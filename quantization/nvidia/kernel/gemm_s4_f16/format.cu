// Copyright (c) OpenMMLab. All rights reserved.

#include "common.h"
#include <iostream>

#include "format.h"

namespace turbomind {

__device__ void atomic_assign_u4(uint32_t* address, uint32_t index, uint32_t value) {
    uint32_t old = *address;
    uint32_t assumed;
    do {
        assumed = old;
        uint32_t tmp = (assumed & ~(0xfu << (index * 4u))) | (value << (index * 4u));
        old = atomicCAS(address, assumed, tmp);
    } while (assumed != old);
}

__device__ uint32_t read_u4(const uint32_t* address, uint32_t index) {
    return (*address >> (index * 4u)) & 0xfu;
}

template <int... Ds>
__global__ void permute_u4(uint* dst, const uint* src, Array<int, sizeof...(Ds)> dims) {
    constexpr int N = sizeof...(Ds);

    size_t count = 1;
    PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
        count *= dims[i];
    }

    constexpr int order[] = {Ds...};

    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < count; i += blockDim.x * gridDim.x) {
        int indices[N]{};

        PRAGMA_UNROLL
        for (int j = N - 1, ii = i; j >= 0; --j) {
            indices[j] = ii % dims[j];
            ii /= dims[j];
        }

        auto data = read_u4(src + i / 8, i % 8);

        int index = 0;

        PRAGMA_UNROLL
        for (int j = N - 1, stride = 1; j >= 0; --j) {
            index += indices[order[j]] * stride;
            stride *= dims[order[j]];
        }

        atomic_assign_u4(dst + index / 8, index % 8, data);
    }
}

void reformat_s4_k8_m(uint32_t* dst, const uint32_t* src, int m, int k, cudaStream_t st) {
    // permutation for [k/8, m] layout
    Array<int, 10> shape{k / 32, 2, 2, m / 32, 2, 2, 8, 2, 2, 2};
    //        |warp|  lane  | 2x2 |  a0-7  |
    permute_u4<0, 3, 6, 8, 9, 1, 4, 7, 2, 5><<<512, 512, 0, st>>>(dst, src, shape);
}

void reformat_s4_k_m8(uint32_t* dst, const uint32_t* src, int m, int k, cudaStream_t st) {
    // permutation for [k, m/8] layout
    Array<int, 10> shape{k / 32, 2, 2, 4, 2, m / 32, 2, 2, 2, 4};
    //        |warp|  lane  | 2x2 |  a0-7  |
    permute_u4<0, 5, 9, 8, 3, 1, 6, 4, 2, 7><<<512, 512, 0, st>>>(dst, src, shape);
}

__global__ void dequantize_s4_offset_64(uint4* dst, const uint32_t* src, size_t count) {
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < count; i += blockDim.x * gridDim.x) {
        dst[i] = dequantize_s4_to_fp16x2_v2(src[i]);
    }
}

__global__ void merge_Q(half2* Q, const half* scales, const half* zeros, int count) {
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < count; i += blockDim.x * gridDim.x) {
        if (TURBOMIND_S4_DEQUANT_USE_FMA) {
            // dequant via HFMA2 has numerical statbility issue
            Q[i] = __halves2half2(-zeros[i] * scales[i], scales[i]);
        } else {
            Q[i] = __halves2half2(zeros[i], scales[i]);
        }
    }
}

void convert_s4_k_m8(uint32_t* A_dst,
                     half2* Q_dst,
                     half* workspace,
                     const uint32_t* A_src,
                     const half* scales,
                     const half* zeros,  // const uint32_t* qzeros,
                     int m,
                     int k,
                     int group_size,
                     cudaStream_t st) {
    // zeros is quantized
    // dequantize_s4_offset_64<<<256, 256, 0, st>>>((uint4*)workspace, qzeros, k / group_size * m / 8);
    // merge_Q<<<256, 256, 0, st>>>(Q_dst, scales, workspace, k / group_size * m);

    reformat_s4_k_m8(A_dst, A_src, m, k, st);
}

void transpose_qk_s4_k_m8_hf(uint32_t* dst, const uint32_t* src, int m, int k, int size_per_head, cudaStream_t st) {
    Array<int, 7> shape{k, m / size_per_head, 2, size_per_head / 2 / 8, 2, 2, 2};
    //      dequant   transpose    quant
    // 0123456 -> 0123564 -> 0135642 -> 0135264
    permute_u4<0, 1, 3, 5, 2, 6, 4><<<512, 512, 0, st>>>(dst, src, shape);
}

// [2, k, m/8] -> [k, m/8, 2]
void fuse_w1_w3_s4_k_m8(uint32_t* dst, const uint32_t* src, int m, int k, cudaStream_t st) {
    Array<int, 6> shape{2, k, m / 8, 2, 2, 2};
    //     dequant   transpose   quant
    // 012345 -> 012453 -> 124530 -> 124053
    permute_u4<1, 2, 4, 0, 5, 3><<<512, 512, 0, st>>>(dst, src, shape);
}

__global__ void dequantize_s4_kernel(uint4* dst, const uint* src, size_t count) {
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < count; i += blockDim.x * gridDim.x) {
        dst[i] = dequantize_s4_to_fp16x2(src[i]);
    }
}

void dequantize_s4(uint4* dst, const uint32_t* src, size_t count, cudaStream_t st) {
    dequantize_s4_kernel<<<512, 512>>>(dst, src, count);
}

}  // namespace turbomind

// convert awq to lmdeploy format
void convert_awq_to_lmdeploy(torch::Tensor qweight,
                             torch::Tensor qweight_dst,
                             int in_features,
                             int out_features) {
    turbomind::convert_s4_k_m8(reinterpret_cast<uint32_t*>(qweight_dst.data_ptr<int>()),
                               nullptr,
                               nullptr,
                               reinterpret_cast<uint32_t*>(qweight.data_ptr<uint8_t>()),
                               nullptr,
                               nullptr,
                               out_features,
                               in_features,
                               -1,  // useless
                               0);
}

__global__ void convert_ours_to_awq_kernel(uint8_t* qweight_src,
                                           uint32_t* qweight_dst,
                                           int in_features,
                                           int out_features) {
    int dst_width = out_features / 8;     // use uint32_t (transpose 2x4 to 4x2)
    int count = in_features * dst_width;  // mappint thread to output tensor (qweight_dst)

    int src_width = in_features / 2;

    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < count; i += blockDim.x * gridDim.x) {
        int dst_offset_row = i / dst_width;
        int dst_offset_col = i % dst_width;

        int src_offset_col = dst_offset_row / 2;
        // [src_offset_row, src_offset_row + 7]
        int src_offset_row = dst_offset_col * 8;

        int src_offset = 0;

        int v[8];
        for (int j = 0; j < 8; j++) {
            src_offset = (src_offset_row + j) * src_width + src_offset_col;

            if (dst_offset_row % 2) {
                // low
                v[j] = qweight_src[src_offset] & 0x0F;
            } else {
                // high
                v[j] = qweight_src[src_offset] >> 4;
            }
        }

        uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(&qweight_dst[i]);
        dst_ptr[0] = (v[2] << 4) + v[0];
        dst_ptr[1] = (v[6] << 4) + v[4];
        dst_ptr[2] = (v[3] << 4) + v[1];
        dst_ptr[3] = (v[7] << 4) + v[5];
    }
}

// [uint8_t]  qweight_src:  out_features,   in_features // 2
// [uint32_t]  qweight_dst: in_features,    out_features // 8
void convert_ours_to_awq(torch::Tensor qweight_src,
                         torch::Tensor qweight_dst,
                         int in_features,
                         int out_features) {
    convert_ours_to_awq_kernel<<<512, 512>>>(qweight_src.data_ptr<uint8_t>(),
                                             reinterpret_cast<uint32_t*>(qweight_dst.data_ptr<uint8_t>()),
                                             in_features,
                                             out_features);
}

// [half]  zeros:           out_features,               in_features // group_size
// [half]  scales:          out_features,               in_features // group_size
// [half2] zeros_scales:    in_features // group_size,  out_features
__global__ void transpose_merge_zeros_scales_kernel(half* zeros,
                                                    half* scales,
                                                    half2* zeros_scales,
                                                    int in_features,
                                                    int out_features,
                                                    int group_size) {
    int width = in_features / group_size;
    int count = out_features * width;

    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < count; i += blockDim.x * gridDim.x) {
        half zero_offset = __hadd(zeros[i], __float2half(64.0f));

        int row = i / width;
        int col = i % width;
        int transposed_index = col * out_features + row;

        zeros_scales[transposed_index] = __halves2half2(zero_offset, scales[i]);
    }
}

c {
    transpose_merge_zeros_scales_kernel<<<256, 256>>>(
        reinterpret_cast<half*>(zeros.data_ptr<at::Half>()),
        reinterpret_cast<half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<half2*>(zeros_scales.data_ptr<at::Half>()),
        in_features,
        out_features,
        group_size);
}
