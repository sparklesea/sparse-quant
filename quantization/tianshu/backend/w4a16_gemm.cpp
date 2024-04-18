#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include "mma.h"

// #include "utils.cuh"

#define WARP_SIZE 64
#define WARP_SIZE_HALF 32
#define FLOAT_BANK_SIZE 32
#define MAX_HEAD_SIZE 128
#define MAX_LEN_GROUP 64
#define MAX_LOOP_SPACE 2

#define MAX_THREADS_PER_BLOCK 1024

#define DIV_UP(x, y) ((x) + (y) - 1) / (y)
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), "Tensor " #x " must be on CUDA")
#define CHECK_DTYPE(x, true_dtype) TORCH_CHECK(x.dtype() == true_dtype, "Tensor " #x " must have dtype (" #true_dtype ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "Tensor " #x " must be contiguous")

#define FULLMASK 0xffffffffffffffff

template<typename T>
__device__ __forceinline__ T blockReduceSum_blockdimx_128(T reducing,
                                          T *shared_mem) {
  // const int32_t WPT = blockDim.x / WARP_SIZE > WARP_SIZE_HALF ? WARP_SIZE :
  // (blockDim.x / WARP_SIZE);
  const int32_t lane_id = threadIdx.x % WARP_SIZE;
  const int32_t warp_id = threadIdx.x / WARP_SIZE;

  reducing += __shfl_xor_sync(FULLMASK, reducing, 32, WARP_SIZE);
  reducing += __shfl_xor_sync(FULLMASK, reducing, 16, WARP_SIZE);
  reducing += __shfl_xor_sync(FULLMASK, reducing, 8, WARP_SIZE);
  reducing += __shfl_xor_sync(FULLMASK, reducing, 4, WARP_SIZE);
  reducing += __shfl_xor_sync(FULLMASK, reducing, 2, WARP_SIZE);
  reducing += __shfl_xor_sync(FULLMASK, reducing, 1, WARP_SIZE);

  // reducing += __shfl_xor_down(FULLMASK, reducing, 32, WARP_SIZE);
  // reducing += __shfl_xor_down(FULLMASK, reducing, 16, WARP_SIZE);
  // reducing += __shfl_xor_down(FULLMASK, reducing, 8, WARP_SIZE);
  // reducing += __shfl_xor_down(FULLMASK, reducing, 4, WARP_SIZE);
  // reducing += __shfl_xor_down(FULLMASK, reducing, 2, WARP_SIZE);
  // reducing += __shfl_xor_down(FULLMASK, reducing, 1, WARP_SIZE);

  if (lane_id == 0)
    shared_mem[warp_id] = reducing;
  __syncthreads();

  if (lane_id < 2)
    reducing = shared_mem[lane_id];

  reducing += __shfl_xor_sync(FULLMASK, reducing, 1, WARP_SIZE);

  reducing = __shfl_sync(FULLMASK, reducing, 0, WARP_SIZE);
  return reducing;
}

// reference: https://github.com/InternLM/lmdeploy/blob/main/src/turbomind/kernels/gemm_s_f16/common.h#L115
__inline__ __device__ uint4 dequantize_s4_to_fp16x2_v2(uint32_t const& source) {
    uint4 result;  // 4 * sizeof(uint) == 16Bytes ---> 8 half elements

    // source, uint32_t ---> 8 uint4b_t elements ---> dequant to 8 half elements
    uint32_t* h = reinterpret_cast<uint32_t*>(&result);
    uint32_t const& i4s = reinterpret_cast<uint32_t const&>(source);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOT_MASK = 0x000f000f;
    static constexpr uint32_t TOP_MASK = 0x00f000f0;
    static constexpr uint32_t MAGIC_NUM_0 = 0x64006400;        // `1024`
    static constexpr uint32_t MAGIC_NUM_1 = 0x54005400;        // `64`
    static constexpr uint32_t MAGIC_NUM_2 = MAGIC_NUM_1 >> 4;  // `64` >> 4, 0x05400540

    // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW
    // dependency if we issue immediately before required.

    // source  : X7 X6 X5 X4 X3 X2 X1 X0
    // top_i4s :  0  0 X7 X6 X5 X4 X3 X2
    const uint32_t top_i4s = i4s >> 8;

    {
        //  64 only, trade 4 hfma2 with 2 shifts
        // source  : X7 X6 X5 X4 X3 X2 X1 X0
        // top_i4s :  0  0 X7 X6 X5 X4 X3 X2

        // 0x 0-0-0-X4-0-0-0-X0, 0x(0-5-4-X4-0-5-4-X0)
        // asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[0]) : "r"(i4s), "n"(BOT_MASK), "n"(MAGIC_NUM_2), "n"(immLut));
        h[0] = (i4s & BOT_MASK) | MAGIC_NUM_2;

        // 0x 0-0-X5-0-0-0-X1-0, 0x(5-4-X5-0-5-4-X1-0)
        // asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[1]) : "r"(i4s), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
        h[1] = (i4s & TOP_MASK) | MAGIC_NUM_1;

        // 0x 0-0-0-X6-0-0-0-X2, 0x(0-5-4-X6-0-5-4-X2)
        // asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[2]) : "r"(top_i4s), "n"(BOT_MASK), "n"(MAGIC_NUM_2), "n"(immLut));
        h[2] = (top_i4s & BOT_MASK) | MAGIC_NUM_2;

        // 0x 0-0-X7-0-0-0-X3-0, 0x(5-4-X7-0-5-4-X3-0)
        // asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[3]) : "r"(top_i4s), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
        h[3] = (top_i4s & TOP_MASK) | MAGIC_NUM_1;

        h[0] <<= 4;  // 0x 5-4-X4-0-5-4-X0-0, h[1]: 0x 5-4-X5-0-5-4-X1-0
        h[2] <<= 4;  // 0x 5-4-X6-0-5-4-X2-0, h[3]: 0x 5-4-X7-0-5-4-X3-0
        // X4 : [0, 15]
        // [64, 79] <---> [0, 15] <---> uint4b_t

        // we don't need to subtract the magic nums because zeros will go through the same dequant function
        // and carry the same magic constant, the magic num will be canceled out after subtracting zeros
    }

    return result;
}


// /opt/maca/tools/cu-bridge/bin/cucc -I/opt/cu-bridge/CUDA_DIR/include -I/opt/maca/tools/cu-bridge/include -I/opt/maca/include -I/opt/maca/include/mcr -I/opt/maca/include/mcblas -I/opt/maca/include/mcfft -I/opt/maca/include/mcsolver -I/opt/maca/include/mcdnn -I/opt/maca/include/common -I/opt/maca/include/mcsparse -I/opt/maca/include/mcrand -I/opt/maca/include/mckl -I/opt/maca/include/mcsml -I/opt/maca/include/mctx -I/opt/maca/include/thrust/detail -c w4a16_gemm.cu --expt-relaxed-constexpr --compiler-options '-fPIC' -D_GLIBCXX_USE_CXX11_ABI=1 -gencode=arch=compute_80,code=compute_80 -gencode=arch=compute_80,code=sm_80 -DUSE_MACA -DUSE_MACA -DNV_ARCH_A100 -std=c++17

__global__ __forceinline__ void w4a16_bs1_kernel(uint32_t* qweight,
                                                 half2* zeros_scales,
                                                 half* vec,
                                                 half* res,
                                                 unsigned int K,
                                                 unsigned int num_per_thread,
                                                 const uint group_size) {
    unsigned int tid = threadIdx.x;
    unsigned int row = (blockIdx.y * blockDim.y + threadIdx.y) << 1;
    unsigned int start_idx = threadIdx.x;

    half vec_val[8];

    // Qweight
    uint32_t qweight_reg[2];
    half2 zeros_scales_reg[2];
    half* zeros_scales_ptr = reinterpret_cast<half*>(&zeros_scales_reg[0]);

    half mat_val[16];

    half sum[2] = {__float2half(0.0f)};

    #pragma unroll
    for (int iter = 0; iter < DIV_UP(num_per_thread, 8); iter++) {
        unsigned int j = (start_idx + iter * blockDim.x) << 3;
        if (j >= K) {
            break;
        }

        *(float4*)(&vec_val[0]) = *(float4*)(&vec[j]);

        // *(float4*)(&mat_val[0]) = *(float4*)(&mat[row * K + j]);
        // *(float4*)(&mat_val[8]) = *(float4*)(&mat[(row + 1) * K + j]);

        // Qweight
        qweight_reg[0] = qweight[(row * K + j) / 8];

        // zeros & scales
        zeros_scales_reg[0] = zeros_scales[row * K / group_size + j / group_size];

        // dequant
        *(uint4*)(&mat_val[0]) = dequantize_s4_to_fp16x2_v2(qweight_reg[0]);

        // unroll
        mat_val[0] = __hmul(__hsub(mat_val[0],
                                   zeros_scales_ptr[0]),
                            zeros_scales_ptr[1]);
        mat_val[1] = __hmul(__hsub(mat_val[1],
                                   zeros_scales_ptr[0]),
                            zeros_scales_ptr[1]);
        mat_val[2] = __hmul(__hsub(mat_val[2],
                                   zeros_scales_ptr[0]),
                            zeros_scales_ptr[1]);
        mat_val[3] = __hmul(__hsub(mat_val[3],
                                   zeros_scales_ptr[0]),
                            zeros_scales_ptr[1]);
        mat_val[4] = __hmul(__hsub(mat_val[4],
                                   zeros_scales_ptr[0]),
                            zeros_scales_ptr[1]);
        mat_val[5] = __hmul(__hsub(mat_val[5],
                                   zeros_scales_ptr[0]),
                            zeros_scales_ptr[1]);
        mat_val[6] = __hmul(__hsub(mat_val[6],
                                   zeros_scales_ptr[0]),
                            zeros_scales_ptr[1]);
        mat_val[7] = __hmul(__hsub(mat_val[7],
                                   zeros_scales_ptr[0]),
                            zeros_scales_ptr[1]);

        sum[0] = __hadd(sum[0], __hmul(vec_val[0], mat_val[0]));
        sum[0] = __hadd(sum[0], __hmul(vec_val[1], mat_val[1]));
        sum[0] = __hadd(sum[0], __hmul(vec_val[2], mat_val[2]));
        sum[0] = __hadd(sum[0], __hmul(vec_val[3], mat_val[3]));
        sum[0] = __hadd(sum[0], __hmul(vec_val[4], mat_val[4]));
        sum[0] = __hadd(sum[0], __hmul(vec_val[5], mat_val[5]));
        sum[0] = __hadd(sum[0], __hmul(vec_val[6], mat_val[6]));
        sum[0] = __hadd(sum[0], __hmul(vec_val[7], mat_val[7]));

        qweight_reg[1] = qweight[((row + 1) * K + j) / 8];
        *(uint4*)(&mat_val[8]) = dequantize_s4_to_fp16x2_v2(qweight_reg[1]);
        zeros_scales_reg[1] = zeros_scales[(row + 1) * K / group_size + j / group_size];

        mat_val[8] = __hmul(__hsub(mat_val[8],
                                   zeros_scales_ptr[2]),
                            zeros_scales_ptr[3]);
        mat_val[9] = __hmul(__hsub(mat_val[9],
                                   zeros_scales_ptr[2]),
                            zeros_scales_ptr[3]);
        mat_val[10] = __hmul(__hsub(mat_val[10],
                                    zeros_scales_ptr[2]),
                             zeros_scales_ptr[3]);
        mat_val[11] = __hmul(__hsub(mat_val[11],
                                    zeros_scales_ptr[2]),
                             zeros_scales_ptr[3]);
        mat_val[12] = __hmul(__hsub(mat_val[12],
                                    zeros_scales_ptr[2]),
                             zeros_scales_ptr[3]);
        mat_val[13] = __hmul(__hsub(mat_val[13],
                                    zeros_scales_ptr[2]),
                             zeros_scales_ptr[3]);
        mat_val[14] = __hmul(__hsub(mat_val[14],
                                    zeros_scales_ptr[2]),
                             zeros_scales_ptr[3]);
        mat_val[15] = __hmul(__hsub(mat_val[15],
                                    zeros_scales_ptr[2]),
                             zeros_scales_ptr[3]);

        sum[1] = __hadd(sum[1], __hmul(vec_val[0], mat_val[8]));
        sum[1] = __hadd(sum[1], __hmul(vec_val[1], mat_val[9]));
        sum[1] = __hadd(sum[1], __hmul(vec_val[2], mat_val[10]));
        sum[1] = __hadd(sum[1], __hmul(vec_val[3], mat_val[11]));
        sum[1] = __hadd(sum[1], __hmul(vec_val[4], mat_val[12]));
        sum[1] = __hadd(sum[1], __hmul(vec_val[5], mat_val[13]));
        sum[1] = __hadd(sum[1], __hmul(vec_val[6], mat_val[14]));
        sum[1] = __hadd(sum[1], __hmul(vec_val[7], mat_val[15]));
    }
//     float gsum_float_temp[2];
//     static __shared__ float warpLevelsum[2];

//     gsum_float_temp[0] = __half2float(sum[0]);
//     gsum_float_temp[1] = __half2float(sum[1]);

//     gsum_float_temp[0] = blockReduceSum_blockdimx_128(gsum_float_temp[0], &warpLevelsum[0]);
//     gsum_float_temp[1] = blockReduceSum_blockdimx_128(gsum_float_temp[1], &warpLevelsum[0]);

//     sum[0] = __float2half(gsum_float_temp[0]);
//     sum[1] = __float2half(gsum_float_temp[1]);

    static __shared__ half warpLevelsum[2];

    sum[0] = blockReduceSum_blockdimx_128(sum[0], &warpLevelsum[0]);
    sum[1] = blockReduceSum_blockdimx_128(sum[1], &warpLevelsum[0]);

    if (tid == 0) {
        res[row] = sum[0];
        res[row + 1] = sum[1];
    }
}

__global__ __forceinline__ void w4a16_bs2_kernel(uint32_t* qweight,
                                                 half2* zeros_scales,
                                                 half* vec,
                                                 half* res,
                                                 unsigned int K,
                                                 unsigned int N,
                                                 unsigned int num_per_thread,
                                                 const uint group_size) {
    unsigned int tid = threadIdx.x;
    unsigned int row = (blockIdx.y * blockDim.y + threadIdx.y) << 1;
    unsigned int start_idx = threadIdx.x;

    half vec_val[16];

    // Qweight
    uint32_t qweight_reg[2];
    half2 zeros_scales_reg[2];
    half* zeros_scales_ptr = reinterpret_cast<half*>(&zeros_scales_reg[0]);

    half mat_val[16];

    half sum[4] = {__float2half(0.0f)};

    // #pragma unroll
    for (int iter = 0; iter < DIV_UP(num_per_thread, 8); iter++) {
        unsigned int j = (start_idx + iter * blockDim.x) << 3;
        if (j >= K) {
            break;
        }

        *(float4*)(&vec_val[0]) = *(float4*)(&vec[j]);

        // *(float4*)(&mat_val[0]) = *(float4*)(&mat[row * K + j]);
        // *(float4*)(&mat_val[8]) = *(float4*)(&mat[(row + 1) * K + j]);

        // Qweight
        qweight_reg[0] = qweight[(row * K + j) / 8];

        // zeros & scales
        zeros_scales_reg[0] = zeros_scales[row * K / group_size + j / group_size];

        // dequant
        *(uint4*)(&mat_val[0]) = dequantize_s4_to_fp16x2_v2(qweight_reg[0]);

        // unroll
        mat_val[0] = __hmul(__hsub(mat_val[0],
                                   zeros_scales_ptr[0]),
                            zeros_scales_ptr[1]);
        mat_val[1] = __hmul(__hsub(mat_val[1],
                                   zeros_scales_ptr[0]),
                            zeros_scales_ptr[1]);
        mat_val[2] = __hmul(__hsub(mat_val[2],
                                   zeros_scales_ptr[0]),
                            zeros_scales_ptr[1]);
        mat_val[3] = __hmul(__hsub(mat_val[3],
                                   zeros_scales_ptr[0]),
                            zeros_scales_ptr[1]);
        mat_val[4] = __hmul(__hsub(mat_val[4],
                                   zeros_scales_ptr[0]),
                            zeros_scales_ptr[1]);
        mat_val[5] = __hmul(__hsub(mat_val[5],
                                   zeros_scales_ptr[0]),
                            zeros_scales_ptr[1]);
        mat_val[6] = __hmul(__hsub(mat_val[6],
                                   zeros_scales_ptr[0]),
                            zeros_scales_ptr[1]);
        mat_val[7] = __hmul(__hsub(mat_val[7],
                                   zeros_scales_ptr[0]),
                            zeros_scales_ptr[1]);

        sum[0] = __hadd(sum[0], __hmul(vec_val[0], mat_val[0]));
        sum[0] = __hadd(sum[0], __hmul(vec_val[1], mat_val[1]));
        sum[0] = __hadd(sum[0], __hmul(vec_val[2], mat_val[2]));
        sum[0] = __hadd(sum[0], __hmul(vec_val[3], mat_val[3]));
        sum[0] = __hadd(sum[0], __hmul(vec_val[4], mat_val[4]));
        sum[0] = __hadd(sum[0], __hmul(vec_val[5], mat_val[5]));
        sum[0] = __hadd(sum[0], __hmul(vec_val[6], mat_val[6]));
        sum[0] = __hadd(sum[0], __hmul(vec_val[7], mat_val[7]));

        *(float4*)(&vec_val[8]) = *(float4*)(&vec[j + 1 * K]);

        sum[2] = __hadd(sum[2], __hmul(vec_val[8], mat_val[0]));
        sum[2] = __hadd(sum[2], __hmul(vec_val[9], mat_val[1]));
        sum[2] = __hadd(sum[2], __hmul(vec_val[10], mat_val[2]));
        sum[2] = __hadd(sum[2], __hmul(vec_val[11], mat_val[3]));
        sum[2] = __hadd(sum[2], __hmul(vec_val[12], mat_val[4]));
        sum[2] = __hadd(sum[2], __hmul(vec_val[13], mat_val[5]));
        sum[2] = __hadd(sum[2], __hmul(vec_val[14], mat_val[6]));
        sum[2] = __hadd(sum[2], __hmul(vec_val[15], mat_val[7]));

        qweight_reg[1] = qweight[((row + 1) * K + j) / 8];
        *(uint4*)(&mat_val[8]) = dequantize_s4_to_fp16x2_v2(qweight_reg[1]);
        zeros_scales_reg[1] = zeros_scales[(row + 1) * K / group_size + j / group_size];

        mat_val[8] = __hmul(__hsub(mat_val[8],
                                   zeros_scales_ptr[2]),
                            zeros_scales_ptr[3]);
        mat_val[9] = __hmul(__hsub(mat_val[9],
                                   zeros_scales_ptr[2]),
                            zeros_scales_ptr[3]);
        mat_val[10] = __hmul(__hsub(mat_val[10],
                                    zeros_scales_ptr[2]),
                             zeros_scales_ptr[3]);
        mat_val[11] = __hmul(__hsub(mat_val[11],
                                    zeros_scales_ptr[2]),
                             zeros_scales_ptr[3]);
        mat_val[12] = __hmul(__hsub(mat_val[12],
                                    zeros_scales_ptr[2]),
                             zeros_scales_ptr[3]);
        mat_val[13] = __hmul(__hsub(mat_val[13],
                                    zeros_scales_ptr[2]),
                             zeros_scales_ptr[3]);
        mat_val[14] = __hmul(__hsub(mat_val[14],
                                    zeros_scales_ptr[2]),
                             zeros_scales_ptr[3]);
        mat_val[15] = __hmul(__hsub(mat_val[15],
                                    zeros_scales_ptr[2]),
                             zeros_scales_ptr[3]);

        sum[1] = __hadd(sum[1], __hmul(vec_val[0], mat_val[8]));
        sum[1] = __hadd(sum[1], __hmul(vec_val[1], mat_val[9]));
        sum[1] = __hadd(sum[1], __hmul(vec_val[2], mat_val[10]));
        sum[1] = __hadd(sum[1], __hmul(vec_val[3], mat_val[11]));
        sum[1] = __hadd(sum[1], __hmul(vec_val[4], mat_val[12]));
        sum[1] = __hadd(sum[1], __hmul(vec_val[5], mat_val[13]));
        sum[1] = __hadd(sum[1], __hmul(vec_val[6], mat_val[14]));
        sum[1] = __hadd(sum[1], __hmul(vec_val[7], mat_val[15]));

        sum[3] = __hadd(sum[3], __hmul(vec_val[8], mat_val[8]));
        sum[3] = __hadd(sum[3], __hmul(vec_val[9], mat_val[9]));
        sum[3] = __hadd(sum[3], __hmul(vec_val[10], mat_val[10]));
        sum[3] = __hadd(sum[3], __hmul(vec_val[11], mat_val[11]));
        sum[3] = __hadd(sum[3], __hmul(vec_val[12], mat_val[12]));
        sum[3] = __hadd(sum[3], __hmul(vec_val[13], mat_val[13]));
        sum[3] = __hadd(sum[3], __hmul(vec_val[14], mat_val[14]));
        sum[3] = __hadd(sum[3], __hmul(vec_val[15], mat_val[15]));
    }

//     float gsum_float_temp[4];
//     static __shared__ float warpLevelsum[2];

//     gsum_float_temp[0] = __half2float(sum[0]);
//     gsum_float_temp[1] = __half2float(sum[1]);

//     gsum_float_temp[0] = blockReduceSum_blockdimx_128(gsum_float_temp[0], &warpLevelsum[0]);
//     gsum_float_temp[1] = blockReduceSum_blockdimx_128(gsum_float_temp[1], &warpLevelsum[0]);

//     sum[0] = __float2half(gsum_float_temp[0]);
//     sum[1] = __float2half(gsum_float_temp[1]);

//     gsum_float_temp[2] = __half2float(sum[2]);
//     gsum_float_temp[3] = __half2float(sum[3]);

//     gsum_float_temp[2] = blockReduceSum_blockdimx_128(gsum_float_temp[2], &warpLevelsum[0]);
//     gsum_float_temp[3] = blockReduceSum_blockdimx_128(gsum_float_temp[3], &warpLevelsum[0]);

//     sum[2] = __float2half(gsum_float_temp[2]);
//     sum[3] = __float2half(gsum_float_temp[3]);

    static __shared__ half warpLevelsum[2];

    sum[0] = blockReduceSum_blockdimx_128(sum[0], &warpLevelsum[0]);
    sum[1] = blockReduceSum_blockdimx_128(sum[1], &warpLevelsum[0]);

    sum[2] = blockReduceSum_blockdimx_128(sum[2], &warpLevelsum[0]);
    sum[3] = blockReduceSum_blockdimx_128(sum[3], &warpLevelsum[0]);

    if (tid == 0) {
        res[row] = sum[0];
        res[row + 1] = sum[1];
        res[row + N] = sum[2];
        res[row + N + 1] = sum[3];
    }
}

// ING: select more effective params
// 16, 16, 256, 264, 24
template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void w4a16_gemm_wmma_kernel_origin(uint32_t* __restrict__ qweight,
                                                       half2* zeros_scales,
                                                       const half* __restrict__ input,
                                                       half* __restrict__ output,
                                                       const int M, const int N, const int K,
                                                       const int group_size) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tid = threadIdx.x;
    int wid = tid / 64;

    __shared__ half smem[(BM + BN) * LDK];
    half* s_a = smem;
    half* s_b = smem + BM * LDK;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> frag_c;

    nvcuda::wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid / 32) * 2;
    int load_a_smem_k = (tid % 32) * 8;
    // 8 x 32, 2 times
    int load_b_smem_n = (tid / 32) * 2;
    int load_b_smem_k = (tid % 32) * 8;

    half* load_a_smem_addrs[2];
    load_a_smem_addrs[0] = s_a + OFFSET(load_a_smem_m, load_a_smem_k, LDK);
    load_a_smem_addrs[1] = load_a_smem_addrs[0] + LDK;
    half* load_b_smem_addrs[2];
    load_b_smem_addrs[0] = s_b + OFFSET(load_b_smem_n, load_b_smem_k, LDK);
    load_b_smem_addrs[1] = load_b_smem_addrs[0] + LDK;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = load_a_smem_k;
    int load_b_gmem_k = load_b_smem_k;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    // each uint32_t data has 8 4bit weight
    int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_gmem_k / 8, K / 8);

    uint32_t qweight_reg[2];
    half weight_reg[16];
    half2 zeros_scales_reg[2];
    half* zeros_scales_ptr = reinterpret_cast<half*>(&zeros_scales_reg[0]);

    for (int bk = 0; bk < K / BK; bk++) {
        if (load_a_gmem_m < M) {
            *(float4*)(load_a_smem_addrs[0]) = *(float4*)(&input[load_a_gmem_addr]);
            *(float4*)(load_a_smem_addrs[1]) = *(float4*)(&input[load_a_gmem_addr + K]);
        }

        {
            qweight_reg[0] = qweight[load_b_gmem_addr];
            qweight_reg[1] = qweight[load_b_gmem_addr + K / 8];

            int zeros_scales_index = load_b_gmem_n * K / group_size + load_b_gmem_k / group_size;
            zeros_scales_reg[0] = zeros_scales[zeros_scales_index];
            zeros_scales_reg[1] = zeros_scales[zeros_scales_index + K / group_size];

            // dequant
            *(uint4*)(&weight_reg[0]) = dequantize_s4_to_fp16x2_v2(qweight_reg[0]);
            *(uint4*)(&weight_reg[8]) = dequantize_s4_to_fp16x2_v2(qweight_reg[1]);

            weight_reg[0] = __hmul(__hsub(weight_reg[0], zeros_scales_ptr[0]), zeros_scales_ptr[1]);
            weight_reg[1] = __hmul(__hsub(weight_reg[1], zeros_scales_ptr[0]), zeros_scales_ptr[1]);
            weight_reg[2] = __hmul(__hsub(weight_reg[2], zeros_scales_ptr[0]), zeros_scales_ptr[1]);
            weight_reg[3] = __hmul(__hsub(weight_reg[3], zeros_scales_ptr[0]), zeros_scales_ptr[1]);
            weight_reg[4] = __hmul(__hsub(weight_reg[4], zeros_scales_ptr[0]), zeros_scales_ptr[1]);
            weight_reg[5] = __hmul(__hsub(weight_reg[5], zeros_scales_ptr[0]), zeros_scales_ptr[1]);
            weight_reg[6] = __hmul(__hsub(weight_reg[6], zeros_scales_ptr[0]), zeros_scales_ptr[1]);
            weight_reg[7] = __hmul(__hsub(weight_reg[7], zeros_scales_ptr[0]), zeros_scales_ptr[1]);

            weight_reg[8] = __hmul(__hsub(weight_reg[8], zeros_scales_ptr[2]), zeros_scales_ptr[3]);
            weight_reg[9] = __hmul(__hsub(weight_reg[9], zeros_scales_ptr[2]), zeros_scales_ptr[3]);
            weight_reg[10] = __hmul(__hsub(weight_reg[10], zeros_scales_ptr[2]), zeros_scales_ptr[3]);
            weight_reg[11] = __hmul(__hsub(weight_reg[11], zeros_scales_ptr[2]), zeros_scales_ptr[3]);
            weight_reg[12] = __hmul(__hsub(weight_reg[12], zeros_scales_ptr[2]), zeros_scales_ptr[3]);
            weight_reg[13] = __hmul(__hsub(weight_reg[13], zeros_scales_ptr[2]), zeros_scales_ptr[3]);
            weight_reg[14] = __hmul(__hsub(weight_reg[14], zeros_scales_ptr[2]), zeros_scales_ptr[3]);
            weight_reg[15] = __hmul(__hsub(weight_reg[15], zeros_scales_ptr[2]), zeros_scales_ptr[3]);

            // STS
            *(uint4*)(load_b_smem_addrs[0]) = *(uint4*)(&weight_reg[0]);
            *(uint4*)(load_b_smem_addrs[1]) = *(uint4*)(&weight_reg[8]);
        }

        __syncthreads();

        nvcuda::wmma::load_matrix_sync(frag_a, &s_a[wid * 16], LDK);
        nvcuda::wmma::load_matrix_sync(frag_b, &s_b[wid * 16], LDK);
        nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

        nvcuda::wmma::load_matrix_sync(frag_a, &s_a[wid * 16 + 64], LDK);
        nvcuda::wmma::load_matrix_sync(frag_b, &s_b[wid * 16 + 64], LDK);
        nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

        nvcuda::wmma::load_matrix_sync(frag_a, &s_a[wid * 16 + 128], LDK);
        nvcuda::wmma::load_matrix_sync(frag_b, &s_b[wid * 16 + 128], LDK);
        nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

        nvcuda::wmma::load_matrix_sync(frag_a, &s_a[wid * 16 + 192], LDK);
        nvcuda::wmma::load_matrix_sync(frag_b, &s_b[wid * 16 + 192], LDK);
        nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += (BK / 8);
        load_b_gmem_k += BK;
    }

    nvcuda::wmma::store_matrix_sync(&smem[wid * BM * LDN], frag_c, LDN, nvcuda::wmma::layout_t::mem_row_major);

    __syncthreads();

    int smem_c_m = tid / 16;
    int smem_c_n = tid % 16;
    int smem_c_addr = OFFSET(smem_c_m, smem_c_n, LDN);

    int gmem_c_m = by * BM + smem_c_m;
    int gmem_c_n = bx * BN + smem_c_n;
    int gmem_c_addr = OFFSET(gmem_c_m, gmem_c_n, N);

    smem[smem_c_addr] = __hadd(smem[smem_c_addr], smem[smem_c_addr + BM * LDN]);
    smem[smem_c_addr] = __hadd(smem[smem_c_addr], smem[smem_c_addr + BM * LDN * 2]);
    smem[smem_c_addr] = __hadd(smem[smem_c_addr], smem[smem_c_addr + BM * LDN * 3]);

    if (gmem_c_m < M) {
        output[gmem_c_addr] = smem[smem_c_addr];
    }
}

template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void w4a16_gemm_nvcuda::wmma_kernel(uint32_t* __restrict__ qweight,
                                                       half2* zeros_scales,
                                                       const half* __restrict__ input,
                                                       half* __restrict__ output,
                                                       const int M, const int N, const int K,
                                                       const int group_size) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tid = threadIdx.x;
    int wid = tid / 64;

    __shared__ half smem[(BM + BN) * LDK];
    half* s_a = smem;
    half* s_b = smem + BM * LDK;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> frag_c;

    nvcuda::wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid / 32) * 2;
    int load_a_smem_k = (tid % 32) * 8;
    // 8 x 32, 2 times
    int load_b_smem_n = (tid / 32) * 2;
    int load_b_smem_k = (tid % 32) * 8;

    half* load_a_smem_addrs[2];
    load_a_smem_addrs[0] = s_a + OFFSET(load_a_smem_m, load_a_smem_k, LDK);
    load_a_smem_addrs[1] = load_a_smem_addrs[0] + LDK;
    half* load_b_smem_addrs[2];
    load_b_smem_addrs[0] = s_b + OFFSET(load_b_smem_n, load_b_smem_k, LDK);
    load_b_smem_addrs[1] = load_b_smem_addrs[0] + LDK;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = load_a_smem_k;
    int load_b_gmem_k = load_b_smem_k;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    // each uint32_t data has 8 4bit weight
    int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_gmem_k / 8, K / 8);

    uint32_t qweight_reg[2];
    half weight_reg[16];
    half2 zeros_scales_reg[2];
    half* zeros_scales_ptr = reinterpret_cast<half*>(&zeros_scales_reg[0]);

    for (int bk = 0; bk < K / BK; bk++) {
        if (load_a_gmem_m < M) {
            *(float4*)(load_a_smem_addrs[0]) = *(float4*)(&input[load_a_gmem_addr]);
            *(float4*)(load_a_smem_addrs[1]) = *(float4*)(&input[load_a_gmem_addr + K]);
        }

        {
            qweight_reg[0] = qweight[load_b_gmem_addr];
            qweight_reg[1] = qweight[load_b_gmem_addr + K / 8];

            int zeros_scales_index = load_b_gmem_n * K / group_size + load_b_gmem_k / group_size;
            zeros_scales_reg[0] = zeros_scales[zeros_scales_index];
            zeros_scales_reg[1] = zeros_scales[zeros_scales_index + K / group_size];

            // dequant
            *(uint4*)(&weight_reg[0]) = dequantize_s4_to_fp16x2_v2(qweight_reg[0]);
            *(uint4*)(&weight_reg[8]) = dequantize_s4_to_fp16x2_v2(qweight_reg[1]);

            weight_reg[0] = __hmul(__hsub(weight_reg[0], zeros_scales_ptr[0]), zeros_scales_ptr[1]);
            weight_reg[1] = __hmul(__hsub(weight_reg[1], zeros_scales_ptr[0]), zeros_scales_ptr[1]);
            weight_reg[2] = __hmul(__hsub(weight_reg[2], zeros_scales_ptr[0]), zeros_scales_ptr[1]);
            weight_reg[3] = __hmul(__hsub(weight_reg[3], zeros_scales_ptr[0]), zeros_scales_ptr[1]);
            weight_reg[4] = __hmul(__hsub(weight_reg[4], zeros_scales_ptr[0]), zeros_scales_ptr[1]);
            weight_reg[5] = __hmul(__hsub(weight_reg[5], zeros_scales_ptr[0]), zeros_scales_ptr[1]);
            weight_reg[6] = __hmul(__hsub(weight_reg[6], zeros_scales_ptr[0]), zeros_scales_ptr[1]);
            weight_reg[7] = __hmul(__hsub(weight_reg[7], zeros_scales_ptr[0]), zeros_scales_ptr[1]);

            weight_reg[8] = __hmul(__hsub(weight_reg[8], zeros_scales_ptr[2]), zeros_scales_ptr[3]);
            weight_reg[9] = __hmul(__hsub(weight_reg[9], zeros_scales_ptr[2]), zeros_scales_ptr[3]);
            weight_reg[10] = __hmul(__hsub(weight_reg[10], zeros_scales_ptr[2]), zeros_scales_ptr[3]);
            weight_reg[11] = __hmul(__hsub(weight_reg[11], zeros_scales_ptr[2]), zeros_scales_ptr[3]);
            weight_reg[12] = __hmul(__hsub(weight_reg[12], zeros_scales_ptr[2]), zeros_scales_ptr[3]);
            weight_reg[13] = __hmul(__hsub(weight_reg[13], zeros_scales_ptr[2]), zeros_scales_ptr[3]);
            weight_reg[14] = __hmul(__hsub(weight_reg[14], zeros_scales_ptr[2]), zeros_scales_ptr[3]);
            weight_reg[15] = __hmul(__hsub(weight_reg[15], zeros_scales_ptr[2]), zeros_scales_ptr[3]);

            // STS
            *(uint4*)(load_b_smem_addrs[0]) = *(uint4*)(&weight_reg[0]);
            *(uint4*)(load_b_smem_addrs[1]) = *(uint4*)(&weight_reg[8]);
        }

        __syncthreads();

        nvcuda::wmma::load_matrix_sync(frag_a, &s_a[wid * 16], LDK);
        nvcuda::wmma::load_matrix_sync(frag_b, &s_b[wid * 16], LDK);
        nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

        nvcuda::wmma::load_matrix_sync(frag_a, &s_a[wid * 16 + 64], LDK);
        nvcuda::wmma::load_matrix_sync(frag_b, &s_b[wid * 16 + 64], LDK);
        nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

        nvcuda::wmma::load_matrix_sync(frag_a, &s_a[wid * 16 + 128], LDK);
        nvcuda::wmma::load_matrix_sync(frag_b, &s_b[wid * 16 + 128], LDK);
        nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

        nvcuda::wmma::load_matrix_sync(frag_a, &s_a[wid * 16 + 192], LDK);
        nvcuda::wmma::load_matrix_sync(frag_b, &s_b[wid * 16 + 192], LDK);
        nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += (BK / 8);
        load_b_gmem_k += BK;
    }

    nvcuda::wmma::store_matrix_sync(&smem[wid * BM * LDN], frag_c, LDN, nvcuda::wmma::layout_t::mem_row_major);

    __syncthreads();

    int smem_c_m = tid / 16;
    int smem_c_n = tid % 16;
    int smem_c_addr = OFFSET(smem_c_m, smem_c_n, LDN);

    int gmem_c_m = by * BM + smem_c_m;
    int gmem_c_n = bx * BN + smem_c_n;
    int gmem_c_addr = OFFSET(gmem_c_m, gmem_c_n, N);

    smem[smem_c_addr] = __hadd(smem[smem_c_addr], smem[smem_c_addr + BM * LDN]);
    smem[smem_c_addr] = __hadd(smem[smem_c_addr], smem[smem_c_addr + BM * LDN * 2]);
    smem[smem_c_addr] = __hadd(smem[smem_c_addr], smem[smem_c_addr + BM * LDN * 3]);

    if (gmem_c_m < M) {
        output[gmem_c_addr] = smem[smem_c_addr];
    }
}

