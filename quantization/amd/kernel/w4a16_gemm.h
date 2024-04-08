#include <rocwmma/rocwmma.hpp>

#include "utils.h"

__global__ __forceinline__ void w4a16_bs1_kernel(uint32_t* qweight,
                                                 half2* zeros_scales,
                                                 half* vec,
                                                 half* res,
                                                 unsigned int K,
                                                 unsigned int num_per_thread,
                                                 const uint group_size) {
    // each thread load num_per_thread elements from global
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
    half gsum[2];

    // #pragma unroll
    for (int iter = 0; iter < DIV_UP(num_per_thread, 8); iter++) {
        unsigned int j = (start_idx + iter * blockDim.x) << 3;
        if (j >= K) {
            break;
        }

        *(float4*)(&vec_val[0]) = *(float4*)(&vec[j]); //取4*32即8个half

        // Qweight
        qweight_reg[0] = qweight[(row * K + j) / 8];

        // zeros & scales
        zeros_scales_reg[0] = zeros_scales[row * K / group_size + j / group_size];

        // dequant
        *(uint4*)(&mat_val[0]) = dequantize_s4_to_fp16x2_v2(qweight_reg[0]); //解量化为8个half

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

    gsum[0] = sum[0];
    gsum[1] = sum[1];

    float gsum_float_temp[2];
    gsum_float_temp[0] = __half2float(gsum[0]);
    gsum_float_temp[1] = __half2float(gsum[1]);

    static __shared__ float warpLevelSums[WARP_SIZE];

    gsum_float_temp[0] = blockReduceSum_blockdimx_128(gsum_float_temp[0], warpLevelSums);
    gsum_float_temp[1] = blockReduceSum_blockdimx_128(gsum_float_temp[1], warpLevelSums);

    gsum[0] = __float2half(gsum_float_temp[0]);
    gsum[1] = __float2half(gsum_float_temp[1]);

    if (tid == 0) {
        res[row] = gsum[0];
        res[row + 1] = gsum[1];
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

    float gsum_float_temp[4];
    static __shared__ float warpLevelsum[WARP_SIZE * 2];

    gsum_float_temp[0] = __half2float(sum[0]);
    gsum_float_temp[1] = __half2float(sum[1]);

    gsum_float_temp[0] = blockReduceSum_blockdimx_128(gsum_float_temp[0], &warpLevelsum[0]);
    gsum_float_temp[1] = blockReduceSum_blockdimx_128(gsum_float_temp[1], &warpLevelsum[0]);

    sum[0] = __float2half(gsum_float_temp[0]);
    sum[1] = __float2half(gsum_float_temp[1]);

    gsum_float_temp[2] = __half2float(sum[2]);
    gsum_float_temp[3] = __half2float(sum[3]);

    gsum_float_temp[2] = blockReduceSum_blockdimx_128(gsum_float_temp[2], &warpLevelsum[WARP_SIZE]);
    gsum_float_temp[3] = blockReduceSum_blockdimx_128(gsum_float_temp[3], &warpLevelsum[WARP_SIZE]);

    sum[2] = __float2half(gsum_float_temp[2]);
    sum[3] = __float2half(gsum_float_temp[3]);

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
__global__ __forceinline__ void w4a16_gemm_wmma_kernel(uint32_t* __restrict__ qweight,
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

    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, half, rocwmma::row_major> frag_a;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, half, rocwmma::col_major> frag_b;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, half> frag_c;

    rocwmma::fill_fragment(frag_c, __float2half(0.0f));

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

        rocwmma::load_matrix_sync(frag_a, &s_a[wid * 16], LDK);
        rocwmma::load_matrix_sync(frag_b, &s_b[wid * 16], LDK);
        rocwmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

        rocwmma::load_matrix_sync(frag_a, &s_a[wid * 16 + 64], LDK);
        rocwmma::load_matrix_sync(frag_b, &s_b[wid * 16 + 64], LDK);
        rocwmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

        rocwmma::load_matrix_sync(frag_a, &s_a[wid * 16 + 128], LDK);
        rocwmma::load_matrix_sync(frag_b, &s_b[wid * 16 + 128], LDK);
        rocwmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

        rocwmma::load_matrix_sync(frag_a, &s_a[wid * 16 + 192], LDK);
        rocwmma::load_matrix_sync(frag_b, &s_b[wid * 16 + 192], LDK);
        rocwmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += (BK / 8);
        load_b_gmem_k += BK;
    }

    rocwmma::store_matrix_sync(&smem[wid * BM * LDN], frag_c, LDN, rocwmma::mem_row_major);

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
