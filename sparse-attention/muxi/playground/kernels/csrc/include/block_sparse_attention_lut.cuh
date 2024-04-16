
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

#include "utils.h"

using namespace std;


__global__ void sparse_attention_prefill_fwd_warp_kernel(
    at::Half* Q, at::Half* K, at::Half* V, float sm_scale, at::Half* out, int* lut, float* m_i, float* l_i,
     int bsz, int head, int seq_len, int hidden_dim, int lut_block, int lut_size, int NNZ, float* p
) {
    int start_m = blockIdx.x;
    int off_hz = blockIdx.y;
    int lut_indicator = blockIdx.y % head;
    int qvk_offset = off_hz * (seq_len * hidden_dim);
    int lut_offset = lut_indicator * (lut_block * lut_size);
    int ml_offset = off_hz * seq_len;

    int tx = threadIdx.x / 64;
    int ty = threadIdx.x % 64;

    int offs_m = start_m * 64 + tx * 4;
    int offs_n = ty;

    __shared__ float acc[64][128];
    __shared__ float m_ij[64];
    __shared__ float l_ij[64];
    __shared__ float p_scale[64];
    __shared__ float acc_scale[64];
    for (int i = 0; i < 4; i++) {
        acc[tx * 4 + i][ty] = 0;
        acc[tx * 4 + i][ty + 64] = 0;
    }
    m_ij[ty] = -INFINITY;
    l_ij[ty] = 0;
    p_scale[ty] = 0;
    acc_scale[ty] = 0;
    __syncthreads();
    float qk_scale = sm_scale * 1.44269504;
    int last_nnz_id = -1;

    at::Half q_val[32];
    at::Half k_val[8];
    at::Half v_val[8];
    at::Half sum[4] = {__float2half(0.f)};
    float gsum[4];
    float m[4] = {-INFINITY};
    float l[4] = {0};
    float m_i_new[4] = {-INFINITY};
    float l_i_new[4] = {0};
    float alpha[4];
    float beta[4];

    for (int nnz_id = 0; nnz_id < NNZ; nnz_id++) {
        int present_nnz_id = lut[lut_offset + start_m * lut_size + nnz_id];
        int start_n = present_nnz_id * 64;

        for (int iter = 0; iter < 16; iter++) {
            if (offs_m < seq_len) {
                *(float4 *)(&q_val[0]) = *(float4 *)(&Q[qvk_offset + offs_m * 128 + iter * 8]);
            }
            if (offs_m + 1 < seq_len) {
                *(float4 *)(&q_val[8]) = *(float4 *)(&Q[qvk_offset + (offs_m + 1) * 128 + iter * 8]);
            }
            if (offs_m + 2 < seq_len) {
                *(float4 *)(&q_val[16]) = *(float4 *)(&Q[qvk_offset + (offs_m + 2) * 128 + iter * 8]);
            }
            if (offs_m + 3 < seq_len) {
                *(float4 *)(&q_val[24]) = *(float4 *)(&Q[qvk_offset + (offs_m + 3) * 128 + iter * 8]);
            }

            if (start_n + offs_n < seq_len) {
                *(float4 *)(&k_val[0]) = *(float4 *)(&K[qvk_offset + (start_n + offs_n) * 128 + iter * 8]);
            }

            if (offs_m < seq_len && start_n + offs_n < seq_len && offs_m >= start_n + offs_n) {
                sum[0] = __hadd(sum[0], __hmul(q_val[0], k_val[0]));
                sum[0] = __hadd(sum[0], __hmul(q_val[1], k_val[1]));
                sum[0] = __hadd(sum[0], __hmul(q_val[2], k_val[2]));
                sum[0] = __hadd(sum[0], __hmul(q_val[3], k_val[3]));
                sum[0] = __hadd(sum[0], __hmul(q_val[4], k_val[4]));
                sum[0] = __hadd(sum[0], __hmul(q_val[5], k_val[5]));
                sum[0] = __hadd(sum[0], __hmul(q_val[6], k_val[6]));
                sum[0] = __hadd(sum[0], __hmul(q_val[7], k_val[7]));
            }

            if (offs_m + 1 < seq_len && start_n + offs_n < seq_len && offs_m + 1 >= start_n + offs_n) {
                sum[1] = __hadd(sum[1], __hmul(q_val[8], k_val[0]));
                sum[1] = __hadd(sum[1], __hmul(q_val[9], k_val[1]));
                sum[1] = __hadd(sum[1], __hmul(q_val[10], k_val[2]));
                sum[1] = __hadd(sum[1], __hmul(q_val[11], k_val[3]));
                sum[1] = __hadd(sum[1], __hmul(q_val[12], k_val[4]));
                sum[1] = __hadd(sum[1], __hmul(q_val[13], k_val[5]));
                sum[1] = __hadd(sum[1], __hmul(q_val[14], k_val[6]));
                sum[1] = __hadd(sum[1], __hmul(q_val[15], k_val[7]));
            }

            if (offs_m + 2 < seq_len && start_n + offs_n < seq_len && offs_m + 2 >= start_n + offs_n) {
                sum[2] = __hadd(sum[2], __hmul(q_val[16], k_val[0]));
                sum[2] = __hadd(sum[2], __hmul(q_val[17], k_val[1]));
                sum[2] = __hadd(sum[2], __hmul(q_val[18], k_val[2]));
                sum[2] = __hadd(sum[2], __hmul(q_val[19], k_val[3]));
                sum[2] = __hadd(sum[2], __hmul(q_val[20], k_val[4]));
                sum[2] = __hadd(sum[2], __hmul(q_val[21], k_val[5]));
                sum[2] = __hadd(sum[2], __hmul(q_val[22], k_val[6]));
                sum[2] = __hadd(sum[2], __hmul(q_val[23], k_val[7]));
            }

            if (offs_m + 3 < seq_len && start_n + offs_n < seq_len && offs_m + 3 >= start_n + offs_n) {
                sum[3] = __hadd(sum[3], __hmul(q_val[24], k_val[0]));
                sum[3] = __hadd(sum[3], __hmul(q_val[25], k_val[1]));
                sum[3] = __hadd(sum[3], __hmul(q_val[26], k_val[2]));
                sum[3] = __hadd(sum[3], __hmul(q_val[27], k_val[3]));
                sum[3] = __hadd(sum[3], __hmul(q_val[28], k_val[4]));
                sum[3] = __hadd(sum[3], __hmul(q_val[29], k_val[5]));
                sum[3] = __hadd(sum[3], __hmul(q_val[30], k_val[6]));
                sum[3] = __hadd(sum[3], __hmul(q_val[31], k_val[7]));
            }

        }

        if (offs_m < seq_len && start_n + offs_n < seq_len && offs_m >= start_n + offs_n) {
            gsum[0] = __half2float(sum[0]) * qk_scale;
        } else {
            gsum[0] = -INFINITY;
        }

        if (offs_m + 1 < seq_len && start_n + offs_n < seq_len && offs_m + 1 >= start_n + offs_n) {
            gsum[1] = __half2float(sum[1]) * qk_scale;
        } else {
            gsum[1] = -INFINITY;
        }

        if (offs_m + 2 < seq_len && start_n + offs_n < seq_len && offs_m + 2 >= start_n + offs_n) {
            gsum[2] = __half2float(sum[2]) * qk_scale;
        } else {
            gsum[2] = -INFINITY;
        }

        if (offs_m + 3 < seq_len && start_n + offs_n < seq_len && offs_m + 3 >= start_n + offs_n) {
            gsum[3] = __half2float(sum[3]) * qk_scale;
        } else {
            gsum[3] = -INFINITY;
        }

        p[(tx * 4 + 0) * 64 + ty] = gsum[0];
        p[(tx * 4 + 1) * 64 + ty] = gsum[1];
        p[(tx * 4 + 2) * 64 + ty] = gsum[2];
        p[(tx * 4 + 3) * 64 + ty] = gsum[3];

        // gsum[0] = (tx * 4 + 0) * 64 + ty;
        // gsum[1] = (tx * 4 + 1) * 64 + ty;
        // gsum[2] = (tx * 4 + 2) * 64 + ty;
        // gsum[3] = (tx * 4 + 3) * 64 + ty;
        // if (offs_m == 4) {
        //     printf("offs_m 5, ty %d, %f \n", ty, gsum[1]);
        // }

        float temp_0 = gsum[0];
        float temp_1 = gsum[1];
        float temp_2 = gsum[2];
        float temp_3 = gsum[3];
        m[0] = temp_0;
        m[1] = temp_1;
        m[2] = temp_2;
        m[3] = temp_3;

        SHFL_DOWN_MAX(m[0], temp_0)
        SHFL_DOWN_MAX(m[1], temp_1)
        SHFL_DOWN_MAX(m[2], temp_2)
        SHFL_DOWN_MAX(m[3], temp_3)

        if (ty == 0) {
            m_ij[tx * 4 + 0] = m[0];
            m_ij[tx * 4 + 1] = m[1];
            m_ij[tx * 4 + 2] = m[2];
            m_ij[tx * 4 + 3] = m[3];
        }
        __syncthreads();
        m[0] = m_ij[tx * 4 + 0];
        m[1] = m_ij[tx * 4 + 1];
        m[2] = m_ij[tx * 4 + 2];
        m[3] = m_ij[tx * 4 + 3];

        // if (offs_m < seq_len) {
        //     printf("offs_m: %d, m_ij is %f \n", offs_m, m[0]);
        // }
        // if (offs_m + 1 < seq_len) {
        //     printf("offs_m: %d, m_ij is %f \n", offs_m + 1, m[1]);
        // }
        // if (offs_m + 2 < seq_len) {
        //     printf("offs_m: %d, m_ij is %f \n", offs_m + 2, m[2]);
        // }
        // if (offs_m + 3 < seq_len) {
        //     printf("offs_m: %d, m_ij is %f \n", offs_m + 3, m[3]);
        // }

        if (m[0] == -INFINITY) {
            gsum[0] = 0;
        } else {
            gsum[0] = exp2f(gsum[0] - m[0]);
        }
        if (m[1] == -INFINITY) {
            gsum[1] = 0;
        } else {
            gsum[1] = exp2f(gsum[1] - m[1]);
        }
        if (m[2] == -INFINITY) {
            gsum[2] = 0;
        } else {
            gsum[2] = exp2f(gsum[2] - m[2]);
        }
        if (m[3] == -INFINITY) {
            gsum[3] = 0;
        } else {
            gsum[3] = exp2f(gsum[3] - m[3]);
        }

        if (last_nnz_id == present_nnz_id) {
            gsum[0] = 0;
            gsum[1] = 0;
            gsum[2] = 0;
            gsum[3] = 0;
        }
        l[0] = gsum[0];
        l[1] = gsum[1];
        l[2] = gsum[2];
        l[3] = gsum[3];

        SHFL_DOWN_SUM(l[0])
        SHFL_DOWN_SUM(l[1])
        SHFL_DOWN_SUM(l[2])
        SHFL_DOWN_SUM(l[3])

        if (ty == 0) {
            l_ij[tx * 4 + 0] = l[0];
            l_ij[tx * 4 + 1] = l[1];
            l_ij[tx * 4 + 2] = l[2];
            l_ij[tx * 4 + 3] = l[3];
        }
        __syncthreads();

        l[0] = l_ij[tx * 4 + 0];
        l[1] = l_ij[tx * 4 + 1];
        l[2] = l_ij[tx * 4 + 2];
        l[3] = l_ij[tx * 4 + 3];

        m_i_new[0] = m[0];
        m_i_new[1] = m[1];
        m_i_new[2] = m[2];
        m_i_new[3] = m[3];

        for (int i = 0; i < 4; i++) {
            if (ty == 0 && offs_m + i < seq_len) {
                m_i_new[i] = max(m_i_new[i], m_i[ml_offset + offs_m + i]);
                alpha[i] = exp2f(m_i[ml_offset + offs_m + i] - m_i_new[i]);
                beta[i] = exp2f(m[i] - m_i_new[i]);
                l_i[ml_offset + offs_m + i] *= alpha[i];
                l_i_new[i] = l_i[ml_offset + offs_m + i] + beta[i] * l[i];
                p_scale[tx * 4 + i] = beta[i] / l_i_new[i];
                acc_scale[tx * 4 + i] = l_i[ml_offset + offs_m + i] / l_i_new[i];

                l_i[ml_offset + offs_m + i] = l_i_new[i];
                m_i[ml_offset + offs_m + i] = l_i_new[i];
            }
        }
        __syncthreads();
        gsum[0] *= p_scale[tx * 4 + 0];
        gsum[1] *= p_scale[tx * 4 + 1];
        gsum[2] *= p_scale[tx * 4 + 2];
        gsum[3] *= p_scale[tx * 4 + 3];

        acc[tx * 4 + 0][ty] *= acc_scale[tx * 4 + 0];
        acc[tx * 4 + 1][ty] *= acc_scale[tx * 4 + 1];
        acc[tx * 4 + 2][ty] *= acc_scale[tx * 4 + 2];
        acc[tx * 4 + 3][ty] *= acc_scale[tx * 4 + 3];
        acc[tx * 4 + 0][ty + 64] *= acc_scale[tx * 4 + 0];
        acc[tx * 4 + 1][ty + 64] *= acc_scale[tx * 4 + 1];
        acc[tx * 4 + 2][ty + 64] *= acc_scale[tx * 4 + 2];
        acc[tx * 4 + 3][ty + 64] *= acc_scale[tx * 4 + 3];

        // if (ty == 0) {
        //     printf("entering p * v \n");
        // }
        // __syncthreads();

        /* p * v */
        for (int iter = 0; iter < 16; iter++) {
            if (start_n + offs_n < seq_len) {
                *(float4 *)(&v_val[0]) = *(float4 *)(&V[qvk_offset + (start_n + offs_n) * 128 + iter * 8]);
                float acc_temp;
                for (int i = 0; i < 8; i++) {
                    acc_temp = gsum[0] * __half2float(v_val[i]);
                    atomicAdd(&acc[tx * 4 + 0][iter * 8 + i], acc_temp);
                    acc_temp = gsum[1] * __half2float(v_val[i]);
                    atomicAdd(&acc[tx * 4 + 1][iter * 8 + i], acc_temp);
                    acc_temp = gsum[2] * __half2float(v_val[i]);
                    atomicAdd(&acc[tx * 4 + 2][iter * 8 + i], acc_temp);
                    acc_temp = gsum[3] * __half2float(v_val[i]);
                    atomicAdd(&acc[tx * 4 + 3][iter * 8 + i], acc_temp);
                }
            }
        }
        last_nnz_id = present_nnz_id;
        __syncthreads();

    }

    if (offs_m + 0 < seq_len) {
        out[qvk_offset + (offs_m + 0) * hidden_dim + ty] = __float2half(acc[tx * 4 + 0][ty]);
        out[qvk_offset + (offs_m + 0) * hidden_dim + ty + 64] = __float2half(acc[tx * 4 + 0][ty + 64]);
    }
    if (offs_m + 1 < seq_len) {
        out[qvk_offset + (offs_m + 1) * hidden_dim + ty] = __float2half(acc[tx * 4 + 1][ty]);
        out[qvk_offset + (offs_m + 1) * hidden_dim + ty + 64] = __float2half(acc[tx * 4 + 1][ty + 64]);
    }
    if (offs_m + 2 < seq_len) {
        out[qvk_offset + (offs_m + 2) * hidden_dim + ty] = __float2half(acc[tx * 4 + 2][ty]);
        out[qvk_offset + (offs_m + 2) * hidden_dim + ty + 64] = __float2half(acc[tx * 4 + 2][ty + 64]);
    }
    if (offs_m + 3 < seq_len) {
        out[qvk_offset + (offs_m + 3) * hidden_dim + ty] = __float2half(acc[tx * 4 + 3][ty]);
        out[qvk_offset + (offs_m + 3) * hidden_dim + ty + 64] = __float2half(acc[tx * 4 + 3][ty + 64]);
    }

}


__global__ void sparse_attention_prefill_fwd_kernel(
    at::Half* Q, at::Half* K, at::Half* V, float sm_scale, at::Half* out, int* lut, float* m_i, float* l_i, 
    int bsz, int head, int seq_len, int hidden_dim, int lut_block, int lut_size, 
    int NNZ, float* p
) {

    int start_m = blockIdx.x;
    int off_hz = blockIdx.y;
    int lut_indicator = blockIdx.y % head;
    int qvk_offset = off_hz * (seq_len * hidden_dim);
    int lut_offset = lut_indicator * (lut_block * lut_size);
    int ml_offset = off_hz * seq_len;

    int offs_m = start_m * 64 + threadIdx.x;
    int offs_n = threadIdx.y * 4;

    // __shared__ float m_i[64];
    // m_i[threadIdx.x] = -INFINITY;
    // __shared__ float l_i[64];
    // l_i[threadIdx.x] = 0;
    __shared__ float acc[64][128];
    for (int i = 0; i < hidden_dim; i++) {
        acc[threadIdx.x][i] = 0;
    }
    __syncthreads();

    float qk_scale = sm_scale * 1.44269504;

    // __shared__ half q[64][128];
    // for (int i = 0; i < hidden_dim; i++) {
    //     if (offs_m < seq_len) {
    //         q[threadIdx.x][i] = __float2half(__half2float(Q[qvk_offset + offs_m * hidden_dim + i]) * qk_scale);
    //     } else {
    //         q[threadIdx.x][i] = 0;
    //     }
    // }
    // __syncthreads();

    int last_nnz_id = -1;

    // __shared__ half k[64][128];
    __shared__ float qk[64][64];
    // __shared__ float m_ij[64];
    // __shared__ float l_ij[64];
    // __shared__ float m_i_new[64];
    // __shared__ float l_i_new[64];
    // __shared__ float p_scale[64];
    // __shared__ float acc_scale[64];
    // __shared__ half v[64][128];

    for (int nnz_id = 0; nnz_id < NNZ; nnz_id++) {
        int present_nnz_id = lut[lut_offset + start_m * lut_size + nnz_id];
        int start_n = present_nnz_id * 64;

        /* load k block*/
        // for (int i = 0; i < hidden_dim; i++) {
        //     if (start_n + threadIdx.y < seq_len) {
        //         k[threadIdx.y][i] = K[qvk_offset + (start_n + threadIdx.y) * hidden_dim + i];
        //     } else {
        //         k[threadIdx.y][i] = 0;
        //     }
        // }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            qk[threadIdx.x][threadIdx.y * 4 + i] = 0;
        }
        // qk[threadIdx.x][threadIdx.y] = 0;
        __syncthreads();

        /* q * k */
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (offs_m < seq_len && start_n + offs_n + i < seq_len && offs_m >= start_n + offs_n + i) {
                // float temp = 0;
                for (int j = 0; j < hidden_dim; j++) {
                    // if (offs_m == 0 && start_n + offs_n + i == 0) {
                    //     printf("%f * %f * %f \n", __half2float(Q[qvk_offset + offs_m * hidden_dim + j]), qk_scale, __half2float(K[qvk_offset + (start_n + offs_n + i) * hidden_dim + j]));
                    // }
                    qk[threadIdx.x][threadIdx.y * 4 + i] += __half2float(Q[qvk_offset + offs_m * hidden_dim + j]) * qk_scale * __half2float(K[qvk_offset + (start_n + offs_n + i) * hidden_dim + j]);
                }
                // qk[threadIdx.x][threadIdx.y * 4 + i] = temp;
            } else {
                qk[threadIdx.x][threadIdx.y * 4 + i] = -INFINITY;
            }
        }
        __syncthreads();
        for (int i = 0; i < 4; i++) {
            p[threadIdx.x * 64 + threadIdx.y * 4 + i] = qk[threadIdx.x][threadIdx.y * 4 + i];
        }
        __syncthreads();

        /* m_ij */
        float m_ij = qk[threadIdx.x][0];
        for (int i = 0; i < 64; i++) {
            if (m_ij < qk[threadIdx.x][i]) {
                m_ij = qk[threadIdx.x][i];
            }
        }
        __syncthreads();
        // if (threadIdx.y == 0 && offs_m < seq_len) {
        //     printf("m %d m_ij is %f \n", offs_m, m_ij);
        // }

        /* p */
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (m_ij == -INFINITY) {
                qk[threadIdx.x][offs_n + i] = 0;
            } else {
                qk[threadIdx.x][offs_n + i] = exp2f(qk[threadIdx.x][offs_n + i] - m_ij);
            }
        }
        __syncthreads();
        // for (int i = 0; i < 4; i++) {
        //     p[threadIdx.x * 64 + threadIdx.y * 4 + i] = qk[threadIdx.x][threadIdx.y * 4 + i];
        // }
        // __syncthreads();
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (last_nnz_id == present_nnz_id) {
                qk[threadIdx.x][offs_n + i] = 0;
            }
        }
        __syncthreads();

        /* l_ij */
        float l_ij = 0;
        for (int i = 0; i < 64; i++) {
            l_ij += qk[threadIdx.x][i];
        }
        // if (threadIdx.y == 0 && offs_m < seq_len) {
        //     printf("m %d, l_ij %f \n", offs_m, l_ij);
        // }
        __syncthreads();

        /* update m_i and l_i */
        if (threadIdx.y == 0 && offs_m < seq_len) {
            // float m_i_new = max(m_i[ml_offset + offs_m], m_ij);
            float m_i_new = m_ij;
            if (m_i_new < m_i[ml_offset + offs_m]) {
                m_i_new = m_i[ml_offset + offs_m];
            }
            float alpha = exp2f(m_i[ml_offset + offs_m] - m_i_new);
            float beta = exp2f(m_ij - m_i_new);
            l_i[ml_offset + offs_m] *= alpha;
            float l_i_new = l_i[ml_offset + offs_m] + beta * l_ij;

            float p_scale = beta / l_i_new;
            // if (std::isnan(p_scale)) {
            //     printf("p_scale is nan, offs_m is %d, m_ij is %f, m_i[ml_offset + offs_m] is %f, l_i[ml_offset + offs_m] is %f, l_ij is %f \n", offs_m, m_ij, m_i[ml_offset + offs_m], l_i[ml_offset + offs_m], l_ij);
            // }
            for (int i = 0; i < 64; i++) {
                qk[threadIdx.x][i] *= p_scale;
            }

            float acc_scale = l_i[ml_offset + offs_m] / l_i_new;
            // if (std::isnan(acc_scale)) {
            //     printf("acc_scale is nan, offs_m is %d, m_ij is %f, m_i[ml_offset + offs_m] is %f, l_i[ml_offset + offs_m] is %f, l_ij is %f \n", offs_m, m_ij, m_i[ml_offset + offs_m], l_i[ml_offset + offs_m], l_ij);
            // }
            for (int i = 0; i < hidden_dim; i++) {
                acc[threadIdx.x][i] *= acc_scale;
            }

            // printf("off_hz %d, m %d, m_ij %f, alpha %f, beta %f, l_ij %f, p_scale %f, acc_scale %f, l_i_new %f \n", off_hz,  offs_m, m_ij, alpha, beta,  l_ij, p_scale, acc_scale, l_i_new);

            l_i[ml_offset + offs_m] = l_i_new;
            m_i[ml_offset + offs_m] = m_i_new;
        }
        __syncthreads();

        /* p * v */
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (start_n + offs_n + i < seq_len) {
                for (int j = 0; j < hidden_dim; j++) {
                    float temp = qk[threadIdx.x][offs_n + i] * __half2float(V[qvk_offset + (start_n + offs_n + i) * hidden_dim + j]);
                    atomicAdd(&acc[threadIdx.x][j], temp);
                }
            }
        }
        
        last_nnz_id = present_nnz_id;
        __syncthreads();
    }

    if (threadIdx.y == 0 && offs_m < seq_len) {
        for (int i = 0; i < hidden_dim; i++) {
            out[qvk_offset + offs_m * hidden_dim + i] = __float2half(acc[threadIdx.x][i]);
        }
    }
    __syncthreads();
}


__global__ void add_kernel(at::Half* a, at::Half* b, at::Half* c) {
    // int eid = blockIdx.x * blockDim.x + threadIdx.x;
    // int eid = threadIdx.x * blockDim.y + threadIdx.y;
    int eid = threadIdx.x;
    // printf("%d ", threadIdx.x);
    // printf("%f \n", a[threadIdx.x]);
    // printf("a[%d] = %f, b[%d] = %f, c[%d] = %f\n", threadIdx.x, a[threadIdx.x], threadIdx.x, b[threadIdx.x], threadIdx.x, c[threadIdx.x]);
    c[eid] = __float2half(__half2float(a[eid]) + __half2float(b[eid]));
    // c[eid] = __hadd(a[eid], b[eid]);
}

