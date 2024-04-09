
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

using namespace std;


__global__ void sparse_attention_prefill_fwd_warp_kernel(
    __half* Q, __half* K, __half* V, float sm_scale, __half* out, int* lut, int bsz, int head, int seq_len, int hidden_dim, int lut_block, int lut_size, int NNZ
) {
    int start_m = blockIdx.x;
    int off_hz = blockIdx.y;
    int lut_indicator = blockIdx.y % head;
    int qvk_offset = off_hz
}


__global__ void sparse_attention_prefill_fwd_kernel(
    at::Half* Q, at::Half* K, at::Half* V, float sm_scale, at::Half* out, int* lut, float* m_i, float* l_i, 
    int bsz, int head, int seq_len, int hidden_dim, int lut_block, int lut_size, 
    int NNZ
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
        // if (offs_m < seq_len && start_n + offs_n < seq_len && offs_m >= start_n + offs_n) {
        //     for (int i = 0; i < hidden_dim; i++) {
        //         qk[threadIdx.x][threadIdx.y] += __half2float(Q[qvk_offset + offs_m * hidden_dim + i]) * qk_scale * __half2float(K[qvk_offset + (start_n + offs_n) * hidden_dim + i]);
        //     }
        // } else {
        //     qk[threadIdx.x][threadIdx.y] = -INFINITY;
        // }
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

        // if (threadIdx.y == 0) {
        //     float temp = qk[threadIdx.x][0];
        //     for (int i = 0; i < blockDim.y; i++) {
        //         if (temp < qk[threadIdx.x][i]) {
        //             temp = qk[threadIdx.x][i];
        //         }
        //     }
        //     m_ij[threadIdx.x] = temp;
        // }
        // __syncthreads();
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

        // if (last_nnz_id != present_nnz_id && m_ij != -INFINITY) {
        //     qk[threadIdx.x][threadIdx.y] = expf(qk[threadIdx.x][threadIdx.y] - m_ij);
        // } else {
        //     qk[threadIdx.x][threadIdx.y] = 0;
        // }
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
        // if (threadIdx.y == 0) {
        //     l_ij[threadIdx.x] = 0;
        //     for (int i = 0; i < blockDim.y; i++) {
        //         l_ij[threadIdx.x] += qk[threadIdx.x][i];
        //     }
        // }
        // __syncthreads();

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
        // for (int i = 0; i < 4; i++) {
        //     p[threadIdx.x * 64 + threadIdx.y * 4 + i] = qk[threadIdx.x][threadIdx.y * 4 + i];
        // }
        // __syncthreads();
        // if (threadIdx.y == 0) {
        //     if (m_i[threadIdx.x] > m_ij[threadIdx.x]) {
        //         m_i_new[threadIdx.x] = m_i[threadIdx.x];
        //     } else {
        //         m_i_new[threadIdx.x] = m_ij[threadIdx.x];
        //     }
        //     float alpha = expf(m_i[threadIdx.x] - m_i_new[threadIdx.x]);
        //     float beta = expf(m_ij[threadIdx.x] - m_i_new[threadIdx.x]);
        //     l_i[threadIdx.x] *= alpha;
        //     l_i_new[threadIdx.x] = l_i[threadIdx.x] + beta * l_ij[threadIdx.x];

        //     p_scale[threadIdx.x] = beta / l_i_new[threadIdx.x];
        //     acc_scale[threadIdx.x] = l_i[threadIdx.x] / l_i_new[threadIdx.x];

        //     for (int i = 0; i < hidden_dim; i++) {
        //         acc[threadIdx.x][i] *= acc_scale[threadIdx.x];
        //     }
        // }
        // __syncthreads();
        // qk[threadIdx.x][threadIdx.y] *= p_scale[threadIdx.x];
        // __syncthreads();

        /* load v */
        // for (int i = 0; i < hidden_dim; i++) {
        //     if (start_n + offs_n < seq_len) {
        //         v[threadIdx.y][i] = V[qvk_offset + (start_n + offs_n) * hidden_dim + i];
        //     } else {
        //         v[threadIdx.y][i] = 0;
        //     }
        // }

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

        // if (start_n + offs_n < seq_len) {
        //     for (int i = 0; i < hidden_dim; i++) {
        //         float temp = qk[threadIdx.x][threadIdx.y] * __half2float(V[qvk_offset + (start_n + offs_n) * hidden_dim + i]);
        //         atomicAdd(&acc[threadIdx.x][i], temp);
        //     }
        // }

        /* update m_i and l_i */
        // if (threadIdx.y == 0) {
        //     // l_i[threadIdx.x] = l_i_new[threadIdx.x];
        //     // m_i[threadIdx.x] = m_i_new[threadIdx.x];
        //     l_i[threadIdx.x] = l_i_new;
        //     m_i[threadIdx.x] = m_i_new;
        // }
        last_nnz_id = present_nnz_id;
        __syncthreads();
    }
    // if (threadIdx.y == 0 && offs_m < seq_len) {
    //     for (int i = 0; i < 64; i++) {
    //         acc[threadIdx.x][i] = qk[threadIdx.x][i];
    //     }
    // }
    if (threadIdx.y == 0 && offs_m < seq_len) {
        for (int i = 0; i < hidden_dim; i++) {
            out[qvk_offset + offs_m * hidden_dim + i] = __float2half(acc[threadIdx.x][i]);
        }
    }
    __syncthreads();
}


__global__ void add_kernel(at::Half* a, at::Half* b, at::Half* c) {
    // int eid = blockIdx.x * blockDim.x + threadIdx.x;
    int eid = threadIdx.x * blockDim.y + threadIdx.y;
    // printf("%d ", threadIdx.x);
    // printf("%f \n", a[threadIdx.x]);
    // printf("a[%d] = %f, b[%d] = %f, c[%d] = %f\n", threadIdx.x, a[threadIdx.x], threadIdx.x, b[threadIdx.x], threadIdx.x, c[threadIdx.x]);
    c[eid] = __float2half(__half2float(a[eid]) + __half2float(b[eid]));
    // c[eid] = __hadd(a[eid], b[eid]);
}

