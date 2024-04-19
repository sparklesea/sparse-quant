
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



__global__ void sparse_attention_prefill_fwd_p(
    at::Half* Q, at::Half* K, float sm_scale, float* P, int* lut, 
    int bsz, int head, int seq_len, int hidden_dim, int lut_block, int lut_size, int NNZ
) {
    int start_m = blockIdx.x;
    int off_hz = blockIdx.y;
    int lut_indicator = blockIdx.y % head;
    int qvk_offset = off_hz * (seq_len * hidden_dim);
    int lut_offset = lut_indicator * (lut_block * lut_size);
    int p_offset = off_hz * (seq_len * seq_len);

    int tx = threadIdx.x / 64;
    int ty = threadIdx.x % 64;

    int offs_m = start_m * 64 + tx * 4;
    int offs_n = ty;

    __shared__ at::Half q[64][128];

    if (offs_m + 0 < seq_len) {
        q[tx * 4 + 0][ty + 0] = Q[qvk_offset + (offs_m + 0) * 128 + ty];
        q[tx * 4 + 0][ty + 64] = Q[qvk_offset + (offs_m + 0) * 128 + ty + 64];
    } else {
        q[tx * 4 + 0][ty + 0] = 0;
        q[tx * 4 + 0][ty + 64] = 0;
    }
    if (offs_m + 1 < seq_len) {
        q[tx * 4 + 1][ty + 0] = Q[qvk_offset + (offs_m + 1) * 128 + ty];
        q[tx * 4 + 1][ty + 64] = Q[qvk_offset + (offs_m + 1) * 128 + ty + 64];
    } else {
        q[tx * 4 + 1][ty + 0] = 0;
        q[tx * 4 + 1][ty + 64] = 0;
    }
    if (offs_m + 2 < seq_len) {
        q[tx * 4 + 2][ty + 0] = Q[qvk_offset + (offs_m + 2) * 128 + ty];
        q[tx * 4 + 2][ty + 64] = Q[qvk_offset + (offs_m + 2) * 128 + ty + 64];
    } else {
        q[tx * 4 + 2][ty + 0] = 0;
        q[tx * 4 + 2][ty + 64] = 0;
    }
    if (offs_m + 3 < seq_len) {
        q[tx * 4 + 3][ty + 0] = Q[qvk_offset + (offs_m + 3) * 128 + ty];
        q[tx * 4 + 3][ty + 64] = Q[qvk_offset + (offs_m + 3) * 128 + ty + 64];
    } else {
        q[tx * 4 + 3][ty + 0] = 0;
        q[tx * 4 + 3][ty + 64] = 0;
    }
    __syncthreads();
    // if (offs_m + 0 < seq_len) {
    //     Q_load[qvk_offset + (offs_m + 0) * 128 + ty + 0] = q[tx * 4 + 0][ty + 0];
    //     Q_load[qvk_offset + (offs_m + 0) * 128 + ty + 64] = q[tx * 4 + 0][ty + 64];
    // }
    // if (offs_m + 1 < seq_len) {
    //     Q_load[qvk_offset + (offs_m + 1) * 128 + ty + 0] = q[tx * 4 + 1][ty + 0];
    //     Q_load[qvk_offset + (offs_m + 1) * 128 + ty + 64] = q[tx * 4 + 1][ty + 64];
    // }
    // if (offs_m + 2 < seq_len) {
    //     Q_load[qvk_offset + (offs_m + 2) * 128 + ty + 0] = q[tx * 4 + 2][ty + 0];
    //     Q_load[qvk_offset + (offs_m + 2) * 128 + ty + 64] = q[tx * 4 + 2][ty + 64];
    // }
    // if (offs_m + 3 < seq_len) {
    //     Q_load[qvk_offset + (offs_m + 3) * 128 + ty + 0] = q[tx * 4 + 3][ty + 0];
    //     Q_load[qvk_offset + (offs_m + 3) * 128 + ty + 64] = q[tx * 4 + 3][ty + 64];
    // }

    at::Half k[128] = {__float2half(0.f)};
    float gsum[4] = {0};

    float qk_scale = sm_scale * 1.44269504;
    int last_nnz_id = -1;

    for (int nnz_id = 0; nnz_id < NNZ; nnz_id++) {
        int present_nnz_id = lut[lut_offset + start_m * lut_size + nnz_id];
        int start_n = present_nnz_id * 64;

        gsum[0] = 0;
        gsum[1] = 0;
        gsum[2] = 0;
        gsum[3] = 0;

        if (start_n + offs_n < seq_len) {
            for (int iter = 0; iter < 16; iter++) {
                *(float4*)(&k[iter * 8]) = *(float4*)(&K[qvk_offset + (start_n + offs_n) * 128 + iter * 8]);
            }
            // for (int iter = 0; iter < 16; iter++) {
            //     *(float4*)(&K_load[qvk_offset + (start_n + offs_n) * 128 + iter * 8]) = *(float4*)(&k[iter * 8]);
            // }
        }

        for (int i = 0; i < 128; i++) {
            gsum[0] += __half2float(q[tx * 4 + 0][i]) * __half2float(k[i]);
            gsum[1] += __half2float(q[tx * 4 + 1][i]) * __half2float(k[i]);
            gsum[2] += __half2float(q[tx * 4 + 2][i]) * __half2float(k[i]);
            gsum[3] += __half2float(q[tx * 4 + 3][i]) * __half2float(k[i]);
        }
        gsum[0] *= qk_scale;
        gsum[1] *= qk_scale;
        gsum[2] *= qk_scale;
        gsum[3] *= qk_scale;

        // P[p_offset + start_m * 64 * seq_len + start_n + tx * 4 + 0 + ty] = gsum[0];
        if (offs_m + 0 < seq_len && start_n + offs_n < seq_len) {
            P[p_offset + (offs_m + 0) * seq_len + start_n + offs_n] = gsum[0];
        }
        if (offs_m + 1 < seq_len && start_n + offs_n < seq_len) {
            P[p_offset + (offs_m + 1) * seq_len + start_n + offs_n] = gsum[1];
        }
        if (offs_m + 2 < seq_len && start_n + offs_n < seq_len) {
            P[p_offset + (offs_m + 2) * seq_len + start_n + offs_n] = gsum[2];
        }
        if (offs_m + 3 < seq_len && start_n + offs_n < seq_len) {
            P[p_offset + (offs_m + 3) * seq_len + start_n + offs_n] = gsum[3];
        }
    }
}

__global__ void sparse_attention_prefill_fwd_p_64(
    at::Half* Q, at::Half* K, float sm_scale, float* P, int* lut, 
    int bsz, int head, int seq_len, int hidden_dim, int lut_block, int lut_size, int NNZ
) {
    int start_m = blockIdx.x;
    int off_hz = blockIdx.y;
    int lut_indicator = blockIdx.y % head;
    int qvk_offset = off_hz * (seq_len * hidden_dim);
    int lut_offset = lut_indicator * (lut_block * lut_size);
    int p_offset = off_hz * (seq_len * seq_len);

    int tx = threadIdx.x / 64;
    int ty = threadIdx.x % 64;

    int offs_m = start_m * 64 + tx * 4;
    int offs_n = ty;

    __shared__ at::Half q[64][64];

    if (offs_m + 0 < seq_len) {
        q[tx * 4 + 0][ty + 0] = Q[qvk_offset + (offs_m + 0) * 64 + ty];
    } else {
        q[tx * 4 + 0][ty + 0] = 0;
    }
    if (offs_m + 1 < seq_len) {
        q[tx * 4 + 1][ty + 0] = Q[qvk_offset + (offs_m + 1) * 64 + ty];
    } else {
        q[tx * 4 + 1][ty + 0] = 0;
    }
    if (offs_m + 2 < seq_len) {
        q[tx * 4 + 2][ty + 0] = Q[qvk_offset + (offs_m + 2) * 64 + ty];
    } else {
        q[tx * 4 + 2][ty + 0] = 0;
    }
    if (offs_m + 3 < seq_len) {
        q[tx * 4 + 3][ty + 0] = Q[qvk_offset + (offs_m + 3) * 64 + ty];
    } else {
        q[tx * 4 + 3][ty + 0] = 0;
    }
    __syncthreads();

    at::Half k[64] = {__float2half(0.f)};
    float gsum[4] = {0};

    float qk_scale = sm_scale * 1.44269504;
    int last_nnz_id = -1;

    for (int nnz_id = 0; nnz_id < NNZ; nnz_id++) {
        int present_nnz_id = lut[lut_offset + start_m * lut_size + nnz_id];
        int start_n = present_nnz_id * 64;

        gsum[0] = 0;
        gsum[1] = 0;
        gsum[2] = 0;
        gsum[3] = 0;

        if (start_n + offs_n < seq_len) {
            for (int iter = 0; iter < 8; iter++) {
                *(float4*)(&k[iter * 8]) = *(float4*)(&K[qvk_offset + (start_n + offs_n) * 64 + iter * 8]);
            }
            // for (int iter = 0; iter < 16; iter++) {
            //     *(float4*)(&K_load[qvk_offset + (start_n + offs_n) * 128 + iter * 8]) = *(float4*)(&k[iter * 8]);
            // }
        }

        for (int i = 0; i < 64; i++) {
            gsum[0] += __half2float(q[tx * 4 + 0][i]) * __half2float(k[i]);
            gsum[1] += __half2float(q[tx * 4 + 1][i]) * __half2float(k[i]);
            gsum[2] += __half2float(q[tx * 4 + 2][i]) * __half2float(k[i]);
            gsum[3] += __half2float(q[tx * 4 + 3][i]) * __half2float(k[i]);
        }
        gsum[0] *= qk_scale;
        gsum[1] *= qk_scale;
        gsum[2] *= qk_scale;
        gsum[3] *= qk_scale;

        // P[p_offset + start_m * 64 * seq_len + start_n + tx * 4 + 0 + ty] = gsum[0];
        if (offs_m + 0 < seq_len && start_n + offs_n < seq_len) {
            P[p_offset + (offs_m + 0) * seq_len + start_n + offs_n] = gsum[0];
        }
        if (offs_m + 1 < seq_len && start_n + offs_n < seq_len) {
            P[p_offset + (offs_m + 1) * seq_len + start_n + offs_n] = gsum[1];
        }
        if (offs_m + 2 < seq_len && start_n + offs_n < seq_len) {
            P[p_offset + (offs_m + 2) * seq_len + start_n + offs_n] = gsum[2];
        }
        if (offs_m + 3 < seq_len && start_n + offs_n < seq_len) {
            P[p_offset + (offs_m + 3) * seq_len + start_n + offs_n] = gsum[3];
        }
    }
}


