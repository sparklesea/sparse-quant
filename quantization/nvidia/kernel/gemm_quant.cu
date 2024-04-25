#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "gemm_s4_f16/gemm_s4_f16.h"
#include "utils.h"

void gemm_awq_api(half* I, int* W, half* W_zeros_scales, half* O, half* workspace,
              int m, int k, int n, const int group_size) {
    // if use lmdeploy(M <= 8 || M > 64), WorkSpace can be a None Tensor

    // if use marlin(8 < M <= 64), WorkSpace value must be all zero,
    // and its length must not less than [output_dim // 128 * 16]

    // NOTE: only use lmdeploy NOW.

    // lmdeploy
    static turbomind::GemmS4F16 gemm_s4_f16;

    int algo_id = -1;
    if (m > 16 && m <= 24 && k == 11008 && n == 4096) {
        algo_id = 12;
    }

    gemm_s4_f16.Run(O,
                    reinterpret_cast<const unsigned int*>(W),
                    I,
                    reinterpret_cast<const half2*>(W_zeros_scales),
                    n,
                    m,
                    k,
                    group_size,
                    gemm_s4_f16.Type::kGemm,
                    algo_id);
}

void Gemm_awq_pure(at::Tensor I,
                    at::Tensor W,
                    at::Tensor W_zeros_scales,
                    c10::optional<at::Tensor> WorkSpace,
                    at::Tensor O,
                    const int group_size) {
    // I: [bs, seq, input_dim], must be contiguous
    // W:               int, length: input_dim * output_dim / 8,              must be contiguous
    // W_zeros_scales: half, length: input_dim * output_dim / group_size * 2, must be contiguous
    // O: [bs, seq, output_dim], must be contiguous

    // WorkSpace(half), M = bs * seq
    // if use lmdeploy(M <= 8 || M > 64), WorkSpace can be a None Tensor
    // if use marlin(8 < M <= 64), WorkSpace value must be all zero,
    // and its length must not less than [output_dim // 128 * 16]

    CHECK_DEVICE(I);
    CHECK_DEVICE(W);
    CHECK_DEVICE(W_zeros_scales);
    CHECK_DEVICE(O);

    CHECK_CONTIGUOUS(I);
    CHECK_CONTIGUOUS(W);
    CHECK_CONTIGUOUS(W_zeros_scales);
    CHECK_CONTIGUOUS(O);

    CHECK_DTYPE(I, at::kHalf);
    // CHECK_DTYPE(W, at::kInt);
    CHECK_DTYPE(W_zeros_scales, at::kHalf);
    CHECK_DTYPE(O, at::kHalf);

    CHECK_DIMS(I, 3);
    CHECK_DIMS(W, 2);
    CHECK_DIMS(W_zeros_scales, 2);
    CHECK_DIMS(O, 3);

    if (WorkSpace.has_value()) {
        CHECK_DEVICE(WorkSpace.value());
        CHECK_CONTIGUOUS(WorkSpace.value());
        CHECK_DTYPE(WorkSpace.value(), at::kHalf);
        CHECK_NUMEL(WorkSpace.value(), 0);
    }

    int m = I.size(0) * I.size(1);
    int k = I.size(2);
    int n = O.size(2);

    gemm_awq_api(reinterpret_cast<half*>(I.data_ptr<at::Half>()),
             reinterpret_cast<int*>(W.data_ptr()),
             reinterpret_cast<half*>(W_zeros_scales.data_ptr<at::Half>()),
             reinterpret_cast<half*>(O.data_ptr<at::Half>()),
             nullptr,  // use lmdeploy NOW
             m, k, n,
             group_size);
}

at::Tensor gemm_awq_ut(at::Tensor I, at::Tensor W, at::Tensor W_zeros_scales,
                       const int M, const int N, const int K,
                       const int group_size) {
    // I:               [bs, seqlen, K]
    // W:               [K, N // 8], int
    // W_zeros_scales:  [K // group_size, N * 2]
    // O:               [bs, seqlen, N]

    // WorkSpace(half), M = bs * seqlen
    // if use lmdeploy, WorkSpace can be a None Tensor
    // if use marlin, WorkSpace value must be all zero and its length must not less than [N // 128 * 16]

    // NOTE: only use lmdeploy NOW

    at::Tensor O = torch::empty({I.size(0), I.size(1), N},
                                at::device(I.device()).dtype(I.dtype()));

    Gemm_awq_pure(I,
                    W,
                    W_zeros_scales,
                    c10::nullopt,
                    O,
                    group_size);

    return O;
}