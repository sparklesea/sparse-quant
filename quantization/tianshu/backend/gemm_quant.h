// #pragma once
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
// #include "utils.h"

////////////////////////////// op for PyTorch //////////////////////////////
// at::Tensor Gemm_awq_pure(at::Tensor I,
//             at::Tensor W,
//             at::Tensor W_zeros_scales,
//             c10::optional<at::Tensor> WorkSpace,
//             // at::Tensor O,
//             const int N,
//             const int group_size);

at::Tensor gemm_awq_ut(at::Tensor I, at::Tensor W, at::Tensor W_zeros_scales,
                       const int M, const int N, const int K,
                       const int group_size);

void test_wmma_float(at::Tensor A, at::Tensor B, at::Tensor C);

void test_wmma_32(at::Tensor A, at::Tensor B, at::Tensor C);

void test_wmma(at::Tensor A, at::Tensor B, at::Tensor C);