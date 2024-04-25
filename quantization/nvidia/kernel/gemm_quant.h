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

void transpose_merge_zeros_scales(torch::Tensor zeros,
                                  torch::Tensor scales,
                                  torch::Tensor zeros_scales,
                                  int in_features,
                                  int out_features,
                                  int group_size);

void convert_ours_to_awq(torch::Tensor qweight_src,
                         torch::Tensor qweight_dst,
                         int in_features,
                         int out_features);

void convert_awq_to_lmdeploy(torch::Tensor qweight,
                             torch::Tensor qweight_dst,
                             int in_features,
                             int out_features);