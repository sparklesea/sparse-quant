// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

namespace turbomind {

void reformat_s4_k8_m(uint32_t* dst, const uint32_t* src, int m, int k, cudaStream_t st = {});

void reformat_s4_k_m8(uint32_t* dst, const uint32_t* src, int m, int k, cudaStream_t st = {});

void convert_s4_k_m8(uint32_t* A_dst,
                     half2* Q_dst,
                     half* workspace,
                     const uint32_t* A_src,
                     const half* scales,
                     const half* zeros,  // const uint32_t* qzeros,
                     int m,
                     int k,
                     int group_size,
                     cudaStream_t st = {});

void transpose_qk_s4_k_m8_hf(uint32_t* dst, const uint32_t* src, int m, int k, int size_per_head, cudaStream_t st = {});

void fuse_w1_w3_s4_k_m8(uint32_t* dst, const uint32_t* src, int m, int k, cudaStream_t st = {});

void dequantize_s4(uint4* dst, const uint32_t* src, size_t count, cudaStream_t st = {});

}  // namespace turbomind

void convert_ours_to_awq(torch::Tensor qweight_src,
                         torch::Tensor qweight_dst,
                         int in_features,
                         int out_features);

void convert_awq_to_lmdeploy(torch::Tensor qweight,
                             torch::Tensor qweight_dst,
                             int in_features,
                             int out_features);

void transpose_merge_zeros_scales(torch::Tensor zeros,
                                  torch::Tensor scales,
                                  torch::Tensor zeros_scales,
                                  int in_features,
                                  int out_features,
                                  int group_size);