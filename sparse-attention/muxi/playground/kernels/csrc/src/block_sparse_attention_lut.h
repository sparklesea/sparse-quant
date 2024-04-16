
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/python.h>
#include <torch/torch.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// torch::Tensor sparse_attention_prefill_warp(
std::vector<torch::Tensor> sparse_attention_prefill_warp(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, float sm_scale, torch::Tensor lut
);

// torch::Tensor sparse_attention_prefill(
std::vector<torch::Tensor> sparse_attention_prefill(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, float sm_scale, torch::Tensor lut
);

torch::Tensor add(torch::Tensor a, torch::Tensor b);

