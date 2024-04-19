
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/python.h>
#include <torch/torch.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

torch::Tensor add(torch::Tensor a, torch::Tensor b);


torch::Tensor sparse_attention_prefill_p(
    torch::Tensor Q, torch::Tensor K, float sm_scale, torch::Tensor lut
);

torch::Tensor sparse_attention_prefill_p_64(
    torch::Tensor Q, torch::Tensor K, float sm_scale, torch::Tensor lut
);
