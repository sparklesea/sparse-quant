

#include <cmath>

#include <torch/all.h>
#include <torch/extension.h>
#include <torch/python.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <cuda_fp16.h>

#include "block_sparse_attention_lut.h"
#include "../include/block_sparse_attention_lut.cuh"

torch::Tensor sparse_attention_prefill(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, float sm_scale, torch::Tensor lut
) {
    int bsz = Q.size(0);
    int head = Q.size(1);
    int seq_len = Q.size(2);
    int hidden_dim = Q.size(3);
    int lut_block = lut.size(1);
    int lut_size = lut.size(2);

    auto devid = Q.device().index();

    auto options_out = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, devid);
    auto out = torch::zeros({bsz, head, seq_len, hidden_dim}, options_out);
    auto options_ml = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    // auto m_i = torch::full({bsz, head, seq_len}, -std::numeric_limits<float>::infinity(), options_ml);
    auto m_i = torch::full({bsz, head, seq_len}, -INFINITY, options_ml);
    // auto m_i = torch::full({bsz, head, seq_len}, -99999, options_ml);
    auto l_i = torch::zeros({bsz, head, seq_len}, options_ml);
    auto p = torch::zeros({64, 64}, options_ml);

    dim3 gridDim((seq_len + 64 - 1) / 64, bsz * head, 1);
    dim3 blockDim(64, 16, 1);
    // dim3 gridDim(1, 2, 3);
    // dim3 blockDim(64, 63, 62);
    int NNZ = min((int)(lut.size(2)), (int)(ceil(seq_len / 64.0)));

    // printf("entering kernel and NNZ is %d | %d, final is %d \n", (int)(lut.size(2)), (int)(ceil(seq_len / 64.0)), NNZ);
    // printf("entering kernel and bsz is: %d, head is: %d, seq_len is %d, hidden_dim is %d, lut_block is %d, lut_size is %d, NNZ is %d, gridDim.x is %d, gridDim.y is %d, gridDim.z is %d, blockDim.x is %d, blockDim.y is %d, blockDim.z is %d \n", bsz, head, seq_len, hidden_dim, lut_block, lut_size, NNZ, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);

    sparse_attention_prefill_fwd_kernel<<<gridDim, blockDim>>>(Q.data_ptr<at::Half>(), K.data_ptr<at::Half>(), V.data_ptr<at::Half>(), sm_scale, out.data_ptr<at::Half>(), lut.data_ptr<int>(), m_i.data_ptr<float>(), l_i.data_ptr<float>(), bsz, head, seq_len, hidden_dim, lut_block, lut_size, NNZ);
    // sparse_attention_prefill_fwd_kernel<<<gridDim, blockDim>>>(Q.data_ptr<at::Half>());

    // printf("out kernel \n");

    return out;
}


torch::Tensor add(torch::Tensor a, torch::Tensor b){
    a = a.contiguous();
    b = b.contiguous();
    auto out = torch::zeros_like(a);
    int length = a.size(0);
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(64, 64, 1);
    add_kernel<<<gridDim, blockDim>>>(a.data_ptr<at::Half>(), b.data_ptr<at::Half>(), out.data_ptr<at::Half>());
    // auto arr = a.data_ptr<float>();
    // printf("%f, %f, %f, %f, %f \n", arr[0], arr[1], arr[2], arr[3], arr[4]);
    return out;
}




