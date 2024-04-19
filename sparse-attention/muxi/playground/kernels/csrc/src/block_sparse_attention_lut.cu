

#include <cmath>

#include <torch/all.h>
#include <torch/extension.h>
#include <torch/python.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <cuda_fp16.h>

#include "block_sparse_attention_lut.h"
#include "../include/block_sparse_attention_lut.cuh"


torch::Tensor add(torch::Tensor a, torch::Tensor b){
    a = a.contiguous();
    b = b.contiguous();
    auto out = torch::zeros_like(a);
    int length = a.size(0);
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(1024, 1, 1);
    add_kernel<<<gridDim, blockDim>>>(a.data_ptr<at::Half>(), b.data_ptr<at::Half>(), out.data_ptr<at::Half>());
    // auto arr = a.data_ptr<float>();
    // printf("%f, %f, %f, %f, %f \n", arr[0], arr[1], arr[2], arr[3], arr[4]);
    return out;
}


torch::Tensor sparse_attention_prefill_p(
    torch::Tensor Q, torch::Tensor K, float sm_scale, torch::Tensor lut
) {
    int bsz = Q.size(0);
    int head = Q.size(1);
    int seq_len = Q.size(2);
    int hidden_dim = Q.size(3);
    int lut_block = lut.size(1);
    int lut_size = lut.size(2);

    auto devid = Q.device().index();

    auto options_p = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    // auto P = torch::zeros({bsz, head, seq_len, seq_len}, options_p);
    auto P = torch::full({bsz, head, seq_len, seq_len}, -INFINITY, options_p);
    // auto Q_load = torch::zeros_like(Q);
    // auto K_load = torch::zeros_like(K);

    dim3 gridDim((seq_len + 64 - 1) / 64, bsz * head, 1);
    dim3 blockDim(1024, 1, 1);
    int NNZ = min((int)(lut.size(2)), (int)(ceil(seq_len / 64.0)));

    // printf("entering sparse_attention_prefill_p kernel and bsz is: %d, head is: %d, seq_len is %d, hidden_dim is %d, lut_block is %d, lut_size is %d, NNZ is %d, gridDim.x is %d, gridDim.y is %d, gridDim.z is %d, blockDim.x is %d, blockDim.y is %d, blockDim.z is %d \n", bsz, head, seq_len, hidden_dim, lut_block, lut_size, NNZ, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);

    sparse_attention_prefill_fwd_p<<<gridDim, blockDim>>>(Q.data_ptr<at::Half>(), K.data_ptr<at::Half>(), sm_scale, P.data_ptr<float>(), lut.data_ptr<int>(), bsz, head, seq_len, hidden_dim, lut_block, lut_size, NNZ);

    // return {P, Q_load, K_load};
    return P;
}

torch::Tensor sparse_attention_prefill_p_64(
    torch::Tensor Q, torch::Tensor K, float sm_scale, torch::Tensor lut
) {
    int bsz = Q.size(0);
    int head = Q.size(1);
    int seq_len = Q.size(2);
    int hidden_dim = Q.size(3);
    int lut_block = lut.size(1);
    int lut_size = lut.size(2);

    auto devid = Q.device().index();

    auto options_p = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    // auto P = torch::zeros({bsz, head, seq_len, seq_len}, options_p);
    auto P = torch::full({bsz, head, seq_len, seq_len}, -INFINITY, options_p);
    // auto Q_load = torch::zeros_like(Q);
    // auto K_load = torch::zeros_like(K);

    dim3 gridDim((seq_len + 64 - 1) / 64, bsz * head, 1);
    dim3 blockDim(1024, 1, 1);
    int NNZ = min((int)(lut.size(2)), (int)(ceil(seq_len / 64.0)));

    // printf("entering sparse_attention_prefill_fwd_p_64 kernel and bsz is: %d, head is: %d, seq_len is %d, hidden_dim is %d, lut_block is %d, lut_size is %d, NNZ is %d, gridDim.x is %d, gridDim.y is %d, gridDim.z is %d, blockDim.x is %d, blockDim.y is %d, blockDim.z is %d \n", bsz, head, seq_len, hidden_dim, lut_block, lut_size, NNZ, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);

    sparse_attention_prefill_fwd_p_64<<<gridDim, blockDim>>>(Q.data_ptr<at::Half>(), K.data_ptr<at::Half>(), sm_scale, P.data_ptr<float>(), lut.data_ptr<int>(), bsz, head, seq_len, hidden_dim, lut_block, lut_size, NNZ);

    // return {P, Q_load, K_load};
    return P;
}


