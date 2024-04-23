#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "w4a16_gemm.cuh"

void test_wmma_float(at::Tensor A, at::Tensor B, at::Tensor C){
    const int warp_size = 64;
    const int M = 16, N = 16, K = 16;
	dim3 gridDim, blockDim;
	// 16 warps in one block

    WMMAF16TensorCore<<<(1, 1), (256)>>>(
        reinterpret_cast<float *>(A.data_ptr()), 
        reinterpret_cast<float *>(B.data_ptr()), 
        reinterpret_cast<float *>(C.data_ptr()));
}

void test_wmma_32(at::Tensor A, at::Tensor B, at::Tensor C){
    const int warp_size = 64;
    const int M = 16, N = 16, K = 32;
	dim3 gridDim, blockDim;
	// 16 warps in one block

    WMMAF16TensorCore_32<<<(1, 1), (64)>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()), 
        reinterpret_cast<half *>(B.data_ptr<at::Half>()), 
        reinterpret_cast<float *>(C.data_ptr()));
}

void test_wmma(at::Tensor A, at::Tensor B, at::Tensor C){
    const int warp_size = 64;
    const int M = 16, N = 16, K = 16;
	dim3 gridDim, blockDim;
	// 16 warps in one block

    WMMAF16TensorCore<<<(1, 1), (128)>>>(
        reinterpret_cast<float *>(A.data_ptr()), 
        reinterpret_cast<float *>(B.data_ptr()), 
        reinterpret_cast<float *>(C.data_ptr()),
        M, N, K,
        A.size(0), B.size(1), B.size(0));
}



void w4a16_gemm(at::Tensor X, at::Tensor W, at::Tensor zeros_scales,
                at::Tensor O, const uint group_size) {
    if (O.dim() == 3 && O.size(2) != W.size(0)){
        throw std::invalid_argument("output dim mismatch!");
    }

    if (O.dim() == 4 && O.size(2) * O.size(3) != W.size(0)){
        throw std::invalid_argument("output dim mismatch!");
    }   
    if (X.size(2) % 128 != 0) {
        throw std::invalid_argument("embbed dim must be a multiple of 128!");
    }

    // support prefill
    int bs = X.size(0) * X.size(1);
    int dim_in = X.size(2);
    int dim_out = W.size(0);

    if (bs == 1) {
        // std::cout << "bs==1" << std::endl;
        w4a16_bs1_kernel<<<dim3(1, dim_out / 2), dim3(128, 1)>>>(
            reinterpret_cast<uint32_t *>(W.data_ptr()),
            reinterpret_cast<half2 *>(zeros_scales.data_ptr<at::Half>()),
            reinterpret_cast<half *>(X.data_ptr<at::Half>()),
            reinterpret_cast<half *>(O.data_ptr<at::Half>()), dim_in,
            DIV_UP(dim_in, 128), group_size);
    } else if (bs == 2) {
        // batch size 2, seq 1. OR
        // batch size 1, seq 2.
        // std::cout << "bs==2" << std::endl;
        w4a16_bs2_kernel<<<dim3(1, dim_out / 2), dim3(128, 1)>>>(
            reinterpret_cast<uint32_t *>(W.data_ptr()),
            reinterpret_cast<half2 *>(zeros_scales.data_ptr<at::Half>()),
            reinterpret_cast<half *>(X.data_ptr<at::Half>()),
            reinterpret_cast<half *>(O.data_ptr<at::Half>()), dim_in, dim_out,
            DIV_UP(dim_in, 128), group_size);
    } 
    else {
        // speedup for batch size 1 to 16
        // w4a16_gemm_wmma_kernel_32<16, 16, 256, 264, 24>
        w4a16_gemm_wmma_kernel_float<16, 16, 256, 264, 24>
            <<<dim3(DIV_UP(dim_out, 16), DIV_UP(bs, 16)), dim3(256)>>>(
                reinterpret_cast<uint32_t *>(W.data_ptr()),
                reinterpret_cast<half2 *>(zeros_scales.data_ptr<at::Half>()),
                reinterpret_cast<half *>(X.data_ptr<at::Half>()),
                reinterpret_cast<half *>(O.data_ptr<at::Half>()),
                bs, dim_out, dim_in,
                group_size);
    }
}


at::Tensor Gemm_awq_pure(at::Tensor I,
                    at::Tensor W,
                    at::Tensor W_zeros_scales,
                    c10::optional<at::Tensor> WorkSpace,
                    at::Tensor O,
                    const int group_size) {

    CHECK_DEVICE(I);CHECK_DEVICE(W);
    CHECK_DEVICE(W_zeros_scales);CHECK_DEVICE(O);

    CHECK_CONTIGUOUS(I);CHECK_CONTIGUOUS(W);
    CHECK_CONTIGUOUS(W_zeros_scales);CHECK_CONTIGUOUS(O);

    CHECK_DTYPE(I, at::kHalf);
    //CHECK_DTYPE(W, at::kInt);
    CHECK_DTYPE(W_zeros_scales, at::kHalf);CHECK_DTYPE(O, at::kHalf);

    w4a16_gemm(I, W, W_zeros_scales, O, group_size);

    return O;
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

at::Tensor dequant(at::Tensor W, at::Tensor zeros_scales, const int N, const int K, const int group_size) {
    at::Tensor W_load = torch::zeros({N, K}, at::device(W.device()).dtype(torch::kFloat16));

    dequant_W_kernel<<<dim3(DIV_UP(N, 16)), dim3(256)>>>(
        reinterpret_cast<uint32_t*>(W.data_ptr()), 
        reinterpret_cast<half2*>(zeros_scales.data_ptr<at::Half>()), 
        reinterpret_cast<half*>(W_load.data_ptr<at::Half>()), 
        N, K, group_size
    );

    return W_load;
}
