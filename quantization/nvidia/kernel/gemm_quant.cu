#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "../../../kernels/nvidia/gemm_s4_f16/gemm_s4_f16.h"
#include "../../../kernels/nvidia/marlin/marlin_cuda.h"
#include "../../../kernels/utils.h"
#include "../../../kernels/nvidia/rope.cuh"
#include "../../../kernels/nvidia/residual.cuh"
#include "../../../kernels/nvidia/activation.cuh"

void gemm_awq_api(half* I, int* W, half* W_zeros_scales, half* O, half* workspace,
              int m, int k, int n, const int group_size) {
    // if use lmdeploy(M <= 8 || M > 64), WorkSpace can be a None Tensor

    // if use marlin(8 < M <= 64), WorkSpace value must be all zero,
    // and its length must not less than [output_dim // 128 * 16]

    // NOTE: only use lmdeploy NOW.
    if (1) {
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
    } else {
        // 8 < m <= 64
        // marlin
        int err = marlin_cuda(I,
                              W,
                              O,
                              W_zeros_scales + k * n / group_size,  // scales ptr
                              W_zeros_scales,                       // zeros ptr
                              m,
                              n,
                              k,
                              workspace,
                              group_size,
                              0, 0, -1, -1, -1, 16);

        if (err == ERR_PROB_SHAPE) {
            AT_ERROR(
                "Problem (m=", m, ", n=", n, ", k=", k, ")",
                " not compatible.");
        } else if (err == ERR_KERN_SHAPE) {
            AT_ERROR(
                "No kernel implementation.");
        }
    }
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

// gemm_awq_api + residual
// O: residual & output
void gemm_awq_residual(half* I, int* W, half* W_zeros_scales, half* linear_output, half* O,
                       int m, int k, int n, const int group_size) {
    gemm_awq_api(I, W, W_zeros_scales, linear_output,
             nullptr,  // use lmdeploy NOW
             m, k, n,
             group_size);

    residual_kernel<<<dim3(m, DIV_UP(n, 256)), dim3(128, 1)>>>(O, linear_output, m, n, O);
}

// O: residual & output
void Gemm_awq_residual(at::Tensor I,
                        at::Tensor W,
                        at::Tensor W_zeros_scales,
                        c10::optional<at::Tensor> WorkSpace,
                        at::Tensor O,
                        const int group_size) {
    // I: [bs, seq, input_dim], must be contiguous
    // W:               int, length: input_dim * output_dim / 8,              must be contiguous
    // W_zeros_scales: half, length: input_dim * output_dim / group_size * 2, must be contiguous
    // WorkSpace:    [bs, seq, output_dim], must be contiguous
    // O(residual): [bs, seq, output_dim], must be contiguous

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
        CHECK_NUMEL(WorkSpace.value(), O.numel());
    } else {
        throw std::invalid_argument("WorkSpace must be a valid tensor.");
    }

    int m = I.size(0) * I.size(1);
    int k = I.size(2);
    int n = O.size(2);

    gemm_awq_residual(reinterpret_cast<half*>(I.data_ptr<at::Half>()),
                      reinterpret_cast<int*>(W.data_ptr()),
                      reinterpret_cast<half*>(W_zeros_scales.data_ptr<at::Half>()),
                      reinterpret_cast<half*>(WorkSpace.value().data_ptr<at::Half>()),
                      reinterpret_cast<half*>(O.data_ptr<at::Half>()),
                      m, k, n,
                      group_size);
}

void gemm_awq_packed_silu_dot(half* I, int* W13, half* W13_zeros_scales, half* W13_output, half* O,
                       int m, int k, int n, const int group_size) {
    gemm_awq_api(I, W13, W13_zeros_scales, W13_output,
             nullptr,  // use lmdeploy NOW
             m, k, n * 2,
             group_size);

    // silu(W1O) * W3O ---> O
    silu_dot_kernel<<<256, 256>>>(W13_output,
                                    W13_output + n,
                                    O,
                                    m,
                                    n,
                                    n * 2);
}

void Gemm_awq_packed_silu_dot(at::Tensor I,
                                at::Tensor W,
                                at::Tensor W_zeros_scales,
                                c10::optional<at::Tensor> WorkSpace,
                                at::Tensor O,
                                const int group_size) {
    // I: [bs, seq, input_dim], must be contiguous
    // W:                 int, length: input_dim * (output_dim * 2) / 8, must be contiguous
    // W_zeros_scales:   half, length: input_dim * (output_dim * 2) / group_size * 2, must be contiguous
    // WorkSpace: [bs, seq, output_dim * 2], , must be contiguous
    // O: [bs, seq, output_dim], must be contiguous

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
        CHECK_NUMEL(WorkSpace.value(), O.numel() * 2);
    } else {
        throw std::invalid_argument("WorkSpace must be a valid tensor.");
    }

    int m = I.size(0) * I.size(1);
    int k = I.size(2);
    int n = O.size(2);

    gemm_awq_packed_silu_dot(reinterpret_cast<half*>(I.data_ptr<at::Half>()),
                            reinterpret_cast<int*>(W.data_ptr()),
                            reinterpret_cast<half*>(W_zeros_scales.data_ptr<at::Half>()),
                            reinterpret_cast<half*>(WorkSpace.value().data_ptr<at::Half>()),
                            reinterpret_cast<half*>(O.data_ptr<at::Half>()),
                            m, k, n,
                            group_size);
}

void gemm_awq_silu_dot(half* I, int* W1, int* W2, half* W1_zeros_scales, half* W2_zeros_scales, 
                       half* workspace, half* O,
                       int m, int k, int n, const int group_size) {
    gemm_awq_api(I, W1, W1_zeros_scales, workspace,
             nullptr,  // use lmdeploy NOW
             m, k, n,
             group_size);
    gemm_awq_api(I, W2, W2_zeros_scales, O,
             nullptr,  // use lmdeploy NOW
             m, k, n,
             group_size);

    // silu(W1O) * W2O ---> O
    silu_dot_kernel<<<256, 256>>>(workspace,
                                    O,
                                    O,
                                    m,
                                    n,
                                    n);
}

void Gemm_awq_silu_dot(at::Tensor I,
            at::Tensor W1, 
            at::Tensor W2, 
            at::Tensor W1_zeros_scales,
            at::Tensor W2_zeros_scales,
            c10::optional<at::Tensor> WorkSpace,
            at::Tensor O,
            const int group_size) {
    // I: [bs, seq, input_dim], must be contiguous
    // W1:                int, length: input_dim * output_dim / 8,              must be contiguous
    // W2:                int, length: input_dim * output_dim / 8,              must be contiguous
    // W1_zeros_scales:   half, length: input_dim * output_dim / group_size * 2, must be contiguous
    // W2_zeros_scales:   half, length: input_dim * output_dim / group_size * 2, must be contiguous
    // WorkSpace: [bs, seq, output_dim], must be contiguous
    // O: [bs, seq, output_dim], must be contiguous

    CHECK_DEVICE(I); CHECK_DEVICE(W1); CHECK_DEVICE(W2); 
    CHECK_DEVICE(W1_zeros_scales); CHECK_DEVICE(W2_zeros_scales); CHECK_DEVICE(O);

    CHECK_CONTIGUOUS(I); CHECK_CONTIGUOUS(W1); CHECK_CONTIGUOUS(W2);
    CHECK_CONTIGUOUS(W1_zeros_scales); CHECK_CONTIGUOUS(W2_zeros_scales); CHECK_CONTIGUOUS(O);

    CHECK_DTYPE(I, at::kHalf);// CHECK_DTYPE(W1, at::kInt); CHECK_DTYPE(W2, at::kInt);
    CHECK_DTYPE(W1_zeros_scales, at::kHalf); CHECK_DTYPE(W2_zeros_scales, at::kHalf); CHECK_DTYPE(O, at::kHalf);

    CHECK_DIMS(I, 3); CHECK_DIMS(W1, 2); CHECK_DIMS(W2, 2);
    CHECK_DIMS(W1_zeros_scales, 2); CHECK_DIMS(W2_zeros_scales, 2); CHECK_DIMS(O, 3);

    if (WorkSpace.has_value()) {
        CHECK_DEVICE(WorkSpace.value());
        CHECK_CONTIGUOUS(WorkSpace.value());
        CHECK_DTYPE(WorkSpace.value(), at::kHalf);
        CHECK_NUMEL(WorkSpace.value(), O.numel());
    } else {
        throw std::invalid_argument("WorkSpace must be a valid tensor.");
    }

    int m = I.size(0) * I.size(1);
    int k = I.size(2);
    int n = O.size(2);

    gemm_awq_silu_dot(reinterpret_cast<half*>(I.data_ptr<at::Half>()),
                      reinterpret_cast<int*>(W1.data_ptr()),
                      reinterpret_cast<int*>(W2.data_ptr()),
                      reinterpret_cast<half*>(W1_zeros_scales.data_ptr<at::Half>()),
                      reinterpret_cast<half*>(W2_zeros_scales.data_ptr<at::Half>()),
                      reinterpret_cast<half*>(WorkSpace.value().data_ptr<at::Half>()),
                      reinterpret_cast<half*>(O.data_ptr<at::Half>()),
                      m, k, n,
                      group_size);
}

template <ROPE_TYPE ROPE, SEQ_DIM_TYPE SEQ_DIM, FREQ_ALIGNED ALIGN>
void gemm_awq_qkvpacked_rope(half* I,
                             int* WQKV,
                             half* WQKV_zeros_scales,
                             half* BQKV,
                             half* QKV_Proj_Output,
                             float* F,
                             int* TokenIndex,
                             half* Q, half* K, half* V,
                             int m, int k, int n, int n_kv,
                             int kv_stride_bs,
                             int kv_stride_seq,
                             int len,
                             int seqlen,
                             int hs,
                             const int group_size) {
    int hn = n / hs;
    int hn_kv = n_kv / hs;

    int bs = m / seqlen;

    // QKV Projection
    gemm_awq_api(I,
             WQKV,
             WQKV_zeros_scales,
             QKV_Proj_Output,
             nullptr,  // use lmdeploy NOW
             m, k, n + n_kv * 2,
             group_size);

    if (BQKV) {
        add_bias<<<dim3(m, 3), dim3(DIV_UP((n + n_kv * 2) >> 3, 3))>>>(
            QKV_Proj_Output,
            BQKV,
            m,
            (n + n_kv * 2) >> 3);
    }

    // QKV rope & update
    // Input: QKV (QKV_Proj_Output), Output: Q & K & V
    if (SEQ_DIM == SEQ_DIM_TYPE::FIRST) {
        update_kv_cache_rope<ROPE, ALIGN><<<dim3(seqlen, bs, (hn + hn_kv * 2)), dim3(hs >> 1)>>>(
            QKV_Proj_Output, (n + n_kv * 2), (n + n_kv * 2) * bs,
            QKV_Proj_Output + n, (n + n_kv * 2), (n + n_kv * 2) * bs,
            QKV_Proj_Output + n + n_kv, (n + n_kv * 2), (n + n_kv * 2) * bs,
            F, TokenIndex,
            Q, n, n * bs,
            K, kv_stride_bs, kv_stride_seq,
            V, kv_stride_bs, kv_stride_seq,
            len, hn, hn_kv, hs);
    } else if (SEQ_DIM == SEQ_DIM_TYPE::SECOND) {
        update_kv_cache_rope<ROPE, ALIGN><<<dim3(seqlen, bs, (hn + hn_kv * 2)), dim3(hs >> 1)>>>(
            QKV_Proj_Output, (n + n_kv * 2) * seqlen, (n + n_kv * 2),
            QKV_Proj_Output + n, (n + n_kv * 2) * seqlen, (n + n_kv * 2),
            QKV_Proj_Output + n + n_kv, (n + n_kv * 2) * seqlen, (n + n_kv * 2),
            F, TokenIndex,
            Q, n * seqlen, n,
            K, kv_stride_bs, kv_stride_seq,
            V, kv_stride_bs, kv_stride_seq,
            len, hn, hn_kv, hs);
    } else {
        throw std::invalid_argument("SEQ_DIM_TYPE error! Only support FIRST or SECOND, but get OTHERS.");
    }
}

// only lmdeploy
template <ROPE_TYPE ROPE, SEQ_DIM_TYPE SEQ_DIM, FREQ_ALIGNED ALIGN>
void Gemm_awq_qkvpacked_rope(at::Tensor I,
                             at::Tensor WQKV,
                             at::Tensor WQKV_zeros_scales,
                             c10::optional<at::Tensor> BQKV,
                             c10::optional<at::Tensor> WorkSpace, 
                             c10::optional<at::Tensor> F,
                             c10::optional<at::Tensor> TokenIndex,
                             at::Tensor Q, at::Tensor K, at::Tensor V,
                             int len,
                             const int group_size) {
    // I: [bs, seq, input_dim], must be contiguous
    // WQKV:                int,    length: (output_dim_q + output_dim_kv * 2) * input_dim / 8
    // WQKV_zeros_scales,  half,    length: (output_dim_q + output_dim_kv * 2) * input_dim / group_size * 2
    // WorkSpace: [bs, seq, output_dim_q + output_dim_kv * 2], must give this GPU buffer

    // BQKV: [output_dim_q + output_dim_kv * 2], must be contiguous
    // Q: [bs, seq, hn, hs], must be contiguous
    // K: [bs, seq, hn_kv, hs], must be contiguous in last dim

    // Check tensor device
    CHECK_DEVICE(I);
    CHECK_DEVICE(WQKV);
    CHECK_DEVICE(WQKV_zeros_scales);
    CHECK_DEVICE(Q);
    CHECK_DEVICE(K);
    CHECK_DEVICE(V);

    if (BQKV.has_value()) CHECK_DEVICE(BQKV.value());
    if (F.has_value()) CHECK_DEVICE(F.value());
    if (TokenIndex.has_value()) CHECK_DEVICE(TokenIndex.value());

    // Check tensor contiguous
    CHECK_CONTIGUOUS(I);
    CHECK_CONTIGUOUS(WQKV);
    CHECK_CONTIGUOUS(WQKV_zeros_scales);
    CHECK_CONTIGUOUS(Q);
    CHECK_LASTDIM_CONTIGUOUS(K);
    CHECK_LASTDIM_CONTIGUOUS(V);

    if (BQKV.has_value()) CHECK_CONTIGUOUS(BQKV.value());
    if (F.has_value()) CHECK_CONTIGUOUS(F.value());
    if (TokenIndex.has_value()) CHECK_CONTIGUOUS(TokenIndex.value());

    // Check tensor dtype
    CHECK_DTYPE(I, at::kHalf);
    CHECK_DTYPE(WQKV_zeros_scales, at::kHalf);
    CHECK_DTYPE(Q, at::kHalf);
    CHECK_DTYPE(K, at::kHalf);
    CHECK_DTYPE(V, at::kHalf);

    if (BQKV.has_value()) CHECK_DTYPE(BQKV.value(), at::kHalf);
    if (F.has_value()) CHECK_DTYPE(F.value(), at::kFloat);
    if (TokenIndex.has_value()) CHECK_DTYPE(TokenIndex.value(), at::kInt);

    // Check tensor dims
    CHECK_DIMS(I, 3);
    CHECK_DIMS(WQKV, 2);
    CHECK_DIMS(WQKV_zeros_scales, 2);
    CHECK_DIMS(Q, 4);
    CHECK_DIMS(K, 4);
    CHECK_DIMS(V, 4);

    if (BQKV.has_value()) CHECK_DIMS(BQKV.value(), 1);
    if (TokenIndex.has_value()) CHECK_DIMS(TokenIndex.value(), 1);

    int bs = SEQ_DIM == SEQ_DIM_TYPE::SECOND ? I.size(0) : I.size(1);
    int seqlen = SEQ_DIM == SEQ_DIM_TYPE::SECOND ? I.size(1) : I.size(0);
    int dim_in = I.size(2);
    int hn = Q.size(-2);
    int hn_kv = K.size(-2);
    int hs = K.size(-1);
    int kv_dim_out = hn_kv * hs;
    int dim_out = hn * hs;  // don't use shape of WQKV

    int kv_stride_bs = SEQ_DIM == SEQ_DIM_TYPE::SECOND ? K.stride(0) : K.stride(1);
    int kv_stride_seq = SEQ_DIM == SEQ_DIM_TYPE::SECOND ? K.stride(1) : K.stride(0);
    
    // Check tensor shapes
    CHECK_SHAPE(Q, I.size(0), I.size(1), hn, hs);
    CHECK_SHAPE(K, K.size(0), K.size(1), hn_kv, hs);
    CHECK_SHAPE(V, K.size(0), K.size(1), hn_kv, hs);

    if (TokenIndex.has_value()) CHECK_SHAPE(TokenIndex.value(), bs * seqlen);

    if (WorkSpace.has_value()) {
        CHECK_DEVICE(WorkSpace.value());
        CHECK_CONTIGUOUS(WorkSpace.value());
        CHECK_DTYPE(WorkSpace.value(), at::kHalf);
        CHECK_NUMEL(WorkSpace.value(), bs * seqlen * (dim_out + kv_dim_out * 2));
    } else {
        throw std::invalid_argument("WorkSpace must be a valid tensor.");
    }

    half* bqkv = BQKV.has_value() ? reinterpret_cast<half*>(BQKV.value().data_ptr<at::Half>()) : nullptr;
    float* f = F.has_value() ? reinterpret_cast<float*>(F.value().data_ptr<float>()) : nullptr;
    int* tokenindex = TokenIndex.has_value() ? reinterpret_cast<int*>(TokenIndex.value().data_ptr<int>()) : nullptr;

    gemm_awq_qkvpacked_rope<ROPE, SEQ_DIM, ALIGN>(
        reinterpret_cast<half*>(I.data_ptr<at::Half>()),
        reinterpret_cast<int*>(WQKV.data_ptr()),
        reinterpret_cast<half*>(WQKV_zeros_scales.data_ptr<at::Half>()),
        bqkv,
        reinterpret_cast<half*>(WorkSpace.value().data_ptr<at::Half>()),
        f,
        tokenindex,
        reinterpret_cast<half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<half*>(V.data_ptr<at::Half>()),
        bs * seqlen, dim_in, dim_out, kv_dim_out, 
        kv_stride_bs, kv_stride_seq,
        len, seqlen, hs,
        group_size);
}

// Instantiating template functions explicitly <Gemm_awq_qkvpacked_rope>
#define GEMM_AWQ_QKVPACKER_ROPE(ROPE, SEQ_DIM, ALIGN)            \
    template void Gemm_awq_qkvpacked_rope<ROPE, SEQ_DIM, ALIGN>( \
        at::Tensor I,                                            \
        at::Tensor WQKV,                                         \
        at::Tensor WQKV_zeros_scales,                            \
        c10::optional<at::Tensor> BQKV,                          \
        c10::optional<at::Tensor> WorkSpace,                     \
        c10::optional<at::Tensor> F,                             \
        c10::optional<at::Tensor> TokenIndex,                    \
        at::Tensor Q, at::Tensor K, at::Tensor V,                \
        int len, const int group_size)
GEMM_AWQ_QKVPACKER_ROPE(ROPE_TYPE::HALF_ROPE, SEQ_DIM_TYPE::FIRST, FREQ_ALIGNED::YES);
GEMM_AWQ_QKVPACKER_ROPE(ROPE_TYPE::FULL_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::YES);
GEMM_AWQ_QKVPACKER_ROPE(ROPE_TYPE::HALF_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::YES);
GEMM_AWQ_QKVPACKER_ROPE(ROPE_TYPE::NO_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::YES);
GEMM_AWQ_QKVPACKER_ROPE(ROPE_TYPE::FULL_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::NO);
GEMM_AWQ_QKVPACKER_ROPE(ROPE_TYPE::HALF_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::NO);
GEMM_AWQ_QKVPACKER_ROPE(ROPE_TYPE::NO_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::NO);
#undef GEMM_AWQ_QKVPACKER_ROPE


template <ROPE_TYPE ROPE, SEQ_DIM_TYPE SEQ_DIM, FREQ_ALIGNED ALIGN>
void gemm_awq_qkvunpacked_rope(half* I, 
                               int* WQ, int* WK, int* WV, 
                               half* WQ_zeros_scales,
                               half* WK_zeros_scales,
                               half* WV_zeros_scales,
                               half* BQ, half* BK, half* BV, 
                               half* WorkSpace, 
                               float* F, 
                               int* TokenIndex, 
                               half* Q, half* K, half* V, 
                               int m, int k, int n, int n_kv, 
                               int kv_stride_bs, 
                               int kv_stride_seq, 
                               int len, 
                               int seqlen, 
                               int hs,
                               const int group_size) {
    int hn = n / hs;
    int hn_kv = n_kv / hs;
    
    half* K_tmp;
    half* V_tmp;
    if (ALIGN == FREQ_ALIGNED::YES) {
        K_tmp = WorkSpace;
        V_tmp = WorkSpace;
    } else {
        K_tmp = K;
        V_tmp = V;
    }

    int bs = m / seqlen;

    // V Projection
    // Input: I, Output: V_tmp --> V
    gemm_awq_api(I, WV, WV_zeros_scales, V_tmp, nullptr, m, k, n_kv, group_size);
    if (BV) {
        add_bias<<<dim3(m, 1), dim3(n_kv >> 3)>>>(V_tmp, BV, m, n_kv >> 3);
    }
    
    if (ALIGN == FREQ_ALIGNED::YES) {
        if (SEQ_DIM == SEQ_DIM_TYPE::FIRST)
            update_cache<<<dim3(seqlen, 1, bs), dim3(n_kv >> 3)>>>(
                        V_tmp, V + (size_t)len * kv_stride_seq, 
                        n_kv >> 3, (bs * n_kv) >> 3, 
                        kv_stride_bs >> 3, kv_stride_seq >> 3);
        else if (SEQ_DIM == SEQ_DIM_TYPE::SECOND)
            update_cache<<<dim3(seqlen, 1, bs), dim3(n_kv >> 3)>>>(
                        V_tmp, V + (size_t)len * kv_stride_seq, 
                        (seqlen * n_kv) >> 3, n_kv >> 3, 
                        kv_stride_bs >> 3, kv_stride_seq >> 3);
        else
            throw std::invalid_argument("SEQ_DIM_TYPE error! Only support FIRST or SECOND, but get OTHERS.");
    } else {
        // Nothing need to do, because V_tmp == V
    }

    // Q & K Projection + RoPE
    // Input: I, Output: Q & K_tmp --> Q & K
    gemm_awq_api(I, WQ, WQ_zeros_scales, Q, nullptr, m, k, n, group_size);
    gemm_awq_api(I, WK, WK_zeros_scales, K_tmp, nullptr, m, k, n_kv, group_size);
    if (BQ) {
        add_bias<<<dim3(m, 1), dim3(n_kv >> 3)>>>(K_tmp, BK, m, n_kv >> 3);
        add_bias<<<dim3(m, 1), dim3(n >> 3)>>>(Q, BQ, m, n >> 3);
    }
    
    if (SEQ_DIM == SEQ_DIM_TYPE::FIRST)
        update_kv_cache_rope<ROPE, ALIGN><<<dim3(seqlen, bs, (hn + hn_kv)), dim3(hs >> 1)>>>(
                Q, n, n * bs,
                K_tmp, n_kv, n_kv * bs,
                V_tmp, n_kv, n_kv * bs,
                F, TokenIndex,
                Q, n, n * bs,
                K, kv_stride_bs, kv_stride_seq, 
                V, kv_stride_bs, kv_stride_seq, 
                len, hn, hn_kv, hs);
    else if (SEQ_DIM == SEQ_DIM_TYPE::SECOND)
        update_kv_cache_rope<ROPE, ALIGN><<<dim3(seqlen, bs, (hn + hn_kv)), dim3(hs >> 1)>>>(
                Q, n * seqlen, n,
                K_tmp, n_kv * seqlen, n_kv,
                V_tmp, n_kv * seqlen, n_kv, // not used
                F, TokenIndex,
                Q, n * seqlen, n,
                K, kv_stride_bs, kv_stride_seq, 
                V, kv_stride_bs, kv_stride_seq, // not used
                len, hn, hn_kv, hs);
    else
        throw std::invalid_argument("SEQ_DIM_TYPE error! Only support FIRST or SECOND, but get OTHERS.");
}

// only lmdeploy
template <ROPE_TYPE ROPE, SEQ_DIM_TYPE SEQ_DIM, FREQ_ALIGNED ALIGN>
void Gemm_awq_qkvunpacked_rope(at::Tensor I,
                                at::Tensor WQ, 
                                at::Tensor WK, 
                                at::Tensor WV, 
                                at::Tensor WQ_zeros_scales,
                                at::Tensor WK_zeros_scales,
                                at::Tensor WV_zeros_scales,
                                c10::optional<at::Tensor> BQ, 
                                c10::optional<at::Tensor> BK, 
                                c10::optional<at::Tensor> BV, 
                                c10::optional<at::Tensor> WorkSpace, 
                                c10::optional<at::Tensor> F,
                                c10::optional<at::Tensor> TokenIndex,
                                at::Tensor Q, 
                                at::Tensor K, 
                                at::Tensor V,
                                int len,
                                const int group_size) {

    // I: [bs, seq, input_dim], must be contiguous
    // WQ:                int, length: output_dim_q * input_dim / 8
    // WK:                int, length: output_dim_kv * input_dim / 8
    // WV:                int, length: output_dim_kv * input_dim / 8
    // WQ_zeros_scales:   half, length: output_dim_q * input_dim / group_size * 2
    // WK_zeros_scales:   half, length: output_dim_kv * input_dim / group_size * 2
    // WV_zeros_scales:   half, length: output_dim_kv * input_dim / group_size * 2
    // BQ: [output_dim_q], must be contiguous
    // BK: [output_dim_kv], must be contiguous
    // BV: [output_dim_kv], must be contiguous
    // WorkSpace: [bs, seq, output_dim_kv], must be contiguous
    
    // Check tensor device
    CHECK_DEVICE(I); CHECK_DEVICE(WQ); CHECK_DEVICE(WK); CHECK_DEVICE(WV); 
    CHECK_DEVICE(WQ_zeros_scales); CHECK_DEVICE(WK_zeros_scales); CHECK_DEVICE(WV_zeros_scales);
    CHECK_DEVICE(Q); CHECK_DEVICE(K); CHECK_DEVICE(V);
    if (BQ.has_value()) CHECK_DEVICE(BQ.value());
    if (BK.has_value()) CHECK_DEVICE(BK.value());
    if (BV.has_value()) CHECK_DEVICE(BV.value());
    if (WorkSpace.has_value()) CHECK_DEVICE(WorkSpace.value());
    if (F.has_value()) CHECK_DEVICE(F.value());
    if (TokenIndex.has_value()) CHECK_DEVICE(TokenIndex.value());
    // Check tensor contiguous
    CHECK_CONTIGUOUS(I); CHECK_CONTIGUOUS(WQ); CHECK_CONTIGUOUS(WK); CHECK_CONTIGUOUS(WV); 
    CHECK_CONTIGUOUS(WQ_zeros_scales); CHECK_CONTIGUOUS(WK_zeros_scales); CHECK_CONTIGUOUS(WV_zeros_scales);
    CHECK_CONTIGUOUS(Q); CHECK_LASTDIM_CONTIGUOUS(K); CHECK_LASTDIM_CONTIGUOUS(V);
    if (BQ.has_value()) CHECK_CONTIGUOUS(BQ.value());
    if (BK.has_value()) CHECK_CONTIGUOUS(BK.value());
    if (BV.has_value()) CHECK_CONTIGUOUS(BV.value());
    if (WorkSpace.has_value()) CHECK_CONTIGUOUS(WorkSpace.value());
    if (F.has_value()) CHECK_CONTIGUOUS(F.value());
    if (TokenIndex.has_value()) CHECK_CONTIGUOUS(TokenIndex.value());
    // Check tensor dtype
    CHECK_DTYPE(I, at::kHalf); 
    CHECK_DTYPE(WQ_zeros_scales, at::kHalf); CHECK_DTYPE(WK_zeros_scales, at::kHalf); CHECK_DTYPE(WV_zeros_scales, at::kHalf); 
    CHECK_DTYPE(Q, at::kHalf); CHECK_DTYPE(K, at::kHalf); CHECK_DTYPE(V, at::kHalf);
    if (BQ.has_value()) CHECK_DTYPE(BQ.value(), at::kHalf);
    if (BK.has_value()) CHECK_DTYPE(BK.value(), at::kHalf);
    if (BV.has_value()) CHECK_DTYPE(BV.value(), at::kHalf);
    if (WorkSpace.has_value()) CHECK_DTYPE(WorkSpace.value(), at::kHalf);
    if (F.has_value()) CHECK_DTYPE(F.value(), at::kFloat);
    if (TokenIndex.has_value()) CHECK_DTYPE(TokenIndex.value(), at::kInt);
    // Check tensor dims
    CHECK_DIMS(I, 3); CHECK_DIMS(WQ, 2); CHECK_DIMS(WK, 2); CHECK_DIMS(WV, 2); 
    CHECK_DIMS(WQ_zeros_scales, 2); CHECK_DIMS(WK_zeros_scales, 2); CHECK_DIMS(WV_zeros_scales, 2);
    CHECK_DIMS(Q, 4); CHECK_DIMS(K, 4); CHECK_DIMS(V, 4);
    if (BQ.has_value()) CHECK_DIMS(BQ.value(), 1);
    if (BK.has_value()) CHECK_DIMS(BK.value(), 1);
    if (BV.has_value()) CHECK_DIMS(BV.value(), 1);
    if (TokenIndex.has_value()) CHECK_DIMS(TokenIndex.value(), 1);

    int bs = SEQ_DIM == SEQ_DIM_TYPE::SECOND ? I.size(0) : I.size(1);
    int seqlen = SEQ_DIM == SEQ_DIM_TYPE::SECOND ? I.size(1) : I.size(0);
    int dim_in = I.size(2);
    int hn = Q.size(-2);
    int hn_kv = K.size(-2);
    int hs = K.size(-1);
    int kv_dim_out = hn_kv * hs;
    int dim_out = hn * hs;  // don't use shape of WQ

    int kv_stride_bs = SEQ_DIM == SEQ_DIM_TYPE::SECOND ? K.stride(0) : K.stride(1);
    int kv_stride_seq = SEQ_DIM == SEQ_DIM_TYPE::SECOND ? K.stride(1) : K.stride(0);

    // Check tensor shapes
    CHECK_SHAPE(Q, I.size(0), I.size(1), hn, hs);
    CHECK_SHAPE(K, K.size(0), K.size(1), hn_kv, hs);
    CHECK_SHAPE(V, K.size(0), K.size(1), hn_kv, hs);
    if (ALIGN == FREQ_ALIGNED::YES) {
        if (!WorkSpace.has_value()) throw std::invalid_argument("WorkSpace is required.");
        else CHECK_NUMEL(WorkSpace.value(), bs * seqlen * kv_dim_out);
    }
    if (TokenIndex.has_value()) CHECK_SHAPE(TokenIndex.value(), bs * seqlen);

    half* bq = BQ.has_value() ? reinterpret_cast<half *>(BQ.value().data_ptr<at::Half>()) : nullptr;
    half* bk = BK.has_value() ? reinterpret_cast<half *>(BK.value().data_ptr<at::Half>()) : nullptr;
    half* bv = BV.has_value() ? reinterpret_cast<half *>(BV.value().data_ptr<at::Half>()) : nullptr;
    half* workspace = WorkSpace.has_value() ? reinterpret_cast<half *>(WorkSpace.value().data_ptr<at::Half>()) : nullptr;
    float* f = F.has_value() ? reinterpret_cast<float *>(F.value().data_ptr<float>()) : nullptr;
    int* tokenindex = TokenIndex.has_value() ? reinterpret_cast<int *>(TokenIndex.value().data_ptr<int>()) : nullptr;

    gemm_awq_qkvunpacked_rope<ROPE, SEQ_DIM, ALIGN>(
        reinterpret_cast<half*>(I.data_ptr<at::Half>()),
        reinterpret_cast<int*>(WQ.data_ptr()), reinterpret_cast<int*>(WK.data_ptr()), reinterpret_cast<int*>(WV.data_ptr()),
        reinterpret_cast<half*>(WQ_zeros_scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(WK_zeros_scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(WV_zeros_scales.data_ptr<at::Half>()),
        bq, bk, bv,
        workspace,
        f, tokenindex,
        reinterpret_cast<half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<half*>(V.data_ptr<at::Half>()),
        bs * seqlen, dim_in, dim_out, kv_dim_out,
        kv_stride_bs, kv_stride_seq,
        len, seqlen, hs,
        group_size);
}

// Instantiating template functions explicitly <Gemm_awq_qkvunpacked_rope>
#define GEMM_AWQ_QKVUNPACKED_ROPE(ROPE, SEQ_DIM, ALIGN)            \
    template void Gemm_awq_qkvunpacked_rope<ROPE, SEQ_DIM, ALIGN>( \
        at::Tensor I,                                              \
        at::Tensor WQ,                                             \
        at::Tensor WK,                                             \
        at::Tensor WV,                                             \
        at::Tensor WQ_zeros_scales,                                \
        at::Tensor WK_zeros_scales,                                \
        at::Tensor WV_zeros_scales,                                \
        c10::optional<at::Tensor> BQ,                              \
        c10::optional<at::Tensor> BK,                              \
        c10::optional<at::Tensor> BV,                              \
        c10::optional<at::Tensor> WorkSpace,                       \
        c10::optional<at::Tensor> F,                               \
        c10::optional<at::Tensor> TokenIndex,                      \
        at::Tensor Q,                                              \
        at::Tensor K,                                              \
        at::Tensor V,                                              \
        int len,                                                   \
        const int group_size)

GEMM_AWQ_QKVUNPACKED_ROPE(ROPE_TYPE::HALF_ROPE, SEQ_DIM_TYPE::FIRST, FREQ_ALIGNED::YES);
GEMM_AWQ_QKVUNPACKED_ROPE(ROPE_TYPE::FULL_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::YES);
GEMM_AWQ_QKVUNPACKED_ROPE(ROPE_TYPE::HALF_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::YES);
GEMM_AWQ_QKVUNPACKED_ROPE(ROPE_TYPE::NO_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::YES);
GEMM_AWQ_QKVUNPACKED_ROPE(ROPE_TYPE::FULL_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::NO);
GEMM_AWQ_QKVUNPACKED_ROPE(ROPE_TYPE::HALF_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::NO);
GEMM_AWQ_QKVUNPACKED_ROPE(ROPE_TYPE::NO_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::NO);

#undef GEMM_AWQ_QKVUNPACKED_ROPE