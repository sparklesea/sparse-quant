#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "../../../kernels/amd/w4a16_gemm.h"
#include "../../../kernels/amd/residual.h"
#include "../../../kernels/amd/activation.h"
#include "../../../kernels/amd/rope.h"

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
        w4a16_bs1_kernel<<<dim3(1, dim_out / 2), dim3(128, 1)>>>(
            reinterpret_cast<uint32_t *>(W.data_ptr()),
            reinterpret_cast<half2 *>(zeros_scales.data_ptr<at::Half>()),
            reinterpret_cast<half *>(X.data_ptr<at::Half>()),
            reinterpret_cast<half *>(O.data_ptr<at::Half>()), dim_in,
            DIV_UP(dim_in, 128), group_size);
    } else if (bs == 2) {
        // batch size 2, seq 1. OR
        // batch size 1, seq 2.
        w4a16_bs2_kernel<<<dim3(1, dim_out / 2), dim3(128, 1)>>>(
            reinterpret_cast<uint32_t *>(W.data_ptr()),
            reinterpret_cast<half2 *>(zeros_scales.data_ptr<at::Half>()),
            reinterpret_cast<half *>(X.data_ptr<at::Half>()),
            reinterpret_cast<half *>(O.data_ptr<at::Half>()), dim_in, dim_out,
            DIV_UP(dim_in, 128), group_size);
    } else {
        // speedup for batch size 1 to 16
        w4a16_gemm_wmma_kernel<16, 16, 256, 264, 24>
            <<<dim3(DIV_UP(dim_out, 16), DIV_UP(bs, 16)), dim3(256, 1)>>>(
                reinterpret_cast<uint32_t *>(W.data_ptr()),
                reinterpret_cast<half2 *>(zeros_scales.data_ptr<at::Half>()),
                reinterpret_cast<half *>(X.data_ptr<at::Half>()),
                reinterpret_cast<half *>(O.data_ptr<at::Half>()),
                bs, dim_out, dim_in,
                group_size);
    }
}


void Gemm_awq_pure(at::Tensor I,
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

}

// gemm_awq_api + residual
// O: residual & output
void gemm_awq_residual(at::Tensor I, at::Tensor W, at::Tensor W_zeros_scales, at::Tensor linear_output, at::Tensor O,
                       int m, int k, int n, const int group_size) {
    
    w4a16_gemm(I, W, W_zeros_scales, linear_output, group_size);

    residual_kernel<<<dim3(m, DIV_UP(n, 256)), dim3(128, 1)>>>(
    reinterpret_cast<half*>(O.data_ptr<at::Half>()), 
    reinterpret_cast<half*>(linear_output.data_ptr<at::Half>()),
    m, n, 
    reinterpret_cast<half*>(O.data_ptr<at::Half>()));
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

    CHECK_DEVICE(I);CHECK_DEVICE(W);
    CHECK_DEVICE(W_zeros_scales);CHECK_DEVICE(O);

    CHECK_CONTIGUOUS(I);CHECK_CONTIGUOUS(W);
    CHECK_CONTIGUOUS(W_zeros_scales);CHECK_CONTIGUOUS(O);

    CHECK_DTYPE(I, at::kHalf);//CHECK_DTYPE(W, at::kInt);
    CHECK_DTYPE(W_zeros_scales, at::kHalf);CHECK_DTYPE(O, at::kHalf);

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

    gemm_awq_residual(I, W, W_zeros_scales, WorkSpace.value(),
                      O, m, k, n, group_size);
}




void gemm_awq_silu_dot(at::Tensor I, at::Tensor W1, at::Tensor W2, at::Tensor W1_zeros_scales, at::Tensor W2_zeros_scales, 
                       at::Tensor WorkSpace, at::Tensor O,
                       int m, int k, int n, const int group_size) {

    w4a16_gemm(I, W1, W1_zeros_scales, WorkSpace, group_size);

    //todo use O? 
    w4a16_gemm(I, W2, W2_zeros_scales, O, group_size);

    // silu(W1O) * W2O ---> O
    silu_dot_kernel_quant<<<256, 256>>>(reinterpret_cast<half*>(WorkSpace.data_ptr<at::Half>()),
                                    reinterpret_cast<half*>(O.data_ptr<at::Half>()),
                                    reinterpret_cast<half*>(O.data_ptr<at::Half>()),
                                    m, n, n);
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

    CHECK_DTYPE(I, at::kHalf); //CHECK_DTYPE(W1, at::kInt); CHECK_DTYPE(W2, at::kInt);
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

    gemm_awq_silu_dot(I, W1, W2,
                      W1_zeros_scales,
                      W2_zeros_scales,
                      WorkSpace.value(), O,
                      m, k, n,
                      group_size);
}

void gemm_awq_packed_silu_dot(at::Tensor I, at::Tensor W13, at::Tensor W13_zeros_scales, at::Tensor W13_output, at::Tensor O,
                       int m, int k, int n, const int group_size) {
    //todo use w13_output? 
    w4a16_gemm(I, W13, W13_zeros_scales, W13_output, group_size);

    // silu(W1O) * W3O ---> O
    silu_dot_kernel_quant<<<256, 256>>>(reinterpret_cast<half*>(W13_output.data_ptr<at::Half>()),
                                    reinterpret_cast<half*>(W13_output.data_ptr<at::Half>()) + n,
                                    reinterpret_cast<half*>(O.data_ptr<at::Half>()),
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

    CHECK_DEVICE(I);CHECK_DEVICE(W);
    CHECK_DEVICE(W_zeros_scales);CHECK_DEVICE(O);

    CHECK_CONTIGUOUS(I);CHECK_CONTIGUOUS(W);
    CHECK_CONTIGUOUS(W_zeros_scales);CHECK_CONTIGUOUS(O);

    CHECK_DTYPE(I, at::kHalf); //CHECK_DTYPE(W, at::kInt);
    CHECK_DTYPE(W_zeros_scales, at::kHalf);CHECK_DTYPE(O, at::kHalf);

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

    gemm_awq_packed_silu_dot(I, W, W_zeros_scales,
                            WorkSpace.value(), O,
                            m, k, n,
                            group_size);
}



template <ROPE_TYPE ROPE, SEQ_DIM_TYPE SEQ_DIM, FREQ_ALIGNED ALIGN>
void gemm_awq_qkvpacked_rope(at::Tensor I,
                             at::Tensor WQKV,
                             at::Tensor WQKV_zeros_scales,
                             c10::optional<at::Tensor> BQKV,
                             c10::optional<at::Tensor> QKV_Proj_Output,
                             c10::optional<at::Tensor> F,
                             c10::optional<at::Tensor> TokenIndex,
                             at::Tensor Q, at::Tensor K, at::Tensor V,
                             int m, int k, int n, int n_kv,
                             int kv_stride_bs,
                             int kv_stride_seq,
                             int len,
                             int seqlen,
                             int hs,
                             const int group_size) {

    float* f = F.has_value() ? reinterpret_cast<float*>(F.value().data_ptr<float>()) : nullptr;
    int* TokenIndex_ptr = TokenIndex.has_value() ? reinterpret_cast<int*>(TokenIndex.value().data_ptr<int>()) : nullptr;

    half *I_ptr = reinterpret_cast<half *>(I.data_ptr<at::Half>());
    half *BQKV_ptr = BQKV.has_value() ? reinterpret_cast<half *>(BQKV.value().data_ptr<at::Half>()) : nullptr;
    half *QKV_Proj_Output_ptr =
        QKV_Proj_Output.has_value() ? reinterpret_cast<half *>(QKV_Proj_Output.value().data_ptr<at::Half>()) : nullptr;
    half *Q_ptr = reinterpret_cast<half *>(Q.data_ptr<at::Half>());
    half *K_ptr = reinterpret_cast<half *>(K.data_ptr<at::Half>());
    half *V_ptr = reinterpret_cast<half *>(V.data_ptr<at::Half>());

    int hn = n / hs;
    int hn_kv = n_kv / hs;

    int bs = m / seqlen;

    // QKV Projection
    w4a16_gemm(I, WQKV, WQKV_zeros_scales,
                QKV_Proj_Output.value(), group_size);

    if (BQKV_ptr) {
        add_bias<<<dim3(m, 3), dim3(DIV_UP((n + n_kv * 2) >> 3, 3))>>>(
            QKV_Proj_Output_ptr,
            BQKV_ptr,
            m,
            (n + n_kv * 2) >> 3);
    }

    if (SEQ_DIM == SEQ_DIM_TYPE::FIRST) {
        update_kv_cache_rope<ROPE, ALIGN><<<dim3(seqlen, bs, (hn + hn_kv * 2)), dim3(hs >> 1)>>>(
            QKV_Proj_Output_ptr, (n + n_kv * 2), (n + n_kv * 2) * bs,
            QKV_Proj_Output_ptr + n, (n + n_kv * 2), (n + n_kv * 2) * bs,
            QKV_Proj_Output_ptr + n + n_kv, (n + n_kv * 2), (n + n_kv * 2) * bs,
            f, TokenIndex_ptr,
            Q_ptr, n, n * bs,
            K_ptr, kv_stride_bs, kv_stride_seq,
            V_ptr, kv_stride_bs, kv_stride_seq,
            len, hn, hn_kv, hs);
    } else if (SEQ_DIM == SEQ_DIM_TYPE::SECOND) {
        update_kv_cache_rope<ROPE, ALIGN><<<dim3(seqlen, bs, (hn + hn_kv * 2)), dim3(hs >> 1)>>>(
            QKV_Proj_Output_ptr, (n + n_kv * 2) * seqlen, (n + n_kv * 2),
            QKV_Proj_Output_ptr + n, (n + n_kv * 2) * seqlen, (n + n_kv * 2),
            QKV_Proj_Output_ptr + n + n_kv, (n + n_kv * 2) * seqlen, (n + n_kv * 2),
            f, TokenIndex_ptr,
            Q_ptr, n * seqlen, n,
            K_ptr, kv_stride_bs, kv_stride_seq,
            V_ptr, kv_stride_bs, kv_stride_seq,
            len, hn, hn_kv, hs);
    }else {
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
    gemm_awq_qkvpacked_rope<ROPE, SEQ_DIM, ALIGN>(
        I, WQKV, WQKV_zeros_scales,
        BQKV, WorkSpace, F, TokenIndex,
        Q, K, V,
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
void gemm_awq_qkvunpacked_rope(at::Tensor I, 
                               at::Tensor WQ, at::Tensor WK, at::Tensor WV, 
                               at::Tensor WQ_zeros_scales,
                               at::Tensor WK_zeros_scales,
                               at::Tensor WV_zeros_scales,
                               c10::optional<at::Tensor> BQ, 
                               c10::optional<at::Tensor> BK, 
                               c10::optional<at::Tensor> BV, 
                               c10::optional<at::Tensor> WorkSpace, 
                               c10::optional<at::Tensor> F, 
                               c10::optional<at::Tensor> TokenIndex, 
                               at::Tensor Q, at::Tensor K, at::Tensor V, 
                               int m, int k, int n, int n_kv, 
                               int kv_stride_bs, 
                               int kv_stride_seq, 
                               int len, 
                               int seqlen, 
                               int hs,
                               const int group_size) {

    half* BQ_ptr = BQ.has_value() ? reinterpret_cast<half *>(BQ.value().data_ptr<at::Half>()) : nullptr;
    half* BK_ptr = BK.has_value() ? reinterpret_cast<half *>(BK.value().data_ptr<at::Half>()) : nullptr;
    half* BV_ptr = BV.has_value() ? reinterpret_cast<half *>(BV.value().data_ptr<at::Half>()) : nullptr;
    half* WorkSpace_ptr = WorkSpace.has_value() ? reinterpret_cast<half *>(WorkSpace.value().data_ptr<at::Half>()) : nullptr;
    float* f = F.has_value() ? reinterpret_cast<float *>(F.value().data_ptr<float>()) : nullptr;
    int* TokenIndex_ptr = TokenIndex.has_value() ? reinterpret_cast<int *>(TokenIndex.value().data_ptr<int>()) : nullptr;

    half *I_ptr = reinterpret_cast<half *>(I.data_ptr<at::Half>());
    
    half *WQ_zeros_scales_ptr = reinterpret_cast<half *>(WQ_zeros_scales.data_ptr<at::Half>());
    half *WK_zeros_scales_ptr = reinterpret_cast<half *>(WK_zeros_scales.data_ptr<at::Half>());
    half *WV_zeros_scales_ptr = reinterpret_cast<half *>(WV_zeros_scales.data_ptr<at::Half>());
    
    half *Q_ptr = reinterpret_cast<half *>(Q.data_ptr<at::Half>());
    half *K_ptr = reinterpret_cast<half *>(K.data_ptr<at::Half>());
    half *V_ptr = reinterpret_cast<half *>(V.data_ptr<at::Half>());



    int hn = n / hs;
    int hn_kv = n_kv / hs;
    
    half* K_tmp;
    half* V_tmp;
    at::Tensor K_at_tmp;
    at::Tensor V_at_tmp;
    if (ALIGN == FREQ_ALIGNED::YES) {
        K_tmp = WorkSpace_ptr;
        V_tmp = WorkSpace_ptr;
        K_at_tmp = WorkSpace.value();
        V_at_tmp = WorkSpace.value();
    } else {
        K_tmp = K_ptr;
        V_tmp = V_ptr;
        K_at_tmp = K;
        V_at_tmp = V;
    }

    int bs = m / seqlen;

    // V Projection
    // Input: I, Output: V_tmp --> V
    w4a16_gemm(I, WV, WV_zeros_scales, V_at_tmp, group_size);
    if (BV_ptr) {
        add_bias<<<dim3(m, 1), dim3(n_kv >> 3)>>>(V_tmp, BV_ptr, m, n_kv >> 3);
    }

    if (ALIGN == FREQ_ALIGNED::YES) {
        if (SEQ_DIM == SEQ_DIM_TYPE::FIRST)
            update_cache<<<dim3(seqlen, 1, bs), dim3(n_kv >> 3)>>>(
                        V_tmp, V_ptr + (size_t)len * kv_stride_seq, 
                        n_kv >> 3, (bs * n_kv) >> 3, 
                        kv_stride_bs >> 3, kv_stride_seq >> 3);
        else if (SEQ_DIM == SEQ_DIM_TYPE::SECOND)
            update_cache<<<dim3(seqlen, 1, bs), dim3(n_kv >> 3)>>>(
                        V_tmp, V_ptr + (size_t)len * kv_stride_seq, 
                        (seqlen * n_kv) >> 3, n_kv >> 3, 
                        kv_stride_bs >> 3, kv_stride_seq >> 3);
        else
            throw std::invalid_argument("SEQ_DIM_TYPE error! Only support FIRST or SECOND, but get OTHERS.");
    } 
    else {
        // Nothing need to do, because V_tmp == V
    }

    // Q & K Projection + RoPE
    // Input: I, Output: Q & K_tmp --> Q & K
    w4a16_gemm(I, WQ, WQ_zeros_scales, Q, group_size);
    w4a16_gemm(I, WK, WK_zeros_scales, K_at_tmp, group_size);
    if (BQ_ptr) {
        add_bias<<<dim3(m, 1), dim3(n_kv >> 3)>>>(K_tmp, BK_ptr, m, n_kv >> 3);
        add_bias<<<dim3(m, 1), dim3(n >> 3)>>>(Q_ptr, BQ_ptr, m, n >> 3);
    }
    
    if (SEQ_DIM == SEQ_DIM_TYPE::FIRST)
        update_kv_cache_rope<ROPE, ALIGN><<<dim3(seqlen, bs, (hn + hn_kv)), dim3(hs >> 1)>>>(
                Q_ptr, n, n * bs,
                K_tmp, n_kv, n_kv * bs,
                V_tmp, n_kv, n_kv * bs,
                f, TokenIndex_ptr,
                Q_ptr, n, n * bs,
                K_ptr, kv_stride_bs, kv_stride_seq, 
                V_ptr, kv_stride_bs, kv_stride_seq, 
                len, hn, hn_kv, hs);
    else if (SEQ_DIM == SEQ_DIM_TYPE::SECOND)
        update_kv_cache_rope<ROPE, ALIGN><<<dim3(seqlen, bs, (hn + hn_kv)), dim3(hs >> 1)>>>(
                Q_ptr, n * seqlen, n,
                K_tmp, n_kv * seqlen, n_kv,
                V_tmp, n_kv * seqlen, n_kv, // not used
                f, TokenIndex_ptr,
                Q_ptr, n * seqlen, n,
                K_ptr, kv_stride_bs, kv_stride_seq, 
                V_ptr, kv_stride_bs, kv_stride_seq, // not used
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

    gemm_awq_qkvunpacked_rope<ROPE, SEQ_DIM, ALIGN>(
        I, WQ, WK, WV,
        WQ_zeros_scales, WK_zeros_scales, WV_zeros_scales,
        BQ, BK, BV, WorkSpace, F, TokenIndex,
        Q, K, V,
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