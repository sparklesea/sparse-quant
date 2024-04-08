#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "../op.h"

///////////////////////////////  For unit test  ////////////////////////////////////////
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



at::Tensor gemm_awq_residual_ut(at::Tensor I, at::Tensor W, at::Tensor W_zeros_scales,
                                at::Tensor R,
                                const int M, const int N, const int K,
                                const int group_size) {
    // I:               [bs, seqlen, K]
    // W:               [K, N // 8], int
    // W_zeros_scales:  [K // group_size, N * 2]
    // R:               [bs, seqlen, N]
    // O:               [bs, seqlen, N]

    at::Tensor WorkSpace = torch::empty({I.size(0), I.size(1), N},
                                       at::device(I.device()).dtype(I.dtype()));

    Gemm_awq_residual(I,
                      W,
                      W_zeros_scales,
                      WorkSpace,
                      R,
                      group_size);

    return R;
}

at::Tensor gemm_awq_silu_dot_ut(at::Tensor I, 
                                c10::optional<at::Tensor> W1, 
                                c10::optional<at::Tensor> W3, 
                                c10::optional<at::Tensor> W1_zeros_scales,
                                c10::optional<at::Tensor> W3_zeros_scales,
                                c10::optional<at::Tensor> W13, 
                                c10::optional<at::Tensor> W13_zeros_scales,
                                const int M, const int N, const int K,
                                const int group_size) {
    // I:                   [bs, seqlen, K]
    // W1/W3:               [K, N // 8], int
    // W1/W3_zeros_scales:  [K // group_size, N * 2]
    // W13:                 [K, N // 8 * 2], int
    // W13_zeros_scales:    [K // group_size, N * 2 * 2]
    // O:                   [bs, seqlen, N]

    bool w13_packed = W13.has_value();
    at::Tensor O = torch::empty({I.size(0), I.size(1), N},
                                at::device(I.device()).dtype(I.dtype()));
    at::Tensor WorkSpace;
    if (w13_packed) {
        WorkSpace = torch::empty({I.size(0), I.size(1), N * 2},
                                at::device(I.device()).dtype(I.dtype()));
    } else {
        WorkSpace = torch::empty({I.size(0), I.size(1), N}, 
                                at::device(I.device()).dtype(I.dtype()));
    }

    if (w13_packed) {
        Gemm_awq_packed_silu_dot(I, W13.value(), W13_zeros_scales.value(), WorkSpace, O, group_size);
    } else {
        Gemm_awq_silu_dot(I, W1.value(), W3.value(), W1_zeros_scales.value(), W3_zeros_scales.value(), WorkSpace, O, group_size);
    }

    return O;
}

at::Tensor gemm_awq_rope_ut(at::Tensor I,
                            c10::optional<at::Tensor> WQ, 
                            c10::optional<at::Tensor> WK, 
                            c10::optional<at::Tensor> WV,
                            c10::optional<at::Tensor> WQ_zeros_scales,
                            c10::optional<at::Tensor> WK_zeros_scales,
                            c10::optional<at::Tensor> WV_zeros_scales,
                            c10::optional<at::Tensor> BQ, 
                            c10::optional<at::Tensor> BK, 
                            c10::optional<at::Tensor> BV,
                            c10::optional<at::Tensor> WQKV,
                            c10::optional<at::Tensor> WQKV_zeros_scales,
                            c10::optional<at::Tensor> BQKV,
                            c10::optional<at::Tensor> F,
                            c10::optional<at::Tensor> TokenIndex,
                            at::Tensor K, at::Tensor V, int len,
                            int hn, 
                            const int group_size) {
    bool qkv_packed = WQKV.has_value();
    int freq_type = F.has_value() ? F.value().size(1) == K.size(-1) ? 2 : 1 : 0;  // 0 means no F, 1 means F is half, 2 means F is full
    bool aligned = !TokenIndex.has_value();

    int bs = I.size(0);
    int seqlen = I.size(1);
    int hn_kv = K.size(-2);
    int hs = K.size(-1);
    int dim_out_kv = hn_kv * hs;
    int dim_out = hn * hs;

    at::Tensor Q = torch::empty({bs, seqlen, hn, hs},
                                at::device(I.device()).dtype(I.dtype()));
    at::Tensor WorkSpace;
    if (qkv_packed) {
        WorkSpace = torch::empty({bs, seqlen, dim_out + dim_out_kv * 2},
                                 at::device(I.device()).dtype(I.dtype()));
    } else {
        WorkSpace = torch::empty({bs, seqlen, dim_out_kv}, 
                    at::device(I.device()).dtype(I.dtype()));
    }

    c10::optional<at::Tensor> none = c10::nullopt;

    if (aligned) {
        if (qkv_packed) {
            if (freq_type == 0)
                Gemm_awq_qkvpacked_rope<ROPE_TYPE::NO_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::YES>(
                    I, WQKV.value(), WQKV_zeros_scales.value(), BQKV, WorkSpace, none, none, Q, K, V, len, group_size);
            else if (freq_type == 1)
                Gemm_awq_qkvpacked_rope<ROPE_TYPE::HALF_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::YES>(
                    I, WQKV.value(), WQKV_zeros_scales.value(), BQKV, WorkSpace, F, none, Q, K, V, len, group_size);
            else
                Gemm_awq_qkvpacked_rope<ROPE_TYPE::FULL_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::YES>(
                    I, WQKV.value(), WQKV_zeros_scales.value(), BQKV, WorkSpace, F, none, Q, K, V, len, group_size);
        } else {
            if (freq_type == 0)
                Gemm_awq_qkvunpacked_rope<ROPE_TYPE::NO_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::YES>(
                    I, WQ.value(), WK.value(), WV.value(), WQ_zeros_scales.value(), WK_zeros_scales.value(), WV_zeros_scales.value(), BQ, BK, BV, WorkSpace, none, none, Q, K, V, len, group_size);
            else if (freq_type == 1)
                Gemm_awq_qkvunpacked_rope<ROPE_TYPE::HALF_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::YES>(
                    I, WQ.value(), WK.value(), WV.value(), WQ_zeros_scales.value(), WK_zeros_scales.value(), WV_zeros_scales.value(), BQ, BK, BV, WorkSpace, F, none, Q, K, V, len, group_size);
            else
                Gemm_awq_qkvunpacked_rope<ROPE_TYPE::FULL_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::YES>(
                    I, WQ.value(), WK.value(), WV.value(), WQ_zeros_scales.value(), WK_zeros_scales.value(), WV_zeros_scales.value(), BQ, BK, BV, WorkSpace, F, none, Q, K, V, len, group_size);
        }
    } else {
        if (qkv_packed) {
            if (freq_type == 0)
                Gemm_awq_qkvpacked_rope<ROPE_TYPE::NO_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::NO>(
                    I, WQKV.value(), WQKV_zeros_scales.value(), BQKV, WorkSpace, none, none, Q, K, V, len, group_size);
            else if (freq_type == 1)
                Gemm_awq_qkvpacked_rope<ROPE_TYPE::HALF_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::NO>(
                    I, WQKV.value(), WQKV_zeros_scales.value(), BQKV, WorkSpace, F, TokenIndex, Q, K, V, len, group_size);
            else
                Gemm_awq_qkvpacked_rope<ROPE_TYPE::FULL_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::NO>(
                    I, WQKV.value(), WQKV_zeros_scales.value(), BQKV, WorkSpace, F, TokenIndex, Q, K, V, len, group_size);
        } else {
            if (freq_type == 0)
                Gemm_awq_qkvunpacked_rope<ROPE_TYPE::NO_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::NO>(
                    I, WQ.value(), WK.value(), WV.value(), WQ_zeros_scales.value(), WK_zeros_scales.value(), WV_zeros_scales.value(), BQ, BK, BV, WorkSpace, none, none, Q, K, V, len, group_size);
            else if (freq_type == 1)
                Gemm_awq_qkvunpacked_rope<ROPE_TYPE::HALF_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::NO>(
                    I, WQ.value(), WK.value(), WV.value(), WQ_zeros_scales.value(), WK_zeros_scales.value(), WV_zeros_scales.value(), BQ, BK, BV, WorkSpace, F, TokenIndex, Q, K, V, len, group_size);
            else
                Gemm_awq_qkvunpacked_rope<ROPE_TYPE::FULL_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::NO>(
                    I, WQ.value(), WK.value(), WV.value(), WQ_zeros_scales.value(), WK_zeros_scales.value(), WV_zeros_scales.value(), BQ, BK, BV, WorkSpace, F, TokenIndex, Q, K, V, len, group_size);
        }
    }

    return Q;
}

at::Tensor gemm_pure_ut(at::Tensor I, at::Tensor W) {
    // I: [bs, seqlen, dim_in] or [seqlen, bs, dim_in]
    // W: [dim_out, dim_in]
    // O: [bs, seqlen, dim_out] or [seqlen, bs, dim_out]

    at::Tensor O = torch::empty({I.size(0), I.size(1), W.size(0)},
                                at::device(I.device()).dtype(I.dtype()));

    c10::optional<at::Tensor> none = c10::nullopt;
    Gemm_pure(I, W, none, O);

    return O;
}

at::Tensor gemm_residual_ut(at::Tensor I, at::Tensor W, at::Tensor R) {
    // I: [bs, seqlen, dim_in] or [seqlen, bs, dim_in]
    // W: [dim_out, dim_in]
    // O: [bs, seqlen, dim_out] or [seqlen, bs, dim_out]

    c10::optional<at::Tensor> none = c10::nullopt;
    Gemm_residual(I, W, none, R);

    return R;
}

at::Tensor gemm_silu_dot_ut(at::Tensor I, at::Tensor W1, at::Tensor W2) {
    // I: [bs, seqlen, dim_in] or [seqlen, bs, dim_in]
    // W1: [dim_out, dim_in]
    // W2: [dim_out, dim_in]
    // O: [bs, seqlen, dim_out] or [seqlen, bs, dim_out]

    at::Tensor O = torch::empty({I.size(0), I.size(1), W1.size(0)},
                                at::device(I.device()).dtype(I.dtype()));

    c10::optional<at::Tensor> none = c10::nullopt;
    Gemm_silu_dot(I, W1, W2, none, O);

    return O;
}

at::Tensor gemm_rope_ut(at::Tensor I,
                        c10::optional<at::Tensor> WQ, c10::optional<at::Tensor> WK, c10::optional<at::Tensor> WV,
                        c10::optional<at::Tensor> BQ, c10::optional<at::Tensor> BK, c10::optional<at::Tensor> BV,
                        c10::optional<at::Tensor> WQKV, c10::optional<at::Tensor> BQKV,
                        c10::optional<at::Tensor> F, c10::optional<at::Tensor> TokenIndex,
                        at::Tensor K, at::Tensor V, int len) {
    bool qkv_packed = WQKV.has_value();
    int freq_type = F.has_value() ? F.value().size(1) == K.size(-1) ? 2 : 1 : 0;  // 0 means no F, 1 means F is half, 2 means F is full
    bool aligned = !TokenIndex.has_value();

    int bs = I.size(0);
    int seqlen = I.size(1);
    int hn_kv = K.size(-2);
    int hs = K.size(-1);
    int dim_out_kv = hn_kv * hs;

    int dim_out = qkv_packed ? WQKV.value().size(0) - dim_out_kv * 2 : WQ.value().size(0);
    int hn = dim_out / hs;

    at::Tensor Q = torch::empty({bs, seqlen, hn, hs},
                                at::device(I.device()).dtype(I.dtype()));
    at::Tensor WorkSpace;
    if (qkv_packed) {
        WorkSpace = torch::empty({bs, seqlen, dim_out + dim_out_kv * 2},
                                 at::device(I.device()).dtype(I.dtype()));
    } else {
        WorkSpace = torch::empty({bs, seqlen, dim_out_kv}, 
                    at::device(I.device()).dtype(I.dtype()));
    }

    c10::optional<at::Tensor> none = c10::nullopt;

    if (aligned) {
        if (qkv_packed) {
            if (freq_type == 0)
                Gemm_qkvpacked_rope<ROPE_TYPE::NO_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::YES>(
                    I, WQKV.value(), BQKV, WorkSpace, none, none, Q, K, V, len);
            else if (freq_type == 1)
                Gemm_qkvpacked_rope<ROPE_TYPE::HALF_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::YES>(
                    I, WQKV.value(), BQKV, WorkSpace, F, none, Q, K, V, len);
            else
                Gemm_qkvpacked_rope<ROPE_TYPE::FULL_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::YES>(
                    I, WQKV.value(), BQKV, WorkSpace, F, none, Q, K, V, len);
        } else {
            if (freq_type == 0)
                Gemm_qkvunpacked_rope<ROPE_TYPE::NO_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::YES>(
                    I, WQ.value(), WK.value(), WV.value(), BQ, BK, BV, WorkSpace, none, none, Q, K, V, len);
            else if (freq_type == 1)
                Gemm_qkvunpacked_rope<ROPE_TYPE::HALF_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::YES>(
                    I, WQ.value(), WK.value(), WV.value(), BQ, BK, BV, WorkSpace, F, none, Q, K, V, len);
            else
                Gemm_qkvunpacked_rope<ROPE_TYPE::FULL_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::YES>(
                    I, WQ.value(), WK.value(), WV.value(), BQ, BK, BV, WorkSpace, F, none, Q, K, V, len);
        }
    } else {
        if (qkv_packed) {
            if (freq_type == 0)
                Gemm_qkvpacked_rope<ROPE_TYPE::NO_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::NO>(
                    I, WQKV.value(), BQKV, WorkSpace, none, none, Q, K, V, len);
            else if (freq_type == 1)
                Gemm_qkvpacked_rope<ROPE_TYPE::HALF_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::NO>(
                    I, WQKV.value(), BQKV, WorkSpace, F, TokenIndex, Q, K, V, len);
            else
                Gemm_qkvpacked_rope<ROPE_TYPE::FULL_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::NO>(
                    I, WQKV.value(), BQKV, WorkSpace, F, TokenIndex, Q, K, V, len);
        } else {
            if (freq_type == 0)
                Gemm_qkvunpacked_rope<ROPE_TYPE::NO_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::NO>(
                    I, WQ.value(), WK.value(), WV.value(), BQ, BK, BV, WorkSpace, none, none, Q, K, V, len);
            else if (freq_type == 1)
                Gemm_qkvunpacked_rope<ROPE_TYPE::HALF_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::NO>(
                    I, WQ.value(), WK.value(), WV.value(), BQ, BK, BV, WorkSpace, F, TokenIndex, Q, K, V, len);
            else
                Gemm_qkvunpacked_rope<ROPE_TYPE::FULL_ROPE, SEQ_DIM_TYPE::SECOND, FREQ_ALIGNED::NO>(
                    I, WQ.value(), WK.value(), WV.value(), BQ, BK, BV, WorkSpace, F, TokenIndex, Q, K, V, len);
        }
    }

    return Q;
}

at::Tensor attention_ut(at::Tensor Q, at::Tensor K, at::Tensor V, c10::optional<at::Tensor> alibi_slopes,
                        const float scale, const float attn_max, const int strategy) {
    // Q: [bs, seqlen, hn, hs]
    // K: [..., hn, hs]
    // V: [..., hn, hs]

    bool alibi_mask = alibi_slopes.has_value();

    int bs = Q.size(0);
    int seqlen = Q.size(1);
    int hn = Q.size(2);
    int hs = Q.size(3);

    at::Tensor H = torch::empty({bs, seqlen, hn, hs}, at::device(Q.device()).dtype(Q.dtype()));
    at::Tensor Workspace = torch::empty({bs, hn, strategy > 1 ? strategy : 0, hs + 1}, at::device(Q.device()).dtype(at::ScalarType::Float));

    c10::optional<at::Tensor> none = c10::nullopt;

    if (alibi_mask)
        Attention<MASK_TYPE::ALIBI_MASK, SEQ_DIM_TYPE::SECOND>(Q, K, V, strategy > 1 ? Workspace : none, alibi_slopes, scale, attn_max, strategy, H);
    else
        Attention<MASK_TYPE::NO_MASK, SEQ_DIM_TYPE::SECOND>(Q, K, V, strategy > 1 ? Workspace : none, none, scale, attn_max, strategy, H);

    return H;
}

at::Tensor attention_serving_ut(at::Tensor Q, at::Tensor K, at::Tensor V, c10::optional<at::Tensor> alibi_slopes,
                                c10::optional<at::Tensor> Q_context_length, c10::optional<at::Tensor> K_context_length,
                                c10::optional<at::Tensor> Block_table, c10::optional<at::Tensor> Cache_seqlens, const float scale) {
    bool alibi_mask = alibi_slopes.has_value();
    bool prefill = Q.dim() == 3 ? true : false;

    if (prefill) {
        int tokens = Q.size(0);
        int hn = Q.size(1);
        int hs = Q.size(2);
        at::Tensor H = torch::empty({tokens, hn, hs},
                                    at::device(Q.device()).dtype(Q.dtype()));

        if (alibi_mask)
            Attention_serving<MASK_TYPE::ALIBI_MASK>(Q, K, V, alibi_slopes, Q_context_length, K_context_length, Block_table, Cache_seqlens, scale, H);
        else
            Attention_serving<MASK_TYPE::NO_MASK>(Q, K, V, alibi_slopes, Q_context_length, K_context_length, Block_table, Cache_seqlens, scale, H);

        return H;
    } else {
        // error here
        int bs = Q.size(0);
        int seqlen = Q.size(1);
        int hn = Q.size(2);
        int hs = Q.size(3);
        at::Tensor H = torch::empty({bs, seqlen, hn, hs},
                                    at::device(Q.device()).dtype(Q.dtype()));

        if (alibi_mask)
            Attention_serving<MASK_TYPE::ALIBI_MASK>(Q, K, V, alibi_slopes, Q_context_length, K_context_length, Block_table, Cache_seqlens, scale, H);
        else
            Attention_serving<MASK_TYPE::NO_MASK>(Q, K, V, alibi_slopes, Q_context_length, K_context_length, Block_table, Cache_seqlens, scale, H);

        return H;
    }
}

at::Tensor rmsnorm_ut(at::Tensor X, at::Tensor RW, const float eps) {
    // X: [bs, seqlen, dim]
    // RW: [dim]

    at::Tensor O = torch::empty({X.size(0), X.size(1), X.size(2)},
                                at::device(X.device()).dtype(X.dtype()));

    Rmsnorm(X, RW, O, eps);

    return O;
}

std::tuple<at::Tensor, at::Tensor> residual_rmsnorm_ut(at::Tensor X, at::Tensor R, at::Tensor RW, const float eps) {
    // X: [bs, seqlen, dim]
    // R: [bs, seqlen, dim]
    // RW: [dim]

    at::Tensor O = torch::empty({X.size(0), X.size(1), X.size(2)},
                                at::device(X.device()).dtype(X.dtype()));
    at::Tensor RO = torch::empty({X.size(0), X.size(1), X.size(2)},
                                 at::device(X.device()).dtype(X.dtype()));

    Residual_rmsnorm(X, R, RW, RO, O, eps);

    return {RO, O};
}

at::Tensor residual_ut(at::Tensor R, at::Tensor X) {
    // R: [bs, seqlen, dim]
    // X: [bs, seqlen, dim]

    at::Tensor O = torch::empty({X.size(0), X.size(1), X.size(2)},
                                at::device(R.device()).dtype(R.dtype()));

    Residual(R, X, O);

    return O;
}

at::Tensor embedding_ut(at::Tensor X, at::Tensor W) {
    // X: [bs, seqlen]
    // W: [vocab_size, dim]

    at::Tensor O = torch::empty({X.size(0), X.size(1), W.size(1)},
                                at::device(W.device()).dtype(W.dtype()));

    Embedding(X, W, O);

    return O;
}

void cache_ut(at::Tensor K, at::Tensor V, at::Tensor K_cache, at::Tensor V_cache, at::Tensor slot_mapping) {
    Cache(K, V, K_cache, V_cache, slot_mapping);
}