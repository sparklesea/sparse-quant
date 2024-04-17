"""
Fused Attention
===============

This is a Triton implementation of the prefill and decode attention kernel. It support block sparse.

"""
import torch
from torch_scatter import scatter

import triton
import triton.language as tl

"""
the single density kernel, which is the fastest for now
sparse_attention_prefill
"""

@triton.jit
def _sparse_attention_prefill_fwd_kernel(
    Q, K, V, sm_scale,
    Out,
    lut,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_lz, stride_lh, stride_lx,
    Z, H, N_CTX,
    NNZ: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    lut_indicator = tl.program_id(1) % H
    qvk_offset = off_hz * stride_qh
    lut_offset = lut_indicator * stride_lz
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # credits to: Adam P. Goucher (https://github.com/apgoucher):
    # scale sm_scale by 1/log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504

    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(tl.float16)

    # last_nnz_id: used when padding
    last_nnz_id = -1

    # loop over k, v and update accumulator
    for nnz_id in range(NNZ):
        # we use nnz_id to indicate which block will be computed
        present_nnz_id = tl.load(lut + lut_offset + start_m * stride_lh + nnz_id * stride_lx)
        start_n = present_nnz_id * BLOCK_N
        present_nnz_id = present_nnz_id.to(tl.int32)

        # -- compute qk ----
        k = tl.load(tl.advance(K_block_ptr, (0, start_n)))
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)

        # if a blcok in the upper triangular is selected in 'casual' case,
        # that is, m_ij == '-inf', in this case, 
        # p should be set to '0', or exp2('inf') will be met
        p = tl.where(m_ij == float("-inf"), tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32), tl.math.exp2(qk - m_ij[:, None]))
        p = tl.where(last_nnz_id == present_nnz_id, tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32), p)
        l_ij = tl.sum(p, 1)

        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_i_new)
        beta = tl.math.exp2(m_ij - m_i_new)
        l_i *= alpha
        l_i_new = l_i + beta * l_ij

        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]

        # scale acc
        acc_scale = l_i / l_i_new
        acc = acc * acc_scale[:, None]

        # update acc
        v = tl.load(tl.advance(V_block_ptr, (start_n, 0)))
        p = p.to(tl.float16)
        acc += tl.dot(p, v)

        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

        # update last_nnz_id
        last_nnz_id = present_nnz_id

    # write back l and m
    # l_ptrs = L + off_hz * N_CTX + offs_m
    # m_ptrs = M + off_hz * N_CTX + offs_m
    # tl.store(l_ptrs, l_i)
    # tl.store(m_ptrs, m_i)

    # write back O
    tl.store(O_block_ptr, acc.to(tl.float16))

class _sparse_attention_prefill(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, lut, BLOCK_M: int = 64, BLOCK_N: int = 64) -> torch.Tensor:
        """
        input:
            q, k, v: (Z, H, N_CTX, L)
            sm_scale: float
            lut: (H, N_CTX/BLOCK_M, nnz)
            BLOCK_M, BLOCK_N: int
        output:
            o: (Z, H, N_CTX, L)
        """
        dtype = q.dtype
        assert dtype == torch.float16

        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        # assert Lk in {16, 32, 64, 128}

        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        # L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        # m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        NNZ = lut.shape[-1]

        num_warps = 4 if Lk <= 64 else 8
        num_stages = 4 if BLOCK_M <= 32 else 2

        _sparse_attention_prefill_fwd_kernel[grid](
            q, k ,v, sm_scale,
            o,
            lut,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            lut.stride(0), lut.stride(1), lut.stride(2),
            q.shape[0], q.shape[1], q.shape[2], NNZ,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=Lk, BLOCK_N=BLOCK_N, 
            num_warps=num_warps,
            num_stages=num_stages)
        
        return o
    
sparse_attention_prefill = _sparse_attention_prefill.apply


"""
the scatter kernel
sparse_attention_prefill_scatter
"""

@triton.jit
def _sparse_attention_prefill_fwd_scatter_kernel(
    Q, K, V, sm_scale,
    L, M,
    Out,
    lut, lut_for_head,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_lz, stride_lh, stride_lx,
    Z, H, N_CTX,
    NNZ: tl.constexpr,
    new_num_heads: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    lut_indicator = tl.program_id(1) % new_num_heads
    head_indicator = tl.load(lut_for_head + lut_indicator)
    bsz_indicator = tl.program_id(1) // new_num_heads

    start_m = tl.program_id(0)
    off_hz = bsz_indicator * H + head_indicator

    qvk_offset = off_hz * stride_qh
    o_offset = tl.program_id(1) * stride_oh
    lut_offset = lut_indicator * stride_lz

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # credits to: Adam P. Goucher (https://github.com/apgoucher):
    # scale sm_scale by 1/log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504

    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(tl.float16)

    # loop over k, v and update accumulator
    for nnz_id in range(NNZ):
        # we use nnz_id to indicate which block will be computed
        start_n = tl.load(lut + lut_offset + start_m * stride_lh + nnz_id * stride_lx)
        start_n = start_n * BLOCK_N

        # -- compute qk ----
        k = tl.load(tl.advance(K_block_ptr, (0, start_n)))
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)

        # if a blcok in the upper triangular is selected in 'casual' case,
        # that is, m_ij == '-inf', in this case, 
        # p should be set to '0', or exp2('inf') will be met
        p = tl.where(m_ij == float("-inf"), tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32), tl.math.exp2(qk - m_ij[:, None]))
        l_ij = tl.sum(p, 1)

        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.where(m_i_new == float("-inf"), 0, tl.math.exp2(m_i - m_i_new))
        beta = tl.where(m_i_new == float("-inf"), 0, tl.math.exp2(m_ij - m_i_new))
        l_i *= alpha
        l_i_new = l_i + beta * l_ij

        # scale p
        p_scale = tl.where(m_i_new == float("-inf"), 0, beta / l_i_new)
        p = p * p_scale[:, None]

        # scale acc
        acc_scale = tl.where(m_i_new == float("-inf"), 0, l_i / l_i_new)
        acc = acc * acc_scale[:, None]

        # update acc
        v = tl.load(tl.advance(V_block_ptr, (start_n, 0)))
        p = p.to(tl.float16)
        acc += tl.dot(p, v)


        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    # write back l and m
    l_ptrs = L + tl.program_id(1) * N_CTX + offs_m
    m_ptrs = M + tl.program_id(1) * N_CTX + offs_m
    tl.store(l_ptrs, l_i)
    tl.store(m_ptrs, tl.math.exp2(m_i))

    # write back O
    tl.store(O_block_ptr, acc.to(tl.float16))

class _sparse_attention_prefill_scatter(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, lut, lut_for_head, BLOCK_M: int = 64, BLOCK_N: int = 64) -> torch.Tensor:
        """
        input:
            q, k, v: (Z, H, N_CTX, L)
            sm_scale: float
            lut: (H_new, N_CTX/BLOCK_M, nnz)
            lut_for_head: (H_new)
            BLOCK_M, BLOCK_N: int
        output:
            o: (Z, H_new, N_CTX, L)
            L: (Z*H_new, N_CTX) row sum
            m: (Z*H_new, N_CTX) max element of each row
        """
        dtype = q.dtype
        assert dtype == torch.float16

        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        # assert Lk in {16, 32, 64, 128}

        o = torch.empty((q.shape[0], lut.shape[0], q.shape[2], q.shape[-1]), device=q.device, dtype=dtype)
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * lut.shape[0], 1)
        L = torch.empty((q.shape[0] * lut.shape[0], q.shape[2]), device=q.device, dtype=torch.float32)
        m = torch.empty((q.shape[0] * lut.shape[0], q.shape[2]), device=q.device, dtype=torch.float32)
        
        # L_sum = torch.empty(q.shape[0], lut.shape[0], q.shape[2], device=q.device, dtype=dtype)
        # output = torch.empty_like(q, device=q.device, dtype=dtype)

        NNZ = lut.shape[-1]

        num_heads = q.shape[1]
        new_num_heads = lut.shape[0]

        num_warps = 4 if Lk <= 64 else 8
        num_stages = 4 if BLOCK_M <= 32 else 2

        _sparse_attention_prefill_fwd_scatter_kernel[grid](
            q, k ,v, sm_scale,
            L, m,
            o,
            lut, lut_for_head,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            lut.stride(0), lut.stride(1), lut.stride(2),
            q.shape[0], q.shape[1], q.shape[2], NNZ, 
            new_num_heads, 
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=Lk, BLOCK_N=BLOCK_N, 
            num_warps=num_warps,
            num_stages=num_stages)

        L = L * m
        L = L.reshape(q.shape[0], new_num_heads, -1).to(dtype)
        # L_sum.scatter_add_(1, lut_for_head.unsqueeze(0).unsqueeze(2).expand(q.shape[0], -1, q.shape[2]), L)
        # L_sum.scatter_add_(1, lut_for_head.unsqueeze(0).unsqueeze(2), L)
        L_sum = scatter(L, lut_for_head, 1, reduce='sum')
        L_sum_expanded = torch.index_select(L_sum, 1, lut_for_head)
        L = L / L_sum_expanded

        o = o * (L.unsqueeze(dim=-1))
        # output.scatter_add_(1, lut_for_head.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(o.shape[0], -1, o.shape[2], o.shape[3]), o)
        # output.scatter_add_(1, lut_for_head.unsqueeze(0).unsqueeze(2).unsqueeze(3), o)
        output = scatter(o, lut_for_head, 1, reduce='sum')

        return output
    
sparse_attention_prefill_scatter = _sparse_attention_prefill_scatter.apply


"""
the simple reduce kernel
sparse_attention_prefill_simple_reduce
"""

@triton.jit
def _sparse_attention_prefill_fwd_simple_reduce_kernel(
    Q, K, V, sm_scale,
    L, M,
    Out,
    lut, lut_for_head, lut_for_o,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_lz, stride_lh, stride_lx,
    Z, H, N_CTX,
    NNZ: tl.constexpr,
    N: tl.constexpr,
    new_num_heads: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    lut_indicator = tl.program_id(1) % new_num_heads
    head_indicator = tl.load(lut_for_head + lut_indicator)
    o_indicator = tl.load(lut_for_o + lut_indicator)
    bsz_indicator = tl.program_id(1) // new_num_heads

    start_m = tl.program_id(0)
    off_hz = bsz_indicator * H + head_indicator
    off_o = bsz_indicator * H + o_indicator

    qvk_offset = off_hz * stride_qh
    o_offset = off_o * stride_oh
    lut_offset = lut_indicator * stride_lz

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # credits to: Adam P. Goucher (https://github.com/apgoucher):
    # scale sm_scale by 1/log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504

    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(tl.float16)

    # loop over k, v and update accumulator
    for nnz_id in range(NNZ):
        # we use nnz_id to indicate which block will be computed
        start_n = tl.load(lut + lut_offset + start_m * stride_lh + nnz_id * stride_lx)
        start_n = start_n * BLOCK_N

        # -- compute qk ----
        k = tl.load(tl.advance(K_block_ptr, (0, start_n)))
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)

        # if a blcok in the upper triangular is selected in 'casual' case,
        # that is, m_ij == '-inf', in this case, 
        # p should be set to '0', or exp2('inf') will be met
        p = tl.where(m_ij == float("-inf"), tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32), tl.math.exp2(qk - m_ij[:, None]))
        l_ij = tl.sum(p, 1)

        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.where(m_i_new == float("-inf"), 0, tl.math.exp2(m_i - m_i_new))
        beta = tl.where(m_i_new == float("-inf"), 0, tl.math.exp2(m_ij - m_i_new))
        l_i *= alpha
        l_i_new = l_i + beta * l_ij

        # scale p
        p_scale = tl.where(m_i_new == float("-inf"), 0, beta / l_i_new)
        p = p * p_scale[:, None]

        # scale acc
        acc_scale = tl.where(m_i_new == float("-inf"), 0, l_i / l_i_new)
        acc = acc * acc_scale[:, None]

        # update acc
        v = tl.load(tl.advance(V_block_ptr, (start_n, 0)))
        p = p.to(tl.float16)
        acc += tl.dot(p, v)


        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    # write back l and m
    l_ptrs = L + off_o * N_CTX + offs_m
    # m_ptrs = M + off_o * N_CTX + offs_m
    tl.store(l_ptrs, l_i * tl.math.exp2(m_i))
    # tl.store(m_ptrs, tl.math.exp2(m_i))

    # write back O
    tl.store(O_block_ptr, acc.to(tl.float16))

class _sparse_attention_prefill_simple_reduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, lut, lut_for_head, lut_for_o, N, BLOCK_M: int = 64, BLOCK_N: int = 64) -> torch.Tensor:
        """
        input:
            q, k, v: (Z, H, N_CTX, L)
            sm_scale: float
            lut: (H_new, N_CTX/BLOCK_M, nnz)
            lut_for_head: (H_new)
            lut_for_o: (H_new)
            N: int
            BLOCK_M, BLOCK_N: int
        output:
            o: (Z, H_new, N_CTX, L)
            L: (Z*H_new, N_CTX) row sum
            m: (Z*H_new, N_CTX) max element of each row
        """

        dtype = q.dtype
        # assert dtype == torch.float16
        # assert Lk in {16, 32, 64, 128}
        # assert Lq == Lk and Lk == Lv

        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        
        
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * lut.shape[0], 1)
        o = torch.zeros((q.shape[0], q.shape[1] * N, q.shape[2], q.shape[-1]), device=q.device, dtype=dtype)
        L = torch.zeros((q.shape[0] * q.shape[1] * N, q.shape[2]), device=q.device, dtype=torch.float32)
        # m = torch.zeros((q.shape[0] * N * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        NNZ = lut.shape[-1]

        num_heads = q.shape[1]
        new_num_heads = lut.shape[0]

        num_warps = 4 if Lk <= 64 else 8
        num_stages = 4 if BLOCK_M <= 32 else 2

        _sparse_attention_prefill_fwd_simple_reduce_kernel[grid](
            q, k ,v, sm_scale,
            L, L,
            o,
            lut, lut_for_head, lut_for_o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            lut.stride(0), lut.stride(1), lut.stride(2),
            q.shape[0], q.shape[1], q.shape[2], NNZ, N,
            new_num_heads, 
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=Lk, BLOCK_N=BLOCK_N, 
            num_warps=num_warps,
            num_stages=num_stages)
        
        
        o = o.view(q.shape[0] * q.shape[1], N, q.shape[2], q.shape[-1])
        L = L.view(q.shape[0] * q.shape[1], N, q.shape[2])
        # m = m.view(q.shape[0] * q.shape[1], N, q.shape[2])

        # L = L * m
        # L_sum = L.to(dtype).sum(dim=1, keepdim=True)
        # .repeat(1, N).reshape(q.shape[0]*q.shape[1]*N, q.shape[2])
        L = L / L.to(dtype).sum(dim=1, keepdim=True)

        # o = o * (L.unsqueeze(dim=-1))
        output = (o * (L.unsqueeze(dim=-1))).sum(dim=1).view(q.shape[0], q.shape[1], q.shape[2], q.shape[-1])
        return output
    
sparse_attention_prefill_simple_reduce = _sparse_attention_prefill_simple_reduce.apply


"""
kernel by genghan
"""
@triton.jit
def _sparse_attention_prefill_fwd_kernel_v2(
    Q, K, V, sm_scale,
    L,
    Out,
    lut,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_lh, stride_lx, stride_ly,
    H, N_CTX,
    NNZ: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_h = off_hz % H
    qvk_offset = off_hz * stride_qh
    lut_offset = off_h * stride_lh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize lut
    lut_base = lut + lut_offset + start_m * stride_lx
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(tl.float16)
    # loop over k, v and update accumulator
    for nnz_id in range(NNZ):
        # we use nnz_id to indicate which block will be computed
        start_n = tl.load(lut_base + nnz_id * stride_ly) * BLOCK_N
        
        # -- load kv ----
        k = tl.load(tl.advance(K_block_ptr, (0, start_n)))
        v = tl.load(tl.advance(V_block_ptr, (start_n, 0)))
        # -- compute qk ----
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        qk += tl.dot(q, k, out_dtype=tl.float16)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(tl.float16), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
    
    # write back l
    acc = acc / l_i[:, None]
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(tl.float16))
        
        
@triton.jit
def _bwd_preprocess(
    Out, DO,
    Delta,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    # compute
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_m, delta)
    
@triton.jit
def _sparse_attention_prefill_bwd_kernel_v2_fp16(
    Q, K, V, sm_scale, DO,
    DQ, DK, DV,
    L,
    D,
    lut,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_lh, stride_lx, stride_ly,
    H, N_CTX, num_block,
    NNZ: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hz = tl.program_id(0)
    qvk_offset = off_hz * stride_qh
    qk_scale = sm_scale * 1.44269504
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    ) # Use K and K^T in bwd
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    ) # Only use V^T in bwd
    DO_block_ptr = tl.make_block_ptr(
        base=DO + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    DQ_block_ptr = tl.make_block_ptr(
        base=DQ + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    DK_block_ptr = tl.make_block_ptr(
        base=DK + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    off_h = off_hz % H
    lut_base = lut + off_h * stride_lh
    D_ptrs = D + off_hz * N_CTX
    l_ptrs = L + off_hz * N_CTX
    offs_n = tl.arange(0, BLOCK_N)

    for start_m in range(0, num_block):
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        # -- Update lut ptr --
        lut_ptrs = lut_base + start_m * stride_lx
        # -- load q --
        Q_block_ptr_curr = tl.advance(Q_block_ptr, (start_m * BLOCK_M, 0))
        q = tl.load(Q_block_ptr_curr)
        # -- load li, Di, do --
        DO_block_ptr_curr = tl.advance(DO_block_ptr, (start_m * BLOCK_M, 0))
        do = tl.load(DO_block_ptr_curr)
        Di = tl.load(D_ptrs + offs_m)
        l_i = tl.load(l_ptrs + offs_m)
        # -- Initialize dq --
        dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float16)
        # -- loop over n --
        for nnz_id in range(NNZ):
            # -- load start_n from lut --
            start_n = tl.load(lut_ptrs + nnz_id * stride_ly) * BLOCK_N
            # -- load k, v, dk, dv --
            K_block_ptr_curr = tl.advance(K_block_ptr, (start_n, 0))
            k = tl.load(K_block_ptr_curr)
            # -- compute p = softmax(qk^T, dim=-1) --
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), float(0.), float("-inf"))
            qk += tl.dot(q, tl.trans(k))
            qk *= qk_scale
            p = tl.math.exp2(qk - l_i[:, None]).to(tl.float16)
            # -- Increment dv --
            DV_block_ptr_curr = tl.advance(DV_block_ptr, (start_n, 0))
            dv = tl.load(DV_block_ptr_curr)
            dv += tl.dot(tl.trans(p), do, out_dtype=tl.float16) # Like K, p is used as p and p^T in bwd
            tl.store(DV_block_ptr_curr, dv)
            # -- compute dp --
            V_block_ptr_curr = tl.advance(V_block_ptr, (0, start_n))
            v = tl.load(V_block_ptr_curr)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, v)
            # -- compute ds --
            ds = p * dp * sm_scale # Like K, ds is used as ds and ds^T in bwd
            # -- Increment dq --
            dq += tl.dot(ds.to(tl.float16), k, out_dtype=tl.float16)
            # -- Increment dk --
            DK_block_ptr_curr = tl.advance(DK_block_ptr, (start_n, 0))
            dk = tl.load(DK_block_ptr_curr)
            dk += tl.dot(tl.trans(ds.to(tl.float16)), q, out_dtype=tl.float16)
            tl.store(DK_block_ptr_curr, dk)
        # -- write back dq --
        DQ_block_ptr_curr = tl.advance(DQ_block_ptr, (start_m * BLOCK_M, 0))
        tl.store(DQ_block_ptr_curr, dq)
    
@triton.jit
def _sparse_attention_prefill_bwd_kernel_v2_rawptr_fp16(
    Q, K, V, sm_scale, DO,
    DQ, DK, DV,
    L,
    D,
    lut,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_lh, stride_lx, stride_ly,
    H, N_CTX, num_block,
    NNZ: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hz = tl.program_id(0)
    qvk_offset = off_hz * stride_qh
    qk_scale = sm_scale * 1.44269504
    Q += qvk_offset
    K += qvk_offset
    V += qvk_offset
    DO += qvk_offset
    DQ += qvk_offset
    DK += qvk_offset
    DV += qvk_offset
    off_h = off_hz % H
    lut_base = lut + off_h * stride_lh
    D_ptrs = D + off_hz * N_CTX
    l_ptrs = L + off_hz * N_CTX
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    offs_m = tl.arange(0, BLOCK_M)
    
    for start_m in range(0, num_block):
        offs_m_curr = start_m * BLOCK_M + offs_m
        # -- Update lut ptr --
        lut_ptrs = lut_base + start_m * stride_lx
        # -- load q --
        q_ptrs = Q + offs_m_curr[:, None] * stride_qm + offs_k[None, :] * stride_qk
        q = tl.load(q_ptrs)
        # -- load li, Di, do --
        do_ptrs = DO + offs_m_curr[:, None] * stride_qm + offs_k[None, :] * stride_qk
        do = tl.load(do_ptrs)
        Di = tl.load(D_ptrs + offs_m_curr)
        l_i = tl.load(l_ptrs + offs_m_curr)
        # -- Initialize dq --
        dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float16)
        # -- loop over n --
        for nnz_id in range(0, NNZ):
            # -- load start_n from lut --
            start_n = tl.load(lut_ptrs + nnz_id * stride_ly)
            offs_n_curr = start_n * BLOCK_N + offs_n
            # -- load k --
            k_ptrs = K + offs_n_curr[:, None] * stride_kn + offs_k[None, :] * stride_kk
            k = tl.load(k_ptrs)
            # -- compute p = softmax(qk^T, dim=-1) --
            qk = tl.where(offs_m_curr[:, None] >= offs_n_curr[None, :], float(0.), float("-inf"))
            qk += tl.dot(q, tl.trans(k))
            qk *= qk_scale
            p = tl.math.exp2(qk - l_i[:, None])
            # -- Increment dv --
            dv_ptrs = DV + offs_n_curr[:, None] * stride_vn + offs_k[None, :] * stride_vk
            dv = tl.load(dv_ptrs)
            dv += tl.dot(tl.trans(p.to(tl.float16)), do, out_dtype=tl.float16)
            tl.store(dv_ptrs, dv)
            # -- compute dp --
            v_ptrs = V + offs_n_curr[:, None] * stride_vn + offs_k[None, :] * stride_vk
            v = tl.load(v_ptrs)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, tl.trans(v))
            # -- compute ds --
            ds = p * dp * sm_scale
            # -- Increment dq --
            dq += tl.dot(ds.to(tl.float16), k, out_dtype=tl.float16)
            # -- Increment dk --
            dk_ptrs = DK + offs_n_curr[:, None] * stride_kn + offs_k[None, :] * stride_kk
            dk = tl.load(dk_ptrs)
            dk += tl.dot(tl.trans(ds.to(tl.float16)), q, out_dtype=tl.float16)
            tl.store(dk_ptrs, dk)
        # -- write back dq --
        dq_ptrs = DQ + offs_m_curr[:, None] * stride_qm + offs_k[None, :] * stride_qk
        tl.store(dq_ptrs, dq)

@triton.jit
def _sparse_attention_prefill_bwd_kernel_v2_fp32(
    Q, K, V, sm_scale, DO,
    DQ, DK, DV,
    L,
    D,
    lut,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_lh, stride_lx, stride_ly,
    H, N_CTX, num_block,
    NNZ: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hz = tl.program_id(0)
    qvk_offset = off_hz * stride_qh
    qk_scale = sm_scale * 1.44269504
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    ) # Use K and K^T in bwd
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    ) # Only use V^T in bwd
    DO_block_ptr = tl.make_block_ptr(
        base=DO + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    DQ_block_ptr = tl.make_block_ptr(
        base=DQ + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    DK_block_ptr = tl.make_block_ptr(
        base=DK + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    off_h = off_hz % H
    lut_base = lut + off_h * stride_lh
    D_ptrs = D + off_hz * N_CTX
    l_ptrs = L + off_hz * N_CTX
    offs_n = tl.arange(0, BLOCK_N)

    for start_m in range(0, num_block):
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        # -- Update lut ptr --
        lut_ptrs = lut_base + start_m * stride_lx
        # -- load q --
        Q_block_ptr_curr = tl.advance(Q_block_ptr, (start_m * BLOCK_M, 0))
        q = tl.load(Q_block_ptr_curr)
        # -- load li, Di, do --
        DO_block_ptr_curr = tl.advance(DO_block_ptr, (start_m * BLOCK_M, 0))
        do = tl.load(DO_block_ptr_curr)
        Di = tl.load(D_ptrs + offs_m)
        l_i = tl.load(l_ptrs + offs_m)
        # -- Initialize dq --
        dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # -- loop over n --
        for nnz_id in range(NNZ):
            # -- load start_n from lut --
            start_n = tl.load(lut_ptrs + nnz_id * stride_ly) * BLOCK_N
            # -- load k, v, dk, dv --
            K_block_ptr_curr = tl.advance(K_block_ptr, (start_n, 0))
            k = tl.load(K_block_ptr_curr)
            # -- compute p = softmax(qk^T, dim=-1) --
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), float(0.), float("-inf"))
            qk += tl.dot(q, tl.trans(k))
            qk *= qk_scale
            p = tl.math.exp2(qk - l_i[:, None]).to(tl.float16)
            # -- Increment dv --
            DV_block_ptr_curr = tl.advance(DV_block_ptr, (start_n, 0))
            dv = tl.load(DV_block_ptr_curr)
            dv += tl.dot(tl.trans(p), do) # Like K, p is used as p and p^T in bwd
            tl.store(DV_block_ptr_curr, dv)
            # -- compute dp --
            V_block_ptr_curr = tl.advance(V_block_ptr, (0, start_n))
            v = tl.load(V_block_ptr_curr)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, v)
            # -- compute ds --
            ds = p * dp * sm_scale # Like K, ds is used as ds and ds^T in bwd
            # -- Increment dq --
            dq += tl.dot(ds.to(tl.float16), k)
            # -- Increment dk --
            DK_block_ptr_curr = tl.advance(DK_block_ptr, (start_n, 0))
            dk = tl.load(DK_block_ptr_curr)
            dk += tl.dot(tl.trans(ds.to(tl.float16)), q)
            tl.store(DK_block_ptr_curr, dk)
        # -- write back dq --
        DQ_block_ptr_curr = tl.advance(DQ_block_ptr, (start_m * BLOCK_M, 0))
        tl.store(DQ_block_ptr_curr, dq)

class _sparse_attention_prefill_v2(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, lut, BLOCK_M: int = 64, BLOCK_N: int = 64) -> torch.Tensor:        
        """
        input:
            q, k, v: (Z, H, N_CTX, L)
            casual: bool
            sm_scale: float
            lut: (H, N_CTX/BLOCK_M, nnz)
            BLOCK_M, BLOCK_N: int
        output:
            o: (Z, H, N_CTX, L)
        """
        dtype = q.dtype
        assert dtype == torch.float16
        
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        
        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        
        NNZ = lut.shape[-1]
        
        num_warps = 4
        num_stages = 4 if Lk <= 64 else 3
        
        _sparse_attention_prefill_fwd_kernel_v2[grid](
            q, k, v, sm_scale,
            L,
            o,
            lut,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            lut.stride(0), lut.stride(1), lut.stride(2),
            q.shape[1], q.shape[2], NNZ,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Lk,
            num_warps=num_warps,
            num_stages=num_stages)

        ctx.save_for_backward(q, k, v, o, L, lut)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        ctx.NNZ = NNZ
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L, lut = ctx.saved_tensors
        do = do.contiguous()
        # Different from Triton example, because we use QO-stationary instead of KV-stationary.
        # dq = torch.zeros_like(q, dtype=torch.float32)
        # dk = torch.zeros_like(k, dtype=torch.float32)
        # dv = torch.zeros_like(v, dtype=torch.float32)
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        delta = torch.empty_like(L)
        _bwd_preprocess[(ctx.grid[0] * ctx.grid[1],)](
            o, do,
            delta,
            BLOCK_M=ctx.BLOCK_M, D_HEAD=ctx.BLOCK_DMODEL,
        )
        # _sparse_attention_prefill_bwd_kernel_v2_rawptr_fp16[(ctx.grid[1],)](
        _sparse_attention_prefill_bwd_kernel_v2_fp16[(ctx.grid[1],)](
        # _sparse_attention_prefill_bwd_kernel_v2_fp32[(ctx.grid[1],)](
            q, k, v, ctx.sm_scale, do,
            dq, dk, dv,
            L, delta,
            lut,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            lut.stride(0), lut.stride(1), lut.stride(2),
            q.shape[1], q.shape[2], ctx.grid[0], ctx.NNZ, 
            BLOCK_M=ctx.BLOCK_M, BLOCK_DMODEL=ctx.BLOCK_DMODEL, BLOCK_N=ctx.BLOCK_N,
            num_warps=4, 
            num_stages=1)
        return dq, dk, dv, None, None, None, None
    
sparse_attention_prefill_v2 = _sparse_attention_prefill_v2.apply