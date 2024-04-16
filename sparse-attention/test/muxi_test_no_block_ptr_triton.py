

import torch
import math

import triton
import triton.language as tl

# @triton.jit
# def _sparse_attention_prefill_fwd_kernel(
#     Q, K, V, sm_scale, 
#     Out,
#     lut, p_ref, 
#     stride_qz, stride_qh, stride_qm, stride_qk,
#     stride_kz, stride_kh, stride_kk, stride_kn,
#     stride_vz, stride_vh, stride_vk, stride_vn,
#     stride_oz, stride_oh, stride_om, stride_on,
#     stride_lz, stride_lh, stride_lx,
#     Z, H, N_CTX, 
#     NNZ: tl.constexpr,
#     BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
#     BLOCK_N: tl.constexpr,
# ):
#     # tl.static_print("static entered kernel")
#     start_m = tl.program_id(0)
#     off_hz = tl.program_id(1)
#     lut_indicator = tl.program_id(1) % H
#     qvk_offset = off_hz * stride_qh
#     lut_offset = lut_indicator * stride_lz

#     tx = tl.arange(0, BLOCK_M)
#     ty = tl.arange(0, BLOCK_N)
#     tk = tl.arange(0, BLOCK_DMODEL)
#     Q_ptrs = Q + qvk_offset + (start_m * BLOCK_M) * stride_qm + (tx[:, None] * stride_qm + tk[None, :] * stride_qk)
#     K_ptrs = K + qvk_offset + (tk[:, None] * stride_kk + ty[None, :] * stride_kn)
#     V_ptrs = V + qvk_offset + (ty[:, None] * stride_vk + tk[None, :] * stride_vn)
#     O_ptrs = Out + qvk_offset + (start_m * BLOCK_M) * stride_om + (tx[:, None] * stride_om + tk[None, :] * stride_on)
#     P_ptrs = p_ref + (tx[:, None] * 64 + ty[None, :] * 1)

#     offs_m = start_m * BLOCK_M + tx
#     offs_n = ty

#     m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
#     l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
#     acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

#     qk_scale = sm_scale * 1.44269504

#     # tl.static_print("static loading q")
#     q = tl.load(Q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
#     tl.device_assert(qk_scale == 1)
#     # if off_hz == 1:
#     #     tl.store(P_ptrs, q.to(tl.float16))
#     # tl.static_print("static loaded q")
#     q = (q * qk_scale).to(tl.float16)

#     last_nnz_id = -1

#     for nnz_id in range(NNZ):
#         present_nnz_id = tl.load(lut + lut_offset + start_m * stride_lh + nnz_id * stride_lx)
#         start_n = present_nnz_id * BLOCK_N
#         start_n = tl.multiple_of(start_n, BLOCK_N)
#         present_nnz_id = present_nnz_id.to(tl.int32)

#         # tl.static_print("static loading k and start_n is", start_n)
#         k = tl.load(K_ptrs + BLOCK_N * start_n * stride_kn, mask=(start_n * BLOCK_N + offs_n[None, :]) < N_CTX, other=0.0)
#         # tl.static_print("static loaded k")

#         qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
#         qk += tl.dot(q, k)
#         # if off_hz == 1:
#         #     tl.store(P_ptrs, qk)
#         qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
#         qk = tl.where((offs_m[:, None] < N_CTX) & ((start_n + offs_n)[None, :] < N_CTX), qk, float("-inf"))
#         if off_hz == 1:
#             tl.store(P_ptrs, qk)

#         m_ij = tl.max(qk, 1)
#         # p = tl.math.exp2(qk - m_ij[:, None])
#         p = tl.exp((qk - m_ij[:, None]) * 0.6931471805599453)
#         # if off_hz == 1:
#         #     tl.store(P_ptrs, p)
#         p = tl.where(m_ij[:, None] == tl.full((BLOCK_M, BLOCK_N), float("-inf"), tl.float32), 0.0, tl.exp((qk - m_ij[:, None]) * 0.6931471805599453))
#         # if off_hz == 1:
#         #     tl.store(P_ptrs, p)
#         p = p * (last_nnz_id!=present_nnz_id)

#         l_ij = tl.sum(p, 1)

#         m_i_new = tl.maximum(m_i, m_ij)
#         # alpha = tl.math.exp2(m_i - m_i_new)
#         alpha = tl.exp((m_i - m_i_new) * 0.6931471805599453)
#         # beta = tl.math.exp2(m_ij - m_i_new)
#         beta = tl.exp((m_ij - m_i_new) * 0.6931471805599453)
#         l_i *= alpha
#         l_i_new = l_i + beta * l_ij

#         p_scale = beta / l_i_new
#         p = p * p_scale[:, None]
#         # tl.static_print(p)
#         # if off_hz == 1:
#         #     tl.store(P_ptrs, p)

#         acc_scale = l_i / l_i_new
#         acc = acc * acc_scale[:, None]

#         # tl.static_print("static loading v")
#         v = tl.load(V_ptrs + BLOCK_N * start_n * stride_vk, mask=(start_n * BLOCK_N + tx[:, None]) < N_CTX, other=0.0)
#         # tl.static_print("static loaded v")
#         p = p.to(tl.float16)
#         acc += tl.dot(p, v)

#         l_i = l_i_new
#         m_i = m_i_new
#         last_nnz_id = present_nnz_id

#     O_mask = offs_m[:, None] < N_CTX
#     tl.store(O_ptrs, acc.to(tl.float16), mask=O_mask)
#     # tl.store(Q_ptrs, acc.to(tl.float16))

@triton.jit
def _sparse_attention_prefill_fwd_kernel(
    Q, K, V, sm_scale, 
    Out,
    lut, p_ref, acc_ref,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kk, stride_kn,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_lz, stride_lh, stride_lx,
    Z, H, N_CTX, 
    NNZ: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # tl.static_print("static entered kernel")
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    lut_indicator = tl.program_id(1) % H
    qvk_offset = off_hz * stride_qh
    lut_offset = lut_indicator * stride_lz

    tx = tl.arange(0, BLOCK_M)
    ty = tl.arange(0, BLOCK_N)
    tk = tl.arange(0, BLOCK_DMODEL)
    Q_ptrs = Q + qvk_offset + (start_m * BLOCK_M) * stride_qm + (tx[:, None] * stride_qm + tk[None, :] * stride_qk)
    K_ptrs = K + qvk_offset + (tk[:, None] * stride_kk + ty[None, :] * stride_kn)
    V_ptrs = V + qvk_offset + (ty[:, None] * stride_vk + tk[None, :] * stride_vn)
    O_ptrs = Out + qvk_offset + (start_m * BLOCK_M) * stride_om + (tx[:, None] * stride_om + tk[None, :] * stride_on)
    P_ptrs = p_ref + (tx[:, None] * 64 + ty[None, :] * 1)
    ACC_Ptrs = acc_ref + (tx[:, None] * 64 + ty[None, :] * 1)

    offs_m = start_m * BLOCK_M + tx
    offs_n = ty

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504

    # tl.static_print("static loading q")
    q = tl.load(Q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    # if off_hz == 1:
    #     tl.store(P_ptrs, q.to(tl.float16))
    # tl.static_print("static loaded q")
    q = (q * qk_scale).to(tl.float16)

    last_nnz_id = -1

    for nnz_id in range(NNZ):
        present_nnz_id = tl.load(lut + lut_offset + start_m * stride_lh + nnz_id * stride_lx)
        start_n = present_nnz_id * BLOCK_N
        start_n = tl.multiple_of(start_n, BLOCK_N)
        present_nnz_id = present_nnz_id.to(tl.int32)

        # tl.static_print("static loading k and start_n is", start_n)
        k = tl.load(K_ptrs + start_n * stride_kn, mask=(start_n + offs_n[None, :]) < N_CTX, other=0.0)
        # tl.static_print("static loaded k")

        # qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # qk += tl.dot(q, k)
        qk = tl.dot(q,k)
        # if off_hz == 1:
        #     tl.store(P_ptrs, qk)
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]) and offs_m[:, None] < N_CTX, qk, float("-inf"))
        # qk = tl.where(offs_m[:, None] < N_CTX, qk, float("-inf"))
        if off_hz == 1:
            tl.store(P_ptrs, qk)

        m_ij = tl.max(qk, 1)
        # p = tl.math.exp2(qk - m_ij[:, None])
        p = tl.exp((qk - m_ij[:, None]) * 0.6931471805599453)
        # if off_hz == 1:
        #     tl.store(P_ptrs, p)
        p = tl.where(m_ij[:, None] == tl.full((BLOCK_M, BLOCK_N), float("-inf"), tl.float32), 0.0, tl.exp((qk - m_ij[:, None]) * 0.6931471805599453))
        # if off_hz == 1:
        #     tl.store(P_ptrs, p)
        p = p * (last_nnz_id!=present_nnz_id)

        l_ij = tl.sum(p, 1)

        m_i_new = tl.maximum(m_i, m_ij)
        # alpha = tl.math.exp2(m_i - m_i_new)
        alpha = tl.exp((m_i - m_i_new) * 0.6931471805599453)
        # beta = tl.math.exp2(m_ij - m_i_new)
        beta = tl.exp((m_ij - m_i_new) * 0.6931471805599453)
        l_i *= alpha
        l_i_new = l_i + beta * l_ij

        # p_scale = beta / l_i_new
        # p = p * p_scale[:, None]
        # tl.static_print(p)
        # if off_hz == 1:
        #     tl.store(P_ptrs, p)

        acc_scale = l_i / l_i_new
        acc = acc * acc_scale[:, None]

        # tl.static_print("static loading v")
        v = tl.load(V_ptrs + start_n * stride_vk, mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)
        # tl.static_print("static loaded v")
        p = p.to(tl.float16)
        # if off_hz == 1:
        #     tl.store(P_ptrs, v)
        # if off_hz == 1:
        #     tl.store(ACC_Ptrs, acc)
        acc += tl.dot(p, v)
        if off_hz == 0 and nnz_id == 0:
            tl.store(ACC_Ptrs, acc)

        l_i = l_i_new
        m_i = m_i_new
        last_nnz_id = present_nnz_id

    O_mask = offs_m[:, None] < N_CTX
    if off_hz == 0 and start_m == 0:
        tl.store(O_ptrs, acc.to(tl.float16), mask=O_mask)


def forward(q, k, v, sm_scale, lut, p_ref, acc_ref, BLOCK_M: int = 64, BLOCK_N: int = 64) -> torch.Tensor:
    dtype = q.dtype
    assert dtype == torch.float16

    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    k = k.transpose(2, 3)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    o = torch.zeros_like(q)
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)

    NNZ = min(lut.shape[-1], math.ceil(q.shape[2] / BLOCK_N))
    print("nnz", NNZ)
    num_warps = 4 if Lk <= 64 else 8
    num_stages = 4 if BLOCK_M <= 32 else 2

    print("entering kernel")
    print(f"grid is {grid}")

    log_2 = 0.6931471805599453

    print(k.stride(0), k.stride(1), k.stride(2), k.stride(3))

    _sparse_attention_prefill_fwd_kernel[grid](
        q, k, v, sm_scale, 
        o, 
        lut, p_ref, acc_ref,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3), 
        k.stride(0), k.stride(1), k.stride(2), k.stride(3), 
        v.stride(0), v.stride(1), v.stride(2), v.stride(3), 
        o.stride(0), o.stride(1), o.stride(2), o.stride(3), 
        lut.stride(0), lut.stride(1), lut.stride(2), 
        q.shape[0], q.shape[1], q.shape[2], NNZ, 
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=Lk, BLOCK_N=BLOCK_N,
        num_warps=num_warps, 
        num_stages=num_stages
    )

    print("leaving kernel")
    return o

BS, NUM_HEAD, SEQ_LEN, DIM = 1, 32, 62, 64

torch.manual_seed(0)

# q = torch.randn((BS, NUM_HEAD, SEQ_LEN, DIM), dtype=torch.float16).to("cuda")
# k = torch.randn((BS, NUM_HEAD, SEQ_LEN, DIM), dtype=torch.float16).to("cuda")
# v = torch.randn((BS, NUM_HEAD, SEQ_LEN, DIM), dtype=torch.float16).to("cuda")
q = torch.ones((BS, NUM_HEAD, SEQ_LEN, DIM), dtype=torch.float16).to("cuda")
k = torch.ones((BS, NUM_HEAD, SEQ_LEN, DIM), dtype=torch.float16).to("cuda")
v = torch.ones((BS, NUM_HEAD, SEQ_LEN, DIM), dtype=torch.float16).to("cuda")
lut = torch.load("lut.pth").to("cuda")
sm_scale = torch.load("sm_scale.pth")
qk_scale = sm_scale * 1.44269504
p = torch.zeros((64, 64), dtype=torch.float16).to("cuda")
acc = torch.zeros((64, 64), dtype=torch.float16).to("cuda")

out = forward(q, k, v, sm_scale, lut, p, acc)

k = k.transpose(2, 3)
p_ref = torch.matmul(q, k) * qk_scale # bs, num_head, seq_len, seq_len
p_ref = torch.where(torch.tril(torch.ones_like(p_ref)) == 1, p_ref, -torch.inf)

s = torch.softmax(p_ref, -1)
ref = torch.matmul(s, v)

# print("v: ", v[0][1][0])
# torch.set_printoptions(profile="full")
# print(acc.shape, acc)
# torch.set_printoptions(profile="default")
print(acc.shape, acc)
# print(p)
# p_ref = q[0][1].reshape((6, 128))
# p_ref = p_ref[0][1].reshape((62, 62))
# # s = s[0][1].reshape((6, 6))

# for i in range(6):
#     print(i, p[i])
#     print(i, p_ref[i])

# mask = torch.isnan(out)
# idx = torch.nonzero(mask)
# print(idx.shape)
# if (idx.shape[0] != 0):
#     print("-"*70)
#     bsz, head, seq_len, hidden_dim = idx[0]
#     print(idx[0], ref[bsz][head][seq_len][hidden_dim].item(), out[bsz][head][seq_len][hidden_dim].item())

print(out.shape)
print(ref.shape)
print(abs(ref - out).max().item())
torch.set_printoptions(profile="full")
print(out[0][0])
torch.set_printoptions(profile="default")
# print(out[0][0])
print(ref[0][0])




