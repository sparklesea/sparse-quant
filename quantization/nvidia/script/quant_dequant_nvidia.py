import torch
from inficom import transpose_merge_zeros_scales, convert_ours_to_awq, convert_awq_to_lmdeploy

# W4A16 reformat quant param
def quant_lmdeploy(weight, input_features, output_features, group_size):
    qweight, zeros, scales, w_fp16 = generate_quant(weight, input_features, 4, group_size)

    zeros_scales = torch.empty((input_features // group_size, output_features * 2), dtype=torch.float16, device="cuda")
    transpose_merge_zeros_scales(zeros, 
                                 scales,
                                 zeros_scales, 
                                 input_features,
                                 output_features,
                                 group_size)

    qweight_awq = torch.empty((input_features, output_features // 2), dtype=torch.uint8, device="cuda")
    convert_ours_to_awq(qweight, 
                        qweight_awq, 
                        input_features, 
                        output_features)

    qweight_lmdeploy = torch.empty((input_features, output_features // 8), dtype=torch.int, device="cuda")
    convert_awq_to_lmdeploy(qweight_awq,
                            qweight_lmdeploy,
                            input_features,
                            output_features)

    return qweight_lmdeploy, zeros_scales, w_fp16

def quantize_tensor(w, n_bit, q_group_size):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2

    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2 ** n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    w = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)
    w_fp16 = (w - zeros) * scales
    w = w.reshape(org_w_shape)
    w_fp16 = w_fp16.reshape(org_w_shape)

    scales = scales.reshape(org_w_shape[0], -1)
    zeros = zeros.reshape(org_w_shape[0], -1)
    return w, scales, zeros, w_fp16

def write_4bit_tensor(w_q):
    w_q_org_shape = w_q.shape
    w_q = w_q.type(torch.uint8).reshape(-1, 2)
    w_int4 = torch.zeros(w_q.shape[0], 1, dtype=torch.uint8, device=w_q.device)

    w_int4[:, 0] = w_q[:, 0] << 4
    w_int4[:, 0] += w_q[:, 1]

    new_shape = w_q_org_shape[:-1] + (w_q_org_shape[-1] * 4 // 8,)
    return w_int4.reshape(new_shape)


def generate_quant(weight_fp16, input_features, n_bit, group_size):
    assert (n_bit == 4 or n_bit == 8), "Quant bit must be 4 or 8"

    # quantize weight
    intweight = []
    zeros = []
    scales = []

    qweight_temp, scales_temp, zeros_temp, w_fp16 = quantize_tensor(weight_fp16, n_bit, group_size)

    if n_bit == 4:
        intweight.append(write_4bit_tensor(qweight_temp))
    else:
        # 8 bit
        intweight.append(qweight_temp.type(torch.uint8))

    scales.append(scales_temp)
    zeros.append(zeros_temp)

    scales = torch.cat(scales, dim=1).clone().half()
    qweight = torch.cat(intweight, dim=1).clone()
    zeros = torch.cat(zeros, dim=1).clone().half()

    return qweight, zeros, scales, w_fp16


import numpy as np

DEV = torch.device('cuda:0')

def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single

_perm, _scale_perm, _scale_perm_single = _get_perms()


def marlin_pack(w, scales, zeros, k, n, group_size, quant_w=None):
    tile = 16
    maxq = 2 ** 4 - 1
    s = scales.t() # [k // group_size, n]
    z = zeros.t()  # [k // group_size, n]

    if group_size != k:
        w = w.reshape((-1, group_size, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((group_size, -1))
        s = s.reshape((1, -1))
        # AWQ
        z = z.reshape((1, -1))

    w = torch.round(w / s)
    w += z
    w = torch.clamp(w, 0, maxq).int()

    if group_size != k:
        w = w.reshape((group_size, -1, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((k, n)).contiguous()
        s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        # AWQ
        z = z.reshape((-1, len(_scale_perm)))[:, _scale_perm]
    else:
        s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
        # AWQ
        z = z.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
    s = s.reshape((-1, n)).contiguous()
    # AWQ
    z = z.reshape((-1, n)).contiguous()

    # use quant weight directly
    w = quant_w.int().clone()
    w = w.reshape((k // tile, tile, n // tile, tile))

    # k // tile, n // tile, 16, 16
    # int32, k x n
    w = w.permute((0, 2, 1, 3))
    w = w.reshape((k // tile, n * tile))
    res = w

    res = res.reshape((-1, _perm.numel()))[:, _perm].reshape(res.shape)
    q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
    res = res.cpu().numpy().astype(np.uint32)
    for i in range(8):
        q |= res[:, i::8] << 4 * i
    q = torch.from_numpy(q.astype(np.int32)).to(w.device)

    return q, s, z

def quantize_tensor(w, n_bit, q_group_size):
    # N, K
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2

    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2 ** n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    w = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)
    w_fp16 = (w - zeros) * scales

    w = w.reshape(org_w_shape)
    w_fp16 = w_fp16.reshape(org_w_shape)

    scales = scales.reshape(org_w_shape[0], -1).contiguous()
    zeros = zeros.reshape(org_w_shape[0], -1).contiguous()

    return w, scales, zeros, w_fp16

def quant_marlin(w, k, n, groupsize=-1):
    tile = 16
    maxq = 2 ** 4 - 1
    # w = torch.randn((k, n), dtype=torch.half, device=DEV)
    w_bak = w.clone()

    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))   # [K // gs, gs, N]
        w = w.permute(1, 0, 2)              # [gs, K // gs, N] (permuted)
        w = w.reshape((groupsize, -1))      # [gs, K // gs * N]

    w_awq, scales_awq, zeros_awq, w_awq_fp16 = quantize_tensor(w_bak.t(), 4, groupsize if groupsize != -1 else k)

    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    ref = (w - (maxq + 1) // 2).half() * s
    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((k, n)).contiguous()
            return w
        ref = reshape(ref)
        w = reshape(w)
    s = s.reshape((-1, n)).contiguous()

    ##############
    # awq
    if groupsize == -1:
        groupsize = k

    pack_b, pack_s, pack_z = marlin_pack(w_awq_fp16.t(), scales_awq, zeros_awq, k, n, groupsize, w_awq.t().int())
    pack_z -= 8

    zeros_scales = torch.cat((pack_z, pack_s), dim=0).contiguous()
    return w_awq_fp16.t(), pack_b, zeros_scales
