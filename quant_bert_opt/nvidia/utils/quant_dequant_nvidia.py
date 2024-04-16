import torch
from inficom import transpose_merge_zeros_scales, convert_ours_to_awq, convert_awq_to_lmdeploy

# W4A16 reformat quant param
def quant_lmdeploy(weight, input_features, output_features, group_size, n_bit):

    qweight, zeros, scales, w_fp16 = generate_quant(weight, input_features, n_bit, group_size)

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
    
    if n_bit == 2:
        qweight_lmdeploy = convert_4bit_to_2bit(qweight_lmdeploy)

    return qweight_lmdeploy, zeros_scales, w_fp16

def convert_4bit_to_2bit(qweight_4bit):
    origin_shape = qweight_4bit.shape
    qweight_2bit = torch.zeros((origin_shape[0] * origin_shape[1] // 2), dtype=torch.int, device=qweight_4bit.device)
    qweight_4bit_flatten = qweight_4bit.view(-1, 2)

    for i in range(1, 9):
        qweight_2bit[:] += (qweight_4bit_flatten[:, 0] & (0x30000000 >> 4 * (i - 1)) << 2 * i)
        qweight_2bit[:] += (qweight_4bit_flatten[:, 1] & (0x30000000 >> 4 * (i - 1)) >> 16 - 2 * i)

    return qweight_2bit.reshape(origin_shape[0], origin_shape[1] // 2)

def convert_2bit_to_4bit(qweight_2bit):
    origin_shape = qweight_2bit.shape
    qweight_4bit = torch.zeros((origin_shape[0] * origin_shape[1], 2), dtype=torch.int, device=qweight_2bit.device)
    qweight_2bit_flatten = qweight_2bit.view(-1)

    for i in range(1, 9):
        qweight_4bit[:, 0] += (qweight_2bit_flatten[:] & (0xC0000000 >> 2 * (i - 1)) >> 2 * i)
        qweight_4bit[:, 1] += (qweight_2bit_flatten[:] & (0x0000C000 >> 2 * (i - 1)) << 16 - 2 * i)

    return qweight_4bit.reshape(origin_shape[0], origin_shape[1] * 2)

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

def write_2bit_tensor(w_q):
    w_q_org_shape = w_q.shape
    w_q = w_q.type(torch.uint8).reshape(-1, 4)
    w_int2 = torch.zeros(w_q.shape[0], 1, dtype=torch.uint8, device=w_q.device)

    w_int2[:, 0] = w_q[:, 0] << 6
    w_int2[:, 0] += (w_q[:, 1] << 4)
    w_int2[:, 0] += (w_q[:, 2] << 2)
    w_int2[:, 0] += w_q[:, 3]

    new_shape = w_q_org_shape[:-1] + (w_q_org_shape[-1] * 2 // 8,)
    return w_int2.reshape(new_shape)

def generate_quant(weight_fp16, input_features, n_bit, group_size):
    assert (n_bit == 4 or n_bit == 2)

    # quantize weight
    intweight = []
    zeros = []
    scales = []

    qweight_2bit, scales_temp, zeros_temp, w_fp16 = quantize_tensor(weight_fp16, n_bit, group_size)

    intweight.append(write_4bit_tensor(qweight_2bit))

    scales.append(scales_temp)
    zeros.append(zeros_temp)

    scales = torch.cat(scales, dim=1).clone().half()
    qweight = torch.cat(intweight, dim=1).clone()
    zeros = torch.cat(zeros, dim=1).clone().half()

    return qweight, zeros, scales, w_fp16

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