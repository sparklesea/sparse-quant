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