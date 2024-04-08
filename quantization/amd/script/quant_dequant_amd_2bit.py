import torch

# W4A16 reformat quant param
def quant_amd(weight, input_features, output_features, group_size):
    qweight, zeros_scales = generate_quant_amd(weight, input_features, group_size, 4)
    fp16_weight_dequant = dequant_to_fp16_amd(qweight, zeros_scales, group_size)

    reformat_weight = weight.reshape(-1, 4, 2).permute(0, 2, 1).reshape(output_features, input_features).clone()
    qweight_4bit_ref, zeros_scales_4bit_ref = generate_quant_amd(reformat_weight, input_features, group_size, 4)
    
    reformat_weight = weight.reshape(-1, 4, 2).permute(0, 2, 1).reshape(output_features, input_features).clone()
    qweight_2bit, zeros_scales_2bit = generate_quant_amd(reformat_weight, input_features, group_size, 2)

    qweight_4bit = convert_2bit_to_4bit(qweight_2bit)

    print("qweight_4bit:", qweight_4bit.shape, qweight_4bit)
    print("qweight_2bit:", qweight_2bit.shape, qweight_2bit)

    return qweight_4bit_ref, zeros_scales_4bit_ref, fp16_weight_dequant

def convert_2bit_to_4bit(qweight_2bit):
    origin_shape = qweight_2bit.shape
    qweight_4bit = torch.empty((origin_shape[0] * origin_shape[1], 2), dtype=torch.uint8, device=qweight_2bit.device)
    qweight_2bit_flatten = qweight_2bit.view(-1)

    qweight_4bit[:, 0] = (qweight_2bit_flatten[:] & 0xC0) >> 2
    qweight_4bit[:, 0] += ((qweight_2bit_flatten[:] & 0x30) >> 4)
    qweight_4bit[:, 1] = (qweight_2bit_flatten[:] & 0x0C) << 2
    qweight_4bit[:, 1] += (qweight_2bit_flatten[:] & 0x03)
    return qweight_4bit.reshape(origin_shape[0], origin_shape[1] * 2)

def quantize_tensor_amd(w, n_bit, q_group_size):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size) #在每个group算scale和zero
    assert w.dim() == 2

    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2 ** n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    w = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)
    w = w.reshape(org_w_shape)
    scales = scales.reshape(org_w_shape[0], -1)
    zeros = zeros.reshape(org_w_shape[0], -1)
    return w, scales, zeros

def write_4bit_tensor_amd(w_q):
    w_q_org_shape = w_q.shape
    w_q = w_q.type(torch.uint8).reshape(-1, 2)
    w_int4 = torch.zeros(w_q.shape[0], 1, dtype=torch.uint8, device=w_q.device)

    # w_int4[:, 0] = w_q[:, 0] << 4
    # w_int4[:, 0] += w_q[:, 1]
    w_int4[:, 0] = w_q[:, 1] << 4
    w_int4[:, 0] += w_q[:, 0]

    new_shape = w_q_org_shape[:-1] + (w_q_org_shape[-1] * 4 // 8,)
    return w_int4.reshape(new_shape)

def write_2bit_tensor_amd(w_q):
    w_q_org_shape = w_q.shape
    w_q = w_q.type(torch.uint8).reshape(-1, 4)
    w_int4 = torch.zeros(w_q.shape[0], 1, dtype=torch.uint8, device=w_q.device)

    w_int4[:, 0] = w_q[:, 3] << 6
    w_int4[:, 0] += (w_q[:, 2] << 4)
    w_int4[:, 0] += (w_q[:, 1] << 2)
    w_int4[:, 0] += w_q[:, 0]

    new_shape = w_q_org_shape[:-1] + (w_q_org_shape[-1] * 2 // 8,)
    return w_int4.reshape(new_shape)

# NOTE: real little-endian
def deq_4bit_tensor_amd(w_int4):
    w_int4_org_shape = w_int4.shape
    new_shape = w_int4_org_shape[:-1] + (w_int4_org_shape[-1] * 8 // 4,)
    w_int4 = w_int4.reshape(-1, 1)
    w_q = torch.zeros(w_int4.shape[0], 2, dtype=torch.uint8, device=w_int4.device)

    w_q[:, 0] = (w_int4[:, 0] << 4) >> 4
    w_q[:, 1] = w_int4[:, 0] >> 4

    return w_q.reshape(new_shape)

def generate_quant_amd(weight_fp16, input_features, group_size, q_bit):

    # quantize weight
    intweight = []
    zeros = []
    scales = []

    end = input_features
    qweight_temp, scales_temp, zeros_temp = quantize_tensor_amd(weight_fp16[:, :end], q_bit, group_size)

    if q_bit==4:
        intweight.append(write_4bit_tensor_amd(qweight_temp))
    elif q_bit==2:
        intweight.append(write_2bit_tensor_amd(qweight_temp))

    scales.append(scales_temp)
    zeros.append(zeros_temp)

    scales = torch.cat(scales, dim=1).clone().half()
    qweight = torch.cat(intweight, dim=1).clone()
    zeros = torch.cat(zeros, dim=1).clone().half() + 64

    zeros_1d = zeros.flatten()
    scales_1d = scales.flatten()

    zeros_scales_1d = torch.zeros(zeros_1d.size(0) + scales_1d.size(0), dtype=zeros.dtype, device=zeros.device)

    zeros_scales_1d[0::2] = zeros_1d
    zeros_scales_1d[1::2] = scales_1d

    zeros_scales = zeros_scales_1d.view(zeros.size(0), zeros.size(1) * 2)

    return qweight, zeros_scales

def dequant_to_fp16_amd(qweight, zeros_scales, group_size):
    out_features = qweight.shape[0]
    in_features = qweight.shape[1] * 2

    fp16_weight = deq_4bit_tensor_amd(qweight).reshape(-1, group_size)
    zeros_scales = zeros_scales.reshape(-1, 2)
    zeros = zeros_scales[:, 0].reshape(-1, 1) - 64
    scales = zeros_scales[:, 1].reshape(-1, 1)

    dequant_fp16_weight = (fp16_weight - zeros) * scales

    return dequant_fp16_weight.reshape(out_features, in_features)
