import torch

@torch.no_grad()
def pseudo_quantize_tensor(tensor, n_bit=8, zero_point=True, group_size=128, granularity="per_group", inplace=False):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    if n_bit == 16:
        return tensor

    org_tensor_shape = tensor.shape
    # print(org_tensor_shape)

    tensor = tensor.reshape(-1, 4, 2).permute(0, 2, 1).reshape(org_tensor_shape)

    if granularity == "per_group":
        assert org_tensor_shape[-1] % group_size == 0
        tensor = tensor.reshape(-1, group_size)
    elif granularity == "per_token" or granularity == "per channel":
        tensor = tensor.reshape(-1, org_tensor_shape[-1])
    elif granularity == "per_tensor":
        tensor = tensor.reshape(1, -1)
    else:
        raise NotImplementedError
    assert tensor.dim() == 2

    if zero_point:
        max_val = tensor.amax(dim=1, keepdim=True)
        min_val = tensor.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:
        max_val = tensor.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    if inplace:
        # ((tensor.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)).mul_(scales)
        (tensor.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int)
    else:
        tensor = (torch.clamp(torch.round(tensor / scales) + zeros, min_int, max_int) - zeros) * scales

    assert torch.isnan(tensor).sum() == 0

    # tensor = tensor.reshape(org_tensor_shape)

    # return tensor

    # print(tensor.shape)

    tensor = tensor.type(torch.uint8).reshape(-1, 8 // n_bit)

    if n_bit == 4:
        compressed_tensor = torch.zeros(tensor.shape[0], 1, dtype=torch.uint8, device=tensor.device)
        compressed_tensor[:, 0] = tensor[:, 0] << 4
        compressed_tensor[:, 0] += tensor[:, 0]

    elif n_bit == 2:
        compressed_tensor = torch.zeros(tensor.shape[0], 1, dtype=torch.uint8, device=tensor.device)
        compressed_tensor[:, 0] = tensor[:, 3] << 6
        compressed_tensor[:, 0] += (tensor[:, 2] << 4)
        compressed_tensor[:, 0] += (tensor[:, 1] << 2)
        compressed_tensor[:, 0] += tensor[:, 0]

    zeros_scales = torch.stack((zeros, scales)).transpose(0, 1).reshape(org_tensor_shape[0], -1)
    
    return compressed_tensor.reshape(org_tensor_shape[0], org_tensor_shape[1] * n_bit // 8), zeros_scales

# W4A16 reformat quant param
def quant_tianshu(weight, input_features, output_features, n_bit, group_size):

    reformat_weight = weight.reshape(-1, 4, 2).permute(0, 2, 1).reshape(output_features, input_features).clone()
    qweight, zeros_scales = generate_quant_tianshu(reformat_weight, input_features, group_size, n_bit)

    # if n_bit==2:
    #     qweight = convert_2bit_to_4bit(qweight)

    return qweight, zeros_scales

def convert_2bit_to_4bit(qweight_2bit):
    origin_shape = qweight_2bit.shape
    qweight_4bit = torch.empty((origin_shape[0] * origin_shape[1], 2), dtype=torch.uint8, device=qweight_2bit.device)
    qweight_2bit_flatten = qweight_2bit.view(-1)

    qweight_4bit[:, 0] = (qweight_2bit_flatten[:] & 0xC0) >> 2
    qweight_4bit[:, 0] += ((qweight_2bit_flatten[:] & 0x30) >> 4)
    qweight_4bit[:, 1] = (qweight_2bit_flatten[:] & 0x0C) << 2
    qweight_4bit[:, 1] += (qweight_2bit_flatten[:] & 0x03)
    return qweight_4bit.reshape(origin_shape[0], origin_shape[1] * 2)

def quantize_tensor_tianshu(w, n_bit, q_group_size):
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

def write_4bit_tensor_tianshu(w_q):
    w_q_org_shape = w_q.shape
    w_q = w_q.type(torch.uint8).reshape(-1, 2)
    w_int4 = torch.zeros(w_q.shape[0], 1, dtype=torch.uint8, device=w_q.device)

    # w_int4[:, 0] = w_q[:, 0] << 4
    # w_int4[:, 0] += w_q[:, 1]
    w_int4[:, 0] = w_q[:, 1] << 4
    w_int4[:, 0] += w_q[:, 0]

    new_shape = w_q_org_shape[:-1] + (w_q_org_shape[-1] * 4 // 8,)
    return w_int4.reshape(new_shape)

def write_2bit_tensor_tianshu(w_q):
    w_q_org_shape = w_q.shape
    w_q = w_q.type(torch.uint8).reshape(-1, 4)
    w_int4 = torch.zeros(w_q.shape[0], 1, dtype=torch.uint8, device=w_q.device)

    w_int4[:, 0] = w_q[:, 0] << 4
    w_int4[:, 0] += (w_q[:, 1] << 6)
    w_int4[:, 0] += w_q[:, 2]
    w_int4[:, 0] += (w_q[:, 3] << 2)

    new_shape = w_q_org_shape[:-1] + (w_q_org_shape[-1] * 2 // 8,)
    return w_int4.reshape(new_shape)

def generate_quant_tianshu(weight_fp16, input_features, group_size, q_bit):

    # quantize weight
    intweight = []
    zeros = []
    scales = []

    end = input_features
    qweight_temp, scales_temp, zeros_temp = quantize_tensor_tianshu(weight_fp16[:, :end], q_bit, group_size)

    if q_bit==4:
        intweight.append(write_4bit_tensor_tianshu(qweight_temp))
    elif q_bit==2:
        intweight.append(write_2bit_tensor_tianshu(qweight_temp))

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
