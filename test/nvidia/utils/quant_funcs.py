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
        ((tensor.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)).mul_(scales)
        # (tensor.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int)
    else:
        tensor = (torch.clamp(torch.round(tensor / scales) + zeros, min_int, max_int) - zeros) * scales

    assert torch.isnan(tensor).sum() == 0

    tensor = tensor.reshape(org_tensor_shape)

    return tensor

# @torch.no_grad()
# def pseudo_quantize_tensor(tensor, n_bit=8, zero_point=True, group_size=128, granularity="per_group", inplace=False):
#     """
#     The basic quantization function for weight, activation and KV cache.
#     """
#     if n_bit == 16:
#         return tensor

#     org_tensor_shape = tensor.shape
#     # print(org_tensor_shape)

#     tensor = tensor.reshape(-1, 4, 2).permute(0, 2, 1).reshape(org_tensor_shape)

#     if granularity == "per_group":
#         assert org_tensor_shape[-1] % group_size == 0
#         tensor = tensor.reshape(-1, group_size)
#     elif granularity == "per_token" or granularity == "per channel":
#         tensor = tensor.reshape(-1, org_tensor_shape[-1])
#     elif granularity == "per_tensor":
#         tensor = tensor.reshape(1, -1)
#     else:
#         raise NotImplementedError
#     assert tensor.dim() == 2

#     if zero_point:
#         max_val = tensor.amax(dim=1, keepdim=True)
#         min_val = tensor.amin(dim=1, keepdim=True)
#         max_int = 2**n_bit - 1
#         min_int = 0
#         scales = (max_val - min_val).clamp(min=1e-5) / max_int
#         zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
#     else:
#         max_val = tensor.abs().amax(dim=1, keepdim=True)
#         max_val = max_val.clamp(min=1e-5)
#         max_int = 2 ** (n_bit - 1) - 1
#         min_int = -(2 ** (n_bit - 1))
#         scales = max_val / max_int
#         zeros = 0

#     if inplace:
#         # ((tensor.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)).mul_(scales)
#         (tensor.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int)
#     else:
#         tensor = (torch.clamp(torch.round(tensor / scales) + zeros, min_int, max_int) - zeros) * scales

#     assert torch.isnan(tensor).sum() == 0

#     # tensor = tensor.reshape(org_tensor_shape)

#     # return tensor

#     # print(tensor.shape)

#     tensor = tensor.type(torch.uint8).reshape(-1, 8 // n_bit)

#     if n_bit == 4:
#         compressed_tensor = torch.zeros(tensor.shape[0], 1, dtype=torch.uint8, device=tensor.device)
#         compressed_tensor[:, 0] = tensor[:, 0] << 4
#         compressed_tensor[:, 0] += tensor[:, 1]

#     elif n_bit == 2:
#         compressed_tensor = torch.zeros(tensor.shape[0], 1, dtype=torch.uint8, device=tensor.device)
#         compressed_tensor[:, 0] = tensor[:, 3] << 6
#         compressed_tensor[:, 0] += (tensor[:, 2] << 4)
#         compressed_tensor[:, 0] += (tensor[:, 1] << 2)
#         compressed_tensor[:, 0] += tensor[:, 0]

#     zeros_scales = torch.stack((zeros, scales)).transpose(0, 1).reshape(org_tensor_shape[0], -1)
    
#     return compressed_tensor.reshape(org_tensor_shape[0], org_tensor_shape[1] * n_bit // 8), zeros_scales
