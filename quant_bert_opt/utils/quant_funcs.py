import torch


@torch.no_grad()
def pseudo_quantize_tensor(tensor, n_bit=8, zero_point=True, group_size=128, granularity="per_group", inplace=False):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    if n_bit == 16:
        return tensor

    org_tensor_shape = tensor.shape
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
    else:
        tensor = (torch.clamp(torch.round(tensor / scales) + zeros, min_int, max_int) - zeros) * scales

    assert torch.isnan(tensor).sum() == 0

    tensor = tensor.reshape(org_tensor_shape)

    return tensor
