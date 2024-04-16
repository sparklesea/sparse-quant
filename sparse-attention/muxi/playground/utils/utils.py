import torch
from torch import Tensor
import warnings

def check_tensor_sanity(tensor: Tensor) -> Tensor:
    """
    Check if a tensor contains NaN or Inf values.
    """
    # Check for NaN values
    if torch.isnan(tensor).any():
        warnings.warn("The tensor contains NaN values.")

    # Check for Inf values
    if torch.isinf(tensor).any():
        warnings.warn("The tensor contains Inf values.")