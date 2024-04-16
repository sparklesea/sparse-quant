"""
Fused Attention
===============

This is a Triton implementation of the prefill, decoder attention kernel. It support block sparse.

"""
import math
import torch

import triton
import triton.language as tl

from triton.ops.blocksparse import matmul as blocksparse_matmul  
from triton.ops.blocksparse import softmax as blocksparse_softmax 

"""
prefill kernel (not fused)
"""
def create_block_sparse_attention_kernels(layout, block_size, device):
    # blocksparse operators
    sparse_dot_sdd = blocksparse_matmul(
        layout,
        block_size,
        "sdd",
        trans_a=False,
        trans_b=True,
        device=device,
    )

    sparse_dot_dsd = blocksparse_matmul(
        layout,
        block_size,
        "dsd",
        trans_a=False,
        trans_b=False,
        device=device,
    )

    sparse_softmax = blocksparse_softmax(
        layout,
        block_size,
        device=device,
    )

    return sparse_dot_sdd, sparse_dot_dsd, sparse_softmax
