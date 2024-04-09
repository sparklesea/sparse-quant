import torch
from torch import Tensor
import time
from typing import List, Union
import random
import math
from tqdm import tqdm
import os
import random

def gen_causal_pattern(num_query, num_key, dtype: torch.dtype = torch.bool, num_layer: int = 1, num_head: int = 1) -> Tensor:
    """
    generate the causal pattern, represented as torch tensor
    An example of causal pattern of a singe head:
        [[1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1]]
    """
    causal_mask = torch.zeros((num_query, num_key), dtype=torch.bool)
    mask_cond = torch.arange(causal_mask.size(-1))
    causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), 1), True)
    causal_mask = causal_mask[None, None, :, :].expand(num_layer, num_head, num_query, num_key)

    return causal_mask.to(dtype)

def gen_band_pattern(num_query, num_key, band_width: Union[int, Tensor], dtype: torch.dtype = torch.bool, num_layer: int = 1, num_head: int = 1) -> Tensor:
    """
    generate the band pattern, represented as torch tensor
    band_width: a number, a tensor of shape (num_layer), or a tensor of shape (num_layer, num_head)
    An example of band pattern of a singe head with band_width = 3:
        [[1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1]]
    """
    # Adjust the shape of band_width based on its type
    if isinstance(band_width, int):
        band_width = torch.full((num_layer, num_head), band_width, dtype=torch.int64)
    elif band_width.dim() == 1:
        band_width = band_width[:, None].expand(num_layer, num_head)
    else:
        assert band_width.shape == (num_layer, num_head)

    assert torch.all(band_width % 2 == 1)

    # Create a range tensor for num_query and num_key
    query_range = torch.arange(num_query)
    key_range = torch.arange(num_key)

    # Compute the start and end ranges for all layers and heads
    start = query_range[None, None, :, None] - (band_width.view(num_layer, num_head, 1, 1) - 1) // 2
    end = query_range[None, None, :, None] + (band_width.view(num_layer, num_head, 1, 1) - 1) // 2 + 1

    # Create a key_range tensor that broadcasts to the desired shape for comparison
    key_range = key_range[None, None, None, :].expand(num_layer, num_head, num_query, num_key)

    # Compute the band mask using broadcasting and vectorized operations
    band_pattern = (key_range >= start) & (key_range < end)

    return band_pattern.to(dtype)

def gen_global_pattern(num_query: int, num_key: int, key_pos: List[int], dtype: torch.dtype = torch.bool, num_layer: int = 1, num_head: int = 1):
    """
    generate the global pattern, represented as torch tensor
    key_pos: list[int]
    An example of global pattern of a singe head with key_pos = [0, 2, 5]:
        [[1, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 0]]
    for now, only support num_layer=1, num_head=1
    """
    # assert(num_layer == 1 and num_head == 1)
    
    global_mask = torch.zeros((num_layer, num_head, num_query, num_key), dtype=torch.bool)
    global_mask[:, :, :, torch.tensor(key_pos, dtype=int)] = True
    # global_mask.masked_fill_(key_pos[:, :, None, None] == torch.arange(num_key)[None, None, None, :], True)

    return global_mask.to(dtype)

import torch
import random

def gen_row_rand_pattern(num_query: int, num_key: int, num_rand_block_per_row: int, dtype: torch.dtype = torch.bool, num_layer: int = 1, num_head: int = 1):
    """
    Generate the random pattern, represented as a torch tensor. Each row has the same number of random blocks.
    """
    # Initialize the mask with zeros
    rand_mask = torch.zeros((num_layer, num_head, num_query, num_key), dtype=torch.bool)

    # For each layer and head, generate random indices for each query
    for l in range(num_layer):
        for h in range(num_head):
            # Generate random indices for all queries at once
            rand_indices = [random.sample(range(num_key), num_rand_block_per_row) for _ in range(num_query)]
            # Convert the list of lists into a 2D tensor of shape (num_query, num_rand_block_per_row)
            rand_indices_tensor = torch.tensor(rand_indices, dtype=torch.long)

            # Use advanced indexing to set the selected positions to True
            # We expand rand_indices_tensor to match the shape of rand_mask for broadcasting
            rand_mask[l, h, torch.arange(num_query).unsqueeze(1), rand_indices_tensor] = True

    return rand_mask.to(dtype)

def gen_bigbird_pattern(num_query: int, num_key: int, num_band_block: int = 3, num_global_block: int = 2, num_rand_block: int = 3, dtype: torch.dtype = torch.bool, num_layer: int = 1, num_head: int = 1):
    """
    generate the bigbird pattern
    num_query: number of queries
    num_key: number of keys
    num_window_block: number of window block
    num_global_block: number of global block
    num_rand_block: number of random block
    dtype: data type of the mask
    num_layer: number of layers
    num_head: number of heads
        An example of BigBird pattern of a single head with num_query = 8, num_key = 8, num_window_block = 3, num_global_block = 1, num_rand_block = 1:
        [[1, 1, 1, 0, 0, 1, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 0, 0],
        [1, 1, 0, 1, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 1]]
    """

    assert num_query == num_key
    seq_len = num_query

    band_pattern = gen_band_pattern(num_query=num_query, num_key=num_key, band_width=num_band_block, dtype=torch.bool, num_layer=num_layer, num_head=num_head)
    
    global_pattern = gen_global_pattern(num_query=num_query, num_key=num_key, key_pos=[i for i in range(num_global_block)], dtype=torch.bool, num_layer=num_layer, num_head=num_head)
    global_pattern = global_pattern | global_pattern.transpose(-2,-1)

    rand_pattern = gen_row_rand_pattern(num_query=num_query, num_key=num_key, num_rand_block_per_row=num_rand_block, dtype=torch.bool, num_layer=num_layer, num_head=num_head)

    bigbird_pattern = band_pattern | global_pattern | rand_pattern

    return bigbird_pattern.to(dtype)

def gen_block_pattern(layout: torch.Tensor, block_size: int = 64, dtype: torch.dtype = torch.bool):
    """
    generate the block mask, represented as torch tensor
    layout: tensor of shape (num_layer, num_head, num_query_block, num_key_block) or (num_head, num_query_block, num_key_block) or (num_query_block, num_key_block)
    An example of block pattern of a singe head with block_size = 2:
        [[1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1]]
    The corresponding layout is:
        [[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 1]]
    """
    shape = layout.shape
    if layout.ndim == 2:
        layout = layout.unsqueeze(0).unsqueeze(0)
    if layout.ndim == 3:
        layout = layout.unsqueeze(0)
    num_layer, num_head, num_query_block, num_key_block = layout.shape
    block_mask = layout.unsqueeze(-1).unsqueeze(-1).permute(0, 1, 2, 4, 3, 5).repeat(1, 1, 1, block_size, 1, block_size).view(num_layer, num_head, num_query_block * block_size, num_key_block * block_size).contiguous()

    return block_mask.to(dtype)

def layout_to_causal_mask(layout: torch.Tensor, block_size: int, dtype: torch.dtype = torch.bool):
    num_layer, num_head, num_query_block, num_key_block = layout.shape
    num_query = num_query_block * block_size
    num_key = num_key_block * block_size
    block_mask = layout.unsqueeze(-1).unsqueeze(-1).permute(0, 1, 2, 4, 3, 5).repeat(1, 1, 1, block_size, 1, block_size).view(num_layer, num_head, num_query, num_key).contiguous().to(dtype)
    causal_mask = gen_causal_pattern(num_query, num_key, dtype, num_layer, num_head)
    return block_mask & causal_mask
    
def gen_spTrans_masks(mask_save_path: str, num_layers: int = 32, num_heads: int = 32, seq_len: int = 4096, stride: int = 128, c: int = 32, tri=True, dtype: torch.dtype = torch.bool):
    """
    generate the spTrans mask and layout, represented as torch tensor
    args:
        num_layers: number of layers
        num_heads: number of heads
        seq_len: sequence length
        stride: stride
        c: c
        tri: whether to use triangular mask
        dtype: data type of the mask
    note:
        spTrans default config: stride=128, c=32    
    """
    num_queries = seq_len
    num_keys = seq_len
    k = stride//c

    mask = torch.zeros((num_layers, num_heads, num_queries, num_keys), dtype=torch.bool)
    # layout = torch.zeros((num_layers, num_heads, num_queries//c, num_keys//c), dtype=torch.bool)
    causal_mask = torch.zeros((num_queries,num_keys), dtype=torch.bool)
    mask_cond = torch.arange(causal_mask.size(-1))
    causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), 1), True)
    
    # Sparse Transformer (fixed)
    for q in tqdm(range(num_queries)):
        # A1 
        qs = math.floor(q/stride)
        mask[:, :, q, qs*stride:min((qs+1)*stride, q+1)] = True
        # A2
        for ks in range(qs):
            mask[:, :, q, ks*stride+stride-c:(ks+1)*stride] = True

    real_density = mask.sum() / causal_mask.expand_as(mask).sum()
    print("real_density:" + str(real_density))
    
    torch.save(mask, os.path.join(mask_save_path, 'SpTrans_mask_'+str(seq_len)+'_{:.4f}.pt'.format(real_density.item())))
    
if __name__ == "__main__":
    num_layer = 32
    num_head = 32
    num_query = 2048
    num_key = num_query
    band_width = 513
    block_size = 4

    # example of block sparse pattern
    layout = [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1]]
    layout = torch.tensor(layout, dtype=torch.bool).expand(num_layer, num_head, -1, -1)
    causal_block_mask = layout_to_causal_mask(layout, block_size)
    print(causal_block_mask[0][0])

    # start_time = time.time()
    # print("Start generating pattern...")
    # block_pattern = gen_block_pattern(layout=layout, block_size=block_size)
    # print("Time elapsed: ", time.time() - start_time)
    # print(block_pattern.shape)
    # torch.save(block_pattern, "local/data/mask/block_pattern_{}_{}_{}_{}.pt".format(block_pattern.shape[0], block_pattern.shape[1], block_pattern.shape[2], block_pattern.shape[3]))

    # start_time = time.time()
    # print("Start generating pattern...")
    # causal_pattern = gen_causal_pattern(num_query, num_key, num_layer=num_layer, num_head=num_head)
    # pattern = causal_pattern
    # print("Time elapsed: ", time.time() - start_time)
    # print(pattern.shape)
    # torch.save(pattern, "local/causal_pattern_{}_{}_{}_{}.pt".format(num_layer, num_head, num_query, num_key))

    