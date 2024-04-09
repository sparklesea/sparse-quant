
from typing import List, Optional, Tuple, Union
import torch

from transformers import OPTModel
from transformers.cache_utils import Cache, DynamicCache

from types import MethodType

"""
efficient llama attention using lut
"""
from playground.kernels.block_sparse_attention_lut import sparse_attention


def OPTAttention_block_sparse_lut_forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)
        
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(-2)

        original_shape = (bsz, self.num_heads, -1, self.head_dim)
        q = query_states.view(*original_shape)
        k = key_states.view(*original_shape)
        v = value_states.view(*original_shape)
        # lut: (self.num_heads, seq_len/BLOCK_M, nnz)
        # give any lut you want here, but each row should have a valid block
        lut = self.lut
        # print(f"q shape is: {q.shape}")
        # print(f"k shape is: {k.shape}")
        # print(f"v shape is: {v.shape}")
        # print(f"lut shape is: {lut.shape}")
        # print(lut[0][31])
        attn_output = self.efficent_attention(q, k, v, self.scaling, lut, self.block_size, self.block_size).half()
        # print(f"attn_output shape is: {attn_output.shape}")

        if not output_attentions:
            attn_weights_reshaped = None
        
        # print(f"attn_output shape is: {attn_output.shape} and bsz={bsz}, tgt_len={tgt_len}, self.embed_dim={self.embed_dim}")
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

    # print(f"q size is: {query_states.shape}")
    # print(f"k size is: {key_states.shape}")
    # print(f"v size is: {value_states.shape}")
    # print(f"lut size is: {self.lut.shape}")
    
        ### efficient attention implementation ###
        attn_output = self.efficent_attention(
            # for prefill
            query_states, 
            key_states, 
            value_states, 
            self.head_dim**-0.5, 
            self.lut, 
            self.block_size, 
            self.block_size,
            # for decode
            attention_mask,
            self.attention_dropout if self.training else 0.0 # noqa
        )
        ### end efficient attention implementation ###



def OPTDecoderLayer_set_static_attention_lut(self, lut, lut_for_head, block_size: int, device: str = "cuda"):
    """
    Set the attention layout of the decoder layer

    lut: a tuple has 'layer' elements, each element has the size of [lut_num_heads, num_block, nnz]
    lut_for_head: a tuple has 'layer' elements, each element has the size of [lut_num_heads]
                  we use it as an indicator when combine heads
    block_size: int
    device: str
    """
    # self.self_attn.efficent_attention = sparse_attention_prefill
    self.self_attn.efficent_attention = sparse_attention
    self.self_attn.lut = lut
    self.self_attn.lut_for_head = lut_for_head
    self.self_attn.block_size = block_size


def OPTModel_use_block_sparse_attention_lut(self):
    """
    Set the model instance to use efficient attention instead of llama attention
    """
    
    for layer in self.layers:
        layer.self_attn.forward = MethodType(OPTAttention_block_sparse_lut_forward, layer.self_attn)
        layer.set_static_attention_lut = MethodType(OPTDecoderLayer_set_static_attention_lut, layer)

