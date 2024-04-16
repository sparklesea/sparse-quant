# coding=utf-8
# Align with transformer==4.36.2
"""PyTorch BERT model."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import add_start_docstrings_to_model_forward, logging


from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from types import MethodType


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BertConfig"


# begin sparse attention mask
def BertEncoder_static_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = False,
    output_hidden_states: Optional[bool] = False,
    return_dict: Optional[bool] = True,
) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = (
        () if output_attentions and self.config.add_cross_attention else None
    )

    batch_size, seq_length = hidden_states.shape[:2]
    ori_attention_mask = attention_mask

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    past_key_values_length = 0
    if use_cache:
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    next_decoder_cache = () if use_cache else None
    for i, layer_module in enumerate(self.layer):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_head_mask = head_mask[i] if head_mask is not None else None
        past_key_value = past_key_values[i] if past_key_values is not None else None

        ##### start: support sparse static attention #####
        if seq_length == 1:
            # decode
            # TODO: support sparse for decode
            pass
        else:
            # prefill
            attention_mask = (
                layer_module.static_attention_mask[
                    :,
                    past_key_values_length : (seq_length + past_key_values_length),
                    : (seq_length + past_key_values_length),
                ][None, :, :, :].expand(
                    batch_size, -1, -1, -1
                )  # + (ori_attention_mask != 0)
            ) * torch.finfo(hidden_states.dtype).min
        # mask is not of shape (batch, 1, query, key) but (batch, head, query, key), which will trigger ValueError in LlamaAttention, but it can be ignored
        ##### end: support sparse static attention #####
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                layer_module.__call__,
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
        else:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache += (layer_outputs[-1],)
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_decoder_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )


def BertAttention_static_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor]:
    mixed_query_layer = self.query(hidden_states)

    # If this is instantiated as a cross-attention module, the keys
    # and values come from an encoder; the attention mask needs to be
    # such that the encoder's padding tokens are not attended to.
    is_cross_attention = encoder_hidden_states is not None

    bsz, q_len, _ = hidden_states.size()

    if is_cross_attention and past_key_value is not None:
        # reuse k,v, cross_attentions
        key_layer = past_key_value[0]
        value_layer = past_key_value[1]
        attention_mask = encoder_attention_mask
    elif is_cross_attention:
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        attention_mask = encoder_attention_mask
    elif past_key_value is not None:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
        value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
    else:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

    query_layer = self.transpose_for_scores(mixed_query_layer)

    use_cache = past_key_value is not None
    if self.is_decoder:
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value = (key_layer, value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    query_length, key_length = query_layer.shape[2], key_layer.shape[2]

    if (
        self.position_embedding_type == "relative_key"
        or self.position_embedding_type == "relative_key_query"
    ):
        query_length, key_length = query_layer.shape[2], key_layer.shape[2]
        if use_cache:
            position_ids_l = torch.tensor(
                key_length - 1, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
        else:
            position_ids_l = torch.arange(
                query_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
        position_ids_r = torch.arange(
            key_length, dtype=torch.long, device=hidden_states.device
        ).view(1, -1)
        distance = position_ids_l - position_ids_r

        positional_embedding = self.distance_embedding(
            distance + self.max_position_embeddings - 1
        )
        positional_embedding = positional_embedding.to(
            dtype=query_layer.dtype
        )  # fp16 compatibility

        if self.position_embedding_type == "relative_key":
            relative_position_scores = torch.einsum(
                "bhld,lrd->bhlr", query_layer, positional_embedding
            )
            attention_scores = attention_scores + relative_position_scores
        elif self.position_embedding_type == "relative_key_query":
            relative_position_scores_query = torch.einsum(
                "bhld,lrd->bhlr", query_layer, positional_embedding
            )
            relative_position_scores_key = torch.einsum(
                "bhrd,lrd->bhlr", key_layer, positional_embedding
            )
            attention_scores = (
                attention_scores
                + relative_position_scores_query
                + relative_position_scores_key
            )

    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        ##### start static attention #####
        if (attention_mask.size() != (bsz, 1, query_length, key_length)) and (
            attention_mask.size()
            != (bsz, self.num_attention_heads, query_length, key_length)
        ):
            ##### end static attention #####
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, query_length, key_length)}, but is {attention_mask.size()}"
            )

        attention_scores = attention_scores + attention_mask

    # Normalize the attention scores to probabilities.
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.dropout(attention_probs)

    # Mask heads if we want to
    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    outputs = (
        (context_layer, attention_probs) if output_attentions else (context_layer,)
    )

    if self.is_decoder:
        outputs = outputs + (past_key_value,)
    return outputs


def BertModel_set_static_attention_mask(self, attention_mask_path: str):
    """
    Store the static attention mask in the model
    """
    # load the attention mask from the file
    predefined_attention_mask = torch.load(attention_mask_path)

    num_queries = predefined_attention_mask.size(-2)
    num_keys = predefined_attention_mask.size(-1)

    causal_mask = torch.zeros((num_queries, num_keys))
    mask_cond = torch.arange(causal_mask.size(-1))
    causal_mask.masked_fill_(
        mask_cond < (mask_cond + 1).view(causal_mask.size(-1), 1), 1
    )

    num_layer, num_head = predefined_attention_mask.size()[:2]

    print(
        "density: ",
        torch.sum(
            predefined_attention_mask[:, :, :num_queries, :num_keys],
            dtype=torch.float32,
        )
        / (num_layer * num_head * torch.sum(causal_mask)),
    )

    for layer_index, layer in enumerate(self.encoder.layer):
        layer.register_buffer(
            "static_attention_mask", ~predefined_attention_mask[layer_index]
        )


def BertModel_use_static_attention(self):
    """
    Set the model instance to use head-wise static attention mask
    """
    self.set_static_attention_mask = MethodType(
        BertModel_set_static_attention_mask, self
    )
    self.encoder.forward = MethodType(
        BertEncoder_static_attention_forward, self.encoder
    )
    for layer in self.encoder.layer:
        layer.attention.self.forward = MethodType(
            BertAttention_static_attention_forward, layer.attention.self
        )


# end sparse attention mask

# # begin sparse attention lut
# from playground.kernels.block_sparse_attention_lut import sparse_attention


# def BertAttention_block_sparse_lut_forward(
#     self,
#     hidden_states: torch.Tensor,
#     attention_mask: Optional[torch.FloatTensor] = None,
#     head_mask: Optional[torch.FloatTensor] = None,
#     encoder_hidden_states: Optional[torch.FloatTensor] = None,
#     encoder_attention_mask: Optional[torch.FloatTensor] = None,
#     past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
#     output_attentions: Optional[bool] = False,
# ) -> Tuple[torch.Tensor]:
#     mixed_query_layer = self.query(hidden_states)

#     # If this is instantiated as a cross-attention module, the keys
#     # and values come from an encoder; the attention mask needs to be
#     # such that the encoder's padding tokens are not attended to.
#     is_cross_attention = encoder_hidden_states is not None

#     if is_cross_attention and past_key_value is not None:
#         # reuse k,v, cross_attentions
#         key_layer = past_key_value[0]
#         value_layer = past_key_value[1]
#         attention_mask = encoder_attention_mask
#     elif is_cross_attention:
#         key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
#         value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
#         attention_mask = encoder_attention_mask
#     elif past_key_value is not None:
#         key_layer = self.transpose_for_scores(self.key(hidden_states))
#         value_layer = self.transpose_for_scores(self.value(hidden_states))
#         key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
#         value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
#     else:
#         key_layer = self.transpose_for_scores(self.key(hidden_states))
#         value_layer = self.transpose_for_scores(self.value(hidden_states))

#     query_layer = self.transpose_for_scores(mixed_query_layer)

#     use_cache = past_key_value is not None
#     if self.is_decoder:
#         # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
#         # Further calls to cross_attention layer can then reuse all cross-attention
#         # key/value_states (first "if" case)
#         # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
#         # all previous decoder key/value_states. Further calls to uni-directional self-attention
#         # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
#         # if encoder bi-directional self-attention `past_key_value` is always `None`
#         past_key_value = (key_layer, value_layer)

#     assert self.position_embedding_type != "relative_key" and self.position_embedding_type != "relative_key_query", "relative attention is not supported"
#     ### efficient attention implementation ###
#     context_layer = self.efficent_attention(
#         # for prefill
#         query_layer,
#         key_layer,
#         value_layer,
#         self.head_dim**-0.5,
#         self.lut,
#         self.block_size,
#         self.block_size,
#         # for decode
#         attention_mask,
#         self.attention_dropout if self.training else 0.0,  # noqa
#     )
#     ### end efficient attention implementation ###

#     context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#     new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#     context_layer = context_layer.view(new_context_layer_shape)

#     # outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
#     assert output_attentions == False, "output_attentions is not supported"
#     outputs = (context_layer,)

#     if self.is_decoder:
#         outputs = outputs + (past_key_value,)
#     return outputs


# def BertDecoderLayer_set_static_attention_lut(self, lut, lut_for_head, block_size: int, device: str = "cuda"):
#     """
#     Set the attention layout of the decoder layer

#     lut: a tuple has 'layer' elements, each element has the size of [lut_num_heads, num_block, nnz]
#     lut_for_head: a tuple has 'layer' elements, each element has the size of [lut_num_heads]
#                   we use it as an indicator when combine heads
#     block_size: int
#     device: str
#     """
#     # self.self_attn.efficent_attention = sparse_attention_prefill
#     self.self_attn.efficent_attention = sparse_attention
#     self.self_attn.lut = lut
#     self.self_attn.lut_for_head = lut_for_head
#     self.self_attn.block_size = block_size


# def BertModel_use_block_sparse_attention_lut(self):
#     """
#     Set the model instance to use efficient attention instead of llama attention
#     """

#     for layer in self.layer:
#         layer.attention.self.forward = MethodType(BertAttention_block_sparse_lut_forward, layer.attention.self)
#         layer.set_static_attention_lut = MethodType(BertDecoderLayer_set_static_attention_lut, layer)


# # end sparse attention lut

# # begin grad analysis
# import deepspeed


# def BertAttention_grad_analysis_forward(
#     self,
#     hidden_states: torch.Tensor,
#     attention_mask: Optional[torch.FloatTensor] = None,
#     head_mask: Optional[torch.FloatTensor] = None,
#     encoder_hidden_states: Optional[torch.FloatTensor] = None,
#     encoder_attention_mask: Optional[torch.FloatTensor] = None,
#     past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
#     output_attentions: Optional[bool] = False,
# ) -> Tuple[torch.Tensor]:
#     mixed_query_layer = self.query(hidden_states)

#     # If this is instantiated as a cross-attention module, the keys
#     # and values come from an encoder; the attention mask needs to be
#     # such that the encoder's padding tokens are not attended to.
#     is_cross_attention = encoder_hidden_states is not None

#     if is_cross_attention and past_key_value is not None:
#         # reuse k,v, cross_attentions
#         key_layer = past_key_value[0]
#         value_layer = past_key_value[1]
#         attention_mask = encoder_attention_mask
#     elif is_cross_attention:
#         key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
#         value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
#         attention_mask = encoder_attention_mask
#     elif past_key_value is not None:
#         key_layer = self.transpose_for_scores(self.key(hidden_states))
#         value_layer = self.transpose_for_scores(self.value(hidden_states))
#         key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
#         value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
#     else:
#         key_layer = self.transpose_for_scores(self.key(hidden_states))
#         value_layer = self.transpose_for_scores(self.value(hidden_states))

#     query_layer = self.transpose_for_scores(mixed_query_layer)

#     use_cache = past_key_value is not None
#     if self.is_decoder:
#         # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
#         # Further calls to cross_attention layer can then reuse all cross-attention
#         # key/value_states (first "if" case)
#         # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
#         # all previous decoder key/value_states. Further calls to uni-directional self-attention
#         # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
#         # if encoder bi-directional self-attention `past_key_value` is always `None`
#         past_key_value = (key_layer, value_layer)

#     # Take the dot product between "query" and "key" to get the raw attention scores.
#     attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

#     if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
#         query_length, key_length = query_layer.shape[2], key_layer.shape[2]
#         if use_cache:
#             position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(-1, 1)
#         else:
#             position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
#         position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
#         distance = position_ids_l - position_ids_r

#         positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
#         positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

#         if self.position_embedding_type == "relative_key":
#             relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
#             attention_scores = attention_scores + relative_position_scores
#         elif self.position_embedding_type == "relative_key_query":
#             relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
#             relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
#             attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

#     attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#     if attention_mask is not None:
#         # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
#         attention_scores = attention_scores + attention_mask

#     # Normalize the attention scores to probabilities.
#     attention_probs = nn.functional.softmax(attention_scores, dim=-1)

#     ### begin grad analysis ###
#     if attention_probs.requires_grad:
#         attention_probs.retain_grad()
#         self.attn_weights = attention_probs
#     ### end grad analysis ###

#     # This is actually dropping out entire tokens to attend to, which might
#     # seem a bit unusual, but is taken from the original Transformer paper.
#     attention_probs = self.dropout(attention_probs)

#     # Mask heads if we want to
#     if head_mask is not None:
#         attention_probs = attention_probs * head_mask

#     context_layer = torch.matmul(attention_probs, value_layer)

#     context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#     new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#     context_layer = context_layer.view(new_context_layer_shape)

#     outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

#     if self.is_decoder:
#         outputs = outputs + (past_key_value,)
#     return outputs


# def BertModel_get_attention_matrix_log(self):
#     """
#     Get the attention matrix gradient
#     """
#     grad_A_list = []
#     mat_A_list = []
#     effect_list = []

#     # effects = grad_A * mat_A
#     # mat_A or grad_A shape: (batch, head, query, key)
#     # print('get attention matrix log')
#     for layer in self.layers:
#         grad_A = layer.self_attn.attn_weights.grad.detach()
#         mat_A = layer.self_attn.attn_weights.detach()
#         multiply = grad_A * mat_A
#         # use the following delta to avoid numerical issue
#         delta = torch.finfo(grad_A.dtype).eps
#         effect = mat_A * (grad_A - torch.sum(multiply, dim=-1, keepdim=True)) / (delta + 1 - mat_A)

#         # deal with the first row of effects
#         effect[..., 0, :] = multiply[..., 0, :]
#         effect = -effect

#         # free the GPU memory
#         effect = effect.cpu()
#         grad_A = grad_A.cpu()
#         mat_A = mat_A.cpu()
#         layer.self_attn.attn_weights.grad = None
#         layer.self_attn.attn_weights = None
#         # torch.cuda.empty_cache()

#         effect_list.append(effect)
#         grad_A_list.append(grad_A)
#         mat_A_list.append(mat_A)

#     # free the CPU memory
#     grad_A = torch.stack(grad_A_list, dim=1)
#     mat_A = torch.stack(mat_A_list, dim=1)
#     effect = torch.stack(effect_list, dim=1)

#     # if the attention matrix is larger than 4k, flush the cache
#     if grad_A.shape[-1] >= 4096:
#         deepspeed.get_accelerator().empty_cache()

#     return {"grad": grad_A, "matrix": mat_A, "sum_effect": effect}


# def BertModel_use_attention_matrix_grad_log(self):
#     """
#     Set the model instance to use flash attention instead of llama attention
#     """
#     self.attention_matrix_log = defaultdict(list)
#     for layer in self.layer:
#         layer.attention.self.forward = MethodType(BertAttention_grad_analysis_forward, layer.attention.self)
#     self.get_attention_matrix_log = MethodType(BertModel_get_attention_matrix_log, self)


# # end grad analysis
