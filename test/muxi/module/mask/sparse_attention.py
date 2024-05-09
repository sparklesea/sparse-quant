import torch
from torch.nn import ModuleList
from torch import Tensor
from typing import Union
from transformers import PreTrainedModel
import math
import torch.nn.functional as F


def gen_causal_pattern(
    num_query,
    num_key,
    dtype: torch.dtype = torch.bool,
    num_layer: int = 1,
    num_head: int = 1,
) -> Tensor:
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


class StaticAttentionMask:
    """
    The rule to cut the attention for each layer with static attention mask
    The mask function is called in the forward function of the model, at each layer of the model. For example, the OPTDecoderLayer.
    """
    def __init__(self, attention_mask: torch.BoolTensor, layer_index: int, layer_type: type):

        self.static_attention_mask = attention_mask # size = layer_num, head_num, seq_len, seq_len
        self.layer_index: int = layer_index
        self.layer_type_name = layer_type.__name__

    def __call__(self, attention_mask: torch.BoolTensor = None) -> torch.Tensor:
        """
        The input attention mask is the attention mask of the layer, with shape (num_batch, num_head, num_queries, num_keys). 
        Note that num_queries = num_keys for prefill phase, and num_queries = 1 for decode phase
        """
        if self.layer_type_name == 'OPTDecoderLayer':
            return self.opt_mask(attention_mask)
        elif self.layer_type_name == 'GLMBlock':
            return self.chatglm_mask(attention_mask)
        else:
            raise NotImplementedError
        
    def opt_mask(self, attention_mask: torch.BoolTensor) -> torch.Tensor:
        # use the attention mask to cut the attention
        num_batch = attention_mask.size(0)
        # num_head = attention_mask.size(1) is usually 1, meaning uniform mask for all heads
        num_queries = attention_mask.size(2) # num_queries = num_keys for prefill phase, and num_queries = 1 for decode phase
        num_keys = attention_mask.size(3)
        device = attention_mask.device
        
        dtype = attention_mask.dtype

        assert(dtype == torch.bool)

        defined_attention_mask = self.static_attention_mask[:, :, num_keys-num_queries:num_keys, :num_keys].expand(num_batch, -1, -1, -1).to(dtype).to(device)
        
        return defined_attention_mask

    def chatglm_mask(self, attention_mask: torch.BoolTensor) -> torch.Tensor:
        """
        For chatglm, 1 means masked, 0 means not masked.
        """

        # use the attention mask to cut the attention
        num_batch = attention_mask.size(0)
        # num_head = attention_mask.size(1) is usually 1, meaning uniform mask for all heads
        num_queries = attention_mask.size(2) # num_queries = num_keys for prefill phase, and num_queries = 1 for decode phase
        num_keys = attention_mask.size(3)
        device = attention_mask.device
        
        dtype = attention_mask.dtype

        assert(dtype == torch.bool)

        defined_attention_mask = ~self.static_attention_mask[:, :, num_keys-num_queries:num_keys, :num_keys].expand(num_batch, -1, -1, -1).to(dtype).to(device)

        # make defined_attention_mask symmetric, and masked by attention_mask
        if num_queries == num_keys:
            # prefill phase
            defined_attention_mask = (defined_attention_mask * defined_attention_mask.transpose(-1, -2)) | attention_mask
        
        return defined_attention_mask


class DynamicSkipAttention:
    # the rule to cut the attention for each layer
    def __init__(self, layer_index: int):
        self.layer_index: int = layer_index

        self.topk = 0.2

        self._record_attention_matrix = True
        self.attention_matrix_record = []

    def get_attention_matrix_record(self):
        # return the attention matrix record with shape (num_data, layer_num, head_num, seq_len, seq_len)
        # move to cpu
        attention_matrix_record = [matrix.cpu() for matrix in self.attention_matrix_record]
        # stack the attention matrix
        return torch.stack(attention_matrix_record, dim=0)

    def __call__(self, attention_matrix: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        num_heads = attention_matrix.size(0)
        num_queries = attention_matrix.size(1)
        num_keys = attention_matrix.size(2)

        if attention_mask is None:
            attention_mask = (attention_matrix != 0.0)
        else:
            # select the attention mask for the current layer
            attention_mask = attention_mask[0]
            if attention_mask.size(0) == 1:
                # uniform attention mask for all heads
                attention_mask = attention_mask.expand(num_heads, -1, -1)
            attention_mask = ~(attention_mask).to(bool).expand(attention_matrix.shape)
            # # crop the mask to the same size as the attention matrix
            # attention_mask = self.attention_mask[:, :num_queries, :num_keys]
            if attention_mask.shape != attention_matrix.shape:
                raise ValueError(
                    f"Attention mask shape {attention_mask.shape} should be the same as attention matrix shape {attention_matrix.shape}."
                )

        num_remove_attention_for_query = torch.floor((self.topk) * torch.sum(attention_mask, dim=-1)).to(int).unsqueeze(-1).repeat(1, 1, num_keys) # shape = num_heads, num_queries, num_keys

        matrix_remove = torch.arange(num_keys, device=num_remove_attention_for_query.device).unsqueeze(0).expand(attention_matrix.shape)

        matrix_remove = torch.where(matrix_remove <= num_remove_attention_for_query, 1, 0)

        sorted_values, sorted_indices = torch.sort(attention_matrix, dim=-1, descending=True)

        pruned_sorted_values = sorted_values * matrix_remove

        original_indices = torch.argsort(sorted_indices, dim=-1)

        attention_matrix = torch.gather(pruned_sorted_values, -1, original_indices)

        if self._record_attention_matrix:
            self.attention_matrix_record.append(attention_matrix)

        return attention_matrix

class DynamicAverageAttention:
    # replace the attention head with the average attention
    def __init__(self, layer_index: int, attention_mask_path: str = None):
        self.layer_index: int = layer_index

        self.average_score_record = []
        self.attention_matrix_record = []

        self._record_average_score = False
        self._record_attention_matrix = False

        self.mode = 'predefined' # 'predefined' or 'dynamic'

        if self.mode == 'predefined':
            if attention_mask_path is None:
                raise ValueError("attention_mask_path should not be None when mode is 'predefined'")
            self.average_mask = torch.load(attention_mask_path) # shape (num_layer, num_head)

    def __call__(self, attention_matrix: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        if self.mode == 'predefined':
            return self.predefined_last_k(attention_matrix, attention_mask)
        elif self.mode == 'dynamic':
            return self.dynamic_last_k(attention_matrix, attention_mask)
        else:
            raise ValueError("mode should be 'predefined' or 'dynamic'")

    def predefined_last_k(self, attention_matrix: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        num_batch = attention_mask.size(0)
        num_heads = attention_matrix.size(0) / num_batch
        num_queries = attention_matrix.size(1)
        num_keys = attention_matrix.size(2)

        # make a lower triangle matrix, where each element is 1/n
        average_attention = torch.tril(torch.ones((num_queries, num_keys), dtype=attention_matrix.dtype, device=attention_matrix.device), diagonal=0)
        average_attention = average_attention / torch.sum(average_attention, dim=-1, keepdim=True)

        # replace the attention head with the average attention if the average mask is 1
        # repeat the average mask for batch_size times
        attention_matrix[self.average_mask[self.layer_index].repeat(num_batch)] = average_attention

        return attention_matrix

    def dynamic_last_k(self, attention_matrix: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        num_heads = attention_matrix.size(0)
        num_queries = attention_matrix.size(1)
        num_keys = attention_matrix.size(2)

        # TODO: only support uniform attention mask for all heads

        # make a lower triangle matrix, where each element is 1/n
        average_attention = torch.tril(torch.ones((num_queries, num_keys), dtype=attention_matrix.dtype, device=attention_matrix.device), diagonal=0)
        average_attention = average_attention / torch.sum(average_attention, dim=-1, keepdim=True)

        # use the 2-norm distance between the attention matrix and the attention mask to cut the attention of each head
        attention_score = torch.norm(attention_matrix - average_attention, dim=(1,2), keepdim=False)
        if self._record_average_score:
            self.average_score_record.append(attention_score)

        # replace the last_k attention head with the average attention
        last_k = 0.1
        num_replace_attention_head = math.floor(last_k * num_heads)
        _, indices = torch.topk(attention_score, num_replace_attention_head, dim=-1, largest=False)
        attention_matrix[indices] = average_attention

        if self._record_attention_matrix:
            self.attention_matrix_record.append(attention_matrix)

        return attention_matrix

    def get_average_score_record(self):
        # return the average score record tensor of shape (num_sentence, num_heads)
        # move to cpu
        self.average_score_record = [score.to('cpu') for score in self.average_score_record]
        return torch.stack(self.average_score_record, dim=0)

def set_static_attention_mask(model: PreTrainedModel, attention_mask_path: str):
    """
    Store the static attention mask in the model
    """
    # load the attention mask from the file
    predefined_attention_mask = torch.load(attention_mask_path)
    
    num_queries = predefined_attention_mask.size(-2)
    num_keys = predefined_attention_mask.size(-1)

    causal_mask = torch.zeros((num_queries, num_keys))
    mask_cond = torch.arange(causal_mask.size(-1))
    causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), 1), 1)

    num_layer, num_head = predefined_attention_mask.size()[:2]

    print("density: ", torch.sum(predefined_attention_mask[:,:,:num_queries,:num_keys], dtype=torch.float32) / (num_layer * num_head * torch.sum(causal_mask)))
    
    
    model.model.static_attention_mask = predefined_attention_mask
    return model


def set_static_attention_rule(model: PreTrainedModel, attention_mask_path: str, max_length: int = None, model_layers: ModuleList = None):
    """
    Apply the static attention rule to the model
    """

    # load the attention mask from the file
    predefined_attention_mask = torch.load(attention_mask_path)
    
    # set the attention mask rule for each layer, mask the upper triangle
    num_queries = predefined_attention_mask.size(-2) if max_length is None else max_length
    num_keys = predefined_attention_mask.size(-1) if max_length is None else max_length

    causal_mask = torch.zeros((num_queries, num_keys))
    mask_cond = torch.arange(causal_mask.size(-1))
    causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), 1), 1)

    num_layer, num_head = predefined_attention_mask.size()[:2]

    print("density: ", torch.sum(predefined_attention_mask[:,:,:num_queries,:num_keys], dtype=torch.float32) / (num_layer * num_head * torch.sum(causal_mask)))

    # set the attention mask rule for each layer
    static_attention_rules = []
    for layer_index, layer in enumerate(model_layers):
        static_attention_rule = StaticAttentionMask(predefined_attention_mask[layer_index].unsqueeze(0), layer_index, type(layer))
        static_attention_rules.append(static_attention_rule)
        layer.set_static_attention_mask(static_attention_rule)
    return model, static_attention_rules

def set_static_attention_layout(attention_layout_path: str, max_length: int = None, model_layers: ModuleList = None, block_size: int = 64):
    """
    Apply the static attention rule to the model and add causal mask
    """

    # load the attention mask from the file
    layout = torch.load(attention_layout_path)
    
    # set the attention mask rule for each layer, mask the upper triangle
    num_query_block = layout.size(-2) if max_length is None else max_length // block_size
    num_key_block = layout.size(-1) if max_length is None else max_length // block_size

    causal_mask = torch.zeros((num_query_block, num_key_block))
    mask_cond = torch.arange(causal_mask.size(-1))
    causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), 1), 1)

    num_layer, num_head = layout.size()[:2]

    density = (torch.sum(layout[:,:,:num_query_block,:num_key_block], dtype=torch.float32) / (num_layer * num_head * torch.sum(causal_mask))).cpu().item()
    print("laytout size: {}, density: {}".format(layout.size(), density))

    # set the attention mask rule for each layer
    for layer_index, layer in enumerate(model_layers):
        layer.set_static_attention_layout(layout[layer_index,:,:num_query_block,:num_key_block], block_size)

def set_static_attention_lut(attention_lut_path: str, attention_lut_for_head_path: str, model_layers: ModuleList = None, block_size: int = 64):
    """
    Apply the efficient attention
    """

    # load the attention mask from the file
    lut = torch.load(attention_lut_path)

    # set the attention lut and lut_for_head for each layer
    for layer_index, layer in enumerate(model_layers):
        layer.set_static_attention_lut(lut[layer_index].to('cuda'), None, block_size)

def set_OPT_dynamic_attention_rule(model: PreTrainedModel):
    # set the attention mask rule for each layer
    dynamic_attention_rules = []
    for layer_id, layer in enumerate(model.model.decoder.layers):
        # type(layer) = OPTDecoderLayer
        dynamic_attention_rule = DynamicSkipAttention(layer_id)
        dynamic_attention_rules.append(dynamic_attention_rule)
        layer.set_dynamic_attention_rule(dynamic_attention_rule)
    return model, dynamic_attention_rules

def set_OPT_average_attention_rule(model: PreTrainedModel):
    # set the attention mask rule for each layer
    average_attention_rules = []
    for layer_id, layer in enumerate(model.model.decoder.layers):
        # type(layer) = OPTDecoderLayer
        dynamic_attention_rule = DynamicAverageAttention(layer_id)
        average_attention_rules.append(dynamic_attention_rule)
        layer.set_dynamic_attention_rule(dynamic_attention_rule)
    return model, average_attention_rules

### utils for block sparse attention

def block_sparse_to_dense(sparse_matrix, layout, batch_size, num_heads, token_length, block_size):
    '''
    sparse_matrix: shape: (batch_size, num_non_zero_blocks, block_size, block_size)
    layout: shape: (num_heads, num_blocks, num_blocks)
    '''
    precision = sparse_matrix.dtype
    device = sparse_matrix.device

    layout_flatten = layout.reshape(-1) # shape: (num_heads * num_blocks * num_blocks)
    num_blocks = layout.shape[1]
    # insert zero matrix to sparse matrix
    num_non_zero_blocks = sparse_matrix.shape[1]
    block_fill_index = torch.cumsum(layout_flatten, dim=0) - 1 # shape: (num_heads * num_blocks * num_blocks)
    block_fill_index[layout_flatten==0] = num_non_zero_blocks
    zero_block = torch.zeros((batch_size, 1, block_size, block_size), dtype=precision, device=device)
    # fill in the zero blocks into the sparse matrix based on the layout
    unfold_dense_matrix = torch.cat([sparse_matrix, zero_block], dim=1) # shape: (batch_size, num_non_zero_blocks + num_zero_blocks, block_size, block_size)
    dense_matrix = unfold_dense_matrix[:, block_fill_index] # shape: (batch_size, num_heads * num_blocks * num_blocks, block_size, block_size)

    # reshape the dense matrix to the dense attention weights
    dense_matrix = dense_matrix.view(batch_size, num_heads, num_blocks, num_blocks, block_size, block_size)
    dense_matrix = dense_matrix.permute(0, 1, 2, 4, 3, 5) # shape: (batch_size, num_heads, block_size, num_blocks, block_size, num_blocks)
    dense_matrix = dense_matrix.reshape(batch_size, num_heads, token_length, token_length) # shape: (batch_size, num_heads, token_length, token_length)

    return dense_matrix


def pattern_to_layout(mask: torch.Tensor, block_size: int) -> torch.Tensor:
    r"""
    <from xformers>
    Given a mask pattern and blocksize, return the corresponding layout
    which makes sure that all the positives in the mask are covered
    """
    assert mask.ndim >= 2, "We're expecting [Heads, Seq, Seq] or [Seq, Seq]"
    _should_squeeze = False

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
        _should_squeeze = True

    assert (
        mask.shape[1] % block_size == 0 and mask.shape[2] % block_size == 0
    ), "We're only handling masks divisible by block_size"

    # Now mark the mask
    layout = torch.nn.functional.max_pool2d(
        mask.to(torch.float), kernel_size=block_size, stride=block_size
    )
    layout = layout.to(torch.long)

    if _should_squeeze:
        layout.squeeze_(0)

    return layout

### utils for efficient attention
def layout_to_lut(layout: torch.BoolTensor):
    """
    input:
        layout: (layer, num_heads, num_block, num_block)
    output:
        lut: a tuple has 'layer' elements, each element has the size of (lut_num_heads, num_block, nnz)
        lut_for_head: a tuple has 'layer' elements, each element has the size of (lut_num_heads)
                      we use it as an indicator when combine heads 
        copy_times: (layer, num_heads, 1)
    """
    DeprecationWarning("This function is deprecated. Use layout2lut_single_density instead.")

    assert layout.dim() == 4, "The layout should have 4 dimensions: (layer, num_heads, num_block, num_block)"

    layer = layout.shape[0]
    num_heads = layout.shape[1]
    num_block = layout.shape[2]

    nnz_min_num = torch.full((layer,1), num_block)
    copy_times = torch.empty(layer, num_heads, 1)

    lut = ()
    lut_for_head = ()

    for i in range(layer):
        copy = torch.zeros(num_heads, 1)
        for j in range(num_heads):
            # to find out the head with the most density
            line = layout[i, j, -1, :]
            nnz = torch.sum(line).item()
            if(nnz <= nnz_min_num[i]):
                nnz_min_num[i] = nnz

        cnt = 0
        for j in range(num_heads):
            line = layout[i, j, -1, :]
            nnz = torch.sum(line).item()
            cnt += nnz/nnz_min_num[i]
            copy[j] = nnz/nnz_min_num[i]
            copy_times[i, j] = int(copy[j].item())

        lut_num_heads = cnt
        head_lut = torch.empty((int(lut_num_heads.item()), num_block, nnz_min_num[i]))
        indicator = torch.empty(int(lut_num_heads.item()))

        for j in range(num_heads):
            if(j == 0):
                sum = 0
            else:
                sum = int(torch.sum(copy[:j]).item())
            for k in range(int(copy[j].item())):
                for l in range(num_block):
                    for m in range(nnz_min_num[i].item()):
                        index = k*nnz_min_num[i] + m + 1
                        line = layout[i, j, l, :]
                        nnz_indices = torch.nonzero(line).squeeze()
                        nnz_index = nnz_indices[index-1].item()
                        head_lut[sum+k, l, m] = nnz_index
                indicator[sum+k] = j

        lut = lut + (head_lut.to(torch.int64),)
        lut_for_head = lut_for_head + (indicator.to(torch.int64),)

    return lut, lut_for_head, copy_times.to(torch.int64)

# old implementation using inner loop. deprecated
def layout2lut_single_density_old(layout: torch.BoolTensor):
    """
    input:
        layout: (layer, num_heads, num_block, num_block)
    output:
        lut: a tuple has 'layer' elements, each element has the size of (num_heads, num_block, nnz)
    """
    DeprecationWarning("This function is deprecated. Use layout2lut_single_density instead.")

    assert layout.dim() == 4, "The layout should have 4 dimensions: (layer, num_heads, num_block, num_block)"

    layer = layout.shape[0]
    num_heads = layout.shape[1]
    num_block = layout.shape[2]

    lut = ()

    for i in range(layer):
        max_nnz = 0
        for j in range(num_heads):
            line = layout[i, j, -1, :]
            nnz = torch.sum(line).item()
            if(nnz > max_nnz):
                max_nnz = nnz
        lut_layer = torch.empty((num_heads, num_block, max_nnz))
        for j in range(num_heads):
            for k in range(num_block):
                cnt = 0
                last_nnz = -1
                for l in range(num_block):
                    if(layout[i, j, k, l] == True):
                        lut_layer[j, k, cnt] = l
                        cnt += 1
                        last_nnz = l
                if(cnt < max_nnz):
                    for l in range(cnt, max_nnz):
                        lut_layer[j, k, l] = last_nnz
        lut += (lut_layer.to(torch.int64), )

    return lut


## layout2lut without using inner loop
def layout2lut_single_density(layout: torch.BoolTensor):
    """
    input:
        layout: (layer, num_heads, num_block, num_block)
    output:
        lut: a tuple has 'layer' elements, each element has the size of (num_heads, num_block, nnz)
    """
    assert layout.dim() == 4, "The layout should have 4 dimensions: (layer, num_heads, num_block, num_block)"

    layer = layout.shape[0]
    num_heads = layout.shape[1]
    num_block = layout.shape[2]

    lut = ()

    for i in range(layer):
        layer_mask = layout[i]
        max_nnz = torch.sum(layer_mask, dim=-1).max().cpu().item()

        one_matrix = torch.ones_like(layer_mask, dtype=torch.int, device=layer_mask.device)
        cum_matrix = torch.cumsum(one_matrix, dim=-1)
        masked_cum_matrix = cum_matrix * layer_mask # keep only entries that are True in attention mask. The value of each entry is column index plus one.
        max_matrix = masked_cum_matrix.max(dim=-1, keepdim=True)[0].repeat(1, 1, num_block)
        filled_matrix = masked_cum_matrix.detach().clone()
        filled_matrix[~layer_mask] = max_matrix[~layer_mask] # fill missing entries with largest value in the row.
        lut_layer = torch.sort(filled_matrix, dim=-1)[0] - 1 # make the index start from zero instead of one.

        lut_layer = lut_layer[:, :, :max_nnz]
        lut += (lut_layer.to(torch.int64), )

    return lut


### lut to density ###
def lut_to_density(lut_path: str) -> list:
    """
    input: 
        lut_path: the path to load lut
    output:
        density_list: a list of density for each layer
    """

    density_list = []
    lut = torch.load(lut_path)
    layer = len(lut)
    for i in range (layer):
        N = lut[i].shape[1]
        n = lut[i].shape[2]
        density = (n*(n+1)/2+n*(N-n))/(N*(N+1)/2)
        density_list.append(density)

    return density_list

def layout_to_mask(layout: torch.BoolTensor, block_size: int = 64, is_causal: bool = True) -> torch.BoolTensor:
    """
    Convert the layout to mask
    input:
        layout: (num_layer, num_head, num_block_x, num_block_y) or (num_head, num_block_x, num_block_y)
        block_size: int, the block size
        is_causal: bool, whether the attention is causal
    output:
        mask: (num_layer, num_head, num_block_x*block_size, num_block_y*block_size) or (num_head, num_block_x*block_size, num_block_y*block_size)
    """
    # Check the number of dimensions in layout and adjust accordingly
    dim = layout.dim()
    if dim == 4:
        num_layer, num_head, num_block_x, num_block_y = layout.shape
    elif dim == 3:
        num_layer = 1  # Set num_layer to 1 if it's missing
        num_head, num_block_x, num_block_y = layout.shape
        layout = layout.unsqueeze(0)  # Add a dimension for num_layer
    else:
        raise ValueError("Invalid layout shape. Expected 3 or 4 dimensions, got {}.".format(dim))

    mask = layout.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, block_size, block_size)
    mask = mask.permute(0, 1, 2, 4, 3, 5)
    mask = mask.reshape(num_layer, num_head, num_block_x*block_size, num_block_y*block_size).contiguous()

    if is_causal:
        # Assume gen_causal_pattern is defined elsewhere and handles num_layer correctly
        causal_mask = gen_causal_pattern(num_query=mask.size(-2), num_key=mask.size(-1), dtype=mask.dtype, num_layer=num_layer, num_head=num_head).to(mask.device)
        mask = mask & causal_mask
    
    if dim == 3:
        mask = mask.squeeze(0)

    return mask


### nnz & num_block to density ###
def n_to_density(n: Union[int, Tensor], N: int) -> Union[float, Tensor]:
    """
    input: 
        n: nnz
        N: num_block
    output:
        density: the density of the layout
    """
    density = (n*(n+1)/2+n*(N-n))/(N*(N+1)/2)
    return density

def mask_to_layout(mask: Tensor, block_size: int) -> Tensor:
    '''
    Input:
        mask: Tensor of shape (..., token_length, token_length)
    Output:
        layout: Boolean tensor of shape (..., num_block_x, num_block_y)
    '''
    avg = F.avg_pool2d(mask.float(), block_size)
    return avg > 0.5
