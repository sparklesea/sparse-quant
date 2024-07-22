import torch
from torch import nn
from datasets import load_dataset
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.rwkv.modeling_rwkv import RwkvForCausalLM
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer


def get_module_by_name_suffix(model, module_name: str):
    for name, module in model.named_modules():
        if name.endswith(module_name):
            return module


def get_op_by_name(module, op_name):
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")


def set_op_by_name(layer, name, new_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_op_name(module, op):
    # get the name of the op relative to the module
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f"Cannot find op {op} in module {module}")


def append_str_prefix(x, prefix):
    if isinstance(x, str):
        return prefix + x
    elif isinstance(x, tuple):
        return tuple([append_str_prefix(y, prefix) for y in x])
    elif isinstance(x, list):
        return [append_str_prefix(y, prefix) for y in x]
    else:
        return x


def get_blocks(model):
    if model.__class__.__name__ == "LlamaForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        # layers = [model.model.layers, model.model.vision_tower.vision_tower.vision_model.encoder.layers]
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    elif isinstance(model, RwkvForCausalLM):
        layers = model.rwkv.blocks
    else:
        raise NotImplementedError(type(model))
    return layers


def move_embed(model, device):
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(device)
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    elif "bigcode" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.wpe = model.transformer.wpe.to(device)
        model.transformer.drop = model.transformer.drop.to(device)
    elif "neox" in str(model.__class__).lower():
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)
        model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.to(device)
        model.embed_out = model.embed_out.to(device)
    else:
        raise NotImplementedError(type(model))


def build_model_and_enc(model_path, use_flash_attn, kv_bit=16, kv_group_size=128, quantized=False):
    print(f"* Building model {model_path}")

    # weither trust remote code
    if "chatglm" in model_path or "mpt" in model_path or "world" in model_path:
        trust_remote_code = True
    else:
        trust_remote_code = False

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if use_flash_attn and "chatglm" not in model_path and "mpt" not in model_path:
        config._flash_attn_2_enabled = True
        config._attn_implementation = "flash_attention_2"
    elif use_flash_attn and "mpt" in model_path:
        config.attn_config["attn_impl"] = "triton"
    else:
        config._flash_attn_2_enabled = False
        config._attn_implementation = None

    # add the kv quantization parameters
    config.kv_bit = kv_bit
    config.kv_group_size = kv_group_size

    # load tokenizer
    if "mpt" in model_path or "rwkv" in model_path:
        use_fast = True
    else:
        use_fast = False
    enc = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast, trust_remote_code=trust_remote_code)

    # load model
    kwargs = {"torch_dtype": torch.float16, "device_map": "balanced"}
    if "opt" in model_path.lower():
        if quantized:
            from module.opt.modeling_opt_ours import OPTForCausalLM
            print("load quantized model")
        else:
            from module.opt.modeling_opt import OPTForCausalLM

        model = OPTForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=trust_remote_code, **kwargs)
    elif "llama" in model_path.lower():
        if quantized:
            from module.llama.modeling_llama_ours import LlamaForCausalLM
            print("load quantized model")
        else:
            from module.llama.modeling_llama import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=trust_remote_code, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=trust_remote_code, **kwargs)
    return model, enc
