import torch
from torch import nn
from datasets import load_dataset
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.rwkv.modeling_rwkv import RwkvForCausalLM
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
from module.opt.modeling_opt_ours import OPTConfig

def build_model_and_enc(model_path, use_flash_attn, kv_bit=16, kv_group_size=128):
    print(f"* Building model {model_path}")

    # weither trust remote code
    if "chatglm" in model_path or "mpt" in model_path or "world" in model_path:
        trust_remote_code = True
    else:
        trust_remote_code = False

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    # config = OPTConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
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

    from module.opt.modeling_opt_ours import OPTForCausalLM

    model = OPTForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=trust_remote_code, **kwargs)

    return model, enc

model_path = '/home/huangshan/huangshan/project/quant_bert_opt/quantized_model'
model, enc = build_model_and_enc(model_path, False, 16, 128)

for i in range(32):
    print(model.model.decoder.layers[i].self_attn.out_proj.zeros_scales)

for name,_ in model.named_modules():
    print(name)