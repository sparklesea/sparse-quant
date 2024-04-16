
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Tuple
from datasets import load_dataset, load_from_disk

from transformers import AutoTokenizer, OPTForCausalLM, OPTConfig

import argparse

from playground.attention.sparse_attention import set_static_attention_lut

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='/share/huangshan/opt-6.7b/', help='path to model')
parser.add_argument('--max_length', type=int, default=2048, help='max length of the sequence')
parser.add_argument('--dataset_dir', type=str,default='EleutherAI/wikitext_document_level', help='path to dataset')
parser.add_argument('--subset', type=str, default='wikitext-103-v1')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--column', type=str, default='page')
parser.add_argument('--loss_type', choices=['cross_entropy', 'ppl'], default='ppl', help='loss type')
parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16')
parser.add_argument('--use_layout', action='store_true')
parser.add_argument('--layout_path', type=str, default=None)
parser.add_argument('--use_lut', action='store_true')
parser.add_argument('--lut_path', type=str, default=None)
parser.add_argument('--block_size', type=int, default=64)

args = parser.parse_args()

def load_model(
    model_name: str,
    max_length: int,
    use_layout: bool = False,
    layout_path: Optional[str] = None,
    use_lut: bool = False,
    lut_path: Optional[str] = None,
    block_size: int = 64,
    use_flash_attention: bool = False,
    dtype=torch.float16,
) -> Tuple[AutoTokenizer, nn.Module]:
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # define the model
    config = OPTConfig.from_pretrained(model_name)
    if use_flash_attention:
        config._attn_implementation = "flash_attention_2"
    else:
        config._attn_implementation_internal = "eager"
    model = OPTForCausalLM.from_pretrained(model_name, config=config, device_map='auto', torch_dtype=dtype, attn_implementation='eager' if not use_flash_attention else 'flash_attention_2').eval()

    # if use_layout:
    #     from playground.models.llama.modeling_llama import LlamaModel_use_block_sparse_attention
    #     model.model.use_block_sparse_attention = LlamaModel_use_block_sparse_attention.__get__(model.model)
    #     model.model.use_block_sparse_attention()
    #     set_static_attention_layout(layout_path, max_length, model.model.layers, block_size)

    if use_lut:
        from playground.models.opt.modeling_opt import OPTModel_use_block_sparse_attention_lut
        model.model.decoder.use_block_sparse_attention_lut = OPTModel_use_block_sparse_attention_lut.__get__(model.model.decoder)
        model.model.decoder.use_block_sparse_attention_lut()
        set_static_attention_lut(lut_path, None, model.model.decoder.layers, block_size)
    
    return model, tokenizer


if __name__ == '__main__':
    if args.dtype == 'fp32':
        dtype = torch.float32
    elif args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError("unsupported data type")

    model, tokenizer = load_model(
        model_name=args.model_name,
        max_length=args.max_length,
        use_layout=args.use_layout,
        layout_path=args.layout_path,
        use_lut=args.use_lut,
        lut_path=args.lut_path,
        dtype=dtype,
        block_size=args.block_size
    )

    model = model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = False

    # CUDA_VISIBLE_DEVICES=6 python test_opt.py --use_lut --lut_path=/home/yuzhen/simple-evaluation-master/examples/opt_lut_density_26.pt
    prompt = "the man worked as a"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    generated_ids = model.generate(input_ids, max_length=512)
    out = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    print(out)

    # dataset = load_dataset(args.dataset_dir, args.subset, split=args.split)

    # sep = tokenizer.encode("\n\n", add_special_tokens=False)

    # # concat the dataset into one long string
    # def tokenization(example):
    #     return tokenizer(example[args.column], add_special_tokens=False)
    
    # dataset = dataset.map(tokenization, batched=True, batch_size=16)

    # batched_ids = []
    # for ids in dataset['input_ids']:
    #     batched_ids.extend(ids + sep)

    # # split into max_length chunks
    # tokenized_list = []
    # max_length = args.max_length
    # for i in range(0, len(batched_ids), max_length):
    #     tokenized_list.append(batched_ids[i:i+max_length])
    # tokenized_list = tokenized_list[:-1] # remove the last one because it's not a full chunk
    # chunks = tokenizer.batch_decode(tokenized_list, skip_special_tokens=False)

    # loss_list = []

    # pbar = tqdm(total=len(chunks))

    # for data_sample in chunks:

    #     data_sample = tokenizer(data_sample, return_tensors='pt', padding = 'max_length', max_length=args.max_length, truncation=True)['input_ids']
    #     loss = model(data_sample.to(next(model.parameters()).device), labels=data_sample.to(next(model.parameters()).device))[0] # cross entropy loss
    #     if args.loss_type == 'ppl':
    #         loss = torch.exp(loss) # use ppl as loss
    #     elif args.loss_type == 'cross_entropy':
    #         pass
    #     else:
    #         raise NotImplementedError

    #     loss_list.append(loss.cpu().item())

    #     pbar.update(1)

    # pbar.close()

    # loss = sum(loss_list) / len(loss_list)
    # print(f"{args.loss_type}: {loss:.4f}")


