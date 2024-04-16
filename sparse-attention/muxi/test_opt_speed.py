import torch
import torch.nn as nn
import json
from typing import Optional, Tuple
# from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, OPTForCausalLM, OPTConfig
from pathlib import Path

from playground.attention.sparse_attention import set_static_attention_lut

Configs = [
    # # batch, input_length, output_length

    # (1, 128, 128),
    # (1, 1024, 128),
    # (1, 2*1024, 128),
    # (1, 4*1024, 128),
    # (1, 8*1024, 128),
    # (1, 32*1024, 128),

    # (2, 128, 128),
    # (2, 1024, 128),
    # (2, 2*1024, 128),
    # (2, 4*1024, 128),
    # (2, 8*1024, 128),
    # (2, 32*1024, 128),

    # (4, 128, 128),
    # (4, 1024, 128),
    # (4, 2*1024, 128),
    # (4, 4*1024, 128),
    # (4, 8*1024, 128),
    # (4, 32*1024, 128),

    # (8, 128, 128),
    # (8, 1024, 128),
    # (8, 2*1024, 128),
    # (8, 4*1024, 128),
    # (8, 8*1024, 128),
    # (8, 32*1024, 128),
    (1, 32, 32),
    (1, 64, 64),
    (1, 128, 128),
    (1, 256, 256),
    # (1, 512, 512),


]


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


def main(model_name, backend='hf', gpu=0):
    torch.cuda.empty_cache()
    print('\nModel: ', model_name, '\nBackend: ', backend, '\n')

    ckpt_dir = model_name

    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(gpu))
    else:
        device = torch.device("cpu")
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)

    max_seq_len  = 4*1024+128+2

    if backend == 'hf':
        # model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code = True).half().to(device)

        
        model = OPTForCausalLM.from_pretrained(model_name, 
                                               ignore_mismatched_sizes = True, 
                                               max_position_embeddings = max_seq_len,).half().to(device)
    elif backend == 'ours':

        model, tokenizer = load_model(
            model_name=model_name,
            max_length=2048,
            use_layout=False,
            layout_path=None,
            use_lut=True,
            lut_path="/home/yuzhen/simple-evaluation-master/examples/opt_lut_density_26.pt",
            dtype=torch.float16,
            block_size=64
        )
        # model = OPTForCausalLM_ours.from_pretrained(model_name, 
        #                                             ignore_mismatched_sizes = True, 
        #                                             max_position_embeddings = max_seq_len,
        #                                             max_seq_len = max_seq_len).half().to(device)

        # model = OPTForCausalLM_ours.from_pretrained(model_name).half().to(device)

        
        # model = AutoModelForCausalLM.from_pretrained(model_name, 
        #                                              ignore_mismatched_sizes = True, 
        #                                              max_position_embeddings = max_seq_len).half().to(device)
        
    
    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)
    warmup, freq = 2, 4

    print("batch  query_length  answer_length  query_latency(ms)  answer_latency(ms)  total_latency(ms)  tokens/second(total) tokens/second(answer)")
    for batch, input_length, output_length in Configs:
        torch.cuda.empty_cache()
        torch.cuda.nvtx.range_push('{} batch {} input_length {} output_length {}'.format(model_name, batch, input_length, output_length))
        # prepare inputs
        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
        
        input_ids = [[666] * input_length] * batch
        input_ids_t = torch.tensor(input_ids, device=device)

        # warm up
        for _ in range(warmup):
            if backend == 'hf':
                logits = model.generate(input_ids_t, num_beams=1, max_length=input_length+1, use_cache=True)
            elif backend == 'ours':
                logits = model.generate(input_ids_t, num_beams=1, max_length=input_length+1, use_cache=True)
            
        query_latency = 0

        for _ in range(freq):

            st.record()

            if backend == 'hf':
                logits = model.generate(input_ids_t, num_beams=1, max_length=input_length+1, use_cache=True)
            elif backend == 'ours':
                logits = model.generate(input_ids_t, num_beams=1, max_length=input_length+1, use_cache=True)
           
            ed.record()
            ed.synchronize()
            query_latency += st.elapsed_time(ed) / freq


        total_latency = 0
        current_len = 0

        for _ in range(freq):

            st.record()

            if backend == 'hf':
                logits = model.generate(input_ids_t, num_beams=1, max_length=input_length+output_length+1, use_cache=True)
            elif backend == 'ours':
                logits = model.generate(input_ids_t, num_beams=1, max_length=input_length+output_length+1, use_cache=True)
           
            ed.record()
            ed.synchronize()
            total_latency += st.elapsed_time(ed) / freq    

            
            for i in range(batch):
                current_len += len(logits[i]) - input_length - 1


        output_length = current_len / (batch * freq)

    
        answer_lantency = total_latency - query_latency
        answer_token_output_latency = answer_lantency / output_length
        answer_tokens_per_second = (1000 / answer_token_output_latency) * batch
        total_token_output_latency = total_latency / (output_length + 1)
        total_tokens_per_second = (1000 / total_token_output_latency) * batch

        print(str(batch).ljust(len('batch')) + "  " +
                str(input_length).ljust(len('query_length')) + "  " +
                str(output_length).ljust(len('answer_length')) + "  " +
                "{:.3f}".format(query_latency).ljust(len('query_latency(ms)')) + "  " +
                "{:.3f}".format(answer_lantency).ljust(len('answer_latency(ms)')) +  "  " +
                "{:.3f}".format(total_latency).ljust(len('total_latency(ms)')) + "  " +
                "{:.3f}".format(total_tokens_per_second).ljust(len('total_tokens_second')) + "  " +
                "{:.3f}".format(answer_tokens_per_second).ljust(len('answer_tokens_second'))) 


if __name__ == "__main__":

    
    # main('/share/chenkangdi/public_model/models--facebook--opt-6.7b', 'hf')
    # main('/share/chenkangdi/public_model/models--facebook--opt-6.7b', 'ours')

    # main('/share/chenkangdi/public_model/models--facebook--opt-13b', 'hf')
    # main('/share/chenkangdi/public_model/models--facebook--opt-13b', 'ours')
    main('/share/huangshan/opt-6.7b/', 'hf')
    main('/share/huangshan/opt-6.7b/', 'ours')
