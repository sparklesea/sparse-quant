import argparse

from utils.utils import build_model_and_enc
from module.lm_eval_adaptor import LMEvalAdaptor
from lm_eval import evaluator

from quantizer.opt_quantizer import OPTQuantizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path of the hf model")
parser.add_argument("--output_path", type=str, help="path to save the quantized model")
parser.add_argument("--use_flash_attn", action="store_true")
parser.add_argument("--tasks", type=str, default=None)
parser.add_argument("--metrics", type=str, default="mc1,mc2")
parser.add_argument("--w_group_size", type=int, default=128)
parser.add_argument("--w_bit", type=int, default=16)
parser.add_argument("--a_group_size", type=int, default=128)
parser.add_argument("--a_bit", type=int, default=16)
parser.add_argument("--kv_group_size", type=int, default=128)
parser.add_argument("--kv_bit", type=int, default=16)
parser.add_argument("--rep_file", type=str, default=None)
parser.add_argument("--mask_path", type=str, default=None)
parser.add_argument("--quantized", action="store_true")
parser.add_argument("--lut_path", type=str, default=None)
parser.add_argument("--sample", nargs="+", type=int, default=[0, 1, 2, 3, 4])
args = parser.parse_args()

prompts = [
    "sentence 1", 
    "sentence 2", 
    "sentence 3", 
    "sentence 4", 
    "sentence 5", 
    "sentence 6", 
    "sentence 7", 
    "sentence 8", 
    "sentence 9", 
    "sentence 10", 
    "sentence 11", 
    "sentence 12", 
    "sentence 13", 
    "sentence 14", 
    "sentence 15", 
    "sentence 16", 
    "sentence 17", 
    "sentence 18", 
    "sentence 19", 
    "sentence 20", 
]


def main():
    print("* Quantization Format: kv_{}_w_{}_a_{}".format(args.kv_bit, args.w_bit, args.a_bit))
    if "falcon" in args.model_path.lower():
        args.kv_group_size = 64
        args.w_group_size = 64

    # a hack here to auto set model group
    model, enc = build_model_and_enc(args.model_path, args.use_flash_attn, args.kv_bit, args.kv_group_size, args.quantized)
    model.eval()
    if args.mask_path is not None:
        from module.mask.sparse_attention import set_static_attention_rule

        print("use masks")
        model, _ = set_static_attention_rule(
            model, args.mask_path, model_layers=model.model.decoder.layers
        )
    if args.lut_path is not None:
        from playground.models.opt.modeling_opt import OPTModel_use_block_sparse_attention_lut
        from module.mask.sparse_attention import set_static_attention_lut

        print("use lut")
        model.model.decoder.use_block_sparse_attention_lut = OPTModel_use_block_sparse_attention_lut.__get__(model.model.decoder)
        model.model.decoder.use_block_sparse_attention_lut()
        set_static_attention_lut(args.lut_path, None, model.model.decoder.layers, 64)
    model=model.cuda()
    if not args.quantized:
        if args.w_bit<16 or args.a_bit<16:
            quantizer=OPTQuantizer(rep_file=args.rep_file,w_bit=args.w_bit,w_group_size=args.w_group_size,a_bit=args.a_bit,a_group_size=args.a_group_size,a_granularity="per_group")
            model = quantizer(model)
            # print(model)
        # # save the quantized model
        if args.output_path:
            model.save_pretrained(args.output_path, safe_serialization=False)
            # model.save_pretrained(args.output_path, safe_serialization=True)
            enc.save_pretrained(args.output_path)

    # CUDA_VISIBLE_DEVICES=6 python eval_compress_opt_support_3090.py --model_path=/share/huangshan/opt-6.7b/ --lut_path masks/opt_lut_density_26.pt
    print(args.sample)
    sample_prompts = [prompts[i] for i in args.sample]
    # print(sample_prompts)
    for prompt in sample_prompts:
        input_ids = enc(prompt, return_tensors="pt").input_ids.cuda()
        generated_ids = model.generate(input_ids, max_length=512)
        out = enc.batch_decode(generated_ids, skip_special_tokens=True)

        print(out)


if __name__ == "__main__":
    # first gen quanted model
    # CUDA_VISIBLE_DEVICES=6 python opt_infer.py --model_path /share/huangshan/opt-6.7b --w_bit 4 --output_path quantized_model/opt

    # then run
    # CUDA_VISIBLE_DEVICES=6 python opt_infer.py --model_path quantized_model/opt --lut_path masks/opt_lut_density_26.pt --w_bit 4 --quantized
    main()