import argparse

from utils.utils import build_model_and_enc
from module.lm_eval_adaptor import LMEvalAdaptor
from lm_eval import evaluator

from quantizer.opt_quantizer import OPTQuantizer

import torch

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
parser.add_argument("--sample", nargs="+", type=int)
parser.add_argument("--eval", action="store_true")
args = parser.parse_args()


# prompts = [
#     "In the future city, between skyscrapers towering into the clouds, people travel through", 
#     "In the morning, the first ray of sunlight filtered through the curtains, spilling onto the bed. I slowly opened my eyes and found", 
#     "Imagine you are an explorer searching for a lost ancient city in an unknown jungle", 
#     "In the future underwater city, humans coexist harmoniously with marine life, creating a", 
#     "On a distant planet, the passage of time differs from that on Earth, and a day might be equivalent to", 
#     "Amidst the entanglement of war and peace, a little girl changed the fate of an entire country with her innocence and courage", 
#     "Deep in the distant space, a spaceship encountered an unknown crisis, and the crew had to", 
#     "In the world of the future, virtual reality technology has reached its peak, allowing people to", 
#     "In the endless universe, there is a creature that possesses the ability to control time, known as", 
#     "On a journey filled with fantasy and adventure, a young warrior and his companions encountered", 
#     "Describe the adventure of a lifetime for a group of friends who found a map leading to a parallel universe.", 
#     "What were the major contributing factors to the fall of the Roman Empire?", 
#     "How did the invention of the printing press revolutionize European society?", 
#     "What are the effects of quantitative easing?", 
#     "What were the economic and philosophical factors that led to the fall of the Soviet Union?", 
#     "What was the influence of the Khmer Empire on Southeast Asia's history and culture?", 
#     "What is the Voynich manuscript, and why has it perplexed scholars for centuries?", 
#     "What is the 'Dyatlov Pass incident' and why does it remain a mystery?", 
#     "What is the 'Phantom Time Hypothesis' proposed by Heribert Illig?", 
#     "Who were the 'Dog-Headed Saint' and the 'Lion-Faced Saint' in medieval Christian traditions?", 
# ]

prompts = [
    "In that tumultuous era, heroes emerged, and none was more legendary than", 
    "Once upon a time, in a faraway forest, there lived a", 
    "In the morning, the first ray of sunlight filtered through the curtains, spilling onto the bed. I slowly opened my eyes and found", 
    "In the future underwater city, humans coexist harmoniously with marine life, creating a", 
    "On a distant planet, the passage of time differs from that on Earth, and a day might be equivalent to", 
    "As night fell, a lonely traveler entered a mysterious tavern and encountered", 
    "In the future city, between skyscrapers towering into the clouds, people travel through", 
    "In a kingdom filled with magic and miracles, a young mage decided to save his country", 
    "When robots acquire emotions and consciousness, how will their relationship with humans change", 
    "In a parallel world where technology and magic coexist, humans create a weapon", 
    "On a journey filled with fantasy and adventure, a young warrior and his companions encountered", 
    "Deep in the distant space, a human spaceship encountered a friendly monster", 
    "What is the Voynich manuscript, and why has it perplexed scholars for centuries?", 
    "What was Project A119 and what were its objectives?", 
    "What is the 'Phantom Time Hypothesis' proposed by Heribert Illig?", 
    "What were the major contributing factors to the fall of the Roman Empire?", 
    "How did the invention of the printing press revolutionize European society?", 
    "What were the economic and philosophical factors that led to the fall of the Soviet Union?", 
    "What are 'zombie stars' in the context of astronomy?", 
    "What was the influence of the Khmer Empire on Southeast Asia's history and culture?", 
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

    if args.eval:
        if args.sample:
            print(args.sample)
            sample_prompts = [prompts[i] for i in args.sample]
        else:
            sample_id = torch.randperm(20).tolist()[:5]
            sample_prompts = [prompts[i] for i in sample_id]
            print(sample_id)
        # print(sample_prompts)
        for prompt in sample_prompts:
            input_ids = enc(prompt, return_tensors="pt").input_ids.cuda()
            generated_ids = model.generate(input_ids, max_length=64)
            out = enc.batch_decode(generated_ids, skip_special_tokens=True)

            print("prompt: ", prompt)
            print("output: ", repr(out[0])[1:-1], "\n")


if __name__ == "__main__":
    # first gen quanted model
    # CUDA_VISIBLE_DEVICES=6 python eval_compress_opt_support_3090.py --model_path $MODEL_OPT_PATH --w_bit 4 --output_path quantized_model/opt

    # then run
    # CUDA_VISIBLE_DEVICES=6 python eval_compress_opt_support_3090.py --model_path quantized_model/opt --lut_path /share/huangshan/masks/opt_lut_density_26.pt --w_bit 4 --quantized
    main()
