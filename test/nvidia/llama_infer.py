
import argparse

from utils.utils import build_model_and_enc

from quantizer.llama_quantizer import LlamaQuantizer

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
# parser.add_argument("--sample", nargs="+", type=int, default=[0, 1, 2, 3, 4])
parser.add_argument("--sample", nargs="+", type=int)
parser.add_argument("--eval", action="store_true")
args = parser.parse_args()

prompts = [
    "北京有什么好玩的地方？", 
    "请帮我做一个上海地区的详细旅游攻略，分三天安排。", 
    "你认为在科学领域，生命的意义是什么？", 
    "请形容一下鱼这种生物。包括外形，栖息地，可食用性等。", 
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
    if not args.quantized:
        if args.w_bit<16 or args.a_bit<16:
            quantizer=LlamaQuantizer(rep_file=args.rep_file,w_bit=args.w_bit,w_group_size=args.w_group_size,a_bit=args.a_bit,a_group_size=args.a_group_size,a_granularity="per_group")
            model = quantizer(model)
            # print(model)
        # # save the quantized model
        if args.output_path:
            model.save_pretrained(args.output_path, safe_serialization=False)
            # model.save_pretrained(args.output_path, safe_serialization=True)
            enc.save_pretrained(args.output_path)
    
    model.eval()
    model=model.cuda()
    if args.eval:
        if args.sample:
            print(args.sample)
            # sample_prompts = [prompts[i] for i in args.sample]
            sample_prompts = [f"### Instruction:{prompts[i].strip()}  ### Response:\n" for i in args.sample]
        else:
            sample_id = torch.randperm(20).tolist()[:5]
            # sample_prompts = [prompts[i] for i in sample_id]
            sample_prompts = [f"### Instruction:{prompts[i].strip()}  ### Response:\n" for i in sample_id]
            print("randomly selected ids: ", sample_id)
        for prompt in sample_prompts:
            inputs = enc(prompt, return_tensors="pt").to("cuda")
            # generated_ids = model.generate(inputs.input_ids, do_sample=True, max_new_tokens=512, top_k=10, top_p=0.85, temperature=1, repetition_penalty=1.15, eos_token_id=2, bos_token_id=1, pad_token_id=0)
            generated_ids = model.generate(inputs.input_ids, do_sample=False, max_new_tokens=512, temperature=1, repetition_penalty=1.15, eos_token_id=2, bos_token_id=1, pad_token_id=0)
            out = enc.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print("prompt: ", prompt)
            print("output: ", out, "\n")


if __name__ == "__main__":
    main()


