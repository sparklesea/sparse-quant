
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
    "谈谈自然语言处理的应用领域。", 
    "请形容一下鱼这种生物。包括外形，栖息地，可食用性等。", 
    "写一段对OpenAI这家公司的介绍。", 
    "写一段对OpenAI这家公司的介绍，包含如下内容：公司创始人，公司所在地，核心产品以及股东构成，字数不超过200字。", 
    "把”今天的天气怎么样？“翻译成英文。", 
    "用一句话总结如下内容：”区块链，就是一个又一个区块组成的链条。每一个区块中保存了一定的信息，它们按照各自产生的时间顺序连接成链条。这个链条被保存在所有的服务器中，只要整个系统中有一台服务器可以工作，整条区块链就是安全的。这些服务器在区块链系统中被称为节点，它们为整个区块链系统提供存储空间和算力支持。如果要修改区块链中的信息，必须征得半数以上节点的同意并修改所有节点中的信息，而这些节点通常掌握在不同的主体手中，因此篡改区块链中的信息是一件极其困难的事。相比于传统的网络，区块链具有两大核心特点：一是数据难以篡改、二是去中心化。基于这两个特点，区块链所记录的信息更加真实可靠，可以帮助解决人们互不信任的问题。“", 
    "简要介绍一下阿里巴巴。", 
    "用一句话总结如下内容：”5 月 1 日，福建三明一游乐场内，一位女演员在进行高空表演走钢丝的时候，由于失误发生了意外，女子直接被挂在半空，此事引发大量网友关注。2 日，事发游乐场工作人员回应：女演员在进行高空表演时，被防坠落的装置卡住了，事情发生后，迅速安排救援人员进行救援，女演员在空中挂了一二十分钟后被救下来了，没有生命危险。因为安全保护措施太多了，起到了反作用。“", 
    "给咖啡店做一个面向年轻人的菜单，在给出最终答案之前，请在回复中采用 step by step 的方式。", 
    "给我推荐 2 本科幻小说，包含作者，出版时间，推荐理由。", 
    "请帮我写一个c++的helloworld程序。", 
    "请帮我写一个python的helloworld程序。", 
    "写首春天的诗。", 
    "请写出圆周率", 
    "以下是一段python程序：a=1+10。执行之后a的值是多少？", 
    "你如何看待死亡？", 
    "请解释一下质数的意思，并举几个例子。", 
    "你是一个野外求生专家，请写一篇在原始森林里的求生指南。", 
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
            # print("prompt: ", prompt)
            print("output: ", out, "\n")


if __name__ == "__main__":
    main()


