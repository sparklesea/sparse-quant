import argparse
import torch
from utils.utils import build_model_and_enc
from module.qlinear.qlinear import WALinear

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
parser.add_argument("--lut_path", type=str, default=None)
args = parser.parse_args()

def main():
    print("* Quantization Format: kv_{}_w_{}_a_{}".format(args.kv_bit, args.w_bit, args.a_bit))

    # a hack here to auto set model group
    from module.bert.modeling_bert_ours import BertForNextSentencePrediction

    kwargs = {"torch_dtype": torch.float16}
    model = BertForNextSentencePrediction.from_pretrained(args.model_path, **kwargs)

    if args.lut_path is not None:
        from module.bert.modeling_bert import BertModel_use_block_sparse_attention_lut
        from module.mask.sparse_attention import set_static_attention_lut

        model.bert.use_static_attention = BertModel_use_block_sparse_attention_lut.__get__(model.bert)
        model.bert.use_static_attention()
        print("Using sparse lut {}".format(args.lut_path))
        set_static_attention_lut(args.lut_path, None, model.bert.encoder.layer, 64)

    model.eval()

    sparsity = count_sparsity(args.lut_path)
    model_info_list=[{"weight_count":0,"4bit_count":0,"2bit_count":0,"avg_bit_width":0, "sparsity":(sparsity[i].mean()).item()} for i in range(24)]
    model_info_list.append({"weight_count":0,"4bit_count":0,"2bit_count":0,"avg_bit_width":0})
    # print(model_info_list)
    for name, module in model.named_modules():
        if isinstance(module, WALinear):
            if "pooler.dense" in name:
                pos = 24
            else:
                pos = int(name.split(".")[3])
            total_count = module.in_features * module.out_features
            # print(int(name.split(".")[3]))
            model_info_list[pos]["weight_count"] += total_count
            if module.w_bit == 4:
                model_info_list[pos]["4bit_count"] += total_count
                model_info_list[pos]["avg_bit_width"] += total_count * module.w_bit
            elif module.w_bit == 2:
                model_info_list[pos]["2bit_count"] += total_count * module.w_bit
                model_info_list[pos]["avg_bit_width"] += total_count * module.w_bit

    total_weight_count,total_4bit_count,total_2bit_count=0,0,0
    total_avg_bit=0
    for i in range(25):
        total_weight_count += model_info_list[i]["weight_count"]
        total_4bit_count += model_info_list[i]["4bit_count"]
        total_2bit_count += model_info_list[i]["2bit_count"]
        model_info_list[i]["avg_bit_width"] /= model_info_list[i]["weight_count"]
        total_avg_bit+=model_info_list[i]["avg_bit_width"]
        if i < 24:
            print(f"layer_{i}: {model_info_list[i]}")
        else:
            print(f"pooler: {model_info_list[i]}")
    print(f"total_weight_count: {total_weight_count}, total_4bit_count: {total_4bit_count}, total_2bit_count: {total_2bit_count}")
    print("total_avg_bit_width: ", total_avg_bit/25)

    spar = torch.sum(sparsity) / (sparsity.shape[0] * sparsity.shape[1])
    print("total sparsity: ", spar)

def count_mask_sparsity(mask):
    x = mask.shape[0]
    block = 0
    for i in range(x):
        block += torch.unique(mask[i]).shape[0]
    
    return 1 - block / (x * x)

def count_sparsity(lut_path):
    lut = torch.load(lut_path)

    layer_num = len(lut)
    head_num = lut[0].shape[0]

    sparsity = torch.zeros((layer_num, head_num))

    for layer in range(layer_num):
        for head in range(head_num):
            mask = lut[layer][head]
            sparsity[layer][head] = count_mask_sparsity(mask)

    return sparsity

if __name__ == "__main__":
    main()
