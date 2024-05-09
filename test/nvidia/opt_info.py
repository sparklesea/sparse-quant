import argparse

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
    model, enc = build_model_and_enc(args.model_path, args.use_flash_attn, args.kv_bit, args.kv_group_size, quantized=True)
    model.eval()
    if args.lut_path is not None:
        from playground.models.opt.modeling_opt import OPTModel_use_block_sparse_attention_lut
        from module.mask.sparse_attention import set_static_attention_lut

        print("use lut")
        model.model.decoder.use_block_sparse_attention_lut = OPTModel_use_block_sparse_attention_lut.__get__(model.model.decoder)
        model.model.decoder.use_block_sparse_attention_lut()
        set_static_attention_lut(args.lut_path, None, model.model.decoder.layers, 64)
    
    model_info_list=[{"total_weight_count":0,"4bit_count":0,"2bit_count":0,"avg_bit_width":0} for _ in range(32)]
    # print(model_info_list)
    for name, module in model.named_modules():
        if isinstance(module, WALinear):
            total_count = module.in_features * module.out_features
            # print(int(name.split(".")[3]))
            model_info_list[int(name.split(".")[3])]["total_weight_count"] += total_count
            if module.w_bit == 4:
                model_info_list[int(name.split(".")[3])]["4bit_count"] += total_count
                model_info_list[int(name.split(".")[3])]["avg_bit_width"] += total_count * module.w_bit
            elif module.w_bit == 2:
                model_info_list[int(name.split(".")[3])]["2bit_count"] += total_count * module.w_bit
                model_info_list[int(name.split(".")[3])]["avg_bit_width"] += total_count * module.w_bit

    
    total_avg_bit=0
    for i in range(32):
        model_info_list[i]["avg_bit_width"] /= model_info_list[i]["total_weight_count"]
        total_avg_bit+=model_info_list[i]["avg_bit_width"]
        print(f"layer_{i}: {model_info_list[i]}")
    print("total_avg_bit_width: ", total_avg_bit/32)

if __name__ == "__main__":
    main()
