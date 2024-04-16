import argparse

from utils.utils import build_model_and_enc
from module.lm_eval_adaptor import LMEvalAdaptor
from lm_eval import evaluator

from quantizer.opt_quantizer import OPTQuantizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path of the hf model")
parser.add_argument("--output_path", type=str, help="path to save the quantized model")
parser.add_argument("--use_flash_attn", action="store_true")
parser.add_argument("--tasks", type=str, default="truthfulqa_mc")
parser.add_argument("--metrics", type=str, default="mc1,mc2")
parser.add_argument("--w_group_size", type=int, default=128)
parser.add_argument("--w_bit", type=int, default=16)
parser.add_argument("--a_group_size", type=int, default=128)
parser.add_argument("--a_bit", type=int, default=16)
parser.add_argument("--kv_group_size", type=int, default=128)
parser.add_argument("--kv_bit", type=int, default=16)
parser.add_argument("--rep_file", type=str, default=None)
parser.add_argument("--mask_path", type=str, default=None)
args = parser.parse_args()


def main():
    print("* Quantization Format: kv_{}_w_{}_a_{}".format(args.kv_bit, args.w_bit, args.a_bit))
    if "falcon" in args.model_path.lower():
        args.kv_group_size = 64
        args.w_group_size = 64

    # a hack here to auto set model group
    model, enc = build_model_and_enc(args.model_path, args.use_flash_attn, args.kv_bit, args.kv_group_size)
    model.eval()
    if args.mask_path is not None:
        from module.mask.sparse_attention import set_static_attention_rule

        print("use masks")
        model, _ = set_static_attention_rule(
            model, args.mask_path, model_layers=model.model.decoder.layers
        )
    model=model.cuda()
    if args.w_bit<16 or args.a_bit<16:
        quantizer=OPTQuantizer(rep_file=args.rep_file,w_bit=args.w_bit,w_group_size=args.w_group_size,a_bit=args.a_bit,a_group_size=args.a_group_size,a_granularity="per_group")
        model = quantizer(model)
        print(model)
    # # save the quantized model
    # if args.output_path:
    #     model.save_pretrained(args.output_path, safe_serialization=False)
    #     enc.save_pretrained(args.output_path)

    # evaluation
    lm_eval_model = LMEvalAdaptor(args.model_path, model, enc, 1)

    if args.tasks is not None:
        task_names = args.tasks.split(",")

        results = evaluator.simple_evaluate(
            model=lm_eval_model,
            tasks=task_names,
            batch_size=1,
            no_cache=True,
            num_fewshot=0,
        )
        print(results)
        print(evaluator.make_table(results))


if __name__ == "__main__":
    main()
