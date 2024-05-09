from transformers import AutoTokenizer
from datasets import Dataset
import argparse
from torch.utils.data import DataLoader
import torch
import json
from tqdm import tqdm
from quantizer.bert_quantizer import BERTQuantizer


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="bert_model/bert-large-cased-lambada")
parser.add_argument("--w_group_size", type=int, default=128)
parser.add_argument("--w_bit", type=int, default=16)
parser.add_argument("--a_group_size", type=int, default=128)
parser.add_argument("--a_bit", type=int, default=16)
parser.add_argument("--kv_group_size", type=int, default=64)
parser.add_argument("--kv_bit", type=int, default=16)
parser.add_argument("--mask_path", type=str, default=None)  # /share/liutengxuan/NLP-playground/examples/sparse_attention/model/bert/bigbird_pattern_24_16_512_512.pt")
parser.add_argument('--lut_path', type=str, default=None)
parser.add_argument("--output_path", type=str, help="path to save the quantized model")
parser.add_argument("--quantized", action="store_true")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--sample", nargs="+", type=int, default=[0, 5, 8, 15, 18])
args = parser.parse_args()

enc = AutoTokenizer.from_pretrained("bert-large-cased")

def tokenize_function(examples):

    inputs = enc(examples["sentence1"], examples["sentence2"], truncation=True)
    return inputs

dataset = {
    "sentence1": [
        "pre sentence 1", 
        "pre sentence 2", 
        "pre sentence 3", 
        "pre sentence 4", 
        "pre sentence 5", 
        "pre sentence 6", 
        "pre sentence 7", 
        "pre sentence 8", 
        "pre sentence 9", 
        "pre sentence 10", 
        "pre sentence 11", 
        "pre sentence 12", 
        "pre sentence 13", 
        "pre sentence 14", 
        "pre sentence 15", 
        "pre sentence 16", 
        "pre sentence 17", 
        "pre sentence 18", 
        "pre sentence 19", 
        "pre sentence 20"
    ], 
    "sentence2": [
        "next sentence 1", 
        "next sentence 2", 
        "next sentence 3", 
        "next sentence 4", 
        "next sentence 5", 
        "next sentence 6", 
        "next sentence 7", 
        "next sentence 8", 
        "next sentence 9", 
        "next sentence 10", 
        "next sentence 11", 
        "next sentence 12", 
        "next sentence 13", 
        "next sentence 14", 
        "next sentence 15", 
        "next sentence 16", 
        "next sentence 17", 
        "next sentence 18", 
        "next sentence 19", 
        "next sentence 20"
    ],
    "labels": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

eval_dataset = Dataset.from_dict(dataset)
eval_dataset = eval_dataset[args.sample]
eval_dataset = Dataset.from_dict(eval_dataset)
print(eval_dataset)

def collate_fn(data):
    tensor_input_ids, tensor_token_type_ids, tensor_attention_mask, idx, labels = [], [], [], [], []
    for item in data:
        tensor_input_ids.append(item["input_ids"])
        tensor_token_type_ids.append(item["token_type_ids"])
        tensor_attention_mask.append(item["attention_mask"])
        # idx.append(item["idx"])
        labels.append(item["labels"])
        tensor_input_ids = torch.tensor(tensor_input_ids, dtype=int).cuda()
        tensor_token_type_ids = torch.tensor(tensor_token_type_ids, dtype=int).cuda()
        tensor_attention_mask = torch.tensor(tensor_attention_mask, dtype=int).cuda()
    # idx = torch.tensor(idx, dtype=int).cuda()
    labels = torch.tensor(labels, dtype=int).cuda()
    return {"labels": labels, "input_ids": tensor_input_ids, "token_type_ids": tensor_token_type_ids, "attention_mask": tensor_attention_mask}

eval_dataset = eval_dataset.map(tokenize_function, batched=True).shuffle(seed=42)
eval_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=collate_fn)

if not args.quantized:
    from transformers import BertForNextSentencePrediction
else:
    from module.bert.modeling_bert_ours import BertForNextSentencePrediction

kwargs = {"torch_dtype": torch.float16}
model = BertForNextSentencePrediction.from_pretrained(args.model_path, **kwargs)
if args.mask_path is not None:
    from module.bert.modeling_bert import BertModel_use_static_attention

    model.bert.use_static_attention = BertModel_use_static_attention.__get__(model.bert)
    model.bert.use_static_attention()
    print("Using sparse mask {}".format(args.mask_path))
    model.bert.set_static_attention_mask(args.mask_path)
if args.lut_path is not None:
    from module.bert.modeling_bert import BertModel_use_block_sparse_attention_lut
    from module.mask.sparse_attention import set_static_attention_lut

    model.bert.use_static_attention = BertModel_use_block_sparse_attention_lut.__get__(model.bert)
    model.bert.use_static_attention()
    print("Using sparse lut {}".format(args.lut_path))
    set_static_attention_lut(args.lut_path, None, model.bert.encoder.layer, 64)

model = model.to("cuda")
if not args.quantized:
    quantizer=BERTQuantizer(w_bit=args.w_bit,a_bit=args.a_bit,w_group_size=args.w_group_size)
    model = quantizer(model)
    print(model)
    if args.output_path:
        model.save_pretrained(args.output_path, safe_serialization=True)
        # enc.torch_dtype="float16"
        enc.save_pretrained(args.output_path, safe_serialization=True)
model.eval()


with torch.no_grad():
    for batch in tqdm(eval_dataloader):
        logits = model(**batch)[1]
        pred = torch.argmax(logits, dim=-1)
        print(pred)


# first gen quanted model
# CUDA_VISIBLE_DEVICES=6 python eval_compress_bert_support_3090.py --model_path bert_model/bert-large-cased-lambada --w_bit 4 --lut_path masks/bert_large_lut.pt --output_path quantized_model/bert

# then run
# CUDA_VISIBLE_DEVICES=6 python eval_compress_bert_support_3090.py --model_path quantized_model/bert --w_bit 4 --lut_path masks/bert_large_lut.pt --quantized

