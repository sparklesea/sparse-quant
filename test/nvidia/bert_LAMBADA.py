from transformers import AutoTokenizer
from datasets import Dataset
import argparse
from torch.utils.data import DataLoader
import torch
import json
from tqdm import tqdm
from quantizer.bert_quantizer import BERTQuantizer


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="$MODEL_BERT_PATH")
parser.add_argument("--w_group_size", type=int, default=128)
parser.add_argument("--w_bit", type=int, default=16)
parser.add_argument("--a_group_size", type=int, default=128)
parser.add_argument("--a_bit", type=int, default=16)
parser.add_argument("--kv_group_size", type=int, default=64)
parser.add_argument("--kv_bit", type=int, default=16)
parser.add_argument("--mask_path", type=str, default=None)  
parser.add_argument('--lut_path', type=str, default=None)
parser.add_argument("--output_path", type=str, help="path to save the quantized model")
parser.add_argument("--quantized", action="store_true")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--fake", action="store_true")
args = parser.parse_args()

enc = AutoTokenizer.from_pretrained("bert-large-cased")


def tokenize_function(examples):

    inputs = enc(examples["sentence1"], examples["sentence2"], truncation=True)
    # inputs = enc(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)
    # sep_token_id = enc.sep_token_id
    # for i in range(len(inputs["token_type_ids"])):
    #     sep_position = inputs["input_ids"][i].index(sep_token_id)
    #     inputs["token_type_ids"][i] = (sep_position + 1) * [0] + (len(inputs["token_type_ids"][i]) - sep_position - 1) * [1]
    return inputs


with open("test.json", "r") as f:
    test_data = json.load(f)
    eval_dataset = Dataset.from_dict(test_data)

inputs_data,outputs_data=[],[]
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
    quantizer=BERTQuantizer(w_bit=args.w_bit,a_bit=args.a_bit,w_group_size=args.w_group_size,fake_quant=args.fake)
    model = quantizer(model)
    print(model)
    if args.output_path:
        model.save_pretrained(args.output_path, safe_serialization=True)
        # enc.torch_dtype="float16"
        enc.save_pretrained(args.output_path, safe_serialization=True)
model.eval()

if args.eval:
    acc, cnt = 0, 0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            cnt += len(batch["labels"])
            logits = model(**batch)[1]
            pred = torch.argmax(logits, dim=-1)
            # acc += torch.sum(pred == batch["labels"])
            acc += torch.sum(pred == batch["labels"])
    print("Accurate:", acc)
    print("Sum:", cnt)
    print("Accuracy:", acc / cnt)
