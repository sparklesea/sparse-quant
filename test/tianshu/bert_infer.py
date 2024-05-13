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
parser.add_argument("--sample", nargs="+", type=int, default=[0, 5, 8, 15, 18])
args = parser.parse_args()

enc = AutoTokenizer.from_pretrained("bert-large-cased")

def tokenize_function(examples):

    inputs = enc(examples["sentence1"], examples["sentence2"], truncation=True)
    return inputs

dataset = {
    "sentence1": [
        "in my palm is a clear stone , and inside it is a small ivory statuette . a guardian angel . `` figured if you 're going to be out at night getting hit by cars , you might as well have some backup . '' i look at him , feeling stunned . like this is some sort of sign ", 
        "in my palm is a clear stone , and inside it is a small ivory statuette . a guardian angel . `` figured if you 're going to be out at night getting hit by cars , you might as well have some backup . '' i look at him , feeling stunned . like this is some sort of sign ", 
        "give me a minute to change and i 'll meet you at the docks . '' she 'd forced those words through her teeth . `` no need to change . we wo n't be that long . '' shane gripped her arm and started leading her to the dock ", 
        "give me a minute to change and i 'll meet you at the docks . '' she 'd forced those words through her teeth . `` no need to change . we wo n't be that long . '' shane gripped her arm and started leading her to the dock ", 
        "`` only one source i know of that would be likely to cough up enough money to finance a phony sleep research facility and pay people big bucks to solve crimes in their dreams , '' farrell concluded dryly . `` what can i say ? '' ellis unfolded his arms and widened his hands . `` your tax dollars at work . '' before farrell could respond , leila 's voice rose from inside the house . `` no insurance ? '' she wailed ", 
        "`` only one source i know of that would be likely to cough up enough money to finance a phony sleep research facility and pay people big bucks to solve crimes in their dreams , '' farrell concluded dryly . `` what can i say ? '' ellis unfolded his arms and widened his hands . `` your tax dollars at work . '' before farrell could respond , leila 's voice rose from inside the house . `` no insurance ? '' she wailed ", 
        "helen 's heart broke a little in the face of miss mabel 's selfless courage . she thought that because she was old , her life was of less value than the others ' . for all helen knew , miss mabel had a lot more years to live than she did ", 
        "helen 's heart broke a little in the face of miss mabel 's selfless courage . she thought that because she was old , her life was of less value than the others ' . for all helen knew , miss mabel had a lot more years to live than she did ", 
        "preston had been the last person to wear those chains , and i knew what i 'd see and feel if they were slipped onto my skin-the reaper 's unending hatred of me . i 'd felt enough of that emotion already in the amphitheater . i did n't want to feel anymore . `` do n't put those on me , '' i whispered . `` please ", 
        "preston had been the last person to wear those chains , and i knew what i 'd see and feel if they were slipped onto my skin-the reaper 's unending hatred of me . i 'd felt enough of that emotion already in the amphitheater . i did n't want to feel anymore . `` do n't put those on me , '' i whispered . `` please ", 
        "she knew that basha was a decent young man , that he was pretty sweet and friendly with her . jawen knew they had a bit of a history , but she thought that this time she would get along better with him , that she could overlook those problems . they kissed , and she knew that she liked basha , but then hastin interfered ", 
        "she knew that basha was a decent young man , that he was pretty sweet and friendly with her . jawen knew they had a bit of a history , but she thought that this time she would get along better with him , that she could overlook those problems . they kissed , and she knew that she liked basha , but then hastin interfered ", 
        "he heard rhinna speak `` the queen wants you in her carriage . '' tom spoke `` no , i 'm not going in some asylum . '' ran was seen standing next to him spoke `` it 's just for a private talk with you that 's all ", 
        "he heard rhinna speak `` the queen wants you in her carriage . '' tom spoke `` no , i 'm not going in some asylum . '' ran was seen standing next to him spoke `` it 's just for a private talk with you that 's all ", 
        "there was no way he would come here on his own . he ordered a cup of coffee , and then we just sat in silence . `` so , '' aidan finally said , `` how 's it going ? '' i laughed . `` not much has changed since the last time i saw you ", 
        "there was no way he would come here on his own . he ordered a cup of coffee , and then we just sat in silence . `` so , '' aidan finally said , `` how 's it going ? '' i laughed . `` not much has changed since the last time i saw you ", 
        "`` why ? '' `` i would have thought you 'd find him rather dry , '' she said . `` i do n't know about that , '' said gabriel . `` he was a great craftsman , '' said heather . `` that he was , '' said flannery ", 
        "`` why ? '' `` i would have thought you 'd find him rather dry , '' she said . `` i do n't know about that , '' said gabriel . `` he was a great craftsman , '' said heather . `` that he was , '' said flannery ", 
        "escorting drunk humans out of the bar is different from going up against a tiger-wildcat who eats raw steak for breakfast and is dying for a fight . '' `` i bet he could win with just his breath , '' ronan said . sean chuckled . `` take it seriously , ronan . these guys are seasoned ", 
        "both its sun-speckled shade and the cool grass beneath were a welcome respite after the stifling kitchen , and i was glad to relax against the tree 's rough , brittle bark and begin my breakfast of buttery , toasted bread and fresh fruit . even the water was tasty , it was so clean and cold "
    ], 
    "sentence2": [
        " but as i stare at harlin , his mouth curved in a confident grin , i do n't care about signs", 
        " `` i can make it there on my own , shane", 
        " `` i can make it there on my own , shane", 
        " `` what do you mean you do n't have any insurance", 
        " `` what do you mean you do n't have any insurance", 
        " `` not going to happen , '' replied helen", 
        " `` not going to happen , '' replied helen", 
        " '' sergei looked at me , surprised by my low , raspy please , but he put down the chains", 
        " '' sergei looked at me , surprised by my low , raspy please , but he put down the chains", 
        " she was so angry that she immediately said , once they were out of earshot of basha , `` you do n't mean anything to me anymore , hastin", 
        " she was so angry that she immediately said , once they were out of earshot of basha , `` you do n't mean anything to me anymore , hastin", 
        " '' tom groaned and went inside the carriage to sit down next to the queen", 
        " '' tom groaned and went inside the carriage to sit down next to the queen", 
        " '' `` ya know , you eat here a lot , '' said aidan", 
        " '' `` ya know , you eat here a lot , '' said aidan", 
        " `` and polish , to boot , '' said gabriel", 
        " `` and polish , to boot , '' said gabriel", 
        " it almost made up for the lack of coffee", 
        " if marquez has a champion , it means he 's won a good share of the fights", 
        " if marquez has a champion , it means he 's won a good share of the fights"
    ],
    "labels": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

eval_dataset = Dataset.from_dict(dataset)
if args.sample is not None:
    eval_dataset = eval_dataset[args.sample]
else:
    sample_id = torch.randperm(20).tolist()[:5]
    eval_dataset = eval_dataset[sample_id]
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
if not args.quantized and args.w_bit is not None:
    quantizer=BERTQuantizer(w_bit=args.w_bit,a_bit=args.a_bit,w_group_size=args.w_group_size)
    model = quantizer(model)
    print(model)
    if args.output_path:
        model.save_pretrained(args.output_path, safe_serialization=True)
        # enc.torch_dtype="float16"
        enc.save_pretrained(args.output_path, safe_serialization=True)
model.eval()

if args.eval:
    with torch.no_grad():
        print("randomly selected ids: ", sample_id)
        for batch in tqdm(eval_dataloader):
            logits = model(**batch)[1]
            pred = torch.argmax(logits, dim=-1)
            print("input: ", repr(enc.batch_decode(batch["input_ids"])[0]))
            if pred == batch["labels"]:
                print("predicted: ", "no" if pred.item() else "yes", ", groudtruth: ", "no" if batch["labels"].item() else "yes", ", correct\n")
            else:
                print("predicted: ", "no" if pred.item() else "yes", ", groudtruth: ", "no" if batch["labels"].item() else "yes", ", wrong\n")


# first gen quanted model
# CUDA_VISIBLE_DEVICES=6 python eval_compress_bert_support_3090.py --model_path $MODEL_BERT_PATH --w_bit 4 --lut_path /share/huangshan/masks/bert_large_lut.pt --output_path quantized_model/bert

# then run
# CUDA_VISIBLE_DEVICES=6 python eval_compress_bert_support_3090.py --model_path quantized_model/bert --w_bit 4 --lut_path /share/huangshan/masks/bert_large_lut.pt --quantized

