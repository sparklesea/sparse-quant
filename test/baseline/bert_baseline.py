from transformers import BertTokenizer, BertModel
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path of the hf model")
args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained('bert-large-uncased').to("cuda")

for seq_len in [8, 16, 32, 64, 128]:
    input_ids = torch.randint(1, 10000, (1, seq_len), dtype=torch.long, device="cuda")
    # text = "hello" * seq_len
    # print(text)
    # input_ids = toenizer

    warmup = 5
    for i in range(warmup):
        result = model(**input_ids)

    repeat = 10
    time = 0
    for i in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        result = model(**input_ids)
        torch.cuda.synchronize()
        end.record()
        time += start.elapsed_time(end)

    time /= repeat
    print(f"{seq_len} prefill time:", time)
