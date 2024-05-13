import torch
import time
import os
from transformers import BertTokenizer, BertModel, AutoTokenizer, BertForNextSentencePrediction

tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
model = BertForNextSentencePrediction.from_pretrained(os.environ.get('MODEL_BERT_PATH'), torch_dtype=torch.float32).to("cuda")

# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
# model = BertModel.from_pretrained("bert-large-uncased").to("cuda")

for prefill_size in [8, 16, 32, 64, 128]:
    words = ["life" for _ in range(prefill_size)]
    
    text = " ".join(words)
    encoded_input = tokenizer(text, return_tensors='pt').to("cuda")
    # print(encoded_input.input_ids.shape)
    output = model(**encoded_input)

    warmup = 10
    freq = 200

    for _ in range(warmup):
        output = model(**encoded_input)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(freq):
        output = model(**encoded_input)
    torch.cuda.synchronize()
    end = time.time()

    query_latency = end - start
    query_latency /= freq
    query_latency *= 1000

    print("prefill_size: ", prefill_size, "latency: ", query_latency)