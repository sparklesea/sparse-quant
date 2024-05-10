from transformers import pipeline, BertTokenizer
import torch

unmasker = pipeline('fill-mask', model='bert-large-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

for seq_len in [8, 128]:
    input_ids = torch.randint(1, 10000, (1, seq_len), dtype=torch.long, device="cuda")
    input_ctx_list = tokenizer.batch_decode(input_ids)
    input_ctx=''
    for i in range(len(input_ctx_list)):
        input_ctx += input_ctx_list[i]
        input_ctx += ' '
    input_ctx += '[MASK]'

    warmup = 50
    for i in range(warmup):
        result = unmasker(input_ctx)

    repeat = 1000
    time = 0
    for i in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        result = unmasker(input_ctx)
        torch.cuda.synchronize()
        end.record()
        time += start.elapsed_time(end)

    time /= repeat
    print(f"{seq_len} prefill time:", time)


# from transformers import AutoTokenizer, BertGenerationDecoder, BertGenerationConfig
# import torch

# tokenizer = AutoTokenizer.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
# config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
# config.is_decoder = True
# model = BertGenerationDecoder.from_pretrained(
#     "google/bert_for_seq_generation_L-24_bbc_encoder", config=config
# )

# inputs = tokenizer("Hello, my dog is cute", return_token_type_ids=False, return_tensors="pt")

# print(inputs)

# outputs = model(**inputs)

# prediction_logits = outputs.logits

# print(prediction_logits.size())

# outputs = torch.argmax(prediction_logits,-1)

# print(outputs)
# # outputs = tokenizer.convert_ids_to_tokens(outputs[0])

# outputs = tokenizer.decode(outputs[0])

# print(outputs)
