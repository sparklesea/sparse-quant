
from transformers import AutoTokenizer
from module.llama.modeling_llama import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("/share/yuzhen/llama2-chinese/")
# model = model.to("cuda")
print(model)
tokenizer = AutoTokenizer.from_pretrained("/share/yuzhen/llama2-chinese/")
prompt = "北京有什么好玩的地方？"

prompt = f"### Instruction:{prompt.strip()}  ### Response:"
inputs = tokenizer(prompt, return_tensors="pt")
print(prompt)
generate_ids = model.generate(inputs.input_ids, do_sample=True, max_new_tokens=512, top_k=10, top_p=0.85, temperature=1, repetition_penalty=1.15, eos_token_id=2, bos_token_id=1, pad_token_id=0)
print("model forward pass!")
response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
# response = response.lstrip(prompt)
print(response)


