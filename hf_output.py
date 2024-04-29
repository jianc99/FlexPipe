import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

PREFIX_LEN = 128
DEVICE = 'cuda:0'

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(DEVICE)

input_ids = torch.full((1, PREFIX_LEN), 11, device=DEVICE)

# Generate logits
with torch.no_grad():
    outputs = model(input_ids=input_ids)
    logits = outputs.logits

print(logits)