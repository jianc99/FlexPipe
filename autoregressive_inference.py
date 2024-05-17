import argparse
import time
import torch
import numpy as np
import os
import torch.distributed as dist
from pipe.utils import initialized_dist, args_parse, make_causal_mask, sample, setup_seed
from pipe.pipleline import LLM_Pipeline
from transformers import LlamaTokenizer



args=args_parse()
pp_config=initialized_dist(args.tp_groups,args.layer_partition)
print(args)
print("="*80)
print(pp_config)
global_rank=dist.get_rank()

setup_seed(123)

MAX_LEN = args.M
DEC_LEN = args.D
MODEL_NAME = args.model
DTYPE = torch.float16
DEVICE = torch.device("cuda", 0)
# DEVICE = torch.device("cuda", global_rank)
T = args.T
WARM_UP = 10
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")

prompt="Pittsburgh is a city located in "



engine = LLM_Pipeline(max_length=MAX_LEN, model_name=args.model, device=DEVICE, pp_config=pp_config)

input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
PREFIX_LEN = input_ids.size(1)

attention_mask = make_causal_mask((MAX_LEN, MAX_LEN), dtype=DTYPE, device=DEVICE)
attention_mask = attention_mask[None, None, :, :]
position_ids = torch.arange(PREFIX_LEN, device=DEVICE).unsqueeze(0)
prefix_storage_ids = torch.arange(PREFIX_LEN, device=DEVICE)
logits = engine.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., :PREFIX_LEN,:], storage_ids=prefix_storage_ids)
seq_offset=PREFIX_LEN

# dist.broadcast_object_list

dist.barrier()
torch.cuda.synchronize()
if global_rank==0:
    t1 = time.time()
    next_token=sample(logits[:,-1,:], 1, 0.9, 0.6).view(1,-1)
    output = next_token.clone()
    while output.size(1)<128:
        control_tensor = torch.tensor([1], device=DEVICE)
        dist.broadcast(control_tensor,0)
        input_ids=next_token
        position_ids = torch.full((1,1),seq_offset, device=DEVICE)
        storage_ids = torch.tensor(seq_offset, device=DEVICE)
        logits = engine.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., seq_offset,:], storage_ids=storage_ids)
        next_token=sample(logits[:,-1,:], 1, 0.9, 0.6).view(1,-1)
        output = torch.cat((output, next_token),dim=-1)
        seq_offset+=1
    control_tensor = torch.tensor([-1], device=DEVICE)
    dist.broadcast(control_tensor,0)
    torch.cuda.synchronize()
    t2=time.time()
    print(tokenizer.decode(output[0]), (t2-t1)/(output.size(1)))

else:
    control_tensor = torch.tensor([-1], device=DEVICE)
    while True:
        dist.broadcast(control_tensor,0)
        if control_tensor.item() == -1:
            break
        else:
            input_ids=torch.full((1,1),0, device=DEVICE)
            position_ids = torch.full((1,1),seq_offset, device=DEVICE)
            storage_ids = torch.tensor(seq_offset, device=DEVICE)
            engine.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., seq_offset,:], storage_ids=storage_ids)
            seq_offset+=1