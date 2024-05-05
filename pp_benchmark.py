import argparse
import time
import torch
import torch.distributed as dist
import numpy as np
import os
import torch.distributed as dist
from pipe.utils import initialized_dist, args_parse, make_causal_mask
from pipe.pipleline import LLM_Pipeline



args=args_parse()
pp_config=initialized_dist(args.tp_groups,args.layer_partition)
print(args)
print("="*80)
print(pp_config)
global_rank=dist.get_rank()

MAX_LEN = args.M
DEC_LEN = args.D
MODEL_NAME = args.model
DTYPE = torch.float16
# DEVICE = torch.device("cuda", 0)
DEVICE = torch.device("cuda", global_rank)
PREFIX_LEN=128
T = args.T
WARM_UP = 10

engine = LLM_Pipeline(max_length=MAX_LEN, model_name=args.model, device=DEVICE, pp_config=pp_config)
# input_ids = torch.randint(low=3, high=30000, size=(1, PREFIX_LEN), device=DEVICE)
input_ids = torch.full((1, PREFIX_LEN),11,device=DEVICE)
attention_mask = make_causal_mask((MAX_LEN, MAX_LEN), dtype=DTYPE, device=DEVICE)
attention_mask = attention_mask[None, None, :, :]
position_ids = torch.arange(PREFIX_LEN, device=DEVICE).unsqueeze(0)
prefix_storage_ids = torch.arange(PREFIX_LEN, device=DEVICE)
logits = engine.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., :PREFIX_LEN,:], storage_ids=prefix_storage_ids)
print(logits)

input_ids = torch.randint(low=3, high=30000, size=(1, DEC_LEN), device=DEVICE)
storage_ids = torch.arange(DEC_LEN, device=DEVICE) + PREFIX_LEN
position_ids = storage_ids.clone().unsqueeze(0)
attention_mask = attention_mask[..., PREFIX_LEN: PREFIX_LEN + DEC_LEN,:].clone()

for _ in range(WARM_UP):
    engine.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, storage_ids=storage_ids)

torch.cuda.synchronize()
t1 = time.time()
for _ in range(T):
    engine.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, storage_ids=storage_ids)
torch.cuda.synchronize()
t2 = time.time()
if dist.get_rank() == 0:
    print("Max Length :{}, Decode Length :{}, Prefix Length :{}, inference time:{}s".format(MAX_LEN, DEC_LEN, PREFIX_LEN, (t2 - t1)/ T))