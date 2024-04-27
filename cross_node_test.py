from llm_dist import LLMEngine
import argparse
import time
import torch
import torch.distributed as dist
import numpy as np
import os
import torch.distributed as dist

rank = int(os.environ["RANK"])
world_size= int(os.environ["WORLD_SIZE"])

def forward(input):
    local_rank=dist.get_rank()+1
    return input*local_rank

dist.init_process_group(backend='nccl',init_method='tcp://172.24.46.47:9991',world_size=world_size,rank=rank)
dist.barrier()

group1=dist.new_group([0])
group2=dist.new_group([1])
local_rank=dist.get_rank()

# if local_rank in [0]:
#     print(dist.get_world_size(group1),dist.get_world_size())
#     print(dist.get_rank(group1),dist.get_rank())
# elif local_rank in [1,2]:
#     print(dist.get_world_size(group2),dist.get_world_size())
#     print(dist.get_rank(group2),dist.get_rank())

DEVICE=torch.device("cuda", 0)
input=torch.tensor([10],device=DEVICE)
output=torch.tensor([0],device=DEVICE)
hidden=torch.tensor([0],device=DEVICE)

if local_rank in [0]:
    hidden=forward(input)
    print(hidden,local_rank)
    if local_rank == 0:
        dist.send(hidden,1)
    dist.broadcast(output,1)

if local_rank in [1]:
    dist.recv(hidden,0)
    print(hidden,local_rank)
    hidden=forward(hidden)
    print(hidden,local_rank)
    output=hidden
    dist.broadcast(output,1)
print(output,local_rank)
