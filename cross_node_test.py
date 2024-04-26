from llm_dist import LLMEngine
import argparse
import time
import torch
import torch.distributed as dist
import numpy as np
import os
import torch.distributed as dist


# gpu_id = int(os.environ["LOCAL_RANK"])
# global_rank = int(os.environ["RANK"])


# print(gpu_id,global_rank)
# print(os.environ)
def forward(input):
    local_rank=dist.get_rank()
    return input*local_rank

dist.init_process_group(backend='nccl')
dist.barrier()

group1=dist.new_group([0])
group2=dist.new_group([1,2])

local_rank=dist.get_rank()
if local_rank in [0]:
    print(dist.get_world_size(group1),dist.get_world_size())
    print(dist.get_rank(group1),dist.get_rank())
elif local_rank in [1,2]:
    print(dist.get_world_size(group2),dist.get_world_size())
    print(dist.get_rank(group2),dist.get_rank())

DEVICE=torch.device("cuda", 0)
input=torch.tensor([10],device=DEVICE)
output=torch.tensor([0],device=DEVICE)
hidden=torch.tensor([0],device=DEVICE)
print(input)

if local_rank in [0]:
    hidden=forward(input)
    print(hidden)
    dist.send(hidden,1)
    dist.broadcast(output,2)

if local_rank in [1,2]:
    if local_rank==1:
        dist.recv(hidden,0)
    dist.broadcast(hidden,2,group2)
    print(hidden)
    hidden=forward(hidden)
    print(hidden)
    dist.all_reduce(hidden,group=group2)
    print(hidden)
    output=hidden
    dist.broadcast(output,2)
print(output)
