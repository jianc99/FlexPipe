import argparse
import time
import torch
import torch.distributed as dist
import numpy as np
import os
import torch.distributed as dist
from spec.utils import initialized_dist, args_parse, make_causal_mask, sample, setup_seed, convert_dataset
from spec.pipleline import LLM_Pipeline

from transformers import LlamaTokenizer, DataCollatorForLanguageModeling
from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator
from tqdm import tqdm



args=args_parse()
target_pp_config, draft_pp_config = initialized_dist(args)
print(args)
print("="*80)
print(target_pp_config,draft_pp_config)
global_rank=dist.get_rank()

setup_seed(args.seed)

top_k=args.top_k
top_p=args.top_p
temperature=args.temperature

spec_depth = args.depth

MAX_LEN = args.M
DEC_LEN = args.D
TARGET_MODEL_NAME = args.target_model
DRAFT_MODEL_NAME = args.draft_model
DTYPE = torch.float16
DEVICE = torch.device("cuda", 0)
# DEVICE = torch.device("cuda", global_rank)
# T = args.T
# WARM_UP = 10

# prompt="I love Pittsburgh, it is a great city"
target_engine=None
draft_engine=None

if draft_pp_config!=None:
    draft_engine = LLM_Pipeline(max_length=MAX_LEN, model_name=DRAFT_MODEL_NAME, device=DEVICE, pp_config=draft_pp_config)

if target_pp_config!= None:
    target_engine = LLM_Pipeline(max_length=MAX_LEN, model_name=TARGET_MODEL_NAME, device=DEVICE, pp_config=target_pp_config)

dist.barrier()
torch.cuda.synchronize()
# time.sleep(100)
if global_rank == 0:
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_dataset_eval = convert_dataset(tokenizer=tokenizer,file_path="../dataset/c4_small.json").select(list(range(0,20)))
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    dataloader = DataLoader(tokenized_dataset_eval, batch_size=1, collate_fn=data_collator, shuffle=False)
    accelerator = Accelerator()
    dataloader = accelerator.prepare(dataloader)
    num_eval_steps = len(dataloader)
    for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
        input_ids = batch['input_ids'][..., :128]
        labels = batch['labels'][..., :128]
        terminate = False
        if labels[0][-1] == -100: continue


prompt = "Pittsburgh is a city located in "
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
PROMPT_LEN = input_ids.size(1)
target_seq_offset=PROMPT_LEN
draft_seq_offset=PROMPT_LEN

attention_mask = make_causal_mask((MAX_LEN, MAX_LEN), dtype=DTYPE, device=DEVICE)
attention_mask = attention_mask[None, None, :, :]
position_ids = torch.arange(PROMPT_LEN, device=DEVICE).unsqueeze(0)
prefix_storage_ids = torch.arange(PROMPT_LEN, device=DEVICE)
if target_pp_config!=None:
    logits = target_engine.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., :PROMPT_LEN,:], storage_ids=prefix_storage_ids)
if draft_pp_config!=None:
    draft_engine.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., :PROMPT_LEN,:], storage_ids=prefix_storage_ids)

if global_rank==0:
    t1 = time.time()

    bonus_token=sample(logits[:,-1,:], top_k=top_k, top_p=top_p, temperature=temperature).view(1,-1)

    output = bonus_token.clone()
    accepted_len = None
    itr=0
    all_accept=0

    while output.size(1)<128:
        itr+=1
        # Run Draft Model to Speculate
        control_tensor = torch.tensor([2,draft_seq_offset,bonus_token.size(1)], device=DEVICE)
        dist.broadcast(control_tensor,0)

        draft_next_token = bonus_token.clone()
        draft_output = draft_next_token.clone()[:,-1].unsqueeze(0)
        for i in range(spec_depth):
            input_ids=draft_next_token

            # print("Draft Input:")
            # print(input_ids, tokenizer.decode(input_ids[0]))

            storage_ids = torch.arange(input_ids.size(1), device=DEVICE) + draft_seq_offset
            position_ids = storage_ids.view(1,-1)
            logits = draft_engine.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., draft_seq_offset:draft_seq_offset+input_ids.size(1),:], storage_ids=storage_ids)
            draft_next_token=sample(logits[:,-1,:], 1, 0.9, 0.6).view(1,-1)
            draft_output = torch.cat((draft_output, draft_next_token),dim=-1)
            draft_seq_offset+=input_ids.size(1)

        # print("Draft Output:")
        # print(tokenizer.decode(draft_output[0]))

        # Run Target Model to Verify
        control_tensor = torch.tensor([1, target_seq_offset, 0], device=DEVICE)
        dist.broadcast(control_tensor,0)
        input_ids=draft_output
        storage_ids = torch.arange(input_ids.size(1), device=DEVICE)+target_seq_offset
        position_ids = storage_ids.view(1,-1)
        logits = target_engine.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., target_seq_offset:target_seq_offset+input_ids.size(1),:], storage_ids=storage_ids)
        accepted_len = 0

        # print("Target Seq:")
        # for i in range(input_ids.size(1)):
        #     target_token = sample(logits[:, i, :], top_k=top_k, top_p=top_p, temperature=temperature)
        #     print(tokenizer.decode(target_token),target_token)

        for i in range(input_ids.size(1)-1):
            target_token = sample(logits[:, i, :], top_k=top_k, top_p=top_p, temperature=temperature)
            if target_token != input_ids[0, i+1]:
                bonus_token = target_token.view(1,-1)
                output = torch.cat((output, bonus_token),dim=-1)
                break
            else:
                accepted_len+=1
                output = torch.cat((output, target_token.view(1,-1)),dim=-1)
        if accepted_len == spec_depth:
            target_token =  sample(logits[:, -1, :], top_k=top_k, top_p=top_p, temperature=temperature).view(1,-1)
            output = torch.cat((output, target_token),dim=-1)
            bonus_token = output[:,-2:]
        else:
            draft_seq_offset = draft_seq_offset - spec_depth + 1 + accepted_len
        target_seq_offset+=1+accepted_len

        all_accept+=accepted_len
        # print(accepted_len, draft_seq_offset, target_seq_offset)
        
    # torch.cuda.synchronize()
    # t2=time.time()

    control_tensor = torch.tensor([-1,0,0], device=DEVICE)
    dist.broadcast(control_tensor,0)
    # torch.cuda.synchronize()
    # t2=time.time()
    # print(tokenizer.decode(output[0]), (t2-t1)/(seq_offset-PREFIX_LEN))
    print(tokenizer.decode(output[0]))
    print(all_accept/itr)
    # print((t2-t1)/output.size(1))
    # print(tokenizer.decode(draft_output[0]))

else:
    control_tensor = torch.tensor([-1,0,0], device=DEVICE)
    while True:
        dist.broadcast(control_tensor,0)
        if control_tensor[0] == -1:
            break
        # Run Target Model
        elif control_tensor[0] == 1 and target_pp_config!=None:
            target_seq_offset = control_tensor[1]
            input_ids=torch.full((1,1+spec_depth),0, device=DEVICE)
            storage_ids = torch.arange(spec_depth+1, device=DEVICE)+target_seq_offset
            position_ids = storage_ids.view(1,-1)
            target_engine.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., target_seq_offset:target_seq_offset+input_ids.size(1),:], storage_ids=storage_ids)
            
        # Run Draft Model
        elif control_tensor[0] == 2 and draft_pp_config!=None:
            input_size = control_tensor[2]
            draft_seq_offset = control_tensor[1]
            for i in range(spec_depth):
                input_ids=torch.full((1,input_size),0, device=DEVICE)
                storage_ids = torch.arange(input_ids.size(1), device=DEVICE) + draft_seq_offset
                position_ids = storage_ids.view(1,-1)
                draft_engine.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., draft_seq_offset:draft_seq_offset+input_ids.size(1),:], storage_ids=storage_ids)
                draft_seq_offset+=input_size
                input_size=1