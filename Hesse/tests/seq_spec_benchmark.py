import argparse
import time
import torch
import torch.distributed as dist
import numpy as np
import os
import sys
sys.path.append("..")
import torch.distributed as dist
from Hesse.Engine.utils import initialized_dist_spec, args_parse_spec, make_causal_mask, sample, setup_seed, convert_dataset
from Hesse.Engine.pipleline import LLM_Pipeline
from transformers import LlamaTokenizer, DataCollatorForLanguageModeling
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm



args=args_parse_spec()
target_pp_config, draft_pp_config, target_last_stage_rank0, draft_last_stage_rank0 = initialized_dist_spec(args)
# print(args)
# print("="*80)
# print(target_pp_config,draft_pp_config)
global_rank=dist.get_rank()
setup_seed(args.seed)

top_k=args.top_k
top_p=args.top_p
temperature=args.temperature
spec_depth = args.depth
MAX_LEN = args.M
TARGET_MODEL_NAME = args.target_model
DRAFT_MODEL_NAME = args.draft_model
DTYPE = torch.float16
# DEVICE = torch.device("cuda", 0)
DEVICE = torch.device("cuda", global_rank)

draft_engine = LLM_Pipeline(max_length=MAX_LEN, model_name=DRAFT_MODEL_NAME, device=DEVICE, pp_config=draft_pp_config, type="spec", last_stage_rank_0=target_last_stage_rank0, cg_list=[2,128])
target_engine = LLM_Pipeline(max_length=MAX_LEN, model_name=TARGET_MODEL_NAME, device=DEVICE, pp_config=target_pp_config, type="spec", last_stage_rank_0=draft_last_stage_rank0, cg_list=[1,2,128])

tokenizer = LlamaTokenizer.from_pretrained(TARGET_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenized_dataset_eval = convert_dataset(tokenizer=tokenizer,file_path="../dataset/c4_small.json").select(list(range(args.start,args.end)))
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
dataloader = DataLoader(tokenized_dataset_eval, batch_size=1, collate_fn=data_collator, shuffle=False)
num_eval_steps = len(dataloader)
for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
    dist.barrier()
    input_ids = batch['input_ids'][..., :128].to(DEVICE)
    labels = batch['labels'][..., :128]
    terminate = False
    if labels[0][-1] == -100: continue
    prompt_len = input_ids.size(1)
    target_seq_offset=prompt_len
    draft_seq_offset=prompt_len

    attention_mask = make_causal_mask((MAX_LEN, MAX_LEN), dtype=DTYPE, device=DEVICE)
    attention_mask = attention_mask[None, None, :, :]
    position_ids = torch.arange(prompt_len, device=DEVICE).unsqueeze(0)
    prefix_storage_ids = torch.arange(prompt_len, device=DEVICE)

    # Prefill
    logits = target_engine.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., :prompt_len,:], storage_ids=prefix_storage_ids)
    draft_engine.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., :prompt_len,:], storage_ids=prefix_storage_ids)

    dist.barrier()
    torch.cuda.synchronize()

    t1 = time.time()

    bonus_token=sample(logits[:,-1,:], top_k=top_k, top_p=top_p, temperature=temperature).view(1,-1)

    output = bonus_token.clone()
    accepted_len = None
    itr=0
    all_accept=0

    while output.size(1)<128 and terminate == False:
        itr+=1
        # Run Draft Model to Speculate
        draft_next_token = bonus_token.clone()
        draft_output = draft_next_token.clone()[:,-1].unsqueeze(0)
        for i in range(spec_depth):
            input_ids=draft_next_token
            storage_ids = torch.arange(input_ids.size(1), device=DEVICE) + draft_seq_offset
            position_ids = storage_ids.view(1,-1)
            logits = draft_engine.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., draft_seq_offset:draft_seq_offset+input_ids.size(1),:], storage_ids=storage_ids)
            draft_next_token=sample(logits[:,-1,:], 1, 0.9, 0.6).view(1,-1)
            draft_output = torch.cat((draft_output, draft_next_token),dim=-1)
            draft_seq_offset+=input_ids.size(1)

        # Run Target Model to Verify
        input_ids=draft_output
        storage_ids = torch.arange(input_ids.size(1), device=DEVICE)+target_seq_offset
        position_ids = storage_ids.view(1,-1)
        logits = target_engine.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., target_seq_offset:target_seq_offset+input_ids.size(1),:], storage_ids=storage_ids)
        accepted_len = 0

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
        
    torch.cuda.synchronize()
    t2=time.time()
    draft_engine.clear_kv()
    target_engine.clear_kv()
    # torch.cuda.synchronize()
    # t2=time.time()
    # print(tokenizer.decode(output[0]), (t2-t1)/(seq_offset-PREFIX_LEN))
    if global_rank == 0:
        print(tokenizer.decode(output[0]))
        print(all_accept/itr)
        print((t2-t1)/output.size(1))
    # print(tokenizer.decode(draft_output[0]))