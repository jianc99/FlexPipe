import time
import torch
import numpy as np
import sys
sys.path.append("..")
import torch.distributed as dist
from Hesse.Engine.utils import initialized_dist_baseline, args_parse_baseline, make_causal_mask, sample, setup_seed, convert_dataset
from Hesse.Engine.pipleline import LLM_Pipeline
from transformers import LlamaTokenizer, DataCollatorForLanguageModeling
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm



args=args_parse_baseline()
pp_config=initialized_dist_baseline(args.tp_groups,args.layer_partition)
print("="*80)
print(pp_config)
global_rank=dist.get_rank()
if global_rank == 0:
    print(args)

setup_seed(args.seed)

MAX_LEN = args.M
MODEL_NAME = args.model
DTYPE = torch.float16
# DEVICE = torch.device("cuda", 0)
DEVICE = torch.device("cuda", global_rank)

engine = LLM_Pipeline(max_length=MAX_LEN, model_name=MODEL_NAME, device=DEVICE, pp_config=pp_config, type = "baseline", cg_list=[1,128])

tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenized_dataset_eval = convert_dataset(tokenizer=tokenizer,file_path="../dataset/c4_small.json").select(list(range(args.start,args.end)))
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
dataloader = DataLoader(tokenized_dataset_eval, batch_size=1, collate_fn=data_collator, shuffle=False)
num_eval_steps = len(dataloader)
for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
    input_ids = batch['input_ids'][..., :128].to(DEVICE)
    labels = batch['labels'][..., :128]
    terminate = False
    if labels[0][-1] == -100: continue
    prefix_len = input_ids.size(1)
    attention_mask = make_causal_mask((MAX_LEN, MAX_LEN), dtype=DTYPE, device=DEVICE)
    attention_mask = attention_mask[None, None, :, :]
    position_ids = torch.arange(prefix_len, device=DEVICE).unsqueeze(0)
    prefix_storage_ids = torch.arange(prefix_len, device=DEVICE)
    logits = engine.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., :prefix_len,:], storage_ids=prefix_storage_ids)
    seq_offset=prefix_len

    dist.barrier()
    torch.cuda.synchronize()
    t1 = time.time()
    next_token=sample(logits[:,-1,:], args.top_k, args.top_p, args.temperature).view(1,-1)
    output = next_token.clone()
    while output.size(1)<128 and terminate == False:
        input_ids=next_token
        position_ids = torch.full((1,1),seq_offset, device=DEVICE)
        storage_ids = torch.tensor(seq_offset, device=DEVICE)
        logits = engine.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., seq_offset,:], storage_ids=storage_ids)
        next_token=sample(logits[:,-1,:], args.top_k, args.top_p, args.temperature).view(1,-1)
        output = torch.cat((output, next_token),dim=-1)
        seq_offset+=1
        if next_token[0] == 2 or next_token[0] == 0: terminate = True
    torch.cuda.synchronize()
    t2=time.time()
    if global_rank == 0:
        print(tokenizer.decode(output[0]), (t2-t1)/(output.size(1)))