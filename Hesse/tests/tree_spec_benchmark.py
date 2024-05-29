import sys
sys.path.append("..")
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
import torch.distributed as dist
import torch
import numpy as np 
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import argparse
from Hesse.Tree.GreedyTree import GreedySTree
import time
from Hesse.Tree.utils import cuda_graph_for_sampling_argmax
from Hesse.Engine.pipleline import LLM_Pipeline
from Hesse.Engine.utils import initialized_dist_spec, setup_seed
from Hesse.Data.data_converter import convert_wiki_dataset, convert_cnn_dataset, convert_c4_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="meta-llama/Llama-2-7b-hf", type=str, help='model')
parser.add_argument('--target', default="meta-llama/Llama-2-70b-hf", type=str, help='target model')
parser.add_argument('--dataset', type=str, default="cnn", help='dataset path')
parser.add_argument('--growmap', type=str, default="demo_tree.pt", help='growmap path')
parser.add_argument('--start', type=int, default=0, help='start')
parser.add_argument('--end', type=int, default=200, help='end')
parser.add_argument('--T', type=float, default=0.6, help='temperature')
parser.add_argument('--P', type=float, default=0.9, help='top_p')
parser.add_argument('--M', type=int, default=256, help='max length')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--Mode', type=str, default="benchmark")
# Target model information
parser.add_argument('--target_layer_partition', nargs='+', type=int, help='Layer partitioning as a list of integers.')
parser.add_argument('--target_tp_groups', nargs='+', type=int, help='TP groups as a list of integers.')
parser.add_argument('--target_group', nargs='+', type=int, help='Target group of ranks')
# Draft model information
parser.add_argument('--draft_layer_partition', nargs='+', type=int, help='Layer partitioning as a list of integers.')
parser.add_argument('--draft_tp_groups', nargs='+', type=int, help='TP groups as a list of integers.')
parser.add_argument('--draft_group', nargs='+', type=int, help='Target group of ranks')
args = parser.parse_args()
# print(args)
setup_seed(args.seed)
target_pp_config, draft_pp_config, target_last_stage_rank0, draft_last_stage_rank0 = initialized_dist_spec(args)
global_rank=dist.get_rank()


def simulation_fast(target_model : LLM_Pipeline, draft_model: LLM_Pipeline, dataloader: DataLoader, T=0.6, top_p=0.9, 
            max_length=512, grow_map=None, sampling_callables = None,
            sample_gather_indices = None):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    num_large_model_steps = 0
    total_time = 0.0
    dtype = torch.float16
    attn_mask = torch.full((max_length, max_length), torch.finfo(dtype).min, dtype=dtype, device=DEVICE)
    sequence = torch.tensor(list(range(max_length)), device=DEVICE).long().unsqueeze(-1)
    new_tokens_buffer =  torch.zeros(max_length).long().to(DEVICE)
    parents_buffer =  torch.zeros(max_length).long().to(DEVICE)
    position_ids = torch.zeros(max_length).long().to(DEVICE)
    
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            input_ids = batch['input_ids'][..., :128]
            labels = batch['labels'][..., :128]
            terminate = False
            if labels[0][-1] == -100: terminate = True
            draft_kv_len = 0
            target_kv_len = 0
            attn_mask.fill_(torch.finfo(dtype).min)
            spectree = GreedySTree(prefix=input_ids.squeeze(0), device=DEVICE, temperature=T,
                                    top_p=top_p,
                                    draft_kv_len=draft_kv_len, target_kv_len=target_kv_len,
                                    draft_model_engine=draft_model, target_model_engine=target_model, max_length=max_length, grow_map=grow_map,
                                    attn_mask = attn_mask, sequence = sequence, new_tokens_buffer = new_tokens_buffer, 
                                    parents_buffer = parents_buffer, 
                                    position_ids = position_ids,
                                    sampling_callables=sampling_callables,
                                    sample_gather_indices = sample_gather_indices)
            torch.cuda.synchronize()
            t1 = time.time()
            while input_ids.shape[1] < 256 and terminate == False:
                spectree.construct_grow_map()
                valid_tokens, draft_kv_len, target_kv_len,terminate = spectree.verify()
                
                num_decoding_steps += (valid_tokens.shape[0] - input_ids.shape[1])
                num_large_model_steps += 1
                input_ids = valid_tokens.unsqueeze(0)
                if (input_ids[0][-1] == 2) or (input_ids[0][-1] == 0): terminate = True
            
            torch.cuda.synchronize()
            t2 = time.time()
            total_time += (t2 - t1)
            draft_model.clear_kv()
            target_model.clear_kv()
            if global_rank == 0:
                print(tokenizer.decode(input_ids[0]))
                print("total time :{:.5f}s, latency :{:.5f}s, decoding step: {}, large model step: {}".format(total_time, total_time / num_decoding_steps, num_decoding_steps, num_large_model_steps))
    return num_decoding_steps / num_large_model_steps

def simulation_benchmark(target_model : LLM_Pipeline, draft_model: LLM_Pipeline, dataloader: DataLoader, T=0.6, top_p=0.9,  
                max_length=512, grow_map=None, sampling_callables = None,
                sample_gather_indices = None):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    num_large_model_steps = 0
    initialize_time = 0.0
    speculate_time = 0.0
    verify_time = 0.0
    large_model_run = 0.0
    accept_loop = 0.0
    kv_select = 0.0
    sample_time = 0.0
    small_model_compute = 0.0
    dtype = torch.float16
    attn_mask = torch.full((max_length, max_length), torch.finfo(dtype).min, dtype=dtype, device=DEVICE)
    sequence = torch.tensor(list(range(max_length)), device=DEVICE).long().unsqueeze(-1)
    new_tokens_buffer =  torch.zeros(max_length).long().to(DEVICE)
    parents_buffer =  torch.zeros(max_length).long().to(DEVICE)
    position_ids = torch.zeros(max_length).long().to(DEVICE)
    
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            input_ids = batch['input_ids'][..., :128]
            labels = batch['labels'][..., :128]
            terminate = False
            if labels[0][-1] == -100: terminate = True
            draft_kv_len = 0
            target_kv_len = 0
            attn_mask.fill_(torch.finfo(dtype).min)
            spectree = GreedySTree(prefix=input_ids.squeeze(0), device=DEVICE, temperature=T,
                                    top_p=top_p,
                                    draft_kv_len=draft_kv_len, target_kv_len=target_kv_len,
                                    draft_model_engine=draft_model, target_model_engine=target_model, max_length=max_length, grow_map=grow_map,
                                    attn_mask = attn_mask, sequence = sequence, new_tokens_buffer = new_tokens_buffer, 
                                    parents_buffer = parents_buffer, 
                                    position_ids = position_ids,
                                    sampling_callables=sampling_callables,
                                    sample_gather_indices = sample_gather_indices)
            while input_ids.shape[1] < 256 and terminate == False:
                torch.cuda.synchronize()
                t1 = time.time()
                torch.cuda.synchronize()
                t2 = time.time()
                a, b = spectree.construct_grow_map(benchmark=True)
                torch.cuda.synchronize()
                t3 = time.time()
                valid_tokens, draft_kv_len, target_kv_len, x, y, z, terminate = spectree.verify(benchmark=True)
                torch.cuda.synchronize()
                t4 = time.time()
                initial_size = input_ids.shape[1]
                input_ids = valid_tokens.unsqueeze(0)
                
                if (input_ids[0] == 2)._is_any_true() or (input_ids[0] == 0)._is_any_true() or input_ids.shape[1] >= 256: 
                    terminate = True
                if not terminate:
                    sample_time += a
                    small_model_compute += b
                    large_model_run += x
                    accept_loop += y
                    kv_select += z
                    initialize_time += (t2 - t1)
                    speculate_time += (t3 - t2)
                    verify_time += (t4 - t3)
                    num_decoding_steps += (valid_tokens.shape[0] - initial_size)
                    num_large_model_steps += 1
            draft_model.clear_kv()
            target_model.clear_kv()
            if num_large_model_steps > 0 and global_rank==0:
                print(num_decoding_steps / num_large_model_steps)
            if global_rank == 0:
                print("total decoding steps: {}".format(num_decoding_steps), "large model steps: {}".format(num_large_model_steps), "avg decoding step: {}".format(num_decoding_steps / num_large_model_steps))
                print("initialization time:{}".format(initialize_time / num_large_model_steps), "speculate time: {}".format(speculate_time / num_large_model_steps),  "verify time: {}".format(verify_time / num_large_model_steps))
                print("large model run: {}".format(large_model_run / num_large_model_steps) , "accept loop: {}".format(accept_loop / num_large_model_steps), "kv select: {}".format(kv_select / num_large_model_steps))
                print("small model run: {}".format(small_model_compute / num_large_model_steps) , "sample time: {}".format(sample_time / num_large_model_steps))
    return num_decoding_steps / num_large_model_steps


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
if args.dataset == 'wiki':
    tokenized_dataset_eval = convert_wiki_dataset(tokenizer=tokenizer).select(list(range(args.start, args.end)))
elif args.dataset == 'cnn':
    tokenized_dataset_eval = convert_cnn_dataset(tokenizer=tokenizer).select(list(range(args.start, args.end)))
else:
    tokenized_dataset_eval = convert_c4_dataset(tokenizer=tokenizer,file_path="../dataset/c4_small.json").select(list(range(args.start, args.end)))
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
dataloader = DataLoader(tokenized_dataset_eval, batch_size=1, collate_fn=data_collator, shuffle=False)

MAX_LEN = args.M
TARGET_MODEL_NAME = args.target
DRAFT_MODEL_NAME = args.model
DTYPE = torch.float16
DEVICE = torch.device("cuda", global_rank)

path = args.growmap
grow_map = torch.load(path)
tree_size = grow_map["size"]
idx_lists = grow_map["roots"]
branch_lists = grow_map['branches']
draft_step = len(grow_map["roots"])

sampling_callables = {}
sample_gather_indices = {}
for i in range(draft_step - 1):
    idx_len = len(idx_lists[i])
    num_samples = max(branch_lists[i])
    sampling_callables[i] = cuda_graph_for_sampling_argmax(device=DEVICE,
        max_length=args.M, idx_len=idx_len, num_samples=num_samples,
        temperature=args.T, tree_size=tree_size)  
for i in range(draft_step - 1):
    ith_gather_list = []
    max_num_samples = max(branch_lists[i])
    for j, branch in enumerate(branch_lists[i]):
        branch_index = torch.arange(branch, device=DEVICE, dtype=torch.long)
        branch_index = branch_index + j * max_num_samples
        ith_gather_list.append(branch_index)
    ith_gather_list = torch.cat(ith_gather_list)
    sample_gather_indices[i] = ith_gather_list

dist.barrier()

cg_list_target = [tree_size]
cg_list_draft = [sum(x) for x in branch_lists]
cg_list_draft.append(1)

draft_model =  LLM_Pipeline(max_length=MAX_LEN, model_name=DRAFT_MODEL_NAME, device=DEVICE, pp_config=draft_pp_config, type="spec", last_stage_rank_0=draft_last_stage_rank0, cg_list=cg_list_draft)
target_model = LLM_Pipeline(max_length=MAX_LEN, model_name=TARGET_MODEL_NAME, device=DEVICE, pp_config=target_pp_config, type="spec", last_stage_rank_0=target_last_stage_rank0, cg_list=cg_list_target)

if args.Mode == "fast":
    simulation_fast(target_model=target_model, draft_model=draft_model, dataloader=dataloader, T=args.T, top_p=args.P,
                                     max_length=args.M, grow_map = grow_map, sampling_callables=sampling_callables, sample_gather_indices = sample_gather_indices)
else:
    simulation_benchmark(target_model=target_model, draft_model=draft_model, dataloader=dataloader, T=args.T, top_p=args.P,
                                               max_length=args.M, grow_map = grow_map, sampling_callables=sampling_callables, sample_gather_indices = sample_gather_indices)