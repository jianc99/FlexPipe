import os
import sys
sys.path.append("..")
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torch
import argparse
from Hesse.Engine.utils import initialized_dist_spec, setup_seed
from Hesse.Data.data_converter import convert_wiki_dataset, convert_cnn_dataset, convert_c4_dataset
from Hesse.Engine.pipleline import LLM_Pipeline
from Hesse.Tree.GreedyTree import GreedySTreeTest

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="meta-llama/Llama-2-7b-hf", type=str, help='model')
parser.add_argument('--target', default="meta-llama/Llama-2-70b-hf", type=str, help='target model')
parser.add_argument('--T', type=float, default=0.6, help='temperature')
parser.add_argument('--P', type=float, default=0.9, help='top_p')
parser.add_argument('--W', type=int, default=32, help='max width')
parser.add_argument('--M', type=int, default=288, help='max length')
parser.add_argument('--dataset', type=str, default="cnn", help='dataset path')
parser.add_argument('--start', type=int, default=0, help='start')
parser.add_argument('--end', type=int, default=200, help='end')
parser.add_argument('--dst', type=str, default="acceptance-rate-vector.pt", help='destination for accepetance rate vector')
# Target model information
parser.add_argument('--target_layer_partition', nargs='+', type=int, help='Layer partitioning as a list of integers.')
parser.add_argument('--target_tp_groups', nargs='+', type=int, help='TP groups as a list of integers.')
parser.add_argument('--target_group', nargs='+', type=int, help='Target group of ranks')
# Draft model information
parser.add_argument('--draft_layer_partition', nargs='+', type=int, help='Layer partitioning as a list of integers.')
parser.add_argument('--draft_tp_groups', nargs='+', type=int, help='TP groups as a list of integers.')
parser.add_argument('--draft_group', nargs='+', type=int, help='Target group of ranks')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
args = parser.parse_args()
# print(args)

def simulation_greedy(target_model : LLM_Pipeline, draft_model: LLM_Pipeline, dataloader: DataLoader, T=0.6, top_p=0.9, w=4, max_length=512):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    num_large_model_steps = 0
    dtype = torch.float16
    attn_mask = torch.full((max_length, max_length), torch.finfo(dtype).min, dtype=dtype, device=DEVICE)
    sequence = torch.tensor(list(range(max_length)), device=DEVICE).long().unsqueeze(-1)
    new_tokens_buffer =  torch.zeros(max_length).long().to(DEVICE)
    parents_buffer =  torch.zeros(max_length).long().to(DEVICE)
    position_ids = torch.zeros(max_length).long().to(DEVICE)
    branch_prob = torch.zeros(w + 1).to(DEVICE)
    output_branch_prob = torch.zeros(w + 2).to(DEVICE)
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            input_ids = batch['input_ids'][..., :128]
            labels = batch['labels'][..., :128]
            terminate = False
            if labels[0][-1] == -100: terminate = True
            draft_kv_len = 0
            target_kv_len = 0
            while input_ids.shape[1] < 256 and terminate == False:
                attn_mask.fill_(torch.finfo(dtype).min)
                spectree = GreedySTreeTest(prefix=input_ids.squeeze(0), device=DEVICE, temperature=T,
                                    top_p=top_p, 
                                    draft_kv_len=draft_kv_len, target_kv_len=target_kv_len,
                                    draft_model_engine=draft_model, target_model_engine=target_model, max_length=max_length,
                                    attn_mask = attn_mask, sequence = sequence, new_tokens_buffer = new_tokens_buffer, 
                                    parents_buffer = parents_buffer, 
                                    position_ids = position_ids, max_width=w)
                
                valid_tokens, draft_kv_len, target_kv_len,  b, terminate = spectree.verify()
                initial_size = input_ids.shape[1]
                input_ids = valid_tokens.unsqueeze(0)
                
                
                if (input_ids[0] == 2)._is_any_true() or (input_ids[0] == 0)._is_any_true(): terminate = True
                if not terminate:
                    branch_prob[b] += 1
                    num_decoding_steps += (valid_tokens.shape[0] - initial_size)
                    num_large_model_steps += 1

            draft_model.clear_kv()
            target_model.clear_kv()
            if num_large_model_steps > 0 and GLOBAL_RANK == 0:
                print(num_decoding_steps / num_large_model_steps)
    if GLOBAL_RANK==0:
        print("total decoding steps: {}".format(num_decoding_steps), "large model steps: {}".format(num_large_model_steps), "avg decoding step: {}".format(num_decoding_steps / num_large_model_steps))
        branch_prob = branch_prob / branch_prob.sum(dim=-1) 
        output_branch_prob[1:] = branch_prob
        print(output_branch_prob)
        torch.save(output_branch_prob, args.dst)
    return num_decoding_steps / num_large_model_steps

target_pp_config, draft_pp_config, target_last_stage_rank0, draft_last_stage_rank0 = initialized_dist_spec(args)
GLOBAL_RANK=dist.get_rank()
setup_seed(args.seed)

MAX_LEN = args.M
TARGET_MODEL_NAME = args.target
DRAFT_MODEL_NAME = args.model
DTYPE = torch.float16
DEVICE = torch.device("cuda", GLOBAL_RANK)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
if args.dataset == 'wiki':
    tokenized_dataset_eval = convert_wiki_dataset(tokenizer=tokenizer).select(list(range(args.start, args.end)))
elif args.dataset == 'cnn':
    tokenized_dataset_eval = convert_cnn_dataset(tokenizer=tokenizer).select(list(range(args.start, args.end)))
else:
    tokenized_dataset_eval = convert_c4_dataset(tokenizer=tokenizer,file_path="../dataset/c4_small.json").select(list(range(args.start, args.end)))

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
dataloader = DataLoader(tokenized_dataset_eval, batch_size=1, collate_fn=data_collator)

cg_list_draft = [128, 1, args.W]
cg_list_target = [128+args.W, args.W+1]

target_model = LLM_Pipeline(max_length=MAX_LEN, model_name=TARGET_MODEL_NAME, device=DEVICE, pp_config=target_pp_config, type="spec", last_stage_rank_0=target_last_stage_rank0, cg_list=cg_list_target)
draft_model =  LLM_Pipeline(max_length=MAX_LEN, model_name=DRAFT_MODEL_NAME, device=DEVICE, pp_config=draft_pp_config, type="spec", last_stage_rank_0=draft_last_stage_rank0, cg_list=cg_list_draft)

simulation_greedy(target_model=target_model, draft_model=draft_model, dataloader=dataloader, T=args.T, top_p=args.P, w=args.W, max_length=args.M)
