import os
import sys
sys.path.append("..")
from transformers import LlamaTokenizer, DataCollatorForLanguageModeling
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torch
import numpy as np 
from torch.nn.functional import softmax
import argparse
from Hesse.Engine.utils import initialized_dist_spec, make_causal_mask, sample, setup_seed, convert_dataset
from Hesse.Engine.pipleline import LLM_Pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="meta-llama/Llama-2-7b-hf", type=str, help='model')
parser.add_argument('--target', default="meta-llama/Llama-2-70b-hf", type=str, help='target model')
parser.add_argument('--T', type=float, default=0.6, help='temperature')
parser.add_argument('--P', type=float, default=0.9, help='top_p')
parser.add_argument('--DP', type=float, default=0.9, help='draft_top_p')
parser.add_argument('--W', type=int, default=32, help='max width')
parser.add_argument('--M', type=int, default=512, help='max length')
parser.add_argument('--dataset', type=str, default="../dataset/c4_small.json", help='dataset path')
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

def get_residual(p: torch.Tensor, q:torch.Tensor):
    residual = p - q
    residual[residual < 0] = 0.0
    residual = residual / (residual.sum(dim=-1).unsqueeze(-1) + 1e-9)
    
    return residual

def evaluate(target_model : LLM_Pipeline, draft_model: LLM_Pipeline, dataloader: DataLoader, k:int, T=0.6, top_p=0.9, draft_top_p=0.99):
    num_eval_steps = len(dataloader)
    acceptance_rate = torch.zeros(k)
    num_samples = 0
    draft_model_prob = []
    token_accept_rate = []
    sampled_token_sets = []
    real_budget = 0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            attention_mask = make_causal_mask((MAX_LEN, MAX_LEN), dtype=DTYPE, device=DEVICE)
            attention_mask = attention_mask[None, None, :, :]
            input_ids = batch['input_ids'].to(DEVICE)
            prefix_len = input_ids.size(1)
            position_ids = torch.arange(prefix_len, device=DEVICE).unsqueeze(0)
            prefix_storage_ids = torch.arange(prefix_len, device=DEVICE)
            target_logits : torch.Tensor = target_model.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., :prefix_len,:], storage_ids=prefix_storage_ids).clone()
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(target_logits, descending=True)
                cumulative_probs = torch.cumsum(
                torch.nn.functional.softmax(sorted_logits / T, dim=-1), dim=-1)
                filter = cumulative_probs > top_p
                filter[..., 1:] = filter[..., :-1].clone()
                filter[..., 0] = 0
                indices_to_remove = filter.scatter(-1, sorted_indices, filter)
                target_logits[indices_to_remove] = float('-inf')

            
            draft_logits = draft_model.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., :prefix_len,:], storage_ids=prefix_storage_ids).clone()
            target_prob = softmax(target_logits / T, dim=-1).squeeze(0)
            q = softmax(draft_logits / T, dim=-1).squeeze(0)
            
            for i in range(128, target_prob.shape[0]):
                token_acceptance_rate = torch.zeros(k)
                draft_tokens = []
                if batch['labels'][0][i] == -100 or batch['labels'][0][i] == 0: continue
                num_samples = num_samples + 1
                token_target_prob = target_prob[i]
                # token_draft_prob = q[i]
                #draft_model_prob.append(q[i].cpu())
                token_draft_logits = draft_logits[0][i]

                if draft_top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(token_draft_logits, descending=True)
                    cumulative_probs = torch.cumsum(
                    torch.nn.functional.softmax(sorted_logits / T, dim=-1), dim=-1)
                    filter = cumulative_probs > draft_top_p
                    filter[..., 1:] = filter[..., :-1].clone()
                    filter[..., 0] = 0
                    indices_to_remove = filter.scatter(-1, sorted_indices, filter)
                    token_draft_logits[indices_to_remove] = float('-inf')

                token_draft_prob = softmax(token_draft_logits / T, dim=-1).squeeze(0)
                sampled_token = token_draft_prob.multinomial(num_samples=1, replacement=True)
                draft_tokens.append(sampled_token.item())
                real_budget = real_budget + 1
                token_acceptance_rate[0] = min(1.0, (token_target_prob[sampled_token]/ token_draft_prob[sampled_token]))

                token_target_prob = get_residual(token_target_prob, token_draft_prob)
                
                
                for j in range(k-1):
                    token_draft_logits[sampled_token] = - torch.inf
                    token_draft_prob = softmax(token_draft_logits / (T), dim=-1).squeeze(0)
                    if torch.isnan(token_draft_prob).long().sum() >= 1:
                        break
                    token_draft_prob = token_draft_prob / token_draft_prob.sum(-1)
                    sampled_token = token_draft_prob.multinomial(num_samples=1, replacement=True)
                    draft_tokens.append(sampled_token.item())
                    real_budget = real_budget + 1
                    branch_token_acceptance_rate = min(1, token_target_prob[sampled_token]/ token_draft_prob[sampled_token])
                    token_acceptance_rate[j+1] = (1 - token_acceptance_rate.sum()) * branch_token_acceptance_rate
                    
                    token_target_prob = get_residual(token_target_prob, token_draft_prob)
                acceptance_rate = acceptance_rate + token_acceptance_rate
                token_accept_rate.append(token_acceptance_rate.cpu())
                sampled_token_sets.append(draft_tokens)
                draft_model_prob.append(q[i][draft_tokens].cpu()) 
            draft_model.clear_kv()
            target_model.clear_kv()
    return acceptance_rate / num_samples

target_pp_config, draft_pp_config, target_last_stage_rank0, draft_last_stage_rank0 = initialized_dist_spec(args)
global_rank=dist.get_rank()
setup_seed(args.seed)

MAX_LEN = args.M
TARGET_MODEL_NAME = args.target
DRAFT_MODEL_NAME = args.model
DTYPE = torch.float16
DEVICE = torch.device("cuda", global_rank)

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenized_dataset_eval = convert_dataset(tokenizer=tokenizer,file_path="../dataset/c4_small.json").select(list(range(args.start, args.end)))
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
dataloader = DataLoader(tokenized_dataset_eval, batch_size=1, collate_fn=data_collator)

target_model = LLM_Pipeline(max_length=MAX_LEN, model_name=TARGET_MODEL_NAME, device=DEVICE, pp_config=target_pp_config, type="spec", last_stage_rank_0=target_last_stage_rank0, cg_list=[256])
draft_model =  LLM_Pipeline(max_length=MAX_LEN, model_name=DRAFT_MODEL_NAME, device=DEVICE, pp_config=draft_pp_config, type="spec", last_stage_rank_0=draft_last_stage_rank0, cg_list=[256])

acceptance_rate_list = [0]
branch_acceptance_rate_list = [0]

acceptance_rate = evaluate(target_model, draft_model, dataloader, k=args.W, T=args.T, top_p=args.P, draft_top_p=args.DP)
x = torch.zeros(len(acceptance_rate) + 1)
x[1:] = acceptance_rate
if global_rank == 0:
    torch.save(x, args.dst)
    print(x)
