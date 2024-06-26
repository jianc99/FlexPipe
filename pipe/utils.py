import torch
import torch.distributed as dist
import argparse
import numpy as np
import random

#Each stage's global ranks
#Send list and receive rank
#If last stage of first stage
# embed+ First stage layers + send to next
# Receive from previous, layers , send to next
# receive from previous, last stage layers + epsilon+ norm + lm head + broadcast

def make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask

def args_parse():
    parser = argparse.ArgumentParser(description='Process model configuration and partitions.')
    
    parser.add_argument('--model', type=str, default="meta-llama/Llama-2-70b-hf", help='Model identifier.')
    parser.add_argument('--T', type=int, default=2000, help='Repeat times.')
    parser.add_argument('--B', type=int, default=1, help='Batch size.')
    parser.add_argument('--P', type=int, default=128, help='Prefix length.')
    parser.add_argument('--M', type=int, default=512, help='Maximum length.')
    parser.add_argument('--D', type=int, default=1, help='Decrement length.')
    parser.add_argument('--layer_partition', nargs='+', type=int, help='Layer partitioning as a list of integers.')
    parser.add_argument('--tp_groups', nargs='+', type=int, help='TP groups as a list of integers.')

    args = parser.parse_args()
    
    return args

def gen_tp_rank_groups(tp_groups):
    tp_rank_groups = []
    current_tp_rank = 0
    for tp_rank in tp_groups:
        new_tp_group = []
        for _ in range(tp_rank):
            new_tp_group.append(current_tp_rank)
            current_tp_rank += 1
        tp_rank_groups.append(new_tp_group)
    return tp_rank_groups

def get_group_for_rank(rank, index_mapping):
    return index_mapping.get(rank)

def gen_include_layers(current_stage, layer_partition):
    start_idx = 0
    stage_indices = []
    for part in layer_partition:
        indices = list(range(start_idx, start_idx + part))
        stage_indices.append(indices)
        start_idx += part
    return stage_indices[current_stage]

def generate_index_mapping(original_lists):
    index_mapping = {}
    for index, group in enumerate(original_lists):
        for rank in group:
            index_mapping[rank] = index
    return index_mapping

def initialized_dist(tp_groups,layer_partition):
    dist.init_process_group(backend='nccl')
    dist.barrier()
    global_rank=dist.get_rank()
    stage_num = len(tp_groups)
    tp_rank_groups = gen_tp_rank_groups(tp_groups)
    index_mapping = generate_index_mapping(tp_rank_groups)
    current_stage = get_group_for_rank(global_rank,index_mapping)
    process_groups = []
    for process_group in tp_rank_groups:
        process_groups.append(dist.new_group(process_group))
    current_tp_group=process_groups[current_stage]
    current_stage_layers=gen_include_layers(current_stage,layer_partition)
    pp_config={
        'num_stages':stage_num,
        'process_groups':process_groups,
        'groups_indices':tp_rank_groups,
        'current_group':current_tp_group,
        'current_stage':current_stage,
        'current_layers':current_stage_layers,
    }
    dist.barrier()
    # torch.cuda.set_device(0)
    torch.cuda.set_device(global_rank)
    return pp_config

def modify_logits_for_top_p_filtering(logits, top_p):
    """Set the logits for none top-p values to -inf."""
    if top_p <= 0.0:
        return
    # First sort and calculate cumulative sum of probabilities.
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
     # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float('-inf'))


def sample(logits, top_k=1, top_p=0.0, temperature=1.0):
    """Sample from top-k logits.
    Arguments:
        logits: Tensor of shape (batch_size, vocab_size)
    """
    if top_k == 1:  # Short-circuit for greedy decoding
        return logits.argmax(dim=-1)
    else:
        if top_p > 0.0:
            assert top_p <= 1.0, 'top-p should be in (0, 1].'
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Safety check
            logits_top, indices = torch.topk(logits, top_k, dim=-1)
            logits_top /= temperature
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return indices[
                torch.arange(indices.shape[0], device=indices.device),
                torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(dim=-1)
            ]
        else:
            logits_top = logits / temperature
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(dim=-1)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    tp_groups=[2,4,2]
    layer_partition=[20,40,20]
    print(gen_include_layers(0,layer_partition))
