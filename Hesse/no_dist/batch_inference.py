from llm import LLMEngine
import argparse
import time
import torch
import sys
sys.path.append("..")
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from Hesse.Tree.BatchTree import BatchSTree
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from Hesse.Data.data_converter import convert_wiki_dataset, convert_cnn_dataset, convert_c4_dataset
from Hesse.Tree.utils import cuda_graph_for_sampling_argmax
from Hesse.Engine.utils import setup_seed

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="JackFram/llama-68m", type=str, help='model')
parser.add_argument('--target', default="princeton-nlp/Sheared-LLaMA-1.3B", type=str, help='target model')
parser.add_argument('--growmap', type=str, default="1.3b-70b_tree.pt", help='growmap path')
parser.add_argument('--start', type=int, default=0, help='start')
parser.add_argument('--end', type=int, default=200, help='end')
parser.add_argument('--T', type=float, default=0.6, help='temperature')
parser.add_argument('--P', type=float, default=0.9, help='top_p')
parser.add_argument('--B', type=int, default=20, help='batch_size')
parser.add_argument('--M', type=int, default=256, help='max length')
parser.add_argument('--seed', type=int, default=123, help='random seed')
args = parser.parse_args()
print(args)
setup_seed(args.seed)

# test = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
# print(test[0][1:], test[0,1:])
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
# tokenizer.pad_token = tokenizer.eos_token
# # tokenized_dataset_eval = convert_wiki_dataset(tokenizer=tokenizer).select(list(range(args.start, args.end)))
# # tokenized_dataset_eval = convert_c4_dataset(tokenizer=tokenizer,file_path="../dataset/c4_small.json").select(list(range(args.start, args.end)))
# tokenized_dataset_eval = convert_cnn_dataset(tokenizer=tokenizer).select(list(range(args.start, args.end)))
# data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
# dataloader = DataLoader(tokenized_dataset_eval, batch_size=args.B, collate_fn=data_collator, shuffle=False)
# num_eval_steps = len(dataloader)
# for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
#             input_ids = batch['input_ids'][..., :128]
#             labels = batch['labels'][..., :128]
#             print((labels[:, -1] == -100)._is_any_true())
#             terminate = False
#             if (labels[:, -1] == -100)._is_any_true(): terminate = True
# batch_size = 3
# seq_len = 5
# num_selected_indices = 2
# tokens = torch.arange(0, seq_len).repeat(batch_size,1)
# selected_indices = torch.randint(0, seq_len, size=(batch_size, num_selected_indices))

# Use torch.gather to select tokens in parallel
# selected_tokens = torch.gather(tokens, 1, selected_indices)

# print("Selected tokens shape:", tokens, selected_indices, selected_tokens)
# time.sleep(1000)

def simulation_fast(target_model : LLMEngine, draft_model: LLMEngine, dataloader: DataLoader, T=0.6, top_p=0.9, 
            max_length=512, grow_map=None, sampling_callables = None,
            sample_gather_indices = None):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    num_large_model_steps = 0
    total_time = 0.0
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            input_ids = batch['input_ids'][..., :128]
            labels = batch['labels'][..., :128]
            terminate = False
            if (labels[:, -1] == -100)._is_any_true(): terminate = True
            spectree = BatchSTree (prefix=input_ids, device=DEVICE, temperature=T, top_p=top_p,
                                    draft_model_engine=draft_model, target_model_engine=target_model, max_length=max_length, grow_map=grow_map,
                                   sampling_callables=sampling_callables, sample_gather_indices = sample_gather_indices, batch_size=BATCH_SIZE, max_target_seq=256
                                    )
            torch.cuda.synchronize()
            t1 = time.time()
            longest=128

            while longest < 256 and terminate == False:
                spectree.construct_grow_map()
                num_nodes,inference_time, sample_time, verify_kv_time, terminate = spectree.verify(benchmark=True)
                longest = num_nodes.max()
                num_large_model_steps += 1
                print(inference_time, sample_time, verify_kv_time)

            torch.cuda.synchronize()
            t2 = time.time()
            num_decoding_steps += (num_nodes.sum() - input_ids.shape[1]*BATCH_SIZE)
            total_time += (t2 - t1)
            for i in range(BATCH_SIZE):
                print(tokenizer.decode(spectree.tokens[i,:num_nodes[i]]))
            draft_model.clear_kv()
            target_model.clear_kv()
            print("total time :{:.5f}s, latency :{:.5f}s, decoding step: {}, large model step: {}".format(total_time, total_time / num_decoding_steps, num_decoding_steps, num_large_model_steps))
    return num_decoding_steps / num_large_model_steps

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenized_dataset_eval = convert_cnn_dataset(tokenizer=tokenizer).select(list(range(args.start, args.end)))
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
dataloader = DataLoader(tokenized_dataset_eval, batch_size=args.B, collate_fn=data_collator, shuffle=False)

path = args.growmap
grow_map = torch.load(path)
tree_size = grow_map["size"]
idx_lists = grow_map["roots"]
branch_lists = grow_map['branches']
draft_step = len(grow_map["roots"])

MAX_LEN = args.M + tree_size
TARGET_MODEL_NAME = args.target
DRAFT_MODEL_NAME = args.model
DTYPE = torch.float16
DEVICE = torch.device("cuda", 0)
BATCH_SIZE = args.B
torch.cuda.set_device(DEVICE)

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

cg_list_target = [tree_size]
cg_list_draft = [sum(x) for x in branch_lists]
cg_list_draft.append(1)

# print(sample_gather_indices, cg_list_draft, cg_list_target)


draft_model =  LLMEngine(max_length=MAX_LEN, model_name=DRAFT_MODEL_NAME, device=DEVICE, batch_size=args.B, dtype=torch.float16)
target_model = LLMEngine(max_length=MAX_LEN, model_name=TARGET_MODEL_NAME, device=DEVICE, batch_size=args.B, dtype=torch.float16)

draft_model.initialize_cuda_graph(cg_list_draft)
target_model.initialize_cuda_graph(cg_list_target)

simulation_fast(target_model=target_model, draft_model=draft_model, dataloader=dataloader, T=args.T, top_p=args.P,
                                     max_length=MAX_LEN, grow_map = grow_map, sampling_callables=sampling_callables, sample_gather_indices = sample_gather_indices)

