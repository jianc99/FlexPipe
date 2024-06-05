export CUDA_VISIBLE_DEVICES=1,2,5,6
export OMP_NUM_THREADS=48
# torchrun --nproc_per_node=4 tests/btree_spec_benchmark.py --target_layer_partition 80 --target_tp_groups 4 --target_group 0 1 2 3 --draft_layer_partition 32 --draft_tp_groups 4 --draft_group 0 1 2 3 --model meta-llama/Llama-2-7b-hf --target meta-llama/Llama-2-70b-hf --T 0.6 --P 0.9 --M 256 --B 16 --growmap 7b-70b_tree.pt
torchrun --nproc_per_node=4 tests/pipeline_benchmark.py --target_layer_partition 32 --target_tp_groups 2 --target_group 0 1 --draft_layer_partition 24 --draft_tp_groups 2 --draft_group 2 3 --model princeton-nlp/Sheared-LLaMA-1.3B --target meta-llama/Llama-2-7b-hf --T 0.6 --P 0.9 --M 256 --B 4 --growmap 1.3b-70b_tree.pt