export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=48
torchrun --nproc_per_node=4 tests/btree_spec_benchmark.py --target_layer_partition 80 --target_tp_groups 4 --target_group 0 1 2 3 --draft_layer_partition 24 --draft_tp_groups 1 --draft_group 0 --model princeton-nlp/Sheared-LLaMA-1.3B --target meta-llama/Llama-2-70b-hf --T 0.6 --P 0.9 --M 256 --B 16 --growmap 1.3b-70b_tree.pt