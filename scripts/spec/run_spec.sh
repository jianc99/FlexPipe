export CUDA_VISIBLE_DEVICES=1,4,6,9
export OMP_NUM_THREADS=48
torchrun --nproc_per_node=8 spec_benchmark.py --target_layer_partition 80 --target_tp_groups 4 --target_group 0 1 2 3 --draft_layer_partition 32 --draft_tp_groups 4 --draft_group 0 1 2 3