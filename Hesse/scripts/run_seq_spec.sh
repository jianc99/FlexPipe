export CUDA_VISIBLE_DEVICES=2,3,8,9
export OMP_NUM_THREADS=48
torchrun --nproc_per_node=4 tests/seq_spec_benchmark.py --target_layer_partition 40 40 --target_tp_groups 2 2 --target_group 0 1 2 3 --draft_layer_partition 32 --draft_tp_groups 4 --draft_group 0 1 2 3