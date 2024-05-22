export CUDA_VISIBLE_DEVICES=0,1,4,5,6,7,8,9
export OMP_NUM_THREADS=48
torchrun --nproc_per_node=8 tests/baseline_benchmark.py --layer_partition 20 20 20 20 --tp_groups 2 2 2 2