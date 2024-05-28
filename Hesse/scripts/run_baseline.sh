export CUDA_VISIBLE_DEVICES=3,4,5,6
export OMP_NUM_THREADS=48
torchrun --nproc_per_node=4 --master_port=13456 tests/baseline_benchmark.py --layer_partition 32 --tp_groups 4