CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=48 \
torchrun --nproc_per_node=1 pp_benchmark.py --layer_partition 32 --tp_groups 1