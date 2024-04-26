CUDA_VISIBLE_DEVICES=5,6 \
OMP_NUM_THREADS=48 \
torchrun --nproc_per_node=2 llm_dist_benchmark.py