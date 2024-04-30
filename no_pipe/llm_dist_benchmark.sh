CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=48 \
torchrun --nproc_per_node=2 llm_dist_benchmark.py