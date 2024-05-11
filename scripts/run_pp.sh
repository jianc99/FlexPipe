CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
OMP_NUM_THREADS=48 \
torchrun --nproc_per_node=8 autoregressive_inference.py --layer_partition 32 --tp_groups 8