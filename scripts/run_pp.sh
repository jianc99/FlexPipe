export CUDA_VISIBLE_DEVICES=5,6,7,8
export OMP_NUM_THREADS=48
torchrun --nproc_per_node=4 autoregressive_inference.py --layer_partition 80 --tp_groups 4