export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=48
torchrun --nproc_per_node=2 autoregressive_inference.py --layer_partition 32 --tp_groups 2