export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=48
torchrun --nproc_per_node=4 autoregressive_inference.py --layer_partition 80 --tp_groups 4