export CUDA_VISIBLE_DEVICES=2,3,4,5
export OMP_NUM_THREADS=48
torchrun --nproc_per_node=4 batchsize_latency_plot.py --layer_partition 80 --tp_groups 4