export CUDA_VISIBLE_DEVICES=1,2,3,4
export OMP_NUM_THREADS=48
torchrun --nproc_per_node=4 tests/plot.py --layer_partition 24 --tp_groups 4 --model princeton-nlp/Sheared-LLaMA-1.3B