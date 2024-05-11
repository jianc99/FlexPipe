export MASTER_ADDR='172.19.136.149'
export MASTER_PORT=9991
export WORLD_SIZE=6
export RANK=3
export NCCL_SOCKET_IFNAME=eno1

CUDA_VISIBLE_DEVICES=4 python3 autoregressive_inference.py --layer_partition 10 11 11 --tp_groups 2 2 2