export MASTER_ADDR='172.19.136.149'
export MASTER_PORT=9991
export WORLD_SIZE=4
export RANK=2
export NCCL_SOCKET_IFNAME=eno1

CUDA_VISIBLE_DEVICES=7 python3 autoregressive_inference.py --layer_partition 80 --tp_groups 4