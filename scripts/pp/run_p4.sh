export MASTER_ADDR='172.19.136.149'
export MASTER_PORT=9991
export WORLD_SIZE=8
export RANK=4
export NCCL_SOCKET_IFNAME=eno1

CUDA_VISIBLE_DEVICES=5 python3 pp_benchmark.py --layer_partition 16 16 --tp_groups 4 4