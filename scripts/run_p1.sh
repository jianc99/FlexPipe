export MASTER_ADDR='172.19.136.149'
export MASTER_PORT=8092
export WORLD_SIZE=3
export RANK=1
export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES=1 python3 cross_node_test.py