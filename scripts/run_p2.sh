export MASTER_ADDR='172.19.136.149'
export MASTER_PORT=8092
export WORLD_SIZE=3
export RANK=2
export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES=0 python3 cross_node_test.py