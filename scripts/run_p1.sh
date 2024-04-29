# export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_IB_HCA=mlx5_2,mlx5_5
# export NCCL_SOCKET_IFNAME=^docker0,eno1,lo

export MASTER_ADDR='172.19.136.149'
export MASTER_PORT=9991
export WORLD_SIZE=2
export RANK=1
export NCCL_SOCKET_IFNAME=eno1

CUDA_VISIBLE_DEVICES=1 python3 cross_node_test.py