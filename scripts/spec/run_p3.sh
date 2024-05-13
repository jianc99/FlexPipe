export MASTER_ADDR='172.19.136.149'
export MASTER_PORT=9991
export WORLD_SIZE=4
export RANK=3
export NCCL_SOCKET_IFNAME=eno1
export CUDA_VISIBLE_DEVICES=8

python3 spec_benchmark.py --target_layer_partition 80 --target_tp_groups 4 --target_group 0 1 2 3 --draft_layer_partition 32 --draft_tp_groups 4 --draft_group 0 1 2 3