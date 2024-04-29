# FlexSpec
## Installation
### Create Virtual Environment
``` bash
conda create -n flexspec python=3.11
```

### Install Necessary Package
Must ensure NCCL version to be the same across different nodes.

``` bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.36.2
```

## Run Scripts
Run the scripts for each GPU worker. Need to specify the master address and port, and NCCL_SOCKET_IFNAME to specific network interface.
``` bash
export MASTER_ADDR='172.24.46.47'
export MASTER_PORT=9991
export WORLD_SIZE=3
export RANK=0
export NCCL_SOCKET_IFNAME=eno1

CUDA_VISIBLE_DEVICES=1 python3 cross_node_test.py
```


## Performance on A100 80G PCIE
Unit in ms, Prefix = 512, Batch size = 1
| Model / # GPUs | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Sheared-LLaMA-2.7B  |  7.9 |   |   |  |
| Llama-2-7b  | 12.7  |   |   |   |
| Llama-2-13b  | 21.6 |   |   |   |
| Llama-2-70b | x  |   |   |   |
| vicuna-33b-v1.3 | 49.0  |   |   |   |

## Performance on 4090 24G PCIE
Unit in ms, Prefix = 512, Batch size = 1
| Model / # GPUs | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Llama-2-7b  | 17.1  | 11.3  | 7.5  | 5.9  |
| Llama-2-70b | x  |  x | x  | 29.9  |
| vicuna-33b-v1.3 | x  | x  | 25.0  | x  |

## Performance on L40 48G PCIE
Unit in ms, Prefix = 512, Batch size = 1
| Model / # GPUs | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Llama-2-7b  | 22.1  | 14.4  | 9.0  | 7.0  |
<!-- | Llama-2-70b | x  |  x | x  | x  | -->

PP+TP Degree= 4 4 means the first and second pipeline stages are both doing tensor parallelism with degree=4.

| PP+TP Degree | 2 2 | 4 4 | 2 2 2 |
|---|---|---|---|
| Llama-2-7b  | 19.2  | 13.4  | 17.0 |