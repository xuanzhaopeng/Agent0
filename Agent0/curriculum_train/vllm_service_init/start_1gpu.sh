#!/bin/bash

model_path=$1
run_id=$2

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

export VLLM_USE_V1=0 
export VLLM_DISABLE_COMPILE_CACHE=1
CUDA_VISIBLE_DEVICES=1 python vllm_service_init/start_vllm_server_tool.py --port 5000 --gpu_mem_util 0.5 --model_path $model_path --max_model_len 4096 --enforce_eager