#!/bin/bash

executor_agent_path=$1

RUN_ID=$(date +%s%N)
export RUN_ID
echo "RUN_ID=$RUN_ID"

# vLLM server
bash vllm_service_init/start_1gpu.sh $executor_agent_path $RUN_ID
echo "vLLM services started with RUN_ID=$RUN_ID"
