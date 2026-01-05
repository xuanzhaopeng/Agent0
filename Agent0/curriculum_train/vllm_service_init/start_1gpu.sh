model_path=$1
run_id=$2
export VLLM_DISABLE_COMPILE_CACHE=1
CUDA_VISIBLE_DEVICES=0 python vllm_service_init/start_vllm_server_tool.py --port 5000 --gpu_mem_util 0.5 --model_path $model_path --max_model_len 2048 &
