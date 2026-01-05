#!/bin/bash

project_name=agent0

curriculum_agent_path=$1
save_path=$2
echo "save_path: $save_path"

echo "Start training curriculum: $curriculum_agent_path -> $save_path"

# Set environment variable to indicate single vLLM service usage
export MULTIPLE_VLLM_SERVICE=0

CUDA_VISIBLE_DEVICES=0 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=2048 \
    worker.actor.model.model_path=$curriculum_agent_path \
    trainer.experiment_name=$save_path \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
    trainer.total_epochs=1000 \
    worker.reward.reward_function=./examples/reward_function/curriculum_reward.py:compute_score \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=1 \
    data.format_prompt=./examples/format_prompt/questioner.jinja \
    worker.rollout.n=4 \
    worker.actor.global_batch_size=64 \
    trainer.logger=['console'] \
    trainer.project_name=$project_name \
    trainer.max_steps=6 \
    trainer.save_freq=1 \
    worker.rollout.gpu_memory_utilization=0.5 \
    worker.rollout.tensor_parallel_size=1


sleep 5

echo "merging model"
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/$save_path/global_step_5/actor

sleep 10

pkill python

echo "curriculum agent training finished"
