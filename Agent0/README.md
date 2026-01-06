# Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2511.16043-b31b1b.svg)](https://arxiv.org/abs/2511.16043)

This repository is the official implementation of **Agent0**.

![](./figs/fig1.png)

**Agent0** is a fully autonomous framework that evolves high-performing agents from scratch without relying on any human-curated data. It employs a symbiotic competition between a **Curriculum Agent** (proposing tasks) and an **Executor Agent** (solving tasks with tools), driving a self-reinforcing cycle of improvement.

## 🔥 Key Features

* **Zero Data:** Eliminates dependency on external data or human annotations.
* **Co-Evolution:** Establishes a competition where the curriculum agent creates increasingly difficult tasks and the executor agent learns to solve them.
* **Tool Integration:** Integrates external tools (e.g., code interpreter) to enhance problem-solving and pressure the curriculum agent to generate complex, tool-aware tasks.

## 📊 Results

Empirically, Agent 0 substantially boosts reasoning capabilities on the Qwen3-8B-Base model:

* **+18%** improvement on Mathematical Reasoning benchmarks.
* **+24%** improvement on General Reasoning benchmarks.

## 🚀 Quickstart Guide

### 1. Configure Environment and Prepare Dirs

```bash
# Download models to /workspace which is Persistent volume in Runpod
huggingface-cli login
huggingface-cli download Qwen/Qwen3-0.6B-Base --local-dir /workspace/models/Qwen/Qwen3-0.6B-Base --local-dir-use-symlinks False

pip install nvitop
```

```bash
git clone https://github.com/xuanzhaopeng/Agent0.git

cd Agent0/Agent0

# For curriculum training
conda create -n curriculum python==3.12
conda activate curriculum
cd curriculum_train/
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install stopit flask vllm transformers torch requests # for vllm_service

# For executor training
conda deactivate
conda create -n executor python==3.12
conda activate executor
cd ../
pip install -r requirements.txt
cd executor_train/
pip install -e verl
pip install "flash-attn==2.8.3" --no-build-isolation --no-cache-dir
```

### 2. Sandbox Service

You need to deploy a code sandbox service for subsequent code compilation. You can dynamically adjust the number or specific configuration of the sandbox service based on your hardware settings such as CPU memory to meet higher concurrency requirements. Please refer to [this](https://github.com/bytedance/SandboxFusion) for more details.

Here is a sample script for the sandbox setup. We deployed four sandbox services. Each service is assigned an IP address and a corresponding port.

```bash
git clone https://github.com/xuanzhaopeng/SandboxFusion.git

cd SandboxFusion
bash scripts/build.sh
```

```bash
# Start 4 sandboxes with python runtime

cd Agent0/Agent0
bash start_sandboxes.sh
```

### 3. Train the Curriculum Agent

First, you need to fill out the code lines L36-41 with info of sandbox service from Step 2 in `curriculum_train/vllm_service_init/start_vllm_server_tool.py`.

```python
SANDBOX_API_URLS = [
    'IP1:PORT1/run_code',
    'IP2:PORT2/run_code',
    'IP3:PORT3/run_code',
    'IP4:PORT4/run_code'
]
```

Then use the script to train the curriculum agent. This step will be relatively slow due to limitations such as rollout and concurrency restrictions of the sandbox service. So if you would like to change the `max_turns`, please refer to `generate_with_tool_use()` in `curriculum_train/vllm_service_init/start_vllm_server_tool.py`.

```bash
cd curriculum_train/
conda activate curriculum

export MULTIPLE_VLLM_SERVICE=0
export STORAGE_PATH="/workspace/curriculum"
export HUGGINGFACENAME=""
export WANDB_API_KEY=""

mkdir -p \
  "$STORAGE_PATH/evaluation" \
  "$STORAGE_PATH/models" \
  "$STORAGE_PATH/generated_question" \
  "$STORAGE_PATH/temp_results"

# if we enable wandb
wandb login

# Initialize first iteration with base model
bash scripts/1_prepare_verl.sh
bash scripts/2_start_vllm_service.sh Qwen/Qwen3-0.6B-Base
bash scripts/3_curriculum_train_1gpu.sh Qwen/Qwen3-0.6B-Base qwen3_0_6b_curriculum_v1

bash scripts/2_start_vllm_service.sh Qwen/Qwen3-4B-Base
bash scripts/3_curriculum_train_1gpu.sh Qwen/Qwen3-4B-Base qwen3_8b_curriculum_v1
```

### 4. Data Curation

Then we need to construct the training data for the execution agent with filtering mechanism using self-consistency score.

```bash
executor_agent_path=Qwen/Qwen3-4B-Base
curriculum_agent_path=${STORAGE_PATH}/models/qwen3_4b_curriculum_v1/global_step_5/actor/huggingface
experiment_name=qwen3_4b_executor_v1

export VLLM_DISABLE_COMPILE_CACHE=1
echo 'start generate question'
bash question_generate/question_generate.bash $curriculum_agent_path 1000 $experiment_name

echo 'start evaluate generated question'
bash question_evaluate/evaluate.sh $executor_agent_path $experiment_name

echo 'start upload'
LOCAL_DATA_PATH=$(python question_evaluate/upload.py --max_score 0.8 --min_score 0.3 --experiment_name ${experiment_name})
echo "training data saved to: ${LOCAL_DATA_PATH}"
```

## 🙏 Acknowledgements

The framework is based on [VeRL](https://github.com/volcengine/verl). We use code from [R-Zero](https://github.com/Chengsong-Huang/R-Zero) and [VeRL-Tool](https://github.com/TIGER-AI-Lab/verl-tool) as the codebase and [SandboxFusion](https://github.com/bytedance/SandboxFusion) as the sandbox server. We thank the authors for releasing their code.

## 🖊️ Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@article{xia2025agent0,
  title={Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning},
  author={Xia, Peng and Zeng, Kaide and Liu, Jiaqi and Qin, Can and Wu, Fang and Zhou, Yiyang and Xiong, Caiming and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2511.16043},
  year={2025}
}
```
