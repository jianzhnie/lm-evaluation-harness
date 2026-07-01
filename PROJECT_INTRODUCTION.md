# Language Model Evaluation Harness (lm-eval) 项目介绍

> **版本**: v0.4.12 | **维护方**: EleutherAI | **许可证**: MIT
> **仓库地址**: https://github.com/EleutherAI/lm-evaluation-harness

---

## 一、项目概述

**Language Model Evaluation Harness**（简称 lm-eval）是一个统一的语言模型评估框架，用于对生成式语言模型进行大规模、多任务的标准化评测。

该框架是 Hugging Face **Open LLM Leaderboard** 的后端评测引擎，已被数百篇学术论文引用，并被 NVIDIA、Cohere、BigScience、BigCode、Nous Research、Mosaic ML 等数十家组织内部使用。

### 核心特性

- **丰富的评测任务**: 内置 60+ 标准学术基准测试，包含数百个子任务和变体
- **多后端模型支持**: HuggingFace Transformers、vLLM、SGLang、OpenAI API、Anthropic API、NVIDIA NeMo、Megatron-LM 等
- **多硬件加速器支持**: NVIDIA GPU (CUDA)、华为昇腾 NPU、Intel GPU (XPU)、Apple Silicon (MPS)、寒武纪 MLU 等
- **灵活的任务配置**: 基于 YAML 配置文件和 Jinja2 模板的 Prompt 设计
- **分布式评估**: 支持多 GPU/NPU 数据并行 (DP)、张量并行 (TP)、专家并行 (EP)
- **可复现性**: 使用公开可用的 Prompt 进行评估，确保不同论文之间的可比性
- **结果可视化**: 支持 Weights & Biases (W&B) 和 Zeno 平台集成

---

## 二、项目结构

```
lm-evaluation-harness/
├── lm_eval/                  # 核心包
│   ├── __main__.py           # CLI 入口
│   ├── _cli/                 # 命令行子命令 (run, ls, validate)
│   ├── api/                  # 外部 API 集成
│   ├── device.py             # 多设备管理 (CUDA/NPU/XPU/MPS 等)
│   ├── models/               # 模型后端实现
│   │   ├── hf/               # HuggingFace Transformers
│   │   ├── vllm_causallms.py # vLLM 推理
│   │   ├── sglang.py         # SGLang 推理
│   │   ├── megatron_lm.py    # Megatron-LM
│   │   ├── mindspeed_lm.py   # MindSpeed-LLM (华为昇腾 NPU)
│   │   ├── openai/           # OpenAI API
│   │   ├── anthropic/        # Anthropic API
│   │   └── ...               # 其他后端
│   ├── tasks/                # 评测任务定义 (YAML + Python)
│   ├── filters/              # 输出过滤器
│   ├── prompts/              # Prompt 模板
│   ├── loggers/              # 日志记录器
│   ├── caching/              # 结果缓存
│   ├── config/               # 配置管理
│   ├── evaluator.py          # 核心评估引擎
│   └── utils.py              # 工具函数
├── docs/                     # 文档
│   ├── interface.md          # CLI 参考手册
│   ├── config_files.md       # YAML 配置指南
│   ├── new_task_guide.md     # 新任务开发指南
│   ├── model_guide.md        # 模型使用指南
│   ├── python-api.md         # Python API 文档
│   └── task_guide.md         # 任务配置指南
├── scripts/                  # 辅助脚本
├── examples/                 # 使用示例 (Jupyter notebooks)
├── templates/                # 模板文件
├── tests/                    # 测试用例
└── pyproject.toml            # 项目配置 & 依赖声明
```

---

## 三、安装方式

### 基础安装

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

基础安装仅包含核心评估框架，**模型后端需单独安装**：

```bash
# HuggingFace Transformers 模型
pip install "lm_eval[hf]"

# vLLM 推理加速
pip install "lm_eval[vllm]"

# API 模型 (OpenAI, Anthropic 等)
pip install "lm_eval[api]"

# 同时安装多个后端
pip install "lm_eval[hf,vllm,api]"
```

### 其他可选依赖

| 类别 | 安装标识 | 说明 |
|------|---------|------|
| 模型后端 | `[hf]` | HuggingFace Transformers |
| 模型后端 | `[vllm]` | vLLM 推理加速 |
| 模型后端 | `[api]` | OpenAI / Anthropic 等 API 模型 |
| 模型后端 | `[gptq]` / `[gptqmodel]` | GPTQ 量化模型 |
| 模型后端 | `[optimum]` | Intel OpenVINO 模型 |
| 模型后端 | `[ipex]` | Intel IPEX 后端 |
| 模型后端 | `[habana]` | Intel Gaudi 后端 |
| 模型后端 | `[neuronx]` | AWS Inferentia2 |
| 模型后端 | `[winml]` | Windows ML (CPU/GPU/NPU) |
| 任务依赖 | `[tasks]` | 所有任务依赖 |
| 任务依赖 | `[math]` | 数学答案校验 |
| 任务依赖 | `[ifeval]` | IFEval 任务 |
| 开发工具 | `[dev]` | 代码检查 & 贡献工具 |
| 可视化 | `[wandb]` | Weights & Biases 日志 |
| 可视化 | `[zeno]` | Zeno 结果可视化 |

---

## 四、快速上手

### 1. CLI 基本用法

```bash
# 查看帮助
lm-eval -h
lm-eval run -h

# 列出所有可用任务
lm-eval ls tasks

# 评估 HuggingFace 模型
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

### 2. 使用 YAML 配置文件

```bash
lm-eval run --config my_eval_config.yaml
```

### 3. Python API 调用

```python
import lm_eval

results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=EleutherAI/gpt-j-6B",
    tasks=["hellaswag", "arc_easy"],
    batch_size=8,
)
```

### 4. 评估 vLLM 模型

```bash
lm_eval --model vllm \
    --model_args pretrained=meta-llama/Llama-2-7b,tensor_parallel_size=2,dtype=auto \
    --tasks lambada_openai \
    --batch_size auto
```

### 5. 评估 API 模型

```bash
export OPENAI_API_KEY=YOUR_KEY_HERE
lm_eval --model openai-completions \
    --model_args model=davinci-002 \
    --tasks lambada_openai,hellaswag
```

---

## 五、支持的模型后端

| 后端 | `--model` 参数 | 请求类型支持 |
|------|---------------|-------------|
| HuggingFace Transformers | `hf` | generate_until, loglikelihood, loglikelihood_rolling |
| vLLM | `vllm` | generate_until, loglikelihood, loglikelihood_rolling |
| SGLang | `sglang` | generate_until, loglikelihood, loglikelihood_rolling |
| OpenAI Completions | `openai-completions` | generate_until, loglikelihood, loglikelihood_rolling |
| OpenAI ChatCompletions | `openai-chat-completions` | generate_until (无 logprobs) |
| Anthropic | `anthropic` | generate_until (无 logprobs) |
| Anthropic Chat | `anthropic-chat` | generate_until (无 logprobs) |
| TextSynth | `textsynth` | generate_until, loglikelihood, loglikelihood_rolling |
| llama.cpp (GGUF/GGML) | `gguf` / `ggml` | generate_until, loglikelihood |
| NVIDIA NeMo | `nemo_lm` | generate_until, loglikelihood, loglikelihood_rolling |
| NVIDIA Megatron-LM | `megatron_lm` | generate_until, loglikelihood, loglikelihood_rolling |
| MindSpeed-LLM (昇腾 NPU) | `mindspeed_lm` | generate_until, loglikelihood, loglikelihood_rolling |
| Mamba | `mamba_ssm` | generate_until, loglikelihood, loglikelihood_rolling |
| OpenVINO | `openvino` | generate_until, loglikelihood, loglikelihood_rolling |
| Intel IPEX | `ipex` | generate_until, loglikelihood, loglikelihood_rolling |
| Intel Gaudi | `habana` | generate_until, loglikelihood, loglikelihood_rolling |
| AWS Inferentia2 | `neuronx` | generate_until, loglikelihood, loglikelihood_rolling |
| Windows ML | `winml` | generate_until, loglikelihood, loglikelihood_rolling |
| Watsonx.ai | `watsonx_llm` | generate_until, loglikelihood |
| 本地推理服务器 | `local-completions` | generate_until, loglikelihood, loglikelihood_rolling |

---

## 六、评测任务体系

框架内置了丰富的评测任务，涵盖以下类别：

### 常见基准测试
- **HellaSwag** - 常识推理
- **MMLU** - 多领域知识理解
- **ARC (Easy/Challenge)** - 科学推理
- **WinoGrande** - 共指消解
- **PIQA** - 物理直觉推理
- **BoolQ** - 布尔问答
- **OpenBookQA** - 开卷问答
- **LAMBADA** - 语言建模
- **GSM8K** - 数学推理
- **TruthfulQA** - 真实性评估
- **BIG-Bench-Hard** - 高难度综合推理
- **Belebele** - 多语言阅读理解

### Open LLM Leaderboard 任务
专门对齐 Hugging Face Open LLM Leaderboard 的评测标准，可通过 `leaderboard` 任务组直接调用。

### 任务类型
- `generate_until` - 生成式任务（适用于所有模型）
- `loglikelihood` - 对数似然计算（需模型提供 logprobs）
- `loglikelihood_rolling` - 滚动对数似然
- `multiple_choice` - 多选题

---

## 七、高级功能

### 多 GPU 评估

```bash
# 数据并行 (每张 GPU 加载完整模型副本)
accelerate launch -m lm_eval --model hf \
    --tasks lambada_openai,arc_easy --batch_size 16

# 模型并行 (大模型分片到多张 GPU)
lm_eval --model hf --tasks arc_easy \
    --model_args parallelize=True --batch_size 16

# vLLM 张量并行 + 数据并行
lm_eval --model vllm \
    --model_args pretrained=model_name,tensor_parallel_size=2,data_parallel_size=4 \
    --tasks hellaswag --batch_size auto
```

### 量化模型评估

```bash
# GPTQ 量化
lm_eval --model hf \
    --model_args pretrained=model-path,gptqmodel=True \
    --tasks hellaswag

# 4-bit 量化
lm_eval --model hf \
    --model_args pretrained=model-path,load_in_4bit=True \
    --tasks hellaswag

# GGUF 格式
lm_eval --model hf \
    --model_args pretrained=/path/to/gguf,gguf_file=model.gguf,tokenizer=/path/to/tokenizer \
    --tasks hellaswag
```

### LoRA / PEFT 适配器评估

```bash
lm_eval --model hf \
    --model_args pretrained=base-model,peft=lora-adapter-path \
    --tasks hellaswag
```

### 结果保存与缓存

```bash
# 保存结果和样本
lm_eval --model hf --tasks hellaswag \
    --log_samples --output_path results/

# 使用缓存加速重复评估
lm_eval --model hf --tasks hellaswag --use_cache ./cache/

# 推送到 Hugging Face Hub
lm_eval --model hf --tasks hellaswag --log_samples \
    --output_path results \
    --hf_hub_log_args hub_results_org=MyOrg,hub_repo_name=eval-results,push_results_to_hub=True
```

### 可视化集成

```bash
# Weights & Biases
lm_eval --model hf --tasks hellaswag \
    --wandb_args project=lm-eval-harness --log_samples

# Zeno
python scripts/zeno_visualize.py --data_path output --project_name "My Project"
```

### 华为昇腾 NPU 评估

框架通过 `lm_eval/device.py` 模块提供统一的设备管理，支持华为昇腾 NPU (Ascend) 硬件加速。NPU 设备通过 `torch_npu` 扩展接入，分布式通信使用 HCCL 后端。

#### 环境准备

```bash
# 1. 安装 CANN 驱动和工具链 (参考华为官方文档)
# 2. 安装 torch_npu
pip install torch_npu

# 3. 安装 lm-eval 及 HuggingFace 后端
pip install -e ".[hf]"
```

#### HuggingFace 后端 + NPU

```bash
# 单卡 NPU 评估
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device npu:0 \
    --batch_size 8

# 多卡 NPU 数据并行
accelerate launch -m lm_eval --model hf \
    --tasks hellaswag \
    --batch_size 16
```

#### MindSpeed-LLM 后端 + NPU (推荐)

`mindspeed_lm` 后端基于华为 MindSpeed-LLM 框架，专为 Ascend NPU 优化。内置 `AscendNPUPatch` 自动处理 CUDA→NPU 适配，无需手动配置分布式后端。

```bash
export MEGATRON_PATH=/path/to/MindSpeed-LLM
export CKPT_PATH=/path/to/megatron_ckpt
export TOKENIZER_MODEL=/path/to/tokenizer

# 快速启动（使用预设 run.sh）
bash run.sh

# 单卡 NPU
torchrun --nproc-per-node=1 -m lm_eval --model mindspeed_lm \
    --model_args "load=${CKPT_PATH},tokenizer_type=PretrainedFromHF,tokenizer_name_or_path=${TOKENIZER_MODEL},devices=1" \
    --tasks hellaswag --batch_size 8

# 张量并行 (4 卡 NPU)
torchrun --nproc-per-node=4 -m lm_eval --model mindspeed_lm \
    --model_args "load=${CKPT_PATH},tokenizer_type=PretrainedFromHF,tokenizer_name_or_path=${TOKENIZER_MODEL},devices=4,tensor_model_parallel_size=4" \
    --tasks hellaswag --batch_size 8
```

#### Megatron-LM 后端 + NPU

Megatron-LM 后端同样支持 NPU：

```bash
export MEGATRON_PATH=/path/to/Megatron-LM

torchrun --nproc-per-node=2 -m lm_eval --model megatron_lm \
    --model_args load=/path/to/ckpt,devices=2,tensor_model_parallel_size=2,tokenizer_model=/path/to/tokenizer.model \
    --tasks hellaswag --batch_size 8
```

#### NPU 分布式策略对照

| 并行策略 | `mindspeed_lm` 用法 | 分布式后端 | 说明 |
|----------|---------------------|-----------|------|
| 单卡 | `devices=1` | HCCL | 单 NPU 评估 |
| 数据并行 (DP) | `devices=N, TP=1` | HCCL | 每卡完整模型, 数据分片 |
| 张量并行 (TP) | `devices=N, tensor_model_parallel_size=N` | HCCL | 模型层切分到多卡 |
| 专家并行 (EP) | `devices=N, expert_model_parallel_size=N` | HCCL | MoE 专家分布到多卡 |

> `AscendNPUPatch` 类自动处理 NPU 适配：`torch.cuda` → `torch.npu` API 映射，`nccl` → `hccl` 重定向，默认生成器初始化，`_compile_dependencies` 禁用。启动脚本 `npu_mindspeed-llm_eval.sh` 封装了常用模式。

#### NPU 设备管理 API (`lm_eval/device.py`)

| 功能 | 函数 | NPU 实现 |
|------|------|---------|
| 设备检测 | `_is_torch_npu_available()` | `import torch_npu` + `torch.npu.is_available()` |
| 获取当前设备 | `get_current_device()` | `npu:{LOCAL_RANK}` |
| 设备数量 | `get_device_count()` | `torch.npu.device_count()` |
| 分布式后端 | `get_distributed_backend()` | `hccl` |
| 缓存清理 | `empty_cache()` | `torch.npu.empty_cache()` |
| 设备同步 | `synchronize()` | `torch.npu.synchronize()` |
| 内存查询 | `mem_get_info()` | `torch.npu.mem_get_info()` |
| 设备统计 | `get_npu_device_stats()` | 名称/内存/利用率等 |
| 设备可见性 | `get_visible_devices_keyword()` | `ASCEND_RT_VISIBLE_DEVICES` |

#### NPU 使用注意事项

- 设备可见性通过 `ASCEND_RT_VISIBLE_DEVICES` 环境变量控制（类似 `CUDA_VISIBLE_DEVICES`）
- Megatron-LM 后端会自动传入 `--distributed-backend hccl`，无需手动指定
- Megatron-LM 本身需使用支持 NPU 的版本（华为适配分支），避免内部 `torch.cuda` 硬编码
- NPU 上不支持 Transformer Engine，需使用 `local` transformer_impl

---

## 八、自定义任务开发

框架支持通过 YAML 配置文件快速定义新任务，也可通过继承 Python 类实现复杂逻辑。详见 `docs/new_task_guide.md`。

基本步骤：
1. 创建任务 YAML 配置文件，定义数据集、Prompt 模板、评估指标
2. 使用 Jinja2 语法编写 Prompt
3. 配置输出过滤和答案提取规则
4. 选择评估指标（准确率、F1、BLEU、ROUGE 等）

---

## 九、技术要求

- **Python**: >= 3.10
- **核心依赖**: datasets, numpy, evaluate, jinja2, scikit-learn, sacrebleu, rouge-score
- **GPU/NPU**: 支持 NVIDIA GPU (CUDA)、华为昇腾 NPU (torch_npu)、Intel GPU (XPU)、Apple Silicon (MPS)、CPU
- **多 GPU/NPU**: 需要 `accelerate` 或 `ray` 库；NPU 分布式需 HCCL（随 CANN 安装）

---

## 十、引用信息

```bibtex
@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and others},
  title        = {The Language Model Evaluation Harness},
  month        = 07,
  year         = {2024},
  publisher    = {Zenodo},
  version      = {v0.4.3},
  doi          = {10.5281/zenodo.12608602}
}
```

---

## 相关资源

| 资源 | 链接 |
|------|------|
| GitHub 仓库 | https://github.com/EleutherAI/lm-evaluation-harness |
| CLI 参考手册 | [docs/interface.md](docs/interface.md) |
| YAML 配置指南 | [docs/config_files.md](docs/config_files.md) |
| Python API 文档 | [docs/python-api.md](docs/python-api.md) |
| 新任务开发指南 | [docs/new_task_guide.md](docs/new_task_guide.md) |
| 模型使用指南 | [docs/model_guide.md](docs/model_guide.md) |
| 任务配置指南 | [lm_eval/tasks/README.md](lm_eval/tasks/README.md) |
| Open LLM Leaderboard | https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard |
| EleutherAI Discord | https://discord.gg/eleutherai |
