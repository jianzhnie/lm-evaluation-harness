# NPU + Megatron-LM 评估使用指南

## 快速使用

```shell
# 1. 设置必要变量
export MEGATRON_PATH=/path/to/Megatron-LM
export CKPT_PATH=/path/to/checkpoint
export TOKENIZER_MODEL=/path/to/tokenizer.model

# 2. 选择模式运行
bash npu_megatron_llm_eval.sh single   # 单卡
bash npu_megatron_llm_eval.sh dp       # 4卡数据并行
bash npu_megatron_llm_eval.sh tp       # 2卡张量并行
bash npu_megatron_llm_eval.sh ep       # 4卡专家并行(MoE)

# 3. 自定义参数
NUM_DEVICES=8 TASKS="hellaswag,arc_easy" bash npu_megatron_llm_eval.sh dp
TP_SIZE=4 NUM_DEVICES=4 bash npu_megatron_llm_eval.sh tp
EXTRA_ARGS="--no-rope-fusion --trust-remote-code" bash npu_megatron_llm_eval.sh custom
```

## 脚本特性

- 自动设置 `ASCEND_RT_VISIBLE_DEVICES`，也支持手动覆盖
- 启动前自动检查 `torch_npu`、`MEGATRON_PATH`、checkpoint、tokenizer 是否存在
- 所有参数均可通过环境变量覆盖
- `custom` 模式支持传入任意 Megatron-LM 参数 (`EXTRA_ARGS`)

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MEGATRON_PATH` | `/path/to/Megatron-LM` | Megatron-LM 安装路径 |
| `CKPT_PATH` | `/path/to/checkpoint` | 模型 checkpoint 目录 |
| `TOKENIZER_MODEL` | `/path/to/tokenizer.model` | tokenizer 文件路径 |
| `TOKENIZER_TYPE` | `HuggingFaceTokenizer` | tokenizer 类型 |
| `TASKS` | `hellaswag` | 评测任务（逗号分隔） |
| `BATCH_SIZE` | `8` | 批次大小 |
| `NUM_DEVICES` | 模式相关 | 使用的 NPU 数量 |
| `OUTPUT_PATH` | `results/npu_megatron` | 结果输出路径 |
| `ASCEND_RT_VISIBLE_DEVICES` | (自动) | NPU 可见设备，类似 `CUDA_VISIBLE_DEVICES` |

### custom 模式额外参数

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TP_SIZE` | `1` | 张量并行度 |
| `EP_SIZE` | `1` | 专家并行度 |
| `SEQ_LENGTH` | `4096` | 最大序列长度 |
| `MAX_GEN_TOKS` | `256` | 最大生成 token 数 |
| `EXTRA_ARGS` | (空) | 额外 Megatron-LM 参数 |

## 典型示例

```shell
# 大模型 TP 评估 (4卡)
NUM_DEVICES=4 SEQ_LENGTH=4096 TASKS="hellaswag,mmlu_abstract_algebra" \
    bash npu_megatron_llm_eval.sh tp

# MoE 模型 EP 评估 (8卡)
CKPT_PATH=/data/moe_ckpt NUM_DEVICES=8 \
    bash npu_megatron_llm_eval.sh ep

# 完整自定义
CKPT_PATH=/data/models/llama-7b \
TOKENIZER_MODEL=/data/models/llama-7b/tokenizer.model \
TASKS="hellaswag,arc_easy,winogrande" \
TP_SIZE=2 NUM_DEVICES=2 SEQ_LENGTH=4096 \
OUTPUT_PATH=results/llama-7b \
    bash npu_megatron_llm_eval.sh custom
```
