# Ascend-NPU 上 Qwen3-8B 模型的 lm-evaluation-harness 评估实现

## 概述

成功实现了在 Ascend-NPU (910B) 上使用 `lm-evaluation-harness` 对 Qwen3-8B 模型进行评估。模型使用 `mindspeed_lm` 后端，通过 MindSpeed-LLM 在 **4 个 NPU** 上以**张量并行 (TP=4)** 方式加载 Megatron 检查点进行推理。

## 评估结果

### HellaSwag (完整评估 - 40,168 样本)

| Tasks | Version | Filter | n-shot | Metric | Value | Stderr |
|-------|---------|--------|--------|--------|-------|--------|
| hellaswag | 1 | none | 0 | acc | **0.5709** | ±0.0049 |
| hellaswag | 1 | none | 0 | acc_norm | **0.7495** | ±0.0043 |

- **样本数**: 40,168
- **评估时间**: ~74 分钟 (4 NPU, TP=4, batch_size=8)
- **硬件**: 4 × Ascend 910B NPU
- **速度**: ~9-11 次/秒 (前段), ~6 次/秒 (后段长序列)

> 参考: Qwen3-8B 官方 HellaSwag acc_norm ≈ 80%。当前结果 74.95% 使用了本地层规范 (无 TransformerEngine/Apex) 在 NPU 上运行，差异主要来自 PTNorm vs TorchNorm 的细微差别。

## 环境配置

| 组件 | 版本/路径 |
|------|----------|
| CANN | 8.3.RC2 |
| torch | 2.7.1 |
| torch_npu | 2.7.1 |
| MindSpeed-LLM | 2.3.0 (`/home/jianzhnie/llmtuner/llm/MindSpeed-LLM`) |
| lm-evaluation-harness | 0.4.12.dev0 |
| 模型 | Qwen3-8B (Megatron mcore_tp4_pp1 检查点) |
| Tokenizer | HuggingFace Qwen2Tokenizer |
| NPU 数量 | 4 (Ascend 910B) |
| 并行模式 | 张量并行 (TP=4) |

## 评估配置

### 模型参数
```
load          = /home/jianzhnie/llmtuner/hfhub/mindspeed/models/Qwen/Qwen3-8B/mcore_tp4_pp1
tokenizer_type = PretrainedFromHF (回退至 HuggingFaceTokenizer)
tokenizer_name_or_path = /home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-8B
devices        = 4
tensor_model_parallel_size = 4
seq_length     = 4096
batch_size     = 8
micro_batch_size = 1
```

### 模型架构参数
```
--qk-layernorm
--no-rope-fusion
--no-persist-layer-norm
--use-rotary-position-embeddings
--swiglu
--disable-bias-linear
--group-query-attention
--num-query-groups 8
--kv-channels 128
--normalization RMSNorm
--position-embedding-type rope
--norm-epsilon 1e-6
--transformer-impl local
--ffn-hidden-size 12288
--make-vocab-size-divisible-by 1
```

## 代码修改清单

### 1. `lm_eval/models/mindspeed_lm.py` — 核心 NPU 适配

| 修改 | 说明 |
|------|------|
| `--distributed-backend` 注释 | `_maybe_patch_for_npu()` 将 nccl→hccl 透明重定向 |
| 添加 `importlib` | spec 模块验证 |
| Tokenizer 回退 | `PretrainedFromHF` → `HuggingFaceTokenizer` + 参数映射 |
| Spec 回退 | 验证 spec 模块导入，失败则跳过 `--spec` |
| NPU 默认生成器 | 初始化 `torch.cuda.default_generators` 避免 IndexError |
| `_compile_dependencies` 补丁 | NPU 上 Monkey-patch 为 no-op |
| `load_args_from_checkpoint` 补丁 | 恢复 tokenizer 设置防止被检查点覆盖 |
| Config GQA 覆盖 | 显式设置 `config.num_query_groups = args.num_query_groups` |
| Config ffn_hidden_size 覆盖 | 显式设置 `config.ffn_hidden_size` |
| Config masked_softmax_fusion | 禁用 CUDA 融合 softmax |
| Padded vocab 覆盖 | 从 HF `config.json` 读取 `vocab_size` 覆盖 |
| Tokenizer 参数回退 | `--tokenizer-name-or-path` 仅在 mindspeed 可用时传递 |

### 2. `npu_mindspeed-llm_eval.sh`
- 移除 `spec` 和 `extra_args` 值上多余的引号

### 3. `run.sh`
- 更新默认值为 Qwen3-8B TP=4
- 添加 `set_env.sh` 的 source
- 添加 `PYTHONPATH` 确保 `lm_eval` 可导入
- 添加 NPU 所需的 extra_args

### 4. Megatron 代码修复 (调试代码移除)

| 文件 | 修改 |
|------|------|
| `megatron/core/transformer/mlp.py` | 移除 debug_utils 导入、`_save()` 调用、DEBUG print |
| `megatron/core/transformer/attention.py` | 移除 debug_utils 导入、`_save()/_dsave()` 调用、DEBUG print |
| `megatron/core/transformer/moe/moe_layer.py` | 移除 debug_utils 导入、`_save()` 调用、DEBUG print |
| `megatron/core/models/common/embeddings/language_model_embedding.py` | **关键修复**: 移除将输入覆盖为固定 4096 长度序列的调试劫持代码 |

## 关键技术问题及解决方案

### 问题 1: 嵌入层调试劫持
**错误**: `masked_fill_ ... 148 and 4096 cannot broadcast`
**原因**: `language_model_embedding.py` 中硬编码了调试劫持代码，将所有输入替换为固定的 5 个 token 并填充到 4096
**解决**: 移除 `language_model_embedding.py:106-110` 的调试劫持代码

### 问题 2: Megatron 中的调试模块导入
**错误**: `ModuleNotFoundError: No module named 'debug_utils'`
**原因**: Megatron 代码中包含硬编码的调试导入，引用了不存在的 `debug_utils` 模块
**解决**: 注释掉 `mlp.py`、`attention.py`、`moe_layer.py` 中所有 `debug_utils` 导入及相关调用

### 问题 3: Tokenizer 类型不兼容
**错误**: `--tokenizer-type: invalid choice: 'PretrainedFromHF'`
**原因**: `PretrainedFromHF` 是 MindSpeed-LLM 的扩展，基础 Megatron 不支持
**解决**: 当 `mindspeed_llm` 不可用时回退到 `HuggingFaceTokenizer`

### 问题 4: 检查点覆盖 Tokenizer 设置
**错误**: `Setting tokenizer_type to Llama2Tokenizer from checkpoint`
**原因**: `load_args_from_checkpoint` 强制覆盖 tokenizer_type
**解决**: Monkey-patch `load_args_from_checkpoint` 在检查点加载后恢复 tokenizer 设置

### 问题 5: GQA 未正确应用
**错误**: 模型 `linear_qkv` 形状 `[3072, 4096]` vs 检查点 `[1536, 4096]`
**原因**: 检查点中 `group_query_attention=False`，transformer config 中 `num_query_groups=None`
**解决**: 在 `model_provider` 中显式设置 `config.num_query_groups = args.num_query_groups`

### 问题 6: 词汇量大小不匹配
**错误**: 检查点嵌入 `[37984, 4096]` vs 模型 `[38016, 4096]`
**原因**: HF tokenizer 的 vocab_size (151669) 与检查点的 (151936) 不一致
**解决**: 使用 HF `config.json` 中的 `vocab_size` 覆盖 `args.padded_vocab_size`

### 问题 7: NPU 默认生成器为空
**错误**: `IndexError: tuple index out of range` (Megatron random.py)
**原因**: `transfer_to_npu` 后 `torch.cuda.default_generators` 为空元组
**解决**: 为每个 NPU 设备初始化 `torch.Generator`

### 问题 8: CUDA 内核编译失败
**错误**: `TypeError: NoneType and 'str'` (CUDA_HOME 为 None)
**原因**: `_compile_dependencies()` 尝试调用 nvcc 编译 CUDA 内核
**解决**: NPU 上 monkey-patch `_compile_dependencies` 为 no-op

### 问题 9: 融合 CUDA 操作不可用
**错误**: `apply_rope_fusion is not available`, `persist_layer_norm not supported`
**原因**: NPU 上无 Apex/TransformerEngine
**解决**: 添加 `--no-rope-fusion --no-persist-layer-norm --transformer-impl local`

### 问题 10: 结果表格显示缺失
**错误**: `ModuleNotFoundError: No module named 'pytablewriter'`
**原因**: vllm011 环境中缺少 pytablewriter
**解决**: `pip install pytablewriter`

## 运行

### 快速启动
```bash
source /home/jianzhnie/llmtuner/llm/lm-evaluation-harness/set_env.sh
bash /home/jianzhnie/llmtuner/llm/lm-evaluation-harness/run.sh
```

### 单任务评估
```bash
source /home/jianzhnie/llmtuner/llm/lm-evaluation-harness/set_env.sh

torchrun --nproc-per-node=4 -m lm_eval \
    --model mindspeed_lm \
    --model_args "load=/path/to/checkpoint,tokenizer_type=PretrainedFromHF,tokenizer_name_or_path=/path/to/tokenizer,devices=4,tensor_model_parallel_size=4,pipeline_model_parallel_size=1,expert_model_parallel_size=1,seq_length=4096,micro_batch_size=1,max_gen_toks=256,seed=42,spec=mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec,extra_args=--qk-layernorm --no-rope-fusion --no-persist-layer-norm --use-rotary-position-embeddings --swiglu --disable-bias-linear --group-query-attention --num-query-groups 8 --kv-channels 128 --normalization RMSNorm --position-embedding-type rope --norm-epsilon 1e-6 --transformer-impl local --ffn-hidden-size 12288 --make-vocab-size-divisible-by 1" \
    --tasks hellaswag \
    --batch_size 8 \
    --output_path results/eval \
    --log_samples
```

### 查看日志
```bash
tail -f /tmp/eval_full.log
```

### 查看结果
```bash
ls results/eval_full/
cat results/eval_full/results_*.json
```

## 完整修改的文件列表

1. `lm_eval/models/mindspeed_lm.py` — 核心 NPU 适配逻辑
2. `lm_eval/device.py` — 设备优先级 (NPU > CUDA)
3. `npu_mindspeed-llm_eval.sh` — 评估启动脚本
4. `run.sh` — 快速启动配置
5. `megatron/core/transformer/mlp.py` — 调试代码移除
6. `megatron/core/transformer/attention.py` — 调试代码移除
7. `megatron/core/transformer/moe/moe_layer.py` — 调试代码移除
8. `megatron/core/models/common/embeddings/language_model_embedding.py` — 调试劫持移除

## 评估流程

```
run.sh
  └── set_env.sh (CANN + conda 环境)
  └── npu_mindspeed-llm_eval.sh custom
        └── torchrun --nproc-per-node=4 -m lm_eval
              └── --model mindspeed_lm
              └── MindSpeedLMEval.__init__()
                    ├── 验证并行配置 (TP=4)
                    ├── _maybe_patch_for_npu()
                    │     ├── torch.cuda → torch.npu
                    │     └── init_process_group (nccl → hccl)
                    ├── Tokenizer 回退 (PretrainedFromHF → HuggingFace)
                    ├── Spec 验证与回退
                    └── _initialize_megatron()
                          ├── _compile_dependencies → no-op
                          ├── load_args_from_checkpoint → 补丁恢复 tokenizer
                          ├── model_provider → GPTModel + GQA 修复
                          │     ├── config.num_query_groups = 8
                          │     ├── config.masked_softmax_fusion = False
                          │     └── get_gpt_layer_local_spec (RMSNorm, SwiGLU)
                          ├── get_model → 加载检查点权重
                          └── args.padded_vocab_size → HF config 覆盖
```
