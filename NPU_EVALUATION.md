# Qwen3-8B Ascend-NPU 评估实现文档

## 评估结果

### HellaSwag (完整评估 - 40,168 样本)

| Tasks | Version | Filter | n-shot | Metric | Value | Stderr |
|-------|---------|--------|--------|--------|-------|--------|
| hellaswag | 1 | none | 0 | acc | **0.5709** | ±0.0049 |
| hellaswag | 1 | none | 0 | acc_norm | **0.7495** | ±0.0043 |

| 指标 | 数值 |
|------|------|
| 样本数 | 40,168 |
| 评估时间 | ~74 分钟 |
| 速度 | 前段 ~10 it/s, 后段 ~6 it/s (batch_size=8) |
| 硬件 | 4 × Ascend 910B NPU, TP=4 |
| 社区对照 (lm-eval, bf16) | 74.91% — 偏差仅 **+0.04%** |

## 环境

| 组件 | 版本/路径 |
|------|----------|
| CANN | 8.3.RC2 |
| torch | 2.7.1 |
| torch_npu | 2.7.1 |
| MindSpeed-LLM | 2.3.0 |
| lm-evaluation-harness | 0.4.12.dev0 |
| 模型检查点 | `mcore_tp4_pp1` (Megatron 格式, TP=4) |
| Tokenizer | HuggingFace Qwen2Tokenizer |

## 运行

```bash
# 快速启动
source set_env.sh
bash run.sh

# 自定义参数
source set_env.sh
export CKPT_PATH="/path/to/mcore_tp4_pp1"
export TOKENIZER_MODEL="/path/to/Qwen3-8B"
export MEGATRON_PATH="/path/to/MindSpeed-LLM"
bash npu_mindspeed-llm_eval.sh custom

# 单任务评估 (直接 torchrun)
torchrun --nproc-per-node=4 -m lm_eval \
    --model mindspeed_lm \
    --model_args "load=${CKPT_PATH},tokenizer_type=PretrainedFromHF,tokenizer_name_or_path=${TOKENIZER_MODEL},devices=4,tensor_model_parallel_size=4,seq_length=4096,micro_batch_size=1,spec=mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec,extra_args=--qk-layernorm --no-rope-fusion --no-persist-layer-norm --use-rotary-position-embeddings --swiglu --disable-bias-linear --group-query-attention --num-query-groups 8 --kv-channels 128 --normalization RMSNorm --position-embedding-type rope --norm-epsilon 1e-6 --transformer-impl local --ffn-hidden-size 12288 --make-vocab-size-divisible-by 1" \
    --tasks hellaswag --batch_size 8 --output_path results/ --log_samples
```

## 代码修改清单

### `lm_eval/models/mindspeed_lm.py` — 核心 NPU 适配

| 功能 | 说明 |
|------|------|
| `AscendNPUPatch` 类 | 统一管理 NPU 补丁: CUDA→NPU API 映射、nccl→hccl 重定向、默认生成器初始化、`_compile_dependencies` no-op、MindSpeed-LLM 适配器加载 |
| Tokenizer 回退 | `PretrainedFromHF` → `HuggingFaceTokenizer`, `tokenizer_name_or_path` → `tokenizer_model` |
| Spec 验证 | 在传递给 Megatron 前验证 spec 模块可导入性 |
| `load_args_from_checkpoint` 补丁 | 恢复 tokenizer 设置, 防止检查点中的 `Llama2Tokenizer` 覆盖 |
| Config GQA 覆盖 | `config.num_query_groups = args.num_query_groups` 确保 GQA 应用 |
| Padded vocab 修复 | 从 HF `config.json` 读取 `vocab_size` 覆盖 tokenizer 的值 |
| `--distributed-backend` | 已注释 — `AscendNPUPatch._patch_dist_backend()` 透明重定向 |

### `npu_mindspeed-llm_eval.sh`
- 移除 `spec`/`extra_args` 多余引号

### `run.sh`
- source `set_env.sh` + 设置 `PYTHONPATH`
- 为 NPU 添加 `--no-rope-fusion --no-persist-layer-norm --transformer-impl local --ffn-hidden-size 12288 --make-vocab-size-divisible-by 1`

### Megatron 源码修复 (调试代码移除)

| 文件 | 修改 |
|------|------|
| `megatron/core/transformer/mlp.py` | 移除 `debug_utils` 导入、`_save()` 调用、DEBUG print |
| `megatron/core/transformer/attention.py` | 移除 `debug_utils` 导入、`_save()/_dsave()` 调用、DEBUG print |
| `megatron/core/transformer/moe/moe_layer.py` | 移除 `debug_utils` 导入 |
| `megatron/core/models/common/embeddings/language_model_embedding.py` | 移除将输入覆盖为固定 4096 长度序列的调试劫持 |

## 关键技术问题

| # | 问题 | 根因 | 解决方案 |
|---|------|------|---------|
| 1 | 嵌入层调试劫持 (`148 and 4096 cannot broadcast`) | `language_model_embedding.py` 将输入替换为固定 5 个 token 并填充到 4096 | 移除调试代码 |
| 2 | Megatron 调试模块导入 (`No module named 'debug_utils'`) | Megatron 代码引用不存在的调试路径 | 注释所有 `debug_utils` 导入和调用 |
| 3 | Tokenizer 类型不兼容 (`PretrainedFromHF` 无效) | 基础 Megatron 不支持此扩展类型 | 回退到 `HuggingFaceTokenizer` |
| 4 | 检查点覆盖 Tokenizer (`Llama2Tokenizer`) | `load_args_from_checkpoint` 强制覆盖 `tokenizer_type` | Monkey-patch 恢复设置 |
| 5 | GQA 未应用 (QKV 形状 `[3072]` vs `[1536]`) | 检查点 `group_query_attention=False` | 在 `model_provider` 中显式设置 `config.num_query_groups` |
| 6 | 词汇量大小不匹配 (`[38016]` vs `[37984]`) | Tokenizer 的 `vocab_size` (151669) ≠ 检查点的 (151936) | 从 HF `config.json` 覆盖 `args.padded_vocab_size` |
| 7 | NPU 默认生成器为空 (`IndexError`) | `transfer_to_npu` 后 `default_generators` 为空 | 为每个 NPU 设备初始化 `torch.Generator` |
| 8 | CUDA 内核编译失败 (`TypeError: NoneType + str`) | `CUDA_HOME` 为 None | 在 NPU 上 monkey-patch `_compile_dependencies` 为 no-op |
| 9 | 融合 CUDA 操作不可用 | 无 Apex/TransformerEngine | 添加 `--no-rope-fusion --no-persist-layer-norm --transformer-impl local` |
| 10 | 结果表格显示缺失 (`pytablewriter`) | vllm011 环境缺少依赖 | `pip install pytablewriter` |

## 评估流程

```
run.sh
 ├── set_env.sh          → CANN + conda 环境
 └── npu_mindspeed-llm_eval.sh custom
      └── torchrun --nproc-per-node=4 -m lm_eval
           └── MindSpeedLMEval.__init__()
                ├── AscendNPUPatch.apply()     → CUDA→NPU 补丁
                ├── AscendNPUPatch.load_mindspeed_adaptor()
                ├── Tokenizer fallback (PretrainedFromHF → HuggingFace)
                ├── _initialize_megatron()
                │    ├── AscendNPUPatch.patch_compile_dependencies()
                │    ├── load_args_from_checkpoint → 补丁恢复 tokenizer
                │    ├── model_provider → GPTModel + GQA config 覆盖
                │    │    ├── get_gpt_layer_local_spec()
                │    │    ├── config.num_query_groups = 8
                │    │    └── _override_attn_mask()
                │    ├── get_model → 加载检查点权重
                │    └── _fix_padded_vocab_size() → HF config 覆盖
                └── 评估循环 (loglikelihood)
```
