#!/bin/bash
# =============================================================================
# Quick-start script for NPU + MindSpeed-LLM evaluation
#
# Usage:
#   1. Edit the paths below to match your environment
#   2. Run: bash run.sh
#
# Environment variables (override defaults):
#   CKPT_PATH            - Model checkpoint path
#   TOKENIZER_MODEL      - Tokenizer path (file or directory)
#   TOKENIZER_NAME_OR_PATH - Tokenizer path for PretrainedFromHF type
#   TOKENIZER_TYPE       - Tokenizer type (default: HuggingFaceTokenizer)
#   MEGATRON_PATH        - MindSpeed-LLM installation path
#   TASKS                - Tasks to evaluate (default: hellaswag)
#   TP_SIZE              - Tensor parallelism size (default: 1)
#   NUM_DEVICES          - Number of NPUs to use (default: 1)
#   SPEC                 - Custom layer spec module path (optional)
#   EXTRA_ARGS           - Extra Megatron-LM arguments (optional)
# =============================================================================
source /home/jianzhnie/llmtuner/llm/lm-evaluation-harness/set_env.sh 2>/dev/null
# Ensure lm_eval is importable from the source directory
export PYTHONPATH="/home/jianzhnie/llmtuner/llm/lm-evaluation-harness:${PYTHONPATH:-}"
cd /home/jianzhnie/llmtuner/llm/MindSpeed-LLM

# ----- Edit these paths to match your environment -----
ckpt_path="/home/jianzhnie/llmtuner/hfhub/mindspeed/models/Qwen/Qwen3-8B/mcore_tp4_pp1"
tokenizer_path="/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-8B"
mindspeed_llm_path="/home/jianzhnie/llmtuner/llm/MindSpeed-LLM"

CKPT_PATH="${CKPT_PATH:-$ckpt_path}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-$tokenizer_path}"
MEGATRON_PATH="${MEGATRON_PATH:-$mindspeed_llm_path}"

# ----- Export for npu_mindspeed-llm_eval.sh -----
export CKPT_PATH
export MEGATRON_PATH
export TOKENIZER_MODEL

# ----- Model & evaluation config -----
export TASKS="${TASKS:-hellaswag}"
export TP_SIZE="${TP_SIZE:-4}"
export PP_SIZE="${PP_SIZE:-1}"
export NUM_DEVICES="${NUM_DEVICES:-4}"
export SEQ_LENGTH="${SEQ_LENGTH:-4096}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export SEED="${SEED:-42}"
export OUTPUT_PATH="${OUTPUT_PATH:-results/eval}"

# ----- Tokenizer config -----
# For PretrainedFromHF tokenizer type, uncomment and set:
export TOKENIZER_TYPE="PretrainedFromHF"
export TOKENIZER_NAME_OR_PATH="${TOKENIZER_MODEL}"

# ----- Optional: Custom layer spec -----
# Uncomment for specific models:
# Qwen3:
export SPEC="mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec"
# Qwen2 MoE:
# export SPEC="mindspeed_llm.tasks.models.spec.qwen2_moe_spec layer_spec"

# ----- Optional: Extra Megatron-LM arguments -----
# Uncomment and customize for your model:
# Qwen3-0.6B:
# export USE_CHECKPOINT_ARGS=false
# export EXTRA_ARGS="--qk-layernorm --use-rotary-position-embeddings --swiglu --disable-bias-linear --group-query-attention --num-query-groups 8 --kv-channels 128 --normalization RMSNorm --position-embedding-type rope --norm-epsilon 1e-6 --padded-vocab-size 151936 --make-vocab-size-divisible-by 1 --rotary-base 1000000 --num-layers 28 --hidden-size 1024 --num-attention-heads 16 --ffn-hidden-size 3072"

# Qwen3-8B (with use_checkpoint_args=true, mcore checkpoint):
# NOTE: --no-rope-fusion and --no-persist-layer-norm are required when
# mindspeed_llm is not available (no Apex/TE installed on NPU).
# --ffn-hidden-size 12288 and --make-vocab-size-divisible-by 1 are required
# to match the checkpoint (prevent auto-computation and vocab padding).
export EXTRA_ARGS="--qk-layernorm --no-rope-fusion --no-persist-layer-norm --use-rotary-position-embeddings --swiglu --disable-bias-linear --group-query-attention --num-query-groups 8 --kv-channels 128 --normalization RMSNorm --position-embedding-type rope --norm-epsilon 1e-6 --transformer-impl local --ffn-hidden-size 12288 --make-vocab-size-divisible-by 1"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

bash "${SCRIPT_DIR}/npu_mindspeed-llm_eval.sh" custom
