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
#   MEGATRON_PATH        - MindSpeed-LLM installation path
#   TASKS                - Tasks to evaluate (default: hellaswag)
#   TP_SIZE              - Tensor parallelism size (default: 1)
#   NUM_DEVICES          - Number of NPUs to use (default: 1)
#   SPEC                 - Custom layer spec module path (optional)
#   EXTRA_ARGS           - Extra Megatron-LM arguments (optional)
# =============================================================================
source ~/.bashrc
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
export TP_SIZE="${TP_SIZE:-1}"
export PP_SIZE="${PP_SIZE:-1}"
export NUM_DEVICES="${NUM_DEVICES:-1}"
export SEQ_LENGTH="${SEQ_LENGTH:-4096}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export SEED="${SEED:-42}"
export OUTPUT_PATH="${OUTPUT_PATH:-results/eval}"

# ----- Optional: Custom layer spec (uncomment for specific models) -----
# export SPEC="mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec"

# ----- Optional: Extra Megatron-LM arguments (uncomment as needed) -----
# export EXTRA_ARGS="--qk-layernorm --use-rotary-position-embeddings --swiglu --disable-bias-linear --group-query-attention --num-query-groups 8 --kv-channels 128 --normalization RMSNorm --position-embedding-type rope --norm-epsilon 1e-6"

bash "$(dirname "$0")/npu_mindspeed-llm_eval.sh" custom
