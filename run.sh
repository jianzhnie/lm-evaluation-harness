#!/bin/bash
# =============================================================================
# Quick-start script for NPU + MindSpeed-LLM evaluation
# Edit the variables below to match your environment, then run: bash run.sh
# =============================================================================
source ~/.bashrc
cd /home/jianzhnie/llmtuner/llm/MindSpeed-LLM

# ----- Edit these paths -----
ckpt_path="/home/jianzhnie/llmtuner/hfhub/mindspeed/models/Qwen/Qwen3-8B/mcore_tp4_pp1"
tokenizer_path="/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-8B"
mindspeed_llm_path="/home/jianzhnie/llmtuner/llm/MindSpeed-LLM"

CKPT_PATH="${CKPT_PATH:-$ckpt_path}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-$tokenizer_path}"
MEGATRON_PATH="${MEGATRON_PATH:-$mindspeed_llm_path}"

# ----- Model & evaluation config -----
export CKPT_PATH
export MEGATRON_PATH
export TOKENIZER_MODEL
export TASKS="${TASKS:-hellaswag,arc_easy,winogrande}"
export TP_SIZE="${TP_SIZE:-4}"
export PP_SIZE="${PP_SIZE:-1}"
export NUM_DEVICES="${NUM_DEVICES:-4}"
export SEQ_LENGTH="${SEQ_LENGTH:-4096}"
export OUTPUT_PATH="${OUTPUT_PATH:-results/eval}"

bash "$(dirname "$0")/npu_mindspeed-llm_eval.sh" custom
