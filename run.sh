#!/bin/bash
source ~/.bashrc
cd /home/jianzhnie/llmtuner/llm/MindSpeed-LLM


ckpt_path="/home/jianzhnie/llmtuner/hfhub/mindspeed/models/Qwen/Qwen3-8B/mcore_tp4_pp1"
tokenizer_path="/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-8B"
export CKPT_PATH=$ckpt_path
export MEGATRON_PATH=/home/jianzhnie/llmtuner/llm/MindSpeed-LLM
export TOKENIZER_MODEL=$tokenizer_path
export TASKS="hellaswag,arc_easy,winogrande"
export TP_SIZE=4
export PP_SIZE=1
export NUM_DEVICES=4
export SEQ_LENGTH=4096
export OUTPUT_PATH=results/Qwen3-8B/

bash /home/jianzhnie/llmtuner/llm/lm-evaluation-harness/npu_mindspeed-llm_eval.sh custom