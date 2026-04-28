#!/bin/bash
# =============================================================================
# NPU (Ascend) + MindSpeed-LLM Evaluation Script
# =============================================================================
# Prerequisites:
#   1. CANN driver & toolkit installed (see Huawei Ascend documentation)
#   2. torch_npu installed:  pip install torch_npu
#   3. Megatron-LM installed or MEGATRON_PATH set
#   4. lm-eval installed:    pip install -e ".[hf]"
#
# Usage:
#   bash npu_mindspeed-llm_eval.sh
#   bash npu_mindspeed-llm_eval.sh <mode>
#
# Modes:
#   single  - Single NPU evaluation (default)
#   dp      - Data Parallelism (4 NPUs)
#   tp      - Tensor Parallelism (2 NPUs)
#   ep      - Expert Parallelism for MoE models (4 NPUs)
#   custom  - Custom configuration via environment variables
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configurable variables (override via environment or edit directly)
# ---------------------------------------------------------------------------
# MindSpeed-LLM path (set MEGATRON_PATH to MindSpeed-LLM root)
: "${MEGATRON_PATH:=/path/to/MindSpeed-LLM}"
export MEGATRON_PATH
# Checkpoint & tokenizer
: "${CKPT_PATH:=/path/to/checkpoint}"
: "${TOKENIZER_MODEL:=/path/to/tokenizer.model}"
: "${TOKENIZER_TYPE:=HuggingFaceTokenizer}"

# Evaluation tasks
: "${TASKS:=hellaswag}"
: "${BATCH_SIZE:=8}"
: "${OUTPUT_PATH:=results/npu_megatron}"

# NPU device visibility (override to select specific NPUs)
# e.g. ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
: "${ASCEND_RT_VISIBLE_DEVICES:=}"

# Mode selection
MODE="${1:-single}"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') $*"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $*" >&2
}

check_env() {
    if [ ! -d "${MEGATRON_PATH}" ]; then
        log_error "MEGATRON_PATH not found: ${MEGATRON_PATH}"
        log_error "Please set MEGATRON_PATH to your Megatron-LM installation directory"
        exit 1
    fi

    if [ ! -d "${CKPT_PATH}" ]; then
        log_error "Checkpoint path not found: ${CKPT_PATH}"
        log_error "Please set CKPT_PATH to your Megatron-LM checkpoint directory"
        exit 1
    fi

    if [ ! -e "${TOKENIZER_MODEL}" ]; then
        log_error "Tokenizer model not found: ${TOKENIZER_MODEL}"
        log_error "Please set TOKENIZER_MODEL to your tokenizer file path"
        exit 1
    fi

    python -c "import torch_npu" 2>/dev/null || {
        log_error "torch_npu is not installed. Run: pip install torch_npu"
        exit 1
    }

    log_info "Environment check passed"
    log_info "  MEGATRON_PATH          = ${MEGATRON_PATH}"
    log_info "  CKPT_PATH              = ${CKPT_PATH}"
    log_info "  TOKENIZER_MODEL        = ${TOKENIZER_MODEL}"
    log_info "  ASCEND_RT_VISIBLE_DEVICES = ${ASCEND_RT_VISIBLE_DEVICES:-<all>}"
}

# Base model args shared across all modes
base_model_args() {
    echo "load=${CKPT_PATH},tokenizer_type=${TOKENIZER_TYPE},tokenizer_model=${TOKENIZER_MODEL}"
}

# ---------------------------------------------------------------------------
# Mode: Single NPU
# ---------------------------------------------------------------------------
run_single() {
    log_info "=== Mode: Single NPU ==="

    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

    torchrun --nproc-per-node=1 -m lm_eval \
        --model mindspeed_lm \
        --model_args "$(base_model_args),devices=1" \
        --tasks "${TASKS}" \
        --batch_size "${BATCH_SIZE}" \
        --output_path "${OUTPUT_PATH}/single" \
        --log_samples
}

# ---------------------------------------------------------------------------
# Mode: Data Parallelism (4 NPUs)
#   Each NPU loads a full model replica, data is distributed across NPUs.
# ---------------------------------------------------------------------------
run_dp() {
    local num_devices="${NUM_DEVICES:-4}"
    log_info "=== Mode: Data Parallelism (${num_devices} NPUs) ==="

    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$(seq -s, 0 $((num_devices - 1)))}"

    torchrun --nproc-per-node="${num_devices}" -m lm_eval \
        --model mindspeed_lm \
        --model_args "$(base_model_args),devices=${num_devices}" \
        --tasks "${TASKS}" \
        --batch_size "${BATCH_SIZE}" \
        --output_path "${OUTPUT_PATH}/dp_${num_devices}" \
        --log_samples
}

# ---------------------------------------------------------------------------
# Mode: Tensor Parallelism (2 NPUs)
#   Model layers are split across NPUs using Tensor Parallelism.
# ---------------------------------------------------------------------------
run_tp() {
    local num_devices="${NUM_DEVICES:-2}"
    log_info "=== Mode: Tensor Parallelism (TP=${num_devices}) ==="

    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$(seq -s, 0 $((num_devices - 1)))}"

    torchrun --nproc-per-node="${num_devices}" -m lm_eval \
        --model mindspeed_lm \
        --model_args "$(base_model_args),devices=${num_devices},tensor_model_parallel_size=${num_devices}" \
        --tasks "${TASKS}" \
        --batch_size "${BATCH_SIZE}" \
        --output_path "${OUTPUT_PATH}/tp_${num_devices}" \
        --log_samples
}

# ---------------------------------------------------------------------------
# Mode: Expert Parallelism for MoE models (4 NPUs)
#   MoE experts are distributed across NPUs, tokens routed via All-to-All.
# ---------------------------------------------------------------------------
run_ep() {
    local num_devices="${NUM_DEVICES:-4}"
    log_info "=== Mode: Expert Parallelism (EP=${num_devices}) ==="

    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$(seq -s, 0 $((num_devices - 1)))}"

    torchrun --nproc-per-node="${num_devices}" -m lm_eval \
        --model mindspeed_lm \
        --model_args "$(base_model_args),devices=${num_devices},expert_model_parallel_size=${num_devices}" \
        --tasks "${TASKS}" \
        --batch_size "${BATCH_SIZE}" \
        --output_path "${OUTPUT_PATH}/ep_${num_devices}" \
        --log_samples
}

# ---------------------------------------------------------------------------
# Mode: Custom (all parameters via environment variables)
#   Allows arbitrary Megatron-LM configurations.
#
#   Additional environment variables:
#     TP_SIZE              - Tensor parallelism size (default: 1)
#     EP_SIZE              - Expert parallelism size (default: 1)
#     SEQ_LENGTH           - Maximum sequence length (default: 4096)
#     MAX_GEN_TOKS         - Maximum tokens to generate (default: 256)
#     USE_CHECKPOINT_ARGS  - Whether to load model args from checkpoint (default: false)
#                            Set to "true" only if checkpoint tokenizer is compatible.
#     EXTRA_ARGS           - Extra Megatron-LM arguments (space-separated)
#
#   Example:
#     TP_SIZE=2 EP_SIZE=1 NUM_DEVICES=2 \
#     EXTRA_ARGS="--no-rope-fusion --trust-remote-code" \
#     bash npu_megatron_llm_eval.sh custom
# ---------------------------------------------------------------------------
run_custom() {
    local num_devices="${NUM_DEVICES:-1}"
    local tp_size="${TP_SIZE:-1}"
    local pp_size="${PP_SIZE:-1}"
    local ep_size="${EP_SIZE:-1}"
    local seq_length="${SEQ_LENGTH:-4096}"
    local max_gen_toks="${MAX_GEN_TOKS:-256}"

    log_info "=== Mode: Custom ==="
    log_info "  devices=${num_devices}, TP=${tp_size}, PP=${pp_size}, EP=${ep_size}"
    log_info "  seq_length=${seq_length}, max_gen_toks=${max_gen_toks}"

    # CUDA_DEVICE_MAX_CONNECTIONS=1 is recommended for tensor parallelism
    if [ "${tp_size}" -gt 1 ]; then
        export CUDA_DEVICE_MAX_CONNECTIONS=1
    fi

    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$(seq -s, 0 $((num_devices - 1)))}"

    local use_ckpt_args="${USE_CHECKPOINT_ARGS:-false}"

    local model_args
    model_args="$(base_model_args),devices=${num_devices},tensor_model_parallel_size=${tp_size},pipeline_model_parallel_size=${pp_size},expert_model_parallel_size=${ep_size},seq_length=${seq_length},max_gen_toks=${max_gen_toks}"

    # Only pass use_checkpoint_args when explicitly set (default is True in Python)
    if [ -n "${USE_CHECKPOINT_ARGS:-}" ]; then
        model_args="${model_args},use_checkpoint_args=${USE_CHECKPOINT_ARGS}"
    fi

    if [ -n "${EXTRA_ARGS:-}" ]; then
        model_args="${model_args},extra_args=\"${EXTRA_ARGS}\""
        log_info "  extra_args=${EXTRA_ARGS}"
    fi

    torchrun --nproc-per-node="${num_devices}" -m lm_eval \
        --model mindspeed_lm \
        --model_args "${model_args}" \
        --tasks "${TASKS}" \
        --batch_size "${BATCH_SIZE}" \
        --output_path "${OUTPUT_PATH}/custom" \
        --log_samples
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    log_info "Starting NPU Megatron-LM evaluation (mode: ${MODE})"
    check_env

    case "${MODE}" in
        single)
            run_single
            ;;
        dp)
            run_dp
            ;;
        tp)
            run_tp
            ;;
        ep)
            run_ep
            ;;
        custom)
            run_custom
            ;;
        *)
            log_error "Unknown mode: ${MODE}"
            echo ""
            echo "Usage: bash $0 [single|dp|tp|ep|custom]"
            echo ""
            echo "Modes:"
            echo "  single  - Single NPU evaluation (default)"
            echo "  dp      - Data Parallelism (4 NPUs)"
            echo "  tp      - Tensor Parallelism (2 NPUs)"
            echo "  ep      - Expert Parallelism for MoE (4 NPUs)"
            echo "  custom  - Custom config via environment variables"
            echo ""
            echo "Examples:"
            echo "  bash $0 single"
            echo "  NUM_DEVICES=8 bash $0 dp"
            echo "  NUM_DEVICES=4 bash $0 tp"
            echo "  CKPT_PATH=/data/moe_ckpt NUM_DEVICES=8 bash $0 ep"
            echo "  TP_SIZE=2 NUM_DEVICES=2 EXTRA_ARGS='--no-rope-fusion' bash $0 custom"
            exit 1
            ;;
    esac

    log_info "Evaluation completed. Results saved to: ${OUTPUT_PATH}"
}

main
