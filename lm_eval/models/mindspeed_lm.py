r"""
MindSpeed-LLM backend for lm-evaluation-harness.

This module provides support for evaluating Megatron-LM models, including
both standard checkpoints and distributed checkpoints (.distcp format).

Parallelism Modes:
    1. Single GPU: devices=1, TP=1, PP=1
       - Standard single GPU evaluation

    2. Data Parallelism (DP): devices > 1, TP=1, PP=1
       - Each GPU has a full model replica
       - Data is distributed across GPUs
       - Results are gathered from all ranks

    3. Tensor Parallelism (TP): TP == devices, PP=1
       - Model layers are split across GPUs using Tensor Parallelism
       - No data parallelism

    4. Expert Parallelism (EP) for MoE models: EP > 1, TP=1, PP=1, devices=EP
       - Each GPU holds different experts
       - Tokens are routed via All-to-All communication
       - EP cannot be combined with TP or PP

Note: Pipeline Parallelism is supported for TP + PP combinations (devices = TP × PP).

Requirements:
    - MindSpeed-LLM must be installed and accessible via MEGATRON_PATH environment variable
    - PyTorch with CUDA or NPU (Ascend, via torch_npu) support
    - torch_npu package for NPU (Ascend) support

Usage Examples:
    # Set MEGATRON_PATH environment variable
    export MEGATRON_PATH=/path/to/MindSpeed-LLM

    # Single GPU evaluation (CUDA)
    torchrun --nproc-per-node=1 -m lm_eval --model mindspeed_lm \
        --model_args load=/path/to/ckpt,tokenizer_model=/path/to/tokenizer.model \
        --tasks arc_easy --batch_size 8

    # Data Parallelism (4 GPUs, each with full model replica)
    torchrun --nproc-per-node=4 -m lm_eval --model mindspeed_lm \
        --model_args load=/path/to/ckpt,devices=4,tokenizer_model=/path/to/tokenizer.model \
        --tasks arc_easy --batch_size 8

    # Tensor Parallelism (2 GPUs for TP)
    torchrun --nproc-per-node=2 -m lm_eval --model mindspeed_lm \
        --model_args load=/path/to/ckpt,devices=2,tensor_model_parallel_size=2,tokenizer_model=/path/to/tokenizer.model \
        --tasks arc_easy --batch_size 8

    # Expert Parallelism for MoE models (6 GPUs, EP=6)
    torchrun --nproc-per-node=6 -m lm_eval --model mindspeed_lm \
        --model_args load=/path/to/moe_ckpt,devices=6,expert_model_parallel_size=6,tokenizer_model=/path/to/tokenizer.model \
        --tasks arc_easy --batch_size 8

    # NPU (Ascend) evaluation - set device visibility and use torch_npu
    export ASCEND_RT_VISIBLE_DEVICES=0,1
    torchrun --nproc-per-node=2 -m lm_eval --model mindspeed_lm \
        --model_args load=/path/to/ckpt,devices=2,tensor_model_parallel_size=2,tokenizer_model=/path/to/tokenizer.model \
        --tasks arc_easy --batch_size 8
"""

import importlib
import logging
import os
import sys
from copy import deepcopy

import torch
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.device import (
    _is_torch_npu_available,
    get_current_device,
)
from lm_eval.models.utils import Collator
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window


eval_logger = logging.getLogger(__name__)


def _add_megatron_to_path():
    """Add Megatron-LM to Python path.

    The MEGATRON_PATH environment variable must be set to the Megatron-LM installation directory.
    """
    megatron_path = os.environ.get("MEGATRON_PATH")
    if megatron_path is None:
        raise OSError(
            "MEGATRON_PATH environment variable is not set. "
            "Please set it to the path of your Megatron-LM installation: "
            "export MEGATRON_PATH=/path/to/Megatron-LM"
        )
    if not os.path.isdir(megatron_path):
        raise FileNotFoundError(f"Megatron-LM directory not found at: {megatron_path}")
    if megatron_path not in sys.path:
        sys.path.insert(0, megatron_path)
    return megatron_path


class AscendNPUPatch:
    """NPU (Ascend) compatibility layer for Megatron-LM.

    Consolidates all NPU-specific monkey-patches and workarounds needed
    to run Megatron-LM on Ascend hardware without full MindSpeed-LLM stack.
    """

    @staticmethod
    def is_available() -> bool:
        return _is_torch_npu_available()

    @staticmethod
    def apply():
        """Apply all NPU patches before any Megatron imports."""
        if not AscendNPUPatch.is_available():
            return

        AscendNPUPatch._patch_cuda_apis()
        AscendNPUPatch._patch_dist_backend()
        AscendNPUPatch._init_default_generators()

    @staticmethod
    def _patch_cuda_apis():
        """Map torch.cuda.* to torch.npu.* for Megatron compatibility."""
        if torch.cuda.is_available():
            return

        eval_logger.info(
            "NPU detected (CUDA unavailable). "
            "Patching torch.cuda APIs to torch.npu equivalents."
        )
        # Core API replacements
        for attr in ("is_available", "device_count", "set_device", "current_device"):
            if hasattr(torch.npu, attr) and not hasattr(torch.cuda, attr):
                setattr(torch.cuda, attr, getattr(torch.npu, attr))
        # CUDA Stream/Graph APIs (some Megatron branches may reference)
        if hasattr(torch.npu, "Stream") and not hasattr(torch.cuda, "Stream"):
            torch.cuda.Stream = torch.npu.Stream
        if hasattr(torch.npu, "set_stream"):
            torch.cuda.set_stream = torch.npu.set_stream

    @staticmethod
    def _patch_dist_backend():
        """Redirect distributed backend nccl -> hccl for NPU."""
        _orig_init_pg = torch.distributed.init_process_group

        def _patched_init_process_group(*args, **kwargs):
            backend = kwargs.get("backend") or (args[0] if args else None)
            if backend == "nccl":
                if "backend" in kwargs:
                    kwargs["backend"] = "hccl"
                elif args:
                    args = ("hccl",) + args[1:]
            return _orig_init_pg(*args, **kwargs)

        if torch.distributed.init_process_group is not _patched_init_process_group:
            torch.distributed.init_process_group = _patched_init_process_group

    @staticmethod
    def _init_default_generators():
        """Initialize torch.cuda.default_generators for Megatron random.py."""
        n_devices = torch.npu.device_count()
        if not torch.cuda.default_generators and n_devices > 0:
            torch.cuda.default_generators = tuple(
                torch.Generator(device="npu") for _ in range(n_devices)
            )
            eval_logger.info(
                f"Initialized {n_devices} NPU default generators"
            )

    @staticmethod
    def patch_compile_dependencies():
        """Patch _compile_dependencies to no-op on NPU (no CUDA nvcc)."""
        try:
            import megatron.training.initialize as megatron_init
            if hasattr(megatron_init, "_compile_dependencies"):
                megatron_init._compile_dependencies = lambda: None
                eval_logger.info("NPU: _compile_dependencies patched to no-op")
        except ImportError:
            pass

    @staticmethod
    def load_mindspeed_adaptor():
        """Try loading MindSpeed-LLM megatron_adaptor. Returns True if loaded."""
        if not AscendNPUPatch.is_available():
            return False
        try:
            from mindspeed_llm import megatron_adaptor  # noqa: F401
            eval_logger.info("MindSpeed-LLM megatron_adaptor loaded")
            return True
        except ImportError:
            eval_logger.info("mindspeed_llm not available; using manual NPU patches")
            return False

    @staticmethod
    def get_tokenizer_fallback(tokenizer_type):
        """Fall back tokenizer type and args when mindspeed_llm missing.

        Returns (tokenizer_type, tokenizer_model_override).
        """
        if tokenizer_type == "PretrainedFromHF":
            eval_logger.info(
                "mindspeed_llm not available: falling back "
                "PretrainedFromHF -> HuggingFaceTokenizer"
            )
            return "HuggingFaceTokenizer", None
        return tokenizer_type, None


def _check_dist_ckpt(load_path: str) -> bool:
    """Check if the checkpoint is in distributed checkpoint format."""
    if not os.path.isdir(load_path):
        return False
    # Check for .distcp files
    for f in os.listdir(load_path):
        if f.endswith(".distcp"):
            return True
    # Check for metadata.json
    return os.path.exists(os.path.join(load_path, "metadata.json"))


def _parse_extra_args(extra_args: str | None) -> list[str]:
    """
    Parse extra_args string into a list of command line arguments.

    Uses space-separated arguments with shell-style quote handling.

    Examples:
        "--no-rope-fusion --trust-remote-code" -> ["--no-rope-fusion", "--trust-remote-code"]
        "--expert-tensor-parallel-size 1 --no-rope-fusion" -> ["--expert-tensor-parallel-size", "1", "--no-rope-fusion"]
    """
    import shlex

    if not extra_args:
        return []

    try:
        return shlex.split(extra_args)
    except ValueError as e:
        eval_logger.warning(
            f"Failed to parse extra_args with shlex: {e}, falling back to simple split"
        )
        return extra_args.split()


@register_model("mindspeed_lm")
class MindSpeedLMEval(LM):
    """
    MindSpeed-LLM model adapter for lm-evaluation-harness.

    Extends Megatron-LM evaluation with NPU (Ascend) support, automatic
    CUDA-to-NPU monkey-patching, and MindSpeed-LLM's initialization.

    See module docstring for parallelism modes and usage examples.

    Args:
        load: Megatron checkpoint path (parent directory containing iter_xxx subdirectories)
        ckpt_step: Checkpoint step to load (e.g., 40000 loads iter_0040000), defaults to latest
        tokenizer_type: Tokenizer type (e.g., HuggingFaceTokenizer, PretrainedFromHF)
        tokenizer_model: Tokenizer model file path (used with most tokenizer types)
        tokenizer_name_or_path: Tokenizer name or path for PretrainedFromHF tokenizer type
        vocab_file: Vocabulary file path (optional)
        merge_file: BPE merge file path (optional)
        devices: Total number of GPUs to use (default: 1)
        tensor_model_parallel_size: Tensor parallelism degree (default: 1)
        pipeline_model_parallel_size: Pipeline parallelism degree (default: 1)
        expert_model_parallel_size: Expert parallelism degree for MoE models (default: 1)
        seq_length: Maximum sequence length
        micro_batch_size: Micro batch size (optional, uses checkpoint value if not specified)
        max_gen_toks: Maximum number of tokens to generate
        use_checkpoint_args: Whether to load model args from checkpoint. Default True.
            Set to False if the checkpoint overrides your tokenizer type or if you want
            to use HuggingFaceTokenizer with a directory path.
        use_dist_ckpt: Whether to use distributed checkpoint format (auto-detected)
        extra_args: Extra MCore command line arguments, space-separated
        seed: Random seed for reproducibility (default: 42)
        spec: Custom layer spec module path, e.g. "mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec".
            Required for models with custom architectures (Qwen3, etc.).
        num_layers: Number of transformer layers (if not using --use-checkpoint-args)
        hidden_size: Hidden dimension size (if not using --use-checkpoint-args)
        num_attention_heads: Number of attention heads (if not using --use-checkpoint-args)
        ffn_hidden_size: FFN hidden dimension size (if not using --use-checkpoint-args)
        num_query_groups: Number of query groups for GQA (if not using --use-checkpoint-args)
        max_position_embeddings: Maximum position embeddings (if not using --use-checkpoint-args)
        padded_vocab_size: Padded vocabulary size (may be needed even with --use-checkpoint-args)
        make_vocab_size_divisible_by: Make vocab size divisible by this value
        rotary_base: Base for rotary position embeddings (default: 10000)
    """

    def __init__(
        self,
        load: str,
        ckpt_step: int | None = None,
        tokenizer_type: str = "HuggingFaceTokenizer",
        tokenizer_model: str | None = None,
        tokenizer_name_or_path: str | None = None,
        vocab_file: str | None = None,
        merge_file: str | None = None,
        devices: int = 1,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        expert_model_parallel_size: int = 1,
        seq_length: int = 4096,
        micro_batch_size: int = 1,
        max_gen_toks: int = 256,
        use_checkpoint_args: bool = True,
        use_dist_ckpt: bool | None = None,
        extra_args: str | None = None,
        seed: int = 42,
        # Custom layer spec module path (e.g. "mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec")
        spec: str | None = None,
        # Model parameters (if not using --use-checkpoint-args)
        num_layers: int | None = None,
        hidden_size: int | None = None,
        num_attention_heads: int | None = None,
        ffn_hidden_size: int | None = None,
        num_query_groups: int | None = None,
        max_position_embeddings: int | None = None,
        padded_vocab_size: int | None = None,
        make_vocab_size_divisible_by: int | None = None,
        rotary_base: float | None = None,
        **kwargs,
    ):
        super().__init__()

        self._max_length = seq_length
        self._batch_size = micro_batch_size if micro_batch_size is not None else 1
        self._max_gen_toks = max_gen_toks
        self._load_path = load
        self._ckpt_step = ckpt_step
        self._tp_size = tensor_model_parallel_size
        self._pp_size = pipeline_model_parallel_size
        self._ep_size = expert_model_parallel_size
        self._devices = devices

        # Validate parallelism configuration (NeMo-style)
        self._validate_parallelism_config(
            devices,
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            expert_model_parallel_size,
        )

        # Add Megatron to path
        _add_megatron_to_path()

        # Auto-detect distributed checkpoint
        if use_dist_ckpt is None:
            # Check iteration directories
            iter_dirs = [d for d in os.listdir(load) if d.startswith("iter_")]
            if iter_dirs:
                latest_iter = sorted(iter_dirs)[-1]
                iter_path = os.path.join(load, latest_iter)
                use_dist_ckpt = _check_dist_ckpt(iter_path)
            else:
                use_dist_ckpt = _check_dist_ckpt(load)

        self._use_dist_ckpt = use_dist_ckpt
        eval_logger.info(f"Using distributed checkpoint: {use_dist_ckpt}")

        # Initialize Megatron and load model
        self._initialize_megatron(
            load=load,
            ckpt_step=ckpt_step,
            tokenizer_type=tokenizer_type,
            tokenizer_model=tokenizer_model,
            tokenizer_name_or_path=tokenizer_name_or_path,
            vocab_file=vocab_file,
            merge_file=merge_file,
            devices=devices,
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            use_checkpoint_args=use_checkpoint_args,
            use_dist_ckpt=use_dist_ckpt,
            extra_args=extra_args,
            seed=seed,
            spec=spec,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            ffn_hidden_size=ffn_hidden_size,
            num_query_groups=num_query_groups,
            max_position_embeddings=max_position_embeddings,
            padded_vocab_size=padded_vocab_size,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by,
            rotary_base=rotary_base,
        )

        eval_logger.info(f"MindSpeed-LLM model loaded from {load}")
        eval_logger.info(f"Max sequence length: {self._max_length}")
        eval_logger.info(f"Batch size: {self._batch_size}")
        eval_logger.info(f"Parallelism mode: {self._parallelism_mode}")
        eval_logger.info(
            f"Devices: {self._devices}, TP: {self._tp_size}, PP: {self._pp_size}, EP: {self._ep_size}"
        )

    def _validate_parallelism_config(self, devices: int, tp: int, pp: int, ep: int):
        """
        Validate parallelism configuration.

        Supported modes:
        1. Single GPU: devices=1, TP=1, PP=1, EP=1
        2. Data Parallelism: tp=1, pp=1, devices>1
        3. Tensor Parallelism: tp == devices, pp=1
        4. Expert Parallelism (MoE): ep == devices, tp=1, pp=1

        For Expert Parallelism (EP > 1):
        - EP cannot be combined with TP or PP (must have TP=1, PP=1)
        - EP must equal devices (all ranks form one model-parallel group)
        - EP is model parallelism (world_size=1), not data parallelism

        Note: Pipeline Parallelism is supported for TP + PP combinations (devices = TP × PP).
        """
        # Validate EP configuration
        if ep > 1:
            # EP cannot be combined with TP or PP
            if tp > 1 or pp > 1:
                raise ValueError(
                    f"Expert Parallelism (EP={ep}) cannot be combined with "
                    f"Tensor Parallelism (TP={tp}) or Pipeline Parallelism (PP={pp}). "
                    f"Please use EP alone with TP=1, PP=1."
                )
            # EP must equal devices
            if devices != ep:
                raise ValueError(
                    f"Invalid Expert Parallelism configuration: devices={devices}, EP={ep}. "
                    f"When using Expert Parallelism (EP > 1), devices must equal expert-model-parallel-size."
                )

        # Determine parallelism mode
        total_model_parallel = tp * pp
        if ep > 1:
            self._parallelism_mode = "expert_parallel"
            eval_logger.info(f"Parallelism mode: Expert Parallel (EP={ep}, devices={devices})")
        elif tp == 1 and pp == 1:
            if devices == 1:
                self._parallelism_mode = "single"
                eval_logger.info("Parallelism mode: Single GPU")
            else:
                self._parallelism_mode = "data_parallel"
                eval_logger.info(
                    f"Parallelism mode: Data Parallel with {devices} replicas"
                )
        elif devices == total_model_parallel:
            # TP and/or PP: all ranks form one model-parallel group
            parts = []
            if tp > 1:
                parts.append(f"TP={tp}")
            if pp > 1:
                parts.append(f"PP={pp}")
            self._parallelism_mode = "tensor_parallel"
            eval_logger.info(
                f"Parallelism mode: {' + '.join(parts)} (devices={devices})"
            )
        else:
            raise ValueError(
                f"Invalid parallelism configuration: devices={devices}, TP={tp}, PP={pp}. "
                f"For model parallelism, TP × PP must equal devices ({total_model_parallel}). "
                f"For data parallelism, set TP=1, PP=1."
            )

    def _initialize_megatron(self, **kwargs):
        """Initialize Megatron distributed environment and load model."""
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

        # NPU environment setup
        if AscendNPUPatch.is_available():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")

        _tp_for_env = kwargs.get("tensor_model_parallel_size", 1)
        if _tp_for_env > 1:
            os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

        # Apply NPU patches + try loading MindSpeed-LLM adaptor
        AscendNPUPatch.apply()
        mindspeed_available = AscendNPUPatch.load_mindspeed_adaptor()

        # Tokenizer fallback when mindspeed_llm not available
        saved_tokenizer_type, _ = AscendNPUPatch.get_tokenizer_fallback(kwargs["tokenizer_type"])
        if saved_tokenizer_type != kwargs["tokenizer_type"]:
            kwargs["tokenizer_type"] = saved_tokenizer_type
            if not kwargs.get("tokenizer_model") and kwargs.get("tokenizer_name_or_path"):
                kwargs["tokenizer_model"] = kwargs["tokenizer_name_or_path"]

        # Import Megatron
        from megatron.training import get_args, get_model, get_tokenizer
        from megatron.training.arguments import core_transformer_config_from_args
        from megatron.training.checkpointing import load_checkpoint

        try:
            from mindspeed_llm.training.initialize import initialize_megatron
        except ImportError:
            from megatron.training import initialize_megatron

        # Patch compile dependencies on NPU
        if not mindspeed_available:
            AscendNPUPatch.patch_compile_dependencies()

        devices = kwargs["devices"]
        tp_size = kwargs["tensor_model_parallel_size"]
        pp_size = kwargs["pipeline_model_parallel_size"]
        ep_size = kwargs["expert_model_parallel_size"]
        actual_tp = tp_size
        actual_pp = pp_size

        # Build command line arguments (align with megatron_lm.py)
        argv = [
            sys.argv[0],
            "--load", kwargs["load"],
            "--tensor-model-parallel-size", str(actual_tp),
            "--pipeline-model-parallel-size", str(actual_pp),
            "--expert-model-parallel-size", str(ep_size),
            "--seq-length", str(kwargs["seq_length"]),
            "--tokenizer-type", kwargs["tokenizer_type"],
            "--use-mcore-models",
            "--no-load-optim", "--no-load-rng",
            "--bf16",
            "--no-masked-softmax-fusion",
            "--no-bias-gelu-fusion",
            "--no-bias-dropout-fusion",
            "--no-gradient-accumulation-fusion",
            "--attention-softmax-in-fp32",
            "--exit-on-missing-checkpoint",
            "--hidden-dropout", "0",
            "--attention-dropout", "0",
            "--seed", str(kwargs.get("seed", 42)),
        ]

        argv.extend(["--micro-batch-size", str(kwargs["micro_batch_size"])])

        if kwargs.get("ckpt_step") is not None:
            argv.extend(["--ckpt-step", str(kwargs["ckpt_step"])])

        if kwargs.get("use_dist_ckpt"):
            argv.append("--use-dist-ckpt")
            argv.append("--auto-detect-ckpt-format")

        # Spec: validate import before passing to Megatron
        spec_arg = kwargs.get("spec")
        if spec_arg:
            spec_parts = spec_arg.split()
            try:
                importlib.import_module(spec_parts[0])
                argv.extend(["--spec"] + spec_parts)
                eval_logger.info(f"Using custom layer spec: {spec_arg}")
            except ImportError:
                eval_logger.warning(
                    f"Cannot import spec '{spec_parts[0]}'; "
                    f"falling back to built-in layer spec"
                )

        if kwargs.get("tokenizer_model"):
            argv.extend(["--tokenizer-model", kwargs["tokenizer_model"]])
        if mindspeed_available and kwargs.get("tokenizer_name_or_path"):
            argv.extend(["--tokenizer-name-or-path", kwargs["tokenizer_name_or_path"]])
        if kwargs.get("vocab_file"):
            argv.extend(["--vocab-file", kwargs["vocab_file"]])
        if kwargs.get("merge_file"):
            argv.extend(["--merge-file", kwargs["merge_file"]])

        # Use checkpoint args (same as megatron_lm.py)
        argv.append("--use-checkpoint-args")

        # Monkey-patch load_args_from_checkpoint to preserve tokenizer settings
        # when mindspeed_llm is not available (checkpoint stores Llama2Tokenizer).
        if not mindspeed_available:
            from megatron.training import checkpointing as ckpt_module
            _orig_load_args = ckpt_module.load_args_from_checkpoint
            def _patched_load_args_from_checkpoint(args):
                _orig_load_args(args)
                args.tokenizer_type = kwargs["tokenizer_type"]
                if kwargs.get("tokenizer_model"):
                    args.tokenizer_model = kwargs["tokenizer_model"]
                eval_logger.info(
                    f"Restored tokenizer_type: {kwargs['tokenizer_type']}"
                )
            ckpt_module.load_args_from_checkpoint = _patched_load_args_from_checkpoint
            import megatron.training.initialize as megatron_init
            if hasattr(megatron_init, "load_args_from_checkpoint"):
                megatron_init.load_args_from_checkpoint = _patched_load_args_from_checkpoint
            eval_logger.info("Patched load_args_from_checkpoint for NPU tokenizer compatibility")

        # Add extra MCore arguments (appended last so they take precedence)
        extra_args_list = _parse_extra_args(kwargs.get("extra_args"))
        if extra_args_list:
            argv.extend(extra_args_list)
            eval_logger.info(f"Extra MCore args: {extra_args_list}")

        # Save original argv and replace
        original_argv = sys.argv
        sys.argv = argv
        eval_logger.info(f"Initializing Megatron with args: {' '.join(argv[1:])}")

        try:
            init_kwargs = {
                "extra_args_provider": None,
                "args_defaults": {"tokenizer_type": kwargs["tokenizer_type"]},
            }
            if AscendNPUPatch.is_available():
                init_kwargs["allow_no_cuda"] = True
            initialize_megatron(**init_kwargs)

            args = get_args()
            self._args = args

            from megatron.core import parallel_state
            self._parallel_state = parallel_state

            self._is_pipeline_last_stage = parallel_state.is_pipeline_last_stage()
            self._is_pipeline_first_stage = parallel_state.is_pipeline_first_stage()
            self._tp_rank = parallel_state.get_tensor_model_parallel_rank()
            self._pp_rank = parallel_state.get_pipeline_model_parallel_rank()
            self._dp_rank = parallel_state.get_data_parallel_rank()

            self._device = get_current_device()
            self._global_rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )

            if self._parallelism_mode == "data_parallel":
                self._rank = self._global_rank
                self._world_size = devices
            else:
                # Model Parallelism (TP/EP): all ranks work together as a single logical worker.
                # From lm_eval's perspective, this is a single worker (world_size=1)
                # because TP/EP handles computation distribution, not data distribution.
                # All ranks must process the same requests so they stay in sync
                # (TP all-reduce, EP all-to-all).
                self._rank = 0
                self._world_size = 1

            eval_logger.info(
                f"Parallel state - TP rank: {self._tp_rank}, PP rank: {self._pp_rank}, "
                f"DP rank: {self._dp_rank}, is_last_stage: {self._is_pipeline_last_stage}"
            )

            # Get tokenizer and fix padded_vocab_size if needed
            self.tokenizer = get_tokenizer()
            if not mindspeed_available and kwargs.get("tokenizer_name_or_path"):
                self._fix_padded_vocab_size(args, kwargs)

            # Create model_provider (aligned with megatron_lm.py)
            def model_provider(pre_process=True, post_process=True, config=None, pg_collection=None):
                """Build GPT model."""
                from megatron.core.models.gpt import GPTModel
                from megatron.core.models.gpt.gpt_layer_specs import (
                    get_gpt_decoder_block_spec,
                    get_gpt_layer_local_spec,
                    get_gpt_layer_with_transformer_engine_spec,
                )

                if config is None:
                    config = core_transformer_config_from_args(args)

                # When mindspeed_llm is unavailable, the checkpoint may not have
                # GQA args (num_query_groups/group_query_attention). Force them
                # from extra_args so the model matches the checkpoint.
                gqa_groups = getattr(args, "num_query_groups", None)
                if gqa_groups is not None:
                    config.num_query_groups = gqa_groups

                # Select layer spec (aligned with megatron_lm.py)
                transformer_impl = getattr(args, "transformer_impl", "local")
                use_transformer_engine = transformer_impl == "transformer_engine"
                if args.num_experts:
                    transformer_layer_spec = get_gpt_decoder_block_spec(
                        config, use_transformer_engine=use_transformer_engine,
                        normalization=getattr(args, "normalization", None),
                        qk_l2_norm=getattr(args, "qk_l2_norm", False),
                    )
                elif getattr(args, "heterogeneous_layers_config_path", None):
                    try:
                        from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
                            get_gpt_heterogeneous_layer_spec,
                        )
                        transformer_layer_spec = get_gpt_heterogeneous_layer_spec(
                            config, use_transformer_engine=use_transformer_engine
                        )
                    except ImportError:
                        eval_logger.warning("heterogeneous_layer_spec not available; falling back to local spec")
                        transformer_layer_spec = get_gpt_layer_local_spec(
                            getattr(args, "num_experts", None),
                            getattr(args, "moe_grouped_gemm", False),
                            getattr(args, "qk_layernorm", False),
                            getattr(args, "multi_latent_attention", False),
                        )
                elif use_transformer_engine:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        getattr(args, "num_experts", None),
                        getattr(args, "moe_grouped_gemm", False),
                        getattr(args, "qk_layernorm", False),
                        getattr(args, "multi_latent_attention", False),
                    )
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        getattr(args, "num_experts", None),
                        getattr(args, "moe_grouped_gemm", False),
                        getattr(args, "qk_layernorm", False),
                        getattr(args, "multi_latent_attention", False),
                    )

                # Override attn_mask_type to arbitrary for proper padding handling
                self._override_attn_mask(transformer_layer_spec)

                model = GPTModel(
                    config=config,
                    transformer_layer_spec=transformer_layer_spec,
                    vocab_size=args.padded_vocab_size,
                    max_sequence_length=args.seq_length,
                    pre_process=pre_process,
                    post_process=post_process,
                    fp16_lm_cross_entropy=getattr(args, "fp16_lm_cross_entropy", False),
                    parallel_output=False,
                    share_embeddings_and_output_weights=not getattr(
                        args, "untie_embeddings_and_output_weights", False
                    ),
                    position_embedding_type=getattr(args, "position_embedding_type", "learned_absolute"),
                    rotary_percent=getattr(args, "rotary_percent", 1.0),
                    rotary_base=getattr(args, "rotary_base", 10000),
                    seq_len_interpolation_factor=getattr(args, "rotary_seq_len_interpolation_factor", None),
                )
                return model

            # Get model and load checkpoint
            self._model = get_model(model_provider, wrap_with_ddp=False)
            load_checkpoint(self._model, None, None, strict=True)

            assert len(self._model) == 1, f"Expected 1 model, got {len(self._model)}"
            self.model = self._model[0]
            self.model.eval()
            eval_logger.info("Model loaded successfully!")

        finally:
            sys.argv = original_argv

    @staticmethod
    def _override_attn_mask(transformer_layer_spec):
        """Force SelfAttention attn_mask_type to arbitrary for left-padding."""
        from megatron.core.transformer.enums import AttnMaskType
        try:
            self_attention = getattr(
                getattr(transformer_layer_spec, "submodules", None),
                "self_attention", None,
            )
            params = getattr(self_attention, "params", None)
            if isinstance(params, dict):
                params["attn_mask_type"] = AttnMaskType.arbitrary
            layer_specs = getattr(transformer_layer_spec, "layer_specs", None)
            if layer_specs:
                for ls in layer_specs:
                    p = getattr(getattr(ls, "submodules", None), "self_attention", None)
                    if p and isinstance(getattr(p, "params", None), dict):
                        p.params["attn_mask_type"] = AttnMaskType.arbitrary
        except (AttributeError, TypeError):
            pass  # Best-effort override

    @staticmethod
    def _fix_padded_vocab_size(args, kwargs):
        """Override padded_vocab_size from HF config.json (checkpoint mismatch)."""
        if kwargs.get("tokenizer_name_or_path"):
            try:
                import json as _json
                cfg_path = os.path.join(kwargs["tokenizer_name_or_path"], "config.json")
                with open(cfg_path) as f:
                    hf_vocab_size = _json.load(f).get("vocab_size")
                if hf_vocab_size is not None:
                    eval_logger.info(
                        f"Overriding padded_vocab_size: {args.padded_vocab_size} -> {hf_vocab_size}"
                    )
                    args.padded_vocab_size = hf_vocab_size
            except (OSError, ValueError, KeyError):
                pass  # Best-effort HF config load

    @property
    def eot_token_id(self) -> int:
        """End of text token ID."""
        try:
            return self.tokenizer.eod
        except AttributeError:
            try:
                return self.tokenizer.eos_token_id
            except AttributeError:
                return self.tokenizer.eos_id

    @property
    def prefix_token_id(self) -> int:
        """Prefix token ID for loglikelihood (typically BOS or EOS)."""
        # Try to get BOS token first, fall back to EOT
        try:
            if (
                hasattr(self.tokenizer, "bos_token_id")
                and self.tokenizer.bos_token_id is not None
            ):
                return self.tokenizer.bos_token_id
        except AttributeError:
            pass
        return self.eot_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def accelerator(self):
        """Return accelerator interface for distributed operations (NeMo-style)."""
        return self._Accelerator(self._world_size, self._device)

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-gather a tensor across data-parallel ranks."""
        return self.accelerator.gather(tensor)

    def gather_object(self, obj, dst: int = 0):
        """Gather Python objects and return list on dst, None on others."""
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return [obj]

        gathered_objects = [None] * self.world_size
        torch.distributed.all_gather_object(gathered_objects, obj)
        return gathered_objects if self.rank == dst else None

    def barrier(self) -> None:
        """Synchronize processes."""
        self.accelerator.wait_for_everyone()

    class _Accelerator:
        """
        Internal accelerator class for distributed operations.

        Provides NeMo-style interface for synchronization and result gathering.
        """

        def __init__(self, world_size, device):
            self.world_size = world_size
            self.device = device

        def wait_for_everyone(self):
            """Synchronize all processes."""
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        def gather(self, local_tensor):
            """Gather tensors from all processes.

            Handles both 0-dimensional (scalar) and multi-dimensional tensors.
            """
            if not torch.distributed.is_initialized() or self.world_size == 1:
                return local_tensor

            # Handle 0-dimensional (scalar) tensors by reshaping to 1-d
            is_scalar = local_tensor.dim() == 0
            if is_scalar:
                local_tensor = local_tensor.unsqueeze(0)

            # Create list of tensors to gather into
            gathered_tensors = [
                torch.zeros_like(local_tensor) for _ in range(self.world_size)
            ]
            torch.distributed.all_gather(gathered_tensors, local_tensor)

            # Concatenate results
            result = torch.cat(gathered_tensors)

            return result

        def gather_object(self, local_obj):
            """Gather Python objects from all processes."""
            if not torch.distributed.is_initialized() or self.world_size == 1:
                return [local_obj]

            gathered_objects = [None] * self.world_size
            torch.distributed.all_gather_object(gathered_objects, local_obj)
            return gathered_objects

    def tok_encode(self, string: str, add_special_tokens: bool = False) -> list[int]:
        """Tokenize string."""
        try:
            return self.tokenizer.tokenize(string)
        except AttributeError:
            return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

    def tok_decode(self, tokens: list[int]) -> str:
        """Decode tokens to string."""
        try:
            return self.tokenizer.detokenize(tokens)
        except AttributeError:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _encode_pair(
        self, context: str, continuation: str
    ) -> tuple[list[int], list[int]]:
        """Encode context-continuation pair."""
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def _model_forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Model forward pass (aligned with megatron_lm.py)."""
        batch_size, seq_len = input_ids.shape

        causal_mask = torch.ones(
            (batch_size, 1, seq_len, seq_len), dtype=torch.bool, device=input_ids.device
        ).triu(diagonal=1)

        if attention_mask is not None and attention_mask.dim() == 2:
            if position_ids is None:
                position_ids = torch.arange(
                    seq_len, dtype=torch.long, device=input_ids.device
                ).unsqueeze(0).expand(batch_size, -1)

            padding_mask = (1 - attention_mask).unsqueeze(1).unsqueeze(2).bool()
            attention_mask = causal_mask | padding_mask
        else:
            if position_ids is None:
                position_ids = torch.arange(
                    seq_len, dtype=torch.long, device=input_ids.device
                ).unsqueeze(0).expand(batch_size, -1)
            attention_mask = causal_mask

        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

        return output

    def _distribute_requests(self, requests: list) -> tuple[list, list[int]]:
        """
        Distribute requests across ranks for Data Parallelism.

        NOTE: When world_size > 1, lm_eval's evaluator already distributes
        requests to each rank based on lm.rank and lm.world_size.
        So we should NOT do additional distribution here - just return
        the requests as-is.

        Returns:
            local_requests: Requests for this rank (unchanged when world_size > 1)
            all_sizes: Number of requests per rank
        """
        # lm_eval already handles data distribution when world_size > 1
        # We just pass through the requests without additional splitting
        return requests, [len(requests)]

    def _gather_results(self, local_results: list, sizes: list[int]) -> list:
        """
        Gather results from all ranks for Data Parallelism.

        NOTE: When world_size > 1, lm_eval's evaluator already handles
        result gathering (via gather_object in evaluator.py line ~692).
        So we should NOT do additional gathering here - just return
        the results as-is.

        Args:
            local_results: Results from this rank
            sizes: Number of results per rank

        Returns:
            all_results: Results from this rank (unchanged when world_size > 1)
        """
        # lm_eval already handles result gathering when world_size > 1
        # We just return results without additional gathering
        return local_results

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Compute log-likelihood with Data Parallelism support.

        Handles BOS token correctly: some tokenizers automatically prepend BOS,
        so we check and move it to context if present in continuation.
        """
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                # Encode continuation without special tokens to avoid duplicate BOS
                continuation_enc = self.tok_encode(
                    continuation, add_special_tokens=False
                )
                # Handle BOS token: if continuation starts with prefix_token_id,
                # use it as context; otherwise use prefix_token_id as context
                if (
                    len(continuation_enc) > 0
                    and continuation_enc[0] == self.prefix_token_id
                ):
                    # Continuation already has BOS, move it to context
                    context_enc = continuation_enc[:1]
                    continuation_enc = continuation_enc[1:]
                else:
                    # Use prefix_token_id (BOS/EOS) as context
                    context_enc = [self.prefix_token_id]
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)
            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def _loglikelihood_tokens(
        self,
        requests: list[tuple],
        disable_tqdm: bool = False,
    ) -> list[tuple[float, bool]]:
        """
        Compute log-likelihood based on tokens.

        With Data Parallelism:
        - Requests are distributed across ranks by lm_eval's evaluator
        - Each rank processes its share
        - Results are gathered by lm_eval's evaluator

        With Model Parallelism (TP/PP):
        - All TP ranks compute the same result (TP is transparent)
        - Only PP last stage has logits, which are broadcast to all stages
        - Results are computed on all ranks consistently

        With Expert Parallelism (EP > 1):
        - MoE layers use all-to-all which provides implicit synchronization
        - All EP ranks process identical requests (world_size=1, model parallelism)
        """
        # Distribute requests for Data Parallelism
        local_requests, sizes = self._distribute_requests(requests)

        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(local_requests, sort_fn=_collate)
        chunks = re_ord.get_batched(n=self.batch_size, batch_fn=None)

        # Only show progress bar on global rank 0
        pbar = tqdm(
            total=len(local_requests),
            disable=disable_tqdm or (self._global_rank != 0),
            desc="Running loglikelihood requests",
        )

        for chunk in chunks:
            inps = []
            ctxlens = []
            contlens = []

            for _, context_enc, continuation_enc in chunk:
                # Truncate to max length
                inp = (context_enc + continuation_enc)[-(self.max_length) :]
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - self.max_length
                )
                ctxlens.append(ctxlen)
                contlens.append(len(continuation_enc))
                inps.append(inp)

            # Pad sequences
            max_len = max(len(inp) for inp in inps)
            padded_inps = []
            attention_mask_list = []
            for inp in inps:
                pad_len = max_len - len(inp)
                padded = [self.eot_token_id] * pad_len + inp
                padded_inps.append(padded)
                # Attention mask: 0 for padding, 1 for real tokens
                attention_mask_list.append([0] * pad_len + [1] * len(inp))

            input_ids = torch.tensor(padded_inps, dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(
                attention_mask_list, dtype=torch.long, device=self.device
            )

            # Forward pass: with PP > 1, ALL stages must participate in
            # the forward pass for P2P send/recv, but only the last PP
            # stage produces valid logits.
            logits = self._model_forward(input_ids, attention_mask=attention_mask)

            # Compute results only on the last PP stage
            if not self._is_pipeline_last_stage:
                for _, ctx, cont in chunk:
                    res.append((0.0, False))
                pbar.update(len(chunk))
                continue

            log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)

            for i, (ctxlen, contlen) in enumerate(zip(ctxlens, contlens, strict=True)):
                # Get padding length
                pad_len = max_len - len(inps[i])

                # Compute log probability of continuation
                cont_log_probs = []
                greedy_tokens = []

                start_idx = pad_len + ctxlen - 1
                end_idx = pad_len + ctxlen + contlen - 1

                for j in range(start_idx, end_idx):
                    next_token = input_ids[i, j + 1].item()
                    cont_log_probs.append(log_probs[i, j, next_token].item())
                    greedy_tokens.append(torch.argmax(log_probs[i, j]).item())

                logprob = sum(cont_log_probs)

                # Check if greedy
                actual_tokens = input_ids[i, start_idx + 1 : end_idx + 1].cpu().tolist()
                is_greedy = greedy_tokens == actual_tokens

                answer = (logprob, is_greedy)
                res.append(answer)

                cache_key = chunk[i][0]
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                pbar.update(1)

        pbar.close()

        # Reorder local results
        local_results = re_ord.get_original(res)

        # Gather results from all ranks (for Data Parallelism)
        all_results = self._gather_results(local_results, sizes)

        return all_results

    def loglikelihood_rolling(
        self,
        requests: list[Instance],
        disable_tqdm: bool = False,
    ) -> list[float]:
        """Compute rolling log-likelihood (for perplexity) with Data Parallelism support."""
        # Distribute requests for Data Parallelism
        local_requests, sizes = self._distribute_requests(
            [req.args for req in requests]
        )

        loglikelihoods = []

        for (string,) in tqdm(
            local_requests,
            disable=disable_tqdm or (self._global_rank != 0),
            desc="Running loglikelihood_rolling requests",
        ):
            rolling_token_windows = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length - 1,
                        context_len=1,
                    ),
                )
            )

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]
            string_nll = self._loglikelihood_tokens(
                rolling_token_windows, disable_tqdm=True
            )
            string_nll = [x[0] for x in string_nll]
            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

            self.cache_hook.add_partial("loglikelihood_rolling", (string,), string_nll)

        # Gather results from all ranks
        all_results = self._gather_results(loglikelihoods, sizes)

        return all_results

    def generate_until(
        self,
        requests: list[Instance],
        disable_tqdm: bool = False,
    ) -> list[str]:
        """
        Generate text until stop condition with Data Parallelism support.

        Supports batched generation for improved throughput.
        With PP > 1, generation requires coordination between pipeline stages.

        Uses Collator to sort requests by length, minimizing padding within batches.
        This is critical for RoPE-based models where position offsets affect results.
        """
        # Distribute requests for Data Parallelism
        local_requests, sizes = self._distribute_requests(requests)

        if not local_requests:
            return self._gather_results([], sizes)

        results = []
        batch_size = self.batch_size if self.batch_size != "auto" else 1

        # Use Collator to sort by TOKEN length (negative for descending order)
        # This minimizes padding within each batch, which is critical for RoPE models
        # NOTE: We need to tokenize first to get the actual token count, not string length!
        def _collate_gen(req):
            # Sort by negative token length (longest first)
            ctx = req.args[0]
            # Tokenize to get actual token count for proper sorting
            tokens = self.tok_encode(ctx)
            return -len(tokens), ctx

        # group_by="gen_kwargs" ensures each batch has the same generation parameters
        # This is important when running multiple tasks with different gen_kwargs
        re_ord = Collator(
            local_requests,
            sort_fn=_collate_gen,
            group_by="gen_kwargs",
            group_fn=lambda x: x.args[1],
        )
        chunks = re_ord.get_batched(n=batch_size, batch_fn=None)

        pbar = tqdm(
            total=len(local_requests),
            disable=disable_tqdm or (self._global_rank != 0),
            desc="Running generate_until requests",
        )

        for chunk in chunks:
            batch_requests = chunk
            actual_batch_size = len(batch_requests)

            # Extract generation parameters from batch
            contexts = []
            until_list = []
            max_gen_toks_list = []
            temperature_list = []
            top_p_list = []
            top_k_list = []

            for request in batch_requests:
                context, gen_kwargs = request.args
                gen_kwargs = deepcopy(gen_kwargs)

                contexts.append(context)

                until = gen_kwargs.pop("until", [])
                if isinstance(until, str):
                    until = [until]
                until_list.append(until)

                max_gen_toks_list.append(
                    gen_kwargs.pop("max_gen_toks", self.max_gen_toks)
                )
                temperature_list.append(gen_kwargs.pop("temperature", 0.0))
                top_p_list.append(gen_kwargs.pop("top_p", 1.0))
                top_k_list.append(gen_kwargs.pop("top_k", 0))

            # Use the max of all max_gen_toks in batch
            max_gen_toks = max(max_gen_toks_list)
            # For simplicity, use first sample's parameters for the whole batch
            # (in practice, lm_eval usually uses same params for all requests)
            temperature = temperature_list[0]
            top_p = top_p_list[0]
            top_k = top_k_list[0]

            # Tokenize all contexts
            context_tokens_list = []
            for ctx in contexts:
                tokens = self.tok_encode(ctx)
                tokens = tokens[-(self.max_length - max_gen_toks) :]
                context_tokens_list.append(tokens)

            # Left-pad to same length
            max_ctx_len = max(len(t) for t in context_tokens_list)

            padded_input_ids = []
            attention_mask_list = []
            for tokens in context_tokens_list:
                pad_len = max_ctx_len - len(tokens)
                padded_tokens = [self.eot_token_id] * pad_len + tokens
                padded_input_ids.append(padded_tokens)
                # Attention mask: 0 for padding, 1 for real tokens
                mask = [0] * pad_len + [1] * len(tokens)
                attention_mask_list.append(mask)

            input_ids = torch.tensor(
                padded_input_ids, dtype=torch.long, device=self.device
            )
            attention_mask = torch.tensor(
                attention_mask_list, dtype=torch.long, device=self.device
            )

            # Track generation state for each sample in batch
            generated_tokens = [[] for _ in range(actual_batch_size)]
            finished = [False] * actual_batch_size

            # Autoregressive generation loop
            # For EP mode: ALL ranks must execute same number of forward passes
            for _step in range(max_gen_toks):
                # EP synchronization FIRST: check if ALL ranks have ALL samples finished
                # This MUST be before any early exit to prevent hang
                if torch.distributed.is_initialized() and self._ep_size > 1:
                    all_finished_local = all(finished)
                    finished_tensor = torch.tensor(
                        [1 if all_finished_local else 0],
                        dtype=torch.int32,
                        device=self.device,
                    )
                    torch.distributed.all_reduce(
                        finished_tensor, op=torch.distributed.ReduceOp.MIN
                    )
                    if finished_tensor.item() == 1:
                        # All ranks agree to exit
                        break
                else:
                    # Non-EP mode: can exit immediately when local batch is done
                    if all(finished):
                        break

                # Truncate if too long
                if input_ids.shape[1] > self.max_length:
                    input_ids = input_ids[:, -self.max_length :]
                    attention_mask = attention_mask[:, -self.max_length :]

                # Forward pass - ALL ranks must participate for EP All-to-All sync
                logits = self._model_forward(input_ids, attention_mask=attention_mask)

                # Only process results if this rank's batch is not finished
                if not all(finished):
                    # Get next token logits for the last position
                    next_token_logits = logits[
                        :, -1, :
                    ].float()  # [batch_size, vocab_size]

                    # Apply sampling strategies (batched)
                    if temperature > 0:
                        next_token_logits = next_token_logits / temperature

                        # Top-K filtering (batched)
                        if top_k > 0:
                            top_k_vals, _ = torch.topk(
                                next_token_logits,
                                min(top_k, next_token_logits.size(-1)),
                            )
                            threshold = top_k_vals[:, -1].unsqueeze(-1)
                            next_token_logits = torch.where(
                                next_token_logits < threshold,
                                torch.full_like(next_token_logits, float("-inf")),
                                next_token_logits,
                            )

                        # Top-P (nucleus) filtering (batched)
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(
                                next_token_logits, descending=True
                            )
                            cumulative_probs = torch.cumsum(
                                torch.nn.functional.softmax(sorted_logits, dim=-1),
                                dim=-1,
                            )
                            sorted_indices_to_remove = cumulative_probs > top_p
                            # Shift right to keep at least one token
                            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[
                                :, :-1
                            ].clone()
                            sorted_indices_to_remove[:, 0] = False
                            # Scatter back to original indices
                            indices_to_remove = sorted_indices_to_remove.scatter(
                                1, sorted_indices, sorted_indices_to_remove
                            )
                            next_token_logits = next_token_logits.masked_fill(
                                indices_to_remove, float("-inf")
                            )

                        # Sample from distribution
                        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                        next_tokens = torch.multinomial(
                            probs, num_samples=1
                        )  # [batch_size, 1]
                    else:
                        # Greedy decoding
                        next_tokens = torch.argmax(
                            next_token_logits, dim=-1, keepdim=True
                        )  # [batch_size, 1]

                    # For Tensor Parallelism, ensure all ranks produce consistent next_tokens.
                    # While TP's internal all-reduce makes logits identical across ranks,
                    # sampling (temperature > 0) uses torch.multinomial which is nondeterministic.
                    # Broadcast from rank 0 to guarantee all TP ranks generate the same tokens.
                    if (
                        self._parallelism_mode == "tensor_parallel"
                        and torch.distributed.is_initialized()
                        and temperature > 0
                    ):
                        torch.distributed.broadcast(next_tokens, src=0)

                    # Process each sample in the batch
                    for i in range(actual_batch_size):
                        if finished[i]:
                            continue

                        next_token_id = next_tokens[i].item()
                        generated_tokens[i].append(next_token_id)

                        # Check EOS
                        if next_token_id == self.eot_token_id:
                            finished[i] = True
                            continue

                        # Check stop sequences
                        generated_text = self.tok_decode(generated_tokens[i])
                        for stop_seq in until_list[i]:
                            if stop_seq in generated_text:
                                finished[i] = True
                                break

                    # Update input_ids and attention_mask for next step
                    input_ids = torch.cat([input_ids, next_tokens], dim=1)
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(
                                (actual_batch_size, 1),
                                dtype=torch.long,
                                device=self.device,
                            ),
                        ],
                        dim=1,
                    )

            # Post-process: decode and truncate at stop sequences
            for i in range(actual_batch_size):
                continuation = self.tok_decode(generated_tokens[i])

                # Truncate at stop sequences
                for stop_seq in until_list[i]:
                    if stop_seq in continuation:
                        continuation = continuation.split(stop_seq)[0]
                        break

                results.append(continuation)
                self.cache_hook.add_partial(
                    "generate_until", batch_requests[i].args, continuation
                )

            pbar.update(actual_batch_size)

        pbar.close()

        # Reorder results to match original request order
        results = re_ord.get_original(results)

        # Gather results from all ranks
        all_results = self._gather_results(results, sizes)

        return all_results
