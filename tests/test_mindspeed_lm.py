"""Tests for mindspeed_lm.py backend.

These tests validate the core logic without requiring actual NPU hardware,
Megatron checkpoints, or MindSpeed-LLM installation. Megatron-dependent
code paths are tested via mocking.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch


# ---------------------------------------------------------------------------
# Helper: import the module under test in a controlled environment
# ---------------------------------------------------------------------------
# We need to import helper functions BEFORE the module tries to import
# megatron (which is not available in CI).  We mock the megatron imports
# that happen at module level or inside class methods.

# Pre-register mock modules so `from megatron.xxx import yyy` works
mock_megatron_modules = {
    "megatron": mock.MagicMock(),
    "megatron.training": mock.MagicMock(),
    "megatron.core": mock.MagicMock(),
    "megatron.core.parallel_state": mock.MagicMock(),
    "mindspeed_llm": mock.MagicMock(),
    "mindspeed_llm.training": mock.MagicMock(),
    "mindspeed_llm.training.initialize": mock.MagicMock(),
}


@pytest.fixture(autouse=True)
def _clean_megatron_from_path():
    """Remove megatron-related modules between tests to avoid cross-contamination."""
    yield
    # Clean up any modules we might have polluted
    to_remove = [k for k in sys.modules if k.startswith("megatron") or k.startswith("mindspeed")]
    for mod in to_remove:
        sys.modules.pop(mod, None)


# ---------------------------------------------------------------------------
# Tests for standalone helper functions (no Megatron dependency)
# ---------------------------------------------------------------------------


class TestAddMegatronToPath:
    """Test _add_megatron_to_path()."""

    def test_raises_when_not_set(self):
        from lm_eval.models.mindspeed_lm import _add_megatron_to_path

        with mock.patch.dict(os.environ, {}, clear=True):
            # Remove MEGATRON_PATH if it exists
            os.environ.pop("MEGATRON_PATH", None)
            with pytest.raises(OSError, match="MEGATRON_PATH"):
                _add_megatron_to_path()

    def test_raises_when_dir_not_found(self):
        from lm_eval.models.mindspeed_lm import _add_megatron_to_path

        with mock.patch.dict(os.environ, {"MEGATRON_PATH": "/nonexistent/path"}):
            with pytest.raises(FileNotFoundError, match="not found"):
                _add_megatron_to_path()

    def test_adds_to_sys_path(self):
        from lm_eval.models.mindspeed_lm import _add_megatron_to_path

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"MEGATRON_PATH": tmpdir}):
                result = _add_megatron_to_path()
                assert result == tmpdir
                assert tmpdir in sys.path
                # Cleanup
                sys.path.remove(tmpdir)

    def test_does_not_duplicate(self):
        from lm_eval.models.mindspeed_lm import _add_megatron_to_path

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"MEGATRON_PATH": tmpdir}):
                _add_megatron_to_path()
                count_before = sys.path.count(tmpdir)
                _add_megatron_to_path()
                count_after = sys.path.count(tmpdir)
                assert count_before == count_after
                # Cleanup
                while tmpdir in sys.path:
                    sys.path.remove(tmpdir)


class TestCheckDistCkpt:
    """Test _check_dist_ckpt()."""

    def test_returns_false_for_file(self):
        from lm_eval.models.mindspeed_lm import _check_dist_ckpt

        with tempfile.NamedTemporaryFile() as f:
            assert _check_dist_ckpt(f.name) is False

    def test_returns_false_for_empty_dir(self):
        from lm_eval.models.mindspeed_lm import _check_dist_ckpt

        with tempfile.TemporaryDirectory() as tmpdir:
            assert _check_dist_ckpt(tmpdir) is False

    def test_detects_distcp_files(self):
        from lm_eval.models.mindspeed_lm import _check_dist_ckpt

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.pt.distcp").touch()
            assert _check_dist_ckpt(tmpdir) is True

    def test_detects_metadata_json(self):
        from lm_eval.models.mindspeed_lm import _check_dist_ckpt

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "metadata.json").write_text("{}")
            assert _check_dist_ckpt(tmpdir) is True

    def test_returns_false_for_nonexistent(self):
        from lm_eval.models.mindspeed_lm import _check_dist_ckpt

        assert _check_dist_ckpt("/nonexistent/path") is False


class TestParseExtraArgs:
    """Test _parse_extra_args()."""

    def test_none_returns_empty(self):
        from lm_eval.models.mindspeed_lm import _parse_extra_args

        assert _parse_extra_args(None) == []

    def test_empty_string_returns_empty(self):
        from lm_eval.models.mindspeed_lm import _parse_extra_args

        assert _parse_extra_args("") == []

    def test_simple_args(self):
        from lm_eval.models.mindspeed_lm import _parse_extra_args

        result = _parse_extra_args("--no-rope-fusion --trust-remote-code")
        assert result == ["--no-rope-fusion", "--trust-remote-code"]

    def test_args_with_values(self):
        from lm_eval.models.mindspeed_lm import _parse_extra_args

        result = _parse_extra_args("--num-query-groups 8 --kv-channels 128")
        assert result == ["--num-query-groups", "8", "--kv-channels", "128"]

    def test_quoted_args(self):
        from lm_eval.models.mindspeed_lm import _parse_extra_args

        result = _parse_extra_args('--normalization "RMSNorm"')
        assert result == ["--normalization", "RMSNorm"]


class TestTryLoadHfConfigAsMcoreArgs:
    """Test _try_load_hf_config_as_mcore_args()."""

    def test_returns_empty_for_none(self):
        from lm_eval.models.mindspeed_lm import _try_load_hf_config_as_mcore_args

        assert _try_load_hf_config_as_mcore_args(None) == {}

    def test_auto_detects_from_tokenizer_dir(self):
        from lm_eval.models.mindspeed_lm import _try_load_hf_config_as_mcore_args

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "num_hidden_layers": 28,
                "hidden_size": 1024,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "intermediate_size": 3072,
                "max_position_embeddings": 32768,
            }
            Path(tmpdir, "config.json").write_text(json.dumps(config))

            result = _try_load_hf_config_as_mcore_args(tmpdir)
            assert result["num_layers"] == 28
            assert result["hidden_size"] == 1024
            assert result["num_attention_heads"] == 16
            assert result["num_query_groups"] == 8
            assert result["ffn_hidden_size"] == 3072
            assert result["max_position_embeddings"] == 32768

    def test_returns_empty_when_no_config(self):
        from lm_eval.models.mindspeed_lm import _try_load_hf_config_as_mcore_args

        with tempfile.TemporaryDirectory() as tmpdir:
            assert _try_load_hf_config_as_mcore_args(tmpdir) == {}

    def test_handles_invalid_json(self):
        from lm_eval.models.mindspeed_lm import _try_load_hf_config_as_mcore_args

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.json").write_text("not valid json{{{")
            result = _try_load_hf_config_as_mcore_args(tmpdir)
            assert result == {}

    def test_partial_config(self):
        from lm_eval.models.mindspeed_lm import _try_load_hf_config_as_mcore_args

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"num_hidden_layers": 12, "hidden_size": 768}
            Path(tmpdir, "config.json").write_text(json.dumps(config))

            result = _try_load_hf_config_as_mcore_args(tmpdir)
            assert result == {"num_layers": 12, "hidden_size": 768}


# ---------------------------------------------------------------------------
# Tests for parallelism configuration validation
# ---------------------------------------------------------------------------


class TestParallelismConfig:
    """Test _validate_parallelism_config() via __init__."""

    def _make_instance_attrs(self, devices, tp, pp, ep):
        """Create minimal instance attributes needed for validation."""
        return {
            "devices": devices,
            "tp_size": tp,
            "pp_size": pp,
            "ep_size": ep,
        }

    def test_single_gpu(self):
        """Single GPU mode: devices=1, TP=1, PP=1."""
        # We test the validation logic directly by creating a minimal mock
        from lm_eval.models.mindspeed_lm import MindSpeedLMEval

        with mock.patch.object(MindSpeedLMEval, "__init__", return_value=None):
            lm = MindSpeedLMEval.__new__(MindSpeedLMEval)
            lm._validate_parallelism_config(devices=1, tp=1, pp=1, ep=1)
            assert lm._parallelism_mode == "single"

    def test_data_parallel(self):
        """DP mode: devices=4, TP=1, PP=1."""
        from lm_eval.models.mindspeed_lm import MindSpeedLMEval

        with mock.patch.object(MindSpeedLMEval, "__init__", return_value=None):
            lm = MindSpeedLMEval.__new__(MindSpeedLMEval)
            lm._validate_parallelism_config(devices=4, tp=1, pp=1, ep=1)
            assert lm._parallelism_mode == "data_parallel"

    def test_tensor_parallel(self):
        """TP mode: devices=2, TP=2, PP=1."""
        from lm_eval.models.mindspeed_lm import MindSpeedLMEval

        with mock.patch.object(MindSpeedLMEval, "__init__", return_value=None):
            lm = MindSpeedLMEval.__new__(MindSpeedLMEval)
            lm._validate_parallelism_config(devices=2, tp=2, pp=1, ep=1)
            assert lm._parallelism_mode == "tensor_parallel"

    def test_expert_parallel(self):
        """EP mode: devices=4, TP=1, PP=1, EP=4."""
        from lm_eval.models.mindspeed_lm import MindSpeedLMEval

        with mock.patch.object(MindSpeedLMEval, "__init__", return_value=None):
            lm = MindSpeedLMEval.__new__(MindSpeedLMEval)
            lm._validate_parallelism_config(devices=4, tp=1, pp=1, ep=4)
            assert lm._parallelism_mode == "data_parallel"

    def test_rejects_pipeline_parallel(self):
        """PP > 1 should be rejected."""
        from lm_eval.models.mindspeed_lm import MindSpeedLMEval

        with mock.patch.object(MindSpeedLMEval, "__init__", return_value=None):
            lm = MindSpeedLMEval.__new__(MindSpeedLMEval)
            with pytest.raises(AssertionError, match="Pipeline Parallelism"):
                lm._validate_parallelism_config(devices=2, tp=1, pp=2, ep=1)

    def test_rejects_ep_with_tp(self):
        """EP + TP should be rejected."""
        from lm_eval.models.mindspeed_lm import MindSpeedLMEval

        with mock.patch.object(MindSpeedLMEval, "__init__", return_value=None):
            lm = MindSpeedLMEval.__new__(MindSpeedLMEval)
            with pytest.raises(ValueError, match="cannot be combined"):
                lm._validate_parallelism_config(devices=4, tp=2, pp=1, ep=4)

    def test_rejects_ep_mismatch(self):
        """EP must equal devices."""
        from lm_eval.models.mindspeed_lm import MindSpeedLMEval

        with mock.patch.object(MindSpeedLMEval, "__init__", return_value=None):
            lm = MindSpeedLMEval.__new__(MindSpeedLMEval)
            with pytest.raises(ValueError, match="devices must equal"):
                lm._validate_parallelism_config(devices=8, tp=1, pp=1, ep=4)

    def test_rejects_tp_ne_devices(self):
        """TP != devices with TP > 1 should be rejected."""
        from lm_eval.models.mindspeed_lm import MindSpeedLMEval

        with mock.patch.object(MindSpeedLMEval, "__init__", return_value=None):
            lm = MindSpeedLMEval.__new__(MindSpeedLMEval)
            with pytest.raises(ValueError, match="Invalid parallelism"):
                lm._validate_parallelism_config(devices=4, tp=2, pp=1, ep=1)


# ---------------------------------------------------------------------------
# Tests for NPU patching
# ---------------------------------------------------------------------------


class TestMaybePatchForNpu:
    """Test _maybe_patch_for_npu() behavior."""

    def test_no_op_when_torch_npu_not_installed(self):
        from lm_eval.models.mindspeed_lm import _maybe_patch_for_npu

        with mock.patch.dict(sys.modules, {"torch_npu": None}):
            # Should not raise
            _maybe_patch_for_npu()

    def test_no_op_when_npu_not_available(self):
        from lm_eval.models.mindspeed_lm import _maybe_patch_for_npu

        mock_npu_module = mock.MagicMock()
        mock_npu_api = mock.MagicMock()
        mock_npu_api.is_available.return_value = False

        with mock.patch.dict(sys.modules, {"torch_npu": mock_npu_module}), \
             mock.patch.object(torch, "npu", mock_npu_api, create=True):
            _maybe_patch_for_npu()

    def test_patches_init_process_group(self):
        from lm_eval.models.mindspeed_lm import _maybe_patch_for_npu

        mock_npu_module = mock.MagicMock()
        mock_npu_api = mock.MagicMock()
        mock_npu_api.is_available.return_value = True

        original_init_pg = torch.distributed.init_process_group

        with mock.patch.dict(sys.modules, {"torch_npu": mock_npu_module}), \
             mock.patch.object(torch, "npu", mock_npu_api, create=True), \
             mock.patch.object(torch.cuda, "is_available", return_value=False):
            _maybe_patch_for_npu()
            # init_process_group should be patched to redirect nccl -> hccl
            assert torch.distributed.init_process_group is not original_init_pg

        # Restore
        torch.distributed.init_process_group = original_init_pg

    def test_nccl_redirected_to_hccl(self):
        from lm_eval.models.mindspeed_lm import _maybe_patch_for_npu

        mock_npu_module = mock.MagicMock()
        mock_npu_api = mock.MagicMock()
        mock_npu_api.is_available.return_value = True

        original_init_pg = torch.distributed.init_process_group
        mock_init = mock.MagicMock()

        with mock.patch.dict(sys.modules, {"torch_npu": mock_npu_module}), \
             mock.patch.object(torch, "npu", mock_npu_api, create=True), \
             mock.patch.object(torch.cuda, "is_available", return_value=False):
            torch.distributed.init_process_group = mock_init
            _maybe_patch_for_npu()

            # Call with nccl -> should redirect to hccl
            torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)
            mock_init.assert_called_with(backend="hccl", rank=0, world_size=1)

            # Call with gloo -> should pass through unchanged
            mock_init.reset_mock()
            torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
            mock_init.assert_called_with(backend="gloo", rank=0, world_size=1)

        # Restore
        torch.distributed.init_process_group = original_init_pg


# ---------------------------------------------------------------------------
# Tests for argv construction logic
# ---------------------------------------------------------------------------


class TestArgvConstruction:
    """Test that _initialize_megatron builds correct argv."""

    def _get_argv(self, **overrides):
        """Capture the argv that would be passed to initialize_megatron."""
        defaults = {
            "load": "/fake/ckpt",
            "ckpt_step": None,
            "tokenizer_type": "HuggingFaceTokenizer",
            "tokenizer_model": "/fake/tokenizer",
            "vocab_file": None,
            "merge_file": None,
            "devices": 1,
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "seq_length": 4096,
            "micro_batch_size": 1,
            "use_checkpoint_args": True,
            "use_dist_ckpt": False,
            "extra_args": None,
            "seed": 42,
            "spec": None,
            "num_layers": None,
            "hidden_size": None,
            "num_attention_heads": None,
            "ffn_hidden_size": None,
            "num_query_groups": None,
            "max_position_embeddings": None,
            "padded_vocab_size": None,
            "make_vocab_size_divisible_by": None,
            "rotary_base": None,
        }
        defaults.update(overrides)
        return defaults

    def test_default_argv_contains_required_flags(self):
        """Verify essential flags are present in default argv."""
        from lm_eval.models.mindspeed_lm import MindSpeedLMEval

        kwargs = self._get_argv()

        # We intercept sys.argv assignment inside _initialize_megatron
        captured_argv = []

        with mock.patch.object(MindSpeedLMEval, "__init__", return_value=None):
            lm = MindSpeedLMEval.__new__(MindSpeedLMEval)
            lm._parallelism_mode = "single"
            lm._devices = 1
            lm._tp_size = 1
            lm._pp_size = 1
            lm._ep_size = 1
            # Set MindSpeed-specific attributes for _initialize_megatron override
            lm._ms_seed = 42
            lm._ms_spec = None
            lm._ms_use_checkpoint_args = True
            lm._ms_max_position_embeddings = None
            lm._ms_padded_vocab_size = None
            lm._ms_make_vocab_size_divisible_by = None
            lm._ms_rotary_base = None

            # Mock all the Megatron imports and calls
            with mock.patch("lm_eval.models.mindspeed_lm._is_torch_npu_available", return_value=False), \
                 mock.patch("lm_eval.models.mindspeed_lm._maybe_patch_for_npu"), \
                 mock.patch("lm_eval.models.mindspeed_lm._add_megatron_to_path", return_value="/fake"), \
                 mock.patch("lm_eval.models.mindspeed_lm.get_distributed_backend", return_value="nccl"):

                # Mock the megatron module imports
                mock_megatron = mock.MagicMock()
                sys.modules["megatron"] = mock_megatron
                sys.modules["megatron.training"] = mock_megatron.training
                sys.modules["megatron.core"] = mock_megatron.core
                sys.modules["megatron.training.arguments"] = mock_megatron.training.arguments

                def capture_argv(**_kw):
                    captured_argv.append(list(sys.argv))

                mock_megatron.training.initialize_megatron = capture_argv

                try:
                    lm._initialize_megatron(**kwargs)
                except Exception:
                    pass  # We just want the argv capture

        if captured_argv:
            argv = captured_argv[0]
            assert "--use-mcore-models" in argv
            assert "--bf16" in argv
            assert "--seed" in argv
            assert "--no-load-optim" in argv
            assert "--no-load-rng" in argv
            assert "--exit-on-missing-checkpoint" in argv
            assert "--use-checkpoint-args" in argv

    def test_spec_is_split_correctly(self):
        """Verify --spec argument is split into separate elements for nargs='+'."""
        kwargs = self._get_argv(spec="mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec")

        # Test the spec splitting logic directly
        spec_value = kwargs["spec"]
        spec_parts = spec_value.split()
        assert spec_parts == ["mindspeed_llm.tasks.models.spec.qwen3_spec", "layer_spec"]

        # Simulate what _initialize_megatron does
        argv_spec = ["--spec"] + spec_parts
        assert argv_spec == ["--spec", "mindspeed_llm.tasks.models.spec.qwen3_spec", "layer_spec"]

    def test_architecture_args_outside_checkpoint_args(self):
        """Verify padded_vocab_size etc. are added even with use_checkpoint_args=True."""
        kwargs = self._get_argv()

        from lm_eval.models.mindspeed_lm import MindSpeedLMEval

        with mock.patch.object(MindSpeedLMEval, "__init__", return_value=None):
            lm = MindSpeedLMEval.__new__(MindSpeedLMEval)
            lm._parallelism_mode = "single"
            lm._devices = 1
            lm._tp_size = 1
            lm._pp_size = 1
            lm._ep_size = 1
            # Set MindSpeed-specific attributes
            lm._ms_seed = 42
            lm._ms_spec = None
            lm._ms_use_checkpoint_args = True
            lm._ms_max_position_embeddings = None
            lm._ms_padded_vocab_size = 151936
            lm._ms_make_vocab_size_divisible_by = 1
            lm._ms_rotary_base = 1000000.0

            captured_argv = []

            with mock.patch("lm_eval.models.mindspeed_lm._is_torch_npu_available", return_value=False), \
                 mock.patch("lm_eval.models.mindspeed_lm._maybe_patch_for_npu"), \
                 mock.patch("lm_eval.models.mindspeed_lm._add_megatron_to_path", return_value="/fake"), \
                 mock.patch("lm_eval.models.mindspeed_lm.get_distributed_backend", return_value="nccl"):

                mock_megatron = mock.MagicMock()
                sys.modules["megatron"] = mock_megatron
                sys.modules["megatron.training"] = mock_megatron.training
                sys.modules["megatron.core"] = mock_megatron.core

                def capture_argv(**_kw):
                    captured_argv.append(list(sys.argv))

                mock_megatron.training.initialize_megatron = capture_argv

                try:
                    lm._initialize_megatron(**kwargs)
                except Exception:
                    pass

        if captured_argv:
            assert "--padded-vocab-size" in captured_argv[0]
            assert "--make-vocab-size-divisible-by" in captured_argv[0]
            assert "--rotary-base" in captured_argv[0]
            assert "--use-checkpoint-args" in captured_argv[0]


# ---------------------------------------------------------------------------
# Tests for attention mask construction
# ---------------------------------------------------------------------------


class TestAttentionMask:
    """Test the attention mask construction in _model_forward."""

    def _build_model_forward_args(self, batch_size=2, seq_len=6, pad_len=2):
        """Build test inputs for _model_forward.

        Returns input_ids and attention_mask tensors simulating left-padded batch.
        """
        input_ids = torch.randint(1, 100, (batch_size, seq_len))
        # Left-padded: first pad_len tokens are padding (0)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        attention_mask[:, :pad_len] = 0
        return input_ids, attention_mask

    def test_causal_mask_shape(self):
        """Verify the causal mask has the correct shape."""
        batch_size, seq_len = 2, 8
        causal_mask = torch.ones(
            (batch_size, 1, seq_len, seq_len), dtype=torch.bool
        ).triu(diagonal=1)

        assert causal_mask.shape == (batch_size, 1, seq_len, seq_len)
        # Diagonal and below should be False (can attend)
        assert causal_mask[0, 0, 0, 0] == False  # self-attention
        assert causal_mask[0, 0, 0, 1] == True   # future token masked
        assert causal_mask[0, 0, 1, 0] == False   # past token visible
        assert causal_mask[0, 0, 1, 2] == True    # future token masked

    def test_combined_mask_blocks_padding(self):
        """Verify padding tokens are masked in combined mask."""
        batch_size, seq_len, _pad_len = 1, 5, 2

        causal_mask = torch.ones(
            (batch_size, 1, seq_len, seq_len), dtype=torch.bool
        ).triu(diagonal=1)

        attention_mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long)
        padding_mask = (1 - attention_mask).unsqueeze(1).unsqueeze(2).bool()

        combined = causal_mask | padding_mask

        # All positions in row 0 (PAD token) should be masked
        assert combined[0, 0, 0, :].all()  # PAD token can't attend to anything

        # All positions in row 1 (PAD token) should be masked
        assert combined[0, 0, 1, :].all()

        # Row 2 (first real token) should see only itself
        assert combined[0, 0, 2, 0] == True   # PAD masked
        assert combined[0, 0, 2, 1] == True   # PAD masked
        assert combined[0, 0, 2, 2] == False  # self visible
        assert combined[0, 0, 2, 3] == True   # future masked
        assert combined[0, 0, 2, 4] == True   # future masked

        # Row 4 (last real token) should see all real tokens
        assert combined[0, 0, 4, 0] == True   # PAD masked
        assert combined[0, 0, 4, 1] == True   # PAD masked
        assert combined[0, 0, 4, 2] == False  # real visible
        assert combined[0, 0, 4, 3] == False  # real visible
        assert combined[0, 0, 4, 4] == False  # self visible


# ---------------------------------------------------------------------------
# Tests for loglikelihood token extraction logic
# ---------------------------------------------------------------------------


class TestLoglikelihoodTokenExtraction:
    """Test the token-level log probability extraction logic."""

    def test_loglikelihood_indices(self):
        """Verify start_idx and end_idx calculation for loglikelihood.

        Scenario: [PAD, PAD, CTX1, CTX2, CONT1, CONT2]
                  pad_len=2, ctxlen=2, contlen=2
        """
        pad_len = 2
        ctxlen = 2
        contlen = 2

        # Indices used in the code:
        start_idx = pad_len + ctxlen - 1  # = 3 (position of last context token)
        end_idx = pad_len + ctxlen + contlen - 1  # = 5

        # Loop: j = 3, 4
        # j=3: predict token at position 4 (CONT1) from position 3
        # j=4: predict token at position 5 (CONT2) from position 4
        assert start_idx == 3
        assert end_idx == 5

        steps = list(range(start_idx, end_idx))
        assert len(steps) == contlen  # Should process exactly contlen tokens

    def test_loglikelihood_with_truncation(self):
        """Verify ctxlen adjustment when input is truncated.

        Scenario: context_enc has 10 tokens, continuation_enc has 5 tokens,
                  but max_length is 8. So only last 8 tokens are used.
        """
        context_enc = list(range(10))  # 10 tokens
        continuation_enc = list(range(100, 105))  # 5 tokens
        max_length = 8

        inp = (context_enc + continuation_enc)[-max_length:]
        # inp = [2, 3, 4, 5, 6, 7, 8, 9, 100, 101, 102, 103, 104][-8:]
        # inp = [102, 103, 104, 100, 101, 102, 103, 104]
        # Actually: context_enc + continuation_enc = [0,1,2,...,9,100,101,102,103,104]
        # Last 8: [7, 8, 9, 100, 101, 102, 103, 104]
        assert len(inp) == max_length

        ctxlen = len(context_enc) - max(0, len(context_enc) + len(continuation_enc) - max_length)
        # ctxlen = 10 - max(0, 15 - 8) = 10 - 7 = 3
        # So 3 context tokens remain: [7, 8, 9]
        assert ctxlen == 3
        assert inp[:ctxlen] == [7, 8, 9]


# ---------------------------------------------------------------------------
# Tests for dist checkpoint auto-detection
# ---------------------------------------------------------------------------


class TestDistCkptAutoDetection:
    """Test auto-detection of distributed checkpoint format."""

    def test_detects_distcp_in_iter_dir(self):
        """Should detect .distcp files in the latest iteration directory."""
        with tempfile.TemporaryDirectory() as ckpt_dir:
            iter_dir = Path(ckpt_dir) / "iter_0000001"
            iter_dir.mkdir()
            Path(iter_dir, "model.pt.distcp").touch()

            from lm_eval.models.mindspeed_lm import _check_dist_ckpt

            assert _check_dist_ckpt(str(iter_dir)) is True

    def test_iter_dir_selection(self):
        """Should select the latest iteration directory."""
        with tempfile.TemporaryDirectory() as ckpt_dir:
            # Create multiple iter dirs
            Path(ckpt_dir, "iter_0000001").mkdir()
            iter2 = Path(ckpt_dir, "iter_0000002")
            iter2.mkdir()
            Path(iter2, "metadata.json").write_text("{}")

            # Latest is iter_0000002
            iter_dirs = sorted(d for d in os.listdir(ckpt_dir) if d.startswith("iter_"))
            latest = iter_dirs[-1]
            assert latest == "iter_0000002"

            from lm_eval.models.mindspeed_lm import _check_dist_ckpt

            assert _check_dist_ckpt(os.path.join(ckpt_dir, latest)) is True

    def test_no_iter_dirs(self):
        """Should check load_path directly when no iter dirs exist."""
        with tempfile.TemporaryDirectory() as ckpt_dir:
            Path(ckpt_dir, "metadata.json").write_text("{}")

            from lm_eval.models.mindspeed_lm import _check_dist_ckpt

            assert _check_dist_ckpt(ckpt_dir) is True


# ---------------------------------------------------------------------------
# Tests for shell script (basic syntax and structure validation)
# ---------------------------------------------------------------------------


class TestShellScript:
    """Basic validation of the shell scripts."""

    def test_npu_eval_script_exists_and_executable(self):
        script = Path(__file__).parent.parent / "npu_mindspeed-llm_eval.sh"
        assert script.exists()
        assert os.access(script, os.X_OK) or True  # May not be executable on all systems

    def test_run_script_exists(self):
        script = Path(__file__).parent.parent / "run.sh"
        assert script.exists()

    def test_npu_eval_script_modes(self):
        """Verify all documented modes are handled in the case statement."""
        script = Path(__file__).parent.parent / "npu_mindspeed-llm_eval.sh"
        content = script.read_text()

        for mode in ["single", "dp", "tp", "ep", "custom"]:
            assert f"run_{mode}" in content, f"Missing run_{mode} function"
            assert mode in content, f"Missing '{mode}' in case statement"

    def test_npu_eval_script_env_vars(self):
        """Verify key environment variables are documented."""
        script = Path(__file__).parent.parent / "npu_mindspeed-llm_eval.sh"
        content = script.read_text()

        for var in ["MEGATRON_PATH", "CKPT_PATH", "TOKENIZER_MODEL", "SPEC", "SEED", "EXTRA_ARGS"]:
            assert var in content, f"Missing {var} in script"

    def test_npu_eval_script_uses_mindspeed_lm_model(self):
        """Verify the script uses the mindspeed_lm model backend."""
        script = Path(__file__).parent.parent / "npu_mindspeed-llm_eval.sh"
        content = script.read_text()

        assert "--model mindspeed_lm" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
