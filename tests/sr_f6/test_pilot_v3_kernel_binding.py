"""CPU source-correspondence controls, never actual new trainer numerical PASS."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from sr_f6 import pilot_v3_kernel_binding as binding


@pytest.fixture
def training_sources(tmp_path):
    current = Path(binding.__file__).parent / "training.py"
    text = current.read_text()
    old, new = tmp_path / "old.py", tmp_path / "new.py"
    old.write_text(text)
    new.write_text(text)
    return old, new, text


def test_exact_training_function_mapping_includes_full_update(training_sources):
    old, new, _ = training_sources
    result = binding._function_map(old, new, "sr_f6.training")
    names = {row["function"] for row in result["functions"]}
    assert "sr_f6.training._update" in names
    assert "sr_f6.training._verify_prepared" in names
    assert all(
        row["producer_source_sha256"] == row["current_source_sha256"] for row in result["functions"]
    )
    assert "status" not in result


@pytest.mark.parametrize("mutation", ["arithmetic", "constant", "import", "shadow"])
def test_changed_numerical_body_constants_imports_and_shadowing_rejected(
    training_sources, mutation
):
    old, new, text = training_sources
    if mutation == "arithmetic":
        tree = ast.parse(text)
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_update")
        lines = text.splitlines(keepends=True)
        lines.insert(node.end_lineno, "    return 'changed numerical path'\n")
        text = "".join(lines)
    elif mutation == "constant":
        text = text.replace("BATCH_SIZE = 128", "BATCH_SIZE = 64")
    elif mutation == "import":
        text = text.replace(
            "from .objective import sequence_loss", "from other.objective import sequence_loss"
        )
    else:
        text += "\nsequence_loss = lambda *args: 0\n"
    new.write_text(text)
    with pytest.raises(PermissionError):
        binding._function_map(old, new, "sr_f6.training")


def test_source_routing_changes_do_not_claim_executed_numerical_pass(training_sources):
    old, new, text = training_sources
    # This helper controls output location, not the registered update arithmetic.
    text = text.replace(
        'root = Path(runtime.identity["run_root"])',
        'root = Path(runtime.identity["run_root"])\n    # V3 route',
    )
    new.write_text(text)
    result = binding._function_map(old, new, "sr_f6.training")
    assert result["functions"]
    assert all("PASS" not in str(item) for item in result["functions"])
    assert any("Adam" in item for item in binding.LIMITATIONS)
    assert any("reference=current" in item for item in binding.LIMITATIONS)


def test_runtime_call_graph_bodies_and_dependency_modes_are_explicit():
    names = binding.FUNCTIONS["sr_f6.runtime"]
    assert set(
        (
            "Runtime.prepare_input",
            "Runtime._batch_forward",
            "Runtime._native_forward",
            "Runtime.batch_reference_forward",
            "Runtime._generate_native_group",
        )
    ) <= set(names)
    assert set(
        (
            "sr_f6.numerical_replay",
            "sr_f6.replay",
            "sr_f6.objective",
            "sr_f6.credit",
            "sr_f5.runtime",
            "sr_f12.runtime",
            "sr_f3_ce.runtime",
            "mm_dev.runtime",
        )
    ) <= set(binding.WHOLE_MODULES)


def test_loaded_origin_must_be_current_deployment(tmp_path):
    with pytest.raises(PermissionError, match="different deployment"):
        binding._loaded_origins(tmp_path)


@pytest.mark.parametrize("mutation", ["tf32", "dtype", "top_level", "duplicate_function", "guard"])
def test_loader_and_unlisted_source_cannot_escape_numerical_closure(tmp_path, mutation):
    """Concrete TF32 reproducer plus full-module bypasses; no model is loaded."""
    text = (Path(binding.__file__).parent / "runtime.py").read_text()
    old, new = tmp_path / "runtime-old.py", tmp_path / "runtime-new.py"
    old.write_text(text)
    if mutation == "tf32":
        anchor = "    configure_audited_backend()\n"
        assert text.count(anchor) == 1
        text = text.replace(anchor, anchor + "    torch.backends.cuda.matmul.allow_tf32 = True\n")
    elif mutation == "dtype":
        anchor = "    configure_audited_backend()\n"
        text = text.replace(anchor, anchor + "    torch.set_default_dtype(torch.float16)\n")
    elif mutation == "top_level":
        text += "\nimport torch\ntorch.backends.cuda.matmul.allow_tf32 = True\n"
    elif mutation == "duplicate_function":
        text += "\ndef generation_recipe(*, protocol='structured'):\n    return {}\n"
    else:
        text = text.replace(
            "return EXECUTION_IDENTITY_FIELDS + (V3_INPUT_IDENTITY_FIELDS",
            "return EXECUTION_IDENTITY_FIELDS + (()",
        )
    new.write_text(text)
    with pytest.raises(PermissionError):
        binding._function_map(old, new, "sr_f6.runtime")


def test_exact_projection_checks_all_module_statements():
    assert "sr_f6.runtime" in binding.ROUTING_TRANSFORMS
    # A harmless-looking change to an unlisted old function is still not one of
    # the exact frozen routing operations. It cannot be labeled nonnumerical.
    original = "def unrelated():\n    return 1\n"
    changed = "def unrelated():\n    return 2\n"
    with pytest.raises(PermissionError):
        binding._approved_module(original, changed, "sr_f6.runtime")
