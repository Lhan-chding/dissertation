"""Reject incomplete CPU preparation and unbound or corrupted ENGINE certificates."""

import json

import pytest

from sr_f1.contract import PLAN_ID, file_hash, load_config
from sr_f1.freeze import verify_engine_receipt, verify_execution


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize(
    "status", ["LOCAL_PREPARED", "CPU_VERIFIED", "PASS", "CPU_FREEZE_REJECTED"]
)
def test_only_full_frozen_cpu_state_allows_gpu(tmp_path, status):
    write(tmp_path / "EXECUTION_FREEZE.json", dict(plan_id=PLAN_ID, status=status))
    with pytest.raises(PermissionError, match="has not been frozen"):
        verify_execution(load_config(), tmp_path)


def test_configuration_change_is_not_an_execution_override(tmp_path):
    config = load_config()
    config["training"]["updates"] = 97
    with pytest.raises(PermissionError, match="authenticated contract"):
        verify_execution(config, tmp_path)


def engine_fixture(root, surrogate=False):
    write(root / "EXECUTION_FREEZE.json", {"fixture": "CPU identity"})
    write(root / "adapter/weights.json", {"fixture": "adapter bytes"})
    write(
        root / "COMMON_START.json",
        dict(
            plan_id=PLAN_ID,
            status="VERIFIED",
            model_id="SRF1_COMMON_START",
            freeze_sha256=file_hash(root / "EXECUTION_FREEZE.json"),
            adapter_path="adapter",
            adapter_file_hashes={"weights.json": file_hash(root / "adapter/weights.json")},
        ),
    )
    traces, artifacts = [], {}
    for mode in ["natural", "stress"] if surrogate else ["natural"]:
        relative = f"engineering/engine/{mode}/raw.json"
        write(root / relative, {"fixture": "trace"})
        evidence = {relative: file_hash(root / relative)}
        constant = surrogate and mode == "natural"
        trace = dict(
            plan_id=PLAN_ID,
            mode=mode,
            status="EXACT_CONSTANT_REWARD_TRACE" if constant else "EXACT_NONZERO_TRACE",
            stress_allowed=constant,
            logical_steps=4,
            physical_updates=8,
            valid_trace_updates=8,
            completion_attempts=1024,
            valid_trace_completions=1024,
            exact_state_and_rng=True,
            exact_token_and_slot_paths=True,
            all_probability_gates_passed=True,
            artifact_hashes=evidence,
        )
        comparison = f"engineering/engine/{mode}/COMPARISON.json"
        write(root / comparison, trace)
        artifacts.update(evidence)
        artifacts[comparison] = file_hash(root / comparison)
        traces.append(trace)
    receipt = dict(
        plan_id=PLAN_ID,
        status="PASS_NONZERO_SURROGATE_KERNEL_RESUME" if surrogate else "PASS_NATURAL_GRPO_RESUME",
        freeze_sha256=file_hash(root / "EXECUTION_FREEZE.json"),
        common_start_sha256=file_hash(root / "COMMON_START.json"),
        natural=traces[0],
        stress=traces[1] if surrogate else None,
        artifact_hashes=artifacts,
    )
    write(root / "ENGINE_PROBABILITY_GRADIENT_RESUME.json", receipt)
    return receipt


@pytest.mark.parametrize("surrogate", [False, True])
def test_complete_engine_receipts_and_immutable_traces(tmp_path, surrogate):
    receipt = engine_fixture(tmp_path, surrogate)
    assert verify_engine_receipt(tmp_path) == receipt
    write(tmp_path / "engineering/engine/natural/raw.json", {"fixture": "changed"})
    with pytest.raises(PermissionError, match="artifact changed"):
        verify_engine_receipt(tmp_path)


def test_plain_pass_does_not_certify_native_engine(tmp_path):
    receipt = engine_fixture(tmp_path)
    receipt["status"] = "PASS"
    write(tmp_path / "ENGINE_PROBABILITY_GRADIENT_RESUME.json", receipt)
    with pytest.raises(PermissionError, match="certification missing"):
        verify_engine_receipt(tmp_path)


def test_engine_is_bound_to_freeze_and_common_start(tmp_path):
    engine_fixture(tmp_path)
    write(tmp_path / "EXECUTION_FREEZE.json", {"fixture": "new freeze"})
    with pytest.raises(PermissionError, match="different execution freeze"):
        verify_engine_receipt(tmp_path)


def test_counterfeit_status_without_full_trace_fails(tmp_path):
    receipt = engine_fixture(tmp_path)
    receipt["natural"]["valid_trace_completions"] = 512
    write(tmp_path / "ENGINE_PROBABILITY_GRADIENT_RESUME.json", receipt)
    with pytest.raises(PermissionError, match="Incomplete ENGINE evidence"):
        verify_engine_receipt(tmp_path)
