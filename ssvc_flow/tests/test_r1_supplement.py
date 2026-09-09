"""RED/GREEN controls for the fixed-bank supplement; no GPU claim from fixtures."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from test_r1_reference_smoke import setup

from src.core import file_hash, source_commit, write_json
from src.optimizer_fork import load_checkpoint, parameter_hash, save_checkpoint
from src.r1_reference_smoke import run_reference_smoke
from src.r1_supplement import _ExecutionMeter, run_supplement

SOURCE_FILES = (
    "src/r1_reference_smoke.py",
    "src/r1_parity.py",
    "src/grpo_update.py",
    "src/smoke_runtime.py",
    "src/optimizer_fork.py",
    "src/model_adapters/base.py",
)


@pytest.fixture
def evidence(tmp_path):
    config, scenes, adapter, certificate = setup(tmp_path / "data")
    data = Path(config["data_root"])
    (data / "calibration.jsonl").write_text("".join(json.dumps(row) + "\n" for row in scenes))
    root = tmp_path / "source"
    root.mkdir()
    revision = source_commit()
    certificate["source_commit"] = revision
    result = run_reference_smoke(config, scenes, root / "smoke", adapter, certificate)
    assert result["passed"]
    write_json(root / "production_path_validation.json", certificate)
    write_json(root / "training_smoke.json", result)
    write_json(
        root / "manifest.json",
        {
            "phase": "R1",
            "files": [
                {
                    "path": name,
                    "sha256": file_hash(root / name),
                    "bytes": (root / name).stat().st_size,
                }
                for name in ("production_path_validation.json", "training_smoke.json")
            ],
        },
    )
    write_json(
        root / "status.json",
        {
            "status": "PASS",
            "source_commit": revision,
            "execution_kind": "CPU_FAKE_ADAPTER_FIXTURE",
            "source_files": {
                name: file_hash(Path(__file__).parents[1] / name) for name in SOURCE_FILES
            },
        },
    )
    (root / "source-commit.txt").write_text(revision + "\n")
    return root, adapter


def snapshot(root):
    return {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()}


def run_fixture(evidence, tmp_path):
    root, adapter = evidence
    return run_supplement(root, tmp_path / "supplement", _adapter_factory=lambda *a, **k: adapter)


def test_warm_zero_lambda_and_order_controls_use_no_new_samples(evidence, tmp_path):
    root, adapter = evidence
    before_files = snapshot(root)
    before_model = parameter_hash(adapter.model, trainable=True)
    generated_before = adapter.generation_calls
    result = run_fixture(evidence, tmp_path)
    assert result["status"] == "PASS", result
    assert result["execution_kind"] == "CPU_FAKE_ADAPTER_FIXTURE"
    assert result["selected_checkpoint_step"] == 1
    assert result["new_rollouts"] == 0
    assert adapter.generation_calls == generated_before
    assert result["zero_gradient_adam"]["all_gradients_explicit_zero"]
    assert result["zero_gradient_adam"]["manual_zero_tensor_equal"]
    assert result["zero_gradient_adam"]["none_control_unchanged"]
    assert result["zero_gradient_adam"]["moments_decay_as_expected"]
    assert result["zero_gradient_adam"]["parameters_changed"]
    assert result["lambda_zero_control"]["independent_reward_implementations"]
    assert result["lambda_zero_control"]["passed"]
    assert result["candidate_order"]["orders"] == [[0.0, 1.0], [1.0, 0.0]]
    assert result["candidate_order"]["passed"]
    assert result["measurement"]["backward_calls_observed"] == 18
    assert result["candidate_order"]["lambda_candidates_have_distinct_advantages"]
    assert result["measurement"]["optimizer_step_calls_observed"] == 9
    assert result["batched_model_forward"] == "NOT_ADOPTED_UNVERIFIED"
    assert all(result["scratch_restoration"].values())
    assert parameter_hash(adapter.model, trainable=True) == before_model
    assert snapshot(root) == before_files


@pytest.mark.parametrize("which", ["certificate", "config", "source", "bank_policy"])
def test_evidence_drift_fails_before_updates(evidence, tmp_path, which):
    root, _adapter = evidence
    if which == "certificate":
        path = root / "production_path_validation.json"
        value = json.loads(path.read_text())
        value["status"] = "FAIL"
    elif which == "config":
        path = root / "smoke/runtime_lock.json"
        value = json.loads(path.read_text())
        value["config"]["optimizer"]["learning_rate"] *= 2
    elif which == "source":
        path = root / "status.json"
        value = json.loads(path.read_text())
        value["source_files"]["src/grpo_update.py"] = "wrong"
    else:
        path = root / "smoke/samples.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        for row in rows:
            if row["optimizer_step"] == 1:
                row["adapter_hash"] = "wrong"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        value = None
    if value is not None:
        write_json(path, value)
    result = run_fixture(evidence, tmp_path)
    assert result["status"] == "FAIL"
    assert result["measurement"].get("optimizer_step_calls_observed", 0) == 0
    assert not result["passed"]


def test_missing_step1_checkpoint_does_not_fallback_to_step4(evidence, tmp_path):
    root, _ = evidence
    (root / "smoke/resume_probe.pt").unlink()
    result = run_fixture(evidence, tmp_path)
    assert result["status"] == "BLOCKED"
    assert "resume_probe.pt" in result["error"]


def test_manifest_detects_tampering_in_otherwise_unused_measurement(evidence, tmp_path):
    root, _ = evidence
    path = root / "training_smoke.json"
    value = json.loads(path.read_text())
    value["generation_seconds"] += 1
    write_json(path, value)
    result = run_fixture(evidence, tmp_path)
    assert result["status"] == "FAIL"
    assert "manifest hash/size mismatch" in result["error"]


def test_empty_adam_is_rejected(evidence, tmp_path):
    root, _ = evidence
    lock = json.loads((root / "smoke/runtime_lock.json").read_text())
    path = root / "smoke/resume_probe.pt"
    state = load_checkpoint(path, lock["identity"])
    state["optimizer"]["state"] = {}
    save_checkpoint(path, state, lock["identity"])
    result = run_fixture(evidence, tmp_path)
    assert result["status"] == "FAIL"
    assert "checkpoint" in result["error"]


def test_stored_old_likelihood_mismatch_stops_before_updates(evidence, tmp_path):
    root, _ = evidence
    path = root / "smoke/samples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if row["optimizer_step"] == 1:
            row["old_logprobs"] = [value + 0.2 for value in row["old_logprobs"]]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = run_fixture(evidence, tmp_path)
    assert result["status"] == "FAIL"
    assert "old likelihood differs" in result["error"]
    assert result["measurement"]["optimizer_step_calls_observed"] == 0
    assert all(result["scratch_restoration"].values())


def test_generic_lambda_zero_bug_is_detected_independently(evidence, tmp_path):
    from src.grpo_update import reward_channels

    def incorrect(category, arm="X_BASE", auxiliary_weight=None):
        value = reward_channels(category, arm, auxiliary_weight)
        return {**value, "sum": value["sum"] + 1} if auxiliary_weight == 0 else value

    with patch("src.r1_supplement.reward_channels", side_effect=incorrect):
        result = run_fixture(evidence, tmp_path)
    assert result["status"] == "FAIL"
    assert result["lambda_zero_control"]["rewards_equal"] is False
    assert all(result["scratch_restoration"].values())


def test_refuses_output_inside_source_evidence(evidence):
    root, _ = evidence
    before = snapshot(root)
    with pytest.raises(ValueError, match="outside"):
        run_supplement(root, root / "new")
    assert snapshot(root) == before


def test_meter_observes_actual_checkpoint_layer_recompute_and_restores_torch_api():
    import types

    from torch.utils.checkpoint import checkpoint

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Module()
            self.language_model.layers = torch.nn.ModuleList([torch.nn.Linear(3, 3)])

        def forward(self, input_ids, attention_mask):
            return checkpoint(self.language_model.layers[0], input_ids, use_reentrant=False)

    model = torch.nn.Module()
    model.model = Backbone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    original = torch.autograd.backward
    meter = _ExecutionMeter(types.SimpleNamespace(model=model), optimizer)
    try:
        with meter.scope("real_checkpoint_probe"):
            output = model.model(input_ids=torch.ones(2, 3), attention_mask=torch.ones(2, 3))
            output.square().sum().backward()
            optimizer.step()
        report = meter.report()
        assert report["backward_calls_observed"] == 1
        assert report["optimizer_step_calls_observed"] == 1
        assert report["by_phase"]["real_checkpoint_probe"]["top_level_forward_calls"] == 1
        calls = report["language_layer_forward_calls"]
        assert calls["real_checkpoint_probe/ordinary_forward/grad_enabled=True"] == 1
        assert calls["real_checkpoint_probe/during_backward/grad_enabled=True"] == 1
    finally:
        meter.close()
    assert torch.autograd.backward is original
