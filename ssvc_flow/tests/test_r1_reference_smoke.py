"""CPU orchestration regressions; these never certify actual Qwen numerics."""

import copy
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml
from test_model_smoke import FakeAdapter, panel

from src.optimizer_fork import parameter_hash
from src.r1_reference_smoke import run_reference_smoke


class ValidMixAdapter(FakeAdapter):
    def generate(self, prepared, *, seed, max_new_tokens):
        self.generation_calls += 1
        token = (0, 1, 3)[seed % 3]
        return {
            "raw_completion": {0: "[1,2,3,4]", 1: "[1,2,3,5]", 3: "invalid"}[token],
            "token_ids": [token, 2],
            "completion_length": 2,
            "stop_reason": "eos",
            "elapsed_seconds": 0.001,
            "behavior_token_logprobs": self.model.lora.log_softmax(-1)[[token, 2]]
            .detach()
            .tolist(),
        }


def setup(tmp_path, adapter_class=ValidMixAdapter):
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/next_stage.yaml").read_text())
    config["data_root"] = str(tmp_path)
    adapter = adapter_class("qwen35_9b", {"id": config["model"]["id"]})
    config["model"]["revision"] = adapter.revision
    adapter.audit.update(
        {
            "chat_template_hash": "fixture-template",
            "frozen_parameter_hash": parameter_hash(adapter.model, trainable=False),
            "probability_execution": "uncached_prefix_recompute",
        }
    )
    certificate = {
        "status": "PASS",
        "selected_path": "uncached_prefix_recompute",
        "model_revision": adapter.revision,
        "initial_adapter_hash": parameter_hash(adapter.model, trainable=True),
        "fixed_sequence_boundaries": {
            "passed": True,
            "checks_completed": 27,
            "checks_expected": 27,
            "not_model_generated_not_training": True,
        },
        "model_audit": dict(adapter.audit),
        "thresholds": {k: v for k, v in config["R1"].items() if k.startswith("parity_alarm_")},
        "checks": {
            name: {
                "passed": True,
                "sequences": 120,
                "failed_sequence_checks": 0,
                "top1_measured": name in ("reference_repeat", "behavior"),
            }
            for name in (
                "reference_repeat",
                "behavior",
                "evaluation_likelihood",
                "training_likelihood",
            )
        },
    }
    return config, panel(tmp_path), adapter, certificate


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda c: None, "PASS reference-path certificate"),
        (lambda c: {**c, "status": "FAIL"}, "PASS reference-path certificate"),
        (lambda c: {**c, "model_revision": "b" * 40}, "model revision differs"),
        (lambda c: {**c, "initial_adapter_hash": "changed"}, "initial adapter hash differs"),
        (lambda c: {**c, "checks": {}}, "check is missing or failed"),
        (lambda c: {**c, "checks": {"reference_repeat": True}}, "check is missing or failed"),
        (
            lambda c: {**c, "model_audit": {**c["model_audit"], "processor_hash": "changed"}},
            "model audit drift",
        ),
        (lambda c: {**c, "thresholds": {}}, "threshold drift"),
        (lambda c: {**c, "fixed_sequence_boundaries": {}}, "boundary certificate is incomplete"),
    ],
)
def test_rejects_missing_failed_or_drifted_certificate_before_updates(tmp_path, mutation, match):
    config, scenes, adapter, certificate = setup(tmp_path)
    with pytest.raises(ValueError, match=match):
        run_reference_smoke(config, scenes, tmp_path / "smoke", adapter, mutation(certificate))
    assert adapter.generation_calls == 0
    assert not (tmp_path / "smoke/checkpoint.pt").exists()


def test_four_updates_use_128_rollouts_and_discard_scratch_state(tmp_path):
    config, scenes, adapter, certificate = setup(tmp_path)
    rng = torch.get_rng_state().clone()
    result = run_reference_smoke(config, scenes, tmp_path / "smoke", adapter, certificate)
    assert result["passed"], result
    assert result["execution_kind"] == "CPU_FAKE_ADAPTER_FIXTURE"
    assert result["raw_sample_count"] == result["generation_calls_this_invocation"] == 128
    assert len(result["training_steps"]) == 4
    assert result["initial_rollout_count"] == 0
    assert result["replay_optimizer_updates"] == 2
    assert result["invariant_optimizer_updates"] == 2
    assert result["implementation_invariants"]["gradient_accumulation"]["passed"]
    assert (
        result["implementation_invariants"]["gradient_accumulation"]["batched_model_forward_tested"]
        is False
    )
    assert all(result["scratch_restoration"].values())
    assert parameter_hash(adapter.model, trainable=True) == certificate["initial_adapter_hash"]
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(
        '"phase": "R1"' in line
        for line in (tmp_path / "smoke/samples.jsonl").read_text().splitlines()
    )
    assert result["I4"] == "NOT_RUN"


def test_missing_natural_valid_contrast_is_inconclusive(tmp_path):
    config, scenes, adapter, certificate = setup(tmp_path, FakeAdapter)
    result = run_reference_smoke(config, scenes, tmp_path / "smoke", adapter, certificate)
    assert result["status"] == "INCONCLUSIVE"
    assert result["checks"]["real_optimizer_steps"]
    assert result["implementation_invariants"]["status"] == "INCONCLUSIVE"
    assert result["invariant_optimizer_updates"] == 0


def test_updated_bank_likelihood_failure_stops_and_restores(tmp_path):
    class DriftAdapter(FakeAdapter):
        def logprobs(self, prepared, completion, require_grad=False):
            value = super().logprobs(prepared, completion, require_grad=require_grad)
            return (
                value + 0.2
                if not require_grad and bool(self.model.lora.detach().abs().sum())
                else value
            )

    config, scenes, adapter, certificate = setup(tmp_path, DriftAdapter)
    result = run_reference_smoke(config, scenes, tmp_path / "smoke", adapter, certificate)
    assert result["status"] == "FAIL"
    assert len(result["training_steps"]) == 1
    assert result["raw_sample_count"] == 64
    assert "likelihood checks" in result["error"]["message"]
    assert all(result["scratch_restoration"].values())
    assert parameter_hash(adapter.model, trainable=True) == certificate["initial_adapter_hash"]


def test_invalid_image_stops_before_generation(tmp_path):
    config, scenes, adapter, certificate = setup(tmp_path)
    with patch("src.r1_reference_smoke._image_audit", return_value={"passed": False}):
        result = run_reference_smoke(config, scenes, tmp_path / "smoke", adapter, certificate)
    assert result["status"] == "FAIL"
    assert result["raw_sample_count"] == 0
    assert result["training_steps"] == []
    assert all(result["scratch_restoration"].values())


def test_rejects_incomplete_numeric_panel_and_unmeasured_top1(tmp_path):
    config, scenes, adapter, certificate = setup(tmp_path)
    certificate["checks"]["behavior"]["sequences"] = 119
    with pytest.raises(ValueError, match="check is missing or failed"):
        run_reference_smoke(config, scenes, tmp_path / "smoke", adapter, certificate)
    certificate["checks"]["behavior"]["sequences"] = 120
    certificate["checks"]["behavior"]["top1_measured"] = False
    with pytest.raises(ValueError, match="check is missing or failed"):
        run_reference_smoke(config, scenes, tmp_path / "smoke", adapter, certificate)


def test_actual_frozen_weight_drift_cannot_use_stale_audit_hash(tmp_path):
    config, scenes, adapter, certificate = setup(tmp_path)
    with torch.no_grad():
        adapter.model.frozen[0] += 1
    with pytest.raises(ValueError, match="frozen parameters differ"):
        run_reference_smoke(config, scenes, tmp_path / "smoke", adapter, certificate)
    assert adapter.generation_calls == 0


def test_calibration_only_and_budget_are_enforced(tmp_path):
    config, scenes, adapter, certificate = setup(tmp_path)
    wrong = copy.deepcopy(scenes)
    wrong[0]["split"] = "train"
    with pytest.raises(ValueError, match="calibration"):
        run_reference_smoke(config, wrong, tmp_path / "smoke", adapter, certificate)
    config["R1"]["max_smoke_training_rollouts"] = 127
    with pytest.raises(ValueError, match="128-rollout budget"):
        run_reference_smoke(config, scenes, tmp_path / "smoke", adapter, certificate)
