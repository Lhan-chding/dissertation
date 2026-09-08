"""CPU fixture orchestration is not NTU model evidence."""

import json
from typing import ClassVar
from unittest.mock import patch

import pytest
from test_model_smoke import FakeAdapter, panel

from src.core import load_config, write_json


def fixture_inputs(tmp_path):
    scenes = [{**s, "split": "dev"} for s in panel(tmp_path)[:2]]
    config = {**load_config(), "data_root": str(tmp_path)}
    config["models"]["qwen35_9b"]["revision"] = "a" * 40
    return config, scenes


class FrozenFake(FakeAdapter):
    calls: ClassVar[list] = []
    fail_at = None

    def generate(self, prepared, *, seed, max_new_tokens, do_sample=True):
        assert not any(p.requires_grad for p in self.model.parameters())
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise RuntimeError("injected interruption")
        self.calls.append((seed, do_sample))
        self.forward_calls += 2
        return {
            **super().generate(prepared, seed=seed, max_new_tokens=max_new_tokens),
            "decoding_mode": "sample" if do_sample else "greedy",
        }


@pytest.fixture(autouse=True)
def reset_fake():
    FrozenFake.calls = []
    FrozenFake.fail_at = None


def test_frozen_exact_panel_no_training_and_resume(tmp_path):
    from src.frozen_runtime import run_frozen

    config, scenes = fixture_inputs(tmp_path)
    out = tmp_path / "P3"
    result = run_frozen(config, scenes, out, "qwen35_9b", _adapter_factory=FrozenFake)
    assert result["passed"]
    assert result["execution_kind"] == "CPU_FAKE_ADAPTER_FIXTURE"
    assert result["raw_sample_count"] == 68
    assert result["optimizer_updates"] == result["backward_calls"] == 0
    rows = [json.loads(x) for x in (out / "samples.jsonl").read_text().splitlines()]
    assert sum(r["decode_mode"] == "sample" for r in rows) == 64
    assert sum(r["decode_mode"] == "greedy" for r in rows) == 4
    before = (out / "samples.jsonl").read_bytes()
    FrozenFake.calls = []
    resumed = run_frozen(config, scenes, out, "qwen35_9b", resume=True, _adapter_factory=FrozenFake)
    assert resumed["passed"] and not FrozenFake.calls
    assert (out / "samples.jsonl").read_bytes() == before
    config["sample_seed"] += 1
    with pytest.raises(ValueError, match="identity"):
        run_frozen(config, scenes, out, "qwen35_9b", resume=True, _adapter_factory=FrozenFake)
    assert (out / "samples.jsonl").read_bytes() == before


def test_resume_interruption_matches_uninterrupted(tmp_path):
    from src.frozen_runtime import run_frozen

    config, scenes = fixture_inputs(tmp_path)
    FrozenFake.fail_at = 9
    with pytest.raises(RuntimeError, match="injected interruption"):
        run_frozen(config, scenes, tmp_path / "partial", "qwen35_9b", _adapter_factory=FrozenFake)
    FrozenFake.fail_at = None
    run_frozen(
        config, scenes, tmp_path / "partial", "qwen35_9b", resume=True, _adapter_factory=FrozenFake
    )
    run_frozen(config, scenes, tmp_path / "whole", "qwen35_9b", _adapter_factory=FrozenFake)

    def outputs(name):
        return [
            (r["sample_key"], r["sample_seed"], r["raw_completion"], r["token_ids"])
            for r in map(json.loads, (tmp_path / name / "samples.jsonl").read_text().splitlines())
        ]

    assert outputs("partial") == outputs("whole")


def test_gates_fail_before_loading_and_no_confirm(tmp_path):
    from src.frozen_runtime import run_frozen

    config, scenes = fixture_inputs(tmp_path)
    with patch("src.model_adapters.load_adapter") as loader:
        with pytest.raises((ValueError, PermissionError), match="P1"):
            run_frozen(config, scenes, tmp_path / "P3", "qwen35_9b")
        loader.assert_not_called()
    with pytest.raises(ValueError, match="dev"):
        run_frozen(
            config,
            [{**scenes[0], "split": "confirm"}],
            tmp_path / "confirm",
            "qwen35_9b",
            _adapter_factory=FrozenFake,
        )


def test_resume_rejects_extra_records_and_image_changes(tmp_path):
    from src.frozen_runtime import run_frozen

    config, scenes = fixture_inputs(tmp_path)
    out = tmp_path / "P3"
    run_frozen(config, scenes, out, "qwen35_9b", _adapter_factory=FrozenFake)
    ledger = out / "samples.jsonl"
    rows = list(map(json.loads, ledger.read_text().splitlines()))
    rows[0]["sample_key"] = "unexpected"
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match=r"ledger|sample"):
        run_frozen(config, scenes, out, "qwen35_9b", resume=True, _adapter_factory=FrozenFake)
    (tmp_path / scenes[0]["image_path"]).write_text("tampered")
    with pytest.raises(ValueError, match="image"):
        run_frozen(config, scenes, out, "qwen35_9b", resume=True, _adapter_factory=FrozenFake)


def test_p1_evidence_cannot_use_fixture(tmp_path):
    from src.frozen_runtime import validate_p1_evidence

    root = tmp_path / "P1"
    root.mkdir()
    write_json(root / "status.json", {"phase": "P1", "status": "PASS"})
    write_json(
        root / "model_audit.json", {"passed": True, "execution_kind": "CPU_FAKE_ADAPTER_FIXTURE"}
    )
    write_json(root / "validated_runtime_lock.json", {"runtime_status": "P1_PASSED"})
    with pytest.raises(ValueError, match="real CUDA"):
        validate_p1_evidence(root, load_config(), "qwen35_9b")


def test_frozen_dry_run_and_gate_does_not_need_gpu(tmp_path):
    from src.rollout import main

    assert (
        main(
            [
                "--phase",
                "frozen",
                "--model",
                "qwen25vl_3b",
                "--dry-run",
                "--out",
                str(tmp_path / "dry"),
            ]
        )
        == 0
    )
    status = json.loads((tmp_path / "dry/status.json").read_text())
    assert status["details"]["rollout_count"] == 4896
    assert status["details"]["backward_count"] == 0
    assert main(["--phase", "frozen", "--out", str(tmp_path / "blocked")]) == 2
    assert "P1" in (tmp_path / "blocked/status.json").read_text()


def test_real_peft_context_restores_flags_without_invalidating_frozen_run(tmp_path):
    import torch

    peft = pytest.importorskip("peft")
    from test_model_official_tiny import tiny_model

    from src.frozen_runtime import run_frozen

    class ActualPeft(FrozenFake):
        def __init__(self, key, spec, **kwargs):
            super().__init__(key, spec, **kwargs)
            self.model = peft.get_peft_model(
                tiny_model(),
                peft.LoraConfig(
                    r=2,
                    target_modules=["model.language_model.layers.0.mlp.gate_proj"],
                    task_type="CAUSAL_LM",
                ),
            )

        def generate(self, prepared, *, seed, max_new_tokens, do_sample=True):
            assert not any(p.requires_grad for p in self.model.parameters())
            return {
                "raw_completion": "invalid",
                "token_ids": [2],
                "completion_length": 1,
                "stop_reason": "eos",
                "elapsed_seconds": 0.001,
                "behavior_token_logprobs": [-1.0],
            }

        def logprobs(self, prepared, completion, require_grad=False):
            return torch.tensor([-1.0])

    config, scenes = fixture_inputs(tmp_path)
    result = run_frozen(config, scenes, tmp_path / "peft", "qwen35_9b", _adapter_factory=ActualPeft)
    assert result["passed"]


def p1_certificate(root, config):
    import importlib.metadata

    from src.core import file_hash
    from src.frozen_runtime import DEPENDENCIES, P1_CHECKS

    root.mkdir()
    freeze = "".join(f"{name}=={importlib.metadata.version(name)}\n" for name in DEPENDENCIES)
    (root / "pip-freeze.txt").write_text(freeze)
    model = config["models"]["qwen35_9b"]
    lock = {
        "model_revision": model["revision"],
        "resolved_config": {**config, "_model_key": "qwen35_9b"},
        "pip_freeze_sha256": file_hash(root / "pip-freeze.txt"),
        "runtime_status": "CANDIDATE_UNTIL_P1_PASS",
    }
    audit = {
        "passed": True,
        "execution_kind": "REAL_CUDA_MODEL",
        "model_id": model["id"],
        "model_revision": model["revision"],
        "checks": {k: True for k in P1_CHECKS},
        "runtime_lock": lock,
    }
    write_json(root / "model_audit.json", audit)
    write_json(root / "status.json", {"phase": "P1", "status": "PASS"})
    write_json(
        root / "manifest.json",
        {"files": [{"path": "model_audit.json", "sha256": file_hash(root / "model_audit.json")}]},
    )
    write_json(
        root / "validated_runtime_lock.json",
        {**lock, "runtime_status": "P1_PASSED", "execution_kind": "REAL_CUDA_MODEL"},
    )


def test_p1_certificate_accepts_exact_model_and_rejects_dependency_or_file_drift(tmp_path):
    from src.frozen_runtime import validate_p1_evidence

    config = load_config()
    root = tmp_path / "P1"
    p1_certificate(root, config)
    gate = validate_p1_evidence(root, config, "qwen35_9b")
    assert gate["audit_sha256"]
    with pytest.raises(ValueError, match="model/revision"):
        validate_p1_evidence(root, config, "qwen25vl_3b")
    with (
        patch("src.frozen_runtime.importlib.metadata.version", return_value="drift"),
        pytest.raises(ValueError, match="dependency drift"),
    ):
        validate_p1_evidence(root, config, "qwen35_9b")
    (root / "model_audit.json").write_text((root / "model_audit.json").read_text() + " ")
    with pytest.raises(ValueError, match="manifest hash"):
        validate_p1_evidence(root, config, "qwen35_9b")


def test_human_review_binds_exact_images_and_calibration(tmp_path):
    from src.core import file_hash
    from src.frozen_runtime import validate_human_review

    scenes = panel(tmp_path)
    selected = scenes + [{**s, "base_scene_id": s["base_scene_id"] + "-second"} for s in scenes]
    (tmp_path / "calibration.jsonl").write_text("calibration fixture")
    manifest = {
        "sheets": [{"scenes": selected}],
        "source_calibration_sha256": file_hash(tmp_path / "calibration.jsonl"),
    }
    path = tmp_path / "contact.json"
    write_json(path, manifest)
    review = {
        "status": "PASS",
        "reviewer": "Human fixture",
        "reviewed_at": "2026-09-08",
        "contact_manifest_sha256": file_hash(path),
        "checked_scene_ids": [s["base_scene_id"] for s in selected],
    }
    review_path = tmp_path / "review.json"
    write_json(review_path, review)
    assert validate_human_review(review_path, path, tmp_path)["image_count"] == 36
    write_json(review_path, {**review, "checked_scene_ids": review["checked_scene_ids"][:-1]})
    with pytest.raises(PermissionError, match="exact 36"):
        validate_human_review(review_path, path, tmp_path)
    write_json(review_path, review)
    (tmp_path / "calibration.jsonl").write_text("drift")
    with pytest.raises(ValueError, match="calibration"):
        validate_human_review(review_path, path, tmp_path)


def test_legacy_runtime_keeps_prompt_parser_and_length(tmp_path):
    from src.frozen_runtime import run_frozen
    from src.legacy_frozen import REPOSITORY, load_legacy_scenes

    config, _ = fixture_inputs(tmp_path)
    root = REPOSITORY / "artifacts/v5/study_c2/data"
    scenes = load_legacy_scenes(root / "reward_fibers.jsonl", root / "reward_fibers_manifest.json")[
        :2
    ]

    class LegacyFake(FrozenFake):
        def prepare(self, prompt, root):
            assert prompt["system"] is None
            assert prompt["messages"] == [{"role": "user", "content": prompt["user"]}]
            return super().prepare(prompt, root)

        def generate(self, prepared, **kwargs):
            assert kwargs["max_new_tokens"] == 48
            return super().generate(prepared, **kwargs)

    result = run_frozen(
        config, scenes, tmp_path / "L", "qwen35_9b", track="L", _adapter_factory=LegacyFake
    )
    assert result["passed"] and result["raw_sample_count"] == 34
    metrics = json.loads((tmp_path / "L/metrics.json").read_text())
    assert metrics["track"] == "L"
    assert metrics["overall"]["pooled"]["pI"] == 1.0  # Fake array outside legacy domain.


def test_same_output_rejects_concurrent_writer_and_releases_after_error(tmp_path):
    from src.frozen_runtime import frozen_writer

    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="interruption"), frozen_writer(out):
        with pytest.raises(RuntimeError, match="writer"), frozen_writer(out):
            pytest.fail("second writer acquired the lock")
        raise RuntimeError("interruption")
    with frozen_writer(out):
        assert (out / ".writer.lock").exists()


def test_cli_concurrent_fresh_run_rejection_preserves_other_writer_evidence(tmp_path):
    from src import rollout
    from src.core import canonical_hash

    out = tmp_path / "shared"
    original = {"phase": "P3", "status": "PASS", "owner": "other process"}

    def competing_run(*args, **kwargs):
        write_json(out / "status.json", original)
        write_json(out / "identity.json", {"hash": canonical_hash(original)})
        raise FileExistsError("Another frozen writer already owns this output")

    with (
        patch("src.frozen_runtime.validate_p1_evidence", return_value={}),
        patch("src.frozen_runtime.load_frozen_dev", return_value=[]),
        patch("src.frozen_runtime.run_frozen", side_effect=competing_run),
    ):
        assert (
            rollout.main(
                [
                    "--phase",
                    "frozen",
                    "--model",
                    "qwen35_9b",
                    "--config",
                    "configs/frozen.json",
                    "--out",
                    str(out),
                    "--p1-run-dir",
                    str(tmp_path / "P1"),
                ]
            )
            != 0
        )
    assert json.loads((out / "status.json").read_text()) == original
    assert not (out / "failures.jsonl").exists()


def test_wrong_model_key_spec_rejected_before_loader(tmp_path):
    from src.frozen_runtime import run_frozen

    config, scenes = fixture_inputs(tmp_path)
    config["models"]["qwen25vl_7b"] = {
        **config["models"]["qwen25vl_3b"],
        "revision": "a" * 40,
    }
    with patch("src.model_adapters.load_adapter") as loader:
        with pytest.raises(ValueError, match="official model"):
            run_frozen(
                config, scenes, tmp_path / "badmodel", "qwen25vl_7b", _adapter_factory=FrozenFake
            )
        loader.assert_not_called()
