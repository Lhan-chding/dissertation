"""Recursive recovery keeps each execution's real identity and historical stop."""

import json
import shutil
from pathlib import Path

import pytest
from test_r4_continuation_gate import (  # noqa: F401
    _build_continued_evidence,
    continued_evidence,
    stopped_evidence,
)
from test_r4_gate import _seal, _write_rows, complete_evidence  # noqa: F401

from src.core import canonical_hash, file_hash, write_json


def _read(path):
    return json.loads(Path(path).read_text())


@pytest.fixture
def stopped_continuation(continued_evidence):  # noqa: F811
    from src.optimizer_fork import load_checkpoint
    from src.r4_gate import _probe
    from src.r4_metrics import control_kl_diagnostic
    from src.r4_report_gate import _curves
    from src.r4_runtime import _write_csv

    root, gate, r2, r3 = continued_evidence
    for path in (root / "X_BASE").iterdir():
        if int(path.name.removeprefix("step_")) > 51:
            shutil.rmtree(path)
    for name in ("X_VALID", "evaluation/X_VALID", "evaluation/X_BASE/step_64"):
        shutil.rmtree(root / name)
    manifest = _read(root / "checkpoint_manifest.json")
    manifest["checkpoints"] = [
        e for e in manifest["checkpoints"] if e["arm"] == "X_BASE" and e["step"] <= 51
    ]
    manifest["X_BASE_step64_for_R3_warm"] = None
    write_json(root / "checkpoint_manifest.json", manifest)
    runtime = _read(root / "runtime_lock.json")
    origin = load_checkpoint(root / "inherited_parent/origin.pt", runtime["origin_identity"])
    probe = _probe(_read(root / "two_arm_training_config.json"), r3, origin)
    attempt = root / "X_BASE/step_51/updates/attempt_0000"
    score_path = attempt / "control_scores/samples.jsonl"
    scores = list(map(json.loads, score_path.read_text().splitlines()))
    for row in scores:
        row["new_token_logprobs"] = [v - 0.6 for v in row["new_token_logprobs"]]
    _write_rows(score_path, scores)
    diagnostic = control_kl_diagnostic(
        probe,
        {r["proposal_sample_key"]: r["new_token_logprobs"] for r in scores},
        eos_token_ids=[0],
    )
    assert diagnostic["alarms"] == {"mean_token_kl": True, "sequence_log_ratio_p99_abs": True}
    result = _read(attempt / "result.json")
    result.update(status="DIAGNOSTIC_STOP", control_diagnostic=diagnostic)
    write_json(attempt / "result.json", result)
    write_json(attempt / "control_diagnostic.json", diagnostic)
    (attempt / "completed.json").unlink()
    _seal(attempt)
    write_json(
        attempt / "completed.json",
        {
            "status": "PASS",
            "identity_hash": canonical_hash(_read(attempt / "identity.json")),
            "manifest_sha256": file_hash(attempt / "manifest.json"),
        },
    )
    write_json(
        root / "alarm_stop.json",
        {**result, "attempt": str(attempt), "reused_completed_unit": False},
    )
    cost = _read(root / "runtime_profile.json")
    cost["invocations"][0]["observed"].update(
        optimizer_step_calls_observed=16, backward_calls_observed=512
    )
    cost.update(optimizer_steps_observed_at_least=51, backward_calls_observed_at_least=1632)
    write_json(root / "runtime_profile.json", cost)
    write_json(
        root / "invocations/attempt_0000/runtime_profile.json", cost["invocations"][0]["observed"]
    )
    write_json(
        root / "invocations/attempt_0000/completion.json",
        {
            "status": "BLOCKED",
            "details": {"final_scratch_origin_restored": True},
        },
    )
    status = _read(root / "status.json")
    status.update(
        status="BLOCKED",
        details={
            "training_started": True,
            "final_scratch_origin_restored": True,
            "runtime_counts": {k: v for k, v in cost.items() if k != "invocations"},
        },
    )
    write_json(root / "status.json", status)
    summaries = []
    for step in range(1, 52):
        segment = root / "inherited_parent" if step <= 35 else root
        summary = _read(segment / f"X_BASE/step_{step:02d}/updates/attempt_0000/result.json")
        summary["source_segment"] = "parent" if step <= 35 else "current"
        summaries.append(summary)
    _write_csv(root / "learning_curves.csv", _curves(summaries))
    # The real writer preserves its terminal stop in the warning history as well.
    warnings = _read(root / "diagnostic_warnings.json")
    warnings.append(
        {
            "arm": "X_BASE",
            "step": 51,
            "status": "DIAGNOSTIC_STOP",
            "source_segment": "current",
            "control_diagnostic": diagnostic,
            "continuation_hash": runtime["continuation"]["continuation_hash"],
        }
    )
    write_json(root / "diagnostic_warnings.json", warnings)
    _seal(root)
    return root, gate, r2, r3


def test_stopped_continuation_preserves_owner_status_identity_and_combined_cost(
    stopped_continuation,
):
    from src.r4_gate import audit_stopped_r4, validate_r4_gate

    root, gate, r2, r3 = stopped_continuation
    before = {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()}
    audit = audit_stopped_r4(root, gate, r2, r3)
    runtime = _read(root / "runtime_lock.json")
    assert audit["continuation"] == runtime["continuation"]
    assert audit["logical_sampling_identity"] == _read(root / "logical_sampling_identity.json")
    assert audit["logical_sampling_identity"] != audit["identity"]
    assert audit["stop"]["step"] == 51
    assert audit["stop"]["diagnostic"]["alarms"]["mean_token_kl"] is True
    assert audit["inherited_counts"]["optimizer_updates"] == 51
    assert audit["inherited_counts"]["new_outputs"] == 2784
    with pytest.raises(ValueError):
        validate_r4_gate(root, gate, r2, r3)
    assert before == {
        str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "fault",
    [
        "hidden_warning",
        "missing_terminal_stop",
        "relabeled_terminal_stop",
        "duplicated_cost",
        "ancestor_status",
        "logical_identity",
    ],
)
def test_stopped_continuation_rejects_resealed_recursive_faults(stopped_continuation, fault):
    from src.r4_gate import audit_stopped_r4

    root, gate, r2, r3 = stopped_continuation
    if fault == "hidden_warning":
        write_json(root / "diagnostic_warnings.json", [])
    elif fault in ("missing_terminal_stop", "relabeled_terminal_stop"):
        warnings = _read(root / "diagnostic_warnings.json")
        if fault == "missing_terminal_stop":
            warnings.pop()
        else:
            warnings[-1]["status"] = "PASS"
        write_json(root / "diagnostic_warnings.json", warnings)
    elif fault == "duplicated_cost":
        cost = _read(root / "runtime_profile.json")
        cost["optimizer_steps_observed_at_least"] += 35
        write_json(root / "runtime_profile.json", cost)
    elif fault == "logical_identity":
        write_json(
            root / "logical_sampling_identity.json", _read(root / "runtime_lock.json")["identity"]
        )
    else:
        path = root / "inherited_parent/X_BASE/step_35/updates/attempt_0000/result.json"
        result = _read(path)
        result["status"] = "DIAGNOSTIC_WARNING"
        write_json(path, result)
        _seal(root / "inherited_parent")
    _seal(root)
    with pytest.raises(ValueError):
        audit_stopped_r4(root, gate, r2, r3)


@pytest.fixture
def recursive_complete(stopped_continuation, complete_evidence, tmp_path):  # noqa: F811
    return _build_continued_evidence(
        stopped_continuation, complete_evidence[0], tmp_path / "second_recovery", schema_version=2
    )


def test_recursive_completion_checks_all_three_sources_and_retains_two_real_stops(
    recursive_complete,
):
    from src.r4_gate import validate_r4_gate

    root, gate, r2, r3 = recursive_complete
    binding = validate_r4_gate(root, gate, r2, r3)
    assert binding["completion_status"] == "COMPLETED_WITH_DIAGNOSTIC_WARNINGS"
    assert binding["source_commit"] == "3" * 40
    assert binding["warm_checkpoint"]["step"] == 64
    parent = binding["continuation"]["parent_binding"]
    assert parent["stop"]["step"] == 51
    assert parent["continuation"]["parent_binding"]["stop"]["step"] == 35
    assert (
        binding["continuation"]["logical_sampling_identity"] == parent["logical_sampling_identity"]
    )
    warnings = _read(root / "diagnostic_warnings.json")
    assert [(w["arm"], w["step"], w["status"], w["source_segment"]) for w in warnings] == [
        ("X_BASE", 35, "DIAGNOSTIC_STOP", "parent"),
        ("X_BASE", 36, "DIAGNOSTIC_WARNING", "parent"),
        ("X_BASE", 51, "DIAGNOSTIC_STOP", "parent"),
        ("X_BASE", 52, "DIAGNOSTIC_WARNING", "current"),
        ("X_VALID", 1, "DIAGNOSTIC_WARNING", "current"),
    ]
    assert all(w["control_diagnostic"]["alarms"]["mean_token_kl"] for w in warnings[-2:])
    cost = _read(root / "runtime_profile.json")
    assert cost["optimizer_steps_observed_at_least"] == 128
    assert cost["invocations"][0]["observed"]["optimizer_step_calls_observed"] == 77
    assert cost["inherited_runtime_counts"]["optimizer_steps_observed_at_least"] == 51


@pytest.mark.parametrize(
    "fault", ["actual_identity", "relabeled_parent_stop", "wrong_owner", "hidden_current_warning"]
)
def test_recursive_completion_rejects_resealed_identity_and_warning_forgery(
    recursive_complete, fault
):
    from src.r4_gate import validate_r4_gate

    root, gate, r2, r3 = recursive_complete
    if fault == "actual_identity":
        path = root / "X_BASE/step_52/rollouts/samples.jsonl"
        rows = list(map(json.loads, path.read_text().splitlines()))
        rows[0]["execution_identity_hash"] = canonical_hash(
            _read(root / "inherited_parent/runtime_lock.json")["identity"]
        )
        _write_rows(path, rows)
    elif fault in ("relabeled_parent_stop", "hidden_current_warning"):
        warnings = _read(root / "diagnostic_warnings.json")
        if fault == "relabeled_parent_stop":
            warnings[2]["status"] = "DIAGNOSTIC_WARNING"
        else:
            warnings.pop()
        write_json(root / "diagnostic_warnings.json", warnings)
    else:
        manifest = _read(root / "checkpoint_manifest.json")
        entry = next(e for e in manifest["checkpoints"] if (e["arm"], e["step"]) == ("X_BASE", 36))
        entry["checkpoint_path"] = (
            "inherited_parent/inherited_parent/X_BASE/step_36/updates/attempt_0000/checkpoint.pt"
        )
        write_json(root / "checkpoint_manifest.json", manifest)
    _seal(root)
    with pytest.raises(ValueError):
        validate_r4_gate(root, gate, r2, r3)
