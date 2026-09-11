"""Audited continuation preserves measured stops and immutable parent evidence."""

import json
import shutil
from pathlib import Path

import pytest
from test_r4_gate import _seal, _write_rows, complete_evidence  # noqa: F401

from src.core import canonical_hash, file_hash, write_json


def _read(path):
    return json.loads(Path(path).read_text())


@pytest.fixture
def stopped_evidence(complete_evidence, tmp_path):  # noqa: F811
    from src import r4_gate
    from src.optimizer_fork import load_checkpoint
    from src.r4_metrics import control_kl_diagnostic

    original, gate, r2, r3, _ = complete_evidence
    root = tmp_path / "stopped"
    shutil.copytree(original, root)
    for path in (root / "X_BASE").iterdir():
        if int(path.name.removeprefix("step_")) > 35:
            shutil.rmtree(path)
    for name in ("X_VALID", "evaluation/X_VALID", "evaluation/X_BASE/step_64"):
        shutil.rmtree(root / name)
    runtime = _read(root / "runtime_lock.json")
    write_json(root / "identity.json", runtime["identity"])
    checkpoints = _read(root / "checkpoint_manifest.json")
    checkpoints["checkpoints"] = [
        item
        for item in checkpoints["checkpoints"]
        if item["arm"] == "X_BASE" and item["step"] <= 35
    ]
    for item in checkpoints["checkpoints"]:
        item["checkpoint_path"] = str(root / Path(item["checkpoint_path"]).relative_to(original))
    checkpoints["X_BASE_step64_for_R3_warm"] = None
    write_json(root / "checkpoint_manifest.json", checkpoints)
    origin = load_checkpoint(root / "origin.pt", {**runtime["identity"], "unit": "initial_origin"})
    probe = r4_gate._probe(_read(root / "two_arm_training_config.json"), r3, origin)
    attempt = root / "X_BASE/step_35/updates/attempt_0000"
    score_path = attempt / "control_scores/samples.jsonl"
    scores = list(map(json.loads, score_path.read_text().splitlines()))
    for row in scores[:2]:
        row["new_token_logprobs"] = [v - 0.6 for v in row["new_token_logprobs"]]
    _write_rows(score_path, scores)
    candidates = {row["proposal_sample_key"]: row["new_token_logprobs"] for row in scores}
    diagnostic = control_kl_diagnostic(probe, candidates, eos_token_ids=[0])
    assert diagnostic["alarms"] == {"mean_token_kl": False, "sequence_log_ratio_p99_abs": True}
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
    observed = cost["invocations"][0]["observed"]
    observed.update(optimizer_step_calls_observed=35, backward_calls_observed=1120)
    cost.update(optimizer_steps_observed_at_least=35, backward_calls_observed_at_least=1120)
    write_json(root / "runtime_profile.json", cost)
    write_json(root / "invocations/attempt_0000/runtime_profile.json", observed)
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
    from src.r4_report_gate import _curves
    from src.r4_runtime import _write_csv

    summaries = [
        _read(root / "X_BASE" / f"step_{step:02d}/updates/attempt_0000/result.json")
        for step in range(1, 36)
    ]
    _write_csv(root / "learning_curves.csv", _curves(summaries))
    _seal(root)
    return root, gate, r2, r3


def test_stopped_audit_preserves_warning_and_cannot_be_used_as_warm_pass(stopped_evidence):
    from src.r4_gate import audit_stopped_r4, validate_r4_gate

    root, gate, r2, r3 = stopped_evidence
    before = {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()}
    audit = audit_stopped_r4(root, gate, r2, r3)
    assert audit["status"] == "PASS_STOPPED_AUDIT"
    assert (audit["stop"]["arm"], audit["stop"]["step"]) == ("X_BASE", 35)
    assert audit["stop"]["diagnostic"]["should_stop"] is True
    assert audit["inherited_counts"]["new_outputs"] == 2272
    assert audit["inherited_counts"]["optimizer_updates"] == 35
    assert "warm_checkpoint" not in audit
    assert audit["audit_hash"] == canonical_hash(
        {k: v for k, v in audit.items() if k not in ("root", "audit_hash")}
    )
    with pytest.raises(ValueError):
        validate_r4_gate(root, gate, r2, r3)
    assert before == {
        str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()
    }


def test_stopped_audit_relocates_snapshot_without_rewriting_original_paths(
    stopped_evidence, tmp_path
):
    from src.r4_gate import audit_stopped_r4

    root, gate, r2, r3 = stopped_evidence
    original = audit_stopped_r4(root, gate, r2, r3)
    snapshot = tmp_path / "inherited_parent"
    shutil.copytree(root, snapshot)
    assert audit_stopped_r4(snapshot, gate, r2, r3, _recorded_root=root) == original


@pytest.mark.parametrize(
    "fault", ["fault_marker", "missing_stop", "source", "checkpoint", "old_logp"]
)
def test_stopped_audit_rejects_resealed_faults(stopped_evidence, fault):
    from src.r4_gate import audit_stopped_r4

    root, gate, r2, r3 = stopped_evidence
    if fault == "fault_marker":
        write_json(root / "measurement_fault.json", {"reason": "test failure"})
    elif fault == "missing_stop":
        (root / "alarm_stop.json").unlink()
    elif fault == "source":
        runtime = _read(root / "runtime_lock.json")
        runtime["source"]["source_commit"] = "unavailable"
        write_json(root / "runtime_lock.json", runtime)
    elif fault == "checkpoint":
        (root / "X_BASE/step_35/updates/attempt_0000/checkpoint.pt").write_bytes(b"bad checkpoint")
    else:
        path = root / "X_BASE/step_35/rollouts/samples.jsonl"
        rows = list(map(json.loads, path.read_text().splitlines()))
        rows[0]["old_logprobs"][0] -= 1
        _write_rows(path, rows)
    _seal(root)
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        audit_stopped_r4(root, gate, r2, r3)


def test_report_steps_preserve_reviewed_warnings_but_reject_unreviewed_mean_alarm():
    from src.r4_report_gate import _steps

    policy = {"reviewed_warning_policy": "sequence_p99_only_two_arms_to_step64"}
    summaries = []
    for arm in ("X_BASE", "X_VALID"):
        for step in range(1, 65):
            summaries.append(
                {
                    "arm": arm,
                    "step": step,
                    "status": "PASS",
                    "source_segment": "current",
                    "training_category_counts": {"X": 32, "S": 0, "W": 0, "I": 0},
                    "zero_advantage_groups": 4,
                    "control_diagnostic": {
                        "mean_token_kl": 0.01,
                        "sequence_log_ratio_p99_abs": 0.0,
                        "status": "WITHIN_ENGINEERING_LIMITS",
                        "should_stop": False,
                        "alarms": {"mean_token_kl": False, "sequence_log_ratio_p99_abs": False},
                    },
                    "update": {"loss": 0.0, "grad_norm_preclip": 0.0, "actual_step_norm": 0.0},
                }
            )
    warning = summaries[34]
    warning.update(status="DIAGNOSTIC_STOP", source_segment="parent")
    warning["control_diagnostic"].update(
        status="STOP_DIAGNOSE",
        should_stop=True,
        sequence_log_ratio_p99_abs=2.1,
        alarms={"mean_token_kl": False, "sequence_log_ratio_p99_abs": True},
    )
    with pytest.raises(ValueError):
        _steps(summaries)
    assert _steps(summaries, continuation=policy)[34]["status"] == "DIAGNOSTIC_STOP"
    warning["control_diagnostic"].update(
        mean_token_kl=0.10001, alarms={"mean_token_kl": True, "sequence_log_ratio_p99_abs": True}
    )
    with pytest.raises(ValueError):
        _steps(summaries, continuation=policy)


@pytest.fixture
def continued_evidence(stopped_evidence, complete_evidence, tmp_path):  # noqa: F811
    import copy

    from src import r4_gate
    from src.optimizer_fork import load_checkpoint, save_checkpoint
    from src.r4_continuation import REVIEWED_POLICY, prepare_continuation
    from src.r4_metrics import control_kl_diagnostic

    parent, gate, r2, r3 = stopped_evidence
    complete = complete_evidence[0]
    audit = r4_gate.audit_stopped_r4(parent, gate, r2, r3)
    source = {**audit["source"], "source_commit": "2" * 40}
    root = tmp_path / "continued"
    decision = {
        "schema_version": 1,
        "authorization": {
            "user_request": "Resume after diagnosis",
            "reason": "Sequence alarm reviewed",
        },
        "parent_audit_hash": audit["audit_hash"],
        "execution_source_hash": canonical_hash(source),
        "policy": copy.deepcopy(REVIEWED_POLICY),
    }
    ctx = prepare_continuation(parent, root, decision, audit, source, allow_training=True)
    contract = ctx["runtime_binding"]
    runtime = _read(complete / "runtime_lock.json")
    old_identity = runtime["identity"]
    identity = {
        **old_identity,
        "source_hash": canonical_hash(source),
        "continuation_hash": ctx["continuation_hash"],
    }
    runtime.update(
        identity=identity,
        source=source,
        continuation=contract,
        origin_identity={**old_identity, "unit": "initial_origin"},
    )
    write_json(root / "runtime_lock.json", runtime)
    write_json(root / "identity.json", identity)
    for name in (
        "gate_binding.json",
        "two_arm_training_config.json",
        "initial_alignment.json",
        "step32_metrics.json",
        "endpoint_metrics.json",
        "L_family_equal_sensitivity.json",
        "N_L_OOD_effects.csv",
        "learning_curves.csv",
        "pilot_report.md",
        "report_zh.md",
    ):
        shutil.copy2(complete / name, root / name)
    for arm, start in (("X_BASE", 36), ("X_VALID", 1)):
        for step in range(start, 65):
            relative = Path(arm) / f"step_{step:02d}"
            shutil.copytree(complete / relative, root / relative)
        for step in (64,) if arm == "X_BASE" else (32, 64):
            relative = Path("evaluation") / arm / f"step_{step:02d}"
            shutil.copytree(complete / relative, root / relative)
    for path in root.rglob("samples.jsonl"):
        if path.is_relative_to(root / "inherited_parent"):
            continue
        rows = list(map(json.loads, path.read_text().splitlines()))
        if "raw_completion" not in rows[0]:
            continue
        for row in rows:
            row["execution_identity_hash"] = canonical_hash(identity)
        _write_rows(path, rows)
        ledger_path = path.parent / "identity.json"
        ledger = _read(ledger_path)
        ledger.update(identity)
        write_json(ledger_path, ledger)
    entries = []
    source_manifest = _read(complete / "checkpoint_manifest.json")
    parent_mapping = {(e["arm"], e["step"]): e for e in audit["checkpoint_manifest"]["checkpoints"]}
    origin = load_checkpoint(root / "inherited_parent/origin.pt", runtime["origin_identity"])
    probe = r4_gate._probe(_read(root / "two_arm_training_config.json"), r3, origin)
    new_warning = None
    for original_entry in source_manifest["checkpoints"]:
        arm, step = original_entry["arm"], original_entry["step"]
        if step == 0:
            entries.append(
                {
                    **original_entry,
                    "checkpoint_path": "inherited_parent/origin.pt",
                    "source_segment": "parent",
                }
            )
            continue
        if (arm, step) in parent_mapping:
            entry = parent_mapping[(arm, step)]
            relative = Path(entry["checkpoint_path"]).relative_to(parent)
            entries.append(
                {
                    **entry,
                    "checkpoint_path": str(Path("inherited_parent") / relative),
                    "source_segment": "parent",
                }
            )
            continue
        relative = Path(arm) / f"step_{step:02d}/updates/attempt_0000"
        attempt = root / relative
        rows = list(
            map(
                json.loads,
                (root / arm / f"step_{step:02d}/rollouts/samples.jsonl").read_text().splitlines(),
            )
        )
        unit = {
            **_read(attempt / "identity.json"),
            **identity,
            "sample_hash": canonical_hash([r["record_hash"] for r in rows]),
        }
        checkpoint_identity = {**unit, "unit": "post_update_checkpoint"}
        post = load_checkpoint(attempt / "checkpoint.pt", original_entry["checkpoint_identity"])
        save_checkpoint(attempt / "checkpoint.pt", post, checkpoint_identity)
        write_json(attempt / "identity.json", unit)
        result = _read(attempt / "result.json")
        result.update(
            checkpoint_identity=checkpoint_identity,
            checkpoint_sha256=file_hash(attempt / "checkpoint.pt"),
        )
        for directory, inputs in (("preupdate_parity", rows), ("control_scores", probe)):
            old_store = _read(attempt / directory / "identity.json")
            store = {**old_store, **unit, "unit": old_store["unit"]}
            write_json(attempt / directory / "identity.json", store)
            measured = list(
                map(json.loads, (attempt / directory / "samples.jsonl").read_text().splitlines())
            )
            source_rows = {r["sample_key"]: r for r in inputs}
            for record in measured:
                source_row = source_rows[record["proposal_sample_key"]]
                record.update(
                    sample_key=canonical_hash([store, source_row["sample_key"]]),
                    proposal_record_hash=source_row["record_hash"],
                )
            if arm == "X_BASE" and step == 36 and directory == "control_scores":
                for record in measured[:2]:
                    record["new_token_logprobs"] = [v - 0.6 for v in record["new_token_logprobs"]]
                diagnostic = control_kl_diagnostic(
                    probe,
                    {r["proposal_sample_key"]: r["new_token_logprobs"] for r in measured},
                    eos_token_ids=[0],
                )
                result.update(status="DIAGNOSTIC_WARNING", control_diagnostic=diagnostic)
                write_json(attempt / "control_diagnostic.json", diagnostic)
                write_json(
                    attempt / "reviewed_warning.json",
                    {
                        "continuation_hash": ctx["continuation_hash"],
                        "reviewed_warning_policy": contract["reviewed_warning_policy"],
                        "arm": arm,
                        "step": step,
                        "checkpoint_sha256": result["checkpoint_sha256"],
                        "control_diagnostic": diagnostic,
                    },
                )
                new_warning = {
                    "arm": arm,
                    "step": step,
                    "status": result["status"],
                    "control_diagnostic": diagnostic,
                    "source_segment": "current",
                    "continuation_hash": ctx["continuation_hash"],
                }
            _write_rows(attempt / directory / "samples.jsonl", measured)
        write_json(attempt / "result.json", result)
        (attempt / "completed.json").unlink()
        _seal(attempt)
        write_json(
            attempt / "completed.json",
            {
                "status": "PASS",
                "identity_hash": canonical_hash(unit),
                "manifest_sha256": file_hash(attempt / "manifest.json"),
            },
        )
        entries.append(
            {
                **original_entry,
                "checkpoint_identity": checkpoint_identity,
                "checkpoint_sha256": result["checkpoint_sha256"],
                "checkpoint_path": str(relative / "checkpoint.pt"),
                "source_segment": "current",
            }
        )
    write_json(
        root / "checkpoint_manifest.json",
        {
            "identity": identity,
            "checkpoints": entries,
            "X_BASE_step64_for_R3_warm": next(
                e for e in entries if e["arm"] == "X_BASE" and e["step"] == 64
            ),
        },
    )
    warnings = [
        {
            "arm": "X_BASE",
            "step": 35,
            "status": "DIAGNOSTIC_STOP",
            "control_diagnostic": audit["stop"]["diagnostic"],
            "source_segment": "parent",
            "continuation_hash": ctx["continuation_hash"],
        },
        new_warning,
    ]
    write_json(root / "diagnostic_warnings.json", warnings)
    shutil.copytree(complete / "invocations", root / "invocations")
    cost = _read(complete / "runtime_profile.json")
    cost["invocations"][0]["observed"].update(
        optimizer_step_calls_observed=93, backward_calls_observed=2976
    )
    inherited = _read(parent / "runtime_profile.json")
    cost["inherited_runtime_counts"] = {k: v for k, v in inherited.items() if k != "invocations"}
    write_json(
        root / "invocations/attempt_0000/runtime_profile.json", cost["invocations"][0]["observed"]
    )
    write_json(
        root / "invocations/attempt_0000/completion.json",
        {
            "status": "COMPLETED_WITH_DIAGNOSTIC_WARNINGS",
            "details": {"final_scratch_origin_restored": True},
        },
    )
    write_json(root / "runtime_profile.json", cost)
    status = _read(complete / "status.json")
    status.update(source, status="COMPLETED_WITH_DIAGNOSTIC_WARNINGS")
    status["details"]["runtime_counts"] = {k: v for k, v in cost.items() if k != "invocations"}
    write_json(root / "status.json", status)
    _seal(root)
    return root, gate, r2, r3


def test_complete_continuation_binds_both_sources_all_outputs_and_original_warning(
    continued_evidence,
):
    from src.r4_gate import validate_r4_gate

    root, gate, r2, r3 = continued_evidence
    binding = validate_r4_gate(root, gate, r2, r3)
    assert binding["completion_status"] == "COMPLETED_WITH_DIAGNOSTIC_WARNINGS"
    assert binding["source_commit"] == "2" * 40
    assert binding["warm_checkpoint"]["step"] == 64
    assert binding["warm_checkpoint"]["source_segment"] == "current"
    assert binding["continuation"]["parent_binding"]["stop"]["step"] == 35


@pytest.mark.parametrize(
    "fault",
    ["hidden_warning", "raw_execution_source", "segment", "parent_source", "inherited_cost"],
)
def test_complete_continuation_rejects_resealed_provenance_and_alarm_tampering(
    continued_evidence, fault
):
    from src.r4_gate import validate_r4_gate

    root, gate, r2, r3 = continued_evidence
    if fault == "hidden_warning":
        write_json(root / "diagnostic_warnings.json", [])
    elif fault == "raw_execution_source":
        path = root / "X_BASE/step_36/rollouts/samples.jsonl"
        rows = list(map(json.loads, path.read_text().splitlines()))
        rows[0]["execution_identity_hash"] = "0" * 64
        _write_rows(path, rows)
    elif fault == "segment":
        value = _read(root / "checkpoint_manifest.json")
        value["checkpoints"][35]["source_segment"] = "current"
        write_json(root / "checkpoint_manifest.json", value)
    elif fault == "parent_source":
        value = _read(root / "inherited_parent/runtime_lock.json")
        value["source"]["source_commit"] = "2" * 40
        write_json(root / "inherited_parent/runtime_lock.json", value)
        _seal(root / "inherited_parent")
    else:
        value = _read(root / "runtime_profile.json")
        value["inherited_runtime_counts"]["optimizer_steps_observed_at_least"] = 34
        write_json(root / "runtime_profile.json", value)
    _seal(root)
    with pytest.raises(ValueError):
        validate_r4_gate(root, gate, r2, r3)


def _diagnostic_pair():
    import copy

    stored = {
        "mean_token_kl": 0.019,
        "sequence_log_ratio_p99_abs": 2.079,
        "status": "STOP_DIAGNOSE",
        "should_stop": True,
        "alarms": {"mean_token_kl": False, "sequence_log_ratio_p99_abs": True},
        "thresholds": {"mean_token_kl": 0.1, "sequence_log_ratio_p99_abs": 2.0},
        "by_prompt": {"prompt": {"mean_token_kl": 0.019, "selected_token_count": 2}},
        "by_group": {"group": {"mean_token_kl": 0.019, "fixed_weight": 1 / 6}},
        "sequence_records": [
            {
                "token_k3_sum": 0.038,
                "mean_token_kl": 0.019,
                "sequence_log_ratio_candidate_minus_step0": -2.079,
                "sample_key": "immutable",
                "selected_token_count": 2,
            }
        ],
    }
    return stored, copy.deepcopy(stored)


def test_kl_comparator_allows_only_expm1_derived_roundoff_without_mutation():
    import copy
    import math

    from src.r4_gate import _compare_kl_diagnostic

    stored, computed = _diagnostic_pair()
    computed["mean_token_kl"] = math.nextafter(stored["mean_token_kl"], math.inf)
    computed["by_prompt"]["prompt"]["mean_token_kl"] = computed["mean_token_kl"]
    before = copy.deepcopy((stored, computed))
    differences = _compare_kl_diagnostic(stored, computed)
    assert len(differences) == 2
    assert max(d["absolute_difference"] for d in differences) < 1e-17
    assert (stored, computed) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "p99",
        "sequence_ratio",
        "weight",
        "count_type",
        "threshold",
        "alarm",
        "status",
        "missing",
        "large",
        "nonfinite",
    ],
)
def test_kl_comparator_rejects_non_allowlisted_changes(mutation):
    import math

    from src.r4_gate import _compare_kl_diagnostic

    stored, computed = _diagnostic_pair()
    if mutation == "p99":
        computed["sequence_log_ratio_p99_abs"] = math.nextafter(2.079, math.inf)
    elif mutation == "sequence_ratio":
        computed["sequence_records"][0]["sequence_log_ratio_candidate_minus_step0"] = (
            math.nextafter(-2.079, math.inf)
        )
    elif mutation == "weight":
        computed["by_group"]["group"]["fixed_weight"] = math.nextafter(1 / 6, math.inf)
    elif mutation == "count_type":
        computed["by_prompt"]["prompt"]["selected_token_count"] = 2.0
    elif mutation == "threshold":
        computed["thresholds"]["mean_token_kl"] = math.nextafter(0.1, math.inf)
    elif mutation == "alarm":
        computed["alarms"]["mean_token_kl"] = True
    elif mutation == "status":
        computed["status"] = "WITHIN_ENGINEERING_LIMITS"
    elif mutation == "missing":
        del computed["by_prompt"]["prompt"]["selected_token_count"]
    elif mutation == "large":
        computed["mean_token_kl"] += 1e-12
    else:
        computed["mean_token_kl"] = math.inf
    with pytest.raises(ValueError):
        _compare_kl_diagnostic(stored, computed)


@pytest.mark.parametrize("flip_recorded_alarm", [False, True])
def test_kl_roundoff_cannot_waive_a_mean_threshold_crossing(flip_recorded_alarm):
    import math

    from src.r4_gate import _compare_kl_diagnostic

    stored, computed = _diagnostic_pair()
    stored["mean_token_kl"] = 0.1
    computed["mean_token_kl"] = math.nextafter(0.1, math.inf)
    if flip_recorded_alarm:
        computed["alarms"]["mean_token_kl"] = True
    with pytest.raises(ValueError):
        _compare_kl_diagnostic(stored, computed)


def test_parent_audit_identity_uses_saved_values_when_cpu_kl_rounding_differs(
    stopped_evidence, monkeypatch
):
    import math

    from src import r4_metrics
    from src.r4_gate import audit_stopped_r4

    root, gate, r2, r3 = stopped_evidence
    before = audit_stopped_r4(root, gate, r2, r3)
    original = r4_metrics.control_kl_diagnostic

    def rounded(*args, **kwargs):
        result = original(*args, **kwargs)
        if result["mean_token_kl"]:
            result["mean_token_kl"] = math.nextafter(result["mean_token_kl"], math.inf)
        return result

    monkeypatch.setattr(r4_metrics, "control_kl_diagnostic", rounded)
    observations = []
    after = audit_stopped_r4(root, gate, r2, r3, roundoff_observer=observations.append)
    assert before == after
    assert observations and any(o["differences"] for o in observations)


def test_continuation_gate_identity_and_original_warning_are_cpu_roundoff_independent(
    continued_evidence, monkeypatch
):
    import math

    from src import r4_metrics
    from src.r4_gate import validate_r4_gate

    root, gate, r2, r3 = continued_evidence
    before = validate_r4_gate(root, gate, r2, r3)
    original = r4_metrics.control_kl_diagnostic

    def rounded(*args, **kwargs):
        result = original(*args, **kwargs)
        if result["mean_token_kl"]:
            result["mean_token_kl"] = math.nextafter(result["mean_token_kl"], math.inf)
        return result

    monkeypatch.setattr(r4_metrics, "control_kl_diagnostic", rounded)
    observations = []
    after = validate_r4_gate(root, gate, r2, r3, roundoff_observer=observations.append)
    assert before == after
    assert {o["step"] for o in observations} == {35, 36}
