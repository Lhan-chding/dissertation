"""CPU report recomputation from nondegenerate sampled R4 evidence."""

import copy
import csv
import importlib
import importlib.util
import json
import shutil

import pytest

from src.core import file_hash, write_json
from src.r3_runtime import _write_csv
from src.r4_metrics import analyze_sampled_endpoints
from src.r4_runtime import _reports

ARMS = ("X_BASE", "X_VALID")
FAMILIES = ("cross_series", "duplicate_encoding", "trend")
INTERFACES = ("IMAGE_CUE_FRESH", "SYMBOLIC_FRESH")


def test_r4_report_gate_api_exists():
    assert importlib.util.find_spec("src.r4_report_gate") is not None


def _rows(track, sizes, *, step=64, initial=False, k=8):
    rows = []
    interfaces = ("collision", "separating") if track == "L" else INTERFACES
    for family, count in sizes.items():
        for scene in range(count):
            for interface_index, interface in enumerate(interfaces):
                prompt = f"{track}-{family}-{scene}-{interface}"
                for arm in ("INITIAL",) if initial else ARMS:
                    for sample in range(k):
                        x = (scene + interface_index) % 5
                        valid = 4 + scene % 4
                        if arm == "X_VALID":
                            x += scene % 3 - 1
                            valid = min(8, valid + 1)
                        category = (
                            "X"
                            if sample % 8 < max(0, x)
                            else "S"
                            if sample % 8 < 5
                            else "W"
                            if sample % 8 < valid
                            else "I"
                        )
                        rows.append(
                            {
                                "track": track,
                                "arm": arm,
                                "checkpoint_step": 0 if initial else step,
                                "base_scene_id": f"{track}-{family}-{scene}",
                                "prompt_id": prompt,
                                "family": family,
                                "interface": interface,
                                "sample_key": f"{arm}-{step}-{prompt}-{sample}",
                                "sample_index": sample,
                                "decode_mode": "sample",
                                "category": category,
                            }
                        )
    return rows


def _curve(summary):
    return {
        "arm": summary["arm"],
        "step": summary["step"],
        "mean_token_kl": summary["control_diagnostic"]["mean_token_kl"],
        "sequence_log_ratio_p99_abs": summary["control_diagnostic"]["sequence_log_ratio_p99_abs"],
        "pX_training_bank": summary["training_category_counts"]["X"] / 32,
        "v_training_bank": 1 - summary["training_category_counts"]["I"] / 32,
        "zero_advantage_groups_retained": summary["zero_advantage_groups"],
        **summary["update"],
        "status": summary["status"],
    }


@pytest.fixture(scope="module")
def complete_reports(tmp_path_factory):
    root = tmp_path_factory.mktemp("r4_reports")
    endpoint = (
        _rows("N", dict.fromkeys(FAMILIES, 48))
        + _rows("L", {"legacy_a": 24, "legacy_b": 64})
        + _rows("OOD", {"cross_series": 36})
    )
    shared = _rows("N", dict.fromkeys(FAMILIES, 12), step=0, initial=True)
    middle = _rows("N", dict.fromkeys(FAMILIES, 12), step=32)
    summaries = [
        {
            "arm": arm,
            "step": step,
            "status": "PASS",
            "control_diagnostic": {
                "mean_token_kl": step / 10000,
                "sequence_log_ratio_p99_abs": step / 1000,
            },
            "training_category_counts": {"X": step % 9, "S": 8, "W": 8, "I": 16 - step % 9},
            "zero_advantage_groups": step % 3,
            "update": {
                "loss": -step / 100,
                "grad_norm_preclip": step / 100 + 1,
                "actual_step_norm": step / 100000,
            },
            "attempt": f"{arm}/step_{step:02d}/attempt_0",
            "reused_completed_unit": step % 2 == 0,
        }
        for arm in ARMS
        for step in range(1, 65)
    ]
    details = {
        "status": "PASS",
        "execution_kind": "REAL_CUDA_TRAINING",
        "training_started": True,
        "final_scratch_origin_restored": True,
        "distinct_optimizer_updates": 128,
        "training_rollouts": 4096,
        "shared_step0_outputs": 576,
        "step32_outputs": 1152,
        "step64_outputs": 8576,
        "new_outputs": 14400,
        "fixed_control_sequence_scores": 6144,
        "preupdate_parity_sequence_scores": 4096,
        "postupdate_training_sequence_scores": 4096,
        "new_control_outputs": 0,
        "arms": dict.fromkeys(ARMS, 64),
    }
    initial = {"N": shared}
    _reports(root, endpoint, initial, details)
    write_json(root / "step32_metrics.json", analyze_sampled_endpoints(middle, shared, track="N"))
    _write_csv(root / "learning_curves.csv", [_curve(s) for s in summaries])
    return root, {
        "endpoint_rows": endpoint,
        "step32_rows": middle,
        "shared_initial_rows": shared,
        "initial_rows_by_track": initial,
        "step_summaries": summaries,
    }


@pytest.fixture
def report_case(complete_reports, tmp_path):
    source, inputs = complete_reports
    root = tmp_path / "reports"
    shutil.copytree(source, root)
    return root, copy.deepcopy(inputs)


def _validate(case):
    root, inputs = case
    return importlib.import_module("src.r4_report_gate").validate_r4_response_artifacts(
        root, **inputs
    )


def test_real_recomputation_binds_read_only_complete_reports(report_case):
    root, inputs = report_case
    before = {p.name: file_hash(p) for p in root.iterdir()}
    original = copy.deepcopy(inputs)
    result = _validate(report_case)
    assert result["status"] == "PASS"
    assert result["execution_kind"] == "CPU_MATH"
    assert result["bootstrap"] == {"replicates": 5000, "seed": 20260909}
    assert result["counts"]["endpoint_outputs_by_track"] == {"N": 4608, "L": 2816, "OOD": 1152}
    assert result["counts"]["learning_curve_rows"] == 128
    assert result["safety_status"] == "NOT_CERTIFIED"
    assert result["artifact_sha256"]["endpoint_metrics.json"] == before["endpoint_metrics.json"]
    assert {p.name: file_hash(p) for p in root.iterdir()} == before
    assert inputs == original
    metrics = json.loads((root / "endpoint_metrics.json").read_text())
    interval = metrics["N"]["comparisons"]["X_VALID_minus_X_BASE"]["responses"]["overall"]["pX"][
        "pointwise_ci"
    ]
    assert interval["status"] == "ESTIMATED" and interval["half_width"] > 0
    primary = metrics["L"]["comparisons"]["X_VALID_minus_X_BASE"]["panel"]
    sensitivity = json.loads((root / "L_family_equal_sensitivity.json").read_text())
    assert (
        primary["fixed_group_weights"]
        != sensitivity["comparisons"]["X_VALID_minus_X_BASE"]["panel"]["fixed_group_weights"]
    )


@pytest.mark.parametrize(
    "artifact", ["endpoint_metrics.json", "step32_metrics.json", "L_family_equal_sensitivity.json"]
)
@pytest.mark.parametrize("mutation", ["placeholder", "estimate", "bootstrap", "denominator"])
def test_rejects_statistical_tampering(report_case, artifact, mutation):
    path = report_case[0] / artifact
    value = json.loads(path.read_text())
    result = value["N"] if artifact == "endpoint_metrics.json" else value
    comparison = result["comparisons"]["X_VALID_minus_X_BASE"]
    if mutation == "placeholder":
        value = {"status": "PASS"}
    elif mutation == "estimate":
        comparison["responses"]["overall"]["pX"]["estimate"] += 1e-12
    elif mutation == "bootstrap":
        result["bootstrap_contract"]["seed"] += 1
    else:
        first = next(iter(comparison["panel"]["target_counts_by_prompt"].values()))
        first["n"] = 16
    write_json(path, value)
    with pytest.raises(ValueError):
        _validate(report_case)


def _csv(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def _save_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("name", ["N_L_OOD_effects.csv", "learning_curves.csv"])
@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "wrong_value", "extra_column", "duplicate_column"]
)
def test_rejects_csv_coverage_and_value_forgery(report_case, name, mutation):
    path = report_case[0] / name
    fields, rows = _csv(path)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = rows[0].copy()
    elif mutation == "wrong_value":
        rows[0]["estimate" if name.startswith("N_") else "pX_training_bank"] = "0.123456789"
    elif mutation == "extra_column":
        fields.append("extra")
    else:
        fields.append(fields[0])
    _save_csv(path, fields, rows)
    with pytest.raises(ValueError):
        _validate(report_case)


def test_accepts_reordered_inputs_json_and_csv(report_case):
    root, inputs = report_case
    expected = _validate(report_case)
    for key in ("endpoint_rows", "step32_rows", "shared_initial_rows", "step_summaries"):
        inputs[key].reverse()
    inputs["initial_rows_by_track"]["N"].reverse()
    for name in ("N_L_OOD_effects.csv", "learning_curves.csv"):
        fields, rows = _csv(root / name)
        _save_csv(root / name, fields[::-1], rows[::-1])
    for name in ("endpoint_metrics.json", "step32_metrics.json"):
        path = root / name
        value = json.loads(path.read_text())
        path.write_text(json.dumps(dict(reversed(list(value.items())))))
    actual = _validate(report_case)
    assert actual["counts"] == expected["counts"]
    assert actual["input_sha256"] == expected["input_sha256"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "swapped_track",
        "unequal_k",
        "wrong_step",
        "initial_mismatch",
        "extra_initial",
        "missing_step",
        "duplicate_step",
    ],
)
def test_fixed_coverage_validation_precedes_statistics(report_case, mutation, monkeypatch):
    api = importlib.import_module("src.r4_report_gate")
    inputs = report_case[1]
    if mutation == "missing":
        inputs["endpoint_rows"].pop()
    elif mutation == "duplicate":
        inputs["endpoint_rows"][-1] = inputs["endpoint_rows"][0].copy()
    elif mutation == "swapped_track":
        inputs["endpoint_rows"][0]["track"] = "L"
    elif mutation == "unequal_k":
        inputs["endpoint_rows"][0]["prompt_id"] = inputs["endpoint_rows"][16]["prompt_id"]
    elif mutation == "wrong_step":
        inputs["step32_rows"][0]["checkpoint_step"] = 64
    elif mutation == "initial_mismatch":
        inputs["initial_rows_by_track"] = {"N": copy.deepcopy(inputs["shared_initial_rows"])}
        inputs["initial_rows_by_track"]["N"][0]["category"] = "I"
    elif mutation == "extra_initial":
        inputs["initial_rows_by_track"]["OOD"] = []
    elif mutation == "missing_step":
        inputs["step_summaries"].pop()
    else:
        inputs["step_summaries"][-1] = inputs["step_summaries"][0].copy()

    def unexpected(*args, **kwargs):
        pytest.fail("Coverage failure must precede bootstrap recomputation")

    monkeypatch.setattr(api, "analyze_sampled_endpoints", unexpected)
    with pytest.raises(ValueError):
        _validate(report_case)


@pytest.mark.parametrize(
    "mutation",
    [
        "placeholder",
        "budget",
        "training_claim",
        "bootstrap_scope",
        "duplicate_json",
        "not_completed",
    ],
)
def test_rejects_pilot_report_missing_or_false_facts(report_case, mutation):
    path = report_case[0] / "pilot_report.md"
    text = path.read_text()
    if mutation == "placeholder":
        text = "# TODO\n"
    elif mutation == "budget":
        text = text.replace('"step64_outputs": 8576', '"step64_outputs": 9999')
    elif mutation == "training_claim":
        text = text.replace("seed 17", "seed 18")
    elif mutation == "bootstrap_scope":
        text = text.replace("单训练 seed 为探索性结果", "已证明所有 seed 无伤害")
    elif mutation == "duplicate_json":
        text = text.replace('"new_outputs": 14400', '"new_outputs": 14400, "new_outputs": 14400')
    else:
        text = text.replace('"status": "PASS"', '"status": "FAIL"')
    path.write_text(text)
    with pytest.raises(ValueError):
        _validate(report_case)


def test_historical_k16_initials_keep_track_specific_denominators(report_case):
    root, inputs = report_case
    inputs["initial_rows_by_track"] = {
        "N": _rows("N", dict.fromkeys(FAMILIES, 48), step=0, initial=True, k=16),
        "L": _rows("L", {"legacy_a": 24, "legacy_b": 64}, step=0, initial=True, k=16),
    }
    details = json.loads(
        (root / "pilot_report.md").read_text().split("```json\n")[1].split("```")[0]
    )
    _reports(root, inputs["endpoint_rows"], inputs["initial_rows_by_track"], details)
    result = _validate(report_case)
    assert result["counts"]["initial_outputs_by_track"] == {"N": 4608, "L": 2816}
    results = json.loads((root / "endpoint_metrics.json").read_text())
    for track in ("N", "L"):
        panel = results[track]["comparisons"]["X_VALID_minus_initial"]["panel"]
        assert {p["n"] for p in panel["reference_counts_by_prompt"].values()} == {16}
        assert {p["n"] for p in panel["target_counts_by_prompt"].values()} == {8}


@pytest.mark.parametrize(
    "name",
    ["endpoint_metrics.json", "N_L_OOD_effects.csv", "learning_curves.csv", "pilot_report.md"],
)
def test_missing_artifact_cannot_pass(report_case, name):
    (report_case[0] / name).unlink()
    with pytest.raises(ValueError):
        _validate(report_case)


@pytest.mark.parametrize("mutation", ["duplicate_field", "nan", "overflow", "escaping_symlink"])
def test_json_parsing_and_artifact_boundary_are_strict(report_case, mutation):
    root = report_case[0]
    path = root / "endpoint_metrics.json"
    if mutation == "escaping_symlink":
        outside = root.parent / "outside.json"
        path.rename(outside)
        path.symlink_to(outside)
    else:
        prefix = {
            "duplicate_field": '{"N": {}, "N": {}, ',
            "nan": '{"invalid": NaN, ',
            "overflow": '{"invalid": 1e999, ',
        }[mutation]
        path.write_text(prefix + path.read_text()[1:])
    with pytest.raises(ValueError):
        _validate(report_case)


def test_csv_nested_scope_values_and_independent_track_keys_are_bound(report_case):
    path = report_case[0] / "N_L_OOD_effects.csv"
    fields, rows = _csv(path)
    group_row = next(row for row in rows if row["scope"] != "overall")
    group_row["simultaneous_ci_groups"] = "['other-track/other-family']"
    _save_csv(path, fields, rows)
    with pytest.raises(ValueError):
        _validate(report_case)


def test_replacing_l_primary_with_equal_family_sensitivity_is_rejected(report_case):
    root = report_case[0]
    path = root / "endpoint_metrics.json"
    metrics = json.loads(path.read_text())
    metrics["L"] = json.loads((root / "L_family_equal_sensitivity.json").read_text())
    write_json(path, metrics)
    with pytest.raises(ValueError):
        _validate(report_case)


CUMULATIVE_POLICY_NAME = "finite_cumulative_kl_or_sequence_p99_two_arms_to_step64"


def _recursive_summaries():
    summaries = []
    for arm in ARMS:
        for step in range(1, 65):
            inherited = arm == "X_BASE" and step <= 51
            mean, p99, status = 0.01, 0.1, "PASS"
            if arm == "X_BASE" and step >= 35:
                mean = 0.101 if step >= 51 else 0.02
                p99 = 2.1
                status = "DIAGNOSTIC_STOP" if step in (35, 51) else "DIAGNOSTIC_WARNING"
            elif arm == "X_VALID" and step > 50:
                mean, status = 0.11, "DIAGNOSTIC_WARNING"
            alarms = {"mean_token_kl": mean > 0.1, "sequence_log_ratio_p99_abs": p99 > 2.0}
            summaries.append(
                {
                    "arm": arm,
                    "step": step,
                    "status": status,
                    "source_segment": "parent" if inherited else "current",
                    "training_category_counts": {"X": 32, "S": 0, "W": 0, "I": 0},
                    "zero_advantage_groups": 4,
                    "control_diagnostic": {
                        "mean_token_kl": mean,
                        "sequence_log_ratio_p99_abs": p99,
                        "status": "STOP_DIAGNOSE"
                        if any(alarms.values())
                        else "WITHIN_ENGINEERING_LIMITS",
                        "should_stop": any(alarms.values()),
                        "alarms": alarms,
                        "thresholds": {
                            "mean_token_kl": 0.1,
                            "sequence_log_ratio_p99_abs": 2.0,
                            "comparison": "strict_greater_than",
                        },
                    },
                    "update": {"loss": 0.0, "grad_norm_preclip": 0.0, "actual_step_norm": 0.0},
                }
            )
    return summaries


def test_recursive_report_steps_preserve_original_stop_warning_stop_history():
    from src.r4_report_gate import _steps

    summaries = _recursive_summaries()
    original = copy.deepcopy(summaries)
    actual = _steps(
        summaries,
        continuation={"schema_version": 2, "reviewed_warning_policy": CUMULATIVE_POLICY_NAME},
    )
    assert actual == original == summaries
    assert [s["step"] for s in actual if s["status"] == "DIAGNOSTIC_STOP"] == [35, 51]
    assert actual[35]["source_segment"] == "parent"
    assert actual[35]["status"] == "DIAGNOSTIC_WARNING"
    assert actual[51]["source_segment"] == "current"
    assert actual[51]["status"] == "DIAGNOSTIC_WARNING"


@pytest.mark.parametrize(
    "mutation",
    [
        "hidden_alarm",
        "current_stop",
        "false_warning",
        "flag",
        "threshold",
        "nan",
        "negative",
        "unknown_source",
        "unknown_policy",
        "wrong_schema",
        "nonfinite_update",
        "nonfinite_group",
    ],
)
def test_recursive_report_steps_reject_hidden_or_unreviewed_diagnostics(mutation):
    from src.r4_report_gate import _steps

    summaries = _recursive_summaries()
    policy = {"schema_version": 2, "reviewed_warning_policy": CUMULATIVE_POLICY_NAME}
    sample = summaries[51]
    diagnostic = sample["control_diagnostic"]
    if mutation == "hidden_alarm":
        sample["status"] = "PASS"
    elif mutation == "current_stop":
        sample["status"] = "DIAGNOSTIC_STOP"
    elif mutation == "false_warning":
        summaries[0]["status"] = "DIAGNOSTIC_WARNING"
    elif mutation == "flag":
        diagnostic["alarms"]["mean_token_kl"] = False
    elif mutation == "threshold":
        diagnostic["thresholds"]["mean_token_kl"] = 0.2
    elif mutation == "nan":
        diagnostic["mean_token_kl"] = float("nan")
    elif mutation == "negative":
        diagnostic["mean_token_kl"] = -0.1
    elif mutation == "unknown_source":
        sample["source_segment"] = "parent/parent"
    elif mutation == "unknown_policy":
        policy["reviewed_warning_policy"] = "waive_anything"
    elif mutation == "wrong_schema":
        policy["schema_version"] = 1
    elif mutation == "nonfinite_update":
        sample["update"]["grad_norm_preclip"] = float("inf")
    elif mutation == "nonfinite_group":
        diagnostic["group_mean_token_kl"] = {"cross_series/SYMBOLIC_FRESH": float("nan")}
    with pytest.raises(ValueError):
        _steps(summaries, continuation=policy)


CUMULATIVE_POLICY_PROSE = (
    "本次按记录的两臂一致恢复决策继续有限累计平均 KL 或 sequence p99 告警；"
    "非有限数及测量、污染、哈希、parity 故障仍停止。"
)


def _recursive_pilot(report_case):
    root, inputs = report_case
    summaries = _recursive_summaries()
    inputs["step_summaries"] = summaries
    continuation = {
        "schema_version": 2,
        "reviewed_warning_policy": CUMULATIVE_POLICY_NAME,
        "continuation_hash": "a" * 64,
    }
    inputs["continuation"] = continuation
    path = root / "pilot_report.md"
    text = path.read_text()
    prose = text.split("```json", 1)[0].replace(
        "状态：PASS", "状态：COMPLETED_WITH_DIAGNOSTIC_WARNINGS"
    )
    prose = prose.replace(
        "未采用诊断恢复决策；任何固定 control 超线仍停止。", CUMULATIVE_POLICY_PROSE
    )
    details = json.loads(text.split("```json\n")[1].split("```", 1)[0])
    details.update(
        status="COMPLETED_WITH_DIAGNOSTIC_WARNINGS",
        continuation_hash=continuation["continuation_hash"],
        reviewed_warning_policy=CUMULATIVE_POLICY_NAME,
        diagnostic_warning_count=sum(s["status"] != "PASS" for s in summaries),
        diagnostic_stop_history=[
            {
                key: s[key]
                for key in ("arm", "step", "status", "source_segment", "control_diagnostic")
            }
            for s in summaries
            if s["status"] == "DIAGNOSTIC_STOP"
        ],
    )
    _write_csv(root / "learning_curves.csv", [_curve(s) for s in summaries])
    path.write_text(prose + "```json\n" + json.dumps(details, ensure_ascii=False) + "\n```\n")
    return details


@pytest.mark.parametrize(
    "mutation",
    ["missing_stop", "changed_stop", "count", "hash", "policy", "prose", "stale_policy_prose"],
)
def test_recursive_report_pilot_requires_bound_diagnostic_history(report_case, mutation):
    from src.r4_report_gate import _pilot

    details = _recursive_pilot(report_case)
    if mutation == "missing_stop":
        details["diagnostic_stop_history"].pop(0)
    elif mutation == "changed_stop":
        details["diagnostic_stop_history"][0]["status"] = "DIAGNOSTIC_WARNING"
    elif mutation == "count":
        details["diagnostic_warning_count"] -= 1
    elif mutation == "hash":
        details["continuation_hash"] = "b" * 64
    elif mutation == "policy":
        details["reviewed_warning_policy"] = "sequence_p99_only_two_arms_to_step64"
    path = report_case[0] / "pilot_report.md"
    prose = path.read_text().split("```json", 1)[0]
    if mutation == "prose":
        prose = prose.replace(CUMULATIVE_POLICY_PROSE, "")
    elif mutation == "stale_policy_prose":
        prose += "平均 KL 超线、非有限数及测量、污染、哈希、parity 故障仍停止。\n"
    path.write_text(prose + "```json\n" + json.dumps(details, ensure_ascii=False) + "\n```\n")
    with pytest.raises(ValueError):
        _pilot(
            report_case[0],
            continuation=report_case[1]["continuation"],
            step_summaries=report_case[1]["step_summaries"],
        )


def test_recursive_complete_report_binds_stop_history_and_warning_status(report_case):
    from src.core import canonical_hash

    details = _recursive_pilot(report_case)
    result = _validate(report_case)
    assert result["status"] == "PASS"
    assert result["completion_status"] == "COMPLETED_WITH_DIAGNOSTIC_WARNINGS"
    assert result["continuation_hash"] == details["continuation_hash"]
    assert result["reviewed_warning_policy"] == CUMULATIVE_POLICY_NAME
    assert result["diagnostic_warning_count"] == details["diagnostic_warning_count"]
    assert result["diagnostic_stop_history_sha256"] == canonical_hash(
        details["diagnostic_stop_history"]
    )
