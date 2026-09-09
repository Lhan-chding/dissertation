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
