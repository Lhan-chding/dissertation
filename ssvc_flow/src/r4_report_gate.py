"""Read-only CPU acceptance of complete R4 sampled reports.

The caller verifies provenance, annotations, sampled actions and log probabilities,
and the full optimizer/checkpoint chain. This gate reconstructs the reported
statistics from those rows; a PASS here is neither CUDA evidence nor safety proof.
"""

from __future__ import annotations

import ast
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from .core import canonical_hash, file_hash
from .r3_report_gate import _csv_value, _file, _json, _parse, _same_json
from .r4_metrics import analyze_sampled_endpoints

ARMS = ("X_BASE", "X_VALID")
TRACK_COUNTS = {"N": 4608, "L": 2816, "OOD": 1152}
FAMILIES = {"cross_series", "duplicate_encoding", "trend"}
BOOTSTRAP = {"replicates": 5000, "seed": 20260909}


def _bank(rows, *, track, step, arms, prompts, k, family_scenes=None):
    if not isinstance(rows, (list, tuple)) or len(rows) != len(arms) * prompts * k:
        raise ValueError(f"Incomplete {track} step {step} sampled bank")
    keys, groups, identities = set(), defaultdict(dict), {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Malformed sampled report input")
        key, prompt, arm = row.get("sample_key"), row.get("prompt_id"), row.get("arm")
        identity = (
            row.get("family", row.get("constraint_family")),
            row.get("base_scene_id"),
            row.get("interface"),
        )
        if (
            not isinstance(key, str)
            or not key
            or key in keys
            or not isinstance(prompt, str)
            or not prompt
            or any(not isinstance(v, str) or not v for v in identity)
            or arm not in arms
            or row.get("track") != track
            or type(row.get("checkpoint_step")) is not int
            or row["checkpoint_step"] != step
            or row.get("decode_mode", "sample") not in ("sample", "sampled")
            or row.get("category") not in ("X", "S", "W", "I")
        ):
            raise ValueError("Duplicate, malformed, or wrong-track/checkpoint sampled input")
        index = row.get("sample_index")
        if type(index) is not int or index not in range(k) or index in groups[arm, prompt]:
            raise ValueError("Prompt sample indices must be the complete fixed K bank")
        if prompt in identities and identities[prompt] != identity:
            raise ValueError("Prompt identity differs between sampled rows")
        keys.add(key)
        identities[prompt] = identity
        groups[arm, prompt][index] = row
    if len(identities) != prompts or set(groups) != {(a, p) for a in arms for p in identities}:
        raise ValueError("Sampled arms do not cover the same complete prompt panel")
    if any(set(group) != set(range(k)) for group in groups.values()):
        raise ValueError("Prompt sample denominator differs from fixed K")
    pairs, scenes = set(), {}
    for family, scene, interface in identities.values():
        if (scene, interface) in pairs or (scene in scenes and scenes[scene] != family):
            raise ValueError("Duplicate scene/interface or conflicting scene family")
        pairs.add((scene, interface))
        scenes[scene] = family
    interfaces = (
        ("collision", "separating") if track == "L" else ("IMAGE_CUE_FRESH", "SYMBOLIC_FRESH")
    )
    if pairs != {(scene, interface) for scene in scenes for interface in interfaces}:
        raise ValueError("Both locked interfaces are required for every scene")
    if family_scenes is not None and Counter(scenes.values()) != family_scenes:
        raise ValueError("Fixed family scene coverage differs from R4 protocol")
    return identities


def _rows_hash(rows):
    return canonical_hash(sorted(rows, key=lambda row: row["sample_key"]))


def _steps(summaries, *, continuation=None):
    if not isinstance(summaries, (list, tuple)) or len(summaries) != 128:
        raise ValueError("Exactly 128 successful step summaries are required")
    result = {}
    for summary in summaries:
        if not isinstance(summary, dict):
            raise ValueError("Malformed step summary")
        arm, step = summary.get("arm"), summary.get("step")
        if (
            arm not in ARMS
            or type(step) is not int
            or step not in range(1, 65)
            or (arm, step) in result
        ):
            raise ValueError("Duplicate or unknown training arm/step")
        counts = summary.get("training_category_counts")
        zero = summary.get("zero_advantage_groups")
        allowed = {"PASS"}
        if (
            continuation is not None
            and continuation.get("reviewed_warning_policy")
            == "sequence_p99_only_two_arms_to_step64"
        ):
            allowed.add(
                "DIAGNOSTIC_STOP"
                if summary.get("source_segment") == "parent"
                else "DIAGNOSTIC_WARNING"
            )
        if (
            summary.get("status") not in allowed
            or not isinstance(counts, dict)
            or set(counts) != set("XSWI")
            or any(type(v) is not int or v < 0 for v in counts.values())
            or sum(counts.values()) != 32
            or type(zero) is not int
            or zero not in range(5)
        ):
            raise ValueError("Step status, 32-output denominator, or retained-group count differs")
        diagnostic, update = summary.get("control_diagnostic"), summary.get("update")
        if not isinstance(diagnostic, dict) or not isinstance(update, dict):
            raise ValueError("Missing measured step diagnostic/update")
        for values, fields in (
            (diagnostic, ("mean_token_kl", "sequence_log_ratio_p99_abs")),
            (update, ("loss", "grad_norm_preclip", "actual_step_norm")),
        ):
            for field in fields:
                value = values.get(field)
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(f"Missing or nonfinite step scalar: {field}")
        if continuation is not None:
            alarmed = diagnostic.get("should_stop") is True
            if (
                diagnostic["mean_token_kl"] > 0.1
                or (
                    alarmed
                    and (
                        diagnostic.get("alarms")
                        != {"mean_token_kl": False, "sequence_log_ratio_p99_abs": True}
                        or diagnostic.get("status") != "STOP_DIAGNOSE"
                        or diagnostic["sequence_log_ratio_p99_abs"] <= 2.0
                        or summary["status"] == "PASS"
                    )
                )
                or (not alarmed and summary["status"] != "PASS")
            ):
                raise ValueError("R4 continuation cannot hide or waive an unreviewed diagnostic")
        result[arm, step] = summary
    if set(result) != {(arm, step) for arm in ARMS for step in range(1, 65)}:
        raise ValueError("Incomplete 64-step coverage for both arms")
    return [result[key] for key in sorted(result)]


def _inputs(
    endpoint_rows,
    step32_rows,
    shared_initial_rows,
    initial_rows_by_track,
    step_summaries,
    *,
    continuation=None,
):
    if not isinstance(endpoint_rows, (list, tuple)) or len(endpoint_rows) != 8576:
        raise ValueError("Exactly 8576 independent-track endpoint rows are required")
    endpoint = {track: [] for track in TRACK_COUNTS}
    for row in endpoint_rows:
        if not isinstance(row, dict) or row.get("track") not in endpoint:
            raise ValueError("Unknown endpoint track")
        endpoint[row["track"]].append(row)
    identities = {}
    for track, rows in endpoint.items():
        identities[track] = _bank(
            rows,
            track=track,
            step=64,
            arms=ARMS,
            prompts=TRACK_COUNTS[track] // 16,
            k=8,
            family_scenes=dict.fromkeys(FAMILIES, 48)
            if track == "N"
            else {"cross_series": 36}
            if track == "OOD"
            else None,
        )
    panel = _bank(
        step32_rows,
        track="N",
        step=32,
        arms=ARMS,
        prompts=72,
        k=8,
        family_scenes=dict.fromkeys(FAMILIES, 12),
    )
    shared = _bank(
        shared_initial_rows,
        track="N",
        step=0,
        arms=("INITIAL",),
        prompts=72,
        k=8,
        family_scenes=dict.fromkeys(FAMILIES, 12),
    )
    if panel != shared or any(identities["N"].get(p) != identity for p, identity in shared.items()):
        raise ValueError("Shared step0/step32 panel must be the same paired subset of N step64")
    new_rows = [*endpoint_rows, *step32_rows, *shared_initial_rows]
    if len({row["sample_key"] for row in new_rows}) != len(new_rows):
        raise ValueError("New evaluation banks reused sampled trajectory keys")
    if not isinstance(initial_rows_by_track, dict) or set(initial_rows_by_track) not in (
        {"N"},
        {"N", "L"},
    ):
        raise ValueError("N initial panel is required; only verified historical L is optional")
    for track, rows in initial_rows_by_track.items():
        if track == "N" and isinstance(rows, (list, tuple)) and len(rows) == 576:
            _bank(
                rows,
                track="N",
                step=0,
                arms=("INITIAL",),
                prompts=72,
                k=8,
                family_scenes=dict.fromkeys(FAMILIES, 12),
            )
            if _rows_hash(rows) != _rows_hash(shared_initial_rows):
                raise ValueError("Fallback N initial bank differs from the shared step0 rows")
        else:
            historical = _bank(
                rows,
                track=track,
                step=0,
                arms=("INITIAL",),
                prompts=TRACK_COUNTS[track] // 16,
                k=16,
            )
            if historical != identities[track]:
                raise ValueError("Historical K16 initial must cover its own full endpoint track")
    return endpoint, _steps(step_summaries, continuation=continuation)


def _effects(results):
    rows = []
    for track, result in results.items():
        for comparison, value in result["comparisons"].items():
            for scope, metrics in value.get("responses", {}).items():
                for metric, values in metrics.items():
                    row = {
                        "track": track,
                        "comparison": comparison,
                        "scope": scope,
                        "metric": metric,
                    }
                    for field, value in values.items():
                        if isinstance(value, dict):
                            row.update({f"{field}_{k}": v for k, v in value.items()})
                        else:
                            row[field] = value
                    rows.append(row)
    return rows


def _curves(summaries):
    return [
        {
            "arm": s["arm"],
            "step": s["step"],
            "mean_token_kl": s["control_diagnostic"]["mean_token_kl"],
            "sequence_log_ratio_p99_abs": s["control_diagnostic"]["sequence_log_ratio_p99_abs"],
            "pX_training_bank": s["training_category_counts"]["X"] / 32,
            "v_training_bank": 1 - s["training_category_counts"]["I"] / 32,
            "zero_advantage_groups_retained": s["zero_advantage_groups"],
            "loss": s["update"]["loss"],
            "grad_norm_preclip": s["update"]["grad_norm_preclip"],
            "actual_step_norm": s["update"]["actual_step_norm"],
            "status": s["status"],
        }
        for s in summaries
    ]


def _check_csv(root, name, expected, key_fields):
    references = {tuple(row[k] for k in key_fields): row for row in expected}
    fields = {key for row in expected for key in row}
    if not expected or len(references) != len(expected):
        raise ValueError("Invalid recomputed CSV key coverage")
    seen = set()
    try:
        with _file(root, name).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if (
                not reader.fieldnames
                or len(reader.fieldnames) != len(fields)
                or set(reader.fieldnames) != fields
            ):
                raise ValueError("CSV columns differ from the fixed report schema")
            for row in reader:
                if set(row) != fields or any(value is None for value in row.values()):
                    raise ValueError("Malformed CSV row")
                key = tuple(_csv_value(row[k], expected[0][k]) for k in key_fields)
                if key not in references or key in seen:
                    raise ValueError("Unknown or duplicate CSV semantic key")
                for field in fields:
                    wanted = references[key].get(field)
                    if isinstance(wanted, (list, dict)):
                        value = ast.literal_eval(row[field])
                        _same_json(value, wanted, f"{name} {key}/{field}")
                    elif _csv_value(row[field], wanted) != wanted:
                        raise ValueError(f"CSV value differs: {key}/{field}")
                seen.add(key)
    except (csv.Error, SyntaxError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}: {exc}") from exc
    if seen != set(references):
        raise ValueError(f"Incomplete {name} measured coverage")
    return len(seen)


def _pilot(root, *, continuation=None):
    text = _file(root, "pilot_report.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```json\s*\n(.*?)\n```", text, flags=re.DOTALL)
    if len(blocks) != 1:
        raise ValueError("Pilot report must contain its one structured completion record")
    details = _parse(blocks[0], "pilot completion record")
    completion_status = "COMPLETED_WITH_DIAGNOSTIC_WARNINGS" if continuation is not None else "PASS"
    expected = {
        "status": completion_status,
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
        "arms": {"X_BASE": 64, "X_VALID": 64},
    }
    if not isinstance(details, dict):
        raise ValueError("Malformed pilot completion record")
    _same_json({key: details.get(key) for key in expected}, expected, "Pilot completion facts")
    prose = re.sub(r"\s+", "", text.split("```json", 1)[0])
    # These facts already occur in the production writer. No manual narrative or
    # reworded interpretation is requested from the experiment operator.
    facts = (
        f"状态：{completion_status}；执行类型：REAL_CUDA_TRAINING",
        "相同seed17LoRA/空Adam起点",
        "当前策略新生成B4K8轨迹",
        "每步一次实际Adam更新",
        "没有新增control生成",
        "KL是固定旧前缀条件方向，六群体等权",
        "不是当前策略occupancy的轨迹KL",
        "0/16/32/64checkpoint包含完整Adam、RNG、sampler",
        "N64、原L48和graphOOD分别报告",
        "变化仅用事前固定共享step0子panel",
        "L主结果按原固定题分布，家族等权另列敏感性",
        "不筛掉零奖励组",
        "base_scene家族分层配对bootstrap5000次",
        "max-stat仅同一指标的声明群体内",
        "单训练seed为探索性结果",
        "未执行A训练臂、新模型、在线SSVC、额外SFT或R3-warm",
    )
    if any(fact not in prose for fact in facts):
        raise ValueError("Pilot report omits or changes required execution/statistical scope facts")
    return canonical_hash(details)


def validate_r4_response_artifacts(
    root,
    *,
    endpoint_rows,
    step32_rows,
    shared_initial_rows,
    initial_rows_by_track,
    step_summaries,
    continuation=None,
):
    """Bind all R4 report values to caller-verified complete raw rows and steps.

    Raises ValueError/FloatingPointError for incomplete, malformed, or changed
    reports. Independent N/L/OOD ratios and optional initial K16 denominators are
    recomputed without smoothing, resampling outputs, or pooling across tracks.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("R4 report root is missing")
    endpoint, summaries = _inputs(
        endpoint_rows,
        step32_rows,
        shared_initial_rows,
        initial_rows_by_track,
        step_summaries,
        continuation=continuation,
    )
    results = {
        track: analyze_sampled_endpoints(rows, initial_rows_by_track.get(track), track=track)
        for track, rows in endpoint.items()
    }
    groups = {f"{row['family']}/{row['interface']}" for row in endpoint["L"]}
    recomputed = {
        "step32_metrics.json": analyze_sampled_endpoints(
            step32_rows, shared_initial_rows, track="N"
        ),
        "endpoint_metrics.json": results,
        "L_family_equal_sensitivity.json": analyze_sampled_endpoints(
            endpoint["L"],
            initial_rows_by_track.get("L"),
            track="L",
            group_weights={g: 1 / len(groups) for g in groups},
        ),
    }
    for name, value in recomputed.items():
        _same_json(_json(root, name), value, name)
    effects_count = _check_csv(
        root, "N_L_OOD_effects.csv", _effects(results), ("track", "comparison", "scope", "metric")
    )
    curves_count = _check_csv(root, "learning_curves.csv", _curves(summaries), ("arm", "step"))
    pilot_hash = _pilot(root, continuation=continuation)
    names = (*recomputed, "N_L_OOD_effects.csv", "learning_curves.csv", "pilot_report.md")
    return {
        "status": "PASS",
        "execution_kind": "CPU_MATH",
        "scope": (
            "Read-only R4 sampled endpoint, initial-change, sensitivity "
            "and learning-curve recomputation"
        ),
        "counts": {
            "endpoint_outputs_by_track": {track: len(rows) for track, rows in endpoint.items()},
            "step32_outputs": len(step32_rows),
            "shared_step0_outputs": len(shared_initial_rows),
            "initial_outputs_by_track": {
                track: len(rows) for track, rows in initial_rows_by_track.items()
            },
            "optimizer_steps": len(summaries),
            "effects_rows": effects_count,
            "learning_curve_rows": curves_count,
        },
        "input_sha256": {
            "endpoint_rows": _rows_hash(endpoint_rows),
            "step32_rows": _rows_hash(step32_rows),
            "shared_initial_rows": _rows_hash(shared_initial_rows),
            "initial_rows_by_track": {
                track: _rows_hash(rows) for track, rows in initial_rows_by_track.items()
            },
            "step_summaries": canonical_hash(summaries),
        },
        "artifact_sha256": {name: file_hash(_file(root, name)) for name in names},
        "recomputed_sha256": {name: canonical_hash(value) for name, value in recomputed.items()},
        "pilot_details_sha256": pilot_hash,
        "bootstrap": dict(BOOTSTRAP),
        "safety_status": "NOT_CERTIFIED",
    }
