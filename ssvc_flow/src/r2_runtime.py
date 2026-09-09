"""Hash-bound, resumable R2 inference on the certified fresh Qwen policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .constraint_solver import satisfies, solve
from .core import (
    PROJECT_ROOT,
    RunStore,
    canonical_hash,
    file_hash,
    frozen_writer,
    phase_artifacts,
    source_commit,
    write_json,
)
from .optimizer_fork import parameter_hash, state_hash
from .r1_reference_smoke import _certificate_check, _helper_config
from .smoke_runtime import Telemetry, _seed_everything, _verify_scene_images
from .verifiers import annotate

MAIN_CONDITIONS = (
    "SYM_ORIGINAL",
    "SYM_CLEAR",
    "SYM_NO_OPERATION",
    "IMAGE_CUE",
    "IMAGE_ONLY",
    "ORACLE_INDEX",
)
LONG_CONDITIONS = ("SYM_LONG", "SYM_THINKING")
METRICS = (
    "pX",
    "pS",
    "pW",
    "pI",
    "copy_observation",
    "repair_changed_coordinate",
    "destroyed_any_correct_coordinate",
    "destroyed_correct_coordinate_fraction",
    "relation_constraints_satisfied",
    "full_constraints_satisfied",
    "truncated",
)
INTERVENTIONS = {
    "SYM_ORIGINAL": "original symbolic repair; original information; 64 tokens",
    "SYM_CLEAR": "fixed expression change; same information; 64 tokens",
    "SYM_NO_OPERATION": "downstream sentence removed; evaluator operation retained; 64 tokens",
    "IMAGE_CUE": "adds true chart to symbolic repair; more information; 64 tokens",
    "IMAGE_ONLY": "direct chart reading; observation and cue removed; 64 tokens",
    "ORACLE_INDEX": "adds true error location, no correct value; more information; 64 tokens",
    "SYM_LONG": "original symbolic repair; non-thinking budget 256 tokens",
    "SYM_THINKING": (
        "original symbolic repair; official thinking template; total budget 1024 tokens"
    ),
}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _invalid_type(raw):
    if not raw.strip():
        return "empty"
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError, RecursionError):
        if raw.lstrip().startswith("[") and raw.count("[") > raw.count("]"):
            return "truncated_or_unclosed"
        if not raw.lstrip().startswith("[") or raw.lstrip().startswith("``"):
            return "extra_text_or_thinking"
        return "other"
    if not isinstance(parsed, list) or len(parsed) != 4:
        return "wrong_length_or_structure"
    if any(type(value) is not int for value in parsed):
        return "non_integer"
    if any(not 0 <= value < 100 for value in parsed):
        return "out_of_domain"
    return None


def annotate_diagnostic(raw, scene, *, thinking=None):
    """All strict fields use complete raw text; segmented thinking is separate."""
    annotation = annotate(raw, scene["truth_world"], scene["operation"], scene["cue"])
    parsed = annotation["parsed_world"]
    truth, observed, changed = scene["truth_world"], scene["observed_world"], scene["changed_index"]
    valid = parsed is not None
    hamming_observed = sum(a != b for a, b in zip(parsed, observed, strict=True)) if valid else None
    damage = sum(parsed[i] != truth[i] for i in range(4) if i != changed) if valid else None
    relations = bool(valid and satisfies(parsed, scene["cue"]))
    final_status = (
        thinking.get(
            "final_status",
            "resolved_final" if thinking.get("status") == "RESOLVED" else "unresolved_final",
        )
        if thinking is not None
        else "not_applicable"
    )
    final = (
        annotate(thinking["final_text"], truth, scene["operation"], scene["cue"])
        if thinking is not None
        and final_status == "resolved_final"
        and isinstance(thinking.get("final_text"), str)
        else None
    )
    cue = scene["cue"]
    if valid and cue["family"] == "cross_series":
        checks = [parsed[i] + parsed[j] == total for i, j, total in cue["edges"]]
    elif valid and cue["family"] == "trend":
        a, b, c, d = parsed
        checks = [b - a == c - b, c - b == d - c]
    elif valid:
        checks = [parsed[cue["known_index"]] == cue["known_value"]]
    else:
        checks = []
    return {
        **annotation,
        "extracted_action": parsed,
        "parse_result": "valid" if valid else "invalid",
        "malformed_type": _invalid_type(raw) if not valid else None,
        "copy_observation": valid and parsed == observed,
        "repair_changed_coordinate": valid and parsed[changed] == truth[changed],
        "destroyed_correct_coordinate_count": damage,
        "destroyed_correct_coordinate_fraction": damage / 3 if valid else None,
        "destroyed_any_correct_coordinate": valid and damage > 0,
        "hamming_to_observed": hamming_observed,
        "hamming_to_truth": sum(a != b for a, b in zip(parsed, truth, strict=True))
        if valid
        else None,
        "relation_constraints_satisfied": relations,
        "full_constraints_satisfied": relations and hamming_observed == 1,
        "constraint_results": checks,
        "constraint_satisfaction_fraction": sum(checks) / len(checks) if checks else None,
        "diagnostic_final_status": final_status,
        "diagnostic_final_X": final["category"] == "X" if final is not None else None,
        "diagnostic_final_category": final["category"] if final is not None else None,
        "diagnostic_final_parsed_world": final["parsed_world"] if final is not None else None,
    }


def generation_checks(generation, tokenizer, eos_ids, max_new_tokens):
    """Return faults so returned malformed outputs can be persisted before abort."""
    errors = []
    tokens, scores = generation.get("token_ids", []), generation.get("behavior_token_logprobs", [])
    if (
        not tokens
        or len(tokens) > max_new_tokens
        or any(type(t) is not int or t < 0 for t in tokens)
    ):
        errors.append("invalid token ids or token budget")
    if generation.get("completion_length") != len(tokens) or len(scores) != len(tokens):
        errors.append("token score length mismatch")
    if any(
        not isinstance(score, (float, int)) or not math.isfinite(score) or score > 1e-5
        for score in scores
    ):
        errors.append("invalid behavior log probability")
    if tokens:
        terminal = tokens[-1] in eos_ids
        if any(t in eos_ids for t in tokens[:-1]):
            errors.append("tokens continue past a sampled EOS")
        expected_stop = "eos" if terminal else "length"
        if generation.get("stop_reason") != expected_stop or (
            not terminal and len(tokens) != max_new_tokens
        ):
            errors.append("EOS or truncation mismatch")
        decoded = tokenizer.decode(
            tokens[:-1] if terminal else tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != generation.get("raw_completion"):
            errors.append("raw token decode mismatch")
    return errors


def request_ledger(requests, identity):
    rows = []
    for request in requests:
        protocol = {k: v for k, v in request.items() if k not in {"sample_key", "sample_seed"}}
        rng = {
            "phase": "R2",
            "seed_root": 20260909,
            "model_hash": identity.get("model_hash"),
            "adapter_hash": identity.get("initial_adapter_hash"),
            "protocol_version": identity.get("protocol_version"),
            "base_scene_id": protocol["base_scene_id"],
            "prompt_id": protocol["prompt_id"],
            "condition": protocol["condition"],
            "decode_mode": protocol["decode_mode"],
            "rollout_index": protocol["rollout_index"],
        }
        # The input builder cannot know the measured initial adapter identity.
        # Bind the actual stream here and save it in request_manifest.json.
        # Statistical pairing is by scene, independent of token-wise RNG coupling.
        rng_key = canonical_hash(rng)
        rows.append(
            {
                **protocol,
                "sample_rng_key": rng_key,
                "sample_seed": int(rng_key[:8], 16) % (2**31),
                "sample_index": protocol["rollout_index"],
                "sample_key": canonical_hash(
                    {"run_identity": identity, "request": protocol, "rng": rng}
                ),
            }
        )
    if len({r["sample_key"] for r in rows}) != len(rows):
        raise ValueError("duplicate R2 sample requests")
    return rows


def validate_existing_rows(records, requests):
    wanted = {r["sample_key"]: r for r in requests}
    for key, row in records.items():
        if key not in wanted or any(row.get(k) != v for k, v in wanted[key].items()):
            raise ValueError("R2 resumed ledger request identity mismatch")
        if row.get("record_hash") != canonical_hash(
            {k: v for k, v in row.items() if k != "record_hash"}
        ):
            raise ValueError("R2 resumed ledger content hash mismatch")
        if row.get("execution_checks", {}).get("passed") is not True:
            raise ValueError("R2 preserved failed execution cannot be skipped on resume")


def _vector(row):
    return np.asarray(
        [
            *(float(row["category"] == c) for c in "XSWI"),
            *(float(row.get(key) or 0) for key in METRICS[4:-1]),
            float(row.get("stop_reason") == "length"),
        ]
    )


def _endpoint(values):
    valid = values[..., :3].sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = values[..., 0] / valid
    return np.concatenate((values, ratio[..., None]), axis=-1)


def paired_effects(rows, *, repeats=5000, seed=20260909):
    """Scene paired, family stratified bootstrap; never resample token outputs."""
    if repeats < 2:
        raise ValueError("at least two bootstrap replicates required")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["decode_mode"], row["condition"], row["family"], row["base_scene_id"])].append(
            row
        )
    means = {key: np.mean([_vector(r) for r in group], axis=0) for key, group in groups.items()}
    result = []
    for mode in sorted({key[0] for key in means}):
        conditions = sorted({key[1] for key in means if key[0] == mode} - {"SYM_ORIGINAL"})
        for condition in conditions:
            target = {
                (f, s): value
                for (m, c, f, s), value in means.items()
                if m == mode and c == condition
            }
            baseline = {
                (f, s): value
                for (m, c, f, s), value in means.items()
                if m == mode and c == "SYM_ORIGINAL"
            }
            if condition in LONG_CONDITIONS:
                baseline = {key: value for key, value in baseline.items() if key in target}
            if not target or target.keys() != baseline.keys():
                raise ValueError("missing paired scene support in R2 condition contrast")
            families = sorted({key[0] for key in target})
            rng = np.random.default_rng(seed)
            strata, draws = {}, {}
            for family in families:
                ids = sorted(key for key in target if key[0] == family)
                values = np.stack([[baseline[key], target[key]] for key in ids])
                strata[family] = values
                draws[family] = values[rng.integers(0, len(ids), size=(repeats, len(ids)))].mean(
                    axis=1
                )
            scopes = {
                **{f: ([f], "family") for f in families},
                "family_standardized": (families, "family_standardized"),
            }
            for label, (included, scope) in scopes.items():
                point = _endpoint(sum(strata[f].mean(axis=0) for f in included) / len(included))
                boot = _endpoint(sum(draws[f] for f in included) / len(included))
                differences = boot[:, 1] - boot[:, 0]
                for i, metric in enumerate((*METRICS, "qX_pool")):
                    point_delta = point[1, i] - point[0, i]
                    finite = bool(
                        np.isfinite(differences[:, i]).all() and math.isfinite(point_delta)
                    )
                    enough = all(len(strata[f]) >= 2 for f in included)
                    result.append(
                        {
                            "condition": condition,
                            "baseline_condition": "SYM_ORIGINAL",
                            "decode_mode": mode,
                            "scope": scope,
                            "family": label,
                            "metric": metric,
                            "baseline_estimate": float(point[0, i])
                            if math.isfinite(point[0, i])
                            else None,
                            "condition_estimate": float(point[1, i])
                            if math.isfinite(point[1, i])
                            else None,
                            "estimate": float(point_delta) if math.isfinite(point_delta) else None,
                            "low": float(np.quantile(differences[:, i], 0.025))
                            if finite and enough
                            else None,
                            "high": float(np.quantile(differences[:, i], 0.975))
                            if finite and enough
                            else None,
                            "CI_status": "ESTIMATED" if finite and enough else "UNKNOWN",
                            "scene_count": sum(len(strata[f]) for f in included),
                            "statistical_unit": "base_scene",
                            "bootstrap_replicates": repeats,
                            "bootstrap_seed": seed,
                            "CI_scope": "scene sampling only; fixed calibration panel policy",
                            "intervention": INTERVENTIONS.get(condition, condition),
                        }
                    )
    return result


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"python_type": type(value).__name__, "repr": repr(value)}


def _source():
    return {
        "source_commit": source_commit(),
        "source_files": {
            str(path.relative_to(PROJECT_ROOT)): file_hash(path)
            for path in sorted((PROJECT_ROOT / "src").rglob("*.py"))
        },
    }


def _load_panel(data_root):
    from .r2_inputs import select_panel

    root = Path(data_root)
    manifest = _json(root / "manifest.json")
    calibration_hash = file_hash(root / "calibration.jsonl")
    if manifest["files"]["calibration.jsonl"]["sha256"] != calibration_hash:
        raise ValueError("R2 calibration data manifest hash mismatch")
    scenes = [
        json.loads(line)
        for line in (root / "calibration.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len({s["base_scene_id"] for s in scenes}) != len(scenes) or any(
        s["split"] != "calibration" for s in scenes
    ):
        raise ValueError("R2 requires distinct calibration scenes")
    panel = select_panel(scenes)
    _verify_scene_images(panel, root)
    for scene in panel:
        changed = [i for i in range(4) if scene["truth_world"][i] != scene["observed_world"][i]]
        if solve(scene["observed_world"], scene["cue"]) != [scene["truth_world"]] or changed != [
            scene["changed_index"]
        ]:
            raise ValueError("R2 solver uniqueness, truth or changed-index audit failed")
    return panel, {
        "calibration_sha256": calibration_hash,
        "dataset_manifest_sha256": file_hash(root / "manifest.json"),
        "panel_hash": canonical_hash(panel),
        "selected_scene_ids": [s["base_scene_id"] for s in panel],
    }


def _write_analysis(out, rows):
    effects = paired_effects(rows)
    _csv(out / "paired_condition_effects.csv", effects)
    cells = defaultdict(list)
    for row in rows:
        cells[(row["condition"], row["decode_mode"], row["family"])].append(row)
    summaries, invalid = [], []
    for (condition, mode, family), group in sorted(cells.items()):
        prompts = defaultdict(list)
        for row in group:
            prompts[row["base_scene_id"]].append(row)
        mean = _endpoint(
            np.mean(
                [np.mean([_vector(r) for r in sample], axis=0) for sample in prompts.values()],
                axis=0,
            )
        )
        counts = Counter(r["category"] for r in group)
        record = {
            "condition": condition,
            "decode_mode": mode,
            "family": family,
            "base_scenes": len(prompts),
            "outputs": len(group),
            **{f"n_{c}": counts[c] for c in "XSWI"},
            **{
                metric: float(value) if math.isfinite(value) else None
                for metric, value in zip((*METRICS, "qX_pool"), mean, strict=True)
            },
            "mean_generated_tokens": sum(r["n_generated_tokens"] for r in group) / len(group),
            "intervention": INTERVENTIONS[condition],
        }
        if condition == "SYM_THINKING":
            record.update(
                {
                    "resolved_final_count": sum(
                        r["diagnostic_final_status"] == "resolved_final" for r in group
                    ),
                    "unresolved_final_count": sum(
                        r["diagnostic_final_status"] != "resolved_final" for r in group
                    ),
                    "diagnostic_final_X_count": sum(r["diagnostic_final_X"] is True for r in group),
                    "diagnostic_final_X_all_output_rate": sum(
                        r["diagnostic_final_X"] is True for r in group
                    )
                    / len(group),
                }
            )
        summaries.append(record)
        taxonomy = Counter(
            (r["malformed_type"], r["stop_reason"]) for r in group if r["category"] == "I"
        )
        for (malformed, stop), count in sorted(taxonomy.items()):
            invalid.append(
                {
                    "condition": condition,
                    "decode_mode": mode,
                    "family": family,
                    "malformed_type": malformed,
                    "stop_reason": stop,
                    "count": count,
                    "total_outputs": len(group),
                }
            )
    _csv(out / "condition_metrics.csv", summaries)
    _csv(out / "invalid_taxonomy.csv", invalid or [{"status": "NO_INVALID_OUTPUTS", "count": 0}])
    write_json(
        out / "condition_metrics.json",
        {
            "groups": summaries,
            "statistical_unit": "base_scene",
            "within_prompt_weight": "equal; sampled and greedy separate",
        },
    )
    lines = [
        "# R2 诊断结果",
        "",
        "所有条件保留原始输出；主指标严格解析完整输出。thinking 分段 final 指标单独统计。",
        "",
        "|条件|模式|家族|场景|输出|pX|复制率|修复坏坐标|新增错误事件率|完整约束|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"|{row['condition']}|{row['decode_mode']}|{row['family']}|{row['base_scenes']}|{row['outputs']}|{row['pX']:.4f}|{row['copy_observation']:.4f}|{row['repair_changed_coordinate']:.4f}|{row['destroyed_any_correct_coordinate']:.4f}|{row['full_constraints_satisfied']:.4f}|"
        )
    lines.extend(
        [
            "",
            "## 统计与输入范围",
            "",
            "72 个 calibration 场景的六个主条件分别比较；"
            "长输出诊断复用其中 12 个场景的 64-token 对照。采样与 greedy 分开。"
            "差值为条件减 SYM_ORIGINAL，按家族分层、base_scene 配对 bootstrap 5,000 次，"
            "seed=20260909；不对每条输出二次抽样。qX_pool 每次以总体 pX/validity 重算。"
            "CI 仅反映场景抽样，不涵盖训练 seed 不确定性，也不替代完整 P3。",
            "",
            "关系约束满足只考察 cue；完整约束额外要求合法动作且相对观察恰改一处。"
            "新增错误事件率以全部输出为分母，非法输出不计为已观测的坐标损坏；"
            "逐条坐标值在非法输出时为 null。thinking 的 diagnostic_final_X "
            "只计算明确官方分隔后的 final，未解析 final 保持 null，成功率另外给出全输出分母。",
            "",
        ]
    )
    lines.extend(
        f"- {condition}: {description}" for condition, description in INTERVENTIONS.items()
    )
    lines.extend(
        [
            "",
            "## 实际执行与证据",
            "",
            f"执行类型：{rows[0]['execution_kind']}。冻结原始模型与初始化 LoRA；"
            "0 optimizer step，0 backward。单序列 uncached 路径，未启用缓存或批量优化。"
            "CPU_FAKE_ADAPTER_FIXTURE 只检验编排，不构成真实模型测量或数值认证。",
            "",
            f"原始证据：{out / 'samples.jsonl'}；完整输入文本/token/hash、"
            "输出文本/token/behavior logps、逐输出测量保存在此 ledger。"
            "diagnostic_rollouts.jsonl 是其相同内容的阶段命名文件。",
            "",
            "与锁定配置的差异仅为六个预注册输入干预，"
            "以及 256/1024-token 长输出诊断和正式 thinking 模板。"
            "执行异常及失败尝试记录在 execution_attempts.jsonl；低正确率本身不是执行失败。",
            "",
        ]
    )
    (out / "diagnosis_report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")


def _finish(out, status, execution_kind, details):
    write_json(out / "progress.json", {"state": status, **details})
    files = [
        p
        for p in sorted(out.iterdir())
        if p.is_file()
        and p.name not in {"status.json", "manifest.json", "report.md", ".writer.lock"}
    ]
    result = phase_artifacts(out, "R2", status, details, files)
    result["execution_kind"] = execution_kind
    write_json(out / "status.json", result)
    return result


def _attempt(out, value):
    with (out / "execution_attempts.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_r2(
    config, data_root, out, r0_dir, r1_run, supplement_dir, *, resume=False, _adapter_factory=None
):
    import torch

    from .model_adapters import load_adapter
    from .next_stage_runtime import (
        validate_config_against_gate,
        validate_prerequisites,
        validate_runtime_environment,
    )
    from .r2_inputs import (
        build_requests,
        prepare_r2,
        split_thinking_completion,
        variant_prompt,
        write_panel_artifacts,
    )

    gate = validate_prerequisites(r0_dir, r1_run, supplement_dir)
    config = validate_config_against_gate(config, gate)
    if Path(data_root).resolve() != Path(config["data_root"]).resolve():
        raise ValueError("R2 data root differs from certified calibration data")
    _helper_config(config)
    panel, data_binding = _load_panel(data_root)
    if (
        data_binding["calibration_sha256"] != gate["binding"]["calibration_sha256"]
        or data_binding["dataset_manifest_sha256"] != gate["binding"]["data_manifest_sha256"]
    ):
        raise ValueError("R2 calibration data no longer matches the passed R0/R1 evidence")
    requests = build_requests(panel)
    if len(requests) != 2232:
        raise ValueError("R2 requires exactly 2160 main plus 72 long outputs")
    expected = Counter({(c, "sample"): 288 for c in MAIN_CONDITIONS})
    expected.update({(c, "greedy"): 72 for c in MAIN_CONDITIONS})
    expected.update({(c, "sample"): 24 for c in LONG_CONDITIONS})
    expected.update({(c, "greedy"): 12 for c in LONG_CONDITIONS})
    if Counter((r["condition"], r["decode_mode"]) for r in requests) != expected:
        raise ValueError("R2 condition or sampled/greedy allocation changed")
    if not _adapter_factory and not torch.cuda.is_available():
        raise RuntimeError("R2 requires an allocated CUDA device; no model download attempted")
    environment = (
        {"execution_kind": "CPU_FAKE_ADAPTER_FIXTURE", "environment_validation": "NOT_REAL_MODEL"}
        if _adapter_factory
        else validate_runtime_environment(gate)
    )
    source = _source()
    execution = "CPU_FAKE_ADAPTER_FIXTURE" if _adapter_factory else "REAL_CUDA_INFERENCE"
    identity = {
        "phase": "R2",
        "protocol_version": config["protocol_version"],
        "model_hash": canonical_hash(config["model"]),
        "data_hash": canonical_hash(data_binding),
        "config_hash": canonical_hash(config),
        "source_hash": canonical_hash(source),
        "gate_hash": canonical_hash(gate["binding"]),
        "requests_hash": canonical_hash(requests),
        "initial_adapter_hash": gate["certificate"]["initial_adapter_hash"],
        "execution_kind": execution,
    }
    ledger = request_ledger(requests, identity)
    out = Path(out)
    with frozen_writer(out):
        store = RunStore(out, identity, resume=resume)
        validate_existing_rows(store.records, ledger)
        write_panel_artifacts(panel, out)
        write_json(out / "data_binding.json", data_binding)
        write_json(out / "gate_binding.json", gate["binding"])
        write_json(out / "request_manifest.json", {"count": len(ledger), "requests": ledger})
        attempt_id = (
            len((out / "execution_attempts.jsonl").read_text().splitlines())
            if (out / "execution_attempts.jsonl").exists()
            else 0
        )
        started = time.perf_counter()
        generated, current_request = 0, None
        uncommitted_generation = None
        has_uncommitted_generation = False
        before_model, adapter = None, None
        write_json(
            out / "status.json",
            {
                "phase": "R2",
                "status": "RUNNING",
                "execution_kind": execution,
                "source_commit": source["source_commit"],
                "details": {"completed": len(store.records), "total": len(ledger)},
            },
        )
        write_json(
            out / "progress.json",
            {"state": "LOADING_MODEL", "completed": len(store.records), "total": len(ledger)},
        )
        try:
            _seed_everything(17)
            adapter = (_adapter_factory or load_adapter)(
                "qwen35_9b",
                {
                    "id": config["model"]["id"],
                    "revision": config["model"]["revision"],
                    "expected_layers": 32,
                },
                image_token_limit=768,
            )
            if (
                adapter.model_id != config["model"]["id"]
                or adapter.revision != config["model"]["revision"]
            ):
                raise ValueError("R2 loaded model identity mismatch")
            base_hash = _certificate_check(gate["certificate"], adapter, config)
            if not _adapter_factory:
                environment = validate_runtime_environment(gate, adapter.audit)
            adapter_names = {
                name for name, p in adapter.model.named_parameters() if p.requires_grad
            }
            adapter_hash = parameter_hash(adapter.model, trainable=True)
            adapter.model.requires_grad_(False)
            adapter.model.eval()
            before_model = parameter_hash(adapter.model, trainable=False)
            runtime = {
                "identity": identity,
                "config": config,
                **source,
                "model_audit": adapter.audit,
                "environment": environment,
                "base_parameter_hash": base_hash,
                "initial_adapter_hash": adapter_hash,
                "all_parameter_hash": before_model,
                "selected_probability_path": "uncached_prefix_recompute",
                "adapter_state": "fresh initialized LoRA; no scratch checkpoint loaded",
                "optimizer_steps": 0,
                "backward_calls": 0,
                "requests": 2232,
            }
            runtime_path = out / "runtime_lock.json"
            if runtime_path.exists():
                previous = _json(runtime_path)
                for key in (
                    "identity",
                    "config",
                    "base_parameter_hash",
                    "initial_adapter_hash",
                    "all_parameter_hash",
                    "environment",
                ):
                    if previous.get(key) != runtime[key]:
                        raise ValueError("R2 runtime identity or model state drift on resume")
            else:
                write_json(runtime_path, runtime)
            scene_map = {scene["base_scene_id"]: scene for scene in panel}
            prepared, last_prompt = None, None
            telemetry = Telemetry()
            with torch.no_grad():
                for request in ledger:
                    if request["sample_key"] in store.keys:
                        continue
                    current_request = request
                    scene = scene_map[request["base_scene_id"]]
                    if last_prompt != request["prompt_id"]:
                        prompt = variant_prompt(scene, request["condition"])
                        prepared = prepare_r2(
                            adapter, prompt, data_root, enable_thinking=request["enable_thinking"]
                        )
                        last_prompt = request["prompt_id"]
                    audit = prepared["audit"]
                    if audit.get("enable_thinking") != request["enable_thinking"]:
                        raise RuntimeError("R2 thinking template does not match intervention")
                    image_expected = request["condition"] in {"IMAGE_CUE", "IMAGE_ONLY"}
                    if bool(audit.get("image_token_count")) != image_expected or (
                        image_expected and not audit.get("pixel_values_hash")
                    ):
                        raise RuntimeError("R2 image presence does not match intervention")
                    telemetry.reset()
                    before_forward = adapter.forward_calls
                    generation = adapter.generate(
                        prepared,
                        seed=request["sample_seed"],
                        max_new_tokens=request["max_new_tokens"],
                        do_sample=request["decode_mode"] == "sample",
                    )
                    uncommitted_generation = generation
                    has_uncommitted_generation = True
                    generated += 1
                    try:
                        faults = generation_checks(
                            generation,
                            adapter.processor.tokenizer,
                            adapter.eos_ids,
                            request["max_new_tokens"],
                        )
                    except (ValueError, TypeError) as exc:
                        faults = [
                            f"returned generation validation error: {type(exc).__name__}: {exc}"
                        ]
                    if image_expected and generation.get("vision_forward_calls", 0) < 1:
                        faults.append("image did not enter measured visual forward")
                    segments = None
                    if request["enable_thinking"]:
                        try:
                            segments = split_thinking_completion(
                                generation.get("token_ids", []),
                                adapter.processor.tokenizer,
                                adapter.eos_ids,
                            )
                        except (ValueError, TypeError) as exc:
                            faults.append(
                                f"thinking segmentation error: {type(exc).__name__}: {exc}"
                            )
                            segments = {
                                "final_status": "unresolved_final",
                                "final_text": None,
                                "segmentation_error": str(exc),
                            }
                    raw = generation.get("raw_completion", "")
                    annotation = annotate_diagnostic(raw, scene, thinking=segments)
                    row = {
                        **request,
                        "run_id": canonical_hash(identity),
                        "phase": "R2",
                        "execution_kind": execution,
                        "protocol_version": config["protocol_version"],
                        "model_id": adapter.model_id,
                        "model_revision": adapter.revision,
                        "adapter_hash": adapter_hash,
                        "optimizer_state_hash": None,
                        "checkpoint_step": 0,
                        "train_seed": 17,
                        "split": "calibration",
                        "family": scene["constraint_family"],
                        "constraint_family": scene["constraint_family"],
                        "interface": "IMAGE_CUE_FRESH" if image_expected else "SYMBOLIC_FRESH",
                        "chart_type": scene["chart_type"],
                        "operation": scene["operation"],
                        "truth_world": scene["truth_world"],
                        "observed_world": scene["observed_world"],
                        "changed_index": scene["changed_index"],
                        "cue": scene["cue"],
                        "solution_count": 1,
                        "image_hash": scene["image_hash"] if image_expected else None,
                        **audit,
                        **generation,
                        **annotation,
                        "raw_token_ids": generation.get("token_ids", []),
                        "raw_text": raw,
                        "thinking_segments": segments,
                        "input_ids_hash": audit.get("tokenized_prompt_hash"),
                        "actual_image_tokens": audit.get("image_token_count"),
                        "generation_config_hash": canonical_hash(
                            {
                                **config["generation_proposed_N"],
                                "max_new_tokens": request["max_new_tokens"],
                                "enable_thinking": request["enable_thinking"],
                                "do_sample": request["decode_mode"] == "sample",
                            }
                        ),
                        "n_generated_tokens": len(generation.get("token_ids", [])),
                        "per_token_logprob_behavior": generation.get("behavior_token_logprobs", []),
                        "logprob_sequence": sum(generation.get("behavior_token_logprobs", [])),
                        "runtime_forward_by_reason": {
                            "generation": adapter.forward_calls - before_forward,
                            "likelihood": 0,
                            "backward": 0,
                        },
                        "elapsed": generation.get("elapsed_seconds"),
                        "peak_memory": telemetry.snapshot(),
                        "execution_checks": {"passed": not faults, "faults": faults},
                    }
                    row = _json_safe(row)
                    row["record_hash"] = canonical_hash(row)
                    store.append(row)
                    uncommitted_generation = None
                    has_uncommitted_generation = False
                    write_json(
                        out / "progress.json",
                        {
                            "state": "RUNNING",
                            "completed": len(store.records),
                            "total": len(ledger),
                            "last_request": request,
                            "completed_by_condition": dict(
                                Counter(r["condition"] for r in store.records.values())
                            ),
                        },
                    )
                    if faults:
                        raise RuntimeError(
                            "R2 returned output failed execution checks; raw record preserved: "
                            + "; ".join(faults)
                        )
            after_model = parameter_hash(adapter.model, trainable=False)
            after_adapter = state_hash(
                {
                    name: state_hash(p)
                    for name, p in adapter.model.named_parameters()
                    if name in adapter_names
                }
            )
            after_base = state_hash(
                {
                    name: state_hash(p)
                    for name, p in adapter.model.named_parameters()
                    if name not in adapter_names
                }
            )
            if (
                before_model != after_model
                or adapter_hash != after_adapter
                or base_hash != after_base
                or any(p.requires_grad for p in adapter.model.parameters())
            ):
                raise RuntimeError("R2 frozen model or adapter state changed")
            if _source() != source:
                raise RuntimeError("R2 source changed while inference was running")
            if (
                file_hash(Path(data_root) / "calibration.jsonl")
                != data_binding["calibration_sha256"]
            ):
                raise RuntimeError("R2 calibration file changed while inference was running")
            if (
                file_hash(Path(data_root) / "manifest.json")
                != data_binding["dataset_manifest_sha256"]
            ):
                raise RuntimeError("R2 dataset manifest changed while inference was running")
            _verify_scene_images(panel, data_root)
            validate_existing_rows(store.records, ledger)
            if len(store.records) != len(ledger):
                raise RuntimeError("R2 output coverage incomplete")
            rows = list(store.records.values())
            # A byte-identical named stage artifact; samples.jsonl remains the only
            # live append ledger and is never rewritten on resume.
            (out / "diagnostic_rollouts.jsonl").write_bytes((out / "samples.jsonl").read_bytes())
            _write_analysis(out, rows)
            audit = {
                "passed": True,
                "all_parameter_hash_before": before_model,
                "all_parameter_hash_after": after_model,
                "base_hash_before": base_hash,
                "base_hash_after": after_base,
                "adapter_hash_before": adapter_hash,
                "adapter_hash_after": after_adapter,
                "optimizer_steps": 0,
                "backward_calls": 0,
                "generation_calls_this_invocation": generated,
                "raw_sample_count": len(rows),
            }
            write_json(out / "inference_audit.json", audit)
            _attempt(
                out,
                {
                    "attempt": attempt_id,
                    "status": "PASS",
                    "generation_calls": generated,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            return _finish(
                out, "PASS", execution, {**audit, "total": len(ledger), "completed": len(rows)}
            )
        except Exception as exc:
            failure = {
                "type": type(exc).__name__,
                "message": str(exc),
                "current_request": current_request,
            }
            if has_uncommitted_generation:
                # An adapter may return structurally malformed scores/tokens.
                # Preserve the complete returned object even if derivation itself
                # raised, and make this key permanently non-skippable on resume.
                raw = uncommitted_generation
                failed_row = _json_safe(
                    {
                        **current_request,
                        "phase": "R2",
                        "execution_kind": execution,
                        "generation_return": raw,
                        "raw_text": raw.get("raw_completion") if isinstance(raw, dict) else None,
                        "raw_token_ids": raw.get("token_ids") if isinstance(raw, dict) else None,
                        "parse_result": "unmeasured_due_to_execution_error",
                        "category": None,
                        "execution_checks": {"passed": False, "faults": [failure]},
                    }
                )
                failed_row["record_hash"] = canonical_hash(failed_row)
                store.append(failed_row)
            _attempt(
                out,
                {
                    "attempt": attempt_id,
                    "status": "FAIL",
                    "error": failure,
                    "generation_calls": generated,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            (out / "report_zh.md").write_text(
                "# R2 执行失败\n\n"
                + json.dumps(failure, ensure_ascii=False, indent=2)
                + "\n\n已返回的输出均保留于 samples.jsonl；未完成条目不视作测量。\n",
                encoding="utf-8",
            )
            return _finish(
                out,
                "FAIL",
                execution,
                {
                    "error": failure,
                    "completed": len(store.records),
                    "total": len(ledger),
                    "generation_calls_this_invocation": generated,
                },
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--r0-dir", type=Path, required=True)
    parser.add_argument("--r1-run", type=Path, required=True)
    parser.add_argument("--supplement-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    from .next_stage_runtime import validate_config_against_gate, validate_prerequisites

    gate = validate_prerequisites(args.r0_dir, args.r1_run, args.supplement_dir)
    config = gate["config"]
    if args.config:
        import yaml

        supplied = yaml.safe_load(args.config.read_text())
        config = validate_config_against_gate(supplied, gate)
    result = run_r2(
        config,
        args.data_root or config["data_root"],
        args.out,
        args.r0_dir,
        args.r1_run,
        args.supplement_dir,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "execution_kind": result["execution_kind"],
                "out": str(args.out),
            }
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
