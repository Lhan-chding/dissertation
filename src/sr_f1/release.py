"""Unified SR-F1 release: terminal matrix, complete slots and raw-text re-scoring.

A technical terminal failure is preserved as missing evidence, never a wrong
answer. CPU implementation completion cannot create a scientific release.
"""

# Chinese report punctuation is intentional.
# ruff: noqa: RUF001
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .evaluation import (
    BASELINE,
    TEST_POOLS,
    encoded,
    iter_evaluation_slots,
    read_jsonl,
    score_response,
    sha_file,
    validate_raw,
)


class ReleaseBlocked(RuntimeError):
    pass


def terminal_matrix(matrix, registered):
    rows = matrix.get("runs") if isinstance(matrix, dict) else matrix
    if not isinstance(rows, list) or len(rows) != 15:
        raise ReleaseBlocked("All 15 registered scientific paths must be present")
    expected = {r["run_id"]: r for r in registered}
    observed = {r["run_id"]: r for r in rows}
    if len(observed) != 15 or set(observed) != set(expected):
        raise ReleaseBlocked("Scientific matrix identities differ from preregistration")
    for run_id, row in observed.items():
        for field, value in expected[run_id].items():
            if row.get(field) != value:
                raise ReleaseBlocked(f"Changed scientific registration: {run_id}/{field}")
        if row.get("status") not in ("COMPLETE", "TECHNICAL_FAILED"):
            raise ReleaseBlocked("All paths must be terminal before revealing test aggregates")
        if row["status"] == "TECHNICAL_FAILED" and not row.get("failure"):
            raise ReleaseBlocked("Technical terminal failure needs an explicit retained reason")
    return observed


def _write_once(path, content):
    path = Path(path)
    if path.exists():
        if path.read_text() != content:
            raise ReleaseBlocked(f"Existing released artifact changed: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(content)


def _json_once(root, name, value):
    _write_once(Path(root) / name, encoded(value) + "\n")


def _jsonl_once(root, name, rows):
    _write_once(Path(root) / name, "".join(encoded(r) + "\n" for r in rows))


def _compact(row, score, task):
    from .contract import fraction_of, load_json_strict

    output = {
        k: row[k]
        for k in (
            "slot_id",
            "model_id",
            "step",
            "pool",
            "protocol",
            "view",
            "qid",
            "root_id",
            "family",
            "chart",
            "variant",
            "draw",
            "monitor_block",
            "truncated",
        )
    }
    output["independent_score"] = score
    output["completion_token_count"] = len(row["tokens"])
    output["self_sampling_nll"] = float(-np.mean(row["old_logprobs"]))
    try:
        answer = load_json_strict(row["raw_text"])["answer"]
        value = None if answer is None else fraction_of(answer, text=True)
        output["parsed_answer"] = (
            None if value is None else (value.numerator if value.denominator == 1 else str(value))
        )
    except (ValueError, TypeError, KeyError, OverflowError, ZeroDivisionError):
        # Deliberately omit rather than using null, which is a valid answer.
        output["raw_text"] = ""
    if row["pool"] in ("MONITOR", "MONITOR_DIAGNOSTIC_ONLY"):
        from .analysis import response_semantics

        output["semantic_diagnostics"] = response_semantics(row["raw_text"], task, score)
    return output


def _score_retained_training_raw(raw, task, *, applied_update):
    """Retain technical missingness without converting it to a scientific zero."""
    from .json_protocol import score_record

    technical = raw.get("generation_status") != "COMPLETE" or raw.get("technical_validation_errors")
    error = "TECHNICAL_INVALID_GENERATION" if technical else None
    independent = None
    if not technical:
        try:
            independent = score_record(raw, task)
        except (PermissionError, KeyError, TypeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
    if error and applied_update:
        raise ReleaseBlocked(
            "Technically unscorable generation entered an applied update: " + error
        )
    return dict(
        independent_score=independent,
        technical_unscorable=error is not None,
        independent_scoring_error=error,
    )


def audit_training(root, matrix, tasks):
    """Recompute every retained scientific update from its actual raw completions."""
    from .contract import digest, reward_advantages
    from .json_protocol import AMENDMENT_ID, score_record, validate_record
    from .training import format_failure_accounting, token_path_record

    root = Path(root)
    amendment_id = AMENDMENT_ID if (root / "AMENDMENT.json").exists() else None
    summary = []
    for run_id, run in matrix.items():
        directory = root / "training" / run_id
        steps = sorted((directory / "steps").glob("*.json"))
        if run["status"] == "COMPLETE" and len(steps) != 96:
            raise ReleaseBlocked("Completed path lacks its 96 actual-update records")
        seen_steps = set()
        token_count = 0
        format_rounds = []
        for path in steps:
            update = json.loads(path.read_text())
            step = update["logical_step"]
            if step in seen_steps or not 1 <= step <= 96:
                raise ReleaseBlocked("Duplicate/out-of-budget scientific step")
            seen_steps.add(step)
            groups = []
            token_path = []
            files = sorted((directory / "rollouts").glob(f"{step:02d}-*.json"))
            if len(files) != 128:
                raise ReleaseBlocked("An applied update lacks its full 128 actual generations")
            by_slot = {}
            for raw_path in files:
                raw = json.loads(raw_path.read_text())
                if raw["record_hash"] != digest(
                    {k: v for k, v in raw.items() if k != "record_hash"}
                ):
                    raise ReleaseBlocked("Scientific raw evidence hash mismatch")
                if raw.get("generation_status") != "COMPLETE" or raw.get(
                    "technical_validation_errors"
                ):
                    raise ReleaseBlocked("Technically invalid generation entered science loss")
                validate_record(raw, amendment_id)
                if raw["run_id"] != run_id or raw["logical_step"] != step:
                    raise ReleaseBlocked("Training raw identity mismatch")
                key = (raw["slot"], raw["sample_index"])
                if key in by_slot:
                    raise ReleaseBlocked("Duplicated training trajectory")
                by_slot[key] = raw
            slot_ids = sorted({s for s, _ in by_slot})
            if len(slot_ids) != 16:
                raise ReleaseBlocked("Training prompt group count differs from 16")
            for slot in slot_ids:
                group = []
                for draw in range(8):
                    raw = by_slot[(slot, draw)]
                    task = tasks[raw["qid"]]
                    group.append(score_record(raw, task))
                    token_count += len(raw["tokens"])
                    token_path.append(token_path_record(raw))
                groups.append(group)
            if amendment_id and "format_failures" not in update:
                raise ReleaseBlocked("Amended training lacks per-round format accounting")
            if "format_failures" in update:
                accounting = format_failure_accounting(groups)
                if update["format_failures"] != accounting:
                    raise ReleaseBlocked("Training per-round format failures differ from raw")
                format_rounds.append(
                    dict(run_id=run_id, arm=run["arm"], logical_step=step, **accounting)
                )
            advantages, info = reward_advantages(run["arm"], groups)
            if update["scores"] != groups or update["reward_advantage_audit"] != info:
                raise ReleaseBlocked("Raw independent scoring differs from applied training credit")
            if not np.array_equal(
                np.asarray(update["final_advantages"]), np.asarray(advantages).ravel()
            ):
                raise ReleaseBlocked("Loss did not consume the exact final registered advantages")
            if update["token_path_hash"] != digest(token_path):
                raise ReleaseBlocked("Applied token-path hash differs from retained raw")
        all_raw = sorted((directory / "rollouts").glob("*.json"))
        raw_recounts = []
        for raw_path in all_raw:
            raw = json.loads(raw_path.read_text())
            if raw["record_hash"] != digest({k: v for k, v in raw.items() if k != "record_hash"}):
                raise ReleaseBlocked("Retained training raw evidence hash mismatch")
            task = tasks[raw["qid"]]
            raw_recounts.append(
                dict(
                    path=str(raw_path.relative_to(root)),
                    sha256=sha_file(raw_path),
                    request_id=raw["request_id"],
                    qid=raw["qid"],
                    generation_status=raw["generation_status"],
                    applied_update=raw["logical_step"] in seen_steps,
                    technical_errors=raw.get("technical_validation_errors", []),
                    **_score_retained_training_raw(
                        raw, task, applied_update=raw["logical_step"] in seen_steps
                    ),
                )
            )
        audit_path = f"score_audit/training/{run_id}.jsonl"
        _jsonl_once(root, audit_path, raw_recounts)
        summary.append(
            dict(
                run_id=run_id,
                status=run["status"],
                completed_updates=len(steps),
                verified_applied_generations=128 * len(steps),
                retained_raw_records=len(all_raw),
                completion_tokens_in_applied_updates=token_count,
                raw_recount_path=audit_path,
                raw_recount_sha256=sha_file(root / audit_path),
                failure=run.get("failure"),
                format_failures_by_round=format_rounds,
                unfavorable_results_retained=True,
            )
        )
    return summary


def audit_evaluation(root, tasks, fit, matrix):
    """Check the frozen full slot set, then independently score only actual answers."""
    root = Path(root)
    from .json_protocol import AMENDMENT_ID

    amendment_id = AMENDMENT_ID if (root / "AMENDMENT.json").exists() else None
    expected = {s["slot_id"]: s for s in iter_evaluation_slots(tasks, fit)}
    unavailable = {m for m, r in matrix.items() if r["status"] == "TECHNICAL_FAILED"}
    inputs = {r["qid"]: r for r in read_jsonl(root / "manifests/MODEL_INPUTS.jsonl")}
    diagnostics = {
        (r["qid"], r["view"]): r
        for r in read_jsonl(root / "manifests/DIAGNOSTIC_INPUTS_AUDIT_ONLY.jsonl")
    }
    compact, raw_index = [], []
    technical_invalid = []
    observed = set()
    for path in sorted((root / "raw/evaluation").glob("*.jsonl")):
        count = 0
        for row in read_jsonl(path):
            slot_id = row.get("slot_id")
            if slot_id in observed or slot_id not in expected:
                raise ReleaseBlocked("Unexpected/duplicated raw evaluation slot")
            slot = expected[slot_id]
            observed.add(slot_id)
            try:
                validate_raw(row, slot, protocol_amendment_id=amendment_id)
            except ValueError as exc:
                if row.get("model_id") not in unavailable:
                    raise ReleaseBlocked(f"Unresolved technical evaluation failure: {exc}") from exc
                technical_invalid.append(dict(slot_id=slot_id, reason=str(exc)))
                count += 1
                continue
            if row["view"]:
                request = diagnostics[(row["qid"], row["view"])]
                if row["information_granted"] != request["information_granted"]:
                    raise ReleaseBlocked("Diagnostic information grant changed")
                expected_text = request["text"]
                if row["view"] == "neutral_hint":
                    prefix = expected_text + "\n\n"
                    suffix = row["text"][len(prefix) :]
                    candidate = request["runtime_neutral_padding"]["candidate"]
                    if not row["text"].startswith(prefix) or not (
                        candidate * (len(suffix) // len(candidate) + 1)
                    ).startswith(suffix):
                        raise ReleaseBlocked(
                            "Neutral hint is not a prefix of its frozen neutral template"
                        )
                elif row["text"] != expected_text:
                    raise ReleaseBlocked("Diagnostic input text differs from frozen request")
                if row["image_file"] != request["image_file"]:
                    raise ReleaseBlocked("Diagnostic image channel changed")
            else:
                request = inputs[row["qid"]]
                expected_text = (
                    request["plain_text"] if row["protocol"] == "answer_only" else request["text"]
                )
                if row["text"] != expected_text or row["image_file"] != request["image_file"]:
                    raise ReleaseBlocked("Model input differs from the frozen gold-free prompt")
            actual_hash = hashlib.sha256(
                encoded(dict(text=row["text"], image_hash=row["image_hash"])).encode()
            ).hexdigest()
            if row["input_hash"] != actual_hash:
                raise ReleaseBlocked("Raw prompt identity hash mismatch")
            image_file = row["image_file"]
            if image_file is not None:
                image = (root / image_file).resolve()
                if not image.is_relative_to(root.resolve()) or sha_file(image) != row["image_hash"]:
                    raise ReleaseBlocked("Evaluation image identity mismatch")
            if (
                row["view"] == "neutral_hint"
                and abs(row.get("token_matching", {}).get("token_difference", 100)) > 1
            ):
                raise ReleaseBlocked("Neutral hint is not tokenizer matched")
            independent = score_response(
                row["raw_text"], tasks[row["qid"]], row["protocol"], record=row
            )
            if "score" in row and row["score"] != independent:
                raise ReleaseBlocked("Persisted evaluation score differs from raw-text recount")
            compact.append(_compact(row, independent, tasks[row["qid"]]))
            count += 1
        raw_index.append(
            dict(
                path=str(path.relative_to(root)),
                sha256=sha_file(path),
                rows=count,
                kind="synthetic_evaluation",
            )
        )
    missing = set(expected) - observed
    unresolved = [s for s in missing if expected[s]["model_id"] not in unavailable]
    if unresolved:
        raise ReleaseBlocked(
            f"Missing {len(unresolved)} required slots; technical missingness is not zero accuracy"
        )
    failures = Counter(expected[s]["model_id"] for s in missing)
    return (
        compact,
        raw_index,
        dict(
            status="RAW_RECOUNT_VERIFIED",
            expected_slots=len(expected),
            actual_generated_rows=len(compact),
            technical_invalid_rows=technical_invalid,
            missing_slots_by_failed_model=dict(failures),
            synthetic_complete=not missing and not technical_invalid,
            raw_text_independently_rescored=True,
            answer_only_has_no_P_or_J=True,
        ),
    )


def _audit_external(root, matrix):
    from .chartqa import iter_chartqa_slots, load_verified_manifest, score_chartqa

    root = Path(root)
    identity_path = root / "external/chartqa/IDENTITY.json"
    if not identity_path.exists():
        raise ReleaseBlocked("External availability must be attempted and explicitly recorded")
    identity = json.loads(identity_path.read_text())
    if identity["status"] == "UNAVAILABLE":
        return [], [], dict(identity, external_generalization_claim_allowed=False)
    manifest = load_verified_manifest(root)
    questions = {q["qid"]: q for q in manifest}
    models = [BASELINE, *list(matrix)]
    failed = {m for m, r in matrix.items() if r["status"] == "TECHNICAL_FAILED"}
    expected = {s["slot_id"]: s for m in models for s in iter_chartqa_slots(manifest, m)}
    observed, rows, index = set(), [], []
    for path in sorted((root / "raw/chartqa").glob("*.jsonl")):
        count = 0
        for row in read_jsonl(path):
            slot_id = row["slot_id"]
            if slot_id in observed or slot_id not in expected:
                raise ReleaseBlocked("Unexpected/duplicate external evaluation slot")
            validate_raw(row, expected[slot_id])
            observed.add(slot_id)
            score = score_chartqa(questions[row["qid"]]["answer"], row["raw_text"])
            rows.append(
                dict(
                    slot_id=slot_id,
                    model_id=row["model_id"],
                    qid=row["qid"],
                    subset=row["subset"],
                    image_id=row["image_id"],
                    independent_score=score,
                )
            )
            count += 1
        index.append(
            dict(
                path=str(path.relative_to(root)), sha256=sha_file(path), rows=count, kind="chartqa"
            )
        )
    missing = set(expected) - observed
    if any(expected[s]["model_id"] not in failed for s in missing):
        raise ReleaseBlocked("Available full ChartQA evaluation has missing slots")
    return (
        rows,
        index,
        dict(
            status="AVAILABLE_VERIFIED",
            identity_sha256=sha_file(identity_path),
            expected_slots=40000,
            generated_rows=len(rows),
            missing_slots=len(missing),
            external_generalization_claim_allowed=not missing,
        ),
    )


def amendment_report_lines(root, receipts):
    """Disclose the registered protocol and observed engineering coverage only."""
    if "AMENDMENT.json" not in receipts:
        return []
    from .amendment import AMENDMENT_ID, ORIGINAL_FREEZE_SHA256
    from .runtime import amended_format_decision

    root = Path(root)
    amendment = receipts["AMENDMENT.json"]
    common = receipts["FORMAT_AND_BRIDGE_RECEIPT.json"]
    metadata = common["metadata"]
    if (
        amendment.get("amendment_id") != AMENDMENT_ID
        or amendment.get("original_freeze_sha256") != ORIGINAL_FREEZE_SHA256
        or metadata.get("protocol_amendment") != amendment.get("protocol_amendment")
        or metadata.get("protocol_amendment_sha256") != sha_file(root / "AMENDMENT.json")
        or metadata.get("old_bridge_used_for_scientific_start") is not False
        or common.get("bridge_executed") is not False
        or common.get("bridge") is not None
        or common.get("status") != "COMPLETE"
    ):
        raise ReleaseBlocked("Amended common-start disclosure differs from registered evidence")
    before, confirmation = common["format_before"], common["format_confirmation"]
    for panel, count in ((before, 256), (confirmation, 512)):
        if (
            panel.get("responses") != count
            or type(panel.get("covered")) is not int
            or not 0 <= panel["covered"] <= count
            or panel.get("coverage") != panel["covered"] / count
        ):
            raise ReleaseBlocked("Amended format coverage numerator/denominator differs")
    decision = amended_format_decision(confirmation["covered"], confirmation["responses"])
    if decision == "F2_AMENDMENT_REQUIRED" or metadata.get("format_gate_decision") != decision:
        raise ReleaseBlocked("Amended confirmation gate does not permit scientific release")
    return [
        "",
        f"协议修正案：{AMENDMENT_ID}；AMENDMENT.json SHA-256："
        f"{sha_file(root / 'AMENDMENT.json')}；本轮冻结 SHA-256："
        f"{sha_file(root / 'EXECUTION_FREEZE.json')}。科学各臂及ENGINE统一学习率1e-4。",
        f"原SR-F1冻结 SHA-256：{ORIGINAL_FREEZE_SHA256}。历史工程FORMAT为93/256"
        "（36.328125%），原桥接后独立FORMAT_CONFIRM为100/256（39.0625%）。",
        f"SR-F1.1零LoRA起点实测FORMAT为{before['covered']}/{before['responses']}"
        f"（{before['coverage']:.6%}），FORMAT_CONFIRM为"
        f"{confirmation['covered']}/{confirmation['responses']}"
        f"（{confirmation['coverage']:.6%}），冻结门禁决定为{decision}。",
        "原SRF1_FORMAT_BRIDGED及128条gold监督曝光保留为历史工程记录；"
        "该adapter未用于本轮科学起点。本轮公共起点为已验证的零LoRA基座。",
        "预填字符{属于prompt；配平完成的最后一个生成token进入损失，不补EOS。"
        "answer_only协议保持原设定。",
        "",
    ]


def release(root, run_matrix_final):
    from .analysis import (
        absolute_panel_results,
        aggregate_results,
        diagnostic_results,
        diagnostic_view_contrasts,
        monitor_credit_audit,
        paired_variant_results,
        panel_contrasts,
    )
    from .chartqa import chartqa_summary

    root = Path(root).resolve()
    registered = json.loads((root / "manifests/RUN_MATRIX.json").read_text())
    matrix = terminal_matrix(run_matrix_final, registered)
    # An implementation or local renderer PASS cannot stand in for real execution.
    required = (
        "EXECUTION_FREEZE.json",
        "MODEL_ENVIRONMENT_IDENTITY.json",
        "SOURCE_AND_RENDER_MANIFEST.json",
        "FORMAT_AND_BRIDGE_RECEIPT.json",
        "ENGINE_PROBABILITY_GRADIENT_RESUME.json",
        "GPU_ACCOUNTING.json",
    )
    if (root / "AMENDMENT.json").exists():
        from .amendment import EFFECTIVE_CONFIG, effective_plan

        effective_plan(root)
        required += ("AMENDMENT.json", EFFECTIVE_CONFIG)
    receipts = {name: json.loads((root / name).read_text()) for name in required}
    amendment_disclosure = amendment_report_lines(root, receipts)
    if receipts["EXECUTION_FREEZE.json"].get("status") != "FROZEN":
        raise ReleaseBlocked("Real execution freeze is absent")
    if receipts["GPU_ACCOUNTING.json"].get("status") != "COMPLETE":
        raise ReleaseBlocked(
            "Actual GPU allocations and technical failures require complete accounting"
        )
    if receipts["ENGINE_PROBABILITY_GRADIENT_RESUME.json"].get("status") not in (
        "PASS_NATURAL_GRPO_RESUME",
        "PASS_NONZERO_SURROGATE_KERNEL_RESUME",
    ):
        raise ReleaseBlocked(
            "GPU probability/gradient/resume has not passed its registered contract"
        )
    tasks = {r["qid"]: r for r in read_jsonl(root / "manifests/TASKS_GOLD_AUDIT_ONLY.jsonl")}
    fit = json.loads((root / "manifests/TRAIN_FIT_QIDS.json").read_text())
    rows, index, audit = audit_evaluation(root, tasks, fit, matrix)
    training = audit_training(root, matrix, tasks)
    external_rows, external_index, external = _audit_external(root, matrix)
    failed = [m for m, r in matrix.items() if r["status"] == "TECHNICAL_FAILED"]
    scientific_matrix_complete = not failed and audit["synthetic_complete"]
    evidence_status = (
        "COMPLETE_REGISTERED_MATRIX" if scientific_matrix_complete else "PARTIAL_TECHNICAL_FAILURE"
    )
    primary = panel_contrasts(rows, unavailable_models=failed)
    transfer = []
    for pool in TEST_POOLS:
        transfer.extend(panel_contrasts(rows, pool, "answer_only", "A", failed))
        if pool != "TEST_ID":
            transfer.extend(panel_contrasts(rows, pool, "evidence_answer", "J", failed))
            transfer.extend(panel_contrasts(rows, pool, "evidence_answer", "A", failed))
    aggregates = aggregate_results(rows)
    absolute = absolute_panel_results(rows)
    credit = monitor_credit_audit(rows)
    diagnostic = diagnostic_results(rows)
    diagnostic_contrasts = diagnostic_view_contrasts(
        [r for r in rows if r["model_id"] not in failed]
    )
    variants = paired_variant_results(rows, tasks)
    external_results = chartqa_summary(external_rows) if external_rows else {}
    _json_once(
        root,
        "SCORE_AUDIT.json",
        dict(
            audit,
            training_raw_recount_verified=True,
            scientific_matrix_complete=scientific_matrix_complete,
            evidence_status=evidence_status,
        ),
    )
    _json_once(
        root,
        "RAW_OUTPUT_INDEX.json",
        dict(
            files=index + external_index,
            training=training,
            note=(
                "Training raw records and technical incidents remain under training; "
                "checkpoints are not a self-contained retraining bundle."
            ),
        ),
    )
    _json_once(
        root,
        "REWARD_ADVANTAGE_AUDIT.json",
        dict(
            monitor_blocks=credit,
            training=training,
            scope=(
                "actual advantages and truth-event credit; "
                "neither group rate nor credit mass is a parameter gradient"
            ),
        ),
    )
    _jsonl_once(root, "RESULTS_BY_SEED_FAMILY_VARIANT_PROTOCOL.jsonl", aggregates)
    _json_once(root, "RESULTS_ABSOLUTE.json", absolute)
    _json_once(
        root,
        "PRIMARY_CONTRASTS.json",
        dict(primary_metric="TEST_ID evidence_answer J", contrasts=primary),
    )
    _json_once(
        root,
        "DIAGNOSTIC_VIEWS.json",
        dict(summaries=diagnostic, contrasts=diagnostic_contrasts, variant_pairs=variants),
    )
    _jsonl_once(
        root, "MONITOR_SEMANTIC_EVENTS.jsonl", (r for r in rows if "semantic_diagnostics" in r)
    )
    _json_once(
        root,
        "TRANSFER_RESULTS.json",
        dict(contrasts=transfer, chartqa_availability=external, chartqa=external_results),
    )
    _json_once(
        root,
        "TRAINING_SUPPORT_AND_FAILURES.json",
        dict(
            runs=training,
            failed_models=failed,
            technical_missingness=audit["missing_slots_by_failed_model"],
            unfavorable_results_retained=True,
        ),
    )
    limitation = (
        "# 识别限制与下一步\n\n本轮比较固定反馈，未认证数据设计优于其他数据，"
        "也未认证诊断路由或慢尺度匹配。"
        "五项主对照的题根区间仅条件于已实现的训练路径；三个种子不是三个独立预训练模型。"
        "2pp为实际关注量级而非科学PASS门槛。普通问答只有A，不构造P/J。诊断真值视图授予额外信息，其差异不是纯感知或纯计算的内部机制认证。"
        "不得按本轮结果追加奖励、种子、支持训练或延长96步。\n\n"
        f"技术失败路径：{', '.join(failed) if failed else '无'}。"
        f"ChartQA状态：{external['status']}；公开预训练污染未排除。"
        "缺外部结果时不能作外部泛化声明。建议应结合普通问答、各迁移、成本与不利结果，建议不等于自动开始新任务。\n"
    )
    _write_once(root / "LIMITATIONS_AND_NEXT_DECISION_zh.md", limitation)
    report = [
        "# SR-F1 固定反馈实验最终报告",
        "",
        f"证据状态: {evidence_status}。技术收尾 COMPLETE 不等于科学矩阵完整。",
        "",
        "本轮检验答案、联合目标、固定部分奖励、GDPO式组件处理及门控精化对证据—答案联合正确性和普通问答迁移的影响。",
        "",
        f"已完成训练路径：{15 - len(failed)}/15；技术失败：{len(failed)}。"
        f"真实合成评价回答：{audit['actual_generated_rows']}。",
        "模型、桥接、GPU恢复、数据身份与计算成本分别见对应原始回执；"
        "终点固定96步，未选择best checkpoint。",
        "",
        "| 主对照 | J变化(pp) | 三种子变化(pp) | 95%区间 | 99%区间 |",
        "|---|---:|---|---|---|",
    ]
    report[2:2] = amendment_disclosure
    absolute_lookup = {
        (r["model_id"], r["protocol"]): r for r in absolute if r["pool"] == "TEST_ID"
    }
    absolute_table = [
        "",
        "| 模型 | J | A（证据协议） | E | P | 普通问答 A |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in [BASELINE, *matrix]:
        full = absolute_lookup.get((model, "evidence_answer"))
        plain = absolute_lookup.get((model, "answer_only"))
        if full and plain:
            absolute_table.append(
                f"| {model} | {full['J']:.4f} | {full['A']:.4f} | "
                f"{full['E']:.4f} | {full['P']:.4f} | {plain['A']:.4f} |"
            )
        else:
            absolute_table.append(
                f"| {model} | undefined | undefined | undefined | undefined | undefined |"
            )
    for contrast in primary:
        if contrast["status"] == "ESTIMATED":
            report.append(
                f"| {contrast['contrast']} | {contrast['effect_pp']:.3f} | "
                f"{contrast['seed_effects_pp']} | {contrast['CI95_pp']} | {contrast['CI99_pp']} |"
            )
        else:
            report.append(
                f"| {contrast['contrast']} | undefined | 技术缺失 | undefined | undefined |"
            )
    report.extend(absolute_table)
    format_rounds = [r for run in training for r in run["format_failures_by_round"]]
    if format_rounds:
        report.extend(
            [
                "",
                "各臂每轮格式失败率（单位：实际生成序列；分母每轮128）见 "
                "TRAINING_FORMAT_FAILURES_BY_ROUND.jsonl。格式失败保留在奖励和损失中。",
            ]
        )
    report.extend(["", "主要对照解释:"])
    for contrast in primary:
        if contrast["status"] != "ESTIMATED":
            report.append(f"- {contrast['contrast']}: 技术缺失，不能形成完整配对效应。")
            continue
        lo, hi = contrast["CI99_pp"]
        sign = "99%区间全为正" if lo > 0 else ("99%区间全为负" if hi < 0 else "99%区间跨越或触及0")
        report.append(
            f"- {contrast['contrast']}: {contrast['effect_pp']:+.3f}pp，{sign}；"
            f"种子效应范围 {contrast['seed_effect_range_pp']}pp。"
        )
    report.extend(["", "普通问答与逐池迁移:"])
    for contrast in transfer:
        label = (
            f"{contrast['pool']} / {contrast['protocol']} / "
            f"{contrast['metric']} / {contrast['contrast']}"
        )
        if contrast["status"] == "ESTIMATED":
            report.append(
                f"- {label}: {contrast['effect_pp']:+.3f}pp；"
                f"95%题根区间 {contrast['CI95_pp']}pp；种子 {contrast['seed_effects_pp']}pp。"
            )
        else:
            report.append(f"- {label}: undefined（技术缺失）。")
    report.extend(["", "MONITOR信用与支持:"])
    for model in [BASELINE, *matrix]:
        blocks = [b for b in credit if b["model_id"] == model]
        if not blocks:
            report.append(f"- {model}: 无可完整重算的MONITOR块。")
            continue
        for arm in ("A", "J", "PART", "DEC", "GATE"):
            active = sum(sum(b["arms"][arm]["final_advantage_nonconstant_groups"]) for b in blocks)
            total = 16 * len(blocks)
            positive = sum(
                b["arms"][arm]["positive_negative_credit"]["J"]["W_plus"] for b in blocks
            )
            negative = sum(
                b["arms"][arm]["positive_negative_credit"]["J"]["W_minus"] for b in blocks
            )
            report.append(
                f"- {model}/{arm}: 非恒定最终优势组 {active}/{total}；"
                f"J成功轨迹正信用总量 {positive:.4f}，负信用总量 {negative:.4f}。"
            )
    report.append("上述组率与信用总量是优化接口测量；不能替代真实J改善或参数梯度。")
    report.extend(
        [
            "",
            "原始A/E/P/J、字段覆盖、输出长度与自生成NLL见 "
            "RESULTS_BY_SEED_FAMILY_VARIANT_PROTOCOL.jsonl；各seed、题族、变体全部保留。",
            "普通问答、组合、表达和外观迁移逐池见 TRANSFER_RESULTS.json；"
            "不以训练reward代替主要成绩。",
            "MONITOR所有轨迹共用五反馈只读复算，DEC保留固定16提示块；"
            "真值事件正负信用及同分内方差见 REWARD_ADVANTAGE_AUDIT.json。",
            "五种诊断视图及题组双正确率与答案一致率分别见 DIAGNOSTIC_VIEWS.json。"
            "条件事件零支持记undefined而非0。",
            "成本见 GPU_ACCOUNTING.json；保留训练技术失败、真实无效回答及截断回答。",
            "",
            limitation,
        ]
    )
    _write_once(root / "FINAL_REPORT_zh.md", "\n".join(report) + "\n")
    _write_once(
        root / "START_HERE_zh.md",
        "# SR-F1 统一释放\n\n先读 [最终报告](FINAL_REPORT_zh.md)，"
        "再读 [识别限制](LIMITATIONS_AND_NEXT_DECISION_zh.md)。\n\n"
        "原始索引：[RAW_OUTPUT_INDEX.json](RAW_OUTPUT_INDEX.json)；"
        "重评分：[SCORE_AUDIT.json](SCORE_AUDIT.json)。\n技术验收不代表科学效果PASS。\n",
    )
    names = [
        "SCORE_AUDIT.json",
        "RAW_OUTPUT_INDEX.json",
        "REWARD_ADVANTAGE_AUDIT.json",
        "RESULTS_BY_SEED_FAMILY_VARIANT_PROTOCOL.jsonl",
        "RESULTS_ABSOLUTE.json",
        "PRIMARY_CONTRASTS.json",
        "DIAGNOSTIC_VIEWS.json",
        "MONITOR_SEMANTIC_EVENTS.jsonl",
        "TRANSFER_RESULTS.json",
        "TRAINING_SUPPORT_AND_FAILURES.json",
        "LIMITATIONS_AND_NEXT_DECISION_zh.md",
        "FINAL_REPORT_zh.md",
        "START_HERE_zh.md",
    ]
    if format_rounds:
        format_name = "TRAINING_FORMAT_FAILURES_BY_ROUND.jsonl"
        _jsonl_once(root, format_name, format_rounds)
        names.append(format_name)
    hashes = {name: sha_file(root / name) for name in [*required, *names, "RUN_MATRIX_FINAL.json"]}
    _json_once(root, "MANIFEST_SHA256.json", hashes)
    return dict(
        status="COMPLETE",
        artifacts=[*names, "MANIFEST_SHA256.json"],
        metadata=dict(
            unified_release=True,
            scientific_matrix_complete=scientific_matrix_complete,
            evidence_status=evidence_status,
            scientific_effect_is_not_a_technical_pass=True,
            synthetic_complete=audit["synthetic_complete"],
            failed_paths=len(failed),
        ),
    )
