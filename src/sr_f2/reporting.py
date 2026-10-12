"""CPU-only SR-F2 release with live scheduler gates and complete coverage reporting.

Completion is a verified delivery state, never an inference from a COMPLETE filename.
No model is loaded, no model call is made, and no old score is overwritten here.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from sr_f1.chartqa import load_verified_manifest, score_chartqa
from sr_f1.contract import file_hash, load_json_strict
from sr_f1.data import load_tasks
from sr_f12.evaluation import durable_json, group_slots, shard_lease
from sr_f12.evaluation import score_record as score_v1_record
from sr_f12.protocol import BASELINE, TEACHER_QOS, iter_model_slots, object_hash
from sr_f12.protocol import scientific_matrix as old_matrix

from .analysis import FAMILIES, contrast, report_main_contrasts
from .diagnostics import interval_slots, protocol_panel, protocol_slots
from .evaluation import score_record
from .orchestration import Slurm, pool_specs, verify_task
from .protocol import AMENDMENT_ID, PARENT_ROOT, scientific_matrix, verify_registration
from .training import validate_adaptive_state
from .workers import endpoint_slots

FIELDS = (
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
)
METRICS = (
    "A",
    "E",
    "E6",
    "p_read",
    "P",
    "P6",
    "J",
    "J6",
    "L_json",
    "L_answer",
    "missing_count",
    "extra_count",
    "relaxed_accuracy",
    "exact_match",
)
ARMS = ("A", "J", "GATE", "DEC", "EARLY", "J6", "DEC_PRIME", "ADAPT")


def read_json(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise PermissionError(message)


def _contained(root, relative):
    root, relative = Path(root).resolve(strict=True), Path(relative)
    path = (root / relative).resolve(strict=True)
    require(
        not relative.is_absolute() and ".." not in relative.parts and path.is_relative_to(root),
        "Artifact path escapes its registered root",
    )
    return path


def pool_layout(specs):
    counts = Counter(spec["gpu_count"] for spec in specs)
    return " + ".join(f"{count} x {size} GPU" for size, count in sorted(counts.items()))


def verify_release_gate(root, *, slurm=None):
    """Fresh queue/accounting and exact global task ownership checks, every call."""
    from .orchestration import global_task_matrix, verify_pool_task_ownership

    root = Path(root).resolve(strict=True)
    registration = verify_registration(root, require_technical=True)
    registration_hash = object_hash(registration)
    status = read_json(root / "orchestration/STATUS.json")
    require(
        status.get("all_gpu_tasks_complete") is True
        and status.get("endpoint_seal_release_ready") is True,
        "All GPU tasks and all endpoint release prerequisites must complete",
    )
    config = read_json(root / "orchestration/CONFIG.json")
    specs, task_matrix = pool_specs(), global_task_matrix()
    require(
        config.get("root") == str(root)
        and config.get("pools") == specs
        and config.get("global_tasks") == task_matrix
        and config.get("global_task_matrix_sha256") == object_hash(task_matrix)
        and config.get("strict_seed_waves") is False,
        "Dynamic orchestration pool/global task registration differs",
    )
    require(
        len(specs) > 0
        and len({spec["pool_id"] for spec in specs}) == len(specs)
        and len(task_matrix) == 53
        and len({(task["kind"], task["model_id"]) for task in task_matrix}) == 53,
        "All registered resident pools and53 unique registered tasks are required",
    )
    require(
        set(status.get("pools", {})) == {spec["pool_id"] for spec in specs},
        "Scheduler status omits or adds a registered pool",
    )
    slurm = slurm or Slurm()
    queue = slurm.queue()
    pools, seen_jobs, completed_pools = [], set(), []
    for spec in specs:
        directory = root / "orchestration/pools" / spec["pool_id"]
        state, intent, plan, complete = [
            read_json(directory / name)
            for name in ("STATE.json", "INTENT.json", "PLAN.json", "COMPLETE.json")
        ]
        job = state.get("job_id")
        require(
            state.get("status") == "COMPLETE"
            and state.get("pool_id") == spec["pool_id"]
            and state.get("gpu_count") == spec["gpu_count"]
            and state.get("registration_sha256") == registration_hash
            and status["pools"][spec["pool_id"]] == state,
            "Pool is incomplete, unknown, failed, or differs from scheduler status",
        )
        require(
            isinstance(job, str) and job.isdigit() and job not in seen_jobs,
            "Each completed pool needs one unique known job identifier",
        )
        seen_jobs.add(job)
        require(job not in queue, "A registered GPU job is still present in squeue")
        require(
            intent.get("registration_sha256") == registration_hash
            and intent.get("pool_id") == spec["pool_id"]
            and intent.get("plan_sha256") == object_hash(plan)
            and all(plan.get(k) == v for k, v in spec.items())
            and plan.get("registration_sha256") == registration_hash,
            "Pool submission intent or task plan differs",
        )
        terminal = slurm.terminal(job)
        expected = dict(
            job_id=job,
            state="COMPLETED",
            exit_code="0:0",
            user=config["user"],
            account="rose",
            qos=TEACHER_QOS,
            name=intent["job_name"],
        )
        require(
            isinstance(terminal, dict) and all(terminal.get(k) == v for k, v in expected.items()),
            "Fresh sacct is missing, unsuccessful, or bound to another allocation",
        )
        require(
            complete.get("status") == "COMPLETE"
            and complete.get("plan_sha256") == object_hash(plan)
            and isinstance(complete.get("tasks"), list),
            "Pool completion task evidence differs",
        )
        completed_pools.append((spec["pool_id"], complete["tasks"]))
        pools.append(
            dict(
                pool_id=spec["pool_id"],
                job_id=job,
                accounting=terminal,
                absent_from_live_queue=True,
                intent_sha256=object_hash(intent),
                pool_completion_sha256=object_hash(complete),
                tasks=complete["tasks"],
            )
        )
    all_tasks = [verify_task(root, task, registration_hash) for task in task_matrix]
    expected_tasks = {(task["kind"], task["model_id"]): task for task in all_tasks}
    claimed = set()
    for pool_id, receipts in completed_pools:
        verify_pool_task_ownership(root, pool_id, receipts)
        for receipt in receipts:
            key = (receipt.get("kind"), receipt.get("model_id"))
            require(
                key in expected_tasks and key not in claimed and receipt == expected_tasks[key],
                "Pool claims a duplicated, unregistered, or changed global task",
            )
            claimed.add(key)
    require(claimed == set(expected_tasks), "Pool union does not cover all53 global tasks")
    expected_new = {row["model_id"] for row in scientific_matrix()}
    for kind in ("train", "endpoint"):
        require(
            {task["model_id"] for task in all_tasks if task["kind"] == kind} == expected_new,
            "All twelve training and endpoint workers are required",
        )
    return dict(
        status="AUTHORIZED_AFTER_ALL_GPU_TASKS_VERIFIED",
        experiment_id=AMENDMENT_ID,
        registration_sha256=registration_hash,
        pools=pools,
        tasks=all_tasks,
        live_queue_and_accounting_verified=True,
        new_endpoint_records=183600,
        scheduling=(
            f"{pool_layout(specs)} resident pools; shared53-task readiness queue; "
            "no seed-wave barrier"
        ),
        technical_completion_is_not_scientific_confirmation=True,
    )


def _chart_slots(chart, model_id, step):
    return [
        dict(
            slot_id=f"{model_id}|CHARTQA|{r['qid']}",
            model_id=model_id,
            step=step,
            pool="CHARTQA",
            protocol="plain_answer",
            view=None,
            qid=r["qid"],
            root_id=r["image_id"],
            family="CHARTQA",
            chart="external",
            variant=r["subset"],
            draw=0,
            samples_per_prompt=1,
            sampling_seed=0,
            group_seed=0,
            generation=dict(
                do_sample=False,
                temperature=1,
                top_p=1,
                top_k=0,
                max_new_tokens=128,
                num_return_sequences=1,
            ),
        )
        for r in chart
    ]


def validated_panel_rows(directory, expected_slots, *, model_id, source_sha256):
    """Reconstruct exact frozen slot membership; counts alone cannot release a panel."""
    directory = Path(directory).resolve(strict=True)
    complete, plan = read_json(directory / "COMPLETE.json"), read_json(directory / "PLAN.json")
    groups = group_slots(expected_slots)
    expected_ids = [g["group_id"] for g in groups]
    require(
        complete.get("status") == "COMPLETE"
        and complete.get("model_id") == model_id
        and complete.get("records") == len(expected_slots),
        "Panel completion differs",
    )
    require(
        plan.get("expected_records") == len(expected_slots)
        and plan.get("slots_sha256") == object_hash(expected_slots)
        and plan.get("source_sha256") == source_sha256,
        "Panel frozen slots or source differs",
    )
    refs = complete.get("groups", [])
    require([r.get("group_id") for r in refs] == expected_ids, "Panel exact group order differs")
    require(
        {p.stem for p in (directory / "groups").glob("*.json")} == set(expected_ids)
        and not any((directory / "groups").glob("*.pending.jsonl")),
        "Unregistered or partial raw groups remain",
    )
    for group, ref in zip(groups, refs, strict=True):
        path = _contained(directory, "groups/" + ref["group_id"] + ".json")
        raw = read_json(path)
        rows = raw.get("records", [])
        require(
            file_hash(path) == ref["raw_sha256"]
            and raw.get("group_id") == group["group_id"]
            and object_hash(rows) == ref["records_sha256"] == raw.get("records_sha256")
            and len(rows) == len(group["slots"]),
            "Panel raw checksum or group count differs",
        )
        for index, (row, slot) in enumerate(zip(rows, group["slots"], strict=True)):
            require(
                all(row.get(k) == v for k, v in slot.items())
                and row.get("source_sha256") == source_sha256
                and row.get("generation_status") == "COMPLETE"
                and not row.get("technical_validation_errors"),
                "Raw record is technically incomplete or differs from its frozen slot",
            )
            tokens = row.get("tokens")
            require(
                isinstance(tokens, list)
                and tokens
                and row.get("completion_token_count") == len(tokens)
                and len(row.get("sampler_logprobs", [])) == len(tokens),
                "Raw token trajectory is incomplete",
            )
            yield row, dict(path=str(path), sha256=ref["raw_sha256"], row_index=index)


def _compact(row, source, v2, v1=None):
    result = {k: row[k] for k in FIELDS}
    result.update(
        score=v2,
        source=source,
        completion_token_count=row["completion_token_count"],
        finish_reason=row.get("finish_reason"),
        truncated=row.get("truncated"),
    )
    if v1 is not None:
        result["score_v1"] = v1
    if row["protocol"] == "evidence_answer":
        try:
            payload = load_json_strict(row["raw_text"])
            evidence = payload.get("evidence") if isinstance(payload, dict) else None
            result["reported_evidence_items"] = (
                len(evidence) if isinstance(evidence, list) else None
            )
        except (ValueError, TypeError, RecursionError, OverflowError):
            result["reported_evidence_items"] = None
    return result


def _write_jsonl_once(path, rows):
    """Stage once and publish without overwrite; interrupted stages are not a release."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            for row in rows:
                stream.write(
                    json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            require(file_hash(path) == file_hash(temp), "Existing released score bytes differ")
        else:
            os.link(temp, path)
    finally:
        os.unlink(temp)
    return file_hash(path)


def publish_bytes(path, body):
    """Publish text/figure bytes atomically; an existing artifact must be identical."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            require(file_hash(path) == file_hash(temporary), "Existing report artifact differs")
        else:
            os.link(temporary, path)
    finally:
        os.unlink(temporary)


def score_new_endpoints(root, registration, tasks, chart):
    chart_map = {r["qid"]: r for r in chart}
    rows = []
    for model in [r["model_id"] for r in scientific_matrix()]:
        slots = endpoint_slots(tasks, model) + _chart_slots(chart, model, 96)
        require(len(slots) == 15300, "Frozen full endpoint must contain15300 records")
        for raw, source in validated_panel_rows(
            root / "sealed/evaluation" / model,
            slots,
            model_id=model,
            source_sha256=registration["source_manifest_sha256"],
        ):
            task = chart_map[raw["qid"]] if raw["pool"] == "CHARTQA" else tasks[raw["qid"]]
            v1 = (
                score_chartqa(task["answer"], raw["raw_text"])
                if raw["pool"] == "CHARTQA"
                else score_v1_record(raw, task)
            )
            rows.append(_compact(raw, source, score_record(raw, task), v1))
    require(len(rows) == 183600, "Incomplete new endpoint coverage")
    return rows


def load_old_endpoints(root, tasks, chart):
    """Bind the prior CPU v1/v2 audit and recover token lengths from authenticated raw."""
    directory = Path(root) / "prior_audit/rescore"
    summary, path = read_json(directory / "SUMMARY.json"), directory / "AUDIT.jsonl"
    require(
        summary.get("status") == "PASS"
        and summary.get("phase") == "rescore"
        and summary.get("result", {}).get("records") == 227700
        and summary["result"].get("v1_score_mismatches") == 0,
        "Prior full v1/v2 audit must pass with zero official-v1 mismatches",
    )
    require(
        file_hash(path) == summary["audit_jsonl_sha256"]
        and file_hash(directory / "SOURCE_MANIFEST.json") == summary["source_manifest_sha256"],
        "Prior rescore output or source manifest hash differs",
    )
    prior_root = Path(summary["root"]).resolve(strict=True)
    require(prior_root == Path(PARENT_ROOT).resolve(strict=True), "Wrong prior experiment root")
    fit = read_json(Path(root) / "manifests/TRAIN_FIT_QIDS.json")
    expected = {}
    for model in [BASELINE, *[r["model_id"] for r in old_matrix()]]:
        slots = list(iter_model_slots(tasks, fit, model, stage="final", test_released=True))
        if model == BASELINE:
            slots += list(iter_model_slots(tasks, fit, model, stage="train_fit"))
        slots += _chart_slots(chart, model, 0 if model == BASELINE else 96)
        expected.update({s["slot_id"]: {k: s[k] for k in FIELDS} for s in slots})
    require(len(expected) == 227700, "Prior registered endpoint matrix differs")
    rows, seen, cached_path, cached = [], set(), None, None
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("kind") != "endpoint_score":
                continue
            key = record["slot_id"]
            require(
                key in expected
                and key not in seen
                and all(record.get(k) == v for k, v in expected[key].items()),
                "Prior endpoint slot missing, duplicated, or mismatched",
            )
            seen.add(key)
            source = record["source"]
            raw_path = Path(source["path"]).resolve(strict=True)
            require(raw_path.is_relative_to(prior_root), "Historical raw escapes prior experiment")
            if raw_path != cached_path:
                require(
                    file_hash(raw_path) == source["sha256"], "Prior raw bytes changed after audit"
                )
                cached_path, cached = raw_path, read_json(raw_path)
                require(
                    object_hash(cached["records"]) == cached["records_sha256"],
                    "Prior raw record checksum differs",
                )
            raw = cached["records"][source["row_index"]]
            require(raw["slot_id"] == key, "Prior source row index differs")
            rows.append(_compact(raw, source, record["score_v2"], record["score_v1"]))
    require(seen == set(expected), "Prior endpoint coverage is incomplete")
    return rows, dict(
        summary_sha256=file_hash(directory / "SUMMARY.json"),
        audit_sha256=summary["audit_jsonl_sha256"],
        records=len(rows),
        historical_official_scores_unchanged=True,
        requested_chartqa_cohort=summary["result"].get(
            "requested_chartqa_cohort",
            {"status": "UNRESOLVED", "reason": "No verified requested122 cohort definition"},
        ),
    )


def cohort_verification(root):
    """The requested post-hoc122 cohort has its own identity and raw-delivery status."""
    directory = Path(root) / "prior_audit/chartqa122"
    path = directory / "COHORT_VERIFICATION.json"
    if not path.exists():
        return dict(
            status="UNRESOLVED",
            cohort_identity_verified=False,
            raw_evidence_status="UNKNOWN",
            reason="Cohort verification receipt absent",
        )
    receipt = read_json(path)
    if receipt.get("status") != "PASS":
        return dict(
            status="UNRESOLVED",
            cohort_identity_verified=False,
            receipt=receipt,
            receipt_sha256=file_hash(path),
            raw_evidence_status=receipt.get("raw_evidence_status", "UNKNOWN"),
        )
    require(
        receipt.get("cohort_identity_verified") is True
        and receipt.get("count") == 122
        and receipt.get("scheme") == "whitespace_stripped"
        and receipt.get("metric") == "relaxed_accuracy"
        and receipt.get("label") == "POST_HOC_WHITESPACE_SENSITIVITY"
        and receipt.get("official_frozen_scores_changed") is False,
        "Requested122 cohort receipt does not verify the declared post-hoc definition",
    )
    expected_models = {BASELINE, *[run["model_id"] for run in old_matrix()]}
    require(
        receipt.get("baseline_model") == BASELINE
        and set(receipt.get("loss_models", []))
        == {"SRF1_2_GATE_s71001", "SRF1_2_GATE_s71003", "SRF1_2_DEC_s71003"},
        "Requested122 baseline or three-model loss definition differs",
    )
    require(
        file_hash(directory / "SOURCE_MANIFEST.json") == receipt["source_manifest_sha256"],
        "Requested122 source manifest differs",
    )
    reference = receipt["rows"]
    rows_path = _contained(directory, reference["path"])
    require(
        file_hash(rows_path) == reference["sha256"] and reference["count"] == 122,
        "Requested122 cohort row evidence hash/count differs",
    )
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    require(
        len(rows) == 122 and len({row["qid"] for row in rows}) == 122,
        "Requested122 cohort must contain122 unique question identifiers",
    )
    if receipt.get("raw_evidence_status") == "RAW_AND_GOLD_REVERIFIED":
        require(
            set(receipt.get("raw_model_ids", [])) == expected_models
            and receipt.get("raw_output_count") == 1586,
            "Requested122 full evidence requires all13 models and1586 raw outputs",
        )
        for row in rows:
            require(
                isinstance(row.get("gold"), dict) and set(row.get("models", {})) == expected_models,
                "Requested122 gold or all-model coverage missing",
            )
            for model in row["models"].values():
                require(
                    isinstance(model.get("raw_text"), str)
                    and isinstance(model.get("tokens"), list)
                    and model["tokens"]
                    and isinstance(model.get("raw_source"), dict)
                    and all(
                        key in model["raw_source"]
                        for key in ("path", "sha256", "records_sha256", "row_index")
                    ),
                    "Requested122 raw model output or provenance missing",
                )
    return dict(
        status="PASS",
        cohort_identity_verified=True,
        count=122,
        receipt=receipt,
        receipt_sha256=file_hash(path),
        raw_evidence_status=receipt.get("raw_evidence_status", "UNKNOWN"),
        interpretation="Post-hoc whitespace sensitivity cohort; not frozen official scores. "
        "Identity PASS does not imply complete raw evidence delivery.",
    )


def summarize_rows(rows):
    buckets = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k) for k in ("model_id", "pool", "protocol", "view", "family"))
        buckets[key].append(row)
    results = []
    for key, records in sorted(buckets.items(), key=lambda x: str(x[0])):
        item = dict(zip(("model_id", "pool", "protocol", "view", "family"), key, strict=True))
        item.update(
            count=len(records),
            questions=len({r["qid"] for r in records}),
            mean_completion_tokens=sum(r["completion_token_count"] for r in records) / len(records),
        )
        for field in ("score", "score_v1"):
            metrics = {}
            for name in METRICS:
                values = [r[field][name] for r in records if field in r and name in r[field]]
                if values:
                    require(
                        len(values) == len(records)
                        and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values),
                        "Metric is partially missing or nonfinite",
                    )
                    metrics[name] = sum(values) / len(values)
            if metrics:
                item["v2" if field == "score" else "v1"] = metrics
        results.append(item)
    return results


def _rate(rows, model, pool="TEST_ID", protocol="evidence_answer", metric="A", family=None):
    selected = [
        r
        for r in rows
        if r["model_id"] == model
        and r["pool"] == pool
        and r["protocol"] == protocol
        and r.get("view") is None
        and (family is None or r["family"] == family)
    ]
    require(selected, "Missing requested model/pool/protocol score support")
    per_family = defaultdict(list)
    for row in selected:
        per_family[row["family"]].append(row["score"][metric])
    return sum(sum(values) / len(values) for values in per_family.values()) / len(per_family)


def load_trajectories(root):
    trajectories, states, usage, training_metrics = [], {}, {}, {}
    for run in [*old_matrix(), *scientific_matrix()]:
        model, new = run["model_id"], run["model_id"].startswith("SRF2_")
        base = root / ("runs" if new else "history") / model
        if new:
            final = read_json(base / "monitor_state/step-96.json")
            state = final["adaptive_state"]
        else:
            final = read_json(base / "COMPLETE.json")
            state = final["trigger_state"]
        validate_adaptive_state(state)
        require(state["last_monitor_step"] == 96, "Incomplete final trigger trajectory")
        states[model] = state
        for step in range(0, 97, 8):
            folder = base / "monitor" if new else base
            path = folder / f"step-{step:02d}" / "COMPLETE.json"
            measure = read_json(path)
            require(
                measure.get("model_id") == model
                and measure.get("step") == step
                and measure.get("count") == 512
                and measure.get("status") == "COMPLETE",
                "Monitor trajectory coverage differs",
            )
            recorded_measure = dict(state["history"][step // 8]["measurement"])
            # Runner times the whole call after the evaluator publishes its receipt.
            recorded_measure.pop("monitor_seconds", None)
            require(recorded_measure == measure, "Trigger state differs from measured trajectory")
            for family, values in measure["families"].items():
                trajectories.append(
                    dict(
                        model_id=model,
                        arm=run["arm"],
                        seed=run["seed"],
                        step=step,
                        family=family,
                        **values,
                    )
                )
        if new:
            steps = [read_json(base / "steps" / f"{step:02d}.json") for step in range(1, 97)]
            require(
                [s["step"] for s in steps] == list(range(1, 97))
                and all(s["arm"] == run["arm"] and s["groups_total"] == 16 for s in steps),
                "Training step metric coverage differs",
            )
            # Every committed eight-step checkpoint binds that step's diagnostics.
            for step in range(8, 97, 8):
                commit = read_json(base / "checkpoints" / f"commit-{step:02d}.json")
                from mm_core.training import state_hash

                require(
                    commit["field_hashes"]["diagnostics_hash"]
                    == state_hash(object_hash(steps[step - 1])),
                    "Committed step metrics hash differs",
                )
            if run["arm"] == "ADAPT":
                for update in steps:
                    decisions = state["history"][(update["step"] - 1) // 8]["decisions"]
                    active = sorted(f for f in FAMILIES if decisions[f]["active"])
                    audit = update["reward_advantage_audit"]
                    require(
                        update["active_families"] == active
                        and update["semantic_groups_used"] == 4 * len(active)
                        and audit["group_reward_arms"].count("J") == 4 * len(active)
                        and len(audit["group_reward_arms"]) == 16,
                        "ADAPT semantic supervision usage differs from measured trigger routing",
                    )
            training_metrics[model] = steps
            usage[model] = dict(
                semantic_groups=sum(s["semantic_groups_used"] for s in steps),
                total_groups=sum(s["groups_total"] for s in steps),
            )
    require(len(trajectories) == 24 * 13 * 4, "Full eight-arm trajectory table is incomplete")
    return trajectories, states, usage, training_metrics


def trigger_summary(states, old_rows):
    matrix, confusion = [], Counter()
    old_ids = {r["model_id"] for r in old_matrix()}
    for model, state in sorted(states.items()):
        for family in FAMILIES:
            r1 = [h["step"] for h in state["history"] if h["decisions"][family]["R1"]]
            r2 = [h["step"] for h in state["history"] if h["decisions"][family]["R2"]]
            item = dict(
                model_id=model,
                family=family,
                R1_steps=r1,
                R2_steps=r2,
                first_R1=min(r1) if r1 else None,
                first_R2=min(r2) if r2 else None,
                active_at_end=family in state["active_families"],
            )
            if model in old_ids:
                reference = "SRF1_2_J_s" + model.rsplit("_s", 1)[1]
                deficit = 100 * (
                    _rate(old_rows, reference, metric="J", family=family)
                    - _rate(old_rows, model, metric="J", family=family)
                )
                observed, flagged = deficit > 5, bool(r1 or r2)
                item.update(
                    endpoint_J_deficit_vs_J_pp=deficit,
                    endpoint_deficit_strictly_gt_5pp=observed,
                    R1_or_R2_ever_triggered=flagged,
                )
                confusion[
                    "TP"
                    if observed and flagged
                    else "FN"
                    if observed
                    else "FP"
                    if flagged
                    else "TN"
                ] += 1
            matrix.append(item)
    tp, fp, fn = confusion["TP"], confusion["FP"], confusion["FN"]
    return dict(
        matrix=matrix,
        historical_descriptive_confusion=dict(confusion),
        precision=None if tp + fp == 0 else tp / (tp + fp),
        recall=None if tp + fn == 0 else tp / (tp + fn),
        label="endpoint family J more than5pp below seed-matched J control",
        interpretation="descriptive historical association; no causal or prospective validation",
    )


def diagnostic_results(root, registration, tasks, old_rows):
    rows = []
    models = [BASELINE, *[r["model_id"] for r in old_matrix()]]
    for model in models:
        slots = protocol_slots(tasks, model) + interval_slots(tasks, model)
        for raw, source in validated_panel_rows(
            root / "diagnostics" / model,
            slots,
            model_id=model,
            source_sha256=registration["source_manifest_sha256"],
        ):
            rows.append(_compact(raw, source, score_record(raw, tasks[raw["qid"]])))
    require(len(rows) == 34944, "Protocol/INTERVAL diagnostic coverage is incomplete")
    selected = {t["qid"] for t in protocol_panel(tasks)}
    matched = [
        r
        for r in old_rows
        if r["pool"] == "TEST_ID"
        and r.get("view") is None
        and r["qid"] in selected
        and r["protocol"] in ("evidence_answer", "answer_only")
    ]
    require(len(matched) == 13 * 512 * 6, "Matched historical protocol controls are incomplete")
    tables, predictions = [], []
    baseline_free = _rate(rows, BASELINE, protocol="free")
    for model in models:
        row = dict(model_id=model)
        for protocol in ("evidence_answer", "answer_only", "free", "list"):
            source = rows if protocol in ("free", "list") else matched
            part = [
                r
                for r in source
                if r["model_id"] == model and r["pool"] == "TEST_ID" and r["protocol"] == protocol
            ]
            row[protocol] = dict(
                A=_rate(source, model, protocol=protocol),
                count=len(part),
                parse_rate=sum(r["score"]["L_answer"] for r in part) / len(part),
                mean_tokens=sum(r["completion_token_count"] for r in part) / len(part),
            )
        tables.append(row)
        if model != BASELINE:
            predictions.append(
                dict(
                    model_id=model,
                    free_gain_vs_start_pp=100 * (row["free"]["A"] - baseline_free),
                    free_gain_at_least_25pp=row["free"]["A"] - baseline_free >= 0.25,
                    free_minus_answer_only_pp=100 * (row["free"]["A"] - row["answer_only"]["A"]),
                    absolute_gap_below_10pp=abs(row["free"]["A"] - row["answer_only"]["A"]) < 0.1,
                )
            )
    interval = []
    for model in models:
        value = dict(model_id=model)
        for view in ("V0", "V1", "V2", "V3", "V4"):
            part = [r for r in rows if r["model_id"] == model and r["view"] == view]
            require(len(part) == 128, "INTERVAL view requires32questionsK4")
            value[view] = dict(
                A=sum(r["score"]["A"] for r in part) / len(part),
                count=len(part),
                parse_rate=sum(r["score"]["L_answer"] for r in part) / len(part),
                mean_tokens=sum(r["completion_token_count"] for r in part) / len(part),
            )
        value["predictions"] = dict(
            V2_at_least_50pct=value["V2"]["A"] >= 0.5,
            V3_at_least_90pct=value["V3"]["A"] >= 0.9,
            V2_below_20pct_and_V3_at_least_90pct=value["V2"]["A"] < 0.2 and value["V3"]["A"] >= 0.9,
            V3_below_50pct=value["V3"]["A"] < 0.5,
            V1_minus_V0_pp=100 * (value["V1"]["A"] - value["V0"]["A"]),
            V1_approximately_V0_status="NO_NUMERIC_EQUIVALENCE_MARGIN_REGISTERED",
        )
        interval.append(value)
    return rows, dict(
        records=len(rows),
        protocol_questions=512,
        protocol_table=tables,
        protocol_predictions=predictions,
        interval_table=interval,
        all_model_family_metrics=summarize_rows(rows),
    )


def secondary_predictions(new_rows, old_rows, states, steps):
    early = {f: contrast(new_rows, old_rows, "EARLY", "A", f) for f in ("CROSS", "INTERVAL")}
    for value in early.values():
        value["mean_effect_at_least_2pp"] = value["effect_pp"] >= 2
    j6 = {}
    for f in ("CROSS", "THRESHOLD", "INTERVAL"):
        value = contrast(new_rows, old_rows, "J6", "J", f)
        value["mean_abs_effect_at_most_2pp"] = abs(value["effect_pp"]) <= 2
        j6[f] = value
    j6_answer = {}
    for family in ("CROSS", "THRESHOLD", "INTERVAL"):
        value = contrast(new_rows, old_rows, "J6", "A", family)
        value["mean_abs_effect_at_most_2pp"] = abs(value["effect_pp"]) <= 2
        j6_answer[family] = value
    evidence_counts = {}
    for seed in (71001, 71002, 71003):
        model = f"SRF2_J6_s{seed}"
        counts = Counter(
            str(r.get("reported_evidence_items"))
            for r in new_rows
            if r["model_id"] == model
            and r["pool"] == "TEST_ID"
            and r["protocol"] == "evidence_answer"
            and r["family"] == "TOPK"
        )
        evidence_counts[model] = dict(
            counts=counts,
            six_is_unique_mode=counts.get("6", 0)
            > max((n for key, n in counts.items() if key != "6"), default=0),
            concentration_threshold="NOT_NUMERICALLY_REGISTERED",
        )
    dec = {}
    baseline_chart = _rate(
        old_rows, BASELINE, pool="CHARTQA", protocol="plain_answer", metric="relaxed_accuracy"
    )
    for seed in (71001, 71002, 71003):
        model = f"SRF2_DEC_PRIME_s{seed}"
        records = steps[model]
        zero = sum(s["reward_advantage_audit"]["zero_signal_step"] for s in records)
        late = sum(s["reward_advantage_audit"]["contrast_groups"] for s in records[48:]) / 48
        groups = defaultdict(list)
        for row in new_rows:
            if (
                row["model_id"] == model
                and row["pool"] == "TEST_ID"
                and row["protocol"] == "evidence_answer"
            ):
                groups[row["qid"]].append(row["score"])
        require(
            len(groups) == 2048 and all(len(g) == 4 for g in groups.values()),
            "Four-draw failure diagnostic coverage differs",
        )
        all_wrong = {
            metric: sum(all(s[metric] == 0 for s in group) for group in groups.values())
            / len(groups)
            for metric in ("A", "J")
        }
        drop = 100 * (
            baseline_chart
            - _rate(
                new_rows, model, pool="CHARTQA", protocol="plain_answer", metric="relaxed_accuracy"
            )
        )
        dec[model] = dict(
            zero_signal_steps=zero,
            zero_steps_at_most15=zero <= 15,
            late48_mean_contrast_groups=late,
            late_contrast_at_least1_8=late >= 1.8,
            four_wrong_rates=all_wrong,
            four_wrong_at_most6_5pct={k: v <= 0.065 for k, v in all_wrong.items()},
            chartqa_drop_pp=drop,
            chartqa_drop_at_most2pp=drop <= 2,
        )
    trigger_predictions = {}
    for model, state in states.items():
        first = {
            f: next((h["step"] for h in state["history"] if h["decisions"][f]["R1"]), None)
            for f in FAMILIES
        }
        r2_cells = [f for f in FAMILIES if any(h["decisions"][f]["R2"] for h in state["history"])]
        item = dict(first_R1=first, R2_ever_families=r2_cells)
        if "_A_s" in model or "_ADAPT_s" in model:
            item.update(
                threshold_at0=first["THRESHOLD"] == 0,
                topk_by32=first["TOPK"] is not None and first["TOPK"] <= 32,
                cross_interval_never=first["CROSS"] is None and first["INTERVAL"] is None,
            )
        if model.startswith("SRF1_2_"):
            item["R2_only_A_TOPK_pattern"] = (
                r2_cells == ["TOPK"] if "_A_s" in model else not r2_cells
            )
            if any("_" + arm + "_s" in model for arm in ("J", "GATE", "DEC")):
                item["no_R1_after_step8"] = not any(
                    h["decisions"][f]["R1"]
                    for h in state["history"]
                    if h["step"] > 8
                    for f in FAMILIES
                )
        trigger_predictions[model] = item
    return dict(
        early_cross_interval=early,
        j6_other_families_J=j6,
        j6_other_families_A=j6_answer,
        j6_other_family_metric_ambiguity="Document does not specify A or J; both are reported",
        j6_topk_evidence_item_counts=evidence_counts,
        dec_prime=dec,
        trigger_predictions=trigger_predictions,
        semantics="predictions reported individually; no extra training or threshold tuning",
    )


def advantage_energy_report(root, new_steps, registration):
    """Observed final advantage energy as a descriptive covariate, without fitted regression."""
    manifest = read_json(registration["prior_sources"])
    expected = {entry["canonical_path"]: entry["sha256"] for entry in manifest["files"]}
    records, summaries = [], []
    for run in [*old_matrix(), *scientific_matrix()]:
        model = run["model_id"]
        entries = []
        for step in range(1, 97):
            if model in new_steps:
                metric = new_steps[model][step - 1]
                path = root / "runs" / model / "steps" / f"{step:02d}.json"
            else:
                path = (Path(PARENT_ROOT) / "runs" / model / "steps" / f"{step:02d}.json").resolve(
                    strict=True
                )
                require(
                    expected.get(str(path)) == file_hash(path),
                    "Historical advantage step bytes differ from frozen audit",
                )
                metric = read_json(path)
            require(
                metric["step"] == step and metric["arm"] == run["arm"],
                "Advantage energy step identity differs",
            )
            coefficients = metric["reward_advantage_audit"]["final_advantage"]
            require(
                len(coefficients) == 16 and all(len(group) == 8 for group in coefficients),
                "Advantage energy requires complete16x8 coefficients",
            )
            energy = sum(float(value) ** 2 for group in coefficients for value in group)
            require(math.isfinite(energy), "Nonfinite advantage energy")
            if "advantage_energy" in metric:
                require(
                    abs(energy - metric["advantage_energy"]) < 1e-8,
                    "Recorded advantage energy differs from coefficients",
                )
            item = dict(
                model_id=model,
                arm=run["arm"],
                seed=run["seed"],
                step=step,
                advantage_sum_squares=energy,
                source=dict(path=str(path), sha256=file_hash(path)),
            )
            records.append(item)
            entries.append(item)
        phases = {}
        for name, selected in (
            ("steps1_12", entries[:12]),
            ("steps13_96", entries[12:]),
            ("steps1_96", entries),
        ):
            values = [row["advantage_sum_squares"] for row in selected]
            phases[name] = dict(
                steps=len(values),
                mean=sum(values) / len(values),
                total=sum(values),
                zero_steps=sum(value == 0 for value in values),
            )
        summaries.append(dict(model_id=model, arm=run["arm"], seed=run["seed"], phases=phases))
    return dict(
        records=records,
        per_seed_phases=summaries,
        role="descriptive covariate; no added causal regression or experiment",
        definition="sum of128 squared final training advantage coefficients per update",
    )


def prediction_table(results):
    """Render every named secondary/diagnostic prediction without hiding adverse seeds."""
    secondary = results.get("secondary", {})
    entries = []

    def add(subject, prediction, observed, met):
        entries.append((subject, prediction, observed, "MET" if met else "NOT_MET"))

    for family, row in secondary.get("early_cross_interval", {}).items():
        add(
            "EARLY " + family,
            "A-J >=2pp",
            f"{row['effect_pp']:.3f}pp",
            row["mean_effect_at_least_2pp"],
        )
    for metric in ("A", "J"):
        for family, row in secondary.get("j6_other_families_" + metric, {}).items():
            add(
                "J6 " + family,
                f"abs({metric}-J控制臂) <=2pp",
                f"{row['effect_pp']:.3f}pp",
                row["mean_abs_effect_at_most_2pp"],
            )
    for model, row in secondary.get("j6_topk_evidence_item_counts", {}).items():
        entries.append(
            (
                model,
                "报告项数集中在6",
                str(dict(row["counts"])),
                "SIX_IS_MODE" if row["six_is_unique_mode"] else "SIX_NOT_UNIQUE_MODE",
            )
        )
    for model, row in secondary.get("dec_prime", {}).items():
        add(model, "零信号步 <=15", row["zero_signal_steps"], row["zero_steps_at_most15"])
        add(
            model,
            "后48步有对比组均值 >=1.8",
            f"{row['late48_mean_contrast_groups']:.3f}",
            row["late_contrast_at_least1_8"],
        )
        for metric in ("A", "J"):
            add(
                model,
                metric + "四次全错 <=6.5%",
                _percent(row["four_wrong_rates"][metric]),
                row["four_wrong_at_most6_5pct"][metric],
            )
        add(
            model,
            "ChartQA下降 <=2pp",
            f"{row['chartqa_drop_pp']:.3f}pp",
            row["chartqa_drop_at_most2pp"],
        )
    for model, row in secondary.get("trigger_predictions", {}).items():
        for key in (
            "threshold_at0",
            "topk_by32",
            "cross_interval_never",
            "R2_only_A_TOPK_pattern",
            "no_R1_after_step8",
        ):
            if key in row:
                observed = row["R2_ever_families"] if key.startswith("R2_") else row["first_R1"]
                add(model, key, str(observed), row[key])
    for row in results["diagnostics"].get("protocol_predictions", []):
        add(
            row["model_id"],
            "FREE比起点 >=25pp",
            f"{row['free_gain_vs_start_pp']:.3f}pp",
            row["free_gain_at_least_25pp"],
        )
        add(
            row["model_id"],
            "abs(FREE-answer_only) <10pp",
            f"{row['free_minus_answer_only_pp']:.3f}pp",
            row["absolute_gap_below_10pp"],
        )
    for row in results["diagnostics"]["interval_table"]:
        prediction = row.get("predictions", {})
        for key in (
            "V2_at_least_50pct",
            "V3_at_least_90pct",
            "V2_below_20pct_and_V3_at_least_90pct",
            "V3_below_50pct",
        ):
            if key in prediction:
                add(
                    row["model_id"],
                    key,
                    "V2=" + _percent(row["V2"]["A"]) + ", V3=" + _percent(row["V3"]["A"]),
                    prediction[key],
                )
    return [
        "## 次要与诊断预测逐项核对",
        "",
        "| 模型/范围 | 预测 | 观测 | 状态 |",
        "|---|---|---|---|",
        *["| " + " | ".join(map(str, row)) + " |" for row in entries],
        "",
        "模式匹配仅是描述性结果。'集中'与V1≈V0没有登记数值等价界; 不自动转写为通过。",
        "",
    ]


def plot_trajectories(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(4, 8, figsize=(24, 11), sharex=True, sharey=True)
    colors = dict(A="#2766a2", E="#e69b00", P="#36935b", J="#a44786")
    for fi, family in enumerate(FAMILIES):
        for ai, arm in enumerate(ARMS):
            axis = axes[fi, ai]
            selected = [r for r in rows if r["family"] == family and r["arm"] == arm]
            for metric, color in colors.items():
                values = np.asarray(
                    [
                        [
                            next(
                                r[metric]
                                for r in selected
                                if r["step"] == step and r["seed"] == seed
                            )
                            for step in range(0, 97, 8)
                        ]
                        for seed in (71001, 71002, 71003)
                    ]
                )
                axis.plot(range(0, 97, 8), values.mean(0), color=color, label=metric, linewidth=1.5)
                axis.fill_between(
                    range(0, 97, 8), values.min(0), values.max(0), color=color, alpha=0.10
                )
            if fi == 0:
                axis.set_title(arm)
            if ai == 0:
                axis.set_ylabel(family + "\nProbability")
            if fi == 3:
                axis.set_xlabel("Update")
            axis.set_ylim(0, 1.03)
            axis.set_xticks([0, 32, 64, 96])
            axis.grid(alpha=0.18)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4)
    fig.suptitle("SR-F2 MONITOR-lite: seed mean; shaded area is seed minimum-maximum", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, metadata={"Software": "SR-F2 frozen reporting"})
    plt.close(fig)


def _percent(value):
    return f"{100 * value:.2f}%"


def render_report(results):
    cohort = results["requested_chartqa_cohort"]
    if cohort["status"] == "PASS":
        cohort_text = (
            "指定ChartQA122题集合的身份已核验(PASS)。选择使用事后whitespace_stripped敏感性分数, "
            "三模型为GATE_s71001、GATE_s71003、DEC_s71003; 不属于冻结正式成绩。"
            + " 原始证据交付状态: "
            + cohort["raw_evidence_status"]
            + "; 身份通过不等同于原文交付完整。"
        )
    else:
        cohort_text = (
            "指定ChartQA122题集合仍为UNRESOLVED。未获得通过校验的集合身份证据; "
            "CPU审计候选集合仅作诊断, 不能替代该指定集合。"
        )
    lines = [
        "# SR-F2 实验报告",
        "",
        "状态: `COMPLETE_COVERAGE_VERIFIED`。工程完成与科学对比是否成立分别报告。",
        "",
        "已核验全部 GPU 作业离开队列且 sacct 为 COMPLETED / 0:0; 全部冻结任务和原始组通过校验。",
        "新终点183,600条; 旧终点227,700条按v1/v2并列; 诊断34,944条; 新旧轨迹各79,872条。",
        f"按用户授权采用{pool_layout(pool_specs())}常驻池与共享53任务队列; 无整波等待。"
        "数据、模型、奖励、种子和训练步数保持冻结定义。",
        "FREE/LIST使用用户确认的512题、v0/v3、K2。旧正式成绩未改写; "
        "所有种子、不利结果及技术记录保留。",
        cohort_text,
        "",
        "## 四个主对比",
        "",
        "| 对比 | 差值(pp) | 98.75%区间(pp) | 三种子差值(pp) | 判定 | 预测 |",
        "|---|---:|---|---|---|---|",
    ]
    for key, value in results["main"]["contrasts"].items():
        status = value.get("primary_status", value.get("status", "UNKNOWN"))
        prediction = value.get("criteria_status", str(value.get("prediction_met")))
        lines.append(
            f"| {key} {value['contrast']} | {value['effect_pp']:.3f} | "
            f"{value['CI98_75_pp']} | {value['seed_effects_pp']} | {status} | {prediction} |"
        )
    lines += [
        "",
        "## 终点v1/v2并列",
        "",
        "| 模型 | TEST_ID A v1/v2 | TEST_ID J v1/v2 | E8 / E6(v2) | ChartQA v1/v2 |",
        "|---|---:|---:|---:|---:|",
    ]
    metrics = results["endpoint_metrics"]
    for model in sorted({r["model_id"] for r in metrics}):
        selected = [
            r
            for r in metrics
            if r["model_id"] == model
            and r["pool"] == "TEST_ID"
            and r["protocol"] == "evidence_answer"
            and r["view"] is None
        ]
        chart = next(r for r in metrics if r["model_id"] == model and r["pool"] == "CHARTQA")

        def average(version, metric, selected=selected):
            return sum(r[version][metric] for r in selected) / len(selected)

        pairs = [
            _percent(average("v1", name)) + "/" + _percent(average("v2", name))
            for name in ("A", "J")
        ]
        pairs.append(_percent(average("v2", "E")) + "/" + _percent(average("v2", "E6")))
        pairs.append(
            _percent(chart["v1"]["relaxed_accuracy"])
            + "/"
            + _percent(chart["v2"]["relaxed_accuracy"])
        )
        lines.append("| " + model + " | " + " | ".join(pairs) + " |")
    lines += [
        "",
        "C4按预登记-2pp非劣界判定; 允许区间跨零。"
        "其附加标准、分种子结果与语义监督占比见results.json。",
        "次要预测逐项保存于results.json的secondary字段; 未登记数值等价界的“≈”不转写为通过。",
        "",
        "## FREE / LIST 与同512题历史对照",
        "",
        "| 模型 | evidence A | answer_only A | FREE A | LIST A | FREE解析率 | FREE/LIST平均token |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results["diagnostics"]["protocol_table"]:
        lines.append(
            "| "
            + row["model_id"]
            + " | "
            + " | ".join(
                _percent(row[p]["A"]) for p in ("evidence_answer", "answer_only", "free", "list")
            )
            + f" | {_percent(row['free']['parse_rate'])} | "
            + f"{row['free']['mean_tokens']:.1f}/{row['list']['mean_tokens']:.1f} |"
        )
    lines += [
        "",
        "历史evidence为K4, 其余三协议K2; 题目支持相同。生成长度与解析失败均计入报告。",
        "",
        "## INTERVAL-mean 五视图",
        "",
        "| 模型 | V0 | V1 | V2 | V3 | V4 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results["diagnostics"]["interval_table"]:
        lines.append(
            "| "
            + row["model_id"]
            + " | "
            + " | ".join(_percent(row[v]["A"]) for v in ("V0", "V1", "V2", "V3", "V4"))
            + " |"
        )
    lines += [
        "",
        "V4按原生分词器与V1等长; 完整恢复率、解析率、token数和预登记阈值结果见results.json。",
        "",
        "## R1 / R2 首次触发矩阵",
        "",
        "| 模型 | CROSS R1/R2 | THRESHOLD R1/R2 | TOPK R1/R2 | INTERVAL R1/R2 |",
        "|---|---|---|---|---|",
    ]
    matrix = results["triggers"]["matrix"]
    for model in sorted({r["model_id"] for r in matrix}):
        cells = [
            next(r for r in matrix if r["model_id"] == model and r["family"] == f) for f in FAMILIES
        ]
        lines.append(
            "| "
            + model
            + " | "
            + " | ".join(f"{r['first_R1']}/{r['first_R2']}" for r in cells)
            + " |"
        )
    lines += [
        "",
        "`None`为未触发。R1动作只在ADAPT实施; 历史和其他新臂的触发仅作记录。",
        "历史精确率/召回率仅描述与终点差距的关系, 不代表因果解释或新的前瞻验证。",
        "",
        "## 新旧八臂轨迹",
        "",
        "![四族八臂轨迹](trajectories.png)",
        "",
        "线为三种子均值, 阴影为种子最小至最大值。全部逐种子逐时点指标保存在results.json。",
        "",
        "## v1/v2、覆盖与技术记录",
        "",
        "SCORES.jsonl保存新终点逐条v2 score、v1 score_v1、来源路径/SHA和token数; "
        "OLD_SCORES.jsonl保存旧终点并列值。",
        "results.json的endpoint_metrics包含全部模型 x 池 x 协议 x 题族的v1/v2并列表, "
        "未删去任何失利种子。",
        f"技术事件文件数: {len(results['technical_incidents'])}。"
        "原始错误记录留在technical_incidents及各任务日志。",
        "[原CPU证据审计JSONL](../prior_audit/rescore/AUDIT.jsonl)保存GATE异常组、漏项和语法分类。",
        "[指定122题原文与金标](../prior_audit/chartqa122/COHORT_ROWS.jsonl)保留全部13模型输出。",
        "本报告不新增奖励、训练臂、种子或诊断矩阵, 也不据结果触发SR-F3。",
        "",
    ]
    lines += prediction_table(results)
    lines += [
        "## 优势能量描述性协变量",
        "",
        "| 模型 | 第1-12步均值 | 第13-96步均值 | 全96步均值 |",
        "|---|---:|---:|---:|",
    ]
    for row in results.get("advantage_energy", {}).get("per_seed_phases", []):
        phases = row["phases"]
        lines.append(
            "| "
            + row["model_id"]
            + " | "
            + " | ".join(
                f"{phases[key]['mean']:.4f}" for key in ("steps1_12", "steps13_96", "steps1_96")
            )
            + " |"
        )
    lines += [
        "",
        "逐种子逐步128项最终优势平方和保存于results.json。仅作描述性协变量, 未新增因果回归。",
        "",
    ]
    return "\n".join(lines)


def release_and_report(root, *, slurm=None):
    root = Path(root).resolve(strict=True)
    with shard_lease(root / "released"):
        cohort = cohort_verification(root)
        require(
            cohort["status"] == "PASS"
            and cohort["raw_evidence_status"] == "RAW_AND_GOLD_REVERIFIED",
            "Requested122 identity and complete raw/gold delivery "
            "are required before final completion",
        )
        authorization = verify_release_gate(root, slurm=slurm)
        authorization = dict(
            authorization, requested_chartqa_cohort_sha256=cohort["receipt_sha256"]
        )
        durable_json(root / "released/AUTHORIZATION.json", authorization)
        registration = read_json(root / "REGISTRATION.json")
        tasks, chart = load_tasks(root), load_verified_manifest(root)
        new_rows = score_new_endpoints(root, registration, tasks, chart)
        old_rows, old_receipt = load_old_endpoints(root, tasks, chart)
        new_hash = _write_jsonl_once(root / "released/SCORES.jsonl", new_rows)
        old_hash = _write_jsonl_once(root / "released/OLD_SCORES.jsonl", old_rows)
        trajectories, states, usage, steps = load_trajectories(root)
        diag_rows, diagnostics = diagnostic_results(root, registration, tasks, old_rows)
        diagnostic_hash = _write_jsonl_once(root / "released/DIAGNOSTIC_SCORES.jsonl", diag_rows)
        main = report_main_contrasts(
            new_rows, old_rows, trigger_states=states, semantic_group_usage=usage
        )
        incidents = [
            dict(path=str(p.resolve(strict=True).relative_to(root)), sha256=file_hash(p))
            for p in sorted((root / "technical_incidents").glob("*.json"))
        ]
        results = dict(
            status="COMPLETE_COVERAGE_VERIFIED",
            experiment_id=AMENDMENT_ID,
            experiment_complete=True,
            authorization_sha256=object_hash(authorization),
            source_manifest_sha256=registration["source_manifest_sha256"],
            coverage=dict(
                new_endpoint_records=len(new_rows),
                old_endpoint_records=len(old_rows),
                diagnostics=len(diag_rows),
                trajectory_cells=len(trajectories),
                historical_monitor_records=79872,
                new_monitor_records=79872,
            ),
            score_files=dict(
                new_sha256=new_hash, old_sha256=old_hash, diagnostic_sha256=diagnostic_hash
            ),
            prior_rescore=old_receipt,
            requested_chartqa_cohort=cohort,
            main=main,
            endpoint_metrics=summarize_rows([*new_rows, *old_rows]),
            diagnostics=diagnostics,
            triggers=trigger_summary(states, old_rows),
            secondary=secondary_predictions(new_rows, old_rows, states, steps),
            semantic_group_usage=usage,
            advantage_energy=advantage_energy_report(root, steps, registration),
            trajectories=trajectories,
            technical_incidents=incidents,
            scientific_confirmation_separate_from_delivery=True,
        )
        require(
            cohort_verification(root) == cohort,
            "Requested122 verification or delivery changed during report construction",
        )
        durable_json(root / "released/results.json", results)
        report = render_report(results)
        publish_bytes(root / "released/REPORT.md", report.encode())
        figure = root / "released/trajectories.png"
        fd, temporary = tempfile.mkstemp(prefix=".trajectories.", suffix=".png", dir=figure.parent)
        os.close(fd)
        try:
            plot_trajectories(trajectories, temporary)
            publish_bytes(figure, Path(temporary).read_bytes())
        finally:
            Path(temporary).unlink()
        # A final completion receipt is published only after all deliverables exist.
        delivery = dict(
            status="COMPLETE_COVERAGE_VERIFIED",
            experiment_complete=True,
            authorization_sha256=object_hash(authorization),
            coverage=results["coverage"],
            files=[
                dict(
                    path=str(p.resolve(strict=True).relative_to(root)),
                    sha256=file_hash(p),
                    bytes=p.stat().st_size,
                )
                for p in (
                    root / "released" / name
                    for name in (
                        "AUTHORIZATION.json",
                        "SCORES.jsonl",
                        "OLD_SCORES.jsonl",
                        "DIAGNOSTIC_SCORES.jsonl",
                        "results.json",
                        "REPORT.md",
                        "trajectories.png",
                        "../prior_audit/chartqa122/COHORT_VERIFICATION.json",
                        "../prior_audit/chartqa122/COHORT_ROWS.jsonl",
                        "../prior_audit/chartqa122/SOURCE_MANIFEST.json",
                    )
                )
            ],
        )
        durable_json(root / "released/COMPLETE.json", delivery)
        return delivery
