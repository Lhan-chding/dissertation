"""Registered SR-F1.2 endpoint inference after all twelve main paths are verified."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from mm_core.training import state_hash
from sr_f1.chartqa import PROMPT, load_verified_manifest, score_chartqa
from sr_f1.evaluation import read_jsonl, sha_file
from sr_f12.evaluation import (
    durable_json,
    evaluate_shard,
    evaluation_account,
    score_record,
    verify_source_manifest,
)
from sr_f12.protocol import (
    AMENDMENT_ID,
    BASELINE,
    TEACHER_QOS,
    generation_config,
    iter_model_slots,
    object_hash,
    scientific_matrix,
    validate_config,
)


def read_json(path):
    return json.loads(Path(path).read_text())


def bounded_file(directory, relative):
    directory = Path(directory).resolve(strict=True)
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise PermissionError("Artifact path must be relative and contained")
    path = (directory / relative).resolve(strict=True)
    if not path.is_relative_to(directory) or not path.is_file():
        raise PermissionError("Artifact escapes registered directory")
    return path


def verify_main_completion(root, *, state_loader=None):
    """Authenticate actual final state and adapters, not just COMPLETE file existence."""
    from .runner import validate_technical_receipt
    from .training import CHECKPOINT_FIELDS, PLAN_ID

    root = Path(root).resolve(strict=True)
    freeze = read_json(root / "SCIENCE_FREEZE.json")
    plan = validate_config(root / "config/SR_F1_2_FROZEN.json")
    technical = read_json(root / "TECHNICAL_CHECK_SR_F1_2.json")
    predictions = read_json(root / "SEMANTIC_PREDICTIONS.json")
    validate_technical_receipt(technical, plan)
    if (
        freeze.get("status") != "FROZEN_BEFORE_SCIENCE"
        or freeze.get("amendment_id") != AMENDMENT_ID
        or freeze.get("main_matrix") != scientific_matrix()
        or freeze.get("config_sha256") != object_hash(plan)
        or freeze.get("technical_check_sha256") != object_hash(technical)
        or freeze.get("predictions_sha256") != object_hash(predictions)
    ):
        raise PermissionError("Science freeze, technical check or predictions differ")
    if state_loader is None:
        import torch

        def state_loader(path):
            return torch.load(path, map_location="cpu", weights_only=False)

    paths = []
    for run in scientific_matrix():
        directory = root / "runs" / run["model_id"]
        manifest = read_json(directory / "RUN_MANIFEST.json")
        done = read_json(directory / "COMPLETE.json")
        latest = read_json(directory / "checkpoints/LATEST.json")
        committed = read_json(directory / "checkpoints/commit-96.json")
        if (
            any(manifest.get(k) != v for k, v in run.items())
            or manifest.get("config_sha256") != object_hash(plan)
            or manifest.get("technical_only") is not False
            or done.get("status") != "COMPLETE"
            or done.get("step") != 96
            or done.get("technical_only") is not False
            or done.get("model_id") != run["model_id"]
            or done.get("checkpoint") != latest
            or latest != committed
            or latest.get("step") != 96
        ):
            raise PermissionError("Main matrix is not completely and consistently committed")
        checkpoint = bounded_file(directory / "checkpoints", committed["path"])
        if sha_file(checkpoint) != committed["sha256"]:
            raise PermissionError("Final checkpoint byte hash differs")
        state = state_loader(checkpoint)
        if (
            set(state) != CHECKPOINT_FIELDS
            or state["plan_id"] != PLAN_ID
            or state["committed_logical_step"] != 96
            or state["run_identity"] != manifest
            or state_hash(state) != committed["state_hash"]
            or committed["field_hashes"] != {k: state_hash(v) for k, v in state.items()}
            or state["cursor"] != dict(next_logical_step=97, next_slot=0, next_sample_index=0)
            or state["training_recipe"]["learning_rate"] != plan["training"]["lr"]
        ):
            raise PermissionError("Final complete training-state identity differs")
        if not state["optimizer"]["state"] or any(
            int(v["step"]) != 96 for v in state["optimizer"]["state"].values()
        ):
            raise PermissionError("Final Adam state does not contain 96 updates")
        adapter = directory / "adapters/step-96"
        identity = read_json(adapter / "IDENTITY.json")
        if (
            identity.get("step") != 96
            or identity.get("run_identity_sha256") != object_hash(manifest)
            or identity.get("policy_hash") != state_hash(state["parameters"])
            or not identity.get("files")
        ):
            raise PermissionError("Endpoint adapter is not the committed final policy")
        for name, checksum in identity["files"].items():
            if sha_file(bounded_file(adapter, name)) != checksum:
                raise PermissionError("Endpoint adapter file identity differs")
        for step in (32, 64, 96):
            fit_directory = directory / "train_fit" / f"step-{step:02d}"
            fit_done = read_json(fit_directory / "COMPLETE.json")
            records = []
            for index in range(32):
                group = read_json(fit_directory / f"{index:02d}.json")
                if (
                    group.get("sha256") != object_hash(group["records"])
                    or len(group["records"]) != 4
                ):
                    raise PermissionError("Training-fit raw group identity differs")
                records.extend(group["records"])
            if (
                fit_done.get("status") != "COMPLETE"
                or fit_done.get("count") != 128
                or fit_done.get("records_sha256") != object_hash(records)
            ):
                raise PermissionError("Required training-fit panel is incomplete")
        paths.append(
            dict(
                model_id=run["model_id"],
                complete_sha256=sha_file(directory / "COMPLETE.json"),
                checkpoint_sha256=committed["sha256"],
                adapter_identity_sha256=object_hash(identity),
            )
        )
        del state
    return dict(
        status="ALL_TWELVE_MAIN_PATHS_VERIFIED",
        science_freeze_sha256=object_hash(freeze),
        config_sha256=object_hash(plan),
        paths=paths,
    )


def verify_endpoint_authorization(root, *, slurm=None):
    """Recheck the registered acceptance and live job terminal evidence at every TEST entry.

    This is deliberately called by workers and score release, not only the
    controller. An old acceptance or fabricated COMPLETE marker cannot authorize
    TEST while a main job is active, failed, unknown, or has changed identity.
    """
    import getpass

    from .orchestration import Slurm
    from .source_revision import resolve_source_revision

    root = Path(root).resolve(strict=True)
    if (root / "STOP").exists():
        raise PermissionError("Experiment STOP exists")
    acceptance = read_json(root / "endpoint_orchestration/MAIN_ACCEPTANCE.json")
    primary = read_json(root / "orchestration/CONFIG.json")
    endpoint = read_json(root / "endpoint_orchestration/CONFIG.json")
    snapshot = read_json(root / "orchestration/STATE.json")
    source = resolve_source_revision(root)
    freeze = read_json(root / "SCIENCE_FREEZE.json")
    user = getpass.getuser()
    if (
        primary.get("user") != user
        or endpoint.get("user") != user
        or primary.get("root") != str(root)
        or endpoint.get("root") != str(root)
        or primary.get("version") != AMENDMENT_ID
        or endpoint.get("version") != AMENDMENT_ID
        or acceptance.get("status") != "ALL_TWELVE_TERMINAL_AND_STATE_VERIFIED"
        or acceptance.get("training_source_commit") != source["source_commit"]
        or freeze.get("source_commit") != source["source_commit"]
        or (
            "registration_source_commit" in acceptance
            and acceptance["registration_source_commit"] != source["registration_source_commit"]
        )
    ):
        raise PermissionError("Endpoint acceptance owner, run, or training-source binding differs")
    terminals = acceptance.get("terminals")
    models = [run["model_id"] for run in scientific_matrix()]
    if (
        not isinstance(terminals, list)
        or len(terminals) != 12
        or {row.get("model_id") for row in terminals} != set(models)
    ):
        raise PermissionError("Endpoint acceptance lacks the unique twelve main terminals")
    indexed = {row["model_id"]: row for row in terminals}
    backend = slurm or Slurm()
    queue = backend.queue(user)
    jobs = set()
    for model in models:
        directory = root / "orchestration/tasks" / model
        state = read_json(directory / "STATE.json")
        intent = read_json(directory / "INTENT.json")
        recorded = read_json(directory / "TERMINAL.json")
        job = state.get("job_id")
        cached_state = snapshot.get("tasks", {}).get(model, {})
        if (
            state.get("task_id") != model
            or state.get("status") != "COMPLETE"
            or cached_state.get("status") != "COMPLETE"
            or cached_state.get("job_id") != job
            or not isinstance(job, str)
            or not job.isdigit()
            or job in jobs
            or intent.get("task_id") != model
            or intent.get("gpu_count") != 1
            or intent.get("job_name") != "srf12-" + model
        ):
            raise PermissionError("Main scheduler registry is not uniquely complete")
        jobs.add(job)
        if job in queue:
            raise PermissionError("A main job remains in the live queue; TEST is sealed")
        expected = dict(
            job_id=job,
            state="COMPLETED",
            exit_code="0:0",
            user=user,
            account="rose",
            qos=TEACHER_QOS,
            name=intent["job_name"],
        )
        actual = backend.terminal(job)
        if actual != expected or recorded != expected or indexed[model].get("terminal") != expected:
            raise PermissionError("Main job live accounting/terminal identity is not successful")
        script = Path(intent["script"]).resolve(strict=True)
        if (
            not script.is_relative_to(directory.resolve())
            or sha_file(script) != intent["script_sha256"]
        ):
            raise PermissionError("Main job script identity changed")
        verified_path = directory / "VERIFIED_COMPLETE.json"
        if indexed[model].get("completion_receipt_sha256") != sha_file(verified_path):
            raise PermissionError("Accepted main completion receipt changed")
        verified = read_json(verified_path)
        done = read_json(root / "runs" / model / "COMPLETE.json")
        if (
            verified.get("task_id") != model
            or verified.get("receipt_sha256") != object_hash(done)
            or verified.get("checkpoint_sha256") != done.get("checkpoint", {}).get("sha256")
        ):
            raise PermissionError("Verified scheduler completion no longer binds the final state")
    result = verify_main_completion(root)
    if acceptance.get("main_completion") != result or acceptance.get(
        "main_completion_sha256"
    ) != object_hash(result):
        raise PermissionError("Endpoint acceptance no longer binds the complete training state")
    return result


def chartqa_panel(root, model_id):
    """The same pinned 2500 author records, each sampled once greedily."""
    manifest = load_verified_manifest(root)
    if len({r["qid"] for r in manifest}) != 2500:
        raise PermissionError("Duplicate ChartQA question")
    inputs, tasks, slots = {}, {}, []
    for item in manifest:
        image = bounded_file(root, item["image_file"])
        if sha_file(image) != item["image_sha256"]:
            raise PermissionError("ChartQA image hash differs")
        qid = item["qid"]
        inputs[qid] = dict(text=PROMPT + "\n" + item["question"], image_file=item["image_file"])
        tasks[qid] = item
        slots.append(
            dict(
                slot_id=f"{model_id}|CHARTQA|{qid}",
                model_id=model_id,
                step=0 if model_id == BASELINE else 96,
                pool="CHARTQA",
                protocol="plain_answer",
                view=None,
                qid=qid,
                root_id=item["image_id"],
                family="CHARTQA",
                chart="external",
                variant=item["subset"],
                draw=0,
                samples_per_prompt=1,
                sampling_seed=0,
                group_seed=0,
                generation=generation_config("plain_answer"),
            )
        )
    return slots, inputs, tasks


def read_completed_panel(directory):
    """Revalidate raw checksums before any released scoring."""
    directory = Path(directory)
    plan, done = read_json(directory / "PLAN.json"), read_json(directory / "COMPLETE.json")
    if (
        done.get("status") != "COMPLETE"
        or done.get("plan_sha256") != object_hash(plan)
        or len(done["groups"]) != len(plan["assigned_groups"])
        or {g["group_id"] for g in done["groups"]} != set(plan["assigned_groups"])
    ):
        raise PermissionError("Endpoint panel is incomplete")
    rows, seen = [], set()
    for group in done["groups"]:
        path = bounded_file(directory / "groups", group["group_id"] + ".json")
        data = read_json(path)
        if (
            sha_file(path) != group["raw_sha256"]
            or data["records_sha256"] != object_hash(data["records"])
            or len(data["records"]) != group["records"]
        ):
            raise PermissionError("Endpoint raw group byte/record identity differs")
        for row in data["records"]:
            if row["slot_id"] in seen:
                raise PermissionError("Duplicate endpoint slot")
            seen.add(row["slot_id"])
            rows.append(row)
    if len(rows) != plan["expected_records"] or len(rows) != done["generated_records"]:
        raise PermissionError("Endpoint raw coverage differs")
    return rows


def run_endpoint(root, model_id, *, source_manifest, code_root, model_path):
    """One GPU per endpoint; scheduler may run independent models on free teacher-QoS cards."""
    root = Path(root).resolve(strict=True)
    if (root / "STOP").exists():
        raise PermissionError("STOP requested")
    source_hash = verify_source_manifest(source_manifest, code_root)
    release = verify_endpoint_authorization(root)
    matrix = {r["model_id"] for r in scientific_matrix()}
    if model_id not in matrix | {BASELINE}:
        raise PermissionError("Unregistered main endpoint model")
    directory = root / "sealed/evaluation" / model_id
    durable_json(directory / "RELEASE_AUTHORIZATION.json", release)
    identity = read_json(root / "MODEL_ENVIRONMENT_IDENTITY.json")
    if Path(model_path).resolve(strict=True) != Path(identity["model_path"]).resolve(strict=True):
        raise PermissionError("Endpoint base path differs")
    from mm_dev.runtime import configure_audited_backend

    from .runtime import SRF12Runtime

    determinism = configure_audited_backend()
    adapter = root / "runs" / model_id / "adapters/step-96" if model_id != BASELINE else None
    runtime = SRF12Runtime(
        str(model_path),
        adapter_path=str(adapter) if adapter else None,
        device="cuda:0",
        output_root=root,
        account=evaluation_account(root, "endpoint-" + model_id),
    )
    common = None
    if adapter:
        adapter_identity = read_json(adapter / "IDENTITY.json")
        files = adapter_identity["files"]
        common = dict(
            adapter_file_hashes=files,
            model_hash=object_hash(
                dict(
                    base_model_weights_hash=identity["model_weights_hash"],
                    adapter_file_hashes=files,
                )
            ),
        )
    runtime.verify_identity(identity, common=common)
    runtime.identity.update(
        base_model_weights_hash=identity["model_weights_hash"],
        determinism=determinism,
        model_id=model_id,
        step=0 if model_id == BASELINE else 96,
    )
    if (
        adapter
        and runtime.current_adapter_identity()["adapter_parameter_hash"]
        != adapter_identity["policy_hash"]
    ):
        raise PermissionError("Actually loaded endpoint adapter differs from final state")
    # Input acquisition is deliberately after the complete-matrix gate.
    tasks = {r["qid"]: r for r in read_jsonl(root / "manifests/TASKS_GOLD_AUDIT_ONLY.jsonl")}
    inputs = {r["qid"]: r for r in read_jsonl(root / "manifests/MODEL_INPUTS.jsonl")}
    diagnostics = {
        (r["qid"], r["view"]): r
        for r in read_jsonl(root / "manifests/DIAGNOSTIC_INPUTS_AUDIT_ONLY.jsonl")
    }
    fit = read_json(root / "manifests/TRAIN_FIT_QIDS.json")
    panels = [
        (
            "final",
            list(iter_model_slots(tasks, fit, model_id, stage="final", test_released=True)),
            inputs,
            diagnostics,
            tasks,
        )
    ]
    if model_id == BASELINE:
        panels.append(
            (
                "train_fit",
                list(iter_model_slots(tasks, fit, model_id, stage="train_fit")),
                inputs,
                {},
                tasks,
            )
        )
    chart_slots, chart_inputs, chart_tasks = chartqa_panel(root, model_id)
    panels.append(("chartqa", chart_slots, chart_inputs, {}, chart_tasks))
    results = []
    for name, slots, source_inputs, source_diagnostics, source_tasks in panels:
        result = evaluate_shard(
            runtime,
            slots,
            source_inputs,
            source_diagnostics,
            source_tasks,
            root,
            directory / name,
            shard_index=0,
            shard_count=1,
            source_sha256=source_hash,
            score_non_test=False,
        )
        if result["status"] != "COMPLETE":
            return result
        results.append(
            dict(
                panel=name,
                records=result["generated_records"],
                completion_sha256=sha_file(directory / name / "COMPLETE.json"),
            )
        )
    expected = 16452 if model_id == BASELINE else 17604
    if sum(p["records"] for p in results) != expected:
        raise PermissionError("Final evaluation panel counts differ")
    receipt = dict(
        status="COMPLETE",
        model_id=model_id,
        records=expected,
        source_sha256=source_hash,
        main_completion_sha256=object_hash(release),
        panels=results,
        aggregate_scores_released=False,
    )
    durable_json(directory / "COMPLETE.json", receipt)
    return receipt


def release_scores(root, model_id):
    """Score validated outputs only after verifying the complete twelve-path gate again."""
    root = Path(root).resolve(strict=True)
    authorization = verify_endpoint_authorization(root)
    if model_id not in {row["model_id"] for row in scientific_matrix()} | {BASELINE}:
        raise PermissionError("Unregistered endpoint model")
    source = root / "sealed/evaluation" / model_id
    complete = read_json(source / "COMPLETE.json")
    if complete.get("status") != "COMPLETE" or complete.get(
        "main_completion_sha256"
    ) != object_hash(authorization):
        raise PermissionError("Evaluation was not bound to the complete main matrix")
    tasks = {r["qid"]: r for r in read_jsonl(root / "manifests/TASKS_GOLD_AUDIT_ONLY.jsonl")}
    chart = {r["qid"]: r for r in load_verified_manifest(root)}
    scored = []
    for panel in complete["panels"]:
        directory = source / panel["panel"]
        if sha_file(directory / "COMPLETE.json") != panel["completion_sha256"]:
            raise PermissionError("Endpoint completion receipt changed")
        rows = read_completed_panel(directory)
        if len(rows) != panel["records"]:
            raise PermissionError("Endpoint panel count changed")
        for row in rows:
            if row["pool"] == "CHARTQA":
                event = score_chartqa(chart[row["qid"]]["answer"], row["raw_text"])
            else:
                event = score_record(row, tasks[row["qid"]])
            scored.append(
                {
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
                    )
                }
                | dict(score=event)
            )
    receipt = dict(
        status="RELEASED_AFTER_ALL_MAIN_COMPLETE",
        model_id=model_id,
        authorization_sha256=object_hash(authorization),
        evaluation_sha256=object_hash(complete),
        count=len(scored),
        records=scored,
        records_sha256=object_hash(scored),
    )
    durable_json(root / "released/evaluation" / (model_id + ".json"), receipt)
    return receipt


def require_single_teacher_gpu():
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit():
        raise PermissionError("Slurm allocation required")
    result = subprocess.run(
        ["scontrol", "show", "job", job, "--oneliner"], capture_output=True, text=True, check=True
    )
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)", result.stdout))
    if fields.get("QOS") != TEACHER_QOS or fields.get("JobState") != "RUNNING":
        raise PermissionError("Endpoint requires running teacher-QoS allocation")
    import torch

    if torch.cuda.device_count() != 1:
        raise PermissionError("Endpoint process must use exactly one visible GPU")
