"""Synthetic read-only historical reuse qualification; never call a model."""

from __future__ import annotations

import copy
import json

import pytest

from ssvc_flow.src.exposure_position import legacy_trajectory as lt
from ssvc_flow.src.exposure_position import schema
from ssvc_flow.src.exposure_substitution.matched_tasks import make_task


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def history(tmp_path, monkeypatch):
    run, legacy = tmp_path / "new", tmp_path / "legacy"
    source_tasks, tasks, mapping = [], [], []
    for i in range(800):
        root_index = i // 4 if i < 128 else i
        center, j = ((3, 3), (3, 1), (3, 2), (2, 2))[i % 4]
        world, observed = [10, 20, 30, 40], [10, 20, 30, 40]
        observed[j] += 5
        task, _ = make_task(world, observed, center, j, f"source-root-{root_index}", "E_CONFIRM", i)
        source_tasks.append(task)
        if i < 128:
            new = {
                **copy.deepcopy(task),
                "phase_id": schema.PHASE_ID,
                "task_id": f"{i + 1000:032x}",
                "root_id": f"{root_index + 1000:032x}",
                "base_instance_id": f"{root_index + 1000:032x}",
                "split": "DEV_TRAJECTORY",
            }
            tasks.append(new)
            mapping.append(
                dict(
                    panel="DEV_TRAJECTORY",
                    task_id=new["task_id"],
                    root_id=new["root_id"],
                    source_task_id=task["task_id"],
                    source_root_id=task["root_id"],
                )
            )
    write_lines(run / "manifests/DEV_TRAJECTORY/tasks_public.jsonl", tasks)
    write_lines(run / "manifests/development_source_mapping.jsonl", mapping)
    audit = {key: "synthetic-" + key for key in lt.AUDIT_KEYS}
    audit.update(
        model_id="Qwen/Qwen3.5-9B",
        model_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        base_dtype="bfloat16",
        probability_execution="uncached_prefix_recompute",
        eos_token_ids=[7, 8],
        lora_modules=[f"layer-{i}" for i in range(96)],
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0,
    )
    environment = {"python": "synthetic-python", "torch": "synthetic-torch"}
    bindings = dict(
        legacy_run=str(legacy), runtime_environment=environment, parents={}, checkpoint_lookup={}
    )
    bridge = dict(status="PASS", execution_kind="REAL_CUDA_BRIDGE", runtime_receipts=[])
    old_jobs, fingerprints = [], {}
    for parent in schema.PARENTS:
        path = legacy / f"{parent}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(parent.encode())
        checkpoint = dict(
            path=str(path),
            sha256=schema.file_digest(path),
            identity={"parent": parent},
            state_hash="full-state",
        )
        historical_identity = dict(model_hash="snapshot-sha", environment=environment)
        fingerprint = lt.digest(
            dict(parent=parent, checkpoint=checkpoint, generation=lt.pure_generation_options(64))
        )
        bindings["parents"][parent] = dict(
            checkpoint=checkpoint,
            inference_fingerprint=fingerprint,
            historical_runtime_identity=historical_identity,
            historical_model_audit=audit,
            checkpoint_full_cpu_state_verified=True,
        )
        bindings["checkpoint_lookup"]["PARENT." + parent] = dict(
            path=str(path),
            checkpoint_sha256=checkpoint["sha256"],
            source_experiment_id="PROSPECTIVE_SELECTION_V2",
            source_checkpoint_id=parent,
            expected_identity=checkpoint["identity"],
            status="VERIFIED_FULL_CPU_STATE",
        )
        backend = dict(
            execution_kind="REAL_CUDA_FROZEN_INFERENCE",
            checkpoint_id=parent,
            checkpoint=checkpoint,
            inference_fingerprint=fingerprint,
            checkpoint_full_cpu_state_verified=True,
            environment=environment,
            historical_runtime_identity=historical_identity,
            model_audit=audit,
            generation_options=lt.pure_generation_options(64),
            eos_token_ids=[7, 8],
            pad_token_id=7,
            bos_token_id=None,
            forward_config={"use_cache": False, "model_type": "qwen3_5"},
            optimizer_constructed=False,
            backward_calls=0,
            optimizer_updates=0,
            likelihood_rescoring_calls=0,
            all_parameters_frozen=True,
            dropout_disabled=True,
            eval_mode=True,
        )
        bridge["runtime_receipts"].append(dict(case=parent, receipt=copy.deepcopy(backend)))
        for candidate in (c for c in lt._candidates("REUSE_12") if c["parent"] == parent):
            current = copy.deepcopy(backend)
            identity = None
            student_sha = None
            if candidate["block"] is not None:
                path = legacy / f"{parent}-{candidate['block']}-{candidate['arm']}.pt"
                path.write_bytes(path.name.encode())
                student_sha = schema.file_digest(path)
                identity = dict(
                    experiment_id=lt.SOURCE_PHASE_ID,
                    parent=parent,
                    block=candidate["block"],
                    arm=candidate["arm"],
                    technical_only=False,
                )
                key = (
                    f"LEGACY.{parent}.block{candidate['block']}."
                    f"{candidate['logical_arm_id']}.step256"
                )
                bindings["checkpoint_lookup"][key] = dict(
                    path=str(path),
                    checkpoint_sha256=student_sha,
                    source_experiment_id=lt.SOURCE_PHASE_ID,
                    source_checkpoint_id=f"{parent}.block{candidate['block']}.{candidate['arm']}.step256",
                    expected_identity=identity,
                    status="VERIFIED_FULL_CPU_STATE",
                )
                current.update(
                    student_checkpoint_sha256=student_sha,
                    student_identity=identity,
                    student_step=256,
                    inference_fingerprint=lt.canonical_hash(
                        {"parent": fingerprint, "student_sha256": student_sha}
                    ),
                )
            job = {key: candidate[key] for key in ("parent", "block", "arm", "step")}
            job.update(panel="E_CONFIRM", tasks=800, draws=8, kind="confirm_eval_sealed")
            old_jobs.append(job)
            jid = lt._legacy_job_id(candidate)
            fingerprints[jid] = current["inference_fingerprint"]
            directory = legacy / "evaluations" / jid
            write(
                directory / "manifest.json",
                dict(protocol="O0", generation=lt.OLD_MANIFEST_GENERATION),
            )
            write(
                directory / "MODEL_IDENTITY.json",
                dict(
                    execution_kind="REAL_FROZEN_GPU",
                    checkpoint_id=lt.legacy_evaluate.checkpoint_id(job),
                    inference_fingerprint=current["inference_fingerprint"],
                    parent_checkpoint=checkpoint,
                    all_lora_adapters_disabled=False,
                    student_checkpoint_sha256=student_sha,
                    student_identity=identity,
                ),
            )
            write(directory / "completion.json", dict(status="COMPLETE", samples=6400))
            write(
                directory / "attempts" / "one" / "BACKEND.json",
                dict(execution_kind="REAL_FROZEN_GPU", backend=current),
            )
    write(run / "CONTINUITY_BRIDGE.json", bridge)
    calls = []
    options = dict(drop_last=False, corrupt_seed=False)

    def validate(legacy_run, job):
        assert legacy_run == legacy
        calls.append(lt.legacy_evaluate.evaluation_job_id(job))
        raw = []
        for task in source_tasks:
            for draw in range(8):
                identity = lt.legacy_evaluate.sample_identity(job, task, draw)
                raw.append(
                    {
                        **identity,
                        "request_id": lt.legacy_evaluate.digest(identity),
                        "sample_seed": lt.legacy_evaluate.sample_seed(identity),
                        "raw_text": "[10,20,30,40]",
                        "token_ids": [1, 2],
                        "stop_reason": "eos",
                        "generated_length": 2,
                        "elapsed_seconds": 0.001,
                        "execution_kind": "REAL_FROZEN_GPU",
                        "inference_fingerprint": fingerprints[calls[-1]],
                    }
                )
        if options["drop_last"]:
            raw.pop()
        if options["corrupt_seed"]:
            raw[0]["sample_seed"] += 1
        return raw, dict(status="COMPLETE", samples=len(raw))

    monkeypatch.setattr(lt.legacy_schema, "read_bound", lambda _run, _name: copy.deepcopy(old_jobs))
    monkeypatch.setattr(lt.legacy_evaluate, "validate_completed_evaluation", validate)
    monkeypatch.setattr(
        lt.legacy_evaluate, "evaluation_tasks", lambda *_: copy.deepcopy(source_tasks)
    )
    return dict(
        run=run,
        legacy=legacy,
        bindings=bindings,
        bridge=bridge,
        tasks=tasks,
        mapping=mapping,
        calls=calls,
        options=options,
    )


def audit(history, mode="RERUN_24"):
    return lt.audit_historical_trajectory(history["run"], history["bindings"], mode=mode)


def test_reuse_complete_14_models_original_seed_and_identity_retained(history):
    receipt, rows = audit(history, "REUSE_12")
    assert receipt["rows_count"] == len(rows) == 14336
    assert len(receipt["compatible_models"]) == 14 and not receipt["unavailable"]
    assert len(history["calls"]) == 14
    assert receipt["rows_hash"] == schema.digest(rows)
    assert receipt["new_model_calls"] == receipt["new_logical_generations"] == 0
    assert len([row for row in rows if row["logical_arm_id"] == "PARENT"]) == 2048
    for row in rows:
        original = row["original_record"]
        assert row["sample_seed"] == original["sample_seed"]
        assert row["sample_id"] == original["request_id"]
        assert row["raw_text"] == original["raw_text"]
        assert row["source_record_sha256"] == schema.digest(original)
        assert row["source_task_id"] == original["task_id"] != row["task_id"]
        assert row["source_arm"] == original["arm"] and "arm" not in row
        assert row["phase_id"] == lt.SOURCE_PHASE_ID
        assert row["origin"] == "historical_reuse" and row["compatibility_verified"]


def test_rerun_only_two_parent_baselines_never_old_step256(history):
    receipt, rows = audit(history)
    assert len(rows) == 2048 and len(receipt["compatible_models"]) == 2
    assert not receipt["legacy_student_step256_permitted"]
    assert {row["step"] for row in rows} == {0}
    assert {row["logical_arm_id"] for row in rows} == {"PARENT"}
    assert all("baseline.PARENT.step0" in jid for jid in history["calls"])


@pytest.mark.parametrize(
    "part,field,replacement",
    [
        ("model", "inference_fingerprint", "different"),
        ("backend", "environment", {"different": "package"}),
        ("backend", "generation_options", {"use_cache": True}),
        ("backend", "pad_token_id", 999),
        ("backend", "historical_runtime_identity", {"model_hash": "other-base"}),
        ("backend", "model_audit", {"tokenizer_hash": "other-codec"}),
    ],
)
def test_incompatible_parent_baseline_is_unavailable_without_budget_transfer(
    history, part, field, replacement
):
    directory = history["legacy"] / "evaluations/eval.S96.baseline.PARENT.step0.E_CONFIRM"
    path = directory / ("MODEL_IDENTITY.json" if part == "model" else "attempts/one/BACKEND.json")
    payload = json.loads(path.read_text())
    (payload if part == "model" else payload["backend"])[field] = replacement
    write(path, payload)
    receipt, rows = audit(history)
    assert len(rows) == 1024 and len(receipt["unavailable"]) == 1
    assert receipt["unavailable"][0]["parent"] == "S96"
    assert receipt["unavailable"][0]["replacement_generation_permitted"] is False
    assert receipt["new_model_calls"] == 0


def test_source_prompt_difference_cannot_be_reused(history):
    task = history["tasks"][0]
    task["observed"][0] += 1
    write_lines(history["run"] / "manifests/DEV_TRAJECTORY/tasks_public.jsonl", history["tasks"])
    receipt, rows = audit(history)
    assert not rows and len(receipt["unavailable"]) == 2
    assert all("prompt/root differs" in row["reason"] for row in receipt["unavailable"])


@pytest.mark.parametrize("option", ["drop_last", "corrupt_seed"])
def test_entire_legacy_coverage_and_original_request_seed_are_required(history, option):
    history["options"][option] = True
    receipt, rows = audit(history)
    assert not rows and len(receipt["unavailable"]) == 2
    expected = "incomplete" if option == "drop_last" else "seed/request"
    assert all(expected in row["reason"] for row in receipt["unavailable"])


def test_missing_raw_evidence_is_explicit_and_never_replaced(history):
    path = (
        history["legacy"] / "evaluations/eval.S96.baseline.PARENT.step0.E_CONFIRM/completion.json"
    )
    path.unlink()
    receipt, rows = audit(history)
    assert len(rows) == 1024
    assert receipt["unavailable"][0]["error_type"] == "FileNotFoundError"
    assert receipt["new_logical_generations"] == 0


def test_no_requalification_after_freeze_and_invalid_mapping_is_fatal(history):
    write(history["run"] / "FROZEN_PLAN.json", {"status": "FROZEN"})
    with pytest.raises(PermissionError, match="before freeze"):
        audit(history)
    (history["run"] / "FROZEN_PLAN.json").unlink()
    mapping = history["mapping"][:-1]
    write_lines(history["run"] / "manifests/development_source_mapping.jsonl", mapping)
    with pytest.raises(ValueError, match="128 mapped"):
        audit(history)
