"""Prospective CPU acceptance and the one-time scientific execution freeze."""

from __future__ import annotations

from pathlib import Path

from sr_f12.evaluation import durable_json

from .protocol import ENGINE_TASKS, EXPERIMENT_ID, file_hash, now, read_json, verify_run


def verify_native_receipt_revision(config, processor_receipt, mask_receipt):
    from .protocol_revision import revision_from_config

    revision = revision_from_config(config)
    if (
        processor_receipt.get("protocol_revision", "v1") != revision
        or mask_receipt.get("protocol_revision", "v1") != revision
        or (revision != "v1" and mask_receipt.get("serialization") != "indent_2")
    ):
        raise PermissionError(
            "Native processor/mask receipt belongs to a different protocol revision"
        )
    return revision


def execute_p0(root):
    from .native_audit import audit_native_inputs
    from .orchestration import write_task_receipt
    from .runner import dataset_and_store, immutable_files

    root = Path(root)
    _, config = verify_run(root)
    dataset, store = dataset_and_store(root)
    if len(dataset.tasks) != 4096 or len(dataset.runs) != 12:
        raise PermissionError("Frozen dataset coverage changed")
    if read_json(root / "parents/PARENT_AUDIT.json")["status"] != "PASS":
        raise PermissionError("Real parent provenance audit missing")
    tests = read_json(root / "deployment/LOCAL_CHECKS.json")
    if tests["status"] != "PASS" or tests["source_manifest_sha256"] != file_hash(
        root / "deployment/SOURCE_MANIFEST.json"
    ):
        raise PermissionError("Local tests do not bind the deployed source")
    data_audit = read_json(root / "data/FROZEN_DATA_AUDIT.json")
    if data_audit["status"] != "PASS" or data_audit["independent_answer_mismatches"]:
        raise PermissionError("Independent data audit failed")
    history = read_json(root / "data/HISTORICAL_WORLD_AUDIT.json")
    chain = read_json(root / "data/HISTORICAL_COVERAGE_CHAIN.json")
    if (
        history["status"] != "PASS"
        or history["overlapping_qids"]
        or chain["status"] != "REGISTERED_SR_F1_F12_F2_SYNTHETIC_WORLD_COVERAGE_VERIFIED"
    ):
        raise PermissionError("Historical numeric-world coverage or deduplication failed")
    visual = read_json(root / "data/VISUAL_SPOTCHECK_REVIEW.json")
    if (
        visual["reviewed_count"] < 24
        or visual["issues_remaining"]
        or visual["image_manifest_sha256"] != file_hash(root / "data/IMAGE_MANIFEST.jsonl")
    ):
        raise PermissionError("Fixed semantic/image review missing")
    # Every model-visible image is authenticated by the same strict route used
    # by the GPU workers; no model weights or model outputs are involved here.
    for qid in dataset.tasks:
        store.model_input(qid)
    native_path = root / "native/PROCESSOR_RECEIPT.json"
    if native_path.exists():
        from .native_audit import direct_source_identity

        receipt = read_json(native_path)
        actual = {name: row["sha256"] for name, row in direct_source_identity().items()}
        if receipt.get("direct_source_sha256") != actual:
            raise PermissionError("Native acceptance source differs from deployed runtime")
        if receipt["image_manifest_sha256"] != file_hash(root / "data/IMAGE_MANIFEST.jsonl"):
            raise PermissionError("Native acceptance image inventory differs")
        masks = root / "native/SUPPORT_MASK_RECEIPT.json"
        if receipt["mask_receipt_sha256"] != file_hash(masks):
            raise PermissionError("Native mask receipt changed")
        for item in read_json(masks)["files"]:
            if file_hash(root / item["path"]) != item["sha256"]:
                raise PermissionError("Native teacher tokens/masks changed")
        # Inherited source is unchanged; bind its authenticated CPU-audit snapshot.
        snapshot = read_json(root / "NATIVE_AUDIT_SOURCE.json")
        for item in snapshot["files"]:
            if (
                item["path"].startswith("src/")
                and not item["path"].startswith("src/sr_f3_ce/")
                and file_hash(root / "code" / item["path"]) != item["sha256"]
            ):
                raise PermissionError("Inherited processor/runtime source differs")
        prior = read_json(root / "prior/MODEL_ENVIRONMENT_IDENTITY.json")
        for key in ("processor_hash", "tokenizer_hash", "chat_template_hash"):
            if receipt["native_processor_identity"][key] != prior[key]:
                raise PermissionError("Native processor no longer matches verified parent")
        if (
            receipt.get("weights_loaded")
            or receipt.get("cuda_initialized")
            or receipt.get("model_calls")
        ):
            raise PermissionError("Native CPU receipt has an unexpected model call")
        durable_json(
            root / "native/P0_REUSE_VERIFICATION.json",
            dict(
                status="PASS",
                direct_source_sha256=actual,
                native_receipt_sha256=file_hash(native_path),
                inherited_source_manifest_sha256=file_hash(root / "NATIVE_AUDIT_SOURCE.json"),
                source_manifest_sha256=file_hash(root / "deployment/SOURCE_MANIFEST.json"),
            ),
        )
    else:
        receipt = audit_native_inputs(root)
    verify_native_receipt_revision(
        config, receipt, read_json(root / "native/SUPPORT_MASK_RECEIPT.json")
    )
    if receipt["status"] != "PASS" or receipt["input_sidecar_negative_test"] != "PASS":
        raise PermissionError("Native processor/mask/input isolation gate failed")
    artifacts = immutable_files(root, root / "native")
    artifacts.extend(str(p.relative_to(root)) for p in sorted((root / "data").glob("*.json")))
    artifacts += [
        "data/IMAGE_MANIFEST.jsonl",
        "parents/PARENT_AUDIT.json",
        "deployment/LOCAL_CHECKS.json",
    ]
    return write_task_receipt(
        root,
        "p0",
        artifacts=sorted(set(artifacts)),
        evidence={
            k: "PASS"
            for k in (
                "cpu_tests",
                "independent_answers",
                "historical_dedup",
                "render",
                "processor",
                "input_leakage",
            )
        },
    )


def freeze_science(root):
    from .orchestration import Controller, receipt_path, verify_task
    from .runner import immutable_files

    root = Path(root)
    identity, config = verify_run(root)
    scheduler = read_json(root / "orchestration/CONFIG.json")
    controller = Controller(
        root,
        scheduler["code_root"],
        python=scheduler["python"],
        user=scheduler["user"],
        pool_size=scheduler["pool_size"],
        cpu_partition=scheduler["cpu_policy"]["partition"],
        cpu_qos=scheduler["cpu_policy"]["qos"],
        cpu_node=scheduler["cpu_policy"]["cpu_node"],
    )
    status = controller.status()
    if status["blocked"] or not {"p0", *ENGINE_TASKS} <= controller.completed():
        raise PermissionError(
            "All CPU/ENGINE workers need verified successful Slurm terminal state"
        )
    if any((root / name).exists() for name in ("runs", "sealed")):
        raise PermissionError("Scientific freeze must precede scientific training and new TEST")
    for task in ("p0", *ENGINE_TASKS):
        verify_task(root, task)
    external = read_json(root / "external/PANELS.json")
    if external["status"] != "FROZEN" or external["CHARTQA"]["question_count"] != 2500:
        raise PermissionError("External identity/type registry must be frozen before science")
    bound = immutable_files(root, root / "native", root / "parents", root / "external")
    bound += [str(receipt_path(root, key).relative_to(root)) for key in ("p0", *ENGINE_TASKS)]
    bound += ["prior/MODEL_ENVIRONMENT_IDENTITY.json", "deployment/SOURCE_MANIFEST.json"]
    value = dict(
        experiment_id=EXPERIMENT_ID,
        status="FROZEN_BEFORE_SCIENCE",
        execution_utc=now(),
        run_identity_sha256=file_hash(root / "RUN_IDENTITY.json"),
        operator_has_seen_new_test_results=False,
        new_test_outputs_sealed=True,
        plan_sha256=file_hash(root / "plan/CODEX_EXECUTION_PLAN_SR_F3_CE_zh.md"),
        config_sha256=file_hash(root / "plan/config/SR_F3_CE.json"),
        base_revision=config["model"]["revision"],
        parent_checks=read_json(root / "parents/PARENT_AUDIT.json"),
        environment_receipt=read_json(root / "prior/MODEL_ENVIRONMENT_IDENTITY.json"),
        code_commit=identity["code_commit"],
        dirty_diff_sha256=identity["dirty_diff_sha256"],
        source_manifest_sha256=file_hash(root / "deployment/SOURCE_MANIFEST.json"),
        engine_receipts={key: file_hash(receipt_path(root, key)) for key in ENGINE_TASKS},
        engineering_files={p: file_hash(root / p) for p in sorted(set(bound))},
        data_manifest_sha256=file_hash(root / "plan/manifests/TASKS_AUDIT_ONLY.jsonl"),
        image_manifest_sha256=file_hash(root / "data/IMAGE_MANIFEST.jsonl"),
        sampling_and_stop=config["generation"],
        training=config["training"],
        loss_mask_receipt="native/SUPPORT_MASK_RECEIPT.json",
        engine_resume_receipt="tasks/engine--resume/COMPLETE.json",
        maximum_authorized_project_workers=5,
        gpu_time_limit=None,
        research_wallclock_limit=None,
        full_raw_outputs_in_delivery=True,
        data_uses=dict(
            ENGINE="engineering only",
            TRAIN_RL="fixed on-policy training",
            SUPPORT="fixed independent CE",
            MONITOR="measurement only; no routing",
            TEST="sealed fixed endpoint evaluation",
        ),
    )
    if config.get("protocol_revision", "v1") != "v1":
        from .protocol import verify_run_source_revision

        revision = verify_run_source_revision(root)
        history = (
            root
            / "protocol_revision/history_v1"
            / "SR_F3_CE_PROTOCOL_STOP_EVIDENCE_FULL_UNDER400MB_20261016.tar.xz"
        )
        history_receipt = history.with_suffix(history.suffix + ".verification.json")
        if (
            history.stat().st_size != 219281700
            or file_hash(history)
            != "d67af2a3a00f44713c3bcb62d526f90f12e392f5fa0fe2d1872f54a3e1184d60"
            or not history_receipt.is_file()
        ):
            raise PermissionError("The complete delivered v1 evidence must remain preserved")
        value.update(
            protocol_revision=revision["protocol_revision"],
            protocol_revision_request_sha256=revision["request_sha256"],
            protocol_selection_sha256=revision["protocol_selection_sha256"],
            protocol_selection=read_json(root / "PROTOCOL_SELECTION.json"),
            protocol_revision_diff=read_json(root / "plan/PROTOCOL_REVISION.json"),
            prior_v1_records_preserved=dict(
                archive=str(history.relative_to(root)),
                sha256=file_hash(history),
                bytes=history.stat().st_size,
                unique_generations=3904,
                verification_receipt_sha256=file_hash(history_receipt),
            ),
            selection_precedes_revised_engine=True,
        )
        for relative in (
            "authorization/REQUEST.md",
            "PROTOCOL_SELECTION.json",
            str(history.relative_to(root)),
            str(history_receipt.relative_to(root)),
        ):
            value["engineering_files"][relative] = file_hash(root / relative)
    durable_json(root / "EXECUTION_FREEZE.json", value)
    verify_run(root, science=True)
    return value
