"""Synthetic reporting, release-boundary, ledger and NLL regression checks."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest

from ssvc_flow.src.exposure_position import reports
from ssvc_flow.src.exposure_position.schema import digest as scientific_digest
from ssvc_flow.src.exposure_position.schema import file_digest
from ssvc_flow.src.exposure_position.semantics import score_output
from ssvc_flow.src.exposure_position.statistics import ARMS
from ssvc_flow.src.verified_discovery_transfer.queue import digest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def frozen_bindings(run):
    name = "PARENT_AND_ENDPOINT_BINDINGS.json"
    put(
        run / name,
        {
            "checkpoint_lookup": {
                "PARENT.S96": {
                    "path": "/server/legacy/parent.pt",
                    "checkpoint_sha256": "a" * 64,
                    "status": "VERIFIED_FULL_CPU_STATE",
                }
            }
        },
    )
    put(run / "FROZEN_PLAN.json", {"files": {name: {"sha256": file_digest(run / name)}}})


def receipt():
    return dict(
        schema="ser-j23-release-v1",
        status="RELEASED",
        plan_hash="a" * 64,
        matrix_digest="b" * 64,
        all_registered_models_terminal=True,
        all_confirmation_requests_complete=True,
        all_registered_jobs_complete=True,
        registered_training_jobs=12,
        completed_evaluation_receipts=[],
    )


def case():
    public = dict(
        task_id="t",
        root_id="r",
        family="cross_series",
        observed=[10, 20, 30, 45],
        H_original=[[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]],
        b_original=[50, 60, 70],
        legal_domain=[0, 99],
        operation="sum4",
    )
    audit = dict(
        task_id="t",
        root_id="r",
        true_world=[10, 20, 30, 40],
        center=3,
        corrupted_index=3,
        root_cohort="CORE32",
    )
    return public, audit


def test_report_rejects_unsigned_or_incomplete_release_before_truth_access(tmp_path, monkeypatch):
    from ssvc_flow.src.exposure_position import schema

    monkeypatch.setattr(schema, "RoleDataset", lambda *args: pytest.fail("Truth loader reached"))
    for value in (
        {},
        {**receipt(), "all_confirmation_requests_complete": False},
        {**receipt(), "status": "PASS"},
    ):
        with pytest.raises(PermissionError):
            reports.write_reports(tmp_path / "reports", [], release_receipt=value)
    assert not (tmp_path / "reports").exists()


def test_disk_receipt_and_queue_seal_must_match_before_truth_access(tmp_path, monkeypatch):
    from ssvc_flow.src.exposure_position import queue, schema

    release = receipt()
    put(tmp_path / "RELEASE_RECEIPT.json", release)
    monkeypatch.setattr(
        schema, "verify_frozen", lambda *a, **kw: {"plan_hash": release["plan_hash"]}
    )
    monkeypatch.setattr(schema, "RoleDataset", lambda *args: pytest.fail("Truth loader reached"))
    db = SimpleNamespace(
        execute=lambda *args: SimpleNamespace(fetchone=lambda: ("wrong",)), close=lambda: None
    )
    monkeypatch.setattr(
        queue, "registered_queue", lambda run: SimpleNamespace(rows=lambda: [], db=db)
    )
    with pytest.raises(PermissionError, match="seal"):
        reports._release_context(tmp_path / "reports", release)


def test_release_gateway_integrity_seal_receipt_then_scoring(tmp_path, monkeypatch):
    from ssvc_flow.src.exposure_position import evaluate, queue

    events = []
    job = {"job_id": "synthetic-job"}
    monkeypatch.setattr(
        evaluate,
        "integrity",
        lambda run: (events.append("integrity") or receipt(), {job["job_id"]: []}, [job]),
    )
    monkeypatch.setattr(evaluate, "evaluation_job_id", lambda job: job["job_id"])
    fake_queue = SimpleNamespace(
        seal_release=lambda: events.append("seal") or "b" * 64,
        db=SimpleNamespace(close=lambda: events.append("close")),
    )
    monkeypatch.setattr(queue, "registered_queue", lambda run: fake_queue)

    def scoring(run, job, raw):
        assert (tmp_path / "RELEASE_RECEIPT.json").exists()
        assert events.index("integrity") < events.index("seal")
        events.append("score")
        return []

    monkeypatch.setattr(evaluate, "score_rows", scoring)
    monkeypatch.setattr(
        reports,
        "write_reports",
        lambda *a, **kw: events.append("reports") or {"status": "TEST_ONLY"},
    )
    assert reports.release_and_analyze(tmp_path)["status"] == "TEST_ONLY"
    assert events == ["integrity", "seal", "close", "score", "reports"]


@pytest.mark.parametrize("raw_text", ["[10,20,30,40]", "无法确定"])
def test_scored_records_bind_raw_bytes_and_independent_events(tmp_path, raw_text):
    public, audit = case()
    raw = dict(
        raw_text=raw_text,
        stop_reason="eos",
        task_id="t",
        root_id="r",
        draw_index=0,
        sample_id="c" * 64,
        job_id="synthetic-job",
        execution_kind="REAL_FROZEN_GPU",
    )
    path = tmp_path / "evaluations" / raw["job_id"] / "roots/r" / (raw["sample_id"] + ".json")
    put(path, dict(record=raw, record_sha256=digest(raw), manifest_hash="d" * 64))
    scored = {**raw, **score_output(public, audit, raw["raw_text"], raw["stop_reason"])}
    context = dict(
        run=tmp_path,
        jobs=[dict(job_id=raw["job_id"], panel="E_CONFIRM2", draws=1)],
        job_tasks={raw["job_id"]: [public]},
        audits={"E_CONFIRM2": {"t": audit}},
    )
    release = {
        "completed_evaluation_receipts": [
            dict(job_id=raw["job_id"], manifest_hash="d" * 64, records_hash=digest([raw]))
        ]
    }
    reports._verify_scored_against_release([scored], context, release)
    with pytest.raises(ValueError, match="scoring"):
        reports._verify_scored_against_release([{**scored, "X": not scored["X"]}], context, release)
    with pytest.raises(ValueError, match="envelope"):
        reports._verify_scored_against_release([{**scored, "raw_text": "oops"}], context, release)
    with pytest.raises(ValueError, match="every registered"):
        reports._verify_scored_against_release([], context, release)


def test_nll_audit_preserves_original_source_and_last_actual_step(tmp_path):
    source = tmp_path / "legacy-training"
    audits = {"COMMON_TRAIN": [], "DONOR_TRAIN": [], "REPLAY": []}
    for slot in range(17):
        role = (
            "COMMON_TRAIN" if slot < 11 or slot == 16 else "DONOR_TRAIN" if slot == 11 else "REPLAY"
        )
        audits[role].append(
            dict(
                task_id=f"new-{slot}",
                source_task_id=f"old-{slot}",
                logical_arm_id=ARMS[0] if role == "DONOR_TRAIN" else None,
                family="cross_series",
                center=3,
                corrupted_index=3,
            )
        )
    for step in range(193, 257):
        slots = []
        for slot in range(16):
            task_slot = 16 if slot == 0 and step > 193 else slot
            slots.append(
                dict(
                    slot=slot,
                    task_id=f"old-{task_slot}",
                    role="common" if slot < 11 else "donor" if slot == 11 else "replay",
                    sequence_nll=step / 1000,
                )
            )
        put(
            source / f"update{step:03d}.json",
            dict(update=step, status="UPDATE_APPLIED", physical_cost_complete=True, slots=slots),
        )
    job = dict(
        panel="E_CONFIRM2",
        logical_arm_id=ARMS[0],
        parent="S96",
        block=0,
        source="LEGACY",
        checkpoint_binding={"path": str(source / "step256.pt")},
    )
    context = dict(run=tmp_path, jobs=[job], training_audits=audits)
    result = reports.training_nll_audit(context)
    assert result["paths"] == 1 and result["errors"] == []
    assert len(result["exposures"]) == 64 * 16
    last = next(
        row
        for row in result["summaries"]
        if row["row_type"] == "last_per_task" and row["task_id"] == "new-0"
    )
    assert last["last_observed_step"] == 193
    assert last["sequence_nll"] == 0.193
    assert "not_uniform_step256" in last["measurement"]
    source.joinpath("update201.json").unlink()
    missing = reports.training_nll_audit(context)
    assert missing["status"] == "INCOMPLETE" and missing["errors"][0]["step"] == 201
    assert len(missing["exposures"]) == 63 * 16


def test_unimported_accounting_stays_null_not_zero(tmp_path):
    result = reports.gpu_accounting(tmp_path, receipt())
    assert result["status"] == "UNVERIFIED"
    assert result["allocated_gpu_hours"] is None
    assert result["logical_new_training_updates"] == 3072
    assert result["physical_new_training_updates_started"] == 0


def accounting_fixture(tmp_path):
    release = {**receipt(), "registered_training_jobs": 0}
    put(tmp_path / "worker_attempts/attempt/STARTED.json", {"slurm_job_id": "100"})
    for n in range(32):
        put(
            tmp_path / f"bridge/attempts/physical{n:02d}.json",
            dict(status="UPDATE_APPLIED", physical_cost_complete=True),
        )
    raw = tmp_path / "sacct.txt"
    raw.write_text("synthetic scheduler evidence\n")
    jobs = [
        dict(
            slurm_job_id="100",
            elapsed_seconds=3600,
            allocated_gpus=2,
            state="COMPLETED",
            workload_category="formal_worker",
        ),
        dict(
            slurm_job_id="200",
            elapsed_seconds=1800,
            allocated_gpus=1,
            state="COMPLETED",
            workload_category="technical_bridge",
        ),
        dict(
            slurm_job_id="300",
            elapsed_seconds=0,
            allocated_gpus=0,
            state="CANCELLED",
            workload_category="cancelled_before_allocation",
        ),
        dict(
            slurm_job_id="400",
            elapsed_seconds=120,
            allocated_gpus=0,
            state="COMPLETED",
            workload_category="CPU_audit",
        ),
    ]
    account = dict(
        status="VERIFIED",
        source="sacct",
        plan_hash=release["plan_hash"],
        allocation_scope_complete=True,
        submitted_slurm_job_ids=[j["slurm_job_id"] for j in jobs],
        jobs=jobs,
        raw_files=[dict(path="sacct.txt", sha256=file_digest(raw))],
    )
    put(tmp_path / "SCHEDULER_ACCOUNTING.json", account)
    return release, account


def test_accounting_counts_allocations_once_includes_bridge_failures_and_zero_gpu(tmp_path):
    release, _ = accounting_fixture(tmp_path)
    result = reports.gpu_accounting(tmp_path, release)
    assert result["status"] == "VERIFIED", result["reason"]
    assert result["allocated_gpu_hours"] == 2.5
    assert len(result["cpu_only_allocations"]) == 2
    assert result["physical_technical_updates_started"] == 32


@pytest.mark.parametrize(
    "mutation", ["missing_worker", "job_step", "changed_raw", "missing_bridge"]
)
def test_accounting_mismatch_cannot_certify_actual_cost(tmp_path, mutation):
    release, account = accounting_fixture(tmp_path)
    if mutation == "changed_raw":
        (tmp_path / "sacct.txt").write_text("changed")
    elif mutation == "missing_worker":
        account["submitted_slurm_job_ids"].remove("100")
        account["jobs"] = account["jobs"][1:]
    elif mutation == "job_step":
        account["submitted_slurm_job_ids"][0] = "100.batch"
        account["jobs"][0]["slurm_job_id"] = "100.batch"
    else:
        account["jobs"][1]["workload_category"] = "unidentified_GPU_work"
    put(tmp_path / "SCHEDULER_ACCOUNTING.json", account)
    result = reports.gpu_accounting(tmp_path, release)
    assert result["status"] == "UNVERIFIED" and result["allocated_gpu_hours"] is None


def analysis_fixture():
    detail = dict(
        estimate=-0.1,
        ci95_root_conditional=[-0.2, 0.05],
        per_parent_order=[[-0.1] * 3, [-0.1] * 3],
        direction95="uncertain",
        harm_supported=False,
    )
    primary = {**detail, "name": "H3"}
    secondary = {
        name: {**detail, "ci98_333333_bonferroni_three": [-0.25, 0.1], "direction_supported": False}
        for name in ("Psi_E", "G_3_2", "G_3_3")
    }
    noninfer = {
        name: {**detail, "one_sided_97_5_lower": -0.2, "noninferiority_established": False}
        for name in ("N2_c3j3", "N3_c3j3")
    }
    cell = dict(
        parent="S96",
        block=0,
        logical_arm_id=ARMS[0],
        family="cross_series",
        center=3,
        corrupted_index=3,
        X=1.0,
        F_val=1.0,
        F_legal=1.0,
        R=0.0,
        n_answers=8,
        q=[0.0, 0.0, 0.0, 1.0],
        r=[0.0] * 4,
        c=[None] * 4,
        c_undefined_reason="no_non_X_answers",
    )
    return dict(
        status="COMPLETE_CONDITIONAL_INFERENCE",
        primary=primary,
        H2=detail,
        parent_contrasts={f"{ARMS[3]}-PARENT": detail},
        bootstrap={},
        scope={},
        secondary_family={"endpoints": secondary},
        noninferiority_family={"endpoints": noninfer},
        descriptive={"G_3_3_minus_G_3_2": detail},
        all_cell_metrics=[cell],
        all_cell_contrasts=[],
        macro={},
        task_metrics=[],
        value_extra_decomposition={},
    )


def test_writer_delivers_all_tables_but_cannot_complete_without_actual_cost(tmp_path, monkeypatch):
    frozen_bindings(tmp_path)
    context = dict(
        run=tmp_path,
        registry=[{"synthetic_registry": True}],
        public={},
        audits={},
        mode="REUSE_12",
        diagnostic_availability={"unavailable": []},
    )
    monkeypatch.setattr(reports, "_release_context", lambda *args: context)
    monkeypatch.setattr(reports, "_verify_scored_against_release", lambda *args: None)
    seen = {}

    def analysis(rows, *, expected_tasks):
        seen["registry"] = expected_tasks
        return analysis_fixture()

    monkeypatch.setattr(reports, "analyze_confirmation", analysis)
    monkeypatch.setattr(reports, "_enriched_task_rows", lambda *a: [])
    monkeypatch.setattr(
        reports, "_diagnostic_tables", lambda *a: {panel: [] for panel in reports.PANELS[1:]}
    )
    monkeypatch.setattr(
        reports,
        "training_nll_audit",
        lambda *a: dict(status="COMPLETE", summaries=[], exposures=[], errors=[], source_files=[]),
    )
    monkeypatch.setattr(reports, "_plots", lambda *a: [])
    output = tmp_path / "reports"
    scope = reports.write_reports(output, [], release_receipt=receipt())
    assert seen["registry"] == context["registry"]
    assert scope["status"] == "REPORT_COMPLETE_STAGE_INCOMPLETE"
    assert scope["allocated_gpu_hours"] is None and not (tmp_path / "STAGE_COMPLETE.json").exists()
    expected = [
        "PRIMARY_H3.json",
        "SECONDARY_FAMILY.json",
        "NONINFERIORITY_C3J3.json",
        "ALL_CELL_METRICS.csv",
        "EDIT_DENOMINATORS.csv",
        "VALUE_VS_EXTRA_ERRORS.csv",
        "ROOT_LEVEL_METRICS.jsonl",
        "PATH_HETEROGENEITY.csv",
        "FULL_TRAIN_FIT.csv",
        "TRAIN_NLL_AUDIT.csv",
        "DEV_TRAJECTORY.csv",
        "DEV_DELTA.csv",
        "DIRECTION_CHECKS.csv",
        "FINAL_REPORT_zh.md",
        "EXECUTION_SCOPE.json",
        "FINAL_GPU_ACCOUNTING.json",
    ]
    assert all((output / name).is_file() for name in expected)
    with (output / "EDIT_DENOMINATORS.csv").open() as stream:
        edits = list(csv.DictReader(stream))
    assert edits[0]["c"] == "null" and edits[0]["denominator_not_X"] == "0"
    final = (output / "FINAL_REPORT_zh.md").read_text()
    assert "区间跨零" in final and "方向不确定" in final
    assert "未建立非劣不自动等于证实伤害" in final
    assert "UNVERIFIED" in final
    manifest = json.loads((output / "FILE_MANIFEST_SHA256.json").read_text())
    assert all(file_digest(output / name) == item["sha256"] for name, item in manifest.items())
    root_manifest = json.loads((tmp_path / "FILE_MANIFEST_SHA256.json").read_text())
    assert "reports/FILE_MANIFEST_SHA256.json" in root_manifest["files"]
    assert all(
        file_digest(tmp_path / name) == item["sha256"]
        for name, item in root_manifest["files"].items()
    )
    monkeypatch.setattr(
        reports, "gpu_accounting", lambda *a: {"status": "VERIFIED", "allocated_gpu_hours": 1.0}
    )
    scope = reports.write_reports(output, [], release_receipt=receipt())
    assert scope["status"] == "STAGE_COMPLETE"
    final_manifest = json.loads((tmp_path / "FILE_MANIFEST_SHA256.json").read_text())
    assert final_manifest["files"]["STAGE_COMPLETE.json"]["sha256"] == file_digest(
        tmp_path / "STAGE_COMPLETE.json"
    )


def test_derived_report_replacement_preserves_previous_bytes(tmp_path):
    path = tmp_path / "report.json"
    reports._json(path, {"status": "UNVERIFIED"})
    old_hash = file_digest(path)
    reports._json(path, {"status": "VERIFIED"})
    prior = tmp_path / ".history" / old_hash / path.name
    assert json.loads(prior.read_text()) == {"status": "UNVERIFIED"}
    assert file_digest(prior) == old_hash


def test_root_manifest_hashes_weights_keeps_history_and_declares_all_omissions(tmp_path):
    frozen_bindings(tmp_path)
    (tmp_path / "checkpoint.pt").write_bytes(b"synthetic weights")
    (tmp_path / ".student.lock").write_text("held-or-stale-lock")
    (tmp_path / "raw.jsonl.partial").write_text("unfinished")
    (tmp_path / "queue.sqlite-shm").write_bytes(b"transient")
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "uncommitted.pt").write_bytes(b"unfinished")
    (tmp_path / "external.pt").symlink_to("/server/legacy/parent.pt")
    (tmp_path / "linked_directory").symlink_to(tmp_path / "tmp", target_is_directory=True)
    manifest = reports.write_root_manifest(tmp_path)
    assert "checkpoint.pt" in manifest["files"]
    assert "FILE_MANIFEST_SHA256.json" not in manifest["files"]
    omissions = {row["path"]: row for row in manifest["omissions"]}
    assert set(omissions) == {
        ".student.lock",
        "raw.jsonl.partial",
        "queue.sqlite-shm",
        "tmp",
        "external.pt",
        "linked_directory",
    }
    assert omissions["tmp"]["scope"] == "directory_subtree"
    assert omissions["external.pt"]["target"] == "/server/legacy/parent.pt"
    binding = manifest["external_evidence"]["checkpoint_lookup"]["PARENT.S96"]
    assert (
        binding["path"] == "/server/legacy/parent.pt" and binding["checkpoint_sha256"] == "a" * 64
    )
    assert manifest["timing"]["added_to_allocated_gpu_hours"] is False
    old_hash = file_digest(tmp_path / "FILE_MANIFEST_SHA256.json")
    (tmp_path / "new.json").write_text("{}")
    newer = reports.write_root_manifest(tmp_path)
    history_name = f".history/{old_hash}/FILE_MANIFEST_SHA256.json"
    assert newer["files"][history_name]["sha256"] == old_hash
    assert all(
        file_digest(tmp_path / name) == item["sha256"] for name, item in newer["files"].items()
    )


def test_root_manifest_rejects_changed_frozen_external_bindings(tmp_path):
    frozen_bindings(tmp_path)
    put(tmp_path / "PARENT_AND_ENDPOINT_BINDINGS.json", {"checkpoint_lookup": {}})
    with pytest.raises(ValueError, match="differ from the frozen plan"):
        reports.write_root_manifest(tmp_path)
    assert not (tmp_path / "FILE_MANIFEST_SHA256.json").exists()


@pytest.mark.parametrize("mutation", ["file", "inventory"])
def test_root_manifest_rejects_source_drift(tmp_path, monkeypatch, mutation):
    frozen_bindings(tmp_path)
    source = tmp_path / "checkpoint.pt"
    source.write_bytes(b"synthetic weights")

    def racing_digest(path):
        value = file_digest(path)
        if path == source:
            if mutation == "file":
                source.write_bytes(b"changed weights")
            else:
                (tmp_path / "late_record.json").write_text("{}")
        return value

    monkeypatch.setattr(reports, "file_digest", racing_digest)
    with pytest.raises(ValueError, match="changed during manifest"):
        reports.write_root_manifest(tmp_path)
    assert not (tmp_path / "FILE_MANIFEST_SHA256.json").exists()


def test_root_manifest_rejects_symlinked_destination(tmp_path):
    frozen_bindings(tmp_path)
    (tmp_path / "FILE_MANIFEST_SHA256.json").symlink_to(tmp_path / "FROZEN_PLAN.json")
    with pytest.raises(ValueError, match="must not be a symlink"):
        reports.write_root_manifest(tmp_path)


def historical_fixture():
    tasks, audits, rows = {}, {}, []
    for n in range(128):
        public, audit = case()
        public.update(task_id=f"new-{n}", root_id=f"root-{n}")
        audit.update(
            task_id=public["task_id"],
            root_id=public["root_id"],
            source_task_id=f"old-{n}",
            source_root_id=f"old-root-{n}",
        )
        tasks[public["task_id"]], audits[audit["task_id"]] = public, audit
        for draw in range(8):
            sid = digest([n, draw])
            original = dict(
                task_id=f"old-{n}",
                root_id=f"old-root-{n}",
                request_id=sid,
                sample_seed=n * 8 + draw,
                raw_text="无法确定",
                token_ids=[1],
                stop_reason="eos",
            )
            rows.append(
                {
                    **original,
                    "sample_id": sid,
                    "original_record": original,
                    "source_record_sha256": scientific_digest(original),
                    "compatibility_verified": True,
                    "origin": "historical_reuse",
                    "phase_id": "SER_J2_20261007",
                    "panel": "DEV_TRAJECTORY",
                    "task_id": public["task_id"],
                    "root_id": public["root_id"],
                    "parent": "S96",
                    "block": None,
                    "logical_arm_id": "PARENT",
                    "step": 0,
                    "checkpoint_id": "old-S96",
                    "draw_index": draw,
                }
            )
    model = dict(
        parent="S96",
        block=None,
        logical_arm_id="PARENT",
        step=0,
        samples=1024,
        compatible=True,
        alias_records_hash=scientific_digest(rows),
    )
    reuse = dict(
        status="AUDITED",
        mode="RERUN_24",
        compatible_models=[model],
        unavailable=[
            dict(
                parent="REP96",
                block=None,
                logical_arm_id="PARENT",
                step=0,
                reason="synthetic_missing_old_parent_answers",
            )
        ],
        rows_count=len(rows),
        rows_hash=scientific_digest(rows),
        new_model_calls=0,
        new_logical_generations=0,
    )
    return dict(
        mode="RERUN_24",
        historical_receipt=reuse,
        historical_rows=rows,
        public={"DEV_TRAJECTORY": tasks},
        audits={"DEV_TRAJECTORY": audits},
    )


def test_historical_reuse_keeps_unicode_failure_old_identity_and_seed():
    context = historical_fixture()
    scored = reports._historical_trajectory_scores(context)
    assert len(scored) == 1024 and all(row["I"] for row in scored)
    assert scored[0]["phase_id"] == "SER_J2_20261007"
    assert (
        scored[17]["sample_seed"]
        == context["historical_rows"][17]["original_record"]["sample_seed"]
    )
    tasks = reports._enriched_task_rows(scored, context)
    assert len(tasks) == 128 and all(row["historical_reused"] for row in tasks)
    assert all(row["I"] == 1 for row in tasks)


@pytest.mark.parametrize("mutation", ["seed", "root_mapping", "rerun_student", "drop_answer"])
def test_historical_reuse_rejects_resampling_misalignment_or_unbudgeted_terminal(mutation):
    context = historical_fixture()
    rows = context["historical_rows"]
    if mutation == "seed":
        rows[0]["sample_seed"] += 1
    elif mutation == "root_mapping":
        rows[0]["root_id"] = "another-root"
    elif mutation == "rerun_student":
        context["historical_receipt"]["compatible_models"][0]["logical_arm_id"] = ARMS[0]
        context["historical_receipt"]["compatible_models"][0]["step"] = 256
    else:
        rows.pop()
    # Even if an upstream caller consistently rehashes modified aliases, the
    # source-preservation and frozen allocation checks must still reject them.
    context["historical_receipt"]["rows_hash"] = scientific_digest(rows)
    with pytest.raises(ValueError):
        reports._historical_trajectory_scores(context)


def test_diagnostics_keep_domain_violations_denominators_and_within_delta_pairing():
    public, audit = case()
    public["observed"][-1] = 80
    audits, tasks, raw = {}, {}, []
    for panel, step, draws in (("DEV_TRAJECTORY", 64, 8), ("DEV_DELTA", 256, 4)):
        task = {**public, "task_id": panel + "-task"}
        info = {**audit, "task_id": task["task_id"], "signed_delta": 40}
        tasks[panel], audits[panel] = {task["task_id"]: task}, {task["task_id"]: info}
        for block in range(3):
            for arm in ARMS[:2]:
                for draw in range(draws):
                    text = "[10,20,30,40]" if arm == ARMS[0] else "[10,-20,30,40]"
                    raw.append(
                        dict(
                            sample_id=f"{panel}-{arm}-{block}-{draw}",
                            phase_id="SER_J23_20261008",
                            checkpoint_id=f"{arm}-{block}-{step}",
                            panel=panel,
                            parent="S96",
                            block=block,
                            logical_arm_id=arm,
                            step=step,
                            draw_index=draw,
                            **score_output(task, info, text),
                        )
                    )
    context = dict(
        public=tasks,
        audits=audits,
        mode="RERUN_24",
        historical_receipt={},
        delta_legality=[
            dict(
                status="REJECTED_OUT_OF_DOMAIN",
                source_root_id="illegal-root",
                cell="c4j4",
                signed_delta=40,
            )
        ],
    )
    means = reports._enriched_task_rows(raw, context)
    tables = reports._diagnostic_tables(raw, means, context)
    bad = next(
        row
        for row in tables["DEV_DELTA"]
        if row["row_type"] == "cell_mean" and row["logical_arm_id"] == ARMS[1]
    )
    assert bad["I"] == 1 and bad["F_val"] == 1 and bad["R"] == 1
    assert bad["denominator_all_answers"] == 4
    assert bad["old_center_compatible_output_match_by_position"][1] == 1
    assert bad["old_center_compatible_value_out_of_domain_by_position"][1] == 1
    effect = next(
        row
        for row in tables["DEV_TRAJECTORY"]
        if row["row_type"] == "equal_order_parent_effect" and row["metric"] == "X"
    )
    assert effect["estimate"] == -1
    infeasible = next(
        row for row in tables["DEV_DELTA"] if row["row_type"] == "infeasible_construction"
    )
    assert infeasible["answer_denominator"] is None and infeasible["not_a_model_I_answer"]
