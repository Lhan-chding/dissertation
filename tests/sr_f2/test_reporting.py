"""CPU-only reporting tests: no live scheduler, model load, or experiment calls."""

import json

import pytest

from sr_f2 import reporting
from sr_f2.diagnostics import generation_config
from sr_f2.reporting import (
    _write_jsonl_once,
    summarize_rows,
    trigger_summary,
    validated_panel_rows,
    verify_release_gate,
)
from sr_f12.evaluation import group_slots
from sr_f12.protocol import object_hash


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class FakeSlurm:
    def __init__(self):
        self.jobs, self.accounting, self.calls = {}, {}, []

    def queue(self):
        self.calls.append("queue")
        return self.jobs

    def terminal(self, job):
        self.calls.append(job)
        return self.accounting.get(job)


@pytest.fixture
def release_root(tmp_path, monkeypatch):
    registration = {"frozen": True}
    registration_hash = object_hash(registration)
    monkeypatch.setattr(reporting, "verify_registration", lambda root, **kw: registration)

    def verify_task(root, task, sha):
        assert sha == registration_hash
        return dict(task, receipt_sha256="receipt-" + task["model_id"])

    monkeypatch.setattr(reporting, "verify_task", verify_task)
    from sr_f2 import orchestration
    from sr_f2.protocol import technical_matrix

    old = [run["model_id"] for run in reporting.old_matrix()]
    new = [run["model_id"] for run in reporting.scientific_matrix()]
    matrix = (
        [dict(kind="history", model_id=model) for model in old]
        + [dict(kind="diagnostic", model_id=model) for model in [reporting.BASELINE, *old]]
        + [dict(kind="pilot", model_id=run["model_id"]) for run in technical_matrix()]
        + [dict(kind=kind, model_id=model) for kind in ("train", "endpoint") for model in new]
    )
    specs = [
        dict(
            pool_id=f"GPU_POOL_{index}",
            gpu_count=4,
            queue="global",
            task_matrix_sha256=object_hash(matrix),
        )
        for index in (1, 2)
    ]
    monkeypatch.setattr(reporting, "pool_specs", lambda: specs)
    monkeypatch.setattr(orchestration, "global_task_matrix", lambda: matrix, raising=False)
    monkeypatch.setattr(
        orchestration, "verify_pool_task_ownership", lambda *args: True, raising=False
    )
    config = dict(
        root=str(tmp_path),
        user="alice",
        pools=specs,
        global_tasks=matrix,
        global_task_matrix_sha256=object_hash(matrix),
        strict_seed_waves=False,
    )
    write(tmp_path / "orchestration/CONFIG.json", config)
    states = {}
    slurm = FakeSlurm()
    for i, spec in enumerate(specs):
        job = str(100 + i)
        directory = tmp_path / "orchestration/pools" / spec["pool_id"]
        state = dict(
            status="COMPLETE",
            pool_id=spec["pool_id"],
            gpu_count=4,
            job_id=job,
            registration_sha256=registration_hash,
        )
        states[spec["pool_id"]] = state
        plan = dict(spec, registration_sha256=registration_hash)
        intent = dict(
            pool_id=spec["pool_id"],
            registration_sha256=registration_hash,
            plan_sha256=object_hash(plan),
            job_name="srf2-" + spec["pool_id"],
        )
        complete = dict(
            status="COMPLETE",
            plan_sha256=object_hash(plan),
            tasks=[verify_task(tmp_path, t, registration_hash) for t in matrix[i::2]],
        )
        for name, value in [
            ("STATE", state),
            ("PLAN", plan),
            ("INTENT", intent),
            ("COMPLETE", complete),
        ]:
            write(directory / (name + ".json"), value)
        slurm.accounting[job] = dict(
            job_id=job,
            state="COMPLETED",
            exit_code="0:0",
            user="alice",
            account="rose",
            qos=reporting.TEACHER_QOS,
            name=intent["job_name"],
        )
    write(
        tmp_path / "orchestration/STATUS.json",
        dict(all_gpu_tasks_complete=True, endpoint_seal_release_ready=True, pools=states),
    )
    return tmp_path, slurm


def test_release_requires_all_live_terminal_allocations_and_53_tasks(release_root):
    root, slurm = release_root
    result = verify_release_gate(root, slurm=slurm)
    assert result["status"] == "AUTHORIZED_AFTER_ALL_GPU_TASKS_VERIFIED"
    assert len(result["pools"]) == 2 and len(result["tasks"]) == 53
    assert len(slurm.calls) == 3
    assert result["live_queue_and_accounting_verified"] is True


@pytest.mark.parametrize("problem", ["queued", "accounting_missing", "failed", "wrong_owner"])
def test_complete_files_do_not_bypass_live_scheduler_blockers(release_root, problem):
    root, slurm = release_root
    if problem == "queued":
        slurm.jobs["100"] = {"JobState": "COMPLETING"}
    elif problem == "accounting_missing":
        del slurm.accounting["100"]
    elif problem == "failed":
        slurm.accounting["100"].update(state="FAILED", exit_code="1:0")
    else:
        slurm.accounting["100"]["user"] = "bob"
    with pytest.raises(PermissionError, match=r"squeue|sacct"):
        verify_release_gate(root, slurm=slurm)
    assert not (root / "released/AUTHORIZATION.json").exists()


def test_false_scheduler_status_blocks_before_live_or_scoring_work(release_root):
    root, slurm = release_root
    path = root / "orchestration/STATUS.json"
    status = json.loads(path.read_text())
    status["all_gpu_tasks_complete"] = False
    write(path, status)
    with pytest.raises(PermissionError, match="All GPU"):
        verify_release_gate(root, slurm=slurm)
    assert slurm.calls == []


def test_worker_artifact_failure_is_not_replaced_by_pool_complete(release_root, monkeypatch):
    root, slurm = release_root

    def invalid(*args):
        raise PermissionError("Missing authenticated final checkpoint")

    monkeypatch.setattr(reporting, "verify_task", invalid)
    with pytest.raises(PermissionError, match="checkpoint"):
        verify_release_gate(root, slurm=slurm)


def panel_fixture(root):
    slots = [
        dict(
            slot_id=f"s{i}",
            model_id="model",
            step=96,
            pool="TEST_ID",
            protocol="answer_only",
            view=None,
            qid="q",
            root_id="r",
            family="CROSS",
            chart="line",
            variant="v0",
            draw=i,
            samples_per_prompt=2,
            sampling_seed=100 + i,
            group_seed=100,
            generation=generation_config("answer_only"),
        )
        for i in range(2)
    ]
    group = group_slots(slots)[0]
    rows = [
        dict(
            s,
            source_sha256="source",
            generation_status="COMPLETE",
            technical_validation_errors=[],
            tokens=[1, 2],
            completion_token_count=2,
            sampler_logprobs=[-0.1, -0.2],
        )
        for s in slots
    ]
    raw = dict(group_id=group["group_id"], records=rows, records_sha256=object_hash(rows))
    path = root / "groups" / (group["group_id"] + ".json")
    write(path, raw)
    ref = dict(
        group_id=group["group_id"],
        raw_sha256=reporting.file_hash(path),
        records_sha256=object_hash(rows),
    )
    write(
        root / "COMPLETE.json", dict(status="COMPLETE", model_id="model", records=2, groups=[ref])
    )
    write(
        root / "PLAN.json",
        dict(expected_records=2, slots_sha256=object_hash(slots), source_sha256="source"),
    )
    return slots, path


def test_panel_exact_slot_coverage_and_raw_hash_are_rechecked(tmp_path):
    slots, raw_path = panel_fixture(tmp_path)
    result = list(validated_panel_rows(tmp_path, slots, model_id="model", source_sha256="source"))
    assert len(result) == 2 and result[1][1]["row_index"] == 1
    raw = json.loads(raw_path.read_text())
    raw["records"][0]["tokens"] = [3, 4]
    write(raw_path, raw)
    with pytest.raises(PermissionError, match="checksum"):
        list(validated_panel_rows(tmp_path, slots, model_id="model", source_sha256="source"))


def test_panel_rejects_partial_or_unregistered_group(tmp_path):
    slots, _ = panel_fixture(tmp_path)
    (tmp_path / "groups" / "extra.pending.jsonl").write_text("{}\n")
    with pytest.raises(PermissionError, match="partial"):
        list(validated_panel_rows(tmp_path, slots, model_id="model", source_sha256="source"))


def test_summary_keeps_incorrect_and_unparseable_rows_v1_v2():
    rows = [
        dict(
            model_id="m",
            pool="TEST_ID",
            protocol="free",
            view=None,
            family="CROSS",
            qid="q",
            completion_token_count=10,
            score={"A": 0, "L_answer": 0},
            score_v1={"A": 0, "L_answer": 0},
        ),
        dict(
            model_id="m",
            pool="TEST_ID",
            protocol="free",
            view=None,
            family="CROSS",
            qid="q",
            completion_token_count=30,
            score={"A": 1, "L_answer": 1},
            score_v1={"A": 0, "L_answer": 0},
        ),
    ]
    summary = summarize_rows(rows)[0]
    assert summary["count"] == 2 and summary["questions"] == 1
    assert summary["v2"]["A"] == 0.5 and summary["v2"]["L_answer"] == 0.5
    assert summary["v1"]["A"] == 0 and summary["mean_completion_tokens"] == 20
    rows[0]["score"].pop("A")
    with pytest.raises(PermissionError, match="partially"):
        summarize_rows(rows)


def test_trigger_confusion_uses_observed_paired_endpoint_label(monkeypatch):
    model, control = "SRF1_2_A_s71001", "SRF1_2_J_s71001"
    monkeypatch.setattr(reporting, "old_matrix", lambda: [dict(model_id=model)])
    decisions = {f: dict(R1=f in ("THRESHOLD", "TOPK"), R2=False) for f in reporting.FAMILIES}
    states = {
        model: dict(
            active_families=["THRESHOLD", "TOPK"], history=[dict(step=0, decisions=decisions)]
        )
    }
    rows = [
        dict(
            model_id=m,
            pool="TEST_ID",
            protocol="evidence_answer",
            view=None,
            family=f,
            score={"J": value},
        )
        for f in reporting.FAMILIES
        for m, value in [(model, 0), (control, 1)]
    ]
    result = trigger_summary(states, rows)
    assert result["historical_descriptive_confusion"] == {"FN": 2, "TP": 2}
    assert result["precision"] == 1 and result["recall"] == 0.5
    assert result["matrix"][0]["endpoint_J_deficit_vs_J_pp"] == 100


def test_atomic_jsonl_publish_is_reproducible_and_never_overwrites(tmp_path):
    path = tmp_path / "SCORES.jsonl"
    original = [{"score": {"A": 1}}, {"score": {"A": 0}}]
    sha = _write_jsonl_once(path, original)
    assert _write_jsonl_once(path, original) == sha
    with pytest.raises(PermissionError, match="differ"):
        _write_jsonl_once(path, [{"score": {"A": 0}}])
    assert [json.loads(line) for line in path.read_text().splitlines()] == original
    assert not list(tmp_path.glob("*.partial"))


def test_release_failure_never_scores_or_publishes_complete(tmp_path, monkeypatch):
    def blocked(*args, **kwargs):
        raise PermissionError("unverified scheduler")

    monkeypatch.setattr(reporting, "verify_release_gate", blocked)
    monkeypatch.setattr(
        reporting,
        "cohort_verification",
        lambda root: dict(
            status="PASS", raw_evidence_status="RAW_AND_GOLD_REVERIFIED", receipt_sha256="cohort"
        ),
    )
    monkeypatch.setattr(reporting, "score_new_endpoints", lambda *args: pytest.fail("score ran"))
    with pytest.raises(PermissionError, match="scheduler"):
        reporting.release_and_report(tmp_path, slurm=FakeSlurm())
    assert not (tmp_path / "released/COMPLETE.json").exists()
    assert not (tmp_path / "released/AUTHORIZATION.json").exists()


def test_plot_creates_standalone_nonempty_png(tmp_path):
    rows = [
        dict(arm=arm, family=family, seed=seed, step=step, A=0.9, E=0.8, P=0.7, J=0.6)
        for arm in reporting.ARMS
        for family in reporting.FAMILIES
        for seed in (71001, 71002, 71003)
        for step in range(0, 97, 8)
    ]
    path = tmp_path / "trajectories.png"
    reporting.plot_trajectories(rows, path)
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert path.stat().st_size > 10000


def test_requested122_identity_pass_remains_separate_from_raw_delivery(tmp_path):
    from sr_f2.reporting import cohort_verification

    assert cohort_verification(tmp_path)["status"] == "UNRESOLVED"
    directory = tmp_path / "prior_audit/chartqa122"
    rows = directory / "COHORT_ROWS.jsonl"
    directory.mkdir(parents=True)
    rows.write_text("".join(json.dumps(dict(qid=f"q{i}")) + "\n" for i in range(122)))
    receipt = dict(
        status="PASS",
        cohort_identity_verified=True,
        count=122,
        scheme="whitespace_stripped",
        metric="relaxed_accuracy",
        label="POST_HOC_WHITESPACE_SENSITIVITY",
        official_frozen_scores_changed=False,
        raw_evidence_status="SOURCE_REFERENCES_ONLY",
        rows=dict(path=rows.name, sha256=reporting.file_hash(rows), count=122),
    )
    write(directory / "SOURCE_MANIFEST.json", {"files": []})
    receipt.update(
        baseline_model=reporting.BASELINE,
        loss_models=["SRF1_2_GATE_s71001", "SRF1_2_GATE_s71003", "SRF1_2_DEC_s71003"],
        source_manifest_sha256=reporting.file_hash(directory / "SOURCE_MANIFEST.json"),
    )
    write(directory / "COHORT_VERIFICATION.json", receipt)
    result = cohort_verification(tmp_path)
    assert result["status"] == "PASS" and result["count"] == 122
    assert result["raw_evidence_status"] == "SOURCE_REFERENCES_ONLY"
    assert "does not imply" in result["interpretation"]
    rows.write_text(rows.read_text().replace("q0", "q1"))
    with pytest.raises(PermissionError, match="hash/count"):
        cohort_verification(tmp_path)


def test_renderer_uses_c4_noninferiority_status_not_superiority():
    from sr_f2.reporting import render_report

    contrast = dict(
        contrast="ADAPT-J",
        effect_pp=-0.3,
        CI98_75_pp=[-1.0, 0.4],
        seed_effects_pp=[-0.2, -0.3, -0.4],
        status="NOT_CONFIRMED",
        primary_status="NONINFERIORITY_CONFIRMED",
        criteria_status="MET",
    )
    model = "SRF2_ADAPT_s71001"
    metrics = [
        dict(
            model_id=model,
            pool="TEST_ID",
            protocol="evidence_answer",
            view=None,
            family="TOPK",
            v1=dict(A=0.8, J=0.6),
            v2=dict(A=0.9, J=0.7, E=0.8, E6=0.9),
        ),
        dict(
            model_id=model,
            pool="CHARTQA",
            protocol="plain_answer",
            view=None,
            family="CHARTQA",
            v1=dict(relaxed_accuracy=0.7),
            v2=dict(relaxed_accuracy=0.8),
        ),
    ]
    protocol = dict(model_id=model)
    for name in ("evidence_answer", "answer_only", "free", "list"):
        protocol[name] = dict(A=0.7, parse_rate=0.9, mean_tokens=30)
    interval = dict(model_id=model, **{v: dict(A=0.5) for v in ("V0", "V1", "V2", "V3", "V4")})
    results = dict(
        main=dict(contrasts={"C4": contrast}),
        endpoint_metrics=metrics,
        diagnostics=dict(protocol_table=[protocol], interval_table=[interval]),
        triggers=dict(matrix=[]),
        technical_incidents=[],
        requested_chartqa_cohort=dict(status="PASS", raw_evidence_status="SOURCE_REFERENCES_ONLY"),
    )
    text = render_report(results)
    line = next(line for line in text.splitlines() if line.startswith("| C4"))
    assert "NONINFERIORITY_CONFIRMED" in line and "| MET |" in line
    assert "SOURCE_REFERENCES_ONLY" in text and "事后whitespace_stripped" in text
    assert "80.00%/90.00%" in text and "60.00%/70.00%" in text


def test_final_completion_blocks_source_only_or_unresolved122(tmp_path, monkeypatch):
    monkeypatch.setattr(
        reporting,
        "cohort_verification",
        lambda root: dict(status="PASS", raw_evidence_status="SOURCE_REFERENCES_ONLY"),
    )
    monkeypatch.setattr(
        reporting, "verify_release_gate", lambda *args, **kw: pytest.fail("gate ran")
    )
    with pytest.raises(PermissionError, match="complete raw/gold"):
        reporting.release_and_report(tmp_path)
    assert not (tmp_path / "released/COMPLETE.json").exists()
    assert not (tmp_path / "released/SCORES.jsonl").exists()


def test_advantage_energy_preserves_every_step_and_registered_phases(tmp_path, monkeypatch):
    model = "SRF2_EARLY_s71001"
    monkeypatch.setattr(reporting, "old_matrix", lambda: [])
    monkeypatch.setattr(
        reporting, "scientific_matrix", lambda: [dict(model_id=model, arm="EARLY", seed=71001)]
    )
    metrics = [
        dict(
            step=step,
            arm="EARLY",
            advantage_energy=128,
            reward_advantage_audit=dict(final_advantage=[[1.0] * 8] * 16),
        )
        for step in range(1, 97)
    ]
    for step in metrics:
        write(tmp_path / "runs" / model / "steps" / f"{step['step']:02d}.json", step)
    source = tmp_path / "prior-sources.json"
    write(source, dict(files=[]))
    result = reporting.advantage_energy_report(
        tmp_path, {model: metrics}, {"prior_sources": str(source)}
    )
    assert len(result["records"]) == 96
    phases = result["per_seed_phases"][0]["phases"]
    assert phases["steps1_12"] == dict(steps=12, mean=128, total=1536, zero_steps=0)
    assert phases["steps13_96"]["steps"] == 84
    assert "no added causal regression" in result["role"]
    metrics[0]["advantage_energy"] = 127
    with pytest.raises(PermissionError, match="differs from coefficients"):
        reporting.advantage_energy_report(
            tmp_path, {model: metrics}, {"prior_sources": str(source)}
        )


@pytest.mark.parametrize("problem", ["missing", "duplicate"])
def test_dynamic_pool_task_union_must_be_exact53(release_root, problem):
    root, slurm = release_root
    first = root / "orchestration/pools/GPU_POOL_1/COMPLETE.json"
    second = root / "orchestration/pools/GPU_POOL_2/COMPLETE.json"
    receipt = json.loads(second.read_text())
    if problem == "missing":
        receipt["tasks"].pop()
    else:
        receipt["tasks"].append(json.loads(first.read_text())["tasks"][0])
    write(second, receipt)
    with pytest.raises(PermissionError, match=r"union|duplicated"):
        verify_release_gate(root, slurm=slurm)
