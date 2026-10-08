"""No-model tests for frozen allocation, durable attempts and exclusive leases."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json

import pytest

from ssvc_flow.src.exposure_position import evaluate as ev
from ssvc_flow.src.exposure_position import queue as q
from ssvc_flow.src.exposure_position import schema


def task_fixture(index=0):
    return dict(
        phase_id=schema.PHASE_ID,
        task_id=f"{index + 1:032x}",
        root_id=f"{index + 100:032x}",
        base_instance_id=f"{index + 100:032x}",
        split="DEV_TRAJECTORY",
        family="cross_series",
        observed=[10, 20, 30, 45],
        H_original=[[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]],
        b_original=[50, 60, 70],
        legal_domain=[0, 99],
        operation="sum4",
        template_version="historical-O0-text-SER-J2-v1",
        chart_type="grouped_bar",
        interface="SYMBOLIC_FRESH",
        image_path=None,
        image_sha256=None,
    )


@pytest.fixture
def toy(tmp_path, monkeypatch):
    job = dict(
        parent="S96",
        block=None,
        logical_arm_id="PARENT",
        arm="PARENT",
        step=0,
        panel="DEV_TRAJECTORY",
        draws=2,
        tasks=2,
        kind="diagnostic_eval",
        source="PARENT",
    )
    tasks = [task_fixture(i) for i in range(2)]
    frozen, machine = {"plan_hash": "CPU_FIXTURE"}, {"fixture": True}
    monkeypatch.setattr(ev, "_job_context", lambda *_: (tmp_path, job, machine, frozen))
    monkeypatch.setattr(ev, "evaluation_tasks", lambda *_: copy.deepcopy(tasks))
    return tmp_path, job, tasks


class FakeBackend:
    def __init__(self, fail_on=None, *, sha="a" * 64, raw="not JSON", crash=False):
        self.calls, self.fail_on, self.raw, self.crash = [], fail_on, raw, crash
        self.receipt = dict(
            checkpoint_sha256=sha,
            source_experiment_id="LEGACY_PHASE",
            source_checkpoint_id="S96",
            inference_fingerprint="CPU_FIXTURE",
        )

    def generate_public(self, prompt, seed, max_tokens):
        assert set(prompt) == {"system", "user"}
        assert max_tokens == 64
        self.calls.append(seed)
        if len(self.calls) == self.fail_on:
            if self.crash:
                raise KeyboardInterrupt("simulated process termination")
            raise RuntimeError("transport failure")
        return dict(
            raw_text=self.raw,
            token_ids=[1, 2],
            stop_reason="length",
            generated_length=2,
            elapsed_seconds=0.001,
            seed=seed,
            inference_fingerprint="CPU_FIXTURE",
        )

    @contextlib.contextmanager
    def factory(self, *_):
        yield self


@pytest.mark.parametrize("mode,expected_train", [("REUSE_12", 12), ("RERUN_24", 24)])
def test_exact_registered_scientific_matrix(mode, expected_train):
    jobs = ev.build_evaluation_jobs(mode)
    assert len(jobs) == 114
    assert sum(j["tasks"] * j["draws"] for j in jobs) == 259656
    for panel, count, total in [
        ("E_CONFIRM2", 26, 206336),
        ("DEV_TRAJECTORY", 36, 36864),
        ("TRAIN_FIT_FULL", 24, 4416),
        ("DEV_DELTA", 28, 12040),
    ]:
        selected = [j for j in jobs if j["panel"] == panel]
        assert len(selected) == count
        assert sum(j["tasks"] * j["draws"] for j in selected) == total
    parents = [j for j in jobs if j["panel"] == "E_CONFIRM2" and j["logical_arm_id"] == "PARENT"]
    assert len(parents) == 2 and all(j["block"] is None for j in parents)
    arms = tuple(schema.ARMS) if mode == "RERUN_24" else tuple(schema.ARMS)[2:]
    training = [
        dict(parent=p, block=b, logical_arm_id=a)
        for p in schema.PARENTS
        for b in range(3)
        for a in arms
    ]
    assert len(q.validate_training_jobs(training, mode)) == expected_train
    for mutation in (
        lambda x: x.pop(),
        lambda x: x.append(x[0]),
        lambda x: x[0].update(draws=9),
        lambda x: x[0].update(source="LEGACY" if mode == "RERUN_24" else "NEW_TRAINING"),
    ):
        altered = copy.deepcopy(jobs)
        mutation(altered)
        with pytest.raises(ValueError):
            ev.validate_evaluation_jobs(altered, mode=mode)


def test_missing_legacy_dev_reduces_only_registered_cells():
    full = ev.build_evaluation_jobs("RERUN_24")
    missing = [
        dict(job_id=ev.evaluation_job_id(j), reason="checkpoint verified unavailable before freeze")
        for j in full
        if j["panel"] == "DEV_DELTA"
        and j["logical_arm_id"] == ev.A2
        and j["parent"] == "S96"
        and j["block"] == 0
    ]
    assert len(missing) == 2
    jobs = ev.build_evaluation_jobs("RERUN_24", unavailable=missing)
    assert sum(j["tasks"] * j["draws"] for j in jobs) == 259656 - (106 + 109) * 4
    with pytest.raises(ValueError, match="all affected"):
        ev.build_evaluation_jobs("RERUN_24", unavailable=missing[:1])
    with pytest.raises(ValueError, match="Only unavailable"):
        ev.build_evaluation_jobs(
            "RERUN_24", unavailable=[dict(job_id=ev.evaluation_job_id(full[0]), reason="missing")]
        )
    jobs[0]["draws"] += 1
    with pytest.raises(ValueError):
        ev.validate_evaluation_jobs(jobs, mode="RERUN_24", unavailable=missing)


def test_seed_is_exact_registered_canonical_json_sha(toy):
    _, job, tasks = toy
    model = FakeBackend().receipt
    identity = ev.sample_identity(job, tasks[0], 0, model)
    expected = (
        int(
            hashlib.sha256(
                json.dumps(
                    {k: identity[k] for k in ev.SEED_FIELDS}, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()[:16],
            16,
        )
        % 2**63
    )
    assert ev.sample_seed(identity) == expected
    assert ev.sample_seed({**identity, "source_checkpoint_id": "source alias only"}) == expected
    for key, value in [
        ("checkpoint_sha256", "b" * 64),
        ("logical_arm_id", ev.A3),
        ("panel", "E_CONFIRM2"),
        ("task_id", "c" * 32),
        ("draw_index", 1),
    ]:
        assert ev.sample_seed({**identity, key: value}) != expected
    with pytest.raises(ValueError):
        ev.sample_identity(job, tasks[0], True, model)
    with pytest.raises(ValueError):
        ev.sample_identity(job, tasks[0], 0, {**model, "checkpoint_sha256": "placeholder"})


def test_raw_invalid_answers_written_once_without_truth_or_scorer(toy, monkeypatch):
    run, job, _ = toy

    def forbidden(*_args, **_kwargs):
        raise AssertionError("generation attempted audit/scoring")

    monkeypatch.setattr(schema.RoleDataset, "audit", forbidden)
    monkeypatch.setattr(ev, "score_rows", forbidden)
    backend = FakeBackend()
    receipt = ev.evaluate_job(run, ev.evaluation_job_id(job), backend_factory=backend.factory)
    rows, _ = ev.validate_completed_evaluation(run, job)
    assert receipt["samples"] == len(backend.calls) == 4
    assert all(r["raw_text"] == "not JSON" and r["stop_reason"] == "length" for r in rows)
    assert all(r["execution_kind"] == "CPU_TEST_FIXTURE" for r in rows)
    assert all(
        "truth" not in r and "event" not in r and "chosen_token_logprobs" not in r for r in rows
    )
    assert len({r["sample_id"] for r in rows}) == 4
    again = FakeBackend(raw="correct now")
    assert ev.evaluate_job(run, ev.evaluation_job_id(job), backend_factory=again.factory) == receipt
    assert again.calls == []


def test_failure_retained_and_resume_same_seed(toy):
    run, job, _ = toy
    first = FakeBackend(fail_on=2)
    with pytest.raises(RuntimeError, match="transport failure"):
        ev.evaluate_job(run, ev.evaluation_job_id(job), backend_factory=first.factory)
    failed = next((run / "evaluations").glob("*/attempts/*/*/FAILED.json"))
    second = FakeBackend()
    ev.evaluate_job(run, ev.evaluation_job_id(job), backend_factory=second.factory)
    assert len(second.calls) == 3 and second.calls[0] == first.calls[1]
    assert json.loads(failed.read_text())["sample_seed"] == second.calls[0]
    assert ev.generation_accounting(run)["physical_generation_attempts_started"] == 5
    assert ev.generation_accounting(run)["failed_generation_attempts"] == 1


def test_unknown_attempt_blocks_until_explicit_process_evidence(toy):
    run, job, _ = toy
    first = FakeBackend(fail_on=1, crash=True)
    with pytest.raises(KeyboardInterrupt):
        ev.evaluate_job(run, ev.evaluation_job_id(job), backend_factory=first.factory)
    retry = FakeBackend()
    with pytest.raises(PermissionError, match="UNKNOWN"):
        ev.evaluate_job(run, ev.evaluation_job_id(job), backend_factory=retry.factory)
    assert not retry.calls
    started = next((run / "evaluations").glob("*/attempts/*/*/STARTED.json"))
    with pytest.raises(PermissionError):
        ev.acknowledge_unknown_attempt(
            started.parent,
            reason="ssh timeout",
            old_process_stopped=False,
            scheduler_state="UNKNOWN",
        )
    ev.acknowledge_unknown_attempt(
        started.parent,
        reason="scheduler confirms failed and worker gone",
        old_process_stopped=True,
        scheduler_state="FAILED",
    )
    ev.evaluate_job(run, ev.evaluation_job_id(job), backend_factory=retry.factory)
    assert retry.calls[0] == first.calls[0]
    assert ev.generation_accounting(run)["generation_attempts_with_unknown_completion"] == 1


def test_tampering_stops_without_resampling(toy):
    run, job, _ = toy
    ev.evaluate_job(run, ev.evaluation_job_id(job), backend_factory=FakeBackend().factory)
    path = next((run / "evaluations").glob("*/roots/*/*.json"))
    row = json.loads(path.read_text())
    row["record"]["raw_text"] = "tampered"
    path.write_text(json.dumps(row))
    retry = FakeBackend()
    with pytest.raises(ValueError, match="identity/hash mismatch"):
        ev.evaluate_job(run, ev.evaluation_job_id(job), backend_factory=retry.factory)
    assert retry.calls == []


def test_changed_checkpoint_cannot_resume_accepted_work(toy):
    run, job, _ = toy
    with pytest.raises(RuntimeError):
        ev.evaluate_job(
            run, ev.evaluation_job_id(job), backend_factory=FakeBackend(fail_on=2).factory
        )
    changed = FakeBackend(sha="b" * 64)
    with pytest.raises(ValueError, match="Immutable"):
        ev.evaluate_job(run, ev.evaluation_job_id(job), backend_factory=changed.factory)
    assert not changed.calls


def test_confirm_scorer_and_post_release_generation_forbidden(toy):
    run, job, _ = toy
    job["panel"] = "E_CONFIRM2"
    with pytest.raises(PermissionError, match="sealed"):
        ev.analyze_diagnostic_job(run, ev.evaluation_job_id(job))
    ev.write_once(run / "RELEASE_RECEIPT.json", {"status": "RELEASED"})
    with pytest.raises(PermissionError, match="Released"):
        ev.evaluate_job(run, ev.evaluation_job_id(job), backend_factory=FakeBackend().factory)


def queue_job(jid, block=0, arm=ev.A3, dependencies=()):
    return dict(
        id=jid,
        kind="train",
        parent="S96",
        block=block,
        logical_arm_id=arm,
        gpu_count=1,
        dependencies=list(dependencies),
    )


def test_queue_idempotence_gpu_student_and_unknown_leases(tmp_path):
    queue = q.Queue(tmp_path / "queue.sqlite", maximum=2)
    queue.register(
        [queue_job("one"), queue_job("same-student"), queue_job("other", block=1)], "fixture"
    )
    assert q.claim_registered(queue, "one", "worker1", gpu_key="host:GPU0")["id"] == "one"
    assert q.claim_registered(queue, "one", "worker1", gpu_key="host:GPU0") is None
    for jid, gpu in [("other", "host:GPU0"), ("same-student", "host:GPU1")]:
        with pytest.raises(q.ResourceUnavailable):
            q.claim_registered(queue, jid, "worker2", gpu_key=gpu)
    queue.mark_unknown("one", "worker1", reason="SSH disconnected")
    with pytest.raises(q.ResourceUnavailable):
        q.claim_registered(queue, "one", "worker2", gpu_key="host:GPU1")
    with pytest.raises(PermissionError):
        queue.retry("one", reason="timeout does not establish stopped")
    queue.retry(
        "one",
        reason="scheduler NODE_FAIL and no worker",
        old_process_stopped=True,
        scheduler_state="NODE_FAIL",
    )
    assert q.claim_registered(queue, "one", "worker3", gpu_key="host:GPU0")
    queue.finish("one", "worker3", "COMPLETE", {"receipt": "fixed"})
    assert queue.db.execute("SELECT COUNT(*) FROM events WHERE event='UNKNOWN'").fetchone()[0] == 1
    assert (
        queue.db.execute("SELECT COUNT(*) FROM events WHERE event='EXPLICIT_RETRY'").fetchone()[0]
        == 1
    )
    queue.db.close()


def test_queue_missing_dependencies_and_blocked_matrix_cannot_release(tmp_path):
    queue = q.Queue(tmp_path / "queue.sqlite")
    queue.register(
        [queue_job("train"), queue_job("eval", block=1, dependencies=["train"])], "fixture"
    )
    with pytest.raises(q.ResourceUnavailable):
        q.claim_registered(queue, "eval", "worker", gpu_key="host:GPU0")
    q.claim_registered(queue, "train", "worker", gpu_key="host:GPU0")
    queue.finish("train", "worker", "BLOCKED_TECHNICAL", {"reason": "retained"})
    with pytest.raises(PermissionError, match="COMPLETE"):
        queue.seal_release()
    with pytest.raises(q.ResourceUnavailable):
        q.claim_registered(queue, "eval", "worker", gpu_key="host:GPU0")
    queue.db.close()


def test_atomic_write_once_never_overwrites(tmp_path):
    path = tmp_path / "raw.json"
    ev.write_once(path, {"raw": "invalid answer"})
    ev.write_once(path, {"raw": "invalid answer"})
    with pytest.raises(ValueError, match="Immutable"):
        ev.write_once(path, {"raw": "nicer answer"})
    assert json.loads(path.read_text()) == {"raw": "invalid answer"}


def test_legacy_alias_preserves_source_checkpoint_identity(tmp_path, monkeypatch):
    from ssvc_flow.src.exposure_position import runtime

    path = tmp_path / "old-checkpoint.pt"
    path.write_bytes(b"synthetic serialized checkpoint bytes")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    source_identity = dict(
        experiment_id="SER_J2_20261007",
        parent="S96",
        block=0,
        arm="A_LOCAL_C1",
        technical_only=False,
    )
    state = dict(identity=copy.deepcopy(source_identity), step=256)
    monkeypatch.setattr(runtime, "read_student_checkpoint", lambda *_: copy.deepcopy(state))
    job = next(j for j in ev.build_evaluation_jobs("REUSE_12") if j["logical_arm_id"] == ev.A2)
    job["checkpoint_binding"] = dict(
        path=str(path),
        checkpoint_sha256=sha,
        source_experiment_id="SER_J2_20261007",
        source_checkpoint_id="S96.block0.A_LOCAL_C1.step256",
        expected_identity=source_identity,
    )
    before = path.read_bytes()
    binding = ev.resolve_checkpoint_binding(tmp_path, job)
    assert binding["source_experiment_id"] == "SER_J2_20261007"
    assert binding["expected_identity"]["arm"] == "A_LOCAL_C1"
    assert path.read_bytes() == before and state["identity"] == source_identity
    altered = copy.deepcopy(job)
    altered["checkpoint_binding"]["expected_identity"]["experiment_id"] = schema.PHASE_ID
    with pytest.raises(ValueError, match="source identity"):
        ev.resolve_checkpoint_binding(tmp_path, altered)
    altered = copy.deepcopy(job)
    altered["logical_arm_id"] = ev.B2
    with pytest.raises(ValueError, match="source arm"):
        ev.resolve_checkpoint_binding(tmp_path, altered)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes differ"):
        ev.resolve_checkpoint_binding(tmp_path, job)


def test_accepted_missing_draw_never_treated_as_complete(toy):
    run, job, _ = toy
    first = FakeBackend(fail_on=4)
    with pytest.raises(RuntimeError):
        ev.evaluate_job(run, ev.evaluation_job_id(job), backend_factory=first.factory)
    with pytest.raises(ValueError, match="Missing registered sample"):
        ev.validate_completed_evaluation(run, job)
    rows, _ = ev.validate_completed_evaluation(run, job, require_complete=False)
    assert len(rows) == 3


def test_queue_unknown_counts_toward_maximum_even_on_other_gpu(tmp_path):
    queue = q.Queue(tmp_path / "queue.sqlite", maximum=1)
    queue.register([queue_job("one"), queue_job("two", block=1)], "fixture")
    q.claim_registered(queue, "one", "w1", gpu_key="host:GPU0")
    queue.mark_unknown("one", "w1", reason="network disconnect")
    with pytest.raises(q.ResourceUnavailable, match="concurrency"):
        q.claim_registered(queue, "two", "w2", gpu_key="host:GPU1")
    queue.db.close()
