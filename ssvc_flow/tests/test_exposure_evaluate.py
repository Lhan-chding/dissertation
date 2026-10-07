"""CPU fake-backend checks; none are scientific model observations."""

from __future__ import annotations

import contextlib
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.exposure_substitution import evaluate, schema

DESIGN = Path(__file__).parents[1] / "docs/exposure_substitution/design"


def lines(name):
    return [json.loads(line) for line in (DESIGN / "manifests" / name).read_text().splitlines()]


@pytest.fixture
def toy_job(tmp_path, monkeypatch):
    job = {
        "parent": "S96",
        "block": None,
        "arm": "PARENT",
        "step": 0,
        "panel": "E_DIAG",
        "draws": 2,
        "tasks": 2,
        "kind": "frozen_parent_eval",
    }
    tasks = lines("E_DIAG/tasks_public.jsonl")[:2]
    frozen, machine = {"plan_hash": "CPU_FIXTURE"}, {"fixture": True}
    monkeypatch.setattr(evaluate, "_job_context", lambda *_args: (tmp_path, job, machine, frozen))
    monkeypatch.setattr(evaluate, "evaluation_tasks", lambda *_args: copy.deepcopy(tasks))
    return tmp_path, job, tasks


class FakeBackend:
    def __init__(self, *, fail_on=None, raw_text="not JSON", fingerprint="CPU_FIXTURE"):
        self.calls, self.fail_on, self.raw_text = [], fail_on, raw_text
        self.receipt = {"inference_fingerprint": fingerprint}

    def generate_public(self, prompt, seed, max_new_tokens):
        assert set(prompt) == {"system", "user"}
        assert max_new_tokens == 64
        assert not any(key in prompt for key in ("truth", "arm", "audit", "target"))
        self.calls.append({"prompt": copy.deepcopy(prompt), "seed": seed})
        if len(self.calls) == self.fail_on:
            raise RuntimeError("CPU fixture transport failure")
        return {
            "raw_text": self.raw_text,
            "token_ids": [1, 2],
            "stop_reason": "length",
            "generated_length": 2,
            "elapsed_seconds": 0.001,
            "seed": seed,
            "inference_fingerprint": self.receipt["inference_fingerprint"],
        }

    @contextlib.contextmanager
    def factory(self, *_args):
        yield self


def test_exact_95_jobs_and_145280_samples():
    jobs = lines("evaluation_jobs.jsonl")
    evaluate.validate_evaluation_jobs(jobs)
    assert len({evaluate.evaluation_job_id(job) for job in jobs}) == 95
    assert sum(job["tasks"] * job["draws"] for job in jobs) == 145280
    for mutate in (
        lambda rows: rows.append(rows[-1]),
        lambda rows: rows[0].update(draws=9),
        lambda rows: rows[-1].update(panel="E_CONFIRM"),
        lambda rows: rows[0].update(block_seed=108702),
    ):
        altered = copy.deepcopy(jobs)
        mutate(altered)
        with pytest.raises(ValueError):
            evaluate.validate_evaluation_jobs(altered)


def test_seed_binds_every_identity_dimension():
    job = lines("evaluation_jobs.jsonl")[0]
    task = lines("E_DIAG/tasks_public.jsonl")[0]
    identity = evaluate.sample_identity(job, task, 0)
    seed = evaluate.sample_seed(identity)
    assert 0 <= seed < 2**63
    assert evaluate.sample_seed(copy.deepcopy(identity)) == seed
    for key in identity:
        changed = {**identity, key: str(identity[key]) + "changed"}
        assert evaluate.sample_seed(changed) != seed, key
    other = {**job, "arm": "B_FORWARD_C4"}
    assert evaluate.sample_seed(evaluate.sample_identity(other, task, 0)) != seed


def test_wrong_and_truncated_outputs_are_accepted_once(toy_job):
    run, job, _ = toy_job
    backend = FakeBackend()
    receipt = evaluate.evaluate_job(
        run, evaluate.evaluation_job_id(job), backend_factory=backend.factory
    )
    assert receipt["samples"] == 4 and receipt["status"] == "COMPLETE"
    assert len(backend.calls) == 4
    raw, _ = evaluate.validate_completed_evaluation(run, job)
    assert all(row["raw_text"] == "not JSON" and row["stop_reason"] == "length" for row in raw)
    assert all(row["execution_kind"] == "CPU_TEST_FIXTURE" for row in raw)
    assert all("event" not in row and "truth" not in row for row in raw)
    repeat = FakeBackend(raw_text="a preferable answer")
    assert (
        evaluate.evaluate_job(run, evaluate.evaluation_job_id(job), backend_factory=repeat.factory)
        == receipt
    )
    assert not repeat.calls


def test_failed_attempt_kept_and_restart_reuses_seed(toy_job):
    run, job, _ = toy_job
    first = FakeBackend(fail_on=2)
    with pytest.raises(RuntimeError, match="transport failure"):
        evaluate.evaluate_job(run, evaluate.evaluation_job_id(job), backend_factory=first.factory)
    failures = list((run / "evaluations").glob("*/attempts/*/*/FAILED.json"))
    assert len(failures) == 1
    failed_seed = json.loads(failures[0].read_text())["sample_seed"]
    retry = FakeBackend(raw_text="another invalid answer")
    receipt = evaluate.evaluate_job(
        run, evaluate.evaluation_job_id(job), backend_factory=retry.factory
    )
    assert len(retry.calls) == 3 and retry.calls[0]["seed"] == failed_seed
    raw, _ = evaluate.validate_completed_evaluation(run, job)
    assert raw[0]["raw_text"] == "not JSON"
    assert receipt["samples"] == 4 and failures[0].exists()
    assert len({row["request_id"] for row in raw}) == 4


def test_corrupt_accepted_sample_never_resampled(toy_job):
    run, job, _ = toy_job
    backend = FakeBackend()
    evaluate.evaluate_job(run, evaluate.evaluation_job_id(job), backend_factory=backend.factory)
    path = next((run / "evaluations").glob("*/roots/*/*.json"))
    altered = json.loads(path.read_text())
    altered["record"]["raw_text"] = "tampered"
    path.write_text(json.dumps(altered))
    retry = FakeBackend()
    with pytest.raises(ValueError, match="identity/hash mismatch"):
        evaluate.evaluate_job(run, evaluate.evaluation_job_id(job), backend_factory=retry.factory)
    assert not retry.calls and path.exists()


def test_resume_rejects_changed_model_identity(toy_job):
    run, job, _ = toy_job
    first = FakeBackend(fail_on=2)
    with pytest.raises(RuntimeError):
        evaluate.evaluate_job(run, evaluate.evaluation_job_id(job), backend_factory=first.factory)
    other = FakeBackend(fingerprint="DIFFERENT_CHECKPOINT")
    with pytest.raises(ValueError, match="Immutable evaluation artifact changed"):
        evaluate.evaluate_job(run, evaluate.evaluation_job_id(job), backend_factory=other.factory)
    assert not other.calls


def test_real_evaluation_requires_gpu_authorization(toy_job):
    run, job, _ = toy_job
    with pytest.raises(PermissionError, match="allow-gpu"):
        evaluate.evaluate_job(run, evaluate.evaluation_job_id(job))


def test_direct_real_eval_requires_bound_bridge_before_backend(toy_job, monkeypatch):
    from src.exposure_substitution import training

    run, job, _ = toy_job
    checks = []

    def reject_bridge(bound_run):
        checks.append(bound_run)
        raise PermissionError("bridge missing or STOP")

    monkeypatch.setattr(training, "require_bridge", reject_bridge, raising=False)
    with pytest.raises(PermissionError, match="bridge missing or STOP"):
        evaluate.evaluate_job(run, evaluate.evaluation_job_id(job), allow_gpu=True)
    assert checks == [run]
    assert not (run / "evaluations").exists()


def test_confirm_never_scored_early_and_released_run_cannot_retry(toy_job):
    run, job, _ = toy_job
    job.update(panel="E_CONFIRM")
    backend = FakeBackend()
    receipt = evaluate.evaluate_job(
        run, evaluate.evaluation_job_id(job), backend_factory=backend.factory
    )
    assert receipt["sealed"] is True
    with pytest.raises(PermissionError, match="sealed"):
        evaluate.analyze_diagnostic_job(run, evaluate.evaluation_job_id(job))
    assert not list(run.rglob("*SCORED*"))
    (run / "FINAL_RELEASE.json").write_text("{}")
    with pytest.raises(PermissionError, match="Released"):
        evaluate.evaluate_job(run, evaluate.evaluation_job_id(job), backend_factory=backend.factory)


def test_bad_logp_retains_non_json_transport(toy_job):
    run, job, _ = toy_job
    backend = FakeBackend()
    original = backend.generate_public

    def bad(*args):
        return {**original(*args), "chosen_token_logprobs": [float("nan"), 0.0]}

    backend.generate_public = bad
    with pytest.raises(ValueError, match="log probabilities"):
        evaluate.evaluate_job(run, evaluate.evaluation_job_id(job), backend_factory=backend.factory)
    failed = json.loads(next(run.glob("evaluations/*/attempts/*/*/FAILED.json")).read_text())
    assert failed["raw_observation"]["chosen_token_logprobs"][0] == {"non_json_float": "nan"}
    assert not list(run.glob("evaluations/*/roots/*/*.json"))


def freeze_fixture(run, include_panels=False):
    names = ["evaluation_jobs.jsonl", "training_jobs.jsonl"]
    if include_panels:
        names += [
            "fit_sentinels.json",
            "common_targets.jsonl",
            "replay_targets.jsonl",
            "donors_public.jsonl",
            "E_DIAG/tasks_public.jsonl",
            "E_CONFIRM/tasks_public.jsonl",
        ]
    files = {}
    for name in names:
        path = run / "manifests" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((DESIGN / "manifests" / name).read_bytes())
        files[str(path.relative_to(run))] = {"sha256": schema.file_digest(path)}
    (run / "machine.json").write_text("{}")
    files["machine.json"] = {"sha256": schema.file_digest(run / "machine.json")}
    frozen = {
        "schema": "ser-j2-test-fixture",
        "status": "FROZEN",
        "experiment_id": evaluate.EXPERIMENT,
        "files": files,
    }
    frozen["plan_hash"] = schema.digest(frozen)
    (run / "FROZEN_PLAN.json").write_text(json.dumps(frozen))
    return frozen


def test_sentinels_remain_8_common_4_replay_4_arm_matched_donors(tmp_path):
    freeze_fixture(tmp_path, include_panels=True)
    jobs = [job for job in lines("evaluation_jobs.jsonl") if job["panel"] == "TRAIN_FIT"][:6]
    views = {job["arm"]: evaluate.evaluation_tasks(tmp_path, job) for job in jobs}
    assert set(views) == set(evaluate.ARMS)
    for tasks in views.values():
        assert [task["split"] for task in tasks] == ["COMMON_TRAIN"] * 8 + ["REPLAY"] * 4 + [
            "DONOR_TRAIN"
        ] * 4
        assert len(tasks) == 16
        assert [task["task_id"] for task in tasks[:12]] == [
            task["task_id"] for task in views["A_LOCAL_C1"][:12]
        ]
        assert [task["root_id"] for task in tasks[12:]] == [
            task["root_id"] for task in views["A_LOCAL_C1"][12:]
        ]
    assert len({task["task_id"] for tasks in views.values() for task in tasks[12:]}) == 12


def test_frozen_machine_cannot_be_replaced(tmp_path):
    freeze_fixture(tmp_path)
    job_id = evaluate.evaluation_job_id(lines("evaluation_jobs.jsonl")[0])
    with pytest.raises(ValueError, match="frozen machine"):
        evaluate._job_context(tmp_path, job_id, {"runtime_path": "wrong"})


def test_base_adapter_disable_is_scoped_and_explicit(monkeypatch):
    from src.exposure_substitution import runtime
    from src.protocol_state_probes import inference

    module = SimpleNamespace(lora_A={}, disable_adapters=False)

    class Model:
        def modules(self):
            return [module]

        @contextlib.contextmanager
        def disable_adapter(self):
            module.disable_adapters = True
            try:
                yield
            finally:
                module.disable_adapters = False

    backend = SimpleNamespace(
        adapter=SimpleNamespace(model=Model()), receipt={"inference_fingerprint": "S96"}
    )
    monkeypatch.setattr(runtime, "load_parent_backend", lambda *_args, **_kwargs: backend)
    monkeypatch.setattr(inference, "_model_guard", lambda model: "unchanged_tensors")
    job = {"arm": "BASE", "parent": "BASE_NO_ADAPTER", "panel": "E_DIAG"}
    with evaluate._backend_for_job(Path("fixture"), job, {}, True) as loaded:
        assert module.disable_adapters is True
        assert loaded.receipt["all_lora_adapters_disabled"] is True
        assert loaded.receipt["inference_fingerprint"] != "S96"
    assert module.disable_adapters is False


def test_release_requires_all113_terminals_and_records_all_missing(tmp_path, monkeypatch):
    from src.exposure_substitution import queue, reports

    freeze_fixture(tmp_path)
    queue.build_queue(tmp_path)
    with pytest.raises(PermissionError, match="113"):
        evaluate.release_and_analyze(tmp_path)
    assert not (tmp_path / "FINAL_RELEASE.json").exists()
    q = queue.registered_queue(tmp_path)
    q.db.execute("UPDATE jobs SET status='BLOCKED_TECHNICAL'")
    q.db.close()
    delivered = {}

    def report(_output, rows, **kwargs):
        delivered.update(kwargs, rows=rows)
        return {"status": "INCOMPLETE_NOT_CERTIFIED"}

    monkeypatch.setattr(reports, "write_reports", report)
    assert evaluate.release_and_analyze(tmp_path)["status"] == "INCOMPLETE_NOT_CERTIFIED"
    receipt = json.loads((tmp_path / "FINAL_RELEASE.json").read_text())
    assert receipt["all_registered_terminal"] and not receipt["scientific_completion"]
    assert len(receipt["technical_missing"]) == 113
    assert delivered["rows"] == []
    assert delivered["execution_scope"]["completed_formal_generations"] == 0


def test_legacy_alias_is_not_a_valid_ser_terminal(tmp_path):
    from src.exposure_substitution import queue

    freeze_fixture(tmp_path)
    queue.build_queue(tmp_path)
    q = queue.registered_queue(tmp_path)
    q.db.execute("UPDATE jobs SET status='ALIAS'")
    q.db.close()
    with pytest.raises(ValueError, match="no alias"):
        evaluate.release_and_analyze(tmp_path)
    assert not (tmp_path / "FINAL_RELEASE.json").exists()
