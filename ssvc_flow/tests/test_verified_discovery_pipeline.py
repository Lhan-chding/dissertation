"""Synthetic CPU orchestration fixtures, never model evidence."""

import copy
import json

import pytest

from src.verified_discovery_transfer import cli
from src.verified_discovery_transfer.config import digest as data_digest
from src.verified_discovery_transfer.config import load_protocol
from src.verified_discovery_transfer.evaluate import release_test
from src.verified_discovery_transfer.fresh_cohort import generate_cohort
from src.verified_discovery_transfer.queue import Queue, digest, read_json, write_json
from src.verified_discovery_transfer.statistics import finite_panel_mcse, paired_cluster_bootstrap


@pytest.fixture(scope="module")
def generated():
    return generate_cohort(
        {
            "schema": "vdt-historical-exclusion-v1",
            "complete": True,
            "sources": [{"path": "CPU_FIXTURE", "sha256": "0" * 64, "scope": "fixture"}],
            "truth_orbits": [],
            "exposed_scene_count": 0,
        }
    )[0]


@pytest.fixture
def run(tmp_path, generated):
    from src.verified_discovery_transfer.config import file_digest

    root = tmp_path / "run"
    root.mkdir()
    cohort = root / "cohort"
    cohort.mkdir()
    files = {}

    def rows_file(name, rows):
        path = cohort / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        files[name] = {"sha256": file_digest(path), "rows": len(rows)}

    for split in ("T_train", "V_selection", "E_test", "G_guard"):
        tasks = (
            [t for t in generated[split]["public"] if t["interface"] == "SYMBOLIC_FRESH"][:2]
            if split == "G_guard"
            else [
                next(t for t in generated[split]["public"] if t["family"] == f)
                for f in ("cross_series", "trend")
            ]
        )
        ids = {t["task_id"] for t in tasks}
        rows_file(split + "/tasks_public.jsonl", tasks)
        rows_file(
            split + "/audit_only.jsonl",
            [a for a in generated[split]["audit"] if a["task_id"] in ids],
        )
    replay = []
    for task in generated["T_train"]["public"][:2]:
        task = copy.deepcopy(task)
        task["split"] = "R_replay"
        truth = next(
            a["true_world"]
            for a in generated["T_train"]["audit"]
            if a["task_id"] == task["task_id"]
        )
        replay.append({"task": task, "canonical_vector": truth})
    rows_file("R_replay/verified_targets.jsonl", replay)
    rows_file("R_replay/tasks_public.jsonl", [r["task"] for r in replay])
    manifest = {"status": "FROZEN", "files": files}
    manifest["manifest_digest"] = data_digest(manifest)
    write_json(cohort / "cohort_manifest.json", manifest)
    plan = load_protocol()
    machine = {"max_concurrent_gpus": 5}
    write_json(root / "protocol.json", plan)
    write_json(root / "machine.json", machine)
    write_json(
        root / "freeze.json",
        {
            "protocol_hash": digest(plan),
            "machine_hash": digest(machine),
            "run_identity": "CPU_FIXTURE",
            "parent_bindings": {"S96": {"sha256": "0" * 64}, "REP96": {"sha256": "1" * 64}},
            "input_hashes": {},
        },
    )
    return root


class FakeBackend:
    def __init__(self, fail_on=None):
        self.calls = 0
        self.fail_on = fail_on
        self.receipt = {"inference_fingerprint": "CPU_FAKE_NOT_MODEL_EVIDENCE"}

    def generate_public(self, prompt, seed, max_new_tokens):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("injected interruption")
        return {"raw_text": "failed normally", "token_ids": [1], "elapsed_seconds": 0.001}


def teacher_job(run, split="T_train", repeat=0):
    from src.verified_discovery_transfer.public_tasks import RoleDataset

    role = {"T_train": "teacher_train", "V_selection": "selector", "E_test": "teacher_eval"}[split]
    tasks = RoleDataset(run / "cohort", role).public(split)
    return {
        "id": f"teacher.S96.{split}.{repeat}.0",
        "kind": "teacher",
        "parent": "S96",
        "split": split,
        "repeat": repeat,
        "role": role,
        "task_ids": [t["task_id"] for t in tasks],
        "draws": 16,
        "dependencies": [],
    }


def test_atomic_resume_retains_every_failed_answer(run):
    job = teacher_job(run)
    backend = FakeBackend(fail_on=47)
    with pytest.raises(RuntimeError, match="injected"):
        cli._generate(run, job, backend)
    assert len(list((run / "evidence" / job["id"] / "atomic").glob("*.json"))) == 46
    resumed = FakeBackend()
    result = cli._generate(run, job, resumed)
    assert result["rows"] == 80 and resumed.calls == 34
    records = cli._records(run, "S96", "T_train", 0)
    assert len(records) == 80 and not any(r["public_verifier_pass"] for r in records)
    assert len({r["request_id"] for r in records}) == 80


def test_generation_selection_discovery_and_empty_training_integration(run, monkeypatch):
    for split in ("V_selection", "T_train"):
        cli._generate(run, teacher_job(run, split), FakeBackend())
    queue = Queue(run / "queue.sqlite")
    cli._dispatch(
        run, {"id": "select.S96", "kind": "select", "parent": "S96", "dependencies": []}, queue
    )
    cli._dispatch(
        run,
        {
            "id": "discover.S96.0",
            "kind": "discover",
            "parent": "S96",
            "repeat": 0,
            "dependencies": [],
        },
        queue,
    )
    job = {
        "id": "sft.S96.0.SELF_MIX",
        "kind": "sft",
        "parent": "S96",
        "repeat": 0,
        "arm": "SELF_MIX",
        "seed": 73101,
    }
    monkeypatch.setattr(cli, "_bridge_passed", lambda run: None)
    assert cli._dispatch(run, job, queue)["status"] == "NO_VERIFIED_TARGETS_RETURN_PARENT"
    focus, replay = cli._training_rows(run, {**job, "id": "sft.S96.0.GOLD_ALL", "arm": "GOLD_ALL"})
    assert len(focus) == len(replay) == 2 and all(
        isinstance(r["target"], str) for r in focus + replay
    )
    assert (
        cli._training_rows(run, {**job, "id": "sft.S96.0.GOLD_MATCH_MIX", "arm": "GOLD_MATCH_MIX"})[
            0
        ]
        == []
    )


def test_eval_repeat_one_uses_student_identity(run):
    job = {
        "id": "eval.S96.1.SELF_MIX.V_selection.64",
        "kind": "evaluate",
        "parent": "S96",
        "repeat": 1,
        "arm": "SELF_MIX",
        "source_job": "sft.S96.1.SELF_MIX",
        "split": "V_selection",
        "step": 64,
        "draws": 4,
    }
    assert cli._generate(run, job, FakeBackend())["rows"] == 8


def test_registered_matrix_and_teacher_gold_priorities(run):
    result = cli.build_queue(run)
    assert result["sft_jobs"] == 20
    queue = Queue(run / "queue.sqlite")
    a = queue.claim("one", blocked_kinds=("sft", "r0"))
    b = queue.claim("two", blocked_kinds=("sft", "r0"))
    assert {a["parent"], b["parent"]} == {"S96", "REP96"}
    c = queue.claim("three")
    assert c["kind"] == "sft" and c["arm"] == "GOLD_ALL"
    assert sum(r["payload"]["kind"] == "r0" for r in queue.rows()) == 4


def test_alias_order_accepts_independent_failure_and_gpu_limit(tmp_path):
    queue = Queue(tmp_path / "q", maximum=2)
    queue.register(
        [
            {"id": "gold", "kind": "sft"},
            {"id": "self", "kind": "sft", "after": ["gold"]},
            {"id": "other", "kind": "teacher"},
        ],
        "identity",
    )
    assert queue.claim("a")["id"] == "gold"
    assert queue.claim("b")["id"] == "other"
    assert queue.claim("c") is None
    queue.finish("gold", "a", "BLOCKED_TECHNICAL", {"reason": "unrelated"})
    assert queue.claim("c")["id"] == "self"
    with pytest.raises(ValueError):
        queue.register([{"id": "gold", "kind": "other"}], "identity")


def test_sealed_release_and_immutable_inputs(run):
    jobs = [{"id": "a", "kind": "teacher"}]
    cohort = read_json(run / "cohort/cohort_manifest.json")["manifest_digest"]
    identity = digest({"run": "CPU_FIXTURE", "cohort": cohort, "jobs": jobs})
    queue = Queue(run / "queue.sqlite")
    queue.register(jobs, identity)
    with pytest.raises(PermissionError):
        release_test(run, queue)
    queue.claim("a")
    queue.finish("a", "a", "BLOCKED_TECHNICAL", {"reason": "missing"})
    write_json(
        run / "REGISTERED_MATRIX.json",
        {"identity": identity, "cohort_manifest_digest": cohort, "jobs": jobs},
    )
    assert release_test(run, queue)["scientific_completion"] is False
    with pytest.raises(PermissionError):
        queue.retry("a", reason="cannot retry after release")
    with pytest.raises(PermissionError):
        queue.claim("new")
    write_json(run / "machine.json", {"changed": True})
    with pytest.raises(ValueError):
        release_test(run, queue)


def test_shared_stream_covariance():
    rows = [
        {"family": f, "base_instance_id": f + str(i), "unit": unit, "left": 0.6, "right": 0.5}
        for f in ("cross", "trend")
        for i in range(2)
        for unit in ("S96:0", "S96:1", "REP96:0", "REP96:1")
    ]
    assert paired_cluster_bootstrap(rows, replicates=100)["difference"] == pytest.approx(0.1)
    assert (
        finite_panel_mcse(
            [
                {"stream_id": "same", "successes": 4, "draws": 8, "coefficient": 1},
                {"stream_id": "same", "successes": 4, "draws": 8, "coefficient": -1},
            ]
        )["mcse"]
        == 0
    )


def test_r0_zero_advantage_executes_adam_momentum():
    import torch

    from src.verified_discovery_transfer.r0_reference import zero_gradient_adam_step

    a = torch.nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.AdamW([a], lr=1e-5, weight_decay=0.0)
    a.grad = torch.ones_like(a)
    opt.step()
    before = a.detach().clone()
    zero_gradient_adam_step(opt, [a], 1e-5)
    assert opt.state[a]["step"] == 2 and not torch.equal(a, before)


def test_whole_missing_chunk_cannot_pass_final_gate(run):
    from src.verified_discovery_transfer.reports import validate_completed_artifacts

    job = teacher_job(run, "E_test")
    result = cli._generate(run, job, FakeBackend())
    queue = Queue(run / "queue.sqlite")
    queue.register([job], "fixture")
    queue.claim("test")
    queue.finish(job["id"], "test", "COMPLETE", result)
    assert validate_completed_artifacts(run, queue)
    next((run / "sealed" / job["id"] / "chunks").glob("*.json")).unlink()
    with pytest.raises(ValueError, match="panel is incomplete"):
        validate_completed_artifacts(run, queue)


def test_strict_nonempty_alias_avoids_model_reload(run, monkeypatch):
    import src.verified_discovery_transfer.sft_runner as runner
    from src.verified_discovery_transfer.canonical_targets import gold_targets
    from src.verified_discovery_transfer.public_tasks import RoleDataset

    data = RoleDataset(run / "cohort", "gold")
    tasks = data.public("T_train")
    targets = gold_targets(tasks, data.audit("T_train"))
    self_targets = {
        tid: {
            **target,
            "label_source_provenance": "self_public_verifier",
            "discovery_request_ids": ["CPU_FIXTURE"],
        }
        for tid, target in targets.items()
    }
    write_json(run / "evidence/discover.S96.0/discovery.json", {"J_O0": self_targets})
    monkeypatch.setattr(cli, "_bridge_passed", lambda run: None)
    backend = FakeBackend()
    backend.receipt["checkpoint"] = {"sha256": "0" * 64}
    monkeypatch.setattr(cli, "_backend", lambda run, parent: backend)
    monkeypatch.setattr(
        runner,
        "run_sft",
        lambda backend, **kwargs: {
            "status": "SFT_256_COMPLETE",
            "identity": kwargs["identity"],
            "checkpoints": [],
        },
    )
    gold = {
        "id": "sft.S96.0.GOLD_ALL",
        "kind": "sft",
        "parent": "S96",
        "repeat": 0,
        "arm": "GOLD_ALL",
        "seed": 73101,
    }
    self_job = {**gold, "id": "sft.S96.0.SELF_O0", "arm": "SELF_O0"}
    queue = Queue(run / "queue.sqlite")
    queue.register([gold, self_job], "fixture")
    queue.claim("one")
    result = cli._dispatch(run, gold, queue)
    queue.finish(gold["id"], "one", "COMPLETE", result)
    monkeypatch.setattr(cli, "_backend", lambda *args: pytest.fail("Alias must not load model"))
    alias = cli._dispatch(run, self_job, queue)
    assert alias["status"] == "ALIAS" and alias["alias_of"] == gold["id"]


def test_final_report_reads_semantic_outcomes_only_after_release(run):
    from src.verified_discovery_transfer.reports import report

    source = {
        "id": "sft.S96.0.SELF_O0",
        "kind": "sft",
        "parent": "S96",
        "repeat": 0,
        "arm": "SELF_O0",
    }
    evaluation = {
        "id": "eval.S96.0.SELF_O0.E_test.256",
        "kind": "evaluate",
        "parent": "S96",
        "repeat": 0,
        "arm": "SELF_O0",
        "source_job": source["id"],
        "split": "E_test",
        "step": 256,
        "draws": 8,
    }
    generated = cli._generate(run, evaluation, FakeBackend())
    jobs = [source, evaluation]
    cohort = read_json(run / "cohort/cohort_manifest.json")["manifest_digest"]
    identity = digest({"run": "CPU_FIXTURE", "cohort": cohort, "jobs": jobs})
    queue = Queue(run / "queue.sqlite")
    queue.register(jobs, identity)
    queue.claim("one")
    queue.finish(source["id"], "one", "COMPLETE", {"status": "COMPLETE", "identity": {}})
    queue.claim("two")
    queue.finish(evaluation["id"], "two", "COMPLETE", generated)
    write_json(
        run / "REGISTERED_MATRIX.json",
        {"identity": identity, "cohort_manifest_digest": cohort, "jobs": jobs},
    )
    assert report(run)["test_results_revealed"] is False
    assert not (run / "FINAL_ENDPOINTS.json").exists()
    assert report(run, reveal=True)["test_results_revealed"] is True
    endpoint = read_json(run / "FINAL_ENDPOINTS.json")[evaluation["id"]]
    assert endpoint["pX_family_equal"] == 0 and endpoint["parent_delta"] is None
    semantic = read_json(run / "final_semantics" / (evaluation["id"] + ".json"))
    assert len(semantic["records"]) == 16 and semantic["audit_only"]
