import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from mm_core.execution import BudgetLedger, atomic_json, object_hash, read_jsonl, sha256_file

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/mm_core/recover_panel.py"
spec = importlib.util.spec_from_file_location("mm_core_recovery_helper", SCRIPT)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def prefix(root, relative):
    path = root / relative
    data = path.read_bytes() if path.exists() else b""
    return dict(
        path=relative,
        bytes=len(data),
        rows=len(data.splitlines()),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def raw(request):
    return dict(
        **request,
        status="completed",
        raw_text='{"readings":[12,34],"answer":46}',
        tokens=[10, 20],
        raw_tokens=[10, 20],
        truncated=False,
    )


def scored(response):
    return dict(
        **{key: value for key, value in response.items() if key != "status"},
        status="scored",
        self_field_surprisal={},
        gold_teacher_forced_field_nll={},
        scoring_seconds=0.0,
    )


@pytest.fixture
def run(tmp_path, monkeypatch):
    root = tmp_path
    stage = "FORMAT_BASE_TEST"
    freeze = dict(model_path="never-loaded", model_weights_hash="m", processor_hash="p")
    atomic_json(root / "manifests/PRE_INFERENCE_FREEZE.json", freeze)
    atomic_json(
        root / "manifests/RESOURCE_OVERRIDE.json",
        dict(max_concurrent_gpus=5, max_allocated_gpu_hours=None),
    )
    questions = [
        dict(
            question_id=f"q{i}",
            image_id=f"i{i}",
            root_family_id=f"family{i}",
            split="FORMAT_TUNE",
            image_sha256=f"image{i}",
            prompt_sha256=f"prompt{i}",
            true_values=[12, 34],
            operation="sum",
        )
        for i in range(2)
    ]
    jsonl(root / "data/questions.jsonl", questions)
    slots = [
        dict(question_id=f"q{i}", sample_index=k, shard=i, request_id=f"r{i}-{k}", seed=i * 4 + k)
        for i in range(2)
        for k in range(4)
    ]
    plan = dict(
        shards=2,
        stage=stage,
        pre_freeze_hash=sha256_file(root / "manifests/PRE_INFERENCE_FREEZE.json"),
        common_start_hash=None,
        model_hash="m",
        expected_slots=slots,
    )
    atomic_json(root / f"manifests/STAGE_PLAN_{stage}.json", plan)
    requests = {
        slot["request_id"]: recovery.expected_request(
            stage, slot, questions[slot["shard"]], freeze, plan
        )
        for slot in slots
    }
    original_ids = ["r0-0", "r1-0", "r1-1", "r1-2", "r1-3"]
    jsonl(
        root / "raw/REQUESTS.jsonl",
        [dict(**requests[key], status="REQUESTED") for key in original_ids],
    )
    ledger = BudgetLedger(root)
    for key in original_ids:
        ledger.reserve(
            "completion_attempts",
            1,
            dict(question_id=requests[key]["question_id"], seed=requests[key]["seed"]),
        )
    ledger.reserve("extra_forward_sequences", 7, dict(purpose="original_including_unknown_partial"))
    settled, affected = [], {}
    for shard, keys in ((0, ["r0-0"]), (1, ["r1-0", "r1-1", "r1-2"])):
        directory = root / f"raw/{stage}"
        jsonl(directory / f"outputs_{shard}.jsonl", [raw(requests[key]) for key in keys])
        jsonl(
            directory / f"scores_{shard}.jsonl",
            [] if shard == 0 else [scored(raw(requests[key])) for key in keys],
        )
        claim_path = directory / f"SHARD_{shard}_STARTED.json"
        atomic_json(
            claim_path,
            dict(
                stage=stage,
                shard=shard,
                shards=2,
                slurm_job_id=str(100 + shard),
                freeze_hash=plan["pre_freeze_hash"],
                stage_plan_hash=object_hash(plan),
            ),
        )
        allocation_path = f"accounting/allocations/original-{shard}.json"
        atomic_json(
            root / allocation_path,
            dict(
                status="TERMINAL_VERIFIED",
                stage=stage,
                job_id=str(100 + shard),
                allocation_key=f"original-{shard}",
                terminal=dict(job_id=str(100 + shard), state="CANCELLED", squeue_absent=True),
            ),
        )
        settled.append(dict(path=allocation_path, sha256=sha256_file(root / allocation_path)))
        affected[str(shard)] = dict(
            original_job_id=str(100 + shard),
            original_allocation_key=f"original-{shard}",
            retry_allocation_key=f"retry-{shard}",
            original_claim_sha256=sha256_file(claim_path),
            raw_prefix=prefix(root, f"raw/{stage}/outputs_{shard}.jsonl"),
            scores_prefix=prefix(root, f"raw/{stage}/scores_{shard}.jsonl"),
            unknown_request_ids=[] if shard == 0 else ["r1-3"],
        )
    manifest = dict(
        status="REGISTERED",
        retry_index=1,
        max_explicit_technical_retries=1,
        retry_id="one-preemption-event",
        reason="SLURM_PREEMPTION",
        stage=stage,
        shards=2,
        pre_freeze_hash=plan["pre_freeze_hash"],
        model_hash="m",
        common_start_hash=None,
        stage_plan_hash=sha256_file(root / f"manifests/STAGE_PLAN_{stage}.json"),
        helper_sha256=sha256_file(SCRIPT),
        frozen_source_root=str(Path(recovery.execution.__file__).resolve().parent),
        completed_answers_must_not_be_resampled=True,
        refund_unknown_attempts=False,
        requests_prefix=prefix(root, "raw/REQUESTS.jsonl"),
        ledger_prefix=prefix(root, "accounting/COST_LEDGER.jsonl"),
        settled_original_allocations=settled,
        affected_shards=affected,
    )
    manifest_path = root / "manifests/EXPLICIT_TECHNICAL_RETRY.json"
    atomic_json(manifest_path, manifest)
    monkeypatch.setattr(recovery, "verify_gate", lambda *args: freeze)
    monkeypatch.setattr(recovery, "verify_stage_plan", lambda *args: plan)
    state = dict(generated=[], scored_tokens=[], active_shard=0, fail_after_completion=False)
    monkeypatch.setattr(
        recovery,
        "require_allocation",
        lambda *args: dict(allocation_key=f"retry-{state['active_shard']}", job_id="200"),
    )

    class FakeRuntime:
        def __init__(self, *args, account, **kwargs):
            if state.get("fail_load"):
                raise RuntimeError("simulated model loading failure")
            self.account = account

        def verify_identity(self, *args):
            return True

        def prepare(self, question, root):
            return {"question": question}

        def encode_completion(self, text):
            return [30, 40]

        def field_scores(self, prepared, tokens, *, provenance):
            state["scored_tokens"].append((tokens[:], provenance))
            self.account("extra_forward_sequences", 1, dict(purpose=provenance))
            return {"status": "MEASURED"}

        def generate(self, question, root, seed, *, score_fields, on_completion):
            state["generated"].append((question["question_id"], seed))
            self.account(
                "completion_attempts", 1, dict(question_id=question["question_id"], seed=seed)
            )
            result = dict(
                raw_text='{"readings":[12,34],"answer":46}',
                tokens=[10, 20],
                raw_tokens=[10, 20],
                truncated=False,
                seed=seed,
            )
            on_completion(result)
            if state["fail_after_completion"]:
                raise RuntimeError("simulated scoring failure after durable answer")
            result["self_field_surprisal"] = self.field_scores(
                {}, result["tokens"], provenance="self"
            )
            result["gold_teacher_forced_field_nll"] = self.field_scores(
                {}, [30, 40], provenance="gold"
            )
            result["scoring_seconds"] = 0.0
            return result

    monkeypatch.setattr(recovery, "QwenRuntime", FakeRuntime)
    return root, manifest_path, state


def test_two_shards_share_one_retry_preserve_answers_repair_scores_and_keep_unknown_cost(run):
    root, manifest, state = run
    original = {
        shard: (root / f"raw/FORMAT_BASE_TEST/outputs_{shard}.jsonl").read_bytes()
        for shard in range(2)
    }
    claim = {
        shard: sha256_file(root / f"raw/FORMAT_BASE_TEST/SHARD_{shard}_STARTED.json")
        for shard in range(2)
    }
    first = recovery.recover_panel(root, 0, 2, manifest)
    assert first["preserved_completions"] == 1 and first["new_completions"] == 3
    assert first["repaired_score_sidecars"] == 1
    state["active_shard"] = 1
    second = recovery.recover_panel(root, 1, 2, manifest)
    assert second["new_completions"] == 1 and second["repaired_score_sidecars"] == 0
    assert state["generated"] == [("q0", 1), ("q0", 2), ("q0", 3), ("q1", 7)]
    assert state["scored_tokens"][0] == ([10, 20], "generated_tokens_actual_self_prefix")
    for shard in range(2):
        path = root / f"raw/FORMAT_BASE_TEST/outputs_{shard}.jsonl"
        assert path.read_bytes().startswith(original[shard])
        assert len(read_jsonl(path)) == 4
        assert len(read_jsonl(path.with_name(f"scores_{shard}.jsonl"))) == 4
        assert (
            sha256_file(root / f"raw/FORMAT_BASE_TEST/SHARD_{shard}_STARTED.json") == claim[shard]
        )
    totals = BudgetLedger(root).totals()
    assert totals["completion_attempts"] == 9  # Eight retained answers plus one original unknown.
    assert totals["extra_forward_sequences"] == 17  # Retain the original partial forward charge.
    events = read_jsonl(root / "accounting/events.jsonl")
    assert sum(row["event"] == "EXPLICIT_TECHNICAL_RETRY_REGISTERED" for row in events) == 1
    retained = [row for row in events if row["event"] == "ORIGINAL_UNKNOWN_ATTEMPT_RETAINED"]
    assert (
        retained[0]["original_reserved_count"] == 1
        and retained[0]["extra_conservative_reservation"] == 0
    )
    retried = [row for row in read_jsonl(root / "raw/REQUESTS.jsonl") if "attempt_id" in row]
    assert len(retried) == 5 and len({row["attempt_id"] for row in retried}) == 5
    with pytest.raises(PermissionError, match="completed shard"):
        recovery.recover_panel(root, 1, 2, manifest)


@pytest.mark.parametrize("tamper", ["claim", "raw", "helper", "plan", "unknown", "terminal"])
def test_recovery_fails_closed_on_changed_registration_or_evidence(run, tamper):
    root, path, state = run
    manifest = json.loads(path.read_text())
    if tamper == "claim":
        (root / "raw/FORMAT_BASE_TEST/SHARD_0_STARTED.json").write_text("{}")
    elif tamper == "raw":
        output = root / "raw/FORMAT_BASE_TEST/outputs_0.jsonl"
        output.write_bytes(output.read_bytes().replace(b"46", b"47"))
    elif tamper == "helper":
        manifest["helper_sha256"] = "changed"
    elif tamper == "plan":
        manifest["stage_plan_hash"] = "changed"
    elif tamper == "unknown":
        manifest["affected_shards"]["0"]["unknown_request_ids"] = ["not-original"]
    else:
        (root / "accounting/allocations/original-1.json").write_text("{}")
    atomic_json(path, manifest)
    with pytest.raises(PermissionError):
        recovery.recover_panel(root, 0, 2, path)
    assert not state["generated"]


def test_partial_jsonl_tail_is_never_truncated(run):
    root, manifest, _ = run
    path = root / "raw/FORMAT_BASE_TEST/outputs_0.jsonl"
    original = path.read_bytes() + b'{"partial"'
    path.write_bytes(original)
    with pytest.raises(PermissionError):
        recovery.recover_panel(root, 0, 2, manifest)
    assert path.read_bytes() == original


def test_scoring_failure_preserves_new_answer_and_disallows_second_retry(run):
    root, manifest, state = run
    state["fail_after_completion"] = True
    with pytest.raises(RuntimeError, match="scoring failure"):
        recovery.recover_panel(root, 0, 2, manifest)
    path = root / "raw/FORMAT_BASE_TEST/outputs_0.jsonl"
    assert len(read_jsonl(path)) == 2
    assert read_jsonl(path)[-1]["request_id"] == "r0-1"
    assert not (root / "raw/FORMAT_BASE_TEST/SHARD_0_COMPLETE.json").exists()
    with pytest.raises(PermissionError):
        recovery.recover_panel(root, 0, 2, manifest)
    assert state["generated"] == [("q0", 1)]


def test_unreserved_original_unknown_is_conservatively_consumed_once(run):
    root, path, state = run
    ledger_path = root / "accounting/COST_LEDGER.jsonl"
    rows = [row for row in read_jsonl(ledger_path) if row.get("identity", {}).get("seed") != 7]
    jsonl(ledger_path, rows)
    manifest = json.loads(path.read_text())
    manifest["ledger_prefix"] = prefix(root, "accounting/COST_LEDGER.jsonl")
    atomic_json(path, manifest)
    state["active_shard"] = 1
    recovery.recover_panel(root, 1, 2, path)
    retained = [
        row
        for row in read_jsonl(root / "accounting/events.jsonl")
        if row["event"] == "ORIGINAL_UNKNOWN_ATTEMPT_RETAINED"
    ]
    assert retained[0]["extra_conservative_reservation"] == 1
    assert BudgetLedger(root).totals()["completion_attempts"] == 6


@pytest.mark.parametrize("key", ["requests_prefix", "ledger_prefix"])
def test_shared_prefix_check_ignores_a_concurrent_incomplete_tail(run, key):
    root, manifest_path, _ = run
    manifest = json.loads(manifest_path.read_text())
    spec = manifest[key]
    expected = recovery.prefix_rows(root, spec)
    path = root / spec["path"]
    with path.open("ab") as handle:
        handle.write(b'{"concurrent_worker_in_progress":')
    assert recovery.prefix_rows(root, spec) == expected
    with pytest.raises(PermissionError):
        recovery.prefix_rows(root, spec, exact=True)


def test_model_load_failure_has_a_preserved_one_shot_claim_and_failure_event(run):
    root, manifest, state = run
    state["fail_load"] = True
    with pytest.raises(RuntimeError, match="model loading failure"):
        recovery.recover_panel(root, 0, 2, manifest)
    assert (root / "accounting/recovery/SHARD_0_STARTED.json").exists()
    assert not state["generated"]
    events = read_jsonl(root / "accounting/events.jsonl")
    assert events[-1]["event"] == "EXPLICIT_RETRY_SHARD_FAILED"
    with pytest.raises(FileExistsError):
        recovery.recover_panel(root, 0, 2, manifest)


def test_parallel_recovery_workers_share_budget_and_registration_safely(run, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import local

    root, manifest, _ = run
    worker = local()
    monkeypatch.setattr(
        recovery,
        "require_allocation",
        lambda *args: dict(allocation_key=f"retry-{worker.shard}", job_id="200"),
    )

    def work(shard):
        worker.shard = shard
        return recovery.recover_panel(root, shard, 2, manifest)

    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(work, (0, 1)))
    assert [receipt["completed"] for receipt in receipts] == [4, 4]
    assert BudgetLedger(root).totals()["completion_attempts"] == 9
    events = read_jsonl(root / "accounting/events.jsonl")
    assert sum(row["event"] == "EXPLICIT_TECHNICAL_RETRY_REGISTERED" for row in events) == 1
