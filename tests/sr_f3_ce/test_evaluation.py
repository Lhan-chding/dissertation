import json
from collections import Counter
from pathlib import Path

import pytest

from sr_f3_ce import contract
from sr_f3_ce.evaluation import (
    all_monitor_slots,
    diagnostic_input,
    endpoint_models,
    endpoint_slots,
    engine_slots,
    evaluation_budget,
    group_slots,
    monitor_slots,
    object_hash,
    release_evaluation,
    run_group,
    score_record,
    sha_file,
    summarize_scores,
    validate_coverage,
    verify_panel_cache,
)

PLAN = Path(__file__).resolve().parents[2] / "artifacts/sr_f3_ce_20261016/plan"


@pytest.fixture(scope="module")
def tasks():
    return {
        x["qid"]: x
        for x in map(
            json.loads, (PLAN / "manifests/TASKS_AUDIT_ONLY.jsonl").read_text().splitlines()
        )
    }


def test_exact_fixed_slot_budgets(tasks):
    engine = engine_slots(tasks)
    assert len(engine) == 2880
    assert Counter(s["view"] is None for s in engine) == {True: 2304, False: 576}
    assert set(Counter(s["model_id"] for s in engine).values()) == {960}
    assert len(all_monitor_slots(tasks)) == 26112
    slots = endpoint_slots(tasks)
    assert Counter(s["protocol"] for s in slots) == {
        "structured": 161280,
        "answer_only": 15360,
        "natural_work": 15360,
    }
    assert len({s["slot_id"] for s in slots}) == 192000
    assert all(s["sealed"] for s in slots)
    assert len(group_slots(slots)) == 48000
    assert evaluation_budget()["scientific_and_evaluation_generations"] == 357756
    assert evaluation_budget(0)["scientific_and_evaluation_generations"] == 353916


def test_monitor_shared_parent_and_no_intermediate_endpoint(tasks):
    assert len(monitor_slots(tasks, "SRF2_J6_s71001", 0)) == 512
    with pytest.raises(ValueError, match="shared once"):
        monitor_slots(tasks, "SRF3CE_EXEC_b0", 0)
    with pytest.raises(ValueError):
        monitor_slots(tasks, "SRF3CE_EXEC_b0", 8)
    bad = [
        dict(
            run_id=m["model_id"],
            block=m["block"],
            arm=m["arm"],
            parent_id=m["parent_id"],
            updates=16,
            future_seed=73101 + m["block"],
        )
        for m in endpoint_models()[3:]
    ]
    with pytest.raises(ValueError):
        endpoint_models(bad)


def test_missing_response_blocks_and_invalid_answer_stays_in_denominator(tasks):
    slots = engine_slots(tasks)[:4]
    rows = [dict(s, generation_status="COMPLETE", technical_validation_errors=[]) for s in slots]
    assert validate_coverage(rows, slots)["records"] == 4
    with pytest.raises(RuntimeError, match="missing slots"):
        validate_coverage(rows[:3], slots)
    rows[0]["generation_status"] = "FAILED"
    with pytest.raises(RuntimeError, match="Technical"):
        validate_coverage(rows, slots)
    task = tasks[slots[0]["qid"]]
    valid = contract.canonical(dict(evidence=task["certificate"], answer=task["answer"]))
    scored = [
        dict(
            s,
            score=score_record(
                dict(s, raw_text=valid if i == 0 else "broken", finish_reason="balanced"), task
            ),
        )
        for i, s in enumerate(slots)
    ]
    assert summarize_scores(scored)["equal_family"]["value_correct"] == 0.25
    with pytest.raises(RuntimeError, match="Missing draw"):
        summarize_scores(scored[:3])


def test_natural_and_direct_are_independent_protocols(tasks):
    task = next(t for t in tasks.values() if t["answer"] is not None)
    answer = task["answer"]
    natural = score_record(
        dict(
            pool="ENGINE",
            protocol="natural_work",
            raw_text=f"Work. <final>{answer}</final>",
            finish_reason="final_tag",
        ),
        task,
    )
    assert natural["value_correct"] == natural["presentation"] == natural["finish"] == 1
    assert not natural["certificate_applicable"]
    assert natural["certificate"] == natural["J"] == 0
    duplicate = score_record(
        dict(
            pool="ENGINE",
            protocol="natural_work",
            raw_text=f"<final>{answer}</final><final>{answer}</final>",
            finish_reason="final_tag",
        ),
        task,
    )
    assert duplicate["value_correct"] == 0
    request = dict(prompt=task["prompt"], image_file="image", image_sha256="sha")
    assert "exact chart table" in diagnostic_input(task, request, "true_table")["prompt"]
    with pytest.raises(PermissionError):
        diagnostic_input(dict(task, pool="TEST_ID"), request, "true_table")


class FakeRuntime:
    def __init__(self, fail_after=None):
        self.stable_model_identity = {"base": "fixed"}
        self.current_adapter_identity = {"adapter": "fixed"}
        self.calls = 0
        self.fail_after = fail_after

    def generate_group(self, model_input, root, seeds, *, protocol, generation, on_completion):
        self.calls += 1
        for index, seed in enumerate(seeds):
            if index == self.fail_after:
                raise RuntimeError("simulated hardware failure")
            on_completion(
                dict(
                    generated_text='"answer":0}',
                    raw_text='{"answer":0}',
                    assistant_prefill="{",
                    tokens=[1],
                    raw_tokens=[1],
                    sampler_logprobs=[-0.1],
                    completion_token_count=1,
                    model_identity=self.stable_model_identity,
                    adapter_identity=self.current_adapter_identity,
                    prompt_tensor_hash="tensor",
                    generation_status="COMPLETE",
                    technical_validation_errors=[],
                    finish_reason="balanced",
                    truncated=False,
                    group_seed=seeds[0],
                    group_row_index=index,
                    group_size=len(seeds),
                    protocol=protocol,
                    seed=seed,
                )
            )


def test_durable_groups_resume_and_partial_never_resamples(tmp_path, tasks):
    slots = engine_slots(tasks)[:4]
    group = group_slots(slots)[0]
    qid = slots[0]["qid"]
    image = tmp_path / "image.png"
    image.write_bytes(b"test image")
    inputs = {
        qid: dict(prompt=tasks[qid]["prompt"], image_file="image.png", image_sha256=sha_file(image))
    }
    runtime = FakeRuntime()
    result = run_group(
        runtime, group, inputs, tasks, tmp_path, tmp_path / "ok", source_sha256="source", score=True
    )
    assert len(result["records"]) == len(result["scores"]) == 4
    assert runtime.calls == 1
    assert (
        run_group(runtime, group, inputs, tasks, tmp_path, tmp_path / "ok", source_sha256="source")[
            "records"
        ]
        == result["records"]
    )
    assert runtime.calls == 1
    broken = FakeRuntime(fail_after=2)
    with pytest.raises(RuntimeError, match="hardware"):
        run_group(
            broken, group, inputs, tasks, tmp_path, tmp_path / "partial", source_sha256="source"
        )
    with pytest.raises(RuntimeError, match="without resampling"):
        run_group(
            runtime, group, inputs, tasks, tmp_path, tmp_path / "partial", source_sha256="source"
        )
    assert runtime.calls == 1


def test_parent_test_scores_and_premature_release_are_blocked(tmp_path, tasks):
    slots = endpoint_slots(tasks)[:4]
    group = group_slots(slots)[0]
    with pytest.raises(PermissionError, match="remain sealed"):
        run_group(
            FakeRuntime(),
            group,
            {},
            tasks,
            tmp_path,
            tmp_path / "eval",
            source_sha256="source",
            score=True,
        )
    matrix = json.loads((PLAN / "manifests/RUN_MATRIX.json").read_text())
    with pytest.raises(PermissionError, match="12 scientific"):
        release_evaluation(matrix, [], slots, [], external_status={})
    assert object_hash(matrix) == object_hash(json.loads(json.dumps(matrix)))


def _partial_endpoint(tmp_path, tasks):
    slots = endpoint_slots(tasks)[:4]
    group = group_slots(slots)[0]
    qid = slots[0]["qid"]
    image = tmp_path / "image.png"
    image.write_bytes(b"test image")
    inputs = {
        qid: dict(prompt=tasks[qid]["prompt"], image_file="image.png", image_sha256=sha_file(image))
    }
    (tmp_path / "RUN_IDENTITY.json").write_text('{"frozen":"identity"}')
    source = sha_file(tmp_path / "RUN_IDENTITY.json")
    with pytest.raises(RuntimeError, match="hardware"):
        run_group(
            FakeRuntime(fail_after=2),
            group,
            inputs,
            tasks,
            tmp_path,
            tmp_path / "endpoint",
            source_sha256=source,
        )
    pending = tmp_path / "endpoint/groups" / (group["group_id"] + ".pending.jsonl")
    return group, inputs, source, pending


def _authorize_partial(tmp_path, group, source, pending):
    authorization = dict(
        experiment_id="SR-F3-CE-20261016",
        run_identity_sha256=source,
        authorizations=[
            dict(
                task_id="eval--" + group["slots"][0]["model_id"],
                model_id=group["slots"][0]["model_id"],
                group_id=group["group_id"],
                pending_sha256=sha_file(pending),
                status="AUTHORIZED",
            )
        ],
    )
    path = tmp_path / "recovery/attempt-1/PARTIAL_REPLAY_AUTHORIZATION.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(authorization))
    (tmp_path / "recovery/PARTIAL_REPLAY_AUTHORIZATION.json").write_text(
        json.dumps(dict(authorization_file=str(path.relative_to(tmp_path)), sha256=sha_file(path)))
    )


def test_explicit_partial_replay_preserves_original_and_matches_whole_trajectory(tmp_path, tasks):
    group, inputs, source, pending = _partial_endpoint(tmp_path, tasks)
    original = pending.read_bytes()
    runtime = FakeRuntime()
    cache = verify_panel_cache(
        tmp_path,
        tmp_path / "endpoint",
        group["slots"],
        inputs,
        tasks,
        model_identity=runtime.stable_model_identity,
        adapter_identity=runtime.current_adapter_identity,
        source_sha256=source,
        task_id="eval--" + group["slots"][0]["model_id"],
    )
    assert len(cache["partial_groups"]) == 1 and not cache["authorization_granted"]
    _authorize_partial(tmp_path, group, source, pending)
    result = run_group(
        runtime, group, inputs, tasks, tmp_path, tmp_path / "endpoint", source_sha256=source
    )
    assert len(result["records"]) == 4 and runtime.calls == 1 and not pending.exists()
    recovery = next((tmp_path / "endpoint/recovery" / group["group_id"]).iterdir())
    assert (recovery / "ORIGINAL.partial.jsonl").read_bytes() == original
    receipt = json.loads((recovery / "RECEIPT.json").read_text())
    assert receipt["status"] == "VERIFIED" and not receipt["answer_selection"]
    assert receipt["technical_replay_generated_sequences"] == 4
    assert receipt["original_prefix_rows_verified"] == 2
    run_group(runtime, group, inputs, tasks, tmp_path, tmp_path / "endpoint", source_sha256=source)
    assert runtime.calls == 1


def test_partial_replay_mismatch_and_interrupted_attempt_never_retry_automatically(tmp_path, tasks):
    group, inputs, source, pending = _partial_endpoint(tmp_path, tasks)
    original = [json.loads(line) for line in pending.read_text().splitlines()]
    original[0]["sampler_logprobs"] = [-0.2]
    pending.write_text("".join(json.dumps(row) + "\n" for row in original))
    _authorize_partial(tmp_path, group, source, pending)
    runtime = FakeRuntime()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="partial replay differs"):
            run_group(
                runtime, group, inputs, tasks, tmp_path, tmp_path / "endpoint", source_sha256=source
            )
    assert runtime.calls == 1 and pending.exists()
    assert not pending.with_suffix("").with_suffix(".json").exists()


def test_interrupted_authorized_replay_requires_new_technical_decision(tmp_path, tasks):
    group, inputs, source, pending = _partial_endpoint(tmp_path, tasks)
    _authorize_partial(tmp_path, group, source, pending)
    runtime = FakeRuntime(fail_after=1)
    with pytest.raises(RuntimeError, match="hardware"):
        run_group(
            runtime, group, inputs, tasks, tmp_path, tmp_path / "endpoint", source_sha256=source
        )
    runtime.fail_after = None
    with pytest.raises(RuntimeError, match="new technical decision"):
        run_group(
            runtime, group, inputs, tasks, tmp_path, tmp_path / "endpoint", source_sha256=source
        )
    assert runtime.calls == 1 and pending.exists()
