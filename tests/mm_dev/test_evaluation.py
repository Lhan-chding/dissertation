import contextlib
import copy
import hashlib
import json

import pytest

from mm_core.execution import sha256_file
from mm_dev import evaluation as ev
from mm_dev.contract import PLAN_ID, digest, seed


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def case(tmp_path):
    plan = {
        "data": {"root_counts": {"PROBE": 1, "DEV_EVAL": 1}},
        "sampling": {"probe_eval": {"K": 4}},
    }
    questions = []
    for panel in ("PROBE", "DEV_EVAL"):
        for chart in ("grouped_bar", "line"):
            for op in ("sum", "difference", "range"):
                for v in ("low", "high"):
                    for d in ("low", "high"):
                        qid = f"{panel}-{chart}-{op}-{v}-{d}"
                        questions.append(
                            dict(
                                question_id=qid,
                                split=panel,
                                root_index=0,
                                root_family_id=panel + "-r0",
                                chart_type=chart,
                                operation=op,
                                V=v,
                                D=d,
                                true_values=["30", "20", "10"] if op == "range" else ["30", "20"],
                                image_id=qid + "-image",
                                image_path="data/images/q.png",
                                image_sha256="a" * 64,
                                prompt="Read the visible chart.",
                                prompt_sha256=hashlib.sha256(
                                    b"Read the visible chart."
                                ).hexdigest(),
                            )
                        )
    (tmp_path / "data").mkdir()
    (tmp_path / "data/questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in questions))
    identity = dict(
        state_id="S0",
        model_hash="model",
        state_manifest_hash="state",
        producer_run_id="COMMON",
        processor_hash="processor",
        adapter_path="adapter",
        adapter_file_hashes={"adapter_model.safetensors": "weights"},
        checkpoint_sha256="checkpoint",
        trainable_state_hash="trainable",
    )
    write_json(tmp_path / "manifests/F2_FREEZE.json", {"freeze": "constant"})
    freeze_sha = sha256_file(tmp_path / "manifests/F2_FREEZE.json")
    write_json(tmp_path / "orchestration/REGISTRATION.json", {"measurement_shards": 1})
    return dict(
        plan=plan,
        root=tmp_path,
        panel="PROBE",
        state_id="S0",
        shard=0,
        shards=1,
        identity=identity,
        freeze_sha256=freeze_sha,
    )


class FakeRuntime:
    def __init__(self):
        self.generated = []
        self.self_scored = []
        self.gold_scored = []
        self.fail_self_once = False
        self.mutate_return = False

    def prepare(self, question, root):
        assert "true_values" not in question
        assert "V" not in question
        assert "root_family_id" not in question
        return question

    def generate(self, question, root, sample_seed, *, score_fields, on_completion):
        assert score_fields is False
        self.prepare(question, root)
        self.generated.append((question["question_id"], sample_seed))
        record = dict(
            raw_text='{"readings": [29, 20], "answer": 49}',
            tokens=[10, 11],
            raw_tokens=[10, 11],
            seed=sample_seed,
            prompt_token_count=500,
            completion_token_count=2,
            truncated=False,
            finish_reason="eos",
            generation_seconds=0.01,
            image_routing={"generation_vision_forward_calls": 1},
        )
        on_completion(record)
        if self.mutate_return:
            record["tokens"][0] = 99
        return record

    def encode_completion(self, text):
        return [20, 21, 22]

    def field_scores(self, prepared, tokens, *, provenance):
        if provenance == ev.SELF_PROVENANCE:
            if self.fail_self_once:
                self.fail_self_once = False
                raise RuntimeError("Injected teacher-forcing failure")
            self.self_scored.append((prepared["question_id"], list(tokens)))
        else:
            assert provenance == ev.GOLD_PROVENANCE
            self.gold_scored.append((prepared["question_id"], list(tokens)))
        return dict(
            status="MEASURED",
            provenance=provenance,
            distribution="raw_model_before_temperature_or_top_p",
            boundary_rule="largest_value_character_overlap_then_readings_tie",
            vision_forward_calls=1,
            token_logprobs=[-0.2, -0.3] + [-0.1] * (len(tokens) - 2),
            token_entropy=[0.1, 0.2] + [0.1] * (len(tokens) - 2),
            field_nll={
                "readings": {
                    "token_count": 1,
                    "token_indices": [0],
                    "mean_nll": 0.2,
                    "mean_token_entropy": 0.1,
                },
                "answer": {
                    "token_count": 1,
                    "token_indices": [1],
                    "mean_nll": 0.3,
                    "mean_token_entropy": 0.2,
                },
            },
        )


def run(case, runtime=None, **extra):
    runtime = runtime or FakeRuntime()
    return ev.run_shard(**case, runtime_factory=lambda: (runtime, case["identity"]), **extra)


def directory(case):
    return (
        case["root"]
        / "raw"
        / case["panel"]
        / case["state_id"]
        / f"shard_{case['shard']}of{case['shards']}"
    )


def load_directory(case):
    questions = ev.panel_questions(case["plan"], case["root"], case["panel"])
    _, expected, golds = ev._expected(
        questions,
        case["panel"],
        case["identity"],
        case["freeze_sha256"],
        case["shard"],
        case["shards"],
    )
    return ev._read_directory(case["root"], directory(case), expected, golds, case["panel"])


def repair(case):
    failure = sorted(directory(case).glob("FAILURE_*.json"))[-1]
    receipt = dict(
        plan_id=PLAN_ID,
        task_id=ev.task_name(case["panel"], case["state_id"], case["shard"], case["shards"]),
        approved=True,
        failure_receipt_sha256=sha256_file(failure),
        freeze_sha256=case["freeze_sha256"],
        source_before={"source": "a"},
        source_after={"source": "a"},
        repair_scope="Transient technical injection fixed",
        rerun_comparable_block_required=False,
    )
    path = case["root"] / "manifests/TECHNICAL_REPAIRS" / (receipt["task_id"] + ".json")
    write_json(path, receipt)
    return path


def test_probe_unique_gold_self_and_idempotence(case):
    runtime = FakeRuntime()
    receipt = run(case, runtime)
    assert receipt["status"] == "COMPLETE"
    assert receipt["expected_completions"] == receipt["actual_completions"] == 96
    assert receipt["self_score_count"] == len(runtime.self_scored) == 96
    assert receipt["gold_score_count"] == len(runtime.gold_scored) == 24
    assert len({q for q, _ in runtime.gold_scored}) == 24
    before = {p.name: p.read_bytes() for p in directory(case).iterdir()}
    assert (
        ev.run_shard(**case, runtime_factory=lambda: pytest.fail("Unnecessary model load"))
        == receipt
    )
    assert {p.name: p.read_bytes() for p in directory(case).iterdir()} == before


def test_eval_no_teacher_forcing(case):
    case["panel"] = "DEV_EVAL"
    runtime = FakeRuntime()
    receipt = run(case, runtime)
    assert len(runtime.generated) == 96
    assert not runtime.self_scored and not runtime.gold_scored
    assert receipt["self_score_count"] == receipt["gold_score_count"] == 0
    assert not list(directory(case).glob("*scores*.jsonl"))


def test_seed_pairing_is_independent_of_model_and_partition(case):
    q = ev.panel_questions(case["plan"], case["root"], "PROBE")[0]
    a = ev.request_identity(q, "PROBE", case["identity"], 0, 2, case["freeze_sha256"])
    other = {**case["identity"], "state_id": "S_A_0", "checkpoint_sha256": "other"}
    b = ev.request_identity(q, "PROBE", other, 0, 2, case["freeze_sha256"])
    assert a["seed"] == b["seed"] == seed("evaluation", "PROBE", q["question_id"], 2)
    assert a["request_id"] != b["request_id"]
    for shards in (1, 2, 5):
        expected = []
        questions = ev.panel_questions(case["plan"], case["root"], "PROBE")
        for shard in range(shards):
            _, rows, _ = ev._expected(
                questions, "PROBE", case["identity"], case["freeze_sha256"], shard, shards
            )
            expected.extend(rows)
        assert len(set(expected)) == 96
        assert a["request_id"] in expected


def test_preemption_resume_preserves_bytes_and_never_regenerates(case):
    first = FakeRuntime()
    receipt = run(case, first, stop_requested=lambda: len(first.generated) >= 3)
    assert receipt["status"] == "CHECKPOINTED"
    original = directory(case) / "outputs_000000.jsonl"
    before = original.read_bytes()
    second = FakeRuntime()
    receipt = run(case, second)
    assert receipt["status"] == "COMPLETE"
    assert len(first.generated) + len(second.generated) == 96
    assert not set(first.generated) & set(second.generated)
    assert len(first.self_scored) + len(second.self_scored) == 96
    assert len(first.gold_scored) + len(second.gold_scored) == 24
    assert original.read_bytes() == before
    assert (directory(case) / "outputs_000001.jsonl").exists()


def test_raw_survives_scoring_failure_and_requires_exact_repair(case):
    first = FakeRuntime()
    first.fail_self_once = True
    with pytest.raises(RuntimeError, match="Injected"):
        run(case, first)
    original = directory(case) / "outputs_000000.jsonl"
    before = original.read_bytes()
    assert len(load_directory(case)["outputs"]) == 1
    assert not load_directory(case)["self_scores"]
    with pytest.raises(PermissionError, match="repair receipt"):
        run(case)
    path = repair(case)
    authorized = json.loads(path.read_text())
    write_json(path, {**authorized, "failure_receipt_sha256": "wrong"})
    with pytest.raises(ValueError, match="failure_receipt"):
        run(case)
    write_json(path, authorized)
    second = FakeRuntime()
    receipt = run(case, second)
    assert receipt["actual_completions"] == 96 and len(second.generated) == 95
    assert len(second.self_scored) == 96 and len(second.gold_scored) == 24
    assert original.read_bytes() == before
    unknown = [
        e
        for e in load_directory(case)["technical_events"]
        if e["event"] == "PHYSICAL_CALL_OUTCOME_UNKNOWN"
    ]
    assert [e["call"] for e in unknown] == ["SELF_FORWARD_STARTED"]


def test_incomplete_tail_is_retained_and_cost_unknown(case):
    first = FakeRuntime()
    run(case, first, stop_requested=lambda: len(first.generated) >= 1)
    path = directory(case) / "outputs_000000.jsonl"
    with path.open("ab") as handle:
        handle.write(b'{"request_id": "interrupted')
    before = path.read_bytes()
    second = FakeRuntime()
    receipt = run(case, second)
    assert receipt["incomplete_tail_count"] == 1
    assert len(second.generated) == 95
    assert path.read_bytes() == before
    events = load_directory(case)["technical_events"]
    assert any(
        e["event"] == "INCOMPLETE_WRITE_RETAINED" and e["physical_call_cost"] == "UNKNOWN"
        for e in events
    )


@pytest.mark.parametrize("body", [b'{"a":1}\n{broken}\n', b'not json\n{"a":1}\n', b"[]\n"])
def test_complete_line_corruption_is_not_silently_skipped(tmp_path, body):
    path = tmp_path / "raw.jsonl"
    path.write_bytes(body)
    with pytest.raises(ValueError, match="Corrupt"):
        ev.read_segment(path, tmp_path)


def test_valid_final_record_without_newline_is_preserved(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_bytes(b'{"a":1}')
    rows, events = ev.read_segment(path, tmp_path)
    assert rows == [{"a": 1}]
    assert events[0]["event"] == "VALID_JSON_WITHOUT_TERMINATOR_RETAINED"
    assert path.read_bytes() == b'{"a":1}'


def test_duplicate_raw_rejected(case):
    first = FakeRuntime()
    run(case, first, stop_requested=lambda: len(first.generated) >= 1)
    path = directory(case) / "outputs_000000.jsonl"
    (directory(case) / "outputs_000999.jsonl").write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="Duplicate logical"):
        run(case)


@pytest.mark.parametrize(
    "key,value",
    [("model_hash", "other"), ("seed", 1), ("tokens", [44]), ("completion_token_count", 99)],
)
def test_raw_identity_and_token_integrity_rejected(case, key, value):
    first = FakeRuntime()
    run(case, first, stop_requested=lambda: len(first.generated) >= 1)
    path = directory(case) / "outputs_000000.jsonl"
    row = json.loads(path.read_text())
    row[key] = value
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        run(case)


def test_self_score_cannot_join_another_completion(case):
    first = FakeRuntime()
    run(case, first, stop_requested=lambda: len(first.generated) >= 1)
    path = directory(case) / "self_scores_000000.jsonl"
    row = json.loads(path.read_text())
    row["completion_sha256"] = "other"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="exact saved raw"):
        run(case)


def test_scores_never_loaded_as_raw_outputs(case):
    first = FakeRuntime()
    run(case, first, stop_requested=lambda: len(first.generated) >= 1)
    data = load_directory(case)
    assert len(data["outputs"]) == len(data["self_scores"]) == 1
    assert all("raw_text" in row for row in data["outputs"].values())
    assert all("raw_text" not in row for row in data["self_scores"].values())


def test_changed_runtime_state_stops_before_generation(case):
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match="Loaded model"):
        ev.run_shard(
            **case, runtime_factory=lambda: (runtime, {**case["identity"], "model_hash": "x"})
        )
    assert not runtime.generated
    assert list(directory(case).glob("FAILURE_*.json"))


def test_mutating_callback_result_cannot_change_saved_response(case):
    runtime = FakeRuntime()
    runtime.mutate_return = True
    with pytest.raises(ValueError, match="after durable callback"):
        run(case, runtime)
    saved = next(iter(load_directory(case)["outputs"].values()))
    assert saved["tokens"] == saved["raw_tokens"] == [10, 11]


def test_completed_receipt_detects_artifact_tamper(case):
    run(case)
    path = directory(case) / "events_000000.jsonl"
    with path.open("a") as handle:
        handle.write('{"event":"injected"}\n')
    with pytest.raises(ValueError, match="receipt differs"):
        run(case)


def test_load_panel_missing_is_incomplete_not_zero(case, monkeypatch):
    from mm_dev import common, runtime

    monkeypatch.setattr(common, "load_plan", lambda p: case["plan"])
    monkeypatch.setattr(common, "verify_execution", lambda *a, **k: {})
    monkeypatch.setattr(runtime, "get_state_identity", lambda *a: case["identity"])
    payload = ev.load_panel("unused", case["root"], "PROBE", "S0")
    assert payload["complete"] is False and not payload["outputs"]
    assert payload["self_scores"] == payload["gold_scores"] == {}
    run(case)
    payload = ev.load_panel("unused", case["root"], "PROBE", "S0")
    assert payload["complete"] is True
    assert len(payload["outputs"]) == len(payload["self_scores"]) == 96
    assert len(payload["gold_scores"]) == 24


def test_changed_sharding_or_state_cannot_resume(case):
    first = FakeRuntime()
    run(case, first, stop_requested=lambda: len(first.generated) >= 1)
    with pytest.raises(ValueError, match="partition changed"):
        run({**case, "shards": 2})
    changed = copy.deepcopy(case)
    changed["identity"]["model_hash"] = "other"
    with pytest.raises(ValueError, match="session/model/freeze changed"):
        run(changed)


@pytest.mark.parametrize("operation", ["missing", "duplicate", "wrong-root", "wrong-factor"])
def test_panel_cube_checked_before_runtime(case, operation):
    path = case["root"] / "data/questions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if operation == "missing":
        rows.pop(0)
    elif operation == "duplicate":
        rows.append(rows[0])
    elif operation == "wrong-root":
        rows[0]["root_index"] = 99
    else:
        rows[0]["V"] = "adaptive"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        ev.run_shard(**case, runtime_factory=lambda: pytest.fail("Invalid panel reached model"))


def test_task_identity_validation():
    assert ev.task_name("PROBE", "S0", 0, 1) == "probe_S0"
    assert ev.task_name("DEV_EVAL", "cont_S_AP_1_aC_f1", 4, 5).endswith("_shard4of5")
    for panel, state, shard, shards in (
        ("PROBE", "cont_S0_a0_f0", 0, 1),
        ("DEV_EVAL", "../../escape", 0, 1),
        ("PROBE", "S0", 0, 6),
        ("PROBE", "S0", 5, 5),
    ):
        with pytest.raises(ValueError):
            ev.task_name(panel, state, shard, shards)


def test_request_binds_model_run_and_question_content(case):
    q = ev.panel_questions(case["plan"], case["root"], "PROBE")[0]
    row = ev.request_identity(q, "PROBE", case["identity"], 0, 0, case["freeze_sha256"])
    assert row["run_hash"] == digest(case["identity"])
    assert row["question_sha256"] == digest(q)
    assert row["model_hash"] == case["identity"]["model_hash"]


def test_cli_requires_registered_task_before_model(case, monkeypatch):
    monkeypatch.delenv("MM_DEV_TASK_ID", raising=False)
    with pytest.raises(PermissionError, match="registered worker task"):
        ev.main("PROBE", ["--plan", "unused", "--run-root", str(case["root"]), "--state-id", "S0"])


def test_cli_uses_registered_allocation_and_publishes_completion(case, monkeypatch):
    from mm_dev import common, orchestration, runtime

    write_json(
        case["root"] / "orchestration/REGISTRATION.json",
        {"measurement_shards": 1, "tasks": {"probe_S0": {}}},
    )
    monkeypatch.setenv("MM_DEV_TASK_ID", "probe_S0")
    monkeypatch.setattr(common, "load_plan", lambda p: case["plan"])
    monkeypatch.setattr(common, "verify_execution", lambda *a, **k: {})
    monkeypatch.setattr(runtime, "get_state_identity", lambda *a: case["identity"])
    events = []
    monkeypatch.setattr(common, "require_allocation", lambda *a: events.append("allocation"))
    monkeypatch.setattr(orchestration, "worker_lease", lambda *a: contextlib.nullcontext())
    monkeypatch.setattr(
        orchestration, "complete_task", lambda *a, **k: events.append(("complete", a, k))
    )
    fake = FakeRuntime()

    def factory(*args, **kwargs):
        assert events == ["allocation"]
        events.append("model")
        assert callable(kwargs["account"])
        return fake, case["identity"]

    monkeypatch.setattr(runtime, "create_evaluation_runtime", factory)
    assert (
        ev.main("PROBE", ["--plan", "unused", "--run-root", str(case["root"]), "--state-id", "S0"])
        == 0
    )
    assert events[:2] == ["allocation", "model"]
    assert events[2][0] == "complete"
    assert any(p.endswith("COMPLETE.json") for p in events[2][1][2])


def test_cli_records_constructor_failure(case, monkeypatch):
    from mm_dev import common, orchestration, runtime

    write_json(
        case["root"] / "orchestration/REGISTRATION.json",
        {"measurement_shards": 1, "tasks": {"probe_S0": {}}},
    )
    monkeypatch.setenv("MM_DEV_TASK_ID", "probe_S0")
    monkeypatch.setattr(common, "load_plan", lambda p: case["plan"])
    monkeypatch.setattr(common, "verify_execution", lambda *a, **k: {})
    monkeypatch.setattr(common, "require_allocation", lambda *a: {})
    monkeypatch.setattr(orchestration, "worker_lease", lambda *a: contextlib.nullcontext())
    monkeypatch.setattr(runtime, "get_state_identity", lambda *a: case["identity"])
    failures = []
    monkeypatch.setattr(orchestration, "fail_task", lambda *a: failures.append(a))

    def fail(*args, **kwargs):
        raise RuntimeError("Constructor failure")

    monkeypatch.setattr(runtime, "create_evaluation_runtime", fail)
    with pytest.raises(RuntimeError, match="Constructor"):
        ev.main("PROBE", ["--plan", "unused", "--run-root", str(case["root"]), "--state-id", "S0"])
    assert len(failures) == 1 and failures[0][1] == "probe_S0"
    assert len(list(directory(case).glob("FAILURE_*.json"))) == 1
    assert not list(directory(case).glob("outputs_*.jsonl"))


def test_unmeasured_boundary_is_retained_without_inventing_values(case):
    class BoundaryRuntime(FakeRuntime):
        def field_scores(self, prepared, tokens, *, provenance):
            return dict(
                status="TOKEN_CHARACTER_BOUNDARY_UNVERIFIED",
                provenance=provenance,
                field_nll=None,
                tokens=len(tokens),
            )

    receipt = run(case, BoundaryRuntime())
    assert receipt["status"] == "COMPLETE"
    rows = load_directory(case)
    assert all(row["field_scores"]["field_nll"] is None for row in rows["self_scores"].values())
    assert all(row["field_scores"]["field_nll"] is None for row in rows["gold_scores"].values())


def test_symlink_raw_parent_rejected(case, tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    (case["root"] / "raw").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="Unsafe measurement parent"):
        run(case)


@pytest.mark.parametrize("mutation", ["overlap", "count", "distribution", "missing"])
def test_field_packet_denominators_and_raw_distribution_checked(mutation):
    packet = FakeRuntime().field_scores(
        {"question_id": "q"}, [10, 11], provenance=ev.SELF_PROVENANCE
    )
    if mutation == "overlap":
        packet["field_nll"]["answer"]["token_indices"] = [0]
    elif mutation == "count":
        packet["field_nll"]["readings"]["token_count"] = 2
    elif mutation == "distribution":
        packet["distribution"] = "top_p_logits"
    else:
        packet["field_nll"].pop("answer")
    with pytest.raises(ValueError):
        ev._packet(packet, ev.SELF_PROVENANCE, 2)
