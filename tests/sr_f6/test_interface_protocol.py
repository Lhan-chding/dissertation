"""Actual prompt/scorer boundary tests; no model calls or new science outcomes."""

import copy
import hashlib
import json

import pytest

from sr_f6.data import public_prompt
from sr_f6.interface_protocol import VERSION, format_contract, response_shape
from sr_f6.interface_protocol import public_prompt as public_only_prompt
from sr_f6.interface_repair import prepare_screen, screen_inputs, screen_plan
from sr_f6.measurement import parse, score_response
from srf6_ref.design import build_bank, visible_prompt
from srf6_ref.measurement import full_truth_response


@pytest.fixture(scope="module")
def tasks():
    return [
        task
        for root in build_bank("DEV_CAL", 8, 2026102711)
        for rows in root["tasks"].values()
        for task in rows
    ]


@pytest.fixture(scope="module")
def inputs(tasks):
    result = []
    for task in tasks:
        prompt = public_prompt(task)
        result.append(
            {
                **{
                    key: task[key]
                    for key in ("qid", "root_id", "family", "factor", "variant", "split")
                },
                "protocol": "structured_v4",
                "source_image_status": "GENERATED",
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "model_input": {
                    "prompt": prompt,
                    "image_file": "DATA/images/example.png",
                    "image_sha256": "a" * 64,
                },
            }
        )
    return result


def test_explicit_revision_reaches_production_input_builder_and_preserves_question(tasks):
    for task in tasks:
        old = public_prompt(task)
        assert old == visible_prompt(task)
        new = public_prompt(task, prompt_version=VERSION)
        assert new.startswith(old.split("Return exactly one JSON object")[0])
        assert "array of STRINGS" in new and "computed NUMERIC VALUE" in new
        assert "BEFORE selected" in new and "Put answer last" in new
        assert "..." not in new
        assert public_prompt(task, natural=True, prompt_version=VERSION) == public_prompt(
            task,
            natural=True,
        )


def test_public_only_boundary_rejects_hidden_or_unknown_fields(tasks):
    task = tasks[0]
    for field in ("gold_answer", "world", "factor", "certificate", "unknown_extension"):
        query = {**task["query"], field: "secret"}
        with pytest.raises(ValueError):
            public_only_prompt(task["world"]["categories"], query)
    query = {**task["query"], "k": True}
    with pytest.raises(ValueError):
        public_only_prompt(task["world"]["categories"], query)
    with pytest.raises(ValueError):
        public_prompt(task, prompt_version="unregistered")


def test_private_world_values_and_gold_do_not_change_prompt(tasks):
    task = tasks[0]
    changed = copy.deepcopy(task)
    changed.update(gold_answer="SECRET", factor="SECRET", gold_selected=["SECRET"])
    changed["world"]["series"] = {"Alpha": [999] * 6, "Beta": [-999] * 6}
    assert public_prompt(task, prompt_version=VERSION) == public_prompt(
        changed, prompt_version=VERSION
    )


def test_schema_is_fresh_shape_only_and_exact_parser_remains_authoritative(tasks):
    shape = response_shape()
    assert set(shape["required"]) == {"evidence", "selected", "bindings", "aggregate", "answer"}
    assert shape["properties"]["selected"]["items"] == {"type": "string"}
    shape["properties"]["selected"]["items"]["enum"] = ["secret"]
    assert "enum" not in response_shape()["properties"]["selected"]["items"]
    task = tasks[0]
    obj = json.loads(full_truth_response(task["world"], task["query"]))
    assert parse(json.dumps(obj))["valid"]
    for value in ("Beta", "60-48", True, None, "1/0", "1/1000000001"):
        bad = copy.deepcopy(obj)
        bad["bindings"][0]["operand"] = value
        assert not parse(json.dumps(bad))["valid"]
    bad = copy.deepcopy(obj)
    bad["selected"] = [{"category": name} for name in bad["selected"]]
    assert not parse(json.dumps(bad))["valid"]
    bad = copy.deepcopy(obj)
    bad["bindings"][0]["operand"] = -18
    assert parse(json.dumps(bad))["valid"]
    assert score_response(json.dumps(bad), task["world"], task["query"])["B"] == 0
    # Well-shaped false/duplicate selections remain semantic failure, not repaired.
    bad["selected"] = ["Example_A", "Example_A"]
    assert parse(json.dumps(bad))["valid"]
    assert score_response(json.dumps(bad), task["world"], task["query"])["S"] == 0
    bad["selected"], bad["bindings"], bad["answer"] = [], [], None
    assert parse(json.dumps(bad))["valid"]
    assert score_response(json.dumps(bad), task["world"], task["query"])["A"] == 0


def test_filter_and_difference_evidence_are_public_requirements(tasks):
    for task in tasks:
        prompt = public_prompt(task, prompt_version=VERSION)
        query = task["query"]
        if query["family"] == "TOPK":
            assert f"report {query['selection_series']} for EVERY category" in prompt
        elif query["positive_only"]:
            assert "BEFORE filtering" in prompt
        if query["value"] == "diff":
            assert "original Alpha and Beta values separately" in prompt


def test_smoke_has_fixed_geometry_and_image_and_task_identity(tasks, inputs):
    before = copy.deepcopy(inputs)
    selected = screen_inputs(inputs, tasks)
    plan = screen_plan(selected)
    assert len(selected) == 96 and len(plan["attempts"]) == 576
    assert plan["required_counts"] == {"V": 87, "S_known": 87, "B_known": 87}
    assert plan["gate_status"] == "NOT_SIGNED" and not plan["generation_allowed"]
    old = {row["qid"]: row for row in inputs}
    for row in selected:
        assert row["prompt_version"] == VERSION
        for key in ("image_file", "image_sha256"):
            assert row["model_input"][key] == old[row["qid"]]["model_input"][key]
        assert row["model_input"]["prompt"] != old[row["qid"]]["model_input"]["prompt"]
    assert inputs == before
    by_identity = {}
    for row in plan["attempts"]:
        key = (row["parent"], row["root_id"], row["variant"], row["draw"])
        assert by_identity.setdefault(key, row["seed"]) == row["seed"]
    assert len(by_identity) == 288  # P0/P1 pairing, not 576 independent seeds.


def test_repaired_prompt_reaches_real_runtime_prepare_boundary(
    monkeypatch, tmp_path, tasks, inputs
):
    from sr_f6 import runtime

    selected = screen_inputs(inputs, tasks)[0]
    captured = {}

    def capture(self, model_input, run_root, *, protocol):
        captured.update(model_input=copy.deepcopy(model_input), root=run_root, protocol=protocol)
        return {"routing": {}}

    # Intercept only the processor backend; exercise Runtime.prepare_input itself.
    # This is a CPU call-path check, not a real processor or model execution.
    monkeypatch.setattr(runtime.FP32Runtime, "prepare_input", capture)
    obj = runtime.Runtime.__new__(runtime.Runtime)
    obj._replay_session = lambda: None
    prepared = obj.prepare_input(selected["model_input"], tmp_path, protocol=selected["protocol"])
    assert captured["model_input"] == selected["model_input"]
    assert "computed NUMERIC VALUE" in captured["model_input"]["prompt"]
    assert set(captured["model_input"]) == {"prompt", "image_file", "image_sha256"}
    assert captured["protocol"] == "structured"
    assert prepared["routing"]["sr_f6_protocol_version"] == runtime.PROTOCOL_VERSION


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "private", "prompt", "lineage", "path"]
)
def test_screen_rejects_mixed_or_unbound_original_inputs(tasks, inputs, mutation):
    altered = copy.deepcopy(inputs)
    if mutation == "missing":
        altered.pop()
    elif mutation == "duplicate":
        altered[-1] = altered[0]
    elif mutation == "private":
        altered[0]["model_input"]["world"] = tasks[0]["world"]
    elif mutation == "prompt":
        altered[0]["model_input"]["prompt"] += " changed"
    elif mutation == "lineage":
        altered[0]["split"] = "TEST_ID"
    else:
        altered[0]["model_input"]["image_file"] = "DATA/images/../../secret"
    with pytest.raises(ValueError):
        screen_inputs(altered, tasks)


def test_plan_writer_never_overwrites_old_run(tmp_path, tasks, inputs):
    source = tmp_path / "old"
    data = source / "DATA/DEV_CAL"
    data.mkdir(parents=True)
    for name, rows in (("MODEL_INPUTS", inputs), ("TASKS_PRIVATE", tasks)):
        (data / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(PermissionError):
        prepare_screen(source, source / "repair")
    output = tmp_path / "new"
    result = prepare_screen(source, output)
    assert result["model_calls"] == result["optimizer_updates"] == 0
    assert (output / "FORMAT_CONTRACT.txt").read_text().strip() == format_contract()
    with pytest.raises(FileExistsError):
        prepare_screen(source, output)
