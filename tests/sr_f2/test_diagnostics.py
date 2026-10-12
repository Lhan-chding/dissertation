import copy
import json
from collections import Counter

import pytest

from sr_f1.contract import PACKAGE
from sr_f2.diagnostics import (
    build_diagnostic_manifest,
    generation_config,
    interval_panel,
    interval_slots,
    interval_views,
    monitor_panel,
    monitor_slots,
    prepare_input,
    protocol_panel,
    protocol_prompt,
    protocol_slots,
)
from sr_f12.protocol import BASELINE


def panel_data():
    def load(name):
        return {
            r["qid"]: r
            for r in map(json.loads, (PACKAGE / "manifests" / name).read_text().splitlines())
        }

    return load("TASKS_GOLD_AUDIT_ONLY.jsonl"), load("MODEL_INPUTS.jsonl")


@pytest.fixture(scope="module")
def panels():
    return panel_data()


def test_fixed_panels_exact_coverage_and_document_discrepancy(panels):
    tasks, inputs = panels
    manifest = build_diagnostic_manifest(tasks, inputs)
    assert len(monitor_panel(tasks)) == 128
    assert set(Counter(t["family"] for t in monitor_panel(tasks)).values()) == {32}
    assert len(protocol_panel(tasks)) == 512
    assert set(Counter(t["family"] for t in protocol_panel(tasks)).values()) == {128}
    assert len(interval_panel(tasks)) == 32
    assert len(manifest["protocol_models"]) == 13
    assert manifest["protocol_count_discrepancy"] is True
    assert manifest["protocol_prompt_count"] == 512
    assert manifest["protocol_prompt_count_in_document"] == 1024
    assert manifest["protocol_variants"] == ["v0", "v3"]
    assert len(monitor_slots(tasks, "SRF2_ADAPT_s71001", 8)) == 512
    assert len(protocol_slots(tasks, BASELINE)) == 2048
    assert len(interval_slots(tasks, BASELINE)) == 640
    assert generation_config("answer_only")["max_new_tokens"] == 768
    assert generation_config("evidence_answer")["max_new_tokens"] == 768
    assert generation_config("free")["max_new_tokens"] == 256
    assert generation_config("list")["max_new_tokens"] == 128


def test_selection_independent_of_gold_values_and_scores(panels):
    tasks, _ = panels
    changed = copy.deepcopy(tasks)
    for t in changed.values():
        t["answer"] = "bogus"
        t["world"] = {"bogus": True}
        t["required_evidence"] = []
    for select in (monitor_panel, protocol_panel, interval_panel):
        assert [t["qid"] for t in select(tasks)] == [t["qid"] for t in select(changed)]
    first = monitor_slots(tasks, "A", 8)
    second = monitor_slots(changed, "B", 8)
    assert [s["sampling_seed"] for s in first] == [s["sampling_seed"] for s in second]
    assert [s["slot_id"] for s in first] != [s["slot_id"] for s in second]


def test_unknown_or_incomplete_panels_rejected(panels):
    tasks, _ = panels
    dropped = dict(tasks)
    del dropped[monitor_panel(tasks)[0]["qid"]]
    with pytest.raises(ValueError, match="Incomplete"):
        monitor_panel(dropped)
    with pytest.raises(ValueError, match="0,8"):
        monitor_slots(tasks, "anything", 7)
    with pytest.raises(ValueError, match="historical"):
        protocol_slots(tasks, "new-model")
    with pytest.raises(ValueError, match="explicit"):
        protocol_panel(tasks, protocol_variants=("v0", "v7"))
    with pytest.raises(ValueError, match="explicit"):
        protocol_panel(tasks, protocol_variants=("v0", "v1", "v2", "v3"))


def test_free_and_list_preserve_question_replace_old_protocol(panels):
    tasks, inputs = panels
    row = inputs[protocol_panel(tasks)[0]["qid"]]
    for protocol in ("free", "list"):
        prompt = protocol_prompt(row, protocol)
        assert prompt.partition("\n\n")[0] == row["plain_text"].partition("\n\n")[0]
        assert "do not output evidence or a reasoning trace" not in prompt
        assert "final line" in prompt and '{"answer": X}' in prompt
        assert "not rounded decimals" in prompt
    with pytest.raises(ValueError, match="isolate"):
        protocol_prompt(dict(row, plain_text="invalid"), "free")


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, tokens, **kwargs):
        return "".join(map(chr, tokens))


def test_interval_views_only_frozen_suffix_and_exact_neutral_match(panels):
    tasks, inputs = panels
    task = interval_panel(tasks)[0]
    row = inputs[task["qid"]]
    views = {r["view"]: r for r in interval_views(task, row)}
    assert views["V0"]["text"] == row["text"]
    assert views["V1"]["text"].endswith(
        "February: Alpha=81, Beta=22; March: Alpha=13, Beta=29; April: Alpha=65, Beta=60."
    )
    assert "February: 59; March: -16; April: 5." in views["V2"]["text"]
    assert views["V3"]["text"].endswith("(59 + (-16) + 5)/3.")
    for v in ("V1", "V2", "V3"):
        assert views[v]["text"].startswith(row["text"] + "\n\n")
    slot = dict(qid=task["qid"], view="V4", protocol="evidence_answer")
    prepared = prepare_input(slot, inputs, list(views.values()), CharacterTokenizer())
    assert len(prepared["text"]) == len(views["V1"]["text"])
    assert prepared["token_matching"]["token_difference"] == 0
    assert prepared["image_file"] == row["image_file"]
    with pytest.raises(ValueError, match="tokenizer"):
        prepare_input(slot, inputs, list(views.values()))


def test_diagnostic_cannot_rewrite_chart_or_enter_training(panels):
    tasks, inputs = panels
    task = interval_panel(tasks)[0]
    views = interval_views(task, inputs[task["qid"]])
    slot = dict(qid=task["qid"], view="V1", protocol="evidence_answer")
    views[1]["allowed_in_training"] = True
    with pytest.raises(PermissionError, match="training"):
        prepare_input(slot, inputs, views)
    views[1]["allowed_in_training"] = False
    views[1]["image_file"] = "images/other.png"
    with pytest.raises(PermissionError, match="chart"):
        prepare_input(slot, inputs, views)
