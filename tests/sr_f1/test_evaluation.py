import json
from collections import Counter
from pathlib import Path
from typing import ClassVar

import pytest

from sr_f1.evaluation import (
    BASELINE,
    evaluate_slots,
    iter_evaluation_slots,
    iter_model_slots,
    model_ids,
    neutral_padding,
    score_response,
)

PACKAGE = Path(__file__).resolve().parents[2] / "docs/sr_f1/package"


def panels():
    tasks = {
        r["qid"]: r
        for r in map(
            json.loads, (PACKAGE / "manifests/TASKS_GOLD_AUDIT_ONLY.jsonl").read_text().splitlines()
        )
    }
    fit = json.loads((PACKAGE / "manifests/TRAIN_FIT_QIDS.json").read_text())
    return tasks, fit


def test_all_registered_evaluation_slots_have_exact_budget_and_independent_protocols():
    tasks, fit = panels()
    counts = Counter()
    models = set()
    plain, evidence = {}, {}
    for slot in iter_evaluation_slots(tasks, fit):
        counts[slot["pool"]] += 1
        models.add(slot["model_id"])
        if slot["model_id"] == BASELINE and slot["pool"] == "TEST_COMPOSITION":
            key = (slot["qid"], slot["draw"])
            (plain if slot["protocol"] == "answer_only" else evidence)[key] = slot
    assert counts == {
        "MONITOR": 65536,
        "MONITOR_DIAGNOSTIC_ONLY": 40960,
        "TRAIN_FIT": 5888,
        "TEST_ID": 262144,
        "TEST_COMPOSITION": 16384,
        "TEST_LANGUAGE": 16384,
        "TEST_RENDER": 16384,
    }
    assert sum(counts.values()) == 423680
    assert models == set(model_ids()) and len(models) == 16
    assert plain.keys() == evidence.keys()
    assert all(plain[k]["sampling_seed"] != evidence[k]["sampling_seed"] for k in plain)
    assert all(
        plain[k]["generation"]["max_new_tokens"] == 128
        and evidence[k]["generation"]["max_new_tokens"] == 768
        for k in plain
    )


def test_monitor_blocks_are_exactly_sixteen_by_eight_and_intermediate_test_forbidden():
    tasks, fit = panels()
    slots = list(iter_model_slots(tasks, fit, BASELINE, stage="baseline"))
    blocks = Counter(s["monitor_block"] for s in slots if s["pool"] == "MONITOR")
    assert blocks == dict.fromkeys(range(32), 128)
    with pytest.raises(ValueError, match="Only TRAIN_FIT"):
        list(iter_model_slots(tasks, fit, "SRF1_A_s71001", step=32))


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))

    def decode(self, ids, skip_special_tokens=False):
        return "".join(map(chr, ids))


def test_neutral_padding_uses_actual_full_prompt_token_count():
    text, receipt = neutral_padding(
        "question", "known facts 123", "Neutral wording. ", CharTokenizer()
    )
    assert len(text) - len("question") == len("\n\nknown facts 123")
    assert abs(receipt["token_difference"]) <= 1
    assert "123" not in text


def test_plain_protocol_never_creates_evidence_metrics_and_keeps_null_correct():
    tasks, _ = panels()
    task = next(t for t in tasks.values() if t["answer"] is None)
    result = score_response('{"answer":null}', task, "answer_only")
    assert result["A"] == 1
    assert not {"E", "P", "J"} & result.keys()
    assert score_response('{"answer":null,"evidence":[]}', task, "answer_only")["A"] == 0
    assert score_response('{"answer":0,"answer":null}', task, "answer_only")["A"] == 0


def test_real_technical_invalid_completion_is_durable_before_exception(tmp_path):
    tasks, fit = panels()
    slot = next(iter_model_slots(tasks, fit, BASELINE, stage="baseline"))
    inputs = {slot["qid"]: {"text": "q", "plain_text": "plain", "image_file": None}}

    class BadRuntime:
        identity: ClassVar = {"base": "verified"}
        adapter_identity: ClassVar = {"sha256": "actual-adapter"}

        def generate(self, **kwargs):
            raw = dict(
                raw_text="retained",
                tokens=[1],
                old_logprobs=[None],
                truncated=False,
                generation_status="TECHNICAL_INVALID",
                technical_validation_errors=["NONFINITE"],
            )
            kwargs["on_completion"](raw)
            raise RuntimeError("Invalid generation preserved")

    path = tmp_path / "raw.jsonl"
    with pytest.raises(RuntimeError, match="preserved"):
        evaluate_slots(BadRuntime(), [slot], inputs, {}, tmp_path, path)
    raw = json.loads(path.read_text())
    assert raw["raw_text"] == "retained" and raw["old_logprobs"] == [None]
    with pytest.raises(ValueError, match="Technical missing"):
        evaluate_slots(BadRuntime(), [slot], inputs, {}, tmp_path, path)


def test_checkpoint_is_immutable_prefix_with_no_resampling(tmp_path):
    tasks, fit = panels()
    slot = next(iter_model_slots(tasks, fit, BASELINE, stage="baseline"))
    inputs = {slot["qid"]: {"text": "q", "plain_text": "plain", "image_file": None}}

    boundary = {"requested": False}

    class Runtime:
        identity: ClassVar = {"base": "verified"}
        adapter_identity = "actual"
        calls = 0

        def generate(self, **kwargs):
            self.calls += 1
            raw = dict(raw_text="{}", tokens=[1], old_logprobs=[-1.0], truncated=False)
            kwargs["on_completion"](raw)
            boundary["requested"] = True
            return raw

    runtime = Runtime()
    path = tmp_path / "raw.jsonl"
    result = evaluate_slots(runtime, [slot], inputs, {}, tmp_path, path, boundary)
    assert result["status"] == "CHECKPOINTED" and result["metadata"]["full_state"]
    assert (tmp_path / result["artifacts"][0]).read_bytes() == path.read_bytes()
    assert evaluate_slots(runtime, [slot], inputs, {}, tmp_path, path)["status"] == "COMPLETE"
    assert runtime.calls == 1


def test_actual_entrypoint_stages_cover_every_slot_once(monkeypatch):
    import sr_f1.evaluation as evaluation

    seen = set()

    def capture(runtime, slots, inputs, diagnostics, root, path, boundary=None):
        for slot in slots:
            assert slot["slot_id"] not in seen
            seen.add(slot["slot_id"])
        return {"status": "COMPLETE"}

    monkeypatch.setattr(evaluation, "evaluate_slots", capture)
    evaluation.evaluate_model(object(), PACKAGE, BASELINE, stage="baseline")
    for model in model_ids():
        if model != BASELINE:
            for step in (32, 64, 96):
                evaluation.evaluate_model(object(), PACKAGE, model, stage="train_fit", step=step)
        evaluation.evaluate_model(object(), PACKAGE, model, stage="final")
    assert len(seen) == 423680
    tasks, fit = panels()
    assert seen == {s["slot_id"] for s in iter_evaluation_slots(tasks, fit)}


def test_eval_resume_allows_same_gpu_class_but_not_changed_model_or_software():
    from sr_f1.evaluation import evaluation_model_identity

    first = dict(
        model_hash="base+actual-adapter",
        processor_hash="fixed",
        hardware=dict(
            cuda_uuid="gpu1",
            hostname="host1",
            pci_bus_id="01",
            cuda_name="RTX PRO 6000",
            driver_version="580",
            compute_capability=[12, 0],
        ),
    )
    second = dict(
        first, hardware=dict(first["hardware"], cuda_uuid="gpu2", hostname="host2", pci_bus_id="02")
    )
    assert evaluation_model_identity(first) == evaluation_model_identity(second)
    assert evaluation_model_identity(first) != evaluation_model_identity(
        dict(second, model_hash="changed")
    )
    assert evaluation_model_identity(first) != evaluation_model_identity(
        dict(second, hardware=dict(second["hardware"], driver_version="590"))
    )
