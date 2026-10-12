import json
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from mm_core.training import capture_rng, state_hash
from sr_f1.json_protocol import decoded_protocol
from sr_f2.diagnostics import generation_config
from sr_f2.evaluation import evaluate_monitor, run_group, summarize_scores
from sr_f2.runtime import SRF2Runtime
from sr_f12.evaluation import group_slots
from sr_f12.protocol import object_hash


def fixtures(root, protocol="evidence_answer", count=4, pool="MONITOR_LITE"):
    image = root / "images" / "image.png"
    image.parent.mkdir(exist_ok=True)
    image.write_bytes(b"immutable-test-image")
    slots = [
        dict(
            slot_id=f"{protocol}-{i}",
            model_id="model",
            step=0,
            pool=pool,
            protocol=protocol,
            view=None,
            qid="q",
            root_id="r",
            family="CROSS",
            chart="line",
            variant="v0",
            draw=i,
            samples_per_prompt=count,
            sampling_seed=100 + i,
            group_seed=100,
            generation=generation_config(protocol),
        )
        for i in range(count)
    ]
    inputs = {
        "q": dict(
            text="Q evidence",
            plain_text="Q\n\nold answer instructions",
            image_file="images/image.png",
        )
    }
    tasks = {"q": dict(world={}, query={})}
    return group_slots(slots)[0], inputs, tasks


class Runtime:
    def __init__(self):
        self.processor = SimpleNamespace(tokenizer=object())
        self.model = torch.nn.Linear(1, 1)
        self.calls = 0
        self.error_after = None
        self.invalid = False
        self.version = "base"

    def stable_model_identity(self):
        return {"model": self.version}

    def current_adapter_identity(self):
        return {"adapter": "fixed"}

    def generate_group(
        self, row, root, seeds, *, text, image_path, protocol, generation, on_completion
    ):
        self.calls += 1
        random.random()
        np.random.rand()
        torch.rand(1)
        for i, seed in enumerate(seeds):
            item = dict(
                tokens=[1, 2],
                raw_tokens=[1, 2],
                sampler_logprobs=[-0.1, -0.2],
                completion_token_count=2,
                group_seed=seeds[0],
                group_row_index=i,
                group_size=len(seeds),
                seed=seed,
                protocol=protocol,
                prompt_tensor_hash=object_hash(text),
                model_identity=self.stable_model_identity(),
                adapter_identity=self.current_adapter_identity(),
                generation_status="COMPLETE",
                technical_validation_errors=[],
                finish_reason="balanced",
                truncated=False,
                balanced_token_count=2,
            )
            if protocol in ("free", "list"):
                item.update(
                    raw_text='work {"x":2}\n{"answer":1}',
                    generated_text='work {"x":2}\n{"answer":1}',
                    assistant_prefill="",
                    answer_extraction="last_balanced_json",
                    balanced_token_count=None,
                    finish_reason="eos",
                )
            else:
                item.update(decoded_protocol('"answer":1}'))
            if self.invalid:
                item["sampler_logprobs"] = [-0.1]
            on_completion(item)
            if i == self.error_after:
                raise RuntimeError("device failure")


def run(root, runtime=None, **kwargs):
    runtime = runtime or Runtime()
    group, inputs, tasks = fixtures(root, **kwargs)
    result = run_group(
        runtime, group, inputs, {}, tasks, root, root / "eval", source_sha256="source", score=False
    )
    return runtime, result


def test_raw_resume_without_regeneration_and_identity_guard(tmp_path):
    runtime, receipt = run(tmp_path)
    assert runtime.calls == 1 and len(receipt["records"]) == 4
    runtime, again = run(tmp_path)
    assert runtime.calls == 0 and receipt == again
    runtime.version = "other"
    with pytest.raises(PermissionError, match="identity"):
        run(tmp_path, runtime)


def test_partial_group_is_retained_without_resampling(tmp_path):
    runtime = Runtime()
    runtime.error_after = 1
    with pytest.raises(RuntimeError, match="device failure"):
        run(tmp_path, runtime)
    retry = Runtime()
    with pytest.raises(RuntimeError, match="Incomplete durable"):
        run(tmp_path, retry)
    assert retry.calls == 0
    pending = next((tmp_path / "eval/groups").glob("*.pending.jsonl"))
    assert len(pending.read_text().splitlines()) == 2


def test_complete_pending_group_recovers_and_invalid_group_retained(tmp_path):
    runtime = Runtime()
    runtime.error_after = 3
    with pytest.raises(RuntimeError, match="device failure"):
        run(tmp_path, runtime)
    retry, result = run(tmp_path)
    assert retry.calls == 0 and len(result["records"]) == 4
    assert not list((tmp_path / "eval/groups").glob("*.pending.jsonl"))


def test_invalid_raw_published_before_validation(tmp_path):
    runtime = Runtime()
    runtime.invalid = True
    with pytest.raises(ValueError, match="trajectory"):
        run(tmp_path, runtime)
    path = next((tmp_path / "eval/groups").glob("*.json"))
    assert len(json.loads(path.read_text())["records"]) == 4
    retry = Runtime()
    with pytest.raises(ValueError, match="trajectory"):
        run(tmp_path, retry)
    assert retry.calls == 0


def test_test_scores_require_explicit_release_and_free_keeps_workspace(tmp_path):
    group, inputs, tasks = fixtures(tmp_path, "free", 2, "TEST_ID")
    runtime = Runtime()
    with pytest.raises(PermissionError, match="sealed"):
        run_group(
            runtime,
            group,
            inputs,
            {},
            tasks,
            tmp_path,
            tmp_path / "eval",
            source_sha256="source",
            score=True,
        )
    assert runtime.calls == 0
    _, receipt = run(tmp_path, protocol="free", count=2, pool="TEST_ID")
    assert receipt["records"][0]["raw_text"].count("{") == 2


def test_family_summary_reports_missing_extra_and_within_group_contrasts():
    base = dict(A=1, E=0, E6=1, p_read=0.75, P=0, J=0, P6=1, J6=1, missing_count=2, extra_count=0)
    rows = [
        dict(qid="q", family="TOPK", score=base),
        dict(qid="q", family="TOPK", score=dict(base, A=0)),
    ]
    result = summarize_scores(rows)["TOPK"]
    assert result["A"] == 0.5 and result["missing_keys"] == 2 and result["extra_keys"] == 0
    assert result["A_minus_P"] == 0.5 and result["contract_slack"] == 6
    assert result["group_contrasts"]["A"]["nonconstant_group_fraction"] == 1
    assert result["group_contrasts"]["A"]["mean_within_group_variance"] == 0.25


def test_monitor_restores_all_rng_and_model_mode_even_after_failure(tmp_path):
    runtime = Runtime()
    runtime.model.train()
    group, inputs, tasks = fixtures(tmp_path)
    before = state_hash(capture_rng())
    runtime.error_after = 0
    with (
        patch("sr_f2.evaluation.monitor_slots", return_value=group["slots"]),
        pytest.raises(RuntimeError, match="device failure"),
    ):
        evaluate_monitor(
            runtime,
            tasks,
            inputs,
            tmp_path,
            tmp_path / "eval",
            model_id="model",
            step=0,
            source_sha256="source",
        )
    assert state_hash(capture_rng()) == before
    assert runtime.model.training is True


def test_free_list_prepare_has_no_prefill_and_preserves_protocol(monkeypatch):
    from sr_f1.runtime import SRRuntime

    seen = []

    def parent(self, text, image_path, root, **kwargs):
        seen.append(kwargs["protocol"])
        return {"routing": {}}

    monkeypatch.setattr(SRRuntime, "prepare_text", parent)
    runtime = SRF2Runtime.__new__(SRF2Runtime)
    for protocol in ("free", "list"):
        actual = runtime.prepare_text("question", None, "/root", protocol=protocol)
        assert actual["routing"]["assistant_prefill"] == ""
        assert actual["routing"]["protocol"] == protocol
    assert seen == ["plain_answer", "plain_answer"]


@pytest.mark.parametrize("protocol,cap", [("free", 256), ("list", 128)])
def test_workspace_generation_does_not_stop_at_first_json_and_restores_rng(protocol, cap):
    from transformers import GenerationConfig

    from sr_f1.runtime import generation_recipe

    runtime = SRF2Runtime.__new__(SRF2Runtime)
    runtime.torch, runtime.device, runtime.image_calls = torch, "cpu", 0
    runtime.identity, runtime.account = {}, lambda *args: None
    runtime.eos_ids = [2]
    runtime.generation_config = GenerationConfig(**generation_recipe(), eos_token_id=2)
    runtime.model = torch.nn.Linear(1, 1)
    pieces = {3: '{"work":1}', 4: '\n{"answer":2}', 2: ""}
    runtime.processor = SimpleNamespace(
        tokenizer=SimpleNamespace(decode=lambda tokens, **kw: "".join(pieces[t] for t in tokens))
    )
    ids = torch.tensor([[10, 11]])
    prepared = dict(
        inputs=dict(input_ids=ids, attention_mask=torch.ones_like(ids)),
        routing=dict(input_tensor_hash="prompt", image_token_count=0),
    )
    runtime.prepare_text = lambda *args, **kwargs: prepared
    runtime.current_adapter_identity = lambda: {}
    runtime.stable_model_identity = lambda: {}

    def generate(**kwargs):
        assert "stopping_criteria" not in kwargs
        assert kwargs["generation_config"].max_new_tokens == cap
        assert kwargs["generation_config"].do_sample is True
        seq = torch.cat([kwargs["input_ids"], torch.tensor([[3, 4, 2], [3, 4, 2]])], -1)
        logits = tuple(torch.zeros((2, 12)) for _ in range(3))
        return SimpleNamespace(sequences=seq, logits=logits, scores=logits)

    runtime.model.generate = generate
    before = state_hash(capture_rng())
    records = runtime.generate_group(
        None, "/unused", [100, 101], text="question", protocol=protocol
    )
    assert state_hash(capture_rng()) == before
    assert all(r["tokens"] == [3, 4, 2] for r in records)
    assert all(r["raw_text"] == '{"work":1}\n{"answer":2}' for r in records)
    assert all(r["finish_reason"] == "eos" and r["assistant_prefill"] == "" for r in records)


def test_monitor_completes_reuses_raw_and_keeps_step_directories_separate(tmp_path):
    from sr_f1.contract import PACKAGE
    from sr_f2.diagnostics import monitor_panel
    from sr_f2.evaluation import run_monitor

    all_tasks = [
        json.loads(line)
        for line in (PACKAGE / "manifests/TASKS_GOLD_AUDIT_ONLY.jsonl").read_text().splitlines()
    ]
    tasks = {t["qid"]: t for t in monitor_panel(all_tasks)}
    fixtures(tmp_path)
    inputs = {
        qid: dict(text="question", plain_text="question\n\nanswer", image_file="images/image.png")
        for qid in tasks
    }
    runtime = Runtime()
    before = state_hash(capture_rng())
    training_directory = tmp_path / "runs/model"
    training_directory.mkdir(parents=True)
    (training_directory / "COMPLETE.json").write_text('{"training":true}')
    (tmp_path / "REGISTRATION.json").write_text('{"source_manifest_sha256":"source"}')
    with (
        patch("sr_f2.evaluation.load_tasks", return_value=tasks),
        patch("sr_f2.evaluation.load_inputs", return_value=inputs),
    ):
        first = run_monitor(runtime, tmp_path, training_directory, "model", 0)
        calls = runtime.calls
        again = run_monitor(runtime, tmp_path, training_directory, "model", 0)
        assert runtime.calls == calls and first == again
        second = run_monitor(runtime, tmp_path, training_directory, "model", 8)
    assert first["count"] == second["count"] == 512
    assert len(first["families"]) == 4
    assert state_hash(capture_rng()) == before and runtime.model.training is True
    assert (training_directory / "monitor/step-00/COMPLETE.json").exists()
    assert (training_directory / "monitor/step-08/COMPLETE.json").exists()
    assert json.loads((training_directory / "COMPLETE.json").read_text()) == {"training": True}


def test_chartqa_sealed_until_release_with_both_sides_stripped(tmp_path):
    from sr_f2.diagnostics import prepare_input
    from sr_f2.evaluation import score_record

    slot = dict(qid="external-q", protocol="plain_answer", pool="CHARTQA", view=None)
    inputs = {
        "external-q": dict(text="ChartQA question", image_file="external/chartqa/images/chart.png")
    }
    assert prepare_input(slot, inputs, {})["text"] == "ChartQA question"
    invalid = {"external-q": dict(inputs["external-q"], image_file="images/chart.png")}
    with pytest.raises(ValueError, match="Unsafe"):
        prepare_input(slot, invalid, {})
    score = score_record(dict(protocol="plain_answer", raw_text="\nabc\t"), dict(answer="  abc\n"))
    assert score == {"relaxed_accuracy": 1, "exact_match": 1}
    group, original_inputs, tasks = fixtures(tmp_path, "plain_answer", 1, "CHARTQA")
    runtime = Runtime()
    with pytest.raises(PermissionError, match="sealed"):
        run_group(
            runtime,
            group,
            original_inputs,
            {},
            tasks,
            tmp_path,
            tmp_path / "eval",
            source_sha256="source",
            score=True,
        )
    assert runtime.calls == 0
