import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sr_f1.evaluation import sha_file
from sr_f1.json_protocol import decoded_protocol
from sr_f12.evaluation import (
    collect_scored_shards,
    evaluate_shard,
    group_slots,
    score_record,
    shard_lease,
    verify_source_manifest,
)
from sr_f12.protocol import BASELINE, object_hash


def fixtures():
    slots = []
    for question in range(3):
        for draw in range(8):
            slots.append(
                dict(
                    slot_id=f"{question}-{draw}",
                    model_id=BASELINE,
                    step=0,
                    pool="MONITOR",
                    protocol="evidence_answer",
                    view=None,
                    qid=f"q{question}",
                    root_id=f"r{question}",
                    family="TOPK",
                    chart="bar",
                    variant="v0",
                    draw=draw,
                    samples_per_prompt=8,
                    sampling_seed=question * 100 + draw,
                    group_seed=question * 100,
                    generation=dict(max_new_tokens=768, num_return_sequences=1),
                )
            )
    inputs = {
        f"q{i}": dict(text=f"question {i}", plain_text=f"answer {i}", image_file=None)
        for i in range(3)
    }
    tasks = {key: dict(world={}, query={}) for key in inputs}
    return slots, inputs, tasks


class Runtime:
    def __init__(self):
        self.processor = SimpleNamespace(tokenizer=object())
        self.calls = []
        self.invalid = False
        self.error_after = None
        self.version = "base-sha"
        self.group_offset = 0

    def stable_model_identity(self):
        return dict(model_sha=self.version)

    def current_adapter_identity(self):
        return dict(adapter_path=None, adapter_parameter_hash=None)

    def generate_group(
        self, row, root, seeds, *, text, image_path, protocol, generation, on_completion
    ):
        self.calls.append(seeds)
        assert generation["num_return_sequences"] == len(seeds)
        for index, seed in enumerate(seeds):
            record = dict(
                tokens=[1, 2],
                raw_tokens=[1, 2],
                sampler_logprobs=[-0.2, -0.3],
                completion_token_count=2,
                group_seed=seeds[0] + self.group_offset,
                group_row_index=index,
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
            record.update(decoded_protocol('"answer":1}'))
            if self.invalid:
                record["sampler_logprobs"] = [-0.2]
            on_completion(record)
            if self.error_after == index:
                raise RuntimeError("device failure")


def run(root, runtime=None, rank=0, count=1, **kwargs):
    runtime = runtime or Runtime()
    slots, inputs, tasks = fixtures()
    kwargs.setdefault("score_non_test", False)
    result = evaluate_shard(
        runtime,
        slots,
        inputs,
        {},
        tasks,
        root,
        root / "evaluation" / f"shard{rank:03d}",
        shard_index=rank,
        shard_count=count,
        source_sha256="source-sha",
        **kwargs,
    )
    return runtime, result


def test_group_shards_are_disjoint_whole_questions_and_resume_without_generation(tmp_path):
    observed = []
    for rank in range(2):
        runtime, result = run(tmp_path, rank=rank, count=2)
        observed += runtime.calls
        assert result["status"] == "COMPLETE"
        resumed, again = run(tmp_path, rank=rank, count=2)
        assert not resumed.calls
        assert again == result
    assert sorted(seed for group in observed for seed in group) == sorted(
        s["sampling_seed"] for s in fixtures()[0]
    )
    assert len(observed) == 3


def test_changed_identity_or_input_is_not_skipped(tmp_path):
    run(tmp_path)
    changed = Runtime()
    changed.version = "different-model"
    with pytest.raises(PermissionError, match="Immutable"):
        run(tmp_path, runtime=changed)
    slots, inputs, tasks = fixtures()
    inputs["q0"]["text"] = "different prompt"
    with pytest.raises(PermissionError, match="identity differs"):
        evaluate_shard(
            Runtime(),
            slots,
            inputs,
            {},
            tasks,
            tmp_path,
            tmp_path / "evaluation/shard000",
            shard_index=0,
            shard_count=1,
            source_sha256="source-sha",
            score_non_test=False,
        )


def test_invalid_raw_is_durable_before_validation_or_score(tmp_path):
    runtime = Runtime()
    runtime.invalid = True
    with patch("sr_f12.evaluation.score_record") as scorer:
        with pytest.raises(ValueError, match="trajectory"):
            run(tmp_path, runtime=runtime, score_non_test=True)
        scorer.assert_not_called()
    files = list((tmp_path / "evaluation/shard000/groups").glob("*.json"))
    assert len(files) == 1
    assert len(json.loads(files[0].read_text())["records"]) == 8
    retry = Runtime()
    with pytest.raises(ValueError, match="trajectory"):
        run(tmp_path, runtime=retry)
    assert not retry.calls


def test_partial_group_is_preserved_and_never_silently_resampled(tmp_path):
    runtime = Runtime()
    runtime.error_after = 2
    with pytest.raises(RuntimeError, match="device failure"):
        run(tmp_path, runtime=runtime)
    retry = Runtime()
    with pytest.raises(RuntimeError, match="Incomplete durable group"):
        run(tmp_path, runtime=retry)
    assert not retry.calls
    pending = list((tmp_path / "evaluation/shard000/groups").glob("*.pending.jsonl"))
    assert len(pending) == 1 and len(pending[0].read_text().splitlines()) == 3


def test_complete_pending_group_recovers_without_resampling(tmp_path):
    runtime = Runtime()
    runtime.error_after = 7
    with pytest.raises(RuntimeError, match="device failure"):
        run(tmp_path, runtime=runtime)
    retry, result = run(tmp_path)
    assert result["generated_records"] == 24
    assert len(retry.calls) == 2


def test_actual_runtime_group_seed_mismatch_survives_slot_decoration(tmp_path):
    runtime = Runtime()
    runtime.group_offset = 1
    with pytest.raises(PermissionError, match="identity differs"):
        run(tmp_path, runtime=runtime)
    assert list((tmp_path / "evaluation/shard000/groups").glob("*.json"))


def test_stop_boundary_and_duplicate_worker(tmp_path):
    (tmp_path / "STOP").touch()
    runtime, result = run(tmp_path)
    assert result["status"] == "CHECKPOINTED" and not runtime.calls
    with (
        shard_lease(tmp_path / "same"),
        pytest.raises(RuntimeError, match="Duplicate"),
        shard_lease(tmp_path / "same"),
    ):
        pass


def test_scored_collection_requires_exact_coverage_and_detects_tamper(tmp_path):
    with patch("sr_f12.evaluation.score_record", return_value={"A": 1, "J": 0}):
        for rank in range(2):
            run(tmp_path, rank=rank, count=2, score_non_test=True)
    rows = collect_scored_shards(tmp_path / "evaluation", expected_shards=2, expected_records=24)
    assert len(rows) == 24 and rows[0]["score"]["A"] == 1
    raw = next((tmp_path / "evaluation/shard000/groups").glob("*.json"))
    value = json.loads(raw.read_text())
    value["records"][0]["tokens"][0] = 17
    raw.write_text(json.dumps(value))
    with pytest.raises(PermissionError, match="checksum"):
        collect_scored_shards(tmp_path / "evaluation", expected_shards=2, expected_records=24)


def test_group_slot_validation():
    slots, _, _ = fixtures()
    with pytest.raises(ValueError, match="Incomplete"):
        group_slots(slots[:-1])
    with pytest.raises(ValueError, match="Non-contiguous"):
        group_slots(slots[:8] + slots[8:16] + slots[:8])


def test_source_manifest_verifies_actual_bytes_and_containment(tmp_path):
    source = tmp_path / "code"
    source.mkdir()
    script = source / "runner.py"
    script.write_text("pass\n")
    manifest = {
        "git_commit": "a" * 40,
        "files": [{"path": "runner.py", "sha256": sha_file(script), "bytes": 5}],
    }
    file = tmp_path / "manifest.json"
    file.write_text(json.dumps(manifest))
    assert verify_source_manifest(file, source) == object_hash(manifest)
    script.write_text("fail\n")
    with pytest.raises(PermissionError, match="identity"):
        verify_source_manifest(file, source)
    manifest["files"][0]["path"] = "../manifest.json"
    file.write_text(json.dumps(manifest))
    with pytest.raises(PermissionError, match="contained"):
        verify_source_manifest(file, source)


def test_invalid_protocol_response_is_scored_as_failure_without_legacy_old_logprobs():
    row = dict(
        raw_text="{}junk", format_protocol_error="trailing_after_close", protocol="evidence_answer"
    )
    with patch("sr_f12.evaluation.score_response", return_value={"A": 0}) as original:
        result = score_record(row, {"world": {}, "query": {}})
    original.assert_called_once_with("", {"world": {}, "query": {}}, "evidence_answer")
    assert result["reason"] == "trailing_after_close"
