"""CPU-only regressions; fake tokenizer/routes never certify a native model run."""

from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from sr_f1 import format_review as review
from sr_f1.contract import PACKAGE, PLAN_ID, digest, file_hash, stable_seed
from sr_f1.data import read_jsonl


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


class FakeTokenizer:
    """A deterministic test double, deliberately not a Qwen tokenizer."""

    def __init__(self):
        self.decoded = {1: '{"evidence":[],"answer":0}', 2: "malformed reply"}
        self.text_tokens = {}

    def encode(self, text, add_special_tokens=False):
        if text not in self.text_tokens:
            token = 1000 + len(self.text_tokens)
            self.text_tokens[text] = token
            self.decoded[token] = text
        return [self.text_tokens[text]] * 64

    def decode(self, tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return self.decoded[tokens[0]]


def _record(index=0, *, truncated=False, covered=False):
    tokens = [1 if covered else 2] * (768 if truncated else 3)
    if not truncated:
        tokens += [248044]
    return dict(
        raw_text=FakeTokenizer().decoded[tokens[0]],
        tokens=tokens,
        raw_tokens=tokens[:],
        completion_token_count=len(tokens),
        old_logprobs=[-1.0] * len(tokens),
        sampler_logprobs=[-1.0] * len(tokens),
        old_logprob_source="actual_generate_raw_logits_selected_token",
        finish_reason="length" if truncated else "eos",
        truncated=truncated,
        sample_index=index,
    )


def test_truncation_alone_can_explain_shortfall_so_bridge_stays_blocked():
    with pytest.raises(PermissionError, match="Truncation alone"):
        review.coverage_evidence(
            dict(responses=256, covered=240, truncated=16, untruncated=240, untruncated_covered=240)
        )


def test_original_full_denominator_proves_protocol_deficiency_without_accuracy():
    evidence = review.coverage_evidence(
        dict(responses=256, covered=93, truncated=22, untruncated=234, untruncated_covered=93)
    )
    assert evidence["upper_bound_coverage_if_all_truncated_valid"] == 115 / 256
    assert evidence["untruncated_coverage"] == 93 / 234
    assert evidence["required_coverage"] == 0.95 and evidence["full_denominator"] == 256
    assert evidence["decision_uses_answer_or_evidence_accuracy"] is False
    task = read_jsonl(PACKAGE / "manifests/TASKS_GOLD_AUDIT_ONLY.jsonl")[0]
    raw = _record(covered=True)
    raw["qid"] = task["qid"]
    counts, failures = review._coverage_counts([raw], {task["qid"]: task})
    assert counts["covered"] == 1  # Empty evidence is legal but not task evidence correctness.
    assert not {"A", "E", "P", "J"} & failures["rows"][0]["field_flags"].keys()


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda r: r.update(finish_reason="length"), "Finish reason"),
        (lambda r: r.update(truncated=True), "Truncation flag"),
        (lambda r: r["tokens"].__setitem__(0, 248044), "token streams"),
        (lambda r: r.update(raw_text="repaired response"), "decoding"),
        (lambda r: r["old_logprobs"].__setitem__(0, float("nan")), "log probabilities"),
        (lambda r: r.update(completion_token_count=100), "token streams"),
    ],
)
def test_token_stopping_or_probability_corruption_is_rejected(change, match):
    raw = _record(covered=True)
    change(raw)
    with pytest.raises(PermissionError, match=match):
        review._validate_tokens(raw, FakeTokenizer(), {"eos_token_id": [248044]})


def test_internal_eos_and_short_non_eos_completion_are_rejected():
    raw = _record(covered=True)
    raw["tokens"][1] = 248044
    raw["raw_tokens"] = raw["tokens"][:]
    with pytest.raises(PermissionError, match="before final"):
        review._validate_tokens(raw, FakeTokenizer(), {"eos_token_id": [248044]})
    raw = _record(covered=True)
    raw["tokens"][-1] = 1
    raw["raw_tokens"] = raw["tokens"][:]
    with pytest.raises(PermissionError, match="before the fixed"):
        review._validate_tokens(raw, FakeTokenizer(), {"eos_token_id": [248044]})
    review._validate_tokens(_record(truncated=True), FakeTokenizer(), {"eos_token_id": [248044]})


@pytest.fixture
def cpu_review_fixture(tmp_path, monkeypatch):
    tasks = {r["qid"]: r for r in read_jsonl(PACKAGE / "manifests/TASKS_GOLD_AUDIT_ONLY.jsonl")}
    inputs = {r["qid"]: r for r in read_jsonl(PACKAGE / "manifests/MODEL_INPUTS.jsonl")}
    for name in ("MODEL_INPUTS.jsonl", "TASKS_GOLD_AUDIT_ONLY.jsonl"):
        path = tmp_path / "manifests" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((PACKAGE / "manifests" / name).read_bytes())
    (tmp_path / "config").mkdir()
    (tmp_path / "config/SR_F1.json").write_bytes((PACKAGE / "config/SR_F1.json").read_bytes())
    generation = dict(
        max_new_tokens=768,
        eos_token_id=[248044],
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
    )
    identity = dict.fromkeys(review.MODEL_KEYS, "frozen-test-only")
    identity["generation_config_expanded"] = generation
    model = dict(identity, model_path="test-only-no-model", model_weights_hash="base")
    _write(tmp_path / "MODEL_ENVIRONMENT_IDENTITY.json", model)
    _write(tmp_path / "PROCESSOR_PREFLIGHT.json", {"status": "PASS"})
    _write(tmp_path / "EXECUTION_FREEZE.json", {"test_fixture": True})
    zero = dict(
        plan_id=PLAN_ID,
        status="VERIFIED",
        model_id="SRF1_COMMON_ZERO",
        step=None,
        trainable_state_hash="policy",
        adapter_hash=digest({"adapter_model.safetensors": "bytes"}),
        adapter_file_hashes={"adapter_model.safetensors": "bytes"},
        adapter_path="states/SRF1_COMMON_ZERO",
        base_model_weights_hash="base",
    )
    _write(tmp_path / "COMMON_ZERO_LORA.json", zero)
    adapter = tmp_path / "states/SRF1_COMMON_ZERO/adapter_model.safetensors"
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b"test-only-not-weights")
    route = dict(
        image_token_count=768,
        processed_size=[1024, 768],
        processor_target_pixels=786432,
        chat_text_sha256="chat",
        input_ids_sha256="ids",
        input_tensor_hash="tensors",
        native_input_tensor_hashes={"input_ids": "tensor"},
        source_image_sha256="image",
    )
    routes = [dict(qid=qid, image_file=inputs[qid]["image_file"], **route) for qid in tasks]
    (tmp_path / "PROCESSOR_INPUT_ROUTES.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in routes)
    )
    freeze = dict(
        artifact_hashes={
            p: file_hash(tmp_path / p)
            for p in (
                "MODEL_ENVIRONMENT_IDENTITY.json",
                "PROCESSOR_PREFLIGHT.json",
                "PROCESSOR_INPUT_ROUTES.jsonl",
            )
        }
    )
    qids, _ = review._raw_paths(tasks)
    for ordinal, (qid, draw) in enumerate((q, i) for q in qids for i in range(8)):
        raw = _record(draw, truncated=ordinal >= 234, covered=ordinal < 93)
        raw.update(
            qid=qid,
            seed=stable_seed(PLAN_ID, "FORMAT", "FORMAT", qid, draw),
            policy_hash="policy",
            input_hash=digest(inputs[qid]),
            generation_status="COMPLETE",
            technical_validation_errors=[],
            sampling_parameters=generation,
            sampling_hash=review.hash_json(generation),
            image_routing=dict(route, generation_vision_forward_calls=1),
            input_routing=dict(route, generation_vision_forward_calls=1),
            model_identity=dict(identity, **zero),
            adapter_identity=dict(
                adapter_parameter_hash="policy",
                published_adapter_hash=zero["adapter_hash"],
                base_model_weights_hash="base",
                model_id="SRF1_COMMON_ZERO",
                step=None,
            ),
            prompt_token_count=7,
        )
        raw["record_hash"] = digest(raw)
        _write(tmp_path / f"engineering/format/before/{qid}-{draw}.json", raw)
    coverage = dict(
        panel="FORMAT",
        label="before",
        policy_hash="policy",
        responses=256,
        covered=93,
        coverage=93 / 256,
        truncated=22,
        coverage_uses_gold_accuracy=False,
    )
    _write(tmp_path / review.COVERAGE, coverage)
    block = dict(
        status="PROTOCOL_BLOCKED",
        reason="FORMAT_TRUNCATION_REQUIRES_TECHNICAL_REVIEW",
        bridge_executed=False,
        format_before=coverage,
    )
    _write(tmp_path / "FORMAT_AND_BRIDGE_RECEIPT.json", block)
    calls = Counter()
    tokenizer = FakeTokenizer()

    def prepare(row, root):
        calls[row["qid"]] += 1
        return dict(routing=route, inputs={"input_ids": np.zeros((1, 7), dtype=np.int64)})

    runtime = SimpleNamespace(
        device="cpu",
        model=SimpleNamespace(),
        identity=identity,
        eos_ids=[248044],
        generation_config=SimpleNamespace(to_dict=lambda: generation),
        processor=SimpleNamespace(tokenizer=tokenizer),
        prepare=prepare,
        encode_completion=lambda text: [*tokenizer.encode(text), 248044],
    )
    monkeypatch.setattr(review, "_load_processor", lambda path: runtime)
    # Only the immutable-input chain is stubbed. Slot, token, parser, routing,
    # coverage, artifact-binding and build/verify logic execute normally.
    monkeypatch.setattr(
        review, "_historical_inputs", lambda root: (tasks, inputs, freeze, model, zero)
    )
    return tmp_path, calls, runtime


def test_cpu_review_checks_exact_original_slots_then_remains_valid_after_bridge(cpu_review_fixture):
    root, calls, _ = cpu_review_fixture
    original = (root / "FORMAT_AND_BRIDGE_RECEIPT.json").read_bytes()
    result = review.build_format_review(root)
    assert result["status"] == "VERIFIED_PROTOCOL_NONADHERENCE"
    assert result["permit_bridge"] and result["model_calls"] == 0
    assert result["counts"] == dict(
        responses=256, covered=93, truncated=22, untruncated=234, untruncated_covered=93
    )
    assert len(calls) == 32 and set(calls.values()) == {1}
    assert result["gold_completion_channel_audit"]["pool_counts"] == {
        "FORMAT": 32,
        "FORMAT_CONFIRM": 32,
        "BRIDGE": 128,
    }
    assert (
        result["field_coverage_diagnostics"]["eos_responses"][
            "invalid_whole_json_or_top_level_contract"
        ]
        == 141
    )
    assert (root / review.ARCHIVE).read_bytes() == original
    assert (root / "FORMAT_AND_BRIDGE_RECEIPT.json").read_bytes() == original
    _write(root / "COMMON_START.json", {"new": "bridge start"})
    (root / "engineering/bridge").mkdir()
    (root / "training").mkdir()
    _write(root / "FORMAT_AND_BRIDGE_RECEIPT.json", {"status": "COMPLETE"})
    assert review.verify_format_review(root) == result
    assert review.build_format_review(root) == result  # Idempotent, no second native prepare.
    assert set(calls.values()) == {1}


def test_review_binds_raw_bytes_and_receipt_counts(cpu_review_fixture):
    root, _, _ = cpu_review_fixture
    result = review.build_format_review(root)
    result["counts"]["covered"] = 94
    result["receipt_hash"] = digest({k: v for k, v in result.items() if k != "receipt_hash"})
    _write(root / review.RECEIPT, result)
    with pytest.raises(PermissionError, match="field-coverage counts"):
        review.verify_format_review(root)


def test_review_rejects_changed_raw_even_with_fresh_record_hash(cpu_review_fixture):
    root, _, _ = cpu_review_fixture
    review.build_format_review(root)
    path = next(
        p for p in (root / "engineering/format/before").glob("*.json") if p.name != "COVERAGE.json"
    )
    raw = json.loads(path.read_text())
    raw["raw_text"] = "changed"
    raw["record_hash"] = digest({k: v for k, v in raw.items() if k != "record_hash"})
    _write(path, raw)
    with pytest.raises(PermissionError, match="artifact bytes"):
        review.verify_format_review(root)


def test_review_cannot_override_stop_or_a_prior_bridge(cpu_review_fixture):
    root, _, _ = cpu_review_fixture
    (root / "STOP").touch()
    with pytest.raises(PermissionError, match="STOP"):
        review.build_format_review(root)
    (root / "STOP").unlink()
    (root / "engineering/bridge").mkdir()
    with pytest.raises(PermissionError, match="prior bridge"):
        review.build_format_review(root)


def test_review_rejects_a_gold_completion_that_cannot_fit(cpu_review_fixture):
    root, _, runtime = cpu_review_fixture
    runtime.encode_completion = lambda text: [1] * 768 + [248044]
    with pytest.raises(PermissionError, match="gold completion"):
        review.build_format_review(root)
    assert not (root / review.RECEIPT).exists()
