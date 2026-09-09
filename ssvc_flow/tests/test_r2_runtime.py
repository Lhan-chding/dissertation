"""R2 CPU regressions do not establish real model evidence."""

import json
from collections import Counter

import pytest

from src.core import canonical_hash
from src.r2_runtime import (
    annotate_diagnostic,
    generation_checks,
    paired_effects,
    request_ledger,
    validate_existing_rows,
)

SCENE = {
    "base_scene_id": "s",
    "truth_world": [1, 2, 3, 4],
    "observed_world": [9, 2, 3, 4],
    "changed_index": 0,
    "operation": "sum4",
    "cue": {"family": "duplicate_encoding", "known_index": 0, "known_value": 1},
}


def test_invalid_actions_do_not_satisfy_repair_or_constraints():
    row = annotate_diagnostic("explanation [1,2,3,4]", SCENE)
    assert row["category"] == "I"
    assert not row["repair_changed_coordinate"]
    assert not row["full_constraints_satisfied"]
    assert row["destroyed_correct_coordinate_count"] is None
    assert row["malformed_type"] == "extra_text_or_thinking"


def test_repair_and_damage_are_distinct_from_full_success():
    row = annotate_diagnostic("[1,9,3,4]", SCENE)
    assert row["repair_changed_coordinate"]
    assert row["destroyed_correct_coordinate_count"] == 1
    assert row["destroyed_correct_coordinate_fraction"] == pytest.approx(1 / 3)
    assert row["category"] == "W"
    assert not row["full_constraints_satisfied"]
    copied = annotate_diagnostic("[9,2,3,4]", SCENE)
    assert copied["copy_observation"]
    assert not copied["repair_changed_coordinate"]
    assert not copied["destroyed_any_correct_coordinate"]


def test_thinking_final_metric_never_replaces_strict_action():
    segments = {"status": "RESOLVED", "final_text": "[1,2,3,4]"}
    row = annotate_diagnostic("reasoning</think>[1,2,3,4]", SCENE, thinking=segments)
    assert row["category"] == "I"
    assert row["diagnostic_final_X"] is True
    unresolved = annotate_diagnostic("[1,2,3,4]", SCENE, thinking={"status": "UNRESOLVED"})
    assert unresolved["diagnostic_final_X"] is None


def test_generation_validation_keeps_only_actual_terminal_eos():
    class Tokenizer:
        def decode(self, tokens, **kwargs):
            return ",".join(map(str, tokens))

    gen = {
        "token_ids": [1, 9],
        "raw_completion": "1",
        "completion_length": 2,
        "stop_reason": "eos",
        "behavior_token_logprobs": [-1.0, -0.2],
    }
    assert generation_checks(gen, Tokenizer(), {9}, 64) == []
    assert generation_checks({**gen, "token_ids": [9, 9]}, Tokenizer(), {9}, 64)
    assert generation_checks({**gen, "raw_completion": "changed"}, Tokenizer(), {9}, 64)
    assert generation_checks(
        {**gen, "behavior_token_logprobs": [float("nan"), -1]}, Tokenizer(), {9}, 64
    )


def requests():
    return [
        {
            "base_scene_id": "s",
            "condition": "SYM_ORIGINAL",
            "decode_mode": "sample",
            "rollout_index": 0,
            "max_new_tokens": 64,
            "enable_thinking": False,
            "prompt_id": "p",
        }
    ]


def test_ledger_binds_policy_and_template_and_reuses_rng():
    identity = {"config_hash": "a", "model_hash": "m", "data_hash": "d", "source_hash": "s"}
    first = request_ledger(requests(), identity)
    assert first == request_ledger(requests(), identity)
    changed = request_ledger(requests(), {**identity, "model_hash": "new"})
    assert first[0]["sample_key"] != changed[0]["sample_key"]
    assert first[0]["sample_rng_key"] != changed[0]["sample_rng_key"]
    assert first[0]["sample_seed"] != changed[0]["sample_seed"]
    adapter_changed = request_ledger(requests(), {**identity, "initial_adapter_hash": "new"})
    assert first[0]["sample_seed"] != adapter_changed[0]["sample_seed"]
    with pytest.raises(ValueError, match="duplicate"):
        request_ledger(requests() * 2, identity)


def test_resume_refuses_tampered_and_failed_rows():
    wanted = request_ledger(requests(), {"model_hash": "m"})
    row = {**wanted[0], "execution_checks": {"passed": True}, "category": "X"}
    row["record_hash"] = canonical_hash(row)
    records = {row["sample_key"]: row}
    validate_existing_rows(records, wanted)
    with pytest.raises(ValueError, match="content hash"):
        validate_existing_rows({row["sample_key"]: {**row, "category": "W"}}, wanted)
    bad = {**row, "execution_checks": {"passed": False}}
    bad["record_hash"] = canonical_hash({k: v for k, v in bad.items() if k != "record_hash"})
    with pytest.raises(ValueError, match="failed execution"):
        validate_existing_rows({row["sample_key"]: bad}, wanted)


def analysis_rows():
    rows = []
    for family in ("cross_series", "trend"):
        for i in range(4):
            for condition, success in (("SYM_ORIGINAL", False), ("SYM_CLEAR", i < 2)):
                annotation = annotate_diagnostic("[1,2,3,4]" if success else "[9,2,3,4]", SCENE)
                for sample in range(4):
                    rows.append(
                        {
                            **annotation,
                            "family": family,
                            "condition": condition,
                            "base_scene_id": f"{family}{i}",
                            "decode_mode": "sample",
                            "prompt_id": f"{family}{i}{condition}",
                            "sample_index": sample,
                            "stop_reason": "eos",
                        }
                    )
    return rows


def test_effects_preserve_scene_pairing_and_prompt_weights():
    rows = analysis_rows()
    result = paired_effects(rows, repeats=100, seed=7)
    main = [r for r in result if r["scope"] == "family_standardized" and r["metric"] == "pX"]
    assert len(main) == 1
    assert main[0]["estimate"] == 0.5
    assert main[0]["scene_count"] == 8
    assert main[0]["statistical_unit"] == "base_scene"
    assert main[0]["bootstrap_replicates"] == 100
    amplified = rows + [r for r in rows if r["base_scene_id"] == "trend0"] * 30
    assert paired_effects(amplified, repeats=100, seed=7) == result


def test_missing_pair_is_not_silently_dropped():
    rows = [
        r
        for r in analysis_rows()
        if not (r["base_scene_id"] == "trend0" and r["condition"] == "SYM_CLEAR")
    ]
    with pytest.raises(ValueError, match="paired scene"):
        paired_effects(rows, repeats=100, seed=7)


def test_sampled_and_greedy_have_separate_effects():
    rows = analysis_rows()
    greedy = [{**r, "decode_mode": "greedy"} for r in rows if r["sample_index"] == 0]
    result = paired_effects(rows + greedy, repeats=100, seed=7)
    counts = Counter(r["decode_mode"] for r in result)
    assert counts["sample"] == counts["greedy"]
    json.dumps(result, allow_nan=False)


def test_full_fixture_interrupt_resume_and_manifest_are_honest(tmp_path, monkeypatch):
    """Exercise all 2,232 identities without downloading or claiming a CUDA model."""
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    from test_model_smoke import FakeAdapter
    from test_r1_reference_smoke import setup
    from test_r2_inputs import Tokenizer

    from src import r2_inputs, r2_runtime
    from src.core import file_hash
    from src.optimizer_fork import parameter_hash

    config, _, initial, certificate = setup(tmp_path)
    data_root = Path("data/generated").resolve()
    config["data_root"] = str(data_root)
    binding = {
        "calibration_sha256": file_hash(data_root / "calibration.jsonl"),
        "data_manifest_sha256": file_hash(data_root / "manifest.json"),
    }
    gate = {"config": config, "certificate": certificate, "binding": binding}
    monkeypatch.setitem(
        sys.modules,
        "src.next_stage_runtime",
        SimpleNamespace(
            validate_prerequisites=lambda *args: gate,
            validate_config_against_gate=lambda candidate, checked: (
                candidate if candidate == checked["config"] else None
            ),
            validate_runtime_environment=lambda *args: pytest.fail(
                "fake must not claim real environment validation"
            ),
        ),
    )
    # Source churn in another worker is irrelevant to a synthetic unit fixture.
    monkeypatch.setattr(r2_runtime, "_source", lambda: {"source_commit": "cpu-fixture"})
    adapters = []

    class Adapter(FakeAdapter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.audit = dict(initial.audit)
            self.processor = SimpleNamespace(tokenizer=Tokenizer())
            self.eos_ids = {248046}
            adapters.append(self)

        def generate(self, prepared, *, seed, max_new_tokens, do_sample):
            if len(adapters) == 1 and self.generation_calls == 3:
                raise RuntimeError("fixture interrupted")
            assert not any(p.requires_grad for p in self.model.parameters())
            tokens = [2, 248069, 1, 248046] if prepared["audit"]["enable_thinking"] else [1, 248046]
            self.generation_calls += 1
            self.forward_calls += len(tokens)
            return {
                "raw_completion": self.processor.tokenizer.decode(tokens[:-1]),
                "token_ids": tokens,
                "completion_length": len(tokens),
                "stop_reason": "eos",
                "behavior_token_logprobs": [-1.0] * len(tokens),
                "vision_forward_calls": int(bool(prepared["image"])),
                "elapsed_seconds": 0.001,
            }

    def prepare(adapter, prompt, root, enable_thinking=False):
        value = adapter.prepare(prompt, root)
        value["audit"]["enable_thinking"] = enable_thinking
        return value

    monkeypatch.setattr(r2_inputs, "prepare_r2", prepare)
    out = tmp_path / "r2"
    args = (config, data_root, out, "r0", "r1", "supplement")
    first = r2_runtime.run_r2(*args, _adapter_factory=Adapter)
    assert first["status"] == "FAIL"
    assert first["details"]["completed"] == 3
    before = (out / "samples.jsonl").read_bytes()
    second = r2_runtime.run_r2(*args, resume=True, _adapter_factory=Adapter)
    assert second["status"] == "PASS", second
    assert second["execution_kind"] == "CPU_FAKE_ADAPTER_FIXTURE"
    assert second["details"]["raw_sample_count"] == 2232
    assert second["details"]["generation_calls_this_invocation"] == 2229
    assert (out / "samples.jsonl").read_bytes().startswith(before)
    rows = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
    assert len({row["sample_key"] for row in rows}) == 2232
    assert all(row["execution_kind"] == "CPU_FAKE_ADAPTER_FIXTURE" for row in rows)
    thinking = [r for r in rows if r["condition"] == "SYM_THINKING"]
    assert len(thinking) == 36
    assert all(
        r["category"] == "I" and r["diagnostic_final_status"] == "resolved_final" for r in thinking
    )
    assert second["details"]["adapter_hash_after"] == certificate["initial_adapter_hash"]
    assert not any(p.requires_grad for p in adapters[-1].model.parameters())
    assert (
        parameter_hash(adapters[-1].model, trainable=False)
        == second["details"]["all_parameter_hash_before"]
    )
    third = r2_runtime.run_r2(*args, resume=True, _adapter_factory=Adapter)
    assert third["status"] == "PASS"
    assert third["details"]["generation_calls_this_invocation"] == 0
    assert (out / "samples.jsonl").read_bytes() == (out / "diagnostic_rollouts.jsonl").read_bytes()
    manifest = json.loads((out / "manifest.json").read_text())
    for item in manifest["files"]:
        assert file_hash(out / item["path"]) == item["sha256"]
        assert (out / item["path"]).stat().st_size == item["bytes"]
    attempts = [
        json.loads(line) for line in (out / "execution_attempts.jsonl").read_text().splitlines()
    ]
    assert [a["status"] for a in attempts] == ["FAIL", "PASS", "PASS"]

    class BadScoreAdapter(Adapter):
        def generate(self, prepared, **kwargs):
            returned = super().generate(prepared, **kwargs)
            returned["behavior_token_logprobs"][0] = "bad"
            return returned

    failed_out = tmp_path / "bad-return"
    failed_args = (config, data_root, failed_out, "r0", "r1", "supplement")
    malformed = r2_runtime.run_r2(*failed_args, _adapter_factory=BadScoreAdapter)
    assert malformed["status"] == "FAIL"
    assert malformed["details"]["completed"] == 1
    returned_row = json.loads((failed_out / "samples.jsonl").read_text())
    assert returned_row["generation_return"]["behavior_token_logprobs"][0] == "bad"
    assert returned_row["raw_text"] == "[1,2,3,4]"
    assert returned_row["raw_token_ids"] == [1, 248046]
    assert returned_row["execution_checks"]["passed"] is False
    with pytest.raises(ValueError, match="failed execution"):
        r2_runtime.run_r2(*failed_args, resume=True, _adapter_factory=Adapter)
