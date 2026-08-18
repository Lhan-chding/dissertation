"""Contracts for real S5 exact-cache continuation parity execution."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from compensability_v4.qwen import manual_generation, phase3_cache
from compensability_v4.qwen.cache_continuation import CachedGenerationState
from compensability_v4.qwen.phase2_candidate import CueCondition
from compensability_v4.qwen.phase3_cache import (
    build_cache_parity_plan,
    build_condition_turns,
    execute_cache_parity_plan,
    facts_for_condition,
    summarize_cache_parity,
    write_cache_parity_outputs,
)

S4_PER_SCENE_SHA256 = "e696d12bb8cb3e6142a3d6ecc6de9474c3e72e3ac85e0c7334005a249556a4af"
S4_SUMMARY_SHA256 = "53eab07dcd70fce6970a63ce1831ec6369164e92320f051e717decdeb1b790c0"
CONDITIONS = tuple(CueCondition)


class _Tokenizer:
    eos_token_id = 99

    @staticmethod
    def decode(token_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return ",".join(str(token_id) for token_id in token_ids)


class _Processor:
    tokenizer = _Tokenizer()

    @staticmethod
    def apply_chat_template(messages, *, tokenize, add_generation_prompt):
        assert add_generation_prompt is True
        if tokenize:
            turn = messages[-1]["content"]
            suffix = {
                "no cue": (31, 41),
                "valid facts": (32, 42),
                "sham facts": (33, 43),
                "counterfactual facts": (34, 44),
            }[turn]
            return [[10, 50, 20, *suffix]]
        return json.dumps(messages, separators=(",", ":"))


PROCESSOR = _Processor()


def _state(index: int) -> CachedGenerationState:
    return CachedGenerationState(
        sample_id=f"scene-{index:03d}",
        token_ids=(10, 50, 20),
        attention_mask=(1, 1, 1),
        position_ids=((0, 1, 2), (0, 4, 2), (0, 7, 2)),
        image_token_positions=(1,),
        image_grid_thw=(1, 20, 20),
        visual_token_count=1,
        generation_config={"do_sample": False, "temperature": 0.0},
        rng_seed=20260817 + index,
        past_key_values=object(),
        chat_messages=(
            {"role": "user", "content": "image observation"},
            {"role": "assistant", "content": "7,4,5,9"},
        ),
        generated_token_ids=(20,),
        processor=PROCESSOR,
    )


def _turns(number_of_scenes: int) -> dict[str, dict[CueCondition, str]]:
    values = {
        CueCondition.NO_CUE: "no cue",
        CueCondition.VALID_CUE: "valid facts",
        CueCondition.SHAM_CUE: "sham facts",
        CueCondition.COUNTERFACTUAL_CUE: "counterfactual facts",
    }
    return {f"scene-{index:03d}": dict(values) for index in range(number_of_scenes)}


def test_cache_plan_is_exactly_579_by_four_with_exact_chat_suffixes() -> None:
    states = tuple(_state(index) for index in range(579))

    plan = build_cache_parity_plan(
        states,
        condition_turns=_turns(579),
        expected_scenes=579,
    )

    assert len(plan) == 2316
    assert len({call.call_id for call in plan}) == 2316
    assert len({call.scene_id for call in plan}) == 579
    assert Counter(call.condition for call in plan) == Counter(
        {condition: 579 for condition in CONDITIONS}
    )
    for call in plan:
        full_ids = PROCESSOR.apply_chat_template(
            [*call.cached_state.chat_messages, {"role": "user", "content": call.new_user_text}],
            tokenize=True,
            add_generation_prompt=True,
        )[0]
        assert tuple(full_ids[: len(call.cached_state.token_ids)]) == call.cached_state.token_ids
        assert call.suffix_token_ids == tuple(full_ids[len(call.cached_state.token_ids) :])
        assert call.full_history_messages[-1] == {
            "role": "user",
            "content": call.new_user_text,
        }


def test_s5_frozen_config_gates_decisions_and_reports_full_logit_drift() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load(
        (root / "configs/recoverability/v4_phase_0_3.yaml").read_text(encoding="utf-8")
    )

    contract = payload["phase_3_cache_parity"]
    assert contract["require_exact_generated_logits"] is False
    assert contract["require_stepwise_argmax_parity"] is True
    assert contract["require_realized_token_top1"] is True
    assert contract["report_stepwise_logit_drift"] is True
    assert contract["logit_absolute_tolerance"] == 0.0
    assert contract["logit_relative_tolerance"] == 0.0
    assert "maximum_logit_drift" not in contract


def test_s5_handoff_states_exact_decision_gate_without_logit_magnitude_threshold() -> None:
    root = Path(__file__).resolve().parents[1]
    handoff = (root / "docs/QWEN_V4_SERVER_HANDOFF.md").read_text(encoding="utf-8")

    assert "Token, decision, suffix/MRoPE, and cache-position parity are exact gates." in handoff
    assert "Full-vocabulary logit identity is reported, not required" in handoff
    assert "No logit-drift magnitude threshold is applied." in handoff


class _ParityModel:
    def __init__(self, drift: str | None = None) -> None:
        self.drift = drift
        self.calls = 0
        self.generate_calls = 0

    def compare_cache_and_full_history(self, call, processor, *, max_new_tokens):
        assert processor is PROCESSOR
        assert max_new_tokens == 4
        self.calls += 1
        suffix_length = len(call.suffix_token_ids)
        cache_positions = tuple(
            range(
                len(call.cached_state.token_ids),
                len(call.cached_state.token_ids) + suffix_length,
            )
        )
        suffix_mrope = tuple(
            tuple(axis[-1] + offset for offset in range(1, suffix_length + 1))
            for axis in call.cached_state.position_ids
        )
        trace = {
            "suffix_token_ids": call.suffix_token_ids,
            "cached_generated_token_ids": (1, 1),
            "full_generated_token_ids": (1, 1),
            "cached_generated_logits": (
                torch.tensor([[1.0, 4.0, 2.0]], dtype=torch.bfloat16),
                torch.tensor([[2.0, 5.0, 1.0]], dtype=torch.bfloat16),
            ),
            "full_generated_logits": (
                torch.tensor([[1.0, 4.0, 2.0]], dtype=torch.bfloat16),
                torch.tensor([[2.0, 5.0, 1.0]], dtype=torch.bfloat16),
            ),
            "cached_suffix_position_ids": suffix_mrope,
            "full_suffix_position_ids": suffix_mrope,
            "cached_cache_position": cache_positions,
            "full_cache_position": cache_positions,
            "mrope_axes": 3,
        }
        if self.drift == "token":
            trace["full_generated_token_ids"] = (1, 2)
        elif self.drift == "logit":
            trace["full_generated_logits"] = (
                torch.tensor([[1.5, 4.75, 1.75]], dtype=torch.bfloat16),
                torch.tensor([[2.25, 5.5, 0.75]], dtype=torch.bfloat16),
            )
        elif self.drift == "argmax":
            trace["full_generated_logits"] = (
                torch.tensor([[5.0, 4.0, 2.0]], dtype=torch.bfloat16),
                torch.tensor([[2.0, 5.0, 1.0]], dtype=torch.bfloat16),
            )
        elif self.drift == "realized_not_top1":
            trace["cached_generated_logits"] = trace["full_generated_logits"] = (
                torch.tensor([[5.0, 4.0, 2.0]], dtype=torch.bfloat16),
                torch.tensor([[2.0, 5.0, 1.0]], dtype=torch.bfloat16),
            )
        elif self.drift == "mrope":
            trace["full_suffix_position_ids"] = ((3, 4), (3, 4), (4, 5))
        elif self.drift == "cache_position":
            trace["full_cache_position"] = tuple(position + 1 for position in cache_positions)
        return trace

    def generate(self, **_arguments):
        self.generate_calls += 1
        raise AssertionError("S5 must use the auditable manual cache/full-history paths")


def test_cache_execution_requires_exact_token_decision_mrope_and_cache_position_parity() -> None:
    plan = build_cache_parity_plan(
        (_state(0),),
        condition_turns=_turns(1),
        expected_scenes=1,
    )
    model = _ParityModel()
    progress: list[tuple[int, int]] = []

    records = execute_cache_parity_plan(
        model,
        PROCESSOR,
        plan,
        max_new_tokens=4,
        logit_absolute_tolerance=0.0,
        logit_relative_tolerance=0.0,
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert len(records) == 4
    assert model.calls == 4
    assert model.generate_calls == 0
    assert progress[-1] == (4, 4)
    assert all(record.token_parity_verified for record in records)
    assert all(record.decision_parity_verified for record in records)
    assert all(record.logit_parity_verified for record in records)
    assert all(len(record.logit_step_evidence) == 2 for record in records)
    assert all(record.mrope_parity_verified for record in records)
    assert all(record.cache_position_parity_verified for record in records)
    assert all(record.suffix_parity_verified for record in records)
    assert all(record.mrope_axes == 3 for record in records)
    assert all(record.interface == "I4_exact_cached_natural_continuation" for record in records)
    assert all(record.claim_family == "natural_visual_revision" for record in records)


@pytest.mark.parametrize(
    "drift", ("token", "argmax", "realized_not_top1", "mrope", "cache_position")
)
def test_cache_execution_fails_closed_on_any_parity_drift(drift: str) -> None:
    plan = build_cache_parity_plan(
        (_state(0),),
        condition_turns=_turns(1),
        expected_scenes=1,
    )

    with pytest.raises(RuntimeError, match=r"parity|MRoPE|cache.position"):
        execute_cache_parity_plan(
            _ParityModel(drift),
            PROCESSOR,
            plan,
            max_new_tokens=4,
            logit_absolute_tolerance=0.0,
            logit_relative_tolerance=0.0,
        )


def test_s5_token_divergence_reports_call_step_tokens_and_logit_evidence() -> None:
    class TokenDivergenceModel(_ParityModel):
        def compare_cache_and_full_history(self, call, processor, *, max_new_tokens):
            trace = super().compare_cache_and_full_history(
                call,
                processor,
                max_new_tokens=max_new_tokens,
            )
            return {
                **trace,
                "full_generated_token_ids": (1, 2),
                "full_generated_logits": (
                    torch.tensor([[1.0, 4.0, 2.0]], dtype=torch.float32),
                    torch.tensor([[2.0, 4.0, 6.0]], dtype=torch.float32),
                ),
            }

    call = build_cache_parity_plan(
        (_state(0),),
        condition_turns=_turns(1),
        expected_scenes=1,
    )[0]

    with pytest.raises(RuntimeError) as raised:
        execute_cache_parity_plan(
            TokenDivergenceModel(),
            PROCESSOR,
            (call,),
            max_new_tokens=4,
            logit_absolute_tolerance=0.0,
            logit_relative_tolerance=0.0,
        )

    message = str(raised.value)
    assert "call_id=scene-000.no_cue" in message
    assert "scene_id=scene-000" in message
    assert "cue_condition=no_cue" in message
    assert "first_mismatch_step=1" in message
    assert "cached_token=1" in message
    assert "full_token=2" in message
    assert "common_prefix_length=1" in message
    assert "cached_generated_token_ids=(1, 1)" in message
    assert "full_generated_token_ids=(1, 2)" in message
    assert "cached_logit_shape=(1, 3)" in message
    assert "full_logit_shape=(1, 3)" in message
    assert "cached_logit_dtype=torch.bfloat16" in message
    assert "full_logit_dtype=torch.float32" in message
    assert "cached_argmax=1" in message
    assert "full_argmax=2" in message
    assert "cached_realized_token_logit=5.0" in message
    assert "full_realized_token_logit=6.0" in message
    assert "max_abs_diff=5.0" in message
    assert "max_rel_diff=" in message
    assert "nonzero_count=2" in message
    assert "l2_diff=" in message
    assert "argmax_abs_diff_token_id=2" in message
    assert "allowed_divergence" not in message


def test_cache_summary_and_output_are_objective_and_no_overwrite(tmp_path: Path) -> None:
    plan = build_cache_parity_plan(
        (_state(0), _state(1)),
        condition_turns=_turns(2),
        expected_scenes=2,
    )
    records = execute_cache_parity_plan(
        _ParityModel("logit"),
        PROCESSOR,
        plan,
        max_new_tokens=4,
        logit_absolute_tolerance=0.0,
        logit_relative_tolerance=0.0,
    )

    summary = summarize_cache_parity(records)

    assert summary["number_of_scenes"] == 2
    assert summary["number_of_parity_calls"] == 8
    assert summary["schema_version"] == 2
    assert summary["condition_counts"] == {condition.value: 2 for condition in CONDITIONS}
    assert summary["all_token_parity_verified"] is True
    assert summary["all_decision_parity_verified"] is True
    assert summary["all_logit_parity_verified"] is False
    assert summary["all_mrope_parity_verified"] is True
    assert summary["all_cache_position_parity_verified"] is True
    assert summary["i4_primary_eligible"] is True
    assert summary["subjective_success_threshold_applied"] is False
    assert "minimum_recovery_accuracy" not in summary
    output = tmp_path / "artifacts/v4/cache/cache_parity.json"
    write_cache_parity_outputs(output, records=records, summary=summary)

    payload = json.loads(output.read_text())
    assert len(payload["records"]) == 8
    assert payload["summary"]["number_of_parity_calls"] == 8
    assert payload["records"][0]["decision_parity_verified"] is True
    assert payload["records"][0]["logit_parity_verified"] is False
    assert payload["records"][0]["logit_step_evidence"][0]["exact_identity"] is False
    with pytest.raises(FileExistsError, match="overwrite"):
        write_cache_parity_outputs(output, records=records, summary=summary)


def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location("phase3_cache_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_05_entrypoint_delegates_to_real_s4_hash_bound_runner(monkeypatch) -> None:
    module = _load_script(
        Path(__file__).resolve().parents[1] / "scripts/v4/05_validate_cache_runner.py"
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        module,
        "run_cache_parity_cli",
        lambda **kwargs: captured.update(kwargs) or 19,
    )

    assert module.main() == 19
    assert captured["phase"] == "phase_3_cache_parity"
    assert captured["expected_input_sha256"] == (
        S4_PER_SCENE_SHA256,
        S4_SUMMARY_SHA256,
    )
    assert captured["expected_scenes"] == 579
    assert captured["expected_conditions"] == 4
    assert captured["output_path"].endswith("artifacts/v4/cache/cache_parity.json")


def test_cache_cli_blocks_hash_drift_before_model_loading(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    module = _load_script(
        Path(__file__).resolve().parents[1] / "scripts/v4/05_validate_cache_runner.py"
    )
    model_loaded = False

    def load_model(**_arguments):
        nonlocal model_loaded
        model_loaded = True
        return object(), object()

    inputs = (tmp_path / "per_scene.jsonl", tmp_path / "summary.json")
    for path in inputs:
        path.write_text("{}\n", encoding="utf-8")
    argv = ["05_validate_cache_runner.py", "--execute"]
    for path, digest in zip(
        inputs,
        (S4_PER_SCENE_SHA256, S4_SUMMARY_SHA256),
        strict=True,
    ):
        argv.extend(("--input", str(path), "--input-sha256", digest))
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        module,
        "validate_server_inputs",
        lambda **_arguments: (_ for _ in ()).throw(RuntimeError("S4 hash drift")),
    )
    monkeypatch.setattr(module, "load_pinned_qwen", load_model)

    result = module.run_cache_parity_cli(
        phase="phase_3_cache_parity",
        expected_input_sha256=(S4_PER_SCENE_SHA256, S4_SUMMARY_SHA256),
        expected_scenes=579,
        expected_conditions=4,
        output_path=str(tmp_path / "cache_parity.json"),
    )

    assert result == 2
    assert model_loaded is False
    assert "S4 hash drift" in capsys.readouterr().out


def test_cache_cli_rejects_any_output_ancestor_symlink_before_config_or_model_load(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    module = _load_script(
        Path(__file__).resolve().parents[1] / "scripts/v4/05_validate_cache_runner.py"
    )
    target = tmp_path / "external"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    inputs = (tmp_path / "per_scene.jsonl", tmp_path / "summary.json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "05_validate_cache_runner.py",
            "--execute",
            "--input",
            str(inputs[0]),
            "--input-sha256",
            S4_PER_SCENE_SHA256,
            "--input",
            str(inputs[1]),
            "--input-sha256",
            S4_SUMMARY_SHA256,
        ],
    )
    monkeypatch.setattr(module, "validate_server_inputs", lambda **_arguments: object())
    monkeypatch.setattr(
        module,
        "_load_config",
        lambda _path: (_ for _ in ()).throw(AssertionError("config loaded before output guard")),
    )

    result = module.run_cache_parity_cli(
        phase="phase_3_cache_parity",
        expected_input_sha256=(S4_PER_SCENE_SHA256, S4_SUMMARY_SHA256),
        expected_scenes=579,
        expected_conditions=4,
        output_path=str(linked / "cache" / "cache_parity.json"),
    )

    assert result == 2
    assert "symlink" in capsys.readouterr().out.lower()


def test_s5_condition_facts_rebuild_all_frozen_cue_families() -> None:
    truth = (6, 14, 10, 10)
    observed = (5, 14, 10, 10)
    counterfactual = (5, 15, 10, 10)

    assert (
        facts_for_condition(
            family="trend",
            truth=truth,
            observed=observed,
            counterfactual=counterfactual,
            condition=CueCondition.NO_CUE,
        )
        == ()
    )
    valid_by_family = {
        family: facts_for_condition(
            family=family,
            truth=truth,
            observed=observed,
            counterfactual=counterfactual,
            condition=CueCondition.VALID_CUE,
        )
        for family in ("cross_series", "duplicate_encoding", "trend")
    }
    assert {row["type"] for row in valid_by_family["cross_series"]} == {"pair_sum"}
    assert {row["type"] for row in valid_by_family["duplicate_encoding"]} == {"known_value"}
    assert {row["type"] for row in valid_by_family["trend"]} == {"arithmetic_progression"}
    sham = facts_for_condition(
        family="trend",
        truth=truth,
        observed=observed,
        counterfactual=counterfactual,
        condition=CueCondition.SHAM_CUE,
    )
    counterfactual_facts = facts_for_condition(
        family="trend",
        truth=truth,
        observed=observed,
        counterfactual=counterfactual,
        condition=CueCondition.COUNTERFACTUAL_CUE,
    )
    assert all(row["index"] == 1 and row["value"] == 14 for row in sham)
    assert tuple(row["value"] for row in counterfactual_facts) == counterfactual
    turns = build_condition_turns(
        correction_prompt="Revise from facts.",
        family="trend",
        truth=truth,
        observed=observed,
        counterfactual=counterfactual,
    )
    assert set(turns) == set(CueCondition)
    assert all(turn.startswith("Revise from facts.\n") for turn in turns.values())
    with pytest.raises(ValueError, match="non-empty"):
        build_condition_turns(
            correction_prompt=" ",
            family="trend",
            truth=truth,
            observed=observed,
            counterfactual=counterfactual,
        )


def test_s5_logit_nonidentity_returns_frozen_stepwise_decision_evidence() -> None:
    cached = torch.tensor([[1.0, 4.0, 2.0]], dtype=torch.bfloat16)
    full = torch.tensor([[1.5, 4.75, 1.75]], dtype=torch.bfloat16)

    evidence = phase3_cache._explain_logit_parity(
        (cached,),
        (full,),
        (1,),
        atol=0.0,
        rtol=0.0,
    )

    assert len(evidence) == 1
    step = evidence[0]
    assert isinstance(step, phase3_cache.LogitStepEvidence)
    assert is_dataclass(step)
    with pytest.raises(FrozenInstanceError):
        step.exact_identity = True
    assert step.step_index == 0
    assert step.cached_shape == (1, 3)
    assert step.full_shape == (1, 3)
    assert step.cached_dtype == "torch.bfloat16"
    assert step.full_dtype == "torch.bfloat16"
    assert step.realized_token_id == 1
    assert step.cached_argmax_token_id == 1
    assert step.full_argmax_token_id == 1
    assert step.cached_realized_token_logit == 4.0
    assert step.full_realized_token_logit == 4.75
    assert step.realized_token_logit_delta == 0.75
    assert step.max_abs_diff == 0.75
    assert step.max_rel_diff == pytest.approx(1.0 / 3.0)
    assert step.nonzero_count == 3
    assert step.l2_diff == pytest.approx(0.9354143467)
    assert step.argmax_abs_diff_token_id == 1
    assert step.exact_identity is False
    assert step.decision_parity_verified is True
    assert phase3_cache._logit_hash(cached) != phase3_cache._logit_hash(full)


@pytest.mark.parametrize(
    ("cached", "full", "tokens", "message"),
    [
        (
            torch.tensor([[1.0, 4.0, 2.0]]),
            torch.tensor([[1.0], [4.0], [2.0]]),
            (1,),
            "shape",
        ),
        (
            torch.tensor([[1.0, float("nan"), 2.0]]),
            torch.tensor([[1.0, 4.0, 2.0]]),
            (1,),
            "finite",
        ),
        (
            torch.tensor([[1.0, 4.0, 2.0]]),
            torch.tensor([[float("inf"), 4.0, 2.0]]),
            (1,),
            "finite",
        ),
        (
            torch.tensor([[1.0, 4.0, 2.0]]),
            torch.tensor([[5.0, 4.0, 2.0]]),
            (1,),
            "argmax",
        ),
        (
            torch.tensor([[1.0, 4.0, 2.0]]),
            torch.tensor([[1.5, 4.75, 1.75]]),
            (2,),
            "realized",
        ),
    ],
)
def test_s5_logit_evidence_fails_closed_on_objective_decision_invalidity(
    cached: torch.Tensor,
    full: torch.Tensor,
    tokens: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        phase3_cache._explain_logit_parity(
            (cached,),
            (full,),
            tokens,
            atol=0.0,
            rtol=0.0,
        )


def test_s5_execution_accepts_decision_exact_logit_drift_and_records_it() -> None:
    plan = build_cache_parity_plan(
        (_state(0),),
        condition_turns=_turns(1),
        expected_scenes=1,
    )

    records = execute_cache_parity_plan(
        _ParityModel("logit"),
        PROCESSOR,
        plan,
        max_new_tokens=4,
        logit_absolute_tolerance=0.0,
        logit_relative_tolerance=0.0,
    )

    assert len(records) == 4
    assert all(record.decision_parity_verified is True for record in records)
    assert all(record.logit_parity_verified is False for record in records)
    assert all(
        all(not step.exact_identity for step in record.logit_step_evidence) for record in records
    )


def test_s5_real_trace_joins_qwen_vision_batch_manual_cache_and_full_paths(
    monkeypatch,
) -> None:
    state = _state(0)
    call = build_cache_parity_plan(
        (state,),
        condition_turns=_turns(1),
        expected_scenes=1,
    )[0]
    full_ids = state.token_ids + call.suffix_token_ids

    class RealPathProcessor(_Processor):
        def __call__(self, **arguments):
            assert arguments["images"] == ["frozen-image"]
            assert arguments["videos"] == []
            return {
                "input_ids": torch.tensor([full_ids]),
                "image_grid_thw": torch.tensor([[1, 20, 20]]),
            }

    processor = RealPathProcessor()
    fake_vision_module = SimpleNamespace(
        process_vision_info=lambda _messages: (["frozen-image"], [])
    )
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_vision_module)
    calls: list[dict[str, object]] = []

    def fake_manual(_model, batch, **arguments):
        calls.append({"batch": batch, **arguments})
        cached = len(calls) == 1
        return SimpleNamespace(
            generated_token_ids=(1, 1),
            generated_logits=(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])),
            forward_position_ids=(
                ((3, 4), (3, 4), (3, 4))
                if cached
                else ((0, 1, 2, 3, 4), (0, 1, 2, 3, 4), (0, 1, 2, 3, 4)),
            ),
            forward_cache_positions=((3, 4) if cached else (0, 1, 2, 3, 4),),
        )

    monkeypatch.setattr(phase3_cache, "manual_greedy_generate", fake_manual)

    trace = phase3_cache._real_cache_full_trace(
        SimpleNamespace(device=torch.device("cpu")),
        processor,
        call,
        max_new_tokens=4,
    )

    assert trace["cached_generated_token_ids"] == trace["full_generated_token_ids"]
    assert trace["cached_suffix_position_ids"] == trace["full_suffix_position_ids"]
    assert trace["cached_cache_position"] == trace["full_cache_position"] == (3, 4)
    assert calls[0]["past_key_values"] is not state.past_key_values
    assert calls[0]["prior_token_ids"] == state.token_ids
    assert "past_key_values" not in calls[1]


def test_manual_cached_trace_derives_cache_positions_when_prepare_omits_them() -> None:
    """Match the Transformers 5.14.1 prepare-inputs cache contract."""

    class Cache:
        def __init__(self, sequence_length: int) -> None:
            self.sequence_length = sequence_length

        def get_seq_length(self) -> int:
            return self.sequence_length

    class Model:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.prepare_calls: list[tuple[int, bool, int, int]] = []

        def prepare_inputs_for_generation(self, input_ids, **arguments):
            next_length = arguments.pop("next_sequence_length")
            first_iteration = arguments.pop("is_first_iteration")
            cache = arguments["past_key_values"]
            prepared_ids = input_ids[:, -next_length:]
            self.prepare_calls.append(
                (
                    next_length,
                    first_iteration,
                    cache.get_seq_length(),
                    int(prepared_ids.shape[-1]),
                )
            )
            positions = arguments["position_ids"][:, :, -next_length:]
            return {
                **arguments,
                "input_ids": prepared_ids,
                "position_ids": positions,
                # Transformers 5.14.1 may omit cache_position here.
            }

        def __call__(self, **arguments):
            prepared_length = int(arguments["input_ids"].shape[-1])
            prior_length = arguments["past_key_values"].get_seq_length()
            logits = torch.zeros((1, prepared_length, 8), dtype=torch.bfloat16)
            logits[0, -1, 4] = 1.0
            return SimpleNamespace(
                logits=logits,
                past_key_values=Cache(prior_length + prepared_length),
            )

        @staticmethod
        def _update_model_kwargs_for_generation(output, arguments, *, is_encoder_decoder):
            assert is_encoder_decoder is False
            return {
                **arguments,
                "past_key_values": output.past_key_values,
                "attention_mask": torch.cat(
                    (arguments["attention_mask"], torch.ones((1, 1), dtype=torch.long)),
                    dim=-1,
                ),
            }

    model = Model()

    result = manual_generation.manual_greedy_generate(
        model,
        {
            "input_ids": torch.tensor([[30, 31]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        },
        max_new_tokens=1,
        past_key_values=Cache(2),
        prior_token_ids=(10, 20),
        prior_position_ids=((0, 1), (0, 1), (0, 1)),
    )

    assert model.prepare_calls == [
        (2, True, 2, 2),
        (1, False, 4, 1),
    ]
    assert result.forward_cache_positions == ((2, 3), (4,))
    assert result.generated_token_ids == (4,)
    assert result.generated_logits[0].dtype == torch.bfloat16


def test_s5_full_history_uses_structured_chat_batch_with_runtime_grid() -> None:
    class Batch(dict):
        moved = False

        def to(self, device):
            assert str(device) == "cpu"
            self.moved = True
            return self

    batch = Batch(
        input_ids=torch.tensor([[10, 50, 20, 31, 41]]),
        image_grid_thw=torch.tensor([[1, 20, 20]]),
        pixel_values=torch.tensor([[1.0]]),
    )

    class StructuredProcessor:
        @staticmethod
        def apply_chat_template(messages, **arguments):
            assert messages[-1]["content"] == "no cue"
            assert arguments == {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_dict": True,
                "return_tensors": "pt",
            }
            return batch

        def __call__(self, **_arguments):
            raise AssertionError("structured multimodal chat must not be retokenized")

    prepared = phase3_cache._prepare_full_history_batch(
        StructuredProcessor(),
        (*_state(0).chat_messages, {"role": "user", "content": "no cue"}),
        SimpleNamespace(device=torch.device("cpu")),
    )

    assert prepared is batch
    assert prepared["image_grid_thw"].tolist() == [[1, 20, 20]]
    assert batch.moved is True


def test_s5_plan_summary_and_writer_reject_incomplete_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_cache_parity_plan((), condition_turns={}, expected_scenes=0)
    with pytest.raises(RuntimeError, match="count or identifiers"):
        build_cache_parity_plan((_state(0),), condition_turns=_turns(1), expected_scenes=2)
    with pytest.raises(RuntimeError, match="align"):
        build_cache_parity_plan((_state(0),), condition_turns={}, expected_scenes=1)
    incomplete_turns = _turns(1)
    incomplete_turns["scene-000"].pop(CueCondition.SHAM_CUE)
    with pytest.raises(RuntimeError, match="all four"):
        build_cache_parity_plan(
            (_state(0),),
            condition_turns=incomplete_turns,
            expected_scenes=1,
        )
    with pytest.raises(ValueError, match="non-empty"):
        summarize_cache_parity(())
    with pytest.raises(ValueError, match="must not be empty"):
        write_cache_parity_outputs(
            tmp_path / "empty.json",
            records=(),
            summary={},
        )
