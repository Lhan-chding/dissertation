"""RED contracts for real Phase-2 layerwise constraint assimilation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from compensability_v4.qwen.phase2_layerwise import (
    build_layerwise_plan,
    execute_layerwise_plan,
    summarize_layerwise_records,
    write_layerwise_outputs,
)

from compensability_v4.qwen.phase2_candidate import (
    CandidateScoringCall,
    CandidateScoringRecord,
    CueCondition,
)

LABELS = ("A", "B", "C", "D")
LABEL_TOKEN_IDS = {"A": 1, "B": 2, "C": 3, "D": 4}
CONDITIONS = tuple(CueCondition)


def _messages(condition: CueCondition) -> tuple[dict[str, str], ...]:
    facts: list[dict[str, object]] = []
    if condition is not CueCondition.NO_CUE:
        facts = [{"type": "known_value", "index": 0, "value": 8}]
    return (
        {"role": "system", "content": "Choose one candidate label."},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "observed_world": [7, 4, 5, 9],
                    "candidates": [
                        {"label": label, "world": list(world)}
                        for label, world in zip(
                            LABELS,
                            (
                                (8, 4, 5, 9),
                                (7, 4, 5, 9),
                                (6, 4, 5, 9),
                                (9, 4, 5, 9),
                            ),
                            strict=True,
                        )
                    ],
                    "facts": facts,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
        },
    )


def _scoring_call(scene_index: int, condition: CueCondition) -> CandidateScoringCall:
    worlds = (
        (8, 4, 5, 9),
        (7, 4, 5, 9),
        (6, 4, 5, 9),
        (9, 4, 5, 9),
    )
    return CandidateScoringCall(
        call_id=f"scene-{scene_index:03d}.{condition.value}",
        scene_id=f"scene-{scene_index:03d}",
        family=("cross_series", "duplicate_encoding", "trend")[scene_index % 3],
        condition=condition,
        observed_world=worlds[1],
        true_world=worlds[0],
        counterfactual_world=worlds[3],
        value_domain=tuple(range(2, 19)),
        facts=tuple(json.loads(_messages(condition)[1]["content"])["facts"]),
        candidate_labels=LABELS,
        candidate_worlds=worlds,
        true_label="A",
        observed_label="B",
        counterfactual_label="D",
        messages=_messages(condition),
    )


def _rendered_prompt(call: CandidateScoringCall) -> str:
    return json.dumps([dict(message) for message in call.messages], separators=(",", ":"))


def _scoring_record(call: CandidateScoringCall) -> CandidateScoringRecord:
    prompt = _rendered_prompt(call)
    logits = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}
    return CandidateScoringRecord(
        call_id=call.call_id,
        scene_id=call.scene_id,
        family=call.family,
        condition=call.condition,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        candidate_labels=call.candidate_labels,
        candidate_worlds=call.candidate_worlds,
        true_world=call.true_world,
        observed_world=call.observed_world,
        counterfactual_world=call.counterfactual_world,
        true_label=call.true_label,
        observed_label=call.observed_label,
        counterfactual_label=call.counterfactual_label,
        candidate_logits=tuple(logits.items()),
        candidate_log_probabilities=tuple((label, value - 4.5) for label, value in logits.items()),
        logp_true=-0.5,
        logp_observed=-1.5,
        logp_counterfactual=-3.5,
        margin_true_observed=1.0,
        margin_counterfactual_observed=-2.0,
        true_rank=1,
        observed_rank=2,
        counterfactual_rank=4,
    )


def _source(number_of_scenes: int):
    calls = tuple(
        _scoring_call(scene_index, condition)
        for scene_index in range(number_of_scenes)
        for condition in CONDITIONS
    )
    return calls, tuple(_scoring_record(call) for call in calls)


def test_layerwise_plan_is_exactly_579_by_four_and_preserves_s3_hashes() -> None:
    scoring_calls, scoring_records = _source(579)

    plan = build_layerwise_plan(
        scoring_calls,
        scoring_records,
        expected_scenes=579,
        expected_language_layers=36,
    )

    assert len(plan) == 2316
    assert len({call.call_id for call in plan}) == 2316
    assert len({call.scene_id for call in plan}) == 579
    assert Counter(call.condition for call in plan) == Counter(
        {condition: 579 for condition in CONDITIONS}
    )
    assert all(call.expected_language_layers == 36 for call in plan)
    assert all(call.source_prompt_sha256 for call in plan)
    assert all(call.candidate_labels == LABELS for call in plan)
    assert all(len(set(call.candidate_worlds)) == 4 for call in plan)


class _Processor:
    tokenizer = SimpleNamespace(
        encode=lambda label, *, add_special_tokens: [LABEL_TOKEN_IDS[label]]
    )

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return json.dumps(messages, separators=(",", ":"))

    def __call__(self, *, text, padding, return_tensors):
        assert padding is True and return_tensors == "pt" and len(text) == 1
        payload = text[0]
        cue_token = 2 if '"facts":[]' not in payload else 1
        return {
            "input_ids": torch.tensor([[7, cue_token]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }


class _Head:
    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                torch.tensor(0.0),
                hidden[0],
                hidden[1],
                -hidden[0],
                -hidden[1],
            )
        )


class _LayerwiseModel:
    device = None

    def __init__(self, *, corrupt_final_forward: bool = False) -> None:
        self.config = SimpleNamespace(num_hidden_layers=36)
        self.model = SimpleNamespace(norm=lambda hidden: hidden)
        self._head = _Head()
        self.corrupt_final_forward = corrupt_final_forward
        self.forward_calls = 0
        self.generate_calls = 0

    def get_output_embeddings(self):
        return self._head

    def __call__(self, **arguments):
        assert arguments["output_hidden_states"] is True
        assert arguments["use_cache"] is False
        assert arguments["return_dict"] is True
        self.forward_calls += 1
        input_ids = arguments["input_ids"]
        cue = float(input_ids[0, -1])
        hidden_states = [torch.zeros((1, 2, 2))]
        for layer in range(36):
            hidden = torch.zeros((1, 2, 2))
            hidden[0, -1] = torch.tensor((cue + layer / 100.0, 1.0))
            hidden_states.append(hidden)
        final_logits = self._head(hidden_states[-1][0, -1])
        if self.corrupt_final_forward:
            final_logits = final_logits.clone()
            final_logits[1] += 0.01
        logits = torch.zeros((1, 2, 5))
        logits[0, -1] = final_logits
        return SimpleNamespace(hidden_states=tuple(hidden_states), logits=logits)

    def generate(self, **_arguments):
        self.generate_calls += 1
        raise AssertionError("layerwise scoring must never generate")


def test_layerwise_execution_uses_36_logit_layers_and_exact_forward_parity() -> None:
    scoring_calls, scoring_records = _source(2)
    plan = build_layerwise_plan(
        scoring_calls,
        scoring_records,
        expected_scenes=2,
        expected_language_layers=36,
    )
    model = _LayerwiseModel()
    progress: list[tuple[int, int]] = []

    records = execute_layerwise_plan(
        model,
        _Processor(),
        plan,
        label_token_ids=LABEL_TOKEN_IDS,
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert len(records) == 8
    assert model.forward_calls == 8
    assert model.generate_calls == 0
    assert progress[-1] == (8, 8)
    assert all(record.language_layers == 36 for record in records)
    assert all(len(record.candidate_logits_by_layer) == 36 for record in records)
    assert all(len(record.margins_true_observed) == 36 for record in records)
    assert all(len(record.margins_counterfactual_observed) == 36 for record in records)
    assert all(len(record.delta_f_by_layer) == 36 for record in records)
    assert all(record.final_forward_parity_verified is True for record in records)
    for scene_id in {record.scene_id for record in records}:
        group = {record.condition: record for record in records if record.scene_id == scene_id}
        no_cue = group[CueCondition.NO_CUE]
        for condition, record in group.items():
            margins = (
                record.margins_counterfactual_observed
                if condition is CueCondition.COUNTERFACTUAL_CUE
                else record.margins_true_observed
            )
            baseline = (
                no_cue.margins_counterfactual_observed
                if condition is CueCondition.COUNTERFACTUAL_CUE
                else no_cue.margins_true_observed
            )
            expected = tuple(
                margin - baseline
                for margin, baseline in zip(
                    margins,
                    baseline,
                    strict=True,
                )
            )
            assert record.delta_f_by_layer == pytest.approx(expected)
            assert condition in CONDITIONS
        assert group[CueCondition.VALID_CUE].assimilation_profile in {
            "no_assimilation",
            "transient_assimilation",
            "persistent_but_insufficient_assimilation",
            "successful_revision",
        }


def test_layerwise_execution_rejects_s3_prompt_hash_drift_before_forward() -> None:
    scoring_calls, scoring_records = _source(1)
    drifted = list(scoring_records)
    drifted[0] = replace(drifted[0], prompt_sha256="0" * 64)
    plan = build_layerwise_plan(
        scoring_calls,
        drifted,
        expected_scenes=1,
        expected_language_layers=36,
    )
    model = _LayerwiseModel()

    with pytest.raises(RuntimeError, match=r"hash|prompt"):
        execute_layerwise_plan(
            model,
            _Processor(),
            plan,
            label_token_ids=LABEL_TOKEN_IDS,
        )
    assert model.forward_calls == 0


def test_layerwise_execution_fails_closed_on_final_forward_mismatch() -> None:
    scoring_calls, scoring_records = _source(1)
    plan = build_layerwise_plan(
        scoring_calls,
        scoring_records,
        expected_scenes=1,
        expected_language_layers=36,
    )

    with pytest.raises(RuntimeError, match=r"final-layer|forward pass|parity"):
        execute_layerwise_plan(
            _LayerwiseModel(corrupt_final_forward=True),
            _Processor(),
            plan,
            label_token_ids=LABEL_TOKEN_IDS,
        )


def test_layerwise_summary_and_outputs_are_complete_objective_and_no_overwrite(
    tmp_path: Path,
) -> None:
    scoring_calls, scoring_records = _source(2)
    plan = build_layerwise_plan(
        scoring_calls,
        scoring_records,
        expected_scenes=2,
        expected_language_layers=36,
    )
    records = execute_layerwise_plan(
        _LayerwiseModel(),
        _Processor(),
        plan,
        label_token_ids=LABEL_TOKEN_IDS,
    )

    summary = summarize_layerwise_records(records)

    assert summary["number_of_scenes"] == 2
    assert summary["number_of_forward_calls"] == 8
    assert summary["language_layers"] == 36
    assert summary["condition_counts"] == {condition.value: 2 for condition in CONDITIONS}
    assert summary["final_forward_parity_verified"] is True
    assert summary["generation_invoked"] is False
    assert summary["subjective_success_threshold_applied"] is False
    assert "minimum_accuracy" not in summary
    assert "minimum_assimilation_rate" not in summary
    assert sum(summary["profile_counts"].values()) == 2
    assert set(summary["paired_effects"]) == {
        "valid_minus_no_cue_margin",
        "sham_minus_no_cue_margin",
        "valid_minus_sham_margin",
        "counterfactual_minus_no_cue_target_margin",
        "counterfactual_minus_sham_target_margin",
    }

    records_path = tmp_path / "layerwise_assimilation/per_scene.jsonl"
    summary_path = tmp_path / "layerwise_assimilation/summary.json"
    write_layerwise_outputs(
        records_path=records_path,
        summary_path=summary_path,
        records=records,
        summary=summary,
    )

    rows = tuple(json.loads(line) for line in records_path.read_text().splitlines())
    assert len(rows) == 8
    assert all(len(row["candidate_logits_by_layer"]) == 36 for row in rows)
    assert json.loads(summary_path.read_text())["number_of_forward_calls"] == 8
    with pytest.raises(FileExistsError, match="overwrite"):
        write_layerwise_outputs(
            records_path=records_path,
            summary_path=summary_path,
            records=records,
            summary=summary,
        )


def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location("phase2_layerwise_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_04_entrypoint_delegates_to_real_hash_bound_runner(monkeypatch) -> None:
    module = _load_script(
        Path(__file__).resolve().parents[1] / "scripts/v4/04_layerwise_assimilation.py"
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        module,
        "run_layerwise_assimilation_cli",
        lambda **kwargs: captured.update(kwargs) or 17,
    )

    assert module.main() == 17
    assert captured["phase"] == "phase_2_layerwise_assimilation"
    assert captured["expected_scenes"] == 579
    assert captured["expected_conditions"] == 4
    assert captured["expected_language_layers"] == 36
    assert captured["output_paths"]["per_scene"].endswith(
        "artifacts/v4/layerwise_assimilation/per_scene.jsonl"
    )
    assert captured["output_paths"]["summary"].endswith(
        "artifacts/v4/layerwise_assimilation/summary.json"
    )
    expected_hashes = captured["expected_input_sha256"]
    assert len(expected_hashes) == 7
    assert all(
        isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")
        for value in expected_hashes
    )
