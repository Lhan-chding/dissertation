"""Semantic contracts and optional model-free integration on the actual archive."""

import json
import math
import os

import pytest

from src.decision_modeling.reanalysis import _check_action, _support, reanalyze, score_key
from src.decision_modeling.semantic_schema import relation_counts, reward_vector, semantic_features
from src.decision_modeling.state_table import aggregate_states, build_state, vector_moments


def prompt(family="trend"):
    cues = {
        "trend": {"family": "trend"},
        "cross_series": {"family": "cross_series", "edges": [[0, 1, 3], [1, 2, 5], [2, 3, 7]]},
        "duplicate_encoding": {"family": "duplicate_encoding", "known_index": 0, "known_value": 1},
    }
    return {
        "track": "N",
        "scene": {
            "truth_world": [1, 2, 3, 4],
            "observed_world": [9, 2, 3, 4],
            "operation": "sum4",
            "cue": cues[family],
        },
    }


@pytest.mark.parametrize(
    "raw",
    [
        "prefix [1,2,3,4]",
        "[1,2,3,4] suffix",
        "[1,2,3]",
        "[1,2,3,true]",
        "[1,2,3,100]",
        "[1,2,3,4.0]",
    ],
)
def test_strict_invalid_cannot_be_rescued(raw):
    f = semantic_features(raw, prompt())
    assert (f["event"], f["relation_score"], f["single_edit"], f["valid"], f["answer_correct"]) == (
        "I",
        0,
        0,
        0,
        0,
    )


def test_relation_reward_does_not_look_at_truth():
    p = prompt("cross_series")
    partial = semantic_features("[1,2,4,3]", p)
    assert (partial["event"], partial["relation_numerator"], partial["relation_denominator"]) == (
        "S",
        2,
        3,
    )
    assert partial["relation_score"] == 2 / 3
    assert partial["answer_correct"] == 1 and partial["event"] != "X"
    changed = {**p, "scene": {**p["scene"], "truth_world": [9, 8, 7, 6]}}
    assert semantic_features("[1,2,4,3]", changed)["relation_score"] == 2 / 3
    assert semantic_features("[9,2,3,4]", p)["copy_observation"] == 1


def test_relation_denominator_and_fraction_contracts():
    assert relation_counts([1, 2, 3, 9], {"family": "trend"}) == (1, 2)
    assert relation_counts(
        [1, 2, 3, 9], {"family": "duplicate_encoding", "known_index": 0, "known_value": 1}
    ) == (1, 1)
    with pytest.raises(ValueError, match="nonempty"):
        relation_counts(None, {"family": "cross_series", "edges": []})


def test_legacy_uses_actual_legacy_parser_and_no_invented_reward():
    p = {
        "track": "L",
        "scene": {
            "truth": [3, 8, 5, 18],
            "observation": [3, 8, 5, 17],
            "operation": {"operator": "sum", "indices": [0, 1]},
        },
    }
    f = semantic_features("v1=3,v2=8,v3=5,v4=18", p)
    assert f["event"] == "X" and f["parser_id"] == "C3.semantic_action_parser.parse_p1"
    assert f["relation_score"] is None and f["relation_status"] == "NOT_APPLICABLE_L"
    assert semantic_features("3,8,5,18\nexplanation", p)["event"] == "I"
    with pytest.raises(ValueError, match="not registered"):
        reward_vector(f)


def test_state_preserves_frequency_covariance_and_rational_values():
    x = semantic_features("[1,2,3,4]", prompt("cross_series"))
    w = semantic_features("[1,2,4,4]", prompt("cross_series"))
    s = build_state([x, x, x, w])
    assert s["n"] == 4 and s["Z0"]["pX"] == 0.75
    assert s["Z3"]["event_relation_counts"] == [
        {"event": "W", "numerator": 1, "denominator": 3, "count": 1},
        {"event": "X", "numerator": 3, "denominator": 3, "count": 3},
    ]
    assert s["Z2"]["reward_moments"]["sample_covariance"][0][3] > 0
    assert s["Z4"]["measurement"]["no_X_means_absent"] is False
    a = aggregate_states({"a": s, "b": build_state([w])}, {"a": 0.5, "b": 0.5})
    assert a["event_probabilities"]["X"] == 0.375
    with pytest.raises(ValueError):
        aggregate_states({"a": s}, {"a": 0.4})


def test_covariance_of_mean_and_constant_draws():
    m = vector_moments([[1, -1], [0, 0]])
    assert m["sample_covariance"] == [[0.5, -0.5], [-0.5, 0.5]]
    assert m["covariance_of_mean"] == [[0.25, -0.25], [-0.25, 0.25]]
    assert vector_moments([[0.1, 0.2]] * 9)["sample_covariance"] == [[0, 0], [0, 0]]


def test_core_key_keeps_input_and_termination_identity():
    g = {
        "input_hash": "input1",
        "token_ids": "[1,2]",
        "stop_reason": "eos",
        "max_new_tokens": "64",
        "eos_token_ids": "[2]",
    }
    score = {"inference_fingerprint": "fp", "probability_execution": "prefix"}
    assert score_key(score, g) != score_key(score, {**g, "input_hash": "input2"})
    assert score_key(score, g) != score_key(score, {**g, "token_ids": "[1,2,3]"})


def test_no_X_support_keeps_unknown_tail_and_qX_unidentified():
    w = semantic_features("[9,2,3,4]", prompt())
    b = _support([{**w, "sequence_logp": math.log(0.9)}])
    assert b["no_X_support"] and b["bounds"]["X"][1] == pytest.approx(0.1)
    assert b["qX"] is None and not b["numerical_error_bounded"]
    assert b["evidence_kind"] == "CONDITIONAL_ON_SCORER"


def test_no_prefix_or_post_EOS_atoms():
    base = {"token_ids": "[1]", "eos_token_ids": "[2]", "max_new_tokens": "64"}
    with pytest.raises(ValueError, match="prefix"):
        _check_action(base)
    with pytest.raises(ValueError, match="prefix"):
        _check_action({**base, "token_ids": "[2,1,2]"})


@pytest.mark.skipif(
    not os.environ.get("DECISION_D0_ARCHIVE"),
    reason="set DECISION_D0_ARCHIVE to the unchanged historical compact archive",
)
def test_actual_D0_archive(tmp_path):
    import pyarrow.parquet as pq

    out = tmp_path / "D0"
    s = reanalyze(os.environ["DECISION_D0_ARCHIVE"], out)
    assert (s["generation_count"], s["score_count"], s["unique_core_scoring_keys"]) == (
        2304,
        3072,
        446,
    )
    assert s["independent_diagnostics_max_error"] < 1e-12
    assert s["no_X_support_prompt_policy_count"] == 2
    assert pq.read_table(out / "state_table.parquet").num_rows == 2304
    states = json.loads((out / "states_Z0_Z4.json").read_text())
    assert sum(row["n"] for row in states) == 2304
    assert all(row["Z4"]["measurement"]["extra_output_observations"] == 0 for row in states)
    assert reanalyze(os.environ["DECISION_D0_ARCHIVE"], out) == s
    (out / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        reanalyze(os.environ["DECISION_D0_ARCHIVE"], out)
