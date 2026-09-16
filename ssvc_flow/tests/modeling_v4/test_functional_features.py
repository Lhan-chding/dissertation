import copy

import numpy as np
import pytest

from src.modeling_v4.functional_features import (
    DIAGNOSTIC_NAMES,
    aggregate_diagnostics,
    build_signature,
    build_state_features,
    output_diagnostics,
)

SCENE = {
    "truth_world": [1, 2, 3, 4],
    "observed_world": [1, 2, 3, 9],
    "operation": "sum4",
    "cue": {"family": "trend"},
}


def test_illegal_outputs_are_unconditional_zero_and_conditional_is_separate():
    invalid = output_diagnostics("prefix [1,2,3,4]", SCENE)
    assert invalid["category"] == "I"
    assert all(invalid[n] == 0 for n in DIAGNOSTIC_NAMES)
    assert invalid["constraint_fraction_conditional"] is None
    result = aggregate_diagnostics(["[1,2,3,4]", "bad"], SCENE)
    np.testing.assert_array_equal(result["diagnostics"], np.full(7, 0.5) * [1, 1, 1, 1, 1, 1, 0])
    assert result["constraint_fraction_conditional"] == 1
    assert result["valid_count"] == 1 and result["count"] == 2


def test_constraints_are_relation_fraction_not_all_or_none():
    result = output_diagnostics("[1,2,3,8]", SCENE)
    assert result["constraint_fraction"] == 0.5
    assert result["edit_exactly_one"] == 1
    scene = {**SCENE, "cue": {"family": "cross_series", "edges": [[0, 1, 3], [2, 3, 7]]}}
    assert output_diagnostics("[1,2,3,8]", scene)["constraint_fraction"] == 0.5
    scene = {**SCENE, "cue": {"family": "duplicate_encoding", "known_index": 2, "known_value": 3}}
    assert output_diagnostics("[1,2,3,9]", scene)["copy_observation"] == 1
    assert aggregate_diagnostics(["bad"], scene)["constraint_fraction_conditional"] is None
    with pytest.raises(ValueError):
        output_diagnostics("bad", {**SCENE, "cue": {"family": "unknown"}})


def test_origin_features_reject_endpoint_leakage_and_keep_raw_diagnostics():
    state = {
        "stage": "ORIGIN",
        "origin_id": "o",
        "panel_role": "signature",
        "event_probabilities": [[0.1, 0.2, 0.3, 0.4]],
        "diagnostic_names": list(DIAGNOSTIC_NAMES),
        "diagnostics": [[0.1] * 7],
    }
    assert build_state_features(state, origin_id="o", mode="EVENT_ONLY").shape == (1, 4)
    assert build_state_features(state, origin_id="o", mode="EVENT_PLUS_DIAGNOSTICS").shape == (
        1,
        11,
    )
    for extra in [
        {"stage": "CANDIDATE"},
        {"origin_id": "other"},
        {"endpoint_diagnostics": [0] * 7},
    ]:
        with pytest.raises(ValueError):
            build_state_features({**state, **extra}, origin_id="o", mode="EVENT_PLUS_DIAGNOSTICS")


def signature_fixture():
    prompts = [f"p{i}" for i in range(36)]
    samples = []
    for i, p in enumerate(prompts):
        for d in range(16):
            samples.append(
                {
                    "sample_id": f"{p}:{d}",
                    "origin_id": "o",
                    "prompt_id": p,
                    "base_scene_id": f"s{i // 2}",
                    "role": "signature",
                    "rng_namespace": "sig",
                    "draw_index": d,
                    "token_ids": [3, 2],
                    "action_mask": [1, 1],
                    "stop_reason": "eos",
                    "eos_seen": True,
                    "truncated": False,
                    "max_new_tokens": 64,
                    "input_hash": p,
                    "runtime_identity": "normalized-policy-v1",
                }
            )

    def scored(fp, scale):
        return [
            {
                **{
                    k: r[k]
                    for k in [
                        "sample_id",
                        "prompt_id",
                        "token_ids",
                        "input_hash",
                        "role",
                        "rng_namespace",
                        "runtime_identity",
                    ]
                },
                "candidate_id": fp,
                "inference_fingerprint": fp,
                "token_logprobs": [-scale, -2 * scale],
                "sequence_logp": -3 * scale,
            }
            for r in samples
        ]

    panel = {
        "panel_id": "panel",
        "runtime_identity": "normalized-policy-v1",
        "origin_id": "o",
        "prompt_ids": prompts,
        "eos_token_ids": [2],
        "splits": {
            r: {"base_scene_ids": [r], "prompt_ids": [r], "sample_ids": [], "rng_namespaces": [r]}
            for r in ["train", "observation", "reference"]
        },
    }
    panel["splits"]["signature"] = {
        "base_scene_ids": [f"s{i}" for i in range(18)],
        "prompt_ids": prompts,
        "sample_ids": [r["sample_id"] for r in samples],
        "rng_namespaces": ["sig"],
    }
    return samples, scored("b", 1), scored("u", 2), panel


def test_signature_preserves_native_576_full_sequence_scores_and_identity():
    samples, b, u, panel = signature_fixture()
    result = build_signature(b[::-1], u, sample_rows=samples[::-1], panel_manifest=panel)
    np.testing.assert_array_equal(result["values"], np.full(576, -3.0))
    assert result["provenance"]["native_width"] == 576
    assert result["provenance"]["ordered_sample_ids"][:2] == ["p0:0", "p0:1"]
    assert result["provenance"]["target_outcomes_used"] is False
    assert result["provenance"]["paid_score_actions"] is None


def test_signature_joins_actual_shared_token_hash_without_duplicate_score_tokens():
    from src.modeling_v3.io import canonical_hash

    samples, baseline, candidate, panel = signature_fixture()
    for row in baseline + candidate:
        row["shared_token_identity"] = canonical_hash(row.pop("token_ids"))
    result = build_signature(baseline, candidate, sample_rows=samples, panel_manifest=panel)
    np.testing.assert_array_equal(result["values"], np.full(576, -3.0))
    candidate[0]["shared_token_identity"] = "0" * 64
    with pytest.raises(ValueError, match="token"):
        build_signature(baseline, candidate, sample_rows=samples, panel_manifest=panel)


@pytest.mark.parametrize(
    "corruption", ["length_average", "tokens", "eos", "rng", "scene", "duplicate"]
)
def test_signature_rejects_scoring_or_split_mismatch(corruption):
    samples, b, u, panel = signature_fixture()
    if corruption == "length_average":
        u[0]["sequence_logp"] /= 2
    if corruption == "tokens":
        u[0]["token_ids"] = [4, 2]
    if corruption == "eos":
        samples[0]["token_ids"] = [3, 4]
    if corruption == "rng":
        panel["splits"]["reference"]["rng_namespaces"] = ["sig"]
    if corruption == "scene":
        panel["splits"]["observation"]["base_scene_ids"] = ["s0"]
    if corruption == "duplicate":
        samples[-1] = copy.deepcopy(samples[0])
    with pytest.raises(ValueError):
        build_signature(b, u, sample_rows=samples, panel_manifest=panel)
