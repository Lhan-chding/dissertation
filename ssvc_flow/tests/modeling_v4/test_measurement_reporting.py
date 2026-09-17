"""Regression tests for finite-sample uncertainty and immutable analysis provenance."""

import json
import math

import numpy as np
import pytest

from src.modeling_v3.io import canonical_hash, sha256_file
from src.modeling_v3.observation_geometry import ContributionBatch
from src.modeling_v4.gpu_collect import pair_observation_diagnostics
from src.modeling_v4.measurement_reporting import (
    count_difference_intervals,
    stable_joint_covariance,
)
from src.modeling_v4.observations import observe_packet
from src.modeling_v4.response_fit import _reference_statistics


def test_identical_floating_draws_have_exactly_zero_joint_covariance_and_no_precision():
    values = np.full((64, 2, 4), 0.1)
    batch = ContributionBatch(
        values,
        tuple(map(str, range(64))),
        "rng",
        "p",
        ("a", "b"),
        (("l", "r"), ("l", "r")),
        ((1,),) * 64,
        "origin",
    )
    assert np.count_nonzero(stable_joint_covariance(values)) == 0
    report = observe_packet(batch, bootstrap_repetitions=12)
    for result in report["methods"].values():
        assert np.count_nonzero(result["covariance_of_mean"]) == 0
        assert not result["uncertainty_limits"]["zero_empirical_variance_is_precision_proof"]
        assert result["diagnostics"]["raw_mass_standard_error"] == [0, 0]
    reference = _reference_statistics(batch)
    assert not reference["resolved"].any()
    assert np.count_nonzero(reference["covariance_of_mean"]) == 0
    assert np.count_nonzero(reference["se"]) == 0


def test_shared_covariance_retained_and_bootstrap_estimates_not_divided_again():
    rng = np.random.default_rng(47)
    base = rng.normal(size=(32, 1, 4))
    values = np.concatenate((base, -base), axis=1)
    batch = ContributionBatch(
        values,
        tuple(map(str, range(32))),
        "rng",
        "p",
        ("a", "b"),
        (("l", "r"), ("r", "l")),
        ((1,),) * 32,
        "origin",
    )
    report = observe_packet(batch, bootstrap_repetitions=20)
    raw = report["methods"]["RAW4"]["covariance_of_mean"]
    np.testing.assert_allclose(raw[:4, 4:], -raw[:4, :4])
    cross = report["methods"]["CROSSFIT_COV_ZERO_SUM"]
    expected = stable_joint_covariance(cross["uncertainty"]["replicate_estimates"])
    np.testing.assert_array_equal(cross["covariance_of_mean"], expected)
    assert cross["uncertainty"]["refits_per_replicate"] == 2


def test_cp_never_collapses_at_zero_or_saturated_counts():
    values = np.repeat([[1, 0, 0, 0]], 64, axis=0)
    interval = count_difference_intervals(values, values)
    np.testing.assert_allclose(interval[:, 0], -(1 - (0.05 / 4) ** (1 / 64)), atol=1e-12)
    np.testing.assert_allclose(interval[:, 1], 1 - (0.05 / 4) ** (1 / 64), atol=1e-12)
    family = count_difference_intervals(values, values, alpha=0.05 / 96)
    assert np.all(family[:, 0] < interval[:, 0])
    with pytest.raises(ValueError, match="one-hot"):
        count_difference_intervals(np.zeros((2, 4)), values)


def response_fixture(vary=False, direct=True):
    policies = {"left": {"inference_fingerprint": "L"}, "right": {"inference_fingerprint": "R"}}
    packets, mapping = {}, {}
    for role in ("reference", "work", "direct_work", "mixture", "left", "right"):
        packet, scores = [], {"left": [], "right": []}
        for i in range(64):
            category = "W" if vary and i % 2 else "X"
            token = 8 if category == "W" else 7
            row = {
                "sample_key": f"{role}:{i}",
                "prompt_id": "p",
                "input_hash": "input",
                "shared_token_identity": str(token),
                "token_ids": [token, 99],
                "eos_seen": True,
                "max_new_tokens": 64,
                "category": category,
                "event_onehot": [1, 0, 0, 0] if category == "X" else [0, 0, 1, 0],
                "probability_execution": "uncached_prefix_recompute",
                "generation_parity": {"passed": True},
                "generation_sequence_logp": math.log(0.45 if vary else 0.9),
            }
            packet.append(row)
            for endpoint in scores:
                probability = (0.45 if vary else 0.9) + (1e-5 if endpoint == "left" else 0)
                scores[endpoint].append(
                    {
                        **row,
                        "sequence_logp": math.log(probability),
                        "inference_fingerprint": policies[endpoint]["inference_fingerprint"],
                    }
                )
        packets[role] = {"rows": packet, "identity": {"prompt_ids": ["p"]}}
        mapping[role] = {k: {"rows": v} for k, v in scores.items()}
    for key, policy in policies.items():
        policy["candidate_id"] = key
    runtime = {"execution_kind": "CPU_FIXTURE"}
    for role, receipt in packets.items():
        proposal_ids = (
            ["left", "right"] if role == "mixture" else [role] if role in policies else []
        )
        proposal_policies = [policies[p] for p in proposal_ids] or [
            {"inference_fingerprint": "origin", "candidate_id": "origin"}
        ]
        receipt["identity"].update(
            {
                "proposal": "MIX"
                if role == "mixture"
                else "DIRECT"
                if role in policies
                else "ORIGIN",
                "origin_id": "o",
                "draws": 64,
                "rng_namespace": role,
                "role": role,
                "runtime": runtime,
                "policies": proposal_policies,
            }
        )
        receipt.update({"count": 64, "chunks": []})
        for i, row in enumerate(receipt["rows"]):
            sample_key = canonical_hash([role, "p", i])
            selected = (
                int(canonical_hash([sample_key, "mixture_source"])[0], 16) % 2
                if role == "mixture"
                else 0
            )
            row.update(
                {
                    "sample_key": sample_key,
                    "sample_id": sample_key,
                    "sample_seed": int(sample_key[:16], 16) % (2**63),
                    "origin_id": "o",
                    "proposal": receipt["identity"]["proposal"],
                    "proposal_policy_id": proposal_policies[selected]["candidate_id"],
                    "draw_index": i,
                    "rng_namespace": role,
                    "role": role,
                    "runtime_identity": canonical_hash(runtime),
                    "eos_token_ids": [99],
                    "truncated": False,
                    "completion_length": 2,
                    "stop_reason": "eos",
                    "raw_completion": "fixture",
                    "proposal_fingerprint": proposal_policies[selected]["inference_fingerprint"],
                    "shared_token_identity": canonical_hash(row["token_ids"]),
                    "behavior_token_logprobs": [row["generation_sequence_logp"] / 2] * 2,
                }
            )
            for score_receipt in mapping[role].values():
                score = score_receipt["rows"][i]
                score.update(
                    {
                        k: row[k]
                        for k in (
                            "sample_key",
                            "sample_id",
                            "shared_token_identity",
                            "role",
                            "rng_namespace",
                            "runtime_identity",
                        )
                    }
                )
                score["token_logprobs"] = [score["sequence_logp"] / 2] * 2
        for endpoint, score_receipt in mapping[role].items():
            score_receipt["identity"] = {
                "policy": policies[endpoint],
                "runtime": runtime,
                "sample_identity": canonical_hash(receipt["identity"]),
                "sample_chunks": receipt["chunks"],
            }
    check = {
        "bank_id": "calibration_001",
        "left_candidate_id": "left",
        "right_candidate_id": "right",
        "mixture": packets["mixture"],
        "mixture_scores": mapping["mixture"],
        "direct": {k: packets[k] for k in ("left", "right")} if direct else {},
    }
    response = {
        "policies": policies,
        "pair_checks": [check],
        **{k: packets[k] for k in ("reference", "work", "direct_work")},
    }
    for role, key in (
        ("reference", "reference_scores"),
        ("work", "work_scores"),
        ("direct_work", "direct_scores"),
    ):
        response[key] = {k: {"score_receipt": v} for k, v in mapping[role].items()}
    return response


def test_pair_diagnostics_use_real_resolution_and_support_without_model_calls():
    result = pair_observation_diagnostics(response_fixture(), row_reader=lambda r: iter(r["rows"]))
    unit = result["units"][0]
    assert result["all_finite"]
    assert unit["reference_precision"] == "UNRESOLVED"
    assert unit["variance_of_mean"]["ORIGIN"] == [0] * 4
    assert unit["comparisons"]["DIRECT"]["within_diagnostic_interval"] == [None] * 4
    assert unit["endpoint_count_intervals"]["single_event"][0][1] > 0.06
    assert unit["conditional_support_bounds"]["valid_conditional_bound"]
    assert not unit["conditional_support_bounds"]["numerical_systematic_error_bounded"]
    assert unit["conditional_support_bounds"]["model_calls"] == 0
    varied = pair_observation_diagnostics(
        response_fixture(True, False), row_reader=lambda r: iter(r["rows"])
    )
    assert varied["units"][0]["reference_precision"] == "DESCRIPTIVE_PARTIALLY_RESOLVED"
    assert varied["units"][0]["endpoint_count_intervals"] is None
    assert varied["scientific_status"] == "NOT_CERTIFIED"


def test_optional_support_invalidates_identity_disagreement():
    response = response_fixture()
    response["work_scores"]["left"]["score_receipt"]["rows"][0]["input_hash"] = "different"
    result = pair_observation_diagnostics(response, row_reader=lambda r: iter(r["rows"]))
    support = result["units"][0]["conditional_support_bounds"]
    assert not support["valid_conditional_bound"]
    assert support["status"] == "UNAVAILABLE_INVALID_OR_INCOMPLETE_RECORDS"


def test_separate_analysis_provenance_preserves_old_receipts_and_source_guards(
    tmp_path, monkeypatch
):
    from src.modeling_v4 import measurement_report as module

    collection = {"sha256": "old", "files": {"src/a.py": "h"}}
    analysis = {"sha256": "new", "files": {"src/a.py": "new-h"}}
    monkeypatch.setattr(module, "_require_server_cpu", lambda: None)
    monkeypatch.setattr(
        module, "source_identity", lambda root=None: collection if root else analysis
    )
    monkeypatch.setattr(
        module, "pair_observation_diagnostics", lambda *a, **k: {"all_finite": True}
    )
    campaign = tmp_path / "campaign"
    task = {"id": "C_seed_step", "kind": "map"}
    config = {"test": "fixture"}
    tdir = campaign / "tasks" / task["id"]
    tdir.mkdir(parents=True)
    response = tdir / "response.json"
    response.write_text(json.dumps({"origin_id": task["id"]}))
    receipt = {
        "status": "COMPLETED",
        "task": task,
        "execution_kind": "REAL_CUDA_MODEL",
        "source_hash": "old",
        "config_hash": canonical_hash(config),
        "origin_id": task["id"],
        "response": {"path": str(response), "sha256": sha256_file(response)},
    }
    complete = tdir / "COMPLETE.json"
    complete.write_text(json.dumps(receipt))
    old_bytes = complete.read_bytes()
    plan = {"root": str(campaign), "tasks": [task], "source": collection, "config": config}
    plan["analysis_rules"] = module.load_analysis_rules()
    plan["analysis_rules_hash"] = canonical_hash(plan["analysis_rules"])
    plan["task_list_hash"] = canonical_hash(plan)
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(plan))
    out = tmp_path / "analysis"
    result = module.analyze_maps(path, task_ids=[task["id"]], collection_code="frozen", out=out)
    assert result["collection_source"] == collection and result["analysis_source"] == analysis
    assert not result["source_guards_bypassed"]
    assert complete.read_bytes() == old_bytes
    with pytest.raises(ValueError, match="empty"):
        module.analyze_maps(path, task_ids=[task["id"]], collection_code="frozen", out=out)
    monkeypatch.setattr(module, "source_identity", lambda root=None: analysis)
    with pytest.raises(ValueError, match="collection source"):
        module.analyze_maps(
            path, task_ids=[task["id"]], collection_code="frozen", out=tmp_path / "other"
        )


def test_analysis_rejects_changed_registered_precision_rules(tmp_path, monkeypatch):
    from src.modeling_v4 import measurement_report as module

    monkeypatch.setattr(module, "_require_server_cpu", lambda: None)
    rules = module.load_analysis_rules()
    rules["reference_precision"]["probability_scales"] = [0.1]
    plan = {"analysis_rules": rules, "analysis_rules_hash": canonical_hash(rules)}
    plan["task_list_hash"] = canonical_hash(plan)
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="analysis rules"):
        module.analyze_maps(
            path, task_ids=["B_origin_0"], collection_code="unused", out=tmp_path / "out"
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_score",
        "score_input",
        "score_tokens",
        "wrong_event",
        "missing_draw",
        "reused_stream",
        "wrong_count_endpoint",
        "reversed_mixture",
        "wrong_mixture_source",
        "wrong_seed",
    ],
)
def test_primary_identity_errors_cannot_produce_measured_labels(mutation):
    response = response_fixture()
    if mutation == "duplicate_score":
        scores = response["reference_scores"]["left"]["score_receipt"]["rows"]
        scores.append(dict(scores[0]))
    elif mutation == "score_input":
        response["reference_scores"]["left"]["score_receipt"]["rows"][0]["input_hash"] = "bad"
    elif mutation == "score_tokens":
        response["pair_checks"][0]["mixture_scores"]["left"]["rows"][0]["shared_token_identity"] = (
            "bad"
        )
    elif mutation == "wrong_event":
        response["reference"]["rows"][0]["event_onehot"] = [0, 1, 0, 0]
    elif mutation == "missing_draw":
        response["reference"]["rows"].pop()
    elif mutation == "reused_stream":
        response["pair_checks"][0]["direct"]["left"]["identity"]["rng_namespace"] = "reference"
    elif mutation == "wrong_count_endpoint":
        response["pair_checks"][0]["direct"]["left"]["identity"]["policies"] = [
            response["policies"]["right"]
        ]
    elif mutation == "reversed_mixture":
        response["pair_checks"][0]["mixture"]["identity"]["policies"].reverse()
    elif mutation == "wrong_mixture_source":
        response["pair_checks"][0]["mixture"]["rows"][0]["proposal_fingerprint"] = "wrong"
    elif mutation == "wrong_seed":
        response["reference"]["rows"][0]["sample_seed"] += 1
    with pytest.raises(ValueError):
        pair_observation_diagnostics(response, row_reader=lambda r: iter(r["rows"]))


def test_optional_raw_token_mutation_invalidates_support():
    response = response_fixture()
    response["work"]["rows"][0]["token_ids"] = [9, 99]
    result = pair_observation_diagnostics(response, row_reader=lambda r: iter(r["rows"]))
    assert not result["units"][0]["conditional_support_bounds"]["valid_conditional_bound"]


def test_reader_bounds_cache_and_rejects_mutation_even_on_cached_hit(tmp_path, monkeypatch):
    from src.modeling_v4 import measurement_report as module

    root = tmp_path / "campaign"
    (root / "tasks").mkdir(parents=True)
    reads = []

    def read(path):
        reads.append(path)
        return [{"sample_id": path}]

    monkeypatch.setattr(module, "_chunk_rows", read)
    reader = module.VerifiedReceiptReader(root)
    receipts = []
    for i in range(3):
        path = root / "tasks" / f"{i}.parquet"
        path.write_text(f"fixture {i}")
        receipts.append(
            {
                "count": 1,
                "chunks": [{"path": str(path), "start": 0, "stop": 1, "sha256": sha256_file(path)}],
            }
        )
        list(reader.rows(receipts[-1]))
    assert len(reader.cache) == 2
    list(reader.rows(receipts[-1]))
    assert len(reads) == 3
    list(reader.rows(receipts[0]))
    assert len(reads) == 4 and len(reader.cache) == 2
    path = root / "tasks" / "0.parquet"
    path.write_text("modified original")
    with pytest.raises(ValueError, match="artifact changed"):
        list(reader.rows(receipts[0]))
