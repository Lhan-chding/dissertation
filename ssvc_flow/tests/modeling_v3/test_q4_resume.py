"""Small, model-free Q4 invocation and retained-score recovery regressions."""

import copy
import json
from types import SimpleNamespace

import pytest

from src.modeling_v3 import vlm_campaign as c
from src.modeling_v3 import vlm_observation as v

TOLERANCES = {
    "mean_abs_token_logp": 1e-6,
    "max_abs_token_logp": 1e-6,
    "max_abs_sequence_logp": 2e-6,
}


def null_receipt(uuid="GPU-A", difference=0.0):
    return {
        "gpu_identity": uuid,
        "runtime_identity": {"model": "fixture", "source": "unchanged"},
        "tolerances": TOLERANCES,
        "scores": [
            {
                "proposal_sample_key": "fixed-null-action",
                "prompt_record_hash": "prompt",
                "input_audit": {"input_tensor_hash": "same-input"},
                "token_ids": [1, 99],
                "inference_fingerprint": "same-policy",
                "probability_execution": v.PATH,
                "max_new_tokens": 64,
                "eos_token_ids": [99],
                "token_logprobs": [-0.5 + difference, -0.25],
                "sequence_logp": -0.75 + difference,
            }
        ],
    }


def invocation(root, receipt):
    return c._record_q4_null_invocation(
        root, receipt, hardware={"allocated_gpu": {"uuid": receipt["gpu_identity"]}}
    )


def test_null_resume_appends_each_actual_gpu_and_preserves_anchor(tmp_path):
    first = invocation(tmp_path, null_receipt())
    anchor_bytes = (tmp_path / "null_receipt.json").read_bytes()
    second = invocation(tmp_path, null_receipt("GPU-B", 0.5e-6))
    third = invocation(tmp_path, null_receipt("GPU-B", 0.25e-6))
    assert (tmp_path / "null_receipt.json").read_bytes() == anchor_bytes
    assert first["null_anchor"] == second["null_anchor"] == third["null_anchor"]
    assert len({r["null_receipt"]["path"] for r in (first, second, third)}) == 3
    assert c._bound_json(second["null_receipt"])["gpu_identity"] == "GPU-B"
    assert c._bound_json(second["null_comparison"])["status"] == "PASS"
    assert c._bound_json(third["null_comparison"])["status"] == "PASS"
    # The worker result must retain this invocation, not silently select GPU-A.
    c._complete(tmp_path, {"status": "fixture-complete", **second})
    saved = c._read(tmp_path / "result.json")
    assert c._bound_json(saved["null_receipt"])["gpu_identity"] == "GPU-B"


@pytest.mark.parametrize("change", ["threshold", "policy", "action", "runtime", "tolerance"])
def test_null_resume_rejects_changed_identity_or_excess_error_and_retains_attempt(tmp_path, change):
    invocation(tmp_path, null_receipt())
    original = (tmp_path / "null_receipt.json").read_bytes()
    receipt = null_receipt("GPU-B")
    if change == "threshold":
        receipt["scores"][0]["token_logprobs"][0] -= 0.1
    elif change == "policy":
        receipt["scores"][0]["inference_fingerprint"] = "changed-policy"
    elif change == "action":
        receipt["scores"][0]["token_ids"][0] = 2
    elif change == "runtime":
        receipt["runtime_identity"]["source"] = "changed-source"
    else:
        receipt["tolerances"] = {**TOLERANCES, "max_abs_token_logp": 1.0}
    with pytest.raises(ValueError):
        invocation(tmp_path, receipt)
    assert (tmp_path / "null_receipt.json").read_bytes() == original
    assert len(list((tmp_path / "null_invocations").glob("*/null_receipt.json"))) == 2
    statuses = [
        c._read(p)["status"] for p in (tmp_path / "null_invocations").glob("*/comparison.json")
    ]
    assert any(status != "PASS" for status in statuses)


def test_null_anchor_tampering_is_rejected(tmp_path):
    invocation(tmp_path, null_receipt())
    (tmp_path / "null_receipt.json").write_text(json.dumps(null_receipt("forged")))
    with pytest.raises(ValueError, match="missing or changed"):
        invocation(tmp_path, null_receipt("GPU-B"))


def test_same_gpu_comparison_keeps_full_checks_but_cannot_satisfy_cross_gpu_gate():
    left = null_receipt()
    right = null_receipt(difference=0.5e-6)
    assert v.compare_probability_receipts(left, right, tolerances=TOLERANCES)["status"] == "PASS"
    with pytest.raises(ValueError, match="distinct"):
        v.compare_cross_gpu_receipts(left, right, tolerances=TOLERANCES)
    right["scores"][0]["input_audit"] = {"input_tensor_hash": "changed"}
    with pytest.raises(ValueError, match="identity"):
        v.compare_probability_receipts(left, right, tolerances=TOLERANCES)


class KnownBackend:
    def __init__(self):
        self.runtime_identity = {"model": "fixture"}
        self.adapter = SimpleNamespace(
            eos_ids={99},
            processor=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a, **k: [1])),
        )
        self.calls = 0

    def activate(self, policy):
        self.policy = policy

    def score_known_actions(self, prompt, actions, *, max_new_tokens):
        self.calls += 1
        return [
            {
                "token_ids": tokens,
                "token_logprobs": [-0.5, -0.25],
                "sequence_logp": -0.75,
                "prompt_id": prompt["prompt_id"],
                "prompt_record_hash": v.digest(prompt),
                "inference_fingerprint": self.policy["inference_fingerprint"],
                "runtime_identity": self.runtime_identity,
                "probability_execution": v.PATH,
                "eos_token_ids": [99],
                "max_new_tokens": max_new_tokens,
                "category": "X",
                "is_sampling_draw": False,
                "action_source": "PREDECLARED_KNOWN_ACTION",
            }
            for tokens in actions
        ]


def known_inputs():
    return (
        {"origin": {"inference_fingerprint": "fixed-policy", "checkpoint": {"sha256": "fixed"}}},
        [{"prompt_id": "p0", "scene": {"truth_world": {"x": 1}}}],
    )


def test_completed_known_scores_resume_without_new_scoring_and_bind_requests(tmp_path):
    backend = KnownBackend()
    policies, probes = known_inputs()
    first = c._q4_known_event_scores(tmp_path, backend, policies, probes)
    original = (tmp_path / "known_event_scores.json").read_bytes()
    resumed = c._q4_known_event_scores(tmp_path, backend, policies, probes)
    assert first == resumed and backend.calls == 1
    assert (tmp_path / "known_event_scores.json").read_bytes() == original
    changed = copy.deepcopy(policies)
    changed["origin"]["checkpoint"]["sha256"] = "different"
    with pytest.raises(ValueError, match="request"):
        c._q4_known_event_scores(tmp_path, backend, changed, probes)
    assert backend.calls == 1


@pytest.mark.parametrize("mutation", ["bytes", "missing_receipt", "prompt", "policy"])
def test_known_score_reuse_rejects_missing_binding_tamper_and_changed_identity(tmp_path, mutation):
    backend = KnownBackend()
    policies, probes = known_inputs()
    c._q4_known_event_scores(tmp_path, backend, policies, probes)
    if mutation == "bytes":
        with (tmp_path / "known_event_scores.json").open("a") as stream:
            stream.write(" ")
    elif mutation == "missing_receipt":
        (tmp_path / "known_event_scores_receipt.json").unlink()
    elif mutation == "prompt":
        probes[0]["scene"]["truth_world"]["x"] = 2
    else:
        policies["origin"]["inference_fingerprint"] = "changed"
    with pytest.raises(ValueError):
        c._q4_known_event_scores(tmp_path, backend, policies, probes)
    assert backend.calls == 1
