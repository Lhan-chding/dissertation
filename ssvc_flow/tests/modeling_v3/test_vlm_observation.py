"""Small model-free fixtures for real observation execution and failure recovery."""

import copy
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from src.modeling_v3 import vlm_observation as v

TOLERANCE = {"mean_abs_token_logp": 1e-6, "max_abs_token_logp": 1e-6, "max_abs_sequence_logp": 2e-6}


def test_reference_overflowing_second_moments_remain_serializable_unresolved():
    raw = np.tile([1e200, -1e200, 0.0, 0.0], (4096, 1))
    raw[::2] *= -1
    result = v.reference_precision(
        raw, cumulative_draws=4096, protocol={"family_cells": 1}, proposal="ORIGIN"
    )
    assert result["status"] == "REFERENCE_REQUIRES_INDEPENDENT_MIX"
    assert not result["finite_reference_moments"] and result["standard_error"] is None
    json.dumps(result, allow_nan=False)


class Tokenizer:
    def decode(self, tokens, **kwargs):
        return " ".join(str(t) for t in tokens)


class Adapter:
    audit: ClassVar = {"probability_execution": v.PATH}
    eos_ids: ClassVar = {99}
    pad_id = 0

    def __init__(self):
        self.processor = SimpleNamespace(tokenizer=Tokenizer())
        self.forward_calls, self.shift, self.calls = 0, 0, 0
        self.fail_after = None

    def prepare(self, prompt, data_root):
        return {
            "audit": {
                "enable_thinking": False,
                "image_token_count": 0,
                "pixel_values_hash": None,
                "final_prompt_hash": v.digest(prompt),
                "input_tensor_hash": v.digest([prompt, "tensors"]),
            }
        }

    def generate(self, prepared, *, seed, max_new_tokens, do_sample):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("injected OOM")
        self.forward_calls += 2
        return {
            "token_ids": [seed % 3, 99],
            "completion_length": 2,
            "stop_reason": "eos",
            "raw_completion": str(seed % 3),
            "behavior_token_logprobs": [-0.5 - self.shift, -0.25],
        }

    def logprobs(self, prepared, tokens, *, require_grad):
        assert not require_grad
        self.forward_calls += len(tokens)
        return [-0.5 - self.shift, -0.25]


def backend():
    adapter, loads = Adapter(), []

    def activate(policy):
        loads.append(policy["inference_fingerprint"])
        adapter.shift = 0.1 if policy["inference_fingerprint"] == "B" else 0
        return policy["inference_fingerprint"]

    return v.PrefixObservationBackend(
        adapter,
        runtime_identity={"code": "frozen", "model": "fixture"},
        policy_loader=activate,
        annotator=lambda text, scene: {"category": "X" if text == "0" else "I"},
        parity_tolerances=TOLERANCE,
    ), loads


def prompt():
    return {
        "prompt_id": "prompt1",
        "prompt": {"user": "fixture"},
        "scene": {},
        "interface": "SYMBOLIC_FRESH",
    }


def action(tokens, *, stop="eos", raw=None):
    return {
        "token_ids": tokens,
        "completion_length": len(tokens),
        "stop_reason": stop,
        "behavior_token_logprobs": [-1.0] * len(tokens),
        "raw_completion": raw if raw is not None else Tokenizer().decode(tokens[:-1]),
    }


def binding(path):
    return {"path": str(path), "sha256": v.file_digest(path)}


def manifest(
    tmp_path, *, operation="generate", sample_files=(), draws=4, role="pilot", proposal="ORIGIN"
):
    tmp_path.mkdir(exist_ok=True)
    prompts, state = tmp_path / "prompts.json", tmp_path / "state.bin"
    if not prompts.exists():
        prompts.write_text(json.dumps([prompt()]))
    if not state.exists():
        state.write_bytes(b"fixture checkpoint")
    policies = {k: {"inference_fingerprint": k, "checkpoint": binding(state)} for k in ("A", "B")}
    task = v.freeze_observation_task(
        operation=operation,
        origin_id="origin1",
        candidate_id="B",
        prompt_file=binding(prompts),
        prompt_ids=["prompt1"],
        role=role,
        draw_start=0,
        draw_stop=draws,
        rng_namespace="development-1",
        proposal=proposal,
        proposal_candidates=["A", "B"] if proposal == "MIX" else ["A"],
        sample_files=sample_files,
        worker=0,
    )
    return v.freeze_task_manifest([task], policies, {"code": "frozen", "model": "fixture"})


def test_import_does_not_load_model_runtime():
    code = (
        "import sys; import src.modeling_v3.vlm_observation; "
        "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_eos_truncation_sampled_pad_and_invalid_parse_retained():
    assert (
        v.validate_action(action([0, 99]), eos_ids={99}, tokenizer=Tokenizer())["padding_positions"]
        == []
    )
    truncated = action([4] * 64, stop="length", raw=Tokenizer().decode([4] * 64))
    assert v.validate_action(truncated, eos_ids={99}, tokenizer=Tokenizer())["truncated"]
    for malformed, message in [
        (action([99, 2, 99]), "first EOS"),
        (action([3], stop="length"), "full truncation"),
        (action([2, 99], raw="wrong"), "decode"),
    ]:
        with pytest.raises(ValueError, match=message):
            v.validate_action(malformed, eos_ids={99}, tokenizer=Tokenizer())
    b, _ = backend()
    b.activate({"inference_fingerprint": "A"})
    sample = b.generate(prompt(), seed=1)
    assert sample["category"] == "I" and sample["event_onehot"] == [0, 0, 0, 1]
    assert sample["token_ids"][-1] == 99 and sample["generation_sequence_logp"] == -0.75
    assert b.counters["forward_calls"] == 4
    assert b.counters["generated_tokens"] == b.counters["scored_tokens"] == 2


def test_tolerance_required_before_generation_and_identity_check():
    with pytest.raises(ValueError, match="frozen"):
        v.PrefixObservationBackend(Adapter(), runtime_identity={}, policy_loader=lambda p: "A")
    b, loads = backend()
    policy = {"inference_fingerprint": "A"}
    b.activate(policy)
    b.activate(policy)
    assert loads == ["A"]
    sample = {**b.generate(prompt(), seed=0), "sample_key": "draw"}
    changed = copy.deepcopy(prompt())
    changed["prompt"]["user"] = "changed"
    with pytest.raises(ValueError, match="prompt identity"):
        b.score(changed, sample)
    b.policy_loader = lambda p: "incorrect"
    with pytest.raises(ValueError, match="fingerprint"):
        b.activate({"inference_fingerprint": "B"})


def test_rng_roles_disjoint_and_mix_iid_not_balanced_halves():
    rows = {
        role: [
            {
                **v.request_identity(
                    origin_id="o",
                    prompt_id="p",
                    role=role,
                    draw_index=i,
                    rng_namespace="frozen",
                    proposal="MIX",
                    proposal_candidates=["A", "B"],
                ),
                "role": role,
            }
            for i in range(24)
        ]
        for role in ("pilot", "main", "reference")
    }
    assert v.validate_role_independence(**rows)["status"] == "PASS"
    coins = [r["proposal_source_index"] for r in rows["main"]]
    assert any(a == b for a, b in itertools.pairwise(coins))
    assert len(set(coins[:12])) == len(set(coins[12:])) == 2
    overlap = copy.deepcopy(rows)
    overlap["reference"][0]["sample_seed"] = overlap["main"][0]["sample_seed"]
    with pytest.raises(ValueError, match="overlap"):
        v.validate_role_independence(**overlap)


def test_mix_raw_matches_probability_difference_extremes_and_origin_not_clipped():
    a, b = np.log([0.1, 0.2, 0.3, 0.4]), np.log([0.2, 0.1, 0.5, 0.2])
    q = np.logaddexp(a, b) - math.log(2)
    raw, diagnostics = v.signed_contributions(a, b, q, list("XSWI"), proposal="MIX")
    np.testing.assert_allclose(
        raw.sum(axis=1), 2 * (np.exp(b) - np.exp(a)) / (np.exp(a) + np.exp(b))
    )
    assert diagnostics["absolute_weight_bound"] == 2
    a, b = np.array([-1e6, -1, -2]), np.array([-1, -1e6, -2])
    raw, _ = v.signed_contributions(
        a, b, np.logaddexp(a, b) - math.log(2), ["X"] * 3, proposal="MIX"
    )
    np.testing.assert_array_equal(raw[:, 0], [2, -2, 0])
    raw, diag = v.signed_contributions([-1000], [-0.1], [-1000], ["I"], proposal="ORIGIN")
    assert (
        not np.isfinite(raw[0, 3])
        and diag["requires_independent_mix"]
        and not diag["weights_clipped"]
    )
    with pytest.raises(ValueError, match="denominator"):
        v.signed_contributions([-0.1], [-0.2], [-0.7], ["X"], proposal="MIX")


def test_reference_multilook_predeclared_empirical_and_formal_distinct():
    protocol, zero = {"family_cells": 30 * 36 * 2, "alpha": 0.05}, np.zeros((4096, 4))
    result = v.reference_precision(zero, cumulative_draws=4096, protocol=protocol, proposal="MIX")
    assert result["status"] == "REFERENCE_EXTEND"
    assert result["formal_hoeffding_half_width"] > 0.01
    assert not result["formal_precision_met"] and not result["reference_is_exact_truth"]
    assert not result["empirical_precision_met"]
    assert result["zero_variance_event_indices"] == [0, 1, 2, 3]
    assert result["alpha_per_interval"] == 0.05 / (5 * 30 * 36 * 2 * 4)
    rough = zero.copy()
    rough[::2, 0], rough[1::2, 0] = 2, -2
    result = v.reference_precision(rough, cumulative_draws=4096, protocol=protocol, proposal="MIX")
    assert result["status"] == "REFERENCE_EXTEND" and result["next_cumulative_draws"] == 8192
    with pytest.raises(ValueError, match="predeclared"):
        v.reference_precision(zero, cumulative_draws=5000, protocol=protocol, proposal="MIX")
    with pytest.raises(ValueError, match="family"):
        v.reference_precision(zero, cumulative_draws=4096, protocol={}, proposal="MIX")
    result = v.reference_precision(
        zero,
        cumulative_draws=4096,
        protocol=protocol,
        proposal="ORIGIN",
        diagnostics={"reasons": ["ORIGIN_SUPPORT_UNCERTIFIED"]},
    )
    assert result["status"] == "REFERENCE_REQUIRES_INDEPENDENT_MIX"
    assert result["formal_hoeffding_half_width"] is result["next_cumulative_draws"] is None


def test_cross_gpu_null_checks_policy_action_runtime_and_thresholds():
    b, _ = backend()
    b.activate({"inference_fingerprint": "A"})
    sample = {**b.generate(prompt(), seed=0), "sample_key": "draw"}
    score = b.score(prompt(), sample)
    left = {
        "gpu_identity": "GPU-UUID-A",
        "runtime_identity": b.runtime_identity,
        "tolerances": TOLERANCE,
        "scores": [score],
    }
    right = {**copy.deepcopy(left), "gpu_identity": "GPU-UUID-B"}
    assert v.compare_cross_gpu_receipts(left, right, tolerances=TOLERANCE)["status"] == "PASS"
    right["scores"][0]["token_logprobs"][0] -= 0.1
    assert (
        v.compare_cross_gpu_receipts(left, right, tolerances=TOLERANCE)["status"]
        == "FAIL_NULL_PARITY"
    )
    right["scores"][0]["token_ids"][0] = 100
    with pytest.raises(ValueError, match="identity"):
        v.compare_cross_gpu_receipts(left, right, tolerances=TOLERANCE)
    with pytest.raises(ValueError, match="distinct"):
        v.compare_cross_gpu_receipts(left, left, tolerances=TOLERANCE)


def test_semantic_strings_are_event_subset_and_aliases_not_independent_draws():
    samples = [
        {"category": "X", "raw_completion": "[1,2,3,4]", "token_ids": [1, 99]},
        {"category": "X", "raw_completion": " [1, 2, 3, 4] ", "token_ids": [2, 99]},
    ]
    result = v.audit_semantic_aliases(samples)
    assert result["observed_semantic_aliases"]
    assert not result["known_string_is_whole_event"] and not result["enumeration_complete"]
    scores = [
        {
            **row,
            "prompt_record_hash": "p",
            "inference_fingerprint": "A",
            "token_logprobs": [-2, -1],
            "eos_token_ids": [99],
            "max_new_tokens": 64,
        }
        for row in samples
    ]
    known = v.known_action_probability(scores)
    assert known["event_lower_bound"] == 2 * math.exp(-3) and not known["event_probability_exact"]
    with pytest.raises(ValueError, match="Duplicate"):
        v.known_action_probability([scores[0], scores[0]])


def test_generation_scoring_resume_original_shards_and_candidate_grouping(tmp_path):
    b, loads = backend()
    generated, out = manifest(tmp_path / "input"), tmp_path / "generation"
    assert (
        v.execute_observation_tasks(generated, b, out=out, worker_index=0)["status"] == "COMPLETE"
    )
    assert loads == ["A"] and b.counters["generated_sequences"] == 4
    shards = [binding(p) for p in out.rglob("samples.jsonl")]
    before = dict(b.counters)
    v.execute_observation_tasks(generated, b, out=out, worker_index=0, resume=True)
    assert before == b.counters
    with pytest.raises(ValueError, match="resume"):
        v.execute_observation_tasks(generated, b, out=out, worker_index=0)
    scoring = manifest(tmp_path / "input", operation="score", sample_files=shards)
    v.execute_observation_tasks(scoring, b, out=tmp_path / "score", worker_index=0)
    assert loads == ["A", "B"]
    rows = v._read_rows(next((tmp_path / "score").rglob("samples.jsonl")))
    assert all(row["sequence_logp"] == -0.85 and row["proposal_source"] == "A" for row in rows)
    assert b.counters["generated_sequences"] == 4
    assert b.counters["scored_sequences"] + b.counters["score_alias_reuses"] == 8
    b.runtime_identity["code"] = "changed"
    with pytest.raises(ValueError, match="runtime identity"):
        v.execute_observation_tasks(generated, b, out=out, worker_index=0, resume=True)


def test_oom_resume_missing_only_partial_shard_retained(tmp_path):
    m, out = manifest(tmp_path / "input"), tmp_path / "out"
    b, _ = backend()
    b.adapter.fail_after = 2
    with pytest.raises(RuntimeError, match="OOM"):
        v.execute_observation_tasks(m, b, out=out, worker_index=0)
    original = next(out.rglob("samples.jsonl"))
    original.write_bytes(original.read_bytes() + b'{"interrupted":')
    interrupted_bytes = original.read_bytes()
    fresh, _ = backend()
    assert (
        v.execute_observation_tasks(m, fresh, out=out, worker_index=0, resume=True)["status"]
        == "COMPLETE"
    )
    assert fresh.counters["generated_sequences"] == 2 and original.read_bytes() == interrupted_bytes
    assert len(list(out.rglob("samples.jsonl"))) == 2 and len(list(out.rglob("COMPLETE.json"))) == 1


def test_manifest_mutation_duplicate_input_tamper_and_worker_ownership(tmp_path):
    m, (b, _) = manifest(tmp_path / "input"), backend()
    empty = v.execute_observation_tasks(m, b, out=tmp_path / "out", worker_index=1)
    assert not empty["completed_tasks"] and not b.counters["generated_sequences"]
    broken = copy.deepcopy(m)
    broken["tasks"][0]["role"] = "main"
    with pytest.raises(ValueError, match="manifest hash"):
        v.validate_task_manifest(broken, workers=2)
    broken["manifest_hash"] = v.digest(
        {k: val for k, val in broken.items() if k != "manifest_hash"}
    )
    with pytest.raises(ValueError, match="Task identity"):
        v.validate_task_manifest(broken, workers=2)
    repeated = copy.deepcopy(m)
    repeated["tasks"] *= 2
    repeated["manifest_hash"] = v.digest(
        {k: val for k, val in repeated.items() if k != "manifest_hash"}
    )
    with pytest.raises(ValueError, match="duplicate"):
        v.validate_task_manifest(repeated, workers=2)
    Path(m["tasks"][0]["prompt_file"]["path"]).write_text("[]")
    with pytest.raises(ValueError, match="hash changed"):
        v.execute_observation_tasks(m, b, out=tmp_path / "other", worker_index=0)


def test_returned_generation_fault_retained(tmp_path):
    m, (b, _) = manifest(tmp_path / "input", draws=1), backend()
    generate = b.adapter.generate

    def bad_generate(*args, **kwargs):
        return {**generate(*args, **kwargs), "raw_completion": "decoder drift"}

    b.adapter.generate = bad_generate
    with pytest.raises(v.ObservationFault, match="decode"):
        v.execute_observation_tasks(m, b, out=tmp_path / "out", worker_index=0)
    assert (
        json.loads(next((tmp_path / "out").rglob("RETURNED_FAULT.json")).read_text())[
            "raw_completion"
        ]
        == "decoder drift"
    )
    assert not list((tmp_path / "out").rglob("COMPLETE.json"))


def test_mix_packet_binds_actual_endpoint_pair_and_refuses_pooled_prompts(tmp_path):
    m, (b, _) = manifest(tmp_path / "input", proposal="MIX"), backend()
    out = tmp_path / "generated"
    v.execute_observation_tasks(m, b, out=out, worker_index=0)
    samples = v._read_rows(next(out.rglob("samples.jsonl")))
    b.activate(m["policies"]["A"])
    a = [b.score(prompt(), sample) for sample in samples]
    b.activate(m["policies"]["B"])
    endpoint_b = [b.score(prompt(), sample) for sample in samples]
    raw, diagnostics = v.packet_contributions(samples, a, endpoint_b, proposal="MIX")
    assert raw.shape == (4, 4) and not diagnostics["requires_independent_mix"]
    wrong_endpoint = copy.deepcopy(endpoint_b)
    for row in wrong_endpoint:
        row["inference_fingerprint"] = "unrelated_policy"
    with pytest.raises(ValueError, match="actual MIX"):
        v.packet_contributions(samples, a, wrong_endpoint, proposal="MIX")
    pooled = copy.deepcopy(samples)
    pooled[1]["prompt_record_hash"] = "second_prompt"
    with pytest.raises(ValueError, match="one prompt"):
        v.packet_contributions(pooled, a, endpoint_b, proposal="MIX")
    wrong_coin = copy.deepcopy(samples)
    wrong_coin[0]["proposal_source_index"] = 1 - wrong_coin[0]["proposal_source_index"]
    b.activate(m["policies"]["A"])
    a_bad = [b.score(prompt(), sample) for sample in wrong_coin]
    b.activate(m["policies"]["B"])
    b_bad = [b.score(prompt(), sample) for sample in wrong_coin]
    with pytest.raises(ValueError, match="source coin"):
        v.packet_contributions(wrong_coin, a_bad, b_bad, proposal="MIX")


def test_resumed_partial_shards_can_be_scored_and_second_writer_excluded(tmp_path):
    import fcntl

    m, (b, _) = manifest(tmp_path / "input"), backend()
    out = tmp_path / "generated"
    b.adapter.fail_after = 1
    with pytest.raises(RuntimeError, match="OOM"):
        v.execute_observation_tasks(m, b, out=out, worker_index=0)
    partial = next(out.rglob("samples.jsonl"))
    partial.write_bytes(partial.read_bytes() + b'{"half":')
    fresh, _ = backend()
    v.execute_observation_tasks(m, fresh, out=out, worker_index=0, resume=True)
    scoring = manifest(
        tmp_path / "input",
        operation="score",
        sample_files=[binding(p) for p in out.rglob("samples.jsonl")],
    )
    v.execute_observation_tasks(scoring, fresh, out=tmp_path / "scores", worker_index=0)
    lock = out / "worker_0" / "writer.lock"
    with lock.open("a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="active writer"):
            v.execute_observation_tasks(m, fresh, out=out, worker_index=0, resume=True)


def test_cli_authorization_precedes_any_model_import():
    code = """import sys
from src.modeling_v3.vlm_observation import observe_vlm
try:
    observe_vlm({}, {}, task_manifest={}, worker_index=0, out='unused')
except PermissionError:
    pass
else:
    raise AssertionError('authorization was bypassed')
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_known_action_baseline_rejects_unterminated_prefixes():
    scores = [
        {
            "category": "X",
            "prompt_record_hash": "p",
            "inference_fingerprint": "A",
            "token_ids": [1, 2],
            "token_logprobs": [-1, -1],
            "max_new_tokens": 64,
            "eos_token_ids": [99],
        }
    ]
    with pytest.raises(ValueError, match="complete finite-horizon"):
        v.known_action_probability(scores)


def test_exact_policy_aliases_reuse_physical_scores_and_keep_alias_receipt(tmp_path):
    generated = manifest(tmp_path / "input", draws=4)
    original, _ = backend()
    v.execute_observation_tasks(generated, original, out=tmp_path / "generated", worker_index=0)
    sample_files = [binding(p) for p in (tmp_path / "generated").rglob("samples.jsonl")]
    plan = manifest(tmp_path / "input", operation="score", sample_files=sample_files)
    fields = {
        k: val
        for k, val in plan["tasks"][0].items()
        if k not in {"task_id", "output_path", "request_keys"}
    }
    tasks = [
        v.freeze_observation_task(**{**fields, "candidate_id": candidate})
        for candidate in ("A", "A_alias")
    ]
    policies = {**plan["policies"], "A_alias": copy.deepcopy(plan["policies"]["A"])}
    alias_plan = v.freeze_task_manifest(tasks, policies, plan["runtime_identity"])
    measured, loads = backend()
    v.execute_observation_tasks(alias_plan, measured, out=tmp_path / "aliases", worker_index=0)
    assert measured.counters["scored_sequences"] == 0
    assert measured.counters["score_alias_reuses"] == 8
    rows = [r for p in (tmp_path / "aliases").rglob("samples.jsonl") for r in v._read_rows(p)]
    assert len(rows) == 8 and all(r["score_execution"] == "INFERENCE_ALIAS" for r in rows)
    assert all(not r["score_is_independent_draw"] and not r["training_resume_alias"] for r in rows)
    assert loads == ["A"]  # identical checkpoint spec was already actually verified
    bad_tasks = [
        tasks[0],
        v.freeze_observation_task(**{**fields, "candidate_id": "A_alias", "worker": 1}),
    ]
    with pytest.raises(ValueError, match=r"same|share"):
        v.freeze_task_manifest(bad_tasks, policies, plan["runtime_identity"])


def test_paid_fault_calls_remain_in_counters(tmp_path):
    plan, (measured, _) = manifest(tmp_path / "input", draws=1), backend()
    original = measured.adapter.logprobs

    def wrong_score(*args, **kwargs):
        values = original(*args, **kwargs)
        return [values[0] - 0.1, values[1]]

    measured.adapter.logprobs = wrong_score
    with pytest.raises(v.ObservationFault, match="parity"):
        v.execute_observation_tasks(plan, measured, out=tmp_path / "out", worker_index=0)
    assert measured.counters["generated_sequences"] == measured.counters["scored_sequences"] == 1
    assert measured.counters["generated_tokens"] == measured.counters["scored_tokens"] == 2
    assert measured.counters["accepted_samples"] == 0


def test_known_string_baseline_is_actual_scoring_without_fabricated_sampling():
    measured, _ = backend()
    measured.activate({"inference_fingerprint": "A"})
    rows = measured.score_known_actions(prompt(), [[0, 99]])
    result = v.known_action_probability(rows)
    assert result["known_X_action_mass"] == math.exp(-0.75)
    assert measured.counters["generated_sequences"] == 0
    assert measured.counters["scored_sequences"] == 1
    assert rows[0]["is_sampling_draw"] is False and "sample_key" not in rows[0]
    assert not result["event_probability_exact"]


def _tiny_q4_worker(tmp_path):
    from src.modeling_v3.io import source_identity

    config = {
        "qwen": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 64,
            "two_gpu_bridge": {"dev_banks": 1, "probes": 1, "initial_draws": 8},
        },
        "observation": {"pilot_fraction": 0.25, "pilot_b_l1_cap": 8.0},
    }
    tmp_path.mkdir(exist_ok=True)
    original = manifest(tmp_path / "input")
    measured, _ = backend()
    measured.runtime_identity = {
        **measured.runtime_identity,
        "config_hash": v.digest(config),
        "source_hash": source_identity()["sha256"],
    }
    state = original["policies"]["A"]["checkpoint"]
    policies = {
        name: {"checkpoint": state, "inference_fingerprint": fp}
        for name, fp in (
            ("origin", "A"),
            ("bank0_joint_0", "A"),
            ("bank0_joint_1", "B"),
            ("bank0_no_x_off_1", "C"),
        )
    }
    endpoints = ["bank0_joint_1", "bank0_joint_0"]
    kinds = [
        ("pilot", "ORIGIN", ["origin"], 0, 2),
        ("main", "ORIGIN", ["origin"], 2, 8),
        ("reference", "ORIGIN", ["origin"], 0, 8),
        ("reference", "MIX", endpoints, 0, 8),
    ]
    kinds += [("reference", "DIRECT", [p], 0, 8) for p in endpoints]
    tasks = [
        v.freeze_observation_task(
            operation="generate",
            origin_id="fixture_origin",
            candidate_id=sources[0],
            prompt_file=original["tasks"][0]["prompt_file"],
            prompt_ids=["prompt1"],
            role=role,
            draw_start=start,
            draw_stop=stop,
            rng_namespace="tiny-q4",
            proposal=proposal,
            proposal_candidates=sources,
            worker=0,
        )
        for role, proposal, sources, start, stop in kinds
    ]
    generation = v.freeze_task_manifest(tasks, policies, measured.runtime_identity, workers=1)
    (tmp_path / "generation_tasks.json").write_text(json.dumps(generation))
    v.execute_observation_tasks(
        generation, measured, out=tmp_path / "generation", worker_index=0, workers=1
    )
    scoring_tasks = []
    for task in tasks:
        if task["proposal"] == "DIRECT":
            continue
        base = tmp_path / "generation" / "worker_0" / task["output_path"]
        sample_files = [binding(p) for p in base.glob("attempt_*/samples.jsonl")]
        fields = {
            k: val for k, val in task.items() if k not in {"task_id", "request_keys", "output_path"}
        }
        for candidate in policies if task["proposal"] == "ORIGIN" else endpoints:
            scoring_tasks.append(
                v.freeze_observation_task(
                    **{
                        **fields,
                        "operation": "score",
                        "candidate_id": candidate,
                        "sample_files": sample_files,
                    }
                )
            )
    scoring = v.freeze_task_manifest(scoring_tasks, policies, measured.runtime_identity, workers=1)
    (tmp_path / "scoring_tasks.json").write_text(json.dumps(scoring))
    v.execute_observation_tasks(
        scoring, measured, out=tmp_path / "scoring", worker_index=0, workers=1
    )
    return config


def test_q4_analysis_runs_geometry_direct_and_independent_noisy_references(tmp_path):
    config = _tiny_q4_worker(tmp_path)
    result = v.analyze_q4_worker(tmp_path, config)
    report = json.loads(Path(result["path"]).read_text())
    assert result["bank_prompt_units"] == 1 and result["direct_count_units"] == 2
    methods = report["observations"][0]["estimators"]
    assert {"RAW4", "EQUAL_ZERO_SUM", "PRESERVE_XI", "PILOT_SHRINK_ZERO_SUM"} <= set(methods)
    assert methods["RAW4"]["n"] == 8 and methods["PILOT_SHRINK_ZERO_SUM"]["n"] == 6
    assert len(report["independent_origin_references"]) == 3
    assert len(report["independent_mix_crosschecks"]) == 1
    assert len(report["independent_direct_contrasts"]) == 1
    assert report["independent_direct_contrasts"][0]["total_generated_sequences"] == 16
    assert not report["independent_direct_contrasts"][0]["same_cost_claim"]
    assert all(
        row["status"] == "REFERENCE_UNRESOLVED" for row in report["independent_origin_references"]
    )
    assert all(not row["ranking_allowed"] for row in report["noisy_reference_error_aggregates"])
    assert report["scientific_method_winner"] is None and not report["online_ssvc_validated"]
    with np.load(report["raw_contributions"]["path"], allow_pickle=False) as raw:
        assert raw["unit_0000_pilot_raw4"].shape == (2, 3, 4)
        assert raw["unit_0000_reference_raw4"].shape == (8, 3, 4)
    assert v.analyze_q4_worker(tmp_path, config) == result
    original = next((tmp_path / "generation").rglob("samples.jsonl"))
    original.write_bytes(original.read_bytes() + b" ")
    with pytest.raises(ValueError, match=r"bytes changed|hash changed"):
        v.analyze_q4_worker(tmp_path, config)
