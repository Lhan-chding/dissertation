"""Read-only full R4 acceptance before exposing the X_BASE step64 warm origin.

Only saved LoRA/Adam tensors and the pinned local CPU processor are loaded. This
module never loads the 9B model, generates, changes a checkpoint, or opens confirm.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from .core import PROJECT_ROOT, canonical_hash, file_hash
from .optimizer_fork import load_checkpoint, state_hash
from .r3_gate import _check_optimizer_state, _valid_hash

ARMS = ("X_BASE", "X_VALID")
KIND = "REAL_CUDA_TRAINING"


KL_FLOAT_COMPARISON = {
    "contract": "R4_DERIVED_KL_FLOAT64_ROUNDOFF_V1",
    "epsilon_multiplier": 64,
    "float64_epsilon": sys.float_info.epsilon,
    "bound": "64 * float64_epsilon * max(1, abs(stored), abs(recomputed))",
    "scope": (
        "Expm1-derived KL quantities only; decisions, thresholds, sequence ratios "
        "and all other evidence exact"
    ),
}


def _derived_kl_float(path):
    return (
        path == ("mean_token_kl",)
        or (len(path) == 3 and path[0] in ("by_prompt", "by_group") and path[2] == "mean_token_kl")
        or (
            len(path) == 3
            and path[0] == "sequence_records"
            and type(path[1]) is int
            and path[2] in ("token_k3_sum", "mean_token_kl")
        )
    )


def _compare_kl_diagnostic(stored, recomputed):
    """Tolerate measured CPU expm1 rounding, never an altered alarm decision.

    Raw scores remain exact. The returned audit is optional local information;
    it must not enter a run, gate or continuation identity. Callers retain the
    original stored diagnostic in every artifact-to-artifact comparison.
    """
    differences = []

    def compare(a, b, path=()):
        if type(a) is not type(b):
            raise ValueError(f"R4 KL diagnostic type changed at {path}")
        if isinstance(a, dict):
            if set(a) != set(b):
                raise ValueError(f"R4 KL diagnostic fields changed at {path}")
            for key in sorted(a):
                compare(a[key], b[key], (*path, key))
        elif isinstance(a, list):
            if len(a) != len(b):
                raise ValueError(f"R4 KL diagnostic coverage changed at {path}")
            for index, (left, right) in enumerate(zip(a, b, strict=True)):
                compare(left, right, (*path, index))
        elif isinstance(a, float):
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError(f"R4 KL diagnostic is nonfinite at {path}")
            if canonical_hash(a) == canonical_hash(b):
                return
            if path == ("mean_token_kl",) and (a > 0.1) != (b > 0.1):
                raise ValueError("R4 mean KL threshold crossing cannot be treated as rounding")
            bound = 64 * sys.float_info.epsilon * max(1.0, abs(a), abs(b))
            if not _derived_kl_float(path) or abs(a - b) > bound:
                raise ValueError(
                    f"R4 KL diagnostic differs beyond derived-float rounding at {path}"
                )
            differences.append(
                {
                    "path": list(path),
                    "stored": a,
                    "recomputed": b,
                    "absolute_difference": abs(a - b),
                    "allowed_absolute_difference": bound,
                }
            )
        elif canonical_hash(a) != canonical_hash(b):
            raise ValueError(f"R4 KL diagnostic exact value changed at {path}")

    compare(stored, recomputed)
    return differences


def _read(path):
    return json.loads(Path(path).read_text())


def _equal(actual, expected, message):
    if canonical_hash(actual) != canonical_hash(expected):
        raise ValueError(message)


def _committed_sources(revision):
    """Read Git objects only; later gate fixes need not match the old run checkout."""
    try:
        listing = subprocess.run(
            ["git", "ls-tree", "--full-tree", "-r", "-z", revision, "--", "ssvc_flow/src"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        entries = []
        for entry in listing.split(b"\0"):
            if not entry:
                continue
            header, name = entry.split(b"\t", 1)
            mode, kind, oid = header.split()
            name = name.decode()
            if name.endswith(".py"):
                if kind != b"blob" or mode not in (b"100644", b"100755"):
                    raise ValueError("R4 committed Python source is not a regular file")
                entries.append((name.removeprefix("ssvc_flow/"), oid))
        if not entries:
            raise ValueError("R4 committed source tree is empty")
        output = subprocess.run(
            ["git", "cat-file", "--batch"],
            cwd=PROJECT_ROOT,
            input=b"\n".join(oid for _, oid in entries) + b"\n",
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("R4 immutable source Git objects are unavailable") from exc
    hashes, offset = {}, 0
    for name, oid in entries:
        end = output.index(b"\n", offset)
        actual_oid, kind, size = output[offset:end].split()
        size = int(size)
        start = end + 1
        content = output[start : start + size]
        if (
            actual_oid != oid
            or kind != b"blob"
            or len(content) != size
            or output[start + size : start + size + 1] != b"\n"
        ):
            raise ValueError("R4 Git source object response is incomplete")
        hashes[name] = hashlib.sha256(content).hexdigest()
        offset = start + size + 1
    if offset != len(output):
        raise ValueError("R4 Git source object response has trailing content")
    return hashes


def _check_source(runtime, gate):
    source = runtime["source"]
    revision = source.get("source_commit")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("R4 source revision is not an immutable Git commit")
    committed = _committed_sources(revision)
    _equal(
        source.get("source_files"),
        committed,
        "R4 source files differ from complete committed Python tree",
    )
    frozen = gate["binding"]["source_implementation_hashes"]
    if not frozen or any(committed.get(name) != digest for name, digest in frozen.items()):
        raise ValueError("R4 committed frozen implementation differs from R1 certificate")


def _check_finished_source(status, runtime):
    _equal(
        {k: status.get(k) for k in ("source_commit", "source_files")},
        runtime["source"],
        "R4 source changed during execution or final evidence binding",
    )


def _check_runtime_profile(root, files, cost, runtime, *, completion_status="PASS"):
    histories = cost.get("invocations")
    if not isinstance(histories, list) or not histories:
        raise ValueError("R4 requires original instrumented invocation profiles")
    names = [h["invocation"] for h in histories]
    actual = sorted(p.name for p in (root / "invocations").iterdir() if p.is_dir())
    if names != actual or len(names) != len(set(names)):
        raise ValueError("R4 invocation cost coverage changed")
    for history in histories:
        invocation = root / "invocations" / history["invocation"]
        completed = (invocation / "completion.json").exists()
        if completed != history.get("completed"):
            raise ValueError("R4 invocation completion cost flag changed")
        if completed:
            _path(root, files, invocation / "completion.json")
        observed = history.get("observed")
        profile = invocation / "runtime_profile.json"
        if profile.exists():
            _equal(
                _read(_path(root, files, profile)),
                observed,
                "R4 invocation observed counters changed",
            )
            if (
                observed.get("language_layers_instrumented") != 32
                or observed.get("top_hook_installed") is not True
                or observed.get("vision_hooks_installed") != 1
                or observed.get("internal_recompute_status") != "MEASURED"
                or any(
                    type(observed.get(k)) is not int or observed[k] < 0
                    for k in ("optimizer_step_calls_observed", "backward_calls_observed")
                )
            ):
                raise ValueError("R4 forward/backward instrumentation is incomplete")
        elif observed != {}:
            raise ValueError("R4 nonempty counters without an invocation profile")
        audit_path = invocation / "model_load_audit.json"
        if audit_path.exists():
            audit = _read(_path(root, files, audit_path))
            _equal(
                {
                    k: v
                    for k, v in audit.items()
                    if k not in ("load_seconds", "load_peak_cuda_bytes")
                },
                runtime["model_audit"],
                "R4 actual invocation model differs from runtime lock",
            )
            if (
                type(audit.get("load_peak_cuda_bytes")) is not int
                or audit["load_peak_cuda_bytes"] <= 0
            ):
                raise ValueError("R4 invocation model was not measured on CUDA")
        elif observed:
            raise ValueError("R4 instrumented invocation lacks model load evidence")
    expected = {
        "optimizer_steps_observed_at_least": sum(
            h["observed"].get("optimizer_step_calls_observed", 0) for h in histories
        ),
        "backward_calls_observed_at_least": sum(
            h["observed"].get("backward_calls_observed", 0) for h in histories
        ),
        "incomplete_invocations_with_unknown_extra_cost": sum(
            not h["completed"] for h in histories
        ),
    }
    if any(cost.get(k) != v for k, v in expected.items()):
        raise ValueError("R4 aggregate counts differ from measured invocation history")
    completion = _read(root / "invocations" / names[-1] / "completion.json")
    if (
        completion.get("status") != completion_status
        or completion.get("details", {}).get("final_scratch_origin_restored") is not True
    ):
        raise ValueError("R4 final invocation did not restore its scratch origin")


def _path(root, files, path, *, recorded_root=None):
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    elif recorded_root is not None and path.is_relative_to(Path(recorded_root)):
        path = root / path.relative_to(recorded_root)
    path = path.resolve()
    if not path.is_relative_to(root) or str(path.relative_to(root)) not in files:
        raise ValueError("R4 required artifact is outside the verified manifest")
    return path


def _ledger(path, *, kind=KIND):
    result = {}
    with Path(path).open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError("R4 truncated measured ledger")
            row = json.loads(line)
            key = row.get("sample_key")
            if not isinstance(key, str) or not key or key in result:
                raise ValueError("R4 duplicate or missing measured sample identity")
            if (
                row.get("execution_kind") != kind
                or row.get("execution_checks", {}).get("passed") is not True
            ):
                raise ValueError("R4 fake or failed measured output")
            if row.get("record_hash") != canonical_hash(
                {k: v for k, v in row.items() if k != "record_hash"}
            ):
                raise ValueError("R4 measured output hash mismatch")
            result[key] = row
    return result


def _parameter_hash(state):
    return state_hash({name: state_hash(value) for name, value in state["parameters"].items()})


def _policy(state):
    return {
        "state_hash": state_hash(state),
        "parameter_hash": _parameter_hash(state),
        "optimizer_state_hash": state_hash(state["optimizer"]),
    }


def _check_state(state, origin, arm, step, plan_hash, keys, last_keys):
    import torch

    _check_optimizer_state(state, step=step)
    if (
        state["optimizer"]["param_groups"] != origin["optimizer"]["param_groups"]
        or state["scheduler"] is not None
    ):
        raise ValueError("R4 checkpoint optimizer/scheduler protocol changed")
    if set(state["parameters"]) != set(origin["parameters"]) or any(
        value.shape != origin["parameters"][name].shape
        or value.dtype != origin["parameters"][name].dtype
        for name, value in state["parameters"].items()
    ):
        raise ValueError("R4 trainable parameter identity changed")
    expected = {
        "arm": arm,
        "checkpoint_step": step,
        "sampler": {"plan_hash": plan_hash, "position": step},
        "completed_sample_keys": keys,
        "scheduler": None,
        "grad_scaler": None,
    }
    if step:
        expected["completed_step_sample_keys"] = last_keys
    if len(keys) != step * 32 or len(set(keys)) != len(keys):
        raise ValueError("R4 checkpoint must contain exactly its own completed training keys")
    _equal(state["metadata"], expected, "R4 checkpoint arm/step/RNG-sampler metadata changed")
    if set(state["rng"]) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("R4 checkpoint must retain complete RNG families")
    if len(state["rng"]["cuda"]) != len(origin["rng"]["cuda"]) or any(
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.uint8
        or value.ndim != 1
        or not value.numel()
        for value in [state["rng"]["torch"], *state["rng"]["cuda"]]
    ):
        raise ValueError("R4 checkpoint dropped or changed saved RNG state families")
    if step:
        indices = [i for group in state["optimizer"]["param_groups"] for i in group["params"]]
        for index, parameter in zip(indices, state["parameters"].values(), strict=True):
            moments = state["optimizer"]["state"][index]
            if any(
                moments[k].shape != parameter.shape or moments[k].dtype != parameter.dtype
                for k in ("exp_avg", "exp_avg_sq")
            ) or torch.any(moments["exp_avg_sq"] < 0):
                raise ValueError("R4 Adam moments have invalid shape/dtype/second moment")


class _PreparedInputs(dict):
    """Keep independently reconstructed CPU audits, never full pixel tensors."""

    def __init__(self, adapter, data_root):
        super().__init__()
        self.adapter, self.data_root = adapter, data_root

    def check(self, prompt, row):
        from .r4_runtime import _prepared

        key = prompt["prompt_id"]
        if key not in self:
            prepared = _prepared(self.adapter, prompt, self.data_root)
            self[key] = {**prepared["audit"], "prepared_hash": state_hash(prepared)}
        for name, value in self[key].items():
            if name not in row or canonical_hash(row[name]) != canonical_hash(value):
                raise ValueError(f"R4 input differs from CPU reconstructed preparation: {name}")


def _validate_raw_row(
    row, prompt, identity, policy, config, legacy_lock, tokenizer, eos, input_bindings
):
    import torch

    from .model_adapters.base import _hash_json
    from .r2_runtime import generation_checks
    from .r4_runtime import _annotate_rollout, _scene_fields

    raw, tokens, old = row.get("raw_completion"), row.get("token_ids"), row.get("old_logprobs")
    if (
        not isinstance(raw, str)
        or row.get("raw_text") != raw
        or not isinstance(tokens, list)
        or row.get("raw_token_ids") != tokens
        or not isinstance(old, list)
        or not old
        or len(old) != len(tokens)
        or any(type(v) not in (int, float) or not math.isfinite(v) or v > 1e-5 for v in old)
        or row.get("behavior_token_logprobs") != old
        or row.get("per_token_logprob_behavior") != old
        or row.get("logprob_sequence") != math.fsum(old)
        or row.get("n_generated_tokens") != len(tokens)
        or row.get("execution_kind") != KIND
        or row.get("execution_checks") != {"passed": True, "faults": []}
    ):
        raise ValueError("R4 incomplete raw/token/frozen old-logp evidence")
    faults = generation_checks(row, tokenizer, eos, prompt["max_new_tokens"])
    if faults:
        raise ValueError("R4 raw token decoding/EOS/budget mismatch: " + "; ".join(faults))
    hashes = (
        "final_prompt_hash",
        "tokenized_prompt_hash",
        "input_ids_hash",
        "input_tensor_hash",
        "prepared_hash",
    )
    if any(not _valid_hash(row.get(k)) for k in hashes):
        raise ValueError("R4 original prepared input hash missing")
    text, ids = row.get("final_prompt"), row.get("final_prompt_token_ids")
    if (
        not isinstance(text, str)
        or not text.rstrip().endswith("</think>")
        or prompt["prompt"]["user"] not in text
        or (prompt["prompt"].get("system") and prompt["prompt"]["system"] not in text)
        or row["final_prompt_hash"] != _hash_json(text)
        or not isinstance(ids, list)
        or not ids
        or any(type(t) is not int or t < 0 for t in ids)
        or row.get("prompt_token_count") != len(ids)
        or row["tokenized_prompt_hash"] != state_hash(torch.tensor([ids], dtype=torch.long))
        or row["input_ids_hash"] != row["tokenized_prompt_hash"]
    ):
        raise ValueError("R4 prompt text/token identity changed")
    image = prompt["interface"] == "IMAGE_CUE_FRESH"
    count = row.get("image_token_count")
    if (
        row.get("enable_thinking") is not False
        or type(count) is not int
        or count < 0
        or bool(count) != image
        or row.get("actual_image_tokens") != count
        or (
            image
            and (
                not _valid_hash(row.get("pixel_values_hash"))
                or row.get("vision_forward_calls", 0) < 1
            )
        )
        or (not image and row.get("pixel_values_hash") is not None)
    ):
        raise ValueError("R4 image/thinking intervention changed")
    expected = {
        "run_id": canonical_hash(identity),
        "policy_state_hash": policy["state_hash"],
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "adapter_hash": policy["parameter_hash"],
        "optimizer_state_hash": policy["optimizer_state_hash"],
        "protocol_version": config["protocol_version"],
        "train_seed": 17,
        "generation_config_hash": canonical_hash(
            legacy_lock["generation"] if prompt["track"] == "L" else config["generation_proposed_N"]
        ),
        **_annotate_rollout(raw, prompt),
        **_scene_fields(prompt, row),
    }
    if any(
        k not in row or canonical_hash(row[k]) != canonical_hash(v) for k, v in expected.items()
    ):
        raise ValueError("R4 annotation, source scene, policy or generation protocol changed")
    if (
        type(row.get("elapsed")) not in (int, float)
        or not math.isfinite(row["elapsed"])
        or row["elapsed"] < 0
        or row.get("elapsed") != row.get("elapsed_seconds")
        or row.get("elapsed_unit") != "seconds"
        or row.get("elapsed_scope") != "adapter.generate"
        or row.get("elapsed_status") != "MEASURED"
        or row.get("peak_memory_status") != "CUDA_PHASE_PEAK_AND_PROCESS_RSS_HIGH_WATERMARK"
        or not isinstance(row.get("peak_memory"), dict)
        or any(
            type(row.get("peak_memory", {}).get(k)) is not int or row["peak_memory"][k] <= 0
            for k in ("peak_cuda_bytes", "peak_cuda_reserved_bytes", "peak_cpu_rss_bytes")
        )
        or row.get("runtime_forward_by_reason", {}).get("generation", 0) < 1
    ):
        raise ValueError("R4 measured runtime metadata missing or simulated")
    binding = {
        k: row.get(k) for k in (*hashes, "image_token_count", "pixel_values_hash", "image_hash")
    }
    if isinstance(input_bindings, _PreparedInputs):
        input_bindings.check(prompt, row)
    elif input_bindings.setdefault(prompt["prompt_id"], binding) != binding:
        raise ValueError("R4 same prompt inputs changed across arms/checkpoints")


def _raw_store(
    root,
    files,
    path,
    prompts,
    identity,
    state,
    arm,
    step,
    role,
    config,
    legacy_lock,
    tokenizer,
    eos,
    inputs,
    *,
    sampling_identity=None,
):
    from .r3_runtime import validate_sample_ledger
    from .r4_runtime import build_requests

    policy = _policy(state)
    logical = identity if sampling_identity is None else sampling_identity
    requests = build_requests(prompts, logical, arm, step, role, policy["state_hash"])
    expected_identity = {
        **identity,
        "arm": arm,
        "step": step,
        "role": role,
        "policy_state_hash": policy["state_hash"],
        "request_hash": canonical_hash(requests),
    }
    _equal(
        _read(_path(root, files, path / "identity.json")),
        expected_identity,
        "R4 sample store policy/request identity mismatch",
    )
    rows = _ledger(_path(root, files, path / "samples.jsonl"))
    validate_sample_ledger(rows, requests)
    if set(rows) != {r["sample_key"] for r in requests}:
        raise ValueError("R4 required sampled endpoint/training coverage is incomplete")
    by_prompt = {p["prompt_id"]: p for p in prompts}
    ordered = [rows[r["sample_key"]] for r in requests]
    for row in ordered:
        if sampling_identity is not None and row.get("execution_identity_hash") != canonical_hash(
            identity
        ):
            raise ValueError("R4 continuation raw output lacks its actual execution identity")
        _validate_raw_row(
            row,
            by_prompt[row["prompt_id"]],
            logical,
            policy,
            config,
            legacy_lock,
            tokenizer,
            eos,
            inputs,
        )
    return ordered


def _score_store(root, files, path, rows, identity, policy):
    expected = {
        **identity,
        "scored_parameter_hash": policy["parameter_hash"],
        "scored_optimizer_hash": policy["optimizer_state_hash"],
    }
    _equal(
        _read(_path(root, files, path / "identity.json")),
        expected,
        "R4 score store policy identity changed",
    )
    records = _ledger(_path(root, files, path / "samples.jsonl"))
    wanted, result = set(), {}
    for source in rows:
        key = canonical_hash([expected, source["sample_key"]])
        wanted.add(key)
        row = records.get(key, {})
        values = row.get("new_token_logprobs")
        binding = {
            "proposal_sample_key": source["sample_key"],
            "proposal_record_hash": source["record_hash"],
            "token_ids": source["token_ids"],
            "scored_parameter_hash": policy["parameter_hash"],
            "scored_optimizer_hash": policy["optimizer_state_hash"],
        }
        if any(
            k not in row or canonical_hash(row[k]) != canonical_hash(v) for k, v in binding.items()
        ) or (
            not isinstance(values, list)
            or len(values) != len(source["token_ids"])
            or any(type(v) not in (int, float) or not math.isfinite(v) or v > 0 for v in values)
        ):
            raise ValueError("R4 measured likelihood/source binding or token probabilities changed")
        result[source["sample_key"]] = values
    if set(records) != wanted:
        raise ValueError("R4 measured likelihood coverage differs from fixed proposals")
    return result


def _probe(training, r3_binding, origin):
    r3 = Path(r3_binding["r3_cold_dir"]).resolve()
    files = r3_binding["r3_files"]
    for name in ("samples.jsonl", "runtime_lock.json", "origin.pt", "bank_manifest.json"):
        if file_hash(r3 / name) != files[name]:
            raise ValueError("R4 inherited R3 proposal evidence changed")
    runtime = _read(r3 / "runtime_lock.json")
    source_origin = load_checkpoint(
        r3 / "origin.pt", {**runtime["identity"], "unit": "initial_origin"}
    )
    if any(
        state_hash(source_origin[k]) != state_hash(origin[k]) for k in ("parameters", "optimizer")
    ):
        raise ValueError("R4 initial parameters/empty Adam differ from R3 proposal")
    rows = [
        r
        for r in _ledger(r3 / "samples.jsonl", kind="REAL_CUDA_FORK").values()
        if r["bank_role"] == "control_proposal" and r["sample_index"] == 0
    ]
    prompts = _read(r3 / "bank_manifest.json")["plan"]["control_prompts"]
    if len(rows) != 48 or {r["prompt_id"] for r in rows} != {p["prompt_id"] for p in prompts}:
        raise ValueError("R4 requires the fixed 48 R3 sample-index-zero control proposals")
    binding = training["control_probe"]
    for key, value in {
        "source_r3_gate_hash": canonical_hash(r3_binding),
        "source_samples_sha256": files["samples.jsonl"],
        "sample_keys": [r["sample_key"] for r in rows],
        "record_hashes": [r["record_hash"] for r in rows],
        "new_control_generations": 0,
        "sequences_per_step": 48,
        "planned_sequence_scores": 6144,
    }.items():
        _equal(binding.get(key), value, "R4 fixed control proposal selection changed")
    return rows


def _check_update(result, rows, arm, pre, post):
    from .grpo_update import grouped_advantages, reward_channels

    groups = []
    for prompt in dict.fromkeys(r["prompt_id"] for r in rows):
        group = [r for r in rows if r["prompt_id"] == prompt]
        rewards = [reward_channels(r["category"], arm)["sum"] for r in group]
        groups.append(
            {
                "prompt_id": prompt,
                "group_id": group[0]["group_id"],
                "K": 8,
                "lambda": 0.0 if arm == "X_BASE" else 1.0,
                "category_counts": {c: sum(r["category"] == c for r in group) for c in "XSWI"},
                "reward_vector": rewards,
                "sample_keys": [r["sample_key"] for r in group],
                **grouped_advantages(rewards),
            }
        )
    _equal(
        result["group_statistics"],
        groups,
        "R4 rewards/advantages or zero-variance group retention changed",
    )
    update = result["update"]
    if (
        any(
            update.get(k) != v
            for k, v in {
                "optimizer_updates": 1,
                "backward_calls": 32,
                "sequences": 32,
                "Lnorm": 64,
                "post_update_likelihood_sequences": 32,
            }.items()
        )
        or update.get("post_update_likelihood_forwards", 0) < 1
    ):
        raise ValueError("R4 update/gradient/postscore counters differ from B4 K8")
    tokens = update.get("token_records", [])
    if len(tokens) != 32:
        raise ValueError("R4 update must retain all 32 sequence token-mask audits")
    for row, audit, advantage in zip(
        rows, tokens, [a for g in groups for a in g["advantages"]], strict=True
    ):
        if any(
            audit.get(k) != v
            for k, v in {
                "advantage": advantage,
                "Lnorm": 64,
                "masked_token_count": len(row["token_ids"]),
                "loss_reduction": "sum_generated_tokens/(B*K*64)",
            }.items()
        ):
            raise ValueError("R4 update changed the sampled-token mask or fixed advantage")
    for key in (
        "loss",
        "grad_norm_preclip",
        "grad_norm_postclip",
        "actual_step_norm",
        "clip_fraction",
        "empirical_same_training_bank_kl",
    ):
        if type(update.get(key)) not in (int, float) or not math.isfinite(update[key]):
            raise ValueError("R4 nonfinite update scalar")
    if any(
        type(t.get(k)) not in (int, float) or not math.isfinite(t[k])
        for t in tokens
        for k in ("loss", "clip_fraction")
    ) or any(not 0 <= t["clip_fraction"] <= 1 for t in tokens):
        raise ValueError("R4 sequence loss/clip audit missing or invalid")
    if (
        update["loss"] != sum(t["loss"] for t in tokens)
        or update["clip_fraction"] != sum(t["clip_fraction"] for t in tokens) / 32
    ):
        raise ValueError("R4 aggregate loss/clip differs from its measured sequence audits")
    if (
        update["grad_norm_preclip"] < 0
        or not 0 <= update["grad_norm_postclip"] <= 1.000001
        or not 0 <= update["clip_fraction"] <= 1
    ):
        raise ValueError("R4 gradient clip audit outside fixed bounds")
    delta = math.sqrt(
        sum(
            float((post["parameters"][n] - value).float().square().sum())
            for n, value in pre["parameters"].items()
        )
    )
    if not math.isclose(delta, update["actual_step_norm"], rel_tol=1e-6, abs_tol=1e-12):
        raise ValueError("R4 actual update norm disagrees with saved state transition")
    _equal(
        result.get("training_category_counts"),
        {c: sum(r["category"] == c for r in rows) for c in "XSWI"},
        "R4 training category counts changed",
    )
    if result.get("zero_advantage_groups") != sum(g["zero_variance_flag"] for g in groups):
        raise ValueError("R4 zero-advantage groups were dropped or miscounted")


def _step_evidence(
    root,
    files,
    entry,
    identity,
    pre,
    rows,
    probe,
    plan_hash,
    keys,
    eos,
    *,
    allow_sequence_warning=False,
    recorded_root=None,
    continuation=None,
    roundoff_observer=None,
):
    import numpy as np

    from .r4_metrics import control_kl_diagnostic

    arm, step = entry["arm"], entry["step"]
    path = _path(root, files, entry["checkpoint_path"], recorded_root=recorded_root)
    attempt = path.parent
    expected_parent = root / arm / f"step_{step:02d}" / "updates"
    if attempt.parent != expected_parent or path.name != "checkpoint.pt":
        raise ValueError("R4 checkpoint is outside its arm/step measured attempt")
    if list(expected_parent.glob("attempt_*/completed.json")) != [attempt / "completed.json"]:
        raise ValueError("R4 training step has missing or multiple completed attempts")
    unit = {
        **identity,
        "unit": "training_step",
        "arm": arm,
        "step": step,
        "prestate_hash": state_hash(pre),
        "sample_hash": canonical_hash([r["record_hash"] for r in rows]),
        "probe_hash": canonical_hash([r["record_hash"] for r in probe]),
    }
    expected_checkpoint = {**unit, "unit": "post_update_checkpoint"}
    _equal(
        entry["checkpoint_identity"],
        expected_checkpoint,
        "R4 checkpoint prestate/train/probe identity changed",
    )
    for name in (
        "identity.json",
        "completed.json",
        "manifest.json",
        "result.json",
        "update.json",
        "group_statistics.json",
        "parity.json",
        "control_diagnostic.json",
    ):
        _path(root, files, attempt / name)
    _equal(_read(attempt / "identity.json"), unit, "R4 atomic step identity changed")
    marker = _read(attempt / "completed.json")
    if (
        marker.get("status") != "PASS"
        or marker.get("identity_hash") != canonical_hash(unit)
        or marker.get("manifest_sha256")
        != files[str((attempt / "manifest.json").relative_to(root))]
    ):
        raise ValueError("R4 step lacks its completed measured attempt")
    # The outer verified manifest has already checked every file's bytes.
    nested_entries = _read(attempt / "manifest.json")["files"]
    nested_names = [f["path"] for f in nested_entries]
    expected_names = {
        str(p.relative_to(attempt))
        for p in attempt.rglob("*")
        if p.is_file()
        and p.name not in ("manifest.json", "completed.json", ".writer.lock")
        and not p.name.endswith(".tmp")
    }
    if len(set(nested_names)) != len(nested_names) or set(nested_names) != expected_names:
        raise ValueError("R4 nested attempt manifest is incomplete or duplicated")
    for f in nested_entries:
        nested = _path(root, files, attempt / f["path"])
        if (
            not nested.is_relative_to(attempt)
            or f["sha256"] != files[str(nested.relative_to(root))]
            or nested.stat().st_size != f["bytes"]
        ):
            raise ValueError("R4 nested measured attempt manifest changed")
    result = _read(attempt / "result.json")
    allowed_statuses = {"PASS"}
    if allow_sequence_warning:
        allowed_statuses.add("DIAGNOSTIC_WARNING" if continuation else "DIAGNOSTIC_STOP")
    if (
        result.get("status") not in allowed_statuses
        or result.get("arm") != arm
        or result.get("step") != step
    ):
        raise ValueError("R4 actual step result is incomplete or alarmed")
    if (
        entry.get("checkpoint_sha256") != files[str(path.relative_to(root))]
        or result.get("checkpoint_path") != "checkpoint.pt"
    ):
        raise ValueError("R4 checkpoint file hash/relative identity changed")
    post = load_checkpoint(path, expected_checkpoint)
    policy = _policy(post)
    for key in (*policy, "checkpoint_identity", "checkpoint_sha256"):
        if key in policy:
            if entry.get(key) != policy[key] or result.get(key) != policy[key]:
                raise ValueError("R4 checkpoint state/parameter/Adam hashes changed")
        elif result.get(key) != entry[key]:
            raise ValueError("R4 result/checkpoint manifest binding changed")
    _check_state(post, pre, arm, step, plan_hash, keys, [r["sample_key"] for r in rows])
    if (
        result.get("sample_keys") != [r["sample_key"] for r in rows]
        or result.get("control_sequences_scored") != 48
    ):
        raise ValueError("R4 update trajectory/probe coverage changed")
    _check_update(result, rows, arm, pre, post)
    for name, key in (("update.json", "update"), ("group_statistics.json", "group_statistics")):
        _equal(
            _read(attempt / name), result[key], "R4 update result disagrees with measured artifact"
        )
    parity = _score_store(
        root,
        files,
        attempt / "preupdate_parity",
        rows,
        {**unit, "unit": "preupdate_parity"},
        _policy(pre),
    )
    errors = [
        abs(a - b)
        for r in rows
        for a, b in zip(parity[r["sample_key"]], r["old_logprobs"], strict=True)
    ]
    parity_report = {
        "mean_abs_token_error": float(np.mean(errors)),
        "p99_abs_token_error": float(np.quantile(errors, 0.99)),
        "tokens": len(errors),
    }
    if parity_report["mean_abs_token_error"] > 0.02 or parity_report["p99_abs_token_error"] > 0.1:
        raise ValueError("R4 measured on-policy parity exceeded fixed limits")
    _equal(result["parity"], parity_report, "R4 parity summary differs from measured token scores")
    _equal(
        _read(attempt / "parity.json"),
        parity_report,
        "R4 parity artifact differs from measured token scores",
    )
    scores = _score_store(
        root,
        files,
        attempt / "control_scores",
        probe,
        {**unit, "unit": "fixed_step0_control", "poststate_hash": policy["state_hash"]},
        policy,
    )
    diagnostic = control_kl_diagnostic(probe, scores, eos_token_ids=eos)
    stored_diagnostic = result["control_diagnostic"]
    roundoff = _compare_kl_diagnostic(stored_diagnostic, diagnostic)
    _equal(
        _read(attempt / "control_diagnostic.json"),
        stored_diagnostic,
        "R4 measured KL artifact differs from its original result",
    )
    if diagnostic["should_stop"]:
        if (
            not allow_sequence_warning
            or diagnostic["alarms"] != {"mean_token_kl": False, "sequence_log_ratio_p99_abs": True}
            or diagnostic["status"] != "STOP_DIAGNOSE"
        ):
            raise ValueError("R4 fixed control KL alarm must block warm acceptance")
        expected_status = "DIAGNOSTIC_WARNING" if continuation else "DIAGNOSTIC_STOP"
        if result["status"] != expected_status:
            raise ValueError("R4 measured diagnostic warning cannot be relabelled PASS")
        if continuation is not None:
            _equal(
                _read(_path(root, files, attempt / "reviewed_warning.json")),
                {
                    "continuation_hash": continuation["continuation_hash"],
                    "reviewed_warning_policy": continuation["reviewed_warning_policy"],
                    "arm": arm,
                    "step": step,
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "control_diagnostic": stored_diagnostic,
                },
                "R4 warning lacks its bound post-diagnostic continuation decision",
            )
    elif diagnostic["status"] != "WITHIN_ENGINEERING_LIMITS" or result["status"] != "PASS":
        raise ValueError("R4 step status does not reflect the measured diagnostic")
    if roundoff and roundoff_observer is not None:
        roundoff_observer(
            {
                "arm": arm,
                "step": step,
                "comparison_contract": KL_FLOAT_COMPARISON,
                "differences": roundoff,
            }
        )
    return post, {**result, "attempt": str(attempt), "reused_completed_unit": False}


def _initial_rows(root, gate, runtime, plan, origin, adapter, shared):
    """Replay historical alignment with independently reconstructed CPU inputs."""
    from .r4_runtime import _historical_initial

    class ParameterView:
        requires_grad = True

        def __init__(self, tensor):
            self.tensor = tensor

        def detach(self):
            return self.tensor

    context = SimpleNamespace(
        model=SimpleNamespace(
            named_parameters=lambda: [
                (n, ParameterView(t)) for n, t in origin["parameters"].items()
            ]
        ),
        model_id=gate["config"]["model"]["id"],
        revision=gate["config"]["model"]["revision"],
        audit=runtime["model_audit"],
        processor=adapter.processor,
        eos_ids=gate["certificate"]["model_audit"]["eos_token_ids"],
        prepare=adapter.prepare,
    )
    historical, alignment = _historical_initial(
        context, plan, gate["binding"]["r0_dir"], gate["config"], gate["config"]["data_root"]
    )
    _equal(
        _read(root / "initial_alignment.json"),
        alignment,
        "R4 historical initial alignment was not reproduced",
    )
    historical.setdefault("N", shared)
    return historical


def _audit_r4(
    r4_dir,
    gate,
    r2_binding,
    r3_binding,
    *,
    stopped=False,
    recorded_root=None,
    roundoff_observer=None,
):
    from .next_stage_runtime import _stage
    from .r4_runtime import _load_plan

    root = Path(r4_dir).resolve()
    roundoff_observations = []
    preliminary = (
        _read(root / "runtime_lock.json") if (root / "runtime_lock.json").is_file() else {}
    )
    continuation = preliminary.get("continuation")
    if stopped and continuation is not None:
        raise ValueError("Only an original R4 stopped prefix can seed this continuation contract")
    parent = parent_runtime = parent_root = parent_files = None
    if continuation is not None:
        from .r4_continuation import verify_continuation

        parent_root = root / "inherited_parent"
        declared = continuation["parent_binding"]
        parent = audit_stopped_r4(
            parent_root, gate, r2_binding, r3_binding, _recorded_root=declared["root"]
        )
        _equal(parent, declared, "R4 continuation parent differs from complete stopped audit")
        context = verify_continuation(root, parent, preliminary["source"])
        _equal(context["runtime_binding"], continuation, "R4 continuation runtime contract changed")
        parent_runtime = context["parent_runtime"]
        parent_files = parent["files"]
    required = (
        "runtime_lock.json",
        "gate_binding.json",
        "two_arm_training_config.json",
        "origin.pt",
        "checkpoint_manifest.json",
        "initial_alignment.json",
        "runtime_profile.json",
        "step32_metrics.json",
        "endpoint_metrics.json",
        "L_family_equal_sensitivity.json",
        "N_L_OOD_effects.csv",
        "learning_curves.csv",
        "pilot_report.md",
        "report_zh.md",
    )
    recorded_root = Path(recorded_root).resolve() if recorded_root is not None else root
    if stopped:
        from .finalize_r0 import verify_manifest

        required = (
            *required[:7],
            "identity.json",
            "alarm_stop.json",
            "runtime_profile.json",
            "learning_curves.csv",
        )
        files = verify_manifest(root, required)
        status = _read(root / "status.json")
        if (
            status.get("status") != "BLOCKED"
            or status.get("phase") != "R4"
            or status.get("execution_kind") != KIND
        ):
            raise ValueError("R4 stopped audit requires preserved BLOCKED real CUDA evidence")
    elif continuation is not None:
        from .finalize_r0 import verify_manifest

        required = (
            *(n for n in required if n != "origin.pt"),
            "continuation_binding.json",
            "continuation_decision.json",
            "logical_sampling_identity.json",
            "diagnostic_warnings.json",
            "inherited_parent/origin.pt",
            "identity.json",
        )
        files = verify_manifest(root, required)
        status = _read(root / "status.json")
        if (
            status.get("status") != "COMPLETED_WITH_DIAGNOSTIC_WARNINGS"
            or status.get("phase") != "R4"
            or status.get("execution_kind") != KIND
        ):
            raise ValueError(
                "R4 continuation requires complete real CUDA evidence with preserved warnings"
            )
    else:
        status, files = _stage(root, "R4", KIND, required)
    forbidden = ("measurement_fault.json", "policy_contamination.json")
    for name in forbidden + (() if stopped else ("alarm_stop.json",)):
        if (root / name).exists():
            raise ValueError(
                "R4 preserved measurement fault, contamination or alarm blocks acceptance"
            )
    runtime, training, checkpoint_manifest = (
        _read(root / n)
        for n in ("runtime_lock.json", "two_arm_training_config.json", "checkpoint_manifest.json")
    )
    _check_source(runtime, gate)
    _check_finished_source(status, runtime)
    identity, config = runtime["identity"], gate["config"]
    if stopped or continuation is not None:
        _equal(
            _read(root / "identity.json"), identity, "R4 root ledger identity differs from runtime"
        )
    from .r1_reference_smoke import _helper_config

    _helper_config(config)
    if any(
        config["R4"].get(k) != v
        for k, v in {
            "steps": 64,
            "B": 4,
            "K": 8,
            "seed": 17,
            "microbatch": 1,
            "checkpoint_steps": [0, 16, 32, 64],
            "arms": {"X_BASE": {"lambda": 0.0}, "X_VALID": {"lambda": 1.0}},
        }.items()
    ):
        raise ValueError("R4 fixed two-arm training protocol changed")
    _equal(
        _read(root / "gate_binding.json"),
        {"r0_r1": gate["binding"], "r2": r2_binding, "r3_cold": r3_binding},
        "R4 upstream evidence binding changed",
    )
    if (
        runtime["config"] != config
        or training["canonical_config"] != config
        or runtime["environment"] != r2_binding["environment"]
        or runtime["environment"] != r3_binding["environment"]
    ):
        raise ValueError("R4 canonical config/environment differs from R0/R1/R2/R3")
    plan, data, legacy_lock = _load_plan(config["data_root"], gate, gate["binding"]["r0_dir"])
    _equal(training["plan"], plan, "R4 prompt plan differs from immutable source selection")
    _equal(training["data_binding"], data, "R4 data lock differs from audited source manifests")
    expected_identity = {
        "phase": "R4",
        "model_hash": canonical_hash(config["model"]),
        "config_hash": canonical_hash(config),
        "data_hash": canonical_hash(data),
        "source_hash": canonical_hash(runtime["source"]),
        "gate_hash": canonical_hash(gate["binding"]),
        "r2_gate_hash": canonical_hash(r2_binding),
        "r3_cold_gate_hash": canonical_hash(r3_binding),
        "plan_hash": plan["plan_hash"],
        "initial_adapter_hash": gate["certificate"]["initial_adapter_hash"],
        "execution_kind": KIND,
    }
    if continuation is not None:
        expected_identity["continuation_hash"] = continuation["continuation_hash"]
    _equal(identity, expected_identity, "R4 source/model/initialization identity changed")
    _equal(checkpoint_manifest["identity"], identity, "R4 checkpoint manifest identity changed")
    authorization = {
        "allow_training_flag": True,
        "canonical_config_allow_training": config["R4"]["allow_training"],
    }
    if (
        runtime.get("authorization") != authorization
        or training.get("authorization") != authorization
        or runtime.get("selected_probability_path") != "uncached_prefix_recompute"
        or runtime.get("scheduler") is not None
        or runtime.get("gradient_scaler") is not None
    ):
        raise ValueError("R4 explicit authorization or certified execution path missing")
    for key in ("processor_hash", "tokenizer_hash", "chat_template_hash", "frozen_parameter_hash"):
        if runtime["model_audit"].get(key) != gate["certificate"]["model_audit"].get(key):
            raise ValueError("R4 loaded processor/model differs from R1 certificate")
    origin_root = parent_root if continuation is not None else root
    origin_files = parent_files if continuation is not None else files
    origin_identity = {
        **(parent_runtime["identity"] if continuation is not None else identity),
        "unit": "initial_origin",
    }
    if continuation is not None:
        _equal(
            runtime.get("origin_identity"),
            origin_identity,
            "R4 continuation must retain the original checkpoint identity",
        )
    origin = load_checkpoint(origin_root / "origin.pt", origin_identity)
    _check_state(origin, origin, "INITIAL", 0, plan["plan_hash"], [], [])
    if (
        runtime["model_audit"].get("execution_kind") != "REAL_CUDA_INFERENCE"
        or not origin["rng"]["cuda"]
    ):
        raise ValueError("R4 initial model/RNG evidence is not measured CUDA")
    _equal(
        runtime["model_audit"],
        {
            k: v
            for k, v in gate["certificate"]["model_audit"].items()
            if k not in ("load_seconds", "load_peak_cuda_bytes")
        },
        "R4 actual model architecture/LoRA/dtype differs from R1 certificate",
    )
    if (
        any(
            runtime.get(k) != v
            for k, v in {
                "origin_hash": state_hash(origin),
                "origin_file_sha256": origin_files["origin.pt"],
                "initial_adapter_hash": identity["initial_adapter_hash"],
                "optimizer_initial_hash": state_hash(origin["optimizer"]),
                "frozen_base_hash": gate["certificate"]["model_audit"]["frozen_parameter_hash"],
            }.items()
        )
        or _parameter_hash(origin) != identity["initial_adapter_hash"]
    ):
        raise ValueError("R4 origin is not the certified fresh adapter and empty Adam")
    entries = checkpoint_manifest["checkpoints"]
    mapping = {(e["arm"], e["step"]): e for e in entries}
    stop_record = _read(root / "alarm_stop.json") if stopped else None
    if stopped:
        stop_arm, stop_step = stop_record.get("arm"), stop_record.get("step")
        if stop_arm not in ARMS or type(stop_step) is not int or stop_step not in range(1, 65):
            raise ValueError("R4 preserved stop has invalid arm/step")
        limits = {
            a: (64 if ARMS.index(a) < ARMS.index(stop_arm) else stop_step)
            for a in ARMS[: ARMS.index(stop_arm) + 1]
        }
    else:
        limits = {a: 64 for a in ARMS}
    expected_steps = {(a, s) for a, limit in limits.items() for s in range(limit + 1)}
    if len(entries) != len(expected_steps) or set(mapping) != expected_steps:
        raise ValueError("R4 checkpoint chain is incomplete, duplicated or extends past the stop")
    if any(e.get("milestone") is not (e["step"] in (0, 16, 32, 64)) for e in entries):
        raise ValueError("R4 checkpoint milestones changed")
    probe = _probe(training, r3_binding, origin)
    from .r2_result_audit import load_processor_adapter, validate_processor_certificate

    validate_processor_certificate(runtime, gate)
    processor_adapter = load_processor_adapter(runtime)
    tokenizer = processor_adapter.processor.tokenizer
    eos = gate["certificate"]["model_audit"]["eos_token_ids"]
    inputs = _PreparedInputs(processor_adapter, config["data_root"])
    used_stores = set()

    parent_mapping = (
        {(e["arm"], e["step"]): e for e in parent["checkpoint_manifest"]["checkpoints"]}
        if continuation is not None
        else {}
    )

    def inherited_sample(arm, step, role):
        if continuation is None:
            return False
        if arm == "INITIAL":
            return True
        target = step + 1 if role == "train" else step
        return (arm, target) in parent_mapping and not (
            role == "evaluation"
            and (arm, target) == (parent["stop"]["arm"], parent["stop"]["step"])
        )

    def sample(path, prompts, state, arm, step, role):
        inherited = inherited_sample(arm, step, role)
        sample_root = parent_root if inherited else root
        sample_files = parent_files if inherited else files
        sample_identity = parent_runtime["identity"] if inherited else identity
        if inherited:
            path = parent_root / path.relative_to(root)
        used_stores.add(str((path / "samples.jsonl").relative_to(root)))
        return _raw_store(
            sample_root,
            sample_files,
            path,
            prompts,
            sample_identity,
            state,
            arm,
            step,
            role,
            config,
            legacy_lock,
            tokenizer,
            eos,
            inputs,
            sampling_identity=parent_runtime["identity"]
            if continuation is not None and not inherited
            else None,
        )

    shared = sample(
        root / "evaluation/INITIAL/step_00/N",
        plan["dev_panel_prompts"],
        origin,
        "INITIAL",
        0,
        "evaluation",
    )
    train_prompts = {p["prompt_id"]: p for p in plan["train_prompts"]}
    summaries, endpoint, panel32, training_rows = [], [], [], []
    for arm, limit in limits.items():
        zero = mapping[(arm, 0)]
        _equal(
            zero["checkpoint_identity"],
            origin_identity,
            "R4 arms must share the exact fresh origin",
        )
        if _path(
            root, files, zero["checkpoint_path"], recorded_root=recorded_root
        ) != origin_root / "origin.pt" or zero["state_hash"] != state_hash(origin):
            raise ValueError("R4 arms do not share the same initialization")
        if continuation is not None and zero.get("source_segment") != "parent":
            raise ValueError("R4 continuation step0 must bind the inherited source segment")
        state, keys = origin, []
        for step, prompt_ids in enumerate(plan["train_steps"][:limit], 1):
            rows = sample(
                root / arm / f"step_{step:02d}/rollouts",
                [train_prompts[p] for p in prompt_ids],
                state,
                arm,
                step - 1,
                "train",
            )
            keys.extend(r["sample_key"] for r in rows)
            entry = mapping[(arm, step)]
            inherited = continuation is not None and (arm, step) in parent_mapping
            if continuation is not None:
                expected_segment = "parent" if inherited else "current"
                if entry.get("source_segment") != expected_segment:
                    raise ValueError("R4 checkpoint source segment differs from the audited prefix")
                if inherited:
                    original_entry = parent_mapping[(arm, step)]
                    original_path = Path(original_entry["checkpoint_path"])
                    relative = (
                        original_path.relative_to(parent["root"])
                        if original_path.is_absolute()
                        else original_path
                    )
                    _equal(
                        entry,
                        {
                            **original_entry,
                            "source_segment": "parent",
                            "checkpoint_path": str(Path("inherited_parent") / relative),
                        },
                        "R4 inherited checkpoint was rewritten or omitted",
                    )
            state, summary = _step_evidence(
                parent_root if inherited else root,
                parent_files if inherited else files,
                parent_mapping[(arm, step)] if inherited else entry,
                parent_runtime["identity"] if inherited else identity,
                state,
                rows,
                probe,
                plan["plan_hash"],
                keys,
                eos,
                allow_sequence_warning=(stopped and (arm, step) == (stop_arm, stop_step))
                or continuation is not None,
                recorded_root=parent["root"] if inherited else recorded_root,
                continuation=continuation if continuation is not None and not inherited else None,
                roundoff_observer=roundoff_observations.append,
            )
            if continuation is not None:
                summary["source_segment"] = "parent" if inherited else "current"
            training_rows.extend(rows)
            summaries.append(summary)
            if step == 32 and not (stopped and (arm, step) == (stop_arm, stop_step)):
                panel32.extend(
                    sample(
                        root / "evaluation" / arm / "step_32/N",
                        plan["dev_panel_prompts"],
                        state,
                        arm,
                        step,
                        "evaluation",
                    )
                )
            if step == 64 and not (stopped and (arm, step) == (stop_arm, stop_step)):
                for track, key in (
                    ("N", "dev_prompts"),
                    ("L", "legacy_prompts"),
                    ("OOD", "ood_prompts"),
                ):
                    endpoint.extend(
                        sample(
                            root / "evaluation" / arm / f"step_64/{track}",
                            plan[key],
                            state,
                            arm,
                            step,
                            "evaluation",
                        )
                    )
        del state
    raw = [*shared, *training_rows, *panel32, *endpoint]
    if not stopped and (
        len(raw) != 14400
        or len({r["sample_key"] for r in raw}) != 14400
        or len(training_rows) != 4096
        or len(panel32) != 1152
        or len(endpoint) != 8576
    ):
        raise ValueError("R4 full raw output budget or independent-arm trajectory coverage changed")
    for name in files:
        if (
            name.endswith(".jsonl")
            and name not in used_stores
            and any(
                "raw_completion" in json.loads(line)
                for line in (root / name).read_text().splitlines()
            )
        ):
            raise ValueError("R4 contains unaccounted generated raw outputs")
    if stopped:
        final = summaries[-1]
        if final["status"] != "DIAGNOSTIC_STOP":
            raise ValueError("R4 stopped audit requires the retained completed alarming update")
        for key in (
            "arm",
            "step",
            "status",
            "checkpoint_identity",
            "checkpoint_sha256",
            "state_hash",
            "control_diagnostic",
        ):
            _equal(
                stop_record.get(key),
                final.get(key),
                "R4 stop marker disagrees with completed measured update",
            )
        if len({r["sample_key"] for r in raw}) != len(raw):
            raise ValueError("R4 stopped prefix repeats sampled identities")
        cost = _read(root / "runtime_profile.json")
        _check_runtime_profile(root, files, cost, runtime, completion_status="BLOCKED")
        updates = len(summaries)
        if (
            cost.get("optimizer_steps_observed_at_least", 0) < updates
            or cost.get("backward_calls_observed_at_least", 0) < updates * 32
        ):
            raise ValueError("R4 stopped prefix lacks measured optimizer/backward counters")
        if status.get("details", {}).get("final_scratch_origin_restored") is not True:
            raise ValueError("R4 stopped prefix did not restore its scratch origin")
        _equal(
            status["details"].get("runtime_counts"),
            {k: v for k, v in cost.items() if k != "invocations"},
            "R4 stopped aggregate counters changed",
        )
        _initial_rows(root, gate, runtime, plan, origin, processor_adapter, shared)
        from .r4_report_gate import _check_csv, _curves

        _check_csv(root, "learning_curves.csv", _curves(summaries), ("arm", "step"))
        binding = {
            "status": "PASS_STOPPED_AUDIT",
            "root": str(recorded_root),
            "files": files,
            "manifest_sha256": file_hash(root / "manifest.json"),
            "runtime_lock_sha256": files["runtime_lock.json"],
            "alarm_stop_sha256": files["alarm_stop.json"],
            "source": runtime["source"],
            "identity": identity,
            "environment": runtime["environment"],
            "plan_hash": plan["plan_hash"],
            "stop": {
                "arm": stop_arm,
                "step": stop_step,
                "checkpoint": mapping[(stop_arm, stop_step)],
                "diagnostic": final["control_diagnostic"],
            },
            "checkpoint_manifest": checkpoint_manifest,
            "inherited_counts": {
                "optimizer_updates": updates,
                "training_rollouts": len(training_rows),
                "shared_step0_outputs": len(shared),
                "step32_outputs": len(panel32),
                "step64_outputs": len(endpoint),
                "new_outputs": len(raw),
            },
            "artifact_paths": sorted(files),
            "r3_cold_gate_hash": canonical_hash(r3_binding),
        }
        binding["audit_hash"] = canonical_hash({k: v for k, v in binding.items() if k != "root"})
        if roundoff_observer is not None:
            for observation in roundoff_observations:
                roundoff_observer(observation)
        return binding
    required_counts = {
        "distinct_optimizer_updates": 128,
        "training_rollouts": 4096,
        "shared_step0_outputs": 576,
        "step32_outputs": 1152,
        "step64_outputs": 8576,
        "new_outputs": 14400,
        "fixed_control_sequence_scores": 6144,
        "preupdate_parity_sequence_scores": 4096,
        "postupdate_training_sequence_scores": 4096,
        "new_control_outputs": 0,
        "arms": {a: 64 for a in ARMS},
        "final_scratch_origin_restored": True,
    }
    details, cost = status["details"], _read(root / "runtime_profile.json")
    if continuation is not None:
        parent_cost = _read(parent_root / "runtime_profile.json")
        inherited_cost = {k: v for k, v in parent_cost.items() if k != "invocations"}
        _equal(
            cost.get("inherited_runtime_counts"),
            inherited_cost,
            "R4 inherited execution costs changed",
        )
        current_cost = copy.deepcopy(cost)
        for key in (
            "optimizer_steps_observed_at_least",
            "backward_calls_observed_at_least",
            "incomplete_invocations_with_unknown_extra_cost",
        ):
            current_cost[key] -= inherited_cost[key]
        _check_runtime_profile(
            root,
            files,
            current_cost,
            runtime,
            completion_status="COMPLETED_WITH_DIAGNOSTIC_WARNINGS",
        )
    else:
        _check_runtime_profile(root, files, cost, runtime)
    if (
        any(details.get(k) != v for k, v in required_counts.items())
        or cost.get("optimizer_steps_observed_at_least", 0) < 128
        or cost.get("backward_calls_observed_at_least", 0) < 4096
    ):
        raise ValueError("R4 measured counters/final origin restoration are incomplete")
    _equal(
        details["runtime_counts"],
        {k: v for k, v in cost.items() if k != "invocations"},
        "R4 observed cost counts changed",
    )
    historical = _initial_rows(root, gate, runtime, plan, origin, processor_adapter, shared)
    from .r4_report_gate import validate_r4_response_artifacts

    if continuation is not None:
        expected_warnings = [
            {
                "arm": s["arm"],
                "step": s["step"],
                "status": s["status"],
                "control_diagnostic": s["control_diagnostic"],
                "source_segment": s["source_segment"],
                "continuation_hash": continuation["continuation_hash"],
            }
            for s in summaries
            if s["status"] != "PASS"
        ]
        _equal(
            _read(root / "diagnostic_warnings.json"),
            expected_warnings,
            "R4 complete warning history changed or an alarm was hidden",
        )
    report_options = {"continuation": continuation} if continuation is not None else {}
    report_audit = validate_r4_response_artifacts(
        root,
        endpoint_rows=endpoint,
        step32_rows=panel32,
        shared_initial_rows=shared,
        initial_rows_by_track=historical,
        step_summaries=summaries,
        **report_options,
    )
    warm = mapping[("X_BASE", 64)]
    _equal(
        checkpoint_manifest["X_BASE_step64_for_R3_warm"],
        warm,
        "R4 warm pointer must be the complete X_BASE step64 manifest entry",
    )
    path = _path(root, files, warm["checkpoint_path"])
    binding = {
        "status": "PASS",
        "root": str(root),
        "r4_dir": str(root),
        "files": files,
        "r4_files": files,
        "environment": runtime["environment"],
        "runtime_lock_sha256": files["runtime_lock.json"],
        "source_commit": runtime["source"]["source_commit"],
        "plan_hash": plan["plan_hash"],
        "initial_adapter_hash": runtime["initial_adapter_hash"],
        "r3_cold_plan_hash": r3_binding["plan_hash"],
        "r3_cold_gate_hash": canonical_hash(r3_binding),
        "report_audit": report_audit,
        **(
            {
                "completion_status": "COMPLETED_WITH_DIAGNOSTIC_WARNINGS",
                "continuation": continuation,
                "diagnostic_warnings_sha256": files["diagnostic_warnings.json"],
            }
            if continuation is not None
            else {}
        ),
        "warm_checkpoint": {
            **copy.deepcopy(warm),
            "path": str(path),
            "identity": copy.deepcopy(warm["checkpoint_identity"]),
            "file_sha256": files[str(path.relative_to(root))],
        },
    }

    if roundoff_observer is not None:
        for observation in roundoff_observations:
            roundoff_observer(observation)
    return binding


def audit_stopped_r4(
    r4_dir,
    gate,
    r2_binding,
    r3_binding,
    *,
    _recorded_root=None,
    roundoff_observer=None,
):
    """Verify the entire stopped prefix; this is never a warm-training PASS."""
    return _audit_r4(
        r4_dir,
        gate,
        r2_binding,
        r3_binding,
        stopped=True,
        recorded_root=_recorded_root,
        roundoff_observer=roundoff_observer,
    )


def validate_r4_gate(r4_dir, gate, r2_binding, r3_binding, *, roundoff_observer=None):
    return _audit_r4(r4_dir, gate, r2_binding, r3_binding, roundoff_observer=roundoff_observer)
