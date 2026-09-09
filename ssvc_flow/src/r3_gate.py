"""Read-only acceptance of bound, complete real cold-fork evidence before R4."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .core import canonical_hash, file_hash
from .optimizer_fork import load_checkpoint, state_hash


def _read(path):
    return json.loads(Path(path).read_text())


def _ledger(path):
    rows = {}
    with Path(path).open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError("R3 has a truncated measured ledger")
            row = json.loads(line)
            key = row["sample_key"]
            if key in rows:
                raise ValueError("R3 has a duplicate measured sample")
            if (
                row.get("execution_kind") != "REAL_CUDA_FORK"
                or row.get("execution_checks", {}).get("passed") is not True
            ):
                raise ValueError("R3 has fake or failed measured output")
            if row.get("record_hash") != canonical_hash(
                {k: v for k, v in row.items() if k != "record_hash"}
            ):
                raise ValueError("R3 measured content hash mismatch")
            rows[key] = row
    return rows


def _check_reuse(decision, origin_hash):
    values = decision["validation"]
    if decision.get("validation_bank_indices") != [1, 11] or set(values) != {"1", "11"}:
        raise ValueError("R3 preregistered reuse bank coverage changed")
    for result in values.values():
        checks = result.get("checks", {})
        if (
            result.get("origin_state_hash") != origin_hash
            or result.get("validation_optimizer_updates") != 4
            or set(checks) != {"0.0", "1.0"}
        ):
            raise ValueError("R3 reuse validation state or actual probes missing")
        if result.get("adoptable") is not all(c.get("passed") is True for c in checks.values()):
            raise ValueError("R3 reuse permission disagrees with gradient/Adam probes")
    adopted = all(r["adoptable"] is True for r in values.values())
    if decision.get("reuse_authorized") is not adopted:
        raise ValueError("R3 reuse permission requires both informative validation banks")
    return adopted


def _check_scores(root, identity, candidate, proposal):
    cid = candidate["candidate_id"]
    if "/" in cid or cid in {".", ".."}:
        raise ValueError("R3 candidate identifier escapes score root")
    score_root = root / "control_scores" / cid
    expected = {
        **identity,
        "unit": "control_likelihood",
        "candidate_id": cid,
        "candidate_parameter_hash": candidate["parameter_hash"],
        "candidate_optimizer_hash": candidate["optimizer_state_hash"],
        "proposal_hash": canonical_hash([r["record_hash"] for r in proposal.values()]),
    }
    if _read(score_root / "identity.json") != expected:
        raise ValueError("R3 control score identity differs from candidate/proposal")
    rows = _ledger(score_root / "samples.jsonl")
    if len(rows) != 768 or {r["proposal_sample_key"] for r in rows.values()} != set(proposal):
        raise ValueError("R3 control score coverage is incomplete")
    for row in rows.values():
        source = proposal[row["proposal_sample_key"]]
        scores = row.get("candidate_token_logprobs", [])
        if (
            row["sample_key"] != canonical_hash([expected, source["sample_key"]])
            or row.get("candidate_id") != cid
            or row.get("candidate_parameter_hash") != candidate["parameter_hash"]
            or row.get("proposal_record_hash") != source["record_hash"]
            or row.get("token_ids") != source["token_ids"]
            or len(scores) != len(source["token_ids"])
            or not scores
            or any(
                not isinstance(v, (int, float)) or not math.isfinite(v) or v > 1e-5 for v in scores
            )
        ):
            raise ValueError("R3 control score data or candidate binding changed")
    return len(rows)


def _check_optimizer_state(state, *, step):
    import torch

    groups = state["optimizer"]["param_groups"]
    expected = {"lr": 1e-5, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}
    if not groups or any(any(group.get(k) != v for k, v in expected.items()) for group in groups):
        raise ValueError("R3 checkpoint optimizer hyperparameters differ from fixed AdamW")
    indices = [i for group in groups for i in group["params"]]
    if len(indices) != len(set(indices)) or len(indices) != len(state["parameters"]):
        raise ValueError("R3 optimizer does not own every trainable parameter exactly once")
    if any(not torch.isfinite(p).all() for p in state["parameters"].values()):
        raise ValueError("R3 candidate contains nonfinite trainable parameters")
    moments = state["optimizer"]["state"]
    if step == 0:
        if moments:
            raise ValueError("R3 cold origin Adam must be empty")
        return
    if set(moments) != set(indices):
        raise ValueError("R3 candidate Adam moments do not cover all trainable parameters")
    for index in indices:
        values = moments[index]
        if float(values["step"]) != step or any(
            not torch.isfinite(values[k]).all() for k in ("step", "exp_avg", "exp_avg_sq")
        ):
            raise ValueError("R3 candidate Adam step or finite moments check failed")


def _load_tokenizer(config, audit):
    """Load only the pinned, already-cached processor; never model weights/network."""
    from transformers import AutoProcessor

    from .model_adapters.base import _hash_json

    model = config["model"]
    if not re.fullmatch(r"[0-9a-f]{40}", model.get("revision", "")):
        raise ValueError("R3 token audit requires a pinned model revision")
    tokenizer = AutoProcessor.from_pretrained(
        model["id"], revision=model["revision"], trust_remote_code=False, local_files_only=True
    ).tokenizer
    if _hash_json(tokenizer.get_vocab()) != audit.get("tokenizer_hash"):
        raise ValueError("R3 tokenizer vocabulary differs from the R1 certificate")
    return tokenizer


def _valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _check_raw_rows(records, plan, identity, config, origin, model_audit):
    import torch

    from .model_adapters.base import _hash_json
    from .r2_runtime import annotate_diagnostic, generation_checks

    tokenizer = _load_tokenizer(config, model_audit)
    eos = model_audit.get("eos_token_ids")
    if not isinstance(eos, list) or not eos or any(type(t) is not int or t < 0 for t in eos):
        raise ValueError("R3 certificate has no valid sampled EOS identity")
    prompts = {r["prompt_id"]: r for r in [*plan["train_prompts"], *plan["control_prompts"]]}
    input_bindings = {}
    hash_fields = (
        "final_prompt_hash",
        "tokenized_prompt_hash",
        "input_ids_hash",
        "input_tensor_hash",
        "prepared_hash",
    )
    prompt_fields = (
        *hash_fields,
        "final_prompt",
        "final_prompt_token_ids",
        "prompt_token_count",
        "image_token_count",
        "actual_image_tokens",
        "pixel_values_hash",
        "image_hash",
    )
    for row in records.values():
        prompt = prompts[row["prompt_id"]]
        scene = prompt["scene"]
        raw, tokens, old = row.get("raw_completion"), row.get("token_ids"), row.get("old_logprobs")
        if (
            not isinstance(raw, str)
            or row.get("raw_text") != raw
            or not isinstance(tokens, list)
            or row.get("raw_token_ids") != tokens
            or row.get("n_generated_tokens") != len(tokens)
            or not isinstance(old, list)
            or not old
            or len(old) != len(tokens)
            or any(type(v) not in (int, float) or not math.isfinite(v) or v > 1e-5 for v in old)
            or row.get("behavior_token_logprobs") != old
            or row.get("per_token_logprob_behavior") != old
            or row.get("logprob_sequence") != math.fsum(old)
        ):
            raise ValueError(
                "R3 raw text/token/frozen behavior log-probability evidence is incomplete"
            )
        faults = generation_checks(row, tokenizer, eos, 64)
        if faults:
            raise ValueError("R3 raw generation evidence failed: " + "; ".join(faults))
        expected = {
            "run_id": canonical_hash(identity),
            "model_id": config["model"]["id"],
            "model_revision": config["model"]["revision"],
            "adapter_hash": identity["initial_adapter_hash"],
            "optimizer_state_hash": state_hash(origin["optimizer"]),
            "protocol_version": config["protocol_version"],
            "train_seed": 17,
            "generation_config_hash": canonical_hash(config["generation_proposed_N"]),
            "solution_count": 1,
            **{
                key: scene[key]
                for key in (
                    "truth_world",
                    "observed_world",
                    "changed_index",
                    "cue",
                    "chart_type",
                    "operation",
                )
            },
            **annotate_diagnostic(raw, scene),
        }
        if any(key not in row or row[key] != value for key, value in expected.items()):
            raise ValueError("R3 raw annotation or cold policy/source-scene binding changed")
        if row.get("execution_checks", {}).get("faults") != []:
            raise ValueError("R3 raw execution faults cannot be hidden behind a passed flag")
        if any(not _valid_hash(row.get(key)) for key in hash_fields):
            raise ValueError("R3 prepared input hashes are missing or malformed")
        text, ids = row.get("final_prompt"), row.get("final_prompt_token_ids")
        if (
            not isinstance(text, str)
            or not text.rstrip().endswith("</think>")
            or any(prompt["prompt"][key] not in text for key in ("system", "user"))
            or row["final_prompt_hash"] != _hash_json(text)
            or not isinstance(ids, list)
            or not ids
            or any(type(t) is not int or t < 0 for t in ids)
            or row.get("prompt_token_count") != len(ids)
            or row["tokenized_prompt_hash"] != state_hash(torch.tensor([ids], dtype=torch.long))
            or row["input_ids_hash"] != row["tokenized_prompt_hash"]
        ):
            raise ValueError("R3 original non-thinking prompt text/token hashes changed")
        image = row["interface"] == "IMAGE_CUE_FRESH"
        image_count = row.get("image_token_count")
        if (
            type(image_count) is not int
            or image_count < 0
            or row.get("actual_image_tokens") != image_count
            or bool(image_count) != image
            or row.get("image_hash") != (scene["image_hash"] if image else None)
            or (
                image
                and (
                    not _valid_hash(row.get("pixel_values_hash"))
                    or row.get("vision_forward_calls", 0) < 1
                )
            )
            or (not image and row.get("pixel_values_hash") is not None)
        ):
            raise ValueError("R3 raw input lost the original image/visual-forward binding")
        binding = {key: row.get(key) for key in prompt_fields}
        if input_bindings.setdefault(row["prompt_id"], binding) != binding:
            raise ValueError("R3 same prompt has different input tensors across frozen rollouts")


def _bank_bindings(plan, records):
    """Recreate fork_gradients._bank_binding using recorded full prepared hashes."""
    import torch

    result = {}
    for index, prompts in enumerate(plan["banks"]):
        groups = [
            sorted(
                (r for r in records.values() if r["prompt_id"] == pid),
                key=lambda r: r["sample_index"],
            )
            for pid in prompts
        ]
        if len(groups) != 4 or any(
            [r["sample_index"] for r in group] != list(range(8))
            or any(r["bank_role"] != "train" or r["bank_index"] != index for r in group)
            for group in groups
        ):
            raise ValueError("R3 bank must contain its original four groups of eight rollouts")
        ordered = [r for group in groups for r in group]
        canonical = [
            {
                "prompt_id": r["prompt_id"],
                "sample_key": r["sample_key"],
                "category": r["category"],
                "token_ids": r["token_ids"],
                "old_logprobs": torch.as_tensor(r["old_logprobs"]).to(dtype=torch.float32).tolist(),
                "prepared_hash": r["prepared_hash"],
            }
            for r in ordered
        ]
        result[str(index)] = {
            "sample_hash": canonical_hash([r["record_hash"] for r in ordered]),
            "bank_hash": state_hash(canonical),
        }
    return result


def _check_reuse_bindings(root, files, decision, identity, origin_hash, bank_bindings):
    from .r3_runtime import _load_tensor_payload

    for index in (1, 11):
        result = decision["validation"][str(index)]
        attempt = Path(result.get("attempt", "")).resolve()
        if attempt.parent != (root / "validation" / f"bank_{index:02d}").resolve():
            raise ValueError("R3 reuse probe attempt is outside its registered bank")
        for name in ("identity.json", "result.json", "score_bank.pt"):
            path = (attempt / name).resolve()
            if not path.is_relative_to(root) or str(path.relative_to(root)) not in files:
                raise ValueError("R3 reuse probe evidence is absent from the stage manifest")
        expected = {
            **identity,
            "unit": "reuse_validation",
            "bank_index": index,
            "origin_hash": origin_hash,
            "sample_hash": bank_bindings[str(index)]["sample_hash"],
        }
        if _read(attempt / "identity.json") != expected or _read(attempt / "result.json") != {
            k: v for k, v in result.items() if k not in {"attempt", "reused_completed_unit"}
        }:
            raise ValueError("R3 reuse probe differs from its 32 frozen records/result")
        scores = _load_tensor_payload(
            attempt / "score_bank.pt", expected, result["score_bank_binding"]
        )
        audit = scores.get("audit", {})
        if any(
            audit.get(k) != v
            for k, v in {
                "bank_hash": bank_bindings[str(index)]["bank_hash"],
                "B": 4,
                "K": 8,
                "sequences": 32,
            }.items()
        ):
            raise ValueError("R3 reuse score bank differs from its frozen raw inputs/actions")
        del scores


def validate_r3_cold_gate(r3_cold_dir, gate, r2_binding):
    from .next_stage_runtime import _stage
    from .r3_runtime import LAMBDAS, build_requests, validate_sample_ledger

    root = Path(r3_cold_dir).resolve()
    status, files = _stage(
        root,
        "R3-cold",
        "REAL_CUDA_FORK",
        (
            "identity.json",
            "runtime_lock.json",
            "gate_binding.json",
            "bank_manifest.json",
            "samples.jsonl",
            "origin.pt",
            "reuse_decision.json",
            "candidate_manifest.json",
            "joint_advantage_checks.json",
            "gradients_summary.parquet",
            "paired_response.csv",
            "support_control_report.md",
            "response_bank_00.json",
            "response_bank_06.json",
            "runtime_profile.json",
        ),
    )
    if (root / "policy_contamination.json").exists():
        raise ValueError("R3 contains unresolved policy contamination")
    identity, runtime, bank = (
        _read(root / n) for n in ("identity.json", "runtime_lock.json", "bank_manifest.json")
    )
    expected_binding = {"r0_r1": gate["binding"], "r2": r2_binding}
    if (
        _read(root / "gate_binding.json") != expected_binding
        or runtime.get("config") != gate["config"]
        or runtime.get("environment") != r2_binding["environment"]
    ):
        raise ValueError("R3 original evidence/config/environment binding differs")
    if (
        identity.get("phase") != "R3-cold"
        or identity.get("execution_kind") != "REAL_CUDA_FORK"
        or identity.get("checkpoint_step") != 0
        or identity.get("gate_hash") != canonical_hash(gate["binding"])
        or identity.get("r2_gate_hash") != canonical_hash(r2_binding)
        or identity.get("config_hash") != canonical_hash(gate["config"])
        or identity.get("model_hash") != canonical_hash(gate["config"]["model"])
        or identity.get("source_hash") != canonical_hash(runtime["source"])
        or identity != runtime.get("identity")
        or identity != bank.get("identity")
        or runtime.get("selected_probability_path") != "uncached_prefix_recompute"
        or runtime.get("initial_adapter_hash") != gate["certificate"]["initial_adapter_hash"]
        or identity.get("initial_adapter_hash") != runtime.get("initial_adapter_hash")
        or runtime.get("frozen_base_hash")
        != gate["certificate"]["model_audit"]["frozen_parameter_hash"]
    ):
        raise ValueError("R3 cold initialization/model/source identity is incomplete")
    if files["origin.pt"] != runtime.get("origin_file_sha256"):
        raise ValueError("R3 origin checkpoint file binding changed")
    origin = load_checkpoint(root / "origin.pt", {**identity, "unit": "initial_origin"})
    _check_optimizer_state(origin, step=0)
    origin_hash = state_hash(origin)
    params_hash = state_hash({n: state_hash(t) for n, t in origin["parameters"].items()})
    if (
        origin_hash != runtime.get("origin_hash")
        or files["origin.pt"] != runtime.get("origin_file_sha256")
        or params_hash != runtime["initial_adapter_hash"]
        or origin["optimizer"]["state"]
        or state_hash(origin["optimizer"]) != runtime["optimizer_initial_hash"]
        or origin["scheduler"] is not None
    ):
        raise ValueError("R3 origin is not the certified fresh parameters/empty Adam")
    data = bank["data_binding"]
    if data.get("manifest_sha256") != gate["binding"]["data_manifest_sha256"] or identity.get(
        "data_hash"
    ) != canonical_hash(data):
        raise ValueError("R3 data manifest binding differs from passed R0")
    for name in ("train.jsonl", "control.jsonl"):
        if data["files"].get(name) != gate["binding"]["dataset_files"][name]["sha256"]:
            raise ValueError("R3 train/control data differs from passed R0")
    plan = bank["plan"]
    if (
        plan["plan_hash"] != identity["plan_hash"]
        or plan.get("reuse_check_bank_indices") != [1, 11]
        or plan.get("response_banks") != [0, 6]
        or len(plan["train_prompts"]) != 48
        or len(plan["control_prompts"]) != 48
        or len(plan["banks"]) != 12
    ):
        raise ValueError("R3 fixed panel/bank selection changed")
    if plan["plan_hash"] != canonical_hash({k: v for k, v in plan.items() if k != "plan_hash"}):
        raise ValueError("R3 stored plan content hash mismatch")
    from .r3_runtime import _load_plan

    expected_plan, expected_data = _load_plan(gate["config"]["data_root"], gate)
    if plan != expected_plan or data != expected_data:
        raise ValueError("R3 selected prompts differ from their R0 source data")
    expected_requests = build_requests(plan, {**identity, "origin_hash": origin_hash})
    if (
        bank.get("request_identity") != {**identity, "origin_hash": origin_hash}
        or bank.get("requests") != expected_requests
    ):
        raise ValueError("R3 request bank is not bound to its actual cold state")
    records = _ledger(root / "samples.jsonl")
    validate_sample_ledger(records, expected_requests)
    if len(records) != 1152 or set(records) != {r["sample_key"] for r in expected_requests}:
        raise ValueError("R3 output coverage is incomplete")
    _check_raw_rows(
        records, plan, identity, gate["config"], origin, gate["certificate"]["model_audit"]
    )
    bank_bindings = _bank_bindings(plan, records)
    proposal = {
        r["sample_key"]: records[r["sample_key"]]
        for r in expected_requests
        if r["bank_role"] == "control_proposal"
    }
    reuse = _read(root / "reuse_decision.json")
    allowed = _check_reuse(reuse, origin_hash)
    _check_reuse_bindings(root, files, reuse, identity, origin_hash, bank_bindings)
    candidate_manifest = _read(root / "candidate_manifest.json")
    banks = candidate_manifest["banks"]
    joint = _read(root / "joint_advantage_checks.json")
    if (
        candidate_manifest.get("distinct_candidates") != 60
        or candidate_manifest.get("committed_candidates") != 0
        or candidate_manifest.get("response_bank_indices") != [0, 6]
        or set(banks) != {str(i) for i in range(12)}
        or joint.get("reuse_validation") != reuse["validation"]
        or joint.get("bank_advantage_and_gradient_audits") != banks
    ):
        raise ValueError("R3 candidate/gradient manifest coverage differs")
    ids = set()
    score_count = 0
    for index, summary in banks.items():
        expected_unit = {
            **identity,
            "unit": "bank_forks",
            "bank_index": int(index),
            "origin_hash": origin_hash,
            "reuse_authorized": allowed,
            "sample_hash": bank_bindings[index]["sample_hash"],
        }
        if summary.get("identity") != expected_unit:
            raise ValueError("R3 bank summary is not bound to its 32 frozen rollout records")
        if (
            summary.get("passed") is not True
            or summary.get("status") != "PASS"
            or summary.get("origin_state_hash") != origin_hash
            or summary.get("origin_restored") is not True
            or summary.get("candidate_optimizer_updates") != 5
            or summary.get("replay_optimizer_updates") != 1
            or summary.get("lambda_zero_replay", {}).get("passed") is not True
            or summary.get("lambda_zero_replay", {}).get("bitwise_equal") is not True
            or (summary.get("reuse_adopted") is True and not allowed)
            or any(
                summary.get("answer_diagnostics", {}).get(a, {}).get("optimizer_updates") != 0
                for a in ("A_BASE", "A_VALID")
            )
            or (
                summary.get("all_valid_invariant", {}).get("applicable") is True
                and summary["all_valid_invariant"].get("passed") is not True
            )
            or sorted(c["lambda"] for c in summary["candidates"]) != list(LAMBDAS)
        ):
            raise ValueError("R3 actual optimizer/restoration/replay/gradient controls incomplete")
        for candidate in summary["candidates"]:
            cid = candidate["candidate_id"]
            cid_identity = candidate["checkpoint_identity"]
            if cid in ids or cid != f"bank_{int(index):02d}_lambda_{candidate['lambda']:g}":
                raise ValueError("R3 candidate identifiers are not distinct bank/lambda states")
            ids.add(cid)
            path = Path(candidate["checkpoint_path"]).resolve()
            if (
                not path.is_relative_to(root)
                or str(path.relative_to(root)) not in files
                or file_hash(path) != candidate["checkpoint_sha256"]
            ):
                raise ValueError("R3 candidate checkpoint file binding changed")
            if (
                cid_identity.get("candidate_id") != cid
                or cid_identity.get("lambda") != candidate["lambda"]
                or cid_identity.get("bank_index") != int(index)
                or cid_identity.get("origin_state_hash") != origin_hash
                or cid_identity.get("bank_hash") != bank_bindings[index]["bank_hash"]
                or any(cid_identity.get(k) != v for k, v in expected_unit.items())
            ):
                raise ValueError("R3 candidate checkpoint identity changed")
            state = load_checkpoint(path, cid_identity)
            _check_optimizer_state(state, step=1)
            if (
                state_hash(state) != candidate["state_hash"]
                or state_hash({n: state_hash(t) for n, t in state["parameters"].items()})
                != candidate["parameter_hash"]
                or state_hash(state["optimizer"]) != candidate["optimizer_state_hash"]
            ):
                raise ValueError("R3 candidate checkpoint state hashes changed")
            if int(index) in (0, 6):
                score_count += _check_scores(root, identity, candidate, proposal)
    details = status["details"]
    required_counts = {
        "raw_sample_count": 1152,
        "train_rollouts": 384,
        "control_proposal_rollouts": 768,
        "distinct_candidate_optimizer_updates": 60,
        "distinct_lambda_zero_replays": 12,
        "response_candidates": 10,
        "control_candidate_sequences_scored": 7680,
    }
    counts = _read(root / "runtime_profile.json")
    if (
        len(ids) != 60
        or score_count != 7680
        or any(details.get(k) != v for k, v in required_counts.items())
        or details.get("candidate_commit") is not False
        or details.get("direct_resample_cold") is not False
        or any(
            details.get("scratch_restoration", {}).get(k) is not True
            for k in ("full_origin_state_and_rng", "initial_adapter", "empty_adam", "frozen_base")
        )
        or counts.get("optimizer_steps_observed_at_least", 0) < 80
        or counts.get("backward_calls_observed_at_least", 0) <= 0
        or any(
            details.get("runtime_counts", {}).get(k) != v
            for k, v in counts.items()
            if k != "invocations"
        )
    ):
        raise ValueError("R3 final real execution counts or restoration audit incomplete")
    from .r3_report_gate import validate_response_artifacts

    response_audit = validate_response_artifacts(
        root, proposal_records=list(proposal.values()), bank_summaries=banks
    )
    return {
        "status": "PASS",
        "r3_cold_dir": str(root),
        "r3_files": files,
        "runtime_lock_sha256": files["runtime_lock.json"],
        "environment": runtime["environment"],
        "origin_state_hash": origin_hash,
        "initial_adapter_hash": runtime["initial_adapter_hash"],
        "plan_hash": plan["plan_hash"],
        "bank_manifest_sha256": files["bank_manifest.json"],
        "source_commit": runtime["source"].get("source_commit"),
        "response_artifact_audit": response_audit,
    }
