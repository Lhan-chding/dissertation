"""S1 measured candidate forks and durable direct control observations.

This module never locates or loads a model. The server preflight supplies a bound
plan and an already verified adapter; CPU fixtures use the same execution path.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .core import RunStore, canonical_hash, file_hash, frozen_writer, write_json
from .optimizer_fork import state_hash
from .r2_runtime import _json_safe, annotate_diagnostic, generation_checks
from .r3_runtime import (
    _manifest,
    _optimizer_step,
    _verify_manifest,
    run_atomic_unit,
    validate_sample_ledger,
)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _component(value, label="path"):
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in {".", ".."}
        or "\\" in value
    ):
        raise ValueError(f"{label} must be a single safe path component")
    return value


def _safe_root(path):
    path = Path(path).absolute()
    if ".." in path.parts:
        raise ValueError("output path contains parent traversal")
    for parent in [path, *path.parents]:
        if parent.is_symlink():
            raise ValueError("output path traverses a symbolic link")
    if path.exists():
        for item in path.rglob("*"):
            if item.is_symlink():
                raise ValueError("output path contains a symbolic link")
    return path


def _immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if _json(path) != value:
            raise ValueError(f"immutable completed evidence differs: {path.name}")
        return
    write_json(path, value)


def _publish(source, destination):
    """Publish immutable measured bytes without keeping duplicate tensor storage."""
    destination = Path(destination)
    if destination.exists():
        if file_hash(destination) != file_hash(source):
            raise ValueError("candidate published checkpoint hash mismatch")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def _restore(adapter, optimizer, state, *, sampler=None):
    from .followup_updates import capture_state, restore_state

    try:
        restore_state(adapter.model, optimizer, state, adapter=adapter, sampler=sampler)
        actual = capture_state(
            adapter.model, optimizer, state["metadata"], adapter=adapter, sampler=sampler
        )
        if state_hash(actual) != state_hash(state):
            raise RuntimeError("complete forward/Adam/RNG/sampler restoration failed")
    except Exception:
        adapter._followup_unusable = True
        adapter._followup_state_unusable = True
        raise


def _fault(out, identity, exc, *, kind="failure", details=None):
    root = Path(out) / "faults"
    root.mkdir(parents=True, exist_ok=True)
    attempt = root / f"attempt_{len(list(root.glob('attempt_*'))):04d}"
    attempt.mkdir(exist_ok=False)
    write_json(
        attempt / f"{kind}.json",
        {
            "identity": identity,
            "status": "MEASUREMENT_FAULT",
            "type": type(exc).__name__,
            "message": str(exc),
            "details": details or {},
        },
    )


def _update_budget(root, bank_identity, maximum):
    reservations = sorted((Path(root) / "update_reservations").glob("reservation_*.json"))
    for reference in Path(root).glob("candidates/*/updates/attempt_*/reservation_reference.json"):
        binding = _json(reference)
        name = _component(binding["reservation_file"], "budget reservation")
        reserved = Path(root) / "update_reservations" / name
        if not reserved.is_file() or file_hash(reserved) != binding["reservation_sha256"]:
            raise ValueError("update budget reservation missing or changed")
    confirmed = 0
    for index, path in enumerate(reservations):
        record = _json(path)
        if (
            record.get("bank_identity_hash") != canonical_hash(bank_identity)
            or record.get("maximum_updates") != maximum
            or record.get("index") != index
        ):
            raise ValueError("update budget reservation identity changed")
        attempt = Path(root) / record["attempt"]
        if not attempt.resolve().is_relative_to(Path(root).resolve()):
            raise ValueError("update budget attempt escapes bank root")
        marker = attempt / "adam_observed.json"
        if marker.exists():
            observed = _json(marker)
            if (
                observed.get("reservation_sha256") != file_hash(path)
                or observed.get("optimizer_updates") != 1
            ):
                raise ValueError("observed Adam call differs from budget reservation")
            confirmed += 1
    if len(reservations) > maximum:
        raise RuntimeError("physical update budget exceeded")
    return {
        "maximum_physical_updates": maximum,
        "reserved_possible_optimizer_calls": len(reservations),
        "confirmed_optimizer_calls": confirmed,
        "uncertain_optimizer_calls": len(reservations) - confirmed,
        "remaining_update_reservations": maximum - len(reservations),
        "maximum_possible_backward_sequences": 32 * len(reservations),
        "failed_attempts_release_quota": False,
    }


def _reserve_update(root, bank_identity, maximum, attempt, candidate_id):
    budget = _update_budget(root, bank_identity, maximum)
    if budget["remaining_update_reservations"] == 0:
        raise RuntimeError(
            "physical update budget exhausted; failed or interrupted attempts "
            "retain their reserved quota"
        )
    index = budget["reserved_possible_optimizer_calls"]
    path = Path(root) / "update_reservations" / f"reservation_{index:04d}.json"
    _immutable_json(
        path,
        {
            "index": index,
            "bank_identity_hash": canonical_hash(bank_identity),
            "maximum_updates": maximum,
            "candidate_id": candidate_id,
            "attempt": str(Path(attempt).relative_to(root)),
            "possible_optimizer_calls": 1,
        },
    )
    _immutable_json(
        Path(attempt) / "reservation_reference.json",
        {"reservation_file": path.name, "reservation_sha256": file_hash(path)},
    )
    return path


def _prepared(adapter, prompt, data_root):
    prepared = adapter.prepare(prompt["prompt"], prompt.get("data_root") or data_root)
    audit = prepared["audit"]
    expected_image = bool(prompt["prompt"].get("image_path"))
    if bool(audit.get("image_token_count")) != expected_image or (
        expected_image and not audit.get("pixel_values_hash")
    ):
        raise ValueError("prepared image binding mismatch")
    if audit.get("enable_thinking") is not False:
        raise ValueError("followup requires the locked non-thinking prompt")
    for key in ("final_prompt_hash", "tokenized_prompt_hash", "input_tensor_hash"):
        if not audit.get(key):
            raise ValueError(f"prepared input missing {key}")
    binding = prompt.get("prepared_binding", {})
    for key, expected in binding.items():
        actual = state_hash(prepared) if key == "prepared_hash" else audit.get(key)
        if actual != expected:
            raise ValueError(f"prepared input binding changed: {key}")
    return prepared


def build_direct_requests(
    prompts, identity, bank_id, candidate, *, samples_per_prompt=16, seed_root=20260914
):
    """Share RNG within each bank; retain candidate identity in sample keys."""
    _component(bank_id, "bank")
    if type(samples_per_prompt) is not int or samples_per_prompt < 1:
        raise ValueError("positive samples_per_prompt required")
    requests = []
    for prompt in prompts:
        for index in range(samples_per_prompt):
            rng = {
                "protocol": identity["protocol_version"],
                "origin": identity["origin_state_hash"],
                "bank": bank_id,
                "prompt": prompt["prompt_id"],
                "prompt_hash": prompt["prompt_hash"],
                "sample_index": index,
                "seed_root": seed_root,
            }
            key = canonical_hash(rng)
            request = {
                "protocol_version": identity["protocol_version"],
                "execution_kind": identity["execution_kind"],
                "origin_state_hash": identity["origin_state_hash"],
                "origin_checkpoint_step": 64,
                "bank_id": bank_id,
                **candidate,
                "seed_root": seed_root,
                "sample_rng_key": key,
                "sample_seed": int(key[:8], 16) % (2**31),
                "sample_index": index,
                "rollout_index": index,
                "decode_mode": "sample",
                "do_sample": True,
                "max_new_tokens": 64,
                "train_seed": 17,
                "split": "control",
                **{
                    k: prompt[k]
                    for k in (
                        "prompt_id",
                        "base_scene_id",
                        "family",
                        "interface",
                        "prompt_hash",
                        "scene_hash",
                    )
                },
            }
            request["sample_key"] = canonical_hash({"identity": identity, "request": request})
            requests.append(request)
    if len({row["sample_key"] for row in requests}) != len(requests):
        raise ValueError("duplicate direct prompt/sample identity")
    return requests


def collect_samples(
    adapter, optimizer, state, prompts, requests, out, identity, data_root, *, sampler=None
):
    """Run actual generation and parsing, preserve faults, restore complete state.

    Valid semantic I outputs count normally. Returned malformed generations are
    saved with failed execution checks and cannot be silently skipped on resume.
    """
    import torch

    from .followup_updates import capture_state

    if getattr(adapter, "_followup_unusable", False) or getattr(
        adapter, "_followup_state_unusable", False
    ):
        raise RuntimeError("adapter is unusable after an earlier restoration failure")
    out = _safe_root(out)
    prompt_map = {p["prompt_id"]: p for p in prompts}
    if len(prompt_map) != len(prompts) or len({r["sample_key"] for r in requests}) != len(requests):
        raise ValueError("duplicate prompt/sample requests")
    ledger_identity = {
        **identity,
        "measurement_state_hash": state_hash(state),
        "requests_hash": canonical_hash(requests),
        "prompt_panel_hash": state_hash(prompts),
    }
    store = RunStore(out, ledger_identity, resume=(out / "identity.json").exists())
    validate_sample_ledger(store.records, requests)
    parameter_before = state_hash(state["parameters"])
    frozen_before = {
        n: p._version for n, p in adapter.model.named_parameters() if not p.requires_grad
    }
    failure = None
    _restore(adapter, optimizer, state, sampler=sampler)
    try:
        prepared_cache = {}
        for request in requests:
            if request["sample_key"] in store.keys:
                continue
            prompt = prompt_map[request["prompt_id"]]
            if prompt["prompt_id"] not in prepared_cache:
                prepared_cache[prompt["prompt_id"]] = _prepared(adapter, prompt, data_root)
            prepared = prepared_cache[prompt["prompt_id"]]
            generation = None
            started, before = time.perf_counter(), adapter.forward_calls
            try:
                with torch.no_grad():
                    generation = adapter.generate(
                        prepared,
                        seed=request["sample_seed"],
                        max_new_tokens=request.get("max_new_tokens", 64),
                        do_sample=request.get(
                            "do_sample", request.get("decode_mode", "sample") == "sample"
                        ),
                    )
                faults = generation_checks(
                    generation,
                    adapter.processor.tokenizer,
                    adapter.eos_ids,
                    request.get("max_new_tokens", 64),
                )
                if (
                    prepared["audit"]["image_token_count"]
                    and generation.get("vision_forward_calls", 0) < 1
                ):
                    faults.append("image did not enter measured visual forward")
                if prompt.get("track") == "L":
                    from .r4_runtime import _annotate_rollout

                    annotation = _annotate_rollout(generation["raw_completion"], prompt)
                else:
                    annotation = annotate_diagnostic(generation["raw_completion"], prompt["scene"])
                scores = generation["behavior_token_logprobs"]
                row = _json_safe(
                    {
                        **identity,
                        **request,
                        **prepared["audit"],
                        **generation,
                        **annotation,
                        "operation": prompt["scene"]["operation"],
                        "image_hash_or_null": prompt["scene"].get("image_hash")
                        if prepared["audit"]["image_token_count"]
                        else None,
                        "prepared_hash": state_hash(prepared),
                        "old_logprobs": scores,
                        "parsed_world_or_null": annotation["parsed_world"],
                        "runtime_forward_calls": adapter.forward_calls - before,
                        "elapsed_seconds": time.perf_counter() - started,
                        "execution_checks": {"passed": not faults, "faults": faults},
                    }
                )
                row["record_hash"] = canonical_hash(row)
                store.append(row)
                generation = None
                if faults:
                    raise RuntimeError("generation execution fault; raw output preserved")
            except Exception as exc:
                if generation is not None:
                    row = _json_safe(
                        {
                            **identity,
                            **request,
                            "generation_return": generation,
                            "parse_result": "execution_error",
                            "execution_checks": {"passed": False, "faults": [str(exc)]},
                        }
                    )
                    row["record_hash"] = canonical_hash(row)
                    store.append(row)
                raise
        return [store.records[r["sample_key"]] for r in requests]
    except BaseException as exc:
        failure = exc
        _fault(out, ledger_identity, exc)
        raise
    finally:
        try:
            actual = capture_state(
                adapter.model, optimizer, state["metadata"], adapter=adapter, sampler=sampler
            )
            if state_hash(actual["parameters"]) != parameter_before or state_hash(
                actual["optimizer"]
            ) != state_hash(state["optimizer"]):
                raise RuntimeError("read-only sampling changed parameter/Adam state")
            if frozen_before != {
                n: p._version for n, p in adapter.model.named_parameters() if not p.requires_grad
            }:
                raise RuntimeError("read-only sampling changed frozen parameters")
        except Exception as exc:
            _fault(out, ledger_identity, exc, kind="contamination")
            if failure is None:
                failure = exc
        try:
            _restore(adapter, optimizer, state, sampler=sampler)
        except Exception as exc:
            adapter._followup_unusable = True
            adapter._followup_state_unusable = True
            _fault(out, ledger_identity, exc, kind="restoration_failure")
            if failure is None:
                raise
        # Preserve the original execution exception when both execution and
        # restoration fail. A contamination discovered after success must fail.
        if failure is not None and __import__("sys").exc_info()[0] is None:
            raise failure


def _counts(rows):
    grouped = {}
    for row in rows:
        group = grouped.setdefault(
            row["prompt_id"],
            {
                **{k: row[k] for k in ("prompt_id", "base_scene_id", "family", "interface")},
                "counts": dict.fromkeys(("X", "S", "W", "I"), 0),
                "n": 0,
            },
        )
        if row["category"] not in group["counts"] or row["execution_checks"]["passed"] is not True:
            raise ValueError("only complete valid execution rows enter counts")
        group["counts"][row["category"]] += 1
        group["n"] += 1
    return list(grouped.values())


def _checkpoint(path, identity, record):
    from .followup_updates import load_checkpoint

    if file_hash(path) != record["checkpoint_sha256"]:
        raise ValueError("candidate checkpoint file hash mismatch")
    state = load_checkpoint(path, identity)
    if state_hash(state) != record["candidate_state_hash"]:
        raise ValueError("candidate checkpoint state hash mismatch")
    return state


def _fingerprint(state, identity, adapter, plan):
    from .followup_updates import forward_state_hash

    return canonical_hash(
        {
            "forward_state_hash": forward_state_hash(state),
            "model_hash": identity["model_hash"],
            "model_revision": adapter.revision,
            "processor_and_architecture": {
                key: adapter.audit.get(key)
                for key in (
                    "model_class",
                    "base_dtype",
                    "probability_execution",
                    "eos_token_ids",
                    "lora_rank",
                    "lora_alpha",
                    "lora_dropout",
                    "lora_modules",
                    "trainable_parameters",
                    "trainable_dtypes",
                    "requested_image_token_limit",
                    "processor_max_pixels",
                    "processor_patch_size",
                    "processor_merge_size",
                    "processor_hash",
                    "tokenizer_hash",
                    "chat_template_hash",
                )
            },
            "parser": identity["parser_version_hash"],
            "generation": plan["design"]["generation_N"],
            "control_prompts": state_hash(plan["control_prompts"]),
        }
    )


def _run_bank(plan, unit, spec, root, adapter, optimizer, origin, identity, data_root, fixture):
    from .followup_statistics import summarize_candidate_panel, summarize_direct_response
    from .followup_updates import compare_update_vectors, fork_one_candidate, save_checkpoint

    bank_identity = {**identity, "bank_id": unit["bank_id"], "bank_hash": unit["bank_hash"]}
    _immutable_json(root / "identity.json", bank_identity)
    _immutable_json(root / "source_binding.json", plan.get("bindings", {}))
    _immutable_json(root / "bank_manifest.json", {k: v for k, v in unit.items() if k != "groups"})
    _immutable_json(
        root / "origin_binding.json",
        {
            "origin_state_hash": state_hash(origin),
            "origin_checkpoint_step": 64,
            "optimizer_step": _optimizer_step(origin),
            "permanent_training_commit": False,
        },
    )
    group_stats = [
        {
            "prompt_id": group[0]["prompt_id"],
            "counts": {c: sum(r["category"] == c for r in group) for c in "XSWI"},
            "K": len(group),
        }
        for group in unit["groups"]
    ]
    _immutable_json(root / "group_statistics.json", group_stats)
    records, checkpoint_paths = {}, {}
    execution_specs = [
        *spec["candidates"],
        {"id": "joint_0_replay", "policy": "joint", "auxiliary_weight": 0.0},
    ]
    for candidate in execution_specs:
        cid = _component(candidate["id"], "candidate")
        candidate_identity = {**bank_identity, "candidate_spec": candidate}
        dest = root / "candidates" / cid

        def execute(attempt, candidate=candidate, candidate_identity=candidate_identity, cid=cid):
            reservation = _reserve_update(root, bank_identity, len(execution_specs), attempt, cid)
            started = time.perf_counter()
            measured = fork_one_candidate(
                adapter,
                optimizer,
                origin,
                unit["groups"],
                candidate,
                failure_path=attempt / "update_failure.json",
            )
            write_json(
                attempt / "adam_observed.json",
                {
                    "reservation_sha256": file_hash(reservation),
                    "optimizer_updates": measured["audit"].get("optimizer_updates"),
                },
            )
            state = measured["state"]
            path = attempt / "checkpoint.pt"
            save_checkpoint(path, state, candidate_identity)
            record = {
                "candidate_id": cid,
                "candidate_spec": candidate,
                "candidate_policy_fingerprint": _fingerprint(state, identity, adapter, plan),
                "candidate_state_hash": state_hash(state),
                "candidate_optimizer_state_hash": state_hash(state["optimizer"]),
                "candidate_optimizer_step": _optimizer_step(state),
                "candidate_parameter_hash": state_hash(state["parameters"]),
                "checkpoint_identity": candidate_identity,
                "checkpoint_sha256": file_hash(path),
                "origin_checkpoint_step": 64,
                "permanent_training_commit": False,
                "audit": measured["audit"],
                "elapsed_seconds": time.perf_counter() - started,
            }
            if record["candidate_optimizer_step"] != 65:
                raise ValueError("candidate must advance warm Adam exactly from 64 to 65")
            if (
                record["audit"].get("optimizer_updates") != 1
                or record["audit"].get("backward_calls") != 32
            ):
                raise RuntimeError(
                    "candidate update meter violates one Adam / B4K8 backward budget"
                )
            write_json(attempt / "candidate.json", record)
            return record

        attempt, record, _ = run_atomic_unit(dest / "updates", candidate_identity, execute)
        _publish(attempt / "checkpoint.pt", dest / "checkpoint.pt")
        _immutable_json(dest / "candidate.json", record)
        records[cid], checkpoint_paths[cid] = record, dest / "checkpoint.pt"

    def load(cid):
        return _checkpoint(checkpoint_paths[cid], records[cid]["checkpoint_identity"], records[cid])

    baseline = load("joint_0")
    replay = load("joint_0_replay")
    from .fork_gradients import compare_gradient_bundles
    from .r3_updates import STATE_ATOL, STATE_RTOL, _tensor_parts

    parameter_comparison = compare_gradient_bundles(
        {"gradients": baseline["parameters"]},
        {"gradients": replay["parameters"]},
        atol=STATE_ATOL,
        rtol=STATE_RTOL,
    )
    baseline_tensors, baseline_structure = _tensor_parts(baseline["optimizer"])
    replay_tensors, replay_structure = _tensor_parts(replay["optimizer"])
    optimizer_comparison = compare_gradient_bundles(
        {"gradients": baseline_tensors},
        {"gradients": replay_tensors},
        atol=STATE_ATOL,
        rtol=STATE_RTOL,
    )
    replay_comparison = {
        "state_bitwise_equal": state_hash(baseline) == state_hash(replay),
        "parameters_bitwise_equal": state_hash(baseline["parameters"])
        == state_hash(replay["parameters"]),
        "optimizer_bitwise_equal": state_hash(baseline["optimizer"])
        == state_hash(replay["optimizer"]),
        "parameter_comparison": parameter_comparison,
        "optimizer_comparison": optimizer_comparison,
        "optimizer_structure_equal": baseline_structure == replay_structure,
        "rng_sampler_equal": all(
            state_hash(baseline[k]) == state_hash(replay[k])
            for k in ("rng", "sampler", "scheduler")
        ),
        "tolerance_source": (
            "inherited r3_updates.STATE_ATOL=1e-7,STATE_RTOL=1e-5; "
            "exact hashes reported independently"
        ),
        "comparison": compare_update_vectors(origin, baseline, replay),
    }
    if not (
        parameter_comparison["allclose"]
        and optimizer_comparison["allclose"]
        and replay_comparison["optimizer_structure_equal"]
        and replay_comparison["rng_sampler_equal"]
    ):
        raise RuntimeError(
            "joint0 replay violates inherited frozen numerical tolerance or exact control structure"
        )
    _immutable_json(root / "baseline_replay.json", replay_comparison)
    geometry = []
    ids = [c["id"] for c in spec["candidates"]]
    for left_index, left in enumerate(ids):
        left_state = load(left)
        for right in ids[left_index + 1 :]:
            geometry.append(
                {
                    "left": left,
                    "right": right,
                    **compare_update_vectors(baseline, left_state, load(right)),
                }
            )
    _immutable_json(root / "parameter_geometry.json", geometry)
    limits = plan.get("fixture_limits", {}) if fixture else {}
    sample_count = limits.get("samples_per_prompt", 16)
    aliases, by_fingerprint, panels = {}, {}, {}
    for cid in spec["direct_candidate_ids"]:
        candidate = records[cid]
        fingerprint = candidate["candidate_policy_fingerprint"]
        if fingerprint in by_fingerprint:
            source = by_fingerprint[fingerprint]
            alias = {
                "candidate_id": cid,
                "alias_of": source,
                "candidate_policy_fingerprint": fingerprint,
                "source_ledger": f"candidates/{source}/direct/samples.jsonl",
                "additional_samples": 0,
                "independent_training_state": True,
                "matching_rule": "EXACT_COMPLETE_FORWARD_FINGERPRINT",
            }
            aliases[cid] = alias
            panels[cid] = panels[source]
            _immutable_json(root / "candidates" / cid / "sample_alias.json", alias)
            _immutable_json(
                root / "candidates" / cid / "counts_by_prompt.json",
                {
                    "alias_of": source,
                    "additional_samples": 0,
                    "source_counts": f"candidates/{source}/counts_by_prompt.json",
                },
            )
            continue
        by_fingerprint[fingerprint] = cid
        fields = {
            k: candidate[k]
            for k in (
                "candidate_id",
                "candidate_policy_fingerprint",
                "candidate_optimizer_state_hash",
                "candidate_optimizer_step",
            )
        }
        fields["source_bank_ids"] = sorted(
            {g["bank"] for g in unit.get("source_groups", []) if "bank" in g}
        )
        requests = build_direct_requests(
            plan["control_prompts"],
            bank_identity,
            unit["bank_id"],
            fields,
            samples_per_prompt=sample_count,
            seed_root=plan["design"]["S1"]["direct_evaluation"]["seed_root"],
        )
        panels[cid] = collect_samples(
            adapter,
            optimizer,
            load(cid),
            plan["control_prompts"],
            requests,
            root / "candidates" / cid / "direct",
            bank_identity,
            data_root,
        )
        _restore(adapter, optimizer, origin)
        _immutable_json(root / "candidates" / cid / "counts_by_prompt.json", _counts(panels[cid]))
        _publish(
            root / "candidates" / cid / "direct" / "samples.jsonl",
            root / "candidates" / cid / "samples.jsonl",
        )
    _immutable_json(root / "policy_aliases.json", aliases)
    weights = {p["prompt_id"]: 1 / len(plan["control_prompts"]) for p in plan["control_prompts"]}
    panel_response = summarize_candidate_panel(
        {cid: panels[cid] for cid in by_fingerprint.values()},
        fixed_weights=weights,
        baseline_key="joint_0",
        aliases={cid: alias["alias_of"] for cid, alias in aliases.items()},
        bootstrap_replicates=limits.get("bootstrap_replicates", 5000),
        bootstrap_seed=20260914,
    )
    _immutable_json(root / "all_candidate_response.json", panel_response)
    responses = []
    contrasts = [("joint_0", cid) for cid in spec["direct_candidate_ids"] if cid != "joint_0"]
    if "no_x_off_1" in panels:
        contrasts.append(("no_x_off_1", "joint_1"))
    for left, right in contrasts:
        response = summarize_direct_response(
            panels[left],
            panels[right],
            fixed_weights=weights,
            bootstrap_replicates=limits.get("bootstrap_replicates", 5000),
            bootstrap_seed=20260914,
        )
        responses.append(
            {
                "left_candidate": left,
                "right_candidate": right,
                "same_policy_alias": records[left]["candidate_policy_fingerprint"]
                == records[right]["candidate_policy_fingerprint"],
                "aliases_add_independent_samples": False,
                "uncertainty_scope": (
                    "this pair only; all_candidate_response.json contains "
                    "the joint bank-wide union bounds"
                ),
                **response,
            }
        )
    _immutable_json(root / "paired_response.json", responses)
    budget = _update_budget(root, bank_identity, len(execution_specs))
    _immutable_json(root / "update_budget.json", budget)
    return {
        "bank_id": unit["bank_id"],
        "candidate_updates": len(spec["candidates"]),
        "replay_updates": 1,
        "logical_direct_candidates": len(spec["direct_candidate_ids"]),
        "unique_direct_candidates": len(by_fingerprint),
        "raw_sample_count": sum(len(panels[cid]) for cid in by_fingerprint.values()),
        "aliases": aliases,
        "update_budget": budget,
    }


def run_followup_bank(
    plan, *, output_root, adapter, optimizer, origin, data_root, fixture=False, resume=False
):
    """Execute one selected bank or all six validated S1 units, without training commits."""
    import torch

    from .followup_updates import capture_state

    if getattr(adapter, "_followup_unusable", False) or getattr(
        adapter, "_followup_state_unusable", False
    ):
        raise RuntimeError("adapter is unusable after restoration failure")
    kind = "CPU_FAKE_TORCH" if fixture else "REAL_CUDA_FOLLOWUP"
    if fixture and (
        getattr(adapter, "execution_kind", None) != kind
        or any(p.device.type != "cpu" for p in adapter.model.parameters())
    ):
        raise ValueError("fixture must use an explicitly FAKE CPU adapter")
    if not fixture and (
        plan.get("execution_kind") != kind
        or not torch.cuda.is_available()
        or not plan.get("gates", {}).get("server_preflight_passed")
    ):
        raise PermissionError("actual CUDA execution requires bound server preflight gates")
    if (
        _optimizer_step(origin) != 64
        or origin["metadata"].get("checkpoint_step") != 64
        or origin["metadata"].get("arm") != "X_BASE"
    ):
        raise ValueError("S1 origin must be X_BASE step64 full Adam state")
    expected = {spec["id"]: spec for spec in plan["design"]["S1"]["units"]}
    units = plan["units"]
    if not units or len({u["bank_id"] for u in units}) != len(units):
        raise ValueError("distinct nonempty bank units required")
    for unit in units:
        _component(unit["bank_id"], "bank")
        if (
            unit["bank_id"] not in expected
            or len(unit["groups"]) != 4
            or any(len(g) != 8 for g in unit["groups"])
        ):
            raise ValueError("unknown bank or B4/K8 bank mismatch")
    prompts = plan["control_prompts"]
    if not fixture and (len(prompts) != 48 or len({p["base_scene_id"] for p in prompts}) != 24):
        raise ValueError("real S1 requires the exact 24 scene / 48 prompt control panel")
    controls = {p["base_scene_id"] for p in prompts}
    if any(
        row["base_scene_id"] in controls
        for unit in units
        for group in unit["groups"]
        for row in group
    ):
        raise ValueError("training/control scene leakage")
    identity = {
        **plan["identity"],
        "execution_kind": kind,
        "origin_state_hash": state_hash(origin),
        "design_hash": canonical_hash(plan["design"]),
        "units": [(u["bank_id"], u["bank_hash"]) for u in units],
        "execution_group_bindings": {u["bank_id"]: state_hash(u["groups"]) for u in units},
        "control_panel_hash": state_hash(prompts),
        "fixture_limits": plan.get("fixture_limits", {}) if fixture else {},
    }
    execution_identity = plan.get("execution_identity", {})
    identity["validated_plan_hash"] = execution_identity.get("validated_plan_hash")
    identity["source_hash"] = execution_identity.get("source_hash")
    if fixture:
        identity["validated_plan_hash"] = identity["validated_plan_hash"] or canonical_hash(
            {"execution_kind": kind, "design_hash": identity["design_hash"]}
        )
        identity["source_hash"] = identity["source_hash"] or canonical_hash(
            {"fixture_source": "CPU_FAKE_TORCH_ONLY"}
        )
    elif not identity["validated_plan_hash"] or not identity["source_hash"]:
        raise ValueError("real runtime requires validated plan and source identity hashes")
    identity["units"] = [list(item) for item in identity["units"]]
    for key in (
        "model_hash",
        "data_hash",
        "config_hash",
        "parser_version_hash",
        "protocol_version",
    ):
        if not identity.get(key):
            raise ValueError(f"runtime identity missing {key}")
    out = _safe_root(output_root)
    direct_bank_root = len(units) == 1 and out.name == units[0]["bank_id"]
    if direct_bank_root:
        identity.update({"bank_id": units[0]["bank_id"], "bank_hash": units[0]["bank_hash"]})
    identity["run_id"] = canonical_hash(identity)
    for binding in plan.get("bindings", {}).values():
        if isinstance(binding, dict) and binding.get("path"):
            parent = Path(binding["path"]).resolve()
            if out.resolve() == parent or out.resolve().is_relative_to(parent):
                raise ValueError("output must not lie inside read-only parent evidence")
    out.mkdir(parents=True, exist_ok=True)
    with frozen_writer(out):
        if (out / "identity.json").exists():
            if not resume:
                raise FileExistsError("run exists; explicit identical resume required")
            if _json(out / "identity.json") != identity:
                raise ValueError("resume identity changed")
        elif resume:
            raise FileNotFoundError("cannot resume missing identity")
        elif any(path.name != ".writer.lock" for path in out.iterdir()):
            raise ValueError("orphan output files without runtime identity; use a fresh directory")
        _immutable_json(out / "identity.json", identity)
        if (out / "completed.json").exists():
            completed = _json(out / "completed.json")
            if completed["manifest_sha256"] != file_hash(out / "manifest.json"):
                raise ValueError("completed manifest hash mismatch")
            _verify_manifest(out, _json(out / "manifest.json"))
            return _json(out / "status.json")
        start = time.perf_counter()
        before_calls = adapter.forward_calls
        try:
            results = []
            for unit in units:
                _restore(adapter, optimizer, origin)
                bank_root = out if direct_bank_root else out / unit["bank_id"]
                bank_start, bank_forwards = time.perf_counter(), adapter.forward_calls
                bank_result = _run_bank(
                    plan,
                    unit,
                    expected[unit["bank_id"]],
                    bank_root,
                    adapter,
                    optimizer,
                    origin,
                    identity,
                    data_root,
                    fixture,
                )
                results.append(bank_result)
                if not direct_bank_root:
                    _immutable_json(
                        bank_root / "status.json",
                        {
                            "status": "CPU_TESTED" if fixture else "MEASURED",
                            "execution_kind": kind,
                            **bank_result,
                        },
                    )
                    profile_path = bank_root / "runtime_profile.json"
                    if not profile_path.exists():
                        _immutable_json(
                            profile_path,
                            {
                                "elapsed_seconds": time.perf_counter() - bank_start,
                                "forward_calls_this_completion_attempt": adapter.forward_calls
                                - bank_forwards,
                            },
                        )
                    manifest = _manifest(bank_root, exclude=("manifest.json",))
                    _immutable_json(bank_root / "manifest.json", manifest)
            _restore(adapter, optimizer, origin)
            if state_hash(
                capture_state(adapter.model, optimizer, origin["metadata"], adapter=adapter)
            ) != state_hash(origin):
                raise RuntimeError("final scratch state restoration mismatch")
            result = {
                "status": "CPU_TESTED" if fixture else "MEASURED",
                "execution_kind": kind,
                "gpu_started": not fixture,
                "training_started": False,
                "permanent_training_commit": False,
                "banks": results,
                **{
                    key: sum(r[key] for r in results)
                    for key in (
                        "candidate_updates",
                        "replay_updates",
                        "logical_direct_candidates",
                        "unique_direct_candidates",
                        "raw_sample_count",
                    )
                },
            }
            if not (out / "runtime_profile.json").exists():
                _immutable_json(
                    out / "runtime_profile.json",
                    {
                        "elapsed_seconds": time.perf_counter() - start,
                        "forward_calls_this_completion_attempt": adapter.forward_calls
                        - before_calls,
                        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
                        if not fixture
                        else None,
                        "measurement_scope": (
                            "completion invocation; preserved failed attempts are separate"
                        ),
                    },
                )
            _immutable_json(out / "status.json", result)
            if (out / "manifest.json").exists():
                _verify_manifest(out, _json(out / "manifest.json"))
            else:
                _immutable_json(
                    out / "manifest.json",
                    _manifest(out, exclude=("manifest.json", "completed.json")),
                )
            _immutable_json(
                out / "completed.json",
                {
                    "status": result["status"],
                    "identity_hash": canonical_hash(identity),
                    "manifest_sha256": file_hash(out / "manifest.json"),
                },
            )
            return result
        except BaseException as exc:
            budgets = {}
            for unit in units:
                bank_root = out if direct_bank_root else out / unit["bank_id"]
                bank_identity = {
                    **identity,
                    "bank_id": unit["bank_id"],
                    "bank_hash": unit["bank_hash"],
                }
                try:
                    budgets[unit["bank_id"]] = _update_budget(
                        bank_root, bank_identity, len(expected[unit["bank_id"]]["candidates"]) + 1
                    )
                except Exception as budget_exc:
                    budgets[unit["bank_id"]] = {"status": "UNVERIFIABLE", "error": str(budget_exc)}
            _fault(out, identity, exc, details={"update_budgets": budgets})
            try:
                _restore(adapter, optimizer, origin)
            except Exception as restore_exc:
                adapter._followup_unusable = True
                adapter._followup_state_unusable = True
                _fault(out, identity, restore_exc, kind="restoration_failure")
            raise


def run_s1(plan, adapter, optimizer, origin, out, *, data_root, fixture=False, resume=False):
    return run_followup_bank(
        plan,
        output_root=out,
        adapter=adapter,
        optimizer=optimizer,
        origin=origin,
        data_root=data_root,
        fixture=fixture,
        resume=resume,
    )
