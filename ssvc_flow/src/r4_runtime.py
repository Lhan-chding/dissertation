"""Two independent seed-17 on-policy arms with immutable measured step attempts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .core import RunStore, canonical_hash, file_hash, frozen_writer, phase_artifacts, write_json
from .optimizer_fork import (
    capture_state,
    load_checkpoint,
    parameter_hash,
    save_checkpoint,
    state_hash,
)
from .r1_reference_smoke import _certificate_check, _helper_config
from .r1_supplement import _ExecutionMeter
from .r2_runtime import _json_safe, _source, annotate_diagnostic, generation_checks
from .r3_runtime import (
    _json,
    _manifest,
    _restore,
    _verify_manifest,
    _write_csv,
    run_atomic_unit,
    validate_sample_ledger,
)
from .smoke_runtime import _seed_everything, _verify_scene_images

ARMS = ("X_BASE", "X_VALID")


class NoComparableInitial(ValueError):
    """Valid historical evidence has a different effective comparison protocol."""


class PilotBlocked(ValueError):
    """A prerequisite or a preserved diagnostic stop prohibits training."""


def build_requests(prompts, identity, arm, step, role, policy_state_hash):
    requests = []
    for prompt in prompts:
        for index in range(8):
            request = {
                "phase": "R4",
                "arm": arm,
                "checkpoint_step": step,
                "bank_role": role,
                "base_scene_id": prompt["base_scene_id"],
                "prompt_id": prompt["prompt_id"],
                "group_id": prompt["prompt_id"],
                "family": prompt["family"],
                "constraint_family": prompt["family"],
                "interface": prompt["interface"],
                "split": prompt["split"],
                "protocol_track": prompt["track"],
                "track": "OOD"
                if prompt.get("evaluation_domain") == "graph_OOD"
                else prompt["track"],
                "evaluation_domain": prompt.get("evaluation_domain", "ID"),
                "prompt_hash": prompt["prompt_hash"],
                "scene_hash": canonical_hash(prompt["scene"]),
                "sample_index": index,
                "rollout_index": index,
                "decode_mode": "sample",
                "max_new_tokens": prompt["max_new_tokens"],
                "enable_thinking": False,
                "policy_state_hash": policy_state_hash,
                "run_id": canonical_hash(identity),
            }
            # Arm-independent seed root/order is predeclared. Policy identity still binds
            # randomness; each arm calls its own current policy and owns distinct sample keys.
            rng = {
                "seed_root": 17,
                "phase": "R4",
                "model_hash": identity["model_hash"],
                "data_hash": identity["data_hash"],
                "config_hash": identity["config_hash"],
                "step": step,
                "role": role,
                "prompt_hash": prompt["prompt_hash"],
                "prompt_id": prompt["prompt_id"],
                "index": index,
                "policy": policy_state_hash,
            }
            key = canonical_hash(rng)
            requests.append(
                {
                    **request,
                    "sample_rng_key": key,
                    "sample_seed": int(key[:8], 16) % (2**31),
                    "sample_key": canonical_hash(
                        {"identity": identity, "request": request, "rng": rng}
                    ),
                }
            )
    if len({r["sample_key"] for r in requests}) != len(requests):
        raise ValueError("R4 duplicate sample identity")
    return requests


def _load_plan(data_root, gate, r0_dir):
    from .constraint_solver import solve
    from .legacy_frozen import REPOSITORY, SOURCES, load_legacy_scenes
    from .r4_inputs import build_r4_plan

    root = Path(data_root)
    manifest = _json(root / "manifest.json")
    if file_hash(root / "manifest.json") != gate["binding"]["data_manifest_sha256"]:
        raise ValueError("R4 dataset manifest differs from passed R0")
    separation = _json(Path(r0_dir) / "closure_checks.json")["cross_split"]
    if (
        separation["status"] != "PASS"
        or separation["manifest_sha256"] != file_hash(root / "manifest.json")
        or separation["manifest_global_truth_uniqueness"] is not True
        or manifest.get("global_truth_uniqueness") is not True
        or separation["base_scene_collisions"]
        or separation["truth_structure_collisions"]
    ):
        raise ValueError("R4 global split separation proof is missing or changed")
    splits, hashes = {}, {}
    # confirm remains sealed. The original R0 manifest supplies its separation evidence.
    for split in ("train", "dev", "ood", "calibration", "control"):
        path = root / f"{split}.jsonl"
        digest = file_hash(path)
        if any(
            record["sha256"] != digest
            for record in (
                manifest["files"][path.name],
                gate["binding"]["dataset_files"][path.name],
            )
        ):
            raise ValueError(f"R4 source differs from audited split: {split}")
        if separation["splits"][split]["sha256"] != digest:
            raise ValueError("R4 split differs from global separation proof")
        hashes[path.name] = digest
        splits[split] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    legacy_lock = _json(Path(r0_dir) / "baseline_lock_L.json")
    legacy_root = REPOSITORY / "artifacts/v5/study_c2/data"
    legacy = load_legacy_scenes(
        legacy_root / "reward_fibers.jsonl", legacy_root / "reward_fibers_manifest.json"
    )
    if canonical_hash(legacy) != legacy_lock["data_hash"]:
        raise ValueError("R4 original L data differs from R0 baseline")
    for relative in SOURCES.values():
        if file_hash(REPOSITORY / relative) != legacy_lock["source_files"]["legacy/" + relative]:
            raise ValueError("R4 original L parser/executor/prompt source changed")
    plan = build_r4_plan(
        splits["train"],
        splits["dev"],
        splits["ood"],
        legacy,
        id_scenes=[
            scene for split in ("train", "dev", "calibration", "control") for scene in splits[split]
        ],
        legacy_lock=legacy_lock,
        data_root=str(root.resolve()),
    )
    selected = {
        r["base_scene_id"]: r["scene"]
        for key in ("train_prompts", "dev_prompts", "ood_prompts")
        for r in plan[key]
    }
    _verify_scene_images(list(selected.values()), root)
    for scene in selected.values():
        changed = [i for i in range(4) if scene["truth_world"][i] != scene["observed_world"][i]]
        if solve(scene["observed_world"], scene["cue"]) != [scene["truth_world"]] or changed != [
            scene["changed_index"]
        ]:
            raise ValueError("R4 unique independent repair solver failed")
    return (
        plan,
        {
            "files": hashes,
            "manifest_sha256": file_hash(root / "manifest.json"),
            "legacy_lock_sha256": file_hash(Path(r0_dir) / "baseline_lock_L.json"),
            "legacy_data_hash": canonical_hash(legacy),
            "cross_split_proof": separation,
            "confirm_access": "NOT_OPENED; separation inherited from passed R0",
        },
        legacy_lock,
    )


def _prepared(adapter, prompt, data_root):
    value = adapter.prepare(prompt["prompt"], prompt.get("data_root") or data_root)
    audit = value["audit"]
    image = prompt["interface"] == "IMAGE_CUE_FRESH"
    if audit.get("enable_thinking") is not False:
        raise ValueError("R4 thinking template changed")
    if bool(audit.get("image_token_count")) != image or (
        image and not audit.get("pixel_values_hash")
    ):
        raise ValueError("R4 image intervention binding mismatch")
    return value


def _frozen_versions(adapter):
    return {
        name: parameter._version
        for name, parameter in adapter.model.named_parameters()
        if not parameter.requires_grad
    }


def _check_frozen(adapter, expected, out=None):
    if _frozen_versions(adapter) != expected or any(
        p.grad is not None for p in adapter.model.parameters() if not p.requires_grad
    ):
        if out is not None:
            write_json(
                out / "policy_contamination.json",
                {"reason": "Frozen parameter version/gradient changed"},
            )
        raise RuntimeError("R4 frozen parameter version/gradient changed")


class _Progress:
    def __init__(self, out, invocation, meter):
        self.out, self.invocation, self.meter = out, invocation, meter

    def write(self, state, **details):
        write_json(self.invocation / "runtime_profile.json", self.meter.report())
        write_json(
            self.out / "progress.json",
            {"state": state, "invocation": self.invocation.name, **details},
        )


def _scene_fields(prompt, audit):
    scene = prompt["scene"]
    if prompt["track"] == "L":
        return {
            "truth_world": scene["truth"],
            "observed_world": scene["observation"],
            "changed_index": scene["error_index"],
            "operation": scene["operation"],
            "legacy_scene_id": scene["scene_id"],
            "legacy_facts": scene["facts"],
            "cue": None,
            "solution_count": None,
            "chart_type": None,
            "image_hash": None,
            "not_applicable_fields": {
                "cue": "NOT_APPLICABLE: original L facts are retained in legacy_facts",
                "solution_count": "NOT_APPLICABLE: N unique-repair solver is not applied to L",
                "chart_type": "NOT_APPLICABLE: original L text-only prompt",
                "image_hash": "NOT_APPLICABLE: original L has no image input",
                "constraint_results": "NOT_APPLICABLE: original L semantic parser protocol",
            },
        }
    return {
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
        "solution_count": 1,
        "image_hash": scene["image_hash"] if audit["image_token_count"] else None,
        "not_applicable_fields": {},
    }


def _annotate_rollout(raw, prompt):
    if prompt["track"] != "L":
        return annotate_diagnostic(raw, prompt["scene"])
    from .legacy_frozen import annotate_legacy

    scene = prompt["scene"]
    annotation = annotate_legacy(raw, scene)
    parsed = annotation["parsed_world"]
    return {
        **annotation,
        "extracted_action": parsed,
        "parse_result": "valid" if parsed is not None else "invalid",
        "hamming_to_observed": sum(
            a != b for a, b in zip(parsed, scene["observation"], strict=True)
        )
        if parsed is not None
        else None,
        "hamming_to_truth": sum(a != b for a, b in zip(parsed, scene["truth"], strict=True))
        if parsed is not None
        else None,
        "constraint_results": None,
    }


def _samples(
    adapter,
    optimizer,
    policy,
    prompts,
    store_root,
    identity,
    arm,
    step,
    role,
    data_root,
    config,
    legacy_lock,
    meter,
    progress,
):
    import torch

    policy_hash = state_hash(policy)
    policy_optimizer_hash = state_hash(policy["optimizer"])
    requests = build_requests(prompts, identity, arm, step, role, policy_hash)
    store_identity = {
        **identity,
        "arm": arm,
        "step": step,
        "role": role,
        "policy_state_hash": policy_hash,
        "request_hash": canonical_hash(requests),
    }
    store = RunStore(store_root, store_identity, resume=(store_root / "identity.json").exists())
    validate_sample_ledger(store.records, requests)
    prompt_map = {p["prompt_id"]: p for p in prompts}
    frozen = _frozen_versions(adapter)
    initial_parameters = parameter_hash(adapter.model, trainable=True)
    current, prepared, prepared_hash, scene_fields = None, None, None, None
    sampling_label = f"{arm}/step_{step:02d}/{role}/sampling"
    try:
        for request in requests:
            if request["sample_key"] in store.keys:
                continue
            prompt = prompt_map[request["prompt_id"]]
            if current != request["prompt_id"]:
                prepared = _prepared(adapter, prompt, data_root)
                prepared_hash = state_hash(prepared)
                scene_fields = _scene_fields(prompt, prepared["audit"])
                current = request["prompt_id"]
            generation, returned = None, False
            try:
                before = adapter.forward_calls
                with meter.scope(sampling_label), torch.no_grad():
                    generation = adapter.generate(
                        prepared,
                        seed=request["sample_seed"],
                        max_new_tokens=request["max_new_tokens"],
                        do_sample=True,
                    )
                    returned = True
                faults = generation_checks(
                    generation,
                    adapter.processor.tokenizer,
                    adapter.eos_ids,
                    request["max_new_tokens"],
                )
                if (
                    request["interface"] == "IMAGE_CUE_FRESH"
                    and generation.get("vision_forward_calls", 0) < 1
                ):
                    faults.append("image did not enter visual forward")
                annotation = _annotate_rollout(generation["raw_completion"], prompt)
                scores = generation["behavior_token_logprobs"]
                row = _json_safe(
                    {
                        **request,
                        "execution_kind": identity["execution_kind"],
                        "model_id": adapter.model_id,
                        "model_revision": adapter.revision,
                        "adapter_hash": initial_parameters,
                        "train_seed": 17,
                        "optimizer_state_hash": policy_optimizer_hash,
                        **prepared["audit"],
                        **generation,
                        **annotation,
                        **scene_fields,
                        "protocol_version": config["protocol_version"],
                        "input_ids_hash": prepared["audit"].get("tokenized_prompt_hash"),
                        "prepared_hash": prepared_hash,
                        "image_grid_thw": prepared["audit"].get("image_grid_thw"),
                        "actual_image_tokens": prepared["audit"].get("image_token_count"),
                        "n_generated_tokens": len(generation["token_ids"]),
                        "per_token_logprob_behavior": scores,
                        "runtime_forward_by_reason": {"generation": adapter.forward_calls - before},
                        "elapsed": generation["elapsed_seconds"],
                        "elapsed_unit": "seconds",
                        "elapsed_scope": "adapter.generate",
                        "elapsed_status": "CPU_FIXTURE_ADAPTER_RETURN"
                        if identity["execution_kind"].startswith("CPU_")
                        else "MEASURED",
                        "peak_memory": meter.memory.get(sampling_label),
                        "peak_memory_status": "CPU_FIXTURE; CUDA_MODEL_MEMORY_NOT_MEASURED"
                        if identity["execution_kind"].startswith("CPU_")
                        else "CUDA_PHASE_PEAK_AND_PROCESS_RSS_HIGH_WATERMARK",
                        "generation_config_hash": canonical_hash(
                            legacy_lock["generation"]
                            if prompt["track"] == "L"
                            else config["generation_proposed_N"]
                        ),
                        "raw_text": generation["raw_completion"],
                        "raw_token_ids": generation["token_ids"],
                        "old_logprobs": scores,
                        "logprob_sequence": math.fsum(scores),
                        "execution_checks": {"passed": not faults, "faults": faults},
                    }
                )
                row["record_hash"] = canonical_hash(row)
                store.append(row)
                returned = False
                progress.write(
                    "SAMPLING",
                    arm=arm,
                    step=step,
                    role=role,
                    completed=len(store.records),
                    total=len(requests),
                )
                if faults:
                    raise RuntimeError("R4 raw generation failed execution checks")
            except Exception as exc:
                if returned:
                    row = _json_safe(
                        {
                            **request,
                            "execution_kind": identity["execution_kind"],
                            "protocol_version": config["protocol_version"],
                            **scene_fields,
                            **prepared["audit"],
                            "prepared_hash": prepared_hash,
                            "runtime_forward_by_reason": {
                                "generation": adapter.forward_calls - before
                            },
                            "generation_return": generation,
                            "execution_checks": {
                                "passed": False,
                                "faults": [f"{type(exc).__name__}: {exc}"],
                            },
                        }
                    )
                    row["record_hash"] = canonical_hash(row)
                    store.append(row)
                raise
        _check_frozen(adapter, frozen, progress.out)
        if parameter_hash(adapter.model, trainable=True) != initial_parameters or state_hash(
            optimizer.state_dict()
        ) != state_hash(policy["optimizer"]):
            write_json(
                progress.out / "policy_contamination.json", {"arm": arm, "step": step, "role": role}
            )
            raise RuntimeError("R4 sampling changed policy/optimizer")
        validate_sample_ledger(store.records, requests)
        if len(store.records) != len(requests):
            raise ValueError("R4 sampling ledger incomplete")
        return [store.records[r["sample_key"]] for r in requests]
    finally:
        try:
            _check_frozen(adapter, frozen, progress.out)
            if parameter_hash(adapter.model, trainable=True) != initial_parameters or state_hash(
                optimizer.state_dict()
            ) != state_hash(policy["optimizer"]):
                raise RuntimeError("R4 generation policy/Adam contamination")
        except Exception as exc:
            write_json(
                progress.out / "policy_contamination.json",
                {"arm": arm, "step": step, "reason": str(exc)},
            )
            raise
        finally:
            _restore(adapter, optimizer, policy)


def _control_probe(adapter, optimizer, origin, r3_dir, r3_binding, config, data_root):
    root = Path(r3_dir)
    runtime = _json(root / "runtime_lock.json")
    manifest = _json(root / "bank_manifest.json")
    old_origin = load_checkpoint(
        root / "origin.pt", {**runtime["identity"], "unit": "initial_origin"}
    )
    if (
        state_hash(old_origin["parameters"]) != state_hash(origin["parameters"])
        or state_hash(old_origin["optimizer"]) != state_hash(origin["optimizer"])
        or old_origin["optimizer"]["state"]
        or origin["optimizer"]["state"]
    ):
        raise PilotBlocked(
            "R3 proposal model/initial adapter/empty Adam "
            "differs from R4; no new control budget author"
            "ized"
        )
    if runtime["frozen_base_hash"] != parameter_hash(adapter.model, trainable=False):
        raise PilotBlocked("R3 proposal frozen model differs from R4")
    all_rows = [
        json.loads(line)
        for line in (root / "samples.jsonl").read_text().splitlines()
        if line.strip()
    ]
    rows = [r for r in all_rows if r["bank_role"] == "control_proposal" and r["sample_index"] == 0]
    prompts = manifest["plan"]["control_prompts"]
    if len(rows) != 48 or {r["prompt_id"] for r in rows} != {p["prompt_id"] for p in prompts}:
        raise PilotBlocked("Fixed R3 sample-index-zero probe coverage differs from 48 prompts")
    by_prompt = {p["prompt_id"]: p for p in prompts}
    initial_hash = parameter_hash(adapter.model, trainable=True)
    for row in rows:
        prompt = by_prompt[row["prompt_id"]]
        audit = _prepared(adapter, prompt, data_root)["audit"]
        if (
            row["record_hash"]
            != canonical_hash({k: v for k, v in row.items() if k != "record_hash"})
            or row.get("execution_checks", {}).get("passed") is not True
            or row["generation_config_hash"] != canonical_hash(config["generation_proposed_N"])
            or row["adapter_hash"] != initial_hash
            or row["model_id"] != adapter.model_id
            or row["model_revision"] != adapter.revision
            or row["prompt_hash"] != prompt["prompt_hash"]
            or any(
                row.get(key) != audit.get(key)
                for key in (
                    "final_prompt_hash",
                    "input_tensor_hash",
                    "tokenized_prompt_hash",
                    "pixel_values_hash",
                )
            )
            or generation_checks(row, adapter.processor.tokenizer, adapter.eos_ids, 64)
        ):
            raise PilotBlocked(
                "R3 fixed proposal input/generation/source mismatch; no resampling fallback"
            )
    binding = {
        "selection": (
            "Each of the 48 predeclared R3-co"
            "ld control prompts, sample_index"
            " == 0; selected before outcomes"
        ),
        "source_r3_gate_hash": canonical_hash(r3_binding),
        "source_samples_sha256": file_hash(root / "samples.jsonl"),
        "sample_keys": [r["sample_key"] for r in rows],
        "record_hashes": [r["record_hash"] for r in rows],
        "new_control_generations": 0,
        "sequences_per_step": 48,
        "planned_sequence_scores": 6144,
        "noise_limit": (
            "One proposal completion per prom"
            "pt; fixed-prefix diagnostic, not"
            " current occupancy trajectory KL"
        ),
        "origin_comparison": (
            "Exact parameters and empty Adam; R3/R4 metadata and RNG need not match"
        ),
    }
    return rows, by_prompt, binding


def _scores(adapter, optimizer, rows, prompts, out, identity, data_root, meter, progress, label):
    parameters = parameter_hash(adapter.model, trainable=True)
    adam = state_hash(optimizer.state_dict())
    frozen = _frozen_versions(adapter)
    identity = {**identity, "scored_parameter_hash": parameters, "scored_optimizer_hash": adam}
    try:
        return _scores_unchecked(
            adapter, rows, prompts, out, identity, data_root, meter, progress, label
        )
    finally:
        try:
            _check_frozen(adapter, frozen, progress.out)
            if (
                parameter_hash(adapter.model, trainable=True) != parameters
                or state_hash(optimizer.state_dict()) != adam
            ):
                raise RuntimeError("R4 likelihood changed current policy/Adam")
        except Exception as exc:
            write_json(
                progress.out / "policy_contamination.json", {"phase": label, "reason": str(exc)}
            )
            raise


def _scores_unchecked(adapter, rows, prompts, out, identity, data_root, meter, progress, label):
    import torch

    store = RunStore(out, identity, resume=(out / "identity.json").exists())
    requests = [
        {
            "sample_key": canonical_hash([identity, row["sample_key"]]),
            "proposal_sample_key": row["sample_key"],
            "proposal_record_hash": row["record_hash"],
            "token_ids": row["token_ids"],
            "execution_kind": identity["execution_kind"],
            "scored_parameter_hash": identity["scored_parameter_hash"],
            "scored_optimizer_hash": identity["scored_optimizer_hash"],
        }
        for row in rows
    ]
    validate_sample_ledger(store.records, requests)
    for row in store.records.values():
        values = row.get("new_token_logprobs")
        if (
            not isinstance(values, list)
            or len(values) != len(row["token_ids"])
            or any(type(v) not in (int, float) or not math.isfinite(v) or v > 0 for v in values)
        ):
            raise ValueError("R4 saved likelihood values invalid")
    current, prepared = None, None
    for row, request in zip(rows, requests, strict=True):
        if request["sample_key"] in store.keys:
            continue
        if current != row["prompt_id"]:
            prepared = _prepared(adapter, prompts[row["prompt_id"]], data_root)
            current = row["prompt_id"]
        if any(
            row.get(key) != prepared["audit"].get(key)
            for key in ("final_prompt_hash", "input_tensor_hash")
        ):
            write_json(
                progress.out / "measurement_fault.json",
                {"phase": label, "reason": "Scored continuation input changed", "request": request},
            )
            raise ValueError("R4 scored continuation input changed")
        returned, score = False, None
        try:
            with meter.scope(label), torch.no_grad():
                score = adapter.logprobs(prepared, row["token_ids"])
                returned = True
            values = score.detach().cpu().tolist()
            if len(values) != len(row["token_ids"]) or not all(
                math.isfinite(v) and v <= 0 for v in values
            ):
                raise ValueError("R4 likelihood token count/non-finite fault")
            record = {**request, "new_token_logprobs": values, "execution_checks": {"passed": True}}
            record["record_hash"] = canonical_hash(record)
            store.append(record)
            returned = False
            progress.write(
                "CONTROL_SCORING", label=label, completed=len(store.records), total=len(rows)
            )
        except Exception as exc:
            if returned:
                record = _json_safe(
                    {
                        **request,
                        "likelihood_return": score.detach().cpu().tolist()
                        if hasattr(score, "detach")
                        else score,
                        "execution_checks": {"passed": False, "faults": [str(exc)]},
                    }
                )
                record["record_hash"] = canonical_hash(record)
                store.append(record)
                write_json(
                    progress.out / "measurement_fault.json",
                    {
                        "phase": label,
                        "reason": str(exc),
                        "request": request,
                        "raw_failure_record_hash": record["record_hash"],
                    },
                )
            raise
    return {r["proposal_sample_key"]: r["new_token_logprobs"] for r in store.records.values()}


def _assert_finite_state(state, out, phase):
    import torch

    def finite(value):
        if isinstance(value, torch.Tensor):
            return not value.is_floating_point() or bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return all(finite(item) for item in value)
        return not isinstance(value, float) or math.isfinite(value)

    if not finite({"parameters": state["parameters"], "optimizer": state["optimizer"]}):
        write_json(
            out / "measurement_fault.json",
            {"phase": phase, "reason": "Nonfinite post-update parameter or Adam state"},
        )
        raise FloatingPointError("Nonfinite post-update parameter or Adam state")


def _step(
    adapter,
    optimizer,
    prestate,
    rows,
    prompts,
    probe,
    control_prompts,
    root,
    identity,
    arm,
    step,
    data_root,
    meter,
    progress,
):
    import numpy as np

    from .grpo_update import grouped_advantages, perform_update, reward_channels
    from .r4_metrics import control_kl_diagnostic

    unit_identity = {
        **identity,
        "unit": "training_step",
        "arm": arm,
        "step": step,
        "prestate_hash": state_hash(prestate),
        "sample_hash": canonical_hash([r["record_hash"] for r in rows]),
        "probe_hash": canonical_hash([r["record_hash"] for r in probe]),
    }
    versions = _frozen_versions(adapter)

    def operation(attempt):
        try:
            _restore(adapter, optimizer, prestate)
            prefix = f"{arm}/step_{step:02d}/{attempt.name}"
            parity = _scores(
                adapter,
                optimizer,
                rows,
                prompts,
                attempt / "preupdate_parity",
                {**unit_identity, "unit": "preupdate_parity"},
                data_root,
                meter,
                progress,
                prefix + "/preupdate_parity",
            )
            errors = [
                abs(a - b)
                for row in rows
                for a, b in zip(parity[row["sample_key"]], row["old_logprobs"], strict=True)
            ]
            parity_report = {
                "mean_abs_token_error": float(np.mean(errors)),
                "p99_abs_token_error": float(np.quantile(errors, 0.99)),
                "tokens": len(errors),
            }
            write_json(attempt / "parity.json", parity_report)
            if (
                parity_report["mean_abs_token_error"] > 0.02
                or parity_report["p99_abs_token_error"] > 0.1
            ):
                write_json(
                    progress.out / "measurement_fault.json",
                    {
                        "phase": prefix,
                        "reason": "On-policy behavior/teacher forcing parity failed",
                        "parity": parity_report,
                    },
                )
                raise RuntimeError("R4 on-policy behavior/teacher forcing parity failed")
            groups = []
            for prompt_id in dict.fromkeys(r["prompt_id"] for r in rows):
                group = [r for r in rows if r["prompt_id"] == prompt_id]
                if len(group) != 8 or any(
                    r["arm"] != arm or r["bank_role"] != "train" or r["split"] != "train"
                    for r in group
                ):
                    raise ValueError("R4 gradient group mixes arms/control or misses trajectories")
                prepared = _prepared(adapter, prompts[prompt_id], data_root)
                groups.append(
                    [
                        {
                            **r,
                            "prepared": prepared,
                            "reward_sum": reward_channels(r["category"], arm)["sum"],
                        }
                        for r in group
                    ]
                )
            if len(groups) != 4:
                raise ValueError("R4 requires B4 K8")
            group_statistics = []
            for group in groups:
                rewards = [r["reward_sum"] for r in group]
                group_statistics.append(
                    {
                        "prompt_id": group[0]["prompt_id"],
                        "group_id": group[0]["group_id"],
                        "K": len(group),
                        "lambda": 0.0 if arm == "X_BASE" else 1.0,
                        "category_counts": {
                            category: sum(r["category"] == category for r in group)
                            for category in "XSWI"
                        },
                        "reward_vector": rewards,
                        "sample_keys": [r["sample_key"] for r in group],
                        **grouped_advantages(rewards),
                    }
                )
            write_json(attempt / "group_statistics.json", group_statistics)
            before_steps = meter.report()["optimizer_step_calls_observed"]
            try:
                with meter.scope(prefix + "/update_and_training_postscore"):
                    update = perform_update(
                        adapter, optimizer, groups, lnorm=64, clip_epsilon=0.2, grad_clip=1.0
                    )
            except (FloatingPointError, OverflowError, RuntimeError) as exc:
                if isinstance(exc, (FloatingPointError, OverflowError)) or any(
                    word in str(exc).lower()
                    for word in ("non-finite", "nonfinite", "nan", "infinity")
                ):
                    write_json(
                        progress.out / "measurement_fault.json",
                        {"phase": prefix, "reason": str(exc)},
                    )
                raise
            if (
                meter.report()["optimizer_step_calls_observed"] - before_steps != 1
                or update["optimizer_updates"] != 1
                or update["backward_calls"] != 32
            ):
                raise RuntimeError(
                    "R4 actual optimizer/backward count differs from single B4K8 update"
                )
            _check_frozen(adapter, versions, progress.out)
            # JSON encoding rejects non-finite measured scalars without coercion.
            try:
                canonical_hash(update)
            except ValueError as exc:
                write_json(
                    progress.out / "measurement_fault.json", {"phase": prefix, "reason": str(exc)}
                )
                raise
            post = capture_state(
                adapter.model,
                optimizer,
                {
                    "arm": arm,
                    "checkpoint_step": step,
                    "sampler": {"plan_hash": identity["plan_hash"], "position": step},
                    "completed_step_sample_keys": [r["sample_key"] for r in rows],
                    "completed_sample_keys": [
                        *prestate["metadata"].get("completed_sample_keys", []),
                        *[r["sample_key"] for r in rows],
                    ],
                    "scheduler": None,
                    "grad_scaler": None,
                },
            )
            _assert_finite_state(post, progress.out, prefix)
            checkpoint_identity = {**unit_identity, "unit": "post_update_checkpoint"}
            save_checkpoint(attempt / "checkpoint.pt", post, checkpoint_identity)
            write_json(attempt / "update.json", update)
            candidate = _scores(
                adapter,
                optimizer,
                probe,
                control_prompts,
                attempt / "control_scores",
                {
                    **unit_identity,
                    "unit": "fixed_step0_control",
                    "poststate_hash": state_hash(post),
                },
                data_root,
                meter,
                progress,
                prefix + "/fixed_step0_control",
            )
            try:
                diagnostic = control_kl_diagnostic(probe, candidate, eos_token_ids=adapter.eos_ids)
            except (ValueError, FloatingPointError, OverflowError) as exc:
                write_json(
                    progress.out / "measurement_fault.json",
                    {"phase": prefix, "reason": str(exc), "diagnostic": "fixed_control_KL"},
                )
                raise
            if diagnostic["should_stop"]:
                # Persist the measured stop before any local artifact, checkpoint
                # publication, or atomic completion can be interrupted.
                write_json(
                    progress.out / "alarm_stop.json",
                    {
                        "status": "DIAGNOSTIC_STOP",
                        "arm": arm,
                        "step": step,
                        "unit_identity": unit_identity,
                        "attempt": str(attempt),
                        "checkpoint_path": str(attempt / "checkpoint.pt"),
                        "checkpoint_identity": checkpoint_identity,
                        "control_diagnostic": diagnostic,
                    },
                )
            write_json(attempt / "control_diagnostic.json", diagnostic)
            _restore(adapter, optimizer, post)
            return {
                "status": "DIAGNOSTIC_STOP" if diagnostic["should_stop"] else "PASS",
                "arm": arm,
                "step": step,
                "checkpoint_identity": checkpoint_identity,
                "checkpoint_path": "checkpoint.pt",
                "checkpoint_sha256": file_hash(attempt / "checkpoint.pt"),
                "state_hash": state_hash(post),
                "parameter_hash": parameter_hash(adapter.model, trainable=True),
                "optimizer_state_hash": state_hash(post["optimizer"]),
                "update": update,
                "group_statistics": group_statistics,
                "parity": parity_report,
                "control_diagnostic": diagnostic,
                "training_category_counts": {
                    category: sum(r["category"] == category for r in rows) for category in "XSWI"
                },
                "zero_advantage_groups": sum(
                    all(record["advantage"] == 0 for record in update["token_records"][i : i + 8])
                    for i in range(0, 32, 8)
                ),
                "control_sequences_scored": len(candidate),
                "sample_keys": [r["sample_key"] for r in rows],
            }
        finally:
            _restore(adapter, optimizer, prestate)

    attempt, result, reused = run_atomic_unit(root, unit_identity, operation)
    post = load_checkpoint(attempt / result["checkpoint_path"], result["checkpoint_identity"])
    if state_hash(post) != result["state_hash"]:
        raise ValueError("R4 post-update checkpoint mismatch")
    _restore(adapter, optimizer, post)
    return post, {**result, "attempt": str(attempt), "reused_completed_unit": reused}


def _historical_initial(adapter, plan, r0_dir, config, data_root, *, fixture=False):
    """Reuse only manifest-verified P3 rows with exact effective protocol/input alignment."""
    from .legacy_frozen import annotate_legacy

    results, evidence = {}, {}
    if fixture:
        return results, {
            "status": "NOT_MEASURED",
            "reason": "CPU fixture has no historical CUDA alignment",
        }
    closure = _json(Path(r0_dir) / "closure_checks.json")
    lora_b = [
        (name, p)
        for name, p in adapter.model.named_parameters()
        if p.requires_grad and "lora_B" in name
    ]
    zero_adapter = bool(lora_b) and all(not p.detach().count_nonzero().item() for _, p in lora_b)
    for track, key in (("N", "dev_prompts"), ("L", "legacy_prompts")):
        try:
            if not zero_adapter:
                raise NoComparableInitial("Fresh LoRA zero-B equivalence not established")
            audited = closure["base_" + track]
            root = Path(config["paths"]["readonly_" + track])
            if root.resolve() != Path(audited["source_root"]).resolve():
                raise ValueError("P3 source path differs from R0 audited root")
            if file_hash(root / "manifest.json") != audited["raw_manifest_sha256"]:
                raise ValueError("P3 manifest hash differs from R0")
            _verify_manifest(root, _json(root / "manifest.json"))
            for relative, digest in audited["raw_files"].items():
                if file_hash(root / relative) != digest:
                    raise ValueError("P3 file differs from R0 raw evidence")
            lock = _json(root / "runtime_lock.json")
            baseline = _json(Path(r0_dir) / f"baseline_lock_{track}.json")
            generation = {
                "do_sample": True,
                "max_new_tokens": 48 if track == "L" else 64,
                "min_p": 0.0,
                "num_beams": 1,
                "repetition_penalty": 1.0,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
            }
            if (
                lock["generation_protocol"] != generation
                or lock["optimizer_updates"] != 0
                or lock["identity"]["model_id"] != adapter.model_id
                or lock["identity"]["model_revision"] != adapter.revision
                or baseline["generation"] != generation
                or lock["adapter_state"] != "fresh base; adapter disabled; no checkpoint loaded"
            ):
                raise NoComparableInitial("P3 effective generation/model/base state differs")
            for audit_key in (
                "processor_hash",
                "tokenizer_hash",
                "chat_template_hash",
                "frozen_parameter_hash",
                "probability_execution",
            ):
                if lock["model_audit"][audit_key] != adapter.audit[audit_key]:
                    raise NoComparableInitial(
                        "P3 processor/model/path alignment differs: " + audit_key
                    )
            raw = [
                json.loads(line)
                for line in (root / "samples.jsonl").read_text().splitlines()
                if line.strip()
            ]
            prompts = {(p["base_scene_id"], p["interface"]): p for p in plan[key]}
            audits = {
                pair: _prepared(adapter, prompt, data_root)["audit"]
                for pair, prompt in prompts.items()
            }
            aligned, seen = [], set()
            for row in raw:
                if row["decode_mode"] != "sample":
                    continue
                pair = (row["base_scene_id"], row["interface"])
                if pair not in prompts:
                    raise ValueError("P3 baseline contains an unaligned sampled prompt")
                prompt, audit = prompts[pair], audits[pair]
                if (
                    row["sample_key"] in seen
                    or row["record_hash"]
                    != canonical_hash({k: v for k, v in row.items() if k != "record_hash"})
                    or row["execution_kind"] != "REAL_CUDA_MODEL"
                    or row["optimizer_step"] != 0
                    or row.get("policy_base_identical") is not True
                    or row.get("on_policy_likelihood_passed") is not True
                    or any(
                        row.get(k) != audit.get(k)
                        for k in (
                            "final_prompt_hash",
                            "input_tensor_hash",
                            "tokenized_prompt_hash",
                            "pixel_values_hash",
                        )
                    )
                    or generation_checks(
                        row,
                        adapter.processor.tokenizer,
                        adapter.eos_ids,
                        generation["max_new_tokens"],
                    )
                ):
                    raise ValueError("P3 raw generation or prepared-input alignment fails")
                seen.add(row["sample_key"])
                annotation = (annotate_legacy if track == "L" else annotate_diagnostic)(
                    row["raw_completion"], prompt["scene"]
                )
                if row["category"] != annotation["category"]:
                    raise ValueError("P3 original category differs from current locked parser")
                aligned.append(
                    {
                        **row,
                        "source_prompt_id": row["prompt_id"],
                        "prompt_id": prompt["prompt_id"],
                        "source_record_hash": row["record_hash"],
                        "arm": "INITIAL",
                        "checkpoint_step": 0,
                        "family": prompt["family"],
                        "track": track,
                        "initial_source": "R0_verified_P3",
                    }
                )
            counts = {
                p: sum(row["prompt_id"] == p for row in aligned)
                for p in (r["prompt_id"] for r in plan[key])
            }
            if set(counts.values()) != {16}:
                raise ValueError("P3 baseline expected exactly K16 per fixed prompt")
            results[track] = aligned
            evidence[track] = {
                "status": "PASS",
                "source_root": str(root),
                "raw_manifest_sha256": audited["raw_manifest_sha256"],
                "sample_count": len(aligned),
                "effective_generation": generation,
                "zero_lora_B_matrices": len(lora_b),
                "comparison": (
                    "P3 disabled adapter equals fresh"
                    "ly initialized zero-B LoRA; R1 c"
                    "ertificate additionally verifies"
                    " base likelihood parity"
                ),
                "input_alignment": (
                    "Every sampled prompt/input/pixel hash and strict category verified"
                ),
                "new_generations": 0,
            }
        except NoComparableInitial as exc:
            evidence[track] = {
                "status": "NOT_MEASURED",
                "reason": f"{type(exc).__name__}: {exc}",
                "new_generations": 0,
            }
    return results, evidence


def _reports(out, endpoint_rows, initial_rows, details):
    from .r4_metrics import analyze_sampled_endpoints

    results, effects = {}, []
    for track in ("N", "L", "OOD"):
        rows = [r for r in endpoint_rows if r["track"] == track]
        results[track] = analyze_sampled_endpoints(rows, initial_rows.get(track), track=track)
        for comparison, value in results[track]["comparisons"].items():
            for scope, metrics in value.get("responses", {}).items():
                for metric, values in metrics.items():
                    row = {
                        "track": track,
                        "comparison": comparison,
                        "scope": scope,
                        "metric": metric,
                    }
                    for key, value in values.items():
                        if isinstance(value, dict):
                            row.update({f"{key}_{k}": v for k, v in value.items()})
                        else:
                            row[key] = value
                    effects.append(row)
        if track == "L":
            groups = {r["family"] + "/" + r["interface"] for r in rows}
            sensitivity = analyze_sampled_endpoints(
                rows,
                initial_rows.get(track),
                track=track,
                group_weights={g: 1 / len(groups) for g in groups},
            )
            write_json(out / "L_family_equal_sensitivity.json", sensitivity)
    write_json(out / "endpoint_metrics.json", results)
    _write_csv(out / "N_L_OOD_effects.csv", effects)
    report = "\n".join(
        [
            "# R4 两臂实际运行",
            "",
            f"状态：{details['status']}；执行类型：{details['execution_kind']}。",
            "",
            (
                "两臂从相同 seed 17 LoRA/空 Adam 起点顺序执行，各自使用当前策略新生成 "
                "B4 K8 轨迹；每步一次实际 Adam 更新。"
            ),
            (
                "每步 32 次反向、32 条更新前 parity 评分、32 条 perform_upda"
                "te 内部更新后训练序列评分，与 48 条固定起点 control 评分分别记录；真实 E"
                "OS 保留。"
            ),
            (
                "固定 control 按 R3-cold 每个 prompt 的 sample_index"
                "=0 事前取 48 条，保留原 key 和 source hash；没有新增 contro"
                "l 生成。"
            ),
            (
                "KL 是固定旧前缀条件方向，六群体等权，不是当前策略 occupancy 的轨迹 KL。超"
                "线步骤保留并停止。"
            ),
            (
                "所有成功 step attempt 不可变；中断 attempt 保留，重算计入实际调用。"
                "0/16/32/64 checkpoint 包含完整 Adam、RNG、sampler 和"
                "原始 sample keys。"
            ),
            (
                "N64、原 L48 和 graph OOD 分别报告。初始 P3 只在原 R0 文件、模型"
                "、模板、输入与生成协议严格对齐后复用；N 无可用 P3 时变化仅用事前固定共享 step0"
                " 子 panel。"
            ),
            (
                "L 主结果按原固定题分布，家族等权另列敏感性。"
                "低正确率和未观测 X 不触发执行失败，不筛掉零奖励组。"
            ),
            (
                "base_scene 家族分层配对 bootstrap 5000 次；max-stat 仅"
                "同一指标的声明群体内，单训练 seed 为探索性结果。"
            ),
            "未执行 A 训练臂、新模型、在线 SSVC、额外 SFT 或 R3-warm。",
            "",
            "```json",
            json.dumps(details, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    (out / "pilot_report.md").write_text(report, encoding="utf-8")
    (out / "report_zh.md").write_text(report, encoding="utf-8")


def _finish(out, status, execution, details):
    if status != "PASS" or not (out / "report_zh.md").exists():
        report = (
            f"# R4 {status}\n\n执行类型：{execution}。\n\n"
            + json.dumps(details, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )
        out.mkdir(parents=True, exist_ok=True)
        (out / "report_zh.md").write_text(report, encoding="utf-8")
        (out / "pilot_report.md").write_text(report, encoding="utf-8")
    write_json(out / "progress.json", {"state": status, **details})
    artifacts = [
        out / f["path"]
        for f in _manifest(out, exclude=("status.json", "manifest.json", "report.md"))["files"]
    ]
    result = phase_artifacts(out, "R4", status, details, artifacts)
    result["execution_kind"] = execution
    write_json(out / "status.json", result)
    return result


def run_r4(
    config,
    data_root,
    out,
    r0_dir,
    r1_run,
    supplement_dir,
    r2_dir,
    r3_cold_dir,
    *,
    allow_training=False,
    resume=False,
    _adapter_factory=None,
):
    out = Path(out)
    if not allow_training:
        if out.exists() and any(out.iterdir()):
            raise PilotBlocked("Explicit --allow-training is required; existing evidence preserved")
        return _finish(
            out,
            "BLOCKED",
            "CPU_AUDIT",
            {
                "training_started": False,
                "reason": (
                    "Requires explicit --allow-traini"
                    "ng; canonical YAML allow_trainin"
                    "g remains unchanged"
                ),
            },
        )
    import torch

    from .model_adapters import load_adapter
    from .next_stage_runtime import (
        validate_config_against_gate,
        validate_prerequisites,
        validate_r2_gate,
        validate_r3_cold_gate,
        validate_runtime_environment,
    )

    gate = validate_prerequisites(r0_dir, r1_run, supplement_dir)
    config = validate_config_against_gate(config, gate)
    r2_binding = validate_r2_gate(r2_dir, gate)
    r3_binding = validate_r3_cold_gate(r3_cold_dir, gate, r2_binding)
    if Path(data_root).resolve() != Path(config["data_root"]).resolve():
        raise ValueError("R4 data root differs from certified data")
    _helper_config(config)
    if any(
        config["R4"][key] != wanted
        for key, wanted in {
            "steps": 64,
            "B": 4,
            "K": 8,
            "seed": 17,
            "microbatch": 1,
            "checkpoint_steps": [0, 16, 32, 64],
            "arms": {"X_BASE": {"lambda": 0.0}, "X_VALID": {"lambda": 1.0}},
        }.items()
    ):
        raise ValueError("R4 fixed two-arm training contract changed")
    plan, data_binding, legacy_lock = _load_plan(data_root, gate, r0_dir)
    fixture = _adapter_factory is not None
    execution = "CPU_FAKE_ADAPTER_FIXTURE" if fixture else "REAL_CUDA_TRAINING"
    if not fixture and not torch.cuda.is_available():
        raise RuntimeError("R4 requires an allocated CUDA device")
    environment = (
        {"status": "CPU_FIXTURE_NOT_REAL_ENVIRONMENT"}
        if fixture
        else validate_runtime_environment(gate)
    )
    if not fixture and (
        environment != r2_binding["environment"] or environment != r3_binding["environment"]
    ):
        raise ValueError("R4 environment differs from measured R2/R3")
    source = _source()
    identity = {
        "phase": "R4",
        "model_hash": canonical_hash(config["model"]),
        "config_hash": canonical_hash(config),
        "data_hash": canonical_hash(data_binding),
        "source_hash": canonical_hash(source),
        "gate_hash": canonical_hash(gate["binding"]),
        "r2_gate_hash": canonical_hash(r2_binding),
        "r3_cold_gate_hash": canonical_hash(r3_binding),
        "plan_hash": plan["plan_hash"],
        "initial_adapter_hash": gate["certificate"]["initial_adapter_hash"],
        "execution_kind": execution,
    }
    with frozen_writer(out):
        if (out / "measurement_fault.json").exists():
            raise PilotBlocked(
                "R4 preserved failed measurement cannot be bypassed by a new attempt"
            )
        if (out / "policy_contamination.json").exists():
            raise PilotBlocked("R4 contaminated evidence cannot resume")
        RunStore(out, identity, resume=resume)
        if (out / "alarm_stop.json").exists():
            return _finish(
                out,
                "BLOCKED",
                execution,
                {
                    "training_started": True,
                    "preserved_stop": _json(out / "alarm_stop.json"),
                    "reason": (
                        "Fixed control alarm remains pend"
                        "ing diagnosis; resume cannot ski"
                        "p the offending batch"
                    ),
                },
            )
        write_json(
            out / "gate_binding.json",
            {"r0_r1": gate["binding"], "r2": r2_binding, "r3_cold": r3_binding},
        )
        invocations = out / "invocations"
        invocations.mkdir(exist_ok=True)
        invocation = invocations / f"attempt_{len(list(invocations.iterdir())):04d}"
        invocation.mkdir()
        adapter = optimizer = origin = meter = progress = None
        status, details = "FAIL", {"execution_kind": execution, "training_started": False}
        summaries, endpoint, panel32, checkpoints = [], [], [], []
        try:
            _seed_everything(17)
            adapter = (_adapter_factory or load_adapter)(
                "qwen35_9b",
                {
                    "id": config["model"]["id"],
                    "revision": config["model"]["revision"],
                    "expected_layers": 32,
                },
                image_token_limit=768,
            )
            if (
                adapter.model_id != config["model"]["id"]
                or adapter.revision != config["model"]["revision"]
            ):
                raise ValueError("R4 actual model identity mismatch")
            frozen_hash = _certificate_check(gate["certificate"], adapter, config)
            if not fixture:
                environment = validate_runtime_environment(gate, adapter.audit)
                if (
                    environment != r2_binding["environment"]
                    or environment != r3_binding["environment"]
                ):
                    raise ValueError("R4 loaded environment drift")
            options = config["optimizer"]
            optimizer = torch.optim.AdamW(
                [p for p in adapter.model.parameters() if p.requires_grad],
                lr=options["learning_rate"],
                betas=tuple(options["betas"]),
                eps=options["eps"],
                weight_decay=options["weight_decay"],
            )
            origin = capture_state(
                adapter.model,
                optimizer,
                {
                    "checkpoint_step": 0,
                    "arm": "INITIAL",
                    "completed_sample_keys": [],
                    "sampler": {"plan_hash": plan["plan_hash"], "position": 0},
                    "scheduler": None,
                    "grad_scaler": None,
                },
            )
            if origin["optimizer"]["state"]:
                raise ValueError("R4 requires fresh empty Adam")
            origin_identity = {**identity, "unit": "initial_origin"}
            if (out / "origin.pt").exists():
                if state_hash(load_checkpoint(out / "origin.pt", origin_identity)) != state_hash(
                    origin
                ):
                    raise ValueError("R4 fresh seed17 initial state differs on resume")
            else:
                save_checkpoint(out / "origin.pt", origin, origin_identity)
            runtime = {
                "identity": identity,
                "config": config,
                "environment": environment,
                "source": source,
                "model_audit": {
                    k: v
                    for k, v in adapter.audit.items()
                    if k not in ("load_seconds", "load_peak_cuda_bytes")
                },
                "origin_hash": state_hash(origin),
                "origin_file_sha256": file_hash(out / "origin.pt"),
                "frozen_base_hash": frozen_hash,
                "initial_adapter_hash": parameter_hash(adapter.model, trainable=True),
                "optimizer_initial_hash": state_hash(origin["optimizer"]),
                "selected_probability_path": "uncached_prefix_recompute",
                "scheduler": None,
                "gradient_scaler": None,
                "authorization": {
                    "allow_training_flag": True,
                    "canonical_config_allow_training": config["R4"]["allow_training"],
                },
            }
            if (out / "runtime_lock.json").exists() and _json(out / "runtime_lock.json") != runtime:
                raise ValueError("R4 strict runtime resume binding changed")
            write_json(invocation / "model_load_audit.json", adapter.audit)
            if not (out / "runtime_lock.json").exists():
                write_json(out / "runtime_lock.json", runtime)
            meter = _ExecutionMeter(adapter, optimizer)
            if not fixture and (
                len(meter.language_layer_names) != 32
                or not meter.top_hook_installed
                or meter.vision_hook_count != 1
            ):
                raise RuntimeError("R4 real forward/backward instrumentation incomplete")
            progress = _Progress(out, invocation, meter)
            probe, control_prompts, probe_binding = _control_probe(
                adapter, optimizer, origin, r3_cold_dir, r3_binding, config, data_root
            )
            write_json(
                out / "two_arm_training_config.json",
                {
                    "canonical_config": config,
                    "authorization": runtime["authorization"],
                    "plan": plan,
                    "data_binding": data_binding,
                    "control_probe": probe_binding,
                    "same_seed_contract": (
                        "Both arms reset to exactly the s"
                        "aved origin including RNG; ident"
                        "ical fixed prompt order, separat"
                        "e current-policy trajectories"
                    ),
                    "cost_contract": {
                        "new_outputs": 14400,
                        "training": 4096,
                        "shared_step0": 576,
                        "step32": 1152,
                        "step64": 8576,
                        "fixed_control_scores": 6144,
                        "preupdate_parity_scores": 4096,
                        "postupdate_training_scores": 4096,
                    },
                },
            )
            historical, alignment = _historical_initial(
                adapter, plan, r0_dir, config, data_root, fixture=fixture
            )
            write_json(out / "initial_alignment.json", alignment)
            _restore(adapter, optimizer, origin)
            shared = _samples(
                adapter,
                optimizer,
                origin,
                plan["dev_panel_prompts"],
                out / "evaluation/INITIAL/step_00/N",
                identity,
                "INITIAL",
                0,
                "evaluation",
                data_root,
                config,
                legacy_lock,
                meter,
                progress,
            )
            initial_rows = {**historical}
            initial_rows.setdefault("N", shared)
            train_prompts = {p["prompt_id"]: p for p in plan["train_prompts"]}
            for arm in ARMS:
                _restore(adapter, optimizer, origin)
                state = origin
                checkpoints.append(
                    {
                        "arm": arm,
                        "step": 0,
                        "checkpoint_path": str((out / "origin.pt").resolve()),
                        "checkpoint_identity": origin_identity,
                        "state_hash": state_hash(origin),
                        "milestone": True,
                    }
                )
                for step, prompt_ids in enumerate(plan["train_steps"], 1):
                    prompts = [train_prompts[p] for p in prompt_ids]
                    rows = _samples(
                        adapter,
                        optimizer,
                        state,
                        prompts,
                        out / arm / f"step_{step:02d}" / "rollouts",
                        identity,
                        arm,
                        step - 1,
                        "train",
                        data_root,
                        config,
                        legacy_lock,
                        meter,
                        progress,
                    )
                    details["training_started"] = True
                    state, summary = _step(
                        adapter,
                        optimizer,
                        state,
                        rows,
                        train_prompts,
                        probe,
                        control_prompts,
                        out / arm / f"step_{step:02d}" / "updates",
                        identity,
                        arm,
                        step,
                        data_root,
                        meter,
                        progress,
                    )
                    summaries.append(summary)
                    checkpoint = {
                        k: summary[k]
                        for k in (
                            "arm",
                            "step",
                            "checkpoint_identity",
                            "state_hash",
                            "parameter_hash",
                            "optimizer_state_hash",
                        )
                    }
                    checkpoint.update(
                        {
                            "checkpoint_path": str(
                                (Path(summary["attempt"]) / summary["checkpoint_path"]).resolve()
                            ),
                            "checkpoint_sha256": summary["checkpoint_sha256"],
                            "milestone": step in (16, 32, 64),
                        }
                    )
                    checkpoints.append(checkpoint)
                    write_json(
                        out / "checkpoint_manifest.json",
                        {
                            "identity": identity,
                            "checkpoints": checkpoints,
                            "X_BASE_step64_for_R3_warm": next(
                                (
                                    c
                                    for c in checkpoints
                                    if c["arm"] == "X_BASE" and c["step"] == 64
                                ),
                                None,
                            ),
                        },
                    )
                    _write_csv(
                        out / "learning_curves.csv",
                        [
                            {
                                "arm": s["arm"],
                                "step": s["step"],
                                "mean_token_kl": s["control_diagnostic"]["mean_token_kl"],
                                "sequence_log_ratio_p99_abs": s["control_diagnostic"][
                                    "sequence_log_ratio_p99_abs"
                                ],
                                "pX_training_bank": s["training_category_counts"]["X"] / 32,
                                "v_training_bank": 1 - s["training_category_counts"]["I"] / 32,
                                "zero_advantage_groups_retained": s["zero_advantage_groups"],
                                "loss": s["update"]["loss"],
                                "grad_norm_preclip": s["update"]["grad_norm_preclip"],
                                "actual_step_norm": s["update"]["actual_step_norm"],
                                "status": s["status"],
                            }
                            for s in summaries
                        ],
                    )
                    if summary["status"] != "PASS":
                        write_json(out / "alarm_stop.json", summary)
                        raise PilotBlocked(
                            "Fixed step0 control KL/log-ratio threshold cr"
                            "ossed; offending step retained"
                        )
                    if (
                        step in (16, 32, 64)
                        and parameter_hash(adapter.model, trainable=False) != frozen_hash
                    ):
                        write_json(
                            out / "policy_contamination.json",
                            {"arm": arm, "step": step, "reason": "frozen full hash changed"},
                        )
                        raise RuntimeError("R4 frozen full hash changed")
                    if step == 32:
                        panel32.extend(
                            _samples(
                                adapter,
                                optimizer,
                                state,
                                plan["dev_panel_prompts"],
                                out / "evaluation" / arm / "step_32/N",
                                identity,
                                arm,
                                step,
                                "evaluation",
                                data_root,
                                config,
                                legacy_lock,
                                meter,
                                progress,
                            )
                        )
                    if step == 64:
                        for track, key in (
                            ("N", "dev_prompts"),
                            ("L", "legacy_prompts"),
                            ("OOD", "ood_prompts"),
                        ):
                            endpoint.extend(
                                _samples(
                                    adapter,
                                    optimizer,
                                    state,
                                    plan[key],
                                    out / "evaluation" / arm / f"step_64/{track}",
                                    identity,
                                    arm,
                                    step,
                                    "evaluation",
                                    data_root,
                                    config,
                                    legacy_lock,
                                    meter,
                                    progress,
                                )
                            )
            if (
                len(summaries) != 128
                or len(shared) != 576
                or len(panel32) != 1152
                or len(endpoint) != 8576
            ):
                raise ValueError("R4 complete protocol coverage differs from fixed budget")
            if len({key for s in summaries for key in s["sample_keys"]}) != 4096:
                raise ValueError("R4 arms reused trajectories or incomplete train coverage")
            details.update(
                {
                    "distinct_optimizer_updates": 128,
                    "training_rollouts": 4096,
                    "shared_step0_outputs": len(shared),
                    "step32_outputs": len(panel32),
                    "step64_outputs": len(endpoint),
                    "new_outputs": 14400,
                    "fixed_control_sequence_scores": 6144,
                    "preupdate_parity_sequence_scores": 4096,
                    "postupdate_training_sequence_scores": 4096,
                    "new_control_outputs": 0,
                    "initial_alignment": alignment,
                    "arms": {arm: 64 for arm in ARMS},
                }
            )
            from .r4_metrics import analyze_sampled_endpoints

            write_json(
                out / "step32_metrics.json", analyze_sampled_endpoints(panel32, shared, track="N")
            )
            status = "PASS"
            details["status"] = status
        except PilotBlocked as exc:
            status = "BLOCKED"
            details["error"] = {"type": type(exc).__name__, "message": str(exc)}
        except Exception as exc:
            status = "FAIL"
            details["error"] = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            if adapter is not None and optimizer is not None and origin is not None:
                try:
                    _restore(adapter, optimizer, origin)
                    if parameter_hash(adapter.model, trainable=False) != frozen_hash:
                        write_json(
                            out / "policy_contamination.json",
                            {"reason": "final frozen hash changed"},
                        )
                        raise RuntimeError("R4 final frozen-base audit failed")
                    details["final_scratch_origin_restored"] = True
                except Exception as exc:
                    status = "FAIL"
                    details["restoration_error"] = str(exc)
            if progress is not None:
                progress.write(status)
            if meter is not None:
                meter.close()
            details["status"] = status
            write_json(invocation / "completion.json", {"status": status, "details": details})
        histories = [
            {
                "invocation": p.name,
                "completed": (p / "completion.json").exists(),
                "observed": _json(p / "runtime_profile.json")
                if (p / "runtime_profile.json").exists()
                else {},
            }
            for p in sorted(invocations.iterdir())
        ]
        cost = {
            "invocations": histories,
            "optimizer_steps_observed_at_least": sum(
                p["observed"].get("optimizer_step_calls_observed", 0) for p in histories
            ),
            "backward_calls_observed_at_least": sum(
                p["observed"].get("backward_calls_observed", 0) for p in histories
            ),
            "incomplete_invocations_with_unknown_extra_cost": sum(
                not p["completed"] for p in histories
            ),
            "counter_scope": (
                "Actual instrumented calls includ"
                "ing retries; abrupt unrecorded i"
                "ntervals are unknown extra cost"
            ),
        }
        write_json(out / "runtime_profile.json", cost)
        details["runtime_counts"] = {k: v for k, v in cost.items() if k != "invocations"}
        if status == "PASS":
            try:
                _reports(out, endpoint, initial_rows, details)
            except Exception as exc:
                status = "FAIL"
                details.update({"status": status, "report_error": str(exc)})
        if status != "PASS":
            (out / "pilot_report.md").write_text(
                "# R4 尚未完成\n\n" + json.dumps(details, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return _finish(out, status, execution, details)


def main(argv=None):
    from .next_stage_common import load_yaml

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/next_stage.yaml"))
    for name in ("data-root", "out", "r0-dir", "r1-run", "supplement-dir", "r2-dir", "r3-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--allow-training", action="store_true")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)
    result = run_r4(
        load_yaml(a.config),
        a.data_root,
        a.out,
        a.r0_dir,
        a.r1_run,
        a.supplement_dir,
        a.r2_dir,
        a.r3_dir,
        allow_training=a.allow_training,
        resume=a.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
