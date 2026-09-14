"""S2 three-arm on-policy training with immutable steps and isolated evaluation.

This module never constructs/downloads a model. The explicitly gated backend supplies
an initialized adapter, optimizer, verified data and historical input bindings.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

from .core import canonical_hash, file_hash, frozen_writer, write_json
from .optimizer_fork import parameter_hash, state_hash
from .r3_runtime import _optimizer_step, _verify_manifest, run_atomic_unit

ARMS = {"X_BASE": ("joint", 0.0), "X_VALID": ("joint", 1.0), "X_VALID_NO_X_OFF": ("no_x_off", 1.0)}


def evaluation_plan(plan, step):
    """Fixed panels; endpoint dev-panel rows alias the complete N ledger."""
    spec = plan["design"]["S2"]
    final = spec["steps"]
    result = []

    def add(track, key, count, **extra):
        result.append({"track": track, "prompts": plan[key], "samples_per_prompt": count, **extra})

    if step == final:
        add("N", "dev_prompts", spec["final_eval"]["samples_per_prompt"])
    if step in spec["dev_panel"]["steps"]:
        add(
            "N_PANEL",
            "dev_panel_prompts",
            spec["dev_panel"]["samples_per_prompt"],
            **({"alias_of": "N"} if step == final else {}),
        )
    if step == final:
        add("L", "legacy_prompts", spec["final_eval"]["samples_per_prompt"])
    if step == final or (step == 0 and spec["OOD_initial_eval"]):
        add("OOD", "ood_prompts", spec["final_eval"]["samples_per_prompt"])
    if step in spec["interface_diagnostic"]["steps"]:
        add("R2", "diagnostic_prompts", spec["interface_diagnostic"]["samples_per_condition"])
    return result


def build_training_requests(
    prompts, identity, seed, arm, step, role, policy_hash, *, samples_per_prompt=8
):
    """Stable semantic random stream; each arm still owns distinct sample keys."""
    if type(seed) is not int or seed < 0 or arm not in ARMS:
        raise ValueError("Invalid training seed or arm")
    if type(samples_per_prompt) is not int or samples_per_prompt < 1:
        raise ValueError("Positive integer samples_per_prompt required")
    if len({p["prompt_id"] for p in prompts}) != len(prompts):
        raise ValueError("Duplicate prompt in request panel")
    rows = []
    for prompt in prompts:
        for index in range(samples_per_prompt):
            rng = {
                "namespace": "SSVC_MECHANISM_S2",
                "identity": identity,
                "seed": seed,
                "step": step,
                "role": role,
                "policy_hash": policy_hash,
                "prompt_id": prompt["prompt_id"],
                "prompt_hash": prompt["prompt_hash"],
                "sample_index": index,
            }
            digest = canonical_hash(rng)
            request = {
                key: prompt[key]
                for key in (
                    "prompt_id",
                    "prompt_hash",
                    "base_scene_id",
                    "family",
                    "interface",
                    "split",
                )
            }
            request.update(
                {
                    "phase": "S2",
                    "arm": arm,
                    "train_seed": seed,
                    "seed": seed,
                    "checkpoint_step": step,
                    "bank_role": role,
                    "group_id": prompt["prompt_id"],
                    "constraint_family": prompt["family"],
                    "sample_index": index,
                    "rollout_index": index,
                    "track": prompt.get("track", "N"),
                    "protocol_track": prompt.get("track", "N"),
                    "evaluation_domain": prompt.get("evaluation_domain", "ID"),
                    "decode_mode": "sample",
                    "do_sample": True,
                    "enable_thinking": False,
                    "max_new_tokens": prompt["max_new_tokens"],
                    "policy_state_hash": policy_hash,
                    "sample_rng_key": digest,
                    "sample_seed": int(digest[:8], 16) % 2**31,
                    "run_id": canonical_hash(identity),
                }
            )
            if "diagnostic_condition" in prompt:
                request["condition"] = prompt["diagnostic_condition"]
            request["sample_key"] = canonical_hash({"request": request, "rng": rng})
            rows.append(request)
    return rows


@contextmanager
def isolated_evaluation(adapter, optimizer, *, sampler=None, metadata=None):
    """Restore every forward/training state component, including on exceptions."""
    from .followup_updates import capture_state

    origin = capture_state(adapter.model, optimizer, metadata, adapter=adapter, sampler=sampler)
    frozen = parameter_hash(adapter.model, trainable=False)
    try:
        yield origin
    finally:
        if _restore_finally(adapter, optimizer, origin, sampler=sampler):
            restored = capture_state(
                adapter.model, optimizer, origin["metadata"], adapter=adapter, sampler=sampler
            )
            if state_hash(restored) != state_hash(origin):
                raise RuntimeError("Evaluation altered training RNG/Adam/sampler/modes/buffers")
            if parameter_hash(adapter.model, trainable=False) != frozen:
                raise RuntimeError("Evaluation changed frozen parameters")


def _restore_finally(adapter, optimizer, state, *, sampler=None, failure_path=None):
    """Keep the primary failure when restoration itself also fails."""
    from .followup_updates import restore_state

    primary = sys.exc_info()[1]
    try:
        restore_state(adapter.model, optimizer, state, adapter=adapter, sampler=sampler)
        return True
    except BaseException as restore_error:
        adapter._followup_unusable = adapter._followup_state_unusable = True
        if failure_path is not None:
            write_json(
                failure_path,
                {
                    "type": type(restore_error).__name__,
                    "message": str(restore_error),
                    "primary_failure": str(primary),
                },
            )
        if primary is None:
            raise
        primary.add_note(f"Complete state restoration also failed: {restore_error}")
        return False


def _policy_hash(state):
    # Optimizer and training metadata do not affect inference or common-random seeds.
    return state_hash(
        {
            key: value
            for key, value in state.items()
            if key not in ("metadata", "rng", "optimizer", "sampler", "scheduler")
        }
    )


def _read(path):
    return json.loads(Path(path).read_text())


def _publish(path, value):
    """Idempotent atomic JSON publication without replacing different evidence."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Output symlink rejected")
    if path.exists():
        if canonical_hash(_read(path)) != canonical_hash(value):
            raise ValueError(f"Existing evidence differs: {path.name}")
    else:
        write_json(path, value)


def _safe_output(path):
    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError("Output directory escape/symlink rejected")
    if path.exists() and any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("Output evidence contains a symlink")
    return path


def _completion_manifest(out):
    return {
        "files": [
            {"path": str(p.relative_to(out)), "sha256": file_hash(p), "bytes": p.stat().st_size}
            for p in sorted(out.rglob("*"))
            if p.is_file()
            and p.relative_to(out).parts[0] != "invocations"
            and p.name != ".writer.lock"
            and str(p.relative_to(out))
            not in ("completed.json", "completion_manifest.json", "completion_binding.json")
            and not p.name.endswith(".tmp")
        ]
    }


def _counts(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["category"] not in "XSWI" or len(row["category"]) != 1:
            raise ValueError("Unknown evaluation category")
        groups[(row["family"], row["interface"])].append(row)

    def summarize(values):
        by_prompt = defaultdict(list)
        for row in values:
            by_prompt[row["prompt_id"]].append(row)
        count = {c: sum(row["category"] == c for row in values) for c in "XSWI"}
        means = {
            c: sum(sum(r["category"] == c for r in v) / len(v) for v in by_prompt.values())
            / len(by_prompt)
            for c in "XSWI"
        }
        valid = 1.0 - means["I"]
        return {
            "n": len(values),
            "prompt_count": len(by_prompt),
            "counts": count,
            "pX": means["X"],
            "v": valid,
            "qX": means["X"] / valid if valid else None,
            "qX_support": "DEFINED" if valid else "UNDEFINED_ZERO_VALID",
            "qS": means["S"] / valid if valid else None,
            "qS_support": "DEFINED" if valid else "UNDEFINED_ZERO_VALID",
            "weights": "equal_prompt",
        }

    result = {
        "overall": summarize(rows),
        "groups": {"/".join(key): summarize(value) for key, value in sorted(groups.items())},
    }
    diagnostic_conditions = ("SYM_ORIGINAL", "IMAGE_CUE", "IMAGE_ONLY")
    if any(
        row.get("track") == "R2"
        or "diagnostic_condition" in row
        or row.get("condition") in diagnostic_conditions
        for row in rows
    ):
        by_condition, by_family_condition = defaultdict(list), defaultdict(list)
        for row in rows:
            condition = row.get("diagnostic_condition", row.get("condition"))
            if condition not in diagnostic_conditions or (
                "condition" in row and row["condition"] != condition
            ):
                raise ValueError("R2 rows require consistent diagnostic condition identities")
            by_condition[condition].append(row)
            by_family_condition[(row["family"], condition)].append(row)
        result["conditions"] = {
            condition: summarize(values) for condition, values in sorted(by_condition.items())
        }
        result["condition_groups"] = {
            "/".join(key): summarize(values) for key, values in sorted(by_family_condition.items())
        }
    return result


def _evaluate(plan, adapter, optimizer, state, out, identity, seed, arm, step, data_root):
    from .followup_runtime import collect_samples

    outputs, reports = {}, []
    with isolated_evaluation(adapter, optimizer, metadata=state["metadata"]):
        for panel in evaluation_plan(plan, step):
            track = panel["track"]
            root = out / "evaluation" / f"step_{step:02d}" / track
            if panel.get("alias_of"):
                source = outputs[panel["alias_of"]]
                wanted = {p["prompt_id"] for p in panel["prompts"]}
                rows = [row for row in source if row["prompt_id"] in wanted]
                if len(rows) != len(wanted) * panel["samples_per_prompt"]:
                    raise ValueError("Endpoint N panel does not match complete N evaluation")
                _publish(
                    root / "alias.json",
                    {
                        "source_track": panel["alias_of"],
                        "sample_keys": [r["sample_key"] for r in rows],
                        "source_record_hashes": [r["record_hash"] for r in rows],
                        "additional_samples": 0,
                        "independent_samples": False,
                    },
                )
            else:
                requests = build_training_requests(
                    panel["prompts"],
                    identity,
                    seed,
                    arm,
                    step,
                    f"evaluation/{track}",
                    _policy_hash(state),
                    samples_per_prompt=panel["samples_per_prompt"],
                )
                rows = collect_samples(
                    adapter, optimizer, state, panel["prompts"], requests, root, identity, data_root
                )
            outputs[track] = rows
            report = {
                "track": track,
                "step": step,
                "seed": seed,
                "arm": arm,
                "alias_of": panel.get("alias_of"),
                **_counts(rows),
            }
            _publish(root / "counts.json", report)
            reports.append(report)
    return reports


def _score_rows(adapter, optimizer, state, rows, prompts, data_root):
    """Teacher forcing uses every genuine generated token, including EOS."""
    import torch

    scores, prepared_inputs = {}, {}
    with isolated_evaluation(adapter, optimizer, metadata=state["metadata"]):
        for row in rows:
            prompt = prompts[row["prompt_id"]]
            if row["prompt_id"] not in prepared_inputs:
                prepared_inputs[row["prompt_id"]] = adapter.prepare(
                    prompt["prompt"], prompt.get("data_root") or data_root
                )
            prepared = prepared_inputs[row["prompt_id"]]
            if row.get("prepared_hash") and row["prepared_hash"] != state_hash(prepared):
                raise ValueError("Prepared input binding changed during teacher forcing")
            with torch.no_grad():
                value = adapter.logprobs(prepared, row["token_ids"]).detach().cpu().double()
            if (
                value.ndim != 1
                or len(value) != len(row["token_ids"])
                or not torch.isfinite(value).all()
            ):
                raise FloatingPointError("Nonfinite or malformed teacher forcing likelihood")
            scores[row["sample_key"]] = value.tolist()
    return scores


def _train_step(
    plan, adapter, optimizer, prestate, rows, prompts, root, identity, arm, seed, step, data_root
):
    import numpy as np

    from .followup_updates import (
        apply_gradient_update,
        capture_state,
        direct_loss_gradients,
        load_checkpoint,
        restore_state,
        save_checkpoint,
    )
    from .r4_continuation import allows_reviewed_warning
    from .r4_metrics import control_kl_diagnostic

    unit = {
        **identity,
        "unit": "S2_training_step",
        "arm": arm,
        "seed": seed,
        "step": step,
        "prestate_hash": state_hash(prestate),
        "sample_hash": canonical_hash([r["record_hash"] for r in rows]),
    }
    frozen = parameter_hash(adapter.model, trainable=False)
    policy, weight = ARMS[arm]

    def operation(attempt):
        try:
            restore_state(adapter.model, optimizer, prestate, adapter=adapter)
            parity = _score_rows(adapter, optimizer, prestate, rows, prompts, data_root)
            errors = [
                abs(a - b)
                for row in rows
                for a, b in zip(parity[row["sample_key"]], row["old_logprobs"], strict=True)
            ]
            parity_report = {
                "mean_abs_token_error": float(np.mean(errors)),
                "p99_abs_token_error": float(np.quantile(errors, 0.99)),
            }
            write_json(attempt / "parity.json", parity_report)
            if (
                parity_report["mean_abs_token_error"] > 0.02
                or parity_report["p99_abs_token_error"] > 0.1
            ):
                raise ValueError("On-policy behavior/teacher forcing parity failure")
            groups = []
            for prompt_id in dict.fromkeys(r["prompt_id"] for r in rows):
                selected = [r for r in rows if r["prompt_id"] == prompt_id]
                if len(selected) != plan["design"]["optimization"]["K"] or any(
                    r["arm"] != arm
                    or r["train_seed"] != seed
                    or r["bank_role"] != "train"
                    or r["split"] != "train"
                    for r in selected
                ):
                    raise ValueError("Training group mixes arm/seed/evaluation or misses samples")
                prompt = prompts[prompt_id]
                prepared = adapter.prepare(prompt["prompt"], prompt.get("data_root") or data_root)
                groups.append([{**r, "prepared": prepared} for r in selected])
            if len(groups) != plan["design"]["optimization"]["B"]:
                raise ValueError("Training group B mismatch")
            gradient_result = direct_loss_gradients(
                adapter,
                groups,
                policy=policy,
                auxiliary_weight=weight,
                lnorm=64,
                epsilon=plan["design"]["reward"]["epsilon"],
                clip_epsilon=0.2,
            )
            loss = gradient_result["audit"]
            stats = loss["group_statistics"]
            update = apply_gradient_update(
                adapter.model, optimizer, gradient_result["gradients"], grad_clip=1.0
            )
            if update["optimizer_updates"] != 1 or loss["backward_calls"] != len(rows):
                raise RuntimeError(
                    "Training must perform one Adam step and one backward per rollout"
                )
            if parameter_hash(adapter.model, trainable=False) != frozen:
                raise RuntimeError("Frozen model parameter changed during update")
            metadata = {
                **prestate["metadata"],
                "checkpoint_step": step,
                "sampler": {"schedule_hash": identity["schedule_hash"], "position": step},
                "completed_sample_keys": [
                    *prestate["metadata"].get("completed_sample_keys", []),
                    *[r["sample_key"] for r in rows],
                ],
            }
            post = capture_state(adapter.model, optimizer, metadata, adapter=adapter)
            if _optimizer_step(prestate) != step - 1 or _optimizer_step(post) != step:
                raise ValueError("Adam step counter does not match committed training cursor")
            checkpoint_identity = {**unit, "unit": "S2_checkpoint"}
            save_checkpoint(attempt / "checkpoint.pt", post, checkpoint_identity)
            probe = plan.get("control_probe_rows", [])
            if probe:
                scores = _score_rows(
                    adapter,
                    optimizer,
                    post,
                    probe,
                    {p["prompt_id"]: p for p in plan["control_prompts"]},
                    data_root,
                )
                diagnostic = control_kl_diagnostic(probe, scores, eos_token_ids=adapter.eos_ids)
                write_json(attempt / "control_scores.json", scores)
            elif plan.get("cpu_fixture") is True:
                diagnostic = {"status": "CPU_FIXTURE_NO_PARENT_CONTROL", "should_stop": False}
            else:
                raise ValueError("Verified fixed control probe required for training")
            warning = diagnostic["should_stop"] and allows_reviewed_warning(
                diagnostic, plan["warning_policy"]
            )
            if diagnostic["should_stop"] and not warning:
                write_json(
                    root.parents[1] / "hard_stop.json", {"step": step, "diagnostic": diagnostic}
                )
                raise RuntimeError("Unreviewed fixed-control diagnostic alarm")
            result = {
                "step": step,
                "seed": seed,
                "arm": arm,
                "checkpoint_path": "checkpoint.pt",
                "checkpoint_identity": checkpoint_identity,
                "checkpoint_sha256": file_hash(attempt / "checkpoint.pt"),
                "state_hash": state_hash(post),
                "parameter_hash": state_hash(post["parameters"]),
                "optimizer_state_hash": state_hash(post["optimizer"]),
                "status": "DIAGNOSTIC_WARNING" if warning else "PASS",
                "group_statistics": stats,
                "loss": loss,
                "update": update,
                "control_diagnostic": diagnostic,
                "training_category_counts": dict(Counter(r["category"] for r in rows)),
                "sample_keys": [r["sample_key"] for r in rows],
            }
            write_json(attempt / "group_statistics.json", stats)
            write_json(attempt / "update.json", {"loss": loss, "update": update})
            return result
        finally:
            _restore_finally(
                adapter, optimizer, prestate, failure_path=attempt / "restoration_failure.json"
            )

    attempt, result, reused = run_atomic_unit(root, unit, operation)
    post = load_checkpoint(
        attempt / result["checkpoint_path"],
        result["checkpoint_identity"],
        model=adapter.model,
        optimizer=optimizer,
        expected_file_sha256=result["checkpoint_sha256"],
    )
    if state_hash(post) != result["state_hash"]:
        raise ValueError("S2 atomic checkpoint state hash mismatch")
    restore_state(adapter.model, optimizer, post, adapter=adapter)
    return post, {
        **result,
        "attempt": str(attempt.relative_to(root.parents[1])),
        "reused_completed_unit": reused,
    }


def _validate(plan, seed, arm, adapter, optimizer):
    if arm not in ARMS or type(seed) is not int or seed not in plan["design"]["S2"]["seeds"]:
        raise ValueError("Unapproved seed or arm")
    fixture = plan.get("cpu_fixture") is True and plan.get("execution_kind") == "CPU_FAKE_TORCH"
    if seed == 17 and arm in ("X_BASE", "X_VALID") and not fixture:
        raise ValueError("Existing historical seed17 anchors cannot be silently retrained")
    if not fixture:
        required = (
            "training_authorized",
            "gpu_smoke_passed",
            "s1_measurement_passed",
            "parent_raw_verified",
        )
        if not all(plan.get("gates", {}).get(k) is True for k in required):
            raise ValueError("S2 authorization and measurement gates are not satisfied")
    if adapter is None or optimizer is None:
        raise ValueError("Initialized adapter and owned optimizer required")
    if getattr(adapter, "_followup_unusable", False) or getattr(
        adapter, "_followup_state_unusable", False
    ):
        raise ValueError("Adapter is unusable after complete-state restoration failure")
    if fixture and (
        getattr(adapter, "execution_kind", None) != "CPU_FAKE_TORCH"
        or any(p.device.type != "cpu" for p in adapter.model.parameters())
    ):
        raise ValueError("CPU fixture requires an explicitly fake adapter with only CPU parameters")
    if len(plan["train_steps"]) != plan["design"]["S2"]["steps"]:
        raise ValueError("Schedule does not cover all training steps")
    prompts = {p["prompt_id"]: p for p in plan["train_prompts"]}
    if len(prompts) != len(plan["train_prompts"]):
        raise ValueError("Duplicate training prompt")
    used = [p for batch in plan["train_steps"] for p in batch]
    if any(len(batch) != plan["design"]["optimization"]["B"] for batch in plan["train_steps"]):
        raise ValueError("Training B mismatch")
    if any(p not in prompts for p in used) or len(used) != len(set(used)):
        raise ValueError("Missing or repeated scheduled prompt")
    if plan.get("sampler_seed", seed) != seed:
        raise ValueError("Sampler seed mismatch")
    train_scenes = {p["base_scene_id"] for p in plan["train_prompts"]}
    for key in (
        "dev_panel_prompts",
        "dev_prompts",
        "legacy_prompts",
        "ood_prompts",
        "diagnostic_prompts",
        "control_prompts",
    ):
        if train_scenes & {p["base_scene_id"] for p in plan.get(key, [])}:
            raise ValueError("Train/evaluation/control scene leakage")
    if not fixture:
        spec = plan["design"]["S2"]
        for key, n in [
            ("dev_panel_prompts", 72),
            ("dev_prompts", 288),
            ("legacy_prompts", 176),
            ("ood_prompts", 72),
            ("diagnostic_prompts", 216),
        ]:
            if len(plan.get(key, [])) != n:
                raise ValueError(f"Prescribed evaluation panel size mismatch: {key}")
        if (
            spec["steps"] != 64
            or plan["design"]["optimization"]["B"] != 4
            or plan["design"]["optimization"]["K"] != 8
        ):
            raise ValueError("Production training must preserve 64 steps B4 K8")
        if not plan.get("control_probe_rows") or not plan.get("warning_policy"):
            raise ValueError("Frozen control and reviewed warning policy required")


def run_training_arm(
    plan,
    *,
    seed,
    arm,
    output_root,
    resume=False,
    adapter=None,
    optimizer=None,
    data_root=None,
    initial_state=None,
):
    """Execute one approved S2 run. CPU fixtures require an explicit bounded plan."""
    _validate(plan, seed, arm, adapter, optimizer)
    from .followup_runtime import collect_samples
    from .followup_updates import capture_state, load_checkpoint, restore_state, save_checkpoint

    out = _safe_output(output_root)
    if initial_state is not None:
        restore_state(adapter.model, optimizer, initial_state, adapter=adapter)
    supplied_origin = capture_state(adapter.model, optimizer, adapter=adapter)
    supplied_origin_hash = state_hash({k: v for k, v in supplied_origin.items() if k != "metadata"})
    identity = {
        **plan["identity"],
        "phase": "S2",
        "seed": seed,
        "arm": arm,
        "execution_kind": plan["execution_kind"],
        "design_hash": canonical_hash(plan["design"]),
        "schedule_hash": canonical_hash(plan["train_steps"]),
        "prompt_binding_hash": canonical_hash(
            [{k: p[k] for k in ("prompt_id", "prompt_hash")} for p in plan["train_prompts"]]
        ),
        "warning_policy_hash": canonical_hash(plan.get("warning_policy")),
    }
    identity["initial_state_hash"] = supplied_origin_hash
    # Randomness omits arm so identical initial policies share random numbers;
    # request/sample identities explicitly retain arm ownership.
    random_identity = {k: v for k, v in identity.items() if k != "arm"}
    with frozen_writer(out):
        has_identity = (out / "identity.json").exists()
        if resume and not has_identity:
            raise ValueError("Cannot resume without a saved run identity")
        if not has_identity and any(p.name != ".writer.lock" for p in out.iterdir()):
            raise ValueError("Refusing to overwrite unbound output evidence")
        if has_identity and not resume:
            raise FileExistsError("Existing training run requires --resume")
        _publish(out / "identity.json", identity)
        _publish(out / "execution_identity.json", plan.get("execution_identity", {}))
        if any(out.rglob("restoration_failure.json")):
            raise ValueError("Preserved restoration failure prevents training continuation")
        if (out / "completed.json").exists():
            manifest = _read(out / "completion_manifest.json")
            binding = _read(out / "completion_binding.json")
            if binding["manifest_sha256"] != file_hash(out / "completion_manifest.json"):
                raise ValueError("Completed run manifest binding mismatch")
            if binding["result_hash"] != canonical_hash(_read(out / "completed.json")):
                raise ValueError("Completed run result binding mismatch")
            _verify_manifest(out, manifest)
            if manifest != _completion_manifest(out):
                raise ValueError("Completed run manifest coverage mismatch")
            return _read(out / "completed.json")
        if (out / "hard_stop.json").exists():
            raise ValueError("Preserved hard stop prevents training continuation")
        invocations = out / "invocations"
        invocations.mkdir(exist_ok=True)
        invocation = invocations / f"invocation_{len(list(invocations.iterdir())):04d}"
        invocation.mkdir(exist_ok=False)
        write_json(invocation / "execution_identity.json", plan.get("execution_identity", {}))
        origin_identity = {**identity, "unit": "S2_initial_state"}
        frozen = parameter_hash(adapter.model, trainable=False)

        def save_initial(attempt):
            metadata = {
                "phase": "S2",
                "arm": arm,
                "seed": seed,
                "checkpoint_step": 0,
                "sampler": {"schedule_hash": identity["schedule_hash"], "position": 0},
                "completed_sample_keys": [],
            }
            origin = capture_state(adapter.model, optimizer, metadata, adapter=adapter)
            if _optimizer_step(origin) != 0:
                raise ValueError("S2 requires the cold initial Adam state, not a warm candidate")
            binding = save_checkpoint(attempt / "checkpoint.pt", origin, origin_identity)
            return {
                "checkpoint_path": "checkpoint.pt",
                "checkpoint_sha256": binding["sha256"],
                "state_hash": state_hash(origin),
            }

        initial_attempt, initial_record, _ = run_atomic_unit(
            out / "initial", origin_identity, save_initial
        )
        initial_path = initial_attempt / initial_record["checkpoint_path"]
        origin = load_checkpoint(
            initial_path,
            origin_identity,
            model=adapter.model,
            optimizer=optimizer,
            expected_file_sha256=initial_record["checkpoint_sha256"],
        )
        restore_state(adapter.model, optimizer, origin, adapter=adapter)
        state = origin
        summaries, evaluations = [], []
        origin_restored = False
        try:
            evaluations.extend(
                _evaluate(
                    plan, adapter, optimizer, state, out, random_identity, seed, arm, 0, data_root
                )
            )
            prompt_map = {p["prompt_id"]: p for p in plan["train_prompts"]}
            for step, prompt_ids in enumerate(plan["train_steps"], 1):
                prompts = [prompt_map[p] for p in prompt_ids]
                requests = build_training_requests(
                    prompts,
                    random_identity,
                    seed,
                    arm,
                    step - 1,
                    "train",
                    _policy_hash(state),
                    samples_per_prompt=plan["design"]["optimization"]["K"],
                )
                rows = collect_samples(
                    adapter,
                    optimizer,
                    state,
                    prompts,
                    requests,
                    out / f"step_{step:02d}" / "rollouts",
                    random_identity,
                    data_root,
                )
                state, summary = _train_step(
                    plan,
                    adapter,
                    optimizer,
                    state,
                    rows,
                    prompt_map,
                    out / f"step_{step:02d}" / "updates",
                    identity,
                    arm,
                    seed,
                    step,
                    data_root,
                )
                summaries.append(summary)
                write_json(
                    out / "progress.json",
                    {
                        "completed_step": step,
                        "seed": seed,
                        "arm": arm,
                        "state_hash": state_hash(state),
                    },
                )
                evaluations.extend(
                    _evaluate(
                        plan,
                        adapter,
                        optimizer,
                        state,
                        out,
                        random_identity,
                        seed,
                        arm,
                        step,
                        data_root,
                    )
                )
            if parameter_hash(adapter.model, trainable=False) != frozen:
                raise RuntimeError("Frozen parameter identity changed")
            checkpoints = [
                {
                    "step": 0,
                    "checkpoint_path": str(initial_path.relative_to(out)),
                    "checkpoint_sha256": initial_record["checkpoint_sha256"],
                    "state_hash": state_hash(origin),
                    "milestone": True,
                }
            ]
            checkpoints.extend(
                {
                    "step": s["step"],
                    "checkpoint_path": str(Path(s["attempt"]) / s["checkpoint_path"]),
                    "state_hash": s["state_hash"],
                    "optimizer_state_hash": s["optimizer_state_hash"],
                    "checkpoint_sha256": s["checkpoint_sha256"],
                    "milestone": s["step"] in plan["design"]["S2"]["checkpoint_steps"],
                }
                for s in summaries
            )
            _publish(
                out / "checkpoint_manifest.json", {"identity": identity, "checkpoints": checkpoints}
            )
            normalized = [
                {k: v for k, v in s.items() if k != "reused_completed_unit"} for s in summaries
            ]
            _publish(out / "training_steps.json", normalized)
            _publish(out / "evaluation_summary.json", evaluations)
            result = {
                "status": "COMPLETED_WITH_DIAGNOSTIC_WARNINGS"
                if any(s["status"] == "DIAGNOSTIC_WARNING" for s in summaries)
                else "PASS",
                "execution_kind": plan["execution_kind"],
                "seed": seed,
                "arm": arm,
                "seed_role": "exploratory" if seed == 17 else "prospective",
                "completed_steps": len(summaries),
                "training_outputs": sum(len(s["sample_keys"]) for s in summaries),
                "evaluation_outputs": sum(
                    e["overall"]["n"] for e in evaluations if not e["alias_of"]
                ),
                "final_state_hash": state_hash(state),
                "final_parameter_hash": state_hash(state["parameters"]),
                "final_optimizer_state_hash": state_hash(state["optimizer"]),
                "gpu_started": plan["execution_kind"] != "CPU_FAKE_TORCH",
                "training_started": plan["execution_kind"] != "CPU_FAKE_TORCH",
                "cpu_optimizer_updates": len(summaries)
                if plan["execution_kind"] == "CPU_FAKE_TORCH"
                else 0,
                "safety_status": "NOT_CERTIFIED",
            }
            _restore_finally(
                adapter,
                optimizer,
                origin,
                failure_path=invocation / "restoration_failure.json",
            )
            origin_restored = True
            _publish(out / "completion_manifest.json", _completion_manifest(out))
            _publish(
                out / "completion_binding.json",
                {
                    "manifest_sha256": file_hash(out / "completion_manifest.json"),
                    "result_hash": canonical_hash(result),
                },
            )
            _publish(out / "completed.json", result)
            write_json(
                invocation / "completed.json",
                {
                    "status": result["status"],
                    "reused_steps": sum(s["reused_completed_unit"] for s in summaries),
                },
            )
            return result
        except BaseException as exc:
            write_json(
                invocation / "failure.json", {"type": type(exc).__name__, "message": str(exc)}
            )
            if isinstance(exc, (FloatingPointError, OverflowError)) or any(
                word in str(exc).lower()
                for word in ("nonfinite", "non-finite", "frozen", "parity", "binding")
            ):
                _publish(out / "hard_stop.json", {"type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            if not origin_restored:
                _restore_finally(
                    adapter, optimizer, origin, failure_path=invocation / "restoration_failure.json"
                )
