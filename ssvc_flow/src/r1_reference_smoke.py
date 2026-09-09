"""Isolated R1 updates on a separately certified uncached reference policy.

This does not relax legacy P1 cache gates. Failed optimization paths remain
failed; this runner never calls a cache or I4 implementation.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from .core import RunStore, canonical_hash, source_commit, write_json
from .grpo_update import grouped_advantages, reward_channels, torch_ppo_loss
from .optimizer_fork import (
    capture_state,
    compare_parameters,
    parameter_hash,
    restore_state,
    save_checkpoint,
    state_hash,
)
from .prompts import build_prompt
from .smoke_runtime import (
    INTERFACES,
    Telemetry,
    _check_kl_alarms,
    _collect_group,
    _image_audit,
    _real_update,
    _replay_audit,
    _verify_scene_images,
)

PATH = "uncached_prefix_recompute"
CERTIFICATE_CHECKS = (
    "reference_repeat",
    "behavior",
    "evaluation_likelihood",
    "training_likelihood",
)


class _R1Store(RunStore):
    def append(self, row):
        return super().append({**row, "phase": "R1", "selected_probability_path": PATH})


class _PhaseTelemetry(Telemetry):
    """Retain peaks across helper resets, including post-generation likelihood."""

    def __init__(self):
        self.intervals = []
        self.active = False

    def begin_phase(self):
        self.intervals = []
        self.active = True
        super().reset()

    def reset(self):
        if self.active:
            self.intervals.append(super().snapshot())
        super().reset()

    def phase_snapshot(self):
        readings = [*self.intervals, super().snapshot()]
        return {
            key: max((row[key] for row in readings if row[key] is not None), default=None)
            for key in readings[0]
        }


def _certificate_check(certificate, adapter, config):
    if not isinstance(certificate, dict) or certificate.get("status") != "PASS":
        raise ValueError("A PASS reference-path certificate is required before smoke")
    if certificate.get("selected_path") != PATH:
        raise ValueError("Smoke only accepts the certified uncached prefix path")
    boundaries = certificate.get("fixed_sequence_boundaries", {})
    if (
        boundaries.get("passed") is not True
        or boundaries.get("checks_completed") != 27
        or boundaries.get("checks_expected") != 27
        or boundaries.get("not_model_generated_not_training") is not True
    ):
        raise ValueError("Reference-path fixed-sequence boundary certificate is incomplete")
    for name in CERTIFICATE_CHECKS:
        check = certificate.get("checks", {}).get(name)
        if (
            not isinstance(check, dict)
            or check.get("passed") is not True
            or check.get("sequences") != 120
            or check.get("failed_sequence_checks") != 0
            or (name in ("reference_repeat", "behavior") and check.get("top1_measured") is not True)
        ):
            raise ValueError(f"Reference-path certificate check is missing or failed: {name}")
    if certificate.get("model_revision") != adapter.revision:
        raise ValueError("Certificate model revision differs from the smoke adapter")
    if certificate.get("initial_adapter_hash") != parameter_hash(adapter.model, trainable=True):
        raise ValueError("Certificate initial adapter hash differs from the smoke adapter")
    measured = certificate.get("model_audit", {})
    for key in ("processor_hash", "tokenizer_hash", "chat_template_hash", "frozen_parameter_hash"):
        if not measured.get(key) or measured[key] != adapter.audit.get(key):
            raise ValueError(f"Reference-path certificate model audit drift: {key}")
    if any(audit.get("probability_execution") != PATH for audit in (measured, adapter.audit)):
        raise ValueError("Reference-path certificate probability execution differs")
    for key in (
        "parity_alarm_mean_abs_token_logp",
        "parity_alarm_p99_abs_token_logp",
        "parity_alarm_min_top1_agreement",
    ):
        if certificate.get("thresholds", {}).get(key) != config["R1"][key]:
            raise ValueError(f"Reference-path certificate threshold drift: {key}")
    frozen_hash = parameter_hash(adapter.model, trainable=False)
    if frozen_hash != measured["frozen_parameter_hash"]:
        raise ValueError("Certificate frozen parameters differ from the actual smoke model")
    return frozen_hash


def _helper_config(config):
    """Translate the new lock explicitly, without importing the legacy P1 lock."""
    if config["R1"]["smoke_optimizer_steps"] != 4:
        raise ValueError("R1 reference smoke requires four optimizer steps")
    if config["R1"]["max_smoke_training_rollouts"] < 128:
        raise ValueError("Four B4 K8 updates require a 128-rollout budget")
    if config["generation_proposed_N"]["max_new_tokens"] != 64:
        raise ValueError("R1 reference smoke requires the locked 64-token horizon")
    if config["reward"]["epsilon"] != 1e-4:
        raise ValueError("The reused update helper requires reward epsilon 1e-4")
    expected_generation = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "do_sample": True,
    }
    if any(config["generation_proposed_N"].get(k) != v for k, v in expected_generation.items()):
        raise ValueError("R1 smoke requires the locked pure-softmax sampling distribution")
    if config["loss"]["Lnorm"] != 64 or config["loss"]["beta_KL"] != 0:
        raise ValueError("R1 smoke requires Lnorm 64 and beta_KL zero")
    if config["optimizer"]["weight_decay"] != 0 or config["R4"]["seed"] != 17:
        raise ValueError("R1 smoke requires weight decay zero and seed 17")
    optimizer = config["optimizer"]
    return {
        "data_root": config["data_root"],
        "sample_seed": 17,
        "generation": dict(config["generation_proposed_N"]),
        "training": {
            "B": 4,
            "K": 8,
            "smoke_updates": 4,
            "seed": 17,
            "Lnorm": config["loss"]["Lnorm"],
            "clip_epsilon": config["loss"]["ppo_clip_upper"] - 1,
            "grad_clip": optimizer["max_grad_norm"],
            "lr": optimizer["learning_rate"],
            "betas": optimizer["betas"],
            "adam_eps": optimizer["eps"],
        },
    }


def _sample_checks(groups, config, *, initial=False):
    import numpy as np

    records = []
    for group in groups:
        for row in group:
            old = row["old_logprobs"]
            behavior = row["behavior_token_logprobs"]
            reference = row["base_token_logprobs"]
            if (
                not old
                or len({len(old), len(behavior), len(reference), len(row["token_ids"])}) != 1
            ):
                raise RuntimeError("R1 generated-token likelihood lengths differ")
            if not np.isfinite([*old, *behavior, *reference]).all():
                raise RuntimeError("R1 generated-token likelihood is nonfinite")
            delta = np.asarray(old) - np.asarray(behavior)
            mean = float(np.abs(delta).mean())
            p99 = float(np.quantile(np.abs(delta), 0.99))
            base_delta = np.asarray(old) - np.asarray(reference)
            base_mean = float(np.abs(base_delta).mean())
            base_p99 = float(np.quantile(np.abs(base_delta), 0.99))
            passed = (
                mean <= config["R1"]["parity_alarm_mean_abs_token_logp"]
                and p99 <= config["R1"]["parity_alarm_p99_abs_token_logp"]
                and (
                    not initial
                    or (
                        base_mean <= config["R1"]["parity_alarm_mean_abs_token_logp"]
                        and base_p99 <= config["R1"]["parity_alarm_p99_abs_token_logp"]
                    )
                )
            )
            records.append(
                {
                    "sample_key": row["sample_key"],
                    "mean_abs_token_logp": mean,
                    "p99_abs_token_logp": p99,
                    "sequence_log_ratio_old_minus_behavior": float(delta.sum()),
                    "initial_base_reference_mean_abs_token_logp": base_mean if initial else None,
                    "initial_base_reference_p99_abs_token_logp": base_p99 if initial else None,
                    "top1_agreement": None,
                    "top1_basis": (
                        "initial path certificate; selected-token scores audited per bank"
                    ),
                    "passed": passed,
                }
            )
    return {"passed": all(row["passed"] for row in records), "records": records}


def _gradients(model):
    return {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }


def _compare_gradients(left, right):
    import torch

    same_keys = bool(left) and set(left) == set(right)
    finite = same_keys and all(torch.isfinite(x).all() for x in [*left.values(), *right.values()])
    nonzero = finite and any(bool(x.abs().max() > 0) for x in left.values())
    equal = finite and all(torch.allclose(left[k], right[k], atol=1e-6, rtol=1e-5) for k in left)
    return {
        "finite": bool(finite),
        "nonzero": bool(nonzero),
        "max_abs_difference": max(
            (float((left[k] - right[k]).abs().max()) for k in left), default=0
        )
        if same_keys
        else None,
        "atol": 1e-6,
        "rtol": 1e-5,
        "passed": bool(nonzero and equal),
    }


def _invariant_probes(adapter, optimizer, groups, helper, telemetry):
    """Two existing valid actions form a diagnostic subbank, never a training bank."""
    import torch

    pair = None
    for group in groups:
        exact = [row for row in group if row["category"] == "X"]
        other = [row for row in group if row["category"] in ("S", "W")]
        if exact and other:
            pair = [
                min(exact, key=lambda r: len(r["token_ids"])),
                min(other, key=lambda r: len(r["token_ids"])),
            ]
            break
    if pair is None:
        return {
            "status": "INCONCLUSIVE",
            "reason": "No same-prompt natural X and S/W pair in this on-policy bank",
            "optimizer_updates": 0,
            "backward_calls": 0,
        }
    origin = capture_state(adapter.model, optimizer)
    candidates, gradients, updates = [], [], []
    accumulation_gradients = []
    try:
        for lam in (0.0, 1.0):
            restore_state(adapter.model, optimizer, origin)
            diagnostic = [
                {**row, "reward_sum": reward_channels(row["category"], "X_BASE", lam)["sum"]}
                for row in pair
            ]
            updates.append(_real_update(adapter, optimizer, [diagnostic], helper, telemetry))
            gradients.append(_gradients(adapter.model))
            candidates.append(capture_state(adapter.model, optimizer))
        for combined in (False, True):
            restore_state(adapter.model, optimizer, origin)
            optimizer.zero_grad(set_to_none=True)
            advantages = grouped_advantages([row["reward_sum"] for row in pair])["advantages"]
            losses = []
            for row, advantage in zip(pair, advantages, strict=True):
                new = adapter.logprobs(row["prepared"], row["token_ids"], require_grad=True)
                old = torch.tensor(row["old_logprobs"], device=new.device, dtype=torch.float32)
                loss, _ = torch_ppo_loss(
                    new[None],
                    old[None],
                    torch.tensor([advantage], device=new.device),
                    torch.ones_like(new[None], dtype=torch.bool),
                    lnorm=helper["training"]["Lnorm"],
                    clip_epsilon=helper["training"]["clip_epsilon"],
                    total_sequences=2,
                )
                if combined:
                    losses.append(loss)
                else:
                    loss.backward()
            if combined:
                sum(losses).backward()
            accumulation_gradients.append(_gradients(adapter.model))
        lambda_gradient = _compare_gradients(*gradients)
        lambda_gradient["preclip_norms_equal"] = math.isclose(
            updates[0]["grad_norm_preclip"],
            updates[1]["grad_norm_preclip"],
            rel_tol=1e-5,
            abs_tol=1e-6,
        )
        lambda_gradient["passed"] = (
            lambda_gradient["passed"] and lambda_gradient["preclip_norms_equal"]
        )
        parameter = compare_parameters(*candidates)
        optimizer_equal = state_hash(candidates[0]["optimizer"]) == state_hash(
            candidates[1]["optimizer"]
        )
        accumulation = _compare_gradients(*accumulation_gradients)
        passed = (
            lambda_gradient["passed"]
            and parameter["parameters_allclose"]
            and optimizer_equal
            and accumulation["passed"]
        )
        return {
            "status": "PASS" if passed else "FAIL",
            "sample_keys": [row["sample_key"] for row in pair],
            "selection": "first same-prompt group with natural X and S/W; shortest of each",
            "bank_role": "existing-rollout diagnostic subbank; main training banks unchanged",
            "all_valid_lambda0_lambda1": {
                "gradients": lambda_gradient,
                "parameters": parameter,
                "optimizer_equal": optimizer_equal,
            },
            "gradient_accumulation": {
                **accumulation,
                "comparison": "two single-sequence backward calls versus one summed-loss backward",
                "batched_model_forward_tested": False,
            },
            "optimizer_updates": 2,
            "backward_calls": 7,
            "update_records": updates,
        }
    finally:
        restore_state(adapter.model, optimizer, origin)
        optimizer.zero_grad(set_to_none=True)
        adapter.model.eval()


def run_reference_smoke(config, scenes, out, adapter, certificate):
    """Return complete or failed evidence; always discard isolated update state."""
    import torch

    frozen_before = _certificate_check(certificate, adapter, config)
    if adapter.model_id != config["model"]["id"] or adapter.revision != config["model"]["revision"]:
        raise ValueError("Smoke adapter differs from the configured model lock")
    helper = _helper_config(config)
    if not scenes or any(scene["split"] != "calibration" for scene in scenes):
        raise ValueError("R1 smoke only accepts nonempty calibration scenes")
    if len({s["base_scene_id"] for s in scenes}) != len(scenes):
        raise ValueError("R1 smoke requires unique calibration base_scene_id values")
    panel = sorted(
        scenes, key=lambda s: canonical_hash(["R1-reference-smoke-v1", s["base_scene_id"]])
    )[:16]
    if len(panel) < 4 or len({s["image_hash"] for s in panel}) < 2:
        raise ValueError("R1 smoke needs at least four scenes and two distinct images")
    _verify_scene_images(panel, config["data_root"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    identity = {
        "model_hash": canonical_hash(
            [adapter.model_id, adapter.revision, certificate["initial_adapter_hash"]]
        ),
        "data_hash": canonical_hash(panel),
        "config_hash": canonical_hash(config),
    }
    store = _R1Store(out, identity)
    train = helper["training"]
    optimizer = torch.optim.AdamW(
        [p for p in adapter.model.parameters() if p.requires_grad],
        lr=train["lr"],
        betas=tuple(train["betas"]),
        eps=train["adam_eps"],
        weight_decay=0.0,
    )
    initial = capture_state(adapter.model, optimizer, {"step": 0})
    real_marker = str(adapter.audit.get("execution_kind", "")).startswith("REAL_CUDA")
    real_device = all(p.device.type == "cuda" for p in adapter.model.parameters())
    execution_kind = (
        "REAL_CUDA_TRAINING_SMOKE" if real_marker and real_device else "CPU_FAKE_ADAPTER_FIXTURE"
    )
    telemetry = _PhaseTelemetry()
    started = time.perf_counter()
    forward_start, generation_start = adapter.forward_calls, adapter.generation_calls
    lock = {
        "identity": identity,
        "source_commit": source_commit(),
        "config": config,
        "certificate": certificate,
        "certificate_hash": canonical_hash(certificate),
        "selected_path": PATH,
        "execution_kind": execution_kind,
        "initial_adapter_hash": certificate["initial_adapter_hash"],
        "initial_optimizer_hash": state_hash(initial["optimizer"]),
        "initial_rng_hash": state_hash(initial["rng"]),
        "base_scene_ids": [s["base_scene_id"] for s in panel],
        "main_optimizer_steps": 4,
        "B": 4,
        "K": 8,
        "maximum_new_rollouts": 128,
        "cache_optimization_paths": "NOT_SELECTED; prior failures retained separately",
        "I4": "NOT_RUN",
        "zero_gradient_semantics": (
            "zero_grad(set_to_none=True); backward every sample including zero advantage; "
            "Adam step always called, including all-zero reward gradients"
        ),
    }
    write_json(out / "runtime_lock.json", lock)
    save_checkpoint(out / "initial_checkpoint.pt", initial, identity)
    steps, phases, sampling = [], [], []
    image, replay, invariants = {}, {}, {"status": "INCONCLUSIVE", "reason": "not measured"}
    error = None
    try:
        prepared = {
            (s["base_scene_id"], interface): adapter.prepare(
                build_prompt(s, interface), config["data_root"]
            )
            for s in panel
            for interface in INTERFACES
        }
        telemetry.begin_phase()
        before = adapter.forward_calls
        image = _image_audit(adapter, prepared, panel, helper)
        phases.append(
            {
                "phase": "image_influence",
                "forwards": adapter.forward_calls - before,
                **telemetry.snapshot(),
            }
        )
        write_json(out / "image_influence_audit.json", image)
        if not image["passed"]:
            raise RuntimeError("R1 image influence failed before optimizer updates")
        for step in range(4):
            write_json(
                out / "progress.json",
                {
                    "status": "RUNNING",
                    "stage": "on_policy_sampling",
                    "completed_steps": len(steps),
                    "rollouts": len(store.records),
                },
            )
            before, begin = adapter.forward_calls, time.perf_counter()
            telemetry.begin_phase()
            groups = []
            for position in range(4):
                scene = panel[(step * 4 + position) % len(panel)]
                interface = INTERFACES[position % 2]
                groups.append(
                    _collect_group(
                        adapter,
                        store,
                        scene,
                        interface,
                        prepared[(scene["base_scene_id"], interface)],
                        step,
                        canonical_hash(
                            [
                                "R1",
                                adapter.model_id,
                                adapter.revision,
                                certificate["initial_adapter_hash"],
                                scene["base_scene_id"],
                                interface,
                                "update",
                                step,
                                position,
                            ]
                        ),
                        helper,
                        telemetry,
                    )
                )
            phases.append(
                {
                    "phase": f"sampling_step_{step}",
                    "forwards": adapter.forward_calls - before,
                    "seconds": time.perf_counter() - begin,
                    **telemetry.phase_snapshot(),
                }
            )
            check = _sample_checks(groups, config, initial=step == 0)
            sampling.append({"step": step, **check})
            write_json(out / "sampling_likelihood_audit.json", sampling)
            if not check["passed"]:
                raise RuntimeError(
                    "R1 current on-policy bank failed likelihood checks; no next update"
                )
            if invariants["status"] == "INCONCLUSIVE":
                write_json(
                    out / "progress.json",
                    {
                        "status": "RUNNING",
                        "stage": "implementation_invariants",
                        "completed_steps": len(steps),
                        "rollouts": len(store.records),
                    },
                )
                before, begin = adapter.forward_calls, time.perf_counter()
                telemetry.begin_phase()
                invariants = _invariant_probes(adapter, optimizer, groups, helper, telemetry)
                phases.append(
                    {
                        "phase": f"invariant_probes_step_{step}",
                        "forwards": adapter.forward_calls - before,
                        "seconds": time.perf_counter() - begin,
                        **telemetry.phase_snapshot(),
                    }
                )
                write_json(out / "implementation_invariants.json", invariants)
                if invariants["status"] == "FAIL":
                    raise RuntimeError("R1 actual gradient/update invariant failed")
            origin = capture_state(
                adapter.model, optimizer, {"step": step, "sample_keys": sorted(store.keys)}
            )
            write_json(
                out / "progress.json",
                {
                    "status": "RUNNING",
                    "stage": "optimizer_update",
                    "completed_steps": len(steps),
                    "rollouts": len(store.records),
                },
            )
            before = adapter.forward_calls
            telemetry.begin_phase()
            update = _real_update(adapter, optimizer, groups, helper, telemetry)
            update["trainable_parameters_without_gradient"] = sum(
                p.grad is None for p in adapter.model.parameters() if p.requires_grad
            )
            update["optimizer_parameter_step_counts"] = sorted(
                {int(state["step"]) for state in optimizer.state.values() if "step" in state}
            )
            steps.append({**update, "optimizer_step": step + 1})
            phases.append(
                {
                    "phase": f"update_{step + 1}",
                    "forwards": adapter.forward_calls - before,
                    **update,
                }
            )
            expected = capture_state(
                adapter.model,
                optimizer,
                {"step": step + 1, "sample_keys": sorted(store.keys), "training_steps": steps},
            )
            save_checkpoint(out / "checkpoint.pt", expected, identity)
            write_json(out / "training_steps.json", steps)
            _check_kl_alarms(steps, helper)
            if step == 1:
                before, begin = adapter.forward_calls, time.perf_counter()
                telemetry.begin_phase()
                replay = _replay_audit(
                    adapter, optimizer, origin, expected, groups, helper, telemetry, out, identity
                )
                phases.append(
                    {
                        "phase": "save_resume_replay",
                        "forwards": adapter.forward_calls - before,
                        "seconds": time.perf_counter() - begin,
                        "backward_calls": sum(
                            x["backward_calls"] for x in replay["replay_updates"]
                        ),
                        **telemetry.phase_snapshot(),
                    }
                )
                if not replay["passed"]:
                    raise RuntimeError("R1 save/resume replay failed")
        frozen_after = parameter_hash(adapter.model, trainable=False)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        frozen_after = parameter_hash(adapter.model, trainable=False)
    finally:
        restore_state(adapter.model, optimizer, initial)
        optimizer.zero_grad(set_to_none=True)
        adapter.model.eval()
        restored = capture_state(adapter.model, optimizer, initial["metadata"])
        restoration = {
            "parameters": state_hash(restored["parameters"]) == state_hash(initial["parameters"]),
            "empty_optimizer": not restored["optimizer"]["state"],
            "optimizer": state_hash(restored["optimizer"]) == state_hash(initial["optimizer"]),
            "rng": state_hash(restored["rng"]) == state_hash(initial["rng"]),
        }
        write_json(
            out / "scratch_state_discarded.json",
            {**restoration, "passed": all(restoration.values())},
        )
    rows = list(store.records.values())
    checks = {
        "reference_certificate": True,
        "sampling_likelihood": bool(sampling) and all(x["passed"] for x in sampling),
        "image": image.get("passed", False),
        "frozen_base": frozen_before == frozen_after,
        "nonempty_gradients_and_measured_step": any(
            s["actual_step_norm"] > 0 and s["grad_norm_preclip"] > 0 for s in steps
        ),
        "fork_resume": replay.get("passed", False),
        "real_optimizer_steps": len(steps) == 4,
        "exact_on_policy_rollout_budget": len(rows) == 128,
        "scratch_state_discarded": all(restoration.values()),
        "implementation_invariants": invariants["status"] == "PASS",
    }
    core = [v for k, v in checks.items() if k != "implementation_invariants"]
    status = "PASS" if error is None and all(checks.values()) else "FAIL"
    if error is None and all(core) and invariants["status"] == "INCONCLUSIVE":
        status = "INCONCLUSIVE"
    elapsed = sum(r["elapsed_seconds"] for r in rows)
    tokens = sum(r["completion_length"] for r in rows)
    result = {
        **adapter.audit,
        "status": status,
        "passed": status == "PASS",
        "checks": checks,
        "execution_kind": execution_kind,
        "selected_path": PATH,
        "runtime_lock": lock,
        "training_steps": steps,
        "raw_sample_count": len(rows),
        "initial_rollout_count": 0,
        "initial_prompt_count": 0,
        "generated_tokens": tokens,
        "generation_seconds": elapsed,
        "generation_tokens_per_second": tokens / elapsed if elapsed else None,
        "generation_calls_this_invocation": adapter.generation_calls - generation_start,
        "forward_calls_this_invocation": adapter.forward_calls - forward_start,
        "replay_optimizer_updates": sum(
            x["optimizer_updates"] for x in replay.get("replay_updates", [])
        ),
        "invariant_optimizer_updates": invariants.get("optimizer_updates", 0),
        "backward_calls_including_replays": sum(x["backward_calls"] for x in steps)
        + sum(x["backward_calls"] for x in replay.get("replay_updates", []))
        + invariants.get("backward_calls", 0),
        "phase_measurements": phases,
        "image_audit": image,
        "fork_resume_audit": replay,
        "implementation_invariants": invariants,
        "scratch_restoration": restoration,
        "frozen_base_hash_before": frozen_before,
        "frozen_base_hash_after": frozen_after,
        "device_memory_total_bytes": torch.cuda.get_device_properties(
            next(adapter.model.parameters()).device
        ).total_memory
        if real_device
        else None,
        "wall_seconds": time.perf_counter() - started,
        "budget_estimation": {
            "actual_rollouts": len(rows),
            "mean_update_seconds": math.fsum(s["update_seconds"] for s in steps) / len(steps)
            if steps
            else None,
            "projection_is_not_a_runtime_guarantee": True,
        },
        "error": error,
        "I4": "NOT_RUN",
        "later_phases": "NOT_RUN",
    }
    write_json(out / "training_smoke.json", result)
    write_json(out / "model_audit.json", result)
    write_json(
        out / "progress.json",
        {"status": status, "completed_steps": len(steps), "rollouts": len(rows), "error": error},
    )
    return result
