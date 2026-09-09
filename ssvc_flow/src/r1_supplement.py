"""Read-only R1 evidence input, followed by disposable fixed-bank Adam controls."""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path

from .core import (
    PROJECT_ROOT,
    canonical_hash,
    file_hash,
    phase_artifacts,
    source_commit,
    write_json,
)
from .grpo_update import grouped_advantages, reward_channels
from .optimizer_fork import (
    capture_state,
    compare_parameters,
    load_checkpoint,
    parameter_hash,
    restore_state,
    state_hash,
)
from .prompts import build_prompt
from .r1_reference_smoke import (
    _certificate_check,
    _compare_gradients,
    _gradients,
    _helper_config,
    _PhaseTelemetry,
)
from .smoke_runtime import _real_update, _seed_everything, _verify_scene_images
from .verifiers import annotate

SOURCE_FILES = (
    "src/r1_reference_smoke.py",
    "src/r1_parity.py",
    "src/grpo_update.py",
    "src/smoke_runtime.py",
    "src/optimizer_fork.py",
    "src/model_adapters/base.py",
)


def _json(path):
    return json.loads(Path(path).read_text())


def _parameter_state_hash(state):
    return state_hash({name: state_hash(value) for name, value in state["parameters"].items()})


def _load_evidence(root, fixture):
    required = (
        "status.json",
        "manifest.json",
        "training_smoke.json",
        "source-commit.txt",
        "production_path_validation.json",
        "smoke/runtime_lock.json",
        "smoke/identity.json",
        "smoke/samples.jsonl",
        "smoke/initial_checkpoint.pt",
        "smoke/resume_probe.pt",
        "smoke/checkpoint.pt",
    )
    hashes = {str(root / name): file_hash(root / name) for name in required}
    manifest = _json(root / "manifest.json")
    entries = manifest.get("files", [])
    listed = [item["path"] for item in entries]
    if (
        manifest.get("phase") != "R1"
        or len(listed) != len(set(listed))
        or not {"production_path_validation.json", "training_smoke.json"} <= set(listed)
    ):
        raise ValueError("Original R1 manifest is incomplete or contains duplicate entries")
    for item in entries:
        relative = Path(item["path"])
        target = (root / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(root):
            raise ValueError("Original R1 manifest path escapes the evidence directory")
        if file_hash(target) != item["sha256"] or target.stat().st_size != item["bytes"]:
            raise ValueError(f"Original R1 manifest hash/size mismatch: {relative}")
        hashes[str(target)] = item["sha256"]
    status = _json(root / "status.json")
    original_smoke = _json(root / "training_smoke.json")
    certificate = _json(root / "production_path_validation.json")
    lock = _json(root / "smoke/runtime_lock.json")
    config, identity = lock["config"], lock["identity"]
    if (
        original_smoke.get("status") != "PASS"
        or original_smoke.get("passed") is not True
        or original_smoke["runtime_lock"] != lock
        or original_smoke["raw_sample_count"] != 128
    ):
        raise ValueError("Manifest-bound training smoke disagrees with the supplied runtime lock")
    revision = status["source_commit"]
    if status["status"] != "PASS" or certificate["status"] != "PASS":
        raise ValueError("The original R1 and reference certificate must both have passed")
    if not fixture and status.get("execution_kind") != "REAL_CUDA_TRAINING_SMOKE":
        raise ValueError("The production supplement requires actual CUDA R1 evidence")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Original source commit is not an immutable Git revision")
    if any(
        value != revision
        for value in (
            lock["source_commit"],
            certificate["source_commit"],
            (root / "source-commit.txt").read_text().strip(),
        )
    ):
        raise ValueError("Original source evidence revisions disagree")
    for name in SOURCE_FILES:
        original = subprocess.run(
            ["git", "show", f"{revision}:ssvc_flow/{name}"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
        ).stdout
        expected = status["source_files"].get(name)
        if (
            hashlib.sha256(original).hexdigest() != expected
            or file_hash(PROJECT_ROOT / name) != expected
        ):
            raise ValueError(f"Original or current source implementation drift: {name}")
    if lock["certificate"] != certificate or lock["certificate_hash"] != canonical_hash(
        certificate
    ):
        raise ValueError("Original certificate content or hash differs from the smoke lock")
    if identity != _json(root / "smoke/identity.json") or identity["config_hash"] != canonical_hash(
        config
    ):
        raise ValueError("Original configuration or identity hash differs")
    expected_model = canonical_hash(
        [
            config["model"]["id"],
            config["model"]["revision"],
            certificate["initial_adapter_hash"],
        ]
    )
    if (
        identity["model_hash"] != expected_model
        or certificate["model_revision"] != config["model"]["revision"]
    ):
        raise ValueError("Original model lock differs from the certificate")
    rows = [json.loads(line) for line in (root / "smoke/samples.jsonl").read_text().splitlines()]
    keys = [row["sample_key"] for row in rows]
    if (
        len(rows) != 128
        or len(set(keys)) != 128
        or any(row["split"] != "calibration" for row in rows)
    ):
        raise ValueError("Expected the original 128 unique calibration samples")
    calibration = Path(config["data_root"]) / "calibration.jsonl"
    hashes[str(calibration)] = file_hash(calibration)
    scenes = [json.loads(line) for line in calibration.read_text().splitlines()]
    scene_map = {row["base_scene_id"]: row for row in scenes}
    if len(scene_map) != len(scenes):
        raise ValueError("Calibration contains duplicate scene identifiers")
    panel = [scene_map[key] for key in lock["base_scene_ids"]]
    if canonical_hash(panel) != identity["data_hash"]:
        raise ValueError("Original calibration panel hash differs")
    _verify_scene_images(panel, config["data_root"])
    initial = load_checkpoint(root / "smoke/initial_checkpoint.pt", identity)
    warm = load_checkpoint(root / "smoke/resume_probe.pt", identity)
    final = load_checkpoint(root / "smoke/checkpoint.pt", identity)
    if (
        _parameter_state_hash(initial) != certificate["initial_adapter_hash"]
        or state_hash(initial["optimizer"]) != lock["initial_optimizer_hash"]
        or state_hash(initial["rng"]) != lock["initial_rng_hash"]
    ):
        raise ValueError("Initial checkpoint differs from the original smoke lock")
    if warm["metadata"].get("step") != 1 or final["metadata"].get("step") != 4:
        raise ValueError("Only the original step1 resume probe and step4 completion are supported")
    if final["metadata"]["training_steps"] != original_smoke["training_steps"]:
        raise ValueError("Final checkpoint updates differ from manifest-bound smoke measurements")
    if set(final["metadata"]["sample_keys"]) != set(keys):
        raise ValueError("Final checkpoint sample identities differ from the ledger")
    if set(warm["metadata"]["sample_keys"]) != {
        row["sample_key"] for row in rows if row["optimizer_step"] <= 1
    }:
        raise ValueError("Step1 checkpoint sample identities differ from the ledger")
    bank = [row for row in rows if row["optimizer_step"] == 1]
    warm_hash = _parameter_state_hash(warm)
    warm_start = {
        "parameters": warm_hash,
        "optimizer": state_hash(warm["optimizer"]),
        "rng": state_hash(warm["rng"]),
    }
    if original_smoke["fork_resume_audit"]["starts"] != [warm_start, warm_start]:
        raise ValueError("Step1 checkpoint differs from manifest-bound resume probe measurements")
    if len(bank) != 32 or any(row["adapter_hash"] != warm_hash for row in bank):
        raise ValueError("The step1 bank policy differs from the resume checkpoint")
    if any(
        row["model_revision"] != config["model"]["revision"]
        or row["model_id"] != config["model"]["id"]
        for row in bank
    ):
        raise ValueError("The step1 bank model identity differs")
    return config, certificate, warm, bank, scene_map, hashes, revision


class _ExecutionMeter:
    """Observe backward entry and actual layer forwards during that entry."""

    def __init__(self, adapter, optimizer):
        import torch

        self.phase = "binding"
        self.in_backward = False
        self.counts = collections.defaultdict(lambda: collections.Counter())
        self.layer_calls = collections.Counter()
        self.handles = []
        self.elapsed = collections.Counter()
        self.telemetry = _PhaseTelemetry()
        self.memory = {}
        self.backward_calls = self.optimizer_calls = 0
        self.sync_seconds = 0.0
        self.original_backward = torch.autograd.backward
        base = (
            adapter.model.get_base_model()
            if hasattr(adapter.model, "get_base_model")
            else adapter.model
        )
        backbone = getattr(base, "model", None)
        if backbone is not None:
            self.handles.append(backbone.register_forward_pre_hook(self._top, with_kwargs=True))
        layers = [
            (n, m)
            for n, m in adapter.model.named_modules()
            if re.search(r"\.language_model\.layers\.\d+$", n)
        ]
        self.language_layer_names = [n for n, _ in layers]
        for _, module in layers:
            self.handles.append(module.register_forward_pre_hook(self._layer))
        visual = [
            (n, m)
            for n, m in adapter.model.named_modules()
            if n.endswith(".model.visual") or n == "model.visual"
        ]
        for _, module in visual:
            self.handles.append(module.register_forward_pre_hook(self._vision))
        self.top_hook_installed = backbone is not None
        self.vision_hook_count = len(visual)
        self.handles.append(optimizer.register_step_post_hook(self._step))
        torch.autograd.backward = self._backward

    def _top(self, module, args, kwargs):
        counter = self.counts[self.phase]
        counter["top_level_forward_calls"] += 1
        ids = kwargs.get("input_ids")
        if ids is not None:
            counter["input_token_slots"] += ids.numel()
            mask = kwargs.get("attention_mask")
            if mask is not None and mask.shape == ids.shape:
                counter["padding_token_slots"] += int((mask == 0).sum())

    def _layer(self, module, args):
        import torch

        kind = "during_backward" if self.in_backward else "ordinary_forward"
        key = f"{self.phase}/{kind}/grad_enabled={torch.is_grad_enabled()}"
        self.layer_calls[key] += 1

    def _vision(self, module, args):
        self.counts[self.phase]["vision_encoder_calls"] += 1

    def _step(self, optimizer, args, kwargs):
        self.optimizer_calls += 1
        self.counts[self.phase]["optimizer_step_calls"] += 1

    def _backward(self, *args, **kwargs):
        self.backward_calls += 1
        self.counts[self.phase]["backward_calls"] += 1
        previous = self.in_backward
        self.in_backward = True
        try:
            return self.original_backward(*args, **kwargs)
        finally:
            self.in_backward = previous

    @contextlib.contextmanager
    def scope(self, label):
        import torch

        self.phase = label
        begin_sync = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.sync_seconds += time.perf_counter() - begin_sync
        self.telemetry.begin_phase()
        started = time.perf_counter()
        try:
            yield
        finally:
            begin_sync = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.sync_seconds += time.perf_counter() - begin_sync
            self.elapsed[label] += time.perf_counter() - started
            self.memory[label] = self.telemetry.phase_snapshot()

    def report(self):
        return {
            "backward_calls_observed": self.backward_calls,
            "optimizer_step_calls_observed": self.optimizer_calls,
            "by_phase": {key: dict(value) for key, value in self.counts.items()},
            "language_layer_forward_calls": dict(self.layer_calls),
            "language_layers_instrumented": len(self.language_layer_names),
            "top_hook_installed": self.top_hook_installed,
            "vision_hooks_installed": self.vision_hook_count,
            "internal_recompute_definition": (
                "actual language-layer forward entry while torch.autograd.backward is active; "
                "grad mode retained"
            ),
            "internal_recompute_status": "MEASURED"
            if self.language_layer_names
            else "NOT_INSTRUMENTED",
            "seconds_by_phase": dict(self.elapsed),
            "memory_by_phase": self.memory,
            "device_sync_seconds": self.sync_seconds,
        }

    def close(self):
        import torch

        torch.autograd.backward = self.original_backward
        for handle in self.handles:
            handle.remove()


def _warm_adam_check(state):
    import torch

    values = list(state["optimizer"]["state"].values())
    if not values or state["scheduler"] is not None:
        raise ValueError("A nonempty Adam checkpoint without a scheduler is required")
    if any(
        not {"step", "exp_avg", "exp_avg_sq"} <= set(value) or float(value["step"]) != 1
        for value in values
    ):
        raise ValueError("Expected mature step1 Adam state for every participating parameter")
    if not any(bool(value["exp_avg"].abs().max() > 0) for value in values):
        raise ValueError("Adam checkpoint has no nonzero momentum")
    if any(
        not torch.isfinite(value[key]).all()
        for value in values
        for key in ("exp_avg", "exp_avg_sq")
    ):
        raise ValueError("Adam checkpoint moments are nonfinite")


def _select_diagnostic_rows(bank, scene_map, adapter, config, fixture):
    import numpy as np

    grouped = collections.defaultdict(list)
    for row in bank:
        grouped[(row["interface"] != "SYMBOLIC_FRESH", row["group_id"])].append(row)
    pair = None
    for _, group in sorted(grouped.items()):
        exact = [row for row in group if row["category"] == "X"]
        other = [row for row in group if row["category"] in ("S", "W")]
        invalid = [row for row in group if row["category"] == "I"]
        if exact and other and invalid:
            pair = [
                min(items, key=lambda r: (len(r["token_ids"]), r["sample_key"]))
                for items in (exact, other, invalid)
            ]
            break
    if pair is None:
        raise ValueError("The locked step1 bank has no same-prompt X, S/W and I diagnostic set")
    prepared_rows, audits = [], []
    for row in pair:
        scene = scene_map[row["base_scene_id"]]
        if (
            annotate(row["raw_completion"], scene["truth_world"], scene["operation"], scene["cue"])[
                "category"
            ]
            != row["category"]
        ):
            raise ValueError("Original diagnostic category disagrees with independent reannotation")
        prepared = adapter.prepare(build_prompt(scene, row["interface"]), config["data_root"])
        for key in ("input_tensor_hash", "tokenized_prompt_hash", "final_prompt_hash"):
            expected = row.get(key)
            if (not fixture and not expected) or expected != prepared["audit"].get(key):
                raise ValueError(f"Original diagnostic input metadata drift: {key}")
        observed = adapter.logprobs(prepared, row["token_ids"]).detach().cpu().tolist()
        if len(observed) != len(row["old_logprobs"]) or not observed:
            raise ValueError("Original diagnostic token likelihood length differs")
        delta = np.asarray(observed) - np.asarray(row["old_logprobs"])
        mean, p99 = float(np.abs(delta).mean()), float(np.quantile(np.abs(delta), 0.99))
        passed = bool(
            np.isfinite(delta).all()
            and mean <= config["R1"]["parity_alarm_mean_abs_token_logp"]
            and p99 <= config["R1"]["parity_alarm_p99_abs_token_logp"]
        )
        audits.append(
            {
                "sample_key": row["sample_key"],
                "mean_abs_token_logp": mean,
                "p99_abs_token_logp": p99,
                "sequence_log_ratio": float(delta.sum()),
                "passed": passed,
            }
        )
        if not passed:
            raise ValueError("Original step1 old likelihood differs from the restored policy")
        prepared_rows.append({**row, "prepared": prepared})
    return prepared_rows, audits


def _execute(adapter, optimizer, origin, rows, rewards, helper, meter, label):
    restore_state(adapter.model, optimizer, origin)
    start_hash = state_hash(capture_state(adapter.model, optimizer, origin["metadata"]))
    diagnostic = [{**row, "reward_sum": reward} for row, reward in zip(rows, rewards, strict=True)]
    with meter.scope(label):
        update = _real_update(adapter, optimizer, [diagnostic], helper, meter.telemetry)
    return {
        "start_hash": start_hash,
        "state": capture_state(adapter.model, optimizer, origin["metadata"]),
        "gradients": _gradients(adapter.model),
        "update": update,
        "rewards": rewards,
        "advantages": grouped_advantages(rewards)["advantages"],
    }


def _candidate_compare(left, right):
    gradients = _compare_gradients(left["gradients"], right["gradients"])
    parameters = compare_parameters(left["state"], right["state"])
    optimizer_equal = state_hash(left["state"]["optimizer"]) == state_hash(
        right["state"]["optimizer"]
    )
    rng_equal = state_hash(left["state"]["rng"]) == state_hash(right["state"]["rng"])
    preclip = math.isclose(
        left["update"]["grad_norm_preclip"],
        right["update"]["grad_norm_preclip"],
        abs_tol=1e-6,
        rel_tol=1e-5,
    )
    return {
        "same_start": left["start_hash"] == right["start_hash"],
        "gradients": gradients,
        "parameters": parameters,
        "optimizer_equal": optimizer_equal,
        "rng_equal": rng_equal,
        "preclip_norms_equal": preclip,
        "passed": left["start_hash"] == right["start_hash"]
        and gradients["passed"]
        and parameters["parameters_allclose"]
        and optimizer_equal
        and rng_equal
        and preclip,
    }


def _zero_probe(adapter, optimizer, origin, rows, helper, meter):
    import torch

    actual = _execute(
        adapter, optimizer, origin, rows, [0.0] * len(rows), helper, meter, "zero_reward_loss"
    )
    names = {name for name, p in adapter.model.named_parameters() if p.requires_grad}
    explicit_zero = set(actual["gradients"]) == names and all(
        bool(torch.count_nonzero(g) == 0) for g in actual["gradients"].values()
    )
    restore_state(adapter.model, optimizer, origin)
    optimizer.zero_grad(set_to_none=True)
    with meter.scope("manual_zero_tensor_control"):
        for parameter in adapter.model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
    manual = capture_state(adapter.model, optimizer, origin["metadata"])
    same_manual = state_hash(actual["state"]["parameters"]) == state_hash(
        manual["parameters"]
    ) and state_hash(actual["state"]["optimizer"]) == state_hash(manual["optimizer"])
    restore_state(adapter.model, optimizer, origin)
    optimizer.zero_grad(set_to_none=True)
    with meter.scope("none_gradient_control"):
        optimizer.step()
    none = capture_state(adapter.model, optimizer, origin["metadata"])
    none_unchanged = state_hash(none["parameters"]) == state_hash(
        origin["parameters"]
    ) and state_hash(none["optimizer"]) == state_hash(origin["optimizer"])
    beta1, beta2 = helper["training"]["betas"]
    before, after = origin["optimizer"]["state"], actual["state"]["optimizer"]["state"]
    advanced = set(before) == set(after) and all(
        float(after[k]["step"]) == float(before[k]["step"]) + 1 for k in before
    )
    decayed = set(before) == set(after) and all(
        torch.allclose(after[k][field], before[k][field] * beta, atol=1e-7, rtol=1e-5)
        for k in before
        for field, beta in (("exp_avg", beta1), ("exp_avg_sq", beta2))
    )
    changed = compare_parameters(origin, actual["state"])["parameters_max_abs_difference"] > 0
    return {
        "all_advantages_zero": all(value == 0 for value in actual["advantages"]),
        "all_gradients_explicit_zero": explicit_zero,
        "manual_zero_tensor_equal": same_manual,
        "none_control_unchanged": none_unchanged,
        "optimizer_steps_advanced": advanced,
        "moments_decay_as_expected": decayed,
        "parameters_changed": changed,
        "scheduler": "NOT_CONFIGURED",
        "actual_update": actual["update"],
        "passed": explicit_zero
        and same_manual
        and none_unchanged
        and advanced
        and decayed
        and changed,
    }


def run_supplement(r1_run, out, *, _adapter_factory=None):
    """Return evidence with PASS/FAIL/BLOCKED; never generate or retain updated weights."""
    import torch

    root, out = Path(r1_run).resolve(), Path(out).resolve()
    if out == root or out.is_relative_to(root):
        raise ValueError("Supplement output must be outside the original evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("A fresh supplement output directory is required")
    (out / ".supplement.lock").touch(exist_ok=False)
    fixture = _adapter_factory is not None
    result = {
        "status": "FAIL",
        "passed": False,
        "execution_kind": "CPU_FAKE_ADAPTER_FIXTURE" if fixture else "CPU_AUDIT",
        "new_rollouts": 0,
        "batched_model_forward": "NOT_ADOPTED_UNVERIFIED",
        "measurement": {},
        "scratch_restoration": {},
        "error": None,
    }
    adapter = optimizer = initial = meter = None
    source_hashes = {}
    try:
        config, certificate, warm, bank, scene_map, source_hashes, revision = _load_evidence(
            root, fixture
        )
        helper = _helper_config(config)
        _warm_adam_check(warm)
        if not fixture and not torch.cuda.is_available():
            raise RuntimeError("The production supplement requires an authorized CUDA device")
        from .model_adapters import load_adapter

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
        frozen_hash = _certificate_check(certificate, adapter, config)
        train = helper["training"]
        optimizer = torch.optim.AdamW(
            [p for p in adapter.model.parameters() if p.requires_grad],
            lr=train["lr"],
            betas=tuple(train["betas"]),
            eps=train["adam_eps"],
            weight_decay=0.0,
        )
        initial = capture_state(adapter.model, optimizer)
        if not fixture:
            if not str(adapter.audit.get("execution_kind", "")).startswith("REAL_CUDA") or any(
                p.device.type != "cuda" for p in adapter.model.parameters()
            ):
                raise ValueError("The production adapter is not a measured CUDA model")
            result["execution_kind"] = "REAL_CUDA_FORK"
        generation_start = adapter.generation_calls
        meter = _ExecutionMeter(adapter, optimizer)
        if not fixture and (
            not meter.top_hook_installed
            or len(meter.language_layer_names) != 32
            or meter.vision_hook_count != 1
        ):
            raise ValueError(
                "Required actual model forward/layer/vision instrumentation is unavailable"
            )
        restore_state(adapter.model, optimizer, warm)
        with meter.scope("step1_bank_binding"):
            order_rows, likelihood = _select_diagnostic_rows(
                bank, scene_map, adapter, config, fixture
            )
        rows = order_rows[:2]
        binding = {
            "original_run": str(root),
            "original_source_commit": revision,
            "supplement_source_commit": source_commit(),
            "supplement_source_sha256": file_hash(__file__),
            "original_file_hashes": source_hashes,
            "original_config": config,
            "original_config_sha256": canonical_hash(config),
            "certificate_sha256": canonical_hash(certificate),
            "checkpoint": "smoke/resume_probe.pt",
            "checkpoint_step": 1,
            "checkpoint_state_hash": state_hash(warm),
            "model_revision": adapter.revision,
            "initial_adapter_hash": certificate["initial_adapter_hash"],
            "frozen_parameter_hash": frozen_hash,
            "selected_sample_keys": [r["sample_key"] for r in order_rows],
            "zero_and_lambda0_sample_keys": [r["sample_key"] for r in rows],
            "selection": (
                "first stable same-prompt X, S/W and I group, symbolic preferred; "
                "shortest action in each category"
            ),
            "bank_role": (
                "fixed-bank diagnostic at the exact step1 policy; "
                "no new sampling or formal training"
            ),
            "likelihood_binding": likelihood,
        }
        write_json(out / "input_binding.json", binding)
        result["selected_checkpoint_step"] = 1
        result["zero_gradient_adam"] = _zero_probe(adapter, optimizer, warm, rows, helper, meter)
        write_json(out / "zero_gradient_adam.json", result["zero_gradient_adam"])
        baseline_rewards = [2.0 if row["category"] == "X" else 0.0 for row in rows]
        candidate_rewards = [
            reward_channels(row["category"], "X_VALID", 0.0)["sum"] for row in rows
        ]
        baseline = _execute(
            adapter, optimizer, warm, rows, baseline_rewards, helper, meter, "independent_X_BASE"
        )
        candidate = _execute(
            adapter,
            optimizer,
            warm,
            rows,
            candidate_rewards,
            helper,
            meter,
            "generic_candidate_lambda0",
        )
        comparison = _candidate_compare(baseline, candidate)
        result["lambda_zero_control"] = {
            **comparison,
            "independent_reward_implementations": True,
            "baseline_definition": (
                "2 * indicator(category == X), implemented independently of reward_channels"
            ),
            "candidate_definition": "reward_channels(category, X_VALID, auxiliary_weight=0)",
            "rewards_equal": baseline_rewards == candidate_rewards,
            "advantages_equal": baseline["advantages"] == candidate["advantages"],
            "baseline_rewards": baseline_rewards,
            "candidate_rewards": candidate_rewards,
            "passed": comparison["passed"]
            and baseline_rewards == candidate_rewards
            and baseline["advantages"] == candidate["advantages"],
        }
        write_json(out / "lambda_zero_control.json", result["lambda_zero_control"])
        orders = [[0.0, 1.0], [1.0, 0.0]]
        outcomes = []
        for index, order in enumerate(orders):
            values = {}
            for lam in order:
                rewards = [
                    reward_channels(row["category"], "X_VALID", lam)["sum"] for row in order_rows
                ]
                values[lam] = _execute(
                    adapter,
                    optimizer,
                    warm,
                    order_rows,
                    rewards,
                    helper,
                    meter,
                    f"order{index}_lambda{lam}",
                )
            outcomes.append(values)
        comparisons = {
            str(lam): _candidate_compare(outcomes[0][lam], outcomes[1][lam]) for lam in (0.0, 1.0)
        }
        distinct = outcomes[0][0.0]["advantages"] != outcomes[0][1.0]["advantages"]
        result["candidate_order"] = {
            "orders": orders,
            "comparisons": comparisons,
            "categories": [row["category"] for row in order_rows],
            "lambda_candidates_have_distinct_advantages": distinct,
            "passed": distinct and all(c["passed"] for c in comparisons.values()),
        }
        write_json(out / "candidate_order.json", result["candidate_order"])
        result["frozen_base_unchanged"] = (
            parameter_hash(adapter.model, trainable=False) == frozen_hash
        )
        result["new_rollouts"] = adapter.generation_calls - generation_start
        if result["new_rollouts"] != 0:
            raise RuntimeError("A supplement must never generate new samples")
        result["passed"] = result["frozen_base_unchanged"] and all(
            result[key]["passed"]
            for key in ("zero_gradient_adam", "lambda_zero_control", "candidate_order")
        )
        result["status"] = "PASS" if result["passed"] else "FAIL"
    except FileNotFoundError as exc:
        result.update(status="BLOCKED", passed=False, error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        result.update(status="FAIL", passed=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        if meter is not None:
            result["measurement"] = meter.report()
            meter.close()
        if initial is not None:
            try:
                restore_state(adapter.model, optimizer, initial)
                optimizer.zero_grad(set_to_none=True)
                adapter.model.eval()
                restored = capture_state(adapter.model, optimizer, initial["metadata"])
                result["scratch_restoration"] = {
                    "parameters": state_hash(restored["parameters"])
                    == state_hash(initial["parameters"]),
                    "empty_optimizer": not restored["optimizer"]["state"],
                    "optimizer": state_hash(restored["optimizer"])
                    == state_hash(initial["optimizer"]),
                    "rng": state_hash(restored["rng"]) == state_hash(initial["rng"]),
                }
            except Exception as exc:
                result["scratch_restoration"] = {
                    "parameters": False,
                    "optimizer": False,
                    "empty_optimizer": False,
                    "rng": False,
                }
                result["restoration_error"] = f"{type(exc).__name__}: {exc}"
            if not all(result["scratch_restoration"].values()):
                result.update(status="FAIL", passed=False, error="Scratch state restoration failed")
        result["original_files_unchanged"] = (
            all(
                Path(name).is_file() and file_hash(name) == digest
                for name, digest in source_hashes.items()
            )
            if source_hashes
            else None
        )
        if result["original_files_unchanged"] is False:
            result.update(
                status="FAIL", passed=False, error="Original evidence changed during supplement"
            )
    write_json(out / "measurement.json", result["measurement"])
    write_json(out / "supplement.json", result)
    document = phase_artifacts(
        out, "R1_SUPPLEMENT", result["status"], result, sorted(out.glob("*.json"))
    )
    document["execution_kind"] = result["execution_kind"]
    write_json(out / "status.json", document)
    (out / "report_zh.md").write_text((out / "report.md").read_text())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_supplement(args.r1_run, args.out)
    print(
        json.dumps(
            {key: result[key] for key in ("status", "execution_kind", "passed", "error")},
            ensure_ascii=False,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
