"""Inference-only runtime for Phase 5 policy-support measurements."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .phase5_support import (
    CheckpointSceneMeasurement,
    HeldOutNaturalError,
    PolicyCheckpoint,
    parse_world,
)

_ADAPTER_PATHS = {
    "C0": "C0_format_only/final_adapter",
    "C1": "C1_forward_arithmetic/final_adapter",
    "T": "T_constraint_recovery/final_adapter",
}


@dataclass(frozen=True, slots=True)
class Phase5MeasurementConfig:
    temperature: float
    top_p: float
    top_k: int
    rollout_count: int
    pass_at_k: tuple[int, ...]
    informative_group_size: int
    max_new_tokens: int
    sampling_seed: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Phase5MeasurementConfig:
        expected = {
            "temperature",
            "top_p",
            "top_k",
            "rollout_count",
            "pass_at_k",
            "informative_group_size",
            "max_new_tokens",
            "sampling_seed",
        }
        if set(value) != expected:
            raise ValueError("Phase 5 measurement config fields differ from the frozen contract")
        pass_values = value["pass_at_k"]
        if not isinstance(pass_values, Sequence) or isinstance(pass_values, (str, bytes)):
            raise TypeError("Phase 5 pass_at_k must be a sequence")
        config = cls(
            temperature=value["temperature"],  # type: ignore[arg-type]
            top_p=value["top_p"],  # type: ignore[arg-type]
            top_k=value["top_k"],  # type: ignore[arg-type]
            rollout_count=value["rollout_count"],  # type: ignore[arg-type]
            pass_at_k=tuple(pass_values),  # type: ignore[arg-type]
            informative_group_size=value["informative_group_size"],  # type: ignore[arg-type]
            max_new_tokens=value["max_new_tokens"],  # type: ignore[arg-type]
            sampling_seed=value["sampling_seed"],  # type: ignore[arg-type]
        )
        if config.temperature != 0.7:
            raise ValueError("Phase 5 temperature must remain frozen at 0.7")
        if config.top_p != 1.0 or config.top_k != 0:
            raise ValueError("Phase 5 top-p/top-k contract drifted")
        integer_values = (
            config.rollout_count,
            config.informative_group_size,
            config.max_new_tokens,
            config.sampling_seed,
            *config.pass_at_k,
        )
        if any(type(item) is not int or item <= 0 for item in integer_values):
            raise ValueError("Phase 5 integer measurement values must be positive")
        if config.max_new_tokens != 32:
            raise ValueError("Phase 5 max_new_tokens must remain frozen at 32")
        if tuple(sorted(set(config.pass_at_k))) != config.pass_at_k:
            raise ValueError("Phase 5 pass_at_k must be unique and increasing")
        if max((*config.pass_at_k, config.informative_group_size)) > config.rollout_count:
            raise ValueError("Phase 5 K cannot exceed rollout_count")
        return config

    @classmethod
    def default_for_tests(cls, *, rollout_count: int = 4) -> Phase5MeasurementConfig:
        return cls.from_mapping(
            {
                "temperature": 0.7,
                "top_p": 1.0,
                "top_k": 0,
                "rollout_count": rollout_count,
                "pass_at_k": [1, 2, rollout_count],
                "informative_group_size": rollout_count,
                "max_new_tokens": 32,
                "sampling_seed": 2026082005,
            }
        )

    def to_mapping(self) -> dict[str, object]:
        payload = asdict(self)
        payload["pass_at_k"] = list(self.pass_at_k)
        return payload


def phase5_rollout_seed(base_seed: int, scene_id: str, rollout_index: int) -> int:
    """Derive common-random-number seeds without checkpoint identity."""

    if type(base_seed) is not int or base_seed <= 0:
        raise ValueError("Phase 5 base seed must be a positive integer")
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError("Phase 5 scene_id must be non-empty")
    if type(rollout_index) is not int or rollout_index < 0:
        raise ValueError("Phase 5 rollout index must be nonnegative")
    digest = hashlib.sha256(
        f"phase5-rollout:{base_seed}:{scene_id}:{rollout_index}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def tree_sha256(path: Path) -> str:
    """Hash a closed regular-file tree, including relative paths."""

    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"Phase 5 adapter directory is missing or unsafe: {path}")
    files = tuple(sorted(item for item in path.rglob("*") if item.is_file()))
    if not files:
        raise RuntimeError(f"Phase 5 adapter directory is empty: {path}")
    digest = hashlib.sha256()
    for item in files:
        if item.is_symlink():
            raise RuntimeError("Phase 5 adapter tree must not contain symlinks")
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def checkpoint_tree_hashes(run_root: Path) -> dict[str, str]:
    if run_root.is_symlink() or not run_root.is_dir():
        raise RuntimeError("Phase 5 Phase-4 run root is missing or unsafe")
    hashes: dict[str, str] = {}
    for checkpoint, relative in _ADAPTER_PATHS.items():
        adapter = run_root / relative
        if not (adapter / "adapter_config.json").is_file():
            raise RuntimeError(f"Phase 5 {checkpoint} adapter config is missing")
        model_files = tuple(adapter.glob("adapter_model.*"))
        if len(model_files) != 1 or model_files[0].is_symlink():
            raise RuntimeError(f"Phase 5 {checkpoint} adapter weights are missing or ambiguous")
        hashes[checkpoint] = tree_sha256(adapter)
    return hashes


def checkpoint_adapter_path(run_root: Path, checkpoint: PolicyCheckpoint) -> Path | None:
    relative = _ADAPTER_PATHS.get(checkpoint.value)
    return None if relative is None else run_root / relative


def load_phase5_config(path: Path) -> tuple[dict[str, object], Phase5MeasurementConfig]:
    import yaml

    if path.is_symlink() or not path.is_file():
        raise ValueError("Phase 5 config must be a regular file")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "status",
        "model",
        "authorization",
        "support_dev",
        "measurement",
        "integrity_gates",
        "artifacts",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected
        or payload.get("schema_version") != 1
    ):
        raise ValueError("Phase 5 config schema differs from the frozen contract")
    if payload.get("status") != "PHASE_5_POLICY_SUPPORT_AUTHORIZED":
        raise ValueError("Phase 5 authorization status drifted")
    if payload.get("authorization") != {
        "measurement_authorized": True,
        "training_authorized": False,
        "rl_authorized": False,
        "downloads_authorized": False,
    }:
        raise ValueError("Phase 5 measurement-only authorization drifted")
    support = payload.get("support_dev")
    if not isinstance(support, dict) or support != {
        "intake_scene_count": 576,
        "family_balanced": True,
        "scenes_per_family": 192,
        "selection_seed": 2026082005,
        "value_domain_min": 2,
        "value_domain_max": 18,
        "resized_height": 280,
        "resized_width": 280,
        "stage1_max_new_tokens": 32,
        "exactly_one_stage1_call_per_scene": True,
        "exactly_one_error_required": True,
        "retry_forbidden": True,
        "sample_extension_forbidden": True,
    }:
        raise ValueError("Phase 5 support-dev contract drifted")
    measurement = payload.get("measurement")
    if not isinstance(measurement, dict):
        raise ValueError("Phase 5 measurement contract is malformed")
    checkpoints = measurement.get("checkpoints")
    if checkpoints != [checkpoint.value for checkpoint in PolicyCheckpoint]:
        raise ValueError("Phase 5 checkpoint order drifted")
    extra = {
        "checkpoints",
        "common_random_numbers_across_checkpoints",
        "candidate_scoring",
        "scene_is_statistical_unit",
        "subjective_success_thresholds_forbidden",
    }
    config = Phase5MeasurementConfig.from_mapping(
        {key: value for key, value in measurement.items() if key not in extra}
    )
    if (
        measurement.get("common_random_numbers_across_checkpoints") is not True
        or measurement.get("candidate_scoring") != "truth_vs_observed_sequence_log_probability"
        or measurement.get("scene_is_statistical_unit") is not True
        or measurement.get("subjective_success_thresholds_forbidden") is not True
    ):
        raise ValueError("Phase 5 measurement integrity contract drifted")
    return payload, config


def verify_phase5_package_lock(
    *, lock_path: Path, repository_root: Path, expected_paths: Sequence[str]
) -> str:
    import yaml

    if lock_path.is_symlink() or not lock_path.is_file():
        raise ValueError("Phase 5 package lock must be a regular file")
    payload = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    rows = payload.get("files") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("status") != "FROZEN_PHASE_5_POLICY_SUPPORT_SURFACE"
        or not isinstance(rows, list)
        or not rows
    ):
        raise ValueError("Phase 5 package lock is malformed")
    observed: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError("Phase 5 package lock row is malformed")
        relative, digest = row["path"], row["sha256"]
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or relative in observed
        ):
            raise ValueError("Phase 5 package lock row has invalid fields")
        candidate = repository_root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError(f"Phase 5 package lock missing file: {relative}")
        file_digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if file_digest != digest:
            raise RuntimeError(f"Phase 5 package lock mismatch: {relative}")
        observed.add(relative)
    if observed != set(expected_paths):
        raise RuntimeError("Phase 5 package lock closure mismatch")
    return hashlib.sha256(lock_path.read_bytes()).hexdigest()


def recovery_prompt(error: HeldOutNaturalError) -> str:
    observed = ",".join(str(value) for value in error.observed)
    facts = json.dumps([dict(fact) for fact in error.facts], sort_keys=True, separators=(",", ":"))
    return (
        f"Observed values: {observed}\nFacts: {facts}\n"
        "Use only the observations and facts in this prompt. "
        "Recover the full world. Return exactly four comma-separated integers only."
    )


def _tokenizer(processor: object) -> object:
    return getattr(processor, "tokenizer", processor)


def _render_prefix(processor: object, prompt: str) -> str:
    tokenizer = _tokenizer(processor)
    template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(template):
        template = getattr(processor, "apply_chat_template", None)
    if not callable(template):
        raise RuntimeError("Phase 5 processor has no chat template")
    rendered = template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("Phase 5 chat template returned invalid text")
    return rendered


def _mapping(batch: object) -> dict[str, object]:
    if isinstance(batch, Mapping):
        return dict(batch)
    keys = getattr(batch, "keys", None)
    if callable(keys):
        return {key: batch[key] for key in keys()}  # type: ignore[index]
    raise TypeError("Phase 5 tokenized batch must be mapping-like")


def _decode(tokenizer: object, token_ids: object) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise RuntimeError("Phase 5 tokenizer has no decode()")
    text = decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if not isinstance(text, str):
        raise RuntimeError("Phase 5 tokenizer returned non-text output")
    return text


def generate_completion(
    model: object,
    processor: object,
    prompt: str,
    *,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed: int,
) -> tuple[str, tuple[int, ...]]:
    shortcut = getattr(model, "phase5_generate", None)
    if callable(shortcut):
        result = shortcut(
            prompt,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or not isinstance(result[0], str)
            or not isinstance(result[1], tuple)
            or any(type(item) is not int for item in result[1])
        ):
            raise RuntimeError("Phase 5 test generation shortcut returned malformed evidence")
        return result
    import torch

    tokenizer = _tokenizer(processor)
    rendered = _render_prefix(processor, prompt)
    if not callable(tokenizer):
        raise RuntimeError("Phase 5 tokenizer is not callable")
    batch = tokenizer([rendered], padding=True, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None and callable(getattr(batch, "to", None)):
        batch = batch.to(device)
    inputs = _mapping(batch)
    input_ids = inputs.get("input_ids")
    if input_ids is None or not hasattr(input_ids, "shape"):
        raise RuntimeError("Phase 5 tokenized prompt has no input_ids")
    prompt_length = int(input_ids.shape[-1])
    generate = getattr(model, "generate", None)
    if not callable(generate):
        raise RuntimeError("Phase 5 model has no generate()")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    options: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        options.update(temperature=temperature, top_p=top_p, top_k=top_k)
    with torch.inference_mode():
        output = generate(**inputs, **options)
    continuation = output[0, prompt_length:]
    ids = tuple(int(item) for item in continuation.detach().cpu().tolist())
    return _decode(tokenizer, continuation), ids


def completion_log_probability(
    model: object, processor: object, prompt: str, completion: str
) -> float:
    shortcut = getattr(model, "phase5_score", None)
    if callable(shortcut):
        score = shortcut(prompt, completion)
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise RuntimeError("Phase 5 test scoring shortcut returned an invalid score")
        return float(score)
    import torch

    tokenizer = _tokenizer(processor)
    rendered = _render_prefix(processor, prompt)
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise RuntimeError("Phase 5 tokenizer has no encode()")
    prefix_ids = tuple(int(item) for item in encode(rendered, add_special_tokens=False))
    completion_ids = tuple(int(item) for item in encode(completion, add_special_tokens=False))
    if not prefix_ids or not completion_ids:
        raise RuntimeError("Phase 5 candidate tokenization is empty")
    device = getattr(model, "device", None)
    input_ids = torch.tensor([prefix_ids + completion_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = getattr(outputs, "logits", None)
    if logits is None or logits.ndim != 3:
        raise RuntimeError("Phase 5 candidate forward returned invalid logits")
    log_probs = torch.log_softmax(logits[0], dim=-1)
    start = len(prefix_ids) - 1
    score = sum(
        float(log_probs[start + offset, token_id].item())
        for offset, token_id in enumerate(completion_ids)
    )
    if not math.isfinite(score):
        raise RuntimeError("Phase 5 candidate log probability is non-finite")
    return score


def _world_text(world: Sequence[int]) -> str:
    return ",".join(str(value) for value in world)


def measure_checkpoint(
    *,
    model: object,
    processor: object,
    checkpoint: PolicyCheckpoint,
    checkpoint_sha256: str,
    errors: Sequence[HeldOutNaturalError],
    config: Phase5MeasurementConfig,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[CheckpointSceneMeasurement, ...]:
    """Measure all scenes once, with no retries or result-dependent stopping."""

    if not errors or len({error.scene_id for error in errors}) != len(errors):
        raise ValueError("Phase 5 errors must be non-empty with unique scene identifiers")
    parameters = getattr(model, "parameters", None)
    if callable(parameters) and any(
        bool(getattr(parameter, "requires_grad", False)) for parameter in parameters()
    ):
        raise RuntimeError("Phase 5 model parameters must all be frozen")
    rows: list[CheckpointSceneMeasurement] = []
    for completed, error in enumerate(sorted(errors, key=lambda item: item.scene_id), start=1):
        prompt = recovery_prompt(error)
        greedy_raw, greedy_ids = generate_completion(
            model,
            processor,
            prompt,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=config.max_new_tokens,
            seed=config.sampling_seed,
        )
        greedy = parse_world(greedy_raw)
        truth_text, observed_text = _world_text(error.truth), _world_text(error.observed)
        logp_true = completion_log_probability(model, processor, prompt, truth_text)
        logp_observed = completion_log_probability(model, processor, prompt, observed_text)
        raw_outputs: list[str] = []
        token_ids: list[tuple[int, ...]] = []
        seeds: list[int] = []
        outputs: list[tuple[int, int, int, int] | None] = []
        for rollout_index in range(config.rollout_count):
            seed = phase5_rollout_seed(config.sampling_seed, error.scene_id, rollout_index)
            raw, ids = generate_completion(
                model,
                processor,
                prompt,
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                max_new_tokens=config.max_new_tokens,
                seed=seed,
            )
            raw_outputs.append(raw)
            token_ids.append(ids)
            seeds.append(seed)
            outputs.append(parse_world(raw))
        row = CheckpointSceneMeasurement(
            scene_id=error.scene_id,
            family=error.family,
            split=error.split,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            truth=error.truth,
            observed=error.observed,
            greedy_raw_output=greedy_raw,
            greedy_token_ids=greedy_ids,
            greedy_output=greedy,
            greedy_parse_success=greedy is not None,
            greedy_success=greedy == error.truth,
            greedy_observation_copy=greedy == error.observed,
            candidate_logp_true=logp_true,
            candidate_logp_observed=logp_observed,
            candidate_margin_true_observed=logp_true - logp_observed,
            sample_raw_outputs=tuple(raw_outputs),
            sample_token_ids=tuple(token_ids),
            sample_seeds=tuple(seeds),
            sample_outputs=tuple(outputs),
            sample_parse_success=tuple(output is not None for output in outputs),
            sample_success=tuple(output == error.truth for output in outputs),
            sample_observation_copy=tuple(output == error.observed for output in outputs),
        )
        rows.append(row)
        if progress is not None:
            progress(completed, len(errors))
    return tuple(rows)


def freeze_inference_model(model: object) -> None:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise RuntimeError("Phase 5 model has no parameters()")
    for parameter in parameters():
        requires_grad = getattr(parameter, "requires_grad_", None)
        if not callable(requires_grad):
            raise RuntimeError("Phase 5 model parameter cannot be frozen")
        requires_grad(False)
    eval_mode = getattr(model, "eval", None)
    if not callable(eval_mode):
        raise RuntimeError("Phase 5 model has no eval()")
    eval_mode()
    if any(bool(getattr(parameter, "requires_grad", False)) for parameter in parameters()):
        raise RuntimeError("Phase 5 could not freeze all model parameters")


__all__ = [
    "Phase5MeasurementConfig",
    "checkpoint_adapter_path",
    "checkpoint_tree_hashes",
    "completion_log_probability",
    "freeze_inference_model",
    "generate_completion",
    "load_phase5_config",
    "measure_checkpoint",
    "phase5_rollout_seed",
    "recovery_prompt",
    "tree_sha256",
    "verify_phase5_package_lock",
]
