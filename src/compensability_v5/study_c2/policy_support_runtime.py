"""Foreground, resumable B3 support measurement at the first GPU boundary."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from compensability_v4.qwen.model_loader import load_pinned_qwen, require_server_model
from compensability_v4.qwen.phase5_runtime import phase5_rollout_seed
from compensability_v5.qwen.study_b_runtime import tree_sha256

from .io import read_jsonl, sha256_file, write_json_new
from .paths import FIBER_ROWS, SUPPORT_MANIFEST, SUPPORT_RAW_ROWS, SUPPORT_ROOT, SUPPORT_SUMMARY
from .rewards import classify_completion
from .stages import load_contract
from .support import summarize_policy_support

SUPPORT_ACK = "I_UNDERSTAND_THIS_RUNS_STUDY_C2_FROZEN_B3_SUPPORT"


def _require_offline_cuda() -> None:  # pragma: no cover - server runtime
    required = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    missing = [name for name in required if os.environ.get(name) != "1"]
    if missing:
        raise RuntimeError("offline environment is incomplete: " + ", ".join(missing))
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Study C2 support requires exactly one visible CUDA GPU")
    if "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"Study C2 support requires a 4090, got {torch.cuda.get_device_name(0)}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Study C2 support requires bf16 CUDA support")


def preflight_support(
    *, config_path: Path, b3_adapter: Path, b3_sha256: str
) -> dict[str, object]:  # pragma: no cover - server runtime
    print("PROGRESS: validating the closed Study C2 config", flush=True)
    contract = load_contract(config_path)
    print("PROGRESS: verifying offline mode and the single bf16 4090", flush=True)
    _require_offline_cuda()
    print("PROGRESS: hashing the immutable local Qwen snapshot", flush=True)
    require_server_model()
    print("PROGRESS: hashing the immutable Study B B3 adapter", flush=True)
    observed = tree_sha256(b3_adapter)
    if observed != b3_sha256:
        raise RuntimeError(
            f"B3 adapter SHA-256 mismatch: expected {b3_sha256}, observed {observed}"
        )
    rows = read_jsonl(FIBER_ROWS)
    print("PROGRESS: verifying the frozen 96-prompt support split", flush=True)
    support_rows = [row for row in rows if row.get("split") == "support_audit"]
    if len(support_rows) != 96:
        raise RuntimeError(f"Study C2 support requires 96 prompts, observed {len(support_rows)}")
    return {
        "schema_version": 2,
        "status": "STUDY_C2_FROZEN_SUPPORT_PREFLIGHT_OK",
        "prompt_count": len(support_rows),
        "rollouts_per_prompt": contract["support_rollouts_per_prompt"],
        "b3_adapter_sha256": observed,
        "fiber_rows_sha256": sha256_file(FIBER_ROWS),
        "config_sha256": sha256_file(config_path),
        "gpu_invoked": False,
    }


def _tokenizer(processor: object) -> object:
    return getattr(processor, "tokenizer", processor)


def _newline_eos_ids(tokenizer: object) -> list[int]:  # pragma: no cover - server runtime
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise RuntimeError("Qwen tokenizer does not expose encode()")
    newline = encode("\n", add_special_tokens=False)
    if not isinstance(newline, list) or len(newline) != 1 or type(newline[0]) is not int:
        raise RuntimeError(f"newline is not one tokenizer token: {newline!r}")
    eos = getattr(tokenizer, "eos_token_id", None)
    if type(eos) is not int:
        raise RuntimeError("Qwen tokenizer lacks an integer EOS token")
    return list(dict.fromkeys((eos, newline[0])))


def _generate_first_line(
    model: object,
    processor: object,
    prompt: str,
    *,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, list[int]]:  # pragma: no cover - server runtime
    import torch

    tokenizer = _tokenizer(processor)
    render = getattr(tokenizer, "apply_chat_template", None)
    if not callable(render) or not callable(tokenizer):
        raise RuntimeError("Qwen tokenizer lacks the required chat/call surface")
    rendered = render(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    batch = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    device = getattr(model, "device", "cuda:0")
    inputs = {name: value.to(device) for name, value in batch.items()}
    prompt_length = int(inputs["input_ids"].shape[-1])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            eos_token_id=_newline_eos_ids(tokenizer),
            use_cache=True,
        )
    continuation = generated[0, prompt_length:]
    ids = [int(item) for item in continuation.detach().cpu().tolist()]
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise RuntimeError("Qwen tokenizer lacks decode()")
    text = decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if not isinstance(text, str):
        raise RuntimeError("Qwen tokenizer returned non-text completion")
    return text, ids


def _validate_partial_rows(
    *,
    completed: list[dict[str, object]],
    prompts: tuple[dict[str, object], ...],
    rollouts_per_prompt: int,
    seed: int,
) -> None:
    expected_total = len(prompts) * rollouts_per_prompt
    if len(completed) > expected_total:
        raise RuntimeError("partial Study C2 support trace exceeds registered rollout count")
    for absolute_index, row in enumerate(completed):
        prompt_index, rollout_index = divmod(absolute_index, rollouts_per_prompt)
        prompt = prompts[prompt_index]
        expected_seed = phase5_rollout_seed(seed, str(prompt["scene_id"]), rollout_index)
        if (
            row.get("scene_id") != prompt["scene_id"]
            or row.get("pair_id") != prompt["pair_id"]
            or row.get("rollout_index") != rollout_index
            or row.get("seed") != expected_seed
            or row.get("kind") not in {"X", "S", "F", "U"}
        ):
            raise RuntimeError(
                f"partial Study C2 support trace drifted at rollout {absolute_index}"
            )


def run_frozen_policy_support(
    *,
    config_path: Path,
    b3_adapter: Path,
    b3_sha256: str,
    acknowledgement: str,
) -> dict[str, object]:  # pragma: no cover - server runtime
    if acknowledgement != SUPPORT_ACK:
        raise PermissionError("exact Study C2 support acknowledgement is required")
    preflight = preflight_support(
        config_path=config_path, b3_adapter=b3_adapter, b3_sha256=b3_sha256
    )
    if SUPPORT_RAW_ROWS.exists() or SUPPORT_SUMMARY.exists() or SUPPORT_MANIFEST.exists():
        raise RuntimeError("completed Study C2 support output already exists; overwrite forbidden")
    contract = load_contract(config_path)
    all_rows = read_jsonl(FIBER_ROWS)
    prompts = tuple(row for row in all_rows if row.get("split") == "support_audit")
    partial = SUPPORT_ROOT / "raw_rows.partial.jsonl"
    SUPPORT_ROOT.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, object]] = []
    if partial.exists() and partial.stat().st_size:
        completed = list(read_jsonl(partial))
    rollouts_per_prompt = int(contract["support_rollouts_per_prompt"])
    expected_total = len(prompts) * rollouts_per_prompt
    _validate_partial_rows(
        completed=completed,
        prompts=prompts,
        rollouts_per_prompt=rollouts_per_prompt,
        seed=int(contract["seed"]),
    )
    print("PROGRESS: loading verified Qwen snapshot and immutable B3 adapter", flush=True)
    base, processor = load_pinned_qwen()
    from peft import PeftModel

    model = PeftModel.from_pretrained(base, str(b3_adapter), is_trainable=False)
    model.eval()
    start = len(completed)
    with partial.open("a", encoding="utf-8") as stream:
        for absolute_index in range(start, expected_total):
            prompt_index, rollout_index = divmod(absolute_index, rollouts_per_prompt)
            row = prompts[prompt_index]
            seed = phase5_rollout_seed(int(contract["seed"]), str(row["scene_id"]), rollout_index)
            completion, token_ids = _generate_first_line(
                model,
                processor,
                str(row["prompt"]),
                seed=seed,
                max_new_tokens=int(contract["training"]["max_completion_length"]),  # type: ignore[index]
                temperature=float(contract["training"]["temperature"]),  # type: ignore[index]
                top_p=float(contract["training"]["top_p"]),  # type: ignore[index]
            )
            truth = tuple(int(value) for value in row["truth"])  # type: ignore[union-attr]
            operation = row["operation"]
            if not isinstance(operation, Mapping):
                raise RuntimeError("Study C2 operation is malformed")
            label = classify_completion(completion, truth=truth, operation=operation)
            result = {
                "schema_version": 2,
                "scene_id": row["scene_id"],
                "pair_id": row["pair_id"],
                "condition": row["condition"],
                "family": row["family"],
                "truth": row["truth"],
                "observation": row["observation"],
                "operation": row["operation"],
                "prompt_sha256": row["prompt_sha256"],
                "full_reward_fiber_size": row["full_reward_fiber_size"],
                "one_edit_reward_fiber_size": row["one_edit_reward_fiber_size"],
                "observed_is_answer_equivalent": (row["observed_answer"] == row["gold_answer"]),
                "rollout_index": rollout_index,
                "seed": seed,
                "completion": completion,
                "token_ids": token_ids,
                "parsed_world": list(label.parsed_world) if label.parsed_world else None,
                "kind": label.kind.value,
                "answer_reward": label.answer_reward,
                "state_reward": label.state_reward,
            }
            stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            completed.append(result)
            print(
                f"PROGRESS: support rollout {absolute_index + 1}/{expected_total} "
                f"scene={row['scene_id']} sample={rollout_index + 1}/"
                f"{rollouts_per_prompt} kind={label.kind.value}",
                flush=True,
            )
    partial.replace(SUPPORT_RAW_ROWS)
    summary = summarize_policy_support(
        completed,
        group_candidates=tuple(int(k) for k in contract["group_candidates"]),  # type: ignore[arg-type]
    )
    summary = {
        "schema_version": 2,
        **summary,
        "status": summary["status"],
        "gpu_invoked": True,
    }
    write_json_new(SUPPORT_SUMMARY, summary)
    manifest: dict[str, object] = {
        **preflight,
        "status": "STUDY_C2_FROZEN_SUPPORT_COMPLETE",
        "raw_rows_sha256": sha256_file(SUPPORT_RAW_ROWS),
        "summary_sha256": sha256_file(SUPPORT_SUMMARY),
        "rollout_count": len(completed),
        "action_protocol": "anchored_first_line_world_v1",
        "stopping_rule": "newline_or_eos_with_max_16_tokens",
        "training_invoked": False,
        "rl_invoked": False,
        "gpu_invoked": True,
    }
    write_json_new(SUPPORT_MANIFEST, manifest)
    return manifest


__all__ = ["SUPPORT_ACK", "preflight_support", "run_frozen_policy_support"]
