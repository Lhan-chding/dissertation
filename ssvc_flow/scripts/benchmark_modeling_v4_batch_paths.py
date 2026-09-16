#!/usr/bin/env python3
"""Bounded, separate technical probe; never enables a production batch path.

Only identical prepared prompts are batched. Generation uses the inherited
uncached distribution settings and a *different*, explicitly recorded batch RNG
stream. Timing or token parity here does not certify a production sampler.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class BudgetExpired(RuntimeError):
    pass


def batch_inputs(adapter, prepared, *, batch_size, completions=None):
    """Explicit supported Qwen image/text fields; no guessed tensor axes."""
    import torch

    original = adapter._device_inputs(prepared)
    allowed = {"input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw"}
    if set(original) - allowed or not {"input_ids", "attention_mask"} <= original.keys():
        raise ValueError(
            "Unsupported prepared keys; videos/position overrides are not batch-certified"
        )
    if batch_size not in (1, 2, 4, 8):
        raise ValueError("Registered microbatches are 1,2,4,8")
    ids = original["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or not len(ids[0]):
        raise ValueError("Exactly one nonempty original prompt required")
    for key in ("attention_mask", "mm_token_type_ids"):
        if key in original and original[key].shape != ids.shape:
            raise ValueError("Prompt token fields must have identical axes")
    if not torch.all(original["attention_mask"] == 1):
        raise ValueError("Benchmark only accepts unpadded original prompts")
    result = {
        key: original[key].repeat(batch_size, 1)
        for key in ("input_ids", "attention_mask", "mm_token_type_ids")
        if key in original
    }
    visual = {"pixel_values", "image_grid_thw"} & original.keys()
    if visual and visual != {"pixel_values", "image_grid_thw"}:
        raise ValueError("Visual pixels and grid must be provided together")
    if visual:
        pixels, grid = original["pixel_values"], original["image_grid_thw"]
        if (
            pixels.ndim != 2
            or grid.ndim != 2
            or grid.shape[1] != 3
            or grid.prod(-1).sum().item() != pixels.shape[0]
        ):
            raise ValueError("Unsupported visual patch/grid layout")
        result["pixel_values"] = pixels.repeat(batch_size, 1)
        result["image_grid_thw"] = grid.repeat(batch_size, 1)
    if completions is not None:
        if len(completions) != batch_size or not all(completions):
            raise ValueError("One nonempty completion per batch row required")
        width = max(map(len, completions))
        tail = ids.new_full((batch_size, width), adapter.pad_id)
        mask = original["attention_mask"].new_zeros((batch_size, width))
        for i, tokens in enumerate(completions):
            tail[i, : len(tokens)] = torch.tensor(tokens, dtype=ids.dtype, device=ids.device)
            mask[i, : len(tokens)] = 1
        result["input_ids"] = torch.cat((result["input_ids"], tail), -1)
        result["attention_mask"] = torch.cat((result["attention_mask"], mask), -1)
        if "mm_token_type_ids" in result:
            result["mm_token_type_ids"] = torch.cat(
                (result["mm_token_type_ids"], torch.zeros_like(tail)), -1
            )
    return result


def teacher_scores(adapter, prepared, completions):
    import torch

    inputs = batch_inputs(adapter, prepared, batch_size=len(completions), completions=completions)
    adapter.model.eval()
    adapter._reset_positions()
    with torch.no_grad():
        output = adapter.model(**inputs, use_cache=False)
        start = prepared["audit"]["prompt_token_count"] - 1
        if output.logits.shape[:2] != inputs["input_ids"].shape:
            raise ValueError("Teacher logits must retain the full sequence/batch axes")
        values = []
        for i, tokens in enumerate(completions):
            logits = output.logits[i, start : start + len(tokens)]
            target = torch.tensor(tokens, dtype=torch.long, device=logits.device)
            values.append(
                logits.float()
                .log_softmax(-1)
                .gather(-1, target[:, None])[:, 0]
                .double()
                .cpu()
                .tolist()
            )
    return values


def unpack_generation(result, *, prompt_length, batch_size, eos_ids, pad_id, horizon):
    """Preserve every returned token, with an explicit EOS-inclusive action mask."""
    rows = []
    tails = result.sequences[:, prompt_length:].tolist()
    if len(tails) != batch_size or len(result.scores) != len(tails[0]):
        raise ValueError("Generation score/time axes differ from returned tokens")
    for i, padded in enumerate(tails):
        stop = next((j + 1 for j, token in enumerate(padded) if token in eos_ids), len(padded))
        tokens = padded[:stop]
        if not tokens or (tokens[-1] not in eos_ids and len(tokens) != horizon):
            raise ValueError("Generated action must end at EOS or the frozen horizon")
        if any(token != pad_id for token in padded[stop:]):
            raise ValueError("Generated non-padding tokens remain after EOS")
        values = [
            float(result.scores[j][i].detach().float().log_softmax(-1)[token])
            for j, token in enumerate(tokens)
        ]
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Nonfinite batch generation probability")
        rows.append(
            {
                "token_ids": tokens,
                "returned_token_ids": padded,
                "action_mask": [True] * stop + [False] * (len(padded) - stop),
                "behavior_token_logprobs": values,
                "eos_seen": tokens[-1] in eos_ids,
                "stop_reason": "eos" if tokens[-1] in eos_ids else "length",
                "max_new_tokens": horizon,
            }
        )
    return rows


def batch_generate(adapter, prepared, *, batch_size, seed, horizon=64):
    import torch
    from transformers import GenerationConfig

    from src.model_adapters.base import pure_generation_options

    inputs = batch_inputs(adapter, prepared, batch_size=batch_size)
    adapter.model.eval()
    adapter._reset_positions()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    settings = GenerationConfig(
        **pure_generation_options(horizon),
        eos_token_id=sorted(adapter.eos_ids),
        pad_token_id=adapter.pad_id,
        bos_token_id=adapter.processor.tokenizer.bos_token_id,
    )
    with torch.no_grad():
        result = adapter.model.generate(**inputs, generation_config=settings)
    return unpack_generation(
        result,
        prompt_length=inputs["input_ids"].shape[1],
        batch_size=batch_size,
        eos_ids=adapter.eos_ids,
        pad_id=adapter.pad_id,
        horizon=horizon,
    )


def parity(actual, expected, tolerances):
    if len(actual) != len(expected) or any(
        len(a) != len(b) for a, b in zip(actual, expected, strict=True)
    ):
        raise ValueError("Same-token probability comparison must preserve every EOS/mask position")
    delta = [
        abs(a - b)
        for row, ref in zip(actual, expected, strict=True)
        for a, b in zip(row, ref, strict=True)
    ]
    sequence = [abs(math.fsum(a) - math.fsum(b)) for a, b in zip(actual, expected, strict=True)]
    if not delta or not all(math.isfinite(x) for x in delta + sequence):
        raise ValueError("Finite nonempty probability paths required")
    errors = {
        "mean_abs_token_logp": math.fsum(delta) / len(delta),
        "max_abs_token_logp": max(delta),
        "max_abs_sequence_logp": max(sequence),
    }
    return {
        "passed": all(errors[k] <= tolerances[k] for k in errors),
        "errors": errors,
        "tolerances": tolerances,
        "tokens": len(delta),
        "sequences": len(actual),
        "per_sequence_max_abs_token_logp": [
            max(abs(x - y) for x, y in zip(a, b, strict=True))
            for a, b in zip(actual, expected, strict=True)
        ],
        "per_sequence_abs_sequence_logp": sequence,
    }


def timed(adapter, function):
    import torch

    cuda = str(adapter.device).startswith("cuda")
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    before = adapter.forward_calls
    start = time.perf_counter()
    value = function()
    if cuda:
        torch.cuda.synchronize()
    return value, {
        "seconds": time.perf_counter() - start,
        "model_forward_calls": adapter.forward_calls - before,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
        "includes_input_assembly_and_output_transfer": True,
    }


def run_profile(
    runtime, prepared, samples, *, out, deadline, generation_prompts, generation_draws=8
):
    """No runtime loading here: also exercised by actual tiny torch fixtures."""
    import torch

    from src.modeling_v3.io import atomic_json, canonical_hash
    from src.modeling_v4.gpu_collect import _binding
    from src.optimizer_fork import state_hash

    root = Path(out)
    adapter = runtime["adapter"]
    tolerances = runtime["parity_tolerances"]
    original = runtime["state"].capture({"purpose": "batch_performance_probe_restore"})
    original_hash = state_hash(original)
    adapter.model.eval()
    immutable = runtime["state_guard"]()
    cases = []
    active_partial = {}

    def check():
        if time.perf_counter() >= deadline:
            raise BudgetExpired("Bounded benchmark time expired; no further model calls")

    def case(name, function):
        check()
        started = time.perf_counter()
        forward_start = adapter.forward_calls
        active_partial.clear()
        try:
            value = {"status": "MEASURED_TECHNICAL_ONLY", **function()}
        except (RuntimeError, ValueError, TypeError, AttributeError, NotImplementedError) as exc:
            value = {
                "status": "TIME_BUDGET_EXHAUSTED"
                if isinstance(exc, BudgetExpired)
                else "OOM"
                if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
                else "UNSUPPORTED_OR_FAILED_PATH",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "partial_observations": active_partial.copy(),
                "returned_sequences": len(active_partial.get("records", [])),
            }
            gc.collect()
            adapter._reset_positions()
            if str(adapter.device).startswith("cuda"):
                torch.cuda.empty_cache()
        if runtime["state_guard"]() != immutable:
            raise RuntimeError(
                "Benchmark changed model parameters, optimizer or persistent buffers"
            )
        value = {
            "name": name,
            "total_case_seconds": time.perf_counter() - started,
            "total_case_model_forward_calls": adapter.forward_calls - forward_start,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated()
            if str(adapter.device).startswith("cuda")
            else None,
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()
            if str(adapter.device).startswith("cuda")
            else None,
            **value,
        }
        for field in ("peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes"):
            measured = [
                entry[field]
                for key, entries in active_partial.items()
                if key.endswith("measurements")
                for entry in entries
                if entry.get(field) is not None
            ]
            if value[field] is not None:
                measured.append(value[field])
            value[field] = max(measured) if measured else None
        timings = value.get("generation_measurements", value.get("measurements"))
        if timings is not None:
            seconds = math.fsum(t["seconds"] for t in timings)
            value["throughput"] = {
                "measured_seconds": seconds,
                "sequences_per_second": value["sequences"] / seconds if seconds else None,
                "tokens_per_second": value["tokens"] / seconds if seconds else None,
                "scope": "timed batch calls including input assembly and output transfer",
                "generation_parity_rescoring_excluded": name.startswith("generation_"),
                "whole_campaign_eta_claimed": False,
            }
        atomic_json(root / f"{name}.json", value)
        cases.append(
            {"name": name, "status": value["status"], "artifact": _binding(root / f"{name}.json")}
        )
        if value["status"] == "TIME_BUDGET_EXHAUSTED":
            raise BudgetExpired("Bounded profile stopped after preserving partial case")
        return value

    groups = {}
    for sample in samples:
        groups.setdefault(sample["prompt_id"], []).append(sample)
    reference = {}
    serial_teacher = {}
    status = "MEASURED_TECHNICAL_ONLY"
    try:

        def serial_reference():
            records, metrics = [], []
            active_partial.update(records=records, measurements=metrics)
            for sample in samples:
                check()
                values, timing = timed(
                    adapter,
                    lambda sample=sample: (
                        adapter.logprobs(
                            prepared[sample["prompt_id"]], sample["token_ids"], require_grad=False
                        )
                        .detach()
                        .double()
                        .cpu()
                        .tolist()
                    ),
                )
                reference[sample["sample_key"]] = values
                records.append(
                    {
                        "sample_key": sample["sample_key"],
                        "token_ids": sample["token_ids"],
                        "token_logprobs": values,
                    }
                )
                metrics.append(timing)
            return {
                "sequences": len(records),
                "tokens": sum(len(r["token_ids"]) for r in records),
                "measurements": metrics,
                "records": records,
                "original_behavior_parity": parity(
                    [r["token_logprobs"] for r in records],
                    [r["behavior_token_logprobs"] for r in samples],
                    tolerances,
                ),
            }

        serial = case("serial_prefix_reference", serial_reference)
        if (
            serial["status"] != "MEASURED_TECHNICAL_ONLY"
            or not serial["original_behavior_parity"]["passed"]
        ):
            raise ValueError(
                "The exact historical-origin serial prefix failed; comparisons cannot proceed"
            )
        for microbatch in (1, 2, 4, 8):

            def score_path(microbatch=microbatch):
                records, metrics = [], []
                active_partial.update(records=records, measurements=metrics)
                for pid, rows in groups.items():
                    for start in range(0, len(rows), microbatch):
                        check()
                        batch = rows[start : start + microbatch]
                        actual, metric = timed(
                            adapter,
                            lambda pid=pid, batch=batch: teacher_scores(
                                adapter, prepared[pid], [s["token_ids"] for s in batch]
                            ),
                        )
                        metrics.append(metric)
                        records.extend(
                            {"sample_key": s["sample_key"], "token_logprobs": value}
                            for s, value in zip(batch, actual, strict=True)
                        )
                comparison = parity(
                    [r["token_logprobs"] for r in records],
                    [reference[r["sample_key"]] for r in records],
                    tolerances,
                )
                if microbatch == 1:
                    serial_teacher.update({r["sample_key"]: r["token_logprobs"] for r in records})
                teacher_comparison = (
                    parity(
                        [r["token_logprobs"] for r in records],
                        [serial_teacher[r["sample_key"]] for r in records],
                        tolerances,
                    )
                    if len(serial_teacher) == len(samples)
                    else None
                )
                return {
                    "microbatch": microbatch,
                    "sequences": len(records),
                    "tokens": sum(len(r["token_logprobs"]) for r in records),
                    "measurements": metrics,
                    "records": records,
                    "prefix_parity": comparison,
                    "serial_teacher_parity": teacher_comparison,
                    "production_path_enabled": False,
                }

            case(f"teacher_batch_{microbatch}", score_path)

            def generate_path(microbatch=microbatch):
                records, generation_metrics, rescore_metrics = [], [], []
                active_partial.update(
                    records=records,
                    generation_measurements=generation_metrics,
                    parity_rescore_measurements=rescore_metrics,
                )
                for pid in generation_prompts:
                    for start in range(0, generation_draws, microbatch):
                        check()
                        seed = int(
                            canonical_hash(["V4_TECHNICAL_BATCH_ONLY", pid, microbatch, start])[
                                :16
                            ],
                            16,
                        ) % (2**63)
                        generated, timing = timed(
                            adapter,
                            lambda pid=pid, seed=seed: batch_generate(
                                adapter, prepared[pid], batch_size=microbatch, seed=seed
                            ),
                        )
                        generation_metrics.append(timing)
                        returned_batch = [
                            {
                                **row,
                                "prompt_id": pid,
                                "batch_seed": seed,
                                "batch_row": index,
                                "prefix_score_status": "NOT_RUN",
                            }
                            for index, row in enumerate(generated)
                        ]
                        records.extend(returned_batch)
                        for retained in returned_batch:
                            check()
                            prefix, cost = timed(
                                adapter,
                                lambda row=retained, pid=pid: (
                                    adapter.logprobs(
                                        prepared[pid], row["token_ids"], require_grad=False
                                    )
                                    .detach()
                                    .double()
                                    .cpu()
                                    .tolist()
                                ),
                            )
                            rescore_metrics.append(cost)
                            retained["prefix_token_logprobs"] = prefix
                            retained["prefix_score_status"] = "MEASURED"
                comparison = parity(
                    [r["behavior_token_logprobs"] for r in records],
                    [r["prefix_token_logprobs"] for r in records],
                    tolerances,
                )
                return {
                    "microbatch": microbatch,
                    "sequences": len(records),
                    "tokens": sum(len(r["token_ids"]) for r in records),
                    "generation_measurements": generation_metrics,
                    "parity_rescore_measurements": rescore_metrics,
                    "records": records,
                    "prefix_parity": comparison,
                    "rng_contract": "batch-global RNG; not the production per-draw seed mapping",
                    "independent_scientific_samples_claimed": False,
                    "production_path_enabled": False,
                }

            case(f"generation_batch_{microbatch}", generate_path)
    except BudgetExpired:
        status = "TIME_BUDGET_EXHAUSTED_PARTIAL_TECHNICAL_ONLY"
    except BaseException:
        status = "FAILED_TECHNICAL_ONLY"
        raise
    finally:
        runtime["state"].restore(original)
        # Restoration legitimately increments tensor versions. Verify complete
        # trainable/Adam/RNG state content once, not pre-restore version counters.
        if state_hash(runtime["state"].capture(original.get("metadata", {}))) != original_hash:
            raise RuntimeError("Full original state restore failed")
        atomic_json(
            root / "PROFILE.json",
            {
                "status": status,
                "cases": cases,
                "state_restored": True,
                "predictive_or_statistical_claim": False,
                "production_path_enabled": False,
                "execution_kind": runtime["identity"]["execution_kind"],
            },
        )
    return {"status": status, "cases": cases}


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--tasks", required=True)
    value.add_argument(
        "--samples", required=True, help="Completed first-bank samples/COMPLETE.json (32 rows)"
    )
    value.add_argument("--origin-index", required=True, type=int, choices=(0, 1))
    value.add_argument("--out", required=True)
    value.add_argument("--max-seconds", type=int, default=3000)
    value.add_argument("--execute-gpu", action="store_true")
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if not 60 <= args.max_seconds <= 3300:
        raise ValueError("Bounded probe must reserve Slurm cleanup time: 60..3300 seconds")
    if not args.execute_gpu:
        return {"status": "DRY_RUN_NO_MODEL", "microbatches": [1, 2, 4, 8], "args": vars(args)}
    if platform.system() != "Linux" or not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("Execute only in a one-GPU Linux Slurm allocation")
    import torch

    from src.modeling_v3.io import atomic_json, canonical_hash, source_identity
    from src.modeling_v4 import gpu_collect as gpu

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU required")
    deadline = time.perf_counter() + args.max_seconds
    tasks = gpu._read(args.tasks)
    if (
        tasks["task_list_hash"]
        != canonical_hash({k: v for k, v in tasks.items() if k != "task_list_hash"})
        or tasks["source"] != source_identity()
    ):
        raise ValueError("Frozen campaign task/source identity differs")
    receipt = gpu._read(args.samples)
    gpu.verify_artifact_bindings(receipt)
    samples = list(gpu._rows(receipt))
    if (
        receipt["count"] != 32
        or len(samples) != 32
        or len({s["sample_key"] for s in samples}) != 32
        or len({s["prompt_id"] for s in samples}) != 4
        or any(
            sum(s["prompt_id"] == p for s in samples) != 8
            for p in {s["prompt_id"] for s in samples}
        )
    ):
        raise ValueError("Exactly 32 committed original first-bank draws required")
    by_id = {p["prompt_id"]: p for p in tasks["inputs"]["train_prompts"]}
    if (
        receipt["identity"]["role"] != "train"
        or receipt["identity"]["proposal"] != "ORIGIN"
        or any(s["prompt_id"] not in by_id for s in samples)
    ):
        raise ValueError("Expected an original B natural training bank")
    root = Path(args.out).resolve()
    if root == Path(tasks["root"]).resolve() or root.is_relative_to(Path(tasks["root"]).resolve()):
        raise ValueError("Benchmark output must be outside the production campaign")
    root.mkdir(parents=True, exist_ok=False)
    binding = {
        "kind": "V4_TECHNICAL_BATCH_PROBE",
        "tasks": gpu._binding(args.tasks),
        "samples": gpu._binding(args.samples),
        "script": gpu._binding(__file__),
        "source": tasks["source"],
        "origin": tasks["inputs"]["bridge_origins"][args.origin_index],
        "max_seconds": args.max_seconds,
        "microbatches": [1, 2, 4, 8],
        "scientific_samples": False,
        "generation_draws_per_interface_per_path": 8,
        "probability_tolerances": tasks["bindings"]["v3_probability_tolerances"],
    }
    atomic_json(root / "INPUT.json", binding)
    try:
        return _execute_bound(args, tasks, receipt, samples, by_id, root, binding, deadline)
    except BaseException as exc:
        atomic_json(
            root / "FAILED.json",
            {
                "status": "FAILED_TECHNICAL_ONLY",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "runtime_receipt_exists": (root / "RUNTIME.json").exists(),
                "profile_receipt_exists": (root / "PROFILE.json").exists(),
                "production_path_enabled": False,
            },
        )
        raise


def _execute_bound(args, tasks, receipt, samples, by_id, root, binding, deadline):
    from src.modeling_v3.io import atomic_json, canonical_hash
    from src.modeling_v4 import gpu_collect as gpu

    started = time.perf_counter()
    runtime = gpu.load_runtime(tasks["config"], tasks["bindings"], execute_gpu=True)
    load_seconds = time.perf_counter() - started
    if canonical_hash(receipt["identity"]["runtime"]) != canonical_hash(runtime["identity"]):
        raise ValueError("Sample runtime/config/model binding differs from the measured runtime")
    origin = gpu._bridge_legacy_checkpoint(runtime, binding["origin"])
    runtime["state"].restore(origin)
    fingerprint = gpu._fingerprint(origin, runtime["identity"])
    if any(s["proposal_fingerprint"] != fingerprint for s in samples):
        raise ValueError("Samples were generated from a different actual policy")
    adapter = runtime["adapter"]
    adapter.model.eval()
    adapter._reset_positions()
    prepared = {
        pid: adapter.prepare(
            by_id[pid]["prompt"], by_id[pid].get("data_root") or runtime["data_root"]
        )
        for pid in dict.fromkeys(s["prompt_id"] for s in samples)
    }
    for sample in samples:
        p = prepared[sample["prompt_id"]]
        if p["audit"]["input_tensor_hash"] != sample["input_hash"] or sample["action_mask"] != [
            True
        ] * len(sample["token_ids"]):
            raise ValueError("Original prepared input or token mask changed")
        from src.modeling_v4.score_response import _active_tokens

        _active_tokens(
            sample["token_ids"], adapter.eos_ids, sample["max_new_tokens"], sample["action_mask"]
        )
    selected = {}
    for pid in prepared:
        selected.setdefault(by_id[pid]["interface"], pid)
    atomic_json(
        root / "RUNTIME.json",
        {
            "identity": runtime["identity"],
            "allocated_gpu": runtime["allocated_gpu"],
            "model_load_seconds": load_seconds,
            "generation_prompt_ids": list(selected.values()),
            "input_audits": {pid: p["audit"] for pid, p in prepared.items()},
            "model_dtype": str(next(adapter.model.parameters()).dtype),
        },
    )
    result = run_profile(
        runtime,
        prepared,
        samples,
        out=root,
        deadline=deadline,
        generation_prompts=list(selected.values()),
    )
    atomic_json(
        root / "COMPLETE.json",
        {
            "input": gpu._binding(root / "INPUT.json"),
            "runtime": gpu._binding(root / "RUNTIME.json"),
            "profile": gpu._binding(root / "PROFILE.json"),
            "status": result["status"],
        },
    )
    return result


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
