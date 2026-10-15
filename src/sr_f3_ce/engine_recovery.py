"""Bounded real-model ENGINE probes and a new-process 4 versus 2+2 replay.

Six durable Adam intents are the upper bound, including an interrupted intent
whose outcome is unknown. Gradient probes never request an Adam intent. This
module does not select a scientific recipe from model scores.
"""

from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from mm_core.training import _cpu_tree, capture_rng, restore_rng, state_hash, trainable_state
from sr_f1.training import atomic_torch
from sr_f12.evaluation import durable_json, shard_lease

from .checkpoint import compare_recovery_states, read_checkpoint
from .protocol import digest, file_hash, now, read_json, verify_run
from .training import probe_gradients

NUMERICAL_BOUND = {"absolute": 2e-6, "relative": 2e-5}
PHYSICAL_SEQUENCE = (
    ("continuous", 1),
    ("continuous", 2),
    ("continuous", 3),
    ("continuous", 4),
    ("split", 3),
    ("split", 4),
)


def _engine_root(root):
    return Path(root).resolve() / "engine/resume"


def _intent_rows(root):
    directory = _engine_root(root) / "optimizer_intents"
    paths = sorted(directory.glob("*.json"))
    if len(paths) > 6:
        raise PermissionError("ENGINE exceeded six physical Adam reservations")
    rows = []
    for index, path in enumerate(paths, 1):
        row = read_json(path)
        if path.name != f"{index:02d}.json" or row["index"] != index:
            raise PermissionError("ENGINE Adam intent inventory has a gap or changed ordinal")
        if (row["segment"], row["step"]) != PHYSICAL_SEQUENCE[index - 1]:
            raise PermissionError("ENGINE Adam intents differ from continuous 4 plus replay 2")
        if row["arm"] != "EXEC" or row["microbatch_size"] not in (4, 2, 1):
            raise PermissionError("ENGINE Adam intent recipe changed")
        rows.append(row)
    return rows


def reserve_engine_step(root, metadata):
    """Persist an intent before Adam; unknown outcomes remain budget-consuming."""
    from .orchestration import worker_binding

    binding = worker_binding(root, "engine--resume")
    directory = _engine_root(root) / "optimizer_intents"
    with shard_lease(directory):
        previous = _intent_rows(root)
        ordinal = len(previous)
        if ordinal == 6:
            raise PermissionError("Six physical ENGINE Adam intents already consumed")
        if (metadata.get("segment"), metadata.get("step")) != PHYSICAL_SEQUENCE[ordinal]:
            raise PermissionError(
                "Unresolved/repeated ENGINE Adam intent: no seventh or repeated update"
            )
        if metadata.get("arm") != "EXEC" or metadata.get("microbatch_size") not in (4, 2, 1):
            raise PermissionError("ENGINE requires the registered EXEC update")
        receipt = dict(
            metadata,
            index=ordinal + 1,
            pid=os.getpid(),
            at=now(),
            **binding,
            status="RESERVED_BEFORE_ADAM",
            unknown_outcome_counts_against_budget=True,
        )
        durable_json(directory / f"{ordinal + 1:02d}.json", receipt)
        return receipt


def _live_state(runtime, optimizer, scheduler, reference):
    return state_hash(
        dict(
            parameters=trainable_state(runtime.model),
            optimizer=_cpu_tree(optimizer.state_dict()),
            scheduler=scheduler.state_dict(),
            rng=capture_rng(),
            reference=reference,
            train_mode=runtime.model.training,
        )
    )


def _gradient_comparison(left, right):
    if left.keys() != right.keys():
        return dict(within_fixed_bound=False, exact=False, reason="parameter_inventory")
    exact, within, maximum = True, True, 0.0
    for name, value in left.items():
        other = right[name]
        if value is None or other is None:
            if value is not None or other is not None:
                return dict(
                    within_fixed_bound=False, exact=False, reason="unused_gradient_inventory"
                )
            continue
        if value.shape != other.shape or value.dtype != other.dtype:
            return dict(within_fixed_bound=False, exact=False, reason="shape_or_dtype")
        difference = (value.double() - other.double()).abs()
        maximum = max(maximum, float(difference.max()))
        exact = exact and state_hash(value) == state_hash(other)
        bound = NUMERICAL_BOUND["absolute"] + NUMERICAL_BOUND["relative"] * value.double().abs()
        within = within and bool((difference <= bound).all())
    return dict(within_fixed_bound=within, exact=exact, max_absolute_difference=maximum)


def _support_identity(root, support_records):
    receipt = read_json(Path(root) / "native/SUPPORT_MASK_RECEIPT.json")
    if receipt["status"] != "PASS":
        raise PermissionError("Native support mask gate is incomplete")
    expected_files = {row["path"]: row["sha256"] for row in receipt["files"]}
    checked = []
    for record in support_records:
        pair = []
        for arm in ("PREFIX", "EXEC"):
            relative = f"native/support_masks/{record['qid']}-{arm}.json"
            path = Path(root) / relative
            if file_hash(path) != expected_files.get(relative):
                raise PermissionError("Frozen native teacher audit changed")
            pair.append(read_json(path))
        keys = ("tokens", "answer_mask", "work_mask", "serialized_teacher_target")
        if any(pair[0][key] != pair[1][key] or pair[1][key] != record[key] for key in keys):
            raise PermissionError("Actual ENGINE PREFIX/EXEC token/text/mask identity differs")
        checked.append(
            dict(qid=record["qid"], target_hash=digest({key: pair[0][key] for key in keys}))
        )
    return checked


def verify_engine_probes(root, runtime, optimizer, scheduler, records, reference, support_records):
    """Use the first cached ENGINE batch; make no additional sampled responses."""
    import torch

    directory = _engine_root(root) / "probes"
    before = _live_state(runtime, optimizer, scheduler, reference)
    cache_hash = state_hash(
        [
            [
                {key: row[key] for key in ("qid", "tokens", "sampler_logprobs", "reward")}
                for row in records
            ],
            [
                {key: row[key] for key in ("qid", "tokens", "answer_mask", "work_mask")}
                for row in support_records
            ],
        ]
    )
    actual_cuda = torch.cuda.is_available() and all(
        parameter.device.type == "cuda"
        for parameter in runtime.model.parameters()
        if parameter.requires_grad
    )
    complete = directory / "COMPLETE.json"
    if complete.exists():
        receipt = read_json(complete)
        if receipt["initial_state_hash"] != before or receipt["cache_hash"] != cache_hash:
            raise PermissionError(
                "Existing ENGINE probes belong to different weights or cached inputs"
            )
        if receipt["actual_cuda"] != actual_cuda:
            raise PermissionError("ENGINE probe backend changed")
        for artifact in receipt["artifacts"]:
            if file_hash(Path(root) / artifact["path"]) != artifact["sha256"]:
                raise PermissionError("ENGINE gradient probe artifact changed")
        return receipt
    mask_pairs = _support_identity(root, support_records)
    artifacts, attempts, successful = [], [], {}

    def run(name, arm, size, zero=False):
        try:
            result = probe_gradients(
                runtime,
                optimizer,
                scheduler,
                records,
                arm=arm,
                reference=reference,
                support_records=None if arm == "RL" else support_records,
                step=1,
                microbatch_size=size,
                engineering_zero_aux=zero,
            )
        except BaseException as exc:
            failure = dict(
                name=name,
                exception_type=type(exc).__name__,
                message=str(exc),
                cost=getattr(exc, "training_probe_cost", None),
                live_state_restored=_live_state(runtime, optimizer, scheduler, reference) == before,
            )
            durable_json(directory / f"{name}.FAILURE.json", failure)
            attempts.append(failure)
            raise
        gradients = result.pop("unclipped_gradients")
        path = directory / f"{name}.pt"
        atomic_torch(path, gradients, exclusive=True)
        stats_path = directory / f"{name}.json"
        durable_json(stats_path, result)
        for artifact in (path, stats_path):
            artifacts.append(
                dict(path=str(artifact.relative_to(Path(root))), sha256=file_hash(artifact))
            )
        if _live_state(runtime, optimizer, scheduler, reference) != before:
            raise RuntimeError("No-Adam ENGINE probe changed live training state")
        if (
            result["total_cost"]["optimizer_step_calls_attempted"]
            or result["total_cost"]["global_clip_calls"]
        ):
            raise RuntimeError("An ENGINE gradient probe consumed an optimizer update")
        return result, gradients

    for size in (4, 2, 1):
        try:
            successful[size] = run(f"EXEC-mb{size}", "EXEC", size)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if not successful:
        raise torch.cuda.OutOfMemoryError(
            "All frozen ENGINE probe microbatches failed; backend gate needed"
        )
    selected, baseline = max(successful), min(successful)
    base_result, base_gradient = successful[baseline]
    comparisons = []
    for size, (result, gradient) in successful.items():
        if size == baseline:
            continue
        comparison = _gradient_comparison(base_gradient, gradient)
        losses = {
            key: math.isclose(
                base_result[key],
                result[key],
                rel_tol=NUMERICAL_BOUND["relative"],
                abs_tol=NUMERICAL_BOUND["absolute"],
            )
            for key in ("task_loss", "aux_ce", "loss")
        }
        comparisons.append(
            dict(
                reference_microbatch=baseline, microbatch=size, gradients=comparison, losses=losses
            )
        )
        if not comparison["within_fixed_bound"] or not all(losses.values()):
            durable_json(directory / "MICROBATCH_FAILURE.json", comparisons[-1])
            raise RuntimeError("ENGINE microbatch objective/gradient equivalence failed")
    pure, pure_gradient = run(f"RL-mb{baseline}", "RL", baseline)
    zero, zero_gradient = run(f"EXEC-zero-mb{baseline}", "EXEC", baseline, zero=True)
    zero_equal = state_hash(pure_gradient) == state_hash(zero_gradient)
    nonzero = base_result["gradient_diagnostics"]["unweighted_norms"]["aux_ce"] > 0
    receipt = dict(
        status="PASS"
        if actual_cuda and zero_equal and nonzero
        else "CPU_EVIDENCE_ONLY"
        if not actual_cuda
        else "FAIL",
        actual_cuda=actual_cuda,
        initial_state_hash=before,
        cache_hash=cache_hash,
        selected_microbatch_size=selected,
        feasible_microbatches=sorted(successful, reverse=True),
        numerical_bound=NUMERICAL_BOUND,
        numerical_bound_fixed_before_measurement=True,
        microbatch_comparisons=comparisons,
        microbatch_comparison_scope="AVAILABLE_GPU_MICROBATCHES"
        if comparisons
        else "ONLY_ONE_FEASIBLE",
        ce_zero_equals_pure_rl_exact=zero_equal,
        nonzero_ce_gradient=nonzero,
        prefix_exec_mask_pairs=mask_pairs,
        prefix_exec_same_target_and_answer_mask=True,
        probabilities="PASS",
        physical_updates=0,
        new_rollouts=0,
        state_unchanged=_live_state(runtime, optimizer, scheduler, reference) == before,
        attempts=attempts,
        artifacts=artifacts,
        probe_costs={str(size): result["total_cost"] for size, (result, _) in successful.items()},
        pure_rl_cost=pure["total_cost"],
        zero_ce_cost=zero["total_cost"],
    )
    durable_json(complete, receipt)
    if not zero_equal or not nonzero or not receipt["state_unchanged"]:
        raise RuntimeError("ENGINE CE/gradient/state probe failed; evidence retained")
    return receipt


def record_engine_final_logprobs(root, runtime, segment):
    """Teacher-force the same step-four ENGINE tokens at each final theta_4."""
    import torch

    if segment not in ("continuous", "split"):
        raise ValueError("Unknown ENGINE segment")
    directory = _engine_root(root) / segment
    state, _receipt = read_checkpoint(directory / "checkpoints", step=4)
    if state_hash(trainable_state(runtime.model)) != state_hash(state["parameters"]):
        raise PermissionError("Final ENGINE runtime does not match checkpoint four")
    rows = []
    for slot in range(16):
        source = read_json(_engine_root(root) / "continuous/rollouts" / f"04-{slot:02d}.json")
        if source["records_sha256"] != digest(source["records"]):
            raise PermissionError("ENGINE final-logp cached rows changed")
        rows.extend(source["records"])
    identity = dict(
        policy_hash=state_hash(state["parameters"]),
        token_path_hash=digest([{key: row[key] for key in ("qid", "tokens")} for row in rows]),
    )
    marker, tensors = directory / "FINAL_LOGPROBS.json", directory / "FINAL_LOGPROBS.pt"
    if marker.exists():
        saved = read_json(marker)
        if saved["identity"] != identity or file_hash(tensors) != saved["sha256"]:
            raise PermissionError("Final ENGINE logprob evidence changed")
        return saved
    rng, mode = capture_rng(), runtime.model.training
    values = []
    try:
        with torch.no_grad():
            for row in rows:
                value = runtime.batch_sequence_forward(
                    [row], purpose="engine_final_logprob", grad=False
                )[0]
                if value.numel() != len(row["tokens"]) or not bool(torch.isfinite(value).all()):
                    raise FloatingPointError("Invalid final ENGINE logprob vector")
                values.append(value.detach().float().cpu())
    finally:
        restore_rng(rng)
        runtime.model.train(mode)
    atomic_torch(tensors, values, exclusive=True)
    saved = dict(
        identity=identity,
        sha256=file_hash(tensors),
        vectors_hash=state_hash(values),
        sequences=len(values),
        tokens=sum(value.numel() for value in values),
        new_rollouts=0,
        physical_updates=0,
        input_scope="CACHED_ENGINE_ONLY",
    )
    durable_json(marker, saved)
    return saved


def _copy_step_two(root):
    import torch

    base = _engine_root(root)
    state, receipt = read_checkpoint(base / "continuous/checkpoints", step=2)
    target = base / "split/checkpoints"
    target.mkdir(parents=True, exist_ok=True)
    source = base / "continuous/checkpoints" / receipt["path"]
    destination = target / receipt["path"]
    if destination.exists():
        if file_hash(destination) != receipt["sha256"]:
            raise PermissionError("Copied step-two checkpoint bytes changed")
    else:
        # atomic_torch may serialize equivalent tensors to different file bytes;
        # byte-copy the committed source using an atomic hard-link publication.
        temporary = target / f".copy-{os.getpid()}-{time.time_ns()}"
        try:
            with source.open("rb") as src, temporary.open("xb") as dst:
                while chunk := src.read(8 << 20):
                    dst.write(chunk)
                dst.flush()
                os.fsync(dst.fileno())
            if file_hash(temporary) != receipt["sha256"]:
                raise PermissionError("Step-two copy failed SHA-256 verification")
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    durable_json(target / "commit-02.json", receipt)
    if not (target / "LATEST.json").exists():
        durable_json(target / "LATEST.json", receipt)
    if state_hash(torch.load(destination, map_location="cpu", weights_only=False)) != state_hash(
        state
    ):
        raise PermissionError("Copied ENGINE checkpoint complete-state hash differs")
    durable_json(
        base / "split/INITIALIZATION.json",
        dict(
            source="continuous/checkpoints/commit-02.json",
            source_receipt=receipt,
            copied_state_hash=state_hash(state),
            copied_logical_step=2,
            new_physical_updates=0,
        ),
    )


def _run_segment(root, segment):
    base = _engine_root(root)
    script = Path(__file__).resolve().parents[2] / "scripts/sr_f3_ce/run_worker.py"
    command = [
        sys.executable,
        str(script),
        "--root",
        str(Path(root).resolve()),
        "--task-id",
        "engine--resume",
        "--engine-segment",
        segment,
    ]
    launch, exited = base / f"{segment}.LAUNCH.json", base / f"{segment}.EXIT.json"
    if exited.exists():
        previous = read_json(exited)
        if previous["returncode"] != 0:
            raise RuntimeError(
                "Prior ENGINE subprocess failed; retain and diagnose before recovery"
            )
        if read_json(launch)["command"] != command:
            raise PermissionError("ENGINE subprocess command changed")
        return
    if launch.exists():
        raise RuntimeError("ENGINE subprocess outcome is unknown; no blind replay is permitted")
    durable_json(launch, dict(command=command, parent_pid=os.getpid(), at=now()))
    with (base / f"{segment}.stdout.log").open("xb") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ)
        )
        durable_json(
            base / f"{segment}.PROCESS.json", dict(pid=process.pid, parent_pid=os.getpid())
        )
        code = process.wait()
        log.flush()
        os.fsync(log.fileno())
    durable_json(exited, dict(returncode=code, pid=process.pid, at=now()))
    if code:
        raise RuntimeError(f"ENGINE {segment} subprocess failed with exit code {code}")


def validate_engine_recovery(root):
    """Verify retained model state, raw cache, real probability metrics and intents."""
    import torch

    base = _engine_root(root)
    states = [
        read_checkpoint(base / segment / "checkpoints", step=4)[0]
        for segment in ("continuous", "split")
    ]
    comparison = compare_recovery_states(*states)
    durable_json(base / "STATE_COMPARISON.json", comparison)
    if not comparison["all_logical_fields_exact"]:
        raise RuntimeError(
            "ENGINE complete-state recovery differs; sensitivity numbers do not waive the gate"
        )
    processes = [read_json(base / f"{segment}.PROCESS.json") for segment in ("continuous", "split")]
    if len({row["pid"] for row in processes}) != 2 or any(
        row["pid"] == os.getpid() for row in processes
    ):
        raise PermissionError("ENGINE replay was not performed in two distinct child processes")
    intents = _intent_rows(root)
    if len(intents) != 6:
        raise PermissionError("ENGINE must have exactly six durable physical Adam intents")
    metrics = []
    for segment, expected_steps in (("continuous", (1, 2, 3, 4)), ("split", (3, 4))):
        rows = [
            read_json(path) for path in sorted((base / segment / "update_attempts").glob("*.json"))
        ]
        if [row["step"] for row in rows] != list(expected_steps):
            raise PermissionError("ENGINE update receipt inventory differs from four plus two")
        for row in rows:
            value = row["metrics"]
            if (
                value["optimizer_updates"] != 1
                or value["total_cost"]["optimizer_step_calls_attempted"] != 1
            ):
                raise PermissionError(
                    "ENGINE physical update count differs from actual update metrics"
                )
            if (
                value["sampler_train_mean_absolute_difference"] > 0.01
                or value["tis_truncated_fraction"] > 0.01
            ):
                raise PermissionError(
                    "Actual ENGINE sampler/teacher-forcing probability gate failed"
                )
            metrics.append(value)
    cache_paths = sorted((base / "continuous/rollouts").glob("[0-9][0-9]-[0-9][0-9].json"))
    expected_cache_names = [
        f"{step:02d}-{slot:02d}.json" for step in range(1, 5) for slot in range(16)
    ]
    if [path.name for path in cache_paths] != expected_cache_names or any(
        (base / "split/rollouts").glob("*.json")
    ):
        raise PermissionError(
            "ENGINE must use exactly 64 shared cached groups without split resampling"
        )
    for index, path in enumerate(cache_paths):
        group = read_json(path)
        if len(group["records"]) != 8 or group["records_sha256"] != digest(group["records"]):
            raise PermissionError("ENGINE cached group is incomplete or changed")
        step, slot = divmod(index, 16)
        if group["identity"]["step"] != step + 1 or group["identity"]["slot"] != slot:
            raise PermissionError("ENGINE cached group has the wrong logical step/slot")
        if any(row["qid"] != states[0]["used_task_qids"][index] for row in group["records"]):
            raise PermissionError("ENGINE cached qid differs from committed input history")
    chain = digest([])
    for step in range(1, 5):
        group_hashes = [file_hash(path) for path in cache_paths[(step - 1) * 16 : step * 16]]
        support_hashes = [
            file_hash(base / "continuous/support" / f"{step:02d}-{qid}.json")
            for qid in states[0]["used_support_qids"][(step - 1) * 4 : step * 4]
        ]
        chain = digest([chain, step, group_hashes, support_hashes])
    if chain != states[0]["committed_records_hash"]:
        raise PermissionError("ENGINE raw/support record chain differs from checkpoint")
    accounting_path = Path(root) / "accounting/engine--resume.jsonl"
    accounting = [json.loads(line) for line in accounting_path.read_text().splitlines()]
    if sum(row["count"] for row in accounting if row["kind"] == "completion_attempts") != 512:
        raise PermissionError("Actual ENGINE generation accounting differs from 512 new responses")
    probes = read_json(base / "probes/COMPLETE.json")
    if (
        probes["status"] != "PASS"
        or not probes["actual_cuda"]
        or not probes["state_unchanged"]
        or not probes["ce_zero_equals_pure_rl_exact"]
        or not probes["nonzero_ce_gradient"]
        or probes["physical_updates"] != 0
        or probes["new_rollouts"] != 0
    ):
        raise PermissionError("Real CUDA gradient probe evidence is incomplete")
    for artifact in probes["artifacts"]:
        if file_hash(Path(root) / artifact["path"]) != artifact["sha256"]:
            raise PermissionError("Retained ENGINE gradient probe artifact changed")
    finals, values = [], []
    for segment in ("continuous", "split"):
        marker = read_json(base / segment / "FINAL_LOGPROBS.json")
        path = base / segment / "FINAL_LOGPROBS.pt"
        if file_hash(path) != marker["sha256"]:
            raise PermissionError("Final ENGINE logprob tensor bytes changed")
        tensor_rows = torch.load(path, map_location="cpu", weights_only=False)
        if state_hash(tensor_rows) != marker["vectors_hash"]:
            raise PermissionError("Final ENGINE logprob vectors changed")
        finals.append(marker)
        values.append(torch.cat(tensor_rows))
    if finals[0]["identity"] != finals[1]["identity"] or values[0].shape != values[1].shape:
        raise PermissionError("Final ENGINE logprob input/policy identity differs")
    delta = (values[0].double() - values[1].double()).abs()
    final_comparison = dict(
        exact=state_hash(values[0]) == state_hash(values[1]),
        mean_absolute_difference=float(delta.mean()),
        max_absolute_difference=float(delta.max()),
        scope="CACHED_ENGINE_ONLY_NO_LOCKED_TEST_ACCESS",
    )
    durable_json(base / "FINAL_LOGPROB_COMPARISON.json", final_comparison)
    if not final_comparison["exact"]:
        raise RuntimeError("Final ENGINE logprobs are not exact; retain differences for diagnosis")
    return dict(
        logical_updates=4,
        physical_updates=6,
        new_rollouts=512,
        replayed_rollouts=256,
        restore="PASS",
        masks="PASS",
        probabilities="PASS",
        nonzero_ce_gradient=probes["nonzero_ce_gradient"],
        new_process_boundary=True,
        optimizer_intents=len(intents),
        sampled_cache_files=len(cache_paths),
    )


def execute_engine_recovery(root):
    """Run continuous four, copy its step two, then replay only steps three/four."""
    from .orchestration import worker_binding, write_task_receipt

    root = Path(root).resolve(strict=True)
    verify_run(root)
    worker_binding(root, "engine--resume")
    base = _engine_root(root)
    with shard_lease(base):
        _run_segment(root, "continuous")
        _copy_step_two(root)
        _run_segment(root, "split")
        evidence = validate_engine_recovery(root)
        files = [path for path in base.rglob("*") if path.is_file() and path.name != ".worker.lock"]
        return write_task_receipt(
            root,
            "engine--resume",
            artifacts=[str(path.relative_to(root)) for path in sorted(files)]
            + ["accounting/engine--resume.jsonl"],
            evidence=evidence,
        )
