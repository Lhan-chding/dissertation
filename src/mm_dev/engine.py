"""Real F2 4 versus fresh-process 2+2 GRPO engineering qualification."""

from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from mm_core.training import configure_training, frozen_hash, state_hash, trainable_state
from mm_core.vl_runtime import seed_all

from .contract import PLAN_ID, digest, seed
from .runtime import (
    actual_cuda_identity,
    adapter_identity,
    atomic_json,
    bounded_path,
    file_hash,
    load_runtime,
    read_json,
)
from .training import (
    CHECKPOINT_FIELDS,
    LeaseEnding,
    execute_path,
    group_diagnostics,
    pin_checkpoint,
    rewards,
    stop_at_committed_boundary,
)


def initialize_common(plan, root, freeze, account):
    root = Path(root)
    receipt_path = root / "manifests/COMMON_LORA.json"
    if receipt_path.exists():
        from .runtime import verified_state

        return verified_state(root, "S0")
    runtime = load_runtime(plan, root, account=account, freeze=freeze)
    questions = [
        json.loads(line) for line in (root / "data/questions.jsonl").read_text().splitlines()
    ]
    row = sorted(
        (q for q in questions if q["split"] == "ENGINE_F2"), key=lambda q: q["question_id"]
    )[0]
    prepared = runtime.prepare(row, root)
    inputs = dict(prepared["inputs"])
    inputs.update(logits_to_keep=1, use_cache=False)
    runtime.reserve("extra_forward_sequences", 1, purpose="common_base_invariance")
    with runtime.torch.no_grad():
        baseline = runtime.model(**inputs).logits.detach().cpu()
    seed_all(plan["common_start"]["zero_output_lora_initialization_seed"])
    _, _, training_identity = configure_training(runtime)
    for name, parameter in runtime.model.named_parameters():
        if "lora_B" in name and bool(parameter.detach().ne(0).any()):
            raise RuntimeError("Common adapter was not initialized with exact zero output")
    runtime.reserve("extra_forward_sequences", 1, purpose="common_zero_lora_invariance")
    with runtime.torch.no_grad():
        after = runtime.model(**inputs).logits.detach().cpu()
    if not runtime.torch.equal(baseline, after):
        raise RuntimeError("Zero-output LoRA changed native base logits")
    adapter_path = root / "states/common_lora"
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pending-common-", dir=adapter_path.parent))
    try:
        runtime.model.save_pretrained(staging, safe_serialization=True)
        if adapter_path.exists():
            actual = adapter_identity(root, str(adapter_path.relative_to(root)))
            expected = adapter_identity(root, str(staging.relative_to(root)))
            if actual["adapter_file_hashes"] != expected["adapter_file_hashes"]:
                raise PermissionError(
                    "Uncommitted common adapter differs from fixed initialization"
                )
        else:
            os.rename(staging, adapter_path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    receipt = dict(
        plan_id=PLAN_ID,
        status="VERIFIED",
        state_id="S0",
        producer_run_id="COMMON",
        freeze_sha256=file_hash(root / "manifests/F2_FREEZE.json"),
        base_model_weights_hash=plan["model"]["model_weights_hash"],
        initialization_seed=plan["common_start"]["zero_output_lora_initialization_seed"],
        **adapter_identity(root, str(adapter_path.relative_to(root))),
        trainable_state_hash=state_hash(trainable_state(runtime.model)),
        frozen_base_hash=frozen_hash(runtime.model),
        training_identity=training_identity,
        invariance=dict(
            question_id=row["question_id"],
            routing=prepared["routing"],
            full_last_token_logits_equal=True,
            logits_hash=state_hash(baseline),
            zero_lora_B_verified=True,
            completions=0,
            short_forwards=2,
        ),
    )
    atomic_json(receipt_path, receipt, exclusive=True)
    del runtime
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    return receipt


def engine_run(stress):
    return dict(
        run_id="ENGINE_F2_STRESS" if stress else "ENGINE_F2_NATURAL",
        phase="ENGINE_F2",
        start_state="S0",
        reference_state="S0",
        repeat=0,
        preparation_recipe="AP",
        steps=4,
        schedule_id="ENGINE_e0",
        optimizer_start="fresh",
    )


def release_parent_cuda_cache():
    """Run after COMMON's function locals die, before any child model is launched."""
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def verify_comparison_artifacts(root, result):
    root = Path(root)
    if (
        result.get("plan_id") != PLAN_ID
        or result.get("freeze_sha256") != file_hash(root / "manifests/F2_FREEZE.json")
        or result.get("valid_trace_completions") != 1536
        or result.get("valid_trace_updates") != 8
        or not result.get("artifact_hashes")
    ):
        raise PermissionError("ENGINE comparison does not bind the required frozen evidence")
    for relative, expected in result["artifact_hashes"].items():
        if file_hash(bounded_path(root, relative)) != expected:
            raise PermissionError("ENGINE comparison artifact changed: " + relative)
    if result.get("stress_allowed"):
        if (
            result.get("mode") != "natural"
            or result.get("status") != "EXACT_CONSTANT_REWARD_TRACE"
            or result.get("natural_nonconstant_groups") != 0
        ):
            raise PermissionError("Natural evidence does not authorize surrogate stress")
        for track in ("continuous", "split"):
            for step in range(1, 5):
                metrics = read_json(
                    root / "engineering/natural" / track / "steps" / f"{step:02d}.json"
                )
                if (
                    not metrics["all_natural_advantages_zero"]
                    or metrics["policy_gradient_norm"] != 0
                    or any(
                        not group["zero_contrast"] or any(group["advantages"])
                        for group in metrics["groups"]
                    )
                ):
                    raise PermissionError("Saved natural trace is not entirely constant/zero-PG")
    return result


def run_segment(plan_path, root, *, mode, segment):
    with stop_at_committed_boundary() as boundary:
        return _run_segment(plan_path, root, mode=mode, segment=segment, boundary=boundary)


def _run_segment(plan_path, root, *, mode, segment, boundary):
    from .common import CostLedger, load_plan, require_allocation, verify_execution

    plan_path, root = Path(plan_path).resolve(), Path(root).resolve()
    plan = load_plan(plan_path)
    freeze = verify_execution(plan_path, root)
    allocation = require_allocation(root, "ENGINE_F2")
    if mode not in {"natural", "stress"} or segment not in {"continuous", "first", "resume"}:
        raise PermissionError("Unregistered ENGINE segment")
    if mode == "stress":
        natural = verify_comparison_artifacts(
            root, read_json(root / "engineering/natural/COMPARISON.json")
        )
        if natural["status"] != "EXACT_CONSTANT_REWARD_TRACE" or not natural["stress_allowed"]:
            raise PermissionError(
                "Synthetic surrogate stress is not authorized by natural evidence"
            )
    path = root / "engineering" / mode / ("continuous" if segment == "continuous" else "split")
    marker = path / (segment.upper() + "_COMPLETE.json")
    if marker.exists():
        raise FileExistsError("Completed ENGINE segment cannot execute again")
    ledger = CostLedger(root, "ENGINE_F2")
    runtime = load_runtime(
        plan,
        root,
        state_id="S0",
        freeze=freeze,
        account=lambda kind, count, metadata: ledger.reserve(
            kind, count, {**metadata, "engine_mode": mode, "engine_segment": segment}
        ),
    )
    run = engine_run(mode == "stress")
    schedule = [
        json.loads(line)
        for line in (plan_path.parent / "schedules/ENGINE_e0.jsonl").read_text().splitlines()
    ]
    questions = {
        row["question_id"]: row
        for row in map(json.loads, (root / "data/questions.jsonl").read_text().splitlines())
    }
    if segment == "resume" and not (path / "FIRST_COMPLETE.json").exists():
        raise PermissionError("Fresh resume requires successful independent first two updates")
    if segment != "resume" and (path / "checkpoints/LATEST.json").exists():
        raise PermissionError(
            "Interrupted ENGINE segment requires explicit technical reconciliation"
        )
    result = execute_path(
        runtime,
        root,
        path,
        run,
        schedule,
        questions,
        stop_step=2 if segment == "first" else 4,
        engine=True,
        stress=mode == "stress",
        resume_step=2 if segment == "resume" else None,
        boundary=boundary,
    )
    expected_base = read_json(root / "manifests/COMMON_LORA.json")["frozen_base_hash"]
    actual_base = frozen_hash(runtime.model)
    if actual_base != expected_base:
        raise RuntimeError("Frozen base/reference unexpectedly changed")
    result.update(
        freeze_sha256=file_hash(root / "manifests/F2_FREEZE.json"),
        allocation=allocation,
        frozen_base_hash=actual_base,
        runtime_identity=runtime.identity,
        mode=mode,
        segment=segment,
    )
    atomic_json(marker, result, exclusive=True)
    return result


def independent_checkpoint_recount(root, mode, plan_path):
    """Run the separately implemented element comparison without a CUDA context."""
    script = Path(__file__).resolve().parents[2] / "scripts/mm_dev/verify_engine_checkpoints.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--plan",
            str(plan_path),
            "--run-root",
            str(root),
            "--mode",
            mode,
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=True,
    )
    relative = f"engineering/{mode}/INDEPENDENT_CHECKPOINT_RECOUNT.json"
    result = read_json(Path(root) / relative)
    if (
        result.get("status") != "VERIFIED_ELEMENTWISE_EQUAL"
        or result.get("cuda_initialized") is not False
        or result.get("freeze_sha256") != file_hash(Path(root) / "manifests/F2_FREEZE.json")
        or result.get("mode") != mode
        or {r["step"] for r in result["comparisons"]} != {2, 4}
        or not result.get("artifact_hashes")
    ):
        raise PermissionError("Independent CPU checkpoint recount is incomplete")
    for path, sha in result["artifact_hashes"].items():
        if file_hash(bounded_path(root, path)) != sha:
            raise PermissionError("Independent checkpoint recount evidence changed")
    return relative


def compare_engine(root, mode, plan_path=None):
    import torch

    root = Path(root)
    directory = root / "engineering" / mode
    plan_path = (
        Path(plan_path)
        if plan_path
        else Path(__file__).resolve().parents[2] / "docs/mm_dev_f2/design/config/MM_DEV_F2.json"
    )
    schedule = [
        json.loads(line)
        for line in (plan_path.parent / "schedules/ENGINE_e0.jsonl").read_text().splitlines()
    ]
    schedule_map = {(s["logical_step"], s["slot"]): s for s in schedule}
    questions = {
        q["question_id"]: q
        for q in map(json.loads, (root / "data/questions.jsonl").read_text().splitlines())
    }
    continuous, split = directory / "continuous", directory / "split"
    receipts = [
        read_json(path)
        for path in (
            continuous / "CONTINUOUS_COMPLETE.json",
            split / "FIRST_COMPLETE.json",
            split / "RESUME_COMPLETE.json",
        )
    ]
    if len({r["process_id"] for r in receipts}) != 3:
        raise RuntimeError("ENGINE needs three genuinely independent processes")
    freeze_sha = file_hash(root / "manifests/F2_FREEZE.json")
    if any(r["freeze_sha256"] != freeze_sha for r in receipts):
        raise PermissionError("ENGINE segment used a different freeze")
    if len({digest(r["runtime_identity"]) for r in receipts}) != 1:
        raise RuntimeError("ENGINE runtime identities changed across fresh processes")
    if len({r["frozen_base_hash"] for r in receipts}) != 1:
        raise RuntimeError("Frozen base parameters differ")
    for receipt, track, initial in zip(
        receipts, (continuous, split, split), (0, 0, 2), strict=True
    ):
        process = read_json(track / f"PROCESS-{receipt['process_id']}.json")
        if process["initial_step"] != initial:
            raise RuntimeError(
                "ENGINE segment did not begin at its registered fresh/resume boundary"
            )
        if process["restored_rng_hash"] != process["checkpoint_rng_hash"]:
            raise RuntimeError("ENGINE process did not restore exact RNG")
        if process["runtime_identity"] != receipt["runtime_identity"]:
            raise PermissionError("ENGINE process identity changed")
    comparisons = []
    verified_states = {}
    for step in range(5):
        states = []
        for track in (continuous, split):
            receipt = read_json(track / "checkpoints" / f"commit-{step:02d}.json")
            path = bounded_path(track / "checkpoints", receipt["path"])
            if file_hash(path) != receipt["sha256"]:
                raise PermissionError("ENGINE committed checkpoint bytes changed")
            state = torch.load(path, map_location="cpu", weights_only=False)
            if set(state) != CHECKPOINT_FIELDS or state_hash(state) != receipt["state_hash"]:
                raise PermissionError("ENGINE full checkpoint contents changed")
            if state["committed_logical_step"] != step or state["plan_id"] != PLAN_ID:
                raise PermissionError("ENGINE checkpoint step/plan changed")
            if not state["parameters"] or (step and not state["optimizer"]["state"]):
                raise RuntimeError("ENGINE checkpoint lacks real policy/Adam state")
            if state["input_stream_hash"] != digest(schedule):
                raise PermissionError("ENGINE checkpoint does not bind the frozen schedule")
            if (
                step == 0
                and state_hash(state["parameters"])
                != read_json(root / "manifests/COMMON_LORA.json")["trainable_state_hash"]
            ):
                raise PermissionError("ENGINE did not start from the shared zero-output adapter")
            verified_states[(track.name, step)] = state
            states.append(state)
        fields = {
            key: state_hash(states[0][key]) == state_hash(states[1][key])
            for key in CHECKPOINT_FIELDS
        }
        if not all(fields.values()):
            raise RuntimeError("ENGINE_EXACT_RESUME_MISMATCH: " + json.dumps(fields))
        comparisons.append(dict(step=step, fields=fields, state_hash=state_hash(states[0])))
    all_steps = []
    raw_coverage = 0
    for track in (continuous, split):
        if len(list((track / "rollouts").glob("*.json"))) != 4 * 24 * 8:
            raise PermissionError("ENGINE raw rollout coverage is incomplete or duplicated")
    for step in range(1, 5):
        a, b = [read_json(path / "steps" / f"{step:02d}.json") for path in (continuous, split)]
        if a != b:
            raise RuntimeError("ENGINE deterministic tokens/losses/diagnostics differ")
        for track in (continuous, split):
            checkpoint = read_json(track / "checkpoints" / f"commit-{step:02d}.json")
            state = torch.load(
                bounded_path(track / "checkpoints", checkpoint["path"]),
                map_location="cpu",
                weights_only=False,
            )
            if (
                digest(a) != state["diagnostics_hash"]
                or a["parameter_hash_before"]
                != state_hash(verified_states[(track.name, step - 1)]["parameters"])
                or a["parameter_hash_after"] != state_hash(state["parameters"])
                or a["reference_hash"] != state_hash(state["reference"])
            ):
                raise PermissionError("ENGINE metrics are not bound to committed state")
        token_paths = []
        for track in (continuous, split):
            records = []
            for slot in range(24):
                group_records = []
                qid = schedule_map[(step, slot)]["question_id"]
                row = questions[qid]
                for index in range(8):
                    record = read_json(track / "rollouts" / f"{step:02d}-{slot:02d}-{index}.json")
                    expected_seed = seed("rollout", "ENGINE_F2", 0, qid, index)
                    expected = dict(
                        run_id=engine_run(mode == "stress")["run_id"],
                        logical_step=step,
                        slot=slot,
                        sample_index=index,
                        question_id=qid,
                        seed=expected_seed,
                        policy_hash=a["parameter_hash_before"],
                        sampling_hash=verified_states[(track.name, step)]["sampling_hash"],
                        request_id=digest(
                            [
                                PLAN_ID,
                                "ENGINE_F2",
                                engine_run(mode == "stress")["run_id"],
                                step,
                                slot,
                                index,
                                a["parameter_hash_before"],
                            ]
                        ),
                        input_hash=digest(
                            {
                                key: row[key]
                                for key in ("question_id", "image_path", "image_sha256", "prompt")
                            }
                        ),
                    )
                    if any(record.get(key) != value for key, value in expected.items()):
                        raise PermissionError("ENGINE raw slot differs from frozen schedule/policy")
                    if record.get("generation_status") != "COMPLETE" or record[
                        "record_hash"
                    ] != digest({k: v for k, v in record.items() if k != "record_hash"}):
                        raise PermissionError("ENGINE raw receipt is invalid or changed")
                    if (
                        len(record["tokens"]) != len(record["old_logprobs"])
                        or record["old_logprobs"] != record["sampler_logprobs"]
                        or not all(math.isfinite(v) for v in record["old_logprobs"])
                    ):
                        raise PermissionError("ENGINE raw sampler probabilities are incomplete")
                    group_records.append(dict(reward=rewards(record["raw_text"], row)))
                    records.append(
                        {
                            key: record[key]
                            for key in (
                                "question_id",
                                "tokens",
                                "raw_text",
                                "image_routing",
                                "old_logprobs",
                                "seed",
                            )
                        }
                    )
                    raw_coverage += 1
                expected_group = group_diagnostics(group_records, "AP")
                saved_group = next(g for g in a["groups"] if g["slot"] == slot)
                if any(saved_group.get(key) != value for key, value in expected_group.items()):
                    raise PermissionError("ENGINE rewards/advantages do not match raw completions")
            if digest(records) != a["token_path_hash"]:
                raise PermissionError("ENGINE raw token path is not bound to committed diagnostics")
            token_paths.append(records)
        if token_paths[0] != token_paths[1]:
            raise RuntimeError("ENGINE raw token/text/routing/probability paths differ")
        if a["natural_nonconstant_groups"] != sum(not g["zero_contrast"] for g in a["groups"]) or a[
            "all_natural_advantages_zero"
        ] != all(not any(g["advantages"]) for g in a["groups"]):
            raise PermissionError("ENGINE natural support summary differs from exact rewards")
        probability = a["sampler_teacher_forcing"]
        differences = probability["absolute_differences"]
        if (
            not differences
            or len(differences) != a["completion_tokens"]
            or any(not math.isfinite(v) or v < 0 for v in differences)
            or sum(differences) / len(differences) > 0.005
            or max(differences) > 0.05
        ):
            raise RuntimeError("Saved ENGINE sampler/teacher-forcing probability gate failed")
        all_steps.append(a)
    nonconstant = sum(s["natural_nonconstant_groups"] for s in all_steps)
    natural_zero = all(
        s["all_natural_advantages_zero"] and s["policy_gradient_norm"] == 0 for s in all_steps
    )
    if (
        mode == "natural"
        and nonconstant
        and not all(
            s["policy_gradient_norm"] > 0 and s["parameter_changed"]
            for s in all_steps
            if s["natural_nonconstant_groups"]
        )
    ):
        raise RuntimeError("GRPO_GRADIENT_MISMATCH")
    stress_allowed = mode == "natural" and nonconstant == 0 and natural_zero
    if mode == "natural" and not nonconstant and not stress_allowed:
        raise RuntimeError("Natural constant groups had nonzero policy gradient")
    if mode == "stress" and not all(
        s["policy_gradient_norm"] > 0 and s["parameter_changed"] for s in all_steps
    ):
        raise RuntimeError("Stress did not prove nonzero policy updates")
    status = "EXACT_CONSTANT_REWARD_TRACE" if stress_allowed else "EXACT_NONZERO_TRACE"
    ledger_path = root / "accounting/ENGINE_F2.jsonl"
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    mode_cost = [r for r in ledger if r["metadata"].get("engine_mode") == mode]
    completions = sum(r["count"] for r in mode_cost if r["kind"] == "completion_attempts")
    physical_updates = sum(
        r["count"] for r in mode_cost if r["kind"] == "physical_optimizer_updates"
    )
    if completions < raw_coverage or physical_updates < len(all_steps) * 2:
        raise PermissionError("ENGINE physical-attempt ledger does not cover its valid traces")
    independent_receipt = independent_checkpoint_recount(root, mode, plan_path)
    artifacts = {
        str(path.relative_to(root)): file_hash(path)
        for track in (continuous, split)
        for path in track.rglob("*")
        if path.is_file()
    }
    artifacts.update(
        {
            str(path.relative_to(root)): file_hash(path)
            for path in (directory / "logs").glob("*.log")
            if path.is_file()
        }
    )
    artifacts[independent_receipt] = file_hash(root / independent_receipt)
    interrupted = root / "engineering/interrupted" / mode
    if interrupted.exists():
        artifacts.update(
            {
                str(path.relative_to(root)): file_hash(path)
                for path in interrupted.rglob("*")
                if path.is_file()
            }
        )
    result = dict(
        plan_id=PLAN_ID,
        status=status,
        mode=mode,
        stress_allowed=stress_allowed,
        freeze_sha256=freeze_sha,
        natural_nonconstant_groups=nonconstant,
        comparisons=comparisons,
        process_ids=[r["process_id"] for r in receipts],
        step_diagnostics_hash=digest(all_steps),
        physical_updates=physical_updates,
        completions=completions,
        valid_trace_completions=raw_coverage,
        valid_trace_updates=len(all_steps) * 2,
        artifact_hashes=artifacts,
        physical_cost_rows=mode_cost,
        all_probability_gates_passed=True,
        exact_tokens_losses_and_states=True,
    )
    atomic_json(directory / "COMPARISON.json", result, exclusive=True)
    return result


def prepare_engine_track(root, mode, track, hardware=None):
    """A preempted continuous segment restarts fresh; its old trace is preserved."""
    directory = Path(root) / "engineering" / mode / track
    if not directory.exists():
        return
    done = directory / (
        "CONTINUOUS_COMPLETE.json" if track == "continuous" else "RESUME_COMPLETE.json"
    )
    markers = list(directory.glob("*_COMPLETE.json"))
    hardware_changed = hardware is not None and any(
        read_json(marker)["runtime_identity"].get("hardware") != hardware for marker in markers
    )
    if done.exists() and not hardware_changed:
        return
    if track == "split" and not hardware_changed and (directory / "FIRST_COMPLETE.json").exists():
        # A clean 2-step boundary is exactly the registered fresh-process resume.
        processes = list(directory.glob("PROCESS-*.json"))
        if len(processes) == 1:
            return
    archive = Path(root) / "engineering/interrupted" / mode
    archive.mkdir(parents=True, exist_ok=True)
    attempt = os.environ["MM_DEV_ATTEMPT_ID"]
    target = archive / f"{track}-{attempt}"
    if target.exists():
        raise FileExistsError("Interrupted ENGINE trace archive already exists")
    os.rename(directory, target)


@contextlib.contextmanager
def forwarding_lease_signals():
    lease = dict(requested=False, signal=None, child=None)

    def forward(signum, _frame):
        lease.update(requested=True, signal=signum)
        child = lease["child"]
        if child is not None and child.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                child.send_signal(signum)

    previous = {
        number: signal.signal(number, forward) for number in (signal.SIGTERM, signal.SIGUSR1)
    }
    try:
        yield lease
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def wait_engine_child(arguments, output, lease):
    child = subprocess.Popen(arguments, stdout=output, stderr=subprocess.STDOUT)
    lease["child"] = child
    try:
        if lease["requested"]:
            child.send_signal(lease["signal"])
        code = child.wait()
        return subprocess.CompletedProcess(arguments, code)
    finally:
        lease["child"] = None


def run_engine(plan_path, root):
    with forwarding_lease_signals() as lease:
        return _run_engine(plan_path, root, lease)


def _run_engine(plan_path, root, lease):
    from .common import CostLedger, load_plan, require_allocation, verify_execution
    from .orchestration import checkpoint_task, complete_task, worker_lease

    plan_path, root = Path(plan_path).resolve(), Path(root).resolve()
    plan = load_plan(plan_path)
    freeze = verify_execution(plan_path, root)
    require_allocation(root, "ENGINE_F2")
    with worker_lease(root, "ENGINE_F2"):
        if (root / "manifests/ENGINE_GATE.json").exists():
            verify_execution(plan_path, root, require_engine=True)
            gate = read_json(root / "manifests/ENGINE_GATE.json")
            complete_task(
                root,
                "ENGINE_F2",
                ["manifests/ENGINE_GATE.json", "manifests/COMMON_LORA.json"],
                dict(status=gate["status"]),
            )
            return gate
        initialize_common(plan, root, freeze, CostLedger(root, "ENGINE_F2").reserve)
        release_parent_cuda_cache()
        if lease["requested"]:
            checkpoint_task(root, "ENGINE_F2", "PREEMPTION", ["manifests/COMMON_LORA.json"])
            raise LeaseEnding("PREEMPTION")
        results = []
        for mode in ("natural", "stress"):
            if mode == "stress" and not results[0]["stress_allowed"]:
                break
            comparison = root / "engineering" / mode / "COMPARISON.json"
            if comparison.exists():
                prior = verify_comparison_artifacts(root, read_json(comparison))
                if prior["freeze_sha256"] != file_hash(root / "manifests/F2_FREEZE.json"):
                    raise PermissionError("Completed ENGINE comparison freeze changed")
                results.append(prior)
                continue
            hardware = actual_cuda_identity()
            for track in ("continuous", "split"):
                prepare_engine_track(root, mode, track, hardware)
            for segment in ("continuous", "first", "resume"):
                directory = root / "engineering" / mode
                track = "continuous" if segment == "continuous" else "split"
                if (directory / track / (segment.upper() + "_COMPLETE.json")).exists():
                    continue
                logs = directory / "logs"
                logs.mkdir(parents=True, exist_ok=True)
                attempt = os.environ["MM_DEV_ATTEMPT_ID"]
                with (logs / f"{segment}-{attempt}.log").open("x") as output:
                    completed = wait_engine_child(
                        [
                            sys.executable,
                            str(
                                Path(__file__).resolve().parents[2]
                                / "scripts/mm_dev/run_engine_f2.py"
                            ),
                            "--plan",
                            str(plan_path),
                            "--run-root",
                            str(root),
                            "--segment",
                            segment,
                            "--mode",
                            mode,
                        ],
                        output,
                        lease,
                    )
                failure = root / "orchestration/failures" / (attempt + ".json")
                if (completed.returncode == 75 or lease["requested"]) and not failure.exists():
                    marker = root / "orchestration/checkpoints" / (attempt + ".json")
                    if not marker.exists():
                        checkpoint_dir = directory / track
                        artifacts = (
                            pin_checkpoint(root, checkpoint_dir, "ENGINE_F2")
                            if (checkpoint_dir / "checkpoints/LATEST.json").exists()
                            else ["manifests/COMMON_LORA.json"]
                        )
                        checkpoint_task(root, "ENGINE_F2", "PREEMPTION", artifacts)
                    raise LeaseEnding("PREEMPTION")
                completed.check_returncode()
            results.append(compare_engine(root, mode, plan_path))
        status = (
            "PASS_NONZERO_SURROGATE_KERNEL_RESUME"
            if len(results) == 2
            else "PASS_NATURAL_GRPO_RESUME"
        )
        cost_snapshot = root / "engineering/PHYSICAL_COST_SNAPSHOT.json"
        cost_rows = [
            json.loads(line)
            for line in (root / "accounting/ENGINE_F2.jsonl").read_text().splitlines()
        ]
        snapshot = dict(
            plan_id=PLAN_ID,
            rows=cost_rows,
            generated_tokens_are_recorded_lower_bound=True,
            durable_raw_is_authority_for_known_completion_tokens=True,
        )
        if cost_snapshot.exists():
            if read_json(cost_snapshot) != snapshot:
                raise PermissionError("ENGINE cost snapshot changed after capture")
        else:
            atomic_json(cost_snapshot, snapshot, exclusive=True)
        gate = dict(
            plan_id=PLAN_ID,
            status=status,
            freeze_sha256=file_hash(root / "manifests/F2_FREEZE.json"),
            common_lora_sha256=file_hash(root / "manifests/COMMON_LORA.json"),
            artifact_hashes={
                "engineering/PHYSICAL_COST_SNAPSHOT.json": file_hash(cost_snapshot),
                **{
                    name: value
                    for result in results
                    for name, value in result["artifact_hashes"].items()
                },
                **{
                    f"engineering/{result['mode']}/COMPARISON.json": file_hash(
                        root / "engineering" / result["mode"] / "COMPARISON.json"
                    )
                    for result in results
                },
            },
            natural=results[0],
            stress=results[1] if len(results) == 2 else None,
            claim="Native full-batch GRPO kernel and exact recovery qualified; "
            "synthetic stress is not natural-reward learning evidence",
        )
        atomic_json(root / "manifests/ENGINE_GATE.json", gate, exclusive=True)
        complete_task(
            root,
            "ENGINE_F2",
            ["manifests/ENGINE_GATE.json", "manifests/COMMON_LORA.json"],
            dict(status=status),
        )
        return gate
