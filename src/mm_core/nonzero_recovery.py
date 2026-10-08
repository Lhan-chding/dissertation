"""Separately authorized nonzero SFT recovery check; never a scientific GRPO result.

This extension leaves the original eleven frozen modules and old audit evidence
unchanged. It inherits the real image protocol, native backend and common start.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

from .execution import (
    append_jsonl,
    atomic_json,
    checked_path,
    exclusive_json,
    locked,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
)
from .training import (
    LORA_RECIPE,
    OPTIMIZER_RECIPE,
    capture_rng,
    configure_training,
    deterministic_order,
    frozen_hash,
    load_checkpoint,
    save_checkpoint,
    state_hash,
    trainable_state,
)
from .vl_runtime import MODEL_ID, QwenRuntime, gold_completion, hash_json, seed_all

SCOPE = "NONZERO_SFT_RESUME"
SEED = 2026100901
PRIOR_COSTS = dict(
    completion_attempts=2433, physical_optimizer_updates=8, extra_forward_sequences=4865
)
ADDITIONAL_CAPS = dict(
    completion_attempts=8, physical_optimizer_updates=8, extra_forward_sequences=16
)
GLOBAL_CAPS = dict(
    completion_attempts=4096, physical_optimizer_updates=64, extra_forward_sequences=8192
)
NONZERO_RECIPE = dict(
    scope=SCOPE,
    logical_steps=4,
    physical_optimizer_updates=8,
    sequences_per_update=2,
    microbatch_sequences=1,
    supervised_sequence_exposures=16,
    objective="mean_per_sequence_gold_completion_NLL_then_equal_sequence_mean",
    prompt_and_image_supervised=False,
    gold_supervision_disclosed=True,
    rollout_per_update=1,
    rollout_question="first_question_of_same_step",
    rollout_timing="after_optimizer_update",
    rollout_use="engineering_comparison_only",
    generation="inherit_exact_pre_inference_freeze",
    sampling_seed="random.randrange(2**63-1)_from_restored_global_python_rng",
    optimizer=OPTIMIZER_RECIPE,
    lora=LORA_RECIPE,
    nonzero_required_each_update=True,
    exact_resume_no_tolerance=True,
    fresh_process_segments=[["continuous", 0, 4], ["resumed", 0, 2], ["resumed", 2, 4]],
    replaces_common_start=False,
    certifies_scientific_grpo=False,
)


def verify_nonzero_gate(run_root, *, source_root=None):
    root = Path(run_root).resolve()
    freeze = read_json(root / "manifests/NONZERO_FREEZE.json")
    if (
        freeze.get("run_root") != str(root)
        or not isinstance(freeze.get("original_run_root"), str)
        or not Path(freeze["original_run_root"]).is_absolute()
        or Path(freeze["original_run_root"]).resolve() == root
    ):
        raise PermissionError("Nonzero evidence must use its separately registered new run root")
    flags = dict(
        status="FROZEN",
        scope=SCOPE,
        authorized=True,
        mmdev_authorized=False,
        no_automatic_retry=True,
        common_start_replacement=False,
    )
    if any(
        freeze.get(key) != value or type(freeze.get(key)) is not type(value)
        for key, value in flags.items()
    ):
        raise PermissionError("Separate explicit nonzero-recovery freeze is required")
    if (
        hash_json(freeze.get("recipe")) != hash_json(NONZERO_RECIPE)
        or type(freeze.get("seed")) is not int
        or freeze["seed"] != SEED
    ):
        raise PermissionError("Nonzero recipe/seed differs from registration")
    if freeze.get("prior_costs") != PRIOR_COSTS:
        raise PermissionError("Original consumed costs must remain reserved")
    if freeze.get("resources") != dict(
        max_concurrent_gpus=5, allocated_gpu_hours="accounting_only"
    ):
        raise PermissionError("Nonzero resource envelope differs from authorization")
    for key, path in (
        ("pre_inference_freeze_sha256", "manifests/PRE_INFERENCE_FREEZE.json"),
        ("common_start_sha256", "manifests/COMMON_START.json"),
    ):
        if freeze.get(key) != sha256_file(root / path):
            raise PermissionError("Inherited audit identity changed")
    original = read_json(root / "manifests/PRE_INFERENCE_FREEZE.json")
    common = read_json(root / "manifests/COMMON_START.json")
    if (
        original.get("status") != "FROZEN"
        or original.get("model_id") != MODEL_ID
        or original.get("dev_authorized") is not False
        or common.get("status") != "VERIFIED"
    ):
        raise PermissionError("Original model/protocol/common-start identity is not verified")
    if (
        common.get("untouched_base") is not True
        or common.get("adapter_path") is not None
        or common.get("model_hash") != original.get("model_weights_hash")
        or not original.get("model_weights_hash")
    ):
        raise PermissionError("The inherited untouched base is required")
    project = Path(source_root).resolve() if source_root else Path(__file__).resolve().parents[2]
    source_hashes = freeze.get("source_hashes", {})
    required = {"src/mm_core/nonzero_recovery.py", "scripts/mm_core/run_nonzero_recovery.py"}
    old_sources = original.get("source_hashes", {})
    if set(old_sources) != {
        "__init__.py",
        "allocations.py",
        "contracts.py",
        "execution.py",
        "generator.py",
        "inventory.py",
        "release.py",
        "runner.py",
        "scoring.py",
        "training.py",
        "vl_runtime.py",
    }:
        raise PermissionError("The eleven original frozen modules must all be present")
    for name, expected in old_sources.items():
        relative = "src/mm_core/" + name
        required.add(relative)
        if source_hashes.get(relative) != expected:
            raise PermissionError("An original frozen module was changed")
    if not required.issubset(source_hashes):
        raise PermissionError("Nonzero source manifest is incomplete")
    for relative, expected in source_hashes.items():
        if sha256_file(checked_path(project, relative)) != expected:
            raise PermissionError("Frozen implementation changed: " + relative)
    file_hashes = freeze.get("file_hashes", {})
    if "data/questions.jsonl" not in file_hashes:
        raise PermissionError("Registered question pool is missing")
    if file_hashes["data/questions.jsonl"] != original.get("file_hashes", {}).get(
        "data/questions.jsonl"
    ):
        raise PermissionError("Copied questions differ from the original ENGINE_TEST pool")
    for relative, expected in file_hashes.items():
        if sha256_file(checked_path(root, relative)) != expected:
            raise PermissionError("Frozen input changed: " + relative)
    rows = read_jsonl(root / "data/questions.jsonl")
    pool = [row for row in rows if row["split"] == "ENGINE_TEST"]
    stream = deterministic_order(pool, SEED, 8)
    if freeze.get("stream_question_ids") != [row["question_id"] for row in stream]:
        raise PermissionError("Input stream differs from the outcome-independent selection")
    for row in stream:
        if file_hashes.get(row["image_path"]) != row["image_sha256"]:
            raise PermissionError("Selected image must be frozen byte-for-byte")
        import hashlib

        if hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
            raise PermissionError("Question prompt identity changed")
    return freeze, original, common, stream


def require_nonzero_allocation(run_root, *, environ=None, scheduler_query=None):
    root = Path(run_root).resolve()
    env = os.environ if environ is None else environ
    allocation = read_json(root / "manifests/NONZERO_ALLOCATION.json")
    job = env.get("SLURM_JOB_ID")
    if not job or not str(job).isdigit() or env.get("SLURM_RESTART_COUNT", "0") != "0":
        raise PermissionError("A bound non-restarted Slurm job is required")
    required = dict(
        status="BOUND",
        scope=SCOPE,
        run_root=str(root),
        slurm_job_id=str(job),
        freeze_sha256=sha256_file(root / "manifests/NONZERO_FREEZE.json"),
        allocated_gpus=1,
        gpu_model_contains="PRO 6000",
    )
    if any(allocation.get(key) != value for key, value in required.items()):
        raise PermissionError("Nonzero Slurm allocation identity differs")
    import re
    import subprocess

    from .allocations import verify_live_gpu_resources

    if not isinstance(allocation.get("job_name"), str) or not allocation["job_name"]:
        raise PermissionError("Registered Slurm job name is missing")
    if scheduler_query is None:
        scheduler_text = subprocess.run(
            ["scontrol", "show", "job", str(job), "-o"],
            check=True,
            text=True,
            capture_output=True,
            timeout=20,
        ).stdout
    else:
        scheduler_text = scheduler_query(str(job))
    fields = dict(part.split("=", 1) for part in scheduler_text.split() if "=" in part)
    owner = re.fullmatch(r"[^()]+\((\d+)\)", fields.get("UserId", ""))
    restarts = fields.get("RestartCnt", fields.get("Restarts"))
    if (
        fields.get("JobId") != str(job)
        or fields.get("JobName") != allocation["job_name"]
        or fields.get("JobState") != "RUNNING"
        or fields.get("Requeue") != "0"
        or restarts != "0"
        or not owner
        or int(owner[1]) != os.getuid()
    ):
        raise PermissionError("Live Slurm job ownership/restart identity differs")
    resources = verify_live_gpu_resources(fields, 1)
    evidence = dict(
        allocation=allocation,
        scheduler_output=scheduler_text,
        live_resources=resources,
        hostname=__import__("socket").gethostname(),
    )
    atomic_json(root / f"accounting/allocation_checks/{job}_{os.getpid()}.json", evidence)
    return evidence


class NonzeroLedger:
    """Independent physical reservations, including prior-cost global hard limits."""

    def __init__(self, run_root):
        self.root = Path(run_root)
        self.path = self.root / "accounting/COST_LEDGER.jsonl"

    def totals(self):
        result = dict.fromkeys(ADDITIONAL_CAPS, 0)
        result["allocated_gpu_hours"] = 0.0
        for entry in read_jsonl(self.path):
            kind, amount = entry["kind"], entry["amount"]
            if (
                kind not in result
                or isinstance(amount, bool)
                or not isinstance(amount, (int, float))
                or not math.isfinite(amount)
                or amount <= 0
            ):
                raise ValueError("Malformed nonzero cost ledger")
            if kind != "allocated_gpu_hours" and type(amount) is not int:
                raise ValueError("Physical operation ledger counts must be integers")
            result[kind] += amount
        return result

    def reserve(self, kind, count, metadata):
        if kind not in {*ADDITIONAL_CAPS, "allocated_gpu_hours"}:
            raise ValueError("Unknown nonzero accounting kind")
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, float))
            or not math.isfinite(count)
            or count <= 0
        ):
            raise ValueError("Reservation must be finite and positive")
        if kind != "allocated_gpu_hours" and type(count) is not int:
            raise ValueError("Physical operation counts must be integers")
        with locked(self.root / "accounting/NONZERO_BUDGET.lock"):
            after = self.totals()[kind] + count
            if kind in ADDITIONAL_CAPS and (
                after > ADDITIONAL_CAPS[kind] or PRIOR_COSTS[kind] + after > GLOBAL_CAPS[kind]
            ):
                raise RuntimeError("BUDGET_EXHAUSTED: nonzero " + kind)
            append_jsonl(
                self.path,
                dict(
                    time=utc_now(),
                    kind=kind,
                    amount=count,
                    identity=metadata,
                    status="CONSUMED_OR_RESERVED",
                    prior_costs=PRIOR_COSTS,
                ),
            )


def adam_nonzero_evidence(optimizer, logical_step):
    import torch

    states = list(optimizer.state.values())
    if not states:
        raise RuntimeError("Adam state is missing after update")
    first_nonzero = second_nonzero = 0
    steps = []
    for state in states:
        if "exp_avg" not in state or "exp_avg_sq" not in state or "step" not in state:
            raise RuntimeError("Adam moments/step are missing")
        first, second = state["exp_avg"], state["exp_avg_sq"]
        if not bool(torch.isfinite(first).all()) or not bool(torch.isfinite(second).all()):
            raise RuntimeError("Adam moments are nonfinite")
        first_nonzero += int(torch.count_nonzero(first))
        second_nonzero += int(torch.count_nonzero(second))
        steps.append(int(state["step"]))
    if first_nonzero == 0 or second_nonzero == 0 or any(step != logical_step for step in steps):
        raise RuntimeError("Adam nonzero moment/update-step validation failed")
    return dict(
        first_moment_nonzero_elements=first_nonzero,
        second_moment_nonzero_elements=second_nonzero,
        adam_step_values=sorted(set(steps)),
    )


def nonzero_sft_step(runtime, optimizer, scheduler, rows, run_root, *, branch, logical_step):
    import torch

    if len(rows) != 2:
        raise ValueError("Exactly two fixed gold sequences are required")
    runtime.reserve(
        "physical_optimizer_updates", 1, stage=SCOPE, branch=branch, logical_step=logical_step
    )
    before = trainable_state(runtime.model)
    before_hash = state_hash(before)
    optimizer.zero_grad(set_to_none=True)
    losses, lengths, routing = [], [], []
    for row in rows:
        prepared = runtime.prepare(row, run_root)
        tokens = runtime.encode_completion(gold_completion(row))
        score = runtime.sequence_forward(
            prepared, tokens, purpose="nonzero_gold_completion", grad=True
        )
        loss = -score["logprobs"].mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonzero SFT loss is nonfinite")
        losses.append(float(loss.detach()))
        lengths.append(len(tokens))
        routing.append(
            {**prepared["routing"], "training_vision_forward_calls": score["vision_forward_calls"]}
        )
        (loss / 2).backward()
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    if not bool(torch.isfinite(norm)) or float(norm) <= 0:
        raise RuntimeError("NONZERO_UPDATE_NOT_ESTABLISHED: finite positive gradient required")
    optimizer.step()
    scheduler.step()
    after = trainable_state(runtime.model)
    after_hash = state_hash(after)
    delta_max = max(
        float((after[name].double() - before[name].double()).abs().max()) for name in before
    )
    changed = sum(int(torch.count_nonzero(after[name] != before[name])) for name in before)
    if not math.isfinite(delta_max) or delta_max <= 0 or changed == 0 or after_hash == before_hash:
        raise RuntimeError("NONZERO_UPDATE_NOT_ESTABLISHED: parameters did not change")
    moments = adam_nonzero_evidence(optimizer, logical_step)
    return dict(
        logical_step=logical_step,
        question_ids=[row["question_id"] for row in rows],
        sequence_losses=losses,
        completion_token_counts=lengths,
        loss=sum(losses) / 2,
        gradient_norm_before_clip=float(norm),
        parameter_hash_before=before_hash,
        parameter_hash_after=after_hash,
        parameter_delta_max_abs=delta_max,
        changed_parameter_elements=changed,
        image_routing=routing,
        **moments,
    )


def run_nonzero_segment(run_root, branch, start, stop):
    import torch

    root = Path(run_root).resolve()
    _, original, common, stream = verify_nonzero_gate(root)
    allocation_evidence = require_nonzero_allocation(root)
    if (branch, start, stop) not in {
        (b, s, e) for b, s, e in NONZERO_RECIPE["fresh_process_segments"]
    }:
        raise PermissionError("Unregistered recovery segment")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise PermissionError("Frozen CUBLAS determinism configuration is missing")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or "PRO 6000" not in torch.cuda.get_device_name(0).upper()
    ):
        raise PermissionError("Exactly one visible PRO 6000 device is required")
    destination = root / "engineering/nonzero" / branch
    exclusive_json(
        destination / f"SEGMENT_{start}_{stop}_STARTED.json",
        dict(
            status="RUNNING",
            pid=os.getpid(),
            start=start,
            stop=stop,
            time=utc_now(),
            freeze_sha256=sha256_file(root / "manifests/NONZERO_FREEZE.json"),
        ),
    )
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    seed_all(SEED)
    ledger = NonzeroLedger(root)
    runtime = QwenRuntime(
        original["model_path"],
        adapter_path=common.get("adapter_path"),
        dtype=original.get("dtype", "bfloat16"),
        attention_backend=original.get("attention_backend", "eager"),
        account=ledger.reserve,
    )
    runtime.verify_identity(original, common)
    optimizer, scheduler, modules = configure_training(runtime)
    frozen_base = frozen_hash(runtime.model)
    reference = dict(parameters=trainable_state(runtime.model), base_hash=frozen_base)
    stream_hash = hash_json([row["question_id"] for row in stream])
    resumed_rng_verified = None
    if start:
        checkpoint = load_checkpoint(
            destination / "checkpoint_2.pt", runtime, optimizer, scheduler, stream_hash
        )
        reference = checkpoint["reference"]
        resumed_rng_verified = state_hash(capture_rng()) == state_hash(checkpoint["rng"])
        if not resumed_rng_verified or reference["base_hash"] != frozen_base:
            raise RuntimeError("Restored RNG or frozen reference identity differs")
    else:
        seed_all(SEED)
    reference_hash = state_hash(reference)
    exclusive_json(
        destination / f"SEGMENT_{start}_{stop}_IDENTITY.json",
        dict(
            pid=os.getpid(),
            start=start,
            stop=stop,
            freeze_sha256=sha256_file(root / "manifests/NONZERO_FREEZE.json"),
            started_sha256=sha256_file(destination / f"SEGMENT_{start}_{stop}_STARTED.json"),
            allocation_evidence=allocation_evidence,
            device_identity=dict(
                hostname=__import__("socket").gethostname(),
                visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                name=torch.cuda.get_device_name(0),
                uuid=str(getattr(torch.cuda.get_device_properties(0), "uuid", "UNAVAILABLE")),
                total_memory=torch.cuda.get_device_properties(0).total_memory,
            ),
            reference_hash=reference_hash,
            frozen_base_hash=frozen_base,
            stream_hash=stream_hash,
            rng_restored_exact=resumed_rng_verified,
            runtime_identity=runtime.identity,
            **modules,
        ),
    )
    checkpoint_manifests = {}
    for step in range(start, stop):
        rows = stream[step * 2 : (step + 1) * 2]
        rng_before = state_hash(capture_rng())
        metrics = nonzero_sft_step(
            runtime, optimizer, scheduler, rows, root, branch=branch, logical_step=step + 1
        )
        # Preserve the completed update even if the later generation fails.
        append_jsonl(
            destination / "TRAINING_UPDATES.jsonl", {**metrics, "rng_before_step_hash": rng_before}
        )
        # A real independently sampled response exercises post-update generation and RNG.
        seed = random.randrange(2**63 - 1)
        request = dict(
            stage=SCOPE,
            branch=branch,
            logical_step=step + 1,
            seed=seed,
            question_id=rows[0]["question_id"],
            image_sha256=rows[0]["image_sha256"],
            prompt_sha256=rows[0]["prompt_sha256"],
            request_id=hash_json([SCOPE, branch, step, seed]),
            purpose="engineering_only_post_update_sample",
        )
        append_jsonl(root / "raw/REQUESTS.jsonl", {**request, "status": "REQUESTED"})
        raw = runtime.generate(
            rows[0],
            root,
            seed=seed,
            score_fields=False,
            on_completion=lambda record, request=request: append_jsonl(
                destination / "RAW_COMPLETIONS.jsonl", {**request, **record}
            ),
        )
        append_jsonl(
            destination / "STEPS.jsonl",
            {
                **metrics,
                "rng_before_step_hash": rng_before,
                "rng_after_step_hash": state_hash(capture_rng()),
                "generation_seed": seed,
                "generation_tokens": raw["tokens"],
                "generation_raw_text": raw["raw_text"],
            },
        )
        if state_hash(reference) != reference_hash:
            raise RuntimeError("Frozen reference changed")
        if step + 1 in (2, 4):
            checkpoint_path = destination / f"checkpoint_{step + 1}.pt"
            if checkpoint_path.exists():
                raise FileExistsError("Existing recovery checkpoint must not be overwritten")
            saved_state = save_checkpoint(
                destination / f"checkpoint_{step + 1}.pt",
                runtime,
                optimizer,
                scheduler,
                step + 1,
                reference,
                stream_hash=stream_hash,
            )
            checkpoint_manifests[checkpoint_path.name] = checkpoint_evidence(
                checkpoint_path, saved_state
            )
    if frozen_hash(runtime.model) != frozen_base:
        raise RuntimeError("Frozen base/vision/projector parameters changed")
    exclusive_json(
        destination / f"SEGMENT_{start}_{stop}_COMPLETE.json",
        dict(
            status="COMPLETED",
            pid=os.getpid(),
            start=start,
            stop=stop,
            freeze_sha256=sha256_file(root / "manifests/NONZERO_FREEZE.json"),
            started_sha256=sha256_file(destination / f"SEGMENT_{start}_{stop}_STARTED.json"),
            identity_sha256=sha256_file(destination / f"SEGMENT_{start}_{stop}_IDENTITY.json"),
            checkpoint_manifests=checkpoint_manifests,
            records_hashes={
                name: hash_json(
                    [
                        row
                        for row in read_jsonl(destination / name)
                        if start < row["logical_step"] <= stop
                    ]
                )
                for name in ("STEPS.jsonl", "TRAINING_UPDATES.jsonl", "RAW_COMPLETIONS.jsonl")
            },
            physical_updates=stop - start,
            image_calls=runtime.image_calls,
            reference_unchanged=True,
            frozen_base_unchanged=True,
            time=utc_now(),
        ),
    )


CHECKPOINT_FIELDS = {
    "committed_logical_step",
    "parameters",
    "optimizer",
    "scheduler",
    "rng",
    "reference",
    "input_stream_hash",
}


def checkpoint_evidence(path, state):
    if set(state) != CHECKPOINT_FIELDS:
        raise ValueError("Checkpoint fields differ from the complete recovery contract")
    return dict(
        sha256=sha256_file(path),
        state_hash=state_hash(state),
        field_hashes={key: state_hash(value) for key, value in state.items()},
    )


def validate_checkpoint_binding(state, identity, step, stream_hash):
    from types import SimpleNamespace

    if (
        set(state) != CHECKPOINT_FIELDS
        or state["committed_logical_step"] != step
        or state["input_stream_hash"] != stream_hash
    ):
        raise ValueError("Checkpoint step/stream/field binding differs")
    expected_names = set(identity["trainable_names"])
    if not expected_names or set(state["parameters"]) != expected_names:
        raise ValueError("Checkpoint trainable parameters are incomplete")
    if (
        set(state["reference"]) != {"parameters", "base_hash"}
        or set(state["reference"]["parameters"]) != expected_names
        or state_hash(state["reference"]) != identity["reference_hash"]
        or state["reference"]["base_hash"] != identity["frozen_base_hash"]
    ):
        raise ValueError("Checkpoint frozen reference binding differs")
    if set(state["rng"]) != {"python", "numpy", "cpu", "cuda"} or len(state["rng"]["cuda"]) != 1:
        raise ValueError("Checkpoint complete single-GPU RNG state missing")
    if state["scheduler"].get("last_epoch") != step:
        raise ValueError("Checkpoint scheduler step differs")
    adam_nonzero_evidence(SimpleNamespace(state=state["optimizer"]["state"]), step)


def compare_nonzero(run_root):
    import torch

    root = Path(run_root).resolve()
    _, original, _, stream = verify_nonzero_gate(root)
    freeze_hash = sha256_file(root / "manifests/NONZERO_FREEZE.json")
    stream_hash = hash_json([row["question_id"] for row in stream])
    comparisons = {}
    base = root / "engineering/nonzero"
    for branch, start, stop in NONZERO_RECIPE["fresh_process_segments"]:
        folder = base / branch
        started_path = folder / f"SEGMENT_{start}_{stop}_STARTED.json"
        identity_path = folder / f"SEGMENT_{start}_{stop}_IDENTITY.json"
        started, identity = read_json(started_path), read_json(identity_path)
        complete = read_json(folder / f"SEGMENT_{start}_{stop}_COMPLETE.json")
        expected_common = dict(
            freeze_sha256=freeze_hash, start=start, stop=stop, pid=started["pid"]
        )
        bound = all(
            record.get(key) == value
            for record in (started, identity, complete)
            for key, value in expected_common.items()
        )
        bound = bound and all(
            record.get("started_sha256") == sha256_file(started_path)
            for record in (identity, complete)
        )
        bound = bound and complete.get("identity_sha256") == sha256_file(identity_path)
        bound = bound and identity["stream_hash"] == stream_hash
        for key in (
            "model_id",
            "processor_hash",
            "chat_template_hash",
            "generation_config_expanded",
            "linear_kernel_identity",
            "linear_kernel_identity_hash",
        ):
            bound = bound and identity["runtime_identity"].get(key) == original.get(key)
        expected_checkpoints = {f"checkpoint_{step}.pt" for step in (2, 4) if start < step <= stop}
        bound = bound and set(complete["checkpoint_manifests"]) == expected_checkpoints
        for filename in expected_checkpoints:
            path = folder / filename
            expected = complete["checkpoint_manifests"][filename]
            if sha256_file(path) != expected["sha256"]:
                raise ValueError("Checkpoint bytes differ from completed receipt")
            state = torch.load(path, map_location="cpu", weights_only=False)
            validate_checkpoint_binding(
                state, identity, int(filename.split("_")[1].split(".")[0]), stream_hash
            )
            bound = bound and checkpoint_evidence(path, state) == expected
        for filename, expected in complete["records_hashes"].items():
            bound = (
                bound
                and hash_json(
                    [
                        row
                        for row in read_jsonl(folder / filename)
                        if start < row["logical_step"] <= stop
                    ]
                )
                == expected
            )
        comparisons[f"segment_{branch}_{start}_{stop}_provenance"] = bound
    for step in (2, 4):
        states = [
            torch.load(
                base / branch / f"checkpoint_{step}.pt", map_location="cpu", weights_only=False
            )
            for branch in ("continuous", "resumed")
        ]
        comparisons[f"step_{step}_state_keys"] = states[0].keys() == states[1].keys()
        for key in set(states[0]) | set(states[1]):
            comparisons[f"step_{step}_{key}"] = (
                key in states[0]
                and key in states[1]
                and state_hash(states[0][key]) == state_hash(states[1][key])
            )
    steps = [read_jsonl(base / branch / "STEPS.jsonl") for branch in ("continuous", "resumed")]
    comparisons["all_per_step_losses_updates_rng_exact"] = steps[0] == steps[1]
    comparisons["four_logical_updates_each"] = all(
        [row["logical_step"] for row in branch] == [1, 2, 3, 4] for branch in steps
    )
    comparisons["registered_question_slots_exact"] = all(
        row["question_ids"] == [q["question_id"] for q in stream[index * 2 : index * 2 + 2]]
        for branch in steps
        for index, row in enumerate(branch)
    )
    updates = [
        read_jsonl(base / branch / "TRAINING_UPDATES.jsonl") for branch in ("continuous", "resumed")
    ]
    comparisons["persisted_training_updates_exact"] = (
        updates[0] == updates[1] and len(updates[0]) == 4
    )
    comparisons["update_metrics_match_final_steps"] = all(
        all(step_row.get(key) == value for key, value in update.items())
        for branch_updates, branch_steps in zip(updates, steps, strict=True)
        for update, step_row in zip(branch_updates, branch_steps, strict=True)
    )
    comparisons["nonzero_every_physical_update"] = all(
        row["gradient_norm_before_clip"] > 0
        and row["parameter_delta_max_abs"] > 0
        and row["changed_parameter_elements"] > 0
        and row["first_moment_nonzero_elements"] > 0
        and row["second_moment_nonzero_elements"] > 0
        and row["parameter_hash_before"] != row["parameter_hash_after"]
        for branch in steps
        for row in branch
    )
    raw = [
        read_jsonl(base / branch / "RAW_COMPLETIONS.jsonl") for branch in ("continuous", "resumed")
    ]
    keys = (
        "logical_step",
        "seed",
        "question_id",
        "tokens",
        "raw_text",
        "truncated",
        "image_routing",
    )
    comparisons["post_update_raw_samples_exact"] = [
        {key: row[key] for key in keys} for row in raw[0]
    ] == [{key: row[key] for key in keys} for row in raw[1]]
    comparisons["four_samples_each"] = len(raw[0]) == len(raw[1]) == 4
    pids = []
    for branch, start, stop in NONZERO_RECIPE["fresh_process_segments"]:
        completed = read_json(base / branch / f"SEGMENT_{start}_{stop}_COMPLETE.json")
        comparisons[f"segment_{branch}_{start}_{stop}"] = (
            completed["status"] == "COMPLETED"
            and completed["reference_unchanged"]
            and completed["frozen_base_unchanged"]
            and completed["image_calls"] >= (stop - start) * 3
        )
        pids.append(completed["pid"])
    comparisons["three_distinct_worker_pids"] = len(set(pids)) == 3
    initial = [
        read_json(
            base
            / branch
            / (
                "SEGMENT_0_4_IDENTITY.json"
                if branch == "continuous"
                else "SEGMENT_0_2_IDENTITY.json"
            )
        )
        for branch in ("continuous", "resumed")
    ]
    comparisons["initial_reference_and_backend_identity_exact"] = all(
        initial[0][key] == initial[1][key]
        for key in (
            "reference_hash",
            "frozen_base_hash",
            "stream_hash",
            "runtime_identity",
            "device_identity",
        )
    )
    comparisons["resume_rng_restored_exact"] = (
        read_json(base / "resumed/SEGMENT_2_4_IDENTITY.json")["rng_restored_exact"] is True
    )
    totals = NonzeroLedger(root).totals()
    comparisons["physical_counts_exact"] = all(
        totals[key] == value for key, value in ADDITIONAL_CAPS.items()
    )
    receipt = dict(
        status="PASS_NONZERO_SFT_RESUME"
        if all(comparisons.values())
        else "NONZERO_RECOVERY_NOT_READY",
        executed=True,
        scope=SCOPE,
        comparisons=comparisons,
        physical_costs=totals,
        prior_costs=PRIOR_COSTS,
        combined_costs={k: PRIOR_COSTS[k] + totals[k] for k in PRIOR_COSTS},
        exact_no_tolerance=True,
        recipe=NONZERO_RECIPE,
        common_start_replaced=False,
        scientific_grpo_certified=False,
        mmdev_authorized=False,
        claim="Nonzero supervised visual parameter/optimizer recovery; no training benefit claim.",
    )
    exclusive_json(root / "engineering/NONZERO_COMPARE.json", receipt)
    return receipt
