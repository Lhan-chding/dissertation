#!/usr/bin/env python3
"""Read-only, CPU-only verification of a completed MM-CORE evidence directory.

This helper never samples, trains, repairs, or changes registered evidence. Only its
own two report files are written. Checkpoints must be this run's own generated files;
loading their Python/NumPy RNG states uses torch.load(weights_only=False) on CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import random
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DIRECTORIES = (
    "tables",
    "scoring",
    "raw",
    "accounting",
    "manifests",
    "engineering",
    "report",
    "data",
)
REPORT = "report/FINAL_EVIDENCE_VERIFICATION.json"
MANIFEST = "report/FINAL_EVIDENCE_SHA256.json"
SEGMENTS = (("continuous", 0, 4), ("resumed", 0, 2), ("resumed", 2, 4))
CHECKPOINT_KEYS = {
    "committed_logical_step",
    "parameters",
    "optimizer",
    "scheduler",
    "rng",
    "reference",
    "input_stream_hash",
}
REQUIRED_BINDINGS = {
    "manifests/PRE_INFERENCE_FREEZE.json",
    "manifests/COMMON_START.json",
    "manifests/ENVIRONMENT.json",
    "manifests/PROCESSOR_ENVIRONMENT_LOCK.json",
    "data/questions.jsonl",
    "tables/FORMAT_CHECK_REPORT.json",
}


class MissingEvidence(Exception):
    """Expected evidence has not yet been produced or settled."""


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe evidence path")
    root = Path(root).resolve()
    path = root / relative
    # Reject links even inside the run: checkpoint provenance must be unambiguous.
    require(
        not any(p.is_symlink() for p in (path, *path.parents) if p != root.parent),
        f"Symlink evidence is not supported: {relative}",
    )
    path.resolve().relative_to(root)
    return path


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def read_json(root, relative):
    return strict_json(safe_path(root, relative).read_bytes())


def jsonl_bytes(data, label):
    require(not data or data.endswith(b"\n"), f"Incomplete JSONL tail: {label}")
    rows = [strict_json(line) for line in data.splitlines()]
    require(all(isinstance(row, dict) for row in rows), f"Nonobject JSONL row: {label}")
    return rows


def read_jsonl(root, relative):
    return jsonl_bytes(safe_path(root, relative).read_bytes(), relative)


def verify_bindings(root, bindings, required=()):
    require(
        isinstance(bindings, dict) and set(required) <= set(bindings),
        "Missing required bound_files identities",
    )
    for name, expected in bindings.items():
        require(sha256(safe_path(root, name)) == expected, f"Bound file changed: {name}")
    return dict(bindings)


def verify_prefix(root, descriptor, expected_path):
    require(descriptor.get("path") == expected_path, "Wrong registered prefix path")
    length = descriptor.get("bytes")
    require(type(length) is int and length >= 0, "Invalid registered prefix length")
    with safe_path(root, expected_path).open("rb") as handle:
        data = handle.read(length)
    require(
        len(data) == length and hashlib.sha256(data).hexdigest() == descriptor.get("sha256"),
        f"Registered prefix changed: {expected_path}",
    )
    rows = jsonl_bytes(data, expected_path)
    require(len(rows) == descriptor.get("rows"), "Registered prefix row count changed")
    return rows


def frozen_modules(source_root, freeze):
    sys.dont_write_bytecode = True
    source = Path(source_root).resolve()
    if source.name != "mm_core":
        source = source / "mm_core"
    expected = freeze.get("source_hashes", {})
    require(
        expected and set(expected) == {p.name for p in source.glob("*.py")},
        "Frozen source module set differs",
    )
    verify_bindings(source, expected)
    # Load a separate package, avoiding whichever mm_core happens to be on PYTHONPATH.
    name = (
        "_mm_core_final_evidence_"
        + hashlib.sha256((str(source) + json.dumps(expected, sort_keys=True)).encode()).hexdigest()[
            :16
        ]
    )
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, source / "__init__.py", submodule_search_locations=[str(source)]
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
    return {
        module: importlib.import_module(f"{name}.{module}")
        for module in ("execution", "training", "vl_runtime", "release")
    }


def verify_identity(root, modules):
    execution, training, runtime = (modules[key] for key in ("execution", "training", "vl_runtime"))
    freeze = execution._verify_freeze(root)
    common = execution.verify_common_start(root)
    execution.verify_format_report(root, "FORMAT_CHECK")
    engine = read_json(root, "manifests/ENGINE_TEST_FREEZE.json")
    require(
        engine.get("status") == "FROZEN" and engine.get("matching_prior_receipt") is False,
        "ENGINE freeze is not an original registered test",
    )
    require(
        engine.get("pre_freeze_hash") == sha256(root / "manifests/PRE_INFERENCE_FREEZE.json"),
        "ENGINE prefreeze differs",
    )
    require(
        engine.get("recipe") == training.ENGINE_RECIPE and type(engine.get("seed")) is int,
        "ENGINE recipe or seed differs",
    )
    bindings = verify_bindings(root, engine.get("bound_files"), REQUIRED_BINDINGS)
    questions = read_jsonl(root, "data/questions.jsonl")
    require(len({q["question_id"] for q in questions}) == len(questions), "Duplicate question ID")
    stream = training.deterministic_order(
        [q for q in questions if q["split"] == "ENGINE_TEST"], engine["seed"], 8
    )
    ids = [q["question_id"] for q in stream]
    require(
        engine.get("stream_question_ids") == ids
        and engine.get("stream_hash") == runtime.hash_json(ids),
        "ENGINE input stream differs from deterministic frozen questions",
    )
    submission = read_json(root, "manifests/ENGINE_PRE_SUBMISSION.json")
    require(
        submission.get("engine_freeze_hash") == sha256(root / "manifests/ENGINE_TEST_FREEZE.json"),
        "ENGINE pre-submission refers to another freeze",
    )
    verify_bindings(
        root, submission.get("bound_files"), ["tables/MEASUREMENT_AUDIT/SCORING_RECEIPT.json"]
    )
    prefix_ledger = verify_prefix(
        root, submission["pre_engine_ledger_prefix"], "accounting/COST_LEDGER.jsonl"
    )
    prefix_requests = verify_prefix(
        root, submission["pre_engine_requests_prefix"], "raw/REQUESTS.jsonl"
    )
    require(
        not any(row.get("stage") == "ENGINE" for row in prefix_requests),
        "Pre-submission snapshot already contains ENGINE requests",
    )
    require(
        not any(
            row.get("identity", {}).get("stage") == "ENGINE"
            or row.get("identity", {}).get("purpose", "").startswith("engine_")
            for row in prefix_ledger
            if row["kind"] != "allocated_gpu_hours"
        ),
        "Pre-submission snapshot already contains ENGINE work",
    )
    return dict(
        freeze=freeze,
        common=common,
        engine=engine,
        stream=stream,
        bindings=bindings,
        prefix_ledger=prefix_ledger,
        prefix_requests=prefix_requests,
    )


def expected_engine_requests(context, runtime, branch):
    engine, stream = context["engine"], context["stream"]
    rng = random.Random(engine["seed"])
    rows, states = [], {}
    for step in range(4):
        for slot, question in enumerate(stream[step * 2 : (step + 1) * 2]):
            for sample in range(8):
                seed = rng.randrange(2**63 - 1)
                # QwenRuntime.generate calls seed_all(seed), including random.seed.
                rng.seed(seed)
                rows.append(
                    dict(
                        stage="ENGINE",
                        branch=branch,
                        logical_step=step + 1,
                        slot=slot,
                        sample_index=sample,
                        request_id=runtime.hash_json(["ENGINE", branch, step, slot, sample]),
                        seed=seed,
                        question_id=question["question_id"],
                        image_sha256=question["image_sha256"],
                        model_hash=context["common"]["model_hash"],
                        processor_hash=context["freeze"]["processor_hash"],
                    )
                )
        states[step + 1] = rng.getstate()
    return rows, states


def verify_rollouts(root, context, modules):
    training, runtime = modules["training"], modules["vl_runtime"]
    questions = {q["question_id"]: q for q in context["stream"]}
    requests = [r for r in read_jsonl(root, "raw/REQUESTS.jsonl") if r.get("stage") == "ENGINE"]
    require(len({r["request_id"] for r in requests}) == len(requests), "Duplicate ENGINE request")
    request_map = {r["request_id"]: r for r in requests}
    branches = {}
    for branch in ("continuous", "resumed"):
        folder = f"engineering/engine/{branch}"
        raw = read_jsonl(root, f"{folder}/RAW_COMPLETIONS.jsonl")
        rollouts = read_jsonl(root, f"{folder}/ROLLOUTS.jsonl")
        expected, _ = expected_engine_requests(context, runtime, branch)
        require(len(raw) <= 64 and len(rollouts) <= 64, "Extra ENGINE completions/rollouts")
        for index, row in enumerate(raw):
            required = expected[index]
            require(
                all(row.get(key) == value for key, value in required.items()),
                f"ENGINE raw request/seed/stream changed: {branch}/{index}",
            )
            request = request_map.get(row["request_id"])
            require(
                request is not None
                and request.get("status") == "REQUESTED"
                and all(request.get(key) == value for key, value in required.items()),
                "ENGINE completion has no matching registered request",
            )
            tokens = row.get("tokens")
            require(
                isinstance(tokens, list)
                and tokens
                and all(type(t) is int and t >= 0 for t in tokens),
                "Invalid saved ENGINE tokens",
            )
            require(
                row.get("raw_tokens") == tokens
                and row.get("completion_token_count") == len(tokens),
                "ENGINE token copies/count differ",
            )
            question = questions[row["question_id"]]
            routing = row.get("image_routing", {})
            required_routing = dict(
                source_image_sha256=question["image_sha256"],
                processed_pixel_sha256=question["processed_pixel_sha256"],
                processor_hash=context["freeze"]["processor_hash"],
                chat_template_hash=context["freeze"]["chat_template_hash"],
                chat_template_kwargs=runtime.CHAT_TEMPLATE_KWARGS,
                chat_template_kwargs_hash=runtime.hash_json(runtime.CHAT_TEMPLATE_KWARGS),
            )
            require(
                all(routing.get(key) == value for key, value in required_routing.items())
                and type(routing.get("generation_vision_forward_calls")) is int
                and routing["generation_vision_forward_calls"] > 0
                and routing.get("image_token_count", 0) > 0,
                "ENGINE actual image routing identity differs",
            )
        require(len(rollouts) <= len(raw), "Rollout exists without persisted raw completion")
        for index, row in enumerate(rollouts):
            require(
                set(row) == set(raw[index]) | {"reward"}
                and all(row.get(key) == value for key, value in raw[index].items()),
                f"ENGINE raw/rollout differs: {branch}/{index}",
            )
            require(
                type(row.get("reward")) is int
                and row["reward"]
                == training.answer_reward(row["raw_text"], questions[row["question_id"]]),
                "ENGINE reward differs from answer-only exact recomputation",
            )
        if len(raw) != 64 or len(rollouts) != 64:
            raise MissingEvidence(
                f"{branch}: {len(raw)} raw completions, {len(rollouts)} rollouts of 64"
            )
        steps = read_jsonl(root, f"{folder}/STEPS.jsonl")
        require(len(steps) <= 4, "Extra physical ENGINE updates")
        for index, step in enumerate(steps):
            group = rollouts[index * 16 : (index + 1) * 16]
            rewards = [r["reward"] for r in group]
            losses = step.get("sequence_losses", [])
            require(
                step.get("logical_step") == index + 1
                and step.get("question_ids") == [r["question_id"] for r in group]
                and step.get("rewards") == rewards
                and step.get("zero_contrast_groups")
                == [len(set(rewards[i : i + 8])) == 1 for i in (0, 8)],
                "ENGINE step question/reward/normalization inputs differ",
            )
            require(
                len(losses) == 16
                and all(type(x) in (int, float) and math.isfinite(x) for x in losses)
                and step.get("loss") == sum(losses) / 16
                and type(step.get("gradient_norm")) in (int, float)
                and math.isfinite(step["gradient_norm"])
                and step["gradient_norm"] >= 0,
                "ENGINE step losses or gradient norm invalid",
            )
        if len(steps) != 4:
            raise MissingEvidence(f"{branch}: {len(steps)} completed physical updates of 4")
        branches[branch] = rollouts
    require(len(requests) == 128, "ENGINE physical request count differs from 128")
    keys = ("logical_step", "slot", "sample_index", "seed", "question_id", "tokens", "reward")
    require(
        [{k: r[k] for k in keys} for r in branches["continuous"]]
        == [{k: r[k] for k in keys} for r in branches["resumed"]],
        "Continuous/resumed tokens, slots, seeds, or rewards differ",
    )
    return dict(
        rollouts=128,
        branch_counts={k: len(v) for k, v in branches.items()},
        exact_tokens_slots_seeds_rewards=True,
        answer_rewards_recomputed=True,
        raw_text_is_saved_tokenizer_output_not_redecoded=True,
    )


def verify_segments(root, context, modules):
    import torch

    training, runtime = modules["training"], modules["vl_runtime"]
    parent = read_json(root, "engineering/ENGINE_STARTED.json")
    require(type(parent.get("pid")) is int and parent["pid"] > 0, "Invalid ENGINE parent PID")
    freeze_hash = sha256(root / "manifests/ENGINE_TEST_FREEZE.json")
    pids, identities, checkpoints, hashes = [], [], {}, {}
    _, rng_states = expected_engine_requests(context, runtime, "continuous")
    for branch, start, stop in SEGMENTS:
        folder = f"engineering/engine/{branch}"
        stem = f"{folder}/SEGMENT_{start}_{stop}"
        started, identity, completed = (
            read_json(root, f"{stem}_{suffix}.json")
            for suffix in ("STARTED", "IDENTITY", "COMPLETE")
        )
        require(
            started.get("start") == start
            and started.get("stop") == stop
            and started.get("freeze_hash") == freeze_hash,
            "Segment started with different freeze/range",
        )
        pid = started.get("pid")
        require(
            type(pid) is int and pid > 0 and pid != parent["pid"] and pid not in pids,
            "ENGINE segments do not record distinct fresh child PIDs",
        )
        pids.append(pid)
        require(
            identity.get("stream_hash") == context["engine"]["stream_hash"],
            "Segment stream differs",
        )
        modules_selected = training.language_qv_modules(identity.get("target_modules", []))
        require(
            identity.get("target_modules") == modules_selected
            and identity.get("trainable_dtype") == "float32"
            and identity.get("model_forward_mode") == "eval_dropout_zero_grad_enabled",
            "Segment module/dtype/mode identity differs",
        )
        require(
            completed.get("status") == "COMPLETED"
            and completed.get("physical_updates") == stop - start
            and completed.get("reference_unchanged") is True
            and completed.get("frozen_base_unchanged") is True
            and type(completed.get("image_calls")) is int
            and completed["image_calls"] > 0,
            "Segment completion is not verified",
        )
        checkpoint_path = safe_path(root, f"{folder}/checkpoint_{stop}.pt")
        digest = sha256(checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        require(sha256(checkpoint_path) == digest, "Checkpoint changed while loading")
        require(
            isinstance(state, dict) and set(state) == CHECKPOINT_KEYS,
            "Checkpoint field set differs",
        )
        require(
            state["committed_logical_step"] == stop
            and state["input_stream_hash"] == identity["stream_hash"],
            "Checkpoint step/stream identity differs",
        )
        require(
            training.state_hash(state) == completed.get("final_state_hash"),
            "Actual checkpoint state differs from completion hash",
        )
        require(
            training.state_hash(state["reference"]) == identity.get("reference_hash"),
            "Checkpoint reference differs from segment identity",
        )
        names = identity.get("trainable_names")
        require(
            isinstance(names, list)
            and len(names) == len(set(names))
            and len(names) == 32
            and set(state["parameters"]) == set(names) == set(state["reference"]["parameters"])
            and all(
                sum(name.endswith(f"{module}.lora_{kind}.default.weight") for name in names) == 1
                for module in modules_selected
                for kind in ("A", "B")
            ),
            "Checkpoint trainable parameter names differ",
        )
        require(
            set(state["reference"]) == {"parameters", "base_hash"}
            and isinstance(state["reference"]["base_hash"], str)
            and len(state["reference"]["base_hash"]) == 64,
            "Checkpoint frozen reference identity missing",
        )
        require(
            all(
                torch.is_tensor(value)
                and value.device.type == "cpu"
                and value.dtype == torch.float32
                and bool(torch.isfinite(value).all())
                for value in state["parameters"].values()
            ),
            "Checkpoint trainable tensors are not finite CPU float32",
        )
        require(
            set(state["rng"]) == {"python", "numpy", "cpu", "cuda"}
            and training.state_hash(state["rng"]["python"])
            == training.state_hash(rng_states[stop]),
            "Checkpoint Python RNG differs from exact recorded sampling trajectory",
        )
        optimizer = state["optimizer"]
        groups = optimizer.get("param_groups", [])
        require(
            len(groups) == 1 and len(groups[0].get("params", [])) == len(names),
            "Checkpoint optimizer parameter coverage differs",
        )
        group = groups[0]
        require(
            group.get("lr") == 1e-5
            and list(group.get("betas", [])) == [0.9, 0.999]
            and group.get("eps") == 1e-8
            and group.get("weight_decay") == 0
            and group.get("amsgrad") is False
            and group.get("maximize") is False,
            "Checkpoint AdamW recipe differs",
        )
        require(
            set(optimizer.get("state", {})) == set(group["params"]),
            "Checkpoint optimizer moments are missing",
        )
        for name, parameter_id in zip(names, group["params"], strict=True):
            moments = optimizer["state"][parameter_id]
            require(
                set(moments) == {"step", "exp_avg", "exp_avg_sq"}
                and float(moments["step"]) == stop
                and all(
                    torch.is_tensor(moments[k])
                    and moments[k].shape == state["parameters"][name].shape
                    and bool(torch.isfinite(moments[k]).all())
                    for k in ("exp_avg", "exp_avg_sq")
                ),
                "Checkpoint optimizer step/moment tensor differs",
            )
        scheduler = state["scheduler"]
        require(
            scheduler.get("last_epoch") == stop
            and scheduler.get("_step_count") == stop + 1
            and scheduler.get("base_lrs") == [1e-5]
            and scheduler.get("_last_lr") == [1e-5],
            "Checkpoint scheduler does not record the expected physical updates",
        )
        require(
            torch.is_tensor(state["rng"]["cpu"])
            and state["rng"]["cpu"].dtype == torch.uint8
            and isinstance(state["rng"]["cuda"], list)
            and len(state["rng"]["cuda"]) == 1
            and all(
                torch.is_tensor(t) and t.device.type == "cpu" and t.dtype == torch.uint8
                for t in state["rng"]["cuda"]
            ),
            "Checkpoint CPU/CUDA RNG evidence missing",
        )
        key = f"{branch}_{start}_{stop}"
        hashes[key] = dict(
            file_sha256=digest,
            state_sha256=training.state_hash(state),
            fields={k: training.state_hash(v) for k, v in state.items()},
        )
        checkpoints[key] = state
        identities.append(identity)
    require(
        all(identity == identities[0] for identity in identities),
        "Segment reference/module identities differ",
    )
    left, right = checkpoints["continuous_0_4"], checkpoints["resumed_2_4"]
    comparisons = {
        key: training.state_hash(left[key]) == training.state_hash(right[key])
        for key in CHECKPOINT_KEYS
    }
    require(all(comparisons.values()), "Actual continuous/resumed checkpoint fields differ")
    receipt = read_json(root, "engineering/ENGINE_COMPARE.json")
    expected = dict(
        comparisons, raw_tokens_slots_seeds_rewards=True, rollout_count=True, image_calls=True
    )
    require(
        receipt.get("status") == "PASS"
        and receipt.get("executed") is True
        and receipt.get("comparisons") == expected
        and receipt.get("physical_updates") == 8
        and receipt.get("rollouts") == 128
        and receipt.get("exact_comparison") is True
        and receipt.get("tolerance") is None
        and receipt.get("recipe") == training.ENGINE_RECIPE
        and receipt.get("engine_model_replaces_common_start") is False
        and receipt.get("scientific_response_evaluation") is False,
        "ENGINE_COMPARE does not agree with independent checkpoint verification",
    )
    return dict(
        checkpoints=hashes,
        comparisons=comparisons,
        recorded_child_pids=pids,
        process_evidence="distinct PIDs in immutable start receipts; no live OS attestation",
        checkpoint_loading="CPU only; own run-generated checkpoints",
    )


def physical_key(row):
    return row.get("stage"), row["request_id"], row.get("attempt_id", "ORIGINAL")


def verify_terminal(root, record):
    """Recompute charged device time from saved Slurm rows, including requeue epochs."""
    terminal = record["terminal"]
    job_id = record["job_id"]
    states = {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
    }
    require(
        terminal.get("state") in states
        and terminal.get("squeue_absent") is True
        and terminal.get("job_id") == job_id
        and terminal.get("gpus") == record["gpus"]
        and type(terminal.get("gpus")) is int
        and type(terminal.get("elapsed_seconds")) is int
        and terminal["elapsed_seconds"] >= 0,
        "Allocation terminal state/resource identity invalid",
    )
    require(
        job_id not in terminal.get("squeue_stdout", "").split(),
        "Terminal job remains in saved squeue",
    )
    receipt = safe_path(root, f"accounting/scheduler/{job_id}.json")
    require(
        sha256(receipt) == record.get("terminal_receipt_sha256")
        and strict_json(receipt.read_bytes()) == terminal,
        "Terminal scheduler receipt hash/content differs",
    )
    command = terminal.get("command", [])
    require(
        command
        and command[0] == "sacct"
        and "-j" in command
        and command[command.index("-j") + 1] == job_id,
        "Scheduler command identity differs",
    )
    formats = [word.split("=", 1)[1] for word in command if word.startswith("--format=")]
    if "-o" in command:
        formats.append(command[command.index("-o") + 1])
    require(len(formats) == 1, "Scheduler column schema missing or ambiguous")
    columns = [word.split("%", 1)[0] for word in formats[0].split(",")]
    expected = {"JobIDRaw", "User", "JobName", "State", "ElapsedRaw", "Start", "End", "AllocTRES"}
    require(len(columns) == 8 and set(columns) == expected, "Unknown scheduler column schema")
    rows = []
    for line in terminal.get("sacct_stdout", "").splitlines():
        if not line.strip():
            continue
        values = line.strip().split("|")
        if len(values) == 9 and values[-1] == "":
            values.pop()
        require(len(values) == len(columns), "Malformed scheduler epoch row")
        rows.append(values)
    require(rows and (len(rows) == 1 or "-D" in command), "Unregistered duplicate scheduler epoch")
    if "epochs" in terminal:
        require(terminal["epochs"] == rows, "Saved epoch array differs from raw sacct rows")
    elapsed, intervals = 0, []
    for values in rows:
        row = dict(zip(columns, values, strict=True))
        state = row["State"].split()[0]
        require(
            row["JobIDRaw"] == job_id
            and row["User"] == terminal.get("owner")
            and row["JobName"] == "mmcore-" + record["allocation_key"]
            and state in states
            and row["ElapsedRaw"].isdigit(),
            "Scheduler epoch owner/job/state identity differs",
        )
        seconds = int(row["ElapsedRaw"])
        tres = dict(item.split("=", 1) for item in row["AllocTRES"].split(",") if "=" in item)
        pending_cancel = (
            seconds == 0
            and row["Start"] in {"None", "Unknown"}
            and state == "CANCELLED"
            and not tres
        )
        if not pending_cancel:
            require(
                tres.get("gres/gpu:pro6000") == str(record["gpus"])
                and tres.get("gres/gpu") == str(record["gpus"])
                and tres.get("node") == "1",
                "Scheduler epoch allocated GPU/node count differs",
            )
            start, end = (datetime.fromisoformat(row[k]) for k in ("Start", "End"))
            require(end >= start, "Scheduler epoch time is reversed")
            if end > start:
                intervals.append((start.isoformat(), end.isoformat(), record["gpus"]))
        elapsed += seconds
    require(
        elapsed == terminal["elapsed_seconds"]
        and dict(zip(columns, rows[-1], strict=True))["State"].split()[0] == terminal["state"],
        "Scheduler aggregate elapsed time/final state differs from all original epochs",
    )
    return dict(
        elapsed_seconds=elapsed,
        gpu_hours=elapsed * record["gpus"] / 3600,
        epoch_count=len(rows),
        intervals=intervals,
    )


def verify_accounting(root, context):
    ledger = read_jsonl(root, "accounting/COST_LEDGER.jsonl")
    requests = read_jsonl(root, "raw/REQUESTS.jsonl")
    totals = dict.fromkeys(
        (
            "completion_attempts",
            "extra_forward_sequences",
            "physical_optimizer_updates",
            "allocated_gpu_hours",
        ),
        0,
    )
    reserved = Counter()
    for row in ledger:
        kind, amount = row["kind"], row["amount"]
        require(
            kind in totals
            and type(amount) in (int, float)
            and math.isfinite(amount)
            and amount > 0,
            "Invalid or refunded ledger entry",
        )
        require(kind == "allocated_gpu_hours" or int(amount) == amount, "Noninteger physical cost")
        totals[kind] += amount
        if kind == "completion_attempts":
            reserved[(row["identity"].get("question_id"), row["identity"].get("seed"))] += amount
    generation = [r for r in requests if r.get("attempt_kind") != "SCORING_ONLY"]
    require(
        len({physical_key(r) for r in generation}) == len(generation),
        "Duplicate physical generation request",
    )
    requested = Counter((r["question_id"], r["seed"]) for r in generation)
    require(
        reserved == requested,
        "Completion ledger differs from ALL physical generation requests "
        "including failures/retries",
    )
    saved = []
    for path in sorted((root / "raw").glob("*/outputs_*.jsonl")):
        saved.extend(
            r for r in read_jsonl(root, path.relative_to(root)) if r.get("status") == "completed"
        )
    for branch in ("continuous", "resumed"):
        path = root / f"engineering/engine/{branch}/RAW_COMPLETIONS.jsonl"
        if path.exists():
            saved.extend(read_jsonl(root, path.relative_to(root)))
    require(
        len({physical_key(r) for r in saved}) == len(saved), "Duplicate saved completion attempt"
    )
    request_map = {physical_key(r): r for r in generation}
    for row in saved:
        request = request_map.get(physical_key(row))
        require(
            request is not None
            and (row["question_id"], row["seed"]) == (request["question_id"], request["seed"]),
            "Saved completion has no matching physical request",
        )
    saved_keys = {physical_key(r) for r in saved}
    unknown = [
        dict(
            request_id=r["request_id"],
            attempt_id=r.get("attempt_id"),
            stage=r["stage"],
            question_id=r["question_id"],
            seed=r["seed"],
        )
        for r in generation
        if physical_key(r) not in saved_keys
    ]
    panel_raw = {physical_key(r): r for r in saved if r.get("stage") != "ENGINE"}
    field_forwards, score_keys = Counter(), set()
    for path in sorted((root / "raw").glob("*/scores_*.jsonl")):
        for row in read_jsonl(root, path.relative_to(root)):
            key = physical_key(row)
            require(
                key in panel_raw and key not in score_keys,
                "Field score has no unique saved completion",
            )
            raw = panel_raw[key]
            require(
                row.get("status") == "scored"
                and all(row.get(k) == v for k, v in raw.items() if k != "status"),
                "Field-score sidecar changed the saved completion",
            )
            score_keys.add(key)
            for field, purpose in (
                ("self_field_surprisal", "generated_tokens_actual_self_prefix"),
                (
                    "gold_teacher_forced_field_nll",
                    "gold_completion_answer_prefix_contains_gold_readings",
                ),
            ):
                if row.get(field) is not None:
                    require(isinstance(row[field], dict), "Invalid field score evidence")
                    field_forwards[purpose] += 1
    charged_field = Counter()
    for row in ledger:
        purpose = row.get("identity", {}).get("purpose")
        if row["kind"] == "extra_forward_sequences" and purpose in {
            "generated_tokens_actual_self_prefix",
            "gold_completion_answer_prefix_contains_gold_readings",
        }:
            charged_field[purpose] += row["amount"]
    require(
        not (field_forwards - charged_field), "Saved field scores exceed charged forward sequences"
    )
    charged_without_sidecar = dict(charged_field - field_forwards)
    prefixes = (context["prefix_ledger"], context["prefix_requests"])
    require(
        ledger[: len(prefixes[0])] == prefixes[0] and requests[: len(prefixes[1])] == prefixes[1],
        "Pre-engine accounting/request prefix changed",
    )
    delta = ledger[len(prefixes[0]) :]
    engine_requests = requests[len(prefixes[1]) :]
    require(
        all(
            r.get("stage") == "ENGINE" and r.get("attempt_kind") != "SCORING_ONLY"
            for r in engine_requests
        ),
        "Unregistered work after ENGINE pre-submission snapshot",
    )
    forward = Counter()
    updates = Counter()
    delta_completions = Counter()
    for row in delta:
        kind, identity = row["kind"], row["identity"]
        if kind == "extra_forward_sequences":
            forward[identity.get("purpose")] += row["amount"]
        elif kind == "physical_optimizer_updates":
            require(identity.get("stage") == "ENGINE", "Non-ENGINE updates after pre-submission")
            updates[(identity.get("branch"), identity.get("logical_step"))] += row["amount"]
        elif kind == "completion_attempts":
            delta_completions[(identity.get("question_id"), identity.get("seed"))] += row["amount"]
    require(
        delta_completions == Counter((r["question_id"], r["seed"]) for r in engine_requests),
        "ENGINE completion accounting mismatch",
    )
    expected_updates = Counter(
        {(branch, step): 1 for branch in ("continuous", "resumed") for step in range(1, 5)}
    )
    complete = (
        len(engine_requests) == 128
        and forward == Counter(engine_behavior_logprob=128, engine_policy_gradient=128)
        and updates == expected_updates
    )
    require(
        len(engine_requests) <= 128
        and not (updates - expected_updates)
        and not (forward - Counter(engine_behavior_logprob=128, engine_policy_gradient=128)),
        "Extra or unregistered ENGINE cost",
    )
    allocations = [
        read_json(root, p.relative_to(root))
        for p in sorted((root / "accounting/allocations").glob("*.json"))
    ]
    unsettled = [r["allocation_key"] for r in allocations if r.get("status") != "TERMINAL_VERIFIED"]
    verified_hours, all_intervals, scheduler_evidence = 0, [], {}
    require(
        len({r["allocation_key"] for r in allocations}) == len(allocations),
        "Duplicate allocation key",
    )
    jobs = [r["job_id"] for r in allocations if r.get("job_id") is not None]
    require(len(set(jobs)) == len(jobs), "Duplicate job ID across allocation records")
    for record in allocations:
        require(
            type(record.get("gpus")) is int
            and 1 <= record["gpus"] <= 5
            and type(record.get("seconds")) is int
            and record["seconds"] > 0,
            "Invalid registered allocation resources",
        )
        require(
            record.get("pre_freeze_hash") == sha256(root / "manifests/PRE_INFERENCE_FREEZE.json"),
            "Allocation prefreeze differs",
        )
        charged = [
            r
            for r in ledger
            if r["kind"] == "allocated_gpu_hours"
            and r.get("identity", {}).get("allocation_key") == record["allocation_key"]
        ]
        identity = {key: record[key] for key in ("allocation_key", "stage", "gpus", "seconds")}
        require(
            len(charged) == 1
            and charged[0]["identity"] == identity
            and charged[0]["amount"] == record["gpus"] * record["seconds"] / 3600,
            "Allocation estimate ledger identity differs (hours are accounting only)",
        )
        if record.get("status") == "TERMINAL_VERIFIED":
            receipt = verify_terminal(root, record)
            scheduler_evidence[record["allocation_key"]] = receipt
            verified_hours += receipt["gpu_hours"]
            all_intervals.extend(receipt["intervals"])
    events = sorted(
        (time, change)
        for start, end, gpus in all_intervals
        for time, change in ((start, gpus), (end, -gpus))
    )
    current, maximum = 0, 0
    for _, change in events:
        current += change
        maximum = max(maximum, current)
    require(maximum <= 5, "Saved scheduler epochs exceed the five-GPU concurrency ceiling")
    if set(panel_raw) != score_keys:
        complete = False
    if not allocations:
        complete = False
    final_path = root / "report/FINAL_AUDIT_STATUS.json"
    if final_path.exists():
        final = read_json(root, str(final_path.relative_to(root)))
        require(
            final.get("budget_reservations") == totals
            and final.get("verified_allocated_gpu_hours") == verified_hours
            and final.get("unknown_allocations") == unsettled
            and final.get("completed_responses") == len(panel_raw),
            "Final report statistics differ from raw evidence/accounting ledger",
        )
    return dict(
        status="PASS" if complete and not unsettled else "INCOMPLETE",
        reservations=totals,
        saved_completions=len(saved),
        generation_requests=len(generation),
        charged_attempts_without_saved_completion=unknown,
        saved_field_score_forwards=dict(field_forwards),
        charged_field_forwards_without_completed_sidecar=charged_without_sidecar,
        scoring_only_requests=sum(r.get("attempt_kind") == "SCORING_ONLY" for r in requests),
        pre_engine_reservations={
            k: sum(r["amount"] for r in prefixes[0] if r["kind"] == k) for k in totals
        },
        engine_extra_forwards=dict(forward),
        engine_physical_updates=sum(updates.values()),
        engine_generation_requests=len(engine_requests),
        unknown_allocations=unsettled,
        verified_allocated_gpu_hours=verified_hours,
        scheduler_evidence=scheduler_evidence,
        maximum_registered_concurrent_gpus=maximum,
        gpu_hours_are_execution_gate=False,
        cost_semantics=(
            "All consumed-or-reserved costs retained, including unknown and failed work; "
            "a reservation is not proof of a successful forward."
        ),
    )


def verify_release(root, modules):
    execution, release = modules["execution"], modules["release"]
    freeze = execution._verify_freeze(root)
    common = execution.verify_common_start(root)
    fmt = execution.verify_format_report(root, "FORMAT_CHECK")
    receipt = read_json(root, "tables/MEASUREMENT_AUDIT/SCORING_RECEIPT.json")
    plan = read_json(root, "manifests/STAGE_PLAN_MEASUREMENT_AUDIT.json")
    integrity, complete = release.release_integrity(root, freeze, common, fmt, receipt, plan)
    require(
        integrity["status"] == "PASS", f"Release evidence integrity failed: {integrity['issues']}"
    )
    if (
        not complete
        or receipt.get("status") != "SCORED"
        or receipt.get("scored_record_count") != 1536
    ):
        raise MissingEvidence(
            "Measurement audit is not completely scored for 1536 registered slots"
        )
    final = read_json(root, "report/FINAL_AUDIT_STATUS.json")
    require(
        final.get("dev_authorized") is False
        and final.get("terminal_state") == "STOP_FOR_REVIEW"
        and final.get("gpu_hours_are_execution_gate") is False,
        "Final report changed authorization/accounting boundary",
    )
    if final.get("status") != "READY_FOR_DEV_REVIEW" or final.get("engine_status") != "PASS":
        raise MissingEvidence(
            "Original final report has not reached READY_FOR_DEV_REVIEW with ENGINE PASS"
        )
    return dict(
        measurement_complete=True,
        measurement_scored=1536,
        release_status=final.get("status"),
        original_engine_status=final.get("engine_status"),
        integrity=integrity,
    )


def inventory(root):
    result = []
    for directory in DIRECTORIES:
        safe_path(root, directory)
        for path in sorted((root / directory).rglob("*")):
            relative = str(path.relative_to(root))
            if relative in {REPORT, MANIFEST}:
                continue
            safe_path(root, relative)
            if path.is_dir():
                continue
            require(path.is_file(), f"Nonregular evidence file: {relative}")
            before = path.stat()
            digest = sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise MissingEvidence(f"Evidence changed during inventory: {relative}")
            result.append(
                dict(
                    path=relative,
                    bytes=after.st_size,
                    sha256=digest,
                    content="hash_only"
                    if path.suffix in {".pt", ".safetensors"}
                    else "evidence_file",
                )
            )
    return result


def audit(root, source_root):
    root = Path(root).resolve()
    report = dict(
        schema_version=1,
        generated_at=datetime.now(timezone.utc).isoformat(),
        status="INCOMPLETE",
        run_root=str(root),
        source_root=str(Path(source_root).resolve()),
        helper_sha256=sha256(__file__),
        checks={},
        issues=[],
        new_model_calls=0,
        new_training_updates=0,
        dev_authorized=False,
        scope=(
            "Independent evidence verification; never authorizes DEV or rewrites "
            "original PASS/failure receipts"
        ),
    )

    def run_check(name, callback):
        try:
            value = callback()
        except (MissingEvidence, FileNotFoundError, ModuleNotFoundError) as error:
            report["issues"].append(
                dict(
                    check=name,
                    severity="INCOMPLETE",
                    error_type=type(error).__name__,
                    detail=str(error),
                )
            )
            report["checks"][name] = dict(status="INCOMPLETE")
            return None
        except Exception as error:
            report["issues"].append(
                dict(
                    check=name, severity="FAIL", error_type=type(error).__name__, detail=str(error)
                )
            )
            report["checks"][name] = dict(status="FAIL")
            return None
        status = value.get("status") if isinstance(value, dict) else None
        status = status if status in {"PASS", "INCOMPLETE", "FAIL"} else "PASS"
        report["checks"][name] = dict(status=status, result=value)
        if status != "PASS":
            report["issues"].append(
                dict(check=name, severity=status, detail="Evidence remains partial or unsettled")
            )
        return value

    before = run_check("inventory_before", lambda: inventory(root))
    freeze = run_check(
        "read_prefreeze", lambda: read_json(root, "manifests/PRE_INFERENCE_FREEZE.json")
    )
    modules = None
    if freeze is not None:
        try:
            modules = frozen_modules(source_root, freeze)
            report["checks"]["frozen_source"] = dict(
                status="PASS", source_hashes=freeze["source_hashes"]
            )
        except Exception as error:
            severity = "INCOMPLETE" if isinstance(error, FileNotFoundError) else "FAIL"
            report["checks"]["frozen_source"] = dict(status=severity)
            report["issues"].append(
                dict(check="frozen_source", severity=severity, detail=str(error))
            )
    if modules:
        context = run_check("engine_identity", lambda: verify_identity(root, modules))
        # Avoid embedding source manifests, entire questions, and original ledgers in the report.
        if context is not None:
            report["checks"]["engine_identity"]["result"] = dict(
                bound_files=context["bindings"],
                stream_question_ids=context["engine"]["stream_question_ids"],
                stream_hash=context["engine"]["stream_hash"],
            )
            run_check("engine_rollouts", lambda: verify_rollouts(root, context, modules))
            run_check("engine_checkpoints", lambda: verify_segments(root, context, modules))
            run_check("accounting", lambda: verify_accounting(root, context))
        run_check("release", lambda: verify_release(root, modules))
        run_check(
            "frozen_source_after",
            lambda: verify_bindings(
                Path(modules["execution"].__file__).parent, freeze["source_hashes"]
            ),
        )
    after = run_check("inventory_after", lambda: inventory(root))
    if before is not None and after is not None and before != after:
        report["issues"].append(
            dict(
                check="stable_snapshot",
                severity="INCOMPLETE",
                detail="Evidence changed during verification; rerun only after all writers stop",
            )
        )
    # Inventory is delivered once in its dedicated manifest, rather than twice in the report.
    for name in ("inventory_before", "inventory_after"):
        if "result" in report["checks"].get(name, {}):
            report["checks"][name]["result"] = dict(
                file_count=len(report["checks"][name]["result"])
            )
    report["checks"].get("read_prefreeze", {}).pop("result", None)
    severities = {i["severity"] for i in report["issues"]}
    report["status"] = "FAIL" if "FAIL" in severities else "INCOMPLETE" if severities else "PASS"
    return report, dict(
        schema_version=1,
        directories=list(DIRECTORIES),
        excluded=[REPORT, MANIFEST],
        files=after or before or [],
        stable_snapshot=before is not None and before == after,
        status=report["status"],
    )


def atomic_report(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".final-evidence-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="Frozen src directory, or its mm_core package directory",
    )
    args = parser.parse_args()
    report, manifest = audit(args.run_root, args.source_root)
    root = args.run_root.resolve()
    atomic_report(safe_path(root, MANIFEST), manifest)
    report["sha256_manifest"] = dict(path=MANIFEST, sha256=sha256(root / MANIFEST))
    atomic_report(safe_path(root, REPORT), report)
    print(
        json.dumps(
            dict(status=report["status"], report=str(root / REPORT), manifest=str(root / MANIFEST))
        )
    )
    return {"PASS": 0, "INCOMPLETE": 2, "FAIL": 1}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
