"""Locked, original-token F6 rollout persistence; no update or scientific gate.

collect_step requires an existing RunTransactions lock and committed checkpoint.
Its registration_ref must equal identity.rollout_registration. The registration
binds identity without that self-reference, split, master_seed, the full schedule,
draw_seeds[step][group][draw], shuffle_seeds[step][group], and public/private/
schedule file references from the frozen DATA manifest. Registration is supplied
by a future phase-freeze layer; this module never creates authorization.

A STARTED group lacking a published COMPLETE marker is ambiguous even with eight
raw rows. It is preserved and blocked, never retried. Completed groups preserve
the RNG after the generation call returned, not an inferred draw-seed state.
Only same-run interruption recovery is implemented. The independently registered
step4-to12 recovery lane and its 1024 new calls require a separate upper layer;
this module never imports another lane's completed responses or changes identity.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import uuid
from pathlib import PurePosixPath

from mm_core.training import capture_rng, restore_rng, state_hash, trainable_state
from mm_core.vl_runtime import hash_json
from srf6_ref.core import FAMILIES, seed_for
from srf6_ref.planning import ARMS

from .common import digest
from .data import public_prompt
from .measurement import score_response
from .planning import draw_seed, root_schedule
from .training import build_credit_group, trajectory_hash
from .transactions import (
    RunTransactions,
    _canonical,
    _fsync_dir,
    _plain_path,
    _read_bytes,
    _require,
    _sha,
)

FORMAT = "srf6-original-token-rollout-v1"


class IncompleteRollout(RuntimeError):
    """Preserved partial or unknown generation requires diagnosis, not resampling."""


def _path(root, relative):
    _require(isinstance(relative, str) and bool(relative), "Relative evidence path required")
    part = PurePosixPath(relative)
    _require(
        not part.is_absolute()
        and ".." not in part.parts
        and part.as_posix() == relative
        and relative != ".",
        "Noncanonical rollout path",
    )
    path = _plain_path(root / relative)
    _require(path.is_relative_to(root), "Rollout path escaped its root")
    return path


def _ref(root, path, data=None):
    data = _read_bytes(path) if data is None else data
    return {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(data).hexdigest()}


def _bound(tx, root, reference, *, kind="json"):
    _require(
        isinstance(reference, dict)
        and set(reference) == {"path", "sha256"}
        and _sha(reference["sha256"]),
        "Content-bound file reference required",
    )
    path = _path(root, reference["path"])
    info = path.stat()
    stamp = (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino)
    key = (str(path), reference["sha256"], kind)
    cache = getattr(tx, "_srf6_rollout_content_cache", None)
    if cache is None:
        cache = tx._srf6_rollout_content_cache = {}
    if key in cache and cache[key][0] == stamp:
        return cache[key][1]
    data = _read_bytes(path)
    _require(
        hashlib.sha256(data).hexdigest() == reference["sha256"],
        "Frozen rollout input content differs",
    )
    if kind == "jsonl":
        value = [json.loads(line) for line in data.splitlines() if line.strip()]
    elif kind == "bytes":
        value = data
    else:
        value = json.loads(data)
    cache[key] = (stamp, value)
    return value


def _publish(path, data):
    path = _plain_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(_read_bytes(path) == data, "Immutable rollout artifact differs")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    # No replace: a published artifact is immutable, including after interruption.
    os.link(temporary, path)
    temporary.unlink()
    _fsync_dir(path.parent)


def _json(path, value):
    _publish(path, _canonical(value) + b"\n")


def _append_raw(path, value):
    # Preserve actual nonfinite backend failures as JSON's explicit NaN/Infinity
    # tokens. Such rows can never receive a COMPLETE marker or enter training.
    payload = json.dumps(value, sort_keys=True, allow_nan=True).encode() + b"\n"
    fd = os.open(_plain_path(path), os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _raw(path):
    try:
        return [json.loads(line) for line in _read_bytes(path).splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise IncompleteRollout("Unreadable/partial raw evidence preserved") from exc


def _save_rng(tx, path, rng):
    import torch

    numpy = rng["numpy"]
    payload = dict(
        python_rng=rng["python"],
        numpy_rng=[numpy[0], numpy[1].tolist(), *numpy[2:]],
        torch_rng=rng["cpu"],
        cuda_rng=rng["cuda"],
    )
    if path.exists():
        reference = _ref(tx.directory, path)
        _require(
            state_hash(_load_rng(tx, reference)) == state_hash(rng),
            "Immutable RNG snapshot differs",
        )
    else:
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        _publish(path, buffer.getvalue())
        reference = _ref(tx.directory, path, buffer.getvalue())
    return reference


def _load_rng(tx, reference):
    import numpy as np
    import torch

    data = _bound(tx, tx.directory, reference, kind="bytes")
    value = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    _require(
        set(value) == {"python_rng", "numpy_rng", "torch_rng", "cuda_rng"},
        "Incomplete RNG snapshot",
    )
    numpy = value["numpy_rng"]
    result = dict(
        python=value["python_rng"],
        numpy=(numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]),
        cpu=value["torch_rng"],
        cuda=value["cuda_rng"],
    )
    _require(
        len(result["cuda"]) == (torch.cuda.device_count() if torch.cuda.is_available() else 0),
        "Rollout RNG device count differs",
    )
    return result


def _v3_overlay(tx, root):
    """Check bound V3 projection identity without granting a live session."""
    from . import pilot_v3_inputs as projection

    identity = tx.identity
    reference = identity.get("input_overlay")
    _require(
        isinstance(reference, dict)
        and reference.get("path") == f"{projection.DIRECTORY}/{projection.MANIFEST_FILE}"
        and reference.get("sha256") == identity.get("input_overlay_sha256")
        and _sha(reference.get("sha256")),
        "Explicit registered V3 overlay identity required",
    )
    overlay = _bound(tx, root, reference)
    _require(
        overlay.get("schema") == projection.SCHEMA
        and overlay.get("phase") == identity["phase"] == "PILOT_V3"
        and overlay.get("phase_identity")
        == dict(path="PHASES/PILOT_V3/IDENTITY.json", sha256=identity["phase_identity_sha256"])
        and overlay.get("actual_source_manifest")
        == dict(path="PHASES/PILOT_V3/SOURCE_MANIFEST.json", sha256=identity["source_sha256"])
        and overlay.get("data_manifest")
        == dict(path="DATA/MANIFEST.json", sha256=identity["data_sha256"])
        and overlay.get("origin_identity", {}).get("sha256") == identity["origin_identity_sha256"]
        and overlay.get("config_sha256") == identity["config_sha256"]
        and overlay.get("prompt_version")
        == identity.get("prompt_version")
        == projection.PROMPT_VERSION
        and overlay.get("wire_protocol") == "structured_v4"
        and overlay.get("old_data_modified") is False
        and overlay.get("training_authorized") is False
        and overlay.get("generation_authorized") is False
        and set(overlay.get("splits", {})) == {"DEV_PILOT", "DEV_EVAL"},
        "V3 overlay differs from the real run's registered source and inputs",
    )
    return overlay


def _registration_inputs(tx, root, registration, config, manifest):
    """Validate immutable split semantics once; no full-split image filesystem scan."""
    _require(
        manifest.get("config_sha256") == digest(config)
        and config.get("experiment_id") == tx.identity["experiment_id"],
        "Config/DATA identity differs",
    )
    phase_contracts = {
        "PILOT_V1": (12, "DEV_PILOT"),
        "PILOT_V3": (12, "DEV_PILOT"),
        "FORMAL_V1": (72, "TRAIN"),
    }
    phase = tx.identity["phase"]
    _require(phase in phase_contracts, "Unregistered rollout phase")
    expected_steps, split = phase_contracts[phase]
    _require(tx.identity["max_steps"] == expected_steps, "Phase update budget/split differs")
    factor = ARMS[tx.identity["arm"]][0]
    _require(
        registration.get("split") == split and registration.get("factor") == factor,
        "Only corresponding DEV_PILOT/TRAIN factor may be sampled",
    )
    if split == "DEV_PILOT":
        _require(
            tx.identity["repeat"] == 0 and tx.identity["arm"] in ("U10", "U11", "C_SHUF"),
            "Unregistered pilot path",
        )
    master = config["master_seed"]
    _require(
        type(master) is int and registration.get("master_seed") == master,
        "Frozen master seed differs",
    )
    counts = 8 if split == "DEV_PILOT" else 48
    _require(config["data"]["roots_per_family"][split] == counts, "Registered root budget differs")
    schedule = root_schedule(
        counts, master, tx.identity["parent"], tx.identity["repeat"], split=split
    )
    _require(
        registration.get("schedule") == schedule and len(schedule) == tx.identity["max_steps"],
        "Frozen root schedule differs",
    )
    files = manifest.get("files")
    _require(
        isinstance(files, list) and all(isinstance(item, dict) for item in files),
        "DATA file inventory required",
    )
    inventory = {item["path"]: item["sha256"] for item in files}
    _require(len(inventory) == len(files), "Duplicate DATA manifest path")
    inputs = registration.get("inputs", {})
    _require(
        set(inputs) == {"public", "private", "schedule"},
        "Three frozen split input references required",
    )
    expected_paths = {
        "public": f"DATA/{split}/MODEL_INPUTS.jsonl",
        "private": f"DATA/{split}/TASKS_PRIVATE.jsonl",
        "schedule": f"DATA/SCHEDULES/{split}-{tx.identity['parent']}-r{tx.identity['repeat']}.json",
    }
    overlay = _v3_overlay(tx, root) if phase == "PILOT_V3" else None
    for name, relative in expected_paths.items():
        if name == "public" and overlay is not None:
            projected = overlay["splits"][split]
            _require(
                inputs[name] == projected["public"]
                and projected["original_public"]
                == dict(path=relative, sha256=inventory.get(relative))
                and projected["private"] == inputs["private"],
                "V3 public projection must bind the same original frozen split",
            )
            continue
        _require(
            inputs[name] == {"path": relative, "sha256": inventory.get(relative)}
            and _sha(inventory.get(relative)),
            "Input is outside the corresponding frozen split manifest",
        )
    _require(
        _bound(tx, root, inputs["schedule"]) == schedule,
        "Manifest schedule differs from registered generator",
    )
    public = _bound(tx, root, inputs["public"], kind="jsonl")
    private = _bound(tx, root, inputs["private"], kind="jsonl")
    if overlay is not None:
        from .interface_v3_protocol import public_prompt as v3_prompt

        original_public = _bound(
            tx, root, overlay["splits"][split]["original_public"], kind="jsonl"
        )
        _require(len(original_public) == len(public), "V3 projection row count changed")
        task_lookup = {task["qid"]: task for task in private}
        for original, projected in zip(original_public, public, strict=True):
            task = task_lookup[original["qid"]]
            expected = copy.deepcopy(original)
            expected["model_input"]["prompt"] = v3_prompt(
                task["world"]["categories"], task["query"]
            )
            if "prompt_sha256" in original:
                expected["prompt_sha256"] = hashlib.sha256(
                    expected["model_input"]["prompt"].encode()
                ).hexdigest()
            expected["prompt_version"] = overlay["prompt_version"]
            _require(projected == expected, "V3 projection changed fields beyond its public prompt")
    expected_n = counts * 6 * 4 * 2
    _require(len(public) == len(private) == expected_n, "Complete paired split ledger required")
    publics, tasks = {row["qid"]: row for row in public}, {row["qid"]: row for row in private}
    roots = [rid for group in schedule for rid in group]
    qids = {f"{rid}_{f}_v{v}" for rid in roots for f in ("P0", "P1") for v in range(4)}
    _require(
        set(publics) == set(tasks) == qids and len(publics) == len(public),
        "Duplicate or missing registered qids",
    )
    for qid, item in publics.items():
        task = tasks[qid]
        _require(
            all(
                item.get(key) == task.get(key)
                for key in ("qid", "root_id", "family", "factor", "variant", "split")
            ),
            "Public/private task identity differs",
        )
        _require(
            task["split"] == split
            and task["family"] in FAMILIES
            and item.get("protocol") == "structured_v4",
            "Wrong split/family/protocol",
        )
        _require(
            type(task["variant"]) is int
            and task["variant"] in range(4)
            and qid == f"{task['root_id']}_{task['factor']}_v{task['variant']}"
            and task["root_id"].startswith(f"{split}_{task['family']}_"),
            "Task root/view lineage differs",
        )
        safe = item.get("model_input")
        _require(
            isinstance(safe, dict) and set(safe) == {"prompt", "image_file", "image_sha256"},
            "Only public model-input whitelist is allowed",
        )
        _require(
            safe["prompt"]
            == (
                v3_prompt(task["world"]["categories"], task["query"])
                if overlay is not None
                else public_prompt(task)
            ),
            "Public prompt differs from model-blind task projection",
        )
        image = safe["image_file"]
        _require(
            isinstance(image, str)
            and image.startswith("DATA/images/")
            and _sha(safe["image_sha256"])
            and inventory.get(image) == safe["image_sha256"],
            "Image must belong to frozen DATA inventory",
        )
        part = PurePosixPath(image)
        _require(part.as_posix() == image and ".." not in part.parts, "Noncanonical image path")
    draws, shuffles = [], []
    domain = "pilot_shuffle" if split == "DEV_PILOT" else "formal_shuffle"
    _require(registration.get("shuffle_domain") == domain, "Unregistered shuffle random stream")
    for step in range(1, len(schedule) + 1):
        draws.append(
            [
                [
                    draw_seed(
                        master, tx.identity["parent"], tx.identity["repeat"], step, g // 4, g % 4, d
                    )
                    for d in range(8)
                ]
                for g in range(16)
            ]
        )
        shuffles.append(
            [
                seed_for(
                    master,
                    domain,
                    tx.identity["parent"],
                    tx.identity["repeat"],
                    tx.identity["arm"],
                    step,
                    g,
                )
                for g in range(16)
            ]
        )
    _require(
        registration.get("draw_seeds") == draws and registration.get("shuffle_seeds") == shuffles,
        "Frozen draw/shuffle seeds differ; no replacement seed allowed",
    )
    return publics, tasks


def _context(tx, runtime, experiment_root, registration_ref):
    _require(isinstance(tx, RunTransactions), "Held RunTransactions context required")
    tx._locked()
    tx._runtime(runtime)
    root = _plain_path(experiment_root)
    _require(root.is_dir() and root.resolve() == root, "Canonical experiment root required")
    _require(runtime.identity.get("run_root") == str(root), "Runtime DATA root differs")
    _require(
        runtime.identity.get("phase") == tx.identity["phase"]
        and runtime.identity.get("execution_mode") == "TRAINING",
        "Explicit matching TRAINING phase required",
    )
    for key in ("phase_identity_sha256", "origin_identity_sha256"):
        if key in tx.identity:
            _require(
                runtime.identity.get(key) == tx.identity[key], "Runtime phase provenance differs"
            )
    if tx.identity["phase"] == "PILOT_V3":
        _require(
            all(
                runtime.identity.get(key) == tx.identity.get(key)
                for key in ("input_overlay", "input_overlay_sha256", "prompt_version")
            ),
            "Actual V3 runtime and transaction projection differ",
        )
    _require(
        tx.identity.get("rollout_registration") == registration_ref,
        "Rollout registration is not frozen in run identity",
    )
    registration = _bound(tx, tx.directory, registration_ref)
    core = {key: value for key, value in tx.identity.items() if key != "rollout_registration"}
    _require(
        registration.get("format") == FORMAT and registration.get("identity") == core,
        "Frozen rollout registration identity differs",
    )
    config = _bound(tx, root, {"path": "CONFIG.json", "sha256": tx.identity["config_sha256"]})
    manifest = _bound(
        tx, root, {"path": "DATA/MANIFEST.json", "sha256": tx.identity["data_sha256"]}
    )
    # Every entry still performs the six bound-file metadata checks. Changed
    # metadata invokes _bound's content hash verification before semantic reuse.
    # The registration hash binds source/config/DATA identities and all input
    # references, so validated semantic maps need no full-split traversal again.
    key = (str(root), registration_ref["sha256"])
    semantics = getattr(tx, "_srf6_rollout_semantics", None)
    if semantics is None:
        semantics = tx._srf6_rollout_semantics = {}
    if key not in semantics:
        semantics[key] = _registration_inputs(tx, root, registration, config, manifest)
    else:
        if tx.identity["phase"] == "PILOT_V3":
            overlay = _v3_overlay(tx, root)
            _bound(tx, root, overlay["splits"]["DEV_PILOT"]["original_public"], kind="jsonl")
        for name in ("public", "private", "schedule"):
            _bound(
                tx,
                root,
                registration["inputs"][name],
                kind="json" if name == "schedule" else "jsonl",
            )
    publics, tasks = semantics[key]
    commits = tx.published()
    _require(bool(commits), "Step zero must be committed before any rollout")
    committed = commits[-1]
    step = committed["step"] + 1
    _require(
        step <= tx.identity["max_steps"] and runtime.identity.get("step") == step - 1,
        "No budget remains or runtime committed cursor differs",
    )
    # Real-time path checks are limited to distinct images actually needed in
    # this step, even when replaying a complete READY. Native preparation checks
    # their contents on each actual generation; no other DATA images are touched.
    images = {
        publics[f"{rid}_{registration['factor']}_v{variant}"]["model_input"]["image_file"]
        for rid in registration["schedule"][step - 1]
        for variant in range(4)
    }
    for image in images:
        _require(_path(root, image).is_file(), "Frozen input image is missing")
    policy = state_hash(trainable_state(runtime.model))
    _require(
        policy == committed["policy_sha256"],
        "Restore the last commit before collecting: in-memory policy changed",
    )
    _require(
        runtime.policy_tensor_identity() == policy,
        "Runtime policy cache differs from committed tensor state",
    )
    return root, registration, publics, tasks, committed, step, policy


def _validate_raw(raw, runtime, safe, seeds, policy, root):
    _require(len(raw) == 8, "Exactly eight original raw draws required")
    for draw, row in enumerate(raw):
        _require(
            isinstance(row, dict) and type(row.get("seed")) is int and row["seed"] == seeds[draw],
            "Raw draw seed/order differs",
        )
        _require(
            type(row.get("group_size")) is int
            and row["group_size"] == 8
            and type(row.get("group_row_index")) is int
            and row["group_row_index"] == draw
            and type(row.get("group_seed")) is int
            and row.get("group_seed") == seeds[0],
            "Native G8 metadata differs",
        )
        if (
            row.get("generation_status") != "COMPLETE"
            or row.get("technical_validation_errors") != []
        ):
            raise IncompleteRollout(
                "TECHNICAL_INVALID/raw backend failure preserved; no replacement draws"
            )
        for key, expected in (
            ("model_input", safe),
            ("run_root", str(root)),
            ("run_id", runtime.identity["run_id"]),
            ("source_sha256", runtime.identity["source_sha256"]),
            ("data_hash", runtime.identity["data_hash"]),
            ("phase", runtime.identity["phase"]),
            ("execution_mode", "TRAINING"),
            ("policy_hash", policy),
            ("policy_tensor_hash", policy),
            ("policy_step", runtime.identity["step"]),
            ("step", runtime.identity["step"] + 1),
        ):
            _require(row.get(key) == expected, f"Raw generation {key} differs")
        adapter = row.get("adapter_identity", {})
        _require(
            all(
                adapter.get(a) == runtime.identity[b]
                for a, b in (("run_id", "run_id"), ("parent", "parent_id"), ("step", "step"))
            ),
            "Raw adapter identity differs",
        )
        for key in (
            "origin_identity_sha256",
            "phase_identity_sha256",
            "pilot_spec_sha256",
            "input_overlay",
            "input_overlay_sha256",
            "prompt_version",
        ):
            if key in runtime.identity:
                _require(row.get(key) == runtime.identity[key], f"Raw {key} differs")
        tokens, logps = row.get("tokens"), row.get("sampler_logprobs")
        _require(
            isinstance(tokens, list)
            and 1 <= len(tokens) <= 1536
            and all(type(t) is int and t >= 0 for t in tokens),
            "Invalid original completion tokens",
        )
        _require(
            row.get("raw_tokens") == tokens and row.get("completion_token_count") == len(tokens),
            "Original token aliases/count differ",
        )
        _require(
            isinstance(logps, list)
            and len(logps) == len(tokens)
            and all(type(v) in (int, float) and math.isfinite(v) and v <= 0 for v in logps),
            "Invalid original sampler probabilities",
        )
        _require(
            row.get("sampler_logprob_source") == "actual_generate_raw_logits_selected_token",
            "Native selected-token sampler probabilities required",
        )
        sampling = row.get("sampling_parameters", {})
        _require(
            all(
                sampling.get(k) == v
                for k, v in dict(
                    max_new_tokens=1536, temperature=1.0, top_p=1.0, top_k=0, do_sample=True
                ).items()
            )
            and row.get("sampling_hash") == hash_json(sampling)
            and row.get("max_new_tokens") == 1536,
            "Registered sampler recipe differs",
        )
        _require(
            row.get("protocol") == "structured_v4"
            and isinstance(row.get("native_prompt_inputs"), dict)
            and bool(row["native_prompt_inputs"])
            and _sha(row.get("prompt_tensor_hash")),
            "Native prompt/token protocol evidence required",
        )
        _require(
            row.get("prompt_sha256") == hashlib.sha256(safe["prompt"].encode()).hexdigest(),
            "Raw prompt identity differs",
        )
        _require(
            isinstance(row.get("full_text"), str)
            and row.get("finish_reason") in ("balanced", "eos", "length"),
            "Raw finish/text required",
        )
        _require(
            row["finish_reason"] != "length" or len(tokens) == 1536,
            "Length finish must retain the registered token limit",
        )
    _require(
        len({r["prompt_tensor_hash"] for r in raw}) == 1
        and len({digest(r["native_prompt_inputs"]) for r in raw}) == 1,
        "G8 mixed native prompt inputs",
    )


def _annotate(raw, task, tx, runtime, step, group_index):
    output = []
    for draw, original in enumerate(raw):
        row = copy.deepcopy(original)
        label = score_response(
            row["full_text"],
            task["world"],
            task["query"],
            finished=row["finish_reason"] == "balanced",
            finish_reason=row["finish_reason"],
        )
        row.update(label)
        # Retain the generated fact; score.finish_reason may normalize an
        # incomplete eos to length under the frozen balanced-only semantics.
        row["finish_reason"] = original["finish_reason"]
        row["generation_finish_reason"] = original["finish_reason"]
        row.update(
            experiment_id=tx.identity["experiment_id"],
            phase=tx.identity["phase"],
            run_id=tx.identity["run_id"],
            parent=tx.identity["parent"],
            repeat=tx.identity["repeat"],
            arm=tx.identity["arm"],
            factor=task["factor"],
            step=step,
            group_id=f"{tx.identity['run_id']}/step-{step:03d}/g{group_index:02d}",
            row_id=f"{tx.identity['run_id']}/step-{step:03d}/g{group_index:02d}/d{draw}",
            qid=task["qid"],
            root_id=task["root_id"],
            family=task["family"],
            variant=task["variant"],
            draw=draw,
            split=task["split"],
            score=label,
        )
        row["token_path_sha256"] = trajectory_hash(row)
        # Runtime's actual trainable policy hash is intentionally never replaced
        # with the frozen parent file-inventory identity used in census rows.
        _require(
            row["policy_hash"] == original["policy_tensor_hash"], "Training policy identity changed"
        )
        output.append(row)
    return output


def collect_step(tx, runtime, *, experiment_root, registration_ref, failpoint=None):
    """Collect/reuse exactly the next registered step under the caller's run lock.

    Returns records, credits, rollout/credit references and sampler_state.
    The caller later adds actual PREUPDATE_PASS evidence for transactions.commit.
    failpoint(name, group_index) is solely for bounded local interruption tests.
    """
    root, registration, public, tasks, committed, step, policy = _context(
        tx, runtime, experiment_root, registration_ref
    )
    destination = _plain_path(tx.directory / "rollouts" / f"step-{step:03d}")
    destination.mkdir(parents=True, exist_ok=True)
    step_spec = dict(
        format=FORMAT,
        identity=tx.identity,
        step=step,
        registration_ref=registration_ref,
        policy_before_sha256=policy,
        roots=registration["schedule"][step - 1],
    )
    started_path = destination / "STARTED.json"
    if started_path.exists():
        started = json.loads(_read_bytes(started_path))
        _require(started.get("spec") == step_spec, "Pending step identity differs")
        before_rng = _load_rng(tx, started["rng_before_reference"])
        _require(
            state_hash(before_rng) == committed["rng_sha256"],
            "Pending step RNG differs from committed state",
        )
    else:
        _require(not any(destination.iterdir()), "Orphan step artifacts require diagnosis")
        before_rng = capture_rng()
        _require(
            state_hash(before_rng) == committed["rng_sha256"],
            "Restore committed RNG before a new rollout",
        )
        before_ref = _save_rng(tx, destination / "rng-before.pt", before_rng)
        started = dict(status="STARTED", spec=step_spec, rng_before_reference=before_ref)
        _json(started_path, started)
    # Preflight all existing groups before any further model call. Even eight
    # raw rows cannot establish call completion without its RNG-after marker.
    groups_root = _plain_path(destination / "groups")
    groups_root.mkdir(exist_ok=True)
    for path in groups_root.iterdir():
        _plain_path(path)
        _require(
            path.is_dir() and path.name in {f"{i:02d}" for i in range(16)},
            "Unexpected group artifact",
        )
        if not (path / "COMPLETE.json").exists():
            raise IncompleteRollout(
                "PARTIAL_OR_UNKNOWN_CALL: uncompleted G8 preserved; no resampling"
            )
    existing = sorted(int(path.name) for path in groups_root.iterdir())
    _require(
        existing == list(range(len(existing))), "Completed groups must form the registered prefix"
    )
    if (destination / "READY.json").exists():
        _require(len(existing) == 16, "Published READY is missing completed groups")
    restore_rng(before_rng)
    versions = [(id(p), p._version) for p in runtime.model.parameters() if p.requires_grad]
    raw_groups, complete_refs, new_calls, reused = [], [], 0, 0
    try:
        for group_index in range(16):
            tx._locked()
            _require(
                versions
                == [(id(p), p._version) for p in runtime.model.parameters() if p.requires_grad],
                "Generation changed trainable parameters",
            )
            rid = step_spec["roots"][group_index // 4]
            factor = registration["factor"]
            qid = f"{rid}_{factor}_v{group_index % 4}"
            item, task = public[qid], tasks[qid]
            seeds = registration["draw_seeds"][step - 1][group_index]
            safe = copy.deepcopy(item["model_input"])
            group_path = _plain_path(groups_root / f"{group_index:02d}")
            spec = dict(
                step_spec_sha256=digest(step_spec),
                group_index=group_index,
                qid=qid,
                seeds=seeds,
                model_input_sha256=digest(safe),
                task_sha256=digest(task),
            )
            marker_path = group_path / "COMPLETE.json"
            if marker_path.exists():
                marker = json.loads(_read_bytes(marker_path))
                _require(
                    marker.get("status") == "IMMUTABLE_COMPLETE_G8"
                    and marker.get("spec") == spec
                    and marker.get("rows") == 8,
                    "Completed G8 identity differs",
                )
                group_started = json.loads(_read_bytes(group_path / "STARTED.json"))
                _require(
                    group_started.get("spec") == spec
                    and group_started.get("rng_before_reference") == marker["rng_before_reference"],
                    "G8 STARTED evidence differs",
                )
                saved_before = _load_rng(tx, marker["rng_before_reference"])
                _require(
                    state_hash(saved_before) == state_hash(capture_rng()),
                    "Completed G8 RNG chain differs",
                )
                raw = _bound(tx, tx.directory, marker["raw_reference"], kind="jsonl")
                _validate_raw(raw, runtime, safe, seeds, policy, root)
                after_rng = _load_rng(tx, marker["rng_after_reference"])
                _require(
                    state_hash(after_rng) == marker.get("rng_after_sha256"),
                    "Completed G8 after-RNG differs",
                )
                restore_rng(after_rng)
                reused += 1
            else:
                _require(not group_path.exists(), "Unpublished G8 must not be called again")
                group_path.mkdir()
                group_before = capture_rng()
                group_before_ref = _save_rng(tx, group_path / "rng-before.pt", group_before)
                _json(
                    group_path / "STARTED.json",
                    dict(
                        status="STARTED",
                        spec=spec,
                        rng_before_reference=group_before_ref,
                        requested_draws=8,
                        completion_status="UNKNOWN_UNTIL_COMPLETE",
                    ),
                )
                raw_path = group_path / "raw.jsonl"
                emitted = []

                def callback(row, raw_path=raw_path, emitted=emitted, group_index=group_index):
                    nonlocal new_calls
                    _append_raw(raw_path, row)
                    emitted.append(copy.deepcopy(row))
                    new_calls += 1
                    if len(emitted) > 8:
                        raise IncompleteRollout(
                            "Backend exceeded registered G8; all emitted rows preserved"
                        )
                    if failpoint:
                        failpoint("after_raw_draw", group_index)

                returned = runtime.generate_group(
                    safe, root, list(seeds), protocol="structured_v4", on_completion=callback
                )
                raw = _raw(raw_path)
                _require(safe == item["model_input"], "Generation mutated the public input")
                _require(
                    versions
                    == [(id(p), p._version) for p in runtime.model.parameters() if p.requires_grad],
                    "Generation changed trainable parameters",
                )
                _validate_raw(raw, runtime, safe, seeds, policy, root)
                _require(returned == emitted == raw, "Returned/native callback trajectories differ")
                after_rng = capture_rng()
                group_after_ref = _save_rng(tx, group_path / "rng-after.pt", after_rng)
                marker = dict(
                    status="IMMUTABLE_COMPLETE_G8",
                    spec=spec,
                    rows=8,
                    raw_reference=_ref(tx.directory, raw_path),
                    rng_before_reference=group_before_ref,
                    rng_after_reference=group_after_ref,
                    rng_after_sha256=state_hash(after_rng),
                )
                _json(marker_path, marker)
            complete_refs.append(_ref(tx.directory, marker_path))
            raw_groups.append((raw, task))
            if failpoint:
                failpoint("after_complete_g8", group_index)
        after_rng = capture_rng()
        _require(
            versions
            == [(id(p), p._version) for p in runtime.model.parameters() if p.requires_grad],
            "Generation changed trainable parameters",
        )
        if failpoint:
            failpoint("before_score", 16)
        records, credits = [], []
        for group_index, (raw, task) in enumerate(raw_groups):
            scored = _annotate(raw, task, tx, runtime, step, group_index)
            records.extend(scored)
            credits.append(
                build_credit_group(
                    scored, tx.identity["arm"], registration["shuffle_seeds"][step - 1][group_index]
                )
            )
        _require(
            len(records) == len({r["row_id"] for r in records}) == 128
            and len({r["qid"] for r in records}) == 16,
            "Incomplete final 128-row rollout",
        )
        _require(state_hash(capture_rng()) == state_hash(after_rng), "Scoring/credit changed RNG")
        after_ref = _save_rng(tx, destination / "rng-after.pt", after_rng)
        rollout = dict(
            status="ROLLOUT_READY",
            identity=tx.identity,
            step=step,
            registration_ref=registration_ref,
            policy_before_sha256=policy,
            rng_before_sha256=state_hash(before_rng),
            rng_after_sha256=state_hash(after_rng),
            rng_before_reference=started["rng_before_reference"],
            rng_after_reference=after_ref,
            complete_group_references=complete_refs,
            records=records,
        )
        credit = dict(identity=tx.identity, step=step, groups=credits)
        _json(destination / "rollout.json", rollout)
        _json(destination / "credit.json", credit)
        references = {
            name: _ref(tx.directory, destination / f"{name}.json") for name in ("rollout", "credit")
        }
        previous = committed["sampler_state"]
        sampler = dict(
            next_step=step + 1,
            committed_log_step=step,
            consumed_qids={
                **previous["consumed_qids"],
                str(step): [records[i]["qid"] for i in range(0, 128, 8)],
            },
            logical_calls=previous["logical_calls"] + 128,
            actual_calls=previous["actual_calls"] + 128,
            rollout_cursor=dict(step=step, groups=16, rows=128),
            rollout_rng_reference=references["rollout"],
        )
        _json(
            destination / "READY.json",
            dict(
                status="ROLLOUT_READY",
                spec=step_spec,
                references=references,
                sampler_state=sampler,
                gate_status="NOT_SIGNED",
                training_authorized=False,
            ),
        )
        if failpoint:
            failpoint("after_ready", 16)
        return dict(
            status="ROLLOUT_READY",
            step=step,
            records=records,
            credits=credits,
            references=references,
            sampler_state=sampler,
            new_model_calls=new_calls,
            reused_complete_groups=reused,
            gate_status="NOT_SIGNED",
            training_authorized=False,
        )
    except BaseException as exc:
        _append_raw(
            destination / "INCIDENTS.jsonl",
            dict(
                error_type=type(exc).__name__,
                error=str(exc),
                newly_persisted_rows=new_calls,
                status="PRESERVED_FAILURE_NO_REPLACEMENT_DRAWS",
            ),
        )
        raise
