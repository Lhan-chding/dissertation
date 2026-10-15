"""Fixed SR-F3-CE evaluation slots, durable raw records, and explicit release gates.

Generating an invalid answer is a completed observation. Missing a generation or
its immutable trajectory is a technical failure and cannot reduce a denominator.
All new TEST records, including the three parents, remain sealed until release.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

from sr_f12.evaluation import durable_json, group_slots, shard_lease

from . import contract

EXPERIMENT_ID = "SR-F3-CE-20261016"
ARMS = ("RL", "GOLD", "PREFIX", "EXEC")
PARENTS = tuple(f"SRF2_J6_s{s}" for s in (71001, 71002, 71003))
TEST_POOLS = ("TEST_ID", "TEST_COMPOSITION", "TEST_LANGUAGE", "TEST_RENDER", "TEST_ARITY")
SEALED_POOLS = (*TEST_POOLS, "CHARTQA", "CHARTQAPRO")
PROTOCOLS = ("structured", "natural_work", "answer_only")
DIAGNOSTIC_VIEWS = ("true_table", "selected_operands", "unevaluated_expression")
POOL_COUNTS = dict(
    ENGINE=64,
    MONITOR=64,
    TEST_ID=2048,
    TEST_COMPOSITION=256,
    TEST_LANGUAGE=128,
    TEST_RENDER=128,
    TEST_ARITY=128,
)


def object_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(tasks):
    rows = list(tasks.values()) if isinstance(tasks, dict) else list(tasks)
    if len({t["qid"] for t in rows}) != len(rows):
        raise ValueError("Duplicate task identity")
    return rows


def _pool(tasks, pool):
    rows = sorted((t for t in _rows(tasks) if t["pool"] == pool), key=lambda t: t["qid"])
    n = POOL_COUNTS[pool]
    if len(rows) != n or Counter(t["family"] for t in rows) != {
        f: n // 4 for f in contract.FAMILIES
    }:
        raise ValueError(f"Incomplete or unbalanced registered pool: {pool}")
    return rows


def endpoint_models(run_matrix=None):
    expected = [
        dict(model_id=f"SRF3CE_{arm}_b{b}", block=b, arm=arm, step=64, parent_id=PARENTS[b])
        for b in range(3)
        for arm in ARMS
    ]
    if run_matrix is not None:
        by_id = {r["run_id"]: r for r in run_matrix}
        if len(by_id) != 12 or len(run_matrix) != 12:
            raise ValueError("Exactly 12 registered scientific paths required")
        for model in expected:
            row = by_id.get(model["model_id"], {})
            if any(row.get(k) != model[k] for k in ("block", "arm", "parent_id")) or (
                row.get("updates") != 64 or row.get("future_seed") != 73101 + model["block"]
            ):
                raise ValueError("Run matrix differs from the frozen 12 paths")
    return [
        dict(model_id=p, parent_id=p, block=b, arm="PARENT", step=0) for b, p in enumerate(PARENTS)
    ] + expected


def generation_config(protocol):
    if protocol not in (*PROTOCOLS, "chartqa_plain", "chartqapro_official"):
        raise ValueError("Unknown registered protocol")
    return dict(
        do_sample=protocol in PROTOCOLS,
        temperature=1,
        top_p=1,
        top_k=0,
        max_new_tokens=256 if protocol == "chartqa_plain" else 1024,
        num_return_sequences=1,
    )


def _slots(model, tasks, protocol, k, *, view=None, phase="ENDPOINT"):
    out = []
    for task in tasks:
        identity = [
            EXPERIMENT_ID,
            phase,
            model["model_id"],
            model["step"],
            task["pool"],
            protocol,
            view,
            task["qid"],
        ]
        seeds = [int(object_hash([*identity, d])[:16], 16) % (2**31 - 1) for d in range(k)]
        for draw, seed in enumerate(seeds):
            out.append(
                dict(
                    model,
                    experiment_id=EXPERIMENT_ID,
                    phase=phase,
                    pool=task["pool"],
                    family=task.get("family", "EXTERNAL"),
                    root_id=task["root_id"],
                    qid=task["qid"],
                    mode=task.get("mode"),
                    chart=task.get("chart"),
                    variant=task.get("variant"),
                    image_cluster=task.get("image_cluster"),
                    protocol=protocol,
                    view=view,
                    draw=draw,
                    samples_per_prompt=k,
                    sampling_seed=seed,
                    group_seed=seeds[0],
                    slot_id=object_hash([*identity, draw]),
                    generation=generation_config(protocol),
                    sealed=task["pool"] in SEALED_POOLS,
                )
            )
    return out


def engine_slots(tasks):
    panel = _pool(tasks, "ENGINE")
    # The first four qids in each family are frozen before any inference. Each
    # is an existing ENGINE task; no outcome, answer, or score selects the panel.
    diagnostic = [t for f in contract.FAMILIES for t in [x for x in panel if x["family"] == f][:4]]
    out = []
    for model in endpoint_models()[:3]:
        for protocol in PROTOCOLS:
            out.extend(_slots(model, panel, protocol, 4, phase="ENGINE"))
        for view in DIAGNOSTIC_VIEWS:
            out.extend(_slots(model, diagnostic, "structured", 4, view=view, phase="ENGINE"))
    if len(out) != 2880:
        raise AssertionError("ENGINE registered count changed")
    return out


def monitor_slots(tasks, model_id, step):
    models = {r["model_id"]: r for r in endpoint_models()}
    if model_id not in models:
        raise ValueError("Unregistered monitor model")
    model = models[model_id]
    allowed = (0,) if model["arm"] == "PARENT" else (16, 32, 48, 64)
    if step not in allowed:
        raise ValueError("Step 0 is shared once per parent; only fixed later panels are allowed")
    return _slots(dict(model, step=step), _pool(tasks, "MONITOR"), "structured", 8, phase="MONITOR")


def all_monitor_slots(tasks):
    out = []
    for model in endpoint_models():
        for step in (0,) if model["arm"] == "PARENT" else (16, 32, 48, 64):
            out.extend(monitor_slots(tasks, model["model_id"], step))
    if len(out) != 26112:
        raise AssertionError("MONITOR registered count changed")
    return out


def endpoint_slots(tasks, run_matrix=None, external_tasks=(), *, model_id=None):
    pools = {p: _pool(tasks, p) for p in TEST_POOLS}
    canonical = [t for t in pools["TEST_ID"] if t["mode"] == "repair" and t["variant"] == 2]
    if len(canonical) != 256 or set(Counter(t["root_id"] for t in canonical).values()) != {2}:
        raise ValueError("Canonical repair v2 requires both charts of all 128 TEST_ID roots")
    external = _rows(external_tasks)
    external_pools = defaultdict(list)
    for task in external:
        if task["pool"] not in ("CHARTQA", "CHARTQAPRO"):
            raise ValueError("External task must use a registered external pool")
        external_pools[task["pool"]].append(task)
    if "CHARTQA" in external_pools and len(external_pools["CHARTQA"]) != 2500:
        raise ValueError("ChartQA is the complete 2500-question regression benchmark")
    if len(external_pools["CHARTQAPRO"]) > 256:
        raise ValueError("ChartQAPro exceeds its frozen 256-image limit")
    out = []
    models = endpoint_models(run_matrix)
    if model_id is not None:
        if model_id not in {m["model_id"] for m in models}:
            raise ValueError("Unregistered endpoint model")
        models = [m for m in models if m["model_id"] == model_id]
    for model in models:
        for panel in pools.values():
            out.extend(_slots(model, panel, "structured", 4))
        for protocol in ("answer_only", "natural_work"):
            out.extend(_slots(model, canonical, protocol, 4))
        for pool, protocol in (("CHARTQA", "chartqa_plain"), ("CHARTQAPRO", "chartqapro_official")):
            out.extend(
                _slots(model, sorted(external_pools[pool], key=lambda t: t["qid"]), protocol, 1)
            )
    return out


def evaluation_budget(chartqapro_images=256):
    if type(chartqapro_images) is not int or not 0 <= chartqapro_images <= 256:
        raise ValueError("ChartQAPro frozen image count must be between 0 and 256")
    out = dict(
        scientific_updates=768,
        RL_generations=98304,
        support_sequences=2304,
        monitor_generations=26112,
        structured_test_generations=161280,
        other_protocol_generations=30720,
        ChartQA_generations=37500,
        ChartQAPro_generations=15 * chartqapro_images,
        ENGINE_generations=2880,
        engine_initial_rollouts=512,
        recovery_cached_replay_sequences=256,
        engineering_physical_updates_max=6,
    )
    out["scientific_and_evaluation_generations"] = sum(
        out[k]
        for k in (
            "RL_generations",
            "monitor_generations",
            "structured_test_generations",
            "other_protocol_generations",
            "ChartQA_generations",
            "ChartQAPro_generations",
        )
    )
    return out


def _identity(runtime, name):
    value = getattr(runtime, name, None)
    return value() if callable(value) else value


def _input(inputs, qid, protocol):
    """The data store owns input provenance; audit-sidecar changes cannot route input."""
    if hasattr(inputs, "model_input"):
        return dict(inputs.model_input(qid, protocol=protocol))
    if hasattr(inputs, "get_model_input"):
        return dict(inputs.get_model_input(qid, protocol=protocol))
    row = inputs[qid]
    if protocol == "structured" or protocol.startswith("chartqa"):
        return {key: row[key] for key in ("prompt", "image_file", "image_sha256")}
    from .prompts import prompt_for

    return dict(
        prompt=prompt_for(row, protocol),
        image_file=row["image_file"],
        image_sha256=row["image_sha256"],
    )


def diagnostic_input(task, model_input, view):
    if task["pool"] != "ENGINE" or view not in DIAGNOSTIC_VIEWS:
        raise PermissionError("Oracle views are registered ENGINE-only diagnostics")
    world, query = task["world"], task["query"]
    if view == "true_table":
        extra = "The exact chart table is: " + contract.canonical(world)
    else:
        indices = contract.selected(world, query)
        values = [contract.encode(contract.operand(world, query, i)) for i in indices]
        if view == "selected_operands":
            extra = "The selected operands for the requested aggregate are: " + contract.canonical(
                values
            )
        else:
            extra = (
                "The unevaluated expression is: "
                + query["aggregate"]
                + "("
                + contract.canonical(values)
                + ")"
            )
    return dict(model_input, prompt=model_input["prompt"] + "\n" + extra)


def validate_record(
    row, slot, *, input_hash, model_identity, adapter_identity, source_sha256, index, count
):
    if any(row.get(k) != v for k, v in slot.items()):
        raise PermissionError("Persisted slot differs from registered evaluation")
    expected = dict(
        input_hash=input_hash,
        model_identity=model_identity,
        adapter_identity=adapter_identity,
        source_sha256=source_sha256,
        group_row_index=index,
        group_size=count,
        runtime_group_seed=slot["group_seed"],
        runtime_group_row_index=index,
        runtime_group_size=count,
        runtime_protocol=slot["protocol"],
        runtime_seed=slot["sampling_seed"],
    )
    if any(row.get(k) != v for k, v in expected.items()):
        raise PermissionError("Runtime/input/model/source/group identity differs")
    tokens, probabilities = row.get("tokens"), row.get("sampler_logprobs")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(type(x) is not int or x < 0 for x in tokens)
        or row.get("raw_tokens") != tokens
        or row.get("completion_token_count") != len(tokens)
        or not isinstance(probabilities, list)
        or len(probabilities) != len(tokens)
        or any(
            isinstance(p, bool)
            or not isinstance(p, (int, float))
            or not math.isfinite(p)
            or p > 1e-5
            for p in probabilities
        )
    ):
        raise ValueError("Complete token and sampler-logprob trajectory required")
    if row.get("generation_status") != "COMPLETE" or row.get("technical_validation_errors"):
        raise RuntimeError("Technical generation failure; preserve raw record and block")
    if not row.get("prompt_tensor_hash"):
        raise ValueError("Actual prompt tensor identity is required")
    if len(tokens) > slot["generation"]["max_new_tokens"]:
        raise ValueError("Generation exceeds registered answer budget")
    if row.get("truncated") is not (row.get("finish_reason") == "length"):
        raise ValueError("Truncation metadata differs")
    if row.get("finish_reason") not in ("balanced", "final_tag", "eos", "length"):
        raise ValueError("Unknown generation termination")
    if row.get("finish_reason") == "length" and len(tokens) != slot["generation"]["max_new_tokens"]:
        raise ValueError("Length stop precedes the registered cap")
    prefill = "{" if slot["protocol"] in ("structured", "answer_only") else ""
    if row.get("assistant_prefill") != prefill or row.get("raw_text") != prefill + row.get(
        "generated_text", ""
    ):
        raise ValueError("Context prefill and immutable raw text differ")


def score_record(row, task, *, external_scorer=None):
    if row["pool"] in ("CHARTQA", "CHARTQAPRO"):
        if external_scorer is None:
            raise RuntimeError("Frozen external official and audit scorer is required")
        return external_scorer(row, task)
    result = contract.score(
        row["raw_text"],
        task["world"],
        task["query"],
        finish_reason=row["finish_reason"],
        protocol=row["protocol"],
    )
    result["certificate_applicable"] = row["protocol"] == "structured"
    if row.get("format_protocol_error"):
        result.update(presentation=0, J=0, reason=row["format_protocol_error"])
    return result


def validate_coverage(records, slots):
    expected = {s["slot_id"]: s for s in slots}
    if len(expected) != len(slots):
        raise ValueError("Duplicate expected evaluation slot")
    observed = {}
    for row in records:
        sid = row.get("slot_id")
        if sid in observed or sid not in expected:
            raise ValueError("Duplicate or unregistered evaluation record")
        if any(row.get(k) != v for k, v in expected[sid].items()):
            raise PermissionError("Observed slot metadata differs")
        if row.get("generation_status") != "COMPLETE" or row.get("technical_validation_errors"):
            raise RuntimeError("Technical record failure is not a model error")
        observed[sid] = row
    missing = sorted(set(expected) - set(observed))
    if missing:
        raise RuntimeError(
            f"Incomplete evaluation: {len(missing)} missing slots; no denominator change"
        )
    return dict(status="COMPLETE", records=len(records), slots_sha256=object_hash(slots))


def release_evaluation(run_matrix, training_receipts, slots, records, *, external_status):
    """Produce a content-bound receipt only after all 12 paths and evaluation finish."""
    models = endpoint_models(run_matrix)
    receipts = {r["run_id"]: r for r in training_receipts}
    if len(receipts) != 12 or len(training_receipts) != 12:
        raise PermissionError(
            "All 12 scientific training receipts are required before TEST release"
        )
    for model in models[3:]:
        receipt = receipts.get(model["model_id"], {})
        if (
            receipt.get("status") != "COMPLETED"
            or receipt.get("step") != 64
            or not receipt.get("adapter_identity")
        ):
            raise PermissionError("Every scientific path must finish the registered 64 updates")
    # Stream whole groups: all token trajectories need not coexist in CPU RAM.
    expected = {s["slot_id"]: s for s in slots}
    if len(expected) != len(slots):
        raise ValueError("Duplicate frozen release slot")
    seen, raw_groups, current, current_id = set(), {}, [], None
    for row in records:
        sid, gid = row.get("slot_id"), row.get("group_id")
        if sid not in expected or sid in seen:
            raise ValueError("Duplicate/unregistered raw release record")
        if any(row.get(k) != v for k, v in expected[sid].items()):
            raise PermissionError("Raw release slot differs from frozen metadata")
        if row.get("generation_status") != "COMPLETE" or row.get("technical_validation_errors"):
            raise RuntimeError("Technical generation failure cannot enter release")
        if gid != current_id and current:
            raw_groups[current_id] = object_hash(current)
            current = []
        if not isinstance(gid, str) or gid in raw_groups:
            raise ValueError("Raw release groups must be complete and contiguous")
        current_id = gid
        current.append(row)
        seen.add(sid)
    if current:
        raw_groups[current_id] = object_hash(current)
    if seen != set(expected):
        raise RuntimeError("Incomplete raw release coverage; technical missingness blocks release")
    coverage = dict(records=len(seen), slots_sha256=object_hash(slots))
    pool_counts = Counter(s["pool"] for s in slots)
    if any(
        pool_counts[p] != POOL_COUNTS[p] * 15 * 4 + (30720 if p == "TEST_ID" else 0)
        for p in TEST_POOLS
    ):
        raise PermissionError("Release requires the complete 15-model synthetic endpoint matrix")
    if {s["model_id"] for s in slots} != {m["model_id"] for m in models}:
        raise PermissionError("All three parents and 12 endpoints must be represented")
    if pool_counts["CHARTQA"] != 37500:
        raise PermissionError("ChartQA regression panel is incomplete")
    pro = external_status.get("CHARTQAPRO", {})
    if pro.get("status") not in ("COMPLETE", "UNAVAILABLE"):
        raise PermissionError("ChartQAPro must finish or have an explicit unavailability receipt")
    if pro["status"] == "COMPLETE":
        if type(pro.get("image_count")) is not int or not 0 < pro["image_count"] <= 256:
            raise ValueError("Frozen ChartQAPro root count required")
        if pool_counts["CHARTQAPRO"] != 15 * pro["image_count"]:
            raise PermissionError("ChartQAPro coverage differs from its frozen subset")
    elif pool_counts["CHARTQAPRO"] or not pro.get("reason"):
        raise PermissionError(
            "Unavailable external data needs a reason and cannot hide partial output"
        )
    return dict(
        status="RELEASED",
        experiment_id=EXPERIMENT_ID,
        training_receipts_sha256=object_hash(training_receipts),
        slots_sha256=coverage["slots_sha256"],
        records=coverage["records"],
        raw_groups=raw_groups,
        external_status=external_status,
        scope="fixed 3 parents and 12 endpoints; all outcomes retained",
    )


def _check_release(group_id, rows, release_receipt):
    if (
        not isinstance(release_receipt, dict)
        or release_receipt.get("status") != "RELEASED"
        or release_receipt.get("experiment_id") != EXPERIMENT_ID
        or release_receipt.get("raw_groups", {}).get(group_id) != object_hash(rows)
    ):
        raise PermissionError(
            "TEST scoring needs the release receipt bound to these exact raw records"
        )


def _pending_rows(path):
    lines = Path(path).read_text().splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError("Blank durable completion row")
    return [json.loads(line) for line in lines]


def _write_immutable_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise PermissionError("Immutable partial replay evidence differs")
        return
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _partial_replay_authorization(root, first, group_id, pending, source_sha256):
    """Resolve one immutable, identity-bound technical recovery authorization."""
    entrypoint = root / "recovery/PARTIAL_REPLAY_AUTHORIZATION.json"
    if not entrypoint.exists():
        raise RuntimeError(
            "Incomplete durable group; preserve and explicitly recover without resampling"
        )
    if first["phase"] == "ENDPOINT":
        task_id = "eval--" + first["model_id"]
    elif first["phase"] == "MONITOR" and first["arm"] != "PARENT":
        task_id = "train--" + first["model_id"]
    else:
        raise PermissionError("Partial replay is not permitted for ENGINE or shared step-0 panels")
    authorization_path = entrypoint
    authorization = json.loads(entrypoint.read_text())
    if "authorization_file" in authorization:
        relative = Path(authorization["authorization_file"])
        authorization_path = (root / relative).resolve(strict=True)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not authorization_path.is_relative_to(root / "recovery")
            or sha_file(authorization_path) != authorization.get("sha256")
        ):
            raise PermissionError("Partial replay authorization pointer differs")
        authorization = json.loads(authorization_path.read_text())
    if (
        authorization.get("experiment_id") != EXPERIMENT_ID
        or authorization.get("run_identity_sha256") != source_sha256
        or sha_file(root / "RUN_IDENTITY.json") != source_sha256
    ):
        raise PermissionError("Partial replay run identity differs")
    binding = dict(
        task_id=task_id,
        model_id=first["model_id"],
        group_id=group_id,
        pending_sha256=sha_file(pending),
        status="AUTHORIZED",
    )
    matching = [
        entry
        for entry in authorization.get("authorizations", [])
        if all(entry.get(key) == value for key, value in binding.items())
    ]
    if len(matching) != 1:
        raise PermissionError("No unique explicit authorization for this exact partial group")
    auth_sha = sha_file(authorization_path)
    return dict(
        authorization_file=str(authorization_path.relative_to(root)),
        authorization_sha256=auth_sha,
        binding=binding,
        recovery_key=object_hash([str(authorization_path.relative_to(root)), auth_sha]),
    ), authorization


def _replay_partial(
    group,
    root,
    directory,
    pending,
    original,
    *,
    model,
    adapter,
    input_hash,
    source_sha256,
    generate,
):
    """Replay the entire original batch once, never choose a better replacement answer."""
    first, count = group["slots"][0], len(group["slots"])
    auth, authorization = _partial_replay_authorization(
        root, first, group["group_id"], pending, source_sha256
    )
    if len(original) >= count:
        raise ValueError("Only an incomplete durable batch uses partial replay")
    for index, row in enumerate(original):
        validate_record(
            row,
            group["slots"][index],
            input_hash=input_hash,
            model_identity=model,
            adapter_identity=adapter,
            source_sha256=source_sha256,
            index=index,
            count=count,
        )
    recovery = directory / "recovery" / group["group_id"] / auth["recovery_key"]
    snapshot = recovery / "ORIGINAL.partial.jsonl"
    _write_immutable_bytes(snapshot, pending.read_bytes())
    durable_json(recovery / "AUTHORIZATION.json", authorization)
    replay_path = recovery / "REPLAY.pending.jsonl"
    attempt_path = recovery / "ATTEMPT.json"
    attempt = dict(
        auth,
        experiment_id=EXPERIMENT_ID,
        group_id=group["group_id"],
        input_hash=input_hash,
        model_identity=model,
        adapter_identity=adapter,
        source_sha256=source_sha256,
        original_rows=len(original),
        full_group_size=count,
        seeds=[s["sampling_seed"] for s in group["slots"]],
        replay_policy="one full deterministic batch; exact original prefix equality",
    )
    if attempt_path.exists():
        if json.loads(attempt_path.read_text()) != attempt:
            raise PermissionError("Partial replay attempt identity differs")
        replay = _pending_rows(replay_path) if replay_path.exists() else []
        if len(replay) != count:
            raise RuntimeError(
                "Authorized replay interrupted; preserve it and require a new technical decision"
            )
    else:
        if replay_path.exists():
            raise PermissionError(
                "Replay trajectories exist without their immutable authorization attempt"
            )
        durable_json(attempt_path, attempt)
        replay = generate(replay_path)
    if len(replay) != count:
        raise RuntimeError("Authorized replay did not preserve the full registered group")
    for index, row in enumerate(replay):
        validate_record(
            row,
            group["slots"][index],
            input_hash=input_hash,
            model_identity=model,
            adapter_identity=adapter,
            source_sha256=source_sha256,
            index=index,
            count=count,
        )
    trajectory_keys = (
        "tokens",
        "raw_tokens",
        "sampler_logprobs",
        "raw_text",
        "generated_text",
        "assistant_prefill",
        "finish_reason",
        "truncated",
        "completion_token_count",
        "prompt_tensor_hash",
        "format_protocol_error",
        "boundary_char",
        "boundary_token_count",
        "balanced_cut_char",
        "final_tag_cut_char",
        "decoded_prefix_views",
        "prefix_character_offsets_stable",
    )
    mismatches = [
        dict(
            index=index,
            differing_fields=[
                key for key in trajectory_keys if old.get(key) != replay[index].get(key)
            ],
        )
        for index, old in enumerate(original)
    ]
    mismatches = [item for item in mismatches if item["differing_fields"]]
    receipt = dict(
        attempt,
        status="MISMATCH" if mismatches else "VERIFIED",
        original_snapshot_sha256=sha_file(snapshot),
        replay_sha256=sha_file(replay_path),
        mismatches=mismatches,
        original_prefix_rows_verified=len(original),
        technical_replay_generated_sequences=count,
        repeated_existing_sequences=len(original),
        sampling_cost_accounting=(
            "runtime generation attempt ledger; these counts are explanatory, not additive"
        ),
        answer_selection=False,
    )
    durable_json(recovery / "RECEIPT.json", receipt)
    if mismatches:
        raise RuntimeError(
            "Deterministic partial replay differs; preserve all trajectories and block"
        )
    return replay


def run_group(
    runtime,
    group,
    inputs,
    tasks,
    root,
    directory,
    *,
    source_sha256,
    score=False,
    release_receipt=None,
    external_scorer=None,
):
    """Durably persist each completion before any validation or scientific scoring.

    The caller holds ``shard_lease``. A partial group blocks automatic resampling;
    already generated trajectories must be recovered explicitly and preserved.
    """
    root, directory = Path(root).resolve(strict=True), Path(directory).resolve()
    if not directory.is_relative_to(root):
        raise PermissionError("Evaluation output escapes run root")
    slots = group["slots"]
    checked = group_slots(slots)
    if len(checked) != 1 or checked[0]["group_id"] != group["group_id"]:
        raise ValueError("Group identity differs")
    first, count = slots[0], len(slots)
    if score and first["sealed"] and not release_receipt:
        raise PermissionError("All new TEST scores, including parents, remain sealed")
    model, adapter = (
        _identity(runtime, k) for k in ("stable_model_identity", "current_adapter_identity")
    )
    if not isinstance(model, dict) or not isinstance(adapter, dict):
        raise ValueError("Immutable model and adapter identities required")
    request = _input(inputs, first["qid"], first["protocol"])
    if first["view"]:
        request = diagnostic_input(tasks[first["qid"]], request, first["view"])
    if set(request) != {"prompt", "image_file", "image_sha256"}:
        raise ValueError("Only the three frozen model-input fields may reach the runtime")
    image_root = Path(getattr(inputs, "image_root", root)).resolve(strict=True)
    image = (image_root / request["image_file"]).resolve(strict=True)
    if not image.is_relative_to(image_root) or sha_file(image) != request["image_sha256"]:
        raise PermissionError("Frozen image path or content differs")
    input_hash = object_hash(request)
    path = directory / "groups" / (group["group_id"] + ".json")
    pending = path.with_suffix(".pending.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)

    def generate(destination):
        rows = []

        def persist(completion):
            index = len(rows)
            if index >= count:
                raise RuntimeError("Runtime emitted excess completions")
            row = dict(completion)
            for key in ("group_seed", "group_row_index", "group_size", "protocol", "seed"):
                row["runtime_" + key] = completion.get(key)
            row.update(slots[index])
            row.update(
                model_input=request,
                input_hash=input_hash,
                source_sha256=source_sha256,
                group_id=group["group_id"],
                group_row_index=index,
                group_size=count,
            )
            with destination.open("a") as stream:
                stream.write(
                    json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            rows.append(row)

        runtime.generate_group(
            request,
            image_root,
            [s["sampling_seed"] for s in slots],
            protocol=first["protocol"],
            generation=dict(first["generation"], num_return_sequences=count),
            on_completion=persist,
        )
        if len(rows) != count:
            raise RuntimeError("Runtime omitted required durable completions")
        return rows

    original_pending = _pending_rows(pending) if pending.exists() else None
    recovered = False
    if path.exists():
        value = json.loads(path.read_text())
        rows = value["records"]
        if (
            value.get("records_sha256") != object_hash(rows)
            or value.get("group_id") != group["group_id"]
        ):
            raise PermissionError("Raw evaluation group checksum differs")
    elif original_pending is not None:
        rows = original_pending
    else:
        rows = generate(pending)
    if original_pending is not None and len(original_pending) < count:
        replay = _replay_partial(
            group,
            root,
            directory,
            pending,
            original_pending,
            model=model,
            adapter=adapter,
            input_hash=input_hash,
            source_sha256=source_sha256,
            generate=generate,
        )
        if path.exists() and rows != replay:
            raise PermissionError("Committed group differs from its verified deterministic replay")
        rows, recovered = replay, True
    durable_json(
        path, dict(group_id=group["group_id"], records=rows, records_sha256=object_hash(rows))
    )
    if len(rows) != count:
        raise RuntimeError("Persisted group coverage differs")
    for index, (row, slot) in enumerate(zip(rows, slots, strict=True)):
        validate_record(
            row,
            slot,
            input_hash=input_hash,
            model_identity=model,
            adapter_identity=adapter,
            source_sha256=source_sha256,
            index=index,
            count=count,
        )
    if len({r["prompt_tensor_hash"] for r in rows}) != 1:
        raise PermissionError("Same-question prompt tensors differ")
    if pending.exists():
        expected_pending = original_pending if recovered else rows
        if _pending_rows(pending) != expected_pending:
            raise PermissionError("Pending and committed groups differ")
        pending.unlink()
    scores = None
    if score:
        if first["sealed"]:
            _check_release(group["group_id"], rows, release_receipt)
        scores = [
            dict(
                slot_id=r["slot_id"],
                score=score_record(r, tasks[r["qid"]], external_scorer=external_scorer),
            )
            for r in rows
        ]
        durable_json(
            directory / "scores" / path.name,
            dict(raw_sha256=sha_file(path), scorer_version=EXPERIMENT_ID, scores=scores),
        )
    return dict(
        group_id=group["group_id"],
        records=rows,
        scores=scores,
        raw_sha256=sha_file(path),
        records_sha256=object_hash(rows),
    )


def verify_panel_cache(
    root,
    directory,
    slots,
    inputs,
    tasks,
    *,
    model_identity,
    adapter_identity,
    source_sha256,
    task_id,
):
    """CPU-only verification of existing complete/partial groups; never generates.

    The caller obtains model/adapter identities from its authenticated checkpoint
    or endpoint provenance. Returned partial bindings can be included in an
    explicit scheduler recovery decision; this function grants no authorization.
    """
    root, directory = Path(root).resolve(strict=True), Path(directory).resolve()
    if not directory.is_relative_to(root) or sha_file(root / "RUN_IDENTITY.json") != source_sha256:
        raise PermissionError("Cache recovery run/path identity differs")
    groups = {group["group_id"]: group for group in group_slots(slots)}
    complete, partial = [], []
    for path in sorted((directory / "groups").glob("*")):
        if path.name.endswith(".pending.jsonl"):
            gid, kind = path.name.removesuffix(".pending.jsonl"), "pending"
        elif path.suffix == ".json":
            gid, kind = path.stem, "complete"
        else:
            raise PermissionError("Unregistered evaluation cache file")
        if gid not in groups:
            raise PermissionError("Cache group is not registered for this panel")
        group = groups[gid]
        first, count = group["slots"][0], len(group["slots"])
        request = _input(inputs, first["qid"], first["protocol"])
        if first["view"]:
            request = diagnostic_input(tasks[first["qid"]], request, first["view"])
        if kind == "complete":
            value = json.loads(path.read_text())
            rows = value["records"]
            if (
                value.get("group_id") != gid
                or len(rows) != count
                or value.get("records_sha256") != object_hash(rows)
            ):
                raise PermissionError("Committed evaluation group checksum/coverage differs")
        else:
            rows = _pending_rows(path)
            if len(rows) > count:
                raise ValueError("Pending group has excess trajectories")
        for index, row in enumerate(rows):
            if row.get("model_input") != request:
                raise PermissionError("Cached model input differs from its frozen store")
            validate_record(
                row,
                group["slots"][index],
                input_hash=object_hash(request),
                model_identity=model_identity,
                adapter_identity=adapter_identity,
                source_sha256=source_sha256,
                index=index,
                count=count,
            )
        item = dict(
            group_id=gid,
            path=str(path.relative_to(root)),
            sha256=sha_file(path),
            records=len(rows),
            expected_records=count,
        )
        if kind == "pending" and len(rows) < count:
            partial.append(
                dict(
                    item, task_id=task_id, model_id=first["model_id"], pending_sha256=item["sha256"]
                )
            )
        else:
            complete.append(item)
    return dict(
        status="VERIFIED_EXISTING_CACHE",
        complete_groups=complete,
        partial_groups=partial,
        missing_groups=len(groups) - len({item["group_id"] for item in [*complete, *partial]}),
        source_sha256=source_sha256,
        authorization_granted=False,
    )


def summarize_scores(rows):
    """Mean K draws, then fixed questions within roots, then equal families."""
    if not rows:
        raise ValueError("No complete scored observations")
    metrics = (
        "value_correct",
        "certificate",
        "certificate_truth",
        "certificate_sufficient",
        "presentation",
        "finish",
        "J",
        "legacy_full",
        "value_parseable",
    )
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seen = set()
    for row in rows:
        if row["slot_id"] in seen:
            raise ValueError("Duplicate scored observation")
        seen.add(row["slot_id"])
        cells[row["family"]][row["root_id"]][row["qid"]].append(row)
    family_results = {}
    for family, roots in sorted(cells.items()):
        roots_out = []
        for questions in roots.values():
            question_values = []
            for observations in questions.values():
                k = observations[0]["samples_per_prompt"]
                if sorted(r["draw"] for r in observations) != list(range(k)):
                    raise RuntimeError("Missing draw cannot be excluded from a mean")
                values = {}
                for metric in metrics:
                    raw = [r["score"][metric] for r in observations]
                    if any(v not in (0, 1) for v in raw):
                        raise ValueError("Finite binary scores required")
                    values[metric] = sum(raw) / k
                question_values.append(values)
            roots_out.append(
                {m: sum(v[m] for v in question_values) / len(question_values) for m in metrics}
            )
        family_results[family] = dict(
            roots=len(roots), **{m: sum(v[m] for v in roots_out) / len(roots_out) for m in metrics}
        )
    return dict(
        records=len(rows),
        families=family_results,
        equal_family={
            m: sum(v[m] for v in family_results.values()) / len(family_results) for m in metrics
        },
    )


def evaluate_monitor(runtime, tasks, inputs, root, directory, *, model_id, step, source_sha256):
    """Fixed 512-response monitor, restoring global RNG and model mode even on error.

    Evaluation seeds are slot identities, never taken from the training RNG. The
    parent step-0 panel is shared; this entry point rejects a per-arm step zero.
    """
    from mm_core.training import capture_rng, restore_rng, state_hash, trainable_state

    slots = monitor_slots(tasks, model_id, step)
    groups = group_slots(slots)
    directory, root = Path(directory).resolve(), Path(root).resolve(strict=True)
    if not directory.is_relative_to(root):
        raise PermissionError("Monitor output escapes run root")
    rng, mode = capture_rng(), runtime.model.training
    before = state_hash(trainable_state(runtime.model))
    try:
        runtime.model.eval()
        plan = dict(
            experiment_id=EXPERIMENT_ID,
            model_id=model_id,
            step=step,
            source_sha256=source_sha256,
            slots_sha256=object_hash(slots),
            policy_hash=before,
            expected_records=512,
            model_identity=_identity(runtime, "stable_model_identity"),
            adapter_identity=_identity(runtime, "current_adapter_identity"),
        )
        with shard_lease(directory):
            durable_json(directory / "PLAN.json", plan)
            allowed = {g["group_id"] for g in groups}
            for path in (directory / "groups").glob("*"):
                if path.name.removesuffix(".pending.jsonl").removesuffix(".json") not in allowed:
                    raise PermissionError("Unregistered monitor group")
            scored, receipts = [], []
            for group in groups:
                receipt = run_group(
                    runtime,
                    group,
                    inputs,
                    tasks,
                    root,
                    directory,
                    source_sha256=source_sha256,
                    score=True,
                )
                score_map = {r["slot_id"]: r["score"] for r in receipt["scores"]}
                scored.extend(dict(r, score=score_map[r["slot_id"]]) for r in receipt["records"])
                receipts.append(
                    {k: receipt[k] for k in ("group_id", "raw_sha256", "records_sha256")}
                )
            validate_coverage(scored, slots)
            if state_hash(trainable_state(runtime.model)) != before:
                raise RuntimeError("Monitor changed policy parameters")
            result = dict(
                status="COMPLETE",
                experiment_id=EXPERIMENT_ID,
                model_id=model_id,
                step=step,
                records=len(scored),
                groups=receipts,
                policy_hash=before,
                summary=summarize_scores(scored),
                rng_isolated=True,
                test_access=False,
                changes_training_control=False,
            )
            durable_json(directory / "COMPLETE.json", result)
            return result
    finally:
        runtime.model.train(mode)
        restore_rng(rng)
