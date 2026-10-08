"""Fail-closed phase gates and shared, append-only physical accounting."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

STAGES = {
    "FORMAT_BASE_TEST",
    "FORMAT_TUNE_POST_BRIDGE",
    "FORMAT_CHECK",
    "MEASUREMENT_AUDIT",
    "BRIDGE",
    "ENGINE",
}
CAPS = {
    "completion_attempts": 4096,
    "physical_optimizer_updates": 64,
    "extra_forward_sequences": 8192,
    "allocated_gpu_hours": None,
}
GENERATION = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "max_new_tokens": 192,
    "num_beams": 1,
    "num_return_sequences": 1,
    "use_cache": True,
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def exclusive_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        locked(path.with_suffix(path.suffix + ".lock")),
        path.open("a", encoding="utf-8") as handle,
    ):
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def checked_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Evidence paths must be safe run-relative paths")
    target = (Path(root) / path).resolve()
    target.relative_to(Path(root).resolve())
    return target


def derive_seed(stage, model_hash, row, sample_index):
    payload = [
        stage,
        model_hash,
        row["root_family_id"],
        row["image_id"],
        row["question_id"],
        sample_index,
    ]
    return int(object_hash(payload)[:16], 16) % (2**63 - 1)


class BudgetLedger:
    """Count caps fail closed; GPU hours are retained for accounting only."""

    def __init__(self, run_root):
        self.root = Path(run_root)
        self.path = self.root / "accounting/COST_LEDGER.jsonl"
        config = read_json(self.root / "manifests/RESOURCE_OVERRIDE.json")
        self.caps = dict(CAPS)
        if config["max_concurrent_gpus"] != 5:
            raise ValueError("This run is bound to the user five-GPU override")
        if config.get("max_allocated_gpu_hours") is not None:
            raise ValueError("GPU hours must be accounting-only in the current resource override")

    def totals(self):
        totals = dict.fromkeys(self.caps, 0)
        for row in read_jsonl(self.path):
            if row["kind"] not in totals or not math.isfinite(row["amount"]) or row["amount"] < 0:
                raise ValueError("Invalid budget ledger entry")
            totals[row["kind"]] += row["amount"]
        return totals

    def reserve(self, kind, amount, identity):
        if (
            kind not in self.caps
            or isinstance(amount, bool)
            or not isinstance(amount, (int, float))
        ):
            raise ValueError("Unknown budget kind or amount")
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("Reservation must be positive and finite")
        if kind != "allocated_gpu_hours" and int(amount) != amount:
            raise ValueError("Sequence/update counts must be integers")
        with locked(self.root / "accounting/BUDGET.lock"):
            totals = self.totals()
            if self.caps[kind] is not None and totals[kind] + amount > self.caps[kind]:
                raise RuntimeError(f"BUDGET_EXHAUSTED: {kind}")
            append_jsonl(
                self.path,
                {
                    "time": utc_now(),
                    "kind": kind,
                    "amount": amount,
                    "identity": identity,
                    "status": (
                        "ALLOCATION_TIME_ESTIMATE"
                        if kind == "allocated_gpu_hours"
                        else "CONSUMED_OR_RESERVED"
                    ),
                },
            )


def _verify_model_override(identity):
    from .vl_runtime import CHAT_TEMPLATE_KWARGS, MODEL_ID, hash_json

    if identity.get("model_id") != MODEL_ID:
        raise PermissionError("Current user override requires Qwen/Qwen3.5-9B")
    if identity.get("chat_template_kwargs") != CHAT_TEMPLATE_KWARGS or identity.get(
        "chat_template_kwargs_hash"
    ) != hash_json(CHAT_TEMPLATE_KWARGS):
        raise PermissionError("Qwen3.5 thinking mode is not frozen consistently")


def _verify_freeze(run_root):
    root = Path(run_root)
    freeze = read_json(root / "manifests/PRE_INFERENCE_FREEZE.json")
    if freeze.get("status") != "FROZEN" or freeze.get("dev_authorized") is not False:
        raise PermissionError("A complete execution freeze is required")
    _verify_model_override(freeze)
    if freeze.get("readability_review_status") != "PASS":
        raise PermissionError("Actual processor readability review is required")
    if freeze.get("old_work_stop_status") not in {"already_stopped", "cancelled_verified"}:
        raise PermissionError("SER-J23 stop status is not verified")
    if not freeze.get("account_and_resource_permission_verified"):
        raise PermissionError("Account and concurrent resource verification required")
    for key in (
        "model_weights_hash",
        "processor_hash",
        "environment_lock_hash",
        "chat_template_hash",
        "model_revision",
        "project_commit",
    ):
        if not freeze.get(key):
            raise PermissionError(f"Unresolved identity: {key}")
    for relative, expected in freeze["file_hashes"].items():
        if sha256_file(checked_path(root, relative)) != expected:
            raise PermissionError(f"Frozen evidence changed: {relative}")
    source_root = Path(__file__).parent
    for relative, expected in freeze["source_hashes"].items():
        if sha256_file(checked_path(source_root, relative)) != expected:
            raise PermissionError(f"Frozen implementation changed: {relative}")
    return freeze


PANEL_SPLITS = {
    "FORMAT_BASE_TEST": "FORMAT_TUNE",
    "FORMAT_TUNE_POST_BRIDGE": "FORMAT_TUNE",
    "FORMAT_CHECK": "FORMAT_CHECK",
    "MEASUREMENT_AUDIT": "AUDIT_MEASURE",
}
FORMAT_STAGES = frozenset(PANEL_SPLITS) - {"MEASUREMENT_AUDIT"}


def _prefreeze_hash(root):
    return sha256_file(Path(root) / "manifests/PRE_INFERENCE_FREEZE.json")


def _check_dispatched(root):
    root = Path(root)
    return (
        (root / "manifests/STAGE_PLAN_FORMAT_CHECK.json").exists()
        or any((root / "raw/FORMAT_CHECK").glob("SHARD_*_STARTED.json"))
        or (root / "tables/FORMAT_CHECK_REPORT.json").exists()
    )


def _panel_questions(root, stage):
    if stage not in PANEL_SPLITS:
        raise PermissionError("Only registered sampling stages have shard plans")
    rows = [
        q
        for q in read_jsonl(Path(root) / "data/questions.jsonl")
        if q["split"] == PANEL_SPLITS[stage]
    ]
    expected = 384 if stage == "MEASUREMENT_AUDIT" else 96
    roots = 16 if stage == "MEASUREMENT_AUDIT" else 4
    if (
        len(rows) != expected
        or len({q["question_id"] for q in rows}) != expected
        or len({q["root_family_id"] for q in rows}) != roots
    ):
        raise PermissionError("Registered panel question/root counts are incomplete")
    return rows


def _bridge_trigger(root):
    root = Path(root)
    trigger = read_json(root / "manifests/BRIDGE_TRIGGER.json")
    if trigger.get("triggered") is not True or trigger.get("root_cause") != "FORMAT_COMPLIANCE":
        raise PermissionError("Format-only bridge trigger required")
    if trigger.get("pre_freeze_hash") != _prefreeze_hash(root):
        raise PermissionError("Bridge trigger identity mismatch")
    base = verify_format_report(root, "FORMAT_BASE_TEST")
    if not base["coverage"]["complete"] or base["gate_passed"]:
        raise PermissionError("Bridge requires complete failed base format check")
    if trigger.get("base_report_hash") != sha256_file(root / "tables/FORMAT_BASE_TEST_REPORT.json"):
        raise PermissionError("Bridge trigger base report identity mismatch")
    return trigger


def _bridge_receipt(root):
    from .vl_runtime import hash_json

    root = Path(root)
    _bridge_trigger(root)
    receipt = read_json(root / "engineering/BRIDGE_RECEIPT.json")
    if (
        receipt.get("status") != "COMPLETED"
        or receipt.get("physical_updates") != 16
        or receipt.get("sequence_exposures") != 128
    ):
        raise PermissionError("Fixed bridge endpoint is not complete")
    if receipt.get("pre_freeze_hash") != _prefreeze_hash(root) or receipt.get(
        "trigger_hash"
    ) != sha256_file(root / "manifests/BRIDGE_TRIGGER.json"):
        raise PermissionError("Bridge receipt provenance mismatch")
    adapter = Path(receipt["adapter_path"]).resolve()
    adapter.relative_to(root.resolve())
    hashes = receipt.get("file_hashes", {})
    if not hashes or receipt.get("adapter_hash") != hash_json(hashes):
        raise PermissionError("Bridge adapter manifest mismatch")
    actual_files = {str(p.relative_to(adapter)) for p in adapter.rglob("*") if p.is_file()}
    if set(hashes) != actual_files:
        raise PermissionError("Bridge adapter file set changed")
    for name, expected in hashes.items():
        if sha256_file(checked_path(adapter, name)) != expected:
            raise PermissionError("Bridge adapter bytes changed")
    return receipt


def _common_payload(root):
    root = Path(root)
    freeze = _verify_freeze(root)
    base = verify_format_report(root, "FORMAT_BASE_TEST")
    if not base["coverage"]["complete"]:
        raise PermissionError("A complete base format report is required")
    payload = dict(
        status="VERIFIED",
        pre_freeze_hash=_prefreeze_hash(root),
        base_report_hash=sha256_file(root / "tables/FORMAT_BASE_TEST_REPORT.json"),
    )
    if base["gate_passed"]:
        if (root / "engineering/BRIDGE_RECEIPT.json").exists():
            raise PermissionError("A passing base cannot be replaced by an untriggered bridge")
        payload.update(
            model_hash=freeze["model_weights_hash"],
            adapter_path=freeze.get("adapter_path"),
            source="verified_base_format_pass",
            untouched_base=True,
        )
    else:
        receipt = _bridge_receipt(root)
        payload.update(
            model_hash=object_hash(
                dict(
                    base_model_weights_hash=freeze["model_weights_hash"],
                    adapter_file_hashes=receipt["file_hashes"],
                )
            ),
            adapter_file_hashes=receipt["file_hashes"],
            adapter_path=receipt["adapter_path"],
            adapter_hash=receipt["adapter_hash"],
            bridge_receipt_hash=sha256_file(root / "engineering/BRIDGE_RECEIPT.json"),
            source="one_fixed_common_format_bridge",
            untouched_base=False,
        )
    return payload


def create_common_start(run_root):
    root = Path(run_root)
    with locked(root / "manifests/STAGE_STATE.lock"):
        if _check_dispatched(root):
            raise PermissionError("Independent check already dispatched; common start is sealed")
        common = _common_payload(root)
        path = root / "manifests/COMMON_START.json"
        if path.exists():
            if read_json(path) != common:
                raise PermissionError("Common start cannot be replaced")
        else:
            exclusive_json(path, common)
    return common


def verify_common_start(run_root):
    root = Path(run_root)
    common = read_json(root / "manifests/COMMON_START.json")
    if common != _common_payload(root):
        raise PermissionError("Common start provenance or model identity changed")
    return common


def _stage_identity(root, stage):
    root = Path(root)
    freeze = _verify_freeze(root)
    if stage == "FORMAT_BASE_TEST":
        common_hash, model_hash = None, freeze["model_weights_hash"]
    else:
        common = verify_common_start(root)
        common_hash = sha256_file(root / "manifests/COMMON_START.json")
        model_hash = common["model_hash"]
    return dict(
        pre_freeze_hash=_prefreeze_hash(root),
        common_start_hash=common_hash,
        model_hash=model_hash,
        questions_sha256=sha256_file(root / "data/questions.jsonl"),
    )


def _plan_payload(root, stage, shards):
    if type(shards) is not int or not 1 <= shards <= 5:
        raise ValueError("Registered shard count must be in 1..5")
    identity = _stage_identity(root, stage)
    rows = _panel_questions(root, stage)
    slots = [
        dict(
            question_id=q["question_id"],
            sample_index=k,
            shard=i % shards,
            request_id=object_hash([stage, identity["model_hash"], q["question_id"], k]),
            seed=derive_seed(stage, identity["model_hash"], q, k),
        )
        for i, q in enumerate(rows)
        for k in range(4)
    ]
    return dict(
        stage=stage,
        shards=shards,
        question_ids=[q["question_id"] for q in rows],
        expected_slots=slots,
        **identity,
    )


def register_stage_plan(run_root, stage, shards):
    root = Path(run_root)
    with locked(root / "manifests/STAGE_STATE.lock"):
        verify_gate(root, stage)
        plan = _plan_payload(root, stage, shards)
        path = root / f"manifests/STAGE_PLAN_{stage}.json"
        if path.exists():
            if read_json(path) != plan:
                raise PermissionError(
                    "Immutable stage plan differs from requested shard/model identity"
                )
        else:
            exclusive_json(path, plan)
    return plan


def _load_stage_plan(root, stage):
    root = Path(root)
    plan = read_json(root / f"manifests/STAGE_PLAN_{stage}.json")
    if plan != _plan_payload(root, stage, plan.get("shards")):
        raise PermissionError("Stage plan provenance changed")
    return plan


def verify_stage_plan(run_root, stage, shard, shards):
    if type(shard) is not int or type(shards) is not int or not 0 <= shard < shards <= 5:
        raise ValueError("Invalid registered shard")
    verify_gate(run_root, stage)
    plan = _load_stage_plan(run_root, stage)
    if plan["shards"] != shards:
        raise PermissionError("Shard count differs from immutable stage plan")
    return plan


def _format_report_payload(root, stage):
    from .scoring import build_format_report

    if stage not in FORMAT_STAGES:
        raise ValueError("Only the three format panels produce a format gate")
    root = Path(root)
    plan = _load_stage_plan(root, stage)
    rows = _panel_questions(root, stage)
    by_question = {row["question_id"]: row for row in rows}
    expected = {
        (slot["question_id"], slot["sample_index"]): slot for slot in plan["expected_slots"]
    }
    outputs, seen, files, missing_shards = [], set(), {}, []
    raw_dir = root / f"raw/{stage}"
    if set(p.name for p in raw_dir.glob("outputs_*.jsonl")) - {
        f"outputs_{s}.jsonl" for s in range(plan["shards"])
    }:
        raise PermissionError("Unregistered extra output shard")
    processor_hash = read_json(root / "manifests/PRE_INFERENCE_FREEZE.json")["processor_hash"]
    for shard in range(plan["shards"]):
        path = raw_dir / f"outputs_{shard}.jsonl"
        claim_path = raw_dir / f"SHARD_{shard}_STARTED.json"
        receipt_path = raw_dir / f"SHARD_{shard}_COMPLETE.json"
        if not path.exists() or not claim_path.exists() or not receipt_path.exists():
            missing_shards.append(shard)
        for artifact in (path, claim_path, receipt_path):
            if artifact.exists():
                files[str(artifact.relative_to(root))] = sha256_file(artifact)
        if claim_path.exists():
            claim = read_json(claim_path)
            if (
                claim.get("stage") != stage
                or claim.get("shard") != shard
                or claim.get("shards") != plan["shards"]
                or claim.get("freeze_hash") != plan["pre_freeze_hash"]
            ):
                raise PermissionError("Shard claim provenance mismatch")
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if (
                not path.exists()
                or receipt.get("status") != "completed"
                or receipt.get("stage") != stage
                or receipt.get("shard") != shard
                or receipt.get("shards") != plan["shards"]
                or receipt.get("output_hash") != sha256_file(path)
            ):
                raise PermissionError("Shard completion receipt provenance mismatch")
        local = read_jsonl(path)
        for output in local:
            slot_key = (output.get("question_id"), output.get("sample_index"))
            slot = expected.get(slot_key)
            if slot is None or slot_key in seen or slot["shard"] != shard:
                raise PermissionError("Missing, duplicate, or incorrectly sharded sample identity")
            seen.add(slot_key)
            q = by_question[slot["question_id"]]
            if (
                output.get("stage") != stage
                or output.get("request_id") != slot["request_id"]
                or output.get("seed") != slot["seed"]
                or output.get("model_hash") != plan["model_hash"]
                or output.get("processor_hash") != processor_hash
                or output.get("image_sha256") != q["image_sha256"]
                or output.get("prompt_hash") != q["prompt_sha256"]
            ):
                raise PermissionError("Raw output request provenance mismatch")
            outputs.append(output)
        if receipt_path.exists() and read_json(receipt_path).get("completed") != sum(
            output.get("status") == "completed" for output in local
        ):
            raise PermissionError("Shard receipt completion count mismatch")
    report = build_format_report(rows, outputs, expected_k=4, stage=stage)
    failed_attempts = sum(output.get("status") != "completed" for output in outputs)
    complete = seen == set(expected) and not missing_shards and not failed_attempts
    report["coverage"] = dict(
        complete=complete,
        expected_slots=len(expected),
        recorded_slots=len(seen),
        failed_attempts=failed_attempts,
        missing_shards=missing_shards,
        missing_slots=[list(x) for x in sorted(set(expected) - seen)],
    )
    if not complete:
        report.update(gate_passed=False, status="INCOMPLETE")
    report["provenance"] = dict(
        **_stage_identity(root, stage),
        stage=stage,
        stage_plan_hash=sha256_file(root / f"manifests/STAGE_PLAN_{stage}.json"),
        raw_file_hashes=files,
    )
    return report


def bind_format_report(run_root, stage):
    root = Path(run_root)
    with locked(root / "manifests/STAGE_STATE.lock"):
        report = _format_report_payload(root, stage)
        path = root / f"tables/{stage}_REPORT.json"
        if path.exists():
            if read_json(path) != report:
                raise PermissionError("Immutable format report evidence changed")
        else:
            exclusive_json(path, report)
    return report


def verify_format_report(run_root, stage):
    root = Path(run_root)
    stored = read_json(root / f"tables/{stage}_REPORT.json")
    if stored != _format_report_payload(root, stage):
        raise PermissionError("Format report evidence or provenance changed")
    return stored


def verify_gate(run_root, stage):
    root = Path(run_root)
    if stage not in STAGES:
        raise PermissionError(f"Stage is not authorized: {stage}")
    freeze = _verify_freeze(root)
    if stage in {"BRIDGE", "FORMAT_TUNE_POST_BRIDGE"}:
        if _check_dispatched(root):
            raise PermissionError("Independent check already dispatched; no more format training")
        _bridge_trigger(root)
        if stage == "FORMAT_TUNE_POST_BRIDGE":
            _bridge_receipt(root)
            verify_common_start(root)
    if stage in {"FORMAT_CHECK", "MEASUREMENT_AUDIT", "ENGINE"}:
        common = verify_common_start(root)
        if stage == "FORMAT_CHECK" and not common["untouched_base"]:
            if not (root / "tables/FORMAT_TUNE_POST_BRIDGE_REPORT.json").exists():
                raise PermissionError("Post-bridge descriptive format retest must complete")
            post = verify_format_report(root, "FORMAT_TUNE_POST_BRIDGE")
            if not post["coverage"]["complete"]:
                raise PermissionError("Post-bridge descriptive format retest must complete")
    if stage in {"MEASUREMENT_AUDIT", "ENGINE"}:
        report = verify_format_report(root, "FORMAT_CHECK")
        if report.get("gate_passed") is not True or report.get("status") != "PASS":
            raise PermissionError("Independent format check did not pass")
    if stage == "ENGINE":
        eng = read_json(root / "manifests/ENGINE_TEST_FREEZE.json")
        if eng.get("status") != "FROZEN" or eng.get("matching_prior_receipt") is not False:
            raise PermissionError("Conditional engine freeze required")
        if eng.get("pre_freeze_hash") != _prefreeze_hash(root):
            raise PermissionError("Engine freeze identity mismatch")
    return freeze


def _validate_preflight(root, env, qa, splits):
    from .generator import validate_dataset

    root = Path(root)
    preflight_path = root / "manifests/PROCESSOR_PREFLIGHT.json"
    preflight = read_json(preflight_path)
    if (
        qa.get("status") != "PASS"
        or qa.get("preflight_hash") != sha256_file(preflight_path)
        or preflight.get("status") != "CPU_PASS_PENDING_VISUAL"
    ):
        raise PermissionError("Readability review is not bound to this processor preflight")
    _verify_model_override(env)
    _verify_model_override(preflight)
    if (
        splits.get("status") != "PASS"
        or splits.get("processed_image_audit") != "PASS_ALL_REGISTERED_IMAGES"
    ):
        raise PermissionError("Complete actual processor split audit required")
    for field in (
        "processor_hash",
        "chat_template_hash",
        "tokenizer_hash",
        "environment_lock_hash",
        "chat_template_kwargs_hash",
        "linear_kernel_identity_hash",
    ):
        if not env.get(field) or env[field] != preflight.get(field):
            raise PermissionError(f"Preflight identity mismatch: {field}")
    from .vl_runtime import hash_json

    kernels = env.get("linear_kernel_identity")
    if (
        not isinstance(kernels, dict)
        or kernels != preflight.get("linear_kernel_identity")
        or hash_json(kernels) != env["linear_kernel_identity_hash"]
    ):
        raise PermissionError("Actual hybrid attention kernels must be frozen")
    if env["environment_lock_hash"] != sha256_file(
        root / "manifests/PROCESSOR_ENVIRONMENT_LOCK.json"
    ):
        raise PermissionError("Processor environment lock changed")
    generation = env.get("generation_config_expanded")
    if (
        not isinstance(generation, dict)
        or generation != preflight.get("generation_config_expanded")
        or any(generation.get(key) != value for key, value in GENERATION.items())
        or not generation.get("eos_token_id")
    ):
        raise PermissionError("Complete expanded generation config is not frozen consistently")
    if (
        preflight.get("source_image_count") != 288
        or preflight.get("processed_image_count") != 288
        or preflight.get("question_count") != 864
        or preflight.get("processed_size") != [896, 672]
        or preflight.get("min_raster_grid_gap_pixels", 0) < 4
        or preflight.get("min_pixels_per_delta", 0) < 4
        or splits.get("processed_image_count") != 288
        or splits.get("processed_pixel_count") != 288
        or splits.get("processor_hash") != env["processor_hash"]
    ):
        raise PermissionError("Actual processor coverage or readability geometry incomplete")
    hashes = preflight.get("artifact_sha256", {})
    required = {
        f"data/{name}.jsonl"
        for name in (
            "questions",
            "sidecars",
            "model_inputs",
            "images",
            "processed_images",
            "processor_routing",
            "sources",
        )
    }
    if not required <= hashes.keys():
        raise PermissionError("Processor preflight data manifest incomplete")
    for relative, expected in hashes.items():
        if sha256_file(checked_path(root, relative)) != expected:
            raise PermissionError(f"Processor preflight artifact changed: {relative}")
    for path_key, hash_key in (
        ("processed_manifest_path", "processed_manifest_sha256"),
        ("question_routing_path", "question_routing_sha256"),
    ):
        if sha256_file(checked_path(root, preflight[path_key])) != preflight[hash_key]:
            raise PermissionError("Actual processor routing manifest changed")
    if splits.get("processed_manifest_sha256") != preflight["processed_manifest_sha256"]:
        raise PermissionError("Split audit processed manifest mismatch")
    if preflight.get("generator_report_sha256") != sha256_file(
        root / "manifests/GENERATOR_REPORT.json"
    ):
        raise PermissionError("Generator report changed after processor preflight")
    validation = validate_dataset(root)
    if validation.get("status") != "PASS" or validation.get("processed_images_checked") != 288:
        raise PermissionError("Actual frozen dataset validation did not pass")
    return preflight


def freeze_audit(run_root):
    root = Path(run_root)
    env = read_json(root / "manifests/ENVIRONMENT.json")
    stop = read_json(root / "manifests/SER_J23_STOP_RECEIPT.json")
    qa = read_json(root / "manifests/PROCESSOR_READABILITY_REVIEW.json")
    splits = read_json(root / "manifests/ROOT_SPLIT_AUDIT.json")
    preflight = _validate_preflight(root, env, qa, splits)
    paths = [
        "data/questions.jsonl",
        "data/model_inputs.jsonl",
        "data/sidecars.jsonl",
        "manifests/ROOT_SPLIT_AUDIT.json",
        "manifests/GENERATOR_REPORT.json",
        "manifests/PROCESSOR_READABILITY_REVIEW.json",
        "manifests/ENVIRONMENT.json",
        "manifests/SER_J23_STOP_RECEIPT.json",
        "manifests/RESOURCE_OVERRIDE.json",
        "manifests/PROCESSOR_PREFLIGHT.json",
        "manifests/PROCESSOR_ENVIRONMENT_LOCK.json",
    ]
    paths.extend(preflight["artifact_sha256"])
    for row in read_jsonl(root / "data/questions.jsonl"):
        image_path = checked_path(root, row["image_path"])
        if sha256_file(image_path) != row["image_sha256"]:
            raise ValueError("Image identity mismatch")
        paths.append(row["image_path"])
        processed_path = checked_path(root, row["processed_image_path"])
        if sha256_file(processed_path) != row["processed_image_hash"]:
            raise PermissionError("Processed image changed after visual review")
        paths.append(row["processed_image_path"])
    source = Path(__file__).parent
    freeze = {
        **env,
        "plan_id": "MM-CORE-F1-20261008",
        "status": "FROZEN",
        "time": utc_now(),
        "dev_authorized": False,
        "authorized_stage": "MM-AUDIT",
        "old_work_stop_status": stop["status"],
        "readability_review_status": "PASS",
        "generation_config_expanded": env["generation_config_expanded"],
        "processor_preflight_hash": sha256_file(root / "manifests/PROCESSOR_PREFLIGHT.json"),
        "seed_derivation_version": "sha256-json-stage-model-root-image-question-index-v1",
        "protocol_route": "B_NEW_REGISTERED_DUAL_FIELDS",
        "file_hashes": {p: sha256_file(checked_path(root, p)) for p in sorted(set(paths))},
        "source_hashes": {p.name: sha256_file(p) for p in sorted(source.glob("*.py"))},
    }
    for key in (
        "model_weights_hash",
        "processor_hash",
        "environment_lock_hash",
        "chat_template_hash",
        "model_revision",
        "project_commit",
        "account_and_resource_permission_verified",
    ):
        if not freeze.get(key):
            raise PermissionError(f"Cannot freeze unresolved {key}")
    if stop["status"] not in {"already_stopped", "cancelled_verified"}:
        raise PermissionError("Old work is not verified stopped")
    exclusive_json(root / "manifests/PRE_INFERENCE_FREEZE.json", freeze)
    append_jsonl(
        root / "accounting/events.jsonl",
        {"time": utc_now(), "event": "PRE_INFERENCE_FROZEN", "hash": object_hash(freeze)},
    )
    return freeze
