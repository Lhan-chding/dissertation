"""User-authorized registered DEV execution over the immutable original runtime.

Existing registration/admission receipts are pinned, not reassessed. This module
neither runs delivery/recovery audits nor grants training, FORMAL or TEST access.
Static runtime/access copies preserve actual tensors, ledger locks and budgets.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from sr_f6 import access, evaluation, phases
from sr_f6 import pilot_v3_inputs as overlay
from sr_f6 import runtime as native
from sr_f6.transactions import _plain_path, _read_bytes, _require, _sha

PHASE_SHA = "d9f061d1cf08e14c485083bbed421a412a29868fd8305ad4c265fb6404562284"
CODE = Path("/projects/_ssd/varunssd/louis-ssvc/sr_f6_code_b7e7b06")
ROOT = Path("/projects/_ssd/varunssd/louis-ssvc/sr_f6_semantic_factorial_20261027")
PREFIX = "scripts/sr_f6_dev_eval/"
HISTORY_SCOPE = "EXISTING_PINNED_RECEIPTS_NOT_REASSESSED"


def _bytes(path):
    path = _plain_path(path)
    before = phases._stat(path)
    payload = _read_bytes(path)
    _require(phases._stat(path) == before, "Dependency changed during read")
    return payload


def _bound(root, ref):
    _require(
        isinstance(ref, dict) and set(ref) == {"path", "sha256"} and _sha(ref["sha256"]),
        "Exact byte reference required",
    )
    path = Path(ref["path"])
    path = _plain_path(path if path.is_absolute() else Path(root) / path)
    _require(
        path.is_relative_to(_plain_path(root)) and path != root, "Reference escapes bound root"
    )
    payload = _bytes(path)
    _require(
        hashlib.sha256(payload).hexdigest() == ref["sha256"], "Pinned bytes changed: " + str(path)
    )
    return json.loads(payload)


class ReadOnlyHashes:
    def __init__(self, root):
        self.directory = Path(root) / ".verification-cache"

    def adopt_verified(self, path, expected_sha256, previous):
        return None

    def hash(self, path):
        return hashlib.sha256(_bytes(path)).hexdigest()


def _context_readonly(root, supplied=None):
    root = _plain_path(root)
    _require(
        root == ROOT
        and native._actual_deployment() == CODE
        and overlay._actual_deployment() == CODE,
        "Actual frozen b7 source/root required",
    )
    identity = _bound(root, dict(path="PHASES/PILOT_V3/IDENTITY.json", sha256=PHASE_SHA))
    _require(
        identity["phase"] == "PILOT_V3"
        and identity["schema"] == phases.IDENTITY_SCHEMA
        and identity["deployment"] == str(CODE)
        and identity["root"] == str(root)
        and identity["status"] == "BOUND_NOT_AUTHORIZED"
        and identity["training_allowed"] is False,
        "Pinned phase identity differs",
    )
    phases._deployment(CODE, identity["revision"])
    manifest = _bound(root, identity["source_manifest"])
    _require(manifest["schema"] == phases.MANIFEST_SCHEMA, "Unexpected source manifest")
    phases._verify_manifest(CODE, manifest, ReadOnlyHashes(root), source=True)
    origin = _bound(root, identity["origin"]["identity"])
    _require(
        origin["source_manifest"] == identity["origin"]["source_manifest"],
        "Origin source reference differs",
    )
    before = _bound(root, identity["origin"]["source_manifest"])
    phases._validate_delta(before, manifest, _bound(root, identity["delta"]))
    for upstream in identity["upstream"]:
        for ref in upstream["evidence"]:
            _bound(root, ref)
    config = _bound(root, dict(path="CONFIG.json", sha256=identity["config_sha256"]))
    _require(
        evaluation.digest(config) == identity["config_canonical_sha256"],
        "Configuration identity differs",
    )
    _bound(root, identity["data_manifest"])
    _require(
        identity["implementation_digest"] == phases.digest(phases.AMENDMENTS)
        and phases.digest(json.loads(_bytes(root / "IMPLEMENTATION_REGISTRATION.json")))
        == phases.digest(phases.AMENDMENTS),
        "Original implementation registration differs",
    )
    native._verify_module_origins(CODE, manifest, tuple(sys.modules.values()))
    context = dict(
        phase="PILOT_V3",
        status="BOUND_NOT_AUTHORIZED",
        training_allowed=False,
        deployment=str(CODE),
        identity=identity,
        identity_sha256=PHASE_SHA,
    )
    _require(supplied is None or supplied == context, "Supplied source context differs")
    return context


def _operator(execution_source):
    _require(
        isinstance(execution_source, dict)
        and set(execution_source) == {"revision", "deployment", "binding_sha256"},
        "Explicit new execution source required",
    )
    deployment = _plain_path(execution_source["deployment"])
    _require(
        _plain_path(__file__) == deployment / PREFIX / "runtime_adapter.py",
        "Adapter must run from its actual independent deployment",
    )
    payload = _bytes(deployment / "DEV_SOURCE.json")
    manifest = json.loads(payload)
    encoded = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    _require(
        payload == encoded
        and hashlib.sha256(payload).hexdigest() == execution_source["binding_sha256"],
        "New source manifest differs",
    )
    _require(
        manifest.get("schema") == "srf6-dev-eval-source-v1"
        and manifest.get("revision") == execution_source["revision"]
        and manifest.get("deployment") == str(deployment)
        and set(manifest["files"])
        == {
            PREFIX + n
            for n in ("runtime_adapter.py", "operations.py", "driver.py", "launch.sbatch")
        },
        "Unexpected new source inventory",
    )
    for name, item in manifest["files"].items():
        data = _bytes(deployment / name)
        _require(
            set(item) == {"bytes", "sha256"}
            and type(item["bytes"]) is int
            and len(data) == item["bytes"]
            and hashlib.sha256(data).hexdigest() == item["sha256"],
            "Execution source bytes changed",
        )
    path = deployment / PREFIX / "operations.py"
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("srf6_dev_bound_operator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _require(
        module.identity(execution_source["revision"], execution_source["binding_sha256"])
        == execution_source,
        "Actual deployed source differs",
    )
    return module


def prepare_context(root, *, execution_source, authorization_ref):
    """Check fixed small receipts and actual source; never reassess historical gates."""
    op = _operator(execution_source)
    _require(
        _plain_path(root) == op.ROOT == ROOT
        and authorization_ref.get("path") == str(op.OPS / "CONTRACT.json"),
        "Exact user authorization contract required",
    )
    _require(set(authorization_ref) == {"path", "sha256"}, "Exact authorization reference required")
    op.metadata(execution_source, authorization_ref["sha256"])
    contract = _bound(op.OPS, authorization_ref)
    _require(
        op.encode(contract) == op.encode(op.contract_value(execution_source)),
        "Typed contract differs",
    )
    auth = contract["user_authorization"]
    _require(op.encode(auth) == op.encode(op.USER_AUTHORIZATION), "User DEV authorization differs")
    context = _context_readonly(root)
    spec = _bound(root, auth["spec_ref"])
    prior = _bound(root, auth["admission_ref"])
    jobs = _bound(root, auth["plan_ref"])
    relative_admission = dict(
        path=str(Path(auth["admission_ref"]["path"]).relative_to(root)),
        sha256=auth["admission_ref"]["sha256"],
    )
    _require(
        auth["phase_ref"]
        == dict(path=str(Path(root) / "PHASES/PILOT_V3/IDENTITY.json"), sha256=PHASE_SHA)
        and spec["phase_identity"] == dict(path="PHASES/PILOT_V3/IDENTITY.json", sha256=PHASE_SHA)
        and spec["phase"] == "PILOT_V3",
        "Original phase/spec pin differs",
    )
    identity = context["identity"]
    _require(
        spec["actual_source_manifest"] == identity["source_manifest"]
        and spec["data_manifest"] == identity["data_manifest"]
        and spec["config_sha256"] == identity["config_sha256"]
        and spec["origin_identity"] == identity["origin"]["identity"],
        "Spec source/input identity differs",
    )
    _require(
        prior["root"] == str(root)
        and prior["phase_identity_sha256"] == PHASE_SHA
        and prior["pilot_spec_sha256"] == auth["spec_ref"]["sha256"]
        and prior["input_overlay"] == spec["input_overlay"],
        "Pinned historical receipt identity differs",
    )
    _require(
        jobs["pins"]
        == dict(
            phase="PILOT_V3",
            phase_identity_sha256=PHASE_SHA,
            pilot_spec_sha256=auth["spec_ref"]["sha256"],
            admission_ref=relative_admission,
        ),
        "Original job plan pins differ",
    )
    _bound(root, spec["input_overlay"])
    _bound(root, spec["source_binding"])
    plan = evaluation._plan(spec)
    _require(
        len(plan["endpoints"]) == auth["endpoints"] == 12
        and plan["responses"] == auth["responses"] == 2304,
        "Only the fixed original DEV budget is authorized",
    )
    return dict(
        status=auth["status"],
        scope="PILOT_V3_REGISTERED_ONLY",
        generation_authorized=True,
        evaluation_authorized=True,
        training_authorized=False,
        formal_authorized=False,
        test_authorized=False,
        delivery_audit_waived_by_user=True,
        full_delivery_verified=False,
        historical_receipts_scope=HISTORY_SCOPE,
        phase_context=context,
        pilot_spec=spec,
        pilot_spec_ref=auth["spec_ref"],
        admission_ref=relative_admission,
        plan=plan,
        jobs=jobs,
        execution_source=copy.deepcopy(execution_source),
        authorization_ref=copy.deepcopy(authorization_ref),
    )


def _admitted(
    root,
    phase_identity_sha256,
    pilot_spec_sha256,
    admission_ref,
    *,
    phase,
    execution_source,
    authorization_ref,
):
    result = prepare_context(
        root, execution_source=execution_source, authorization_ref=authorization_ref
    )
    _require(
        phase == "PILOT_V3"
        and phase_identity_sha256 == result["phase_context"]["identity_sha256"]
        and pilot_spec_sha256 == result["pilot_spec_ref"]["sha256"]
        and admission_ref == result["admission_ref"],
        "Caller changed the registered DEV pins",
    )
    return result


def _runtime_source(
    root, *, phase=None, phase_identity_sha256=None, execution_source, authorization_ref
):
    result = prepare_context(
        root, execution_source=execution_source, authorization_ref=authorization_ref
    )
    source = result["phase_context"]
    _require(
        phase == "PILOT_V3" and phase_identity_sha256 == PHASE_SHA,
        "Only fixed PILOT_V3 source allowed",
    )
    return {**source, "origin_identity_sha256": source["identity"]["origin"]["identity"]["sha256"]}


def _input_binding(root, source, pilot_spec_sha256, *, execution_source, authorization_ref):
    result = prepare_context(
        root, execution_source=execution_source, authorization_ref=authorization_ref
    )
    context = {k: v for k, v in source.items() if k != "origin_identity_sha256"}
    _require(
        context == result["phase_context"]
        and pilot_spec_sha256 == result["pilot_spec_ref"]["sha256"],
        "Runtime input identity differs",
    )
    projected = _verify_overlay_readonly(
        root, phase_context=context, overlay_ref=result["pilot_spec"]["input_overlay"]
    )
    return dict(
        pilot_spec_sha256=pilot_spec_sha256,
        input_overlay=copy.deepcopy(projected["manifest_ref"]),
        input_overlay_sha256=projected["manifest_ref"]["sha256"],
        prompt_version=projected["manifest"]["prompt_version"],
    )


def _access_authorization(
    runtime, root, mode, pilot_spec_sha256, admission_ref, *, execution_source, authorization_ref
):
    _require(mode == "EVALUATION", "User authorization is DEV evaluation only")
    bound = access._loaded(runtime, root, mode)
    result = _admitted(
        root,
        runtime.identity["phase_identity_sha256"],
        pilot_spec_sha256,
        admission_ref,
        phase="PILOT_V3",
        execution_source=execution_source,
        authorization_ref=authorization_ref,
    )
    context = result["phase_context"]
    _require(
        result["status"] == "USER_AUTHORIZED_REGISTERED_DEV_EVAL"
        and result["scope"] == "PILOT_V3_REGISTERED_ONLY"
        and result["generation_authorized"] is True
        and result["training_authorized"] is False
        and result["formal_authorized"] is False
        and result["test_authorized"] is False
        and result["full_delivery_verified"] is False
        and result["delivery_audit_waived_by_user"] is True
        and context["identity"] == bound["source"]["identity"]
        and context["identity_sha256"] == bound["source"]["identity_sha256"]
        and context["deployment"] == bound["source"]["deployment"],
        "DEV authority does not bind the actually loaded source",
    )
    spec = result["pilot_spec"]
    _require(
        runtime.identity.get("pilot_spec_sha256") == pilot_spec_sha256
        and runtime.identity.get("input_overlay") == spec["input_overlay"]
        and runtime.identity.get("input_overlay_sha256") == spec["input_overlay"]["sha256"]
        and runtime.identity.get("prompt_version") == spec["prompt_version"],
        "Actual V3 runtime input binding differs",
    )
    return result


def _build_overlay_readonly(root, *, phase_context):
    """Reconstruct both complete public projections, without publishing anything."""
    root = overlay._plain_path(root)
    context = _context_readonly(root, phase_context)
    identity = context["identity"]
    reader = overlay.SimpleNamespace()
    config = overlay._bound(
        reader, root, {"path": "CONFIG.json", "sha256": identity["config_sha256"]}
    )
    data = overlay._bound(reader, root, identity["data_manifest"])
    overlay._require(
        data.get("config_sha256") == overlay.digest(config), "Original DATA/config differs"
    )
    files = data.get("files")
    overlay._require(
        isinstance(files, list) and all(isinstance(item, dict) for item in files),
        "Original DATA inventory required",
    )
    inventory = {item["path"]: item["sha256"] for item in files}
    overlay._require(len(inventory) == len(files), "Duplicate original DATA member")

    def data_ref(relative):
        overlay._require(overlay._sha(inventory.get(relative)), "Missing original DATA member")
        return {"path": relative, "sha256": inventory[relative]}

    schedules = {}
    for parent in overlay.registration.PARENTS:
        reference = data_ref(f"DATA/SCHEDULES/DEV_PILOT-{parent}-r0.json")
        overlay._require(
            overlay._bound(reader, root, reference)
            == overlay.root_schedule(8, config["master_seed"], parent, 0, split="DEV_PILOT"),
            "Original pilot schedule differs from frozen seed generator",
        )
        schedules[parent] = reference
    (projected, artifacts) = ({}, {})
    for split in overlay.SPLITS:
        (projected[split], payload) = overlay._split(reader, root, config, data_ref, split)
        artifacts[f"{split}/MODEL_INPUTS.jsonl"] = payload
    manifest = dict(
        schema=overlay.SCHEMA,
        phase=overlay.PHASE,
        status="INPUT_OVERLAY_NOT_AUTHORIZED",
        phase_identity={
            "path": f"PHASES/{overlay.PHASE}/IDENTITY.json",
            "sha256": context["identity_sha256"],
        },
        actual_source_manifest=identity["source_manifest"],
        origin_identity=identity["origin"]["identity"],
        data_manifest=identity["data_manifest"],
        config_sha256=identity["config_sha256"],
        prompt_version=overlay.PROMPT_VERSION,
        wire_protocol=overlay.interface_v3_protocol.WIRE_PROTOCOL,
        constructor="sr_f6.interface_v3_protocol.public_prompt(categories, query)",
        public_inputs=["world.categories", "query"],
        generator_sources=overlay._source_members(
            root, context, (overlay, overlay.interface_v3_protocol, overlay.registration)
        ),
        splits=projected,
        schedules=schedules,
        permitted_changes=["model_input.prompt", "prompt_version", "prompt_sha256_if_present"],
        old_data_modified=False,
        training_authorized=False,
        generation_authorized=False,
        formal_test_access=False,
        execution_status="NOT_RUN",
    )
    artifacts[overlay.MANIFEST_FILE] = overlay.registration._payload(manifest)
    reference = overlay.registration._reference(
        f"{overlay.DIRECTORY}/{overlay.MANIFEST_FILE}", manifest
    )
    return dict(
        manifest_ref=reference,
        manifest=manifest,
        inputs_by_split={
            split: {name: projected[split][name] for name in ("public", "private")}
            for split in overlay.SPLITS
        },
        files=artifacts,
    )


def _verify_overlay_readonly(root, *, phase_context, overlay_ref):
    """Verify the manifest pin, exact membership and all original row projections."""
    root = overlay._plain_path(root)
    overlay._require(
        isinstance(overlay_ref, dict)
        and set(overlay_ref) == {"path", "sha256"}
        and (overlay_ref.get("path") == f"{overlay.DIRECTORY}/{overlay.MANIFEST_FILE}")
        and overlay._sha(overlay_ref.get("sha256")),
        "Pinned PILOT_V3 overlay manifest required",
    )
    overlay._bound(overlay.SimpleNamespace(), root, overlay_ref)
    bundle = _build_overlay_readonly(root, phase_context=phase_context)
    overlay._require(
        bundle["manifest_ref"] == overlay_ref, "Overlay differs from fixed reconstruction"
    )
    overlay._check_files(overlay._plain_path(root / overlay.DIRECTORY), bundle)
    return {key: bundle[key] for key in ("manifest_ref", "manifest", "inputs_by_split")}


def _inputs(root, plan, *, spec=None, phase_context=None):
    phase = "PILOT_V1" if spec is None else spec["phase"]
    contract = evaluation.pilot_contract.for_phase(phase)
    public_path = "DATA/DEV_EVAL/MODEL_INPUTS.jsonl"
    if contract.phase == "PILOT_V3":
        evaluation._require(
            phase_context is not None and phase_context.get("phase") == contract.phase,
            "Verified V3 phase context required for evaluation overlay",
        )
        overlay = _verify_overlay_readonly(
            root, phase_context=phase_context, overlay_ref=spec["input_overlay"]
        )
        evaluation._require(
            plan["inputs"] == overlay["inputs_by_split"]["DEV_EVAL"],
            "Evaluation inputs differ from verified V3 overlay",
        )
        public_path = overlay["inputs_by_split"]["DEV_EVAL"]["public"]["path"]
    evaluation._require(
        plan["inputs"]["public"]["path"] == public_path
        and plan["inputs"]["private"]["path"] == "DATA/DEV_EVAL/TASKS_PRIVATE.jsonl",
        "Only DEV_EVAL input files may be accessed",
    )
    public = evaluation._bound(root, plan["inputs"]["public"], lines=True)
    private = evaluation._bound(root, plan["inputs"]["private"], lines=True)
    (items, tasks) = ({r["qid"]: r for r in public}, {r["qid"]: r for r in private})
    expected = {r["qid"] for r in plan["seeds_by_parent"]["b0"]}
    evaluation._require(
        len(public) == len(private) == len(items) == len(tasks) == 96
        and set(items) == set(tasks) == expected,
        "Complete frozen DEV_EVAL matrix required",
    )
    for qid, item in items.items():
        task = tasks[qid]
        evaluation._require(
            item["split"] == task["split"] == "DEV_EVAL"
            and item["protocol"] == "structured_v4"
            and (set(item["model_input"]) == {"prompt", "image_file", "image_sha256"}),
            "Public model input whitelist/panel differs",
        )
        evaluation._require(
            all(item[k] == task[k] for k in ("qid", "root_id", "family", "factor", "variant")),
            "Evaluation task lineage differs",
        )
    return (items, tasks)


def _load(
    root,
    parent,
    task_id,
    *,
    training=None,
    mode=None,
    phase=None,
    phase_identity_sha256=None,
    audit_root=None,
    run_id=None,
    pilot_spec_sha256=None,
    execution_source,
    authorization_ref,
):
    """Load a bound parent, with separate provenance and execution semantics.

    Default S0 loading retains its interface but verifies the actual historical
    deployment. New phases require an explicit full identity pin and audit root.
    They remain NOT_AUTHORIZED: phase scientific gates, evaluation-panel access
    locks, and orchestration are not implemented here, so generation is blocked.
    EVALUATION creates no optimizer and cannot perform gradient forwards.
    PILOT REPLAY keeps adapter gradients, creates no optimizer, and requires an
    active original-token session for every forward. It cannot sample/update.
    """
    import torch

    from mm_dev.runtime import actual_cuda_identity, configure_audited_backend
    from sr_f3_ce.training import fresh_optimizer
    from sr_f6.common import write_once

    selected_mode = native._execution_mode(mode, training)
    _require(selected_mode == "EVALUATION", "This adapter authorizes DEV evaluation only")
    if selected_mode == "REPLAY" and phase != "PILOT_V1":
        raise PermissionError("REPLAY is restricted to the registered PILOT_V1 phase")
    if not isinstance(task_id, str) or not native.re.fullmatch(
        "[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", task_id
    ):
        raise PermissionError("Runtime task_id must be one safe execution identifier")
    if selected_mode == "EVALUATION" and phase is None:
        raise PermissionError("Evaluation requires an explicit registered execution phase")
    if phase is None and run_id is not None:
        raise PermissionError("Legacy S0 run_id remains its task_id")
    if phase is not None and selected_mode == "TRAINING" and (run_id is None):
        raise PermissionError("Phase training requires its stable scientific run_id")
    if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
        raise PermissionError("Explicit run_id must be nonempty")
    effective_run_id = task_id if run_id is None else run_id
    root = native.Path(root)
    source = _runtime_source(
        root,
        phase=phase,
        phase_identity_sha256=phase_identity_sha256,
        execution_source=execution_source,
        authorization_ref=authorization_ref,
    )
    input_binding = _input_binding(
        root,
        source,
        pilot_spec_sha256,
        execution_source=execution_source,
        authorization_ref=authorization_ref,
    )
    audit = native._audit_directory(root, source["phase"], audit_root)
    identity_path = native._runtime_artifact(audit, "runtime_identity", task_id + ".json")
    if phase is not None and identity_path.exists():
        raise FileExistsError("Preserve an existing runtime execution receipt; use a new task_id")
    config = native.read(root / "CONFIG.json")
    native.validate_generation_config(config)
    if parent not in config["model"]["parents"]:
        raise ValueError("Unknown registered parent")
    adapter = native.Path(config["model"]["parents"][parent])
    if not adapter.is_dir():
        raise FileNotFoundError(adapter)
    (bound, bound_files_check) = native.verify_binding_inputs(root, config, parent)
    configure_audited_backend()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each SR-F6 worker must own exactly one visible GPU")

    def construct():
        native.seed_all(native.seed_for(config["master_seed"], "load", parent))

        def account(kind, count, metadata):
            if type(count) is not int or count < 0:
                raise ValueError("Invalid model-call accounting")
            native.append(
                native._runtime_artifact(audit, "accounting", task_id + ".jsonl"),
                dict(
                    at=native.now(),
                    task_id=task_id,
                    parent=parent,
                    kind=kind,
                    count=count,
                    metadata=metadata,
                ),
            )

        runtime = native.Runtime(
            config["model"]["model_path"],
            adapter_path=str(adapter),
            output_root=root,
            protocol_revision="v2",
            account=account,
        )
        native.configure_parent_training(
            runtime, parent_adapter_config=native.read(adapter / "adapter_config.json")
        )
        source_dtype = runtime.identity["dtype"]
        runtime.model = runtime.model.float()
        parameter_dtypes = sorted(
            {
                str(v.dtype).removeprefix("torch.")
                for v in runtime.model.parameters()
                if v.is_floating_point()
            }
        )
        buffer_dtypes = sorted(
            {
                str(v.dtype).removeprefix("torch.")
                for v in runtime.model.buffers()
                if v.is_floating_point()
            }
        )
        if parameter_dtypes != ["float32"] or any(v != "float32" for v in buffer_dtypes):
            raise RuntimeError("Full FP32 conversion did not take effect")
        runtime._srf5_numerics = dict(native.NUMERICS_V2)
        runtime.identity.update(
            dtype="float32",
            numerics=dict(native.NUMERICS_V2),
            numerics_recipe_version=2,
            model_cast_from_dtype=source_dtype,
            forward_parameter_dtypes=parameter_dtypes,
            forward_float_buffer_dtypes=buffer_dtypes,
            input_float_dtype="float32",
            probability_dtype="float32",
            parent_id=parent,
            run_id=effective_run_id,
            step=0,
            snapshot=config["model"]["snapshot"],
            sr_f6_protocol_version=native.PROTOCOL_VERSION,
        )
        (_, post_load_files_check) = native.verify_binding_inputs(root, config, parent)
        if (
            _runtime_source(
                root,
                phase=phase,
                phase_identity_sha256=phase_identity_sha256,
                execution_source=execution_source,
                authorization_ref=authorization_ref,
            )
            != source
            or _input_binding(
                root,
                source,
                pilot_spec_sha256,
                execution_source=execution_source,
                authorization_ref=authorization_ref,
            )
            != input_binding
        ):
            raise PermissionError("Source identity changed during model loading")
        identity = source["identity"]
        tensor_hash = native.state_hash(native.trainable_state(runtime.model))
        runtime._srf6_adapter_parameter_names = tuple(
            name for (name, value) in runtime.model.named_parameters() if value.requires_grad
        )
        runtime._srf6_training = selected_mode == "TRAINING"
        runtime._srf6_execution_mode = selected_mode
        runtime._srf6_policy_identity = (0, tensor_hash)
        runtime._srf6_source_context = native.copy.deepcopy(source)
        runtime.identity.update(
            run_root=str(root.resolve(strict=True)),
            audit_root=str(audit),
            task_id=task_id,
            phase=source["phase"],
            execution_mode=selected_mode,
            source_sha256=identity["source_manifest"]["sha256"],
            source_hash=identity["source_manifest"]["sha256"],
            data_hash=identity["data_manifest"]["sha256"],
            config_sha256=identity["config_sha256"],
            origin_identity_sha256=source["origin_identity_sha256"],
            phase_identity_sha256=source["identity_sha256"] if phase is not None else None,
            source_binding_status="BOUND_NOT_AUTHORIZED",
            execution_authorization="NOT_AUTHORIZED"
            if phase is not None
            else "CALLER_CENSUS_GATE_REQUIRED",
            parent_content_hash=bound["parents"][parent]["content_sha256"],
            policy_hash=tensor_hash,
            reference_hash=tensor_hash,
            reference_parent=parent,
            **input_binding,
        )
        runtime._srf6_execution_identity = {
            key: runtime.identity[key] for key in native._execution_identity_fields(source["phase"])
        }
        trainable = [
            {"name": n, "shape": list(p.shape), "dtype": str(p.dtype)}
            for (n, p) in runtime.model.named_parameters()
            if p.requires_grad
        ]
        if selected_mode == "EVALUATION":
            native._freeze_evaluation(runtime, 0, tensor_hash)
        (optimizer, scheduler) = (
            fresh_optimizer(runtime.model) if selected_mode == "TRAINING" else (None, None)
        )
        writer = write_once if phase is not None else native.write
        writer(
            identity_path,
            dict(
                native.copy.deepcopy(runtime.identity),
                hardware=actual_cuda_identity(),
                inherited_adapter_parameters=trainable,
                trainable_parameters=[] if selected_mode == "EVALUATION" else trainable,
                fresh_optimizer=selected_mode == "TRAINING",
                source_binding_note="content binding and loaded identity are separate receipts",
                bound_files_before_load=bound_files_check,
                bound_files_after_load=post_load_files_check,
            ),
        )
        native._LOADED_RUNTIMES[runtime] = dict(
            execution=native.copy.deepcopy(runtime._srf6_execution_identity),
            source=native.copy.deepcopy(source),
            model=runtime.model,
            parent={
                key: runtime.identity[key]
                for key in (
                    "parent_id",
                    "reference_parent",
                    "reference_hash",
                    "parent_content_hash",
                    "config_sha256",
                )
            },
        )
        return (runtime, optimizer, scheduler)

    if selected_mode in {"EVALUATION", "REPLAY"}:
        rng = native.capture_rng()
        try:
            return construct()
        finally:
            native.restore_rng(rng)
    return construct()


def _load_committed_endpoint(
    runtime, tx, *, expected_commit_sha256, execution_source, authorization_ref
):
    """Copy a pinned, published adapter into an EVALUATION runtime, read-only.

    ``tx`` must already hold its exclusive RunTransactions lock. This does not
    create an optimizer, restore training RNG, run evaluation, or authorize any
    panel access. PILOT generation still requires a separately verified active
    evaluation access session after successful loading; FORMAL remains closed.
    """
    import json

    import torch

    from sr_f3_ce.training import finite_tree
    from sr_f6.common import write_once
    from sr_f6.transactions import RunTransactions, _cursor, _file_hash, _read_bytes

    if runtime.execution_mode() != "EVALUATION" or not isinstance(tx, RunTransactions):
        raise PermissionError("Endpoint loading requires EVALUATION and a real locked transaction")
    if not isinstance(expected_commit_sha256, str) or not native.re.fullmatch(
        "[0-9a-f]{64}", expected_commit_sha256
    ):
        raise PermissionError("An exact committed endpoint identity is required")
    identity = runtime.identity
    native._check_loaded_source_identity(runtime)
    if identity.get("phase") not in native.MODEL_PHASES:
        raise PermissionError("Endpoint evaluation requires its registered execution phase")
    loaded_execution = getattr(runtime, "_srf6_execution_identity", None)
    if loaded_execution is None or any(
        identity.get(key) != loaded_execution.get(key)
        for key in native._execution_identity_fields(identity.get("phase"))
    ):
        raise PermissionError("Endpoint runtime execution identity changed after loading")
    if not isinstance(identity.get("task_id"), str) or not native.re.fullmatch(
        "[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", identity["task_id"]
    ):
        raise PermissionError("Endpoint task_id is not a safe execution filename")
    audit = native._audit_directory(identity["run_root"], identity["phase"], identity["audit_root"])
    source = _runtime_source(
        identity["run_root"],
        phase=identity["phase"],
        phase_identity_sha256=identity["phase_identity_sha256"],
        execution_source=execution_source,
        authorization_ref=authorization_ref,
    )
    if source != runtime._srf6_source_context:
        raise PermissionError("Endpoint runtime source binding changed")
    native._audit_directory(identity["run_root"], identity["phase"], tx.directory)
    tx._locked()
    for key, expected in (
        ("phase", identity["phase"]),
        ("parent", identity["parent_id"]),
        ("source_sha256", identity["source_sha256"]),
        ("data_sha256", identity["data_hash"]),
        ("config_sha256", identity["config_sha256"]),
        ("reference_sha256", identity["reference_hash"]),
        ("experiment_id", identity["experiment_id"]),
    ):
        if tx.identity.get(key) != expected:
            raise PermissionError("Endpoint transaction differs from runtime " + key)
    if identity["phase"] == "PILOT_V3":
        for key in ("input_overlay", "input_overlay_sha256", "prompt_version"):
            if tx.identity.get(key) != identity[key]:
                raise PermissionError("Endpoint transaction V3 input binding differs")
    if tx.identity["run_id"] == identity["run_id"]:
        raise PermissionError(
            "Evaluation run and checkpoint producer must have distinct identities"
        )
    markers = tx.published()
    matched = [marker for marker in markers if marker["commit_sha256"] == expected_commit_sha256]
    if len(matched) != 1:
        raise PermissionError("Pinned endpoint is not a unique published commit")
    marker = matched[0]
    step = marker["step"]
    checkpoint = tx.checkpoints / f"step-{step:03d}"
    receipt_path = native._runtime_artifact(
        audit, "runtime_identity", identity["task_id"] + "-endpoint.json"
    )
    if receipt_path.exists():
        raise FileExistsError("Endpoint execution receipt already exists")
    rng = native.capture_rng()
    try:
        path = checkpoint / "state.pt"
        before = path.stat()
        if _file_hash(path) != marker["state_sha256"]:
            raise PermissionError("Endpoint checkpoint state content differs")
        state = torch.load(path, map_location="cpu", weights_only=True)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise PermissionError("Endpoint state changed while loading")
        if (
            state.get("identity") != tx.identity
            or state.get("step") != step
            or state.get("sampler_state") != marker["sampler_state"]
        ):
            raise PermissionError("Endpoint serialized metadata differs from its commit")
        parameters = dict(runtime.model.named_parameters())
        names = runtime._srf6_adapter_parameter_names
        policy = state.get("policy")
        if (
            not isinstance(policy, dict)
            or set(policy) != set(names)
            or set(marker.get("parameter_names", [])) != set(names)
            or (not finite_tree(policy))
            or (native.state_hash(policy) != marker["policy_sha256"])
        ):
            raise PermissionError("Endpoint adapter content/names differ")
        for name, value in policy.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.float32
                or parameters[name].dtype != torch.float32
                or (value.shape != parameters[name].shape)
            ):
                raise PermissionError("Endpoint adapter shape/dtype differs")
        if step:
            update_bytes = _read_bytes(checkpoint / "UPDATE.json")
            import hashlib

            if hashlib.sha256(update_bytes).hexdigest() != marker["update_sha256"]:
                raise PermissionError("Endpoint update receipt changed")
            update = json.loads(update_bytes)
            qids = tx._evidence(
                marker["references"],
                update,
                step,
                marker["policy_before_sha256"],
                marker["rng_sha256"],
            )
            if update.get("policy_after_sha256") != marker["policy_sha256"]:
                raise PermissionError("Endpoint update policy differs from commit")
            _cursor(
                state["sampler_state"],
                step,
                markers[step - 1]["sampler_state"],
                qids,
                marker["references"]["rollout"],
            )
        with torch.no_grad():
            for name, value in policy.items():
                parameters[name].copy_(value)
        actual = native.state_hash(
            {name: parameters[name].detach().cpu().clone() for name in names}
        )
        if actual != marker["policy_sha256"]:
            raise PermissionError("Actual loaded endpoint parameters differ")
        native._freeze_evaluation(runtime, step, actual)
        identity.update(
            step=step,
            policy_hash=actual,
            checkpoint_step=step,
            checkpoint_run_id=tx.identity["run_id"],
            checkpoint_commit_sha256=expected_commit_sha256,
            checkpoint_state_sha256=marker["state_sha256"],
            execution_authorization="NOT_AUTHORIZED",
        )
        report = {
            "status": "ENDPOINT_LOADED_NOT_AUTHORIZED",
            "runtime_identity": native.copy.deepcopy(identity),
            "checkpoint_identity": native.copy.deepcopy(tx.identity),
            "checkpoint": str(checkpoint),
            "optimizer_created": False,
            "training_rng_restored": False,
            "model_calls": 0,
            "optimizer_updates": 0,
        }
        write_once(receipt_path, report)
        native._LOADED_ENDPOINTS[runtime] = dict(
            tx=tx,
            run_id=tx.identity["run_id"],
            step=step,
            commit_sha256=expected_commit_sha256,
            policy_sha256=actual,
        )
        return report
    finally:
        native.restore_rng(rng)


@access.contextmanager
def _open_evaluation_access(
    runtime,
    *,
    experiment_root,
    pilot_spec_sha256,
    admission_ref,
    ledger,
    execution_source,
    authorization_ref,
):
    root = access._plain_path(experiment_root)
    admission = _access_authorization(
        runtime,
        root,
        "EVALUATION",
        pilot_spec_sha256,
        admission_ref,
        execution_source=execution_source,
        authorization_ref=authorization_ref,
    )
    contract = access.for_phase(admission["phase_context"]["phase"])
    access._require(
        type(ledger) is evaluation.EvaluationLedger, "Actual held EvaluationLedger required"
    )
    ledger._locked()
    plan = admission["pilot_spec"]["evaluation"]
    access._require(
        ledger.plan == plan
        and ledger.identity.get("execution_origin") == "SERVER_ACTUAL"
        and (ledger.root == root),
        "Synthetic or unregistered evaluation ledger",
    )
    endpoint = next(
        (e for e in plan["endpoints"] if e["endpoint_id"] == ledger.identity["endpoint_id"]), None
    )
    access._require(endpoint is not None and endpoint == ledger.endpoint, "Unregistered endpoint")
    access._require(
        ledger.directory
        == access._plain_path(root / contract.evaluation_directory / endpoint["endpoint_id"])
        and runtime.identity["audit_root"] == str(ledger.directory),
        "Evaluation directory differs",
    )
    (public, tasks) = _inputs(
        root, plan, spec=admission["pilot_spec"], phase_context=admission["phase_context"]
    )
    access._require(
        ledger.public == public
        and ledger.tasks == tasks
        and (ledger.schedule == plan["seeds_by_parent"][endpoint["parent"]]),
        "Evaluation input/seed panel changed",
    )
    mapping = {
        "evaluation_run_id": "run_id",
        "parent": "parent_id",
        "checkpoint_step": "step",
        "policy_tensor_sha256": "policy_hash",
        "phase_identity_sha256": "phase_identity_sha256",
        "source_sha256": "source_sha256",
        "data_sha256": "data_hash",
        "reference_sha256": "reference_hash",
    }
    access._require(
        all(ledger.identity.get(k) == runtime.identity.get(v) for (k, v) in mapping.items())
        and ledger.identity.get("pilot_spec_sha256") == pilot_spec_sha256
        and (ledger.identity.get("admission_ref") == admission_ref)
        and (ledger.identity.get("input_refs") == plan["inputs"]),
        "Evaluation loaded identity differs",
    )
    if contract.phase == "PILOT_V3":
        access._require(
            all(
                ledger.identity.get(key) == runtime.identity.get(key)
                for key in ("phase", "input_overlay", "input_overlay_sha256", "prompt_version")
            ),
            "Evaluation V3 source or input overlay changed",
        )
    access._require(
        runtime.identity["step"] == endpoint["checkpoint_step"]
        and runtime.identity["parent_id"] == endpoint["parent"],
        "Evaluation actual endpoint differs",
    )
    if endpoint["kind"] == "FROZEN_PARENT":
        access._require(
            runtime.policy_tensor_identity() == endpoint["tensor_sha256"]
            and runtime.identity.get("checkpoint_commit_sha256") is None,
            "Frozen parent differs",
        )
    else:
        from sr_f6.runtime import _LOADED_ENDPOINTS

        bound = _LOADED_ENDPOINTS.get(runtime)
        access._require(bound is not None, "Actually loaded committed endpoint required")
        bound["tx"]._locked()
        access._require(
            bound["tx"].directory
            == root / access.Path(endpoint["expected_commit_path"]).parent.parent.parent
            and bound["run_id"] == endpoint["run_id"]
            and (bound["step"] == 12)
            and (ledger.identity.get("checkpoint_commit_sha256") == bound["commit_sha256"])
            and (runtime.identity.get("checkpoint_commit_sha256") == bound["commit_sha256"])
            and (runtime.identity.get("checkpoint_run_id") == bound["run_id"]),
            "Committed PRIMARY endpoint binding differs",
        )
    session = access._new_session(runtime, root, ledger, admission, "EVALUATION")
    session._endpoint_binding = bound if endpoint["kind"] != "FROZEN_PARENT" else None
    session._evaluation_inputs = {
        qid: access.copy.deepcopy(row["model_input"]) for (qid, row) in public.items()
    }
    session._evaluation_schedule = {row["qid"]: list(row["seeds"]) for row in ledger.schedule}
    access._ACTIVE[id(runtime)] = session
    try:
        session.assert_active()
        yield session
    finally:
        access._ACTIVE.pop(id(runtime), None)


def _run_endpoint(
    root,
    *,
    endpoint_id,
    task_id,
    phase_identity_sha256,
    pilot_spec_sha256,
    admission_ref,
    phase="PILOT_V1",
    execution_source,
    authorization_ref,
):
    """Run only one registered endpoint; no runtime or sampler override exists."""
    from mm_core.training import capture_rng, restore_rng

    root = evaluation._plain_path(root)
    evaluation._require(root.is_dir(), "Existing canonical experiment root required")
    admitted = _admitted(
        root,
        phase_identity_sha256,
        pilot_spec_sha256,
        admission_ref,
        phase=phase,
        execution_source=execution_source,
        authorization_ref=authorization_ref,
    )
    spec = admitted["pilot_spec"]
    plan = evaluation._plan(spec)
    endpoint = evaluation._endpoint(plan, endpoint_id)
    (public, tasks) = _inputs(root, plan, spec=spec, phase_context=admitted["phase_context"])
    with evaluation._source(root, spec, endpoint) as source:
        identity = evaluation._identity(
            root, admitted, endpoint, source, pilot_spec_sha256, admission_ref
        )
        with evaluation.EvaluationLedger(root, identity, plan, public, tasks) as ledger:
            groups = ledger.completed()
            if groups is not None:
                report = evaluation._endpoint_report(ledger, groups)
                evaluation._write(ledger.directory / "COMPLETE.json", report)
                return report
            evaluation._require(
                not (ledger.directory / "COMPLETE.json").exists(),
                "Endpoint completion marker contradicts incomplete groups",
            )
            rng = capture_rng()
            try:
                (runtime, optimizer, scheduler) = _load(
                    root,
                    endpoint["parent"],
                    task_id,
                    mode="EVALUATION",
                    phase=evaluation.pilot_contract.for_phase(phase).phase,
                    phase_identity_sha256=phase_identity_sha256,
                    audit_root=ledger.directory,
                    run_id=identity["evaluation_run_id"],
                    **{"pilot_spec_sha256": pilot_spec_sha256} if phase == "PILOT_V3" else {},
                    execution_source=execution_source,
                    authorization_ref=authorization_ref,
                )
                evaluation._require(
                    type(runtime) is native.Runtime
                    and optimizer is None
                    and (scheduler is None)
                    and (runtime.execution_mode() == "EVALUATION"),
                    "Production evaluation requires actual read-only native runtime",
                )
                evaluation._require(
                    runtime.torch.cuda.is_available() and runtime.torch.cuda.device_count() == 1,
                    "CPU cannot produce actual evaluation evidence",
                )
                if source["tx"] is not None:
                    _load_committed_endpoint(
                        runtime,
                        source["tx"],
                        expected_commit_sha256=source["commit"],
                        execution_source=execution_source,
                        authorization_ref=authorization_ref,
                    )
                evaluation._require(
                    runtime.policy_tensor_identity() == source["policy"]
                    and (not runtime.model.training)
                    and all(
                        not p.requires_grad and p.grad is None for p in runtime.model.parameters()
                    ),
                    "Evaluation policy is not frozen read-only",
                )
                with _open_evaluation_access(
                    runtime,
                    experiment_root=root,
                    pilot_spec_sha256=pilot_spec_sha256,
                    admission_ref=admission_ref,
                    ledger=ledger,
                    execution_source=execution_source,
                    authorization_ref=authorization_ref,
                ):
                    report = evaluation._execute(ledger, runtime)
                evaluation._require(
                    runtime.policy_tensor_identity() == source["policy"]
                    and all(p.grad is None for p in runtime.model.parameters()),
                    "Evaluation modified policy/gradients",
                )
                evaluation._write(ledger.directory / "COMPLETE.json", report)
                return report
            finally:
                restore_rng(rng)


def _verify_evaluation(
    root,
    *,
    phase_identity_sha256,
    pilot_spec_sha256,
    admission_ref,
    summary_ref=None,
    phase="PILOT_V1",
    execution_source,
    authorization_ref,
):
    """Read and replay every complete endpoint; missing/UNKNOWN never become zeros."""
    root = evaluation._plain_path(root)
    admitted = _admitted(
        root,
        phase_identity_sha256,
        pilot_spec_sha256,
        admission_ref,
        phase=phase,
        execution_source=execution_source,
        authorization_ref=authorization_ref,
    )
    spec = admitted["pilot_spec"]
    plan = evaluation._plan(spec)
    (public, tasks) = _inputs(root, plan, spec=spec, phase_context=admitted["phase_context"])
    (reports, inputs) = ([], [])
    with evaluation.ExitStack() as locks:
        for endpoint in plan["endpoints"]:
            source = locks.enter_context(evaluation._source(root, spec, endpoint))
            identity = evaluation._identity(
                root, admitted, endpoint, source, pilot_spec_sha256, admission_ref
            )
            ledger = locks.enter_context(
                evaluation.EvaluationLedger(root, identity, plan, public, tasks)
            )
            groups = ledger.completed()
            evaluation._require(
                groups is not None, "All 12 endpoints must be complete before aggregation"
            )
            report = evaluation._endpoint_report(ledger, groups)
            evaluation._require(
                evaluation._read(ledger.directory / "COMPLETE.json") == report,
                "Endpoint completion report differs from original raw replay",
            )
            reports.append(report)
            inputs.append(evaluation._reference(root, ledger.directory / "COMPLETE.json"))
        result = evaluation.summarize_reports(reports)
        result.update(
            status="COMPLETE_DESCRIPTIVE_NOT_GATE_SIGNED",
            phase_identity_sha256=phase_identity_sha256,
            pilot_spec_sha256=pilot_spec_sha256,
            admission_ref=admission_ref,
            inputs=inputs,
            **dict(
                phase=phase,
                input_overlay=spec["input_overlay"],
                prompt_version=spec["prompt_version"],
            )
            if phase == "PILOT_V3"
            else {},
        )
        if summary_ref is not None:
            evaluation._require(
                summary_ref.get("path")
                == evaluation.pilot_contract.for_phase(phase).evaluation_directory + "/SUMMARY.json"
                and evaluation._bound(root, summary_ref) == result,
                "Saved evaluation summary differs from complete original evidence",
            )
        return result


def _pins(context):
    return dict(
        phase="PILOT_V3",
        phase_identity_sha256=context["phase_context"]["identity_sha256"],
        pilot_spec_sha256=context["pilot_spec_ref"]["sha256"],
        admission_ref=context["admission_ref"],
    )


def run_endpoint(root, *, endpoint_id, task_id, execution_source, authorization_ref):
    context = prepare_context(
        root, execution_source=execution_source, authorization_ref=authorization_ref
    )
    return _run_endpoint(
        root,
        endpoint_id=endpoint_id,
        task_id=task_id,
        execution_source=execution_source,
        authorization_ref=authorization_ref,
        **_pins(context),
    )


def aggregate(root, *, execution_source, authorization_ref):
    context = prepare_context(
        root, execution_source=execution_source, authorization_ref=authorization_ref
    )
    return _verify_evaluation(
        root,
        execution_source=execution_source,
        authorization_ref=authorization_ref,
        **_pins(context),
    )
