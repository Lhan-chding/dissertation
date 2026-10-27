"""Immutable 54-path FORMAL_V1 training registration, never an admission gate.

The executing deployment must be the registered FORMAL source. Historical PILOT
provenance is verified at its own explicit deployment; its frozen plan supplies
references to original S0 parent tensor evidence, never trained pilot endpoints.
No model is loaded and no optimizer, rollout, evaluation, or scientific PASS is
created here. Build returns unpublished bytes; freeze publishes all 109 files
under one exclusive lock with ordinary rename, supported by the project Ceph.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

from srf6_ref.core import seed_for
from srf6_ref.planning import ARMS

from . import phases
from .contract import DIAGNOSTIC_STEPS
from .planning import draw_seed
from .registration import DIAGNOSTIC_GROUPS, DIAGNOSTIC_ROWS, PARENTS, _payload, _reference
from .registration import DIRECTORY as PILOT_DIRECTORY
from .registration import SCHEMA as PILOT_SCHEMA
from .registration import SPEC_FILE as PILOT_SPEC_FILE
from .registration import _inputs as _source_inputs
from .rollouts import FORMAT as ROLLOUT_FORMAT
from .rollouts import _bound, _path, _registration_inputs
from .transactions import _fsync_dir, _plain_path, _read_bytes, _require, _sha

PHASE = "FORMAL_V1"
DIRECTORY = "PHASE_RUNS/FORMAL_V1/FORMAL_FROZEN"
SPEC_FILE = "FORMAL_SPEC_FROZEN.json"
SCHEMA = "srf6-formal-training-registration-v1"
REPEATS = (0, 1, 2)
PERMANENT_CHECKPOINTS = (0, 12, 36, 72)


def _actual_deployment():
    return Path(__file__).resolve().parents[2]


def _context(root, supplied):
    _require(
        isinstance(supplied, dict)
        and supplied.get("phase") == PHASE
        and _sha(supplied.get("identity_sha256")),
        "Explicit pinned FORMAL_V1 phase context required",
    )
    deployment = _actual_deployment()
    _require(
        supplied.get("deployment") == str(deployment),
        "Formal registration must execute its declared deployment",
    )
    actual = phases.verify_phase(
        root, PHASE, deployment=deployment, expected_identity_sha256=supplied["identity_sha256"]
    )
    _require(actual == supplied, "Formal phase context differs from actual provenance")
    _require(
        actual["status"] == "BOUND_NOT_AUTHORIZED" and actual["training_allowed"] is False,
        "Source provenance cannot authorize formal training",
    )
    identity = actual["identity"]
    _require(
        identity["source_manifest"]["path"] == f"PHASES/{PHASE}/SOURCE_MANIFEST.json"
        and identity["source_manifest"]["sha256"]
        != identity["origin"]["source_manifest"]["sha256"],
        "Formal source must remain distinct from original S0",
    )
    return actual


def _pilot(root, context, reference):
    """Read immutable upstream provenance at its producer, without authorizing it."""
    _require(
        isinstance(reference, dict)
        and reference.get("path") == f"{PILOT_DIRECTORY}/{PILOT_SPEC_FILE}",
        "Explicit original frozen PILOT specification reference required",
    )
    reader = SimpleNamespace()
    spec = _bound(reader, root, reference)
    upstream = [
        item for item in context["identity"]["upstream"] if item["producer_phase"] == "PILOT_V1"
    ]
    _require(len(upstream) == 1, "Formal source must bind the original PILOT producer")
    identity_ref = dict(path="PHASES/PILOT_V1/IDENTITY.json", sha256=upstream[0]["identity_sha256"])
    identity = _bound(reader, root, identity_ref)
    pilot = phases.verify_phase(
        root,
        "PILOT_V1",
        deployment=identity["deployment"],
        expected_identity_sha256=identity_ref["sha256"],
    )
    # The already-verified upstream receipt tree must actually cite this plan;
    # caller-supplied valid-looking bytes are not a substitute for that binding.
    pending, visited, anchored = list(upstream[0]["evidence"]), set(), False
    while pending and not anchored:
        item = pending.pop(0)
        if item == reference:
            anchored = True
            break
        if item["path"] in visited or not item["path"].endswith(".json"):
            continue
        visited.add(item["path"])
        report = _bound(reader, root, item)
        if isinstance(report, dict):
            dependencies = [*report.get("inputs", []), *report.get("evidence", [])]
            anchored = any(
                child.get("path") == reference["path"]
                and child.get("sha256") == reference["sha256"]
                for child in dependencies
            )
            pending.extend(dependencies)
    _require(anchored, "Frozen PILOT spec must be bound by its FORMAL upstream producer evidence")
    _require(
        spec.get("schema") == PILOT_SCHEMA
        and spec.get("phase") == "PILOT_V1"
        and spec.get("phase_identity") == identity_ref
        and spec.get("actual_source_manifest") == pilot["identity"]["source_manifest"],
        "PILOT plan must retain its actual historical producer",
    )
    formal = context["identity"]
    for key, expected in (
        ("config_sha256", formal["config_sha256"]),
        ("data_manifest", formal["data_manifest"]),
        ("origin_identity", formal["origin"]["identity"]),
    ):
        _require(spec.get(key) == expected, "PILOT and FORMAL original config/DATA/S0 differ")
    _require(
        pilot["identity"]["origin"] == formal["origin"]
        and pilot["deployment"] != context["deployment"]
        and pilot["identity"]["source_manifest"]["sha256"] != formal["source_manifest"]["sha256"],
        "FORMAL cannot relabel the historical PILOT deployment/source",
    )
    return spec, dict(
        phase="PILOT_V1",
        deployment=pilot["deployment"],
        phase_identity=identity_ref,
        source_manifest=pilot["identity"]["source_manifest"],
        specification=copy.deepcopy(reference),
        use="Original S0 parent tensor references only; not admission or trained endpoints",
    )


def build_formal_registration(root, *, phase_context, pilot_spec_ref):
    """Build unpublished deterministic registration; requires actual FORMAL binding.

    The status describes frozen plan content, with published=False. Neither a
    real binding nor a CPU fixture becomes SERVER_ACTUAL execution evidence.
    No file/image/model inventory is rescanned beyond existing metadata caches.
    """
    root = _plain_path(root)
    context = _context(root, phase_context)
    pilot, producer = _pilot(root, context, pilot_spec_ref)
    # Reuse original S0 tensor/file/source checks, not the PILOT current-source
    # verifier. The helper's nine-path selection is discarded; full frozen order
    # is read below. This does not import any PILOT endpoint or optimizer state.
    reader, config, data, data_ref, order_ref, _, parents, binding_ref = _source_inputs(
        root, context, pilot.get("parent_references")
    )
    _require(pilot.get("parent_tensors") == parents, "PILOT original parent tensor binding differs")
    _require(pilot.get("source_binding") == binding_ref, "Original SOURCE_BINDING differs")
    order = _bound(reader, root, order_ref)
    combinations = {(p, r, a) for p in PARENTS for r in REPEATS for a in ARMS}
    _require(
        len(order) == 54
        and {(item["parent"], item["repeat"], item["arm"]) for item in order} == combinations,
        "All 54 frozen parent/repeat/arm combinations required",
    )
    identity = context["identity"]
    shared_inputs = {
        "public": data_ref("DATA/TRAIN/MODEL_INPUTS.jsonl"),
        "private": data_ref("DATA/TRAIN/TASKS_PRIVATE.jsonl"),
    }
    files, paths = {}, []
    for index, ordered in enumerate(order):
        parent, repeat, arm = ordered["parent"], ordered["repeat"], ordered["arm"]
        directory = f"runs/{parent}/r{repeat}/{arm}"
        core = dict(
            experiment_id=config["experiment_id"],
            run_id=ordered["run_id"],
            phase=PHASE,
            parent=parent,
            arm=arm,
            repeat=repeat,
            lane="PRIMARY",
            max_steps=72,
            source_sha256=identity["source_manifest"]["sha256"],
            data_sha256=identity["data_manifest"]["sha256"],
            config_sha256=identity["config_sha256"],
            reference_sha256=parents[parent]["tensor_sha256"],
            parent_content_sha256=parents[parent]["parent_content_sha256"],
            parent_reference_evidence=parents[parent]["runtime_identity"],
            phase_identity_sha256=context["identity_sha256"],
            origin_identity_sha256=identity["origin"]["identity"]["sha256"],
            diagnostic_row_indices=list(DIAGNOSTIC_ROWS),
        )
        inputs = {
            **shared_inputs,
            "schedule": data_ref(f"DATA/SCHEDULES/TRAIN-{parent}-r{repeat}.json"),
        }
        registration = dict(
            format=ROLLOUT_FORMAT,
            identity=core,
            split="TRAIN",
            factor=ordered["factor"],
            master_seed=config["master_seed"],
            schedule=copy.deepcopy(_bound(reader, root, inputs["schedule"])),
            inputs=inputs,
            shuffle_domain="formal_shuffle",
            draw_seeds=[
                [
                    [
                        draw_seed(config["master_seed"], parent, repeat, step, g // 4, g % 4, d)
                        for d in range(8)
                    ]
                    for g in range(16)
                ]
                for step in range(1, 73)
            ],
            shuffle_seeds=[
                [
                    seed_for(config["master_seed"], "formal_shuffle", parent, repeat, arm, step, g)
                    for g in range(16)
                ]
                for step in range(1, 73)
            ],
        )
        reader.identity = core
        _registration_inputs(reader, root, registration, config, data)
        run_identity = {
            **core,
            "rollout_registration": _reference("ROLLOUT_REGISTRATION.json", registration),
        }
        files[f"{directory}/ROLLOUT_REGISTRATION.json"] = registration
        files[f"{directory}/RUN_IDENTITY.json"] = run_identity
        paths.append(
            dict(
                order=index + 1,
                parent=parent,
                repeat=repeat,
                arm=arm,
                factor=ordered["factor"],
                credit=ordered["credit"],
                block_order=ordered["block_order"],
                arm_order_seed=ordered["arm_order_seed"],
                run_id=ordered["run_id"],
                run_directory=directory,
                audit_root=f"{DIRECTORY}/{directory}",
                updates=72,
                responses=9216,
                transaction_identity=_reference(f"{directory}/RUN_IDENTITY.json", run_identity),
                rollout_registration=_reference(
                    f"{directory}/ROLLOUT_REGISTRATION.json", registration
                ),
            )
        )
    spec = dict(
        schema=SCHEMA,
        phase=PHASE,
        status="FROZEN_NOT_AUTHORIZED",
        scope="TRAINING_REGISTRATION_ONLY_NO_EXECUTION_EVIDENCE",
        phase_identity=dict(
            path=f"PHASES/{PHASE}/IDENTITY.json", sha256=context["identity_sha256"]
        ),
        actual_source_manifest=identity["source_manifest"],
        origin_identity=identity["origin"]["identity"],
        pilot_producer=producer,
        config_sha256=identity["config_sha256"],
        data_manifest=identity["data_manifest"],
        source_binding=binding_ref,
        parent_references=copy.deepcopy(pilot["parent_references"]),
        parent_tensors=parents,
        order_source=order_ref,
        paths=paths,
        budgets=dict(
            primary_paths=54,
            updates_per_path=72,
            primary_logical_updates=3888,
            responses_per_update=128,
            primary_responses=497664,
            roots_per_path=288,
            exposures_per_root=1,
            groups_per_update=16,
            draws_per_group=8,
        ),
        diagnostics=dict(
            steps=list(DIAGNOSTIC_STEPS),
            group_indices=list(DIAGNOSTIC_GROUPS),
            row_indices=list(DIAGNOSTIC_ROWS),
            selection="Four slots; each slot's v0/v1; all eight draws; independent of outputs",
        ),
        initialization=dict(
            checkpoint="Corresponding original S0 frozen parent at step0",
            fresh_optimizer=True,
            optimizer_state_inheritance=False,
            pilot_endpoint_inheritance=False,
        ),
        checkpoints=dict(
            permanent_steps=list(PERMANENT_CHECKPOINTS),
            retain_all_checkpoint_states=True,
            primary_endpoint=72,
            raw_credit_update_retention="Permanent for every registered step, including failures",
            cleanup_implemented=False,
        ),
        scheduling=dict(
            within_block="Frozen FORMAL_PATH_ORDER six-arm order within each parent/repeat",
            between_blocks="Independent parent/repeat blocks may run concurrently",
            root_order="Frozen TRAIN schedule; identical for all six arms in each parent/repeat",
            draw_stream="Original draw_seed(master,parent,repeat,step,slot,variant,draw); no arm",
            shuffle_stream="seed_for(master,formal_shuffle,parent,repeat,arm,step,group_index)",
            additional_repeats_or_resampling=False,
        ),
        outstanding_requirements=dict(
            scope="TRAINING_REGISTRATION_ONLY",
            scientific_admission_included=False,
            runtime_access_included=False,
            evaluation_plan_included=False,
            separate_bound_plans_and_admission_required=True,
            step36_dev_eval=dict(
                authorization=False,
                checkpoints=54,
                questions=96,
                draws=1,
                total_responses=5184,
                checkpoint_selection_or_early_stopping=False,
            ),
            final_evaluation=dict(
                authorization=False,
                checkpoint_count=57,
                responses_per_checkpoint=6010,
                total_responses=342570,
                all_54_step72_commits_required_before_test_model_access=True,
            ),
        ),
        gate_status="NOT_SIGNED",
        training_authorized=False,
        generation_authorized=False,
        evaluation_authorized=False,
        formal_test_access=False,
        execution_status="NOT_RUN",
    )
    files[SPEC_FILE] = spec
    return dict(
        status="FROZEN_NOT_AUTHORIZED",
        published=False,
        directory=str(root / DIRECTORY),
        spec=spec,
        files=files,
        spec_sha256=hashlib.sha256(_payload(spec)).hexdigest(),
        training_authorized=False,
        generation_authorized=False,
        evaluation_authorized=False,
    )


def _check_files(destination, bundle):
    for relative, value in bundle["files"].items():
        _require(
            _read_bytes(_path(destination, relative)) == _payload(value),
            "Frozen formal registration artifact differs",
        )
    runs = _plain_path(destination / "runs")
    _require({p.name for p in runs.iterdir()} == set(PARENTS), "Unexpected formal parent directory")
    actual = set()
    for parent in PARENTS:
        directory = _plain_path(runs / parent)
        _require(
            {p.name for p in directory.iterdir()} == {f"r{r}" for r in REPEATS},
            "Unexpected formal repeat directory",
        )
        for repeat in REPEATS:
            for path in _plain_path(directory / f"r{repeat}").iterdir():
                _require(_plain_path(path).is_dir(), "Unexpected formal run entry")
                actual.add(path.relative_to(destination).as_posix())
    _require(
        actual == {p["run_directory"] for p in bundle["spec"]["paths"]},
        "Unexpected/missing formal run directory",
    )


def verify_formal_registration(root, *, phase_context, expected_spec_sha256):
    """Verify exact published construction and all original bound training inputs."""
    root = _plain_path(root)
    _require(_sha(expected_spec_sha256), "Pinned formal specification hash required")
    destination = _plain_path(root / DIRECTORY)
    content = _read_bytes(destination / SPEC_FILE)
    _require(hashlib.sha256(content).hexdigest() == expected_spec_sha256, "Formal spec pin differs")
    spec = json.loads(content)
    bundle = build_formal_registration(
        root,
        phase_context=phase_context,
        pilot_spec_ref=spec.get("pilot_producer", {}).get("specification"),
    )
    _require(bundle["spec_sha256"] == expected_spec_sha256, "Formal construction differs")
    _check_files(destination, bundle)
    return _result(bundle, "VERIFIED_REGISTRATION_NOT_AUTHORIZED")


def _result(bundle, status):
    return dict(
        status=status,
        published=True,
        directory=bundle["directory"],
        spec_sha256=bundle["spec_sha256"],
        spec=bundle["spec"],
        gate_status="NOT_SIGNED",
        training_authorized=False,
        generation_authorized=False,
        evaluation_authorized=False,
    )


def freeze_formal_registration(root, *, phase_context, pilot_spec_ref):
    """Publish 109 files atomically; exact retry is read-only, failure stays intact."""
    root = _plain_path(root)
    bundle = build_formal_registration(
        root, phase_context=phase_context, pilot_spec_ref=pilot_spec_ref
    )
    destination = _plain_path(root / DIRECTORY)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        _plain_path(destination.parent / ".formal-registration.lock"),
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
        0o600,
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if destination.exists():
            _check_files(destination, bundle)
        else:
            staging = destination.with_name(f".formal-registration-{uuid.uuid4().hex}")
            staging.mkdir()
            for relative, value in bundle["files"].items():
                path = _path(staging, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(_payload(value))
                    stream.flush()
                    os.fsync(stream.fileno())
            for directory, _, _ in os.walk(staging, topdown=False):
                _fsync_dir(directory)
            _require(
                not destination.exists() and not destination.is_symlink(),
                "Never replace an existing formal registration",
            )
            os.rename(staging, destination)
            _fsync_dir(destination.parent)
        _check_files(destination, bundle)
    finally:
        os.close(fd)
    return _result(bundle, "FROZEN_NOT_AUTHORIZED")
