"""Freeze the nine PILOT_V1 paths without authorizing generation or training.

build_pilot_registration constructs deterministic artifacts; freeze publishes
them atomically at PHASE_RUNS/PILOT_V1/PILOT_FROZEN; verify reconstructs and checks
the published plan. All three reverify actual phase provenance. Parent tensor
references come from immutable, explicitly supplied S0 runtime_identity files
using the 017b60a schema, which has no phase/execution_mode fields.

No reference cycle: rollout registration binds identity_core without its own
reference, then the final transaction identity binds the serialized registration;
the overall pilot spec references both. The run identity never references the
overall spec. These engineering conventions must be frozen before pilot output.
Actual b0/U11 step4 recovery import and scientific prerequisite gates are separate
work: this module records their requirements and never implements a bypass.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import uuid
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from srf6_ref.core import FAMILIES, seed_for

from . import phases
from .common import digest
from .data import public_prompt
from .planning import draw_seed, formal_paths, root_schedule
from .rollouts import FORMAT as ROLLOUT_FORMAT
from .rollouts import _bound, _path, _registration_inputs
from .transactions import _canonical, _fsync_dir, _plain_path, _read_bytes, _require, _sha

PHASE = "PILOT_V1"
DIRECTORY = "PHASE_RUNS/PILOT_V1/PILOT_FROZEN"
SPEC_FILE = "PILOT_SPEC_FROZEN.json"
SCHEMA = "srf6-pilot-registration-v1"
PARENTS = ("b0", "b1", "b2")
ARMS = ("U10", "U11", "C_SHUF")
DIAGNOSTIC_GROUPS = (0, 1, 4, 5, 8, 9, 12, 13)
DIAGNOSTIC_ROWS = tuple(g * 8 + d for g in DIAGNOSTIC_GROUPS for d in range(8))


def _actual_deployment():
    return Path(__file__).resolve().parents[2]


def _payload(value):
    return _canonical(value) + b"\n"


def _reference(path, value):
    return {"path": path, "sha256": hashlib.sha256(_payload(value)).hexdigest()}


def _context(root, supplied):
    _require(
        isinstance(supplied, dict) and supplied.get("phase") == PHASE,
        "Verified PILOT_V1 context required",
    )
    _require(_sha(supplied.get("identity_sha256")), "Explicit phase identity pin required")
    deployment = _actual_deployment()
    _require(
        supplied.get("deployment") == str(deployment),
        "Registration is not executing its declared phase source",
    )
    actual = phases.verify_phase(
        root, PHASE, deployment=deployment, expected_identity_sha256=supplied["identity_sha256"]
    )
    _require(actual == supplied, "Caller phase context differs from fresh provenance verification")
    _require(
        actual.get("status") == "BOUND_NOT_AUTHORIZED" and actual.get("training_allowed") is False,
        "Phase provenance must not be interpreted as scientific authorization",
    )
    identity = actual["identity"]
    _require(
        identity.get("source_manifest", {}).get("path") == f"PHASES/{PHASE}/SOURCE_MANIFEST.json",
        "Actual PILOT_V1 source manifest required",
    )
    _require(
        identity["source_manifest"]["sha256"] != identity["origin"]["source_manifest"]["sha256"],
        "New registration cannot claim historical S0 source",
    )
    return actual


def _inputs(root, context, parent_references):
    identity = context["identity"]
    reader = SimpleNamespace()
    config = _bound(reader, root, {"path": "CONFIG.json", "sha256": identity["config_sha256"]})
    data = _bound(reader, root, identity["data_manifest"])
    _require(data.get("config_sha256") == digest(config), "Frozen DATA/config digest differs")
    files = data.get("files")
    _require(
        isinstance(files, list) and all(isinstance(item, dict) for item in files),
        "Frozen DATA inventory required",
    )
    inventory = {item["path"]: item["sha256"] for item in files}
    _require(len(inventory) == len(files), "Duplicate frozen DATA path")

    def data_ref(relative):
        _require(_sha(inventory.get(relative)), "Required pilot input missing from frozen DATA")
        return {"path": relative, "sha256": inventory[relative]}

    order_ref = data_ref("DATA/FORMAL_PATH_ORDER.json")
    order = _bound(reader, root, order_ref)
    _require(
        order == formal_paths(config), "Frozen formal path order differs from registered generator"
    )
    selected = [item for item in order if item["repeat"] == 0 and item["arm"] in ARMS]
    _require(
        len(selected) == 9
        and {(item["parent"], item["arm"]) for item in selected}
        == {(p, a) for p in PARENTS for a in ARMS},
        "Exactly nine fixed parent/arm pilot paths required",
    )
    _require(
        isinstance(parent_references, dict) and set(parent_references) == set(PARENTS),
        "Three explicit S0 parent runtime evidence references required",
    )
    _require(
        len({ref.get("path") for ref in parent_references.values() if isinstance(ref, dict)}) == 3,
        "Parent evidence references must be distinct",
    )
    source_binding_path = root / "SOURCE_BINDING.json"
    binding_bytes = _read_bytes(source_binding_path)
    binding = json.loads(binding_bytes)
    _require(
        binding.get("config_file_sha256") == identity["config_sha256"],
        "Original parent file binding/config differs",
    )
    binding_ref = dict(path="SOURCE_BINDING.json", sha256=hashlib.sha256(binding_bytes).hexdigest())
    parents = {}
    for parent in PARENTS:
        reference = parent_references[parent]
        report = _bound(reader, root, reference)
        relative = Path(reference["path"])
        _require(
            relative.parent == Path("runtime_identity")
            and relative.suffix == ".json"
            and report.get("run_id") == relative.stem,
            "Original S0 runtime_identity path/run identifier required",
        )
        for key, expected in (
            ("parent_id", parent),
            ("reference_parent", parent),
            ("run_root", str(root)),
            ("source_sha256", identity["origin"]["source_manifest"]["sha256"]),
            ("source_hash", identity["origin"]["source_manifest"]["sha256"]),
            ("data_hash", identity["data_manifest"]["sha256"]),
            ("snapshot", config["model"]["snapshot"]),
            ("dtype", "float32"),
            ("input_float_dtype", "float32"),
            ("probability_dtype", "float32"),
        ):
            _require(report.get(key) == expected, f"S0 parent runtime {key} differs")
        _require(
            type(report.get("step")) is int
            and report["step"] == 0
            and report.get("fresh_optimizer") is False,
            "Frozen read-only S0 parent evidence at step zero required",
        )
        tensor_hash = report.get("reference_hash")
        _require(
            _sha(tensor_hash) and report.get("policy_hash") == tensor_hash,
            "Actual parent policy/reference tensor identity required",
        )
        parent_binding = binding["parents"][parent]
        _require(
            parent_binding.get("path") == config["model"]["parents"][parent]
            and _sha(parent_binding.get("content_sha256"))
            and report.get("parent_content_hash") == parent_binding["content_sha256"],
            "Parent file inventory evidence differs",
        )
        _require(
            tensor_hash != parent_binding["content_sha256"],
            "Parent file inventory cannot substitute for tensor identity",
        )
        parameters = report.get("trainable_parameters")
        _require(
            isinstance(parameters, list)
            and bool(parameters)
            and all(
                isinstance(p, dict)
                and isinstance(p.get("name"), str)
                and bool(p["name"])
                and p.get("dtype") == "torch.float32"
                and isinstance(p.get("shape"), list)
                and all(type(n) is int and n >= 0 for n in p["shape"])
                for p in parameters
            ),
            "Original loaded adapter parameter inventory required",
        )
        _require(
            len({p["name"] for p in parameters}) == len(parameters),
            "Duplicate loaded adapter parameter name",
        )
        _require(
            isinstance(report.get("bound_files_before_load"), dict)
            and isinstance(report.get("bound_files_after_load"), dict),
            "Original before/after load file evidence required",
        )
        parents[parent] = dict(
            runtime_identity=copy.deepcopy(reference),
            tensor_sha256=tensor_hash,
            parent_content_sha256=parent_binding["content_sha256"],
            parameter_inventory_sha256=digest(parameters),
            evidence_producer="CENSUS_S0",
        )
    return reader, config, data, data_ref, order_ref, selected, parents, binding_ref


def _evaluation(reader, root, config, data_ref, paths, parents):
    inputs = {
        "public": data_ref("DATA/DEV_EVAL/MODEL_INPUTS.jsonl"),
        "private": data_ref("DATA/DEV_EVAL/TASKS_PRIVATE.jsonl"),
    }
    public = _bound(reader, root, inputs["public"], kind="jsonl")
    private = _bound(reader, root, inputs["private"], kind="jsonl")
    expected = {
        f"DEV_EVAL_{family}_{index:04d}_P0_v{view}"
        for family in FAMILIES
        for index in range(4)
        for view in range(4)
    }
    _require(
        len(public) == len(private) == 96
        and {r["qid"] for r in public} == {r["qid"] for r in private} == expected,
        "Exactly 96 frozen DEV_EVAL questions required",
    )
    tasks = {r["qid"]: r for r in private}
    for item in public:
        task = tasks[item["qid"]]
        _require(
            all(
                item.get(k) == task.get(k)
                for k in ("qid", "root_id", "family", "variant", "split", "factor")
            )
            and item.get("split") == "DEV_EVAL"
            and item.get("factor") == "P0"
            and item.get("protocol") == "structured_v4",
            "DEV_EVAL identity/split/protocol differs",
        )
        _require(
            isinstance(item.get("model_input"), dict)
            and set(item["model_input"]) == {"prompt", "image_file", "image_sha256"},
            "Evaluation model input must use public whitelist",
        )
        _require(
            type(task.get("variant")) is int
            and task["variant"] in range(4)
            and task["family"] in FAMILIES
            and item["qid"] == f"{task['root_id']}_P0_v{task['variant']}"
            and task["root_id"].startswith(f"DEV_EVAL_{task['family']}_"),
            "DEV_EVAL root/view lineage differs",
        )
        safe = item["model_input"]
        _require(safe["prompt"] == public_prompt(task), "DEV_EVAL public prompt differs")
        image = safe["image_file"]
        _require(
            isinstance(image, str)
            and image.startswith("DATA/images/")
            and PurePosixPath(image).as_posix() == image
            and ".." not in PurePosixPath(image).parts
            and _sha(safe["image_sha256"])
            and data_ref(image)["sha256"] == safe["image_sha256"],
            "DEV_EVAL image differs from frozen inventory",
        )
    endpoints = [
        dict(
            kind="FROZEN_PARENT",
            endpoint_id=f"{parent}/parent",
            parent=parent,
            checkpoint_step=0,
            reference=parents[parent]["runtime_identity"],
            tensor_sha256=parents[parent]["tensor_sha256"],
            responses=192,
        )
        for parent in PARENTS
    ]
    endpoints.extend(
        dict(
            kind="PILOT_ENDPOINT",
            endpoint_id=f"{item['parent']}/{item['arm']}/step12",
            parent=item["parent"],
            run_id=item["run_id"],
            checkpoint_step=12,
            expected_commit_path=f"{DIRECTORY}/{item['run_directory']}/checkpoints/step-012/COMMIT.json",
            actual_commit_binding="REQUIRED_AFTER_PUBLICATION_BEFORE_EVALUATION",
            responses=192,
        )
        for item in paths
    )
    return dict(
        split="DEV_EVAL",
        inputs=inputs,
        questions=96,
        draws_per_question=2,
        checkpoint_count=12,
        responses=2304,
        endpoints=endpoints,
        seed_domain="pilot_dev_eval",
        descriptive_summary=dict(
            status="DESCRIPTIVE_ONLY_NOT_GATE",
            master_seed=2026102701,
            bootstrap_replicates=2000,
            confidence_level=0.95,
            quantiles=[0.025, 0.975],
            quantile_interpolation="LINEAR_INDEX_N_MINUS_ONE_P",
            resampling="SIX_FAMILIES_EACH_FOUR_ROOTS_WITH_REPLACEMENT",
            paired_cluster="ROOT_FOUR_VIEWS_TWO_DRAWS",
            absolute_seed_parts=[
                "master_seed",
                "pilot_dev_eval_summary",
                "endpoint_id",
                "label",
                "statistic",
            ],
            absolute_labels=["A", "J", "V", "S", "B"],
            absolute_statistics=["known_rate", "pass_rate_given_known"],
            difference_seed_parts=[
                "master_seed",
                "pilot_dev_eval_summary",
                "endpoint_id",
                "parent_difference",
                "metric",
            ],
            difference_metrics=["A", "J", "V", "S", "B", "S_known", "B_known"],
            semantic_difference_denominator="BOTH_ENDPOINT_AND_PARENT_KNOWN",
            zero_denominator_bootstrap="OMIT_AND_REPORT_DEFINED_REPLICATES_ALL_UNDEFINED_IS_NULL",
            training_repeat_variance_estimated=False,
            significance_gate=None,
        ),
        seeds_by_parent={
            parent: [
                {
                    "qid": qid,
                    "seeds": [
                        seed_for(config["master_seed"], "pilot_dev_eval", parent, qid, draw)
                        for draw in range(2)
                    ],
                }
                for qid in sorted(expected)
            ]
            for parent in PARENTS
        },
        recovery_endpoint_is_additional_checkpoint=False,
        formal_test_access=False,
        execution_status="NOT_RUN",
    )


def build_pilot_registration(root, *, phase_context, parent_references):
    """Construct deterministic pilot-only artifacts; reverify actual phase source.

    Returns the file mapping without publishing it. This verifies existing
    provenance and references; it cannot establish scientific gate completion.
    """
    root = _plain_path(root)
    context = _context(root, phase_context)
    reader, config, data, data_ref, order_ref, order, parents, binding_ref = _inputs(
        root, context, parent_references
    )
    phase_identity = context["identity"]
    source_hash = phase_identity["source_manifest"]["sha256"]
    files, paths = {}, []
    shared_inputs = {
        "public": data_ref("DATA/DEV_PILOT/MODEL_INPUTS.jsonl"),
        "private": data_ref("DATA/DEV_PILOT/TASKS_PRIVATE.jsonl"),
    }
    for index, ordered in enumerate(order):
        parent, arm = ordered["parent"], ordered["arm"]
        run_id = f"{config['experiment_id']}/PILOT/{parent}/r0/{arm}"
        run_directory = f"runs/{parent}/r0/{arm}"
        identity_core = dict(
            experiment_id=config["experiment_id"],
            run_id=run_id,
            phase=PHASE,
            parent=parent,
            arm=arm,
            repeat=0,
            lane="PRIMARY",
            max_steps=12,
            source_sha256=source_hash,
            data_sha256=phase_identity["data_manifest"]["sha256"],
            config_sha256=phase_identity["config_sha256"],
            reference_sha256=parents[parent]["tensor_sha256"],
            parent_content_sha256=parents[parent]["parent_content_sha256"],
            parent_reference_evidence=parents[parent]["runtime_identity"],
            phase_identity_sha256=context["identity_sha256"],
            origin_identity_sha256=phase_identity["origin"]["identity"]["sha256"],
            diagnostic_row_indices=list(DIAGNOSTIC_ROWS),
        )
        inputs = {
            **shared_inputs,
            "schedule": data_ref(f"DATA/SCHEDULES/DEV_PILOT-{parent}-r0.json"),
        }
        registration = dict(
            format=ROLLOUT_FORMAT,
            identity=identity_core,
            split="DEV_PILOT",
            factor="P1",
            master_seed=config["master_seed"],
            schedule=root_schedule(8, config["master_seed"], parent, 0, split="DEV_PILOT"),
            inputs=inputs,
            shuffle_domain="pilot_shuffle",
            draw_seeds=[
                [
                    [
                        draw_seed(
                            config["master_seed"], parent, 0, step, group // 4, group % 4, draw
                        )
                        for draw in range(8)
                    ]
                    for group in range(16)
                ]
                for step in range(1, 13)
            ],
            shuffle_seeds=[
                [
                    seed_for(config["master_seed"], "pilot_shuffle", parent, 0, arm, step, group)
                    for group in range(16)
                ]
                for step in range(1, 13)
            ],
        )
        reader.identity = identity_core
        _registration_inputs(reader, root, registration, config, data)
        registration_ref = _reference("ROLLOUT_REGISTRATION.json", registration)
        identity = {**identity_core, "rollout_registration": registration_ref}
        files[f"{run_directory}/ROLLOUT_REGISTRATION.json"] = registration
        files[f"{run_directory}/RUN_IDENTITY.json"] = identity
        paths.append(
            dict(
                order=index + 1,
                parent=parent,
                arm=arm,
                repeat=0,
                factor="P1",
                updates=12,
                responses=1536,
                run_id=run_id,
                run_directory=run_directory,
                audit_root=f"{DIRECTORY}/{run_directory}",
                transaction_identity=_reference(f"{run_directory}/RUN_IDENTITY.json", identity),
                rollout_registration=_reference(
                    f"{run_directory}/ROLLOUT_REGISTRATION.json", registration
                ),
            )
        )
    b0_u11 = next(item for item in paths if item["parent"] == "b0" and item["arm"] == "U11")
    spec = dict(
        schema=SCHEMA,
        phase=PHASE,
        status="PILOT_PLAN_NOT_AUTHORIZED",
        phase_identity={
            "path": f"PHASES/{PHASE}/IDENTITY.json",
            "sha256": context["identity_sha256"],
        },
        actual_source_manifest=phase_identity["source_manifest"],
        origin_identity=phase_identity["origin"]["identity"],
        config_sha256=phase_identity["config_sha256"],
        data_manifest=phase_identity["data_manifest"],
        source_binding=binding_ref,
        parent_references=copy.deepcopy(parent_references),
        parent_tensors=parents,
        order_source=order_ref,
        paths=paths,
        budgets=dict(
            primary_paths=9,
            updates_per_path=12,
            primary_logical_updates=108,
            responses_per_update=128,
            primary_responses=13824,
            recovery_new_responses=1024,
            dev_eval_responses=2304,
        ),
        diagnostics=dict(
            group_indices=list(DIAGNOSTIC_GROUPS),
            row_indices=list(DIAGNOSTIC_ROWS),
            registered_all_steps=[1, 4, 8, 12, 36, 72],
            pilot_steps=[1, 4, 8, 12],
            selection="Four slots; each slot's v0/v1; all eight draws; independent of outputs",
        ),
        recovery=dict(
            parent="b0",
            arm="U11",
            repeat=0,
            logical_run_id=b0_u11["run_id"],
            storage_lane="RECOVERY",
            physical_directory="PHASE_RUNS/PILOT_V1/RECOVERY/b0/r0/U11",
            binding_path="PHASE_RUNS/PILOT_V1/RECOVERY/b0/r0/U11/RECOVERY_BINDING.json",
            logical_identity_policy="Byte-identical PRIMARY RUN_IDENTITY and rollout registration",
            execution_journal=(
                "Per-process BEGIN/COMMIT references/END; actual native call accounting"
            ),
            continuity_rule=(
                "Primary starts at step0; all twelve updates in one process; "
                "1536 new responses; no prior data attempt"
            ),
            independent_process_proof=(
                "SERVER_ACTUAL BEGIN/END bind hostname, Linux boot_id, pid and /proc start_ticks; "
                "primary and recovery process identities must differ"
            ),
            source_run_directory=b0_u11["run_directory"],
            source_checkpoint_step=4,
            target_checkpoint_step=12,
            update_steps=list(range(5, 13)),
            new_generated_responses=1024,
            expected_source_commit_path=f"{DIRECTORY}/{b0_u11['run_directory']}/checkpoints/step-004/COMMIT.json",
            actual_source_commit_binding="REQUIRED_SEPARATELY_AFTER_STEP4_PUBLICATION",
            source_prefix_policy=(
                "Verified state and consumption prefix; "
                "separate immutable recovery identity binding"
            ),
            reuse_primary_steps5_to12_raw=False,
            regenerate_original_registered_seeds=True,
            adds_scientific_repeat=False,
            comparison=dict(
                rule="EXACT_TREE_NO_POSTHOC_TOLERANCE",
                required_sections=[
                    "adapter_parameter_names_dtypes_shapes_tensors",
                    "adam_all_tensors_and_steps",
                    "scheduler",
                    "python_numpy_cpu_cuda_rng",
                    "sampler_consumption_cursor_and_scientific_step",
                    "original_tokens_and_sampler_logprobs",
                    "native_prompt_inputs_finish_and_full_text",
                    "labels_oracle_witnesses_credit_strata_mappings_coefficients",
                    "per_step_numerical_and_gradient_content",
                ],
                execution_metadata_only_exclusions=[
                    "time",
                    "host",
                    "pid",
                    "job_id",
                    "storage_lane",
                    "execution_id",
                ],
                exact_exclusion_locations={
                    "original_row_top_level": ["task_id", "generation_seconds"],
                    "original_row_model_identity": ["task_id", "audit_root"],
                    "update_top_level": ["training_seconds"],
                    "preupdate_evidence_top_level": [
                        "at",
                        "audit_root",
                        "task_id",
                        "source_audit_event",
                    ],
                },
                metadata_exclusion_scope=(
                    "Execution receipts only; never recursive raw-text removal"
                ),
                audit_slice_exclusion_requires=(
                    "Verify original bytes SHA and event content equality first"
                ),
                audit_prefix_import=(
                    "Only original bytes through step4 event end; later events are generated anew"
                ),
                compared_hardware_fields=[
                    "cuda_name",
                    "cuda_total_memory_bytes",
                    "compute_capability",
                    "cuda_runtime",
                    "driver_version",
                ],
                compared_environment_fields=[
                    "dtype",
                    "numerics",
                    "attention_backend",
                    "torch_version",
                    "transformers_version",
                    "forward_parameter_dtypes",
                    "probability_dtype",
                ],
                physical_gpu_uuid_policy=(
                    "Retain both actual UUIDs; equal UUID is not required "
                    "for the registered two-worker plan"
                ),
                continuity_required=True,
                scope="Only registered b0/U11 on the actually verified hardware/kernel/environment",
            ),
            implementation_status="IMPLEMENTED_NOT_EXECUTED",
        ),
        evaluation=_evaluation(reader, root, config, data_ref, paths, parents),
        engineering_conventions=dict(
            status="PROSPECTIVE_ENGINEERING_CONVENTIONS",
            freeze_before_first_pilot_output=True,
            path_order=(
                "Filter frozen FORMAL_PATH_ORDER to repeat0 and U10/U11/C_SHUF, preserving order"
            ),
            run_id_pattern="{experiment}/PILOT/{parent}/r0/{arm}",
            primary_lane="PRIMARY",
            registration_directory=DIRECTORY,
            diagnostic_groups=list(DIAGNOSTIC_GROUPS),
            shuffle_seed="seed_for(master,pilot_shuffle,parent,0,arm,step,group_index)",
            evaluation_seed="seed_for(master,pilot_dev_eval,parent,qid,draw)",
            lane_and_execution_id_excluded_from_scientific_seeds=True,
            scheduling="Registered order within a parent; no scheduler/job allocation occurs here",
            retain_all_checkpoint_states=True,
        ),
        gate_status="NOT_SIGNED",
        training_authorized=False,
        generation_authorized=False,
        execution_status="NOT_RUN",
    )
    files[SPEC_FILE] = spec
    return dict(
        status="READY_TO_FREEZE_NOT_AUTHORIZED",
        directory=str(root / DIRECTORY),
        spec=spec,
        files=files,
        spec_sha256=hashlib.sha256(_payload(spec)).hexdigest(),
        training_authorized=False,
        generation_authorized=False,
    )


def _check_files(destination, bundle):
    for relative, value in bundle["files"].items():
        _require(
            _read_bytes(_path(destination, relative)) == _payload(value),
            "Frozen registration artifact differs",
        )
    expected = {item["run_directory"] for item in bundle["spec"]["paths"]}
    runs = _plain_path(destination / "runs")
    _require({p.name for p in runs.iterdir()} == set(PARENTS), "Unexpected pilot parent directory")
    actual = set()
    for parent in PARENTS:
        parent_dir = _plain_path(runs / parent)
        _require(
            {p.name for p in parent_dir.iterdir()} == {"r0"}, "Unexpected pilot repeat directory"
        )
        for path in _plain_path(parent_dir / "r0").iterdir():
            _require(_plain_path(path).is_dir(), "Unexpected pilot run entry")
            actual.add(path.relative_to(destination).as_posix())
    _require(actual == expected, "Unexpected/missing pilot run directory")


def verify_pilot_registration(root, *, phase_context, expected_spec_sha256):
    """Verify published bytes and rebuild fixed science/engineering choices."""
    root = _plain_path(root)
    _require(_sha(expected_spec_sha256), "Pinned pilot specification hash required")
    destination = _plain_path(root / DIRECTORY)
    spec_bytes = _read_bytes(destination / SPEC_FILE)
    _require(
        hashlib.sha256(spec_bytes).hexdigest() == expected_spec_sha256,
        "Pilot specification pin differs",
    )
    spec = json.loads(spec_bytes)
    bundle = build_pilot_registration(
        root, phase_context=phase_context, parent_references=spec.get("parent_references")
    )
    _require(
        bundle["spec_sha256"] == expected_spec_sha256,
        "Published specification differs from registered construction",
    )
    _check_files(destination, bundle)
    return dict(
        status="VERIFIED_REGISTRATION_NOT_AUTHORIZED",
        directory=str(destination),
        spec_sha256=expected_spec_sha256,
        spec=bundle["spec"],
        gate_status="NOT_SIGNED",
        training_authorized=False,
        generation_authorized=False,
    )


def freeze_pilot_registration(root, *, phase_context, parent_references):
    """Publish all nineteen immutable registration files together, never replace."""
    root = _plain_path(root)
    bundle = build_pilot_registration(
        root, phase_context=phase_context, parent_references=parent_references
    )
    destination = _plain_path(root / DIRECTORY)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        _plain_path(destination.parent / ".pilot-registration.lock"),
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
        0o600,
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if destination.exists():
            _check_files(destination, bundle)
        else:
            staging = destination.with_name(f".pilot-registration-{uuid.uuid4().hex}")
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
                "Never replace an existing pilot registration",
            )
            os.rename(staging, destination)
            _fsync_dir(destination.parent)
        _check_files(destination, bundle)
    finally:
        os.close(fd)
    return dict(
        status="FROZEN_NOT_AUTHORIZED",
        directory=str(destination),
        spec_sha256=bundle["spec_sha256"],
        spec=bundle["spec"],
        gate_status="NOT_SIGNED",
        training_authorized=False,
        generation_authorized=False,
    )
