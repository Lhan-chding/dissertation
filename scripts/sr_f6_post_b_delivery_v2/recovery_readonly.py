"""Read-only recovery delivery consumer for already frozen PILOT_V3 evidence.

Recovery arithmetic, state/RNG/raw comparison and prefix checks are copied from
b7 without weakening assertions. Provenance uses the original phase verifier
with a cold, read-only hash provider. Historical confirmation is consumed from
its existing bound transport proof; it is NOT recomputed or newly admitted.
This module writes nothing, does not sample/train, and does not patch any frozen
module or insert prevalidated values into its caches. The caller must bind this
new consumer independently and hold both existing transaction locks.
"""

from __future__ import annotations

import copy
import hashlib
import re
import sys
from pathlib import Path

from sr_f6 import historical_closure as historical
from sr_f6 import phases, pilot_audit
from sr_f6 import pilot_v3_inputs as overlay
from sr_f6 import pilot_v3_registration as v3_registration
from sr_f6 import recovery as frozen
from sr_f6.common import digest
from sr_f6.runtime import _verify_module_origins
from sr_f6.transactions import _require


class ReadOnlyHashes:
    """Never trusts or writes a phase cache, even when legacy cache data exists."""

    def __init__(self, root):
        # Only read by phases._verify_manifest; adopt_verified ignores its data.
        self.directory = phases._path(root / ".verification-cache", exists=False)

    def adopt_verified(self, path, expected_sha256, previous):
        # Deliberately no adoption. hash() below always reads the actual bytes.
        return None

    def hash(self, path):
        path = phases._path(path)
        before = phases._stat(path)
        value = phases.file_hash(path)
        _require(phases._stat(path) == before, "File changed while hashing")
        return value


def _context_readonly(root, supplied):
    _require(
        isinstance(supplied, dict)
        and supplied.get("phase") == overlay.PHASE
        and frozen._sha(supplied.get("identity_sha256")),
        "Explicit pinned PILOT_V3 phase context required",
    )
    root = phases._root(root)
    deployment = overlay._actual_deployment()
    _require(supplied.get("deployment") == str(deployment), "Actual PILOT_V3 source required")
    context = phases._verify_phase(root, overlay.PHASE, deployment, ReadOnlyHashes(root), set())
    _require(context == supplied, "PILOT_V3 context differs from fresh provenance verification")
    _require(
        context.get("status") == "BOUND_NOT_AUTHORIZED"
        and context.get("training_allowed") is False
        and context["identity"]["source_manifest"]["path"]
        == f"PHASES/{overlay.PHASE}/SOURCE_MANIFEST.json"
        and context["identity"]["source_manifest"]["sha256"]
        != context["identity"]["origin"]["source_manifest"]["sha256"],
        "PILOT_V3 provenance is separate from historical source and authorization",
    )
    overlay._source_members(
        root, context, (overlay, overlay.interface_v3_protocol, overlay.registration)
    )
    manifest = historical._json(
        phases._file(root, context["identity"]["source_manifest"]["path"]).read_bytes()
    )
    _verify_module_origins(deployment, manifest, tuple(sys.modules.values()))
    return context


def _historical_input_hash(reference):
    _require(set(reference) == {"path", "sha256"}, "Exact saved historical input required")
    phases._sha(reference["sha256"])
    before = historical._state(reference["path"])
    target = phases._path(Path(before["target"]))
    _require(historical.file_hash(target) == reference["sha256"], "Saved historical input changed")
    _require(
        historical._state(reference["path"]) == before,
        "Historical alias/input changed while reading",
    )


def _proof_bound(root, reference, *, json_value=True):
    """Cold proof transport read; never consult or populate a legacy cache."""
    _require(
        isinstance(reference, dict)
        and set(reference) == {"path", "sha256"}
        and frozen._sha(reference["sha256"]),
        "Explicit proof reference required",
    )
    path = frozen._path(root, reference["path"])
    before = frozen._stamp(path)
    payload = frozen._read_bytes(path)
    _require(hashlib.sha256(payload).hexdigest() == reference["sha256"], "Proof content differs")
    _require(frozen._stamp(path) == before, "Proof changed while reading")
    return frozen._json(payload) if json_value else payload


def _confirmation_readonly(root, *, phase_context, confirmation_binding_ref):
    """Validate existing proof transport and all its inputs; never run BOOTSTRAP."""
    root = phases._root(root)
    context = _context_readonly(root, phase_context)
    _require(
        set(confirmation_binding_ref) == {"path", "sha256"}
        and confirmation_binding_ref["path"] == historical.BINDING_PATH,
        "Fixed V3 confirmation binding required",
    )
    binding = _proof_bound(root, confirmation_binding_ref)
    registration_ref = binding["closure_registration_ref"]
    receipt_ref = binding["corrected_confirmation_receipt_ref"]
    registered = historical._request(registration_ref, receipt_ref)
    original_ref = binding["original_verification_ref"]
    original_path = phases._file(root, original_ref["path"])
    _require(
        original_path.parent.parent == root / historical.DIRECTORY
        and original_path.name == "RESULT.json"
        and re.fullmatch(r"[0-9a-f]{32}", original_path.parent.name),
        "Original actual attempt changed",
    )
    original = _proof_bound(root, original_ref)
    _require(
        original["schema"] == historical.SCHEMA
        and original["status"] == "CONFIRMATION_RECOMPUTED_NOT_AUTHORIZED"
        and all(
            original[k] is False
            for k in ("training_authorized", "generation_authorized", "formal_authorized")
        ),
        "Existing original confirmation proof required",
    )
    request = dict(
        nonce=original_path.parent.name,
        followup_root=str(historical.FOLLOWUP_ROOT),
        followup_deployment=str(historical.FOLLOWUP_DEPLOYMENT),
        producer_deployment=str(historical.PRODUCER_DEPLOYMENT),
        closure_root=str(historical.CLOSURE_ROOT),
        closure_deployment=str(historical.CLOSURE_DEPLOYMENT),
        closure_pins=copy.deepcopy(historical.CLOSURE_PINS),
        registration_ref=registration_ref,
        receipt_ref=receipt_ref,
    )
    _require(original["request"] == request, "Saved historical request changed")
    expected_transcript = [
        original_path.parent / name for name in ("REQUEST.json", "stdout.json", "stderr.log")
    ]
    _require(
        [row["path"] for row in original["transcript"]]
        == [str(path.relative_to(root)) for path in expected_transcript],
        "Complete original transcript required",
    )
    transcript = [_proof_bound(root, row, json_value=False) for row in original["transcript"]]
    submitted = historical._json(transcript[0])
    envelope = historical._json(transcript[1])
    execution = original["execution"]
    _require(
        set(submitted) == {"schema", "request", "execution"}
        and submitted["schema"] == historical.SCHEMA
        and submitted["request"] == request
        and execution
        == dict(
            submitted["execution"],
            completed_at=execution["completed_at"],
            child_pid=envelope["child_pid"],
            returncode=0,
        )
        and submitted["execution"]["command"]
        == [submitted["execution"]["command"][0], "-I", "-B", "-c", historical.BOOTSTRAP]
        and submitted["execution"]["cwd"] == str(historical.FOLLOWUP_DEPLOYMENT)
        and submitted["execution"]["bootstrap_sha256"]
        == hashlib.sha256(historical.BOOTSTRAP.encode()).hexdigest(),
        "Original isolated execution transport differs",
    )
    facts = historical._facts(envelope, request, registered)
    _require(
        original["facts"] == facts and original["facts_sha256"] == digest(facts),
        "Saved historical facts differ",
    )
    before = historical._snapshot(facts["inputs"])
    _require(
        len({item["path"] for item in facts["inputs"]}) == len(facts["inputs"]),
        "Duplicate historical input closure",
    )
    for item in facts["inputs"]:
        _historical_input_hash(item)
    _require(
        historical._request(registration_ref, receipt_ref) == registered,
        "Historical registration changed during verification",
    )
    _require(
        historical._snapshot(facts["inputs"]) == before, "Historical inventory changed during read"
    )
    _require(
        _proof_bound(root, confirmation_binding_ref) == binding
        and _proof_bound(root, original_ref) == original
        and [_proof_bound(root, row, json_value=False) for row in original["transcript"]]
        == transcript,
        "Historical proof transport changed during read",
    )
    verification = dict(original, report_ref=original_ref)
    expected = historical._binding(context, registration_ref, receipt_ref, verification)
    _require(binding == expected, "Handwritten or stale confirmation binding")
    return dict(
        binding_ref=copy.deepcopy(confirmation_binding_ref),
        binding=binding,
        verification=verification,
    )


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
    projected, artifacts = ({}, {})
    for split in overlay.SPLITS:
        projected[split], payload = overlay._split(reader, root, config, data_ref, split)
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
            root,
            context,
            (overlay, overlay.interface_v3_protocol, overlay.registration),
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


def _build_registration_readonly(
    root, *, phase_context, parent_references, overlay_ref, confirmation_binding_ref
):
    """Build new-source artifacts from original inputs, without publishing."""
    root = v3_registration._plain_path(root)
    context = _context_readonly(root, phase_context)
    v3_registration.pilot_v3_inputs._source_members(root, context, (v3_registration,))
    overlay = _verify_overlay_readonly(root, phase_context=context, overlay_ref=overlay_ref)
    confirmation = _confirmation_readonly(
        root, phase_context=context, confirmation_binding_ref=confirmation_binding_ref
    )
    v3_registration._require(
        confirmation.get("binding_ref") == confirmation_binding_ref,
        "Historical confirmation verifier returned another binding",
    )
    _, config, _, data_ref, order_ref, order, parents, binding_ref = (
        v3_registration.registration._inputs(root, context, parent_references)
    )
    bundle = v3_registration.registration._assemble_registration(
        root,
        context,
        parent_references,
        config,
        data_ref,
        order_ref,
        order,
        parents,
        binding_ref,
        shared_inputs=overlay["inputs_by_split"]["DEV_PILOT"],
        evaluation_inputs=overlay["inputs_by_split"]["DEV_EVAL"],
        phase=v3_registration.PHASE,
        directory=v3_registration.DIRECTORY,
        schema=v3_registration.SCHEMA,
        run_namespace=v3_registration.PHASE,
        identity_bindings={
            "input_overlay": overlay["manifest_ref"],
            "input_overlay_sha256": overlay["manifest_ref"]["sha256"],
            "prompt_version": v3_registration.pilot_v3_inputs.PROMPT_VERSION,
        },
        spec_bindings={
            "input_overlay": overlay["manifest_ref"],
            "prompt_version": v3_registration.pilot_v3_inputs.PROMPT_VERSION,
            "confirmation_binding_ref": confirmation_binding_ref,
        },
    )
    return {
        **bundle,
        "gate_status": "NOT_SIGNED",
        "overlay": overlay,
        "confirmation_binding": confirmation,
    }


def _verify_registration_readonly(root, *, phase_context, expected_spec_sha256):
    """Rebuild every artifact from verified v3 source, original DATA and parents."""
    root = v3_registration._plain_path(root)
    v3_registration._require(
        v3_registration._sha(expected_spec_sha256), "Pinned PILOT_V3 specification required"
    )
    destination = v3_registration._plain_path(root / v3_registration.DIRECTORY)
    payload = v3_registration._read_bytes(destination / v3_registration.SPEC_FILE)
    v3_registration._require(
        v3_registration.hashlib.sha256(payload).hexdigest() == expected_spec_sha256,
        "PILOT_V3 specification pin differs",
    )
    spec = v3_registration.json.loads(payload)
    v3_registration._require(
        spec.get("schema") == v3_registration.SCHEMA and spec.get("phase") == v3_registration.PHASE,
        "Explicit PILOT_V3 specification required",
    )
    bundle = _build_registration_readonly(
        root,
        phase_context=phase_context,
        parent_references=spec.get("parent_references"),
        overlay_ref=spec.get("input_overlay"),
        confirmation_binding_ref=spec.get("confirmation_binding_ref"),
    )
    v3_registration._require(
        bundle["spec_sha256"] == expected_spec_sha256,
        "PILOT_V3 specification differs from fixed reconstruction",
    )
    v3_registration.registration._check_files(destination, bundle)
    return v3_registration._result(bundle, "VERIFIED_REGISTRATION_NOT_AUTHORIZED")


def _registration_readonly(root, phase_context, pilot_spec_sha256):
    value = _registration_bundle_readonly(
        root, phase_context=phase_context, pilot_spec_sha256=pilot_spec_sha256
    )
    spec = value["spec"]
    item = next(p for p in spec["paths"] if (p["parent"], p["arm"]) == ("b0", "U11"))
    primary = frozen._path(root, item["audit_root"])
    identity = frozen._json(frozen._read_bytes(primary / "RUN_IDENTITY.json"))
    frozen._require(
        identity["lane"] == "PRIMARY"
        and identity["repeat"] == 0
        and (identity["max_steps"] == 12)
        and (identity["phase"] == frozen.pilot_contract.for_phase(phase_context["phase"]).phase),
        "Only registered b0/U11 primary identity can be imported",
    )
    return (spec, primary, identity)


def _verify_recovery_readonly(
    root, *, phase_context, pilot_spec_sha256, binding_ref, verify_payloads=True
):
    """Validate fixed storage binding; payload verification is needed once before cached access."""
    root = frozen._plain_path(root)
    contract = frozen.pilot_contract.for_phase(phase_context["phase"])
    frozen._require(
        binding_ref.get("path") == contract.recovery_directory + "/RECOVERY_BINDING.json",
        "Only the unique registered recovery binding is allowed",
    )
    binding = frozen._bound(root, binding_ref)
    _, primary, identity = _registration_readonly(root, phase_context, pilot_spec_sha256)
    contract = frozen.pilot_contract.for_phase(phase_context["phase"])
    destination = frozen._path(root, contract.recovery_directory)
    frozen._require(
        binding.get("format") == frozen.FORMAT
        and binding.get("directory") == contract.recovery_directory
        and (binding.get("storage_lane") == "RECOVERY")
        and (binding.get("logical_identity") == identity)
        and (binding.get("primary_directory") == primary.relative_to(root).as_posix())
        and (binding.get("phase_identity_sha256") == phase_context["identity_sha256"])
        and (binding.get("pilot_spec_sha256") == pilot_spec_sha256)
        and (binding.get("source_step") == 4)
        and (binding.get("target_step") == 12)
        and (binding.get("imported_actual_calls") == 512)
        and (binding.get("additional_new_calls") == 1024),
        "Recovery binding is outside the frozen contract",
    )
    frozen._require(
        frozen._json(frozen._read_bytes(destination / "RUN_IDENTITY.json")) == identity,
        "Imported identity changed",
    )
    source = frozen._bound(root, binding["source_commit_reference"])
    frozen._require(
        binding["source_commit_reference"]["path"]
        == (primary / "checkpoints/step-004/COMMIT.json").relative_to(root).as_posix()
        and source["commit_sha256"] == binding["source_commit_sha256"]
        and (source["sampler_state"] == binding["source_sampler_state"])
        and (source["rng_sha256"] == binding["source_rng_sha256"])
        and (source["policy_sha256"] == binding["source_policy_sha256"]),
        "Source checkpoint binding differs",
    )
    chain = frozen._markers(destination, identity)
    frozen._require(
        len(chain) >= 5 and [m["commit_sha256"] for m in chain[:5]] == binding["prefix_commits"],
        "Imported original commit prefix differs",
    )
    execution = frozen._execution(
        root,
        binding["primary_execution_ref"],
        primary,
        identity,
        require_continuous=True,
        pilot_spec_sha256=pilot_spec_sha256,
    )
    frozen._require(
        binding["scope"] == execution["end"]["scope"],
        "Recovery scope cannot upgrade synthetic evidence",
    )
    stamps = frozen._prefix_stamps(primary, destination, binding["prefix_files"])
    previous_stamps = None
    if verify_payloads and previous_stamps != stamps:
        if previous_stamps is None:
            original = frozen.RunTransactions(primary, identity)
            expected = frozen._prefix_files(
                original,
                frozen._markers(primary, identity),
                synthetic=binding["scope"] == "SYNTHETIC_ONLY",
            )
            lengths = frozen._audit_prefixes(original, frozen._markers(primary, identity))
            frozen._require(
                [v["path"] for v in binding["prefix_files"]] == expected,
                "Incomplete/unexpected imported reference closure",
            )
            frozen._require(
                {
                    v["path"]: v["source_byte_count"]
                    for v in binding["prefix_files"]
                    if "source_byte_count" in v
                }
                == lengths,
                "Original audit prefix boundary differs",
            )
        old = dict(previous_stamps or ())
        for item in binding["prefix_files"]:
            for directory in (primary, destination):
                path = frozen._path(directory, item["path"])
                if old.get(str(path)) != frozen._stamp(path):
                    actual = (
                        frozen.hashlib.sha256(frozen._inventory_bytes(directory, item)).hexdigest()
                        if "source_byte_count" in item
                        else frozen._file_hash(path)
                    )
                    frozen._require(actual == item["sha256"], "Imported prefix content differs")
        frozen._require(
            frozen._prefix_stamps(primary, destination, binding["prefix_files"]) == stamps,
            "Prefix changed during verification",
        )
    return dict(
        directory=str(destination),
        identity=identity,
        binding_ref=frozen.copy.deepcopy(binding_ref),
        binding=binding,
        source_step=4,
        target_step=12,
        imported_actual_calls=512,
        additional_new_calls=1024,
        scope=binding["scope"],
        status="VERIFIED_IMPORT_NOT_AUTHORIZED",
        gate_status="NOT_SIGNED",
        training_authorized=False,
    )


def _comparison_facts_readonly(
    primary_tx,
    recovery_tx,
    *,
    root,
    phase_context,
    pilot_spec_sha256,
    binding_ref,
    primary_execution_ref,
    recovery_execution_ref,
):
    """Save exact delivery facts. Numeric mismatches are FAIL; invalid evidence raises."""
    primary_tx._locked()
    recovery_tx._locked()
    root = frozen._plain_path(root)
    bound = _verify_recovery_readonly(
        root,
        phase_context=phase_context,
        pilot_spec_sha256=pilot_spec_sha256,
        binding_ref=binding_ref,
    )
    frozen._require(
        primary_tx.directory == root / bound["binding"]["primary_directory"]
        and recovery_tx.directory == frozen.Path(bound["directory"])
        and (primary_tx.identity == recovery_tx.identity == bound["identity"]),
        "Comparison runs differ",
    )
    frozen._require(
        primary_execution_ref == bound["binding"]["primary_execution_ref"],
        "Primary execution differs",
    )
    primary = frozen._execution(
        root,
        primary_execution_ref,
        primary_tx.directory,
        primary_tx.identity,
        require_continuous=True,
        pilot_spec_sha256=pilot_spec_sha256,
    )
    recovered = frozen._execution(
        root,
        recovery_execution_ref,
        recovery_tx.directory,
        recovery_tx.identity,
        pilot_spec_sha256=pilot_spec_sha256,
    )
    frozen._require(
        recovered["begin"]["storage_lane"] == "RECOVERY"
        and recovered["begin"]["start_step"] == 4
        and (recovered["begin"]["recovery_binding_ref"] == binding_ref)
        and (recovered["new_model_calls"] == 1024)
        and all(
            r["new_model_calls"] == 128 and r["reused_complete_groups"] == 0
            for r in recovered["commits"]
        )
        and (recovered["end"]["scope"] == primary["end"]["scope"] == bound["scope"]),
        "Independent registered eight-step/1024-new-call execution required",
    )
    independent_process = (
        primary["begin"]["process_identity"] != recovered["begin"]["process_identity"]
    )
    frozen._require(
        bound["scope"] != "SERVER_ACTUAL" or independent_process,
        "Registered recovery must execute in a new independent OS process",
    )
    for folder in (recovery_tx.directory / "EXECUTIONS").iterdir():
        if folder.name != recovered["end"]["task_id"]:
            frozen._require(
                (folder / "END.json").is_file(), "Unclosed independent recovery attempt"
            )
            frozen._require(
                not any(
                    e.get("kind") == "completion_attempts" and e.get("count", 0)
                    for e in frozen._events(
                        recovery_tx.directory / "accounting" / (folder.name + ".jsonl")
                    )
                ),
                "Extra recovery data calls exceed registered budget",
            )
    for ledger in (recovery_tx.directory / "accounting").glob("*.jsonl"):
        if ledger.name != recovered["end"]["task_id"] + ".jsonl":
            frozen._require(
                not any(
                    e.get("kind") == "completion_attempts" and e.get("count", 0)
                    for e in frozen._events(ledger)
                ),
                "Extra/orphan recovery data calls exceed budget",
            )
    left, right = (primary_tx.published(), recovery_tx.published())
    frozen._require(len(left) == len(right) == 13, "Both actual endpoints must be step12")
    differences = []
    for key in frozen.ENVIRONMENT_KEYS:
        frozen._differences(
            primary["runtime"].get(key),
            recovered["runtime"].get(key),
            f"environment/{key}",
            differences,
        )
    for key in frozen.HARDWARE_KEYS:
        frozen._differences(
            primary["runtime"].get("hardware", {}).get(key),
            recovered["runtime"].get("hardware", {}).get(key),
            f"hardware/{key}",
            differences,
        )
    step_facts = []
    for step in range(4, 13):
        a, b = (
            frozen._state(primary_tx, left[step], gpu=bound["scope"] == "SERVER_ACTUAL"),
            frozen._state(recovery_tx, right[step], gpu=bound["scope"] == "SERVER_ACTUAL"),
        )
        start = len(differences)
        for key in (
            "policy",
            "optimizer",
            "scheduler",
            "python_rng",
            "numpy_rng",
            "torch_rng",
            "cuda_rng",
        ):
            frozen._differences(a[key], b[key], f"step{step}/state/{key}", differences)
        frozen._differences(
            left[step]["parameter_names"],
            right[step]["parameter_names"],
            f"step{step}/parameter_order",
            differences,
        )
        frozen._differences(
            frozen._without(a["sampler_state"], ("rollout_rng_reference",)),
            frozen._without(b["sampler_state"], ("rollout_rng_reference",)),
            f"step{step}/sampler",
            differences,
        )
        if step > 4:
            frozen._differences(
                frozen._originals(
                    primary_tx, left[step], synthetic=bound["scope"] == "SYNTHETIC_ONLY"
                ),
                frozen._originals(
                    recovery_tx, right[step], synthetic=bound["scope"] == "SYNTHETIC_ONLY"
                ),
                f"step{step}/original_raw_rng",
                differences,
            )
            ar = frozen._bound(primary_tx.directory, left[step]["references"]["rollout"])
            br = frozen._bound(recovery_tx.directory, right[step]["references"]["rollout"])
            frozen._differences(
                [frozen._row(r) for r in ar["records"]],
                [frozen._row(r) for r in br["records"]],
                f"step{step}/original_rows",
                differences,
            )
            for key in (
                "identity",
                "step",
                "policy_before_sha256",
                "rng_before_sha256",
                "rng_after_sha256",
            ):
                frozen._differences(
                    ar.get(key), br.get(key), f"step{step}/rollout/{key}", differences
                )
            for kind, exclusions in (("credit", ()), ("evidence", frozen.EVIDENCE_EXECUTION_KEYS)):
                av = frozen._bound(primary_tx.directory, left[step]["references"][kind])
                bv = frozen._bound(recovery_tx.directory, right[step]["references"][kind])
                if kind == "evidence":
                    frozen._audit_event(
                        primary_tx.directory, av, required=bound["scope"] == "SERVER_ACTUAL"
                    )
                    frozen._audit_event(
                        recovery_tx.directory, bv, required=bound["scope"] == "SERVER_ACTUAL"
                    )
                frozen._differences(
                    frozen._without(av, exclusions),
                    frozen._without(bv, exclusions),
                    f"step{step}/{kind}",
                    differences,
                )
            au = frozen._json(
                frozen._read_bytes(primary_tx.checkpoints / f"step-{step:03d}/UPDATE.json")
            )
            bu = frozen._json(
                frozen._read_bytes(recovery_tx.checkpoints / f"step-{step:03d}/UPDATE.json")
            )
            frozen._differences(
                frozen._without(au, frozen.UPDATE_EXECUTION_KEYS),
                frozen._without(bu, frozen.UPDATE_EXECUTION_KEYS),
                f"step{step}/numerical_gradient",
                differences,
            )
        step_facts.append(
            dict(
                step=step,
                exact=len(differences) == start,
                primary_commit=left[step]["commit_sha256"],
                recovery_commit=right[step]["commit_sha256"],
            )
        )
    result = dict(
        format=frozen.FORMAT,
        status="FAIL"
        if differences
        else "PASS"
        if bound["scope"] == "SERVER_ACTUAL"
        else "SYNTHETIC_ONLY",
        scope=bound["scope"],
        exact=not differences,
        binding_ref=binding_ref,
        primary_execution_ref=primary_execution_ref,
        recovery_execution_ref=recovery_execution_ref,
        new_generated_responses=1024,
        independent_process_verified=independent_process and bound["scope"] == "SERVER_ACTUAL",
        process_pair={
            "primary": primary["begin"]["process_identity"],
            "recovery": recovered["begin"]["process_identity"],
        },
        hardware_pair={
            "primary": primary["runtime"].get("hardware"),
            "recovery": recovered["runtime"].get("hardware"),
        },
        compared_hardware_fields=list(frozen.HARDWARE_KEYS),
        compared_environment_fields=list(frozen.ENVIRONMENT_KEYS),
        differences=differences,
        steps=step_facts,
        gate_status="NOT_SIGNED",
        training_authorized=False,
    )
    return result


def _registration_bundle_readonly(root, *, phase_context, pilot_spec_sha256):
    _require(phase_context["phase"] == "PILOT_V3", "Only actual PILOT_V3 delivery is supported")
    return _verify_registration_readonly(
        root, phase_context=phase_context, expected_spec_sha256=pilot_spec_sha256
    )


def verify_saved_recovery(
    primary_tx,
    recovery_tx,
    *,
    root,
    phase_context,
    pilot_spec_sha256,
    binding_ref,
    comparison_ref,
    recovery_execution_ref,
):
    """Recompute recovery delivery; consume, never rerun, historical admission."""
    primary_tx._locked()
    recovery_tx._locked()
    _require(phase_context["phase"] == "PILOT_V3", "PILOT_V3 context required")
    root = frozen._plain_path(root)
    path = frozen._path(root, comparison_ref["path"])
    _require(
        path.parent == recovery_tx.directory / "COMPARISONS"
        and re.fullmatch(r"[0-9a-f]{32}\.json", path.name),
        "Fixed comparison receipt path required",
    )
    saved = frozen._bound(root, comparison_ref)
    before = {tx.directory: pilot_audit._snapshot(tx.directory) for tx in (primary_tx, recovery_tx)}
    actual = _comparison_facts_readonly(
        primary_tx,
        recovery_tx,
        root=root,
        phase_context=phase_context,
        pilot_spec_sha256=pilot_spec_sha256,
        binding_ref=binding_ref,
        primary_execution_ref=saved["primary_execution_ref"],
        recovery_execution_ref=recovery_execution_ref,
    )
    _require(saved == actual, "Saved comparison differs from recomputed facts")
    _require(actual["scope"] == "SERVER_ACTUAL", "Actual recovery evidence required")
    compared = {
        **actual,
        "receipt_reference": copy.deepcopy(comparison_ref),
        "verified_from_saved_facts": True,
    }
    history = pilot_audit._comparison_inventory(root, recovery_tx.directory, compared)
    _require(
        all(pilot_audit._snapshot(path) == stamp for path, stamp in before.items()),
        "Recovery inventory changed during read",
    )
    return dict(
        status="RECOVERY_FACTS_RECHECKED_NOT_GATE_SIGNED",
        comparison=compared,
        comparison_history=history,
        gate_status="NOT_SIGNED",
        evaluation_authorized=False,
        full_delivery_verified=False,
        historical_confirmation_scope="EXISTING_BOUND_PROOF_AND_INPUT_BYTES_VERIFIED_NOT_RECOMPUTED",
        new_responses=0,
        new_updates=0,
    )
