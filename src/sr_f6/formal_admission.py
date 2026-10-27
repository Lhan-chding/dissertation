"""Actual FORMAL training/DEV36 admission; TEST always needs a separate grant.

Cold assessment/sealing/runtime entry replays historical PILOT in its isolated
producer and verifies both independent frozen plans. New bridge execution IDs
are retained, but equality is over the complete original facts and scientific
dependencies, never those fresh transport paths. Every historical bridge
attempt is enumerated; unresolved failures/partials cannot be selected away.

Only a successful in-process recomputation populates the PID-local cache. Warm
calls check full prerequisite membership/stat identities; changed content and
new attempts invalidate it. Call at session/update boundaries, never per token,
prepare_input, or forward. No persisted cache or status string grants access.
Formal training outputs are not their own admission prerequisites.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import os
import re
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from . import formal_evaluation_plan as evaluation
from . import formal_registration as registration
from . import historical_pilot as history
from . import phases, pilot_audit
from .admission import _synthetic
from .common import digest, file_hash
from .source_binding import verify_bound_files
from .transactions import _fsync_dir

SCHEMA = "srf6-formal-admission-v1"
PATH = "PHASES/FORMAL_V1/ADMISSION.json"
ASSESSMENTS = "PHASES/FORMAL_V1/ADMISSION_ASSESSMENTS"
SCOPE = "FORMAL_V1_REGISTERED_TRAINING_AND_DEV36_ONLY"
_CACHE = {}


def _actual_deployment():
    return Path(__file__).resolve().parents[2]


class _Unknown(PermissionError):
    pass


class _VerifiedAssessment(dict):
    """Private live verification proof; JSON round-tripping never preserves it."""


def _require(value, message):
    if not value:
        raise PermissionError(message)


def _ref(value):
    _require(
        isinstance(value, dict) and set(value) == {"path", "sha256"}, "Exact reference required"
    )
    phases._relative(value["path"])
    phases._sha(value["sha256"])
    return value


class _Evidence:
    def __init__(self, root):
        self.root, self.inputs, self.external = root, {}, {}
        self.synthetic = False
        self.verified = {}
        self.bound_files = {}

    def adopt(self, refs):
        for ref in refs:
            _ref(ref)
            _require(
                self.inputs.setdefault(ref["path"], copy.deepcopy(ref)) == ref,
                "Conflicting prerequisite content pins",
            )

    def read(self, ref, *, opaque=False):
        _ref(ref)
        path = phases._file(self.root, ref["path"])
        before = history._stamp(path)
        raw = path.read_bytes()
        _require(
            history._stamp(path) == before and hashlib.sha256(raw).hexdigest() == ref["sha256"],
            "Prerequisite changed while reading",
        )
        value = raw if opaque else history._json(raw)
        self.adopt([ref])
        self.verified[str(path)] = before
        self.synthetic |= _synthetic(value)
        return value

    def outside(self, path, sha):
        path = Path(path)
        phases._sha(sha)
        _require(
            path.is_absolute() and path.resolve(strict=True).is_relative_to(phases.PROJECT),
            "External prerequisite outside project",
        )
        item = dict(path=str(path), resolved_path=str(path.resolve(strict=True)), sha256=sha)
        _require(self.external.setdefault(str(path), item) == item, "External content pin differs")


def _history_request(request):
    return dict(
        formal_phase_identity_sha256=request["phase_identity_sha256"],
        pilot_phase_identity_sha256=request["pilot_phase_identity_sha256"],
        pilot_spec_sha256=request["pilot_spec_sha256"],
        admission_ref=request["pilot_admission_ref"],
        recovery_binding_ref=request["recovery_binding_ref"],
        recovery_comparison_ref=request["recovery_comparison_ref"],
        evaluation_summary_ref=request["evaluation_summary_ref"],
    )


def _arguments(request):
    values = _history_request(request)
    values["phase_identity_sha256"] = values.pop("pilot_phase_identity_sha256")
    del values["formal_phase_identity_sha256"]
    return values


def _request(**values):
    _require(
        set(values)
        == {
            "phase_identity_sha256",
            "formal_spec_sha256",
            "evaluation_spec_sha256",
            "pilot_phase_identity_sha256",
            "pilot_spec_sha256",
            "pilot_admission_ref",
            "recovery_binding_ref",
            "recovery_comparison_ref",
            "evaluation_summary_ref",
        },
        "Exact formal admission request required",
    )
    for key in (
        "phase_identity_sha256",
        "formal_spec_sha256",
        "evaluation_spec_sha256",
        "pilot_phase_identity_sha256",
        "pilot_spec_sha256",
    ):
        phases._sha(values[key])
    for key in (
        "pilot_admission_ref",
        "recovery_binding_ref",
        "recovery_comparison_ref",
        "evaluation_summary_ref",
    ):
        _ref(values[key])
    return copy.deepcopy(values)


def _source(evidence, request):
    consumer, producer = history._contexts(
        evidence.root, request["phase_identity_sha256"], request["pilot_phase_identity_sha256"]
    )
    _require(
        _actual_deployment() == Path(consumer["deployment"]),
        "Formal admission is not executing its actual source",
    )
    evidence.synthetic |= _synthetic(consumer) or _synthetic(producer)
    return consumer, producer


def _plan_inputs(evidence, context, request):
    trained = registration.verify_formal_registration(
        evidence.root, phase_context=context, expected_spec_sha256=request["formal_spec_sha256"]
    )
    evaluated = evaluation.verify_formal_evaluation_plan(
        evidence.root, phase_context=context, expected_spec_sha256=request["evaluation_spec_sha256"]
    )
    _require(
        trained.get("status") == "VERIFIED_REGISTRATION_NOT_AUTHORIZED"
        and evaluated.get("status") == "VERIFIED_PLAN_NOT_AUTHORIZED",
        "Both verified formal plans required",
    )
    train, plan = trained["spec"], evaluated["spec"]
    for key in (
        "phase_identity",
        "actual_source_manifest",
        "origin_identity",
        "config_sha256",
        "data_manifest",
        "source_binding",
        "parent_references",
        "parent_tensors",
        "order_source",
    ):
        _require(
            train.get(key) == plan.get(key), "Training/evaluation common identity differs: " + key
        )
    _require(
        train["phase_identity"]
        == dict(path="PHASES/FORMAL_V1/IDENTITY.json", sha256=context["identity_sha256"])
        and train["actual_source_manifest"] == context["identity"]["source_manifest"],
        "Plans claim another actual source",
    )
    pilot_ref = dict(path=history.SPEC_PATH, sha256=request["pilot_spec_sha256"])
    _require(
        train["pilot_producer"] == plan["pilot_parent_provenance"]
        and train["pilot_producer"]["specification"] == pilot_ref
        and train["pilot_producer"]["phase_identity"]["sha256"]
        == request["pilot_phase_identity_sha256"],
        "Plans refer to different historical PILOT",
    )
    paths = train["paths"]
    _require(
        len(paths) == 54 and len({p["run_id"] for p in paths}) == 54,
        "Exactly 54 formal paths required",
    )
    evidence.adopt(
        [
            dict(
                path=f"{registration.DIRECTORY}/{registration.SPEC_FILE}",
                sha256=request["formal_spec_sha256"],
            )
        ]
    )
    for path in paths:
        for name in ("transaction_identity", "rollout_registration"):
            ref = path[name]
            evidence.adopt(
                [dict(path=f"{registration.DIRECTORY}/{ref['path']}", sha256=ref["sha256"])]
            )
    _require(len(evaluated["frozen_files"]) == 11, "Complete independent evaluation plan required")
    evidence.adopt(evaluated["frozen_files"])
    # Provider verifiers checked these small plan payloads; establish their
    # exact verified metadata here, rather than attaching later stat to old SHA.
    with phases._ContentCache(evidence.root) as cache:
        for ref in list(evidence.inputs.values()):
            path = phases._file(evidence.root, ref["path"])
            before = history._stamp(path)
            _require(
                cache.hash(path) == ref["sha256"] and history._stamp(path) == before,
                "Formal plan changed after verification",
            )
            evidence.verified[str(path)] = before
    selectors_ref = plan["endpoint_selectors"]
    path = phases._file(evidence.root, f"{evaluation.DIRECTORY}/{selectors_ref['path']}")
    _require(file_hash(path) == selectors_ref["sha256"], "Evaluation selectors changed")
    selectors = [history._json(line) for line in path.read_bytes().splitlines() if line.strip()]
    for stage, step in (("DEV36", 36), ("FINAL", 72)):
        actual = [
            p
            for p in selectors
            if p["stage"] == stage and p["kind"] == "FUTURE_FORMAL_COMMIT_SELECTOR"
        ]
        expected = {
            (p["run_id"], p["parent"], p["repeat"], p["arm"], p["audit_root"]) for p in paths
        }
        _require(
            len(actual) == 54
            and {
                (p["run_id"], p["parent"], p["repeat"], p["arm"], p["expected_run_directory"])
                for p in actual
            }
            == expected
            and all(p["checkpoint_step"] == step for p in actual),
            "Evaluation endpoint selectors differ from training paths",
        )
    _require(
        plan["access"]["training_grant_implies_test_access"] is False,
        "Training grant may never open TEST",
    )
    evidence.synthetic |= _synthetic(trained) or _synthetic(evaluated)
    return train, plan


def _closure(evidence, refs, visited=None):
    visited = set() if visited is None else visited
    for reference in refs:
        evidence.adopt([reference])
        if reference["path"] in visited or not reference["path"].endswith(".json"):
            continue
        visited.add(reference["path"])
        value = evidence.read(reference)
        if isinstance(value, dict):
            for key in ("inputs", "evidence"):
                dependencies = value.get(key, [])
                _require(isinstance(dependencies, list), "Explicit prerequisite closure required")
                _closure(evidence, dependencies, visited)


def _dependency_inputs(evidence, context, request, plan):
    """Adopt already-verified inventories; do not read all images/model weights again."""
    identity = context["identity"]
    pending = [
        (dict(path="PHASES/FORMAL_V1/IDENTITY.json", sha256=context["identity_sha256"]), identity)
    ]
    seen, deployments = set(), []
    while pending:
        reference, item = pending.pop()
        if reference["path"] in seen:
            continue
        seen.add(reference["path"])
        evidence.read(reference)
        evidence.read(item["delta"])
        manifest = evidence.read(item["source_manifest"])
        deployment = Path(item["deployment"])
        deployments.append(str(deployment))
        for member in manifest["files"]:
            evidence.outside(deployment / member["path"], member["sha256"])
        for upstream in item["upstream"]:
            _closure(evidence, upstream["evidence"])
            if upstream["producer_phase"] == phases.ORIGIN:
                continue
            ref = dict(
                path=f"PHASES/{upstream['producer_phase']}/IDENTITY.json",
                sha256=upstream["identity_sha256"],
            )
            pending.append((ref, evidence.read(ref)))
    origin = evidence.read(identity["origin"]["identity"])
    manifest = evidence.read(origin["source_manifest"])
    origin_deployment = Path(identity["origin"]["deployment"])
    deployments.append(str(origin_deployment))
    for member in manifest["files"]:
        evidence.outside(origin_deployment / member["path"], member["sha256"])
    data = evidence.read(identity["data_manifest"])
    evidence.adopt([dict(path=m["path"], sha256=m["sha256"]) for m in data["files"]])
    evidence.read(dict(path="CONFIG.json", sha256=identity["config_sha256"]))
    admitted = evidence.read(request["pilot_admission_ref"])
    evidence.adopt(admitted["inputs"])
    for item in admitted["external_inputs"]:
        evidence.outside(item["path"], item["sha256"])
    protocol = evidence.read(plan["retention_protocol_binding"])
    evidence.outside(protocol["original_receipt"]["path"], protocol["original_receipt"]["sha256"])
    for item in [*protocol["files"], *protocol["inherited_native_files"]]:
        for name in ("original_path", "formal_path"):
            evidence.outside(item[name], item["sha256"])
    # Reuse the actual source-binding file metadata for large frozen objects.
    # The later proof still compares the expected content pin and current stat;
    # it must never pair an old SHA with an unverified newly observed stat.
    source = evidence.read(plan["source_binding"])
    inventories = [
        *source.get("source_files", []),
        *source.get("model_files", []),
        *source.get("retention", {}).values(),
        *[row for parent in source.get("parents", {}).values() for row in parent["files"]],
        *[row for rows in source.get("retention_images", {}).values() for row in rows],
    ]
    if "previous_config" in source:
        inventories.append(source["previous_config"])
    for item in inventories:
        item = {
            key: item[key] for key in ("path", "resolved_path", "bytes", "sha256", "stat_cache_key")
        }
        _require(
            evidence.bound_files.setdefault(item["path"], item) == item,
            "Conflicting original source inventory",
        )
    return sorted(set(deployments))


def _attempts(evidence, context, producer, request):
    folder = phases._path(evidence.root / history.DIRECTORY, exists=False)
    if not folder.exists():
        return []
    reports = []
    for attempt in sorted(folder.iterdir()):
        phases._path(attempt)
        _require(
            attempt.is_dir() and re.fullmatch(r"[0-9a-f]{32}", attempt.name),
            "Unknown historical attempt entry",
        )
        names = {p.name for p in attempt.iterdir()}
        if names != {"REQUEST.json", "stdout.json", "stderr.log", "RESULT.json", "tmp", "cache"}:
            raise _Unknown("Historical verification failure/partial is preserved: " + attempt.name)
        for name in ("tmp", "cache"):
            _require(phases._path(attempt / name).is_dir(), "Historical scratch path changed")
        refs = {
            name: history._ref(evidence.root, phases._file(attempt, name))
            for name in ("REQUEST.json", "stdout.json", "stderr.log", "RESULT.json")
        }
        saved = evidence.read(refs["REQUEST.json"])
        envelope = evidence.read(refs["stdout.json"])
        result = evidence.read(refs["RESULT.json"])
        evidence.read(refs["stderr.log"], opaque=True)
        _require(
            saved["format"] == history.FORMAT
            and saved["context"] == context["identity"]
            and saved["producer_context"] == producer["identity"],
            "Historical attempt producer/consumer differs",
        )
        transport = dict(
            root=str(evidence.root),
            deployment=producer["deployment"],
            nonce=attempt.name,
            arguments=_arguments(request),
        )
        _require(saved["transport"] == transport, "Historical attempt is outside pinned request")
        facts = history._result(envelope, transport, producer)
        execution = result["execution"]
        _require(
            saved["execution"]["command"] == [sys.executable, "-I", "-B", "-c", history.BOOTSTRAP]
            and saved["execution"]["cwd"] == producer["deployment"]
            and saved["execution"]["bootstrap_sha256"]
            == hashlib.sha256(history.BOOTSTRAP.encode()).hexdigest()
            and all(execution.get(k) == v for k, v in saved["execution"].items())
            and execution["returncode"] == 0
            and type(execution["child_pid"]) is int
            and execution["child_pid"] > 0,
            "Historical execution command/return identity differs",
        )
        start, end = (datetime.fromisoformat(execution[k]) for k in ("started_at", "completed_at"))
        _require(
            start.tzinfo is not None and end.tzinfo is not None and end >= start,
            "Actual historical UTC interval required",
        )
        _require(
            result["format"] == history.FORMAT
            and result["status"] == "FACTS_RECOMPUTED_NOT_AUTHORIZED"
            and result["context"] == context["identity"]
            and result["producer_context"] == producer["identity"]
            and result["request"] == _arguments(request)
            and result["historical_result"] == facts
            and result["historical_result_sha256"] == digest(facts)
            and result["gate_status"] == "NOT_SIGNED"
            and all(
                result.get(k) is False
                for k in (
                    "training_authorized",
                    "generation_authorized",
                    "evaluation_authorized",
                    "formal_authorized",
                )
            ),
            "Historical result is not the original non-authorizing verifier output",
        )
        expected_inputs = history._references(
            evidence.root,
            [
                dict(path=history.SPEC_PATH, sha256=request["pilot_spec_sha256"]),
                *[
                    request[k]
                    for k in (
                        "pilot_admission_ref",
                        "recovery_binding_ref",
                        "recovery_comparison_ref",
                        "evaluation_summary_ref",
                    )
                ],
                refs["REQUEST.json"],
                refs["stdout.json"],
                refs["stderr.log"],
                *facts["inputs"],
            ],
        )
        _require(result["inputs"] == expected_inputs, "Historical result input closure differs")
        evidence.adopt(result["inputs"])
        reports.append(dict(reference=refs["RESULT.json"], facts=facts))
    return reports


def _pilot_facts(facts):
    expected = pilot_audit._exposure(facts["runs"])
    _require(
        expected == facts["exposure"], "Historical U11 exposure differs from all original groups"
    )
    for run in facts["runs"]:
        _require(
            run["scope"] == "SERVER_ACTUAL"
            and run["updates"] == 12
            and run["responses"] == 1536
            and len(run["groups"]) == 192
            and len(run["steps"]) == 12,
            "Incomplete original pilot run facts",
        )
    if facts["status"] == "RECOVERY_COMPARISON_HISTORY_UNKNOWN":
        raise _Unknown("Historical recovery comparison attempt remains unresolved")
    if facts["status"] == "RECOVERY_MISMATCH":
        raise PermissionError("Actual recovery comparison mismatch")
    if not expected["registered_exposure_met"]:
        return dict(status="INSUFFICIENT_REGISTERED_EXPOSURE", exposure=expected)
    _require(
        facts["status"] == "COMPLETE_FACTS_NOT_GATE_SIGNED",
        "Historical status contradicts actual exposure",
    )
    recovery = facts["recovery"]
    _require(
        recovery["status"] == "PASS"
        and recovery["scope"] == "SERVER_ACTUAL"
        and recovery["exact"] is True
        and recovery["independent_process_verified"] is True
        and recovery["new_generated_responses"] == 1024
        and facts["recovery_comparison_history"]["status"] == "ALL_SAVED_FACTS_VERIFIED",
        "Exact independent registered recovery required",
    )
    result = facts["evaluation"]
    _require(
        result["status"] == "COMPLETE_DESCRIPTIVE_NOT_GATE_SIGNED"
        and result["checkpoints"] == 12
        and result["responses"] == 2304,
        "Complete descriptive DEV_EVAL required",
    )
    return dict(status="PASS", exposure=expected, effectiveness_gate=None)


def _layout(root, context, request, deployments):
    snapshot = history._snapshot(root, _arguments(request))
    # All original files, including checkpoints not exposed as flat references,
    # participate in the metadata boundary. Formal outputs deliberately do not.
    trees = {"history": sorted(snapshot)}
    for deployment in deployments:
        base = Path(deployment)
        trees[deployment] = sorted(
            phases._source_paths(
                base, legacy=deployment == context["identity"]["origin"]["deployment"]
            )
        )
    frozen = phases._path(root / registration.DIRECTORY)
    trees["training_layout"] = sorted(
        str(p.relative_to(frozen))
        for pattern in ("runs/*", "runs/*/*", "runs/*/*/*")
        for p in frozen.glob(pattern)
    )
    evaluated = phases._path(root / evaluation.DIRECTORY)
    trees["evaluation_layout"] = sorted(str(p.relative_to(evaluated)) for p in evaluated.rglob("*"))
    folder = phases._path(root / history.DIRECTORY, exists=False)
    trees["bridge_attempts"] = (
        sorted(str(p.relative_to(folder)) for pattern in ("*", "*/*") for p in folder.glob(pattern))
        if folder.exists()
        else []
    )
    for relative in trees["bridge_attempts"]:
        path = phases._path(folder / relative)
        snapshot[str(path)] = history._stamp(path)
    return trees, snapshot


def _proof(evidence, context, request, deployments, before=None):
    trees, snapshot = _layout(evidence.root, context, request, deployments)
    pins = {
        str(phases._file(evidence.root, ref["path"])): ref["sha256"]
        for ref in evidence.inputs.values()
    }
    root_paths = set(pins)
    for item in evidence.external.values():
        _require(
            pins.setdefault(item["path"], item["sha256"]) == item["sha256"],
            "Conflicting dependency pin",
        )
    with phases._ContentCache(evidence.root) as cache:
        for name, sha in pins.items():
            path = Path(name)
            stamp = history._stamp(path)
            if name not in evidence.verified:
                bound = evidence.bound_files.get(name)
                if bound is not None:
                    _require(bound["sha256"] == sha, "Original source pin differs")
                    verify_bound_files([bound], boundary=phases.PROJECT)
                else:
                    _require(cache.hash(path) == sha, "Adopted prerequisite content differs")
                _require(history._stamp(path) == stamp, "Adopted prerequisite changed")
                evidence.verified[name] = stamp
            snapshot[name] = stamp
    _require(
        all(snapshot.get(name) == stamp for name, stamp in evidence.verified.items()),
        "Evidence changed after its actual content verification",
    )
    if before is not None:
        _require(
            all(snapshot.get(name) == stamp for name, stamp in before["snapshot"].items()),
            "Frozen prerequisites changed during historical replay",
        )
        _require(
            {k: v for k, v in trees.items() if k != "bridge_attempts"}
            == {k: v for k, v in before["trees"].items() if k != "bridge_attempts"},
            "Frozen prerequisite layout changed during replay",
        )
    return dict(trees=trees, snapshot=snapshot, pins=pins, root_paths=root_paths)


def _assert_verified(root, assessment):
    _require(
        type(assessment) is _VerifiedAssessment
        and hasattr(assessment, "_verification")
        and assessment._verification["assessment_sha256"] == digest(assessment),
        "A live, unchanged actual assessment is required",
    )
    proof = assessment._verification
    trees, snapshot = _layout(
        root, assessment["phase_context"], assessment["request"], assessment["deployments"]
    )
    _require(trees == proof["trees"], "Prerequisite layout changed before cache installation")
    for name, expected in proof["snapshot"].items():
        if name in proof["root_paths"]:
            phases._file(root, Path(name).relative_to(root).as_posix())
        value = snapshot[name] if name in snapshot else history._stamp(Path(name))
        _require(value == expected, "Verified prerequisite changed before cache installation")
    return proof


@contextmanager
def _lock(root):
    directory = phases._path(root / "PHASES/FORMAL_V1")
    path = phases._path(directory / ".admission.lock", exists=False)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _assess(root, request):
    evidence = _Evidence(root)
    checks, context, producer, train, plan, fresh, deployments = (
        {},
        None,
        None,
        None,
        None,
        None,
        [],
    )
    verified = None

    def check(name, function):
        try:
            value = function()
            checks[name] = (
                value if isinstance(value, dict) and "status" in value else dict(status="PASS")
            )
            return value
        except (OSError, ValueError, TypeError, KeyError, ImportError, PermissionError) as exc:
            checks[name] = dict(
                status="UNKNOWN"
                if isinstance(
                    exc,
                    (
                        FileNotFoundError,
                        ImportError,
                        _Unknown,
                        history.HistoricalPilotVerificationError,
                    ),
                )
                else "FAIL",
                error=type(exc).__name__ + ": " + str(exc),
            )
            return None

    source = check("source", lambda: _source(evidence, request))
    if source:
        context, producer = source
        plans = check("plans", lambda: _plan_inputs(evidence, context, request))
        if plans:
            train, plan = plans
            dep = check(
                "dependencies", lambda: _dependency_inputs(evidence, context, request, plan)
            )
            if dep is not None:
                deployments = dep
                prior = check(
                    "prior_attempts", lambda: _attempts(evidence, context, producer, request)
                )
                if prior is not None:
                    # An old negative result is not a candidate to replace.
                    prior_ok = check("prior_facts", lambda: _all_previous(prior))
                    if prior_ok is not None and prior_ok["status"] == "PASS":
                        before = check(
                            "initial_boundary",
                            lambda: _proof(evidence, context, request, deployments),
                        )
                        if before is None:
                            prior_ok = None
                    if prior_ok is not None and prior_ok["status"] == "PASS":
                        fresh = check(
                            "historical_replay",
                            lambda: history.verify_historical_pilot(
                                root, **_history_request(request)
                            ),
                        )
                        if fresh:
                            _require(
                                fresh.get("status") == "FACTS_RECOMPUTED_NOT_AUTHORIZED",
                                "Actual historical bridge result required",
                            )
                            checks["historical_replay"] = dict(
                                status="PASS", bridge_status=fresh["status"]
                            )
                            evidence.synthetic |= _synthetic(fresh)
                            reports = check(
                                "complete_attempts",
                                lambda: _attempts(evidence, context, producer, request),
                            )
                            if reports is not None:
                                evidence.read(fresh["report_ref"])
                                check("historical_facts", lambda: _matching_facts(reports, fresh))
                            verified = check(
                                "stable_inputs",
                                lambda: _proof(evidence, context, request, deployments, before),
                            )
    required = {
        "source",
        "plans",
        "dependencies",
        "prior_attempts",
        "prior_facts",
        "initial_boundary",
        "historical_replay",
        "complete_attempts",
        "historical_facts",
        "stable_inputs",
    }
    for name in required - checks.keys():
        checks[name] = dict(status="UNKNOWN", error="Required actual facts unavailable")
    states = [checks[name]["status"] for name in required]
    status = (
        "PASS"
        if all(v == "PASS" for v in states)
        else "FAIL"
        if "FAIL" in states
        else "INSUFFICIENT_REGISTERED_EXPOSURE"
        if "INSUFFICIENT_REGISTERED_EXPOSURE" in states
        else "UNKNOWN"
    )
    stable = None
    if context is not None and train is not None and plan is not None and fresh is not None:
        stable = dict(
            context_sha256=digest(context["identity"]),
            producer_sha256=digest(producer["identity"]),
            historical_result_sha256=digest(fresh["historical_result"]),
            historical_inputs=history._references(root, fresh["historical_result"]["inputs"]),
            formal_spec_sha256=request["formal_spec_sha256"],
            evaluation_spec_sha256=request["evaluation_spec_sha256"],
            parent_tensors=train["parent_tensors"],
            config_sha256=train["config_sha256"],
            data_manifest=train["data_manifest"],
            order_source=train["order_source"],
            pilot_specification=train["pilot_producer"]["specification"],
            dependency_pins=[
                evidence.inputs[p]
                for p in sorted(evidence.inputs)
                if not p.startswith(history.DIRECTORY + "/")
            ],
            external_pins=[evidence.external[p] for p in sorted(evidence.external)],
        )
    result = _VerifiedAssessment(
        schema=SCHEMA,
        status="SYNTHETIC_ONLY" if evidence.synthetic else status,
        scientific_status=status,
        execution_origin="SYNTHETIC_TEST" if evidence.synthetic else "ACTUAL_EVIDENCE_VALIDATION",
        scope=SCOPE,
        context=None if context is None else context["identity"],
        phase_context=context,
        request=copy.deepcopy(request),
        checks=checks,
        stable_facts=stable,
        inputs=[evidence.inputs[p] for p in sorted(evidence.inputs)],
        external_inputs=[evidence.external[p] for p in sorted(evidence.external)],
        deployments=deployments,
        historical_report_ref=None if fresh is None else fresh["report_ref"],
        formal_spec=train,
        evaluation_spec=plan,
        training_authorized=False,
        generation_authorized=False,
        dev36_authorized=False,
        formal_test_access=False,
        final_evaluation_authorized=False,
        gate_status="NOT_SIGNED",
    )
    if result["status"] == "PASS" and verified is not None:
        result._verification = {**verified, "assessment_sha256": digest(result)}
    return result


def _all_previous(reports):
    for item in reports:
        result = _pilot_facts(item["facts"])
        if result["status"] != "PASS":
            return result
    return dict(status="PASS")


def _matching_facts(reports, fresh):
    _require(
        reports and any(r["reference"] == fresh["report_ref"] for r in reports),
        "Fresh actual verification attempt missing",
    )
    value = fresh["historical_result"]
    _require(
        all(r["facts"] == value for r in reports),
        "Historical verification attempts disagree; favorable selection forbidden",
    )
    return _pilot_facts(value)


def assess_formal_admission(root, **request):
    """Actual assessment (including preserved bridge attempts), never grants access."""
    root = phases._root(root)
    request = _request(**request)
    with _lock(root):
        return copy.deepcopy(dict(_assess(root, request)))


def _publish(path, value):
    path = phases._path(path, exists=False)
    payload = history._payload(value)
    if path.exists():
        _require(
            path.read_bytes() == payload, "Preserve differing/partial formal admission artifact"
        )
        return
    missing = []
    directory = path.parent
    while not directory.exists():
        missing.append(directory)
        directory = directory.parent
    path.parent.mkdir(parents=True, exist_ok=True)
    for directory in reversed(missing):
        _fsync_dir(directory.parent)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _require(not path.exists() and not path.is_symlink(), "Never replace formal admission")
    os.rename(temporary, path)
    _fsync_dir(path.parent)


def _receipt(root, assessment, assessment_ref):
    _require(
        assessment["status"] == "PASS"
        and assessment["execution_origin"] == "ACTUAL_EVIDENCE_VALIDATION"
        and not _synthetic(assessment),
        "Only actual recomputed formal prerequisites may be sealed",
    )
    return dict(
        schema=SCHEMA,
        status="PASS",
        scope=SCOPE,
        root=str(root),
        context=assessment["context"],
        request=assessment["request"],
        stable_facts=assessment["stable_facts"],
        assessment_ref=assessment_ref,
        inputs=[*assessment["inputs"], assessment_ref],
        external_inputs=assessment["external_inputs"],
        training_authorized=True,
        generation_authorized=True,
        dev36_authorized=True,
        formal_test_access=False,
        final_evaluation_authorized=False,
        gate_status="SIGNED_FROM_RECOMPUTED_FACTS",
        effectiveness_gate=None,
        limitation=(
            "Registered TRAIN and committed DEV_EVAL step36 only; runtime session locks "
            "required; TEST requires all54 step72 and separate grant"
        ),
    )


def _authorized(receipt, ref, assessment):
    return dict(
        status="VERIFIED_FORMAL_ADMISSION",
        scope=SCOPE,
        training_authorized=True,
        generation_authorized=True,
        dev36_authorized=True,
        formal_test_access=False,
        final_evaluation_authorized=False,
        admission_ref=ref,
        receipt=receipt,
        phase_context=assessment["phase_context"],
        formal_spec=assessment["formal_spec"],
        evaluation_spec=assessment["evaluation_spec"],
        formal_spec_ref=dict(
            path=f"{registration.DIRECTORY}/{registration.SPEC_FILE}",
            sha256=assessment["request"]["formal_spec_sha256"],
        ),
        evaluation_spec_ref=dict(
            path=f"{evaluation.DIRECTORY}/{evaluation.SPEC_FILE}",
            sha256=assessment["request"]["evaluation_spec_sha256"],
        ),
    )


def _baseline(evidence, receipt):
    original = evidence.read(receipt["assessment_ref"])
    _require(
        _receipt(evidence.root, original, receipt["assessment_ref"]) == receipt,
        "Saved permission differs from its original complete assessment",
    )
    with phases._ContentCache(evidence.root) as cache:
        for ref in receipt["inputs"]:
            _ref(ref)
            path = phases._file(evidence.root, ref["path"])
            _require(cache.hash(path) == ref["sha256"], "Sealed prerequisite changed")
    return original


def seal_formal_admission(root, **request):
    """Freeze a real phase-bound training/DEV36 receipt; preserve every failed prefix."""
    root = phases._root(root)
    request = _request(**request)
    with _lock(root):
        existing = None
        path = phases._path(root / PATH, exists=False)
        if path.exists():
            existing = history._json(path.read_bytes())
            _require(
                existing.get("request") == request, "Existing formal admission request differs"
            )
            _baseline(_Evidence(root), existing)
        assessment = _assess(root, request)
        assessment_ref = dict(
            path=f"{ASSESSMENTS}/{digest(assessment)}.json",
            sha256=hashlib.sha256(history._payload(assessment)).hexdigest(),
        )
        _publish(root / assessment_ref["path"], assessment)
        if assessment["status"] != "PASS":
            return dict(
                status=assessment["status"],
                assessment=assessment,
                assessment_ref=assessment_ref,
                training_authorized=False,
                generation_authorized=False,
                dev36_authorized=False,
                formal_test_access=False,
                final_evaluation_authorized=False,
                gate_status="NOT_SIGNED",
            )
        receipt = _receipt(root, assessment, assessment_ref) if existing is None else existing
        _require(
            receipt["stable_facts"] == assessment["stable_facts"],
            "New actual facts differ from sealed science",
        )
        _assert_verified(root, assessment)
        _publish(path, receipt)
        ref = history._ref(root, path)
        result = _authorized(receipt, ref, assessment)
        _remember(root, result, assessment)
        return copy.deepcopy(result)


def _key(root, phase, ref, train, evaluated):
    return (os.getpid(), str(root), phase, ref["sha256"], train, evaluated)


def _remember(root, result, assessment):
    _require(
        assessment["status"] == "PASS"
        and assessment["execution_origin"] == "ACTUAL_EVIDENCE_VALIDATION"
        and not _synthetic(assessment),
        "Only actual full verification may populate runtime cache",
    )
    proof = _assert_verified(root, assessment)
    refs = [*assessment["inputs"], *result["receipt"]["inputs"], result["admission_ref"]]
    pins = {str(phases._file(root, r["path"])): r["sha256"] for r in refs}
    for ref in [*assessment["external_inputs"], *result["receipt"]["external_inputs"]]:
        pins[ref["path"]] = ref["sha256"]
    trees, snapshot = copy.deepcopy(proof["trees"]), copy.deepcopy(proof["snapshot"])
    root_paths = set(proof["root_paths"])
    with phases._ContentCache(root) as cache:
        for name, sha in pins.items():
            if name in proof["pins"]:
                _require(proof["pins"][name] == sha, "Cache pin differs from verified input")
                continue
            # Only new small assessment/admission records can be absent from the
            # completed proof; their bytes are checked before recording stat.
            path = Path(name)
            _require(
                path.is_relative_to(root)
                and (name == str(root / PATH) or path.is_relative_to(root / ASSESSMENTS)),
                "Unverified late cache dependency",
            )
            phases._file(root, path.relative_to(root).as_posix())
            stamp = history._stamp(path)
            _require(
                cache.hash(path) == sha and history._stamp(path) == stamp,
                "Published receipt changed before cache installation",
            )
            snapshot[name] = stamp
            root_paths.add(name)
    _assert_verified(root, assessment)
    request = assessment["request"]
    _CACHE[
        _key(
            root,
            request["phase_identity_sha256"],
            result["admission_ref"],
            request["formal_spec_sha256"],
            request["evaluation_spec_sha256"],
        )
    ] = dict(
        result=copy.deepcopy(result),
        request=copy.deepcopy(request),
        context=copy.deepcopy(assessment["phase_context"]),
        deployments=assessment["deployments"],
        pins=pins,
        trees=trees,
        snapshot=snapshot,
        root_paths=root_paths,
    )


def _unchanged(root, cached):
    trees, current = _layout(root, cached["context"], cached["request"], cached["deployments"])
    if trees != cached["trees"]:
        return False
    for name in cached["snapshot"]:
        path = Path(name)
        if name in cached["root_paths"]:
            phases._file(root, path.relative_to(root).as_posix())
        value = current[name] if name in current else history._stamp(path)
        _require(value[0] == cached["snapshot"][name][0], "Verified prerequisite target changed")
        if value != cached["snapshot"][name]:
            if name not in cached["pins"]:
                return False
            _require(
                file_hash(path) == cached["pins"][name] and history._stamp(path) == value,
                "Admission dependency changed after actual verification",
            )
            cached["snapshot"][name] = value
    deployment = _actual_deployment()
    _require(
        str(deployment) == cached["context"]["deployment"],
        "Cached permission belongs to another source",
    )
    manifest = history._json(
        phases._file(root, cached["context"]["identity"]["source_manifest"]["path"]).read_bytes()
    )
    history._loaded(deployment, manifest)
    return True


def require_formal_admission(
    root, *, phase_identity_sha256, admission_ref, formal_spec_sha256, evaluation_spec_sha256
):
    """Cold original evidence replay, then immutable PID-local prerequisite reuse."""
    root = phases._root(root)
    _ref(admission_ref)
    _require(admission_ref["path"] == PATH, "Only fixed FORMAL admission receipt allowed")
    key = _key(
        root, phase_identity_sha256, admission_ref, formal_spec_sha256, evaluation_spec_sha256
    )
    cached = _CACHE.get(key)
    if cached is not None and _unchanged(root, cached):
        return copy.deepcopy(cached["result"])
    with _lock(root):
        evidence = _Evidence(root)
        receipt = evidence.read(admission_ref)
        request = receipt["request"]
        _require(
            receipt["schema"] == SCHEMA
            and request["phase_identity_sha256"] == phase_identity_sha256
            and request["formal_spec_sha256"] == formal_spec_sha256
            and request["evaluation_spec_sha256"] == evaluation_spec_sha256,
            "Formal receipt phase/plans pins differ",
        )
        _baseline(evidence, receipt)
        assessment = _assess(root, _request(**request))
        _require(
            assessment["status"] == "PASS"
            and assessment["stable_facts"] == receipt["stable_facts"],
            "Actual formal prerequisites do not match sealed science: " + assessment["status"],
        )
        result = _authorized(receipt, admission_ref, assessment)
        _remember(root, result, assessment)
        return copy.deepcopy(result)
