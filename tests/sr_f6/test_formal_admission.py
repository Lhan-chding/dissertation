"""Synthetic control fixtures, not actual formal admission or experiment evidence.

The historical transport uses a real subprocess with the explicitly synthetic
verifier from test_historical_pilot. Plan providers are controlled stubs (their
real construction is tested separately). Positive control tests suppress only
the fixture's synthetic source marker to exercise publication/cache mechanics;
they cannot be used as a scientific or server receipt.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from sr_f6 import formal_admission as admission
from sr_f6 import historical_pilot as history
from sr_f6 import pilot_audit, source_binding
from sr_f6.common import file_hash
from sr_f6.planning import formal_paths

_spec = importlib.util.spec_from_file_location(
    "formal_admission_transport_fixture", Path(__file__).with_name("test_historical_pilot.py")
)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


def write(root, relative, value):
    return _fixture.write(root, relative, value)


@pytest.fixture
def case(tmp_path, monkeypatch):
    value = _fixture.case.__wrapped__(tmp_path, monkeypatch)
    root = value.root
    admission._CACHE.clear()
    monkeypatch.setattr(admission, "_actual_deployment", lambda: value.current)
    source = admission._source
    value.synthetic = True

    def controlled_source(evidence, request):
        result = source(evidence, request)
        evidence.synthetic |= value.synthetic
        return result

    monkeypatch.setattr(admission, "_source", controlled_source)
    config = json.loads((root / "CONFIG.json").read_bytes())
    identity = value.consumer["identity"]
    order = formal_paths(config)
    paths = []
    for item in order:
        relative = f"runs/{item['parent']}/r{item['repeat']}/{item['arm']}"
        run_identity = dict(
            run_id=item["run_id"],
            phase="FORMAL_V1",
            parent=item["parent"],
            repeat=item["repeat"],
            arm=item["arm"],
        )
        identity_ref = write(
            root, f"{admission.registration.DIRECTORY}/{relative}/RUN_IDENTITY.json", run_identity
        )
        rollout_ref = write(
            root,
            f"{admission.registration.DIRECTORY}/{relative}/ROLLOUT_REGISTRATION.json",
            dict(control="SYNTHETIC"),
        )
        paths.append(
            dict(
                parent=item["parent"],
                repeat=item["repeat"],
                arm=item["arm"],
                run_id=item["run_id"],
                audit_root=f"{admission.registration.DIRECTORY}/{relative}",
                transaction_identity=dict(
                    path=f"{relative}/RUN_IDENTITY.json", sha256=identity_ref["sha256"]
                ),
                rollout_registration=dict(
                    path=f"{relative}/ROLLOUT_REGISTRATION.json", sha256=rollout_ref["sha256"]
                ),
            )
        )
    parent_refs = {
        p: write(root, f"runtime_identity/{p}.json", dict(parent=p, tensor_sha256=str(i + 1) * 64))
        for i, p in enumerate(("b0", "b1", "b2"))
    }
    producer = dict(
        phase="PILOT_V1",
        deployment=value.producer["deployment"],
        source_manifest=value.producer["identity"]["source_manifest"],
        phase_identity=dict(
            path="PHASES/PILOT_V1/IDENTITY.json", sha256=value.producer["identity_sha256"]
        ),
        specification=dict(path=history.SPEC_PATH, sha256=value.request["pilot_spec_sha256"]),
    )
    common = dict(
        phase_identity=dict(
            path="PHASES/FORMAL_V1/IDENTITY.json", sha256=value.consumer["identity_sha256"]
        ),
        actual_source_manifest=identity["source_manifest"],
        origin_identity=identity["origin"]["identity"],
        config_sha256=identity["config_sha256"],
        data_manifest=identity["data_manifest"],
        source_binding=history._ref(root, root / "SOURCE_BINDING.json"),
        parent_references=parent_refs,
        parent_tensors={
            p: dict(tensor_sha256=str(i + 1) * 64) for i, p in enumerate(("b0", "b1", "b2"))
        },
        order_source=write(root, "DATA/FORMAL_PATH_ORDER_CONTROL.json", order),
    )
    train = dict(**common, pilot_producer=producer, paths=paths)
    train_ref = write(
        root, f"{admission.registration.DIRECTORY}/{admission.registration.SPEC_FILE}", train
    )
    external = value.project / "external-control.json"
    external.write_text('{"control":"SYNTHETIC"}\n')
    protocol_ref = write(
        root,
        "PHASES/FORMAL_V1/RETENTION_PROTOCOL_BINDING.json",
        dict(
            original_receipt=dict(path=str(external), sha256=file_hash(external)),
            files=[],
            inherited_native_files=[],
        ),
    )
    selectors = [
        dict(
            stage=stage,
            kind="FUTURE_FORMAL_COMMIT_SELECTOR",
            parent=p["parent"],
            repeat=p["repeat"],
            arm=p["arm"],
            run_id=p["run_id"],
            expected_run_directory=p["audit_root"],
            checkpoint_step=step,
        )
        for stage, step in (("DEV36", 36), ("FINAL", 72))
        for p in paths
    ]
    selector_path = root / admission.evaluation.DIRECTORY / "ENDPOINT_SELECTORS.jsonl"
    selector_path.parent.mkdir(parents=True)
    selector_path.write_bytes(b"".join(history._payload(p) for p in selectors))
    selector_ref = dict(path="ENDPOINT_SELECTORS.jsonl", sha256=file_hash(selector_path))
    plan = dict(
        **common,
        pilot_parent_provenance=producer,
        endpoint_selectors=selector_ref,
        retention_protocol_binding=protocol_ref,
        access=dict(training_grant_implies_test_access=False),
    )
    eval_ref = write(
        root, f"{admission.evaluation.DIRECTORY}/{admission.evaluation.SPEC_FILE}", plan
    )
    frozen = [history._ref(root, selector_path), eval_ref]
    for index in range(9):
        frozen.append(
            write(root, f"{admission.evaluation.DIRECTORY}/control-{index}.json", dict(index=index))
        )
    train_result = dict(status="VERIFIED_REGISTRATION_NOT_AUTHORIZED", spec=train)
    eval_result = dict(status="VERIFIED_PLAN_NOT_AUTHORIZED", spec=plan, frozen_files=frozen)
    monkeypatch.setattr(
        admission.registration,
        "verify_formal_registration",
        lambda *args, **kwargs: copy.deepcopy(train_result),
    )
    monkeypatch.setattr(
        admission.evaluation,
        "verify_formal_evaluation_plan",
        lambda *args, **kwargs: copy.deepcopy(eval_result),
    )
    # The frozen bridge fixture's admission lists this extra external input;
    # keep its pin by using a separate protocol dependency above instead.
    value.external = external
    value.train, value.plan = train_result, eval_result
    value.control_request = dict(
        phase_identity_sha256=value.request["formal_phase_identity_sha256"],
        formal_spec_sha256=train_ref["sha256"],
        evaluation_spec_sha256=eval_ref["sha256"],
        pilot_phase_identity_sha256=value.request["pilot_phase_identity_sha256"],
        pilot_spec_sha256=value.request["pilot_spec_sha256"],
        pilot_admission_ref=value.request["admission_ref"],
        recovery_binding_ref=value.request["recovery_binding_ref"],
        recovery_comparison_ref=value.request["recovery_comparison_ref"],
        evaluation_summary_ref=value.request["evaluation_summary_ref"],
    )
    runs = []
    families = ["BINDING", "IMPACT", "TIES", "SIGNED", "INTERVAL", "K_DIRECTION"]
    for parent in ("b0", "b1", "b2"):
        for arm in ("U10", "U11", "C_SHUF"):
            groups = [
                dict(
                    step=step,
                    qid=f"{parent}-{step}-{g}",
                    root_id=f"root-{step}-{g // 4}",
                    family=families[(step + g // 4) % 6],
                    nonzero=arm == "U11",
                    exchangeable=False,
                    changed=False,
                    source_step=step - 1,
                    commit_sha256=f"{step:064x}",
                    original_reference=dict(path=f"raw/{step}/{g}", sha256="a" * 64),
                )
                for step in range(1, 13)
                for g in range(16)
            ]
            runs.append(
                dict(
                    parent=parent,
                    arm=arm,
                    scope="SERVER_ACTUAL",
                    updates=12,
                    responses=1536,
                    groups=groups,
                    steps=list(range(1, 13)),
                )
            )
    value.facts.update(
        runs=runs,
        exposure=pilot_audit._exposure(runs),
        recovery=dict(
            status="PASS",
            scope="SERVER_ACTUAL",
            exact=True,
            independent_process_verified=True,
            new_generated_responses=1024,
        ),
        recovery_comparison_history=dict(status="ALL_SAVED_FACTS_VERIFIED"),
        evaluation=dict(
            status="COMPLETE_DESCRIPTIVE_NOT_GATE_SIGNED", checkpoints=12, responses=2304
        ),
    )
    write(root, "CONTROL_FACTS.json", value.facts)
    return value


@pytest.fixture
def positive_control(case):
    """White-box publication/cache control only, not authentic source evidence."""
    case.synthetic = False
    return case


def seal(case):
    return admission.seal_formal_admission(case.root, **case.control_request)


def require(case, result):
    return admission.require_formal_admission(
        case.root,
        phase_identity_sha256=case.control_request["phase_identity_sha256"],
        admission_ref=result["admission_ref"],
        formal_spec_sha256=case.control_request["formal_spec_sha256"],
        evaluation_spec_sha256=case.control_request["evaluation_spec_sha256"],
    )


def calls(case):
    path = case.root / "CONTROL_CALLS.jsonl"
    return len(path.read_text().splitlines()) if path.exists() else 0


def test_synthetic_all_apparent_pass_can_never_seal(case):
    result = seal(case)
    assert result["status"] == "SYNTHETIC_ONLY"
    assert result["assessment"]["scientific_status"] == "PASS"
    assert not result["training_authorized"] and not (case.root / admission.PATH).exists()
    assert admission._CACHE == {}


def test_control_seal_cold_recheck_and_warm_immutable_no_test_grant(positive_control):
    case = positive_control
    result = seal(case)
    assert result["status"] == "VERIFIED_FORMAL_ADMISSION"
    assert result["training_authorized"] and result["dev36_authorized"]
    assert not result["formal_test_access"] and not result["final_evaluation_authorized"]
    before = (case.root / admission.PATH).read_bytes()
    assert calls(case) == 1
    result["formal_spec"]["paths"].clear()
    warm = require(case, result)
    assert len(warm["formal_spec"]["paths"]) == 54 and calls(case) == 1
    admission._CACHE.clear()
    cold = require(case, warm)
    assert calls(case) == 2 and not cold["formal_test_access"]
    assert (case.root / admission.PATH).read_bytes() == before
    assert len(list((case.root / history.DIRECTORY).glob("*/RESULT.json"))) == 2


@pytest.mark.parametrize(
    "status",
    [
        "INSUFFICIENT_REGISTERED_EXPOSURE",
        "RECOVERY_MISMATCH",
        "RECOVERY_COMPARISON_HISTORY_UNKNOWN",
    ],
)
def test_negative_or_unknown_original_facts_block_without_effect_gate(positive_control, status):
    case = positive_control
    case.facts["status"] = status
    if status == "INSUFFICIENT_REGISTERED_EXPOSURE":
        for run in case.facts["runs"]:
            for group in run["groups"]:
                group["nonzero"] = False
        case.facts["exposure"] = pilot_audit._exposure(case.facts["runs"])
    write(case.root, "CONTROL_FACTS.json", case.facts)
    result = seal(case)
    assert result["status"] != "PASS" and not result["training_authorized"]
    assert not (case.root / admission.PATH).exists()
    assert (case.root / result["assessment_ref"]["path"]).is_file()


@pytest.mark.parametrize(
    "key",
    [
        "origin_identity",
        "data_manifest",
        "parent_tensors",
        "order_source",
        "pilot_parent_provenance",
    ],
)
def test_two_plans_cannot_mix_identity(positive_control, key):
    case = positive_control
    case.plan["spec"][key] = {"wrong": True}
    result = seal(case)
    assert result["status"] == "FAIL" and calls(case) == 0
    assert not result["training_authorized"]


def test_phase_source_confusion_rejected_before_child(positive_control, monkeypatch):
    case = positive_control
    monkeypatch.setattr(admission, "_actual_deployment", lambda: case.pilot)
    result = seal(case)
    assert result["status"] == "FAIL" and calls(case) == 0


@pytest.mark.parametrize("kind", ["orphan", "failure", "partial_result"])
def test_added_historical_attempt_invalidates_warm_and_blocks_without_new_child(
    positive_control, kind
):
    case = positive_control
    result = seal(case)
    directory = case.root / history.DIRECTORY / ("f" * 32)
    directory.mkdir()
    if kind != "orphan":
        write(directory, "REQUEST.json", {"status": "PASS"})
        (directory / ("FAILURE.json" if kind == "failure" else "RESULT.json")).write_text(
            '{"status":'
        )
    with pytest.raises(PermissionError):
        require(case, result)
    assert calls(case) == 1 and directory.exists()


@pytest.mark.parametrize("kind", ["delta", "run_identity", "eval_payload", "external"])
def test_changed_prerequisite_invalidates_warm_cache(positive_control, kind):
    case = positive_control
    result = seal(case)
    path = (
        case.root / "PHASES/FORMAL_V1/DELTA.json"
        if kind == "delta"
        else case.root / case.train["spec"]["paths"][0]["audit_root"] / "RUN_IDENTITY.json"
        if kind == "run_identity"
        else case.root / admission.evaluation.DIRECTORY / "control-0.json"
        if kind == "eval_payload"
        else case.external
    )
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(PermissionError):
        require(case, result)
    assert calls(case) == 1


def test_same_bytes_symlink_cannot_change_cached_boundary(positive_control):
    case = positive_control
    result = seal(case)
    path = case.root / case.train["spec"]["paths"][0]["audit_root"] / "RUN_IDENTITY.json"
    other = case.project / "same-content.json"
    other.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(other)
    with pytest.raises(PermissionError):
        require(case, result)


def test_plan_changed_during_historical_child_cannot_be_cached(positive_control, monkeypatch):
    case = positive_control
    original = history.verify_historical_pilot
    path = case.root / case.train["spec"]["paths"][0]["audit_root"] / "RUN_IDENTITY.json"

    def changed(*args, **kwargs):
        value = original(*args, **kwargs)
        path.write_bytes(path.read_bytes() + b" ")
        return value

    monkeypatch.setattr(history, "verify_historical_pilot", changed)
    result = seal(case)
    assert result["status"] == "FAIL" and not result["training_authorized"]
    assert admission._CACHE == {} and not (case.root / admission.PATH).exists()


def test_change_between_assessment_and_cache_install_is_rejected(positive_control, monkeypatch):
    case = positive_control
    original = admission._remember
    path = case.external

    def changed(*args):
        path.write_bytes(path.read_bytes() + b" ")
        return original(*args)

    monkeypatch.setattr(admission, "_remember", changed)
    with pytest.raises(PermissionError, match="before cache installation"):
        seal(case)
    assert admission._CACHE == {}


def test_external_changed_before_initial_proof_cannot_acquire_verified_stat(
    positive_control, monkeypatch
):
    case = positive_control
    original = admission._dependency_inputs

    def changed(*args):
        value = original(*args)
        case.external.write_bytes(case.external.read_bytes() + b" ")
        return value

    monkeypatch.setattr(admission, "_dependency_inputs", changed)
    result = seal(case)
    assert result["status"] == "FAIL" and not result["training_authorized"]
    assert admission._CACHE == {} and calls(case) == 0
    assert not (case.root / admission.PATH).exists()


def test_no_per_step_original_cold_replay_or_unchanged_image_hash(positive_control, monkeypatch):
    case = positive_control
    result = seal(case)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unchanged warm boundary must not hash/replay all evidence")

    monkeypatch.setattr(admission, "file_hash", forbidden)
    monkeypatch.setattr(history, "verify_historical_pilot", forbidden)
    for _ in range(2):
        assert require(case, result)["dev36_authorized"] is True
    assert calls(case) == 1


def test_original_bound_metadata_avoids_reading_large_source_again(positive_control, monkeypatch):
    case = positive_control
    bound = source_binding.inventory_file(case.external, boundary=case.project)
    original = admission._dependency_inputs

    def with_bound(evidence, *args):
        result = original(evidence, *args)
        evidence.bound_files[str(case.external)] = bound
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("Unchanged original inventory must not rehash its payload")

    original_hash = admission.phases.file_hash

    def cached_hash(path):
        if Path(path) == case.external:
            forbidden()
        return original_hash(path)

    monkeypatch.setattr(admission, "_dependency_inputs", with_bound)
    monkeypatch.setattr(source_binding, "file_sha256", forbidden)
    monkeypatch.setattr(admission.phases, "file_hash", cached_hash)
    assert seal(case)["training_authorized"] is True


def test_formal_training_outputs_are_not_their_own_admission_inputs(positive_control):
    case = positive_control
    result = seal(case)
    path = (
        case.root
        / case.train["spec"]["paths"][0]["audit_root"]
        / "checkpoints/step-000/CONTROL.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    assert require(case, result)["training_authorized"] is True and calls(case) == 1


def test_handwritten_permission_and_partial_admission_are_not_overwritten(positive_control):
    case = positive_control
    path = case.root / admission.PATH
    path.write_text('{"status":"PASS"')
    original = path.read_bytes()
    with pytest.raises((PermissionError, ValueError)):
        seal(case)
    assert path.read_bytes() == original and calls(case) == 0


def test_rename_failure_keeps_partial_publication_and_syncs_success(positive_control, monkeypatch):
    case = positive_control
    original = admission.os.rename

    def failed(source, destination):
        if Path(destination).name == "ADMISSION.json":
            raise OSError("controlled publication interruption")
        original(source, destination)

    monkeypatch.setattr(admission.os, "rename", failed)
    with pytest.raises(OSError, match="publication interruption"):
        seal(case)
    assert not (case.root / admission.PATH).exists() and admission._CACHE == {}
    assert list((case.root / "PHASES/FORMAL_V1").glob(".ADMISSION.json.*.tmp"))
    monkeypatch.setattr(admission.os, "rename", original)
    synced = []
    sync = admission._fsync_dir
    monkeypatch.setattr(
        admission, "_fsync_dir", lambda path: (synced.append(Path(path)), sync(path))
    )
    assert seal(case)["training_authorized"] is True
    assert case.root / "PHASES/FORMAL_V1" in synced
