"""Synthetic source-boundary tests; no fixture grants a real scientific PASS."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from sr_f6 import confirm_analysis, confirm_evidence
from sr_f6_followup import source


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(source._payload(value))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = tmp_path.resolve()
    root = project / "followup"
    deployment = project / "sr_f6_followup_code_abcdef1"
    producer_root, producer_code = project / "producer", project / "producer_code"
    for name, value in (
        ("PROJECT", project),
        ("ROOT", root),
        ("PRODUCER_ROOT", producer_root),
        ("PRODUCER_DEPLOYMENT", producer_code),
    ):
        monkeypatch.setattr(source, name, value)
    for name in ("__init__.py", "source.py"):
        path = deployment / "src/sr_f6_followup" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"""SYNTHETIC_ONLY source."""\n')
    monkeypatch.setattr(source, "_actual_deployment", lambda: deployment)
    monkeypatch.setattr(source, "_verify_module_origins", lambda *args, **kwargs: None)
    producer_root.mkdir()
    producer_code.mkdir()
    qid = "DEV_SCHEMA_CONFIRM_SYNTHETIC_0000_P0_v0"
    selection = dict(
        qid=qid,
        parents=["b0", "b1", "b2"],
        draws=list(range(8)),
        sequences=24,
        new_generated_responses=0,
        optimizer_updates=0,
        authorization="SEPARATE_ACTUAL_NUMERICAL_ADMISSION_REQUIRED",
    )
    requirements = dict(
        independent_response_review=dict(
            count=360,
            seed=2026102701,
            new_original_answers_only=True,
            authorization="SEPARATE_INDEPENDENT_REVIEW_REQUIRED",
        ),
        numerical_replay_selection=selection,
    )
    spec = dict(
        gates=dict(review_seed=2026102701, new_independent_response_review=360),
        followup_requirements=requirements,
    )
    context = dict(
        root=producer_root,
        deployment=producer_code,
        registration_sha256=source.PRODUCER_REGISTRATION_SHA256,
        registration=dict(
            revision=source.PRODUCER_REVISION, source_manifest_sha256="a" * 64, data_sha256="b" * 64
        ),
        spec=spec,
        spec_sha256="c" * 64,
        spec_object_sha256=source._digest(spec),
        tasks={qid: dict(qid=qid)},
        public=[dict(qid=qid)],
        config={},
    )
    calls = []
    monkeypatch.setattr(
        source, "_producer_context", lambda: calls.append("producer") or copy.deepcopy(context)
    )
    source._SUPPORT_CACHE.clear()
    source._SOURCE_VERIFIED.clear()
    return SimpleNamespace(
        project=project,
        root=root,
        deployment=deployment,
        producer_root=producer_root,
        producer_code=producer_code,
        context=context,
        calls=calls,
        monkeypatch=monkeypatch,
    )


def registered(setup):
    return source.register(setup.root, "abcdef1" + "0" * 33)


def save_analysis(setup, *, status=None):
    """Install explicitly mocked frozen validators, never a production fixture."""
    root = setup.producer_root
    raw_file = root / "CONFIRM/raw/SYNTHETIC.json"
    model = setup.project / "model.bin"
    write(raw_file, {"fixture": "SYNTHETIC_ONLY"})
    model.write_bytes(b"SYNTHETIC_ONLY_MODEL")
    evidence = dict(
        status="VERIFIED_STORED_CONFIRMATION_NOT_GATE_SIGNED",
        registration_sha256=source.PRODUCER_REGISTRATION_SHA256,
        source_rows=9216,
        actual_recorded_calls=9216,
        records=[dict(row_id=f"synthetic-{i}") for i in range(9216)],
        raw_by_row={f"synthetic-{i}": dict(text="SYNTHETIC_ONLY") for i in range(9216)},
        saved_token_validation=dict(status="PASS", checked_responses=9216),
        input_producer=dict(deployment=str(setup.producer_code), revision=source.PRODUCER_REVISION),
        consumer_source=dict(kind="SYNTHETIC_FROZEN_VALIDATOR"),
        inputs=[dict(path=str(raw_file.relative_to(root)), sha256=source._hash(raw_file))],
        external_inputs=[dict(path=str(model), sha256=source._hash(model))],
    )
    validation = {k: v for k, v in evidence.items() if k not in ("records", "raw_by_row")}
    files = {"VALIDATION.json": validation, "SUPPORT.json": dict(status="PASS")}
    report = dict(
        schema=confirm_analysis.SCHEMA,
        status=status or source.SUPPORT_STATUS,
        support_status="PASS",
        registration_sha256=source.PRODUCER_REGISTRATION_SHA256,
    )
    directory = root / source.ANALYSIS
    refs = {}
    for name, value in files.items():
        write(directory / name, value)
        refs[name] = dict(path=f"{source.ANALYSIS}/{name}", sha256=source._hash(directory / name))
    saved = dict(
        report,
        training_allowed=False,
        generation_allowed=False,
        evaluation_allowed=False,
        gate_status="NOT_SIGNED",
        new_model_calls=0,
        optimizer_updates=0,
        reports=refs,
    )
    write(directory / "ASSESSMENT.json", saved)
    write(
        directory / "SEAL.json",
        dict(
            schema="srf6-confirm-analysis-seal-v1",
            status=source.SUPPORT_STATUS,
            assessment=dict(
                path=f"{source.ANALYSIS}/ASSESSMENT.json",
                sha256=source._hash(directory / "ASSESSMENT.json"),
            ),
            registration_sha256=source.PRODUCER_REGISTRATION_SHA256,
            gate_status="NOT_SIGNED",
            training_allowed=False,
            generation_allowed=False,
        ),
    )
    scans = []
    setup.monkeypatch.setattr(
        confirm_evidence,
        "validate_saved_confirm",
        lambda root: scans.append("full") or copy.deepcopy(evidence),
    )
    setup.monkeypatch.setattr(
        confirm_evidence, "_snapshot", lambda root: {"raw": {"CONFIRM/raw/SYNTHETIC.json"}}
    )
    setup.monkeypatch.setattr(
        confirm_analysis,
        "_analyze",
        lambda root, value: (copy.deepcopy(report), copy.deepcopy(files)),
    )
    return SimpleNamespace(
        evidence=evidence, report=report, files=files, scans=scans, raw_file=raw_file, model=model
    )


def test_registration_freezes_distinct_consumer_and_original_contract(setup):
    result = registered(setup)
    assert result["consumer"]["revision"] != source.PRODUCER_REVISION
    assert result["registration"]["producer"]["revision"] == source.PRODUCER_REVISION
    assert result["spec"]["requirements"] == setup.context["spec"]["followup_requirements"]
    assert result["spec"]["training_allowed"] is False
    assert result["registration"]["status"] == "BOUND_NOT_AUTHORIZED"
    assert result["producer_context"]["root"] == setup.producer_root
    with pytest.raises(PermissionError):
        registered(setup)
    with pytest.raises(FileNotFoundError):
        source.require_support(setup.root, "review")


@pytest.mark.parametrize("external", [False, True])
def test_registration_symlink_rejected_before_producer_verification(setup, external):
    registered(setup)
    path = setup.root / "REGISTRATION.json"
    destination = (
        setup.project.parent / (setup.project.name + "-external-registration.json")
        if external
        else setup.project / "copied-registration.json"
    )
    destination.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(destination)
    setup.calls.clear()
    with pytest.raises(PermissionError, match="Symlinked"):
        source.verify_registered(setup.root, require_support=False)
    assert setup.calls == []


def test_actual_launchers_are_bound_and_later_added_source_is_rejected(setup):
    launcher = setup.deployment / "scripts/sr_f6_followup/t03.sbatch"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/bin/bash\n# SYNTHETIC_ONLY launcher\n")
    context = registered(setup)
    assert str(launcher) in {row["path"] for row in context["source_manifest"]["files"]}
    launcher.write_text("#!/bin/bash\n# changed launcher\n")
    with pytest.raises(PermissionError, match="source changed"):
        source.verify_registered(setup.root)


def test_consumer_cannot_deploy_shadow_producer_package(setup):
    (setup.deployment / "src/sr_f6").mkdir()
    with pytest.raises(PermissionError, match="shadow"):
        registered(setup)


def test_source_content_cache_only_reuses_unchanged_registered_files(setup):
    registered(setup)
    calls = []
    original = source._hash
    setup.monkeypatch.setattr(
        source, "_hash", lambda path: calls.append(str(path)) or original(path)
    )
    source.verify_registered(setup.root)
    assert not any(name.endswith("source.py") for name in calls)
    path = setup.deployment / "src/sr_f6_followup/source.py"
    path.write_text("changed")
    with pytest.raises(PermissionError, match="source changed"):
        source.verify_registered(setup.root)


@pytest.mark.parametrize("mutation", ["seed", "root", "permission", "source", "extra_source"])
def test_changed_registration_or_contract_rejected(setup, mutation):
    registered(setup)
    reg = setup.root / "REGISTRATION.json"
    value = json.loads(reg.read_text())
    if mutation == "root":
        value["root"] = str(setup.project)
    elif mutation == "permission":
        value["training_allowed"] = True
    elif mutation == "source":
        (setup.deployment / "src/sr_f6_followup/source.py").write_text("changed\n")
    elif mutation == "extra_source":
        (setup.deployment / "src/sr_f6_followup/new.py").write_text("changed\n")
    else:
        specfile = setup.root / "SPEC.json"
        spec = json.loads(specfile.read_text())
        spec["requirements"]["independent_response_review"]["seed"] += 1
        write(specfile, spec)
        value["spec"]["sha256"] = source._hash(specfile)
    write(reg, value)
    with pytest.raises(PermissionError):
        source.verify_registered(setup.root)


def test_path_and_alias_boundaries(setup):
    with pytest.raises(PermissionError):
        source._root(setup.project)
    with pytest.raises(PermissionError):
        source._path(setup.project.parent)
    target = setup.project / "target"
    target.write_text("bound")
    alias = setup.project / "alias"
    alias.symlink_to(target)
    observed = source._state(alias)
    assert observed["resolved_path"] == str(target) and observed["aliases"]
    with pytest.raises(PermissionError):
        source._path(alias)
    outside = setup.project.parent / (setup.project.name + "-outside")
    outside.write_text("outside")
    alias.unlink()
    alias.symlink_to(outside)
    with pytest.raises(PermissionError):
        source._state(alias)


def test_support_requires_full_recomputation_and_warm_cache_does_not_hash_models(setup):
    registered(setup)
    fixture = save_analysis(setup)
    hash_calls = []
    real = source._hash
    setup.monkeypatch.setattr(
        source, "_hash", lambda path: hash_calls.append(Path(path)) or real(path)
    )
    result = source.bind_support(setup.root)
    assert fixture.scans == ["full"]
    assert fixture.model not in hash_calls
    assert result["consumer"] != result["evidence"]["consumer_source"]
    assert result["training_allowed"] is result["generation_allowed"] is False
    result["records"][0]["row_id"] = "caller_mutation"
    again = source.require_support(setup.root, "review")
    assert again["records"][0]["row_id"] == "synthetic-0"
    assert fixture.scans == ["full"] and fixture.model not in hash_calls
    source._SUPPORT_CACHE.clear()
    source.require_support(setup.root, "review")
    assert fixture.scans == ["full", "full"] and fixture.model not in hash_calls


def test_unchanged_support_binding_is_not_read_or_hashed_in_warm_calls(setup):
    registered(setup)
    fixture = save_analysis(setup)
    source.bind_support(setup.root)
    binding = setup.root / "SUPPORT_BINDING.json"
    reads, hashes = [], []
    original_read, original_hash = Path.read_bytes, source._hash

    def read_bytes(path):
        reads.append(path)
        return original_read(path)

    def file_hash(path):
        hashes.append(Path(path))
        return original_hash(path)

    setup.monkeypatch.setattr(Path, "read_bytes", read_bytes)
    setup.monkeypatch.setattr(source, "_hash", file_hash)
    for _ in range(3):
        source.require_support(setup.root, "review")
    assert binding not in reads and binding not in hashes
    assert fixture.scans == ["full"]
    source._SUPPORT_CACHE.clear()
    source.require_support(setup.root, "review")
    assert reads.count(binding) == 1 and binding not in hashes
    assert fixture.scans == ["full", "full"]


@pytest.mark.parametrize(
    "name", ["SUPPORT_BINDING.json", "SUPPORT_BINDING.attempt.json", "SUPPORT_BINDING.failure.json"]
)
def test_dangling_support_evidence_is_never_treated_as_missing(setup, name):
    registered(setup)
    save_analysis(setup)
    (setup.root / name).symlink_to(setup.project / "absent")
    with pytest.raises(PermissionError, match="prior or partial"):
        source.bind_support(setup.root)


def test_dangling_failure_blocks_warm_support_cache(setup):
    registered(setup)
    save_analysis(setup)
    source.bind_support(setup.root)
    (setup.root / "SUPPORT_BINDING.failure.json").symlink_to(setup.project / "absent")
    with pytest.raises(PermissionError, match="Failed support"):
        source.require_support(setup.root, "review")


@pytest.mark.parametrize("kind", ["model", "raw", "alias", "membership", "binding"])
def test_warm_cache_rejects_all_changes_before_reusing_permission(setup, kind):
    registered(setup)
    fixture = save_analysis(setup)
    source.bind_support(setup.root)
    if kind in ("model", "raw"):
        target = fixture.model if kind == "model" else fixture.raw_file
        target.write_text("changed")
    elif kind == "alias":
        target = setup.project / "new-model"
        target.write_bytes(fixture.model.read_bytes())
        fixture.model.unlink()
        fixture.model.symlink_to(target)
    elif kind == "membership":
        setup.monkeypatch.setattr(confirm_evidence, "_snapshot", lambda root: {"raw": {"extra"}})
    else:
        path = setup.root / "SUPPORT_BINDING.json"
        binding = json.loads(path.read_text())
        binding["training_allowed"] = True
        write(path, binding)
    with pytest.raises(PermissionError):
        source.require_support(setup.root, "review")
    assert fixture.scans == ["full"]


@pytest.mark.parametrize("kind", ["negative", "unknown", "false_pass", "partial", "bad_seal"])
def test_support_failure_never_binds_and_attempt_is_preserved(setup, kind):
    registered(setup)
    fixture = save_analysis(setup, status="CONFIRM_FAIL_STOP" if kind == "negative" else None)
    if kind == "unknown":
        fixture.report["support_status"] = "UNKNOWN"
    elif kind == "false_pass":
        fixture.files["SUPPORT.json"]["status"] = "FAIL"
    elif kind == "partial":
        fixture.evidence["records"].pop()
    elif kind == "bad_seal":
        path = setup.producer_root / source.ANALYSIS / "SEAL.json"
        seal = json.loads(path.read_text())
        seal["generation_allowed"] = True
        write(path, seal)
    with pytest.raises(PermissionError):
        source.bind_support(setup.root)
    assert not (setup.root / "SUPPORT_BINDING.json").exists()
    assert (setup.root / "SUPPORT_BINDING.failure.json").exists()
    with pytest.raises(PermissionError):
        source.bind_support(setup.root)


def test_cold_process_does_not_trust_self_consistent_fake_binding(setup):
    registered(setup)
    fixture = save_analysis(setup)
    source.bind_support(setup.root)
    source._SUPPORT_CACHE.clear()
    path = setup.root / "SUPPORT_BINDING.json"
    binding = json.loads(path.read_text())
    binding["records_sha256"] = "f" * 64
    write(path, binding)
    with pytest.raises(PermissionError, match="cold reconstruction"):
        source.require_support(setup.root, "review")
    assert len(fixture.scans) == 2


def test_t03_requires_verified_new_review_not_caller_pass(setup):
    registered(setup)
    save_analysis(setup)
    bound = source.bind_support(setup.root)
    module = ModuleType("sr_f6_followup.review_audit")
    review = dict(
        status="REVIEW_COMPLETE_NOT_GATE_SIGNED",
        completed=True,
        reviewed_original_answers=360,
        independently_reviewed_original_answers=360,
        consumer=bound["consumer"],
        input_producer=bound["evidence"]["input_producer"],
        training_allowed=False,
        gate_status="NOT_SIGNED",
        synthetic=False,
    )
    module.verify_completed = lambda root: copy.deepcopy(review)
    setup.monkeypatch.setitem(sys.modules, module.__name__, module)
    assert source.require_support(setup.root, "t03")["review"] == review
    review["independently_reviewed_original_answers"] = 359
    with pytest.raises(PermissionError, match="360-answer"):
        source.require_support(setup.root, "admission")
    with pytest.raises(PermissionError, match="purpose"):
        source.require_support(setup.root, "training")


def test_actual_module_check_includes_main_and_package_path(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "PROJECT", tmp_path.resolve())
    deployment = tmp_path.resolve() / "consumer"
    package = deployment / "src/sr_f6_followup"
    package.mkdir(parents=True)
    path = package / "entry.py"
    path.write_text("# actual consumer\n")
    module = ModuleType("__main__")
    module.__file__ = str(path)
    module.__spec__ = SimpleNamespace(name="sr_f6_followup.entry")
    refs = [dict(path=str(path))]
    source._verify_module_origins(deployment, refs, [module])
    module.__spec__ = None  # direct script still has an actual consumer file
    source._verify_module_origins(deployment, refs, [module])
    module.__spec__ = SimpleNamespace(name="sr_f6_followup.entry")
    outside = tmp_path / "other.py"
    outside.write_text("# other\n")
    module.__file__ = str(outside)
    with pytest.raises(PermissionError, match="outside its manifest"):
        source._verify_module_origins(deployment, refs, [module])
    module.__file__ = str(path)
    module.__path__ = [str(tmp_path)]
    with pytest.raises(PermissionError, match="search path"):
        source._verify_module_origins(deployment, refs, [module])


def test_real_two_source_imports_keep_producer_and_consumer_distinct(tmp_path):
    """A real isolated Python process, with small marked producer stand-ins."""
    project = tmp_path.resolve()
    frozen = project / "frozen/src/sr_f6"
    consumer = project / "sr_f6_followup_code_abcdef1/src/sr_f6_followup"
    frozen.mkdir(parents=True)
    consumer.mkdir(parents=True)
    (frozen / "__init__.py").write_text('"""SYNTHETIC_ONLY frozen origin."""\n')
    (frozen / "confirm_registration.py").write_text("SYNTHETIC_ONLY = True\n")
    (consumer / "__init__.py").write_text('"""SYNTHETIC_ONLY consumer."""\n')
    (consumer / "source.py").write_bytes(Path(source.__file__).read_bytes())
    code = """
import sys
from pathlib import Path
base = Path(sys.argv[1])
sys.path[:0] = [str(base/'frozen/src'), str(base/'sr_f6_followup_code_abcdef1/src')]
from sr_f6 import confirm_registration
from sr_f6_followup import source
source.PROJECT = base
deployment = base/'sr_f6_followup_code_abcdef1'
files = [dict(path=str(p)) for p in (deployment/'src').rglob('*.py')]
source._verify_module_origins(deployment, files)
assert Path(confirm_registration.__file__).is_relative_to(base/'frozen')
assert source._actual_deployment() == deployment
source.PRODUCER_DEPLOYMENT = base/'other'
try:
    source._producer_context()
except PermissionError as error:
    assert 'actual frozen' in str(error)
else:
    raise AssertionError('wrong producer accepted')
print('ACTUAL_DISTINCT_IMPORTS_ONLY_NO_SCIENTIFIC_PASS')
"""
    process = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(project)],
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    assert process.stdout.strip() == "ACTUAL_DISTINCT_IMPORTS_ONLY_NO_SCIENTIFIC_PASS"
