"""Provenance fixtures only; no fixture authorizes a scientific stage."""

import copy
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f6 import phases
from sr_f6.common import digest, file_hash


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(phases._payload(value))


def file_record(base, relative):
    path = base / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": file_hash(path)}


@pytest.fixture
def project(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(phases, "PROJECT", project)
    root = project / phases.ROOT.name
    root.mkdir()
    old_revision, new_revision = "a" * 40, "b" * 40
    old = project / f"sr_f6_code_{old_revision[:7]}"
    new = project / f"sr_f6_code_{new_revision[:7]}"
    files = {
        "src/library.py": "VALUE = 1\n",
        "src/sr_f6/runtime.py": "PROTOCOL = 'structured_v4'\n",
        "tests/sr_f6/test_native.py": "def test_fixture(): pass\n",
        "scripts/sr_f6/census.sbatch": "#!/bin/bash\n",
    }
    for relative, content in files.items():
        path = old / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    shutil.copytree(old, new)
    (new / "src/sr_f6/phases.py").write_text("VERSION = 1\n")
    original_config = Path(__file__).parents[2] / "configs/sr_f6.json"
    (root / "CONFIG.json").write_bytes(original_config.read_bytes())
    write(root / "IMPLEMENTATION_REGISTRATION.json", phases.AMENDMENTS)
    data_path = root / "DATA/DEV_CAL/INPUT.json"
    write(data_path, {"qid": "FIXTURE_ONLY"})
    data = {"files": [file_record(root, "DATA/DEV_CAL/INPUT.json")]}
    write(root / "DATA/MANIFEST.json", data)
    original_manifest = {"files": [file_record(old, name) for name in sorted(files)]}
    write(root / "SOURCE_MANIFEST.json", original_manifest)
    identity = {
        "config_sha256": file_hash(root / "CONFIG.json"),
        "implementation_digest": digest(phases.AMENDMENTS),
        "revision": old_revision,
        "source_manifest": {
            "path": "SOURCE_MANIFEST.json",
            "sha256": file_hash(root / "SOURCE_MANIFEST.json"),
        },
        "data_manifest": {
            "path": "DATA/MANIFEST.json",
            "sha256": file_hash(root / "DATA/MANIFEST.json"),
        },
    }
    write(root / "IDENTITY.json", identity)
    # A non-PASS upstream must bind without ever allowing training.
    write(root / "EVIDENCE/CENSUS.json", {"status": "NOT_RUN", "context": identity})
    upstream = [
        {
            "producer_phase": phases.ORIGIN,
            "identity_sha256": file_hash(root / "IDENTITY.json"),
            "evidence": [
                {"path": "EVIDENCE/CENSUS.json", "sha256": file_hash(root / "EVIDENCE/CENSUS.json")}
            ],
        }
    ]
    current = phases.build_manifest(new, revision=new_revision)
    delta = phases.source_delta(
        original_manifest,
        current,
        reasons={"src/sr_f6/phases.py": "Add explicit stage provenance before its use"},
    )
    arguments = dict(
        root=root,
        phase="PILOT_V1",
        deployment=new,
        revision=new_revision,
        origin_deployment=old,
        upstream=upstream,
        expected_delta=delta,
    )
    return SimpleNamespace(
        root=root,
        old=old,
        new=new,
        identity=identity,
        original_manifest=original_manifest,
        current=current,
        delta=delta,
        arguments=arguments,
        upstream=upstream,
        revision=new_revision,
    )


def register(project, **changes):
    return phases.register_phase(**{**project.arguments, **changes})


def verify(project, phase="PILOT_V1", **changes):
    return phases.verify_phase(project.root, phase, deployment=project.new, **changes)


def test_explicit_historical_deployment_and_immutable_root(project):
    names = (
        "IDENTITY.json",
        "SOURCE_MANIFEST.json",
        "CONFIG.json",
        "IMPLEMENTATION_REGISTRATION.json",
    )
    original = {name: (project.root / name).read_bytes() for name in names}
    result = phases.verify_origin(project.root, deployment=project.old)
    assert result["identity"] == project.identity
    assert result["deployment"] == str(project.old)
    assert Path(phases.__file__).resolve().is_relative_to(project.old) is False
    bound = register(project)
    assert bound["status"] == "BOUND_NOT_AUTHORIZED"
    assert bound["training_allowed"] is False
    assert (
        bound["identity"]["source_manifest"]["sha256"]
        != project.identity["source_manifest"]["sha256"]
    )
    assert bound["identity"]["origin"]["identity"]["sha256"] == result["identity_sha256"]
    assert original == {name: (project.root / name).read_bytes() for name in names}
    assert verify(project, expected_identity_sha256=bound["identity_sha256"]) == bound


def test_exact_retry_is_idempotent_and_does_not_mutate_arguments(project):
    before = copy.deepcopy(project.arguments)
    first = register(project)
    path = project.root / "PHASES/PILOT_V1/IDENTITY.json"
    stat, content = path.stat(), path.read_bytes()
    assert register(project) == first
    assert path.stat().st_mtime_ns == stat.st_mtime_ns
    assert path.read_bytes() == content
    assert project.arguments == before
    changed = copy.deepcopy(project.delta)
    changed["changes"][0]["reason"] = "Different prospective explanation"
    with pytest.raises(FileExistsError, match="replacement"):
        register(project, expected_delta=changed)
    assert path.read_bytes() == content


def test_added_module_cannot_impersonate_s0(project):
    with pytest.raises(PermissionError):
        phases.verify_origin(project.root, deployment=project.new)
    (project.old / "src/sr_f6/new.py").write_text("VALUE = 2\n")
    with pytest.raises(PermissionError, match="complete and exact"):
        phases.verify_origin(project.root, deployment=project.old)
    with pytest.raises(PermissionError):
        register(project, phase=phases.ORIGIN)


@pytest.mark.parametrize("which", ["old", "new"])
def test_mutated_bound_source_fails(project, which):
    register(project)
    path = getattr(project, which) / "src/library.py"
    path.write_text("VALUE = 2\n")
    with pytest.raises(PermissionError, match="content changed"):
        verify(project)


def test_new_module_after_freeze_fails(project):
    register(project)
    (project.new / "src/undeclared.py").write_text("VALUE = 3\n")
    with pytest.raises(PermissionError, match="complete and exact"):
        verify(project)


def test_root_identity_cannot_be_rewritten_even_self_consistently(project):
    register(project)
    identity = {**project.identity, "annotation": "retrospective rewrite"}
    write(project.root / "IDENTITY.json", identity)
    with pytest.raises(PermissionError, match="Original root identity changed"):
        verify(project)


@pytest.mark.parametrize(
    "target",
    [
        "CONFIG.json",
        "DATA/DEV_CAL/INPUT.json",
        "DATA/MANIFEST.json",
        "IMPLEMENTATION_REGISTRATION.json",
    ],
)
def test_original_config_data_registration_and_manifests_remain_bound(project, target):
    register(project)
    write(project.root / target, {"changed": True})
    with pytest.raises(PermissionError):
        verify(project)


def test_undeclared_modified_source_rejected_before_registration(project):
    (project.new / "src/library.py").write_text("VALUE = 2\n")
    with pytest.raises(PermissionError, match="declared reason"):
        register(project)
    assert not (project.root / "PHASES/PILOT_V1").exists()


def test_declared_changed_and_deleted_sources_have_exact_delta(project):
    (project.new / "src/library.py").write_text("VALUE = 2\n")
    (project.new / "tests/sr_f6/test_native.py").unlink()
    current = phases.build_manifest(project.new, revision=project.revision)
    delta = phases.source_delta(
        project.original_manifest,
        current,
        reasons={
            "src/library.py": "Explicitly reviewed version change",
            "src/sr_f6/phases.py": "New provenance module",
            "tests/sr_f6/test_native.py": "Explicit fixture replacement",
        },
    )
    assert {row["kind"] for row in delta["changes"]} == {"added", "modified", "deleted"}
    assert register(project, expected_delta=delta)["training_allowed"] is False


def test_changed_evidence_or_wrong_producer_context_is_rejected(project):
    register(project)
    write(project.root / "EVIDENCE/CENSUS.json", {"status": "PASS", "context": project.identity})
    with pytest.raises(PermissionError, match="artifact changed"):
        verify(project)


def test_evidence_with_matching_hash_but_foreign_context_is_rejected(project):
    write(
        project.root / "EVIDENCE/CENSUS.json", {"status": "PASS", "context": {"phase": "FORMAL_V1"}}
    )
    upstream = copy.deepcopy(project.upstream)
    upstream[0]["evidence"][0]["sha256"] = file_hash(project.root / "EVIDENCE/CENSUS.json")
    with pytest.raises(PermissionError, match="context differs"):
        register(project, upstream=upstream)


def test_evidence_inputs_are_bound(project):
    leaf = project.root / "EVIDENCE/leaf.txt"
    leaf.write_text("original witness")
    write(
        project.root / "EVIDENCE/CENSUS.json",
        {
            "context": project.identity,
            "status": "FAIL",
            "inputs": [{"path": "EVIDENCE/leaf.txt", "sha256": file_hash(leaf)}],
        },
    )
    upstream = copy.deepcopy(project.upstream)
    upstream[0]["evidence"][0]["sha256"] = file_hash(project.root / "EVIDENCE/CENSUS.json")
    result = register(project, upstream=upstream)
    assert result["training_allowed"] is False
    leaf.write_text("changed witness")
    with pytest.raises(PermissionError, match="dependency changed"):
        verify(project)


@pytest.mark.parametrize("target", ["source", "evidence", "source_directory", "cache"])
def test_symlinked_members_and_ancestors_are_rejected(project, tmp_path, target):
    if target == "source":
        path = project.new / "src/library.py"
        path.unlink()
        path.symlink_to(project.old / "src/library.py")
    elif target == "evidence":
        path = project.root / "EVIDENCE/CENSUS.json"
        other = tmp_path / "evidence.json"
        path.rename(other)
        path.symlink_to(other)
    elif target == "source_directory":
        path = project.new / "src"
        other = tmp_path / "source"
        path.rename(other)
        path.symlink_to(other, target_is_directory=True)
    else:
        path = project.root / ".verification-cache"
        other = tmp_path / "cache"
        path.rename(other)
        path.symlink_to(other, target_is_directory=True)
    with pytest.raises(PermissionError, match="Symlink"):
        register(project)


def test_external_or_misnamed_deployments_are_rejected(project, tmp_path):
    outside = tmp_path / project.new.name
    shutil.copytree(project.new, outside)
    with pytest.raises(PermissionError, match="outside"):
        register(project, deployment=outside)
    inside = project.new.parent / "arbitrary_code"
    shutil.copytree(project.new, inside)
    with pytest.raises(PermissionError, match="declared revision"):
        register(project, deployment=inside)
    with pytest.raises(PermissionError, match="declared revision"):
        register(project, revision="c" * 40)


def test_source_s0_relabel_and_phase_mix_are_rejected(project):
    with pytest.raises(PermissionError, match="relabel"):
        register(project, deployment=project.old, revision="a" * 40)
    upstream = copy.deepcopy(project.upstream)
    upstream[0]["producer_phase"] = "FORMAL_V1"
    with pytest.raises(PermissionError, match="mixed upstream"):
        register(project, upstream=upstream)
    with pytest.raises(PermissionError, match="Required upstream"):
        register(project, phase="FORMAL_V1")


def test_context_cannot_be_pinned_to_another_identity(project):
    register(project)
    with pytest.raises(PermissionError, match="pinned identity"):
        verify(project, expected_identity_sha256="c" * 64)
    with pytest.raises(PermissionError):
        phases.verify_phase(project.root, "CENSUS_REVIEW_V1", deployment=project.new)


def producer_reference(project, context):
    relative = f"EVIDENCE/{context['phase']}.json"
    write(project.root / relative, {"context": context["identity"], "status": "NOT_RUN"})
    return {
        "producer_phase": context["phase"],
        "identity_sha256": context["identity_sha256"],
        "evidence": [{"path": relative, "sha256": file_hash(project.root / relative)}],
    }


def test_review_pilot_formal_chain_preserves_each_producer_context(project):
    review = register(project, phase="CENSUS_REVIEW_V1")
    review_ref = producer_reference(project, review)
    pilot = register(project, upstream=[*project.upstream, review_ref])
    pilot_ref = producer_reference(project, pilot)
    formal = register(
        project, phase="FORMAL_V1", upstream=[*project.upstream, review_ref, pilot_ref]
    )
    assert verify(project, "FORMAL_V1") == formal
    assert all(context["status"] == "BOUND_NOT_AUTHORIZED" for context in (review, pilot, formal))
    assert all(context["training_allowed"] is False for context in (review, pilot, formal))
    assert read_identity(project.root / "IDENTITY.json") == project.identity
    write(
        project.root / "EVIDENCE/CENSUS_REVIEW_V1.json",
        {"context": formal["identity"], "status": "PASS"},
    )
    with pytest.raises(PermissionError):
        verify(project, "FORMAL_V1")


def read_identity(path):
    return json.loads(path.read_text())


def test_verified_content_is_cached_across_calls_and_only_changed_file_rehashes(
    project, monkeypatch
):
    register(project)
    seen = []
    original = phases.file_hash
    monkeypatch.setattr(
        phases, "file_hash", lambda path: (seen.append(Path(path)), original(path))[1]
    )
    verify(project)
    assert seen == []
    path = project.new / "src/library.py"
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    verify(project)
    assert seen == [path]
    verify(project)
    assert seen == [path]


def test_mount_device_number_is_not_a_cache_invalidation_key():
    stat = SimpleNamespace(st_ino=1, st_size=2, st_mtime_ns=3, st_ctime_ns=4, st_mode=5, st_dev=99)
    path = SimpleNamespace(stat=lambda: stat)
    before = phases._stat(path)
    stat.st_dev = 101
    assert phases._stat(path) == before


def test_recursive_entrypoints_are_in_new_complete_manifest(project):
    script = project.new / "scripts/sr_f6/tools/verify.sh"
    script.parent.mkdir()
    script.write_text("#!/bin/sh\nexit 0\n")
    manifest = phases.build_manifest(project.new, revision=project.revision)
    assert "scripts/sr_f6/tools/verify.sh" in {entry["path"] for entry in manifest["files"]}


def test_duplicate_manifest_and_delta_entries_are_rejected(project):
    duplicate = copy.deepcopy(project.current)
    duplicate["files"].append(duplicate["files"][0])
    with pytest.raises(PermissionError, match="duplicate"):
        phases.source_delta(project.original_manifest, duplicate, reasons={})
    delta = copy.deepcopy(project.delta)
    delta["changes"].append(delta["changes"][0])
    with pytest.raises(PermissionError, match="Duplicate"):
        register(project, expected_delta=delta)


@pytest.mark.parametrize("target", ["CONFIG.json", "IMPLEMENTATION_REGISTRATION.json"])
def test_self_consistent_rewrite_cannot_replace_frozen_contract(project, target):
    identity = copy.deepcopy(project.identity)
    if target == "CONFIG.json":
        config = read_identity(project.root / target)
        config["training"]["aux_coefficient"] = 0.1
        write(project.root / target, config)
        identity["config_sha256"] = file_hash(project.root / target)
    else:
        registration = copy.deepcopy(phases.AMENDMENTS)
        registration["unchanged_science"] = "changed after model output"
        write(project.root / target, registration)
        identity["implementation_digest"] = digest(registration)
    write(project.root / "IDENTITY.json", identity)
    with pytest.raises(PermissionError, match="identity/configuration"):
        phases.verify_origin(project.root, deployment=project.old)


def test_ready_evidence_inputs_transitive_chain_is_bound_without_scanning_models(project):
    leaf = project.root / "EVIDENCE/trajectory.bin"
    leaf.write_bytes(b"actual registered trajectory")
    prerequisite = project.root / "EVIDENCE/PREREQUISITES.json"
    write(
        prerequisite,
        {
            "context": project.identity,
            "inputs": [{"path": "EVIDENCE/trajectory.bin", "sha256": file_hash(leaf)}],
            "model_files": [{"path": "/not-authorized/large-model.bin", "sha256": "0" * 64}],
        },
    )
    write(
        project.root / "EVIDENCE/CENSUS.json",
        {
            "context": project.identity,
            "status": "READY_FOR_CENSUS",
            "evidence": [
                {
                    "path": "EVIDENCE/PREREQUISITES.json",
                    "sha256": file_hash(prerequisite),
                    "checks": ["fixture"],
                }
            ],
        },
    )
    upstream = copy.deepcopy(project.upstream)
    upstream[0]["evidence"][0]["sha256"] = file_hash(project.root / "EVIDENCE/CENSUS.json")
    assert register(project, upstream=upstream)["training_allowed"] is False
    leaf.write_bytes(b"replaced trajectory")
    with pytest.raises(PermissionError, match="dependency changed"):
        verify(project)


def test_cyclic_explicit_evidence_chain_is_rejected(project):
    child = project.root / "EVIDENCE/child.json"
    write(child, {"inputs": [{"path": "EVIDENCE/CENSUS.json", "sha256": "0" * 64}]})
    write(
        project.root / "EVIDENCE/CENSUS.json",
        {
            "context": project.identity,
            "evidence": [{"path": "EVIDENCE/child.json", "sha256": file_hash(child)}],
        },
    )
    upstream = copy.deepcopy(project.upstream)
    upstream[0]["evidence"][0]["sha256"] = file_hash(project.root / "EVIDENCE/CENSUS.json")
    with pytest.raises(PermissionError, match="Cyclic evidence"):
        register(project, upstream=upstream)


@pytest.mark.parametrize("cache_state", ["verified", "wrong_sha", "wrong_stat"])
def test_reuse_existing_s0_gate_cache_requires_expected_sha_and_current_stat(
    project, monkeypatch, cache_state
):
    data = project.root / "DATA/DEV_CAL/INPUT.json"
    metadata = {"stat": phases._stat(data)[:4], "sha256": file_hash(data)}
    if cache_state == "wrong_sha":
        metadata["sha256"] = "0" * 64
    elif cache_state == "wrong_stat":
        metadata["stat"][2] -= 1
    cache_path = (
        project.root / ".verification-cache" / f"{project.identity['data_manifest']['sha256']}.json"
    )
    write(cache_path, {str(data): metadata})
    seen = []
    original = phases.file_hash
    monkeypatch.setattr(
        phases, "file_hash", lambda path: (seen.append(Path(path)), original(path))[1]
    )
    phases.verify_origin(project.root, deployment=project.old)
    assert (data not in seen) is (cache_state == "verified")
