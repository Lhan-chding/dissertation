"""Offline source/path boundaries for the existing-output audit; no model calls."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def helper():
    path = Path(
        os.environ.get(
            "SRF3_EXISTING_OUTPUT_AUDIT",
            Path(__file__).resolve().parents[2] / "scripts/sr_f3_ce/audit_existing_outputs.py",
        )
    )
    spec = importlib.util.spec_from_file_location("sr_f3_ce_existing_output_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True) + "\n")


@pytest.fixture
def source_fixture(helper, tmp_path):
    repo, snapshot = tmp_path / "frozen_source", tmp_path / "snapshot"
    files = []
    for relative in (*helper.SOURCE_FILES, "src/frozen_dependency.py", "README.md"):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        content = ("# Source fixture: " + relative + "\n").encode()
        path.write_bytes(content)
        files.append(dict(path=relative, bytes=len(content), sha256=helper.sha(content)))
    manifest = dict(code_commit=helper.COMMIT, files=files)
    deployment_path = snapshot / "deployment/SOURCE_MANIFEST.json"
    write_json(deployment_path, manifest)
    write_json(
        snapshot / "RUN_IDENTITY.json",
        dict(
            code_commit=helper.COMMIT,
            bound_files={
                "deployment/SOURCE_MANIFEST.json": helper.sha(deployment_path.read_bytes())
            },
        ),
    )
    portable_path = tmp_path / "PORTABLE_SOURCE_MANIFEST.json"
    write_json(portable_path, manifest)
    return SimpleNamespace(
        repo=repo,
        snapshot=snapshot,
        manifest=manifest,
        deployment=deployment_path,
        portable=portable_path,
    )


def test_offline_source_verification_never_calls_git(helper, source_fixture, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline verification must not invoke Git")

    monkeypatch.setattr(helper.subprocess, "check_output", forbidden)
    f = source_fixture
    before = {str(p.relative_to(f.repo)): p.read_bytes() for p in f.repo.rglob("*") if p.is_file()}
    source_sha, checked, receipt = helper.verify_sources(f.repo, f.snapshot, f.portable)
    assert not (f.repo / ".git").exists()
    assert source_sha == helper.sha(f.deployment.read_bytes())
    assert set(checked) == set(helper.SOURCE_FILES)
    assert receipt["mode"] == "run_bound_frozen_source_manifest"
    assert receipt["checked_files"] == len(f.manifest["files"])
    assert before == {
        str(p.relative_to(f.repo)): p.read_bytes() for p in f.repo.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("relative", ["src/sr_f3_ce/contract.py", "README.md"])
def test_changed_source_or_nonpython_inventory_file_is_rejected(helper, source_fixture, relative):
    f = source_fixture
    (f.repo / relative).write_text("modified\n")
    with pytest.raises(PermissionError, match="Frozen source file identity differs"):
        helper.verify_sources(f.repo, f.snapshot, f.portable)


def test_changing_source_and_portable_manifest_cannot_self_authorize(helper, source_fixture):
    f = source_fixture
    path = f.repo / helper.SOURCE_FILES[0]
    path.write_text("changed source and forged matching inventory\n")
    altered = copy.deepcopy(f.manifest)
    altered["files"][0].update(bytes=path.stat().st_size, sha256=helper.sha(path.read_bytes()))
    write_json(f.portable, altered)
    with pytest.raises(PermissionError, match="differs from original deployment"):
        helper.verify_sources(f.repo, f.snapshot, f.portable)


def test_changed_original_manifest_is_rejected_by_run_binding(helper, source_fixture):
    f = source_fixture
    altered = copy.deepcopy(f.manifest)
    altered["files"][0]["sha256"] = "0" * 64
    write_json(f.deployment, altered)
    write_json(f.portable, altered)
    with pytest.raises(PermissionError, match="not bound by the frozen run identity"):
        helper.verify_sources(f.repo, f.snapshot, f.portable)


def test_duplicate_json_key_cannot_hide_manifest_byte_tampering(helper, source_fixture):
    f = source_fixture
    original = f.portable.read_text()
    tampered = original.replace('"code_commit": ', '"code_commit": "forged", "code_commit": ', 1)
    assert json.loads(tampered) == json.loads(original)
    f.portable.write_text(tampered)
    with pytest.raises(PermissionError, match="differs from original deployment"):
        helper.verify_sources(f.repo, f.snapshot, f.portable)


@pytest.mark.parametrize("commit", [None, "0" * 40, "71483fb"])
def test_wrong_or_abbreviated_commit_is_rejected(helper, source_fixture, commit):
    f = source_fixture
    altered = copy.deepcopy(f.manifest)
    altered["code_commit"] = commit
    write_json(f.portable, altered)
    with pytest.raises(PermissionError, match="frozen 714 commit"):
        helper.verify_sources(f.repo, f.snapshot, f.portable)


@pytest.mark.parametrize(
    "path",
    [
        "../outside.py",
        "/absolute.py",
        "src/../outside.py",
        "src\\escape.py",
        "src//empty.py",
        "./src/local.py",
    ],
)
def test_manifest_path_escape_is_rejected(helper, source_fixture, path):
    manifest = copy.deepcopy(source_fixture.manifest)
    manifest["files"][0]["path"] = path
    with pytest.raises(PermissionError, match="escapes its root"):
        helper.manifest_inventory(manifest)


def test_duplicate_path_is_rejected(helper, source_fixture):
    manifest = copy.deepcopy(source_fixture.manifest)
    manifest["files"].append(dict(manifest["files"][0]))
    with pytest.raises(ValueError, match="Duplicate source"):
        helper.manifest_inventory(manifest)


@pytest.mark.parametrize("field,value", [("bytes", True), ("bytes", -1), ("sha256", "not-a-sha")])
def test_malformed_identity_is_rejected(helper, source_fixture, field, value):
    manifest = copy.deepcopy(source_fixture.manifest)
    manifest["files"][0][field] = value
    with pytest.raises(ValueError, match="Malformed source file identity"):
        helper.manifest_inventory(manifest)


def test_source_symlink_cannot_escape_even_with_matching_bytes(helper, source_fixture, tmp_path):
    f = source_fixture
    path = f.repo / helper.SOURCE_FILES[0]
    external = tmp_path / "outside.py"
    external.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(external)
    with pytest.raises(PermissionError, match="escapes its registered root"):
        helper.verify_sources(f.repo, f.snapshot, f.portable)


@pytest.mark.parametrize(
    "relative",
    ["src/extra.py", "src/extra.so", "src/package/__pycache__/cached.pyc", "src/legacy.pyo"],
)
def test_unregistered_python_binary_or_bytecode_is_rejected(helper, source_fixture, relative):
    f = source_fixture
    path = f.repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"unbound")
    with pytest.raises(PermissionError, match=r"unregistered source file|unbound Python bytecode"):
        helper.verify_sources(f.repo, f.snapshot, f.portable)


def test_default_mode_keeps_git_show_fixed_commit(helper, source_fixture, monkeypatch):
    f = source_fixture
    calls = []

    def git_show(args, cwd):
        calls.append((args, cwd))
        assert args[:2] == ["git", "show"] and cwd == f.repo
        commit, relative = args[2].split(":", 1)
        assert commit == helper.COMMIT
        return (f.repo / relative).read_bytes()

    monkeypatch.setattr(helper.subprocess, "check_output", git_show)
    _, checked, receipt = helper.verify_sources(f.repo, f.snapshot)
    assert len(calls) == len(checked) == len(helper.SOURCE_FILES)
    assert receipt["mode"] == "git_show_fixed_commit"


def test_optional_input_paths_and_original_defaults(helper, source_fixture, tmp_path):
    f = source_fixture
    arguments = SimpleNamespace(
        repo=f.repo, snapshot=f.snapshot, output=tmp_path / "review", plan=None, prepared=None
    )
    paths = helper.audit_paths(arguments)
    assert paths[3:] == (
        f.repo / "artifacts/sr_f3_ce_20261016/plan",
        f.repo / "artifacts/sr_f3_ce_20261016/prepared",
    )
    arguments.plan, arguments.prepared = tmp_path / "plan", tmp_path / "prepared"
    assert helper.audit_paths(arguments)[3:] == (arguments.plan, arguments.prepared)


@pytest.mark.parametrize("relative", [".", "review", "new/subdirectory"])
def test_output_inside_snapshot_is_rejected_before_source_read(helper, source_fixture, relative):
    f = source_fixture
    output = f.snapshot / relative
    with pytest.raises(PermissionError, match="must not alter the original snapshot"):
        helper.main(["--repo", str(f.repo), "--snapshot", str(f.snapshot), "--output", str(output)])
    if relative != ".":
        assert not output.exists()


def test_output_symlink_into_snapshot_is_rejected(helper, source_fixture, tmp_path):
    f = source_fixture
    alias = tmp_path / "snapshot-alias"
    alias.symlink_to(f.snapshot, target_is_directory=True)
    with pytest.raises(PermissionError, match="must not alter the original snapshot"):
        helper.main(
            [
                "--repo",
                str(f.repo),
                "--snapshot",
                str(f.snapshot),
                "--output",
                str(alias / "review"),
            ]
        )
    assert not (f.snapshot / "review").exists()


def test_rejects_optimized_python_instead_of_silently_disabling_checks(helper):
    result = subprocess.run(
        [sys.executable, "-O", helper.__file__],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode != 0
    assert "requires Python without -O or -OO" in result.stderr


def test_original_independent_parser_self_checks_are_preserved(helper):
    assert helper.self_check() == {"status": "PASS", "cases": 19}
