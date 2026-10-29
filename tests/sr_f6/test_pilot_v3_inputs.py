"""SYNTHETIC CPU-only input fixtures; no model, job, measurement or gate PASS."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
from pathlib import Path

import pytest

from sr_f6 import interface_v3_protocol, pilot_v3_inputs, pilot_v3_registration, registration
from sr_f6.common import file_hash
from tests.sr_f6.test_registration import factory, prototype, write  # noqa: F401


@pytest.fixture
def v3_factory(factory):  # noqa: F811 - imported pytest fixture
    def create(*, mutate_data=None):
        return factory(
            phase="PILOT_V3",
            mutate_data=mutate_data,
            real_source_modules=(
                registration,
                pilot_v3_inputs,
                pilot_v3_registration,
                interface_v3_protocol,
            ),
        )

    return create


@pytest.fixture
def v3_project(v3_factory):
    return v3_factory()


def build(project):
    return pilot_v3_inputs.build_overlay(project.root, phase_context=project.context)


def freeze(project):
    return pilot_v3_inputs.freeze_overlay(project.root, phase_context=project.context)


def verify(project, result):
    return pilot_v3_inputs.verify_overlay(
        project.root, phase_context=project.context, overlay_ref=result["manifest_ref"]
    )


def read_rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def test_complete_projection_originals_byte_unchanged(v3_project):
    project = v3_project
    protected = {p: p.read_bytes() for p in (project.root / "DATA").rglob("*") if p.is_file()}
    before_context = copy.deepcopy(project.context)
    result = freeze(project)
    assert verify(project, result) == result
    assert project.context == before_context
    assert protected == {p: p.read_bytes() for p in protected}
    manifest = result["manifest"]
    assert manifest["training_authorized"] is manifest["generation_authorized"] is False
    assert manifest["formal_test_access"] is manifest["old_data_modified"] is False
    assert set(result["inputs_by_split"]) == {"DEV_PILOT", "DEV_EVAL"}
    assert len(manifest["generator_sources"]) == 3
    for split, count in (("DEV_PILOT", 384), ("DEV_EVAL", 96)):
        entry = manifest["splits"][split]
        original = read_rows(project.root / entry["original_public"]["path"])
        private = read_rows(project.root / entry["private"]["path"])
        rows = read_rows(project.root / entry["public"]["path"])
        assert len(rows) == count == entry["rows"]
        for old, task, row in zip(original, private, rows, strict=True):
            expected = copy.deepcopy(old)
            expected["model_input"]["prompt"] = interface_v3_protocol.public_prompt(
                task["world"]["categories"], task["query"]
            )
            expected["prompt_version"] = "srf6-interface-v3"
            assert row == expected
            assert set(row["model_input"]) == {"prompt", "image_file", "image_sha256"}
        assert {row["factor"] for row in rows} == ({"P0", "P1"} if split == "DEV_PILOT" else {"P0"})


def test_prompt_uses_only_public_labels_and_query(v3_project):
    entry = build(v3_project)["manifest"]["splits"]["DEV_PILOT"]
    row = read_rows(v3_project.root / entry["original_public"]["path"])[0]
    task = read_rows(v3_project.root / entry["private"]["path"])[0]
    changed = copy.deepcopy(task)
    changed["gold_answer"] = "SYNTHETIC hidden answer changed"
    changed["gold_selected"] = ["not public"]
    changed["world"]["series"] = {"Alpha": [-999], "Beta": [999]}
    before = copy.deepcopy((row, task, changed))
    assert pilot_v3_inputs._project_public(row, task) == pilot_v3_inputs._project_public(
        row, changed
    )
    assert (row, task, changed) == before


def test_original_prompt_digest_recomputed_only_when_present(v3_factory):
    def add_metadata(root):
        for split in pilot_v3_inputs.SPLITS:
            path = root / f"DATA/{split}/MODEL_INPUTS.jsonl"
            rows = read_rows(path)
            for row in rows:
                row["prompt_sha256"] = hashlib.sha256(
                    row["model_input"]["prompt"].encode()
                ).hexdigest()
                row["source_image_status"] = "GENERATED"
            write(path, rows, lines=True)

    project = v3_factory(mutate_data=add_metadata)
    result = freeze(project)
    for entry in result["manifest"]["splits"].values():
        for row in read_rows(project.root / entry["public"]["path"]):
            assert (
                row["prompt_sha256"]
                == hashlib.sha256(row["model_input"]["prompt"].encode()).hexdigest()
            )
            assert row["source_image_status"] == "GENERATED"


@pytest.mark.parametrize(
    "kind",
    [
        "reorder",
        "missing",
        "duplicate",
        "split",
        "image",
        "injection",
        "metadata_injection",
        "old_prompt",
        "digest",
        "schedule",
    ],
)
def test_malformed_originals_rejected_with_self_consistent_provenance(v3_factory, kind):
    def mutate(root):
        if kind == "schedule":
            path = root / "DATA/SCHEDULES/DEV_PILOT-b0-r0.json"
            rows = json.loads(path.read_bytes())
            rows.reverse()
            write(path, rows)
            return
        path = root / "DATA/DEV_PILOT/MODEL_INPUTS.jsonl"
        rows = read_rows(path)
        if kind == "reorder":
            rows.reverse()
        elif kind == "missing":
            rows.pop()
        elif kind == "duplicate":
            rows[0] = rows[1]
        elif kind == "split":
            rows[0]["split"] = "TRAIN"
        elif kind == "image":
            rows[0]["model_input"]["image_sha256"] = "9" * 64
        elif kind == "injection":
            rows[0]["model_input"]["answer"] = "injected"
        elif kind == "metadata_injection":
            rows[0]["private_answer"] = "injected"
        elif kind == "old_prompt":
            rows[0]["model_input"]["prompt"] += "changed"
        elif kind == "digest":
            rows[0]["prompt_sha256"] = "9" * 64
        write(path, rows, lines=True)

    project = v3_factory(mutate_data=mutate)
    with pytest.raises(PermissionError):
        build(project)


@pytest.mark.parametrize("split", ["TRAIN", "TEST_ID", "TEST_COMPOSITION", "DEV_CAL", "", None])
def test_unknown_split_is_closed(split):
    with pytest.raises(PermissionError, match="two registered"):
        pilot_v3_inputs._split(None, None, None, None, split)


@pytest.mark.parametrize(
    "kind",
    [
        "overlay_prompt",
        "overlay_order",
        "private",
        "image",
        "config_seed",
        "schedule",
        "extra_file",
        "manifest",
        "source",
        "old_phase",
        "source_path",
    ],
)
def test_frozen_boundaries_reject_changes(v3_project, monkeypatch, kind):
    project = v3_project
    result = freeze(project)
    entry = result["manifest"]["splits"]["DEV_PILOT"]
    if kind in {"overlay_prompt", "overlay_order"}:
        path = project.root / entry["public"]["path"]
        rows = read_rows(path)
        if kind == "overlay_order":
            rows.reverse()
        else:
            rows[0]["model_input"]["prompt"] = read_rows(
                project.root / entry["original_public"]["path"]
            )[0]["model_input"]["prompt"]
        write(path, rows, lines=True)
    elif kind == "private":
        path = project.root / entry["private"]["path"]
        rows = read_rows(path)
        rows[0]["world"]["series"]["Alpha"][0] += 1
        write(path, rows, lines=True)
    elif kind == "image":
        (project.root / "DATA/images/cpu-fixture.png").write_bytes(b"changed synthetic image")
    elif kind == "config_seed":
        config = copy.deepcopy(project.config)
        config["master_seed"] += 1
        write(project.root / "CONFIG.json", config)
    elif kind == "schedule":
        path = project.root / result["manifest"]["schedules"]["b0"]["path"]
        value = json.loads(path.read_bytes())
        value.reverse()
        write(path, value)
    elif kind == "extra_file":
        (project.root / pilot_v3_inputs.DIRECTORY / "extra.json").write_text("{}")
    elif kind == "manifest":
        result["manifest"]["formal_test_access"] = True
        path = project.root / result["manifest_ref"]["path"]
        write(path, result["manifest"])
        result["manifest_ref"]["sha256"] = file_hash(path)
    elif kind == "source":
        Path(interface_v3_protocol.__file__).write_text("# modified after freeze\n")
    elif kind == "source_path":
        monkeypatch.setattr(
            interface_v3_protocol, "__file__", str(project.old / "src/sr_f6/runtime.py")
        )
    else:
        project.context["phase"] = "PILOT_V1"
    with pytest.raises(PermissionError):
        verify(project, result)


def test_idempotent_freeze_preserves_bytes_and_metadata(v3_project):
    result = freeze(v3_project)
    directory = v3_project.root / pilot_v3_inputs.DIRECTORY
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.rglob("*") if p.is_file()
    }
    assert freeze(v3_project) == result
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before} == before


def test_partial_published_tree_never_overwritten(v3_project):
    directory = v3_project.root / pilot_v3_inputs.DIRECTORY
    directory.mkdir(parents=True)
    marker = directory / pilot_v3_inputs.MANIFEST_FILE
    marker.write_bytes(b"SYNTHETIC interrupted publication\n")
    with pytest.raises(PermissionError):
        freeze(v3_project)
    assert marker.read_bytes() == b"SYNTHETIC interrupted publication\n"


def test_failed_rename_preserves_attempt_and_no_published_overlay(v3_project, monkeypatch):
    def fail(*args):
        raise OSError("SYNTHETIC publication interruption")

    monkeypatch.setattr(pilot_v3_inputs.os, "rename", fail)
    with pytest.raises(OSError, match="SYNTHETIC"):
        freeze(v3_project)
    assert not (v3_project.root / pilot_v3_inputs.DIRECTORY).exists()
    attempts = list((v3_project.root / "PHASE_RUNS/PILOT_V3").glob(".pilot-v3-overlay-*"))
    assert len(attempts) == 1
    assert len(list(attempts[0].rglob("*.jsonl"))) == 2


def test_held_publication_lock_and_symlink_rejected(v3_project):
    parent = v3_project.root / "PHASE_RUNS/PILOT_V3"
    parent.mkdir(parents=True)
    lock = parent / ".pilot-v3-overlay.lock"
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            freeze(v3_project)
    target = parent / "INPUTS_V3"
    target.symlink_to(v3_project.root / "DATA", target_is_directory=True)
    with pytest.raises(PermissionError, match="Symlink"):
        freeze(v3_project)
