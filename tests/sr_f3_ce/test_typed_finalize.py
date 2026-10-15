"""Local synthetic fixtures for the operator-side typed-audit finalizer.

No scientific outputs, gold labels, GPU calls, or publication jobs are used.
Set SRF3_TYPED_FINALIZER when this test and the helper are stored separately.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import sys
from pathlib import Path

import pytest

from sr_f1.chartqa import PROMPT
from sr_f3_ce import external, protocol
from sr_f3_ce.evaluation import EXPERIMENT_ID, object_hash, sha_file


@pytest.fixture(scope="module")
def helper():
    candidates = (
        [Path(os.environ["SRF3_TYPED_FINALIZER"])]
        if "SRF3_TYPED_FINALIZER" in os.environ
        else [
            Path(__file__).resolve().parents[2] / "scripts/sr_f3_ce/finalize_external_types.py",
            Path(__file__).with_name("typed_finalize.py"),
        ]
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise RuntimeError("Set SRF3_TYPED_FINALIZER to the reviewed helper path")
    spec = importlib.util.spec_from_file_location("sr_f3_ce_typed_finalize_operator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def review(item, reviewer, kind="UNRESOLVED", comparison=None):
    return dict(
        qid=item["qid"],
        reviewer_id=reviewer,
        input_identity=external.blind_item_identity(item),
        method="independent_contract_rule_check",
        kind=kind,
        comparison=comparison,
        evidence="Synthetic fixture: explicit interface evidence or a recorded ambiguity.",
        used_gold=False,
        used_model_output=False,
    )


def payload(items, reviewer):
    return dict(
        experiment_id=EXPERIMENT_ID,
        reviewer_id=reviewer,
        blind_inputs_only=True,
        used_gold=False,
        used_model_output=False,
        saw_other_review=False,
        reviews=[review(item, reviewer) for item in items.values()],
    )


@pytest.fixture
def simple_items():
    return {
        "q1": dict(qid="q1", question="Which calendar year?", image_sha256="image-one"),
        "q2": dict(qid="q2", question="How many years of duration?", image_sha256="image-two"),
    }


def test_complete_unresolved_pass_is_reviewed_without_inventing_a_type(
    helper, tmp_path, simple_items
):
    document = payload(simple_items, "A")
    path = tmp_path / "A.json"
    write_json(path, document)
    _, result, scope, snapshot = helper.annotation_set(
        path, simple_items, EXPERIMENT_ID, external, object_hash
    )
    assert snapshot == path.read_bytes()
    assert set(result) == set(simple_items)
    assert all(row["kind"] == "UNRESOLVED" for row in result.values())
    assert scope == object_hash(
        {qid: external.blind_item_identity(item) for qid, item in sorted(simple_items.items())}
    )


@pytest.mark.parametrize(
    "change, exception",
    [
        ("missing_row", ValueError),
        ("wrong_identity", PermissionError),
        ("gold_exposure", PermissionError),
        ("saw_other", PermissionError),
        ("nested_gold", PermissionError),
        ("year_tolerance", ValueError),
        ("unresolved_comparison", ValueError),
        ("empty_evidence", ValueError),
    ],
)
def test_invalid_blind_review_is_rejected(helper, tmp_path, simple_items, change, exception):
    document = payload(simple_items, "A")
    if change == "missing_row":
        document["reviews"].pop()
    elif change == "wrong_identity":
        document["reviews"][0]["input_identity"] = "wrong"
    elif change == "gold_exposure":
        document["reviews"][0]["used_gold"] = True
    elif change == "saw_other":
        document["saw_other_review"] = True
    elif change == "nested_gold":
        document["reviews"][0]["evidence"] = {"gold": "prohibited fixture label"}
    elif change == "year_tolerance":
        document["reviews"][0].update(kind="YEAR", comparison="RELATIVE_5PCT")
    elif change == "unresolved_comparison":
        document["reviews"][0]["comparison"] = "EXACT"
    else:
        document["reviews"][0]["evidence"] = " "
    path = tmp_path / "invalid.json"
    write_json(path, document)
    with pytest.raises(exception):
        helper.annotation_set(path, simple_items, EXPERIMENT_ID, external, object_hash)


@pytest.mark.parametrize("marker", ["EXECUTION_FREEZE.json", "runs", "sealed"])
def test_scientific_freeze_training_or_any_sealed_directory_blocks_revision(
    helper, tmp_path, marker
):
    (tmp_path / marker).mkdir()
    with pytest.raises(PermissionError, match="before science"):
        helper.no_science(tmp_path)


def test_raw_evidence_archive_preserves_rules_selftests_and_visual_bytes(helper, tmp_path):
    source, destination = tmp_path / "original", tmp_path / "archive"
    source.mkdir()
    (source / "rules.py").write_text("RULE = 'explicit interface evidence only'\n")
    (source / "selftest.json").write_text('{"fixture_only":true,"cases":3}\n')
    (source / "visual").mkdir()
    (source / "visual/check.bin").write_bytes(b"\x00\x80visual evidence\xff")
    inventory = helper.evidence_inventory(source, sha_file)
    helper.archive_evidence(source, destination, inventory, sha_file)
    assert len(inventory) == 3
    for item in inventory:
        assert (destination / item["path"]).read_bytes() == (source / item["path"]).read_bytes()
    (source / "rules.py").write_text("changed")
    with pytest.raises(PermissionError, match="changed after inventory"):
        helper.archive_evidence(source, destination, inventory, sha_file)


def test_evidence_symlinks_overlap_and_escape_are_rejected(helper, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "rules.py").write_text("fixture")
    (source / "linked").symlink_to(source / "rules.py")
    with pytest.raises(PermissionError, match="symlinks"):
        helper.evidence_inventory(source, sha_file)
    (source / "linked").unlink()
    inventory = helper.evidence_inventory(source, sha_file)
    with pytest.raises(PermissionError, match="overlap"):
        helper.archive_evidence(source, source / "inside", inventory, sha_file)
    malicious = [dict(inventory[0], path="../outside")]
    with pytest.raises(PermissionError, match="escapes"):
        helper.archive_evidence(source, tmp_path / "dest", malicious, sha_file)


def test_interrupted_immutable_write_does_not_poison_final_name(helper, tmp_path, monkeypatch):
    target = tmp_path / "PANELS.v0.json"
    content = b'{"synthetic":"complete index"}\n'
    real_open, real_fdopen = Path.open, os.fdopen

    class InterruptedWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, value):
            self.stream.write(value[:9])
            self.stream.flush()
            raise OSError("injected interruption after a real partial write")

        def __getattr__(self, name):
            return getattr(self.stream, name)

    def interrupted_open(path, *args, **kwargs):
        stream = real_open(path, *args, **kwargs)
        return InterruptedWriter(stream) if path == target and args == ("xb",) else stream

    def interrupted_fdopen(fd, *args, **kwargs):
        stream = real_fdopen(fd, *args, **kwargs)
        return InterruptedWriter(stream) if args == ("wb",) else stream

    # Exercise both the original direct-write implementation and the replacement
    # temporary writer: the injected error happens after actual filesystem I/O.
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", interrupted_open)
        patch.setattr(helper.os, "fdopen", interrupted_fdopen)
        with pytest.raises(OSError, match="real partial write"):
            helper.write_bytes_once(target, content)
    assert not target.exists()
    helper.write_bytes_once(target, content)
    assert target.read_bytes() == content
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("same_content", [True, False])
def test_immutable_publication_never_overwrites_concurrent_writer(
    helper, tmp_path, monkeypatch, same_content
):
    target, content = tmp_path / "receipt.json", b"complete validated bytes\n"
    concurrent = content if same_content else b"another immutable publisher\n"
    real_link = os.link
    link_calls = []

    def concurrent_link(source, destination, **kwargs):
        link_calls.append((source, destination))
        assert Path(source).parent == target.parent
        assert Path(source).read_bytes() == content
        target.write_bytes(concurrent)
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(helper.os, "link", concurrent_link)
    if same_content:
        helper.write_bytes_once(target, content)
    else:
        with pytest.raises(PermissionError, match="Immutable typed-audit artifact differs"):
            helper.write_bytes_once(target, content)
    assert len(link_calls) == 1
    assert target.read_bytes() == concurrent
    assert list(tmp_path.iterdir()) == [target]


def test_directory_sync_interruption_leaves_complete_retryable_publication(
    helper, tmp_path, monkeypatch
):
    target, content = tmp_path / "receipt.json", b"complete validated bytes\n"
    real_fsync = os.fsync
    synced = []

    def interrupt_directory_sync(fd):
        directory = stat.S_ISDIR(os.fstat(fd).st_mode)
        synced.append("directory" if directory else "file")
        if directory:
            assert target.read_bytes() == content
            raise OSError("injected directory fsync interruption")
        return real_fsync(fd)

    with monkeypatch.context() as patch:
        patch.setattr(helper.os, "fsync", interrupt_directory_sync)
        with pytest.raises(OSError, match="directory fsync"):
            helper.write_bytes_once(target, content)
    assert synced == ["file", "directory"]
    assert target.read_bytes() == content
    helper.write_bytes_once(target, content)
    with pytest.raises(PermissionError, match="Immutable typed-audit artifact differs"):
        helper.write_bytes_once(target, b"different")
    assert target.read_bytes() == content


def make_v0_fixture(root):
    directory = root / "external/chartqa"
    directory.mkdir(parents=True)
    (directory / "image.bin").write_bytes(b"synthetic visible-image fixture bytes")
    image_sha = sha_file(directory / "image.bin")
    items = {
        f"q{i:04d}": dict(qid=f"q{i:04d}", question=f"Fixture question {i}", image_sha256=image_sha)
        for i in range(2500)
    }
    inputs = [
        dict(item, prompt=PROMPT + "\n" + item["question"], image_file="image.bin")
        for item in items.values()
    ]
    for row in inputs:
        del row["question"]
    (directory / "MODEL_INPUTS.jsonl").write_text("".join(json.dumps(row) + "\n" for row in inputs))
    registry = external.build_type_registry(list(items.values()), [], outputs_unseen=True)
    write_json(directory / "TYPE_REGISTRY.json", registry)
    write_json(
        directory / "FREEZE.json",
        dict(
            status="FROZEN",
            image_count=2500,
            protocol="chartqa_plain",
            inputs_sha256=sha_file(directory / "MODEL_INPUTS.jsonl"),
            type_registry_sha256=object_hash(registry),
        ),
    )
    pro = root / "external/chartqapro"
    write_json(
        pro / "FREEZE.json", dict(status="UNAVAILABLE", image_count=0, reason="fixture only")
    )
    panels = dict(
        status="FROZEN",
        experiment_id=EXPERIMENT_ID,
        CHARTQA=dict(
            root="external/chartqa",
            question_count=2500,
            freeze_sha256=sha_file(directory / "FREEZE.json"),
        ),
        CHARTQAPRO=dict(
            root="external/chartqapro",
            image_count=0,
            status="UNAVAILABLE",
            freeze_sha256=sha_file(pro / "FREEZE.json"),
        ),
    )
    write_json(root / "external/PANELS.json", panels)
    return items, panels


def test_v0_inputs_require_exact_image_and_do_not_open_gold_sidecars(helper, tmp_path):
    items, panels = make_v0_fixture(tmp_path)
    # There deliberately is no TASKS_AUDIT_ONLY.jsonl or prior answer manifest.
    _, _, recovered, _ = helper.input_panel(
        tmp_path, panels["CHARTQA"], "CHARTQA", external, object_hash, sha_file
    )
    assert recovered == items
    (tmp_path / "external/chartqa/image.bin").write_bytes(b"changed")
    with pytest.raises(PermissionError, match="image identity"):
        helper.input_panel(tmp_path, panels["CHARTQA"], "CHARTQA", external, object_hash, sha_file)


def test_default_main_is_read_only_and_preserves_disagreement(
    helper, tmp_path, monkeypatch, capsys
):
    items, _ = make_v0_fixture(tmp_path)
    a, b = payload(items, "A"), payload(items, "B")
    a["reviews"][0].update(kind="COUNT_INT", comparison="EXACT")
    b["reviews"][0].update(kind="COUNT_INT", comparison="EXACT")
    a["reviews"][1].update(kind="YEAR", comparison="EXACT")
    b["reviews"][1].update(kind="COUNT_INT", comparison="EXACT")
    b["reviews"][2].update(kind="YEAR", comparison="EXACT")
    write_json(tmp_path / "A.json", a)
    write_json(tmp_path / "B.json", b)
    deployed = tmp_path / "code/src/sr_f3_ce/external.py"
    deployed.parent.mkdir(parents=True)
    deployed.write_text("# fixture identity only; real module functions are exercised\n")
    monkeypatch.setattr(external, "__file__", str(deployed))
    monkeypatch.setattr(
        protocol, "verify_run", lambda root: (dict(code_commit="fixture", bound_files={}), {})
    )
    monkeypatch.setattr(sys, "path", copy.copy(sys.path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "typed_finalize",
            "--root",
            str(tmp_path),
            "--review-a",
            str(tmp_path / "A.json"),
            "--review-b",
            str(tmp_path / "B.json"),
        ],
    )
    before = {
        str(path.relative_to(tmp_path)): sha_file(path)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    helper.main()
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "VALIDATED_NOT_PUBLISHED"
    assert output["panels"]["CHARTQA"]["status_counts"] == {"RESOLVED": 1, "UNRESOLVED": 2499}
    after = {
        str(path.relative_to(tmp_path)): sha_file(path)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert not (tmp_path / "external/revisions").exists()
    assert not (tmp_path / "external/chartqa_typed_v1").exists()
    # A publish request missing the original evidence is rejected before writes.
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--publish"])
    with pytest.raises(PermissionError, match="evidence directories"):
        helper.main()
    assert not (tmp_path / "external/revisions").exists()


def test_publish_gate_requires_lh_job_name_without_writing(helper, tmp_path, monkeypatch):
    items, _ = make_v0_fixture(tmp_path)
    for label in ("A", "B"):
        write_json(tmp_path / (label + ".json"), payload(items, label))
        evidence = tmp_path / (label + "-evidence")
        evidence.mkdir()
        (evidence / "rules.py").write_text("# synthetic independent rule fixture\n")
    deployed = tmp_path / "code/src/sr_f3_ce/external.py"
    deployed.parent.mkdir(parents=True)
    deployed.write_text("# fixture identity only\n")
    monkeypatch.setattr(external, "__file__", str(deployed))
    monkeypatch.setattr(
        protocol, "verify_run", lambda root: (dict(code_commit="fixture", bound_files={}), {})
    )
    monkeypatch.setattr(sys, "path", copy.copy(sys.path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "typed_finalize",
            "--root",
            str(tmp_path),
            "--review-a",
            str(tmp_path / "A.json"),
            "--review-b",
            str(tmp_path / "B.json"),
            "--review-a-evidence-dir",
            str(tmp_path / "A-evidence"),
            "--review-b-evidence-dir",
            str(tmp_path / "B-evidence"),
            "--auxiliary-code-commit",
            "a" * 40,
            "--publish",
        ],
    )
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_JOB_NAME", "missing-required-suffix")
    before = {
        str(path.relative_to(tmp_path)): sha_file(path)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    with pytest.raises(PermissionError, match="-LH"):
        helper.main()
    after = {
        str(path.relative_to(tmp_path)): sha_file(path)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert not (tmp_path / "external/revisions").exists()


def test_published_review_bytes_are_the_validated_snapshot_despite_source_change(
    helper, tmp_path, monkeypatch, capsys
):
    items, _ = make_v0_fixture(tmp_path)
    write_json(tmp_path / "RUN_IDENTITY.json", {"fixture": True})
    write_json(tmp_path / "prior/HISTORICAL_IMAGES.json", {"fixture": True})
    old = tmp_path / "external/chartqa"
    (old / "TASKS_AUDIT_ONLY.jsonl").write_bytes(b"")
    snapshots = {}
    for label in ("A", "B"):
        path = tmp_path / (label + ".json")
        write_json(path, payload(items, label))
        snapshots[label] = path.read_bytes()
        evidence = tmp_path / (label + "-evidence")
        evidence.mkdir()
        (evidence / "rules.py").write_text("# independent synthetic check; no gold or outputs\n")
    deployed = tmp_path / "code/src/sr_f3_ce/external.py"
    deployed.parent.mkdir(parents=True)
    deployed.write_text("# fixture identity only\n")

    def prepare_fixture(manifest, image_root, output_root, *, history_receipt, type_reviews):
        output_root.mkdir(parents=True, exist_ok=True)
        for name in ("MODEL_INPUTS.jsonl", "TASKS_AUDIT_ONLY.jsonl", "image.bin"):
            shutil.copyfile(old / name, output_root / name)
        registry = external.build_type_registry(
            list(items.values()), type_reviews, outputs_unseen=True
        )
        write_json(output_root / "TYPE_REGISTRY.json", registry)
        freeze = helper.read(old / "FREEZE.json")
        freeze["type_registry_sha256"] = object_hash(registry)
        write_json(output_root / "FREEZE.json", freeze)
        return freeze

    inventory, changed = helper.evidence_inventory, False

    def inventory_after_source_change(directory, hash_function):
        nonlocal changed
        if not changed:
            for label in snapshots:
                path = tmp_path / (label + ".json")
                value = helper.read(path)
                value["reviews"][0].update(kind="COUNT_INT", comparison="EXACT")
                write_json(path, value)
            changed = True
        return inventory(directory, hash_function)

    monkeypatch.setattr(external, "__file__", str(deployed))
    monkeypatch.setattr(external, "prepare_chartqa", prepare_fixture)
    monkeypatch.setattr(
        protocol, "verify_run", lambda root: (dict(code_commit="fixture", bound_files={}), {})
    )
    monkeypatch.setattr(helper, "evidence_inventory", inventory_after_source_change)
    monkeypatch.setattr(sys, "path", copy.copy(sys.path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "typed_finalize",
            "--root",
            str(tmp_path),
            "--review-a",
            str(tmp_path / "A.json"),
            "--review-b",
            str(tmp_path / "B.json"),
            "--review-a-evidence-dir",
            str(tmp_path / "A-evidence"),
            "--review-b-evidence-dir",
            str(tmp_path / "B-evidence"),
            "--auxiliary-code-commit",
            "a" * 40,
            "--publish",
        ],
    )
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_JOB_NAME", "synthetic-typed-regression-LH")
    helper.main()
    assert json.loads(capsys.readouterr().out)["status"] == "PUBLISHED_TYPED_V1_BEFORE_SCIENCE"
    revision = tmp_path / "external/revisions/typed_v1"
    audit = helper.read(revision / "TYPE_AUDIT_RECEIPT.json")
    registry = helper.read(tmp_path / "external/chartqa_typed_v1/TYPE_REGISTRY.json")
    for label, original in snapshots.items():
        archived = revision / ("REVIEW_" + label + ".json")
        assert (tmp_path / (label + ".json")).read_bytes() != original
        assert archived.read_bytes() == original
        assert audit["review_files"][archived.name] == hashlib.sha256(original).hexdigest()
    archived_reviews = [
        row for label in snapshots for row in json.loads(snapshots[label])["reviews"]
    ]
    assert registry == external.build_type_registry(
        list(items.values()), archived_reviews, outputs_unseen=True
    )
    assert registry["rows"][0]["status"] == "UNRESOLVED"
    # Retrying the same archived input is idempotent; changed source paths are
    # intentionally restored from our synthetic snapshots for this retry.
    for label, original in snapshots.items():
        (tmp_path / (label + ".json")).write_bytes(original)
    before = {
        str(path.relative_to(tmp_path)): sha_file(path)
        for path in (tmp_path / "external").rglob("*")
        if path.is_file()
    }
    helper.main()
    assert json.loads(capsys.readouterr().out)["status"] == "PUBLISHED_TYPED_V1_BEFORE_SCIENCE"
    after = {
        str(path.relative_to(tmp_path)): sha_file(path)
        for path in (tmp_path / "external").rglob("*")
        if path.is_file()
    }
    assert before == after
