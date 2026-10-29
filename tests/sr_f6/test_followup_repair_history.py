"""Synthetic historical negative evidence; no actual scientific authorization."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sr_f6_followup import repair_history, source


@pytest.fixture
def history(tmp_path, monkeypatch):
    project = tmp_path.resolve()
    root = project / "old-followup"
    deployment = project / "sr_f6_followup_code_02fbf89"
    monkeypatch.setattr(source, "PROJECT", project)
    monkeypatch.setattr(source, "ROOT", project / "new-followup")
    monkeypatch.setattr(repair_history, "ROOT", root)
    monkeypatch.setattr(repair_history, "DEPLOYMENT", deployment)
    monkeypatch.setattr(repair_history, "_CACHE", None)

    def write(name, value, *, lines=False):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            "".join(json.dumps(row) + "\n" for row in value).encode()
            if lines
            else source._payload(value)
        )
        path.write_bytes(payload)
        return dict(path=name, sha256=hashlib.sha256(payload).hexdigest())

    for name in ("__init__.py", "source.py"):
        path = deployment / "src/sr_f6_followup" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"""SYNTHETIC_ONLY historical deployed source."""\n')
    mf = write(
        "SOURCE_MANIFEST.json",
        dict(
            schema="srf6-followup-source-v1",
            revision=repair_history.REVISION,
            deployment=str(deployment),
            files=[source._inventory(path) for path in source._source_paths(deployment)],
        ),
    )
    spec = write("SPEC.json", dict(scope="SYNTHETIC_ONLY"))
    producer = dict(
        root=str(source.PRODUCER_ROOT),
        deployment=str(source.PRODUCER_DEPLOYMENT),
        revision=source.PRODUCER_REVISION,
        registration_sha256=source.PRODUCER_REGISTRATION_SHA256,
        source_manifest_sha256="a" * 64,
    )
    reg = write(
        "REGISTRATION.json",
        dict(
            schema="srf6-followup-registration-v1",
            root=str(root),
            deployment=str(deployment),
            revision=repair_history.REVISION,
            status="BOUND_NOT_AUTHORIZED",
            producer=producer,
            source_manifest=dict(mf, path=str(root / mf["path"])),
            spec=dict(spec, path=str(root / spec["path"])),
            training_allowed=False,
            generation_allowed=False,
            evaluation_allowed=False,
            model_calls=0,
            optimizer_updates=0,
        ),
    )
    consumer = dict(
        deployment=str(deployment),
        revision=repair_history.REVISION,
        source_manifest_sha256=mf["sha256"],
        registration_sha256=reg["sha256"],
    )
    input_producer = dict(
        phase="SCHEMA_CONFIRM_V3",
        deployment=str(source.PRODUCER_DEPLOYMENT),
        revision=source.PRODUCER_REVISION,
        source_manifest=dict(path="SOURCE_MANIFEST.json", sha256="a" * 64),
    )
    ledger = write("REVIEW/PACKAGE/private/selection.json", {"fixture": "SYNTHETIC_ONLY"})
    blind = write(
        "REVIEW/PACKAGE/blind/EXTRACTION.jsonl", [{"fixture": "SYNTHETIC_ONLY"}], lines=True
    )
    tasks = write(
        "REVIEW/PACKAGE/judgment/TASKS.jsonl", [{"fixture": "SYNTHETIC_ONLY"}], lines=True
    )
    support = write("SUPPORT_BINDING.json", {"fixture": "SYNTHETIC_ONLY"})
    package_inputs = [reg, mf, spec, support, ledger, blind, tasks]
    package = write(
        "REVIEW/PACKAGE/BINDING.json",
        dict(
            schema="srf6-followup-review-package-v1",
            consumer=consumer,
            input_producer=input_producer,
            phase="SCHEMA_CONFIRM_V3",
            split="DEV_SCHEMA_CONFIRM",
            status="BLIND_PACKAGE_ONLY",
            model_calls=0,
            optimizer_updates=0,
            training_allowed=False,
            generation_allowed=False,
            evaluation_allowed=False,
            ledger=ledger,
            blind=blind,
            judgment_tasks=tasks,
            inputs=package_inputs,
        ),
    )
    execution = write("REVIEW/EXECUTIONS/reviewer/ATTEMPT.json", {"fixture": "SYNTHETIC_ONLY"})
    summary = write(
        "REVIEW/AUDIT/SUMMARY.json",
        dict(
            schema="srf6-followup-independent-review-audit-v1",
            consumer=consumer,
            input_producer=input_producer,
            status="BLOCKING_FINDINGS",
            completed=False,
            synthetic=False,
            training_allowed=False,
            gate_status="NOT_SIGNED",
            package_ref=package,
            reviewed_original_answers=360,
            independently_reviewed_original_answers=360,
            blocking_finding_ids=list(range(8)),
            adjudication_required_ids=list(range(8, 16)),
            inputs=[*package_inputs, package, execution],
            external_inputs=[dict(path="/DO_NOT_READ_SYNTHETIC_MODEL", sha256="f" * 64)],
        ),
    )
    seal = write(
        "REVIEW/AUDIT/BINDING.json",
        dict(
            schema="srf6-followup-review-seal-v1",
            summary=summary,
            consumer=consumer,
            input_producer=input_producer,
            package=package,
            status="BLOCKING_FINDINGS",
            completed=False,
            training_allowed=False,
            gate_status="NOT_SIGNED",
        ),
    )
    write("REVIEW/.audit-attempt.json", dict(schema="srf6-followup-review-seal-attempt-v1"))
    pins = {
        ref["path"]: ref["sha256"]
        for ref in [reg, mf, package, ledger, blind, tasks, summary, seal]
    }
    for path in root.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    monkeypatch.setattr(repair_history, "PINS", pins)
    return dict(root=root, deployment=deployment, write=write, pins=pins, monkeypatch=monkeypatch)


def test_complete_historical_failure_preserved_without_authority(history):
    result = repair_history.verify()
    assert set(result) == {"references", "package", "ledger", "blind", "tasks", "summary"}
    assert result["summary"]["status"] == "BLOCKING_FINDINGS"
    assert result["summary"]["completed"] is False
    assert all(Path(ref["path"]).is_relative_to(source.PROJECT) for ref in result["references"])
    assert not any("DO_NOT_READ" in ref["path"] for ref in result["references"])
    assert len(result["references"]) == len({ref["path"] for ref in result["references"]})
    first = copy.deepcopy(result)
    result["summary"]["status"] = "FORGED_PASS"
    assert repair_history.verify() == first


def test_warm_history_uses_stamps_without_content_reads(history, monkeypatch):
    expected = repair_history.verify()
    monkeypatch.setattr(
        Path, "read_bytes", lambda *args: pytest.fail("Warm history must not reread")
    )
    monkeypatch.setattr(source, "_hash", lambda *args: pytest.fail("Warm history must not rehash"))
    assert repair_history.verify() == expected


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize(
    "name",
    [
        "REGISTRATION.json",
        "SOURCE_MANIFEST.json",
        "REVIEW/PACKAGE/blind/EXTRACTION.jsonl",
        "REVIEW/AUDIT/SUMMARY.json",
        "REVIEW/EXECUTIONS/reviewer/ATTEMPT.json",
    ],
)
def test_historical_evidence_tamper_rejected(history, warm, name):
    if warm:
        repair_history.verify()
    (history["root"] / name).write_text("changed")
    with pytest.raises(PermissionError, match="changed"):
        repair_history.verify()


@pytest.mark.parametrize("warm", [False, True])
def test_extra_old_execution_is_not_hidden(history, warm):
    if warm:
        repair_history.verify()
    history["write"]("REVIEW/EXECUTIONS/extra/ATTEMPT.json", {"fixture": "SYNTHETIC_ONLY"})
    with pytest.raises(PermissionError, match=r"membership|closure"):
        repair_history.verify()


def test_old_deployed_source_change_is_rejected(history):
    repair_history.verify()
    (history["deployment"] / "src/sr_f6_followup/source.py").write_text("changed")
    with pytest.raises(PermissionError, match="changed"):
        repair_history.verify()


def test_added_old_deployment_source_is_rejected_by_warm_cache(history):
    repair_history.verify()
    (history["deployment"] / "src/sr_f6_followup/extra.py").write_text("SYNTHETIC_ONLY = True")
    with pytest.raises(PermissionError, match="source inventory changed"):
        repair_history.verify()


def test_old_and_new_root_may_not_be_same(history, monkeypatch):
    monkeypatch.setattr(source, "ROOT", history["root"])
    with pytest.raises(PermissionError, match="new consumer root"):
        repair_history.verify()


def test_old_evidence_symlink_is_rejected(history):
    path = history["root"] / "REVIEW/AUDIT/SUMMARY.json"
    target = history["root"] / "moved-summary.json"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(PermissionError, match="Symlinked"):
        repair_history.verify()


def test_nonterminal_old_failure_is_rejected_even_with_consistent_pin(history):
    path = history["root"] / "REVIEW/AUDIT/SUMMARY.json"
    value = json.loads(path.read_text())
    value["status"] = "REVIEW_COMPLETE_NOT_GATE_SIGNED"
    ref = history["write"]("REVIEW/AUDIT/SUMMARY.json", value)
    history["pins"][ref["path"]] = ref["sha256"]
    with pytest.raises(PermissionError, match="historical failure"):
        repair_history.verify()


def test_historical_relative_reference_cannot_escape(history):
    with pytest.raises(PermissionError, match="escapes"):
        repair_history._relative(dict(path="../foreign", sha256="a" * 64))


def test_incomplete_or_foreign_staging_is_preserved(history):
    stage = history["root"] / "REVIEW/.PACKAGE.staging"
    stage.mkdir()
    (stage / "preserved").write_text("SYNTHETIC_ONLY partial")
    with pytest.raises(PermissionError):
        repair_history.verify()
    assert (stage / "preserved").exists()
