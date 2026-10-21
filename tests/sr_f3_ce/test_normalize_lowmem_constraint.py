"""Real frozen-submit integration for exact held-job constraint normalization."""

import importlib.util
import json
import re
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

BASE = Path(__file__).resolve().parents[2]
SOURCE = BASE / "scripts/sr_f3_ce/normalize_lowmem_constraint.py"
# Reuse the verified low-memory submission fixture, specialized to the exact
# deployed operation/job. Its scientific worker/history boundary remains a stub;
# held fields, command chains, capacity replay and retirement checks are real.
fixture_source = Path(__file__).with_name("test_lowmem_delivery_submission.py")
body = (
    fixture_source.read_text()
    .replace('"operations/new"', '"operations/538aa3dd36e9d5a53f5aa0040b24d3999eab76e9"')
    .replace('"AUTH.json"', '"RECOVERY_AUTHORIZATION.json"')
)
body = re.sub(r"\b99\b", "216983", body)
fixture_module = ModuleType("constraint_original_fixture")
fixture_module.__file__ = str(fixture_source)
exec(compile(body, str(fixture_source), "exec"), fixture_module.__dict__)
case = fixture_module.case


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def setup(case):
    original, calls, faults, _state = case
    original.retire()
    original.submit()
    partial = original.work / "commands/poll_continue_123"
    real_runner = original.runner
    normal = dict(done=False, update_count=0)

    def runner(argv, **kwargs):
        if argv[:2] == ["scontrol", "update"]:
            calls.append(argv)
            normal["update_count"] += 1
            if faults.get("normalize_timeout"):
                raise subprocess.TimeoutExpired(argv, 30)
            if not faults.get("duplicate_again"):
                normal["done"] = True
            return SimpleNamespace(returncode=faults.get("normalize_rc", 0), stdout="", stderr="")
        result = real_runner(argv, **kwargs)
        if argv[:4] == ["scontrol", "show", "job", "216983"] and not normal["done"]:
            result.stdout = result.stdout.replace("Features=cpu_ok", "Features=cpu_ok&cpu_ok")
        return result

    argv = ["scontrol", "show", "job", "216983", "-o"]
    save(partial / "001_held_job.intent.json", dict(argv=argv, at=1))
    save(
        partial / "001_held_job.result.json",
        dict(argv=argv, at=2, returncode=0, stdout=runner(argv).stdout),
    )
    deploy = original.root / "operations/constraint_revision"
    deploy.mkdir()
    helper = deploy / SOURCE.name
    helper.write_bytes(SOURCE.read_bytes())
    spec = importlib.util.spec_from_file_location("deployed_constraint", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    auth = dict(
        root=str(original.root),
        job_id="216983",
        source=str(helper.relative_to(original.root)),
        source_sha256=module.sha(helper),
        original_authorization=dict(
            path=str(original.authorization.relative_to(original.root)),
            sha256=module.sha(original.authorization),
        ),
        partial_poll=dict(
            path=str(partial.relative_to(original.root)),
            files={p.name: module.sha(p) for p in partial.iterdir()},
        ),
    )
    authorization = deploy / "NORMALIZATION_AUTHORIZATION.json"
    save(authorization, auth)
    operation = module.Sidecar(
        original.root, authorization, runner=runner, validator=lambda *_: None
    )
    return operation, calls, faults, normal


def test_normalizes_once_then_original_worker_submission_chain_accepts(setup):
    op, calls, _, normal = setup
    before = {p.name: p.read_bytes() for p in op.partial.iterdir()}
    result = op.run()
    assert result["status"] == "RELEASED"
    assert normal["update_count"] == 1
    assert op.op.prior_release() == result
    assert op.run() == result
    assert normal["update_count"] == 1
    assert {p.name: p.read_bytes() for p in op.partial.iterdir()} == before
    assert not (op.partial / "RESULT.json").exists()
    assert sum(c[0] == "sbatch" for c in calls) == 1
    assert sum(c[:2] == ["scontrol", "release"] for c in calls) == 1
    embedded = json.loads((op.audit / "SOURCE.json").read_text())
    assert embedded["utf8"] == SOURCE.read_text()


def test_capacity_waits_repeat_after_once_only_normalization(setup):
    op, calls, faults, normal = setup
    faults.update(node="IDLE+COMPLETING+PLANNED", node2="ALLOCATED")
    assert op.run()["status"] == "WAIT_CAPACITY"
    assert op.run()["status"] == "WAIT_CAPACITY"
    assert normal["update_count"] == 1
    assert not any(c[:2] == ["scontrol", "release"] for c in calls)
    faults.update(node2="MIXED")
    assert op.run()["status"] == "RELEASED"
    assert op.op.prior_release()["job_id"] == "216983"


@pytest.mark.parametrize("fault", ["duplicate_again", "normalize_timeout", "normalize_rc"])
def test_unknown_failed_or_reduplicated_normalization_never_retried(setup, fault):
    op, calls, faults, normal = setup
    faults[fault] = 1
    with pytest.raises((PermissionError, subprocess.TimeoutExpired)):
        op.run()
    assert (op.audit / "STOP.json").exists()
    with pytest.raises(PermissionError, match="already attempted"):
        op.run()
    assert normal["update_count"] == 1
    assert not any(c[:2] == ["scontrol", "release"] for c in calls)


@pytest.mark.parametrize(
    "marker", ["STARTED.json", "RELEASE_INTENT.json", "ALLOCATION_VERIFIED.json"]
)
def test_advanced_job_never_normalized(setup, marker):
    op, _, _, normal = setup
    save(op.work / marker, {})
    with pytest.raises(PermissionError, match="advanced"):
        op.run()
    assert normal["update_count"] == 0


def test_unknown_partial_query_cannot_be_exempted(setup):
    op, _, _, normal = setup
    p = op.partial / "001_held_job.result.json"
    value = json.loads(p.read_text())
    value["returncode"] = None
    save(p, value)
    with pytest.raises(PermissionError, match="bytes changed"):
        op.run()
    assert normal["update_count"] == 0


def test_started_scheduler_state_stops_before_update(setup):
    op, _, faults, normal = setup
    faults["started"] = True
    with pytest.raises(PermissionError):
        op.run()
    assert normal["update_count"] == 0


def test_source_or_authority_change_blocks(setup):
    op, _, _, normal = setup
    save(op.authorization, dict(op.auth, job_id="999"))
    with pytest.raises(PermissionError, match="authority changed"):
        op.run()
    assert normal["update_count"] == 0


def test_normalization_proof_tampering_blocks_release(setup):
    op, _, faults, normal = setup
    faults.update(node="DOWN", node2="ALLOCATED")
    assert op.run()["status"] == "WAIT_CAPACITY"
    p = op.audit / "005_held_job.result.json"
    value = json.loads(p.read_text())
    value["stdout"] += " changed"
    save(p, value)
    with pytest.raises(PermissionError, match="audit changed"):
        op.run()
    assert normal["update_count"] == 1


def test_unknown_wait_argv_even_with_rebound_inventory_stops(setup):
    op, calls, faults, normal = setup
    faults.update(node="DOWN", node2="ALLOCATED")
    assert op.run()["status"] == "WAIT_CAPACITY"
    poll = next(p for p in (op.work / "commands").glob("poll_continue_*") if p != op.partial)
    for kind in ("intent", "result"):
        p = poll / f"001_held_job.{kind}.json"
        value = json.loads(p.read_text())
        value["argv"] = ["scontrol", "update", "JobId=216983", "Features=gpu_ok"]
        save(p, value)
    marker = json.loads((poll / "RESULT.json").read_text())
    marker["files"] = {
        p.name: op.module.sha(p) for p in poll.glob("*.json") if p.name != "RESULT.json"
    }
    save(poll / "RESULT.json", marker)
    with pytest.raises(PermissionError, match="Unknown wait command"):
        op.run()
    assert normal["update_count"] == 1
    assert not any(c[:2] == ["scontrol", "release"] for c in calls)


def final_fixture(op):
    import zipfile

    op.run()
    archive = op.root / "delivery/SR_F3_CE_20261016.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as z:
        for path in [*op.audit.glob("*.json"), *op.partial.glob("*.json")]:
            z.write(path, "evidence/" + str(path.relative_to(op.root)))
    original = dict(
        status="FINAL_LOWMEM_RECOVERY_VERIFIED",
        archive=dict(path=str(archive), sha256=op.module.sha(archive)),
    )
    save(archive.parent / "FINAL_LOWMEM_RECOVERY_VERIFIED.json", original)
    # Original scientific/build gates are an explicit isolated boundary here;
    # original submission replay and additional ZIP-byte checks remain real.
    calls = []

    def original_final():
        calls.append(True)
        return original

    op.op.verify_final = original_final
    return archive, original, calls


def test_final_calls_original_gate_and_binds_same_unchanged_zip(setup):
    op, _, _, _ = setup
    archive, original, calls = final_fixture(op)
    result = op.verify_final()
    assert calls == [True]
    assert result["status"] == "FINAL_CONSTRAINT_NORMALIZATION_VERIFIED"
    assert result["archive"] == original["archive"]
    assert op.module.sha(archive) == original["archive"]["sha256"]
    assert op.verify_final() == result


def test_final_archive_hash_change_is_refused(setup):
    op, _, _, _ = setup
    archive, _, _ = final_fixture(op)
    with archive.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(PermissionError, match="archive changed"):
        op.verify_final()
    assert not (archive.parent / "FINAL_CONSTRAINT_NORMALIZATION_VERIFIED.json").exists()
