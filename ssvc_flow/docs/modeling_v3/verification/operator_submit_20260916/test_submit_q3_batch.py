"""Tiny operator fixtures only: every scheduler command is a mocked subprocess."""

import errno
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

FLOW = next(p for p in Path(__file__).resolve().parents if (p / "src/modeling_v3").is_dir())
OPERATORS = FLOW / "docs/modeling_v3/operators"


def module(name):
    spec = importlib.util.spec_from_file_location(name, OPERATORS / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


submitter = module("submit_q3_batch")
prepare = module("prepare_q3_operator")


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate06"
    checkout = candidate / "checkout"
    flow = checkout / "ssvc_flow"
    flow.mkdir(parents=True)
    bundle = candidate / "Q3_OPERATOR_INTERVAL_CALIBRATION"
    (bundle / "requests").mkdir(parents=True)
    (candidate / "Q3_campaign").mkdir()
    (candidate / "Q3_operator_logs").mkdir()
    config_path = flow / "configs/modeling_v3/protocol.json"
    config_path.parent.mkdir(parents=True)
    config = json.loads((FLOW / "configs/modeling_v3/protocol.json").read_text())
    config_path.write_text(json.dumps(config))
    lock_path = candidate / "Q3_SELECTION/SELECTION_LOCK.json"
    lock_path.parent.mkdir()
    lock_path.write_text(json.dumps({"selected": {"fixture": True}}))
    snapshot = checkout / "CODE_SNAPSHOT_MANIFEST.json"
    snapshot.write_text(json.dumps({"ssvc_flow/src/cpu_fixture.py": "0" * 64}))
    roots = [tmp_path / f"historical{i}" for i in range(4)] + [candidate / "Q3_campaign"]
    rows = prepare.calibration_requests(config, lock_path, config_path, roots[-1], roots)
    requests = []
    for row in rows:
        path = bundle / "requests" / f"{row['index']:02d}_seed{row['seed']}.json"
        prepare.seal_json(path, row["request"])
        requests.append({"path": str(path), "sha256": prepare.sha256(path)})
    prepare.seal_json(bundle / "REQUEST_LIST.json", requests)
    prepare.seal_json(
        bundle / "SUBMISSION_INTENTS.json",
        prepare.submission_intents(
            checkout, candidate, bundle, prepare.sha256(bundle / "REQUEST_LIST.json")
        ),
    )
    prepare.seal_json(bundle / "CALIBRATION_MATRIX.json", {"rows": rows})
    binding = {
        "candidate_root": str(candidate),
        "checkout": str(checkout),
        "snapshot_manifest_sha256": prepare.sha256(snapshot),
        "config_path": str(config_path),
        "selection_lock_path": str(lock_path),
        "inventory_basis_path": str(tmp_path / "basis.json"),
        "inventory_basis_sha256": "1" * 64,
        "historical_roots": list(map(str, roots[:-1])),
        "config_file_sha256": prepare.sha256(config_path),
        "selection_hash": "fixture-selection",
    }
    manifest = bundle / "PREPARATION_MANIFEST.json"
    prepare.seal_json(
        manifest,
        {
            "schema": "ssvc-v3-q3-operator-preparation-1",
            "status": "PREPARED_NOT_SUBMITTED",
            "binding": binding,
            "operator_script_sha256": prepare.sha256(OPERATORS / "prepare_q3_operator.py"),
            "files": {
                str(p.relative_to(bundle)): prepare.sha256(p)
                for p in bundle.rglob("*")
                if p.is_file()
            },
            "request_count": 20,
        },
    )
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(submitter, "SERVER_BASE", tmp_path)
    context = {
        "candidate": candidate,
        "bundle": bundle,
        "manifest": manifest,
        "sha256": prepare.sha256(manifest),
        "rows": rows,
        "calls": [],
        "queue": b"",
        "sbatch_stdout": b"155999\n",
        "sbatch_code": 0,
        "verify_code": 0,
        "sacct": b"155999_0|COMPLETED|0:0\n",
        "binding": binding,
    }

    def run(argv, **kwargs):
        context["calls"].append((list(argv), kwargs))
        if "--mode" in argv:
            return subprocess.CompletedProcess(
                argv, context["verify_code"], b'{"status":"VERIFIED_NOT_SUBMITTED"}\n', b""
            )
        if argv[0] == "squeue":
            return subprocess.CompletedProcess(argv, 0, context["queue"], b"")
        if argv[0] == "sacct":
            return subprocess.CompletedProcess(argv, 0, context["sacct"], b"")
        assert argv[0] == "sbatch"
        if context.get("timeout"):
            raise subprocess.TimeoutExpired(argv, 120, output=b"possibly submitted")
        return subprocess.CompletedProcess(
            argv, context["sbatch_code"], context["sbatch_stdout"], b"fixture stderr\n"
        )

    context["run"] = run
    return context


def dispatch(c, batch=0, **kwargs):
    return submitter.submit_batch(c["manifest"], c["sha256"], batch, run=c["run"], **kwargs)


def finish_first(c):
    dispatch(c)
    out = Path(c["rows"][0]["output"])
    out.mkdir()
    binding = {
        "config_sha256": submitter.canonical_hash(
            json.loads(Path(c["binding"]["config_path"]).read_text())
        ),
        "source_hashes": {"src/cpu_fixture.py": "0" * 64},
        "selection_hash": "fixture-selection",
        "seed_subset": [21001],
        "arm_subset": ["X_BASE", "X_VALID"],
        "roles": ["interval_calibration"],
        "stage": "Q3_FROZEN_CAMPAIGN",
    }
    prepare.seal_json(out / "BINDING.json", binding)
    prepare.seal_json(
        out / "COMPLETE.json",
        {
            "status": "COMPLETE",
            "binding": binding,
            "summary": {"stage": "Q3", "status": "COLLECTED_AND_MEASURED"},
            "files": {"BINDING.json": prepare.sha256(out / "BINDING.json")},
        },
    )
    review = c["candidate"] / "review_batch_0.md"
    review.write_text("Fixture human review evidence, not a scientific qualification.\n")
    return {"previous_review": review, "previous_review_sha256": prepare.sha256(review)}


def test_first_batch_exact_single_submission(campaign):
    c = campaign
    result = dispatch(c)
    assert result["status"] == "SUBMITTED"
    assert result["job_id"] == "155999"
    calls = [row for row in c["calls"] if row[0][0] == "sbatch"]
    assert len(calls) == 1
    original = json.loads((c["bundle"] / "SUBMISSION_INTENTS.json").read_text())[0]
    assert calls[0][0] == original["argv"]
    for key, value in original["environment"].items():
        assert calls[0][1]["env"][key] == value
    assert not any(Path(row["output"]).exists() for row in c["rows"])
    assert [row[0][0] for row in c["calls"]][-2:] == ["squeue", "sbatch"]


def test_duplicate_batch_never_submits_twice(campaign):
    dispatch(campaign)
    with pytest.raises(FileExistsError):
        dispatch(campaign)
    assert sum(row[0][0] == "sbatch" for row in campaign["calls"]) == 1


def test_current_output_refused(campaign):
    Path(campaign["rows"][0]["output"]).mkdir()
    with pytest.raises(FileExistsError):
        dispatch(campaign)
    assert not any(row[0][0] == "sbatch" for row in campaign["calls"])


def test_verify_failure_stops_before_queue(campaign):
    campaign["verify_code"] = 1
    with pytest.raises(ValueError):
        dispatch(campaign)
    assert len(campaign["calls"]) == 1


def test_outer_hash_mismatch_stops_before_subprocess(campaign):
    campaign["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        dispatch(campaign)
    assert not campaign["calls"]


@pytest.mark.parametrize("index", [-1, True, 5, "0"])
def test_invalid_batch_index(campaign, index):
    with pytest.raises(ValueError):
        dispatch(campaign, index)


def test_later_batch_needs_prior_completion(campaign):
    with pytest.raises((ValueError, FileNotFoundError)):
        dispatch(campaign, 1)
    assert not any(row[0][0] == "sbatch" for row in campaign["calls"])


def test_later_batch_needs_explicit_review(campaign):
    finish_first(campaign)
    with pytest.raises(ValueError):
        dispatch(campaign, 1)


def test_later_batch_complete_reviewed_and_capacity(campaign):
    review = finish_first(campaign)
    result = dispatch(campaign, 1, **review)
    assert result["status"] == "SUBMITTED"
    assert result["indices"] == [1, 2, 3, 4, 5]
    assert sum(row[0][0] == "sbatch" for row in campaign["calls"]) == 2


def test_previous_failed_slurm_refused(campaign):
    review = finish_first(campaign)
    campaign["sacct"] = b"155999_0|FAILED|1:0\n"
    with pytest.raises(ValueError):
        dispatch(campaign, 1, **review)


def test_previous_original_tamper_refused(campaign):
    review = finish_first(campaign)
    (Path(campaign["rows"][0]["output"]) / "BINDING.json").write_text("{}")
    with pytest.raises(ValueError):
        dispatch(campaign, 1, **review)


def test_review_hash_mismatch_refused(campaign):
    review = finish_first(campaign)
    review["previous_review"].write_text("changed")
    with pytest.raises(ValueError):
        dispatch(campaign, 1, **review)


def test_timeout_preserves_intent_and_prevents_retry(campaign):
    campaign["timeout"] = True
    result = dispatch(campaign)
    assert result["status"] == "SUBMISSION_OUTCOME_UNKNOWN"
    with pytest.raises(FileExistsError):
        dispatch(campaign)


def test_scheduler_rejection_preserved(campaign):
    campaign["sbatch_code"] = 1
    campaign["sbatch_stdout"] = b""
    assert dispatch(campaign)["status"] == "SUBMISSION_REJECTED"


def test_inherited_scheduler_and_python_overrides_removed(campaign):
    with patch.dict(
        os.environ, {"SBATCH_GRES": "gpu:8", "PYTHONPATH": "/bad", "CUDA_VISIBLE_DEVICES": "0"}
    ):
        dispatch(campaign)
    env = next(row[1]["env"] for row in campaign["calls"] if row[0][0] == "sbatch")
    assert "SBATCH_GRES" not in env and "PYTHONPATH" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == ""


def test_queue_element_count_and_rejections():
    qos = submitter.QOS
    assert submitter.queue_elements(
        f"42_0|u|{qos}|RUNNING\n42_1|u|{qos}|PENDING\n".encode(), "u"
    ) == ["42_0", "42_1"]
    for payload in [
        f"42_[0-5]|u|{qos}|PENDING\n",
        f"42_0|u|{qos}|PENDING\n" * 2,
        "42|other|other|RUNNING\n",
    ]:
        with pytest.raises(ValueError):
            submitter.queue_elements(payload.encode(), "u")


def test_capacity_checks_pending_elements(campaign):
    user = submitter.current_user()
    campaign["queue"] = "".join(
        f"42_{i}|{user}|{submitter.QOS}|PENDING\n" for i in range(5)
    ).encode()
    with pytest.raises(ValueError):
        dispatch(campaign)
    assert not any(row[0][0] == "sbatch" for row in campaign["calls"])


def test_sealed_gpu_command_rejected_even_with_new_manifest_hash(campaign):
    c = campaign
    path = c["bundle"] / "SUBMISSION_INTENTS.json"
    intents = json.loads(path.read_text())
    intents[0]["argv"].insert(1, "--gres=gpu:2")
    path.write_text(json.dumps(intents))
    manifest = json.loads(c["manifest"].read_text())
    manifest["files"][path.name] = prepare.sha256(path)
    c["manifest"].write_text(json.dumps(manifest))
    c["sha256"] = prepare.sha256(c["manifest"])
    with pytest.raises(ValueError):
        dispatch(c)
    assert not any(row[0][0] == "sbatch" for row in c["calls"])


def test_five_element_batch_needs_five_free_slots(campaign):
    review = finish_first(campaign)
    user = submitter.current_user()
    campaign["queue"] = f"42_0|{user}|{submitter.QOS}|PENDING\n".encode()
    with pytest.raises(ValueError):
        dispatch(campaign, 1, **review)
    assert sum(row[0][0] == "sbatch" for row in campaign["calls"]) == 1


def test_preserved_empty_submission_directory_blocks_retry(campaign):
    (campaign["candidate"] / "Q3_SUBMISSIONS/batch_00").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        dispatch(campaign)
    assert not any(row[0][0] == "sbatch" for row in campaign["calls"])


def test_malformed_success_stdout_is_uncertain_and_not_retried(campaign):
    campaign["sbatch_stdout"] = b"unexpected success text\n"
    assert dispatch(campaign)["status"] == "SUBMISSION_OUTCOME_UNKNOWN"
    with pytest.raises(FileExistsError):
        dispatch(campaign)


def test_previous_sbatch_original_tamper(campaign):
    review = finish_first(campaign)
    (campaign["candidate"] / "Q3_SUBMISSIONS/batch_00/sbatch.stdout").write_text("666\n")
    with pytest.raises(ValueError):
        dispatch(campaign, 1, **review)


def test_previous_incomplete_raw_output(campaign):
    review = finish_first(campaign)
    (Path(campaign["rows"][0]["output"]) / "COMPLETE.json").unlink()
    with pytest.raises(FileNotFoundError):
        dispatch(campaign, 1, **review)


def test_submission_intent_quota_error_prevents_sbatch(campaign):
    original_fsync = os.fsync

    def fail_intent_fsync(fd):
        # This test process only calls fsync when sealing its pre-submit intent.
        raise OSError(errno.EDQUOT, "Disk quota exceeded")

    with patch.object(os, "fsync", fail_intent_fsync), pytest.raises(OSError) as raised:
        dispatch(campaign)
    assert raised.value.errno == errno.EDQUOT
    assert os.fsync is original_fsync
    assert not any(row[0][0] == "sbatch" for row in campaign["calls"])
    with pytest.raises(FileExistsError):
        dispatch(campaign)


def test_submit_receipt_preserves_exact_stdout_stderr_bytes(campaign):
    result = dispatch(campaign)
    assert Path(result["stdout"]["path"]).read_bytes() == campaign["sbatch_stdout"]
    assert Path(result["stderr"]["path"]).read_bytes() == b"fixture stderr\n"


def test_sacct_requires_every_expected_element():
    with pytest.raises(ValueError):
        submitter._completed_elements(b"51_1|COMPLETED|0:0\n", "51", [1, 2])
    with pytest.raises(ValueError):
        submitter._completed_elements(b"51_1|COMPLETED|0:0\n51_2|TIMEOUT|0:15\n", "51", [1, 2])
