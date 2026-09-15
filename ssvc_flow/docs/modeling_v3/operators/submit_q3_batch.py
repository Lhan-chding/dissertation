#!/usr/bin/env python3
"""Explicitly submit one previously sealed Q3 CPU batch; never advance a batch.

Usage on the server, after reviewing the prepared artifacts:
  python submit_q3_batch.py --prepared-manifest /.../PREPARATION_MANIFEST.json \
    --sha256 RECORDED_PREPARATION_SHA256 --batch-index 0

For later batches also supply --previous-review PATH and
--previous-review-sha256 SHA256 for an already reviewed result receipt. This
script records that attestation; it does not create a scientific decision.
Any existing batch directory, including an uncertain submission attempt, blocks
another submission. Investigate the actual scheduler state instead of retrying.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
import pwd
import re
import subprocess
import sys
from pathlib import Path

SERVER_BASE = Path("/projects/varunssd/louis-ssvc/modeling_v3_20260915")
QOS = "soujanya-poria-startfund-2026-03"
MAX_SUBMITTED_ELEMENTS = 5


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def current_user():
    return pwd.getpwuid(os.getuid()).pw_name


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def _check_hash(path, expected):
    if (
        not isinstance(expected, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        or Path(path).is_symlink()
        or not Path(path).is_file()
        or sha256_file(path) != expected
    ):
        raise ValueError(f"original path/hash mismatch: {path}")


def _environment(sealed):
    # Do not inherit SBATCH_*, SLURM_*, PYTHONPATH, loader overrides or GPU flags.
    context = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TZ")
        if key in os.environ
    }
    return {**context, **sealed}


def _bytes(value):
    if value is None:
        return b""
    return value if isinstance(value, bytes) else value.encode()


def _probe(argv, *, run, env, cwd, timeout=120):
    result = run(argv, env=env, cwd=str(cwd), capture_output=True, timeout=timeout, check=False)
    record = {
        "argv": argv,
        "returncode": result.returncode,
        "stdout": _bytes(result.stdout).decode("utf-8"),
        "stderr": _bytes(result.stderr).decode("utf-8"),
    }
    if result.returncode != 0:
        raise ValueError("read-only preflight failed: " + json.dumps(record))
    return record


def queue_elements(stdout, username):
    """Count expanded array elements, never compressed array lines."""
    elements = []
    for line in _bytes(stdout).decode("utf-8").splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split("|")]
        if len(values) != 4:
            raise ValueError("squeue element inventory has an unexpected format")
        job, user, qos, state = values
        if (
            re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job) is None
            or user != username
            or qos != QOS
            or not state
            or job in elements
        ):
            raise ValueError("squeue must return unique expanded elements for this user/QOS")
        elements.append(job)
    return elements


def _completed_elements(stdout, job_id, indices):
    expected = {f"{job_id}_{index}" for index in indices}
    found = set()
    for line in _bytes(stdout).decode("utf-8").splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split("|")]
        if len(values) != 3:
            raise ValueError("sacct completion inventory has an unexpected format")
        job, state, code = values
        if "." in job:
            continue  # Job steps do not represent array elements.
        if job != job_id and job not in expected:
            raise ValueError("previous Slurm job has unexpected array elements")
        if state != "COMPLETED" or code != "0:0":
            raise ValueError("previous Slurm batch has a failed or unfinished allocation")
        if job in expected:
            if job in found:
                raise ValueError("duplicate previous Slurm array element")
            found.add(job)
    if found != expected:
        raise ValueError("previous Slurm batch completion is missing array elements")


def _verify_complete(operator, row, prepared, snapshot_files):
    root = Path(row["output"])
    receipt_path = root / "COMPLETE.json"
    receipt = _read_json(receipt_path)
    files = receipt.get("files")
    if (
        root.is_symlink()
        or receipt_path.is_symlink()
        or receipt.get("status") != "COMPLETE"
        or not isinstance(files, dict)
        or not files
        or receipt.get("summary", {}).get("stage") != "Q3"
        or receipt.get("summary", {}).get("status") != "COLLECTED_AND_MEASURED"
    ):
        raise ValueError("previous output requires an actual completed Q3 campaign")
    binding = receipt["binding"]
    config = _read_json(prepared["binding"]["config_path"])
    expected = {
        "config_sha256": canonical_hash(config),
        "stage": "Q3_FROZEN_CAMPAIGN",
        "selection_hash": prepared["binding"]["selection_hash"],
        "seed_subset": [row["seed"]],
        "arm_subset": config["cpu"]["arms"],
        "roles": ["interval_calibration"],
    }
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError("previous campaign seed/role/config/selection binding differs")
    sources = binding.get("source_hashes")
    if not sources or any(
        snapshot_files.get("ssvc_flow/" + relative) != digest
        for relative, digest in sources.items()
    ):
        raise ValueError("previous campaign source binding differs from the verified snapshot")
    for relative, digest in files.items():
        _check_hash(operator.safe_file(root, relative), digest)
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path != receipt_path and ".pending-" not in path.name
    }
    if actual != set(files):
        raise ValueError("previous campaign original inventory differs")
    return {"path": str(receipt_path), "sha256": sha256_file(receipt_path), "seed": row["seed"]}


@contextlib.contextmanager
def _locked_submission(root):
    path = root / f".q3_cpu_submit_{os.getuid()}.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def _seal_bytes(path, payload):
    with Path(path).open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def submit_batch(
    prepared_manifest,
    prepared_sha256,
    batch_index,
    *,
    previous_review=None,
    previous_review_sha256=None,
    run=None,
):
    """Execute exactly one sealed sbatch after all read-only gates succeed."""
    if sys.platform != "linux":
        raise ValueError("Q3 submission requires the Linux server")
    if type(batch_index) is not int or batch_index < 0:
        raise ValueError("batch index must be a nonnegative integer")
    run = subprocess.run if run is None else run
    _check_hash(prepared_manifest, prepared_sha256)
    path = Path(prepared_manifest).resolve()
    prepared = _read_json(path)
    binding = prepared["binding"]
    candidate = Path(binding["candidate_root"]).resolve()
    checkout = Path(binding["checkout"]).resolve()
    bundle = candidate / "Q3_OPERATOR_INTERVAL_CALIBRATION"
    if (
        path != bundle / "PREPARATION_MANIFEST.json"
        or candidate.parent != SERVER_BASE.resolve()
        or not re.fullmatch(r"candidate[0-9]+", candidate.name)
        or checkout != candidate / "checkout"
        or bundle.is_symlink()
        or prepared.get("schema") != "ssvc-v3-q3-operator-preparation-1"
        or prepared.get("status") != "PREPARED_NOT_SUBMITTED"
        or prepared.get("request_count") != 20
    ):
        raise ValueError("a complete original server Q3 preparation is required")
    operator_path = Path(__file__).resolve().with_name("prepare_q3_operator.py")
    _check_hash(operator_path, prepared["operator_script_sha256"])
    spec = importlib.util.spec_from_file_location("_q3_verified_operator", operator_path)
    operator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(operator)
    initial_env = _environment(operator.CPU_ENV)
    verify = _probe(
        [
            str(operator.SERVER_PYTHON),
            str(operator_path),
            "--mode",
            "verify",
            "--candidate-root",
            str(candidate),
            "--checkout",
            str(checkout),
            "--snapshot-sha256",
            binding["snapshot_manifest_sha256"],
            "--inventory-basis",
            binding["inventory_basis_path"],
            "--inventory-basis-sha256",
            binding["inventory_basis_sha256"],
            "--prepared-sha256",
            prepared_sha256,
        ],
        run=run,
        env=initial_env,
        cwd=checkout / "ssvc_flow",
        timeout=None,  # Hash all frozen development originals without an arbitrary time cap.
    )
    if json.loads(verify["stdout"]).get("status") != "VERIFIED_NOT_SUBMITTED":
        raise ValueError("original preparation operator did not verify the sealed bundle")
    # Recheck payload hashes locally as well as through the original operator.
    _check_hash(path, prepared_sha256)
    for relative, digest in prepared["files"].items():
        _check_hash(operator.safe_file(bundle, relative), digest)
    list_path = bundle / "REQUEST_LIST.json"
    list_hash = sha256_file(list_path)
    batches = _read_json(bundle / "SUBMISSION_INTENTS.json")
    expected = operator.submission_intents(checkout, candidate, bundle, list_hash)
    if batches != expected:
        raise ValueError("sealed submission argv/environment differs from the CPU-only template")
    if batch_index >= len(batches):
        raise ValueError("batch index is outside the complete sealed list")
    batch = batches[batch_index]
    config = _read_json(binding["config_path"])
    campaign = candidate / "Q3_campaign"
    expected_rows = operator.calibration_requests(
        config,
        binding["selection_lock_path"],
        binding["config_path"],
        campaign,
        [*binding["historical_roots"], campaign],
    )
    rows = _read_json(bundle / "CALIBRATION_MATRIX.json")["rows"]
    if rows != expected_rows:
        raise ValueError("calibration request identities differ from the frozen role matrix")
    requests = _read_json(list_path)
    if len(requests) != len(rows):
        raise ValueError("request list and calibration matrix differ")
    for row, request in zip(rows, requests, strict=True):
        expected_path = bundle / "requests" / f"{row['index']:02d}_seed{row['seed']}.json"
        if Path(request["path"]) != expected_path:
            raise ValueError("request paths differ from their sealed ordered indices")
        _check_hash(expected_path, request["sha256"])
        if _read_json(expected_path) != row["request"]:
            raise ValueError("sealed CPU request changed its seed/output/command")
    submissions = candidate / "Q3_SUBMISSIONS"
    destination = submissions / f"batch_{batch_index:02d}"
    if submissions.is_symlink() or destination.exists() or destination.is_symlink():
        raise FileExistsError("existing submission intent/receipt or unsafe submission directory")
    selected_rows = [rows[index] for index in batch["indices"]]
    if campaign.is_symlink() or any(
        Path(row["output"]).exists() or Path(row["output"]).is_symlink() for row in selected_rows
    ):
        raise FileExistsError("current batch already has an output; new submission is forbidden")
    env = _environment(batch["environment"])
    prior = None
    if batch_index == 0:
        if previous_review is not None or previous_review_sha256 is not None:
            raise ValueError("first batch has no previous result review")
    else:
        if previous_review is None or previous_review_sha256 is None:
            raise ValueError("later batches require an already reviewed result receipt path/hash")
        _check_hash(previous_review, previous_review_sha256)
        if not Path(previous_review).read_text().strip():
            raise ValueError("previous result review cannot be empty")
        previous = batches[batch_index - 1]
        previous_dir = submissions / f"batch_{batch_index - 1:02d}"
        previous_receipt = _read_json(previous_dir / "RECEIPT.json")
        if (
            previous_receipt.get("status") != "SUBMITTED"
            or previous_receipt.get("prepared_sha256") != prepared_sha256
            or previous_receipt.get("batch_index") != batch_index - 1
            or previous_receipt.get("indices") != previous["indices"]
            or re.fullmatch(r"[0-9]+", str(previous_receipt.get("job_id", ""))) is None
        ):
            raise ValueError("previous batch has no verified successful scheduler submission")
        _check_hash(previous_dir / "INTENT.json", previous_receipt["intent_sha256"])
        previous_intent = _read_json(previous_dir / "INTENT.json")
        if (
            previous_receipt.get("argv") != previous["argv"]
            or previous_intent.get("argv") != previous["argv"]
            or previous_intent.get("prepared_sha256") != prepared_sha256
            or previous_intent.get("indices") != previous["indices"]
            or previous_intent.get("batch_index") != batch_index - 1
        ):
            raise ValueError("previous submission receipt differs from its sealed intent")
        for stream_name in ("stdout", "stderr"):
            stream_path = previous_dir / ("sbatch." + stream_name)
            if previous_receipt[stream_name]["path"] != str(stream_path):
                raise ValueError("previous submission stream escaped its original directory")
            _check_hash(stream_path, previous_receipt[stream_name]["sha256"])
        submitted_id = (previous_dir / "sbatch.stdout").read_text().strip().split(";")[0]
        if submitted_id != previous_receipt["job_id"]:
            raise ValueError("previous job id differs from the actual sbatch response")
        snapshot = _read_json(checkout / "CODE_SNAPSHOT_MANIFEST.json")
        snapshot_files = snapshot.get("files", snapshot)
        completed = [
            _verify_complete(operator, rows[index], prepared, snapshot_files)
            for index in previous["indices"]
        ]
        accounting = _probe(
            [
                "sacct",
                "--allocations",
                "--noheader",
                "--parsable2",
                "--jobs",
                previous_receipt["job_id"],
                "--format=JobID%64,State%40,ExitCode",
            ],
            run=run,
            env=env,
            cwd=candidate,
        )
        _completed_elements(accounting["stdout"], previous_receipt["job_id"], previous["indices"])
        prior = {
            "receipt": {
                "path": str(previous_dir / "RECEIPT.json"),
                "sha256": sha256_file(previous_dir / "RECEIPT.json"),
            },
            "completed_outputs": completed,
            "scheduler_accounting": accounting,
            "operator_review": {
                "path": str(Path(previous_review).resolve()),
                "sha256": previous_review_sha256,
                "scope": "OPERATOR_SUPPLIED_REVIEW_NOT_AUTOMATIC_SCIENTIFIC_DECISION",
            },
        }
    # Serialize these operator invocations; unrelated submissions still race and
    # Slurm remains the authority enforcing QOS limits. A rejection is preserved.
    with _locked_submission(candidate.parent):
        if destination.exists() or any(Path(row["output"]).exists() for row in selected_rows):
            raise FileExistsError("batch was submitted or produced output during preflight")
        username = current_user()
        queue = _probe(
            [
                "squeue",
                "--array",
                "--noheader",
                "--user",
                username,
                "--qos",
                QOS,
                "--format=%i|%u|%q|%T",
            ],
            run=run,
            env=env,
            cwd=candidate,
        )
        elements = queue_elements(queue["stdout"], username)
        if len(elements) + len(batch["indices"]) > MAX_SUBMITTED_ELEMENTS:
            raise ValueError(
                "ordinary QOS MaxSubmitPU=5 has insufficient expanded-element capacity"
            )
        submissions.mkdir(exist_ok=True)
        destination.mkdir()  # exclusive reservation; failures are never silently retried
        intent = {
            "schema": "ssvc-v3-q3-batch-submit-intent-1",
            "status": "INTENT_FROZEN_BEFORE_SCHEDULER_CALL",
            "batch_index": batch_index,
            "indices": batch["indices"],
            "prepared_manifest": str(path),
            "prepared_sha256": prepared_sha256,
            "request_list_sha256": list_hash,
            "argv": batch["argv"],
            "environment": batch["environment"],
            "process_environment": env,
            "requests": [requests[i] for i in batch["indices"]],
            "outputs": [row["output"] for row in selected_rows],
            "preparation_verification": verify,
            "previous_batch": prior,
            "queue_snapshot": queue,
            "counted_queue_elements": elements,
            "qos_max_submit_pu": MAX_SUBMITTED_ELEMENTS,
            "username": username,
            "submitter_sha256": sha256_file(__file__),
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "next_batch_automatic": False,
            "scientific_decision_created": False,
        }
        operator.seal_json(destination / "INTENT.json", intent)
        directory_fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        # An ENOSPC/EDQUOT/fsync failure above prevents any scheduler call.
        stdout, stderr, error, code, job_id = b"", b"", None, None, None
        status = "SUBMISSION_OUTCOME_UNKNOWN"
        try:
            result = run(
                batch["argv"],
                env=env,
                cwd=str(candidate),
                capture_output=True,
                timeout=120,
                check=False,
            )
            stdout, stderr, code = _bytes(result.stdout), _bytes(result.stderr), result.returncode
            match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?", stdout.decode().strip())
            if code == 0 and match:
                status, job_id = "SUBMITTED", match.group(1)
            elif code != 0 and not match:
                status = "SUBMISSION_REJECTED"
        except BaseException as exc:
            stdout = _bytes(getattr(exc, "stdout", None))
            stderr = _bytes(getattr(exc, "stderr", None))
            error = {"type": type(exc).__name__, "message": str(exc)}
        _seal_bytes(destination / "sbatch.stdout", stdout)
        _seal_bytes(destination / "sbatch.stderr", stderr)
        receipt = {
            "schema": "ssvc-v3-q3-batch-submit-receipt-1",
            "status": status,
            "batch_index": batch_index,
            "indices": batch["indices"],
            "prepared_sha256": prepared_sha256,
            "job_id": job_id,
            "argv": batch["argv"],
            "returncode": code,
            "exception": error,
            "intent_sha256": sha256_file(destination / "INTENT.json"),
            "stdout": {
                "path": str(destination / "sbatch.stdout"),
                "sha256": sha256_file(destination / "sbatch.stdout"),
            },
            "stderr": {
                "path": str(destination / "sbatch.stderr"),
                "sha256": sha256_file(destination / "sbatch.stderr"),
            },
            "next_batch_automatic": False,
        }
        operator.seal_json(destination / "RECEIPT.json", receipt)
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--batch-index", required=True, type=int)
    parser.add_argument("--previous-review")
    parser.add_argument("--previous-review-sha256")
    args = parser.parse_args(argv)
    result = submit_batch(
        args.prepared_manifest,
        args.sha256,
        args.batch_index,
        previous_review=args.previous_review,
        previous_review_sha256=args.previous_review_sha256,
    )
    print(json.dumps(result))
    return 0 if result["status"] == "SUBMITTED" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from None
