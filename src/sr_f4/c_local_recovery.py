"""Recover only the diagnosed, pre-model C-local missing-resource incident."""

from __future__ import annotations

import argparse
import copy
import sys
from contextlib import ExitStack
from pathlib import Path

from .c_local import require_frozen, task_matrix
from .common import now, read, write
from .deployment import check_imports
from .pool import ACCOUNT, QOS, Slurm, _locked, _save, _task_lease

RECEIPT = "RECOVERY_RUNTIME_RESOURCES.json"
MISSING = "docs/sr_f1/package/PACKAGE_SHA256.json"


def recover(root, job_id, revision, *, slurm=None):
    """Fail closed on live leases, other errors, new attempts or model output."""
    root, slurm = Path(root), slurm or Slurm()
    require_frozen(root)
    receipt_path = root / RECEIPT
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt.get("job_id") != job_id or receipt.get("revision") != revision:
            raise ValueError("Different resource recovery already registered")
        if receipt.get("status") == "APPLIED":
            return receipt
    imported = check_imports(Path(__file__).resolve().parents[2], sys.executable)
    if job_id in slurm.user_job_ids("varun024") or job_id in slurm.queue():
        raise RuntimeError("Original allocation is still queued or running")
    terminal = slurm.terminal(job_id)
    if not terminal or any(
        terminal.get(k) != v
        for k, v in {
            "job_id": job_id,
            "state": "FAILED",
            "user": "varun024",
            "qos": QOS,
            "account": ACCOUNT,
            "exit_code": "1:0",
        }.items()
    ):
        raise PermissionError("Original allocation terminal identity differs")
    # This incident happened before runtime construction. Do not generalize it
    # to checkpoint or partially generated tasks.
    for name in ("accounting", "runtime_identity", "EVAL", "tasks"):
        if any(p.is_file() for p in (root / name).rglob("*")):
            raise ValueError("Model output or completion evidence exists")
    if any(p.is_file() and p.name != "run_identity.json" for p in (root / "RUNS").rglob("*")):
        raise ValueError("Training output beyond initial identity exists")
    expected = {t["task_id"] for t in task_matrix() if not t["dependencies"]}
    with _locked(root) as state, ExitStack() as leases:
        jobs = [j for j in state["jobs"].values() if j.get("job_id") == job_id]
        if len(jobs) != 1 or jobs[0].get("user") != "varun024":
            raise PermissionError("Original job is not this pool's allocation")
        if any(
            j["status"] in ("SUBMITTING", "SUBMISSION_UNKNOWN", "RUNNING", "SUBMITTED")
            for j in state["jobs"].values()
        ):
            raise RuntimeError("Another allocation or reservation is active")
        touched = {k for k, v in state["tasks"].items() if v["attempts"]}
        if touched != expected:
            raise ValueError("Incident does not match the initial ready task set")
        for key in sorted(expected):
            row = state["tasks"][key]
            attempts = row["attempts"]
            if (
                len(attempts) != 1
                or attempts[0].get("status") != "FAILED"
                or attempts[0].get("owner", {}).get("job_id") != job_id
                or attempts[0].get("returncode") != 1
                or attempts[0].get("complete_receipt") is not False
            ):
                raise ValueError("Not the original failed pre-model attempt")
            recovered = any(r.get("recovery") == RECEIPT for r in row.get("recoveries", []))
            if row["status"] != "FAILED" and not (row["status"] == "PENDING" and recovered):
                raise ValueError("Task is not a diagnosed failure")
            log = root / "pool/logs" / f"{key}.attempt-001.log"
            content = log.read_text()
            error = (
                "FileNotFoundError: [Errno 2] No such file or directory: "
                f"'{jobs[0]['code']}/{MISSING}'"
            )
            if (
                content.splitlines()[-1] != error
                or "from sr_f3_ce.training import fresh_optimizer" not in content
                or 'reference_module("semantic_contract")' not in content
            ):
                raise ValueError("Attempt has a different failure")
            lease = _task_lease(root, key)
            if lease is None:
                raise RuntimeError("Task lease is still held")
            leases.enter_context(lease)
        receipt = (
            read(receipt_path)
            if receipt_path.exists()
            else dict(
                status="PREPARED",
                at=now(),
                job_id=job_id,
                revision=revision,
                imports=imported,
                terminal=terminal,
                before=copy.deepcopy(state),
                tasks=sorted(expected),
                model_calls=0,
                unchanged="Frozen inputs, budget, seeds, runtime algorithms "
                "and all original attempts",
            )
        )
        write(receipt_path, receipt)
        for key in sorted(expected):
            row = state["tasks"][key]
            if row["status"] == "FAILED":
                row.setdefault("recoveries", []).append(
                    dict(
                        at=now(),
                        recovery=RECEIPT,
                        failed_attempt=1,
                        terminal=terminal,
                        reason=row.get("reason"),
                    )
                )
                row.update(status="PENDING", reason="Missing deployed runtime resources restored")
                row.pop("owner", None)
        jobs[0].update(status="TERMINAL", scheduler_terminal=terminal)
        _save(root, state)
        receipt.update(status="APPLIED", applied_at=now())
        write(receipt_path, receipt)
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    result = recover(args.root, args.job_id, args.revision)
    print(dict(status=result["status"], tasks=len(result["tasks"])))
