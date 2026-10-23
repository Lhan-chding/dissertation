"""Repair the missing scheduler resume flag without changing the pilot freeze."""

from __future__ import annotations

import argparse
from pathlib import Path

from .c_pilot import pool_task_matrix, require_frozen, task_matrix
from .common import now, read, write
from .pool import ACCOUNT, QOS, Slurm, _locked, _save, _task_lease

RECEIPT = "RECOVERY_TIMEOUT_RESUMPTION.json"


def enable_resumption(root, revision):
    root = Path(root)
    require_frozen(root)
    expected = {t["task_id"]: t for t in task_matrix()}
    amended = {t["task_id"]: t for t in pool_task_matrix()}
    with _locked(root) as state:
        if set(state["tasks"]) != set(expected) or any(
            row["spec"] not in (expected[key], amended[key]) for key, row in state["tasks"].items()
        ):
            raise ValueError("Pool differs from the registered pilot task matrix")
        path = root / RECEIPT
        if path.exists():
            receipt = read(path)
            if receipt["revision"] != revision or receipt["status"] not in ("PREPARED", "APPLIED"):
                raise ValueError("Unknown timeout recovery record")
        else:
            receipt = dict(
                status="PREPARED",
                at=now(),
                revision=revision,
                original_specs={k: row["spec"] for k, row in state["tasks"].items()},
                unchanged="TASKS.json, FROZEN.json, CONFIG.json, all model/data/seed/budget "
                "identities, completed statuses, failed attempts, checkpoints and answers",
            )
            write(path, receipt)
        changed = []
        for key, row in state["tasks"].items():
            if row["spec"] != amended[key]:
                row["spec"] = amended[key]
                changed.append(key)
        if changed:
            _save(root, state)
        if receipt["status"] != "APPLIED":
            receipt.update(status="APPLIED", applied_at=now())
            write(path, receipt)
    return changed


def recover_misclassified_timeouts(root, *, slurm=None):
    """Only retry failures already attributed to a verified, released TIMEOUT."""
    root, slurm = Path(root), slurm or Slurm()
    if read(root / RECEIPT)["status"] != "APPLIED":
        raise ValueError("Resume capability amendment must be applied first")
    require_frozen(root)
    with _locked(root) as state:
        candidates = {
            k: row
            for k, row in state["tasks"].items()
            if row["status"] == "FAILED"
            and row.get("reason") == "lease ended: TIMEOUT"
            and row["spec"].get("resume_supported") is True
            and row["attempts"]
            and row["attempts"][-1]["status"] == "FAILED"
            and row["attempts"][-1].get("terminal", {}).get("state") == "TIMEOUT"
        }
        jobs = list(state["jobs"].values())
    queue, terminal, resumed = slurm.queue(), {}, []
    for key, previous in candidates.items():
        attempt = previous["attempts"][-1]
        job = attempt["owner"].get("job_id")
        if not job or job in queue or attempt["terminal"].get("job_id") != job:
            continue
        if job not in terminal:
            terminal[job] = slurm.terminal(job)
        record = terminal[job]
        if not record or record["state"] != "TIMEOUT":
            continue
        registered = [j for j in jobs if j.get("job_id") == job]
        if (
            len(registered) != 1
            or record.get("job_id") != job
            or record.get("qos") != QOS
            or record.get("account") != ACCOUNT
            or record.get("user") != registered[0].get("user")
        ):
            raise PermissionError("Timeout does not match this pool's teacher allocation")
        spec = previous["spec"]
        if spec["kind"] == "c_pilot_train":
            directory = root / "RUNS/C" / f"b{spec['block']}" / spec["arm"]
            progress = directory / "PROGRESS.json"
            if (
                progress.exists()
                and read(progress)["successful_updates"] > 0
                and not (directory / "checkpoints/latest.pt").is_file()
            ):
                raise FileNotFoundError("Committed pilot training checkpoint is missing")
        lease = _task_lease(root, key)
        if lease is None:
            continue
        try:
            with _locked(root) as state:
                row = state["tasks"][key]
                if row != previous:
                    continue
                row.setdefault("recoveries", []).append(
                    dict(
                        at=now(),
                        recovery=RECEIPT,
                        failed_attempt=len(row["attempts"]),
                        terminal=record,
                        reason=row["reason"],
                    )
                )
                row.update(status="PENDING", reason="Resume after TIMEOUT capability repair")
                row.pop("owner", None)
                _save(root, state)
                resumed.append(key)
        finally:
            lease.close()
    return resumed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    changed = enable_resumption(args.root, args.revision)
    resumed = recover_misclassified_timeouts(args.root)
    print(dict(resume_flags_enabled=len(changed), resumed=resumed))


if __name__ == "__main__":
    main()
