#!/usr/bin/env python3
"""Scheduling-only overlay; workers and frozen production source stay unchanged."""

import argparse
import os
from pathlib import Path

from sr_f12.orchestration import (
    Controller, TEACHER_QOS, file_hash, read_json, run_controller, save_json,
    scientific_matrix, teacher_usage,
)

FROZEN_FILES = {
    "SCIENCE_FREEZE.json", "config/SR_F1_2_FROZEN.json",
    "TECHNICAL_CHECK_SR_F1_2.json", "SEMANTIC_PREDICTIONS.json",
}


def submission_usage(queue, attempts):
    actual = sum(row.get("QOS") == TEACHER_QOS for row in queue.values())
    reserved = sum(
        row["status"] not in ("COMPLETE", "FAILED") and row.get("job_id") not in queue
        for row in attempts
    )
    return actual + reserved


class RollingController(Controller):
    """Use the original submission, release, identity and terminal-state gates."""

    def identity(self):
        source_hash = super().identity()
        auth = read_json(self.authorization)
        if (
            auth["policy"] != "REGISTERED_MATRIX_ROLLING_CAPACITY"
            or auth["run_root"] != str(self.root)
            or auth["production_source_sha256"] != source_hash
            or auth["overlay_sha256"] != file_hash(Path(__file__))
            or auth["max_submit_pu"] != 5
            or set(auth["frozen_file_sha256"]) != FROZEN_FILES
        ):
            raise PermissionError("Rolling scheduling authorization differs")
        for name, checksum in auth["frozen_file_sha256"].items():
            if file_hash(self.root / name) != checksum:
                raise PermissionError("Frozen scheduling dependency changed: " + name)
        return source_hash

    def submit(self, key, queue):
        if submission_usage(queue, self.attempts()) >= 5:
            return False
        return super().submit(key, queue)

    def tick(self):
        snapshot = super().tick()
        if snapshot["stage"] != "MAIN_TRAINING":
            return snapshot
        changed = False
        for run in scientific_matrix():
            states = {row["task_id"]: row for row in self.attempts()}
            if any(row["status"] in ("FAILED", "SUBMISSION_UNKNOWN", "RELEASE_UNKNOWN")
                   for row in states.values()):
                break
            key = run["model_id"]
            if key in states:
                continue
            queue = self.slurm.queue(self.user)
            if teacher_usage(queue, list(states.values())) >= 5:
                break
            if submission_usage(queue, list(states.values())) >= 5:
                break
            changed = True
            if not self.submit(key, queue):
                break
        # Refresh after dispatch so the ordinary controller exposes unknown/failure states.
        if changed:
            snapshot = super().tick()
        snapshot["scheduling_policy"] = "REGISTERED_MATRIX_ROLLING_CAPACITY"
        snapshot["scheduler_job_id"] = os.environ.get("SLURM_JOB_ID")
        snapshot["scheduling_authorization"] = str(self.authorization)
        save_json(self.directory / "STATE.json", snapshot)
        return snapshot


def main():
    parser = argparse.ArgumentParser()
    for name in ("run-root", "code-root", "model-path", "source-manifest", "authorization"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    import sr_f12.orchestration as production
    if Path(production.__file__).resolve() != args.code_root.resolve() / "src/sr_f12/orchestration.py":
        raise PermissionError("Overlay must import the registered production controller")
    controller = RollingController(
        args.run_root, args.code_root, args.model_path, args.source_manifest,
        args.source_commit, python=args.python,
    )
    controller.authorization = args.authorization.resolve(strict=True)
    controller.identity()
    result = run_controller(controller)
    if result["stage"].startswith("BLOCKED_"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
