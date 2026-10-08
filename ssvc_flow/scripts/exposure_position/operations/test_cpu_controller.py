"""Pure metadata tests: SQLite/scheduler text only, no tasks, truth or model calls."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "ser_j23_operations_controller", HERE / "cpu_controller.py"
)
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="controller-metadata-")
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.ids = ["301", "302"]
        jobs = [{"id": "train.S96.0.A3", "kind": "train"}, {"id": "eval.S96.0.A3", "kind": "eval"}]
        frozen = {"status": "FROZEN", "phase_id": "SER_J23_20261008", "plan_hash": "synthetic-plan"}
        json_file(self.run / "FROZEN_PLAN.json", frozen)
        registry = {
            "identity": "synthetic-registered-identity",
            "plan_hash": controller.digest_json(frozen),
            "jobs": jobs,
        }
        json_file(self.run / "REGISTERED_MATRIX.json", registry)
        with sqlite3.connect(self.run / "queue.sqlite") as connection:
            connection.executescript(
                "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);"
                "CREATE TABLE jobs(id TEXT PRIMARY KEY,payload TEXT,status TEXT,"
                "worker TEXT,started REAL,finished REAL);"
            )
            connection.execute("INSERT INTO metadata VALUES('identity',?)", (registry["identity"],))
            for job in jobs:
                connection.execute(
                    "INSERT INTO jobs VALUES(?,?,'PENDING',NULL,NULL,NULL)",
                    (job["id"], json.dumps(job)),
                )
        self.config, self.session = controller.start_session(
            self.run, self.root / "receipts", "attempt-1", self.ids, 60
        )

    def tearDown(self):
        self.temporary.cleanup()

    def state(self, identifier, state, worker=None):
        with sqlite3.connect(self.run / "queue.sqlite") as connection:
            connection.execute(
                "UPDATE jobs SET status=?,worker=? WHERE id=?", (state, worker, identifier)
            )

    def runner(self, *, states=None, present=(), failure=None, malformed=False):
        states = states or {identifier: "RUNNING" for identifier in self.ids}

        def invoke(command, _timeout):
            name = command[0]
            if failure == name:
                return dict(
                    returncode=1,
                    stdout=b"",
                    stderr=b"scheduler temporarily unavailable",
                    error_type=None,
                    elapsed_seconds=0.001,
                )
            if name == "sacct":
                text = (
                    "\n".join(
                        identifier
                        + "|"
                        + state
                        + "|0:0|60|cpu=4,gres/gpu=1|2026-10-08T10:00:00|"
                        + ("2026-10-08T10:01:00" if state in controller.TERMINAL else "Unknown")
                        for identifier, state in states.items()
                    )
                    + "\n"
                )
                if malformed:
                    text += "malformed|row\n"
            else:
                text = "".join(identifier + "|RUNNING\n" for identifier in present)
            return dict(
                returncode=0,
                stdout=text.encode(),
                stderr=b"",
                error_type=None,
                elapsed_seconds=0.001,
            )

        return invoke

    def test_normal_complete_is_candidate_only_and_preserves_sqlite(self):
        for identifier in ("train.S96.0.A3", "eval.S96.0.A3"):
            self.state(identifier, "COMPLETE")
        before = controller.file_sha(self.run / "queue.sqlite")
        event = controller.observe(
            self.config, self.session, 0, runner=self.runner(present=self.ids)
        )
        self.assertEqual(event["status"], "SCIENTIFIC_MATRIX_COMPLETE_CANDIDATE")
        self.assertFalse(event["release_performed"])
        self.assertFalse(event["scheduler_accounting_certified"])
        self.assertFalse((self.run / "STOP").exists())
        self.assertFalse((self.run / "RELEASE_RECEIPT.json").exists())
        self.assertEqual(before, controller.file_sha(self.run / "queue.sqlite"))

    def test_blocked_science_writes_stop_then_waits_without_mutation(self):
        self.state("train.S96.0.A3", "BLOCKED_TECHNICAL", "host:301:777")
        before = controller.file_sha(self.run / "queue.sqlite")
        event = controller.observe(
            self.config, self.session, 0, runner=self.runner(present=self.ids)
        )
        self.assertEqual(event["status"], "STOP_REQUESTED_DRAINING")
        stop = controller.read_json(self.run / "STOP")
        self.assertEqual(stop["reasons"][0]["scientific_status"], "BLOCKED_TECHNICAL")
        self.assertTrue(stop["inflight_tasks_may_finish"])
        self.assertFalse(stop["automatic_retry_permitted"])
        self.assertEqual(before, controller.file_sha(self.run / "queue.sqlite"))
        event = controller.observe(
            self.config,
            self.session,
            1,
            runner=self.runner(states={i: "COMPLETED" for i in self.ids}),
        )
        self.assertEqual(event["status"], "STOPPED_WORKER_ALLOCATIONS_TERMINAL")
        self.assertFalse(event["stop"]["created"])

    def test_scientific_unknown_is_stop_even_when_scheduler_query_fails(self):
        self.state("train.S96.0.A3", "UNKNOWN", "host:301:777")
        event = controller.observe(
            self.config, self.session, 0, runner=self.runner(failure="sacct")
        )
        self.assertEqual(event["status"], "STOP_REQUESTED_DRAINING")
        self.assertTrue((self.run / "STOP").exists())
        self.assertEqual(event["stop_reasons"][0]["kind"], "SCIENTIFIC_TASK_REQUIRES_OPERATOR")
        self.assertFalse(event["scheduler"]["301"]["terminal_verified"])

    def test_scheduler_failure_does_not_infer_running_worker_dead(self):
        self.state("train.S96.0.A3", "RUNNING", "host:301:777")
        event = controller.observe(
            self.config,
            self.session,
            0,
            runner=self.runner(states={i: "COMPLETED" for i in self.ids}, failure="squeue"),
        )
        self.assertEqual(event["status"], "OBSERVING")
        self.assertFalse((self.run / "STOP").exists())
        self.assertTrue(all(row["observation"] == "UNKNOWN" for row in event["scheduler"].values()))
        self.assertIn(
            b"temporarily unavailable",
            (self.session / "polls/000000/squeue.stderr.txt").read_bytes(),
        )

    def test_verified_dead_allocation_with_running_task_requests_stop(self):
        self.state("train.S96.0.A3", "RUNNING", "host:301:777")
        event = controller.observe(
            self.config,
            self.session,
            0,
            runner=self.runner(states={"301": "NODE_FAIL", "302": "COMPLETED"}),
        )
        self.assertEqual(event["status"], "STOPPED_WORKER_ALLOCATIONS_TERMINAL")
        self.assertEqual(event["stop_reasons"][0]["slurm_job_id"], "301")
        self.assertEqual(
            event["stop_reasons"][0]["kind"], "TERMINAL_ALLOCATION_WITH_RUNNING_SCIENTIFIC_TASK"
        )
        with sqlite3.connect(self.run / "queue.sqlite") as connection:
            self.assertEqual(
                connection.execute("SELECT status FROM jobs WHERE id='train.S96.0.A3'").fetchone()[
                    0
                ],
                "RUNNING",
            )

    def test_squeue_presence_overrides_terminal_sacct_observation(self):
        self.state("train.S96.0.A3", "RUNNING", "host:301:777")
        event = controller.observe(
            self.config,
            self.session,
            0,
            runner=self.runner(states={i: "COMPLETED" for i in self.ids}, present=["301"]),
        )
        self.assertEqual(event["status"], "OBSERVING")
        self.assertFalse((self.run / "STOP").exists())
        self.assertFalse(event["scheduler"]["301"]["terminal_verified"])

    def test_custom_worker_uses_persisted_attempt_allocation(self):
        self.state("train.S96.0.A3", "RUNNING", "custom-worker")
        json_file(
            self.run / "worker_attempts/one/STARTED.json",
            dict(job_id="train.S96.0.A3", worker="custom-worker", slurm_job_id="301"),
        )
        event = controller.observe(
            self.config, self.session, 0, runner=self.runner(states={i: "FAILED" for i in self.ids})
        )
        self.assertEqual(event["running_task_allocation_mapping"]["train.S96.0.A3"], "301")
        self.assertTrue((self.run / "STOP").exists())

    def test_existing_stop_is_preserved_and_sessions_polls_refuse_overwrite(self):
        self.state("train.S96.0.A3", "UNKNOWN")
        controller.write_once_bytes(self.run / "STOP", b"original reason\n")
        event = controller.observe(self.config, self.session, 0, runner=self.runner())
        self.assertFalse(event["stop"]["created"])
        self.assertEqual((self.run / "STOP").read_bytes(), b"original reason\n")
        with self.assertRaises(FileExistsError):
            controller.observe(self.config, self.session, 0, runner=self.runner())
        with self.assertRaises(FileExistsError):
            controller.start_session(self.run, self.root / "receipts", "attempt-1", self.ids, 60)
        with self.assertRaises(FileExistsError):
            controller.write_once_bytes(self.run / "STOP", b"original reason\n")

    def test_registered_metadata_mutation_prevents_complete_candidate(self):
        data = controller.read_json(self.run / "REGISTERED_MATRIX.json")
        data["jobs"].pop()
        json_file(self.run / "REGISTERED_MATRIX.json", data)
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            controller.observe(self.config, self.session, 0, runner=self.runner())
        self.assertFalse((self.run / "STOP").exists())

    def test_malformed_scheduler_output_does_not_prove_termination(self):
        self.state("train.S96.0.A3", "RUNNING", "host:301:777")
        event = controller.observe(
            self.config,
            self.session,
            0,
            runner=self.runner(states={i: "COMPLETED" for i in self.ids}, malformed=True),
        )
        self.assertEqual(event["status"], "OBSERVING")
        self.assertFalse((self.run / "STOP").exists())

    def test_cpu_and_allocation_input_guards(self):
        controller.require_cpu_allocation({"SLURM_GPUS": "0"})
        with self.assertRaises(PermissionError):
            controller.require_cpu_allocation({"SLURM_JOB_GPUS": "0"})
        for ids in ([], ["1", "1"], ["123.batch"], ["123_4"], ["123;scancel"], ["0"]):
            with self.assertRaises(ValueError):
                controller.allocation_ids(ids)

    def test_completion_during_scheduler_query_does_not_create_stale_stop(self):
        self.state("train.S96.0.A3", "RUNNING", "host:301:777")
        underlying = self.runner(states={i: "COMPLETED" for i in self.ids})

        def finish_while_querying(command, timeout):
            if command[0] == "sacct":
                for identifier in ("train.S96.0.A3", "eval.S96.0.A3"):
                    self.state(identifier, "COMPLETE")
            return underlying(command, timeout)

        event = controller.observe(self.config, self.session, 0, runner=finish_while_querying)
        self.assertEqual(event["status"], "SCIENTIFIC_MATRIX_COMPLETE_CANDIDATE")
        self.assertFalse((self.run / "STOP").exists())

    def test_pending_matrix_without_live_workers_is_incomplete_not_success(self):
        event = controller.observe(
            self.config,
            self.session,
            0,
            runner=self.runner(states={i: "COMPLETED" for i in self.ids}),
        )
        self.assertEqual(event["status"], "INCOMPLETE_NO_LIVE_WORKER_ALLOCATIONS")
        self.assertFalse(event["all_scientific_complete"])
        self.assertFalse((self.run / "STOP").exists())


if __name__ == "__main__":
    unittest.main()
