"""Resource revision preserves registration and old costs; no real Slurm calls."""

import copy
import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sr_f1.orchestration import (
    ENGINE_MULTIGPU_REPAIR_ID,
    MULTIGPU_CAPACITY_MODE,
    TEACHER_QOS,
    SlurmBackend,
    atomic_json,
    checkpoint_task,
    digest,
    file_hash,
    read_json,
    resource_override,
    verify_multigpu_resources,
)

REPO = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIXTURES = load_module(
    "multigpu_io_fixture", Path(__file__).with_name("test_engine_io_orchestration.py")
)


class MultiGPURecoveryTests(unittest.TestCase):
    def setUp(self):
        fixture = FIXTURES.EngineIORecoveryTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root, self.scheduler, self.backend = fixture.root, fixture.scheduler, fixture.backend
        self.simulation = fixture.simulation
        fixture.authorize()
        self.scheduler.tick()
        self.maintained = self.scheduler.state["tasks"]["ENGINE"]["attempts"][-1]
        self.backend.jobs[self.maintained["job_id"]]["state"] = "CANCELLED"
        self.scheduler.tick()
        self.prior_attempts = copy.deepcopy(self.scheduler.state["tasks"]["ENGINE"]["attempts"])
        self.prior_authorizations = {
            key: copy.deepcopy(self.scheduler.state[key])
            for key in (
                "engine_memory_repair_resume",
                "engine_storage_repair_resume",
                "engine_io_repair_resume",
            )
        }
        atomic_json(
            self.root / "ENGINE_MULTIGPU_REPAIR.json", {"repair_id": ENGINE_MULTIGPU_REPAIR_ID}
        )
        self.repair = {
            "repair_id": ENGINE_MULTIGPU_REPAIR_ID,
            "repair_sha256": file_hash(self.root / "ENGINE_MULTIGPU_REPAIR.json"),
            "source_commit": "d" * 40,
            "original_execution_freeze_sha256": file_hash(self.root / "EXECUTION_FREEZE.json"),
            "gpu_worker_constraint": "highmem",
            "gpu_count": 4,
            "cpus_per_gpu": 4,
            "minimum_gpu_host_memory_gb_per_gpu": 80,
            "worker_source_sha256": file_hash(REPO / "scripts/sr_f1/run_worker.py"),
            "previous_worker_source_sha256": self.scheduler.registration["source_hashes"][
                "scripts/sr_f1/run_worker.py"
            ],
            "maintained_attempt_id": self.maintained["attempt_id"],
            "maintained_job_id": self.maintained["job_id"],
        }
        stub = patch(
            "sr_f1.freeze.verify_engine_multigpu_repair", return_value=self.repair, create=True
        )
        self.verifier = stub.start()
        self.addCleanup(stub.stop)
        self.other_teacher_gpus = 1
        original_submit, original_observe = self.backend.submit, self.backend.observe

        def submit(command):
            try:
                return original_submit(command)
            finally:
                job = self.backend.jobs[str(1000 + len(self.backend.jobs) - 1)]
                job["gpus"] = (
                    int(command[command.index("--gres") + 1].rsplit(":", 1)[1])
                    if "--gres" in command
                    else 0
                )

        def observe(attempt, permission):
            result = original_observe(attempt, permission)
            for row in result["accounting"]:
                count = self.backend.jobs[row["JobIDRaw"]]["gpus"]
                if count > 1:
                    row["AllocTRES"] = f"gres/gpu={count},gres/gpu:pro6000={count}"
            return result

        self.backend.submit, self.backend.observe = submit, observe

        def project_capacity(permission):
            jobs = [{"job_id": "other", "gpus": self.other_teacher_gpus, "state": "RUNNING"}]
            jobs.extend(
                {"job_id": key, "gpus": value["gpus"], "state": value["state"]}
                for key, value in self.backend.jobs.items()
                if value["state"] in {"PENDING", "RUNNING"}
            )
            return {
                "owner": permission["owner"],
                "qos": TEACHER_QOS,
                "jobs": jobs,
                "capacity_mode": MULTIGPU_CAPACITY_MODE,
                "repair_sha256": self.repair["repair_sha256"],
                "maximum_reserved_gpus": 5,
            }

        self.backend.project_capacity = project_capacity
        self.resource_edits = {}

        def detail(command):
            job_id = command[3]
            job = self.backend.jobs[job_id]
            count = job["gpus"]
            tres = (
                f"cpu={4 * count},mem={90 * count}G,node=1,"
                f"gres/gpu={count},gres/gpu:pro6000={count}"
            )
            fields = {
                "JobId": job_id,
                "JobName": job["name"],
                "Comment": job["comment"],
                "UserId": "owner(1)",
                "Account": "rose",
                "QOS": TEACHER_QOS,
                "JobState": "PENDING",
                "Reason": "JobHeldUser",
                "Features": "highmem",
                "NumNodes": "1",
                "ReqTRES": tres,
                "AllocTRES": tres,
            }
            fields.update(self.resource_edits)
            return {
                "returncode": 0,
                "stdout": " ".join(f"{key}={value}" for key, value in fields.items()),
            }

        self.backend._run = Mock(side_effect=detail)

    def authorize(self):
        return self.scheduler.resume_repaired_engine_multigpu()

    def test_once_authorization_preserves_registration_prior_history_and_markers(self):
        hashes = {
            relative: file_hash(self.root / relative)
            for relative in (
                "EXECUTION_FREEZE.json",
                "orchestration/REGISTRATION.json",
                "manifests/ALLOCATION_PERMISSION.json",
            )
        }
        journal = (self.root / "orchestration/journal.jsonl").read_bytes()
        result = self.authorize()
        self.assertEqual(result["permitted_next_attempt_id"], "ENGINE_attempt0004")
        self.assertEqual(self.scheduler.state["tasks"]["ENGINE"]["attempts"], self.prior_attempts)
        self.assertEqual(
            {key: self.scheduler.state[key] for key in self.prior_authorizations},
            self.prior_authorizations,
        )
        self.assertTrue(
            (self.root / "orchestration/journal.jsonl").read_bytes().startswith(journal)
        )
        for relative, checksum in hashes.items():
            self.assertEqual(file_hash(self.root / relative), checksum)
        self.assertEqual(
            [file_hash(path) for path in self.fixture.marker_paths], self.fixture.marker_hashes
        )
        with self.assertRaisesRegex(ValueError, "ALREADY_AUTHORIZED"):
            self.authorize()

    def test_four_gpu_request_and_terminal_cost_are_authentic(self):
        self.authorize()
        self.scheduler.tick()
        attempt = self.scheduler.state["tasks"]["ENGINE"]["attempts"][-1]
        self.assertEqual(attempt["attempt_id"], "ENGINE_attempt0004")
        self.assertTrue(attempt["released"])
        self.assertIn("gpu:pro6000:4", attempt["command"])
        self.assertIn("--cpus-per-task=16", attempt["command"])
        self.assertIn("--nodes=1", attempt["command"])
        self.assertIn("--constraint=highmem", attempt["command"])
        self.assertFalse(any(value.startswith("--mem") for value in attempt["command"]))
        allocation = read_json(self.root / "manifests/ALLOCATIONS.json")["allocations"][-1]
        self.assertEqual(allocation["gpus"], 4)
        self.assertEqual(self.scheduler.state["capacity"]["teacher_qos_reserved_gpus"], 1)
        self.backend.jobs[attempt["job_id"]].update(state="FAILED", exit_code="1:0")
        self.scheduler.tick()
        self.assertEqual(attempt["accounting"]["gpu_seconds"], 40)
        self.assertEqual(
            self.scheduler.state["tasks"]["ENGINE"]["attempts"][3]["accounting"]["gpu_seconds"], 10
        )

    def test_pending_four_gpu_job_prevents_second_science_submission(self):
        self.authorize()
        self.scheduler.tick()
        self.simulation.engine_verified = True
        self.assertFalse(self.scheduler._submit("SRF1_A_s71001"))
        self.assertEqual(self.scheduler.state["capacity"]["teacher_qos_reserved_gpus"], 5)
        self.assertEqual(self.scheduler.state["capacity"]["requested_gpus"], 4)

    def test_unknown_four_gpu_submission_is_reserved_without_duplication(self):
        self.authorize()
        self.backend.ambiguous = True
        self.scheduler.tick()
        attempt = self.scheduler.state["tasks"]["ENGINE"]["attempts"][-1]
        count = len(self.backend.jobs)
        self.backend.jobs.pop(str(1000 + count - 1))
        self.assertFalse(self.scheduler._capacity_available("SRF1_A_s71001"))
        self.assertEqual(self.scheduler.state["capacity"]["teacher_qos_reserved_gpus"], 5)
        self.assertEqual(attempt["resource_override"]["gpus"], 4)
        self.backend.outage = True
        self.scheduler.tick()
        self.scheduler.tick()
        self.assertEqual(len(self.scheduler.state["tasks"]["ENGINE"]["attempts"]), 5)

    def test_release_waits_when_another_teacher_job_arrives(self):
        self.authorize()
        project = self.backend.project_capacity
        calls = 0

        def changing(permission):
            nonlocal calls
            calls += 1
            self.other_teacher_gpus = 1 if calls == 1 else 2
            return project(permission)

        self.backend.project_capacity = changing
        self.scheduler.tick()
        attempt = self.scheduler.state["tasks"]["ENGINE"]["attempts"][-1]
        self.assertNotIn("released", attempt)
        self.assertTrue(attempt["release_waiting_for_teacher_gpu_capacity"])
        count = len(self.backend.jobs)
        self.scheduler.tick()
        self.assertEqual(len(self.backend.jobs), count)
        self.backend.project_capacity = project
        self.other_teacher_gpus = 1
        self.scheduler.tick()
        self.assertTrue(attempt["released"])

    def test_insufficient_actual_resources_stay_held(self):
        self.authorize()
        self.resource_edits["ReqTRES"] = "cpu=16,mem=90G,node=1,gres/gpu=4,gres/gpu:pro6000=4"
        self.scheduler.tick()
        attempt = self.scheduler.state["tasks"]["ENGINE"]["attempts"][-1]
        self.assertNotIn("released", attempt)
        self.assertEqual(attempt["status"], "UNKNOWN")
        self.resource_edits.clear()
        self.resource_edits["NumNodes"] = "2"
        self.scheduler.tick()
        self.assertNotIn("released", attempt)

    def test_graceful_maintenance_marker_is_preserved_but_not_spliced(self):
        task = self.scheduler.state["tasks"]["ENGINE"]
        atomic_json(self.root / "recovery_checkpoints/ENGINE/test.json", {"step": 0})
        environment = {
            "SR_F1_TASK_ID": "ENGINE",
            "SR_F1_ATTEMPT_ID": self.maintained["attempt_id"],
            "SR_F1_REGISTRATION_HASH": digest(self.scheduler.registration),
        }
        with patch.dict(os.environ, environment):
            checkpoint_task(
                self.root,
                "ENGINE",
                "PREEMPTION",
                ["recovery_checkpoints/ENGINE/test.json"],
                {
                    "full_state": True,
                    "identity_verified": True,
                    "restart_engine_track_required": True,
                },
            )
        marker = f"orchestration/checkpoints/{self.maintained['attempt_id']}.json"
        task.update(status="RETRYABLE", resume_checkpoint=marker)
        self.scheduler._save("TEST_GRACEFUL_MAINTENANCE")
        result = self.authorize()
        self.assertEqual(result["maintenance_checkpoint_sha256"], file_hash(self.root / marker))
        self.assertNotIn("resume_checkpoint", task)
        self.scheduler.tick()
        self.assertTrue(task["attempts"][-1]["released"])

    def test_live_maintenance_or_prior_optimizer_update_prevents_restart(self):
        self.backend.jobs[self.maintained["job_id"]]["state"] = "RUNNING"
        with self.assertRaisesRegex(ValueError, "STILL_ACTIVE"):
            self.authorize()
        self.backend.jobs[self.maintained["job_id"]]["state"] = "CANCELLED"
        atomic_json(
            self.root / "accounting/ENGINE.jsonl",
            {"kind": "physical_optimizer_updates", "count": 1},
        )
        with self.assertRaisesRegex(ValueError, "PRIOR_OPTIMIZER_UPDATE"):
            self.authorize()

    def test_only_engine_and_training_use_auxiliary_gpus(self):
        self.assertEqual(
            resource_override(self.root, self.scheduler.registration, "ENGINE")["gpus"], 4
        )
        self.assertEqual(
            resource_override(self.root, self.scheduler.registration, "SRF1_A_s71001")["gpus"], 4
        )
        self.assertIsNone(resource_override(self.root, self.scheduler.registration, "BASELINE"))
        self.assertIsNone(resource_override(self.root, self.scheduler.registration, "ANALYZE"))

    def test_worker_revision_cannot_rewrite_or_bypass_original_registration(self):
        before = file_hash(self.root / "orchestration/REGISTRATION.json")
        self.repair["worker_source_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "REVISED_WORKER_SOURCE_CHANGED"):
            self.authorize()
        self.repair["worker_source_sha256"] = file_hash(REPO / "scripts/sr_f1/run_worker.py")
        self.repair["previous_worker_source_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "IMMUTABLE_REGISTRATION_CHANGED"):
            self.authorize()
        self.assertEqual(file_hash(self.root / "orchestration/REGISTRATION.json"), before)

    def test_changed_attempt_override_is_rejected_before_projection_or_release(self):
        self.authorize()
        self.scheduler.tick()
        attempt = self.scheduler.state["tasks"]["ENGINE"]["attempts"][-1]
        attempt["resource_override"]["gpus"] = 3
        with self.assertRaisesRegex(ValueError, "ATTEMPT_RESOURCES_CHANGED"):
            self.scheduler._projections()

    def test_wrong_qos_or_model_type_rejects_resource_release(self):
        self.authorize()
        self.resource_edits["QOS"] = "rose"
        self.scheduler.tick()
        attempt = self.scheduler.state["tasks"]["ENGINE"]["attempts"][-1]
        self.assertNotIn("released", attempt)
        self.resource_edits["QOS"] = TEACHER_QOS
        self.resource_edits["ReqTRES"] = "cpu=16,mem=360G,node=1,gres/gpu=4,gres/gpu:h100=4"
        self.scheduler.tick()
        self.assertNotIn("released", attempt)

    def test_worker_requires_actual_requested_and_allocated_resources(self):
        self.authorize()
        self.scheduler.tick()
        attempt = self.scheduler.state["tasks"]["ENGINE"]["attempts"][-1]
        worker = load_module("multigpu_worker", REPO / "scripts/sr_f1/run_worker.py")
        environment = {
            "SR_F1_TASK_ID": "ENGINE",
            "SR_F1_ATTEMPT_ID": attempt["attempt_id"],
            "SR_F1_REGISTRATION_HASH": digest(self.scheduler.registration),
            "SLURM_JOB_ID": attempt["job_id"],
        }
        self.resource_edits["JobState"] = "RUNNING"
        with patch.dict(os.environ, environment):
            self.assertEqual(
                worker.validate_allocation(self.root, "ENGINE", backend=self.backend)["operation"],
                "engine",
            )
        (self.root / f"orchestration/worker_starts/{attempt['attempt_id']}.json").unlink()
        self.resource_edits["AllocTRES"] = "cpu=16,mem=90G,node=1,gres/gpu=4,gres/gpu:pro6000=4"
        with (
            patch.dict(os.environ, environment),
            self.assertRaisesRegex(ValueError, "HOST_MEMORY_BELOW"),
        ):
            worker.validate_allocation(self.root, "ENGINE", backend=self.backend)

    def test_cli_is_exclusive_and_does_not_submit(self):
        cli = load_module("multigpu_cli", REPO / "scripts/sr_f1/submit_matrix.py")
        scheduler = Mock()
        scheduler.resume_repaired_engine_multigpu.return_value = {"status": "RETRYABLE"}
        args = [
            "submit_matrix.py",
            "--plan",
            str(self.simulation.plan),
            "--run-root",
            str(self.root),
            "--code-root",
            str(REPO),
            "--python",
            "/usr/bin/python3",
            "--resume-repaired-engine-multigpu",
        ]
        with patch("sys.argv", args), patch.object(cli, "Scheduler", return_value=scheduler):
            self.assertEqual(cli.main(), 0)
        scheduler.resume_repaired_engine_multigpu.assert_called_once_with()
        scheduler.tick.assert_not_called()
        with patch("sys.argv", [*args, "--watch"]), self.assertRaises(SystemExit):
            cli.main()


class MultiGPUQueueTests(unittest.TestCase):
    def test_only_teacher_qos_jobs_are_counted_with_pending_and_array_totals(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "ENGINE_MULTIGPU_REPAIR.json").write_text("{}")
            backend = SlurmBackend()
            replies = [
                {
                    "returncode": 0,
                    "stdout": (
                        f"10|owner|{TEACHER_QOS}|RUNNING\n"
                        f"11_2|owner|{TEACHER_QOS}|PENDING\n12|owner|rose|PENDING\n"
                    ),
                },
                {
                    "returncode": 0,
                    "stdout": (
                        f"JobId=10 UserId=owner(1) QOS={TEACHER_QOS} "
                        "ReqTRES=gres/gpu=1 AllocTRES=gres/gpu=1"
                    ),
                },
                {
                    "returncode": 0,
                    "stdout": (
                        f"JobId=11_2 UserId=owner(1) QOS={TEACHER_QOS} "
                        "ReqTRES=gres/gpu=4,gres/gpu:pro6000=4 AllocTRES="
                    ),
                },
            ]
            backend._run = Mock(side_effect=replies)
            with patch(
                "sr_f1.orchestration.multigpu_repair", return_value={"repair_sha256": "a" * 64}
            ):
                result = backend.project_capacity(
                    {"owner": "owner", "qos": TEACHER_QOS, "run_root": str(root)}
                )
            self.assertEqual(sum(row["gpus"] for row in result["jobs"]), 5)
            self.assertEqual([row["job_id"] for row in result["jobs"]], ["10", "11_2"])
            self.assertEqual(backend._run.call_count, 3)


class MultiGPUNodeRangeTests(unittest.TestCase):
    def setUp(self):
        self.override = {
            "nodes": 1,
            "gpus": 3,
            "cpus_per_task": 12,
            "minimum_host_memory_gb": 240,
            "constraint": "highmem",
        }
        # Real held-job representation observed for validation job 196459.
        self.fields = {
            "JobState": "PENDING",
            "Reason": "JobHeldUser",
            "NumNodes": "1-1",
            "Features": "highmem",
            "ReqTRES": "cpu=12,mem=270G,node=1,billing=12,gres/gpu=3,gres/gpu:pro6000=3",
            "AllocTRES": "",
        }

    def test_held_exact_single_node_request_range_is_accepted(self):
        verify_multigpu_resources(self.fields, self.override)

    def test_multi_node_or_variable_range_is_rejected(self):
        for nodes in ("2-2", "1-2", "2", "0-1"):
            with self.subTest(nodes=nodes), self.assertRaisesRegex(ValueError, "SINGLE_NODE"):
                verify_multigpu_resources({**self.fields, "NumNodes": nodes}, self.override)

    def test_running_or_allocated_range_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "SINGLE_NODE"):
            verify_multigpu_resources({**self.fields, "JobState": "RUNNING"}, self.override)
        with self.assertRaisesRegex(ValueError, "SINGLE_NODE"):
            verify_multigpu_resources(self.fields, self.override, allocated=True)

    def test_actual_single_node_requires_both_tres_node_counts(self):
        fields = {
            **self.fields,
            "JobState": "RUNNING",
            "NumNodes": "1",
            "AllocTRES": self.fields["ReqTRES"],
        }
        verify_multigpu_resources(fields, self.override, allocated=True)
        for key in ("ReqTRES", "AllocTRES"):
            for count in ("2", "1-1", ""):
                with (
                    self.subTest(key=key, count=count),
                    self.assertRaisesRegex(ValueError, "TRES_SINGLE_NODE"),
                ):
                    modified = fields[key].replace("node=1", "node=" + count)
                    verify_multigpu_resources(
                        {**fields, key: modified}, self.override, allocated=True
                    )

    def test_held_range_still_requires_tres_single_node(self):
        with self.assertRaisesRegex(ValueError, "TRES_SINGLE_NODE"):
            verify_multigpu_resources(
                {**self.fields, "ReqTRES": self.fields["ReqTRES"].replace("node=1,", "")},
                self.override,
            )


if __name__ == "__main__":
    unittest.main()
