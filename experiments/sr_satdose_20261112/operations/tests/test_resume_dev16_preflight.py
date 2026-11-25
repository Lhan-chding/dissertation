import os
import sys
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

OP = Path(__file__).resolve().parents[1] / "satdose_resume_dev16_preflight.py"
spec = spec_from_file_location("resume_dev16_test", OP)
m = module_from_spec(spec)
spec.loader.exec_module(m)


class CPURecovery(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name).resolve() / "prepared"
        root.mkdir()
        pub = root / "execution_releases"
        pub.mkdir()
        side = pub / "fresh_cpu_recovery"
        side.mkdir()
        self.root = root
        self.pub = pub
        self.side = side
        source = b"operator-source"
        self.src = source
        self.raw = m.old_local()
        self.request = m.bind_intent(self.raw)
        self.patch(
            patch.multiple(
                m,
                ROOT=root,
                CODE=root / "train_code_71018e5",
                PUB=pub,
                SIDE=side,
                __file__=side / "RECOVERY_SOURCE.py",
            )
        )
        self.patch(patch.object(m, "bind_intent", return_value=self.request))
        m.write(side / "INTENT.json", {"source_sha256": m.digest(source)})
        m.write(side / "RECOVERY_SOURCE.py", source)
        m.write(side / "ORIGINAL_INTENT.json", self.raw)
        for n in ("FIRST_TWO_LAUNCHER.lock", "DIAGNOSTIC_WORKER.lock"):
            m.write(root / n, b"")
        self.budget = {"required_bytes": 42, "max_observation_age_seconds": 300}
        self.risk = {"pending_required_dev": ["11031/T1/16"]}
        self.release = {"storage_budget": m.prior_ref("STORAGE_BUDGET")}
        self.patch(patch.object(m, "stable_inventory"))
        self.patch(
            patch.object(m, "verified_inputs", return_value=(self.release, self.budget, self.risk))
        )
        self.patch(patch.object(m, "unoccupied", return_value={"queue": {"matching": 0}}))
        self.patch(
            patch.object(
                m, "prior_process_observation", return_value={"status": "ORIGINAL_PID_ABSENT"}
            )
        )
        self.patch(patch.dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        execution = ModuleType("execution")
        self.c = Mock()
        self.c.ROOT = m.ROOT
        self.c.PACKAGE = m.CODE
        self.c.__file__ = str(m.CODE / "execution/trained_dev_control.py")
        self.c._process_identity.return_value = {"pid": 234}
        self.c.preflight.return_value = {
            "release": self.release,
            "risk": self.risk,
            "budget": self.budget,
            "original": {"execution_sources": self.request["execution_sources"]},
            "manifest": list(range(108)),
            "requests": list(range(108)),
            "base": Mock(),
            "canary": Mock(),
        }
        measured = m.datetime.now(m.timezone.utc)
        self.c.storage._now.return_value = measured
        self.c.storage._budget.return_value = self.budget
        self.c.storage_check.return_value = {
            "checked_at_utc": measured.isoformat(),
            "schema": "sr-satdose-trained-dev-storage-v1",
            "budget": self.release["storage_budget"],
        }
        self.c.tx = SimpleNamespace(
            write_once=m.write, canonical=lambda v: m.canonical(v).rstrip(b"\n")
        )
        self.c.ref = lambda path: {
            "path": str(path.relative_to(m.ROOT)),
            "sha256": m.digest(m.file(path)),
        }
        execution.trained_dev_control = self.c
        self.patch(patch.dict(sys.modules, execution=execution))
        self.patch(patch.object(sys, "path", sys.path.copy()))

    def patch(self, ctx):
        val = ctx.start()
        self.addCleanup(ctx.stop)
        return val

    def test_real_preflight_and_storage_and_provenance_required_before_result(self):
        m.run_worker()
        self.c.preflight.assert_called_once_with(
            m.artifact("RELEASE"), m.PRIOR_REMOTE["RELEASE"], 11031, "T1", 16
        )
        self.c.storage_check.assert_called_once_with(m.PUB, self.release["storage_budget"])
        value = m.remote_collect(self.src, self.raw)
        self.assertEqual(value["status"], "CPU_RECOVERY_INDEPENDENT_BYTES_VERIFIED")
        self.assertEqual(value["provenance"]["intended_result"]["committed_updates"], 64)
        self.assertEqual(value["provenance"]["original_transport_exit"], 255)
        with self.assertRaises(FileExistsError):
            m.run_worker()

    def test_risk_change_stops_before_storage_and_write(self):
        self.c.preflight.return_value["risk"] = {"changed": True}
        with self.assertRaisesRegex(ValueError, "preflight differs"):
            m.run_worker()
        self.c.storage_check.assert_not_called()
        self.assertFalse(m.artifact("PREPARATION_RESULT").exists())

    def test_storage_failure_does_not_publish(self):
        self.c.storage_check.side_effect = PermissionError("quota")
        with self.assertRaisesRegex(PermissionError, "quota"):
            m.run_worker()
        self.assertFalse(m.artifact("PREPARATION_RESULT").exists())

    def test_target_exists_blocks_old_preparation_recovery(self):
        m.unoccupied.side_effect = ValueError("occupied")
        with self.assertRaisesRegex(ValueError, "occupied"):
            m.run_worker()
        self.c.preflight.assert_not_called()

    def test_partial_result_cannot_be_collected_as_verified(self):
        m.run_worker()
        m.artifact("PREPARATION_RESULT").write_bytes(b"partial")
        with self.assertRaisesRegex(ValueError, "evidence chain"):
            m.remote_collect(self.src, self.raw)


class OriginalAttemptTests(unittest.TestCase):
    def test_local_original_bytes_are_bound(self):
        self.assertEqual(m.bind_intent(m.old_local())["step"], 16)
        with self.assertRaisesRegex(ValueError, "intent SHA"):
            m.bind_intent(m.old_local() + b" ")

    def test_read_only_scheduler_queries_are_limited_to_exact_endpoint(self):
        with TemporaryDirectory() as tmp:
            pub = Path(tmp).resolve()
            for k in m.PRIOR_REMOTE:
                (pub / (m.PREFIX + k + ".json")).write_bytes(b"original")
            success = SimpleNamespace(stderr="", stdout="")
            with (
                patch.object(m, "PUB", pub),
                patch.object(m.subprocess, "run", return_value=success) as proc,
            ):
                m.unoccupied()
                calls = [call.args[0] for call in proc.call_args_list]
                self.assertEqual([cmd[0] for cmd in calls], ["squeue", "sacct"])
                self.assertTrue(all("--name=" + m.JOB in cmd for cmd in calls))
                proc.return_value = SimpleNamespace(stderr="", stdout="123|" + m.JOB + "|PENDING\n")
                with self.assertRaisesRegex(ValueError, "Matching"):
                    m.unoccupied()
