"""Concurrency, admission and actual write-once endpoint evidence on CPU."""

import copy
import fcntl
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent


def standalone(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


p = standalone("parallel_main_control_cpu_tests", PACKAGE / "execution/parallel_main_control.py")
serial_fixture = standalone("parallel_main_serial_fixture", HERE / "test_main_control.py")
coverage = standalone(
    "parallel_main_coverage_for_controller", PACKAGE / "operations/satdose_parallel_main_plan.py"
)


class ParallelGateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        root = patch.object(p, "ROOT", self.root)
        root.start()
        self.addCleanup(root.stop)
        (self.root / "DIAGNOSTIC_WORKER.lock").touch()
        p.release_directory().mkdir(parents=True)
        for seed in p.SEEDS:
            (p.release_directory() / f"LANE_{seed}.lock").touch()

    def test_two_lanes_coexist_but_duplicate_or_legacy_worker_cannot_start(self):
        with p.LaneLease(11031), p.LaneLease(11032):
            with self.assertRaises(BlockingIOError), p.LaneLease(11031):
                pass
            with (
                (self.root / "DIAGNOSTIC_WORKER.lock").open("rb+") as legacy,
                self.assertRaises(BlockingIOError),
            ):
                fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (self.root / "DIAGNOSTIC_WORKER.lock").open("rb+") as legacy:
            fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError), p.LaneLease(11032):
                pass

    def test_replaced_lane_file_fails_closed(self):
        with p.LaneLease(11031) as lease:
            path = p.release_directory() / "LANE_11031.lock"
            path.unlink()
            path.touch()
            with self.assertRaisesRegex(PermissionError, "Replaced"):
                lease.active()

    def test_two_complete_results_required_before_next_wave(self):
        prior = p.endpoint_keys([p.WAVES[0]])
        m = SimpleNamespace(
            _existing_slots=lambda: dict.fromkeys(prior), endpoint_key=serial_fixture.c.endpoint_key
        )
        p.check_slots(m, 1, 11031, active=False, worker=False)
        prior.remove("11032/START/0")
        with self.assertRaisesRegex(PermissionError, "predecessor"):
            p.check_slots(m, 1, 11031, active=False, worker=True)
        prior.add("11032/START/0")
        prior.add("11032/T0/8")
        p.check_slots(m, 1, 11031, active=False, worker=True)
        with self.assertRaises(PermissionError):
            p.check_slots(m, 1, 11031, active=False, worker=False)
        prior.add("11032/T1/24")
        with self.assertRaises(PermissionError):
            p.check_slots(m, 1, 11031, active=False, worker=True)

    def test_pair_cannot_cross_releases_jobs_or_waves(self):
        rel = dict(path="release", sha256="1" * 64)
        pair = dict(
            schema="sr-satdose-main-pair-v1",
            release=rel,
            wave=0,
            arm="START",
            step=0,
            jobs={"11031": 17, "11032": 18},
            previous_results={},
        )
        p.validate_pair(pair, rel, 0, 11032, "18")
        for args in [
            (rel, 0, 11032, "17"),
            (rel, 1, 11032, "18"),
            (dict(rel, sha256="2" * 64), 0, 11032, "18"),
        ]:
            with self.assertRaises(PermissionError):
                p.validate_pair(pair, *args)
        with self.assertRaisesRegex(PermissionError, "job IDs"):
            p.validate_pair(dict(pair, jobs={"11031": 17, "11032": 17}), rel, 0)
        with self.assertRaises(PermissionError):
            p.validate_released({"status": "HELD"}, pair, {"path": "pair"})

    def test_peer_cancellation_stops_further_generation(self):
        with (
            patch.object(
                p.subprocess,
                "run",
                return_value=SimpleNamespace(
                    returncode=0, stdout="JobId=24 JobState=CANCELLED ExitCode=0:15"
                ),
            ),
            self.assertRaisesRegex(PermissionError, "Peer job failed"),
        ):
            p.check_peer_scheduler(24)
        with (
            patch.object(
                p.subprocess,
                "run",
                return_value=SimpleNamespace(
                    returncode=0, stdout="JobId=24 JobState=PENDING Reason=JobHeldUser Priority=0"
                ),
            ),
            self.assertRaisesRegex(PermissionError, "held"),
        ):
            p.check_peer_scheduler(24)

    def test_invalid_endpoint_never_loads_project_or_hardware(self):
        with patch.object(p, "bootstrap") as boot:
            with self.assertRaises(PermissionError):
                p.preflight(Path("x"), "a" * 64, 11031, "T1", 2)
            boot.assert_not_called()

    def test_operator_coverage_cannot_exclude_a_chunk_or_dev_audit(self):
        refs = {str(i): {} for i in range(12)}
        terminal = SimpleNamespace(
            receipt={"sha256": "0" * 64},
            spec={"original_train_release": {}, "report": {}},
            audit_spec={"continuation_results": refs},
        )
        value = dict(
            schema="sr-satdose-parallel-main-admission-v1",
            status="TERMINAL_AND_OPERATOR_AUDITS_VERIFIED",
            terminal_admission=terminal.receipt,
            terminal_audit={},
            original_train_release={},
            continuation_audits=dict(refs),
            trained_dev_audits={
                f"{s}/{a}/{t}": {} for s in p.SEEDS for a in ("T0", "T1") for t in (2, 8, 16)
            },
        )
        p.validate_admission(value, terminal)
        value["trained_dev_audits"].pop("11032/T1/16")
        with self.assertRaisesRegex(PermissionError, "audit population"):
            p.validate_admission(value, terminal)


class ParallelExecutionTests(unittest.TestCase):
    """Exercise the two live writers using the real frozen bridge and synthetic sampled tokens."""

    def setUp(self):
        self.real_preflight = p.preflight
        f = serial_fixture.MainControlTests()
        self.addCleanup(f.doCleanups)
        f.setUp()
        self.f = f
        m = serial_fixture.c
        self.addCleanup(patch.stopall)
        patch.object(p, "ROOT", f.root).start()
        p.release_directory().mkdir(parents=True)
        for s in p.SEEDS:
            (p.release_directory() / f"LANE_{s}.lock").touch()
        root = p.wave_directory(0)
        root.mkdir()
        f.write("plan.json", {})
        changes = dict(
            parallel_root=str(PACKAGE),
            parallel_sources={"execution/parallel_main_control.py": m.sha(Path(p.__file__))},
            plan=f.ref("plan.json"),
            terminal_admission=f.ref("admission.json"),
            terminal_publication=f.ref("terminal.json"),
            terminal_manifest=f.ref("terminal.json"),
        )
        f.release.update(changes)
        self.release = p.release_directory() / "RELEASE.json"
        self.release.write_text(json.dumps(f.release))
        self.release_ref = m.ref(self.release)
        self.pair = dict(
            schema="sr-satdose-main-pair-v1",
            release=self.release_ref,
            wave=0,
            arm="START",
            step=0,
            jobs={"11031": 23, "11032": 24},
            previous_results={},
        )
        (root / "PAIR.json").write_text(json.dumps(self.pair))
        self.pair_ref = m.ref(root / "PAIR.json")
        (root / "RELEASE_RESULT.json").write_text(
            json.dumps(
                dict(
                    status="MAIN_PARALLEL_PAIR_RELEASE_CONFIRMED",
                    wave=0,
                    jobs=self.pair["jobs"],
                    pair=self.pair_ref,
                    release=self.release_ref,
                )
            )
        )
        self.released = m.ref(root / "RELEASE_RESULT.json")
        self.ctx = {}
        # Build the real registry/endpoint identity then project two requests
        # for the concurrent kernel-free transaction exercise. Plan cardinality
        # and terminal inputs are validated separately on the full frozen data.
        for seed in p.SEEDS:
            f.release["endpoint"] = dict(start_seed=seed, arm="START", step=0)
            f.update_release()
            with patch.object(m, "_prior", return_value=(m.TA.Observed(f.root), [])):
                c = m.preflight(f.release_path, f.release_sha, seed, "START", 0)
            c["sources"]["authority"] = dict(path=p.__file__, sha256=m.sha(Path(p.__file__)))
            c.update(
                m=m,
                release_ref=self.release_ref,
                released=self.released,
                pair=self.pair_ref,
                wave=0,
                job_id=self.pair["jobs"][str(seed)],
                previous_results={},
                sources_sha256=m.backend.object_sha(c["sources"]),
                count=2,
                requests=c["requests"][:2],
                manifest=c["manifest"][:2],
            )
            self.ctx[seed] = c
        self.thread_data = threading.local()
        patch.object(
            m.first_two,
            "_allocation",
            side_effect=lambda: {
                "job_id": str(self.thread_data.job),
                "observation": {"JobName": f"LH-SATDOSE-MAIN-{self.thread_data.seed}-START-s00"},
            },
        ).start()
        patch.object(p, "check_peer_scheduler").start()
        patch.object(
            p, "preflight", side_effect=lambda path, sha, seed, arm, step, worker: self.ctx[seed]
        ).start()

    def run_lane(self, seed):
        self.thread_data.job = self.pair["jobs"][str(seed)]
        self.thread_data.seed = seed
        return p.execute(self.release, self.release_ref["sha256"], seed, "START", 0)

    def test_actual_concurrent_draws_complete_independently_without_retries(self):
        gate = threading.Barrier(2, timeout=20)
        self.f.hook = lambda _: gate.wait()
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(self.run_lane, p.SEEDS))
        m = serial_fixture.c
        for result in results:
            self.assertEqual(result["responses"], 2)
            self.assertEqual(result["release"], self.release_ref)
            self.assertEqual(result["pair"], self.pair_ref)
            receipt = m.read(m.bound(result["receipt"]))
            self.assertEqual(result["job_id"], self.pair["jobs"][str(result["start_seed"])])
            self.assertEqual(len(receipt["complete_files"]), 2)
            self.assertEqual(receipt["optimizer_steps"], 0)
        self.assertEqual(len(set(self.f.attempts)), 4)
        with self.assertRaises(PermissionError):
            self.run_lane(11032)

    def test_sibling_incident_stops_remaining_draws(self):
        def fail_sibling(_):
            (p.wave_directory(0) / "ERROR_11032.json").write_text("{}")

        self.f.hook = fail_sibling
        with self.assertRaisesRegex(PermissionError, "incident"):
            self.run_lane(11031)
        slot = self.ctx[11031]["directory"]
        self.assertTrue((slot / "HOLD.json").exists())
        self.assertFalse((slot / "RESULT.json").exists())
        self.assertEqual(len(self.f.attempts), 1)

    def test_cpu_preflight_resolves_original_registry_and_rejects_changed_plan(self):
        f = self.f
        m = serial_fixture.c
        rows = [
            r for r in f.manifest if (r["start_seed"], r["arm"], r["step"]) == (11031, "START", 0)
        ]
        ep = dict(
            responses=1952, request_ids_sha256=m.backend.object_sha([r["request_id"] for r in rows])
        )
        plan = {"lanes": {"11031": {"endpoints": [ep]}}}
        f.write("plan.json", plan)
        f.release["plan"] = f.ref("plan.json")
        self.release.write_text(json.dumps(f.release))
        self.release_ref = m.ref(self.release)
        helper = SimpleNamespace(
            build_plan=lambda _: copy.deepcopy(plan), lane_endpoints=coverage.lane_endpoints
        )
        with patch.object(p, "bootstrap", return_value=(m, f.release, f.terminal, helper)):
            ctx = self.real_preflight(self.release, self.release_ref["sha256"], 11031, "START", 0)
            self.assertEqual(ctx["count"], 1952)
            self.assertEqual(ctx["previous_results"], {})
            self.assertEqual(ctx["sources"]["authority"]["path"], p.__file__)
            plan["lanes"]["11031"]["endpoints"][0]["request_ids_sha256"] = "7" * 64
            with self.assertRaisesRegex(PermissionError, "frozen plan"):
                self.real_preflight(self.release, self.release_ref["sha256"], 11031, "START", 0)
        f.factory_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
