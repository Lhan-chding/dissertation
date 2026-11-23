"""CPU-only checks for paired Slurm admission and durable recovery."""

import base64
import json
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[1] / "satdose_launch_parallel_main.py"
spec = spec_from_file_location("parallel_launcher_under_test", SOURCE)
launch = module_from_spec(spec)
spec.loader.exec_module(launch)


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.batch = self.root / "prepared"
        self.batch.mkdir()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(
            patch.multiple(launch, ROOT=self.root, BATCH=self.batch, CODE=self.root / "code")
        )
        self.stack.enter_context(patch.object(launch, "no_scheduler_duplicates"))
        self.room = self.stack.enter_context(
            patch.object(launch, "capacity", return_value={"status": "CAPACITY_READY"})
        )
        launch.put(self.batch / "RELEASE.json", {"wave_budget": 14})
        self.release = launch.ref(self.batch / "RELEASE.json")
        self.commands = []
        self.fault = None
        self.unlocked = False

        def run(cmd):
            self.commands.append(cmd)
            if cmd[0] == "sbatch":
                job = 101 if len(self.commands) == 1 else 102
                response = {"returncode": 0, "stdout": str(job), "stderr": ""}
                if len(self.commands) == 2 and self.fault in ("partial", "duplicate"):
                    response["stdout"] = "" if self.fault == "partial" else "101"
                return dict(response, argv=cmd)
            self.assertEqual(cmd, ["scontrol", "release", "101,102"])
            self.assertTrue((self.batch / "wave-00/PAIR.json").is_file())
            self.assertTrue((self.batch / "wave-00/RELEASE_INTENT.json").is_file())
            return dict(argv=cmd, returncode=int(self.fault == "release"), stdout="", stderr="")

        self.stack.enter_context(patch.object(launch, "run", side_effect=run))

        def job_fields(job):
            seed = 11031 if job == 101 else 11032
            released = bool(self.commands and self.commands[-1][0] == "scontrol")
            if self.fault == "bad_hold" and seed == 11032:
                released = True
            return dict(
                JobId=str(job),
                JobName=launch.name_for(seed, 0),
                Account="rose",
                QOS=launch.QOS,
                Requeue="0",
                Partition="cluster02",
                UserId="varun024(123)",
                NumCPUs="8",
                NumTasks="1",
                NumNodes="1",
                Command=str(launch.CODE / launch.LAUNCHER),
                WorkDir=str(launch.CODE),
                StdOut=str(launch.wave_dir(0) / f"{seed}-{job}.out"),
                StdErr=str(launch.wave_dir(0) / f"{seed}-{job}.err"),
                JobState="PENDING",
                Reason="Resources" if released else "JobHeldUser",
                Priority="2" if released else "0",
                TimeLimit="1-00:00:00",
                ReqTRES="cpu=8,mem=66G,node=1,gres/gpu=2,gres/gpu:pro6000=2",
            )

        self.stack.enter_context(patch.object(launch, "fields_for", side_effect=job_fields))

    def dispatch(self):
        def unlock():
            self.assertFalse((launch.wave_dir(0) / "RELEASE_RESULT.json").exists())
            self.unlocked = True

        return launch.dispatch(0, self.release, {}, lambda: None, unlock)

    def test_pair_release_once_then_read_only_recovery(self):
        outcome = self.dispatch()["result"]
        self.assertEqual(outcome["jobs"], {"11031": 101, "11032": 102})
        self.assertTrue(self.unlocked)
        for i, cmd in enumerate(self.commands[:2]):
            self.assertIn("--hold", cmd)
            self.assertIn("--gpus=pro6000:2", cmd)
            self.assertIn("--qos=" + launch.QOS, cmd)
            self.assertIn("--time=24:00:00", cmd)
            self.assertEqual(cmd[-4:], [str(11031 + i), "START", "0", "0"])
        with patch.object(launch, "observed", return_value=""):
            self.assertEqual(launch.inspect_wave(0)["result"], outcome)
        with self.assertRaises(FileExistsError):
            self.dispatch()
        self.assertEqual(len(self.commands), 3)

    def test_capacity_defers_before_any_slot_or_submission(self):
        self.room.return_value = {"status": "CAPACITY_DEFERRED"}
        self.assertIn("deferred", self.dispatch())
        self.assertFalse(launch.wave_dir(0).exists())
        self.assertFalse(self.commands)

    def failure(self, fault, release_calls):
        self.fault = fault
        with self.assertRaises(ValueError):
            self.dispatch()
        self.assertTrue((launch.wave_dir(0) / "ERROR.json").is_file())
        self.assertFalse((launch.wave_dir(0) / "RELEASE_RESULT.json").exists())
        self.assertEqual(sum(cmd[0] == "scontrol" for cmd in self.commands), release_calls)
        self.assertEqual(launch.inspect_wave(0)["status"], "WAVE_INCOMPLETE_UNKNOWN_NO_RETRY")
        with self.assertRaises(FileExistsError):
            self.dispatch()

    def test_ambiguous_second_submission_preserves_first_without_release(self):
        self.failure("partial", 0)

    def test_repeated_id_blocks_pair(self):
        self.failure("duplicate", 0)

    def test_foreign_hold_blocks_both(self):
        self.failure("bad_hold", 0)

    def test_partial_release_never_authorizes_generation(self):
        self.failure("release", 1)

    def test_tampered_prior_accounting_blocks_next_wave(self):
        self.dispatch()
        with (
            patch.object(
                launch, "observed", return_value="101|bad|COMPLETED|0:0|rose|qos|cluster02\n"
            ),
            self.assertRaisesRegex(ValueError, "accounting"),
        ):
            launch.verify_finished_jobs(1, self.release)

    def test_atomic_evidence_immutable(self):
        original = launch.checked(self.batch / "RELEASE.json")
        with self.assertRaises(FileExistsError):
            launch.put(self.batch / "RELEASE.json", {"unexpected": 1})
        self.assertEqual(launch.checked(self.batch / "RELEASE.json"), original)


class DeploymentAndAdmissionTests(unittest.TestCase):
    def test_atomic_complete_package_and_idempotent_reinspection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            code = root / "new_parallel_package"
            blobs = {name: name.encode() for name in launch.ASSETS}
            request = dict(
                sources={k: launch.digest(v) for k, v in blobs.items()},
                blobs={k: base64.b64encode(v).decode() for k, v in blobs.items()},
            )
            with patch.multiple(launch, ROOT=root, CODE=code):
                value = launch.deploy(request)
                self.assertEqual(value, launch.deploy(request))
                self.assertEqual(launch.inventory(code), request["sources"])
                (code / launch.CONTROL).write_bytes(b"foreign")
                with self.assertRaisesRegex(ValueError, "deployment differs"):
                    launch.deploy(request)
                self.assertEqual((code / launch.CONTROL).read_bytes(), b"foreign")

    def test_deployment_bytes_travel_on_stdin(self):
        package = "a" * 200_000
        with patch.object(launch.subprocess, "run") as process:
            launch.ssh_action({"blobs": {"example": package}}, b"operator")
        args, kwargs = process.call_args
        self.assertLess(max(len(v) for v in args[0]), 2048)
        self.assertNotIn(package, args[0][-1])
        self.assertEqual(json.loads(kwargs["input"])["blobs"]["example"], package)

    def test_corrupt_transfer_cannot_publish_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            code = root / "new_parallel_package"
            blobs = {n: b"file" for n in launch.ASSETS}
            request = dict(
                sources={k: launch.digest(v) for k, v in blobs.items()},
                blobs={k: base64.b64encode(v).decode() for k, v in blobs.items()},
            )
            request["blobs"][launch.CONTROL] = base64.b64encode(b"changed").decode()
            with (
                patch.multiple(launch, ROOT=root, CODE=code),
                self.assertRaisesRegex(ValueError, "Invalid shipped source"),
            ):
                launch.deploy(request)
            self.assertFalse(code.exists())

    def test_missing_actual_terminal_cannot_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp).resolve()
            stack.enter_context(
                patch.multiple(launch, ROOT=root, BATCH=root / "admission", CODE=root / "capsule")
            )
            stack.enter_context(patch.dict(launch.os.environ, CUDA_VISIBLE_DEVICES=""))
            sources = {name: "1" * 64 for name in launch.ASSETS}
            stack.enter_context(patch.object(launch, "inventory", return_value=sources))
            stack.enter_context(
                patch.object(launch, "capacity", return_value={"status": "CAPACITY_READY"})
            )
            prep = Mock()
            prep.existing_lease.side_effect = lambda *a: nullcontext(lambda: None)
            prep.authenticate_remote.side_effect = ValueError("terminal is incomplete")
            stack.enter_context(patch.object(launch, "load", side_effect=[prep, Mock()]))
            dispatch = stack.enter_context(patch.object(launch, "dispatch"))
            with self.assertRaisesRegex(ValueError, "terminal is incomplete"):
                launch.remote_launch({"wave": 0, "parallel_sources": sources})
            dispatch.assert_not_called()
            self.assertFalse((root / "admission").exists())


class CapacityTests(unittest.TestCase):
    def test_all_qos_jobs_and_shared_user_slots(self):
        def observed(cmd):
            if cmd[0] == "sacctmgr":
                return launch.QOS + "|gres/gpu:pro6000=8|5\n"
            return "10|varun024|RUNNING\n11|someone|PENDING\n12|varun024|PENDING\n"

        def fields(job):
            env = {
                "UserId": "someone(1)" if job == "11" else "varun024(2)",
                "QOS": launch.QOS,
                "Reason": "Resources",
                "Priority": "1",
            }
            if job == "10":
                env.update(JobState="RUNNING", AllocTRES="cpu=8,gres/gpu=2,gres/gpu:pro6000=2")
            elif job == "11":
                env.update(JobState="PENDING", ReqTRES="gres/gpu=3")
            else:
                env.update(
                    JobState="PENDING",
                    Reason="JobHeldUser",
                    Priority="0",
                    ReqTRES="gres/gpu:pro6000=4",
                )
            return env

        with (
            patch.object(launch, "observed", side_effect=observed),
            patch.object(launch, "fields_for", side_effect=fields),
        ):
            value = launch.capacity()
        self.assertEqual(
            [
                value[k]
                for k in ("running_gpu", "runnable_gpu", "held_jobs", "free_slots", "gpu_headroom")
            ],
            [2, 3, 1, 3, 3],
        )
        self.assertEqual(value["status"], "CAPACITY_DEFERRED")

    def test_scheduler_values_can_be_empty_unquoted_or_contain_apostrophe(self):
        output = (
            "JobId=29 Comment=operator's note WorkDir=/scratch/one two/owner=abc "
            'StdErr= StdOut="/scratch/two words" QOS=test ReqTRES=gres/gpu=2 Socks/Node=*'
        )
        with patch.object(launch, "observed", return_value=output):
            fields = launch.fields_for(29)
        self.assertEqual(fields["WorkDir"], "/scratch/one two/owner=abc")
        self.assertEqual(fields["StdOut"], "/scratch/two words")
        self.assertEqual(fields["StdErr"], "")
        self.assertEqual(fields["ReqTRES"], "gres/gpu=2")

    def test_ambiguous_critical_scheduler_field_fails_closed(self):
        with (
            patch.object(launch, "observed", return_value="JobId=29 QOS=test Comment=x QOS=other"),
            self.assertRaisesRegex(ValueError, "Ambiguous"),
        ):
            launch.fields_for(29)

    def test_scheduler_lookup_failure_fails_closed_without_claim(self):
        with patch.object(launch, "observed", side_effect=ValueError("queue changed")):
            self.assertEqual(launch.capacity()["reason"], "CAPACITY_UNOBSERVED")

    def test_unregistered_wave_endpoint_budget(self):
        for wave, (arm, step) in enumerate(launch.WAVES):
            for seed in launch.SEEDS:
                cmd = launch.intended_command(seed, wave, {"sha256": "a" * 64})
                self.assertIn("--time=" + ("24:00:00" if step in (0, 24) else "06:00:00"), cmd)
                self.assertEqual(cmd[-4:], [str(seed), arm, str(step), str(wave)])


if __name__ == "__main__":
    unittest.main()
