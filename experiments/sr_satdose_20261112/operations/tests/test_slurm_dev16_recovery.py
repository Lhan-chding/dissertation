import subprocess
import unittest
from contextlib import nullcontext
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

OP = Path(__file__).resolve().parents[1] / "satdose_slurm_dev16_recovery.py"
spec = spec_from_file_location("slurm_cpu_recovery", OP)
m = module_from_spec(spec)
spec.loader.exec_module(m)


class SlurmRecovery(unittest.TestCase):
    def setUp(self):
        self.old = (OP.parent / m.V1_NAME).read_bytes()
        self.helper = (OP.parent / m.HELPER_NAME).read_bytes()
        self.v, self.s = m.load_dependencies(self.old, self.helper)
        self.probe = self.v.file(
            self.v.OLD / "interruption_20261125/SLURM_RECOVERY_INSPECTION.json"
        )
        self.raw = self.v.old_local()
        self.tres = "billing=2,cpu=2,mem=6144M,node=1"

    def test_frozen_inputs_and_shell(self):
        m.observation(self.v, self.probe)
        with self.assertRaises(ValueError):
            m.observation(self.v, self.probe + b" ")
        args = m.sbatch()
        for item in (
            "--cpus-per-task=2",
            "--mem=6G",
            "--time=04:00:00",
            "--hold",
            "--qos=" + m.QOS,
        ):
            self.assertIn(item, args)
        self.assertFalse(any("gpu" in flag.lower() for flag in args))
        subprocess.run(["bash", "-n"], input=m.command(m.sha(OP.read_bytes())), check=True)

    def test_resource_identity(self):
        m.resources(self.s, self.tres)
        m.resources(self.s, self.tres.replace("6144M", "6G"))
        for bad in (
            self.tres + ",gres/gpu=1",
            self.tres + ",gres/gpu:pro6000=1",
            self.tres.replace("cpu=2", "cpu=1"),
            self.tres.replace("6144M", "64G"),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                m.resources(self.s, bad)

    def accounting(self, state="COMPLETED", exitcode="0:0", batch=True, alloc=None):
        rows = [
            "|".join(
                [
                    "42",
                    m.CPU_JOB,
                    "rose",
                    m.QOS,
                    "cluster02",
                    state,
                    "0:0",
                    self.tres,
                    self.tres if alloc is None else alloc,
                ]
            )
        ]
        if batch:
            rows.append("|".join(["42.batch", "batch", "rose", "", "", state, exitcode, "", ""]))
        self.s.observed = Mock(return_value="\n".join(rows))
        return m.account(self.s, 42)

    def test_accounting_requires_parent_and_successful_batch(self):
        self.assertTrue(self.accounting()[0])
        self.assertFalse(self.accounting(state="RUNNING")[0])
        for kw in ({"batch": False}, {"exitcode": "1:0"}, {"alloc": self.tres + ",gres/gpu=2"}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.accounting(**kw)

    def test_submit_uncertain_does_not_retry_or_release(self):
        with TemporaryDirectory() as d:
            base = Path(d).resolve() / "cpu"
            with (
                patch.object(m, "BASE", base),
                patch.object(m, "old_side"),
                patch.object(m, "original_leases", return_value=nullcontext(lambda: None)),
                patch.object(m, "no_cpu_job"),
                patch.multiple(
                    self.v, unoccupied=Mock(), verified_inputs=Mock(), stable_inventory=Mock()
                ),
            ):
                self.s.run = Mock(return_value=dict(returncode=1, stdout="", stderr="unknown"))
                for _ in range(2):
                    with self.assertRaises(ValueError):
                        m.submit(
                            self.v,
                            self.s,
                            OP.read_bytes(),
                            self.old,
                            self.helper,
                            self.raw,
                            self.probe,
                        )
                self.s.run.assert_called_once_with(m.sbatch())
                self.assertTrue((base / "INTENT.json").is_file())
                self.assertTrue((base / "SUBMIT_RESPONSE.json").is_file())
                self.assertFalse((base / "RELEASE_INTENT.json").exists())
