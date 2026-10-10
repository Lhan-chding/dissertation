"""Exercise the deployed submission function with scheduler boundary fixtures."""

import ast
import re
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).with_name("srf11_multigpu_autochain_v2.py")
SUBMIT = next(
    node
    for node in ast.parse(SOURCE.read_text()).body
    if isinstance(node, ast.FunctionDef) and node.name == "submit"
)
PROGRAM = compile(ast.Module(body=[SUBMIT], type_ignores=[]), str(SOURCE), "exec")


class DeploymentDependencyTests(unittest.TestCase):
    def check_submission(self, accounting, queue, expected_dependency):
        with tempfile.TemporaryDirectory() as directory:
            saved = {}
            commands = []

            def save(name, value):
                self.assertNotIn(name, saved)
                saved[name] = value

            def run(args):
                commands.append(args)
                if args[0] == "sacct":
                    output = accounting
                elif args[0] == "squeue":
                    output = queue
                elif args[0] == "sbatch":
                    self.assertIn("--hold", args)
                    output = "777\n"
                else:
                    self.assertEqual(args, ["scontrol", "release", "777"])
                    self.assertEqual(
                        saved["AUTO_V2_MAINTENANCE_SUBMISSION.json"]["stdout"],
                        "777\n",
                    )
                    output = ""
                return {"returncode": 0, "stdout": output, "stderr": ""}

            namespace = {
                "inc": Path(directory),
                "py": "python",
                "re": re,
                "save": save,
                "run": run,
            }
            exec(PROGRAM, namespace)
            if expected_dependency is None:
                with self.assertRaises(AssertionError):
                    namespace["submit"]("maintenance", ["11", "22"])
                self.assertFalse(any(args[0] == "sbatch" for args in commands))
                return
            self.assertEqual(namespace["submit"]("maintenance", ["11", "22"]), "777")
            submitted = next(args for args in commands if args[0] == "sbatch")
            self.assertEqual(
                [arg for arg in submitted if arg.startswith("--dependency")],
                expected_dependency,
            )

    def test_completed_dependency_is_verified_then_omitted(self):
        self.check_submission("11|COMPLETED|0:0\n22|COMPLETED|0:0\n", "", [])

    def test_live_dependency_remains_afterok(self):
        self.check_submission(
            "11|COMPLETED|0:0\n22|RUNNING|0:0\n",
            "22|cpu|RUNNING\n",
            ["--dependency=afterok:22"],
        )

    def test_missing_parent_job_does_not_use_successful_batch_step(self):
        self.check_submission("11|COMPLETED|0:0\n22.batch|COMPLETED|0:0\n", "", None)

    def test_failed_parent_cannot_be_skipped(self):
        self.check_submission("11|COMPLETED|0:0\n22|FAILED|1:0\n", "", None)


if __name__ == "__main__":
    unittest.main()
