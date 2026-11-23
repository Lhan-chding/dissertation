import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "satdose_watch_parallel_main.py"
spec = importlib.util.spec_from_file_location("watch_parallel_main", MODULE)
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def result(wave):
    return dict(
        status="MAIN_PARALLEL_PAIR_RELEASE_CONFIRMED",
        wave=wave,
        release={"path": "execution_releases/parallel_main_v1/RELEASE.json", "sha256": "a" * 64},
        jobs={"11031": 11 + wave * 2, "11032": 12 + wave * 2},
    )


class SupervisorTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.state = Path(tmp.name).resolve()

    def test_incomplete_training_observes_only(self):
        with (
            patch.object(
                watch,
                "readiness",
                return_value={"status": "WAIT_FOR_TRAINING_AND_AUDITS", "missing": ["step-024"]},
            ),
            patch.object(watch, "local") as action,
        ):
            self.assertEqual(watch.advance(self.state)["missing"], ["step-024"])
            action.assert_not_called()
        self.assertEqual(list(self.state.iterdir()), [])

    def test_only_two_complete_clean_parents_can_advance(self):
        self.assertEqual(
            watch.pair_state([11, 12], "11|COMPLETED|0:0\n12|COMPLETED|0:0\n"), "COMPLETE"
        )
        for raw in ("", "11|COMPLETED|0:0", "11|RUNNING|0:0\n12|PENDING|0:0"):
            self.assertEqual(watch.pair_state([11, 12], raw), "WAIT")
        for raw in (
            "13|COMPLETED|0:0",
            "11|FAILED|1:0",
            "11|COMPLETED|0:9",
            "11|COMPLETED|0:0\n11|COMPLETED|0:0",
        ):
            with self.assertRaises(RuntimeError):
                watch.pair_state([11, 12], raw)

    def test_unknown_local_submission_recovers_read_only(self):
        (self.state / "wave-00-attempt-0001").mkdir()
        with (
            patch.object(watch, "inspect", return_value=result(0)) as lookup,
            patch.object(watch, "local") as execute,
        ):
            self.assertEqual(watch.launch(self.state, 0, []), result(0))
            lookup.assert_called_once_with(0)
            execute.assert_not_called()

    def test_replayed_or_unproven_capacity_response_is_not_a_deferral(self):
        r = {
            "returncode": 3,
            "status": "PARALLEL_MAIN_OPERATOR_CAPTURED",
            "response": {
                "deferred": {"wave": 2, "status": "CAPACITY_DEFERRED", "submission_mutations": 0}
            },
        }
        self.assertTrue(watch.deferred(r, 2))
        self.assertFalse(watch.deferred(r, 3))
        r["response"]["deferred"]["submission_mutations"] = 1
        self.assertFalse(watch.deferred(r, 2))

    def test_running_wave_blocks_all_subsequent_submissions(self):
        watch.save(self.state / "TERMINAL_BINDING.json", [])
        watch.save(self.state / "wave-00-accepted.json", result(0))
        with (
            patch.object(watch, "ssh", return_value="11|COMPLETED|0:0\n12|RUNNING|0:0"),
            patch.object(watch, "launch") as execute,
        ):
            self.assertEqual(watch.advance(self.state)["wave"], 0)
            execute.assert_not_called()

    def test_reused_ids_or_changed_release_stop_the_next_wave(self):
        watch.save(self.state / "TERMINAL_BINDING.json", [])
        watch.save(self.state / "wave-00-accepted.json", result(0))
        with patch.object(watch, "ssh", return_value="11|COMPLETED|0:0\n12|COMPLETED|0:0"):
            for key in ("jobs", "release"):
                other = result(1)
                if key == "jobs":
                    other["jobs"]["11031"] = 11
                else:
                    other["release"]["sha256"] = "b" * 64
                watch.save(self.state / "wave-01-accepted.json", other, replace=True)
                with self.assertRaises(RuntimeError):
                    watch.advance(self.state)

    def test_all_fourteen_jobs_reach_only_one_merge(self):
        watch.save(self.state / "TERMINAL_BINDING.json", [])

        def accounting(argv):
            self.assertIn("-X", argv)
            return "\n".join(
                f"{job}|COMPLETED|0:0" for job in argv[argv.index("-j") + 1].split(",")
            )

        def launch(_, wave, args):
            return result(wave)

        with (
            patch.object(watch, "ssh", side_effect=accounting),
            patch.object(watch, "launch", side_effect=launch) as call,
            patch.object(watch, "finish_audit", return_value={"status": "verified"}) as audit,
        ):
            self.assertEqual(watch.advance(self.state), {"status": "verified"})
            self.assertEqual(call.call_count, 7)
            self.assertEqual(len(audit.call_args.args[2]), 14)

    def test_no_audit_success_without_bound_receipt(self):
        attempt = self.state / "final-audit-0001" / "bundle"
        attempt.mkdir(parents=True)
        watch.save(attempt / "AUDIT_RECEIPT.json", dict(status="OK"))
        with self.assertRaisesRegex(RuntimeError, "Final audit receipt"):
            watch.finish_audit(self.state, result(0)["release"], set())


if __name__ == "__main__":
    unittest.main()
