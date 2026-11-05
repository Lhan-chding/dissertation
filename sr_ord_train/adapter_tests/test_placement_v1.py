"""CPU-only scheduling safety tests; no cluster calls or model imports."""
import contextlib
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scheduling import placement_v1 as p


OWNER = "researcher"


def ledger(*requests):
    return {"at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "basis": "Audited saved submissions and terminal receipts.",
            "unresolved_requests": list(requests)}


def row(job=1, owner=OWNER, qos=p.TARGET_QOS, name="LH-ORD-X", state="RUNNING", count=2, nodes=1):
    return f"{job}|{owner}|{qos}|{name}|{state}|gres/gpu:pro6000:{count}|{nodes}"


def unknown(job=1, **kwargs):
    result = dict(job_id=str(job), owner=OWNER, qos=p.TARGET_QOS, name="LH-ORD-X",
                  gpu_type="pro6000", gpus_per_node=2, nodes=1)
    result.update(kwargs)
    return result


class PlacementTests(unittest.TestCase):
    def plan(self, rows, audit=None):
        return p.plan("\n".join(rows), audit if audit is not None else ledger(), OWNER)

    def test_six_cards_block_new_job(self):
        result = self.plan([row(i) for i in (1, 2, 3)])
        self.assertEqual(result["status"], "BLOCKED_CAPACITY")
        self.assertEqual(result["before"]["experiment_gpus"], 6)
        self.assertIn("--exclude=gpu-pro6000-12", result["planned_resource_args"])
        self.assertFalse(result["submission_performed"])

    def test_four_plus_two_ready(self):
        result = self.plan([row(1), row(2)])
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["hypothetical_after_one_new_job"]["experiment_gpus"], 6)

    def test_pending_and_unknown_are_counted(self):
        result = self.plan([row(1), row(2, state="PENDING")], ledger(unknown(3)))
        self.assertEqual(result["status"], "BLOCKED_CAPACITY")
        self.assertEqual(result["requests"][-1]["state"], "UNKNOWN")

    def test_unknown_survives_queue_absence(self):
        result = self.plan([], ledger(unknown(1, nodes=3)))
        self.assertEqual(result["before"]["experiment_gpus"], 6)
        self.assertEqual(result["status"], "BLOCKED_CAPACITY")

    def test_other_user_consumes_shared_qos(self):
        result = self.plan([row(1), row(2, owner="other", name="UNRELATED", count=6)])
        self.assertEqual(result["before"]["experiment_gpus"], 2)
        self.assertIn("shared_target_qos_pro6000", result["blocking_limits"])

    def test_five_job_limit(self):
        result = self.plan([row(i, name="UNRELATED", count=1) for i in range(1, 6)])
        self.assertEqual(result["blocking_limits"], ["owner_target_qos_jobs"])

    def test_other_user_jobs_do_not_consume_owner_job_slots(self):
        result = self.plan([row(i, owner="other", name="UNRELATED", count=1) for i in range(1, 6)])
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["before"]["owner_target_qos_jobs"], 0)

    def test_other_qos_is_separate(self):
        result = self.plan([row(1), row(2, qos="rose", name="UNRELATED", count=12)])
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["other_qos_job_ids"], ["2"])
        self.assertEqual(result["before"]["shared_target_qos_pro6000"], 2)

    def test_unrelated_other_qos_array_does_not_block(self):
        unrelated = "123_[1-20]|other|rose|any name allowed|PENDING|gres/gpu:2|1"
        result = self.plan([row(1), unrelated])
        self.assertEqual(result["status"], "READY")
        self.assertEqual(len(result["requests"]), 1)
        for relevant in (unrelated.replace("other", OWNER),
                         unrelated.replace("rose", p.TARGET_QOS),
                         unrelated.replace("any name allowed", "LH-ORD-OTHER")):
            with self.subTest(relevant=relevant), self.assertRaises(p.UnsafeInput):
                self.plan([relevant])

    def test_unrelated_basic_malformed_row_still_rejected(self):
        for invalid in ("1|other|rose|bad", "1|other|rose|name|PENDING||1"):
            with self.subTest(invalid=invalid), self.assertRaises(p.UnsafeInput):
                self.plan([invalid])

    def test_experiment_in_other_qos_still_counts_experiment_cap(self):
        result = self.plan([row(1, qos="rose", count=6)])
        self.assertEqual(result["blocking_limits"], ["experiment_gpus"])

    def test_multinode_is_multiplied(self):
        result = self.plan([row(nodes=3)])
        self.assertEqual(result["before"]["experiment_gpus"], 6)

    def test_identical_duplicates_and_ledger_overlap_count_once(self):
        result = self.plan([row(1), row(1)], ledger(unknown(1), unknown(1)))
        self.assertEqual(result["before"]["experiment_gpus"], 2)
        self.assertEqual(len(result["requests"]), 1)

    def test_conflicting_duplicates_fail_closed(self):
        for second in (row(1, count=1), row(1, owner="other"), row(1, state="PENDING"),
                       row(1, owner="other", qos="rose", name="UNRELATED")):
            with self.subTest(second=second), self.assertRaises(p.UnsafeInput):
                self.plan([row(1), second])
        with self.assertRaises(p.UnsafeInput):
            self.plan([row(1)], ledger(unknown(1, nodes=2)))

    def test_malformed_queue_fail_closed(self):
        invalid = ["garbage", row().replace("gres/gpu:pro6000:2", "gres/gpu:2"),
                   row().replace("gres/gpu:pro6000:2", "gres/gpu:pro6000:2,gres/gpu:a100:1"),
                   row(nodes=0), row(count=0), row(job="123_[1-3]"), row(state="UNKNOWN"),
                   row(state="COMPLETED"), row().replace("pro6000", "a100")]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(p.UnsafeInput):
                self.plan([value])

    def test_null_per_node_tres_cannot_prove_zero_gpu(self):
        for resource in ("(null)", "N/A"):
            with self.subTest(resource=resource), self.assertRaises(p.UnsafeInput):
                self.plan([row(name="MAYBE_CPU").replace("gres/gpu:pro6000:2", resource)])

    def test_ledger_requires_audit_even_when_empty(self):
        for value in ({}, {"unresolved_requests": []}, ledger() | {"basis": ""},
                      ledger() | {"at_utc": "bad"}, ledger() | {"at_utc": "2026-11-05"}):
            with self.subTest(value=value), self.assertRaises(p.UnsafeInput):
                self.plan([], value)

    def test_stale_and_future_ledger_fail_closed(self):
        now = datetime(2026, 11, 5, 13, tzinfo=timezone.utc)
        for delta in (-901, 1):
            audit = ledger()
            audit["at_utc"] = (now + timedelta(seconds=delta)).isoformat().replace("+00:00", "Z")
            with self.subTest(delta=delta), self.assertRaises(p.UnsafeInput):
                p.plan("", audit, OWNER, now=now)
        audit["at_utc"] = (now - timedelta(seconds=900)).isoformat().replace("+00:00", "Z")
        self.assertEqual(p.plan("", audit, OWNER, now=now)["status"], "READY")

    def test_ledger_invalid_values_and_conflicts(self):
        for entry in (unknown(gpus_per_node=True), unknown(nodes=-1), unknown(job_id="x"),
                      unknown(gpu_type="none", gpus_per_node=2)):
            with self.subTest(entry=entry), self.assertRaises(p.UnsafeInput):
                self.plan([], ledger(entry))
        with self.assertRaises(p.UnsafeInput):
            self.plan([], ledger(unknown(), unknown(owner="other")))

    def test_input_is_immutable(self):
        audit = ledger(unknown(3))
        original = copy.deepcopy(audit)
        self.plan([row(1)], audit)
        self.assertEqual(audit, original)

    def run_cli(self, response=None, error=None):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ledger.json"
            path.write_text(json.dumps(ledger()))
            output = io.StringIO()
            def respond(command, **kwargs):
                if error is not None:
                    raise error
                if command[0] == "squeue":
                    return response
                job = command[-1]
                return subprocess.CompletedProcess(command, 0,
                    f"JobId={job} JobName=LH-ORD-X JobState=RUNNING UserId={OWNER}(1000) QOS={p.TARGET_QOS} NumNodes=1 "
                    "ReqTRES=cpu=8,node=1,gres/gpu=2,gres/gpu:pro6000=2 "
                    "AllocTRES=cpu=8,node=1,gres/gpu=2,gres/gpu:pro6000=2", "")
            with patch.object(p.subprocess, "run", side_effect=respond) as run:
                with contextlib.redirect_stdout(output):
                    code = p.main(["--owner", OWNER, "--unresolved-ledger", str(path)])
                self.assertEqual(run.call_args_list[0].args[0][0], "squeue")
                for call in run.call_args_list:
                    self.assertIn(call.args[0][0], ("squeue", "scontrol"))
                    self.assertTrue(call.kwargs["check"])
                    self.assertNotIn("shell", call.kwargs)
                    self.assertFalse(any(key.startswith("SQUEUE_") for key in call.kwargs["env"]))
                self.assertIn("--states=all", run.call_args_list[0].args[0])
            return code, json.loads(output.getvalue())

    def test_cli_read_only_success(self):
        code, result = self.run_cli(subprocess.CompletedProcess([], 0, row(1), ""))
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "READY")
        self.assertFalse(result["submission_performed"])
        self.assertEqual(result["scontrol_verified_job_ids"], ["1"])

    def test_cli_filters_environment_without_mutating_caller(self):
        with patch.dict(os.environ, {"SQUEUE_USERS": "someone", "SQUEUE_STATES": "PENDING"}):
            before = dict(os.environ)
            code, _ = self.run_cli(subprocess.CompletedProcess([], 0, row(1), ""))
            self.assertEqual(code, 0)
            self.assertEqual(dict(os.environ), before)

    def test_scontrol_resource_crosscheck(self):
        request = p.parse_queue(row(1))[0]["1"]
        valid = (f"JobId=1 JobName=LH-ORD-X JobState=RUNNING UserId={OWNER}(1000) QOS={p.TARGET_QOS} NumNodes=1 "
                 "ReqTRES=cpu=8,node=1,gres/gpu=2,gres/gpu:pro6000=2")
        with self.assertRaises(p.UnsafeInput):
            p.verify_job_detail(valid, request)
        for invalid in (valid.replace("NumNodes=1", "NumNodes=1-4"),
                        valid.replace("NumNodes=1", "NumNodes=2"),
                        valid.replace("gres/gpu:pro6000=2", "gres/gpu:pro6000=4"),
                        valid.replace("gres/gpu=2", "gres/gpu=4"),
                        valid.replace("node=1", "node=2"), valid.replace("JobId=1", "JobId=2"),
                        valid.replace("JobName=LH-ORD-X", "JobName=OTHER"),
                        valid.replace("JobState=RUNNING", "JobState=COMPLETED"),
                        valid.replace(OWNER, "other"), valid.replace(p.TARGET_QOS, "rose"),
                        valid.replace("gres/gpu:pro6000=2", "gres/gpu:a100=2"),
                        valid + " NumNodes=1", valid + ",gres/gpu:pro6000=2", ""):
            with self.subTest(invalid=invalid), self.assertRaises(p.UnsafeInput):
                p.verify_job_detail(invalid, request)
        with self.assertRaises(p.UnsafeInput):
            p.verify_job_detail(valid, request, expected_state="PENDING")
        p.verify_job_detail(valid + " AllocTRES=cpu=8,node=1,gres/gpu=2,gres/gpu:pro6000=2", request)
        p.verify_job_detail(valid + " AllocTRES=cpu=8,node=1,gres/gpu=1,gres/gpu:pro6000=1", request)
        for allocation in ("gres/gpu=4,gres/gpu:pro6000=4", "gres/gpu=4,gres/gpu:pro6000=2",
                           "gres/gpu=2,gres/gpu:a100=2", "cpu=8", "(null)", "N/A"):
            with self.subTest(allocation=allocation), self.assertRaises(p.UnsafeInput):
                p.verify_job_detail(valid + " AllocTRES=" + allocation, request)

    def test_pending_unallocated_still_counts_complete_request(self):
        queued = row(274069, state="PENDING", count=1)
        request = p.parse_queue(queued)[0]["274069"]
        detail = (f"JobId=274069 JobName=LH-ORD-X JobState=PENDING UserId={OWNER}(1000) "
                  f"QOS={p.TARGET_QOS} NumNodes=1 "
                  "ReqTRES=cpu=4,mem=33G,node=1,billing=8,gres/gpu=1,gres/gpu:pro6000=1")
        for allocation in ("(null)", "N/A"):
            with self.subTest(allocation=allocation):
                p.verify_job_detail(detail + " AllocTRES=" + allocation, request, "PENDING")
                result = self.plan([queued])
                self.assertEqual(result["before"]["experiment_gpus"], 1)
                self.assertEqual(result["before"]["shared_target_qos_pro6000"], 1)
                self.assertEqual(result["before"]["owner_target_qos_jobs"], 1)
                with self.assertRaises(p.UnsafeInput):
                    p.verify_job_detail(detail.replace("JobState=PENDING", "JobState=RUNNING")
                                        + " AllocTRES=" + allocation, request, "RUNNING")

    def test_cli_scontrol_failure_cannot_return_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ledger.json"
            path.write_text(json.dumps(ledger()))
            outputs = [subprocess.CompletedProcess([], 0, row(1), ""),
                       subprocess.CalledProcessError(1, "scontrol")]
            stream = io.StringIO()
            with patch.object(p.subprocess, "run", side_effect=outputs), contextlib.redirect_stdout(stream):
                code = p.main(["--owner", OWNER, "--unresolved-ledger", str(path)])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(stream.getvalue())["status"], "BLOCKED_INPUT")

    def test_cli_timeout_nonzero_stderr_never_ready(self):
        for error in (subprocess.TimeoutExpired("squeue", 15), subprocess.CalledProcessError(1, "squeue"), OSError("missing")):
            with self.subTest(error=error):
                code, result = self.run_cli(error=error)
                self.assertEqual(code, 2)
                self.assertEqual(result["status"], "BLOCKED_INPUT")
        code, result = self.run_cli(subprocess.CompletedProcess([], 0, "", "warning"))
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "BLOCKED_INPUT")

    def test_existing_bound_sources_untouched_by_planning(self):
        package = Path(__file__).resolve().parents[1]
        files = [path for directory in ("adapter", "gpu_production", "gpu_production_v2", "diagnostics")
                 for path in (package / directory).rglob("*.py")]
        self.assertTrue(files)
        before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        self.plan([row(1), row(2)])
        after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        self.assertEqual(before, after)
        self.assertEqual(p.PRODUCTION_COMMIT, "f594e32705581d8bcfdd609560eda87fd11cbc31")
        self.assertIn("--mem=66G", p.PLANNED_ARGS)
        self.assertIn("--time=48:00:00", p.PLANNED_ARGS)


if __name__ == "__main__":
    unittest.main()
