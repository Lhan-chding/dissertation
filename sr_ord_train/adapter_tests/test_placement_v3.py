"""V3 safety regressions, retaining v2 coverage without changing historical tests."""
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

from scheduling import placement_v3 as p


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
        self.assertIn("--exclude=gpu-pro6000-6,gpu-pro6000-11,gpu-pro6000-12", result["planned_resource_args"])
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

class FixedNodeRepresentationTests(unittest.TestCase):
    def detail(self, nodes=1):
        return (f'JobId=275612 JobName=LH-ORD-X JobState=PENDING UserId={OWNER}(1000) '
                f'QOS={p.TARGET_QOS} NumNodes={nodes}-{nodes} '
                f'ReqTRES=cpu=8,node={nodes},gres/gpu={2*nodes},gres/gpu:pro6000={2*nodes} AllocTRES=(null)')

    def test_equal_range_counts_full_request(self):
        for nodes in (1,2,3):
            queued=row(275612,state='PENDING',nodes=nodes)
            p.verify_job_detail(self.detail(nodes),p.parse_queue(queued)[0]['275612'],'PENDING')
            result=p.plan(queued,ledger(),OWNER)
            self.assertEqual(result['before']['experiment_gpus'],2*nodes)
            self.assertEqual(result['schema'],'SR_ORD_PLACEMENT_V3')
            self.assertEqual(result['status'],'READY' if nodes<3 else 'BLOCKED_CAPACITY')

    def test_variable_and_malformed_ranges_rejected(self):
        for value in ('1-2','2-1','0-0','01-01','1-01','1-1-1','1-','1..1','-1','1 - 1'):
            with self.subTest(value=value),self.assertRaises(p.UnsafeInput):
                p.fixed_node_count(value)

    def test_equal_range_cannot_bypass_crosschecks(self):
        req=p.parse_queue(row(275612,state='PENDING'))[0]['275612'];valid=self.detail()
        for bad in (valid.replace('NumNodes=1-1','NumNodes=2-2'),valid.replace('node=1','node=2'),
                    valid.replace('gres/gpu:pro6000=2','gres/gpu:pro6000=4'),valid.replace('gres/gpu=2','gres/gpu=4'),
                    valid+' NumNodes=1-1',valid.replace('JobState=PENDING','JobState=COMPLETED')):
            with self.subTest(bad=bad),self.assertRaises(p.UnsafeInput):p.verify_job_detail(bad,req,'PENDING')

    def test_cli_regression_and_v1_unchanged(self):
        from scheduling import placement_v1
        queued=row(275612,state='PENDING');detail=self.detail()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'ledger.json';path.write_text(json.dumps(ledger()));output=io.StringIO()
            results=[subprocess.CompletedProcess([],0,queued,''),subprocess.CompletedProcess([],0,detail,'')]
            with patch.object(p.subprocess,'run',side_effect=results),contextlib.redirect_stdout(output):
                code=p.main(['--owner',OWNER,'--unresolved-ledger',str(path)])
            self.assertEqual(code,0);self.assertEqual(json.loads(output.getvalue())['before']['experiment_gpus'],2)
        with self.assertRaises(placement_v1.UnsafeInput):
            placement_v1.verify_job_detail(detail,placement_v1.parse_queue(queued)[0]['275612'],'PENDING')


class JobTotalGPURequestTests(unittest.TestCase):
    """A missing per-node display is resolved from evidence, never treated as zero."""

    @staticmethod
    def queue(job="278509", owner=OWNER, qos="rose", name="UNRELATED",
              state="PENDING", resource="N/A", nodes=1):
        return f"{job}|{owner}|{qos}|{name}|{state}|{resource}|{nodes}"

    @staticmethod
    def detail(job="278509", owner=OWNER, qos="rose", name="UNRELATED",
               state="PENDING", gpu_type="6000ada", count=4, nodes="1-1",
               allocation="(null)"):
        return (f"JobId={job} JobName={name} UserId={owner}(1000) QOS={qos} "
                f"JobState={state} NumNodes={nodes} "
                f"ReqTRES=cpu=16,mem=256G,node=1,gres/gpu={count},gres/gpu:{gpu_type}={count} "
                f"AllocTRES={allocation} TresPerJob=gres/gpu:{gpu_type}:{count} TresPerTask=cpu:16")

    def plan(self, queue=None, detail=None, audit=None):
        queue = self.queue() if queue is None else queue
        detail = self.detail() if detail is None else detail
        return p.plan(queue, ledger() if audit is None else audit, OWNER,
                      job_details={"278509": detail})

    def cli(self, queue, details, audit=None):
        """Only mocked, fully observed read-only commands are allowed."""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ledger.json"
            path.write_text(json.dumps(ledger() if audit is None else audit))
            output = io.StringIO()
            def respond(command, **kwargs):
                self.assertTrue(kwargs["check"])
                self.assertNotIn("shell", kwargs)
                self.assertFalse(any(k.startswith("SQUEUE_") for k in kwargs["env"]))
                if command[0] == "squeue":
                    self.assertEqual(command, ["squeue", "--all", "--states=all", "--noheader",
                                              "--format=" + p.SQUEUE_FORMAT])
                    return subprocess.CompletedProcess(command, 0, queue, "")
                self.assertEqual(command[:4], ["scontrol", "show", "job", "-o"])
                response = details[command[-1]]
                if isinstance(response, Exception):
                    raise response
                if isinstance(response, subprocess.CompletedProcess):
                    return response
                return subprocess.CompletedProcess(command, 0, response, "")
            with patch.object(p.subprocess, "run", side_effect=respond) as run:
                with contextlib.redirect_stdout(output):
                    code = p.main(["--owner", OWNER, "--unresolved-ledger", str(path)])
            return code, json.loads(output.getvalue()), [call.args[0] for call in run.call_args_list]

    def test_actual_job_total_other_qos_is_visible_and_does_not_consume_target_cap(self):
        for resource in ("N/A", "(null)"):
            with self.subTest(resource=resource):
                result = self.plan(queue=self.queue(resource=resource))
                self.assertEqual(result["status"], "READY")
                self.assertEqual(result["before"], {"experiment_gpus": 0,
                    "shared_target_qos_pro6000": 0, "owner_target_qos_jobs": 0})
                self.assertEqual(result["other_qos_job_ids"], ["278509"])
                self.assertEqual(result["requests"][0]["total_gpus"], 4)
                self.assertEqual(result["requests"][0]["gpu_type"], "6000ada")
                self.assertEqual(result["requests"][0]["state"], "PENDING")
                self.assertFalse(result["submission_performed"])

    def test_target_pending_total_fully_counts_and_blocks_shared_cap(self):
        result = self.plan(queue=self.queue(qos=p.TARGET_QOS),
                           detail=self.detail(qos=p.TARGET_QOS, gpu_type="pro6000", count=8))
        self.assertEqual(result["before"]["shared_target_qos_pro6000"], 8)
        self.assertEqual(result["before"]["owner_target_qos_jobs"], 1)
        self.assertEqual(result["blocking_limits"], ["shared_target_qos_pro6000"])
        self.assertEqual(result["status"], "BLOCKED_CAPACITY")

    def test_other_qos_experiment_total_still_counts_experiment_cap(self):
        result = self.plan(queue=self.queue(name="LH-ORD-OTHER"),
                           detail=self.detail(name="LH-ORD-OTHER", count=6))
        self.assertEqual(result["before"]["experiment_gpus"], 6)
        self.assertEqual(result["before"]["shared_target_qos_pro6000"], 0)
        self.assertEqual(result["blocking_limits"], ["experiment_gpus"])

    def test_five_job_limit_includes_total_gpu_rows(self):
        queues, details = [], {}
        for job in range(1, 6):
            queues.append(self.queue(job=str(job), qos=p.TARGET_QOS))
            details[str(job)] = self.detail(job=str(job), qos=p.TARGET_QOS,
                                            gpu_type="pro6000", count=1)
        result = p.plan("\n".join(queues), ledger(), OWNER, job_details=details)
        self.assertEqual(result["before"]["owner_target_qos_jobs"], 5)
        self.assertEqual(result["before"]["shared_target_qos_pro6000"], 5)
        self.assertEqual(result["blocking_limits"], ["owner_target_qos_jobs"])

    def test_unknown_overlap_deduplicates_exactly_and_conflicts_fail_closed(self):
        entry = unknown("278509", qos="rose", name="UNRELATED",
                        gpu_type="6000ada", gpus_per_node=4)
        result = self.plan(audit=ledger(entry))
        self.assertEqual(len(result["requests"]), 1)
        self.assertEqual(result["requests"][0]["total_gpus"], 4)
        self.assertTrue(result["requests"][0]["in_unresolved_ledger"])
        for field, value in (("gpus_per_node", 2), ("owner", "other"), ("nodes", 2),
                             ("gpu_type", "pro6000"), ("name", "OTHER")):
            with self.subTest(field=field), self.assertRaises(p.UnsafeInput):
                self.plan(audit=ledger(dict(entry, **{field: value})))

    def test_absent_unknown_experiment_remains_counted(self):
        result = self.plan(audit=ledger(unknown("99", gpus_per_node=6)))
        self.assertEqual(result["before"]["experiment_gpus"], 6)
        self.assertEqual(result["status"], "BLOCKED_CAPACITY")
        absent = next(r for r in result["requests"] if r["job_id"] == "99")
        self.assertEqual(absent["state"], "UNKNOWN")

    def test_missing_or_ambiguous_job_total_evidence_blocks(self):
        valid = self.detail()
        invalid = [
            "", valid.replace(" TresPerJob=gres/gpu:6000ada:4", ""),
            valid.replace("TresPerJob=gres/gpu:6000ada:4", "TresPerJob=(null)"),
            valid.replace("TresPerJob=gres/gpu:6000ada:4", "TresPerJob=gres/gpu:4"),
            valid.replace("TresPerJob=gres/gpu:6000ada:4", "TresPerJob=gres/gpu:6000ada:0"),
            valid + " TresPerJob=gres/gpu:6000ada:4",
            valid.replace("TresPerJob=gres/gpu:6000ada:4", "TresPerJob=gres/gpu:6000ada:4,gres/gpu:a100:1"),
            valid.replace("gres/gpu=4,", ""),
            valid.replace(",gres/gpu:6000ada=4", ""),
            valid.replace("gres/gpu=4", "gres/gpu=2"),
            valid.replace("gres/gpu:6000ada=4", "gres/gpu:6000ada=2"),
            valid.replace("TresPerJob=gres/gpu:6000ada:4", "TresPerJob=gres/gpu:6000ada:2"),
            valid.replace("TresPerJob=gres/gpu:6000ada:4", "TresPerJob=gres/gpu:pro6000:4"),
            valid.replace("gres/gpu:6000ada=4", "gres/gpu:6000ada=4,gres/gpu:a100=1"),
            valid.replace("gres/gpu:6000ada=4", "gres/gpu:6000ada=4,gres/gpu:6000ada=4"),
            valid + " ReqTRES=node=1,gres/gpu=4,gres/gpu:6000ada=4",
            valid.replace("ReqTRES=cpu=16,mem=256G,node=1,gres/gpu=4,gres/gpu:6000ada=4",
                          "ReqTRES=cpu=16,mem=256G,node=1"),
        ]
        for detail in invalid:
            with self.subTest(detail=detail), self.assertRaises(p.UnsafeInput):
                self.plan(detail=detail)

    def test_conflicting_gpu_request_forms_fail_closed(self):
        valid = self.detail()
        for change in (" TresPerNode=gres/gpu:6000ada:4", " TresPerSocket=gres/gpu:6000ada:4",
                       " TresPerTask=gres/gpu:6000ada:4"):
            with self.subTest(change=change), self.assertRaises(p.UnsafeInput):
                self.plan(detail=valid.replace(" TresPerTask=cpu:16", "") + change)

    def test_identity_state_node_changes_or_terminal_detail_fail_closed(self):
        valid = self.detail()
        replacements = (("JobId=278509", "JobId=278510"), ("JobName=UNRELATED", "JobName=OTHER"),
                        (f"UserId={OWNER}(1000)", "UserId=other(1000)"),
                        ("QOS=rose", "QOS=" + p.TARGET_QOS),
                        ("JobState=PENDING", "JobState=COMPLETED"),
                        ("JobState=PENDING", "JobState=RUNNING"),
                        ("NumNodes=1-1", "NumNodes=1-2"), ("NumNodes=1-1", "NumNodes=2-2"),
                        ("node=1", "node=2"))
        for old, new in replacements:
            with self.subTest(new=new), self.assertRaises(p.UnsafeInput):
                self.plan(detail=valid.replace(old, new))
        for field in ("JobId=278509", "JobName=UNRELATED", f"UserId={OWNER}(1000)", "QOS=rose",
                      "JobState=PENDING", "NumNodes=1-1"):
            with self.subTest(field=field), self.assertRaises(p.UnsafeInput):
                self.plan(detail=valid + " " + field)
        with self.assertRaises(p.UnsafeInput):
            self.plan(queue=self.queue(nodes=2), detail=valid.replace("NumNodes=1-1", "NumNodes=2-2").replace("node=1", "node=2"))

    def test_running_total_requires_valid_allocation(self):
        queue = self.queue(state="RUNNING")
        valid = self.detail(state="RUNNING", allocation="cpu=16,node=1,gres/gpu=4,gres/gpu:6000ada=4")
        self.assertEqual(self.plan(queue=queue, detail=valid)["requests"][0]["total_gpus"], 4)
        for allocation in ("(null)", "N/A", "cpu=16", "gres/gpu=5,gres/gpu:6000ada=5",
                           "gres/gpu=4,gres/gpu:pro6000=4", "gres/gpu=4,gres/gpu:6000ada=3"):
            with self.subTest(allocation=allocation), self.assertRaises(p.UnsafeInput):
                self.plan(queue=queue, detail=self.detail(state="RUNNING", allocation=allocation))
        with self.assertRaises(p.UnsafeInput):
            self.plan(queue=queue, detail=valid.replace(" AllocTRES=cpu=16,node=1,gres/gpu=4,gres/gpu:6000ada=4", ""))

    def test_plan_without_details_cannot_accept_na(self):
        for details in (None, {}):
            with self.subTest(details=details), self.assertRaises(p.UnsafeInput):
                p.plan(self.queue(), ledger(), OWNER, job_details=details)

    def test_supplied_details_also_verify_typed_queue_rows(self):
        queue = row(1)
        valid = (f"JobId=1 JobName=LH-ORD-X JobState=RUNNING UserId={OWNER}(1000) "
                 f"QOS={p.TARGET_QOS} NumNodes=1 ReqTRES=node=1,gres/gpu=2,gres/gpu:pro6000=2 "
                 "AllocTRES=node=1,gres/gpu=2,gres/gpu:pro6000=2")
        for details in ({}, {"1": valid.replace("JobId=1", "JobId=2")}):
            with self.subTest(details=details), self.assertRaises(p.UnsafeInput):
                p.plan(queue, ledger(), OWNER, job_details=details)
        self.assertEqual(p.plan(queue, ledger(), OWNER, job_details={"1": valid})["status"], "READY")

    def test_cli_queries_every_relevant_row_and_keeps_raw_observations(self):
        queue = "\n".join((self.queue(), self.queue(job="11", qos=p.TARGET_QOS, resource="gres/gpu:pro6000:1", state="RUNNING")))
        typed = (f"JobId=11 JobName=UNRELATED UserId={OWNER}(1000) QOS={p.TARGET_QOS} "
                 "JobState=RUNNING NumNodes=1 ReqTRES=node=1,gres/gpu=1,gres/gpu:pro6000=1 "
                 "AllocTRES=node=1,gres/gpu=1,gres/gpu:pro6000=1")
        details = {"278509": self.detail(), "11": typed}
        with patch.dict(os.environ, {"SQUEUE_USERS": "filtered", "SQUEUE_STATES": "RUNNING"}):
            environment = dict(os.environ)
            code, report, commands = self.cli(queue, details)
            self.assertEqual(dict(os.environ), environment)
        self.assertEqual(code, 0)
        self.assertEqual(report["schema"], "SR_ORD_PLACEMENT_V3")
        self.assertEqual(report["scontrol_verified_job_ids"], ["11", "278509"])
        self.assertEqual({c[-1] for c in commands if c[0] == "scontrol"}, set(details))
        self.assertEqual(len(commands), 3)
        self.assertEqual(report["before"]["shared_target_qos_pro6000"], 1)
        self.assertEqual(report["other_qos_job_ids"], ["278509"])
        observation = json.dumps(report["scheduler_observation"])
        self.assertIn(queue.splitlines()[0], observation)
        self.assertIn("TresPerJob=gres/gpu:6000ada:4", observation)
        self.assertFalse(report["submission_performed"])

    def test_cli_disappeared_job_or_diagnostic_stderr_never_ready(self):
        for response in (subprocess.CalledProcessError(1, "scontrol"),
                         subprocess.TimeoutExpired("scontrol", 15), "",
                         subprocess.CompletedProcess([], 0, self.detail(), "warning")):
            with self.subTest(response=response):
                code, report, _ = self.cli(self.queue(), {"278509": response})
                self.assertEqual(code, 2)
                self.assertEqual(report["status"], "BLOCKED_INPUT")
                self.assertFalse(report["submission_performed"])

    def test_old_versions_still_reject_na_and_remain_unmodified(self):
        from scheduling import placement_v1, placement_v2
        files = [Path(placement_v1.__file__), Path(placement_v2.__file__)]
        before = [hashlib.sha256(f.read_bytes()).hexdigest() for f in files]
        for old in (placement_v1, placement_v2):
            with self.assertRaises(old.UnsafeInput):
                old.plan(self.queue(), ledger(), OWNER)
        self.plan()
        self.assertEqual(before, [hashlib.sha256(f.read_bytes()).hexdigest() for f in files])

    def test_na_resolution_does_not_mutate_inputs(self):
        details = {"278509": self.detail()}
        audit = ledger()
        before = copy.deepcopy((details, audit))
        p.plan(self.queue(), audit, OWNER, job_details=details)
        self.assertEqual((details, audit), before)


if __name__ == "__main__":
    unittest.main()
