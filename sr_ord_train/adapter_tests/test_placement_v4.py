"""Positive zero-GPU evidence and retained GPU admission safety for v4."""
import contextlib
import copy
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scheduling import placement_v4 as p


OWNER = "researcher"


def ledger(*requests):
    return {"at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "basis": "All submission records and terminal receipts audited.",
            "unresolved_requests": list(requests)}


def cpu_queue(job="383711", owner=OWNER, qos="override-limits-but-killable",
              name="scarce-launch-gate-SM", state="PENDING", tres="N/A", nodes=1):
    return f"{job}|{owner}|{qos}|{name}|{state}|{tres}|{nodes}"


def cpu_detail(job="383711", owner=OWNER, qos="override-limits-but-killable",
               name="scarce-launch-gate-SM", state="PENDING", nodes="1",
               requested="cpu=1,mem=3G,node=1", allocated="(null)", **extra):
    fields = {"JobId": job, "JobName": name, "UserId": owner + "(1000)",
              "QOS": qos, "JobState": state, "NumNodes": nodes,
              "ReqTRES": requested, "AllocTRES": allocated, "TresPerTask": "cpu:1"}
    fields.update(extra)
    return " ".join(k + "=" + str(v) for k, v in fields.items() if v is not None)


def gpu_queue(job="1", name="LH-ORD-EVAL", state="RUNNING", count=2, qos=p.TARGET_QOS):
    return f"{job}|{OWNER}|{qos}|{name}|{state}|gres/gpu:pro6000:{count}|1"


def gpu_detail(job="1", name="LH-ORD-EVAL", state="RUNNING", count=2,
               qos=p.TARGET_QOS, per_job=False):
    resources = f"cpu=8,mem=66G,node=1,gres/gpu={count},gres/gpu:pro6000={count}"
    fields = dict(JobId=job, JobName=name, UserId=OWNER + "(1000)", QOS=qos,
                  JobState=state, NumNodes="1", ReqTRES=resources,
                  AllocTRES=resources if state != "PENDING" else "(null)")
    fields["TresPerJob" if per_job else "TresPerNode"] = f"gres/gpu:pro6000:{count}"
    return " ".join(k + "=" + str(v) for k, v in fields.items())


class CPUOnlyEvidenceTests(unittest.TestCase):
    def plan_cpu(self, queue=None, detail=None, audit=None):
        queue = cpu_queue() if queue is None else queue
        detail = cpu_detail() if detail is None else detail
        return p.plan(queue, ledger() if audit is None else audit, OWNER,
                      job_details={queue.split("|", 1)[0]: detail})

    def test_real_pending_cpu_only_row_is_retained_with_zero_cards(self):
        for tres in ("N/A", "(null)"):
            with self.subTest(tres=tres):
                result = self.plan_cpu(queue=cpu_queue(tres=tres))
                self.assertEqual(result["schema"], "SR_ORD_PLACEMENT_V4")
                self.assertEqual(result["status"], "READY")
                self.assertEqual(result["before"], {"experiment_gpus": 0,
                    "shared_target_qos_pro6000": 0, "owner_target_qos_jobs": 0})
                self.assertEqual(result["other_qos_job_ids"], ["383711"])
                self.assertEqual(len(result["requests"]), 1)
                request = result["requests"][0]
                self.assertEqual((request["gpu_type"], request["gpus_per_node"],
                                  request["total_gpus"]), ("none", 0, 0))
                self.assertIn("383711", result["scheduler_observation"]["scontrol"])
                self.assertFalse(result["submission_performed"])

    def test_running_cpu_requires_matching_positive_cpu_and_node_allocation(self):
        result = self.plan_cpu(queue=cpu_queue(state="RUNNING"),
            detail=cpu_detail(state="RUNNING", allocated="cpu=1,mem=3G,node=1"))
        self.assertEqual(result["requests"][0]["state"], "RUNNING")
        self.assertEqual(result["requests"][0]["total_gpus"], 0)
        for alloc in (None, "(null)", "N/A", "cpu=1", "node=1", "cpu=0,node=1",
                      "cpu=2,node=1", "cpu=1,node=2", "cpu=1,node=0"):
            with self.subTest(alloc=alloc), self.assertRaises(p.UnsafeInput):
                self.plan_cpu(queue=cpu_queue(state="RUNNING"),
                              detail=cpu_detail(state="RUNNING", allocated=alloc))

    def test_zero_gpu_target_qos_jobs_still_consume_submission_slots(self):
        rows, details = [], {}
        for job in map(str, range(1, 6)):
            rows.append(cpu_queue(job=job, qos=p.TARGET_QOS))
            details[job] = cpu_detail(job=job, qos=p.TARGET_QOS)
        result = p.plan("\n".join(rows), ledger(), OWNER, job_details=details)
        self.assertEqual(result["before"]["owner_target_qos_jobs"], 5)
        self.assertEqual(result["before"]["shared_target_qos_pro6000"], 0)
        self.assertEqual(result["blocking_limits"], ["owner_target_qos_jobs"])
        self.assertEqual(result["status"], "BLOCKED_CAPACITY")
        self.assertEqual(len(result["requests"]), 5)

    def test_other_owners_cpu_job_in_target_qos_does_not_consume_owner_slot(self):
        result = self.plan_cpu(queue=cpu_queue(owner="other", qos=p.TARGET_QOS),
                              detail=cpu_detail(owner="other", qos=p.TARGET_QOS))
        self.assertEqual(result["before"]["owner_target_qos_jobs"], 0)
        self.assertEqual(len(result["requests"]), 1)

    def test_na_without_positive_detail_is_never_assumed_zero(self):
        for details in (None, {}, {"383711": ""}):
            with self.subTest(details=details), self.assertRaises(p.UnsafeInput):
                p.plan(cpu_queue(), ledger(), OWNER, job_details=details)
        for req in (None, "N/A", "(null)", "node=1", "cpu=1", "cpu=0,node=1",
                    "cpu=-1,node=1", "cpu=1.0,node=1", "cpu=1,node=0"):
            with self.subTest(req=req), self.assertRaises(p.UnsafeInput):
                self.plan_cpu(detail=cpu_detail(requested=req))

    def test_gpu_mentions_in_totals_never_become_cpu_only(self):
        for field in ("requested", "allocated"):
            for gpu in ("gres/gpu=1", "gres/gpu:pro6000=1", "gres/gpu=0",
                        "gres/gpu:pro6000=0", "gres/gpu", "gpu=1", "gres/gpu:a100=2"):
                with self.subTest(field=field, gpu=gpu), self.assertRaises(p.UnsafeInput):
                    self.plan_cpu(detail=cpu_detail(**{field: "cpu=1,node=1," + gpu}))

    def test_gpu_in_every_request_mode_is_not_ignored(self):
        for field in ("TresPerJob", "TresPerNode", "TresPerTask", "TresPerSocket"):
            for value in ("gres/gpu:1", "gres/gpu:pro6000:1", "gres/gpu:pro6000:0",
                          "cpu:1,gres/gpu:pro6000:1", "gpu:1"):
                with self.subTest(field=field, value=value), self.assertRaises(p.UnsafeInput):
                    self.plan_cpu(detail=cpu_detail(**{field: value}))

    def test_malformed_and_duplicate_totals_fail_closed(self):
        for field in ("requested", "allocated"):
            for value in ("cpu=1,node=1,cpu=1", "cpu=1,node=1,node=1", "cpu=1,node=1,",
                          "cpu=1,,node=1", "cpu=1,node=1,broken", "cpu=1=node=1",
                          "cpu=1,node=1,unknown=1"):
                with self.subTest(field=field, value=value), self.assertRaises(p.UnsafeInput):
                    self.plan_cpu(detail=cpu_detail(**{field: value}))

    def test_malformed_request_modes_fail_closed(self):
        for field in ("TresPerJob", "TresPerNode", "TresPerTask", "TresPerSocket"):
            for value in ("cpu", "cpu:1,cpu:1", "cpu:1,", "cpu:1,,mem:3G",
                          "unknown:1", "cpu:0", "cpu:-1"):
                with self.subTest(field=field, value=value), self.assertRaises(p.UnsafeInput):
                    self.plan_cpu(detail=cpu_detail(**{field: value}))

    def test_additional_resource_fields_cannot_hide_unexamined_gpu_evidence(self):
        for field in ("Gres", "GresDetail", "TresPerUnknown", "OtherTRES"):
            for value in ("gpu:pro6000:1", "(null)"):
                with self.subTest(field=field, value=value), self.assertRaises(p.UnsafeInput):
                    self.plan_cpu(detail=cpu_detail(**{field: value}))

    def test_memory_units_and_request_modes_must_agree_with_request(self):
        result = self.plan_cpu(queue=cpu_queue(state="RUNNING"), detail=cpu_detail(
            state="RUNNING", allocated="cpu=1,mem=3072M,node=1",
            TresPerNode="cpu:1,mem:3072M", NumCPUs="1"))
        self.assertEqual(result["requests"][0]["total_gpus"], 0)
        for changes in ({"allocated": "cpu=1,mem=4G,node=1"},
                        {"allocated": "cpu=1,node=1,billing=1"},
                        {"TresPerTask": "cpu:2"}, {"TresPerNode": "mem:4G"},
                        {"TresPerSocket": "billing:1"}, {"TresPerJob": "N/A"},
                        {"NumCPUs": "2"}, {"NumCPUs": "0"}):
            with self.subTest(changes=changes), self.assertRaises(p.UnsafeInput):
                self.plan_cpu(detail=cpu_detail(**changes))

    def test_duplicate_scontrol_evidence_is_ambiguous(self):
        for field, value in (("JobId", "383711"), ("JobName", "scarce-launch-gate-SM"),
                             ("UserId", OWNER + "(1000)"), ("QOS", "override-limits-but-killable"),
                             ("JobState", "PENDING"), ("NumNodes", "1"),
                             ("ReqTRES", "cpu=1,node=1"), ("AllocTRES", "(null)"),
                             ("TresPerTask", "cpu:1")):
            with self.subTest(field=field), self.assertRaises(p.UnsafeInput):
                self.plan_cpu(detail=cpu_detail() + " " + field + "=" + value)

    def test_empty_optional_resource_fields_are_not_treated_as_absent(self):
        for field in ("TresPerJob", "TresPerNode", "TresPerTask", "TresPerSocket",
                      "AllocTRES", "NumCPUs"):
            with self.subTest(field=field), self.assertRaises(p.UnsafeInput):
                self.plan_cpu(detail=cpu_detail(**{field: ""}))

    def test_empty_unknown_resource_fields_do_not_disappear(self):
        for field in ("Gres", "GresDetail", "TresPerUnknown", "OtherTRES"):
            with self.subTest(field=field), self.assertRaises(p.UnsafeInput):
                self.plan_cpu(detail=cpu_detail(**{field: ""}))

    def test_empty_critical_fields_and_empty_duplicates_fail_closed(self):
        for field in ("JobId", "JobName", "UserId", "QOS", "JobState", "NumNodes", "ReqTRES"):
            cases = (cpu_detail(**{field: ""}),
                     cpu_detail() + " " + field + "=",
                     field + "= " + cpu_detail())
            for index, detail in enumerate(cases):
                with self.subTest(field=field, index=index), self.assertRaises(p.UnsafeInput):
                    self.plan_cpu(detail=detail)

    def test_identity_state_and_node_disagreements_fail_closed(self):
        for changes in ({"job": "383712"}, {"owner": "other"}, {"qos": p.TARGET_QOS},
                        {"name": "OTHER"}, {"state": "RUNNING"}, {"state": "COMPLETED"},
                        {"nodes": "2"}, {"nodes": "1-2"}, {"requested": "cpu=1,node=2"}):
            with self.subTest(changes=changes), self.assertRaises(p.UnsafeInput):
                self.plan_cpu(detail=cpu_detail(**changes))
        with self.assertRaises(p.UnsafeInput):
            self.plan_cpu(queue=cpu_queue(nodes=2),
                          detail=cpu_detail(nodes="2", requested="cpu=1,node=2"))

    def test_experiment_zero_gpu_identity_is_rejected(self):
        for qos in (p.TARGET_QOS, "rose"):
            with self.subTest(qos=qos), self.assertRaises(p.UnsafeInput):
                self.plan_cpu(queue=cpu_queue(name="LH-ORD-X", qos=qos),
                              detail=cpu_detail(name="LH-ORD-X", qos=qos))

    def test_unresolved_ledger_cannot_claim_zero_gpu_or_override_cpu_proof(self):
        entry = dict(job_id="383711", owner=OWNER, qos="override-limits-but-killable",
                     name="scarce-launch-gate-SM", gpu_type="none", gpus_per_node=0, nodes=1)
        with self.assertRaises(p.UnsafeInput):
            self.plan_cpu(audit=ledger(entry))
        with self.assertRaises(p.UnsafeInput):
            self.plan_cpu(audit=ledger(dict(entry, gpu_type="pro6000", gpus_per_node=2)))

    def test_cpu_duplicates_deduplicate_but_identity_changes_do_not(self):
        queue = cpu_queue()
        result = p.plan(queue + "\n" + queue, ledger(), OWNER,
                        job_details={"383711": cpu_detail()})
        self.assertEqual(len(result["requests"]), 1)
        with self.assertRaises(p.UnsafeInput):
            p.plan(queue + "\n" + cpu_queue(name="OTHER"), ledger(), OWNER,
                   job_details={"383711": cpu_detail()})

    def test_input_mappings_are_not_mutated(self):
        audit = ledger(); details = {"383711": cpu_detail()}
        original = copy.deepcopy((audit, details))
        p.plan(cpu_queue(), audit, OWNER, job_details=details)
        self.assertEqual((audit, details), original)


class GPUAndCLICompatibilityTests(unittest.TestCase):
    def test_cpu_rows_do_not_hide_gpu_experiment_or_shared_caps(self):
        rows = [cpu_queue()]; details = {"383711": cpu_detail()}
        for job in ("1", "2", "3"):
            rows.append(gpu_queue(job)); details[job] = gpu_detail(job)
        result = p.plan("\n".join(rows), ledger(), OWNER, job_details=details)
        self.assertEqual(result["before"]["experiment_gpus"], 6)
        self.assertIn("experiment_gpus", result["blocking_limits"])
        self.assertEqual(len(result["requests"]), 4)
        self.assertEqual(result["planned_resource_args"][:-1], list(p.PLANNED_ARGS))
        self.assertEqual(result["production_commit"], "f594e32705581d8bcfdd609560eda87fd11cbc31")

    def test_valid_job_total_gpu_is_still_gpu_not_zero(self):
        queue = gpu_queue(state="PENDING").replace("gres/gpu:pro6000:2", "N/A")
        result = p.plan(queue, ledger(), OWNER,
                        job_details={"1": gpu_detail(state="PENDING", per_job=True)})
        self.assertEqual(result["requests"][0]["total_gpus"], 2)
        self.assertEqual(result["before"]["experiment_gpus"], 2)
        self.assertEqual(result["status"], "READY")
        with self.assertRaises(p.UnsafeInput):
            p.plan(queue, ledger(), OWNER, job_details={"1": gpu_detail(
                state="PENDING", per_job=True).replace("gres/gpu=2", "gres/gpu=1")})

    def test_typed_gpu_allocation_mismatch_still_blocks(self):
        with self.assertRaises(p.UnsafeInput):
            p.plan(gpu_queue(), ledger(), OWNER, job_details={"1": gpu_detail().replace(
                "AllocTRES=cpu=8,mem=66G,node=1,gres/gpu=2,gres/gpu:pro6000=2",
                "AllocTRES=cpu=8,node=1,gres/gpu=4,gres/gpu:pro6000=2")})

    def test_unknown_gpu_requests_remain_conservatively_counted(self):
        unknown = dict(job_id="9", owner=OWNER, qos=p.TARGET_QOS, name="LH-ORD-UNKNOWN",
                       gpu_type="pro6000", gpus_per_node=6, nodes=1)
        result = p.plan(cpu_queue(), ledger(unknown), OWNER,
                        job_details={"383711": cpu_detail()})
        self.assertEqual(result["before"]["experiment_gpus"], 6)
        self.assertEqual(next(r for r in result["requests"] if r["job_id"] == "9")["state"], "UNKNOWN")
        self.assertEqual(result["status"], "BLOCKED_CAPACITY")

    def test_cpu_exception_does_not_relax_stale_ledger_or_terminal_queue(self):
        audit = ledger(); audit["at_utc"] = (datetime.now(timezone.utc) - timedelta(minutes=16)).isoformat().replace("+00:00", "Z")
        for queue, detail, recorded in ((cpu_queue(), cpu_detail(), audit),
                (cpu_queue(state="COMPLETED"), cpu_detail(state="COMPLETED"), ledger())):
            with self.subTest(queue=queue), self.assertRaises(p.UnsafeInput):
                p.plan(queue, recorded, OWNER, job_details={"383711": detail})

    def test_cli_keeps_cpu_and_gpu_rows_and_only_calls_read_only_commands(self):
        queue = cpu_queue() + "\n" + gpu_queue()
        details = {"383711": cpu_detail(), "1": gpu_detail()}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ledger.json"; path.write_text(json.dumps(ledger()))
            output = io.StringIO()
            def respond(command, **kwargs):
                self.assertTrue(kwargs["check"]); self.assertNotIn("shell", kwargs)
                if command[0] == "squeue":
                    self.assertIn("--all", command); self.assertIn("--states=all", command)
                    return subprocess.CompletedProcess(command, 0, queue, "")
                self.assertEqual(command[:4], ["scontrol", "show", "job", "-o"])
                return subprocess.CompletedProcess(command, 0, details[command[-1]], "")
            with patch.object(p.subprocess, "run", side_effect=respond) as run:
                with contextlib.redirect_stdout(output):
                    code = p.main(["--owner", OWNER, "--unresolved-ledger", str(path)])
            result = json.loads(output.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(result["schema"], "SR_ORD_PLACEMENT_V4")
            self.assertEqual({call.args[0][-1] for call in run.call_args_list if call.args[0][0] == "scontrol"}, set(details))
            self.assertEqual(len(result["requests"]), 2)
            self.assertEqual(result["scheduler_observation"]["squeue"], queue)
            self.assertFalse(result["submission_performed"])


if __name__ == "__main__":
    unittest.main()
