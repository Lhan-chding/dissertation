"""Full 34-path CPU gate simulations, with virtual immutable payload files.

The production cardinalities, schedules, seeds and checkpoint field schema are
kept. Only file IO/hash boundaries and the already-tested panel loader are mocked;
no model is constructed and no Slurm command is issued.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from mm_core.training import state_hash
from mm_dev import common
from mm_dev.contract import PLAN_ID, digest, run_matrix, seed
from mm_dev.orchestration import digest as registration_digest
from mm_dev.orchestration import summarize_accounting, tasks_for

REPOSITORY = Path(__file__).resolve().parents[2]
PLAN_PATH = REPOSITORY / "docs/mm_dev_f2/design/config/MM_DEV_F2.json"


def fixture_hash(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


class VirtualCompleteRun:
    def __init__(self, root):
        self.root = root
        self.plan = json.loads(PLAN_PATH.read_text())
        self.matrix = run_matrix()
        self.runs = {run["run_id"]: run for run in self.matrix}
        self.streams = {
            schedule: [
                json.loads(line)
                for line in (PLAN_PATH.parent / "schedules" / (schedule + ".jsonl"))
                .read_text()
                .splitlines()
            ]
            for schedule in {run["schedule_id"] for run in self.matrix}
        }
        self.slots = {
            key: {(row["logical_step"], row["slot"]): row for row in rows}
            for key, rows in self.streams.items()
        }
        self.questions = {}
        for rows in self.streams.values():
            for row in rows:
                qid = row["question_id"]
                truth = [30, 20, 40] if row["operation"] == "range" else [30, 20]
                self.questions[qid] = {
                    "question_id": qid,
                    "image_path": "images/" + qid + ".png",
                    "image_sha256": fixture_hash(qid),
                    "prompt": "frozen visible prompt",
                    "true_values_decimal": list(map(str, truth)),
                    "operation": row["operation"],
                    "delta_decimal": "1",
                    "chart_type": row["chart_type"],
                }
        (root / "data").mkdir()
        (root / "data/questions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.questions.values())
        )
        (root / "accounting").mkdir()
        self.ledger = root / "accounting/ENGINE_F2.jsonl"
        self.ledger.write_text('{"kind":"completion_attempts","count":1}\n')
        self.permission = {"owner": "u", "account": "a", "qos": "q"}
        self.registration = {
            "plan_id": PLAN_ID,
            "tasks": tasks_for(self.matrix),
            "permission": self.permission,
            "measurement_shards": 1,
        }
        self.scheduler_state = {"tasks": {}}
        self.cost = {
            "status": "COMPLETE",
            "plan_id": PLAN_ID,
            "policy": "ACCOUNTING_ONLY",
            "registration_hash": registration_digest(self.registration),
            "allocations": [],
        }
        self.observations = {}
        for index, (task_id, task) in enumerate(self.registration["tasks"].items()):
            attempt = {
                "attempt_id": task_id + "_attempt0000",
                "job_id": str(1000 + index),
                "job_name": "job-" + task_id,
                "comment": "bound-" + task_id,
            }
            self.scheduler_state["tasks"][task_id] = {"status": "COMPLETE", "attempts": [attempt]}
            if not task["gpus"]:
                continue
            relative = "orchestration/observations/" + attempt["attempt_id"] + ".json"
            accounting = [
                {
                    "JobIDRaw": attempt["job_id"],
                    "JobName": attempt["job_name"],
                    "Comment": attempt["comment"],
                    "User": "u",
                    "Account": "a",
                    "QOS": "q",
                    "State": "COMPLETED",
                    "AllocTRES": "gres/gpu=1,gres/gpu:pro6000=1",
                    "ElapsedRaw": "10",
                    "Start": "2026-10-09T00:00:00",
                    "End": "2026-10-09T00:00:10",
                    "ExitCode": "0:0",
                }
            ]
            self.observations[relative] = {"queue": [], "accounting": accounting}
            summary = summarize_accounting(accounting, attempt, self.permission, 1)
            evidence = {"path": relative, "sha256": fixture_hash(relative)}
            attempt.update(accounting=summary, observation=evidence, status="COMPLETED")
            self.cost["allocations"].append(
                {"task_id": task_id, **attempt, **summary, "scheduler_observation": evidence}
            )
        self.cost.update(
            gpu_task_count=len(self.cost["allocations"]),
            actual_gpu_seconds=750,
            actual_gpu_hours=750 / 3600,
        )
        self.metrics_cache = {}
        self.missing_path = None
        self.endpoint_mismatch = None
        self.group_short = False
        self.raw_mutation = None
        self.raw_rehash = True
        self.panel_incomplete = None
        self.read_counts = {}
        self.panel_calls = []

    def policy_hash(self, run_id, step):
        return fixture_hash(f"parameters:{run_id}:{step}")

    def raw(self, run, step, slot, index):
        schedule = self.slots[run["schedule_id"]][step, slot]
        qid = schedule["question_id"]
        question = self.questions[qid]
        truth = list(map(int, question["true_values_decimal"]))
        answer = {
            "sum": sum(truth),
            "difference": truth[0] - truth[1],
            "range": max(truth) - min(truth),
        }[question["operation"]]
        policy_hash = self.policy_hash(run["run_id"], step - 1)
        raw = {
            "run_id": run["run_id"],
            "logical_step": step,
            "slot": slot,
            "sample_index": index,
            "question_id": qid,
            "seed": seed("rollout", run["phase"], run["repeat"], qid, index),
            "policy_hash": policy_hash,
            "sampling_hash": "native-training-config-sha",
            "input_hash": digest(
                {
                    key: question[key]
                    for key in ("question_id", "image_path", "image_sha256", "prompt")
                }
            ),
            "request_id": digest(
                [PLAN_ID, run["phase"], run["run_id"], step, slot, index, policy_hash]
            ),
            "generation_status": "COMPLETE",
            "technical_validation_errors": [],
            "raw_text": json.dumps({"readings": truth, "answer": answer}),
            "tokens": [10, 11],
            "raw_tokens": [10, 11],
            "old_logprobs": [-0.5, -0.7],
            "sampler_logprobs": [-0.5, -0.7],
            "completion_token_count": 2,
            "prompt_token_count": 20,
            "truncated": False,
            "image_routing": {"generation_vision_forward_calls": 1},
        }
        raw["record_hash"] = digest(raw)
        return raw

    def metrics(self, run, step):
        key = run["run_id"], step
        if key not in self.metrics_cache:
            groups, token_path = [], []
            for slot in [s for s in self.streams[run["schedule_id"]] if s["logical_step"] == step]:
                group = {
                    **slot,
                    "completion_rewards": [
                        {"rA": "1", "q_read": "1", "rAP": "1", "format_valid": True}
                        for _ in range(8)
                    ],
                    "rewards": ["1"] * 8,
                    "advantages": [0] * 8,
                }
                groups.append(group)
                for index in range(8):
                    raw = self.raw(run, step, slot["slot"], index)
                    token_path.append(
                        {
                            key: raw[key]
                            for key in (
                                "question_id",
                                "tokens",
                                "raw_text",
                                "image_routing",
                                "old_logprobs",
                                "seed",
                            )
                        }
                    )
            self.metrics_cache[key] = {
                "logical_step": step,
                "groups": groups,
                "effective_sequences": 192,
                "parameter_hash_before": self.policy_hash(run["run_id"], step - 1),
                "parameter_hash_after": self.policy_hash(run["run_id"], step),
                "token_path_hash": digest(token_path),
                "stress_coefficient_override": False,
            }
        return self.metrics_cache[key]

    def checkpoint(self, run, step):
        relative = f"training/{run['run_id']}/checkpoints/step-{step:02d}-fixture.pt"
        metrics = self.metrics(run, step)
        return {
            "step": step,
            "path": f"step-{step:02d}-fixture.pt",
            "sha256": fixture_hash(relative),
            "state_hash": fixture_hash([run["run_id"], step]),
            "field_hashes": {
                "diagnostics_hash": state_hash(digest(metrics)),
                "parameters": self.policy_hash(run["run_id"], step),
                "sampling_hash": state_hash("native-training-config-sha"),
                "input_stream_hash": state_hash(digest(self.streams[run["schedule_id"]])),
                "token_path_hash": state_hash(metrics["token_path_hash"]),
            },
        }

    def state_identity(self, plan, root, state_id):
        run = next(
            run for run in self.matrix if state_id in {run["run_id"], run.get("output_state")}
        )
        adapter = "states/" + run["run_id"]
        result = {
            "state_id": state_id,
            "producer_run_id": run["run_id"],
            "logical_step": 32,
            "checkpoint_sha256": self.checkpoint(run, 32)["sha256"],
            "trainable_state_hash": self.policy_hash(run["run_id"], 32),
            "adapter_path": adapter,
            "adapter_file_hashes": {
                "adapter.safetensors": fixture_hash(adapter + "/adapter.safetensors")
            },
        }
        if self.endpoint_mismatch and run["run_id"] == "prep_A_0":
            result[self.endpoint_mismatch] = "wrong-binding"
        return result

    def read_json(self, path):
        relative = str(Path(path).relative_to(self.root))
        self.read_counts[relative] = self.read_counts.get(relative, 0) + 1
        if relative == self.missing_path:
            raise FileNotFoundError(relative)
        packets = {
            "manifests/GPU_ACCOUNTING_FINAL.json": self.cost,
            "orchestration/REGISTRATION.json": self.registration,
            "orchestration/STATE.json": self.scheduler_state,
            **self.observations,
        }
        if relative in packets:
            return packets[relative]
        pieces = relative.split("/")
        if pieces[0] != "training":
            raise AssertionError("Unexpected virtual JSON read: " + relative)
        run = self.runs[pieces[1]]
        if pieces[-1] == "COMPLETE.json":
            return {"final_step": 32, "checkpoint": self.checkpoint(run, 32)}
        if pieces[-1] == "LATEST.json":
            return self.checkpoint(run, 32)
        if pieces[2] == "steps":
            step = int(Path(pieces[-1]).stem)
            metrics = self.metrics(run, step)
            if self.group_short and run["run_id"] == "prep_A_0" and step == 1:
                metrics["groups"][0]["completion_rewards"] = ["reward"] * 7
            return metrics
        if pieces[2] == "checkpoints" and pieces[-1].startswith("commit-"):
            return self.checkpoint(run, int(Path(pieces[-1]).stem.split("-")[1]))
        if pieces[2] == "rollouts":
            step, slot, index = map(int, Path(pieces[-1]).stem.split("-"))
            raw = self.raw(run, step, slot, index)
            if (
                self.raw_mutation
                and run["run_id"] == "prep_A_0"
                and (step, slot, index) == (1, 0, 0)
            ):
                raw[self.raw_mutation] = {
                    "seed": 999,
                    "tokens": [10, 12],
                }.get(self.raw_mutation, "WRONG")
                if self.raw_rehash:
                    raw["record_hash"] = digest(
                        {k: v for k, v in raw.items() if k != "record_hash"}
                    )
            return raw
        raise AssertionError("Unexpected virtual training JSON: " + relative)

    def file_hash(self, path):
        path = Path(path)
        if path == self.ledger:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        return fixture_hash(str(path.relative_to(self.root)))

    def panel(self, plan, root, panel, state_id):
        self.panel_calls.append((panel, state_id))
        count = 6144 if panel == "PROBE" else 12288
        if self.panel_incomplete == (panel, state_id):
            count -= 1
        relative = f"raw/{panel}/{state_id}/shard_0of1/outputs_000000.jsonl"
        return {
            "complete": True,
            "outputs": range(count),
            "self_scores": range(6144) if panel == "PROBE" else [],
            "gold_scores": range(1536) if panel == "PROBE" else [],
            "receipts": [
                {"shard": 0, "shards": 1, "artifacts": {relative: fixture_hash(relative)}}
            ],
        }

    def verify(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(common, "load_plan", return_value=self.plan))
            execution = stack.enter_context(
                patch.object(common, "verify_execution", return_value={})
            )
            stack.enter_context(patch.object(common, "read_json", new=self.read_json))
            stack.enter_context(patch.object(common, "sha256_file", new=self.file_hash))
            stack.enter_context(
                patch.object(common, "checked_path", new=lambda root, rel: root / rel)
            )
            stack.enter_context(patch("mm_dev.runtime.get_state_identity", new=self.state_identity))
            stack.enter_context(patch("mm_dev.evaluation.load_panel", new=self.panel))
            result = common.verify_analysis_readiness(PLAN_PATH, self.root)
            execution.assert_called_once_with(PLAN_PATH, self.root, require_engine=True)
            return result


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.fixture = VirtualCompleteRun(self.root)

    def test_complete_production_cardinalities_and_stable_receipt(self):
        first = self.fixture.verify()
        second = self.fixture.verify()
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "COMPLETE")
        self.assertEqual(first["scientific_training_paths"], 34)
        self.assertEqual(first["counts"]["training_rollouts"], 208896)
        self.assertEqual(first["counts"]["probe_completions"], 30720)
        self.assertEqual(first["counts"]["evaluation_completions"], 430080)
        self.assertEqual(first["counts"]["probe_gold_score_records"], 7680)
        self.assertEqual(len(set(self.fixture.panel_calls)), 40)
        self.assertNotIn("time", first)
        self.assertNotIn("verified_at", first)

    def test_missing_34th_scientific_path_is_not_a_complete_33_path_run(self):
        last = run_matrix()[-1]["run_id"]
        self.fixture.missing_path = f"training/{last}/COMPLETE.json"
        with self.assertRaises(FileNotFoundError):
            self.fixture.verify()

    def test_wrong_step32_checkpoint_binding_is_rejected(self):
        self.fixture.endpoint_mismatch = "checkpoint_sha256"
        with self.assertRaisesRegex(PermissionError, "endpoint"):
            self.fixture.verify()

    def test_wrong_step32_parameter_identity_is_rejected(self):
        self.fixture.endpoint_mismatch = "trainable_state_hash"
        with self.assertRaises(PermissionError):
            self.fixture.verify()

    def test_seven_reward_records_are_not_an_eight_sample_group(self):
        self.fixture.group_short = True
        with self.assertRaisesRegex(PermissionError, "eight"):
            self.fixture.verify()

    def test_raw_seed_and_input_identity_checked_even_with_valid_self_hash(self):
        for key in ("seed", "input_hash"):
            self.fixture.raw_mutation = key
            with self.subTest(key=key), self.assertRaises(PermissionError):
                self.fixture.verify()

    def test_technical_invalid_raw_is_not_a_scientific_completed_slot(self):
        self.fixture.raw_mutation = "generation_status"
        with self.assertRaises(PermissionError):
            self.fixture.verify()

    def test_raw_policy_request_sampling_identity_checked_with_valid_self_hash(self):
        for key in ("policy_hash", "request_id", "sampling_hash"):
            self.fixture.raw_mutation = key
            with self.subTest(key=key), self.assertRaises(PermissionError):
                self.fixture.verify()

    def test_raw_token_trajectory_checked_even_with_valid_self_hash(self):
        self.fixture.raw_mutation = "tokens"
        with self.assertRaisesRegex(PermissionError, "token trajectory"):
            self.fixture.verify()

    def test_one_missing_panel_slot_rejects_complete_status(self):
        self.fixture.panel_incomplete = ("PROBE", "S0")
        with self.assertRaisesRegex(PermissionError, "panel is incomplete"):
            self.fixture.verify()

    def test_nonterminal_scheduler_source_rejected_despite_complete_cost_marker(self):
        observation = next(iter(self.fixture.observations.values()))
        observation["accounting"][0]["State"] = "RUNNING"
        with self.assertRaises((PermissionError, ValueError)):
            self.fixture.verify()

    def test_ledger_change_between_checks_changes_readiness_digest(self):
        before = self.fixture.verify()
        with self.fixture.ledger.open("a") as stream:
            stream.write('{"kind":"completion_attempts","count":1}\n')
        after = self.fixture.verify()
        self.assertNotEqual(before["evidence_set_sha256"], after["evidence_set_sha256"])
        self.assertEqual(before["counts"], after["counts"])


if __name__ == "__main__":
    unittest.main()
