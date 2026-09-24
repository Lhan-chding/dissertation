"""Durable, deduplicated task DAG and prospective test authorization.

No scheduler calls occur unless an explicit callback is supplied. Submission intents
are permanent: an interrupted or ambiguous sbatch requires operator reconciliation,
never an automatic second submission.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LEVELS = (
    "Z0_MOMENTS_GLOBAL",
    "Z1_MOMENTS_STRATIFIED",
    "Z2_REWARD_DISTRIBUTION",
    "Z3_REPAIR_STRUCTURE",
)
BASELINES = ("GDPO_R4", "SAW_R4", "DIRECT_REPAIR_R4")
KEY_FIELDS = {
    "source": ("protocol", "lineage_id", "source_recipe"),
    "prestate": ("lineage_id", "origin_id", "policy_id", "panel_id", "snapshot_t", "draw_role"),
    "branch": ("origin_id", "recipe_id", "repeat", "schedule_id"),
    "evaluation": (
        "lineage_id",
        "origin_id",
        "repeat",
        "policy_id",
        "panel_id",
        "horizon",
        "draw_role",
    ),
    "smoke": ("protocol", "smoke_id"),
    "historical_e": ("policy_id", "panel_id", "draw_role"),
}


def _bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def content_id(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _publish(path: Path, value: Any) -> None:
    """Publish a complete immutable JSON file without overwriting an existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _bytes(value)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"immutable record differs: {path}")
        return
    fd, name = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(name, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"immutable record differs: {path}") from None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.unlink(name)


class TaskRegistry:
    def __init__(self, root: str | Path, protocol: Mapping[str, Any]):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.protocol = json.loads(json.dumps(protocol))
        self.protocol_id = str(self.protocol["protocol"])
        _publish(self.root / "protocol.json", self.protocol)
        self.lineages = {}
        for role, rows in self.protocol["origins"].items():
            if role in ("development", "tuning", "test_pool"):
                for row in rows:
                    self.lineages[int(row["seed"])] = row

    @contextmanager
    def _lock(self):
        with (self.root / ".registry.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _path(self, area: str, identifier: str) -> Path:
        if not identifier or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in identifier
        ):
            raise ValueError("unsafe record identifier")
        return self.root / area / f"{identifier}.json"

    def tasks(self) -> list[dict]:
        return [_read(p) for p in sorted((self.root / "tasks").glob("*.json"))]

    def task(self, task_id: str) -> dict:
        task = _read(self._path("tasks", task_id))
        expected = content_id({"kind": task["kind"], "key": task["key"]})
        if (
            expected != task_id
            or task["task_id"] != task_id
            or task["key"] != [task["payload"][field] for field in KEY_FIELDS[task["kind"]]]
        ):
            raise ValueError("task identity mismatch")
        return task

    def _lineage(self, payload: Mapping) -> dict:
        try:
            row = self.lineages[int(payload["lineage_id"])]
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("unknown lineage") from exc
        if payload.get("source_recipe") != row["source_recipe"]:
            raise ValueError("source recipe differs from registration")
        if "role" in payload and payload["role"] != row["role"]:
            raise ValueError("lineage role differs from registration")
        if "origin_step" in payload and payload["origin_step"] not in row["anchors"]:
            raise ValueError("unregistered origin step")
        if (
            "origin_id" in payload
            and payload["origin_id"]
            != f"{int(payload['lineage_id'])}_t{payload.get('origin_step')}"
        ):
            raise ValueError("origin_id must match its canonical registered lineage and step")
        return row

    def load_freeze(self) -> dict:
        path = self.root / "freeze.json"
        if not path.exists():
            raise ValueError("test requires frozen selectors and sample size")
        freeze = _read(path)
        body = {k: v for k, v in freeze.items() if k != "freeze_id"}
        if freeze.get("freeze_id") != content_id(body):
            raise ValueError("freeze identity mismatch")
        for level, digest in freeze["selector_ids"].items():
            model = _read(self._path("frozen_selectors", digest))
            if content_id(model) != digest:
                raise ValueError(f"frozen selector content changed: {level}")
        return freeze

    def write_freeze(self, freeze: Mapping) -> dict:
        with self._lock():
            value = json.loads(json.dumps(freeze))
            models = value.pop("frozen_selectors", None)
            required = {
                "source_version",
                "model",
                "recipes",
                "feature_schemas",
                "scaling",
                "alphas",
                "best_static",
                "T_panel",
                "test_n",
                "test_draws",
                "primary_comparison",
                "horizon",
                "secondary_analyses",
                "selection_rule",
            }
            if required - value.keys():
                raise ValueError(f"missing freeze fields: {sorted(required - value.keys())}")
            if not models or set(models) != set(LEVELS):
                raise ValueError("freeze requires four serialized selector models")
            for field in ("feature_schemas", "scaling", "alphas"):
                if set(value[field]) != set(LEVELS):
                    raise ValueError(f"freeze {field} must contain exactly four levels")
            if (
                value["model"] != self.protocol["model"]
                or value["recipes"] != self.protocol["actions"]
            ):
                raise ValueError("freeze model/recipes differ from protocol")
            if (
                value["test_n"] not in self.protocol["origins"]["test_n_choices"]
                or value["test_draws"] not in self.protocol["panels"]["T"]["draws_choices"]
            ):
                raise ValueError("unregistered final N/m")
            if value["horizon"] != 32 or value["primary_comparison"] != list(LEVELS[2:][::-1]):
                raise ValueError("primary test must be frozen Z3-Z2 at H32")
            if value["best_static"] not in self.protocol["actions"]["selectable"]:
                raise ValueError("invalid BestStatic")
            if not value["source_version"] or not value["T_panel"] or not value["selection_rule"]:
                raise ValueError("freeze requires version, panel and selection rule")
            ids = {level: content_id(model) for level, model in models.items()}
            if "selector_ids" in value and value["selector_ids"] != ids:
                raise ValueError("selector IDs must bind serialized content")
            value["selector_ids"] = ids
            value.pop("freeze_id", None)
            value["freeze_id"] = content_id(value)
            existing = self.root / "freeze.json"
            if existing.exists():
                _publish(existing, value)
                return self.load_freeze()
            if any(self._is_test(t["payload"]) for t in self.tasks()):
                raise ValueError("cannot freeze after test tasks exist")
            for level, model in models.items():
                _publish(self._path("frozen_selectors", ids[level]), model)
            _publish(existing, value)
            return self.load_freeze()

    def _is_test(self, payload: Mapping) -> bool:
        return int(payload.get("lineage_id", -1)) in {
            int(r["seed"]) for r in self.protocol["origins"]["test_pool"]
        }

    def _test_origin(self, payload: Mapping, freeze: Mapping) -> None:
        row = self._lineage(payload)
        allowed = self.protocol["origins"]["test_pool"][: freeze["test_n"]]
        if row not in allowed:
            raise ValueError("lineage outside frozen first-N test registry")
        if "origin_step" in payload and payload["origin_step"] != row["anchors"][0]:
            raise ValueError("test uses exactly one registered origin")

    def branch_output(self, origin_id: str, recipe_id: str, repeat: int) -> Path:
        self._path("decisions", origin_id)
        if recipe_id not in (*self.protocol["actions"]["selectable"], *BASELINES) or repeat not in (
            1,
            2,
        ):
            raise ValueError("invalid branch identity")
        return self.root / "branches" / origin_id / recipe_id / f"repeat_{repeat}"

    def write_decision(self, decision: Mapping) -> dict:
        with self._lock():
            freeze = self.load_freeze()
            value = json.loads(json.dumps(decision))
            value["lineage_id"] = int(value["lineage_id"])
            self._test_origin(value, freeze)
            self._validate_decision(value, freeze)
            value.pop("decision_id", None)
            value["decision_id"] = content_id(value)
            path = self._path("decisions", value["origin_id"])
            if path.exists():
                _publish(path, value)
                return value
            if any(
                t["kind"] in ("branch", "evaluation")
                and t["payload"].get("origin_id") == value["origin_id"]
                for t in self.tasks()
            ):
                raise ValueError("decision must precede future task creation")
            future = self.root / "branches" / value["origin_id"]
            if future.exists() and any(p.is_file() for p in future.rglob("*")):
                raise ValueError("future training artifacts already exist before decision")
            _publish(path, value)
            return value

    def _validate_decision(self, decision: Mapping, freeze: Mapping) -> None:
        if (
            decision.get("freeze_id") != freeze["freeze_id"]
            or decision.get("selector_ids") != freeze["selector_ids"]
        ):
            raise ValueError("decision does not bind frozen models")
        if decision.get("made_before_run") is not True:
            raise ValueError("prospective decision required")
        if "origin_step" not in decision or "origin_id" not in decision:
            raise ValueError("decision requires registered origin identity")
        choices = decision.get("choices", {})
        if set(choices) != set(LEVELS) or not set(choices.values()) <= set(
            self.protocol["actions"]["selectable"]
        ):
            raise ValueError("four selectors must choose from the common R0-R7 set")

    def decision(self, origin_id: str) -> dict:
        path = self._path("decisions", origin_id)
        if not path.exists():
            raise ValueError("test future work requires a pre-existing decision")
        decision = _read(path)
        if decision.get("decision_id") != content_id(
            {k: v for k, v in decision.items() if k != "decision_id"}
        ):
            raise ValueError("decision changed after immutable publication")
        freeze = self.load_freeze()
        self._test_origin(decision, freeze)
        self._validate_decision(decision, freeze)
        return decision

    def _guard(self, kind: str, payload: dict) -> None:
        if kind in ("source", "prestate", "branch", "evaluation"):
            self._lineage(payload)
        if kind in ("prestate", "branch", "evaluation"):
            if "origin_step" not in payload:
                raise ValueError("origin_step required")
            self._path("decisions", payload["origin_id"])
        if kind == "branch":
            if payload["recipe_id"] not in (*self.protocol["actions"]["selectable"], *BASELINES):
                raise ValueError("unregistered recipe")
            repeats = 2 if self._is_test(payload) else 1
            if payload["repeat"] not in range(1, repeats + 1):
                raise ValueError("unregistered continuation repeat")
            expected = self.protocol["training"]["branch_schedule_seeds"][payload["repeat"] - 1]
            prefix = f"continuation-{payload['repeat']}-"
            suffix = str(payload["schedule_id"])[len(prefix) :]
            if (
                payload.get("branch_schedule_seed") != expected
                or not str(payload["schedule_id"]).startswith(prefix)
                or len(suffix) != 16
                or any(c not in "0123456789abcdef" for c in suffix)
            ):
                raise ValueError("branch must use common registered repeat schedule")
        if self._is_test(payload):
            freeze = self.load_freeze()
            self._test_origin(payload, freeze)
            if kind in ("branch", "evaluation"):
                decision = self.decision(payload["origin_id"])
                for field in ("lineage_id", "origin_step", "source_recipe"):
                    if decision[field] != payload[field]:
                        raise ValueError("task origin differs from decision")
                allowed = set(decision["choices"].values()) | {
                    freeze["best_static"],
                    "R0",
                    *BASELINES,
                }
                if payload.get("recipe_id") not in allowed:
                    raise ValueError("test recipe outside frozen decision/baseline union")
                if payload["repeat"] not in (1, 2):
                    raise ValueError("test requires repeats 1 and 2")
                if kind == "evaluation" and (
                    payload["panel_id"],
                    payload["horizon"],
                    payload.get("n_draws"),
                ) not in (("T", 32, freeze["test_draws"]), ("T8", 8, 8)):
                    raise ValueError("test evaluation differs from frozen panel/horizon/draws")

    def register_task(self, kind: str, payload: Mapping, dependencies=()) -> dict:
        with self._lock():
            if kind not in KEY_FIELDS:
                raise ValueError("unknown task kind")
            body = json.loads(json.dumps(payload))
            if "lineage_id" in body:
                body["lineage_id"] = int(body["lineage_id"])
            body.pop("selector_id", None)  # consumers reference a shared physical task
            body.setdefault("protocol", self.protocol_id)
            if body["protocol"] != self.protocol_id:
                raise ValueError("task protocol mismatch")
            self._guard(kind, body)
            try:
                key = [body[field] for field in KEY_FIELDS[kind]]
            except KeyError as exc:
                raise ValueError(f"missing task key field: {exc}") from exc
            task_id = content_id({"kind": kind, "key": key})
            deps = json.loads(json.dumps(list(dependencies)))
            for dep in deps:
                dep_id = dep if isinstance(dep, str) else dep["task_id"]
                self.task(dep_id)
                if isinstance(dep, dict):
                    artifact = Path(dep["completed_artifact"]).resolve()
                    expected = (
                        self.root
                        / "sources"
                        / str(body.get("lineage_id"))
                        / f"H{int(body.get('origin_step', -1)):02d}.json"
                    )
                    if self.task(dep_id)["kind"] != "source" or artifact != expected:
                        raise ValueError(
                            "source dependency must use its registered origin checkpoint receipt"
                        )
            kinds = {}
            for dep in deps:
                dep_id = dep if isinstance(dep, str) else dep["task_id"]
                prerequisite = self.task(dep_id)
                dp = prerequisite["payload"]
                if kind in ("prestate", "branch", "evaluation"):
                    if dp.get("lineage_id") != body["lineage_id"]:
                        raise ValueError("dependency must belong to the same lineage")
                    if (
                        prerequisite["kind"] != "source"
                        and dp.get("origin_id") != body["origin_id"]
                    ):
                        raise ValueError("dependency must belong to the same origin")
                kinds[prerequisite["kind"]] = prerequisite
            required_kinds = {
                "prestate": {"source"},
                "branch": {"source", "prestate"},
                "evaluation": {"branch"},
            }.get(kind, set())
            if not required_kinds <= kinds.keys():
                raise ValueError(f"{kind} requires explicit dependencies: {sorted(required_kinds)}")
            if kind == "evaluation":
                bp = kinds["branch"]["payload"]
                if any(bp[field] != body[field] for field in ("recipe_id", "repeat")):
                    raise ValueError("evaluation dependency must match recipe and repeat")
            if kind == "branch":
                _publish(
                    self._path("schedules", f"repeat-{body['repeat']}"),
                    {
                        "schedule_id": body["schedule_id"],
                        "branch_schedule_seed": body["branch_schedule_seed"],
                    },
                )
            record = {
                "task_id": task_id,
                "kind": kind,
                "key": key,
                "payload": body,
                "dependencies": deps,
                "gpus": 1,
            }
            if self._is_test(body):
                record["freeze_id"] = self.load_freeze()["freeze_id"]
                if kind in ("branch", "evaluation"):
                    record["decision_id"] = content_id(self.decision(body["origin_id"]))
            _publish(self._path("tasks", task_id), record)
            return record

    def register_test_branches(
        self, origin_id: str, dependencies=(), branch_schedules: Mapping | None = None
    ) -> list[dict]:
        decision = self.decision(origin_id)
        freeze = self.load_freeze()
        if branch_schedules is None or set(branch_schedules) != {"1", "2"}:
            raise ValueError("prepared schedules for exactly two repeats are required")
        recipes = sorted(
            set(decision["choices"].values()) | {freeze["best_static"], "R0", *BASELINES}
        )
        base = {k: decision[k] for k in ("origin_id", "lineage_id", "origin_step", "source_recipe")}
        result = []
        for recipe in recipes:
            for repeat in (1, 2):
                seed = self.protocol["training"]["branch_schedule_seeds"][repeat - 1]
                schedule = branch_schedules[str(repeat)]
                if schedule.get("sampler_seed") != seed:
                    raise ValueError("prepared branch schedule seed differs from protocol")
                payload = {
                    **base,
                    "role": "locked_test",
                    "recipe_id": recipe,
                    "repeat": repeat,
                    "schedule_id": schedule.get("schedule_id", seed),
                    "branch_schedule_seed": seed,
                }
                result.append(self.register_task("branch", payload, dependencies))
        return result

    def _dependencies_complete(self, task: Mapping) -> bool:
        for dep in task["dependencies"]:
            if isinstance(dep, str):
                if not self._path("completed", dep).exists():
                    return False
            else:
                path = Path(dep["completed_artifact"])
                if not path.is_file():
                    return False
                try:
                    receipt = _read(path)
                    checkpoint = receipt["checkpoint"]
                    source_root = self.root / "sources" / str(task["payload"]["lineage_id"])
                    cp = Path(checkpoint["path"]).resolve()
                    if (
                        receipt["step"] != task["payload"]["origin_step"]
                        or not {"path", "sha256", "identity", "state_hash"} <= checkpoint.keys()
                        or not cp.is_relative_to(source_root)
                        or not cp.is_file()
                    ):
                        return False
                except (KeyError, TypeError, ValueError, OSError):
                    return False
        return True

    def assert_can_execute(self, task_id: str) -> dict:
        task = self.task(task_id)
        self._guard(task["kind"], task["payload"])
        if self._is_test(task["payload"]):
            if task.get("freeze_id") != self.load_freeze()["freeze_id"]:
                raise ValueError("task freeze changed")
            if task["kind"] in ("branch", "evaluation") and task.get("decision_id") != content_id(
                self.decision(task["payload"]["origin_id"])
            ):
                raise ValueError("task decision changed")
        if not self._dependencies_complete(task):
            raise ValueError("task dependencies are not complete")
        if self._path("completed", task_id).exists():
            raise ValueError("task already complete")
        return task

    def mark_complete(self, task_id: str, receipt: Mapping) -> dict:
        with self._lock():
            self.task(task_id)
            if receipt.get("status") != "COMPLETE":
                raise ValueError("completion receipt must explicitly report COMPLETE")
            value = {"task_id": task_id, "receipt": dict(receipt)}
            if not self._path("completed", task_id).exists():
                self.assert_can_execute(task_id)
            _publish(self._path("completed", task_id), value)
            return value

    def _latest_submission_path(self, task_id: str) -> Path:
        recovery = self.root / "recoveries" / task_id / "attempt_1"
        if (recovery / "intent.json").exists():
            return recovery / "submission.json"
        return self._path("submissions", task_id)

    def _occupied_slots(self, active_ids: set[str], *, terminal_task_id: str | None = None) -> int:
        unresolved = 0
        for path in (self.root / "intents").glob("*.json"):
            tid = _read(path)["task_id"]
            if tid == terminal_task_id or self._path("completed", tid).exists():
                continue
            receipt_path = self._latest_submission_path(tid)
            if not receipt_path.exists() or str(_read(receipt_path)["job_id"]) not in active_ids:
                unresolved += 1
        return len(active_ids) + unresolved

    def resubmit_terminal(
        self,
        task_id: str,
        terminal_receipt: Mapping,
        submit: Callable[[dict, Path], str],
        *,
        reason: str,
        query_active: Callable[[], list[Mapping]],
        max_gpu_jobs: int = 5,
        available_gpus: int = 5,
    ) -> dict:
        """One explicit recovery after a verified Slurm terminal failure.

        The caller must obtain a fresh sacct observation; this method makes no
        remote calls and never infers failure from absence in squeue. Receipt
        fields: source='sacct', job_id, state, observed_at (timezone ISO timestamp),
        code_version, failure_class. NUMERICAL faults additionally require a
        nonempty reviewed_correction. The reason and evidence become immutable
        before the single submission callback. One recovery per task is allowed;
        an uncertain callback outcome, or a second failure, needs separate review.
        """
        if not 1 <= max_gpu_jobs <= 5 or available_gpus < 1:
            raise ValueError("GPU concurrency must be 1..5 with hardware available")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("explicit recovery reason required")
        evidence = json.loads(json.dumps(terminal_receipt))
        fields = {"source", "job_id", "state", "observed_at", "code_version", "failure_class"}
        if fields - evidence.keys() or evidence.get("source") != "sacct":
            raise ValueError("observed sacct terminal receipt and code version required")
        terminal_failures = {
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
            "BOOT_FAIL",
            "PREEMPTED",
            "DEADLINE",
            "REVOKED",
        }
        if evidence["state"] not in terminal_failures:
            raise ValueError(
                "recovery requires an explicit terminal failure, never unknown/live/completed"
            )
        if evidence["failure_class"] not in {
            "INFRASTRUCTURE",
            "RESOURCE",
            "IMPLEMENTATION",
            "NUMERICAL",
        }:
            raise ValueError("reviewed failure class required; unknown failure cannot be retried")
        if not isinstance(evidence["code_version"], str) or not evidence["code_version"].strip():
            raise ValueError("reviewed code version required")
        correction = evidence.get("reviewed_correction")
        if evidence["failure_class"] == "NUMERICAL" and (
            not isinstance(correction, str) or not correction.strip()
        ):
            raise ValueError("numerical failure requires an explicit reviewed correction")
        try:
            observed = datetime.fromisoformat(evidence["observed_at"])
        except (TypeError, ValueError) as exc:
            raise ValueError("terminal observation requires a timezone ISO timestamp") from exc
        if observed.tzinfo is None or observed > datetime.now(timezone.utc):
            raise ValueError("terminal observation timestamp must be timezone-aware and not future")
        with self._lock():
            task = self.assert_can_execute(task_id)
            recovery = self.root / "recoveries" / task_id / "attempt_1"
            if (recovery / "intent.json").exists():
                raise ValueError(
                    "one recovery attempt already recorded; never retry an uncertain attempt"
                )
            previous_path = self._latest_submission_path(task_id)
            if not previous_path.exists():
                raise ValueError(
                    "original submission outcome unknown; no recorded job ID to reconcile"
                )
            previous = _read(previous_path)
            submitted_at = datetime.fromisoformat(
                _read(self._path("intents", task_id))["created_at"]
            )
            if observed < submitted_at:
                raise ValueError("terminal observation predates the recorded submission intent")
            if str(evidence["job_id"]) != str(previous["job_id"]):
                raise ValueError("terminal receipt does not match the recorded submission job ID")
            if (
                self._is_test(task["payload"])
                and evidence["code_version"] != self.load_freeze()["source_version"]
            ):
                raise ValueError("test recovery cannot change the frozen source version")
            active = list(query_active())
            if any(int(row.get("gpus", 1)) != 1 for row in active):
                raise ValueError("project jobs must each request exactly one GPU")
            active_ids = {str(row["job_id"]) for row in active}
            if str(previous["job_id"]) in active_ids:
                raise ValueError(
                    "recorded job is still active; terminal evidence conflicts with live state"
                )
            occupied = self._occupied_slots(active_ids, terminal_task_id=task_id)
            if occupied >= min(max_gpu_jobs, available_gpus):
                raise ValueError("no free project GPU slot for recovery")
            intent = {
                "task_id": task_id,
                "attempt": 1,
                "previous_job_id": previous["job_id"],
                "created_at": _now(),
                "status": "RECOVERY_INTENT_UNRESOLVED",
                "gpus": 1,
                "reason": reason,
                "terminal_receipt": evidence,
            }
            _publish(recovery / "intent.json", intent)
            try:
                job_id = str(submit(task, self._path("tasks", task_id))).strip()
                if not job_id or not job_id.split(";")[0].isdigit():
                    raise ValueError("sbatch returned an unrecognized job ID")
            except Exception as exc:
                _publish(
                    recovery / "submission_error.json",
                    {
                        "task_id": task_id,
                        "attempt": 1,
                        "status": "SUBMISSION_UNKNOWN_DO_NOT_RESUBMIT",
                        "error_type": type(exc).__name__,
                    },
                )
                raise RuntimeError("recovery submission outcome unknown; do not resubmit") from exc
            result = {
                "task_id": task_id,
                "attempt": 1,
                "job_id": job_id.split(";")[0],
                "previous_job_id": previous["job_id"],
                "status": "RECOVERY_SUBMITTED",
            }
            _publish(recovery / "submission.json", result)
            return result

    def submit_ready(
        self,
        query_active: Callable[[], list[Mapping]],
        submit: Callable[[dict, Path], str],
        *,
        max_gpu_jobs: int = 5,
        available_gpus: int = 5,
    ) -> list[dict]:
        """Callbacks query *this project's* active jobs and perform one sbatch.

        Active rows: job_id, gpus (must equal 1). Unknown intents consume slots
        even when a queue query sees no job; queue lag cannot trigger resubmission.
        """
        if not 1 <= max_gpu_jobs <= 5 or available_gpus < 1:
            raise ValueError("GPU concurrency must be 1..5 with hardware available")
        with self._lock():
            active = list(query_active())
            if any(int(row.get("gpus", 1)) != 1 for row in active):
                raise ValueError("project jobs must each request exactly one GPU")
            active_ids = {str(row["job_id"]) for row in active}
            slots = min(max_gpu_jobs, available_gpus) - self._occupied_slots(active_ids)
            submissions = []
            for task in self.tasks():
                if slots <= 0:
                    break
                tid = task["task_id"]
                if (
                    self._path("intents", tid).exists()
                    or self._path("completed", tid).exists()
                    or not self._dependencies_complete(task)
                ):
                    continue
                self.assert_can_execute(tid)
                intent = {
                    "task_id": tid,
                    "created_at": _now(),
                    "status": "SUBMISSION_INTENT_UNRESOLVED",
                    "gpus": 1,
                }
                _publish(self._path("intents", tid), intent)
                try:
                    job_id = str(submit(task, self._path("tasks", tid))).strip()
                    if not job_id or not job_id.split(";")[0].isdigit():
                        raise ValueError("sbatch returned an unrecognized job ID")
                except Exception as exc:
                    _publish(
                        self._path("submission_errors", tid),
                        {
                            "task_id": tid,
                            "status": "SUBMISSION_UNKNOWN_DO_NOT_RESUBMIT",
                            "error_type": type(exc).__name__,
                        },
                    )
                    raise RuntimeError(
                        "submission outcome unknown; reconcile intent, do not resubmit"
                    ) from exc
                receipt = {"task_id": tid, "job_id": job_id.split(";")[0], "status": "SUBMITTED"}
                _publish(self._path("submissions", tid), receipt)
                submissions.append(receipt)
                slots -= 1
            return submissions
