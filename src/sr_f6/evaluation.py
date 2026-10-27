"""Fixed PILOT DEV_EVAL execution; raw trajectories are never scored in place.

The production entry verifies admission, the frozen 12-endpoint registration,
and a live evaluation access session. A two-draw STARTED group with no verified
COMPLETE marker is UNKNOWN and cannot be resampled. Complete groups are replayed
from their original persisted bytes. Synthetic execution has a separate root
and always reports SYNTHETIC_ONLY; it cannot create production evidence.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path

from srf6_ref.core import FAMILIES, seed_for

from .census_evidence import _strict_json
from .common import digest, write_once
from .measurement import score_response
from .registration import DIRECTORY as REGISTRATION_DIRECTORY
from .transactions import RunTransactions, _plain_path, _read_bytes, _require, _sha

FORMAT = "srf6-pilot-evaluation-v1"
DIRECTORY = "PHASE_RUNS/PILOT_V1/EVALUATION"
SYNTHETIC_DIRECTORY = "SYNTHETIC_EVALUATION"
LABELS = ("A", "J", "V", "S", "B")
BOOTSTRAPS = 2000
SUMMARY_RECIPE = dict(
    status="DESCRIPTIVE_ONLY_NOT_GATE",
    master_seed=2026102701,
    bootstrap_replicates=BOOTSTRAPS,
    confidence_level=0.95,
    quantiles=[0.025, 0.975],
    quantile_interpolation="LINEAR_INDEX_N_MINUS_ONE_P",
    resampling="SIX_FAMILIES_EACH_FOUR_ROOTS_WITH_REPLACEMENT",
    paired_cluster="ROOT_FOUR_VIEWS_TWO_DRAWS",
    absolute_seed_parts=[
        "master_seed",
        "pilot_dev_eval_summary",
        "endpoint_id",
        "label",
        "statistic",
    ],
    absolute_labels=list(LABELS),
    absolute_statistics=["known_rate", "pass_rate_given_known"],
    difference_seed_parts=[
        "master_seed",
        "pilot_dev_eval_summary",
        "endpoint_id",
        "parent_difference",
        "metric",
    ],
    difference_metrics=[*LABELS, "S_known", "B_known"],
    semantic_difference_denominator="BOTH_ENDPOINT_AND_PARENT_KNOWN",
    zero_denominator_bootstrap="OMIT_AND_REPORT_DEFINED_REPLICATES_ALL_UNDEFINED_IS_NULL",
    training_repeat_variance_estimated=False,
    significance_gate=None,
)


class IncompleteEvaluation(PermissionError):
    """An attempted generation has unknown/partial outcome; preserve, never redraw."""


def _read(path):
    return _strict_json(_read_bytes(path))


def _reference(root, path):
    path = _plain_path(path)
    return dict(
        path=path.relative_to(root).as_posix(), sha256=hashlib.sha256(_read_bytes(path)).hexdigest()
    )


def _bound(root, ref, *, lines=False):
    _require(
        isinstance(ref, dict) and set(ref) == {"path", "sha256"} and _sha(ref["sha256"]),
        "Exact evidence reference required",
    )
    relative = Path(ref["path"])
    _require(not relative.is_absolute() and ".." not in relative.parts, "Evidence path escaped run")
    payload = _read_bytes(root / relative)
    _require(
        hashlib.sha256(payload).hexdigest() == ref["sha256"], "Bound evaluation evidence changed"
    )
    return (
        [_strict_json(line) for line in payload.splitlines() if line.strip()]
        if lines
        else _strict_json(payload)
    )


def _write(path, value):
    path = _plain_path(path)
    write_once(path, value)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _plan(spec):
    plan = spec["evaluation"]
    _require(
        plan.get("descriptive_summary") == SUMMARY_RECIPE,
        "Prospectively frozen evaluation summary recipe required",
    )
    _require(
        all(
            plan.get(k) == v
            for k, v in dict(
                split="DEV_EVAL",
                questions=96,
                draws_per_question=2,
                checkpoint_count=12,
                responses=2304,
                formal_test_access=False,
                recovery_endpoint_is_additional_checkpoint=False,
            ).items()
        ),
        "Fixed DEV_EVAL budget/panel required",
    )
    expected_ids = {f"{p}/parent" for p in ("b0", "b1", "b2")} | {
        f"{p}/{a}/step12" for p in ("b0", "b1", "b2") for a in ("U10", "U11", "C_SHUF")
    }
    endpoints = plan["endpoints"]
    _require(
        len(endpoints) == 12 and {e["endpoint_id"] for e in endpoints} == expected_ids,
        "Exactly three parents and nine PRIMARY endpoints required",
    )
    qids = {f"DEV_EVAL_{f}_{i:04d}_P0_v{v}" for f in FAMILIES for i in range(4) for v in range(4)}
    for endpoint in endpoints:
        parent = endpoint["parent"]
        is_parent = endpoint["endpoint_id"] == f"{parent}/parent"
        _require(
            endpoint["responses"] == 192
            and endpoint["checkpoint_step"] == (0 if is_parent else 12)
            and endpoint["kind"] == ("FROZEN_PARENT" if is_parent else "PILOT_ENDPOINT"),
            "Endpoint kind/step differs",
        )
    _require(
        set(plan["seeds_by_parent"]) == {"b0", "b1", "b2"},
        "Three fixed evaluation seed streams required",
    )
    for rows in plan["seeds_by_parent"].values():
        _require(
            len(rows) == 96 and [r["qid"] for r in rows] == sorted(qids),
            "Fixed sorted evaluation qids required",
        )
        _require(
            all(len(r["seeds"]) == 2 and all(type(s) is int for s in r["seeds"]) for r in rows),
            "Exactly two registered integer draw seeds required",
        )
    return plan


def _inputs(root, plan):
    _require(
        plan["inputs"]["public"]["path"] == "DATA/DEV_EVAL/MODEL_INPUTS.jsonl"
        and plan["inputs"]["private"]["path"] == "DATA/DEV_EVAL/TASKS_PRIVATE.jsonl",
        "Only DEV_EVAL input files may be accessed",
    )
    public = _bound(root, plan["inputs"]["public"], lines=True)
    private = _bound(root, plan["inputs"]["private"], lines=True)
    items, tasks = {r["qid"]: r for r in public}, {r["qid"]: r for r in private}
    expected = {r["qid"] for r in plan["seeds_by_parent"]["b0"]}
    _require(
        len(public) == len(private) == len(items) == len(tasks) == 96
        and set(items) == set(tasks) == expected,
        "Complete frozen DEV_EVAL matrix required",
    )
    for qid, item in items.items():
        task = tasks[qid]
        _require(
            item["split"] == task["split"] == "DEV_EVAL"
            and item["protocol"] == "structured_v4"
            and set(item["model_input"]) == {"prompt", "image_file", "image_sha256"},
            "Public model input whitelist/panel differs",
        )
        _require(
            all(item[k] == task[k] for k in ("qid", "root_id", "family", "factor", "variant")),
            "Evaluation task lineage differs",
        )
    return items, tasks


def _endpoint(plan, identifier):
    matched = [e for e in plan["endpoints"] if e["endpoint_id"] == identifier]
    _require(len(matched) == 1, "Unregistered or recovery evaluation endpoint")
    return copy.deepcopy(matched[0])


class EvaluationLedger:
    """Exclusive endpoint ledger; access.py binds this live object, not a flag."""

    def __init__(self, root, identity, plan, public, tasks):
        self.root = _plain_path(root)
        self.identity = copy.deepcopy(identity)
        self.plan = copy.deepcopy(plan)
        self.endpoint = _endpoint(plan, identity["endpoint_id"])
        self.public, self.tasks = copy.deepcopy(public), copy.deepcopy(tasks)
        self.schedule = copy.deepcopy(plan["seeds_by_parent"][self.endpoint["parent"]])
        base = (
            SYNTHETIC_DIRECTORY if identity["execution_origin"] == "SYNTHETIC_TEST" else DIRECTORY
        )
        _require(
            identity["execution_origin"] in ("SYNTHETIC_TEST", "SERVER_ACTUAL"),
            "Explicit evaluation execution origin required",
        )
        self.directory = _plain_path(self.root / base / identity["endpoint_id"])
        self._fd = self._pid = None
        self.pending = None
        self._generation_consumed = False
        self._active_reservation = None
        self._verified_groups = {}
        self._runtime_refs = {}

    def __enter__(self):
        _require(self._fd is None, "Evaluation lock already held")
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(
            _plain_path(self.directory / ".evaluation.lock"),
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
            0o600,
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._fd, self._pid = fd, os.getpid()
            _write(self.directory / "IDENTITY.json", self.identity)
            return self
        except BaseException:
            self._fd = self._pid = None
            os.close(fd)
            raise

    def __exit__(self, *_):
        self.pending = None
        self._active_reservation = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = self._pid = None

    def _locked(self):
        _require(self._fd is not None and self._pid == os.getpid(), "Live evaluation lock required")
        _require(
            os.fstat(self._fd).st_ino
            == _plain_path(self.directory / ".evaluation.lock").stat().st_ino
            and _read(self.directory / "IDENTITY.json") == self.identity,
            "Evaluation lock or identity changed",
        )

    def _group(self, qid):
        _require(qid in self.public, "Unregistered evaluation question")
        return _plain_path(self.directory / "groups" / qid)

    def _start(self, item):
        return dict(
            format=FORMAT,
            status="STARTED_OUTCOME_UNKNOWN_UNTIL_COMPLETE",
            identity=self.identity,
            qid=item["qid"],
            seeds=item["seeds"],
            model_input=self.public[item["qid"]]["model_input"],
            responses=2,
        )

    def cached(self, item):
        self._locked()
        folder = self._group(item["qid"])
        if not folder.exists():
            return None
        names = {p.name for p in folder.iterdir()}
        if names != {
            "STARTED.json",
            "GENERATING.json",
            "draw-0.json",
            "draw-1.json",
            "COMPLETE.json",
        }:
            raise IncompleteEvaluation(
                "UNKNOWN/PARTIAL evaluation group preserved; no resampling: " + item["qid"]
            )
        _require(
            _read(folder / "STARTED.json") == self._start(item),
            "Evaluation STARTED identity differs",
        )
        _require(
            _read(folder / "GENERATING.json")
            == dict(self._start(item), status="GENERATION_RESERVED_NO_RETRY"),
            "Evaluation durable generation reservation differs",
        )
        raw = [_read(folder / f"draw-{draw}.json") for draw in range(2)]
        expected = self._complete(item, raw)
        _require(
            _read(folder / "COMPLETE.json") == expected,
            "Evaluation completion differs from raw/score replay",
        )
        self._verified_groups[item["qid"]] = expected
        return expected

    def begin(self, item):
        self._locked()
        _require(
            self.pending is None and self.cached(item) is None,
            "Evaluation group is already complete or active",
        )
        # No caller can skip an earlier incomplete or absent group.
        for earlier in self.schedule:
            if earlier == item:
                break
            _require(
                earlier["qid"] in self._verified_groups,
                "Evaluation groups must follow registered order",
            )
        else:
            raise PermissionError("Unregistered evaluation seed/qid pair")
        _write(self._group(item["qid"]) / "STARTED.json", self._start(item))
        self.pending = copy.deepcopy(item)
        self._generation_consumed = False
        self._active_reservation = digest(self._start(item))

    def authorize_generation(self, model_input, seeds):
        """Called once by the live runtime access session, after durable STARTED."""
        self._locked()
        _require(
            self.pending is not None
            and not self._generation_consumed
            and self._active_reservation is not None,
            "No unconsumed evaluation reservation",
        )
        item = self.pending
        _require(
            item in self.schedule and self._active_reservation == digest(self._start(item)),
            "Pending generation was not opened by this live ledger",
        )
        _require(
            list(seeds) == item["seeds"] and model_input == self.public[item["qid"]]["model_input"],
            "Generation differs from registered pending DEV_EVAL group",
        )
        _require(
            _read(self._group(item["qid"]) / "STARTED.json") == self._start(item),
            "Evaluation reservation changed",
        )
        _require(
            {p.name for p in self._group(item["qid"]).iterdir()} == {"STARTED.json"},
            "UNKNOWN/partial/already consumed generation cannot be reserved again",
        )
        _write(
            self._group(item["qid"]) / "GENERATING.json",
            dict(self._start(item), status="GENERATION_RESERVED_NO_RETRY"),
        )
        self._generation_consumed = True

    def capture(self, row):
        self._locked()
        _require(
            self.pending is not None and self._generation_consumed,
            "Generation lacks a live reserved evaluation group",
        )
        draw = row.get("group_row_index")
        _require(type(draw) is int and draw in (0, 1), "Unexpected evaluation draw callback")
        path = self._group(self.pending["qid"]) / f"draw-{draw}.json"
        _require(not path.exists(), "Duplicate evaluation completion callback")
        # Persist the original native row before technical or scoring checks.
        # Even nonfinite native failure output is retained; strict replay will
        # reject it, and STARTED can never be mistaken for a completed group.
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = (json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n").encode()
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    def _complete(self, item, raw):
        _require(len(raw) == 2, "Exactly two original evaluation trajectories required")
        _require(
            raw[0].get("task_id") == raw[1].get("task_id"),
            "Two draws must come from the same native generation process",
        )
        scores = []
        executions = {}
        task = self.tasks[item["qid"]]
        for draw, row in enumerate(raw):
            _validate_raw(row, self.identity, self.public[item["qid"]], item["seeds"], draw)
            if self.identity["execution_origin"] == "SERVER_ACTUAL":
                for ref in self._execution_evidence(row):
                    executions[ref["path"]] = ref
            score = score_response(
                row["full_text"],
                task["world"],
                task["query"],
                finished=row["finish_reason"] == "balanced",
                finish_reason=row["finish_reason"],
            )
            scores.append(
                dict(
                    qid=item["qid"],
                    root_id=task["root_id"],
                    family=task["family"],
                    variant=task["variant"],
                    draw=draw,
                    seed=item["seeds"][draw],
                    score=score,
                )
            )
        raw_refs = [
            _reference(self.root, self._group(item["qid"]) / f"draw-{d}.json") for d in range(2)
        ]
        execution_refs = [executions[k] for k in sorted(executions)]
        return dict(
            format=FORMAT,
            status="COMPLETE",
            identity=self.identity,
            qid=item["qid"],
            raw=raw_refs,
            executions=execution_refs,
            inputs=[*raw_refs, *execution_refs],
            expected_accounting=dict(
                task_id=raw[0].get("task_id"),
                events=[
                    dict(
                        kind="completion_attempts",
                        count=2,
                        metadata=dict(group_seed=item["seeds"][0], protocol="structured"),
                    ),
                    *[
                        dict(
                            kind="generated_tokens",
                            count=len(row["tokens"]),
                            metadata=dict(group_seed=item["seeds"][0], row_index=draw),
                        )
                        for draw, row in enumerate(raw)
                    ],
                ],
            ),
            scores=scores,
        )

    def _execution_evidence(self, row):
        task_id = row.get("task_id")
        _require(
            isinstance(task_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", task_id),
            "Actual runtime task identity required",
        )
        if task_id in self._runtime_refs:
            return self._runtime_refs[task_id]
        path = self.directory / "runtime_identity" / (task_id + ".json")
        report = _read(path)
        for key, expected in dict(
            task_id=task_id,
            audit_root=str(self.directory),
            parent_id=self.identity["parent"],
            execution_mode="EVALUATION",
            phase="PILOT_V1",
            phase_identity_sha256=self.identity["phase_identity_sha256"],
            run_id=self.identity["evaluation_run_id"],
            run_root=str(self.root),
            source_sha256=self.identity["source_sha256"],
            data_hash=self.identity["data_sha256"],
            reference_hash=self.identity["reference_sha256"],
            fresh_optimizer=False,
            trainable_parameters=[],
        ).items():
            _require(
                report.get(key) == expected, "Actual evaluation loader evidence differs: " + key
            )
        refs = [_reference(self.root, path)]
        if self.identity["checkpoint_commit_sha256"] is not None:
            path = self.directory / "runtime_identity" / (task_id + "-endpoint.json")
            loaded = _read(path)
            _require(
                loaded.get("status") == "ENDPOINT_LOADED_NOT_AUTHORIZED"
                and loaded.get("optimizer_created") is False
                and loaded.get("training_rng_restored") is False,
                "Read-only endpoint loader evidence required",
            )
            actual = loaded["runtime_identity"]
            _require(
                actual.get("checkpoint_commit_sha256") == self.identity["checkpoint_commit_sha256"]
                and actual.get("policy_hash") == self.identity["policy_tensor_sha256"],
                "Actual loaded endpoint content differs",
            )
            refs.append(_reference(self.root, path))
        else:
            _require(
                report.get("policy_hash") == self.identity["policy_tensor_sha256"]
                and report.get("step") == 0,
                "Frozen parent tensor evidence differs",
            )
        self._runtime_refs[task_id] = refs
        return refs

    def finish(self, returned):
        self._locked()
        _require(
            self.pending is not None and self._generation_consumed,
            "No active evaluation generation",
        )
        item = self.pending
        raw = [_read(self._group(item["qid"]) / f"draw-{d}.json") for d in range(2)]
        _require(raw == returned, "Native return differs from independently persisted callbacks")
        complete = self._complete(item, raw)
        _write(self._group(item["qid"]) / "COMPLETE.json", complete)
        self._verified_groups[item["qid"]] = complete
        self.pending = None
        self._active_reservation = None
        return complete

    def completed(self):
        self._locked()
        groups = _plain_path(self.directory / "groups")
        if groups.exists():
            _require(
                {p.name for p in groups.iterdir()} <= set(self.public),
                "Unexpected evaluation question directory",
            )
        result = [self.cached(item) for item in self.schedule]
        missing = False
        for item in result:
            if item is None:
                missing = True
            else:
                _require(not missing, "Evaluation cache has an unregistered gap")
        return None if any(value is None for value in result) else result


def _validate_raw(row, identity, item, seeds, draw):
    _require(
        isinstance(row, dict)
        and row.get("generation_status") == "COMPLETE"
        and not row.get("technical_validation_errors"),
        "Technical-invalid raw evaluation preserved, not complete",
    )
    for key, expected in dict(
        model_input=item["model_input"],
        seed=seeds[draw],
        group_seed=seeds[0],
        group_row_index=draw,
        group_size=2,
        protocol="structured_v4",
        execution_mode="EVALUATION",
        policy_tensor_hash=identity["policy_tensor_sha256"],
        policy_hash=identity["policy_tensor_sha256"],
        policy_step=identity["checkpoint_step"],
        step=identity["checkpoint_step"],
        source_sha256=identity["source_sha256"],
        data_hash=identity["data_sha256"],
        phase="PILOT_V1",
        phase_identity_sha256=identity["phase_identity_sha256"],
        run_root=identity["run_root"],
        run_id=identity["evaluation_run_id"],
    ).items():
        _require(row.get(key) == expected, "Original evaluation identity differs: " + key)
    if identity["checkpoint_commit_sha256"] is not None:
        _require(
            row.get("checkpoint_commit_sha256") == identity["checkpoint_commit_sha256"]
            and row.get("checkpoint_run_id") == identity["checkpoint_run_id"]
            and row.get("checkpoint_step") == 12,
            "Raw endpoint commit differs",
        )
    _require(
        row.get("adapter_identity", {}).get("parent") == identity["parent"]
        and row.get("adapter_identity", {}).get("step") == identity["checkpoint_step"],
        "Raw native adapter parent/step differs",
    )
    tokens, logps = row.get("tokens"), row.get("sampler_logprobs")
    _require(
        isinstance(tokens, list)
        and 1 <= len(tokens) <= 1536
        and all(type(t) is int and t >= 0 for t in tokens)
        and row.get("raw_tokens") == tokens,
        "Original evaluation tokens incomplete",
    )
    _require(
        isinstance(logps, list)
        and len(logps) == len(tokens)
        and all(type(x) in (int, float) and math.isfinite(x) and x <= 0 for x in logps),
        "Original sampling probabilities incomplete/nonfinite",
    )
    reason = row.get("finish_reason")
    _require(
        reason in ("balanced", "eos", "length")
        and row.get("completion_token_count") == len(tokens)
        and row.get("max_new_tokens") == 1536
        and row.get("truncated") is (reason == "length")
        and (reason != "length" or len(tokens) == 1536),
        "Original evaluation termination metadata differs",
    )
    _require(
        row.get("balanced_token_count") == (len(tokens) if reason == "balanced" else None)
        and row.get("boundary_token_count") == (len(tokens) if reason == "balanced" else None),
        "Original stop boundary differs",
    )
    _require(
        isinstance(row.get("full_text"), str)
        and _sha(row.get("prompt_tensor_hash"))
        and isinstance(row.get("native_prompt_inputs"), dict)
        and bool(row["native_prompt_inputs"])
        and row.get("sampler_logprob_source") == "actual_generate_raw_logits_selected_token",
        "Original native prompt/text evidence missing",
    )
    _require(
        identity["execution_origin"] != "SERVER_ACTUAL"
        or row.get("execution_origin") != "SYNTHETIC_TEST",
        "Synthetic raw cannot become actual evaluation evidence",
    )


def _endpoint_report(ledger, groups):
    records = [row for group in groups for row in group["scores"]]
    _require(len(records) == 192, "Complete endpoint denominator required")
    accounting = (
        _accounting_evidence(ledger, groups)
        if ledger.identity["execution_origin"] == "SERVER_ACTUAL"
        else dict(status="SYNTHETIC_ONLY", actual_recorded_calls=None, inputs=[])
    )
    return dict(
        format=FORMAT,
        status="SYNTHETIC_ONLY"
        if ledger.identity["execution_origin"] == "SYNTHETIC_TEST"
        else "COMPLETE_NOT_GATE_SIGNED",
        execution_origin=ledger.identity["execution_origin"],
        identity=ledger.identity,
        endpoint=ledger.endpoint,
        responses=192,
        groups=96,
        accounting=accounting,
        optimizer_updates=0,
        training_authorized=False,
        gate_status="NOT_SIGNED",
        inputs=[
            _reference(ledger.root, ledger._group(item["qid"]) / "COMPLETE.json")
            for item in ledger.schedule
        ]
        + accounting["inputs"],
        metrics=_metrics(records),
        records=records,
    )


def _event_key(event):
    kind, count, metadata = event.get("kind"), event.get("count"), event.get("metadata")
    _require(
        type(count) is int and count > 0 and isinstance(metadata, dict),
        "Invalid actual evaluation accounting event",
    )
    if kind == "completion_attempts":
        _require(
            set(metadata) == {"group_seed", "protocol"}
            and metadata["protocol"] == "structured"
            and type(metadata["group_seed"]) is int,
            "Unknown evaluation generation accounting shape",
        )
        return kind, metadata["group_seed"], metadata["protocol"], count
    if kind == "generated_tokens":
        _require(
            set(metadata) == {"group_seed", "row_index"}
            and type(metadata["group_seed"]) is int
            and type(metadata["row_index"]) is int
            and metadata["row_index"] in (0, 1),
            "Unknown evaluation token accounting shape",
        )
        return kind, metadata["group_seed"], metadata["row_index"], count
    raise PermissionError("Unknown or extra evaluation model-call accounting kind")


def _accounting_evidence(ledger, groups):
    """Bind final native event multisets once, after all 96 groups are complete.

    Runtime emits one batched count=2 attempt under the first draw seed and two
    token events with row_index 0/1. Independent per-row RNG does not split this
    event. Repeated seeds retain multiplicity; no event is keyed only by seed.
    """
    ledger._locked()
    _require(
        ledger.pending is None and len(groups) == 96,
        "Only a terminal endpoint can bind its call ledger",
    )
    expected = defaultdict(Counter)
    for group in groups:
        entry = group["expected_accounting"]
        task_id = entry["task_id"]
        _require(
            isinstance(task_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", task_id),
            "Every completed evaluation group needs an actual process identity",
        )
        expected[task_id].update(_event_key(event) for event in entry["events"])
    directory = _plain_path(ledger.directory / "accounting")
    _require(directory.is_dir(), "Missing actual evaluation call accounting")
    _require(
        {p.name for p in directory.iterdir()} == {f"{task}.jsonl" for task in expected},
        "Missing or extra/unknown evaluation process accounting; preserve and diagnose",
    )
    refs, calls = [], 0
    for task_id, required in sorted(expected.items()):
        path = _plain_path(directory / f"{task_id}.jsonl")
        before = path.stat()
        payload = _read_bytes(path)
        after = path.stat()
        _require(
            (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "Evaluation call ledger still changing at terminal binding",
        )
        events = [_strict_json(line) for line in payload.splitlines() if line.strip()]
        observed = Counter()
        for event in events:
            _require(
                event.get("task_id") == task_id
                and event.get("parent") == ledger.identity["parent"],
                "Actual evaluation accounting process/parent differs",
            )
            observed[_event_key(event)] += 1
            if event["kind"] == "completion_attempts":
                calls += event["count"]
        _require(
            observed == required,
            "Actual evaluation event multiset differs from original two-draw trajectories",
        )
        refs.append(
            dict(
                path=path.relative_to(ledger.root).as_posix(),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    _require(calls == 192, "Exact original 192 endpoint calls required")
    return dict(
        status="VERIFIED_FINAL_NATIVE_EVENTS",
        actual_recorded_calls=calls,
        inputs=refs,
        process_count=len(expected),
        unknown_or_extra_calls=False,
    )


def _execute(ledger, runtime):
    for item in ledger.schedule:
        if item["qid"] in ledger._verified_groups or ledger.cached(item) is not None:
            continue
        ledger.begin(item)
        # The production Runtime invokes the active access session before sampling.
        if ledger.identity["execution_origin"] == "SYNTHETIC_TEST":
            ledger.authorize_generation(ledger.public[item["qid"]]["model_input"], item["seeds"])
        rows = runtime.generate_group(
            ledger.public[item["qid"]]["model_input"],
            ledger.root,
            item["seeds"],
            protocol="structured_v4",
            on_completion=ledger.capture,
        )
        if ledger.identity["execution_origin"] == "SERVER_ACTUAL":
            _require(
                runtime.policy_tensor_identity() == ledger.identity["policy_tensor_sha256"]
                and all(not p.requires_grad and p.grad is None for p in runtime.model.parameters()),
                "Evaluation generation changed policy or gradients",
            )
        ledger.finish(rows)
    # The live exclusive ledger has already validated each cached/new group.
    # Reuse those exact objects instead of rereading every raw trajectory again.
    report = _endpoint_report(
        ledger, [ledger._verified_groups[item["qid"]] for item in ledger.schedule]
    )
    return report


def _metrics(records):
    result = {}
    for key in LABELS:
        values = [r["score"][key] for r in records]
        known = [v for v in values if v is not None]
        _require(
            all(type(v) is int and v in (0, 1) for v in known)
            and (key in ("S", "B") or len(known) == len(values)),
            "Invalid evaluation labels",
        )
        result[key] = dict(
            total=len(values),
            known=len(known),
            unknown=len(values) - len(known),
            known_rate=len(known) / len(values),
            passed=sum(known),
            failed=len(known) - sum(known),
            pass_rate_given_known=sum(known) / len(known) if known else None,
        )
    return result


def run_synthetic_endpoint(root, *, spec, endpoint_id, runtime, policy_tensor_sha256="e" * 64):
    """Explicit CPU fixture core; never writes into the production namespace."""
    root = _plain_path(root)
    plan = _plan(spec)
    endpoint = _endpoint(plan, endpoint_id)
    public, tasks = _inputs(root, plan)
    identity = dict(
        format=FORMAT,
        execution_origin="SYNTHETIC_TEST",
        run_root=str(root),
        evaluation_run_id="SYNTHETIC/" + endpoint_id,
        endpoint_id=endpoint_id,
        parent=endpoint["parent"],
        checkpoint_step=endpoint["checkpoint_step"],
        policy_tensor_sha256=policy_tensor_sha256,
        checkpoint_commit_sha256=None if endpoint["kind"] == "FROZEN_PARENT" else "c" * 64,
        checkpoint_run_id=endpoint.get("run_id"),
        phase_identity_sha256="a" * 64,
        pilot_spec_sha256=digest(spec),
        source_sha256="b" * 64,
        data_sha256="d" * 64,
        input_refs=plan["inputs"],
    )
    with EvaluationLedger(root, identity, plan, public, tasks) as ledger:
        runtime.identity = copy.deepcopy(identity)
        report = _execute(ledger, runtime)
        _write(ledger.directory / "COMPLETE.json", report)
        return report


def _admitted(root, phase_identity_sha256, pilot_spec_sha256, admission_ref):
    from .admission import require_pilot_admission

    result = require_pilot_admission(
        root,
        phase_identity_sha256=phase_identity_sha256,
        pilot_spec_sha256=pilot_spec_sha256,
        admission_ref=admission_ref,
    )
    _require(
        result.get("status") == "VERIFIED_PILOT_ADMISSION"
        and result.get("scope") == "PILOT_V1_REGISTERED_ONLY"
        and result.get("generation_authorized") is True,
        "Actual PILOT scientific admission required",
    )
    return result


@contextmanager
def _source(root, spec, endpoint):
    """Keep the PRIMARY producer lock live across endpoint load and evaluation."""
    if endpoint["kind"] == "FROZEN_PARENT":
        report = _bound(root, endpoint["reference"])
        _require(
            report.get("parent_id") == endpoint["parent"]
            and report.get("step") == 0
            and report.get("policy_hash") == endpoint["tensor_sha256"],
            "Registered parent evidence changed",
        )
        yield dict(
            policy=endpoint["tensor_sha256"],
            commit=None,
            checkpoint_run_id=None,
            refs=[endpoint["reference"]],
            tx=None,
        )
        return
    paths = [p for p in spec["paths"] if p["run_id"] == endpoint["run_id"]]
    _require(len(paths) == 1, "Endpoint producer run not registered")
    item = paths[0]
    run_dir = _plain_path(root / REGISTRATION_DIRECTORY / item["run_directory"])
    ref = dict(item["transaction_identity"])
    ref["path"] = f"{REGISTRATION_DIRECTORY}/{ref['path']}"
    identity = _bound(root, ref)
    _require(
        identity.get("lane") == "PRIMARY"
        and identity.get("max_steps") == 12
        and identity.get("run_id") == endpoint["run_id"]
        and identity.get("parent") == endpoint["parent"],
        "Only the registered PRIMARY step12 endpoint is evaluable",
    )
    with RunTransactions(run_dir, identity) as tx:
        markers = tx.published()
        _require(
            [m["step"] for m in markers] == list(range(13)),
            "All published primary steps 0..12 required before endpoint evaluation",
        )
        marker = markers[-1]
        path = run_dir / "checkpoints/step-012/COMMIT.json"
        _require(
            path.relative_to(root).as_posix() == endpoint["expected_commit_path"]
            and _read(path) == marker,
            "Evaluation endpoint COMMIT path/content differs",
        )
        yield dict(
            policy=marker["policy_sha256"],
            commit=marker["commit_sha256"],
            checkpoint_run_id=identity["run_id"],
            refs=[ref, _reference(root, path)],
            tx=tx,
        )


def _identity(root, admitted, endpoint, source, pilot_spec_sha256, admission_ref):
    context, spec = admitted["phase_context"], admitted["pilot_spec"]
    parent = endpoint["parent"]
    experiment = spec["paths"][0]["run_id"].split("/PILOT/", 1)[0]
    return dict(
        format=FORMAT,
        execution_origin="SERVER_ACTUAL",
        run_root=str(root),
        evaluation_run_id=f"{experiment}/PILOT_DEV_EVAL/{endpoint['endpoint_id']}",
        endpoint_id=endpoint["endpoint_id"],
        parent=parent,
        checkpoint_step=endpoint["checkpoint_step"],
        policy_tensor_sha256=source["policy"],
        checkpoint_commit_sha256=source["commit"],
        checkpoint_run_id=source["checkpoint_run_id"],
        phase_identity_sha256=context["identity_sha256"],
        pilot_spec_sha256=pilot_spec_sha256,
        source_sha256=context["identity"]["source_manifest"]["sha256"],
        data_sha256=context["identity"]["data_manifest"]["sha256"],
        reference_sha256=spec["parent_tensors"][parent]["tensor_sha256"],
        admission_ref=copy.deepcopy(admission_ref),
        input_refs=spec["evaluation"]["inputs"],
        endpoint_refs=source["refs"],
    )


def run_endpoint(
    root, *, endpoint_id, task_id, phase_identity_sha256, pilot_spec_sha256, admission_ref
):
    """Run only one registered endpoint; no runtime or sampler override exists."""
    from mm_core.training import capture_rng, restore_rng

    from . import runtime as native
    from .access import open_evaluation_access

    root = _plain_path(root)
    _require(root.is_dir(), "Existing canonical experiment root required")
    admitted = _admitted(root, phase_identity_sha256, pilot_spec_sha256, admission_ref)
    spec = admitted["pilot_spec"]
    plan = _plan(spec)
    endpoint = _endpoint(plan, endpoint_id)
    public, tasks = _inputs(root, plan)
    with _source(root, spec, endpoint) as source:
        identity = _identity(root, admitted, endpoint, source, pilot_spec_sha256, admission_ref)
        with EvaluationLedger(root, identity, plan, public, tasks) as ledger:
            groups = ledger.completed()
            if groups is not None:
                report = _endpoint_report(ledger, groups)
                _write(ledger.directory / "COMPLETE.json", report)
                return report
            _require(
                not (ledger.directory / "COMPLETE.json").exists(),
                "Endpoint completion marker contradicts incomplete groups",
            )
            rng = capture_rng()
            try:
                runtime, optimizer, scheduler = native.load(
                    root,
                    endpoint["parent"],
                    task_id,
                    mode="EVALUATION",
                    phase="PILOT_V1",
                    phase_identity_sha256=phase_identity_sha256,
                    audit_root=ledger.directory,
                    run_id=identity["evaluation_run_id"],
                )
                _require(
                    type(runtime) is native.Runtime
                    and optimizer is None
                    and scheduler is None
                    and runtime.execution_mode() == "EVALUATION",
                    "Production evaluation requires actual read-only native runtime",
                )
                _require(
                    runtime.torch.cuda.is_available() and runtime.torch.cuda.device_count() == 1,
                    "CPU cannot produce actual evaluation evidence",
                )
                if source["tx"] is not None:
                    native.load_committed_endpoint(
                        runtime, source["tx"], expected_commit_sha256=source["commit"]
                    )
                _require(
                    runtime.policy_tensor_identity() == source["policy"]
                    and not runtime.model.training
                    and all(
                        not p.requires_grad and p.grad is None for p in runtime.model.parameters()
                    ),
                    "Evaluation policy is not frozen read-only",
                )
                with open_evaluation_access(
                    runtime,
                    experiment_root=root,
                    pilot_spec_sha256=pilot_spec_sha256,
                    admission_ref=admission_ref,
                    ledger=ledger,
                ):
                    report = _execute(ledger, runtime)
                _require(
                    runtime.policy_tensor_identity() == source["policy"]
                    and all(p.grad is None for p in runtime.model.parameters()),
                    "Evaluation modified policy/gradients",
                )
                _write(ledger.directory / "COMPLETE.json", report)
                return report
            finally:
                restore_rng(rng)


def _percentile(values, probability):
    values = sorted(values)
    location = (len(values) - 1) * probability
    lower = int(location)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (
        location - lower
    )


def _interval(by_root, *, seed):
    """Stratified paired-root bootstrap, conditional on these 24 roots/checkpoints."""
    groups = defaultdict(list)
    for root, (family, numerator, denominator) in by_root.items():
        groups[family].append((root, numerator, denominator))
    _require(
        set(groups) == set(FAMILIES) and all(len(v) == 4 for v in groups.values()),
        "Intervals require all 24 registered roots, four per family",
    )
    rng = random.Random(seed)
    ratios = []
    ordered = [sorted(groups[family]) for family in FAMILIES]
    for _ in range(BOOTSTRAPS):
        chosen = [rng.choice(family) for family in ordered for _ in range(4)]
        denominator = sum(value[2] for value in chosen)
        if denominator:
            ratios.append(sum(value[1] for value in chosen) / denominator)
    return dict(
        level=0.95,
        lower=_percentile(ratios, 0.025) if ratios else None,
        upper=_percentile(ratios, 0.975) if ratios else None,
        bootstrap_replicates=BOOTSTRAPS,
        defined_replicates=len(ratios),
        seed=seed,
        method="PERCENTILE_STRATIFIED_ROOT_BOOTSTRAP_CONDITIONAL_ON_FIXED_ENDPOINTS",
        sampling_unit="24 roots; four per family; all four views and two draws stay together",
        training_repeat_variance_estimated=False,
    )


def summarize_reports(reports, *, master_seed=2026102701):
    """Describe exactly 12 complete endpoints; never a scientific admission gate."""
    _require(
        type(master_seed) is int and master_seed == SUMMARY_RECIPE["master_seed"],
        "Evaluation summary seed is frozen",
    )
    expected = {f"{p}/parent" for p in ("b0", "b1", "b2")} | {
        f"{p}/{a}/step12" for p in ("b0", "b1", "b2") for a in ("U10", "U11", "C_SHUF")
    }
    indexed = {report["endpoint"]["endpoint_id"]: report for report in reports}
    _require(
        len(reports) == len(indexed) == 12 and set(indexed) == expected,
        "All 12 registered endpoints are required; no partial denominator",
    )
    details, comparisons = {}, {}
    for identifier, report in sorted(indexed.items()):
        _require(
            report["responses"] == 192
            and len(report["records"]) == 192
            and report["status"] in ("SYNTHETIC_ONLY", "COMPLETE_NOT_GATE_SIGNED"),
            "Incomplete endpoint report",
        )
        records = report["records"]
        ids = {(r["qid"], r["draw"]) for r in records}
        roots = Counter(r["root_id"] for r in records)
        _require(
            len(ids) == 192 and len(roots) == 24 and set(roots.values()) == {8},
            "Repeated endpoint row or root denominator differs",
        )
        metrics = _metrics(records)
        for label in LABELS:
            for statistic in ("known_rate", "pass_rate_given_known"):
                roots = {}
                for r in records:
                    root = roots.setdefault(r["root_id"], [r["family"], 0, 0])
                    value = r["score"][label]
                    root[1] += (
                        int(value is not None) if statistic == "known_rate" else int(value == 1)
                    )
                    root[2] += 1 if statistic == "known_rate" else int(value is not None)
                metrics[label][statistic + "_interval"] = _interval(
                    roots,
                    seed=seed_for(
                        master_seed, "pilot_dev_eval_summary", identifier, label, statistic
                    ),
                )
        details[identifier] = metrics
        if report["endpoint"]["kind"] == "FROZEN_PARENT":
            continue
        parent = indexed[f"{report['endpoint']['parent']}/parent"]
        baseline = {(r["qid"], r["draw"]): r for r in parent["records"]}
        _require(set(baseline) == ids, "Endpoint/parent pairing differs")
        paired = {}
        for metric in (*LABELS, "S_known", "B_known"):
            roots = {}
            for r in records:
                before = baseline[r["qid"], r["draw"]]
                _require(r["seed"] == before["seed"], "Paired evaluation seeds differ")
                key = metric.removesuffix("_known")
                a, b = r["score"][key], before["score"][key]
                root = roots.setdefault(r["root_id"], [r["family"], 0, 0])
                if metric.endswith("_known"):
                    root[1] += int(a is not None) - int(b is not None)
                    root[2] += 1
                elif a is not None and b is not None:
                    root[1] += a - b
                    root[2] += 1
            denominator = sum(v[2] for v in roots.values())
            paired[metric] = dict(
                difference=sum(v[1] for v in roots.values()) / denominator if denominator else None,
                paired_denominator=denominator,
                excluded_unknown_pairs=192 - denominator,
                conditioning="Both semantic claims known"
                if metric in ("S", "B")
                else "All registered question/draw pairs",
                interval=_interval(
                    roots,
                    seed=seed_for(
                        master_seed,
                        "pilot_dev_eval_summary",
                        identifier,
                        "parent_difference",
                        metric,
                    ),
                ),
            )
        comparisons[identifier] = paired
    synthetic = any(report["execution_origin"] == "SYNTHETIC_TEST" for report in reports)
    return dict(
        format=FORMAT,
        status="SYNTHETIC_ONLY" if synthetic else "DESCRIPTIVE_INPUTS_NOT_INDEPENDENTLY_VERIFIED",
        checkpoints=12,
        responses=2304,
        endpoints=details,
        differences_from_parent=comparisons,
        gate_status="NOT_SIGNED",
        training_authorized=False,
        significance_gate=None,
        descriptive_summary=copy.deepcopy(SUMMARY_RECIPE),
        interpretation=(
            "Conditional DEV_EVAL description of one pilot repeat per fixed parent/arm. "
            "Root-resampling intervals do not estimate training-repeat variance, establish "
            "72-step power, or authorize formal tests/training."
        ),
    )


def verify_evaluation(
    root, *, phase_identity_sha256, pilot_spec_sha256, admission_ref, summary_ref=None
):
    """Read and replay every complete endpoint; missing/UNKNOWN never become zeros."""
    root = _plain_path(root)
    admitted = _admitted(root, phase_identity_sha256, pilot_spec_sha256, admission_ref)
    spec = admitted["pilot_spec"]
    plan = _plan(spec)
    public, tasks = _inputs(root, plan)
    reports, inputs = [], []
    with ExitStack() as locks:
        for endpoint in plan["endpoints"]:
            source = locks.enter_context(_source(root, spec, endpoint))
            identity = _identity(root, admitted, endpoint, source, pilot_spec_sha256, admission_ref)
            ledger = locks.enter_context(EvaluationLedger(root, identity, plan, public, tasks))
            groups = ledger.completed()
            _require(groups is not None, "All 12 endpoints must be complete before aggregation")
            report = _endpoint_report(ledger, groups)
            _require(
                _read(ledger.directory / "COMPLETE.json") == report,
                "Endpoint completion report differs from original raw replay",
            )
            reports.append(report)
            inputs.append(_reference(root, ledger.directory / "COMPLETE.json"))
        result = summarize_reports(reports)
        result.update(
            status="COMPLETE_DESCRIPTIVE_NOT_GATE_SIGNED",
            phase_identity_sha256=phase_identity_sha256,
            pilot_spec_sha256=pilot_spec_sha256,
            admission_ref=admission_ref,
            inputs=inputs,
        )
        if summary_ref is not None:
            _require(
                summary_ref.get("path") == DIRECTORY + "/SUMMARY.json"
                and _bound(root, summary_ref) == result,
                "Saved evaluation summary differs from complete original evidence",
            )
        return result


def aggregate(root, *, phase_identity_sha256, pilot_spec_sha256, admission_ref):
    """Publish the deterministic description after the read-only evidence replay."""
    result = verify_evaluation(
        root,
        phase_identity_sha256=phase_identity_sha256,
        pilot_spec_sha256=pilot_spec_sha256,
        admission_ref=admission_ref,
    )
    _write(_plain_path(root) / DIRECTORY / "SUMMARY.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "aggregate"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--phase-identity-sha256", required=True)
    parser.add_argument("--pilot-spec-sha256", required=True)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument("--endpoint-id")
    parser.add_argument("--task-id")
    args = parser.parse_args()
    kwargs = dict(
        phase_identity_sha256=args.phase_identity_sha256,
        pilot_spec_sha256=args.pilot_spec_sha256,
        admission_ref={"path": "PHASES/PILOT_V1/ADMISSION.json", "sha256": args.admission_sha256},
    )
    if args.action == "run":
        if not args.endpoint_id or not args.task_id:
            parser.error("run requires --endpoint-id and --task-id")
        result = run_endpoint(
            args.root, endpoint_id=args.endpoint_id, task_id=args.task_id, **kwargs
        )
    else:
        result = aggregate(args.root, **kwargs)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "records"}, ensure_ascii=False, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
