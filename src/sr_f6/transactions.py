"""F6 checkpoint transactions only: no generation, optimizer update, or gate signing.

Hold RunTransactions' lock across the caller's complete attempt. Only published
step-NNN directories are commits; staging is preserved and never recovered.
The caller must freeze this interface in its phase registration before use.

Identity requires experiment_id/run_id/phase/parent/arm/repeat/lane/max_steps and
source/data/config/reference SHA256s. sampler_state.consumed_qids maps committed
step strings to sixteen qids; actual schedule validity belongs to the caller. References
are relative files below this run, with path/sha256. Rollout and credit JSON
envelopes bind the same identity and step; rollout contains all 128 records,
policy_before_sha256, rng_before_sha256, rng_after_sha256, and ROLLOUT_READY.
Credit contains all sixteen native credit receipts in groups. Evidence is the
PREUPDATE_PASS JSON produced by training.update. This layer validates structural
completeness and persistence; the trainer remains responsible for actual scores,
credit mathematics and probability gates. No failed attempt is deleted here.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import uuid
from pathlib import Path, PurePosixPath

from mm_core.training import capture_rng, restore_rng, state_hash, trainable_state
from sr_f3_ce.training import finite_tree, validate_optimizer, validate_optimizer_state
from sr_f4.training import save_checkpoint

FORMAT = "srf6-checkpoint-transaction-v1"
FAILPOINTS = ("after_state", "after_marker", "before_rename", "after_rename")


def _require(condition, message):
    if not condition:
        raise PermissionError(message)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _plain_path(path):
    """Reject symlinks in every existing component, including run ancestors."""
    path = Path(os.path.abspath(path))
    for candidate in (*reversed(path.parents), path):
        _require(not candidate.is_symlink(), f"Symlink forbidden: {candidate}")
    return path


def _read_bytes(path):
    path = _plain_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        _require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "Regular file required")
        return stream.read()


def _read_json(path):
    return json.loads(_read_bytes(path))


def _file_hash(path):
    path = _plain_path(path)
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        _require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "Regular state file required")
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    path = _plain_path(path)
    with path.open("xb") as stream:
        stream.write(_canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_dir(path):
    fd = os.open(_plain_path(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _identity(identity):
    _require(isinstance(identity, dict), "Explicit run identity required")
    for key in ("experiment_id", "run_id", "phase", "lane"):
        _require(
            isinstance(identity.get(key), str) and bool(identity[key]), f"Missing identity {key}"
        )
    _require(identity.get("parent") in ("b0", "b1", "b2"), "Unregistered parent")
    _require(
        identity.get("arm") in ("U00", "U01", "U10", "U11", "C_GLOBAL", "C_SHUF"),
        "Unregistered arm",
    )
    _require(
        type(identity.get("repeat")) is int and identity["repeat"] in range(3), "Invalid repeat"
    )
    _require(
        type(identity.get("max_steps")) is int and identity["max_steps"] in (12, 72),
        "Pilot/formal limit must be explicit",
    )
    for key in ("source_sha256", "data_sha256", "config_sha256", "reference_sha256"):
        _require(_sha(identity.get(key)), f"Missing identity {key}")
    _canonical(identity)


def _cursor(sampler, step, previous=None, qids=None, rollout_reference=None):
    _require(isinstance(sampler, dict), "Complete sampler state required")
    _require("rollout_rng_reference" in sampler, "Explicit rollout RNG reference required")
    _require(
        type(sampler.get("next_step")) is int and sampler["next_step"] == step + 1,
        "Wrong next-step cursor",
    )
    _require(
        type(sampler.get("committed_log_step")) is int and sampler["committed_log_step"] == step,
        "Wrong committed cursor",
    )
    consumed = sampler.get("consumed_qids")
    _require(
        isinstance(consumed, dict) and set(consumed) == {str(i) for i in range(1, step + 1)},
        "Incomplete step-indexed consumption",
    )
    for values in consumed.values():
        _require(
            isinstance(values, list)
            and len(values) == 16
            and all(isinstance(v, str) and v for v in values)
            and len(set(values)) == 16,
            "Sixteen distinct qids per consumed step required",
        )
    _require(
        _integer(sampler.get("logical_calls")) and sampler["logical_calls"] == 128 * step,
        "Logical call count must retain 128 per committed update",
    )
    _require(
        _integer(sampler.get("actual_calls"))
        and sampler["actual_calls"] >= sampler["logical_calls"],
        "Physical calls cannot disappear from accounting",
    )
    expected = {"step": step, "groups": 16 if step else 0, "rows": 128 if step else 0}
    _require(
        sampler.get("rollout_cursor") == expected
        and all(type(v) is int for v in sampler["rollout_cursor"].values()),
        "Incomplete rollout cursor",
    )
    if step == 0:
        _require(
            sampler["actual_calls"] == 0 and sampler.get("rollout_rng_reference") is None,
            "Step zero requires explicit empty rollout/RNG state",
        )
    else:
        _require(
            sampler.get("rollout_rng_reference") == rollout_reference,
            "Rollout RNG reference differs",
        )
        if qids is not None:
            _require(consumed[str(step)] == qids, "Consumption differs from actual G8 qids")
    if previous is not None:
        _require(
            all(consumed.get(k) == v for k, v in previous["consumed_qids"].items()),
            "Previously consumed qids changed",
        )
        _require(
            sampler["actual_calls"] >= previous["actual_calls"] + 128,
            "New registered rollout calls missing",
        )
    _canonical(sampler)


def _optimizer_cursor(optimizer_state, scheduler_state, step, parameters):
    import torch

    validate_optimizer_state(optimizer_state, scheduler_state)
    _require(
        finite_tree(optimizer_state) and finite_tree(scheduler_state),
        "Nonfinite optimizer/scheduler state",
    )
    _require(
        scheduler_state.get("last_epoch") == step
        and scheduler_state.get("_step_count") == step + 1,
        "Scheduler did not step exactly once per commit",
    )
    ids = [pid for group in optimizer_state["param_groups"] for pid in group["params"]]
    _require(
        len(ids) == len(parameters) and len(set(ids)) == len(parameters),
        "Optimizer parameter ownership mismatch",
    )
    states = optimizer_state["state"]
    _require(set(states) == (set(ids) if step else set()), "Partial or inherited Adam state")
    for pid, parameter in zip(ids, parameters, strict=True):
        _require(parameter.dtype == torch.float32, "FP32 trainable parameters required")
        if not step:
            continue
        item = states[pid]
        _require(float(item.get("step", -1)) == step, "Adam did not step exactly once per commit")
        _require("exp_avg" in item and "exp_avg_sq" in item, "Incomplete Adam moments")
        for key in ("exp_avg", "exp_avg_sq"):
            _require(
                isinstance(item[key], torch.Tensor)
                and item[key].dtype == torch.float32
                and item[key].shape == parameter.shape,
                "Adam moment shape/dtype differs from owned parameter",
            )


class RunTransactions:
    """One locked F6 run/lane. Methods require this object's live context."""

    def __init__(self, directory, identity):
        _identity(identity)
        self.directory = _plain_path(directory)
        self.identity = copy.deepcopy(identity)
        self.checkpoints = self.directory / "checkpoints"
        self._fd = None
        self._pid = None

    def __enter__(self):
        _require(self._fd is None, "Transaction lock already acquired")
        _plain_path(self.directory).mkdir(parents=True, exist_ok=True)
        lock = _plain_path(self.directory / ".run.lock")
        fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._fd, self._pid = fd, os.getpid()
            _plain_path(self.checkpoints)
            identity_path = self.directory / "RUN_IDENTITY.json"
            if identity_path.exists() or identity_path.is_symlink():
                _require(_read_json(identity_path) == self.identity, "Run identity differs")
            else:
                _require(
                    not self.checkpoints.exists() or not any(self.checkpoints.iterdir()),
                    "Cannot adopt checkpoints without run identity",
                )
                _write_json(identity_path, self.identity)
                _fsync_dir(self.directory)
            _plain_path(self.checkpoints).mkdir(exist_ok=True)
            return self
        except BaseException:
            self._fd, self._pid = None, None
            os.close(fd)
            raise

    def __exit__(self, *_):
        if self._fd is not None:
            os.close(self._fd)
            self._fd, self._pid = None, None

    def _locked(self):
        _require(
            self._fd is not None and self._pid == os.getpid(), "Live exclusive run lock required"
        )
        path = _plain_path(self.directory / ".run.lock")
        _require(os.fstat(self._fd).st_ino == path.stat().st_ino, "Lock inode changed")
        _require(
            _read_json(self.directory / "RUN_IDENTITY.json") == self.identity,
            "Run identity changed while locked",
        )

    def _reference(self, reference):
        _require(
            isinstance(reference, dict)
            and set(reference) == {"path", "sha256"}
            and _sha(reference["sha256"]),
            "Explicit path/SHA256 reference required",
        )
        value = reference["path"]
        _require(isinstance(value, str) and bool(value), "Relative evidence path required")
        relative = PurePosixPath(value)
        _require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and relative.as_posix() == value
            and value != ".",
            "Evidence must be a canonical run descendant",
        )
        path = _plain_path(self.directory / value)
        _require(path.is_relative_to(self.directory), "Evidence escaped run")
        data = _read_bytes(path)
        _require(
            hashlib.sha256(data).hexdigest() == reference["sha256"], "Evidence content hash differs"
        )
        return json.loads(data)

    def published(self):
        """Validate small commit-chain receipts; never read checkpoint tensors."""
        self._locked()
        markers = []
        for path in sorted(self.checkpoints.iterdir()):
            _plain_path(path)
            if path.name.startswith(".staging-"):
                continue
            _require(
                path.is_dir() and re.fullmatch(r"step-\d{3}", path.name) is not None,
                "Unexpected checkpoint entry",
            )
            marker = _read_json(path / "COMMIT.json")
            _require(isinstance(marker, dict), "Invalid commit marker")
            step = marker.get("step")
            _require(
                _integer(step)
                and step <= self.identity["max_steps"]
                and path.name == f"step-{step:03d}",
                "Commit step/path mismatch",
            )
            _require(
                marker.get("format") == FORMAT
                and marker.get("status") == "COMMITTED"
                and marker.get("identity") == self.identity,
                "Commit identity mismatch",
            )
            digest = marker.get("commit_sha256")
            _require(
                _sha(digest)
                and digest == _digest({k: v for k, v in marker.items() if k != "commit_sha256"}),
                "Commit marker checksum differs",
            )
            _require(
                set(p.name for p in path.iterdir())
                == (
                    {"state.pt", "COMMIT.json", "UPDATE.json"}
                    if step
                    else {"state.pt", "COMMIT.json"}
                ),
                "Partial or unexpected checkpoint files",
            )
            _require(
                _plain_path(path / "state.pt").is_file() and _sha(marker.get("state_sha256")),
                "Missing state file/identity",
            )
            _require(
                _sha(marker.get("policy_sha256")) and _sha(marker.get("rng_sha256")),
                "Missing policy/RNG identity",
            )
            if step:
                _require(
                    _plain_path(path / "UPDATE.json").is_file()
                    and _sha(marker.get("update_sha256")),
                    "Missing UPDATE receipt/identity",
                )
            _require(step == len(markers), "Published checkpoint chain has a gap")
            previous = markers[-1] if markers else None
            _require(
                marker.get("previous_commit_sha256")
                == (previous["commit_sha256"] if previous else None),
                "Commit chain differs",
            )
            _require(
                marker.get("policy_before_sha256")
                == (previous["policy_sha256"] if previous else self.identity["reference_sha256"]),
                "Policy chain differs",
            )
            _cursor(
                marker.get("sampler_state"),
                step,
                previous["sampler_state"] if previous else None,
                rollout_reference=marker.get("references", {}).get("rollout"),
            )
            markers.append(marker)
        return markers

    def _runtime(self, runtime):
        for runtime_key, identity_key in (
            ("run_id", "run_id"),
            ("parent_id", "parent"),
            ("source_sha256", "source_sha256"),
            ("data_hash", "data_sha256"),
            ("reference_hash", "reference_sha256"),
        ):
            _require(
                runtime.identity.get(runtime_key) == self.identity[identity_key],
                f"Runtime {runtime_key} differs",
            )
        _require(
            runtime.identity.get("reference_parent") == self.identity["parent"],
            "Wrong reference parent",
        )

    def _evidence(self, references, update, step, policy_before, rng_hash):
        import torch

        from .training import trajectory_hash

        _require(
            isinstance(references, dict) and set(references) == {"rollout", "credit", "evidence"},
            "Rollout/credit/evidence references required",
        )
        rollout = self._reference(references["rollout"])
        credit = self._reference(references["credit"])
        evidence = self._reference(references["evidence"])
        for item in (rollout, credit):
            _require(
                isinstance(item, dict)
                and item.get("identity") == self.identity
                and item.get("step") == step,
                "Cached artifact identity/step differs",
            )
        _require(
            rollout.get("status") == "ROLLOUT_READY"
            and rollout.get("policy_before_sha256") == policy_before,
            "Partial or wrong-policy rollout cache",
        )
        _require(
            _sha(rollout.get("rng_before_sha256")) and rollout.get("rng_after_sha256") == rng_hash,
            "Rollout-after RNG differs from update/checkpoint",
        )
        records, groups = rollout.get("records"), credit.get("groups")
        _require(
            isinstance(records, list)
            and len(records) == 128
            and isinstance(groups, list)
            and len(groups) == 16,
            "Full 128 rows and sixteen credit G8s required",
        )
        row_ids, qids, group_ids = set(), [], set()
        for offset, group in zip(range(0, 128, 8), groups, strict=True):
            block = records[offset : offset + 8]
            _require(
                all(isinstance(row, dict) for row in block) and isinstance(group, dict),
                "Malformed G8 cache",
            )
            qid, gid = block[0].get("qid"), block[0].get("group_id")
            _require(
                isinstance(qid, str)
                and bool(qid)
                and qid not in qids
                and isinstance(gid, str)
                and bool(gid)
                and gid not in group_ids,
                "Distinct sixteen qid/G8 identities required",
            )
            qids.append(qid)
            group_ids.add(gid)
            ids = []
            for draw, row in enumerate(block):
                rid = row.get("row_id")
                _require(
                    isinstance(rid, str) and bool(rid) and rid not in row_ids,
                    "Duplicate/missing raw row identity",
                )
                row_ids.add(rid)
                ids.append(rid)
                _require(
                    row.get("qid") == qid
                    and row.get("group_id") == gid
                    and type(row.get("draw")) is int
                    and row["draw"] == draw,
                    "Partial or mixed G8 draws",
                )
                for key, expected in (
                    ("run_id", self.identity["run_id"]),
                    ("parent", self.identity["parent"]),
                    ("source_sha256", self.identity["source_sha256"]),
                    ("step", step),
                    ("policy_hash", policy_before),
                ):
                    _require(row.get(key) == expected, f"Cached row {key} differs")
                tokens, logps = row.get("tokens"), row.get("sampler_logprobs")
                _require(
                    isinstance(tokens, list)
                    and 1 <= len(tokens) <= 1536
                    and all(_integer(t) for t in tokens),
                    "Original completion tokens required",
                )
                _require(
                    isinstance(logps, list)
                    and len(logps) == len(tokens)
                    and all(type(p) in (int, float) and math.isfinite(p) and p <= 0 for p in logps),
                    "Original finite sampler probabilities required",
                )
                _require(_integer(row.get("seed")), "Original draw seed required")
            _require(
                group.get("qid") == qid
                and group.get("group_id") == gid
                and group.get("row_ids") == ids
                and group.get("draws") == list(range(8))
                and group.get("arm") == self.identity["arm"]
                and group.get("policy_hash") == policy_before,
                "Credit cache does not bind actual eight rows",
            )
            scored = group.get("scored_rows")
            _require(
                isinstance(scored, list) and len(scored) == 8, "Full scored credit rows required"
            )
            _require(
                _sha(group.get("scored_group_sha256"))
                and _digest(scored) == group["scored_group_sha256"],
                "Scored credit group checksum differs",
            )
            for row, scored_row in zip(block, scored, strict=True):
                score = row.get("score", row)
                _require(isinstance(score, dict), "Malformed nested score")
                for key in ("J", "V", "finished", "S", "B"):
                    _require(key in score, "Missing scored field")
                    _require(
                        key not in row or row[key] == score[key],
                        "Conflicting top-level and nested score",
                    )
                required = {
                    "row_id",
                    "draw",
                    "qid",
                    "group_id",
                    "policy_hash",
                    "parent",
                    "factor",
                    "root_id",
                    "family",
                    "J",
                    "V",
                    "finished",
                    "S",
                    "B",
                }
                _require(
                    isinstance(scored_row, dict) and required <= set(scored_row),
                    "Incomplete scored credit row",
                )
                for key, value in scored_row.items():
                    source = score if key in ("J", "V", "finished", "S", "B") else row
                    _require(
                        key in source and source[key] == value,
                        "Raw labels/identity differ from applied credit",
                    )
        _require(
            isinstance(evidence, dict)
            and evidence.get("status") == "PREUPDATE_PASS"
            and evidence.get("run_id") == self.identity["run_id"]
            and evidence.get("committed_source_step") == step - 1
            and evidence.get("policy_before_sha256") == policy_before,
            "Preupdate evidence identity differs",
        )
        _require(
            isinstance(update, dict)
            and update.get("status") == "UPDATED_UNCOMMITTED"
            and update.get("optimizer_updates") == 1
            and update.get("sequence_count") == 128
            and update.get("group_sizes") == [8] * 16,
            "One complete uncommitted update required",
        )
        _require(
            update.get("actual_credit_groups") == groups
            and update.get("policy_before_sha256") == policy_before
            and update.get("reference_sha256") == self.identity["reference_sha256"],
            "Update policy/reference/credit differs",
        )
        traces = update.get("probability_vectors")
        _require(
            isinstance(traces, list) and len(traces) == 128,
            "All 128 actual probability traces required",
        )
        for row, trace in zip(records, traces, strict=True):
            _require(isinstance(trace, dict), "Malformed probability trace")
            for key in ("row_id", "qid", "tokens", "token_path_sha256"):
                _require(
                    key in row and trace.get(key) == row[key],
                    "Raw trajectory differs from update probability trace",
                )
            _require(
                _sha(row["token_path_sha256"]) and trajectory_hash(row) == row["token_path_sha256"],
                "Raw token/prompt trajectory hash differs",
            )
            # training.update records the sampler vector after its registered
            # FP32 conversion; compare that exact representation, never a tolerance.
            sampler = torch.tensor(row["sampler_logprobs"], dtype=torch.float32).tolist()
            _require(
                trace.get("sampler_logprobs") == sampler,
                "Raw sampler probabilities differ from update trace",
            )
            for key in ("train_logprobs", "reference_logprobs"):
                values = trace.get(key)
                _require(
                    isinstance(values, list)
                    and len(values) == len(row["tokens"])
                    and all(
                        type(v) in (int, float) and math.isfinite(v) and v <= 0 for v in values
                    ),
                    "Incomplete/nonfinite actual probability trace",
                )
        _require(
            update.get("rng_restored") is True
            and update.get("rng_before_sha256") == rng_hash
            and update.get("rng_after_sha256") == rng_hash,
            "Update RNG restoration not evidenced",
        )
        _canonical(update)
        return qids

    def commit(
        self,
        runtime,
        optimizer,
        scheduler,
        *,
        step,
        sampler_state,
        update=None,
        references=None,
        failpoint=None,
    ):
        """Publish step0 or one consecutive update. Exceptions require restoration.

        failpoint, when supplied for local tests, is called with the four named
        boundaries in FAILPOINTS. A post-rename exception still leaves a commit.
        """
        self._locked()
        self._runtime(runtime)
        markers = self.published()
        _require(
            _integer(step) and step == len(markers) and step <= self.identity["max_steps"],
            "Commit must be unique and consecutive",
        )
        previous = markers[-1] if markers else None
        _require(
            type(runtime.identity.get("step")) is int
            and runtime.identity["step"] == max(0, step - 1),
            "Runtime must retain previous committed step",
        )
        validate_optimizer(optimizer, scheduler, model=runtime.model)
        policy = trainable_state(runtime.model)
        _require(bool(policy) and finite_tree(policy), "Nonfinite/empty trainable state")
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        names = {id(p): name for name, p in runtime.model.named_parameters() if p.requires_grad}
        parameter_names = [names[id(p)] for p in parameters]
        _optimizer_cursor(optimizer.state_dict(), scheduler.state_dict(), step, parameters)
        policy_hash, rng_hash = state_hash(policy), state_hash(capture_rng())
        before = previous["policy_sha256"] if previous else self.identity["reference_sha256"]
        refs = copy.deepcopy(references if references is not None else {})
        if step:
            qids = self._evidence(refs, update, step, before, rng_hash)
            _require(
                update.get("policy_after_sha256") == policy_hash,
                "Actual post-update policy differs",
            )
        else:
            _require(
                update is None and not refs and policy_hash == before,
                "Step zero must be original parent and empty update",
            )
            qids = None
        sampler = copy.deepcopy(sampler_state)
        _cursor(
            sampler,
            step,
            previous["sampler_state"] if previous else None,
            qids,
            refs.get("rollout"),
        )
        target = self.checkpoints / f"step-{step:03d}"
        _require(not target.exists(), "Published checkpoints are immutable")
        staging = self.checkpoints / f".staging-{step:03d}-{uuid.uuid4().hex}"
        staging.mkdir()
        save_checkpoint(
            staging / "state.pt",
            runtime,
            optimizer,
            scheduler,
            identity=self.identity,
            step=step,
            sampler_state=sampler,
        )
        if failpoint:
            failpoint("after_state")
        marker = dict(
            format=FORMAT,
            status="COMMITTED",
            identity=self.identity,
            step=step,
            previous_commit_sha256=previous["commit_sha256"] if previous else None,
            policy_before_sha256=before,
            policy_sha256=policy_hash,
            parameter_names=parameter_names,
            rng_sha256=rng_hash,
            state_sha256=_file_hash(staging / "state.pt"),
            sampler_state=sampler,
            references=refs,
            gate_status="NOT_SIGNED",
            training_authorized=False,
        )
        if step:
            _write_json(staging / "UPDATE.json", update)
            marker["update_sha256"] = hashlib.sha256(
                _read_bytes(staging / "UPDATE.json")
            ).hexdigest()
        marker["commit_sha256"] = _digest(marker)
        _write_json(staging / "COMMIT.json", marker)
        if failpoint:
            failpoint("after_marker")
        _fsync_dir(staging)
        if failpoint:
            failpoint("before_rename")
        _require(
            not target.exists() and not target.is_symlink(),
            "Never overwrite a published checkpoint",
        )
        os.rename(staging, target)
        _fsync_dir(self.checkpoints)
        if failpoint:
            failpoint("after_rename")
        runtime.identity["step"] = step
        runtime.identity["policy_hash"] = policy_hash
        runtime._srf6_policy_identity = (step, policy_hash)
        return copy.deepcopy(marker)

    def restore_latest(self, runtime, optimizer, scheduler):
        """Restore highest published state, or None if no commit exists.

        A corrupt highest commit is an error, never permission to fall back.
        Same-step restoration refreshes the policy identity cache unconditionally.
        """
        import numpy as np
        import torch

        self._locked()
        self._runtime(runtime)
        markers = self.published()
        if not markers:
            return None
        marker = markers[-1]
        step = marker["step"]
        path = self.checkpoints / f"step-{step:03d}"
        _require(
            _file_hash(path / "state.pt") == marker["state_sha256"],
            "Checkpoint state content differs",
        )
        state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
        _require(
            state.get("identity") == self.identity
            and state.get("step") == step
            and state.get("sampler_state") == marker["sampler_state"],
            "Serialized checkpoint metadata differs",
        )
        policy = state["policy"]
        current = {
            name: param for name, param in runtime.model.named_parameters() if param.requires_grad
        }
        _require(set(policy) == set(current), "Trainable parameter names changed")
        for name, value in policy.items():
            _require(
                isinstance(value, torch.Tensor)
                and value.dtype == torch.float32
                and value.shape == current[name].shape
                and value.dtype == current[name].dtype,
                "Trainable parameter shape/dtype differs",
            )
        _require(
            finite_tree(policy) and state_hash(policy) == marker["policy_sha256"],
            "Checkpoint tensor policy differs",
        )
        validate_optimizer(optimizer, scheduler, model=runtime.model)
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        names = {id(p): name for name, p in current.items()}
        _require(
            [names[id(p)] for p in parameters] == marker.get("parameter_names")
            and [len(g["params"]) for g in optimizer.param_groups]
            == [len(g["params"]) for g in state["optimizer"]["param_groups"]],
            "Optimizer parameter order/group layout changed",
        )
        _optimizer_cursor(state["optimizer"], state["scheduler"], step, parameters)
        np_rng = state["numpy_rng"]
        rng = dict(
            python=state["python_rng"],
            numpy=(np_rng[0], np.asarray(np_rng[1], dtype=np.uint32), *np_rng[2:]),
            cpu=state["torch_rng"],
            cuda=state["cuda_rng"],
        )
        _require(state_hash(rng) == marker["rng_sha256"], "Checkpoint RNG differs")
        _require(
            len(rng["cuda"]) == (torch.cuda.device_count() if torch.cuda.is_available() else 0),
            "Checkpoint CUDA RNG device count differs",
        )
        if step:
            update_bytes = _read_bytes(path / "UPDATE.json")
            _require(
                hashlib.sha256(update_bytes).hexdigest() == marker["update_sha256"],
                "UPDATE receipt content differs",
            )
            update = json.loads(update_bytes)
            qids = self._evidence(
                marker["references"],
                update,
                step,
                marker["policy_before_sha256"],
                marker["rng_sha256"],
            )
            _require(
                update.get("policy_after_sha256") == marker["policy_sha256"],
                "Restored update policy differs",
            )
            _cursor(
                state["sampler_state"],
                step,
                markers[-2]["sampler_state"],
                qids,
                marker["references"]["rollout"],
            )
        with torch.no_grad():
            for name, value in policy.items():
                current[name].copy_(value)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        optimizer.zero_grad(set_to_none=True)
        restore_rng(rng)
        runtime.identity["step"] = step
        runtime.identity["policy_hash"] = marker["policy_sha256"]
        runtime._srf6_policy_identity = (step, marker["policy_sha256"])
        return {
            "step": step,
            "sampler_state": copy.deepcopy(state["sampler_state"]),
            "commit": marker,
        }
