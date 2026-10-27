"""Real torch serialization with synthetic CPU rollout/update fixtures, not GPU PASS."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mm_core.training import capture_rng, restore_rng, state_hash, trainable_state
from sr_f3_ce.training import fresh_optimizer
from sr_f6 import transactions
from sr_f6.credit import build_credit_group
from sr_f6.training import trajectory_hash
from sr_f6.transactions import FAILPOINTS, RunTransactions


def setup():
    random.seed(1907)
    np.random.seed(1907)
    torch.manual_seed(1907)
    model = torch.nn.Linear(2, 1).float()
    optimizer, scheduler = fresh_optimizer(model)
    reference = state_hash(trainable_state(model))
    identity = dict(
        experiment_id="CPU_FIXTURE_ONLY",
        run_id="fixture/PILOT/b0/r0/U11",
        phase="PILOT_V1",
        parent="b0",
        arm="U11",
        repeat=0,
        lane="MAIN",
        max_steps=12,
        source_sha256="a" * 64,
        data_sha256="b" * 64,
        config_sha256="c" * 64,
        reference_sha256=reference,
    )
    runtime = SimpleNamespace(
        model=model,
        identity=dict(
            run_id=identity["run_id"],
            parent_id="b0",
            step=0,
            source_sha256=identity["source_sha256"],
            data_hash=identity["data_sha256"],
            reference_hash=reference,
            reference_parent="b0",
        ),
        _srf6_policy_identity=(0, "stale-cache"),
    )
    return runtime, optimizer, scheduler, identity


def zero_sampler():
    return dict(
        next_step=1,
        committed_log_step=0,
        consumed_qids={},
        logical_calls=0,
        actual_calls=0,
        rollout_cursor=dict(step=0, groups=0, rows=0),
        rollout_rng_reference=None,
    )


def save_json(directory, relative, value):
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    path.write_bytes(data)
    return dict(path=relative, sha256=hashlib.sha256(data).hexdigest())


def rollout(directory, identity, runtime, step, previous):
    """Synthetic full G8 fixture; only the caller owns generation/cache lifecycle."""
    before_rng = capture_rng()
    policy = state_hash(trainable_state(runtime.model))
    records, groups = [], []
    for group_index in range(16):
        block = []
        for draw in range(8):
            block.append(
                dict(
                    row_id=f"step-{step}/q-{group_index}/draw-{draw}",
                    group_id=f"step-{step}/group-{group_index}",
                    qid=f"q-{group_index}",
                    draw=draw,
                    run_id=identity["run_id"],
                    parent="b0",
                    source_sha256=identity["source_sha256"],
                    step=step,
                    policy_hash=policy,
                    factor="P1",
                    root_id=f"root-{group_index // 4}",
                    family="BINDING",
                    variant=group_index % 4,
                    tokens=[
                        random.randrange(1, 100),
                        int(np.random.randint(1, 100)),
                        int(torch.randint(1, 100, ()).item()),
                    ],
                    sampler_logprobs=[-1.0, -2.0, -3.0],
                    seed=step * 1000 + group_index * 8 + draw,
                    J=0,
                    V=1,
                    finished=1,
                    S=draw % 2,
                    B=1,
                    prompt_tensor_hash="d" * 64,
                    native_prompt_inputs={"input_ids": [[1, 2]], "attention_mask": [[1, 1]]},
                    model_input=dict(prompt="CPU fixture", image_file=None, image_sha256=None),
                    protocol="structured_v4",
                    sampling_hash="e" * 64,
                )
            )
            block[-1]["token_path_sha256"] = trajectory_hash(block[-1])
        records.extend(block)
        groups.append(build_credit_group(block, "U11", step * 100 + group_index))
    after_rng = capture_rng()
    envelope = dict(
        status="ROLLOUT_READY",
        identity=identity,
        step=step,
        policy_before_sha256=policy,
        rng_before_sha256=state_hash(before_rng),
        rng_after_sha256=state_hash(after_rng),
        records=records,
    )
    prefix = f"artifacts/step-{step:03d}"
    references = dict(
        rollout=save_json(directory, f"{prefix}/rollout.json", envelope),
        credit=save_json(
            directory, f"{prefix}/credit.json", dict(identity=identity, step=step, groups=groups)
        ),
        evidence=save_json(
            directory,
            f"{prefix}/evidence.json",
            dict(
                status="PREUPDATE_PASS",
                run_id=identity["run_id"],
                committed_source_step=step - 1,
                policy_before_sha256=policy,
                fixture_only=True,
            ),
        ),
    )
    sampler = dict(
        next_step=step + 1,
        committed_log_step=step,
        consumed_qids={**previous["consumed_qids"], str(step): [f"q-{i}" for i in range(16)]},
        logical_calls=step * 128,
        actual_calls=previous["actual_calls"] + 128,
        rollout_cursor=dict(step=step, groups=16, rows=128),
        rollout_rng_reference=references["rollout"],
    )
    return dict(
        records=records,
        groups=groups,
        rng=after_rng,
        references=references,
        sampler=sampler,
        before=policy,
    )


def tiny_update(runtime, optimizer, scheduler, identity, cached):
    """Actual AdamW and LambdaLR; this fixture does not pretend to run F6 loss."""
    # A real rollout adapter must restore this persisted after-rollout RNG when
    # replaying its complete cache. Transaction recovery itself restores COMMIT.
    restore_rng(cached["rng"])
    before_rng = capture_rng()
    optimizer.zero_grad(set_to_none=True)
    features = torch.tensor([random.random(), float(np.random.random())]) + torch.rand(2)
    loss = (runtime.model(features).sum() - 0.125).square()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(runtime.model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    restore_rng(before_rng)
    return dict(
        status="UPDATED_UNCOMMITTED",
        optimizer_updates=1,
        sequence_count=128,
        group_sizes=[8] * 16,
        actual_credit_groups=cached["groups"],
        probability_vectors=[
            dict(
                row_id=row["row_id"],
                qid=row["qid"],
                tokens=list(row["tokens"]),
                sampler_logprobs=list(row["sampler_logprobs"]),
                train_logprobs=list(row["sampler_logprobs"]),
                reference_logprobs=list(row["sampler_logprobs"]),
                token_path_sha256=row["token_path_sha256"],
            )
            for row in cached["records"]
        ],
        policy_before_sha256=cached["before"],
        policy_after_sha256=state_hash(trainable_state(runtime.model)),
        reference_sha256=identity["reference_sha256"],
        rng_restored=True,
        rng_before_sha256=state_hash(before_rng),
        rng_after_sha256=state_hash(before_rng),
        fixture_only=True,
    )


def full_state(runtime, optimizer, scheduler):
    return state_hash(
        dict(
            policy=trainable_state(runtime.model),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            rng=capture_rng(),
        )
    )


def commit_step(tx, runtime, optimizer, scheduler, step, cached, update, failpoint=None):
    return tx.commit(
        runtime,
        optimizer,
        scheduler,
        step=step,
        sampler_state=cached["sampler"],
        update=update,
        references=cached["references"],
        failpoint=failpoint,
    )


def baseline(directory):
    runtime, optimizer, scheduler, identity = setup()
    with RunTransactions(directory, identity) as tx:
        tx.commit(runtime, optimizer, scheduler, step=0, sampler_state=zero_sampler())
        sampler = zero_sampler()
        for step in (1, 2):
            cached = rollout(directory, identity, runtime, step, sampler)
            update = tiny_update(runtime, optimizer, scheduler, identity, cached)
            commit_step(tx, runtime, optimizer, scheduler, step, cached, update)
            sampler = cached["sampler"]
        return (
            full_state(runtime, optimizer, scheduler),
            sampler,
            cached["records"],
            cached["groups"],
        )


@pytest.mark.parametrize("location", FAILPOINTS)
def test_crash_boundaries_recover_exact_baseline_and_single_logical_commit(tmp_path, location):
    expected, sampler, records, groups = baseline(tmp_path / "baseline")
    directory = tmp_path / "crashed"
    runtime, optimizer, scheduler, identity = setup()
    with RunTransactions(directory, identity) as tx:
        tx.commit(runtime, optimizer, scheduler, step=0, sampler_state=zero_sampler())
        pending = rollout(directory, identity, runtime, 1, zero_sampler())
        update = tiny_update(runtime, optimizer, scheduler, identity, pending)

        def fail(name):
            if name == location:
                raise RuntimeError("injected " + name)

        with pytest.raises(RuntimeError, match="injected"):
            commit_step(tx, runtime, optimizer, scheduler, 1, pending, update, fail)
        assert runtime.identity["step"] == 0
    # Fresh model/optimizer simulates losing failed in-memory state.
    runtime, optimizer, scheduler, _ = setup()
    with RunTransactions(directory, identity) as tx:
        restored = tx.restore_latest(runtime, optimizer, scheduler)
        assert restored["step"] == (1 if location == "after_rename" else 0)
        if restored["step"] == 0:
            # Reuse the complete same tokens/logprobs/credit/RNG. No new rollout.
            update = tiny_update(runtime, optimizer, scheduler, identity, pending)
            commit_step(tx, runtime, optimizer, scheduler, 1, pending, update)
            assert len(list((directory / "checkpoints").glob(".staging-*"))) == 1
        else:
            assert not list((directory / "checkpoints").glob(".staging-*"))
        later = rollout(directory, identity, runtime, 2, pending["sampler"])
        later_update = tiny_update(runtime, optimizer, scheduler, identity, later)
        commit_step(tx, runtime, optimizer, scheduler, 2, later, later_update)
        assert [m["step"] for m in tx.published()] == [0, 1, 2]
        assert full_state(runtime, optimizer, scheduler) == expected
        assert later["sampler"] == sampler
        assert later["records"] == records
        assert later["groups"] == groups
        assert later["sampler"]["actual_calls"] == 256
        # Transaction ledger scopes qids to steps; actual schedule validity belongs to caller.
        assert sampler["consumed_qids"]["1"] == sampler["consumed_qids"]["2"]
        with pytest.raises(PermissionError, match="unique and consecutive"):
            commit_step(tx, runtime, optimizer, scheduler, 2, later, later_update)
        assert len(list((directory / "checkpoints").glob("step-*"))) == 3


@pytest.fixture
def committed(tmp_path):
    runtime, optimizer, scheduler, identity = setup()
    directory = tmp_path / "run"
    with RunTransactions(directory, identity) as tx:
        tx.commit(runtime, optimizer, scheduler, step=0, sampler_state=zero_sampler())
        pending = rollout(directory, identity, runtime, 1, zero_sampler())
        update = tiny_update(runtime, optimizer, scheduler, identity, pending)
        commit_step(tx, runtime, optimizer, scheduler, 1, pending, update)
    return directory, runtime, optimizer, scheduler, identity, pending


def test_same_step_restore_refreshes_hash_rng_adam_and_clears_grad(committed):
    directory, runtime, optimizer, scheduler, identity, _ = committed
    expected = full_state(runtime, optimizer, scheduler)
    with torch.no_grad():
        for parameter in runtime.model.parameters():
            parameter.add_(0.7)
            parameter.grad = torch.ones_like(parameter)
    runtime._srf6_policy_identity = (1, "wrong cached hash")
    runtime.identity["policy_hash"] = "wrong public hash"
    random.random()
    np.random.random()
    torch.rand(3)
    with RunTransactions(directory, identity) as tx:
        receipt = tx.restore_latest(runtime, optimizer, scheduler)
    assert full_state(runtime, optimizer, scheduler) == expected
    assert runtime._srf6_policy_identity == (1, receipt["commit"]["policy_sha256"])
    assert runtime.identity["policy_hash"] == receipt["commit"]["policy_sha256"]
    assert all(p.grad is None for p in runtime.model.parameters())
    assert receipt["commit"]["gate_status"] == "NOT_SIGNED"
    assert receipt["commit"]["training_authorized"] is False


@pytest.mark.parametrize(
    "target", ["state.pt", "UPDATE.json", "COMMIT.json", "rollout", "credit", "evidence"]
)
def test_tampered_highest_state_or_receipt_is_rejected_no_fallback(committed, target):
    directory, runtime, optimizer, scheduler, identity, pending = committed
    path = (
        directory / "checkpoints/step-001" / target
        if "." in target
        else directory / pending["references"][target]["path"]
    )
    path.write_bytes(path.read_bytes() + b" ")
    if target == "COMMIT.json":
        content = json.loads(path.read_bytes())
        content["step"] = 2
        path.write_text(json.dumps(content))
    before = full_state(runtime, optimizer, scheduler)
    with RunTransactions(directory, identity) as tx, pytest.raises(PermissionError):
        tx.restore_latest(runtime, optimizer, scheduler)
    assert full_state(runtime, optimizer, scheduler) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("parent", "b1"),
        ("source_sha256", "f" * 64),
        ("data_sha256", "f" * 64),
        ("phase", "WRONG_PHASE"),
        ("run_id", "another-run"),
        ("reference_sha256", "f" * 64),
        ("lane", "RECOVERY"),
    ],
)
def test_wrong_explicit_run_identity_cannot_reuse_commits(committed, field, value):
    directory, _, _, _, identity, _ = committed
    wrong = {**identity, field: value}
    with (
        pytest.raises(PermissionError, match="identity differs"),
        RunTransactions(directory, wrong),
    ):
        pass


def test_second_lock_and_unlocked_or_inherited_object_are_rejected(committed):
    directory, _, _, _, identity, _ = committed
    first = RunTransactions(directory, identity)
    with pytest.raises(PermissionError, match="lock"):
        first.published()
    with first:
        with pytest.raises(BlockingIOError), RunTransactions(directory, identity):
            pass
        first._pid -= 1
        with pytest.raises(PermissionError, match="lock"):
            first.published()
        first._pid += 1
    with RunTransactions(directory, identity) as second:
        assert len(second.published()) == 2


def test_second_process_cannot_acquire_the_same_run_lock(committed):
    directory, _, _, _, identity, _ = committed
    source_root = Path(transactions.__file__).resolve().parents[1]
    required = [str(source_root), str(source_root.parent)]
    inherited = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(required + [p for p in inherited if p not in required]),
    }
    code = """
import json, sys
from sr_f6.transactions import RunTransactions
try:
    with RunTransactions(sys.argv[1], json.loads(sys.argv[2])):
        raise SystemExit(9)
except BlockingIOError:
    raise SystemExit(0)
"""
    with RunTransactions(directory, identity):
        result = subprocess.run(
            [sys.executable, "-c", code, str(directory), json.dumps(identity)],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env=environment,
        )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("defect", ["data_identity", "optimizer_order"])
def test_restore_rejects_wrong_runtime_data_or_optimizer_layout(committed, defect):
    directory, runtime, optimizer, scheduler, identity, _ = committed
    if defect == "data_identity":
        runtime.identity["data_hash"] = "e" * 64
    else:
        optimizer.param_groups[0]["params"].reverse()
    before = full_state(runtime, optimizer, scheduler)
    with RunTransactions(directory, identity) as tx, pytest.raises(PermissionError):
        tx.restore_latest(runtime, optimizer, scheduler)
    assert full_state(runtime, optimizer, scheduler) == before


@pytest.mark.parametrize("target", ["run", "ancestor", "state", "marker", "evidence", "staging"])
def test_symlink_paths_are_rejected(committed, tmp_path, target):
    directory, runtime, optimizer, scheduler, identity, pending = committed
    if target in ("run", "ancestor"):
        link = tmp_path / "link"
        link.symlink_to(
            directory if target == "run" else directory.parent, target_is_directory=True
        )
        with pytest.raises(PermissionError, match="Symlink"):
            RunTransactions(link if target == "run" else link / directory.name, identity)
        return
    if target == "staging":
        (directory / "checkpoints/.staging-orphan").symlink_to(tmp_path, target_is_directory=True)
    else:
        path = (
            directory / pending["references"]["evidence"]["path"]
            if target == "evidence"
            else directory
            / "checkpoints/step-001"
            / ("state.pt" if target == "state" else "COMMIT.json")
        )
        original = path.with_suffix(".original")
        path.rename(original)
        path.symlink_to(original)
    with RunTransactions(directory, identity) as tx, pytest.raises(PermissionError):
        tx.restore_latest(runtime, optimizer, scheduler)


@pytest.mark.parametrize(
    "bad",
    [
        "partial_G8",
        "duplicate_draw",
        "duplicate_row",
        "fake_count",
        "credit_rows",
        "source",
        "outside_reference",
        "wrong_rng",
        "wrong_consumption",
        "wrong_step",
        "wrong_optimizer_step",
    ],
)
def test_incomplete_or_mismatched_pending_artifacts_never_commit(tmp_path, bad):
    runtime, optimizer, scheduler, identity = setup()
    directory = tmp_path / "run"
    with RunTransactions(directory, identity) as tx:
        tx.commit(runtime, optimizer, scheduler, step=0, sampler_state=zero_sampler())
        pending = rollout(directory, identity, runtime, 1, zero_sampler())
        update = tiny_update(runtime, optimizer, scheduler, identity, pending)
        ref = pending["references"]["rollout"]
        envelope = json.loads((directory / ref["path"]).read_bytes())
        if bad == "partial_G8":
            envelope["records"].pop()
        elif bad == "duplicate_draw":
            envelope["records"][1]["draw"] = 0
        elif bad == "duplicate_row":
            envelope["records"][1]["row_id"] = envelope["records"][0]["row_id"]
        elif bad == "fake_count":
            envelope.pop("records")
            envelope["row_count"] = 128
        elif bad == "source":
            envelope["records"][0]["source_sha256"] = "f" * 64
        elif bad == "credit_rows":
            update["actual_credit_groups"][0]["row_ids"].pop()
        elif bad == "wrong_rng":
            random.random()
        elif bad == "wrong_consumption":
            pending["sampler"]["consumed_qids"]["1"].pop()
        elif bad == "wrong_step":
            runtime.identity["step"] = 1
        elif bad == "wrong_optimizer_step":
            for state in optimizer.state.values():
                state["step"].add_(1)
        else:
            ref["path"] = "../other/rollout.json"
        if bad in ("partial_G8", "duplicate_draw", "duplicate_row", "fake_count", "source"):
            pending["references"]["rollout"] = save_json(directory, ref["path"], envelope)
            pending["sampler"]["rollout_rng_reference"] = pending["references"]["rollout"]
        with pytest.raises((PermissionError, FileNotFoundError)):
            commit_step(tx, runtime, optimizer, scheduler, 1, pending, update)
        assert [m["step"] for m in tx.published()] == [0]


def test_listing_commits_does_not_read_state_or_large_update(committed, monkeypatch):
    from sr_f6 import transactions

    directory, _, _, _, identity, _ = committed
    real_read = transactions._read_bytes

    def read_small_only(path):
        assert path.name not in ("state.pt", "UPDATE.json")
        return real_read(path)

    monkeypatch.setattr(transactions, "_read_bytes", read_small_only)
    monkeypatch.setattr(
        transactions, "_file_hash", lambda path: pytest.fail("listing hashed state")
    )
    with RunTransactions(directory, identity) as tx:
        assert len(tx.published()) == 2


@pytest.mark.parametrize(
    "change",
    [
        "tokens",
        "sampler",
        "prompt",
        "labels",
        "root_id",
        "missing_trace",
        "partial_trace",
        "trace_qid",
        "trace_reference",
    ],
)
def test_updated_rollout_reference_cannot_rebind_a_different_update(tmp_path, change):
    runtime, optimizer, scheduler, identity = setup()
    directory = tmp_path / "run"
    with RunTransactions(directory, identity) as tx:
        tx.commit(runtime, optimizer, scheduler, step=0, sampler_state=zero_sampler())
        pending = rollout(directory, identity, runtime, 1, zero_sampler())
        update = tiny_update(runtime, optimizer, scheduler, identity, pending)
        ref = pending["references"]["rollout"]
        envelope = json.loads((directory / ref["path"]).read_bytes())
        row = envelope["records"][0]
        if change == "tokens":
            row["tokens"][0] += 1
        elif change == "sampler":
            row["sampler_logprobs"][0] -= 0.1
        elif change == "prompt":
            row["model_input"]["prompt"] = "different prompt"
        elif change == "labels":
            row["S"] = 1 - row["S"]
        elif change == "root_id":
            row["root_id"] = "wrong-root"
        elif change == "missing_trace":
            update.pop("probability_vectors")
        elif change == "partial_trace":
            update["probability_vectors"].pop()
        elif change == "trace_qid":
            update["probability_vectors"][0]["qid"] = "different qid"
        else:
            update["probability_vectors"][0]["reference_logprobs"] = [float("nan")] * 3
        row["token_path_sha256"] = trajectory_hash(row)
        pending["references"]["rollout"] = save_json(directory, ref["path"], envelope)
        pending["sampler"]["rollout_rng_reference"] = pending["references"]["rollout"]
        with pytest.raises(PermissionError):
            commit_step(tx, runtime, optimizer, scheduler, 1, pending, update)
        assert [m["step"] for m in tx.published()] == [0]


def test_scored_credit_checksum_is_bound_even_if_reference_is_rehashed(tmp_path):
    runtime, optimizer, scheduler, identity = setup()
    directory = tmp_path / "run"
    with RunTransactions(directory, identity) as tx:
        tx.commit(runtime, optimizer, scheduler, step=0, sampler_state=zero_sampler())
        pending = rollout(directory, identity, runtime, 1, zero_sampler())
        update = tiny_update(runtime, optimizer, scheduler, identity, pending)
        pending["groups"][0]["scored_group_sha256"] = "f" * 64
        ref = pending["references"]["credit"]
        pending["references"]["credit"] = save_json(
            directory, ref["path"], dict(identity=identity, step=1, groups=pending["groups"])
        )
        with pytest.raises(PermissionError, match="Scored credit group checksum"):
            commit_step(tx, runtime, optimizer, scheduler, 1, pending, update)
        assert [m["step"] for m in tx.published()] == [0]


def test_nested_flat_conflict_cannot_be_hidden_by_rehashed_rollout_reference(tmp_path):
    runtime, optimizer, scheduler, identity = setup()
    directory = tmp_path / "run"
    with RunTransactions(directory, identity) as tx:
        tx.commit(runtime, optimizer, scheduler, step=0, sampler_state=zero_sampler())
        pending = rollout(directory, identity, runtime, 1, zero_sampler())
        row = pending["records"][0]
        row["score"] = {key: row[key] for key in ("J", "V", "finished", "S", "B")}
        ref = pending["references"]["rollout"]
        envelope = json.loads((directory / ref["path"]).read_bytes())
        envelope["records"] = pending["records"]
        pending["references"]["rollout"] = save_json(directory, ref["path"], envelope)
        pending["sampler"]["rollout_rng_reference"] = pending["references"]["rollout"]
        update = tiny_update(runtime, optimizer, scheduler, identity, pending)
        row["S"] = 1 - row["S"]  # Nested score and applied credit retain the old value.
        pending["references"]["rollout"] = save_json(directory, ref["path"], envelope)
        pending["sampler"]["rollout_rng_reference"] = pending["references"]["rollout"]
        with pytest.raises(PermissionError, match="Conflicting top-level and nested score"):
            commit_step(tx, runtime, optimizer, scheduler, 1, pending, update)
        assert [m["step"] for m in tx.published()] == [0]


def test_native_weights_only_format_and_missing_published_state_fail_closed(committed):
    directory, runtime, optimizer, scheduler, identity, _ = committed
    path = directory / "checkpoints/step-001/state.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["identity"] == identity
    assert state["step"] == 1
    assert set(state) == {
        "identity",
        "step",
        "policy",
        "optimizer",
        "scheduler",
        "sampler_state",
        "python_rng",
        "numpy_rng",
        "torch_rng",
        "cuda_rng",
    }
    path.unlink()
    with (
        RunTransactions(directory, identity) as tx,
        pytest.raises(PermissionError, match="Partial"),
    ):
        tx.restore_latest(runtime, optimizer, scheduler)


def test_staging_with_complete_marker_is_not_committed_and_max_step_is_explicit(tmp_path):
    runtime, optimizer, scheduler, identity = setup()
    with RunTransactions(tmp_path / "run", identity) as tx:
        assert tx.restore_latest(runtime, optimizer, scheduler) is None

        def fail(location):
            if location == "after_marker":
                raise RuntimeError("fixture failure")

        with pytest.raises(RuntimeError):
            tx.commit(
                runtime, optimizer, scheduler, step=0, sampler_state=zero_sampler(), failpoint=fail
            )
        assert tx.published() == []
        assert tx.restore_latest(runtime, optimizer, scheduler) is None
        tx.commit(runtime, optimizer, scheduler, step=0, sampler_state=zero_sampler())
        assert len(list(tx.checkpoints.glob(".staging-*/COMMIT.json"))) == 1
        with pytest.raises(PermissionError):
            tx.commit(runtime, optimizer, scheduler, step=13, sampler_state=zero_sampler())
    with pytest.raises(PermissionError):
        RunTransactions(tmp_path / "bad", {**identity, "max_steps": 13})
