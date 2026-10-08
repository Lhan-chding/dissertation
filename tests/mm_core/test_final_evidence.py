"""CPU-only regressions for the independent final-evidence verifier."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mm_core import execution, release, training, vl_runtime

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/mm_core/verify_final_evidence.py"
spec = importlib.util.spec_from_file_location("final_evidence_helper", SCRIPT)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
MODULES = dict(execution=execution, release=release, training=training, vl_runtime=vl_runtime)


def write(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def jsonl(root, relative, rows):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def cost(kind, identity):
    return dict(kind=kind, amount=1, identity=identity, status="CONSUMED_OR_RESERVED")


def checkpoint(context, names, stop):
    _, rng_states = verifier.expected_engine_requests(context, vl_runtime, "continuous")
    params = {name: torch.tensor([[0.5]], dtype=torch.float32) for name in names}
    moments = {
        i: dict(
            step=torch.tensor(float(stop)),
            exp_avg=torch.zeros((1, 1)),
            exp_avg_sq=torch.zeros((1, 1)),
        )
        for i in range(len(names))
    }
    return dict(
        committed_logical_step=stop,
        parameters=params,
        optimizer=dict(
            state=moments,
            param_groups=[
                dict(
                    params=list(range(len(names))),
                    lr=1e-5,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0.0,
                    amsgrad=False,
                    maximize=False,
                )
            ],
        ),
        scheduler=dict(last_epoch=stop, _step_count=stop + 1, base_lrs=[1e-5], _last_lr=[1e-5]),
        reference=dict(parameters=copy.deepcopy(params), base_hash="a" * 64),
        rng=dict(
            python=rng_states[stop],
            numpy=np.random.RandomState(4).get_state(),
            cpu=torch.tensor([3, 4], dtype=torch.uint8),
            cuda=[torch.tensor([1, 2], dtype=torch.uint8)],
        ),
        input_stream_hash=context["engine"]["stream_hash"],
    )


@pytest.fixture
def engine_run(tmp_path):
    root = tmp_path
    stream = [
        dict(
            question_id=f"q{i}",
            split="ENGINE_TEST",
            image_sha256=f"image{i}",
            processed_pixel_sha256=f"pixel{i}",
            true_values=[12, 34],
            operation="sum",
        )
        for i in range(8)
    ]
    context = dict(
        freeze=dict(processor_hash="processor", chat_template_hash="chat"),
        common=dict(model_hash="common"),
        engine=dict(seed=42, stream_hash=vl_runtime.hash_json([q["question_id"] for q in stream])),
        stream=stream,
        prefix_ledger=[],
        prefix_requests=[],
    )
    write(root, "manifests/PRE_INFERENCE_FREEZE.json", context["freeze"])
    write(root, "manifests/ENGINE_TEST_FREEZE.json", context["engine"])
    write(root, "engineering/ENGINE_STARTED.json", dict(pid=100, status="RUNNING"))
    modules = sorted(
        f"model.language_model.layers.{i}.self_attn.{kind}"
        for i in range(3, 32, 4)
        for kind in ("q_proj", "v_proj")
    )
    names = [
        f"base_model.model.{module}.lora_{kind}.default.weight"
        for module in modules
        for kind in ("A", "B")
    ]
    requests, ledger = [], []
    for branch in ("continuous", "resumed"):
        expected, _ = verifier.expected_engine_requests(context, vl_runtime, branch)
        requests.extend(dict(r, status="REQUESTED") for r in expected)
        raw = []
        for request in expected:
            i = int(request["question_id"][1:])
            raw.append(
                dict(
                    request,
                    raw_text='{"answer":46}',
                    tokens=[1, 2],
                    raw_tokens=[1, 2],
                    completion_token_count=2,
                    image_routing=dict(
                        source_image_sha256=f"image{i}",
                        processed_pixel_sha256=f"pixel{i}",
                        processor_hash="processor",
                        chat_template_hash="chat",
                        chat_template_kwargs=vl_runtime.CHAT_TEMPLATE_KWARGS,
                        chat_template_kwargs_hash=vl_runtime.hash_json(
                            vl_runtime.CHAT_TEMPLATE_KWARGS
                        ),
                        image_token_count=588,
                        generation_vision_forward_calls=1,
                    ),
                )
            )
            ledger.append(
                cost(
                    "completion_attempts",
                    dict(question_id=request["question_id"], seed=request["seed"]),
                )
            )
            for purpose in ("engine_behavior_logprob", "engine_policy_gradient"):
                ledger.append(
                    cost("extra_forward_sequences", dict(purpose=purpose, completion_tokens=2))
                )
        jsonl(root, f"engineering/engine/{branch}/RAW_COMPLETIONS.jsonl", raw)
        jsonl(
            root,
            f"engineering/engine/{branch}/ROLLOUTS.jsonl",
            [dict(row, reward=1) for row in raw],
        )
        steps = []
        for step in range(4):
            group = raw[step * 16 : (step + 1) * 16]
            steps.append(
                dict(
                    logical_step=step + 1,
                    question_ids=[r["question_id"] for r in group],
                    rewards=[1] * 16,
                    zero_contrast_groups=[True, True],
                    sequence_losses=[0.0] * 16,
                    loss=0.0,
                    gradient_norm=0.0,
                )
            )
            ledger.append(
                cost(
                    "physical_optimizer_updates",
                    dict(stage="ENGINE", branch=branch, logical_step=step + 1),
                )
            )
        jsonl(root, f"engineering/engine/{branch}/STEPS.jsonl", steps)
    for index, (branch, start, stop) in enumerate(verifier.SEGMENTS):
        folder = f"engineering/engine/{branch}"
        state = checkpoint(context, names, stop)
        torch.save(state, root / folder / f"checkpoint_{stop}.pt")
        stem = f"{folder}/SEGMENT_{start}_{stop}"
        write(
            root,
            stem + "_STARTED.json",
            dict(
                pid=101 + index,
                start=start,
                stop=stop,
                freeze_hash=verifier.sha256(root / "manifests/ENGINE_TEST_FREEZE.json"),
            ),
        )
        write(
            root,
            stem + "_IDENTITY.json",
            dict(
                reference_hash=training.state_hash(state["reference"]),
                stream_hash=context["engine"]["stream_hash"],
                target_modules=modules,
                trainable_names=names,
                trainable_dtype="float32",
                model_forward_mode="eval_dropout_zero_grad_enabled",
            ),
        )
        write(
            root,
            stem + "_COMPLETE.json",
            dict(
                status="COMPLETED",
                physical_updates=stop - start,
                final_state_hash=training.state_hash(state),
                reference_unchanged=True,
                frozen_base_unchanged=True,
                image_calls=200,
            ),
        )
    write(
        root,
        "engineering/ENGINE_COMPARE.json",
        dict(
            status="PASS",
            executed=True,
            comparisons={
                **dict.fromkeys(verifier.CHECKPOINT_KEYS, True),
                "raw_tokens_slots_seeds_rewards": True,
                "rollout_count": True,
                "image_calls": True,
            },
            physical_updates=8,
            rollouts=128,
            exact_comparison=True,
            tolerance=None,
            engine_model_replaces_common_start=False,
            scientific_response_evaluation=False,
            recipe=training.ENGINE_RECIPE,
        ),
    )
    jsonl(root, "raw/REQUESTS.jsonl", requests)
    terminal = dict(
        job_id="123",
        state="COMPLETED",
        squeue_absent=True,
        elapsed_seconds=100,
        gpus=1,
        owner="owner",
        squeue_stdout="",
        command=[
            "sacct",
            "-X",
            "-j",
            "123",
            "-o",
            "JobIDRaw,State,ElapsedRaw,Start,End,AllocTRES%200,User,JobName%100",
        ],
        sacct_stdout="123|COMPLETED|100|2026-10-09T00:00:00|2026-10-09T00:01:40|"
        "gres/gpu:pro6000=1,gres/gpu=1,node=1|owner|mmcore-engine|\n",
    )
    write(root, "accounting/scheduler/123.json", terminal)
    allocation_identity = dict(allocation_key="engine", stage="ENGINE", seconds=3600, gpus=1)
    write(
        root,
        "accounting/allocations/engine.json",
        dict(
            allocation_identity,
            pre_freeze_hash=verifier.sha256(root / "manifests/PRE_INFERENCE_FREEZE.json"),
            status="TERMINAL_VERIFIED",
            job_id="123",
            terminal=terminal,
            terminal_receipt_sha256=verifier.sha256(root / "accounting/scheduler/123.json"),
        ),
    )
    ledger.append(cost("allocated_gpu_hours", allocation_identity))
    jsonl(root, "accounting/COST_LEDGER.jsonl", ledger)
    return root, context


def test_complete_cpu_evidence_verifies_actual_checkpoints_and_all_rollouts(engine_run):
    root, context = engine_run
    assert verifier.verify_rollouts(root, context, MODULES)["rollouts"] == 128
    checked = verifier.verify_segments(root, context, MODULES)
    assert all(checked["comparisons"].values())
    assert checked["recorded_child_pids"] == [101, 102, 103]
    accounting = verifier.verify_accounting(root, context)
    assert accounting["status"] == "PASS"
    assert accounting["reservations"]["completion_attempts"] == 128
    assert accounting["reservations"]["extra_forward_sequences"] == 256
    assert accounting["reservations"]["physical_optimizer_updates"] == 8


def test_pass_receipt_cannot_hide_checkpoint_mutation(engine_run):
    root, context = engine_run
    path = root / "engineering/engine/resumed/checkpoint_4.pt"
    state = torch.load(path, weights_only=False, map_location="cpu")
    state["parameters"][next(iter(state["parameters"]))].add_(1)
    torch.save(state, path)
    with pytest.raises(ValueError, match="Actual checkpoint state"):
        verifier.verify_segments(root, context, MODULES)


def test_self_consistent_changed_checkpoint_still_fails_continuous_comparison(engine_run):
    root, context = engine_run
    path = root / "engineering/engine/resumed/checkpoint_4.pt"
    state = torch.load(path, weights_only=False, map_location="cpu")
    state["parameters"][next(iter(state["parameters"]))].add_(1)
    torch.save(state, path)
    relative = "engineering/engine/resumed/SEGMENT_2_4_COMPLETE.json"
    complete = verifier.read_json(root, relative)
    complete["final_state_hash"] = training.state_hash(state)
    write(root, relative, complete)
    with pytest.raises(ValueError, match="continuous/resumed checkpoint fields"):
        verifier.verify_segments(root, context, MODULES)


@pytest.mark.parametrize("field,value", [("pid", 101), ("freeze_hash", "changed")])
def test_reused_pid_or_changed_freeze_rejected(engine_run, field, value):
    root, context = engine_run
    relative = "engineering/engine/resumed/SEGMENT_2_4_STARTED.json"
    started = verifier.read_json(root, relative)
    started[field] = value
    write(root, relative, started)
    with pytest.raises(ValueError):
        verifier.verify_segments(root, context, MODULES)


@pytest.mark.parametrize("field,value", [("reward", 0), ("seed", 123), ("tokens", [999])])
def test_rollout_reward_seed_and_tokens_recomputed(engine_run, field, value):
    root, context = engine_run
    relative = "engineering/engine/resumed/ROLLOUTS.jsonl"
    rows = verifier.read_jsonl(root, relative)
    rows[0][field] = value
    jsonl(root, relative, rows)
    with pytest.raises(ValueError):
        verifier.verify_rollouts(root, context, MODULES)


def test_both_branches_agreeing_on_wrong_seed_do_not_pass(engine_run):
    root, context = engine_run
    for branch in ("continuous", "resumed"):
        for name in ("RAW_COMPLETIONS", "ROLLOUTS"):
            relative = f"engineering/engine/{branch}/{name}.jsonl"
            rows = verifier.read_jsonl(root, relative)
            rows[0]["seed"] = 123
            jsonl(root, relative, rows)
    with pytest.raises(ValueError, match="raw request/seed/stream"):
        verifier.verify_rollouts(root, context, MODULES)


def test_partial_engine_explicitly_incomplete(engine_run):
    root, context = engine_run
    relative = "engineering/engine/resumed/ROLLOUTS.jsonl"
    rows = verifier.read_jsonl(root, relative)
    jsonl(root, relative, rows[:-1])
    with pytest.raises(verifier.MissingEvidence, match="63 rollouts"):
        verifier.verify_rollouts(root, context, MODULES)


def test_unknown_original_attempt_and_failed_forward_are_never_refunded(engine_run):
    root, context = engine_run
    old_request = dict(
        stage="FORMAT_BASE_TEST", request_id="original-unknown", question_id="old-q", seed=7
    )
    old_costs = [
        cost("completion_attempts", dict(question_id="old-q", seed=7)),
        cost("extra_forward_sequences", dict(purpose="generated_tokens_actual_self_prefix")),
    ]
    context["prefix_requests"] = [old_request]
    context["prefix_ledger"] = old_costs
    jsonl(
        root, "raw/REQUESTS.jsonl", [old_request, *verifier.read_jsonl(root, "raw/REQUESTS.jsonl")]
    )
    jsonl(
        root,
        "accounting/COST_LEDGER.jsonl",
        [*old_costs, *verifier.read_jsonl(root, "accounting/COST_LEDGER.jsonl")],
    )
    checked = verifier.verify_accounting(root, context)
    assert checked["status"] == "PASS"
    assert checked["reservations"]["completion_attempts"] == 129
    assert checked["saved_completions"] == 128
    assert checked["reservations"]["extra_forward_sequences"] == 257
    assert (
        checked["charged_attempts_without_saved_completion"][0]["request_id"] == "original-unknown"
    )
    assert checked["charged_field_forwards_without_completed_sidecar"] == {
        "generated_tokens_actual_self_prefix": 1
    }
    ledger = verifier.read_jsonl(root, "accounting/COST_LEDGER.jsonl")
    jsonl(root, "accounting/COST_LEDGER.jsonl", ledger[1:])
    with pytest.raises(ValueError, match="Completion ledger"):
        verifier.verify_accounting(root, context)


def test_prefix_hash_cannot_be_replaced_by_later_ledger(tmp_path):
    data = b'{"kind":"old"}\n'
    (tmp_path / "ledger.jsonl").write_bytes(data + b'{"kind":"new"}\n')
    desc = dict(
        path="ledger.jsonl", bytes=len(data), rows=1, sha256=hashlib.sha256(data).hexdigest()
    )
    assert verifier.verify_prefix(tmp_path, desc, "ledger.jsonl") == [{"kind": "old"}]
    (tmp_path / "ledger.jsonl").write_bytes(b'{"kind":"bad"}\n')
    with pytest.raises(ValueError, match="prefix changed"):
        verifier.verify_prefix(tmp_path, desc, "ledger.jsonl")


def test_inventory_covers_all_scopes_and_hashes_binaries_without_self_reference(tmp_path):
    for directory in verifier.DIRECTORIES:
        write(tmp_path, f"{directory}/evidence.json", {"directory": directory})
    (tmp_path / "engineering/checkpoint.pt").write_bytes(b"opaque checkpoint bytes")
    write(tmp_path, verifier.REPORT, {"old": "report"})
    write(tmp_path, verifier.MANIFEST, {"old": "manifest"})
    entries = verifier.inventory(tmp_path)
    assert len(entries) == len(verifier.DIRECTORIES) + 1
    assert not {verifier.REPORT, verifier.MANIFEST} & {r["path"] for r in entries}
    assert next(r for r in entries if r["path"].endswith(".pt"))["content"] == "hash_only"


def test_external_checkpoint_and_directory_symlinks_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.write_bytes(b"not loaded")
    root = tmp_path / "run"
    (root / "engineering").mkdir(parents=True)
    (root / "engineering/checkpoint.pt").symlink_to(outside)
    with pytest.raises(ValueError, match="Symlink"):
        verifier.inventory(root)
    (root / "engineering/checkpoint.pt").unlink()
    (root / "raw").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        verifier.inventory(root)


def test_wrong_frozen_source_rejected_before_import(tmp_path):
    source = tmp_path / "mm_core"
    source.mkdir()
    (source / "__init__.py").write_text('raise RuntimeError("must not execute")\n')
    with pytest.raises(ValueError, match="Bound file changed"):
        verifier.frozen_modules(source, {"source_hashes": {"__init__.py": "wrong"}})


def test_fake_pass_without_prerequisite_evidence_is_incomplete(tmp_path):
    write(tmp_path, "engineering/ENGINE_COMPARE.json", dict(status="PASS"))
    before = verifier.inventory(tmp_path)
    report, manifest = verifier.audit(tmp_path, tmp_path / "missing-src")
    assert report["status"] == "INCOMPLETE"
    assert manifest["stable_snapshot"] is True
    assert report["new_model_calls"] == report["new_training_updates"] == 0
    assert verifier.inventory(tmp_path) == before
    assert not (tmp_path / verifier.REPORT).exists()


def test_unstable_snapshot_cannot_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(
        verifier, "inventory", lambda root: [{"path": str(len(list(root.iterdir())))}]
    )

    def alter(root, relative):
        (root / "new-file").write_text("writer was active")
        raise FileNotFoundError(relative)

    monkeypatch.setattr(verifier, "read_json", alter)
    report, _ = verifier.audit(tmp_path, tmp_path)
    assert report["status"] == "INCOMPLETE"
    assert any(i["check"] == "stable_snapshot" for i in report["issues"])


def identity_fixture(root, context, monkeypatch):
    monkeypatch.setattr(execution, "_verify_freeze", lambda root: context["freeze"])
    monkeypatch.setattr(execution, "verify_common_start", lambda root: context["common"])
    monkeypatch.setattr(execution, "verify_format_report", lambda *args: {"gate_passed": True})
    for name in verifier.REQUIRED_BINDINGS - {
        "manifests/PRE_INFERENCE_FREEZE.json",
        "data/questions.jsonl",
    }:
        write(root, name, {"registered": name})
    jsonl(root, "data/questions.jsonl", context["stream"])
    stream = training.deterministic_order(context["stream"], 42, 8)
    ids = [q["question_id"] for q in stream]
    engine = dict(
        status="FROZEN",
        matching_prior_receipt=False,
        seed=42,
        recipe=training.ENGINE_RECIPE,
        pre_freeze_hash=verifier.sha256(root / "manifests/PRE_INFERENCE_FREEZE.json"),
        bound_files={name: verifier.sha256(root / name) for name in verifier.REQUIRED_BINDINGS},
        stream_question_ids=ids,
        stream_hash=vl_runtime.hash_json(ids),
    )
    write(root, "manifests/ENGINE_TEST_FREEZE.json", engine)
    name = "tables/MEASUREMENT_AUDIT/SCORING_RECEIPT.json"
    write(root, name, {"status": "SCORED"})
    empty = dict(bytes=0, rows=0, sha256=hashlib.sha256(b"").hexdigest())
    submission = dict(
        engine_freeze_hash=verifier.sha256(root / "manifests/ENGINE_TEST_FREEZE.json"),
        bound_files={name: verifier.sha256(root / name)},
        pre_engine_ledger_prefix=dict(empty, path="accounting/COST_LEDGER.jsonl"),
        pre_engine_requests_prefix=dict(empty, path="raw/REQUESTS.jsonl"),
    )
    write(root, "manifests/ENGINE_PRE_SUBMISSION.json", submission)
    return engine, submission


def test_freeze_and_presubmission_bindings_are_actually_checked(engine_run, monkeypatch):
    root, context = engine_run
    identity_fixture(root, context, monkeypatch)
    checked = verifier.verify_identity(root, MODULES)
    assert checked["engine"]["stream_question_ids"] == [q["question_id"] for q in checked["stream"]]
    write(root, "manifests/COMMON_START.json", {"changed": True})
    with pytest.raises(ValueError, match="Bound file changed"):
        verifier.verify_identity(root, MODULES)


def test_wrong_question_stream_cannot_be_self_consistently_refrozen(engine_run, monkeypatch):
    root, context = engine_run
    engine, submission = identity_fixture(root, context, monkeypatch)
    engine["stream_question_ids"].reverse()
    engine["stream_hash"] = vl_runtime.hash_json(engine["stream_question_ids"])
    write(root, "manifests/ENGINE_TEST_FREEZE.json", engine)
    submission["engine_freeze_hash"] = verifier.sha256(root / "manifests/ENGINE_TEST_FREEZE.json")
    write(root, "manifests/ENGINE_PRE_SUBMISSION.json", submission)
    with pytest.raises(ValueError, match="input stream differs"):
        verifier.verify_identity(root, MODULES)


def test_changed_presubmission_scoring_receipt_fails(engine_run, monkeypatch):
    root, context = engine_run
    identity_fixture(root, context, monkeypatch)
    write(
        root, "tables/MEASUREMENT_AUDIT/SCORING_RECEIPT.json", {"status": "SCORED", "changed": True}
    )
    with pytest.raises(ValueError, match="Bound file changed"):
        verifier.verify_identity(root, MODULES)


@pytest.mark.parametrize("state", [None, "RUNNING", "PENDING"])
def test_nonterminal_scheduler_status_cannot_certify_gpu_time(engine_run, state):
    root, context = engine_run
    relative = "accounting/allocations/engine.json"
    record = verifier.read_json(root, relative)
    record["terminal"]["state"] = state
    write(root, relative, record)
    with pytest.raises(ValueError, match="terminal state/resource"):
        verifier.verify_accounting(root, context)


@pytest.mark.parametrize("gpus", [0, 6, 1.0, True])
def test_invalid_registered_gpu_counts_rejected(engine_run, gpus):
    root, context = engine_run
    relative = "accounting/allocations/engine.json"
    record = verifier.read_json(root, relative)
    record["gpus"] = gpus
    write(root, relative, record)
    with pytest.raises(ValueError, match="allocation resources"):
        verifier.verify_accounting(root, context)


def test_requeued_scheduler_epochs_preserve_preemption_and_pending_cancel(engine_run):
    root, _ = engine_run
    record = verifier.read_json(root, "accounting/allocations/engine.json")
    terminal = record["terminal"]
    terminal.update(
        state="CANCELLED",
        elapsed_seconds=271,
        command=[
            "sacct",
            "-D",
            "-X",
            "-nP",
            "-j",
            "123",
            "--format=JobIDRaw,User,JobName%100,State,ElapsedRaw,Start,End,AllocTRES%200",
        ],
        epochs=[
            [
                "123",
                "owner",
                "mmcore-engine",
                "PREEMPTED",
                "271",
                "2026-10-09T00:00:00",
                "2026-10-09T00:04:31",
                "gres/gpu:pro6000=1,gres/gpu=1,node=1",
            ],
            [
                "123",
                "owner",
                "mmcore-engine",
                "CANCELLED by 10",
                "0",
                "None",
                "2026-10-09T00:05:00",
                "",
            ],
        ],
    )
    terminal["sacct_stdout"] = "".join("|".join(row) + "\n" for row in terminal["epochs"])
    write(root, "accounting/scheduler/123.json", terminal)
    record["terminal_receipt_sha256"] = verifier.sha256(root / "accounting/scheduler/123.json")
    checked = verifier.verify_terminal(root, record)
    assert checked["elapsed_seconds"] == 271
    assert checked["epoch_count"] == 2
    assert checked["gpu_hours"] == 271 / 3600
    terminal["elapsed_seconds"] = 0
    write(root, "accounting/scheduler/123.json", terminal)
    record["terminal_receipt_sha256"] = verifier.sha256(root / "accounting/scheduler/123.json")
    with pytest.raises(ValueError, match="aggregate elapsed time"):
        verifier.verify_terminal(root, record)
