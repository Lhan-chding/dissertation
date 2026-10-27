"""Synthetic CPU access-boundary tests; mocked admission never certifies a run."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

from mm_core.training import state_hash, trainable_state
from sr_f6 import access, admission, rollouts, runtime, training
from sr_f6.common import digest
from sr_f6.registration import DIAGNOSTIC_ROWS, DIRECTORY
from sr_f6.transactions import RunTransactions


def fixture_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def case(tmp_path, monkeypatch):
    f = fixture_module("test_rollouts")
    case = f.prepare(tmp_path, f.prototype.__wrapped__())
    case["identity"]["diagnostic_row_indices"] = list(DIAGNOSTIC_ROWS)
    registration_path = case["run"] / "ROLLOUT_REGISTRATION.json"
    registration = json.loads(registration_path.read_bytes())
    registration["identity"]["diagnostic_row_indices"] = list(DIAGNOSTIC_ROWS)
    case["ref"]["sha256"] = f.write(registration_path, registration)
    case["identity"]["rollout_registration"] = copy.deepcopy(case["ref"])
    root = case["experiment"]
    folder = root / DIRECTORY / "runs/b0/r0/U11"
    folder.parent.mkdir(parents=True)
    shutil.move(case["run"], folder)
    case["run"] = folder
    r = runtime.Runtime.__new__(runtime.Runtime)
    r.__dict__.update(case["runtime"].__dict__)
    r.torch = torch
    r.identity.update(
        source_hash=r.identity["source_sha256"],
        audit_root=str(folder),
        task_id="CPU-fixture",
        phase_identity_sha256="d" * 64,
        origin_identity_sha256="e" * 64,
        config_sha256=case["identity"]["config_sha256"],
        parent_content_hash="f" * 64,
        diagnostic_row_indices=list(DIAGNOSTIC_ROWS),
    )
    source = dict(
        phase="PILOT_V1",
        identity_sha256="d" * 64,
        origin_identity_sha256="e" * 64,
        deployment="/SYNTHETIC_TEST_NOT_A_DEPLOYMENT",
        identity=dict(
            source_manifest=dict(path="SOURCE.json", sha256=r.identity["source_sha256"]),
            data_manifest=dict(path="DATA/MANIFEST.json", sha256=r.identity["data_hash"]),
        ),
    )
    r._srf6_source_context = source
    r._srf6_execution_mode = "TRAINING"
    r._srf6_execution_identity = {k: r.identity[k] for k in runtime.EXECUTION_IDENTITY_FIELDS}
    runtime._LOADED_RUNTIMES[r] = dict(
        execution=copy.deepcopy(r._srf6_execution_identity),
        source=copy.deepcopy(source),
        model=r.model,
        parent={
            k: r.identity[k]
            for k in (
                "parent_id",
                "reference_parent",
                "reference_hash",
                "parent_content_hash",
                "config_sha256",
            )
        },
    )
    r._srf6_adapter_parameter_names = tuple(dict(r.model.named_parameters()))
    case["runtime"] = r
    ref = {"path": "PHASES/PILOT_V1/ADMISSION.json", "sha256": "8" * 64}
    identity_sha = f.write(folder / "RUN_IDENTITY.json", case["identity"])
    path = dict(
        run_id=r.identity["run_id"],
        audit_root=folder.relative_to(root).as_posix(),
        transaction_identity=dict(path="runs/b0/r0/U11/RUN_IDENTITY.json", sha256=identity_sha),
    )
    grant = dict(
        status="VERIFIED_PILOT_ADMISSION",
        scope="PILOT_V1_REGISTERED_ONLY",
        training_authorized=True,
        generation_authorized=True,
        phase_context=source,
        admission_ref=ref,
        pilot_spec_ref=dict(sha256="9" * 64),
        pilot_spec=dict(paths=[path], diagnostics=dict(row_indices=list(DIAGNOSTIC_ROWS))),
    )
    calls = []

    def fake_admission(*args, **kwargs):
        calls.append(kwargs)
        return copy.deepcopy(grant)

    monkeypatch.setattr(admission, "require_pilot_admission", fake_admission)

    def prepared(self, safe, run_root, *, protocol):
        value = dict(
            inputs={
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
                "mm_token_type_ids": torch.tensor([[0, 0]]),
            },
            routing=dict(input_tensor_hash="d" * 64),
        )
        self._srf5_prepared = value
        return value

    monkeypatch.setattr(runtime.FP32Runtime, "prepare_input", prepared)
    monkeypatch.setattr(runtime.FP32Runtime, "reserve", lambda *args, **kwargs: None)

    def native(self, safe, root, seeds, *, protocol, generation, on_completion):
        self.prepare_input(safe, root, protocol=protocol)
        self.reserve("completion_attempts", len(seeds), group_seed=seeds[0])

        def completed(row):
            row["prompt_token_count"] = 2
            row["prompt"] = safe["prompt"]
            on_completion(row)

        rows = f.FakeRuntime.generate_group(
            self, safe, root, seeds, protocol="structured_v4", on_completion=completed
        )
        for i, row in enumerate(rows):
            self.reserve("generated_tokens", len(row["tokens"]), group_seed=seeds[0], row_index=i)
        return rows

    r._generate_native_group = MethodType(native, r)
    case.update(module=f, grant=grant, admission_ref=ref, calls=calls)
    return case


def opened(case, tx):
    return access.open_training_access(
        case["runtime"],
        tx,
        experiment_root=case["experiment"],
        pilot_spec_sha256="9" * 64,
        admission_ref=case["admission_ref"],
    )


def test_unregistered_runtime_and_missing_admission_never_authorize(case, monkeypatch):
    r = case["runtime"]
    with pytest.raises(PermissionError, match="NOT_AUTHORIZED"):
        r.prepare_input({}, case["experiment"])
    with pytest.raises(PermissionError, match="NOT_AUTHORIZED"):
        training.update(r, None, None, [], [], {}, [])
    with pytest.raises(PermissionError, match="NOT_AUTHORIZED"):
        access.active_session(r)
    with RunTransactions(case["run"], case["identity"]) as tx:
        case["module"].start(tx, case)
        record = runtime._LOADED_RUNTIMES.pop(r)
        with pytest.raises(PermissionError, match="Actual runtime load"), opened(case, tx):
            pass
        runtime._LOADED_RUNTIMES[r] = record
        monkeypatch.setattr(
            admission,
            "require_pilot_admission",
            lambda *a, **k: (_ for _ in ()).throw(
                PermissionError("scientific evidence incomplete")
            ),
        )
        with (
            pytest.raises(PermissionError, match="scientific evidence incomplete"),
            opened(case, tx),
        ):
            pass
    assert r.calls == 0


def test_registered_collect_ready_reuses_originals_and_update_binds_raw(case, monkeypatch):
    r, f = case["runtime"], case["module"]
    with RunTransactions(case["run"], case["identity"]) as tx:
        f.start(tx, case)
        with opened(case, tx) as session:
            result = f.collect(tx, case)
            assert r.calls == 128 and len(result["records"]) == 128
            again = f.collect(tx, case)
            assert r.calls == 128 and again["new_model_calls"] == 0
            assert session._ready_records() == (result["records"], result["credits"])
            seen = []

            def numerical_body(
                runtime, optimizer, scheduler, records, groups, reference, credits, **kwargs
            ):
                row = records[0]
                prepared = runtime.prepare_input(row["model_input"], case["experiment"])
                session.check_records([dict(row, prepared=prepared)])
                session.check_batch(prepared, [row["tokens"]])
                with pytest.raises(PermissionError, match="Completion tokens"):
                    session.check_batch(prepared, [[999999]])
                with pytest.raises(PermissionError, match="Unbound"):
                    session.check_batch(copy.deepcopy(prepared), [row["tokens"]])
                with pytest.raises(PermissionError, match="Direct native"):
                    runtime._native_forward(
                        {}, completion_start=2, lengths=[3], purpose="training_policy", grad=True
                    )
                seen.append(True)
                for i in DIAGNOSTIC_ROWS:
                    runtime.reserve(
                        "diagnostic_gradient_calls", 3, row_id=records[i]["row_id"], step=1
                    )
                return {"status": "SYNTHETIC_ONLY"}

            monkeypatch.setattr(training, "_update", numerical_body)
            ref = trainable_state(r.model)
            changed = copy.deepcopy(result["records"])
            changed[0]["tokens"][0] += 1
            with pytest.raises(PermissionError, match="persisted original"):
                training.update(r, None, None, changed, [8] * 16, ref, result["credits"])
            assert (
                training.update(r, None, None, result["records"], [8] * 16, ref, result["credits"])[
                    "status"
                ]
                == "SYNTHETIC_ONLY"
            )
            with pytest.raises(PermissionError, match="No original-token"):
                training.update(r, None, None, result["records"], [8] * 16, ref, result["credits"])
            assert seen == [True]
        with pytest.raises(PermissionError, match="NOT_AUTHORIZED"):
            r.prepare_input(result["records"][0]["model_input"], case["experiment"])


def test_direct_wrong_seed_panel_prepared_and_identity_downgrade_are_blocked(case):
    r, f = case["runtime"], case["module"]
    with RunTransactions(case["run"], case["identity"]) as tx:
        f.start(tx, case)
        with opened(case, tx) as session:
            g = session.groups[0]
            inspection = session.groups
            inspection[0]["seeds"] = [1] * 8
            inspection[0]["model_input"]["prompt"] = "TEST prompt"
            inspection_spec = session.step_spec
            inspection_spec["step"] = 72
            assert session.groups[0] == g and session.step_spec["step"] == 1
            with pytest.raises(PermissionError, match="registered 16 G8"):
                r.generate_group(g["model_input"], case["experiment"], [1] * 8)
            with pytest.raises(PermissionError, match="registered 16 G8"):
                r.generate_group(
                    dict(g["model_input"], prompt="TEST prompt"), case["experiment"], g["seeds"]
                )
            with pytest.raises(PermissionError, match="live generation"):
                r.prepare_input(g["model_input"], case["experiment"])
            with pytest.raises(PermissionError, match="Direct native generation"):
                r.reserve("completion_attempts", 8, group_seed=g["seeds"][0])
            with pytest.raises(PermissionError, match="Unbound"):
                r._batch_forward(
                    {"inputs": {}, "routing": {}}, [[1]], purpose="training_policy", grad=True
                )
            r.identity["phase"] = "CENSUS_S0"
            with pytest.raises(PermissionError, match="actual loaded source"):
                r.generate_group(g["model_input"], case["experiment"], g["seeds"])
    assert r.calls == 0


def test_interrupted_generation_cannot_be_retried_in_fresh_session(case):
    r, f = case["runtime"], case["module"]
    r.defect = "unknown_call"
    with RunTransactions(case["run"], case["identity"]) as tx:
        f.start(tx, case)
        with opened(case, tx), pytest.raises(RuntimeError, match="interrupted"):
            f.collect(tx, case)
        r.defect = None
        with opened(case, tx) as session:
            g = session.groups[0]
            with pytest.raises(PermissionError, match="already attempted"):
                r.generate_group(g["model_input"], case["experiment"], g["seeds"])
            with pytest.raises(rollouts.IncompleteRollout):
                f.collect(tx, case)
    assert r.calls == 0


@pytest.mark.parametrize("mutation", ["raw", "credit", "ready"])
def test_ready_tamper_rejected_before_update(case, mutation):
    f = case["module"]
    with RunTransactions(case["run"], case["identity"]) as tx:
        f.start(tx, case)
        with opened(case, tx) as session:
            f.collect(tx, case)
            folder = tx.directory / "rollouts/step-001"
            if mutation == "raw":
                path = folder / "groups/00/raw.jsonl"
                path.write_bytes(path.read_bytes().replace(b'"seed":', b'"changed_seed":', 1))
            elif mutation == "credit":
                path = folder / "credit.json"
                obj = json.loads(path.read_bytes())
                obj["groups"][0]["arm"] = "U10"
                path.write_text(json.dumps(obj))
            else:
                path = folder / "READY.json"
                obj = json.loads(path.read_bytes())
                obj["spec"]["step"] = 2
                path.write_text(json.dumps(obj))
            with pytest.raises(PermissionError):
                session._ready_records()


def test_caller_constructed_session_cannot_authorize(case):
    fake = access.PilotSession()
    fake.runtime = case["runtime"]
    with pytest.raises(PermissionError, match="NOT_AUTHORIZED"):
        fake.assert_active()
    assert digest({"fixture": "SYNTHETIC_ONLY"})


def test_wrong_physical_lane_and_released_lock_do_not_authorize(case):
    f = case["module"]
    with RunTransactions(case["run"], case["identity"]) as tx:
        f.start(tx, case)
    with pytest.raises(PermissionError, match="lock"), opened(case, tx):
        pass
    wrong = case["experiment"] / "PHASE_RUNS/PILOT_V1/arbitrary-copy"
    shutil.copytree(case["run"], wrong)
    with (
        RunTransactions(wrong, case["identity"]) as other,
        pytest.raises(PermissionError, match="physical recovery lane"),
        opened(case, other),
    ):
        pass


@pytest.fixture
def evaluation_case(case):
    from sr_f6 import evaluation

    f = fixture_module("test_evaluation")
    root, spec, _ = f.fixture.__wrapped__(case["experiment"], f.prototype.__wrapped__())
    r = case["runtime"]
    policy = r.policy_tensor_identity()
    folder = root / evaluation.DIRECTORY / "b0/parent"
    r.identity.update(
        audit_root=str(folder),
        run_id="CPU-evaluation-fixture",
        execution_mode="EVALUATION",
        policy_hash=policy,
    )
    r._srf6_execution_mode = "EVALUATION"
    r._srf6_execution_identity = {k: r.identity[k] for k in runtime.EXECUTION_IDENTITY_FIELDS}
    runtime._LOADED_RUNTIMES[r]["execution"] = copy.deepcopy(r._srf6_execution_identity)
    runtime._freeze_evaluation(r, 0, policy)
    spec["evaluation"]["endpoints"][0]["tensor_sha256"] = policy
    case["grant"]["pilot_spec"] = spec
    plan = spec["evaluation"]
    public, tasks = evaluation._inputs(root, plan)
    identity = dict(
        execution_origin="SERVER_ACTUAL",
        endpoint_id="b0/parent",
        parent="b0",
        checkpoint_step=0,
        evaluation_run_id=r.identity["run_id"],
        policy_tensor_sha256=policy,
        checkpoint_commit_sha256=None,
        checkpoint_run_id=None,
        phase_identity_sha256="d" * 64,
        pilot_spec_sha256="9" * 64,
        source_sha256=r.identity["source_sha256"],
        data_sha256=r.identity["data_hash"],
        reference_sha256=r.identity["reference_hash"],
        admission_ref=case["admission_ref"],
        input_refs=plan["inputs"],
    )
    ledger = evaluation.EvaluationLedger(root, identity, plan, public, tasks)
    case.update(ledger=ledger, eval_module=evaluation)
    return case


def eval_open(case, ledger=None):
    return access.open_evaluation_access(
        case["runtime"],
        experiment_root=case["experiment"],
        pilot_spec_sha256="9" * 64,
        admission_ref=case["admission_ref"],
        ledger=ledger or case["ledger"],
    )


def test_evaluation_exact_panel_two_draws_no_forwards_or_update(evaluation_case):
    case, r = evaluation_case, evaluation_case["runtime"]
    with case["ledger"] as ledger, eval_open(case) as session:
        item = ledger.schedule[0]
        safe = ledger.public[item["qid"]]["model_input"]
        ledger.begin(item)
        with (
            pytest.raises(PermissionError, match="frozen DEV_EVAL"),
            session.generation(
                safe,
                case["experiment"],
                [1, 2],
                protocol="structured_v4",
                access=access._RUNTIME_ACCESS,
            ),
        ):
            pass
        with session.generation(
            safe,
            case["experiment"],
            item["seeds"],
            protocol="structured_v4",
            access=access._RUNTIME_ACCESS,
        ):
            prepared = r.prepare_input(safe, case["experiment"])
            with pytest.raises(PermissionError, match="teacher forcing"):
                r._batch_forward(prepared, [[1]], purpose="training_policy", grad=True)
            r.reserve("completion_attempts", 2, group_seed=item["seeds"][0])
            for draw in range(2):
                r.reserve("generated_tokens", 1, group_seed=item["seeds"][0], row_index=draw)
        with (
            pytest.raises(PermissionError),
            session.generation(
                safe,
                case["experiment"],
                item["seeds"],
                protocol="structured_v4",
                access=access._RUNTIME_ACCESS,
            ),
        ):
            pass
        with pytest.raises(PermissionError, match="No original-token"):
            training.update(r, None, None, [], [], {}, [])
        assert all(not p.requires_grad and p.grad is None for p in r.model.parameters())
    with pytest.raises(PermissionError, match="NOT_AUTHORIZED"):
        r.prepare_input(safe, case["experiment"])
    with (
        case["ledger"] as ledger,
        eval_open(case),
        pytest.raises(case["eval_module"].IncompleteEvaluation),
    ):
        ledger.begin(item)


@pytest.mark.parametrize(
    "field", ["policy_tensor_sha256", "phase_identity_sha256", "evaluation_run_id"]
)
def test_evaluation_mismatched_identity_rejected_before_panel(evaluation_case, field):
    case = evaluation_case
    case["ledger"].identity[field] = "changed"
    with case["ledger"], pytest.raises(PermissionError, match="loaded identity"), eval_open(case):
        pass


def test_evaluation_cannot_claim_arbitrary_committed_step(evaluation_case):
    case, r = evaluation_case, evaluation_case["runtime"]
    ledger = case["ledger"]
    endpoint = copy.deepcopy(ledger.plan["endpoints"][3])
    endpoint["expected_commit_path"] = (
        DIRECTORY + "/runs/b0/r0/U10/checkpoints/step-012/COMMIT.json"
    )
    case["grant"]["pilot_spec"]["evaluation"]["endpoints"][3] = endpoint
    ledger.plan = copy.deepcopy(case["grant"]["pilot_spec"]["evaluation"])
    ledger.endpoint = endpoint
    ledger.identity.update(
        endpoint_id=endpoint["endpoint_id"],
        checkpoint_step=12,
        checkpoint_run_id=endpoint["run_id"],
        checkpoint_commit_sha256="a" * 64,
    )
    ledger.directory = case["experiment"] / case["eval_module"].DIRECTORY / endpoint["endpoint_id"]
    r.identity.update(step=12, audit_root=str(ledger.directory))
    runtime._LOADED_RUNTIMES[r]["execution"]["audit_root"] = str(ledger.directory)
    runtime._freeze_evaluation(r, 12, r.identity["policy_hash"])
    with ledger, pytest.raises(PermissionError, match="Actually loaded committed"), eval_open(case):
        pass


def test_deleting_private_source_does_not_downgrade_actual_loaded_runtime(case):
    r = case["runtime"]
    del r._srf6_source_context
    del r._srf6_execution_identity
    r.identity["phase"] = "CENSUS_S0"
    for operation in (
        lambda: r.prepare_input({}, case["experiment"]),
        lambda: r.generate_group({}, case["experiment"], list(range(8))),
        lambda: training.update(r, None, None, [], [], {}, []),
    ):
        with pytest.raises(PermissionError, match="cannot downgrade"):
            operation()


def numerical_native_fixture(case, monkeypatch, *, fail=False):
    """Only replace unavailable VL model math, retaining all Runtime dispatch."""
    r = case["runtime"]
    baseline = sum(p.detach().sum() for p in r.model.parameters()).clone()
    r.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    r.eos_ids = [0]
    accounts, forwards = [], []
    monkeypatch.setattr(
        runtime.FP32Runtime,
        "reserve",
        lambda self, kind, count, **meta: accounts.append((kind, count, meta)),
    )
    monkeypatch.setattr(runtime.FP32Runtime, "_validate_record_revision", lambda *args: None)
    monkeypatch.setattr(runtime.FP32Runtime, "_validate_prepared_revision", lambda *args: None)

    def actual_tiny_forward(self, inputs, *, completion_start, lengths, purpose, grad):
        self.reserve(
            "extra_forward_sequences", len(lengths), purpose=purpose, completion_tokens=sum(lengths)
        )
        forwards.append((purpose, grad, state_hash(trainable_state(self.model))))
        if fail and purpose == "training_policy":
            raise RuntimeError("SYNTHETIC native forward interruption")
        value = sum(p.sum() for p in self.model.parameters()) - baseline
        return [
            -torch.arange(1, n + 1, dtype=torch.float32)
            + value * inputs["input_ids"][i, completion_start].float() / 100
            for i, n in enumerate(lengths)
        ]

    monkeypatch.setattr(runtime.FP32Runtime, "_native_forward", actual_tiny_forward)
    return accounts, forwards


def test_real_training_body_access_diagnostics_reference_restore_and_commit(case, monkeypatch):
    r, f = case["runtime"], case["module"]
    accounts, forwards = numerical_native_fixture(case, monkeypatch)
    reference = trainable_state(r.model)
    initial = state_hash(reference)
    versions_before = [p._version for p in r.model.parameters()]
    with RunTransactions(case["run"], case["identity"]) as tx:
        f.start(tx, case)
        with opened(case, tx) as session:
            ready = f.collect(tx, case)
            result = training.update(
                r,
                case["optimizer"],
                case["scheduler"],
                ready["records"],
                [8] * 16,
                reference,
                ready["credits"],
                diagnostic_row_indices=DIAGNOSTIC_ROWS,
            )
            assert result["status"] == "UPDATED_UNCOMMITTED" and r.identity["step"] == 0
            assert result["optimizer_updates"] == 1 and case["scheduler"].last_epoch == 1
            assert all(int(state["step"]) == 1 for state in case["optimizer"].state.values())
            assert len(forwards) == 256
            assert forwards[::2] == [("training_reference", False, initial)] * 128
            assert forwards[1::2] == [("training_policy", True, initial)] * 128
            assert state_hash(reference) == initial
            assert all(
                p._version > old
                for p, old in zip(r.model.parameters(), versions_before, strict=True)
            )
            assert result["policy_after_sha256"] != initial
            diagnostics = [event for event in accounts if event[0] == "diagnostic_gradient_calls"]
            assert [(e[1], e[2]["row_id"], e[2]["step"]) for e in diagnostics] == [
                (3, ready["records"][i]["row_id"], 1) for i in DIAGNOSTIC_ROWS
            ]
            assert session._diagnostic_seen == session._diagnostic_row_ids
            with pytest.raises(PermissionError, match="No live generation"):
                group = session.groups[0]
                r.generate_group(group["model_input"], case["experiment"], group["seeds"])
            event = json.loads(
                (tx.directory / "NUMERICAL_TOKEN_AUDIT.jsonl").read_text().splitlines()[-1]
            )
            path = "rollouts/step-001/SYNTHETIC_PREUPDATE.json"
            ref = dict(path=path, sha256=f.write(tx.directory / path, event))
            marker = tx.commit(
                r,
                case["optimizer"],
                case["scheduler"],
                step=1,
                sampler_state=ready["sampler_state"],
                update=result,
                references={**ready["references"], "evidence": ref},
            )
            assert marker["step"] == r.identity["step"] == 1
            with pytest.raises(PermissionError, match="committed step changed"):
                session.assert_active()
        restored = tx.restore_latest(r, case["optimizer"], case["scheduler"])
        assert restored["step"] == 1 and r.identity["policy_hash"] == result["policy_after_sha256"]


def test_real_update_failure_closes_access_and_cannot_generate_again(case, monkeypatch):
    r, f = case["runtime"], case["module"]
    numerical_native_fixture(case, monkeypatch, fail=True)
    original = trainable_state(r.model)
    with RunTransactions(case["run"], case["identity"]) as tx:
        f.start(tx, case)
        with opened(case, tx) as session:
            ready = f.collect(tx, case)
            with pytest.raises(RuntimeError, match="native forward interruption"):
                training.update(
                    r,
                    case["optimizer"],
                    case["scheduler"],
                    ready["records"],
                    [8] * 16,
                    original,
                    ready["credits"],
                    diagnostic_row_indices=DIAGNOSTIC_ROWS,
                )
            assert state_hash(trainable_state(r.model)) == state_hash(original)
            assert not case["optimizer"].state and case["scheduler"].last_epoch == 0
            assert r._srf6_access_poisoned
            with pytest.raises(PermissionError, match="requires reload"):
                group = session.groups[0]
                r.generate_group(group["model_input"], case["experiment"], group["seeds"])
        assert [m["step"] for m in tx.published()] == [0]
