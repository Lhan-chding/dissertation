"""Actual torch/Adam/RNG transactions with synthetic CPU data, never GPU evidence."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mm_core.training import state_hash, trainable_state
from sr_f6 import recovery, registration
from sr_f6.transactions import RunTransactions


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


_t = module("recovery_transaction_fixture", "test_transactions.py")
_r = module("recovery_registration_fixture", "test_registration.py")
factory, prototype = _r.factory, _r.prototype


def runtime_files(case, runtime, task, directory):
    runtime.identity.update(
        run_root=str(case.root),
        audit_root=str(directory),
        task_id=task,
        phase="PILOT_V1",
        execution_mode="TRAINING",
        source_sha256=case.identity["source_sha256"],
        data_hash=case.identity["data_sha256"],
        run_id=case.identity["run_id"],
    )
    for key in ("phase_identity_sha256", "origin_identity_sha256", "config_sha256"):
        if key in case.identity:
            runtime.identity[key] = case.identity[key]
    _t.save_json(directory, f"runtime_identity/{task}.json", runtime.identity)
    return runtime


def accounting(runtime, records):
    path = (
        Path(runtime.identity["audit_root"])
        / "accounting"
        / (runtime.identity["task_id"] + ".jsonl")
    )
    path.parent.mkdir(exist_ok=True)
    with path.open("a") as out:
        for offset in range(0, 128, 8):
            group = records[offset : offset + 8]
            seed = group[0].get("group_seed", group[0]["seed"])
            events = [("completion_attempts", 8, {"group_seed": seed, "protocol": "structured"})]
            events += [
                ("generated_tokens", len(row["tokens"]), {"group_seed": seed, "row_index": i})
                for i, row in enumerate(group)
            ]
            for kind, count, metadata in events:
                out.write(
                    json.dumps(
                        dict(
                            task_id=runtime.identity["task_id"],
                            parent="b0",
                            kind=kind,
                            count=count,
                            metadata=metadata,
                        )
                    )
                    + "\n"
                )


def step(tx, runtime, optimizer, scheduler, number, *, mutate=None):
    cached = _t.rollout(
        tx.directory, tx.identity, runtime, number, tx.published()[-1]["sampler_state"]
    )
    for row in cached["records"]:
        row.update(
            task_id=runtime.identity["task_id"],
            generation_seconds=1.3,
            full_text='{"time":"actual model content, not execution metadata"}',
            finish_reason="balanced",
        )
    if mutate:
        mutate(cached)
    envelope = json.loads((tx.directory / cached["references"]["rollout"]["path"]).read_bytes())
    envelope["records"] = cached["records"]
    cached["references"]["rollout"] = _t.save_json(
        tx.directory, cached["references"]["rollout"]["path"], envelope
    )
    cached["sampler"]["rollout_rng_reference"] = cached["references"]["rollout"]
    accounting(runtime, cached["records"])
    update = _t.tiny_update(runtime, optimizer, scheduler, tx.identity, cached)
    marker = _t.commit_step(tx, runtime, optimizer, scheduler, number, cached, update)
    return marker, dict(
        references=cached["references"], new_model_calls=128, reused_complete_groups=0
    )


@pytest.fixture
def case(factory):
    data = factory()
    runtime, opt, scheduler, _ = _t.setup()
    value = json.loads((data.root / data.refs["b0"]["path"]).read_bytes())
    value["reference_hash"] = value["policy_hash"] = state_hash(trainable_state(runtime.model))
    _r.write(data.root / data.refs["b0"]["path"], value)
    data.refs["b0"]["sha256"] = recovery._file_hash(data.root / data.refs["b0"]["path"])
    frozen = registration.freeze_pilot_registration(
        data.root, phase_context=data.context, parent_references=data.refs
    )
    spec_hash = frozen["spec_sha256"]
    _, primary, identity = recovery._registration(data.root, data.context, spec_hash)
    result = SimpleNamespace(
        root=data.root,
        context=data.context,
        spec_hash=spec_hash,
        primary=primary,
        identity=identity,
        runtime=runtime,
        optimizer=opt,
        scheduler=scheduler,
    )
    runtime_files(result, runtime, "primary-continuous", primary)
    return result


def primary(case, *, resumed=False, stop=12):
    tx = RunTransactions(case.primary, case.identity)
    with tx:
        tx.commit(
            case.runtime, case.optimizer, case.scheduler, step=0, sampler_state=_t.zero_sampler()
        )
        with recovery.ExecutionJournal.synthetic(
            tx, case.runtime, experiment_root=case.root, resumed_after_data=resumed
        ) as journal:
            for number in range(1, stop + 1):
                journal.record_commit(
                    *step(tx, case.runtime, case.optimizer, case.scheduler, number)
                )
            if stop == 12:
                ref = journal.finish()
            else:
                return None
    return ref


def prepare(case, pref):
    with RunTransactions(case.primary, case.identity) as tx:
        return recovery.prepare_recovery(
            tx,
            experiment_root=case.root,
            phase_context=case.context,
            pilot_spec_sha256=case.spec_hash,
            source_commit_sha256=tx.published()[4]["commit_sha256"],
            primary_execution_ref=pref,
        )


def recover(case, bound, *, mutate_step=None, mutate=None):
    runtime, optimizer, scheduler, _ = _t.setup()
    directory = Path(bound["directory"])
    runtime_files(case, runtime, "independent-recovery", directory)
    with RunTransactions(directory, case.identity) as tx:
        restored = tx.restore_latest(runtime, optimizer, scheduler)
        assert restored["step"] == 4
        with recovery.ExecutionJournal.synthetic(
            tx,
            runtime,
            experiment_root=case.root,
            storage_lane="RECOVERY",
            restored=True,
            recovery_binding_ref=bound["binding_ref"],
        ) as journal:
            for number in range(5, 13):
                journal.record_commit(
                    *step(
                        tx,
                        runtime,
                        optimizer,
                        scheduler,
                        number,
                        mutate=mutate if number == mutate_step else None,
                    )
                )
            ref = journal.finish()
    return ref


def compare(case, bound, pref, rref):
    with (
        RunTransactions(case.primary, case.identity) as ptx,
        RunTransactions(bound["directory"], case.identity) as rtx,
    ):
        return recovery.compare_recovery(
            ptx,
            rtx,
            root=case.root,
            phase_context=case.context,
            pilot_spec_sha256=case.spec_hash,
            binding_ref=bound["binding_ref"],
            primary_execution_ref=pref,
            recovery_execution_ref=rref,
        )


def test_real_cpu_prefix_restore_new_generation_and_exact_all_steps(case):
    pref = primary(case)
    bound = prepare(case, pref)
    assert bound["scope"] == "SYNTHETIC_ONLY"
    assert bound["identity"] == case.identity
    lane = Path(bound["directory"])
    assert not (lane / "artifacts/step-005").exists()
    assert not (lane / "checkpoints/step-005").exists()
    for number in range(5):
        assert (lane / f"checkpoints/step-{number:03d}/state.pt").read_bytes() == (
            case.primary / f"checkpoints/step-{number:03d}/state.pt"
        ).read_bytes()
    again = prepare(case, pref)
    assert again["binding_ref"] == bound["binding_ref"]
    rref = recover(case, bound)
    result = compare(case, bound, pref, rref)
    assert result["status"] == "SYNTHETIC_ONLY" and result["exact"]
    assert result["new_generated_responses"] == 1024
    assert all(s["exact"] for s in result["steps"])
    assert len(list((lane / "checkpoints").glob("step-*"))) == 13


def test_resumed_primary_finishes_honestly_but_cannot_be_recovery_baseline(case):
    pref = primary(case, resumed=True)
    assert not recovery._bound(case.root, pref)["continuity_verified"]
    with pytest.raises(PermissionError, match="continuous"):
        prepare(case, pref)
    assert not (case.root / recovery.DIRECTORY).exists()


def test_completed_step12_can_close_new_process_without_updating(case):
    primary(case)
    runtime_files(case, case.runtime, "after-publish-crash", case.primary)
    with (
        RunTransactions(case.primary, case.identity) as tx,
        recovery.ExecutionJournal.synthetic(
            tx, case.runtime, experiment_root=case.root, restored=True, resumed_after_data=True
        ) as journal,
    ):
        ref = journal.finish()
    end = recovery._bound(case.root, ref)
    assert end["new_model_calls"] == 0 and end["continuity_verified"] is False


def test_partial_segment_retained_and_general_resume_allowed(case):
    assert primary(case, stop=4) is None
    assert (case.primary / "EXECUTIONS/primary-continuous/FAILURE.json").exists()
    runtime_files(case, case.runtime, "resume-general", case.primary)
    with (
        RunTransactions(case.primary, case.identity) as tx,
        recovery.ExecutionJournal.synthetic(
            tx, case.runtime, experiment_root=case.root, restored=True, resumed_after_data=True
        ) as journal,
    ):
        for n in range(5, 13):
            journal.record_commit(*step(tx, case.runtime, case.optimizer, case.scheduler, n))
        ref = journal.finish()
    assert not recovery._bound(case.root, ref)["continuity_verified"]
    with pytest.raises(PermissionError, match="continuous"):
        prepare(case, ref)


def test_model_text_time_is_not_metadata_and_failure_is_saved(case):
    pref = primary(case)
    bound = prepare(case, pref)
    rref = recover(
        case,
        bound,
        mutate_step=7,
        mutate=lambda cached: cached["records"][0].update(
            full_text='{"time":"changed model answer"}'
        ),
    )
    result = compare(case, bound, pref, rref)
    assert result["status"] == "FAIL"
    assert any("original_rows/0/full_text" in path for path in result["differences"])
    assert (case.root / result["receipt_reference"]["path"]).is_file()


@pytest.mark.parametrize("what", ["identity", "state", "marker", "symlink"])
def test_import_tampering_fails_closed(case, what):
    pref = primary(case)
    bound = prepare(case, pref)
    lane = Path(bound["directory"])
    if what == "identity":
        path = lane / "RUN_IDENTITY.json"
        value = json.loads(path.read_bytes())
        value["lane"] = "RECOVERY"
        path.write_text(json.dumps(value))
    elif what == "state":
        with (lane / "checkpoints/step-004/state.pt").open("ab") as stream:
            stream.write(b"bad")
    elif what == "marker":
        path = lane / "checkpoints/step-004/COMMIT.json"
        value = json.loads(path.read_bytes())
        value["rng_sha256"] = "0" * 64
        path.write_text(json.dumps(value))
    else:
        path = lane / "artifacts/step-001/rollout.json"
        original = case.primary / "artifacts/step-001/rollout.json"
        path.unlink()
        path.symlink_to(original)
    with pytest.raises(PermissionError):
        recovery.verify_recovery(
            case.root,
            phase_context=case.context,
            pilot_spec_sha256=case.spec_hash,
            binding_ref=bound["binding_ref"],
        )


def test_unclosed_recovery_attempt_cannot_disappear(case):
    pref = primary(case)
    bound = prepare(case, pref)
    rref = recover(case, bound)
    orphan = Path(bound["directory"]) / "EXECUTIONS/unclosed"
    orphan.mkdir()
    (orphan / "BEGIN.json").write_text("{}")
    with pytest.raises(PermissionError, match="Unclosed"):
        compare(case, bound, pref, rref)
    assert list((Path(bound["directory"]) / "COMPARISONS").glob("*-FAILURE.json"))


def test_production_journal_refuses_cpu_fixture(case):
    with RunTransactions(case.primary, case.identity) as tx:
        tx.commit(
            case.runtime, case.optimizer, case.scheduler, step=0, sampler_state=_t.zero_sampler()
        )
        with (
            pytest.raises(PermissionError, match="single-GPU"),
            recovery.ExecutionJournal(tx, case.runtime, experiment_root=case.root),
        ):
            pass


def test_duplicate_record_commit_and_unlocked_journal_rejected(case):
    with (
        pytest.raises(PermissionError, match="lock"),
        recovery.ExecutionJournal.synthetic(
            RunTransactions(case.primary, case.identity), case.runtime, experiment_root=case.root
        ),
    ):
        pass
    with RunTransactions(case.primary, case.identity) as tx:
        tx.commit(
            case.runtime, case.optimizer, case.scheduler, step=0, sampler_state=_t.zero_sampler()
        )
        with recovery.ExecutionJournal.synthetic(
            tx, case.runtime, experiment_root=case.root
        ) as journal:
            marker, cached = step(tx, case.runtime, case.optimizer, case.scheduler, 1)
            journal.record_commit(marker, cached)
            with pytest.raises(PermissionError, match="exactly once"):
                journal.record_commit(marker, cached)


@pytest.mark.parametrize(
    "text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{"x":-1e999}']
)
def test_strict_json_rejects_ambiguous_nonfinite_evidence(text):
    with pytest.raises(PermissionError):
        recovery._json(text)


def test_exact_tree_retains_types_dtypes_shape_and_nested_time():
    assert recovery._differences({"time": "a"}, {"time": "b"}) == ["/time"]
    assert recovery._differences(1, True)
    assert recovery._differences(torch.tensor([1.0]), torch.tensor([1.0], dtype=torch.float64))
    assert recovery._differences(torch.ones(2), torch.ones(1, 2))


def test_repeat_verification_does_not_reload_prefix_tensor_states(case, monkeypatch):
    pref = primary(case)
    bound = prepare(case, pref)
    calls = []
    original = torch.load

    def observe(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", observe)
    again = recovery.verify_recovery(
        case.root,
        phase_context=case.context,
        pilot_spec_sha256=case.spec_hash,
        binding_ref=bound["binding_ref"],
    )
    assert again["binding_ref"] == bound["binding_ref"]
    assert calls == []
    # Metadata invalidates the cache; immutable content is then rechecked.
    target = Path(bound["directory"]) / "checkpoints/step-003/state.pt"
    with target.open("ab") as out:
        out.write(b"changed")
    with pytest.raises(PermissionError, match="prefix content"):
        recovery.verify_recovery(
            case.root,
            phase_context=case.context,
            pilot_spec_sha256=case.spec_hash,
            binding_ref=bound["binding_ref"],
        )
    assert calls == []  # A known immutable hash rejects changed bytes before tensor loading.


@pytest.mark.parametrize("earlier_calls", [0, 8])
def test_predata_failure_remains_but_only_prior_actual_data_breaks_continuity(case, earlier_calls):
    folder = case.primary / "EXECUTIONS/preload-failure"
    folder.mkdir(parents=True)
    (folder / "FAILURE.json").write_text('{"status":"PRELOAD_FAILURE"}')
    ledger = case.primary / "accounting/preload-failure.jsonl"
    ledger.parent.mkdir()
    ledger.write_text(
        json.dumps(
            dict(
                task_id="preload-failure",
                parent="b0",
                kind="completion_attempts",
                count=earlier_calls,
                metadata={},
            )
        )
        + "\n"
    )
    ref = primary(case)
    assert recovery._bound(case.root, ref)["continuity_verified"] == (earlier_calls == 0)
    assert (folder / "FAILURE.json").is_file()


def test_wrong_seed_with_correct_totals_rejected_and_failure_retained(case, monkeypatch):
    original = accounting

    def changed(runtime, records):
        original(runtime, records)
        path = (
            Path(runtime.identity["audit_root"])
            / "accounting"
            / (runtime.identity["task_id"] + ".jsonl")
        )
        events = recovery._events(path)
        events[-1]["metadata"]["group_seed"] += 1
        path.write_text("".join(json.dumps(e) + "\n" for e in events))

    monkeypatch.setitem(accounting.__globals__, "accounting", changed)
    with pytest.raises(PermissionError, match="G8/draw accounting"):
        primary(case)
    assert (case.primary / "EXECUTIONS/primary-continuous/FAILURE.json").is_file()


def test_actual_raw_rng_closure_and_annotated_trajectory_binding(tmp_path, prototype):
    # The real collector's CPU runtime creates 16 COMPLETE G8, original callback
    # JSONL and native CPU RNG files; this is still synthetic data, not GPU proof.
    fixture = module("recovery_raw_rollout_fixture", "test_rollouts.py")
    case = fixture.prepare(tmp_path, prototype)
    with RunTransactions(case["run"], case["identity"]) as tx:
        tx.commit(
            case["runtime"],
            case["optimizer"],
            case["scheduler"],
            step=0,
            sampler_state=fixture.initial_sampler(),
        )
        result = fixture.rollouts.collect_step(
            tx, case["runtime"], experiment_root=case["experiment"], registration_ref=case["ref"]
        )
        marker = dict(references=result["references"])
        actual = recovery._originals(tx, marker)
        assert len(actual["original_raw"]) == 128 and len(actual["rollout_rng"]) == 34
        envelope = recovery._bound(tx.directory, result["references"]["rollout"])
        envelope["records"][0]["tokens"][0] += 1
        marker["references"]["rollout"] = _t.save_json(
            tx.directory, result["references"]["rollout"]["path"], envelope
        )
        with pytest.raises(PermissionError, match="raw trajectory"):
            recovery._originals(tx, marker)


def test_changing_scope_cannot_upgrade_cpu_journal_to_actual_gpu(case):
    ref = primary(case)
    end_path = case.root / ref["path"]
    end = recovery._bound(case.root, ref)
    begin_path = case.root / end["begin_reference"]["path"]
    begin = json.loads(begin_path.read_bytes())
    begin["scope"] = "SERVER_ACTUAL"
    begin_path.write_text(json.dumps(begin))
    end["scope"] = "SERVER_ACTUAL"
    end["begin_reference"] = recovery._ref(case.root, begin_path)
    end_path.write_text(json.dumps(end))
    with pytest.raises(PermissionError, match=r"Actual OS process|synthetic cannot be upgraded"):
        prepare(case, recovery._ref(case.root, end_path))


def test_real_pilot_drive_bridge_prefix_and_independent_recovery(case, prototype):
    from contextlib import nullcontext

    from sr_f3_ce.training import fresh_optimizer
    from sr_f6 import pilot
    from srf6_ref.measurement import full_truth_response

    fixture = module("recovery_real_pilot_fixture", "test_pilot.py")
    parent_model = copy.deepcopy(case.runtime.model)

    class BoundCPUFixture(fixture.CPUTrainerRuntime):
        def generate_group(self, safe, root, seeds, *, protocol, on_completion):
            def capture(row):
                for key in ("phase_identity_sha256", "origin_identity_sha256"):
                    row[key] = self.identity[key]
                on_completion(row)

            return super().generate_group(
                safe, root, seeds, protocol=protocol, on_completion=capture
            )

    def fresh(directory, task):
        runtime = BoundCPUFixture(
            copy.deepcopy(parent_model),
            {**case.runtime.identity, "step": 0},
            full_truth_response(prototype["world"], prototype["query"]),
        )
        runtime_files(case, runtime, task, directory)
        return runtime, *fresh_optimizer(runtime.model)

    runtime, optimizer, scheduler = fresh(case.primary, "real-primary")
    with RunTransactions(case.primary, case.identity) as tx:
        result = pilot.run_synthetic(
            tx,
            runtime,
            optimizer,
            scheduler,
            experiment_root=case.root,
            registration_ref=case.identity["rollout_registration"],
        )
    pref = result["execution"]
    bound = prepare(case, pref)
    lane = Path(bound["directory"])
    audit_item = next(
        item
        for item in bound["binding"]["prefix_files"]
        if item["path"] == "NUMERICAL_TOKEN_AUDIT.jsonl"
    )
    source_bytes = (case.primary / audit_item["path"]).read_bytes()
    imported = (lane / audit_item["path"]).read_bytes()
    assert imported == source_bytes[: audit_item["source_byte_count"]]
    assert len(imported) < len(source_bytes) and len(imported.splitlines()) == 4
    assert not (lane / "rollouts/step-005").exists()
    runtime, optimizer, scheduler = fresh(lane, "real-recovery")
    with RunTransactions(lane, case.identity) as tx:
        recovered = pilot._drive(
            tx,
            runtime,
            optimizer,
            scheduler,
            experiment_root=case.root,
            registration_ref=case.identity["rollout_registration"],
            access_factory=nullcontext,
            journal_factory=recovery.ExecutionJournal.synthetic,
            scope="SYNTHETIC_ONLY",
            storage_lane="RECOVERY",
            recovery_binding_ref=bound["binding_ref"],
        )
    assert runtime.calls == 1024 and recovered["segment_updates"] == 8
    rref = recovered["execution"]
    primary_evidence = json.loads(
        (case.primary / "PREUPDATE_EVIDENCE/step-005/real-primary.json").read_bytes()
    )
    recovery_evidence = json.loads(
        (lane / "PREUPDATE_EVIDENCE/step-005/real-recovery.json").read_bytes()
    )
    assert primary_evidence["task_id"] == "real-primary"
    assert recovery_evidence["task_id"] == "real-recovery"
    # Only the top-level execution ID is ignored, after both original slices verify.
    recovery._audit_event(case.primary, primary_evidence, required=True)
    recovery._audit_event(lane, recovery_evidence, required=True)
    exact = compare(case, bound, pref, rref)
    assert exact["exact"] and exact["status"] == "SYNTHETIC_ONLY", exact["differences"][:5]
    with (
        RunTransactions(case.primary, case.identity) as ptx,
        RunTransactions(lane, case.identity) as rtx,
    ):
        before = set((lane / "COMPARISONS").iterdir())
        verified = recovery.verify_comparison(
            ptx,
            rtx,
            root=case.root,
            phase_context=case.context,
            pilot_spec_sha256=case.spec_hash,
            binding_ref=bound["binding_ref"],
            primary_execution_ref=pref,
            recovery_execution_ref=rref,
            comparison_ref=exact["receipt_reference"],
        )
        assert verified["verified_from_saved_facts"]
        assert set((lane / "COMPARISONS").iterdir()) == before
        # Keep the actual evidence JSON unchanged but corrupt its referenced
        # append-only bytes. No exclusion can turn that into matching evidence.
        path = lane / "NUMERICAL_TOKEN_AUDIT.jsonl"
        original = path.read_bytes()
        fifth = original.splitlines(keepends=True)[4]
        replacement = fifth.replace(b'"audit_root"', b'"audit_roox"', 1)
        assert replacement != fifth and len(replacement) == len(fifth)
        path.write_bytes(original.replace(fifth, replacement, 1))
        with pytest.raises(PermissionError, match="audit slice"):
            recovery.verify_comparison(
                ptx,
                rtx,
                root=case.root,
                phase_context=case.context,
                pilot_spec_sha256=case.spec_hash,
                binding_ref=bound["binding_ref"],
                primary_execution_ref=pref,
                recovery_execution_ref=rref,
                comparison_ref=exact["receipt_reference"],
            )


def test_failure_difference_order_is_stable_and_does_not_follow_hash_set_order():
    left = {key: 0 for key in ("zeta", "beta", "alpha", "gamma", "epsilon", "delta")}
    right = dict.fromkeys(left, 1)
    assert recovery._differences(left, right) == ["/" + key for key in sorted(left)]


def test_process_identity_is_stable_and_another_actual_process_is_distinct():
    import os
    import subprocess
    import sys

    current = recovery._process_identity()
    assert current == recovery._process_identity() and current["pid"] == os.getpid()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(recovery.__file__).parents[1])
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from sr_f6.recovery import _process_identity; "
            "print(json.dumps(_process_identity()))",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    child = json.loads(result.stdout)
    assert child != current and child["pid"] != current["pid"]
    # OS identity is an engineering fact, not evidence of any GPU execution.


def test_same_process_recovery_rejected_before_gpu_check_or_any_generation(case):
    pref = primary(case)
    bound = prepare(case, pref)
    # Deliberately forged scope headers can never obtain actual execution. They
    # exercise the earlier independent-process rejection without faking CUDA.
    end_path = case.root / pref["path"]
    end = recovery._bound(case.root, pref)
    begin_path = case.root / end["begin_reference"]["path"]
    begin = json.loads(begin_path.read_bytes())
    begin["scope"] = "SERVER_ACTUAL"
    begin_path.write_text(json.dumps(begin))
    end["scope"] = "SERVER_ACTUAL"
    end["begin_reference"] = recovery._ref(case.root, begin_path)
    end_path.write_text(json.dumps(end))
    binding_path = case.root / bound["binding_ref"]["path"]
    binding = json.loads(binding_path.read_bytes())
    binding["primary_execution_ref"] = recovery._ref(case.root, end_path)
    binding_path.write_text(json.dumps(binding))
    runtime, optimizer, scheduler, _ = _t.setup()
    runtime_files(case, runtime, "must-not-start", Path(bound["directory"]))
    with RunTransactions(bound["directory"], case.identity) as tx:
        tx.restore_latest(runtime, optimizer, scheduler)
        with (
            pytest.raises(PermissionError, match="independent OS process before generation"),
            recovery.ExecutionJournal(
                tx,
                runtime,
                experiment_root=case.root,
                storage_lane="RECOVERY",
                restored=True,
                recovery_binding_ref=recovery._ref(case.root, binding_path),
            ),
        ):
            pytest.fail("Same-process recovery must stop before entering its body")
        assert tx.published()[-1]["step"] == 4
    assert not (Path(bound["directory"]) / "EXECUTIONS/must-not-start").exists()


def test_only_top_level_evidence_task_id_is_execution_metadata():
    left = {"task_id": "first", "model_payload": {"task_id": "literal model text"}}
    right = {"task_id": "second", "model_payload": {"task_id": "literal model text"}}

    def projected(value):
        return recovery._without(value, recovery.EVIDENCE_EXECUTION_KEYS)

    assert recovery._differences(projected(left), projected(right)) == []
    right["model_payload"]["task_id"] = "changed literal model text"
    assert recovery._differences(projected(left), projected(right)) == ["/model_payload/task_id"]
