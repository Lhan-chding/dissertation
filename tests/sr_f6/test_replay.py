"""CPU synthetic evidence and mocked native forwards; no actual T03 receipt."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mm_core.training import state_hash, trainable_state
from sr_f6 import census, replay, runtime
from sr_f6.common import digest, file_hash
from sr_f12.training import probability_diagnostics
from srf6_ref.core import FAMILIES
from srf6_ref.design import make_root
from srf6_ref.measurement import full_truth_response


def save(root, relative, value, *, lines=False):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x) + "\n" for x in value) if lines else json.dumps(value))
    return {"path": relative, "sha256": file_hash(path)}


@pytest.fixture
def fx(tmp_path, monkeypatch):
    root = tmp_path
    config = json.loads((Path(__file__).parents[2] / "configs/sr_f6.json").read_text())
    save(root, "CONFIG.json", config)
    manifest = save(root, "DATA/MANIFEST.json", {"synthetic": True})
    proto = make_root("DEV_CAL", "BINDING", 0, 2026102711)["tasks"]["P0"][0]
    public, tasks = [], []
    for family in FAMILIES:
        for n in range(8):
            rid = f"DEV_CAL_{family}_{n:04d}"
            for factor in census.FACTORS:
                for variant in range(4):
                    qid = f"{rid}_{factor}_v{variant}"
                    task = dict(
                        proto, qid=qid, root_id=rid, family=family, factor=factor, variant=variant
                    )
                    tasks.append(task)
                    item = {
                        k: task[k]
                        for k in ("qid", "root_id", "family", "factor", "variant", "split")
                    }
                    public.append(
                        dict(
                            item,
                            protocol="structured_v4",
                            model_input=dict(
                                prompt="synthetic", image_file="fixture.png", image_sha256="e" * 64
                            ),
                        )
                    )
    save(root, "DATA/DEV_CAL/MODEL_INPUTS.jsonl", public, lines=True)
    save(root, "DATA/DEV_CAL/TASKS_PRIVATE.jsonl", tasks, lines=True)
    frozen = census.freeze_spec(root, dict.fromkeys(census.PARENTS, "a" * 64))
    origin = {"source_manifest": {"sha256": "b" * 64}, "data_manifest": manifest}
    origin_ref = save(root, "IDENTITY.json", origin)
    ready_ref = save(
        root,
        "PREFLIGHT/CENSUS_READY.json",
        dict(
            context=origin,
            status="READY_FOR_CENSUS",
            execution_origin="SERVER_ACTUAL",
            phase="DEV_CAL",
            optimizer_updates=0,
        ),
    )
    source = dict(
        phase="PILOT_V1",
        status="BOUND_NOT_AUTHORIZED",
        training_allowed=False,
        identity_sha256="d" * 64,
        origin_identity_sha256=origin_ref["sha256"],
        identity=dict(
            source_manifest={"sha256": "c" * 64},
            data_manifest=manifest,
            upstream=[
                dict(
                    producer_phase="CENSUS_S0",
                    identity_sha256=origin_ref["sha256"],
                    evidence=[ready_ref],
                )
            ],
        ),
    )
    (root / "PHASES/PILOT_V1").mkdir(parents=True)
    monkeypatch.setattr(replay, "_source", lambda *a: copy.deepcopy(source))
    r = runtime.Runtime.__new__(runtime.Runtime)
    r.model, r.torch = torch.nn.Linear(1, 1), torch
    r.model.register_buffer("frozen_buffer", torch.zeros(1))
    r._srf6_adapter_parameter_names = tuple(n for n, _ in r.model.named_parameters())
    policy = state_hash(trainable_state(r.model))
    audit = root / "PHASE_RUNS/PILOT_V1/REPLAY_RUNS/test"
    r.identity = dict(
        run_root=str(root),
        audit_root=str(audit),
        task_id="test",
        run_id="t03-b0",
        phase="PILOT_V1",
        execution_mode="REPLAY",
        phase_identity_sha256="d" * 64,
        parent_id="b0",
        step=0,
        policy_hash=policy,
        reference_hash=policy,
        parent_content_hash="a" * 64,
        source_sha256="c" * 64,
        source_hash="c" * 64,
        data_hash=manifest["sha256"],
        origin_identity_sha256=origin_ref["sha256"],
    )
    r._srf6_source_context = copy.deepcopy(source)
    r._srf6_execution_identity = {k: r.identity[k] for k in runtime.EXECUTION_IDENTITY_FIELDS}
    r._srf6_execution_mode, r._srf6_policy_identity = "REPLAY", (0, policy)
    save(
        root,
        "PHASE_RUNS/PILOT_V1/REPLAY_RUNS/test/runtime_identity/test.json",
        dict(r.identity, fresh_optimizer=False),
    )
    public, task_map = census.census_inputs(root)
    item, task_id = public[0], "CENSUS-b0-0-of-2"
    spec = census.group_spec(config, "b0", item, task_map[item["qid"]], frozen)
    sr = dict(
        run_id=task_id,
        run_root=str(root),
        parent_id="b0",
        step=0,
        reference_parent="b0",
        parent_content_hash="a" * 64,
        source_sha256="b" * 64,
        source_hash="b" * 64,
        data_hash=manifest["sha256"],
        fresh_optimizer=False,
        dtype="float32",
        probability_dtype="float32",
        sr_f6_protocol_version=census.PROTOCOL_VERSION,
        policy_hash=policy,
        reference_hash=policy,
    )
    save(root, f"runtime_identity/{task_id}.json", sr)
    native = {
        "input_ids": [[1]],
        "attention_mask": [[1]],
        "mm_token_type_ids": [[1]],
        "image_grid_thw": [[1, 1, 1]],
    }
    raw = [
        dict(
            run_id=task_id,
            run_root=str(root),
            policy_hash=policy,
            policy_tensor_hash=policy,
            parent_content_hash="a" * 64,
            source_sha256="b" * 64,
            source_hash="b" * 64,
            data_hash=manifest["sha256"],
            step=0,
            policy_step=0,
            protocol="structured_v4",
            sr_f6_protocol_version=census.PROTOCOL_VERSION,
            sampler_logprob_source="actual_generate_raw_logits_selected_token",
            adapter_identity=dict(run_id=task_id, parent="b0", step=0),
            prompt_tensor_hash="e" * 64,
            native_prompt_inputs=native,
            seed=seed,
            generation_status="COMPLETE",
            technical_validation_errors=[],
            tokens=[2, 3],
            raw_tokens=[2, 3],
            completion_token_count=2,
            max_new_tokens=1536,
            group_size=8,
            group_row_index=draw,
            group_seed=spec["seeds"][0],
            truncated=False,
            balanced_token_count=2,
            boundary_token_count=2,
            sampler_logprobs=[-1.0, -2.0],
            model_input=item["model_input"],
            full_text=full_truth_response(proto["world"], proto["query"]),
            finish_reason="balanced",
        )
        for draw, seed in enumerate(spec["seeds"])
    ]
    raw[-1].update(finish_reason="eos", balanced_token_count=None, boundary_token_count=None)
    records = census.annotate_group(raw, task_map[item["qid"]], spec)
    save(
        root,
        f"CENSUS/attempts/b0/{item['qid']}.json",
        dict(status="STARTED", task_id=task_id, spec=spec),
    )
    save(root, f"CENSUS/raw/b0/{item['qid']}.jsonl", raw, lines=True)
    save(
        root,
        f"CENSUS/groups/b0/{item['qid']}.json",
        dict(
            status="COMPLETE",
            spec=spec,
            records=records,
            credit=census.counterfactual(records, spec),
        ),
    )
    save(
        root,
        f"CENSUS/alignment/{task_id}.json",
        dict(
            status="PASS",
            execution_origin="SERVER_ACTUAL",
            phase="PARENT_CENSUS",
            parent="b0",
            group_id=spec["group_id"],
            parameter_tensor_sha256=policy,
            original_registered_responses=8,
            checked_sequences=8,
            new_generated_responses=0,
            optimizer_updates=0,
            identity=dict(
                frozen_spec_sha256=spec["frozen_spec_sha256"],
                group_spec_sha256=digest(spec),
                raw_trajectories_sha256=digest(
                    [
                        {k: row[k] for k in ("tokens", "sampler_logprobs", "prompt_tensor_hash")}
                        for row in records
                    ]
                ),
            ),
            traces=[
                dict(
                    row_id=row["row_id"],
                    token_ids=row["tokens"],
                    sampled_logprobs=row["sampler_logprobs"],
                    current_logprobs=row["sampler_logprobs"],
                )
                for row in records
            ],
            diagnostics=probability_diagnostics(
                [torch.zeros(2)] * 8, [torch.zeros(2, dtype=torch.bool)] * 8
            ),
        ),
    )
    prepared = dict(
        inputs={
            **{k: torch.tensor(v) for k, v in native.items()},
            "pixel_values": torch.ones(1, 1),
        },
        routing=dict(input_tensor_hash="e" * 64),
    )
    r.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    monkeypatch.setattr(runtime.FP32Runtime, "prepare_input", lambda *a, **k: prepared)
    monkeypatch.setattr(r, "_validate_prepared_revision", lambda *a: None)
    monkeypatch.setattr(
        runtime.FP32Runtime,
        "_prepared_records",
        lambda self, rows: (
            rows[0].get("prepared") or self.prepare_input(rows[0]["model_input"], root)
        ),
    )
    calls = []

    def native_forward(self, inputs, **kw):
        calls.append((inputs, kw))
        return [torch.tensor([-1.0, -2.0], requires_grad=kw["grad"])]

    monkeypatch.setattr(runtime.FP32Runtime, "_native_forward", native_forward)
    return SimpleNamespace(
        root=root,
        runtime=r,
        prepared=prepared,
        item=item,
        raw=raw,
        calls=calls,
        policy=policy,
        spec_ref=replay.freeze_spec(root, phase_identity_sha256="d" * 64),
    )


def test_freeze_rules_without_other_parents_and_exact_idempotence(fx):
    spec = replay.verify_spec(fx.root, fx.spec_ref, phase_identity_sha256="d" * 64)
    assert set(spec["groups"]) == {"b0", "b1", "b2"}
    assert spec["rules"]["atol"] == 1e-6 and spec["rules"]["rtol"] == 1e-5
    assert spec["rules"]["probability_mean_and_tis_denominator"] == (
        "all completion tokens in the selected original G8"
    )
    assert spec["rules"]["frozen_base_buffer_mutation"] == (
        "FAIL; poison runtime; discard/reload; not claimed restored"
    )
    assert replay.freeze_spec(fx.root, phase_identity_sha256="d" * 64) == fx.spec_ref
    assert not (fx.root / "CENSUS/workers").exists()
    assert not (fx.root / "CENSUS/raw/b1").exists()


def test_original_tokens_source_and_eos_survive_guarded_native_forward(fx):
    r = fx.runtime
    with replay.open_replay(r, fx.spec_ref) as session:
        assert session.records[0]["source_sha256"] == "b" * 64
        assert session.raw_records[-1]["finish_reason"] == "eos"
        assert session.records[-1]["finish_reason"] == "length"
        prepared = r.prepare_input(fx.item["model_input"], fx.root)
        record = dict(session.records[0], prepared=prepared)
        r.batch_sequence_forward([record], purpose="t03_native", grad=True)
        r.batch_reference_forward([record], session.reference)
        inputs, arguments = fx.calls[-1]
        assert inputs["input_ids"].tolist() == [[1, 2, 3]]
        assert arguments["completion_start"] == 1 and arguments["lengths"] == [2]
        assert arguments["grad"] is False
        with pytest.raises(PermissionError, match="Native forward requires"):
            r._native_forward(inputs, **arguments)
    with pytest.raises(PermissionError, match="Inactive"):
        session.assert_active()
    with pytest.raises(PermissionError, match="open_replay"):
        r._batch_forward(prepared, [[2, 3]], purpose="t03_native", grad=True)


@pytest.mark.parametrize(
    "defect",
    ["input", "tokens", "prepared", "prepared_mutation", "record", "phase", "mode", "spec"],
)
def test_replay_access_rejects_unbound_or_mutated_values(fx, defect):
    r = fx.runtime
    with replay.open_replay(r, fx.spec_ref) as session:
        prepared = r.prepare_input(fx.item["model_input"], fx.root)
        original_identity, original_spec = copy.deepcopy(r.identity), copy.deepcopy(session.spec)
        try:
            with pytest.raises(PermissionError):
                if defect == "input":
                    r.prepare_input(dict(fx.item["model_input"], prompt="other panel"), fx.root)
                elif defect == "tokens":
                    r._batch_forward(prepared, [[2, 4]], purpose="t03_native", grad=True)
                elif defect == "prepared":
                    r._batch_forward(
                        copy.deepcopy(prepared), [[2, 3]], purpose="t03_native", grad=True
                    )
                elif defect == "prepared_mutation":
                    prepared["inputs"]["pixel_values"].add_(1)
                    r.prepare_input(fx.item["model_input"], fx.root)
                elif defect == "record":
                    r.batch_sequence_forward(
                        [dict(session.records[0], seed=1)], purpose="t03_native"
                    )
                elif defect in {"phase", "mode"}:
                    r.identity["phase" if defect == "phase" else "execution_mode"] = (
                        "CENSUS_S0" if defect == "phase" else "CENSUS"
                    )
                    r.prepare_input(fx.item["model_input"], fx.root)
                else:
                    session.spec["rules"]["atol"] = 100
                    session.assert_active()
        finally:
            r.identity, session.spec = original_identity, original_spec


@pytest.mark.parametrize("target", ["raw", "attempts", "groups", "alignment"])
def test_parent_evidence_must_be_original_before_any_forward(fx, target):
    if target == "alignment":
        path = fx.root / "CENSUS/alignment/CENSUS-b0-0-of-2.json"
        value = json.loads(path.read_text())
        value["traces"][0]["token_ids"] = [999, 3]
    else:
        ext = "jsonl" if target == "raw" else "json"
        path = fx.root / f"CENSUS/{target}/b0/{fx.item['qid']}.{ext}"
        if target == "raw":
            value = copy.deepcopy(fx.raw)
            value[0]["tokens"] = [999, 3]
            path.write_text("".join(json.dumps(row) + "\n" for row in value))
        else:
            value = json.loads(path.read_text())
            value["spec"]["qid"] = "other"
    if target != "raw":
        path.write_text(json.dumps(value))
    with pytest.raises((PermissionError, ValueError)), replay.open_replay(fx.runtime, fx.spec_ref):
        pytest.fail("Changed original evidence opened")
    assert not fx.calls


def test_reference_cache_and_restoration_check_only_adapter(fx, monkeypatch):
    with replay.open_replay(fx.runtime, fx.spec_ref) as session:
        original = replay.state_hash
        calls = []
        monkeypatch.setattr(
            replay, "state_hash", lambda value: (calls.append(value), original(value))[1]
        )
        for _ in range(8):
            session.check_reference(session.reference)
        assert calls == []
        with torch.no_grad():
            next(fx.runtime.model.parameters()).add_(1)
        with pytest.raises(PermissionError, match="original parent"):
            session.refresh_adapter_versions()
        with torch.no_grad():
            for n, p in fx.runtime.model.named_parameters():
                p.copy_(session.reference[n])
        session.refresh_adapter_versions()


def test_base_buffer_mutation_poisoning_is_not_a_reusable_session(fx):
    with (
        pytest.raises(PermissionError, match="base/buffers"),
        replay.open_replay(fx.runtime, fx.spec_ref) as session,
    ):
        fx.runtime.model.frozen_buffer.add_(1)
        session.assert_active()
    assert fx.runtime._srf6_replay_poisoned is True


def test_lookalike_session_and_private_sampler_cannot_authorize_generation(fx):
    fx.runtime._srf6_replay_session = SimpleNamespace(active=True)
    with pytest.raises(PermissionError, match="open_replay"):
        fx.runtime._prepared_records([{}])
    with replay.open_replay(fx.runtime, fx.spec_ref):
        fx.runtime.generation_config = SimpleNamespace()
        fx.runtime.protocol_revision = "v2"
        with pytest.raises(PermissionError, match="forbids generation"):
            runtime.Runtime._generate_native_group(
                fx.runtime, fx.item["model_input"], fx.root, list(range(8))
            )


def test_live_session_cannot_mint_a_prepared_or_native_permit(fx):
    with replay.open_replay(fx.runtime, fx.spec_ref) as session:
        with pytest.raises(PermissionError, match="Only the verified runtime"):
            session.bind_prepared(fx.prepared)
        with (
            pytest.raises(PermissionError, match="Only the verified runtime"),
            session.native_call({}, completion_start=1, lengths=[1], purpose="test", grad=True),
        ):
            pytest.fail("Arbitrary native input gained permission")


def test_unknown_parent_raw_does_not_select_another_group(fx):
    spec = replay.verify_spec(fx.root, fx.spec_ref, phase_identity_sha256="d" * 64)
    with pytest.raises((OSError, PermissionError)):
        replay._parent_evidence(fx.root, spec, "b1")


def test_replay_spec_is_content_pinned_and_cannot_relax_fixed_rules(fx):
    path = fx.root / replay.SPEC_PATH
    spec = json.loads(path.read_text())
    spec["rules"]["rtol"] = 1
    path.write_text(json.dumps(spec))
    with pytest.raises(PermissionError, match="bytes changed"):
        replay.verify_spec(fx.root, fx.spec_ref, phase_identity_sha256="d" * 64)
    new_ref = dict(fx.spec_ref, sha256=file_hash(path))
    with pytest.raises(PermissionError, match="rules or provenance"):
        replay.verify_spec(fx.root, new_ref, phase_identity_sha256="d" * 64)
