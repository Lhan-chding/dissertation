"""Synthetic trajectories test boundaries; these are not actual model evidence."""

from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import pytest
import torch

from sr_f6 import confirm_census as cc
from sr_f6.common import digest, read, write_once
from sr_f6.interface_v3_protocol import VERSION
from sr_f6.runtime import PROTOCOL_VERSION, generation_recipe
from srf6_ref.core import seed_for
from srf6_ref.design import build_bank
from srf6_ref.measurement import full_truth_response


@pytest.fixture(scope="module")
def task():
    return build_bank(cc.SPLIT, 1, cc.MASTER)[0]["tasks"]["P0"][0]


@pytest.fixture
def inputs(task):
    item = {k: task[k] for k in ("qid", "root_id", "family", "factor", "variant", "split")}
    item.update(
        model_input={
            "prompt": "synthetic public question",
            "image_file": "DATA/images/x.png",
            "image_sha256": "a" * 64,
        },
        prompt_sha256=hashlib.sha256(b"synthetic public question").hexdigest(),
    )
    plan = dict(
        attempts=[
            dict(
                parent="b0",
                qid=item["qid"],
                draw=d,
                seed=seed_for(
                    cc.MASTER,
                    "srf6-confirm-v3",
                    "confirm",
                    "b0",
                    task["root_id"],
                    task["variant"],
                    d,
                ),
            )
            for d in range(8)
        ]
    )
    context = dict(
        spec=plan,
        spec_object_sha256=digest(plan),
        registration_sha256="1" * 64,
        registration=dict(
            source_manifest_sha256="2" * 64,
            data_sha256="3" * 64,
            parent_content_hashes={"b0": "4" * 64},
        ),
        private={task["qid"]: task},
    )
    spec = cc.group_spec(context, "b0", item)
    text = full_truth_response(task["world"], task["query"])
    raw = []
    for draw, seed in enumerate(spec["seeds"]):
        row = {
            k: spec[k]
            for k in (
                "phase",
                "registration_sha256",
                "source_sha256",
                "data_hash",
                "parent_content_hash",
            )
        }
        row.update(
            policy_tensor_hash="5" * 64,
            generation_status="COMPLETE",
            technical_validation_errors=[],
            execution_mode="READONLY_SCHEMA_CONFIRM",
            optimizer_updates=0,
            step=0,
            policy_step=0,
            protocol="structured_v4",
            prompt_version=VERSION,
            sr_f6_protocol_version=PROTOCOL_VERSION,
            seed_semantics="independent_per_draw_torch_generator",
            group_size=8,
            group_row_index=draw,
            group_seed=spec["seeds"][0],
            seed=seed,
            adapter_identity={"parent": "b0", "step": 0},
            tokens=[10],
            raw_tokens=[10],
            sampler_logprobs=[-0.5],
            completion_token_count=1,
            max_new_tokens=1536,
            sampler_logprob_source="actual_generate_raw_logits_selected_token",
            finish_reason="balanced",
            full_text=text,
            task_id="LH-SRF6-Confirm-123-b0-0",
            model_input=item["model_input"],
            prompt_sha256=item["prompt_sha256"],
            sampling_parameters=generation_recipe(),
            prompt_tensor_hash="6" * 64,
        )
        raw.append(row)
    return context, item, spec, raw


def test_original_g8_credit_and_native_eos_are_preserved(task, inputs):
    _, _, spec, raw = inputs
    before = copy.deepcopy(raw)
    raw[0]["finish_reason"] = "eos"
    records = cc.annotate(raw, spec, task)
    assert records[0]["finish_reason"] == "eos"
    assert records[0]["raw_finish_reason"] == "eos"
    assert records[0]["finished"] == 0
    assert records[0]["S"] is records[0]["B"] is None
    assert raw[1:] == before[1:]
    group = dict(
        status="COMPLETE", spec=spec, records=records, credit=cc.counterfactual(records, spec)
    )
    assert cc.validate_group(group, spec, task) == records
    group["records"][1]["A"] = 0
    with pytest.raises(PermissionError, match="labels or G8"):
        cc.validate_group(group, spec, task)


@pytest.mark.parametrize(
    "change,error",
    [
        ({"phase": "PARENT_CENSUS"}, PermissionError),
        ({"registration_sha256": "0" * 64}, PermissionError),
        ({"execution_mode": "READONLY_INTERFACE_SCREEN"}, PermissionError),
        ({"group_size": 2}, PermissionError),
        ({"optimizer_updates": True}, PermissionError),
        ({"technical_validation_errors": ["bad"]}, PermissionError),
        ({"seed": 1}, ValueError),
        ({"tokens": [True], "raw_tokens": [True]}, ValueError),
        ({"sampler_logprobs": [float("nan")]}, ValueError),
        ({"sampler_logprobs": [0.1]}, ValueError),
        ({"finish_reason": "length"}, ValueError),
        ({"sampling_parameters": {}}, PermissionError),
        ({"parent_content_hash": "0" * 64}, PermissionError),
    ],
)
def test_tampered_or_partial_original_is_rejected(task, inputs, change, error):
    _, _, spec, raw = inputs
    raw[0].update(change)
    with pytest.raises(error):
        cc.annotate(raw, spec, task)


def test_seeds_are_frozen_factor_matched_and_no_duplicate_or_partial_draws(task, inputs):
    context, item, spec, raw = inputs
    twin = dict(item, factor="P1")
    assert cc.group_spec(context, "b0", twin)["seeds"] == spec["seeds"]
    for bad in (raw[:-1], raw + raw[:1], [raw[0]] * 8):
        with pytest.raises(ValueError):
            cc.annotate(bad, spec, task)
    context["spec"]["attempts"][0]["seed"] += 1
    with pytest.raises(PermissionError, match="prospective rule"):
        cc.group_spec(context, "b0", item)


def test_preflight_route_enforced(task, inputs):
    _, _, spec, raw = inputs
    route = dict(
        input_tensor_hash="6" * 64, processor_hash="p", chat_template_hash="c", chat_text_sha256="t"
    )
    with pytest.raises(PermissionError, match="processor"):
        cc.annotate(raw, spec, task, preflight_route=route)
    for row in raw:
        row["input_routing"] = route
    assert len(cc.annotate(raw, spec, task, preflight_route=route)) == 8


def test_actual_trace_recomputation_failure_retention_and_no_replay(task, inputs, tmp_path):
    _, _, spec, raw = inputs
    records = cc.annotate(raw, spec, task)

    def replay(rows, *, on_replay):
        values = [torch.tensor(r["sampler_logprobs"]) for r in rows]
        for i, value in enumerate(values):
            on_replay(i, value)
        return values

    runtime = SimpleNamespace(
        torch=torch,
        policy_tensor_identity=lambda: "5" * 64,
        replay_alignment=replay,
    )
    path = tmp_path / "alignment.json"
    report = cc.check_alignment(runtime, raw, records, spec, path)
    assert report["status"] == "PASS"
    assert report["diagnostics"]["sampler_train_max_absolute_difference"] == 0
    with pytest.raises(FileExistsError):
        cc.check_alignment(runtime, raw, records, spec, path)
    report["traces"][0]["token_ids"] = [11]
    with pytest.raises(PermissionError, match="original draw"):
        cc.validate_alignment(report, spec, records)

    def failed_replay(rows, *, on_replay):
        for i, _ in enumerate(rows):
            on_replay(i, torch.tensor([0.0]))
        raise RuntimeError("synthetic numerical rejection after actual vectors")

    runtime.replay_alignment = failed_replay
    failure = tmp_path / "failure.json"
    with pytest.raises(RuntimeError):
        cc.check_alignment(runtime, raw, records, spec, failure)
    assert read(failure)["status"] == "FAIL"
    assert len(read(failure)["traces"]) == 8


def test_technical_load_failure_consumes_fixed_shard_slot(inputs, tmp_path, monkeypatch):
    from sr_f6 import confirm_registration, confirm_runtime

    context, item, _, _ = inputs
    context.update(
        root=tmp_path,
        public=[item] * 384,
        preflight={"inputs": [{"qid": item["qid"], "routing": {}}]},
    )
    context["spec"]["workers"] = [dict(parent="b0", shard=0, qids=[item["qid"]] * 192)]
    monkeypatch.setattr(confirm_registration, "verify_registered", lambda *a, **k: context)
    calls = []

    def failure(*args):
        calls.append(args)
        raise RuntimeError("synthetic model load failure")

    monkeypatch.setattr(confirm_runtime, "load_confirm", failure)
    with pytest.raises(RuntimeError, match="model load"):
        cc.run(tmp_path, "b0", 0, "LH-SRF6-Confirm-123-b0-0")
    assert (
        read(tmp_path / "CONFIRM/failures/LH-SRF6-Confirm-123-b0-0.json")["status"]
        == "TECHNICAL_FAILURE_STOP"
    )
    with pytest.raises(PermissionError):
        cc.run(tmp_path, "b0", 0, "LH-SRF6-Confirm-124-b0-0")
    assert len(calls) == 1


def test_worker_cannot_follow_unregistered_public_order(inputs, tmp_path, monkeypatch):
    from sr_f6 import confirm_registration

    context, item, _, _ = inputs
    public = [dict(item, qid=f"synthetic-{i:04d}") for i in range(384)]
    context.update(root=tmp_path, public=public)
    context["spec"]["workers"] = [
        dict(parent="b0", shard=0, qids=[r["qid"] for r in public[::2]][::-1])
    ]
    monkeypatch.setattr(confirm_registration, "verify_registered", lambda *a, **k: context)
    with pytest.raises(PermissionError, match="frozen worker assignment"):
        cc.run(tmp_path, "b0", 0, "LH-SRF6-Confirm-123-b0-0")
    context["public"] = public[::-1]
    with pytest.raises(PermissionError, match="sorted qid order"):
        cc.run(tmp_path, "b0", 0, "LH-SRF6-Confirm-123-b0-0")
    assert not (tmp_path / "CONFIRM").exists()


def test_execution_publication_failure_is_recorded_before_other_parents(
    inputs, tmp_path, monkeypatch
):
    from sr_f6 import confirm_registration

    context, item, _, _ = inputs
    context.update(
        root=tmp_path,
        public=[item] * 384,
        preflight={"inputs": [{"qid": item["qid"], "routing": {}}]},
    )
    context["spec"]["workers"] = [dict(parent="b0", shard=0, qids=[item["qid"]] * 192)]
    monkeypatch.setattr(confirm_registration, "verify_registered", lambda *a, **k: context)
    task_id = "LH-SRF6-Confirm-123-b0-0"
    original = dict(status="orphan synthetic observation")
    write_once(tmp_path / f"CONFIRM/executions/{task_id}.json", original)
    with pytest.raises(FileExistsError):
        cc.run(tmp_path, "b0", 0, task_id)
    assert read(tmp_path / f"CONFIRM/executions/{task_id}.json") == original
    assert (tmp_path / "CONFIRM/execution_slots/b0-0.json").exists()
    assert read(tmp_path / f"CONFIRM/failures/{task_id}.json")["status"] == "TECHNICAL_FAILURE_STOP"
