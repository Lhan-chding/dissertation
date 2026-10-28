from __future__ import annotations

import copy
import hashlib
import json

import pytest

from sr_f6.common import digest, write_once
from sr_f6.interface_v3_protocol import VERSION
from sr_f6.interface_v3_screen import (
    PHASE,
    annotate,
    output_path,
    run,
    summarize_records,
    validate_group,
)
from sr_f6.runtime import PROTOCOL_VERSION, generation_recipe
from srf6_ref.design import build_bank
from srf6_ref.measurement import full_truth_response


@pytest.fixture(scope="module")
def task():
    return build_bank("DEV_CAL", 1, 2026102711)[0]["tasks"]["P0"][0]


@pytest.fixture
def spec(task):
    return dict(
        phase=PHASE,
        screen_stage=1,
        parent="b0",
        registration_sha256="1" * 64,
        source_sha256="2" * 64,
        data_hash="3" * 64,
        parent_content_hash="4" * 64,
        model_input_sha256=digest(
            {"prompt": "question", "image_file": "DATA/images/q.png", "image_sha256": "a" * 64}
        ),
        prompt_sha256=hashlib.sha256(b"question").hexdigest(),
        private_sha256=digest(task),
        seeds=[123, 456],
        **{k: task[k] for k in ("qid", "root_id", "family", "factor", "variant")},
    )


@pytest.fixture
def raw(task, spec):
    text = full_truth_response(task["world"], task["query"])
    return [
        dict(
            **{
                k: spec[k]
                for k in (
                    "phase",
                    "screen_stage",
                    "registration_sha256",
                    "source_sha256",
                    "data_hash",
                    "parent_content_hash",
                )
            },
            policy_tensor_hash="5" * 64,
            generation_status="COMPLETE",
            technical_validation_errors=[],
            execution_mode="READONLY_INTERFACE_SCREEN",
            optimizer_updates=0,
            step=0,
            policy_step=0,
            protocol="structured_v4",
            prompt_version=VERSION,
            sr_f6_protocol_version=PROTOCOL_VERSION,
            seed_semantics="independent_per_draw_torch_generator",
            group_size=2,
            group_row_index=i,
            group_seed=123,
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
            task_id="screen-execution",
            model_input={
                "prompt": "question",
                "image_file": "DATA/images/q.png",
                "image_sha256": "a" * 64,
            },
            prompt_sha256=spec["prompt_sha256"],
            sampling_parameters=generation_recipe(),
        )
        for i, seed in enumerate(spec["seeds"])
    ]


def test_strict_two_draws_labels_and_immutable_group(raw, spec, task):
    records = annotate(raw, spec, task)
    assert [r["wire_score"]["V"] for r in records] == [1, 1]
    assert [r["wire_score"]["S"] for r in records] == [1, 1]
    group = dict(status="COMPLETE", spec=spec, records=records)
    assert validate_group(group, spec, task) == records
    group["records"][0]["wire_score"]["V"] = 0
    with pytest.raises(PermissionError, match="strict rescoring"):
        validate_group(group, spec, task)


def test_eos_does_not_gain_balanced_completion(raw, spec, task):
    raw[0]["finish_reason"] = "eos"
    row = annotate(raw, spec, task)[0]
    assert row["finish_reason"] == "eos"
    assert row["wire_score"]["finished"] == 0
    assert row["wire_score"]["S"] is None
    assert row["diagnostic"]["generation_finish_reason"] == "eos"


@pytest.mark.parametrize(
    "change,error",
    [
        ({"seed": 999}, ValueError),
        ({"generation_status": "TECHNICAL_INVALID"}, PermissionError),
        ({"technical_validation_errors": ["BAD"]}, PermissionError),
        ({"optimizer_updates": 1}, PermissionError),
        ({"step": 1}, PermissionError),
        ({"phase": "CENSUS_S0"}, PermissionError),
        ({"policy_tensor_hash": None}, PermissionError),
        ({"sampler_logprobs": [float("nan")]}, ValueError),
        ({"tokens": [True], "raw_tokens": [True]}, ValueError),
        ({"tokens": []}, ValueError),
        ({"finish_reason": "length"}, ValueError),
        ({"sampling_parameters": {"temperature": 0.5}}, PermissionError),
    ],
)
def test_invalid_actual_sampling_fails_closed(raw, spec, task, change, error):
    raw[0].update(change)
    with pytest.raises(error):
        annotate(raw, spec, task)


def test_missing_duplicate_or_additional_draws_rejected(raw, spec, task):
    for rows in (raw[:1], [raw[0], raw[0]], [*raw, raw[0]]):
        with pytest.raises(ValueError):
            annotate(rows, spec, task)


def test_processor_preflight_is_enforced(raw, spec, task):
    route = dict(
        input_tensor_hash="x", processor_hash="y", chat_template_hash="z", chat_text_sha256="q"
    )
    with pytest.raises(PermissionError, match="processor"):
        annotate(raw, spec, task, preflight_route=route)
    for row in raw:
        row.update(prompt_tensor_hash="x", input_routing=route)
    assert len(annotate(raw, spec, task, preflight_route=route)) == 2


def test_output_path_rejects_symlinks_and_escapes(tmp_path):
    with pytest.raises(PermissionError):
        output_path(tmp_path, "../outside")
    (tmp_path / "bad").symlink_to(tmp_path)
    with pytest.raises(PermissionError):
        output_path(tmp_path, "bad/file")
    assert output_path(tmp_path, "SCREEN/raw/file").parent.is_dir()


def summary_rows(*, stage=1, parents=("b0", "b1", "b2"), valid_per_factor=87):
    return [
        dict(
            row_id=f"{p}/{f}/{i}",
            parent=p,
            factor=f,
            family="synthetic",
            screen_stage=stage,
            finish_reason="balanced",
            wire_score=dict(
                V=int(i < valid_per_factor),
                S=0 if i < valid_per_factor else None,
                B=0 if i < valid_per_factor else None,
                A=0,
                J=0,
            ),
            diagnostic=dict(J_core_diag=None),
        )
        for p in parents
        for f in ("P0", "P1")
        for i in range(96)
    ]


def test_pooled_stage_gate_and_factor_reports_do_not_invent_factor_pass():
    records = summary_rows()
    result = summarize_records(records)
    assert result["status"] == "SCREEN_PASS" and not result["training_allowed"]
    records[0]["wire_score"]["V"] = 0
    result = summarize_records(records)
    assert result["status"] == "SCREEN_PASS"  # 173 pooled, one factor86 is reported, not hidden.
    assert result["cells"][0]["counts"]["V"] == 86
    records[1]["wire_score"]["V"] = 0
    result = summarize_records(records)
    assert result["status"] == "STAGE2_REQUIRED"
    assert result["decision"]["stage2_parents"] == ["b0"]
    second = summary_rows(stage=2, parents=("b0",))
    final = summarize_records(second, stage=2, parents=("b0",), prior=result["decision"])
    assert final["status"] == "SCREEN_PASS"
    assert final["decision"]["parents"]["b0"]["combined_counts"]["V"] == 346
    second[0]["wire_score"]["V"] = 0
    assert (
        summarize_records(second, stage=2, parents=("b0",), prior=result["decision"])["status"]
        == "SCREEN_FAIL_STOP"
    )


def test_partial_or_cross_stage_cannot_reach_gate():
    for records in (summary_rows()[:-1], summary_rows(stage=2)):
        with pytest.raises(ValueError):
            summarize_records(records)
    with pytest.raises(ValueError):
        summarize_records(summary_rows() + summary_rows()[:1])


def _prior_fixture(tmp_path, monkeypatch, records):
    from sr_f6 import interface_v3_registration as registration
    from sr_f6 import interface_v3_screen as screen
    from sr_f6.common import file_hash

    output = tmp_path / "STAGE1"
    output.mkdir()
    context = dict(root=output, stage=1, registration_sha256="a" * 64)
    monkeypatch.setattr(registration, "verify_registered", lambda *a, **kw: context)
    accounting = dict(
        actual_completion_attempts=576, actual_generated_tokens=576, optimizer_updates=0
    )
    monkeypatch.setattr(screen, "collect_records", lambda *a: (records, accounting))
    report = screen.summarize_records(records)
    report.update(
        records_sha256=digest(records),
        accounting=accounting,
        saved_token_validation=dict(
            status="PASS", checked_responses=576, checked_tokens=576, new_model_calls=0
        ),
    )
    path = output / "SCREEN/SCREEN_ASSESSMENT.json"
    write_once(path, report)
    seal = dict(
        schema="srf6-interface-v3-assessment-seal-v1",
        screen_stage=1,
        registration_sha256="a" * 64,
        assessment_sha256=file_hash(path),
        training_allowed=False,
    )
    write_once(output / "SCREEN/ASSESSMENT_SEAL.json", seal)
    return screen, path, seal


def test_stage2_requires_complete_sealed_gray_stage_and_exact_parent(tmp_path, monkeypatch):
    records = summary_rows()
    for i in (0, 1):
        records[i]["wire_score"]["V"] = 0
    screen, _, _ = _prior_fixture(tmp_path, monkeypatch, records)
    assert screen.stage2_authorized(tmp_path) == ["b0"]
    assert screen.stage2_authorized(tmp_path, "b0") is True
    with pytest.raises(PermissionError, match="not prospectively"):
        screen.stage2_authorized(tmp_path, "b1")
    records[2]["wire_score"]["V"] = 0
    with pytest.raises(PermissionError, match="original records"):
        screen.stage2_authorized(tmp_path, "b0")


@pytest.mark.parametrize("valid", [81, 96])
def test_no_stage2_after_hard_failure_or_pass(tmp_path, monkeypatch, valid):
    screen, _, _ = _prior_fixture(tmp_path, monkeypatch, summary_rows(valid_per_factor=valid))
    with pytest.raises(PermissionError, match="non-failing gray"):
        screen.stage2_authorized(tmp_path)


def test_stage2_rejects_unsealed_changed_report(tmp_path, monkeypatch):
    records = summary_rows()
    for i in (0, 1):
        records[i]["wire_score"]["V"] = 0
    screen, path, _ = _prior_fixture(tmp_path, monkeypatch, records)
    obj = json.loads(path.read_text())
    obj["status"] = "SCREEN_PASS"
    path.write_text(json.dumps(obj))
    with pytest.raises(PermissionError, match="seal differs"):
        screen.stage2_authorized(tmp_path)


def test_run_writes_to_stage_root_and_refuses_partial_resampling(tmp_path, monkeypatch, task, raw):
    from sr_f6 import interface_v3_registration as registration
    from sr_f6 import interface_v3_runtime as runtime

    base = tmp_path
    out = base / "STAGE1"
    out.mkdir()
    item = dict(
        **{k: task[k] for k in ("qid", "root_id", "family", "factor", "variant")},
        model_input=raw[0]["model_input"],
        prompt_sha256=raw[0]["prompt_sha256"],
    )
    route = dict(
        input_tensor_hash="x", processor_hash="y", chat_template_hash="z", chat_text_sha256="q"
    )
    context = dict(
        root=out,
        registration_root=base,
        stage=1,
        registration_sha256="1" * 64,
        registration=dict(
            source_manifest_sha256="2" * 64,
            data_sha256="3" * 64,
            parent_content_hashes={"b0": "4" * 64},
        ),
        public=[item],
        private={task["qid"]: task},
        preflight=dict(inputs=[dict(qid=task["qid"], routing=route)]),
        plan=dict(
            attempts=[
                dict(parent="b0", qid=task["qid"], draw=i, seed=r["seed"])
                for i, r in enumerate(raw)
            ]
        ),
    )
    calls = []

    def verify(root, **kwargs):
        assert root == base and kwargs["stage"] == 1 and kwargs["parent"] == "b0"
        return context

    monkeypatch.setattr(registration, "verify_registered", verify)

    class Partial:
        def generate_registered(self, qid, *, on_completion):
            first = copy.deepcopy(raw[0])
            first.update(prompt_tensor_hash="x", input_routing=route)
            on_completion(first)
            raise RuntimeError("simulated interrupted native generation")

    def load(*args):
        calls.append(1)
        return Partial()

    monkeypatch.setattr(runtime, "load_screen", load)
    with pytest.raises(RuntimeError, match="interrupted"):
        run(base, "b0", 0, "first", stage=1)
    with pytest.raises(RuntimeError, match="Partial screen unit"):
        run(base, "b0", 0, "second", stage=1)
    assert len(calls) == 1
    assert len(list((out / "SCREEN/failures").glob("*.json"))) == 2
    assert not (base / "SCREEN").exists()
    assert len((out / "SCREEN/raw/b0" / f"{task['qid']}.jsonl").read_text().splitlines()) == 1


def test_bool_stage_provenance_is_rejected(raw, spec, task):
    raw[0]["screen_stage"] = True
    with pytest.raises(PermissionError):
        annotate(raw, spec, task)
