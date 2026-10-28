"""Frozen screen accounting, strict labels, immutable recovery, and no-G8 claims."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from sr_f6.common import digest, write_once
from sr_f6.interface_protocol import VERSION
from sr_f6.interface_screen import (
    PHASE,
    annotate,
    group_spec,
    output_path,
    run,
    summarize_records,
    validate_group,
    verify_accounting,
    verify_saved_tokens,
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


def summary_rows():
    return [
        dict(
            row_id=f"{parent}/{factor}/{i}",
            parent=parent,
            factor=factor,
            root_id=f"root{i // 8}",
            engineered_collision_anchor=i // 8 < 6,
            family=f"family{i // 16}",
            finish_reason="balanced",
            wire_score={
                "V": int(i < 87),
                "S": 0 if i < 87 else None,
                "B": 0 if i < 87 else None,
                "A": 0,
                "J": 0,
            },
            diagnostic={"J_core_diag": None},
        )
        for parent in ("b0", "b1", "b2")
        for factor in ("P0", "P1")
        for i in range(96)
    ]


def test_all_18_counts_and_wrong_known_semantics():
    records = summary_rows()
    result = summarize_records(records)
    assert result["status"] == "SCREEN_PASS"
    assert len(result["cells"]) == 6
    assert len(result["family_counts"]) == 36
    assert all(cell["counts"]["S_known"] == 87 for cell in result["cells"])
    assert result["training_allowed"] is False
    assert result["complete_G8_credit"] == "NOT_APPLICABLE"
    assert result["effect_estimated"] == "NOT_RUN"
    for key in ("V", "S", "B"):
        changed = copy.deepcopy(records)
        changed[0]["wire_score"][key] = 0 if key == "V" else None
        assert summarize_records(changed)["status"] == "SCREEN_FAIL_STOP"


def test_missing_rows_or_cell_substitution_is_not_a_scientific_fail():
    records = summary_rows()
    with pytest.raises(ValueError):
        summarize_records(records[:-1])
    records[0]["parent"] = "b1"
    with pytest.raises(ValueError):
        summarize_records(records)


def test_output_path_rejects_symlinks_and_escapes(tmp_path):
    with pytest.raises(PermissionError):
        output_path(tmp_path, "../outside")
    (tmp_path / "bad").symlink_to(tmp_path)
    with pytest.raises(PermissionError):
        output_path(tmp_path, "bad/file")
    assert output_path(tmp_path, "SCREEN/raw/file").parent.is_dir()


def test_group_spec_uses_fixed_two_seeds(task, spec):
    item = {
        **{k: task[k] for k in ("qid", "root_id", "family", "factor", "variant")},
        "model_input": {"prompt": "q"},
        "prompt_sha256": "p",
    }
    context = dict(
        registration_sha256="r",
        private={task["qid"]: task},
        registration=dict(
            source_manifest_sha256="s", data_sha256="d", parent_content_hashes={"b0": "h"}
        ),
        plan={
            "attempts": [
                dict(parent="b0", qid=task["qid"], draw=i, seed=s) for i, s in enumerate((123, 456))
            ]
        },
    )
    assert group_spec(context, "b0", item)["seeds"] == [123, 456]
    context["plan"]["attempts"].append(context["plan"]["attempts"][0])
    with pytest.raises(PermissionError):
        group_spec(context, "b0", item)


def test_actual_ledger_reconciles_exact_budget_and_identity(tmp_path):
    records = [
        dict(
            task_id="task",
            parent="b0",
            policy_tensor_hash="h",
            tokens=[1, 2],
            qid=f"q{i // 2}",
            group_seed=i // 2,
            group_row_index=i % 2,
        )
        for i in range(576)
    ]
    context = {"registration_sha256": "r"}
    write_once(
        tmp_path / "runtime_identity/task.json",
        dict(
            registration_sha256="r",
            phase=PHASE,
            execution_mode="READONLY_INTERFACE_SCREEN",
            fresh_optimizer=False,
            trainable_parameters=[],
            optimizer_updates=0,
            policy_hash="h",
            parent_id="b0",
        ),
    )
    path = output_path(tmp_path, "accounting/task.jsonl")
    ledger = []
    for row in records:
        meta = dict(
            qid=row["qid"], group_seed=row["group_seed"], phase=PHASE, registration_sha256="r"
        )
        if row["group_row_index"] == 0:
            ledger.append(
                dict(
                    task_id="task",
                    parent="b0",
                    kind="completion_attempts",
                    count=2,
                    metadata=dict(meta, protocol="structured"),
                )
            )
        ledger.append(
            dict(
                task_id="task",
                parent="b0",
                kind="generated_tokens",
                count=2,
                metadata=dict(meta, row_index=row["group_row_index"]),
            )
        )
    path.write_text("\n".join(map(json.dumps, ledger)) + "\n")
    assert verify_accounting(tmp_path, records, context)["actual_completion_attempts"] == 576
    ledger[0]["count"] += 1
    path.write_text("\n".join(map(json.dumps, ledger)) + "\n")
    with pytest.raises(PermissionError, match="reservations"):
        verify_accounting(tmp_path, records, context)


def test_saved_text_and_stopping_must_match_original_tokens(monkeypatch):
    from types import SimpleNamespace

    from sr_f1 import json_protocol
    from sr_f3_ce import runtime as native

    tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda *args, **kwargs: '"answer":1}')
    monkeypatch.setattr(json_protocol, "decode_generated", lambda *args: '"answer":1}')
    monkeypatch.setattr(native, "first_boundary_token", lambda *args, **kwargs: (1, 12))
    row = dict(
        tokens=[1],
        full_text='{"answer":1}',
        generated_text_with_special_tokens='"answer":1}',
        finish_reason="balanced",
        backend_sequence_tokens=[1, 0],
        backend_padding_count=1,
    )
    assert verify_saved_tokens([row], tokenizer, [2])["checked_responses"] == 1
    bad = dict(row, full_text='{"answer":2}')
    with pytest.raises(PermissionError, match="text differs"):
        verify_saved_tokens([bad], tokenizer, [2])
    with pytest.raises(PermissionError, match="stopping"):
        verify_saved_tokens([dict(row, finish_reason="eos")], tokenizer, [2])
    with pytest.raises(PermissionError, match="padding"):
        verify_saved_tokens([dict(row, backend_sequence_tokens=[1, 3])], tokenizer, [2])


def test_launcher_refuses_external_root_before_any_write(tmp_path):
    script = Path(__file__).parents[2] / "scripts/sr_f6/interface_screen.sbatch"
    env = dict(
        os.environ,
        SRF6_CODE=str(tmp_path),
        SRF6_ROOT=str(tmp_path),
        SRF6_PARENT="b0",
        SLURM_JOB_ID="123",
        SLURM_JOB_NAME="LH-SRF6-Screen",
    )
    result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "wrong run root" in result.stderr
    assert not list(tmp_path.iterdir())


def test_launcher_cache_environment_stays_with_registered_execution():
    script = Path(__file__).parents[2] / "scripts/sr_f6/interface_screen.sbatch"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    for variable in (
        "TMPDIR",
        "TMP",
        "TEMP",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_MODULES_CACHE",
        "HF_DATASETS_CACHE",
        "HF_ASSETS_CACHE",
        "TORCH_HOME",
        "TORCH_EXTENSIONS_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
    ):
        assert variable + '="$cache/' in text


@pytest.mark.parametrize("saved_rows", [1, 2])
def test_resume_never_replaces_an_interrupted_draw(tmp_path, monkeypatch, task, raw, saved_rows):
    from sr_f6 import interface_screen_registration, interface_screen_runtime

    route = dict(
        input_tensor_hash="x", processor_hash="y", chat_template_hash="z", chat_text_sha256="q"
    )
    for row in raw:
        row.update(prompt_tensor_hash="x", input_routing=route)
    item = {
        **{key: task[key] for key in ("qid", "root_id", "family", "factor", "variant")},
        "model_input": raw[0]["model_input"],
        "prompt_sha256": raw[0]["prompt_sha256"],
    }
    context = dict(
        root=tmp_path,
        registration_sha256="1" * 64,
        public=[item],
        private={task["qid"]: task},
        preflight={"inputs": [{"qid": task["qid"], "routing": route}]},
        registration=dict(
            source_manifest_sha256="2" * 64,
            data_sha256="3" * 64,
            parent_content_hashes={"b0": "4" * 64},
        ),
        plan={
            "attempts": [
                dict(parent="b0", qid=task["qid"], draw=i, seed=s) for i, s in enumerate((123, 456))
            ]
        },
    )
    monkeypatch.setattr(interface_screen_registration, "verify_registered", lambda *a, **k: context)

    def cannot_load(*args, **kwargs):
        raise AssertionError("Saved raw recovery must never sample replacement responses")

    monkeypatch.setattr(interface_screen_runtime, "load_screen", cannot_load)
    attempt_path = output_path(tmp_path, f"SCREEN/attempts/b0/{task['qid']}.json")
    write_once(attempt_path, {"spec": group_spec(context, "b0", item)})
    raw_path = output_path(tmp_path, f"SCREEN/raw/b0/{task['qid']}.jsonl")
    original = "\n".join(json.dumps(row) for row in raw[:saved_rows]) + "\n"
    raw_path.write_text(original)
    if saved_rows == 1:
        with pytest.raises(RuntimeError, match="no automatic resampling"):
            run(tmp_path, "b0", 0, "recovery")
        assert (tmp_path / "SCREEN/failures/recovery.json").exists()
    else:
        result = run(tmp_path, "b0", 0, "recovery")
        assert result["generated_this_process"] == 0
        assert result["recovered_raw_rows"] == 2
    assert raw_path.read_text() == original
