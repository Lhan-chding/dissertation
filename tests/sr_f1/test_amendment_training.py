"""SR-F1.1 optimizer, raw-protocol and completion-boundary regression checks."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from mm_core.training import trainable_state
from sr_f1 import training
from sr_f1.contract import PACKAGE, gold_evidence, score
from sr_f1.evaluation import evaluate_slots, score_response, validate_raw
from sr_f1.json_protocol import AMENDMENT_ID, decoded_protocol
from sr_f1.runtime import reference_parameters


def amended_runtime():
    return SimpleNamespace(
        model=torch.nn.Linear(2, 2, bias=False),
        protocol_amendment={"id": AMENDMENT_ID},
        training_learning_rate=1e-4,
    )


def base_config(runtime):
    optimizer = torch.optim.AdamW(runtime.model.parameters(), lr=1e-5, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return optimizer, scheduler, {"fixture": "real_CPU_AdamW"}


def test_amendment_changes_actual_optimizer_and_scheduler_rate_and_binds_resume(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(training, "configure_training", base_config)
    runtime = amended_runtime()
    optimizer, scheduler, identity = training.configure_scientific_training(runtime)
    reference = trainable_state(runtime.model)
    assert identity["learning_rate"] == 1e-4
    assert optimizer.param_groups[0]["lr"] == scheduler.get_last_lr()[0] == 1e-4
    runtime.model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    state = training.checkpoint_state(
        runtime,
        optimizer,
        scheduler,
        run_identity=identity,
        step=1,
        reference=reference,
        stream_hash="fixture",
        sampling_hash="fixture",
    )
    training.commit_checkpoint(tmp_path / "valid", state)
    new_optimizer, new_scheduler, _ = training.configure_scientific_training(runtime)
    restored = training.load_checkpoint(
        tmp_path / "valid",
        runtime,
        new_optimizer,
        new_scheduler,
        run_identity=identity,
        reference=reference,
        stream_hash="fixture",
        sampling_hash="fixture",
    )
    assert restored["optimizer"]["state"]
    assert new_optimizer.param_groups[0]["lr"] == new_scheduler.get_last_lr()[0] == 1e-4
    # Hash-valid complete state with an unauthorized optimizer/scheduler rate is rejected.
    for name in ("optimizer", "scheduler"):
        changed = copy.deepcopy(state)
        if name == "optimizer":
            changed[name]["param_groups"][0]["lr"] = 1e-5
        else:
            changed[name]["base_lrs"] = [1e-5]
        training.commit_checkpoint(tmp_path / name, changed)
        with pytest.raises(PermissionError, match="learning rate"):
            training.load_checkpoint(
                tmp_path / name,
                runtime,
                new_optimizer,
                new_scheduler,
                run_identity=identity,
                reference=reference,
                stream_hash="fixture",
                sampling_hash="fixture",
            )
    legacy = SimpleNamespace(model=torch.nn.Linear(2, 2))
    old_optimizer, old_scheduler, old_identity = training.configure_scientific_training(legacy)
    assert old_optimizer.param_groups[0]["lr"] == old_scheduler.get_last_lr()[0] == 1e-5
    assert old_identity == {"fixture": "real_CPU_AdamW"}


def raw_record(generated_text, *, tokens=None, unbalanced=False):
    tokens = [1, 2] if tokens is None else tokens
    routing = dict(
        protocol_amendment_id=AMENDMENT_ID,
        protocol_amendment={"id": AMENDMENT_ID},
        protocol="evidence_answer",
        assistant_prefill="{",
        assistant_prefill_token_ids=[90],
        base_prompt_token_count=10,
    )
    return dict(
        **decoded_protocol(generated_text),
        tokens=tokens,
        raw_tokens=tokens[:],
        old_logprobs=[-1.0] * len(tokens),
        sampler_logprobs=[-1.0] * len(tokens),
        completion_token_count=len(tokens),
        balanced_token_count=None if unbalanced else len(tokens),
        finish_reason="length" if unbalanced else "balanced",
        truncated=unbalanced,
        input_routing=routing,
        image_routing=routing,
        prompt_token_count=11,
        generation_status="COMPLETE",
        technical_validation_errors=[],
    )


def test_failclosed_reward_scoring_and_per_round_format_accounting():
    with (PACKAGE / "manifests/TASKS_GOLD_AUDIT_ONLY.jsonl").open() as stream:
        task = json.loads(next(stream))
    text = json.dumps(
        {"evidence": gold_evidence(task["world"], task["query"]), "answer": task["answer"]},
        ensure_ascii=False,
    )
    valid = raw_record(text[1:])
    trailing = raw_record(text[1:] + "\n```")
    unbalanced = raw_record(text[1:-1], unbalanced=True)
    passed = score_response(valid["raw_text"], task, "evidence_answer", record=valid)
    assert passed == score(text, task["world"], task["query"])
    failed = []
    for record, reason in ((trailing, "trailing_after_close"), (unbalanced, "unbalanced")):
        scored = score_response(record["raw_text"], task, "evidence_answer", record=record)
        assert scored["reason"] == reason
        assert all(scored[key] == 0 for key in ("A", "J", "E", "p_read", "L_json"))
        failed.append(scored)
    assert training.format_failure_accounting([[passed, *failed]]) == dict(
        unit="generated_sequence",
        denominator=3,
        numerator=2,
        format_failure_rate=2 / 3,
        reasons={"trailing_after_close": 1, "unbalanced": 1},
    )
    # The raw classification is ordinary format failure, not technical missingness.
    slot = {"protocol": "evidence_answer"}
    for record in (valid, trailing, unbalanced):
        record.update(
            slot, status="GENERATED", input_hash="i", model_identity="m", adapter_identity="a"
        )
        validate_raw(record, slot, protocol_amendment_id=AMENDMENT_ID)
    changed = dict(valid, protocol_amendment_id="legacy")
    with pytest.raises(ValueError, match="amendment identity"):
        validate_raw(changed, slot, protocol_amendment_id=AMENDMENT_ID)


def test_evaluation_protocol_explicit_and_amended_resume_rejects_legacy(tmp_path):
    class Runtime:
        def __init__(self):
            self.calls = []
            self.identity = {"model_hash": "fixed"}
            self.adapter_identity = "adapter"
            self.protocol_amendment = {"id": AMENDMENT_ID}

        def generate(self, **kwargs):
            self.calls.append(kwargs["protocol"])
            if kwargs["protocol"] == "evidence_answer":
                raw = raw_record('"evidence":[],"answer":null}')
            else:
                raw = dict(
                    raw_text='{"answer":null}', tokens=[1], old_logprobs=[-1.0], truncated=False
                )
            kwargs["on_completion"](raw)

    slots = [
        dict(
            slot_id=protocol, protocol=protocol, qid="q", view=None, sampling_seed=3, generation={}
        )
        for protocol in ("evidence_answer", "answer_only")
    ]
    inputs = {"q": dict(text="evidence", plain_text="answer", image_file=None)}
    runtime = Runtime()
    path = tmp_path / "raw.jsonl"
    assert evaluate_slots(runtime, slots, inputs, {}, tmp_path, path)["status"] == "COMPLETE"
    assert runtime.calls == ["evidence_answer", "answer_only"]
    assert evaluate_slots(runtime, slots, inputs, {}, tmp_path, path)["status"] == "COMPLETE"
    assert len(runtime.calls) == 2
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0].pop("protocol_amendment_id")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="metadata is incomplete"):
        evaluate_slots(runtime, slots, inputs, {}, tmp_path, path)
    assert len(runtime.calls) == 2


def test_loss_uses_last_balanced_generated_token_without_prompt_prefill_or_synthetic_eos():
    class Runtime:
        def __init__(self):
            self.model = torch.nn.Linear(2, 2, bias=False)
            self.calls = []

        def prepare(self, row, root):
            return {"prompt_ids": [90], "x": torch.tensor(row["x"])}

        def reserve(self, *args, **kwargs):
            pass

        def sequence_forward(self, prepared, tokens, *, purpose, grad=False):
            # Vocabulary only has IDs 0 and 1. Token 90 is prompt-only; no EOS ID exists.
            assert tokens == [0, 1]
            self.calls.append((tuple(tokens), purpose))
            with torch.enable_grad() if grad else torch.no_grad():
                return {"logprobs": self.model(prepared["x"]).log_softmax(-1)[tokens]}

        def reference_forward(self, prepared, tokens, reference):
            with reference_parameters(self.model, reference):
                return self.sequence_forward(prepared, tokens, purpose="reference")

    runtime = Runtime()
    optimizer, scheduler, _ = base_config(runtime)
    reference = trainable_state(runtime.model)
    examples = []
    for i in range(128):
        row = {"x": [1.0, i / 128]}
        old = runtime.sequence_forward(runtime.prepare(row, None), [0, 1], purpose="sampler")[
            "logprobs"
        ].tolist()
        examples.append(
            dict(
                row=row,
                root=None,
                advantage=(-1.0) ** i,
                record=dict(tokens=[0, 1], old_logprobs=old),
            )
        )
    result = training.update(runtime, optimizer, scheduler, reference, examples)
    assert result["completion_tokens"] == 256
    assert result["sampler_teacher_forcing"]["token_count"] == 256
    assert len(runtime.calls) == 384


def test_unapplied_technical_invalid_raw_is_retained_unscorable_in_partial_release(tmp_path):
    from sr_f1.contract import digest
    from sr_f1.release import audit_training

    with (PACKAGE / "manifests/TASKS_GOLD_AUDIT_ONLY.jsonl").open() as stream:
        task = json.loads(next(stream))
    run_id = "SRF1_A_s71001"
    directory = tmp_path / "training" / run_id / "rollouts"
    directory.mkdir(parents=True)
    raw = dict(
        run_id=run_id,
        qid=task["qid"],
        logical_step=1,
        request_id="failed-request",
        protocol_amendment_id=AMENDMENT_ID,
        generation_status="TECHNICAL_INVALID",
        technical_validation_errors=["NONFINITE_LOGPROB"],
        raw_text="retained partial text",
    )
    raw["record_hash"] = digest(raw)
    (directory / "01-00-0.json").write_text(json.dumps(raw))
    summary = audit_training(
        tmp_path,
        {run_id: dict(arm="A", status="TECHNICAL_FAILED", failure="NONFINITE_LOGPROB")},
        {task["qid"]: task},
    )
    assert summary[0]["retained_raw_records"] == 1
    assert summary[0]["verified_applied_generations"] == 0
    audit = json.loads((tmp_path / summary[0]["raw_recount_path"]).read_text())
    assert audit["independent_score"] is None
    assert audit["technical_unscorable"] is True
    assert audit["technical_errors"] == ["NONFINITE_LOGPROB"]
    assert audit["applied_update"] is False


def test_amendment_report_uses_verified_numerators_and_discloses_archived_bridge(tmp_path):
    from sr_f1.amendment import ORIGINAL_FREEZE_SHA256
    from sr_f1.evaluation import sha_file
    from sr_f1.release import ReleaseBlocked, amendment_report_lines

    amendment = dict(
        amendment_id=AMENDMENT_ID,
        original_freeze_sha256=ORIGINAL_FREEZE_SHA256,
        protocol_amendment={"id": AMENDMENT_ID},
    )
    (tmp_path / "AMENDMENT.json").write_text(json.dumps(amendment))
    (tmp_path / "EXECUTION_FREEZE.json").write_text('{"status":"FROZEN"}')
    common = dict(
        status="COMPLETE",
        bridge_executed=False,
        bridge=None,
        format_before=dict(covered=248, responses=256, coverage=248 / 256),
        format_confirmation=dict(covered=485, responses=512, coverage=485 / 512),
        metadata=dict(
            protocol_amendment={"id": AMENDMENT_ID},
            protocol_amendment_sha256=sha_file(tmp_path / "AMENDMENT.json"),
            old_bridge_used_for_scientific_start=False,
            format_gate_decision="PASS_90_WITH_FORMAT_REPORTING",
        ),
    )
    receipts = {"AMENDMENT.json": amendment, "FORMAT_AND_BRIDGE_RECEIPT.json": common}
    report = "\n".join(amendment_report_lines(tmp_path, receipts))
    assert "248/256" in report and "485/512" in report
    assert "128条gold" in report and "未用于本轮科学起点" in report
    assert ORIGINAL_FREEZE_SHA256 in report
    assert sha_file(tmp_path / "AMENDMENT.json") in report
    common["format_confirmation"]["coverage"] = 0.95
    with pytest.raises(ReleaseBlocked, match="numerator/denominator"):
        amendment_report_lines(tmp_path, receipts)
