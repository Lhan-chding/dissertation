import pytest

from sr_f1.runtime import amended_format_decision


@pytest.mark.parametrize(
    "covered,expected",
    [
        (0, "F2_AMENDMENT_REQUIRED"),
        (460, "F2_AMENDMENT_REQUIRED"),
        (461, "PASS_90_WITH_FORMAT_REPORTING"),
        (486, "PASS_90_WITH_FORMAT_REPORTING"),
        (487, "PASS_95"),
        (512, "PASS_95"),
    ],
)
def test_confirmation_uses_exact_preregistered_integer_boundaries(covered, expected):
    assert amended_format_decision(covered, 512) == expected


@pytest.mark.parametrize("covered,total", [(240, 256), (-1, 512), (513, 512), (True, 512)])
def test_incomplete_or_invalid_confirmation_cannot_release_engine(covered, total):
    with pytest.raises(ValueError):
        amended_format_decision(covered, total)


def test_confirmation_sixteen_slots_resume_without_generating_and_reject_legacy(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import torch

    from sr_f1 import data, json_protocol, runtime
    from sr_f1.amendment import build_amended_config
    from sr_f1.contract import digest

    amendment = build_amended_config()["protocol_amendment"]
    (tmp_path / "AMENDMENT.json").write_text("{}\n")
    inputs = {f"confirm-{i:02d}": {"qid": f"confirm-{i:02d}"} for i in range(32)}
    tasks = {qid: {"pool": "FORMAT_CONFIRM"} for qid in inputs}
    monkeypatch.setattr(data, "load_inputs", lambda root: inputs)
    monkeypatch.setattr(data, "load_tasks", lambda root: tasks)
    monkeypatch.setattr(
        json_protocol,
        "score_record",
        lambda record, task: dict(L_json=1, L_answer=1, L_evidence=1, A=0, E=0, P=0, J=0),
    )
    calls = []
    routing = dict(
        protocol_amendment_id=amendment["id"],
        protocol_amendment=amendment,
        protocol="evidence_answer",
        assistant_prefill="{",
        assistant_prefill_token_ids=[90],
        base_prompt_token_count=1,
    )

    def generate(row, root, seed, on_completion):
        calls.append((row["qid"], seed))
        on_completion(
            dict(
                **json_protocol.decoded_protocol('"evidence":[],"answer":0}'),
                tokens=[5],
                raw_tokens=[5],
                old_logprobs=[-1.0],
                sampler_logprobs=[-1.0],
                prompt_token_count=2,
                completion_token_count=1,
                balanced_token_count=1,
                finish_reason="balanced",
                truncated=False,
                generation_status="COMPLETE",
                image_routing=routing,
                input_routing=routing,
            )
        )

    obj = SimpleNamespace(
        model=torch.nn.Linear(1, 1), protocol_amendment=amendment, generate_training=generate
    )
    first = runtime.format_panel(obj, tmp_path, "FORMAT_CONFIRM", "confirm", samples_per_prompt=16)
    assert first["responses"] == 512 and len(calls) == 512 and len(set(calls)) == 512
    resumed = runtime.format_panel(
        obj, tmp_path, "FORMAT_CONFIRM", "confirm", samples_per_prompt=16
    )
    assert resumed == first and len(calls) == 512
    path = tmp_path / "engineering/format/confirm/confirm-00-15.json"
    record = runtime.read_json(path)
    for key in json_protocol.AMENDMENT_RECORD_FIELDS:
        record.pop(key)
    record["input_routing"] = record["image_routing"] = {}
    record["record_hash"] = digest({k: v for k, v in record.items() if k != "record_hash"})
    runtime.atomic_json(path, record)
    with pytest.raises(PermissionError, match="metadata is incomplete"):
        runtime.format_panel(obj, tmp_path, "FORMAT_CONFIRM", "confirm", samples_per_prompt=16)
    assert len(calls) == 512
