"""Census identity, gold boundaries, and independent span-review regressions."""

import copy
import json
from pathlib import Path

import pytest

from sr_f4.common import write, write_rows
from sr_f5.calibration import (
    aggregate,
    annotate_group,
    census_inputs,
    frozen_census,
    geometry_check,
    group_spec,
    independent_review,
    tokenizer_fixtures,
    validate_group,
)
from sr_f5.runtime import Runtime, backend_protocol
from srf5_ref.credit import credits
from srf5_ref.generator import make_root
from srf5_ref.oracle import canonical_evidence, encode, evaluate_a


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


@pytest.fixture
def example():
    task = make_root("dev", "BINDING", 0, 122)[0]
    public = dict(
        qid=task["qid"],
        root_id=task["root_id"],
        protocol="structured_v3",
        model_input=dict(prompt="Frozen public question", image_file=None, image_sha256=None),
    )
    config = dict(master_seed=123, parents=["b0", "b1", "b2"])
    spec = group_spec(config, "b0", public, task)
    raw = []
    gold = evaluate_a(task["world"], task["query"])
    for i, seed in enumerate(spec["seeds"]):
        text = json.dumps(
            dict(
                evidence=canonical_evidence(task["world"], task["query"]),
                selected=gold["selected"] if i < 7 else [],
                answer=encode(gold["answer"]),
            )
        )
        ids = list(map(ord, text[1:]))
        raw.append(
            dict(
                tokens=ids,
                sampler_logprobs=[-0.1] * len(ids),
                seed=seed,
                model_input=public["model_input"],
                adapter_identity=dict(parent="b0", step=0),
                full_text=text,
                assistant_prefill="{",
                finish_reason="balanced",
                generation_status="COMPLETE",
                technical_validation_errors=[],
                token_prefix_character_ends=list(range(2, len(text) + 1)),
                prefix_character_offsets_stable=True,
                truncated=False,
            )
        )
    records = annotate_group(raw, task, spec, CharacterTokenizer())
    group = dict(
        status="COMPLETE",
        spec=spec,
        records=records,
        credit=credits(records, "LOCAL", spec["shuffle_seed"]),
    )
    return task, public, config, spec, group


def test_completed_group_recomputes_credit_and_preserves_identity(example):
    _, _, _, spec, group = example
    assert len(validate_group(group, spec)) == 8
    changed = copy.deepcopy(group)
    changed["records"][0]["seed"] += 1
    with pytest.raises(ValueError, match="registered draws"):
        validate_group(changed, spec)
    changed = copy.deepcopy(group)
    changed["credit"]["aux"][0] += 1
    with pytest.raises(ValueError, match="credit"):
        validate_group(changed, spec)


def test_partial_raw_group_cannot_be_annotated_as_complete(example):
    task, _, _, spec, group = example
    with pytest.raises(ValueError, match="partial"):
        annotate_group(group["records"][:7], task, spec, CharacterTokenizer())


def test_independent_review_detects_semantic_and_mask_tampering(example):
    task, _, _, _, group = example
    row = group["records"][0]
    assert independent_review(row, task)["status"] == "PASS"
    wrong = copy.deepcopy(row)
    wrong["S"] = "FAIL"
    assert "independent_selection_disagreement" in independent_review(wrong, task)["errors"]
    wrong = copy.deepcopy(row)
    wrong["mask"][0] = True
    assert "native_prefix_mask_disagreement" in independent_review(wrong, task)["errors"]
    wrong = copy.deepcopy(row)
    wrong["V"] = 0
    assert "independent_structure_disagreement" in independent_review(wrong, task)["errors"]


def test_known_wrong_selection_remains_reviewable(example):
    task, _, _, _, group = example
    row = group["records"][-1]
    assert row["S"] == "FAIL" and row["V"] == 1
    assert independent_review(row, task)["status"] == "PASS"


def test_tokenizer_fixtures_require_no_model_calls():
    result = tokenizer_fixtures(CharacterTokenizer())
    assert result["status"] == "PASS" and result["model_calls"] == 0
    assert len(result["cases"]) == 9


def test_processor_geometry_uses_actual_scaling():
    geometry = dict(pixels_per_value_unit=5.2, minimum_major_label_ink_height_px=15)
    prepared = dict(routing=dict(source_size=[1024, 768], processed_size=[1024, 768]))
    assert geometry_check(prepared, geometry)["status"] == "PASS"
    prepared["routing"]["processed_size"] = [512, 384]
    assert geometry_check(prepared, geometry)["status"] == "FAIL"


def test_protocol_alias_isolated_from_native_module():
    from sr_f3_ce.runtime import PROTOCOLS

    assert backend_protocol("structured_v3") == "structured"
    assert "structured_v3" not in PROTOCOLS
    with pytest.raises(PermissionError):
        backend_protocol("unregistered")


def test_runtime_record_validation_maps_alias_without_mutation():
    runtime = Runtime.__new__(Runtime)
    runtime.protocol_revision = "v2"
    row = dict(protocol="structured_v3", protocol_revision="v2", assistant_prefill="{")
    runtime._validate_record_revision(row)
    assert row["protocol"] == "structured_v3"


def test_census_rejects_extra_gold_at_model_boundary(tmp_path):
    from srf5_ref.generator import make_bank

    tasks = make_bank("dev", 8, 11)
    public = [
        dict(
            qid=t["qid"],
            root_id=t["root_id"],
            protocol="structured_v3",
            model_input=dict(prompt="question", image_file=None, image_sha256=None),
        )
        for t in tasks
    ]
    public[0]["model_input"]["gold_answer"] = 1
    write_rows(tmp_path / "DATA/dev/TASKS_PRIVATE.jsonl", tasks)
    write_rows(tmp_path / "DATA/dev/MODEL_INPUTS.jsonl", public)
    with pytest.raises(PermissionError, match="Private metadata"):
        census_inputs(tmp_path)


def test_census_without_frozen_registration_cannot_launch(tmp_path, example):
    _, public, config, _, _ = example
    write(tmp_path / "G2_SPEC_FROZEN.json", {})
    with pytest.raises(PermissionError, match="Frozen census"):
        frozen_census(tmp_path, config, [public])


def test_aggregate_missing_groups_never_passes(tmp_path, monkeypatch, example):
    task, public, config, _, _ = example
    import sr_f5.calibration as module

    write(tmp_path / "CONFIG.json", config)
    monkeypatch.setattr(module, "census_inputs", lambda _: ([public], {task["qid"]: task}))
    monkeypatch.setattr(module, "frozen_census", lambda *args: {})
    report = aggregate(tmp_path)
    assert report["actual_response_count"] == 0
    assert report["gate"]["status"] == "NOT_COMPLETE"
    assert len(report["missing_groups"]) == 3


def test_no_gpu_fixture_uses_character_tokenizer_only_in_tests():
    source = Path("src/sr_f5/calibration.py").read_text()
    assert "CharacterTokenizer" not in source
