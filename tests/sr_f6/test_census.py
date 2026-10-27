from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sr_f6 import census
from srf6_ref.core import FAMILIES
from srf6_ref.design import make_root, visible_prompt
from srf6_ref.measurement import full_truth_response


@pytest.fixture
def prepared(tmp_path):
    config = json.loads((Path(__file__).parents[2] / "configs/sr_f6.json").read_text())
    (tmp_path / "CONFIG.json").write_text(json.dumps(config))
    directory = tmp_path / "DATA/DEV_CAL"
    directory.mkdir(parents=True)
    (tmp_path / "DATA/MANIFEST.json").write_text("{}")
    proto = make_root("DEV_CAL", "BINDING", 0, config["data"]["seeds"]["DEV_CAL"])["tasks"]["P0"][0]
    tasks, public = [], []
    for family in FAMILIES:
        for i in range(8):
            root_id = f"DEV_CAL_{family}_{i:04d}"
            for factor in ("P0", "P1"):
                for variant in range(4):
                    task = dict(
                        proto,
                        family=family,
                        root_id=root_id,
                        factor=factor,
                        variant=variant,
                        qid=f"{root_id}_{factor}_v{variant}",
                    )
                    item = {
                        k: task[k]
                        for k in ("qid", "root_id", "family", "factor", "variant", "split")
                    }
                    item.update(
                        protocol="structured_v4",
                        model_input=dict(
                            prompt=visible_prompt(task), image_file=None, image_sha256=None
                        ),
                    )
                    tasks.append(task)
                    public.append(item)
    for name, values in [("TASKS_PRIVATE", tasks), ("MODEL_INPUTS", public)]:
        (directory / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in values))
    return tmp_path, config


def test_complete_input_matrix_and_spec_are_exact(prepared):
    root, config = prepared
    public, tasks = census.census_inputs(root)
    assert len(public) == 384
    frozen = census.freeze_spec(root, {p: str(i) * 64 for i, p in enumerate(census.PARENTS)})
    assert frozen["expected_response_rows"] == 9216
    assert census.validate_census_spec(root, config, public, tasks) == frozen
    changed = copy.deepcopy(public)
    changed[0]["model_input"]["prompt"] += " changed"
    with pytest.raises(PermissionError):
        census.validate_census_spec(root, config, changed, tasks)
    with pytest.raises(FileExistsError):
        census.freeze_spec(root, {p: "a" * 64 for p in census.PARENTS})


def test_missing_factor_or_duplicate_view_is_rejected(prepared):
    root, _ = prepared
    path = root / "DATA/DEV_CAL/MODEL_INPUTS.jsonl"
    original = path.read_text().splitlines()
    path.write_text("\n".join(original[:-1]) + "\n")
    with pytest.raises(ValueError):
        census.census_inputs(root)


def test_group_recomputation_preserves_nulls_and_rejects_tampering(prepared):
    root, config = prepared
    public, tasks = census.census_inputs(root)
    frozen = census.freeze_spec(root, {p: "a" * 64 for p in census.PARENTS})
    item = public[0]
    task = tasks[item["qid"]]
    spec = census.group_spec(config, "b0", item, task, frozen)
    raw = [
        dict(
            seed=seed,
            generation_status="COMPLETE",
            technical_validation_errors=[],
            tokens=[1, 2],
            sampler_logprobs=[-1.0, -2.0],
            model_input=item["model_input"],
            adapter_identity={"parent": "b0", "step": 0},
            parent_content_hash=spec["policy_hash"],
            policy_tensor_hash="c" * 64,
            full_text=full_truth_response(task["world"], task["query"]),
            finish_reason="balanced",
        )
        for seed in spec["seeds"]
    ]
    wrong_parent = copy.deepcopy(raw)
    wrong_parent[0]["parent_content_hash"] = "d" * 64
    with pytest.raises(PermissionError):
        census.annotate_group(wrong_parent, task, spec)
    mixed_policy = copy.deepcopy(raw)
    mixed_policy[0]["policy_tensor_hash"] = "d" * 64
    with pytest.raises(PermissionError):
        census.annotate_group(mixed_policy, task, spec)
    missing_policy = copy.deepcopy(raw)
    for row in missing_policy:
        row.pop("policy_tensor_hash")
    with pytest.raises(PermissionError):
        census.annotate_group(missing_policy, task, spec)
    records = census.annotate_group(raw, task, spec)
    assert records[0]["A"] == records[0]["J"] == records[0]["S"] == records[0]["B"] == 1
    group = dict(
        status="COMPLETE", spec=spec, records=records, credit=census.counterfactual(records, spec)
    )
    assert census.validate_group(group, spec, task) == records
    changed = copy.deepcopy(group)
    changed["records"][0]["S"] = 0
    with pytest.raises(PermissionError):
        census.validate_group(changed, spec, task)
    changed = copy.deepcopy(group)
    changed["records"][0]["policy_hash"] = "b" * 64
    with pytest.raises(PermissionError):
        census.validate_group(changed, spec, task)
    changed = copy.deepcopy(group)
    changed["records"][0]["draw"] = 1
    with pytest.raises(ValueError):
        census.validate_group(changed, spec, task)


def test_factor_anchors_share_seed_but_remain_distinct_registered_rows(prepared):
    root, config = prepared
    public, tasks = census.census_inputs(root)
    frozen = census.freeze_spec(root, {p: "a" * 64 for p in census.PARENTS})
    p0 = next(p for p in public if p["factor"] == "P0" and p["variant"] == 0)
    p1 = next(
        p
        for p in public
        if p["factor"] == "P1" and p["root_id"] == p0["root_id"] and p["variant"] == 0
    )
    a = census.group_spec(config, "b0", p0, tasks[p0["qid"]], frozen)
    b = census.group_spec(config, "b0", p1, tasks[p1["qid"]], frozen)
    assert a["seeds"] == b["seeds"]
    assert a["group_id"] != b["group_id"]


def test_summary_rejects_incomplete_census(prepared):
    root, _ = prepared
    census.freeze_spec(root, {p: "a" * 64 for p in census.PARENTS})
    with pytest.raises(FileNotFoundError):
        census.complete_summary(root)


def test_real_token_replay_is_read_only_and_failure_is_retained(tmp_path):
    from types import SimpleNamespace

    import torch

    from mm_core.training import state_hash, trainable_state

    model = torch.nn.Linear(2, 1)
    runtime = SimpleNamespace(torch=torch, model=model)
    records = [
        dict(
            tokens=[1, 2], sampler_logprobs=[-1.0, -2.0], prompt_tensor_hash="a" * 64, row_id=str(i)
        )
        for i in range(8)
    ]
    spec = dict(parent="b0", group_id="g", frozen_spec_sha256="b" * 64)
    runtime.batch_sequence_forward = lambda batch, **kw: [
        torch.tensor(batch[0]["sampler_logprobs"])
    ]
    before = state_hash(trainable_state(model))
    output = tmp_path / "good.json"
    report = census.check_generation_alignment(runtime, records, spec, output)
    assert report["status"] == "PASS"
    assert report["diagnostics"]["token_count"] == 16
    assert state_hash(trainable_state(model)) == before
    assert census.validate_alignment_receipt(output, spec) == report
    runtime.batch_sequence_forward = lambda batch, **kw: [torch.tensor([5.0, 5.0])]
    failed = tmp_path / "failed.json"
    with pytest.raises(RuntimeError):
        census.check_generation_alignment(runtime, records, spec, failed)
    assert json.loads(failed.read_text())["status"] == "FAIL"
    assert len(json.loads(failed.read_text())["traces"]) == 8
    runtime.batch_sequence_forward = lambda batch, **kw: [
        torch.tensor(batch[0]["sampler_logprobs"])
    ]
    with pytest.raises(PermissionError):
        census.check_generation_alignment(runtime, records, spec, failed)
