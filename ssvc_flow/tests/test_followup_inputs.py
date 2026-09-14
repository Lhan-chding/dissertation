"""CPU fixtures use metadata shapes, never pretend to recover parent raw rows."""

import copy
import json
from pathlib import Path

import pytest

from src.core import canonical_hash

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def bank_fixture():
    from src.followup_parent import load_parent_metadata
    from src.r3_runtime import build_requests

    docs = load_parent_metadata(ROOT / "reports/SSVC_GPT_PRO_DELIVERY_20260914")
    r, b, c = (docs[k] for k in ("runtime_lock", "bank_manifest", "candidate_manifest"))
    for identity in (r["identity"], b["identity"], b["request_identity"]):
        identity["execution_kind"] = "CPU_FAKE_ADAPTER_FIXTURE"
    b["requests"] = build_requests(b["plan"], b["request_identity"])
    for item in c["banks"].values():
        item["identity"]["execution_kind"] = "CPU_FAKE_ADAPTER_FIXTURE"
        for candidate in item["candidates"]:
            candidate["checkpoint_identity"]["execution_kind"] = "CPU_FAKE_ADAPTER_FIXTURE"
    rows = []
    prompt_map = {p["prompt_id"]: p for p in b["plan"]["train_prompts"]}
    for request in b["requests"]:
        if request["bank_role"] != "train":
            continue
        prompt = prompt_map[request["prompt_id"]]
        counts = next(
            p["counts"]
            for p in c["banks"][str(request["bank_index"])]["group_composition"]["prompts"]
            if p["prompt_id"] == request["prompt_id"]
        )
        categories = [k for k in ("X", "S", "W", "I") for _ in range(counts[k])]
        category = categories[request["sample_index"]]
        row = {
            **request,
            "execution_kind": "CPU_FAKE_ADAPTER_FIXTURE",
            "category": category,
            "model_id": r["config"]["model"]["id"],
            "model_revision": r["config"]["model"]["revision"],
            "adapter_hash": r["initial_adapter_hash"],
            "train_seed": 17,
            "protocol_version": r["identity"]["protocol_version"],
            "optimizer_state_hash": r["optimizer_initial_hash"],
            "token_ids": [1, 2, 3],
            "raw_token_ids": [1, 2, 3],
            "old_logprobs": [-1.0, -2.0, -3.0],
            "behavior_token_logprobs": [-1.0, -2.0, -3.0],
            "per_token_logprob_behavior": [-1.0, -2.0, -3.0],
            "n_generated_tokens": 3,
            "raw_text": "CPU fixture only",
            "raw_completion": "CPU fixture only",
            "generation_config_hash": canonical_hash(r["config"]["generation_proposed_N"]),
            "final_prompt_hash": canonical_hash([prompt["prompt_id"], "final"]),
            "tokenized_prompt_hash": canonical_hash([prompt["prompt_id"], "tokens"]),
            "input_ids_hash": canonical_hash([prompt["prompt_id"], "tokens"]),
            "input_tensor_hash": canonical_hash([prompt["prompt_id"], "input"]),
            "prepared_hash": canonical_hash([prompt["prompt_id"], "prepared"]),
            "image_hash": prompt["scene"]["image_hash"]
            if prompt["interface"] == "IMAGE_CUE_FRESH"
            else None,
            "execution_checks": {"passed": True, "faults": []},
        }
        row["record_hash"] = canonical_hash(row)
        rows.append(row)
    for index, item in c["banks"].items():
        ordered = sorted(
            (row for row in rows if row["bank_index"] == int(index)),
            key=lambda row: (
                b["plan"]["banks"][int(index)].index(row["prompt_id"]),
                row["sample_index"],
            ),
        )
        digest = canonical_hash([row["record_hash"] for row in ordered])
        item["identity"]["sample_hash"] = digest
        for candidate in item["candidates"]:
            candidate["checkpoint_identity"]["sample_hash"] = digest
    return docs, rows


def _build(docs, rows, index=3, **kw):
    from src.followup_inputs import build_followup_bank

    config = json.loads(
        (ROOT / "ssvc_flow/docs/mechanism_followup/design/configs/followup_design.json").read_text()
    )
    spec = (
        next(u for u in config["S1"]["units"] if u.get("parent_bank_index") == index)
        if isinstance(index, int)
        else config["S1"]["units"][-1]
    )
    return build_followup_bank(
        docs["bank_manifest"],
        rows,
        spec,
        candidate_manifest=docs["candidate_manifest"],
        runtime_lock=docs["runtime_lock"],
        **kw,
    )


@pytest.mark.parametrize("index", [0, 3, 4, 5, 11, "composite"])
def test_fixed_banks_composite_preserve_original_rows(bank_fixture, index):
    docs, rows = bank_fixture
    before = copy.deepcopy(rows)
    bank = _build(docs, rows, index)
    assert len(bank["groups"]) == 4 and all(len(g) == 8 for g in bank["groups"])
    assert len(set(bank["prompt_ids"])) == 4
    lookup = {r["sample_key"]: r for r in rows}
    assert all(row == lookup[row["sample_key"]] for group in bank["groups"] for row in group)
    assert rows == before
    assert bank["execution_kind"] == "CPU_FAKE_ADAPTER_FIXTURE"
    if index == "composite":
        assert [(g["bank"], g["group"]) for g in bank["source_groups"]] == [
            (3, 2),
            (11, 2),
            (0, 0),
            (0, 1),
        ]
    bank["groups"][0][0]["token_ids"][0] = 99
    assert rows == before


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate",
        "missing",
        "sample_index",
        "origin",
        "model",
        "parser",
        "prepared",
        "image",
        "tokens",
        "old",
        "nan",
        "hash",
        "control",
        "data",
        "category",
        "bool_index",
        "tokenized",
    ],
)
def test_bank_rejects_integrity_binding_and_leakage_faults(bank_fixture, fault):
    original_docs, original_rows = bank_fixture
    docs, rows = copy.deepcopy(original_docs), copy.deepcopy(original_rows)
    row = next(r for r in rows if r["bank_index"] == 3)
    if fault == "duplicate":
        rows.append(copy.deepcopy(row))
    elif fault == "missing":
        rows.remove(row)
    elif fault == "sample_index":
        row["sample_index"] = 7
    elif fault == "origin":
        row["origin_state_hash"] = "0" * 64
    elif fault == "model":
        row["model_revision"] = "other"
    elif fault == "parser":
        row["protocol_version"] = "other"
    elif fault == "prepared":
        row["prepared_hash"] = "0" * 64
    elif fault == "image":
        row["image_hash"] = "0" * 64
    elif fault == "tokens":
        row["token_ids"][0] = 19
    elif fault == "old":
        row["old_logprobs"][0] = -19
    elif fault == "nan":
        row["old_logprobs"][0] = float("nan")
    elif fault == "hash":
        row["record_hash"] = "0" * 64
    elif fault == "control":
        docs["bank_manifest"]["plan"]["control_prompts"][0]["base_scene_id"] = row["base_scene_id"]
    elif fault == "data":
        row["scene_hash"] = "0" * 64
    elif fault == "category":
        row["category"] = "W"
    elif fault == "bool_index":
        row["sample_index"] = True
    else:
        row["tokenized_prompt_hash"] = "0" * 64
    if fault not in ("hash", "nan"):
        row["record_hash"] = canonical_hash({k: v for k, v in row.items() if k != "record_hash"})
    with pytest.raises(ValueError):
        _build(docs, rows)


def test_bank_selection_does_not_use_new_outcomes(bank_fixture):
    docs, rows = bank_fixture
    changed = copy.deepcopy(docs)
    changed["bank_manifest"]["new_control_results"] = {"bank3": "bad", "bank6": "good"}
    assert _build(docs, rows) == _build(changed, rows)


def test_missing_raw_rejected(bank_fixture):
    with pytest.raises(ValueError, match=r"missing|complete|raw"):
        _build(bank_fixture[0], [])


def test_seed17_exact_and_new_seed_schedules_deterministic():
    from src.followup_inputs import build_training_schedule
    from src.r4_inputs import build_train_schedule

    scenes = [
        json.loads(line)
        for line in (ROOT / "ssvc_flow/data/generated/train.jsonl").read_text().splitlines()
    ]
    before = copy.deepcopy(scenes)
    assert build_training_schedule(scenes, seed=17) == build_train_schedule(scenes)
    schedules = [build_training_schedule(scenes, seed=seed) for seed in (17, 29, 41)]
    for seed, schedule in zip((17, 29, 41), schedules, strict=True):
        assert schedule == build_training_schedule(list(reversed(scenes)), seed=seed)
        assert len(schedule["train_steps"]) == 64
        assert len({p for step in schedule["train_steps"] for p in step}) == 256
    assert len({canonical_hash(s["train_steps"]) for s in schedules}) == 3
    assert scenes == before
    with pytest.raises(ValueError):
        build_training_schedule(scenes, seed=True)
    with pytest.raises(ValueError):
        build_training_schedule(scenes, seed=17, preserve_legacy17=False)


def test_prepared_attachment_checks_actual_tensor_bytes(bank_fixture):
    import torch

    from src.optimizer_fork import state_hash

    docs, rows = copy.deepcopy(bank_fixture)
    pids = docs["bank_manifest"]["plan"]["banks"][3]
    prepared = {}
    for pid in pids:
        group = [r for r in rows if r["prompt_id"] == pid]
        audit = {
            k: group[0][k]
            for k in ("final_prompt_hash", "tokenized_prompt_hash", "input_tensor_hash")
        }
        prepared[pid] = {"audit": audit, "inputs": {"input_ids": torch.tensor([[1, 2]])}}
        for row in group:
            row["prepared_hash"] = state_hash(prepared[pid])
            row["record_hash"] = canonical_hash(
                {k: v for k, v in row.items() if k != "record_hash"}
            )
    summary = docs["candidate_manifest"]["banks"]["3"]
    digest = canonical_hash(
        [r["record_hash"] for pid in pids for r in rows if r["prompt_id"] == pid]
    )
    summary["identity"]["sample_hash"] = digest
    for candidate in summary["candidates"]:
        candidate["checkpoint_identity"]["sample_hash"] = digest
    assembled = _build(docs, rows, prepared_inputs=prepared)
    assert assembled["prepared_inputs_verified"]
    assert all("prepared" in row for group in assembled["groups"] for row in group)
    assembled["groups"][0][0]["prepared"]["inputs"]["input_ids"][0, 0] = 999
    assert prepared[pids[0]]["inputs"]["input_ids"][0, 0] == 1
    prepared[pids[0]]["inputs"]["input_ids"][0, 0] = 998
    with pytest.raises(ValueError, match=r"prepared"):
        _build(docs, rows, prepared_inputs=prepared)


def test_metadata_import_and_schedule_do_not_import_model_libraries():
    import subprocess
    import sys

    script = """
import sys
from src.followup_parent import audit_parent_evidence
from src.followup_inputs import build_training_schedule
assert not any(k in sys.modules for k in ('torch','transformers','peft'))
"""
    subprocess.run([sys.executable, "-c", script], check=True)
