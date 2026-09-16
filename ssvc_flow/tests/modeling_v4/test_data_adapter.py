import copy
import json
from collections import Counter

import pytest

from src.modeling_v4.data_adapter import build_bank_plan, build_panels, build_training_schedule


def training_prompts():
    return [
        {
            "prompt_id": f"{g}-{i}",
            "split": "train",
            "family": f"f{g // 2}",
            "interface": ("SYMBOLIC_FRESH", "IMAGE_CUE_FRESH")[g % 2],
        }
        for g in range(6)
        for i in range(96)
    ]


def test_nested_banks_keep_full_disjoint_pools_and_no_outcomes():
    prompts = training_prompts()
    before = copy.deepcopy(prompts)
    small = build_bank_plan(prompts, origin_id="o", calibration_banks=32, query_banks=16)
    full = build_bank_plan(prompts, origin_id="o", calibration_banks=96, query_banks=24)
    assert len(small["pools"]["calibration"]) == 384 and len(small["pools"]["query"]) == 192
    assert not set(small["pools"]["calibration"]) & set(small["pools"]["query"])
    assert small["banks"][:32] == full["banks"][:32]
    assert small["banks"][32:] == full["banks"][96:112]
    assert len({p for b in full["banks"] for p in b["prompt_ids"]}) == 480
    assert prompts == before


def test_source_schedule_is_seed_bound_arm_independent_and_unique():
    a = build_training_schedule(training_prompts(), 41001)
    b = build_training_schedule(training_prompts(), 41002)
    assert len(a["train_steps"]) == 128
    assert len({p for s in a["train_steps"] for p in s}) == 512
    assert a["train_steps"] != b["train_steps"]
    with pytest.raises(ValueError):
        build_bank_plan(training_prompts()[:-1], origin_id="o", calibration_banks=4, query_banks=0)


def test_panels_metadata_only_disjoint_and_fixed_prefix(monkeypatch):
    monkeypatch.setattr(
        "src.r4_inputs._n_record",
        lambda s, i, r: {
            "base_scene_id": s["base_scene_id"],
            "prompt_id": s["base_scene_id"] + i,
            "family": s["constraint_family"],
            "interface": i,
        },
    )
    scenes = [
        {
            "base_scene_id": f"{f}-{c}-{o}-{s}",
            "split": "control",
            "constraint_family": f,
            "chart_type": c,
            "operation": o,
        }
        for f in range(3)
        for c in range(3)
        for o in range(2)
        for s in range(4)
    ]
    panel = build_panels(scenes)
    assert len(panel["observation"]) == 72 and len(panel["signature"]) == 36
    assert len(panel["cross_prompt"]) == 36
    assert not {p["base_scene_id"] for p in panel["cross_prompt"]} & {
        p["base_scene_id"] for role in ("observation", "signature") for p in panel[role]
    }
    assert panel["first_map"] == panel["observation"][:24]
    assert (
        sorted(Counter((p["family"], p["interface"]) for p in panel["first_map"]).values())
        == [4] * 6
    )
    assert not {p["base_scene_id"] for p in panel["observation"]} & {
        p["base_scene_id"] for p in panel["signature"]
    }
    assert panel == build_panels(list(reversed(scenes)))
    with pytest.raises(ValueError):
        build_panels([{**scenes[0], "split": "confirm"}])


def test_prepare_inputs_resolves_original_relative_control_binding(tmp_path, monkeypatch):
    from src.modeling_v3.io import sha256_file
    from src.modeling_v4.data_adapter import prepare_inputs

    control = tmp_path / "control.jsonl"
    control.write_text("{}\n")
    parent = {
        "training_data_binding": {"files": {"control.jsonl": sha256_file(control)}},
        "paths": {"raw_dataset_root": str(tmp_path)},
        "training_plan": {"train_prompts": training_prompts()},
    }
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent))
    origins = []
    for arm in ("X_BASE", "X_VALID"):
        checkpoint = tmp_path / (arm + ".pt")
        checkpoint.write_bytes(arm.encode())
        origins.append(
            {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "identity": {"arm": arm, "step": 64},
            }
        )
    monkeypatch.setattr(
        "src.modeling_v3.vlm_campaign.build_q4_bridge_plan",
        lambda *a: {"probes": [], "null_origin": {}, "null_actions": []},
    )
    monkeypatch.setattr(
        "src.modeling_v4.data_adapter.build_panels",
        lambda rows, **kw: {"observed": rows, "root": kw["data_root"]},
    )
    result = prepare_inputs(
        {},
        {
            "parent_validated_plan": {"path": str(parent_path), "sha256": sha256_file(parent_path)},
            "bridge_origins": origins,
            "v3_probability_tolerances": {},
        },
    )
    assert result["panels"]["observed"] == [{}]
