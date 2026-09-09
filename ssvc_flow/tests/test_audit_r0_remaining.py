from pathlib import Path

import pytest

from src.audit_r0_remaining import (
    audit_cross_split,
    audit_symbolic_prompts,
    compare_contact_manifest,
    decode_raw_completion,
    select_36,
)


class Tok:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(
            {1: "[", 2: "1", 3: ",", 4: "2", 5: "]", 9: "<think>"}.get(i, "?") for i in ids
        )


def test_decode_only_trailing_eos_preserves_think():
    text, ids = decode_raw_completion([9, 1, 2, 3, 4, 5, 248046], Tok(), [248044, 248046])
    assert text == "<think>[1,2]"
    assert ids[-1] == 5


def test_cross_split_passes_generated_dataset():
    r = audit_cross_split(Path("data/generated"))
    assert r["status"] == "PASS"
    assert not r["base_scene_collisions"]


def test_hash_selection_and_reuse_manifest():
    root = Path("data/generated")
    selected, missing = select_36(root)
    assert len(selected) == 36 and not missing
    cm = Path("docs/local_evidence/P0/contact_sheets_manifest.json")
    r = compare_contact_manifest(root, cm)
    assert r["can_reuse_human_review"] is True


def test_symbolic_prompt_no_undeclared_labels():
    r = audit_symbolic_prompts(Path("data/generated"))
    assert r["status"] == "PASS"
    assert r["checked"] == 144


def test_processor_detects_visual_token_drift(monkeypatch):
    from src.audit_r0_remaining import compare_processor_to_p3
    from src.model_adapters.qwen35 import Qwen35Adapter

    metadata = dict(
        image_grid_thw=[[1, 32, 48]],
        processor_width=768,
        processor_height=512,
        image_token_count=384,
    )
    monkeypatch.setattr(Qwen35Adapter, "prepare", lambda *a: {"audit": metadata})
    monkeypatch.setattr("src.audit_r0_remaining.build_prompt", lambda *a: {})
    scene = {"base_scene_id": "dev-one"}
    old = {**scene, **metadata, "interface": "IMAGE_CUE_FRESH", "image_token_count": 385}
    result = compare_processor_to_p3([scene], [old], None, Path("."), ["dev-one"])
    assert result["status"] == "FAIL"


def test_processor_requires_every_selected_scene(monkeypatch):
    from src.audit_r0_remaining import compare_processor_to_p3
    from src.model_adapters.qwen35 import Qwen35Adapter

    metadata = dict(
        image_grid_thw=[[1, 32, 48]],
        processor_width=768,
        processor_height=512,
        image_token_count=384,
    )
    monkeypatch.setattr(Qwen35Adapter, "prepare", lambda *a: {"audit": metadata})
    monkeypatch.setattr("src.audit_r0_remaining.build_prompt", lambda *a: {})
    scenes = [{"base_scene_id": x} for x in ["dev-one", "dev-two"]]
    old = {**scenes[0], **metadata, "interface": "IMAGE_CUE_FRESH"}
    result = compare_processor_to_p3(scenes, [old], None, Path("."), ["dev-one", "dev-two"])
    assert result["status"] == "INCONCLUSIVE"


@pytest.mark.parametrize("field", ["input_tensor_hash", "image_token_count", "final_prompt_hash"])
def test_historical_replay_checks_every_record(monkeypatch, field):
    from src.finalize_r0 import replay_historical_inputs

    scene = {"base_scene_id": "dev-one"}
    metadata = {
        "input_tensor_hash": "tensor",
        "image_token_count": 384,
        "final_prompt_hash": "prompt",
    }

    class Adapter:
        def prepare(self, *args):
            return {"audit": metadata}

    monkeypatch.setattr("src.finalize_r0.INPUT_FIELDS", tuple(metadata))
    monkeypatch.setattr("src.finalize_r0.build_prompt", lambda *a: {})
    old = {**scene, **metadata, "interface": "IMAGE_CUE_FRESH"}
    rows = [{**old, "sample_key": "a"}, {**old, "sample_key": "b", field: "changed"}]
    result = replay_historical_inputs([scene], rows, Adapter(), Path("."), "N")
    assert result["status"] == "FAIL"
    assert result["checked_prompts"] == 1
    assert result["checked_records"] == 2
    assert result["failures"][0]["sample_key"] == "b"


def test_manifest_closure_rejects_changed_evidence(tmp_path):
    from src.core import file_hash, write_json
    from src.finalize_r0 import verify_manifest

    artifact = tmp_path / "samples.jsonl"
    artifact.write_text("{}\n")
    write_json(
        tmp_path / "manifest.json",
        {
            "files": [
                {
                    "path": artifact.name,
                    "bytes": artifact.stat().st_size,
                    "sha256": file_hash(artifact),
                }
            ]
        },
    )
    assert verify_manifest(tmp_path, [artifact.name])[artifact.name] == file_hash(artifact)
    artifact.write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="content mismatch"):
        verify_manifest(tmp_path, [artifact.name])


def test_human_review_is_bound_to_original_p3_confirmation(tmp_path):
    from src.core import file_hash, write_json
    from src.finalize_r0 import human_review_binding

    ids = [str(i) for i in range(36)]
    selection = {
        "can_reuse_human_review": True,
        "selected_scene_ids": ids,
        "contact_manifest_sha256": "panel",
    }
    path = tmp_path / "review.json"
    write_json(
        path,
        {
            "checked_scene_ids": ids,
            "status": "PASS",
            "contact_manifest_sha256": "panel",
            "reviewer": "fixture",
        },
    )
    runtime = {
        "P1_evidence": {
            "human_review": {
                "status": "PASS",
                "review_sha256": file_hash(path),
                "contact_manifest_sha256": "panel",
            }
        }
    }
    assert human_review_binding(selection, path, runtime)["status"] == "PASS"
    write_json(
        path, {"checked_scene_ids": ids, "status": "PASS", "contact_manifest_sha256": "other-panel"}
    )
    assert human_review_binding(selection, path, runtime)["status"] == "FAIL"


@pytest.mark.parametrize("failure", ["missing", "modified", "wrong_count"])
def test_dataset_manifest_requires_every_declared_split(tmp_path, failure):
    from src.core import file_hash, write_json
    from src.finalize_r0 import verify_dataset_manifest

    names = ("train", "control", "calibration", "dev", "confirm", "ood", "natural_pool")
    for name in names:
        (tmp_path / f"{name}.jsonl").write_text('{"base_scene_id":"fixture"}\n')
    manifest = {
        "split_sizes": {name: 1 for name in names},
        "files": {
            f"{name}.jsonl": {
                "sha256": file_hash(tmp_path / f"{name}.jsonl"),
                "bytes": (tmp_path / f"{name}.jsonl").stat().st_size,
            }
            for name in names
        },
    }
    write_json(tmp_path / "manifest.json", manifest)
    assert verify_dataset_manifest(tmp_path)["status"] == "PASS"
    if failure == "missing":
        (tmp_path / "train.jsonl").unlink()
    elif failure == "modified":
        (tmp_path / "control.jsonl").write_text("{}\n")
    else:
        manifest["split_sizes"]["confirm"] = 2
        write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError):
        verify_dataset_manifest(tmp_path)
