import copy
import json
from collections import UserDict
from types import SimpleNamespace

import pytest
import torch

from mm_core.training import state_hash
from sr_f3_ce.contract import support_target
from sr_f3_ce.masks import serialize_target
from sr_f3_ce.native_audit import (
    TARGET_PIXELS,
    audit_native_inputs,
    audit_sidecar_independence,
    processor_pixel_budget,
)


class OpaqueBatch(UserDict):
    """Reproduce BatchFeature's non-dict path and hidden tensor middle values."""

    def __repr__(self):
        return "BatchFeature({'pixel_values': tensor([0., ..., 0.])})"


class FixedStore:
    image_root = "/registered/images"

    def model_input(self, qid, protocol="structured"):
        return {"prompt": "frozen " + protocol, "image_file": None, "image_sha256": None}


def sidecar_fixture():
    row = dict(
        qid="q",
        pool="ENGINE",
        root_id="root",
        answer=7,
        world={"nested": [1, 2]},
        query={"operation": "sum"},
        teacher={"target": "original"},
        certificate=[1],
    )
    return SimpleNamespace(tasks={"q": row}), FixedStore()


def test_actual_sidecar_poison_is_observed_and_original_object_restored():
    dataset, store = sidecar_fixture()
    row, nested = dataset.tasks["q"], dataset.tasks["q"]["world"]
    original, seen = copy.deepcopy(row), []

    def prepare(*args, **kwargs):
        seen.append(copy.deepcopy(dataset.tasks["q"]))
        return {
            "inputs": OpaqueBatch(pixel_values=torch.zeros(4096), input_ids=torch.tensor([[1, 2]]))
        }

    receipt = audit_sidecar_independence(
        SimpleNamespace(prepare_input=prepare), dataset, store, "q"
    )
    assert seen[0]["answer"] == 7 and seen[1]["answer"] == "POISON_GOLD"
    assert seen[1]["qid"] == "POISON_QID" and seen[1]["world"] == {"POISON_WORLD": True}
    assert dataset.tasks["q"] is row and dataset.tasks["q"]["world"] is nested
    assert dataset.tasks["q"] == original
    assert receipt["actual_sidecar_mutated"] and receipt["actual_sidecar_restored"]
    assert receipt["original_sidecar_sha256"] == receipt["restored_sidecar_sha256"]
    assert receipt["original_sidecar_sha256"] != receipt["poisoned_sidecar_sha256"]
    assert receipt["input_tensor_sha256_before"] == receipt["input_tensor_sha256_after"]


def test_tensor_middle_mutation_is_detected_even_with_identical_truncated_repr():
    dataset, store = sidecar_fixture()
    original = copy.deepcopy(dataset.tasks)
    batches = []

    def prepare(*args, **kwargs):
        tensor = torch.zeros(4096)
        if dataset.tasks["q"]["answer"] == "POISON_GOLD":
            tensor[2048] = 1
        batch = OpaqueBatch(pixel_values=tensor)
        batches.append(batch)
        return {"inputs": batch}

    with pytest.raises(PermissionError, match="changed native model inputs"):
        audit_sidecar_independence(SimpleNamespace(prepare_input=prepare), dataset, store, "q")
    assert state_hash(batches[0]) == state_hash(batches[1])  # old implementation missed this
    assert state_hash(dict(batches[0])) != state_hash(dict(batches[1]))
    assert dataset.tasks == original


def test_prompt_leak_and_processor_exception_both_restore_real_sidecar():
    dataset, _ = sidecar_fixture()
    original = copy.deepcopy(dataset.tasks)

    class LeakingStore(FixedStore):
        def model_input(self, qid, protocol="structured"):
            return {
                **super().model_input(qid, protocol),
                "prompt": str(dataset.tasks[qid]["answer"]),
            }

    runtime = SimpleNamespace(
        prepare_input=lambda *args, **kwargs: {"inputs": OpaqueBatch(ids=torch.ones(1))}
    )
    with pytest.raises(PermissionError, match="changed native model inputs"):
        audit_sidecar_independence(runtime, dataset, LeakingStore(), "q")
    assert dataset.tasks == original

    def fail_on_poison(*args, **kwargs):
        if dataset.tasks["q"]["answer"] == "POISON_GOLD":
            raise RuntimeError("processor incident")
        return {"inputs": OpaqueBatch(ids=torch.ones(1))}

    with pytest.raises(RuntimeError, match="processor incident"):
        audit_sidecar_independence(
            SimpleNamespace(prepare_input=fail_on_poison), dataset, FixedStore(), "q"
        )
    assert dataset.tasks == original


class ImageProcessor:
    patch_size = 16

    def __init__(self, minimum=TARGET_PIXELS, maximum=TARGET_PIXELS):
        self.minimum, self.maximum = minimum, maximum

    def to_dict(self):
        return {
            "size": {"shortest_edge": self.minimum, "longest_edge": self.maximum},
            "patch_size": self.patch_size,
            "merge_size": 2,
            "do_resize": True,
        }


def test_native_budget_checks_effective_size_aliases_not_only_receipt_constant():
    runtime = SimpleNamespace(
        processor=SimpleNamespace(image_processor=ImageProcessor()),
        identity={"processor_target_pixels": TARGET_PIXELS},
    )
    result = processor_pixel_budget(runtime)
    assert result["effective_min_pixels"] == result["effective_max_pixels"] == 786432
    runtime.processor.image_processor.maximum = 602112
    with pytest.raises(PermissionError, match="max pixels"):
        processor_pixel_budget(runtime)
    runtime.processor.image_processor.maximum = TARGET_PIXELS
    runtime.processor.image_processor.min_pixels = 602112
    with pytest.raises(PermissionError, match="min pixels"):
        processor_pixel_budget(runtime)


def test_current_native_qwen_image_processor_preserves_fixed_canvas_without_weights():
    from PIL import Image
    from transformers import Qwen2VLImageProcessor

    processor = Qwen2VLImageProcessor(
        min_pixels=TARGET_PIXELS,
        max_pixels=TARGET_PIXELS,
        patch_size=16,
        merge_size=2,
        temporal_patch_size=2,
    )
    runtime = SimpleNamespace(
        processor=SimpleNamespace(image_processor=processor),
        identity={"processor_target_pixels": TARGET_PIXELS},
    )
    assert processor_pixel_budget(runtime)["status"] == "PASS"
    output = processor(images=[Image.new("RGB", (1024, 768), "white")], return_tensors="pt")
    assert output["image_grid_thw"].tolist() == [[1, 48, 64]]
    assert not torch.cuda.is_initialized()


def test_full_native_audit_retains_real_negative_test_and_complete_mask_inventory(
    tmp_path, monkeypatch
):
    import sr_f3_ce.runner as runner
    from sr_f3_ce.runtime import SRF3CERuntime

    world = {
        "categories": ["January", "February"],
        "domain": [10, 90],
        "series": {"Alpha": [62, 83], "Beta": [31, 12]},
    }
    query = {"family": "CROSS", "aggregate": "mean", "value": "Alpha"}
    teachers = {
        arm: support_target(world, query, arm != "GOLD") for arm in ("GOLD", "PREFIX", "EXEC")
    }
    tasks = {}
    for pool, count in (("SUPPORT", 256), ("ENGINE", 16)):
        for i in range(count):
            qid = f"{pool}_{i:03d}"
            tasks[qid] = dict(
                qid=qid,
                pool=pool,
                root_id=qid,
                answer=7,
                world=world,
                query=query,
                teacher=teachers,
            )
    dataset, store = SimpleNamespace(tasks=tasks), FixedStore()
    store.image_root = tmp_path / "data"
    (tmp_path / "prior").mkdir()
    store.image_root.mkdir()
    identity = dict(
        processor_hash="p",
        tokenizer_hash="t",
        chat_template_hash="c",
        processor_target_pixels=TARGET_PIXELS,
    )
    (tmp_path / "prior/MODEL_ENVIRONMENT_IDENTITY.json").write_text(
        json.dumps(dict(identity, model_path="/local/frozen-model"))
    )
    (store.image_root / "RENDER_AUDIT.json").write_text(
        json.dumps(dict(pixels_per_value_unit=5.2, minimum_major_label_ink_height_px=15))
    )
    (store.image_root / "SEMANTIC_RENDER_SPOTCHECK.json").write_text(
        json.dumps({"items": [{"qid": "ENGINE_000"}]})
    )
    (store.image_root / "IMAGE_MANIFEST.jsonl").write_text("frozen image manifest fixture\n")

    class Runtime:
        def __init__(self):
            self.identity = identity
            self.processor = SimpleNamespace(image_processor=ImageProcessor())
            self.poison_observed = False

        def prepare_input(self, safe, root, protocol="structured"):
            if tasks["ENGINE_000"]["answer"] == "POISON_GOLD":
                self.poison_observed = True
            return dict(
                inputs=OpaqueBatch(
                    image_grid_thw=torch.tensor([[1, 48, 64]]),
                    pixel_values=torch.zeros(20),
                    input_ids=torch.tensor([[1, 2]]),
                ),
                routing=dict(
                    source_size=[1024, 768],
                    processed_size=[1024, 768],
                    processor_target_pixels=TARGET_PIXELS,
                ),
            )

        def freeze_support_serialization(self, cases, root):
            assert len(cases) == 816
            return dict(status="PASS", serialization="canonical", target_count=len(cases))

        def prepare_support(self, safe, root, target, arm, serialization="canonical"):
            return dict(
                tokens=[1, 2, 3],
                answer_mask=[0, 0, 1],
                work_mask=[0, int(arm != "GOLD"), 0],
                mask_audit={"actual_offsets": []},
                serialized_teacher_target=serialize_target(target, serialization),
            )

    runtime = Runtime()
    monkeypatch.setattr(runner, "dataset_and_store", lambda root: (dataset, store))
    monkeypatch.setattr(
        runner,
        "engine_schedule",
        lambda tasks: [{"support_qids": [f"ENGINE_{i:03d}" for i in range(16)]}],
    )
    monkeypatch.setattr(SRF3CERuntime, "processor_only", lambda *args, **kwargs: runtime)
    receipt = audit_native_inputs(tmp_path)
    assert runtime.poison_observed and tasks["ENGINE_000"]["answer"] == 7
    assert receipt["native_pixel_budget"]["effective_max_pixels"] == TARGET_PIXELS
    assert receipt["input_sidecar_audit"]["actual_sidecar_restored"]
    masks = json.loads((tmp_path / "native/SUPPORT_MASK_RECEIPT.json").read_text())
    assert masks["science_support_examples"] == 256 and masks["engineering_support_examples"] == 16
    assert masks["target_cases"] == 816 and masks["prefix_exec_pairs"] == 272
    assert len(masks["files"]) == 816
    assert receipt["model_calls"] == 0 and not receipt["weights_loaded"]
