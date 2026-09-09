import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from src.diagnose_repair_inputs import variant_prompt
from src.prompts import OPERATIONS, build_prompt


@pytest.fixture
def scenes():
    return [
        json.loads(line)
        for line in Path("data/generated/calibration.jsonl").read_text().splitlines()
    ]


def test_operation_intervention_removes_exactly_one_line(scenes):
    scene = scenes[0]
    original = build_prompt(scene, "SYMBOLIC_FRESH")
    modified = variant_prompt(scene, "SYM_NO_OPERATION")
    line = f"The downstream calculation is: {OPERATIONS[scene['operation']]}.\n"
    assert modified["user"] == original["user"].replace(line, "")
    assert modified["system"] == original["system"]


def test_original_conditions_and_truth_leakage(scenes):
    for scene in scenes[:18]:
        assert variant_prompt(scene, "SYM_ORIGINAL") == build_prompt(scene, "SYMBOLIC_FRESH")
        assert variant_prompt(scene, "IMAGE_CUE") == build_prompt(scene, "IMAGE_CUE_FRESH")
        hidden = {
            **scene,
            "truth_world": [91, 92, 93, 94],
            "changed_index": (scene["changed_index"] + 1) % 4,
        }
        for condition in (
            "SYM_ORIGINAL",
            "SYM_CLEAR",
            "SYM_NO_OPERATION",
            "IMAGE_CUE",
            "IMAGE_ONLY",
        ):
            assert variant_prompt(scene, condition) == variant_prompt(hidden, condition)


def test_clear_image_only_oracle_and_unknown(scenes):
    scene = next(s for s in scenes if s["constraint_family"] == "trend")
    clear = variant_prompt(scene, "SYM_CLEAR")["user"]
    assert "b - a = c - b\nc - b = d - c" in clear
    assert "other three values are correct" in clear
    only = variant_prompt(scene, "IMAGE_ONLY")
    assert "a and b are the two series" in only["user"]
    assert f"The downstream calculation is: {OPERATIONS[scene['operation']]}" in only["user"]
    assert "observed record" not in only["user"]
    assert "relationships" not in only["user"]
    assert only["image_path"] == scene["image_path"]
    assert (
        f"coordinate {'abcd'[scene['changed_index']]}"
        in variant_prompt(scene, "ORACLE_INDEX")["user"]
    )
    with pytest.raises(ValueError, match="condition"):
        variant_prompt(scene, "unknown")


def test_image_only_omits_observation_and_cue_from_prompt_identity(scenes):
    scene = scenes[0]
    changed = {**scene, "observed_world": [91, 92, 93, 94], "cue": {"family": "trend"}}
    assert variant_prompt(scene, "IMAGE_ONLY") == variant_prompt(changed, "IMAGE_ONLY")
    assert (
        variant_prompt(scene, "IMAGE_ONLY")["prompt_hash"]
        != variant_prompt({**scene, "image_hash": "changed-image"}, "IMAGE_ONLY")["prompt_hash"]
    )


def test_panel_is_hash_selected_balanced_and_order_independent(scenes):
    from src.r2_inputs import select_long_panel, select_panel

    original = copy.deepcopy(scenes)
    panel = select_panel(scenes)
    assert panel == select_panel(list(reversed(scenes)))
    assert scenes == original
    assert len(panel) == 72
    assert set(
        Counter((s["constraint_family"], s["chart_type"], s["operation"]) for s in panel).values()
    ) == {4}
    long = select_long_panel(panel)
    assert len(long) == 12
    assert set(s["base_scene_id"] for s in long) <= set(s["base_scene_id"] for s in panel)
    assert set(
        Counter((s["constraint_family"], s["chart_type"], s["operation"]) for s in long).values()
    ) == {1}
    assert set(s["constraint_family"] for s in long) == {"cross_series", "trend"}
    with pytest.raises(ValueError, match="duplicate"):
        select_panel([*scenes, scenes[0]])
    with pytest.raises(ValueError, match="calibration"):
        select_panel([{**scenes[0], "split": "dev"}, *scenes[1:]])
    with pytest.raises(ValueError, match="Insufficient"):
        select_panel(scenes[:18])


def test_request_counts_identity_and_long_reuse(scenes):
    from src.r2_inputs import build_requests, select_panel

    panel = select_panel(scenes)
    requests = build_requests(panel)
    assert requests == build_requests(list(reversed(panel)))
    assert len(requests) == len({r["sample_key"] for r in requests}) == 2232
    assert Counter((r["max_new_tokens"], r["decode_mode"]) for r in requests) == {
        (64, "sample"): 1728,
        (64, "greedy"): 432,
        (256, "sample"): 24,
        (256, "greedy"): 12,
        (1024, "sample"): 24,
        (1024, "greedy"): 12,
    }
    assert sum(r["enable_thinking"] for r in requests) == 36
    assert len({r["prompt_id"] for r in requests}) == 456
    assert json.loads(json.dumps(requests)) == requests


class Tokenizer:
    mapping: ClassVar[dict[int, str]] = {
        248068: "<think>",
        248069: "</think>",
        1: "[1,2,3,4]",
        2: "reason",
        3: "\n",
        248046: "<|im_end|>",
    }

    def convert_tokens_to_ids(self, value):
        return {v: k for k, v in self.mapping.items()}.get(value)

    def encode(self, text, add_special_tokens=False):
        return [self.convert_tokens_to_ids(text)] if text in self.mapping.values() else [99]

    def decode(self, ids, **kwargs):
        return "".join(self.mapping.get(t, "x") for t in ids)


@pytest.mark.parametrize(
    "ids,reasoning,final,resolved",
    [
        ([2, 248069, 1, 248046], "reason", "[1,2,3,4]", True),
        ([248069, 1], "", "[1,2,3,4]", True),
        ([1, 2], "[1,2,3,4]reason", None, False),
        ([1, 248046], "[1,2,3,4]", None, False),
        ([248068, 2, 248069, 1], "reason", "[1,2,3,4]", True),
        ([248068, 2], "reason", None, False),
        ([2, 248069], "reason", "", True),
    ],
)
def test_thinking_segments_require_delimiter(ids, reasoning, final, resolved):
    from src.r2_inputs import split_thinking_completion

    result = split_thinking_completion(ids, Tokenizer(), {248046})
    assert result["reasoning_text"] == reasoning
    assert result["final_text"] == final
    assert result["unresolved_final"] is not resolved
    assert result["reasoning_token_count"] + result["final_token_count"] + result[
        "delimiter_token_count"
    ] + result["eos_token_count"] == len(ids)
    assert result["completion_token_count"] == len(ids)


def test_ambiguous_thinking_segments_are_unresolved():
    from src.r2_inputs import split_thinking_completion

    result = split_thinking_completion([2, 248069, 1, 248069, 1], Tokenizer(), {248046})
    assert result["unresolved_final"]
    assert result["final_text"] is None


def test_non_thinking_prepare_is_exact_delegation():
    from src.r2_inputs import prepare_r2

    sentinel = object()
    adapter = SimpleNamespace(prepare=lambda prompt, root: sentinel)
    assert prepare_r2(adapter, {}, Path(".")) is sentinel


def test_thinking_prepare_uses_explicit_true_and_binds_inputs(monkeypatch):
    import torch

    from src import r2_inputs

    class Processor:
        chat_template = "fixed-fixture-template"
        tokenizer = Tokenizer()

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is True
            self.last_messages = messages
            return "user\n<|im_start|>assistant\n<think>\n"

        def __call__(self, **kwargs):
            assert set(kwargs) == {"text", "return_tensors", "padding"}
            return {
                "input_ids": torch.tensor([[99, 248068, 3]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            }

    monkeypatch.setattr(
        r2_inputs,
        "PINNED_CHAT_TEMPLATE_SHA256",
        hashlib.sha256(Processor.chat_template.encode()).hexdigest(),
    )
    processor = Processor()
    adapter = SimpleNamespace(processor=processor)
    prompt = {"system": "system", "user": "user 42"}
    prepared = r2_inputs.prepare_r2(adapter, prompt, ".", True)
    assert prepared["audit"]["enable_thinking"] is True
    assert prepared["audit"]["final_prompt_token_ids"] == [99, 248068, 3]
    assert prepared["audit"]["thinking_open_token_id"] == 248068
    assert prepared["audit"]["thinking_close_token_id"] == 248069
    assert prepared["audit"]["input_tensor_hash"]
    assert prepared["audit"]["image_token_count"] == 0
    with pytest.raises(ValueError, match="symbolic"):
        r2_inputs.prepare_r2(adapter, {**prompt, "image_path": "x.png"}, ".", True)
    processor.chat_template = "drifted"
    with pytest.raises(ValueError, match="template"):
        r2_inputs.prepare_r2(adapter, prompt, ".", True)


def test_panel_artifacts_record_real_diffs_and_hashes(scenes, tmp_path):
    from src.r2_inputs import select_panel, write_panel_artifacts

    panel = select_panel(scenes)
    manifest = write_panel_artifacts(panel, tmp_path)
    assert manifest["request_count"] == 2232
    assert len(manifest["rows"]) == 456
    text = (tmp_path / "prompt_diffs.md").read_text()
    assert "--- " in text and "+++ " in text
    assert "-The downstream calculation is:" in text
    assert (
        manifest["prompt_diffs_sha256"]
        == hashlib.sha256((tmp_path / "prompt_diffs.md").read_bytes()).hexdigest()
    )
    assert write_panel_artifacts(list(reversed(panel)), tmp_path) == manifest
    path = tmp_path / "prompt_diffs.md"
    path.write_text(path.read_text() + "corrupt")
    with pytest.raises(ValueError, match="diffs changed"):
        write_panel_artifacts(panel, tmp_path)
