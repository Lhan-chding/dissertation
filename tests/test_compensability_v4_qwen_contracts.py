"""RED fake-runtime contracts for Qwen-specific Phase 1--3 infrastructure."""

from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import pytest

from compensability_v4.qwen.cache_continuation import (
    CachedGenerationState,
    assert_cache_parity,
    build_suffix_token_ids,
)
from compensability_v4.qwen.candidate_scoring import (
    find_single_token_labels,
    score_candidate_labels,
)
from compensability_v4.qwen.interface_runner import InterfaceName, interface_family
from compensability_v4.qwen.introspect_model import introspect_model
from compensability_v4.qwen.layerwise_assimilation import validate_final_layer_logits
from compensability_v4.qwen.model_loader import (
    MODEL_PATH,
    MODEL_SNAPSHOT_SHA256,
    require_server_model,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.table = {"A": [11], "B": [12], "C": [13], "D": [14], "AA": [11, 11]}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return self.table.get(text, [99, 99])


class FakeProcessor:
    tokenizer = FakeTokenizer()


class FakeModel:
    def next_token_logits(self, prompt: str) -> dict[int, float]:
        assert prompt == "prompt"
        return {11: -1.0, 12: 2.5, 13: 0.0, 14: 1.5}


def test_candidate_labels_are_single_token_and_world_mapping_survives_order_changes() -> None:
    labels = find_single_token_labels(
        FakeTokenizer(), candidates=["AA", "D", "B", "A", "C"], minimum=4
    )
    assert labels == ("D", "B", "A", "C")
    worlds = [(8, 4, 5, 6), (9, 4, 5, 6), (7, 4, 5, 6), (10, 4, 5, 6)]

    first = score_candidate_labels(FakeModel(), FakeProcessor(), "prompt", labels, worlds)
    second = score_candidate_labels(
        FakeModel(),
        FakeProcessor(),
        "prompt",
        tuple(reversed(labels)),
        tuple(reversed(worlds)),
    )

    assert first == second
    assert first[(9, 4, 5, 6)] == 2.5


def test_last_layer_candidate_logits_must_match_standard_forward() -> None:
    validate_final_layer_logits([{"A": -1.0}, {"A": 2.0, "B": 1.0}], {"A": 2.0, "B": 1.0})
    with pytest.raises(RuntimeError, match="final-layer"):
        validate_final_layer_logits([{"A": 1.9, "B": 1.0}], {"A": 2.0, "B": 1.0})


def test_cache_suffix_and_state_provenance_are_exact_and_immutable() -> None:
    assert build_suffix_token_ids([1, 2, 3, 4], [1, 2, 3, 4, 8, 9]) == (8, 9)
    with pytest.raises(ValueError, match="prefix"):
        build_suffix_token_ids([1, 2, 3], [1, 5, 3, 8])

    state = CachedGenerationState(
        sample_id="scene-1",
        token_ids=(1, 2, 3),
        attention_mask=(1, 1, 1),
        position_ids=((0, 1, 2),),
        image_token_positions=(1,),
        image_grid_thw=(1, 20, 20),
        visual_token_count=100,
        generation_config=MappingProxyType({"do_sample": False}),
        rng_seed=20260817,
        past_key_values=object(),
    )
    assert state.sample_id == "scene-1"
    with pytest.raises(TypeError):
        state.generation_config["do_sample"] = True  # type: ignore[index]


def test_cache_parity_is_an_execution_gate() -> None:
    assert_cache_parity((4, 5, 6), (4, 5, 6))
    with pytest.raises(RuntimeError, match="parity"):
        assert_cache_parity((4, 5, 6), (4, 5, 7))


def test_interface_names_preserve_visual_revision_claim_boundary() -> None:
    assert interface_family(InterfaceName.I0_HARD_TEXT) == "symbolic_downstream_recovery"
    assert interface_family(InterfaceName.I3_SAME_CONVERSATION) == "natural_visual_revision"
    assert interface_family(InterfaceName.I4_EXACT_CACHE) == "natural_visual_revision"


def test_model_introspection_uses_runtime_config_and_named_modules() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=36, vision_config=SimpleNamespace(depth=32)),
        named_modules=lambda: iter(
            [("", object()), ("model.layers.0", object()), ("visual.merger", object())]
        ),
    )
    result = introspect_model(model)
    assert result["language_layers"] == 36
    assert result["vision_config"]["depth"] == 32
    assert "visual.merger" in result["module_names"]


def test_model_introspection_supports_composite_qwen_text_config() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(num_hidden_layers=36),
            vision_config=SimpleNamespace(depth=32),
        ),
        named_modules=lambda: iter(
            (("", object()), ("model.language_model.layers.0", object()))
        ),
    )

    result = introspect_model(model)

    assert result["language_layers"] == 36


def test_server_model_loader_is_pinned_and_fails_closed_locally(tmp_path) -> None:
    assert MODEL_PATH == "/model/ModelScope/Qwen/Qwen2.5-VL-3B-Instruct"
    assert MODEL_SNAPSHOT_SHA256 == (
        "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87"
    )
    with pytest.raises(RuntimeError, match="snapshot"):
        require_server_model(tmp_path / "missing", MODEL_SNAPSHOT_SHA256)
