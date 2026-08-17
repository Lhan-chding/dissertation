"""Offline fake-runtime coverage for the Qwen v4 helper layer."""

from __future__ import annotations

import math
import sys
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from compensability_v4.qwen import cache_continuation as cache
from compensability_v4.qwen import candidate_scoring as scoring
from compensability_v4.qwen import interface_runner as interfaces
from compensability_v4.qwen import introspect_model as introspection
from compensability_v4.qwen import layerwise_assimilation as assimilation
from compensability_v4.qwen import manual_generation as generation
from compensability_v4.qwen import model_loader


class TableTokenizer:
    eos_token_id = 9

    def __init__(self, table: dict[str, object] | None = None) -> None:
        self.table = table or {"A": [1], "B": [2], "AA": [1, 1]}

    def encode(self, label: str, *, add_special_tokens: bool) -> object:
        assert add_special_tokens is False
        return self.table[label]

    def decode(self, token_ids: object, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return "decoded:" + ",".join(str(value) for value in token_ids)  # type: ignore[arg-type]


def make_cached_state(**changes: object) -> cache.CachedGenerationState:
    values = {
        "sample_id": "scene-1",
        "token_ids": (10, 20),
        "attention_mask": (1, 1),
        "position_ids": ((0, 1),),
        "image_token_positions": (1,),
        "image_grid_thw": (1, 10, 10),
        "visual_token_count": 1,
        "generation_config": {"do_sample": False, "nested": {"values": [1, 2]}},
        "rng_seed": 17,
        "past_key_values": object(),
        "chat_messages": ({"role": "user", "content": "look"},),
        "generated_token_ids": (),
    }
    values.update(changes)
    return cache.CachedGenerationState(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"sample_id": " "}, "sample_id"),
        ({"token_ids": ()}, "token ids"),
        ({"token_ids": (-1,), "attention_mask": (1,), "position_ids": ((0,),)}, "token ids"),
        ({"attention_mask": (1,)}, "align"),
        ({"attention_mask": (1, 2)}, "binary"),
        ({"position_ids": ((0,),)}, "position ids"),
        ({"image_token_positions": (2,)}, "outside"),
        ({"visual_token_count": -1}, "visual token count"),
        ({"image_token_positions": (), "visual_token_count": 1}, "requires saved"),
        ({"image_grid_thw": (1, 0, 1)}, "image_grid_thw"),
        ({"rng_seed": True}, "rng_seed"),
        ({"past_key_values": None}, "past_key_values"),
        ({"generated_token_ids": (99,)}, "provenance"),
        ({"generation_config": {1: "bad"}}, "keys"),
    ],
)
def test_cached_generation_state_rejects_corrupt_provenance(
    changes: dict[str, object], error: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        make_cached_state(**changes)


def test_cached_generation_state_deep_freezes_config_and_messages() -> None:
    state = make_cached_state()

    assert state.generation_config["nested"]["values"] == (1, 2)  # type: ignore[index]
    assert isinstance(state.generation_config, MappingProxyType)
    with pytest.raises(TypeError):
        state.chat_messages[0]["role"] = "assistant"  # type: ignore[index]


def test_cache_suffix_parity_and_sample_guards_cover_edge_cases() -> None:
    assert cache.build_suffix_token_ids((1,), (1, 2)) == (2,)
    with pytest.raises(ValueError, match="non-empty suffix"):
        cache.build_suffix_token_ids((1,), (1,))
    with pytest.raises(RuntimeError, match="sample 'scene-1'"):
        cache.require_state_sample(make_cached_state(), "scene-2")
    cache.ParityGate("scene-1", (3, 4), (3, 4))
    with pytest.raises(RuntimeError, match="generated token 1"):
        cache.assert_cache_parity((3, 4), (3, 5), sample_id="scene-1")


@pytest.mark.parametrize("value", [[[1], [2]], object(), "tokens"])
def test_cache_token_normalization_rejects_unsupported_values(value: object) -> None:
    with pytest.raises(RuntimeError):
        cache._token_ids(value)


def test_append_turn_continues_from_exact_suffix_and_builds_new_state(monkeypatch) -> None:
    tokenizer = TableTokenizer()

    class Processor:
        def __init__(self) -> None:
            self.tokenizer = tokenizer

        def apply_chat_template(self, messages: object, **kwargs: object) -> list[list[int]]:
            assert kwargs == {"tokenize": True, "add_generation_prompt": True}
            assert messages[-1] == {"role": "user", "content": "repair"}  # type: ignore[index]
            return [[10, 20, 30]]

    processor = Processor()
    state = make_cached_state(processor=processor)
    fake_result = SimpleNamespace(
        generated_token_ids=(40,),
        all_token_ids=(10, 20, 30, 40),
        attention_mask=(1, 1, 1, 1),
        position_ids=((0, 1, 2, 3),),
        generation_config={"do_sample": False, "max_new_tokens": 2},
        rng_seed=17,
        past_key_values="next-cache",
    )
    calls: dict[str, object] = {}

    def fake_generate(model: object, batch: object, **kwargs: object) -> object:
        calls.update({"model": model, "batch": batch, **kwargs})
        return fake_result

    monkeypatch.setattr(generation, "manual_greedy_generate", fake_generate)
    model = SimpleNamespace(device="cpu")
    result = cache.append_turn_and_continue(
        model,
        state,
        "repair",
        max_new_tokens=2,
        parity_reference_token_ids=(40,),
    )

    assert result["text"] == "decoded:40"
    assert result["suffix_token_ids"] == (30,)
    assert result["parity_verified"] is True
    assert result["state"].past_key_values == "next-cache"  # type: ignore[union-attr]
    assert calls["prior_token_ids"] == (10, 20)
    assert calls["past_key_values"] is state.past_key_values
    assert calls["batch"]["input_ids"].tolist() == [[30]]  # type: ignore[index]


def test_append_turn_reports_non_roundtrip_template_as_diagnostic_only() -> None:
    class NonRoundtripProcessor:
        tokenizer = TableTokenizer()

        @staticmethod
        def apply_chat_template(_messages: object, **_kwargs: object) -> list[list[int]]:
            return [[99, 20, 30]]

    state = make_cached_state(processor=NonRoundtripProcessor())
    with pytest.raises(RuntimeError, match=r"decode/re-encode.*diagnostic"):
        cache.append_turn_and_continue(object(), state, "repair")


@pytest.mark.parametrize(
    ("state", "text", "sample_id", "message"),
    [
        (make_cached_state(), " ", None, "non-empty"),
        (make_cached_state(), "ok", "other", "belongs"),
        (make_cached_state(chat_messages=(), processor=object()), "ok", None, "chat-message"),
        (make_cached_state(processor=None), "ok", None, "processor"),
    ],
)
def test_append_turn_fails_closed_before_generation(
    state: cache.CachedGenerationState, text: str, sample_id: str | None, message: str
) -> None:
    with pytest.raises((ValueError, RuntimeError), match=message):
        cache.append_turn_and_continue(object(), state, text, sample_id=sample_id)


def test_candidate_helpers_cover_fake_and_standard_forward_paths() -> None:
    class Processor:
        tokenizer = TableTokenizer()

        def __call__(self, *args: object, **kwargs: object) -> dict[str, torch.Tensor]:
            assert kwargs["return_tensors"] == "pt"
            return {
                "input_ids": torch.tensor([[7, 8, 0]]),
                "attention_mask": torch.tensor([[1, 1, 0]]),
            }

    class Model:
        device = torch.device("cpu")

        def __call__(self, **kwargs: object) -> object:
            assert kwargs["use_cache"] is False
            logits = torch.zeros((1, 3, 4))
            logits[0, 1, 1] = 2.0
            logits[0, 1, 2] = -1.0
            return SimpleNamespace(logits=logits)

    assert scoring.next_token_logits(Model(), Processor(), "prompt").tolist() == [
        0.0,
        2.0,
        -1.0,
        0.0,
    ]
    scored = scoring.score_candidate_labels(
        Model(), Processor(), "prompt", ("A", "B"), ((1, 2, 3, 4), (4, 3, 2, 1))
    )
    assert scored[(1, 2, 3, 4)] == 2.0
    assert scoring.candidate_margin(scored, (1, 2, 3, 4), (4, 3, 2, 1)) == 3.0
    probabilities = scoring.candidate_log_probabilities(scored)
    assert math.isclose(sum(math.exp(value) for value in probabilities.values()), 1.0)


def test_candidate_prompt_fallback_and_device_mapping() -> None:
    class MoveValue:
        def __init__(self) -> None:
            self.devices: list[str] = []

        def to(self, device: str) -> MoveValue:
            self.devices.append(device)
            return self

    class PositionalProcessor:
        def __call__(self, *args: object, **kwargs: object) -> dict[str, object]:
            if "text" in kwargs:
                raise TypeError("positional only")
            assert args == ("prompt",)
            return {"value": MoveValue()}

    batch = scoring._prepare_prompt(PositionalProcessor(), "prompt")
    moved = scoring._move_to_model_device(batch, SimpleNamespace(device="cpu"))
    assert moved["value"].devices == ["cpu"]
    with pytest.raises(TypeError, match="callable"):
        scoring._prepare_prompt(object(), "prompt")


@pytest.mark.parametrize(
    ("labels", "worlds", "message"),
    [
        ((), (), "aligned"),
        (("AA",), ((1, 2, 3, 4),), "single token"),
        (("A", "A"), ((1, 2, 3, 4), (4, 3, 2, 1)), "unique"),
        (("A",), ((1, 2, 3),), "four values"),
        (("A", "B"), ((1, 2, 3, 4), (1, 2, 3, 4)), "worlds must be unique"),
    ],
)
def test_candidate_scoring_rejects_invalid_candidate_contracts(
    labels: tuple[str, ...], worlds: tuple[tuple[int, ...], ...], message: str
) -> None:
    model = SimpleNamespace(next_token_logits=lambda _prompt: {1: 1.0, 2: 2.0})
    with pytest.raises((ValueError, RuntimeError), match=message):
        scoring.score_candidate_labels(model, TableTokenizer(), "prompt", labels, worlds)


def test_candidate_label_and_score_error_paths() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        scoring.find_single_token_labels(TableTokenizer(), ("A",), True)
    with pytest.raises(RuntimeError, match="insufficient"):
        scoring.find_single_token_labels(TableTokenizer(), ("AA",), 1)
    with pytest.raises(TypeError, match="encode"):
        scoring.find_single_token_labels(object(), ("A",), 1)
    with pytest.raises(RuntimeError, match="invalid ids"):
        scoring.find_single_token_labels(TableTokenizer({"A": "bad"}), ("A",), 1)
    with pytest.raises(RuntimeError, match="negative"):
        scoring.find_single_token_labels(TableTokenizer({"A": [-1]}), ("A",), 1)
    with pytest.raises(RuntimeError, match="finite"):
        scoring._logit_at({1: float("nan")}, 1)
    with pytest.raises(ValueError, match="non-empty"):
        scoring.candidate_log_probabilities({})
    with pytest.raises(ValueError, match="both"):
        scoring.candidate_margin({(1, 2, 3, 4): 1.0}, (1, 2, 3, 4), (4, 3, 2, 1))


def test_interface_runner_enforces_claim_and_cache_parity_boundaries() -> None:
    runner = interfaces.InterfaceRunner(
        {
            interfaces.InterfaceName.I0_HARD_TEXT: lambda **kwargs: f"hard:{kwargs['sample_id']}",
            interfaces.InterfaceName.I1_SOFT_REPORT: lambda **_kwargs: {"text": "soft"},
            interfaces.InterfaceName.I4_EXACT_CACHE: lambda **_kwargs: {
                "text": "cache",
                "parity_verified": True,
                "proof": "exact-token-equality",
            },
        }
    )

    hard = runner.run(interfaces.InterfaceName.I0_HARD_TEXT, sample_id="s")
    assert hard.primary_eligible is True
    assert hard.family == "symbolic_downstream_recovery"
    assert (
        runner.run(interfaces.InterfaceName.I1_SOFT_REPORT, sample_id="s").primary_eligible is False
    )
    exact = runner.run(interfaces.InterfaceName.I4_EXACT_CACHE, sample_id="s", primary_result=True)
    assert exact.parity_verified is True
    assert exact.metadata["proof"] == "exact-token-equality"
    assert interfaces.interface_family(interfaces.InterfaceName.I2_CANDIDATE_WORLD) == (
        "intervention_diagnostic"
    )


def test_interface_runner_rejects_invalid_dispatch_results() -> None:
    with pytest.raises(TypeError, match="InterfaceName"):
        interfaces.InterfaceRunner({}).run("I0", sample_id="s")  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="no runtime handler"):
        interfaces.InterfaceRunner({}).run(interfaces.InterfaceName.I0_HARD_TEXT, sample_id="s")
    with pytest.raises(RuntimeError, match="return text"):
        interfaces.InterfaceRunner(
            {interfaces.InterfaceName.I0_HARD_TEXT: lambda **_kwargs: {"text": 1}}
        ).run(interfaces.InterfaceName.I0_HARD_TEXT, sample_id="s")
    with pytest.raises(RuntimeError, match="parity"):
        interfaces.InterfaceRunner(
            {interfaces.InterfaceName.I4_EXACT_CACHE: lambda **_kwargs: {"text": "x"}}
        ).run(interfaces.InterfaceName.I4_EXACT_CACHE, sample_id="s", primary_result=True)


class DictConfig:
    def __init__(self, value: dict[str, object]) -> None:
        self.value = value

    def to_dict(self) -> dict[str, object]:
        return self.value


def test_introspection_detaches_config_and_writes_both_artifacts(tmp_path) -> None:
    config = DictConfig({"num_hidden_layers": 2, "vision_config": {"depth": 3}})
    config.num_hidden_layers = 2
    config.vision_config = {"depth": 3}
    model = SimpleNamespace(
        config=config,
        named_modules=lambda: iter((("", object()), ("model.layer.0", object()))),
    )

    result = introspection.write_model_introspection(model, tmp_path / "nested")
    config.value["later"] = "mutation"
    assert "later" not in result["model_config"]
    assert (tmp_path / "nested/model_introspection.json").is_file()
    assert (tmp_path / "nested/module_manifest.txt").read_text() == "\nmodel.layer.0\n"


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (object(), "no runtime config"),
        (SimpleNamespace(config=SimpleNamespace(num_hidden_layers=True)), "num_hidden_layers"),
        (SimpleNamespace(config=SimpleNamespace(num_hidden_layers=0)), "must be positive"),
        (SimpleNamespace(config=SimpleNamespace(num_hidden_layers=1)), "vision_config"),
        (
            SimpleNamespace(
                config=SimpleNamespace(num_hidden_layers=1, vision_config={}), named_modules=None
            ),
            "named_modules",
        ),
        (
            SimpleNamespace(
                config=SimpleNamespace(num_hidden_layers=1, vision_config={}),
                named_modules=lambda: iter(()),
            ),
            "empty or non-unique",
        ),
        (
            SimpleNamespace(
                config=SimpleNamespace(num_hidden_layers=1, vision_config={}),
                named_modules=lambda: iter((("x", 1), ("x", 2))),
            ),
            "empty or non-unique",
        ),
    ],
)
def test_introspection_fails_closed_on_invalid_runtime_metadata(
    model: object, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        introspection.introspect_model(model)


def test_config_mapping_supports_mapping_and_rejects_non_mapping_results() -> None:
    assert introspection._config_mapping({"a": 1}) == {"a": 1}
    assert introspection._config_mapping(SimpleNamespace(a=1)) == {"a": 1}
    with pytest.raises(RuntimeError, match="not introspectable"):
        introspection._config_mapping(object())
    with pytest.raises(RuntimeError, match="produce a mapping"):
        introspection._config_mapping(DictConfig([]))  # type: ignore[arg-type]


class ProjectionModel:
    def __init__(self, *, attention: bool = True) -> None:
        self.config = SimpleNamespace(num_hidden_layers=2)
        self.model = SimpleNamespace(norm=lambda value: value + 1)
        self.attention = attention

    def get_output_embeddings(self):
        return lambda hidden: torch.tensor([0.0, float(hidden[0]), float(hidden[1])])

    def __call__(self, **kwargs: object) -> object:
        assert kwargs["output_hidden_states"] is True
        embedding = torch.zeros((1, 3, 2))
        first = torch.tensor([[[0.0, 0.0], [1.0, 3.0], [9.0, 9.0]]])
        final = torch.tensor([[[0.0, 0.0], [5.0, 2.0], [8.0, 7.0]]])
        logits = torch.zeros((1, 3, 3))
        target = 1 if self.attention else 2
        logits[0, target] = self.get_output_embeddings()(final[0, target])
        return SimpleNamespace(hidden_states=(embedding, first, final), logits=logits)


def test_layerwise_projection_uses_runtime_norm_head_and_forward_parity() -> None:
    batch = {"input_ids": torch.tensor([[4, 5, 0]]), "attention_mask": torch.tensor([[1, 1, 0]])}
    values = assimilation.layerwise_candidate_logits(ProjectionModel(), batch, {"A": 1, "B": 2})
    assert values == [{"A": 2.0, "B": 4.0}, {"A": 5.0, "B": 2.0}]
    assert assimilation.layerwise_margins(values, true_label="A", observed_label="B") == (
        -2.0,
        3.0,
    )
    no_attention = assimilation.layerwise_candidate_logits(
        ProjectionModel(attention=False), {"input_ids": torch.tensor([[4, 5, 6]])}, (1, 2)
    )
    assert no_attention[-1] == {"1": 8.0, "2": 7.0}


@pytest.mark.parametrize(
    ("base", "cue", "expected"),
    [
        ((-2.0, -1.0), (-2.0, -1.0), "no_assimilation"),
        ((-2.0, -1.0), (-1.0, 1.0), "successful_revision"),
        ((-2.0, -2.0), (-2.0, -1.0), "persistent_but_insufficient_assimilation"),
        ((-2.0, -1.0), (-1.0, -2.0), "transient_assimilation"),
    ],
)
def test_assimilation_profiles_use_only_sign_and_aligned_layerwise_differences(
    base: tuple[float, ...], cue: tuple[float, ...], expected: str
) -> None:
    assert assimilation.classify_assimilation_profile(base, cue) == expected


def test_layerwise_error_paths_are_objective_runtime_invariants() -> None:
    with pytest.raises(ValueError, match="non-empty and unique"):
        assimilation._candidate_ids((1, 1))
    with pytest.raises(ValueError, match="non-negative"):
        assimilation._candidate_ids((-1,))
    with pytest.raises(RuntimeError, match="finite"):
        assimilation._as_float(float("inf"))
    with pytest.raises(ValueError, match="non-negative"):
        assimilation.validate_final_layer_logits([{"A": 1.0}], {"A": 1.0}, absolute_tolerance=-1)
    with pytest.raises(RuntimeError, match="do not match"):
        assimilation.validate_final_layer_logits([], {"A": 1.0})
    with pytest.raises(ValueError, match="every layer"):
        assimilation.layerwise_margins([{"A": 1.0}], true_label="A", observed_label="B")
    with pytest.raises(ValueError, match="aligned"):
        assimilation.classify_assimilation_profile((1.0,), ())


def test_layerwise_runtime_structure_errors() -> None:
    missing_head = SimpleNamespace(model=SimpleNamespace(norm=lambda value: value))
    with pytest.raises(RuntimeError, match="output embedding"):
        assimilation._runtime_projection_modules(missing_head)
    missing_norm = SimpleNamespace(
        get_output_embeddings=lambda: lambda value: value, model=object()
    )
    with pytest.raises(RuntimeError, match="final language norm"):
        assimilation._runtime_projection_modules(missing_norm)
    model = ProjectionModel()
    model.config.num_hidden_layers = 3
    with pytest.raises(RuntimeError, match="hidden-state count"):
        assimilation.layerwise_candidate_logits(model, {"input_ids": torch.tensor([[1]])}, {"A": 1})
    model.config.num_hidden_layers = True
    with pytest.raises(RuntimeError, match="layer count"):
        assimilation.layerwise_candidate_logits(model, {"input_ids": torch.tensor([[1]])}, {"A": 1})
    with pytest.raises(RuntimeError, match="no attended token"):
        assimilation.layerwise_candidate_logits(
            ProjectionModel(),
            {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[0]])},
            {"A": 1},
        )


def test_manual_generation_result_and_input_contract_errors() -> None:
    valid = {
        "prompt_token_ids": (1,),
        "generated_token_ids": (2,),
        "all_token_ids": (1, 2),
        "attention_mask": (1, 1),
        "position_ids": ((0, 1),),
        "past_key_values": object(),
        "generation_config": {},
        "rng_seed": 0,
    }
    result = generation.ManualGenerationResult(**valid)
    assert isinstance(result.generation_config, MappingProxyType)
    for key, value, message in (
        ("all_token_ids", (1, 3), "provenance"),
        ("attention_mask", (1,), "attention"),
        ("position_ids", ((0,),), "position"),
    ):
        invalid = {**valid, key: value}
        with pytest.raises(ValueError, match=message):
            generation.ManualGenerationResult(**invalid)
    with pytest.raises(TypeError, match="mapping-like"):
        generation._mapping(object())
    with pytest.raises(RuntimeError, match="exactly one"):
        generation._one_dimensional_ids([[1], [2]])
    with pytest.raises(TypeError, match="sequence-like"):
        generation._one_dimensional_ids(object())
    with pytest.raises(RuntimeError, match="position_ids are empty"):
        generation._position_tuple([], 1)


class GreedyModel:
    device = torch.device("cpu")

    def __init__(self, token: int = 4, cache: object = "cache") -> None:
        self.token = token
        self.cache = cache
        self.calls = 0

    def __call__(self, **kwargs: object) -> object:
        self.calls += 1
        length = int(kwargs["input_ids"].shape[1])  # type: ignore[union-attr]
        logits = torch.zeros((1, length, 8))
        logits[0, -1, self.token] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=self.cache)


def test_manual_greedy_generation_runs_on_cpu_without_transformers() -> None:
    model = GreedyModel()
    result = generation.manual_greedy_generate(
        model,
        {"input_ids": torch.tensor([[1, 2]])},
        max_new_tokens=3,
        eos_token_ids=(4, 7),
        rng_seed=23,
    )

    assert result.prompt_token_ids == (1, 2)
    assert result.generated_token_ids == (4,)
    assert result.all_token_ids == (1, 2, 4)
    assert result.attention_mask == (1, 1, 1)
    assert result.position_ids == ((0, 1, 2),)
    assert result.past_key_values == "cache"
    assert model.calls == 2


class PreparedGreedyModel(GreedyModel):
    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, **kwargs: object) -> dict:
        return {**kwargs, "input_ids": input_ids}

    def _get_initial_cache_position(self, _input_ids: torch.Tensor, kwargs: dict) -> dict:
        return {**kwargs, "cache_position": torch.tensor([0])}

    def _update_model_kwargs_for_generation(
        self, output: object, kwargs: dict, *, is_encoder_decoder: bool
    ) -> dict:
        assert is_encoder_decoder is False
        return {
            **kwargs,
            "past_key_values": output.past_key_values,
            "attention_mask": torch.cat((kwargs["attention_mask"], torch.ones((1, 1))), dim=-1),
        }


def test_manual_generation_supports_paired_prior_cache_and_prepare_hooks() -> None:
    result = generation.manual_greedy_generate(
        PreparedGreedyModel(),
        {"input_ids": torch.tensor([[3]]), "attention_mask": torch.tensor([[1, 1, 1]])},
        max_new_tokens=1,
        past_key_values="prior-cache",
        prior_token_ids=(1, 2),
        prior_position_ids=((0, 1),),
    )
    assert result.prompt_token_ids == (1, 2, 3)
    assert result.all_token_ids == (1, 2, 3, 4)
    assert result.position_ids == ((0, 1, 2, 3),)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"max_new_tokens": True}, TypeError, "integer"),
        ({"max_new_tokens": 0}, ValueError, "positive"),
        ({"max_new_tokens": 1, "generation_config": {"do_sample": True}}, RuntimeError, "greedy"),
        (
            {"max_new_tokens": 1, "generation_config": {"temperature": float("nan")}},
            RuntimeError,
            "temperature",
        ),
    ],
)
def test_manual_generation_rejects_nondeterministic_or_invalid_settings(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        generation.manual_greedy_generate(GreedyModel(), {}, **kwargs)  # type: ignore[arg-type]


def test_manual_generation_rejects_bad_batches_and_unpaired_cache() -> None:
    with pytest.raises(RuntimeError, match="no input_ids"):
        generation.manual_greedy_generate(GreedyModel(), {}, max_new_tokens=1)
    with pytest.raises(RuntimeError, match="exactly one"):
        generation.manual_greedy_generate(
            GreedyModel(), {"input_ids": torch.tensor([[1], [2]])}, max_new_tokens=1
        )
    with pytest.raises(ValueError, match="provided together"):
        generation.manual_greedy_generate(
            GreedyModel(),
            {"input_ids": torch.tensor([[1]])},
            max_new_tokens=1,
            prior_token_ids=(1,),
        )
    with pytest.raises(RuntimeError, match="no token/cache"):
        generation.manual_greedy_generate(
            GreedyModel(cache=None), {"input_ids": torch.tensor([[1]])}, max_new_tokens=1
        )


def test_position_helpers_cover_qwen_mrope_and_rank_validation() -> None:
    two = torch.tensor([[0, 1]])
    three = torch.tensor([[[0, 1]], [[0, 1]], [[0, 1]]])
    assert generation._extend_position_ids(two, 1).tolist() == [[0, 1, 2]]
    assert generation._extend_position_ids(three, 1).shape == (3, 1, 3)
    assert generation._extend_position_ids(two, 0) is two
    with pytest.raises(RuntimeError, match="rank"):
        generation._extend_position_ids(torch.tensor([1]), 1)
    assert generation._position_tuple(three, 2) == ((0, 1), (0, 1), (0, 1))
    with pytest.raises(RuntimeError, match="align"):
        generation._position_tuple([[0]], 2)


def test_visual_observation_capture_uses_fake_vision_runtime(monkeypatch) -> None:
    class Batch(dict):
        moved = False

        def to(self, device: object) -> Batch:
            assert str(device) == "cpu"
            self.moved = True
            return self

    tokenizer = TableTokenizer()

    class Processor:
        def __init__(self) -> None:
            self.tokenizer = tokenizer
            self.batch = Batch(
                input_ids=torch.tensor([[50, 5]]), image_grid_thw=torch.tensor([[1, 2, 3]])
            )

        def apply_chat_template(self, messages: object, **kwargs: object) -> str:
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return "template"

        def __call__(self, **kwargs: object) -> Batch:
            assert kwargs["images"] == ["image-input"]
            return self.batch

    processor = Processor()
    fake_vision = SimpleNamespace(
        process_vision_info=lambda _messages: (["image-input"], ["video-input"])
    )
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_vision)
    fake_result = generation.ManualGenerationResult(
        prompt_token_ids=(50, 5),
        generated_token_ids=(9,),
        all_token_ids=(50, 5, 9),
        attention_mask=(1, 1, 1),
        position_ids=((0, 1, 2),),
        past_key_values="cache",
        generation_config={"do_sample": False},
        rng_seed=7,
    )
    monkeypatch.setattr(generation, "manual_greedy_generate", lambda *_args, **_kwargs: fake_result)
    model = SimpleNamespace(device=torch.device("cpu"), config=SimpleNamespace(image_token_id=50))

    result = generation.generate_observation_with_cache(
        model,
        processor,
        "image",
        "prompt",
        sample_id="scene",
        resized_height=280,
        resized_width=280,
        rng_seed=7,
    )

    assert result["text"] == "decoded:9"
    assert result["state"].image_grid_thw == (1, 2, 3)  # type: ignore[union-attr]
    assert result["state"].image_token_positions == (0,)  # type: ignore[union-attr]
    assert processor.batch.moved is True


@pytest.mark.parametrize(("height", "width"), [(0, 280), (281, 280)])
def test_visual_observation_rejects_noncanonical_dimensions(height: int, width: int) -> None:
    with pytest.raises(ValueError, match=r"positive|multiples"):
        generation.generate_observation_with_cache(
            object(),
            object(),
            object(),
            "prompt",
            sample_id="s",
            resized_height=height,
            resized_width=width,
        )


def test_model_loader_verifies_exact_local_snapshot_and_load_options(tmp_path, monkeypatch) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    monkeypatch.setattr(model_loader, "MODEL_PATH", str(snapshot))
    monkeypatch.setattr(
        model_loader, "snapshot_sha256", lambda path: model_loader.MODEL_SNAPSHOT_SHA256
    )
    assert (
        model_loader.require_server_model(snapshot, model_loader.MODEL_SNAPSHOT_SHA256) == snapshot
    )
    with pytest.raises(RuntimeError, match="expectation"):
        model_loader.require_server_model(snapshot, "0" * 64)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(RuntimeError, match="frozen server path"):
        model_loader.require_server_model(other, model_loader.MODEL_SNAPSHOT_SHA256)
    monkeypatch.setattr(model_loader, "snapshot_sha256", lambda path: "0" * 64)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        model_loader.require_server_model(snapshot, model_loader.MODEL_SNAPSHOT_SHA256)


def test_model_loader_loads_offline_without_transformers(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    class ModelClass:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> object:
            calls["model"] = (path, kwargs)
            return SimpleNamespace(eval=lambda: calls.setdefault("eval", True))

    class ProcessorClass:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> object:
            calls["processor"] = (path, kwargs)
            return "processor"

    monkeypatch.setattr(model_loader, "require_server_model", lambda *_args: tmp_path)
    model, processor = model_loader.load_pinned_qwen(
        model_path=tmp_path,
        device_map={"": "cpu"},
        torch_dtype="float",
        model_class=ModelClass,
        processor_class=ProcessorClass,
    )
    assert processor == "processor"
    assert calls["eval"] is True
    assert calls["model"][1]["local_files_only"] is True  # type: ignore[index]
    assert calls["processor"][1]["trust_remote_code"] is False  # type: ignore[index]
    assert model_loader.os.environ["HF_HUB_OFFLINE"] == "1"
    assert model_loader.os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert model is not None


def test_model_loader_rejects_model_without_eval(monkeypatch, tmp_path) -> None:
    loader = SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object())
    monkeypatch.setattr(model_loader, "require_server_model", lambda *_args: tmp_path)
    with pytest.raises(RuntimeError, match="eval"):
        model_loader.load_pinned_qwen(
            model_path=tmp_path,
            torch_dtype="float",
            model_class=loader,
            processor_class=loader,
        )
