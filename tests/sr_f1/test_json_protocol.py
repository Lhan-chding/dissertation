from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from sr_f1.contract import _reference, gold_output, score
from sr_f1.json_protocol import (
    AMENDMENT_ID,
    BalancedJSONStop,
    balanced_cut,
    decoded_protocol,
    first_balanced_token,
    score_record,
    validate_record,
)
from sr_f1.runtime import SRRuntime, generation_recipe


@pytest.mark.parametrize(
    "text,expected",
    [
        ("{}", 2),
        ('{"x": {"y": [1, {"z": 2}]}} tail', 27),
        ('{"x": "} { \\"", "y": 1}', 23),
        ('{"x": "\\\\"}', 11),
        ('{"x": "unfinished }', None),
        ('{"x": 1', None),
        ("prose {}", None),
        (" {}", None),
    ],
)
def test_balanced_cut_strings_escapes_nesting_and_no_prose_extraction(text, expected):
    assert balanced_cut(text) == expected


class Tokenizer:
    def __init__(self, pieces=()):
        self.pieces = {i + 100: piece for i, piece in enumerate(pieces)}
        self.pieces[90] = "{"
        self.pieces[999] = ""

    def encode(self, text, **kwargs):
        if text == "{":
            return [90]
        if text.startswith('{"'):
            return [91, 101]
        if text.startswith("{\n"):
            return [90, 100, 101]
        raise ValueError("Unexpected fixture encoding")

    def decode(self, tokens, **kwargs):
        return "".join(self.pieces[token] for token in tokens)


class Inputs(dict):
    def to(self, device):
        return self


class Processor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bad_boundary = False

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return messages[0]["content"][-1]["text"] + (
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    def __call__(self, *, text, **kwargs):
        tokens = [2, 3]
        if text[0].endswith("{"):
            tokens = [2, 4, 90] if self.bad_boundary else [*tokens, 90]
        ids = torch.tensor([tokens])
        return Inputs(input_ids=ids, attention_mask=torch.ones_like(ids))


class Config(SimpleNamespace):
    def to_dict(self):
        return vars(self)


class Model:
    config = SimpleNamespace(image_token_id=99)

    def __init__(self, tokens):
        self.tokens, self.generated_tokens = tokens, []

    def named_parameters(self):
        return []

    def eval(self):
        return self

    def generate(self, *, input_ids, generation_config, stopping_criteria=None, **kwargs):
        self.initial_ids = input_ids.tolist()[0]
        self.criteria_seen = stopping_criteria
        sequence, logits = input_ids.clone(), []
        for token in self.tokens[: generation_config.max_new_tokens]:
            self.generated_tokens.append(token)
            sequence = torch.cat([sequence, sequence.new_tensor([[token]])], dim=1)
            logits.append(torch.zeros((1, 1000)))
            # HF checks criteria after sampling, including the completing token.
            stopped = stopping_criteria(sequence, logits[-1]) if stopping_criteria else False
            if bool(stopped) or token == 999:
                break
        return SimpleNamespace(sequences=sequence, logits=logits, scores=logits)


def runtime(pieces, *, amended=True, tokens=None):
    obj = SRRuntime.__new__(SRRuntime)
    obj.protocol_amendment = {"id": AMENDMENT_ID} if amended else None
    obj.processor = Processor(Tokenizer(pieces))
    obj.model = Model(tokens if tokens is not None else list(range(100, 100 + len(pieces))))
    obj.device, obj.torch, obj.image_calls = "cpu", torch, 0
    obj.adapter_path, obj.eos_ids = None, [999]
    obj.identity = dict(processor_hash="p", chat_template_hash="c")
    obj.generation_config = Config(**generation_recipe())
    obj.events = []
    obj.reserve = lambda kind, count, **metadata: obj.events.append((kind, count, metadata))
    return obj


def test_prefill_appended_to_native_prompt_not_generated_logprobs_or_accounting(tmp_path):
    obj = runtime(['"answer": 1', "}\n\n", "DO NOT GENERATE"])
    record = obj.generate(text="query", run_root=tmp_path, seed=7)
    assert obj.model.initial_ids == [2, 3, 90]
    assert obj.model.generated_tokens == [100, 101]
    assert record["tokens"] == record["raw_tokens"] == [100, 101]
    assert len(record["old_logprobs"]) == len(record["sampler_logprobs"]) == 2
    assert record["prompt_token_count"] == 3
    assert record["completion_token_count"] == 2
    assert record["generated_text"] == '"answer": 1}\n\n'
    assert record["raw_text"] == '{"answer": 1}'
    assert record["finish_reason"] == "balanced"
    assert record["balanced_token_count"] == 2
    assert record["truncated"] is False
    assert record["format_protocol_error"] is None
    assert validate_record(record, AMENDMENT_ID)
    assert [(kind, count) for kind, count, _ in obj.events] == [
        ("completion_attempts", 1),
        ("generated_tokens", 2),
    ]


@pytest.mark.parametrize("protocol", ["answer_only", "plain_answer"])
def test_non_evidence_channels_and_legacy_remain_unmodified(tmp_path, protocol):
    for amended in (True, False):
        obj = runtime(["42", ""], amended=amended, tokens=[100, 999])
        record = obj.generate(text="query", run_root=tmp_path, seed=8, protocol=protocol)
        assert obj.model.initial_ids == [2, 3]
        assert obj.model.criteria_seen is None
        assert record["raw_text"] == "42"
        assert record["tokens"] == [100, 999]
        assert "protocol_amendment_id" not in record
        assert "assistant_prefill" not in record["input_routing"]
        assert record["finish_reason"] == "eos"
    legacy = runtime(['{"answer": 1}', ""], amended=False, tokens=[100, 999])
    assert legacy.generate(text="query", run_root=tmp_path, seed=9)["raw_text"] == '{"answer": 1}'


def test_prefill_rejects_boundary_merge_and_wrong_registered_token(tmp_path):
    obj = runtime([])
    obj.processor.bad_boundary = True
    with pytest.raises(PermissionError, match="boundary"):
        obj.prepare_text("query", None, tmp_path)
    obj.processor.bad_boundary = False
    obj.processor.tokenizer.encode = lambda text, **kwargs: [91]
    with pytest.raises(PermissionError, match="token 90"):
        obj.prepare_text("query", None, tmp_path)


def task():
    return dict(
        world={
            "categories": ["January", "February", "March", "April"],
            "series": {"Alpha": [30, 42, 26, 17], "Beta": [20, 50, 14, 21]},
            "unit": "count",
        },
        query=_reference.base_query("CROSS"),
    )


def test_closing_token_nonwhitespace_suffix_is_retained_and_zero_scored(tmp_path):
    question = task()
    gold = json.dumps(gold_output(question["world"], question["query"]))
    obj = runtime([gold[1:-1], "}\n```", "DO NOT GENERATE"])
    record = obj.generate(text="query", run_root=tmp_path, seed=5)
    assert obj.model.generated_tokens == [100, 101]
    assert record["raw_text"] == gold + "\n```"
    assert record["generated_text"] == gold[1:] + "\n```"
    assert record["format_protocol_error"] == "trailing_after_close"
    assert record["generation_status"] == "COMPLETE"
    scored = score_record(record, question)
    assert scored["reason"] == "trailing_after_close"
    assert all(scored[field] == 0 for field in ("L_json", "L_answer", "L_evidence", "A", "J"))


def test_valid_record_uses_exact_unchanged_official_answer_type_rule(tmp_path):
    question = task()
    gold = gold_output(question["world"], question["query"])
    gold["answer"] = "56.0"  # Strict registered rational-string syntax rejects this.
    text = json.dumps(gold)
    record = runtime([text[1:]]).generate(text="query", run_root=tmp_path, seed=10)
    result = score_record(record, question)
    assert result == score(text, question["world"], question["query"])
    assert result["L_json"] == 1 and result["L_answer"] == 0


@pytest.mark.parametrize("ending", ["eos", "length"])
def test_unbalanced_eos_or_768_is_format_failure_without_synthetic_eos(tmp_path, ending):
    tokens = [100, 999] if ending == "eos" else [100] * 768
    obj = runtime(['"'], tokens=tokens)
    record = obj.generate(text="query", run_root=tmp_path, seed=11)
    assert record["tokens"] == tokens
    assert record["finish_reason"] == ending
    assert record["balanced_token_count"] is None
    assert record["balanced_cut_char"] is None
    assert record["format_protocol_error"] == "unbalanced"
    assert record["truncated"] is (ending == "length")
    assert score_record(record, task())["reason"] == "unbalanced"


def test_balanced_at_token_limit_is_not_length_truncation(tmp_path):
    obj = runtime([" ", "}"], tokens=[100] * 767 + [101])
    record = obj.generate(text="query", run_root=tmp_path, seed=12)
    assert len(record["tokens"]) == 768
    assert record["finish_reason"] == "balanced" and not record["truncated"]
    assert record["balanced_token_count"] == 768


def test_independent_first_token_replay_and_stop_observation_mismatch_are_technical(tmp_path):
    tokenizer = Tokenizer(['"x":', "{}", "}", "post"])
    assert first_balanced_token(tokenizer, [100, 101, 102, 103]) == (3, 8)
    stop = BalancedJSONStop(tokenizer, 2)
    assert not bool(stop(torch.tensor([[1, 90, 100, 101]]), None))
    assert bool(stop(torch.tensor([[1, 90, 100, 101, 102]]), None))
    with pytest.raises(RuntimeError, match="continued"):
        stop(torch.tensor([[1, 90, 100, 101, 102, 103]]), None)
    obj = runtime(["}", "post", ""], tokens=[100, 101, 999])
    original_generate = obj.model.generate

    def faulty_generate(**kwargs):
        kwargs.pop("stopping_criteria")
        return original_generate(**kwargs)

    obj.model.generate = faulty_generate
    saved = []
    with pytest.raises(RuntimeError, match="BALANCED_STOP_REPLAY_MISMATCH"):
        obj.generate(text="query", run_root=tmp_path, seed=13, on_completion=saved.append)
    assert saved[0]["tokens"] == [100, 101, 999]
    assert saved[0]["generation_status"] == "TECHNICAL_INVALID"


def test_durable_metadata_cannot_be_stripped_or_rewritten_before_score(tmp_path):
    record = runtime(["}"]).generate(text="query", run_root=tmp_path, seed=14)
    assert validate_record(record, AMENDMENT_ID)
    for field, replacement in (
        ("raw_text", "changed"),
        ("balanced_token_count", 2),
        ("generated_text", "}extra"),
        ("format_protocol_error", "unbalanced"),
        ("old_logprobs", []),
    ):
        modified = {**record, field: replacement}
        with pytest.raises(PermissionError):
            validate_record(modified, AMENDMENT_ID)
    modified = copy.deepcopy(record)
    del modified["protocol_amendment_id"]
    with pytest.raises(PermissionError, match="incomplete"):
        score_record(modified, task())
    with pytest.raises(PermissionError, match="incomplete"):
        validate_record({"raw_text": "{}"}, AMENDMENT_ID)


def test_amended_gold_slices_prefill_and_does_not_append_eos():
    obj = runtime(["\n", "}"])
    assert obj.encode_completion("{\n}") == [100, 101]
    assert obj.encode_completion("{\n}", eos=False) == [100, 101]
    with pytest.raises(PermissionError, match="independent token 90"):
        obj.encode_completion('{"answer": 1}')
    legacy = runtime(["\n", "}"], amended=False)
    assert legacy.encode_completion("{\n}") == [90, 100, 101, 999]


def test_protocol_error_does_not_hide_whitespace_or_internal_syntax():
    assert decoded_protocol('"answer":1}\n\t')["raw_text"] == '{"answer":1}'
    assert decoded_protocol('"answer":1} x')["format_protocol_error"] == "trailing_after_close"
    # Balancing is termination, not validation: the official scorer rejects this later.
    assert decoded_protocol('"answer":}')["format_protocol_error"] is None


def test_native_qwen_generate_stops_at_one_real_draw_and_teacher_forces_same_token(tmp_path):
    """Tiny CPU architecture check; the actual 9B CUDA ENGINE gate remains required."""
    import transformers

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(71010)
        config = transformers.Qwen3_5Config(
            text_config=dict(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=48,
                num_hidden_layers=2,
                layer_types=["linear_attention", "full_attention"],
                head_dim=8,
                linear_num_key_heads=2,
                linear_num_value_heads=4,
                linear_key_head_dim=8,
                linear_value_head_dim=8,
                linear_conv_kernel_dim=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                bos_token_id=1,
                eos_token_id=None,
                pad_token_id=0,
                rope_parameters=dict(
                    rope_type="default",
                    mrope_section=[1, 1, 2],
                    mrope_interleaved=True,
                    partial_rotary_factor=1.0,
                ),
            ),
            vision_config=dict(
                depth=1,
                hidden_size=32,
                intermediate_size=48,
                num_heads=4,
                patch_size=2,
                spatial_merge_size=2,
                temporal_patch_size=2,
                out_hidden_size=32,
                num_position_embeddings=16,
            ),
            image_token_id=120,
            video_token_id=121,
            vision_start_token_id=118,
            vision_end_token_id=119,
        )
        config._attn_implementation = "eager"
        obj = runtime([])
        obj.model = transformers.Qwen3_5ForConditionalGeneration(config).eval()
        # Every native sampled ID decodes to a close in this tokenizer fixture;
        # no logits processor forces, filters or otherwise modifies the draw.
        obj.processor.tokenizer.decode = lambda tokens, **kwargs: "}" * len(tokens)
        obj.eos_ids = []
        obj.generation_config = transformers.GenerationConfig(
            **generation_recipe(), eos_token_id=None, pad_token_id=0, bos_token_id=1
        )
        record = obj.generate(text="query", run_root=tmp_path, seed=71010)
        assert record["finish_reason"] == "balanced"
        assert len(record["tokens"]) == record["balanced_token_count"] == 1
        assert record["prompt_token_count"] == 3
        assert record["old_logprobs"] == record["sampler_logprobs"]
        prepared = obj.prepare_text("query", None, tmp_path)
        result = obj.cached_training_forward(
            prepared, record["tokens"], purpose="amendment_cpu_fixture", grad=True
        )
        torch.testing.assert_close(
            result["logprobs"].detach(), torch.tensor(record["old_logprobs"]), atol=0, rtol=0
        )
        assert result["logprobs"].requires_grad
        assert result["cached_model_forward_calls"] == 1
        result["logprobs"].sum().backward()
        assert torch.isfinite(obj.model.lm_head.weight.grad).all()
    finally:
        torch.set_num_threads(previous_threads)
