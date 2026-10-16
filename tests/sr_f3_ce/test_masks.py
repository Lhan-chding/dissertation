import copy
import json

import pytest

from sr_f3_ce.contract import support_target
from sr_f3_ce.masks import (
    MaskAlignmentError,
    field_masks,
    freeze_serialization,
    serialize_target,
    target_spans,
    tokenize_with_offsets,
)


@pytest.fixture
def teacher():
    world = {
        "categories": ["January", "February"],
        "domain": [10, 90],
        "series": {"Alpha": [62, 83], "Beta": [31, 12]},
    }
    query = {"family": "CROSS", "aggregate": "mean", "value": "Alpha"}
    return support_target(world, query, True)


def character_offsets(text, start=0):
    return [(i + start, i + start + 1) for i in range(len(text))]


def test_fields_have_disjoint_actual_character_masks_and_context_zero(teacher):
    origin = 71
    offsets = [(0, 0), *character_offsets(teacher["text"], origin)]
    audit = field_masks(teacher, offsets, target_start=origin, token_ids=list(range(len(offsets))))
    assert not audit["answer_mask"][0] and not audit["work_mask"][0]
    assert not audit["answer_mask"][1] and not audit["work_mask"][1]  # prefill
    assert (
        sum(audit["answer_mask"]) == teacher["spans"]["answer"][1] - teacher["spans"]["answer"][0]
    )
    assert sum(audit["work_mask"]) == teacher["spans"]["work"][1] - teacher["spans"]["work"][0]
    assert all(not (a and w) for a, w in zip(audit["answer_mask"], audit["work_mask"], strict=True))


def test_structural_comma_can_join_answer_key_without_evidence_leak(teacher):
    answer = teacher["spans"]["answer"][0]
    offsets = [
        *character_offsets(teacher["text"][: answer - 1]),
        (answer - 1, len(teacher["text"])),
    ]
    audit = field_masks(teacher, offsets)
    assert audit["answer_mask"][-1] == 1
    assert audit["work_mask"][-1] == 0
    assert (
        audit["boundary_adjustments"][-1]["reason"] == "only_json_structure_crosses_field_boundary"
    )


def test_work_key_is_not_silently_supervised_with_its_opening_quote(teacher):
    lo, _hi = teacher["spans"]["work"]
    key_start = teacher["text"].index('"work"')
    offsets = [*character_offsets(teacher["text"][:key_start]), (key_start, lo + 1)]
    offsets += character_offsets(teacher["text"][lo + 1 :], lo + 1)
    audit = field_masks(teacher, offsets)
    assert audit["token_owners"][key_start] == "context"
    assert any(audit["work_mask"]) and any(audit["answer_mask"])


def test_cross_field_semantic_token_is_rejected(teacher):
    _lo, hi = teacher["spans"]["work"]
    # The token contains work contents and letters from the supervised answer key.
    offsets = character_offsets(teacher["text"][: hi - 2])
    offsets += [(hi - 2, teacher["spans"]["answer"][0] + 4)]
    offsets += character_offsets(
        teacher["text"][teacher["spans"]["answer"][0] + 4 :], teacher["spans"]["answer"][0] + 4
    )
    with pytest.raises(MaskAlignmentError, match="semantic fields"):
        field_masks(teacher, offsets)


def test_punctuation_inside_string_is_semantic_content():
    text = '{"evidence":[],"work":";","answer":1}'
    _, spans = target_spans(text)
    target = {"text": text, "spans": spans}
    begin = text.index(";")
    end = spans["answer"][0] + 3
    offsets = [*character_offsets(text[:begin]), (begin, end), *character_offsets(text[end:], end)]
    with pytest.raises(MaskAlignmentError, match="semantic fields"):
        field_masks(target, offsets)


def test_native_offset_identity_and_empty_masks_fail_closed(teacher):
    with pytest.raises(ValueError, match="length"):
        field_masks(teacher, [(0, 1)], token_ids=[1, 2])
    with pytest.raises(MaskAlignmentError, match="Empty"):
        field_masks(teacher, [(0, 0)])
    with pytest.raises(MaskAlignmentError, match="monotone"):
        field_masks(teacher, [(10, 11), (9, 10)])


def test_target_spans_ignore_escaped_lookalike_keys_and_reject_tampering(teacher):
    target = copy.deepcopy(teacher)
    target["object"]["work"] = 'look at "answer": and \\ and 😺'
    import json

    target["text"] = json.dumps(target["object"], ensure_ascii=False, separators=(",", ":"))
    _, target["spans"] = target_spans(target["text"])
    assert serialize_target(target)["text"] == target["text"]
    target["spans"]["work"][0] += 1
    with pytest.raises(ValueError, match="spans"):
        serialize_target(target)


def test_uniform_newline_fallback_is_one_corpus_freeze(teacher):
    calls = []

    def encode(target, index):
        calls.append((target["serialization"], index))
        if target["serialization"] == "canonical" and index == 1:
            raise MaskAlignmentError("registered tokenizer boundary case")
        return field_masks(target, character_offsets(target["text"]))

    receipt = freeze_serialization([teacher, teacher, teacher], encode)
    assert receipt["serialization"] == "field_newlines_v1"
    assert receipt["target_count"] == 3
    assert receipt["canonical_failures"][0]["index"] == 1
    assert calls == [
        ("canonical", 0),
        ("canonical", 1),
        ("field_newlines_v1", 0),
        ("field_newlines_v1", 1),
        ("field_newlines_v1", 2),
    ]


def test_real_fast_tokenizer_offsets_include_full_prefix_and_gold_answer(teacher):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        [teacher["text"], "native assistant context\n"],
        trainers.BpeTrainer(vocab_size=256, min_frequency=100, special_tokens=["[UNK]"]),
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    target = serialize_target(teacher)
    audit = tokenize_with_offsets(tokenizer, target, prefix="native assistant context\n")
    encoded = tokenizer(
        "native assistant context\n" + target["text"],
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    assert audit["token_ids"] == encoded["input_ids"]
    assert audit["offsets"] == [list(pair) for pair in encoded["offset_mapping"]]
    assert any(audit["answer_mask"]) and any(audit["work_mask"])
    gold_text = '{"evidence":[],"answer":1}'
    _, gold_spans = target_spans(gold_text)
    gold = tokenize_with_offsets(tokenizer, {"text": gold_text, "spans": gold_spans})
    assert not any(gold["work_mask"])


@pytest.mark.parametrize("answer", ["98/3", None, 42])
def test_prescribed_indent_two_recomputes_exact_fraction_null_and_newline_spans(teacher, answer):
    obj = copy.deepcopy(teacher["object"])
    obj["work"] = 'row one\nrow two; quote="answer"; 中文'
    obj["answer"] = answer
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    _, spans = target_spans(text)
    target = {"text": text, "spans": spans, "object": obj}
    selected = serialize_target(target, "indent_2")
    assert selected["text"] == json.dumps(obj, indent=2, ensure_ascii=False)
    assert list(selected["object"]) == ["evidence", "work", "answer"]
    work = selected["spans"]["work"]
    answer_span = selected["spans"]["answer"]
    assert selected["text"][slice(*work)] == json.dumps(obj["work"], ensure_ascii=False)
    assert selected["text"][slice(*answer_span)] == (
        '"answer": ' + json.dumps(answer, ensure_ascii=False) + "\n}"
    )
    assert selected["spans"] != target["spans"]
    assert serialize_target(selected, "indent_2")["text"] == selected["text"]
    assert target["text"] == text and target["spans"] == spans


def test_prescribed_indent_two_alignment_failure_never_tries_another_serialization(teacher):
    calls = []

    def encode(target, index):
        calls.append((target["serialization"], index))
        raise MaskAlignmentError("actual tokenizer semantic boundary conflict")

    with pytest.raises(MaskAlignmentError, match="Prescribed support serialization failed"):
        freeze_serialization([teacher, teacher], encode, serializations=("indent_2",))
    assert calls == [("indent_2", 0)]
    for policy in ((), ("indent_2", "indent_2"), ("unknown",)):
        with pytest.raises(ValueError, match="policy"):
            freeze_serialization([teacher], encode, serializations=policy)
