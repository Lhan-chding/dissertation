"""Pure CPU measurement, replay, and actual tensor-gradient acceptance checks."""

import copy
import json

import pytest
import torch

from sr_f3_ce.prompts import CONTRACT_V2, question_from_prompt
from sr_f4 import c_local_credit as credit
from sr_f4 import task_contract as contract
from sr_f4.training import native_advantages


def task():
    return dict(qid="q", world_id="w", root_id="r", world=dict(
        categories=["January", "February", "March", "April"], domain=[10, 90],
        series=dict(Alpha=[80, 60, 40, 20], Beta=[30, 50, 70, 90])),
        query=dict(family="TOPK", selection_series="Alpha", direction="largest", k=2,
                   value="Beta", aggregate="sum"))


def record(t=None, *, selection=None, computation=None, answer=80, evidence=None,
           work=None, ensure_ascii=False):
    t = t or task()
    selection = selection or dict(categories=["January", "February"], rule=credit.expected_rule(t))
    computation = computation or dict(categories=["January", "February"], value="Beta",
                                       aggregate="sum", operands=[30, 50])
    if evidence is None:
        evidence = contract.canonical_certificate(t["world"], t["query"])
    work = (work if work is not None else "SELECT " + json.dumps(selection)
            + " COMPUTE " + json.dumps(computation))
    text = json.dumps(dict(evidence=evidence, work=work, answer=answer), ensure_ascii=ensure_ascii)
    return dict(row_id="row-0", group_id="group", qid=t["qid"], full_text=text,
                assistant_prefill="{", tokens=list(range(len(text) - 1)),
                score=contract.score(text, t["world"], t["query"], finish_reason="balanced"))


class CharacterTokenizer:
    def __init__(self, text, *, pieces=None):
        self.pieces = list(text[1:]) if pieces is None else pieces

    def decode(self, ids, **kwargs):
        assert kwargs == dict(skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return "".join(self.pieces[i] for i in ids)


def test_prompt_preserves_native_question_parser_and_exact_rule_fields():
    assert "\n" not in credit.PROTOCOL_INSTRUCTION
    prompt = "Question? " + credit.PROTOCOL_INSTRUCTION + "\n" + CONTRACT_V2
    assert question_from_prompt(prompt).startswith("Question?")
    t = task()
    t["query"] = dict(family="INTERVAL", start=0, end=2, include_start=False,
                      include_end=True, positive_only=True, value="diff", aggregate="sum")
    assert credit.expected_rule(t) == dict(family="INTERVAL", start="January", end="March",
                                         include_start=False, include_end=True, positive_only=True)


def test_explicit_measurement_and_span_exclusion():
    r = record()
    original = copy.deepcopy(r)
    label = credit.measure(task(), r)
    assert (label["P"], label["S"], label["O"], label["E"]) == ("PASS",) * 4
    assert label["normalization"]["j"] == 1
    assert r == original
    lo, hi = label["selection_span"]
    assert r["full_text"][lo:hi].startswith(r'{\"categories\"')
    assert "COMPUTE" not in r["full_text"][lo:hi]
    masks = credit.token_masks(r, label, CharacterTokenizer(r["full_text"]))
    assert sum(masks["selection_mask"]) == hi - lo
    assert not any(a and b for a, b in zip(
        masks["evidence_mask"], masks["selection_mask"], strict=True))
    assert not label["human_audited"]


@pytest.mark.parametrize("change", ["direction", "series", "k", "set"])
def test_wrong_rule_or_adopted_set_fails_even_if_answer_correct(change):
    s = dict(categories=["January", "February"], rule=credit.expected_rule(task()))
    c = dict(categories=s["categories"], value="Beta", aggregate="sum", operands=[30, 50])
    if change == "direction":
        s["rule"]["direction"] = "smallest"
    elif change == "series":
        s["rule"]["selection_series"] = "Beta"
    elif change == "k":
        s["rule"]["k"] = 3
    else:
        s["categories"] = c["categories"] = ["March", "April"]
    r = record(selection=s, computation=c)
    assert r["score"]["value_correct"] == 1
    assert credit.measure(task(), r)["S"] == "FAIL"


def test_compute_category_conflict_abstains_but_calculation_errors_do_not_gate_selection():
    c = dict(categories=["March", "April"], value="Beta", aggregate="sum", operands=[70, 90])
    label = credit.measure(task(), record(computation=c))
    assert label["S"] == "UNKNOWN" and label["selection_span"] is None
    assert any("CONFLICTING" in reason for reason in label["reasons"])
    c["categories"] = ["January", "February"]
    label = credit.measure(task(), record(computation=c, answer=161))
    assert label["S"] == "PASS" and label["O"] == "FAIL" and label["E"] == "FAIL"
    assert label["normalization"]["j"] == 0
    # Compute order may differ, provided the operands follow that order.
    c.update(categories=["February", "January"], operands=[50, 30])
    assert credit.measure(task(), record(computation=c))["O"] == "PASS"


@pytest.mark.parametrize("work", ["", "Selected: January, February", "COMPUTE {} SELECT {}",
                                  'SELECT {"categories":[],"rule":{}} COMPUTE {}',
                                  'SELECT {"categories":[],"categories":[]} COMPUTE {}'])
def test_unsupported_work_unknown_without_inferring_from_answer(work):
    label = credit.measure(task(), record(work=work))
    assert label["S"] == "UNKNOWN" and label["selection_span"] is None


def test_truthful_nonempty_evidence_is_not_sufficient_evidence_and_false_is_fail():
    r = record(evidence=[dict(series="Alpha", category="January", value=80)])
    label = credit.measure(task(), r)
    assert label["P"] == "PASS" and not label["normalization"]["sufficient"]
    assert label["normalization"]["j"] == 0
    r = record(evidence=[dict(series="Alpha", category="January", value=81)])
    assert credit.measure(task(), r)["P"] == "FAIL"
    assert credit.measure(task(), record(evidence=[]))["P"] == "UNKNOWN"


def test_json_unicode_escapes_map_to_original_text_including_surrogate_pairs():
    t = task()
    t["world"]["categories"][0] = '一月😀"'
    s = dict(categories=[t["world"]["categories"][0], "February"], rule=credit.expected_rule(t))
    c = dict(categories=s["categories"], value="Beta", aggregate="sum", operands=[30, 50])
    r = record(t, selection=s, computation=c, ensure_ascii=True)
    label = credit.measure(t, r)
    assert label["S"] == "PASS"
    lo, hi = label["selection_span"]
    assert json.loads('"' + r["full_text"][lo:hi] + '"') == json.dumps(s)
    # Exercise raw JSON-string unicode escapes rather than escaped internal slashes.
    raw = json.dumps('A一😀"B', ensure_ascii=True)
    spans = credit._decoded_string_spans(raw, 0, len(raw))
    assert len(spans) == 5
    assert raw[slice(*spans[2])] == r"\ud83d\ude00"


def inputs(js=None, ps=None, ss=None):
    js = js or [0, 0, 1, 1, 0, 0, 1, 1]
    ps = ps or ["PASS"] * 8
    ss = ss or ["PASS", "FAIL"] * 4
    records, labels = [], []
    for i in range(8):
        r = record()
        r["row_id"] = f"row-{i}"
        r["score"]["J"] = js[i]
        label = credit.measure(task(), r)
        label.update(P=ps[i], S=ss[i])
        records.append(r)
        labels.append(label)
    return records, labels


def test_shared_cohort_expands_to_successes_and_exact_bounded_channels():
    records, labels = inputs()
    original = copy.deepcopy((records, labels))
    outputs = {a: credit.group_credit(task(), records, labels, arm=a, seed=71) for a in credit.ARMS}
    assert outputs["SELECT"]["known_indices"] == list(range(8))
    assert outputs["SELECT"]["selection_advantages"] == [.125, -.125] * 4
    assert outputs["SELECT"]["evidence_advantages"] == [0] * 8
    assert outputs["SELECT"]["native_advantages"] == native_advantages([0, 0, 1, 1, 0, 0, 1, 1])
    assert outputs["J"]["selection_advantages"] == [0] * 8
    assert outputs["EVID"]["selection_advantages"] == [0] * 8
    assert (records, labels) == original
    assert all(o["candidate_indices"] == list(range(8)) for o in outputs.values())


def test_unknown_zeroes_only_auxiliary_and_below_coverage_abstains_whole_group():
    records, labels = inputs(ss=["PASS", "FAIL"] * 3 + ["UNKNOWN", "UNKNOWN"])
    result = credit.group_credit(task(), records, labels, arm="SELECT", seed=3)
    assert result["active"] and result["known_indices"] == list(range(6))
    assert result["selection_advantages"][6:] == [0, 0]
    assert result["native_advantages"][6:] != [0, 0]
    labels[5]["S"] = "UNKNOWN"
    result = credit.group_credit(task(), records, labels, arm="SELECT", seed=3)
    assert result["eligibility_reason"] == "KNOWN_COVERAGE_BELOW_75_PERCENT"
    assert result["selection_advantages"] == result["evidence_advantages"] == [0] * 8


def test_exact_shuffle_preserves_j_p_strata_and_allows_identity():
    records, labels = inputs(ps=["PASS", "PASS", "PASS", "PASS", "FAIL", "FAIL", "FAIL", "FAIL"])
    result = credit.group_credit(task(), records, labels, arm="SHUF", seed=21)
    perm = result["permutation"]
    assert sorted(perm) == list(range(8))
    for dst, src in enumerate(perm):
        assert records[dst]["score"]["J"] == records[src]["score"]["J"]
        assert labels[dst]["P"] == labels[src]["P"]
    ps = [int(label["P"] == "PASS" and labels[perm[i]]["S"] == "PASS")
          for i, label in enumerate(labels)]
    assert result["selection_advantages"] == [.25 * (v - sum(ps) / 8) for v in ps]
    records, labels = inputs(ss=["PASS"] * 8)
    result = credit.group_credit(task(), records, labels, arm="SHUF", seed=21)
    assert result["shuffle"]["changed_count"] == 0
    assert not result["selection_signal"]


def test_native_identity_and_mask_identity_are_fail_closed():
    records, labels = inputs()
    labels[0]["normalization"]["j"] = 1
    with pytest.raises(ValueError, match="identity"):
        credit.group_credit(task(), records, labels, arm="SELECT", seed=1)
    r = record()
    label = credit.measure(task(), r)
    label["row_id"] = "other"
    with pytest.raises(ValueError, match="identity"):
        credit.token_masks(r, label, CharacterTokenizer(r["full_text"]))


def test_boundary_crossing_tokens_are_excluded_and_zero_width_tokens_uncredited():
    r = record()
    label = credit.measure(task(), r)
    lo, hi = label["selection_span"]
    text = r["full_text"]
    pieces = [text[1:lo - 1], text[lo - 1:lo + 1], text[lo + 1:hi - 1],
              text[hi - 1:hi + 1], text[hi + 1:], ""]
    r["tokens"] = list(range(6))
    result = credit.token_masks(r, label, CharacterTokenizer(text, pieces=pieces))
    assert result["selection_mask"] == [False, False, True, False, False, False]
    assert result["diagnostics"]["excluded_boundary_tokens"]["selection"] == [1, 3]


def test_unstable_prefix_full_decode_mismatch_and_bad_spans_abstain():
    r = record()
    label = credit.measure(task(), r)

    class Unstable(CharacterTokenizer):
        def decode(self, ids, **kwargs):
            return "�" if len(ids) == 1 else super().decode(ids, **kwargs)

    result = credit.token_masks(r, label, Unstable(r["full_text"]))
    assert result["diagnostics"]["reason"] == "UNSTABLE_TOKEN_PREFIX_DECODE"
    assert not any(result["selection_mask"] + result["evidence_mask"])
    altered = copy.deepcopy(r)
    altered["full_text"] += " "
    assert credit.token_masks(altered, label, CharacterTokenizer(r["full_text"]))[
        "diagnostics"]["reason"] == "FULL_DECODE_MISMATCH"
    label["selection_span"] = [-1, 5]
    assert credit.token_masks(r, label, CharacterTokenizer(r["full_text"]))[
        "diagnostics"]["reason"] == "INVALID_RAW_CHARACTER_SPAN"


def objective(*, ev=0, sel=0, em=None, sm=None, kl=.02):
    current = torch.tensor([-.2, -.3, -.4, -.5, -.6], requires_grad=True)
    sampler = current.detach() - torch.tensor([0, .1, .2, .8, 1.])
    reference = current.detach() + .2
    result = credit.sequence_objective(current, sampler, reference, -.5,
                                       evidence_advantage=ev, selection_advantage=sel,
                                       evidence_mask=em, selection_mask=sm, kl_coefficient=kl)
    gradient, = torch.autograd.grad(result["loss"], current)
    return result, gradient


def test_j_native_loss_gradient_equivalence_and_kl_once():
    from sr_f12.training import sequence_objective as native_objective

    current = torch.tensor([-.2, -.3, -.4], requires_grad=True)
    sampler, reference = current.detach() - .2, current.detach() + .3
    expected = native_objective(current, sampler, reference, .4)
    actual = credit.sequence_objective(current, sampler, reference, .4)
    assert torch.equal(actual["loss"], expected["loss"])
    assert torch.equal(torch.autograd.grad(actual["loss"], current, retain_graph=True)[0],
                       torch.autograd.grad(expected["loss"], current)[0])
    with_aux, _ = objective(ev=.2, sel=-.2, em=[True, False, False, False, False],
                            sm=[False, True, True, False, False])
    base, _ = objective()
    assert torch.equal(with_aux["kl"], base["kl"])
    assert torch.equal(with_aux["loss"] - with_aux["policy"], base["loss"] - base["policy"])


def test_selection_gradient_only_on_span_and_tis_cap_unchanged():
    base, before = objective()
    local, after = objective(sel=.2, sm=[False, False, True, True, False])
    diff = after - before
    assert torch.equal(diff[[0, 1, 4]], torch.zeros(3))
    assert torch.allclose(diff[2:4], -.2 * local["weights"][2:4] / 2)
    assert local["weights"].max() == 2
    assert torch.equal(base["weights"], local["weights"])
    assert torch.equal(base["tis_truncated"], local["tis_truncated"])


def test_evidence_gradient_only_span_empty_masks_abstain_and_span_mean_not_length_diluted():
    _, before = objective()
    _, after = objective(ev=.125, em=[True, False, False, False, False])
    assert torch.allclose(after - before, torch.tensor([-.125, 0, 0, 0, 0]))
    empty, gradient = objective(ev=.2, sel=-.1, em=[False] * 5, sm=[False] * 5)
    assert torch.equal(gradient, before)
    assert empty["evidence_policy"] == empty["selection_policy"] == 0
    for size in (5, 50):
        cur = torch.full((size,), -.2, requires_grad=True)
        result = credit.sequence_objective(cur, cur.detach(), cur.detach(), 0,
                                           selection_advantage=.25,
                                           selection_mask=[True] * 2 + [False] * (size - 2))
        grad, = torch.autograd.grad(result["loss"], cur)
        assert torch.equal(grad[:2], torch.tensor([-.125, -.125]))
        assert torch.count_nonzero(grad[2:]) == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), .251,
                                   torch.tensor(.1, requires_grad=True), [.1, .2]])
def test_invalid_or_trainable_auxiliary_coefficient_rejected(value):
    with pytest.raises(ValueError, match="coefficient"):
        objective(sel=value, sm=[False, True, False, False, False])


@pytest.mark.parametrize("mask", [[1, 0, 0, 0, 0], [True],
                                  torch.ones(5, requires_grad=True), [float("nan")] * 5])
def test_invalid_mask_rejected(mask):
    with pytest.raises(ValueError, match="mask"):
        objective(sel=.1, sm=mask)


def test_overlap_rejected():
    with pytest.raises(ValueError, match="overlap"):
        objective(ev=.1, sel=.1, em=[True] * 5, sm=[True] * 5)
