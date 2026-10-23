import copy
import itertools
import json
from dataclasses import replace

import pytest

from sr_f4.credit_design_replay import (
    POLICIES,
    diagnose_group,
    proposed_residual,
    run,
    shuffle_change_probability,
    summarize,
)
from srf4_ref.credit import Observation, semantic_residual

FROZEN, KNOWN, PRIME_STRICT, PRIME = POLICIES


def row(i, p="PASS", s="PASS", **kw):
    return replace(Observation(str(i), 0, True, True, True, True, p, s), **kw)


def group(rows, source="C_TRAIN_PROBE"):
    return dict(
        group_id="g",
        source=source,
        family="TOPK",
        block="b0",
        world_id="w",
        verifier_version="test",
        labels=[
            dict(
                row_id=r.row_id,
                verifier_version="test",
                normalization=dict(
                    j=r.j,
                    protocol_valid=r.protocol_valid,
                    finished=r.finished,
                    sufficient=r.sufficient,
                    defined=r.defined,
                ),
                dimensions={
                    k: dict(status=v)
                    for k, v in zip("PSE", [r.perception, r.selection, "UNKNOWN"], strict=True)
                },
                native_score=dict(value_correct=1),
            )
            for r in rows
        ],
    )


def test_original_policy_matches_reference_for_labels_and_shuffles():
    # Exhaust the semantic pairs for a two-row eligible group, with an outsider.
    pairs = list(itertools.product(["PASS", "FAIL", "UNKNOWN"], repeat=2))
    for a, b in itertools.product(pairs, repeat=2):
        rows = [row(0, *a), row(1, *b), row(2, sufficient=False)]
        for arm in ("J", "EVID", "SELECT", "SELECT_SHUF"):
            for seed in (0, 1, 2026102205):
                assert (
                    proposed_residual(rows, arm, FROZEN, shuffle_seed=seed)
                    == (semantic_residual(rows, arm, shuffle_seed=seed)[0])
                )


def test_p_variation_alone_does_not_count_as_semantic_signal():
    g = diagnose_group(group([row(0), row(1, p="FAIL")]))
    for p in g["policies"].values():
        assert p["active"] and p["select_minus_evid_changed"]
        assert not p["signal"]
        assert p["shuffle_changed"] == 0
        assert p["l1_numerators"]["select_total"] > 0
        assert p["l1_numerators"]["select_minus_shuffle_exact"] == 0


def test_two_mixed_selection_rows_have_half_shuffle_change_probability():
    g = diagnose_group(group([row(0), row(1, s="FAIL")]))
    p = g["policies"][PRIME.name]
    assert p["signal"]
    assert p["expected_shuffle_change_probability"] == 0.5
    assert p["l1_numerators"]["select_minus_shuffle_exact"] == 0.125
    assert shuffle_change_probability(4, 2) == pytest.approx(5 / 6)
    assert shuffle_change_probability(5, 2) == 0.9


def test_exact_shuffle_l1_agrees_with_exhaustive_permutations():
    for n in range(2, 7):
        for k in range(1, n):
            original = [1] * k + [0] * (n - k)
            outcomes = set(itertools.permutations(original))
            expected = sum(
                sum(0.125 * abs(a - b) for a, b in zip(original, outcome, strict=True))
                for outcome in outcomes
            ) / len(outcomes)
            p = diagnose_group(
                group([row(i, s="PASS" if s else "FAIL") for i, s in enumerate(original)])
            )["policies"][PRIME.name]
            assert p["l1_numerators"]["select_minus_shuffle_exact"] == pytest.approx(expected)


def test_sufficiency_and_known_subset_are_independent_ablations():
    rows = [row(0), row(1, s="FAIL", sufficient=False), row(2), row(3, s="UNKNOWN")]
    result = diagnose_group(group(rows))["policies"]
    assert not result[FROZEN.name]["active"]
    assert not result[KNOWN.name]["active"]  # 2/3 < 75%
    assert not result[PRIME_STRICT.name]["active"]
    assert result[PRIME.name]["active"] and result[PRIME.name]["signal"]
    for residual in result[PRIME.name]["residuals"].values():
        assert residual[3] == 0
        assert sum(residual) == pytest.approx(0)


def test_subthreshold_coverage_abstains_every_arm():
    rows = [row(0), row(1, s="FAIL"), row(2, s="UNKNOWN")]
    for arm in ("J", "EVID", "SELECT", "SELECT_SHUF"):
        assert proposed_residual(rows, arm, PRIME) == [0, 0, 0]


def test_total_select_credit_can_be_positive_when_s_fails():
    rows = [row(0), row(1, s="FAIL")] + [row(i, p="FAIL", s="FAIL") for i in range(2, 8)]
    g = diagnose_group(group(rows))
    p = g["policies"][PRIME.name]
    assert p["residuals"]["SELECT"] == [0.203125, 0.078125] + [-0.046875] * 6
    assert p["positive_components_overlapping"]["selection_fail"] == 1


def test_unknown_and_success_rows_remain_zero_and_inputs_immutable():
    rows = [row(0), row(1, s="FAIL"), row(2), row(3, s="UNKNOWN"), row(4, j=1)]
    original = copy.deepcopy(rows)
    for arm in ("J", "EVID", "SELECT", "SELECT_SHUF"):
        result = proposed_residual(rows, arm, PRIME)
        assert result[3:] == [0, 0]
        assert sum(result) == pytest.approx(0)
    assert rows == original


def test_zero_native_denominator_is_not_infinite_power():
    g = diagnose_group(group([row(0), row(1, s="FAIL")]))
    summary = summarize([g], PRIME.name)
    assert summary["zero_native_l1_groups"] == 1
    assert summary["zero_native_l1_select_signal"] > 0
    assert all(value is None for value in summary["R"].values())


@pytest.mark.parametrize("source", ["B_CONFIRM", "C_CONFIRM_ID", "UNKNOWN"])
def test_confirmation_and_unregistered_sources_rejected(source):
    with pytest.raises(ValueError, match="development"):
        diagnose_group(group([row(0)], source=source))


def test_corrupt_base_or_flags_and_duplicate_identity_rejected():
    g = group([row(0), row(1, s="FAIL")])
    g["A_J"] = [1, 0]
    with pytest.raises(ValueError, match="advantages"):
        diagnose_group(g)
    del g["A_J"]
    g["labels"][0]["normalization"]["finished"] = "false"
    with pytest.raises(ValueError, match="booleans"):
        diagnose_group(g)
    with pytest.raises(ValueError, match="Duplicate"):
        diagnose_group(group([row(0), row(0)]))


def test_run_never_overwrites_or_double_counts_versions(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(group([row(0), row(1, s="FAIL")])) + "\n")
    output = tmp_path / "result"
    summary = run([source], output)
    assert summary["strata"][0]["policies"][PRIME.name]["N_sig"] == 1
    with pytest.raises(FileExistsError):
        run([source], output)
    with pytest.raises(ValueError, match="Duplicate group"):
        run([source, source], tmp_path / "duplicate")
    assert not (tmp_path / "duplicate").exists()


def test_world_identity_and_label_version_required():
    g = group([row(0), row(1, s="FAIL")])
    g["world_id"] = None
    with pytest.raises(ValueError, match="identity"):
        diagnose_group(g)
    g["world_id"] = "w"
    g["labels"][0]["verifier_version"] = "another"
    with pytest.raises(ValueError, match="versions disagree"):
        diagnose_group(g)


def test_mixed_measurement_versions_cannot_be_pooled(tmp_path):
    a = group([row(0), row(1, s="FAIL")])
    b = group([row(2), row(3, s="FAIL")])
    b.update(group_id="g2", verifier_version="another")
    for label in b["labels"]:
        label["verifier_version"] = "another"
    source = tmp_path / "mixed.jsonl"
    source.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    with pytest.raises(ValueError, match="separate runs"):
        run([source], tmp_path / "out")
