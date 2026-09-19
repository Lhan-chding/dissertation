import copy
import math

import numpy as np
import pytest

from src.decision_modeling.decision_readout import (
    compare_candidate,
    select_candidate,
    utility_interval,
)
from src.decision_modeling.evaluation import (
    fixed_panel_mc_error,
    independent_reference_comparison,
    paired_scene_bootstrap,
    replay_representations,
)
from src.decision_modeling.intervals import (
    FixedLookAllocation,
    conditional_mass_bounds,
    contribution_interval,
    difference_interval,
    fixed_panel_hoeffding,
    joint_mass_feasible,
)


def test_mass_bounds_qx_and_joint_region():
    result = conditional_mass_bounds({"X": 0.2, "S": 0.1, "W": 0.1, "I": 0.1}, relation_mass=0.3)
    assert result["bounds"]["qX"] == pytest.approx([0.2 / 0.9, 0.7 / 0.9])
    assert result["bounds"]["V"] == pytest.approx([0.4, 0.9])
    assert result["evidence_kind"] == "CONDITIONAL_ON_SCORER"
    assert result["certification"] == "NOT_CERTIFIED"
    assert joint_mass_feasible(result, {"X": 0.7, "S": 0.1, "W": 0.1, "I": 0.1})
    # Simultaneously taking all four marginal upper bounds is impossible.
    assert not joint_mass_feasible(result, {k: result["bounds"][f"p{k}"][1] for k in "XSWI"})
    for allocation in np.random.default_rng(42).dirichlet(np.ones(4), size=100):
        p = {k: result["known_event_mass"][k] + 0.5 * allocation[i] for i, k in enumerate("XSWI")}
        qx = p["X"] / sum(p[k] for k in "XSW")
        assert result["bounds"]["qX"][0] <= qx <= result["bounds"]["qX"][1]


def test_zero_valid_mass_is_unidentified_not_zero():
    result = conditional_mass_bounds({"X": 0, "S": 0, "W": 0, "I": 0.5})
    assert result["bounds"]["qX"] is None
    assert result["bounds"]["pX"] == [0, 0.5]
    assert result["qX_status"].startswith("UNIDENTIFIED")
    with pytest.raises(ValueError, match="exceeds one"):
        conditional_mass_bounds({"X": 0.7, "S": 0.4, "W": 0, "I": 0})
    with pytest.raises(ValueError, match="valid mass"):
        conditional_mass_bounds({"X": 0.1, "S": 0, "W": 0, "I": 0.5}, relation_mass=0.2)


def test_hoeffding_alpha_ledger_unequal_weights_and_zero_count():
    allocation = FixedLookAllocation(("a", "b"), ("pX", "V"))
    result = fixed_panel_hoeffding(
        [[0] * 32, [1] * 32],
        allocation=allocation,
        candidate="a",
        metric="pX",
        look=32,
        prompt_weights=[0.75, 0.25],
    )
    assert result["allocated_cells"] == 12
    assert result["alpha"] == pytest.approx(0.05 / 12)
    assert result["estimate"] == 0.25
    assert result["radius"] == pytest.approx(math.sqrt(math.log(480) * (0.75**2 + 0.25**2) / 64))
    zero = fixed_panel_hoeffding(
        [[0] * 32], allocation=allocation, candidate="a", metric="pX", look=32
    )
    assert zero["interval"][1] > 0
    with pytest.raises(ValueError, match="Unregistered"):
        fixed_panel_hoeffding(
            [[0] * 33], allocation=allocation, candidate="a", metric="pX", look=33
        )
    with pytest.raises(ValueError, match="frozen look"):
        fixed_panel_hoeffding(
            [[0] * 31], allocation=allocation, candidate="a", metric="pX", look=32
        )


def test_allocation_freezes_mutable_config_and_total_error_budget():
    candidates = ["a", "b"]
    allocation = FixedLookAllocation(candidates, ["group1:pX:RAW", "group1:pX:MIX"])
    candidates.append("unregistered")
    assert allocation.candidates == ("a", "b")
    assert sum(
        allocation.cell_alpha(candidate, metric, look)
        for candidate in allocation.candidates
        for metric in allocation.metrics
        for look in allocation.looks
    ) == pytest.approx(allocation.alpha)


def test_mix_range_is_four_and_origin_not_bounded_by_observed_max():
    allocation = FixedLookAllocation(("a",), ("raw", "mix"))
    raw = contribution_interval(
        [[0] * 32], method="RAW", allocation=allocation, candidate="a", metric="raw", look=32
    )
    mix = contribution_interval(
        [[0] * 32], method="MIX", allocation=allocation, candidate="a", metric="mix", look=32
    )
    assert mix["radius"] == pytest.approx(4 * raw["radius"])
    origin = contribution_interval([[0] * 32], method="ORIGIN")
    assert origin["interval"] is None
    assert origin["status"] == "UNRESOLVED"
    with pytest.raises(ValueError, match="Control variates"):
        contribution_interval([[0] * 32], method="MIX", control_variate=True)
    with pytest.raises(ValueError, match="range"):
        contribution_interval(
            [[2.1] * 32], method="MIX", allocation=allocation, candidate="a", metric="mix", look=32
        )


def test_decisions_require_all_groups_and_containment():
    groups = ("g1", "g2")
    kwargs = {"required_groups": groups, "evidence_kind": "CONDITIONAL_ON_SCORER"}
    missing = compare_candidate({"g1": [0, 0.1]}, [-0.1, 0.1], **kwargs)
    assert missing["feasibility"] == "UNKNOWN"
    assert missing["missing_groups"] == ["g2"]
    assert missing["target_status"] == "UNRESOLVED"
    safe = compare_candidate({"g1": [-0.01, 0.1], "g2": [0, 0.1]}, [-0.01, 0.01], **kwargs)
    assert safe["conclusions"] == ["NONINFERIOR_AT_SCALE", "EQUIVALENT_AT_SCALE"]
    assert safe["certification"] == "NOT_CERTIFIED"
    harm = compare_candidate({"g1": [-0.1, -0.011]}, [0.1, 0.2], **kwargs)
    assert harm["status"] == "HARM_DETECTED"


def test_regret_retains_unknown_possible_best():
    result = select_candidate(
        {"safe": [0.5, 0.6], "unknown": [0.1, 0.95], "harm": [0.98, 1]},
        {"safe": "FEASIBLE", "unknown": "UNKNOWN", "harm": "INFEASIBLE"},
        evidence_kind="TEST",
        simultaneous=True,
    )
    assert result["selected"] == "safe"
    assert result["regret_upper"] == pytest.approx(0.45)
    assert result["candidate_set"] == ["safe", "unknown"]
    unresolved = select_candidate(
        {"safe": [0.5, 0.6], "unknown": None},
        {"safe": "FEASIBLE", "unknown": "UNKNOWN"},
        evidence_kind="TEST",
        simultaneous=True,
    )
    assert unresolved["regret_upper"] is None
    assert unresolved["status"] == "UNRESOLVED"
    with pytest.raises(ValueError, match="simultaneous"):
        select_candidate({"a": [0, 1]}, {"a": "FEASIBLE"}, evidence_kind="TEST", simultaneous=False)


def test_preferences_and_interval_difference():
    assert utility_interval(
        {"pX": [0.2, 0.4], "V": [0.5, 0.7], "relation_score": [0, 1]}, "w1"
    ) == pytest.approx([0.21, 0.49])
    assert utility_interval({"pX": [0, 1]}, "w1") is None
    assert difference_interval([0.1, 0.2], [0.3, 0.5]) == pytest.approx([-0.4, -0.1])


def test_scene_bootstrap_keeps_interfaces_and_endpoints_together():
    rows = []
    for family in ("a", "b", "c"):
        for scene in range(4):
            # Opposite paired-interface changes cancel exactly within scene.
            for interface, delta in (("text", 1), ("image", -1)):
                rows.append(
                    {
                        "family": family,
                        "base_scene": f"{family}{scene}",
                        "interface": interface,
                        "left": 2 + scene + delta,
                        "right": 2 + scene,
                    }
                )
    result = paired_scene_bootstrap(rows)
    assert result["replicates"] == 5000
    assert result["interval"] == [0, 0]
    assert result["nested_completion_resampling"] is False
    assert result["finite_sample_certificate"] is False
    with pytest.raises(ValueError, match="Missing paired"):
        paired_scene_bootstrap(rows[:-1])


def test_fixed_panel_mc_report_is_separate_and_zero_se_not_certified():
    result = fixed_panel_mc_error([[1] * 32, [0] * 32])
    assert result["estimate"] == 0.5
    assert result["empirical_standard_error"] == 0
    assert result["zero_empirical_se_is_certainty"] is False
    assert result["scene_sampling_included"] is False


def test_replay_never_passes_future_and_isolates_mutations():
    streams = {"p1": [{"value": n} for n in range(512)], "p2": [{"value": n} for n in range(512)]}
    original = copy.deepcopy(streams)
    seen = []

    def builder(prefix, level):
        n = len(prefix["p1"])
        assert all(len(rows) == n and rows[-1]["value"] == n - 1 for rows in prefix.values())
        seen.append((level, n))
        prefix["p1"][0]["value"] = -999
        return {"n": n, "level": level}

    def rule(state):
        return state.pop("n")

    result = replay_representations(
        streams, representation_builder=builder, rules={f"Z{i}": rule for i in range(5)}
    )
    assert streams == original
    assert len(seen) == 15
    for row in result:
        assert set(row["decisions"].values()) == {row["look"]}
    short = {"p1": original["p1"][:32], "p2": original["p2"]}
    with pytest.raises(ValueError, match="Every stream"):
        replay_representations(
            short, representation_builder=builder, rules={f"Z{i}": rule for i in range(5)}
        )


def test_reference_remains_uncertain():
    result = independent_reference_comparison([0.3, 0.5], {"a": [0.4, 0.6], "b": [0.7, 0.8]})
    assert result["regret_interval"] == pytest.approx([0.2, 0.5])
    assert result["reference_unresolved_fraction"] == 0.5
    assert result["reference_is_exact_truth"] is False
