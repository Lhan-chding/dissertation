"""Scientific information-boundary counterexamples, no model/GPU claims."""

import copy
import json

import numpy as np
import pytest

from src.prospective_selection.features import LEVELS, PreDecisionPacket, build_predecision_packet
from src.prospective_selection.kernels import FeatureKernel
from src.prospective_selection.semantics import (
    event_probabilities,
    repair_features,
    reward_bin,
    validate_repair,
)


def row(event="W", b=0, prompt="p", numerator=0, denominator=2):
    return dict(
        prompt_id=prompt,
        family="trend",
        interface="symbolic",
        event=event,
        relation_numerator=numerator,
        relation_denominator=denominator,
        F=None if event == "I" else int(event == "X"),
        B=None if event == "I" else b,
        M=None if event == "I" else b + int(event == "X"),
    )


def packet(level=LEVELS[3], rows=None, past=None, lineage="l0", step=32, origin=None, meta=None):
    rows = [row()] if rows is None else rows
    return build_predecision_packet(
        origin_id=origin or f"{lineage}:{step}",
        lineage_id=lineage,
        source_recipe="R0",
        step=step,
        current_rows=rows,
        history_rows=rows if past is None else past,
        feature_level=level,
        known_training_metadata=meta,
    )


def test_repair_identities_invalid_atom_and_exact_relation_keys():
    assert reward_bin("W", 1, 2) == reward_bin("W", 2, 4)
    assert reward_bin("W", 999, 1000) != reward_bin("W", 1, 1)
    observed, truth = [9, 2, 3, 4], [1, 2, 3, 4]
    for output in (observed, truth, [1, 9, 3, 4], [8, 9, 7, 4]):
        repair = repair_features(observed, truth, output)
        assert (output == truth) == (repair["F"] == 1 and repair["B"] == 0)
        assert repair["coord_accuracy"] == (repair["F"] + 3 - repair["B"]) / 4
        assert repair["Bdamage"] == repair["B"] / 3
    invalid = repair_features(observed, truth, "[1,2,3,4]")
    assert invalid["F"] is invalid["B"] is invalid["M"] is None
    assert invalid["Bdamage"] == 1 and invalid["coord_accuracy"] == 0
    with pytest.raises(ValueError):
        repair_features(truth, truth, truth)
    with pytest.raises(ValueError):
        validate_repair({**row(), "M": 4})
    assert sum(event_probabilities(0.2, 0.4, 0.7).values()) == 1


def test_reward_equivalent_states_remain_hidden_at_z2_and_visible_at_z3():
    a2, b2 = packet(LEVELS[2]), packet(LEVELS[2], [row(b=2)])
    assert a2.payload_json == b2.payload_json
    a3, b3 = packet(), packet(rows=[row(b=2)])
    assert a3.payload_json != b3.payload_json
    k2, k3 = FeatureKernel.fit([a2, b2]), FeatureKernel.fit([a3, b3])
    assert np.allclose(k2.gram([a2, b2]), 1)
    assert k3.gram([a3, b3])[0, 1] < k3.gram([a3, b3])[0, 0]


def test_layer_whitelists_nested_outcomes_and_no_raw_path_access():
    p = packet(LEVELS[2])
    assert "repair" not in p.payload_json and "parsed_world" not in p.payload_json
    for key, value in (("repair_histograms", {}), ("future_H32", {}), ("raw_completion", "secret")):
        data = p.to_dict()
        data["current"][key] = value
        with pytest.raises(ValueError):
            PreDecisionPacket.from_dict(data)
    data = p.to_dict()
    data["known_training_metadata"]["lineage_id"] = 1
    with pytest.raises(ValueError):
        PreDecisionPacket.from_dict(data)
    with pytest.raises(ValueError):
        packet(rows=[{**row(), "outcomes": {"R0": 0.9}}])
    data = p.to_dict()
    data["current"]["n"] = 999
    assert p.to_dict()["current"]["n"] == 1  # copy cannot mutate immutable packet


def test_counts_no_pseudoprior_invalid_marginal_and_history_alignment():
    p = packet(rows=[row("I"), row(), row()])
    s = p.to_dict()["current"]
    assert s["repair_histograms"]["p"]["counts"]["INVALID"] == 1
    assert s["repair_marginals"]["p"]["B"]["counts"] == {"0": 2, "INVALID": 1}
    assert s["reward_histograms"]["p"]["n"] == 3
    with pytest.raises(ValueError):
        packet(past=[row(prompt="different")])
    with pytest.raises(ValueError):
        build_predecision_packet(
            origin_id="o",
            lineage_id="l",
            source_recipe="R0",
            step=32,
            current_rows=[row()],
            history_rows=[row()],
            history_step=25,
            feature_level=LEVELS[0],
        )


@pytest.mark.parametrize("level", LEVELS)
def test_kernel_psd_lineage_invariance_and_fit_only_scaling(level):
    packets = [
        packet(
            level,
            [row(b=b)] * (b + 1),
            past=[row()],
            lineage=f"fit{b}",
            step=32 + 8 * b,
            meta={"gradient_norm_mean": b + 1},
        )
        for b in range(3)
    ]
    kernel = FeatureKernel.fit(packets)
    k = kernel.gram(packets)
    assert np.linalg.eigvalsh(k).min() >= -1e-10
    serialized = copy.deepcopy(kernel.to_dict())
    altered = packets[0].to_dict()
    altered["origin_id"], altered["lineage_id"] = "unseen-id", "unseen-lineage"
    assert np.allclose(
        kernel.gram([packets[0]], packets),
        kernel.gram([PreDecisionPacket.from_dict(altered)], packets),
    )
    test = packet(level, [row(b=3)], lineage="test", step=96, meta={"gradient_norm_mean": 1000})
    kernel.gram([test], packets)
    assert kernel.to_dict() == serialized
    assert FeatureKernel.from_dict(json.loads(json.dumps(serialized))).gram(
        packets
    ) == pytest.approx(k)


def test_histogram_replication_and_trend_redundancy_identity():
    rows = [row("X", numerator=2), row("W", numerator=2, b=1), row("W", numerator=1), row("I")]
    p, q = packet(rows=rows), packet(rows=rows * 3)
    kernel = FeatureKernel.fit([p, q])
    assert np.allclose(kernel.gram([p, q]), 1)
    values = np.array([r["relation_numerator"] / r["relation_denominator"] for r in rows])
    px = sum(r["event"] == "X" for r in rows) / len(rows)
    assert 2 * np.mean(values**2) - np.mean(values) - px == 0.25


def test_serialized_histograms_reject_smuggled_atoms_and_inconsistent_projections():
    p = packet()
    d = p.to_dict()
    d["current"]["reward_histograms"]["p"]["counts"] = {"W:0/1|future_h32_reward=.9": 1}
    with pytest.raises(ValueError, match="reward atom"):
        PreDecisionPacket.from_dict(d)
    d = p.to_dict()
    d["current"]["moments_global"][0] = 0.9
    with pytest.raises(ValueError, match="moments"):
        PreDecisionPacket.from_dict(d)
    d = p.to_dict()
    d["current"]["repair_marginals"]["p"]["B"]["counts"] = {"2": 1}
    with pytest.raises(ValueError, match="marginals"):
        PreDecisionPacket.from_dict(d)
