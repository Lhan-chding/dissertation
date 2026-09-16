import copy

import numpy as np
import pytest

from src.modeling_v4.nonlinear import fit_signature_residual, validate_training_partition


def dataset():
    rng = np.random.default_rng(4)
    signatures = rng.normal(size=(768, 576)) * 0.02
    state = np.repeat(rng.normal(size=(6, 2)), 128, axis=0)
    y = np.zeros((768, 2, 4))
    y[:, :, 0] = signatures[:, :2]
    y[:, :, 3] = -signatures[:, 2:4]
    records, provenance, states = [], [], []
    for i in range(768):
        origin = f"o{i // 128}"
        seed = i // 256 + 1
        records.append(
            dict(
                seed=seed,
                origin_id=origin,
                pair_id=f"p{i}",
                role="development",
                bank_role="calibration",
                is_alias=False,
            )
        )
        provenance.append(
            dict(
                native_width=576,
                prompts=36,
                draws_per_prompt=16,
                panel_id=f"panel-{origin}",
                panel_role="signature",
                origin_id=origin,
                baseline_fingerprint=f"base-{origin}",
                candidate_fingerprint=f"candidate-{i}",
                sample_identity_sha256="a" * 64,
                split_identity_sha256="b" * 64,
                split_guard="VERIFIED",
                score_definition="FULL_SEQUENCE_LOGPROB_DIFFERENCE",
                target_outcomes_used=False,
                paid_feature=True,
                paid_score_actions=1152,
                required_score_entries=1152,
            )
        )
        states.append(dict(stage="ORIGIN", origin_id=origin, panel_role="signature"))
    return (
        signatures,
        state,
        y,
        records,
        dict(
            development_seeds=[1, 2, 3],
            validation_seed=3,
            signature_provenance=provenance,
            state_provenance=states,
            probe_groups=["group-a", "group-a"],
        ),
    )


def test_whole_seed_and_actual_unique_pair_guards():
    e, z, y, records, kwargs = dataset()
    partition = validate_training_partition(e, z, y, records, **kwargs)
    assert partition["training_indices"].tolist() == list(range(512))
    assert partition["validation_indices"].tolist() == list(range(512, 768))
    assert partition["nonalias_training_pair_count"] == 512
    altered = copy.deepcopy(records)
    altered[0]["role"] = "interval_calibration"
    with pytest.raises(ValueError, match="development"):
        validate_training_partition(e, z, y, altered, **kwargs)
    changed = copy.deepcopy(kwargs)
    changed["signature_provenance"][1]["candidate_fingerprint"] = changed["signature_provenance"][
        0
    ]["candidate_fingerprint"]
    with pytest.raises(ValueError, match=r"duplicate.*pair"):
        validate_training_partition(e, z, y, records, **changed)
    with pytest.raises(ValueError, match="512"):
        validate_training_partition(
            e[1:],
            z[1:],
            y[1:],
            records[1:],
            **{
                **kwargs,
                "signature_provenance": kwargs["signature_provenance"][1:],
                "state_provenance": kwargs["state_provenance"][1:],
            },
        )


def test_provenance_forbids_endpoint_state_and_target_outcomes():
    e, z, y, records, kwargs = dataset()
    kwargs["signature_provenance"][0]["target_outcomes_used"] = True
    with pytest.raises(ValueError, match="signature"):
        validate_training_partition(e, z, y, records, **kwargs)
    kwargs["signature_provenance"][0]["target_outcomes_used"] = False
    kwargs["state_provenance"][0]["stage"] = "CANDIDATE_ENDPOINT"
    with pytest.raises(ValueError, match="origin"):
        validate_training_partition(e, z, y, records, **kwargs)
    with pytest.raises(ValueError, match="576"):
        validate_training_partition(e[:, :2], z, y, records, **kwargs)


def test_three_real_networks_zero_anchor_and_holdout_labels_do_not_train():
    pytest.importorskip("torch")
    e, z, y, records, kwargs = dataset()
    model = fit_signature_residual(e, z, y, records, epochs=1, **kwargs)
    modified = y.copy()
    modified[512:] += 100
    other = fit_signature_residual(e, z, modified, records, epochs=1, **kwargs)
    np.testing.assert_array_equal(model["predict"](e[:2], z[:2]), other["predict"](e[:2], z[:2]))
    np.testing.assert_array_equal(model["predict"](np.zeros((2, 576)), z[:2]), np.zeros((2, 2, 4)))
    assert model["predict_members"](e[:2], z[:2]).shape == (3, 2, 2, 4)
    assert not np.array_equal(
        model["predict_members"](e[:2], z[:2])[0], model["predict_members"](e[:2], z[:2])[1]
    )
    assert model["metadata"]["source_training_seed_count"] == 2
    assert model["metadata"]["network_initializations_are_source_seeds"] is False
    assert model["metadata"]["group_channels"] == ["delta_pX", "delta_v=-delta_pI"]
    assert len(model["validation_metrics"]) == 3
    assert model["validation_metrics"][0]["raw4_mse"] != other["validation_metrics"][0]["raw4_mse"]
