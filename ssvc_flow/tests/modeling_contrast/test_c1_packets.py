import os
from pathlib import Path

import numpy as np
import pytest

from src.modeling_contrast.c1_packets import repair_c1_auxiliary, same_bank_counts


def regression_parent_root():
    from src.modeling_contrast.study import legacy_root

    explicit = os.environ.get("SSVC_TEST_PARENT_ROOT")
    if explicit is not None:
        root = Path(explicit)
        assert (root / "manifest.json").is_file(), "explicit regression evidence unavailable"
        return root
    return legacy_root()


def test_same_bank_counts_does_not_resolve_through_another_bank():
    row = {"array_prefix": "r5b5_", "sample_names": ["shared", "candidate"]}
    primitive = {
        "r5b5_sample_0_counts": np.array([[1, 2, 3, 10]]),
        "r5b13_sample_0_counts": np.array([[9, 2, 3, 2]]),
    }
    np.testing.assert_array_equal(same_bank_counts(primitive, row)["shared"], [[1, 2, 3, 10]])
    assert same_bank_counts(primitive, {"array_prefix": "r5b2_", "sample_names": []}) == {}


def test_real_seed101_anchor24_c1_overlay_preserves_bank_and_source(tmp_path):
    import json

    from src.modeling_contrast.observation_study import load_packet
    from src.modeling_contrast.packet_codec import decode_arrays
    from src.modeling_contrast.study import digest

    root = regression_parent_root()
    path = Path("runs/modeling_contrast_v2/N1/packets/seed101_X_VALID/a1/O_IND_n64")
    if not (root / "manifest.json").is_file() or not (path / "packets.json").is_file():
        pytest.skip("local immutable legacy and N1 regression evidence unavailable")
    entry = next(
        e
        for e in json.loads((root / "manifest.json").read_text())["trajectories"]
        if e["id"] == "seed101_X_VALID"
    )
    original = load_packet(path)
    before = digest(path / "packet_arrays.npz")
    with np.load(path / "packet_arrays.npz", allow_pickle=False) as handle:
        primitives = decode_arrays(handle)
    rows = original["metadata"]["packets"]
    row5 = next(r for r in rows if r["noise_replica"] == 5 and r["bank"] == 5)
    row13 = next(r for r in rows if r["noise_replica"] == 5 and r["bank"] == 13)
    fp = row5["policy_fingerprints"][0][0]
    assert fp == row13["policy_fingerprints"][0][0]
    local = same_bank_counts(primitives, row5)[fp]
    evaluation = same_bank_counts(primitives, row13)[fp]
    assert not np.array_equal(local, evaluation)
    fixed = repair_c1_auxiliary(root, entry, 1, original, tmp_path / "overlay")
    np.testing.assert_array_equal(fixed["legacy_total_levels"][5, 5, 0] * 64, local)
    assert not np.array_equal(fixed["legacy_total_levels"][5, 5, 0] * 64, evaluation)
    np.testing.assert_array_equal(
        fixed["legacy_total_levels"][:5], original["legacy_total_levels"][:5]
    )
    np.testing.assert_array_equal(fixed["legacy_origin_counts"], original["legacy_origin_counts"])
    assert digest(path / "packet_arrays.npz") == before
    np.testing.assert_array_equal(
        original["legacy_total_levels"], primitives["legacy_total_levels"]
    )
    assert fixed["c1_auxiliary_sha256"] != before
    replay = repair_c1_auxiliary(root, entry, 1, original, tmp_path / "overlay")
    assert replay["c1_auxiliary_sha256"] == fixed["c1_auxiliary_sha256"]
    np.testing.assert_array_equal(replay["legacy_total_levels"], fixed["legacy_total_levels"])


def test_future_c1_packets_use_same_bank_counts_and_explicit_data_role(tmp_path):
    import json

    from src.modeling_contrast.observation_study import measure_unit
    from src.modeling_contrast.packet_codec import decode_arrays

    root = regression_parent_root()
    if not (root / "manifest.json").is_file():
        pytest.skip("local immutable legacy regression evidence unavailable")
    entry = next(
        e
        for e in json.loads((root / "manifest.json").read_text())["trajectories"]
        if e["id"] == "seed101_X_VALID"
    )
    out = tmp_path / "new_packet"
    # The data_role string is an interface witness; input identity remains the
    # actual legacy trajectory and is never presented as a new training seed.
    packet = measure_unit(
        root,
        entry,
        1,
        "O_IND",
        64,
        out,
        replicas=6,
        banks=(5, 13),
        data_role="fresh_interface_test_only",
    )
    assert packet["metadata"]["v2_role"] == "fresh_interface_test_only"
    assert packet["metadata"]["c1_auxiliary_fit_evaluation_independent"] is True
    row0 = next(
        r for r in packet["metadata"]["packets"] if r["noise_replica"] == 0 and r["bank"] == 5
    )
    assert row0["observation_origin"] == "FRESH_CPU_COUNTS"
    row5 = next(
        r for r in packet["metadata"]["packets"] if r["noise_replica"] == 5 and r["bank"] == 5
    )
    with np.load(out / "packet_arrays.npz", allow_pickle=False) as handle:
        primitives = decode_arrays(handle)
    fp = row5["policy_fingerprints"][0][0]
    np.testing.assert_array_equal(
        packet["legacy_total_levels"][5, 5, 0] * 64, same_bank_counts(primitives, row5)[fp]
    )
