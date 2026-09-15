import numpy as np
import pytest

from src.modeling_contrast.observations import ToyWorldService, measure
from src.modeling_contrast.packet_codec import decode_arrays, encode_arrays
from src.modeling_contrast.packet_reconstruction import reconstruct_measurements


def as_unit(packet):
    prefix = "r0b0_"
    primitives = {
        prefix + key: value
        for key, value in packet.arrays(include_covariance=False).items()
        if key.startswith("sample_")
    }
    row = packet.metadata()
    row.update(bank=0, noise_replica=0, array_prefix=prefix, sample_names=list(packet.samples))
    metadata = {
        "method": packet.method,
        "n": packet.n,
        "replicas": 1,
        "banks": [0],
        "packets": [row],
    }
    return primitives, metadata


@pytest.mark.parametrize("method", ["O_IND", "O_CRN", "O_LR_ORIGIN", "O_LR_MIX"])
@pytest.mark.parametrize("aliased", [False, True])
def test_paid_primitives_reconstruct_every_float_bit(method, aliased):
    p = np.array([[0.2, 0.15, 0.2, 0.1, 0.2, 0.15]])
    q = p + np.array([[0.001, -0.002, 0.002, 0, 0, -0.001]])
    tables = {"b": p, "u": p if aliased else q, "v": q, "o": p}
    service = ToyWorldService.from_action_probabilities(tables, [[2, 3, 0, 1, 2, 3]])
    packet = measure(service, [("b", "u"), ("b", "v")], origin_id="o", method=method, n=64, seed=31)
    primitives, metadata = as_unit(packet)
    decoded = decode_arrays(encode_arrays(primitives))
    reconstructed = reconstruct_measurements(decoded, metadata)
    for key, original in [
        ("raw", packet.raw_event_estimate),
        ("helmert", packet.helmert_estimate),
        ("covariance", packet.covariance),
    ]:
        assert reconstructed[key][0, 0].tobytes() == original.tobytes()


def test_reversed_mixture_pair_and_legacy_unsigned_counts_reconstruct():
    p = np.array([[0.2, 0.15, 0.2, 0.1, 0.2, 0.15]])
    q = p + np.array([[0.001, -0.002, 0.002, 0, 0, -0.001]])
    tables = {"b": p, "u": q, "o": p}
    cats = [[2, 3, 0, 1, 2, 3]]
    for method in ("O_LR_MIX", "O_IND"):
        service = ToyWorldService.from_action_probabilities(tables, cats)
        counts = {
            "b": np.array([[4, 1, 9, 2]], dtype=np.uint16),
            "u": np.array([[3, 1, 9, 3]], dtype=np.uint16),
        }
        packet = measure(
            service,
            [("b", "u"), ("u", "b")],
            origin_id="o",
            method=method,
            n=16,
            seed=3,
            legacy_counts=counts if method == "O_IND" else None,
        )
        primitives, metadata = as_unit(packet)
        result = reconstruct_measurements(primitives, metadata)
        assert result["raw"][0, 0].tobytes() == packet.raw_event_estimate.tobytes()
        assert result["covariance"][0, 0].tobytes() == packet.covariance.tobytes()


def test_missing_paid_scores_and_duplicate_packet_metadata_fail():
    p = np.array([[0.2, 0.15, 0.2, 0.1, 0.2, 0.15]])
    q = p + np.array([[0.001, -0.002, 0.002, 0, 0, -0.001]])
    packet = measure(
        ToyWorldService.from_action_probabilities({"b": p, "u": q, "o": p}, [[2, 3, 0, 1, 2, 3]]),
        [("b", "u"), ("b", "u")],
        origin_id="o",
        method="O_LR_ORIGIN",
        n=16,
        seed=3,
    )
    primitives, metadata = as_unit(packet)
    key = next(key for key in primitives if key.endswith("candidate_logp"))
    del primitives[key]
    with pytest.raises(ValueError, match="Missing"):
        reconstruct_measurements(primitives, metadata)
    primitives, metadata = as_unit(packet)
    metadata["packets"].append(metadata["packets"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        reconstruct_measurements(primitives, metadata)


def test_stored_core_legacy_path_needs_no_primitive_reconstruction():
    arrays = {key: np.ones((1,)) for key in ("raw", "helmert", "covariance")}
    result = reconstruct_measurements(arrays, {})
    assert all(np.array_equal(result[key], arrays[key]) for key in arrays)
