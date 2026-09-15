import numpy as np
import pytest

from src.modeling_contrast.packet_codec import decode_arrays, encode_arrays


def assert_bits_equal(original, decoded):
    assert original.keys() == decoded.keys()
    for key in original:
        assert original[key].dtype == decoded[key].dtype
        assert original[key].shape == decoded[key].shape
        assert original[key].tobytes() == decoded[key].tobytes()


def test_codec_sparse_paid_action_scores_roundtrip_npz(tmp_path):
    actions = np.array([[0, 5, 0, 5], [2, 2, 7, 2]], dtype=np.int16)
    labels = np.array([[0, 3, 0, 3], [2, 2, 3, 2]], dtype=np.int8)
    logp = np.array([[-0.1, -0.5, -0.1, -0.5], [-0.2, -0.2, -0.7, -0.2]])
    arrays = {
        "r0b0_sample_0_actions": actions,
        "r0b0_sample_0_labels": labels,
        "r0b0_sample_0_logrho": logp,
        "r0b0_sample_1_baseline_logp": logp.copy(),
        "raw": np.zeros((1, 2, 4)),
    }
    encoded = encode_arrays(arrays)
    assert all(value.dtype.kind != "O" for value in encoded.values())
    assert "r0b0_sample_0_logrho" not in encoded
    path = tmp_path / "encoded.npz"
    np.savez_compressed(path, **encoded)
    with np.load(path, allow_pickle=False) as loaded:
        decoded = decode_arrays(loaded)
    assert_bits_equal(arrays, decoded)
    selected = decode_arrays(encoded, keys=["raw"])
    assert set(selected) == {"raw"}


def test_codec_bitwise_signed_zero_nan_payload_and_endianness():
    bits = np.array([0, 1 << 63, 0x7FF8000000000042, 0xFFF0000000000000], dtype=np.uint64)
    arrays = {
        "edge_logp": bits.view(np.float64),
        "big_endian_logp": np.array([0.0, -0.0, -0.3], dtype=">f8"),
        "float32_logp": np.array([-1.0, -1.0, -2.0], dtype=np.float32),
    }
    assert_bits_equal(arrays, decode_arrays(encode_arrays(arrays)))


def test_codec_same_action_different_scores_falls_back_without_loss():
    arrays = {
        "sample_0_actions": np.array([[1, 1, 2]], dtype=np.int16),
        "sample_0_logrho": np.array([[1.0, np.nextafter(1.0, 2.0), 2.0]]),
    }
    assert_bits_equal(arrays, decode_arrays(encode_arrays(arrays)))


def test_codec_rejects_object_and_reserved_names():
    with pytest.raises(ValueError):
        encode_arrays({"bad": np.array([object()])})
    with pytest.raises(ValueError):
        encode_arrays({"__packet_codec_manifest__": np.array([1])})


def test_codec_plain_legacy_arrays_load_and_bad_version_rejected():
    plain = {"x": np.array([1.0, 2.0])}
    assert_bits_equal(plain, decode_arrays(plain))
    encoded = encode_arrays({"x_logp": np.array([1.0, 1.0])})
    encoded["__packet_codec_manifest__"] = np.frombuffer(b'{"version":99}', dtype=np.uint8)
    with pytest.raises(ValueError, match="version"):
        decode_arrays(encoded)
