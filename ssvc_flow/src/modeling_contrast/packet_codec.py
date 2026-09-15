"""Lossless NPZ codec for repeated paid action scores.

Only observed (prompt, action) values are stored. Missing actions are never
filled in, scored, or inferred. Float payloads are dictionary encoded as raw
integer bits, preserving signed zero, byte order, and NaN payloads exactly.
Ordinary arrays retain their keys so fitting can load its small summaries
without decoding primitive samples.
"""

from __future__ import annotations

import json
import re

import numpy as np

PREFIX = "__packet_codec_"
MANIFEST = PREFIX + "manifest__"
VERSION = 1
_SAMPLE = re.compile(r"^(.*?)(sample_\d+_)(?:baseline_logp|candidate_logp|logrho)$")


def _small_uint(maximum):
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if maximum <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    raise ValueError("Integer index exceeds uint64")


def _action_reference(name, array, arrays):
    match = _SAMPLE.match(name)
    if not match or array.ndim != 2:
        return None
    prefix, sample = match.groups()
    # LR-ORIGIN scores in sample_1/sample_2 all refer to sample_0 actions.
    for key in (prefix + sample + "actions", prefix + "sample_0_actions"):
        if key in arrays:
            actions = arrays[key]
            if actions.shape == array.shape and actions.dtype.kind in "iu":
                return key
    return None


def _sparse_scores(bits, actions):
    if not actions.size or (actions < 0).any():
        return None
    stride = int(actions.max()) + 1
    if stride * len(actions) > np.iinfo(np.uint64).max:
        return None
    joint = (np.arange(len(actions), dtype=np.uint64)[:, None] * stride + actions).ravel()
    keys, positions, inverse = np.unique(joint, return_index=True, return_inverse=True)
    values = bits.ravel()[positions]
    if not np.array_equal(values[inverse], bits.ravel()):
        # Arbitrary inputs may contain nonconstant score values for one action.
        # Fall back to the full exact bit dictionary rather than overwriting.
        return None
    return keys, values, stride


def encode_arrays(arrays):
    """Return primitive arrays suitable for np.savez_compressed, no pickle."""
    originals = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or name.startswith(PREFIX):
            raise ValueError("Nonreserved string array names required")
        value = np.asarray(value)
        if value.dtype.hasobject:
            raise ValueError("Object arrays are forbidden")
        originals[name] = value
    encoded, records, groups, key_parts = {}, [], {}, []
    key_offset = 0
    # Group the same bank/policy/score field across replicas so equivalent sparse
    # score tables fall within DEFLATE's 32 KiB window. This changes storage order
    # only; original names, sample order and all bytes survive in the manifest.
    storage_order = sorted(originals, key=lambda name: (re.sub(r"^r\d+(?=b\d+_)", "", name), name))
    for name in storage_order:
        array = originals[name]
        if (
            array.dtype.kind != "f"
            or array.dtype.itemsize not in (2, 4, 8)
            or not name.endswith(("logp", "logrho"))
        ):
            encoded[name] = array
            continue
        contiguous = np.ascontiguousarray(array)
        bit_dtype = np.dtype(f"u{array.dtype.itemsize}")
        bits = contiguous.view(bit_dtype).reshape(array.shape)
        action_name = _action_reference(name, array, originals)
        sparse = _sparse_scores(bits, originals[action_name]) if action_name is not None else None
        record = {
            "name": name,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "word_bytes": array.dtype.itemsize,
        }
        if sparse is None:
            values = bits.ravel()
            record["mode"] = "bit_dictionary"
        else:
            keys, values, stride = sparse
            record.update(
                mode="paid_action_lookup",
                actions=action_name,
                stride=stride,
                key_offset=key_offset,
                key_count=len(keys),
            )
            key_parts.append(keys)
            key_offset += len(keys)
        group = groups.setdefault(array.dtype.itemsize, {"parts": [], "size": 0})
        record.update(value_offset=group["size"], value_count=len(values))
        group["parts"].append(values)
        group["size"] += len(values)
        records.append(record)
    for word_bytes, group in groups.items():
        all_bits = np.concatenate(group["parts"])
        unique, inverse = np.unique(all_bits, return_inverse=True)
        encoded[f"{PREFIX}dictionary_{word_bytes}__"] = unique
        encoded[f"{PREFIX}indices_{word_bytes}__"] = inverse.astype(
            _small_uint(max(len(unique) - 1, 0))
        )
    if key_parts:
        keys = np.concatenate(key_parts)
        encoded[PREFIX + "action_keys__"] = keys.astype(_small_uint(int(keys.max())))
    manifest = {
        "version": VERSION,
        "original_keys": list(originals),
        "records": records,
        "float_contract": "bitwise_exact",
        "unqueried_action_values_stored": False,
    }
    encoded[MANIFEST] = np.frombuffer(
        json.dumps(manifest, separators=(",", ":"), ensure_ascii=True).encode(), dtype=np.uint8
    )
    return encoded


def decode_arrays(encoded, *, keys=None):
    """Decode all arrays or selected original names; accepts an NpzFile mapping."""
    if MANIFEST not in encoded:
        names = list(encoded) if keys is None else list(keys)
        return {name: np.array(encoded[name], copy=True) for name in names}
    manifest = json.loads(np.asarray(encoded[MANIFEST], dtype=np.uint8).tobytes())
    if manifest.get("version") != VERSION:
        raise ValueError("Unsupported packet codec version")
    original_keys = manifest["original_keys"]
    if len(original_keys) != len(set(original_keys)):
        raise ValueError("Duplicate original array names")
    names = original_keys if keys is None else list(keys)
    if not set(names).issubset(original_keys):
        raise ValueError("Requested array absent from codec manifest")
    records = {record["name"]: record for record in manifest["records"]}
    result = {}
    dictionary_cache, index_cache = {}, {}
    action_keys = None
    for name in names:
        if name not in records:
            result[name] = np.array(encoded[name], copy=True)
            continue
        record = records[name]
        word_bytes = record["word_bytes"]
        dtype = np.dtype(record["dtype"])
        if dtype.kind != "f" or dtype.itemsize != word_bytes or word_bytes not in (2, 4, 8):
            raise ValueError("Invalid encoded float word type")
        if word_bytes not in dictionary_cache:
            dictionary_cache[word_bytes] = np.asarray(encoded[f"{PREFIX}dictionary_{word_bytes}__"])
            index_cache[word_bytes] = np.asarray(encoded[f"{PREFIX}indices_{word_bytes}__"])
        dictionary, indices = dictionary_cache[word_bytes], index_cache[word_bytes]
        offset, count = record["value_offset"], record["value_count"]
        if offset < 0 or count < 0 or offset + count > len(indices):
            raise ValueError("Invalid dictionary index bounds")
        codes = indices[offset : offset + count]
        if codes.dtype.kind != "u" or (codes.size and int(codes.max()) >= len(dictionary)):
            raise ValueError("Invalid float dictionary index")
        values = dictionary[codes]
        shape = tuple(record["shape"])
        if record["mode"] == "paid_action_lookup":
            if action_keys is None:
                action_keys = np.asarray(encoded[PREFIX + "action_keys__"])
            offset, key_count = record["key_offset"], record["key_count"]
            if offset < 0 or key_count != count or offset + key_count > len(action_keys):
                raise ValueError("Invalid paid action lookup bounds")
            sparse_keys = action_keys[offset : offset + key_count]
            actions = np.asarray(encoded[record["actions"]])
            if (
                actions.shape != shape
                or actions.ndim != 2
                or actions.dtype.kind not in "iu"
                or (actions < 0).any()
            ):
                raise ValueError("Invalid action reference")
            joint = (
                np.arange(len(actions), dtype=np.uint64)[:, None] * record["stride"] + actions
            ).ravel()
            indices = np.searchsorted(sparse_keys, joint)
            if (indices >= len(sparse_keys)).any() or not np.array_equal(
                sparse_keys[indices], joint
            ):
                raise ValueError("Missing paid action score; no inference permitted")
            values = values[indices]
        elif record["mode"] != "bit_dictionary":
            raise ValueError("Unknown packet codec mode")
        if values.size != int(np.prod(shape, dtype=np.int64)):
            raise ValueError("Decoded float shape mismatch")
        result[name] = np.ascontiguousarray(values).view(dtype).reshape(shape).copy()
    return result
