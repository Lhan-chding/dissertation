import io
import json
import math
import struct

import pytest

from scripts.sr_f3_ce.compact_codec import MAGIC, Decoder, Encoder, Limits, dumps, loads


def signature(value):
    if type(value) is float:
        return ("float", struct.pack("<d", value))
    if type(value) is dict:
        return ("dict", [(key, signature(child)) for key, child in value.items()])
    if type(value) is list:
        return ("list", [signature(child) for child in value])
    return (type(value).__name__, value)


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        0,
        -1,
        2**300,
        -(2**300),
        "",
        "你好",
        "\ud800",
        0.0,
        -0.0,
        1.5,
        math.pi,
        1e300,
        5e-324,
        [],
        {},
        [0, -3, 2**100],
        [0.0, -0.0, 1.5],
        [1.5, math.pi],
        {"z": [False, {"a": "重复", "b": "重复"}], "a": [1, 1.0, None]},
    ],
)
def test_exact_values(value):
    assert signature(loads(dumps(value))) == signature(value)


def test_repeated_records_and_independence():
    value = {"identity": {"prompt": "x" * 68000, "nested": {"a": [1, 2]}}, "b": "x"}
    stream = io.BytesIO()
    enc = Encoder(stream)
    enc.write(value)
    first_size = stream.tell()
    enc.write(value)
    assert stream.tell() - first_size < 10
    stream.seek(0)
    dec = Decoder(stream)
    first, second = dec.read(), dec.read()
    first["identity"]["nested"]["a"][0] = 900
    assert second["identity"]["nested"]["a"] == [1, 2]
    with pytest.raises(EOFError):
        dec.read()


def test_numeric_array_width():
    narrow = dumps([1.5] * 100)
    wide = dumps([math.pi] * 100)
    assert len(wide) - len(narrow) == 400
    assert narrow[len(MAGIC) + 1 : len(MAGIC) + 2] == b"F"


def test_table_budget_exhaustion():
    limits = Limits(max_intern_entries=2, max_intern_bytes=12)
    values = [{"k": "abcdefgh"}, {"x": ["abcdefgh", "z"]}] * 10
    stream = io.BytesIO()
    enc = Encoder(stream, limits)
    for value in values:
        enc.write(value)
    assert len(enc.strings) + len(enc.containers) <= 2
    assert enc.intern_bytes <= 12
    stream.seek(0)
    dec = Decoder(stream, limits)
    assert [dec.read() for _ in values] == values


@pytest.mark.parametrize(
    "data",
    [
        MAGIC + b"R?",
        MAGIC + b"Rr\x00",
        MAGIC + b"Rq\x00",
        MAGIC + b"RC\x02q\x00",
        MAGIC + b"RS\x05abc",
        MAGIC + b"R",
        MAGIC + b"Ri\x80",
        MAGIC + b"Rg" + struct.pack("<f", float("inf")),
        MAGIC + b"RD\x01" + struct.pack("<d", float("nan")),
        MAGIC + b"Ro\x02s\x01anis\x01an",
        MAGIC + b"RC\x02n",
        MAGIC + b"RC\x03o\x00",
        MAGIC + b"Rs\x01\xff",
        MAGIC + b"Rna",
    ],
)
def test_bad_data(data):
    with pytest.raises(ValueError):
        loads(data)


def test_duplicate_keys():
    with pytest.raises(ValueError, match="duplicate"):
        loads(MAGIC + b"Ro\x02s\x01ans\x01an")


@pytest.mark.parametrize(
    "value", [float("nan"), [float("inf")], {"a": float("nan")}, {1: "key"}, (1, 2), object()]
)
def test_non_json_rejected(value):
    with pytest.raises(ValueError):
        dumps(value)


@pytest.mark.parametrize(
    ("value", "limits"),
    [
        ("1234", Limits(max_string_bytes=3)),
        ([1, 2], Limits(max_array_items=1)),
        (1000, Limits(max_int_bits=3)),
        ([[[None]]], Limits(max_depth=2)),
        ([1, 2, 3], Limits(max_record_values=3)),
    ],
)
def test_bounds(value, limits):
    with pytest.raises(ValueError):
        dumps(value, limits)
    encoded = dumps(value)
    with pytest.raises(ValueError):
        loads(encoded, limits)


def test_canonical_json_roundtrip_recipe():
    original = json.dumps({"b": -0.0, "a": ["你好", 1.25, 1.0000000000000002]}, indent=2) + "\n"
    value = loads(dumps(json.loads(original)))
    assert json.dumps(value, indent=2) + "\n" == original


def test_intern_reference_cannot_bypass_value_or_depth_bound():
    stream = io.BytesIO()
    enc = Encoder(stream)
    value = {"k": [1, 2, 3]}
    enc.write(value)
    enc.write([value, value])
    dec = Decoder(io.BytesIO(stream.getvalue()), Limits(max_record_values=8))
    assert dec.read() == value
    with pytest.raises(ValueError, match="bound"):
        dec.read()
    shallow = io.BytesIO()
    enc = Encoder(shallow)
    enc.write(value)
    enc.write([[value]])
    dec = Decoder(io.BytesIO(shallow.getvalue()), Limits(max_depth=3))
    assert dec.read() == value
    with pytest.raises(ValueError, match="bound"):
        dec.read()


def test_numeric_list_depth_bound():
    with pytest.raises(ValueError, match="depth"):
        dumps([1], Limits(max_depth=0))
    with pytest.raises(ValueError, match="depth"):
        loads(dumps([1]), Limits(max_depth=0))


def test_definition_wrapper_is_not_counted_as_value():
    assert loads(dumps({"a": 1}, Limits(max_record_values=3)), Limits(max_record_values=3)) == {
        "a": 1
    }


def test_existing_container_reference_survives_full_table():
    stream = io.BytesIO()
    enc = Encoder(stream, Limits(max_intern_entries=1))
    value = {"k": "x" * 1000}
    enc.write(value)
    before = stream.tell()
    enc.write(value)
    assert stream.tell() - before == 3
    dec = Decoder(io.BytesIO(stream.getvalue()), Limits(max_intern_entries=1))
    assert dec.read() == dec.read() == value


def test_float_array_chunk_boundary():
    values = [1.5, -0.0] * 32769
    assert signature(loads(dumps(values))) == signature(values)
