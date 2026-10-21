"""Bounded, reversible JSON-value binary stream; no pickle or executable payloads.

This preserves ordered JSON values and numeric types, not whitespace, gzip bytes or
original numeric spellings. Wrap the binary stream in LZMAFile for compression.
"""

from __future__ import annotations

import copy
import io
import json
import math
import struct
from dataclasses import dataclass
from typing import BinaryIO

MAGIC = b"SRFJSON1\x00"


@dataclass(frozen=True)
class Limits:
    max_depth: int = 100
    max_string_bytes: int = 16_000_000
    max_array_items: int = 10_000_000
    max_record_values: int = 20_000_000
    max_int_bits: int = 4096
    max_intern_entries: int = 8192
    max_intern_bytes: int = 64_000_000
    max_container_bytes: int = 128_000


def _uint(value: int) -> bytes:
    out = bytearray()
    while value >= 128:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def _float32(value: float) -> bool:
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0] == value
    except OverflowError:
        return False


def _reference_count(value: object, depth: int, limits: Limits, remaining: int) -> int:
    """Bound the expanded value before copying an interned container."""
    count = 0
    stack = [(value, depth)]
    while stack:
        item, level = stack.pop()
        count += 1
        if count > remaining or level > limits.max_depth:
            raise ValueError("expanded reference exceeds record or depth bound")
        if type(item) is list:
            stack.extend((child, level + 1) for child in item)
        elif type(item) is dict:
            stack.extend((child, level + 1) for child in item.values())
            stack.extend((key, level + 1) for key in item)
    return count


def _reference_shape(value: object) -> tuple[int, int]:
    """Count expanded JSON values and maximum relative depth once per definition."""
    count = maximum_depth = 0
    stack = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        count += 1
        maximum_depth = max(maximum_depth, depth)
        if type(item) is list:
            stack.extend((child, depth + 1) for child in item)
        elif type(item) is dict:
            stack.extend((child, depth + 1) for child in item.values())
            stack.extend((key, depth + 1) for key in item)
    return count, maximum_depth


class Encoder:
    """Write independent records while retaining bounded intern dictionaries."""

    def __init__(self, stream: BinaryIO, limits: Limits | None = None):
        self.stream = stream
        self.limits = limits or Limits()
        self.strings: dict[bytes, int] = {}
        self.containers: dict[bytes, int] = {}
        self.intern_bytes = 0
        self.values = 0
        self.stream.write(MAGIC)

    def _room(self, size: int) -> bool:
        return (
            len(self.strings) + len(self.containers) < self.limits.max_intern_entries
            and self.intern_bytes + size <= self.limits.max_intern_bytes
        )

    def write(self, value: object) -> None:
        self.values = 0
        self.stream.write(b"R")
        self._write(value, 0)

    def _write(self, value: object, depth: int, intern: bool = True) -> None:
        if depth > self.limits.max_depth:
            raise ValueError("maximum nesting depth exceeded")
        self.values += 1
        if self.values > self.limits.max_record_values:
            raise ValueError("maximum record values exceeded")
        out = self.stream
        if value is None:
            out.write(b"n")
        elif type(value) is bool:
            out.write(b"t" if value else b"f")
        elif type(value) is int:
            if value.bit_length() > self.limits.max_int_bits:
                raise ValueError("integer too large")
            out.write(b"i" + _uint(value * 2 if value >= 0 else -value * 2 - 1))
        elif type(value) is float:
            if not math.isfinite(value):
                raise ValueError("nonfinite number is not JSON")
            if _float32(value):
                out.write(b"g" + struct.pack("<f", value))
            else:
                out.write(b"d" + struct.pack("<d", value))
        elif type(value) is str:
            # surrogatepass preserves legal Python json.loads outputs such as "\\ud800".
            raw = value.encode("utf-8", "surrogatepass")
            if len(raw) > self.limits.max_string_bytes:
                raise ValueError("string too large")
            if raw in self.strings:
                out.write(b"r" + _uint(self.strings[raw]))
            else:
                define = self._room(len(raw))
                out.write((b"S" if define else b"s") + _uint(len(raw)) + raw)
                if define:
                    self.strings[raw] = len(self.strings)
                    self.intern_bytes += len(raw)
        elif type(value) is dict:
            if len(value) > self.limits.max_array_items:
                raise ValueError("object too large")
            if any(type(key) is not str for key in value):
                raise ValueError("JSON object keys must be strings")
            # C JSON serializer is efficient for repeated metadata. Do not serialize
            # record dictionaries containing large arrays merely to find a cache key.
            eligible = (
                intern
                and len(value) <= 64
                and not any(type(v) is list and len(v) > 64 for v in value.values())
            )
            raw = None
            if eligible:
                try:
                    candidate = json.dumps(
                        value, ensure_ascii=True, allow_nan=False, separators=(",", ":")
                    ).encode("ascii")
                except (ValueError, TypeError, RecursionError) as exc:
                    raise ValueError("invalid JSON container") from exc
                if len(candidate) <= self.limits.max_container_bytes:
                    raw = candidate
            if raw is not None and raw in self.containers:
                self.values += (
                    _reference_count(
                        value, depth, self.limits, self.limits.max_record_values - self.values + 1
                    )
                    - 1
                )
                out.write(b"q" + _uint(self.containers[raw]))
                return
            define = raw is not None and self._room(len(raw))
            if define:
                # Reserve ID before children, as nested definitions may follow.
                self.containers[raw] = len(self.containers)
                self.intern_bytes += len(raw)
                out.write(b"C" + _uint(len(raw)))
            out.write(b"o" + _uint(len(value)))
            for key, child in value.items():
                self._write(key, depth + 1)
                self._write(child, depth + 1)
        elif type(value) is list:
            count = len(value)
            if count and depth >= self.limits.max_depth:
                raise ValueError("maximum nesting depth exceeded")
            if count > self.limits.max_array_items:
                raise ValueError("array too large")
            if count and all(type(v) is int for v in value):
                out.write(b"I" + _uint(count))
                for item in value:
                    if item.bit_length() > self.limits.max_int_bits:
                        raise ValueError("integer too large")
                    out.write(_uint(item * 2 if item >= 0 else -item * 2 - 1))
                self.values += count
            elif count and all(type(v) is float for v in value):
                if not all(math.isfinite(v) for v in value):
                    raise ValueError("nonfinite number is not JSON")
                narrow = all(_float32(v) for v in value)
                out.write((b"F" if narrow else b"D") + _uint(count))
                kind = "f" if narrow else "d"
                for offset in range(0, count, 65536):
                    block = value[offset : offset + 65536]
                    out.write(struct.pack(f"<{len(block)}{kind}", *block))
                self.values += count
            else:
                out.write(b"a" + _uint(count))
                for child in value:
                    self._write(child, depth + 1)
            if self.values > self.limits.max_record_values:
                raise ValueError("maximum record values exceeded")
        else:
            raise ValueError(f"unsupported JSON type: {type(value).__name__}")


class Decoder:
    """Decode records; raises EOFError only at a clean record boundary.

    ``copy_references=False`` is only for trusted read-only consumers: decoded
    interned containers alias across records and MUST NOT be mutated. The default
    returns independent containers. Cached shape bounds remain enforced in both modes.
    """

    def __init__(
        self, stream: BinaryIO, limits: Limits | None = None, *, copy_references: bool = True
    ):
        self.stream = stream
        self.limits = limits or Limits()
        self.strings: list[str] = []
        self.containers: list[object] = []
        self.container_shapes: list[tuple[int, int] | None] = []
        self.copy_references = copy_references
        self.intern_bytes = 0
        self.values = 0
        if self._read(len(MAGIC)) != MAGIC:
            raise ValueError("invalid stream magic")

    def _read(self, count: int) -> bytes:
        data = self.stream.read(count)
        if len(data) != count:
            raise ValueError("truncated binary JSON stream")
        return data

    def _uint(self, bits: int = 64) -> int:
        result = 0
        for shift in range(0, bits + 7, 7):
            byte = self._read(1)[0]
            result |= (byte & 127) << shift
            if not byte & 128:
                if result.bit_length() > bits:
                    raise ValueError("integer bound exceeded")
                return result
        raise ValueError("unterminated variable integer")

    def _define(self, size: int) -> None:
        if (
            len(self.strings) + len(self.containers) >= self.limits.max_intern_entries
            or self.intern_bytes + size > self.limits.max_intern_bytes
        ):
            raise ValueError("intern dictionary bound exceeded")
        self.intern_bytes += size

    def read(self) -> object:
        marker = self.stream.read(1)
        if not marker:
            raise EOFError
        if marker != b"R":
            raise ValueError("invalid record marker")
        self.values = 0
        return self._value(0)

    def _value(self, depth: int, object_only: bool = False) -> object:
        if depth > self.limits.max_depth:
            raise ValueError("maximum nesting depth exceeded")
        self.values += 1
        if self.values > self.limits.max_record_values:
            raise ValueError("maximum record values exceeded")
        tag = self._read(1)
        if object_only and tag != b"o":
            raise ValueError("intern definition must directly contain object")
        if tag == b"n":
            return None
        if tag in (b"t", b"f"):
            return tag == b"t"
        if tag == b"i":
            number = self._uint(self.limits.max_int_bits + 1)
            value = number // 2 if number % 2 == 0 else -(number // 2) - 1
            if value.bit_length() > self.limits.max_int_bits:
                raise ValueError("integer too large")
            return value
        if tag in (b"g", b"d"):
            value = struct.unpack(
                "<f" if tag == b"g" else "<d", self._read(4 if tag == b"g" else 8)
            )[0]
            if not math.isfinite(value):
                raise ValueError("nonfinite number")
            return value
        if tag in (b"s", b"S"):
            size = self._uint()
            if size > self.limits.max_string_bytes:
                raise ValueError("string too large")
            try:
                value = self._read(size).decode("utf-8", "surrogatepass")
            except UnicodeDecodeError as exc:
                raise ValueError("invalid UTF-8 string") from exc
            if tag == b"S":
                self._define(size)
                self.strings.append(value)
            return value
        if tag in (b"r", b"q"):
            index = self._uint()
            table = self.strings if tag == b"r" else self.containers
            if index >= len(table) or table[index] is _PENDING:
                raise ValueError("invalid intern reference")
            if tag == b"r":
                return table[index]
            count, relative_depth = self.container_shapes[index]
            if (
                self.values + count - 1 > self.limits.max_record_values
                or depth + relative_depth > self.limits.max_depth
            ):
                raise ValueError("expanded reference exceeds record or depth bound")
            self.values += count - 1
            return copy.deepcopy(table[index]) if self.copy_references else table[index]
        if tag == b"C":
            size = self._uint()
            if size > self.limits.max_container_bytes:
                raise ValueError("container intern size exceeded")
            self._define(size)
            index = len(self.containers)
            self.containers.append(_PENDING)
            self.container_shapes.append(None)
            self.values -= 1  # The definition wrapper is not a JSON value.
            value = self._value(depth, object_only=True)
            if type(value) is not dict:
                raise ValueError("interned container must be object")
            canonical = json.dumps(
                value, ensure_ascii=True, allow_nan=False, separators=(",", ":")
            ).encode("ascii")
            if len(canonical) != size:
                raise ValueError("container intern size mismatch")
            self.container_shapes[index] = _reference_shape(value)
            self.containers[index] = copy.deepcopy(value) if self.copy_references else value
            return value
        if tag in (b"a", b"o", b"I", b"F", b"D"):
            count = self._uint()
            if count and depth >= self.limits.max_depth:
                raise ValueError("maximum nesting depth exceeded")
            if count > self.limits.max_array_items:
                raise ValueError("container item bound exceeded")
            if tag == b"o":
                result = {}
                for _ in range(count):
                    key = self._value(depth + 1)
                    if type(key) is not str or key in result:
                        raise ValueError("invalid or duplicate object key")
                    result[key] = self._value(depth + 1)
                return result
            if tag == b"a":
                return [self._value(depth + 1) for _ in range(count)]
            self.values += count
            if self.values > self.limits.max_record_values:
                raise ValueError("maximum record values exceeded")
            if tag == b"I":
                values = []
                for _ in range(count):
                    number = self._uint(self.limits.max_int_bits + 1)
                    value = number // 2 if number % 2 == 0 else -(number // 2) - 1
                    if value.bit_length() > self.limits.max_int_bits:
                        raise ValueError("integer too large")
                    values.append(value)
                return values
            kind, width = ("f", 4) if tag == b"F" else ("d", 8)
            values = []
            for offset in range(0, count, 65536):
                size = min(65536, count - offset)
                values.extend(struct.unpack(f"<{size}{kind}", self._read(width * size)))
            if not all(math.isfinite(v) for v in values):
                raise ValueError("nonfinite number")
            return values
        raise ValueError(f"unknown binary JSON tag: {tag!r}")


_PENDING = object()


def dumps(value: object, limits: Limits | None = None) -> bytes:
    stream = io.BytesIO()
    Encoder(stream, limits).write(value)
    return stream.getvalue()


def loads(data: bytes, limits: Limits | None = None) -> object:
    stream = io.BytesIO(data)
    value = Decoder(stream, limits).read()
    if stream.read(1):
        raise ValueError("trailing data after single record")
    return value
