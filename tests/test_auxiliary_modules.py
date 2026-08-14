"""Behavioral coverage for public evaluation, intervention, and JSONL helpers."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from types import MappingProxyType

import pytest

from compbias.envs.cva_world.schema import ErrorSpec
from compbias.eval.equilibrium import summarize_endpoint_labels, summarize_equilibria
from compbias.interventions.counterfactual import pair_error_mechanism_shift
from compbias.interventions.error_catalog import (
    apply_catalog_error,
    index_error_catalog,
    validate_error_catalog,
)
from compbias.io.jsonl import JsonlDecodeError, append_jsonl, read_jsonl, write_jsonl


def _truth_error() -> ErrorSpec:
    return ErrorSpec(error_id="truth", family="truth", severity=0, parameters={})


def _offset_error() -> ErrorSpec:
    return ErrorSpec(
        error_id="numeric_offset:+2",
        family="numeric_offset",
        severity=2,
        parameters={"field": "value", "delta": 2},
    )


def test_equilibrium_summary_is_sorted_detached_and_deeply_read_only() -> None:
    endpoints = [
        {"endpoint_label": "truthful"},
        {"endpoint_label": "compensatory"},
        {"endpoint_label": "truthful"},
    ]

    summary = summarize_endpoint_labels(endpoints)
    endpoints[0]["endpoint_label"] = "changed-after-call"

    assert summary.total == 3
    assert summary.counts == {"compensatory": 1, "truthful": 2}
    assert summary.proportions == pytest.approx({"compensatory": 1 / 3, "truthful": 2 / 3})
    assert isinstance(summary.counts, MappingProxyType)
    with pytest.raises(TypeError):
        summary.counts["new"] = 1  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        summary.total = 0  # type: ignore[misc]

    exported = summary.to_mapping()
    exported_counts = exported["counts"]
    assert isinstance(exported_counts, dict)
    exported_counts["truthful"] = 0
    assert summary.counts["truthful"] == 2


def test_equilibrium_alias_and_custom_label_field_accept_string_endpoints() -> None:
    assert summarize_equilibria(("left", "right", "left")).to_mapping() == {
        "total": 3,
        "counts": {"left": 2, "right": 1},
        "proportions": {"left": 2 / 3, "right": 1 / 3},
    }
    assert summarize_endpoint_labels(({"terminal": "stable"},), label_field="terminal").counts == {
        "stable": 1
    }


@pytest.mark.parametrize("label_field", ["", None, 0])
def test_equilibrium_rejects_invalid_label_field(label_field: object) -> None:
    with pytest.raises(ValueError, match="label_field must be a non-empty string"):
        summarize_endpoint_labels(("stable",), label_field=label_field)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("endpoints", "message"),
    [
        ((), "endpoints must not be empty"),
        (({"endpoint_label": ""},), "every terminal endpoint_label"),
        (({"endpoint_label": "   "},), "every terminal endpoint_label"),
        (({"other": "stable"},), "every terminal endpoint_label"),
        ((1,), "every terminal endpoint_label"),
    ],
)
def test_equilibrium_rejects_missing_or_non_string_terminal_labels(
    endpoints: tuple[object, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        summarize_endpoint_labels(endpoints)


@dataclass
class _DataclassRecord:
    sample_id: str
    error_mechanism: str
    payload: dict[str, object]


class _SerializableRecord:
    def __init__(self, payload: dict[object, object]) -> None:
        self.payload = payload

    def to_mapping(self) -> dict[object, object]:
        return self.payload


class _InvalidSerializableRecord:
    def to_mapping(self) -> list[object]:
        return []


def test_counterfactual_pairing_supports_all_record_forms_and_detaches_nested_data() -> None:
    source_payload = {"values": [1, {"nested": "source"}]}
    counterfactual_payload = {"values": (1, {"nested": "source"})}
    source = [
        {
            "sample_id": "b",
            "split_keys": {"error_mechanism": "standard", "visual_style": "font_a"},
            "payload": source_payload,
            "error_catalog": [{"error_id": "truth"}],
        },
        _DataclassRecord("a", "standard", {"values": [3]}),
    ]
    counterfactual = [
        _SerializableRecord(
            {
                "sample_id": "b",
                "split_keys": {"error_mechanism": "shifted", "visual_style": "font_a"},
                "payload": counterfactual_payload,
                "error_catalog": [{"error_id": "numeric_offset:+2"}],
            }
        ),
        {"sample_id": "a", "error_mechanism": "shifted", "payload": {"values": [3]}},
    ]

    pairs = pair_error_mechanism_shift(source, counterfactual)
    source_payload["values"][1]["nested"] = "mutated"  # type: ignore[index]
    counterfactual_payload["values"][1]["nested"] = "mutated"  # type: ignore[index]

    assert tuple(pair.sample_id for pair in pairs) == ("a", "b")
    assert pairs[1].source_error_mechanism == "standard"
    assert pairs[1].counterfactual_error_mechanism == "shifted"
    assert pairs[1].shifted_factors == ("error_mechanism",)
    assert pairs[1].source["payload"]["values"][1]["nested"] == "source"  # type: ignore[index]
    assert pairs[1].counterfactual["payload"]["values"][1]["nested"] == (  # type: ignore[index]
        "source"
    )
    with pytest.raises(TypeError):
        pairs[1].source["sample_id"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        pairs[1].source["payload"]["values"][1]["nested"] = "changed"  # type: ignore[index]

    exported = pairs[1].to_mapping()
    assert exported["shifted_factors"] == ["error_mechanism"]
    exported["source"]["payload"]["values"][1]["nested"] = "changed"  # type: ignore[index]
    assert pairs[1].source["payload"]["values"][1]["nested"] == "source"  # type: ignore[index]


@pytest.mark.parametrize(
    ("changed_field", "counterfactual_value"),
    [
        ("scene", {"value": 8}),
        (
            "split_keys",
            {"error_mechanism": "shifted", "visual_style": "font_b"},
        ),
    ],
)
def test_counterfactual_pairing_rejects_changes_outside_mechanism_and_catalog(
    changed_field: str, counterfactual_value: object
) -> None:
    source = {
        "sample_id": "a",
        "scene": {"value": 7},
        "split_keys": {"error_mechanism": "standard", "visual_style": "font_a"},
        "error_catalog": [{"error_id": "truth"}],
    }
    counterfactual = {
        "sample_id": "a",
        "scene": {"value": 7},
        "split_keys": {"error_mechanism": "shifted", "visual_style": "font_a"},
        "error_catalog": [{"error_id": "numeric_offset:+2"}],
    }
    counterfactual[changed_field] = counterfactual_value

    with pytest.raises(ValueError, match="changes fields other than"):
        pair_error_mechanism_shift((source,), (counterfactual,))


@pytest.mark.parametrize(
    ("source", "counterfactual", "exception", "message"),
    [
        ((), ({"sample_id": "a", "error_mechanism": "shifted"},), ValueError, "source"),
        (({"sample_id": "a", "error_mechanism": "base"},), (), ValueError, "counterfactual"),
        (
            ({"error_mechanism": "base"},),
            ({"sample_id": "a", "error_mechanism": "shifted"},),
            ValueError,
            "sample_id",
        ),
        (
            (
                {"sample_id": "a", "error_mechanism": "base"},
                {"sample_id": "a", "error_mechanism": "base"},
            ),
            ({"sample_id": "a", "error_mechanism": "shifted"},),
            ValueError,
            "duplicate sample_id",
        ),
        (
            ({"sample_id": "a", "error_mechanism": "base"},),
            ({"sample_id": "b", "error_mechanism": "shifted"},),
            ValueError,
            "paired sample_id values",
        ),
        (
            ({"sample_id": "a", "error_mechanism": "base"},),
            ({"sample_id": "a", "error_mechanism": "base"},),
            ValueError,
            "does not change",
        ),
        (
            ({"sample_id": "a"},),
            ({"sample_id": "a", "error_mechanism": "shifted"},),
            ValueError,
            "identify its error_mechanism",
        ),
        (
            (_InvalidSerializableRecord(),),
            ({"sample_id": "a", "error_mechanism": "shifted"},),
            TypeError,
            "mappings or dataclasses",
        ),
    ],
)
def test_counterfactual_pairing_rejects_unpaired_or_ambiguous_records(
    source: tuple[object, ...],
    counterfactual: tuple[object, ...],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        pair_error_mechanism_shift(source, counterfactual)


def test_error_catalog_validation_indexing_and_application_are_immutable() -> None:
    truth = _truth_error()
    offset = _offset_error()
    catalog = [truth, offset]

    validated = validate_error_catalog(catalog)
    catalog.reverse()
    index = index_error_catalog(validated)
    scene = {"value": 7, "metadata": {"source": "fixture"}}
    changed = apply_catalog_error(scene, validated, "numeric_offset:+2")

    assert validated == (truth, offset)
    assert tuple(index) == ("truth", "numeric_offset:+2")
    assert changed["value"] == 9
    assert scene == {"value": 7, "metadata": {"source": "fixture"}}
    with pytest.raises(TypeError):
        index["new"] = truth  # type: ignore[index]
    with pytest.raises(TypeError):
        changed["value"] = 11  # type: ignore[index]


@pytest.mark.parametrize(
    ("catalog", "exception", "message"),
    [
        ((), ValueError, "must not be empty"),
        ((_truth_error(), _truth_error()), ValueError, "duplicate error_id"),
        ((_offset_error(),), ValueError, "truth intervention"),
        ((object(),), TypeError, "ErrorSpec"),
    ],
)
def test_error_catalog_rejects_incomplete_or_invalid_catalogs(
    catalog: tuple[object, ...], exception: type[Exception], message: str
) -> None:
    with pytest.raises(exception, match=message):
        validate_error_catalog(catalog)


@pytest.mark.parametrize("error_id", ["", None, 0])
def test_apply_catalog_error_rejects_invalid_error_identifiers(error_id: object) -> None:
    with pytest.raises(ValueError, match="error_id must be a non-empty string"):
        apply_catalog_error({"value": 7}, (_truth_error(),), error_id)  # type: ignore[arg-type]


def test_apply_catalog_error_reports_unknown_catalog_member() -> None:
    with pytest.raises(KeyError, match="unknown error_id: missing"):
        apply_catalog_error({"value": 7}, (_truth_error(),), "missing")


def test_jsonl_round_trip_is_canonical_and_append_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "records.jsonl"
    records = ({"z": 2, "a": "café"}, {"sample_id": "b", "values": [1, 2]})

    returned = write_jsonl(path, records)
    append_jsonl(path, ({"sample_id": "c"},))

    assert returned == path
    assert path.read_text(encoding="utf-8").splitlines() == [
        '{"a":"café","z":2}',
        '{"sample_id":"b","values":[1,2]}',
        '{"sample_id":"c"}',
    ]
    assert read_jsonl(path) == (*records, {"sample_id": "c"})
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_jsonl_append_creates_a_missing_file_and_empty_write_is_readable(tmp_path: Path) -> None:
    appended = tmp_path / "new.jsonl"
    empty = tmp_path / "empty.jsonl"

    append_jsonl(appended, ({"row": 1},))
    write_jsonl(empty, ())

    assert read_jsonl(appended) == ({"row": 1},)
    assert empty.read_bytes() == b""
    assert read_jsonl(empty) == ()


@pytest.mark.parametrize(
    ("contents", "line_number", "message"),
    [
        ('{"ok":1}\n\n', 2, "blank JSONL row"),
        ('{"ok":1}\n{"bad":}\n', 2, "Expecting value"),
        ('{"ok":1}\n[1,2]\n', 2, "JSONL row must be an object"),
    ],
)
def test_jsonl_reader_reports_path_line_and_failure_kind(
    tmp_path: Path, contents: str, line_number: int, message: str
) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(JsonlDecodeError, match=message) as captured:
        read_jsonl(path)

    assert captured.value.path == path
    assert captured.value.line_number == line_number
    assert str(captured.value).startswith(f"{path}:{line_number}:")


def test_jsonl_batch_validation_preserves_existing_destination(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    write_jsonl(path, ({"stable": True},))

    with pytest.raises(TypeError, match="JSONL row 2 must be a mapping"):
        write_jsonl(path, ({"replacement": 1}, object()))  # type: ignore[arg-type]

    assert read_jsonl(path) == ({"stable": True},)
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_jsonl_failed_replace_removes_temporary_file(tmp_path: Path) -> None:
    directory_destination = tmp_path / "destination"
    directory_destination.mkdir()

    with pytest.raises(OSError):
        write_jsonl(directory_destination, ({"row": 1},))

    assert directory_destination.is_dir()
    assert list(tmp_path.glob(f".{directory_destination.name}.*.tmp")) == []


@pytest.mark.parametrize("path", ["", "/"])
def test_jsonl_requires_a_file_name(path: str) -> None:
    with pytest.raises(ValueError, match="JSONL path must name a file"):
        write_jsonl(path, ())
