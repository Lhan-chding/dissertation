"""Lossless document assembly, source identity, and completion boundaries."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def api():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_final_fact_record.py"
    spec = importlib.util.spec_from_file_location("fact_record_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def case(tmp_path):
    work = tmp_path / "work"
    source = work / "server_sources"
    stage = source / "example"
    stage.mkdir(parents=True)
    shared = {f"long_observation_field_name_{index}": index for index in range(40)}
    payload = json.dumps({"first": shared, "second": shared, "number": "LEXEME"}).replace(
        '"LEXEME"', "0.12345678901234567890123456789"
    )
    (stage / "values.json").write_text(payload)
    with (stage / "table.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "value", "note"])
        writer.writerow(["a", "0.00000000000000000001", " padded | <br>\n多行\t "])
        writer.writerow(["b", "9007199254740993", "guard-cell"])
    (work / "document_spec.json").write_text(json.dumps({"stages": ["example"]}))
    files = []
    for path in sorted(stage.iterdir()):
        content = path.read_bytes()
        files.append(
            {
                "stage": "example",
                "relative_path": str(path.relative_to(source)),
                "server_path": "/recorded/" + path.name,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    (source / "source_manifest.json").write_text(json.dumps({"files": files}))
    return work, source, work / "result.md"


def test_written_document_restores_numbers_tables_and_shared_objects(api, case):
    work, source, output = case
    api.build(work, source, output)
    result = output.read_text()
    assert "0.12345678901234567890123456789" in result
    assert "0.00000000000000000001" in result
    assert "9007199254740993" in result
    coverage = json.loads((work / "document_coverage.json").read_text())
    assert coverage["status"] == "DRAFT"
    assert coverage["shared_json_subtrees"] > 0
    assert coverage["shared_field_schemas"] > 0
    assert coverage["rendered_verification"]["status"] == "PASS"
    assert coverage["rendered_verification"]["data_rows_restored"] == 2


def test_source_hash_mismatch_stops_before_publication(api, case):
    work, source, output = case
    path = source / "example" / "values.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="Transfer identity mismatch"):
        api.build(work, source, output)
    assert not output.exists()


def test_file_aliases_follow_rendered_stage_order(api, case):
    work, source, output = case
    other = source / "second"
    other.mkdir()
    (other / "values.json").write_bytes((source / "example" / "values.json").read_bytes())
    (work / "document_spec.json").write_text(json.dumps({"stages": ["second", "example"]}))
    api.build(work, source, output)
    coverage = json.loads((work / "document_coverage.json").read_text())
    assert coverage["rendered_verification"]["status"] == "PASS"
    assert any(item.get("same_content_as") for item in coverage["file_coverage"])


def test_alias_cannot_hide_json_field_not_in_csv(api, case):
    work, source, output = case
    stage = source / "example"
    (source / "source_manifest.json").unlink()
    (stage / "grouped.json").write_text('{"a":[{"value":1,"hidden":2}]}')
    (stage / "grouped.csv").write_text("group,value\na,1\n")
    (work / "document_spec.json").write_text(
        json.dumps(
            {
                "stages": ["example"],
                "json_csv_equivalences": [
                    {
                        "json": "example/grouped.json",
                        "csv": "example/grouped.csv",
                        "group_key": "group",
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="fields differ"):
        api.build(work, source, output)


def test_csv_alias_restores_original_json_scalar_types_and_empty_groups(api, case):
    work, source, output = case
    stage = source / "example"
    (stage / "grouped.json").write_text(
        '{"empty":[],"a":[{"value":1.000,"defined":true,"note":"1.000"}]}'
    )
    (stage / "grouped.csv").write_text("group,value,defined,note\na,1.000,True,1.000\n")
    spec = {
        "stages": ["example"],
        "json_csv_equivalences": [
            {"json": "example/grouped.json", "csv": "example/grouped.csv", "group_key": "group"}
        ],
    }
    (work / "document_spec.json").write_text(json.dumps(spec))
    api.build(work, source, output)
    coverage = json.loads((work / "document_coverage.json").read_text())
    assert coverage["rendered_verification"]["csv_alias_json_files_restored"] == 1
    output.write_text(output.read_text().replace('"defined":"boolean"', '"defined":"string"'))
    with pytest.raises(AssertionError, match="Written CSV alias differs"):
        api.verify_rendered_document(output, api.collect_files(source, work, spec))


@pytest.mark.parametrize(
    "replacement", [{"status": "PASS"}, {"status": "PASS", "all_stage_sources_verified": 1}]
)
def test_finalization_requires_explicit_typed_acceptance(api, case, replacement):
    work, source, output = case
    receipt = work / "accepted.json"
    receipt.write_text(json.dumps(replacement))
    with pytest.raises(ValueError, match="acceptance receipt"):
        api.build(work, source, output, receipt)
    assert not output.exists()


@pytest.mark.parametrize("failure", [None, "hash", "omitted", "status", "role", "unconfigured"])
def test_finalization_binds_included_actual_acceptance_sources(api, case, failure):
    work, source, output = case
    audit = work / "audit.json"
    audit.write_text(json.dumps({"result": {"status": "FAIL" if failure == "status" else "PASS"}}))
    spec = {
        "stages": ["example"],
        "receipts": [] if failure == "omitted" else ["audit.json"],
        "completion_sources": {
            "audit": {
                "path": "audit.json",
                "checks": [{"field": ["result", "status"], "equals": "PASS"}],
            }
        },
    }
    if failure == "unconfigured":
        del spec["completion_sources"]
    (work / "document_spec.json").write_text(json.dumps(spec))
    binding = {
        "path": str(audit),
        "bytes": audit.stat().st_size,
        "sha256": "wrong" if failure == "hash" else hashlib.sha256(audit.read_bytes()).hexdigest(),
    }
    receipt = work / "complete.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "PASS",
                "all_stage_sources_verified": True,
                "verified_sources": {} if failure == "role" else {"audit": binding},
            }
        )
    )
    if failure:
        with pytest.raises(ValueError, match=r"Acceptance source|acceptance source bindings"):
            api.build(work, source, output, receipt)
        assert not output.exists()
    else:
        api.build(work, source, output, receipt)
        coverage = json.loads((work / "document_coverage.json").read_text())
        assert coverage["status"] == "COMPLETE"


def test_written_markdown_corruption_is_detected_independently(api, case):
    work, source, output = case
    api.build(work, source, output)
    output.write_text(output.read_text().replace("guard-cell", "changed-cell"))
    spec = json.loads((work / "document_spec.json").read_text())
    records = api.collect_files(source, work, spec)
    with pytest.raises(AssertionError, match="Written table value differs"):
        api.verify_rendered_document(output, records)


def test_render_verification_failure_preserves_existing_delivery(api, case, monkeypatch):
    work, source, output = case
    output.write_text("previous verified document")

    def reject(*_args):
        raise AssertionError("rendered verification failed")

    monkeypatch.setattr(api, "verify_rendered_document", reject)
    with pytest.raises(AssertionError, match="rendered verification failed"):
        api.build(work, source, output)
    assert output.read_text() == "previous verified document"
    assert not list(work.glob("result.md.*.tmp"))


def test_parquet_embedded_json_remains_reconstructible(api, case):
    arrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    work, source, output = case
    values = {f"observation_field_{index}": index / 10 for index in range(80)}
    data = arrow.Table.from_pylist(
        [
            {"candidate": "x", "audit_json": json.dumps(values)},
            {"candidate": "y", "audit_json": json.dumps(values)},
        ]
    )
    parquet.write_table(data, source / "example" / "summary.parquet")
    api.build(work, source, output)
    coverage = json.loads((work / "document_coverage.json").read_text())
    assert coverage["rendered_verification"]["tables_restored"] == 2
    assert coverage["rendered_verification"]["data_rows_restored"] == 4
