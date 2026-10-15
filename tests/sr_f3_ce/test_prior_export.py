import json
from pathlib import Path

import pytest

from sr_f3_ce.preflight import file_hash, object_hash
from sr_f3_ce.prior_export import (
    RAW_FIELDS,
    MissingEvidence,
    PriorExporter,
    _image_exports,
    write_jsonl_once,
)


def source_file(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


@pytest.fixture
def exporter(tmp_path):
    parent, output = tmp_path / "old", tmp_path / "new"
    parent.mkdir()
    output.mkdir()
    return PriorExporter(parent, output, {"tasks": []}, {"bound_files": {}})


def raw_group(exporter):
    key = ("SRF2_J6_s71001", 96, "CHARTQA", "plain_answer", None, "chartqa-0000")
    row = dict.fromkeys(RAW_FIELDS, None)
    row.update(zip(("model_id", "step", "pool", "protocol", "view", "qid"), key, strict=True))
    row.update(
        draw=0,
        experiment_id="SR-F2-20261013",
        raw_text="50%",
        text="Actual prompt",
        generation={"max_new_tokens": 128},
        sampling_parameters={"do_sample": False},
        tokens=[1],
        sampler_logprobs=[-0.2],
        prompt_tensor_hash="0" * 64,
        input_hash="1" * 64,
        image_hash="2" * 64,
        completion_token_count=1,
        finish_reason="length",
        truncated=True,
    )
    group_id = object_hash(key)
    relative = f"sealed/evaluation/{key[0]}/groups/{group_id}.json"
    value = dict(group_id=group_id, records=[row], records_sha256=object_hash([row]))
    path = source_file(exporter.root, relative, value)
    refs = {
        group_id: dict(
            group_id=group_id, raw_sha256=file_hash(path), records_sha256=value["records_sha256"]
        )
    }
    return relative, key, refs, value


def test_raw_export_preserves_bytes_prompt_stop_and_source_pointer(exporter):
    relative, key, refs, _ = raw_group(exporter)
    source = (exporter.root / relative).read_bytes()
    count = exporter.group(
        str(Path(relative).parents[1]),
        refs,
        key,
        1,
        gold_location={"source_path": "manifest.jsonl", "source_line": 9},
    )
    assert count == 1
    assert (exporter.output / "prior/raw" / relative).read_bytes() == source
    assert (exporter.root / relative).read_bytes() == source
    record = exporter.records[0]
    assert record["actual_prompt_text"] == "Actual prompt"
    assert record["finish_reason"] == "length" and record["truncated"] is True
    assert record["source_line"] == 1 and record["source_json_pointer"] == "/records/0"
    assert record["gold_source"]["source_line"] == 9


def test_changed_bytes_are_not_certified_or_copied(exporter):
    relative, key, refs, value = raw_group(exporter)
    value["records"][0]["raw_text"] = "altered"
    source_file(exporter.root, relative, value)
    with pytest.raises(MissingEvidence, match="hash mismatch"):
        exporter.group(str(Path(relative).parents[1]), refs, key, 1, gold_location={})
    assert not (exporter.output / "prior/raw" / relative).exists()


def test_missing_actual_prompt_is_retained_and_marked_missing(exporter):
    relative, key, refs, value = raw_group(exporter)
    del value["records"][0]["text"]
    value["records_sha256"] = object_hash(value["records"])
    path = source_file(exporter.root, relative, value)
    refs[object_hash(key)].update(
        raw_sha256=file_hash(path), records_sha256=value["records_sha256"]
    )
    with pytest.raises(MissingEvidence, match="raw fields absent"):
        exporter.group(str(Path(relative).parents[1]), refs, key, 1, gold_location={})
    assert (exporter.output / "prior/raw" / relative).exists()
    assert exporter.records[0]["missing_raw_fields"] == ["text"]


def test_panel_requires_released_worker_receipt(exporter):
    source_file(
        exporter.root, "tasks/endpoint/SRF2_J6_s71001/COMPLETE.json", dict(status="COMPLETE")
    )
    with pytest.raises(MissingEvidence, match="released authorization"):
        exporter.panel("endpoint", "SRF2_J6_s71001")


def test_missing_panel_stays_in_required_item_counts(exporter):
    spec = dict(label="CHARTQA", questions=[{"qid": "q1"}, {"qid": "q2"}], k=1)
    result = exporter.export_model("endpoint", "SRF2_J6_s71001", [spec])
    assert result[0]["status"] == "MISSING"
    assert result[0]["expected_records"] == 2 and result[0]["exported_records"] == 0


def test_export_root_cannot_touch_old_tree(tmp_path):
    old = tmp_path / "old"
    child = old / "child"
    child.mkdir(parents=True)
    with pytest.raises(PermissionError, match="independent"):
        PriorExporter(old, child, {}, {"bound_files": {}})


def test_jsonl_publication_is_idempotent_and_never_replaces(tmp_path):
    first = write_jsonl_once(tmp_path, "records.jsonl", [{"raw": "unfavorable"}])
    assert write_jsonl_once(tmp_path, "records.jsonl", [{"raw": "unfavorable"}]) == first
    with pytest.raises(PermissionError, match="Existing output differs"):
        write_jsonl_once(tmp_path, "records.jsonl", [{"raw": "rewritten"}])
    assert json.loads((tmp_path / "records.jsonl").read_text())["raw"] == "unfavorable"


def test_authenticated_task_completion_binds_raw_group(exporter):
    relative, key, refs, _ = raw_group(exporter)
    directory = str(Path(relative).parents[1])
    done = source_file(
        exporter.root,
        directory + "/COMPLETE.json",
        dict(
            status="COMPLETE",
            model_id=key[0],
            records=1,
            groups=list(refs.values()),
        ),
    )
    task = dict(
        kind="endpoint",
        model_id=key[0],
        status="COMPLETE",
        records=1,
        registration_sha256=object_hash(exporter.registration),
        artifacts=[dict(path=directory + "/COMPLETE.json", sha256=file_hash(done))],
    )
    source_file(exporter.root, f"tasks/endpoint/{key[0]}/COMPLETE.json", task)
    exporter.authorization["tasks"] = [
        dict(kind="endpoint", model_id=key[0], receipt_sha256=object_hash(task))
    ]
    assert exporter.panel("endpoint", key[0]) == (directory, refs)
    task["records"] = 2
    source_file(exporter.root, f"tasks/endpoint/{key[0]}/COMPLETE.json", task)
    with pytest.raises(MissingEvidence, match="released authorization"):
        exporter.panel("endpoint", key[0])


def test_image_bridge_checks_pixels_and_preserves_original_image_path(exporter):
    from PIL import Image

    image_file = "external/chartqa/images/tiny.png"
    path = exporter.root / image_file
    path.parent.mkdir(parents=True)
    Image.new("RGB", (2, 3), (19, 30, 41)).save(path)
    digest = file_hash(path)
    identity = dict(files=[dict(path=image_file, sha256=digest, bytes=path.stat().st_size)])
    source_file(exporter.root, "INPUT_IDENTITY.json", identity)
    rows = {
        f"chartqa-{index:04d}": dict(
            qid=f"chartqa-{index:04d}", image_file=image_file, image_sha256=digest
        )
        for index in range(2500)
    }
    manifest = "external/chartqa/TEST_MANIFEST.jsonl"
    (exporter.root / manifest).write_text("".join(json.dumps(row) + "\n" for row in rows.values()))
    result = _image_exports(exporter, identity, manifest, rows)
    assert result["status"] == "COMPLETE"
    assert result["checked_images"] == result["copied_chartqa_images"] == 1
    copied = exporter.output / "external/legacy_images" / image_file
    assert copied.read_bytes() == path.read_bytes()
    history = json.loads((exporter.output / "prior/HISTORICAL_IMAGES.json").read_text())
    assert history["rows"][0]["pixel_sha256"]
    assert history["model_calls"] == 0 and history["new_test_outputs_accessed"] is False
