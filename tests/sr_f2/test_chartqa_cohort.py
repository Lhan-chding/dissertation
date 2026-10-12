"""Synthetic cohorts only: private user question identifiers never enter public tests."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "chartqa_cohort", Path(__file__).resolve().parents[2] / "scripts/sr_f2/verify_chartqa_cohort.py"
)
cohort = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cohort)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixtures(tmp_path):
    definition = dict(
        baseline="BASE",
        models=["M1", "M2", "M3"],
        count=1,
        qids=["synthetic-0"],
        per_model_lost_counts=dict(M1=1, M2=2, M3=1),
        relaxed_accuracy_percent=dict(BASE=100, M1=75, M2=50, M3=75),
        questions_per_model=4,
        expected_model_count=4,
    )
    defpath = tmp_path / "definition.json"
    save(defpath, definition)
    source = tmp_path / "prior/audit/sidecar.jsonl"
    source.parent.mkdir(parents=True)
    rows, raw_rows = [], {}
    for model in ["BASE", "M1", "M2", "M3"]:
        for i in range(4):
            qid = f"synthetic-{i}"
            correct = model == "BASE" or (i != 0 and not (model == "M2" and i == 1))
            prediction = "yes\n" if correct else "no"
            row = dict(
                label=cohort.LABEL,
                model_id=model,
                qid=qid,
                root_id=f"image-{i}",
                variant="human",
                slot_id=f"{model}|CHARTQA|{qid}",
                frozen=cohort.score_chartqa("yes", prediction),
                whitespace_stripped=cohort.score_chartqa("yes", prediction.strip()),
            )
            rows.append(row)
            raw_rows[(model, qid)] = dict(
                row, raw_text=prediction, tokens=[1, 2], finish_reason="eos"
            )
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return definition, defpath, source, raw_rows


def raw_fixture(tmp_path, raw_rows):
    root = tmp_path / "raw"
    goldpath = root / "external/chartqa/TEST_MANIFEST.jsonl"
    goldpath.parent.mkdir(parents=True)
    goldpath.write_text(
        "".join(
            json.dumps(dict(qid=f"synthetic-{i}", answer="yes", image_id=f"image-{i}")) + "\n"
            for i in range(4)
        )
    )
    save(
        goldpath.parent / "IDENTITY.json",
        dict(status="AVAILABLE_VERIFIED", manifest_sha256=cohort.sha(goldpath)),
    )
    refs = []
    for model in sorted({key[0] for key in raw_rows}):
        relative = cohort.group_relative(model, "synthetic-0", "BASE")
        payload = dict(records=[raw_rows[(model, "synthetic-0")]])
        payload["records_sha256"] = cohort.object_hash(payload["records"])
        save(root / relative, payload)
        directory = root / "sealed/evaluation" / model
        finish = dict(
            groups=[
                dict(
                    group_id=Path(relative).stem, records=1, raw_sha256=cohort.sha(root / relative)
                )
            ]
        )
        save(directory / "chartqa/COMPLETE.json", finish)
        done = dict(
            panels=[
                dict(
                    panel="chartqa",
                    completion_sha256=cohort.sha(directory / "chartqa/COMPLETE.json"),
                )
            ]
        )
        save(directory / "COMPLETE.json", done)
        release = dict(evaluation_sha256=cohort.object_hash(done))
        save(root / "released/evaluation" / f"{model}.json", release)
        refs.append(dict(model_id=model, release_sha256=cohort.object_hash(release)))
    save(
        root / "endpoint_orchestration/COMPLETE.json",
        dict(status="EVALUATIONS_COMPLETE_REPORT_PENDING", releases=refs),
    )
    return root


def test_complete_identity_keeps_posthoc_separate(tmp_path):
    _, definition, source, _ = fixtures(tmp_path)
    result = cohort.run(definition, source, tmp_path / "output")
    assert result["status"] == "PASS" and result["cohort_identity_verified"] is True
    assert result["count"] == 1 and result["loss_counts"] == dict(M1=1, M2=2, M3=1)
    assert result["raw_evidence_status"] == "SOURCE_REFERENCES_ONLY"
    assert result["label"] == cohort.LABEL
    assert result["official_frozen_scores_changed"] is False
    rows = [
        json.loads(line)
        for line in (tmp_path / "output/COHORT_ROWS.jsonl").read_text().splitlines()
    ]
    assert rows[0]["models"]["BASE"]["score_sidecar"]["frozen"]["relaxed_accuracy"] == 0
    assert (
        rows[0]["models"]["BASE"]["score_sidecar"]["whitespace_stripped"]["relaxed_accuracy"] == 1
    )
    assert rows[0]["source_sidecar_sha256"] == cohort.sha(source)
    assert "gold" not in rows[0]


def test_wrong_user_set_fails_and_preserves_failure_receipt(tmp_path):
    data, definition, source, _ = fixtures(tmp_path)
    data["qids"] = ["synthetic-3"]
    save(definition, data)
    with pytest.raises(ValueError, match="exact computed intersection"):
        cohort.run(definition, source, tmp_path / "output")
    receipt = json.loads((tmp_path / "output/COHORT_VERIFICATION.json").read_text())
    assert receipt["status"] == "FAIL" and receipt["cohort_identity_verified"] is False


@pytest.mark.parametrize("fault", ["duplicate", "missing", "count", "accuracy", "selection"])
def test_reject_partial_or_changed_evidence(tmp_path, fault):
    data, definition, source, _ = fixtures(tmp_path)
    if fault == "duplicate":
        source.write_text(source.read_text() + source.read_text().splitlines()[0] + "\n")
    elif fault == "missing":
        source.write_text("\n".join(source.read_text().splitlines()[:-1]) + "\n")
    elif fault == "count":
        data["per_model_lost_counts"]["M1"] = 2
    elif fault == "accuracy":
        data["relaxed_accuracy_percent"]["M1"] = 74
    else:
        data["models"] = ["M1", "M2"]
        data["per_model_lost_counts"].pop("M3")
        data["relaxed_accuracy_percent"].pop("M3")
    save(definition, data)
    with pytest.raises(ValueError):
        cohort.run(definition, source, tmp_path / "output")


def test_original_raw_and_gold_reverification(tmp_path):
    _, definition, source, raw_rows = fixtures(tmp_path)
    root = raw_fixture(tmp_path, raw_rows)
    result = cohort.run(definition, source, tmp_path / "output", root)
    assert result["status"] == "PASS"
    assert result["raw_evidence_status"] == "RAW_AND_GOLD_REVERIFIED"
    assert result["gold_type_counts"] == {"nonnumeric_text": 1}
    record = json.loads((tmp_path / "output/COHORT_ROWS.jsonl").read_text())
    assert record["gold"]["answer"] == "yes"
    assert record["models"]["BASE"]["raw_text"] == "yes\n"
    assert record["models"]["M1"]["raw_text"] == "no"
    rawpath = root / cohort.group_relative("M1", "synthetic-0", "BASE")
    rawpath.write_text("{}")
    with pytest.raises(ValueError, match="Source SHA differs"):
        cohort.run(definition, source, tmp_path / "tamper_output", root)


def test_no_overwrite_or_write_into_original_source(tmp_path):
    _, definition, source, _ = fixtures(tmp_path)
    with pytest.raises(PermissionError):
        cohort.run(definition, source, source.parent / "output")
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError):
        cohort.run(definition, source, output)


def test_raw_delivery_keeps_unselected_model_outputs(tmp_path):
    data, definition, source, raw_rows = fixtures(tmp_path)
    data["expected_model_count"] = 5
    save(definition, data)
    extra = []
    for i in range(4):
        qid = f"synthetic-{i}"
        original = raw_rows[("BASE", qid)]
        raw_rows[("CONTROL", qid)] = dict(
            original, model_id="CONTROL", slot_id=f"CONTROL|CHARTQA|{qid}"
        )
        extra.append(
            {
                k: v
                for k, v in raw_rows[("CONTROL", qid)].items()
                if k not in ("raw_text", "tokens", "finish_reason")
            }
        )
    source.write_text(source.read_text() + "".join(json.dumps(row) + "\n" for row in extra))
    root = raw_fixture(tmp_path, raw_rows)
    result = cohort.run(definition, source, tmp_path / "output", root)
    assert result["loss_models"] == ["M1", "M2", "M3"]
    assert result["raw_output_count"] == 5
    record = json.loads((tmp_path / "output/COHORT_ROWS.jsonl").read_text())
    assert set(record["models"]) == {"BASE", "CONTROL", "M1", "M2", "M3"}
    assert record["models"]["CONTROL"]["raw_text"] == "yes\n"
