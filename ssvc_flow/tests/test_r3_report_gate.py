"""Report acceptance recomputes real CPU statistics from immutable score ledgers."""

import copy
import csv
import importlib
import importlib.util
import json
import math
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.core import canonical_hash, file_hash, write_json
from src.r3_response import analyze_control_responses
from src.r3_runtime import _write_reports

LAMBDAS = (0, 0.01, 0.25, 1, 2)


def test_report_validator_api_exists():
    assert importlib.util.find_spec("src.r3_report_gate") is not None


def _seal(row):
    row.pop("record_hash", None)
    row["record_hash"] = canonical_hash(row)
    return row


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(scope="module")
def complete_reports(tmp_path_factory):
    root = tmp_path_factory.mktemp("r3_report_fixture")
    proposal = []
    for family in ("cross_series", "duplicate", "trend"):
        for scene in range(8):
            for interface in ("IMAGE_CUE_FRESH", "SYMBOLIC_FRESH"):
                prompt = f"{family}-{scene}-{interface}"
                for sample in range(16):
                    length = sample % 3 + 1
                    x_count = scene + int(interface == "SYMBOLIC_FRESH")
                    category = (
                        "X"
                        if sample < x_count
                        else "S"
                        if sample < 8
                        else "W"
                        if sample < 12
                        else "I"
                    )
                    proposal.append(
                        _seal(
                            {
                                "sample_key": f"{prompt}-{sample}",
                                "prompt_id": prompt,
                                "base_scene_id": f"{family}-{scene}",
                                "family": family,
                                "interface": interface,
                                "category": category,
                                "token_ids": [sample + 1] * length,
                                "old_logprobs": [-4.0] * length,
                                "sample_index": sample,
                            }
                        )
                    )
    summaries, responses = {}, {}
    for bank in range(12):
        candidates, scores = [], {}
        for lam in LAMBDAS:
            cid = f"bank_{bank:02d}_lambda_{lam:g}"
            candidate = {
                "lambda": lam,
                "candidate_id": cid,
                "parameter_hash": canonical_hash([bank, lam, "parameters"]),
                "optimizer_state_hash": canonical_hash([bank, lam, "optimizer"]),
                "gradient_delta_l2": lam * 0.17,
                "parameter_auxiliary_delta_l2": lam * 1e-5,
                "all_valid": False,
                "optional": None,
                "gradient": {"finite": True, "nested": {"norm": 0.25 + lam}},
            }
            candidates.append(candidate)
            if bank not in (0, 6):
                continue
            score_root = root / "control_scores" / cid
            score_root.mkdir(parents=True)
            identity = {
                "unit": "control_likelihood",
                "candidate_id": cid,
                "candidate_parameter_hash": candidate["parameter_hash"],
                "candidate_optimizer_hash": candidate["optimizer_state_hash"],
                "proposal_hash": canonical_hash([row["record_hash"] for row in proposal]),
            }
            write_json(score_root / "identity.json", identity)
            rows, scores[cid] = [], {}
            for row in proposal:
                old = row["old_logprobs"]
                shift = (0.01 + lam * (1 if row["category"] == "X" else -0.2)) / 20
                new = [old[0] + shift, *old[1:]]
                scores[cid][row["sample_key"]] = new
                rows.append(
                    _seal(
                        {
                            "sample_key": canonical_hash([identity, row["sample_key"]]),
                            "proposal_sample_key": row["sample_key"],
                            "candidate_id": cid,
                            "candidate_parameter_hash": candidate["parameter_hash"],
                            "proposal_record_hash": row["record_hash"],
                            "token_ids": row["token_ids"],
                            "candidate_token_logprobs": new,
                            "sequence_logprob": math.fsum(new),
                            "execution_kind": "CPU_FAKE_ADAPTER_FIXTURE",
                            "execution_checks": {"passed": True, "faults": []},
                        }
                    )
                )
            _write_rows(score_root / "samples.jsonl", rows)
        summaries[bank] = {"status": "PASS", "candidates": candidates}
        if bank in (0, 6):
            responses[bank] = analyze_control_responses(
                proposal, scores, baseline_key=f"bank_{bank:02d}_lambda_0", bank_index=bank
            )
            write_json(root / f"response_bank_{bank:02d}.json", responses[bank])
    _write_reports(
        root, summaries, responses, {}, {"execution_kind": "CPU_FIXTURE", "status": "PASS"}
    )
    # Match the caller's actual sorted JSON manifest roundtrip and string bank keys.
    summaries = json.loads((root / "candidate_manifest.json").read_text())["banks"]
    return root, proposal, summaries


@pytest.fixture
def report_case(complete_reports, tmp_path):
    source, proposal, summaries = complete_reports
    root = tmp_path / "reports"
    shutil.copytree(source, root)
    return root, copy.deepcopy(proposal), copy.deepcopy(summaries)


def _validate(case):
    api = importlib.import_module("src.r3_report_gate")
    root, proposal, summaries = case
    return api.validate_response_artifacts(
        root, proposal_records=proposal, bank_summaries=summaries
    )


def test_real_report_recomputation_is_read_only_and_bound(report_case):
    root, proposal, summaries = report_case
    inputs = copy.deepcopy((proposal, summaries))
    before = {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()}
    result = _validate(report_case)
    after = {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()}
    assert before == after
    assert (proposal, summaries) == inputs
    assert result["status"] == "PASS"
    assert result["execution_kind"] == "CPU_MATH"
    assert result["proposal_sequences"] == 768
    assert result["response_candidates"] == 10
    assert result["paired_response_rows"] == 280
    assert result["gradient_candidate_rows"] == 60
    assert result["bootstrap"] == {"replicates": 5000, "seed": 20260909}
    assert set(result["response_banks"]) == {"0", "6"}
    assert result["response_banks"]["6"]["baseline_candidate_id"] == "bank_06_lambda_0"
    assert result["artifact_sha256"]["paired_response.csv"] == before["paired_response.csv"]
    response = json.loads((root / "response_bank_00.json").read_text())
    interval = response["candidates"]["bank_00_lambda_1"]["responses"]["overall"]["pX"]["ci"]
    assert interval["estimate"]["half_width"] > 0
    assert result["safety_status"] == "NOT_CERTIFIED"


@pytest.mark.parametrize("mutation", ["placeholder", "estimate", "ci", "warning", "bootstrap"])
def test_rejects_semantically_forged_response(report_case, mutation):
    path = report_case[0] / "response_bank_06.json"
    result = json.loads(path.read_text())
    values = result["candidates"]["bank_06_lambda_1"]["responses"]["overall"]["pX"]
    if mutation == "placeholder":
        result = {}
    elif mutation == "estimate":
        values["absolute_delta"] += 1e-12
    elif mutation == "ci":
        values["ci"]["auxiliary_delta"]["low"] = 123.0
    elif mutation == "warning":
        result["safety_status"] = "CERTIFIED"
    else:
        result["bootstrap_contract"]["seed"] = 0
    write_json(path, result)
    with pytest.raises(ValueError, match="response"):
        _validate(report_case)


def _csv_rows(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def _csv_write(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("mutation", ["value", "missing", "duplicate", "none", "nan"])
def test_rejects_forged_or_incomplete_paired_csv(report_case, mutation):
    path = report_case[0] / "paired_response.csv"
    fields, rows = _csv_rows(path)
    if mutation == "value":
        rows[-1]["auxiliary_delta"] = "0.123"
    elif mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = rows[0].copy()
    elif mutation == "none":
        rows[0]["observed_delta"] = "0"
    else:
        rows[0]["estimate"] = "NaN"
    _csv_write(path, fields, rows)
    with pytest.raises(ValueError, match="CSV"):
        _validate(report_case)


@pytest.mark.parametrize("mutation", ["audit", "scalar", "missing", "unmeasured", "duplicate"])
def test_rejects_forged_gradient_parquet(report_case, mutation):
    path = report_case[0] / "gradients_summary.parquet"
    rows = pq.ParquetFile(path).read().to_pylist()
    if mutation == "audit":
        rows[-1]["audit_json"] = "{}"
    elif mutation == "scalar":
        rows[-1]["gradient_delta_l2"] += 0.01
    elif mutation == "missing":
        rows.pop()
    elif mutation == "unmeasured":
        rows[5]["probability_response"] = "MEASURED_CONTROL_IS"
    else:
        rows[-1] = rows[0].copy()
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="parquet"):
        _validate(report_case)


def test_accepts_semantic_json_parquet_and_csv_reordering(report_case):
    root = report_case[0]
    parquet = root / "gradients_summary.parquet"
    rows = pq.ParquetFile(parquet).read().to_pylist()
    for row in rows:
        row["audit_json"] = json.dumps(json.loads(row["audit_json"]), sort_keys=True)
    pq.write_table(pa.Table.from_pylist(list(reversed(rows))), parquet)
    path = root / "paired_response.csv"
    fields, rows = _csv_rows(path)
    _csv_write(path, list(reversed(fields)), list(reversed(rows)))
    assert _validate(report_case)["status"] == "PASS"


@pytest.mark.parametrize(
    "mutation", ["binding", "duplicate", "truncated", "rehashed_score", "token_boolean"]
)
def test_rejects_changed_score_evidence(report_case, mutation):
    path = report_case[0] / "control_scores/bank_00_lambda_0/samples.jsonl"
    rows = _read_rows(path)
    if mutation == "binding":
        rows[0]["proposal_record_hash"] = "forged"
        _seal(rows[0])
    elif mutation == "duplicate":
        rows[-1] = rows[0].copy()
    elif mutation == "truncated":
        path.write_text(path.read_text().rstrip("\n"))
    elif mutation == "token_boolean":
        rows[0]["token_ids"] = [True]
        _seal(rows[0])
    else:
        rows[0]["candidate_token_logprobs"][0] += 0.1
        rows[0]["sequence_logprob"] = math.fsum(rows[0]["candidate_token_logprobs"])
        _seal(rows[0])
    if mutation != "truncated":
        _write_rows(path, rows)
    with pytest.raises(ValueError):
        _validate(report_case)


@pytest.mark.parametrize("mutation", ["proposal", "banks", "candidate", "path"])
def test_rejects_invalid_boundary_inputs(report_case, mutation):
    _, proposal, summaries = report_case
    if mutation == "proposal":
        proposal.pop()
    elif mutation == "banks":
        del summaries["11"]
    elif mutation == "candidate":
        summaries["11"]["candidates"].pop()
    else:
        summaries["0"]["candidates"][0]["candidate_id"] = "../escape"
    with pytest.raises(ValueError):
        _validate(report_case)


@pytest.mark.parametrize(
    ("relative", "payload"),
    [
        ("gradients_summary.parquet", "{}"),
        ("response_bank_00.json", '{"status": "PASS", "status": "FAIL"}'),
        ("response_bank_00.json", '{"estimate": NaN}'),
        ("response_bank_00.json", '{"estimate": 1e9999}'),
    ],
)
def test_rejects_invalid_report_serializations(report_case, relative, payload):
    (report_case[0] / relative).write_text(payload)
    with pytest.raises(ValueError):
        _validate(report_case)
