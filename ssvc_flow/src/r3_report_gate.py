"""Read-only R3 report acceptance by deterministic CPU statistical recomputation.

The caller must already verify the run manifest, raw proposal annotations, model
provenance, and candidate checkpoints. This additional gate binds report content
to that evidence; it neither executes a model nor certifies overlap or safety.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .core import canonical_hash, file_hash
from .r3_response import analyze_control_responses

LAMBDAS = (0, 0.01, 0.25, 1, 2)
RESPONSE_BANKS = (0, 6)
BOOTSTRAP = {"replicates": 5000, "seed": 20260909}


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _parse(text, context):
    try:
        result = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
        canonical_hash(result)  # Also rejects finite JSON literals that overflow to infinity.
        return result
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {context}: {exc}") from exc


def _file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or escaping report artifact: {relative}")
    return path


def _json(root, relative):
    return _parse(_file(root, relative).read_text(encoding="utf-8"), str(relative))


def _same_json(actual, expected, context):
    if canonical_hash(actual) != canonical_hash(expected):
        raise ValueError(f"{context} differs from recomputed evidence")


def _summaries(value):
    if not isinstance(value, dict):
        raise ValueError("Bank summaries must contain all twelve banks")
    banks = {}
    for key, summary in value.items():
        if type(key) not in (str, int) or str(key) not in {str(i) for i in range(12)}:
            raise ValueError("Unknown bank index")
        index = int(key)
        if index in banks or not isinstance(summary, dict):
            raise ValueError("Duplicate or malformed bank summary")
        candidates = summary.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 5:
            raise ValueError("Every bank must have five candidate summaries")
        lambdas = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("Malformed candidate summary")
            lam = candidate.get("lambda")
            if type(lam) not in (float, int) or lam not in LAMBDAS or lam in lambdas:
                raise ValueError("Candidate lambda grid differs from the fixed five values")
            lambdas.add(lam)
            if candidate.get("candidate_id") != f"bank_{index:02d}_lambda_{lam:g}":
                raise ValueError("Candidate ID does not bind its bank and lambda")
            if "bank_index" in candidate and candidate["bank_index"] != index:
                raise ValueError("Candidate bank_index differs from its enclosing bank")
            for field in ("parameter_hash", "optimizer_state_hash"):
                if not isinstance(candidate.get(field), str) or not candidate[field]:
                    raise ValueError(f"Candidate {field} is missing")
            canonical_hash(candidate)
        banks[index] = summary
    if set(banks) != set(range(12)):
        raise ValueError("Bank summaries must contain all twelve banks")
    return {index: banks[index] for index in range(12)}


def _proposals(records):
    if not isinstance(records, (list, tuple)) or len(records) != 768:
        raise ValueError("Exactly 768 verified proposal records are required")
    result = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Malformed proposal record")
        key = row.get("sample_key")
        if not isinstance(key, str) or not key or key in result:
            raise ValueError("Missing or duplicate proposal sample_key")
        if row.get("record_hash") != canonical_hash(
            {k: v for k, v in row.items() if k != "record_hash"}
        ):
            raise ValueError("Proposal record hash mismatch")
        result[key] = row
    return result


def _scores(root, candidate, proposal, proposal_hash, hashes):
    cid = candidate["candidate_id"]
    prefix = Path("control_scores") / cid
    identity_path = prefix / "identity.json"
    identity = _json(root, identity_path)
    expected = {
        "unit": "control_likelihood",
        "candidate_id": cid,
        "candidate_parameter_hash": candidate["parameter_hash"],
        "candidate_optimizer_hash": candidate["optimizer_state_hash"],
        "proposal_hash": proposal_hash,
    }
    if not isinstance(identity, dict) or any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Control score identity differs from supplied evidence: {cid}")
    relative = prefix / "samples.jsonl"
    path = _file(root, relative)
    content = path.read_text(encoding="utf-8")
    if not content or not content.endswith("\n"):
        raise ValueError(f"Truncated control score ledger: {cid}")
    result = {}
    for line in content.splitlines():
        row = _parse(line, f"control score {cid}")
        if not isinstance(row, dict):
            raise ValueError(f"Malformed control score record: {cid}")
        source = row.get("proposal_sample_key")
        if not isinstance(source, str) or source not in proposal or source in result:
            raise ValueError(f"Unknown or duplicate control score proposal key: {cid}")
        original = proposal[source]
        expected_row = {
            "sample_key": canonical_hash([identity, source]),
            "candidate_id": cid,
            "candidate_parameter_hash": candidate["parameter_hash"],
            "proposal_record_hash": original["record_hash"],
            "token_ids": original.get("token_ids"),
        }
        checks = row.get("execution_checks")
        if (
            row.get("record_hash")
            != canonical_hash({k: v for k, v in row.items() if k != "record_hash"})
            or any(canonical_hash(row.get(k)) != canonical_hash(v) for k, v in expected_row.items())
            or not isinstance(checks, dict)
            or checks.get("passed") is not True
            or checks.get("faults") != []
        ):
            raise ValueError(f"Control score hash, source, or execution binding mismatch: {cid}")
        values = row.get("candidate_token_logprobs")
        tokens = original.get("token_ids")
        if (
            not isinstance(tokens, list)
            or not isinstance(values, list)
            or not values
            or len(values) != len(tokens)
            or any(type(v) not in (int, float) or not math.isfinite(v) or v > 1e-5 for v in values)
            or row.get("sequence_logprob") != math.fsum(values)
        ):
            raise ValueError(f"Invalid control score token probabilities: {cid}")
        result[source] = values
    if set(result) != set(proposal):
        raise ValueError(f"Incomplete control score coverage: {cid}")
    hashes[str(identity_path)] = file_hash(_file(root, identity_path))
    hashes[str(relative)] = file_hash(path)
    return result


def _response_rows(results):
    rows = []
    for index, result in results.items():
        for cid, candidate in result["candidates"].items():
            for scope, metrics in candidate["responses"].items():
                for metric, values in metrics.items():
                    row = {
                        "bank_index": index,
                        "candidate_id": cid,
                        "scope": scope,
                        "metric": metric,
                        **{k: v for k, v in values.items() if k != "ci"},
                    }
                    for ci_type, interval in values["ci"].items():
                        row.update({f"{ci_type}_CI_{k}": v for k, v in interval.items()})
                    rows.append(row)
    return rows


def _csv_value(value, expected):
    if expected is None:
        if value != "":
            raise ValueError("Expected an empty CSV NA cell")
        return None
    if type(expected) is bool:
        if value not in ("True", "False"):
            raise ValueError("Expected a literal CSV boolean")
        return value == "True"
    if type(expected) is int:
        return int(value)
    if type(expected) is float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Nonfinite CSV number")
        return result
    return value


def _check_csv(root, expected):
    key_fields = ("bank_index", "candidate_id", "scope", "metric")
    expected_rows = {tuple(row[k] for k in key_fields): row for row in expected}
    fields = {k for row in expected for k in row}
    seen = set()
    try:
        with _file(root, "paired_response.csv").open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if (
                not reader.fieldnames
                or len(reader.fieldnames) != len(fields)
                or set(reader.fieldnames) != fields
            ):
                raise ValueError("CSV report columns differ from the estimator schema")
            for row in reader:
                if set(row) != fields or any(value is None for value in row.values()):
                    raise ValueError("Malformed CSV report row")
                key = (int(row["bank_index"]), *(row[k] for k in key_fields[1:]))
                if key not in expected_rows or key in seen:
                    raise ValueError("Unknown or duplicate CSV response key")
                reference = expected_rows[key]
                for field in fields:
                    value = reference.get(field)
                    if _csv_value(row[field], value) != value:
                        raise ValueError(f"CSV response differs at {key}/{field}")
                seen.add(key)
    except (csv.Error, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid paired response CSV: {exc}") from exc
    if len(expected_rows) != 280 or seen != set(expected_rows):
        raise ValueError("CSV response does not contain all 280 measured cells")
    return len(seen)


def _scalar_equal(actual, expected):
    if expected is None or type(expected) is bool:
        return actual is expected
    if type(expected) in (float, int):
        return type(actual) in (float, int) and math.isfinite(actual) and actual == expected
    return type(actual) is type(expected) and actual == expected


def _check_parquet(root, banks):
    import pyarrow as pa
    import pyarrow.parquet as pq

    expected = {}
    for index, summary in banks.items():
        for candidate in summary["candidates"]:
            scalars = {
                k: v
                for k, v in candidate.items()
                if v is None or isinstance(v, (bool, int, float, str))
            }
            expected[candidate["candidate_id"]] = {
                "bank_index": index,
                **scalars,
                "probability_response": "MEASURED_CONTROL_IS"
                if index in RESPONSE_BANKS
                else "NOT_MEASURED",
                "audit_json": candidate,
            }
    try:
        table = pq.ParquetFile(_file(root, "gradients_summary.parquet")).read()
    except (pa.ArrowException, OSError, ValueError) as exc:
        raise ValueError(f"Invalid gradient parquet artifact: {exc}") from exc
    # Match Table.from_pylist: its output columns come from the first row. Full
    # candidate content, including later optional scalars, is bound by audit_json.
    fields = set(next(iter(expected.values())))
    if (
        table.num_rows != 60
        or len(table.column_names) != len(fields)
        or set(table.column_names) != fields
    ):
        raise ValueError("Gradient parquet does not contain the expected 60-row schema")
    seen = set()
    for row in table.to_pylist():
        cid = row.get("candidate_id")
        if not isinstance(cid, str) or cid not in expected or cid in seen:
            raise ValueError("Unknown or duplicate gradient parquet candidate")
        reference = expected[cid]
        for field in fields - {"audit_json"}:
            if not _scalar_equal(row[field], reference.get(field)):
                raise ValueError(f"Gradient parquet scalar differs: {cid}/{field}")
        audit = _parse(row["audit_json"], f"parquet audit_json {cid}")
        _same_json(audit, reference["audit_json"], f"Gradient parquet audit_json {cid}")
        seen.add(cid)
    if seen != set(expected):
        raise ValueError("Gradient parquet candidate coverage is incomplete")
    return len(seen)


def validate_response_artifacts(root, *, proposal_records, bank_summaries):
    """Return a PASS audit only after every measured report value is reconstructed.

    Raises ValueError/FloatingPointError on missing, altered, malformed, or
    nonfinite evidence. The returned execution kind describes only this CPU
    check; real CUDA provenance remains an independent caller-owned gate.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("R3 evidence root is missing")
    banks = _summaries(bank_summaries)
    proposal = _proposals(proposal_records)
    proposal_hash = canonical_hash([r["record_hash"] for r in proposal_records])
    results, audits, hashes = {}, {}, {}
    for index in RESPONSE_BANKS:
        candidates = banks[index]["candidates"]
        scores = {
            c["candidate_id"]: _scores(root, c, proposal, proposal_hash, hashes) for c in candidates
        }
        baseline = next(c["candidate_id"] for c in candidates if c["lambda"] == 0)
        recomputed = analyze_control_responses(
            proposal_records,
            scores,
            baseline_key=baseline,
            bank_index=index,
            bootstrap_replicates=BOOTSTRAP["replicates"],
            seed=BOOTSTRAP["seed"],
        )
        relative = f"response_bank_{index:02d}.json"
        stored = _json(root, relative)
        _same_json(stored, recomputed, f"R3 response bank {index}")
        hashes[relative] = file_hash(_file(root, relative))
        audits[str(index)] = {
            "status": "PASS",
            "baseline_candidate_id": baseline,
            "candidate_count": len(candidates),
            "proposal_sequences": len(proposal),
            "response_sha256": hashes[relative],
            "recomputed_sha256": canonical_hash(recomputed),
        }
        results[index] = recomputed
    csv_count = _check_csv(root, _response_rows(results))
    parquet_count = _check_parquet(root, banks)
    for relative in ("paired_response.csv", "gradients_summary.parquet"):
        hashes[relative] = file_hash(_file(root, relative))
    return {
        "status": "PASS",
        "execution_kind": "CPU_MATH",
        "scope": "Read-only statistical recomputation of R3 control IS and gradient reports",
        "proposal_sequences": len(proposal),
        "proposal_record_hashes_sha256": proposal_hash,
        "response_candidates": sum(a["candidate_count"] for a in audits.values()),
        "response_banks": audits,
        "paired_response_rows": csv_count,
        "gradient_candidate_rows": parquet_count,
        "artifact_sha256": hashes,
        "bootstrap": dict(BOOTSTRAP),
        "safety_status": "NOT_CERTIFIED",
        "unmeasured_response_banks": [i for i in banks if i not in RESPONSE_BANKS],
    }
