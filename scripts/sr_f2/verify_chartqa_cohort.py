#!/usr/bin/env python3
"""Verify a user-defined, post-hoc ChartQA loss cohort without changing frozen scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sr_f1.chartqa import score_chartqa

LABEL = "POST_HOC_WHITESPACE_SENSITIVITY"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class Sources:
    def __init__(self):
        self.files = {}

    def add(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        entry = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
        require(expected is None or entry["sha256"] == expected, f"Source SHA differs: {path}")
        if str(path) in self.files:
            require(self.files[str(path)] == entry, f"Source changed during audit: {path}")
        self.files[str(path)] = entry
        return entry

    def read(self, path, expected=None):
        self.add(path, expected)
        return json.loads(Path(path).read_text())

    def stable(self):
        for path, entry in list(self.files.items()):
            require(self.add(path) == entry, f"Source changed during verification: {path}")


def validate_definition(definition):
    baseline, models, qids = definition["baseline"], definition["models"], definition["qids"]
    require(
        isinstance(baseline, str)
        and isinstance(models, list)
        and models
        and all(isinstance(m, str) for m in models)
        and len(models) == len(set(models))
        and baseline not in models,
        "Invalid baseline/model selection",
    )
    require(
        isinstance(qids, list)
        and len(qids) == len(set(qids)) == definition["count"]
        and all(isinstance(q, str) and q for q in qids),
        "Duplicate/missing user cohort qids",
    )
    require(set(definition["per_model_lost_counts"]) == set(models), "Expected loss counts differ")
    require(
        set(definition["relaxed_accuracy_percent"]) == {baseline, *models},
        "Expected accuracy model identities differ",
    )
    return baseline, models, set(qids)


def read_scores(source, sources, definition):
    baseline, models, expected_qids = validate_definition(definition)
    source_identity = dict(sources.add(source))
    rows, all_slots, count = defaultdict(dict), set(), 0
    with Path(source).open() as stream:
        for line_number, line in enumerate(stream, 1):
            require(bool(line.strip()), f"Blank sidecar row at {line_number}")
            row = json.loads(line)
            model, qid, slot = row["model_id"], row["qid"], row["slot_id"]
            require(
                isinstance(model, str) and re.fullmatch(r"[A-Za-z0-9_-]+", model),
                "Unsafe model identifier",
            )
            require(row["label"] == LABEL, "Input is not the post-hoc whitespace sidecar")
            require(
                qid not in rows[model] and slot not in all_slots, "Duplicate score-sidecar slot"
            )
            require(slot == f"{model}|CHARTQA|{qid}", "ChartQA slot identity differs")
            for scheme in ("frozen", "whitespace_stripped"):
                require(
                    set(row[scheme]) == {"relaxed_accuracy", "exact_match"}
                    and all(type(v) is int and v in (0, 1) for v in row[scheme].values()),
                    "Malformed binary ChartQA score",
                )
            rows[model][qid] = dict(record=row, source_line=line_number)
            all_slots.add(slot)
            count += 1
    n = definition.get("questions_per_model", 2500)
    require(len(rows) == definition.get("expected_model_count", 13), "Incomplete model matrix")
    require(baseline in rows and set(models) <= set(rows), "Requested models absent")
    questions = set(rows[baseline])
    require(
        len(questions) == n and all(set(model_rows) == questions for model_rows in rows.values()),
        "ChartQA qid coverage differs between models",
    )
    for qid in questions:
        require(
            len({(v[qid]["record"]["root_id"], v[qid]["record"]["variant"]) for v in rows.values()})
            == 1,
            "Question/image/subset identity differs",
        )
    scores = {
        m: {q: r["record"]["whitespace_stripped"]["relaxed_accuracy"] for q, r in group.items()}
        for m, group in rows.items()
    }
    hits = {m: sum(v.values()) for m, v in scores.items()}
    accuracy = {m: hits[m] * 100 / n for m in scores}
    # Exact integer comparison implements the user's >4 percentage-point rule.
    selected = {m for m in scores if m != baseline and (hits[baseline] - hits[m]) * 100 > 4 * n}
    require(selected == set(models), "User model set differs from >4pp loss selection")
    lost = {
        m: {q for q in questions if scores[baseline][q] == 1 and scores[m][q] == 0} for m in models
    }
    actual = set.intersection(*(lost[m] for m in models))
    differences = dict(
        missing_user_qids=sorted(actual - expected_qids),
        extra_user_qids=sorted(expected_qids - actual),
    )
    require(
        actual == expected_qids,
        "User cohort differs from exact computed intersection: " + json.dumps(differences),
    )
    loss_counts = {m: len(lost[m]) for m in models}
    require(
        loss_counts == definition["per_model_lost_counts"], "Per-model lost-question counts differ"
    )
    require(
        all(
            Decimal(str(accuracy[m])) == Decimal(str(v))
            for m, v in definition["relaxed_accuracy_percent"].items()
        ),
        "Post-hoc relaxed accuracy percentages differ",
    )
    source_identity["rows"] = count
    return rows, dict(
        count=len(actual),
        qids=sorted(actual),
        baseline_model=baseline,
        loss_models=models,
        loss_counts=loss_counts,
        accuracy_percent=accuracy,
        selected_by_loss_over_four_pp=sorted(selected),
        source=source_identity,
        questions_per_model=n,
        model_count=len(rows),
        cohort_identity_verified=True,
    )


def gold_type(value):
    try:
        number = Decimal(str(value).strip())
        if number.is_finite():
            return "integer" if number == number.to_integral_value() else "decimal"
    except ArithmeticError:
        pass
    return "nonnumeric_text"


def group_relative(model, qid, baseline):
    group = object_hash(
        (model, 0 if model == baseline else 96, "CHARTQA", "plain_answer", None, qid)
    )
    return f"sealed/evaluation/{model}/chartqa/groups/{group}.json"


def index_references(source, sources, baseline, models, qids):
    path = Path(source).parent / "audit_raw_group_index.json"
    if not path.is_file():
        return {}
    wanted = {group_relative(m, q, baseline) for m in [baseline, *models] for q in qids}
    records = sources.read(path)
    result = {r["path"]: r for r in records if r["path"] in wanted}
    require(set(result) == wanted, "Archived raw-group index does not cover every cohort output")
    return result


def raw_evidence(root, sources, baseline, models, qids, score_rows):
    """Read only selected raw groups through their released endpoint hash chains."""
    root = Path(root).resolve(strict=True)
    identity = sources.read(root / "external/chartqa/IDENTITY.json")
    require(identity["status"] == "AVAILABLE_VERIFIED", "ChartQA gold is unavailable")
    manifest = root / "external/chartqa/TEST_MANIFEST.jsonl"
    sources.add(manifest, identity["manifest_sha256"])
    gold = {}
    for line in manifest.open():
        item = json.loads(line)
        require(item["qid"] not in gold, "Duplicate ChartQA gold qid")
        gold[item["qid"]] = item
    require(set(gold) == set(score_rows[baseline]), "Gold manifest qid coverage differs")
    final = sources.read(root / "endpoint_orchestration/COMPLETE.json")
    require(final["status"] == "EVALUATIONS_COMPLETE_REPORT_PENDING", "Prior scores unreleased")
    releases = {r["model_id"]: r for r in final["releases"]}
    result = defaultdict(dict)
    for model in [baseline, *models]:
        release = sources.read(root / "released/evaluation" / f"{model}.json")
        require(
            object_hash(release) == releases[model]["release_sha256"], "Release identity differs"
        )
        directory = root / "sealed/evaluation" / model
        done = sources.read(directory / "COMPLETE.json")
        require(object_hash(done) == release["evaluation_sha256"], "Endpoint completion differs")
        panel = next(p for p in done["panels"] if p["panel"] == "chartqa")
        finished = sources.read(directory / "chartqa/COMPLETE.json", panel["completion_sha256"])
        groups = {g["group_id"]: g for g in finished["groups"]}
        for qid in qids:
            relative = group_relative(model, qid, baseline)
            group_id = Path(relative).stem
            reference = groups[group_id]
            payload = sources.read(root / relative, reference["raw_sha256"])
            require(
                len(payload["records"]) == reference["records"] == 1
                and object_hash(payload["records"]) == payload["records_sha256"],
                "Raw ChartQA group content differs",
            )
            raw, expected = payload["records"][0], score_rows[model][qid]["record"]
            require(
                all(
                    raw[k] == expected[k]
                    for k in ("slot_id", "model_id", "qid", "root_id", "variant")
                ),
                "Raw/sidecar model/question/image identity differs",
            )
            item = gold[qid]
            require(item["image_id"] == raw["root_id"], "Gold/raw image identity differs")
            require(
                score_chartqa(item["answer"], raw["raw_text"]) == expected["frozen"]
                and score_chartqa(str(item["answer"]).strip(), raw["raw_text"].strip())
                == expected["whitespace_stripped"],
                "Raw gold/output rescoring differs from sidecar",
            )
            result[qid][model] = dict(
                raw_text=raw["raw_text"],
                tokens=raw["tokens"],
                finish_reason=raw.get("finish_reason"),
                raw_source=dict(
                    path=str(root / relative),
                    sha256=reference["raw_sha256"],
                    records_sha256=payload["records_sha256"],
                    row_index=0,
                ),
            )
    return gold, result


def run(definition_path, source, output, raw_root=None):
    sources = Sources()
    definition = sources.read(definition_path)
    output = Path(output).resolve()
    for protected in [
        Path(source).resolve().parent,
        *([Path(raw_root).resolve()] if raw_root else []),
    ]:
        if output == protected or output.is_relative_to(protected):
            raise PermissionError("Output must be separate from prior source/raw roots")
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        status="RUNNING",
        label=LABEL,
        scheme="whitespace_stripped",
        metric="relaxed_accuracy",
        definition=sources.add(definition_path),
        official_frozen_scores_changed=False,
        generated_model_outputs=0,
        started_at=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(__file__),
    )
    try:
        score_rows, result = read_scores(source, sources, definition)
        receipt.update(result)
        baseline, qids = result["baseline_model"], result["qids"]
        all_models = sorted(score_rows)
        other_models = [m for m in all_models if m != baseline]
        indexed = index_references(source, sources, baseline, other_models, qids)
        gold, raw = (
            raw_evidence(raw_root, sources, baseline, other_models, qids, score_rows)
            if raw_root
            else ({}, {})
        )
        types = Counter()
        with (output / "COHORT_ROWS.jsonl").open("x") as stream:
            for qid in qids:
                item = dict(
                    qid=qid,
                    label=LABEL,
                    source_sidecar_sha256=result["source"]["sha256"],
                    models={},
                )
                for model in all_models:
                    reference = score_rows[model][qid]
                    model_row = dict(
                        score_sidecar=reference["record"], source_line=reference["source_line"]
                    )
                    relative = group_relative(model, qid, baseline)
                    if relative in indexed:
                        model_row["archived_raw_source"] = indexed[relative]
                    if raw:
                        model_row.update(raw[qid][model])
                    item["models"][model] = model_row
                if gold:
                    item["gold"] = gold[qid]
                    item["gold_type"] = gold_type(gold[qid]["answer"])
                    types[item["gold_type"]] += 1
                stream.write(
                    json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                )
        receipt["raw_evidence_status"] = (
            "RAW_AND_GOLD_REVERIFIED" if raw else "SOURCE_REFERENCES_ONLY"
        )
        receipt["gold_type_counts"] = dict(types) if raw else None
        receipt["raw_model_ids"] = all_models
        receipt["raw_output_count"] = len(qids) * len(all_models) if raw else 0
        receipt["rows"] = dict(
            path="COHORT_ROWS.jsonl", sha256=sha(output / "COHORT_ROWS.jsonl"), count=len(qids)
        )
        sources.stable()
        receipt["status"] = "PASS"
    except Exception:
        receipt.update(status="FAIL", cohort_identity_verified=False, error=traceback.format_exc())
        raise
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest = output / "SOURCE_MANIFEST.json"
        manifest.write_text(
            json.dumps(dict(files=list(sources.files.values())), indent=2, sort_keys=True) + "\n"
        )
        receipt["source_manifest_sha256"] = sha(manifest)
        (output / "COHORT_VERIFICATION.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n"
        )
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    args = parser.parse_args(argv)
    receipt = run(args.definition, args.source, args.output, args.raw_root)
    print(json.dumps({k: receipt[k] for k in ("status", "count", "label", "raw_evidence_status")}))


if __name__ == "__main__":
    main()
