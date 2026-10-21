"""Offline native/semantic replay, blinded audit preparation and C evidence gate."""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from srf4_ref.credit import combined_advantages
from srf4_ref.validation import c_gate, wilson_lower

from .verification import VERIFIER_VERSION, observation, raw_text, verify_semantics

SHUFFLE_SEEDS = tuple(2026102205 + i for i in range(10))


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _delta(base, updated, observations):
    changes = [b - a for a, b in zip(base, updated, strict=False)]
    failures = [i for i, row in enumerate(observations) if row.j == 0]
    return dict(
        l1=sum(abs(x) for x in changes),
        linf=max(map(abs, changes), default=0),
        zero_to_nonzero=any(abs(a) <= 1e-8 < abs(b) for a, b in zip(base, updated, strict=False)),
        sign_flips=sum(a * b < 0 for a, b in zip(base, updated, strict=False)),
        failure_ranking_changed=any(
            (base[i] > base[j]) != (updated[i] > updated[j])
            or (base[i] < base[j]) != (updated[i] < updated[j])
            for offset, i in enumerate(failures)
            for j in failures[offset + 1 :]
        ),
    )


def replay_group(group, native_advantage):
    """Replay one complete group using an injected, audited native std function.

    Input: {group_id, source, block, family, world_id, rows:[{task,record,score}]}.
    The callable must also accept the scalar offline candidates. It owns the
    native ddof/epsilon behavior; this module supplies no default replacement.
    """
    rows = group["rows"]
    if not rows:
        raise ValueError("Empty replay group")
    labels = [
        row.get("label") or verify_semantics(row["task"], row["record"], row["score"])
        for row in rows
    ]
    observations = [observation(label) for label in labels]
    if len({row.row_id for row in observations}) != len(rows):
        raise ValueError("Duplicate answer identity within replay group")
    rewards = [row.j for row in observations]
    base = list(map(float, native_advantage(rewards)))
    if len(base) != len(rows) or not all(map(math.isfinite, base)):
        raise ValueError("Native advantage callable returned invalid output")
    result = {
        key: group.get(key) for key in ("group_id", "source", "block", "family", "world_id", "step")
    }
    result.update(
        verifier_version=VERIFIER_VERSION,
        row_ids=[x.row_id for x in observations],
        raw_J=rewards,
        native_scores=[label["native_score"] for label in labels],
        labels=labels,
        A_J=base,
        scalar={},
        bounded={},
        shuffle=[],
    )
    failures = [i for i, row in enumerate(observations) if row.j == 0]
    unknown = any(
        observations[i].perception == "UNKNOWN" or observations[i].selection == "UNKNOWN"
        for i in failures
    )
    for mode in ("P", "PS"):
        for weight in (0.01, 0.1, 0.25):
            candidate = list(rewards)
            if not unknown:
                for i in failures:
                    row = observations[i]
                    candidate[i] += (
                        weight
                        * (row.perception == "PASS")
                        * (mode == "P" or row.selection == "PASS")
                    )
            changed = list(map(float, native_advantage(candidate)))
            result["scalar"][f"{mode}_{weight}"] = dict(
                rewards=candidate,
                advantages=changed,
                abstained_unknown=unknown,
                **_delta(base, changed, observations),
            )
    for arm in ("J", "EVID", "SELECT"):
        updated, meta = combined_advantages(base, observations, arm, beta=0.25)
        result["bounded"][arm] = dict(
            advantages=updated,
            residual=[b - a for a, b in zip(base, updated, strict=False)],
            metadata=meta,
            **_delta(base, updated, observations),
        )
    for seed in SHUFFLE_SEEDS:
        updated, meta = combined_advantages(
            base, observations, "SELECT_SHUF", beta=0.25, shuffle_seed=seed
        )
        result["shuffle"].append(
            dict(
                seed=seed,
                advantages=updated,
                residual=[b - a for a, b in zip(base, updated, strict=False)],
                metadata=meta,
                **_delta(base, updated, observations),
            )
        )
    result["select_vs_evid_changed"] = any(
        abs(a - b) > 1e-8
        for a, b in zip(
            result["bounded"]["SELECT"]["advantages"],
            result["bounded"]["EVID"]["advantages"],
            strict=False,
        )
    )
    return result


def replay_groups(groups, native_advantage, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts, coverage = Counter(), Counter()
    signal_groups, signal_worlds, by_block = Counter(), defaultdict(set), Counter()
    group_ids, row_ids = set(), set()
    with (output_dir / "GROUP_REPLAY.jsonl").open("w") as stream:
        for group in groups:
            identity = (group.get("source"), group.get("block"), group["group_id"])
            if identity in group_ids:
                raise ValueError(f"Duplicate replay group {identity}")
            group_ids.add(identity)
            result = replay_group(group, native_advantage)
            for row_id in result["row_ids"]:
                if row_id in row_ids:
                    raise ValueError(f"Duplicate replay answer {row_id}")
                row_ids.add(row_id)
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            counts[str(group.get("source"))] += 1
            j = result["raw_J"]
            group_type = "ALL_ZERO" if not any(j) else "ALL_ONE" if all(j) else "MIXED"
            for label in result["labels"]:
                for dimension, value in label["dimensions"].items():
                    key = (
                        str(group.get("source")),
                        str(group.get("family")),
                        str(group.get("block")),
                        group_type,
                        str(group.get("step", 0)),
                        dimension,
                        value["status"],
                    )
                    coverage[key] += 1
            if group.get("source") == "C_TRAIN_PROBE" and result["select_vs_evid_changed"]:
                family, block = group["family"], str(group["block"])
                block = f"b{block}" if block in ("0", "1", "2") else block
                signal_groups[family] += 1
                signal_worlds[family].add(group["world_id"])
                by_block[block] += 1
    summary = dict(
        status="REPLAY_COMPLETE",
        verifier_version=VERIFIER_VERSION,
        groups_by_source=dict(counts),
        answers=len(row_ids),
        signal={
            family: dict(
                select_vs_evid_changed_groups=signal_groups[family],
                independent_worlds=len(signal_worlds[family]),
            )
            for family in ("INTERVAL", "TOPK")
        },
        signal_by_block=dict(by_block),
        coverage=[
            dict(
                source=k[0],
                family=k[1],
                block=k[2],
                group_type=k[3],
                step=k[4],
                dimension=k[5],
                status=k[6],
                count=v,
            )
            for k, v in sorted(coverage.items())
        ],
    )
    _json(output_dir / "SUMMARY.json", summary)
    return summary


def prepare_blind_audit(
    groups, output_dir, *, seed=2026102223, representative_n=120, positive_n=80
):
    """Create reproducible blinded records, with private sampling metadata apart.

    Input groups are replay inputs with optional replay result in ``replay``.
    The caller must freeze the parser before calling. No result is auto-HUMAN.
    """
    rng = random.Random(seed)
    strata, positive = defaultdict(list), defaultdict(list)
    identifiers = set()
    for group in groups:
        if "CONFIRM" in str(group.get("source", "")):
            raise ValueError("Confirmation data cannot enter verifier development or audit")
        replay = group.get("replay")
        labels = (
            replay["labels"]
            if replay
            else [
                x.get("label") or verify_semantics(x["task"], x["record"], x["score"])
                for x in group["rows"]
            ]
        )
        observations = [observation(label) for label in labels]
        residual, _ = combined_advantages([0.0] * len(labels), observations, "SELECT")
        for index, (row, label) in enumerate(zip(group["rows"], labels, strict=False)):
            if label["row_id"] in identifiers:
                raise ValueError("Duplicate row in human audit pool")
            identifiers.add(label["row_id"])
            native = label["native_score"]
            stratum = (
                str(group.get("family")),
                str(group.get("block")),
                int(bool(native["J"])),
                int(label["normalization"]["protocol_valid"]),
            )
            item = dict(row=row, label=label, positive=residual[index] > 1e-8, stratum=stratum)
            strata[stratum].append(item)
            if item["positive"]:
                positive[label["world_id"]].append(item)
    # Rarity is fixed as <=10% for the observed (P,S,S-rule) pattern within
    # family/block/J/protocol strata. This only stratifies the blinded sample.
    refined = defaultdict(list)
    for base, values in strata.items():

        def pattern(item):
            dims = item["label"]["dimensions"]
            return (dims["P"]["status"], dims["S"]["status"], dims["S"]["rule_id"])

        frequencies = Counter(pattern(item) for item in values)
        for item in values:
            rarity = "RARE" if frequencies[pattern(item)] / len(values) <= 0.1 else "COMMON"
            item["stratum"] = (*base, rarity)
            refined[item["stratum"]].append(item)
    strata = refined
    for values in strata.values():
        rng.shuffle(values)
    selected, used, per_stratum = [], set(), Counter()
    keys = sorted(strata)
    rng.shuffle(keys)
    while len(selected) < representative_n:
        added = False
        for key in keys:
            offset = per_stratum[key]
            if offset >= len(strata[key]) or len(selected) == representative_n:
                continue
            item = strata[key][offset]
            selected.append(("REPRESENTATIVE", item))
            used.add(item["label"]["row_id"])
            per_stratum[key] += 1
            added = True
        if not added:
            break
    positive_remaining = {
        world: [x for x in values if x["label"]["row_id"] not in used]
        for world, values in positive.items()
    }
    worlds = [world for world, values in positive_remaining.items() if values]
    rng.shuffle(worlds)
    chosen_worlds = worlds[:positive_n]
    for world in chosen_worlds:
        item = rng.choice(positive_remaining[world])
        selected.append(("POSITIVE_CREDIT", item))
        used.add(item["label"]["row_id"])
    rng.shuffle(selected)
    blind, private = [], []
    for index, (cohort, item) in enumerate(selected):
        task, record, label = item["row"]["task"], item["row"]["record"], item["label"]
        audit_id = f"AUDIT_{index:03d}"
        # Whitelist fields: never copy a row wholesale into an annotator file.
        blind.append(
            dict(
                audit_id=audit_id,
                image=task.get("image_path", task.get("chart")),
                world=task["world"],
                query=task["query"],
                question=task.get("question", task.get("prompt")),
                response=raw_text(record),
                gold_program=task["query"],
                second_review_required=index < min(40, len(selected)),
            )
        )
        if cohort == "REPRESENTATIVE":
            probability = per_stratum[item["stratum"]] / len(strata[item["stratum"]])
        else:
            probability = (
                len(chosen_worlds) / len(worlds) / len(positive_remaining[label["world_id"]])
            )
        private.append(
            dict(
                audit_id=audit_id,
                cohort=cohort,
                row_id=label["row_id"],
                world_id=label["world_id"],
                stratum=list(item["stratum"]),
                conditional_selection_probability=probability,
                positive_credit=item["positive"],
                automatic_dimensions=label["dimensions"],
                verifier_version=VERIFIER_VERSION,
            )
        )
    output_dir = Path(output_dir)
    _jsonl(output_dir / "BLINDED_RECORDS.jsonl", blind)
    _jsonl(output_dir / "private" / "SAMPLE_KEY.jsonl", private)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "audit_id",
        "reviewer_id",
        "source_type",
        "reviewed_at",
        "P",
        "S",
        "O",
        "E",
        "evidence_note",
        "second_reviewer_id",
        "second_P",
        "second_S",
        "adjudicated_P",
        "adjudicated_S",
    ]
    with (output_dir / "HUMAN_REVIEW.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({"audit_id": row["audit_id"]} for row in blind)
    receipt = dict(
        status="AWAITING_HUMAN_REVIEW",
        verifier_version=VERIFIER_VERSION,
        seed=seed,
        representative_sampled=sum(c == "REPRESENTATIVE" for c, _ in selected),
        positive_credit_worlds_sampled=len(chosen_worlds),
        total_sampled=len(selected),
        reserved_supplement_max=100,
        second_review_requested=min(40, len(selected)),
        human_audit_complete=False,
        automatic_labels_are_human=False,
        rarity_definition=(
            "P/S/S-rule pattern frequency <=0.10 within family/block/J/protocol stratum"
        ),
        probability_note=(
            "Representative n/N within fixed strata; enriched probabilities conditional "
            "on representative exclusions."
        ),
    )
    _json(output_dir / "AUDIT_PREPARATION.json", receipt)
    return receipt


def assess_human_audit(review_path, key_path):
    """Summarize genuine provided reviewer records; missing templates stay blocked."""
    empty = dict(human_audit_complete=False, reason="MISSING_HUMAN_REVIEW", audit={})
    if not Path(review_path).is_file() or not Path(key_path).is_file():
        return empty
    private = [json.loads(line) for line in Path(key_path).read_text().splitlines() if line.strip()]
    keys = {row["audit_id"]: row for row in private}
    if len(keys) != len(private):
        raise ValueError("Duplicate private audit IDs")
    with Path(review_path).open(newline="") as stream:
        reviewed = list(csv.DictReader(stream))
    if len({row["audit_id"] for row in reviewed}) != len(reviewed):
        raise ValueError("Duplicate human audit IDs")
    valid, worlds, false_rules = [], {"P": defaultdict(list), "S": defaultdict(list)}, []
    for row in reviewed:
        if row["audit_id"] not in keys:
            raise ValueError("Unknown audit ID")
        if not (
            row.get("source_type") == "HUMAN"
            and row.get("reviewer_id", "").strip()
            and row.get("reviewed_at", "").strip()
        ):
            continue
        if any(row.get(d) not in ("PASS", "FAIL", "UNKNOWN") for d in "PSOE"):
            continue
        key = keys[row["audit_id"]]
        if key["verifier_version"] != VERIFIER_VERSION:
            raise ValueError("Human labels refer to a different frozen parser")
        valid.append(row)
        for dim in ("P", "S"):
            human = row.get("adjudicated_" + dim) or row[dim]
            if human not in ("PASS", "FAIL", "UNKNOWN"):
                raise ValueError("Invalid adjudicated human label")
            auto = key["automatic_dimensions"][dim]["status"]
            if auto != "UNKNOWN" and human != "UNKNOWN" and auto != human:
                false_rules.append(dict(audit_id=row["audit_id"], dimension=dim))
            if key["positive_credit"] and auto != "UNKNOWN":
                # Unknown human judgments do not count as confirmed correct.
                worlds[dim][key["world_id"]].append(human != "UNKNOWN" and human == auto)
    audit = {}
    for name, dim in (("perception", "P"), ("selection", "S")):
        n = len(worlds[dim])
        correct = sum(all(values) for values in worlds[dim].values())
        audit[name] = dict(
            reward_positive_n=n,
            reward_positive_correct=correct,
            wilson_lower=wilson_lower(correct, n) if n else None,
        )
    second = sum(
        bool(row.get("second_reviewer_id"))
        and row.get("second_reviewer_id") != row.get("reviewer_id")
        and row.get("second_P") in ("PASS", "FAIL", "UNKNOWN")
        and row.get("second_S") in ("PASS", "FAIL", "UNKNOWN")
        for row in valid
    )
    complete = len(valid) >= 200 and len(valid) == len(private) and not false_rules
    return dict(
        human_audit_complete=complete,
        reason="HUMAN_REVIEW_AVAILABLE" if complete else "MEASUREMENT_NOT_READY",
        audit=audit,
        reviewed_rows=len(valid),
        expected_rows=len(private),
        second_reviewed=second,
        review_mode="MULTI_REVIEW" if second >= 40 else "SINGLE_OR_INSUFFICIENT_SECOND_REVIEW",
        rule_mismatches=false_rules,
        annotation_source="PROVIDED_HUMAN_RECORDS",
    )


def write_c_gate(output_path, prerequisites, replay_summary=None, human_audit=None):
    report = dict(prerequisites)
    report.update(
        signal=(replay_summary or {}).get("signal", {}),
        signal_by_block=(replay_summary or {}).get("signal_by_block", {}),
    )
    report.update(
        human_audit or dict(human_audit_complete=False, audit={}, reason="MISSING_HUMAN_REVIEW")
    )
    report.update(c_gate(report))
    if report.get("registered_probe_coverage_complete") is False:
        report["failed_requirements"].append("registered_probe_coverage_complete")
        report["decision"] = "BLOCK_C"
    report["verifier_version"] = VERIFIER_VERSION
    _json(output_path, report)
    return report


def _read(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def _small_record(record, **identity):
    keys = (
        "qid",
        "row_id",
        "block",
        "model_id",
        "repeat",
        "step",
        "group_id",
        "draw",
        "generation_status",
        "truncated",
        "finish_reason",
        "protocol",
        "protocol_revision",
    )
    result = {key: record[key] for key in keys if key in record}
    result.update(full_text=raw_text(record), **identity)
    return result


def _check_label(label, record, score):
    if label["verifier_version"] != VERIFIER_VERSION or label["native_score"] != score:
        raise ValueError("Semantic label version or native score differs")
    for dimension in label["dimensions"].values():
        if dimension["status"] not in ("PASS", "FAIL", "UNKNOWN"):
            raise ValueError("Unregistered semantic label")
        if dimension["status"] != "UNKNOWN":
            if not dimension["rule_id"] or not dimension["source_spans"]:
                raise ValueError("A known label lacks its source evidence")
            if any(not 0 <= a < b <= len(raw_text(record)) for a, b in dimension["source_spans"]):
                raise ValueError("Semantic source span is outside its response")


def replay_stage(root):
    """CPU continuation after all nine parent probe tasks, including old RL B3.

    Reads data records once, not checkpoints or weight hashes. Old artifacts stay
    read-only. Missing historical groups are explicit technical gaps, never a
    reduced denominator or an invented successful gate.
    """
    from . import task_contract
    from .common import rows as read_rows
    from .common import write
    from .training import native_advantages

    root = Path(root)
    config = _read(root / "CONFIG.json")
    tasks = {task["qid"]: task for task in read_rows(root / "DATA/TASKS.jsonl")}
    old_root = Path(config["old_root"])
    all_groups, gaps, golden_differences = [], [], []
    golden_count, golden_types, raw_binary = 0, Counter(), True
    max_advantage_diff, max_reward_diff = 0.0, 0.0
    sources = ("B0", "B2_DEV", "C_TRAIN_PROBE")
    expected_answers = {"B0": 1280, "B2_DEV": 1600, "C_TRAIN_PROBE": 384}
    probe_counts = {}
    for source in sources:
        for block in range(3):
            directory = root / "PROBES" / source / f"b{block}" / "groups"
            answer_count = 0
            for path in sorted(directory.glob("*.json")):
                saved = _read(path)
                records = saved["rows"]
                if not records:
                    raise ValueError(f"Empty probe group: {path}")
                qid = records[0]["qid"]
                task = tasks[qid]
                if "CONFIRM" in task.get("split", ""):
                    raise ValueError("Confirmation task found in development replay")
                joined = []
                for draw, record in enumerate(records):
                    if record["qid"] != qid:
                        raise ValueError("Mixed-qid probe group")
                    compact = _small_record(
                        record, block=f"b{block}", step=0, group_id=saved["group_id"], draw=draw
                    )
                    label = record.get("semantic")
                    if label is None:
                        label = verify_semantics(task, compact, record["score"])
                    if label["verifier_version"] != VERIFIER_VERSION:
                        raise ValueError("Probe labels use a different frozen parser")
                    _check_label(label, compact, record["score"])
                    joined.append(
                        dict(task=task, record=compact, score=record["score"], label=label)
                    )
                    raw_binary = raw_binary and record["score"].get("J") in (0, 1)
                all_groups.append(
                    dict(
                        group_id=saved["group_id"],
                        source=source,
                        block=f"b{block}",
                        family=task["query"]["family"],
                        world_id=task.get("world_id", task.get("root_id")),
                        step=0,
                        rows=joined,
                    )
                )
                answer_count += len(records)
            probe_counts[f"{source}/b{block}"] = answer_count
            if answer_count != expected_answers[source]:
                gaps.append(
                    dict(
                        source=source,
                        block=f"b{block}",
                        expected_answers=expected_answers[source],
                        actual_answers=answer_count,
                    )
                )
    historical_count = Counter()
    for block in range(3):
        directory = old_root / "runs" / f"SRF3CE_RL_b{block}"
        metrics_by_step = {}
        for path in sorted((directory / "update_attempts").glob("*.json")):
            saved = _read(path)
            metrics = saved.get("metrics", {})
            if "reward_groups" in metrics:
                metrics_by_step[int(saved["step"])] = metrics["reward_groups"]
        for path in sorted((directory / "rollouts").glob("*")):
            if not re.fullmatch(r"\d{2}-\d{2}\.json(?:\.gz)?", path.name):
                continue
            saved = _read(path)
            identity, records = saved["identity"], saved["records"]
            step, slot, qid = int(identity["step"]), int(identity["slot"]), identity["qid"]
            if not 1 <= step <= 64 or not 0 <= slot < 16 or len(records) != 8:
                raise ValueError(f"Historical RL group outside registered schedule: {path}")
            task = tasks[qid]
            group_id = f"HIST_RL_b{block}_s{step:02d}_g{slot:02d}"
            joined, rewards = [], []
            for draw, record in enumerate(records):
                compact = _small_record(
                    record,
                    row_id=f"{group_id}/d{draw}",
                    model_id=f"SRF3CE_RL_b{block}",
                    block=f"b{block}",
                    repeat=0,
                    step=step,
                    group_id=group_id,
                    draw=draw,
                )
                score = record["score"]
                rescored = task_contract.score(
                    compact["full_text"],
                    task["world"],
                    task["query"],
                    finish_reason=record.get("finish_reason", "balanced"),
                    protocol="structured",
                )
                rewards.append(score["J"])
                raw_binary = raw_binary and score.get("J") in (0, 1)
                reward_diff = abs(float(rescored["J"]) - float(score["J"]))
                max_reward_diff = max(max_reward_diff, reward_diff)
                if reward_diff:
                    golden_differences.append(
                        dict(
                            group_id=group_id,
                            draw=draw,
                            kind="J_RESCORE",
                            saved=score["J"],
                            recomputed=rescored["J"],
                        )
                    )
                label = verify_semantics(task, compact, score)
                _check_label(label, compact, score)
                joined.append(dict(task=task, record=compact, score=score, label=label))
            expected = metrics_by_step.get(step)
            if expected is None or len(expected) != 16:
                gaps.append(
                    dict(
                        source="HISTORICAL_RL",
                        block=f"b{block}",
                        step=step,
                        kind="MISSING_NATIVE_UPDATE_METRICS",
                    )
                )
            else:
                native_group = expected[slot]
                advantages = native_advantages(rewards)
                if native_group["qid"] != qid or list(native_group["J"]) != rewards:
                    golden_differences.append(
                        dict(group_id=group_id, kind="SAVED_SAMPLE_ORDER_OR_J")
                    )
                differences = [
                    abs(a - float(b))
                    for a, b in zip(advantages, native_group["advantages"], strict=False)
                ]
                maximum = max(differences, default=float("inf"))
                max_advantage_diff = max(max_advantage_diff, maximum)
                if len(native_group["advantages"]) != 8 or maximum > 1e-6:
                    golden_differences.append(
                        dict(
                            group_id=group_id,
                            kind="NATIVE_ADVANTAGE",
                            maximum_absolute_difference=maximum,
                        )
                    )
                golden_count += 1
                kind = "ALL_ZERO" if not any(rewards) else "ALL_ONE" if all(rewards) else "MIXED"
                golden_types[(f"b{block}", task["query"]["family"], kind)] += 1
            all_groups.append(
                dict(
                    group_id=group_id,
                    source="HISTORICAL_RL",
                    block=f"b{block}",
                    family=task["query"]["family"],
                    world_id=task.get("world_id", task.get("root_id")),
                    step=step,
                    rows=joined,
                )
            )
            historical_count[f"b{block}"] += 1
        if historical_count[f"b{block}"] != 1024:
            gaps.append(
                dict(
                    source="HISTORICAL_RL",
                    block=f"b{block}",
                    expected_groups=1024,
                    actual_groups=historical_count[f"b{block}"],
                )
            )
    summary = replay_groups(all_groups, native_advantages, root / "REPLAY")
    golden = dict(
        status="PASS"
        if not golden_differences and golden_count == 3072
        else "INCOMPLETE_OR_MISMATCH",
        compared_groups=golden_count,
        expected_groups=3072,
        max_J_difference=max_reward_diff,
        max_advantage_absolute_difference=max_advantage_diff,
        differences=golden_differences,
        group_types=[
            dict(block=k[0], family=k[1], group_type=k[2], count=v)
            for k, v in sorted(golden_types.items())
        ],
        unavailable_types=[
            dict(block=b, family=f, group_type=t)
            for b in ("b0", "b1", "b2")
            for f in ("CROSS", "INTERVAL", "THRESHOLD", "TOPK")
            for t in ("ALL_ZERO", "MIXED", "ALL_ONE")
            if (b, f, t) not in golden_types
        ],
        scope="Saved native J, sample order and advantage; no model/weight hash sweep",
    )
    write(root / "REPLAY/HISTORICAL_NATIVE_COMPARISON.json", golden)
    write(
        root / "VERIFIER/RULES.json",
        dict(
            verifier_version=VERIFIER_VERSION,
            source="src/sr_f4/verification.py",
            measurement_scope="Observable response claims only",
            labels_location="REPLAY/GROUP_REPLAY.jsonl: each group.labels includes every answer",
            source_spans=(
                "Unicode codepoint offsets in raw response; work spans cite serialized work value"
            ),
            human_labels_location="VERIFIER/audit/HUMAN_REVIEW.csv",
            ai_labels_are_human=False,
            unknown_policy="Unsupported, conflicting, ambiguous and unfinished claims abstain",
        ),
    )
    write(
        root / "VERIFIER/COVERAGE.json",
        dict(
            verifier_version=VERIFIER_VERSION,
            coverage=summary["coverage"],
            denominator="All observations, including UNKNOWN and invalid responses",
        ),
    )
    audit_directory = root / "VERIFIER/audit"
    if not (audit_directory / "AUDIT_PREPARATION.json").exists():
        audit = prepare_blind_audit(all_groups, audit_directory)
    else:
        audit = _read(audit_directory / "AUDIT_PREPARATION.json")
        if audit["verifier_version"] != VERIFIER_VERSION:
            raise ValueError(
                "Existing blind audit belongs to another parser; do not overwrite human work"
            )
    human = assess_human_audit(
        audit_directory / "HUMAN_REVIEW.csv", audit_directory / "private/SAMPLE_KEY.jsonl"
    )
    data_ready = _read(root / "DATA/DATA_READY.json")
    prerequisites = dict(config.get("c_gate_prerequisites", {}))
    prerequisites.update(
        native_advantage_parity=golden["status"] == "PASS",
        binary_J_verified=raw_binary,
        label_lineage_verified=True,
        heldout_isolation_verified=data_ready.get("status") == "DATA_READY",
        registered_probe_coverage_complete=not gaps,
    )
    gate = write_c_gate(root / "C_GATE_REPORT.json", prerequisites, summary, human)
    result = dict(
        status="COMPLETED" if not gaps else "INCOMPLETE_TECHNICAL",
        probe_answers=probe_counts,
        historical_groups=dict(historical_count),
        golden=golden["status"],
        replay=summary,
        audit=audit,
        c_gate=gate["decision"],
        failed_requirements=gate["failed_requirements"],
        technical_gaps=gaps,
    )
    write(root / "REPLAY/STAGE_SUMMARY.json", result)
    write(
        root / "FREEZE_C_F1.json",
        dict(
            status=gate["decision"],
            verifier_version=VERIFIER_VERSION,
            gate_report="C_GATE_REPORT.json",
            c_paths="SKIPPED_BY_PREREG_GATE"
            if gate["decision"] == "BLOCK_C"
            else "REGISTERED_24_PATHS",
            human_review=human["reason"],
            training_started=False,
        ),
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--native-advantage", required=True, help="Audited module:function; no reference fallback"
    )
    parser.add_argument("--prepare-audit", action="store_true")
    args = parser.parse_args()
    module, function = args.native_advantage.split(":", 1)
    native_advantage = getattr(importlib.import_module(module), function)
    groups = [json.loads(line) for line in args.groups.read_text().splitlines() if line.strip()]
    summary = replay_groups(groups, native_advantage, args.output)
    if args.prepare_audit:
        prepare_blind_audit(groups, args.output / "audit")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
