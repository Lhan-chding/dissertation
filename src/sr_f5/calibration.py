"""Registered, restartable G2 census. No optimizer and no formal test outputs.

Completed groups are immutable resumable units. An interrupted, unaccounted
generation is a diagnostic stop, never silently resampled with a new seed.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

from sr_f4.common import append, now, read, rows, write, write_rows
from srf5_ref.credit import credits
from srf5_ref.gates import check_calibration
from srf5_ref.generator import FAMILIES, seed_for
from srf5_ref.oracle import evaluate_b

from .measurement import map_token_mask, parse_response, score_response
from .runtime import PROTOCOL_VERSION, load


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def census_inputs(root):
    root = Path(root)
    public = list(rows(root / "DATA/dev/MODEL_INPUTS.jsonl"))
    private = list(rows(root / "DATA/dev/TASKS_PRIVATE.jsonl"))
    if len(public) != 192 or len(private) != 192:
        raise ValueError("G2 requires exactly 192 registered DEV_CAL questions")
    by_qid = {row["qid"]: row for row in private}
    if len(by_qid) != 192 or len({r["qid"] for r in public}) != 192:
        raise ValueError("Duplicate DEV_CAL identity")
    if set(by_qid) != {r["qid"] for r in public}:
        raise ValueError("Public/private DEV_CAL identities differ")
    for row in public:
        task = by_qid[row["qid"]]
        if row["protocol"] != "structured_v3" or row["root_id"] != task["root_id"]:
            raise ValueError("Unexpected DEV_CAL protocol or root")
        if set(row["model_input"]) != {"prompt", "image_file", "image_sha256"}:
            raise PermissionError("Private metadata at model input boundary")
    if Counter(t["cell"] for t in private) != Counter({f: 32 for f in FAMILIES}):
        raise ValueError("Unexpected DEV_CAL family balance")
    return sorted(public, key=lambda r: r["qid"]), by_qid


def group_spec(config, parent, public, task, frozen=None):
    return dict(
        parent=parent,
        qid=public["qid"],
        root_id=public["root_id"],
        cell=task["cell"],
        purpose="G2_parent_census",
        protocol_version=PROTOCOL_VERSION,
        config_sha256=digest(config),
        public_input_sha256=digest(public),
        model_input_sha256=digest(public["model_input"]),
        task_sha256=digest(task),
        frozen_census_identity=digest(frozen) if frozen is not None else None,
        seeds=[seed_for(config["master_seed"], "G2", parent, public["qid"], i) for i in range(8)],
        shuffle_seed=seed_for(config["master_seed"], "G2_shuffle", parent, public["qid"]),
    )


def frozen_census(root, config, public):
    frozen = read(Path(root) / "G2_SPEC_FROZEN.json")
    config_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (
        frozen.get("config_digest") != config_digest
        or set(frozen.get("question_ids", [])) != {r["qid"] for r in public}
        or frozen.get("questions_per_parent") != 192
        or frozen.get("expected_responses") != 4608
        or frozen.get("optimization_allowed") is not False
        or frozen.get("test_outputs_generated") is not False
        or frozen.get("master_seed") != config["master_seed"]
        or frozen.get("parents") != config["parents"]
    ):
        raise PermissionError("Frozen census specification does not match current inputs")
    ready = read(Path(root) / "CPU_READY.json")
    if ready.get("status") != "CPU_READY_FOR_G2_PREFLIGHT":
        raise PermissionError("Server CPU preparation has not completed")
    return frozen


def validate_group(group, spec):
    if group.get("status") != "COMPLETE" or group.get("spec") != spec:
        raise PermissionError("Completed G2 group identity changed")
    records = group.get("records", [])
    if len(records) != 8 or [r["seed"] for r in records] != spec["seeds"]:
        raise ValueError("G2 group does not contain its eight registered draws")
    if len({r["row_id"] for r in records}) != 8:
        raise ValueError("Duplicate G2 draw identity")
    for i, row in enumerate(records):
        if row["qid"] != spec["qid"] or row["row_id"] != f"{spec['parent']}/{spec['qid']}/{i}":
            raise ValueError("G2 draw identity mismatch")
        if row.get("technical_validation_errors") or row.get("generation_status") != "COMPLETE":
            raise ValueError("Technical failure retained in G2 group")
        if len(row["tokens"]) != len(row["sampler_logprobs"]):
            raise ValueError("Sampler probability/token length mismatch")
        if digest(row["model_input"]) != spec["model_input_sha256"]:
            raise ValueError("Generated response used a different model input")
        if (
            row["adapter_identity"].get("parent") != spec["parent"]
            or row["adapter_identity"].get("step") != 0
        ):
            raise ValueError("G2 response did not use its frozen parent")
    expected = credits(records, "LOCAL", spec["shuffle_seed"])
    if expected != group["credit"]:
        raise ValueError("Persisted G2 credit differs from recomputation")
    return records


def annotate_group(raw, task, spec, tokenizer):
    if len(raw) != 8 or [r["seed"] for r in raw] != spec["seeds"]:
        raise ValueError("Cannot complete partial or mismatched raw group")

    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    output = []
    for index, row in enumerate(raw):
        if row.get("technical_validation_errors") or row.get("generation_status") != "COMPLETE":
            raise ValueError("Technical invalid response requires diagnosis")
        label = score_response(
            row["full_text"],
            task["world"],
            task["query"],
            finish_reason=row["finish_reason"],
            protocol="structured_v3",
        )
        mask, reason = map_token_mask(
            row["tokens"],
            row["full_text"],
            label["selected_span"],
            decode,
            prefill=row["assistant_prefill"],
        )
        output.append(
            dict(
                row,
                **label,
                qid=task["qid"],
                root_id=task["root_id"],
                cell=task["cell"],
                row_id=f"{spec['parent']}/{task['qid']}/{index}",
                parent=spec["parent"],
                draw=index,
                mask=mask,
                selected_mask=mask,
                score=label,
                mask_ok=reason == "MAPPED",
                mask_reason=reason,
                seed_semantics="independent_per_draw_torch_generator",
                label_version=PROTOCOL_VERSION,
            )
        )
    return output


def tokenizer_fixtures(tokenizer):
    fixtures = [
        ("ascii", '{"evidence":[],"selected":["Mug","Lamp"],"answer":1}', "KNOWN"),
        ("escaped", '{"evidence":[],"selected":["M\\u0075g"],"answer":1}', "KNOWN"),
        (
            "unicode_work",
            '{"evidence":[],"work":"选择与计算","selected":["Mug"],"answer":1}',
            "KNOWN",
        ),
        ("boundary", '{"evidence":[],"selected":[],"answer":1}', "KNOWN"),
        ("duplicate_key", '{"evidence":[],"selected":[],"selected":[],"answer":1}', "UNKNOWN"),
        ("truncated", '{"evidence":[],"selected":["Mug"', "UNKNOWN"),
        ("missing", '{"evidence":[],"answer":1}', "UNKNOWN"),
        ("multiple", '{"evidence":[],"selected":[],"answer":1}{}', "UNKNOWN"),
        (
            "decoy_work",
            '{"evidence":[],"work":"selected=[Mug]; x=3","selected":["Lamp"],"answer":1}',
            "KNOWN",
        ),
    ]
    evidence = []

    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    for name, text, expected in fixtures:
        token_ids = tokenizer.encode(text[1:], add_special_tokens=False)
        extraction = parse_response(text)
        mask, reason = map_token_mask(
            token_ids, text, extraction["selected_span"], decode, prefill="{"
        )
        if extraction["status"] != expected:
            raise RuntimeError(f"Real tokenizer fixture parser failure: {name}")
        if name == "ascii" and reason != "MAPPED":
            raise RuntimeError("Real ASCII selected span is not mappable")
        if expected == "UNKNOWN" and any(mask):
            raise RuntimeError("Unknown fixture has active tokens")
        evidence.append(
            dict(
                name=name,
                text=text,
                token_ids=token_ids,
                mask=mask,
                reason=reason,
                selected_span=extraction["selected_span"],
                extraction_status=extraction["status"],
            )
        )
    return dict(
        status="PASS",
        kind="real_tokenizer_fixture_no_generation",
        model_calls=0,
        unicode_prefix_instability="abstain_and_report",
        cases=evidence,
    )


def geometry_check(prepared, geometry):
    route = prepared["routing"]
    source, processed = route.get("source_size"), route.get("processed_size")
    if not source or not processed:
        raise ValueError("Missing actual processor dimensions")
    source_metrics = geometry.get("geometry", geometry.get("metrics", geometry))
    scale_x, scale_y = processed[0] / source[0], processed[1] / source[1]
    pixels = source_metrics["pixels_per_value_unit"] * scale_y
    height = source_metrics["minimum_major_label_ink_height_px"] * scale_y
    return dict(
        status="PASS" if pixels >= 3 and height >= 9 else "FAIL",
        reviewer="program",
        source_size=source,
        processed_size=processed,
        processed_pixels_per_value_unit=pixels,
        minimum_processed_label_height_px=height,
        scale_x=scale_x,
        scale_y=scale_y,
        processor_hash=route.get("processor_hash"),
        processed_pixel_sha256=route.get("processed_pixel_sha256"),
        processed_label_boxes=[
            dict(
                text=label["text"],
                bounding_box=[
                    value * (scale_x if index % 2 == 0 else scale_y)
                    for index, value in enumerate(label["bounding_box"])
                ],
            )
            for label in source_metrics.get("labels", [])
        ],
        claim="geometric_readability_only_not_human_semantic_review",
    )


def run(root, parent, shard=0, shards=1):
    root = Path(root).resolve()
    config = read(root / "CONFIG.json")
    if parent not in config["parents"] or type(shards) is not int or not 0 <= shard < shards:
        raise ValueError("Invalid registered parent/shard")
    public, tasks = census_inputs(root)
    frozen = frozen_census(root, config, public)
    selected = public[shard::shards]
    task_id = f"G2-{parent}-{shard}-of-{shards}"
    specs = {r["qid"]: group_spec(config, parent, r, tasks[r["qid"]], frozen) for r in selected}
    group_root = root / "G2/groups" / parent
    pending = []
    for item in selected:
        path = group_root / f"{item['qid']}.json"
        if path.exists():
            validate_group(read(path), specs[item["qid"]])
        else:
            pending.append(item)
    if not pending:
        return dict(status="COMPLETE", task_id=task_id, resumed_groups=len(selected), model_calls=0)
    runtime, _, _ = load(root, parent, task_id, training=False)
    write(
        root / "G2/tokenizer" / f"{task_id}.json", tokenizer_fixtures(runtime.processor.tokenizer)
    )
    geometry = {r["image_file"]: r for r in rows(root / "DATA/RENDER_GEOMETRY.jsonl")}
    for item in pending:
        qid, task = item["qid"], tasks[item["qid"]]
        spec = specs[qid]
        path = group_root / f"{qid}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if path.exists():
                validate_group(read(path), spec)
                continue
            prepared = runtime.prepare_input(item["model_input"], root, protocol=item["protocol"])
            geo = geometry_check(prepared, geometry[item["model_input"]["image_file"]])
            write(root / "G2/readability" / parent / f"{qid}.json", geo)
            if geo["status"] != "PASS":
                raise RuntimeError("Native processor geometric readability gate failed")
            raw_path = root / "G2/raw" / parent / f"{qid}.jsonl"
            attempt_path = root / "G2/attempts" / parent / f"{qid}.json"
            if attempt_path.exists():
                if read(attempt_path)["spec"] != spec:
                    raise PermissionError("Interrupted G2 attempt specification changed")
                raw = list(rows(raw_path)) if raw_path.exists() else []
                if len(raw) != 8:
                    raise RuntimeError(f"Partial G2 attempt {parent}/{qid}; preserve and diagnose")
            else:
                write(attempt_path, dict(status="STARTED", at=now(), spec=spec, task_id=task_id))
                runtime.generate_group(
                    item["model_input"],
                    root,
                    spec["seeds"],
                    protocol=item["protocol"],
                    on_completion=lambda row, target=raw_path: append(target, row),
                )
                raw = list(rows(raw_path))
            records = annotate_group(raw, task, spec, runtime.processor.tokenizer)
            group = dict(
                status="COMPLETE",
                at=now(),
                spec=spec,
                records=records,
                credit=credits(records, "LOCAL", spec["shuffle_seed"]),
                readability=geo,
            )
            validate_group(group, spec)
            write(path, group)
            print(json.dumps(dict(task_id=task_id, qid=qid, status="COMPLETE")), flush=True)
    result = dict(
        status="COMPLETE",
        task_id=task_id,
        groups=len(selected),
        newly_completed_groups=len(pending),
        purpose="G2_parent_census_no_updates",
    )
    write(root / "G2/workers" / f"{task_id}.json", result)
    return result


def independent_review(row, task):
    """Audit raw declarations without calling either production score function.

    Oracle B independently supplies answer/selection truth. Exact scalar grammar
    and certificate sufficiency remain the registered native low-level rules;
    this is not a second independent proof of that certificate implementation.
    """
    from sr_f4 import task_contract as legacy

    errors = []
    text, world, query = row["full_text"], task["world"], task["query"]
    truth = evaluate_b(world, query)

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    decoder = json.JSONDecoder(
        object_pairs_hook=unique_pairs,
        parse_float=Decimal,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")),
    )
    # A balanced object can be structurally complete while invalid JSON. Finish
    # therefore needs a lexical boundary, separately from successful JSON parse.
    boundary, depth, quoted, escaped = None, 0, False, False
    start = len(text) - len(text.lstrip())
    if text[start : start + 1] == "{":
        for index, char in enumerate(text[start:], start):
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    boundary = index + 1
                    break
    prefix, whole, end = None, None, None
    if boundary is not None and boundary <= 64000:
        try:
            candidate, end = decoder.raw_decode(text, start)
            if isinstance(candidate, dict) and end == boundary:
                prefix = candidate
                if not text[end:].strip() and len(text) <= 64000:
                    whole = candidate
        except (ValueError, TypeError, ArithmeticError, RecursionError):
            pass
    known_structure = bool(
        whole is not None
        and set(whole)
        in ({"evidence", "selected", "answer"}, {"evidence", "work", "selected", "answer"})
        and isinstance(whole["evidence"], list)
        and ("work" not in whole or isinstance(whole["work"], str))
        and isinstance(whole["selected"], list)
        and all(type(x) is str for x in whole["selected"])
        and (whole["answer"] is None or type(whole["answer"]) in (str, int, Decimal))
        and all(
            isinstance(fact, dict)
            and set(fact) == {"series", "category", "value"}
            and isinstance(fact["series"], str)
            and isinstance(fact["category"], str)
            for fact in whole["evidence"]
        )
    )
    expected_span, expected_s = None, "UNKNOWN"
    if known_structure:
        selected = [x.strip().casefold() for x in whole["selected"]]
        target = {x.strip().casefold() for x in truth["selected"]}
        expected_s = (
            "PASS" if len(selected) == len(set(selected)) and set(selected) == target else "FAIL"
        )
        # Walk original raw members rather than trusting the stored span or
        # reserializing Decimal literals. The strict decoder already validated it.
        pos = start + 1
        while True:
            while text[pos].isspace():
                pos += 1
            key, pos = decoder.raw_decode(text, pos)
            while text[pos].isspace():
                pos += 1
            pos += 1
            while text[pos].isspace():
                pos += 1
            value_start = pos
            _, pos = decoder.raw_decode(text, pos)
            if key == "selected":
                expected_span = [value_start, pos]
            while text[pos].isspace():
                pos += 1
            if text[pos] == "}":
                break
            pos += 1

    def facts_from(obj):
        if not isinstance(obj, dict) or not isinstance(obj.get("evidence"), list):
            raise ValueError("evidence type")
        if len(obj["evidence"]) > 32:
            raise ValueError("evidence length")
        facts = {}
        for fact in obj["evidence"]:
            if not isinstance(fact, dict) or set(fact) != {"series", "category", "value"}:
                raise ValueError("fact fields")
            series, category = fact["series"], fact["category"]
            if series not in world["series"] or category not in world["categories"]:
                raise ValueError("unknown entity")
            key = series, world["categories"].index(category)
            if key in facts:
                raise ValueError("duplicate fact")
            value = legacy.rational(fact["value"])
            if value.denominator != 1 or not world["domain"][0] <= value <= world["domain"][1]:
                raise ValueError("fact domain")
            facts[key] = value
        return facts

    expected = dict(
        V=int(known_structure),
        S=expected_s,
        P="UNKNOWN",
        value_parseable=0,
        value_correct=0,
        finish=int(row["finish_reason"] in ("balanced", "eos") and boundary is not None),
        certificate_truth=0,
        certificate_sufficient=0,
        certificate=0,
    )
    try:
        answer = None if prefix["answer"] is None else legacy.rational(prefix["answer"])
        expected.update(value_parseable=1, value_correct=int(answer == truth["answer"]))
    except (ValueError, TypeError, KeyError, ArithmeticError, RecursionError):
        pass
    valid_facts = False
    try:
        facts = facts_from(prefix)
        valid_facts = True
        fact_truth = all(v == world["series"][s][i] for (s, i), v in facts.items())
        if (
            "include_end" in query or "selection_series" in query
        ) and not expected["value_parseable"]:
            # The preserved extension scorer returns its legacy diagnostic
            # certificate components before extension recomputation when the
            # answer is unparseable. Preserve that branch; its J remains zero.
            sufficiency = legacy.native.sufficient_template(
                world, legacy.normalized_interval(query), facts
            )
        else:
            sufficiency = legacy.sufficient_template(world, query, facts)
        expected.update(
            certificate_truth=int(fact_truth),
            certificate_sufficient=int(sufficiency),
            certificate=int(fact_truth and sufficiency),
        )
    except (ValueError, TypeError, KeyError, ArithmeticError, RecursionError):
        pass
    try:
        facts = facts_from(whole)
        if facts:
            expected["P"] = (
                "PASS" if all(v == world["series"][s][i] for (s, i), v in facts.items()) else "FAIL"
            )
    except (ValueError, TypeError, KeyError, ArithmeticError, RecursionError):
        pass
    expected["A"] = expected["value_correct"]
    expected["J_M3"] = int(
        expected["V"] and expected["finish"] and expected["A"] and expected["certificate"]
    )
    expected["J"] = expected["J_M3"]
    # Projection retains the old presentation limit, while J_M3 uses the v3
    # structural gate exactly as registered. Do not silently merge the two Js.
    expected["J_legacy_projection"] = (
        int(
            expected["A"]
            and expected["finish"]
            and expected["certificate"]
            and expected["value_parseable"]
            and valid_facts
            and len(whole.get("work", "")) <= 16000
        )
        if known_structure
        else None
    )
    for name, value in expected.items():
        if row.get(name) != value:
            errors.append(
                {
                    "V": "independent_structure_disagreement",
                    "S": "independent_selection_disagreement",
                }.get(name, f"independent_{name}_disagreement")
            )
    if row.get("selected_span") != expected_span:
        errors.append("selected_character_span_disagreement")
    ends = row.get("token_prefix_character_ends", [])
    if len(ends) != len(row["tokens"]) or len(row["mask"]) != len(ends):
        errors.append("token_span_length_mismatch")
    else:
        expected_mask = [False] * len(ends)
        if expected_span and row.get("prefix_character_offsets_stable"):
            starts = [len(row["assistant_prefill"]), *ends[:-1]]
            if (
                not ends
                or ends[-1] != len(text)
                or any(
                    type(b) is not int or not 0 <= a <= b <= len(text)
                    for a, b in zip(starts, ends, strict=True)
                )
            ):
                errors.append("native_prefix_offsets_inconsistent")
            else:
                lo, hi = expected_span
                expected_mask = [
                    b > a and lo <= a and b <= hi for a, b in zip(starts, ends, strict=True)
                ]
        if row["mask"] != expected_mask or row["mask_ok"] != any(expected_mask):
            errors.append("native_prefix_mask_disagreement")
        if "selected_mask" in row and row["selected_mask"] != expected_mask:
            errors.append("selected_mask_alias_disagreement")
    return dict(
        row_id=row["row_id"],
        qid=row["qid"],
        cell=row["cell"],
        parent=row["parent"],
        reviewer="program",
        status="PASS" if not errors else "FAIL",
        errors=errors,
        expected_fields=expected,
        expected_selected_span=expected_span,
        evidence="independent_JSON_oracle_B_answer_selection_raw_fact_truth_and_native_prefix_intervals",
        inherited_rules="native_exact_scalar_grammar_and_registered_certificate_sufficiency",
        certificate_independence="sufficiency_rule_reused_not_independently_proved",
        free_text_reasoning_review="NOT_APPLICABLE_final_selected_only",
    )


def aggregate(root):
    root = Path(root)
    config = read(root / "CONFIG.json")
    public, tasks = census_inputs(root)
    frozen = frozen_census(root, config, public)
    blocks, all_reviews, failures, missing, all_records = {}, [], [], [], []
    for parent in config["parents"]:
        groups = []
        for item in public:
            path = root / "G2/groups" / parent / f"{item['qid']}.json"
            if not path.exists():
                missing.append(f"{parent}/{item['qid']}")
                continue
            group = read(path)
            validate_group(group, group_spec(config, parent, item, tasks[item["qid"]], frozen))
            groups.append(group)
            if group["readability"]["status"] != "PASS":
                failures.append(f"{parent}/{item['qid']}:processor_readability")
        records = [r for g in groups for r in g["records"]]
        all_records.extend(records)
        count = len(records)
        known = [r for r in records if r["S"] in ("PASS", "FAIL")]
        signal = [g for g in groups if g["credit"]["nonzero_group"]]
        family_known = {}
        reviewed_counts = {}
        for family in FAMILIES:
            subset = [r for r in records if r["cell"] == family]
            family_known[family] = (
                sum(r["S"] != "UNKNOWN" for r in subset) / len(subset) if subset else 0
            )
            unique = {}
            for row in sorted(subset, key=lambda r: r["row_id"]):
                unique.setdefault((row["qid"], row["full_text"]), row)
            selected = list(unique.values())[:20]
            reviewed_counts[family] = len(selected)
            for row in selected:
                review = independent_review(row, tasks[row["qid"]])
                all_reviews.append(review)
                failures.extend(f"{row['row_id']}:{e}" for e in review["errors"])
            if len(selected) < 20:
                failures.append(f"{parent}/{family}:fewer_than_20_unique_reviews")
        parseable = sum(r["value_parseable"] for r in records)
        if parseable == 0:
            failures.append(f"{parent}:no_exact_parseable_responses")
        blocks[parent] = dict(
            groups=len(groups),
            responses=count,
            known_fraction=len(known) / count if count else 0,
            minimum_family_known_fraction=min(family_known.values()),
            family_known_fraction=family_known,
            mask_success_given_known=sum(r["mask_ok"] for r in known) / len(known) if known else 0,
            nonzero_groups=len(signal),
            swappable_groups=sum(g["credit"]["swappable_group"] for g in groups),
            changed_groups=sum(g["credit"]["changed_group"] for g in groups),
            families_with_signal=len({g["spec"]["cell"] for g in signal}),
            roots_with_signal=len({g["spec"]["root_id"] for g in signal}),
            truncation_fraction=sum(r["truncated"] for r in records) / count if count else 0,
            exact_parseable_responses=parseable,
            unique_reviewed_by_family=reviewed_counts,
            all_J_zero_groups=sum(g["credit"]["all_J_zero"] for g in groups),
            all_J_one_groups=sum(g["credit"]["all_J_one"] for g in groups),
            mixed_J_groups=sum(
                not g["credit"]["all_J_zero"] and not g["credit"]["all_J_one"] for g in groups
            ),
            state_counts=dict(
                Counter(f"A{r['value_correct']}_J{r['J']}_S{r['S']}" for r in records)
            ),
            P_known_fraction=sum(r["P"] != "UNKNOWN" for r in records) / count if count else 0,
        )
    report = dict(
        actual_response_count=len(all_records),
        unique_reviewed_responses=len(all_reviews),
        unresolved_systematic_measurement_errors=len(failures),
        blocks=blocks,
        missing_groups=missing,
        failures=failures,
        protocol_version=PROTOCOL_VERSION,
        scope="G2 feasibility; no performance selection or formal model outputs",
    )
    gate = check_calibration(report)
    if missing or failures:
        gate["status"] = "NOT_COMPLETE" if missing else "FAIL"
        gate["errors"].extend(failures)
    report["gate"] = gate
    write(root / "G2/METRICS.json", report)
    write_rows(root / "G2/PROGRAM_REVIEWS.jsonl", all_reviews)
    write(
        root / "GATES/G2.json",
        dict(
            gate,
            at=now(),
            evidence="G2/METRICS.json",
            raw_evidence="G2/groups",
            actual_response_count=len(all_records),
        ),
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", choices=("b0", "b1", "b2"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        result = aggregate(args.root)
    elif args.parent is not None:
        result = run(args.root, args.parent, args.shard, args.shards)
    else:
        parser.error("--parent is required without --aggregate")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
