"""Frozen, deterministic DEV_CAL review sampling; never an independent review.

Package docs 03 section 3.6 defines uniqueness inside each parent/factor cell as
(qid, exact full text). The same scope applies to the root cap of four: a global
cap would allow only 48 * 4 = 192 answers, contradicting the required 360.
Explicit provenance aliases share an origin; matching input/seed/text alone
does not make different qids aliases. The sampler consumes existing labels for
stratification, never uses an oracle or extracts text using a gold answer.

Each of 36 parent/factor/family blocks has ten slots. In the fixed TARGETS order,
cover each present rare/unknown/invalid target once if feasible. Then repeatedly
choose the stratum with fewest selected, least remaining supply, and lexical
(A,J,S,B) tie-break; order candidates by the registered candidate hash domain.
Check residual capacity before accepting a candidate, so local greedy choices
cannot prevent a feasible 360. Unrepresented strata and rare-case shortfalls
are reported, never silently filled with copies. Separate blind-order hashing
is independent of selection order and labels. No output signs a review gate.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

from srf6_ref.core import FAMILIES

VERSION = "srf6-blind-review-v1.1"
REVIEW_SEED = 2026102701
PARENTS = ("b0", "b1", "b2")
FACTORS = ("P0", "P1")
PER_FAMILY = 10
ROOT_CAP = 4
TOTAL = 360
PROTOCOL = "structured_v4-srf6-1.0"
DOMAINS = {
    "candidate": "SRF6_REVIEW_V1_CANDIDATE",
    "representative": "SRF6_REVIEW_V1_REPRESENTATIVE",
    "blind_order": "SRF6_REVIEW_V1_BLIND_ORDER",
}
RARE = ("J1_S0", "J1_B0", "S1_B0", "S0_B1")
TARGETS = (*RARE, "S_UNKNOWN", "B_UNKNOWN", "V_INVALID", "UNFINISHED")
LABELS = ("A", "J", "S", "B", "V", "finish")
BLOCKS = tuple((p, f, family) for p in PARENTS for f in FACTORS for family in FAMILIES)


def _digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _rank(seed, domain, value):
    return _digest([DOMAINS[domain], seed, value])


def _state(value):
    return "UNKNOWN" if value is None else ("PASS" if value == 1 else "FAIL")


def _targets(labels):
    a, b, j = labels["S"], labels["B"], labels["J"]
    conditions = (
        j == 1 and a == 0,
        j == 1 and b == 0,
        a == 1 and b == 0,
        a == 0 and b == 1,
        a is None,
        b is None,
        labels["V"] == 0,
        labels["finish"] == 0,
    )
    return tuple(name for name, active in zip(TARGETS, conditions, strict=True) if active)


def _labels(row):
    score = row.get("score", {})
    if not isinstance(score, dict):
        raise ValueError("score must be a label mapping")
    result = {}
    for key in LABELS:
        values = [obj[key] for obj in (row, score) if key in obj]
        if key == "finish":
            values.extend(obj["finished"] for obj in (row, score) if "finished" in obj)
        if any(
            not ((key in ("S", "B") and value is None) or (type(value) is int and value in (0, 1)))
            for value in values
        ):
            raise ValueError(f"Invalid {key} label")
        if not values or any(value != values[0] for value in values):
            raise ValueError(f"Missing or inconsistent {key} label")
        result[key] = values[0]
    return result


def _alias(row):
    provenance = row.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a mapping")
    references = [
        obj[key]
        for obj, keys in (
            (row, ("source_row_id", "cached_from_row_id")),
            (provenance, ("source_row_id", "cached_from_row_id", "alias_of")),
        )
        for key in keys
        if key in obj
    ]
    if any(not isinstance(ref, str) or not ref for ref in references):
        raise ValueError("Explicit source-row aliases must be nonempty strings")
    references = {ref for ref in references if ref != row["row_id"]}
    if len(references) > 1:
        raise ValueError("Conflicting explicit provenance aliases")
    return next(iter(references), None)


def _raw_evidence(row, raw_by_row):
    """Bind the unmodified sampler row; annotated finish_reason is not raw."""
    raw = raw_by_row.get(row["row_id"])
    if not isinstance(raw, dict):
        raise ValueError("Original raw sampling record required for every review row")
    if raw.get("qid") != row["qid"] or raw.get("row_id", row["row_id"]) != row["row_id"]:
        raise PermissionError("Raw sampling qid/row identity differs")
    if (
        type(row.get("seed")) is not int
        or type(raw.get("seed")) is not int
        or raw["seed"] != row["seed"]
    ):
        raise PermissionError("Raw sampling seed differs")
    for source in (raw, row):
        tokens = source.get("tokens")
        if (
            not isinstance(tokens, list)
            or not 1 <= len(tokens) <= 1536
            or any(type(token) is not int or token < 0 for token in tokens)
        ):
            raise ValueError("Original sampling tokens must be complete integer IDs")
    if raw["tokens"] != row["tokens"] or raw.get("full_text") != row["full_text"]:
        raise PermissionError("Raw sampling tokens/full_text differ")
    if (
        not isinstance(raw.get("model_input"), dict)
        or not isinstance(row.get("model_input"), dict)
        or raw["model_input"] != row["model_input"]
    ):
        raise PermissionError("Raw sampling model input differs")
    reason = raw.get("finish_reason")
    if reason not in ("balanced", "eos", "length"):
        raise ValueError("Original sampling finish reason is required")
    expected_flags = {
        "truncated": reason == "length",
        "balanced_token_count": len(raw["tokens"]) if reason == "balanced" else None,
        "boundary_token_count": len(raw["tokens"]) if reason == "balanced" else None,
    }
    for key, expected in expected_flags.items():
        if key in raw and (raw[key] != expected or type(raw[key]) is not type(expected)):
            raise PermissionError("Raw sampling finish reason contradicts native stop flags")
        if key in row and key in raw and row[key] != raw[key]:
            raise PermissionError("Annotated native stop flags differ from raw sampling")
    evidence = {
        key: raw[key]
        for key in ("qid", "seed", "tokens", "full_text", "model_input", "finish_reason")
    }
    return reason, _digest(evidence)


def _pool(records, seed, raw_by_row):
    by_id = {}
    for source in records:
        row = dict(source)
        parent, factor, family = (row.get(key) for key in ("parent", "factor", "family"))
        if (parent, factor, family) not in BLOCKS or row.get("phase") != "PARENT_CENSUS":
            raise PermissionError("Review may only sample registered DEV_CAL census cells")
        root, qid, draw = row.get("root_id"), row.get("qid"), row.get("draw")
        roots = {f"DEV_CAL_{family}_{index:04d}" for index in range(8)}
        if root not in roots or qid not in {f"{root}_{factor}_v{v}" for v in range(4)}:
            raise PermissionError("Review question is outside registered DEV_CAL")
        if type(draw) is not int or draw not in range(8):
            raise ValueError("Registered census draw must be 0 through 7")
        row_id = row.get("row_id")
        if row_id != f"{parent}/{qid}/{draw}" or row_id in by_id:
            raise ValueError("Invalid or duplicate original row identity")
        text = row.get("full_text")
        if not isinstance(text, str) or row.get("raw_text", text) != text:
            raise ValueError("Original full_text must be present and unmodified")
        if row.get("finish_reason") not in ("balanced", "eos", "length"):
            raise ValueError("Annotated program finish reason is required")
        raw_reason, raw_hash = _raw_evidence(row, raw_by_row)
        labels = _labels(row)
        key = _digest([parent, factor, qid, text])
        by_id[row_id] = dict(
            row_id=row_id,
            block=(parent, factor, family),
            root_id=root,
            qid=qid,
            full_text=text,
            finish_reason=raw_reason,
            finish_reason_source="CENSUS_RAW_BY_ROW",
            program_finish_reason=row["finish_reason"],
            raw_sampling_evidence_sha256=raw_hash,
            labels=labels,
            unit_id=key,
            alias=_alias(row),
            rank=_rank(seed, "representative", row_id),
        )
    if set(raw_by_row) != set(by_id):
        raise PermissionError("Raw mapping must match the exact annotated census row set")
    for row in by_id.values():
        visited = {row["row_id"]}
        origin = row["alias"] or row["row_id"]
        while origin in by_id and by_id[origin]["alias"]:
            if origin in visited:
                raise ValueError("Cyclic source-row provenance")
            visited.add(origin)
            if by_id[origin]["full_text"] != row["full_text"]:
                raise ValueError("Explicit aliases disagree on original text")
            origin = by_id[origin]["alias"]
        if origin in by_id and by_id[origin]["full_text"] != row["full_text"]:
            raise ValueError("Explicit aliases disagree on original text")
        row["origin_id"] = origin
    units = {}
    for row in sorted(by_id.values(), key=lambda item: (item["rank"], item["row_id"])):
        key = row["unit_id"]
        if key not in units:
            labels = row["labels"]
            units[key] = dict(
                unit_id=key,
                block=row["block"],
                root_id=row["root_id"],
                qid=row["qid"],
                text=row["full_text"],
                labels=labels,
                stratum=(labels["A"], labels["J"], _state(labels["S"]), _state(labels["B"])),
                targets=_targets(labels),
                rank=_rank(seed, "candidate", key),
                rows=[],
                options={},
            )
        unit = units[key]
        if unit["labels"] != row["labels"]:
            raise ValueError("Identical cell/qid/text has inconsistent labels")
        unit["rows"].append(row)
        unit["options"].setdefault(row["origin_id"], row)
    return units, by_id


class _Flow:
    """Small integral max flow, used only to preserve hard sampling capacities."""

    def __init__(self):
        self.graph = defaultdict(list)

    def edge(self, left, right, capacity):
        forward = [right, capacity, None]
        reverse = [left, 0, forward]
        forward[2] = reverse
        self.graph[left].append(forward)
        self.graph[right].append(reverse)

    def maximum(self):
        total = 0
        while True:
            level, queue = {"start": 0}, deque(["start"])
            while queue:
                node = queue.popleft()
                for target, capacity, _ in self.graph[node]:
                    if capacity and target not in level:
                        level[target] = level[node] + 1
                        queue.append(target)
            if "end" not in level:
                return total
            cursor = defaultdict(int)

            def send(node, limit, *, level=level, cursor=cursor):
                if node == "end":
                    return limit
                edges = self.graph[node]
                while cursor[node] < len(edges):
                    edge = edges[cursor[node]]
                    if edge[1] and level.get(edge[0]) == level[node] + 1:
                        pushed = send(edge[0], min(limit, edge[1]))
                        if pushed:
                            edge[1] -= pushed
                            edge[2][1] += pushed
                            return pushed
                    cursor[node] += 1
                return 0

            while pushed := send("start", TOTAL):
                total += pushed


def _capacity(units, selected):
    counts = Counter(units[key]["block"] for key in selected)
    roots = Counter((units[key]["block"], units[key]["root_id"]) for key in selected)
    used = {row["origin_id"] for row in selected.values()}
    available = []
    origin_counts = Counter()
    for key, unit in units.items():
        block, root = unit["block"], unit["root_id"]
        if key in selected or counts[block] == PER_FAMILY or roots[block, root] == ROOT_CAP:
            continue
        origins = set(unit["options"]) - used
        if origins:
            available.append((key, unit, origins))
            origin_counts.update(origins)
    if not any(count > 1 for count in origin_counts.values()):
        by_root = Counter((unit["block"], unit["root_id"]) for _, unit, _ in available)
        by_block = Counter()
        for (block, root), count in by_root.items():
            by_block[block] += min(ROOT_CAP - roots[block, root], count)
        return sum(min(PER_FAMILY - counts[block], by_block[block]) for block in BLOCKS)
    flow = _Flow()
    for block in BLOCKS:
        flow.edge("start", ("block", block), PER_FAMILY - counts[block])
    added_roots, added_origins = set(), set()
    for key, unit, origins in available:
        block, root = unit["block"], unit["root_id"]
        if (block, root) not in added_roots:
            flow.edge(("block", block), ("root", block, root), ROOT_CAP - roots[block, root])
            added_roots.add((block, root))
        flow.edge(("root", block, root), ("unit", key), 1)
        for origin in sorted(origins):
            flow.edge(("unit", key), ("origin", origin), 1)
            if origin not in added_origins:
                flow.edge(("origin", origin), "end", 1)
                added_origins.add(origin)
    return flow.maximum()


def _select(units):
    selected, reasons = {}, {}
    maximum = _capacity(units, selected)
    by_block = {block: [u for u in units.values() if u["block"] == block] for block in BLOCKS}

    def take(candidates, reason):
        used = {row["origin_id"] for row in selected.values()}
        for unit in sorted(candidates, key=lambda value: (value["rank"], value["unit_id"])):
            key = unit["unit_id"]
            if key in selected:
                continue
            siblings = [u for u in selected if units[u]["block"] == unit["block"]]
            if len(siblings) >= PER_FAMILY:
                continue
            if sum(units[u]["root_id"] == unit["root_id"] for u in siblings) >= ROOT_CAP:
                continue
            for origin, row in unit["options"].items():
                if origin in used:
                    continue
                trial = {**selected, key: row}
                if len(trial) + _capacity(units, trial) >= maximum:
                    selected[key] = row
                    reasons[key] = reason
                    return True
        return False

    for block in BLOCKS:
        pool = by_block[block]
        for target in TARGETS:
            if any(
                target in units[key]["targets"] for key in selected if units[key]["block"] == block
            ):
                continue
            take([u for u in pool if target in u["targets"]], f"coverage:{target}")
        while sum(units[key]["block"] == block for key in selected) < PER_FAMILY:
            selected_strata = Counter(
                units[key]["stratum"] for key in selected if units[key]["block"] == block
            )
            remaining = defaultdict(list)
            for unit in pool:
                if unit["unit_id"] not in selected:
                    remaining[unit["stratum"]].append(unit)
            ordered = sorted(
                remaining,
                key=lambda state: (selected_strata[state], len(remaining[state]), state),
            )
            if not any(take(remaining[state], "stratum_round_robin") for state in ordered):
                break
    if len(selected) != maximum:
        raise RuntimeError("Deterministic selection failed to achieve proven feasible capacity")
    return selected, reasons, maximum


def _coverage(units, selected):
    reports = []
    for block in BLOCKS:
        pool = [u for u in units.values() if u["block"] == block]
        chosen = [u for u in pool if u["unit_id"] in selected]
        strata = []
        for state in sorted({u["stratum"] for u in pool}):
            available = [u for u in pool if u["stratum"] == state]
            count = sum(u["stratum"] == state for u in chosen)
            strata.append(
                dict(
                    stratum=list(state),
                    raw_count=sum(len(u["rows"]) for u in available),
                    unique_count=len(available),
                    selected_count=count,
                    coverage_target=1,
                    coverage_shortfall=int(count == 0),
                )
            )
        targets = {}
        for target in TARGETS:
            available = [u for u in pool if target in u["targets"]]
            count = sum(target in u["targets"] for u in chosen)
            targets[target] = dict(
                raw_count=sum(len(u["rows"]) for u in available),
                unique_count=len(available),
                selected_count=count,
                coverage_target=1,
                coverage_shortfall=int(count == 0),
                unreviewed_unique_count=len(available) - count,
                shortfall_reason=(
                    "NOT_OBSERVED"
                    if not available
                    else "QUOTA_ROOT_CAP_OR_EXPLICIT_ALIAS_CONSTRAINT"
                    if not count
                    else None
                ),
            )
        reports.append(
            dict(
                parent=block[0],
                factor=block[1],
                family=block[2],
                target=PER_FAMILY,
                raw_count=sum(len(u["rows"]) for u in pool),
                unique_count=len(pool),
                selected_count=len(chosen),
                shortfall=PER_FAMILY - len(chosen),
                strata=strata,
                targets=targets,
                root_counts=dict(sorted(Counter(u["root_id"] for u in chosen).items())),
            )
        )
    return reports


def build_review(records, *, raw_by_row, source_identity, seed=REVIEW_SEED):
    """Return ``(private_ledger, blind_items)``; neither is a completed review.

    ``records`` are original annotated census rows, not oracle/task objects.
    ``raw_by_row`` maps each registered row_id to its original sampling row.
    The caller adds qid from the raw source filename when the sampler did not
    store it; qid, seed, tokens, full_text and model_input must match. Do not
    construct this mapping from annotated rows: their finish_reason may have
    been normalized by the scorer. No scorer reason is used as a fallback.
    ``source_identity`` names externally verified source SHA256 identities.
    Incomplete/duplicate-constrained pools produce an explicit insufficient
    result and only their actual unique selections, never fabricated fillers.
    """
    if type(seed) is not int or seed != REVIEW_SEED:
        raise ValueError("Review seed is frozen at 2026102701; post hoc reselection is prohibited")
    if not isinstance(raw_by_row, dict):
        raise ValueError("An explicit original raw sampling row mapping is required")
    if not isinstance(source_identity, dict) or not source_identity:
        raise ValueError("Explicit source identities required")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
        for name, value in source_identity.items()
    ):
        raise ValueError("Source identities must be named lowercase SHA256 values")
    units, rows = _pool(records, seed, raw_by_row)
    selected, reasons, maximum = _select(units)
    order = sorted(selected, key=lambda key: (_rank(seed, "blind_order", key), key))
    blind, mapping = [], []
    for index, key in enumerate(order):
        row, unit = selected[key], units[key]
        anonymous_id = f"R{index + 1:04d}"
        blind.append(
            dict(
                anonymous_id=anonymous_id,
                raw_text=row["full_text"],
                protocol_version=PROTOCOL,
                generation_finish_reason=row["finish_reason"],
            )
        )
        mapping.append(
            dict(
                anonymous_id=anonymous_id,
                row_id=row["row_id"],
                origin_id=row["origin_id"],
                unit_id=key,
                parent=unit["block"][0],
                factor=unit["block"][1],
                family=unit["block"][2],
                root_id=unit["root_id"],
                qid=unit["qid"],
                raw_text_sha256=hashlib.sha256(row["full_text"].encode("utf-8")).hexdigest(),
                program_labels=dict(unit["labels"]),
                program_finish_reason=row["program_finish_reason"],
                generation_finish_reason=row["finish_reason"],
                generation_finish_reason_source=row["finish_reason_source"],
                raw_sampling_evidence_sha256=row["raw_sampling_evidence_sha256"],
                stratum=list(unit["stratum"]),
                selection_reason=reasons[key],
                source_row_ids=[r["row_id"] for r in unit["rows"]],
            )
        )
    rare_cases = []
    for key, unit in sorted(units.items()):
        patterns = [target for target in RARE if target in unit["targets"]]
        if patterns:
            rare_cases.append(
                dict(
                    unit_id=key,
                    parent=unit["block"][0],
                    factor=unit["block"][1],
                    family=unit["block"][2],
                    root_id=unit["root_id"],
                    qid=unit["qid"],
                    patterns=patterns,
                    program_labels=dict(unit["labels"]),
                    source_row_ids=[r["row_id"] for r in unit["rows"]],
                    selected=key in selected,
                )
            )
    ledger = dict(
        version=VERSION,
        status="BLIND_PACKAGE_ONLY",
        review_status="REVIEW_NOT_RUN",
        sampling_status="COMPLETE_360" if maximum == TOTAL else "INSUFFICIENT_UNIQUE_RESPONSES",
        reviewer_type=None,
        execution_origin="OFFLINE_SAMPLING_ONLY",
        seed=seed,
        generation_finish_reason_source="CENSUS_RAW_BY_ROW",
        seed_domains=dict(DOMAINS),
        source_identity=dict(source_identity),
        selection_inputs_sha256=_digest([rows[key] for key in sorted(rows)]),
        blind_items_sha256=_digest(blind),
        raw_count=len(rows),
        unique_units=len(units),
        selected_count=len(selected),
        required_count=TOTAL,
        shortfall=TOTAL - len(selected),
        root_cap=ROOT_CAP,
        root_cap_scope="parent_factor",
        uniqueness_scope="parent_factor_qid_exact_full_text",
        alias_policy="explicit_source_row_provenance_only",
        coverage_priority=list(TARGETS),
        per_family=PER_FAMILY,
        stratum_tie_break="selected_count, remaining_unique_supply, lexical_A_J_S_B",
        candidate_tie_break=(
            "candidate_domain_sha256, unit_id; representative_domain_sha256, row_id"
        ),
        unmet_coverage_policy="report; never duplicate, relax caps, or alter source labels",
        interpretation="Coverage sample, not a population rate or independent correctness review",
        selected=mapping,
        blocks=_coverage(units, selected),
        rare_cases=rare_cases,
    )
    return ledger, blind


def write_review(records, output_dir, *, raw_by_row, source_identity, seed=REVIEW_SEED):
    """Create exclusive private/selection.json and blind/EXTRACTION.jsonl files.

    Only the blind directory may be provided to a blinded extractor. Preserve
    the private ledger until extraction is complete. Future text-model reviews
    must identify MODEL_TEXT, never HUMAN. The caller owns source verification
    and registration of this independent review module; no runtime is changed.
    """
    ledger, blind = build_review(
        records, raw_by_row=raw_by_row, seed=seed, source_identity=source_identity
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    private, public = output / "private", output / "blind"
    private.mkdir(mode=0o700)
    public.mkdir()
    selection_path, blind_path = private / "selection.json", public / "EXTRACTION.jsonl"
    with selection_path.open("x", encoding="utf-8") as stream:
        selection_path.chmod(0o600)
        json.dump(ledger, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with blind_path.open("x", encoding="utf-8") as stream:
        for row in blind:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return dict(
        private_ledger=str(selection_path),
        blind_package=str(blind_path),
        status=ledger["status"],
        review_status=ledger["review_status"],
        sampling_status=ledger["sampling_status"],
        selected_count=ledger["selected_count"],
    )
