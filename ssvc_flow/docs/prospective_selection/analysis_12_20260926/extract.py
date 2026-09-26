"""Read only the 12 already verified branches; never load a model or submit jobs."""

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

RECEIPTS = [
    "FIRST_BRANCHES_VERIFIED_20260925_0746.json",
    "BRANCH_R4_SAW_VERIFIED_20260925.json",
    "BRANCH_T96_FIRST4_VERIFIED_20260925.json",
    "BRANCH_61003_T32_FIRST4_VERIFIED_20260926.json",
]


def read(path):
    return json.loads(Path(path).read_text())


def evaluation(directory, prompts, draws):
    manifest = read(directory / "MANIFEST.json")
    complete = read(directory / "COMPLETE.json")
    assert complete["status"] == "EVALUATION_COMPLETE"
    assert complete["identity"] == manifest
    commits = sorted((directory / "prompts").glob("*/COMMIT.json"))
    assert len(commits) == prompts
    rows, seen = [], set()
    for path in commits:
        chunk = read(path)
        records = [
            json.loads(line) for line in Path(chunk["samples"]["path"]).read_text().splitlines()
        ]
        assert len(records) == chunk["count"] == draws
        assert len({r["prompt_id"] for r in records}) == 1
        for r in records:
            assert r["sample_id"] not in seen
            seen.add(r["sample_id"])
            for key in (
                "lineage_id",
                "origin_id",
                "policy_id",
                "panel_id",
                "horizon",
                "repeat",
                "role",
            ):
                assert r[key] == manifest[key]
        rows.extend(records)
    assert len(rows) == complete["generated_outputs"] == prompts * draws
    return rows, complete


def prompt_counts(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["prompt_id"]].append(row)
    result = []
    for pid, records in sorted(groups.items()):
        first = records[0]
        events = Counter(r["semantic"]["event"] for r in records)
        result.append(
            dict(
                prompt_id=pid,
                base_scene_id=first["base_scene_id"],
                family=first["family"],
                interface=first["interface"],
                n=len(records),
                events=dict(events),
            )
        )
    return result


def repair_buckets(rows):
    buckets = defaultdict(Counter)
    for r in rows:
        s = r["semantic"]
        key = (r["prompt_id"], s["event"], s["relation_numerator"], s["relation_denominator"])
        repair = ("INVALID",) if s["event"] == "I" else (s["F"], s["B"], s["M"])
        buckets[key][repair] += 1
    return [
        dict(
            reward_key=list(key),
            n=sum(counts.values()),
            repair_counts=[dict(repair=list(k), n=v) for k, v in sorted(counts.items())],
        )
        for key, counts in sorted(buckets.items())
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    branches = []
    for filename in RECEIPTS:
        receipt = read(args.root / filename)
        assert receipt["status"] == "VERIFIED"
        for verified in receipt["branches"]:
            origin, recipe = verified["origin"], verified["recipe"]
            directory = args.root / "campaign/branches" / origin / recipe / "repeat_1"
            evaluated = {}
            for horizon, prompts, draws in ((8, 24, 8), (32, 144, 16)):
                rows, complete = evaluation(
                    directory / f"evaluations/H{horizon:02d}", prompts, draws
                )
                assert (
                    abs(complete["summary"]["J"] - verified["evaluations"][str(horizon)]["J"])
                    < 1e-12
                )
                evaluated[str(horizon)] = dict(
                    summary=complete["summary"], prompts=prompt_counts(rows)
                )
            updates = []
            for path in sorted(directory.glob("segments/*/COMMIT.json")):
                updates.extend(read(binding["path"]) for binding in read(path)["updates"])
            assert sorted(u["step"] for u in updates) == list(range(1, 33))
            small = []
            for u in sorted(updates, key=lambda r: r["step"]):
                advantages = u["reward_statistics"]["advantages"]
                small.append(
                    dict(
                        step=u["step"],
                        loss=u["loss"],
                        grad_norm=u["grad_norm_preclip"],
                        sampling_seconds=u["sampling_seconds"],
                        update_seconds=u["update_seconds"],
                        effective_groups=sum(any(v != 0 for v in g) for g in advantages),
                        groups=len(advantages),
                    )
                )
                assert all(math.isfinite(small[-1][k]) for k in ("loss", "grad_norm"))
            branches.append(
                dict(
                    origin=origin,
                    recipe=recipe,
                    verified=verified,
                    evaluations=evaluated,
                    updates=small,
                )
            )
    assert len(branches) == 12 and len({(b["origin"], b["recipe"]) for b in branches}) == 12
    prestates = []
    for origin in sorted({b["origin"] for b in branches}):
        step = int(origin.split("_t")[1])
        for t in (step - 8, step):
            rows, complete = evaluation(args.root / "campaign/prestate" / origin / f"t{t}", 72, 32)
            prestates.append(
                dict(
                    origin=origin,
                    snapshot=t,
                    summary=complete["summary"],
                    prompts=prompt_counts(rows),
                    repair_buckets=repair_buckets(rows),
                )
            )
    result = dict(
        schema="partial12-descriptive-v1",
        extracted_at=datetime.now(timezone.utc).isoformat(),
        source_root=str(args.root),
        receipt_files=RECEIPTS,
        scope="12 completed development branches only; no GPU, no final test, no selector fitting",
        verification=(
            "Previously verified raw bindings reused; "
            "identity/count/summary checks repeated only for extraction."
        ),
        branches=branches,
        prestates=prestates,
    )
    with args.out.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(branches=len(branches), prestates=len(prestates), out=str(args.out))))


if __name__ == "__main__":
    main()
