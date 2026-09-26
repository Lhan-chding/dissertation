"""Reproduce descriptive tables and figures from the fixed partial-12 export."""

import csv
import itertools
import json
import math
import os
import statistics
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ssvc-partial12-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "analysis_input.json").read_text())
ORIGINS = ["61001_t32", "61001_t96", "61003_t32"]


def write_csv(name, rows):
    with (ROOT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return statistics.mean(values)


def seconds(sacct):
    h, m, s = map(int, sacct.splitlines()[0].split("|")[3].split(":"))
    return h * 3600 + m * 60 + s


def prob(prompt):
    return prompt["events"].get("X", 0) / prompt["n"]


def panel_mc_variance(prompts):
    # Same 24 prompts in each of six equally weighted strata.
    return sum(prob(p) * (1 - prob(p)) / (p["n"] - 1) for p in prompts) / len(prompts) ** 2


summaries, strata_rows, diagnostics, pairs, audits = [], [], [], [], []
indexed = {}
for branch in DATA["branches"]:
    origin, recipe = branch["origin"], branch["recipe"]
    ev = branch["evaluations"]["32"]
    s, prompts = ev["summary"], ev["prompts"]
    assert len(prompts) == 144 and sum(p["n"] for p in prompts) == 2304
    assert len({p["base_scene_id"] for p in prompts}) == 72
    assert abs(mean(prob(p) for p in prompts) - s["J"]) < 1e-12
    se = math.sqrt(panel_mc_variance(prompts))
    assert abs(se - s["fixed_panel_mc_se"]) < 1e-12
    indexed[(origin, recipe)] = branch
    strata = s["strata"]
    error_total = sum(v["n"] * (1 - v["pX"]) for v in strata.values())
    hard_errors = sum(
        v["n"] * (1 - v["pX"])
        for k, v in strata.items()
        if k in ("cross_series/SYMBOLIC_FRESH", "trend/SYMBOLIC_FRESH")
    )
    events = {
        event: sum(p["events"].get(event, 0) for p in prompts) / 2304
        for event in ("X", "S", "W", "I")
    }
    assert abs(sum(events.values()) - 1) < 1e-12
    updates = branch["updates"]
    summary = dict(
        origin=origin,
        recipe=recipe,
        J=s["J"],
        mc_se=se,
        image_pX=s["interfaces"]["IMAGE_CUE_FRESH"],
        symbolic_pX=s["interfaces"]["SYMBOLIC_FRESH"],
        pS=events["S"],
        pW=events["W"],
        pI=events["I"],
        coord_accuracy=mean(v["coord_accuracy"] for v in strata.values()),
        damage_probability=mean(
            v["damage_any_given_valid"] * v["valid_n"] / v["n"] for v in strata.values()
        ),
        full_relation_wrong=mean(v["relation_full_wrong"] for v in strata.values()),
        hard_symbolic_error_share=hard_errors / error_total,
        total_hours=seconds(branch["verified"]["sacct"]) / 3600,
        train_hours=branch["verified"]["training_seconds"] / 3600,
        h32_eval_hours=branch["verified"]["evaluations"]["32"]["sampling_seconds"] / 3600,
        nonzero_advantage_group_fraction=sum(u["effective_groups"] for u in updates)
        / sum(u["groups"] for u in updates),
        zero_gradient_steps=sum(u["grad_norm"] == 0 for u in updates),
    )
    summaries.append(summary)
    for key, v in strata.items():
        strata_rows.append(dict(origin=origin, recipe=recipe, stratum=key, **v))
    h8 = branch["evaluations"]["8"]
    p8 = {p["prompt_id"]: p for p in h8["prompts"]}
    same = [p for p in prompts if p["prompt_id"] in p8]
    assert len(same) == 24
    assert all(p8[p["prompt_id"]]["base_scene_id"] == p["base_scene_id"] for p in same)
    diagnostics.append(
        dict(
            origin=origin,
            recipe=recipe,
            prompts=24,
            H8_J=h8["summary"]["J"],
            H32_same_prompts_J=mean(prob(p) for p in same),
            aligned_change=mean(prob(p) for p in same) - h8["summary"]["J"],
        )
    )

for origin in ORIGINS:
    recipes = sorted(r for o, r in indexed if o == origin)
    for left, right in itertools.combinations(recipes, 2):
        a = indexed[(origin, left)]["evaluations"]["32"]
        b = indexed[(origin, right)]["evaluations"]["32"]
        pa, pb = ({p["prompt_id"]: p for p in e["prompts"]} for e in (a, b))
        assert pa.keys() == pb.keys()
        differences = [prob(pa[k]) - prob(pb[k]) for k in pa]
        delta = mean(differences)
        se = math.sqrt(panel_mc_variance(list(pa.values())) + panel_mc_variance(list(pb.values())))
        pairs.append(
            dict(
                origin=origin,
                left=left,
                right=right,
                delta=delta,
                mc_se_difference=se,
                descriptive_mc_low=delta - 1.96 * se,
                descriptive_mc_high=delta + 1.96 * se,
                prompts_left_higher=sum(d > 0 for d in differences),
                prompts_right_higher=sum(d < 0 for d in differences),
                prompts_tied=sum(d == 0 for d in differences),
            )
        )

for pre in DATA["prestates"]:
    valid = [b for b in pre["repair_buckets"] if b["reward_key"][1] != "I"]
    eligible = [b for b in valid if b["n"] >= 2]
    mixed = [b for b in eligible if len(b["repair_counts"]) > 1]
    pairs_total = sum(b["n"] * (b["n"] - 1) // 2 for b in eligible)
    pairs_different = pairs_total - sum(
        r["n"] * (r["n"] - 1) // 2 for b in eligible for r in b["repair_counts"]
    )
    mixed_supported = [b for b in mixed if b["n"] >= 8]
    audits.append(
        dict(
            origin=pre["origin"],
            snapshot=pre["snapshot"],
            raw_outputs=2304,
            valid_reward_buckets=len(valid),
            repeated_reward_buckets=len(eligible),
            mixed_repair_buckets=len(mixed),
            mixed_with_n_ge_8=len(mixed_supported),
            prompts_with_mixed_repair=len({b["reward_key"][0] for b in mixed}),
            same_reward_pairs=pairs_total,
            different_repair_pairs=pairs_different,
            pair_disagreement=pairs_different / pairs_total if pairs_total else None,
            singleton_reward_buckets=sum(b["n"] == 1 for b in valid),
        )
    )

summaries.sort(key=lambda r: (ORIGINS.index(r["origin"]), -r["J"]))
write_csv("branch_summary.csv", summaries)
write_csv("six_strata.csv", strata_rows)
write_csv("within_origin_comparisons.csv", pairs)
write_csv("aligned_H8_H32.csv", diagnostics)
write_csv("prestate_information_audit.csv", audits)
write_csv(
    "training_updates.csv",
    [
        dict(origin=b["origin"], recipe=b["recipe"], **u)
        for b in DATA["branches"]
        for u in b["updates"]
    ],
)
facts = dict(
    branches=12,
    independent_lineages=2,
    origins=3,
    training_outputs=12288,
    H8_outputs=2304,
    H32_outputs=27648,
    prestate_outputs=13824,
    total_gpu_hours=sum(s["total_hours"] for s in summaries),
    average_branch_hours=mean(s["total_hours"] for s in summaries),
    origins_detail={
        o: dict(
            recipes=[s["recipe"] for s in summaries if s["origin"] == o],
            observed_spread=max(s["J"] for s in summaries if s["origin"] == o)
            - min(s["J"] for s in summaries if s["origin"] == o),
        )
        for o in ORIGINS
    },
    branches_summary=summaries,
    information_audit=audits,
    caution=(
        "Descriptive development snapshot; incomplete unequal action sets; "
        "MC error conditional on fixed endpoints and panel, "
        "not training-seed or unseen-lineage uncertainty. "
        "No multiplicity-adjusted confirmatory test. "
        "Nonzero-advantage group counts are exact floating-point nonzero checks "
        "and not comparable learning-signal measures across GDPO and other algorithms."
    ),
)
(ROOT / "analysis_summary.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2) + "\n")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharex=True)
for ax, origin in zip(axes, ORIGINS, strict=True):
    selected = [s for s in summaries if s["origin"] == origin]
    ax.errorbar(
        [100 * s["J"] for s in selected],
        range(4),
        xerr=[196 * s["mc_se"] for s in selected],
        fmt="o",
        color="#165D8C",
        capsize=4,
    )
    ax.set_yticks(range(4), [s["recipe"] for s in selected])
    ax.invert_yaxis()
    ax.set_title(origin)
    ax.set_xlabel("H32 exact correctness (%)")
    ax.grid(axis="x", alpha=0.25)
fig.suptitle("12 completed development branches — endpoint estimates", fontsize=13)
fig.text(
    0.5,
    0.01,
    "Bars: approximate +/-1.96 fixed-panel Monte Carlo SE only; not across-training uncertainty.",
    ha="center",
    fontsize=8,
)
fig.tight_layout(rect=(0, 0.06, 1, 0.92))
fig.savefig(ROOT / "endpoint_scores.png", dpi=180)
fig.savefig(ROOT / "endpoint_scores.pdf")
plt.close(fig)

keys = sorted(DATA["branches"][0]["evaluations"]["32"]["summary"]["strata"])
matrix = np.array(
    [
        [
            100
            * indexed[(s["origin"], s["recipe"])]["evaluations"]["32"]["summary"]["strata"][k]["pX"]
            for k in keys
        ]
        for s in summaries
    ]
)
fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(matrix, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
ax.set_yticks(range(12), [s["origin"] + " / " + s["recipe"] for s in summaries])
ax.set_xticks(
    range(6),
    [
        k.replace("cross_series", "Cross-\nseries")
        .replace("duplicate_encoding", "Duplicate")
        .replace("trend", "Trend")
        .replace("/IMAGE_CUE_FRESH", "\nImage")
        .replace("/SYMBOLIC_FRESH", "\nSymbolic")
        for k in keys
    ],
)
for i in range(12):
    for j in range(6):
        ax.text(
            j,
            i,
            f"{matrix[i, j]:.1f}",
            ha="center",
            va="center",
            color="white" if matrix[i, j] > 65 else "black",
        )
ax.set_title("H32 exact correctness by family and interface (%)")
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
fig.tight_layout()
fig.savefig(ROOT / "strata_heatmap.png", dpi=180)
fig.savefig(ROOT / "strata_heatmap.pdf")
plt.close(fig)
print(
    json.dumps(
        {k: v for k, v in facts.items() if k not in ("branches_summary", "information_audit")},
        ensure_ascii=False,
        indent=2,
    )
)
