"""A/B endpoint summaries with paired-world uncertainty and explicit C blocking."""

# Chinese report paragraphs and figure captions are intentionally human-readable.
# ruff: noqa: E501, RUF001

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from srf4_ref.sampling import seed_for

from .common import read, rows, write, write_rows
from .evaluation import ARMS, build_a_eval_groups

METRICS = (
    "value_correct",
    "J",
    "presentation",
    "finish",
    "certificate_truth",
    "certificate_sufficient",
    "certificate",
    "historical_code.relaxed_accuracy",
    "historical_code.exact_match",
    "v2.relaxed_accuracy",
    "v2.exact_match",
    "typed_audit.typed_score",
    "official",
)


def _metric(score, key):
    value = score
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _block(value):
    if isinstance(value, str):
        value = value.removeprefix("b")
    value = int(value)
    if value not in (0, 1, 2):
        raise ValueError("Unknown evaluation block")
    return value


def _raw_rows(path):
    if path.suffix == ".jsonl":
        yield from rows(path)
        return
    value = read(path)
    if isinstance(value, dict) and isinstance(value.get("records"), list):
        yield from value["records"]
    elif isinstance(value, dict) and isinstance(value.get("rows"), list):
        yield from value["rows"]


def load_evaluations(root, tasks):
    """Accept raw JSONL or atomic group JSON, never count aggregate score files."""
    root = Path(root)
    registry = {}
    task_file = root / "TASKS.json"
    if task_file.exists():
        value = read(task_file)
        registry = {
            item["task_id"]: item
            for item in (value.get("tasks", []) if isinstance(value, dict) else value)
        }
    result, seen = [], set()
    paths = sorted((root / "EVAL").rglob("*.jsonl")) + sorted((root / "EVAL").rglob("*.json"))
    for path in paths:
        if ".chunk" in path.name:
            continue
        for record in _raw_rows(path):
            if not isinstance(record, dict) or "qid" not in record or "score" not in record:
                continue
            meta = registry.get(record.get("task_id"), {})
            block = _block(record.get("block", meta.get("block")))
            arm = record.get("arm", meta.get("arm"))
            if arm not in ARMS:
                raise ValueError(f"Unexpected evaluation arm {arm!r} in {path}")
            task = tasks.get(record["qid"], {})
            panel = record.get("panel", meta.get("panel", task.get("pool")))
            protocol = record.get("protocol", "structured")
            draw = record.get("draw", record.get("draw_index", record.get("group_row_index")))
            if not isinstance(draw, int):
                raise ValueError("Evaluation row has no integer draw identity")
            identity = (block, arm, panel, protocol, record["qid"], draw)
            if identity in seen:
                raise ValueError(f"Duplicate evaluation answer {identity}")
            seen.add(identity)
            score = record["score"]
            applicable_metrics = (
                METRICS
                if protocol == "structured" or protocol == "chartqa_plain"
                else tuple(
                    key
                    for key in METRICS
                    if key
                    not in ("J", "certificate_truth", "certificate_sufficient", "certificate")
                )
            )
            normalized = dict(
                block=block,
                arm=arm,
                panel=panel,
                protocol=protocol,
                qid=record["qid"],
                draw=draw,
                world_id=task.get("world_id", task.get("root_id", record["qid"])),
                family=task.get("family", "EXTERNAL"),
                variant=task.get("variant"),
                mode=task.get("mode"),
                cell=task.get("cell"),
                purpose=task.get("purpose"),
                pair_id=task.get("pair_id"),
                score={
                    key: _metric(score, key)
                    for key in applicable_metrics
                    if _metric(score, key) is not None
                },
                truncated=bool(record.get("truncated", record.get("finish_reason") == "length")),
                completion_tokens=record.get(
                    "completion_token_count", len(record.get("tokens", []))
                ),
                work_present=score.get("work_present"),
            )
            semantic = record.get("semantic", record.get("semantic_state", record.get("label", {})))
            normalized["semantic_status"] = {
                key: value.get("status") for key, value in semantic.get("dimensions", {}).items()
            }
            result.append(normalized)
    return result


def summarize(items):
    values = list(items)
    result = dict(answers=len(values), worlds=len({row["world_id"] for row in values}))
    for metric in METRICS:
        valid = [
            row["score"][metric]
            for row in values
            if isinstance(row["score"].get(metric), (int, float))
            and math.isfinite(row["score"][metric])
        ]
        if valid:
            result[metric] = dict(mean=sum(valid) / len(valid), n=len(valid), sum=sum(valid))
    if values:
        result.update(
            mean_completion_tokens=sum(row["completion_tokens"] for row in values) / len(values),
            truncated_fraction=sum(row["truncated"] for row in values) / len(values),
        )
    return result


def paired_world_comparison(
    values, panel, arm, reference, *, metric="value_correct", replicates=10000
):
    """Paired world resampling, preserving all prompts/draws and model blocks.

    Stratification is by family on old panels and purpose on B_CONFIRM. Strata
    retain their frozen world-count weights; parent blocks are equally weighted.
    This measures item uncertainty conditional on these three trained blocks.
    """
    if panel == "CHARTQA" and "PARENT" in (arm, reference):
        return dict(
            status="NOT_REGISTERED",
            metric=metric,
            panel=panel,
            arm=arm,
            reference=reference,
            reason="ChartQA is registered only for the two b0 A endpoints; no parent measurement",
        )

    import numpy as np

    blocks = (0,) if panel == "CHARTQA" else (0, 1, 2)
    shared_parent_draws = panel == "B_CONFIRM" and "PARENT" in (arm, reference)
    selected = [
        row
        for row in values
        if row["panel"] == panel
        and row["arm"] in (arm, reference)
        and (not shared_parent_draws or 0 <= row["draw"] < 4)
    ]
    index = defaultdict(dict)
    for row in selected:
        if not isinstance(row["score"].get(metric), (int, float)):
            return dict(
                status="METRIC_UNAVAILABLE",
                metric=metric,
                panel=panel,
                arm=arm,
                reference=reference,
            )
        key = (row["qid"], row["draw"], row["protocol"])
        index[row["block"], row["arm"]][key] = row
    reference_keys = set(index[blocks[0], reference])
    if not reference_keys or any(
        set(index[block, candidate]) != reference_keys
        for block in blocks
        for candidate in (arm, reference)
    ):
        return dict(
            status="INCOMPLETE_PAIRED_COVERAGE",
            metric=metric,
            panel=panel,
            arm=arm,
            reference=reference,
            observed_counts={
                f"b{block}/{candidate}": len(index[block, candidate])
                for block in blocks
                for candidate in (arm, reference)
            },
        )
    by_world = defaultdict(list)
    for key in sorted(reference_keys):
        row = index[blocks[0], reference][key]
        by_world[row["world_id"]].append(key)
    world_ids = sorted(by_world)
    matrix = np.asarray(
        [
            [
                sum(
                    index[block, arm][key]["score"][metric]
                    - index[block, reference][key]["score"][metric]
                    for key in by_world[world]
                )
                / len(by_world[world])
                for block in blocks
            ]
            for world in world_ids
        ],
        dtype=float,
    )
    strata = defaultdict(list)
    for i, world in enumerate(world_ids):
        row = index[blocks[0], reference][by_world[world][0]]
        strata[
            (row.get("purpose") or "representative") if panel == "B_CONFIRM" else row["family"]
        ].append(i)
    rng = np.random.default_rng(seed_for(2026102225, "A", panel, arm, reference, metric))
    distributions = np.zeros((replicates, len(blocks)))
    for indices in strata.values():
        sampled = rng.choice(indices, size=(replicates, len(indices)), replace=True)
        distributions += matrix[sampled].mean(axis=1) * len(indices) / len(world_ids)
    combined = distributions.mean(axis=1)
    block_means = matrix.mean(axis=0)
    result = dict(
        status="COMPLETE",
        panel=panel,
        metric=metric,
        arm=arm,
        reference=reference,
        answers_per_model=len(reference_keys),
        worlds=len(world_ids),
        bootstrap_replicates=replicates,
        strata={name: len(indices) for name, indices in strata.items()},
        weighting="within-world prompt/draw mean; frozen world-count strata weights; blocks equal",
        difference=float(block_means.mean()),
        paired_world_ci95=np.quantile(combined, [0.025, 0.975]).tolist(),
        one_sided_world_lower95=float(np.quantile(combined, 0.05)),
        per_block=[
            dict(
                block=block,
                difference=float(block_means[i]),
                paired_world_ci95=np.quantile(distributions[:, i], [0.025, 0.975]).tolist(),
            )
            for i, block in enumerate(blocks)
        ],
        inference_scope="item uncertainty conditional on fixed trained models; not training-seed replication",
    )
    if shared_parent_draws:
        result["paired_draw_indices"] = [0, 1, 2, 3]
        result["parent_draws_4_to_7"] = (
            "B measurement only; not extra A comparison or training repeats"
        )
    if len(blocks) == 3:
        # df=2 t critical is descriptive only, never a replacement for seed replication.
        half_width = 4.302652729911275 * float(block_means.std(ddof=1)) / math.sqrt(3)
        result["descriptive_block_t_ci95_df2"] = [
            result["difference"] - half_width,
            result["difference"] + half_width,
        ]
    if reference == "PARENT" and panel == "TEST_ID":
        result["practical_retention_margin"] = -0.02
        result["conditional_item_noninferiority"] = result["one_sided_world_lower95"] > -0.02
        result["retention_caveat"] = (
            "Fixed models and aggregate panel only; not simultaneous worst-variant or new-training-seed noninferiority"
        )
    return result


def coverage(values, tasks, external):
    observed = defaultdict(set)
    for row in values:
        observed[row["block"], row["arm"]].add(
            (row["panel"], row["protocol"], row["qid"], row["draw"])
        )
    output = []
    for block in range(3):
        for arm in ARMS:
            schedule = build_a_eval_groups(tasks, block, external, arm=arm)
            expected = {
                (g["panel"], g["protocol"], g["qid"], draw)
                for g in schedule
                for draw in range(g["G"])
            }
            actual = observed[block, arm]
            output.append(
                dict(
                    block=block,
                    arm=arm,
                    expected=len(expected),
                    actual=len(actual),
                    missing=len(expected - actual),
                    unexpected=len(actual - expected),
                    status="COMPLETE" if actual == expected else "INCOMPLETE_TECHNICAL",
                )
            )
    return output


def _plots(root, comparisons, by_variant):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = root / "ANALYSIS/figures"
    directory.mkdir(parents=True, exist_ok=True)
    selected = [
        row
        for row in comparisons
        if row["status"] == "COMPLETE"
        and row["arm"] == "LZ"
        and row["reference"] == "J_UNIFORM_NEW_REPEAT"
        and row["metric"] == "value_correct"
    ]
    if selected:
        fig, ax = plt.subplots(figsize=(9, max(3, len(selected) * 0.65)))
        for i, row in enumerate(selected):
            mean = 100 * row["difference"]
            low, high = [100 * x for x in row["paired_world_ci95"]]
            ax.errorbar(
                mean,
                i,
                xerr=[[max(0, mean - low)], [max(0, high - mean)]],
                color="black",
                capsize=3,
            )
            for block in row["per_block"]:
                ax.scatter(
                    100 * block["difference"],
                    i + (block["block"] - 1) * 0.13,
                    s=25,
                    color=("#4477aa", "#ee6677", "#228833")[block["block"]],
                    label=f"b{block['block']}" if i == 0 else None,
                )
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_yticks(range(len(selected)), [row["panel"] for row in selected])
        ax.set_xlabel("LZ minus uniform, percentage points")
        ax.set_title(
            "A fixed step 64: B_CONFIRM confirmation; TEST/other panels regression\n3 fixed parent blocks, 1 training repeat/block; paired-world 95% intervals"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / "A_PAIRED_ENDPOINTS.png", dpi=180)
        fig.savefig(directory / "A_PAIRED_ENDPOINTS.pdf")
        plt.close(fig)
    retained = [row for row in by_variant if row["panel"] == "TEST_ID"]
    if retained:
        lookup = {
            (row["block"], row["arm"], row["family"], row["variant"]): row for row in retained
        }
        cells = sorted({(row["family"], row["variant"]) for row in retained})
        fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
        for ax, arm in zip(axes, ("J_UNIFORM_NEW_REPEAT", "LZ"), strict=True):
            for block in range(3):
                delta = []
                for family, variant in cells:
                    current = (
                        lookup.get((block, arm, family, variant), {})
                        .get("value_correct", {})
                        .get("mean")
                    )
                    parent = (
                        lookup.get((block, "PARENT", family, variant), {})
                        .get("value_correct", {})
                        .get("mean")
                    )
                    delta.append(
                        float("nan")
                        if current is None or parent is None
                        else 100 * (current - parent)
                    )
                ax.plot(range(len(cells)), delta, marker=".", label=f"b{block}")
            ax.axhline(-2, color="red", linestyle="--", linewidth=0.8)
            ax.axhline(0, color="gray", linewidth=0.7)
            ax.set_ylabel(f"{arm}\nminus parent (pp)")
            ax.legend(loc="best")
        axes[-1].set_xticks(
            range(len(cells)), [f"{f}_v{v}" for f, v in cells], rotation=70, ha="right"
        )
        fig.suptitle(
            "A fixed step 64; old TEST_ID regression; all 3 blocks and 32 variants\n-2 pp is a practical margin, not a point-estimate noninferiority test"
        )
        fig.tight_layout()
        fig.savefig(directory / "A_RETENTION_BY_VARIANT.png", dpi=180)
        fig.savefig(directory / "A_RETENTION_BY_VARIANT.pdf")
        plt.close(fig)


def _coverage_rates(counters):
    coverage_rows = []
    for (source, dimension), counts in sorted(counters.items()):
        denominator = sum(counts.values())
        coverage_rows.append(
            dict(
                source=source,
                dimension=dimension,
                denominator=denominator,
                counts=dict(counts),
                rates={key: value / denominator for key, value in counts.items()},
            )
        )
    return coverage_rows


def confirmation_measurement(values):
    """Reuse every parent B_CONFIRM draw, separate from development and A fits."""
    selected = [row for row in values if row["panel"] == "B_CONFIRM" and row["arm"] == "PARENT"]
    groups = defaultdict(list)
    for row in selected:
        groups[row["block"], row["qid"]].append(row)
    counters, strata, missing_labels = defaultdict(Counter), Counter(), 0
    for (block, _), items in groups.items():
        j = [row["score"].get("J") for row in items]
        group_type = (
            "UNAVAILABLE"
            if any(value not in (0, 1) for value in j)
            else "ALL_ZERO"
            if not any(j)
            else "ALL_ONE"
            if all(j)
            else "MIXED"
        )
        for row in items:
            labels = row.get("semantic_status", {})
            missing_labels += any(
                labels.get(key) not in ("PASS", "FAIL", "UNKNOWN") for key in "PSOE"
            )
            for dimension in "PSOE":
                status = labels.get(dimension)
                if status not in ("PASS", "FAIL", "UNKNOWN"):
                    status = "MISSING"
                counters["B2_CONFIRM_PARENT", dimension][status] += 1
                strata[block, row["family"], row.get("purpose"), group_type, dimension, status] += 1
    per_block = []
    prompt_sets = []
    for block in range(3):
        items = [row for row in selected if row["block"] == block]
        prompts = {row["qid"] for row in items}
        prompt_sets.append(prompts)
        complete = len(prompts) == 56 and all(
            len(groups[block, qid]) == 8
            and {row["draw"] for row in groups[block, qid]} == set(range(8))
            for qid in prompts
        )
        per_block.append(
            dict(
                block=block,
                expected_answers=448,
                answers=len(items),
                prompts=len(prompts),
                worlds=len({row["world_id"] for row in items}),
                status="COMPLETE" if complete else "INCOMPLETE_TECHNICAL",
            )
        )
    complete = (
        all(row["status"] == "COMPLETE" for row in per_block)
        and prompt_sets[0] == prompt_sets[1] == prompt_sets[2]
        and not missing_labels
    )
    return dict(
        status="COMPLETE" if complete else "INCOMPLETE_TECHNICAL",
        source="B2_CONFIRM_PARENT",
        scope="sealed confirmation, parent step 0 only; never development or training repeats",
        expected_answers=1344,
        answers=len(selected),
        groups=len(groups),
        draws_per_prompt=8,
        missing_semantic_answers=missing_labels,
        per_block=per_block,
        coverage=_coverage_rates(counters),
        coverage_by_stratum=[
            dict(
                block=k[0],
                family=k[1],
                purpose=k[2],
                group_type=k[3],
                dimension=k[4],
                status=k[5],
                count=v,
            )
            for k, v in sorted(strata.items(), key=lambda item: str(item[0]))
        ],
        reuse="Draws 0–3 also supply paired A-parent contrasts; all 0–7 enter B measurement once",
    )


def measurement_summary(root, values=()):
    path = Path(root) / "REPLAY/SUMMARY.json"
    replay = read(path) if path.exists() else dict(status="MISSING_REPLAY")
    confirmation = confirmation_measurement(values)
    counters = defaultdict(Counter)
    for row in replay.get("coverage", []):
        counters[row["source"], row["dimension"]][row["status"]] += row["count"]
    coverage_rows = _coverage_rates(counters) + confirmation["coverage"]
    return dict(
        status="MEASUREMENT_COMPLETE"
        if replay["status"] == "REPLAY_COMPLETE" and confirmation["status"] == "COMPLETE"
        else "INCOMPLETE_TECHNICAL",
        replay_status=replay["status"],
        verifier_version=replay.get("verifier_version"),
        groups_by_source=dict(
            replay.get("groups_by_source", {}), B2_CONFIRM_PARENT=confirmation["groups"]
        ),
        answers=replay.get("answers", 0) + confirmation["answers"],
        development_and_historical_answers=replay.get("answers", 0),
        confirmation=confirmation,
        signal=replay.get("signal", {}),
        signal_by_block=replay.get("signal_by_block", {}),
        signal_scope="C gate signal from development/training probes only; confirmation is excluded",
        coverage=coverage_rows,
        denominator="All answers, including invalid, unfinished and UNKNOWN",
        measurement_scope="observable response claims, not hidden reasoning or human labels",
    )


def training_exposure(root):
    result = []
    for block in range(3):
        for arm in ARMS[1:]:
            directory = root / "RUNS/A" / f"b{block}/r1" / arm
            progress = directory / "PROGRESS.json"
            committed = read(progress).get("successful_updates", 0) if progress.exists() else 0
            counts, sizes, last_seen = Counter(), defaultdict(Counter), {}
            for path in sorted((directory / "assignments").glob("*.json")):
                value = read(path)
                if value["step"] > committed:
                    continue
                for group in value["groups"]:
                    variant = group["variant_id"]
                    counts[variant] += group["g"]
                    sizes[variant][str(group["g"])] += 1
                    last_seen[variant] = value["step"]
            result.append(
                dict(
                    block=block,
                    arm=arm,
                    committed_updates=committed,
                    answers=sum(counts.values()),
                    expected_answers=64 * 128,
                    variants=[
                        dict(
                            variant=variant,
                            answers=count,
                            group_counts=dict(sizes[variant]),
                            last_observed_step=last_seen[variant],
                            staleness=committed - last_seen[variant],
                        )
                        for variant, count in sorted(counts.items())
                    ],
                )
            )
    return result


def _measurement_exposure_plots(root, measurement, exposure, summaries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = root / "ANALYSIS/figures"
    output.mkdir(parents=True, exist_ok=True)
    covered = measurement["coverage"]
    if covered:
        fig, ax = plt.subplots(figsize=(11, max(4, len(covered) * 0.3)))
        left = [0.0] * len(covered)
        for status, color in (
            ("PASS", "#3a8c63"),
            ("FAIL", "#cc6a61"),
            ("UNKNOWN", "#8b929e"),
            ("MISSING", "#d6a84b"),
        ):
            widths = [100 * row["rates"].get(status, 0) for row in covered]
            ax.barh(range(len(covered)), widths, left=left, label=status, color=color)
            left = [a + b for a, b in zip(left, widths, strict=True)]
        ax.set_yticks(
            range(len(covered)), [f"{row['source']} / {row['dimension']}" for row in covered]
        )
        ax.set_xlabel("Percent of all answers (UNKNOWN retained)")
        ax.set_title(
            "B parent step0 / historical RL steps1–64; development and sealed confirmation separately\nObservable P/S/O/E labels; automated labels are not human validation"
        )
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(output / "B_SEMANTIC_COVERAGE.png", dpi=180)
        plt.close(fig)
    if any(row["variants"] for row in exposure):
        variants = sorted({value["variant"] for row in exposure for value in row["variants"]})
        matrix = []
        for row in exposure:
            counts = {value["variant"]: value["answers"] for value in row["variants"]}
            matrix.append([counts.get(variant, 0) for variant in variants])
        fig, ax = plt.subplots(figsize=(13, 4))
        image = ax.imshow(matrix, aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(variants)), variants, rotation=70, ha="right")
        ax.set_yticks(range(len(exposure)), [f"b{r['block']} / {r['arm']}" for r in exposure])
        ax.set_title(
            "A old TRAIN_RL exposure through committed updates (target64)\nAnswer counts include G8/G32; this is training exposure, not evaluation accuracy"
        )
        fig.colorbar(image, ax=ax, label="Generated training answers")
        fig.tight_layout()
        fig.savefig(output / "A_SAMPLING_EXPOSURE.png", dpi=180)
        plt.close(fig)
    old = [row for row in summaries if row["panel"] == "TEST_ID"]
    if old:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        labels = [f"b{r['block']}/{r['arm']}" for r in old]
        axes[0].bar(range(len(old)), [r["mean_completion_tokens"] for r in old])
        axes[0].set_ylabel("Mean completion tokens")
        axes[1].bar(range(len(old)), [100 * r["truncated_fraction"] for r in old], color="#cc6a61")
        axes[1].set_ylabel("Truncated (%)")
        axes[1].set_xticks(range(len(old)), labels, rotation=30, ha="right")
        fig.suptitle(
            "A fixed step64 versus shared parent step0; old TEST_ID regression\nLength and truncation side effects, all3blocks"
        )
        fig.tight_layout()
        fig.savefig(output / "A_LENGTH_TRUNCATION.png", dpi=180)
        plt.close(fig)


def report(root):
    root = Path(root)
    tasks = {row["qid"]: row for row in rows(root / "DATA/TASKS.jsonl")}
    config = read(root / "CONFIG.json")
    external_path = Path(config["old_root"]) / "external/chartqa/TASKS_AUDIT_ONLY.jsonl"
    external = list(rows(external_path))
    all_tasks = dict(tasks, **{row["qid"]: row for row in external})
    values = load_evaluations(root, all_tasks)
    coverage_rows = coverage(values, tasks, external)
    grouped, variants = defaultdict(list), defaultdict(list)
    for row in values:
        grouped[row["block"], row["arm"], row["panel"]].append(row)
        variants[row["block"], row["arm"], row["panel"], row["family"], row["variant"]].append(row)
    summaries = [
        dict(block=block, arm=arm, panel=panel, **summarize(items))
        for (block, arm, panel), items in sorted(grouped.items())
    ]
    by_variant = [
        dict(block=block, arm=arm, panel=panel, family=family, variant=variant, **summarize(items))
        for (block, arm, panel, family, variant), items in sorted(
            variants.items(), key=lambda x: str(x[0])
        )
    ]
    comparisons = []
    for panel in sorted({row["panel"] for row in values}):
        pairs = (
            ("LZ", "J_UNIFORM_NEW_REPEAT"),
            ("LZ", "PARENT"),
            ("J_UNIFORM_NEW_REPEAT", "PARENT"),
        )
        for arm, reference in pairs[:1] if panel == "CHARTQA" else pairs:
            metrics = (
                ("v2.relaxed_accuracy",)
                if panel == "CHARTQA"
                else ("value_correct",)
                if panel in ("NATURAL_WORK", "ANSWER_ONLY")
                else ("value_correct", "J")
            )
            for metric in metrics:
                comparisons.append(
                    paired_world_comparison(values, panel, arm, reference, metric=metric)
                )
    gate_path = root / "C_GATE_REPORT.json"
    gate = (
        read(gate_path)
        if gate_path.exists()
        else dict(status="BLOCK_C", reasons=["MEASUREMENT_NOT_READY: human audit not supplied"])
    )
    measurement = measurement_summary(root, values)
    exposure = training_exposure(root)
    status = (
        "A_B_ENDPOINTS_COMPLETE_C_BLOCKED"
        if all(row["status"] == "COMPLETE" for row in coverage_rows)
        and measurement["status"] == "MEASUREMENT_COMPLETE"
        and gate.get("decision", gate.get("status")) == "BLOCK_C"
        else "INCOMPLETE_TECHNICAL"
    )
    result = dict(
        status=status,
        track_c=gate,
        coverage=coverage_rows,
        comparisons=comparisons,
        measurement=measurement,
        training_exposure=exposure,
        inference_limits=[
            "A has one new paired training repeat per fixed parent block",
            "LZ changes exposure, G and effective update weighting together",
            "Old panels are design-reused regression, not independent confirmation",
            "C BLOCK_C does not show semantic information is useless",
            "No C training advantage or new-world improvement claim is made",
        ],
    )
    write(root / "ANALYSIS/A_B_ENDPOINT_ANALYSIS.json", result)
    write_rows(root / "ANALYSIS/ENDPOINT_BY_MODEL.jsonl", summaries)
    write_rows(root / "ANALYSIS/ENDPOINT_BY_VARIANT.jsonl", by_variant)
    write_rows(root / "ANALYSIS/NORMALIZED_ENDPOINTS.jsonl", values)
    write(root / "ANALYSIS/B_MEASUREMENT_SUMMARY.json", measurement)
    write(root / "ANALYSIS/A_TRAINING_EXPOSURE.json", exposure)
    _plots(root, comparisons, by_variant)
    _measurement_exposure_plots(root, measurement, exposure, summaries)
    lines = [
        "# SR-F4 A/B 固定终点结果",
        "",
        f"当前状态：`{status}`。",
        "",
        "本次新增结果与旧报告分开记录。A 的六条路径使用固定第64次更新；三个父模型各评一次零步基线。旧 TEST_ID、组合、类别数、语言、渲染和 ChartQA 均是已参与设计的回归面板，B_CONFIRM 是本次新世界确认面板。",
        "",
        "C 状态及缺口：",
        "",
        "```json",
        json.dumps(gate, ensure_ascii=False, indent=2),
        "```",
        "",
        "缺少真实人工盲核时，C 不启动。这个阻断不能解释为选择语义没有价值，也没有 SELECT、EVID 或 SHUF 的学习效果可以比较。",
        "",
        "|块|路径|预定回答|实际回答|缺失|状态|",
        "|---|---|---:|---:|---:|---|",
    ]
    lines += [
        f"|b{r['block']}|{r['arm']}|{r['expected']}|{r['actual']}|{r['missing']}|{r['status']}|"
        for r in coverage_rows
    ]
    lines += [
        "",
        "A 的 LZ 与均匀采样比较：",
        "",
        "|面板|指标|LZ−均匀(pp)|配对world 95%区间(pp)|b0/b1/b2 点差(pp)|",
        "|---|---|---:|---|---|",
    ]
    for row in comparisons:
        if (
            row["status"] != "COMPLETE"
            or row["arm"] != "LZ"
            or row["reference"] != "J_UNIFORM_NEW_REPEAT"
        ):
            continue
        interval = ", ".join(f"{100 * x:.2f}" for x in row["paired_world_ci95"])
        blocks = ", ".join(f"b{x['block']}:{100 * x['difference']:.2f}" for x in row["per_block"])
        lines.append(
            f"|{row['panel']}|{row['metric']}|{100 * row['difference']:.2f}|[{interval}]|{blocks}|"
        )
    lines += [
        "",
        "B 的原生 J 是答案正确、证据真实且充分、协议格式与完成状态的合取。证据列表为空时，原生 truth 字段可能因空集合逻辑为1；本轮 P 额外要求非空，不能把 certificate 总门控直接当读数真实性。S 只核验外显回答最终采用的对象集合，模糊或矛盾过程记 UNKNOWN。",
        "",
        "父模型 B2 封存确认固定为3父模型×56题×8draw=1,344条，以下以 B2_CONFIRM_PARENT 独立列示，复用原回答附带的 P/S/O/E 标签。A 与父模型的 B_CONFIRM 比较仅使用双方同seed的前4draw；额外4draw只增加 B 测量，不作为训练重复，也不新增回答。",
        "",
        f"B2 确认实际回答：{measurement['confirmation']['answers']}/1,344；状态：`{measurement['confirmation']['status']}`；缺少语义标签的回答：{measurement['confirmation']['missing_semantic_answers']}。",
        "",
        "|开发/历史/封存确认来源|维度|全部回答数|PASS|FAIL|UNKNOWN|标签缺失|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"|{r['source']}|{r['dimension']}|{r['denominator']}|{r['counts'].get('PASS', 0)}|{r['counts'].get('FAIL', 0)}|{r['counts'].get('UNKNOWN', 0)}|{r['counts'].get('MISSING', 0)}|"
        for r in measurement["coverage"]
    ]
    lines += [
        "",
        "区间来自10,000次分层配对world重采样，保留同一世界所有问题、图像、draw与模型对应记录。它只描述固定已训练模型的题目不确定性，不能用回答数量替代训练重复数量。A每块只有一个新重复；三个父模型块也不是大量独立训练种子。",
        "",
        "逐块、逐变体与最差保持差值见 `ANALYSIS/ENDPOINT_BY_VARIANT.jsonl` 和保持图。−2pp仅是预定实用界；点差没有低于该界不能自动证明非劣，最差子组还有多重查看的不确定性。",
        "",
        "原生答案正确率、J、读数真实性、证据充分性、格式、完成率、长度与截断分别报告。natural_work/answer_only 的原生 J 不适用，不把其恒零值解释为任务失败。ChartQA单列原评分，不并入合成任务正确率，仅评b0两条训练臂；未安排父模型ChartQA，也不报告其未测差值。",
        "",
        "LZ结果只回答这一整套成功率采样配方是否改善固定终点，不能归因为单独的G、曝光或在线状态，也不能替代C对新增选择信息的检验。所有负向块、负向变体和技术缺失均保留。",
    ]
    (root / "REPORT_ZH.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.root), ensure_ascii=False, indent=2))
