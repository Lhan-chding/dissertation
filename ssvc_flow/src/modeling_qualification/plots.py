"""Five deterministic figures backed by published source tables."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _cap(value):
    return 24 if str(value) == "FULL" else _number(value)


def _save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def render_figures(m3: Path, m4: Path, out: Path) -> list[dict]:
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120}
    )
    records = []
    comparison_root = m3.parent / "M3_primary_tables"
    rows = _rows(comparison_root / "primary_aggregate_metrics.csv")
    rows += [
        r
        for r in _rows(comparison_root / "robust_aggregate_metrics.csv")
        if r["n"] in ("16", "256")
    ]

    primary = [
        r
        for r in rows
        if r.get("fit_budget") == "8"
        and r.get("representation") == "per_prompt"
        and r.get("input_mode") == "actual_d"
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    methods = list(dict.fromkeys(r["method"] for r in primary))
    labels_by_method = {
        "B2_FULL_PARAMETER_SKETCH": "B2 full sketch",
        "B4_ENERGY95": "B4 95% energy",
        "B5_EXACT_SELECTED_DIAGNOSTIC": "B5 exact rule",
    }

    def method_label(method):
        return labels_by_method.get(method, method.split("_")[0])

    for col, n in enumerate(["exact", "16", "64", "256"]):
        for mi, method in enumerate(methods):
            selected = sorted(
                [r for r in primary if r["n"] == n and r["method"] == method],
                key=lambda r: _cap(r["rank_cap"]),
            )
            if not selected:
                continue
            x = [_cap(r["rank_cap"]) for r in selected]
            axes[0, col].plot(
                x,
                [_number(r["delta_mae"]) for r in selected],
                ".-",
                label=method_label(method),
                color=plt.get_cmap("tab20")(mi),
                lw=1,
            )
            axes[1, col].plot(
                x,
                [
                    max(_number(r["group_delta_pX_q95_max"]), _number(r["group_delta_v_q95_max"]))
                    for r in selected
                ],
                ".-",
                color=plt.get_cmap("tab20")(mi),
                lw=1,
            )
        axes[0, col].set_title(f"n = {n}")
        axes[1, col].axhline(0.01, color="black", ls=":", lw=1)
        axes[1, col].set_xlabel("rank cap (24 tick = FULL)")
        for ax in axes[:, col]:
            ax.set_xticks([0, 4, 8, 16, 24], ["0", "4", "8", "16", "FULL"])
            ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Response MAE (probability)")
    axes[1, 0].set_ylabel("Worst group |error| q95")
    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), frameon=False)
    fig.suptitle("Locked test: error vs response rank; missing budgets have no full grid")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    _save(fig, out, "01_error_dimension")
    records.append(
        {
            "figure": "01_error_dimension",
            "source": (
                "M3_primary_tables/primary_aggregate_metrics.csv + robust_aggregate_metrics.csv"
            ),
        }
    )

    scatter = _rows(comparison_root / "response_scatter.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    for group in range(6):
        subset = [r for r in scatter if int(r["group"]) == group]
        ax.scatter(
            [_number(r["true_delta_pX"]) for r in subset],
            [_number(r["pred_delta_pX"]) for r in subset],
            s=6,
            alpha=0.4,
            label=f"group {group}",
        )
    all_x = [_number(r["true_delta_pX"]) for r in scatter]
    all_y = [_number(r["pred_delta_pX"]) for r in scatter]
    limit = max([0.01, *map(abs, all_x), *map(abs, all_y)]) * 1.05
    ax.plot([-limit, limit], [-limit, limit], "k:", lw=1)
    ax.axvspan(-0.005, 0.005, color="grey", alpha=0.1)
    ax.set(
        xlabel="True group delta pX",
        ylabel="Predicted group delta pX",
        title=f"Frozen exact-selected r=1; n=64, noise=0; {len(scatter)} group responses",
    )
    ax.legend(fontsize=7, frameon=False)
    _save(fig, out, "02_response_scatter")
    records.append(
        {"figure": "02_response_scatter", "source": "M3_primary_tables/response_scatter.csv"}
    )

    example = _rows(m4 / "example_exact_selected_timeseries.csv")
    example = [r for r in example if str(r.get("n")) == "64"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for group, ax in enumerate(axes.flat):
        subset = sorted(
            [r for r in example if int(r["group"]) == group and r["event"] == "pX"],
            key=lambda r: int(r["elapsed_step"]),
        )
        x = [_number(r["elapsed_step"]) for r in subset]
        ax.plot(x, [_number(r["truth"]) for r in subset], label="exact evaluator", lw=1.5)
        ax.plot(
            x, [_number(r["raw_prediction"]) for r in subset], label="frozen prediction", lw=1.5
        )
        ax.fill_between(
            x,
            [_number(r["envelope_lower"]) for r in subset],
            [_number(r["envelope_upper"]) for r in subset],
            alpha=0.2,
            label="empirical envelope",
        )
        ax.set_title(f"group {group}")
        ax.set_ylabel("pX")
        ax.set_xlabel("steps since anchor")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Preassigned seed 301 / X_BASE / anchor 8; exact-selected r=1, n=64")
    fig.tight_layout()
    _save(fig, out, "03_fixed_window")
    records.append(
        {"figure": "03_fixed_window", "source": "M4/example_exact_selected_timeseries.csv"}
    )

    tracking = _rows(m4 / "tracking_rank_horizon_grid.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for n, ax in zip(["exact", "64"], axes, strict=True):
        heat = np.full((7, 5), np.nan)
        for i, cap in enumerate(["0", "1", "2", "4", "8", "16", "FULL"]):
            for j, horizon in enumerate([1, 2, 4, 8, 16]):
                subset = [
                    r
                    for r in tracking
                    if r["method"] == "B5_WORST_GROUP_RANK"
                    and r["rank_cap"] == cap
                    and r["n"] == n
                    and int(r["H"]) == horizon
                    and r.get("telemetry") == "PROJECTION_AND_NET_DISPLACEMENT"
                ]
                if subset:
                    heat[i, j] = max(
                        _number(subset[0]["group_delta_pX_q95_max"]),
                        _number(subset[0]["group_delta_v_q95_max"]),
                    )
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad("#dddddd")
        im = ax.imshow(heat, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(5), ["1", "2", "4", "8", "16"])
        ax.set_yticks(range(7), ["0", "1", "2", "4", "8", "16", "FULL"])
        ax.set(xlabel="H (all intermediate steps scored)", ylabel="rank cap", title=f"n={n}")
        fig.colorbar(im, ax=ax, label="Worst group q95 error")
    fig.suptitle("Rank and fixed-window error; grey cells are not evaluated")
    _save(fig, out, "04_dimension_window")
    records.append({"figure": "04_dimension_window", "source": "M4/tracking_rank_horizon_grid.csv"})

    selection_rows = [
        r
        for r in primary
        if r["n"] == "64"
        and (r["rank_cap"] in ("4", "FULL") or r["method"].startswith(("B0_", "B1_")))
    ]
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(selection_rows))
    ax.bar(
        x - 0.18,
        [_number(r["delta_mae"]) for r in selection_rows],
        width=0.36,
        label="mean response error",
    )
    ax.bar(
        x + 0.18,
        [
            max(_number(r["group_delta_pX_q95_max"]), _number(r["group_delta_v_q95_max"]))
            for r in selection_rows
        ],
        width=0.36,
        label="worst group q95 error",
    )
    ax.set_xticks(
        x,
        [f"{method_label(r['method'])} r={r['rank_cap']}" for r in selection_rows],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("Probability error")
    ax.set_title(
        "n=64; B2-B6 share 8 paid fit banks (24 candidate panels); B0/B1 shown as references"
    )
    ax.legend(frameon=False)
    _save(fig, out, "05_equal_budget")
    records.append(
        {
            "figure": "05_equal_budget",
            "source": (
                "M3_primary_tables/primary_aggregate_metrics.csv + robust_aggregate_metrics.csv"
            ),
        }
    )
    return records
