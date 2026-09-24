"""Fit and freeze from development evidence; inference never opens outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..modeling_v4 import gpu_collect as gpu
from .features import LEVELS, PreDecisionPacket
from .jobs import TaskRegistry
from .protocol import get_lineage
from .selectors import ACTIONS, FrozenSelector, tune_selector


def implementation_id():
    from ..modeling_v3.io import canonical_hash, sha256_file

    source = Path(__file__).parents[1]
    files = sorted(source.rglob("*.py"))
    return canonical_hash({str(p.relative_to(source)): sha256_file(p) for p in files})


def verify_frozen_execution(frozen, prepared=None):
    if frozen.get("implementation_id") != implementation_id():
        raise ValueError("Implementation changed after scientific freeze")
    if prepared is not None and (
        prepared.get("data_id") != frozen["T_panel"]["data_id"]
        or prepared["panels"]["T"] != frozen["T_panel"]["prompts"]
    ):
        raise ValueError("Runtime panel differs from frozen T")


def _read(path):
    return json.loads(Path(path).read_text())


def development_evidence(config, root):
    root = Path(root)
    registry = TaskRegistry(root, config)
    packets, utilities, outcomes = {level: [] for level in LEVELS}, [], []
    for spec in config["origins"]["development"] + config["origins"]["tuning"]:
        for step in spec["anchors"]:
            origin = f"{spec['seed']}_t{step}"
            for level in LEVELS:
                packet = PreDecisionPacket.from_dict(
                    _read(root / "prestate" / origin / f"{level}.json")
                )
                if packet.lineage_id != str(spec["seed"]) or packet.origin_id != origin:
                    raise ValueError("Development packet identity mismatch")
                packets[level].append(packet)
            values = {}
            for recipe in (*ACTIONS, *config["actions"]["external_baselines"]):
                branch = registry.branch_output(origin, recipe, 1)
                complete = _read(branch / "COMPLETE.json")
                if complete["status"] != "TRAINING_COMPLETE" or complete["steps"] != 32:
                    raise ValueError("Complete real H32 development matrix required")
                value = _read(branch / "EVAL_H32.json")
                if value["status"] != "EVALUATION_COMPLETE" or value["identity"]["panel_id"] != "D":
                    raise ValueError("Development labels require real independent D evaluation")
                if value["identity"]["draws"] != 16:
                    raise ValueError("Do not mix evaluation sampling protocols")
                values[recipe] = value["summary"]["J"]
                outcomes.append(
                    {
                        "lineage_id": str(spec["seed"]),
                        "source_recipe": spec["source_recipe"],
                        "origin_step": step,
                        "recipe_id": recipe,
                        "J": values[recipe],
                    }
                )
            utilities.append([values[a] for a in ACTIONS])
    return packets, np.asarray(utilities), outcomes


def fit_selectors(config, root, out):
    packets, utilities, outcomes = development_evidence(config, root)
    nfit = sum(len(s["anchors"]) for s in config["origins"]["development"])
    models = {}
    for level in LEVELS:
        models[level] = tune_selector(
            packets[level][:nfit], utilities[:nfit], packets[level][nfit:], utilities[nfit:]
        ).to_dict()
        gpu._publish(Path(out) / f"{level}.json", models[level])
    result = {
        "status": "DEVELOPMENT_SELECTORS_FIT",
        "models": models,
        "outcomes": outcomes,
        "contexts": len(utilities),
        "independent_lineages": 16,
        "final_test_used": False,
    }
    gpu._publish(Path(out) / "FIT.json", result)
    return result


def precision_from_development(config, root, fit):
    from .precision import plan_precision
    from .selectors import best_static

    packets, utilities, _ = development_evidence(config, root)
    specs = config["origins"]["development"] + config["origins"]["tuning"]
    anchors, differences = {}, []
    # Source recipes alternate R0/R1. Pair-index parity alternates the stage;
    # this balances all four cells without consulting observed utilities.
    for i, spec in enumerate(specs):
        step = (32, 96)[(i // 2) % 2]
        lid = str(spec["seed"])
        anchors[lid] = {"source_recipe": spec["source_recipe"], "origin_step": step}
        selected = {}
        for level in LEVELS[2:]:
            train = [j for j, p in enumerate(packets[level]) if p.lineage_id != lid]
            held = next(
                j
                for j, p in enumerate(packets[level])
                if p.lineage_id == lid and p.to_dict()["step"] == step
            )
            train_packets = [packets[level][j] for j in train]
            static = best_static(train_packets, utilities[train])
            model = FrozenSelector.fit(
                train_packets,
                utilities[train],
                alpha=fit["models"][level]["alpha"],
                default_action=static,
            )
            action = model.choose_action(packets[level][held])["recipe"]
            selected[level] = utilities[held, ACTIONS.index(action)]
        differences.append(
            {
                "lineage_id": lid,
                **anchors[lid],
                "role": spec["role"],
                "difference": float(selected[LEVELS[3]] - selected[LEVELS[2]]),
            }
        )
    result = plan_precision(
        differences,
        anchors,
        test_outcomes_exist=bool(list((Path(root) / "decisions").glob("*.json"))),
    )
    result["planning_rows"] = differences
    result["hyperparameter_source"] = "development/tuning fixed alpha; LLO coefficient/static refit"
    return result


def freeze_selectors(config, root, fit, precision, prepared, source_version):
    if prepared.get("test_panel_status") != "READY" or len(prepared["panels"]["T"]) != 288:
        raise ValueError("Final freeze requires audited, unexposed T panel")
    models = fit["models"]
    defaults = {model["best_static"] for model in models.values()}
    if len(defaults) != 1 or fit["contexts"] != 32 or fit["independent_lineages"] != 16:
        raise ValueError("Freeze needs complete development and tuning, shared BestStatic")
    value = dict(
        source_version=source_version,
        implementation_id=implementation_id(),
        model=config["model"],
        recipes=config["actions"],
        frozen_selectors=models,
        feature_schemas={level: "prospective-predecision-v1" for level in LEVELS},
        scaling={level: m["kernel"] for level, m in models.items()},
        alphas={level: m["alpha"] for level, m in models.items()},
        best_static=defaults.pop(),
        T_panel={"prompts": prepared["panels"]["T"], "data_id": prepared["data_id"]},
        test_n=precision["N"],
        test_draws=precision["mc"]["m"],
        precision=precision,
        primary_comparison=config["selector"]["primary_comparison"],
        horizon=32,
        secondary_analyses=config["statistics"]["secondary_Holm_family"],
        selection_rule={"practical_tie": 0.005, "tie_break": "recipe ID", "invalid": "BestStatic"},
    )
    return TaskRegistry(root, config).write_freeze(value)


def decide(config, root, origin_id):
    """No outcome paths or result stores are accepted or opened here."""
    registry = TaskRegistry(root, config)
    frozen = registry.load_freeze()
    verify_frozen_execution(frozen)
    details, choices = {}, {}
    identity = None
    for level in LEVELS:
        packet = PreDecisionPacket.from_dict(
            _read(Path(root) / "prestate" / origin_id / f"{level}.json")
        )
        model = FrozenSelector.from_dict(
            _read(Path(root) / "frozen_selectors" / f"{frozen['selector_ids'][level]}.json")
        )
        meta = packet.to_dict()
        current_identity = (
            packet.origin_id,
            packet.lineage_id,
            meta["step"],
            meta["source_recipe"],
        )
        if packet.origin_id != origin_id or packet.feature_level != level:
            raise ValueError("Packet origin/layer differs from requested decision")
        if identity is not None and identity != current_identity:
            raise ValueError("Four selectors must receive the same predecision origin")
        identity = current_identity
        details[level] = model.choose_action(packet)
        choices[level] = details[level]["recipe"]
    meta = packet.to_dict()
    spec = get_lineage(config, packet.lineage_id)
    decision = dict(
        origin_id=origin_id,
        lineage_id=int(packet.lineage_id),
        origin_step=meta["step"],
        source_recipe=spec["source_recipe"],
        freeze_id=frozen["freeze_id"],
        selector_ids=frozen["selector_ids"],
        choices=choices,
        details=details,
        made_before_run=True,
    )
    return registry.write_decision(decision)
