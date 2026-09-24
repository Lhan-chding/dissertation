"""Fit and freeze from development evidence; inference never opens outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..modeling_v4 import gpu_collect as gpu
from .features import LEVELS, SCHEMA, PreDecisionPacket
from .jobs import TaskRegistry
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


def _origin_spec(config, origin_id, *, test=False):
    roles = ("test_pool",) if test else ("development", "tuning")
    matches = [
        (spec, step)
        for role in roles
        for spec in config["origins"][role]
        for step in spec["anchors"]
        if origin_id == f"{spec['seed']}_t{step}"
    ]
    if len(matches) != 1:
        raise ValueError("Requested origin is not registered for this phase")
    return matches[0]


def _read_packet(root, origin, spec, step, level, *, fallback=False):
    """Malformed observations may fall back; conflicting provenance never may."""
    try:
        raw = _read(Path(root) / "prestate" / origin / f"{level}.json")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        if fallback:
            return None, f"UNREADABLE_PACKET:{type(exc).__name__}"
        raise
    expected = dict(
        origin_id=origin,
        lineage_id=str(spec["seed"]),
        source_recipe=spec["source_recipe"],
        step=step,
        history_step=step - 8,
        feature_level=level,
    )
    if isinstance(raw, dict) and any(
        key in raw and raw[key] != value for key, value in expected.items()
    ):
        raise ValueError("Packet origin, source, window or layer differs from registration")
    try:
        packet = PreDecisionPacket.from_dict(raw)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        if fallback:
            return None, f"CORRUPT_PACKET:{type(exc).__name__}:{exc}"
        raise
    if not packet.valid_observations or not packet.to_dict()["history"]["n"]:
        if fallback:
            return None, "MISSING_CURRENT_OR_HISTORY_OBSERVATIONS"
        raise ValueError("Development packets require current and history observations")
    return packet, None


def _check_shared_observations(packets):
    """Compare overlapping observables; levels cannot gain different sample budgets."""
    data = {level: packet.to_dict() for level, packet in packets.items() if packet is not None}
    for time in ("current", "history"):
        snapshots = [value[time] for value in data.values()]
        for key in (
            "n",
            "moments_global",
            "moments_stratified",
            "reward_histograms",
            "prompt_strata",
        ):
            values = [snapshot[key] for snapshot in snapshots if key in snapshot]
            if values and any(value != values[0] for value in values[1:]):
                raise ValueError(f"Four layers must share identical {time} observations: {key}")
    metadata = [value["known_training_metadata"] for value in data.values()]
    if metadata and any(value != metadata[0] for value in metadata[1:]):
        raise ValueError("Four layers must share identical known training metadata")


def _test_exposure_exists(config, root):
    root = Path(root)
    if any((root / "decisions").glob("*.json")):
        return True
    for spec in config["origins"]["test_pool"]:
        for step in spec["anchors"]:
            directory = root / "branches" / f"{spec['seed']}_t{step}"
            if directory.exists() and any(p.is_file() for p in directory.rglob("*")):
                return True
    return False


def development_evidence(config, root):
    root = Path(root)
    registry = TaskRegistry(root, config)
    packets, utilities, outcomes = {level: [] for level in LEVELS}, [], []
    for spec in config["origins"]["development"] + config["origins"]["tuning"]:
        for step in spec["anchors"]:
            origin = f"{spec['seed']}_t{step}"
            context = {}
            for level in LEVELS:
                packet, _ = _read_packet(root, origin, spec, step, level)
                context[level] = packet
                packets[level].append(packet)
            _check_shared_observations(context)
            values = {}
            for recipe in (*ACTIONS, *config["actions"]["external_baselines"]):
                branch = registry.branch_output(origin, recipe, 1)
                complete = _read(branch / "COMPLETE.json")
                if complete["status"] != "TRAINING_COMPLETE" or complete["steps"] != 32:
                    raise ValueError("Complete real H32 development matrix required")
                expected = dict(
                    origin_id=origin,
                    lineage_id=spec["seed"],
                    recipe_id=recipe,
                    repeat=1,
                    role=spec["role"],
                )
                if any(complete["identity"].get(k) != v for k, v in expected.items()):
                    raise ValueError("Development training identity mismatch")
                value = _read(branch / "EVAL_H32.json")
                if value["status"] != "EVALUATION_COMPLETE" or value["identity"]["panel_id"] != "D":
                    raise ValueError("Development labels require real independent D evaluation")
                if value["identity"]["draws"] != 16:
                    raise ValueError("Do not mix evaluation sampling protocols")
                expected_eval = dict(
                    origin_id=origin,
                    lineage_id=spec["seed"],
                    repeat=1,
                    policy_id=f"{origin}:{recipe}:1:H32",
                    horizon=32,
                    role="evaluation",
                )
                if any(value["identity"].get(k) != v for k, v in expected_eval.items()):
                    raise ValueError("Development evaluation identity mismatch")
                if value["generated_outputs"] != config["panels"]["D"]["prompts"] * 16:
                    raise ValueError("Development evaluation sample count mismatch")
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
        "independent_lineages": len({p.lineage_id for p in packets[LEVELS[0]]}),
        "final_test_used": False,
    }
    gpu._publish(Path(out) / "FIT.json", result)
    return result


def precision_from_development(config, root, fit):
    from .precision import plan_precision

    if _test_exposure_exists(config, root):
        raise ValueError("Precision planning is forbidden after test decisions or future artifacts")
    packets, utilities, _ = development_evidence(config, root)
    specs = config["origins"]["development"] + config["origins"]["tuning"]
    fit_lineages = {str(spec["seed"]) for spec in config["origins"]["development"]}
    anchors, differences, folds = {}, [], []
    # Source recipes alternate R0/R1. Pair-index parity alternates the stage;
    # this balances all four cells without consulting observed utilities.
    for i, spec in enumerate(specs):
        step = (32, 96)[(i // 2) % 2]
        lid = str(spec["seed"])
        anchors[lid] = {"source_recipe": spec["source_recipe"], "origin_step": step}
        selected = {}
        for level in LEVELS[2:]:
            train = [
                j
                for j, p in enumerate(packets[level])
                if p.lineage_id != lid and p.lineage_id in fit_lineages
            ]
            tune = [
                j
                for j, p in enumerate(packets[level])
                if p.lineage_id != lid and p.lineage_id not in fit_lineages
            ]
            held = next(
                j
                for j, p in enumerate(packets[level])
                if p.lineage_id == lid and p.to_dict()["step"] == step
            )
            model = tune_selector(
                [packets[level][j] for j in train],
                utilities[train],
                [packets[level][j] for j in tune],
                utilities[tune],
            )
            action = model.choose_action(packets[level][held])["recipe"]
            selected[level] = utilities[held, ACTIONS.index(action)]
            folds.append(
                dict(
                    held_lineage=lid,
                    level=level,
                    alpha=model.to_dict()["alpha"],
                    best_static=model.to_dict()["best_static"],
                    selected=action,
                    fit_lineages=sorted({packets[level][j].lineage_id for j in train}),
                    tuning_lineages=sorted({packets[level][j].lineage_id for j in tune}),
                )
            )
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
        test_outcomes_exist=False,
    )
    result["planning_rows"] = differences
    result["planning_folds"] = folds
    result["hyperparameter_source"] = "NESTED_LINEAGE_HOLDOUT_ALPHA_STATIC_AND_COEFFICIENTS"
    result["planning_limitations"] = [
        "Development-only planning estimate; not independent final-test evidence.",
        "Each outer fold fits one fewer source lineage than the final frozen selector.",
        "Finite development data and Monte Carlo outcomes do not guarantee future power.",
    ]
    return result


def freeze_selectors(config, root, fit, precision, prepared, source_version):
    if prepared.get("test_panel_status") != "READY" or len(prepared["panels"]["T"]) != 288:
        raise ValueError("Final freeze requires audited, unexposed T panel")
    models = fit["models"]
    if set(models) != set(LEVELS) or fit.get("final_test_used") is not False:
        raise ValueError("Freeze requires exactly four development-only selectors")
    if _test_exposure_exists(config, root):
        raise ValueError("Cannot freeze after test decisions or future artifacts")
    expected_origins = {
        f"{spec['seed']}_t{step}"
        for role in ("development", "tuning")
        for spec in config["origins"][role]
        for step in spec["anchors"]
    }
    fit_packets = {}
    for level, serialized in models.items():
        model = FrozenSelector.from_dict(serialized)
        if model.feature_level != level:
            raise ValueError("Frozen selector layer mismatch")
        packets = [PreDecisionPacket.from_dict(p) for p in serialized["training_packets"]]
        if {p.origin_id for p in packets} != expected_origins:
            raise ValueError("Final selector must refit all registered development/tuning origins")
        for packet in packets:
            spec, step = _origin_spec(config, packet.origin_id)
            d = packet.to_dict()
            if (
                packet.lineage_id != str(spec["seed"])
                or d["source_recipe"] != spec["source_recipe"]
                or d["step"] != step
                or d["history_step"] != step - 8
            ):
                raise ValueError("Frozen fit packet registration mismatch")
            fit_packets.setdefault(packet.origin_id, {})[level] = packet
    for packets in fit_packets.values():
        _check_shared_observations(packets)
    defaults = {model["best_static"] for model in models.values()}
    if len(defaults) != 1 or fit["contexts"] != 32 or fit["independent_lineages"] != 16:
        raise ValueError("Freeze needs complete development and tuning, shared BestStatic")
    value = dict(
        source_version=source_version,
        implementation_id=implementation_id(),
        model=config["model"],
        recipes=config["actions"],
        frozen_selectors=models,
        feature_schemas={level: SCHEMA for level in LEVELS},
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
    """Load frozen models and prior P packets only; never open outcome contents."""
    spec, step = _origin_spec(config, origin_id, test=True)
    registry = TaskRegistry(root, config)
    frozen = registry.load_freeze()
    verify_frozen_execution(frozen)
    if spec not in config["origins"]["test_pool"][: frozen["test_n"]]:
        raise ValueError("Requested origin lies outside the frozen test cohort")
    packets, reasons, models = {}, {}, {}
    for level in LEVELS:
        packets[level], reasons[level] = _read_packet(
            root, origin_id, spec, step, level, fallback=True
        )
        models[level] = FrozenSelector.from_dict(
            _read(Path(root) / "frozen_selectors" / f"{frozen['selector_ids'][level]}.json")
        )
        serialized = models[level].to_dict()
        if (
            models[level].feature_level != level
            or serialized["best_static"] != frozen["best_static"]
        ):
            raise ValueError("Frozen selector identity/default differs from freeze")
    _check_shared_observations(packets)
    details, choices = {}, {}
    for level in LEVELS:
        if packets[level] is None:
            details[level] = dict(
                selector_id=models[level].selector_id,
                origin_id=origin_id,
                recipe=frozen["best_static"],
                predictions=None,
                predicted_best=None,
                margin_to_default=None,
                status="FALLBACK_BEST_STATIC",
                reason=reasons[level],
            )
        else:
            details[level] = models[level].choose_action(packets[level])
        choices[level] = details[level]["recipe"]
    decision = dict(
        origin_id=origin_id,
        lineage_id=spec["seed"],
        origin_step=step,
        source_recipe=spec["source_recipe"],
        freeze_id=frozen["freeze_id"],
        selector_ids=frozen["selector_ids"],
        choices=choices,
        details=details,
        made_before_run=True,
    )
    return registry.write_decision(decision)
