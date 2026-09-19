"""Read-only, reproducible reports from completed observation prefixes/receipts.

No experiment is launched here. Missing, partial and fixture evidence remains
explicit, and retrospective reports never redefine the frozen candidate budget.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

from ..modeling_v3.vlm_observation import digest
from .decision_readout import compare_candidate, select_candidate
from .evaluation import (
    fixed_panel_mc_error,
    independent_reference_comparison,
    paired_scene_bootstrap,
    replay_representations,
)
from .intervals import FixedLookAllocation, difference_interval, fixed_panel_hoeffding
from .state_table import build_state

REPORTS = (
    "RUNTIME_MEASUREMENT_zh.md",
    "D1_OBSERVABILITY_zh.md",
    "MULTIREWARD_BLOCK_RESULTS_zh.md",
    "DECISION_READOUT_zh.md",
    "MODELING_DECISION_zh.md",
)


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _group(row):
    return f"{row['family']}|{row['interface']}"


def _load_stream(root):
    identity = _read(root / "samples" / "IDENTITY.json")
    streams = defaultdict(list)
    sample_ids = set()
    for path in sorted((root / "samples").glob("*/*.json")):
        if not re.fullmatch(r"\d+_\d+\.json", path.name):
            continue
        chunk = _read(path)
        if chunk["identity"] != digest(identity) or chunk["rows_hash"] != digest(chunk["rows"]):
            raise ValueError(f"Observation identity/hash mismatch: {path.name}")
        for row in chunk["rows"]:
            if row["prompt_id"] != path.parent.name or row["role"] != identity["role"]:
                raise ValueError("Sample prompt/role differs from stream identity")
            if row["rng_stream_id"] != identity["stream_id"] or row["sample_id"] in sample_ids:
                raise ValueError("Duplicate sample identity or changed RNG stream")
            sample_ids.add(row["sample_id"])
            streams[row["prompt_id"]].append(row)
    for rows in streams.values():
        rows.sort(key=lambda row: row["draw_index"])
        if [row["draw_index"] for row in rows] != list(range(len(rows))):
            raise ValueError("Missing, repeated or noncontiguous observation draws")
        for key in (
            "family",
            "interface",
            "base_scene_id",
            "prompt_record_hash",
            "inference_fingerprint",
        ):
            if len({row[key] for row in rows}) != 1:
                raise ValueError(f"Prompt/policy identity changes within stream: {key}")
    if not streams:
        raise ValueError("No completed observation chunks")
    return identity, dict(streams)


def _panel_reasons(streams, config):
    reasons = []
    expected = config["panels"]
    if len(streams) != expected["prompts_per_panel"]:
        reasons.append("INCOMPLETE_PANEL_PROMPTS")
    scenes = defaultdict(set)
    for rows in streams.values():
        scenes[rows[0]["base_scene_id"]].add(rows[0]["interface"])
    if len(scenes) != expected["scenes_per_panel"] or any(
        interfaces != set(expected["interfaces"]) for interfaces in scenes.values()
    ):
        reasons.append("INCOMPLETE_PAIRED_SCENES")
    groups = {_group(rows[0]) for rows in streams.values()}
    if groups != {"|".join(group) for group in config["statistics"]["key_groups"]}:
        reasons.append("INCOMPLETE_KEY_GROUPS")
    sizes = [sum(_group(rows[0]) == group for rows in streams.values()) for group in groups]
    if sizes and len(set(sizes)) != 1:
        reasons.append("UNBALANCED_FIXED_PANEL_GROUPS")
    return reasons


def _observations(root, config):
    observations, issues = [], []
    for folder in sorted({path.parent for path in root.rglob("LOOK_*.json")}):
        try:
            identity, streams = _load_stream(folder)
            for path in sorted(folder.glob("LOOK_*.json")):
                meta = _read(path)
                look = meta["look"]
                if type(look) is not int or look <= 0:
                    raise ValueError("Invalid declared look")
                if meta["role"] != identity["role"]:
                    raise ValueError("Look role differs from frozen stream")
                if any(len(rows) < look for rows in streams.values()):
                    issues.append(
                        {"path": str(path), "status": "PARTIAL", "reason": "INCOMPLETE_LOOK"}
                    )
                    continue
                prefix = {key: rows[:look] for key, rows in streams.items()}
                flat = [row for rows in prefix.values() for row in rows]
                if meta["total_rows"] != len(flat):
                    raise ValueError("Look count differs from acquired prefix")
                if meta.get("panel_identity") != identity["prompts"]:
                    raise ValueError("Look panel identity differs from frozen stream")
                reasons = _panel_reasons(prefix, config)
                missing = [
                    key
                    for key in ("origin_id", "horizon", "candidate_id", "panel")
                    if key not in meta
                ]
                if missing:
                    reasons.append("MISSING_COMPARISON_IDENTITY:" + ",".join(missing))
                if meta.get("panel") not in ("P", "E"):
                    reasons.append("DEBUG_OR_UNREGISTERED_PANEL")
                if meta.get("execution_kind") != "REAL_CUDA_MODEL":
                    reasons.append("NOT_REAL_CUDA_MODEL")
                role = meta["role"]
                if role not in ("endpoint", "reference"):
                    reasons.append("NON_ENDPOINT_PROPOSAL")
                registered = config["observations"][
                    "reference_looks" if role == "reference" else "looks"
                ]
                if look not in registered:
                    reasons.append("UNREGISTERED_LOOK")
                states = {key: build_state(rows) for key, rows in prefix.items()}
                observations.append(
                    {
                        "path": str(folder),
                        "metadata": meta,
                        "identity": identity,
                        "streams": prefix,
                        "states": states,
                        "decision_blockers": reasons,
                        "sample_count": len(flat),
                        "scoring_failures": sum(
                            not row.get("scoring_usable", False) for row in flat
                        ),
                        "integrity": "PER_CHUNK_ROWS_HASH_AND_STREAM_IDENTITY",
                    }
                )
        except (ValueError, KeyError, TypeError) as error:
            issues.append({"path": str(folder), "status": "INVALID_EVIDENCE", "reason": str(error)})
    return observations, issues


def _candidate_registry(config, origin, observed):
    recipes = list(config["blocks"].get(f"{origin}_recipes", []))
    budget = config["statistics"]["decision_episode"]["candidates"]
    if len(observed) > budget:
        raise ValueError("Observed candidates exceed frozen alpha allocation")
    if recipes and not set(observed) <= set(recipes):
        raise ValueError("Candidate is not in the frozen origin recipe registry")
    names = recipes or sorted(observed)
    while len(names) < budget:
        names.append(f"UNMEASURED_SLOT_{len(names)}")
    return tuple(names)


def _intervals(observation, config, registry):
    meta, streams = observation["metadata"], observation["streams"]
    groups = ["|".join(group) for group in config["statistics"]["key_groups"]]
    metrics = tuple([f"pX:{group}" for group in groups] + ["J:w0", "J:w1", "J:w2"])
    allocation = FixedLookAllocation(
        registry,
        metrics,
        tuple(
            config["observations"]["reference_looks" if meta["role"] == "reference" else "looks"]
        ),
        config["statistics"]["alpha"] / 2,
    )
    # Half the episode alpha is reserved for independent reference inference;
    # union of endpoint and reference intervals still has error <= total alpha.
    result = {}
    for metric in metrics:
        if metric.startswith("pX:"):
            arrays = [
                [int(row["event"] == "X") for row in rows]
                for rows in streams.values()
                if _group(rows[0]) == metric[3:]
            ]
        else:
            weights = config["statistics"]["evaluation_preferences"][metric[2:]]
            arrays = [
                [
                    weights[0] * int(row["event"] == "X")
                    + weights[1] * row["valid"]
                    + weights[2] * row["relation_score"]
                    for row in rows
                ]
                for rows in streams.values()
            ]
        result[metric] = fixed_panel_hoeffding(
            arrays,
            allocation=allocation,
            candidate=meta["candidate_id"],
            metric=metric,
            look=meta["look"],
        )
    return result


def _same_panel(left, right):
    if set(left["streams"]) != set(right["streams"]):
        return False
    return all(
        left["streams"][key][0]["prompt_record_hash"]
        == right["streams"][key][0]["prompt_record_hash"]
        for key in left["streams"]
    )


def _descriptive_comparison(candidate, baseline, config):
    """Paired scene uncertainty and fixed-panel generation MC remain separate."""
    readers = {
        "pX": lambda row: int(row["event"] == "X"),
        "pA": lambda row: row["answer_correct"],
        "V": lambda row: row["valid"],
        "C": lambda row: row["relation_score"],
    }
    result = {}
    for metric, read in readers.items():
        scene_rows, difference_draws = [], []
        for prompt, left in candidate["streams"].items():
            right = baseline["streams"][prompt]
            scene_rows.append(
                {
                    "family": left[0]["family"],
                    "base_scene": left[0]["base_scene_id"],
                    "interface": left[0]["interface"],
                    "left": math.fsum(read(row) for row in left) / len(left),
                    "right": math.fsum(read(row) for row in right) / len(right),
                }
            )
            difference_draws.append([read(a) - read(b) for a, b in zip(left, right, strict=True)])
        result[metric] = {
            "scene_sampling": paired_scene_bootstrap(
                scene_rows,
                replicates=config["statistics"].get("bootstrap_repetitions", 5000),
                seed=config.get("seed", 20260919),
                interfaces=config["panels"]["interfaces"],
            ),
            "fixed_panel_generation_mc": fixed_panel_mc_error(difference_draws),
        }
    return result


def _additional_targets(look_result, panel, config):
    """Use acquired development intervals only; no independent reference input."""
    if panel != "P":
        return []
    look = look_result["look"]
    later = [n for n in config["observations"]["looks"] if n > look]
    contenders = set()
    for selection in look_result["selections"].values():
        if (
            selection["regret_upper"] is None
            or selection["regret_upper"] > config["statistics"]["epsilon_J"]
        ):
            contenders.update(selection["candidate_set"])
    targets = []
    for candidate in sorted(contenders):
        if candidate.startswith("UNMEASURED_SLOT_"):
            continue
        if candidate in look_result["bounds"] and not later:
            continue
        comparison = look_result["comparisons"].get(candidate, {})
        targets.append(
            {
                "candidate_id": candidate,
                "next_look": min(later)
                if candidate in look_result["bounds"]
                else min(config["observations"]["looks"]),
                "unresolved_groups": comparison.get(
                    "unresolved_groups",
                    ["|".join(group) for group in config["statistics"]["key_groups"]],
                ),
                "reason": "CURRENT_INTERVALS_CAN_CHANGE_CHOICE",
                "uses_reference_or_test": False,
                "automatic_execution": False,
            }
        )
    return targets


def _replay(items, config, registry):
    looks = sorted({item["metadata"]["look"] for item in items})
    by_candidate = {}
    for item in sorted(items, key=lambda item: item["metadata"]["look"]):
        by_candidate[item["metadata"]["candidate_id"]] = item

    def builder(prefix, level):
        return {
            key: {"state": build_state(rows)[level], "n": len(rows)} for key, rows in prefix.items()
        }

    def rule(states):
        means = defaultdict(list)
        for key, item in states.items():
            means[key.split("::", 1)[0]].append(item["state"]["pX"])
        utility = {key: math.fsum(values) / len(values) for key, values in means.items()}
        best = max(utility.values())
        return {
            "preference": "w0",
            "candidate_set": sorted(key for key, value in utility.items() if value == best),
            "pX": utility,
            "status": "DESCRIPTIVE_MEAN_RANKING",
            "evidence_kind": "SAME_PREFIX_EMPIRICAL_MEANS",
            "noninferiority": "NOT_ASSERTED_BY_PLUGIN_RANKING",
            "joint_information_gain": "NOT_ESTABLISHED",
        }

    result = []
    for look in looks:
        streams = {
            f"{candidate}::{prompt}": rows
            for candidate, item in by_candidate.items()
            if item["metadata"]["look"] >= look
            for prompt, rows in item["streams"].items()
        }
        result.extend(
            replay_representations(
                streams,
                representation_builder=builder,
                rules={f"Z{i}": rule for i in range(5)},
                looks=[look],
            )
        )
    return result


def _decisions(observations, config):
    episodes, results = defaultdict(list), []
    for item in observations:
        meta = item["metadata"]
        if not item["decision_blockers"] and meta["role"] == "endpoint":
            episodes[(meta["origin_id"], meta["horizon"], meta["panel"])].append(item)
    for (origin, horizon, panel), items in sorted(episodes.items()):
        episode = {
            "origin_id": origin,
            "horizon": horizon,
            "panel": panel,
            "looks": [],
            "issues": [],
            "experiment_scope": "D2_D4_BLOCK"
            if origin in config["blocks"]["origins"] and horizon in (0, 8, 32)
            else "EXPLORATORY_D1",
        }
        results.append(episode)
        try:
            registry = _candidate_registry(
                config, origin, {item["metadata"]["candidate_id"] for item in items}
            )
            actual_registry = [name for name in registry if not name.startswith("UNMEASURED_SLOT_")]
            episode["candidate_registry"] = actual_registry
            episode["alpha_candidate_slots"] = len(registry)
            if any(not _same_panel(items[0], item) for item in items[1:]):
                raise ValueError("Same named panel has differing prompt identities")
            duplicate_keys = [
                (item["metadata"]["candidate_id"], item["metadata"]["look"]) for item in items
            ]
            if len(duplicate_keys) != len(set(duplicate_keys)):
                raise ValueError(
                    "Duplicate candidate/look streams; choose one frozen stream explicitly"
                )
            for look in sorted({item["metadata"]["look"] for item in items}):
                current = {
                    item["metadata"]["candidate_id"]: item
                    for item in items
                    if item["metadata"]["look"] == look
                }
                bounds = {
                    name: _intervals(item, config, registry) for name, item in current.items()
                }
                comparisons = {}
                groups = ["|".join(group) for group in config["statistics"]["key_groups"]]
                for name, item in current.items():
                    baseline = "R1" if name in ("R1", "R5", "R6", "R7") else "R0"
                    baseline = item["metadata"].get("baseline_candidate", baseline)
                    if baseline not in current:
                        comparisons[name] = {
                            "status": "UNRESOLVED",
                            "reason": "MATCHED_BASELINE_NOT_MEASURED",
                            "feasibility": "UNKNOWN",
                        }
                        continue
                    delta = {
                        group: difference_interval(
                            bounds[name][f"pX:{group}"]["interval"],
                            bounds[baseline][f"pX:{group}"]["interval"],
                        )
                        for group in groups
                    }
                    comparisons[name] = compare_candidate(
                        delta,
                        difference_interval(
                            bounds[name]["J:w0"]["interval"], bounds[baseline]["J:w0"]["interval"]
                        ),
                        required_groups=groups,
                        evidence_kind="FINITE_SAMPLE_FIXED_PANEL_HOEFFDING",
                        tau_x=config["statistics"]["tau_X"],
                        epsilon=config["statistics"]["epsilon_J"],
                    )
                    # A baseline is identically itself, not two independently
                    # unknown expectations. Its contrast is exactly zero.
                    if name == baseline:
                        delta = {group: [0, 0] for group in groups}
                        comparisons[name] = compare_candidate(
                            {group: [0, 0] for group in groups},
                            [0, 0],
                            required_groups=groups,
                            evidence_kind="IDENTICAL_POLICY_CONTRAST",
                            tau_x=config["statistics"]["tau_X"],
                            epsilon=config["statistics"]["epsilon_J"],
                        )
                    comparisons[name]["baseline_candidate"] = baseline
                    comparisons[name]["group_differences"] = delta
                    comparisons[name]["descriptive_uncertainty"] = _descriptive_comparison(
                        item,
                        current[baseline],
                        config,
                    )
                selections = {}
                for preference in ("w0", "w1", "w2") if panel == "P" else ():
                    # Keep main-task families separate, including unmeasured recipes.
                    for family, excluded in (
                        ("exact", {"R1", "R5", "R6", "R7"}),
                        ("answer", set(actual_registry) - {"R1", "R5", "R6", "R7"}),
                    ):
                        candidates = [name for name in actual_registry if name not in excluded]
                        if not candidates:
                            continue
                        selections[f"{family}:{preference}"] = select_candidate(
                            {
                                name: bounds[name][f"J:{preference}"]["interval"]
                                if name in bounds
                                else [0, 1]
                                for name in candidates
                            },
                            {
                                name: comparisons.get(name, {}).get("feasibility", "UNKNOWN")
                                for name in candidates
                            },
                            evidence_kind="FINITE_SAMPLE_FIXED_PANEL_HOEFFDING",
                            simultaneous=True,
                        )
                look_result = {
                    "look": look,
                    "bounds": bounds,
                    "comparisons": comparisons,
                    "selections": selections,
                    "selection_scope": "DEVELOPMENT_P"
                    if panel == "P"
                    else "EVALUATION_ONLY_NO_SELECTION",
                }
                look_result["additional_observation_targets"] = _additional_targets(
                    look_result, panel, config
                )
                episode["looks"].append(look_result)
            episode["same_prefix_replay"] = _replay(items, config, registry) if panel == "P" else []
            episode["independent_reference"] = _references(items, observations, config, registry)
            for look_result in episode["looks"]:
                matches = [
                    row
                    for row in episode["independent_reference"]
                    if row.get("look") == look_result["look"]
                    and row["status"] == "INDEPENDENT_REFERENCE_MEASURED"
                ]
                reference_by_candidate = {
                    row["candidate_id"]: row
                    for row in sorted(matches, key=lambda row: row["reference_look"])
                }
                look_result["independent_reference_selection"] = {}
                for key, selection in look_result["selections"].items():
                    selected = selection["selected"]
                    if selected not in reference_by_candidate:
                        look_result["independent_reference_selection"][key] = {
                            "status": "NOT_RUN",
                            "reason": "SELECTED_CANDIDATE_REFERENCE_MISSING",
                        }
                        continue
                    preference = key.split(":")[1]
                    candidate_bounds = {
                        name: reference_by_candidate[name]["bounds"][f"J:{preference}"]["interval"]
                        if name in reference_by_candidate
                        else None
                        for name in selection["candidate_set"]
                    }
                    look_result["independent_reference_selection"][key] = (
                        independent_reference_comparison(
                            reference_by_candidate[selected]["bounds"][f"J:{preference}"][
                                "interval"
                            ],
                            candidate_bounds,
                        )
                    )
            episode["status"] = "ANALYZED_FIXED_PANEL"
        except (ValueError, KeyError, TypeError) as error:
            episode["status"] = "UNRESOLVED"
            episode["issues"].append(str(error))
    return results


def _references(items, observations, config, registry):
    references = []
    for endpoint in items:
        meta = endpoint["metadata"]
        for reference in observations:
            other = reference["metadata"]
            if reference["decision_blockers"] or other["role"] != "reference":
                continue
            if any(
                meta[key] != other[key] for key in ("origin_id", "horizon", "candidate_id", "panel")
            ):
                continue
            independent = endpoint["identity"]["stream_id"] != reference["identity"]["stream_id"]
            same_policy = _same_panel(endpoint, reference) and all(
                endpoint["streams"][key][0]["inference_fingerprint"]
                == reference["streams"][key][0]["inference_fingerprint"]
                for key in endpoint["streams"]
            )
            endpoint_ids = {
                row["sample_id"] for rows in endpoint["streams"].values() for row in rows
            }
            independent = independent and not any(
                row["sample_id"] in endpoint_ids
                for rows in reference["streams"].values()
                for row in rows
            )
            if not independent or not same_policy:
                references.append(
                    {
                        "candidate_id": meta["candidate_id"],
                        "status": "INVALID_REFERENCE_IDENTITY_OR_RNG",
                    }
                )
                continue
            reference_bounds = _intervals(reference, config, registry)
            interval = reference_bounds["J:w0"]["interval"]
            endpoint_interval = _intervals(endpoint, config, registry)["J:w0"]["interval"]
            references.append(
                {
                    "candidate_id": meta["candidate_id"],
                    "look": meta["look"],
                    "reference_look": other["look"],
                    "status": "INDEPENDENT_REFERENCE_MEASURED",
                    "interval": interval,
                    "bounds": reference_bounds,
                    "comparison": independent_reference_comparison(
                        endpoint_interval, {meta["candidate_id"]: interval}
                    ),
                }
            )
    return references or [{"status": "NOT_RUN"}]


def _blocks(root):
    result = []
    for path in sorted(root.rglob("MANIFEST.json")):
        manifest = _read(path)
        if manifest.get("kind") != "DECISION_MODELING_BLOCK":
            continue
        commits = [_read(p) for p in sorted(path.parent.glob("steps/H*/COMMIT.json"))]
        issues = []
        if [item["step"] for item in commits] != list(range(1, len(commits) + 1)):
            issues.append("NONCONTIGUOUS_STEP_RECEIPTS")
        if any(item["manifest_hash"] != digest(manifest) for item in commits):
            issues.append("STEP_MANIFEST_HASH_MISMATCH")
        endpoints = [h for h in (0, 8, 32) if (path.parent / f"H{h:02d}.json").exists()]
        if any(h > len(commits) for h in endpoints):
            issues.append("ENDPOINT_WITHOUT_COMMITTED_STEPS")
        result.append(
            {
                "path": str(path.parent),
                "origin_id": manifest["origin_id"],
                "recipe": manifest["recipe"],
                "fixture": manifest.get("fixture", False),
                "committed_steps": len(commits),
                "endpoint_receipts": endpoints,
                "training_outputs": sum(item["training_outputs"] for item in commits),
                "sampling_seconds": math.fsum(item["sampling_seconds"] for item in commits),
                "update_seconds": math.fsum(item["update_seconds"] for item in commits),
                "status": "INVALID_RECEIPTS"
                if issues
                else "CPU_FIXTURE"
                if manifest.get("fixture")
                else "RECEIPTS_PRESENT",
                "issues": issues,
                "checkpoint_payload_rehashed": False,
            }
        )
    return result


def report(root, out, config):
    """Create report artifacts in a fresh directory; source files remain read-only."""
    root, out = Path(root).resolve(), Path(out).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if out == root or root.is_relative_to(out) or (out.exists() and any(out.iterdir())):
        raise FileExistsError("Use a new empty report directory, not the source root")
    if isinstance(config, (str, Path)):
        config = _read(config)
    observations, issues = _observations(root, config)
    blocks = _blocks(root)
    planned_blocks = [
        {
            "origin_id": origin,
            "recipe": recipe,
            "status": next(
                (
                    item["status"]
                    for item in blocks
                    if item["origin_id"] == origin and item["recipe"] == recipe
                ),
                "NOT_RUN",
            ),
        }
        for origin in ("O1", "O2")
        for recipe in config["blocks"].get(f"{origin}_recipes", [])
    ]
    costs = [
        {"path": str(path), "cost": _read(path), "scope": "IMMUTABLE_INVOCATION_RECEIPT"}
        for path in sorted(root.glob("**/costs/*.json"))
    ]
    costs += [
        {
            "path": str(path),
            "cost": _read(path),
            "scope": "LEGACY_LAST_INVOCATION_ONLY_NOT_CUMULATIVE",
        }
        for path in sorted(root.rglob("LAST_COST.json"))
    ]
    d0 = [
        {"path": str(path), "summary": _read(path)}
        for path in sorted(root.rglob("summary.json"))
        if _read(path).get("status") == "D0_REANALYZED"
    ]
    decisions = _decisions(observations, config)
    lr = [
        {"path": str(path), "analysis": _read(path)}
        for path in sorted(root.rglob("ANALYSIS.json"))
        if _read(path).get("proposal") in ("ORIGIN", "MIX")
    ]
    summary = {
        "status": "REPORT_COMPLETE",
        "new_model_calls": 0,
        "new_training_steps": 0,
        "observation_looks": [
            {key: value for key, value in item.items() if key not in ("streams", "states")}
            for item in observations
        ],
        "observation_issues": issues,
        "blocks": blocks,
        "costs": costs,
        "d0": d0,
        "decision_episodes": decisions,
        "scope": "FIXED_PANEL_ENDPOINT_ANALYSIS; NO_TRAINING_SEED_GENERALIZATION",
        "joint_information_gain": "NOT_ESTABLISHED",
        "automatic_successor": False,
        "lr_observations": lr,
        "planned_blocks": planned_blocks,
    }
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "analysis.json", summary)
    _write(
        out / "endpoint_states.json",
        [
            {"path": item["path"], "metadata": item["metadata"], "states": item["states"]}
            for item in observations
        ],
    )
    runtime = [
        "# 运行成本",
        "",
        "所有数值来自现有回执；本报告未调用模型。",
        "",
        "costs/ 内为逐次不可变回执；历史 LAST_COST 仅代表最后一次调用，不能当作累计耗时。"
        "缓存命中率不能解释成墙钟加速率。",
        "",
    ]
    for item in costs:
        runtime += [f"- `{item['path']}`：`{json.dumps(item['cost'], ensure_ascii=False)}`"]
    for item in blocks:
        runtime += [
            f"- {item['origin_id']}/{item['recipe']}：采样 {item['sampling_seconds']:.6f} 秒；"
            f"更新 {item['update_seconds']:.6f} 秒；fixture={item['fixture']}。"
        ]
    for item in d0:
        runtime += [
            f"- D0：{item['summary'].get('generation_tokens')} 生成 token；"
            f"生成行级秒数和 {item['summary'].get('generation_seconds_sum')}；"
            f"评分秒数 {item['summary'].get('scoring_seconds_status')}。"
        ]
    if not costs and not blocks and not d0:
        runtime += ["NOT_RUN：尚无运行成本回执。"]
    observability = [
        "# D1 可观测性",
        "",
        "逐题状态、频数、reward 协方差及精确有理关系联合计数见 endpoint_states.json。"
        "未知尾部未被当作确定事件；零 X 样本不能证明 pX=0。",
        "",
    ]
    for item in observations:
        meta = item["metadata"]
        observability += [
            f"- {meta.get('candidate_id', 'UNKNOWN')} / {meta['role']} / look={meta['look']}："
            f"{item['sample_count']} 输出；评分不一致 {item['scoring_failures']}；"
            f"判定阻塞 `{item['decision_blockers']}`。"
        ]
        groups = defaultdict(list)
        for prompt, rows in item["streams"].items():
            groups[_group(rows[0])].append(item["states"][prompt])
        for group, states in sorted(groups.items()):
            px = math.fsum(state["Z0"]["pX"] for state in states) / len(states)
            valid = math.fsum(state["Z0"]["V"] for state in states) / len(states)
            answer = math.fsum(
                state["Z1"]["event_probabilities"]["X"] + state["Z1"]["event_probabilities"]["S"]
                for state in states
            ) / len(states)
            relation = (
                math.fsum(state["Z2"]["reward_moments"]["mean"][3] for state in states)
                / len(states)
                if all(state["Z2"]["reward_moments"] is not None for state in states)
                else None
            )
            qx = px / valid if valid else None
            observability += [
                f"  - {group}：{len(states)} prompts；pX={px:.6f}；pA={answer:.6f}；"
                f"V={valid:.6f}；C={relation}；qX={qx}（pX/V）"
                "（逐题等权经验均值）。"
            ]
    if not observations:
        observability += ["NOT_RUN：无新端点观测。"]
    for item in lr:
        observability += [
            f"- {item['analysis']['proposal']}：{item['analysis'].get('status')}；"
            f"{len(item['analysis'].get('units', []))} 个逐题 RAW4/PRESERVE_XI、"
            "协方差及条件质量界记录，详见 analysis.json；经验协方差不是有限样本证书。"
        ]
    observability += [
        "",
        f"完整性问题：`{json.dumps(issues, ensure_ascii=False)}`。",
        "评分不一致只禁用对应 LR/质量解释；合法端点采样仍可单列分析。"
        "质量界未覆盖数值误差时仍为 CONDITIONAL_ON_SCORER / NOT_CERTIFIED。",
    ]
    block_lines = [
        "# 多奖励短训练块",
        "",
        "记录 H0/H8/H32 回执与连续提交步数；回执存在不等于已重新校验服务器 checkpoint 原件。",
        "",
    ]
    block_lines += [
        f"- {item['origin_id']}/{item['recipe']}：{item['status']}；"
        f"{item['committed_steps']} 步，{item['training_outputs']} 输出，"
        f"端点 {item['endpoint_receipts']}；问题 {item['issues']}。"
        for item in blocks
    ]
    if not blocks:
        block_lines += ["NOT_RUN：尚无 D2/D4 训练块回执。"]
    block_lines += [
        f"- 登记分支 {item['origin_id']}/{item['recipe']}：{item['status']}。"
        for item in planned_blocks
    ]
    decision_lines = [
        "# 决策读出",
        "",
        "固定面板 Hoeffding 使用冻结候选预算、6 个群体及 3 个效用、"
        "预定 look 的联合 alpha 分配；端点与独立参考各预留总 alpha 的一半。"
        "不同起点、窗口或面板不能合并比较。",
        "",
        "完整界、缺失候选、非劣/等效/伤害及 regret 见 analysis.json。"
        "未观测候选保留 [0,1] / UNKNOWN。",
        "",
    ]
    for episode in decisions:
        decision_lines += [
            f"- {episode['origin_id']}/H{episode['horizon']}/{episode['panel']}："
            f"{episode['status']}；问题 {episode['issues']}。"
        ]
        for look in episode["looks"]:
            for key, selection in look["selections"].items():
                decision_lines += [
                    f"  - look={look['look']}，{key}：selected={selection['selected']}；"
                    f"regret_upper={selection['regret_upper']}；"
                    f"UNKNOWN={selection['unknown_candidates']}。"
                ]
    if not decisions:
        decision_lines += ["NOT_RUN/UNRESOLVED：没有身份、样本前缀与完整面板均合规的可比较端点。"]
    decision_lines += [
        "",
        "Z0-Z4 对同一前缀回放的 w0 均值排序为描述性结果；未将其当作群体非劣证书。"
        "均值效用的排序不预设联合结构必然改善。"
        "独立参考若缺失明确标记 NOT_RUN；参考区间不是精确真值。",
    ]
    decision_lines += [
        "追加目标 additional_observation_targets 仅从当前 P 面板区间、"
        "候选集合与未判定群体导出，不读取 reference/E，不自动执行。"
        "独立参考 256→1024 的重要选择理由由采集入口显式记录。"
        "经验方法校准：NOT_IMPLEMENTED；无经验零方差证书。"
    ]
    modeling = [
        "# 建模阶段事实",
        "",
        f"D0 重分析 {len(d0)} 份；新观测 look {len(observations)} 份；"
        f"训练块回执 {len(blocks)} 份；可分析决策 episode {len(decisions)} 份。",
        "",
        "联合结构的额外决策收益：NOT_ESTABLISHED。"
        "当前输出提供状态、同证据回放与保守决策区间；"
        "没有据此宣称胜过 SAW/GDPO、完成在线控制或独立训练 seed 迁移。",
        "",
        "固定面板生成随机性与场景总体、训练 seed 差异不能混合。"
        "完整匹配端点的 pX/pA/V/C 配对场景 bootstrap 默认 5000 次，按家族分层，"
        "两接口共同重抽且不嵌套 completion；固定面板 MC 与 Hoeffding 另列。"
        "未启动任何后继实验。",
    ]
    for name, lines in zip(
        REPORTS, (runtime, observability, block_lines, decision_lines, modeling), strict=True
    ):
        (out / name).write_text("\n".join(lines) + "\n")
    return summary
