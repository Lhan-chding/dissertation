"""Finite fresh cohorts, exhaustive unique repair checks, and proven replay."""

import copy
import json
import random
from collections import Counter
from pathlib import Path

from ..constraint_solver import solve
from ..protocol_state_probes.ptlc import dpe
from ..protocol_state_probes.transforms import matrix_from_scene
from .config import PROTOCOLS, TEMPLATE_VERSION, digest, file_digest, load_protocol
from .public_tasks import compile_case, public_verifier, validate_public, verify_raw


def truth_orbit(world):
    if len(world) != 4 or any(type(v) is not int or not 0 <= v <= 99 for v in world):
        raise ValueError("Four legal truth integers required")
    return tuple(sorted(world))


def trend_worlds():
    return [(a, a + d, a + 2 * d, a + 3 * d) for d in range(1, 34) for a in range(100 - 3 * d)]


def solve_public(task):
    """Enumerate all 4*99 repairs independently of generator construction."""
    validate_public(task)
    solutions = []
    for j in range(4):
        for value in range(100):
            if value != task["observed"][j]:
                candidate = list(task["observed"])
                candidate[j] = value
                if public_verifier(task, candidate):
                    solutions.append(candidate)
    return solutions


def validate_historical_index(index):
    if index.get("schema") != "vdt-historical-exclusion-v1" or index.get("complete") is not True:
        raise ValueError("Explicit complete historical exclusion index required")
    sources = index.get("sources")
    if not sources or any(not all(s.get(k) for k in ("path", "sha256", "scope")) for s in sources):
        raise ValueError("Historical metadata source inventory with hashes required")
    if any(len(s["sha256"]) != 64 for s in sources):
        raise ValueError("Invalid historical source digest")
    if type(index.get("exposed_scene_count")) is not int or index["exposed_scene_count"] < 0:
        raise ValueError("Historical exposure count required")
    orbits = set()
    for world in index["truth_orbits"]:
        key = truth_orbit(world)
        if list(key) != world:
            raise ValueError("Historical truth_orbits must be sorted")
        orbits.add(key)
    if index["exposed_scene_count"] < len(orbits):
        raise ValueError("Orbit count exceeds exposed scene inventory")
    return orbits


def _cue(world, family, j, center):
    if family == "trend":
        return {"family": family}
    if family == "duplicate_encoding":
        return {"family": family, "known_index": j, "known_value": world[j]}
    edges = sorted(tuple(sorted((center, leaf))) for leaf in range(4) if leaf != center)
    return {"family": family, "edges": [[i, k, world[i] + world[k]] for i, k in edges]}


def _task(world, family, j, center, split, ordinal, rng, seed):
    observed = list(world)
    replacement = rng.randrange(99)
    observed[j] = replacement + (replacement >= world[j])
    cue = _cue(world, family, j, center)
    H, b = matrix_from_scene({"cue": cue})
    sid = digest([TEMPLATE_VERSION, split, seed, ordinal])[:32]
    operations = ("sum4", "difference_pairs", "range4")
    task = {
        "task_id": sid,
        "base_instance_id": sid,
        "split": split,
        "family": family,
        "observed": observed,
        "H_original": H,
        "b_original": b,
        "legal_domain": [0, 99],
        "operation": operations[ordinal % 3],
        "template_version": TEMPLATE_VERSION,
        "chart_type": ("grouped_bar", "line")[ordinal % 2],
        "interface": "SYMBOLIC_FRESH",
        "image_path": None,
        "image_sha256": None,
    }
    if solve_public(task) != [list(world)] or solve(observed, cue) != [list(world)]:
        raise ValueError("Generator/public solver/independent cue solver contract disagreement")
    audit = {
        "task_id": sid,
        "base_instance_id": sid,
        "split": split,
        "true_world": list(world),
        "corrupted_index": j,
        "center": center,
        "truth_orbit_key": list(truth_orbit(world)),
        "truth_all_in_0_49": all(v <= 49 for v in world),
        "independent_solver_solutions": [list(world)],
        "DPE1": {
            p: dpe(compile_case(task, p)["H_display"], j, compile_case(task, p)["output_order"])
            for p in PROTOCOLS[family]
        },
    }
    return task, audit


def normalize_replay_candidate(candidate):
    candidate = copy.deepcopy(candidate)
    if "task" not in candidate:
        record = candidate["record"]
        scene = record["scene"]
        H, b = matrix_from_scene(scene)
        sid = scene["base_scene_id"]
        candidate["task"] = {
            "task_id": "replay-" + sid,
            "base_instance_id": sid,
            "split": "R_replay",
            "family": record.get("family", scene["cue"]["family"]),
            "observed": scene["observed_world"],
            "H_original": H,
            "b_original": b,
            "legal_domain": [0, 99],
            "operation": record.get("operation", scene["operation"]),
            "template_version": TEMPLATE_VERSION,
            "chart_type": record.get("chart_type", scene.get("chart_type", "grouped_bar")),
            "interface": "SYMBOLIC_FRESH",
            "image_path": None,
            "image_sha256": None,
        }
        source = candidate["source"]
        if record.get("split", scene.get("split")) != "train":
            raise ValueError("Replay historical record itself is not train-only")
        if record.get("interface") != "SYMBOLIC_FRESH":
            raise ValueError("Replay historical record is not symbolic")
        if source["base_scene_id"] != sid:
            raise ValueError("Replay provenance identity mismatch")
    task, source = candidate["task"], candidate["source"]
    if "record" in candidate:
        historical_prompt = candidate["record"]["prompt"]
        prompt = compile_case(task, "O0")["prompt"]
        if any(historical_prompt.get(k) != prompt[k] for k in ("system", "user")):
            raise ValueError("Replay historical input differs from exact O0 template")
    validate_public(task)
    if (
        task["split"] != "R_replay"
        or source.get("split") != "train"
        or source.get("interface") != "SYMBOLIC_FRESH"
        or source.get("protocol") != "O0"
    ):
        raise ValueError("Replay must be historical train-only symbolic O0")
    if source.get("panel") in {"P", "D", "U22", "D48", "V_selection", "E_test"}:
        raise ValueError("Development/evaluation records forbidden from replay")
    if not all(
        source.get(k) for k in ("request_id", "base_scene_id", "artifact_path", "artifact_sha256")
    ):
        raise ValueError("Replay proof requires completed request and hashed source artifact")
    if len(source["artifact_sha256"]) != 64 or source["base_scene_id"] != task["base_instance_id"]:
        raise ValueError("Invalid replay provenance")
    scored = verify_raw(task, "O0", candidate["raw_completion"])
    if not scored["public_verifier_pass"]:
        raise ValueError("Replay raw response did not publicly verify")
    if candidate.get("canonical_vector", scored["canonical_vector"]) != scored["canonical_vector"]:
        raise ValueError("Replay canonical target does not match raw O0 response")
    if solve_public(task) != [scored["canonical_vector"]]:
        raise ValueError("Replay public repair must be unique")
    return {
        "task": task,
        "canonical_vector": scored["canonical_vector"],
        "raw_completion": candidate["raw_completion"],
        "source": source,
        "verified": True,
    }


def select_replay(candidates):
    by_scene = {}
    for candidate in candidates:
        item = normalize_replay_candidate(candidate)
        sid = item["task"]["base_instance_id"]
        if sid in by_scene and by_scene[sid]["canonical_vector"] != item["canonical_vector"]:
            raise ValueError("Conflicting replay targets")
        # Stable selection across duplicate accepted outputs, independent of outcomes.
        if (
            sid not in by_scene
            or item["source"]["request_id"] < by_scene[sid]["source"]["request_id"]
        ):
            by_scene[sid] = item
    ordered = [by_scene[s] for s in sorted(by_scene)]
    selected = []
    for family, count in [("duplicate_encoding", 32), ("cross_series", 16), ("trend", 16)]:
        selected.extend([r for r in ordered if r["task"]["family"] == family][:count])
    selected_ids = {r["task"]["task_id"] for r in selected}
    selected.extend(
        [r for r in ordered if r["task"]["task_id"] not in selected_ids][: 64 - len(selected)]
    )
    if not selected:
        raise ValueError("No proven historical replay target; cannot freeze SFT cohort")
    return sorted(selected, key=lambda r: r["task"]["base_instance_id"]), {
        "available_unique": len(ordered),
        "selected_count": len(selected),
        "shortfall": 64 - len(selected),
        "composition": dict(Counter(r["task"]["family"] for r in selected)),
        "selection": "scene-ID order; preferred32/16/16 then deterministic available fill",
    }


def generate_cohort(historical_index, protocol=None):
    config = protocol or load_protocol()
    seen = validate_historical_index(historical_index)
    pools = [w for w in trend_worlds() if truth_orbit(w) not in seen]
    required = sum(
        config["data"]["new_task_counts"][s] // 2 for s in ("T_train", "V_selection", "E_test")
    )
    capacity = {
        "total_trend_orbits": 1617,
        "historically_excluded_trend_orbits": 1617 - len(pools),
        "remaining_trend_orbits": len(pools),
        "required_trend_orbits": required,
    }
    if len(pools) < required:
        raise ValueError("Insufficient nonconstant trend orbit capacity: " + json.dumps(capacity))
    datasets = {
        s: {"public": [], "audit": []} for s in ("T_train", "V_selection", "E_test", "G_guard")
    }
    # Reserve all scarce trend worlds before cross/duplicate allocation.
    for split in ("T_train", "V_selection", "E_test"):
        seed = config["data"]["generator_seeds"][split]
        rng = random.Random(seed)
        available = [w for w in pools if truth_orbit(w) not in seen]
        rng.shuffle(available)
        count = config["data"]["new_task_counts"][split] // 2
        for n, ascending in enumerate(available[:count]):
            world = list(ascending if rng.randrange(2) else ascending[::-1])
            j, within = n // (count // 4), n % (count // 4)
            task, audit = _task(world, "trend", j, None, split, within * 4 + j, rng, seed)
            task["chart_type"] = ("grouped_bar", "line")[within % 2]
            datasets[split]["public"].append(task)
            datasets[split]["audit"].append(audit)
            seen.add(truth_orbit(world))
    for split in ("T_train", "V_selection", "E_test"):
        seed = config["data"]["generator_seeds"][split]
        rng = random.Random(seed + 1000000)
        half = config["data"]["new_task_counts"][split] // 2
        for center in range(4):
            for j in range(4):
                for within in range(half // 16):
                    for _ in range(100000):
                        world = [rng.randrange(100) for _ in range(4)]
                        if truth_orbit(world) not in seen:
                            break
                    else:
                        raise ValueError("Cross orbit sampling capacity exhausted")
                    ordinal = half + within * 16 + center * 4 + j
                    task, audit = _task(world, "cross_series", j, center, split, ordinal, rng, seed)
                    task["chart_type"] = ("grouped_bar", "line")[within % 2]
                    datasets[split]["public"].append(task)
                    datasets[split]["audit"].append(audit)
                    seen.add(truth_orbit(world))
        for kind in ("public", "audit"):
            datasets[split][kind].sort(key=lambda r: r["task_id"])
    seed = config["data"]["generator_seeds"]["G_guard"]
    rng = random.Random(seed)
    # Corruption-balanced fixed E selection; shared base identities are intentional.
    audit_by_id = {r["task_id"]: r for r in datasets["E_test"]["audit"]}
    for family in ("cross_series", "trend"):
        for j in range(4):
            candidates = [
                t
                for t in datasets["E_test"]["public"]
                if t["family"] == family and audit_by_id[t["task_id"]]["corrupted_index"] == j
            ]
            candidates.sort(key=lambda t: digest([seed, t["task_id"]]))
            for source in candidates[:8]:
                task, audit = copy.deepcopy(source), copy.deepcopy(audit_by_id[source["task_id"]])
                gid = digest(["image-guard", source["task_id"], seed])[:32]
                task.update(task_id=gid, split="G_guard", interface="IMAGE_CUE_FRESH")
                audit.update(task_id=gid, split="G_guard", paired_E_task_id=source["task_id"])
                datasets["G_guard"]["public"].append(task)
                datasets["G_guard"]["audit"].append(audit)
    for n in range(32):
        for _ in range(100000):
            world = [rng.randrange(100) for _ in range(4)]
            if truth_orbit(world) not in seen:
                break
        else:
            raise ValueError("Duplicate guard orbit sampling capacity exhausted")
        task, audit = _task(world, "duplicate_encoding", n % 4, None, "G_guard", n, rng, seed)
        task["chart_type"] = ("grouped_bar", "line")[(n // 4) % 2]
        datasets["G_guard"]["public"].append(task)
        datasets["G_guard"]["audit"].append(audit)
        seen.add(truth_orbit(world))
    return datasets, capacity


def prepare_cohort(output_dir, historical_index, replay_rows, protocol=None, render_images=True):
    """Create a new immutable cohort directory; never overwrite an existing cohort."""
    config = protocol or load_protocol()
    if isinstance(historical_index, (str, Path)):
        historical_index = json.loads(Path(historical_index).read_text())
    if isinstance(replay_rows, (str, Path)):
        replay_rows = [
            json.loads(line) for line in Path(replay_rows).read_text().splitlines() if line.strip()
        ]
    replay, replay_report = select_replay(replay_rows)
    historical_orbits = validate_historical_index(historical_index)
    if any(truth_orbit(r["canonical_vector"]) not in historical_orbits for r in replay):
        raise ValueError("Replay scene absent from supposedly complete historical exclusion index")
    datasets, capacity = generate_cohort(historical_index, config)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    if render_images:
        from ..render_charts import render_chart

        audit_by_id = {r["task_id"]: r for r in datasets["G_guard"]["audit"]}
        for task in datasets["G_guard"]["public"]:
            if task["interface"] == "IMAGE_CUE_FRESH":
                relative = "images/" + task["task_id"] + ".png"
                path = root / relative
                path.parent.mkdir(exist_ok=True)
                render_chart(audit_by_id[task["task_id"]]["true_world"], task["chart_type"], path)
                task.update(image_path=relative, image_sha256=file_digest(path))
    datasets["R_replay"] = {"public": [r["task"] for r in replay], "targets": replay}
    manifest = {
        "schema": "vdt-cohort-v1",
        "status": "FROZEN" if render_images else "BLOCKED_IMAGES_PENDING",
        "cohort_name": config["data"]["cohort_name"],
        "protocol_digest": digest(config),
        "historical_index_digest": digest(historical_index),
        "historical_sources": historical_index["sources"],
        "historical_completeness_scope": historical_index.get("completeness_scope"),
        "historical_limitations": historical_index.get("limitations", []),
        "capacity": capacity,
        "replay": replay_report,
        "files": {},
        "guard_image_scenes_shared_with_E": 64,
        "independent_guard_scenes": 32,
        "distribution": {},
    }
    for split, groups in datasets.items():
        for kind, rows in groups.items():
            name = {
                "public": "tasks_public.jsonl",
                "audit": "audit_only.jsonl",
                "targets": "verified_targets.jsonl",
            }[kind]
            relative = split + "/" + name
            path = root / relative
            path.parent.mkdir(exist_ok=True)
            path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
            manifest["files"][relative] = {"sha256": file_digest(path), "rows": len(rows)}
        audits = groups.get("audit", [])
        by_id = {r["task_id"]: r for r in audits}
        if audits:
            manifest["distribution"][split] = {
                "family_by_j": dict(
                    Counter(
                        t["family"] + ":" + str(by_id[t["task_id"]]["corrupted_index"])
                        for t in groups["public"]
                    )
                ),
                "cross_center_by_j": dict(
                    Counter(
                        str(a["center"]) + ":" + str(a["corrupted_index"])
                        for a in audits
                        if a["center"] is not None
                    )
                ),
                "truth_all_in_0_49": sum(a["truth_all_in_0_49"] for a in audits),
                "count": len(audits),
            }
    manifest["manifest_digest"] = digest(manifest)
    (root / "cohort_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
