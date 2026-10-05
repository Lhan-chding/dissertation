"""Read-only fixed-panel statistics from atomically committed frozen outputs.

No inference, training, outcome-dependent stopping, or completion bootstrap occurs
here. Aliases identify the same random variable. Missing cells remain missing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta

BINARY_METRICS = (
    "pX",
    "pS",
    "pW",
    "pI",
    "pA",
    "pV",
    "parse_four_ints",
    "out_of_domain",
    "copy",
    "anchor_joint",
    "anchor_given_parse",
    "match_V1",
    "match_V2",
    "other",
    "match_V1_given_parse",
    "match_V2_given_parse",
    "F_given_valid",
    "control_success",
    "outside_reference_union",
    "default_forward_output",
    "truncated",
)
EFFECT_METRICS = (
    "pX",
    "pA",
    "pV",
    "pI",
    "copy",
    "anchor_joint",
    "match_V1",
    "match_V2",
    "other",
    "outside_reference_union",
    "C_orig",
    "C_display",
)
CONTINUOUS_METRICS = (
    "C_orig",
    "C_display",
    "C_alg_orig",
    "C_alg_display",
    "B_given_valid",
    "M_given_valid",
)
CORE_PAIRS = (("A1", "O0"), ("B1", "B0"), ("BR", "O0"), ("B0", "O0"))
PRIMARY_CHECKPOINTS = ("S32", "S96", "R4_128", "DIRECT_128")
ALL_CHECKPOINTS = (*PRIMARY_CHECKPOINTS, "REP32", "REP96")


def _json(path: Path):
    return json.loads(path.read_text())


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _stable_seed(*parts: Any) -> int:
    payload = json.dumps(parts, sort_keys=True, default=str).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _write_json(path: Path, value: Any):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["status"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def clopper_pearson(successes: int, total: int, alpha: float = 0.05):
    """Exact two-sided marginal interval; n=0 is unavailable, never [0, 0]."""
    if not isinstance(successes, int) or not isinstance(total, int) or not 0 <= successes <= total:
        raise ValueError("integer binomial counts must satisfy 0 <= successes <= total")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if total == 0:
        return None, None
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    high = (
        1.0
        if successes == total
        else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    )
    return low, high


def family_weights(families: list[str]) -> list[float]:
    """Each present prespecified family gets equal weight; scenes equal within it."""
    counts = Counter(families)
    return [1.0 / (len(counts) * counts[family]) for family in families]


class ContrastEngine:
    """Linear contrasts with shared alias variables and paired-scene bootstrap.

    A scene contains terms [(independent_evidence_cell_key, coefficient), ...].
    A cell key includes metric and checkpoint. Identical keys are consolidated
    before MC variance or posterior evaluation. Bootstrap resamples scene contrasts
    in family/structure strata, never independently resampling their endpoints.
    """

    def __init__(
        self, samples, *, posterior_draws=20000, bootstrap_draws=5000, seed=20261005, expected=None
    ):
        self.samples = samples
        self.posterior_draws = posterior_draws
        self.bootstrap_draws = bootstrap_draws
        self.seed = seed
        self.expected = expected if expected is not None else {}
        self._posterior_cache = OrderedDict()

    def _posterior(self, key, prior, support):
        cache_key = (key, prior, tuple(support), self.posterior_draws)
        if cache_key in self._posterior_cache:
            self._posterior_cache.move_to_end(cache_key)
            return self._posterior_cache[cache_key]
        values = np.asarray(self.samples[key], dtype=float)
        rng = np.random.default_rng(_stable_seed(self.seed, key, prior, support))
        if tuple(support) == (0.0, 1.0):
            successes = int(values.sum())
            result = rng.beta(
                successes + prior, len(values) - successes + prior, self.posterior_draws
            )
        else:
            counts = np.asarray([np.sum(values == outcome) for outcome in support])
            if int(counts.sum()) != len(values):
                raise ValueError("Observed value outside registered categorical support")
            result = rng.dirichlet(counts + prior, self.posterior_draws) @ np.asarray(support)
        self._posterior_cache[cache_key] = result
        if len(self._posterior_cache) > 128:
            self._posterior_cache.popitem(last=False)
        return result

    def estimate(self, scenes, *, binary=True, support=None):
        empty = dict(
            estimate=None,
            estimate_pp=None,
            mc_se=None,
            jeffreys_low=None,
            jeffreys_high=None,
            uniform_low=None,
            uniform_high=None,
            scene_bootstrap_low=None,
            scene_bootstrap_high=None,
            scene_count=len(scenes),
            independent_cells=0,
            posterior_draws=self.posterior_draws,
            scene_bootstrap_draws=self.bootstrap_draws,
            prior_direction_sensitive=None,
        )
        if not scenes:
            return dict(
                empty,
                status="N/A_EMPTY_REGISTERED_STRATUM",
                missing_cells=[],
                scene_bootstrap_status="N/A",
            )
        scene_ids = [scene["scene"] for scene in scenes]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("one contrast may contain each base_scene only once")
        required = {key for scene in scenes for key, coefficient in scene["terms"] if coefficient}
        missing = sorted(key for key in required if not len(self.samples.get(key, [])))
        if missing:
            return dict(
                empty,
                status="MISSING_ENDPOINTS",
                missing_cells=missing,
                complete_scene_count=sum(
                    all(
                        len(self.samples.get(key, []))
                        for key, coefficient in scene["terms"]
                        if coefficient
                    )
                    for scene in scenes
                ),
                scene_bootstrap_status="MISSING_ENDPOINTS",
            )
        weights = family_weights([scene["family"] for scene in scenes])
        coefficients = defaultdict(float)
        per_scene = []
        for scene, weight in zip(scenes, weights, strict=True):
            local = defaultdict(float)
            for key, coefficient in scene["terms"]:
                local[key] += coefficient
                coefficients[key] += coefficient * weight
            per_scene.append(
                sum(
                    coefficient * float(np.mean(self.samples[key]))
                    for key, coefficient in local.items()
                    if coefficient
                )
            )
        coefficients = {key: value for key, value in coefficients.items() if abs(value) > 1e-15}
        estimate = float(np.dot(weights, per_scene))
        variance_terms = [
            coefficient**2 * float(np.var(self.samples[key], ddof=1)) / len(self.samples[key])
            if len(self.samples[key]) > 1
            else None
            for key, coefficient in coefficients.items()
        ]
        mc_se = None if None in variance_terms else math.sqrt(sum(variance_terms))
        partial = any(
            len(self.samples[key]) < self.expected.get(key, len(self.samples[key]))
            for key in required
        )
        result = dict(
            empty,
            status="PARTIAL_FIXED_N" if partial else "COMPLETE",
            estimate=estimate,
            estimate_pp=100 * estimate,
            mc_se=mc_se,
            missing_cells=[],
            independent_cells=len(coefficients),
            complete_scene_count=len(scenes),
            posterior_model="not_binomial_no_Beta_claim",
            prior_direction_sensitive=None,
        )
        if binary or support is not None:
            support = (0.0, 1.0) if support is None else tuple(support)
            result["posterior_model"] = (
                "independent_cell_Beta"
                if support == (0.0, 1.0)
                else "independent_cell_Dirichlet_signed_preference"
            )
            for prior, label in ((0.5, "jeffreys"), (1.0, "uniform")):
                draws = np.zeros(self.posterior_draws)
                for key, coefficient in coefficients.items():
                    draws += coefficient * self._posterior(key, prior, support)
                result[label + "_low"], result[label + "_high"] = map(
                    float, np.quantile(draws, [0.025, 0.975])
                )

            def direction(label):
                return (
                    1 if result[label + "_low"] > 0 else (-1 if result[label + "_high"] < 0 else 0)
                )

            result["prior_direction_sensitive"] = direction("jeffreys") != direction("uniform")
            result["conditional_direction"] = {1: "POSITIVE", -1: "NEGATIVE", 0: "UNRESOLVED"}[
                direction("jeffreys")
            ]
        if len(scenes) < 3:
            result["scene_bootstrap_status"] = "LESS_THAN_3_SCENES_USE_PROMPT_ROWS"
        else:
            groups = defaultdict(list)
            for index, scene in enumerate(scenes):
                groups[(scene["family"], scene.get("stratum", "all"))].append(index)
            draws = np.zeros(self.bootstrap_draws)
            for group, indices in sorted(groups.items()):
                group_weight = sum(weights[index] for index in indices)
                rng = np.random.default_rng(
                    _stable_seed(self.seed, "paired_scene_only", group, sorted(scene_ids))
                )
                sampled = rng.integers(0, len(indices), size=(self.bootstrap_draws, len(indices)))
                group_values = np.asarray([per_scene[index] for index in indices])
                draws += group_weight * group_values[sampled].mean(axis=1)
            result["scene_bootstrap_low"], result["scene_bootstrap_high"] = map(
                float, np.quantile(draws, [0.025, 0.975])
            )
            result["scene_bootstrap_status"] = "DESCRIPTIVE_SCENE_ONLY"
        return result


def read_committed_rows(run_path, manifest=None):
    """Only COMMIT-referenced chunks; immutable collision/hash checks fail closed."""
    root = Path(run_path)
    rows, keys, draws, receipt_count, smoke_count = [], {}, set(), 0, 0
    bindings = {}
    if manifest is not None:
        from .protocol import digest, sample_identity

        for binding_path in (root / "checkpoint_bindings").glob("*.json"):
            binding = _json(binding_path)
            expected_identity = digest(
                {
                    "run_identity": manifest["identity"],
                    "checkpoint": binding["checkpoint_record"],
                    "runtime_sha256": binding["runtime_sha256"],
                }
            )
            if (
                binding["run_identity"] != manifest["identity"]
                or binding["identity"] != expected_identity
            ):
                raise ValueError(f"Checkpoint binding identity mismatch: {binding_path}")
            bindings[binding["checkpoint"]] = binding
    for receipt_path in sorted((root / "raw").glob("*/*/*/chunks/*.COMMIT.json")):
        commit = _json(receipt_path)
        file_name = commit["file"]
        if Path(file_name).name != file_name:
            raise ValueError(f"Unsafe chunk COMMIT path: {receipt_path}")
        chunk_path = receipt_path.parent / file_name
        content = chunk_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != commit["sha256"]:
            raise ValueError(f"Committed chunk hash mismatch: {chunk_path}")
        chunk = json.loads(content)
        chunk_rows = chunk["rows"]
        if (
            len(chunk_rows) != commit["n_rows"]
            or [row["sample_key"] for row in chunk_rows] != commit["sample_keys"]
        ):
            raise ValueError(f"Committed chunk row identity mismatch: {chunk_path}")
        receipt_count += 1
        for row in chunk_rows:
            if manifest is not None:
                binding = bindings.get(row["checkpoint"])
                if binding is None or chunk.get("identity") != binding["identity"]:
                    raise ValueError(f"Committed chunk checkpoint binding mismatch: {chunk_path}")
                stable_key, stable_seed = sample_identity(
                    manifest["protocol_id"],
                    row["checkpoint"],
                    row["canonical_case_owner"],
                    row["role"],
                    row["draw_index"],
                )
                if row["sample_key"] != stable_key or row.get("seed") != stable_seed:
                    raise ValueError(f"Committed chunk seed/sample identity mismatch: {chunk_path}")
            key = row["sample_key"]
            identity = (
                row["checkpoint"],
                row["canonical_case_owner"],
                row["role"],
                row["draw_index"],
            )
            if key in keys or identity in draws:
                raise ValueError(f"Duplicate/conflicting committed sample identity: {key}")
            keys[key] = row
            draws.add(identity)
            if row["role"] == "smoke":
                smoke_count += 1
            elif row["role"] == "frozen_probe":
                rows.append(row)
            else:
                raise ValueError(f"Unregistered output role: {row['role']}")
    return rows, {
        "committed_chunks": receipt_count,
        "smoke_outputs_excluded": smoke_count,
        "main_outputs": len(rows),
        "integrity": "COMMIT_SHA256_AND_UNIQUE_SAMPLE_AND_DRAW",
    }


def _prediction_value(case, audit, prediction, features, metric, b1_pair=None):
    parsed = bool(features.get("parse_four_ints", False))
    valid = bool(features.get("domain_valid", features.get("in_domain", False)))
    event = features.get("event", features.get("E"))
    control = case["panel"].startswith("CONTROL_")
    if metric in ("pX", "pS", "pW", "pI", "pA", "pV"):
        if case["kind"] != "repair":
            return None
        return float(
            event
            in {
                "pX": {"X"},
                "pS": {"S"},
                "pW": {"W"},
                "pI": {"I"},
                "pA": {"X", "S"},
                "pV": {"X", "S", "W"},
            }[metric]
        )
    if metric == "parse_four_ints":
        return float(parsed)
    if metric == "out_of_domain":
        return float(parsed and not valid)
    if metric == "copy":
        return float(parsed and bool(features.get("copy_parsed", features.get("copy", False))))
    if metric in ("anchor_joint", "anchor_given_parse"):
        if metric == "anchor_given_parse" and not parsed:
            return None
        return float(
            parsed and bool(features.get("anchor_all_parsed", features.get("anchor_all", False)))
        )
    if metric in ("match_V1", "match_V2", "match_V1_given_parse", "match_V2_given_parse"):
        if metric.endswith("given_parse") and not parsed:
            return None
        return float(parsed and bool(features.get(metric.split("_given")[0], False)))
    if metric in ("other", "outside_reference_union"):
        if prediction is None:
            return None
        outside = not any(
            features.get(key, False) for key in ("copy_parsed", "match_V1", "match_V2")
        )
        return float((parsed and outside) if metric == "other" else (not parsed or outside))
    if metric in ("C_orig", "C_display"):
        value = features.get(metric)
        return float(value) if value is not None else (0.0 if not valid else None)
    if metric in ("C_alg_orig", "C_alg_display"):
        return float(features[metric]) if parsed and features.get(metric) is not None else None
    if metric in ("F_given_valid", "B_given_valid", "M_given_valid"):
        name = metric[0]
        return float(features[name]) if valid and features.get(name) is not None else None
    if metric == "control_success":
        return (
            float(bool(features[metric])) if control and features.get(metric) is not None else None
        )
    if metric == "default_forward_output":
        if not control:
            return None
        expected = audit.get("control_expected_canonical") or audit.get("truth_world")
        return float(
            parsed
            and case.get("output_order") != [0, 1, 2, 3]
            and features.get("emitted_world") == expected
        )
    if metric == "truncated":
        return float(bool(features.get("truncated", False)))
    if metric.startswith("b1_"):
        if not b1_pair:
            return None
        _, variant, target = metric.split("_")
        old_prediction, new_prediction = b1_pair
        world = features.get("canonical_world")
        old = old_prediction["programs"][variant]["canonical"]
        new = new_prediction["programs"][variant]["canonical"]
        old_match = parsed and world == old
        new_match = parsed and world == new
        if target == "preference":
            return float(new_match) - float(old_match)
        return float(new_match if target == "new" else old_match)
    if metric.startswith("relation_"):
        _, relation_kind, mode, index = metric.split("_")
        if mode == "conditional" and not parsed:
            return None
        flags = features.get("relation_flags_" + relation_kind)
        return float(parsed and flags is not None and bool(flags[int(index)]))
    if metric.startswith("local_coordinate_"):
        index = metric.rsplit("_", 1)[-1]
        flags = (features.get("local_relation_flags_by_coordinate") or {}).get(index)
        return float(parsed and flags is not None and all(flags))
    if metric.startswith("anchor_coordinate_"):
        index = int(metric.rsplit("_", 1)[-1])
        mask = features.get("anchor_mask")
        return float(parsed and mask is not None and bool(mask[index]))
    raise KeyError(metric)


class RunAnalysis:
    def __init__(self, root, *, posterior_draws, bootstrap_draws, seed):
        self.root = Path(root)
        self.protocol = _json(self.root / "protocol.json")
        self.manifest = _json(self.root / "execution_manifest.json")
        from .protocol import digest, file_hash, validate_protocol

        if self.manifest.get("identity") != digest(
            {key: value for key, value in self.manifest.items() if key != "identity"}
        ):
            raise ValueError("Execution manifest identity is corrupt")
        verification = self.manifest["verification"]
        if not verification.get("recompiled_from_prepared") or not verification.get(
            "all_fields_match"
        ):
            raise ValueError("Run lacks verified prepared-data recompilation")
        for name, expected_hash in self.manifest["file_hashes"].items():
            if Path(name).name != name or file_hash(self.root / name) != expected_hash:
                raise ValueError(f"Frozen run artifact changed: {name}")
        validate_protocol(self.protocol)
        # Read-only reports remain portable: remote runtime/checkpoint payload paths
        # are provenance here, never reopened or interpreted as local paths.
        self.cases = {case["case_id"]: case for case in _jsonl(self.root / "cases.jsonl")}
        self.audits = {
            audit["case_id"]: audit for audit in _jsonl(self.root / "audit_labels.jsonl")
        }
        self.predictions = {
            prediction["case_id"]: prediction
            for prediction in _jsonl(self.root / "program_predictions.jsonl")
        }
        self.aliases = _json(self.root / "alias_map.json")
        self.jobs = self.manifest["jobs"]
        exposure_path = self.root / "U22_EXPOSURE_AUDIT.json"
        self.exposure_audit = (
            _json(exposure_path)
            if exposure_path.exists()
            else {"status": "U22_EXPOSURE_UNVERIFIED"}
        )
        self.expected = {
            (job["checkpoint_id"], self.owner(job["case_id"])): job["draws"] for job in self.jobs
        }
        self.rows, self.receipts = read_committed_rows(self.root, self.manifest)
        self.cells = defaultdict(list)
        for row in self.rows:
            key = (row["checkpoint"], row["canonical_case_owner"])
            if key not in self.expected:
                raise ValueError(f"Unscheduled committed output cell: {key}")
            if row["case_id"] != key[1] or row["panel"] != self.cases[key[1]]["panel"]:
                raise ValueError(f"Committed output metadata differs from frozen case: {key}")
            self.cells[key].append(row)
        for key, rows in self.cells.items():
            indices = sorted(row["draw_index"] for row in rows)
            if any(index < 0 or index >= self.expected[key] for index in indices):
                raise ValueError(f"Draw outside preregistered fixed budget: {key}")
            if len(indices) != len(set(indices)):
                raise ValueError(f"Duplicate draw: {key}")
        self.by_scene = defaultdict(dict)
        for case in self.cases.values():
            if case["kind"] == "repair" and not case["panel"].startswith("CONTROL_"):
                self.by_scene[(case["panel"], case["base_scene_id"])][case["protocol"]] = case[
                    "case_id"
                ]
        self.samples, self.sample_expected = {}, {}
        self.engine = ContrastEngine(
            self.samples,
            posterior_draws=posterior_draws,
            bootstrap_draws=bootstrap_draws,
            seed=seed,
            expected=self.sample_expected,
        )

    def panel_role(self, panel):
        if panel != "U22":
            return "execution_control" if panel.startswith("CONTROL_") else "development_diagnostic"
        if self.exposure_audit.get(
            "status"
        ) == "CONTAMINATED_DOWNGRADED" or self.exposure_audit.get("downgraded_to_development"):
            return "development_diagnostic"
        if self.exposure_audit.get("status") in (
            "VERIFIED_UNTOUCHED",
            "UNTOUCHED_VERIFIED",
            "SAFE",
        ):
            return "untouched_outcome_replication"
        return "exposure_unverified"

    def owner(self, case_id):
        visited = set()
        while self.aliases.get(case_id, case_id) != case_id:
            if case_id in visited:
                raise ValueError("Alias cycle")
            visited.add(case_id)
            case_id = self.aliases[case_id]
        return case_id

    def metric_key(self, checkpoint, case_id, metric):
        owner = self.owner(case_id)
        key = "|".join((checkpoint, owner, metric))
        if key not in self.samples:
            case = self.cases[owner]
            scene_cases = self.by_scene.get((case["panel"], case["base_scene_id"]), {})
            b1_pair = None
            if "B0" in scene_cases and "B1" in scene_cases:
                b1_pair = (self.predictions[scene_cases["B0"]], self.predictions[scene_cases["B1"]])
            values = []
            for row in self.cells.get((checkpoint, owner), []):
                features = dict(row["features"])
                features.setdefault("truncated", row.get("stop_reason") == "length")
                value = _prediction_value(
                    case, self.audits[owner], self.predictions.get(owner), features, metric, b1_pair
                )
                if value is not None:
                    values.append(value)
            self.samples[key] = values
            self.sample_expected[key] = self.expected.get((checkpoint, owner), 0)
        return key

    def scene_specs(self, panel, endpoints, metric, selector=None):
        scenes = []
        for (scene_panel, scene_id), protocols in sorted(self.by_scene.items()):
            if scene_panel != panel or any(
                protocol not in protocols for _, protocol, _ in endpoints
            ):
                continue
            base = self.cases[next(iter(protocols.values()))]
            if selector and not selector(base, protocols):
                continue
            terms = [
                (self.metric_key(checkpoint, protocols[protocol], metric), coefficient)
                for checkpoint, protocol, coefficient in endpoints
            ]
            anchor_case = protocols[endpoints[0][1]]
            audit = self.audits[anchor_case]
            stratum = f"{int(audit['DPE1_old'])}->{int(audit['DPE1_new'])}"
            scenes.append(
                {"scene": scene_id, "family": base["family"], "stratum": stratum, "terms": terms}
            )
        return scenes

    def effect(self, panel, endpoints, metric, selector=None, **metadata):
        scenes = self.scene_specs(panel, endpoints, metric, selector)
        result = self.engine.estimate(
            scenes,
            binary=metric in BINARY_METRICS
            or (metric.startswith("b1_") and not metric.endswith("preference")),
            support=(-1.0, 0.0, 1.0) if metric.endswith("preference") else None,
        )
        endpoint_values = {}
        for checkpoint, protocol, _ in endpoints:
            endpoint_specs = self.scene_specs(panel, [(checkpoint, protocol, 1)], metric)
            # Evaluate endpoints on exactly the contrast's scene set, never a larger pool.
            allowed = {scene["scene"] for scene in scenes}
            endpoint_specs = [scene for scene in endpoint_specs if scene["scene"] in allowed]
            values = []
            for scene in endpoint_specs:
                key = scene["terms"][0][0]
                values.append(float(np.mean(self.samples[key])) if self.samples[key] else None)
            endpoint_values[checkpoint + ":" + protocol] = (
                float(np.dot(family_weights([scene["family"] for scene in endpoint_specs]), values))
                if values and None not in values
                else None
            )
        return dict(
            panel=panel,
            metric=metric,
            effective_split_role=self.panel_role(panel),
            U22_exposure_audit_status=self.exposure_audit["status"] if panel == "U22" else None,
            **metadata,
            endpoints=endpoint_values,
            **result,
        )

    def prompt_metrics(self):
        table, controls, mass, atoms = [], [], [], []
        for checkpoint in sorted({job["checkpoint_id"] for job in self.jobs}):
            for case_id, case in sorted(self.cases.items()):
                owner = self.owner(case_id)
                if (checkpoint, owner) not in self.expected:
                    continue
                rows = self.cells.get((checkpoint, owner), [])
                expected_n = self.expected[(checkpoint, owner)]
                audit, prediction = self.audits[case_id], self.predictions.get(case_id, {})
                programs = prediction.get("programs", {})
                metadata = dict(
                    checkpoint=checkpoint,
                    panel=case["panel"],
                    case_id=case_id,
                    canonical_case_owner=owner,
                    prompt_identity=case.get("prompt_identity"),
                    base_scene_id=case["base_scene_id"],
                    family=case["family"],
                    protocol=case["protocol"],
                    kind=case["kind"],
                    effective_split_role=self.panel_role(case["panel"]),
                    U22_exposure_audit_status=self.exposure_audit["status"]
                    if case["panel"] == "U22"
                    else None,
                    original_split=case.get("original_split"),
                    is_alias=case_id != owner,
                    observed_outputs=len(rows),
                    expected_outputs=expected_n,
                    DPE1_old=audit.get("DPE1_old"),
                    DPE1_new=audit.get("DPE1_new"),
                    program_V1_is_truth=programs.get("V1", {}).get("canonical")
                    == audit.get("truth_world")
                    if programs
                    else None,
                    program_V2_is_truth=programs.get("V2", {}).get("canonical")
                    == audit.get("truth_world")
                    if programs
                    else None,
                    program_V1_is_copy=programs.get("V1", {}).get("canonical")
                    == prediction.get("copy_canonical")
                    if programs
                    else None,
                    program_V2_is_copy=programs.get("V2", {}).get("canonical")
                    == prediction.get("copy_canonical")
                    if programs
                    else None,
                    V1_prediction_in_domain=programs.get("V1", {}).get("in_domain"),
                )
                metrics = list(BINARY_METRICS + CONTINUOUS_METRICS)
                if case["kind"] != "repair":
                    metrics = [
                        "parse_four_ints",
                        "out_of_domain",
                        "copy",
                        "control_success",
                        "default_forward_output",
                        "truncated",
                        "C_orig",
                        "C_display",
                        "C_alg_orig",
                        "C_alg_display",
                    ]
                for kind in ("orig", "display"):
                    field = "H_original" if kind == "orig" else "H_display"
                    for index in range(len(case.get(field, []))):
                        metrics.extend(
                            [
                                f"relation_{kind}_joint_{index}",
                                f"relation_{kind}_conditional_{index}",
                            ]
                        )
                metrics += [
                    f"anchor_coordinate_{index}"
                    for index in prediction.get("anchor_coordinates", [])
                ]
                positions = {
                    coordinate: index for index, coordinate in enumerate(case["output_order"])
                }
                closing = {
                    max((k for k, coefficient in enumerate(row) if coefficient), key=positions.get)
                    for row in case.get("H_display", [])
                }
                metrics += [f"local_coordinate_{index}" for index in sorted(closing)]
                for metric in metrics:
                    values = self.samples[self.metric_key(checkpoint, case_id, metric)]
                    n = len(values)
                    binary = metric in BINARY_METRICS or metric.startswith(
                        ("relation_", "anchor_coordinate_", "local_coordinate_")
                    )
                    successes = int(sum(values)) if binary else None
                    low, high = clopper_pearson(successes, n) if binary else (None, None)
                    item = dict(
                        metadata,
                        metric=metric,
                        n=n,
                        successes=successes,
                        estimate=float(np.mean(values)) if n else None,
                        cp95_low=low,
                        cp95_high=high,
                        zero_count_one_sided95_upper=1 - 0.05 ** (1 / n)
                        if binary and n and successes == 0
                        else None,
                        status="NOT_OBSERVED"
                        if not n
                        else ("COMPLETE" if len(rows) == expected_n else "PARTIAL_FIXED_N"),
                        denominator="parse_four_ints"
                        if "given_parse" in metric or "conditional" in metric or "C_alg" in metric
                        else (
                            "domain_valid" if "given_valid" in metric else "all_committed_outputs"
                        ),
                    )
                    if not case["panel"].startswith("CONTROL_"):
                        table.append(item)
                    else:
                        controls.append(item)
                if case_id != owner:
                    continue
                event_counts = Counter(
                    row["features"].get("event", row["features"].get("E")) for row in rows
                )
                matches = Counter(row["features"].get("match_mask") for row in rows)
                invalid_events = [
                    event
                    for event in event_counts
                    if case["kind"] == "repair" and event not in ("X", "S", "W", "I")
                ]
                nesting_failures = sum(
                    row["features"].get("event") in ("X", "S", "W")
                    and not row["features"].get(
                        "domain_valid", row["features"].get("in_domain", False)
                    )
                    for row in rows
                )
                mask_failures = sum(
                    value is not None and (type(value) is not int or not 0 <= value <= 7)
                    for value in matches
                )
                # Unparseable outcomes form their own null atom; the union complement includes them.
                counts = Counter()
                for row in rows:
                    features = row["features"]
                    atom = {
                        key: features.get(key)
                        for key in (
                            "event",
                            "parse_four_ints",
                            "domain_valid",
                            "match_mask",
                            "edit_mask",
                            "anchor_mask",
                            "relation_flags_orig",
                            "relation_flags_display",
                            "F",
                            "B",
                            "M",
                        )
                    }
                    counts[json.dumps(atom, sort_keys=True)] += 1
                for atom, count in sorted(counts.items()):
                    atoms.append(
                        dict(
                            checkpoint=checkpoint,
                            panel=case["panel"],
                            case_id=case_id,
                            base_scene_id=case["base_scene_id"],
                            family=case["family"],
                            protocol=case["protocol"],
                            kind=case["kind"],
                            atom_json=atom,
                            count=count,
                            denominator=len(rows),
                            probability=count / len(rows),
                        )
                    )
                mass.append(
                    dict(
                        checkpoint=checkpoint,
                        panel=case["panel"],
                        case_id=case_id,
                        n=len(rows),
                        event_counts={str(key): value for key, value in event_counts.items()},
                        match_mask_counts={str(key): value for key, value in matches.items()},
                        atoms_total=sum(counts.values()),
                        probability_total=sum(counts.values()) / len(rows) if rows else None,
                        nesting_failures=nesting_failures,
                        mask_failures=mask_failures,
                        invalid_events=invalid_events,
                        status="NOT_OBSERVED"
                        if not rows
                        else (
                            "PASS"
                            if not invalid_events
                            and not nesting_failures
                            and not mask_failures
                            and sum(counts.values()) == len(rows)
                            else "FAIL"
                        ),
                    )
                )
        return table, controls, mass, atoms

    def effect_tables(self):
        paired, transitions, t1, t2, factorial, b1 = [], [], [], [], [], []
        checkpoints = sorted({job["checkpoint_id"] for job in self.jobs})
        panels = sorted(
            {
                case["panel"]
                for case in self.cases.values()
                if case["kind"] == "repair" and not case["panel"].startswith("CONTROL_")
            }
        )
        for checkpoint in checkpoints:
            available_panels = {
                self.cases[owner]["panel"] for cp, owner in self.expected if cp == checkpoint
            }
            for panel in panels:
                if panel not in available_panels:
                    continue
                for changed, baseline in CORE_PAIRS:
                    endpoints = [(checkpoint, changed, 1), (checkpoint, baseline, -1)]
                    comparison = changed + "-" + baseline
                    for metric in EFFECT_METRICS:
                        paired.append(
                            self.effect(
                                panel,
                                endpoints,
                                metric,
                                comparison=comparison,
                                checkpoint=checkpoint,
                                scope="family_balanced",
                            )
                        )
                        families = sorted(
                            {
                                self.cases[protocols[changed]]["family"]
                                for (p, _), protocols in self.by_scene.items()
                                if p == panel and changed in protocols
                            }
                        )
                        for family in families:
                            paired.append(
                                self.effect(
                                    panel,
                                    endpoints,
                                    metric,
                                    lambda case, _, family=family: case["family"] == family,
                                    comparison=comparison,
                                    checkpoint=checkpoint,
                                    scope="family",
                                    family=family,
                                )
                            )
                        # All paired scene rows remain visible, including errors and reversals.
                        for (p, scene), protocols in sorted(self.by_scene.items()):
                            if p != panel or changed not in protocols or baseline not in protocols:
                                continue
                            paired.append(
                                self.effect(
                                    panel,
                                    endpoints,
                                    metric,
                                    lambda case, _, scene=scene: case["base_scene_id"] == scene,
                                    comparison=comparison,
                                    checkpoint=checkpoint,
                                    scope="scene",
                                    base_scene_id=scene,
                                )
                            )
                        if changed != "B0":
                            for old, new in ((0, 1), (1, 0), (0, 0), (1, 1)):

                                def select_transition(
                                    case, protocols, old=old, new=new, changed=changed
                                ):
                                    audit = self.audits[protocols[changed]]
                                    return (
                                        int(audit["DPE1_old"]) == old
                                        and int(audit["DPE1_new"]) == new
                                    )

                                transitions.append(
                                    self.effect(
                                        panel,
                                        endpoints,
                                        metric,
                                        select_transition,
                                        comparison=comparison,
                                        checkpoint=checkpoint,
                                        scope="DPE_transition",
                                        DPE_transition=f"{old}->{new}",
                                    )
                                )
                self._b1_rows(checkpoint, panel, b1)
        for label, first, second, target in (
            ("T1", "S96", "S32", t1),
            ("T2", "DIRECT_128", "R4_128", t2),
            ("REP_T1", "REP96", "REP32", t1),
        ):
            for panel in panels:
                if label == "REP_T1" and panel != "D48":
                    continue
                for changed, baseline in CORE_PAIRS[:3]:
                    endpoints = [
                        (first, changed, 1),
                        (first, baseline, -1),
                        (second, changed, -1),
                        (second, baseline, 1),
                    ]
                    for metric in EFFECT_METRICS:
                        target.append(
                            self.effect(
                                panel,
                                endpoints,
                                metric,
                                interaction=label,
                                comparison=changed + "-" + baseline,
                                scope="family_balanced",
                            )
                        )
                        families = sorted(
                            {
                                self.cases[protocols[changed]]["family"]
                                for (p, _), protocols in self.by_scene.items()
                                if p == panel and changed in protocols
                            }
                        )
                        for family in families:
                            target.append(
                                self.effect(
                                    panel,
                                    endpoints,
                                    metric,
                                    lambda case, _, family=family: case["family"] == family,
                                    interaction=label,
                                    comparison=changed + "-" + baseline,
                                    scope="family",
                                    family=family,
                                )
                            )
                        for transition in ("0->1", "1->0", "0->0", "1->1"):

                            def selected(case, protocols, transition=transition, changed=changed):
                                audit = self.audits[protocols[changed]]
                                return (
                                    f"{int(audit['DPE1_old'])}->{int(audit['DPE1_new'])}"
                                    == transition
                                )

                            target.append(
                                self.effect(
                                    panel,
                                    endpoints,
                                    metric,
                                    selected,
                                    interaction=label,
                                    comparison=changed + "-" + baseline,
                                    scope="DPE_transition",
                                    DPE_transition=transition,
                                )
                            )
        factorial_specs = {
            "L00": [("L00", 1)],
            "L01": [("L01", 1)],
            "L10": [("L10", 1)],
            "L11": [("L11", 1)],
            "input_main_effect": [("L10", 0.5), ("L00", -0.5), ("L11", 0.5), ("L01", -0.5)],
            "output_main_effect": [("L01", 0.5), ("L00", -0.5), ("L11", 0.5), ("L10", -0.5)],
            "input_at_forward_output": [("L10", 1), ("L00", -1)],
            "input_at_reverse_output": [("L11", 1), ("L01", -1)],
            "output_at_forward_input": [("L01", 1), ("L00", -1)],
            "output_at_reverse_input": [("L11", 1), ("L10", -1)],
            "input_output_interaction": [("L11", 1), ("L10", -1), ("L01", -1), ("L00", 1)],
            "naming_forward_L00-O0": [("L00", 1), ("O0", -1)],
            "naming_reverse_L01-A1": [("L01", 1), ("A1", -1)],
        }
        for checkpoint in ("S32", "S96"):
            for name, specification in factorial_specs.items():
                endpoints = [
                    (checkpoint, protocol, coefficient) for protocol, coefficient in specification
                ]
                for metric in EFFECT_METRICS:
                    factorial.append(
                        self.effect(
                            "D48",
                            endpoints,
                            metric,
                            comparison=name,
                            checkpoint=checkpoint,
                            scope="family_balanced",
                        )
                    )
                    for family in ("cross_series", "trend"):
                        factorial.append(
                            self.effect(
                                "D48",
                                endpoints,
                                metric,
                                lambda case, _, family=family: case["family"] == family,
                                comparison=name,
                                checkpoint=checkpoint,
                                scope="family",
                                family=family,
                            )
                        )
        return {
            "paired_protocol_effects": paired,
            "DPE_transition_effects": transitions,
            "T1_interactions": t1,
            "T2_interactions": t2,
            "input_output_factorial": factorial,
            "B1_exact_error_shifts": b1,
        }

    def _b1_rows(self, checkpoint, panel, output):
        registered = []
        for (p, scene), protocols in sorted(self.by_scene.items()):
            if p != panel or "B1" not in protocols or "B0" not in protocols:
                continue
            old, new = self.predictions[protocols["B0"]], self.predictions[protocols["B1"]]
            audit = self.audits[protocols["B1"]]
            case = self.cases[protocols["B1"]]
            diagnostic = new.get("b1_diagnostic", {})
            changed = {
                v: old["programs"][v]["canonical"] != new["programs"][v]["canonical"]
                for v in ("V1", "V2")
            }
            category = (
                "both"
                if all(changed.values())
                else "V1_only"
                if changed["V1"]
                else "V2_only"
                if changed["V2"]
                else "unchanged"
            )
            first_leaf_risk = diagnostic.get("first_leaf_risk")
            if first_leaf_risk is None:
                coordinates = [
                    set(index for index, coefficient in enumerate(row) if coefficient)
                    for row in case["H_original"]
                ]
                centers = set.intersection(*coordinates) if coordinates else set()
                first_leaf = min(set(range(4)) - centers) if len(centers) == 1 else None
                first_leaf_risk = audit["corrupted_coordinate"] == first_leaf
            registered.append((scene, changed, bool(first_leaf_risk), old, new, audit))
            for variant in ("V1", "V2"):
                metric = f"b1_{variant}_preference"
                endpoints = [(checkpoint, "B1", 1), (checkpoint, "B0", -1)]
                result = self.effect(
                    panel,
                    endpoints,
                    metric,
                    lambda case, _, scene=scene: case["base_scene_id"] == scene,
                    checkpoint=checkpoint,
                    scope="scene",
                    base_scene_id=scene,
                    variant=variant,
                    comparison="(B1_new-B1_old)-(B0_new-B0_old)",
                )
                result.update(
                    old_prediction=old["programs"][variant]["canonical"],
                    new_prediction=new["programs"][variant]["canonical"],
                    prediction_changed=changed[variant],
                    change_category=category,
                    first_leaf_risk=bool(first_leaf_risk),
                    new_prediction_in_domain=new["programs"][variant]["in_domain"],
                    new_prediction_integer=new["programs"][variant]["integer_prediction"],
                    old_prediction_is_truth=old["programs"][variant]["canonical"]
                    == audit["truth_world"],
                    new_prediction_is_truth=new["programs"][variant]["canonical"]
                    == audit["truth_world"],
                    DPE1_old=audit["DPE1_old"],
                    DPE1_new=audit["DPE1_new"],
                )
                for protocol in ("B0", "B1"):
                    for label in ("old", "new"):
                        values = self.samples[
                            self.metric_key(
                                checkpoint, protocols[protocol], f"b1_{variant}_{label}"
                            )
                        ]
                        count, n = int(sum(values)), len(values)
                        prefix = protocol + "_match_" + label
                        result[prefix + "_count"], result[prefix + "_n"] = count, n
                        result[prefix + "_probability"] = count / n if n else None
                        result[prefix + "_cp95_low"], result[prefix + "_cp95_high"] = (
                            clopper_pearson(count, n)
                        )
                if not changed[variant] and result["estimate"] is not None:
                    # Old==new is exactly the same event, not two categorical outcomes.
                    for key in (
                        "estimate",
                        "estimate_pp",
                        "mc_se",
                        "jeffreys_low",
                        "jeffreys_high",
                        "uniform_low",
                        "uniform_high",
                    ):
                        result[key] = 0.0
                    result["posterior_model"] = "identical_prediction_events_exact_zero"
                    result["prior_direction_sensitive"] = False
                output.append(result)
        for variant in ("V1", "V2"):
            selections = {
                "all_prediction_changed": {
                    scene for scene, changed, _, _, _, _ in registered if changed[variant]
                },
                "first_leaf_prediction_changed": {
                    scene
                    for scene, changed, risk, _, _, _ in registered
                    if changed[variant] and risk
                },
                "new_out_of_domain_prediction_changed": {
                    scene
                    for scene, changed, _, _, new, _ in registered
                    if changed[variant] and not new["programs"][variant]["in_domain"]
                },
                "DPE_0_to_1": {
                    scene
                    for scene, _, _, _, _, audit in registered
                    if not audit["DPE1_old"] and audit["DPE1_new"]
                },
            }
            for group, scene_ids in selections.items():
                output.append(
                    self.effect(
                        panel,
                        [(checkpoint, "B1", 1), (checkpoint, "B0", -1)],
                        f"b1_{variant}_preference",
                        lambda case, _, scene_ids=scene_ids: case["base_scene_id"] in scene_ids,
                        checkpoint=checkpoint,
                        variant=variant,
                        scope=group,
                        comparison="(B1_new-B1_old)-(B0_new-B0_old)",
                    )
                )

    def workload(self):
        table = []
        grouped = defaultdict(list)
        for job in self.jobs:
            grouped[(job["checkpoint_id"], job["phase"], job["panel"])].append(job)
        for (checkpoint, phase, panel), jobs in sorted(grouped.items()):
            rows = [
                row
                for job in jobs
                for row in self.cells.get((checkpoint, self.owner(job["case_id"])), [])
            ]
            completed = sum(
                len(self.cells.get((checkpoint, self.owner(job["case_id"])), [])) == job["draws"]
                for job in jobs
            )
            elapsed = [
                float(row["elapsed_seconds"])
                for row in rows
                if row.get("elapsed_seconds") is not None
            ]
            lengths = [
                int(row.get("generated_length", len(row.get("token_ids", [])))) for row in rows
            ]
            planned = sum(job["draws"] for job in jobs)
            remaining = planned - len(rows)
            seconds_per_output = sum(elapsed) / len(elapsed) if elapsed else None
            table.append(
                dict(
                    checkpoint=checkpoint,
                    phase=phase,
                    panel=panel,
                    planned_cells=len(jobs),
                    complete_cells=completed,
                    missing_cells=sum(
                        not self.cells.get((checkpoint, self.owner(job["case_id"]))) for job in jobs
                    ),
                    planned_outputs=planned,
                    committed_outputs=len(rows),
                    remaining_outputs=remaining,
                    observed_generation_seconds=sum(elapsed),
                    measured_seconds_per_output=seconds_per_output,
                    estimated_remaining_gpu_hours=remaining * seconds_per_output / 3600
                    if seconds_per_output is not None
                    else None,
                    token_length_p50=float(np.quantile(lengths, 0.5)) if lengths else None,
                    token_length_p90=float(np.quantile(lengths, 0.9)) if lengths else None,
                    peak_memory_allocated_bytes=max(
                        (
                            row["peak_memory_allocated_bytes"]
                            for row in rows
                            if row.get("peak_memory_allocated_bytes") is not None
                        ),
                        default=None,
                    ),
                    peak_memory_reserved_bytes=max(
                        (
                            row["peak_memory_reserved_bytes"]
                            for row in rows
                            if row.get("peak_memory_reserved_bytes") is not None
                        ),
                        default=None,
                    ),
                    memory_peak_scope="allocator high-water mark since checkpoint load",
                    status="COMPLETE" if completed == len(jobs) else "INCOMPLETE",
                )
            )
        return table


def summarize(run_path, *, posterior_draws=20000, bootstrap_draws=5000, seed=20261005):
    """Generate all registered tables/reports without importing a model backend.

    Nondefault draw counts exist for CPU contract tests; reports label them and
    cannot mark their statistical settings as the registered final analysis.
    """
    if posterior_draws <= 0 or bootstrap_draws <= 0:
        raise ValueError("Positive Monte Carlo replication counts required")
    analysis = RunAnalysis(
        run_path, posterior_draws=posterior_draws, bootstrap_draws=bootstrap_draws, seed=seed
    )
    output = analysis.root / "tables"
    output.mkdir(parents=True, exist_ok=True)
    prompt, controls, mass, atoms = analysis.prompt_metrics()
    tables = analysis.effect_tables()
    tables["action_rankings"], ranking_flips = action_rankings(tables["paired_protocol_effects"])
    tables.update(
        prompt_metrics=prompt,
        execution_controls=controls,
        probability_mass_checks=mass,
        workload_actual=analysis.workload(),
    )
    for name, rows in tables.items():
        write_csv(output / (name + ".csv"), rows)
    # Sparse atomic counts preserve dependence; no independent-Beta joint model.
    import pandas as pd

    atom_fields = [
        "checkpoint",
        "panel",
        "case_id",
        "base_scene_id",
        "family",
        "protocol",
        "kind",
        "atom_json",
        "count",
        "denominator",
        "probability",
    ]
    pd.DataFrame(atoms, columns=atom_fields).to_parquet(
        output / "joint_behavior_atoms.parquet", index=False
    )
    planned_outputs = sum(job["draws"] for job in analysis.jobs)
    completed_cells = sum(
        len(analysis.cells.get((job["checkpoint_id"], analysis.owner(job["case_id"])), []))
        == job["draws"]
        for job in analysis.jobs
    )
    availability_path = analysis.root / "CHECKPOINT_AVAILABILITY.json"
    technical_failures = sum(
        len(_jsonl(path)) for path in analysis.root.glob("**/technical_failures.jsonl")
    )
    summary = {
        "protocol_id": analysis.protocol["protocol_id"],
        "status": "COMPLETE" if completed_cells == len(analysis.jobs) else "INCOMPLETE",
        "planned_cells": len(analysis.jobs),
        "completed_cells": completed_cells,
        "planned_outputs": planned_outputs,
        "committed_outputs": len(analysis.rows),
        "missing_outputs": planned_outputs - len(analysis.rows),
        "receipts": analysis.receipts,
        "U22_exposure_audit": analysis.exposure_audit,
        "U22_effective_split_role": analysis.panel_role("U22"),
        "technical_failure_records": technical_failures,
        "checkpoint_availability": _json(availability_path) if availability_path.exists() else None,
        "statistics": {
            "posterior_draws": posterior_draws,
            "scene_bootstrap_draws": bootstrap_draws,
            "seed": seed,
            "registered_replication_counts": posterior_draws == 20000 and bootstrap_draws == 5000,
            "CP": "95% marginal exact; not simultaneous",
            "posterior": "conditional independent-cell Beta(0.5,0.5); sensitivity Beta(1,1)",
            "B1_preference": (
                "within-cell mutually exclusive old/new/other "
                "Dirichlet(0.5,0.5,0.5); sensitivity Dirichlet(1,1,1)"
            ),
            "nonbinary_scores": (
                "empirical mean, sample MC SE and descriptive "
                "scene bootstrap; no Beta interval claimed"
            ),
            "bootstrap": (
                "5000 registered; scenes only; same b"
                "ase_scene keeps every checkpoint/pro"
                "tocol; family/structure stratified"
            ),
            "aliases": "one cell, coefficients consolidated, same posterior variable",
            "D_U": "separate fixed panels; equal family weights",
        },
        "mass_check_failures": sum(row["status"] == "FAIL" for row in mass),
        "table_rows": {name: len(rows) for name, rows in tables.items()},
        "joint_behavior_atoms": len(atoms),
        "workload": tables["workload_actual"],
        "action_rankings": tables["action_rankings"],
        "observed_ranking_flips": ranking_flips,
    }
    if summary["mass_check_failures"]:
        summary["status"] = "INVALID_PROBABILITY_MASS"
    from .reports import write_reports

    write_reports(analysis.root, summary, tables)
    _write_json(analysis.root / "SUMMARY.json", summary)
    return summary


def action_rankings(paired_rows):
    """Post-hoc point rankings at identical family panels; no learned routing claim."""
    groups = {}
    for row in paired_rows:
        if row["metric"] != "pX" or row["scope"] != "family":
            continue
        key = (row["panel"], row["family"], row["checkpoint"])
        group = groups.setdefault(key, {"values": {}, "complete": True})
        if row["status"] != "COMPLETE":
            group["complete"] = False
        for endpoint, value in row["endpoints"].items():
            action = endpoint.split(":", 1)[1]
            if action != "B0":
                group["values"][action] = value
    rows, by_key = [], {}
    for (panel, family, checkpoint), group in sorted(groups.items()):
        values = group["values"]
        available = {key: value for key, value in values.items() if value is not None}
        best = (
            sorted(key for key, value in available.items() if value == max(available.values()))
            if len(available) == len(values) and available
            else []
        )
        row = {
            "panel": panel,
            "family": family,
            "checkpoint": checkpoint,
            "action_values": values,
            "point_best_actions": best,
            "status": "COMPLETE" if group["complete"] else "INCOMPLETE",
            "interpretation": "post_hoc_optimistic_point_ranking_not_routing_value",
        }
        rows.append(row)
        by_key[(panel, family, checkpoint)] = row
    flips = []
    for panel, family in sorted({(row["panel"], row["family"]) for row in rows}):
        for label, early, late in (
            ("T1", "S32", "S96"),
            ("T2", "R4_128", "DIRECT_128"),
            ("REP_T1", "REP32", "REP96"),
        ):
            first, second = by_key.get((panel, family, early)), by_key.get((panel, family, late))
            if first is None or second is None:
                continue
            complete = first["status"] == second["status"] == "COMPLETE"
            flips.append(
                {
                    "panel": panel,
                    "family": family,
                    "comparison": label,
                    "first_checkpoint": early,
                    "second_checkpoint": late,
                    "first_point_best": first["point_best_actions"],
                    "second_point_best": second["point_best_actions"],
                    "disjoint_point_best_actions": not bool(
                        set(first["point_best_actions"]) & set(second["point_best_actions"])
                    )
                    if complete
                    else None,
                    "status": "COMPLETE" if complete else "INCOMPLETE",
                    "route_benefit_established": False,
                }
            )
    return rows, flips
