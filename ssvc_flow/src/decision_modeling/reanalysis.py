"""D0: reproduce the compact preview facts without a model or historical writes."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from .semantic_schema import EVENTS, reward_vector, semantic_features
from .state_table import build_state, vector_moments


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _canonical(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


class PreviewSource:
    """Read allowlisted members in place. Never extract an archive."""

    def __init__(self, source):
        self.path = Path(source)
        self.archive = zipfile.ZipFile(self.path) if self.path.is_file() else None
        self.names = (
            self.archive.namelist()
            if self.archive
            else [str(p.relative_to(self.path)) for p in self.path.rglob("*") if p.is_file()]
        )
        if len(set(self.names)) != len(self.names):
            raise ValueError("duplicate archive member names")

    def read(self, suffix):
        matches = [n for n in self.names if n == suffix or n.endswith("/" + suffix)]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one source member: {suffix}")
        name = matches[0]
        return self.archive.read(name) if self.archive else (self.path / name).read_bytes()

    def table(self, name):
        payload = self.read("tables/" + name + ".csv")
        return list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))), _sha(payload)

    def close(self):
        if self.archive:
            self.archive.close()


def _check_action(row):
    tokens = _json(row["token_ids"])
    eos_ids = _json(row["eos_token_ids"])
    n = int(row["max_new_tokens"])
    if n != 64 or not tokens or any(type(t) is not int or t < 0 for t in tokens) or len(tokens) > n:
        raise ValueError("invalid complete token sequence")
    eos = tokens[-1] in eos_ids
    if any(t in eos_ids for t in tokens[:-1]) or (not eos and len(tokens) != n):
        raise ValueError("prefix or tokens beyond allowed EOS are not complete actions")
    expected_stop = "eos" if eos else "length"
    if (
        row["stop_reason"] != expected_stop
        or _json(row["eos"]) != eos
        or _json(row["truncated"]) != (not eos)
    ):
        raise ValueError("EOS/length flags disagree with complete tokens")
    if (
        int(row["completion_length"]) != len(tokens)
        or (eos and int(row["terminal_token_id"]) != tokens[-1])
        or (not eos and row["terminal_token_id"] not in ("", "null"))
    ):
        raise ValueError("completion length/terminal token mismatch")
    for key in ("action_mask", "token_mask"):
        if _json(row[key]) != [True] * len(tokens):
            raise ValueError("scored action mask omits sampled tokens")
    logs = _json(row["behavior_token_logprobs"])
    if len(logs) != len(tokens) or not all(math.isfinite(v) and v <= 0 for v in logs):
        raise ValueError("invalid generation token probabilities")
    if not math.isclose(math.fsum(logs), float(row["generation_sequence_logp"]), abs_tol=1e-9):
        raise ValueError("generation sequence log probability mismatch")
    if not _json(row["generation_parity"])["passed"]:
        raise ValueError("historical generation/scoring parity failed")
    return tokens


def score_key(score, generation):
    """Core key includes actual input and complete termination contract."""
    return (
        score["inference_fingerprint"],
        generation["input_hash"],
        tuple(_json(generation["token_ids"])),
        generation["stop_reason"],
        int(generation["max_new_tokens"]),
        tuple(_json(generation["eos_token_ids"])),
        score["probability_execution"],
    )


def _support(rows):
    """Conditional mass bounds on a deduplicated set of complete atoms."""
    mass = math.fsum(math.exp(row["sequence_logp"]) for row in rows)
    if not math.isfinite(mass) or mass > 1 + 1e-10:
        raise ValueError("scored complete support has mass above one")
    tail = max(0.0, 1 - mass)
    events = {
        e: math.fsum(math.exp(row["sequence_logp"]) for row in rows if row["event"] == e)
        for e in EVENTS
    }
    known = {
        **events,
        "A": events["X"] + events["S"],
        "V": mass - events["I"],
        "C": math.fsum(math.exp(row["sequence_logp"]) * row["relation_score"] for row in rows),
        "single_edit": math.fsum(
            math.exp(row["sequence_logp"]) * row["single_edit"] for row in rows
        ),
    }
    identified_qx = events["X"] + events["S"] > 0
    return {
        "method": "conditional_mass_bounds",
        "evidence_kind": "CONDITIONAL_ON_SCORER",
        "known_mass": mass,
        "unknown_tail": tail,
        "unique_complete_actions": len(rows),
        "known_event_mass": events,
        "bounds": {k: [v, min(1.0, v + tail)] for k, v in known.items()},
        "qX": [
            events["X"] / (events["X"] + events["S"] + tail),
            (events["X"] + tail) / (events["X"] + events["S"] + tail),
        ]
        if identified_qx
        else None,
        "qX_status": "CONDITIONAL_ON_SCORER"
        if identified_qx
        else "UNIDENTIFIED_DENOMINATOR_CONTAINS_ZERO",
        "joint_constraints": ["pX+pS+pW+pI=1", "A=pX+pS", "V=1-pI", "pX<=A<=V", "0<=C<=V"],
        "numerical_error_bounded": False,
        "statistical_confidence_interval": False,
        "no_X_support": events["X"] == 0,
        "no_X_means_absent": False,
    }


def _write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )


def reanalyze(source, out):
    """Validate records, preserve draw frequency, write actual Parquet and receipts."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(out)
    src = PreviewSource(source)
    try:
        raw, gh = src.table("PREVIEW_RAW_GENERATIONS")
        scores, sh = src.table("PREVIEW_RAW_SCORES")
        prompt_rows, ph = src.table("PREVIEW_PROMPTS")
        identity = {
            "generations_csv_sha256": gh,
            "scores_csv_sha256": sh,
            "prompts_csv_sha256": ph,
            "implementation_sha256": {
                name: _sha(Path(__file__).with_name(name).read_bytes())
                for name in ("reanalysis.py", "semantic_schema.py", "state_table.py")
            },
        }
        if (out / "COMPLETE.json").exists():
            prior = json.loads((out / "COMPLETE.json").read_text())
            if prior["source_identity"] != identity:
                raise ValueError("refusing to overwrite completed D0 with different source")
            for name, digest in prior["output_sha256"].items():
                if _sha((out / name).read_bytes()) != digest:
                    raise ValueError("completed D0 artifact integrity failure")
            return json.loads((out / "summary.json").read_text())
        if out.exists() and any(out.iterdir()):
            raise ValueError(
                "output is nonempty without completion receipt; "
                "preserve it and choose a fresh directory"
            )
        prompts = {row["prompt_id"]: _json(row["full_prompt_record"]) for row in prompt_rows}
        if len(prompts) != len(prompt_rows):
            raise ValueError("duplicate prompt identity")
        by_sample, facts, groups = {}, [], defaultdict(list)
        aliases = defaultdict(dict)
        for row in raw:
            key = row["sample_key"]
            if key in by_sample or row["sample_id"] != key:
                raise ValueError("duplicate/mismatched sampling identity")
            tokens = _check_action(row)
            if row["probability_execution"] != "uncached_prefix_recompute":
                raise ValueError("D0 source changed its probability execution path")
            expected_key = _canonical(
                [row["rng_namespace"], row["prompt_id"], int(row["draw_index"])]
            )
            if key != expected_key or int(row["sample_seed"]) != int(expected_key[:16], 16) % (
                2**63
            ):
                raise ValueError("sample key/seed does not match frozen RNG namespace")
            if row["shared_token_identity"] != _canonical(tokens):
                raise ValueError("token identity hash mismatch")
            prompt = prompts[row["prompt_id"]]
            features = semantic_features(row["raw_completion"], prompt)
            annotation = _json(row["annotation"])
            if (
                features["event"] != row["category"]
                or annotation["parsed_world"] != features["parsed_world"]
                or _json(row["event_onehot"]) != [int(features["event"] == e) for e in EVENTS]
            ):
                raise ValueError("strict parser/event recomputation mismatch")
            if row["proposal"] not in {"ORIGIN", "DIRECT", "MIX"} or row["role"] not in {
                "work",
                "reference",
                "direct_work",
                "direct",
            }:
                raise ValueError("unknown source role/proposal")
            audit = _json(row["input_audit"])
            if (
                row["input_hash"] != audit["input_tensor_hash"]
                or row["input_tensor_hash"] != row["input_hash"]
            ):
                raise ValueError("input identity mismatch")
            fact = {
                **features,
                "sample_key": key,
                "prompt_id": row["prompt_id"],
                "base_scene_id": row["base_scene_id"],
                "family": prompt["family"],
                "interface": prompt["interface"],
                "proposal": row["proposal"],
                "source_role": row["role"],
                "inference_fingerprint": row["inference_fingerprint"],
                "proposal_policy_id": row["proposal_policy_id"],
                "rng_namespace": row["rng_namespace"],
                "draw_index": int(row["draw_index"]),
                "full_tokens": tokens,
                "stop_condition": row["stop_reason"],
                "completion_length": len(tokens),
                "generation_sequence_logp": float(row["generation_sequence_logp"]),
                "elapsed_generation_seconds": float(row["elapsed_seconds"]),
                "vision_forward_calls": int(row["vision_forward_calls"]),
                "input_identity": row["input_hash"],
                "frequency": 1,
            }
            facts.append(fact)
            by_sample[key] = (row, fact)
            # A MIX stream is a single proposal law, not two endpoint samples.
            stream = (row["prompt_id"], row["rng_namespace"])
            groups[stream].append(fact)
        keyed, scores_by_sample, scored_facts = {}, defaultdict(dict), []
        max_duplicate_span = 0.0
        fp_candidates = defaultdict(set)
        for score in scores:
            sid = score["proposal_sample_key"]
            if sid not in by_sample:
                raise ValueError("score has no linked generation")
            gen, fact = by_sample[sid]
            for field in (
                "prompt_id",
                "prompt_record_hash",
                "input_hash",
                "shared_token_identity",
                "probability_execution",
                "max_new_tokens",
                "eos_token_ids",
                "runtime_identity",
            ):
                if score[field] != gen[field]:
                    raise ValueError(f"score/generation {field} differs")
            if score["category"] != fact["event"] or _json(score["score_is_independent_draw"]):
                raise ValueError("score labels/independence invalid")
            logs = _json(score["token_logprobs"])
            logp = float(score["sequence_logp"])
            if (
                len(logs) != len(fact["full_tokens"])
                or not all(math.isfinite(v) and v <= 0 for v in logs)
                or not math.isclose(math.fsum(logs), logp, abs_tol=1e-9)
            ):
                raise ValueError("score full sequence log probability mismatch")
            fp = score["inference_fingerprint"]
            if fp in scores_by_sample[sid]:
                raise ValueError("duplicate logical score association")
            scores_by_sample[sid][fp] = logp
            fp_candidates[fp].add(score["candidate_id"])
            core = score_key(score, gen)
            value = {
                **fact,
                "sequence_logp": logp,
                "scoring_fingerprint": fp,
                "score_source_role": score["role"],
                "scoring_path": score["probability_execution"],
                "score_key": _canonical(core),
            }
            if core in keyed:
                prior = keyed[core]
                span = abs(prior["sequence_logp"] - logp)
                max_duplicate_span = max(max_duplicate_span, span)
                if span > 1e-10 or prior["event"] != fact["event"]:
                    raise ValueError("same complete scoring key has inconsistent scores/labels")
            else:
                keyed[core] = value
            scored_facts.append(value)
        support_groups = defaultdict(list)
        for row in keyed.values():
            support_groups[(row["prompt_id"], row["scoring_fingerprint"])].append(row)
        support = [
            {"prompt_id": pid, "inference_fingerprint": fp, **_support(rows)}
            for (pid, fp), rows in sorted(support_groups.items())
        ]
        states = []
        endpoints = []
        contrasts = []
        comparison_max_error = 0.0
        for (_pid, stream), rows in sorted(groups.items()):
            # Z4 uses only outputs in this same stream, preserving the Z0-Z4
            # information budget. A data-dependent discovered set is legal for
            # mass bounds; it is not reused as an unbiased tail MC sample.
            stream_atoms = {}
            for fact in rows:
                logs = scores_by_sample.get(fact["sample_key"], {})
                if fact["proposal"] == "MIX":
                    if len(logs) != 2:
                        raise ValueError("MIX law needs both endpoint probabilities")
                    left_logp, right_logp = logs.values()
                    peak = max(left_logp, right_logp)
                    sequence_logp = peak + math.log(
                        (math.exp(left_logp - peak) + math.exp(right_logp - peak)) / 2
                    )
                else:
                    sequence_logp = fact["generation_sequence_logp"]
                atom = (fact["input_identity"], tuple(fact["full_tokens"]), fact["stop_condition"])
                stream_atoms.setdefault(atom, {**fact, "sequence_logp": sequence_logp})
            measurement = _support(list(stream_atoms.values()))
            measurement.update(
                {
                    "support_source": "same_observed_stream",
                    "extra_output_observations": 0,
                    "tail_MC_unbiasedness_claimed": False,
                    "fixed_panel_MC": {
                        "method": "HOEFFDING_POINTWISE_DESCRIPTIVE",
                        "alpha": 0.05,
                        "half_width": math.sqrt(math.log(40) / (2 * len(rows))),
                        "simultaneous_decision_interval": False,
                    },
                    "behavior_variance": "Z2.reward_moments.behavior_covariance",
                    "scene_heterogeneity": (
                        "stratify_by_base_scene_id_and_family; three_debug_scenes_only"
                    ),
                }
            )
            state = build_state(rows, measurement=measurement)
            meta = {
                k: rows[0][k]
                for k in (
                    "prompt_id",
                    "base_scene_id",
                    "family",
                    "interface",
                    "proposal",
                    "source_role",
                )
            }
            states.append({**meta, "rng_namespace": stream, **state})
            if rows[0]["proposal"] == "DIRECT":
                endpoints.append(
                    {
                        **meta,
                        "policy": rows[0]["proposal_policy_id"],
                        "n": len(rows),
                        "counts": state["Z1"]["event_counts"],
                    }
                )
        for index, prompt_row in enumerate(prompt_rows):
            pid = prompt_row["prompt_id"]
            response = json.loads(
                src.read(f"preview_20260918/prompts/{index:02d}/response/COMPLETE.json")
            )
            diag = json.loads(src.read(f"preview_20260918/prompts/{index:02d}/DIAGNOSTICS.json"))[
                "units"
            ][0]
            left = response["policies"]["calibration_001_joint_1"]["inference_fingerprint"]
            right = response["policies"]["calibration_001_joint_0"]["inference_fingerprint"]
            source_bound = diag["conditional_support_bounds"]
            for fp, endpoint in ((left, "left"), (right, "right")):
                computed = next(
                    r for r in support if r["prompt_id"] == pid and r["inference_fingerprint"] == fp
                )
                if abs(computed["known_mass"] - source_bound[f"seen_mass_{endpoint}"]) > 1e-10:
                    raise ValueError(
                        "recomputed support mass disagrees with historical diagnostics"
                    )
            direct = {r["policy"]: r for r in endpoints if r["prompt_id"] == pid}
            direct_difference = [
                direct["calibration_001_joint_1"]["counts"][e]
                / direct["calibration_001_joint_1"]["n"]
                - direct["calibration_001_joint_0"]["counts"][e]
                / direct["calibration_001_joint_0"]["n"]
                for e in EVENTS
            ]
            if (
                max(
                    abs(x - y)
                    for x, y in zip(direct_difference, diag["means"]["DIRECT"], strict=True)
                )
                > 1e-10
            ):
                raise ValueError("endpoint counts disagree with historical diagnostics")
            for candidate, policy in response["policies"].items():
                aliases[policy["inference_fingerprint"]][candidate] = {
                    k: policy.get(k)
                    for k in (
                        "training_state_hash",
                        "optimizer_state_hash",
                        "complete_training_state_retained",
                    )
                }
            for proposal in ("ORIGIN", "MIX"):
                selected = [
                    r
                    for r in facts
                    if r["prompt_id"] == pid
                    and r["source_role"] == "reference"
                    and r["proposal"] == proposal
                ]
                contributions = []
                for fact in selected:
                    logs = scores_by_sample[fact["sample_key"]]
                    left_logp, right_logp = logs[left], logs[right]
                    if proposal == "ORIGIN":
                        anchor = fact["generation_sequence_logp"]
                        # exp(r-anchor)*expm1(l-r) avoids subtraction near null.
                        weight = math.exp(right_logp - anchor) * math.expm1(left_logp - right_logp)
                    else:
                        weight = 2 * math.tanh((left_logp - right_logp) / 2)
                    contributions.append(
                        [weight * int(fact["event"] == e) for e in EVENTS]
                        + [weight * v for v in reward_vector(fact)]
                    )
                moments = vector_moments(contributions)
                error = max(abs(moments["mean"][j] - diag["means"][proposal][j]) for j in range(4))
                covariance_error = max(
                    abs(
                        moments["covariance_of_mean"][j][k]
                        - diag["covariance_of_mean"][proposal][j][k]
                    )
                    for j in range(4)
                    for k in range(4)
                )
                comparison_max_error = max(comparison_max_error, error, covariance_error)
                if max(error, covariance_error) > 1e-10:
                    raise ValueError(
                        "recomputed LR moments disagree with independent historical diagnostics"
                    )
                contrasts.append(
                    {
                        "prompt_id": pid,
                        "proposal": proposal,
                        "contrast": "joint_1 - joint_0",
                        "columns": [
                            "X",
                            "S",
                            "W",
                            "I",
                            "reward_X",
                            "reward_A",
                            "reward_V",
                            "reward_C",
                        ],
                        "evidence_kind": "EMPIRICAL_APPROXIMATION"
                        if proposal == "ORIGIN"
                        else "BOUNDED_MIX_DRAWS",
                        "contribution_range": None if proposal == "ORIGIN" else [-2, 2],
                        **moments,
                    }
                )
        status, _ = src.table("PREVIEW_STATUS")
        counts = Counter(row["event"] for row in facts)
        reuse = {
            "logical_scores": len(scores),
            "unique_core_keys": len(keyed),
            "repeated_score_rows": len(scores) - len(keyed),
            "potential_reuse_fraction": 1 - len(keyed) / len(scores),
            "historical_cache_enabled": False,
            "observed_wall_clock_speedup": None,
            "maximum_duplicate_sequence_logp_span": max_duplicate_span,
            "key_fields": [
                "inference_fingerprint",
                "input_identity",
                "full_tokens",
                "stop_condition",
                "max_new_tokens",
                "eos_token_ids",
                "scoring_path",
            ],
            "sampling_observations_preserved": len(facts),
            "scores_are_new_observations": False,
        }
        summary = {
            "status": "D0_REANALYZED",
            "scientific_status": "NOT_CERTIFIED",
            "model_calls": 0,
            "source_identity": identity,
            "generation_count": len(facts),
            "score_count": len(scores),
            "unique_core_scoring_keys": len(keyed),
            "prompt_count": len(prompts),
            "scene_count": len({r["base_scene_id"] for r in facts}),
            "origin_count": len({r["origin_id"] for r in raw}),
            "candidate_inference_policies": len(fp_candidates),
            "all_generation_event_counts": dict(counts),
            "endpoints": endpoints,
            "relation_plus_single_edit_equivalence": {
                "scope": "observed_outputs_only_not_full_domain_proof",
                "disagreements_with_X": sum(
                    (
                        r["relation_numerator"] == r["relation_denominator"]
                        and bool(r["single_edit"])
                    )
                    != (r["event"] == "X")
                    for r in facts
                ),
            },
            "strict_parser_mismatches": 0,
            "score_association_mismatches": 0,
            "complete_action_validation_failures": 0,
            "independent_diagnostics_max_error": comparison_max_error,
            "no_X_support_prompt_policy_count": sum(r["no_X_support"] for r in support),
            "generation_seconds_sum": math.fsum(r["elapsed_generation_seconds"] for r in facts),
            "generation_tokens": sum(r["completion_length"] for r in facts),
            "generation_vision_forward_calls": sum(r["vision_forward_calls"] for r in facts),
            "prompt_elapsed_seconds_this_attempt": [
                {"prompt_id": r["prompt_id"], "seconds": float(r["elapsed_seconds_this_attempt"])}
                for r in status
            ],
            "scoring_seconds": None,
            "scoring_seconds_status": "NOT_SEPARATELY_RECORDED_IN_RAW_SCORE_TABLE",
            "predictor_test": "NOT_RUN",
            "new_training_updates": 0,
            "scope": (
                "one historical origin; two scored inference policies; six debugging prompts; "
                "no independent training-seed inference"
            ),
        }
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(facts), out / "state_table.parquet")
        pq.write_table(pa.Table.from_pylist(scored_facts), out / "scored_observations.parquet")
        _write_json(out / "states_Z0_Z4.json", states)
        _write_json(out / "conditional_mass_bounds.json", support)
        _write_json(out / "lr_moments.json", contrasts)
        _write_json(
            out / "inference_aliases.json",
            {
                "aliases": dict(aliases),
                "training_states_are_not_interchangeable": True,
                "origin_generation_fingerprints": sorted(
                    {r["inference_fingerprint"] for r in raw if r["proposal"] == "ORIGIN"}
                ),
            },
        )
        _write_json(out / "score_reuse_summary.json", reuse)
        _write_json(out / "summary.json", summary)
        report = _report(summary, support, reuse)
        (out / "D0_REANALYSIS_zh.md").write_text(report)
        if pq.read_table(out / "state_table.parquet").num_rows != len(raw):
            raise RuntimeError("Parquet draw count failed readback")
        hashes = {p.name: _sha(p.read_bytes()) for p in out.iterdir() if p.is_file()}
        _write_json(out / "COMPLETE.json", {"source_identity": identity, "output_sha256": hashes})
        return summary
    finally:
        src.close()


def _report(summary, support, reuse):
    lines = [
        "# D0 历史预览重分析",
        "",
        "状态：D0_REANALYZED；科学结论仍为 NOT_CERTIFIED。无模型调用、无新训练、无预测器测试。",
        "",
        (
            f"读取 {summary['generation_count']:,} 条生成、{summary['score_count']:,} 条评分；"
            f"{summary['prompt_count']} 题/{summary['scene_count']} 场景，"
            "一个历史原点、两个被评分推理策略。"
            "严格解析、完整 token/EOS/length、评分关联复核未发现不一致。"
        ),
        (
            f"完整评分键 {reuse['unique_core_keys']} 个；"
            f"重复评分行 {reuse['repeated_score_rows']} 条，"
            f"潜在复用比例 {reuse['potential_reuse_fraction']:.6%}。"
            "这是计算复用潜力，未实测新缓存墙钟加速。"
            f"{summary['generation_count']:,} 个抽样观察均保留，重复 token 不从统计样本去重。"
        ),
        "",
        "| 接口 | 家族 | 端点 | n | X | S | W | I |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["endpoints"]:
        lines.append(
            f"| {row['interface']} | {row['family']} | {row['policy']} | {row['n']} | "
            + " | ".join(str(row["counts"][e]) for e in EVENTS)
            + " |"
        )
    lines += [
        "",
        (
            "ORIGIN/MIX 的联合均值及协方差独立复算与历史 DIAGNOSTICS 最大绝对差 "
            f"{summary['independent_diagnostics_max_error']:.3g}。"
            "完整非对角协方差保存在 lr_moments.json；"
            "ORIGIN 没有由样本最大值推断总体权重上界，MIX 贡献范围为 [-2,2]。"
        ),
        "",
        (
            "Z0–Z4 由同一组样本生成，逐题、流、角色分开保存。"  # noqa: RUF001
            "Z3 保留 event×关系分子/分母及 event×single_edit；"  # noqa: RUF001
            "不同关系家族分母没有阈值化。"
            "state_table.parquet 为逐观察事实，不是只含均值的汇总。"
        ),
        "",
        "| 题目序号 | 策略指纹前缀 | 已知质量 | 未知尾部 | 无 X 支持 |",
        "|---|---|---:|---:|---|",
    ]
    pids = [r["prompt_id"] for r in summary["prompt_elapsed_seconds_this_attempt"]]
    for row in support:
        lines.append(
            f"| {pids.index(row['prompt_id']) + 1} | {row['inference_fingerprint'][:12]} "
            f"| {row['known_mass']:.10f} | {row['unknown_tail']:.10f} | {row['no_X_support']} |"
        )
    lines += [
        "",
        (
            "质量界名称为 conditional_mass_bounds，全部标记 CONDITIONAL_ON_SCORER。"
            "未界定评分数值误差，不属于统计置信区间，不产生 CERTIFIED_SAFE。"
            "没有 X 支持不等于 X 不存在；qX 分母区间含 0 时返回未识别。"
            "未知尾部没有分配为确定事件。"
        ),
        "",
        (
            "历史生成记录 elapsed_seconds 合计 "
            f"{summary['generation_seconds_sum']:.6f} 秒，"
            f"生成 token {summary['generation_tokens']:,} 个，记录的生成 vision_forward_calls "
            f"{summary['generation_vision_forward_calls']:,}。"
            "此处是行级生成耗时求和，不是任务墙钟。原始评分表未单列评分秒数，保留未记录；"
            "逐题当次尝试耗时见 summary.json，不能据此分解评分/加载成本。"
        ),
        "",
        (
            "inference_aliases.json 保留推理别名与训练/Adam 身份；"
            "共享推理指纹不表示可交换未来训练状态。"
            "六题只作历史调试，不进入新 P/E 面板。未执行 D1/D2/D3/D4。"
        ),
        "",
        (
            "产物：state_table.parquet、scored_observations.parquet、states_Z0_Z4.json、"
            "conditional_mass_bounds.json、lr_moments.json、inference_aliases.json、"
            "score_reuse_summary.json、summary.json；"
            "COMPLETE.json 记录 CSV 输入哈希与所有输出 SHA-256。"
        ),
        "",
    ]
    return "\n".join(lines)
