"""Fixed, prompt-wise exploratory measurement of existing V4 candidates.

No training, held-out evaluation, campaign mutation, or automatic expansion.
Candidate provenance remains old; measurement provenance is the actual new code.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from ..core import frozen_writer
from ..modeling_v3.io import canonical_hash, source_identity
from . import gpu_collect as gpu
from .analysis_rules import load_analysis_rules
from .config import atomic_json
from .data_adapter import CONTRASTS, bound_json

COMPATIBILITY_KEYS = (
    "protocol_version",
    "execution_kind",
    "model_hash",
    "data_hash",
    "config_hash",
    "bindings_hash",
    "environment",
)
PRIMARY = CONTRASTS[0][0]


def _outside_campaign(out, campaign):
    out, campaign = Path(out).resolve(), Path(campaign).resolve()
    if out == campaign or out.is_relative_to(campaign) or campaign.is_relative_to(out):
        raise ValueError("Preview output must be separate from the original campaign")
    return out


def _tasks(path):
    tasks = gpu._read(path)
    if (
        canonical_hash({k: v for k, v in tasks.items() if k != "task_list_hash"})
        != tasks["task_list_hash"]
    ):
        raise ValueError("Frozen task list changed")
    if (
        tasks["analysis_rules"] != load_analysis_rules()
        or canonical_hash(tasks["analysis_rules"]) != tasks["analysis_rules_hash"]
    ):
        raise ValueError("Frozen analysis rules changed")
    return tasks


def select_banks(records):
    """Only training metadata; no response labels or claimed held-out status."""
    if len(records) != 8 or [r["bank_id"] for r in records] != [
        f"calibration_{i:03d}" for i in range(8)
    ]:
        raise ValueError("All first eight registered calibration banks must be complete")
    alias, active = [], []
    for record in records:
        if record["role"] != "calibration" or not record["origin_state_restored"]:
            raise ValueError("Expected genuine same-origin calibration forks")
        if set(record["policies"]) != {"joint_0", "joint_1", "no_x_off_1"}:
            raise ValueError("All three candidate policies are required")
        contrast = record["contrasts"][PRIMARY]
        same = (
            record["policies"]["joint_1"]["inference_fingerprint"]
            == record["policies"]["joint_0"]["inference_fingerprint"]
        )
        if same != contrast["exact_inference_alias"] or not math.isfinite(contrast["norm"]):
            raise ValueError("Primary alias or norm metadata differs")
        if same:
            alias.append(record)
        elif contrast["norm"] > 0:
            active.append(record)
    if not alias or not active:
        raise ValueError("Fixed first-eight pool lacks an alias control or active primary pair")
    return [alias[0], active[0]]


def prepare(tasks_path, collection_code, out):
    """Metadata only: freeze scope before any preview response is observed."""
    tasks = _tasks(tasks_path)
    out = _outside_campaign(out, tasks["root"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new empty preview directory")
    if source_identity(collection_code) != tasks["source"]:
        raise ValueError("Original collection source differs")
    task_id = "map_41001_X_BASE_32"
    task = next(t for t in tasks["tasks"] if t["id"] == task_id)
    if (task["stage"], task["kind"], task["seed"], task["step"]) != ("C", "map", 41001, 32):
        raise ValueError("Preview requires the registered first development origin")
    folder = Path(tasks["root"]) / "tasks" / task_id / "forks"
    pool = [gpu._binding(folder / f"calibration_{i:03d}" / "COMPLETE.json") for i in range(8)]
    selected = select_banks([bound_json(b) for b in pool])
    prompts = tasks["inputs"]["panels"]["observation"][: task["prompts"]]
    ids = gpu._diagnostic_prompt_ids(prompts)
    if len(ids) != 6:
        raise ValueError("Expected six family/interface strata")
    probes = [next(p for p in prompts if p["prompt_id"] == pid) for pid in ids]
    plan = {
        "schema": "ssvc-v4-measurement-preview-1",
        "scope": "EXPLORATORY_CALIBRATION_MEASUREMENT_ONLY",
        "tasks": gpu._binding(tasks_path),
        "collection_code": str(Path(collection_code).resolve()),
        "collection_source": tasks["source"],
        "measurement_source": source_identity(),
        "source_task_id": task_id,
        "root": str(out),
        "selection_pool": pool,
        "banks": [pool[int(b["bank_id"].rsplit("_", 1)[1])] for b in selected],
        "origin_policy": gpu._binding(folder / "origin_policy.json"),
        "probes": probes,
        "draws": 64,
        "reference_draws": 64,
        "selection_rule": (
            "First exact primary alias and first nonalias positive-norm primary "
            "in calibration_000..007; first prompt per family/interface "
            "in the frozen 24-prompt panel"
        ),
        "progress_unit": (
            "One complete prompt, including independent ORIGIN, MIX and endpoint COUNT"
        ),
        "stop_after": "SIX_FIXED_PROMPTS_NO_AUTOMATIC_EXPANSION",
        "reference_is_high_precision_truth": False,
        "observation_score_mode": "uncached_prefix_recompute",
        "predictive_evaluation": "NOT_RUN",
        "heldout_query_labels_used": False,
        "new_adam_updates": 0,
        "online_ssvc": False,
    }
    plan["plan_hash"] = canonical_hash(plan)
    gpu._publish(out / "PLAN.json", plan)
    return plan


def _load_plan(path):
    plan = gpu._read(path)
    if canonical_hash({k: v for k, v in plan.items() if k != "plan_hash"}) != plan["plan_hash"]:
        raise ValueError("Preview plan changed")
    if plan["measurement_source"] != source_identity():
        raise ValueError("Preview measurement source changed")
    tasks = bound_json(plan["tasks"])
    _tasks(plan["tasks"]["path"])
    root = _outside_campaign(plan["root"], tasks["root"])
    task = next(t for t in tasks["tasks"] if t["id"] == "map_41001_X_BASE_32")
    panel = tasks["inputs"]["panels"]["observation"][: task["prompts"]]
    expected_ids = gpu._diagnostic_prompt_ids(panel)
    expected_probes = [next(p for p in panel if p["prompt_id"] == pid) for pid in expected_ids]
    if (
        plan["source_task_id"] != task["id"]
        or plan["probes"] != expected_probes
        or len(expected_probes) != 6
        or plan["draws"] != 64
        or plan["reference_draws"] != 64
        or plan["observation_score_mode"] != "uncached_prefix_recompute"
        or plan["heldout_query_labels_used"] is not False
        or plan["predictive_evaluation"] != "NOT_RUN"
    ):
        raise ValueError("Fixed exploratory scope differs")
    if Path(path).resolve() != root / "PLAN.json":
        raise ValueError("Preview plan location changed")
    if tasks["source"] != plan["collection_source"]:
        raise ValueError("Original source identity changed")
    folder = (Path(tasks["root"]) / "tasks" / plan["source_task_id"] / "forks").resolve()
    for binding in [plan["origin_policy"], *plan["selection_pool"], *plan["banks"]]:
        if not Path(binding["path"]).resolve().is_relative_to(folder):
            raise ValueError("Imported metadata outside the source origin")
    selected = select_banks([bound_json(b) for b in plan["selection_pool"]])
    banks = [bound_json(b) for b in plan["banks"]]
    if selected != banks:
        raise ValueError("Frozen bank selection differs")
    origin = bound_json(plan["origin_policy"])
    if (
        origin["candidate_id"] != "origin"
        or origin["checkpoint"]["identity"].get("candidate_id") != "origin"
        or origin["checkpoint"]["identity"].get("origin_id") != plan["source_task_id"]
    ):
        raise ValueError("Original origin policy label differs")
    for bank in banks:
        for name, policy in bank["policies"].items():
            identity = policy["checkpoint"]["identity"]
            if (
                identity.get("origin_id") != plan["source_task_id"]
                or identity.get("bank_id") != bank["bank_id"]
                or identity.get("candidate_id") != bank["bank_id"] + "_" + name
                or policy["candidate_id"] != identity["candidate_id"]
            ):
                raise ValueError("Imported candidate origin/bank/label differs")
    policies = [origin, *(p for b in banks for p in b["policies"].values())]
    expected = origin["checkpoint"]["identity"]
    if expected["source_hash"] != tasks["source"]["sha256"] or expected[
        "config_hash"
    ] != canonical_hash(tasks["config"]):
        raise ValueError("Imported policy collection source/config differs")
    for policy in policies:
        checkpoint = policy["checkpoint"]
        if not Path(checkpoint["path"]).resolve().is_relative_to(folder):
            raise ValueError("Imported checkpoint outside the source origin")
        for key in (*COMPATIBILITY_KEYS, "source_hash"):
            if key not in expected or checkpoint["identity"].get(key) != expected[key]:
                raise ValueError("Imported policy identity differs: " + key)
    return plan, tasks, banks, origin


def _check_runtime(runtime, origin, banks):
    expected = origin["checkpoint"]["identity"]
    for key in COMPATIBILITY_KEYS:
        if key not in expected or runtime["identity"].get(key) != expected[key]:
            raise ValueError("New measurement runtime differs: " + key)
    layout = gpu._policy_layout(runtime)
    for policy in [origin, *(p for b in banks for p in b["policies"].values())]:
        if any(policy.get(key) != value for key, value in layout.items()):
            raise ValueError("Imported policy parameter layout/scaling differs")


def _write_summary(root, plan, receipts):
    lines = [
        "# V4 首批概率测量",
        "",
        "状态：探索性测量；未检验预测模型；不代表完整 C 阶段完成。",
        "",
        f"已完成问题：{len(receipts)}/{len(plan['probes'])}；每题各独立抽样流 64 次。",
        "",
        (
            "全部差值单位为百分点。COUNT 区间为单事件保守 95% 区间；"
            "ORIGIN/MIX 标准误差是经验抽样误差，不含数值系统误差。"
        ),
        "零经验标准误差不证明真实误差为零；各题区间不能当作整个面板同时覆盖保证。",
        "",
        "|问题|候选组|对比|事件|ORIGIN ± SE|MIX ± SE|COUNT [区间]|",
        "|---|---|---|---|---|---|---|",
    ]
    for receipt in receipts:
        diagnostic = bound_json(receipt["diagnostics"])
        for unit in diagnostic["units"]:
            if unit["status"] != "MEASURED":
                continue
            for i, event in enumerate(("X", "S", "W", "I")):
                values = []
                for method in ("ORIGIN", "MIX"):
                    mean = 100 * unit["means"][method][i]
                    se = 100 * math.sqrt(max(0, unit["variance_of_mean"][method][i]))
                    values.append(f"{mean:.6g} ± {se:.6g}")
                count = "未测"
                if unit["endpoint_count_intervals"]:
                    lo, hi = unit["endpoint_count_intervals"]["single_event"][i]
                    count = (
                        f"{100 * unit['means']['DIRECT'][i]:.6g} [{100 * lo:.6g}, {100 * hi:.6g}]"
                    )
                fields = [
                    unit["prompt_id"],
                    unit["bank_id"],
                    unit["contrast_id"],
                    event,
                    *values,
                    count,
                ]
                lines.append("|" + "|".join(str(v).replace("|", "\\|") for v in fields) + "|")
    lines += [
        "",
        "完整协方差、重叠诊断、条件概率质量界与所有原始测量的路径在逐题 JSON 中。",
        "已选同策略对照保留在候选元数据及共享评分中；精确别名复用评分不算独立测量。",
    ]
    # This is a replaceable view; all per-prompt evidence is immutable.
    temporary = root / "FIRST_RESULTS_zh.md.pending"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(root / "FIRST_RESULTS_zh.md")


def run(plan_path, *, execute_gpu=False, resume=False, runtime_factory=None):
    if execute_gpu is not True:
        raise PermissionError("New preview measurement requires --execute-gpu")
    plan, tasks, banks, origin = _load_plan(plan_path)
    root = Path(plan["root"])
    if (root / "RUN_IDENTITY.json").exists() and not resume:
        raise FileExistsError("Use --resume for an existing preview run")
    with frozen_writer(root):
        gpu._publish(
            root / "RUN_IDENTITY.json",
            {"plan_hash": plan["plan_hash"], "source": plan["measurement_source"]},
        )
        runtime, receipts = None, []
        observation_id = f"preview:{plan['plan_hash']}:{plan['source_task_id']}"
        forks = {"origin_id": observation_id, "origin_policy": origin, "banks": banks}
        for index, prompt in enumerate(plan["probes"]):
            folder = root / "prompts" / f"{index:02d}"
            done = folder / "COMPLETE.json"
            if done.exists():
                receipt = gpu._read(done)
                if (
                    receipt["plan_hash"] != plan["plan_hash"]
                    or receipt["prompt_id"] != prompt["prompt_id"]
                ):
                    raise ValueError("Completed preview prompt identity differs")
                gpu.verify_artifact_bindings(receipt)
            else:
                if runtime is None:
                    runtime = (runtime_factory or gpu.load_runtime)(
                        tasks["config"], tasks["bindings"], device="cuda:0", execute_gpu=True
                    )
                    _check_runtime(runtime, origin, banks)
                    runtime["analysis_rules"] = tasks["analysis_rules"]
                    gpu._publish(root / "RUNTIME.json", runtime["identity"])
                    # Existing fixed actions, no new generation/performance probe.
                    gpu._null_receipt(runtime, tasks["inputs"], root)
                    # The diagnostic packet validator certifies this path only.
                    runtime["observation_score_mode"] = plan["observation_score_mode"]
                    state = runtime["checkpoint_cache"].load(origin)
                    runtime["state"].restore(state)
                started = time.perf_counter()
                atomic_json(
                    root / "LATEST.json",
                    {
                        "status": "MEASURING_PROMPT",
                        "completed_prompts": len(receipts),
                        "total_prompts": len(plan["probes"]),
                        "prompt_id": prompt["prompt_id"],
                    },
                )
                response = gpu.collect_response_map(
                    tasks["config"],
                    runtime,
                    forks,
                    [prompt],
                    out=folder / "response",
                    draws=plan["draws"],
                    reference_draws=plan["reference_draws"],
                    bridge=True,
                    resume=resume,
                )
                diagnostics = gpu.pair_observation_diagnostics(response)
                gpu._publish(folder / "DIAGNOSTICS.json", diagnostics)
                receipt = {
                    "plan_hash": plan["plan_hash"],
                    "prompt_id": prompt["prompt_id"],
                    "source_origin_id": plan["source_task_id"],
                    "observation_origin_id": observation_id,
                    "response": gpu._binding(folder / "response" / "COMPLETE.json"),
                    "diagnostics": gpu._binding(folder / "DIAGNOSTICS.json"),
                    "scope": plan["scope"],
                    "scientific_status": "NOT_CERTIFIED",
                    "elapsed_seconds_this_attempt": time.perf_counter() - started,
                    "new_adam_updates": 0,
                }
                gpu._publish(done, receipt)
            receipts.append(receipt)
            _write_summary(root, plan, receipts)
            atomic_json(
                root / "LATEST.json",
                {
                    "status": "PROMPT_COMPLETE",
                    "completed_prompts": len(receipts),
                    "total_prompts": len(plan["probes"]),
                    "prompt_id": prompt["prompt_id"],
                },
            )
        result = {
            "plan_hash": plan["plan_hash"],
            "status": "PREVIEW_MEASUREMENTS_COMPLETE",
            "prompts": receipts,
            "predictive_evaluation": "NOT_RUN",
            "scientific_status": "NOT_CERTIFIED",
            "automatic_successor": False,
        }
        gpu._publish(root / "COMPLETE.json", result)
        atomic_json(
            root / "LATEST.json",
            {
                "status": result["status"],
                "completed_prompts": len(receipts),
                "total_prompts": len(plan["probes"]),
            },
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--tasks", required=True)
    p.add_argument("--collection-code", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("--plan", required=True)
    p.add_argument("--execute-gpu", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        plan = prepare(args.tasks, args.collection_code, args.out)
        _load_plan(Path(plan["root"]) / "PLAN.json")
        print(plan["root"] + "/PLAN.json")
    else:
        print(run(args.plan, execute_gpu=args.execute_gpu, resume=args.resume)["status"])


if __name__ == "__main__":
    main()
