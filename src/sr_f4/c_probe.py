"""Bounded TOPK development probe: fixed draws, independent GPU workers, CPU report.

Never trains, reads confirmation data, or modifies the historical C gate. New
answers retain native scores/v1 labels; the frozen r5 measurement is a sidecar.
"""

# Chinese report prose uses full-width punctuation intentionally.
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from srf4_ref.sampling import seed_for

from . import c_probe_data as data
from .common import now, read, rows, write, write_rows

SHARDS = 4


def task_matrix():
    # Round-robin blocks make the initial allocation cover all three parents.
    probes = [
        dict(
            task_id=f"CTOPK-b{block}-s{shard}",
            kind="c_topk_probe",
            block=block,
            shard=shard,
            shards=SHARDS,
            dependencies=[],
        )
        for shard in range(SHARDS)
        for block in range(3)
    ]
    return [
        *probes,
        dict(
            task_id="CTOPK-report",
            kind="c_topk_report",
            dependencies=[p["task_id"] for p in probes],
        ),
    ]


def assignments(root, block, shard):
    if (
        type(block) is not int
        or block not in range(3)
        or type(shard) is not int
        or shard not in range(SHARDS)
    ):
        raise ValueError("Unregistered probe block/shard")
    result = [
        dict(g, protocol="structured", panel=data.SPLIT)
        for g in rows(Path(root) / "DATA/PROBE_GROUPS.jsonl")
        if g["block"] == f"b{block}" and g["group_index"] % SHARDS == shard
    ]
    if len(result) != data.QUERY_COUNT // SHARDS:
        raise ValueError("Probe shard inventory differs from the frozen budget")
    for group in result:
        group["seeds"] = [
            seed_for(
                data.GENERATION_SEED,
                "C_TOPK_DEV_R1",
                block,
                group["group_index"],
                group["qid"],
                draw,
            )
            for draw in range(data.DRAWS_PER_GROUP)
        ]
    return result


def prepare(root, source_root, revision):
    """Create an isolated root using already registered model/parent references."""
    from sr_f3_ce.rendering import render_chart

    from .pool import init_pool
    from .verification_v3 import VERIFIER_VERSION

    root, source_root = Path(root), Path(source_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Preserve the existing probe; do not prepare over live outputs")
    if VERIFIER_VERSION != data.MEASUREMENT_VERSION:
        raise ValueError("Prospective probe measurement version differs")
    source = read(source_root / "CONFIG.json")
    config = {
        k: source[k] for k in ("model_path", "parents", "python", "qos", "account", "partition")
    }
    if set(config["parents"]) != {"0", "1", "2"}:
        raise ValueError("The three registered parent references are required")
    config.update(
        experiment_id=data.VERSION,
        source_root=str(source_root),
        code_revision=revision,
        gpu_limit=8,
        max_submit_jobs=5,
    )
    data.prepare_data(root / "DATA")
    tasks = list(rows(root / "DATA/TASKS.jsonl"))
    rendered, images, geometry = {}, [], []
    for task in tasks:
        world_id = task["world_id"]
        if world_id not in rendered:
            payload, metrics = render_chart(task)
            relative = f"images/{world_id}.png"
            destination = root / "DATA" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            # Native runtime requires an image identity; compute once from the
            # newly rendered bytes, without scanning model or old output files.
            rendered[world_id] = dict(
                image_file=relative, image_sha256=hashlib.sha256(payload).hexdigest()
            )
            geometry.append(dict(world_id=world_id, **metrics))
        images.append(dict(qid=task["qid"], **rendered[world_id]))
    write_rows(root / "DATA/IMAGE_MANIFEST.jsonl", images)
    write_rows(root / "DATA/RENDER_GEOMETRY.jsonl", geometry)
    write(root / "CONFIG.json", config)
    matrix = task_matrix()
    write(root / "TASKS.json", matrix)
    seeds = [
        dict(task_id=s["task_id"], assignments=assignments(root, s["block"], s["shard"]))
        for s in matrix
        if s["kind"] == "c_topk_probe"
    ]
    write(root / "DRAW_PLAN.json", seeds)
    frozen = dict(
        data.plan(),
        at=now(),
        code_revision=revision,
        parent_references=config["parents"],
        model_path=config["model_path"],
        scientific_status="PROSPECTIVE_DEVELOPMENT_PROBE_NOT_C_TRAINING",
        training_updates=0,
        confirmation_access=False,
        group_resume="same identity and per-draw seed; retain interrupted attempts",
        generation=dict(
            protocol="structured-v2",
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=1024,
            do_sample=True,
            training=False,
        ),
    )
    write(root / "FROZEN.json", frozen)
    result = init_pool(root, matrix)
    write(
        root / "READY.json",
        dict(
            at=now(),
            status="READY",
            images=len(rendered),
            answers=data.RESPONSE_COUNT,
            tasks=len(matrix),
        ),
    )
    return result


def require_frozen(root):
    from .verification_v3 import VERIFIER_VERSION

    frozen = read(Path(root) / "FROZEN.json")
    config = read(Path(root) / "CONFIG.json")
    if (
        any(frozen.get(k) != v for k, v in data.plan().items())
        or frozen["measurement_version"] != VERIFIER_VERSION
        or frozen["training_updates"] != 0
        or frozen["confirmation_access"] is not False
        or config["model_path"] != frozen["model_path"]
        or config["parents"] != frozen["parent_references"]
        or config["code_revision"] != frozen["code_revision"]
    ):
        raise ValueError("Probe scope or measurement differs from the prospective freeze")
    return frozen


def coverage(root):
    """Small file-presence inventory, including when a failed shard blocks report.

    The full report validates contents. This receipt never certifies responses
    from filenames alone; its fixed denominators include every registered group.
    """
    root = Path(root)
    cells, missing = defaultdict(Counter), []
    for group in rows(root / "DATA/PROBE_GROUPS.jsonl"):
        present = (
            root / "PROBES" / group["block"] / "groups" / f"{group['group_index']:05d}.json"
        ).is_file()
        for key in (
            "all",
            f"cell/{group['cell']}",
            f"block/{group['block']}",
            f"cell_block/{group['cell']}/{group['block']}",
        ):
            cells[key]["planned_groups"] += 1
            cells[key]["planned_answers"] += group["G"]
            cells[key]["present_group_files"] += present
            cells[key]["missing_groups"] += not present
        if not present:
            missing.append(group["group_id"])
    result = dict(
        at=now(),
        status="FILES_PRESENT_CONTENTS_NOT_YET_VALIDATED"
        if not missing
        else "INCOMPLETE_TECHNICAL",
        counts=dict(cells),
        missing_groups=missing,
    )
    write(root / "ANALYSIS/COVERAGE.json", result)
    return result


def run_slice(root, spec):
    from .run import group, image_index, task_index
    from .runtime import load

    root = Path(root)
    require_frozen(root)
    block, shard = spec["block"], spec["shard"]
    assigned = assignments(root, block, shard)
    registered = next(
        s["assignments"] for s in read(root / "DRAW_PLAN.json") if s["task_id"] == spec["task_id"]
    )
    if assigned != registered:
        raise ValueError("Draw plan differs; do not generate")
    tasks, images = task_index(root), image_index(root)
    runtime, _, _ = load(root, block, spec["task_id"])
    directory = root / "PROBES" / f"b{block}"
    count = 0
    for position, assignment in enumerate(assigned, 1):
        task = tasks[assignment["qid"]]
        if task["split"] != data.SPLIT or task["family"] != "TOPK":
            raise ValueError("Only the frozen new TOPK development source may be sampled")
        values = group(
            root,
            runtime,
            task,
            images[task["qid"]],
            assignment,
            directory / "groups",
            task_id=f"{data.SPLIT}-b{block}",
            block=block,
            master=data.GENERATION_SEED,
            source=data.SPLIT,
        )
        count += len(values)
        write(
            directory / f"PROGRESS_s{shard}.json",
            dict(at=now(), answers=count, groups_done=position, target_groups=len(assigned)),
        )
    return dict(
        status="PROBE_SLICE_COMPLETE", answers=count, groups=len(assigned), block=block, shard=shard
    )


def report(root):
    from .credit_design_replay import POLICIES, diagnose_group, summarize
    from .replay import replay_group
    from .run import task_index
    from .training import native_advantages
    from .verification_v3 import VERIFIER_VERSION, verify_semantics

    root = Path(root)
    require_frozen(root)
    completeness = coverage(root)
    if completeness["missing_groups"]:
        return completeness
    tasks = task_index(root)
    expected_draws = {
        (g["block"], g["group_index"]): g
        for spec in read(root / "DRAW_PLAN.json")
        for g in spec["assignments"]
    }
    labels, diagnostics, strata = [], [], defaultdict(list)
    seen, measurement = set(), Counter()
    native = defaultdict(Counter)
    for assignment in rows(root / "DATA/PROBE_GROUPS.jsonl"):
        block = int(assignment["block"][1:])
        saved = read(
            root / "PROBES" / f"b{block}" / "groups" / f"{assignment['group_index']:05d}.json"
        )
        if (
            saved["group_id"] != assignment["group_id"]
            or len(saved["rows"]) != data.DRAWS_PER_GROUP
        ):
            raise ValueError("Missing or mismatched probe group; retain raw outputs")
        task = tasks[assignment["qid"]]
        joined = []
        expected = expected_draws[(assignment["block"], assignment["group_index"])]
        for draw, record in enumerate(saved["rows"]):
            if (
                record["row_id"] in seen
                or record["qid"] != task["qid"]
                or record["seed"] != expected["seeds"][draw]
                or record["draw"] != draw
                or record["source"] != data.SPLIT
                or record["block"] != block
                or record["arm"] != "PARENT"
                or record["row_id"] != f"{saved['group_id']}/d{draw:02d}"
                or record["group_id"] != saved["group_id"]
            ):
                raise ValueError("Duplicate or mismatched probe answer identity")
            seen.add(record["row_id"])
            label = verify_semantics(task, record, record["score"])
            for key in (
                "all",
                f"cell/{task['cell']}",
                f"block/b{block}",
                f"cell_block/{task['cell']}/b{block}",
                f"boundary/{task['boundary_subtype']}",
            ):
                native[key]["answers"] += 1
                native[key]["technical_invalid"] += record.get("generation_status") != "COMPLETE"
                for field in (
                    "J",
                    "value_correct",
                    "certificate_sufficient",
                    "presentation",
                    "finish",
                ):
                    native[key][field] += record["score"][field] == 1
            for dimension in "PS":
                measurement[
                    (task["cell"], f"b{block}", dimension, label["dimensions"][dimension]["status"])
                ] += 1
            joined.append(dict(task=task, record=record, score=record["score"], label=label))
        replay = replay_group(
            dict(
                group_id=saved["group_id"],
                source=data.SPLIT,
                step=0,
                block=f"b{block}",
                family="TOPK",
                world_id=task["world_id"],
                rows=joined,
            ),
            native_advantages,
        )
        replay.update(
            verifier_version=VERIFIER_VERSION,
            cell=task["cell"],
            development_only=True,
            frozen_gate_eligible=False,
            measurement_mode="PROGRAMMATIC_ONLY",
        )
        labels.append(replay)
        diagnostic = diagnose_group(replay)
        diagnostics.append(diagnostic)
        for key in (
            "all",
            f"cell/{task['cell']}",
            f"block/b{block}",
            f"cell_block/{task['cell']}/b{block}",
            f"boundary/{task['boundary_subtype']}",
        ):
            strata[key].append(diagnostic)
    if len(seen) != data.RESPONSE_COUNT or len(labels) != data.GROUP_COUNT:
        raise ValueError("Incomplete probe budget")
    result = dict(
        status=(
            "INCOMPLETE_TECHNICAL"
            if native["all"]["technical_invalid"]
            else "PROBE_COMPLETE_NOT_C_TRAINING_AUTHORIZATION"
        ),
        at=now(),
        answers=len(seen),
        groups=len(labels),
        verifier_version=VERIFIER_VERSION,
        training_updates=0,
        confirmation_access=False,
        native_statistics={
            key: dict(
                counts=dict(counts),
                rates={
                    field: count / counts["answers"]
                    for field, count in counts.items()
                    if field != "answers"
                },
            )
            for key, counts in native.items()
        },
        strata={
            key: {p.name: summarize(gs, p.name) for p in POLICIES} for key, gs in strata.items()
        },
        measurement=[
            dict(cell=k[0], block=k[1], dimension=k[2], label=k[3], count=v)
            for k, v in sorted(measurement.items())
        ],
    )
    directory = root / "ANALYSIS"
    write_rows(directory / "GROUP_REPLAY.jsonl", labels)
    write_rows(directory / "GROUP_DIAGNOSTICS.jsonl", diagnostics)
    write(directory / "SUMMARY.json", result)
    write(
        directory / "COVERAGE.json",
        dict(
            completeness,
            status=(
                "INCOMPLETE_TECHNICAL"
                if native["all"]["technical_invalid"]
                else "COMPLETE_CONTENTS_VALIDATED"
            ),
            validated_answers=len(seen),
            validated_groups=len(labels),
            technical_invalid_answers=native["all"]["technical_invalid"],
        ),
    )
    primary = result["strata"]["all"]["F_PRIME_KNOWN_75"]
    lines = [
        "# C 的 TOPK 小规模机会探针",
        "",
        f"记录 {len(seen)} 条回答、{len(labels)} 个组；三个固定父模型，没有训练更新。",
        f"状态：{result['status']}；技术无效回答：{native['all']['technical_invalid']}。",
        f"主口径 F′/75% 已知覆盖：N_act={primary['N_act']}，"
        f"N_sig={primary['N_sig']}，W_sig={primary['W_sig']}。",
        "",
        "| 分层 | 活跃组 | 选择分歧组 | 不同 world | 候选失败回答 | 已知回答 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, policies in result["strata"].items():
        p = policies["F_PRIME_KNOWN_75"]
        lines.append(
            f"| {key} | {p['N_act']} | {p['N_sig']} | {p['W_sig']} | "
            f"{p['candidate_rows']} | {p['known_rows']} |"
        )
    lines += [
        "",
        "| 原生指标分层 | 回答数 | J | 答案正确率 | 证据充分率 | 完成率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, value in result["native_statistics"].items():
        r = value["rates"]
        lines.append(
            f"| {key} | {value['counts']['answers']} | {r['J']:.2%} | "
            f"{r['value_correct']:.2%} | {r['certificate_sufficient']:.2%} | {r['finish']:.2%} |"
        )
    lines += [
        "",
        "完整 JSON 同时保留原 F、F/75% 已知子集、F′严格和 F′/75% 四种口径，"
        "以及 P/S 未知率、信号 L1 比例和剔除原因。",
        "这些是固定父模型的开发测量：分歧出现不等于训练有效，也不自动打开旧 C 门控。",
        "新题生成没有按模型表现或答案碰撞筛选；全部四个 cell、三个 block、技术失败记录均保留。",
        "若新回答出现未支持语法，应先做带 AI 来源的复核；"
        "不得把 UNKNOWN 填成期待的 S，或悄悄改本次冻结的测量版本。",
        "",
    ]
    (root / "REPORT_ZH.md").write_text("\n".join(lines))
    return dict(
        status=result["status"],
        answers=len(seen),
        groups=len(labels),
        N_act=primary["N_act"],
        N_sig=primary["N_sig"],
        W_sig=primary["W_sig"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "report", "coverage"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.command == "prepare":
        if not args.source_root or not args.revision:
            parser.error("prepare requires --source-root and --revision")
        print(prepare(args.root, args.source_root, args.revision))
    elif args.command == "report":
        print(report(args.root))
    else:
        print(coverage(args.root))


if __name__ == "__main__":
    main()
