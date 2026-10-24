"""Frozen Local-credit pilot orchestration; independent of the historical C gate."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from srf4_ref.sampling import seed_for

from . import c_local_data as data
from .c_pilot_images import local_image
from .common import now, read, rows, write, write_rows

SHARDS = 4


def protocol():
    return dict(
        reward=dict(
            policy="RELIABLE_ALL_OUTCOMES_LOCAL_CREDIT_V1",
            lambda_evidence=0.25,
            lambda_selection=0.25,
            eligible="valid_finished_defined_both_J0_and_J1",
            shared_known_cohort="P_and_S_known_minimum_two_and_75_percent_eligible",
            evidence="P_minus_known_mean_P",
            selection="P_times_S_minus_known_mean_P_times_S",
            shuffle="one_uniform_permutation_within_joint_J_P_strata_no_retry",
            reduction="native_full_sequence_mean_plus_each_auxiliary_own_span_mean",
            normalization="no_auxiliary_std_no_combined_second_whitening",
            parser=data.MEASUREMENT_VERSION,
            credit_scope="evidence_value_tokens_and_selection_declaration_tokens_only",
            conflict="selection_and_compute_category_disagreement_is_UNKNOWN",
        ),
        optimizer=dict(
            name="AdamW", lr=1e-5, betas=[0.9, 0.999], eps=1e-8,
            weight_decay=0, scheduler="constant", grad_clip=1.0,
            kl_beta=0.02, policy_clip=0.2, tis_cap=2.0,
            updates=data.STEPS, answers_per_update=128, reference="fixed_initial_parent",
        ),
        generation=dict(
            protocol="structured-v2-with-selection-declaration-v1", temperature=1.0,
            top_p=1.0, top_k=0, max_new_tokens=1024, do_sample=True,
            chartqa_protocol="chartqa_plain", retention_prompt="original_unchanged_v2",
            all_new_endpoint_prompts="identical_for_parent_and_all_four_arms",
        ),
        statistics=dict(
            primary="new_value_correct_SELECT_minus_SHUF",
            secondary=["native_J", "SELECT_minus_J", "SELECT_minus_EVID", "SELECT_minus_PARENT"],
            bootstrap_replicates=2000, bootstrap_seed=2026102406, confidence=0.95,
            unit="paired_root_lineage_within_family_equal_family_equal_parent_weight",
            interpretation="conditional_descriptive_not_training_seed_CI",
            per_block_required=True, adverse_results_required=True, adaptive_extension=False,
        ),
        confirmation_access=False,
        review_origin="AI_development_checks_not_human_review",
        historical_C_gate="UNCHANGED",
        authorization="2026-10-24 user authorized new tasks and local-credit four-arm pilot",
    )


def prelaunch_gate():
    """Require the committed CPU/replay receipt before any preparation or model load."""
    path = (Path(__file__).resolve().parents[2]
            / "docs/sr_f4/validation/C_LOCAL_CPU_GATE_20261024.json")
    gate = read(path)
    if (gate.get("schema") != "C_LOCAL_PRELAUNCH_V1" or gate.get("status") != "PASS"
            or gate.get("data_version") != data.VERSION
            or gate.get("measurement_version") != data.MEASUREMENT_VERSION
            or gate.get("protocol") != protocol()
            or gate.get("data_plan") != data.plan()
            or gate.get("offline_replay", {}).get("counts", {}).get("answers") != 6144
            or gate.get("checks", {}).get("cpu_gradient") != "PASS"
            or gate.get("checks", {}).get("integration") != "PASS"
            or gate.get("checks", {}).get("review") != "PASS"):
        raise PermissionError("New pilot requires the matching completed CPU/replay gate")
    return gate


def task_matrix():
    matrix = []
    # Each completed model's evaluations appear before later training tasks.
    # A free worker therefore starts ready evaluation without a matrix barrier.
    for arm in data.ARMS:
        for block in data.BLOCKS:
            train_id = f"CLC-T-b{block}-{arm}"
            matrix.append(
                dict(task_id=train_id, kind="c_local_train", block=block, arm=arm, dependencies=[])
            )
            matrix.extend(
                dict(
                    task_id=f"CLC-E-b{block}-{arm}-s{s}",
                    kind="c_local_eval",
                    block=block,
                    arm=arm,
                    shard=s,
                    shards=SHARDS,
                    dependencies=[train_id],
                )
                for s in range(SHARDS)
            )
    for block in data.BLOCKS:
        matrix.extend(
            dict(
                task_id=f"CLC-E-b{block}-PARENT-s{s}",
                kind="c_local_eval",
                block=block,
                arm="PARENT",
                shard=s,
                shards=SHARDS,
                dependencies=[],
            )
            for s in range(SHARDS)
        )
    matrix.append(
        dict(
            task_id="CLC-REPORT", kind="c_local_report", dependencies=[s["task_id"] for s in matrix]
        )
    )
    return matrix


def build_eval_groups(tasks, block):
    groups = list(data.eval_assignments(tasks, block))
    panels = [("OLD_TEST_ID", [t for t in tasks if t.get("pool") == "TEST_ID"], 4, "structured")]
    if block == 0:
        panels.append(
            ("CHARTQA", [t for t in tasks if t.get("pool") == "CHARTQA"], 1, "chartqa_plain")
        )
    for panel, selected, draws, mode in panels:
        for task in sorted(selected, key=lambda t: t["qid"]):
            groups.append(
                dict(
                    qid=task["qid"],
                    group_index=len(groups),
                    G=draws,
                    protocol=mode,
                    panel=panel,
                    seeds=[
                        seed_for(data.MASTER, "eval", block, panel, task["qid"], draw)
                        for draw in range(draws)
                    ],
                )
            )
    if len(groups) != 512 + (250 if block == 0 else 0):
        raise ValueError("Endpoint task inventory differs from the fixed budget")
    return groups


def pool_task_matrix():
    """Scheduling capabilities are separate from frozen experimental inputs."""
    return [dict(spec, resume_supported=True)
            if spec["kind"] in ("c_local_train", "c_local_eval") else spec
            for spec in task_matrix()]


def prepare(root, source_root, revision):
    from sr_f3_ce.rendering import render_chart

    from .c_local_credit import VERIFIER_VERSION
    from .pool import init_pool

    root, source_root = Path(root), Path(source_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Pilot root is not empty; preserve existing work")
    if VERIFIER_VERSION != data.MEASUREMENT_VERSION:
        raise ValueError("Pilot parser differs from the registered local-credit measurement")
    gate = prelaunch_gate()
    source = read(source_root / "CONFIG.json")
    config = {
        k: source[k]
        for k in ("model_path", "parents", "python", "qos", "account", "partition", "old_root")
    }
    config.update(
        experiment_id=data.VERSION,
        source_root=str(source_root),
        code_revision=revision,
        gpu_limit=8,
        max_submit_jobs=5,
    )
    if set(config["parents"]) != {"0", "1", "2"}:
        raise ValueError("All three fixed parent blocks are required")
    if (config["qos"], config["account"], config["partition"]) != (
        "soujanya-poria-startfund-2026-03", "rose", "cluster02"
    ):
        raise PermissionError("New pilot must use the registered teacher allocation")
    tasks = data.generate_tasks()
    retained = data.select_retention(rows(source_root / "DATA/TASKS.jsonl"))
    external = data.select_chartqa(
        rows(Path(source["old_root"]) / "external/chartqa/TASKS_AUDIT_ONLY.jsonl")
    )
    routes = {r["qid"]: r for r in rows(source_root / "DATA/CHARTQA_IMAGES.jsonl")}
    external = [
        dict(
            t,
            split="C_LOCAL_CHARTQA_R1",
            prompt=routes[t["qid"]]["prompt"],
            world_id=t["qid"],
            root_id=t["qid"],
        )
        for t in external
    ]
    old_images = {r["qid"]: r for r in rows(source_root / "DATA/IMAGE_MANIFEST.jsonl")}
    rendered, images, geometry = {}, [], []
    for task in tasks:
        world_id = task["world_id"]
        if world_id not in rendered:
            payload, metrics = render_chart(task)
            relative = f"images/{world_id}.png"
            destination = root / "DATA" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            # Image identity is required by the existing runtime: derive once
            # from newly rendered bytes, with no old-model/file hash scan.
            rendered[world_id] = dict(
                image_file=relative, image_sha256=hashlib.sha256(payload).hexdigest()
            )
            geometry.append(dict(world_id=world_id, **metrics))
        images.append(dict(qid=task["qid"], **rendered[world_id]))
    for task in [*retained, *external]:
        image = dict((routes if task.get("pool") == "CHARTQA" else old_images)[task["qid"]])
        images.append(local_image(root, source_root, image))
    tasks = [*tasks, *retained, *external]
    if len({t["qid"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate pilot task identity")
    write_rows(root / "DATA/TASKS.jsonl", tasks)
    write_rows(root / "DATA/IMAGE_MANIFEST.jsonl", images)
    write(root / "FROZEN_INPUTS.json", dict(tasks=tasks, images=images))
    write_rows(root / "DATA/RENDER_GEOMETRY.jsonl", geometry)
    write(root / "CONFIG.json", config)
    for block in data.BLOCKS:
        write_rows(root / "DATA" / f"EVAL_GROUPS_b{block}.jsonl", build_eval_groups(tasks, block))
        write(
            root / "DATA" / f"TRAIN_PLAN_b{block}.json",
            {str(step): data.training_assignments(tasks, block, step)
             for step in range(1, data.STEPS + 1)},
        )
    matrix = task_matrix()
    write(root / "TASKS.json", matrix)
    frozen = dict(
        data.plan(),
        **protocol(),
        at=now(),
        code_revision=revision,
        parent_references=config["parents"],
        model_path=config["model_path"],
        scientific_status="PROSPECTIVE_BOUNDED_C_LOCAL_TRAINING_PILOT",
    )
    frozen["prelaunch_gate"] = gate
    write(root / "PRELAUNCH_GATE.json", gate)
    write(root / "FROZEN.json", frozen)
    result = init_pool(root, pool_task_matrix())
    write(
        root / "READY.json",
        dict(
            at=now(),
            status="READY",
            tasks=len(matrix),
            training_answers=18432,
            eval_answers=31970,
            images_created=len(rendered),
        ),
    )
    return result


def require_frozen(root):
    from .c_local_credit import VERIFIER_VERSION

    root = Path(root)
    frozen, config = read(root / "FROZEN.json"), read(root / "CONFIG.json")
    expected = dict(data.plan(), **protocol())
    if (
        any(frozen.get(k) != v for k, v in expected.items())
        or VERIFIER_VERSION != data.MEASUREMENT_VERSION
        or config["parents"] != frozen["parent_references"]
        or config["model_path"] != frozen["model_path"]
        or config["code_revision"] != frozen["code_revision"]
        or read(root / "TASKS.json") != task_matrix()
    ):
        raise ValueError("Pilot differs from the prospective freeze")
    if frozen.get("prelaunch_gate") != prelaunch_gate() or (
        read(root / "PRELAUNCH_GATE.json") != frozen["prelaunch_gate"]
    ):
        raise ValueError("Prelaunch receipt differs from the frozen checks")
    inputs = read(root / "FROZEN_INPUTS.json")
    tasks = list(rows(root / "DATA/TASKS.jsonl"))
    if (
        tasks != inputs["tasks"]
        or list(rows(root / "DATA/IMAGE_MANIFEST.jsonl")) != inputs["images"]
    ):
        raise ValueError("Pilot task or image manifest differs from the prospective freeze")
    for block in data.BLOCKS:
        training = read(root / "DATA" / f"TRAIN_PLAN_b{block}.json")
        expected_training = {
            str(s): data.training_assignments(tasks, block, s) for s in range(1, data.STEPS + 1)
        }
        if training != expected_training or list(
            rows(root / "DATA" / f"EVAL_GROUPS_b{block}.jsonl")
        ) != build_eval_groups(tasks, block):
            raise ValueError("Pilot draw plan differs from the prospective freeze")
    return frozen


def run_eval(root, spec):
    import torch

    from sr_f1.runtime import copy_parameters

    from .run import group, image_index, task_index
    from .runtime import load

    root = Path(root)
    require_frozen(root)
    block, arm, shard = spec["block"], spec["arm"], spec["shard"]
    if spec not in task_matrix() or spec["kind"] != "c_local_eval":
        raise ValueError("Unregistered pilot evaluation")
    for dependency in spec["dependencies"]:
        if read(root / "tasks" / dependency / "COMPLETE.json").get("status") != "COMPLETE":
            raise PermissionError("This endpoint's training is not complete")
    tasks, images = task_index(root), image_index(root)
    all_groups = list(rows(root / "DATA" / f"EVAL_GROUPS_b{block}.jsonl"))
    if all_groups != build_eval_groups(list(tasks.values()), block):
        raise ValueError("Endpoint draw plan differs from the freeze")
    assigned = [g for g in all_groups if g["group_index"] % SHARDS == shard]
    runtime, _, _ = load(root, block, spec["task_id"])
    if arm != "PARENT":
        checkpoint = root / "RUNS/C_LOCAL" / f"b{block}" / arm / "checkpoints/step-012.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            state["step"] != data.STEPS
            or state["identity"]["arm"] != arm
            or state["identity"]["block"] != block
        ):
            raise ValueError("Wrong trained endpoint")
        copy_parameters(runtime.model, state["policy"])
        runtime.identity["step"] = data.STEPS
        del state
    directory = root / "EVAL" / f"b{block}" / arm / f"shard{shard}"
    count = 0
    for number, assignment in enumerate(assigned, 1):
        task = tasks[assignment["qid"]]
        values = group(
            root,
            runtime,
            task,
            images[task["qid"]],
            assignment,
            directory / "groups",
            task_id=f"CLC-E-b{block}-{arm}",
            block=block,
            master=data.MASTER,
            step=0 if arm == "PARENT" else data.STEPS,
            arm=arm,
            source="C_LOCAL_ENDPOINT_R1",
        )
        count += len(values)
        write(
            directory / "PROGRESS.json",
            dict(at=now(), groups_done=number, groups_total=len(assigned), answers=count),
        )
    return dict(block=block, arm=arm, shard=shard, answers=count, groups=len(assigned))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    prepare(args.root, args.source_root, args.revision)


if __name__ == "__main__":
    main()
