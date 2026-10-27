"""Registered DEV_CAL census with immutable G8 resume units and complete identity checks.

No training, test access, resampling, or automatic PASS signing occurs here.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
from collections import Counter
from pathlib import Path

from sr_f4.common import append, now, read, write
from srf6_ref.census import summarize
from srf6_ref.core import FAMILIES, seed_for
from srf6_ref.credit import credit

from .runtime import PROTOCOL_VERSION, load
from .source_binding import file_sha256, object_sha256

PARENTS = ("b0", "b1", "b2")
FACTORS = ("P0", "P1")


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def census_inputs(root):
    root = Path(root)
    public = rows(root / "DATA/DEV_CAL/MODEL_INPUTS.jsonl")
    private = rows(root / "DATA/DEV_CAL/TASKS_PRIVATE.jsonl")
    if len(public) != 384 or len(private) != 384:
        raise ValueError("Census requires all 384 DEV_CAL factor/question rows")
    tasks = {row["qid"]: row for row in private}
    if len(tasks) != 384 or len({r["qid"] for r in public}) != 384:
        raise ValueError("Duplicate census qid")
    if set(tasks) != {r["qid"] for r in public}:
        raise ValueError("Public/private DEV_CAL qids differ")
    expected = Counter({(f, p): 32 for f in FAMILIES for p in FACTORS})
    if Counter((r["family"], r["factor"]) for r in private) != expected:
        raise ValueError("DEV_CAL family/factor matrix is incomplete")
    lineages = {}
    for item in public:
        task = tasks[item["qid"]]
        if any(
            item[k] != task[k] for k in ("qid", "root_id", "family", "factor", "variant", "split")
        ):
            raise ValueError("DEV_CAL public/private identity differs")
        if item["split"] != "DEV_CAL" or item["protocol"] != "structured_v4":
            raise PermissionError("Census may only access registered DEV_CAL")
        if set(item["model_input"]) != {"prompt", "image_file", "image_sha256"}:
            raise PermissionError("Private metadata at model input boundary")
        lineages.setdefault((task["root_id"], task["factor"]), []).append(task["variant"])
    if len(lineages) != 96 or any(sorted(v) != [0, 1, 2, 3] for v in lineages.values()):
        raise ValueError("Every registered root/factor requires all four views")
    return sorted(public, key=lambda row: row["qid"]), tasks


def _spec(root, config, public, tasks, policy_hashes):
    if tuple(config["model"]["parents"]) != PARENTS or set(policy_hashes) != set(PARENTS):
        raise PermissionError("All three registered parent identities required")
    if any(
        not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
        for h in policy_hashes.values()
    ):
        raise ValueError("Parent content SHA256 identities required")
    return dict(
        phase="PARENT_CENSUS",
        split="DEV_CAL",
        protocol_version=PROTOCOL_VERSION,
        config_file_sha256=file_sha256(Path(root) / "CONFIG.json"),
        data_manifest_sha256=file_sha256(Path(root) / "DATA/MANIFEST.json"),
        parents=list(PARENTS),
        factors=list(FACTORS),
        questions_per_parent=384,
        expected_response_rows=9216,
        draws_per_question=8,
        optimizer_updates=0,
        parent_policy_hashes=dict(policy_hashes),
        seed_rule="seed_for(master,census,parent,root_id,variant,draw); factor excluded",
        duplicate_sampling_note=(
            "Identical factor inputs/seeds are correlated; never independent replicates"
        ),
        questions=[
            dict(
                qid=r["qid"],
                root_id=r["root_id"],
                family=r["family"],
                factor=r["factor"],
                variant=r["variant"],
                public_sha256=object_sha256(r),
                private_sha256=object_sha256(tasks[r["qid"]]),
            )
            for r in public
        ],
    )


def freeze_spec(root, parent_policy_hashes):
    root = Path(root)
    config = read(root / "CONFIG.json")
    public, tasks = census_inputs(root)
    spec = _spec(root, config, public, tasks, parent_policy_hashes)
    path = root / "CENSUS_SPEC_FROZEN.json"
    with path.open("x") as stream:
        json.dump(spec, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return spec


def validate_census_spec(root, config, public, tasks):
    spec = read(Path(root) / "CENSUS_SPEC_FROZEN.json")
    if spec != _spec(root, config, public, tasks, spec.get("parent_policy_hashes", {})):
        raise PermissionError(
            "Census freeze differs from exact current source/data/config identities"
        )
    return spec


def group_spec(config, parent, item, task, frozen):
    if parent not in PARENTS:
        raise ValueError("Unknown parent")
    return dict(
        phase="PARENT_CENSUS",
        parent=parent,
        factor=item["factor"],
        qid=item["qid"],
        root_id=item["root_id"],
        family=item["family"],
        variant=item["variant"],
        group_id=f"{parent}/{item['qid']}",
        policy_hash=frozen["parent_policy_hashes"][parent],
        frozen_spec_sha256=object_sha256(frozen),
        model_input_sha256=object_sha256(item["model_input"]),
        public_sha256=object_sha256(item),
        private_sha256=object_sha256(task),
        seeds=[
            seed_for(config["master_seed"], "census", parent, item["root_id"], item["variant"], i)
            for i in range(8)
        ],
        shuffle_seed=seed_for(
            config["master_seed"],
            "census_shuffle",
            parent,
            item["factor"],
            f"{parent}/{item['qid']}",
        ),
    )


def annotate_group(raw, task, spec):
    from .measurement import score_response

    if len(raw) != 8 or [r["seed"] for r in raw] != spec["seeds"]:
        raise ValueError("Census group requires its original eight registered draws")
    tensor_hashes = {row.get("policy_tensor_hash") for row in raw}
    if len(tensor_hashes) != 1:
        raise PermissionError("Census G8 mixes loaded policy tensors")
    tensor_hash = next(iter(tensor_hashes))
    if (
        not isinstance(tensor_hash, str)
        or len(tensor_hash) != 64
        or any(c not in "0123456789abcdef" for c in tensor_hash)
    ):
        raise PermissionError("Actual loaded policy tensor SHA256 is required")
    result = []
    for draw, row in enumerate(raw):
        if row.get("parent_content_hash") != spec["policy_hash"]:
            raise PermissionError("Loaded parent content differs from the frozen census binding")
        if row.get("generation_status") != "COMPLETE" or row.get("technical_validation_errors"):
            raise RuntimeError("Preserved technical failure must be diagnosed before completion")
        tokens, logps = row["tokens"], row["sampler_logprobs"]
        if not 1 <= len(tokens) <= 1536 or len(logps) != len(tokens):
            raise ValueError("Original tokens/probabilities are incomplete or over budget")
        if any(type(t) is not int or t < 0 for t in tokens) or not all(map(math.isfinite, logps)):
            raise ValueError("Invalid sampled tokens or nonfinite probabilities")
        if object_sha256(row["model_input"]) != spec["model_input_sha256"]:
            raise PermissionError("Raw response input differs from frozen question")
        if (
            row["adapter_identity"].get("parent") != spec["parent"]
            or row["adapter_identity"].get("step") != 0
        ):
            raise PermissionError("Census response did not use its corresponding frozen parent")
        label = score_response(
            row["full_text"],
            task["world"],
            task["query"],
            finished=row["finish_reason"] == "balanced",
            finish_reason=row["finish_reason"],
        )
        identity = {
            key: spec[key]
            for key in (
                "phase",
                "parent",
                "factor",
                "qid",
                "root_id",
                "family",
                "group_id",
                "policy_hash",
            )
        }
        result.append(
            dict(
                row,
                **label,
                **identity,
                draw=draw,
                row_id=f"{spec['parent']}/{task['qid']}/{draw}",
                score=label,
                raw_text=row["full_text"],
                raw_token_ids=tokens,
                sample_logprobs=logps,
                frozen_spec_sha256=spec["frozen_spec_sha256"],
            )
        )
    return result


def counterfactual(records, spec):
    return {
        mode: credit(records, mode, seed=spec["shuffle_seed"])
        for mode in ("J", "COND", "GLOBAL", "SHUF")
    }


def validate_group(group, spec, task):
    if group.get("status") != "COMPLETE" or group.get("spec") != spec:
        raise PermissionError("Persisted census group identity changed")
    records = group.get("records", [])
    if [r.get("draw") for r in records] != list(range(8)):
        raise ValueError("Persisted census G8 draws missing or duplicated")
    rebuilt = annotate_group(records, task, spec)
    if rebuilt != records or group.get("credit") != counterfactual(records, spec):
        raise PermissionError("Persisted labels/credits differ from frozen recomputation")
    return records


def check_generation_alignment(runtime, records, spec, output):
    """Replay the first registered G8 without drawing new responses or updating."""
    from mm_core.training import capture_rng, restore_rng, state_hash, trainable_state

    torch = runtime.torch
    if len(records) != 8:
        raise ValueError("Alignment must replay exactly the registered first G8")
    output = Path(output)
    expected = dict(
        frozen_spec_sha256=spec["frozen_spec_sha256"],
        group_spec_sha256=object_sha256(spec),
        raw_trajectories_sha256=object_sha256(
            [
                {k: row[k] for k in ("tokens", "sampler_logprobs", "prompt_tensor_hash")}
                for row in records
            ]
        ),
    )
    if output.exists():
        previous = read(output)
        if previous.get("status") != "PASS" or previous.get("identity") != expected:
            raise PermissionError(
                "Stored generation alignment failed or changed; preserve and diagnose"
            )
        return previous
    before = state_hash(trainable_state(runtime.model))
    rng = capture_rng()
    traces = []
    result = dict(
        status="FAIL",
        identity=expected,
        phase="PARENT_CENSUS",
        parent=spec["parent"],
        group_id=spec["group_id"],
        reviewer_type="PROGRAM",
        execution_origin="SERVER_ACTUAL",
        new_generated_responses=0,
        optimizer_updates=0,
        original_registered_responses=8,
    )
    try:
        differences, truncated = [], []
        for row in records:
            forwarded = runtime.batch_sequence_forward(
                [row], purpose="census_token_alignment", grad=False
            )
            if len(forwarded) != 1 or forwarded[0].numel() != len(row["tokens"]):
                raise RuntimeError("Actual saved-token teacher forcing is not aligned")
            current = forwarded[0].detach().float().cpu()
            sampler = torch.tensor(row["sampler_logprobs"], dtype=torch.float32)
            if current.ndim != 1 or not bool(torch.isfinite(current).all()):
                raise FloatingPointError(
                    "Nonfinite or invalid actual teacher-forced log probabilities"
                )
            delta = current - sampler
            differences.append(delta.abs())
            truncated.append(delta > math.log(2.0))
            traces.append(
                dict(
                    row_id=row.get("row_id"),
                    token_ids=row["tokens"],
                    sampled_logprobs=sampler.tolist(),
                    current_logprobs=current.tolist(),
                )
            )
        from sr_f12.training import enforce_probability_gate, probability_diagnostics

        diagnostics = probability_diagnostics(differences, truncated)
        result["diagnostics"] = diagnostics
        enforce_probability_gate(diagnostics)
        if diagnostics["sampler_train_max_absolute_difference"] > 5.0:
            raise RuntimeError("Maximum sampler/train discrepancy exceeds 5 nat")
        if state_hash(trainable_state(runtime.model)) != before:
            raise RuntimeError("Read-only generation replay changed trainable parameters")
        result.update(
            status="PASS",
            parameter_tensor_sha256=before,
            checked_sequences=8,
            actual_token_path="saved completion IDs; one sequence per forward",
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        restore_rng(rng)
        result["traces"] = traces
        write(output, result)
    return result


def validate_alignment_receipt(path, spec):
    report = read(path)
    if (
        report.get("status") != "PASS"
        or report.get("phase") != "PARENT_CENSUS"
        or report.get("identity", {}).get("frozen_spec_sha256") != spec["frozen_spec_sha256"]
        or report.get("identity", {}).get("group_spec_sha256") != object_sha256(spec)
        or report.get("original_registered_responses") != 8
        or report.get("new_generated_responses") != 0
        or report.get("optimizer_updates") != 0
        or report.get("checked_sequences") != 8
    ):
        raise PermissionError("Missing or failed real first-G8 token replay")
    d = report["diagnostics"]
    keys = (
        "sampler_train_mean_absolute_difference",
        "sampler_train_max_absolute_difference",
        "tis_truncated_fraction",
    )
    if any(not math.isfinite(d[k]) or d[k] < 0 for k in keys) or len(report.get("traces", [])) != 8:
        raise PermissionError("Incomplete or nonfinite persisted probability replay")
    if (
        d["sampler_train_mean_absolute_difference"] > 0.01
        or d["sampler_train_max_absolute_difference"] > 5
        or d["tis_truncated_fraction"] > 0.01
    ):
        raise PermissionError("Persisted replay exceeds registered probability gates")
    return report


def run(root, parent, shard=0, shards=1):
    from .gates import require_census_ready

    root = Path(root)
    require_census_ready(root, config_path=root / "CONFIG.json")
    config = read(root / "CONFIG.json")
    if (
        parent not in PARENTS
        or type(shards) is not int
        or type(shard) is not int
        or not 0 <= shard < shards
    ):
        raise ValueError("Invalid registered parent/shard")
    public, tasks = census_inputs(root)
    frozen = validate_census_spec(root, config, public, tasks)
    selected = public[shard::shards]
    task_id = f"CENSUS-{parent}-{shard}-of-{shards}"
    runtime = None
    actual_calls = 0
    first = selected[0]
    first_spec = group_spec(config, parent, first, tasks[first["qid"]], frozen)
    alignment_path = root / "CENSUS/alignment" / f"{task_id}.json"
    aligned = alignment_path.exists()
    if aligned:
        validate_alignment_receipt(alignment_path, first_spec)
    for item in selected:
        task = tasks[item["qid"]]
        spec = group_spec(config, parent, item, task, frozen)
        directory = root / "CENSUS/groups" / parent
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{item['qid']}.json"
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if path.exists():
                existing = validate_group(read(path), spec, task)
                if not aligned:
                    runtime, _, _ = load(root, parent, task_id, training=False)
                    check_generation_alignment(runtime, existing, spec, alignment_path)
                    aligned = True
                continue
            raw_path = root / "CENSUS/raw" / parent / f"{item['qid']}.jsonl"
            attempt = root / "CENSUS/attempts" / parent / f"{item['qid']}.json"
            if attempt.exists():
                if read(attempt)["spec"] != spec:
                    raise PermissionError("Interrupted census attempt identity changed")
                raw = rows(raw_path) if raw_path.exists() else []
                if len(raw) != 8:
                    raise RuntimeError("Partial census attempt preserved; no automatic resampling")
            else:
                if raw_path.exists():
                    raise RuntimeError("Orphan raw census evidence must be diagnosed")
                if runtime is None:
                    runtime, _, _ = load(root, parent, task_id, training=False)
                write(attempt, dict(status="STARTED", at=now(), spec=spec, task_id=task_id))
                runtime.generate_group(
                    item["model_input"],
                    root,
                    spec["seeds"],
                    protocol="structured_v4",
                    on_completion=lambda row, target=raw_path: append(target, row),
                )
                actual_calls += 8
                raw = rows(raw_path)
            records = annotate_group(raw, task, spec)
            if not aligned:
                if runtime is None:
                    runtime, _, _ = load(root, parent, task_id, training=False)
                check_generation_alignment(runtime, records, spec, alignment_path)
                aligned = True
            group = dict(
                status="COMPLETE",
                at=now(),
                spec=spec,
                records=records,
                credit=counterfactual(records, spec),
            )
            validate_group(group, spec, task)
            write(path, group)
            print(json.dumps(dict(task_id=task_id, qid=item["qid"], status="COMPLETE")), flush=True)
    receipt = dict(
        status="COMPLETE",
        phase="PARENT_CENSUS",
        task_id=task_id,
        groups=len(selected),
        actual_generated_responses_this_process=actual_calls,
        optimizer_updates=0,
        gate_status="NOT_SIGNED",
        generation_alignment_sha256=file_sha256(alignment_path),
        frozen_spec_sha256=object_sha256(frozen),
    )
    write(root / "CENSUS/workers" / f"{task_id}.json", receipt)
    return receipt


def complete_summary(root):
    root = Path(root)
    config = read(root / "CONFIG.json")
    public, tasks = census_inputs(root)
    frozen = validate_census_spec(root, config, public, tasks)
    records = []
    expected_paths = set()
    for parent in PARENTS:
        for item in public:
            path = root / "CENSUS/groups" / parent / f"{item['qid']}.json"
            expected_paths.add(path)
            spec = group_spec(config, parent, item, tasks[item["qid"]], frozen)
            records.extend(validate_group(read(path), spec, tasks[item["qid"]]))
    actual_paths = set((root / "CENSUS/groups").glob("*/*.json"))
    if actual_paths != expected_paths:
        raise PermissionError("Unexpected or missing census groups outside exact DEV_CAL matrix")
    if len(records) != 9216 or len({r["row_id"] for r in records}) != 9216:
        raise ValueError("Complete census must retain exactly 9216 registered rows")
    if Counter((r["parent"], r["factor"]) for r in records) != Counter(
        {(p, f): 1536 for p in PARENTS for f in FACTORS}
    ):
        raise ValueError("Incomplete census parent/factor denominators")
    worker_paths = list((root / "CENSUS/workers").glob("*.json"))
    covered_parents = set()
    for worker_path in worker_paths:
        worker = read(worker_path)
        if worker.get("status") != "COMPLETE" or worker.get("frozen_spec_sha256") != object_sha256(
            frozen
        ):
            raise PermissionError("Incomplete or changed census worker evidence")
        task_id = worker["task_id"]
        _, parent, shard_text, _, shards_text = task_id.split("-")
        shard, shards = int(shard_text), int(shards_text)
        if parent not in PARENTS or not 0 <= shard < shards <= len(public):
            raise PermissionError("Invalid completed worker identity")
        first = public[shard::shards][0]
        alignment_path = root / "CENSUS/alignment" / f"{task_id}.json"
        validate_alignment_receipt(
            alignment_path, group_spec(config, parent, first, tasks[first["qid"]], frozen)
        )
        if file_sha256(alignment_path) != worker.get("generation_alignment_sha256"):
            raise PermissionError("Worker probability evidence changed")
        covered_parents.add(parent)
    if covered_parents != set(PARENTS):
        raise PermissionError("All parents require completed real-token probability evidence")
    for parent in PARENTS:
        if len({r.get("policy_tensor_hash") for r in records if r["parent"] == parent}) != 1:
            raise PermissionError("Parent census mixed different loaded policy tensors")
    result = summarize(records, config["master_seed"])
    result.update(
        status="COMPLETE_NOT_GATE_SIGNED",
        frozen_spec_sha256=object_sha256(frozen),
        source_rows=9216,
        optimizer_updates=0,
        real_measurement_review="NOT_RUN",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "summarize"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", choices=PARENTS)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    result = (
        run(args.root, args.parent, args.shard, args.shards)
        if args.command == "run"
        else complete_summary(args.root)
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
