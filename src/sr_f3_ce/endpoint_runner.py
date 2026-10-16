"""Executable 15-model sealed endpoints and content-bound final raw rescoring.

This layer has no reward-based scheduling or model selection. The orchestrator's
verified Slurm terminal receipts gate both generation and disclosure.
"""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from sr_f12.evaluation import durable_json, shard_lease

from .evaluation import (
    EXPERIMENT_ID,
    TEST_POOLS,
    endpoint_models,
    endpoint_slots,
    group_slots,
    object_hash,
    release_evaluation,
    score_record,
    sha_file,
    validate_record,
)
from .external import ExternalInputStore, load_official_scorer, score_external


def _read(path):
    return json.loads(Path(path).read_text())


def _run_revision(root):
    from .protocol_revision import revision_from_config

    return revision_from_config(_read(Path(root) / "plan/config/SR_F3_CE.json"))


def _lines(path):
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                raise ValueError("Blank artifact row")
            yield json.loads(line)


def _bounded(root, relative):
    relative, root = Path(relative), Path(root).resolve(strict=True)
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError("Artifact path escapes run root")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise PermissionError("Artifact resolves outside run root")
    return path


def _completed(root):
    """Use the same actual accounting, process, artifact and freeze checks as dispatch."""
    from .orchestration import Controller

    config = _read(Path(root) / "orchestration/CONFIG.json")
    controller = Controller(
        root,
        config["code_root"],
        python=config["python"],
        user=config["user"],
        pool_size=config["pool_size"],
        cpu_partition=config["cpu_policy"]["partition"],
        cpu_qos=config["cpu_policy"]["qos"],
        cpu_node=config["cpu_policy"]["cpu_node"],
    )
    return controller.completed()


def _require_terminal(root, *, evaluations=False):
    models = endpoint_models()
    required = {"train--" + m["model_id"] for m in models if m["arm"] != "PARENT"}
    if evaluations:
        required |= {"eval--" + m["model_id"] for m in models}
    missing = required - _completed(root)
    if missing:
        raise PermissionError("Scientific terminal receipts missing: " + ", ".join(sorted(missing)))


def prepare_external_panels(
    root, *, prior_manifest=None, history_path=None, legacy_image_root=None, sources_root=None
):
    """Run before outputs on an authorized CPU worker, after historical image export."""
    from .external import acquire_chartqapro, freeze_chartqapro, prepare_chartqa

    root = Path(root).resolve(strict=True)
    if any((root / "sealed/endpoint").glob("*/RAW_MANIFEST.json")):
        raise PermissionError("Cannot alter external panels after endpoint output exists")
    prior_manifest = Path(prior_manifest or root / "prior/raw/external/chartqa/TEST_MANIFEST.jsonl")
    history_path = Path(history_path or root / "prior/HISTORICAL_IMAGES.json")
    legacy_image_root = Path(legacy_image_root or root / "external/legacy_images")
    sources = Path(sources_root or root / "external/chartqapro_sources")
    history = _read(history_path)
    chartqa_root, pro_root = root / "external/chartqa", root / "external/chartqapro"
    chartqa = prepare_chartqa(
        prior_manifest, legacy_image_root, chartqa_root, history_receipt=history
    )
    sources_receipt = acquire_chartqapro(sources)
    if sources_receipt["status"] == "UNAVAILABLE":
        pro = dict(
            status="UNAVAILABLE",
            image_count=0,
            reason=sources_receipt["reason"],
            source_receipt_sha256=object_hash(sources_receipt),
            external_transfer_certified=False,
        )
        durable_json(pro_root / "FREEZE.json", pro)
    else:
        pro = freeze_chartqapro(
            sources / "test-00000-of-00001.parquet",
            pro_root,
            history_receipt=history,
            scorer_source=sources / "evaluate_predictions.py",
            prompt_source=sources / "vlmeval_chartqapro_utils.py",
            anls_wheel=sources / "anls-0.0.2-py3-none-any.whl",
        )
    receipt = dict(
        status="FROZEN",
        experiment_id=EXPERIMENT_ID,
        CHARTQA=dict(
            root="external/chartqa",
            freeze_sha256=sha_file(chartqa_root / "FREEZE.json"),
            status=chartqa["status"],
            question_count=2500,
        ),
        CHARTQAPRO=dict(
            root="external/chartqapro",
            freeze_sha256=sha_file(pro_root / "FREEZE.json"),
            status=pro["status"],
            image_count=pro["image_count"],
            reason=pro.get("reason"),
        ),
    )
    durable_json(root / "external/PANELS.json", receipt)
    return receipt


def _external(root, *, load_scorer=True):
    root = Path(root)
    panels = _read(root / "external/PANELS.json")
    if panels.get("status") != "FROZEN" or panels.get("experiment_id") != EXPERIMENT_ID:
        raise PermissionError("External panels have not been frozen")
    tasks, stores, registries = {}, {}, {}
    official = None
    for pool in ("CHARTQA", "CHARTQAPRO"):
        item = panels[pool]
        directory = _bounded(root, item["root"])
        if sha_file(directory / "FREEZE.json") != item["freeze_sha256"]:
            raise PermissionError("External freeze changed")
        freeze = _read(directory / "FREEZE.json")
        if pool == "CHARTQAPRO" and freeze["status"] == "UNAVAILABLE":
            if not freeze.get("reason") or freeze.get("image_count") != 0:
                raise ValueError("Explicit external-only unavailability receipt required")
            continue
        stores[pool] = ExternalInputStore(directory)
        registries[pool] = _read(directory / "TYPE_REGISTRY.json")
        path = directory / "TASKS_AUDIT_ONLY.jsonl"
        if sha_file(path) != freeze["audit_tasks_sha256"]:
            raise PermissionError("External audit task identity differs")
        for task in _lines(path):
            if task["qid"] in tasks or task["pool"] != pool:
                raise ValueError("Duplicate or mislabeled external task")
            tasks[task["qid"]] = task
        if pool == "CHARTQAPRO" and load_scorer:
            official = load_official_scorer(
                directory / "sources/evaluate_predictions.py",
                directory / "sources/anls-0.0.2-py3-none-any.whl",
            )

    def scorer(row, task):
        return score_external(
            row, task, type_registry=registries[task["pool"]], official_scorer=official
        )

    return tasks, stores, panels, scorer


def _load_endpoint_runtime(root, model, task_id):
    from mm_core.training import state_hash, trainable_state
    from sr_f1.contract import digest
    from sr_f1.runtime import copy_parameters

    from .checkpoint import read_checkpoint
    from .prompts import prefill_for
    from .runner import load_parent_runtime

    runtime = load_parent_runtime(root, model["parent_id"], task_id)
    revision = runtime.protocol_revision
    provenance = dict(
        model=model, policy_source="verified parent adapter", protocol_revision=revision
    )
    if model["arm"] != "PARENT":
        directory = Path(root) / "runs" / model["model_id"]
        state, checkpoint = read_checkpoint(directory / "checkpoints", step=64)
        identity = state["run_identity"]
        if (
            state["committed_logical_step"] != 64
            or identity.get("run_id") != model["model_id"]
            or any(identity.get(k) != model[k] for k in ("arm", "block", "parent_id"))
            or identity.get("stage") != "SCIENCE"
            or identity.get("experiment_id") != EXPERIMENT_ID
            or identity.get("run_identity_sha256") != sha_file(Path(root) / "RUN_IDENTITY.json")
            or identity.get("updates") != 64
            or identity.get("future_seed") != 73101 + model["block"]
            or identity.get("protocol_revision", "v1") != revision
            or identity.get("assistant_prefill", "{") != prefill_for("structured", revision)
            or state["reference_hash"] != state_hash(trainable_state(runtime.model))
        ):
            raise PermissionError("Scientific final checkpoint identity differs")
        exported = directory / "adapters/step-64"
        adapter = _read(exported / "IDENTITY.json")
        if (
            adapter.get("step") != 64
            or adapter.get("policy_hash") != state_hash(state["parameters"])
            or adapter.get("run_identity_sha256") != digest(identity)
            or any(
                sha_file(_bounded(exported, path)) != value
                for path, value in adapter["files"].items()
            )
        ):
            raise PermissionError("Final published adapter/committed policy identity differs")
        copy_parameters(runtime.model, state["parameters"])
        if state_hash(trainable_state(runtime.model)) != adapter["policy_hash"]:
            raise PermissionError("Loaded endpoint tensors differ from committed step 64")
        runtime.adapter_path = str(exported)
        runtime.identity["adapter_hash"] = sha_file(exported / "IDENTITY.json")
        provenance.update(
            policy_source="verified committed step-64 parameters",
            checkpoint=checkpoint,
            adapter_export=adapter,
        )
        del state
    runtime.identity.update(model_id=model["model_id"], step=model["step"])
    provenance.update(
        policy_hash=state_hash(trainable_state(runtime.model)),
        model_identity=runtime.stable_model_identity(),
        adapter_identity=runtime.current_adapter_identity(),
    )
    return runtime, provenance


def execute_endpoint(root, model_id):
    """Generate all fixed outputs, including failures, with no TEST scoring or peeking."""
    from .orchestration import write_task_receipt
    from .runner import dataset_and_store, evaluate_panel

    root = Path(root).resolve(strict=True)
    models = {m["model_id"]: m for m in endpoint_models()}
    if model_id not in models:
        raise PermissionError("Unregistered endpoint")
    _require_terminal(root)
    dataset, store = dataset_and_store(root)
    revision = _run_revision(root)
    external_tasks, external_stores, panels, _ = _external(root, load_scorer=False)
    tasks = dict(dataset.tasks, **external_tasks)
    slots = endpoint_slots(
        dataset.tasks, dataset.runs, external_tasks, model_id=model_id, revision=revision
    )
    runtime, provenance = _load_endpoint_runtime(root, models[model_id], "eval--" + model_id)
    if runtime.protocol_revision != revision:
        raise PermissionError("Endpoint runtime differs from the frozen protocol revision")
    destination = root / "sealed/endpoint" / model_id
    durable_json(destination / "MODEL_IDENTITY.json", provenance)
    all_manifests, counts = [], Counter()
    for pool in (*TEST_POOLS, "CHARTQA", "CHARTQAPRO"):
        chosen = [s for s in slots if s["pool"] == pool]
        if not chosen:
            continue
        input_store = external_stores.get(pool, store)
        rows, _, coverage = evaluate_panel(
            runtime, root, input_store, tasks, chosen, destination / pool, score=False
        )
        counts.update(r["protocol"] for r in rows)
        for entry in coverage["groups"]:
            path = destination / pool / "groups" / (entry["group_id"] + ".json")
            all_manifests.append(
                dict(
                    path=str(path.relative_to(root)),
                    sha256=entry["sha256"],
                    bytes=path.stat().st_size,
                    group_id=entry["group_id"],
                    pool=pool,
                )
            )
        del rows
    manifest = dict(
        status="SEALED_COMPLETE",
        experiment_id=EXPERIMENT_ID,
        model_id=model_id,
        model_identity_sha256=sha_file(destination / "MODEL_IDENTITY.json"),
        slots_sha256=object_hash(slots),
        records=len(slots),
        raw_groups=all_manifests,
        external_panels_sha256=sha_file(root / "external/PANELS.json"),
        score_access=False,
    )
    durable_json(destination / "RAW_MANIFEST.json", manifest)
    pro = panels["CHARTQAPRO"]
    evidence = dict(
        structured_rows=counts["structured"],
        alternate_protocol_rows=counts["answer_only"] + counts["natural_work"],
        chartqa_rows=counts["chartqa_plain"],
        chartqapro_rows=counts["chartqapro_official"],
        chartqapro_status="UNAVAILABLE" if pro["status"] == "UNAVAILABLE" else "COMPLETE",
        sealed=True,
    )
    return write_task_receipt(
        root,
        "eval--" + model_id,
        artifacts=[
            str((destination / name).relative_to(root))
            for name in ("MODEL_IDENTITY.json", "RAW_MANIFEST.json")
        ],
        evidence=evidence,
    )


def _raw_groups(root, slots):
    """Re-open all expected raw files and validate identities/trajectories, streamwise."""
    from .runner import dataset_and_store

    _, synthetic_store = dataset_and_store(root)
    _, external_stores, _, _ = _external(root, load_scorer=False)
    expected = {g["group_id"]: g for g in group_slots(slots)}
    found = set()
    for model in endpoint_models():
        destination = Path(root) / "sealed/endpoint" / model["model_id"]
        manifest = _read(destination / "RAW_MANIFEST.json")
        provenance = _read(destination / "MODEL_IDENTITY.json")
        if (
            manifest.get("status") != "SEALED_COMPLETE"
            or manifest.get("model_id") != model["model_id"]
            or provenance.get("model") != model
            or manifest["model_identity_sha256"] != sha_file(destination / "MODEL_IDENTITY.json")
            or manifest["external_panels_sha256"] != sha_file(Path(root) / "external/PANELS.json")
        ):
            raise PermissionError("Sealed endpoint manifest identity differs")
        for entry in manifest["raw_groups"]:
            gid = entry["group_id"]
            if gid in found or gid not in expected:
                raise ValueError("Duplicate/unregistered endpoint raw group")
            path = _bounded(root, entry["path"])
            if sha_file(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
                raise PermissionError("Sealed raw group bytes changed")
            value = _read(path)
            rows, group = value["records"], expected[gid]
            if group["slots"][0]["model_id"] != model["model_id"]:
                raise PermissionError("Raw group was attributed to another model")
            if value["group_id"] != gid or value["records_sha256"] != object_hash(rows):
                raise PermissionError("Sealed raw group internal checksum differs")
            if len(rows) != len(group["slots"]):
                raise RuntimeError("Raw group technical coverage incomplete")
            for index, (row, slot) in enumerate(zip(rows, group["slots"], strict=True)):
                store = external_stores.get(slot["pool"], synthetic_store)
                request = store.model_input(slot["qid"], protocol=slot["protocol"])
                if row.get("model_input") != request:
                    raise PermissionError("Raw endpoint input differs from the frozen input store")
                validate_record(
                    row,
                    slot,
                    input_hash=object_hash(request),
                    model_identity=provenance["model_identity"],
                    adapter_identity=provenance["adapter_identity"],
                    source_sha256=sha_file(Path(root) / "RUN_IDENTITY.json"),
                    index=index,
                    count=len(rows),
                )
            found.add(gid)
            yield group, rows, entry
    if found != set(expected):
        raise RuntimeError("Complete frozen raw endpoint groups required")


def _publish_file(temporary, destination):
    if destination.exists():
        if sha_file(temporary) != sha_file(destination):
            raise PermissionError("Immutable release file differs")
        temporary.unlink()
    else:
        os.link(temporary, destination)
        temporary.unlink()


def _external_summary(rows):
    """All original/v2/official outcomes, and explicit unresolved typed counts."""
    from .analysis import contrast
    from .evaluation import ARMS

    output = {}
    for pool in ("CHARTQA", "CHARTQAPRO"):
        output[pool] = {}
        for model in endpoint_models():
            selected = [r for r in rows if r["pool"] == pool and r["model_id"] == model["model_id"]]
            if not selected:
                continue
            clusters = defaultdict(list)
            for row in selected:
                clusters[row["image_cluster"]].append(row)
            summary = dict(
                records=len(selected),
                image_clusters=len(clusters),
                typed_status_counts=dict(
                    Counter(r["score"]["typed_audit"]["status"] for r in selected)
                ),
            )
            typed = [
                r["score"]["typed_audit"]["typed_score"]
                for r in selected
                if r["score"]["typed_audit"]["typed_score"] is not None
            ]
            summary["typed_audit"] = dict(
                total_questions=len(selected),
                defined_questions=len(typed),
                mean_on_defined_only=sum(typed) / len(typed) if typed else None,
                full_benchmark_typed_score=None
                if len(typed) != len(selected)
                else sum(typed) / len(typed),
                unresolved_preserved=True,
                replaces_official_score=False,
            )
            if pool == "CHARTQA":
                for version in ("historical_code", "v2"):
                    summary[version] = {
                        metric: sum(r["score"][version][metric] for r in selected) / len(selected)
                        for metric in ("relaxed_accuracy", "exact_match")
                    }
            else:
                summary["official"] = sum(r["score"]["official"] for r in selected) / len(selected)
            output[pool][model["model_id"]] = summary
        metric_paths = (
            (
                ("historical_code", "relaxed_accuracy"),
                ("historical_code", "exact_match"),
                ("v2", "relaxed_accuracy"),
                ("v2", "exact_match"),
            )
            if pool == "CHARTQA"
            else (("official",),)
        )
        paired = {}
        pool_rows = [r for r in rows if r["pool"] == pool]
        if not pool_rows:
            continue
        for metric_path in metric_paths:
            cells = defaultdict(lambda: defaultdict(dict))
            for row in pool_rows:
                value = row["score"]
                for key in metric_path:
                    value = value[key]
                cluster = row["image_cluster"]
                if not cluster or row["qid"] in cells[row["model_id"]][cluster]:
                    raise ValueError("Unique image-cluster/question external support required")
                cells[row["model_id"]][cluster][row["qid"]] = value
            models = {(m["arm"], m["block"]): m["model_id"] for m in endpoint_models()}
            common = set(cells[models[("PARENT", 0)]])
            if any(set(cells[mid]) != common for mid in models.values()):
                raise RuntimeError("External paired image-cluster support differs between models")
            for cluster in common:
                questions = set(cells[models[("PARENT", 0)]][cluster])
                if any(set(cells[mid][cluster]) != questions for mid in models.values()):
                    raise RuntimeError(
                        "External question missingness cannot change paired denominators"
                    )
            for arm1, arm0 in [
                *((arm, "PARENT") for arm in ARMS),
                ("EXEC", "PREFIX"),
                ("EXEC", "GOLD"),
            ]:
                matrix = [
                    [
                        sum(cells[models[(arm1, b)]][cluster].values())
                        / len(cells[models[(arm1, b)]][cluster])
                        - sum(cells[models[(arm0, b)]][cluster].values())
                        / len(cells[models[(arm0, b)]][cluster])
                        for b in range(3)
                    ]
                    for cluster in sorted(common)
                ]
                if len(matrix) > 1:
                    item = contrast(matrix, ["EXTERNAL"] * len(matrix), alpha=0.05)
                else:
                    item = dict(
                        point=sum(matrix[0]) / 3,
                        per_block=matrix[0],
                        roots=1,
                        CI95=None,
                        CI_simultaneous=None,
                        interval_status="UNAVAILABLE_ONLY_ONE_IMAGE_CLUSTER",
                    )
                item.update(
                    arm1=arm1,
                    arm0=arm0,
                    metric="/".join(metric_path),
                    point_weighting=(
                        "equal image clusters, questions within image, three blocks equal"
                    ),
                    role="secondary descriptive comparison; nominal 95% interval",
                )
                paired["/".join(metric_path) + ":" + arm1 + "-" + arm0] = item
        output[pool]["paired_image_cluster_contrasts"] = paired
    return output


def _prefix_update(accumulator, row, task):
    """Describe real decoded token prefixes; never pretend these are new rollouts."""
    if row["pool"] not in TEST_POOLS:
        return
    from .contract import registered_completion
    from .prompts import prefill_for

    prefixes = row.get("decoded_prefix_views")
    if not isinstance(prefixes, dict) or set(prefixes) != {"128", "256", "512", "1024"}:
        raise RuntimeError("Actual decoded token prefix views are required for budget diagnostics")
    revision = row.get("protocol_revision", "v1")
    prefill = prefill_for(row["protocol"], revision)
    if row.get("assistant_prefill", prefill) != prefill or any(
        not isinstance(text, str) or not text.startswith(prefill) for text in prefixes.values()
    ):
        raise PermissionError("Decoded budget prefixes differ from the recorded context prefill")
    key = row["model_id"] + ":" + row["protocol"]
    panel = accumulator.setdefault(
        key,
        dict(
            records=0,
            generated_tokens=0,
            protocol_revision=revision,
            assistant_prefill=prefill,
            prefill_counted_in_generated_budget=False,
            budgets={
                str(n): dict(value_correct=0, finish=0, presentation=0)
                for n in (128, 256, 512, 1024)
            },
        ),
    )
    if panel["protocol_revision"] != revision or panel["assistant_prefill"] != prefill:
        raise PermissionError("Budget diagnostic panel mixes protocol revisions or prefills")
    panel["records"] += 1
    panel["generated_tokens"] += len(row["tokens"])
    for budget, prefix in prefixes.items():
        reason = (
            ("final_tag" if row["protocol"] == "natural_work" else "balanced")
            if registered_completion(prefix, row["protocol"])
            else "length"
        )
        view = dict(row, raw_text=prefix, finish_reason=reason, format_protocol_error=None)
        scores = score_record(view, task)
        for metric in ("value_correct", "finish", "presentation"):
            panel["budgets"][budget][metric] += scores[metric]


def execute_release(root):
    """Release only after 12+15 terminal jobs; archive then rescore every actual raw row."""
    from .analysis import primary_analysis
    from .orchestration import write_task_receipt
    from .runner import dataset_and_store

    root = Path(root).resolve(strict=True)
    _require_terminal(root, evaluations=True)
    dataset, _ = dataset_and_store(root)
    revision = _run_revision(root)
    external_tasks, _, panels, external_scorer = _external(root)
    tasks = dict(dataset.tasks, **external_tasks)
    slots = endpoint_slots(dataset.tasks, dataset.runs, external_tasks, revision=revision)
    training = []
    for model in endpoint_models()[3:]:
        provenance = _read(root / "sealed/endpoint" / model["model_id"] / "MODEL_IDENTITY.json")
        training.append(
            dict(
                run_id=model["model_id"],
                status="COMPLETED",
                step=64,
                adapter_identity=provenance["adapter_identity"],
            )
        )
    pro = panels["CHARTQAPRO"]
    external_status = {
        "CHARTQAPRO": dict(
            status="UNAVAILABLE" if pro["status"] == "UNAVAILABLE" else "COMPLETE",
            image_count=pro["image_count"],
            reason=pro.get("reason"),
        )
    }
    destination = root / "release"
    with shard_lease(destination):
        raw_iterator = (row for _, rows, _ in _raw_groups(root, slots) for row in rows)
        receipt = release_evaluation(
            dataset.runs, training, slots, raw_iterator, external_status=external_status
        )
        durable_json(destination / "RELEASE.json", receipt)
        fd, temporary_name = tempfile.mkstemp(
            prefix="ENDPOINT_RAW.", suffix=".pending.gz", dir=destination
        )
        temporary = Path(temporary_name)
        compact, group_scores, prefix = [], {}, {}
        with os.fdopen(fd, "wb") as binary:
            with gzip.GzipFile(fileobj=binary, mode="wb", filename="", mtime=0) as compressed:
                for group, rows, entry in _raw_groups(root, slots):
                    scored = []
                    for row in rows:
                        compressed.write(
                            (
                                json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
                                + "\n"
                            ).encode()
                        )
                        score = score_record(
                            row, tasks[row["qid"]], external_scorer=external_scorer
                        )
                        _prefix_update(prefix, row, tasks[row["qid"]])
                        scored.append(dict(slot_id=row["slot_id"], score=score))
                        compact.append(
                            dict(
                                group["slots"][row["draw"]],
                                group_id=group["group_id"],
                                generation_status=row["generation_status"],
                                technical_validation_errors=row.get(
                                    "technical_validation_errors", []
                                ),
                                score=score,
                            )
                        )
                    group_scores[group["group_id"]] = object_hash(scored)
                    durable_json(
                        destination / "scores" / (group["group_id"] + ".json"),
                        dict(
                            raw_sha256=entry["sha256"], scores=scored, scorer_version=EXPERIMENT_ID
                        ),
                    )
            binary.flush()
            os.fsync(binary.fileno())
        raw_path = destination / "ENDPOINT_RAW.jsonl.gz"
        _publish_file(temporary, raw_path)
        # A second pass reads the delivered compressed raw artifact itself, not
        # previously derived scores or algebraic identities between score arrays.
        count, mismatches = 0, []
        scores_by_slot = {r["slot_id"]: r["score"] for r in compact}
        with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                again = score_record(row, tasks[row["qid"]], external_scorer=external_scorer)
                if again != scores_by_slot[row["slot_id"]]:
                    mismatches.append(row["slot_id"])
                count += 1
        if mismatches or count != len(slots):
            raise RuntimeError("Delivered raw full rescoring failed; preserve all evidence")
        rescoring = dict(
            status="PASS",
            records=count,
            mismatches=mismatches,
            raw_sha256=sha_file(raw_path),
            gzip_read_to_eof=True,
            method="full delivered raw reparse and production rescoring",
            independently_implemented_scorer_claimed=False,
            group_scores_sha256=object_hash(group_scores),
        )
        durable_json(destination / "RAW_RESCORING.json", rescoring)
        scored_receipt = dict(
            receipt,
            raw_rescoring="PASS",
            projection_kind="verified_raw_rescore_projection_v1",
            analysis_rows_sha256=object_hash(compact),
            rescoring_receipt_sha256=sha_file(destination / "RAW_RESCORING.json"),
        )
        durable_json(destination / "RELEASE_SCORED.json", scored_receipt)
        analysis = primary_analysis(compact, dataset.tasks, scored_receipt)
        durable_json(destination / "PRIMARY_ANALYSIS.json", analysis)
        durable_json(destination / "EXTERNAL_ANALYSIS.json", _external_summary(compact))
        for panel in prefix.values():
            panel["mean_generated_tokens"] = panel["generated_tokens"] / panel["records"]
            for metrics in panel["budgets"].values():
                for key in metrics:
                    metrics[key] /= panel["records"]
        durable_json(
            destination / "PREFIX_DIAGNOSTICS.json",
            dict(
                status="OBSERVED_TRAJECTORY_PREFIXES",
                panels=prefix,
                independent_rollouts=False,
                counterfactual_short_budget_claim=False,
                note=(
                    "Actual per-token decoded prefixes; "
                    "full-trajectory completion/cost description only"
                ),
            ),
        )
        score_bytes = "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            for row in compact
        ).encode()
        fd, temporary_name = tempfile.mkstemp(prefix="ENDPOINT_SCORES.", dir=destination)
        with os.fdopen(fd, "wb") as stream:
            stream.write(score_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        _publish_file(Path(temporary_name), destination / "ENDPOINT_SCORES.jsonl")
        artifacts = [
            "RELEASE.json",
            "RELEASE_SCORED.json",
            "RAW_RESCORING.json",
            "ENDPOINT_RAW.jsonl.gz",
            "ENDPOINT_SCORES.jsonl",
            "PRIMARY_ANALYSIS.json",
            "EXTERNAL_ANALYSIS.json",
            "PREFIX_DIAGNOSTICS.json",
        ]
        manifest = dict(
            status="RAW_AND_NUMERICAL_RELEASE_COMPLETE",
            files=[
                dict(
                    path=name,
                    sha256=sha_file(destination / name),
                    bytes=(destination / name).stat().st_size,
                )
                for name in artifacts
            ],
            scope="all 15 endpoints; raw text, tokens, prompts and scores retained",
            final_research_delivery_complete=False,
        )
        durable_json(destination / "MANIFEST.json", manifest)
    return write_task_receipt(
        root,
        "release",
        artifacts=["release/" + name for name in [*artifacts, "MANIFEST.json"]],
        evidence=dict(training_paths=12, evaluated_models=15, raw_rescoring="PASS"),
    )
