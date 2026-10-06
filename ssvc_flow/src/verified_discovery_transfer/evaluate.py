"""Outcome summaries and immutable release gate for E/G artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from .queue import digest, read_json, write_json
from .statistics import pass_at_k, wilson


def release_test(run, queue):
    run = Path(run)
    from .cli import _context

    _context(run)
    frozen = read_json(run / "freeze.json")
    if (
        digest(read_json(run / "protocol.json")) != frozen["protocol_hash"]
        or digest(read_json(run / "machine.json")) != frozen["machine_hash"]
    ):
        raise ValueError("Frozen protocol/machine identity changed")
    if not queue.all_terminal():
        raise PermissionError("E/G remain sealed until every registered task is terminal")
    rows = queue.rows()
    registry = read_json(run / "REGISTERED_MATRIX.json")
    if registry["jobs"] != [row["payload"] for row in rows]:
        raise ValueError("Registered matrix differs from queue jobs")
    identity = queue.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
    if not identity or identity[0] != registry["identity"]:
        raise ValueError("Registered queue identity differs")
    from .reports import validate_completed_artifacts

    validate_completed_artifacts(run, queue)
    if queue.seal_release() != digest(rows):
        raise ValueError("Matrix changed during release")
    receipt = {
        "schema": "verified-discovery-release-v1",
        "status": "RELEASED",
        "queue_identity": frozen["run_identity"],
        "matrix_digest": digest(rows),
        "technical_missing": [r["id"] for r in rows if r["status"] == "BLOCKED_TECHNICAL"],
        "scientific_completion": all(r["status"] != "BLOCKED_TECHNICAL" for r in rows),
    }
    write_json(run / "TEST_RELEASE.json", receipt)
    return receipt


def summarize_raw(records, *, expected_draws):
    streams = defaultdict(list)
    ids = set()
    for row in records:
        if row["request_id"] in ids:
            raise ValueError("Duplicate atomic sample")
        ids.add(row["request_id"])
        streams[(row["family"], row["task_id"])].append(row)
    if not streams or any(len(rows) != expected_draws for rows in streams.values()):
        raise ValueError("Incomplete fixed evaluation panel")
    tasks = []
    families = defaultdict(list)
    for (family, task_id), rows in sorted(streams.items()):
        count = sum(bool(row["public_verifier_pass"]) for row in rows)
        successes = [row["public_verifier_pass"] for row in rows]
        record = {
            "family": family,
            "task_id": task_id,
            "draws": expected_draws,
            "successes": count,
            "pX": count / expected_draws,
            "wilson_95": wilson(count, expected_draws),
            "unseen_success": count == 0,
            "contrast_state": "all-X"
            if all(successes)
            else "mixed-X"
            if any(successes)
            else "no-X",
            "pass_at_k": {
                str(k): pass_at_k(count, expected_draws, k)
                for k in (1, 2, 4, 8)
                if k <= expected_draws
            },
        }
        tasks.append(record)
        families[family].append(record["pX"])
    return {
        "pX_family_equal": mean(mean(values) for values in families.values()),
        "families": {family: mean(values) for family, values in families.items()},
        "tasks": tasks,
        "contrast_state_counts": dict(Counter(task["contrast_state"] for task in tasks)),
    }


def generate_frozen(backend, prompt, seed, *, data_root=None, do_sample=True):
    """Inherited uncached raw-only path, with an explicit public image boundary."""
    if set(prompt) == {"system", "user"} and do_sample:
        return backend.generate_public(prompt, seed, 64)
    import copy

    import torch

    from ..modeling_v3.vlm_observation import ObservationFault, validate_action
    from ..protocol_state_probes.inference import _memory_peaks, _model_guard

    if set(prompt) not in ({"system", "user"}, {"system", "user", "image_path"}) or any(
        not isinstance(v, str) for v in prompt.values()
    ):
        raise ValueError("Only public strings and registered image path are accepted")
    if _model_guard(backend.adapter.model) != backend._guard_baseline:
        raise RuntimeError("Frozen model changed")
    if prompt.get("image_path"):
        if data_root is None:
            raise ValueError("Image requires registered cohort root")
        image = (Path(data_root) / prompt["image_path"]).resolve()
        if not image.is_relative_to(Path(data_root).resolve()):
            raise ValueError("Image escaped cohort root")
    prepared = backend.adapter.prepare(copy.deepcopy(prompt), data_root or backend.data_root)
    device = torch.device(backend.adapter.device)
    devices = [device.index or 0] if device.type == "cuda" else []
    raw = None
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            raw = backend.adapter.generate(
                prepared, seed=seed, max_new_tokens=64, do_sample=do_sample
            )
        flags = validate_action(
            raw,
            eos_ids=backend.adapter.eos_ids,
            max_new_tokens=64,
            tokenizer=backend.adapter.processor.tokenizer,
        )
    except Exception as exc:
        raise ObservationFault(str(exc), raw) from exc
    finally:
        if _model_guard(backend.adapter.model) != backend._guard_baseline:
            raise RuntimeError("Frozen image/greedy generation mutated the model")
    return {
        "raw_text": raw["raw_completion"],
        "token_ids": raw["token_ids"],
        "stop_reason": raw["stop_reason"],
        "generated_length": raw["completion_length"],
        "elapsed_seconds": raw["elapsed_seconds"],
        "chosen_token_logprobs": raw["behavior_token_logprobs"],
        "input_audit": copy.deepcopy(prepared["audit"]),
        "public_prompt_hash": digest(prompt),
        "inference_fingerprint": backend.receipt["inference_fingerprint"],
        "seed": seed,
        "extra_rescoring": False,
        "do_sample": do_sample,
        **_memory_peaks(backend.adapter.device),
        **flags,
    }


def audit_endpoint(records, tasks, audits, pre_zero16_ids=()):
    """Derive final-only semantic diagnostics; never return targets to generation.

    Callers obtain audits using RoleDataset(..., 'final_eval') after release.
    This pure analysis function accepts E/G only and keeps each protocol separate.
    PRE_ZERO16 IDs must come from the frozen parent O0 16-draw panel, never from
    these post-training records. Partial repair is explicitly F=1 and B>0;
    partial relation satisfaction is a different, separately named diagnostic.
    """
    from ..protocol_state_probes.ptlc import anchors, dpe
    from ..protocol_state_probes.semantics import score_raw
    from ..protocol_state_probes.transforms import FWD, predict_equations
    from .public_tasks import compile_case, public_verifier, validate_public

    by_task = {task["task_id"]: task for task in tasks}
    by_audit = {audit["task_id"]: audit for audit in audits}
    if len(by_task) != len(tasks) or len(by_audit) != len(audits) or set(by_task) != set(by_audit):
        raise ValueError("Audit requires one exact matching privileged row per public task")
    if not tasks or len({t["split"] for t in tasks}) != 1:
        raise ValueError("One nonempty final split required")
    if any(t["split"] not in ("E_test", "G_guard") for t in tasks):
        raise PermissionError("Semantic endpoint audits are final E/G analysis only")
    for task in tasks:
        validate_public(task)
        audit = by_audit[task["task_id"]]
        if audit["split"] != task["split"] or audit["base_instance_id"] != task["base_instance_id"]:
            raise ValueError("Audit/public identity mismatch")
        truth, j = audit["true_world"], audit["corrupted_index"]
        if not public_verifier(task, truth) or [
            k for k in range(4) if truth[k] != task["observed"][k]
        ] != [j]:
            raise ValueError("Audit truth violates the registered one-edit public contract")
        if audit.get("DPE1", {}).get("O0") != dpe(task["H_original"], j, FWD):
            raise ValueError("Audit DPE1_O0 disagrees with original public relations")
        if audit["truth_all_in_0_49"] != all(0 <= value <= 49 for value in truth):
            raise ValueError("Audit numeric-range stratum disagrees with truth")
    pre_zero16 = set(pre_zero16_ids)
    streams, output, requests, identities = defaultdict(list), [], set(), set()
    cases = {}
    for record in records:
        tid = record["task_id"]
        if tid not in by_task or record["request_id"] in requests:
            raise ValueError("Unknown audit task or duplicate accepted request")
        requests.add(record["request_id"])
        task, audit = by_task[tid], by_audit[tid]
        if (
            record["split"] != task["split"]
            or record["base_instance_id"] != task["base_instance_id"]
        ):
            raise ValueError("Generation split/base identity does not match audit task")
        # Different steps, student jobs, parents, or repeats are distinct endpoints.
        identity = tuple(
            record.get(k)
            for k in ("parent_id", "model_revision", "pipeline_repeat", "source_job", "step")
        )
        identities.add(identity)
        if len(identities) != 1:
            raise ValueError("Do not pool different models/checkpoints/repeats into one endpoint")
        protocol = record["protocol_id"]
        key = (tid, protocol)
        if key not in cases:
            case = compile_case(task, protocol)
            case.update(panel=task["split"], kind="repair")
            prediction = {
                "case_id": case["case_id"],
                "programs": predict_equations(
                    case["H_display"], case["b_display"], task["observed"], case["output_order"]
                ),
            }
            cases[key] = (case, prediction)
        case, prediction = cases[key]
        inherited_audit = {
            "case_id": case["case_id"],
            "truth_world": audit["true_world"],
            "corrupted_coordinate": audit["corrupted_index"],
            "control_expected_canonical": None,
        }
        semantic = score_raw(record["raw_completion"], case, inherited_audit, prediction)
        if "public_verifier_pass" in record and record["public_verifier_pass"] != semantic["X"]:
            raise ValueError("Public verification and privileged exact-world event disagree")
        canonical = semantic["canonical_world"]
        original_anchors = anchors(task["H_original"], FWD)
        original_mask = (
            [canonical[k] == task["observed"][k] if k in original_anchors else None for k in FWD]
            if canonical is not None
            else None
        )
        orig_kept = (
            all(canonical[k] == task["observed"][k] for k in original_anchors)
            if canonical is not None
            else None
        )
        algebraic = {}
        if canonical is not None:
            j, truth = audit["corrupted_index"], audit["true_world"]
            algebraic = {
                "F_alg": int(canonical[j] == truth[j]),
                "B_alg": sum(canonical[k] != truth[k] for k in FWD if k != j),
                "M_alg": sum(canonical[k] != task["observed"][k] for k in FWD),
            }
        aliases = {
            "truth": bool(semantic["X"]),
            "copy": bool(semantic["copy"]),
            "V1": bool(semantic["match_V1"]),
            "V2": bool(semantic["match_V2"]),
        }
        row = {
            key: record.get(key)
            for key in (
                "request_id",
                "task_id",
                "base_instance_id",
                "split",
                "parent_id",
                "model_revision",
                "pipeline_repeat",
                "source_job",
                "step",
                "protocol_id",
                "draw_index",
                "stop_reason",
            )
        }
        row.update(semantic)
        row.update(algebraic)
        row.update(
            family=task["family"],
            interface=task["interface"],
            operation=task["operation"],
            corrupted_index=audit["corrupted_index"],
            DPE1_O0=audit["DPE1"]["O0"],
            DPE1_protocol=dpe(case["H_display"], audit["corrupted_index"], case["output_order"]),
            truth_all_in_0_49=audit["truth_all_in_0_49"],
            PRE_ZERO16=tid in pre_zero16 or task["base_instance_id"] in pre_zero16,
            original_anchor_coordinates=original_anchors,
            original_anchor_mask=original_mask,
            original_anchor_all=orig_kept,
            original_anchor_all_legal=bool(semantic["domain_valid"] and orig_kept),
            alias_flags=aliases,
            alias_mask=sum(
                int(aliases[name]) << index
                for index, name in enumerate(("truth", "copy", "V1", "V2"))
            ),
            noncopy_nontruth_ptlc_match=bool(
                (aliases["V1"] or aliases["V2"]) and not aliases["copy"] and not aliases["truth"]
            ),
            noncopy_nontruth_ptlc_match_legal=bool(
                semantic["domain_valid"]
                and (aliases["V1"] or aliases["V2"])
                and not aliases["copy"]
                and not aliases["truth"]
            ),
            all_original_relations_wrong=bool(
                semantic["domain_valid"]
                and all(semantic["relation_flags_orig"])
                and not semantic["X"]
            ),
            all_original_relations_wrong_alg=bool(
                canonical is not None and all(semantic["relation_flags_orig"]) and not semantic["X"]
            ),
            partial_repair=semantic["F"] == 1 and semantic["B"] > 0,
            partial_original_relations=bool(
                semantic["domain_valid"] and 0 < semantic["C_orig"] < 1
            ),
        )
        output.append(row)
        streams[key].append(row)
    if {key[0] for key in streams} != set(by_task):
        raise ValueError("Incomplete endpoint task coverage")
    metrics = []
    for (_tid, protocol), rows in sorted(streams.items()):
        draws = [row["draw_index"] for row in rows]
        if any(type(draw) is not int for draw in draws) or sorted(draws) != list(range(len(rows))):
            raise ValueError("Each protocol requires unique contiguous draws from zero")
        rows.sort(key=lambda row: row["draw_index"])
        n = len(rows)
        counts = Counter(row["event"] for row in rows)
        x = counts["X"]
        reference = rows[0]
        values = {row["C_orig"] for row in rows}
        algebraic_values = {row["C_alg_orig"] for row in rows if row["C_alg_orig"] is not None}
        contrast = {
            "state": "all-X" if x == n else "mixed-X" if x else "no-X",
            "has_partial_repair": any(row["partial_repair"] for row in rows),
            "has_partial_original_relations": any(
                row["partial_original_relations"] for row in rows
            ),
            "C_orig_varies": len(values) > 1,
            "C_alg_orig_varies_among_parsed": len(algebraic_values) > 1,
            "C_orig_values": sorted(values),
            "C_alg_orig_values": sorted(algebraic_values),
        }
        metric = {
            key: reference[key]
            for key in (
                "task_id",
                "base_instance_id",
                "split",
                "family",
                "interface",
                "operation",
                "protocol_id",
                "corrupted_index",
                "DPE1_O0",
                "DPE1_protocol",
                "truth_all_in_0_49",
                "PRE_ZERO16",
            )
        }
        metric.update(
            n=n,
            **{"n" + event: counts[event] for event in ("X", "S", "W", "I")},
            **{"p" + event: counts[event] / n for event in ("X", "S", "W", "I")},
            pass_at_k={str(k): pass_at_k(x, n, k) for k in (1, 2, 4, 8) if k <= n},
            wilson_95=wilson(x, n),
            contrast=contrast,
            K8_contrast=contrast
            if reference["split"] == "E_test" and protocol == "O0" and n == 8
            else None,
            original_anchor_preservation=sum(row["original_anchor_all_legal"] for row in rows) / n,
            PTLC_alias_counts=dict(Counter(str(row["alias_mask"]) for row in rows)),
            partial_repair_count=sum(row["partial_repair"] for row in rows),
            copy_count=sum(row["copy"] for row in rows),
            all_original_relations_wrong_count=sum(
                row["all_original_relations_wrong"] for row in rows
            ),
            noncopy_nontruth_ptlc_count=sum(
                row["noncopy_nontruth_ptlc_match_legal"] for row in rows
            ),
        )
        for field in ("C_orig", "C_display", "F", "B", "M"):
            available = [row[field] for row in rows if row[field] is not None]
            metric[field + "_mean"] = mean(available) if available else None
            metric[field + "_count"] = len(available)
        metrics.append(metric)
    return {
        "audit_only": True,
        "no_generation_feedback": True,
        "pre_zero16_definition": "parent O0 pretraining fixed 16-draw no-X; not pX=0",
        "partial_repair_definition": (
            "legal F=1 and B>0; repaired corrupted value but damaged another"
        ),
        "K8_scope": "post-training E O0 eight-draw fixed evaluation groups only",
        "records": output,
        "tasks": metrics,
    }
