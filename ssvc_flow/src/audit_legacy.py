"""P0: read legacy evidence as data, without importing or executing legacy code.

Only allowlisted research directories are searched. Archives are streamed, never
extracted; checkpoint blobs are inventoried without loading model weights.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import yaml

from .core import canonical_hash, file_hash, phase_artifacts, write_json

SEARCH_ROOTS = (
    "configs/v5/study_c3_resolution_validity.yaml",
    "configs/v5/server_package_lock.yaml",
    "src/compensability/study_c3",
    "src/compensability_v5/study_c2",
    "src/compensability_v5/qwen/study_b_backend.py",
    "scripts/v5/study_c3",
    "tests/study_c3",
    "artifacts/v5/study_c3",
    "artifacts/v5/study_c2/data",
    "artifacts/v5/study_c2/training",
    "artifacts/v5/study_c2/evaluation",
)
TEXT_SUFFIXES = frozenset((".py", ".json", ".jsonl", ".yaml", ".yml", ".md", ".csv"))
BINARY_SUFFIXES = frozenset((".safetensors", ".bin", ".pt", ".pth"))
MAX_PARSE_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
REQUIRED = (
    "model",
    "training_seeds",
    "learning_rate",
    "group_size",
    "reward_vectors",
    "parser_source",
    "prompt_data",
    "package_versions",
    "training_logs",
    "checkpoint_binaries",
    "initial_adapter_config",
    "group_std_convention",
    "loss_reduction",
    "prompts_per_optimizer_update",
    "effective_chat_template",
    "parser_runtime_qualification",
    "source_commit_checkout_verified",
)
NEW_PROTOCOL = {
    "model": "Qwen/Qwen3.5-9B",
    "training_seeds": [17],
    "learning_rate": 1e-5,
    "group_size": 8,
    "reward_vectors": {
        "A_BASE": [2, 2, 0, 0],
        "X_BASE": [2, 0, 0, 0],
        "A_VALID": [3, 3, 1, 0],
        "X_VALID": [3, 1, 1, 0],
    },
    "group_std_convention": "sqrt(population_variance + 1e-8)",
    "loss_reduction": "fixed token constant Lnorm=64, denominator B*K*64",
    "prompts_per_optimizer_update": 4,
    "parser_source": "strict four-key JSON, exact keys a,b,c,d, integer values",
}


def _entry(reference, content, kind, size=None):
    return {
        "source": reference,
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content) if size is None else size,
        "kind": kind,
    }


def _category(name):
    path = PurePosixPath(name)
    if path.suffix in BINARY_SUFFIXES:
        return "checkpoint_binary"
    if "trainer_log" in name or "reward_trace" in name:
        return "training_log"
    if path.suffix == ".py":
        return "source"
    return "data" if path.suffix == ".jsonl" else "config_or_report"


def _readable(path, root):
    return not any(p.is_symlink() for p in (path, *path.parents) if p != root.parent)


def _paths(root):
    found = set()
    for relative in SEARCH_ROOTS:
        candidate = root / relative
        if not _readable(candidate, root):
            continue
        if candidate.is_file():
            found.add(candidate)
        elif candidate.is_dir():
            for directory, subdirs, files in os.walk(candidate, followlinks=False):
                subdirs[:] = sorted(d for d in subdirs if not (Path(directory) / d).is_symlink())
                found.update(
                    Path(directory) / f for f in files if not (Path(directory) / f).is_symlink()
                )
    return tuple(sorted(p for p in found if p.suffix in TEXT_SUFFIXES | BINARY_SUFFIXES))


def _needs_content(name, size):
    if size > MAX_PARSE_BYTES:
        return False
    path = PurePosixPath(name)
    return path.suffix in {".py", ".yaml", ".yml"} or path.name in {
        "reward_fibers.jsonl",
        "arm_config.json",
        "factorial_execution_contract.json",
        "manifest.json",
        "sha256_manifest.json",
        "adapter_config.json",
    }


def _directory_evidence(root):
    inventory, contents = [], {}
    for path in _paths(root):
        name, size = path.relative_to(root).as_posix(), path.stat().st_size
        item = {"source": name, "sha256": None, "bytes": size, "kind": _category(name)}
        if path.suffix in BINARY_SUFFIXES and size > MAX_PARSE_BYTES:
            item = {**item, "hash_status": "not_hashed_large_checkpoint"}
        else:
            item = {**item, "sha256": file_hash(path)}
        inventory.append(item)
        if _needs_content(name, size):
            contents[name] = path.read_bytes()
    return inventory, contents


def _archive_evidence(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("legacy archive is missing or unsafe")
    inventory, contents, seen, total = [], {}, set(), 0
    with tarfile.open(path, "r|*") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or member.name in seen:
                raise ValueError("unsafe or duplicated archive member")
            seen.add(member.name)
            if member.isdir():
                continue
            if not member.isfile() or member.size < 0:
                raise ValueError("unsafe archive member type")
            total += member.size
            if total > MAX_ARCHIVE_BYTES or len(seen) > 10000:
                raise ValueError("legacy archive exceeds bounded audit budget")
            stream = archive.extractfile(member)
            digest, chunks = hashlib.sha256(), []
            save = _needs_content(member.name, member.size)
            for chunk in iter(lambda stream=stream: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                if save:
                    chunks.append(chunk)
            reference = f"archive:{path.name}!/{member.name}"
            inventory.append(
                {
                    "source": reference,
                    "sha256": digest.hexdigest(),
                    "bytes": member.size,
                    "kind": _category(member.name),
                }
            )
            if save:
                contents[reference] = b"".join(chunks)
    return (
        inventory,
        contents,
        {
            "name": path.name,
            "sha256": file_hash(path),
            "member_count": len(inventory),
            "uncompressed_bytes": total,
        },
    )


def _fact(value, item, field):
    return {
        "value": value,
        "evidence": [{"source": item["source"], "sha256": item["sha256"], "field": field}],
    }


def _config_facts(payload, item):
    fields = {key: key for key in ("model", "training_seeds")}
    fields |= {
        key: f"training.{key}"
        for key in (
            "learning_rate",
            "group_size",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "optimizer",
            "precision",
            "kl_beta",
            "temperature",
            "top_p",
            "max_completion_length",
            "optimizer_steps",
            "checkpoint_steps",
        )
    }
    fields |= {
        "evaluation_rollouts": "evaluation.sampled_rollouts",
        "evaluation_decoders": "evaluation.decoders",
        "evaluation_free_max_tokens": "evaluation.free_max_completion_length",
    }
    result = {}
    for name, field in fields.items():
        value = payload
        for key in field.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            result[name] = _fact(value, item, field)
    return result


def _source_facts(text, item):
    tree, result = ast.parse(text), {}
    name = item["source"]
    if name.endswith("semantic_action_parser.py"):
        functions = sorted(
            n.name
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name.startswith("parse_")
        )
        result["parser_source"] = _fact(functions, item, "AST function definitions; source-only")
    if name.endswith("test_reward_argmax_preservation.py"):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            try:
                value = ast.literal_eval(node)
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict) and set(value) == {"A_BIN", "X_BIN", "A_LEX", "X_LEX"}:
                result["reward_vectors"] = _fact(
                    value, item, f"literal reward test fixture, line {node.lineno}"
                )
    for function, label in (
        ("_render_prompt", "evaluation_prompt_constructor"),
        ("build_grpo_config_kwargs", "training_backend_config_source"),
        ("create_training_trainer", "training_backend_source"),
    ):
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function]
        if nodes:
            n = nodes[0]
            result[label] = _fact(
                {
                    "function": function,
                    "line": n.lineno,
                    "source_fragment_sha256": canonical_hash(ast.get_source_segment(text, n)),
                },
                item,
                function,
            )
    return result


def _data_inventory(body, item):
    rows = [json.loads(line) for line in body.decode("utf-8").splitlines() if line.strip()]
    if any(not isinstance(r, dict) for r in rows):
        raise ValueError("legacy data contains non-object rows")
    evaluation = [r for r in rows if r.get("split") in {"dev", "test", "positive_control"}]
    pairs = defaultdict(list)
    for row in evaluation:
        pairs[row.get("pair_id")].append(row)
    scene_ids = [r.get("scene_id") for r in evaluation]
    matched = all(
        isinstance(key, str)
        and len(group) == 2
        and {r.get("condition") for r in group} == {"collision", "separating"}
        for key, group in pairs.items()
    )
    prompt_hash_failures = sum(
        isinstance(r.get("prompt"), str)
        and r.get("prompt_sha256") is not None
        and hashlib.sha256(r["prompt"].encode()).hexdigest() != r["prompt_sha256"]
        for r in rows
    )
    return {
        "source": item["source"],
        "sha256": item["sha256"],
        "row_count": len(rows),
        "split_counts": dict(sorted(Counter(r.get("split", "missing") for r in rows).items())),
        "evaluation_scene_count": len(evaluation),
        "evaluation_pair_count": len(pairs),
        "unique_evaluation_scene_count": len(set(scene_ids)),
        "all_pairs_have_two_matched_scenes": bool(pairs)
        and matched
        and len(set(scene_ids)) == len(scene_ids),
        "prompt_count": sum(isinstance(r.get("prompt"), str) for r in rows),
        "prompt_hash_mismatch_count": prompt_hash_failures,
        "operation_metadata_count": sum(isinstance(r.get("operation"), dict) for r in rows),
        "row_schema_keys": sorted({key for row in rows for key in row}),
    }


def _merge_facts(facts, addition, conflicts):
    result = dict(facts)
    for key, value in addition.items():
        previous = result.get(key)
        if previous and previous["value"] != value["value"]:
            conflicts.append({"field": key, "evidence": previous["evidence"] + value["evidence"]})
        elif previous:
            result[key] = {**previous, "evidence": previous["evidence"] + value["evidence"]}
        else:
            result[key] = value
    return result


def _extract(inventory, contents):
    facts, data, errors, conflicts = {}, [], [], []
    for item in inventory:
        name, body = item["source"], contents.get(item["source"])
        if body is None:
            continue
        try:
            addition = {}
            if name.endswith(".py"):
                addition = _source_facts(body.decode("utf-8"), item)
            elif name.endswith("reward_fibers.jsonl"):
                counts = _data_inventory(body, item)
                data.append(counts)
                addition = {
                    "prompt_data": _fact(
                        {"prompt_count": counts["prompt_count"]}, item, "JSONL prompt fields"
                    )
                }
            elif name.endswith((".yaml", ".yml")):
                payload = yaml.safe_load(body)
                if not isinstance(payload, dict):
                    raise ValueError("legacy YAML must contain a mapping")
                if name.endswith("study_c3_resolution_validity.yaml"):
                    addition = _config_facts(payload, item)
                elif name.endswith("server_package_lock.yaml"):
                    addition = {
                        "package_versions": _fact(
                            payload.get("packages"),
                            item,
                            "packages; historical lock, not new dependencies",
                        )
                    }
            elif name.endswith("adapter_config.json"):
                payload = json.loads(body)
                allowed = {
                    k: payload[k]
                    for k in ("r", "lora_alpha", "lora_dropout", "target_modules")
                    if k in payload
                }
                addition = {
                    "initial_adapter_config": _fact(
                        allowed, item, "adapter configuration; identity unverified"
                    )
                }
            elif name.endswith("factorial_execution_contract.json"):
                payload = json.loads(body)
                addition = {
                    k: _fact(payload[k], item, k)
                    for k in (
                        "training_prompt_count",
                        "b3_adapter_sha256",
                        "model_snapshot_sha256",
                        "git_commit_sha",
                    )
                    if k in payload
                }
            facts = _merge_facts(facts, addition, conflicts)
        except (ValueError, SyntaxError, UnicodeError, yaml.YAMLError) as exc:
            errors.append({"source": name, "error": type(exc).__name__})
    logs = [i for i in inventory if i["kind"] == "training_log"]
    if logs:
        facts["training_logs"] = _fact(
            [i["source"] for i in logs], logs[0], "available log files, not new results"
        )
    return facts, data, errors, conflicts


def _archive_checks(inventory, contents):
    checks = []
    actual = {i["source"]: i["sha256"] for i in inventory}
    for name, content in contents.items():
        if not name.endswith("sha256_manifest.json") or "!/" not in name:
            continue
        try:
            expected = json.loads(content).get("files", {})
            if not isinstance(expected, dict):
                raise ValueError("invalid hash manifest")
            prefix = name.split("!/", 1)[0] + "!/"
            failures = [p for p, digest in expected.items() if actual.get(prefix + p) != digest]
            checks.append(
                {
                    "source": name,
                    "checked_file_count": len(expected),
                    "mismatches": failures,
                    "verified": bool(expected) and not failures,
                }
            )
        except (ValueError, TypeError, AttributeError):
            checks.append({"source": name, "verified": False, "error": "invalid hash manifest"})
    return checks


def _validate_locations(root, out, legacy_root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("legacy workspace root is missing or unsafe")
    resolved = out.resolve()
    for relative in SEARCH_ROOTS:
        protected = (root / relative).resolve()
        if resolved == protected or protected in resolved.parents:
            raise ValueError("output would modify a protected read-only legacy directory")
    if legacy_root is not None and (
        resolved == legacy_root.resolve() or legacy_root.resolve() in resolved.parents
    ):
        raise ValueError("output must be outside the read-only legacy root")


def _report(result):
    lines = [
        "# Legacy versus new protocol",
        "",
        "Historical evidence is read-only. No legacy code was executed.",
        "Archived results describe the historical experiment only; "
        "no new model/GPU result was produced.",
        "",
        f"legacy_exact_reproduction={str(result['legacy_exact_reproduction']).lower()}",
        "",
        "| Field | L: observed legacy evidence | N: proposed new protocol |",
        "|---|---|---|",
    ]
    for key in sorted(set(REQUIRED) | set(NEW_PROTOCOL)):
        fact = result["facts"].get(key)
        value = json.dumps(fact["value"], ensure_ascii=False) if fact else "missing"
        new = json.dumps(
            NEW_PROTOCOL.get(key, "requires phase-specific verification"), ensure_ascii=False
        )
        lines.append(f"| {key} | {value.replace('|', '/')} | {new} |")
    lines.extend(
        [
            "",
            "Missing exact-reproduction evidence: " + ", ".join(result["missing"]),
            "",
            "Source hashes and field/function provenance are recorded in audit.json. "
            "Source files alone do not verify the archived checkout or effective library defaults.",
        ]
    )
    for data in result["data_inventory"]:
        pairing = str(data["all_pairs_have_two_matched_scenes"]).lower()
        lines.append(
            f"\nObserved data: {data['row_count']} rows; "
            f"splits {json.dumps(data['split_counts'])}; "
            f"evaluation {data['evaluation_scene_count']} scenes / "
            f"{data['evaluation_pair_count']} pairs; two-condition pairing verified={pairing}."
        )
    return "\n".join(lines) + "\n"


def audit_legacy(*, root, out, legacy_root=None, legacy_archive=None):
    """Produce reviewable P0 provenance; incomplete history is a valid audit result."""
    root, out = Path(root), Path(out)
    legacy_root = Path(legacy_root) if legacy_root is not None else None
    _validate_locations(root, out, legacy_root)
    source = legacy_root or root
    if source.is_symlink() or not source.is_dir():
        raise ValueError("legacy root is missing or unsafe")
    inventory, contents = _directory_evidence(source)
    archives = []
    if legacy_archive is not None:
        items, archive_contents, archive = _archive_evidence(Path(legacy_archive))
        inventory = inventory + items
        contents = {**contents, **archive_contents}
        archives = [archive]
    facts, data, errors, conflicts = _extract(inventory, contents)
    checks = _archive_checks(inventory, contents)
    missing = sorted(key for key in REQUIRED if key not in facts)
    result = {
        "schema_version": 1,
        "phase": "P0",
        "legacy_exact_reproduction": False,
        "gpu_invoked": False,
        "legacy_code_executed": False,
        "facts": facts,
        "missing": missing,
        "inventory": inventory,
        "data_inventory": data,
        "archives": archives,
        "archive_hash_checks": checks,
        "parse_errors": errors,
        "conflicting_evidence": conflicts,
        "new_gpu_results": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "audit.json", result)
    diff = out / "legacy_vs_new_protocol.md"
    diff.write_text(_report(result), encoding="utf-8")
    write_json(out / "legacy_hash_manifest.json", {"files": inventory, "archives": archives})
    status = (
        "AUDIT_COMPLETE_HISTORY_INCOMPLETE"
        if not errors and not conflicts and all(c["verified"] for c in checks)
        else "AUDIT_EVIDENCE_REQUIRES_REVIEW"
    )
    phase_artifacts(
        out,
        "P0",
        status,
        result,
        artifacts=(out / "audit.json", diff, out / "legacy_hash_manifest.json"),
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--legacy-archive", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "phase": "P0",
                    "prompt_count": 0,
                    "rollout_count": 0,
                    "max_tokens": 0,
                    "forward_calls": 0,
                    "backward_calls": 0,
                    "budget": "CPU read-only source/archive hash audit",
                }
            )
        )
        return 0
    result = audit_legacy(
        root=args.root,
        out=args.out,
        legacy_root=args.legacy_root,
        legacy_archive=args.legacy_archive,
    )
    print(
        json.dumps(
            {
                "legacy_exact_reproduction": result["legacy_exact_reproduction"],
                "missing": result["missing"],
                "evidence_file_count": len(result["inventory"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
