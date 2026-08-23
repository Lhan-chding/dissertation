"""Artifact-producing runtimes for Study C3-A and Study C3-B."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from compensability_v5.qwen.study_b_runtime import tree_sha256
from compensability_v5.study_c2.evaluation_runtime import preflight_evaluation

from .action_audit_runtime import audit_existing_rows, summarize_action_audit
from .config_runtime import (
    load_config,
    require_offline_snapshot,
    select_evaluation_rows,
    validate_c2_evaluation_sources,
)
from .evaluation_runtime import evaluate_grid, summarize_evaluation_rows
from .io import read_json, read_jsonl, sha256_file, write_json_new, write_jsonl_new
from .paths import (
    ACTION_AUDIT_EXAMPLES,
    ACTION_AUDIT_MANIFEST,
    ACTION_AUDIT_ROWS,
    ACTION_AUDIT_SUMMARY,
    C2_EVALUATION_RAW,
    C2_FIBER_ROWS,
    CONFIG,
    EXISTING_EVAL_MANIFEST,
    EXISTING_EVAL_MASKS,
    EXISTING_EVAL_RAW,
    EXISTING_EVAL_SCENES,
    EXISTING_EVAL_SUMMARY,
    TRIE_MANIFEST,
    TRIE_PAYLOAD,
    TRIE_VALIDATION_MANIFEST,
    TRIE_VALIDATION_SUMMARY,
)
from .qwen_backend import (
    build_pinned_valid_world_trie,
    checkpoint_sampler_factory,
    load_frozen_valid_world_trie,
    load_pinned_processor,
    load_token_counter,
)
from .valid_world_trie import ValidWorldTrie


def _write_text_new(path: Path, text: str) -> None:
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"Study C3 output overwrite is forbidden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def build_failure_examples(rows: Sequence[Mapping[str, object]], *, limit_per_cell: int = 3) -> str:
    if type(limit_per_cell) is not int or limit_per_cell <= 0:
        raise ValueError("Study C3 failure-example limit must be positive")
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        arm = row.get("arm")
        category = row.get("failure_category")
        completion = row.get("completion")
        if not all(isinstance(value, str) for value in (arm, category, completion)):
            raise ValueError("Study C3 failure example row is malformed")
        if category != "canonical_valid_in_domain":
            grouped[(str(arm), str(category))].append(row)
    lines = ["# Study C3 Action-channel Failure Examples", ""]
    for (arm, category), group in sorted(grouped.items()):
        lines.extend([f"## {arm} / {category}", ""])
        for row in group[:limit_per_cell]:
            completion = json.dumps(str(row["completion"]), ensure_ascii=False)
            lines.append(f"- scene_id={row.get('scene_id')}; completion={completion}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def preflight_action_audit() -> dict[str, object]:
    runtime = require_offline_snapshot()
    provenance = validate_c2_evaluation_sources()
    raw = read_jsonl(C2_EVALUATION_RAW)
    fibers = read_jsonl(C2_FIBER_ROWS)
    if len(raw) != 5632:
        raise ValueError("Study C3-A requires all 5,632 frozen Study C2 completions")
    return {
        "schema_version": 3,
        "status": "STUDY_C3_ACTION_CHANNEL_AUDIT_PREFLIGHT_OK",
        "source_raw_row_count": len(raw),
        "fiber_row_count": len(fibers),
        "model_snapshot_sha256": runtime["model_snapshot_sha256"],
        "package_lock_sha256": runtime["package_lock_sha256"],
        **provenance,
        "training_invoked": False,
        "optimizer_step_invoked": False,
        "rl_invoked": False,
        "gpu_invoked": False,
    }


def run_action_audit() -> dict[str, object]:
    preflight = preflight_action_audit()
    audited = audit_existing_rows(
        read_jsonl(C2_EVALUATION_RAW),
        read_jsonl(C2_FIBER_ROWS),
        token_counter=load_token_counter(),
    )
    summary = summarize_action_audit(audited)
    write_jsonl_new(ACTION_AUDIT_ROWS, audited)
    write_json_new(ACTION_AUDIT_SUMMARY, summary)
    _write_text_new(ACTION_AUDIT_EXAMPLES, build_failure_examples(audited))
    manifest = {
        **preflight,
        "status": "STUDY_C3_ACTION_CHANNEL_AUDIT_COMPLETE",
        "per_row_sha256": sha256_file(ACTION_AUDIT_ROWS),
        "summary_sha256": sha256_file(ACTION_AUDIT_SUMMARY),
        "failure_examples_sha256": sha256_file(ACTION_AUDIT_EXAMPLES),
        "row_count": len(audited),
    }
    write_json_new(ACTION_AUDIT_MANIFEST, manifest)
    return manifest


def preflight_trie_build() -> dict[str, object]:
    provenance = require_offline_snapshot()
    load_config(CONFIG)
    return {
        "schema_version": 3,
        "status": "STUDY_C3_VALID_WORLD_TRIE_PREFLIGHT_OK",
        "expected_world_count": 17**4,
        **provenance,
        "gpu_invoked": False,
    }


def run_trie_build() -> dict[str, object]:
    preflight = preflight_trie_build()
    trie = build_pinned_valid_world_trie()
    payload = trie.to_payload()
    write_json_new(TRIE_PAYLOAD, payload)
    manifest = {
        **preflight,
        "status": "STUDY_C3_VALID_WORLD_TRIE_COMPLETE",
        "trie_sha256": sha256_file(TRIE_PAYLOAD),
        "world_count": trie.world_count,
        "node_count": payload["node_count"],
        "eos_token_id": trie.eos_token_id,
    }
    write_json_new(TRIE_MANIFEST, manifest)
    return manifest


def preflight_trie_validation() -> dict[str, object]:
    provenance = require_offline_snapshot()
    manifest = read_json(TRIE_MANIFEST)
    if manifest.get("status") != "STUDY_C3_VALID_WORLD_TRIE_COMPLETE" or manifest.get(
        "trie_sha256"
    ) != sha256_file(TRIE_PAYLOAD):
        raise ValueError("Study C3 valid-world trie manifest drifted")
    return {
        "schema_version": 3,
        "status": "STUDY_C3_VALID_WORLD_TRIE_VALIDATION_PREFLIGHT_OK",
        "trie_sha256": manifest["trie_sha256"],
        **provenance,
        "gpu_invoked": False,
    }


def run_trie_validation() -> dict[str, object]:
    preflight = preflight_trie_validation()
    processor = load_pinned_processor()
    tokenizer = getattr(processor, "tokenizer", processor)
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise RuntimeError("Study C3 Qwen tokenizer lacks encode support")
    trie = ValidWorldTrie.from_payload(read_json(TRIE_PAYLOAD))
    legal_count = 0
    multi_token_number_observed = False
    for action in trie.iter_actions():
        tokens = encode(action, add_special_tokens=False)
        if not isinstance(tokens, list) or not trie.accepts(tokens):
            raise RuntimeError(f"Study C3 trie rejects legal action: {action}")
        legal_count += 1
    for number in range(2, 19):
        tokens = encode(str(number), add_special_tokens=False)
        if isinstance(tokens, list) and len(tokens) > 1:
            multi_token_number_observed = True
    invalid = ("1,3,4,5", "2,3,4,19", "[2,3,4,5]", "2,3,4", "2,3,4,5 prose")
    blocked = 0
    for action in invalid:
        tokens = encode(action, add_special_tokens=False)
        if not isinstance(tokens, list) or trie.accepts(tokens):
            raise RuntimeError(f"Study C3 trie accepts invalid action: {action}")
        blocked += 1
    probe = encode("2,3,4,5", add_special_tokens=False)
    if not isinstance(probe, list) or trie.allowed_next(probe) != {trie.eos_token_id}:
        raise RuntimeError("Study C3 trie does not enforce EOS-only terminal transition")
    summary = {
        "schema_version": 3,
        "status": "STUDY_C3_VALID_WORLD_TRIE_VALIDATED",
        "legal_action_count": legal_count,
        "invalid_fixture_count": len(invalid),
        "invalid_fixture_blocked_count": blocked,
        "eos_only_after_complete_action": True,
        "tokenizer_agnostic_construction": True,
        "multi_token_number_observed": multi_token_number_observed,
        "training_invoked": False,
        "optimizer_step_invoked": False,
        "rl_invoked": False,
        "gpu_invoked": False,
    }
    write_json_new(TRIE_VALIDATION_SUMMARY, summary)
    manifest = {
        **preflight,
        "status": "STUDY_C3_VALID_WORLD_TRIE_VALIDATION_COMPLETE",
        "validation_summary_sha256": sha256_file(TRIE_VALIDATION_SUMMARY),
        "legal_action_count": legal_count,
    }
    write_json_new(TRIE_VALIDATION_MANIFEST, manifest)
    return manifest


def _c2_checkpoints(preflight: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    arms = preflight.get("arm_manifests")
    if not isinstance(arms, Mapping):
        raise ValueError("Study C2 checkpoint provenance is unavailable")
    result: list[dict[str, object]] = []
    for name, verifier in (
        ("C2_answer_reward", "answer"),
        ("C2_exact_state_reward", "state"),
    ):
        details = arms.get(name)
        if not isinstance(details, Mapping):
            raise ValueError(f"Study C2 checkpoint provenance is malformed: {name}")
        adapter_path = Path(str(details["adapter_path"]))
        observed = tree_sha256(adapter_path)
        if observed != details.get("final_adapter_sha256"):
            raise ValueError(f"Study C2 adapter tree hash drifted: {name}")
        result.append(
            {
                "checkpoint_id": name,
                "checkpoint_step": 192,
                "verifier": verifier,
                "validity_channel": "binary",
                "adapter_path": str(adapter_path),
                "adapter_sha256": observed,
                "source_manifest_sha256": details["manifest_sha256"],
            }
        )
    return tuple(result)


def preflight_existing_checkpoint_evaluation() -> dict[str, object]:
    config = load_config(CONFIG)
    trie = load_frozen_valid_world_trie()
    c2 = preflight_evaluation(config_path=Path("configs/v5/study_c2_identifiable_reward.yaml"))
    source = validate_c2_evaluation_sources()
    rows = select_evaluation_rows(read_jsonl(C2_FIBER_ROWS))
    checkpoints = _c2_checkpoints(c2)
    intervention = config["existing_checkpoint_intervention"]
    assert isinstance(intervention, Mapping)
    return {
        "schema_version": 3,
        "status": "STUDY_C3_EXISTING_CHECKPOINT_INTERVENTION_PREFLIGHT_OK",
        "evaluation_scene_count": len(rows),
        "evaluation_pair_count": len(rows) // 2,
        "sampled_rollouts": intervention["sampled_rollouts"],
        "decoders": intervention["decoders"],
        "checkpoints": checkpoints,
        "valid_world_trie_sha256": sha256_file(TRIE_PAYLOAD),
        "valid_world_count": trie.world_count,
        "rollout_seed_base": source["rollout_seed_base"],
        "rollout_seed_algorithm": source["rollout_seed_algorithm"],
        "c2_evaluation_manifest_sha256": source["manifest_sha256"],
        "model_snapshot_sha256": c2["model_snapshot_sha256"],
        "package_lock_sha256": c2["package_lock_sha256"],
        "training_invoked": False,
        "optimizer_step_invoked": False,
        "rl_invoked": False,
        "gpu_invoked": False,
    }


def _original_free16_parity(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    frozen = {
        (str(row["arm"]), str(row["scene_id"]), int(row["rollout_index"])): str(row["completion"])
        for row in read_jsonl(C2_EVALUATION_RAW)
    }
    compared = 0
    matched = 0
    for row in rows:
        if row.get("eval_decoder") != "free_16":
            continue
        key = (
            str(row["checkpoint_id"]),
            str(row["scene_id"]),
            int(row["rollout_index"]),
        )
        if key not in frozen:
            raise ValueError("Study C3 D0 result lacks a frozen Study C2 counterpart")
        compared += 1
        matched += str(row["completion"]) == frozen[key]
    if compared != 5632 or matched != compared:
        raise RuntimeError(
            "Study C3 D0 failed exact completion parity with frozen Study C2 evaluation"
        )
    return {
        "compared_rollout_count": compared,
        "identical_completion_count": matched,
        "identical_completion_rate": None if compared == 0 else matched / compared,
    }


def run_existing_checkpoint_evaluation() -> dict[str, object]:
    preflight = preflight_existing_checkpoint_evaluation()
    rows = select_evaluation_rows(read_jsonl(C2_FIBER_ROWS))
    raw, masks = evaluate_grid(
        rows=rows,
        checkpoints=preflight["checkpoints"],  # type: ignore[arg-type]
        decoders=preflight["decoders"],  # type: ignore[arg-type]
        rollout_count=int(preflight["sampled_rollouts"]),
        seed_base=int(preflight["rollout_seed_base"]),
        sampler_factory=checkpoint_sampler_factory,
    )
    per_scene, summary = summarize_evaluation_rows(
        raw, expected_rollouts=int(preflight["sampled_rollouts"])
    )
    summary = {
        **summary,
        "status": "STUDY_C3_EXISTING_CHECKPOINT_DECODER_INTERVENTION_COMPLETE",
        "free_16_reproduction": _original_free16_parity(raw),
    }
    write_jsonl_new(EXISTING_EVAL_RAW, raw)
    write_jsonl_new(EXISTING_EVAL_MASKS, masks)
    write_jsonl_new(EXISTING_EVAL_SCENES, per_scene)
    write_json_new(EXISTING_EVAL_SUMMARY, summary)
    manifest = {
        **preflight,
        "status": "STUDY_C3_EXISTING_CHECKPOINT_DECODER_INTERVENTION_COMPLETE",
        "raw_rows_sha256": sha256_file(EXISTING_EVAL_RAW),
        "logit_masks_sha256": sha256_file(EXISTING_EVAL_MASKS),
        "per_scene_sha256": sha256_file(EXISTING_EVAL_SCENES),
        "summary_sha256": sha256_file(EXISTING_EVAL_SUMMARY),
        "raw_row_count": len(raw),
        "mask_row_count": len(masks),
        "scene_cell_count": len(per_scene),
        "gpu_invoked": True,
    }
    write_json_new(EXISTING_EVAL_MANIFEST, manifest)
    return manifest


__all__ = [
    "build_failure_examples",
    "preflight_action_audit",
    "preflight_existing_checkpoint_evaluation",
    "preflight_trie_build",
    "preflight_trie_validation",
    "run_action_audit",
    "run_existing_checkpoint_evaluation",
    "run_trie_build",
    "run_trie_validation",
]
