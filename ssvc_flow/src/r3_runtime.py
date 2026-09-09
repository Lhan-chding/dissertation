"""Resumable real cold R3 rollout, isolated optimizer fork, and control scoring."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .constraint_solver import solve
from .core import (
    RunStore,
    canonical_hash,
    file_hash,
    frozen_writer,
    phase_artifacts,
    write_json,
)
from .optimizer_fork import (
    capture_state,
    load_checkpoint,
    parameter_hash,
    restore_state,
    save_checkpoint,
    state_hash,
)
from .r1_reference_smoke import _certificate_check, _helper_config
from .r1_supplement import _ExecutionMeter
from .r2_runtime import _json_safe, _source, annotate_diagnostic, generation_checks
from .smoke_runtime import _seed_everything, _verify_scene_images

LAMBDAS = (0.0, 0.01, 0.25, 1.0, 2.0)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _manifest(root, *, exclude=()):
    return {
        "files": [
            {"path": str(p.relative_to(root)), "sha256": file_hash(p), "bytes": p.stat().st_size}
            for p in sorted(root.rglob("*"))
            if p.is_file()
            and str(p.relative_to(root)) not in set(exclude)
            and not p.name.endswith(".tmp")
            and p.name != ".writer.lock"
        ]
    }


def _verify_manifest(root, manifest):
    paths = set()
    for item in manifest["files"]:
        relative = Path(item["path"])
        path = (root / relative).resolve()
        if (
            relative.is_absolute()
            or not path.is_relative_to(root.resolve())
            or str(relative) in paths
        ):
            raise ValueError("R3 attempt manifest path is duplicated or escapes root")
        paths.add(str(relative))
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or file_hash(path) != item["sha256"]
        ):
            raise ValueError(f"R3 attempt manifest hash/size mismatch: {relative}")


def run_atomic_unit(root, identity, operation):
    """Never overwrite successful measured units or discard interrupted attempts."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    completed = []
    attempts = sorted(root.glob("attempt_*"))
    for attempt in attempts:
        if not (attempt / "identity.json").exists():
            continue  # Preserve a kill between mkdir and the atomic identity write.
        if _json(attempt / "identity.json") != identity:
            raise ValueError("R3 atomic unit identity changed on resume")
        marker = attempt / "completed.json"
        if not marker.exists():
            continue
        record = _json(marker)
        if record.get("status") != "PASS" or record.get("identity_hash") != canonical_hash(
            identity
        ):
            raise ValueError("R3 atomic unit completion binding mismatch")
        if file_hash(attempt / "manifest.json") != record["manifest_sha256"]:
            raise ValueError("R3 atomic unit manifest hash mismatch")
        _verify_manifest(attempt, _json(attempt / "manifest.json"))
        completed.append((attempt, _json(attempt / "result.json"), True))
    if len(completed) > 1:
        raise ValueError("R3 atomic unit has multiple successful attempts")
    if completed:
        return completed[0]
    attempt = root / f"attempt_{len(attempts):04d}"
    attempt.mkdir(exist_ok=False)
    write_json(attempt / "identity.json", identity)
    try:
        result = operation(attempt)
        write_json(attempt / "result.json", result)
        write_json(
            attempt / "manifest.json",
            _manifest(attempt, exclude=("manifest.json", "completed.json")),
        )
        write_json(
            attempt / "completed.json",
            {
                "status": "PASS",
                "identity_hash": canonical_hash(identity),
                "manifest_sha256": file_hash(attempt / "manifest.json"),
            },
        )
        return attempt, result, False
    except Exception as exc:
        write_json(
            attempt / "failure.json",
            {"status": "FAIL", "type": type(exc).__name__, "message": str(exc)},
        )
        raise


def build_requests(plan, identity):
    prompts = {r["prompt_id"]: r for r in [*plan["train_prompts"], *plan["control_prompts"]]}
    assignments = [
        ("train", bank_index, prompt_id, index)
        for bank_index, bank in enumerate(plan["banks"])
        for prompt_id in bank
        for index in range(8)
    ] + [
        ("control_proposal", None, record["prompt_id"], index)
        for record in plan["control_prompts"]
        for index in range(16)
    ]
    requests = []
    for role, bank_index, prompt_id, index in assignments:
        prompt = prompts[prompt_id]
        request = {
            "phase": identity["phase"],
            "bank_role": role,
            "bank_index": bank_index,
            "base_scene_id": prompt["base_scene_id"],
            "prompt_id": prompt_id,
            "family": prompt["family"],
            "interface": prompt["interface"],
            "split": prompt["split"],
            "prompt_hash": prompt["prompt_hash"],
            "scene_hash": prompt["scene_hash"],
            "group_id": prompt_id,
            "sample_index": index,
            "rollout_index": index,
            "decode_mode": "sample",
            "max_new_tokens": 64,
            "enable_thinking": False,
            "origin_state_hash": identity.get("origin_hash"),
            "checkpoint_step": identity.get("checkpoint_step", 0),
        }
        rng = {
            "seed_root": 20260909,
            "phase": identity["phase"],
            "checkpoint_step": identity.get("checkpoint_step"),
            "model_hash": identity.get("model_hash"),
            "adapter_hash": identity.get("initial_adapter_hash"),
            "origin_hash": identity.get("origin_hash"),
            "plan_hash": plan["plan_hash"],
            "prompt_id": prompt_id,
            "role": role,
            "sample_index": index,
        }
        rng_key = canonical_hash(rng)
        requests.append(
            {
                **request,
                "sample_rng_key": rng_key,
                "sample_seed": int(rng_key[:8], 16) % (2**31),
                "sample_key": canonical_hash(
                    {"identity": identity, "request": request, "rng": rng}
                ),
            }
        )
    if len(requests) != 1152 or len({r["sample_key"] for r in requests}) != 1152:
        raise ValueError("R3 requires distinct 384 train and 768 control requests")
    return requests


def validate_sample_ledger(records, requests):
    wanted = {r["sample_key"]: r for r in requests}
    for key, row in records.items():
        if key not in wanted or any(row.get(k) != v for k, v in wanted[key].items()):
            raise ValueError("R3 sample identity mismatch")
        if row.get("record_hash") != canonical_hash(
            {k: v for k, v in row.items() if k != "record_hash"}
        ):
            raise ValueError("R3 sample content hash mismatch")
        if row.get("execution_checks", {}).get("passed") is not True:
            raise ValueError("R3 preserved failed execution cannot be skipped")


def _load_plan(data_root, gate):
    from .r3_inputs import build_r3_plan

    root = Path(data_root)
    manifest = _json(root / "manifest.json")
    if file_hash(root / "manifest.json") != gate["binding"]["data_manifest_sha256"]:
        raise ValueError("R3 dataset manifest differs from passed R0 evidence")
    splits, hashes = {}, {}
    for split in ("train", "control"):
        path = root / f"{split}.jsonl"
        digest = file_hash(path)
        if any(
            record["sha256"] != digest
            for record in (
                manifest["files"][path.name],
                gate["binding"]["dataset_files"][path.name],
            )
        ):
            raise ValueError(f"R3 {split} source hash differs from passed split-separation audit")
        hashes[path.name] = digest
        splits[split] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    plan = build_r3_plan(splits["train"], splits["control"])
    selected = {
        r["base_scene_id"]: r["scene"] for r in [*plan["train_prompts"], *plan["control_prompts"]]
    }
    _verify_scene_images(list(selected.values()), root)
    for scene in selected.values():
        changed = [i for i in range(4) if scene["truth_world"][i] != scene["observed_world"][i]]
        if solve(scene["observed_world"], scene["cue"]) != [scene["truth_world"]] or changed != [
            scene["changed_index"]
        ]:
            raise ValueError("R3 selected scene fails independent unique-repair solver")
    if plan["reuse_check_bank_indices"] != [1, 11] or plan["response_banks"] != [0, 6]:
        raise ValueError("R3 predeclared validation or response bank selection changed")
    return plan, {
        "manifest_sha256": file_hash(root / "manifest.json"),
        "files": hashes,
        "selected_scene_hashes": {key: canonical_hash(scene) for key, scene in selected.items()},
    }


def _restore(adapter, optimizer, origin):
    adapter.model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    if hasattr(adapter, "_reset_positions"):
        adapter._reset_positions()
    restore_state(adapter.model, optimizer, origin)
    restored = capture_state(adapter.model, optimizer, origin["metadata"])
    if state_hash(restored) != state_hash(origin):
        raise RuntimeError("R3 full origin state or RNG restoration mismatch")


def _save_tensor_payload(path, payload, identity):
    import torch

    value = {"identity": identity, "payload_hash": state_hash(payload), "payload": payload}
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)
    return {
        "file": str(path.name),
        "sha256": file_hash(path),
        "payload_hash": value["payload_hash"],
    }


def _load_tensor_payload(path, identity, binding):
    import torch

    if file_hash(path) != binding["sha256"]:
        raise ValueError("R3 score-cache file hash mismatch")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (
        value["identity"] != identity
        or value["payload_hash"] != binding["payload_hash"]
        or state_hash(value["payload"]) != binding["payload_hash"]
    ):
        raise ValueError("R3 score-cache state identity mismatch")
    return value["payload"]


def _prepared(adapter, prompt_record, data_root):
    prepared = adapter.prepare(prompt_record["prompt"], data_root)
    audit = prepared["audit"]
    image = prompt_record["interface"] == "IMAGE_CUE_FRESH"
    if audit.get("enable_thinking") is not False:
        raise ValueError("R3 requires the original non-thinking template")
    if bool(audit.get("image_token_count")) != image or (
        image and not audit.get("pixel_values_hash")
    ):
        raise ValueError("R3 intervention image presence mismatch")
    return prepared


def _collect_samples(adapter, optimizer, origin, plan, requests, store, data_root, meter, profile):
    import torch

    prompt_map = {r["prompt_id"]: r for r in [*plan["train_prompts"], *plan["control_prompts"]]}
    current, prepared = None, None
    generated = 0
    for request in requests:
        if request["sample_key"] in store.keys:
            continue
        prompt_record = prompt_map[request["prompt_id"]]
        if current != request["prompt_id"]:
            prepared = _prepared(adapter, prompt_record, data_root)
            current = request["prompt_id"]
        scene = prompt_record["scene"]
        generation, returned = None, False
        try:
            before = adapter.forward_calls
            with meter.scope("sampling/" + request["bank_role"]), torch.no_grad():
                generation = adapter.generate(
                    prepared, seed=request["sample_seed"], max_new_tokens=64, do_sample=True
                )
                returned = True
            generated += 1
            faults = generation_checks(generation, adapter.processor.tokenizer, adapter.eos_ids, 64)
            if (
                request["interface"] == "IMAGE_CUE_FRESH"
                and generation.get("vision_forward_calls", 0) < 1
            ):
                faults.append("image did not enter measured visual forward")
            annotation = annotate_diagnostic(generation["raw_completion"], scene)
            scores = generation["behavior_token_logprobs"]
            row = _json_safe(
                {
                    **request,
                    "run_id": profile.run_id,
                    "execution_kind": "CPU_FAKE_ADAPTER_FIXTURE"
                    if profile.fixture
                    else "REAL_CUDA_FORK",
                    "model_id": adapter.model_id,
                    "model_revision": adapter.revision,
                    "adapter_hash": profile.adapter_hash,
                    "checkpoint_step": 0,
                    "train_seed": 17,
                    "optimizer_state_hash": state_hash(origin["optimizer"]),
                    "protocol_version": profile.protocol_version,
                    "chart_type": scene["chart_type"],
                    "operation": scene["operation"],
                    "truth_world": scene["truth_world"],
                    "observed_world": scene["observed_world"],
                    "changed_index": scene["changed_index"],
                    "cue": scene["cue"],
                    "solution_count": 1,
                    "image_hash": scene["image_hash"]
                    if prepared["audit"]["image_token_count"]
                    else None,
                    **prepared["audit"],
                    **generation,
                    **annotation,
                    "raw_token_ids": generation["token_ids"],
                    "raw_text": generation["raw_completion"],
                    "old_logprobs": scores,
                    "per_token_logprob_behavior": scores,
                    "logprob_sequence": math.fsum(scores),
                    "generation_config_hash": profile.generation_hash,
                    "input_ids_hash": prepared["audit"].get("tokenized_prompt_hash"),
                    "actual_image_tokens": prepared["audit"].get("image_token_count"),
                    "n_generated_tokens": len(generation["token_ids"]),
                    "runtime_forward_by_reason": {"generation": adapter.forward_calls - before},
                    "peak_memory": meter.memory.get("sampling/" + request["bank_role"]),
                    "execution_checks": {"passed": not faults, "faults": faults},
                }
            )
            row["record_hash"] = canonical_hash(row)
            store.append(row)
            returned = False
            profile.write("SAMPLING", completed=len(store.records), total=len(requests))
            if faults:
                raise RuntimeError("R3 generation execution fault; raw output preserved")
        except Exception as exc:
            if returned:
                failure = _json_safe(
                    {
                        **request,
                        "execution_kind": profile.execution_kind,
                        "generation_return": generation,
                        "parse_result": "unmeasured_due_to_execution_error",
                        "execution_checks": {
                            "passed": False,
                            "faults": [f"{type(exc).__name__}: {exc}"],
                        },
                    }
                )
                failure["record_hash"] = canonical_hash(failure)
                store.append(failure)
            profile.write("SAMPLING_FAILED", completed=len(store.records), total=len(requests))
            raise
    if parameter_hash(adapter.model, trainable=True) != profile.adapter_hash or state_hash(
        optimizer.state_dict()
    ) != state_hash(origin["optimizer"]):
        write_json(
            profile.out / "policy_contamination.json",
            {
                "phase": "sampling",
                "reason": "Adapter or Adam changed before origin restoration",
            },
        )
        raise RuntimeError("R3 sampling changed policy/optimizer; saved rows cannot be resumed")
    _restore(adapter, optimizer, origin)
    return generated


def _bank_groups(plan, records, index, adapter, data_root):
    prompt_map = {r["prompt_id"]: r for r in plan["train_prompts"]}
    groups = []
    for prompt_id in plan["banks"][index]:
        prompt = prompt_map[prompt_id]
        group = sorted(
            (r for r in records.values() if r["prompt_id"] == prompt_id),
            key=lambda r: r["sample_index"],
        )
        if (
            len(group) != 8
            or [r["sample_index"] for r in group] != list(range(8))
            or any(
                r["bank_role"] != "train" or r["split"] != "train" or r["bank_index"] != index
                for r in group
            )
        ):
            raise ValueError("R3 training bank completeness or control-separation error")
        prepared = _prepared(adapter, prompt, data_root)
        if any(
            r["final_prompt_hash"] != prepared["audit"]["final_prompt_hash"]
            or r.get("input_tensor_hash") != prepared["audit"].get("input_tensor_hash")
            for r in group
        ):
            raise ValueError("R3 original rollout prepared-input binding changed")
        groups.append([{**row, "prepared": prepared} for row in group])
    return groups


class _Profile:
    def __init__(self, out, invocation, meter, fixture, config, identity):
        self.out, self.invocation, self.meter = out, invocation, meter
        self.fixture = fixture
        self.protocol_version = config["protocol_version"]
        self.generation_hash = canonical_hash(config["generation_proposed_N"])
        self.execution_kind = "CPU_FAKE_ADAPTER_FIXTURE" if fixture else "REAL_CUDA_FORK"
        self.run_id = canonical_hash(identity)
        self.adapter_hash = identity["initial_adapter_hash"]

    def write(self, state, **progress):
        write_json(self.invocation / "runtime_profile.json", self.meter.report())
        write_json(self.invocation / "progress.json", {"state": state, **progress})
        write_json(
            self.out / "progress.json",
            {"state": state, "invocation": self.invocation.name, **progress},
        )


class _MeterPrefix:
    def __init__(self, meter, prefix):
        self.meter, self.prefix = meter, prefix

    def scope(self, label):
        return self.meter.scope(f"{self.prefix}/{label}")


def _candidate_checkpoint(candidate, attempt):
    path = Path(candidate["checkpoint_path"])
    if not path.is_absolute():
        path = attempt / "engine" / path
    path = path.resolve()
    if not path.is_relative_to(attempt.resolve()):
        raise ValueError("R3 candidate checkpoint path escapes its measured attempt")
    state = load_checkpoint(path, candidate["checkpoint_identity"])
    params = state_hash({name: state_hash(tensor) for name, tensor in state["parameters"].items()})
    if (
        params != candidate["parameter_hash"]
        or state_hash(state["optimizer"]) != candidate["optimizer_state_hash"]
    ):
        raise ValueError("R3 candidate state hashes disagree with saved checkpoint")
    return state


def _score_candidate(
    adapter,
    optimizer,
    origin,
    candidate,
    attempt,
    proposal,
    plan,
    out,
    identity,
    data_root,
    meter,
    profile,
):
    import torch

    state = _candidate_checkpoint(candidate, attempt)
    score_identity = {
        **identity,
        "unit": "control_likelihood",
        "candidate_id": candidate["candidate_id"],
        "candidate_parameter_hash": candidate["parameter_hash"],
        "candidate_optimizer_hash": candidate["optimizer_state_hash"],
        "proposal_hash": canonical_hash([r["record_hash"] for r in proposal]),
    }
    score_root = out / "control_scores" / candidate["candidate_id"]
    if "/" in candidate["candidate_id"] or candidate["candidate_id"] in {".", ".."}:
        raise ValueError("R3 candidate_id must be a single path component")
    scores = RunStore(score_root, score_identity, resume=(score_root / "identity.json").exists())
    requests = [
        {
            "sample_key": canonical_hash([score_identity, row["sample_key"]]),
            "proposal_sample_key": row["sample_key"],
            "candidate_id": candidate["candidate_id"],
        }
        for row in proposal
    ]
    validate_sample_ledger(scores.records, requests)
    prompt_map = {r["prompt_id"]: r for r in plan["control_prompts"]}
    current, prepared = None, None
    try:
        _restore(adapter, optimizer, state)
        for proposal_row, request in zip(proposal, requests, strict=True):
            if request["sample_key"] in scores.keys:
                continue
            if current != proposal_row["prompt_id"]:
                prepared = _prepared(adapter, prompt_map[proposal_row["prompt_id"]], data_root)
                current = proposal_row["prompt_id"]
            if proposal_row.get("input_tensor_hash") != prepared["audit"].get("input_tensor_hash"):
                raise ValueError("R3 control likelihood prepared-input binding changed")
            returned, value = False, None
            try:
                before = adapter.forward_calls
                with (
                    meter.scope("control_likelihood/" + candidate["candidate_id"]),
                    torch.no_grad(),
                ):
                    value = adapter.logprobs(
                        prepared, proposal_row["token_ids"], require_grad=False
                    )
                    returned = True
                token_scores = value.detach().cpu().tolist()
                faults = []
                if len(token_scores) != len(proposal_row["token_ids"]) or any(
                    not math.isfinite(v) or v > 1e-5 for v in token_scores
                ):
                    faults.append("malformed candidate token log probabilities")
                row = _json_safe(
                    {
                        **request,
                        "phase": identity["phase"],
                        "execution_kind": profile.execution_kind,
                        "candidate_parameter_hash": candidate["parameter_hash"],
                        "proposal_record_hash": proposal_row["record_hash"],
                        "token_ids": proposal_row["token_ids"],
                        "candidate_token_logprobs": token_scores,
                        "sequence_logprob": math.fsum(token_scores),
                        "forward_calls": adapter.forward_calls - before,
                        "execution_checks": {"passed": not faults, "faults": faults},
                    }
                )
                row["record_hash"] = canonical_hash(row)
                scores.append(row)
                returned = False
                profile.write(
                    "CONTROL_SCORING",
                    candidate=candidate["candidate_id"],
                    completed=len(scores.records),
                    total=768,
                )
                if faults:
                    raise RuntimeError("R3 candidate score fault; returned probabilities preserved")
            except Exception as exc:
                if returned:
                    row = _json_safe(
                        {
                            **request,
                            "returned_scores": value.detach().cpu().tolist()
                            if hasattr(value, "detach")
                            else value,
                            "execution_checks": {"passed": False, "faults": [str(exc)]},
                        }
                    )
                    row["record_hash"] = canonical_hash(row)
                    scores.append(row)
                raise
        if (
            parameter_hash(adapter.model, trainable=True) != candidate["parameter_hash"]
            or state_hash(optimizer.state_dict()) != candidate["optimizer_state_hash"]
        ):
            raise RuntimeError("R3 candidate changed during control-only likelihood scoring")
        if len(scores.records) != 768:
            raise ValueError("R3 candidate has incomplete control likelihood coverage")
        return {
            r["proposal_sample_key"]: r["candidate_token_logprobs"] for r in scores.records.values()
        }
    finally:
        _restore(adapter, optimizer, origin)


def _write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_reports(out, summaries, response_results, validation, details):
    import pyarrow as pa
    import pyarrow.parquet as pq

    candidates, responses = [], []
    for index, summary in summaries.items():
        for candidate in summary["candidates"]:
            scalars = {
                k: v
                for k, v in candidate.items()
                if v is None or isinstance(v, (bool, int, float, str))
            }
            candidates.append(
                {
                    "bank_index": index,
                    **scalars,
                    "probability_response": "MEASURED_CONTROL_IS"
                    if index in (0, 6)
                    else "NOT_MEASURED",
                    "audit_json": json.dumps(candidate, ensure_ascii=False, allow_nan=False),
                }
            )
    pq.write_table(pa.Table.from_pylist(candidates), out / "gradients_summary.parquet")
    write_json(
        out / "candidate_manifest.json",
        {
            "distinct_candidates": len(candidates),
            "banks": summaries,
            "committed_candidates": 0,
            "response_bank_indices": [0, 6],
        },
    )
    write_json(
        out / "joint_advantage_checks.json",
        {
            "reuse_validation": validation,
            "bank_advantage_and_gradient_audits": summaries,
            "scope": "cold fixed-state single update; no multi-epoch or changed-policy reuse",
        },
    )
    support = [
        "# R3-cold control 支持与响应",
        "",
        "主值为未裁剪普通 importance estimator。"
        "全部差值保留同一 proposal 的配对结构；没有安全认证。",
        "",
    ]
    for index, result in response_results.items():
        for key, candidate in result["candidates"].items():
            for scope, metrics in candidate["responses"].items():
                for metric, values in metrics.items():
                    row = {
                        "bank_index": index,
                        "candidate_id": key,
                        "scope": scope,
                        "metric": metric,
                        **{k: v for k, v in values.items() if k != "ci"},
                    }
                    for ci_type, ci in values["ci"].items():
                        row.update({f"{ci_type}_CI_{name}": value for name, value in ci.items()})
                    responses.append(row)
            support.append(
                f"- bank {index}, candidate {key}: "
                + json.dumps(candidate["warnings"], ensure_ascii=False)
            )
    _write_csv(out / "paired_response.csv", responses)
    support.extend(
        [
            "",
            "每题无观测 X 时，经验重加权 X 贡献为零；"
            "真实 pX/条件类别梯度不可由此估计。ESS 与最大归一化权重警告仅是 overlap 诊断。",
            "",
            "逐题、六群体与整体诊断以及序列 log-ratio/weight 见 res"
            "ponse_bank_00.json 和 response_bank_06."
            "json。",
        ]
    )
    (out / "support_control_report.md").write_text("\n".join(support) + "\n", encoding="utf-8")
    text = [
        "# R3-cold 实际执行报告",
        "",
        f"执行类型：{details['execution_kind']}。状态：{details['status']}。",
        "",
        "已执行固定 train 48 prompts * 8 = 384 条新 on"
        "-policy 输出，control 24 base scenes * 2 "
        "interfaces * 16 = 768 条 proposal；不筛掉无 "
        "X 或零优势组。",
        "12 个 B4K8 bank，各 5 个隔离 Adam 候选，共 60 个不"
        "同候选；λ0 重放和两 bank 复用验证的额外 optimizer/bac"
        "kward 单独记账。所有候选丢弃恢复，不进入长期训练。",
        "只对事前指定 bank 0/6 的 10 个候选完整评分 768 条 con"
        "trol 输出；其他 bank 的概率响应明确 NOT_MEASURED。c"
        "ontrol 不进入训练梯度。",
        "",
        "## 统计单位与范围",
        "",
        "按 base_scene、家族分层 cluster bootstrap 5,"
        "000 次，两个接口配对；每题先平均输出，再按六群体固定权重聚合，每次重算 "
        "q 比值。CI 仅针对该固定 checkpoint 的评估场景不确定性。",
        "",
        "cold 仅为初始状态实现与机制诊断；未执行 warm、cold 直接重采样"
        "、R4 长期训练或在线 SSVC。SGD/固定预条件参考只描述参数空间，不冒"
        "充已测概率响应。",
        "",
        "## 配置与证据",
        "",
        "使用 R1 认证的原始模型、初始化 LoRA、空 Adam 与非 think"
        "ing 64-token uncached 路径。没有加载 smoke 更新"
        " checkpoint，没有修改 LR/clip/reward 配置。",
        "原始 samples.jsonl、bank_manifest.json、candidate_manifest.json、"
        f"control_scores/、validation/、forks/ 均位于 {out}。",
        "",
        "不可变 attempt 目录保留中断和失败过程；成功单位只在逐文件 hash"
        "/size 校验后复用。runtime_profile.json 汇总实际观"
        "测调用；未正常结束的调用区间标为未知额外成本，不称为零。",
        "",
        "## 实际计数与状态",
        "",
        "```json",
        json.dumps(details, ensure_ascii=False, indent=2, allow_nan=False),
        "```",
        "",
    ]
    for name in ("report_zh.md", "cold_report.md"):
        (out / name).write_text("\n".join(text), encoding="utf-8")


def _final_status(out, status, execution, details):
    write_json(out / "progress.json", {"state": status, **details})
    artifacts = [
        out / item["path"]
        for item in _manifest(out, exclude=("status.json", "manifest.json", "report.md"))["files"]
    ]
    result = phase_artifacts(out, "R3-cold", status, details, artifacts)
    result["execution_kind"] = execution
    write_json(out / "status.json", result)
    return result


def run_r3(
    config,
    data_root,
    out,
    r0_dir,
    r1_run,
    supplement_dir,
    r2_dir,
    *,
    state="cold",
    resume=False,
    _adapter_factory=None,
):
    if state != "cold":
        out = Path(out)
        if out.exists() and any(out.iterdir()):
            raise ValueError(
                "Refusing to overwrite an existing run with an unimplemented warm stage"
            )
        result = phase_artifacts(
            out,
            "R3-warm",
            "BLOCKED",
            {
                "warm_implemented": False,
                "reason": (
                    "Cold runner does not implement the complete warm "
                    "checkpoint/direct-resampling protocol"
                ),
            },
        )
        result["execution_kind"] = "CPU_AUDIT"
        write_json(out / "status.json", result)
        return result

    import torch

    from .model_adapters import load_adapter
    from .next_stage_runtime import (
        validate_config_against_gate,
        validate_prerequisites,
        validate_r2_gate,
        validate_runtime_environment,
    )
    from .r3_response import analyze_control_responses
    from .r3_updates import run_bank_forks, validate_reuse_bank

    gate = validate_prerequisites(r0_dir, r1_run, supplement_dir)
    config = validate_config_against_gate(config, gate)
    r2_binding = validate_r2_gate(r2_dir, gate)
    if Path(data_root).resolve() != Path(config["data_root"]).resolve():
        raise ValueError("R3 data root differs from the certified dataset")
    _helper_config(config)
    if (
        config["R3"]["lambda_candidates"] != list(LAMBDAS)
        or config["R3"]["direct_resample_cold"] is not False
    ):
        raise ValueError("R3 cold fixed candidate or direct-sampling protocol changed")
    plan, data_binding = _load_plan(data_root, gate)
    fixture = _adapter_factory is not None
    if not fixture and not torch.cuda.is_available():
        raise RuntimeError("R3-cold requires an allocated CUDA device; no model requested")
    environment = (
        {"status": "CPU_FIXTURE_NOT_REAL_ENVIRONMENT"}
        if fixture
        else validate_runtime_environment(gate)
    )
    if not fixture and environment != r2_binding["environment"]:
        raise ValueError("R3 runtime environment differs from measured R2")
    source = _source()
    execution = "CPU_FAKE_ADAPTER_FIXTURE" if fixture else "REAL_CUDA_FORK"
    identity = {
        "phase": "R3-cold",
        "protocol_version": config["protocol_version"],
        "checkpoint_step": 0,
        "model_hash": canonical_hash(config["model"]),
        "config_hash": canonical_hash(config),
        "data_hash": canonical_hash(data_binding),
        "initial_adapter_hash": gate["certificate"]["initial_adapter_hash"],
        "gate_hash": canonical_hash(gate["binding"]),
        "r2_gate_hash": canonical_hash(r2_binding),
        "source_hash": canonical_hash(source),
        "plan_hash": plan["plan_hash"],
        "execution_kind": execution,
    }
    out = Path(out)
    with frozen_writer(out):
        if (out / "policy_contamination.json").exists():
            raise ValueError("R3 policy-contaminated evidence requires a fresh run")
        store = RunStore(out, identity, resume=resume)
        write_json(out / "gate_binding.json", {"r0_r1": gate["binding"], "r2": r2_binding})
        invocations = out / "invocations"
        invocations.mkdir(exist_ok=True)
        invocation = invocations / f"attempt_{len(list(invocations.iterdir())):04d}"
        invocation.mkdir(exist_ok=False)
        write_json(invocation / "identity.json", identity)
        write_json(
            out / "status.json",
            {
                "phase": "R3-cold",
                "status": "RUNNING",
                "execution_kind": execution,
                "details": {"completed_samples": len(store.records), "total_samples": 1152},
            },
        )
        write_json(
            out / "progress.json",
            {
                "state": "LOADING_MODEL",
                "invocation": invocation.name,
                "completed_samples": len(store.records),
                "total_samples": 1152,
            },
        )
        adapter = optimizer = origin = meter = profile = None
        before_base = None
        summaries, responses, validation = {}, {}, {}
        details = {"execution_kind": execution, "status": "FAIL", "state": "cold"}
        status = "FAIL"
        try:
            _seed_everything(17)
            adapter = (_adapter_factory or load_adapter)(
                "qwen35_9b",
                {
                    "id": config["model"]["id"],
                    "revision": config["model"]["revision"],
                    "expected_layers": 32,
                },
                image_token_limit=768,
            )
            if (
                adapter.model_id != config["model"]["id"]
                or adapter.revision != config["model"]["revision"]
            ):
                raise ValueError("R3 loaded model identity differs from certificate")
            before_base = _certificate_check(gate["certificate"], adapter, config)
            if not fixture:
                environment = validate_runtime_environment(gate, adapter.audit)
                if environment != r2_binding["environment"]:
                    raise ValueError("R3 loaded runtime environment differs from measured R2")
            options = config["optimizer"]
            optimizer = torch.optim.AdamW(
                [p for p in adapter.model.parameters() if p.requires_grad],
                lr=options["learning_rate"],
                betas=tuple(options["betas"]),
                eps=options["eps"],
                weight_decay=options["weight_decay"],
            )
            origin_metadata = {
                "checkpoint_step": 0,
                "state": "cold",
                "sampler": {"plan_hash": plan["plan_hash"], "position": 0},
                "scheduler": None,
                "grad_scaler": None,
            }
            origin = capture_state(adapter.model, optimizer, origin_metadata)
            if origin["optimizer"]["state"] or origin["scheduler"] is not None:
                raise ValueError("Cold R3 must begin with empty Adam and no scheduler")
            origin_identity = {**identity, "unit": "initial_origin"}
            origin_path = out / "origin.pt"
            if origin_path.exists():
                saved = load_checkpoint(origin_path, origin_identity)
                if state_hash(saved) != state_hash(origin):
                    raise ValueError("R3 fresh initialization/RNG differs from saved origin")
                origin = saved
            else:
                save_checkpoint(origin_path, origin, origin_identity)
            request_identity = {**identity, "origin_hash": state_hash(origin)}
            requests = build_requests(plan, request_identity)
            validate_sample_ledger(store.records, requests)
            write_json(
                out / "bank_manifest.json",
                {
                    "identity": identity,
                    "request_identity": request_identity,
                    "plan": plan,
                    "data_binding": data_binding,
                    "requests": requests,
                },
            )
            runtime = {
                "identity": identity,
                "config": config,
                "environment": environment,
                "source": source,
                "model_audit": adapter.audit,
                "origin_hash": state_hash(origin),
                "origin_file_sha256": file_hash(origin_path),
                "frozen_base_hash": before_base,
                "initial_adapter_hash": parameter_hash(adapter.model, trainable=True),
                "optimizer_initial_hash": state_hash(origin["optimizer"]),
                "gradient_scaler": None,
                "scheduler": None,
                "selected_probability_path": "uncached_prefix_recompute",
            }
            lock = out / "runtime_lock.json"
            if lock.exists():
                previous = _json(lock)
                for key in (
                    "identity",
                    "config",
                    "environment",
                    "origin_hash",
                    "origin_file_sha256",
                    "frozen_base_hash",
                    "initial_adapter_hash",
                ):
                    if previous.get(key) != runtime[key]:
                        raise ValueError(f"R3 runtime resume binding drift: {key}")
            else:
                write_json(lock, runtime)
            meter = _ExecutionMeter(adapter, optimizer)
            if not fixture and (
                len(meter.language_layer_names) != 32
                or not meter.top_hook_installed
                or meter.vision_hook_count != 1
            ):
                raise RuntimeError("R3 actual forward/backward instrumentation is incomplete")
            profile = _Profile(out, invocation, meter, fixture, config, identity)
            profile.write("SAMPLING", completed=len(store.records), total=1152)
            _collect_samples(
                adapter, optimizer, origin, plan, requests, store, data_root, meter, profile
            )
            validate_sample_ledger(store.records, requests)
            if len(store.records) != 1152:
                raise ValueError("R3 train/control rollout coverage is incomplete")
            if parameter_hash(adapter.model, trainable=True) != identity["initial_adapter_hash"]:
                raise ValueError("R3 sampling changed the initial adapter")
            score_banks = {}
            for index in plan["reuse_check_bank_indices"]:
                groups = _bank_groups(plan, store.records, index, adapter, data_root)
                unit_identity = {
                    **identity,
                    "unit": "reuse_validation",
                    "bank_index": index,
                    "origin_hash": state_hash(origin),
                    "sample_hash": canonical_hash(
                        [r["record_hash"] for group in groups for r in group]
                    ),
                }

                def validate(attempt, groups=groups, unit_identity=unit_identity, index=index):
                    try:
                        result = validate_reuse_bank(
                            adapter,
                            optimizer,
                            origin,
                            groups,
                            meter=_MeterPrefix(
                                meter, f"validation/bank_{index:02d}/{attempt.name}"
                            ),
                        )
                        tensor_binding = _save_tensor_payload(
                            attempt / "score_bank.pt", result["score_bank"], unit_identity
                        )
                        return {
                            **{k: v for k, v in result.items() if k != "score_bank"},
                            "score_bank_binding": tensor_binding,
                        }
                    finally:
                        _restore(adapter, optimizer, origin)
                        profile.write("VALIDATING_REUSE", bank_index=index)

                attempt, result, reused = run_atomic_unit(
                    out / "validation" / f"bank_{index:02d}", unit_identity, validate
                )
                validation[index] = {
                    **result,
                    "attempt": str(attempt),
                    "reused_completed_unit": reused,
                }
                score_banks[index] = _load_tensor_payload(
                    attempt / "score_bank.pt", unit_identity, result["score_bank_binding"]
                )
            authorized = all(result["adoptable"] is True for result in validation.values())
            write_json(
                out / "reuse_decision.json",
                {
                    "reuse_authorized": authorized,
                    "validation_bank_indices": plan["reuse_check_bank_indices"],
                    "validation": validation,
                    "fallback": None if authorized else "DIRECT_LOSS_GRADIENTS_FOR_ALL_BANKS",
                },
            )
            candidate_attempts = {}
            for index in range(12):
                groups = _bank_groups(plan, store.records, index, adapter, data_root)
                unit_identity = {
                    **identity,
                    "unit": "bank_forks",
                    "bank_index": index,
                    "origin_hash": state_hash(origin),
                    "reuse_authorized": authorized,
                    "sample_hash": canonical_hash(
                        [r["record_hash"] for group in groups for r in group]
                    ),
                }

                def fork(attempt, groups=groups, index=index, unit_identity=unit_identity):
                    try:
                        result = run_bank_forks(
                            adapter,
                            optimizer,
                            origin,
                            groups,
                            attempt / "engine",
                            unit_identity,
                            reuse_authorized=authorized,
                            score_bank=score_banks.get(index),
                            meter=_MeterPrefix(meter, f"forks/bank_{index:02d}/{attempt.name}"),
                        )
                        if (
                            result.get("status") != "PASS"
                            or result.get("passed") is not True
                            or result.get("candidate_optimizer_updates") != 5
                        ):
                            raise RuntimeError(
                                "R3 bank engine did not complete five isolated candidates"
                            )
                        if sorted(float(c["lambda"]) for c in result["candidates"]) != list(
                            LAMBDAS
                        ):
                            raise ValueError("R3 candidate lambda coverage differs from fixed grid")
                        return result
                    finally:
                        _restore(adapter, optimizer, origin)
                        profile.write("FORKING", bank_index=index)

                attempt, summary, reused = run_atomic_unit(
                    out / "forks" / f"bank_{index:02d}", unit_identity, fork
                )
                summaries[index] = summary
                candidate_attempts[index] = attempt
                profile.write(
                    "FORKING",
                    completed_banks=len(summaries),
                    total_banks=12,
                    reused_completed_unit=reused,
                )
            all_ids = [
                c["candidate_id"] for summary in summaries.values() for c in summary["candidates"]
            ]
            if len(all_ids) != len(set(all_ids)) or len(all_ids) != 60:
                raise ValueError("R3 candidate IDs must identify 60 distinct bank/lambda updates")
            proposal = [
                store.records[r["sample_key"]]
                for r in requests
                if r["bank_role"] == "control_proposal"
            ]
            for index in plan["response_banks"]:
                candidate_scores = {}
                for candidate in summaries[index]["candidates"]:
                    candidate_scores[candidate["candidate_id"]] = _score_candidate(
                        adapter,
                        optimizer,
                        origin,
                        candidate,
                        candidate_attempts[index],
                        proposal,
                        plan,
                        out,
                        identity,
                        data_root,
                        meter,
                        profile,
                    )
                baseline = next(
                    c["candidate_id"] for c in summaries[index]["candidates"] if c["lambda"] == 0
                )
                responses[index] = analyze_control_responses(
                    proposal,
                    candidate_scores,
                    baseline_key=baseline,
                    bank_index=index,
                    bootstrap_replicates=5000,
                    seed=20260909,
                )
                write_json(out / f"response_bank_{index:02d}.json", responses[index])
            if _source() != source:
                raise ValueError("R3 source changed during execution")
            for name, digest in data_binding["files"].items():
                if file_hash(Path(data_root) / name) != digest:
                    raise ValueError("R3 train/control data changed during execution")
            _verify_scene_images(
                [r["scene"] for r in [*plan["train_prompts"], *plan["control_prompts"]]], data_root
            )
            details.update(
                {
                    "raw_sample_count": len(store.records),
                    "train_rollouts": 384,
                    "control_proposal_rollouts": 768,
                    "distinct_candidate_optimizer_updates": 60,
                    "distinct_lambda_zero_replays": sum(
                        s.get("replay_optimizer_updates", 0) for s in summaries.values()
                    ),
                    "response_candidates": 10,
                    "control_candidate_sequences_scored": 7680,
                    "direct_resample_cold": False,
                    "warm_implemented": False,
                    "candidate_commit": False,
                    "reuse_authorized": authorized,
                }
            )
            status = "PASS"
        except Exception as exc:
            details["error"] = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            restoration = {}
            if adapter is not None and optimizer is not None and origin is not None:
                try:
                    _restore(adapter, optimizer, origin)
                    restoration = {
                        "full_origin_state_and_rng": True,
                        "initial_adapter": parameter_hash(adapter.model, trainable=True)
                        == identity["initial_adapter_hash"],
                        "empty_adam": not optimizer.state_dict()["state"],
                        "frozen_base": parameter_hash(adapter.model, trainable=False)
                        == before_base,
                    }
                    if not all(restoration.values()):
                        write_json(
                            out / "policy_contamination.json",
                            {
                                "phase": "final_restoration",
                                "checks": restoration,
                            },
                        )
                        raise RuntimeError("R3 final scratch restoration audit failed")
                except Exception as exc:
                    status = "FAIL"
                    details["restoration_error"] = {"type": type(exc).__name__, "message": str(exc)}
            details["scratch_restoration"] = restoration
            details["status"] = status
            if profile is not None:
                profile.write(status, completed_samples=len(store.records))
            if meter is not None:
                meter.close()
            write_json(invocation / "completion.json", {"status": status, "details": details})
        histories = []
        for prior in sorted(invocations.iterdir()):
            report = prior / "runtime_profile.json"
            histories.append(
                {
                    "invocation": prior.name,
                    "completed": (prior / "completion.json").exists(),
                    "observed": _json(report) if report.exists() else {},
                    "unrecorded_work_possible": not (prior / "completion.json").exists(),
                }
            )
        cost = {
            "invocations": histories,
            "backward_calls_observed_at_least": sum(
                r["observed"].get("backward_calls_observed", 0) for r in histories
            ),
            "optimizer_steps_observed_at_least": sum(
                r["observed"].get("optimizer_step_calls_observed", 0) for r in histories
            ),
            "incomplete_invocations_with_unknown_extra_cost": sum(
                r["unrecorded_work_possible"] for r in histories
            ),
            "counter_scope": (
                "actual observed calls; unfinished intervals may include additional unrecorded work"
            ),
        }
        write_json(out / "runtime_profile.json", cost)
        details["runtime_counts"] = {k: v for k, v in cost.items() if k != "invocations"}
        if status == "PASS":
            _write_reports(out, summaries, responses, validation, details)
        else:
            (out / "report_zh.md").write_text(
                "# R3-cold 执行未通过\n\n"
                + json.dumps(details, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
        return _final_status(out, status, execution, details)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--r0-dir", type=Path, required=True)
    parser.add_argument("--r1-run", type=Path, required=True)
    parser.add_argument("--supplement-dir", type=Path, required=True)
    parser.add_argument("--r2-dir", type=Path, required=True)
    parser.add_argument("--state", choices=("cold", "warm"), default="cold")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    from .next_stage_runtime import validate_config_against_gate, validate_prerequisites

    gate = validate_prerequisites(args.r0_dir, args.r1_run, args.supplement_dir)
    config = gate["config"]
    if args.config:
        import yaml

        config = validate_config_against_gate(yaml.safe_load(args.config.read_text()), gate)
    result = run_r3(
        config,
        args.data_root or config["data_root"],
        args.out,
        args.r0_dir,
        args.r1_run,
        args.supplement_dir,
        args.r2_dir,
        state=args.state,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "execution_kind": result["execution_kind"],
                "out": str(args.out),
            }
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
