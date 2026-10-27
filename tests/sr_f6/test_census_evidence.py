"""Synthetic persistence fixtures, never actual GPU or scientific evidence."""

import copy
import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from sr_f6 import census
from sr_f6 import census_evidence as evidence
from sr_f6.common import digest, file_hash
from sr_f12.training import probability_diagnostics
from srf6_ref.core import FAMILIES
from srf6_ref.design import make_root
from srf6_ref.measurement import full_truth_response


def save(root, relative, value, *, lines=False):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row) + "\n" for row in value) if lines else json.dumps(value)
    path.write_text(payload)
    return path


def worker(task_id, count, frozen):
    return dict(
        task_id=task_id,
        status="COMPLETE",
        phase="PARENT_CENSUS",
        groups=count,
        frozen_spec_sha256=digest(frozen),
        optimizer_updates=0,
        actual_generated_responses_this_process=count * 8,
    )


def alignment(spec, records, tensor_hash):
    return dict(
        identity=dict(
            frozen_spec_sha256=spec["frozen_spec_sha256"],
            group_spec_sha256=digest(spec),
            raw_trajectories_sha256=digest(
                [
                    {k: row[k] for k in ("tokens", "sampler_logprobs", "prompt_tensor_hash")}
                    for row in records
                ]
            ),
        ),
        status="PASS",
        execution_origin="SERVER_ACTUAL",
        phase="PARENT_CENSUS",
        parent=spec["parent"],
        group_id=spec["group_id"],
        parameter_tensor_sha256=tensor_hash,
        original_registered_responses=8,
        checked_sequences=8,
        new_generated_responses=0,
        optimizer_updates=0,
        traces=[
            dict(
                row_id=row["row_id"],
                token_ids=row["tokens"],
                sampled_logprobs=row["sampler_logprobs"],
                current_logprobs=row["sampler_logprobs"],
            )
            for row in records
        ],
        diagnostics=probability_diagnostics(
            [torch.zeros(2)] * 8, [torch.zeros(2, dtype=torch.bool)] * 8
        ),
    )


def test_worker_partition_requires_every_shard_without_overlap():
    public, frozen = [{"qid": str(i)} for i in range(4)], {"synthetic": True}
    workers = {
        f"CENSUS/workers/CENSUS-{p}-{s}-of-2.json": worker(f"CENSUS-{p}-{s}-of-2", 2, frozen)
        for p in census.PARENTS
        for s in range(2)
    }
    assigned = evidence._worker_partition(workers, public, frozen)
    assert len(assigned) == 12
    assert assigned["b0", "1"] == "CENSUS-b0-1-of-2"
    missing = dict(workers)
    missing.pop("CENSUS/workers/CENSUS-b0-1-of-2.json")
    with pytest.raises(PermissionError, match="Incomplete"):
        evidence._worker_partition(missing, public, frozen)
    mixed = dict(workers)
    mixed["CENSUS/workers/CENSUS-b0-0-of-1.json"] = worker("CENSUS-b0-0-of-1", 4, frozen)
    with pytest.raises(PermissionError, match=r"Mixed|Overlapping"):
        evidence._worker_partition(mixed, public, frozen)


@pytest.mark.parametrize(
    "key,value",
    [
        ("groups", 3),
        ("optimizer_updates", 1),
        ("actual_generated_responses_this_process", 9),
        ("status", "RUNNING"),
    ],
)
def test_worker_counts_are_not_inferred_from_a_complete_label(key, value):
    public, frozen = [{"qid": "one"}], {"synthetic": True}
    workers = {
        f"CENSUS/workers/CENSUS-{p}-0-of-1.json": worker(f"CENSUS-{p}-0-of-1", 1, frozen)
        for p in census.PARENTS
    }
    workers["CENSUS/workers/CENSUS-b0-0-of-1.json"][key] = value
    with pytest.raises(PermissionError):
        evidence._worker_partition(workers, public, frozen)


@pytest.fixture
def replay():
    records = [
        dict(
            row_id=f"b0/q/{i}",
            tokens=[1, 2],
            sampler_logprobs=[-1.0, -2.0],
            prompt_tensor_hash="c" * 64,
        )
        for i in range(8)
    ]
    spec = dict(parent="b0", group_id="b0/q", frozen_spec_sha256="d" * 64)
    return spec, records, alignment(spec, records, "a" * 64)


def test_replay_checks_raw_tokens_not_only_pass_and_aggregate(replay):
    spec, records, report = replay
    evidence._alignment(report, spec, records, "a" * 64)
    report["traces"][0]["token_ids"] = [3, 4]
    with pytest.raises(PermissionError, match="original sampled"):
        evidence._alignment(report, spec, records, "a" * 64)


@pytest.mark.parametrize("change", ["policy", "raw_digest", "vector", "summary", "nonfinite"])
def test_replay_mutations_cannot_retain_pass(replay, change):
    spec, records, report = replay
    if change == "policy":
        report["parameter_tensor_sha256"] = "f" * 64
    elif change == "raw_digest":
        report["identity"]["raw_trajectories_sha256"] = "f" * 64
    elif change == "vector":
        report["traces"][0]["current_logprobs"] = [-2.0, -2.0]
    elif change == "summary":
        report["diagnostics"]["token_count"] = 1
    else:
        report["traces"][0]["current_logprobs"] = [float("nan"), -2.0]
    with pytest.raises(PermissionError):
        evidence._alignment(report, spec, records, "a" * 64)


def test_reader_rejects_symlinks_and_detects_midread_mutation(tmp_path, monkeypatch):
    original = save(tmp_path, "original.json", {"synthetic": True})
    (tmp_path / "link.json").symlink_to(original)
    reader = evidence._Reader(tmp_path)
    with pytest.raises(PermissionError, match="Symlink"):
        reader.read("link.json")
    before = Path.read_bytes

    def changed(path):
        data = before(path)
        path.write_bytes(data + b" ")
        return data

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(PermissionError, match="changed while"):
        reader.read("original.json")


def build_complete_fixture(root):
    config = json.loads((Path(__file__).parents[2] / "configs/sr_f6.json").read_text())
    save(root, "CONFIG.json", config)
    save(root, "DATA/MANIFEST.json", {"synthetic": True})
    proto = make_root("DEV_CAL", "BINDING", 0, config["data"]["seeds"]["DEV_CAL"])["tasks"]["P0"][0]
    tasks, public = [], []
    for family in FAMILIES:
        for index in range(8):
            rid = f"DEV_CAL_{family}_{index:04d}"
            for factor in census.FACTORS:
                for variant in range(4):
                    task = dict(
                        proto,
                        family=family,
                        root_id=rid,
                        factor=factor,
                        variant=variant,
                        qid=f"{rid}_{factor}_v{variant}",
                    )
                    item = {
                        k: task[k]
                        for k in ("qid", "root_id", "family", "factor", "variant", "split")
                    }
                    item.update(
                        protocol="structured_v4",
                        model_input=dict(
                            prompt="SYNTHETIC TEST ONLY",
                            image_file="fixture.png",
                            image_sha256="e" * 64,
                        ),
                    )
                    tasks.append(task)
                    public.append(item)
    save(root, "DATA/DEV_CAL/MODEL_INPUTS.jsonl", public, lines=True)
    save(root, "DATA/DEV_CAL/TASKS_PRIVATE.jsonl", tasks, lines=True)
    public, tasks = census.census_inputs(root)
    frozen = census.freeze_spec(root, {p: "a" * 64 for p in census.PARENTS})
    context = dict(source_manifest={"sha256": "b" * 64}, data_manifest={"sha256": "c" * 64})
    truth = full_truth_response(proto["world"], proto["query"])
    for parent in census.PARENTS:
        for shard in range(2):
            task_id = f"CENSUS-{parent}-{shard}-of-2"
            runtime = dict(
                run_id=task_id,
                run_root=str(root),
                parent_id=parent,
                step=0,
                reference_parent=parent,
                parent_content_hash="a" * 64,
                source_sha256="b" * 64,
                source_hash="b" * 64,
                data_hash="c" * 64,
                fresh_optimizer=False,
                dtype="float32",
                probability_dtype="float32",
                sr_f6_protocol_version=census.PROTOCOL_VERSION,
                policy_hash="d" * 64,
                reference_hash="d" * 64,
            )
            save(root, f"runtime_identity/{task_id}.json", runtime)
            selected = public[shard::2]
            accounting = []
            for item in selected:
                qid = item["qid"]
                spec = census.group_spec(config, parent, item, tasks[qid], frozen)
                raw = [
                    dict(
                        run_id=task_id,
                        run_root=str(root),
                        policy_hash="d" * 64,
                        policy_tensor_hash="d" * 64,
                        parent_content_hash="a" * 64,
                        source_sha256="b" * 64,
                        source_hash="b" * 64,
                        data_hash="c" * 64,
                        step=0,
                        policy_step=0,
                        protocol="structured_v4",
                        sr_f6_protocol_version=census.PROTOCOL_VERSION,
                        sampler_logprob_source="actual_generate_raw_logits_selected_token",
                        adapter_identity=dict(run_id=task_id, parent=parent, step=0),
                        prompt_tensor_hash="e" * 64,
                        native_prompt_inputs={"input_ids": [[1]]},
                        seed=seed,
                        generation_status="COMPLETE",
                        technical_validation_errors=[],
                        tokens=[1, 2],
                        raw_tokens=[1, 2],
                        completion_token_count=2,
                        max_new_tokens=1536,
                        group_size=8,
                        group_row_index=draw,
                        group_seed=spec["seeds"][0],
                        truncated=False,
                        balanced_token_count=2,
                        boundary_token_count=2,
                        sampler_logprobs=[-1.0, -2.0],
                        model_input=item["model_input"],
                        full_text=truth,
                        finish_reason="balanced",
                    )
                    for draw, seed in enumerate(spec["seeds"])
                ]
                # Keep a valid unfinished EOS example: annotation normalizes its reason.
                if item is selected[-1]:
                    raw[-1]["finish_reason"] = "eos"
                    raw[-1]["balanced_token_count"] = None
                    raw[-1]["boundary_token_count"] = None
                accounting.append(
                    dict(
                        task_id=task_id,
                        parent=parent,
                        kind="completion_attempts",
                        count=8,
                        metadata=dict(group_seed=spec["seeds"][0], protocol="structured"),
                    )
                )
                accounting.extend(
                    dict(
                        task_id=task_id,
                        parent=parent,
                        kind="generated_tokens",
                        count=2,
                        metadata=dict(group_seed=spec["seeds"][0], row_index=draw),
                    )
                    for draw in range(8)
                )
                records = census.annotate_group(raw, tasks[qid], spec)
                save(
                    root,
                    f"CENSUS/attempts/{parent}/{qid}.json",
                    dict(status="STARTED", task_id=task_id, spec=spec),
                )
                save(root, f"CENSUS/raw/{parent}/{qid}.jsonl", raw, lines=True)
                save(
                    root,
                    f"CENSUS/groups/{parent}/{qid}.json",
                    dict(
                        status="COMPLETE",
                        spec=spec,
                        records=records,
                        credit=census.counterfactual(records, spec),
                    ),
                )
                if item is selected[0]:
                    replay_path = save(
                        root, f"CENSUS/alignment/{task_id}.json", alignment(spec, records, "d" * 64)
                    )
                    accounting.extend(
                        dict(
                            task_id=task_id,
                            parent=parent,
                            kind="extra_forward_sequences",
                            count=1,
                            metadata=dict(purpose="census_token_alignment", completion_tokens=2),
                        )
                        for _ in range(8)
                    )
            report = worker(task_id, len(selected), frozen)
            report["generation_alignment_sha256"] = file_hash(replay_path)
            save(root, f"CENSUS/workers/{task_id}.json", report)
            save(
                root,
                f"accounting/{task_id}.jsonl",
                accounting,
                lines=True,
            )
    return context, public


def test_full_raw_matrix_and_original_eos_are_retained_then_mutations_rejected(
    tmp_path, monkeypatch
):
    context, public = build_complete_fixture(tmp_path)
    calls = []

    def verified_origin(root, *, deployment):
        calls.append((root, deployment))
        return dict(identity=context, identity_sha256="f" * 64)

    monkeypatch.setattr(evidence.phases, "verify_origin", verified_origin)
    result = evidence.validate_saved_census(tmp_path, origin_deployment="SYNTHETIC_OLD_SOURCE")
    assert len(result["records"]) == result["actual_recorded_calls"] == 9216
    assert result["training_authorized"] is False
    assert result["model_calls_by_this_validator"] == 0
    assert result["input_producer"]["identity"] == context
    assert calls == [(tmp_path, "SYNTHETIC_OLD_SOURCE")]
    eos_id = f"b0/{public[-1]['qid']}/7"
    assert result["raw_by_row"][eos_id]["finish_reason"] == "eos"
    assert next(r for r in result["records"] if r["row_id"] == eos_id)["finish_reason"] == "length"
    assert len(result["inputs"]) == 3484  # 3*1152 payloads + 6*4 workers + 4 fixed inputs.
    qid = public[0]["qid"]
    raw_path = tmp_path / f"CENSUS/raw/b0/{qid}.jsonl"
    original = raw_path.read_bytes()
    raw = census.rows(raw_path)
    raw[0]["full_text"] += " altered raw"
    save(tmp_path, str(raw_path.relative_to(tmp_path)), raw, lines=True)
    with pytest.raises(PermissionError, match="independently persisted original raw"):
        evidence.validate_saved_census(tmp_path, origin_deployment="SYNTHETIC_OLD_SOURCE")
    raw_path.write_bytes(original)
    # Coherent group relabeling alone passed the old summary path; new raw comparison rejects it.
    group_path = tmp_path / f"CENSUS/groups/b0/{qid}.json"
    group = json.loads(group_path.read_text())
    altered = copy.deepcopy(group)
    altered["records"][0]["tokens"] = [4, 5]
    altered["records"][0]["raw_token_ids"] = [4, 5]
    census.validate_group(altered, group["spec"], census.census_inputs(tmp_path)[1][qid])
    save(tmp_path, str(group_path.relative_to(tmp_path)), altered)
    with pytest.raises(PermissionError, match="independently persisted original raw"):
        evidence.validate_saved_census(tmp_path, origin_deployment="SYNTHETIC_OLD_SOURCE")
    save(tmp_path, str(group_path.relative_to(tmp_path)), group)
    raw = census.rows(raw_path)
    raw[0].update(
        finish_reason="length", truncated=True, balanced_token_count=None, boundary_token_count=None
    )
    # A coherent reannotation cannot make a two-token cutoff a native 1536-token stop.
    changed_records = census.annotate_group(
        raw, census.census_inputs(tmp_path)[1][qid], group["spec"]
    )
    save(tmp_path, str(raw_path.relative_to(tmp_path)), raw, lines=True)
    save(
        tmp_path,
        str(group_path.relative_to(tmp_path)),
        dict(
            group,
            records=changed_records,
            credit=census.counterfactual(changed_records, group["spec"]),
        ),
    )
    with pytest.raises(PermissionError, match="termination metadata"):
        evidence.validate_saved_census(tmp_path, origin_deployment="SYNTHETIC_OLD_SOURCE")


@pytest.mark.parametrize("mutation", ["seed", "index", "extra_forward", "purpose", "kind"])
def test_accounting_same_totals_cannot_hide_wrong_native_events(mutation):
    rows = [
        dict(
            task_id="worker",
            parent="b0",
            kind="completion_attempts",
            count=8,
            metadata=dict(group_seed=42, protocol="structured"),
        ),
        dict(
            task_id="worker",
            parent="b0",
            kind="generated_tokens",
            count=3,
            metadata=dict(group_seed=42, row_index=0),
        ),
        dict(
            task_id="worker",
            parent="b0",
            kind="extra_forward_sequences",
            count=1,
            metadata=dict(purpose="census_token_alignment", completion_tokens=3),
        ),
    ]
    # Duplicate events intentionally retain their multiplicity for paired factor seeds.
    expected = Counter(
        {
            ("completion_attempts", 42, "structured", 8): 2,
            ("generated_tokens", 42, 0, 3): 2,
            ("extra_forward_sequences", "census_token_alignment", 3, 1): 2,
        }
    )
    rows += copy.deepcopy(rows)
    evidence._accounting(rows, "worker", "b0", expected)
    if mutation == "seed":
        rows[0]["metadata"]["group_seed"] = -12345
    elif mutation == "index":
        rows[1]["metadata"]["row_index"] = 7
    elif mutation == "extra_forward":
        rows[2]["count"] = 1000000
    elif mutation == "purpose":
        rows[2]["metadata"]["purpose"] = "unregistered_sampling_or_update"
    else:
        rows[2]["kind"] = "unregistered_update"
    with pytest.raises(PermissionError):
        evidence._accounting(rows, "worker", "b0", expected)


def test_missing_original_raw_is_never_reconstructed(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence.phases, "verify_origin", lambda *a, **k: {"identity": {}})
    save(tmp_path, "CONFIG.json", {})
    monkeypatch.setattr(census, "census_inputs", lambda root: ([], {}))
    with pytest.raises(PermissionError, match="missing"):
        evidence.validate_saved_census(tmp_path, origin_deployment="SYNTHETIC_OLD_SOURCE")
