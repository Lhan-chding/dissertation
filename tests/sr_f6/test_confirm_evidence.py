"""Synthetic persisted CPU controls, never real confirmation or GPU evidence."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mm_core.training import state_hash
from sr_f6 import confirm_census as census
from sr_f6 import confirm_evidence as evidence
from sr_f6 import confirm_registration as registration
from sr_f6.common import digest, file_hash
from sr_f6.runtime import PROTOCOL_VERSION, generation_recipe
from sr_f6.source_binding import inventory_file
from srf6_ref.core import FAMILIES
from srf6_ref.design import make_root
from srf6_ref.measurement import full_truth_response


def save(root, relative, value, *, lines=False):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in value)
        if lines
        else json.dumps(value, allow_nan=False)
    )
    return dict(path=relative, sha256=file_hash(path))


class Tokenizer:
    pad_token_id = 0

    def __init__(self, text):
        self.text = text

    def decode(self, tokens, **kwargs):
        return "".join({0: "", 1: self.text[1:-1], 2: "}", 3: ""}[t] for t in tokens)


def replay(spec, records):
    traces = [
        dict(
            row_id=r["row_id"],
            token_ids=r["tokens"],
            sampled_logprobs=r["sampler_logprobs"],
            current_logprobs=r["sampler_logprobs"],
        )
        for r in records
    ]
    return dict(
        status="PASS",
        identity=census._alignment_identity(spec, records),
        phase=census.PHASE,
        parent=spec["parent"],
        group_id=spec["group_id"],
        execution_origin="SERVER_ACTUAL",
        parameter_tensor_sha256=records[0]["policy_tensor_hash"],
        checked_sequences=8,
        original_registered_responses=8,
        new_generated_responses=0,
        optimizer_updates=0,
        traces=traces,
        diagnostics=census._diagnostics(traces, records),
    )


@pytest.fixture(scope="module")
def complete(tmp_path_factory):
    """Real full-matrix reconstruction, with ONLY the upstream provider synthesized."""
    project = tmp_path_factory.mktemp("confirm-evidence-synthetic")
    root = project / "confirm"
    root.mkdir()
    config = json.loads((Path(__file__).parents[2] / "configs/sr_f6.json").read_text())
    proto = make_root("DEV_CAL", "BINDING", 0, config["data"]["seeds"]["DEV_CAL"])["tasks"]["P0"][0]
    truth = full_truth_response(proto["world"], proto["query"])
    native = dict(
        input_ids=[[1, 2]],
        attention_mask=[[1, 1]],
        mm_token_type_ids=[[0, 0]],
        image_grid_thw=[[1, 2, 2]],
    )
    route = dict(
        input_tensor_hash="e" * 64,
        processor_hash="f" * 64,
        chat_template_hash="a" * 64,
        chat_text_sha256="b" * 64,
        native_input_tensor_hashes={k: state_hash(torch.tensor(v)) for k, v in native.items()},
    )
    public, private = [], {}
    for family in FAMILIES:
        for index in range(8):
            rid = f"{census.SPLIT}_{family}_{index:04d}"
            for factor in ("P0", "P1"):
                for variant in range(4):
                    qid = f"{rid}_{factor}_v{variant}"
                    task = dict(
                        proto,
                        qid=qid,
                        root_id=rid,
                        split=census.SPLIT,
                        family=family,
                        factor=factor,
                        variant=variant,
                    )
                    private[qid] = task
                    public.append(
                        dict(
                            **{
                                k: task[k]
                                for k in ("qid", "root_id", "split", "family", "factor", "variant")
                            },
                            protocol="structured_v4",
                            prompt_version=evidence.PROMPT_VERSION,
                            prompt_sha256=hashlib.sha256(b"SYNTHETIC ONLY").hexdigest(),
                            model_input=dict(
                                prompt="SYNTHETIC ONLY",
                                image_file="DATA/image.bin",
                                image_sha256="e" * 64,
                            ),
                        )
                    )
    public.sort(key=lambda r: r["qid"])
    parent_hashes = {p: "a" * 64 for p in census.PARENTS}
    plan = registration.make_plan(public, private, parent_hashes)
    refs = {
        "config": save(root, "CONFIG.json", config),
        "spec": save(root, "CONFIRM_SPEC_FROZEN.json", plan),
    }
    dep = project / "synthetic-dependency.bin"
    dep.write_bytes(b"explicit synthetic metadata fixture; no model")
    bound = inventory_file(dep, project)
    manifest = dict(revision="c" * 40, files=[bound])
    refs["source_manifest"] = save(root, "SOURCE_MANIFEST.json", manifest)
    audit = dict(historical_world_exclusion=dict(bound_files=[bound]))
    save(root, "DATA/AUDIT.json", audit)
    save(root, "DATA/MODEL_INPUTS.jsonl", public, lines=True)
    save(root, "DATA/TASKS_PRIVATE.jsonl", list(private.values()), lines=True)
    members = [
        dict(inventory_file(path, root), path=str(path.relative_to(root)))
        for path in sorted((root / "DATA").iterdir())
    ]
    refs["data_manifest"] = save(root, "DATA/MANIFEST.json", dict(files=members))
    record = dict(
        **refs,
        phase=census.PHASE,
        revision="c" * 40,
        source_manifest_sha256=refs["source_manifest"]["sha256"],
        data_sha256=refs["data_manifest"]["sha256"],
        parent_content_hashes=parent_hashes,
        upstream_files=[bound],
        predecessor_screen=dict(
            phase="INTERFACE_SCREEN_V3",
            deployment=str(project / "old-5ca-screen"),
            revision="5ca3fe9",
        ),
    )
    pin = save(root, "REGISTRATION.json", record)["sha256"]
    review_ref = save(root, "PREFLIGHT/observations.json", dict(synthetic=True))
    logpath = root / "PREFLIGHT/native.log"
    logpath.write_text("CPU SYNTHETIC TEST ONLY; not scientific evidence\n")
    native_ref = save(
        root,
        "PREFLIGHT/NATIVE_REGRESSION.json",
        dict(synthetic=True, log=dict(path="PREFLIGHT/native.log", sha256=file_hash(logpath))),
    )
    processor = dict(
        inputs=[dict(qid=r["qid"], routing=route) for r in public],
        images=[],
        identity=dict(
            processor_hash=route["processor_hash"], chat_template_hash=route["chat_template_hash"]
        ),
        verified_dependencies=[bound],
        native_regression=native_ref,
    )
    processor_ref = save(root, "PREFLIGHT/PROCESSOR.json", processor)
    visual = dict(review_evidence=review_ref, synthetic=True)
    visual_ref = save(root, "PREFLIGHT/VISUAL.json", visual)
    seal = dict(processor=processor_ref, visual=visual_ref)
    save(root, "PREFLIGHT/SEAL.json", seal)
    context = dict(
        root=root,
        input_root=root,
        source_root=project / "original",
        registration=record,
        registration_sha256=pin,
        spec=plan,
        plan=plan,
        spec_object_sha256=digest(plan),
        public=public,
        private=private,
        config=config,
        source_manifest=manifest,
        deployment=Path(evidence.__file__).resolve().parents[2],
        source_binding=dict(
            model_files=[bound], parents={p: dict(files=[bound]) for p in census.PARENTS}
        ),
        preflight=dict(processor, visual=visual, seal=seal),
    )
    for parent_index, parent in enumerate(census.PARENTS):
        job = str(100 + parent_index)
        deployment = context["deployment"]
        save(
            root,
            f"SUBMISSIONS/{parent}.attempt.json",
            dict(
                parent=parent,
                registration_sha256=pin,
                status="SUBMISSION_INTENT_NOT_COMPLETION",
                argv=[
                    "sbatch",
                    "--parsable",
                    "--no-requeue",
                    f"--job-name=LH-SRF6-Confirm-{parent}",
                    f"--chdir={root}",
                    f"--output={root / 'slurm'}/confirm-{parent}-%j.log",
                    f"--export=ALL,SRF6_CODE={deployment},SRF6_ROOT={root},SRF6_PARENT={parent}",
                    str(deployment / "scripts/sr_f6/confirm.sbatch"),
                ],
            ),
        )
        save(
            root,
            f"SUBMISSIONS/{parent}.result.json",
            dict(
                parent=parent,
                status="SUBMITTED",
                job_id=job,
                returncode=0,
                stdout=job + "\n",
                stderr="",
            ),
        )
        for shard in (0, 1):
            task_id = f"LH-SRF6-Confirm-{job}-{parent}-{shard}"
            execution = dict(
                status="STARTED",
                phase=census.PHASE,
                task_id=task_id,
                parent=parent,
                shard=shard,
                shards=2,
                registration_sha256=pin,
            )
            save(root, f"CONFIRM/executions/{task_id}.json", execution)
            save(root, f"CONFIRM/execution_slots/{parent}-{shard}.json", execution)
            runtime = dict(
                run_id=task_id,
                task_id=task_id,
                parent_id=parent,
                phase=census.PHASE,
                confirmation_version=evidence.VERSION,
                run_root=str(root),
                registration_root=str(root),
                source_root=str(context["source_root"]),
                registration_sha256=pin,
                source_sha256=record["source_manifest_sha256"],
                data_hash=record["data_sha256"],
                parent_content_hash=parent_hashes[parent],
                step=0,
                optimizer_updates=0,
                fresh_optimizer=False,
                optimizer=None,
                scheduler=None,
                trainable_parameters=[],
                dtype="float32",
                probability_dtype="float32",
                input_float_dtype="float32",
                forward_parameter_dtypes=["float32"],
                forward_float_buffer_dtypes=[],
                prompt_version=evidence.PROMPT_VERSION,
                sr_f6_protocol_version=PROTOCOL_VERSION,
                execution_mode="READONLY_SCHEMA_CONFIRM",
                input_plan_sha256=digest(plan),
                worker_id=f"{parent}-s{shard}-of-2",
                shard=shard,
                first_qid=public[shard]["qid"],
                source_binding_status="REGISTERED_INDEPENDENT_CONFIRMATION_ONLY",
                policy_hash="d" * 64,
                hardware=dict(cuda_visible_device_count=1, cuda_uuid=f"SYNTHETIC-{parent}-{shard}"),
            )
            save(root, f"CONFIRM/runtime_identity/{task_id}.json", runtime)
            ledger = []

            def event(qid, kind, count, ledger=ledger, task_id=task_id, parent=parent, **metadata):
                ledger.append(
                    dict(
                        task_id=task_id,
                        parent=parent,
                        kind=kind,
                        count=count,
                        metadata=dict(
                            metadata,
                            qid=qid,
                            phase=census.PHASE,
                            confirmation_version=evidence.VERSION,
                            registration_sha256=pin,
                        ),
                    )
                )

            for i, item in enumerate(public[shard::2]):
                qid = item["qid"]
                spec = census.group_spec(context, parent, item)
                raw = [
                    dict(
                        phase=census.PHASE,
                        registration_sha256=pin,
                        source_sha256=runtime["source_sha256"],
                        source_hash=runtime["source_sha256"],
                        data_hash=runtime["data_hash"],
                        parent_content_hash=parent_hashes[parent],
                        policy_hash="d" * 64,
                        policy_tensor_hash="d" * 64,
                        run_id=task_id,
                        task_id=task_id,
                        run_root=str(root),
                        root=str(root),
                        confirmation_version=evidence.VERSION,
                        protocol_revision="v2",
                        backend_protocol="structured",
                        generation_status="COMPLETE",
                        technical_validation_errors=[],
                        execution_mode="READONLY_SCHEMA_CONFIRM",
                        optimizer_updates=0,
                        step=0,
                        policy_step=0,
                        protocol="structured_v4",
                        prompt_version=evidence.PROMPT_VERSION,
                        sr_f6_protocol_version=PROTOCOL_VERSION,
                        seed_semantics="independent_per_draw_torch_generator",
                        group_size=8,
                        group_row_index=draw,
                        group_seed=spec["seeds"][0],
                        seed=seed,
                        tokens=[1, 2],
                        raw_tokens=[1, 2],
                        sampler_logprobs=[-1.0, -2.0],
                        completion_token_count=2,
                        max_new_tokens=1536,
                        sampler_logprob_source="actual_generate_raw_logits_selected_token",
                        finish_reason="balanced",
                        truncated=False,
                        balanced_token_count=2,
                        boundary_token_count=2,
                        sampling_parameters=generation_recipe(),
                        model_input=item["model_input"],
                        prompt_sha256=item["prompt_sha256"],
                        prompt_tensor_hash=route["input_tensor_hash"],
                        input_routing=route,
                        native_prompt_inputs=native,
                        adapter_identity=dict(parent=parent, step=0, run_id=task_id),
                        full_text=truth,
                        generated_text_with_special_tokens=truth[1:],
                        backend_sequence_tokens=[1, 2],
                        backend_padding_count=0,
                    )
                    for draw, seed in enumerate(spec["seeds"])
                ]
                rows = census.annotate(raw, spec, private[qid], preflight_route=route)
                stem = f"{parent}/{qid}"
                save(
                    root,
                    f"CONFIRM/attempts/{stem}.json",
                    dict(status="STARTED", task_id=task_id, spec=spec),
                )
                save(root, f"CONFIRM/raw/{stem}.jsonl", raw, lines=True)
                save(
                    root,
                    f"CONFIRM/groups/{stem}.json",
                    dict(
                        status="COMPLETE",
                        spec=spec,
                        records=rows,
                        credit=census.counterfactual(rows, spec),
                    ),
                )
                event(
                    qid,
                    "completion_attempts",
                    8,
                    group_seed=spec["seeds"][0],
                    protocol="structured",
                )
                for draw in range(8):
                    event(qid, "generated_tokens", 2, group_seed=spec["seeds"][0], row_index=draw)
                if i == 0:
                    alignment = save(root, f"CONFIRM/alignment/{task_id}.json", replay(spec, rows))
                    for _ in range(8):
                        event(
                            qid,
                            "extra_forward_sequences",
                            1,
                            purpose="confirm_token_alignment",
                            completion_tokens=2,
                        )
            save(root, f"CONFIRM/accounting/{task_id}.jsonl", ledger, lines=True)
            save(
                root,
                f"CONFIRM/workers/{task_id}.json",
                dict(
                    status="COMPLETE",
                    phase=census.PHASE,
                    split=census.SPLIT,
                    task_id=task_id,
                    parent=parent,
                    shard=shard,
                    shards=2,
                    groups=192,
                    responses=1536,
                    generated_this_process=1536,
                    actual_generated_responses_this_process=1536,
                    recovered_raw_rows=0,
                    reused_rows=0,
                    optimizer_updates=0,
                    registration_sha256=pin,
                    frozen_spec_sha256=digest(plan),
                    generation_alignment_sha256=alignment["sha256"],
                    training_allowed=False,
                    gate_status="NOT_SIGNED",
                ),
            )
    return root, context, Tokenizer(truth)


@pytest.fixture
def bound(complete, monkeypatch):
    root, context, tokenizer = complete
    monkeypatch.setattr(registration, "PROJECT", root.parent)
    calls = []

    def verify(path, *, require_preflight):
        assert Path(path) == root and require_preflight is True
        calls.append(1)
        return context

    monkeypatch.setattr(registration, "verify_registered", verify)
    monkeypatch.setattr(evidence, "_tokenizer", lambda ctx: (tokenizer, [3]))
    return SimpleNamespace(root=root, context=context, tokenizer=tokenizer, calls=calls)


def test_complete_raw_reconstruction_binds_full_9216_matrix_without_permission(bound, monkeypatch):
    from sr_f6 import source_binding

    def no_payload_hash(_):
        raise AssertionError("unchanged registered dependency must use content-bound metadata")

    monkeypatch.setattr(source_binding, "file_sha256", no_payload_hash)
    result = evidence.validate_saved_confirm(bound.root)
    assert result["status"] == "VERIFIED_STORED_CONFIRMATION_NOT_GATE_SIGNED"
    assert len(result["records"]) == len(result["raw_by_row"]) == result["source_rows"] == 9216
    assert result["saved_token_validation"]["checked_responses"] == 9216
    assert result["accounting"] == dict(
        completion_attempts=9216, generated_tokens=18432, extra_forward_sequences=48
    )
    assert result["training_authorized"] is result["evaluation_authorized"] is False
    assert result["gate_status"] == "NOT_SIGNED"
    assert result["consumer_source"]["deployment"] != result["predecessor_screen"]["deployment"]
    assert (
        result["model_calls_by_this_validator"]
        == result["optimizer_updates_by_this_validator"]
        == 0
    )
    assert len({ref["path"] for ref in result["inputs"]}) == len(result["inputs"])
    assert {"PREFLIGHT/native.log", "PREFLIGHT/NATIVE_REGRESSION.json"} <= {
        ref["path"] for ref in result["inputs"]
    }
    assert len(bound.calls) == 2


@pytest.mark.parametrize(
    "kind",
    ["raw", "partial", "group", "attempt", "credit", "runtime", "worker", "slot", "submission"],
)
def test_one_altered_fact_cannot_be_replaced_by_complete_or_pass(bound, kind):
    qid = bound.context["public"][0]["qid"]
    task = "LH-SRF6-Confirm-100-b0-0"
    paths = dict(
        raw=f"CONFIRM/raw/b0/{qid}.jsonl",
        partial=f"CONFIRM/raw/b0/{qid}.jsonl",
        group=f"CONFIRM/groups/b0/{qid}.json",
        attempt=f"CONFIRM/attempts/b0/{qid}.json",
        credit=f"CONFIRM/groups/b0/{qid}.json",
        runtime=f"CONFIRM/runtime_identity/{task}.json",
        worker=f"CONFIRM/workers/{task}.json",
        slot="CONFIRM/execution_slots/b0-0.json",
        submission="SUBMISSIONS/b0.result.json",
    )
    path = bound.root / paths[kind]
    original = path.read_bytes()
    value = (
        [json.loads(line) for line in original.splitlines()]
        if kind in ("raw", "partial")
        else json.loads(original)
    )
    if kind == "raw":
        value[0]["full_text"] = '{"answer": -999}'
    elif kind == "partial":
        value.pop()
    elif kind == "group":
        value["records"][0]["wire_score"]["A"] = 0
    elif kind == "attempt":
        value["task_id"] = "LH-SRF6-Confirm-999-b0-0"
    elif kind == "credit":
        value["credit"]["J"] = {"status": "PASS"}
    elif kind == "runtime":
        value["policy_hash"] = "f" * 64
    elif kind == "worker":
        value["generated_this_process"] = 0
        value["reused_rows"] = 1536
    elif kind == "slot":
        value["task_id"] = "LH-SRF6-Confirm-999-b0-0"
    else:
        value["job_id"] = "999"
        value["stdout"] = "999\n"
    try:
        save(bound.root, paths[kind], value, lines=kind in ("raw", "partial"))
        with pytest.raises((PermissionError, ValueError)):
            evidence.validate_saved_confirm(bound.root)
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize(
    "relative",
    [
        "CONFIRM/failures/stopped.json",
        "CONFIRM/raw/b0/.partial.tmp",
        "CONFIRM/executions/orphan.json",
    ],
)
def test_failures_or_unresolved_extra_files_remain_and_block(bound, relative):
    path = bound.root / relative
    save(bound.root, relative, dict(status="UNKNOWN_SYNTHETIC"))
    try:
        with pytest.raises(PermissionError):
            evidence.validate_saved_confirm(bound.root)
        assert path.exists()
    finally:
        path.unlink()


def test_trace_vectors_are_recomputed_even_when_report_keeps_pass(bound):
    task = "LH-SRF6-Confirm-100-b0-0"
    qid = bound.context["public"][0]["qid"]
    group = json.loads((bound.root / f"CONFIRM/groups/b0/{qid}.json").read_text())
    report = json.loads((bound.root / f"CONFIRM/alignment/{task}.json").read_text())
    census.validate_alignment(report, group["spec"], group["records"])
    report["traces"][0]["current_logprobs"][0] -= 1
    with pytest.raises(Exception, match=r"discrepancy|summary"):
        census.validate_alignment(report, group["spec"], group["records"])


def test_ordered_ledger_requires_alignment_before_next_group(bound):
    task = "LH-SRF6-Confirm-100-b0-0"
    ledger = [
        json.loads(line)
        for line in (bound.root / f"CONFIRM/accounting/{task}.jsonl").read_text().splitlines()
    ]
    expected = []
    for row in ledger:
        m, kind, n = row["metadata"], row["kind"], row["count"]
        tail = (
            (m["group_seed"], m.get("protocol", m.get("row_index")))
            if kind != "extra_forward_sequences"
            else (m["purpose"], m["completion_tokens"])
        )
        expected.append((m["qid"], kind, *tail, n))
    evidence._accounting(ledger, task, "b0", bound.context["registration_sha256"], expected)
    ledger[9], ledger[17] = ledger[17], ledger[9]
    with pytest.raises(PermissionError, match="ordered events"):
        evidence._accounting(ledger, task, "b0", bound.context["registration_sha256"], expected)


def test_native_prompt_and_cpu_stopping_replay_use_original_tokens(bound):
    qid = bound.context["public"][0]["qid"]
    rows = [
        json.loads(line)
        for line in (bound.root / f"CONFIRM/raw/b0/{qid}.jsonl").read_text().splitlines()
    ]
    assert evidence.verify_saved_tokens(rows, bound.tokenizer, [3])["checked_responses"] == 8
    changed = copy.deepcopy(rows)
    changed[0]["tokens"].append(2)
    with pytest.raises(PermissionError, match=r"text differs|stopping"):
        evidence.verify_saved_tokens(changed, bound.tokenizer, [3])
    runtime = json.loads(
        (bound.root / "CONFIRM/runtime_identity/LH-SRF6-Confirm-100-b0-0.json").read_text()
    )
    spec = census.group_spec(bound.context, "b0", bound.context["public"][0])
    route = bound.context["preflight"]["inputs"][0]["routing"]
    rows[0]["native_prompt_inputs"]["input_ids"][0][0] = 999
    with pytest.raises(PermissionError, match="native prompt tensor"):
        evidence._native(rows, runtime, runtime["task_id"], bound.root, spec, route, {})


@pytest.mark.parametrize("payload", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e9999}'])
def test_strict_reader_rejects_ambiguous_or_nonfinite_fact_containers(tmp_path, payload):
    (tmp_path / "raw.jsonl").write_text(payload + "\n")
    with pytest.raises(PermissionError):
        evidence._Reader(tmp_path).read("raw.jsonl", lines=True)


def test_reader_reads_each_leaf_once_and_detects_later_change(tmp_path, monkeypatch):
    save(tmp_path, "fact.json", dict(full_text='{"answer":NaN,"answer":2}'))
    reader = evidence._Reader(tmp_path)
    original = Path.read_bytes
    calls = []

    def observe(path):
        calls.append(str(path))
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", observe)
    assert reader.read("fact.json") == reader.read("fact.json")
    assert len(calls) == 1
    reader.stable()
    (tmp_path / "fact.json").write_bytes(original(tmp_path / "fact.json") + b" ")
    with pytest.raises(PermissionError, match="changed during"):
        reader.stable()


def test_reader_rejects_same_content_symlink_substitution(tmp_path):
    save(tmp_path, "fact.json", dict(synthetic=True))
    save(tmp_path, "other.json", dict(synthetic=True))
    reader = evidence._Reader(tmp_path)
    reader.read("fact.json")
    (tmp_path / "fact.json").unlink()
    (tmp_path / "fact.json").symlink_to(tmp_path / "other.json")
    with pytest.raises(PermissionError, match="Symlink"):
        reader.stable()


def test_confirmation_consumer_cannot_claim_old_screen_deployment(bound):
    context = dict(
        bound.context,
        deployment=Path(bound.context["registration"]["predecessor_screen"]["deployment"]),
    )
    with pytest.raises(PermissionError, match="new registered source"):
        evidence._provenance(evidence._Reader(bound.root), context)


def test_late_raw_drift_cannot_receive_original_hash_receipt(bound, monkeypatch):
    qid = bound.context["public"][0]["qid"]
    path = bound.root / f"CONFIRM/raw/b0/{qid}.jsonl"
    original = path.read_bytes()

    def changed(_):
        path.write_bytes(original + b"\n")
        return bound.tokenizer, [3]

    monkeypatch.setattr(evidence, "_tokenizer", changed)
    try:
        with pytest.raises(PermissionError, match="changed during"):
            evidence.validate_saved_confirm(bound.root)
    finally:
        path.write_bytes(original)


def test_boolean_event_index_is_not_an_original_integer_draw(bound):
    task = "LH-SRF6-Confirm-100-b0-0"
    ledger = [
        json.loads(line)
        for line in (bound.root / f"CONFIRM/accounting/{task}.jsonl").read_text().splitlines()
    ]
    ledger[1]["metadata"]["row_index"] = False
    with pytest.raises(PermissionError, match="integers"):
        evidence._accounting(ledger, task, "b0", bound.context["registration_sha256"], [])


def test_opaque_visual_evidence_is_bound_without_interpreting_image_bytes(tmp_path):
    payload = b"\x89PNG\r\n\x1a\nSYNTHETIC PIXELS ONLY"
    (tmp_path / "review.png").write_bytes(payload)
    reader = evidence._Reader(tmp_path)
    ref = dict(path="review.png", sha256=hashlib.sha256(payload).hexdigest())
    assert reader.reference(ref, opaque=True) == payload
    reader.stable()


def test_native_regression_log_is_a_required_pinned_leaf(bound):
    path = bound.root / "PREFLIGHT/native.log"
    original = path.read_bytes()
    path.write_bytes(original + b" altered")
    try:
        with pytest.raises(PermissionError, match="input pin"):
            evidence._provenance(evidence._Reader(bound.root), bound.context)
    finally:
        path.write_bytes(original)
