"""Identity-bound, append-only F2 probe/evaluation workers and CPU evidence loading.

Only completed generations are observations. A lost call or unparseable trailing
fragment remains technical evidence and is never converted to a wrong answer.
Complete JSON without its final newline is retained and validated as an observation;
the missing terminator is logged and a later attempt always uses a new segment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import signal
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from mm_core.execution import sha256_file, utc_now
from mm_core.vl_runtime import gold_completion

from .contract import PLAN_ID, canonical, digest, run_matrix, seed
from .data import model_input

START_STATES = ("S0", "S_A_0", "S_AP_0", "S_A_1", "S_AP_1")
SELF_PROVENANCE = "generated_tokens_actual_self_prefix"
GOLD_PROVENANCE = "gold_completion_answer_prefix_contains_gold_readings"
IDENTITY_KEYS = (
    "state_id",
    "model_hash",
    "state_manifest_hash",
    "producer_run_id",
    "processor_hash",
    "adapter_path",
    "adapter_file_hashes",
    "checkpoint_sha256",
    "trainable_state_hash",
)
KINDS = ("outputs", "self_scores", "gold_scores", "events")


def _read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=_bad_constant)


def _bad_constant(value):
    raise ValueError("Nonfinite JSON number: " + value)


def _freeze_copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _json_once(path, value):
    """Exclusive fsynced records; a worker lease owns this directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, pending = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(
                json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(pending, path)
    finally:
        os.unlink(pending)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append(path, value):
    data = canonical(_freeze_copy(value)) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        while data:
            written = os.write(descriptor, data)
            if written <= 0:
                raise OSError("Short append to durable measurement evidence")
            data = data[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_segment(path, root):
    """Keep an interrupted final fragment untouched; reject every other corruption."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Measurement segment is not a regular file")
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows, tails = [], []
    offset = 0
    for index, line in enumerate(lines):
        try:
            row = json.loads(line, parse_constant=_bad_constant)
            if not isinstance(row, dict):
                raise ValueError("JSONL record must be an object")
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith((b"\n", b"\r")):
                raise ValueError("Corrupt durable JSONL segment: " + str(path)) from None
            tails.append(
                dict(
                    event="INCOMPLETE_WRITE_RETAINED",
                    path=str(path.relative_to(root)),
                    file_sha256=hashlib.sha256(raw).hexdigest(),
                    offset=offset,
                    bytes=len(line),
                    tail_sha256=hashlib.sha256(line).hexdigest(),
                    physical_call_cost="UNKNOWN",
                    scientific_observation=False,
                )
            )
            break
        if index == len(lines) - 1 and not line.endswith((b"\n", b"\r")):
            tails.append(
                dict(
                    event="VALID_JSON_WITHOUT_TERMINATOR_RETAINED",
                    path=str(path.relative_to(root)),
                    file_sha256=hashlib.sha256(raw).hexdigest(),
                    offset=offset,
                    bytes=len(line),
                    scientific_observation=True,
                    physical_call_cost="RECORDED_IF_IDENTITY_VALIDATES",
                )
            )
        rows.append(row)
        offset += len(line)
    return rows, tails


def task_name(panel, state_id, shard, shards):
    if panel not in {"PROBE", "DEV_EVAL"}:
        raise ValueError("Unregistered measurement panel")
    allowed = set(START_STATES)
    if panel == "DEV_EVAL":
        allowed.update(r["run_id"] for r in run_matrix() if r["phase"] == "CONTINUE")
    if state_id not in allowed:
        raise ValueError("Unregistered model state for panel")
    if type(shards) is not int or type(shard) is not int or not 1 <= shards <= 5:
        raise ValueError("One to five preregistered measurement shards required")
    if not 0 <= shard < shards:
        raise ValueError("Shard is outside its fixed partition")
    prefix = "probe" if panel == "PROBE" else "eval"
    return f"{prefix}_{state_id}" + (f"_shard{shard}of{shards}" if shards > 1 else "")


def state_identity(identity, state_id):
    required = set(IDENTITY_KEYS) - {"checkpoint_sha256"}
    if any(identity.get(k) is None for k in required) or identity.get("state_id") != state_id:
        raise ValueError("Incomplete exact model state identity")
    if not identity["adapter_file_hashes"]:
        raise ValueError("Exact per-file adapter identity missing")
    return {key: copy.deepcopy(identity.get(key)) for key in IDENTITY_KEYS}


def panel_questions(plan, root, panel):
    rows, tails = read_segment(Path(root) / "data/questions.jsonl", Path(root))
    if tails:
        raise ValueError("Frozen question file is incomplete")
    questions = sorted((q for q in rows if q.get("split") == panel), key=lambda q: q["question_id"])
    expected = plan["data"]["root_counts"][panel]
    cube = Counter()
    ids = set()
    root_names = {}
    for question in questions:
        qid = question["question_id"]
        if qid in ids:
            raise ValueError("Duplicate panel question")
        ids.add(qid)
        index = question["root_index"]
        if type(index) is not int or not 0 <= index < expected:
            raise ValueError("Unexpected panel root")
        family = question["root_family_id"]
        if index in root_names and root_names[index] != family:
            raise ValueError("Root index maps to multiple families")
        root_names[index] = family
        v, d = question["V"], question["D"]
        if v not in {"low", "high"} or d not in {"low", "high"}:
            raise ValueError("Unexpected registered factor level")
        if not question.get("image_sha256") or not question.get("prompt"):
            raise ValueError("Question lacks frozen image/prompt identity")
        if question.get("prompt_sha256") != hashlib.sha256(question["prompt"].encode()).hexdigest():
            raise ValueError("Frozen prompt text differs from its UTF-8 SHA256")
        cube[(index, question["chart_type"], question["operation"], v, d)] += 1
    wanted = Counter(
        {
            (r, c, o, v, d): 1
            for r in range(expected)
            for c in ("grouped_bar", "line")
            for o in ("sum", "difference", "range")
            for v in ("low", "high")
            for d in ("low", "high")
        }
    )
    if cube != wanted or len(set(root_names.values())) != expected:
        raise ValueError("Panel is not the complete frozen root-by-24-factor cube")
    if plan["sampling"]["probe_eval"]["K"] != 4:
        raise ValueError("Unregistered measurement sample count")
    return questions


def request_identity(question, panel, identity, slot, sample_index, freeze_sha256):
    """Physical placement never changes paired seeds or logical request identities."""
    checkpoint_hash = identity.get("checkpoint_sha256") or identity["model_hash"]
    return dict(
        plan_id=PLAN_ID,
        stage=panel,
        panel=panel,
        state_id=identity["state_id"],
        run_id=identity["state_id"],
        producer_run_id=identity["producer_run_id"],
        run_hash=digest(identity),
        model_hash=identity["model_hash"],
        state_manifest_hash=identity["state_manifest_hash"],
        processor_hash=identity["processor_hash"],
        freeze_sha256=freeze_sha256,
        question_id=question["question_id"],
        question_sha256=digest(question),
        image_sha256=question["image_sha256"],
        prompt_sha256=question["prompt_sha256"],
        slot=slot,
        sample_index=sample_index,
        seed=seed("evaluation", panel, question["question_id"], sample_index),
        request_id=digest(
            [PLAN_ID, panel, identity["state_id"], 0, slot, sample_index, checkpoint_hash]
        ),
    )


def _check_subset(record, expected):
    for key, value in expected.items():
        if record.get(key) != value or type(record.get(key)) is not type(value):
            raise ValueError("Evidence identity mismatch: " + key)


def validate_completion(row, expected):
    _check_subset(row, expected)
    if row.get("status") != "completed" or not isinstance(row.get("raw_text"), str):
        raise ValueError("Only complete raw model responses are observations")
    tokens = row.get("tokens")
    if not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError("Exact generated token IDs required")
    if row.get("raw_tokens") != tokens or row.get("completion_token_count") != len(tokens):
        raise ValueError("Saved token counts/IDs differ")
    if type(row.get("prompt_token_count")) is not int or row["prompt_token_count"] <= 0:
        raise ValueError("Prompt token accounting missing")
    if type(row.get("truncated")) is not bool:
        raise ValueError("Truncation status missing")
    if row.get("image_routing", {}).get("generation_vision_forward_calls", 0) < 1:
        raise ValueError("No actual image encoder routing evidence")


def _packet(packet, provenance, token_count):
    if not isinstance(packet, dict) or packet.get("provenance") != provenance:
        raise ValueError("Teacher-forcing provenance mismatch")
    if packet.get("status") not in {
        "MEASURED",
        "TOKEN_CHARACTER_BOUNDARY_UNVERIFIED",
        "NO_GENERATED_TOKENS",
    }:
        raise ValueError("Unknown field-scoring outcome")
    if packet["status"] == "NO_GENERATED_TOKENS":
        if token_count != 0 or provenance != SELF_PROVENANCE or packet.get("field_nll") is not None:
            raise ValueError("Empty-token status does not match its actual self completion")
    elif packet["status"] == "TOKEN_CHARACTER_BOUNDARY_UNVERIFIED":
        if packet.get("tokens") != token_count or packet.get("field_nll") is not None:
            raise ValueError("Unverified token boundary packet changed")
    else:
        if packet.get("distribution") != "raw_model_before_temperature_or_top_p":
            raise ValueError("Field metrics are not raw-model teacher forcing")
        if not packet.get("boundary_rule") or packet.get("vision_forward_calls", 0) < 1:
            raise ValueError("Field boundary or actual visual forward evidence missing")
        fields = packet.get("field_nll")
        if not isinstance(fields, dict) or set(fields) != {"readings", "answer"}:
            raise ValueError("Measured field packet has incomplete fields")
        used = set()
        for field in fields.values():
            indices, count = field.get("token_indices"), field.get("token_count")
            if (
                type(count) is not int
                or count < 0
                or not isinstance(indices, list)
                or len(indices) != count
                or len(set(indices)) != count
                or any(type(i) is not int or not 0 <= i < token_count for i in indices)
                or used.intersection(indices)
            ):
                raise ValueError("Field-token denominators/indices changed")
            used.update(indices)
            for key in ("mean_nll", "mean_token_entropy"):
                value = field.get(key)
                if (count == 0 and value is not None) or (
                    count > 0
                    and (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value < 0
                    )
                ):
                    raise ValueError("Field score missingness/finite-number contract changed")
        for key in ("token_logprobs", "token_entropy"):
            values = packet.get(key)
            if (
                not isinstance(values, list)
                or len(values) != token_count
                or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                    for v in values
                )
            ):
                raise ValueError("Per-token scoring evidence incomplete")


def _gold_identity(question, panel, identity, slot, freeze_sha256):
    base = request_identity(question, panel, identity, slot, 0, freeze_sha256)
    base.pop("sample_index")
    base.pop("seed")
    base["request_id"] = digest(["GOLD", base["request_id"], gold_completion(question)])
    base["gold_text_sha256"] = hashlib.sha256(gold_completion(question).encode()).hexdigest()
    return base


def _read_directory(root, directory, expected, gold_expected, panel):
    outputs, selfs, golds, events, tails = {}, {}, {}, [], []
    for kind in KINDS:
        for path in sorted(directory.glob(kind + "_*.jsonl")):
            if not re.fullmatch(kind + r"_\d{6}\.jsonl", path.name):
                raise ValueError("Unexpected measurement segment filename")
            rows, interrupted = read_segment(path, root)
            tails.extend(interrupted)
            for row in rows:
                if kind == "events":
                    events.append(row)
                    continue
                key = row.get("request_id")
                target = {"outputs": outputs, "self_scores": selfs, "gold_scores": golds}[kind]
                if key in target:
                    raise ValueError("Duplicate logical record in immutable segments")
                if kind == "outputs":
                    if key not in expected:
                        raise ValueError("Unregistered raw completion")
                    validate_completion(row, expected[key])
                elif panel != "PROBE":
                    raise ValueError("DEV_EVAL forbids additional field teacher forcing")
                elif kind == "self_scores":
                    if key not in expected:
                        raise ValueError("Unregistered self field record")
                    _check_subset(row, expected[key])
                    if key not in outputs:
                        raise ValueError("Self field record has no saved raw completion")
                    _packet(row.get("field_scores"), SELF_PROVENANCE, len(outputs[key]["tokens"]))
                else:
                    if key not in gold_expected:
                        raise ValueError("Unregistered gold field record")
                    _check_subset(row, gold_expected[key])
                    tokens = row.get("tokens")
                    if (
                        not isinstance(tokens, list)
                        or not tokens
                        or any(type(t) is not int or t < 0 for t in tokens)
                        or row.get("token_count") != len(tokens)
                        or row.get("tokens_sha256") != digest(tokens)
                    ):
                        raise ValueError("Gold teacher-forcing exact token evidence changed")
                    _packet(row.get("field_scores"), GOLD_PROVENANCE, len(tokens))
                target[key] = row
    for key, row in selfs.items():
        if key not in outputs or row.get("completion_sha256") != digest(outputs[key]):
            raise ValueError("Self field record does not bind exact saved raw completion")
        if row.get("tokens_sha256") != digest(outputs[key]["tokens"]):
            raise ValueError("Self field scoring token identity changed")
    unresolved = []
    for event in events:
        kind = event.get("event")
        target = {
            "GENERATION_STARTED": outputs,
            "SELF_FORWARD_STARTED": selfs,
            "GOLD_FORWARD_STARTED": golds,
        }.get(kind)
        if target is None:
            continue
        saved = target.get(event.get("request_id"), {})
        if saved.get("attempt") != event.get("attempt"):
            unresolved.append(
                dict(
                    event="PHYSICAL_CALL_OUTCOME_UNKNOWN",
                    call=kind,
                    attempt=event.get("attempt"),
                    request_id=event.get("request_id"),
                    physical_call_cost="UNKNOWN",
                    scientific_observation=False,
                )
            )
    return dict(
        outputs=outputs,
        self_scores=selfs,
        gold_scores=golds,
        technical_events=events + tails + unresolved,
        interrupted_tails=[t for t in tails if t["event"] == "INCOMPLETE_WRITE_RETAINED"],
    )


def _scope(root, panel, state_id, shard, shards):
    task_name(panel, state_id, shard, shards)
    parent = Path(root) / "raw" / panel / state_id
    for path in (Path(root) / "raw", parent.parent, parent):
        if path.is_symlink():
            raise ValueError("Unsafe measurement parent directory")
    for directory in parent.glob("shard_*"):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Unsafe measurement shard directory")
        match = re.fullmatch(r"shard_(\d+)of([1-5])", directory.name)
        if not match or int(match[2]) != shards or int(match[1]) >= shards:
            raise ValueError("Fixed measurement partition changed")
    return parent / f"shard_{shard}of{shards}"


def _expected(questions, panel, identity, freeze_sha256, shard, shards):
    selected = [(slot, q) for slot, q in enumerate(questions) if slot % shards == shard]
    expected, golds = {}, {}
    for slot, question in selected:
        for sample in range(4):
            value = request_identity(question, panel, identity, slot, sample, freeze_sha256)
            expected[value["request_id"]] = value
        value = _gold_identity(question, panel, identity, slot, freeze_sha256)
        golds[value["request_id"]] = value
    return selected, expected, golds


def _artifacts(root, directory):
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("Unexpected non-regular measurement artifact")
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "COMPLETE.json"
    }


def _completion_receipt(root, directory, base, expected, data, question_count):
    return dict(
        **base,
        status="COMPLETE",
        question_count=question_count,
        expected_completions=len(expected),
        actual_completions=len(data["outputs"]),
        self_score_count=len(data["self_scores"]),
        gold_score_count=len(data["gold_scores"]),
        logical_keys_sha256=digest(sorted(expected)),
        technical_event_count=len(data["technical_events"]),
        incomplete_tail_count=len(data["interrupted_tails"]),
        unterminated_complete_record_count=sum(
            e["event"] == "VALID_JSON_WITHOUT_TERMINATOR_RETAINED" for e in data["technical_events"]
        ),
        artifacts=_artifacts(root, directory),
    )


def _verify_receipt(root, directory, expected_receipt):
    saved = _read_json(directory / "COMPLETE.json")
    if saved != expected_receipt:
        raise ValueError("Measurement completion receipt differs from exact persisted evidence")
    for relative, wanted in saved["artifacts"].items():
        if sha256_file(Path(root) / relative) != wanted:
            raise ValueError("Immutable measurement artifact changed")
    return saved


def _base(panel, identity, freeze_sha256, shard, shards):
    return dict(
        plan_id=PLAN_ID,
        panel=panel,
        state_id=identity["state_id"],
        task_id=task_name(panel, identity["state_id"], shard, shards),
        shard=shard,
        shards=shards,
        freeze_sha256=freeze_sha256,
        state_identity=identity,
    )


def load_panel(plan_path, run_root, panel, state_id):
    """CPU-only canonical observations, unique PROBE sidecars, and explicit coverage."""
    from .common import load_plan, verify_execution
    from .runtime import get_state_identity

    plan = load_plan(plan_path)
    verify_execution(plan_path, run_root, require_engine=True)
    root = Path(run_root).resolve(strict=True)
    identity = state_identity(get_state_identity(plan, root, state_id), state_id)
    freeze_sha = sha256_file(root / "manifests/F2_FREEZE.json")
    questions = panel_questions(plan, root, panel)
    registration = _read_json(root / "orchestration/REGISTRATION.json")
    shards = registration["measurement_shards"]
    combined = dict(
        questions=questions,
        outputs=[],
        self_scores={},
        gold_scores={},
        complete=True,
        identity=identity,
        technical_events=[],
        receipts=[],
    )
    for shard in range(shards):
        directory = _scope(root, panel, state_id, shard, shards)
        selected, expected, golds = _expected(questions, panel, identity, freeze_sha, shard, shards)
        data = _read_directory(root, directory, expected, golds, panel)
        complete = _is_complete(data, expected, golds, panel)
        base = _base(panel, identity, freeze_sha, shard, shards)
        if (directory / "COMPLETE.json").exists():
            if not complete:
                raise ValueError("Completion receipt claims an incomplete panel shard")
            receipt = _completion_receipt(root, directory, base, expected, data, len(selected))
            combined["receipts"].append(_verify_receipt(root, directory, receipt))
        else:
            complete = False
        combined["complete"] &= complete
        combined["outputs"].extend(data["outputs"].values())
        combined["self_scores"].update(data["self_scores"])
        combined["gold_scores"].update({v["question_id"]: v for v in data["gold_scores"].values()})
        combined["technical_events"].extend(data["technical_events"])
    combined["outputs"].sort(key=lambda row: (row["slot"], row["sample_index"]))
    return combined


def _is_complete(data, expected, golds, panel):
    return data["outputs"].keys() == expected.keys() and (
        panel == "DEV_EVAL"
        or (
            data["self_scores"].keys() == expected.keys()
            and data["gold_scores"].keys() == golds.keys()
        )
    )


def _repair_authorization(root, directory, base):
    failures = sorted(directory.glob("FAILURE_*.json"))
    if not failures:
        return None
    failure_hash = sha256_file(failures[-1])
    for path in sorted(directory.glob("STARTED_*.json")):
        previous = _read_json(path).get("repair_authorization") or {}
        if previous.get("failure_receipt_sha256") == failure_hash:
            return previous
    path = root / "manifests/TECHNICAL_REPAIRS" / (base["task_id"] + ".json")
    if not path.exists():
        raise PermissionError(
            "Recorded software failure requires an explicit technical repair receipt"
        )
    receipt = _read_json(path)
    _check_subset(
        receipt,
        dict(
            plan_id=PLAN_ID,
            task_id=base["task_id"],
            approved=True,
            failure_receipt_sha256=failure_hash,
            freeze_sha256=base["freeze_sha256"],
        ),
    )
    if any(
        key not in receipt
        for key in (
            "source_before",
            "source_after",
            "repair_scope",
            "rerun_comparable_block_required",
        )
    ):
        raise PermissionError("Technical repair scope/acknowledgment missing")
    if receipt["rerun_comparable_block_required"] is not False:
        raise PermissionError("Comparable-block repair needs separate orchestration")
    return {**receipt, "receipt_sha256": sha256_file(path)}


def run_shard(
    *,
    plan,
    root,
    panel,
    state_id,
    shard,
    shards,
    identity,
    freeze_sha256,
    runtime_factory,
    stop_requested=lambda: False,
):
    """Run under the caller's registered worker lease; injectable factory supports CPU tests."""
    root = Path(root).resolve(strict=True)
    identity = state_identity(identity, state_id)
    questions = panel_questions(plan, root, panel)
    selected, expected, golds = _expected(questions, panel, identity, freeze_sha256, shard, shards)
    directory = _scope(root, panel, state_id, shard, shards)
    directory.mkdir(parents=True, exist_ok=True)
    base = _base(panel, identity, freeze_sha256, shard, shards)
    session = directory / "SESSION.json"
    if session.exists():
        if _read_json(session) != base:
            raise ValueError("Measurement session/model/freeze changed")
    else:
        _json_once(session, base)
    data = _read_directory(root, directory, expected, golds, panel)
    if (directory / "COMPLETE.json").exists():
        if not _is_complete(data, expected, golds, panel):
            raise ValueError("Completed shard lost observations")
        receipt = _completion_receipt(root, directory, base, expected, data, len(selected))
        return _verify_receipt(root, directory, receipt)
    repair = _repair_authorization(root, directory, base)
    attempts = [int(p.stem.split("_")[-1]) for p in directory.glob("STARTED_*.json")]
    attempt = max(attempts, default=-1) + 1
    suffix = f"{attempt:06d}"
    _json_once(
        directory / f"STARTED_{suffix}.json",
        dict(**base, attempt=attempt, at=utc_now(), repair_authorization=repair),
    )
    paths = {kind: directory / f"{kind}_{suffix}.jsonl" for kind in KINDS}

    def event(kind, **values):
        value = dict(event=kind, attempt=attempt, at=utc_now(), **values)
        _append(paths["events"], value)
        data["technical_events"].append(value)

    try:
        for tail in data["interrupted_tails"]:
            event("PRESERVED_INTERRUPTED_WRITE", evidence=tail, physical_call_cost="UNKNOWN")
        if not _is_complete(data, expected, golds, panel) and not stop_requested():
            runtime, loaded_identity = runtime_factory()
            if state_identity(loaded_identity, state_id) != identity:
                raise ValueError("Loaded model differs from preregistered exact state")
            for slot, question in selected:
                if stop_requested():
                    break
                prepared = None
                for sample in range(4):
                    if stop_requested():
                        break
                    request = request_identity(
                        question, panel, identity, slot, sample, freeze_sha256
                    )
                    request_id = request["request_id"]
                    if request_id not in data["outputs"]:
                        event("GENERATION_STARTED", request_id=request_id, seed=request["seed"])

                        def save_completion(generated, request=request, request_id=request_id):
                            if request_id in data["outputs"]:
                                raise ValueError("Runtime tried to persist a duplicate completion")
                            for key, value in request.items():
                                if key in generated and generated[key] != value:
                                    raise ValueError("Runtime overwrote request identity: " + key)
                            saved = _freeze_copy(
                                {**generated, **request, "status": "completed", "attempt": attempt}
                            )
                            validate_completion(saved, request)
                            _append(paths["outputs"], saved)
                            data["outputs"][request_id] = saved

                        returned = runtime.generate(
                            model_input(question),
                            root,
                            request["seed"],
                            score_fields=False,
                            on_completion=save_completion,
                        )
                        if request_id not in data["outputs"]:
                            raise RuntimeError(
                                "Runtime failed to persist completion before returning"
                            )
                        for key in ("raw_text", "tokens", "raw_tokens", "seed"):
                            if returned.get(key) != data["outputs"][request_id][key]:
                                raise ValueError(
                                    "Runtime changed a completion after durable callback"
                                )
                        event("GENERATION_SAVED", request_id=request_id)
                    if panel == "PROBE" and request_id not in data["self_scores"]:
                        raw = data["outputs"][request_id]
                        if prepared is None:
                            prepared = runtime.prepare(model_input(question), root)
                        event("SELF_FORWARD_STARTED", request_id=request_id)
                        start = time.perf_counter()
                        packet = (
                            runtime.field_scores(
                                prepared, list(raw["tokens"]), provenance=SELF_PROVENANCE
                            )
                            if raw["tokens"]
                            else {
                                "status": "NO_GENERATED_TOKENS",
                                "provenance": SELF_PROVENANCE,
                                "field_nll": None,
                            }
                        )
                        _packet(packet, SELF_PROVENANCE, len(raw["tokens"]))
                        saved = _freeze_copy(
                            dict(
                                **request,
                                attempt=attempt,
                                field_scores=packet,
                                completion_sha256=digest(raw),
                                tokens_sha256=digest(raw["tokens"]),
                                scoring_seconds=time.perf_counter() - start,
                            )
                        )
                        _append(paths["self_scores"], saved)
                        data["self_scores"][request_id] = saved
                        event("SELF_FORWARD_SAVED", request_id=request_id)
                gold_request = _gold_identity(question, panel, identity, slot, freeze_sha256)
                gold_id = gold_request["request_id"]
                if panel == "PROBE" and gold_id not in data["gold_scores"] and not stop_requested():
                    if prepared is None:
                        prepared = runtime.prepare(model_input(question), root)
                    tokens = runtime.encode_completion(gold_completion(question))
                    event("GOLD_FORWARD_STARTED", request_id=gold_id)
                    start = time.perf_counter()
                    packet = runtime.field_scores(
                        prepared, list(tokens), provenance=GOLD_PROVENANCE
                    )
                    _packet(packet, GOLD_PROVENANCE, len(tokens))
                    saved = _freeze_copy(
                        dict(
                            **gold_request,
                            attempt=attempt,
                            field_scores=packet,
                            tokens=tokens,
                            tokens_sha256=digest(tokens),
                            token_count=len(tokens),
                            scoring_seconds=time.perf_counter() - start,
                        )
                    )
                    _append(paths["gold_scores"], saved)
                    data["gold_scores"][gold_id] = saved
                    event("GOLD_FORWARD_SAVED", request_id=gold_id)
        if not _is_complete(data, expected, golds, panel):
            event("PREEMPTION_CHECKPOINT", missing_generations=len(expected) - len(data["outputs"]))
            return dict(
                **base,
                status="CHECKPOINTED",
                artifacts=_artifacts(root, directory),
                actual_completions=len(data["outputs"]),
                expected_completions=len(expected),
            )
        event("SHARD_COMPLETE")
        data = _read_directory(root, directory, expected, golds, panel)
        receipt = _completion_receipt(root, directory, base, expected, data, len(selected))
        _json_once(directory / "COMPLETE.json", receipt)
        return receipt
    except Exception as exc:
        event(
            "SOFTWARE_FAILURE",
            exception_type=type(exc).__name__,
            message=str(exc),
            scientific_observation=False,
        )
        _json_once(
            directory / f"FAILURE_{suffix}.json",
            dict(
                **base,
                status="FAILED",
                attempt=attempt,
                exception_type=type(exc).__name__,
                message=str(exc),
                artifacts=_artifacts(root, directory),
                at=utc_now(),
            ),
        )
        raise


def main(panel, argv=None):
    from .common import CostLedger, load_plan, require_allocation, verify_execution
    from .orchestration import checkpoint_task, complete_task, fail_task, worker_lease
    from .runtime import create_evaluation_runtime, get_state_identity

    parser = argparse.ArgumentParser(description="Frozen F2 " + panel + " worker")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--shard", default=0, type=int)
    parser.add_argument("--shards", default=1, type=int)
    args = parser.parse_args(argv)
    task_id = task_name(panel, args.state_id, args.shard, args.shards)
    if os.environ.get("MM_DEV_TASK_ID") != task_id:
        raise PermissionError("Explicit registered worker task identity required")
    root = args.run_root.resolve(strict=True)
    plan = load_plan(args.plan)
    verify_execution(args.plan, root, require_engine=True)
    registration = _read_json(root / "orchestration/REGISTRATION.json")
    if registration["measurement_shards"] != args.shards or task_id not in registration["tasks"]:
        raise PermissionError("Measurement shard partition differs from registration")
    stopped = []
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in previous:
        signal.signal(sig, lambda signum, frame: stopped.append(signum))
    try:
        with worker_lease(root, task_id):
            require_allocation(root, task_id)
            identity = get_state_identity(plan, root, args.state_id)
            ledger = CostLedger(root, task_id)
            receipt = run_shard(
                plan=plan,
                root=root,
                panel=panel,
                state_id=args.state_id,
                shard=args.shard,
                shards=args.shards,
                identity=identity,
                freeze_sha256=sha256_file(root / "manifests/F2_FREEZE.json"),
                runtime_factory=lambda: create_evaluation_runtime(
                    plan, root, args.state_id, account=ledger.reserve
                ),
                stop_requested=lambda: bool(stopped),
            )
            artifacts = list(receipt["artifacts"])
            if receipt["status"] == "COMPLETE":
                complete_path = (
                    _scope(root, panel, args.state_id, args.shard, args.shards) / "COMPLETE.json"
                )
                artifacts.append(str(complete_path.relative_to(root)))
                marker = root / f"orchestration/completions/{task_id}.json"
                if not marker.exists():
                    complete_task(root, task_id, artifacts, metadata={"measurement": receipt})
            else:
                checkpoint_task(root, task_id, "PREEMPTION", artifacts, metadata=receipt)
    except Exception as exc:
        try:
            fail_task(root, task_id, exc)
        except Exception as marker_error:
            print(
                "Technical failure marker could not be persisted: " + str(marker_error),
                file=sys.stderr,
            )
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0
