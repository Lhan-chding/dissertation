"""Freeze the complete FORMAL evaluation plan, without permitting model access.

This is independent of FORMAL training registration. It records selectors for
future real commits, never fabricated commit hashes. Original retention IDs and
source-native inputs come only from S0-bound files. All protocols use the F6
sampled T=1/1536 recipe, including ChartQA G=1. Runtime adapters are separate work.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib
import json
import os
from collections import Counter
from pathlib import Path

from srf6_ref.core import FAMILIES, seed_for
from srf6_ref.planning import ARMS

from . import formal_registration as formal
from . import phases, review_package, source_binding
from .census_evidence import _strict_json
from .contract import AMENDMENTS, validate_config
from .data import public_prompt
from .registration import PARENTS, _inputs, _payload
from .rollouts import _path, _publish
from .transactions import _plain_path, _read_bytes, _require, _sha

PHASE = "FORMAL_V1"
DIRECTORY = "PHASE_RUNS/FORMAL_V1/FORMAL_EVAL_FROZEN"
SPEC_FILE = "FORMAL_EVAL_SPEC_FROZEN.json"
SCHEMA = "srf6-formal-evaluation-plan-v1"
BINDING_PATH = "PHASES/FORMAL_V1/RETENTION_PROTOCOL_BINDING.json"
BINDING_SCHEMA = "srf6-formal-retention-protocol-binding-v1"
OPERATIONS_RECEIPT = "sr_f6_operations_20261027/RETENTION_INTERFACE_SOURCE_BINDING_20261027.json"
INTERFACE_FILES = (
    "src/sr_f4/run.py",
    "src/sr_f4/evaluation.py",
    "src/sr_f3_ce/prompts.py",
    "src/sr_f3_ce/external.py",
    "src/sr_f1/chartqa.py",
)
INHERITED_FILES = (
    "src/sr_f4/task_contract.py",
    "src/sr_f3_ce/contract.py",
    "src/sr_f3_ce/runtime.py",
)
RETENTION_FILES = (
    "LEGACY_TASKS.jsonl",
    "CHARTQA_TASKS.jsonl",
    "LEGACY_SOURCE_ROUTES.jsonl",
    "CHARTQA_SOURCE_ROUTES.jsonl",
    "FROZEN.json",
)
PANELS = {
    "DEV_EVAL": (96, 1),
    "COMPOSITION": (768, 4),
    "ID": (384, 2),
    "MECHANISM": (192, 4),
    "CLASSOUT": (192, 2),
    "NATURAL_INTERFACE": (128, 2),
    "LEGACY": (128, 4),
    "CHARTQA": (250, 1),
}
SPLITS = {
    "DEV_EVAL": "DEV_EVAL",
    "COMPOSITION": "TEST_COMPOSITION",
    "ID": "TEST_ID",
    "MECHANISM": "TEST_MECHANISM",
    "CLASSOUT": "TEST_CLASSOUT",
    "NATURAL_INTERFACE": "NATURAL_INTERFACE",
}
GENERATION = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=1536)
SOURCE_DEPENDENCIES = (
    *INTERFACE_FILES,
    "src/sr_f4/task_contract.py",
    "src/sr_f3_ce/contract.py",
    "src/sr_f3_ce/runtime.py",
    "src/sr_f6/measurement.py",
    "src/srf6_ref/measurement.py",
    "src/srf6_ref/core.py",
    "src/srf6_ref/statistics.py",
    "src/sr_f6/data.py",
    "src/sr_f6/formal_evaluation_plan.py",
)
AUTHORIZATION = dict(
    training_authorized=False,
    generation_authorized=False,
    evaluation_authorized=False,
    formal_test_access=False,
)


def _actual_deployment():
    return Path(__file__).resolve().parents[2]


def _reference(path, payload):
    return dict(path=path, sha256=hashlib.sha256(payload).hexdigest())


def _read(root, reference, *, lines=False):
    _require(
        isinstance(reference, dict)
        and set(reference) == {"path", "sha256"}
        and _sha(reference["sha256"]),
        "Exact content reference required",
    )
    raw = _read_bytes(_path(root, reference["path"]))
    _require(hashlib.sha256(raw).hexdigest() == reference["sha256"], "Bound content changed")
    return (
        [_strict_json(line) for line in raw.splitlines() if line.strip()]
        if lines
        else _strict_json(raw)
    )


def _base(root, phase_context, pilot_spec_ref):
    root = _plain_path(root)
    _require(
        isinstance(phase_context, dict)
        and phase_context.get("deployment") == str(_actual_deployment()),
        "Evaluation plan must execute its actual FORMAL deployment",
    )
    context = formal._context(root, phase_context)
    pilot, producer = formal._pilot(root, context, pilot_spec_ref)
    _reader, config, data, data_ref, order_ref, _, parents, binding_ref = _inputs(
        root, context, pilot.get("parent_references")
    )
    validate_config(config)
    _require(
        pilot.get("parent_tensors") == parents and pilot.get("source_binding") == binding_ref,
        "Original PILOT parent/source provenance differs",
    )
    manifest = _read(root, context["identity"]["source_manifest"])
    members = phases._members(manifest)
    _require(
        set(SOURCE_DEPENDENCIES) <= set(members), "Formal source lacks evaluation dependencies"
    )
    source_refs = {
        name: dict(path=name, sha256=members[name]["sha256"]) for name in SOURCE_DEPENDENCIES
    }
    return dict(
        root=root,
        context=context,
        pilot=pilot,
        producer=producer,
        config=config,
        data=data,
        data_ref=data_ref,
        order_ref=order_ref,
        parents=parents,
        binding_ref=binding_ref,
        source_refs=source_refs,
        source_members=members,
    )


def _absolute(path):
    path = Path(path)
    _require(
        path.is_absolute() and path == _plain_path(path) and path.is_relative_to(phases.PROJECT),
        "Source path must stay in canonical project",
    )
    return path


def _external(bound, *, lines=False):
    """Small source/task payloads are read exactly; large images use prior pins/cache."""
    path = _absolute(bound["path"])
    _require(path == Path(bound["resolved_path"]), "External source cannot redirect")
    raw = _read_bytes(path)
    _require(
        len(raw) == bound["bytes"] and hashlib.sha256(raw).hexdigest() == bound["sha256"],
        "Original external source content changed",
    )
    return (
        [_strict_json(row) for row in raw.splitlines() if row.strip()]
        if lines
        else _strict_json(raw)
    )


def _protocol_binding(base):
    root, context = base["root"], base["context"]
    original = _read(root, base["binding_ref"])
    receipt_path = _absolute(phases.PROJECT / OPERATIONS_RECEIPT)
    raw = _read_bytes(receipt_path)
    receipt = _strict_json(raw)
    old = _absolute(original["previous_source"])
    _require(
        receipt.get("schema") == "srf6-retention-interface-source-binding-v1"
        and receipt.get("source_deployment") == str(old)
        and receipt.get("status") == "BOUND_CONTENT_ONLY_NOT_AUTHORIZED"
        and receipt.get("execution_origin") == "SERVER_ACTUAL_SOURCE_READ"
        and receipt.get("model_calls") == 0
        and receipt.get("optimizer_updates") == 0
        and all(
            receipt.get(k) is False
            for k in ("training_authorized", "evaluation_authorized", "formal_test_access")
        ),
        "Original five-module content receipt differs; receipt cannot authorize evaluation",
    )
    files = receipt.get("files")
    _require(
        isinstance(files, list)
        and len(files) == len(INTERFACE_FILES)
        and {f.get("relative") for f in files} == set(INTERFACE_FILES),
        "Exactly five original retention interface modules required",
    )
    current = Path(context["deployment"])
    verified = []
    for item in sorted(files, key=lambda x: x["relative"]):
        relative = item["relative"]
        path = _absolute(item["path"])
        _require(
            path == old / relative and item["resolved_path"] == str(path),
            "Original interface path differs from F5 producer",
        )
        # Original content was read by the fixed operations producer; current
        # FORMAL content was verified above by phases. Revalidate changed stats,
        # rather than reread unchanged modules on every plan verification.
        source_binding.verify_bound_files([item], boundary=phases.PROJECT)
        _require(
            base["source_members"][relative]["bytes"] == item["bytes"]
            and base["source_refs"][relative]["sha256"] == item["sha256"],
            "Original F5 and actual FORMAL interface contents differ",
        )
        verified.append(
            dict(
                relative=relative,
                original_path=str(path),
                formal_path=str(current / relative),
                bytes=item["bytes"],
                sha256=item["sha256"],
            )
        )
    inherited = []
    source_files = original.get("source_files", [])
    for relative in INHERITED_FILES:
        matches = [item for item in source_files if item.get("path") == str(old / relative)]
        _require(len(matches) == 1, "Original S0 native scorer/stop binding required")
        item = matches[0]
        _require(
            item.get("resolved_path") == str(_absolute(old / relative)),
            "Original S0 native scorer/stop path changed",
        )
        source_binding.verify_bound_files([item], boundary=phases.PROJECT)
        _require(
            base["source_refs"][relative]["sha256"] == item["sha256"]
            and base["source_members"][relative]["bytes"] == item["bytes"],
            "Actual FORMAL native scorer/stop differs from original S0 binding",
        )
        inherited.append(
            dict(
                relative=relative,
                original_path=item["path"],
                formal_path=str(current / relative),
                bytes=item["bytes"],
                sha256=item["sha256"],
            )
        )
    return dict(
        schema=BINDING_SCHEMA,
        phase=PHASE,
        status="BOUND_CONTENT_ONLY_NOT_AUTHORIZED",
        phase_identity=dict(
            path=f"PHASES/{PHASE}/IDENTITY.json", sha256=context["identity_sha256"]
        ),
        actual_source_manifest=context["identity"]["source_manifest"],
        source_binding=base["binding_ref"],
        pilot_spec_ref=base["producer"]["specification"],
        original_receipt=dict(path=str(receipt_path), sha256=hashlib.sha256(raw).hexdigest()),
        original_deployment=str(old),
        formal_deployment=str(current),
        files=verified,
        inherited_native_files=inherited,
        model_calls=0,
        optimizer_updates=0,
        **AUTHORIZATION,
    )


def bind_retention_protocol(root, *, phase_context, pilot_spec_ref):
    """Bind actual five-module bytes at the sole registered operations receipt path."""
    base = _base(root, phase_context, pilot_spec_ref)
    value = _protocol_binding(base)
    payload = _payload(value)
    _publish(_path(base["root"], BINDING_PATH), payload)
    return dict(binding=value, binding_ref=_reference(BINDING_PATH, payload), **AUTHORIZATION)


def _verify_protocol(base, reference):
    _require(
        isinstance(reference, dict) and reference.get("path") == BINDING_PATH,
        "Only the fixed FORMAL retention binding is permitted",
    )
    saved = _read(base["root"], reference)
    _require(
        saved == _protocol_binding(base), "Retention binding differs from actual source replay"
    )
    return saved


def verify_retention_protocol(root, *, phase_context, pilot_spec_ref, binding_ref):
    base = _base(root, phase_context, pilot_spec_ref)
    return dict(
        binding=_verify_protocol(base, binding_ref),
        binding_ref=copy.deepcopy(binding_ref),
        status="VERIFIED_CONTENT_NOT_AUTHORIZED",
        **AUTHORIZATION,
    )


def _indexed(rows, count, label):
    _require(
        isinstance(rows, list)
        and len(rows) == count
        and all(
            isinstance(row, dict) and isinstance(row.get("qid"), str) and row["qid"] for row in rows
        ),
        f"Wrong {label} row schema/count",
    )
    result = {row["qid"]: row for row in rows}
    _require(len(result) == count, f"Duplicate {label} qid")
    return result


def _row(panel, task, model_input, source_refs, index):
    draws = PANELS[panel][1]
    paired = task.get("paired_id_qid") if panel == "NATURAL_INTERFACE" else None
    stream_panel, stream_qid = ("ID", paired) if paired else (panel, task["qid"])
    domain = "formal_dev_eval" if panel == "DEV_EVAL" else "formal_final_eval"
    row = dict(
        panel=panel,
        qid=task["qid"],
        group_index=index,
        draws=draws,
        seeds=[
            seed_for(2026102701, domain, stream_panel, stream_qid, draw) for draw in range(draws)
        ],
        protocol_profile=panel,
        model_input=copy.deepcopy(model_input),
        source_refs=source_refs,
    )
    for key in ("root_id", "family", "variant"):
        if key in task:
            row[key] = task[key]
    if paired:
        row["paired_id_qid"] = paired
    return row


def _internal_panels(base):
    root, data_ref = base["root"], base["data_ref"]
    panels, private_maps = {}, {}
    for panel, split in SPLITS.items():
        public_ref = data_ref(f"DATA/{split}/MODEL_INPUTS.jsonl")
        private_ref = data_ref(f"DATA/{split}/TASKS_PRIVATE.jsonl")
        public = _indexed(_read(root, public_ref, lines=True), PANELS[panel][0], panel)
        private = _indexed(
            _read(root, private_ref, lines=True), PANELS[panel][0], panel + " private"
        )
        _require(set(public) == set(private), "Public/private qid association differs")
        if panel != "NATURAL_INTERFACE":
            roots = base["config"]["data"]["roots_per_family"][split]
            expected = {
                f"{split}_{f}_{i:04d}_P0_v{v}"
                for f in FAMILIES
                for i in range(roots)
                for v in range(4)
            }
            _require(set(public) == expected, "Frozen internal panel IDs differ")
        else:
            ids = private_maps["ID"]
            selected = [qid for f in FAMILIES for qid in sorted(ids) if ids[qid]["family"] == f]
            anchors = [
                qid
                for f in FAMILIES
                for qid in [q for q in sorted(ids) if ids[q]["family"] == f][:16]
            ]
            remaining = sorted(
                set(selected) - set(anchors),
                key=lambda q: (seed_for(2026102701, "natural_interface", q), q),
            )
            _require(
                set(public) == {"NATURAL_" + q for q in anchors + remaining[:32]},
                "Natural interface must use existing registered ID selection",
            )
        rows = []
        for index, qid in enumerate(sorted(public)):
            item, task = public[qid], private[qid]
            natural = panel == "NATURAL_INTERFACE"
            protocol = "natural_answer" if natural else "structured_v4"
            _require(
                all(
                    item.get(k) == task.get(k)
                    for k in ("qid", "root_id", "family", "variant", "split", "factor")
                )
                and task.get("split") == split
                and task.get("factor") == "P0"
                and task.get("family") in FAMILIES
                and type(task.get("variant")) is int
                and task["variant"] in range(4)
                and item.get("protocol") == protocol,
                "Internal panel public identity/protocol differs",
            )
            lineage = f"{task['root_id']}_P0_v{task['variant']}"
            _require(
                qid == ("NATURAL_" if natural else "") + lineage,
                "Internal root/view lineage differs",
            )
            if natural:
                paired = task.get("paired_id_qid")
                _require(
                    paired == lineage and paired in private_maps["ID"], "Natural ID pairing differs"
                )
                other = private_maps["ID"][paired]
                _require(
                    all(
                        task.get(k) == other.get(k)
                        for k in ("world", "query", "root_id", "family", "variant", "gold_answer")
                    ),
                    "Natural interface private task changed from paired ID",
                )
            safe = item.get("model_input")
            _require(
                isinstance(safe, dict)
                and set(safe) == {"prompt", "image_file", "image_sha256"}
                and safe["prompt"] == public_prompt(task, natural=natural),
                "Internal input must match fixed public-only prompt",
            )
            if natural:
                paired_input = next(
                    r["model_input"]
                    for r in panels["ID"]["rows"]
                    if r["qid"] == task["paired_id_qid"]
                )
                _require(
                    all(safe[k] == paired_input[k] for k in ("image_file", "image_sha256")),
                    "Natural interface must retain its original ID image",
                )
            image = _path(root, safe["image_file"])
            _require(
                safe["image_file"].startswith("DATA/images/")
                and data_ref(safe["image_file"])["sha256"] == safe["image_sha256"]
                and image.is_file(),
                "Internal image must match frozen DATA inventory",
            )
            rows.append(
                _row(panel, task, safe, dict(public=public_ref, private=private_ref), index)
            )
        private_maps[panel] = private
        panels[panel] = dict(
            rows=rows,
            private_source_refs=[private_ref],
            image_root_binding=dict(kind="EXPERIMENT_ROOT", path=str(root)),
        )
    return panels


def _loaded_module_paths():
    return {
        name: Path(importlib.import_module(name).__file__)
        for name in ("sr_f1.chartqa", "sr_f4.evaluation", "sr_f3_ce.prompts")
    }


def _retention_panels(base):
    original = _read(base["root"], base["binding_ref"])
    inventory = original["retention"]
    _require(
        set(inventory) == set(RETENTION_FILES), "Exactly five original retention files required"
    )
    previous = _absolute(original["previous_run"])
    values = {}
    for name in RETENTION_FILES:
        _require(
            inventory[name]["path"] == str(previous / "RETENTION" / name),
            "Retention file route differs",
        )
        values[name] = _external(inventory[name], lines=name.endswith(".jsonl"))
    frozen = values["FROZEN.json"]
    _require(
        frozen.get("model_outputs_used") is False
        and frozen.get("legacy_questions") == 128
        and frozen.get("chartqa_questions") == 250
        and frozen.get("legacy_seed") == 2026102309
        and frozen.get("chartqa_seed") == 2026102310,
        "Frozen retention selection contract differs",
    )
    data_root = _absolute(Path(frozen["legacy_source"]).parent)
    _require(
        original.get("retention_image_root") == str(data_root),
        "Original retention image base differs",
    )
    from sr_f1.chartqa import PROMPT
    from sr_f4.evaluation import model_input_for

    for name, path in _loaded_module_paths().items():
        relative = "src/" + name.replace(".", "/") + ".py"
        _require(
            path == _path(Path(base["context"]["deployment"]), relative),
            "Loaded retention prompt helper differs from actual FORMAL source",
        )

    panels = {}
    for panel in ("LEGACY", "CHARTQA"):
        tasks = values[f"{panel}_TASKS.jsonl"]
        _require(
            hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()
            == frozen[f"{panel.lower()}_task_digest"],
            "Frozen retention task digest differs",
        )
        task_map = _indexed(tasks, PANELS[panel][0], panel)
        routes = _indexed(
            values[f"{panel}_SOURCE_ROUTES.jsonl"], PANELS[panel][0], panel + " routes"
        )
        images = _indexed(original["retention_images"][panel], PANELS[panel][0], panel + " images")
        _require(
            set(task_map) == set(routes) == set(images), "Retention qid routes/image pins differ"
        )
        # Different ChartQA questions may share an image. Reuse each content pin once.
        unique = {}
        for qid in sorted(task_map):
            route, bound = routes[qid], images[qid]
            expected_path = _absolute(source_binding.retention_image_path(frozen, route))
            _require(
                bound["path"] == str(expected_path)
                and bound["resolved_path"] == str(expected_path)
                and bound["sha256"] == route["image_sha256"],
                "Retention image pin or DATA route differs",
            )
            signature = {k: v for k, v in bound.items() if k != "qid"}
            _require(
                unique.setdefault(str(expected_path), signature) == signature,
                "Conflicting shared-image pins",
            )
        source_binding.verify_bound_files(list(unique.values()), boundary=phases.PROJECT)
        rows = []
        if panel == "LEGACY":
            _require(
                Counter(t.get("family") for t in tasks)
                == {f: 32 for f in ("CROSS", "INTERVAL", "THRESHOLD", "TOPK")},
                "Original LEGACY family denominator differs",
            )
        sources = {
            name: dict(
                path=inventory[f"{panel}_{name}.jsonl"]["path"],
                sha256=inventory[f"{panel}_{name}.jsonl"]["sha256"],
            )
            for name in ("TASKS", "SOURCE_ROUTES")
        }
        for index, qid in enumerate(sorted(task_map)):
            task, route = task_map[qid], routes[qid]
            if panel == "LEGACY":
                _require(
                    all(
                        k in task
                        for k in (
                            "prompt",
                            "world",
                            "query",
                            "answer",
                            "root_id",
                            "family",
                            "variant",
                            "pool",
                            "chart",
                            "render_style",
                        )
                    ),
                    "Original LEGACY task schema incomplete",
                )
                _require(
                    route.get("chart") == task["chart"]
                    and route.get("render_style") == task["render_style"],
                    "LEGACY task/render association differs",
                )
                projected = {k: task[k] for k in ("qid", "pool", "chart", "prompt", "render_style")}
                safe = model_input_for(projected, route, protocol="structured")
                _require(
                    safe["prompt"] == task["prompt"],
                    "Frozen LEGACY prompt is not the original v2 interface",
                )
            else:
                _require(
                    all(k in task for k in ("question", "answer", "root_id", "image_sha256"))
                    and task.get("pool") == route.get("pool") == "CHARTQA"
                    and route.get("prompt") == PROMPT + "\n" + task["question"]
                    and task["image_sha256"] == route["image_sha256"],
                    "Original ChartQA task/prompt/image association differs",
                )
                safe = {k: route[k] for k in ("prompt", "image_file", "image_sha256")}
            rows.append(_row(panel, task, safe, sources, index))
        panels[panel] = dict(
            rows=rows,
            private_source_refs=[sources["TASKS"]],
            image_root_binding=dict(
                kind="ORIGINAL_F4_DATA",
                path=str(data_root),
                frozen=dict(
                    path=inventory["FROZEN.json"]["path"], sha256=inventory["FROZEN.json"]["sha256"]
                ),
            ),
        )
    return panels


def _profiles():
    common = dict(
        generation=GENERATION,
        raw_retention="Full original text/tokens/logprobs/finish; no repair or resampling",
    )
    profiles = {}
    for panel in PANELS:
        profile = copy.deepcopy(common)
        if panel == "NATURAL_INTERFACE":
            profile.update(
                protocol="natural_answer",
                assistant_prefill="",
                stop="EOS_OR_LENGTH_NO_ANSWER_LINE_STOP",
                scorer="srf6_ref.measurement.natural_answer_score",
                expected=(
                    "exact(frozen gold_answer), checked against world/query oracle; "
                    "never compare parsed exact to raw string"
                ),
                implementation_status="RUNTIME_ADAPTER_REQUIRED_NOT_IMPLEMENTED",
            )
        elif panel == "LEGACY":
            profile.update(
                protocol="structured",
                revision="v2",
                assistant_prefill="{",
                stop="ORIGINAL_STRUCTURED_BALANCED_JSON_OR_EOS_OR_LENGTH",
                scorer="sr_f4.task_contract.score",
                scorer_kwargs=dict(protocol="structured"),
                primary_field="value_correct",
                implementation_status="RUNTIME_ADAPTER_REQUIRED_NOT_IMPLEMENTED",
            )
        elif panel == "CHARTQA":
            profile.update(
                protocol="chartqa_plain",
                assistant_prefill="",
                stop="ORIGINAL_CHARTQA_EOS_OR_LENGTH",
                scorer="sr_f1.chartqa.score_chartqa",
                wrapper_source="sr_f3_ce.external.score_external",
                scoring_calls={
                    "historical_code": "score_chartqa(str(task['answer']), raw_text)",
                    "v2": "score_chartqa(str(task['answer']).strip(), raw_text.strip())",
                },
                metrics=["relaxed_accuracy", "exact_match"],
                typed_registry_required=False,
                implementation_status="F6_SAMPLED_ADAPTER_REQUIRED_OLD_EXTERNAL_GREEDY_FORBIDDEN",
            )
        else:
            profile.update(
                protocol="structured_v4",
                assistant_prefill="{",
                stop="F6_STRUCTURED_BALANCED_JSON_OR_EOS_OR_LENGTH",
                scorer="sr_f6.measurement.score_response",
                primary_field="A",
            )
        profiles[panel] = profile
    return profiles


def _endpoints(base):
    order = _read(base["root"], base["order_ref"])
    _require(
        len(order) == 54
        and {(r["parent"], r["repeat"], r["arm"]) for r in order}
        == {(p, r, a) for p in PARENTS for r in range(3) for a in ARMS},
        "All 54 formal paths required",
    )
    endpoints = []
    for stage, step, responses in (("DEV36", 36, 96), ("FINAL", 72, 6010)):
        for ordered in order:
            parent, repeat, arm = ordered["parent"], ordered["repeat"], ordered["arm"]
            directory = f"{formal.DIRECTORY}/runs/{parent}/r{repeat}/{arm}"
            endpoints.append(
                dict(
                    stage=stage,
                    endpoint_id=f"{stage}/{parent}/r{repeat}/{arm}",
                    kind="FUTURE_FORMAL_COMMIT_SELECTOR",
                    parent=parent,
                    repeat=repeat,
                    arm=arm,
                    lane="PRIMARY",
                    run_id=ordered["run_id"],
                    checkpoint_step=step,
                    expected_run_directory=directory,
                    expected_commit_path=f"{directory}/checkpoints/step-{step:03d}/COMMIT.json",
                    actual_commit_binding="REQUIRED_AFTER_PUBLICATION_BEFORE_MODEL_ACCESS",
                    access="REGISTERED_DEV36_ONLY"
                    if stage == "DEV36"
                    else "ALL_54_STEP72_PLUS_SEPARATE_TEST_GRANT",
                    responses=responses,
                )
            )
    for parent in PARENTS:
        endpoints.append(
            dict(
                stage="FINAL",
                endpoint_id=f"FINAL/{parent}/parent",
                kind="ORIGINAL_S0_PARENT",
                parent=parent,
                checkpoint_step=0,
                reference=base["parents"][parent]["runtime_identity"],
                tensor_sha256=base["parents"][parent]["tensor_sha256"],
                responses=6010,
                access="ALL_54_STEP72_PLUS_SEPARATE_TEST_GRANT",
            )
        )
    return endpoints


def build_formal_evaluation_plan(
    root, *, phase_context, pilot_spec_ref, retention_protocol_binding_ref
):
    """Return deterministic unpublished bytes; read-only CPU work grants no access."""
    base = _base(root, phase_context, pilot_spec_ref)
    _verify_protocol(base, retention_protocol_binding_ref)
    panels = {**_internal_panels(base), **_retention_panels(base)}
    endpoints, files, descriptions = _endpoints(base), {}, {}
    profiles = _profiles()
    for panel, content in panels.items():
        rows = content.pop("rows")
        payload = b"".join(_payload(row) for row in rows)
        relative = f"panels/{panel}.jsonl"
        files[relative] = payload
        questions, draws = PANELS[panel]
        descriptions[panel] = dict(
            questions=questions,
            draws=draws,
            responses=questions * draws,
            rows_ref=_reference(relative, payload),
            protocol_profile=profiles[panel],
            **content,
        )
    files["ENDPOINT_SELECTORS.jsonl"] = b"".join(_payload(item) for item in endpoints)
    budgets = dict(
        dev36_checkpoints=54,
        dev36_responses=5184,
        final_checkpoints=57,
        final_responses_per_checkpoint=6010,
        final_responses=342570,
        total_responses=347754,
        question_groups=2138,
        stored_draw_seeds=6106,
    )
    _require(
        sum(e["responses"] for e in endpoints) == budgets["total_responses"]
        and sum(v["questions"] for v in descriptions.values()) == 2138
        and sum(v["responses"] for v in descriptions.values()) == 6106,
        "Formal evaluation exact budgets differ",
    )
    identity = base["context"]["identity"]
    spec = dict(
        schema=SCHEMA,
        phase=PHASE,
        status="FROZEN_NOT_AUTHORIZED",
        execution_status="NOT_RUN",
        scope="EVALUATION_PLAN_ONLY_NO_EXECUTION_EVIDENCE",
        phase_identity=dict(
            path=f"PHASES/{PHASE}/IDENTITY.json", sha256=base["context"]["identity_sha256"]
        ),
        actual_source_manifest=identity["source_manifest"],
        origin_identity=identity["origin"]["identity"],
        config_sha256=identity["config_sha256"],
        data_manifest=identity["data_manifest"],
        source_binding=base["binding_ref"],
        pilot_parent_provenance=base["producer"],
        parent_references=base["pilot"]["parent_references"],
        parent_tensors=base["parents"],
        retention_protocol_binding=copy.deepcopy(retention_protocol_binding_ref),
        source_dependencies=base["source_refs"],
        order_source=base["order_ref"],
        endpoint_selectors=_reference(
            "ENDPOINT_SELECTORS.jsonl", files["ENDPOINT_SELECTORS.jsonl"]
        ),
        endpoint_count=111,
        panels=descriptions,
        budgets=budgets,
        seed_recipe=dict(
            master_seed=2026102701,
            implementation="srf6_ref.core.seed_for",
            dev36=["master", "formal_dev_eval", "DEV_EVAL", "qid", "draw"],
            final=["master", "formal_final_eval", "stream_panel", "stream_qid", "draw"],
            natural_pairing="stream_panel=ID; stream_qid=paired_id_qid",
            shared_across_all_endpoints=True,
            excluded=["parent", "repeat", "arm", "checkpoint", "lane", "task"],
        ),
        statistics_recipe=dict(
            config=base["config"]["evaluation"],
            retention=AMENDMENTS["retention"],
            source=base["source_refs"]["src/srf6_ref/statistics.py"],
            estimand="draw/qid/root-four-views/family/fixed-parent/repeat equal weighting",
            training_uncertainty="Within each fixed parent three paired repeats; Satterthwaite t",
            zero_training_variance="NA; Holm uses p=1 without deleting comparison",
            primary_tests="Four two-sided Holm alpha=.05; simultaneous 98.75% Bonferroni intervals",
            controls="U11-C_GLOBAL and U11-C_SHUF separate two-comparison Holm",
            bootstrap=(
                "10000; seed2026102731; within-parent paired repeats "
                "and shared within-family root indices"
            ),
            natural="Paired interface probe; not new independent roots or main endpoint",
        ),
        access=dict(
            dev36="REGISTERED_COMMIT_ONLY_DEV_EVAL_NO_EARLY_STOP_OR_SELECTION",
            final="ALL_54_STEP72_AND_SEPARATE_TEST_GRANT_REQUIRED_INCLUDING_THREE_PARENTS",
            test_grant_path="PHASES/FORMAL_V1/TEST_ACCESS_GRANT.json",
            training_grant_implies_test_access=False,
            endpoint_loader=(
                "Actual held RunTransactions; committed endpoint binding required; "
                "no optimizer/grad/RNG restoration"
            ),
            partial_unknown="Preserve all attempts; no resampling or seed replacement",
            independent_training_spec=True,
        ),
        **AUTHORIZATION,
    )
    files[SPEC_FILE] = _payload(spec)
    files["BINDING.json"] = _payload(
        dict(
            schema=SCHEMA,
            phase_identity=spec["phase_identity"],
            files=[_reference(name, value) for name, value in sorted(files.items())],
            **AUTHORIZATION,
        )
    )
    return dict(
        status="FROZEN_NOT_AUTHORIZED",
        published=False,
        directory=str(base["root"] / DIRECTORY),
        spec=spec,
        spec_sha256=hashlib.sha256(files[SPEC_FILE]).hexdigest(),
        files=files,
        **AUTHORIZATION,
    )


def _result(bundle, status):
    return dict(
        status=status,
        published=True,
        directory=bundle["directory"],
        spec=bundle["spec"],
        spec_sha256=bundle["spec_sha256"],
        binding_ref=_reference(f"{DIRECTORY}/BINDING.json", bundle["files"]["BINDING.json"]),
        frozen_files=[
            _reference(f"{DIRECTORY}/{name}", raw) for name, raw in sorted(bundle["files"].items())
        ],
        **AUTHORIZATION,
    )


def freeze_formal_evaluation_plan(
    root, *, phase_context, pilot_spec_ref, retention_protocol_binding_ref
):
    """Exclusive publication with BINDING last on Ceph; preserve every failed prefix."""
    root = _plain_path(root)
    target = review_package._directory(root, DIRECTORY)
    parent = review_package._directory(root, str(Path(DIRECTORY).parent))
    parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(parent / ".formal-evaluation.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bundle = build_formal_evaluation_plan(
            root,
            phase_context=phase_context,
            pilot_spec_ref=pilot_spec_ref,
            retention_protocol_binding_ref=retention_protocol_binding_ref,
        )
        files = bundle["files"]
        stage = review_package._directory(
            root, str(Path(DIRECTORY).parent / ".FORMAL_EVAL_FROZEN.staging")
        )
        if target.exists():
            review_package._compare(target, files)
            if stage.exists():
                review_package._linked_stage(stage, target, files)
        else:
            if stage.exists():
                review_package._compare(stage, files)
            else:
                stage.mkdir(mode=0o700)
                for relative, raw in files.items():
                    path = _path(stage, relative)
                    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    output = os.open(
                        path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600
                    )
                    with os.fdopen(output, "wb") as stream:
                        stream.write(raw)
                        stream.flush()
                        os.fsync(stream.fileno())
                for directory in sorted((p for p in stage.rglob("*") if p.is_dir()), reverse=True):
                    review_package._fsync_directory(directory)
                review_package._fsync_directory(stage)
                review_package._compare(stage, files)
            review_package._publish(stage, target)
        review_package._compare(target, files)
        for directory in (p for p in target.rglob("*") if p.is_dir()):
            review_package._fsync_directory(directory)
        review_package._fsync_directory(target)
        review_package._fsync_directory(parent)
        return _result(bundle, "FROZEN_NOT_AUTHORIZED")
    finally:
        os.close(fd)


def verify_formal_evaluation_plan(root, *, phase_context, expected_spec_sha256):
    """Reconstruct all IDs/seeds/sources; a partial publication is never consumed."""
    root = _plain_path(root)
    target = review_package._directory(root, DIRECTORY)
    review_package._require_commit(target)
    _require(_sha(expected_spec_sha256), "Explicit evaluation spec pin required")
    spec = _read(root, dict(path=f"{DIRECTORY}/{SPEC_FILE}", sha256=expected_spec_sha256))
    bundle = build_formal_evaluation_plan(
        root,
        phase_context=phase_context,
        pilot_spec_ref=spec.get("pilot_parent_provenance", {}).get("specification"),
        retention_protocol_binding_ref=spec.get("retention_protocol_binding"),
    )
    _require(bundle["spec_sha256"] == expected_spec_sha256, "Evaluation plan replay differs")
    review_package._compare(target, bundle["files"])
    return _result(bundle, "VERIFIED_PLAN_NOT_AUTHORIZED")
