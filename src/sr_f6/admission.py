"""Mixed-producer PILOT admission, recomputed from actual saved evidence.

S0 preflight/census, the 86a760b sampler recipe, and actual PILOT execution keep
their own identities. Assessment never authorizes. Sealing accepts references,
not a caller-written PASS report. Cold runtime verification repeats the CPU
facts; a process-local cache can only originate in that successful verification.
Unchanged files are reused; changed metadata is rehashed against the same bound
content. No persisted cache or status string is a scientific authority.

Independent HUMAN/MODEL_TEXT reviews require complete visible execution traces:
schema srf6-review-execution-trace-v1, ACTUAL_REVIEW, reviewer_id/type/identity,
instance_id/task_id, method/source refs, and ordered events. Each event contains
stage, started_at/completed_at, inputs [{reference, content}], and output
{reference, content}; content is the actual decoded submitted/returned artifact.
Extraction sees only its blind subset; judgment adds the sealed extraction and
world/query packet. Adjudication similarly binds its actual inputs and output.
This records actual visible task payloads, not inaccessible platform system
context. MODEL_TEXT may explicitly use model_identity=NOT_EXPOSED; actual task
and instance identity remain required. HUMAN must not invent model fields.
Hashes bind records, not the authenticity of a person's/platform's identity.
Known fabricated traces, gold leakage, oracle errors and false PASS block.

Before each real review/adjudication, create
PHASES/PILOT_V1/REVIEW_EXECUTIONS/<task_id>/ATTEMPT.json with ACTUAL_REVIEW,
phase_identity_sha256, package {path,sha256}, reviewer_id, task_id, instance_id,
and started_at. Finish with exactly one SUBMISSION.json or ADJUDICATION.json.
All executions are enumerated; orphan, unfinished and unresolved FAILURE.json
records block as technical UNKNOWN, not scientific infeasibility. Technical
recovery is not implemented here: preserve original attempts/outputs and
register explicit recovery before retrying; never redraw the 360 answers.
Adjudication uses input_references plus a separate decision_response artifact
containing {items,completed_at}. Its trace binds that response, avoiding a
circular hash through the final manifest's own reviewer.transcript reference.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from . import data, numerical_replay, phases, registration, review_audit, review_package
from .common import digest, file_hash, safe_evidence, write_once
from .source_binding import content_inventory, verify_bound_files

SCHEMA = "srf6-pilot-admission-v1"
PATH = "PHASES/PILOT_V1/ADMISSION.json"
SPEC_PATH = f"{registration.DIRECTORY}/{registration.SPEC_FILE}"
TRACE_SCHEMA = "srf6-review-execution-trace-v1"
REVIEW_DIRECTORY = "PHASES/PILOT_V1/REVIEW_EXECUTIONS"
_CACHE = {}


class _IncompleteEvidence(PermissionError):
    """Unresolved execution/transport evidence, distinct from scientific failure."""


def _require(value, message):
    if not value:
        raise PermissionError(message)


def _ref(value):
    _require(
        isinstance(value, dict) and set(value) == {"path", "sha256"}, "Explicit reference required"
    )
    _require(
        isinstance(value["path"], str) and re.fullmatch(r"[a-f0-9]{64}", value.get("sha256", "")),
        "Full content pin required",
    )
    return value


def _synthetic(value):
    if isinstance(value, dict):
        return any(
            (
                key in ("scope", "execution_origin", "status")
                and isinstance(item, str)
                and item.startswith("SYNTHETIC")
            )
            or _synthetic(item)
            for key, item in value.items()
        )
    return isinstance(value, list) and any(_synthetic(item) for item in value)


class _Facts:
    def __init__(self, root):
        self.root = Path(root)
        _require(
            self.root.is_absolute() and self.root.resolve(strict=True) == self.root,
            "Canonical run root required",
        )
        self.inputs, self.external, self.synthetic = {}, {}, False

    def read(self, relative, *, reference=None, lines=False, opaque=False):
        candidate = self.root / relative
        if not candidate.exists() and not candidate.is_symlink():
            raise FileNotFoundError("Required admission evidence missing: " + str(relative))
        path = safe_evidence(self.root, relative)
        before = _stat(path)
        raw = path.read_bytes()
        _require(before == _stat(path), "Admission input changed while reading")
        ref = dict(path=str(relative), sha256=hashlib.sha256(raw).hexdigest())
        if reference is not None:
            _require(ref == _ref(reference), "Admission input content differs")
        self.adopt([ref])
        if opaque:
            return raw
        value = (
            [review_package._json(line) for line in raw.splitlines() if line.strip()]
            if lines
            else review_package._json(raw)
        )
        self.synthetic |= _synthetic(value)
        return value

    def bound(self, reference, *, lines=False):
        return self.read(_ref(reference)["path"], reference=reference, lines=lines)

    def adopt(self, refs):
        for ref in refs:
            _ref(ref)
            _require(
                self.inputs.setdefault(ref["path"], dict(ref)) == ref,
                "Conflicting admission evidence identities",
            )

    def outside(self, path, sha256, *, lines=False):
        path = Path(path)
        _require(
            path.is_absolute() and path.resolve(strict=True).is_relative_to(phases.PROJECT),
            "External evidence escapes registered project",
        )
        raw = path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == sha256, "External evidence content differs")
        self.external[str(path)] = dict(
            path=str(path), resolved_path=str(path.resolve()), sha256=sha256
        )
        value = (
            [review_package._json(line) for line in raw.splitlines() if line.strip()]
            if lines
            else review_package._json(raw)
        )
        self.synthetic |= _synthetic(value)
        return value


def _stat(path):
    path = Path(path)
    value = path.stat()
    return (
        str(path.resolve(strict=True)),
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
    )


def _phase(facts, pin):
    from .runtime import _verify_module_origins

    deployment = Path(__file__).resolve(strict=True).parents[2]
    result = phases.verify_phase(
        facts.root, "PILOT_V1", deployment=deployment, expected_identity_sha256=pin
    )
    _require(
        result.get("phase") == "PILOT_V1"
        and result.get("identity_sha256") == pin
        and result.get("deployment") == str(deployment),
        "Actual PILOT executing source required",
    )
    _require(
        result.get("status") == "BOUND_NOT_AUTHORIZED" and result.get("training_allowed") is False,
        "Provenance is not admission",
    )
    identity = facts.bound(dict(path="PHASES/PILOT_V1/IDENTITY.json", sha256=pin))
    _require(identity == result["identity"], "Actual phase identity differs")
    manifest = facts.bound(identity["source_manifest"])
    _verify_module_origins(deployment, manifest, tuple(sys.modules.values()))
    origin = facts.bound(identity["origin"]["identity"])
    _require(
        origin["source_manifest"] != identity["source_manifest"], "PILOT cannot claim S0 source"
    )
    old_manifest = facts.bound(origin["source_manifest"])
    data_manifest = facts.bound(origin["data_manifest"])
    facts.adopt([{k: item[k] for k in ("path", "sha256")} for item in data_manifest["files"]])
    for base, source in (
        (deployment, manifest),
        (Path(identity["origin"]["deployment"]), old_manifest),
    ):
        for item in source["files"]:
            path = base / item["path"]
            facts.external[str(path)] = dict(
                path=str(path), resolved_path=str(path), sha256=item["sha256"]
            )
    return result, origin


def _inventory(facts, items):
    verify_bound_files(items, boundary=phases.PROJECT)
    for item in items:
        facts.external[item["path"]] = {
            key: item[key] for key in ("path", "resolved_path", "sha256")
        }


def _source(facts, phase, origin, config):
    from .gates import PREFLIGHT_REQUIRED, _evidence

    ready = facts.read("PREFLIGHT/CENSUS_READY.json")
    _require(
        ready.get("status") == "READY_FOR_CENSUS"
        and ready.get("phase") == "DEV_CAL"
        and ready.get("context") == origin
        and ready.get("execution_origin") == "SERVER_ACTUAL"
        and ready.get("optimizer_updates") == 0,
        "Original actual S0 census preflight required",
    )
    _require(
        any(
            entry.get("producer_phase") == "CENSUS_S0"
            and entry.get("identity_sha256") == phase["identity"]["origin"]["identity"]["sha256"]
            and facts.inputs["PREFLIGHT/CENSUS_READY.json"] in entry.get("evidence", [])
            for entry in phase["identity"]["upstream"]
        ),
        "Original preflight must be a frozen PILOT upstream",
    )
    _evidence(facts.root, ready["evidence"], PREFLIGHT_REQUIRED, origin)
    _collect(facts, facts.inputs["PREFLIGHT/CENSUS_READY.json"])
    source = facts.read("SOURCE_BINDING.json")
    _require(
        source.get("execution_origin") == "SERVER_ACTUAL"
        and source.get("phase") == "SOURCE_BINDING",
        "Actual S0 source inventory required",
    )
    _require(
        source.get("config_file_sha256") == origin["config_sha256"]
        and source.get("config_canonical_sha256") == digest(config),
        "Source configuration differs",
    )
    _require(
        source.get("optimizer_inheritance") is False
        and source.get("model_calls") == source.get("optimizer_updates") == 0,
        "S0 binding must not inherit optimizer or run model",
    )
    _require(
        set(source.get("parents", {})) == {"b0", "b1", "b2"},
        "Three fixed parent inventories required",
    )
    _require(
        source.get("host") and source.get("python") and source.get("versions"),
        "Actual environment facts required",
    )
    for name, version in source["versions"].items():
        _require(
            version and importlib.metadata.version(name) == version,
            "Current dependency environment differs from S0: " + name,
        )
    from .source_binding import SOURCE_FILES

    inherited = source.get("inherited_deployment_files", [])
    _require(
        {r["relative"] for r in inherited} == set(SOURCE_FILES)
        and len(inherited) == len(SOURCE_FILES),
        "Complete inherited native source inventory required",
    )
    _require(
        digest(inherited) == source["source_content_sha256"],
        "Native source inventory digest differs",
    )
    _inventory(facts, source["source_files"])
    for row in inherited:
        path = Path(phase["identity"]["origin"]["deployment"]) / row["relative"]
        _require(
            str(path) == row["path"] and file_hash(path) == row["sha256"],
            "Actual historical inherited source differs",
        )
        facts.external[str(path)] = dict(
            path=str(path), resolved_path=str(path), sha256=row["sha256"]
        )
    _inventory(facts, source["model_files"])
    _require(
        digest(content_inventory(source["model_files"])) == source["model_content_sha256"],
        "Model content inventory differs",
    )
    names = {Path(row["path"]).name for row in source["model_files"]}
    _require(
        {"config.json", "tokenizer_config.json"} <= names
        and any(n.endswith(".safetensors") for n in names),
        "Incomplete model/tokenizer files",
    )
    _require(
        all(
            Path(row["path"]).is_relative_to(config["model"]["model_path"])
            for row in source["model_files"]
        ),
        "Wrong model snapshot path",
    )
    for parent, item in source["parents"].items():
        _require(item["path"] == config["model"]["parents"][parent], "Parent path substituted")
        _require(
            {Path(r["path"]).name for r in item["files"]}
            == {"adapter_config.json", "adapter_model.safetensors"},
            "Parent adapter file set differs",
        )
        _inventory(facts, item["files"])
        _require(
            digest(content_inventory(item["files"])) == item["content_sha256"],
            "Parent content identity differs",
        )
        cfg = next(row for row in item["files"] if row["path"].endswith("adapter_config.json"))
        _require(
            facts.outside(cfg["path"], cfg["sha256"]) == item["lora_config"],
            "Parent LoRA configuration differs",
        )
    _inventory(facts, [source["previous_config"], *source["retention"].values()])
    for panel, count in (("LEGACY", 128), ("CHARTQA", 250)):
        rows = source["retention_images"][panel]
        _require(
            len(rows) == len({r["qid"] for r in rows}) == count,
            "Retention image/ID coverage differs",
        )
        _inventory(facts, rows)
        ref = source["retention"][panel + "_TASKS.jsonl"]
        tasks = facts.outside(ref["path"], ref["sha256"], lines=True)
        _require({r["qid"] for r in tasks} == {r["qid"] for r in rows}, "Retention task IDs differ")
    return dict(
        status="PASS",
        producer="CENSUS_S0",
        source_binding_status=source["status"],
        content_identity_only=True,
    )


def _data(facts, config):
    roots = facts.read("DATA/ROOT_CERTIFICATES_PRIVATE.jsonl", lines=True)
    banks = defaultdict(list)
    for root in roots:
        banks[root["tasks"]["P0"][0]["split"]].append(root)
    actual = data.audit_banks(dict(banks), config)
    _require(
        actual == facts.read("DATA/CPU_AUDIT.json"),
        "Recomputed dual-oracle/data audit differs from S0 receipt",
    )
    all_tasks = []
    for split, bank in banks.items():
        tasks = facts.read(f"DATA/{split}/TASKS_PRIVATE.jsonl", lines=True)
        _require(tasks == data.all_tasks(bank), "Saved private tasks differ from audited root bank")
        public = facts.read(f"DATA/{split}/MODEL_INPUTS.jsonl", lines=True)
        _require(len(public) == len(tasks), "Public/private task count differs")
        for item, task in zip(public, tasks, strict=True):
            model = item["model_input"]
            _require(
                set(model) == {"prompt", "image_file", "image_sha256"}
                and model["prompt"] == data.public_prompt(task),
                "Gold-blind public input differs",
            )
            _require(
                item["qid"] == task["qid"]
                and model["image_file"] == f"DATA/images/{digest(task['world'])}.png",
                "Public image/task lineage differs",
            )
            facts.read(
                model["image_file"],
                reference=dict(path=model["image_file"], sha256=model["image_sha256"]),
                opaque=True,
            )
        all_tasks.extend(tasks)
    natural = facts.read("DATA/NATURAL_INTERFACE/TASKS_PRIVATE.jsonl", lines=True)
    _require(
        natural == data.natural_interface_tasks(banks["TEST_ID"], config["master_seed"]),
        "Natural-interface task source differs",
    )
    all_tasks.extend(natural)
    return dict(
        status="PASS",
        producer="CENSUS_S0",
        oracle_comparisons=actual["question_oracle_comparisons"],
        independent_oracles=actual["independent_native_oracles"],
        question_rows=len(all_tasks),
    ), all_tasks


def _historical(facts, tasks):
    from srf6_ref.core import fingerprint

    source = facts.read("SOURCE_BINDING.json")
    report = facts.read("PREFLIGHT/HISTORICAL_WORLD_AUDIT.json")
    previous = Path(source["previous_run"])
    expected = [
        previous / "DATA/TASKS_PRIVATE.jsonl",
        previous.parent / "sr_f4_c_local_pilot_20261024/DATA/TASKS.jsonl",
        previous.parent / "sr_f4_consolidated_20261022/DATA/TASKS.jsonl",
    ]
    _require(
        [r["path"] for r in report["sources"]] == [str(p) for p in expected],
        "Historical source list differs",
    )
    prior = set()
    for ref in report["sources"]:
        rows = facts.outside(ref["path"], ref["sha256"], lines=True)
        worlds = [r["world"] for r in rows if r.get("world") is not None]
        _require(len(worlds) == ref["world_rows"] > 0, "Historical source count differs")
        _require(
            all(r.get("world") is not None or r.get("pool") == "CHARTQA" for r in rows),
            "Unclassified historical row",
        )
        prior.update(fingerprint(w) for w in worlds)
    _require(not any(fingerprint(t["world"]) in prior for t in tasks), "Historical world overlap")
    _require(
        report.get("status") == "PASS"
        and report.get("collisions") == []
        and report.get("checked_question_rows") == len(tasks)
        and report.get("model_calls") == 0,
        "Historical receipt contradicts facts",
    )
    return dict(status="PASS", producer="CENSUS_S0", checked_question_rows=len(tasks))


def _processor(facts, origin):
    from PIL import Image

    from srf6_ref.core import FAMILIES

    report = facts.read("PREFLIGHT/PROCESSOR.json")
    visual = facts.read("PREFLIGHT/VISUAL_REVIEW.json")
    _require(
        report.get("execution_origin") == "SERVER_ACTUAL"
        and report.get("config_file_sha256") == origin["config_sha256"]
        and report.get("data_manifest_sha256") == origin["data_manifest"]["sha256"],
        "Actual processor input binding differs",
    )
    images = report.get("images", [])
    plan = facts.read("DATA/PROCESSOR_REVIEW_PLAN.json")["items"]
    _require(
        len(images) == 48
        and {r["qid"] for r in images} == {r["qid"] for r in plan}
        and Counter(r["family"] for r in images) == Counter(dict.fromkeys(FAMILIES, 8)),
        "48 planned processor images required",
    )
    plan = {row["qid"]: row for row in plan}
    identity = report["identity"]
    _require(
        identity.get("architecture_verified") is True
        and identity.get("model_type") == "qwen3_5"
        and identity.get("runtime_recipe") == "SR-F6-native-full-sequence-1536",
        "Actual processor architecture/recipe differs",
    )
    generation = identity.get("generation_config_expanded", {})
    for key, expected in (
        ("max_new_tokens", 1536),
        ("do_sample", True),
        ("num_beams", 1),
        ("temperature", 1.0),
        ("top_p", 1.0),
        ("top_k", 0),
        ("repetition_penalty", 1.0),
        ("remove_invalid_values", False),
    ):
        _require(generation.get(key) == expected, "Actual generation recipe differs: " + key)
    reviews = visual.get("items", [])
    _require(
        visual.get("status") == "PASS"
        and visual.get("reviewer_type") in ("HUMAN", "MODEL_VISUAL")
        and visual.get("method")
        and visual.get("processor_receipt_sha256")
        == facts.inputs["PREFLIGHT/PROCESSOR.json"]["sha256"]
        and len(reviews) == 48
        and len({r["qid"] for r in reviews}) == 48,
        "Actual visual review is incomplete",
    )
    reviews = {r["qid"]: r for r in reviews}
    for item in images:
        task, review = plan[item["qid"]], reviews[item["qid"]]
        _require(
            item["source_file"] == task["image_file"]
            and item["source_sha256"] == task["image_sha256"],
            "Processor source image differs",
        )
        for path, sha in (
            (item["source_file"], item["source_sha256"]),
            (item["processed_file"], item["processed_file_sha256"]),
        ):
            facts.read(path, reference=dict(path=path, sha256=sha), opaque=True)
        route = item["routing"]
        _require(
            route.get("source_image_sha256") == item["source_sha256"]
            and route.get("processor_hash") == identity["processor_hash"]
            and route.get("chat_template_hash") == identity["chat_template_hash"]
            and route.get("native_input_tensor_hashes")
            and route.get("input_float_dtype") == route.get("probability_dtype") == "float32"
            and route.get("sr_f6_protocol_version") == "structured_v4-srf6-1.0",
            "Actual processor routing facts differ",
        )
        hashes = route["native_input_tensor_hashes"]
        _require(
            set(hashes)
            == {
                "input_ids",
                "attention_mask",
                "mm_token_type_ids",
                "pixel_values",
                "image_grid_thw",
            }
            and all(re.fullmatch(r"[a-f0-9]{64}", value) for value in hashes.values()),
            "Complete actual native tensor hashes required",
        )
        with Image.open(safe_evidence(facts.root, item["source_file"])) as image:
            _require(
                image.size == (1280, 896) and list(image.size) == route.get("source_size"),
                "Actual original image dimensions differ",
            )
        with Image.open(safe_evidence(facts.root, item["processed_file"])) as image:
            _require(
                list(image.size) == route.get("processed_size")
                and hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
                == item.get("processed_rgb_sha256"),
                "Actual processor reconstruction pixels/dimensions differ",
            )
        grid = route.get("image_grid_thw")
        _require(
            isinstance(grid, list)
            and len(grid) == 1
            and len(grid[0]) == 3
            and grid[0][0] == 1
            and all(type(n) is int and n > 0 for n in grid[0]),
            "Native image grid required",
        )
        _require(
            route["processed_size"] == [16 * grid[0][2], 16 * grid[0][1]]
            and route.get("image_token_count") * 4 == grid[0][1] * grid[0][2],
            "Native image grid/token alignment differs",
        )
        _require(
            review.get("status") == "PASS"
            and review.get("observation")
            and review.get("reviewed_source") is True
            and review.get("reviewed_processed") is True
            and review.get("issues") == []
            and review.get("source_sha256") == item["source_sha256"]
            and review.get("processed_file_sha256") == item["processed_file_sha256"],
            "Unresolved/unviewed visual image",
        )
    fixtures = report.get("tokenizer_fixtures", [])
    _tokenizer_roundtrips(facts, fixtures)
    return dict(
        status="PASS",
        producer="CENSUS_S0",
        images=48,
        reviewer_type=visual["reviewer_type"],
        tokenizer_fixtures=len(fixtures),
    )


def _tokenizer_roundtrips(facts, fixtures):
    from transformers import AutoTokenizer

    _require(
        len(fixtures) >= 3
        and all(
            r.get("exact_roundtrip") is True
            and isinstance(r.get("text"), str)
            and isinstance(r.get("token_ids"), list)
            and bool(r["token_ids"])
            for r in fixtures
        ),
        "Real tokenizer roundtrip facts incomplete",
    )
    config = facts.read("CONFIG.json")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["model_path"], local_files_only=True, trust_remote_code=False
    )
    for item in fixtures:
        _require(
            tokenizer.encode(item["text"], add_special_tokens=False) == item["token_ids"]
            and tokenizer.decode(
                item["token_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            == item["text"],
            "Actual bound tokenizer roundtrip differs",
        )


def _native(facts, origin):
    from sr_f5.measurement import score_response as native_score
    from srf6_ref.core import FAMILIES
    from srf6_ref.measurement import full_truth_response

    from .measurement import score_response

    report = facts.read("PREFLIGHT/NATIVE_REGRESSION.json")
    log = facts.read("PREFLIGHT/NATIVE_REGRESSION.log", opaque=True)
    _require(
        report.get("status") == "PASS"
        and report.get("returncode") == 0
        and report.get("model_calls") == report.get("optimizer_updates") == 0
        and report.get("config_sha256") == origin["config_sha256"]
        and report.get("source_manifest_sha256") == origin["source_manifest"]["sha256"]
        and hashlib.sha256(log).hexdigest() == report.get("log_sha256"),
        "S0 native regression facts differ",
    )
    _require(
        "tests/sr_f6/test_measurement.py" in report.get("command", [])
        and re.search(rb"\b[1-9][0-9]* passed\b", log)
        and not re.search(rb"\b[1-9][0-9]* (failed|error|skipped)\b", log),
        "Actual completed native regression log required",
    )
    _require(
        review_audit._time(report["started_at"]) <= review_audit._time(report["completed_at"]),
        "Native regression chronology differs",
    )
    tasks = facts.read("DATA/DEV_CAL/TASKS_PRIVATE.jsonl", lines=True)
    counts = Counter()
    for family in FAMILIES:
        selected = [task for task in tasks if task["family"] == family][:50]
        _require(len(selected) == 50, "50 native J fixtures per family required")
        for index, task in enumerate(selected):
            obj = json.loads(full_truth_response(task["world"], task["query"]))
            if index % 4 == 1:
                obj["answer"] = "9999"
            elif index % 4 == 2:
                obj["evidence"] = []
            elif index % 4 == 3:
                obj["evidence"][0]["value"] = "18" if obj["evidence"][0]["value"] == "12" else "12"
            native = native_score(
                json.dumps({k: v for k, v in obj.items() if k not in ("bindings", "aggregate")}),
                task["world"],
                task["query"],
            )
            score = score_response(json.dumps(obj), task["world"], task["query"])
            _require(score["V"] == 1 and score["J"] == native["J"], "Native J parity failure")
            for key, value in (
                ("selected", ["unlisted-category", "unlisted-category"]),
                ("bindings", [dict(category="unlisted-category", operand=999)]),
            ):
                mutant = score_response(
                    json.dumps(dict(obj, **{key: value})), task["world"], task["query"]
                )
                _require(
                    mutant["V"] == 1 and mutant["J"] == native["J"],
                    "Semantic-field mutation changed native J",
                )
            counts[family] += 1
    return dict(
        status="PASS",
        producer="CENSUS_S0",
        recheck_execution="PILOT_V1",
        native_fixtures=sum(counts.values()),
        valid_shape_mutations=2 * sum(counts.values()),
        family_counts=dict(counts),
    )


def _visible(facts, reference):
    return facts.bound(reference, lines=reference["path"].endswith(".jsonl"))


def _trace(facts, submission, package, *, adjudication=False):
    reviewer = submission["reviewer"]
    _require(
        submission.get("execution_origin") == "ACTUAL_REVIEW"
        and reviewer.get("type") in ("HUMAN", "MODEL_TEXT"),
        "Actual independent HUMAN/MODEL_TEXT execution required",
    )
    trace = facts.bound(reviewer["transcript"])
    source = facts.bound(reviewer["source"])
    facts.read(reviewer["method"]["path"], reference=reviewer["method"], opaque=True)
    _require(
        trace.get("schema") == TRACE_SCHEMA and trace.get("execution_origin") == "ACTUAL_REVIEW",
        "Complete typed actual execution trace required",
    )
    for key, expected in (
        ("reviewer_id", reviewer["id"]),
        ("reviewer_type", reviewer["type"]),
        ("identity", reviewer["identity"]),
        ("model_identity", reviewer.get("model_identity")),
        ("method", reviewer["method"]),
        ("source", reviewer["source"]),
    ):
        _require(
            trace.get(key) == expected, "Actual trace reviewer/method identity differs: " + key
        )
    for key in ("instance_id", "task_id"):
        _require(
            isinstance(trace.get(key), str)
            and bool(trace[key].strip())
            and source.get(key) == trace[key],
            "Actual review instance/task identity required",
        )
    _require(
        source.get("reviewer_type") == reviewer["type"]
        and source.get("execution_origin") == "ACTUAL_REVIEW",
        "Reviewer source is not an actual independent execution",
    )
    if reviewer["type"] == "HUMAN":
        _require(trace.get("model_identity") is None, "Human trace cannot invent a model identity")
    else:
        _require(
            bool(trace.get("model_identity")),
            "Actual model identity or explicit NOT_EXPOSED required",
        )
    if adjudication:
        response = facts.bound(submission["decision_response"])
        _require(
            response == {k: submission[k] for k in ("items", "completed_at")},
            "Actual adjudication response differs from final manifest",
        )
        stages = [("ADJUDICATION", submission["input_references"], submission["decision_response"])]
    else:
        stages = [
            ("BLIND_EXTRACTION", [submission["blind_input"]], submission["blind_extraction"]),
            (
                "INDEPENDENT_JUDGMENT",
                [
                    submission["blind_input"],
                    package["judgment_tasks"],
                    submission["blind_extraction"],
                ],
                submission["judgments"],
            ),
        ]
    events = trace.get("events", [])
    _require(len(events) == len(stages), "Complete visible execution stage trace required")
    previous = None
    for event, (stage, inputs, output) in zip(events, stages, strict=True):
        _require(
            set(event) == {"stage", "started_at", "completed_at", "inputs", "output"}
            and event["stage"] == stage,
            "Unregistered review trace stage/fields",
        )
        started, completed = (
            review_audit._time(event["started_at"]),
            review_audit._time(event["completed_at"]),
        )
        _require(
            started <= completed and (previous is None or previous < started),
            "Review execution chronology differs",
        )
        previous = completed
        expected = [dict(reference=ref, content=_visible(facts, ref)) for ref in inputs]
        _require(
            event["inputs"] == expected,
            "Trace inputs omit/change the actual payload or expose extra material",
        )
        actual_output = _visible(facts, output)
        _require(
            event["output"] == dict(reference=output, content=actual_output),
            "Trace output differs from actual recorded response",
        )
        _require(
            actual_output.get("completed_at") == event["completed_at"],
            "Trace completion differs from sealed response",
        )
        if not adjudication and stage == "INDEPENDENT_JUDGMENT":
            _require(
                actual_output.get("started_at") == event["started_at"],
                "Judgment began before its recorded actual request",
            )
    return dict(
        reviewer_id=reviewer["id"],
        reviewer_type=reviewer["type"],
        instance_id=trace["instance_id"],
        task_id=trace["task_id"],
        transcript=reviewer["transcript"],
        visible_payloads_verified=True,
    )


def _review_inventory(facts, package_ref, refs, adjudication_refs, traces, phase_pin):
    directory = facts.root / REVIEW_DIRECTORY
    if not directory.exists():
        raise _IncompleteEvidence("Actual review execution inventory missing")
    _require(directory.is_dir() and not directory.is_symlink(), "Unsafe review execution inventory")
    supplied = {ref["path"] for ref in [*refs, *adjudication_refs]}
    completed, instances = set(), set()
    by_task = {row["task_id"]: row for row in traces}
    _require(
        len(by_task) == len(traces),
        "Distinct actual review task identity required for each execution",
    )
    for child in sorted(directory.iterdir()):
        _require(child.is_dir() and not child.is_symlink(), "Unexpected review execution entry")
        relative = child.relative_to(facts.root).as_posix()
        attempt = facts.read(relative + "/ATTEMPT.json")
        _require(
            attempt.get("task_id") == child.name
            and attempt.get("phase_identity_sha256") == phase_pin
            and attempt.get("package") == package_ref
            and attempt.get("execution_origin") == "ACTUAL_REVIEW",
            "Review ATTEMPT differs from fixed package/phase",
        )
        if (child / "FAILURE.json").exists():
            facts.read(relative + "/FAILURE.json")
            raise _IncompleteEvidence(
                "Unresolved technical review failure; recovery is not implemented"
            )
        candidates = [
            child / name
            for name in ("SUBMISSION.json", "ADJUDICATION.json")
            if (child / name).exists()
        ]
        if len(candidates) != 1 or child.name not in by_task:
            raise _IncompleteEvidence("Incomplete or unreported review execution")
        trace = by_task[child.name]
        _require(
            attempt.get("instance_id") == trace["instance_id"]
            and attempt.get("reviewer_id") == trace["reviewer_id"],
            "Review instance differs from pre-execution binding",
        )
        trace_value = facts.bound(trace["transcript"])
        _require(
            review_audit._time(attempt["started_at"])
            <= review_audit._time(trace_value["events"][0]["started_at"]),
            "Review ATTEMPT was not recorded before execution",
        )
        instances.add(trace["instance_id"])
        completed.add(candidates[0].relative_to(facts.root).as_posix())
    _require(completed == supplied, "All actual review/adjudication submissions must be retained")
    return dict(executions=len(completed), instances=len(instances))


def _measurement(facts, phase, request):
    package = facts.bound(request["package_ref"])
    _require(
        package.get("schema") == review_package.SCHEMA,
        "Full-census-replayed v2 package required for admission",
    )
    result = review_audit.aggregate_review(
        facts.root,
        phase_deployment=phase["deployment"],
        phase_identity_sha256=phase["identity_sha256"],
        package_ref=request["package_ref"],
        review_refs=request["review_refs"],
        adjudication_refs=request["adjudication_refs"],
    )
    facts.adopt(result["inputs"])
    facts.synthetic |= _synthetic(result)
    support = facts.bound(package["support"])
    _require(
        result.get("selection_provenance") == "RECOMPUTED_FROM_VERIFIED_FULL_CENSUS"
        and result.get("context") == phase["identity"],
        "Review selection/producer not verified",
    )
    traces = []
    for ref in request["review_refs"]:
        traces.append(_trace(facts, facts.bound(ref), package))
    for ref in request["adjudication_refs"]:
        submission = facts.bound(ref)
        traces.append(_trace(facts, dict(submission, _reference=ref), package, adjudication=True))
    inventory = _review_inventory(
        facts,
        request["package_ref"],
        request["review_refs"],
        request["adjudication_refs"],
        traces,
        phase["identity_sha256"],
    )
    good = (
        result["workflow_state"] == "REVIEW_COMPLETE_NOT_GATE_SIGNED"
        and result["independently_reviewed_original_answers"] == 360
        and not result["blocking_finding_ids"]
        and not result["adjudication_required_ids"]
        and not result["missing_independent_review_ids"]
    )
    return dict(
        status="PASS"
        if good
        else "UNKNOWN"
        if result["workflow_state"]
        in ("REVIEW_NOT_RUN", "REVIEW_INCOMPLETE", "ADJUDICATION_REQUIRED")
        else "FAIL",
        audit=result,
        support=support,
        actual_traces=traces,
        execution_inventory=inventory,
        limitation=(
            "Bound visible execution records; hashes do not authenticate "
            "platform/human identity or hidden system context"
        ),
    )


def _collect(facts, reference, seen=None):
    seen = set() if seen is None else seen
    if reference["path"] in seen:
        return
    seen.add(reference["path"])
    value = facts.read(
        reference["path"], reference=reference, opaque=not reference["path"].endswith(".json")
    )
    if isinstance(value, dict):
        for key in ("inputs", "evidence"):
            for ref in value.get(key, []):
                _collect(facts, {k: ref[k] for k in ("path", "sha256")}, seen)


def _numerical(facts, phase, request):
    result = numerical_replay.summary(
        facts.root,
        phase_identity_sha256=phase["identity_sha256"],
        replay_spec_sha256=request["replay_spec_sha256"],
        receipts=request["numerical_receipts"],
    )
    _require(result.get("scope") == "SERVER_ACTUAL", "Real three-parent T03 required")
    for ref in request["numerical_receipts"]:
        _collect(facts, ref)
    directory = facts.root / "PHASE_RUNS/PILOT_V1/REPLAY_RUNS"
    if directory.exists():
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                facts.read(path.relative_to(facts.root).as_posix(), opaque=True)
    return result


def _registration(facts, phase, spec_pin):
    result = registration.verify_pilot_registration(
        facts.root, phase_context=phase, expected_spec_sha256=spec_pin
    )
    _require(
        result.get("status") == "VERIFIED_REGISTRATION_NOT_AUTHORIZED"
        and result.get("spec_sha256") == spec_pin,
        "Registered pilot plan required",
    )
    for item in result["spec"]["paths"]:
        for key in ("transaction_identity", "rollout_registration"):
            ref = item[key]
            facts.bound(dict(path=f"{registration.DIRECTORY}/{ref['path']}", sha256=ref["sha256"]))
    for reference in result["spec"]["parent_references"].values():
        facts.bound(reference)
    _collect(facts, dict(path=SPEC_PATH, sha256=spec_pin))
    return result


def _loaded(facts, phase):
    from .runtime import _verify_module_origins

    _require(
        phase["deployment"] == str(Path(__file__).resolve(strict=True).parents[2]),
        "Admission cannot execute another deployment's code",
    )
    manifest = facts.bound(phase["identity"]["source_manifest"])
    _verify_module_origins(Path(phase["deployment"]), manifest, tuple(sys.modules.values()))
    return dict(status="PASS")


def assess_pilot_admission(
    root,
    *,
    phase_identity_sha256,
    pilot_spec_sha256,
    package_ref,
    review_refs,
    replay_spec_sha256,
    numerical_receipts,
    adjudication_refs=(),
):
    """Recompute facts without writing, sampling, running GPU, or authorizing."""
    facts = _Facts(root)
    request = dict(
        phase_identity_sha256=phase_identity_sha256,
        pilot_spec_sha256=pilot_spec_sha256,
        package_ref=copy.deepcopy(package_ref),
        review_refs=list(copy.deepcopy(review_refs)),
        adjudication_refs=list(copy.deepcopy(adjudication_refs)),
        replay_spec_sha256=replay_spec_sha256,
        numerical_receipts=list(copy.deepcopy(numerical_receipts)),
    )
    checks = {}
    phase = origin = registered = None

    def check(name, fn):
        try:
            value = fn()
            checks[name] = (
                value if isinstance(value, dict) and "status" in value else dict(status="PASS")
            )
            facts.synthetic |= _synthetic(value)
            return value
        except (OSError, KeyError, ValueError, TypeError, ImportError, PermissionError) as exc:
            checks[name] = dict(
                status="UNKNOWN"
                if isinstance(exc, (FileNotFoundError, ImportError, _IncompleteEvidence))
                else "FAIL",
                error=type(exc).__name__ + ": " + str(exc),
            )
            return None

    bound = check("actual_phase", lambda: _phase(facts, phase_identity_sha256))
    if bound:
        phase, origin = bound
        config = check(
            "config", lambda: facts.bound(dict(path="CONFIG.json", sha256=origin["config_sha256"]))
        )
        if config:
            check("source", lambda: _source(facts, phase, origin, config))
            audited = check("data", lambda: _data(facts, config))
            if audited:
                checks["data"] = audited[0]
                check("historical", lambda: _historical(facts, audited[1]))
            check("processor_visual", lambda: _processor(facts, origin))
            check("native_j", lambda: _native(facts, origin))
        measurement = check("measurement", lambda: _measurement(facts, phase, request))
        if measurement:
            checks["support"] = measurement["support"]
        check("numerical", lambda: _numerical(facts, phase, request))
        registered = check("registration", lambda: _registration(facts, phase, pilot_spec_sha256))
        if registered:
            checks["registration"] = dict(
                status="PASS",
                registration_status=registered["status"],
                spec_sha256=pilot_spec_sha256,
            )
        check("loaded_dependencies", lambda: _loaded(facts, phase))
    required = {
        "actual_phase",
        "source",
        "data",
        "historical",
        "processor_visual",
        "native_j",
        "measurement",
        "support",
        "numerical",
        "registration",
        "loaded_dependencies",
    }
    for name in required - checks.keys():
        checks[name] = dict(status="UNKNOWN", error="Required upstream evidence unavailable")
    states = [checks[name]["status"] for name in sorted(required)]
    scientific = (
        "PASS"
        if all(s == "PASS" for s in states)
        else "FAIL"
        if any(s in ("FAIL", "INVALID_SUPPORT_EVIDENCE") for s in states)
        else "INSUFFICIENT_REGISTERED_EXPOSURE"
        if "INSUFFICIENT_REGISTERED_EXPOSURE" in states
        else "UNKNOWN"
    )
    return dict(
        schema=SCHEMA,
        status="SYNTHETIC_ONLY" if facts.synthetic else scientific,
        scientific_status=scientific,
        execution_origin="SYNTHETIC_TEST" if facts.synthetic else "ACTUAL_EVIDENCE_VALIDATION",
        scope="PILOT_V1_REGISTERED_ONLY",
        training_authorized=False,
        generation_authorized=False,
        gate_status="NOT_SIGNED",
        context=phase["identity"] if phase else None,
        phase_context=phase,
        request=request,
        checks=checks,
        inputs=[facts.inputs[p] for p in sorted(facts.inputs)],
        external_inputs=[facts.external[p] for p in sorted(facts.external)],
        pilot_spec=registered["spec"] if registered else None,
        limitations=[
            "Not a formal-training authorization",
            "No significance or performance threshold applied",
            "Hashes alone do not establish authentic platform/human execution records",
        ],
    )


def _receipt(root, assessment):
    _require(
        assessment["status"] == "PASS"
        and assessment["execution_origin"] == "ACTUAL_EVIDENCE_VALIDATION",
        "Only complete actual recomputed PASS may be sealed",
    )
    return dict(
        schema=SCHEMA,
        status="PASS",
        scope="PILOT_V1_REGISTERED_ONLY",
        training_authorized=True,
        generation_authorized=True,
        gate_status="SIGNED_FROM_RECOMPUTED_FACTS",
        context=assessment["context"],
        request=assessment["request"],
        assessment_sha256=digest(assessment),
        inputs=assessment["inputs"],
        external_inputs=assessment["external_inputs"],
        execution_origin="ACTUAL_EVIDENCE_VALIDATION",
        root=str(Path(root)),
        limitation=(
            "Registered pilot only; independent review identities are recorded, "
            "not cryptographically authenticated"
        ),
    )


def seal_pilot_admission(root, **request):
    """Only freshly recomputed actual PASS can publish the fixed immutable receipt."""
    assessment = assess_pilot_admission(root, **request)
    if assessment["status"] != "PASS":
        assessment_ref = None
        if assessment["phase_context"] is not None:
            relative = f"PHASES/PILOT_V1/ADMISSION_ASSESSMENTS/{digest(assessment)}.json"
            parent = review_package._directory(root, "PHASES/PILOT_V1/ADMISSION_ASSESSMENTS")
            parent.mkdir(mode=0o700, exist_ok=True)
            write_once(Path(root) / relative, assessment)
            (Path(root) / relative).chmod(0o600)
            assessment_ref = dict(path=relative, sha256=file_hash(Path(root) / relative))
        return dict(
            status=assessment["status"],
            training_authorized=False,
            generation_authorized=False,
            gate_status="NOT_SIGNED",
            assessment=assessment,
            assessment_ref=assessment_ref,
        )
    receipt = _receipt(root, assessment)
    write_once(Path(root) / PATH, receipt)
    ref = dict(path=PATH, sha256=file_hash(Path(root) / PATH))
    result = _authorized(receipt, ref, assessment)
    _remember(root, result, assessment)
    return result


def _authorized(receipt, ref, assessment):
    return dict(
        status="VERIFIED_PILOT_ADMISSION",
        scope="PILOT_V1_REGISTERED_ONLY",
        training_authorized=True,
        generation_authorized=True,
        phase_context=assessment["phase_context"],
        admission_ref=ref,
        receipt=receipt,
        pilot_spec_ref=dict(path=SPEC_PATH, sha256=assessment["request"]["pilot_spec_sha256"]),
        pilot_spec=assessment["pilot_spec"],
    )


def _trees(root, phase_context=None):
    roots = (
        "CENSUS/workers",
        "CENSUS/raw",
        "CENSUS/groups",
        "CENSUS/attempts",
        "CENSUS/alignment",
        "PHASE_RUNS/PILOT_V1/REPLAY_RUNS",
        REVIEW_DIRECTORY,
    )
    result = {
        relative: tuple(
            sorted(
                str(p.relative_to(root))
                for p in (root / relative).rglob("*")
                if p.is_file() or p.is_symlink()
            )
        )
        for relative in roots
    }
    # Training outputs live below the frozen arm directories; they are not
    # admission prerequisites and must not invalidate admission after step zero.
    runs = root / registration.DIRECTORY / "runs"
    result["registration_layout"] = tuple(
        sorted(
            str(path.relative_to(root))
            for pattern in ("*", "*/*", "*/*/*")
            for path in runs.glob(pattern)
        )
    )
    if phase_context is not None:
        for deployment in (
            phase_context["deployment"],
            phase_context["identity"]["origin"]["deployment"],
        ):
            base = Path(deployment)
            for relative in ("src", "tests/sr_f6", "scripts/sr_f6"):
                directory = base / relative
                result[str(directory)] = tuple(
                    sorted(
                        str(p)
                        for p in directory.rglob("*")
                        if (p.is_file() or p.is_symlink()) and p.suffix in (".py", ".sh", ".sbatch")
                    )
                )
    return result


def _key(root, pin, ref, spec):
    return (os.getpid(), str(Path(root)), pin, ref["sha256"], spec)


def _remember(root, result, assessment):
    _require(
        assessment["status"] == "PASS"
        and assessment["execution_origin"] == "ACTUAL_EVIDENCE_VALIDATION"
        and result == _authorized(_receipt(root, assessment), result["admission_ref"], assessment),
        "Only recomputed actual authorization may populate admission cache",
    )
    root = Path(root)
    refs = [*assessment["inputs"], result["admission_ref"]]
    files = {
        str(root / r["path"]): dict(sha256=r["sha256"], stat=_stat(root / r["path"])) for r in refs
    }
    for ref in assessment["external_inputs"]:
        files[ref["path"]] = dict(sha256=ref["sha256"], stat=_stat(ref["path"]))
    _CACHE[
        _key(
            root,
            result["phase_context"]["identity_sha256"],
            result["admission_ref"],
            result["pilot_spec_ref"]["sha256"],
        )
    ] = dict(result=copy.deepcopy(result), files=files, trees=_trees(root, result["phase_context"]))


def _cache_unchanged(root, cached):
    if _trees(root, cached["result"]["phase_context"]) != cached["trees"]:
        return False
    for name, item in cached["files"].items():
        current = _stat(name)
        _require(current[0] == item["stat"][0], "Admission evidence link/target changed")
        if current != item["stat"]:
            _require(
                file_hash(name) == item["sha256"] and _stat(name) == current,
                "Admission evidence changed after verification",
            )
            item["stat"] = current
    return True


def require_pilot_admission(root, *, phase_identity_sha256, admission_ref, pilot_spec_sha256):
    """Runtime boundary: cold revalidation, then content-bound process-local reuse."""
    _ref(admission_ref)
    _require(admission_ref["path"] == PATH, "Only the fixed PILOT admission receipt is allowed")
    root = Path(root)
    key = _key(root, phase_identity_sha256, admission_ref, pilot_spec_sha256)
    cached = _CACHE.get(key)
    if cached is not None and _cache_unchanged(root, cached):
        _loaded(_Facts(root), cached["result"]["phase_context"])
        return copy.deepcopy(cached["result"])
    facts = _Facts(root)
    receipt = facts.bound(admission_ref)
    _require(
        receipt.get("schema") == SCHEMA
        and receipt.get("request", {}).get("phase_identity_sha256") == phase_identity_sha256
        and receipt["request"].get("pilot_spec_sha256") == pilot_spec_sha256,
        "Admission phase/spec pins differ",
    )
    assessment = assess_pilot_admission(root, **receipt["request"])
    _require(
        assessment["status"] == "PASS", "Pilot admission facts do not pass: " + assessment["status"]
    )
    _require(
        receipt == _receipt(root, assessment),
        "Handwritten or stale admission receipt differs from actual facts",
    )
    result = _authorized(receipt, admission_ref, assessment)
    _remember(root, result, assessment)
    return result
