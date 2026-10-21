"""Portable, data-only revalidation using the frozen production scoring code.

This entry point does not construct a Controller, load tensors, contact Slurm or
run inference. Original scheduler/checkpoint receipts remain historical evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

SCIENCE_COMMIT = "652e52d530520a276b6420ff0f4ee479dcde6db1"
SOURCE_MANIFEST_SHA = "dae3a506dda4fffee8da0e25a3e0881f4b45fa3259342c4160ecd0c09e03c9e0"
BUILDER_COMMIT = "35800b02b1ef1eff2378e4c933393b4ea2668e4d"
BUILDER_SHA = "d525f83a81851fc1537a18d7823198224acd77b12e6e160380e368ebf69dcfc4"
WEIGHTS = {".pt", ".pth", ".bin", ".safetensors", ".ckpt"}


def _fail(message):
    raise PermissionError(message)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            _fail("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _read(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=lambda value: _fail("Nonfinite JSON: " + value),
    )


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def _bounded(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        _fail("Expected a relative artifact path")
    if any(part in {".", ".."} for part in relative.split("/")):
        _fail("Unsafe artifact path")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        _fail("Artifact escapes the local input root")
    return path


def _source_api(root):
    manifest_path = root / "deployment/SOURCE_MANIFEST.json"
    if _sha(manifest_path) != SOURCE_MANIFEST_SHA:
        _fail("Scientific source manifest differs from the registered run")
    if _read(root / "RUN_IDENTITY.json")["code_commit"] != SCIENCE_COMMIT:
        _fail("Scientific code commit differs")
    records = _read(manifest_path)["files"]
    if not records or len({row["path"] for row in records}) != len(records):
        _fail("Scientific source inventory is ambiguous")
    code = (root / "code").resolve(strict=True)
    for row in records:
        path = _bounded(code, row["path"])
        if path.suffix.lower() in WEIGHTS:
            _fail("Data-only verifier cannot read a source tensor artifact")
        if path.stat().st_size != row["bytes"] or _sha(path) != row["sha256"]:
            _fail("Scientific source changed: " + row["path"])
    if any(p.is_file() for p in code.rglob("*.py[co]")):
        _fail("Unbound scientific bytecode must not be imported")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code / "src"))
    modules = {}
    for name in ("data", "evaluation", "external", "contract"):
        module = importlib.import_module("sr_f3_ce." + name)
        if Path(module.__file__).resolve() != code / "src/sr_f3_ce" / (name + ".py"):
            _fail("Imported scoring source belongs to another checkout")
        modules[name] = module
    path = root / "operations" / BUILDER_COMMIT / "build_delivery.py"
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != BUILDER_SHA:
        _fail("Original delivery verifier source changed")
    spec = importlib.util.spec_from_file_location("frozen_data_delivery", path)
    builder = importlib.util.module_from_spec(spec)
    exec(compile(body, str(path), "exec"), builder.__dict__)
    return SimpleNamespace(**modules), builder


class _ExternalInputs:
    """Explicit path relocation only; frozen prompts and image hashes stay intact."""

    def __init__(self, root, directory, freeze, inputs, original_root):
        self.inputs = inputs
        self.protocol = freeze.get("protocol", "chartqapro_official")
        source = freeze.get("image_root")
        if source is None:
            self.image_root = directory
        else:
            source = Path(source)
            if not source.is_absolute() or not source.is_relative_to(original_root):
                _fail("External image root is outside the registered experiment")
            self.image_root = (root / source.relative_to(original_root)).resolve(strict=True)
        if not self.image_root.is_relative_to(root):
            _fail("Relocated external images escape local inputs")

    def model_input(self, qid, protocol=None):
        if protocol is not None and protocol != self.protocol:
            _fail("External protocol changed")
        row = self.inputs[qid]
        path = _bounded(self.image_root, row["image_file"])
        if _sha(path) != row["image_sha256"]:
            _fail("External source image changed")
        return {key: row[key] for key in ("prompt", "image_file", "image_sha256")}


def _external(root, api):
    panels = _read(root / "external/PANELS.json")
    if (
        panels.get("status") != "FROZEN"
        or panels.get("experiment_id") != api.evaluation.EXPERIMENT_ID
    ):
        _fail("External panels are not frozen")
    tasks, stores, registries = {}, {}, {}
    official = None
    original_root = Path(_read(root / "RUN_IDENTITY.json")["root"])
    for pool in ("CHARTQA", "CHARTQAPRO"):
        item = panels[pool]
        path = _bounded(root, item["root"] + "/FREEZE.json")
        if _sha(path) != item["freeze_sha256"]:
            _fail("External freeze changed")
        directory, freeze = path.parent, _read(path)
        if pool == "CHARTQAPRO" and freeze["status"] == "UNAVAILABLE":
            _fail("This delivery requires all 233340 released records including ChartQAPro")
        if freeze.get("status") != "FROZEN":
            _fail("External inputs not frozen")
        inputs_path = directory / "MODEL_INPUTS.jsonl"
        tasks_path = directory / "TASKS_AUDIT_ONLY.jsonl"
        if (
            _sha(inputs_path) != freeze["inputs_sha256"]
            or _sha(tasks_path) != freeze["audit_tasks_sha256"]
        ):
            _fail("External sidecar identity changed")
        inputs = {}
        for row in api.data.read_jsonl(inputs_path):
            if row["qid"] in inputs:
                _fail("Duplicate external input")
            inputs[row["qid"]] = row
        registry = _read(directory / "TYPE_REGISTRY.json")
        if api.evaluation.object_hash(registry) != freeze["type_registry_sha256"]:
            _fail("External type registry changed")
        registries[pool] = registry
        pool_tasks = api.data.read_jsonl(tasks_path)
        if len(inputs) != freeze["image_count"] or len(pool_tasks) != len(inputs):
            _fail("External task coverage differs")
        for row in pool_tasks:
            if row["qid"] in tasks or row["qid"] not in inputs or row["pool"] != pool:
                _fail("External task identity differs")
            tasks[row["qid"]] = row
        stores[pool] = _ExternalInputs(root, directory, freeze, inputs, original_root)
        if pool == "CHARTQAPRO":
            official = api.external.load_official_scorer(
                directory / "sources/evaluate_predictions.py",
                directory / "sources/anls-0.0.2-py3-none-any.whl",
            )

    def scorer(row, task):
        return api.external.score_external(
            row, task, type_registry=registries[task["pool"]], official_scorer=official
        )

    return tasks, stores, scorer


def _endpoint_groups(root, slots, api, store, external_stores):
    """The frozen _raw_groups validation, with explicit portable input stores."""
    evaluation = api.evaluation
    groups = evaluation.group_slots(slots)
    expected = {group["group_id"]: group for group in groups}
    if len(expected) != len(groups):
        _fail("Duplicate expected endpoint group")
    found = set()
    source_sha = _sha(root / "RUN_IDENTITY.json")
    panels_sha = _sha(root / "external/PANELS.json")
    for model in evaluation.endpoint_models():
        directory = root / "sealed/endpoint" / model["model_id"]
        manifest, provenance = [
            _read(directory / name) for name in ("RAW_MANIFEST.json", "MODEL_IDENTITY.json")
        ]
        if (
            manifest.get("status") != "SEALED_COMPLETE"
            or manifest.get("model_id") != model["model_id"]
            or provenance.get("model") != model
            or manifest["model_identity_sha256"] != _sha(directory / "MODEL_IDENTITY.json")
            or manifest["external_panels_sha256"] != panels_sha
        ):
            _fail("Sealed endpoint identity differs")
        for entry in manifest["raw_groups"]:
            gid = entry["group_id"]
            if gid in found or gid not in expected:
                _fail("Duplicate or unregistered endpoint group")
            path = _bounded(root, entry["path"])
            if _sha(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
                _fail("Original endpoint group changed")
            value, group = _read(path), expected[gid]
            rows = value["records"]
            if (
                group["slots"][0]["model_id"] != model["model_id"]
                or value["group_id"] != gid
                or value["records_sha256"] != evaluation.object_hash(rows)
                or len(rows) != len(group["slots"])
            ):
                _fail("Endpoint raw group identity/coverage differs")
            for index, (row, slot) in enumerate(zip(rows, group["slots"], strict=True)):
                current = external_stores.get(slot["pool"], store)
                request = current.model_input(slot["qid"], protocol=slot["protocol"])
                if row.get("model_input") != request:
                    _fail("Endpoint input differs from frozen prompt/image")
                evaluation.validate_record(
                    row,
                    slot,
                    input_hash=evaluation.object_hash(request),
                    model_identity=provenance["model_identity"],
                    adapter_identity=provenance["adapter_identity"],
                    source_sha256=source_sha,
                    index=index,
                    count=len(rows),
                )
            found.add(gid)
            yield group, rows, entry
    if found != set(expected):
        _fail("Missing registered endpoint groups")


def _training_group(value, qid, task, request, api, builder):
    rows = value["records"]
    if len(rows) != 8 or value.get("records_sha256") != api.evaluation.object_hash(rows):
        _fail("Training raw group count/checksum differs")
    for index, row in enumerate(rows):
        builder.raw_fields(row)
        if row["model_input"] != request:
            _fail("Training input differs from frozen input")
        if row.get("qid") != qid:
            _fail("Training task identity changed")
        if (
            row.get("generation_status") != "COMPLETE"
            or row.get("technical_validation_errors")
            or row.get("group_row_index") != index
            or len(row["tokens"]) != len(row.get("sampler_logprobs", []))
        ):
            _fail("Training technical trajectory or sample alignment differs")
        actual = api.contract.score(
            row["full_text"],
            task["world"],
            task["query"],
            finish_reason=row["finish_reason"],
            protocol="structured",
        )
        if row["score"] != actual or row["reward"] != actual["J"]:
            _fail("Training production rescore differs")


def _training(root, dataset, store, api, builder):
    counts = dict(training_paths=0, training_groups=0, training_records=0, support_examples=0)
    for run in dataset.runs:
        directory = root / "runs" / run["run_id"]
        expected, support_expected = set(), set()
        schedule = dataset.schedules[run["block"]]
        if len(schedule) != 64 or {item["step"] for item in schedule} != set(range(1, 65)):
            _fail("Training schedule differs from fixed 64 steps")
        for item in schedule:
            if len(item["rl_qids"]) != 16:
                _fail("Training schedule must have 16 groups per update")
            for slot, qid in enumerate(item["rl_qids"]):
                name = f"{item['step']:02d}-{slot:02d}.json"
                expected.add(name)
                path = directory / "rollouts" / name
                value = _read(path)
                request = _read(path.with_suffix(".request.json"))
                model_input = store.model_input(qid)
                expected_identity = dict(
                    step=item["step"],
                    slot=slot,
                    qid=qid,
                    input_hash=api.evaluation.object_hash(model_input),
                    task_hash=api.evaluation.object_hash(dataset.tasks[qid]),
                )
                if value["identity"] != request or any(
                    request.get(key) != expected for key, expected in expected_identity.items()
                ):
                    _fail("Training request identity differs from the frozen schedule")
                _training_group(
                    value,
                    qid,
                    dataset.tasks[qid],
                    model_input,
                    api,
                    builder,
                )
                counts["training_groups"] += 1
                counts["training_records"] += 8
            if run["arm"] != "RL":
                if len(item["support_qids"]) != 4:
                    _fail("Support schedule must contain four examples per update")
                for qid in item["support_qids"]:
                    name = f"{item['step']:02d}-{qid}.json"
                    support_expected.add(name)
                    audit = _read(directory / "support" / name)
                    if any(
                        audit.get(key) != value
                        for key, value in dict(
                            qid=qid,
                            step=item["step"],
                            arm=run["arm"],
                            protocol_revision=dataset.audit["protocol_revision"],
                        ).items()
                    ):
                        _fail("Support identity differs from the frozen schedule")
                    native = _read(root / "native/support_masks" / f"{qid}-{run['arm']}.json")
                    if any(
                        audit[k] != native[k]
                        for k in ("tokens", "answer_mask", "work_mask", "mask_audit")
                    ):
                        _fail("Support tokens/masks differ from native audit")
                    if audit["target"] != native["original_target"]:
                        _fail("Support teacher differs from frozen target")
                    counts["support_examples"] += 1
        actual = {path.name for path in (directory / "rollouts").glob("[0-9][0-9]-[0-9][0-9].json")}
        if actual != expected:
            _fail("Training group inventory differs")
        actual_support = {path.name for path in (directory / "support").glob("[0-9][0-9]-*.json")}
        if actual_support != support_expected:
            _fail("Support group inventory differs")
        counts["training_paths"] += 1
    if counts != dict(
        training_paths=12, training_groups=12288, training_records=98304, support_examples=2304
    ):
        _fail("Full training/support raw-data coverage differs")
    return counts


def _verify_in_process(root):
    root = Path(root).resolve(strict=True)
    api, builder = _source_api(root)
    dataset = api.data.load_frozen_dataset(root / "plan")
    store = api.data.ModelInputStore.from_files(
        root / "plan/manifests/MODEL_INPUTS.jsonl",
        root / "data/IMAGE_MANIFEST.jsonl",
        image_root=root / "data",
    )
    external_tasks, external_stores, external_scorer = _external(root, api)
    if set(dataset.tasks) & set(external_tasks):
        _fail("Synthetic and external task identities overlap")
    tasks = dict(dataset.tasks, **external_tasks)
    revision = _read(root / "plan/config/SR_F3_CE.json").get("protocol_revision", "v1")
    slots = api.evaluation.endpoint_slots(
        dataset.tasks, dataset.runs, external_tasks, revision=revision
    )
    if len(slots) != 233340:
        _fail("Fixed 15-model endpoint slot coverage differs")
    coverage = builder.verify_release_stream(
        root,
        _endpoint_groups(root, slots, api, store, external_stores),
        tasks,
        lambda row, task: api.evaluation.score_record(row, task, external_scorer=external_scorer),
    )
    coverage.update(_training(root, dataset, store, api, builder))
    monitor, engine = builder.collect_panels(root, api, dataset, store)
    coverage.update(
        evaluated_models=15,
        monitor_records=sum(item[2] for item in monitor),
        engine_diagnostic_records=sum(item[2] for item in engine),
        protocol_revision=revision,
    )
    signals = builder.training_signal_index(root, dataset.runs)
    return dict(
        schema_version=1,
        status="PASS",
        scope="PORTABLE_RESULTS_DATA_ONLY",
        records=coverage["records"],
        coverage=coverage,
        mismatches=[],
        checks=dict(
            frozen_plan="PASS",
            frozen_production_scoring_source="PASS",
            raw_rescoring_chain="PASS",
            training_raw_production_rescoring="PASS",
            support_raw_masks="PASS",
            monitor_and_engine_raw_coverage="PASS",
            training_signal_inventory="PASS",
        ),
        scorer_code_identity=dict(
            scientific_commit=SCIENCE_COMMIT,
            source_manifest_sha256=SOURCE_MANIFEST_SHA,
            original_builder_sha256=BUILDER_SHA,
            local_verifier_sha256=_sha(Path(__file__)),
        ),
        source_receipts={
            name: _sha(root / name)
            for name in (
                "RUN_IDENTITY.json",
                "EXECUTION_FREEZE.json",
                "release/MANIFEST.json",
                "release/RELEASE.json",
                "release/RELEASE_SCORED.json",
                "release/RAW_RESCORING.json",
                "release/ENDPOINT_RAW.jsonl.gz",
                "release/ENDPOINT_SCORES.jsonl",
                "release/PRIMARY_ANALYSIS.json",
                "release/EXTERNAL_ANALYSIS.json",
                "release/PREFIX_DIAGNOSTICS.json",
            )
        },
        training_signals_sha256=builder.digest(signals),
        boundaries=dict(
            new_model_calls=0,
            tensor_or_checkpoint_payloads_read=0,
            fresh_checkpoint_verification=False,
            fresh_scheduler_verification=False,
            original_seven_check_full_delivery_claimed=False,
            statistical_analysis=(
                "Existing released analysis retained with verified file hashes; not recomputed here"
            ),
            scheduler_and_checkpoint_receipts=(
                "Retained historical evidence, not newly certified locally"
            ),
        ),
    )


def verify_results(root):
    """Isolate imports so an existing checkout cannot replace frozen scorer code."""
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--child",
            str(Path(root).resolve(strict=True)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("Local data rescoring failed:\n" + result.stderr[-16000:])
    receipt = json.loads(result.stdout, object_pairs_hook=_pairs)
    if receipt.get("status") != "PASS" or receipt.get("records") != 233340:
        _fail("Local data rescoring returned incomplete acceptance")
    return receipt


def rescore(root, transport_manifest=None):
    """Compatibility API; the packager independently authenticates transport."""
    return verify_results(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("root")
    args = parser.parse_args()
    value = _verify_in_process(args.root) if args.child else verify_results(args.root)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))
