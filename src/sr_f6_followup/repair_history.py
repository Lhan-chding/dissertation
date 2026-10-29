"""Preserve the failed v1 review as pinned historical evidence, never authority.

No old consumer module is imported and no producer identity is rewritten. Only
the small historical review/source closure is content-checked on a cold call;
the producer's model dependencies remain the responsibility of source.py.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from . import source

ROOT = source.PROJECT / "sr_f6_confirm_followup_v1_20261029"
DEPLOYMENT = source.PROJECT / "sr_f6_followup_code_02fbf89"
REVISION = "02fbf8968d33f5858f7668ba5a38b1fc314b52c1"
PINS = {
    "REGISTRATION.json": "0c0595f7e79b129a4ff50683558005d52543dc98d85749896b53252e99301fd8",
    "SOURCE_MANIFEST.json": "0c1469dc3a998ca3ac5830ef6c788e4929e92d9f0af3af8d2aa95b2915828b7c",
    "REVIEW/PACKAGE/BINDING.json": (
        "83c830d3f898b728a3aee95971d6c8d50fd1b31db930ce0ed51982beb6c4b727"
    ),
    "REVIEW/PACKAGE/private/selection.json": (
        "ad7c4a141e1a428ce50179646fef1c5170a26f860ca698282763c69161e18545"
    ),
    "REVIEW/PACKAGE/blind/EXTRACTION.jsonl": (
        "1e65dd1728c21915a054af78bbf80891376193fb16fe4836ebc425ecfce394c3"
    ),
    "REVIEW/PACKAGE/judgment/TASKS.jsonl": (
        "8ce005ca996a7213633018fb4fe10ea8fd4cb979e41e66951e931e2d223a97f1"
    ),
    "REVIEW/AUDIT/SUMMARY.json": "1d858e6ae8ddc0e432df18f31fdd303e7c4a2d4779b380a90bb45c6b4ba59430",
    "REVIEW/AUDIT/BINDING.json": "b0f92a37a18a9fcfae53b91b69d94d0757f7240b27ae23f2a0c8733b135bb8b7",
}
_CACHE = None


def _membership():
    """Detect extra, replaced, symlinked, or removed historical review objects."""
    directory = source._path(ROOT / "REVIEW")
    result = []
    for path in [directory, *sorted(directory.rglob("*"))]:
        source._path(path)
        source._require(path.is_file() or path.is_dir(), "Unsupported historical review object")
        result.append((str(path), source._stamp(path.stat())))
    return result


def _relative(reference):
    source._require(
        isinstance(reference, dict) and set(reference) == {"path", "sha256"},
        "Exact historical reference required",
    )
    relative = Path(reference["path"])
    source._require(
        not relative.is_absolute() and ".." not in relative.parts,
        "Historical relative reference escapes its root",
    )
    return dict(path=str(source._path(ROOT / relative)), sha256=reference["sha256"])


def verify():
    """Return copied, independently pinned v1 facts with unchanged-file caching."""
    global _CACHE
    source._path(ROOT)
    source._path(DEPLOYMENT)
    source._require(ROOT != source.ROOT, "Historical root cannot be the new consumer root")
    membership = _membership()
    if _CACHE is not None:
        source._require(membership == _CACHE[0], "Historical review closure changed")
        source._unchanged(_CACHE[1])
        source._require(
            {str(path) for path in source._source_paths(DEPLOYMENT)}
            == {
                ref["path"]
                for ref in _CACHE[2]["references"]
                if Path(ref["path"]).is_relative_to(DEPLOYMENT)
            },
            "Historical source inventory changed",
        )
        return copy.deepcopy(_CACHE[2])

    from sr_f6.review_package import _compare, _linked_stage

    references, payloads, states = {}, {}, {}

    def bound(reference, *, lines=False, opaque=False):
        path = source._path(reference["path"])
        name = str(path)
        source._require(
            references.setdefault(name, reference) == reference,
            "Conflicting historical content reference",
        )
        if name not in payloads:
            state = source._state(path)
            payloads[name] = path.read_bytes()
            source._require(
                hashlib.sha256(payloads[name]).hexdigest() == reference["sha256"],
                "Historical evidence content changed",
            )
            source._require(source._state(path) == state, "Historical evidence changed while read")
            states[name] = dict(**state, sha256=reference["sha256"])
        data = payloads[name]
        if opaque:
            return data
        return [json.loads(line) for line in data.splitlines()] if lines else json.loads(data)

    fixed = {relative: _relative(dict(path=relative, sha256=pin)) for relative, pin in PINS.items()}
    for reference in fixed.values():
        bound(reference, opaque=True)
    registration = bound(fixed["REGISTRATION.json"])
    manifest = bound(fixed["SOURCE_MANIFEST.json"])
    package = bound(fixed["REVIEW/PACKAGE/BINDING.json"])
    summary = bound(fixed["REVIEW/AUDIT/SUMMARY.json"])
    seal = bound(fixed["REVIEW/AUDIT/BINDING.json"])
    consumer = dict(
        deployment=str(DEPLOYMENT),
        revision=REVISION,
        source_manifest_sha256=PINS["SOURCE_MANIFEST.json"],
        registration_sha256=PINS["REGISTRATION.json"],
    )
    source._require(
        registration.get("schema") == "srf6-followup-registration-v1"
        and registration.get("root") == str(ROOT)
        and registration.get("deployment") == str(DEPLOYMENT)
        and registration.get("revision") == REVISION
        and registration.get("status") == "BOUND_NOT_AUTHORIZED"
        and type(registration.get("model_calls")) is int
        and registration["model_calls"] == 0
        and type(registration.get("optimizer_updates")) is int
        and registration["optimizer_updates"] == 0
        and registration.get("source_manifest") == fixed["SOURCE_MANIFEST.json"]
        and all(
            registration.get(key) is False
            for key in ("training_allowed", "generation_allowed", "evaluation_allowed")
        ),
        "Historical consumer registration identity differs",
    )
    producer = registration["producer"]
    source._require(
        producer.get("root") == str(source.PRODUCER_ROOT)
        and producer.get("deployment") == str(source.PRODUCER_DEPLOYMENT)
        and producer.get("revision") == source.PRODUCER_REVISION
        and producer.get("registration_sha256") == source.PRODUCER_REGISTRATION_SHA256,
        "Historical review has a different actual producer",
    )
    source._require(
        manifest.get("schema") == "srf6-followup-source-v1"
        and manifest.get("deployment") == str(DEPLOYMENT)
        and manifest.get("revision") == REVISION,
        "Historical source manifest identity differs",
    )
    files = manifest["files"]
    source._require(
        len(files) == len({item["path"] for item in files})
        and {item["path"] for item in files}
        == {str(path) for path in source._source_paths(DEPLOYMENT)},
        "Historical source inventory differs",
    )
    for item in files:
        bound({key: item[key] for key in ("path", "sha256")}, opaque=True)
        source._require(
            {key: item[key] for key in item if key != "sha256"}
            == {key: states[item["path"]][key] for key in states[item["path"]] if key != "sha256"},
            "Historical deployed source changed after registration",
        )
    source._require(
        registration["spec"]["path"] == str(ROOT / "SPEC.json"),
        "Historical specification path differs",
    )
    bound(registration["spec"])
    for value in (package, summary, seal):
        source._require(value.get("consumer") == consumer, "Historical consumer identity differs")
        actual_producer = value.get("input_producer", {})
        source._require(
            actual_producer.get("phase") == "SCHEMA_CONFIRM_V3"
            and actual_producer.get("deployment") == str(source.PRODUCER_DEPLOYMENT)
            and actual_producer.get("revision") == source.PRODUCER_REVISION
            and actual_producer.get("source_manifest", {}).get("sha256")
            == producer["source_manifest_sha256"],
            "Historical input producer identity differs",
        )
    source._require(
        package.get("schema") == "srf6-followup-review-package-v1"
        and package.get("phase") == "SCHEMA_CONFIRM_V3"
        and package.get("split") == "DEV_SCHEMA_CONFIRM"
        and package.get("status") == "BLIND_PACKAGE_ONLY"
        and package.get("model_calls") == package.get("optimizer_updates") == 0
        and all(
            package.get(key) is False
            for key in ("training_allowed", "generation_allowed", "evaluation_allowed")
        )
        and all(
            _relative(package[key]) == fixed[relative]
            for key, relative in (
                ("ledger", "REVIEW/PACKAGE/private/selection.json"),
                ("blind", "REVIEW/PACKAGE/blind/EXTRACTION.jsonl"),
                ("judgment_tasks", "REVIEW/PACKAGE/judgment/TASKS.jsonl"),
            )
        ),
        "Historical frozen review package differs",
    )
    source._require(
        summary.get("schema") == "srf6-followup-independent-review-audit-v1"
        and summary.get("status") == "BLOCKING_FINDINGS"
        and summary.get("completed") is False
        and summary.get("synthetic") is False
        and summary.get("gate_status") == "NOT_SIGNED"
        and summary.get("training_allowed") is False
        and summary.get("reviewed_original_answers") == 360
        and summary.get("independently_reviewed_original_answers") == 360
        and len(summary.get("blocking_finding_ids", [])) == 8
        and len(summary.get("adjudication_required_ids", [])) == 8,
        "The terminal historical failure must remain unchanged",
    )
    source._require(
        seal
        == dict(
            schema="srf6-followup-review-seal-v1",
            summary=dict(
                path="REVIEW/AUDIT/SUMMARY.json", sha256=PINS["REVIEW/AUDIT/SUMMARY.json"]
            ),
            consumer=consumer,
            input_producer=summary["input_producer"],
            package=dict(
                path="REVIEW/PACKAGE/BINDING.json", sha256=PINS["REVIEW/PACKAGE/BINDING.json"]
            ),
            status="BLOCKING_FINDINGS",
            completed=False,
            training_allowed=False,
            gate_status="NOT_SIGNED",
        )
        and summary.get("package_ref") == seal["package"],
        "Historical terminal publication differs",
    )
    for reference in [*package["inputs"], *summary["inputs"]]:
        bound(_relative(reference), opaque=True)
    execution_directory = source._path(ROOT / "REVIEW/EXECUTIONS")
    source._require(
        {str(path) for path in execution_directory.rglob("*") if path.is_file()}
        == {name for name in references if Path(name).is_relative_to(execution_directory)},
        "Historical execution membership differs from the terminal closure",
    )
    failure = ROOT / "REVIEW/.audit-failure.json"
    source._require(
        not (failure.exists() or failure.is_symlink()), "Historical failed seal cannot be ignored"
    )
    # The original seal-attempt is retained even though the pre-seal summary
    # necessarily predates its final publication references.
    attempt_path = source._path(ROOT / "REVIEW/.audit-attempt.json")
    attempt = bound(source._reference(attempt_path))
    source._require(
        attempt.get("schema") == "srf6-followup-review-seal-attempt-v1",
        "Historical seal attempt is missing or invalid",
    )
    for directory in ("REVIEW/PACKAGE", "REVIEW/AUDIT"):
        target = source._path(ROOT / directory)
        expected = {
            str(Path(name).relative_to(target)): data
            for name, data in payloads.items()
            if Path(name).is_relative_to(target)
        }
        _compare(target, expected)
        stage = target.parent / ("." + target.name + ".staging")
        if stage.exists() or stage.is_symlink():
            source._path(stage)
            _linked_stage(stage, target, expected)
    result = dict(
        references=[references[name] for name in sorted(references)],
        package=package,
        ledger=bound(fixed["REVIEW/PACKAGE/private/selection.json"]),
        blind=bound(fixed["REVIEW/PACKAGE/blind/EXTRACTION.jsonl"], lines=True),
        tasks=bound(fixed["REVIEW/PACKAGE/judgment/TASKS.jsonl"], lines=True),
        summary=summary,
    )
    source._unchanged(list(states.values()))
    source._require(_membership() == membership, "Historical review changed during verification")
    _CACHE = (membership, list(states.values()), copy.deepcopy(result))
    return copy.deepcopy(result)
