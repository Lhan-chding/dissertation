"""Synthetic evidence only: these tests do not perform real independent review."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sr_f6 import phases, review, review_audit
from sr_f6.common import digest, file_hash

OBJECT = {
    "evidence": [{"series": "Alpha", "category": "X", "value": 1}],
    "selected": ["X"],
    "bindings": [{"category": "X", "operand": 1}],
    "aggregate": "sum",
    "answer": 1,
}
RAW = json.dumps(OBJECT, ensure_ascii=False, separators=(",", ":"))
LABELS = dict(A=1, J=1, V=1, finish=1, S=1, B=1)
ORACLE = dict(answer="1", selected=["X"], bindings={"X": "1"})


@pytest.mark.parametrize("lines", [False, True])
@pytest.mark.parametrize("number", ["1e999", "-1e999"])
def test_audit_reader_rejects_exponent_overflow(tmp_path, lines, number):
    path = tmp_path / ("input.jsonl" if lines else "input.json")
    path.write_text('{"value":' + number + "}\n")
    ref = dict(path=path.name, sha256=file_hash(path))
    with pytest.raises(ValueError, match="Nonfinite JSON"):
        review_audit._artifact(review_audit._ArtifactReader(tmp_path), ref, lines=lines)


@pytest.mark.parametrize("lines", [False, True])
def test_audit_reader_preserves_finite_numbers_and_full_text_strings(tmp_path, lines):
    item = dict(large=1e308, negative=-1e308, full_text='{"answer":1e999,"answer":-1e999}')
    path = tmp_path / ("input.jsonl" if lines else "input.json")
    path.write_text(json.dumps(item) + "\n")
    ref = dict(path=path.name, sha256=file_hash(path))
    result = review_audit._artifact(review_audit._ArtifactReader(tmp_path), ref, lines=lines)
    assert result == ([item] if lines else item)


class Fixture:
    def __init__(self, root):
        self.root = root
        self.pin = "d" * 64
        records, raw_by_row, tasks = [], {}, {}
        for parent, factor, family in review.BLOCKS:
            for index in range(10):
                root_id = f"DEV_CAL_{family}_{index % 8:04d}"
                qid = f"{root_id}_{factor}_v{index // 8}"
                row_id = f"{parent}/{qid}/0"
                row = dict(
                    phase="PARENT_CENSUS",
                    parent=parent,
                    factor=factor,
                    family=family,
                    root_id=root_id,
                    qid=qid,
                    draw=0,
                    row_id=row_id,
                    full_text=RAW,
                    raw_text=RAW,
                    seed=index,
                    tokens=[10, 20],
                    finish_reason="balanced",
                    model_input={"prompt": "synthetic task"},
                    **LABELS,
                )
                records.append(row)
                raw_by_row[row_id] = dict(row)
                tasks[qid] = dict(
                    qid=qid,
                    split="DEV_CAL",
                    root_id=root_id,
                    family=family,
                    factor=factor,
                    world={"categories": ["X"], "series": {"Alpha": [1], "Beta": [1]}},
                    query={"aggregate": "sum", "k": 1, "value": "Alpha"},
                    gold_answer="1",
                    gold_selected=["X"],
                    gold_bindings={"X": "1"},
                )
        self.ledger, self.blind = review.build_review(
            records, raw_by_row=raw_by_row, source_identity={"census_spec_sha256": "c" * 64}
        )
        task_ref = self.write("DATA/DEV_CAL/TASKS_PRIVATE.jsonl", list(tasks.values()), lines=True)
        manifest = self.write(
            "DATA/MANIFEST.json",
            {"files": [{**task_ref, "bytes": (root / task_ref["path"]).stat().st_size}]},
        )
        self.context = dict(
            phase="CENSUS_REVIEW_V1",
            data_manifest=manifest,
            source_manifest={
                "path": "PHASES/CENSUS_REVIEW_V1/SOURCE_MANIFEST.json",
                "sha256": "f" * 64,
            },
        )
        self.pin = self.write("PHASES/CENSUS_REVIEW_V1/IDENTITY.json", self.context)["sha256"]
        self.producer = dict(
            phase="CENSUS_REVIEW_V1",
            identity=self.context,
            identity_sha256=self.pin,
            status="BOUND_NOT_AUTHORIZED",
            training_allowed=False,
            deployment=str(root / "frozen-review-code"),
        )
        selected = {r["anonymous_id"]: r for r in self.ledger["selected"]}
        packet = [
            {
                "anonymous_id": b["anonymous_id"],
                "world": tasks[selected[b["anonymous_id"]]["qid"]]["world"],
                "query": tasks[selected[b["anonymous_id"]]["qid"]]["query"],
            }
            for b in self.blind
        ]
        self.package = dict(
            schema="srf6-review-package-binding-v1",
            context=self.context,
            phase_identity_sha256=self.pin,
            execution_origin="SYNTHETIC_TEST",
            ledger=self.write("package/private/selection.json", self.ledger),
            blind=self.write("package/blind/EXTRACTION.jsonl", self.blind, lines=True),
            ledger_source_identity=self.ledger["source_identity"],
            task_source=task_ref,
            judgment_tasks=self.write("package/judgment_tasks.jsonl", packet, lines=True),
        )
        self.package_ref = self.write("package/BINDING.json", self.package)

    def write(self, relative, value, *, lines=False):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in value)
            if lines
            else json.dumps(value, ensure_ascii=False)
        )
        path.write_text(payload, encoding="utf-8")
        return dict(path=relative, sha256=file_hash(path))

    def read(self, ref):
        return json.loads((self.root / ref["path"]).read_text())

    def reviewer(self, identifier, batch):
        return dict(
            id=identifier,
            type="MODEL_TEXT",
            identity="synthetic reviewer identity",
            model_identity="SYNTHETIC_MODEL_NO_ACTUAL_CALL",
            method=self.write(f"reviewers/{identifier}/method.json", {"scope": "fixture"}),
            source=self.write(
                f"reviewers/{identifier}/source.json",
                {"source": "synthetic independent implementation"},
            ),
            transcript=self.write(
                f"reviews/{batch}/transcript.json", {"execution_origin": "SYNTHETIC_TEST"}
            ),
        )

    def submission(self, batch="batch-1", *, ids=None, reviewer_id="rater-1"):
        subset = (
            self.blind
            if ids is None
            else [item for item in self.blind if item["anonymous_id"] in ids]
        )
        blind_ref = self.write(f"reviews/{batch}/blind.jsonl", subset, lines=True)
        extraction, judgment = [], []
        for item in subset:
            fields = {}
            for key, value in json.loads(item["raw_text"]).items():
                literal = json.dumps(value, separators=(",", ":"))
                # Locate the member's value, not an equal literal in an earlier member.
                raw = item["raw_text"]
                member_start = raw.index('"' + key + '":') + len(key) + 3
                fields[key] = dict(
                    state="READABLE",
                    span=dict(start=member_start, end=member_start + len(literal), text=literal),
                )
            import hashlib

            extraction.append(
                dict(
                    anonymous_id=item["anonymous_id"],
                    raw_text_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                    object_status="UNAMBIGUOUS_JSON_OBJECT",
                    fields=fields,
                    rationale="SYNTHETIC raw-span fixture",
                )
            )
            judgment.append(
                dict(
                    anonymous_id=item["anonymous_id"],
                    labels=dict(LABELS),
                    unknown_reasons={},
                    oracle=copy.deepcopy(ORACLE),
                    rationale="SYNTHETIC judgment fixture",
                    ambiguities=[],
                )
            )
        extraction_ref = self.write(
            f"reviews/{batch}/extraction.json",
            dict(
                stage="BLIND_EXTRACTION",
                reviewer_id=reviewer_id,
                inputs=[blind_ref],
                exposure=dict(program_labels=False, oracle=False, private_ledger=False),
                completed_at="2026-10-27T01:00:00+00:00",
                items=extraction,
            ),
        )
        judgment_ref = self.write(
            f"reviews/{batch}/judgments.json",
            dict(
                stage="INDEPENDENT_JUDGMENT",
                reviewer_id=reviewer_id,
                inputs=[blind_ref, self.package["judgment_tasks"], extraction_ref],
                program_labels_seen=False,
                started_at="2026-10-27T02:00:00+00:00",
                completed_at="2026-10-27T03:00:00+00:00",
                items=judgment,
            ),
        )
        return self.write(
            f"reviews/{batch}/SUBMISSION.json",
            dict(
                schema="srf6-independent-review-v1",
                package_sha256=self.package_ref["sha256"],
                reviewer=self.reviewer(reviewer_id, batch),
                execution_origin="SYNTHETIC_TEST",
                independence=dict.fromkeys(review_audit.INDEPENDENCE, True),
                blind_input=blind_ref,
                blind_extraction=extraction_ref,
                judgments=judgment_ref,
            ),
        )

    def change(self, ref, component, mutate):
        manifest = self.read(ref)
        if component is None:
            mutate(manifest)
        else:
            value = self.read(manifest[component])
            mutate(value)
            manifest[component] = self.write(manifest[component]["path"], value)
            if component == "blind_extraction":
                judgment = self.read(manifest["judgments"])
                judgment["inputs"][-1] = manifest[component]
                manifest["judgments"] = self.write(manifest["judgments"]["path"], judgment)
        return self.write(ref["path"], manifest)

    def adjudication(
        self, finding_id, decision, *, reviewer_id="adjudicator", batch="adjudication"
    ):
        return self.write(
            f"reviews/{batch}/ADJUDICATION.json",
            dict(
                schema="srf6-review-adjudication-v1",
                package_sha256=self.package_ref["sha256"],
                reviewer=self.reviewer(reviewer_id, batch),
                execution_origin="SYNTHETIC_TEST",
                independent_of_reviewers=True,
                completed_at="2026-10-27T04:00:00+00:00",
                items=[
                    dict(
                        finding_id=finding_id,
                        decision=decision,
                        rationale="SYNTHETIC adjudication fixture",
                        evidence=[self.package["judgment_tasks"]],
                    )
                ],
            ),
        )

    def aggregate(self, refs=(), adjudications=()):
        return review_audit.aggregate_review(
            self.root,
            phase_deployment=self.producer["deployment"],
            phase_identity_sha256=self.pin,
            package_ref=self.package_ref,
            review_refs=list(refs),
            adjudication_refs=list(adjudications),
        )


@pytest.fixture
def fx(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path)

    def verified(root, phase, *, deployment, expected_identity_sha256):
        assert Path(root) == tmp_path
        assert phase == "CENSUS_REVIEW_V1"
        assert deployment == fixture.producer["deployment"]
        assert expected_identity_sha256 == fixture.pin
        return copy.deepcopy(fixture.producer)

    monkeypatch.setattr(phases, "verify_phase", verified)
    return fixture


def test_complete_synthetic_review_never_signs_gate_or_adopts_input_producer_context(fx):
    result = fx.aggregate([fx.submission()])
    assert result["status"] == "SYNTHETIC_ONLY"
    assert result["workflow_state"] == "REVIEW_COMPLETE_NOT_GATE_SIGNED"
    assert result["training_allowed"] is False and result["gate_status"] == "NOT_SIGNED"
    assert (
        result["reviewed_original_answers"]
        == result["independently_reviewed_original_answers"]
        == 360
    )
    assert result["findings"] == []
    assert result["agreement_threshold"] is None
    assert result["census_observability_threshold"]["applied_to_review_sample"] is False
    assert result["input_producer"]["phase"] == "CENSUS_REVIEW_V1"
    assert result["validator"]["phase"] is None and "context" not in result
    assert result["reviewers"][0]["reviewer"]["type"] == "MODEL_TEXT"
    assert result["selection_provenance"] == "NOT_RECOMPUTED_FROM_FULL_CENSUS"
    assert isinstance(result["inputs"], list)
    assert result["named_inputs"]["package"] == fx.package_ref


def test_v2_final_delivery_reuses_one_full_package_verification_across_shards(fx, monkeypatch):
    """Routing fixture only; real package replay is tested in test_review_package."""
    from sr_f6 import review_package

    producer = copy.deepcopy(fx.producer)
    producer["phase"] = producer["identity"]["phase"] = "PILOT_V1"
    producer["deployment"] = str(review_package._deployment())
    sampler = dict(
        identity_sha256="c" * 64, spec={"path": review_package.SPEC_PATH, "sha256": "e" * 64}
    )
    fx.package.update(
        schema=review_package.SCHEMA, context=producer["identity"], sampler_recipe=sampler
    )
    fx.package_ref = fx.write("package/BINDING.json", fx.package)
    calls = []

    def verified(root, package_ref, **kwargs):
        calls.append((root, package_ref, kwargs))
        return dict(
            producer=producer,
            sampler_recipe=sampler,
            input_producer={"phase": "CENSUS_S0", "identity_sha256": "f" * 64},
            inputs=[
                fx.package_ref,
                fx.package["ledger"],
                fx.package["blind"],
                fx.package["task_source"],
            ],
            selection_provenance="RECOMPUTED_FROM_VERIFIED_FULL_CENSUS",
            artifacts=dict(
                package=fx.package,
                ledger=fx.ledger,
                blind=fx.blind,
                tasks={
                    t["qid"]: t
                    for t in [
                        json.loads(line)
                        for line in (fx.root / fx.package["task_source"]["path"])
                        .read_text()
                        .splitlines()
                    ]
                },
            ),
        )

    monkeypatch.setattr(review_package, "verify_package", verified)
    refs = [
        fx.submission(batch="first", ids=[item["anonymous_id"] for item in fx.blind[:180]]),
        fx.submission(batch="second", ids=[item["anonymous_id"] for item in fx.blind[180:]]),
    ]
    result = review_audit.aggregate_review(
        fx.root,
        phase_deployment=producer["deployment"],
        phase_identity_sha256=fx.pin,
        package_ref=fx.package_ref,
        review_refs=refs,
    )
    assert len(calls) == 1
    assert calls[0][2] == dict(
        phase_identity_sha256=fx.pin,
        sampler_phase_identity_sha256="c" * 64,
        review_spec_sha256="e" * 64,
    )
    assert result["status"] == "SYNTHETIC_ONLY"
    assert result["independently_reviewed_original_answers"] == 360
    assert result["validator"]["phase"] == result["context"]["phase"] == "PILOT_V1"
    assert result["sampler_recipe"] == sampler
    assert result["census_input_producer"]["phase"] == "CENSUS_S0"
    assert result["selection_provenance"] == "RECOMPUTED_FROM_VERIFIED_FULL_CENSUS"
    assert result["selection_verification_required_before_admission"] == []
    assert result["gate_status"] == "NOT_SIGNED" and result["training_allowed"] is False
    assert fx.package["task_source"] in result["inputs"]


def test_unknown_package_schema_cannot_claim_an_arbitrary_producer(fx):
    fx.package["schema"] = "arbitrary-producer"
    fx.package_ref = fx.write("package/BINDING.json", fx.package)
    with pytest.raises(ValueError, match="Unknown review package producer"):
        fx.aggregate([])


def test_multiple_reviewers_and_same_reviewer_disjoint_shards_cover_360(fx):
    ids = [row["anonymous_id"] for row in fx.blind]
    shards = [
        fx.submission("a", ids=ids[:100]),
        fx.submission("b", ids=ids[100:200]),
        fx.submission("c", ids=ids[200:], reviewer_id="rater-2"),
    ]
    result = fx.aggregate(shards)
    assert result["workflow_state"] == "REVIEW_COMPLETE_NOT_GATE_SIGNED"
    assert result["independently_reviewed_original_answers"] == 360
    assert len(result["reviewers"]) == 3
    assert result["agreement"]["J"]["comparisons"] == 360


def test_partial_shard_reports_missing_ids_without_new_per_reviewer_quota(fx):
    ids = [row["anonymous_id"] for row in fx.blind[:17]]
    result = fx.aggregate([fx.submission(ids=ids)])
    assert result["workflow_state"] == "REVIEW_INCOMPLETE"
    assert result["independently_reviewed_original_answers"] == 17
    assert len(result["missing_independent_review_ids"]) == 343


def test_overlapping_same_reviewer_claims_are_retained_not_overwritten_or_majority_voted(fx):
    first = fx.submission("first")
    second = fx.submission("second", ids=[fx.blind[0]["anonymous_id"]])
    second = fx.change(second, "judgments", lambda value: value["items"][0]["labels"].update(S=0))
    result = fx.aggregate([first, second])
    finding = next(f for f in result["findings"] if f["kind"] == "LABEL_DISAGREEMENT")
    assert len(finding["claims"]) == 3
    assert finding["suspected_false_pass"] is True
    assert result["agreement"]["S"]["comparisons"] == 361
    assert result["workflow_state"] == "ADJUDICATION_REQUIRED"


def test_empty_reviews_are_not_claimed_as_review_complete(fx):
    result = fx.aggregate()
    assert result["workflow_state"] == "REVIEW_NOT_RUN"
    assert result["reviewed_original_answers"] == 0
    assert result["agreement"]["J"]["rate"] is None


@pytest.mark.parametrize(
    "component,mutation",
    [
        ("judgments", lambda value: value["items"].append(copy.deepcopy(value["items"][0]))),
        ("judgments", lambda value: value["items"].pop()),
        ("blind_extraction", lambda value: value["items"][0].update(anonymous_id="R9999")),
        (
            "blind_extraction",
            lambda value: value["items"][0]["fields"]["selected"]["span"].update(text='["GOLD"]'),
        ),
        ("blind_extraction", lambda value: value["exposure"].update(oracle=True)),
        ("judgments", lambda value: value.update(program_labels_seen=True)),
        ("judgments", lambda value: value.update(started_at="2026-10-27T00:00:00+00:00")),
        ("judgments", lambda value: value["items"][0]["labels"].update(J=None)),
        ("judgments", lambda value: value["items"][0]["labels"].update(V=True)),
    ],
)
def test_rejects_duplicate_missing_unknown_ids_gold_exposure_and_wrong_label_domains(
    fx, component, mutation
):
    ref = fx.change(fx.submission(), component, mutation)
    with pytest.raises(ValueError):
        fx.aggregate([ref])


def test_invalid_protocol_does_not_force_readable_semantic_claims_unknown(fx):
    ref = fx.submission()
    ref = fx.change(
        ref, "judgments", lambda value: value["items"][0]["labels"].update(V=0, J=0, S=1, B=0)
    )
    result = fx.aggregate([ref])
    assert not any(f["kind"] == "REVIEW_CONTRACT_VIOLATION" for f in result["findings"])
    assert {f["field"] for f in result["findings"]} == {"V", "J", "B"}


def test_incomplete_semantics_unknown_preserves_binary_answer_and_protocol(fx):
    ref = fx.submission()

    def incomplete(value):
        value["items"][0]["labels"].update(finish=0, J=0, S=None, B=None, A=1, V=1)
        value["items"][0]["unknown_reasons"] = {
            "S": "Incomplete generation",
            "B": "Incomplete generation",
        }

    ref = fx.change(ref, "judgments", incomplete)
    result = fx.aggregate([ref])
    assert not any(f["kind"] == "REVIEW_CONTRACT_VIOLATION" for f in result["findings"])
    assert result["agreement"]["S"]["review_unknown"] == 1
    assert result["agreement"]["A"]["agreements"] == 360
    assert result["agreement"]["V"]["agreements"] == 360


def test_empty_readable_claims_and_field_local_unknown_rules():
    fields = {key: {"state": "READABLE", "span": {"text": "[]"}} for key in review_audit.FIELDS}
    assert "S_EMPTY_READABLE_CLAIM_IS_FAIL" in review_audit._constraints(LABELS, fields)
    labels = dict(LABELS, V=0, J=0, S=0, B=None)
    fields["bindings"] = {"state": "UNREADABLE", "span": None}
    assert review_audit._constraints(labels, fields) == []


def test_whole_object_ambiguity_differs_from_field_local_invalidity():
    fields = {
        key: {"state": "UNREADABLE", "span": {"text": "locatable fragment"}}
        for key in review_audit.FIELDS
    }
    labels = dict(A=0, J=0, V=0, finish=1, S=None, B=None)
    assert review_audit._constraints(labels, fields, "INVALID_OR_AMBIGUOUS_JSON") == []
    fields["selected"] = {"state": "READABLE", "span": {"text": '["X"]'}}
    assert "S_AMBIGUOUS_OBJECT_CANNOT_PROVIDE_READABLE_CLAIM" in review_audit._constraints(
        labels, fields, "INVALID_OR_AMBIGUOUS_JSON"
    )
    # The same readable selected claim is measurable in an unambiguous object
    # whose separate bindings field is invalid.
    labels.update(A=1, S=1)
    assert review_audit._constraints(labels, fields, "UNAMBIGUOUS_JSON_OBJECT") == []


def test_duplicate_json_keys_keep_unknown_even_when_field_fragments_are_locatable(fx):
    import hashlib

    anonymous_id = fx.blind[0]["anonymous_id"]
    raw = RAW[:-1] + ',"answer":1}'
    fx.blind[0]["raw_text"] = raw
    selected = next(row for row in fx.ledger["selected"] if row["anonymous_id"] == anonymous_id)
    selected["raw_text_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    selected["unit_id"] = digest([selected["parent"], selected["factor"], selected["qid"], raw])
    selected["program_labels"].update(A=0, V=0, J=0, S=None, B=None)
    fx.ledger["blind_items_sha256"] = digest(fx.blind)
    fx.package["blind"] = fx.write(fx.package["blind"]["path"], fx.blind, lines=True)
    fx.package["ledger"] = fx.write(fx.package["ledger"]["path"], fx.ledger)
    fx.package_ref = fx.write(fx.package_ref["path"], fx.package)
    ref = fx.submission()

    def extraction(value):
        item = value["items"][0]
        item["object_status"] = "INVALID_OR_AMBIGUOUS_JSON"
        for field in item["fields"].values():
            field["state"] = "UNREADABLE"  # Spans remain, without implying semantic readability.

    def judgment(value):
        item = value["items"][0]
        item["labels"].update(A=0, V=0, J=0, S=None, B=None)
        item["unknown_reasons"] = {"S": "Duplicate JSON key", "B": "Duplicate JSON key"}

    ref = fx.change(ref, "blind_extraction", extraction)
    ref = fx.change(ref, "judgments", judgment)
    result = fx.aggregate([ref])
    assert result["findings"] == []
    assert result["agreement"]["S"]["review_unknown"] == 1
    assert result["agreement"]["B"]["review_unknown"] == 1


@pytest.mark.parametrize("operand", [None, "not-a-number"])
def test_locatable_null_or_nonnumeric_binding_is_unknown_without_erasing_selection(fx, operand):
    import hashlib

    anonymous_id = fx.blind[0]["anonymous_id"]
    obj = copy.deepcopy(OBJECT)
    obj["bindings"][0]["operand"] = operand
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    fx.blind[0]["raw_text"] = raw
    selected = next(row for row in fx.ledger["selected"] if row["anonymous_id"] == anonymous_id)
    selected["raw_text_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    selected["unit_id"] = digest([selected["parent"], selected["factor"], selected["qid"], raw])
    selected["program_labels"].update(V=0, J=0, B=None)
    fx.ledger["blind_items_sha256"] = digest(fx.blind)
    fx.package["blind"] = fx.write(fx.package["blind"]["path"], fx.blind, lines=True)
    fx.package["ledger"] = fx.write(fx.package["ledger"]["path"], fx.ledger)
    fx.package_ref = fx.write(fx.package_ref["path"], fx.package)
    ref = fx.submission()

    def extraction(value):
        value["items"][0]["fields"]["bindings"]["state"] = "UNREADABLE"

    def judgment(value):
        value["items"][0]["labels"].update(V=0, J=0, B=None)
        value["items"][0]["unknown_reasons"] = {"B": "Operand is null or nonnumeric"}

    ref = fx.change(ref, "blind_extraction", extraction)
    ref = fx.change(ref, "judgments", judgment)
    result = fx.aggregate([ref])
    assert result["findings"] == []
    assert result["agreement"]["B"]["review_unknown"] == 1
    assert result["agreement"]["S"]["review_unknown"] == 0


def test_output_dependency_list_contains_all_verified_leaves_and_catches_later_changes(
    fx, monkeypatch
):
    ref = fx.submission()
    result = fx.aggregate([ref])
    manifest = fx.read(ref)
    required = [
        ref,
        manifest["blind_input"],
        manifest["blind_extraction"],
        manifest["judgments"],
        *[manifest["reviewer"][name] for name in ("method", "source", "transcript")],
    ]
    assert all(reference in result["inputs"] for reference in required)
    # Exercise the actual phase dependency walker, with only its server-path
    # boundary adapted to this synthetic temporary directory.
    monkeypatch.setattr(phases, "_file", lambda root, relative: Path(root) / relative)

    class Cache:
        @staticmethod
        def hash(path):
            return file_hash(path)

    phases._dependencies(fx.root, result, "synthetic-report.json", Cache(), set(), set())
    transcript = fx.root / manifest["reviewer"]["transcript"]["path"]
    transcript.write_bytes(transcript.read_bytes() + b"\n")
    with pytest.raises(PermissionError, match="dependency changed"):
        phases._dependencies(fx.root, result, "synthetic-report.json", Cache(), set(), set())


def test_unknown_requires_field_reason_and_does_not_absorb_unrelated_labels(fx):
    ref = fx.change(
        fx.submission(), "judgments", lambda value: value["items"][0]["labels"].update(S=None)
    )
    with pytest.raises(ValueError, match="UNKNOWN requires"):
        fx.aggregate([ref])


def test_failed_independence_attestation_is_blocking_not_independent_coverage(fx):
    ref = fx.change(
        fx.submission(),
        None,
        lambda value: value["independence"].update(no_shared_production_scorer=False),
    )
    result = fx.aggregate([ref])
    assert result["workflow_state"] == "BLOCKING_FINDINGS"
    assert result["independently_reviewed_original_answers"] == 0


@pytest.mark.parametrize(
    "decision,workflow",
    [
        ("PRODUCTION_CORRECT", "REVIEW_COMPLETE_NOT_GATE_SIGNED"),
        ("PRODUCTION_INCORRECT", "BLOCKING_FINDINGS"),
        ("UNRESOLVED", "ADJUDICATION_REQUIRED"),
    ],
)
def test_each_adjudication_keeps_original_disagreement_and_has_no_agreement_threshold(
    fx, decision, workflow
):
    ref = fx.change(
        fx.submission(), "judgments", lambda value: value["items"][0]["labels"].update(S=0)
    )
    before = fx.aggregate([ref])
    finding = before["findings"][0]
    adjudication = fx.adjudication(finding["finding_id"], decision)
    result = fx.aggregate([ref], [adjudication])
    assert result["workflow_state"] == workflow
    assert result["findings"][0]["claims"] == finding["claims"]
    assert result["agreement"]["S"]["rate"] == 359 / 360
    assert result["training_allowed"] is False


def test_independent_oracle_difference_requires_adjudication_and_confirmed_error_blocks(fx):
    ref = fx.change(
        fx.submission(), "judgments", lambda value: value["items"][0]["oracle"].update(answer="2")
    )
    initial = fx.aggregate([ref])
    finding = next(f for f in initial["findings"] if f["kind"] == "ORACLE_DISAGREEMENT")
    result = fx.aggregate([ref], [fx.adjudication(finding["finding_id"], "PRODUCTION_INCORRECT")])
    assert result["workflow_state"] == "BLOCKING_FINDINGS"


def test_self_adjudication_and_duplicate_submission_are_rejected(fx):
    ref = fx.change(
        fx.submission(), "judgments", lambda value: value["items"][0]["labels"].update(S=0)
    )
    finding = fx.aggregate([ref])["findings"][0]
    with pytest.raises(ValueError, match="independent of original"):
        fx.aggregate(
            [ref],
            [fx.adjudication(finding["finding_id"], "PRODUCTION_CORRECT", reviewer_id="rater-1")],
        )
    with pytest.raises(ValueError, match="Duplicate review submission"):
        fx.aggregate([ref, ref])


def test_model_identity_cannot_be_relabelled_human_and_transcript_must_be_bound(fx):
    ref = fx.change(fx.submission(), None, lambda value: value["reviewer"].update(type="HUMAN"))
    with pytest.raises(ValueError, match="disguise a model"):
        fx.aggregate([ref])


@pytest.mark.parametrize("target", ["blind", "ledger", "task_source"])
def test_frozen_package_artifact_bytes_cannot_change(fx, target):
    path = fx.root / fx.package[target]["path"]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="bytes changed"):
        fx.aggregate()


def test_judgment_packet_cannot_include_private_gold_or_program_labels(fx):
    packet = [
        json.loads(line)
        for line in (fx.root / fx.package["judgment_tasks"]["path"]).read_text().splitlines()
    ]
    packet[0]["gold_answer"] = "1"
    fx.package["judgment_tasks"] = fx.write(
        fx.package["judgment_tasks"]["path"], packet, lines=True
    )
    fx.package_ref = fx.write(fx.package_ref["path"], fx.package)
    with pytest.raises(ValueError, match="no gold or program labels"):
        fx.aggregate()


def test_ambiguities_are_preserved_for_adjudication_and_oracle_equivalent_fractions_agree(fx):
    ref = fx.submission()

    def change(value):
        value["items"][0]["oracle"]["answer"] = "2/2"
        value["items"][0]["ambiguities"] = ["SYNTHETIC uncertainty; do not erase"]

    ref = fx.change(ref, "judgments", change)
    result = fx.aggregate([ref])
    assert [f["kind"] for f in result["findings"]] == ["REVIEW_AMBIGUITY"]
    assert result["workflow_state"] == "ADJUDICATION_REQUIRED"


def test_aggregator_never_calls_production_parser_or_scorer(fx, monkeypatch):
    from sr_f6 import measurement

    def prohibited(*args, **kwargs):
        pytest.fail("Production implementation cannot manufacture independent review")

    for name in ("parse", "score_response", "oracle_a"):
        monkeypatch.setattr(measurement, name, prohibited)
    assert fx.aggregate([fx.submission()])["reviewed_original_answers"] == 360
