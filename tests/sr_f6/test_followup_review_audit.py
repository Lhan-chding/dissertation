"""Synthetic evidence, including actual-format traces; no real reviews are performed."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from sr_f6.common import digest
from sr_f6_followup import review_audit, review_package

spec = importlib.util.spec_from_file_location(
    "followup_package_fixtures", Path(__file__).with_name("test_followup_review_package.py")
)
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


class Fixture(fixtures.Fixture):
    def __init__(self, root, monkeypatch, *, synthetic=False):
        super().__init__(root, monkeypatch)
        review_audit._CACHE.clear()
        if not synthetic:
            # Actual-format local fixtures exercise completion only. The upstream
            # source verifier is mocked; this is never real scientific evidence.
            for row in self.records:
                row["execution_origin"] = "ACTUAL_CONFIRMATION"
            for row in self.raw.values():
                row["execution_origin"] = "ACTUAL_CONFIRMATION"
            self.support.update(
                records_sha256=digest(self.records), raw_records_sha256=digest(self.raw)
            )
        bundle = review_package._derive(root)
        monkeypatch.setattr(review_package, "_derive", lambda root: copy.deepcopy(bundle))
        self.package()
        self.package_ref = self.verification["package_ref"]
        self.package_value = self.verification["artifacts"]["package"]
        self.blind = self.verification["artifacts"]["blind"]
        self.monkeypatch = monkeypatch

    def event(self, stage, inputs, output, started):
        value = self.read(output)
        return dict(
            stage=stage,
            started_at=started,
            completed_at=value["completed_at"],
            inputs=[dict(reference=ref, content=self.read(ref)) for ref in inputs],
            output=dict(reference=output, content=value),
        )

    def reviewer(self, task, instance, reviewer_id):
        prefix = "REVIEW/EXECUTIONS/" + task
        method = self.write(
            prefix + "/METHOD.json", {"method": "Independent raw reading and calculation"}
        )
        origin = self.write(
            prefix + "/SOURCE.json",
            dict(
                execution_origin="ACTUAL_REVIEW",
                reviewer_type="MODEL_TEXT",
                instance_id=instance,
                task_id=task,
            ),
        )
        return dict(
            id=reviewer_id,
            type="MODEL_TEXT",
            identity="local actual-format test fixture",
            model_identity="NOT_EXPOSED",
            method=method,
            source=origin,
        )

    def trace(self, task, reviewer, events):
        source_value = self.read(reviewer["source"])
        trace = dict(
            schema=review_audit.TRACE_SCHEMA,
            execution_origin="ACTUAL_REVIEW",
            reviewer_id=reviewer["id"],
            reviewer_type=reviewer["type"],
            identity=reviewer["identity"],
            model_identity=reviewer["model_identity"],
            instance_id=source_value["instance_id"],
            task_id=task,
            method=reviewer["method"],
            source=reviewer["source"],
            events=events,
        )
        return self.write(f"REVIEW/EXECUTIONS/{task}/TRACE.json", trace)

    def submit(self, task="task-1", *, ids=None, labels=None, oracle=None):
        self.monkeypatch.setattr(review_audit, "now", lambda: "2026-10-29T00:00:00+00:00")
        review_audit.prepare_execution(
            self.root, task_id=task, instance_id="instance-" + task, reviewer_id="reviewer-" + task
        )
        prefix = f"REVIEW/EXECUTIONS/{task}"
        subset = self.blind if ids is None else [r for r in self.blind if r["anonymous_id"] in ids]
        blind_ref = self.write(prefix + "/BLIND.jsonl", subset, lines=True)
        extracted, judged = [], []
        for item in subset:
            raw, fields = item["raw_text"], {}
            for key, value in fixtures.OBJECT.items():
                text = fixtures.json.dumps(value, separators=(",", ":"))
                start = raw.index('"' + key + '":') + len(key) + 3
                fields[key] = dict(
                    state="READABLE", span=dict(start=start, end=start + len(text), text=text)
                )
            extracted.append(
                dict(
                    anonymous_id=item["anonymous_id"],
                    raw_text_sha256=fixtures.hashlib.sha256(raw.encode()).hexdigest(),
                    fields=fields,
                    object_status="UNAMBIGUOUS_JSON_OBJECT",
                    rationale="Fixture literal spans",
                )
            )
            judged.append(
                dict(
                    anonymous_id=item["anonymous_id"],
                    labels=labels or dict(fixtures.LABELS),
                    unknown_reasons={},
                    oracle=oracle or dict(answer="1", selected=["X"], bindings={"X": "1"}),
                    rationale="Fixture independent calculation",
                    ambiguities=[],
                )
            )
        reviewer_id = "reviewer-" + task
        extraction_ref = self.write(
            prefix + "/EXTRACTION.json",
            dict(
                stage="BLIND_EXTRACTION",
                reviewer_id=reviewer_id,
                inputs=[blind_ref],
                exposure=dict(program_labels=False, oracle=False, private_ledger=False),
                completed_at="2026-10-29T01:00:00+00:00",
                items=extracted,
            ),
        )
        inputs = [blind_ref, self.package_value["judgment_tasks"], extraction_ref]
        judgment_ref = self.write(
            prefix + "/JUDGMENT.json",
            dict(
                stage="INDEPENDENT_JUDGMENT",
                reviewer_id=reviewer_id,
                inputs=inputs,
                program_labels_seen=False,
                started_at="2026-10-29T02:00:00+00:00",
                completed_at="2026-10-29T03:00:00+00:00",
                items=judged,
            ),
        )
        reviewer = self.reviewer(task, "instance-" + task, reviewer_id)
        reviewer["transcript"] = self.trace(
            task,
            reviewer,
            [
                self.event(
                    "BLIND_EXTRACTION", [blind_ref], extraction_ref, "2026-10-29T00:30:00+00:00"
                ),
                self.event(
                    "INDEPENDENT_JUDGMENT", inputs, judgment_ref, "2026-10-29T02:00:00+00:00"
                ),
            ],
        )
        return self.write(
            prefix + "/SUBMISSION.json",
            dict(
                schema="srf6-independent-review-v1",
                package_sha256=self.package_ref["sha256"],
                reviewer=reviewer,
                execution_origin="ACTUAL_REVIEW",
                independence=dict.fromkeys(review_audit.rules.INDEPENDENCE, True),
                blind_input=blind_ref,
                blind_extraction=extraction_ref,
                judgments=judgment_ref,
            ),
        )

    def resubmit(self, ref, mutate):
        value = self.read(ref)
        mutate(value)
        return self.write(ref["path"], value)

    def retrace(self, ref, *, extraction=None, judgment=None):
        value = self.read(ref)
        if extraction:
            value["blind_extraction"] = self.write(value["blind_extraction"]["path"], extraction)
        judged = judgment or self.read(value["judgments"])
        judged["inputs"][-1] = value["blind_extraction"]
        value["judgments"] = self.write(value["judgments"]["path"], judged)
        task = value["reviewer"]["source"]["path"].split("/")[2]
        value["reviewer"]["transcript"] = self.trace(
            task,
            value["reviewer"],
            [
                self.event(
                    "BLIND_EXTRACTION",
                    [value["blind_input"]],
                    value["blind_extraction"],
                    "2026-10-29T00:30:00+00:00",
                ),
                self.event(
                    "INDEPENDENT_JUDGMENT",
                    judged["inputs"],
                    value["judgments"],
                    judged["started_at"],
                ),
            ],
        )
        return self.write(ref["path"], value)

    def adjudicate(self, *, decision="PRODUCTION_CORRECT", task="adj-1"):
        self.monkeypatch.setattr(review_audit, "now", lambda: "2026-10-29T03:30:00+00:00")
        attempt = review_audit.prepare_execution(
            self.root,
            task_id=task,
            instance_id="instance-" + task,
            reviewer_id="reviewer-" + task,
            kind="ADJUDICATION",
        )
        packet = self.read(attempt["findings"])
        reviewer = self.reviewer(task, "instance-" + task, "reviewer-" + task)
        response = dict(
            completed_at="2026-10-29T05:00:00+00:00",
            items=[
                dict(
                    finding_id=row["finding_id"],
                    decision=decision,
                    rationale="Fixture explicit adjudication",
                    evidence=[attempt["findings"]],
                )
                for row in packet["findings"]
            ],
        )
        response_ref = self.write(f"REVIEW/EXECUTIONS/{task}/RESPONSE.json", response)
        reviewer["transcript"] = self.trace(
            task,
            reviewer,
            [
                self.event(
                    "ADJUDICATION",
                    attempt["input_references"],
                    response_ref,
                    "2026-10-29T04:00:00+00:00",
                )
            ],
        )
        return self.write(
            f"REVIEW/EXECUTIONS/{task}/ADJUDICATION.json",
            dict(
                schema="srf6-review-adjudication-v1",
                package_sha256=self.package_ref["sha256"],
                execution_origin="ACTUAL_REVIEW",
                reviewer=reviewer,
                independent_of_reviewers=True,
                findings=attempt["findings"],
                input_references=attempt["input_references"],
                decision_response=response_ref,
                **response,
            ),
        )


@pytest.fixture
def fx(tmp_path, monkeypatch):
    return Fixture(tmp_path, monkeypatch)


def test_real_format_trace_complete_seal_and_warm_cache(fx, monkeypatch):
    fx.submit()
    result = review_audit.seal(fx.root)
    assert result["status"] == result["workflow_state"] == "REVIEW_COMPLETE_NOT_GATE_SIGNED"
    assert result["completed"] and result["synthetic"] is False
    assert (
        result["reviewed_original_answers"]
        == result["independently_reviewed_original_answers"]
        == 360
    )
    assert result["consumer"] == fx.consumer and result["input_producer"] == fx.producer
    assert result["training_allowed"] is result["evaluation_allowed"] is False
    assert result["agreement_threshold"] is None
    original = copy.deepcopy(result)
    result["reviewers"].clear()

    def forbidden(*args, **kwargs):
        pytest.fail("Warm verification must not replay package or traces")

    monkeypatch.setattr(review_audit, "aggregate_review", forbidden)
    assert review_audit.verify_completed(fx.root) == original
    assert fx.calls[-1] == "review"


def test_synthetic_package_or_trace_never_complete(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch, synthetic=True)
    fixture.submit()
    out = review_audit.seal(tmp_path)
    assert out["status"] == "SYNTHETIC_ONLY" and not out["completed"]


def test_synthetic_marker_in_actual_format_trace_never_complete(fx):
    ref = fx.submit()
    value = fx.read(ref)
    source_ref = value["reviewer"]["source"]
    origin = fx.read(source_ref)
    origin["status"] = "SYNTHETIC_ONLY"
    value["reviewer"]["source"] = fx.write(source_ref["path"], origin)
    transcript = fx.read(value["reviewer"]["transcript"])
    transcript["source"] = value["reviewer"]["source"]
    value["reviewer"]["transcript"] = fx.write(value["reviewer"]["transcript"]["path"], transcript)
    fx.write(ref["path"], value)
    assert review_audit.aggregate_review(fx.root)["status"] == "SYNTHETIC_ONLY"


@pytest.mark.parametrize("kind", ["extra", "changed", "failure", "summary", "symlink"])
def test_warm_cache_detects_all_review_changes(fx, kind):
    ref = fx.submit()
    review_audit.seal(fx.root)
    if kind == "extra":
        fx.write("REVIEW/EXECUTIONS/new/ATTEMPT.json", {})
    elif kind == "failure":
        fx.write("REVIEW/EXECUTIONS/task-1/FAILURE.json", {"error": "preserve"})
    elif kind == "summary":
        fx.write("REVIEW/AUDIT/SUMMARY.json", {"status": "PASS"})
    elif kind == "symlink":
        (fx.root / "REVIEW/alias").symlink_to(fx.root / ref["path"])
    else:
        value = fx.read(ref)
        value["status"] = "PASS"
        fx.write(ref["path"], value)
    with pytest.raises((PermissionError, ValueError)):
        review_audit.verify_completed(fx.root)


@pytest.mark.parametrize("kind", ["partial", "failure", "unreported", "both"])
def test_omitted_or_failed_attempt_is_never_hidden(fx, kind):
    fx.submit()
    prefix = "REVIEW/EXECUTIONS/task-1"
    if kind == "partial":
        (fx.root / prefix / "SUBMISSION.json").unlink()
    elif kind == "failure":
        fx.write(prefix + "/FAILURE.json", {"error": "preserved"})
    elif kind == "unreported":
        fx.write("REVIEW/EXECUTIONS/orphan/ATTEMPT.json", {})
    else:
        fx.write(prefix + "/ADJUDICATION.json", {})
    with pytest.raises((PermissionError, ValueError, FileNotFoundError)):
        review_audit.seal(fx.root)
    assert (fx.root / "REVIEW/.audit-failure.json").exists()
    with pytest.raises((PermissionError, ValueError), match="Failed audit preserved"):
        review_audit.seal(fx.root)


@pytest.mark.parametrize(
    "kind", ["program", "payload", "omitted", "time", "leak", "span", "unknown"]
)
def test_trace_reading_chronology_leakage_and_labels_fail_closed(fx, kind):
    ref = fx.submit()
    value = fx.read(ref)
    trace = fx.read(value["reviewer"]["transcript"])
    if kind == "program":
        value["reviewer"]["type"] = "PROGRAM"
    elif kind == "payload":
        trace["events"][0]["inputs"][0]["content"][0]["raw_text"] = "rewritten"
    elif kind == "omitted":
        trace["events"].pop()
    elif kind == "time":
        trace["events"][1]["started_at"] = "2026-10-29T00:40:00+00:00"
    elif kind in ("leak", "span"):
        extraction = fx.read(value["blind_extraction"])
        if kind == "leak":
            extraction["exposure"]["oracle"] = True
        else:
            extraction["items"][0]["fields"]["answer"]["span"]["text"] = "999"
        fx.retrace(ref, extraction=extraction)
    else:
        judgment = fx.read(value["judgments"])
        judgment["items"][0]["labels"]["S"] = None
        fx.retrace(ref, judgment=judgment)
    if kind not in ("leak", "span", "unknown"):
        value["reviewer"]["transcript"] = fx.write(value["reviewer"]["transcript"]["path"], trace)
        fx.write(ref["path"], value)
    with pytest.raises((PermissionError, ValueError)):
        review_audit.aggregate_review(fx.root)


def test_incomplete_coverage_and_unresolved_false_pass_are_preserved(fx):
    fx.submit(ids={"R0001"}, labels=dict(fixtures.LABELS, S=0))
    out = review_audit.aggregate_review(fx.root)
    assert out["status"] == "ADJUDICATION_REQUIRED" and not out["completed"]
    assert out["findings"][0]["suspected_false_pass"]
    assert len(out["missing_independent_review_ids"]) == 359


def test_oracle_disagreement_and_independent_adjudication(fx):
    fx.submit(oracle=dict(answer="2", selected=["X"], bindings={"X": "1"}))
    before = review_audit.aggregate_review(fx.root)
    assert before["status"] == "ADJUDICATION_REQUIRED"
    assert all(r["kind"] == "ORACLE_DISAGREEMENT" for r in before["findings"])
    fx.adjudicate()
    after = review_audit.aggregate_review(fx.root)
    assert after["status"] == "REVIEW_COMPLETE_NOT_GATE_SIGNED"
    assert len(after["findings"]) == 360  # Original errors remain in the record.


def test_confirmed_production_error_remains_blocking(fx):
    fx.submit(oracle=dict(answer="2", selected=["X"], bindings={"X": "1"}))
    fx.adjudicate(decision="PRODUCTION_INCORRECT")
    out = review_audit.aggregate_review(fx.root)
    assert out["status"] == "BLOCKING_FINDINGS" and not out["completed"]


@pytest.mark.parametrize("kind", ["empty", "early", "same_instance"])
def test_adjudication_cannot_hide_inputs_or_reuse_original_instance(fx, kind):
    fx.submit(oracle=dict(answer="2", selected=["X"], bindings={"X": "1"}))
    ref = fx.adjudicate()
    value = fx.read(ref)
    trace = fx.read(value["reviewer"]["transcript"])
    attempt_path = "REVIEW/EXECUTIONS/adj-1/ATTEMPT.json"
    attempt = fx.read({"path": attempt_path})
    if kind == "empty":
        value["input_references"] = attempt["input_references"] = []
        trace["events"][0]["inputs"] = []
    elif kind == "early":
        trace["events"][0]["started_at"] = "2026-10-29T02:30:00+00:00"
        attempt["started_at"] = "2026-10-29T02:00:00+00:00"
    else:
        origin = fx.read(value["reviewer"]["source"])
        origin["instance_id"] = trace["instance_id"] = attempt["instance_id"] = "instance-task-1"
        value["reviewer"]["source"] = fx.write(value["reviewer"]["source"]["path"], origin)
        trace["source"] = value["reviewer"]["source"]
    fx.write(attempt_path, attempt)
    value["reviewer"]["transcript"] = fx.write(value["reviewer"]["transcript"]["path"], trace)
    fx.write(ref["path"], value)
    with pytest.raises((PermissionError, ValueError)):
        review_audit.aggregate_review(fx.root)


def test_saved_pass_without_real_trace_cannot_be_verified(fx):
    fx.write(
        "REVIEW/AUDIT/SUMMARY.json",
        {"status": "REVIEW_COMPLETE_NOT_GATE_SIGNED", "completed": True},
    )
    with pytest.raises((PermissionError, ValueError, FileNotFoundError)):
        review_audit.verify_completed(fx.root)


def test_two_real_format_shards_cover_exact_360_and_duplicate_attempt_rejected(fx):
    fx.submit(ids={r["anonymous_id"] for r in fx.blind[:180]})
    incomplete = review_audit.aggregate_review(fx.root)
    assert incomplete["status"] == "REVIEW_INCOMPLETE"
    fx.submit(task="task-2", ids={r["anonymous_id"] for r in fx.blind[180:]})
    report = review_audit.aggregate_review(fx.root)
    assert report["status"] == "REVIEW_COMPLETE_NOT_GATE_SIGNED"
    assert report["execution_inventory"]["executions"] == 2
    with pytest.raises(FileExistsError):
        review_audit.prepare_execution(
            fx.root, task_id="task-1", instance_id="new", reviewer_id="new"
        )


def test_private_cache_checks_upstream_and_does_not_accept_disk_status(fx):
    fx.submit()
    review_audit.seal(fx.root)
    fx.consumer["registration_sha256"] = "0" * 64
    with pytest.raises((PermissionError, ValueError), match="upstream binding"):
        review_audit.verify_completed(fx.root)


def test_unlinked_partial_stage_cannot_hide_beside_final_audit(fx):
    fx.submit()
    review_audit.seal(fx.root)
    review_audit._CACHE.clear()
    path = fx.root / "REVIEW/.AUDIT.staging"
    path.mkdir(mode=0o700)
    (path / "partial").write_text("preserve")
    with pytest.raises((PermissionError, ValueError)):
        review_audit.verify_completed(fx.root)
    assert (path / "partial").read_text() == "preserve"


def test_cache_install_rejects_mutation_during_cold_verification(fx, monkeypatch):
    fx.submit()
    review_audit.seal(fx.root)
    review_audit._CACHE.clear()
    original = review_audit._audit_files

    def changed(report):
        result = original(report)
        fx.write("REVIEW/EXECUTIONS/task-1/EXTRA.json", {"change": "after validation"})
        return result

    monkeypatch.setattr(review_audit, "_audit_files", changed)
    with pytest.raises((PermissionError, ValueError), match="during cold verification"):
        review_audit.verify_completed(fx.root)
    assert not review_audit._CACHE
