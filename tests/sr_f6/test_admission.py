"""Synthetic CPU contracts only; no fixture can seal actual pilot admission."""

from __future__ import annotations

import hashlib
import json
import os

import pytest

from sr_f6 import admission
from sr_f6.common import file_hash


def save(root, relative, value, *, lines=False):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row) + "\n" for row in value) if lines else json.dumps(value)
    path.write_text(text)
    return dict(path=relative, sha256=file_hash(path))


def load(root, ref, *, lines=False):
    text = (root / ref["path"]).read_text()
    return [json.loads(line) for line in text.splitlines()] if lines else json.loads(text)


@pytest.fixture
def admission_request():
    return dict(
        phase_identity_sha256="a" * 64,
        pilot_spec_sha256="b" * 64,
        package_ref=dict(path="package.json", sha256="c" * 64),
        review_refs=[],
        replay_spec_sha256="d" * 64,
        numerical_receipts=[],
        adjudication_refs=[],
    )


@pytest.fixture
def synthetic_assessor(tmp_path, monkeypatch, admission_request):
    """Unit boundaries are simulated; the marker survives every apparent PASS."""
    config = save(tmp_path, "CONFIG.json", {"execution_origin": "SYNTHETIC_TEST"})
    (tmp_path / "PHASES/PILOT_V1").mkdir(parents=True)
    identity = dict(
        source_manifest=dict(path="source.json", sha256="f" * 64),
        origin=dict(deployment=str(tmp_path / "old")),
    )
    phase = dict(identity=identity, identity_sha256="a" * 64, deployment=str(tmp_path / "new"))

    def source(facts, pin):
        facts.synthetic = True
        return phase, dict(config_sha256=config["sha256"])

    monkeypatch.setattr(admission, "_phase", source)
    for name in ("_source", "_historical", "_processor", "_native", "_loaded"):
        monkeypatch.setattr(admission, name, lambda *args: dict(status="PASS"))
    monkeypatch.setattr(admission, "_data", lambda *args: (dict(status="PASS"), []))
    monkeypatch.setattr(
        admission, "_measurement", lambda *args: dict(status="PASS", support=dict(status="PASS"))
    )
    monkeypatch.setattr(
        admission, "_numerical", lambda *args: dict(status="PASS", scope="SYNTHETIC_TEST")
    )
    monkeypatch.setattr(
        admission,
        "_registration",
        lambda *args: dict(
            status="VERIFIED_REGISTRATION_NOT_AUTHORIZED",
            spec={"execution_origin": "SYNTHETIC_TEST"},
        ),
    )
    return tmp_path, admission_request


def test_all_apparent_pass_synthetic_evidence_never_authorizes_or_seals(synthetic_assessor):
    root, request = synthetic_assessor
    result = admission.assess_pilot_admission(root, **request)
    assert result["scientific_status"] == "PASS" and result["status"] == "SYNTHETIC_ONLY"
    assert not result["training_authorized"] and result["gate_status"] == "NOT_SIGNED"
    sealed = admission.seal_pilot_admission(root, **request)
    assert sealed["status"] == "SYNTHETIC_ONLY" and not (root / admission.PATH).exists()
    assert (
        sealed["assessment_ref"]
        and (root / sealed["assessment_ref"]["path"]).stat().st_mode & 0o777 == 0o600
    )
    with pytest.raises(PermissionError, match="actual recomputed PASS"):
        admission._receipt(root, result)
    result["status"] = "PASS"
    with pytest.raises(PermissionError, match="actual recomputed PASS"):
        admission._receipt(root, result)


@pytest.mark.parametrize("state", ["FAIL", "INCOMPLETE", "INCONCLUSIVE"])
def test_numerical_failure_unknown_and_probe_inconclusive_are_retained(
    synthetic_assessor, monkeypatch, state
):
    root, request = synthetic_assessor
    monkeypatch.setattr(
        admission,
        "_numerical",
        lambda *args: dict(status=state, failed_attempts=["retained-attempt"]),
    )
    result = admission.assess_pilot_admission(root, **request)
    assert result["checks"]["numerical"]["status"] == state
    assert result["scientific_status"] == ("FAIL" if state == "FAIL" else "UNKNOWN")
    assert result["status"] == "SYNTHETIC_ONLY" and not result["training_authorized"]


def test_insufficient_support_is_not_hidden_by_complete_review(synthetic_assessor, monkeypatch):
    root, request = synthetic_assessor
    monkeypatch.setattr(
        admission,
        "_measurement",
        lambda *args: dict(
            status="PASS",
            support=dict(
                status="INSUFFICIENT_REGISTERED_EXPOSURE",
                failures=[dict(name="b0/P0.S_known", observed=1382, minimum=1383)],
            ),
        ),
    )
    report = admission.assess_pilot_admission(root, **request)
    assert report["scientific_status"] == "INSUFFICIENT_REGISTERED_EXPOSURE"
    assert report["checks"]["support"]["failures"][0]["observed"] == 1382


def test_transport_failure_is_unknown_not_scientific_infeasibility(synthetic_assessor, monkeypatch):
    root, request = synthetic_assessor

    def unfinished(*args):
        raise admission._IncompleteEvidence("preserved interrupted upload")

    monkeypatch.setattr(admission, "_measurement", unfinished)
    report = admission.assess_pilot_admission(root, **request)
    assert report["checks"]["measurement"]["status"] == "UNKNOWN"
    assert report["scientific_status"] == "UNKNOWN"


def test_missing_real_source_has_explicit_missing_checks_and_no_pass(tmp_path, admission_request):
    result = admission.assess_pilot_admission(tmp_path, **admission_request)
    assert result["status"] != "PASS" and not result["training_authorized"]
    assert result["checks"]["actual_phase"]["status"] in ("FAIL", "UNKNOWN")
    assert result["checks"]["measurement"]["status"] == "UNKNOWN"


def test_handwritten_pass_and_disk_cache_never_substitute_for_actual_facts(synthetic_assessor):
    root, request = synthetic_assessor
    ref = save(
        root,
        admission.PATH,
        dict(schema=admission.SCHEMA, status="PASS", request=request, training_authorized=True),
    )
    save(root, ".verification-cache/admission.json", dict(status="PASS", admission=ref))
    with pytest.raises(PermissionError, match="facts do not pass"):
        admission.require_pilot_admission(
            root,
            phase_identity_sha256=request["phase_identity_sha256"],
            admission_ref=ref,
            pilot_spec_sha256=request["pilot_spec_sha256"],
        )
    assert not admission._CACHE


@pytest.mark.parametrize("pin", ["phase_identity_sha256", "pilot_spec_sha256"])
def test_runtime_wrong_phase_or_spec_pin_rejected(synthetic_assessor, pin):
    root, request = synthetic_assessor
    ref = save(root, admission.PATH, dict(schema=admission.SCHEMA, request=request))
    args = {key: request[key] for key in ("phase_identity_sha256", "pilot_spec_sha256")}
    args[pin] = "0" * 64
    with pytest.raises(PermissionError, match="pins differ"):
        admission.require_pilot_admission(root, admission_ref=ref, **args)


class TraceFixture:
    def __init__(self, root):
        self.root = root
        self.task = "review-task-1"
        self.prefix = admission.REVIEW_DIRECTORY + "/" + self.task
        self.package_ref = dict(path="package.json", sha256="a" * 64)
        self.method = save(
            root,
            self.prefix + "/METHOD.json",
            {"method": "Independent visible input extraction and calculation"},
        )
        self.source = save(
            root,
            self.prefix + "/SOURCE.json",
            dict(
                execution_origin="ACTUAL_REVIEW",
                status="SYNTHETIC_ONLY",
                instance_id="separate-instance-1",
                task_id=self.task,
                reviewer_type="MODEL_TEXT",
            ),
        )
        blind = [
            dict(
                anonymous_id="R0001",
                raw_text='{"answer":1}',
                protocol_version="structured_v4-srf6-1.0",
                generation_finish_reason="balanced",
            )
        ]
        self.blind = save(root, self.prefix + "/BLIND.jsonl", blind, lines=True)
        self.packet = save(
            root,
            "package/tasks.jsonl",
            [dict(anonymous_id="R0001", world={"a": 1}, query={"sum": True})],
            lines=True,
        )
        self.extract = save(
            root,
            self.prefix + "/EXTRACTION.json",
            dict(
                stage="BLIND_EXTRACTION",
                completed_at="2026-01-01T01:00:00+00:00",
                items=[dict(anonymous_id="R0001", value=1)],
            ),
        )
        self.judgment = save(
            root,
            self.prefix + "/JUDGMENT.json",
            dict(
                stage="INDEPENDENT_JUDGMENT",
                started_at="2026-01-01T02:00:00+00:00",
                completed_at="2026-01-01T03:00:00+00:00",
                items=[dict(anonymous_id="R0001", answer="1")],
            ),
        )
        self.trace = dict(
            schema=admission.TRACE_SCHEMA,
            execution_origin="ACTUAL_REVIEW",
            reviewer_id="independent-1",
            reviewer_type="MODEL_TEXT",
            identity="Actual platform instance declared in synthetic fixture",
            model_identity="NOT_EXPOSED",
            instance_id="separate-instance-1",
            task_id=self.task,
            method=self.method,
            source=self.source,
            events=[],
        )
        self.trace["events"] = [
            self.event("BLIND_EXTRACTION", [self.blind], self.extract, "2026-01-01T00:30:00+00:00"),
            self.event(
                "INDEPENDENT_JUDGMENT",
                [self.blind, self.packet, self.extract],
                self.judgment,
                "2026-01-01T02:00:00+00:00",
            ),
        ]
        self.submission = dict(
            execution_origin="ACTUAL_REVIEW",
            reviewer=dict(
                id="independent-1",
                type="MODEL_TEXT",
                identity=self.trace["identity"],
                model_identity="NOT_EXPOSED",
                method=self.method,
                source=self.source,
            ),
            blind_input=self.blind,
            blind_extraction=self.extract,
            judgments=self.judgment,
        )
        self.persist()

    def event(self, stage, inputs, output, started):
        return dict(
            stage=stage,
            started_at=started,
            completed_at=load(self.root, output)["completed_at"],
            inputs=[
                dict(
                    reference=ref,
                    content=load(self.root, ref, lines=ref["path"].endswith(".jsonl")),
                )
                for ref in inputs
            ],
            output=dict(reference=output, content=load(self.root, output)),
        )

    def persist(self):
        self.submission["reviewer"]["transcript"] = save(
            self.root, self.prefix + "/TRACE.json", self.trace
        )
        self.ref = save(self.root, self.prefix + "/SUBMISSION.json", self.submission)
        save(
            self.root,
            self.prefix + "/ATTEMPT.json",
            dict(
                execution_origin="ACTUAL_REVIEW",
                task_id=self.task,
                instance_id=self.trace["instance_id"],
                reviewer_id="independent-1",
                phase_identity_sha256="b" * 64,
                package=self.package_ref,
                started_at="2026-01-01T00:00:00+00:00",
            ),
        )

    def verify(self):
        facts = admission._Facts(self.root)
        trace = admission._trace(facts, self.submission, dict(judgment_tasks=self.packet))
        assert facts.synthetic  # A structurally complete test is never an actual review.
        return facts, trace


@pytest.fixture
def trace(tmp_path):
    return TraceFixture(tmp_path)


def test_not_exposed_model_identity_can_have_bound_actual_instance_and_payload_trace(trace):
    facts, result = trace.verify()
    assert result["visible_payloads_verified"] and result["reviewer_type"] == "MODEL_TEXT"
    assert trace.submission["reviewer"]["transcript"]["path"] in facts.inputs
    inventory = admission._review_inventory(
        facts, trace.package_ref, [trace.ref], [], [result], "b" * 64
    )
    assert inventory == dict(executions=1, instances=1)


@pytest.mark.parametrize(
    "mutation", ["program", "synthetic", "input_gold", "output", "instance", "chronology", "empty"]
)
def test_declared_independence_cannot_replace_actual_trace_binding(trace, mutation):
    if mutation == "program":
        trace.submission["reviewer"]["type"] = "PROGRAM"
    elif mutation == "synthetic":
        trace.submission["execution_origin"] = "SYNTHETIC_TEST"
    elif mutation == "input_gold":
        trace.trace["events"][0]["inputs"].append(
            dict(reference=trace.packet, content=load(trace.root, trace.packet, lines=True))
        )
    elif mutation == "output":
        trace.trace["events"][0]["output"]["content"]["items"][0]["value"] = 999
    elif mutation == "instance":
        trace.trace["instance_id"] = "unbound-instance"
    elif mutation == "chronology":
        trace.trace["events"][1]["started_at"] = "2025-12-31T00:00:00+00:00"
    else:
        trace.trace["events"] = []
    trace.persist()
    with pytest.raises(PermissionError):
        trace.verify()


@pytest.mark.parametrize("kind", ["unfinished", "failure", "unreported"])
def test_all_review_attempts_are_enumerated_and_technical_incidents_retained(trace, kind):
    facts, result = trace.verify()
    if kind == "unfinished":
        (trace.root / trace.ref["path"]).unlink()
    elif kind == "failure":
        save(
            trace.root,
            trace.prefix + "/FAILURE.json",
            dict(status="FAIL", reason="Synthetic transport failure"),
        )
    else:
        ref = trace.prefix + "/SUBMISSION.json"
        trace.ref = dict(path="omitted.json", sha256="f" * 64)
        assert (trace.root / ref).exists()
    with pytest.raises(PermissionError):
        admission._review_inventory(facts, trace.package_ref, [trace.ref], [], [result], "b" * 64)
    if kind == "failure":
        assert (trace.root / trace.prefix / "FAILURE.json").exists()


def test_adjudication_trace_has_no_self_referential_manifest_hash(trace):
    response = save(
        trace.root,
        trace.prefix + "/DECISION_RESPONSE.json",
        dict(
            items=[
                dict(
                    finding_id="finding-1",
                    decision="UNRESOLVED",
                    rationale="synthetic ambiguity remains",
                )
            ],
            completed_at="2026-01-01T05:00:00+00:00",
        ),
    )
    actual = load(trace.root, response)
    trace.submission.update(
        actual, input_references=[trace.extract, trace.judgment], decision_response=response
    )
    trace.trace["events"] = [
        trace.event(
            "ADJUDICATION", [trace.extract, trace.judgment], response, "2026-01-01T04:00:00+00:00"
        )
    ]
    trace.persist()
    facts = admission._Facts(trace.root)
    result = admission._trace(facts, trace.submission, {}, adjudication=True)
    assert result["visible_payloads_verified"] and facts.synthetic
    assert response["path"] in facts.inputs


def test_metadata_cache_reads_only_changed_content_and_detects_new_attempt(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    path.write_text('{"synthetic":true}')
    phase = dict(
        deployment=str(tmp_path / "new"),
        identity=dict(origin=dict(deployment=str(tmp_path / "old"))),
    )
    cached = dict(
        result=dict(phase_context=phase),
        files={str(path): dict(sha256=file_hash(path), stat=admission._stat(path))},
        trees=admission._trees(tmp_path, phase),
    )
    calls, original = [], admission.file_hash
    monkeypatch.setattr(admission, "file_hash", lambda p: (calls.append(str(p)), original(p))[1])
    assert admission._cache_unchanged(tmp_path, cached) and calls == []
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    assert admission._cache_unchanged(tmp_path, cached) and calls == [str(path)]
    assert admission._cache_unchanged(tmp_path, cached) and calls == [str(path)]
    path.write_text('{"synthetic":false}')
    with pytest.raises(PermissionError, match="changed after"):
        admission._cache_unchanged(tmp_path, cached)
    path.write_text('{"synthetic":true}')
    save(
        tmp_path,
        admission.REVIEW_DIRECTORY + "/new-attempt/ATTEMPT.json",
        dict(status="SYNTHETIC_ONLY"),
    )
    assert not admission._cache_unchanged(tmp_path, cached)


def test_new_source_file_invalidates_process_cache_inventory(tmp_path):
    phase = dict(
        deployment=str(tmp_path / "new"),
        identity=dict(origin=dict(deployment=str(tmp_path / "old"))),
    )
    before = admission._trees(tmp_path, phase)
    source = tmp_path / "new/src/sr_f6/new_module.py"
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic code fixture\n")
    assert admission._trees(tmp_path, phase) != before


def test_training_outputs_do_not_become_admission_dependencies(tmp_path, monkeypatch):
    paths = []
    for parent in ("b0", "b1", "b2"):
        for arm in ("U10", "U11", "C_SHUF"):
            relative = f"runs/{parent}/r0/{arm}"
            refs = {}
            for key, name in (
                ("transaction_identity", "RUN_IDENTITY.json"),
                ("rollout_registration", "ROLLOUT_REGISTRATION.json"),
            ):
                ref = save(
                    tmp_path,
                    f"{admission.registration.DIRECTORY}/{relative}/{name}",
                    dict(status="SYNTHETIC_ONLY"),
                )
                refs[key] = dict(path=f"{relative}/{name}", sha256=ref["sha256"])
            paths.append(refs)
    parent_refs = {
        p: save(tmp_path, f"runtime_identity/{p}.json", dict(status="SYNTHETIC_ONLY"))
        for p in ("b0", "b1", "b2")
    }
    spec = dict(paths=paths, parent_references=parent_refs, execution_origin="SYNTHETIC_TEST")
    ref = save(tmp_path, admission.SPEC_PATH, spec)
    monkeypatch.setattr(
        admission.registration,
        "verify_pilot_registration",
        lambda *args, **kwargs: dict(
            status="VERIFIED_REGISTRATION_NOT_AUTHORIZED", spec_sha256=ref["sha256"], spec=spec
        ),
    )
    first = admission._Facts(tmp_path)
    admission._registration(first, {}, ref["sha256"])
    before = admission._trees(tmp_path)
    root = admission.registration.DIRECTORY + "/runs/b0/r0/U11"
    for name in (
        "TRAINER_ATTEMPTS/task/STARTED.json",
        "runtime_identity/task.json",
        "checkpoints/step-000/COMMIT.json",
    ):
        save(tmp_path, root + "/" + name, dict(status="SYNTHETIC_ONLY"))
    second = admission._Facts(tmp_path)
    admission._registration(second, {}, ref["sha256"])
    assert second.inputs == first.inputs and len(second.inputs) == 22
    assert admission._trees(tmp_path) == before


@pytest.fixture
def processor_fixture(tmp_path, monkeypatch):
    from PIL import Image

    from srf6_ref.core import FAMILIES

    origin = dict(config_sha256="a" * 64, data_manifest=dict(sha256="b" * 64))
    source = tmp_path / "DATA/images/source.png"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (1280, 896), "white").save(source)
    processed = tmp_path / "PREFLIGHT/processor/view.png"
    processed.parent.mkdir(parents=True)
    image = Image.new("RGB", (1056, 736), "white")
    image.save(processed)
    image_sha = hashlib.sha256(image.tobytes()).hexdigest()
    plan, images, reviews = [], [], []
    for family in FAMILIES:
        for index in range(8):
            qid = f"DEV_CAL_{family}_{index:04d}_P1_v0"
            plan.append(
                dict(
                    qid=qid,
                    family=family,
                    image_file="DATA/images/source.png",
                    image_sha256=file_hash(source),
                )
            )
            images.append(
                dict(
                    qid=qid,
                    family=family,
                    source_file="DATA/images/source.png",
                    source_sha256=file_hash(source),
                    processed_file="PREFLIGHT/processor/view.png",
                    processed_file_sha256=file_hash(processed),
                    processed_rgb_sha256=image_sha,
                    routing=dict(
                        source_image_sha256=file_hash(source),
                        processor_hash="c" * 64,
                        chat_template_hash="d" * 64,
                        native_input_tensor_hashes=dict.fromkeys(
                            (
                                "input_ids",
                                "attention_mask",
                                "mm_token_type_ids",
                                "pixel_values",
                                "image_grid_thw",
                            ),
                            "e" * 64,
                        ),
                        input_float_dtype="float32",
                        probability_dtype="float32",
                        sr_f6_protocol_version="structured_v4-srf6-1.0",
                        source_size=[1280, 896],
                        processed_size=[1056, 736],
                        image_grid_thw=[[1, 46, 66]],
                        image_token_count=759,
                    ),
                )
            )
            reviews.append(
                dict(
                    qid=qid,
                    status="PASS",
                    observation="SYNTHETIC FIXTURE; not an actual visual inspection",
                    reviewed_source=True,
                    reviewed_processed=True,
                    issues=[],
                    source_sha256=file_hash(source),
                    processed_file_sha256=file_hash(processed),
                )
            )
    report = dict(
        status="SYNTHETIC_ONLY",
        execution_origin="SERVER_ACTUAL",
        config_file_sha256="a" * 64,
        data_manifest_sha256="b" * 64,
        images=images,
        tokenizer_fixtures=[],
        identity=dict(
            architecture_verified=True,
            model_type="qwen3_5",
            runtime_recipe="SR-F6-native-full-sequence-1536",
            processor_hash="c" * 64,
            chat_template_hash="d" * 64,
            generation_config_expanded=dict(
                max_new_tokens=1536,
                do_sample=True,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                remove_invalid_values=False,
            ),
        ),
    )
    processor_ref = save(tmp_path, "PREFLIGHT/PROCESSOR.json", report)
    save(tmp_path, "DATA/PROCESSOR_REVIEW_PLAN.json", dict(items=plan))
    visual = dict(
        status="PASS",
        reviewer_type="MODEL_VISUAL",
        method="SYNTHETIC fixture only",
        processor_receipt_sha256=processor_ref["sha256"],
        items=reviews,
    )
    save(tmp_path, "PREFLIGHT/VISUAL_REVIEW.json", visual)
    monkeypatch.setattr(admission, "_tokenizer_roundtrips", lambda facts, fixtures: None)
    return tmp_path, origin, report, visual


def test_processor_rechecks_48_actual_image_hashes_pixels_and_routing(processor_fixture):
    root, origin, _, _ = processor_fixture
    facts = admission._Facts(root)
    report = admission._processor(facts, origin)
    assert report["images"] == 48 and facts.synthetic


@pytest.mark.parametrize(
    "mutation", ["coverage", "unviewed", "rgb", "grid", "recipe", "program_reviewer"]
)
def test_processor_pass_string_cannot_override_missing_visual_or_native_facts(
    processor_fixture, mutation
):
    root, origin, report, visual = processor_fixture
    if mutation == "coverage":
        report["images"].pop()
    elif mutation == "unviewed":
        visual["items"][0]["reviewed_processed"] = False
    elif mutation == "rgb":
        report["images"][0]["processed_rgb_sha256"] = "f" * 64
    elif mutation == "grid":
        report["images"][0]["routing"]["image_token_count"] = 758
    elif mutation == "recipe":
        report["identity"]["generation_config_expanded"]["max_new_tokens"] = 768
    else:
        visual["reviewer_type"] = "PROGRAM"
    ref = save(root, "PREFLIGHT/PROCESSOR.json", report)
    visual["processor_receipt_sha256"] = ref["sha256"]
    save(root, "PREFLIGHT/VISUAL_REVIEW.json", visual)
    with pytest.raises(PermissionError):
        admission._processor(admission._Facts(root), origin)


def test_native_parity_recomputes_300_fixtures_600_mutations_without_real_model(tmp_path):
    from srf6_ref.design import build_bank

    roots = build_bank("DEV_CAL", 8, 2026102711)
    tasks = [task for root in roots for factor in ("P0", "P1") for task in root["tasks"][factor]]
    save(tmp_path, "DATA/DEV_CAL/TASKS_PRIVATE.jsonl", tasks, lines=True)
    origin = dict(config_sha256="a" * 64, source_manifest=dict(sha256="b" * 64))
    log = tmp_path / "PREFLIGHT/NATIVE_REGRESSION.log"
    log.parent.mkdir(parents=True)
    log.write_text("SYNTHETIC RECEIPT FIXTURE\n83 passed\n")
    save(
        tmp_path,
        "PREFLIGHT/NATIVE_REGRESSION.json",
        dict(
            status="PASS",
            execution_origin="SYNTHETIC_TEST",
            returncode=0,
            model_calls=0,
            optimizer_updates=0,
            config_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
            log_sha256=file_hash(log),
            command=["tests/sr_f6/test_measurement.py"],
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:01:00+00:00",
        ),
    )
    facts = admission._Facts(tmp_path)
    report = admission._native(facts, origin)
    assert report["native_fixtures"] == 300 and report["valid_shape_mutations"] == 600
    assert set(report["family_counts"].values()) == {50} and facts.synthetic


def test_admission_cache_rejects_synthetic_insertion(synthetic_assessor):
    root, request = synthetic_assessor
    assessment = admission.assess_pilot_admission(root, **request)
    with pytest.raises(PermissionError, match="Only recomputed actual"):
        admission._remember(
            root, dict(admission_ref=dict(path=admission.PATH, sha256="a" * 64)), assessment
        )


@pytest.mark.parametrize(
    "payload", ['{"status":"PASS","status":"FAIL"}', '{"value":1e999}', '{"value":-Infinity}']
)
def test_admission_rejects_ambiguous_or_nonfinite_receipt_containers(tmp_path, payload):
    (tmp_path / "receipt.json").write_text(payload)
    with pytest.raises((PermissionError, ValueError)):
        admission._Facts(tmp_path).read("receipt.json")
