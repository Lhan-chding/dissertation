"""Small CPU-only delivery fixtures; no fabricated experimental completion."""

import copy
import gzip
import importlib.util
import json
import py_compile
import struct
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_path = Path(__file__).resolve().parents[2] / "scripts/sr_f3_ce/build_delivery.py"
_spec = importlib.util.spec_from_file_location("srf3_delivery_auxiliary", _path)
delivery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(delivery)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(delivery.encoded(value) + b"\n")


def release_fixture(root):
    directory = root / "release"
    directory.mkdir()
    slot = dict(
        slot_id="s0",
        qid="q0",
        draw=0,
        model_id="CPU_FIXTURE_ONLY",
        pool="TEST_ID",
        protocol="structured",
        sealed=True,
    )
    row = dict(
        slot,
        group_id="g0",
        generation_status="COMPLETE",
        technical_validation_errors=[],
        tokens=[1],
        max_new_tokens=1024,
        model_input={"prompt": "fixture"},
        raw_text="invalid answer",
        finish_reason="eos",
    )
    score = dict(value_correct=0, presentation=0, J=0)
    compact = dict(
        slot,
        group_id="g0",
        generation_status="COMPLETE",
        technical_validation_errors=[],
        score=score,
    )
    with gzip.open(directory / "ENDPOINT_RAW.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    (directory / "ENDPOINT_SCORES.jsonl").write_bytes(delivery.encoded(compact) + b"\n")
    score_rows = [dict(slot_id="s0", score=score)]
    write(
        directory / "scores/g0.json",
        dict(
            raw_sha256="original-group-sha", scores=score_rows, scorer_version=delivery.EXPERIMENT
        ),
    )
    release = dict(
        status="RELEASED",
        experiment_id=delivery.EXPERIMENT,
        records=1,
        raw_groups={"g0": delivery.digest([row])},
    )
    write(directory / "RELEASE.json", release)
    rescoring = dict(
        status="PASS",
        records=1,
        mismatches=[],
        gzip_read_to_eof=True,
        raw_sha256=delivery.sha_file(directory / "ENDPOINT_RAW.jsonl.gz"),
        group_scores_sha256=delivery.digest({"g0": delivery.digest(score_rows)}),
        independently_implemented_scorer_claimed=False,
    )
    write(directory / "RAW_RESCORING.json", rescoring)
    write(
        directory / "RELEASE_SCORED.json",
        dict(
            release,
            raw_rescoring="PASS",
            projection_kind="verified_raw_rescore_projection_v1",
            analysis_rows_sha256=delivery.digest([compact]),
            rescoring_receipt_sha256=delivery.sha_file(directory / "RAW_RESCORING.json"),
        ),
    )
    for name in ("PRIMARY_ANALYSIS.json", "EXTERNAL_ANALYSIS.json", "PREFIX_DIAGNOSTICS.json"):
        write(directory / name, dict(scope="CPU_FIXTURE_ONLY"))
    refresh_manifest(root)
    groups = [(dict(group_id="g0", slots=[slot]), [row], dict(sha256="original-group-sha"))]
    return groups, {"q0": {}}, lambda *_: score


def refresh_manifest(root):
    write(
        root / "release/MANIFEST.json",
        dict(
            files=[
                dict(
                    path=name,
                    sha256=delivery.sha_file(root / "release" / name),
                    bytes=(root / "release" / name).stat().st_size,
                )
                for name in delivery.RELEASE_FILES
            ]
        ),
    )


def test_sealed_gate_uses_controller_not_status_labels_or_file_existence(tmp_path):
    write(tmp_path / "release/RELEASE.json", dict(status="RELEASED"))
    calls = []
    api = SimpleNamespace(
        protocol=SimpleNamespace(verify_run=lambda *a, **k: calls.append("source")),
        endpoint_runner=SimpleNamespace(_completed=lambda root: set()),
        orchestration=SimpleNamespace(TASKS={"release": {}}),
    )
    with pytest.raises(PermissionError, match="sealed/incomplete"):
        delivery.require_final_release(tmp_path, api)
    assert calls == ["source"]


def test_stream_reopens_raw_and_retains_invalid_model_answers(tmp_path):
    groups, tasks, scorer = release_fixture(tmp_path)
    result = delivery.verify_release_stream(tmp_path, iter(groups), tasks, scorer)
    assert result["records"] == 1
    assert result["independent_scorer_claimed"] is False
    assert result["full_original_rescoring"] == "PASS"
    assert groups[0][1][0]["raw_text"] == "invalid answer"


@pytest.mark.parametrize("change", ["raw", "score", "missing", "sealed", "extra"])
def test_release_missing_sealed_or_tampered_content_is_rejected(tmp_path, change):
    groups, tasks, scorer = release_fixture(tmp_path)
    directory = tmp_path / "release"
    if change == "missing":
        (directory / "ENDPOINT_RAW.jsonl.gz").unlink()
    elif change == "sealed":
        value = delivery.read(directory / "RELEASE.json")
        value["status"] = "SEALED_COMPLETE"
        write(directory / "RELEASE.json", value)
        refresh_manifest(tmp_path)
    elif change in ("raw", "extra"):
        changed = copy.deepcopy(groups[0][1][0])
        changed["raw_text"] = "changed" if change == "raw" else "invalid answer"
        with gzip.open(directory / "ENDPOINT_RAW.jsonl.gz", "wt") as stream:
            stream.write(json.dumps(changed) + "\n")
            if change == "extra":
                stream.write(json.dumps(changed) + "\n")
    else:
        value = delivery.loads((directory / "ENDPOINT_SCORES.jsonl").read_text())
        value["score"]["J"] = 1
        write(directory / "ENDPOINT_SCORES.jsonl", value)
        refresh_manifest(tmp_path)
    with pytest.raises((PermissionError, FileNotFoundError)):
        delivery.verify_release_stream(tmp_path, iter(groups), tasks, scorer)


def make_zip(path, *, link="raw.jsonl.gz"):
    writer = delivery.DeliveryZip(path)
    writer.add_bytes("START_HERE.md", f"[raw]({link})".encode())
    writer.add_bytes("REPORT_ZH.md", b"CPU fixture only")
    writer.add_jsonl_gzip("raw.jsonl.gz", iter([dict(raw_text="failure retained", score=0)]))
    writer.finish()


def test_stream_zip_crc_manifest_eof_and_entry_links(tmp_path):
    path = tmp_path / "fixture.zip"
    make_zip(path)
    result = delivery.verify_zip(path)
    assert set(result["checks"].values()) == {"PASS"}
    assert result["aggregate_counts"]["raw.jsonl.gz"]["rows"] == 1
    with (
        zipfile.ZipFile(path) as archive,
        gzip.GzipFile(fileobj=archive.open("raw.jsonl.gz")) as stream,
    ):
        assert delivery.loads(stream.read())["score"] == 0


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "a//b", "C:/escape", "a\\b"])
def test_unsafe_archive_names_rejected(name):
    with pytest.raises(PermissionError, match="Unsafe"):
        delivery.safe_name(name)


def test_duplicate_zip_path_and_bad_links_rejected(tmp_path):
    path = tmp_path / "duplicate.zip"
    make_zip(path)
    with zipfile.ZipFile(path, "a") as archive, pytest.warns(UserWarning, match="Duplicate"):
        archive.writestr("REPORT_ZH.md", "duplicate")
    with pytest.raises(PermissionError, match="Duplicate"):
        delivery.verify_zip(path)
    missing = tmp_path / "missing-link.zip"
    make_zip(missing, link="absent.json")
    with pytest.raises(PermissionError, match="entry link"):
        delivery.verify_zip(missing)


def test_zip_actual_crc_corruption_is_not_accepted(tmp_path):
    path = tmp_path / "crc.zip"
    make_zip(path)
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("raw.jsonl.gz")
    with path.open("r+b") as stream:
        stream.seek(info.header_offset)
        header = stream.read(30)
        filename, extra = struct.unpack_from("<HH", header, 26)
        offset = info.header_offset + 30 + filename + extra
        stream.seek(offset + 15)
        value = stream.read(1)
        stream.seek(offset + 15)
        stream.write(bytes([value[0] ^ 1]))
    with pytest.raises((PermissionError, zipfile.BadZipFile), match="CRC"):
        delivery.verify_zip(path)


def test_original_wrapper_preserves_exact_utf8_bytes(tmp_path):
    source = tmp_path / "raw/group.json"
    source.parent.mkdir()
    source.write_bytes(b'{\n  "records": [{"raw_text": "bad", "score": 0}]\n}\n')
    metadata = dict(
        task_sidecar={},
        _source_identity=dict(sha256=delivery.sha_file(source), bytes=source.stat().st_size),
    )
    row = next(delivery.wrapped_originals(tmp_path, [(source, metadata, 1)]))
    assert row["original_utf8"].encode() == source.read_bytes()
    assert row["source_sha256"] == delivery.sha_file(source)


def test_noncritical_historical_missing_is_retained_without_blocking(tmp_path):
    write(tmp_path / "prior/raw/FILE_MANIFEST.json", dict(files=[]))
    index = tmp_path / "prior/raw/RECORD_INDEX.jsonl"
    index.write_bytes(b"")
    write(
        tmp_path / "prior/RAW_EXPORT.json",
        dict(
            status="MISSING",
            operation_completed=True,
            critical=False,
            blocks_new_science=False,
            file_manifest="prior/raw/FILE_MANIFEST.json",
            record_index=dict(
                path="prior/raw/RECORD_INDEX.jsonl", sha256=delivery.sha_file(index), bytes=0
            ),
            items=[dict(status="MISSING", reason="16 historical roots; requested 32")],
            indexed_raw_records=0,
        ),
    )
    result = delivery.historical_status(tmp_path)
    assert result["status"] == "MISSING"
    assert "16 historical roots" in result["missing"][0]["reason"]


def analysis_fixture():
    analysis = dict(
        status="NUMERICAL_ANALYSIS_COMPLETE",
        experiment_id=delivery.EXPERIMENT,
        all_registered_blocks_retained=True,
        adverse_results_retained=True,
        primary={},
        secondary={},
        canonical_protocols={},
        canonical_protocol_rates={},
        decomposition={},
        retention={},
    )
    for i, (arm0, scope) in enumerate(
        (("PREFIX", "REPAIR"), ("GOLD", "REPAIR"), ("GOLD", "TEST_COMPOSITION")), 1
    ):
        analysis["primary"][f"C{i}"] = dict(
            arm1="EXEC",
            arm0=arm0,
            scope=scope,
            metric="value_correct",
            effect_pp=-2,
            per_block_pp=[-5, 2, -3],
            CI95_pp=[-6, 1],
            CI_simultaneous_pp=[-7, 2],
            paired_t_sensitivity=dict(df=2, CI95_pp=[-12, 8]),
            bootstrap_replicates=10000,
            bootstrap_seed=2026101651,
            confidence_simultaneous=1 - 0.05 / 3,
        )
    families = ("CROSS", "THRESHOLD", "TOPK", "INTERVAL")
    arms = ("PARENT", "RL", "GOLD", "PREFIX", "EXEC")
    metrics = (
        "value_correct",
        "certificate",
        "certificate_truth",
        "certificate_sufficient",
        "presentation",
        "finish",
        "J",
        "legacy_full",
        "value_parseable",
    )
    rate = dict(
        rate=0.1, per_block=[0.1, 0.1, 0.1], family_per_block={f: [0.1, 0.1, 0.1] for f in families}
    )
    contrast = copy.deepcopy(analysis["primary"]["C1"])
    for scope in (
        "REPAIR",
        "LEGACY",
        "TEST_COMPOSITION",
        "TEST_LANGUAGE",
        "TEST_RENDER",
        "TEST_ARITY",
    ):
        analysis["decomposition"][scope] = {
            arm: {m: copy.deepcopy(rate) for m in metrics} for arm in arms
        }
    for arm in arms[1:]:
        analysis["retention"][arm] = {
            m: dict(contrast, by_family={f: contrast for f in families}, worst_family="CROSS")
            for m in ("value_correct", "J")
        }
    for scope in ("REPAIR", "TEST_COMPOSITION"):
        for name in ("GOLD-RL", "PREFIX-GOLD", "EXEC-RL"):
            analysis["secondary"][scope + ":" + name] = contrast
    for protocol in ("structured", "answer_only", "natural_work"):
        for name in ("EXEC-PREFIX", "EXEC-GOLD"):
            analysis["canonical_protocols"][protocol + ":" + name] = contrast
        applicable = (
            metrics
            if protocol == "structured"
            else ("value_correct", "presentation", "finish", "value_parseable")
        )
        analysis["canonical_protocol_rates"][protocol] = {
            arm: {m: rate for m in applicable} for arm in arms
        }
    return analysis


def test_report_preserves_negative_blocks_uncertainty_and_unknown_costs():
    coverage = dict(
        training_paths=12,
        logical_updates=768,
        training_records=98304,
        support_examples=2304,
        monitor_records=26112,
        engine_diagnostic_records=2880,
        records=229500,
    )
    report = delivery.render_report(
        analysis_fixture(),
        {"CHARTQAPRO": {}},
        {"panels": {}},
        coverage,
        {"gpu_hours": "UNKNOWN"},
        {"status": "MISSING"},
        {"CHARTQAPRO": dict(status="UNAVAILABLE", reason="fixture")},
    )
    assert "-5.000 | +2.000 | -3.000" in report
    assert "本设置未确认" in report and "98.3333%" in report and "df=2" in report
    assert "UNKNOWN" in report and "MISSING" in report and "UNAVAILABLE" in report
    assert "没有声称新独立评分器" in report


def test_receipt_publication_is_immutable_and_recoverable_after_zip_publish(tmp_path):
    path = tmp_path / "fixture.zip"
    make_zip(path)
    checked = delivery.verify_zip(path)
    provenance = dict(
        run_identity_sha256="a" * 64,
        scientific_source_manifest_sha256="b" * 64,
        helper_manifest_sha256="c" * 64,
    )
    with pytest.raises(PermissionError, match="structural ZIP"):
        delivery._external_receipt(path, checked, provenance, {"scope": "CPU_FIXTURE_ONLY"})
    # Explicit fixture flag tests receipt mechanics only, not an experiment gate.
    checked["checks"]["raw_rescoring_chain"] = "PASS"
    receipt = delivery._external_receipt(path, checked, provenance, {"scope": "CPU_FIXTURE_ONLY"})
    assert (
        delivery._external_receipt(path, checked, provenance, {"scope": "CPU_FIXTURE_ONLY"})
        == receipt
    )
    with pytest.raises(PermissionError, match="Immutable"):
        delivery._external_receipt(
            path, checked, dict(provenance, helper_manifest_sha256="d" * 64), {}
        )


def test_active_panels_pointer_uses_typed_v1_and_retains_v0_without_counting_it(tmp_path):
    old = dict(outputs_unseen=True, rows=[dict(qid="q0", status="UNRESOLVED")], unresolved=1)
    write(tmp_path / "external/chartqa/TYPE_REGISTRY.json", old)
    current = dict(
        outputs_unseen=True,
        rows=[dict(qid="q1", status="RESOLVED"), dict(qid="q2", status="UNRESOLVED")],
        unresolved=1,
    )
    directory = tmp_path / "external/chartqa_typed_v1"
    write(directory / "TYPE_REGISTRY.json", current)
    write(
        directory / "FREEZE.json",
        dict(status="FROZEN", type_registry_sha256=delivery.digest(current)),
    )
    pro = tmp_path / "external/chartqapro_typed_v1"
    write(pro / "FREEZE.json", dict(status="UNAVAILABLE", image_count=0))
    panels = {
        pool: dict(
            root=str(path.relative_to(tmp_path)),
            freeze_sha256=delivery.sha_file(path / "FREEZE.json"),
        )
        for pool, path in (("CHARTQA", directory), ("CHARTQAPRO", pro))
    }
    write(tmp_path / "external/PANELS.json", panels)
    rows = list(delivery.unresolved_types(tmp_path))
    assert [row["qid"] for row in rows] == ["q2"]
    assert rows[0]["source_registry"] == "external/chartqa_typed_v1/TYPE_REGISTRY.json"
    current["unresolved"] = 0
    write(directory / "TYPE_REGISTRY.json", current)
    with pytest.raises(PermissionError, match="frozen panel"):
        list(delivery.unresolved_types(tmp_path))


def scheduler_fixture(root):
    for job, state in (("123", "COMPLETED"), ("124", "TIMEOUT")):
        directory = root / "orchestration/jobs" / job
        intent = dict(
            key=job,
            qos="teacher",
            job_name="SRF3-test-LH",
            gpu_count=2 if job == "123" else 1,
            task_ids=["fixture-task"],
        )
        write(directory / "INTENT.json", intent)
        write(
            directory / "STATE.json",
            dict(key=job, job_id=job, intent_sha256=delivery.sha_file(directory / "INTENT.json")),
        )
        write(
            directory / "TERMINAL.json",
            dict(
                job_id=job,
                state=state,
                exit_code="0:0",
                user="tester",
                account="rose",
                qos="teacher",
                name="SRF3-test-LH",
            ),
        )
    return (
        "123|COMPLETED|0:0|tester|rose|teacher|SRF3-test-LH|3600|"
        "2026-10-16T00:00:00|2026-10-16T00:10:00|2026-10-16T01:10:00|"
        "cpu=8,gres/gpu=2,gres/gpu:pro6000=2|\n"
        "124|TIMEOUT|0:0|tester|rose|teacher|SRF3-test-LH|1800|"
        "2026-10-16T00:00:00|2026-10-16T00:20:00|2026-10-16T00:50:00|gres/gpu=1|\n"
    )


def test_actual_allocated_costs_include_failed_attempt_without_requested_time(tmp_path):
    stdout = scheduler_fixture(tmp_path)
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["TZ"] == "UTC"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    result, raw = delivery.scheduler_costs(tmp_path, execute=execute)
    assert len(calls) == 1 and "123,124" in calls[0]
    assert raw["stdout"] == stdout
    assert result["totals"]["gpu_hours"] == 2.5
    assert result["successful_attempt_gpu_hours"] == 2
    assert result["failed_attempt_gpu_hours"] == 0.5
    assert result["totals"]["queue_seconds"] == 1800
    assert result["totals"]["total_wall_seconds"] == 4200


def test_scheduler_cost_identity_mismatch_cannot_become_unknown_or_pass(tmp_path):
    stdout = scheduler_fixture(tmp_path).replace("tester", "someone_else", 1)
    with pytest.raises(PermissionError, match="INTENT/TERMINAL"):
        delivery.scheduler_costs(
            tmp_path,
            execute=lambda *a, **k: SimpleNamespace(returncode=0, stdout=stdout, stderr=""),
        )


def test_unavailable_scheduler_cost_is_explicit_unknown_with_retained_error(tmp_path):
    scheduler_fixture(tmp_path)
    result, raw = delivery.scheduler_costs(
        tmp_path,
        execute=lambda *a, **k: SimpleNamespace(
            returncode=1, stdout="", stderr="accounting unavailable"
        ),
    )
    assert result["status"] == "UNKNOWN"
    assert raw["stderr"] == "accounting unavailable"
    assert "nonzero" in result["reason"]


def test_operations_snapshot_omits_mutable_progress_but_keeps_source_and_prior_end(tmp_path):
    directory = tmp_path / "operations/fixture"
    for name in (
        "OPERATION_IDENTITY.json",
        "OPERATION_SOURCES.json",
        "STARTED.json",
        "END.json",
        "PROGRESS.json",
        "QUEUE_SNAPSHOT.json",
    ):
        write(directory / name, dict(scope="CPU_FIXTURE_ONLY"))
    included, _ = delivery.original_inventory(tmp_path, {}, set(), {})
    assert "evidence/operations/fixture/END.json" in included
    assert "evidence/operations/fixture/OPERATION_SOURCES.json" in included
    assert not any(name.endswith(("PROGRESS.json", "QUEUE_SNAPSHOT.json")) for name in included)


def test_report_missing_entire_unfavorable_arm_is_rejected():
    analysis = analysis_fixture()
    del analysis["decomposition"]["REPAIR"]["RL"]
    with pytest.raises(PermissionError, match="omitted"):
        delivery._validate_analysis(analysis)


@pytest.mark.parametrize("allocation", ["", "cpu=8,mem=80G"])
def test_missing_actual_gpu_allocation_is_unknown_instead_of_zero(tmp_path, allocation):
    stdout = scheduler_fixture(tmp_path).replace("cpu=8,gres/gpu=2,gres/gpu:pro6000=2", allocation)
    result, raw = delivery.scheduler_costs(
        tmp_path, execute=lambda *a, **k: SimpleNamespace(returncode=0, stdout=stdout, stderr="")
    )
    assert result["status"] == "UNKNOWN"
    assert result["totals"]["gpu_hours"] == "UNKNOWN"
    assert result["jobs"][0]["status"] == "UNKNOWN"
    assert result["verified_subtotals"]["gpu_hours"] == 0.5
    assert raw["stdout"] == stdout


def test_hf_weight_symlink_is_server_only_by_frozen_filename(tmp_path):
    repository = tmp_path / "hf/model"
    blob = repository / "blobs" / ("a" * 64)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"synthetic weight fixture only")
    snapshot = repository / "snapshots/revision"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").symlink_to(blob)
    (snapshot / "config.json").write_bytes(b"{}")
    write(
        tmp_path / "prior/MODEL_ENVIRONMENT_IDENTITY.json",
        dict(
            model_path=str(snapshot),
            model_files=[
                dict(
                    name=name,
                    sha256=delivery.sha_file(snapshot / name),
                    bytes=(snapshot / name).stat().st_size,
                )
                for name in ("model.safetensors", "config.json")
            ],
        ),
    )
    included, omitted = delivery.model_sources(tmp_path)
    assert set(included) == {"model_source/base/config.json"}
    assert [row["name"] for row in omitted] == ["model.safetensors"]
    assert omitted[0]["server_path"] == str(blob)


def test_complete_zip_label_cannot_replace_required_raw_sidecar_inventory(tmp_path):
    path = tmp_path / "incomplete.zip"
    writer = delivery.DeliveryZip(path)
    writer.add_json("DELIVERY_COVERAGE.json", dict(verified_tasks=33))
    writer.add_bytes("START_HERE.md", b"Incomplete fixture")
    writer.add_bytes("REPORT_ZH.md", b"Incomplete fixture")
    writer.finish()
    with pytest.raises(PermissionError, match="required original/sidecar"):
        delivery.verify_zip(path)


def test_claimed_aggregate_count_cannot_replace_actual_original_records():
    row = dict(
        source_path="runs/x/rollouts/01-00.json",
        source_bytes=2,
        source_sha256=delivery.hashlib.sha256(b"{}").hexdigest(),
        records=98304,
        original_utf8="{}",
        task_sidecar={"qid": "q"},
    )
    with pytest.raises(PermissionError, match="row count/hash"):
        delivery._validate_delivered_original(row, "TRAINING_GROUPS.jsonl.gz", {"q": {"qid": "q"}})


def test_authenticated_aggregate_cannot_change_between_validation_and_stream(tmp_path):
    path = tmp_path / "runs/group.json"
    write(path, dict(records=[{"raw_text": "original"}]))
    receipt = dict(
        artifacts=[
            dict(path="runs/group.json", sha256=delivery.sha_file(path), bytes=path.stat().st_size)
        ]
    )
    collection = delivery.freeze_collections(tmp_path, {"fixture": [(path, {}, 1)]}, {"t": receipt})
    write(path, dict(records=[{"raw_text": "tampered"}]))
    with pytest.raises(PermissionError, match="authenticated source"):
        list(delivery.wrapped_originals(tmp_path, collection["fixture"]))


def test_existing_zip_does_not_bypass_actual_full_raw_rescoring(tmp_path, monkeypatch):
    output = tmp_path / "delivery/existing.zip"
    output.parent.mkdir()
    output.write_bytes(b"A pre-existing filename is not an experimental delivery")
    write(tmp_path / "RUN_IDENTITY.json", dict(code_commit="CPU_FIXTURE_ONLY"))
    write(tmp_path / "deployment/SOURCE_MANIFEST.json", {})
    write(tmp_path / "EXECUTION_FREEZE.json", {})
    write(tmp_path / "tasks/release/COMPLETE.json", {})
    api = SimpleNamespace(
        runner=SimpleNamespace(
            dataset_and_store=lambda _: (SimpleNamespace(tasks={}, runs=[]), None)
        ),
        endpoint_runner=SimpleNamespace(
            _external=lambda _: ({}, {}, {}, None), _raw_groups=lambda *_: iter(())
        ),
        evaluation=SimpleNamespace(endpoint_slots=lambda *_: []),
    )
    monkeypatch.setattr(delivery, "load_frozen_api", lambda _: api)
    monkeypatch.setattr(delivery, "require_final_release", lambda *_: {"release": {}})
    monkeypatch.setattr(delivery, "verify_images_and_types", lambda *_: [])

    def refuse(*_args, **_kwargs):
        raise RuntimeError("actual original rescore sentinel")

    monkeypatch.setattr(delivery, "verify_release_stream", refuse)
    with pytest.raises(RuntimeError, match="original rescore sentinel"):
        delivery.build(tmp_path, output)
    assert not Path(str(output) + ".verification.json").exists()
    failures = list(output.parent.glob("existing.zip.failure-*.json"))
    assert len(failures) == 1
    failure = delivery.read(failures[0])
    assert failure["published"] is True
    assert failure["retained_temporary"] == []
    assert failure["published_archive_sha256"] == delivery.sha_file(output)


def test_preparation_history_is_retained_outside_new_science_task_count(tmp_path):
    for directory in (
        "native_job",
        "native_job_v2",
        "preflight_job",
        "prior_job",
        "type_audit_job",
        "parent_audit_code",
        "native_audit_code_v2",
    ):
        file = tmp_path / directory / "failure.err"
        file.parent.mkdir()
        file.write_text("CPU fixture historical failure retained")
    included, _ = delivery.original_inventory(tmp_path, {}, set(), {})
    assert len(included) == 7
    assert "evidence/native_job/failure.err" in included


def test_archive_binding_compares_actual_sidecars_and_helper_bytes(tmp_path):
    # Binding unit fixture only: complete scientific acceptance is tested by its
    # separate source/release gates, never inferred from these placeholder rows.
    path = tmp_path / "binding.zip"
    helper = tmp_path / "helper.py"
    helper.write_bytes(b"# retained helper fixture\n")
    expected = dict(sha256=delivery.sha_file(helper), bytes=helper.stat().st_size)
    writer = delivery.DeliveryZip(path)
    for name in sorted(delivery.DELIVERY_MEMBERS):
        if name == "TASK_SIDECARS.jsonl.gz":
            writer.add_jsonl_gzip(name, [{"qid": "actual-q"}])
        elif name.endswith(".jsonl.gz"):
            writer.add_jsonl_gzip(name, [])
        elif name == "OPERATIONS_SNAPSHOT.json":
            writer.add_json(
                name, dict(scope="immutable files present before delivery publication", included=[])
            )
        elif name.endswith(".json"):
            writer.add_json(name, {})
        else:
            writer.add_bytes(name, b"CPU fixture only")
    writer.add_path("helper/source/helper.py", helper, expected)
    writer.finish()
    arguments = dict(
        json_values={"DELIVERY_COVERAGE.json": {}},
        jsonl_values={"TASK_SIDECARS.jsonl.gz": lambda: iter([{"qid": "actual-q"}])},
        sources={"helper/source/helper.py": (helper, expected)},
        originals={},
    )
    delivery.verify_bound_archive(path, **arguments)
    with pytest.raises(PermissionError, match="raw/sidecar"):
        delivery.verify_bound_archive(
            path,
            **dict(
                arguments,
                jsonl_values={"TASK_SIDECARS.jsonl.gz": lambda: iter([{"qid": "different-q"}])},
            ),
        )
    helper.write_bytes(b"# changed between validation and streaming\n")
    with pytest.raises(PermissionError, match="Original artifact changed"):
        delivery.verify_bound_archive(path, **arguments)


def test_archived_cost_snapshot_recomputes_original_accounting_and_rejects_changed_totals(tmp_path):
    stdout = scheduler_fixture(tmp_path)
    scheduler, raw = delivery.scheduler_costs(
        tmp_path, execute=lambda *a, **k: SimpleNamespace(returncode=0, stdout=stdout, stderr="")
    )
    costs = dict(
        scheduler=scheduler,
        **{k: scheduler["totals"][k] for k in ("gpu_hours", "queue_seconds", "total_wall_seconds")},
        physical_counts={"fixture": 2},
    )
    delivery.verify_cost_snapshot(tmp_path, costs, raw, costs)
    changed = copy.deepcopy(costs)
    changed["gpu_hours"] = 0
    with pytest.raises(PermissionError, match="scheduler totals"):
        delivery.verify_cost_snapshot(tmp_path, changed, raw, costs)
    forged_raw = dict(raw, stdout=raw["stdout"].replace("|3600|", "|36|"))
    forged_scheduler, _ = delivery.scheduler_costs(
        tmp_path,
        execute=lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=forged_raw["stdout"], stderr=""
        ),
    )
    forged = dict(
        costs,
        scheduler=forged_scheduler,
        **{
            k: forged_scheduler["totals"][k]
            for k in ("gpu_hours", "queue_seconds", "total_wall_seconds")
        },
    )
    with pytest.raises(PermissionError, match="current actual"):
        delivery.verify_cost_snapshot(tmp_path, forged, forged_raw, costs)


def test_terminal_operation_failure_log_is_kept_while_active_log_is_outside_snapshot(tmp_path):
    old, active = tmp_path / "operations/attempts/0001", tmp_path / "operations/attempts/0002"
    write(old / "TERMINAL.json", dict(state="TIMEOUT"))
    active.mkdir(parents=True)
    (old / "slurm-1.log").write_text("historical traceback fixture")
    (active / "slurm-2.log").write_text("active log fixture")
    included, _ = delivery.original_inventory(tmp_path, {}, set(), {})
    assert "evidence/operations/attempts/0001/slurm-1.log" in included
    assert "evidence/operations/attempts/0002/slurm-2.log" not in included


def test_actual_production_scorer_reopens_three_protocols_and_keeps_invalid_answer(tmp_path):
    from sr_f3_ce import contract, evaluation

    world = dict(
        categories=["January", "February", "March", "April"],
        domain=[10, 90],
        series={"Alpha": [30, 27, 50, 33], "Beta": [11, 53, 71, 62]},
    )
    query = dict(family="TOPK", aggregate="sum", k=2, direction="largest")
    task = dict(qid="CPU_ONLY_Q0", pool="TEST_ID", world=world, query=query)
    texts = [
        ("structured", contract.support_target(world, query, False)["text"], "balanced"),
        ("structured", "{", "eos"),
        ("answer_only", '{"answer":133}', "balanced"),
        ("natural_work", 'work\n<final>"133"</final>', "final_tag"),
    ]
    directory = tmp_path / "release"
    rows, compact, groups, raw_hashes, group_hashes = [], [], [], {}, {}
    for index, (protocol, text, finish) in enumerate(texts):
        gid = f"CPU_ONLY_G{index}"
        slot = dict(
            slot_id=f"CPU_ONLY_S{index}", qid=task["qid"], pool="TEST_ID", protocol=protocol, draw=0
        )
        row = dict(
            slot,
            group_id=gid,
            generation_status="COMPLETE",
            technical_validation_errors=[],
            tokens=[1, 2],
            max_new_tokens=1024,
            model_input={"prompt": "CPU synthetic fixture only"},
            raw_text=text,
            finish_reason=finish,
        )
        score = evaluation.score_record(row, task)
        scores = [dict(slot_id=slot["slot_id"], score=score)]
        rows.append(row)
        compact.append(
            dict(
                slot,
                group_id=gid,
                generation_status="COMPLETE",
                technical_validation_errors=[],
                score=score,
            )
        )
        groups.append((dict(group_id=gid, slots=[slot]), [row], dict(sha256=f"fixture-{index}")))
        raw_hashes[gid], group_hashes[gid] = delivery.digest([row]), delivery.digest(scores)
        write(
            directory / f"scores/{gid}.json",
            dict(raw_sha256=f"fixture-{index}", scores=scores, scorer_version=delivery.EXPERIMENT),
        )
    with gzip.open(directory / "ENDPOINT_RAW.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    (directory / "ENDPOINT_SCORES.jsonl").write_bytes(
        b"".join(delivery.encoded(row) + b"\n" for row in compact)
    )
    receipt = dict(
        status="RELEASED", experiment_id=delivery.EXPERIMENT, records=4, raw_groups=raw_hashes
    )
    write(directory / "RELEASE.json", receipt)
    write(
        directory / "RAW_RESCORING.json",
        dict(
            status="PASS",
            records=4,
            mismatches=[],
            gzip_read_to_eof=True,
            raw_sha256=delivery.sha_file(directory / "ENDPOINT_RAW.jsonl.gz"),
            group_scores_sha256=delivery.digest(group_hashes),
        ),
    )
    write(
        directory / "RELEASE_SCORED.json",
        dict(
            receipt,
            raw_rescoring="PASS",
            projection_kind="verified_raw_rescore_projection_v1",
            analysis_rows_sha256=delivery.digest(compact),
            rescoring_receipt_sha256=delivery.sha_file(directory / "RAW_RESCORING.json"),
        ),
    )
    for name in ("PRIMARY_ANALYSIS.json", "EXTERNAL_ANALYSIS.json", "PREFIX_DIAGNOSTICS.json"):
        write(directory / name, {"scope": "CPU_FIXTURE_ONLY"})
    refresh_manifest(tmp_path)
    result = delivery.verify_release_stream(
        tmp_path, iter(groups), {task["qid"]: task}, evaluation.score_record
    )
    assert [row["score"]["value_correct"] for row in compact] == [1, 0, 1, 1]
    assert compact[1]["score"]["finish"] == 0
    assert compact[3]["score"]["value_parseable"] == 1
    assert result["records"] == 4 and result["full_original_rescoring"] == "PASS"


def test_costs_keep_probe_attempts_and_forward_counters_separate_without_double_count(tmp_path):
    row = dict(
        kind="extra_forward_sequences",
        count=2,
        task_id="fixture",
        metadata=dict(purpose="engine_probe", completion_tokens=8),
    )
    accounting = tmp_path / "accounting/fixture.jsonl"
    write(accounting, row)
    cost = dict(update_backward_calls_attempted=32, optimizer_step_calls_attempted=0)
    write(tmp_path / "engine/resume/probes/EXEC-mb4.json", dict(total_cost=cost))
    write(tmp_path / "engine/resume/probes/EXEC-mb2.FAILURE.json", dict(cost=cost))
    write(tmp_path / "engine/resume/probes/COMPLETE.json", dict(probe_costs={"4": cost}))
    costs, _ = delivery.cost_summary(tmp_path)
    assert len(costs["engine_probe_physical_attempts"]) == 2
    assert {row["status"] for row in costs["engine_probe_physical_attempts"]} == {
        "FAILED",
        "MEASURED",
    }
    assert costs["recorded_extra_forward_by_purpose"] == {
        "engine_probe": {"sequences": 2, "completion_tokens": 8}
    }
    assert costs["decode_time"] == "UNKNOWN"
    assert costs["image_encoder_call_count"].startswith("UNKNOWN")
    assert "not summed" in costs["generation_seconds_limitation"]


def test_training_signal_index_retains_repeated_attempt_and_requires_fixed_gradient_diagnostics(
    tmp_path,
):
    run = dict(run_id="CPU_FIXTURE_ONLY", arm="RL")
    numbers = (
        "policy",
        "kl",
        "aux_ce",
        "policy_weighted",
        "kl_weighted",
        "aux_ce_weighted",
        "gradient_norm_before_clip",
        "gradient_norm_after_clip",
        "parameter_max_absolute_change",
        "parameter_l2_change",
        "constant_group_fraction",
        "sampler_train_mean_absolute_difference",
        "sampler_train_p99_absolute_difference",
        "sampler_train_max_absolute_difference",
        "tis_truncated_fraction",
        "tis_cap",
    )
    metrics = dict.fromkeys(numbers, 0.0)
    metrics.update(
        arm="RL",
        finite=True,
        engineering_zero_aux=False,
        scientific_recipe_eligible=True,
        optimizer_updates=1,
        optimizer_steps=[1],
        microbatch_size=4,
        reward_groups=[{} for _ in range(16)],
        support_losses=[],
        gradient_diagnostics={"unweighted_norms": {"policy": 1, "kl": 0}},
        policy=-1,
    )
    directory = tmp_path / "runs/CPU_FIXTURE_ONLY/update_attempts"
    for step in range(1, 65):
        write(
            directory / f"{step:02d}-attempt.json",
            dict(step=step, metrics=dict(metrics, step=step)),
        )
    write(directory / "01-repeat.json", dict(step=1, metrics=dict(metrics, step=1)))
    index = delivery.training_signal_index(tmp_path, [run])["runs"][run["run_id"]]
    assert len(index["logical_steps"]) == 64 and len(index["attempts"]) == 65
    assert index["diagnostic_steps"] == [1, 16, 32, 64]
    assert index["attempts"][0]["fixed_diagnostic"]["policy"] == -1
    write(
        directory / "16-attempt.json",
        dict(step=16, metrics=dict(metrics, step=16, gradient_diagnostics=None)),
    )
    with pytest.raises(PermissionError, match="gradient diagnostic"):
        delivery.training_signal_index(tmp_path, [run])


def extension_fixture(root, monkeypatch):
    write(root / "RUN_IDENTITY.json", dict(experiment_id=delivery.EXPERIMENT, code_commit="7" * 40))
    write(root / "deployment/SOURCE_MANIFEST.json", dict(files=[]))
    directory = root / "native/observability" / ("a" * 40)
    directory.mkdir(parents=True)
    module_name = "fixture_orchestration_" + delivery.digest(str(root))[:12]
    orchestration = ModuleType(module_name)
    orchestration.Controller, orchestration.calls = type("OriginalController", (), {}), []
    monkeypatch.setitem(delivery.sys.modules, module_name, orchestration)
    launcher = (
        f"import {module_name} as original\n"
        "original.calls.append('IMPORT')\n"
        "class InstrumentedController: pass\n"
        "def verify_extension(root, manifest_path, science=False):\n"
        "    original.calls.append(('VERIFY', science))\n"
        "    return {'scope': 'CPU_FIXTURE_ONLY'}\n"
        "def install(root, manifest_path):\n"
        "    original.calls.append('INSTALL')\n"
        "    original.Controller = InstrumentedController\n"
        "    return InstrumentedController\n"
    )
    (directory / "instrumented_execution.py").write_text(launcher)
    (directory / "observe_runtime.py").write_text("# CPU fixture observer, no model calls\n")
    write(
        directory / "CPU_VALIDATION.json",
        dict(
            status="PASS",
            scope="CPU_TOY_TRANSPARENCY_AND_ORCHESTRATION",
            tests_passed=2,
            gpu_validation_performed=False,
            new_model_generations=0,
            new_optimizer_steps=0,
            source_sha256={
                name: delivery.sha_file(directory / name)
                for name in ("instrumented_execution.py", "observe_runtime.py")
            },
        ),
    )
    write(directory / "AMENDMENT.json", dict(scope="CPU_FIXTURE_ONLY"))

    def record(name):
        return dict(
            path=name,
            sha256=delivery.sha_file(directory / name),
            bytes=(directory / name).stat().st_size,
        )

    manifest = dict(
        schema_version=1,
        experiment_id=delivery.EXPERIMENT,
        status="FROZEN_OBSERVABILITY_EXTENSION",
        extension_commit="a" * 40,
        base_code_commit="7" * 40,
        run_identity_sha256=delivery.sha_file(root / "RUN_IDENTITY.json"),
        base_source_manifest_sha256=delivery.sha_file(root / "deployment/SOURCE_MANIFEST.json"),
        sources=[record(name) for name in ("instrumented_execution.py", "observe_runtime.py")],
        cpu_validation=record("CPU_VALIDATION.json"),
    )
    write(directory / "SOURCE_MANIFEST.json", manifest)
    write(
        root / "EXECUTION_FREEZE.json",
        dict(
            engineering_files={
                str((directory / name).relative_to(root)): delivery.sha_file(directory / name)
                for name in ("SOURCE_MANIFEST.json", "CPU_VALIDATION.json", "AMENDMENT.json")
            }
        ),
    )
    api = SimpleNamespace(
        orchestration=orchestration, protocol=SimpleNamespace(verify_run=lambda *a, **k: None)
    )
    return directory / "SOURCE_MANIFEST.json", api


@pytest.mark.parametrize("suffix", [".pyc", ".pyo"])
def test_frozen_package_bytecode_is_rejected_before_any_scientific_import(
    tmp_path, monkeypatch, suffix
):
    path = tmp_path / "code/src/sr_f3_ce/protocol.py"
    path.parent.mkdir(parents=True)
    original = b"# authenticated CPU fixture source only\n"
    path.write_bytes(original)
    write(tmp_path / "RUN_IDENTITY.json", dict(experiment_id=delivery.EXPERIMENT))
    write(
        tmp_path / "deployment/SOURCE_MANIFEST.json",
        dict(
            files=[
                dict(
                    path="src/sr_f3_ce/protocol.py",
                    bytes=len(original),
                    sha256=delivery.sha_file(path),
                )
            ]
        ),
    )
    path.write_text("raise RuntimeError('UNBOUND_PACKAGE_CACHE_EXECUTED')\n")
    cache = (
        Path(importlib.util.cache_from_source(str(path)))
        if suffix == ".pyc"
        else path.with_suffix(suffix)
    )
    py_compile.compile(
        str(path),
        cfile=str(cache),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    path.write_bytes(original)
    cache_hash = delivery.sha_file(cache)
    monkeypatch.setattr(delivery.sys, "dont_write_bytecode", False)

    def forbidden_import(*args, **kwargs):
        pytest.fail("Scientific import ran before the frozen bytecode guard")

    monkeypatch.setattr(delivery.importlib, "import_module", forbidden_import)
    with pytest.raises(PermissionError, match="unbound Python bytecode"):
        delivery.load_frozen_api(tmp_path)
    assert delivery.sys.dont_write_bytecode is True
    assert delivery.sha_file(cache) == cache_hash
    assert path.read_bytes() == original


def test_extension_import_uses_unique_freeze_binding_then_installs_real_controller(
    tmp_path, monkeypatch
):
    manifest, api = extension_fixture(tmp_path, monkeypatch)
    extension = delivery.load_observability_extension(tmp_path, api, manifest)
    assert api.orchestration.calls == ["IMPORT", ("VERIFY", True), "INSTALL"]
    assert api.orchestration.Controller is extension.controller
    assert extension.identity["manifest_sha256"] == delivery.sha_file(manifest)
    api.observability = extension
    api.orchestration.Controller = type("ReplacedController", (), {})
    with pytest.raises(PermissionError, match="replaced"):
        delivery.require_final_release(tmp_path, api)


def test_extension_ignores_unchecked_hash_pyc_and_preserves_module_identity(tmp_path, monkeypatch):
    manifest, api = extension_fixture(tmp_path, monkeypatch)
    launcher = manifest.with_name("instrumented_execution.py")
    original = launcher.read_bytes()
    launcher.write_text("raise RuntimeError('UNBOUND_PYC_EXECUTED')\n")
    cache = Path(
        py_compile.compile(
            str(launcher),
            invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
            doraise=True,
        )
    )
    launcher.write_bytes(original)
    assert struct.unpack("<I", cache.read_bytes()[4:8])[0] == 1
    monkeypatch.setattr(delivery.sys, "dont_write_bytecode", True)
    # Confirm this fixture is a real unchecked cache: the standard loader would use it.
    spec = importlib.util.spec_from_file_location("unsafe_cpu_fixture_only", launcher)
    with pytest.raises(RuntimeError, match="UNBOUND_PYC_EXECUTED"):
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
    extension = delivery.load_observability_extension(tmp_path, api, manifest)
    assert api.orchestration.calls == ["IMPORT", ("VERIFY", True), "INSTALL"]
    assert extension.module.__file__ == str(launcher)
    assert extension.module.__spec__.origin == str(launcher)
    assert delivery.sys.modules[extension.module.__name__] is extension.module


@pytest.mark.parametrize("replacement_point", ["before_buffer_read", "after_buffer_read"])
def test_extension_executes_only_the_same_buffer_it_hashes(
    tmp_path, monkeypatch, replacement_point
):
    manifest, api = extension_fixture(tmp_path, monkeypatch)
    launcher = manifest.with_name("instrumented_execution.py")
    original_read, reads = Path.read_bytes, []
    replacement = b"raise RuntimeError('REPLACED_SOURCE_EXECUTED')\n"

    def racing_read(path):
        if path != launcher:
            return original_read(path)
        reads.append(path)
        if replacement_point == "before_buffer_read":
            path.write_bytes(replacement)
        value = original_read(path)
        if replacement_point == "after_buffer_read":
            path.write_bytes(replacement)
        return value

    monkeypatch.setattr(Path, "read_bytes", racing_read)
    if replacement_point == "before_buffer_read":
        with pytest.raises(PermissionError, match="launcher changed before import"):
            delivery.load_observability_extension(tmp_path, api, manifest)
        assert api.orchestration.calls == []
    else:
        # The fixture verifier records calls only; real verify_extension additionally
        # rejects the changed on-disk source after this authenticated buffer executes.
        extension = delivery.load_observability_extension(tmp_path, api, manifest)
        assert api.orchestration.calls == ["IMPORT", ("VERIFY", True), "INSTALL"]
        assert api.orchestration.Controller is extension.controller
    assert reads == [launcher]
    assert original_read(launcher) == replacement


@pytest.mark.parametrize("mutation", ["unbound", "observer_bytes", "ambiguous", "explicit_other"])
def test_unbound_ambiguous_or_changed_extension_is_rejected_before_python_import(
    tmp_path, monkeypatch, mutation
):
    manifest, api = extension_fixture(tmp_path, monkeypatch)
    freeze = delivery.read(tmp_path / "EXECUTION_FREEZE.json")
    supplied = None
    if mutation == "unbound":
        del freeze["engineering_files"][
            str(manifest.with_name("AMENDMENT.json").relative_to(tmp_path))
        ]
        write(tmp_path / "EXECUTION_FREEZE.json", freeze)
    elif mutation == "observer_bytes":
        manifest.with_name("observe_runtime.py").write_text("# changed source\n")
    elif mutation == "ambiguous":
        freeze["engineering_files"][
            "native/observability/" + "b" * 40 + "/SOURCE_MANIFEST.json"
        ] = "x"
        write(tmp_path / "EXECUTION_FREEZE.json", freeze)
    else:
        supplied = tmp_path / "arbitrary.json"
        write(supplied, {})
    with pytest.raises(PermissionError):
        delivery.load_observability_extension(tmp_path, api, supplied)
    assert api.orchestration.calls == []


def observation_event(scope, event, *, parent=None, kind="generation", outcome="RETURNED"):
    row = dict(
        schema_version=1,
        experiment_id=delivery.EXPERIMENT,
        installation_id="obs-1",
        runtime_id="r1",
        scope_id=scope,
        parent_scope_id=parent,
        kind=kind,
        purpose="fixture",
        protocol="structured",
        pid=42,
        time_unix_ns=100 if event == "SCOPE_BEGIN" else 200,
        event=event,
    )
    if event == "SCOPE_END":
        counts = dict(
            entered=1,
            returned=int(outcome == "RETURNED"),
            raised=0,
            unresolved=0,
            checkpoint_early_stop=int(outcome == "CHECKPOINT_EARLY_STOP"),
            input_batch_items=1,
            input_sequence_slots=2,
            input_element_slots=2,
            image_items=0,
            image_shape_unknown_calls=0,
            shape_unknown_calls=0,
            host_inclusive_ns=50,
            host_exclusive_ns=40,
        )
        row.update(
            outcome=outcome,
            original_call_returned=outcome == "RETURNED",
            exception_type=None,
            host_scope_elapsed_ns=100,
            coverage_status="COMPLETE",
            coverage_issues=[],
            visual_expected=True,
            model_stages={"prefill": counts},
            visual_stages={"prefill": dict(counts, image_items=1, image_shape_unknown_calls=0)},
            model_hook_layers={"ignored_nested_base": dict(entered=99)},
        )
    return row


def write_events(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(delivery.encoded(row) + b"\n" for row in rows))


def test_nested_observation_hooks_count_once_and_scope_host_times_are_not_added(tmp_path):
    path = tmp_path / "events.jsonl"
    write_events(
        path,
        [
            observation_event("outer", "SCOPE_BEGIN"),
            observation_event("inner", "SCOPE_BEGIN", parent="outer", kind="teacher_forcing"),
            observation_event("inner", "SCOPE_END", parent="outer", kind="teacher_forcing"),
            observation_event("outer", "SCOPE_END"),
        ],
    )
    summary = delivery.summarize_observation_events(path)
    assert summary["status"] == "COMPLETE_RECORDED_SCOPES"
    categories = summary["by_kind_purpose_protocol"]
    assert sum(row["model_stages"]["prefill"]["entered"] for row in categories) == 2
    assert sum(row["visual_stages"]["prefill"]["entered"] for row in categories) == 2
    assert all(
        row["scope_host_elapsed_ns"] == {"minimum": 100, "maximum": 100} for row in categories
    )
    assert "not GPU kernel" in summary["timing_scope"]


def test_checkpoint_early_stop_is_recorded_control_flow_not_false_scientific_failure(tmp_path):
    path = tmp_path / "events.jsonl"
    write_events(
        path,
        [
            observation_event("s", "SCOPE_BEGIN", kind="unscoped_model_forward"),
            observation_event(
                "s", "SCOPE_END", kind="unscoped_model_forward", outcome="CHECKPOINT_EARLY_STOP"
            ),
        ],
    )
    summary = delivery.summarize_observation_events(path)
    assert summary["status"] == "COMPLETE_RECORDED_SCOPES"
    assert summary["by_kind_purpose_protocol"][0]["outcomes"] == {"CHECKPOINT_EARLY_STOP": 1}


def test_unclosed_and_truncated_observation_scope_remains_unknown(tmp_path):
    path = tmp_path / "events.jsonl"
    write_events(path, [observation_event("s", "SCOPE_BEGIN")])
    with path.open("ab") as stream:
        stream.write(b'{"event":"SCOPE_END"')
    summary = delivery.summarize_observation_events(path)
    assert summary["status"] == "UNKNOWN" and summary["ended_scopes"] == 0
    assert set(summary["coverage_issues"]) == {"TRUNCATED_FINAL_EVENT", "SCOPES_WITHOUT_END"}


@pytest.mark.parametrize("problem", ["unmatched_end", "wrong_counter", "changed_identity"])
def test_corrupted_observation_pair_or_counters_is_rejected(tmp_path, problem):
    rows = [observation_event("s", "SCOPE_BEGIN"), observation_event("s", "SCOPE_END")]
    if problem == "unmatched_end":
        rows.pop(0)
    elif problem == "wrong_counter":
        rows[1]["model_stages"]["prefill"]["entered"] = 2
    else:
        rows[1]["runtime_id"] = "different-runtime"
    path = tmp_path / "events.jsonl"
    write_events(path, rows)
    with pytest.raises(PermissionError):
        delivery.summarize_observation_events(path)


def observation_attempt_fixture(root):
    identity = dict(
        manifest_sha256="a" * 64, amendment_sha256="b" * 64, observer_source_sha256="c" * 64
    )
    write(root / "RUN_IDENTITY.json", {})
    write(root / "EXECUTION_FREEZE.json", {})
    job_key, task_id = "physical-attempt-1", "train--CPU_FIXTURE_ONLY"
    job = root / "orchestration/jobs" / job_key
    write(
        job / "INTENT.json",
        dict(
            key=job_key,
            task_ids=[task_id],
            claim_id="claim-1",
            extension_manifest_sha256=identity["manifest_sha256"],
            amendment_sha256=identity["amendment_sha256"],
        ),
    )
    write(
        job / "STATE.json", dict(job_id="123", intent_sha256=delivery.sha_file(job / "INTENT.json"))
    )
    binding = dict(
        experiment_id=delivery.EXPERIMENT,
        run_identity_sha256=delivery.sha_file(root / "RUN_IDENTITY.json"),
        execution_freeze_sha256=delivery.sha_file(root / "EXECUTION_FREEZE.json"),
        extension_manifest_sha256=identity["manifest_sha256"],
        amendment_sha256=identity["amendment_sha256"],
        job_key=job_key,
        task_id=task_id,
        job_id="123",
        claim_id="claim-1",
    )
    attempt = root / "accounting/observability" / job_key / task_id
    write(
        attempt / "INSTALLATION.json",
        dict(
            schema_version=1,
            status="INSTALLED",
            execution_binding=binding,
            observer=dict(
                installation_id="obs-1",
                old_engine_observed=False,
                observer_source_sha256=identity["observer_source_sha256"],
            ),
        ),
    )
    write_events(
        attempt / "events.jsonl",
        [
            dict(observation_event("s", event), execution_binding=binding)
            for event in ("SCOPE_BEGIN", "SCOPE_END")
        ],
    )
    write(root / "tasks" / task_id / "COMPLETE.json", {"scope": "CPU_FIXTURE_ONLY"})
    write(
        attempt / "FINAL.json",
        dict(
            schema_version=1,
            status="COMPLETE",
            execution_binding=binding,
            installation_sha256=delivery.sha_file(attempt / "INSTALLATION.json"),
            events=dict(
                path=str((attempt / "events.jsonl").relative_to(root)),
                sha256=delivery.sha_file(attempt / "events.jsonl"),
                bytes=(attempt / "events.jsonl").stat().st_size,
                rows=2,
                scope_begin=1,
                scope_end=1,
            ),
            observer_snapshot=dict(
                schema_version=1,
                installation_id="obs-1",
                sealed=True,
                events_written={"SCOPE_BEGIN": 1, "SCOPE_END": 1},
                observer_overhead_ns={"emit": 1},
                active_scopes=[],
                write_errors=[],
                final_receipt_write_cost="UNKNOWN_NOT_MEASURED_BY_OBSERVER",
            ),
            task_receipt_sha256=delivery.sha_file(root / "tasks" / task_id / "COMPLETE.json"),
        ),
    )
    return identity, task_id, attempt


def test_final_observation_receipt_binds_actual_job_events_and_preserves_raw(tmp_path):
    identity, task_id, attempt = observation_attempt_fixture(tmp_path)
    summary = delivery.observability_costs(tmp_path, identity, [task_id])
    assert summary["status"] == "COMPLETE_RECORDED_ATTEMPTS"
    assert summary["old_engine"].startswith("UNKNOWN")
    assert summary["attempts"][0]["observer_overhead_ns"] == {"emit": 1}
    originals, _ = delivery.original_inventory(tmp_path, {}, set(), {})
    for name in ("events.jsonl", "INSTALLATION.json", "FINAL.json"):
        assert "evidence/" + str((attempt / name).relative_to(tmp_path)) in originals
    final = delivery.read(attempt / "FINAL.json")
    final["events"]["rows"] = 3
    write(attempt / "FINAL.json", final)
    with pytest.raises(PermissionError, match="Final observation receipt"):
        delivery.observability_costs(tmp_path, identity, [task_id])


def test_missing_observation_final_is_unknown_and_cannot_certify_required_task(tmp_path):
    identity, task_id, attempt = observation_attempt_fixture(tmp_path)
    (attempt / "FINAL.json").unlink()
    summary = delivery.observability_costs(tmp_path, identity)
    assert summary["status"] == "UNKNOWN"
    assert summary["attempts"][0]["missing_or_incomplete_final"] is True
    with pytest.raises(PermissionError, match="lack their verified observation"):
        delivery.observability_costs(tmp_path, identity, [task_id])


def test_unsealed_observer_snapshot_cannot_certify_required_task(tmp_path):
    identity, task_id, attempt = observation_attempt_fixture(tmp_path)
    final = delivery.read(attempt / "FINAL.json")
    final["observer_snapshot"]["sealed"] = False
    write(attempt / "FINAL.json", final)
    with pytest.raises(PermissionError, match="Final observation receipt"):
        delivery.observability_costs(tmp_path, identity, [task_id])


def test_complete_scope_cannot_replace_absent_hook_measurements_with_zero(tmp_path):
    rows = [observation_event("s", "SCOPE_BEGIN"), observation_event("s", "SCOPE_END")]
    rows[1]["model_stages"] = {}
    path = tmp_path / "events.jsonl"
    write_events(path, rows)
    with pytest.raises(PermissionError, match="missing model/visual hooks"):
        delivery.summarize_observation_events(path)


def test_actual_cpu_observer_event_schema_integrates_with_delivery_summary(tmp_path):
    import torch

    source = Path(__file__).resolve().parents[2] / "scripts/sr_f3_ce/observe_runtime.py"
    spec = importlib.util.spec_from_file_location("delivery_actual_observer_fixture", source)
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding, self.visual = torch.nn.Embedding(8, 2), torch.nn.Identity()

        def forward(self, input_ids, pixel_values=None):
            return self.embedding(input_ids) + self.visual(pixel_values)[:, None, :]

    class TinyRuntime:
        def __init__(self):
            self.model = TinyModel()

        def _visual_module(self):
            return self.model.visual

        def generate_group(self, values, protocol="structured"):
            self.model(**values)
            return self.model(**dict(values, input_ids=values["input_ids"][:, :1]))

        def _native_forward(self, values, purpose="training_policy", grad=True):
            return self.model(**values)

    events = []
    observer.install(TinyRuntime, events.append)
    try:
        runtime = TinyRuntime()
        inputs = dict(input_ids=torch.tensor([[1, 2, 3]]), pixel_values=torch.ones(1, 2))
        runtime.generate_group(inputs)
        runtime._native_forward(inputs).sum().backward()
        assert observer.snapshot(TinyRuntime)["active_scopes"] == []
        assert observer.seal(TinyRuntime)["sealed"] is True
    finally:
        observer.uninstall(TinyRuntime)
    path = tmp_path / "events.jsonl"
    write_events(path, events)
    summary = delivery.summarize_observation_events(path)
    assert summary["status"] == "COMPLETE_RECORDED_SCOPES"
    categories = summary["by_kind_purpose_protocol"]
    for axis in ("model_stages", "visual_stages"):
        assert sum(stats["entered"] for row in categories for stats in row[axis].values()) == 3
