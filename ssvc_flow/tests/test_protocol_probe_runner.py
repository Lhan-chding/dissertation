"""Deterministic CPU contracts for orchestration, never GPU evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.protocol_state_probes import protocol as p
from src.protocol_state_probes import run as r
from src.protocol_state_probes.cli import parser

DESIGN = Path(__file__).resolve().parents[1] / "docs/protocol_state_probes/design"


class FakeBackend:
    def __init__(self, failures=0, text="not valid JSON"):
        self.calls = []
        self.failures = failures
        self.text = text
        self.counters = {"backward_calls": 0, "optimizer_updates": 0}
        self.receipt = {"execution_kind": "CPU_TEST_DOUBLE_NOT_MODEL_EVIDENCE"}

    def generate_public(self, prompt, *, seed, max_new_tokens):
        assert set(prompt) == {"system", "user"}
        assert max_new_tokens == 64
        self.calls.append((prompt, seed))
        if self.failures:
            self.failures -= 1
            raise OSError("simulated generation transport failure")
        return {
            "raw_text": self.text,
            "token_ids": [1, 2],
            "stop_reason": "eos",
            "elapsed_seconds": 0.01,
            "generated_length": 2,
        }


def make_run(tmp_path, *, jobs=None):
    root = tmp_path / "run"
    root.mkdir()
    cases = p.read_jsonl(DESIGN / "manifests/cases.jsonl")
    audits = p.read_jsonl(DESIGN / "manifests/audit_labels_NOT_FOR_MODEL.jsonl")
    predictions = p.read_jsonl(DESIGN / "manifests/program_predictions.jsonl")
    aliases = {case["case_id"]: case["case_id"] for case in cases}
    aliases.update(
        {
            row["case_id"]: row["owner_case_id"]
            for row in p.read_jsonl(DESIGN / "manifests/prompt_aliases.jsonl")
        }
    )
    protocol = p.read_json(DESIGN / "protocol.json")
    if jobs is None:
        jobs = [
            {
                **p.read_jsonl(DESIGN / "manifests/logical_jobs.jsonl")[0],
                "checkpoint_id": "S96",
                "draws": 8,
            }
        ]
    files = {
        "protocol.json": p.json_bytes(protocol),
        "cases.jsonl": b"".join(p.json_bytes(row) for row in cases),
        "audit_labels.jsonl": b"".join(p.json_bytes(row) for row in audits),
        "program_predictions.jsonl": b"".join(p.json_bytes(row) for row in predictions),
        "alias_map.json": p.json_bytes(aliases),
    }
    for name, payload in files.items():
        p.atomic_bytes(root / name, payload)
    runtime = tmp_path / "runtime.json"
    p.atomic_json(runtime, {"test_only": True})
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "jobs": jobs,
        "file_hashes": {
            name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()
        },
        "source": {"code_hashes": p.source_hashes()},
        "runtime": {"path": str(runtime), "sha256": p.file_hash(runtime)},
        "verification": {"all_fields_match": True, "recompiled_from_prepared": True},
    }
    manifest["identity"] = p.digest(manifest)
    p.atomic_json(root / "execution_manifest.json", manifest)
    p.atomic_json(
        root / "CHECKPOINT_AVAILABILITY.json",
        {
            "checkpoints": [
                {"checkpoint_id": name, "status": "AVAILABLE", "checkpoint": {"sha256": name}}
                for name in ("S96", "S32")
            ]
        },
    )
    return root


def test_stable_sample_seed_isolated_by_role_checkpoint_owner():
    baseline = p.sample_identity("experiment", "S96", "owner", "frozen_probe", 0)
    assert baseline == p.sample_identity("experiment", "S96", "owner", "frozen_probe", 0)
    assert (
        len(
            {
                baseline,
                p.sample_identity("experiment", "S32", "owner", "frozen_probe", 0),
                p.sample_identity("experiment", "S96", "other", "frozen_probe", 0),
                p.sample_identity("experiment", "S96", "owner", "smoke", 0),
                p.sample_identity("experiment", "S96", "owner", "frozen_probe", 1),
            }
        )
        == 5
    )


def test_invalid_answers_are_committed_once_never_retried(tmp_path):
    root = make_run(tmp_path)
    backend = FakeBackend()
    result = r.run_worker(
        root, "S96", allow_gpu=True, backend_factory=lambda *args, **kwargs: backend
    )
    rows = r.committed_rows(root)
    assert result["status"] == "COMPLETE"
    assert len(backend.calls) == len(rows) == 8
    assert all(row["features"]["E"] == "I" for row in rows)
    assert all(
        row["role"] == "frozen_probe" and row["raw_text"] == "not valid JSON" for row in rows
    )
    assert (root / "phase_receipts/S96/primary_core_and_controls.json").exists()

    # A completed resume verifies evidence and does not instantiate another model.
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed resume attempted a model load")

    again = r.run_worker(root, "S96", allow_gpu=True, resume=True, backend_factory=forbidden)
    assert again["status"] == "COMPLETE"
    assert len(r.committed_rows(root)) == 8
    with pytest.raises(ValueError, match="--resume"):
        r.run_worker(root, "S96", allow_gpu=True, backend_factory=forbidden)


def test_two_technical_retries_keep_seed_and_errors(tmp_path):
    root = make_run(tmp_path)
    backend = FakeBackend(failures=2)
    result = r.run_worker(
        root, "S96", allow_gpu=True, backend_factory=lambda *args, **kwargs: backend
    )
    assert result["status"] == "COMPLETE"
    assert len(backend.calls) == 10
    assert backend.calls[0][1] == backend.calls[1][1] == backend.calls[2][1]
    assert len(p.read_jsonl(root / "technical_failures.jsonl")) == 2
    assert len(r.committed_rows(root)) == 8


def test_exhausted_technical_attempts_never_reset_on_resume(tmp_path):
    root = make_run(tmp_path)
    backend = FakeBackend(failures=100)
    result = r.run_worker(
        root, "S96", allow_gpu=True, backend_factory=lambda *args, **kwargs: backend
    )
    assert result["status"] == "PARTIAL_WITH_EXPLICIT_BLOCKERS"
    assert len(backend.calls) == 3
    assert r.committed_rows(root) == []
    r.run_worker(
        root, "S96", allow_gpu=True, resume=True, backend_factory=lambda *args, **kwargs: backend
    )
    assert len(backend.calls) == 3
    assert len(p.read_jsonl(root / "technical_failures.jsonl")) == 3


def test_returned_technical_fault_retains_raw_and_nonfinite_values(tmp_path):
    from src.modeling_v3.vlm_observation import ObservationFault

    root = make_run(tmp_path)
    backend = FakeBackend()

    def faulty(prompt, *, seed, max_new_tokens):
        backend.calls.append((prompt, seed))
        raise ObservationFault(
            "Invalid EOS boundary",
            {"raw_completion": "[1,2,3,4]", "token_ids": [1], "bad_score": float("nan")},
        )

    backend.generate_public = faulty
    result = r.run_worker(
        root, "S96", allow_gpu=True, backend_factory=lambda *args, **kwargs: backend
    )
    assert result["status"] == "PARTIAL_WITH_EXPLICIT_BLOCKERS"
    errors = p.read_jsonl(root / "technical_failures.jsonl")
    assert len(errors) == 3
    assert errors[0]["raw_result"]["raw_completion"] == "[1,2,3,4]"
    assert errors[0]["raw_result"]["bad_score"] == {"nonfinite_float": "nan"}
    assert set(errors[0]["prompt"]) == {"system", "user"}


@pytest.mark.parametrize(
    "reason", ["Frozen model changed during generation", "invalid public prompt contract"]
)
def test_fatal_contract_failure_stops_without_retries(tmp_path, reason):
    from src.modeling_v3.vlm_observation import ObservationFault

    root = make_run(tmp_path)
    backend = FakeBackend()

    def faulty(prompt, *, seed, max_new_tokens):
        backend.calls.append((prompt, seed))
        if "Frozen" in reason:
            raise ObservationFault(reason, {"raw_completion": "retained raw output"})
        raise ValueError(reason)

    backend.generate_public = faulty
    with pytest.raises(ValueError, match=reason):
        r.run_worker(root, "S96", allow_gpu=True, backend_factory=lambda *args, **kwargs: backend)
    assert len(backend.calls) == 1
    assert p.read_jsonl(root / "technical_failures.jsonl")[0]["status"] == "FATAL_CONTRACT_FAILURE"
    with pytest.raises(RuntimeError, match="Stored fatal contract failure"):
        r.run_worker(
            root,
            "S96",
            allow_gpu=True,
            resume=True,
            backend_factory=lambda *args, **kwargs: backend,
        )
    assert len(backend.calls) == 1


def test_generated_raw_survives_scoring_crash_without_regeneration(tmp_path):
    root = make_run(tmp_path)
    backend = FakeBackend(text="[1,2,3,4]")

    def broken_scorer(*args):
        raise RuntimeError("scoring process interrupted")

    with pytest.raises(RuntimeError, match="scoring process interrupted"):
        r.run_worker(
            root,
            "S96",
            allow_gpu=True,
            backend_factory=lambda *args, **kwargs: backend,
            scorer=broken_scorer,
        )
    assert len(backend.calls) == 1
    assert r.committed_rows(root) == []
    result = r.run_worker(
        root, "S96", allow_gpu=True, resume=True, backend_factory=lambda *args, **kwargs: backend
    )
    assert result["status"] == "COMPLETE"
    assert len(backend.calls) == 8


def test_uncommitted_chunk_not_counted_hash_tamper_rejected(tmp_path):
    root = make_run(tmp_path)
    backend = FakeBackend()
    r.run_worker(root, "S96", allow_gpu=True, backend_factory=lambda *args, **kwargs: backend)
    commit_path = next((root / "raw").glob("**/*.COMMIT.json"))
    commit = p.read_json(commit_path)
    data = commit_path.parent / commit["file"]
    saved = commit_path.read_bytes()
    commit_path.unlink()
    assert r.committed_rows(root) == []
    p.atomic_bytes(commit_path, saved)
    data.write_text(data.read_text() + " ")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        r.committed_rows(root)


def test_conflicting_same_key_chunk_preserves_both(tmp_path):
    root = make_run(tmp_path)
    backend = FakeBackend()
    r.run_worker(root, "S96", allow_gpu=True, backend_factory=lambda *args, **kwargs: backend)
    commit = next((root / "raw").glob("**/*.COMMIT.json"))
    path = commit.parent / p.read_json(commit)["file"]
    document = p.read_json(path)
    proposed = [{**row, "raw_text": "different"} for row in document["rows"]]
    with pytest.raises(ValueError, match="DUPLICATE_SAMPLE_CONFLICT"):
        r.commit_chunk(path, proposed, document["identity"])
    assert p.read_json(path) == document
    assert len(list(path.parent.glob("*.CONFLICT-*.json"))) == 1


def test_u22_gate_blocks_without_model_and_allows_explicit_downgrade(tmp_path):
    job = next(
        row
        for row in p.read_jsonl(DESIGN / "manifests/logical_jobs.jsonl")
        if row["panel"] == "U22"
    )
    root = make_run(tmp_path, jobs=[{**job, "checkpoint_id": "S96", "draws": 8}])
    backend = FakeBackend()
    calls = []

    def factory(*args, **kwargs):
        calls.append(True)
        return backend

    result = r.run_worker(root, "S96", allow_gpu=True, backend_factory=factory)
    assert not calls
    assert result["jobs"][0]["status"] == "BLOCKED_U22_EXPOSURE_AUDIT"
    p.atomic_json(
        root / "U22_EXPOSURE_AUDIT.json",
        {"verified": True, "downgraded_to_development": True, "status": "CONTAMINATED_DOWNGRADED"},
    )
    result = r.run_worker(root, "S96", allow_gpu=True, resume=True, backend_factory=factory)
    assert result["status"] == "COMPLETE"
    assert len(calls) == 1 and len(backend.calls) == 8
    rows = r.committed_rows(root)
    frozen_case = p.load_run(root)["cases"][job["case_id"]]
    assert frozen_case["original_split"] == "train"
    assert frozen_case["split_role"] == "untouched_outcome_replication"
    assert all(row["original_split"] == "train" for row in rows)
    assert all(row["split_role"] == "untouched_outcome_replication" for row in rows)
    assert all(row["effective_split_role"] == "development_diagnostic" for row in rows)
    assert all(row["exposure_audit_status"] == "CONTAMINATED_DOWNGRADED" for row in rows)
    assert all(row["prompt"] == frozen_case["prompt"] for row in rows)
    assert all(prompt == frozen_case["prompt"] for prompt, _seed in backend.calls)


def test_smoke_and_main_share_single_load_keep_separate_roles(tmp_path):
    root = make_run(tmp_path)
    backend = FakeBackend()
    loads = []

    def factory(*args, **kwargs):
        loads.append(True)
        return backend

    result = r.run_worker(
        root, "S96", allow_gpu=True, with_smoke=True, lane_id="lane-0", backend_factory=factory
    )
    assert result["status"] == "COMPLETE"
    assert len(loads) == 1 and len(backend.calls) == 24
    rows = r.committed_rows(root)
    assert sum(row["role"] == "smoke" for row in rows) == 16
    assert sum(row["role"] == "frozen_probe" for row in rows) == 8
    assert all(row["panel"].startswith("CONTROL_") for row in rows if row["role"] == "smoke")
    skipped = r.run_worker(
        root, "S32", allow_gpu=True, smoke=True, lane_id="lane-0", backend_factory=factory
    )
    assert skipped["new_outputs"] == 0 and len(loads) == 1


def test_first_eight_resource_receipts_use_measured_rows_and_survive_resume(tmp_path):
    root = make_run(tmp_path)
    backend = FakeBackend()
    original_generate = backend.generate_public

    def measured(prompt, *, seed, max_new_tokens):
        raw = original_generate(prompt, seed=seed, max_new_tokens=max_new_tokens)
        count = len(backend.calls)
        return {
            **raw,
            "peak_memory_allocated_bytes": count * 1024,
            "peak_memory_reserved_bytes": count * 2048,
        }

    backend.generate_public = measured
    r.run_worker(
        root,
        "S96",
        allow_gpu=True,
        with_smoke=True,
        lane_id="lane-0",
        backend_factory=lambda *args, **kwargs: backend,
    )
    smoke_path = root / "first_block_receipts/S96/smoke.json"
    main_path = root / "first_block_receipts/S96/frozen_probe.json"
    smoke, main = p.read_json(smoke_path), p.read_json(main_path)
    assert smoke["committed_outputs"] == main["committed_outputs"] == 8
    assert smoke["generation_seconds"] == pytest.approx(0.08)
    assert main["generation_seconds_per_output"] == pytest.approx(0.01)
    assert smoke["generated_length_p50"] == smoke["generated_length_p90"] == 2
    assert smoke["generated_tokens"] == main["generated_tokens"] == 16
    assert smoke["peak_memory_allocated_bytes"] == 8 * 1024
    assert main["peak_memory_allocated_bytes"] == 24 * 1024
    assert main["peak_memory_reserved_bytes"] == 24 * 2048
    assert main["peak_memory_allocated_bytes_observed_outputs"] == 8
    assert "before_model_load" in main["memory_peak_scope"]
    progress = p.read_jsonl(root / "progress.jsonl")
    assert progress[-1]["peak_memory_allocated_bytes"] == main["peak_memory_allocated_bytes"]
    assert progress[0]["chunk_outputs"] == 2
    saved = smoke_path.read_bytes(), main_path.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed resume attempted model load")

    r.run_worker(
        root,
        "S96",
        allow_gpu=True,
        resume=True,
        with_smoke=True,
        lane_id="lane-0",
        backend_factory=forbidden,
    )
    assert saved == (smoke_path.read_bytes(), main_path.read_bytes())


def test_unavailable_memory_stays_null_not_zero():
    metrics = r._resource_metrics(
        [{"generated_length": 2, "elapsed_seconds": 0.1, "peak_memory_allocated_bytes": None}]
    )
    assert metrics["peak_memory_allocated_bytes"] is None
    assert metrics["peak_memory_reserved_bytes"] is None
    assert metrics["peak_memory_allocated_bytes_observed_outputs"] == 0


def test_technical_smoke_failure_blocks_main_not_invalid_answer_rate(tmp_path):
    root = make_run(tmp_path)
    backend = FakeBackend(failures=100)
    with pytest.raises(RuntimeError, match="TECHNICAL_SMOKE_FAILED"):
        r.run_worker(
            root,
            "S96",
            allow_gpu=True,
            with_smoke=True,
            lane_id="lane-0",
            backend_factory=lambda *args, **kwargs: backend,
        )
    assert len(backend.calls) == 24
    assert not r.committed_rows(root)
    receipt = p.read_json(next((root / "worker_receipts").glob("*-interrupted-*.json")))
    assert any(job["status"] == "BLOCKED_SMOKE_TECHNICAL_FAILURE" for job in receipt["jobs"])


def test_maximum_five_worker_leases(tmp_path):
    from contextlib import ExitStack

    with ExitStack() as stack:
        for index in range(5):
            assert stack.enter_context(r.worker_slot(tmp_path)) == index
        with pytest.raises(RuntimeError, match="GPU_WORKER_LIMIT"), r.worker_slot(tmp_path):
            pytest.fail("Sixth slot was granted")
    with r.worker_slot(tmp_path) as index:
        assert index == 0


def test_frozen_data_code_runtime_drift_and_gpu_permission_fail_early(tmp_path, monkeypatch):
    root = make_run(tmp_path)
    with pytest.raises(ValueError, match="--allow-gpu"):
        r.run_worker(root, "S96")
    real_hashes = p.source_hashes()
    monkeypatch.setattr(p, "source_hashes", lambda: {**real_hashes, "changed.py": "different"})
    with pytest.raises(ValueError, match="implementation changed"):
        p.load_run(root)
    monkeypatch.setattr(p, "source_hashes", lambda: real_hashes)
    runtime = tmp_path / "changed-runtime.json"
    p.atomic_json(runtime, {"different": True})
    with pytest.raises(ValueError, match="Runtime configuration differs"):
        p.load_run(root, runtime)
    (root / "cases.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        p.load_run(root)


def test_alias_owner_never_independently_scheduled(tmp_path):
    root = make_run(tmp_path)
    context = p.load_run(root)
    for alias, owner in context["aliases"].items():
        if alias != owner:
            assert context["cases"][alias]["prompt"] == context["cases"][owner]["prompt"]
    assert all(
        context["aliases"][row["case_id"]] == row["case_id"] for row in context["manifest"]["jobs"]
    )


def test_cli_worker_phase_and_same_load_smoke_contract():
    args = parser().parse_args(
        [
            "worker",
            "--run",
            "/example",
            "--checkpoint",
            "S96",
            "--allow-gpu",
            "--resume",
            "--with-smoke",
            "--lane-id",
            "lane-0",
            "--phase",
            "primary_core_and_controls",
        ]
    )
    assert (
        args.with_smoke and args.phase == "primary_core_and_controls" and args.lane_id == "lane-0"
    )
