from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import tarfile
from pathlib import Path

import pytest

from compensability_v4.data.splits import DatasetSplit
from compensability_v4.qwen.phase5_support import HeldOutNaturalError
from compensability_v5.qwen.study_a_runtime import (
    BASE_SHA256,
    RAW_ARCHIVE_MEMBER,
    RAW_ARCHIVE_SHA256,
    STUDY_A_ACK,
    T_ADAPTER_SHA256,
    build_phase2_study_a_scenarios,
    build_study_a_scenarios,
    capture_phase2a_natural_observations,
    load_phase2a_child,
    load_natural_errors,
    require_study_a_authorization,
    run_study_a,
)
from compensability_v5.server_runtime import phase3 as phase3_server

SHA = "a" * 64


def _load_cli():
    path = Path(__file__).resolve().parents[2] / "scripts/v5/12_run_study_a.py"
    specification = importlib.util.spec_from_file_location("test_v5_study_a_cli", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _error() -> HeldOutNaturalError:
    return HeldOutNaturalError(
        scene_id="natural-a",
        family="cross_series",
        split=DatasetSplit.SUPPORT_DEV,
        truth=(2, 3, 4, 5),
        observed=(9, 3, 4, 5),
        error_indices=(0,),
        facts=(
            {
                "fact_id": "known-2",
                "type": "known_value",
                "index": 2,
                "value": 4,
            },
            {
                "fact_id": "sum-01",
                "type": "pair_sum",
                "left_index": 0,
                "right_index": 1,
                "total": 5,
            },
            {
                "fact_id": "sum-13",
                "type": "pair_sum",
                "left_index": 1,
                "right_index": 3,
                "total": 8,
            },
        ),
        image_path="images/natural-a.png",
        stage1_model_sha256=BASE_SHA256,
        stage1_raw_output="9,3,4,5",
    )


def _trend_error() -> HeldOutNaturalError:
    return HeldOutNaturalError(
        scene_id="natural-trend",
        family="trend",
        split=DatasetSplit.SUPPORT_DEV,
        truth=(2, 3, 4, 5),
        observed=(2, 9, 4, 5),
        error_indices=(1,),
        facts=(
            {
                "fact_id": "trend-012",
                "type": "arithmetic_progression",
                "indices": [0, 1, 2],
            },
            {
                "fact_id": "trend-123",
                "type": "arithmetic_progression",
                "indices": [1, 2, 3],
            },
        ),
        image_path="images/natural-trend.png",
        stage1_model_sha256=BASE_SHA256,
        stage1_raw_output="2,9,4,5",
    )


class _Parameter:
    requires_grad = False

    def requires_grad_(self, value: bool) -> None:
        self.requires_grad = value


class _FakeModel:
    def __init__(self, *, fail_after_generations: int | None = None) -> None:
        self.generate_calls = 0
        self.fail_after_generations = fail_after_generations
        self._parameters = (_Parameter(),)

    def parameters(self):
        return iter(self._parameters)

    def eval(self) -> None:
        return None

    def phase5_generate(self, prompt: str, **options: object):
        assert "Observed values:" in prompt
        assert "Constraint rows (A | b):" in prompt
        assert prompt.endswith("Return exactly four comma-separated integers only.\n")
        self.generate_calls += 1
        if (
            self.fail_after_generations is not None
            and self.generate_calls > self.fail_after_generations
        ):
            raise RuntimeError("simulated interruption")
        return "2,3,4,5", (2, 3, 4, 5)

    def phase5_score(self, prompt: str, completion: str) -> float:
        del prompt
        return -1.0 if completion == "2,3,4,5" else -3.0

    def study_a_observe(self, **kwargs: object):
        assert kwargs["prompt"].startswith("Read the chart")
        return "9,3,4,5", (9, 3, 4, 5)


def _loader(models: dict[str, _FakeModel]):
    return lambda checkpoint: (models[checkpoint], object())


def _phase2a_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "phase2a"
    (root / "images").mkdir(parents=True)
    image = root / "images/example.png"
    image.write_bytes(b"fixed-image")
    row = {
        "schema_version": 1,
        "scene_id": "parent-a-familiar",
        "semantic_scene_id": "parent-a",
        "family": "pair_sum",
        "graph_axis": "familiar",
        "truth": [2, 3, 4, 5],
        "constraint_matrix": [[1, 1, 0, 0], [0, 1, 0, 1]],
        "constraint_targets": [5, 8],
        "answer_operation": {"operator": "sum", "indices": [0, 1]},
        "transformation": {"kind": "identity"},
        "image_path": "images/example.png",
        "image_sha256": hashlib.sha256(b"fixed-image").hexdigest(),
        "observation_status": "pending_server_capture",
    }
    rows = root / "pre_model_rows.jsonl"
    rows.write_text(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "status": "PHASE_2A_PRE_MODEL_FROZEN",
        "row_count": 1,
        "rows_sha256": hashlib.sha256(rows.read_bytes()).hexdigest(),
        "model_calls": 0,
        "observation_capture_required": True,
    }
    (root / "parent_manifest.json").write_text(json.dumps(manifest))
    return root


def test_scenarios_cover_registered_orbit_and_error_axes() -> None:
    scenarios = build_study_a_scenarios(_error())

    assert [scenario.graph_axis for scenario in scenarios] == [
        "canonical",
        "variable_permuted",
        "error_location_permuted",
        "fact_order_permuted",
        "equivalent_basis_graph_ood",
    ]
    assert all(scenario.prompt == scenario.prompt for scenario in scenarios)
    assert scenarios[0].truth == (2, 3, 4, 5)
    assert scenarios[1].truth == (3, 4, 5, 2)
    assert scenarios[1].observed == (3, 4, 5, 9)
    assert scenarios[1].pushforward_permutation == (1, 2, 3, 0)
    assert scenarios[2].truth == scenarios[0].truth
    assert scenarios[2].observed == (2, 10, 4, 5)
    assert scenarios[3].constraint_matrix == tuple(reversed(scenarios[0].constraint_matrix))
    assert scenarios[4].constraint_matrix[0] == tuple(
        left + right
        for left, right in zip(
            scenarios[0].constraint_matrix[0],
            scenarios[0].constraint_matrix[1],
            strict=True,
        )
    )
    assert all(scenario.fiber_size > 0 for scenario in scenarios)
    assert len({scenario.prompt_sha256 for scenario in scenarios}) == 5


def test_trend_facts_are_rendered_as_exact_linear_equations() -> None:
    canonical = build_study_a_scenarios(_trend_error())[0]

    assert canonical.constraint_matrix == ((1, -2, 1, 0), (0, 1, -2, 1))
    assert canonical.constraint_targets == (0, 0)
    assert "1,-2,1,0 = 0" in canonical.prompt


def test_phase2a_capture_freezes_all_parents_and_builds_five_audit_axes(
    tmp_path: Path,
) -> None:
    child = tmp_path / "child"
    frozen, child_hash = capture_phase2a_natural_observations(
        phase2a_root=_phase2a_fixture(tmp_path),
        output_root=child,
        work_root=tmp_path / "capture-work",
        model=_FakeModel(),
        processor=object(),
        expected_parent_count=1,
        seed=19,
    )

    assert len(frozen) == 1
    assert frozen[0]["natural_observation"] == [9, 3, 4, 5]
    assert frozen[0]["capture_label"] == "primary_single_in_domain"
    assert frozen[0]["error_count"] == 1
    assert frozen[0]["prompt"].endswith(
        "Return exactly four comma-separated integers only.\n"
    )
    assert len(build_phase2_study_a_scenarios(frozen[0])) == 5
    loaded, loaded_hash = load_phase2a_child(child)
    assert loaded == frozen
    assert loaded_hash == child_hash
    manifest = json.loads((child / "child_manifest.json").read_text())
    assert manifest["parent_manifest_modified"] is False
    assert manifest["semantic_scene_count"] == 1
    assert manifest["capture_label_counts"] == {"primary_single_in_domain": 1}


def test_run_study_a_executes_base_and_t_and_atomically_publishes(tmp_path: Path) -> None:
    output_root = tmp_path / "published"
    work_root = tmp_path / "resume"
    models = {"Base": _FakeModel(), "T": _FakeModel()}

    summary = run_study_a(
        errors=(_error(),),
        raw_archive_sha256=SHA,
        output_root=output_root,
        work_root=work_root,
        checkpoint_loader=_loader(models),
        k=8,
        sampling_seed=19,
    )

    assert summary["status"] == "V5_STUDY_A_EXECUTED"
    assert summary["scenario_checkpoint_count"] == 10
    assert summary["training_invoked"] is False
    assert summary["rl_invoked"] is False
    assert summary["prompt_search_invoked"] is False
    rows = [
        json.loads(line)
        for line in (output_root / "per_scenario.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 10
    assert {row["checkpoint"] for row in rows} == {"Base", "T"}
    assert all(len(row["sample_raw_outputs"]) == 8 for row in rows)
    assert len({tuple(row["sample_seeds"]) for row in rows}) == 1
    assert all(math.isfinite(row["candidate_margin_true_observed"]) for row in rows)
    assert all(
        row["candidate_margin_true_observed"] == pytest.approx(2.0)
        for row in rows
        if row["graph_axis"] == "canonical"
    )
    assert all("fiber_size" in row for row in rows)
    assert any(row["equivariance_defect"] for row in rows)
    assert (output_root / "raw_trace.jsonl").read_bytes() == (
        work_root / "raw_trace.jsonl"
    ).read_bytes()
    assert set(summary["by_graph_axis"]) == {
        "canonical",
        "variable_permuted",
        "error_location_permuted",
        "fact_order_permuted",
        "equivalent_basis_graph_ood",
    }
    assert set(summary["by_family"]) == {"cross_series"}
    manifest = json.loads((output_root / "manifest.json").read_text())
    assert manifest["source_sha256"] == {
        "Base": BASE_SHA256,
        "T": T_ADAPTER_SHA256,
        "raw_archive": SHA,
    }


def test_resume_uses_completed_raw_trace_without_duplicate_model_calls(tmp_path: Path) -> None:
    output_root = tmp_path / "published"
    work_root = tmp_path / "resume"
    interrupted_base = _FakeModel(fail_after_generations=9)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_study_a(
            errors=(_error(),),
            raw_archive_sha256=SHA,
            output_root=output_root,
            work_root=work_root,
            checkpoint_loader=_loader({"Base": interrupted_base, "T": _FakeModel()}),
            k=8,
            sampling_seed=19,
        )

    assert not output_root.exists()
    assert len((work_root / "raw_trace.jsonl").read_text().splitlines()) == 1

    resumed_base = _FakeModel()
    run_study_a(
        errors=(_error(),),
        raw_archive_sha256=SHA,
        output_root=output_root,
        work_root=work_root,
        checkpoint_loader=_loader({"Base": resumed_base, "T": _FakeModel()}),
        k=8,
        sampling_seed=19,
    )

    assert resumed_base.generate_calls == 4 * 9
    assert len((work_root / "raw_trace.jsonl").read_text().splitlines()) == 10


def test_resume_rejects_tampered_raw_trace_provenance(tmp_path: Path) -> None:
    work_root = tmp_path / "resume"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_study_a(
            errors=(_error(),),
            raw_archive_sha256=SHA,
            output_root=tmp_path / "first-output",
            work_root=work_root,
            checkpoint_loader=_loader(
                {"Base": _FakeModel(fail_after_generations=9), "T": _FakeModel()}
            ),
            k=8,
            sampling_seed=19,
        )
    trace = work_root / "raw_trace.jsonl"
    row = json.loads(trace.read_text())
    row["prompt_sha256"] = "0" * 64
    trace.write_text(json.dumps(row) + "\n")

    with pytest.raises(RuntimeError, match="provenance drifted"):
        run_study_a(
            errors=(_error(),),
            raw_archive_sha256=SHA,
            output_root=tmp_path / "second-output",
            work_root=work_root,
            checkpoint_loader=_loader({"Base": _FakeModel(), "T": _FakeModel()}),
            k=8,
            sampling_seed=19,
        )


def test_load_natural_errors_reads_only_hash_bound_raw_archive_member(tmp_path: Path) -> None:
    archive = tmp_path / "raw.tar.gz"
    payload = (json.dumps(_error().to_mapping(), sort_keys=True) + "\n").encode()
    with tarfile.open(archive, "w:gz") as stream:
        info = tarfile.TarInfo(RAW_ARCHIVE_MEMBER)
        info.size = len(payload)
        stream.addfile(info, io.BytesIO(payload))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()

    errors, observed_digest = load_natural_errors(archive, expected_sha256=digest)

    assert observed_digest == digest
    assert errors == (_error(),)
    with pytest.raises(ValueError, match="archive SHA-256 mismatch"):
        load_natural_errors(archive, expected_sha256="0" * 64)


def test_authorization_is_explicit_offline_and_hashes_are_fully_pinned() -> None:
    assert BASE_SHA256 == "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87"
    assert T_ADAPTER_SHA256 == "807a61c2e3f7b532b162554dee6e7df83d654fb1f10cc464e9dcb5f6f8efd5c7"
    environment = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    require_study_a_authorization(
        execute=True,
        acknowledgement=STUDY_A_ACK,
        environment=environment,
    )
    with pytest.raises(PermissionError, match="acknowledgement"):
        require_study_a_authorization(
            execute=True,
            acknowledgement="wrong",
            environment=environment,
        )
    with pytest.raises(RuntimeError, match="offline"):
        require_study_a_authorization(
            execute=True,
            acknowledgement=STUDY_A_ACK,
            environment={"HF_HUB_OFFLINE": "1"},
        )


def test_cli_blocks_before_touching_missing_inputs_without_explicit_execute(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli()
    monkeypatch.setattr(
        "sys.argv",
        [
            "12_run_study_a.py",
            "--raw-archive",
            "/does/not/exist.tar.gz",
            "--t-adapter",
            "/does/not/exist",
        ],
    )

    assert cli.main() == 2
    assert "explicit --execute" in capsys.readouterr().out


def test_cli_success_path_delegates_to_frozen_runtime_and_prints_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    output_root = tmp_path / "outputs/run"
    work_root = tmp_path / "work/run"
    cli.ALLOWED_OUTPUT_ROOT = tmp_path / "outputs"
    cli.ALLOWED_WORK_ROOT = tmp_path / "work"
    cli.ALLOWED_DATA_ROOT = tmp_path / "data"
    cli.CAPTURE_WORK_ROOT = tmp_path / "work/capture"
    calls: dict[str, object] = {}
    monkeypatch.setattr(cli, "require_study_a_authorization", lambda **kwargs: None)
    monkeypatch.setattr(cli, "load_natural_errors", lambda *args, **kwargs: ((_error(),), SHA))
    monkeypatch.setattr(cli, "require_t_adapter", lambda path: path)
    monkeypatch.setattr(cli, "_require_gpu_runtime", lambda: None)

    def fake_run(**kwargs: object) -> dict[str, object]:
        calls.update(kwargs)
        output_root.mkdir(parents=True)
        for name in ("per_scenario.jsonl", "raw_trace.jsonl", "summary.json", "manifest.json"):
            (output_root / name).write_text("{}\n")
        return {"status": "V5_STUDY_A_EXECUTED"}

    monkeypatch.setattr(cli, "run_phase2a_study_a", fake_run)
    monkeypatch.setattr("sys.argv", [
        "12_run_study_a.py",
        "--execute",
        "--ack",
        STUDY_A_ACK,
        "--phase2a-root",
        str(tmp_path / "phase2a"),
        "--child-root",
        str(tmp_path / "data/child"),
        "--t-adapter",
        str(tmp_path / "adapter"),
        "--output-root",
        str(output_root),
        "--work-root",
        str(work_root),
    ])

    assert cli.main() == 0
    assert calls["k"] == 8
    assert calls["sampling_seed"] == 2026082101
    assert calls["expected_parent_count"] == 96
    output = capsys.readouterr().out
    assert "atomically published" in output
    assert output.count("SHA256") == 4


def test_generic_orbit_callback_delegates_to_study_a_and_publishes_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_archive = tmp_path / "raw.tar.gz"
    raw_archive.write_bytes(b"raw")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    output = tmp_path / "audits/orbit_support.json"
    validation = {
        "schema_version": 1,
        "phase": "phase3_orbit_audit",
        "input_sha256": {str(raw_archive): RAW_ARCHIVE_SHA256},
        "output": str(output),
    }
    monkeypatch.setenv("COMPBIAS_V5_T_ADAPTER", str(adapter))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(phase3_server, "load_natural_errors", lambda path: ((_error(),), SHA))
    monkeypatch.setattr(phase3_server, "require_t_adapter", lambda path: path)

    def fake_run(**kwargs: object) -> dict[str, object]:
        result_root = kwargs["output_root"]
        assert isinstance(result_root, Path)
        result_root.mkdir(parents=True)
        (result_root / "summary.json").write_text("{}\n")
        return {"status": "V5_STUDY_A_EXECUTED", "by_graph_axis": {}}

    monkeypatch.setattr(phase3_server, "run_study_a", fake_run)

    result = phase3_server.run_orbit_audit(validation, {"task": "orbit_audit", "k": 8})

    assert result["status"] == "V5_STUDY_A_ORBIT_CALLBACK_COMPLETE"
    assert json.loads(output.read_text()) == result


def test_generic_gradient_callback_is_explicitly_deferred_and_writes_nothing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "gradient_alignment.json"
    with pytest.raises(RuntimeError, match="DEFERRED_NOT_REQUIRED_BY_4090_DECISIVE_PILOT"):
        phase3_server.run_gradient_alignment(
            {
                "schema_version": 1,
                "phase": "phase3_gradient_alignment",
                "input_sha256": {},
                "output": str(output),
            },
            {"task": "gradient_alignment"},
        )
    assert not output.exists()


def test_orbit_callback_recognizes_a_completed_atomic_study_a_publication(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "orbit_support_study_a"
    result_root.mkdir()
    sources = {
        "raw_archive": SHA,
        "Base": BASE_SHA256,
        "T": T_ADAPTER_SHA256,
    }
    expected = {
        "schema_version": 1,
        "status": "V5_STUDY_A_EXECUTED",
        "source_sha256": sources,
    }
    (result_root / "summary.json").write_text(json.dumps(expected))
    (result_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "V5_STUDY_A_ATOMICALLY_PUBLISHED",
                "source_sha256": sources,
            }
        )
    )

    assert phase3_server._completed_summary(result_root, SHA) == expected
    assert phase3_server._completed_summary(tmp_path / "absent", SHA) is None


def test_cli_gpu_preflight_reports_missing_packages_and_missing_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli()
    original_import = __import__

    def missing_import(name: str, *args: object, **kwargs: object):
        if name in {"transformers", "peft"}:
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", missing_import)
    with pytest.raises(RuntimeError, match="transformers, peft"):
        cli._require_gpu_runtime()

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def is_bf16_supported() -> bool:
            return False

    class _Torch:
        cuda = _Cuda()

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "torch":
            return _Torch()
        if name in {"transformers", "peft"}:
            return object()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        cli._require_gpu_runtime()
