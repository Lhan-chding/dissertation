"""Only tiny immutable fixtures; no large arrays or experiment originals."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).with_name("q1_nonalias_summary.py")
spec = importlib.util.spec_from_file_location("q1_nonalias_helper", MODULE)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
SOURCE_ROOT = next(
    root
    for root in (Path.cwd(), *Path(__file__).resolve().parents)
    if (root / "src/modeling_v3/io.py").is_file()
)
helper._load_frozen_dependencies(fixture_import_root=SOURCE_ROOT)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def completed(path, summary, *, config=None):
    files = {
        str(p.relative_to(path)): helper.sha256_file(p) for p in path.rglob("*") if p.is_file()
    }
    value = {
        "status": "COMPLETE",
        "files": files,
        "summary": summary,
        "binding": {"config_sha256": config} if config else {},
    }
    write_json(path / "COMPLETE.json", value)
    return value


def fixture(tmp_path, *, identity_fault=None, inventory_fault=None, source_role=None):
    source, analysis = tmp_path / "original_Q1", tmp_path / "analysis"
    source.mkdir()
    analysis.mkdir()
    config = "a" * 64
    declared_units = []
    # Two unequal-sized error distributions demonstrate pooled quantiles, not averaged unit tails.
    for index, (case, same, values) in enumerate(
        [
            ("IDENTITY_HASH_SELECTED", False, [0, 0, 0, 0]),
            ("IDENTITY_HASH_SELECTED", False, [1, 1, 1, 10]),
            ("IDENTITY_HASH_SELECTED", True, [0, 0, 0, 0]),
            ("IDENTICAL_POLICY", True, [0, 0, 0, 0]),
        ]
    ):
        origin = f"fixture_seed{index}"
        name = f"{origin}_a8_b0_{case}_origin_n64"
        declared = {
            "id": name,
            "n": 64,
            "proposal": "origin",
            "case": case,
            "seed": index,
            "arm": "X_BASE",
            "anchor": 8,
            "bank": 0,
            "local_bank_index": 0,
            "historical_reused_fixed_policies": True,
            "repeats": 2,
            "metrics": {"RAW4": {}},
            "observed_case_inventory": {
                "identical_policy": same,
                "max_absolute_action_probability_difference": 0 if same else 0.1,
            },
        }
        if index == 0 and inventory_fault:
            inventory_fault(declared)
        declared_units.append(declared)
        originals = []
        for repeat in range(2):
            rep = source / "units" / name / f"repeat{repeat:03d}"
            rep.mkdir(parents=True)
            (rep / "statistics.npz").write_bytes(b"not read by helper")
            seed = helper._seed(name, repeat)
            identity = {
                "seed": seed,
                "n": 64,
                "proposal": "origin",
                "pilot_rng": helper._seed(seed, "pilot"),
                "main_rng": helper._seed(seed, "main"),
                "record_hash": helper.canonical_hash([name, repeat, seed]),
                "origin_id": origin,
                "anchor": 8,
                "bank": 0,
                "local_bank_index": 0,
                "historical_reused_fixed_policies": True,
                "candidate_index": 0 if case == "IDENTICAL_POLICY" else 1,
                "baseline_index": 0,
                "independent_count_status": "IDENTICAL_POLICY_ALIAS_ZERO_NO_DRAWS"
                if same
                else "SAMPLED",
                "source_policy_files": str(source / f"policy{index}.npz"),
                "source_policy_sha256": "b" * 64,
                "prompt_metadata_sha256": "c" * 64,
            }
            if index == repeat == 0 and identity_fault:
                identity_fault(identity)
            write_json(rep / "sampling_identity.json", identity)
            completed(rep, {"repeat": repeat})
            originals.append(
                {
                    "repeat": repeat,
                    "statistics_path": str(rep / "statistics.npz"),
                    "statistics_sha256": helper.sha256_file(rep / "statistics.npz"),
                    "sampling_identity_sha256": helper.sha256_file(rep / "sampling_identity.json"),
                    "completion_sha256": helper.sha256_file(rep / "COMPLETE.json"),
                }
            )
        dest = analysis / "units" / name
        dest.mkdir(parents=True)
        truth = np.zeros((2, 4))  # zero event truth MUST NOT classify a full policy as an alias.
        errors = np.zeros((2, 2, 4))
        errors[..., 0] = np.asarray(values).reshape(2, 2)
        errors[..., 3] = -errors[..., 0]
        np.savez_compressed(
            dest / "EXACT_RECORDS.npz", truth=truth, estimate_RAW4=errors, residual_RAW4=errors
        )
        write_json(dest / "ORIGINALS.json", originals)
        write_json(
            dest / "SUMMARY.json",
            {
                **{
                    k: declared[k]
                    for k in ("id", "n", "proposal", "case", "seed", "arm", "anchor", "bank")
                },
                "expected_repeats": 2,
                "observed_repeats": 2,
                "prompt_count": 2,
                "methods": {"RAW4": {"status": "RECOMPUTED_FROM_ORIGINAL_ESTIMATES"}},
            },
        )
    completed(
        source,
        {
            "stage": source_role or "Q1",
            "pilot": False,
            "scientific_status": "DEVELOPMENT_ONLY",
            "unit_count": 4,
            "units": declared_units,
        },
        config=config,
    )
    binding = {
        "kind": "Q1_ORIGINALS_DERIVED_SUMMARY",
        "fixture": True,
        "config_sha256": config,
        "analysis_source_sha256": "d" * 64,
        "inputs": {str(source): helper.sha256_file(source / "COMPLETE.json")},
    }
    write_json(analysis / "BINDING.json", binding)
    write_json(
        analysis / "Q1_ANALYSIS.json",
        {
            "stage": "Q1_DESCRIPTIVE_ANALYSIS",
            "fixture": True,
            "scientific_status": "FIXTURE_NOT_AUTHORIZATION",
            "originals_modified": False,
            "new_samples_generated": 0,
            "source_bindings": binding["inputs"],
            "unit_count": 4,
        },
    )
    helper.finalize_run(analysis, binding)
    return analysis, source


def run(analysis, out):
    return helper.analyze(
        analysis,
        out,
        expected_complete_sha256=helper.sha256_file(analysis / "COMPLETE.json"),
        fixture=True,
        fixture_import_root=SOURCE_ROOT,
    )


def test_actual_pooled_quantiles_and_zero_truth_not_alias(tmp_path):
    analysis, source = fixture(tmp_path)
    result = run(analysis, tmp_path / "output")
    selected = [r for r in result["rows"] if r["stratum"] == "SELECTED_NONIDENTICAL"]
    x = next(r for r in selected if r["event"] == "X")
    v = next(r for r in selected if r["event"] == "delta_v")
    assert x["count"] == 8 and x["unit_count"] == 2 and x["unit_repeat_count"] == 4
    expected = np.array([0, 0, 0, 0, 1, 1, 1, 10])
    assert x["signed_bias"] == expected.mean()
    assert x["mse"] == np.mean(expected**2)
    assert x["absolute_residual_quantiles"]["q95"] == pytest.approx(np.quantile(expected, 0.95))
    assert x["absolute_residual_quantiles"]["q95"] != pytest.approx(
        (0 + np.quantile(expected[4:], 0.95)) / 2
    )
    assert x["absolute_residual_quantiles"] == v["absolute_residual_quantiles"]
    assert result["historical_full_policy_arrays_reread"] is False
    assert result["classification_from_zero_truth"] is False
    assert {r["stratum"] for r in result["rows"]} == set(helper.STRATA)
    helper.verify_manifest(tmp_path / "output")
    assert all(r["mse"] == 0 for r in result["rows"] if r["stratum"] != "SELECTED_NONIDENTICAL")
    # Policy arrays do not exist and statistics are deliberately invalid NPZ: neither is read.
    assert not list(source.glob("policy*.npz"))


@pytest.mark.parametrize(
    "fault",
    [
        lambda x: x.update(pilot_rng=x["main_rng"]),
        lambda x: x.update(independent_count_status="IDENTICAL_POLICY_ALIAS_ZERO_NO_DRAWS"),
        lambda x: x.update(candidate_index=0),
    ],
)
def test_sampling_identity_conflicts_are_rejected_even_when_manifest_binds_them(tmp_path, fault):
    analysis, _ = fixture(tmp_path, identity_fault=fault)
    with pytest.raises(ValueError, match="identity disagrees"):
        run(analysis, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "fault",
    [
        lambda d: d.pop("observed_case_inventory"),
        lambda d: d["observed_case_inventory"].update(identical_policy=True),
    ],
)
def test_missing_or_conflicting_identity_is_not_recovered_from_zero_truth(tmp_path, fault):
    analysis, _ = fixture(tmp_path, inventory_fault=fault)
    with pytest.raises(ValueError, match="identity inventory"):
        run(analysis, tmp_path / "output")


def test_confirm_source_rejected(tmp_path):
    analysis, _ = fixture(tmp_path, source_role="Q3")
    with pytest.raises(ValueError, match="Q1 development"):
        run(analysis, tmp_path / "output")


def test_actual_source_identity_tamper_rejected(tmp_path):
    analysis, source = fixture(tmp_path)
    p = next(source.glob("units/*/repeat000/sampling_identity.json"))
    p.write_text(p.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        run(analysis, tmp_path / "output")


def test_analysis_manifest_and_expected_complete_tamper_rejected(tmp_path):
    analysis, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        helper.analyze(
            analysis,
            tmp_path / "bad",
            expected_complete_sha256="0" * 64,
            fixture=True,
            fixture_import_root=SOURCE_ROOT,
        )
    p = next(analysis.glob("units/*/EXACT_RECORDS.npz"))
    with p.open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="manifest path/hash"):
        run(analysis, tmp_path / "output")


def test_no_output_overwrite_or_source_overlap(tmp_path):
    analysis, source = fixture(tmp_path)
    run(analysis, tmp_path / "output")
    with pytest.raises(ValueError, match="must not exist"):
        run(analysis, tmp_path / "output")
    with pytest.raises(ValueError, match="overlap"):
        run(analysis, source / "new")


def test_no_production_cpu_bypass(monkeypatch, tmp_path):
    monkeypatch.setattr(helper.platform, "system", lambda: "Linux")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="CPU-only"):
        helper.analyze(tmp_path / "missing", tmp_path / "out", expected_complete_sha256="0" * 64)


@pytest.mark.parametrize("fault", ["residual", "alias_truth"])
def test_rebound_analysis_still_checks_original_residual_arithmetic_and_alias_consistency(
    tmp_path, fault
):
    analysis, _ = fixture(tmp_path)
    if fault == "residual":
        path = next(analysis.glob("units/*seed0*/EXACT_RECORDS.npz"))
    else:
        path = next(analysis.glob("units/*seed2*/EXACT_RECORDS.npz"))
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    if fault == "residual":
        arrays["residual_RAW4"][0, 0, 0] += 1
    else:
        arrays["truth"][0, 0] = 1
        arrays["estimate_RAW4"] = arrays["estimate_RAW4"] + arrays["truth"]
    np.savez_compressed(path, **arrays)
    binding = json.loads((analysis / "BINDING.json").read_text())
    (analysis / "RUN_MANIFEST.json").unlink()
    (analysis / "COMPLETE.json").unlink()
    helper.finalize_run(analysis, binding)
    with pytest.raises(ValueError, match=r"residual ledger|alias contradicts"):
        run(analysis, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("field", ["SLURM_JOB_GPUS", "SLURM_STEP_GPUS"])
def test_slurm_allocated_gpu_is_rejected_even_when_cuda_hidden(monkeypatch, field):
    monkeypatch.setattr(helper.platform, "system", lambda: "Linux")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv(field, "0")
    with pytest.raises(RuntimeError, match="CPU-only"):
        helper._scope(False)


def test_no_checkout_environment_fallback(monkeypatch):
    monkeypatch.delenv("SSVC_V3_CHECKOUT", raising=False)
    with pytest.raises(ValueError, match="SSVC_V3_CHECKOUT"):
        helper._load_frozen_dependencies()


def tiny_snapshot(tmp_path, count=1025):
    checkout = tmp_path / "snapshot"
    src = checkout / "ssvc_flow/src"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("# tiny snapshot fixture\n")
    for i in range(count - 1):
        path = checkout / "tiny_bound_files" / str(i)
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(bytes([i % 256]))
    files = {
        str(p.relative_to(checkout)): helper._local_sha256(p)
        for p in checkout.rglob("*")
        if p.is_file()
    }
    write_json(checkout / "CODE_SNAPSHOT_MANIFEST.json", files)
    return checkout


def test_snapshot_pin_rejects_other_manifest_before_any_src_import(tmp_path, monkeypatch):
    assert (
        helper.SNAPSHOT_SHA256 == "cd703f28725212aa17e86ac8d0be82890f253c955aba7293fc9d84728d34b52d"
    )
    checkout = tiny_snapshot(tmp_path, count=1)
    monkeypatch.setenv("SSVC_V3_CHECKOUT", str(checkout))
    with pytest.raises(ValueError, match="SHA-256 differs"):
        helper._load_frozen_dependencies()


def test_full_inventory_gate_checks_all_1025_hashes_and_rejects_extra_src(tmp_path, monkeypatch):
    checkout = tiny_snapshot(tmp_path)
    # Test-only patch permits a 1025-byte synthetic snapshot to exercise inventory mechanics.
    monkeypatch.setattr(
        helper, "SNAPSHOT_SHA256", helper._local_sha256(checkout / "CODE_SNAPSHOT_MANIFEST.json")
    )
    _, files = helper._verify_snapshot(checkout)
    assert len(files) == 1025
    last = checkout / "tiny_bound_files/1023"
    original = last.read_bytes()
    last.write_bytes(b"changed")
    with pytest.raises(ValueError, match="file changed"):
        helper._verify_snapshot(checkout)
    last.write_bytes(original)
    (checkout / "ssvc_flow/src/extra.py").write_text("# unregistered\n")
    with pytest.raises(ValueError, match="src inventory"):
        helper._verify_snapshot(checkout)


def test_snapshot_count_cannot_be_relaxed(tmp_path, monkeypatch):
    checkout = tiny_snapshot(tmp_path, count=1)
    monkeypatch.setattr(
        helper, "SNAPSHOT_SHA256", helper._local_sha256(checkout / "CODE_SNAPSHOT_MANIFEST.json")
    )
    with pytest.raises(ValueError, match="exactly 1025"):
        helper._verify_snapshot(checkout)


def test_product_import_rejects_foreign_preloaded_src(monkeypatch, tmp_path):
    import types

    monkeypatch.setitem(
        sys.modules,
        "src.foreign_fixture",
        types.SimpleNamespace(__file__=str(tmp_path / "foreign.py")),
    )
    with pytest.raises(ValueError, match="different checkout"):
        helper._load_frozen_dependencies(fixture_import_root=SOURCE_ROOT)


def test_copied_operator_module_imports_no_src_until_controlled_initialization(tmp_path):
    copied = tmp_path / "external/operators/q1_nonalias_summary.py"
    copied.parent.mkdir(parents=True)
    shutil.copyfile(MODULE, copied)
    program = (
        "import importlib.util,sys; "
        f"s=importlib.util.spec_from_file_location('external_operator',{str(copied)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "assert not any(k=='src' or k.startswith('src.') for k in sys.modules); "
        f"r=m._load_frozen_dependencies(fixture_import_root={str(SOURCE_ROOT)!r}); "
        "assert r['mode']=='EXPLICIT_TINY_FIXTURE_IMPORT_ROOT'; "
        f"assert r['src_import_root']=={str(SOURCE_ROOT)!r}; "
        "assert len(r['actual_src_sha256'])==64"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_copied_docs_operator_and_test_remain_runnable(tmp_path):
    copied = tmp_path / "docs/modeling_v3/operators"
    copied.mkdir(parents=True)
    shutil.copyfile(MODULE, copied / MODULE.name)
    testfile = copied / Path(__file__).name
    shutil.copyfile(__file__, testfile)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    # Select one existing small numerical test; do not recursively run this relocation test.
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(testfile), "-q", "-k", "actual_pooled_quantiles"],
        cwd=SOURCE_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


def test_production_analysis_rejects_previous_candidate_measurement_sources(tmp_path, monkeypatch):
    analysis, source = fixture(tmp_path)
    complete = json.loads((source / "COMPLETE.json").read_text())
    complete["binding"]["source_hashes"] = {"src/modeling_v3/cpu_campaign.py": "0" * 64}
    write_json(source / "COMPLETE.json", complete)
    binding = json.loads((analysis / "BINDING.json").read_text())
    binding["fixture"] = False
    binding["analysis_source_sha256"] = helper.sha256_file(
        SOURCE_ROOT / "src/modeling_v3/q1_results.py"
    )
    binding["inputs"] = {str(source): helper.sha256_file(source / "COMPLETE.json")}
    write_json(analysis / "BINDING.json", binding)
    report = json.loads((analysis / "Q1_ANALYSIS.json").read_text())
    report.update(
        fixture=False,
        scientific_status="DEVELOPMENT_ONLY_NOT_CERTIFIED",
        source_bindings=binding["inputs"],
    )
    write_json(analysis / "Q1_ANALYSIS.json", report)
    (analysis / "RUN_MANIFEST.json").unlink()
    (analysis / "COMPLETE.json").unlink()
    helper.finalize_run(analysis, binding)
    # Tiny gate isolation only. No CLI option can override these production checks.
    monkeypatch.setattr(helper, "_scope", lambda fixture: None)
    monkeypatch.setattr(
        helper,
        "_load_frozen_dependencies",
        lambda **kw: {"actual_src_files": {"src/modeling_v3/cpu_campaign.py": "1" * 64}},
    )
    with pytest.raises(ValueError, match=r"measurement source.*candidate06"):
        helper.analyze(
            analysis,
            tmp_path / "out",
            expected_complete_sha256=helper.sha256_file(analysis / "COMPLETE.json"),
        )
    assert not (tmp_path / "out").exists()
