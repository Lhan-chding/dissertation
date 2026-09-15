"""The final family must consume original-backed stage analyses, not p-value files."""

import json
import sys
import types

import pytest

from src.modeling_v3.io import atomic_json, canonical_hash, finalize_run, sha256_file


def test_production_family_refuses_local_execution_before_reading_inputs(tmp_path, monkeypatch):
    from src.modeling_v3.primary_family import finalize_primary_family

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="server Slurm CPU"):
        finalize_primary_family({}, *([tmp_path / "absent"] * 4), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def _inputs(tmp_path, monkeypatch):
    family = {"family_id": "FIXTURE", "hypotheses": [{"id": f"RQ{i}"} for i in range(1, 5)]}
    cpu_lock = tmp_path / "cpu_lock.json"
    atomic_json(cpu_lock, {"selected": {"primary_comparison_family": family}})
    vlm_lock = tmp_path / "vlm_lock.json"
    atomic_json(
        vlm_lock,
        {"parent_cpu_selection": {"path": str(cpu_lock), "sha256": sha256_file(cpu_lock)}},
    )
    paths = []
    for stage in ("CPU", "VLM"):
        root = tmp_path / stage
        root.mkdir()
        report = {"fixture": True, "stage_for_fixture": stage}
        atomic_json(root / "TEST_ANALYSIS.json", report)
        finalize_run(root, {"fixture": True})
        paths.append(root / "TEST_ANALYSIS.json")
    calls = []

    def verifier(stage):
        def verify(config, lock, report, *, fixture):
            assert fixture is True
            assert report["sha256"] == sha256_file(report["path"])
            calls.append(stage)
            return {
                "family": family,
                "results": [
                    {"hypothesis_id": f"RQ{i}"} for i in ([1, 2, 3] if stage == "CPU" else [4])
                ],
            }

        return verify

    from src.modeling_v3 import schema

    monkeypatch.setattr(schema, "verify_selection_lock", lambda c, p: json.loads(p.read_text()))
    monkeypatch.setitem(
        sys.modules,
        "src.modeling_v3.vlm_response",
        types.SimpleNamespace(
            verify_vlm_selection_lock=lambda c, p, **kw: json.loads(p.read_text())
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.modeling_v3.cpu_results",
        types.SimpleNamespace(verify_cpu_primary_comparisons=verifier("CPU")),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.modeling_v3.vlm_results",
        types.SimpleNamespace(verify_vlm_primary_comparison=verifier("VLM")),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.modeling_v3.frozen_comparisons",
        types.SimpleNamespace(
            summarize_primary_family=lambda f, r: {"status": "FIXTURE", "count": len(r)}
        ),
    )
    return cpu_lock, paths[0], vlm_lock, paths[1], family, calls


def test_family_recomputes_both_stages_and_binds_complete_originals(tmp_path, monkeypatch):
    from src.modeling_v3.primary_family import finalize_primary_family

    cpu_lock, cpu, vlm_lock, vlm, family, calls = _inputs(tmp_path, monkeypatch)
    result = finalize_primary_family(
        {}, cpu_lock, cpu, vlm_lock, vlm, tmp_path / "out", fixture=True
    )
    assert calls == ["CPU", "VLM"]
    assert result["family_sha256"] == canonical_hash(family)
    assert result["summary"]["count"] == 4
    assert result["fixture"] is True
    assert result["online_ssvc"] == "NOT_CERTIFIED"
    assert (tmp_path / "out/COMPLETE.json").is_file()


@pytest.mark.parametrize(
    "fault", ["changed_parent", "changed_report", "missing_complete", "wrong_fixture"]
)
def test_family_rejects_broken_analysis_chain_before_recomputation(tmp_path, monkeypatch, fault):
    from src.modeling_v3.primary_family import finalize_primary_family

    cpu_lock, cpu, vlm_lock, vlm, _family, calls = _inputs(tmp_path, monkeypatch)
    if fault == "changed_parent":
        cpu_lock.write_text(cpu_lock.read_text() + " ")
    elif fault == "changed_report":
        cpu.write_text(cpu.read_text() + " ")
    elif fault == "missing_complete":
        (cpu.parent / "COMPLETE.json").unlink()
    else:
        fresh = tmp_path / "production_report"
        fresh.mkdir()
        atomic_json(fresh / "TEST_ANALYSIS.json", {"fixture": False})
        finalize_run(fresh, {})
        cpu = fresh / "TEST_ANALYSIS.json"
    with pytest.raises(ValueError):
        finalize_primary_family({}, cpu_lock, cpu, vlm_lock, vlm, tmp_path / "out", fixture=True)
    assert calls == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("fault", ["different_family", "CPU_result_as_RQ4"])
def test_family_rejects_cross_stage_substitution(tmp_path, monkeypatch, fault):
    from src.modeling_v3.primary_family import finalize_primary_family

    cpu_lock, cpu, vlm_lock, vlm, family, _calls = _inputs(tmp_path, monkeypatch)
    module = sys.modules["src.modeling_v3.vlm_results"]
    original = module.verify_vlm_primary_comparison

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        if fault == "different_family":
            result["family"] = {**family, "family_id": "DIFFERENT"}
        else:
            result["results"] = [{"hypothesis_id": "RQ3"}]
        return result

    monkeypatch.setattr(module, "verify_vlm_primary_comparison", changed)
    with pytest.raises(ValueError):
        finalize_primary_family({}, cpu_lock, cpu, vlm_lock, vlm, tmp_path / "out", fixture=True)
    assert not (tmp_path / "out").exists()
