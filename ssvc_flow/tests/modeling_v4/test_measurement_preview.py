import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.modeling_v3.io import canonical_hash
from src.modeling_v4 import gpu_collect as gpu
from src.modeling_v4 import measurement_preview as preview


def banks():
    result = []
    for i in range(8):
        result.append(
            {
                "bank_id": f"calibration_{i:03d}",
                "role": "calibration",
                "origin_state_restored": True,
                "policies": {
                    "joint_0": {"inference_fingerprint": "base"},
                    "joint_1": {"inference_fingerprint": "base" if i % 2 == 0 else str(i)},
                    "no_x_off_1": {"inference_fingerprint": "ablation"},
                },
                "contrasts": {
                    preview.PRIMARY: {"exact_inference_alias": i % 2 == 0, "norm": float(i % 2)}
                },
            }
        )
    return result


def test_fixed_selection_preserves_roles_and_ignores_outcomes():
    pool = banks()
    before = copy.deepcopy(pool)
    selected = preview.select_banks(pool)
    assert [b["bank_id"] for b in selected] == ["calibration_000", "calibration_001"]
    assert all(b["role"] == "calibration" for b in selected)
    assert pool == before
    pool[1]["unrelated_response"] = "bad"
    assert preview.select_banks(pool)[1]["bank_id"] == "calibration_001"
    with pytest.raises(ValueError, match="first eight"):
        preview.select_banks(pool[:2])
    pool[1]["contrasts"][preview.PRIMARY]["exact_inference_alias"] = True
    with pytest.raises(ValueError, match="alias"):
        preview.select_banks(pool)


def test_authorization_before_reads_and_campaign_write_rejected(tmp_path):
    with pytest.raises(PermissionError):
        preview.run("missing")
    for out in (tmp_path, tmp_path / "new", tmp_path.parent):
        with pytest.raises(ValueError, match="separate"):
            preview._outside_campaign(out, tmp_path)
    assert (
        preview._outside_campaign(tmp_path / "preview", tmp_path / "campaign")
        == tmp_path / "preview"
    )


def test_runtime_allows_new_source_only_and_rejects_layout(monkeypatch):
    old = {k: "same" for k in preview.COMPATIBILITY_KEYS} | {"source_hash": "old"}
    policy = {"checkpoint": {"identity": old}, "parameter_layout": [1], "module_scalings": {"x": 1}}
    runtime = {"identity": old | {"source_hash": "new"}}
    monkeypatch.setattr(
        gpu, "_policy_layout", lambda r: {"parameter_layout": [1], "module_scalings": {"x": 1}}
    )
    preview._check_runtime(runtime, policy, [])
    for field in preview.COMPATIBILITY_KEYS:
        bad = {"identity": runtime["identity"] | {field: "different"}}
        with pytest.raises(ValueError, match=field):
            preview._check_runtime(bad, policy, [])
    policy["module_scalings"] = {"x": 2}
    with pytest.raises(ValueError, match="layout"):
        preview._check_runtime(runtime, policy, [])


def test_streamed_first_prompt_survives_failure_and_resume_skips_it(tmp_path, monkeypatch):
    plan = {
        "root": str(tmp_path),
        "plan_hash": "frozen",
        "measurement_source": {"sha256": "new"},
        "source_task_id": "map_41001_X_BASE_32",
        "scope": "EXPLORATORY",
        "draws": 64,
        "reference_draws": 64,
        "observation_score_mode": "uncached_prefix_recompute",
        "probes": [{"prompt_id": "first"}, {"prompt_id": "second"}],
    }
    tasks = {"config": {}, "bindings": {}, "analysis_rules": {}, "inputs": {}}
    monkeypatch.setattr(preview, "_load_plan", lambda p: (plan, tasks, [], {}))
    monkeypatch.setattr(preview, "_check_runtime", lambda *a: None)
    monkeypatch.setattr(gpu, "_null_receipt", lambda *a: None)
    loads, calls = [], []
    runtime = {
        "identity": {"source_hash": "new"},
        "checkpoint_cache": SimpleNamespace(load=lambda p: {}),
        "state": SimpleNamespace(restore=lambda s: None),
    }

    def factory(*a, **kw):
        loads.append(True)
        return runtime

    fail = [True]

    def collect(config, run, forks, prompts, *, out, draws, reference_draws, bridge, resume):
        pid = prompts[0]["prompt_id"]
        calls.append(pid)
        assert forks["origin_id"] == "preview:frozen:map_41001_X_BASE_32"
        assert draws == reference_draws == 64 and bridge
        if pid == "second" and fail[0]:
            # First result is already visible while later work is still pending.
            assert (tmp_path / "prompts/00/COMPLETE.json").exists()
            assert "1/2" in (tmp_path / "FIRST_RESULTS_zh.md").read_text()
            raise RuntimeError("interrupted")
        response = {"prompt_id": pid}
        gpu._publish(Path(out) / "COMPLETE.json", response)
        return response

    monkeypatch.setattr(gpu, "collect_response_map", collect)
    monkeypatch.setattr(
        gpu,
        "pair_observation_diagnostics",
        lambda r: {"units": [], "scientific_status": "NOT_CERTIFIED"},
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        preview.run("plan", execute_gpu=True, runtime_factory=factory)
    assert not (tmp_path / "COMPLETE.json").exists()
    fail[0] = False
    result = preview.run("plan", execute_gpu=True, resume=True, runtime_factory=factory)
    assert calls == ["first", "second", "second"]
    assert result["predictive_evaluation"] == "NOT_RUN"
    assert result["automatic_successor"] is False
    assert len(loads) == 2
    preview.run("plan", execute_gpu=True, resume=True, runtime_factory=factory)
    assert len(loads) == 2  # A fully completed preview never loads the model again.
    assert canonical_hash(gpu._read(tmp_path / "COMPLETE.json")) == canonical_hash(result)


def metadata_campaign(tmp_path, monkeypatch):
    from src.modeling_v4.analysis_rules import load_analysis_rules

    old_source, new_source = {"sha256": "old"}, {"sha256": "new"}
    monkeypatch.setattr(
        preview, "source_identity", lambda root=None: old_source if root else new_source
    )
    task_id = "map_41001_X_BASE_32"
    root = tmp_path / "campaign"
    folder = root / "tasks" / task_id / "forks"
    config = {"tiny": True}
    identity = {k: "same" for k in preview.COMPATIBILITY_KEYS} | {
        "source_hash": "old",
        "config_hash": canonical_hash(config),
        "origin_id": task_id,
    }
    origin = {
        "candidate_id": "origin",
        "checkpoint": {
            "path": str(folder / "origin.pt"),
            "identity": identity | {"candidate_id": "origin"},
        },
    }
    gpu._publish(folder / "origin_policy.json", origin)
    for bank in banks():
        for name, policy in bank["policies"].items():
            cid = bank["bank_id"] + "_" + name
            policy.update(
                candidate_id=cid,
                checkpoint={
                    "path": str(folder / bank["bank_id"] / (name + ".pt")),
                    "identity": identity | {"candidate_id": cid, "bank_id": bank["bank_id"]},
                },
            )
        gpu._publish(folder / bank["bank_id"] / "COMPLETE.json", bank)
    rules = load_analysis_rules()
    tasks = {
        "config": config,
        "root": str(root),
        "source": old_source,
        "analysis_rules": rules,
        "analysis_rules_hash": canonical_hash(rules),
        "tasks": [
            {"id": task_id, "stage": "C", "kind": "map", "seed": 41001, "step": 32, "prompts": 24}
        ],
        "inputs": {
            "panels": {
                "observation": [
                    {"prompt_id": f"p{i}_{j}", "family": i, "interface": j}
                    for i in range(3)
                    for j in range(2)
                ]
            }
        },
    }
    tasks["task_list_hash"] = canonical_hash(tasks)
    path = tmp_path / "tasks.json"
    gpu._publish(path, tasks)
    return path, folder


def test_prepare_imports_metadata_without_reading_checkpoints(tmp_path, monkeypatch):
    path, _folder = metadata_campaign(tmp_path, monkeypatch)
    out = tmp_path / "preview"
    plan = preview.prepare(path, tmp_path / "old_code", out)
    parsed, _, selected, origin = preview._load_plan(out / "PLAN.json")
    assert parsed == plan
    assert len(plan["probes"]) == 6
    assert plan["draws"] == plan["reference_draws"] == 64
    assert plan["collection_source"] != plan["measurement_source"]
    assert not Path(origin["checkpoint"]["path"]).exists()  # No model reads during preparation.
    assert [b["bank_id"] for b in selected] == ["calibration_000", "calibration_001"]
    altered = plan | {"draws": 4096}
    altered["plan_hash"] = canonical_hash({k: v for k, v in altered.items() if k != "plan_hash"})
    preview.atomic_json(out / "PLAN.json", altered)
    with pytest.raises(ValueError, match="scope"):
        preview._load_plan(out / "PLAN.json")


def test_wrong_bank_checkpoint_label_rejected_before_model_load(tmp_path, monkeypatch):
    path, folder = metadata_campaign(tmp_path, monkeypatch)
    bank_path = folder / "calibration_001" / "COMPLETE.json"
    bank = gpu._read(bank_path)
    bank["policies"]["joint_1"]["checkpoint"]["identity"]["bank_id"] = "calibration_007"
    preview.atomic_json(bank_path, bank)
    out = tmp_path / "preview"
    preview.prepare(path, tmp_path / "old_code", out)
    with pytest.raises(ValueError, match="origin/bank/label"):
        preview._load_plan(out / "PLAN.json")
