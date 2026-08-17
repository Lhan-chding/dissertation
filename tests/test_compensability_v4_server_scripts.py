"""Offline tests for v4 server guards, package locks, and CLI dry-runs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts/v4"


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GUARDS = load_script("_guards", "_guards.py")
PHASE_CLI = load_script("_phase_cli", "_phase_cli.py")
CAPABILITY_SCRIPT = load_script("test_capability_chain_script", "02_run_capability_chain.py")


def valid_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model": {
            "local_path": GUARDS.MODEL_PATH,
            "snapshot_sha256": GUARDS.MODEL_SNAPSHOT_SHA256,
            "model_loading_allowed": True,
            "offline_only": True,
        },
        "authorization": {
            "training_authorized": False,
            "rl_authorized": False,
            "downloads_authorized": False,
        },
        "reporting": {
            "subjective_success_thresholds_forbidden": True,
            "report_scene_clustered_confidence_intervals": True,
            "report_paired_and_family_stratified_effects": True,
            "report_policy_support_and_reward_variance": True,
        },
        "integrity_gates": {"hash_parity": True, "token_parity": True},
        "vision_input": {"resized_height": 280, "resized_width": 280},
        "runtime_evidence": {
            "model_introspection_path": "artifacts/v4/model_introspection.json",
            "model_introspection_sha256": (
                "ed96d19a238d68497617071e29604313e0aae9a41a9e3bd24dbad451d87a0640"
            ),
            "module_manifest_path": "artifacts/v4/module_manifest.txt",
            "module_manifest_sha256": (
                "1c98fd8ba74fa5c30b8f585ffee5020544baf5be61f23e0c28c61a132973e8f0"
            ),
            "model_class": "Qwen2_5_VLForConditionalGeneration",
            "language_layers": 36,
            "vision_layers": 32,
            "module_count": 839,
            "required_modules": [
                "model.visual.blocks.0",
                "model.visual.blocks.31",
                "model.visual.merger",
                "model.language_model.layers.0",
                "model.language_model.layers.35",
                "model.language_model.norm",
                "lm_head",
            ],
        },
        "phase_1_capability_chain": {
            "source_scenes": 580,
            "model_call_cap": 3480,
            "calls_per_scene": 6,
            "t1_calls_per_scene": 1,
            "t1_yes_no_balanced": True,
            "t5_candidate_count": 4,
            "t5_true_label_balanced": True,
            "max_new_tokens": 32,
            "do_sample": False,
            "seed": 2026081701,
            "bootstrap_resamples": 10000,
        },
    }


def write_yaml(path: Path, value: object) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def test_sha256_and_execute_guard_are_deterministic(tmp_path, capsys) -> None:
    evidence = tmp_path / "evidence.bin"
    evidence.write_bytes(b"objective evidence")
    assert GUARDS.sha256(evidence) == hashlib.sha256(b"objective evidence").hexdigest()
    assert GUARDS.blocked_unless_execute(False) is True
    assert "explicit --execute" in capsys.readouterr().out
    assert GUARDS.blocked_unless_execute(True) is False


def test_load_config_accepts_only_objective_reporting_contract(tmp_path, monkeypatch) -> None:
    path = tmp_path / "config.yaml"
    write_yaml(path, valid_config())
    monkeypatch.setattr(GUARDS, "CONFIG_PATH", path)

    loaded = GUARDS._load_config(path)
    assert loaded["authorization"]["training_authorized"] is False
    assert "minimum_visual_repair_rate" not in loaded["reporting"]

    subjective = valid_config()
    subjective["reporting"]["minimum_visual_repair_rate"] = 0.8
    write_yaml(path, subjective)
    with pytest.raises(RuntimeError, match="subjective empirical"):
        GUARDS._load_config(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(schema_version=2), "schema"),
        (lambda value: value.pop("reporting"), "sections"),
        (lambda value: value["model"].update(local_path="/wrong"), "model pin"),
        (lambda value: value["model"].update(offline_only=False), "offline contract"),
        (lambda value: value["authorization"].update(training_authorized=True), "training"),
        (lambda value: value["authorization"].update(rl_authorized=True), "RL"),
        (lambda value: value["authorization"].update(downloads_authorized=True), "offline"),
        (
            lambda value: value["reporting"].update(
                report_scene_clustered_confidence_intervals=False
            ),
            "reporting contract",
        ),
        (lambda value: value["vision_input"].update(resized_height=224), "280x280"),
        (lambda value: value["integrity_gates"].update(hash_parity=False), "integrity gates"),
        (
            lambda value: value["runtime_evidence"].update(language_layers=35),
            "runtime evidence",
        ),
        (
            lambda value: value["runtime_evidence"].update(unreviewed_field=True),
            "runtime evidence",
        ),
        (
            lambda value: value["phase_1_capability_chain"].update(model_call_cap=3479),
            "Phase 1",
        ),
    ],
)
def test_load_config_fails_closed_on_contract_drift(tmp_path, monkeypatch, mutate, message) -> None:
    path = tmp_path / "config.yaml"
    payload = valid_config()
    mutate(payload)
    write_yaml(path, payload)
    monkeypatch.setattr(GUARDS, "CONFIG_PATH", path)
    with pytest.raises(RuntimeError, match=message):
        GUARDS._load_config(path)


def test_canonical_file_rejects_aliases_symlinks_and_missing_files(tmp_path) -> None:
    expected = tmp_path / "expected"
    expected.write_text("x", encoding="utf-8")
    GUARDS._canonical_file(expected, expected, "input")
    alias = tmp_path / "alias"
    alias.symlink_to(expected)
    with pytest.raises(RuntimeError, match="canonical"):
        GUARDS._canonical_file(alias, expected, "input")
    with pytest.raises(RuntimeError, match="canonical"):
        GUARDS._canonical_file(tmp_path / "missing", expected, "input")


def test_package_lock_validates_each_file_hash_and_its_own_digest(tmp_path, monkeypatch) -> None:
    relative_paths = (
        "configs/recoverability/v4_phase_0_3.yaml",
        "configs/recoverability/v4/phase_1_3_prompts.yaml",
        "docs/Qwen25VL_Constraint_Assimilation_Natural_State_RL_Codex_Plan_v4.md",
        "docs/QWEN_V4_SERVER_HANDOFF.md",
        "pyproject.toml",
        "requirements-gpu.lock.txt",
    )
    rows = []
    for relative in relative_paths:
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(relative, encoding="utf-8")
        rows.append({"path": relative, "sha256": GUARDS.sha256(source)})
    lock = tmp_path / "lock.yaml"
    write_yaml(lock, {"files": rows})
    monkeypatch.setattr(GUARDS, "ROOT", tmp_path)
    monkeypatch.setattr(GUARDS, "PACKAGE_LOCK_PATH", lock)
    assert GUARDS._verify_package_lock(lock) == GUARDS.sha256(lock)

    (tmp_path / relative_paths[0]).write_text("drifted", encoding="utf-8")
    with pytest.raises(RuntimeError, match="package lock mismatch"):
        GUARDS._verify_package_lock(lock)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "empty or malformed"),
        ([{"path": "a"}], "row is malformed"),
        ([{"path": 1, "sha256": "0" * 64}], "invalid fields"),
        (
            [
                {"path": "a", "sha256": "0" * 64},
                {"path": "a", "sha256": "1" * 64},
            ],
            "duplicate",
        ),
    ],
)
def test_package_lock_rejects_malformed_rows(tmp_path, monkeypatch, rows, message) -> None:
    lock = tmp_path / "lock.yaml"
    write_yaml(lock, {"files": rows})
    monkeypatch.setattr(GUARDS, "ROOT", tmp_path)
    monkeypatch.setattr(GUARDS, "PACKAGE_LOCK_PATH", lock)
    if message == "duplicate":
        (tmp_path / "a").write_text("content", encoding="utf-8")
        monkeypatch.setattr(GUARDS, "sha256", lambda _path: "0" * 64)
    with pytest.raises(RuntimeError, match=message):
        GUARDS._verify_package_lock(lock)


def test_raw_input_validation_hash_binds_json_and_jsonl(tmp_path) -> None:
    json_file = tmp_path / "rows.json"
    json_file.write_text('{"value": 1}', encoding="utf-8")
    digest = GUARDS.sha256(json_file)
    assert GUARDS._validate_raw_input(json_file, digest)["sha256"] == digest

    jsonl_file = tmp_path / "rows.jsonl"
    jsonl_file.write_text('{"value": 1}\n\n{"value": 2}\n', encoding="utf-8")
    assert GUARDS._validate_raw_input(jsonl_file, GUARDS.sha256(jsonl_file))["path"] == str(
        jsonl_file.resolve()
    )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        GUARDS._validate_raw_input(json_file, "0" * 64)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no records"):
        GUARDS._validate_raw_input(empty, GUARDS.sha256(empty))
    with pytest.raises(RuntimeError, match="non-empty regular"):
        GUARDS._validate_raw_input(tmp_path / "missing", "0" * 64)


def test_runtime_evidence_requires_exact_s1_artifacts(tmp_path, monkeypatch) -> None:
    introspection = tmp_path / "model_introspection.json"
    manifest = tmp_path / "module_manifest.txt"
    modules = [
        "",
        "model.visual.blocks.0",
        "model.visual.blocks.31",
        "model.visual.merger",
        "model.language_model.layers.0",
        "model.language_model.layers.35",
        "model.language_model.norm",
        "lm_head",
    ]
    introspection.write_text(
        json.dumps(
            {
                "model_class": "Qwen2_5_VLForConditionalGeneration",
                "language_layers": 36,
                "module_count": len(modules),
                "vision_config": {"depth": 32},
            }
        ),
        encoding="utf-8",
    )
    manifest.write_text("\n".join(modules) + "\n", encoding="utf-8")
    runtime = {
        **valid_config()["runtime_evidence"],
        "model_introspection_sha256": GUARDS.sha256(introspection),
        "module_manifest_sha256": GUARDS.sha256(manifest),
        "module_count": len(modules),
    }
    monkeypatch.setattr(GUARDS, "MODEL_INTROSPECTION_PATH", introspection)
    monkeypatch.setattr(GUARDS, "MODULE_MANIFEST_PATH", manifest)

    validated = GUARDS.validate_runtime_evidence(runtime)

    assert validated["language_layers"] == 36
    manifest.write_text("lm_head\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256"):
        GUARDS.validate_runtime_evidence(runtime)


def test_validate_server_inputs_runs_all_gates_before_return(monkeypatch, tmp_path) -> None:
    source = tmp_path / "input.json"
    source.write_text("{}", encoding="utf-8")
    calls: list[object] = []
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(GUARDS, "_load_config", lambda path: calls.append(("config", path)))
    monkeypatch.setattr(GUARDS, "_verify_package_lock", lambda path: "lock-hash")
    monkeypatch.setattr(GUARDS, "sha256", lambda path: "config-hash")
    monkeypatch.setattr(
        GUARDS, "require_server_model", lambda path, expected: calls.append((path, expected))
    )
    monkeypatch.setattr(
        GUARDS,
        "_validate_raw_input",
        lambda path, digest: {"path": str(path), "sha256": digest},
    )

    result = GUARDS.validate_server_inputs(
        config=tmp_path / "config",
        package_lock=tmp_path / "lock",
        model_path=tmp_path / "model",
        inputs=(source,),
        input_sha256=("source-hash",),
        expected_input_sha256=("source-hash",),
        require_raw_evidence=True,
    )
    assert result.config_sha256 == "config-hash"
    assert result.package_lock_sha256 == "lock-hash"
    assert result.inputs == ({"path": str(source), "sha256": "source-hash"},)
    assert calls[-1][0] == tmp_path / "model"


def test_validate_server_inputs_rejects_offline_or_evidence_contract_breaks(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(GUARDS, "_load_config", lambda _path: {})
    monkeypatch.setattr(GUARDS, "_verify_package_lock", lambda _path: "lock")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    kwargs = {
        "config": tmp_path / "config",
        "package_lock": tmp_path / "lock",
        "model_path": tmp_path / "model",
        "inputs": (),
        "input_sha256": (),
        "expected_input_sha256": (),
        "require_raw_evidence": False,
    }
    with pytest.raises(RuntimeError, match="HF_HUB_OFFLINE"):
        GUARDS.validate_server_inputs(**kwargs)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="matching"):
        GUARDS.validate_server_inputs(**{**kwargs, "inputs": (tmp_path / "a",)})
    with pytest.raises(RuntimeError, match="requires hash-bound"):
        GUARDS.validate_server_inputs(**{**kwargs, "require_raw_evidence": True})


def test_validate_server_inputs_rejects_caller_selected_evidence_hash(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "input.jsonl"
    source.write_text('{"scene_id":"replacement"}\n', encoding="utf-8")
    replacement_hash = GUARDS.sha256(source)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(GUARDS, "_load_config", lambda _path: {})
    monkeypatch.setattr(GUARDS, "_verify_package_lock", lambda _path: "lock")
    with pytest.raises(RuntimeError, match="frozen evidence SHA-256"):
        GUARDS.validate_server_inputs(
            config=tmp_path / "config",
            package_lock=tmp_path / "lock",
            model_path=tmp_path / "model",
            inputs=(source,),
            input_sha256=(replacement_hash,),
            expected_input_sha256=(GUARDS.LEGACY_SCREEN_RECORDS_SHA256,),
            require_raw_evidence=True,
        )


def test_execution_manifest_is_no_overwrite_and_records_no_subjective_threshold(tmp_path) -> None:
    validation = GUARDS.ValidatedServerInputs("config", "lock", "model", ())
    output = tmp_path / "nested/manifest.json"
    GUARDS.write_execution_manifest(
        output,
        phase="phase_2",
        validation=validation,
        intended_artifacts=("artifact.jsonl",),
        integrity_gates=("exact_logit_parity",),
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "PREWORK_MANIFEST_ONLY_PHASE_NOT_EXECUTED"
    assert payload["phase_specific_gates_executed"] is False
    assert payload["training_invoked"] is False
    assert payload["rl_invoked"] is False
    assert payload["subjective_success_threshold_applied"] is False
    with pytest.raises(FileExistsError, match="overwrite"):
        GUARDS.write_execution_manifest(
            output,
            phase="phase_2",
            validation=validation,
            intended_artifacts=(),
            integrity_gates=(),
        )


def test_phase_cli_is_inert_without_execute(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["phase"])
    assert (
        PHASE_CLI.run_phase_preflight(
            phase="phase",
            description="description",
            default_output_name="phase.json",
            intended_artifacts=(),
            integrity_gates=(),
            expected_input_sha256=(GUARDS.LEGACY_SCREEN_RECORDS_SHA256,),
        )
        == 2
    )
    assert "BLOCKED" in capsys.readouterr().out


def test_phase_cli_execute_success_and_failure_are_reported(monkeypatch, tmp_path, capsys) -> None:
    validation = GUARDS.ValidatedServerInputs("config", "lock", "model", ())
    output = tmp_path / "manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["phase", "--execute", "--output", str(output), "--input", "raw", "--input-sha256", "abc"],
    )
    monkeypatch.setattr(PHASE_CLI, "validate_server_inputs", lambda **_kwargs: validation)
    written: dict[str, object] = {}
    monkeypatch.setattr(
        PHASE_CLI,
        "write_execution_manifest",
        lambda path, **kwargs: written.update(path=path, **kwargs),
    )
    assert (
        PHASE_CLI.run_phase_preflight(
            phase="phase",
            description="d",
            default_output_name="x",
            intended_artifacts=("a",),
            integrity_gates=("g",),
            expected_input_sha256=(GUARDS.LEGACY_SCREEN_RECORDS_SHA256,),
        )
        == 0
    )
    assert written["path"] == output
    success_output = capsys.readouterr().out
    assert "PREPARED" in success_output
    assert "phase not executed" in success_output

    monkeypatch.setattr(
        PHASE_CLI,
        "validate_server_inputs",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("hash drift")),
    )
    assert (
        PHASE_CLI.run_phase_preflight(
            phase="phase",
            description="d",
            default_output_name="x",
            intended_artifacts=(),
            integrity_gates=(),
            expected_input_sha256=(GUARDS.LEGACY_SCREEN_RECORDS_SHA256,),
        )
        == 2
    )
    assert "hash drift" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("filename", "phase"),
    [
        ("03_score_candidates.py", "phase_2_candidate_scoring"),
        ("04_layerwise_assimilation.py", "phase_2_layerwise_assimilation"),
        ("05_validate_cache_runner.py", "phase_3_cache_parity"),
        ("06_run_interface_ladder.py", "phase_3_interface_ladder"),
    ],
)
def test_phase_entrypoints_delegate_only_to_preflight(monkeypatch, filename, phase) -> None:
    module = load_script("test_" + filename.replace(".py", ""), filename)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        module, "run_phase_preflight", lambda **kwargs: captured.update(kwargs) or 7
    )
    assert module.main() == 7
    assert captured["phase"] == phase
    assert captured["integrity_gates"]
    assert captured["expected_input_sha256"] == (GUARDS.LEGACY_SCREEN_RECORDS_SHA256,)
    assert all("percent" not in gate and "rate" not in gate for gate in captured["integrity_gates"])


def test_capability_chain_entrypoint_uses_runtime_execution(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        CAPABILITY_SCRIPT,
        "run_capability_chain_cli",
        lambda **kwargs: captured.update(kwargs) or 11,
    )

    assert CAPABILITY_SCRIPT.main() == 11
    assert captured["phase"] == "phase_1_capability_chain"
    assert captured["expected_input_sha256"] == (GUARDS.LEGACY_SCREEN_RECORDS_SHA256,)
    assert captured["output_paths"]["per_scene"].endswith(
        "artifacts/v4/capability_chain/per_scene.csv"
    )


def test_capability_chain_cli_executes_frozen_contract(monkeypatch, tmp_path, capsys) -> None:
    source = tmp_path / "screen_records.jsonl"
    source.write_text('{"eligible":true}\n', encoding="utf-8")
    config = valid_config()
    validation = GUARDS.ValidatedServerInputs("config", "lock", "model", ({"sha256": "raw"},))
    output_dir = tmp_path / "capability"
    outputs = {
        "per_scene": str(output_dir / "per_scene.csv"),
        "summary_by_family": str(output_dir / "summary_by_family.csv"),
        "paired_gaps": str(output_dir / "paired_gaps.json"),
    }
    scenes = tuple(object() for _ in range(580))
    calls = tuple(type("Call", (), {"call_id": str(index)})() for index in range(3480))
    records = tuple(object() for _ in range(3480))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "02_run_capability_chain.py",
            "--execute",
            "--input",
            str(source),
            "--input-sha256",
            GUARDS.LEGACY_SCREEN_RECORDS_SHA256,
        ],
    )
    monkeypatch.setattr(CAPABILITY_SCRIPT, "validate_server_inputs", lambda **_kwargs: validation)
    monkeypatch.setattr(CAPABILITY_SCRIPT, "_load_config", lambda _path: config)
    monkeypatch.setattr(CAPABILITY_SCRIPT, "validate_runtime_evidence", lambda _runtime: {})
    monkeypatch.setattr(
        CAPABILITY_SCRIPT,
        "_load_prompt_contract",
        lambda _path: (
            {name: name for name in ("T1", "T2", "T3", "T4", "T5", "T6")},
            ("A", "B", "C", "D"),
        ),
    )
    monkeypatch.setattr(CAPABILITY_SCRIPT, "load_legacy_capability_scenes", lambda _path: scenes)
    monkeypatch.setattr(
        CAPABILITY_SCRIPT, "load_pinned_qwen", lambda **_kwargs: (object(), object())
    )
    monkeypatch.setattr(
        CAPABILITY_SCRIPT,
        "find_single_token_labels",
        lambda _tokenizer, _order, minimum: ("A", "B", "C", "D"),
    )
    monkeypatch.setattr(
        CAPABILITY_SCRIPT, "build_capability_calls", lambda *_args, **_kwargs: calls
    )
    monkeypatch.setattr(
        CAPABILITY_SCRIPT,
        "execute_capability_calls",
        lambda *_args, progress, **_kwargs: (progress(3480, 3480), records)[1],
    )
    monkeypatch.setattr(
        CAPABILITY_SCRIPT,
        "summarize_capability_run",
        lambda *_args, **_kwargs: (({"family": "x", "task_type": "T1"},), {"G_loc": {}}),
    )
    written = {}
    monkeypatch.setattr(
        CAPABILITY_SCRIPT,
        "write_capability_outputs",
        lambda path, **kwargs: written.update(path=path, **kwargs),
    )

    result = CAPABILITY_SCRIPT.run_capability_chain_cli(
        phase="phase_1_capability_chain",
        expected_input_sha256=(GUARDS.LEGACY_SCREEN_RECORDS_SHA256,),
        output_paths=outputs,
    )

    assert result == 0
    assert written["path"] == output_dir
    assert written["gaps"]["subjective_success_threshold_applied"] is False
    assert written["gaps"]["model_calls"] == 3480
    assert "PROGRESS: 3480/3480" in capsys.readouterr().out


def test_capability_chain_cli_is_inert_without_execute(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["02_run_capability_chain.py"])
    assert (
        CAPABILITY_SCRIPT.run_capability_chain_cli(
            phase="phase_1_capability_chain",
            expected_input_sha256=(GUARDS.LEGACY_SCREEN_RECORDS_SHA256,),
            output_paths={
                "per_scene": "per_scene.csv",
                "summary_by_family": "summary_by_family.csv",
                "paired_gaps": "paired_gaps.json",
            },
        )
        == 2
    )
    assert "BLOCKED" in capsys.readouterr().out


def test_introspection_cli_dry_run_never_loads_model(monkeypatch, tmp_path, capsys) -> None:
    module = load_script("test_introspect_qwen_script", "01_introspect_qwen.py")
    monkeypatch.setattr(sys, "argv", ["01_introspect_qwen.py", "--artifact-root", str(tmp_path)])
    monkeypatch.setattr(
        module, "validate_server_inputs", lambda **_kwargs: pytest.fail("must stay inert")
    )
    assert module.main() == 2
    assert "BLOCKED" in capsys.readouterr().out


def test_introspection_cli_rejects_redirected_artifact_root(monkeypatch, tmp_path, capsys) -> None:
    module = load_script("test_introspect_qwen_redirect", "01_introspect_qwen.py")
    monkeypatch.setattr(
        sys,
        "argv",
        ["01_introspect_qwen.py", "--execute", "--artifact-root", str(tmp_path)],
    )
    monkeypatch.setattr(
        module,
        "validate_server_inputs",
        lambda **_kwargs: pytest.fail("redirected path must fail before server validation"),
    )

    assert module.main() == 2
    assert "canonical" in capsys.readouterr().out
