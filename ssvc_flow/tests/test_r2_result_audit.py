"""Postflight reclassification must reject internally rehashed false evidence."""

from types import SimpleNamespace

import pytest

from src.core import canonical_hash
from src.r2_runtime import annotate_diagnostic


def fixture_row():
    scene = {
        "base_scene_id": "s",
        "constraint_family": "duplicate_encoding",
        "truth_world": [1, 2, 3, 4],
        "observed_world": [9, 2, 3, 4],
        "changed_index": 0,
        "operation": "sum4",
        "chart_type": "line",
        "image_hash": "image",
        "cue": {"family": "duplicate_encoding", "known_index": 0, "known_value": 1},
    }
    config = {"protocol_version": "test", "generation_proposed_N": {}, "data_root": "/data"}
    runtime = {"config": config, "identity": {}, "initial_adapter_hash": "adapter"}
    runtime["model_audit"] = {
        "model_id": "model",
        "model_revision": "revision",
        "eos_token_ids": [9],
    }
    audit = {"image_token_count": 0, "enable_thinking": False, "final_prompt_hash": "input"}
    row = {
        "base_scene_id": "s",
        "prompt_id": "p",
        "condition": "SYM_ORIGINAL",
        "enable_thinking": False,
        "max_new_tokens": 64,
        "decode_mode": "sample",
        "run_id": canonical_hash({}),
        "protocol_version": "test",
        "model_id": "model",
        "model_revision": "revision",
        "adapter_hash": "adapter",
        "checkpoint_step": 0,
        "optimizer_state_hash": None,
        "train_seed": 17,
        "family": "duplicate_encoding",
        "constraint_family": "duplicate_encoding",
        "interface": "SYMBOLIC_FRESH",
        "split": "calibration",
        "solution_count": 1,
        **{
            key: scene[key]
            for key in (
                "truth_world",
                "observed_world",
                "changed_index",
                "operation",
                "chart_type",
                "cue",
            )
        },
        "image_hash": None,
        "input_ids_hash": None,
        "actual_image_tokens": 0,
        **audit,
        "raw_completion": "[1,2,3,4]",
        "raw_text": "[1,2,3,4]",
        "token_ids": [1, 9],
        "raw_token_ids": [1, 9],
        "completion_length": 2,
        "n_generated_tokens": 2,
        "stop_reason": "eos",
        "behavior_token_logprobs": [-1.0, -0.2],
        "per_token_logprob_behavior": [-1.0, -0.2],
        "logprob_sequence": -1.2,
        "thinking_segments": None,
        "execution_checks": {"passed": True, "faults": []},
        "generation_config_hash": canonical_hash(
            {"max_new_tokens": 64, "enable_thinking": False, "do_sample": True}
        ),
        **annotate_diagnostic("[1,2,3,4]", scene),
    }
    adapter = SimpleNamespace(
        processor=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids, **kw: "[1,2,3,4]"))
    )
    return row, scene, runtime, audit, adapter


def test_recompute_annotations_and_prepared_inputs_without_model(monkeypatch):
    from src import r2_result_audit as module

    row, scene, runtime, audit, adapter = fixture_row()
    monkeypatch.setattr(module, "variant_prompt", lambda *a: {})
    monkeypatch.setattr(module, "prepare_r2", lambda *a, **kw: {"audit": audit})
    result = module.validate_raw_rows([row], [scene], runtime, adapter)
    assert result == {"rows": 1, "prepared_prompts": 1, "thinking_rows": 0, "image_rows": 0}
    for change in (
        {"category": "W"},
        {"raw_text": "changed"},
        {"truth_world": [9, 2, 3, 4]},
        {"final_prompt_hash": "changed"},
        {"logprob_sequence": 0.0},
    ):
        with pytest.raises(ValueError):
            module.validate_raw_rows([{**row, **change}], [scene], runtime, adapter)


def test_processor_reconstruction_uses_only_pinned_cached_config_and_processor(monkeypatch):
    import sys

    from src import r2_result_audit as module
    from src.model_adapters.base import _hash_json

    ip = SimpleNamespace(size={"longest_edge": 1})
    processor = SimpleNamespace(
        image_processor=ip,
        tokenizer=SimpleNamespace(get_vocab=lambda: {"a": 1}),
        chat_template="template",
        to_dict=lambda: {"size": dict(ip.size)},
    )
    calls = []

    def cached_processor(name, **options):
        calls.append(("processor", name, options))
        return processor

    def cached_config(name, **options):
        calls.append(("config", name, options))
        return SimpleNamespace(image_token_id=200)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=cached_processor),
            AutoConfig=SimpleNamespace(from_pretrained=cached_config),
        ),
    )
    audit = {
        "model_id": "Qwen/Qwen3.5-9B",
        "model_revision": "a" * 40,
        "processor_max_pixels": 786432,
        "requested_image_token_limit": 768,
        "processor_hash": _hash_json({"size": {"longest_edge": 786432}}),
        "tokenizer_hash": _hash_json({"a": 1}),
        "chat_template_hash": _hash_json("template"),
    }
    runtime = {
        "model_audit": audit,
        "config": {"model": {"id": audit["model_id"], "revision": audit["model_revision"]}},
    }
    adapter = module.load_processor_adapter(runtime)
    assert adapter.device == "cpu" and not hasattr(adapter.model, "parameters")
    assert adapter.processor is processor and ip.size["longest_edge"] == 786432
    assert {call[0] for call in calls} == {"processor", "config"}
    assert all(
        call[2] == {"revision": "a" * 40, "trust_remote_code": False, "local_files_only": True}
        for call in calls
    )
    with pytest.raises(ValueError, match="processor/tokenizer/template"):
        module.load_processor_adapter(
            {**runtime, "model_audit": {**audit, "tokenizer_hash": "changed"}}
        )


def test_report_recomputation_rejects_changed_summary_without_rewriting_source(
    tmp_path, monkeypatch
):
    import json

    from test_r2_runtime import analysis_rows

    from src import next_stage_runtime
    from src import r2_result_audit as module
    from src.core import file_hash, write_json

    root = tmp_path / "original"
    root.mkdir()
    rows = [
        {**row, "n_generated_tokens": 2, "execution_kind": "CPU_FAKE_ADAPTER_FIXTURE"}
        for row in analysis_rows()
    ]
    (root / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(root / "runtime_lock.json", {"identity": {}})
    write_json(root / "request_manifest.json", {"requests": []})
    write_json(root / "data_binding.json", {})

    def panel_files(panel, directory):
        write_json(directory / "panel_manifest.json", {"fixture": True})
        (directory / "prompt_diffs.md").write_text("fixed template fixture\n")

    panel_files([], root)
    module._write_analysis(root, rows)
    environment = {"fixture": True}

    def verified_binding(*args):
        return {
            "environment": environment,
            "r2_files": {p.name: file_hash(p) for p in root.iterdir() if p.is_file()},
        }

    monkeypatch.setattr(next_stage_runtime, "validate_r2_gate", verified_binding)
    monkeypatch.setattr(next_stage_runtime, "validate_runtime_environment", lambda *a: environment)
    monkeypatch.setattr(module, "_load_panel", lambda *a: ([], {}))
    monkeypatch.setattr(module, "build_requests", lambda *a: [])
    monkeypatch.setattr(module, "request_ledger", lambda *a: [])
    monkeypatch.setattr(module, "load_processor_adapter", lambda *a: None)
    monkeypatch.setattr(module, "validate_processor_certificate", lambda *a: None, raising=False)
    monkeypatch.setattr(module, "validate_raw_rows", lambda *a: {"rows": len(rows)})
    monkeypatch.setattr(module, "write_panel_artifacts", panel_files)
    gate = {"config": {"data_root": "/fixture"}}
    before = verified_binding()["r2_files"]
    result = module.audit_r2_results(root, gate, tmp_path / "accepted")
    assert result["status"] == "PASS" and result["new_model_generations"] == 0
    assert verified_binding()["r2_files"] == before
    assert all(
        v["original_sha256"] == v["recomputed_sha256"]
        for v in result["report_hash_checks"].values()
    )
    (root / "condition_metrics.csv").write_text("forged summary\n")
    with pytest.raises(ValueError, match="derived report"):
        module.audit_r2_results(root, gate, tmp_path / "rejected")
    failed = json.loads((tmp_path / "rejected/audit.json").read_text())
    assert failed["status"] == "FAIL"
    assert (root / "condition_metrics.csv").read_text() == "forged summary\n"
    with pytest.raises(ValueError, match="new directory"):
        module.audit_r2_results(root, gate, tmp_path / "accepted")


def test_processor_and_unused_EOS_drift_cannot_be_accepted_by_self_reported_R2_audit():
    from src import r2_result_audit as module

    audit = {
        "processor_hash": "processor",
        "tokenizer_hash": "tokenizer",
        "chat_template_hash": "template",
        "processor_max_pixels": 786432,
        "processor_patch_size": 16,
        "processor_merge_size": 2,
        "requested_image_token_limit": 768,
        "eos_token_ids": [9],
    }
    gate = {"certificate": {"model_audit": audit}}
    assert module.validate_processor_certificate({"model_audit": dict(audit)}, gate) is None
    for change in (
        {"eos_token_ids": [9, 12345]},
        {"processor_hash": "rehashed"},
        {"processor_max_pixels": 999},
    ):
        with pytest.raises(ValueError, match="R1 processor/EOS"):
            module.validate_processor_certificate({"model_audit": {**audit, **change}}, gate)
