"""Real tiny Torch orchestration fixtures never certify CUDA measurements."""

from src.core import canonical_hash
from src.r4_runtime import build_requests, run_r4


def test_arm_step_policy_and_source_bind_every_sample():
    prompts = [
        {
            "prompt_id": "p",
            "base_scene_id": "b",
            "family": "duplicate",
            "interface": "SYMBOLIC_FRESH",
            "split": "train",
            "track": "N",
            "evaluation_domain": "ID",
            "prompt_hash": "h",
            "max_new_tokens": 64,
            "enable_thinking": False,
            "scene": {"x": 1},
        }
    ]
    identity = {"phase": "R4", "model_hash": "m", "data_hash": "d", "config_hash": "c"}
    args = (prompts, identity, "X_BASE", 3, "train", "state-a")
    rows = build_requests(*args)
    assert rows == build_requests(*args)
    assert len(rows) == len({r["sample_key"] for r in rows}) == 8
    other = build_requests(prompts, identity, "X_VALID", 3, "train", "state-a")
    assert rows[0]["sample_key"] != other[0]["sample_key"]
    assert rows[0]["sample_seed"] == other[0]["sample_seed"]
    changed = build_requests(prompts, identity, "X_BASE", 3, "train", "state-b")
    assert rows[0]["sample_key"] != changed[0]["sample_key"]
    assert rows[0]["sample_seed"] != changed[0]["sample_seed"]
    changed = build_requests(
        prompts, {**identity, "model_hash": "new"}, "X_BASE", 3, "train", "state-a"
    )
    assert rows[0]["sample_seed"] != changed[0]["sample_seed"]
    assert all(r["run_id"] == canonical_hash(identity) for r in rows)


def test_authorization_is_flag_and_does_not_load_model(tmp_path):
    result = run_r4(
        {"R4": {"allow_training": False}},
        tmp_path,
        tmp_path / "out",
        "r0",
        "r1",
        "supp",
        "r2",
        "r3",
    )
    assert result["status"] == "BLOCKED"
    assert result["details"]["training_started"] is False
    assert "# R4 BLOCKED" in (tmp_path / "out/report_zh.md").read_text()


def _fixture(tmp_path, monkeypatch):
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    from test_model_smoke import FakeAdapter
    from test_r1_reference_smoke import setup

    from src import r4_runtime
    from src.core import file_hash, write_json
    from src.legacy_frozen import REPOSITORY, SOURCES, load_legacy_scenes
    from src.r3_inputs import build_r3_plan

    config, _, initial, certificate = setup(tmp_path)
    root = Path("data/generated").resolve()
    config["data_root"] = str(root)
    manifest = json.loads((root / "manifest.json").read_text())
    gate = {
        "certificate": certificate,
        "config": config,
        "binding": {
            "data_manifest_sha256": file_hash(root / "manifest.json"),
            "dataset_files": manifest["files"],
        },
    }
    monkeypatch.setitem(
        sys.modules,
        "src.next_stage_runtime",
        SimpleNamespace(
            validate_prerequisites=lambda *a: gate,
            validate_config_against_gate=lambda c, g: c,
            validate_r2_gate=lambda *a: {"status": "PASS"},
            validate_r3_cold_gate=lambda *a: {"status": "PASS"},
            validate_runtime_environment=lambda *a: (_ for _ in ()).throw(
                AssertionError("not real CUDA")
            ),
        ),
    )
    monkeypatch.setattr(r4_runtime, "_source", lambda: {"source_commit": "CPU-FIXTURE"})
    r0 = tmp_path / "r0"
    legacy_root = REPOSITORY / "artifacts/v5/study_c2/data"
    legacy = load_legacy_scenes(
        legacy_root / "reward_fibers.jsonl", legacy_root / "reward_fibers_manifest.json"
    )
    write_json(
        r0 / "baseline_lock_L.json",
        {
            "actual_max_new_tokens": 48,
            "generation": {
                "do_sample": True,
                "max_new_tokens": 48,
                "min_p": 0.0,
                "num_beams": 1,
                "repetition_penalty": 1.0,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
            },
            "data_hash": canonical_hash(legacy),
            "source_files": {
                "legacy/" + relative: file_hash(REPOSITORY / relative)
                for relative in SOURCES.values()
            },
        },
    )
    write_json(
        r0 / "closure_checks.json",
        {
            "cross_split": {
                "status": "PASS",
                "manifest_sha256": file_hash(root / "manifest.json"),
                "manifest_global_truth_uniqueness": True,
                "base_scene_collisions": {},
                "truth_structure_collisions": {},
                "splits": {
                    name[:-6]: {"sha256": info["sha256"]}
                    for name, info in manifest["files"].items()
                    if name.endswith(".jsonl")
                },
            }
        },
    )
    instances = []

    class Tokenizer:
        def decode(self, ids, **kwargs):
            return "".join({1: "[1,2,3,4]", 3: "invalid", 2: "<eos>"}[i] for i in ids)

    class Adapter(FakeAdapter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.audit = {
                **initial.audit,
                "load_seconds": len(instances) + 0.123,
                "load_peak_cuda_bytes": len(instances),
            }
            self.processor = SimpleNamespace(tokenizer=Tokenizer())
            self.eos_ids = {2}
            instances.append(self)

        def prepare(self, prompt, root):
            prepared = super().prepare(prompt, root)
            prepared["audit"].update(
                {
                    "enable_thinking": False,
                    "input_tensor_hash": canonical_hash(prompt),
                    "tokenized_prompt_hash": canonical_hash(prompt["user"]),
                }
            )
            return prepared

        def generate(self, prepared, *, seed, max_new_tokens, do_sample):
            returned = super().generate(prepared, seed=seed, max_new_tokens=max_new_tokens)
            returned["vision_forward_calls"] = int(bool(prepared["image"]))
            self.forward_calls += len(returned["token_ids"])
            return returned

    def probe(adapter, optimizer, origin, *args):
        def load(split):
            return [json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines()]

        prompts = build_r3_plan(load("train"), load("control"))["control_prompts"]
        rows = []
        for index, prompt in enumerate(prompts):
            audit = adapter.prepare(prompt["prompt"], root)["audit"]
            row = {
                "sample_key": f"source-r3-{index}",
                "prompt_id": prompt["prompt_id"],
                "base_scene_id": prompt["base_scene_id"],
                "family": prompt["family"],
                "interface": prompt["interface"],
                "split": "control",
                "token_ids": [1, 2],
                "old_logprobs": [-1.3862943649291992] * 2,
                "stop_reason": "eos",
                **audit,
            }
            row["record_hash"] = canonical_hash(row)
            rows.append(row)
        return (
            rows,
            {p["prompt_id"]: p for p in prompts},
            {"fixture": True, "new_control_generations": 0},
        )

    monkeypatch.setattr(r4_runtime, "_control_probe", probe)
    return (config, root, tmp_path / "pilot", r0, "r1", "supp", "r2", "r3"), Adapter, instances


def test_full_toy_two_arms_interruption_resume_and_original_L_tokens(tmp_path, monkeypatch):
    import json

    from src import grpo_update
    from src.core import file_hash
    from src.optimizer_fork import load_checkpoint, state_hash

    args, adapter, instances = _fixture(tmp_path, monkeypatch)
    original = grpo_update.perform_update
    interruptions = []

    def interrupt_after_real_update(*args, **kwargs):
        result = original(*args, **kwargs)
        if not interruptions:
            interruptions.append(result)
            raise RuntimeError("interrupted after actual Adam update")
        return result

    monkeypatch.setattr(grpo_update, "perform_update", interrupt_after_real_update)
    result = run_r4(*args, allow_training=True, _adapter_factory=adapter)
    assert result["status"] == "FAIL", result["details"]
    assert "interrupted after actual Adam" in result["details"]["error"]["message"]
    assert "# R4 FAIL" in (args[2] / "report_zh.md").read_text()
    out = args[2]
    prefix = (out / "X_BASE/step_01/rollouts/samples.jsonl").read_bytes()
    result = run_r4(*args, allow_training=True, resume=True, _adapter_factory=adapter)
    assert result["status"] == "PASS", result["details"]
    assert result["execution_kind"] == "CPU_FAKE_ADAPTER_FIXTURE"
    assert result["details"]["new_outputs"] == 14400
    assert result["details"]["runtime_counts"]["optimizer_steps_observed_at_least"] == 129
    assert result["details"]["runtime_counts"]["backward_calls_observed_at_least"] == 129 * 32
    assert (out / "X_BASE/step_01/rollouts/samples.jsonl").read_bytes() == prefix
    assert (out / "X_BASE/step_01/updates/attempt_0000/failure.json").exists()
    groups = json.loads(
        (out / "X_BASE/step_01/updates/attempt_0001/group_statistics.json").read_text()
    )
    assert len(groups) == 4
    assert all(
        g["K"] == 8
        and len(g["reward_vector"]) == len(g["advantages"]) == len(g["sample_keys"]) == 8
        for g in groups
    )
    assert all(g["zero_variance_flag"] and not any(g["advantages"]) for g in groups)
    samples = []
    for path in out.rglob("samples.jsonl"):
        samples.extend(json.loads(line) for line in path.read_text().splitlines())
    raw = [r for r in samples if "raw_text" in r]
    assert len(raw) == 14400
    assert len({r["sample_key"] for r in raw}) == 14400
    legacy = [r for r in raw if r["track"] == "L"]
    assert len(legacy) == 2816
    assert {r["max_new_tokens"] for r in legacy} == {48}
    n_hash = next(r["generation_config_hash"] for r in raw if r["track"] == "N")
    assert all(r["generation_config_hash"] != n_hash for r in legacy)
    checkpoint = json.loads((out / "checkpoint_manifest.json").read_text())
    assert len(checkpoint["checkpoints"]) == 130
    assert sum(c["milestone"] for c in checkpoint["checkpoints"]) == 8
    warm = checkpoint["X_BASE_step64_for_R3_warm"]
    assert warm["step"] == 64
    warm_state = load_checkpoint(warm["checkpoint_path"], warm["checkpoint_identity"])
    assert len(warm_state["metadata"]["completed_sample_keys"]) == 2048
    assert len(set(warm_state["metadata"]["completed_sample_keys"])) == 2048
    assert (
        state_hash(load_checkpoint(warm["checkpoint_path"], warm["checkpoint_identity"]))
        == warm["state_hash"]
    )
    assert args[0]["R4"]["allow_training"] is False
    assert len({r["policy_state_hash"] for r in raw if r["bank_role"] == "train"}) > 2
    before = sum(a.generation_calls for a in instances)
    again = run_r4(*args, allow_training=True, resume=True, _adapter_factory=adapter)
    assert again["status"] == "PASS", again["details"]
    assert sum(a.generation_calls for a in instances) == before
    assert instances[-1].forward_calls == 0
    assert again["details"]["runtime_counts"]["optimizer_steps_observed_at_least"] == 129
    for item in json.loads((out / "manifest.json").read_text())["files"]:
        assert file_hash(out / item["path"]) == item["sha256"]


def test_returned_malformed_raw_is_preserved_and_cannot_resume(tmp_path, monkeypatch):
    import json

    args, adapter, instances = _fixture(tmp_path, monkeypatch)

    class Malformed(adapter):
        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            result["behavior_token_logprobs"][0] = "bad"
            return result

    failed = run_r4(*args, allow_training=True, _adapter_factory=Malformed)
    assert failed["status"] == "FAIL"
    path = args[2] / "evaluation/INITIAL/step_00/N/samples.jsonl"
    row = json.loads(path.read_text())
    assert row["generation_return"]["behavior_token_logprobs"][0] == "bad"
    assert row["generation_return"]["raw_completion"]
    assert row["generation_return"]["token_ids"]
    before = sum(a.generation_calls for a in instances)
    failed = run_r4(*args, allow_training=True, resume=True, _adapter_factory=adapter)
    assert failed["status"] == "FAIL"
    assert "failed execution" in failed["details"]["error"]["message"]
    assert sum(a.generation_calls for a in instances) == before


def test_likelihood_contamination_precedes_restore_and_score_tamper_rejected(tmp_path):
    import json
    from types import SimpleNamespace

    import pytest
    import torch
    from test_model_smoke import FakeAdapter

    from src.r1_supplement import _ExecutionMeter
    from src.r4_runtime import _scores

    adapter = FakeAdapter("fixture", {"id": "fixture"})
    optimizer = torch.optim.AdamW([adapter.model.lora], lr=1e-5)
    meter = _ExecutionMeter(adapter, optimizer)
    prompt = {
        "prompt_id": "p",
        "prompt": {"user": "u", "prompt_hash": "h"},
        "interface": "SYMBOLIC_FRESH",
    }
    old_prepare = adapter.prepare

    def prepare(*args):
        prepared = old_prepare(*args)
        prepared["audit"]["enable_thinking"] = False
        return prepared

    adapter.prepare = prepare
    row = {
        "sample_key": "source",
        "record_hash": "source_hash",
        "prompt_id": "p",
        "token_ids": [1, 2],
        "final_prompt_hash": "h",
        "input_tensor_hash": None,
    }
    identity = {
        "model_hash": "m",
        "config_hash": "c",
        "data_hash": "d",
        "execution_kind": "CPU_FAKE_ADAPTER_FIXTURE",
    }
    progress = SimpleNamespace(out=tmp_path, write=lambda *a, **k: None)
    root = tmp_path / "scores"
    args = (
        adapter,
        optimizer,
        [row],
        {"p": prompt},
        root,
        identity,
        tmp_path,
        meter,
        progress,
        "likelihood",
    )
    try:
        _scores(*args)
        path = root / "samples.jsonl"
        original = json.loads(path.read_text())
        broken = {**original, "token_ids": [3, 2]}
        broken["record_hash"] = canonical_hash(
            {k: v for k, v in broken.items() if k != "record_hash"}
        )
        path.write_text(json.dumps(broken) + "\n")
        with pytest.raises(ValueError, match="identity"):
            _scores(*args)
        path.write_text(
            json.dumps(
                {
                    **original,
                    "new_token_logprobs": [1.0, 0.0],
                    "record_hash": canonical_hash(
                        {
                            **{k: v for k, v in original.items() if k != "record_hash"},
                            "new_token_logprobs": [1.0, 0.0],
                        }
                    ),
                }
            )
            + "\n"
        )
        with pytest.raises(ValueError, match="likelihood values"):
            _scores(*args)
        original_logprobs = adapter.logprobs

        def contaminate(*a, **k):
            returned = original_logprobs(*a, **k)
            with torch.no_grad():
                adapter.model.lora.add_(1.0)
            return returned

        adapter.logprobs = contaminate
        with pytest.raises(RuntimeError, match="changed current policy"):
            _scores(
                adapter,
                optimizer,
                [row],
                {"p": prompt},
                tmp_path / "contaminated",
                identity,
                tmp_path,
                meter,
                progress,
                "likelihood",
            )
        assert (tmp_path / "policy_contamination.json").exists()
        assert torch.equal(adapter.model.lora, torch.ones(4))  # detection precedes any restore
    finally:
        meter.close()


def test_control_alarm_retains_step_and_resume_cannot_skip(tmp_path, monkeypatch):
    from src import r4_metrics

    args, adapter, instances = _fixture(tmp_path, monkeypatch)
    diagnostic = r4_metrics.control_kl_diagnostic

    def alarm(*a, **k):
        result = diagnostic(*a, **k)
        return {
            **result,
            "should_stop": True,
            "mean_token_kl": 0.100001,
            "alarms": {"mean_token_kl": True, "sequence_log_ratio_p99_abs": False},
        }

    monkeypatch.setattr(r4_metrics, "control_kl_diagnostic", alarm)
    result = run_r4(*args, allow_training=True, _adapter_factory=adapter)
    assert result["status"] == "BLOCKED", result["details"]
    assert result["details"]["runtime_counts"]["optimizer_steps_observed_at_least"] == 1
    out = args[2]
    assert (out / "alarm_stop.json").exists()
    assert (out / "X_BASE/step_01/updates/attempt_0000/checkpoint.pt").exists()
    assert not (out / "X_BASE/step_02").exists()
    before = len(instances)
    (out / "report_zh.md").write_text("STALE PASS REPORT")
    again = run_r4(*args, allow_training=True, resume=True, _adapter_factory=adapter)
    assert again["status"] == "BLOCKED"
    assert len(instances) == before
    assert "# R4 BLOCKED" in (out / "report_zh.md").read_text()
    assert "STALE PASS REPORT" not in (out / "report_zh.md").read_text()


def test_historical_alignment_uses_verified_rows_but_corruption_is_not_na(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import pytest
    import torch

    from src import legacy_frozen, r4_runtime
    from src.core import file_hash, write_json

    model = torch.nn.Module()
    model.register_parameter("lora_B", torch.nn.Parameter(torch.zeros(1)))
    audit = {
        key: key + "-fixture"
        for key in (
            "processor_hash",
            "tokenizer_hash",
            "chat_template_hash",
            "frozen_parameter_hash",
            "probability_execution",
        )
    }
    prepared_audit = {
        "enable_thinking": False,
        "image_token_count": 0,
        "final_prompt_hash": "h",
        "input_tensor_hash": "i",
        "tokenized_prompt_hash": "t",
        "pixel_values_hash": None,
    }
    adapter = SimpleNamespace(
        model=model,
        model_id="fixture",
        revision="a" * 40,
        audit=audit,
        eos_ids={2},
        processor=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *a, **k: "[1,2,3,4]")),
        prepare=lambda *a: {"audit": prepared_audit},
    )
    plan = {
        "dev_prompts": [
            {
                "prompt_id": "pN",
                "base_scene_id": "bN",
                "interface": "SYMBOLIC_FRESH",
                "family": "duplicate",
                "prompt": {},
                "scene": {},
            }
        ],
        "legacy_prompts": [
            {
                "prompt_id": "pL",
                "base_scene_id": "bL",
                "interface": "collision",
                "family": "duplicate",
                "prompt": {},
                "scene": {},
            }
        ],
    }
    monkeypatch.setattr(r4_runtime, "annotate_diagnostic", lambda *a: {"category": "X"})
    monkeypatch.setattr(legacy_frozen, "annotate_legacy", lambda *a: {"category": "X"})
    config, closure = {"paths": {}}, {}
    r0 = tmp_path / "r0"
    for track, key in (("N", "dev_prompts"), ("L", "legacy_prompts")):
        source = tmp_path / track
        config["paths"]["readonly_" + track] = str(source)
        generation = {
            "do_sample": True,
            "max_new_tokens": 64 if track == "N" else 48,
            "min_p": 0.0,
            "num_beams": 1,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
        }
        write_json(
            source / "runtime_lock.json",
            {
                "generation_protocol": generation,
                "optimizer_updates": 0,
                "identity": {"model_id": "fixture", "model_revision": "a" * 40},
                "model_audit": audit,
                "adapter_state": "fresh base; adapter disabled; no checkpoint loaded",
            },
        )
        write_json(r0 / f"baseline_lock_{track}.json", {"generation": generation})
        prompt = plan[key][0]
        rows = []
        for index in range(16):
            row = {
                "sample_key": f"{track}-{index}",
                "base_scene_id": prompt["base_scene_id"],
                "interface": prompt["interface"],
                "prompt_id": "old-prompt",
                "decode_mode": "sample",
                "category": "X",
                "execution_kind": "REAL_CUDA_MODEL",
                "optimizer_step": 0,
                "policy_base_identical": True,
                "on_policy_likelihood_passed": True,
                "token_ids": [1, 2],
                "raw_completion": "[1,2,3,4]",
                "completion_length": 2,
                "stop_reason": "eos",
                "behavior_token_logprobs": [-1.0, -1.0],
                **prepared_audit,
            }
            row["record_hash"] = canonical_hash(row)
            rows.append(row)
        (source / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        raw_files = {p.name: file_hash(p) for p in source.iterdir()}
        write_json(
            source / "manifest.json",
            {
                "files": [
                    {"path": name, "sha256": digest, "bytes": (source / name).stat().st_size}
                    for name, digest in raw_files.items()
                ]
            },
        )
        closure["base_" + track] = {
            "source_root": str(source),
            "raw_manifest_sha256": file_hash(source / "manifest.json"),
            "raw_files": raw_files,
        }
    write_json(r0 / "closure_checks.json", closure)
    aligned, evidence = r4_runtime._historical_initial(adapter, plan, r0, config, tmp_path)
    assert {track: len(rows) for track, rows in aligned.items()} == {"N": 16, "L": 16}
    assert all(value["status"] == "PASS" for value in evidence.values())
    assert aligned["L"][0]["source_prompt_id"] == "old-prompt"
    path = tmp_path / "L/samples.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match=r"hash|manifest"):
        r4_runtime._historical_initial(adapter, plan, r0, config, tmp_path)


def test_returned_bad_likelihood_blocks_new_attempt_resume(tmp_path, monkeypatch):
    import pytest
    import torch

    from src.r4_runtime import PilotBlocked

    args, adapter, instances = _fixture(tmp_path, monkeypatch)

    class BadScore(adapter):
        def logprobs(self, *args, **kwargs):
            return torch.full((2,), float("nan"))

    result = run_r4(*args, allow_training=True, _adapter_factory=BadScore)
    assert result["status"] == "FAIL"
    assert (args[2] / "measurement_fault.json").exists()
    assert (args[2] / "X_BASE/step_01/updates/attempt_0000/preupdate_parity/samples.jsonl").exists()
    before = len(instances)
    with pytest.raises(PilotBlocked, match="failed measurement"):
        run_r4(*args, allow_training=True, resume=True, _adapter_factory=adapter)
    assert len(instances) == before


def test_cli_preserves_dry_run_and_explicit_flag_without_mutating_yaml(
    tmp_path, monkeypatch, capsys
):
    from src import r4_runtime, train_two_arm_pilot

    assert train_two_arm_pilot.main(["--arm", "X_BASE", "--dry-run"]) == 0
    assert '"training_started": false' in capsys.readouterr().out
    calls = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "PASS"}

    monkeypatch.setattr(r4_runtime, "run_r4", capture)
    command = ["--allow-training", "--resume", "--out", str(tmp_path / "pilot")]
    for flag in ("--data-root", "--r0-dir", "--r1-run", "--supplement-dir", "--r2-dir", "--r3-dir"):
        command.extend([flag, str(tmp_path / flag[2:])])
    assert train_two_arm_pilot.main(command) == 0
    args, kwargs = calls[0]
    assert args[0]["R4"]["allow_training"] is False
    assert kwargs == {"allow_training": True, "resume": True}
    assert args[-1] == tmp_path / "r3-dir"


def test_fixed_control_reuses_only_index_zero_with_exact_state_and_input(tmp_path, monkeypatch):
    import json

    import pytest
    import torch

    from src import r4_runtime
    from src.core import write_json
    from src.optimizer_fork import capture_state, parameter_hash, save_checkpoint
    from src.r3_inputs import build_r3_plan

    actual_probe = r4_runtime._control_probe
    args, adapter_class, _ = _fixture(tmp_path, monkeypatch)
    config, data_root = args[:2]
    adapter = adapter_class("qwen35_9b", {"id": config["model"]["id"]})
    optimizer = torch.optim.AdamW([adapter.model.lora], lr=1e-5, weight_decay=0.0)
    origin = capture_state(adapter.model, optimizer, {"phase": "R4"})
    source = tmp_path / "cold_source"
    identity = {"phase": "R3-cold", "model_hash": "m"}
    old_origin = {**origin, "metadata": {"phase": "R3-cold"}}
    source.mkdir()
    save_checkpoint(source / "origin.pt", old_origin, {**identity, "unit": "initial_origin"})
    write_json(
        source / "runtime_lock.json",
        {"identity": identity, "frozen_base_hash": parameter_hash(adapter.model, trainable=False)},
    )

    def load(split):
        return [
            json.loads(line) for line in (data_root / f"{split}.jsonl").read_text().splitlines()
        ]

    plan = build_r3_plan(load("train"), load("control"))
    write_json(source / "bank_manifest.json", {"plan": plan})
    rows = []
    for prompt in plan["control_prompts"]:
        prepared = adapter.prepare(prompt["prompt"], data_root)
        for index in (0, 1):
            generated = adapter.generate(prepared, seed=index, max_new_tokens=64, do_sample=True)
            row = {
                "sample_key": f"{prompt['prompt_id']}-{index}",
                "prompt_id": prompt["prompt_id"],
                "prompt_hash": prompt["prompt_hash"],
                "sample_index": index,
                "bank_role": "control_proposal",
                "adapter_hash": parameter_hash(adapter.model, trainable=True),
                "model_id": adapter.model_id,
                "model_revision": adapter.revision,
                "generation_config_hash": canonical_hash(config["generation_proposed_N"]),
                **prepared["audit"],
                **generated,
                "execution_checks": {"passed": True},
            }
            row["record_hash"] = canonical_hash(row)
            rows.append(row)
    path = source / "samples.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    selected, _, binding = actual_probe(
        adapter, optimizer, origin, source, {"status": "PASS"}, config, data_root
    )
    assert len(selected) == 48
    assert {r["sample_index"] for r in selected} == {0}
    assert binding["new_control_generations"] == 0
    assert binding["planned_sequence_scores"] == 6144
    rows[0]["input_tensor_hash"] = "changed"
    rows[0]["record_hash"] = canonical_hash(
        {k: v for k, v in rows[0].items() if k != "record_hash"}
    )
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(r4_runtime.PilotBlocked, match="mismatch"):
        actual_probe(adapter, optimizer, origin, source, {"status": "PASS"}, config, data_root)


def test_nonfinite_control_diagnostic_is_a_persistent_failed_measurement(tmp_path, monkeypatch):
    import pytest

    from src import r4_metrics
    from src.r4_runtime import PilotBlocked

    args, adapter, _ = _fixture(tmp_path, monkeypatch)

    def overflow(*a, **k):
        raise FloatingPointError("overflow in expm1")

    monkeypatch.setattr(r4_metrics, "control_kl_diagnostic", overflow)
    result = run_r4(*args, allow_training=True, _adapter_factory=adapter)
    assert result["status"] == "FAIL"
    assert (args[2] / "measurement_fault.json").exists()
    with pytest.raises(PilotBlocked, match="failed measurement"):
        run_r4(*args, allow_training=True, resume=True, _adapter_factory=adapter)


def test_measured_alarm_survives_interruption_before_atomic_step_completion(tmp_path, monkeypatch):
    import json

    from src import r4_metrics, r4_runtime

    args, adapter, instances = _fixture(tmp_path, monkeypatch)
    original_diagnostic = r4_metrics.control_kl_diagnostic
    original_write = r4_runtime.write_json
    interrupted = []

    def measured_alarm(*a, **k):
        result = original_diagnostic(*a, **k)
        return {
            **result,
            "should_stop": True,
            "mean_token_kl": 0.100001,
            "alarms": {"mean_token_kl": True, "sequence_log_ratio_p99_abs": False},
        }

    def interrupt_after_local_diagnostic(path, value):
        original_write(path, value)
        if path.name == "control_diagnostic.json" and not interrupted:
            interrupted.append(path)
            raise RuntimeError("interrupted after measured diagnostic before atomic completion")

    monkeypatch.setattr(r4_metrics, "control_kl_diagnostic", measured_alarm)
    monkeypatch.setattr(r4_runtime, "write_json", interrupt_after_local_diagnostic)
    first = run_r4(*args, allow_training=True, _adapter_factory=adapter)
    assert first["status"] == "FAIL", first["details"]
    assert "before atomic completion" in first["details"]["error"]["message"]
    attempt = args[2] / "X_BASE/step_01/updates/attempt_0000"
    assert (attempt / "control_diagnostic.json").exists()
    assert not (attempt / "completed.json").exists()
    marker = args[2] / "alarm_stop.json"
    assert marker.exists(), "A measured stop must persist before atomic completion"
    preserved = json.loads(marker.read_text())
    assert preserved["control_diagnostic"]["should_stop"]
    assert preserved["unit_identity"]["arm"] == "X_BASE"
    assert preserved["unit_identity"]["step"] == 1
    assert preserved["checkpoint_path"] == str(attempt / "checkpoint.pt")
    before = (
        len(instances),
        sum(a.generation_calls for a in instances),
        sum(a.forward_calls for a in instances),
    )
    again = run_r4(*args, allow_training=True, resume=True, _adapter_factory=adapter)
    assert again["status"] == "BLOCKED"
    assert (
        len(instances),
        sum(a.generation_calls for a in instances),
        sum(a.forward_calls for a in instances),
    ) == before
    assert not (attempt.parent / "attempt_0001").exists()


def test_raw_rollout_schema_preserves_N_L_OOD_and_train_scene_values(tmp_path, monkeypatch):
    import json

    import torch

    from src import r4_runtime
    from src.core import file_hash
    from src.optimizer_fork import capture_state, state_hash
    from src.r1_supplement import _ExecutionMeter

    args, adapter_class, _ = _fixture(tmp_path, monkeypatch)
    config, data_root, out, r0 = args[:4]
    manifest = json.loads((data_root / "manifest.json").read_text())
    gate = {
        "binding": {
            "data_manifest_sha256": file_hash(data_root / "manifest.json"),
            "dataset_files": manifest["files"],
        }
    }
    plan, _, legacy_lock = r4_runtime._load_plan(data_root, gate, r0)
    adapter = adapter_class("qwen35_9b", {"id": config["model"]["id"]})
    optimizer = torch.optim.AdamW([adapter.model.lora], lr=1e-5, weight_decay=0.0)
    origin = capture_state(adapter.model, optimizer, {"checkpoint_step": 0})
    meter = _ExecutionMeter(adapter, optimizer)
    progress = r4_runtime._Progress(out, out / "invocation", meter)
    identity = {
        "model_hash": "m",
        "config_hash": canonical_hash(config),
        "data_hash": "d",
        "execution_kind": "CPU_FAKE_ADAPTER_FIXTURE",
    }
    cases = [
        (
            "train",
            next(p for p in plan["train_prompts"] if p["interface"] == "IMAGE_CUE_FRESH"),
            "train",
        ),
        (
            "N",
            next(p for p in plan["dev_panel_prompts"] if p["interface"] == "SYMBOLIC_FRESH"),
            "evaluation",
        ),
        ("L", plan["legacy_prompts"][0], "evaluation"),
        (
            "OOD",
            next(p for p in plan["ood_prompts"] if p["interface"] == "IMAGE_CUE_FRESH"),
            "evaluation",
        ),
    ]
    prepared_hash_calls = []
    original_hash = r4_runtime.state_hash

    def counted_hash(value):
        if isinstance(value, dict) and {"inputs", "audit"} <= value.keys():
            prepared_hash_calls.append(value)
        return original_hash(value)

    monkeypatch.setattr(r4_runtime, "state_hash", counted_hash)
    try:
        for name, prompt, role in cases:
            expected_prepared = r4_runtime._prepared(adapter, prompt, data_root)
            rows = r4_runtime._samples(
                adapter,
                optimizer,
                origin,
                [prompt],
                out / name,
                identity,
                "X_BASE",
                0 if role == "train" else 64,
                role,
                data_root,
                config,
                legacy_lock,
                meter,
                progress,
            )
            assert len(rows) == 8
            scene = prompt["scene"]
            for row in rows:
                assert row["protocol_version"] == config["protocol_version"]
                assert row["prepared_hash"] == state_hash(expected_prepared)
                assert row["input_ids_hash"] == expected_prepared["audit"]["tokenized_prompt_hash"]
                assert row["actual_image_tokens"] == expected_prepared["audit"]["image_token_count"]
                assert row["n_generated_tokens"] == len(row["token_ids"]) == 2
                assert (
                    row["per_token_logprob_behavior"]
                    == row["behavior_token_logprobs"]
                    == row["old_logprobs"]
                )
                assert row["runtime_forward_by_reason"] == {"generation": 2}
                assert row["elapsed"] == row["elapsed_seconds"] == 0.001
                assert row["elapsed_status"] == "CPU_FIXTURE_ADAPTER_RETURN"
                assert row["peak_memory_status"].startswith("CPU_FIXTURE")
                assert row["peak_memory"]["peak_cpu_rss_bytes"] > 0
                assert "image_grid_thw" in row
                assert row["group_id"] == prompt["prompt_id"]
                assert row["operation"] == scene["operation"]
                if name == "L":
                    assert row["truth_world"] == scene["truth"]
                    assert row["observed_world"] == scene["observation"]
                    assert row["changed_index"] == scene["error_index"]
                    assert row["legacy_facts"] == scene["facts"]
                    assert row["legacy_scene_id"] == scene["scene_id"]
                    assert all(
                        row[key] is None
                        for key in ("cue", "solution_count", "chart_type", "image_hash")
                    )
                    assert set(row["not_applicable_fields"]) == {
                        "cue",
                        "solution_count",
                        "chart_type",
                        "image_hash",
                        "constraint_results",
                    }
                else:
                    assert row["truth_world"] == scene["truth_world"]
                    assert row["observed_world"] == scene["observed_world"]
                    assert row["changed_index"] == scene["changed_index"]
                    assert row["cue"] == scene["cue"]
                    assert row["chart_type"] == scene["chart_type"]
                    assert row["solution_count"] == 1
                    assert row["image_hash"] == (
                        scene["image_hash"] if row["actual_image_tokens"] else None
                    )
                    assert row["not_applicable_fields"] == {}
        assert len(prepared_hash_calls) == 4  # once per prompt, not per K8 rollout
    finally:
        meter.close()


def test_legacy_rollout_aliases_use_original_semantic_parser_and_no_N_constraints():
    from src.legacy_frozen import REPOSITORY, load_legacy_scenes
    from src.r4_runtime import _annotate_rollout

    root = REPOSITORY / "artifacts/v5/study_c2/data"
    scene = load_legacy_scenes(root / "reward_fibers.jsonl", root / "reward_fibers_manifest.json")[
        0
    ]
    prompt = {"track": "L", "scene": scene}
    valid = _annotate_rollout(",".join(map(str, scene["truth"])), prompt)
    assert valid["category"] == "X"
    assert valid["parse_result"] == "valid"
    assert valid["extracted_action"] == valid["parsed_world"] == scene["truth"]
    assert valid["hamming_to_observed"] == 1
    assert valid["hamming_to_truth"] == 0
    assert valid["constraint_results"] is None
    invalid = _annotate_rollout("invalid", prompt)
    assert invalid["category"] == "I"
    assert invalid["parse_result"] == "invalid"
    assert invalid["extracted_action"] is None
    assert invalid["hamming_to_observed"] is invalid["hamming_to_truth"] is None
    assert invalid["constraint_results"] is None
