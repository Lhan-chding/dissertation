"""Warm state fixtures exercise real tiny Adam without certifying CUDA."""

from src.r3_warm_runtime import build_direct_requests


def test_direct_requests_lock_four_candidates_and_document_seed_coupling():
    from test_r3_runtime import plan

    selected = plan()
    identity = {
        "phase": "R3-warm",
        "model_hash": "m",
        "data_hash": "d",
        "config_hash": "c",
        "origin_hash": "warm",
        "source_hash": "source",
    }
    candidate = {
        "candidate_id": "bank_00_lambda_0",
        "lambda": 0.0,
        "parameter_hash": "a",
        "optimizer_state_hash": "o",
        "optimizer_step": 65,
    }
    requests = build_direct_requests(selected, identity, candidate, 0, "candidate-state")
    assert len(requests) == len({r["sample_key"] for r in requests}) == 768
    assert requests == build_direct_requests(selected, identity, candidate, 0, "candidate-state")
    other = build_direct_requests(
        selected,
        identity,
        {**candidate, "candidate_id": "bank_00_lambda_1", "lambda": 1.0, "parameter_hash": "b"},
        0,
        "other-state",
    )
    assert [r["sample_seed"] for r in requests] == [r["sample_seed"] for r in other]
    assert not ({r["sample_key"] for r in requests} & {r["sample_key"] for r in other})
    changed = build_direct_requests(
        selected, {**identity, "origin_hash": "changed-warm"}, candidate, 0, "candidate-state"
    )
    assert changed[0]["sample_seed"] != requests[0]["sample_seed"]
    assert all(
        r["checkpoint_step"] == 64
        and r["bank_role"] == "direct_control"
        and r["split"] == "control"
        for r in requests
    )


def _fixture(tmp_path, monkeypatch):
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import torch
    from test_model_smoke import FakeAdapter
    from test_r1_reference_smoke import setup
    from test_r3_runtime import plan

    from src import r3_runtime
    from src.core import file_hash
    from src.optimizer_fork import capture_state, parameter_hash, save_checkpoint, state_hash

    config, _, initial, certificate = setup(tmp_path)
    data_root = Path("data/generated").resolve()
    config["data_root"] = str(data_root)
    manifest = json.loads((data_root / "manifest.json").read_text())
    gate = {
        "certificate": certificate,
        "config": config,
        "binding": {
            "data_manifest_sha256": file_hash(data_root / "manifest.json"),
            "dataset_files": manifest["files"],
        },
    }
    optimizer = torch.optim.AdamW([initial.model.lora], lr=1e-5, weight_decay=0.0)
    # Build an actual mature Adam state before the runner's measured interval.
    for _ in range(64):
        optimizer.zero_grad(set_to_none=True)
        initial.model.lora.square().sum().add(
            initial.model.lora @ torch.tensor([1.0, -1.0, 0.5, -0.5])
        ).backward()
        optimizer.step()
    warm = capture_state(
        initial.model,
        optimizer,
        {
            "arm": "X_BASE",
            "checkpoint_step": 64,
            "sampler": {"plan_hash": "R4-plan", "position": 64},
            "completed_sample_keys": [f"R4-own-sample-{i}" for i in range(2048)],
            "scheduler": None,
            "grad_scaler": None,
        },
    )
    r4_root = tmp_path / "r4"
    r4_root.mkdir()
    checkpoint_path = r4_root / "X_BASE_step64.pt"
    checkpoint_identity = {"phase": "R4", "arm": "X_BASE", "step": 64}
    save_checkpoint(checkpoint_path, warm, checkpoint_identity)
    checkpoint = {
        "path": str(checkpoint_path),
        "identity": checkpoint_identity,
        "file_sha256": file_hash(checkpoint_path),
        "state_hash": state_hash(warm),
        "parameter_hash": parameter_hash(initial.model, trainable=True),
        "optimizer_state_hash": state_hash(warm["optimizer"]),
    }
    r2 = {"status": "PASS"}
    cold = {"status": "PASS", "plan_hash": plan()["plan_hash"]}
    r4 = {"status": "PASS", "warm_checkpoint": checkpoint, "r3_cold_plan_hash": cold["plan_hash"]}
    monkeypatch.setitem(
        sys.modules,
        "src.next_stage_runtime",
        SimpleNamespace(
            validate_prerequisites=lambda *a: gate,
            validate_config_against_gate=lambda c, g: c,
            validate_r2_gate=lambda *a: r2,
            validate_r3_cold_gate=lambda *a: cold,
            validate_r4_gate=lambda *a: r4,
            validate_runtime_environment=lambda *a: (_ for _ in ()).throw(
                AssertionError("not real CUDA")
            ),
        ),
    )
    monkeypatch.setattr(r3_runtime, "_source", lambda: {"source_commit": "CPU_FIXTURE"})
    instances = []

    class Tokenizer:
        def decode(self, ids, **kwargs):
            return "".join({1: "[1,2,3,4]", 3: "invalid", 2: "<eos>"}[i] for i in ids)

    class Adapter(FakeAdapter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.audit = {**certificate["model_audit"], "load_seconds": len(instances) + 0.12}
            self.processor = SimpleNamespace(tokenizer=Tokenizer())
            self.eos_ids = {2}
            instances.append(self)

        def prepare(self, prompt, root):
            from src.core import canonical_hash

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
            result = super().generate(prepared, seed=seed, max_new_tokens=max_new_tokens)
            result["vision_forward_calls"] = int(bool(prepared["image"]))
            self.forward_calls += len(result["token_ids"])
            return result

    return (
        (config, data_root, tmp_path / "warm", "r0", "r1", "supp", "r2", "cold", r4_root),
        Adapter,
        instances,
        warm,
        checkpoint,
    )


def test_full_warm_actual_adam_origin_direct_interrupt_and_resume(tmp_path, monkeypatch):
    import json

    from src.core import file_hash
    from src.optimizer_fork import load_checkpoint, state_hash
    from src.r3_warm_runtime import run_r3_warm

    args, adapter, instances, warm, checkpoint = _fixture(tmp_path, monkeypatch)
    interrupted = []

    class InterruptDirect(adapter):
        def generate(self, *args, **kwargs):
            if self.generation_calls == 1155 and not interrupted:
                interrupted.append(True)
                raise RuntimeError("interrupted direct output after three returns")
            return super().generate(*args, **kwargs)

    first = run_r3_warm(*args, _adapter_factory=InterruptDirect)
    assert first["status"] == "FAIL", first["details"]
    assert "interrupted direct" in first["details"]["error"]["message"]
    assert all(first["details"]["scratch_restoration"].values())
    out = args[2]
    direct_path = next((out / "direct_samples").glob("*/samples.jsonl"))
    prefix = direct_path.read_bytes()
    assert len(prefix.splitlines()) == 3
    complete = run_r3_warm(*args, resume=True, _adapter_factory=adapter)
    assert complete["status"] == "PASS", complete["details"]
    assert complete["phase"] == "R3-warm"
    assert complete["execution_kind"] == "CPU_FAKE_ADAPTER_FIXTURE"
    assert complete["details"]["raw_sample_count"] == 4224
    assert complete["details"]["direct_rollouts"] == 3072
    assert complete["details"]["runtime_counts"]["optimizer_steps_observed_at_least"] == 80
    assert direct_path.read_bytes().startswith(prefix)
    lock = json.loads((out / "runtime_lock.json").read_text())
    origin = load_checkpoint(out / "origin.pt", {**lock["identity"], "unit": "initial_origin"})
    assert state_hash(origin) == state_hash(warm) == lock["origin_hash"]
    assert all(float(v["step"]) == 64 for v in origin["optimizer"]["state"].values())
    base = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
    assert len(base) == 1152
    assert all(
        r["checkpoint_step"] == 64 and r["adapter_hash"] == checkpoint["parameter_hash"]
        for r in base
    )
    direct = [
        json.loads(line)
        for p in (out / "direct_samples").glob("*/samples.jsonl")
        for line in p.read_text().splitlines()
    ]
    assert len(direct) == len({r["sample_key"] for r in direct}) == 3072
    assert len({r["candidate_id"] for r in direct}) == 4
    assert all(r["checkpoint_step"] == 64 and r["bank_role"] == "direct_control" for r in direct)
    assert all(r["adapter_hash"] == r["candidate_parameter_hash"] for r in direct)
    assert all(r["origin_checkpoint_step"] == 64 and r["optimizer_step"] == 64 for r in base)
    assert all(
        r["origin_checkpoint_step"] == 64
        and r["optimizer_step"] == r["candidate_optimizer_step"] == 65
        for r in direct
    )
    required = {
        "protocol_version",
        "truth_world",
        "observed_world",
        "changed_index",
        "cue",
        "solution_count",
        "chart_type",
        "operation",
        "image_hash",
        "input_ids_hash",
        "actual_image_tokens",
        "n_generated_tokens",
        "per_token_logprob_behavior",
        "runtime_forward_by_reason",
        "elapsed",
        "peak_memory",
        "memory_measurement_status",
        "prepared_hash",
        "parse_result",
        "extracted_action",
        "constraint_results",
    }
    assert all(required <= r.keys() for r in [*base, *direct])
    assert all(r["elapsed"] >= 0 for r in direct)
    assert all(r["memory_measurement_status"] == "CPU_FIXTURE_CUDA_NOT_MEASURED" for r in direct)
    before = sum(a.generation_calls for a in instances)
    again = run_r3_warm(*args, resume=True, _adapter_factory=adapter)
    assert again["status"] == "PASS", again["details"]
    assert sum(a.generation_calls for a in instances) == before
    assert instances[-1].forward_calls == 0
    assert again["details"]["runtime_counts"]["optimizer_steps_observed_at_least"] == 80
    assert file_hash(checkpoint["path"]) == checkpoint["file_sha256"]
    for item in json.loads((out / "manifest.json").read_text())["files"]:
        assert file_hash(out / item["path"]) == item["sha256"]


def test_explicit_checkpoint_mismatch_is_rejected_before_model_load(tmp_path, monkeypatch):
    import pytest

    from src.r3_warm_runtime import run_r3_warm

    args, adapter, instances, _, _ = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="exactly equal verified"):
        run_r3_warm(*args, checkpoint=tmp_path / "different.pt", _adapter_factory=adapter)
    assert not instances


def test_resealed_checkpoint_with_wrong_adam_step_is_rejected(tmp_path, monkeypatch):
    from src.core import file_hash
    from src.optimizer_fork import save_checkpoint, state_hash
    from src.r3_warm_runtime import run_r3_warm

    args, adapter, instances, warm, checkpoint = _fixture(tmp_path, monkeypatch)
    next(iter(warm["optimizer"]["state"].values()))["step"].fill_(63)
    save_checkpoint(checkpoint["path"], warm, checkpoint["identity"])
    checkpoint.update(
        file_sha256=file_hash(checkpoint["path"]),
        state_hash=state_hash(warm),
        optimizer_state_hash=state_hash(warm["optimizer"]),
    )
    result = run_r3_warm(*args, _adapter_factory=adapter)
    assert result["status"] == "FAIL"
    assert "exact X_BASE step64" in result["details"]["error"]["message"]
    assert instances[0].generation_calls == instances[0].forward_calls == 0


def _direct_unit(tmp_path, monkeypatch):
    """Use one real Adam65 candidate, with no expensive fork/statistic orchestration."""
    import torch
    from test_r3_runtime import plan

    from src.core import RunStore
    from src.optimizer_fork import capture_state, parameter_hash, save_checkpoint, state_hash
    from src.r1_supplement import _ExecutionMeter
    from src.r3_runtime import _Profile, _restore

    args, factory, instances, warm, _checkpoint = _fixture(tmp_path, monkeypatch)
    adapter = factory("qwen35_9b", args[0]["model"])
    optimizer = torch.optim.AdamW([adapter.model.lora], lr=1e-5, weight_decay=0.0)
    _restore(adapter, optimizer, warm)
    adapter.model.lora.square().sum().backward()
    optimizer.step()
    candidate_state = capture_state(adapter.model, optimizer, warm["metadata"])
    identity = {
        "phase": "R3-warm",
        "origin_hash": state_hash(warm),
        "model_hash": "fixture",
        "data_hash": "data",
        "config_hash": "config",
        "source_hash": "fixture",
        "initial_adapter_hash": parameter_hash(adapter.model, trainable=True),
        "checkpoint_step": 64,
    }
    attempt = tmp_path / "fork_attempt"
    attempt.mkdir()
    candidate_path = attempt / "candidate.pt"
    candidate_identity = {"candidate_id": "bank_00_lambda_0", "origin_hash": state_hash(warm)}
    save_checkpoint(candidate_path, candidate_state, candidate_identity)
    candidate = {
        "candidate_id": "bank_00_lambda_0",
        "lambda": 0.0,
        "parameter_hash": parameter_hash(adapter.model, trainable=True),
        "optimizer_state_hash": state_hash(candidate_state["optimizer"]),
        "optimizer_step": 65,
        "checkpoint_path": str(candidate_path),
        "checkpoint_identity": candidate_identity,
    }
    selected = plan()
    requests = build_direct_requests(selected, identity, candidate, 0, state_hash(candidate_state))
    out = args[2]
    store = RunStore(out, identity)
    invocation = out / "invocation"
    invocation.mkdir()
    meter = _ExecutionMeter(adapter, optimizer)
    profile = _Profile(out, invocation, meter, True, args[0], identity)
    return {
        "args": args,
        "factory": factory,
        "instances": instances,
        "adapter": adapter,
        "optimizer": optimizer,
        "origin": warm,
        "state": candidate_state,
        "identity": identity,
        "candidate": candidate,
        "attempt": attempt,
        "plan": selected,
        "requests": requests,
        "store": store,
        "meter": meter,
        "profile": profile,
        "data_root": args[1],
        "out": out,
    }


def _collect_unit(unit, requests=None):
    from src.r3_runtime import _collect_samples

    return _collect_samples(
        unit["adapter"],
        unit["optimizer"],
        unit["state"],
        unit["plan"],
        unit["requests"][:1] if requests is None else requests,
        unit["store"],
        unit["data_root"],
        unit["meter"],
        unit["profile"],
    )


def test_direct_returned_malformed_raw_is_preserved_and_cannot_resume(tmp_path, monkeypatch):
    import pytest

    from src.r3_runtime import validate_sample_ledger

    unit = _direct_unit(tmp_path, monkeypatch)
    generate = unit["adapter"].generate

    def malformed(*args, **kwargs):
        returned = generate(*args, **kwargs)
        returned["behavior_token_logprobs"] = ["bad", -1.0]
        return returned

    monkeypatch.setattr(unit["adapter"], "generate", malformed)
    try:
        with pytest.raises(TypeError):
            _collect_unit(unit)
        row = next(iter(unit["store"].records.values()))
        assert row["generation_return"]["behavior_token_logprobs"] == ["bad", -1.0]
        assert row["candidate_optimizer_step"] == 65
        assert row["execution_checks"]["passed"] is False
        with pytest.raises(ValueError, match="preserved failed execution"):
            validate_sample_ledger(unit["store"].records, unit["requests"][:1])
        assert unit["adapter"].generation_calls == 1
    finally:
        unit["meter"].close()


def test_sampling_exception_cannot_hide_parameter_contamination(tmp_path, monkeypatch):
    import pytest
    import torch

    from src.r3_warm_runtime import run_r3_warm

    unit = _direct_unit(tmp_path, monkeypatch)

    def contaminated(*args, **kwargs):
        with torch.no_grad():
            unit["adapter"].model.lora.add_(1)
        raise RuntimeError("interrupted after mutation before output")

    monkeypatch.setattr(unit["adapter"], "generate", contaminated)
    try:
        with pytest.raises(RuntimeError, match="read-only measurement changed"):
            _collect_unit(unit)
        assert (unit["out"] / "policy_contamination.json").is_file()
    finally:
        unit["meter"].close()
    before = len(unit["instances"])
    with pytest.raises(ValueError, match="policy-contaminated"):
        run_r3_warm(*unit["args"], resume=True, _adapter_factory=unit["factory"])
    assert len(unit["instances"]) == before


def _score_unit(unit, proposal):
    from src.r3_runtime import _score_candidate

    return _score_candidate(
        unit["adapter"],
        unit["optimizer"],
        unit["origin"],
        unit["candidate"],
        unit["attempt"],
        proposal,
        unit["plan"],
        unit["out"],
        unit["identity"],
        unit["data_root"],
        unit["meter"],
        unit["profile"],
    )


def test_candidate_bad_returned_scores_stay_failed_across_resume(tmp_path, monkeypatch):
    import pytest
    import torch

    from src.core import RunStore

    unit = _direct_unit(tmp_path, monkeypatch)
    try:
        _collect_unit(unit)
        proposal = list(unit["store"].records.values())
        monkeypatch.setattr(
            unit["adapter"], "logprobs", lambda *a, **kw: torch.tensor([float("nan"), -1.0])
        )
        with pytest.raises(RuntimeError, match="score fault"):
            _score_unit(unit, proposal)
        root = unit["out"] / "control_scores" / unit["candidate"]["candidate_id"]
        saved = RunStore(
            root, __import__("json").loads((root / "identity.json").read_text()), resume=True
        )
        row = next(iter(saved.records.values()))
        assert row["candidate_token_logprobs"] == ["nan", -1.0]
        assert row["execution_checks"]["passed"] is False
        with pytest.raises(ValueError, match="preserved failed execution"):
            _score_unit(unit, proposal)
    finally:
        unit["meter"].close()


def test_scoring_exception_cannot_hide_frozen_parameter_mutation(tmp_path, monkeypatch):
    import json

    import pytest
    import torch

    unit = _direct_unit(tmp_path, monkeypatch)
    try:
        _collect_unit(unit)
        proposal = list(unit["store"].records.values())

        def bad_score(*args, **kwargs):
            with torch.no_grad():
                unit["adapter"].model.frozen.add_(1)
            raise RuntimeError("measurement interrupted")

        monkeypatch.setattr(unit["adapter"], "logprobs", bad_score)
        with pytest.raises(RuntimeError, match="read-only measurement changed"):
            _score_unit(unit, proposal)
        marker = json.loads((unit["out"] / "policy_contamination.json").read_text())
        assert marker["phase"] == "control_likelihood"
        assert marker["checks"]["frozen_parameter_versions"] is False
    finally:
        unit["meter"].close()


def test_returned_nonfinite_training_score_blocks_new_fork_attempt_before_load(
    tmp_path, monkeypatch
):
    import json

    import pytest

    from src.r3_warm_runtime import run_r3_warm

    args, adapter, instances, _, _ = _fixture(tmp_path, monkeypatch)

    class BadTraining(adapter):
        def logprobs(self, *args, **kwargs):
            value = super().logprobs(*args, **kwargs)
            return value * float("nan") if kwargs.get("require_grad") else value

    first = run_r3_warm(*args, _adapter_factory=BadTraining)
    assert first["status"] == "FAIL"
    marker = args[2] / "measurement_fault.json"
    assert marker.exists()
    failure = json.loads(marker.read_text())
    assert failure["returned_token_logprobs"] == ["nan", "nan"]
    assert failure["identity"]["unit"] == "reuse_validation"
    assert failure["require_grad"] is True
    assert all(first["details"]["scratch_restoration"].values())
    assert len(list((args[2] / "validation" / "bank_01").glob("attempt_*"))) == 1
    before = len(instances)
    with pytest.raises(ValueError, match="measurement fault"):
        run_r3_warm(*args, resume=True, _adapter_factory=adapter)
    assert len(instances) == before


def test_saved_score_cannot_be_resealed_without_candidate_probabilities(tmp_path, monkeypatch):
    import json

    import pytest

    from src.core import canonical_hash

    unit = _direct_unit(tmp_path, monkeypatch)
    try:
        _collect_unit(unit)
        proposal = list(unit["store"].records.values())
        with pytest.raises(ValueError, match="incomplete control likelihood"):
            _score_unit(unit, proposal)
        path = unit["out"] / "control_scores" / unit["candidate"]["candidate_id"] / "samples.jsonl"
        row = json.loads(path.read_text())
        del row["candidate_token_logprobs"]
        row["record_hash"] = canonical_hash({k: v for k, v in row.items() if k != "record_hash"})
        path.write_text(json.dumps(row) + "\n")
        before = unit["adapter"].forward_calls
        with pytest.raises(ValueError, match="saved candidate token probabilities"):
            _score_unit(unit, proposal)
        assert unit["adapter"].forward_calls == before
    finally:
        unit["meter"].close()


def test_engine_assertion_failure_persists_but_interruption_allows_new_attempt(tmp_path):
    from src.r3_runtime import _record_engine_fault

    identity = {"unit": "bank_forks", "bank_index": 0}
    _record_engine_fault(tmp_path, identity, tmp_path / "attempt", RuntimeError("interrupted"))
    assert not (tmp_path / "measurement_fault.json").exists()
    _record_engine_fault(
        tmp_path,
        identity,
        tmp_path / "attempt",
        RuntimeError("Parameters changed during gradient computation"),
    )
    assert (tmp_path / "measurement_fault.json").exists()
