"""Cold-fork orchestration fixtures are explicitly not CUDA evidence."""

import json
from collections import Counter
from pathlib import Path

import pytest

from src.core import canonical_hash
from src.r3_runtime import build_requests, run_atomic_unit, run_r3, validate_sample_ledger


def plan():
    from src.r3_inputs import build_r3_plan

    def load(name):
        return [
            json.loads(line)
            for line in Path(f"data/generated/{name}.jsonl").read_text().splitlines()
        ]

    return build_r3_plan(load("train"), load("control"))


def test_fixed_request_counts_and_checkpoint_bound_rng():
    selected = plan()
    identity = {
        "model_hash": "m",
        "initial_adapter_hash": "a",
        "phase": "R3-cold",
        "checkpoint_step": 0,
    }
    requests = build_requests(selected, identity)
    assert len(requests) == len({r["sample_key"] for r in requests}) == 1152
    assert Counter(r["bank_role"] for r in requests) == {"train": 384, "control_proposal": 768}
    assert Counter(r["bank_index"] for r in requests if r["bank_role"] == "train") == {
        i: 32 for i in range(12)
    }
    assert requests == build_requests(selected, identity)
    changed = build_requests(selected, {**identity, "initial_adapter_hash": "different"})
    assert requests[0]["sample_rng_key"] != changed[0]["sample_rng_key"]
    assert requests[0]["sample_seed"] != changed[0]["sample_seed"]
    assert all(r["decode_mode"] == "sample" and r["max_new_tokens"] == 64 for r in requests)


def test_samples_reject_control_as_training_and_bad_return_on_resume():
    requests = build_requests(plan(), {"phase": "R3-cold", "model_hash": "m"})
    row = {**requests[0], "execution_checks": {"passed": True}}
    row["record_hash"] = canonical_hash(row)
    validate_sample_ledger({row["sample_key"]: row}, requests)
    changed = {**row, "bank_role": "control_proposal"}
    changed["record_hash"] = canonical_hash(
        {k: v for k, v in changed.items() if k != "record_hash"}
    )
    with pytest.raises(ValueError, match="identity"):
        validate_sample_ledger({row["sample_key"]: changed}, requests)
    changed = {**row, "execution_checks": {"passed": False}}
    changed["record_hash"] = canonical_hash(
        {k: v for k, v in changed.items() if k != "record_hash"}
    )
    with pytest.raises(ValueError, match="failed execution"):
        validate_sample_ledger({row["sample_key"]: changed}, requests)


def test_atomic_units_reuse_success_preserve_failed_attempts_and_reject_tamper(tmp_path):
    calls = []

    def fail(out):
        calls.append(out)
        (out / "partial.bin").write_bytes(b"incomplete measured work")
        raise RuntimeError("interrupted unit")

    root, identity = tmp_path / "unit", {"bank": 3, "origin": "a"}
    with pytest.raises(RuntimeError, match="interrupted unit"):
        run_atomic_unit(root, identity, fail)

    def succeed(out):
        calls.append(out)
        (out / "candidate.bin").write_bytes(b"complete candidate")
        return {"status": "PASS", "candidate_optimizer_updates": 5}

    first_path, first, reused = run_atomic_unit(root, identity, succeed)
    assert not reused
    second_path, second, reused = run_atomic_unit(root, identity, succeed)
    assert reused and first_path == second_path and first == second
    assert len(calls) == 2
    assert (calls[0] / "partial.bin").exists()
    (first_path / "candidate.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match=r"hash|manifest"):
        run_atomic_unit(root, identity, succeed)
    assert len(calls) == 2


def test_warm_unimplemented_is_blocked_without_loading_or_certifying(tmp_path):
    result = run_r3({}, tmp_path, tmp_path / "warm", "r0", "r1", "supp", "r2", state="warm")
    assert result["status"] == "BLOCKED"
    assert not result["details"]["warm_implemented"]
    assert result["execution_kind"] == "CPU_AUDIT"


def test_full_cold_fixture_actual_toy_adam_and_resume(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    import torch
    from test_model_smoke import FakeAdapter
    from test_r1_reference_smoke import setup

    from src import r3_runtime, r3_updates
    from src.core import file_hash

    config, _, initial, certificate = setup(tmp_path)
    data_root = Path("data/generated").resolve()
    config["data_root"] = str(data_root)
    manifest = json.loads((data_root / "manifest.json").read_text())
    gate = {
        "config": config,
        "certificate": certificate,
        "binding": {
            "data_manifest_sha256": file_hash(data_root / "manifest.json"),
            "dataset_files": manifest["files"],
        },
    }
    monkeypatch.setitem(
        sys.modules,
        "src.next_stage_runtime",
        SimpleNamespace(
            validate_prerequisites=lambda *args: gate,
            validate_config_against_gate=lambda candidate, checked: candidate,
            validate_r2_gate=lambda *args: {"status": "PASS", "raw_sample_count": 2232},
            validate_runtime_environment=lambda *args: pytest.fail(
                "CPU fake cannot validate real environment"
            ),
        ),
    )
    monkeypatch.setattr(r3_runtime, "_source", lambda: {"source_commit": "cpu-fixture"})
    instances = []
    engine = r3_updates.run_bank_forks
    interrupted_bank = []

    def interrupt_one_completed_engine(*args, **kwargs):
        result = engine(*args, **kwargs)
        if args[5]["bank_index"] == 4 and not interrupted_bank:
            interrupted_bank.append(str(args[4]))
            raise RuntimeError("fixture interrupted after bank engine returned")
        return result

    monkeypatch.setattr(r3_updates, "run_bank_forks", interrupt_one_completed_engine)

    class Tokenizer:
        def decode(self, tokens, **kwargs):
            return "".join({1: "[1,2,3,4]", 3: "invalid", 2: "<eos>"}[t] for t in tokens)

    class Adapter(FakeAdapter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.audit = dict(initial.audit)
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
            if len(instances) == 1 and self.generation_calls == 3:
                raise RuntimeError("CPU fixture interrupted after three returns")
            result = super().generate(prepared, seed=seed, max_new_tokens=max_new_tokens)
            result["vision_forward_calls"] = int(bool(prepared["image"]))
            self.forward_calls += len(result["token_ids"])
            return result

    out = tmp_path / "cold"
    args = (config, data_root, out, "r0", "r1", "supp", "r2")
    failed = run_r3(*args, _adapter_factory=Adapter)
    assert failed["status"] == "FAIL"
    prefix = (out / "samples.jsonl").read_bytes()
    assert len(prefix.splitlines()) == 3
    bank_interrupted = run_r3(*args, resume=True, _adapter_factory=Adapter)
    assert bank_interrupted["status"] == "FAIL"
    assert "bank engine returned" in bank_interrupted["details"]["error"]["message"]
    assert (Path(interrupted_bank[0]) / "summary.json").exists()
    assert len((out / "samples.jsonl").read_bytes().splitlines()) == 1152
    complete = run_r3(*args, resume=True, _adapter_factory=Adapter)
    assert complete["status"] == "PASS", complete["details"]
    assert complete["execution_kind"] == "CPU_FAKE_ADAPTER_FIXTURE"
    assert complete["details"]["distinct_candidate_optimizer_updates"] == 60
    assert complete["details"]["control_candidate_sequences_scored"] == 7680
    # 60 distinct candidates + 12 replays + 8 validation probes + 6 repeated
    # measured updates in the interrupted atomic bank; none is called zero work.
    assert complete["details"]["runtime_counts"]["optimizer_steps_observed_at_least"] == 86
    assert complete["details"]["runtime_counts"]["backward_calls_observed_at_least"] > 0
    assert all(complete["details"]["scratch_restoration"].values())
    assert (out / "samples.jsonl").read_bytes().startswith(prefix)
    rows = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
    assert len(rows) == 1152
    assert Counter(r["bank_role"] for r in rows) == {"train": 384, "control_proposal": 768}
    assert all(r["run_id"] and r["origin_state_hash"] for r in rows)
    assert len({r["origin_state_hash"] for r in rows}) == 1
    before_generation = sum(a.generation_calls for a in instances)
    again = run_r3(*args, resume=True, _adapter_factory=Adapter)
    assert again["status"] == "PASS", again["details"]
    assert sum(a.generation_calls for a in instances) == before_generation
    assert again["details"]["runtime_counts"]["optimizer_steps_observed_at_least"] == 86
    latest = sorted((out / "invocations").iterdir())[-1]
    meter = json.loads((latest / "runtime_profile.json").read_text())
    assert meter["optimizer_step_calls_observed"] == meter["backward_calls_observed"] == 0
    assert instances[-1].forward_calls == 0
    for item in json.loads((out / "manifest.json").read_text())["files"]:
        assert file_hash(out / item["path"]) == item["sha256"]
        assert (out / item["path"]).stat().st_size == item["bytes"]
    assert torch.isfinite(instances[-1].model.lora).all()

    class BadReturnAdapter(Adapter):
        def generate(self, prepared, **kwargs):
            returned = super().generate(prepared, **kwargs)
            returned["behavior_token_logprobs"][0] = "invalid score"
            return returned

    bad_out = tmp_path / "bad-return"
    bad_args = (config, data_root, bad_out, "r0", "r1", "supp", "r2")
    malformed = run_r3(*bad_args, _adapter_factory=BadReturnAdapter)
    assert malformed["status"] == "FAIL"
    failed_row = json.loads((bad_out / "samples.jsonl").read_text())
    assert failed_row["generation_return"]["behavior_token_logprobs"][0] == "invalid score"
    assert failed_row["generation_return"]["raw_completion"]
    assert failed_row["generation_return"]["token_ids"]
    assert not failed_row["execution_checks"]["passed"]
    retry = run_r3(*bad_args, resume=True, _adapter_factory=Adapter)
    assert retry["status"] == "FAIL"
    assert "failed execution" in retry["details"]["error"]["message"]
