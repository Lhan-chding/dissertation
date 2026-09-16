import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PATH = Path(__file__).parents[2] / "scripts/benchmark_modeling_v4_batch_paths.py"
SPEC = importlib.util.spec_from_file_location("v4_batch_benchmark", PATH)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class TinyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(5, dtype=torch.float32))
        self.forward_calls = 0
        self.batch_sizes = []

    def forward(self, input_ids, **kwargs):
        self.forward_calls += 1
        self.batch_sizes.append(input_ids.shape[0])
        return SimpleNamespace(logits=self.weight.expand(*input_ids.shape, 5))

    def generate(self, input_ids, generation_config, **kwargs):
        output = self(input_ids, **kwargs)
        first = output.logits[:, -1]
        tail = torch.tensor([[2, 4]]).repeat(len(input_ids), 1)
        second = self(torch.cat((input_ids, tail[:, :1]), -1)).logits[:, -1]
        return SimpleNamespace(sequences=torch.cat((input_ids, tail), -1), scores=(first, second))


class Adapter:
    def __init__(self):
        self.model = TinyHead()
        self.device = "cpu"
        self.pad_id = 0
        self.eos_ids = {4}
        self.processor = SimpleNamespace(tokenizer=SimpleNamespace(bos_token_id=1))

    @property
    def forward_calls(self):
        return self.model.forward_calls

    def _device_inputs(self, prepared):
        return {k: v.clone() for k, v in prepared["inputs"].items()}

    def _reset_positions(self):
        pass

    def logprobs(self, prepared, tokens, require_grad=False):
        result = []
        for i, token in enumerate(tokens):
            ids = torch.cat(
                (prepared["inputs"]["input_ids"], torch.tensor([tokens[:i]], dtype=torch.long)), -1
            )
            output = self.model(ids)
            result.append(output.logits[0, -1].log_softmax(-1)[token])
        return torch.stack(result)


def prepared(image=False):
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "mm_token_type_ids": torch.zeros(1, 3, dtype=torch.long),
    }
    if image:
        inputs.update(
            pixel_values=torch.arange(8).reshape(4, 2).float(),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
        )
    return {"inputs": inputs, "audit": {"prompt_token_count": 3}}


def runtime_fixture():
    adapter = Adapter()

    class State:
        def capture(self, metadata):
            return {
                "weight": adapter.model.weight.detach().clone(),
                "rng": torch.get_rng_state(),
                "training": adapter.model.training,
            }

        def restore(self, value):
            with torch.no_grad():
                adapter.model.weight.copy_(value["weight"])
            torch.set_rng_state(value["rng"])
            adapter.model.train(value["training"])

    return {
        "adapter": adapter,
        "state": State(),
        "state_guard": lambda: tuple(adapter.model.weight.detach().tolist()),
        "identity": {"execution_kind": "CPU_TINY_FIXTURE"},
        "parity_tolerances": {
            "mean_abs_token_logp": 1e-5,
            "max_abs_token_logp": 1e-5,
            "max_abs_sequence_logp": 1e-4,
        },
    }


def samples_fixture(adapter):
    prompts = {"text": prepared(), "image": prepared(True)}
    samples = []
    for pid, p in prompts.items():
        for i in range(8):
            samples.append(
                {
                    "prompt_id": pid,
                    "sample_key": f"{pid}-{i}",
                    "token_ids": [2, 4],
                    "behavior_token_logprobs": adapter.logprobs(p, [2, 4]).detach().tolist(),
                }
            )
    return prompts, samples


def test_teacher_batch_exact_tokens_eos_and_right_padding():
    adapter = Adapter()
    p = prepared(True)
    tokens = [[2, 4], [3, 2, 4]]
    original = {k: v.clone() for k, v in p["inputs"].items()}
    inputs = bench.batch_inputs(adapter, p, batch_size=2, completions=tokens)
    assert inputs["input_ids"].shape == (2, 6)
    assert inputs["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]
    assert inputs["image_grid_thw"].shape == (2, 3) and inputs["pixel_values"].shape == (8, 2)
    actual = bench.teacher_scores(adapter, p, tokens)
    expected = [adapter.logprobs(p, t).detach().tolist() for t in tokens]
    assert bench.parity(actual, expected, runtime_fixture()["parity_tolerances"])["passed"]
    for key in original:
        torch.testing.assert_close(original[key], p["inputs"][key])
    assert 2 in adapter.model.batch_sizes


def test_reject_unknown_axes_and_nonpadding_after_eos():
    adapter = Adapter()
    p = prepared()
    p["inputs"]["position_ids"] = torch.ones(1, 3)
    with pytest.raises(ValueError, match="Unsupported"):
        bench.batch_inputs(adapter, p, batch_size=2)
    result = SimpleNamespace(
        sequences=torch.tensor([[1, 4, 2]]), scores=(torch.zeros(1, 5), torch.zeros(1, 5))
    )
    with pytest.raises(ValueError, match="after EOS"):
        bench.unpack_generation(
            result, prompt_length=1, batch_size=1, eos_ids={4}, pad_id=0, horizon=64
        )


def test_batch_generation_masks_keep_first_eos_and_exclude_only_padding():
    result = SimpleNamespace(
        sequences=torch.tensor([[1, 4, 0], [1, 2, 4]]),
        scores=(torch.zeros(2, 5), torch.zeros(2, 5)),
    )
    rows = bench.unpack_generation(
        result, prompt_length=1, batch_size=2, eos_ids={4}, pad_id=0, horizon=64
    )
    assert rows[0]["token_ids"] == [4] and rows[0]["action_mask"] == [True, False]
    assert rows[1]["token_ids"] == [2, 4] and rows[1]["action_mask"] == [True, True]


def test_real_batch_generation_preserves_masks_and_measures_same_tokens():
    runtime = runtime_fixture()
    adapter = runtime["adapter"]
    rows = bench.batch_generate(adapter, prepared(True), batch_size=8, seed=23)
    assert len(rows) == 8 and adapter.model.batch_sizes == [8, 8]
    assert all(r["action_mask"] == [True, True] and r["eos_seen"] for r in rows)
    assert bench.parity(
        [r["behavior_token_logprobs"] for r in rows],
        [adapter.logprobs(prepared(True), r["token_ids"]).detach().tolist() for r in rows],
        runtime["parity_tolerances"],
    )["passed"]


def test_full_fixture_profile_is_technical_only_and_preserves_rng(tmp_path):
    runtime = runtime_fixture()
    runtime["state_guard"] = lambda: (
        id(runtime["adapter"].model.weight),
        runtime["adapter"].model.weight._version,
    )
    prompts, samples = samples_fixture(runtime["adapter"])
    rng = torch.get_rng_state().clone()
    result = bench.run_profile(
        runtime,
        prompts,
        samples,
        out=tmp_path,
        deadline=time.perf_counter() + 30,
        generation_prompts=list(prompts),
    )
    assert len(result["cases"]) == 9
    assert all(c["status"] == "MEASURED_TECHNICAL_ONLY" for c in result["cases"])
    torch.testing.assert_close(torch.get_rng_state(), rng)
    for size in (1, 2, 4, 8):
        score = json.loads((tmp_path / f"teacher_batch_{size}.json").read_text())
        assert (
            score["sequences"] == 16
            and score["prefix_parity"]["passed"]
            and score["serial_teacher_parity"]["passed"]
        )
        assert len(score["measurements"]) == 16 // size
        generation = json.loads((tmp_path / f"generation_batch_{size}.json").read_text())
        assert generation["sequences"] == 16 and generation["production_path_enabled"] is False
    assert (
        json.loads((tmp_path / "PROFILE.json").read_text())["predictive_or_statistical_claim"]
        is False
    )


def test_one_oom_path_keeps_costs_and_other_sizes_continue(tmp_path, monkeypatch):
    runtime = runtime_fixture()
    prompts, samples = samples_fixture(runtime["adapter"])
    original = bench.teacher_scores

    def sometimes(adapter, p, completions):
        if len(completions) == 4:
            adapter.model.forward_calls += 1
            raise torch.OutOfMemoryError("injected tiny OOM")
        return original(adapter, p, completions)

    monkeypatch.setattr(bench, "teacher_scores", sometimes)
    result = bench.run_profile(
        runtime,
        prompts,
        samples,
        out=tmp_path,
        deadline=time.perf_counter() + 30,
        generation_prompts=list(prompts),
    )
    assert next(c for c in result["cases"] if c["name"] == "teacher_batch_4")["status"] == "OOM"
    assert result["cases"][-1]["status"] == "MEASURED_TECHNICAL_ONLY"
    failure = json.loads((tmp_path / "teacher_batch_4.json").read_text())
    assert failure["total_case_model_forward_calls"] == 1 and failure["total_case_seconds"] > 0


def test_dry_run_loads_no_runtime_and_creates_no_output(tmp_path):
    result = bench.main(
        [
            "--tasks",
            "unused",
            "--samples",
            "unused",
            "--origin-index",
            "0",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert result["status"] == "DRY_RUN_NO_MODEL"
    assert not (tmp_path / "out").exists()


def test_exhausted_deadline_calls_no_model_and_preserves_original(tmp_path):
    runtime = runtime_fixture()
    prompts, samples = samples_fixture(runtime["adapter"])
    before = runtime["adapter"].forward_calls
    result = bench.run_profile(
        runtime,
        prompts,
        samples,
        out=tmp_path,
        deadline=time.perf_counter() - 1,
        generation_prompts=list(prompts),
    )
    assert result["status"] == "TIME_BUDGET_EXHAUSTED_PARTIAL_TECHNICAL_ONLY"
    assert runtime["adapter"].forward_calls == before


def test_deadline_after_batch_return_keeps_all_unrescored_rows(tmp_path, monkeypatch):
    runtime = runtime_fixture()
    prompts, samples = samples_fixture(runtime["adapter"])
    clock = [0.0]
    monkeypatch.setattr(bench, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    original = bench.batch_generate

    def expire_after_eight(*args, **kwargs):
        rows = original(*args, **kwargs)
        if kwargs["batch_size"] == 8:
            clock[0] = 200.0
        return rows

    monkeypatch.setattr(bench, "batch_generate", expire_after_eight)
    result = bench.run_profile(
        runtime, prompts, samples, out=tmp_path, deadline=100.0, generation_prompts=list(prompts)
    )
    assert result["status"] == "TIME_BUDGET_EXHAUSTED_PARTIAL_TECHNICAL_ONLY"
    partial = json.loads((tmp_path / "generation_batch_8.json").read_text())
    assert partial["status"] == "TIME_BUDGET_EXHAUSTED" and partial["returned_sequences"] == 8
    assert all(
        row["prefix_score_status"] == "NOT_RUN"
        for row in partial["partial_observations"]["records"]
    )


def test_failed_runtime_load_writes_truthful_receipt(tmp_path, monkeypatch):
    from src.modeling_v3 import io
    from src.modeling_v4 import gpu_collect as gpu

    monkeypatch.setattr(bench.platform, "system", lambda: "Linux")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(io, "source_identity", lambda: {"fixture": True})
    monkeypatch.setattr(gpu, "verify_artifact_bindings", lambda _: None)
    samples = [{"prompt_id": f"p{i // 8}", "sample_key": str(i)} for i in range(32)]
    monkeypatch.setattr(gpu, "_rows", lambda _: iter(samples))

    def fail(*args, **kwargs):
        raise RuntimeError("fixture model load rejected")

    monkeypatch.setattr(gpu, "load_runtime", fail)
    tasks = {
        "source": {"fixture": True},
        "root": str(tmp_path / "production"),
        "config": {},
        "bindings": {"v3_probability_tolerances": {}},
        "inputs": {
            "train_prompts": [{"prompt_id": f"p{i}"} for i in range(4)],
            "bridge_origins": [{"fixture_origin": True}],
        },
    }
    tasks["task_list_hash"] = io.canonical_hash(tasks)
    task_path, raw_path = tmp_path / "tasks.json", tmp_path / "raw.json"
    task_path.write_text(json.dumps(tasks))
    raw_path.write_text(
        json.dumps({"count": 32, "identity": {"role": "train", "proposal": "ORIGIN"}})
    )
    out = tmp_path / "profile"
    with pytest.raises(RuntimeError, match="load rejected"):
        bench.main(
            [
                "--tasks",
                str(task_path),
                "--samples",
                str(raw_path),
                "--origin-index",
                "0",
                "--out",
                str(out),
                "--execute-gpu",
            ]
        )
    failed = json.loads((out / "FAILED.json").read_text())
    assert failed["status"] == "FAILED_TECHNICAL_ONLY" and failed["runtime_receipt_exists"] is False
    assert not (out / "COMPLETE.json").exists()
