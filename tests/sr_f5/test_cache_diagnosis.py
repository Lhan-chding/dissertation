"""CPU-only tests for replay identity and unmodified raw-score observation."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def diagnostic():
    spec = importlib.util.spec_from_file_location(
        "diagnose_cache", Path(__file__).parents[2] / "scripts/sr_f5/diagnose_cache.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(diagnostic):
    rows = [{"tokens": [2] * 163}, {"tokens": [1] * 163}]
    observer = diagnostic.ForcedReplay(torch, rows, 3)
    ids = torch.tensor([[4, 5, 6], [4, 5, 6]])
    kwargs = dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        position_ids=torch.tensor([[0, 1, 2], [0, 1, 2]]),
        use_cache=True,
    )
    observer.before_forward(None, (), kwargs)
    logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0]], [[4.0, 3.0, 2.0, 1.0]]])
    observer.after_forward(None, (), kwargs, SimpleNamespace(logits=logits))
    return observer, ids, logits


def test_force_does_not_modify_scores_or_raw_logits(diagnostic):
    observer, ids, logits = prepare(diagnostic)
    scores = logits[:, -1, :]
    original = logits.clone()
    forced = observer(ids, scores)
    assert torch.equal(logits, original)
    assert torch.equal(scores, original[:, -1, :])
    assert forced.argmax(-1).tolist() == [2, 1]
    assert observer.score_checks == [{"position": 0, "raw_logits_scores_equal": True}]
    expected = scores.log_softmax(-1)[[0, 1], [2, 1]].tolist()
    assert observer.probabilities == [[expected[0]], [expected[1]]]
    assert forced.data_ptr() != scores.data_ptr()


def test_transformed_scores_fail_before_forcing(diagnostic):
    observer, ids, logits = prepare(diagnostic)
    scores = logits[:, -1, :].clone()
    scores[0, 0] += 1
    with pytest.raises(PermissionError, match="processor changed"):
        observer(ids, scores)
    assert observer.probabilities == [[], []]


def test_saved_prefix_deviation_fails(diagnostic):
    observer, ids, logits = prepare(diagnostic)
    observer(ids, logits[:, -1, :])
    wrong = torch.cat([ids, torch.tensor([[0], [1]])], dim=1)
    observer.before_forward(None, (), {"input_ids": wrong[:, -1:]})
    observer.after_forward(None, (), {}, SimpleNamespace(logits=logits))
    with pytest.raises(PermissionError, match="departed"):
        observer(wrong, logits[:, -1, :])


def test_hook_records_boundary_and_accounting(diagnostic):
    counts = []
    observer = diagnostic.ForcedReplay(
        torch, [{"tokens": [2] * 163}], 3, reserve=lambda *a, **kw: counts.append((a, kw))
    )
    cache = SimpleNamespace(get_seq_length=lambda: 3)
    observer.before_forward(
        None,
        (),
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "cache_position": torch.tensor([0, 1, 2]),
            "past_key_values": cache,
            "use_cache": True,
        },
    )
    snapshot = observer.forwards[0]
    assert snapshot["phase"] == "prefill"
    assert snapshot["inputs"]["cache_position"]["values"] == [0, 1, 2]
    assert snapshot["inputs"]["past_key_values"]["sequence_length"] == 3
    assert counts[0][0] == ("extra_forward_sequences", 1)
    assert counts[0][1]["purpose"] == "FORCED_TOKEN_REPLAY_ONLY"


def test_replay_config_keeps_distribution_and_rejects_length_eos(diagnostic):
    config = dict(
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        do_sample=True,
        max_new_tokens=1024,
        forced_eos_token_id=None,
        use_cache=True,
    )
    replay = diagnostic.replay_config(config)
    assert replay.do_sample is False and replay.max_new_tokens == 163
    assert replay.use_cache is True
    assert config["max_new_tokens"] == 1024 and config["do_sample"] is True
    with pytest.raises(PermissionError, match="forced EOS"):
        diagnostic.replay_config(dict(config, forced_eos_token_id=2))


def test_repeat_model_hooks_preserve_base_logits(diagnostic):
    class Model(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None):
            return SimpleNamespace(logits=torch.ones(input_ids.shape[0], 1, 8))

    model = Model()
    observer = diagnostic.ForcedReplay(torch, [{"tokens": [2] * 163}], 3)
    before = model.register_forward_pre_hook(observer.before_forward, with_kwargs=True)
    after = model.register_forward_hook(observer.after_forward, with_kwargs=True)
    output = model(input_ids=torch.tensor([[1, 2, 3]]))
    forced = observer(torch.tensor([[1, 2, 3]]), output.logits[:, -1, :])
    before.remove()
    after.remove()
    assert forced.argmax(-1).item() == 2
    assert torch.equal(output.logits, torch.ones_like(output.logits))
    assert len(observer.forwards) == 1


def test_original_cache_disabled_rejected(diagnostic):
    with pytest.raises(PermissionError, match="native cache"):
        diagnostic.replay_config(dict(temperature=1.0, top_p=1.0, top_k=0, use_cache=False))


def test_language_position_hook_records_actual_positions(diagnostic):
    observer, _ids, _logits = prepare(diagnostic)
    observer.before_language_forward(None, (), {"position_ids": torch.tensor([[[2, 3, 4]]])})
    snapshot = observer.forwards[0]
    assert snapshot["language_position_ids_exposed"]
    assert snapshot["language_model_inputs"]["position_ids"]["values"] == [[[2, 3, 4]]]
