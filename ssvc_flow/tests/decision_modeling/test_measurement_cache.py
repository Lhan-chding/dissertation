import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.decision_modeling.measurement import collect_stream, lr_contributions
from src.decision_modeling.score_cache import ScoreCache, score_key


def test_cache_process_and_disk_reuse_namespaces_and_no_mutable_alias(tmp_path):
    path = tmp_path / "cache.sqlite"
    key = score_key("weights", "input", [1, 4], eos_ids=[4])
    calls = []

    def compute():
        calls.append(1)
        return [-0.2, -0.3]

    with ScoreCache(path, namespace={"precision": "bf16"}) as cache:
        first = cache.score(key, length=2, compute=compute)
        first[0] = -999
        assert cache.score(key, length=2, compute=compute) == [-0.2, -0.3]
        assert len(calls) == 1 and cache.counters["memory_hits"] == 1
    with ScoreCache(path, namespace={"precision": "bf16"}) as cache:
        assert cache.score(key, length=2, compute=compute) == [-0.2, -0.3]
        assert cache.counters["persistent_hits"] == 1
    with ScoreCache(path, namespace={"precision": "fp32"}) as cache:
        cache.score(key, length=2, compute=compute)
    assert len(calls) == 2


def test_keys_bind_input_policy_path_and_full_tokens():
    base = score_key("weights", "a", [1, 4], eos_ids=[4])
    assert base != score_key("weights", "b", [1, 4], eos_ids=[4])
    assert base != score_key("other", "a", [1, 4], eos_ids=[4])
    assert base != score_key("weights", "a", [1, 4], eos_ids=[4], scoring_path="full")
    for tokens in ([1], [4, 1, 4], [], [True, 4]):
        with pytest.raises(ValueError):
            score_key("weights", "input", tokens, eos_ids=[4])


def test_nonfinite_score_does_not_poison_persistent_cache(tmp_path):
    with ScoreCache(tmp_path / "s.db", namespace={"v": 1}) as cache:
        with pytest.raises(ValueError):
            cache.score("key", length=1, compute=lambda: [float("nan")])
        assert cache.score("key", length=1, compute=lambda: [-1]) == [-1]


class FakeBackend:
    def __init__(self):
        self.cache = SimpleNamespace(namespace="fixed")
        self.calls = []

    def activate(self, policy):
        self.policy = policy

    def generate(self, prompt, *, seed):
        self.calls.append(seed)
        return {
            "token_ids": [1, 4],
            "event": "I",
            "seed": seed,
            "prompt_id": prompt["prompt_id"],
            "policy": self.policy,
        }


def test_nested_prefix_reuse_and_real_mixture_coins(tmp_path):
    backend = FakeBackend()
    kwargs = dict(out=tmp_path, role="mix", stream_id="independent")
    first = collect_stream(
        backend, [{"id": "a"}, {"id": "b"}], [{"prompt_id": "p"}], draws=32, **kwargs
    )
    second = collect_stream(
        backend, [{"id": "a"}, {"id": "b"}], [{"prompt_id": "p"}], draws=128, **kwargs
    )
    assert second[:32] == first and len(backend.calls) == 128
    assert set(r["proposal_source_index"] for r in first) == {0, 1}
    assert len({r["sample_id"] for r in second}) == 128
    # Identical complete tokens remain 128 independent statistical observations.
    assert len({tuple(r["token_ids"]) for r in second}) == 1
    p = tmp_path / "p/0000_0032.json"
    corrupt = json.loads(p.read_text())
    corrupt["rows"][0]["event"] = "X"
    p.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match="changed"):
        collect_stream(
            backend, [{"id": "a"}, {"id": "b"}], [{"prompt_id": "p"}], draws=32, **kwargs
        )


def test_stable_signed_lr_and_alias_zero():
    sample = {
        "sample_id": "a",
        "event": "W",
        "scoring_usable": True,
        "generation_sequence_logp": -10000,
    }
    left = {"sample_id": "a", "sequence_logp": -10000}
    right = {"sample_id": "a", "sequence_logp": -9999}
    result = lr_contributions([sample], [left], [right], proposal="MIX")
    np.testing.assert_allclose(result[0], [0, 0, 2 * np.tanh(0.5), 0])
    assert not lr_contributions([sample], [left], [left], proposal="ORIGIN").any()
    with pytest.raises(ValueError, match="disabled"):
        lr_contributions([sample | {"scoring_usable": False}], [left], [right], proposal="MIX")
