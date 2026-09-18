"""Real small Torch execution only; this fixture is not scientific GPU evidence."""

import copy
import importlib.util
from pathlib import Path

from src.modeling_v3.io import canonical_hash, sha256_file
from src.modeling_v4 import gpu_collect as gpu
from src.modeling_v4.data_adapter import CONTRASTS


def test_old_candidates_new_preview_stream_actual_collection_and_diagnostics(tmp_path):
    fixture_path = Path(__file__).with_name("test_gpu_collect.py")
    spec = importlib.util.spec_from_file_location("preview_tiny_runtime_fixture", fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    old_runtime, adapter_module = fixture.tiny_runtime()
    old_runtime["identity"]["source_hash"] = "old-collection-source"
    old_origin_id = "map_41001_X_BASE_32"
    origin = old_runtime["state"].capture({"step": 32})
    train_prompts = adapter_module.fake_prompts(4, split="train")
    old_root = tmp_path / "original_campaign"
    original = gpu.run_forks(
        old_runtime,
        origin,
        {
            "origin_id": old_origin_id,
            "banks": [
                {
                    "bank_id": "calibration_000",
                    "role": "calibration",
                    "prompt_ids": [p["prompt_id"] for p in train_prompts],
                }
            ],
        },
        train_prompts,
        out=old_root / "forks",
        bridge=True,
    )
    bank = original["banks"][0]
    assert not bank["contrasts"][CONTRASTS[0][0]]["exact_inference_alias"]
    original_hash = canonical_hash(original)
    original_bytes = {str(p): sha256_file(p) for p in old_root.rglob("*") if p.is_file()}

    runtime, _ = fixture.tiny_runtime()
    runtime["identity"]["source_hash"] = "new-measurement-source"
    runtime["observation_score_mode"] = "uncached_prefix_recompute"
    runtime["state"].restore(runtime["checkpoint_cache"].load(original["origin_policy"]))
    observation_id = f"preview:frozen-plan:{old_origin_id}"
    forks = {**copy.deepcopy(original), "origin_id": observation_id}
    probes = adapter_module.fake_prompts(1)
    response_root = tmp_path / "preview" / "prompts" / "00" / "response"
    response = gpu.collect_response_map(
        {}, runtime, forks, probes, out=response_root, draws=8, reference_draws=8, bridge=True
    )
    assert runtime["identity"]["execution_kind"] == "CPU_FAKE_TORCH"
    assert forks["banks"][0]["role"] == "calibration"
    assert response["scientific_status"] == "NOT_CERTIFIED"
    assert (response_root / "COMPLETE.json").exists()
    assert response["direct_baseline"] == "DIRECT_LR_ORIGIN_INDEPENDENT_PACKET"
    for policy in response["policies"].values():
        assert policy["checkpoint"]["identity"]["origin_id"] == old_origin_id
        assert policy["checkpoint"]["identity"]["source_hash"] == "old-collection-source"

    packets = [response[k] for k in ("work", "reference", "direct_work")]
    for check in response["pair_checks"]:
        packets.append(check["mixture"])
        packets.extend(check["direct"].values())
    assert response["pair_checks"]
    streams, keys, seeds = set(), set(), set()
    for packet in packets:
        identity = packet["identity"]
        assert identity["origin_id"] == observation_id
        assert identity["runtime"]["source_hash"] == "new-measurement-source"
        assert observation_id in identity["rng_namespace"]
        assert identity["rng_namespace"] not in streams
        streams.add(identity["rng_namespace"])
        rows = list(gpu._rows(packet))
        assert len(rows) == packet["count"] == 8
        for row in rows:
            assert row["sample_key"] not in keys
            assert row["sample_seed"] not in seeds
            assert row["role"] == identity["role"]
            assert row["origin_id"] == observation_id
            keys.add(row["sample_key"])
            seeds.add(row["sample_seed"])
        assert all(Path(chunk["path"]).is_file() for chunk in packet["chunks"])

    diagnostic = gpu.pair_observation_diagnostics(response)
    assert diagnostic["all_finite"]
    assert diagnostic["scientific_status"] == "NOT_CERTIFIED"
    expected = {check["contrast_id"]: check for check in response["pair_checks"]}
    for unit in diagnostic["units"]:
        check = expected[unit["contrast_id"]]
        assert unit["bank_id"] == "calibration_000"
        assert unit["left_candidate_id"] == check["left_candidate_id"]
        assert unit["right_candidate_id"] == check["right_candidate_id"]
        assert unit["conditional_support_bounds"]["valid_conditional_bound"]
    primary = next(unit for unit in diagnostic["units"] if unit["contrast_id"] == CONTRASTS[0][0])
    assert primary["endpoint_count_intervals"]["single_event_coverage_at_least"] == 0.95

    before = runtime["adapter"].generation_calls
    resumed = gpu.collect_response_map(
        {},
        runtime,
        forks,
        probes,
        out=response_root,
        draws=8,
        reference_draws=8,
        bridge=True,
        resume=True,
    )
    assert runtime["adapter"].generation_calls == before
    assert canonical_hash(resumed) == canonical_hash(response)
    assert canonical_hash(original) == original_hash
    assert {str(p): sha256_file(p) for p in old_root.rglob("*") if p.is_file()} == original_bytes
