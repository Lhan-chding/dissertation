import copy

import numpy as np
import pytest

from src.modeling_v4.response_fit import (
    _contract_derivative,
    _effective_contrast,
    _packet,
    _reference_statistics,
    _require_server_cpu,
)


def packet_fixture():
    samples = []
    for d, event in enumerate(["X", "S", "W", "I"] * 2):
        samples.append(
            {
                "sample_id": str(d),
                "prompt_id": "p",
                "token_ids": [3, 2],
                "input_hash": "input",
                "role": "work",
                "rng_namespace": "work",
                "runtime_identity": {"model": "fixture"},
                "generation_sequence_logp": -3.0,
                "category": event,
            }
        )
    policies = {
        n: {"candidate_id": n, "inference_fingerprint": n}
        for n in ["joint_0", "joint_1", "no_x_off_1"]
    }
    scores = {
        n: [
            {
                **{
                    k: s[k]
                    for k in [
                        "sample_id",
                        "prompt_id",
                        "token_ids",
                        "input_hash",
                        "role",
                        "rng_namespace",
                        "runtime_identity",
                    ]
                },
                "inference_fingerprint": n,
                "token_logprobs": [-1.0, -2.0 + v],
                "sequence_logp": -3.0 + v,
            }
            for s in samples
        ]
        for n, v in [("joint_0", 0.0), ("joint_1", 0.1), ("no_x_off_1", -0.1)]
    }
    return samples, scores, policies


def test_identified_packet_preserves_all_three_correlated_contrasts():
    samples, scores, policies = packet_fixture()
    batch = _packet(samples, scores, policies)
    assert batch.contributions.shape == (8, 3, 4)
    np.testing.assert_allclose(
        batch.contributions[:, 2], batch.contributions[:, 0] - batch.contributions[:, 1]
    )
    scores["joint_1"][0]["token_ids"] = [9, 2]
    with pytest.raises(ValueError, match="identity"):
        _packet(samples, scores, policies)


def test_alias_uses_exact_fingerprint_and_rejects_inconsistent_scores():
    samples, scores, policies = packet_fixture()
    policies["joint_1"]["inference_fingerprint"] = "joint_0"
    for s in scores["joint_1"]:
        s["inference_fingerprint"] = "joint_0"
    with pytest.raises(ValueError, match="alias"):
        _packet(samples, scores, policies)
    scores["joint_1"] = copy.deepcopy(scores["joint_0"])
    assert np.all(_packet(samples, scores, policies).contributions[:, 0] == 0)


def test_reference_noise_and_unresolved_zero_variance_never_fabricate_precision():
    samples, scores, policies = packet_fixture()
    result = _reference_statistics(_packet(samples, scores, policies))
    assert result["se"].shape == (3, 4) and not result["resolved"].any()
    samples = [{**s, "category": "I"} for s in samples]
    result = _reference_statistics(_packet(samples, scores, policies))
    assert not result["resolved"].any()
    policies["joint_1"]["inference_fingerprint"] = "joint_0"
    scores["joint_1"] = copy.deepcopy(scores["joint_0"])
    result = _reference_statistics(_packet(samples, scores, policies))
    assert result["resolved"][0].all() and np.all(result["se"][0] == 0)


def test_effective_factors_keep_actual_scaling_and_all_coordinates():
    base = {
        "m.lora_A.default.weight": np.array([[1.0, 2.0]]),
        "m.lora_B.default.weight": np.array([[3.0], [4.0]]),
    }
    candidate = {**base, "m.lora_B.default.weight": base["m.lora_B.default.weight"] + 1}
    result = _effective_contrast(candidate, base, {"m": 2.0}, "base")
    assert result["modules"]["m"]["scaling"] == 2.0
    with pytest.raises(ValueError):
        _effective_contrast(
            {**candidate, "extra": np.zeros(1)}, {**base, "extra": np.zeros(1)}, {"m": 2.0}, "base"
        )


def test_full_gradient_contraction_is_group_only_and_checks_all_parameters():
    gradient = {
        "parameter_order": ["a"],
        "grouped_gradients": {"a": np.ones((2, 4, 3))},
        "prompt_gradients": {"a": np.ones((1, 4, 3))},
        "groups": [("f", "i"), ("g", "i")],
        "prompt_subset": ["p"],
        "expansion_point": "ORIGIN",
        "reference_labels_read": False,
    }
    result = _contract_derivative(gradient, {"a": np.array([1.0, 2.0, 3.0])})
    assert result["group"].shape == (2, 4) and np.all(result["group"] == 6)
    assert result["prompt"].shape == (1, 4)
    with pytest.raises(ValueError):
        _contract_derivative(gradient, {"b": np.ones(3)})


def test_real_entry_requires_linux_slurm_cpu(monkeypatch):
    monkeypatch.setattr("src.modeling_v4.response_fit.platform.system", lambda: "Darwin")
    with pytest.raises(RuntimeError):
        _require_server_cpu()
    monkeypatch.setattr("src.modeling_v4.response_fit.platform.system", lambda: "Linux")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_GPUS", "0")
    with pytest.raises(RuntimeError):
        _require_server_cpu()


def completed_fixture(tmp_path):
    import torch

    from src.modeling_v3.io import canonical_hash, sha256_file
    from src.modeling_v4 import gpu_collect as gpu
    from src.optimizer_fork import state_hash

    runtime = {
        "model_hash": "fixture-model",
        "execution_kind": "CPU_FIXTURE",
        "config_hash": "fixture",
        "source_hash": "fixture",
    }
    scene = {
        "truth_world": [1, 2, 3, 4],
        "observed_world": [1, 2, 3, 9],
        "operation": "sum4",
        "cue": {"family": "trend"},
    }
    probes = [
        {
            "prompt_id": f"p{i}",
            "base_scene_id": f"s{i}",
            "family": f"f{i}",
            "interface": "i",
            "scene": scene,
        }
        for i in range(2)
    ]
    task = {
        "id": "map_fixture",
        "seed": 41001,
        "arm": "X_BASE",
        "step": 32,
        "draws": 8,
        "calibration_banks": 2,
        "query_banks": 1,
    }
    names = ["m.lora_A.default.weight", "m.lora_B.default.weight"]

    def policy(name, v):
        parameters = {names[0]: torch.tensor([[1.0, v]]), names[1]: torch.tensor([[v], [1.0]])}
        state = {"parameters": parameters, "buffers": {}}
        spec = gpu._save_state(tmp_path / (name + ".pt"), state, {**runtime, "candidate_id": name})
        fp = gpu._fingerprint(state, runtime)
        return {
            "checkpoint": spec,
            "candidate_id": name,
            "inference_fingerprint": fp,
            "module_scalings": {"m": 2.0},
            "parameter_layout": [
                {"name": n, "shape": list(v.shape)} for n, v in sorted(parameters.items())
            ],
        }, parameters

    origin, _ = policy("origin", 1.0)
    banks = []
    for b in range(3):
        role = "calibration" if b < 2 else "query"
        bank = {
            "bank_id": f"{role}_{b}",
            "role": role,
            "prompt_ids": [f"train{b}"],
            "policies": {},
            "contrasts": {},
        }
        parameters = {}
        for i, name in enumerate(["joint_0", "joint_1", "no_x_off_1"]):
            bank["policies"][name], parameters[name] = policy(
                f"b{b}_{name}", 1.0 + 0.01 * (i + 1) * (b + 1)
            )
        from src.modeling_v4.data_adapter import CONTRASTS

        for target, left, right in CONTRASTS:
            delta = {n: parameters[left][n].double() - parameters[right][n].double() for n in names}
            bank["contrasts"][target] = {
                "left": bank["policies"][left]["checkpoint"],
                "right": bank["policies"][right]["checkpoint"],
                "parameter_order": names,
                "delta_hash": state_hash(delta),
                "exact_inference_alias": False,
            }
        banks.append(bank)
    forks = {"origin_id": task["id"], "origin_policy": origin, "banks": banks}

    def actions(role):
        namespace = "fixture:" + role
        identity = {
            "origin_id": task["id"],
            "role": role,
            "draws": 8,
            "proposal": "ORIGIN",
            "policies": [origin],
            "prompt_hash": canonical_hash(probes),
            "rng_namespace": namespace,
            "runtime": runtime,
        }

        def make(i):
            p, d = probes[i // 8], i % 8
            e = d % 4
            key = canonical_hash([namespace, p["prompt_id"], d])
            tokens = [e + 3, 2]
            return {
                "sample_id": key,
                "sample_key": key,
                "prompt_id": p["prompt_id"],
                "base_scene_id": p["base_scene_id"],
                "origin_id": task["id"],
                "draw_index": d,
                "role": role,
                "rng_namespace": namespace,
                "runtime_identity": canonical_hash(runtime),
                "proposal_fingerprint": origin["inference_fingerprint"],
                "proposal_policy_id": origin["candidate_id"],
                "sample_seed": int(key[:16], 16) % (2**63),
                "token_ids": tokens,
                "eos_token_ids": [2],
                "completion_length": 2,
                "stop_reason": "eos",
                "eos_seen": True,
                "truncated": False,
                "behavior_token_logprobs": [-1.0, -2.0],
                "generation_sequence_logp": -3.0,
                "raw_completion": ["[1,2,3,4]", "[0,3,3,4]", "[1,2,3,5]", "bad"][e],
                "category": ["X", "S", "W", "I"][e],
                "event_onehot": [int(j == e) for j in range(4)],
                "action_mask": [1, 1],
                "input_hash": p["prompt_id"],
            }

        return gpu._chunks(tmp_path / role, identity, make, count=16, resume=False)

    response = {
        "origin_id": task["id"],
        "work": actions("work"),
        "reference": actions("reference"),
        "direct_work": actions("direct_work"),
    }
    for role, field in [
        ("work", "work_scores"),
        ("reference", "reference_scores"),
        ("direct_work", "direct_scores"),
    ]:
        response[field] = {}
        samples = response[role]
        rows = list(gpu._rows(samples))
        for b, bank in enumerate(banks):
            if role != "work" and bank["role"] != "query":
                continue
            for j, p in enumerate(bank["policies"].values()):
                ident = {
                    "sample_identity": canonical_hash(samples["identity"]),
                    "sample_chunks": samples["chunks"],
                    "policy": p,
                    "runtime": runtime,
                }

                def make(i, policy=p, effect=(j + 1) * (b + 1) * 0.01, rows=rows):
                    r = rows[i]
                    v = effect * (1 + r["draw_index"] % 4)
                    return {
                        **{
                            k: r[k]
                            for k in [
                                "sample_id",
                                "prompt_id",
                                "token_ids",
                                "input_hash",
                                "role",
                                "rng_namespace",
                                "runtime_identity",
                            ]
                        },
                        "inference_fingerprint": policy["inference_fingerprint"],
                        "token_logprobs": [-1.0, -2.0 + v],
                        "sequence_logp": -3.0 + v,
                    }

                response[field][p["candidate_id"]] = {
                    "score_receipt": gpu._chunks(
                        tmp_path / f"{role}_{p['candidate_id']}",
                        ident,
                        make,
                        count=16,
                        resume=False,
                    )
                }
    gradient = {
        "parameter_order": names,
        "grouped_gradients": {
            n: torch.ones((2, 4, *({names[0]: (1, 2), names[1]: (2, 1)}[n]))) for n in names
        },
        "prompt_gradients": {
            n: torch.ones((1, 4, *({names[0]: (1, 2), names[1]: (2, 1)}[n]))) for n in names
        },
        "groups": [("f0", "i"), ("f1", "i")],
        "prompt_subset": ["p0"],
        "expansion_point": "ORIGIN",
        "reference_labels_read": False,
        "sample_ids": [r["sample_id"] for r in gpu._rows(response["work"])],
        "rng_stream_ids": ["fixture:work"],
    }
    path = tmp_path / "gradient.pt"
    torch.save(gradient, path)
    receipt = {
        "status": "MEASURED",
        "origin_id": task["id"],
        "reference_used": False,
        "artifact": {"path": str(path), "sha256": sha256_file(path)},
    }
    config = {
        "models": {
            "compression_ranks": [2, "FULL"],
            "ridge_alpha": [1e-8, 1e-5, 0.01],
            "rbf_bandwidth_multipliers": [0.25, 1.0, 4.0],
        },
        "observations": {"reference_primary_n": 8},
    }
    return config, task, forks, response, receipt, probes


@pytest.mark.parametrize("selected", [False, True])
def test_completed_map_fits_then_opens_reference_and_keeps_group_scope(
    tmp_path, monkeypatch, selected
):
    import json

    from src.modeling_v3.io import verify_manifest
    from src.modeling_v4 import response_fit as fit

    args = completed_fixture(tmp_path)
    out = tmp_path / "derived"
    out.mkdir()
    original = fit._action_rows

    def checked(*a, **kw):
        if kw["role"] == "reference":
            assert (out / "PREDICTIONS.parquet").exists()
        return original(*a, **kw)

    monkeypatch.setattr(fit, "_action_rows", checked)
    selection = (
        {
            "gradient_reference": "FULL_SCORE_JVP",
            "primary_models": [
                {
                    "id": "frozen-full",
                    "model": "FULL_DUAL_RIDGE",
                    "representation": "R2",
                    "settings": {},
                },
                {
                    "id": "frozen-j",
                    "model": "FULL_SCORE_JVP",
                    "representation": "R2",
                    "settings": {},
                },
            ],
        }
        if selected
        else None
    )
    result = fit._fit_completed_map(
        *args,
        out,
        binding={"fixture": True},
        alpha=1e-5,
        bandwidth_multiplier=1.0,
        methods=("RAW4",),
        selection=selection,
        completed={},
    )
    assert result["status"] == "COMPLETED" and verify_manifest(out)["status"] == "COMPLETE"
    metrics = json.loads((out / "evaluation" / "METRICS.json").read_text())
    assert {
        r["evaluation_unit"] for r in metrics["summary"] if r["method"] == "FULL_SCORE_JVP"
    } == {"prompt", "semantic_group"}
    assert any(r["method"] == "DIRECT_MEASURE" for r in metrics["summary"])
    assert result["query_response_used_for_fit"] is False and result["new_actions"] == 0
    if selected:
        import pyarrow.parquet as pq

        rows = pq.read_table(out / "PREDICTIONS.parquet").to_pylist()
        chosen = [
            r for r in rows if r["design_id"] == "frozen-j" and r["evaluation_unit"] == "prompt"
        ]
        assert {r["prediction_status"] for r in chosen} == {"PREDICTED", "UNKNOWN"}
        assert all(
            json.loads(r["model_metadata_json"])["frozen_model_id"] == "frozen-j" for r in chosen
        )


def test_signature_primary_cannot_relabel_an_actual_linear_lock(tmp_path):
    from src.modeling_v3.io import atomic_json
    from src.modeling_v4 import response_fit as fit

    path = tmp_path / "SIGNATURE_SELECTION.json"
    atomic_json(path, {"models": [{"id": "RAW4/linear", "spec": {"model": "FULL_DUAL_RIDGE"}}]})
    primary = {
        "id": "false-mlp",
        "representation": "R4",
        "model": "NONLINEAR_SIGNATURE_RESIDUAL",
        "settings": {
            "signature_model_selection": fit._binding(path),
            "signature_model_id": "RAW4/linear",
        },
    }
    with pytest.raises(ValueError, match="actual signature model type"):
        fit._frozen_signature_rows(
            {"primary_models": [primary]}, {"signature": {}}, {}, [], [], tmp_path
        )


def test_streamed_coordinates_match_dense_raw_and_effective_kernels(tmp_path):
    from src.modeling_v4.full_kernels import effective_weight_gram
    from src.modeling_v4.gpu_collect import CheckpointCache
    from src.modeling_v4.response_fit import _endpoint_memmap, _precomputed_inputs, _read_gradient

    _, task, forks, response, receipt, probes = completed_fixture(tmp_path)
    from src.modeling_v4.response_fit import _action_rows

    work = _action_rows(
        response["work"],
        probes,
        role="work",
        origin_id=task["id"],
        policy_fingerprint=forks["origin_policy"]["inference_fingerprint"],
    )
    gradient = _read_gradient(receipt, task["id"], work)
    data, layout, scalings = _endpoint_memmap(
        forks,
        forks["banks"],
        tmp_path / "coordinates.npy",
        CheckpointCache(),
        response["work"]["identity"]["runtime"],
    )
    result = _precomputed_inputs(
        data,
        layout,
        scalings,
        gradient,
        candidate="joint_1",
        baseline="joint_0",
        calibration_count=2,
        base_id="fixture-model",
    )
    from src.modeling_v4.response_fit import compute_response_geometry

    paired = compute_response_geometry(
        data[:, :2],
        layout,
        scalings,
        gradient,
        candidate="candidate",
        baseline="baseline",
        calibration_count=2,
        base_id="fixture-model",
        endpoint_names=("baseline", "candidate"),
    )
    np.testing.assert_array_equal(paired["raw"], result["raw"])
    np.testing.assert_array_equal(paired["effective"], result["effective"])
    delta = data[:, 1].astype(float) - data[:, 0].astype(float)
    np.testing.assert_allclose(result["raw"], delta @ delta.T, atol=1e-15)
    effective = []
    for i in range(3):
        endpoints = [
            {
                n: np.array(data[i, endpoint, v["start"] : v["stop"]]).reshape(v["shape"])
                for n, v in layout.items()
            }
            for endpoint in (1, 0)
        ]
        effective.append(_effective_contrast(*endpoints, scalings, "fixture-model"))
    np.testing.assert_allclose(result["effective"], effective_weight_gram(effective), atol=1e-15)
    direction = {n: delta[2, v["start"] : v["stop"]].reshape(v["shape"]) for n, v in layout.items()}
    np.testing.assert_allclose(
        result["derivative"]["group"][0], _contract_derivative(gradient, direction)["group"]
    )
    forks["banks"][0]["contrasts"]["joint_1_minus_joint_0"]["exact_inference_alias"] = True
    with pytest.raises(ValueError, match="Alias"):
        _endpoint_memmap(
            forks,
            forks["banks"],
            tmp_path / "bad.npy",
            CheckpointCache(),
            response["work"]["identity"]["runtime"],
        )


def test_corrupted_chunks_and_query_train_overlap_are_rejected(tmp_path):
    from src.modeling_v4 import response_fit as fit

    config, task, forks, response, gradient, probes = completed_fixture(tmp_path)
    path = response["work"]["chunks"][0]["path"]
    with open(path, "ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="bytes"):
        list(fit._verified_rows(response["work"]))
    forks["banks"][2]["prompt_ids"] = forks["banks"][0]["prompt_ids"]
    out = tmp_path / "failed"
    out.mkdir()
    with pytest.raises(ValueError, match="partition"):
        fit._fit_completed_map(
            config,
            task,
            forks,
            response,
            gradient,
            probes,
            out,
            binding={"fixture": True},
            alpha=1e-5,
            bandwidth_multiplier=1.0,
            methods=("RAW4",),
        )
    assert not (out / "COMPLETE.json").exists()


def test_calibration_export_never_reads_query_scores_or_reference(tmp_path, monkeypatch):
    import json

    from src.modeling_v3.io import atomic_json, canonical_hash, sha256_file, verify_manifest
    from src.modeling_v4 import response_fit as fit

    config, task, forks, response, _, probes = completed_fixture(tmp_path)
    config.update(qwen={"seed_roles": {"development": [41001]}}, observations={"methods": ["RAW4"]})
    task.update(kind="map", prompts=2)
    source = {"sha256": "a" * 64}
    monkeypatch.setattr(fit, "_require_server_cpu", lambda: None)
    monkeypatch.setattr(fit, "source_identity", lambda: source)
    task_root = tmp_path / "campaign" / "tasks" / task["id"]
    task_root.mkdir(parents=True)
    atomic_json(task_root / "forks.json", forks)
    atomic_json(task_root / "response.json", response)

    def bind(p):
        return {"path": str(p), "sha256": sha256_file(p)}

    atomic_json(
        task_root / "COMPLETE.json",
        {
            "status": "COMPLETED",
            "task": task,
            "execution_kind": "REAL_CUDA_MODEL",
            "origin_id": task["id"],
            "config_hash": canonical_hash(config),
            "source_hash": source["sha256"],
            "forks": bind(task_root / "forks.json"),
            "response": bind(task_root / "response.json"),
        },
    )
    plan = {
        "tasks": [task],
        "config": config,
        "source": source,
        "root": str(tmp_path / "campaign"),
        "inputs": {"panels": {"observation": probes}},
    }
    plan["task_list_hash"] = canonical_hash(plan)
    atomic_json(tmp_path / "tasks.json", plan)
    original = fit._score_rows

    def only_calibration(receipt, *args):
        assert not receipt["identity"]["policy"]["candidate_id"].startswith("b2_")
        assert receipt["identity"]["sample_identity"] == canonical_hash(
            response["work"]["identity"]
        )
        return original(receipt, *args)

    monkeypatch.setattr(fit, "_score_rows", only_calibration)
    with open(response["reference"]["chunks"][0]["path"], "ab") as stream:
        stream.write(b"never opened in calibration export")
    output = fit.export_calibration_labels(
        tmp_path / "tasks.json", task_id=task["id"], out=tmp_path / "labels"
    )
    receipt = json.loads((tmp_path / "labels" / "LABELS.json").read_text())
    assert (
        output["status"] == "COMPLETE"
        and verify_manifest(tmp_path / "labels")["status"] == "COMPLETE"
    )
    assert len(receipt["units"]) == 6 and receipt["reference_labels_read"] is False
    assert np.load(tmp_path / "labels" / "LABELS.npz")["RAW4"].shape == (6, 2, 4)


@pytest.mark.parametrize("subset", [False, True])
def test_actual_mix_fallback_replaces_only_affected_contrast_and_keeps_originals(tmp_path, subset):
    from src.modeling_v3.io import canonical_hash
    from src.modeling_v4 import gpu_collect as gpu
    from src.modeling_v4 import response_fit as fit

    _, task, forks, response, _, probes = completed_fixture(tmp_path)
    bank = forks["banks"][-1]
    policies = bank["policies"]
    left, right = policies["joint_1"], policies["joint_0"]
    original_rows = list(gpu._rows(response["reference"]))
    namespace = "independent:mix:jointcontrast"
    identity = {
        **response["reference"]["identity"],
        "rng_namespace": namespace,
        "proposal": "MIX",
        "policies": [left, right],
        "prompt_ids": [p["prompt_id"] for p in probes[:1] if subset]
        if subset
        else [p["prompt_id"] for p in probes],
        "prompt_hash": canonical_hash(probes[:1] if subset else probes),
    }

    def action(i):
        row = dict(original_rows[i])
        key = canonical_hash([namespace, row["prompt_id"], row["draw_index"]])
        chosen = [left, right][int(canonical_hash([key, "mixture_source"])[0], 16) % 2]
        row.update(
            sample_id=key,
            sample_key=key,
            rng_namespace=namespace,
            sample_seed=int(key[:16], 16) % (2**63),
            proposal_fingerprint=chosen["inference_fingerprint"],
            proposal_policy_id=chosen["candidate_id"],
        )
        return row

    count = 8 if subset else 16
    mixture = gpu._chunks(tmp_path / "actual_mix", identity, action, count=count, resume=False)
    mixed = list(gpu._rows(mixture))
    mixscores = {}
    for name, policy in [("joint_1", left), ("joint_0", right)]:
        scored = list(
            gpu._rows(response["reference_scores"][policy["candidate_id"]]["score_receipt"])
        )

        def score(i, scored=scored):
            return {**scored[i], "sample_id": mixed[i]["sample_id"], "rng_namespace": namespace}

        sid = {
            "policy": policy,
            "runtime": identity["runtime"],
            "sample_identity": canonical_hash(identity),
            "sample_chunks": mixture["chunks"],
        }
        mixscores[policy["candidate_id"]] = gpu._chunks(
            tmp_path / ("mix_score_" + name), sid, score, count=count, resume=False
        )
    response["pair_checks"] = [
        {
            "bank_id": bank["bank_id"],
            "contrast_id": "joint_1_minus_joint_0",
            "left_candidate_id": left["candidate_id"],
            "right_candidate_id": right["candidate_id"],
            "mixture": mixture,
            "mixture_scores": mixscores,
        }
    ]
    raw = fit._action_rows(
        response["reference"],
        probes,
        role="reference",
        origin_id=task["id"],
        policy_fingerprint=forks["origin_policy"]["inference_fingerprint"],
    )
    stats = fit._observations(bank, response, response["reference"], raw, probes, ())
    if subset:
        stats[1]["overlap_usable"][:] = True
    before = np.array([s["estimate"] for s in stats])
    audit = fit._apply_mixture_references(
        bank,
        response,
        probes,
        stats,
        origin_id=task["id"],
        forbidden_ids=set(),
        forbidden_streams=set(),
        forbidden_seeds=set(),
    )
    assert audit["fallback_diagnostics"][0]["selected_proposal"] == (
        "ORIGIN_AND_MIX" if subset else "MIX"
    )
    assert audit["fallback_diagnostics"][1]["selected_proposal"] == "ORIGIN_REQUIRED_MIX_MISSING"
    np.testing.assert_array_equal(audit["original_origin_statistics"]["estimate"], before)
    assert all(np.all(s["covariance_of_mean"][:4, 4:] == 0) for s in stats[: 1 if subset else 2])
    if subset:
        np.testing.assert_array_equal(stats[1]["estimate"], before[1])
        assert stats[1]["selected_proposal"] == ["ORIGIN"] * 3
    assert all(not s["resolved"][1:].any() for s in stats)


def test_shared_token_hash_replaces_duplicate_score_tokens_and_rejects_forgery():
    from src.modeling_v3.io import canonical_hash

    samples, scores, policies = packet_fixture()
    expected = _packet(samples, scores, policies).contributions
    for rows in scores.values():
        for row in rows:
            row["shared_token_identity"] = canonical_hash(row.pop("token_ids"))
    np.testing.assert_array_equal(_packet(samples, scores, policies).contributions, expected)
    scores["joint_1"][0]["shared_token_identity"] = "f" * 64
    with pytest.raises(ValueError, match="token identity"):
        _packet(samples, scores, policies)


def test_reference_overflow_is_unresolved_without_clipping_or_hiding_identity_errors():
    from src.modeling_v4.response_fit import _origin_reference_statistics

    samples, scores, policies = packet_fixture()
    samples = [{**s, "generation_sequence_logp": -1000.0} for s in samples]
    value = _origin_reference_statistics(samples, scores, policies, np.array([False] * 3))
    assert value["origin_nonrepresentable_weight"] == [True] * 3
    assert np.isnan(value["estimate"]).all() and not value["resolved"].any()
    scores["joint_1"][0]["input_hash"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        _origin_reference_statistics(samples, scores, policies, np.array([False] * 3))


def test_public_d_entry_uses_mapping_cache_and_actual_registered_m_n(tmp_path, monkeypatch):
    from src.modeling_v3.io import atomic_json, canonical_hash
    from src.modeling_v4 import gpu_collect as gpu
    from src.modeling_v4 import response_fit as fit

    config, task, forks, response, gradient, probes = completed_fixture(tmp_path)
    config["qwen"] = {
        "seed_roles": {"development": [41001]},
        "generation": {
            "max_new_tokens": 64,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "thinking": False,
        },
    }
    config["observations"]["methods"] = ["RAW4"]
    task.update(
        kind="map",
        stage="D",
        prompts=2,
        role="development",
        nested_bank_prefixes=[1, 2],
        nested_draw_counts=[4, 8],
    )
    source = {"sha256": "a" * 64}
    monkeypatch.setattr(fit, "_require_server_cpu", lambda: None)
    monkeypatch.setattr(fit, "source_identity", lambda: source)
    seen = []

    def phase(_plan, _task, cache):
        cache["actual_phase_verifier_uses_mapping_assignment"] = True
        seen.append(cache)
        return {}

    monkeypatch.setattr(gpu, "_validate_phase_task", phase)
    troot = tmp_path / "campaign" / "tasks" / task["id"]
    troot.mkdir(parents=True)
    atomic_json(troot / "forks.json", forks)
    atomic_json(troot / "response.json", response)
    atomic_json(
        troot / "COMPLETE.json",
        {
            "status": "COMPLETED",
            "task": task,
            "execution_kind": "REAL_CUDA_MODEL",
            "origin_id": task["id"],
            "config_hash": canonical_hash(config),
            "source_hash": source["sha256"],
            "forks": fit._binding(troot / "forks.json"),
            "response": fit._binding(troot / "response.json"),
            "gradient": gradient,
        },
    )
    training = [
        {"prompt_id": f"train{i}", "family": f"f{i % 2}", "interface": "i"} for i in range(3)
    ]
    plan = {
        "tasks": [task],
        "config": config,
        "source": source,
        "root": str(tmp_path / "campaign"),
        "inputs": {"panels": {"observation": probes}, "train_prompts": training},
    }
    plan["task_list_hash"] = canonical_hash(plan)
    atomic_json(tmp_path / "tasks.json", plan)
    result = fit.fit_response_map(
        tmp_path / "tasks.json",
        task_id=task["id"],
        out=tmp_path / "fit",
        calibration_banks=1,
        bank_selector="BLOCK_PIVOT_QR",
        draw_count=4,
    )
    assert seen and result["calibration_banks"] == 1 and result["draws"] == 4
    assert result["observation_packets"]["direct_work"]["used_prefix_draws"] == 4
    assert result["observation_packets"]["work"]["physical_draws"] == 8
    with pytest.raises(ValueError, match="registered nested"):
        fit.fit_response_map(
            tmp_path / "tasks.json", task_id=task["id"], out=tmp_path / "bad", calibration_banks=3
        )


def test_frozen_kernel_specs_reject_reinterpretation_and_keep_only_selected_grid():
    from src.modeling_v4.response_fit import _model_specs

    config = {
        "models": {
            "compression_ranks": [2, "FULL"],
            "ridge_alpha": [1e-5, 0.01],
            "rbf_bandwidth_multipliers": [1.0, 4.0],
        }
    }
    primary = {
        "id": "selected",
        "model": "FULL_RBF_RAW",
        "representation": "R2",
        "settings": {"alpha": 0.01, "bandwidth_multiplier": 4.0},
    }
    result = _model_specs(config, ["RAW4"], 1e-5, 1.0, {"primary_models": [primary]})
    assert [r["alpha"] for r in result if r["frozen_model_id"]] == [0.01]
    assert len(result) == 3  # fixed full/rank-2 baselines plus the single frozen RBF
    for bad in [
        {**primary, "representation": "R3"},
        {
            **primary,
            "model": "FULL_EFFECTIVE_WEIGHT_RIDGE",
            "representation": "R3",
            "settings": {"standardize_blocks": True},
        },
    ]:
        with pytest.raises(ValueError, match="representation"):
            _model_specs(config, ["RAW4"], 1e-5, 1.0, {"primary_models": [bad]})


def test_actual_nested_work_direct_and_reference_validation_keep_original_prefixes(tmp_path):
    from src.modeling_v3.io import canonical_hash
    from src.modeling_v4 import gpu_collect as gpu
    from src.modeling_v4 import response_fit as fit

    _, task, forks, response, _, probes = completed_fixture(tmp_path)

    def extend(role, score_key, extra, extra_scores):
        old = response[role]
        identity = {**old["identity"], "draws": 12}
        rows = list(gpu._rows(old))
        prior = {(r["prompt_id"], r["draw_index"]): r for r in rows}

        def draw(i):
            pid, index = probes[i // 12]["prompt_id"], i % 12
            value = prior[pid, index % 8]
            key = canonical_hash([identity["rng_namespace"], pid, index])
            return {
                **value,
                "sample_id": key,
                "sample_key": key,
                "draw_index": index,
                "sample_seed": int(key[:16], 16) % (2**63),
            }

        measured = gpu._chunks(tmp_path / extra, identity, draw, count=24, resume=False)
        newrows = list(gpu._rows(measured))
        scores = {}
        for candidate, value in response[score_key].items():
            original = value["score_receipt"]
            by_id = {r["sample_id"]: r for r in gpu._rows(original)}

            def score(i, by_id=by_id):
                raw = newrows[i]
                source = prior[raw["prompt_id"], raw["draw_index"] % 8]
                return {**by_id[source["sample_id"]], "sample_id": raw["sample_id"]}

            sid = {
                **original["identity"],
                "sample_identity": canonical_hash(identity),
                "sample_chunks": measured["chunks"],
            }
            scores[candidate] = {
                "score_receipt": gpu._chunks(
                    tmp_path / f"{extra_scores}_{candidate}", sid, score, count=24, resume=False
                )
            }
        response[extra], response[extra_scores] = measured, scores

    extend("work", "work_scores", "ncurve_work", "ncurve_scores")
    extend("direct_work", "direct_scores", "ncurve_direct_work", "ncurve_direct_scores")
    origin = forks["origin_policy"]["inference_fingerprint"]
    nested, raw, provenance = fit._nested_work_packets(response, probes, task["id"], origin, 12)
    assert provenance["work"]["physical_draws"] == provenance["direct_work"]["physical_draws"] == 12
    bank = forks["banks"][-1]
    eight = fit._observations(
        bank, nested, nested["work"], raw["work"], probes, ("RAW4",), draw_count=8
    )
    original = fit._action_rows(
        response["work"], probes, role="work", origin_id=task["id"], policy_fingerprint=origin
    )
    expected = fit._observations(bank, response, response["work"], original, probes, ("RAW4",))
    np.testing.assert_array_equal(eight["RAW4"], expected["RAW4"])
    with pytest.raises(ValueError, match="matched budget"):
        fit._nested_work_packets(response, probes, task["id"], origin, 16)

    extend("reference", "reference_scores", "reference_validation", "reference_validation_scores")
    task["reference_validation"] = {
        "draws": 12,
        "bank_ids": [bank["bank_id"]],
        "prompt_scope": "all_frozen_observation_prompts",
    }
    response["reference_validation_scope"] = task["reference_validation"]
    ref = fit._action_rows(
        response["reference"],
        probes,
        role="reference",
        origin_id=task["id"],
        policy_fingerprint=origin,
    )
    primary = {
        bank["bank_id"]: fit._observations(bank, response, response["reference"], ref, probes, ())
    }
    result = fit._reference_validation(
        response, task, [bank], probes, ref, primary, tmp_path, origin
    )
    assert result["nested_prefix_not_independent_replicates"] is True
    with np.load(result["arrays"]["path"], allow_pickle=False) as data:
        np.testing.assert_allclose(data[f"{bank['bank_id']}__estimate_change"], 0, atol=1e-16)
