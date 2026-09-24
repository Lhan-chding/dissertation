"""CPU-only synthetic workflow integration; no records are real model results."""

import json

import numpy as np
import pytest

from src.prospective_selection import workflow
from src.prospective_selection.features import LEVELS, build_predecision_packet
from src.prospective_selection.jobs import TaskRegistry
from src.prospective_selection.protocol import load_protocol
from src.prospective_selection.selectors import FrozenSelector


def _packet(spec, step, level, count=1):
    sample = dict(
        prompt_id="fixed-p",
        family="trend",
        interface="SYMBOLIC_FRESH",
        event="W",
        relation_numerator=0,
        relation_denominator=2,
        F=0,
        B=0,
        M=0,
    )
    return build_predecision_packet(
        origin_id=f"{spec['seed']}_t{step}",
        lineage_id=str(spec["seed"]),
        source_recipe=spec["source_recipe"],
        step=step,
        current_rows=[sample] * count,
        history_rows=[sample] * count,
        feature_level=level,
    )


def _packets(config):
    return {
        level: [
            _packet(spec, step, level)
            for spec in config["origins"]["development"] + config["origins"]["tuning"]
            for step in spec["anchors"]
        ]
        for level in LEVELS
    }


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    config = load_protocol()
    packets = _packets(config)
    utilities = np.full((32, 8), 0.2)
    utilities[:, 4] = 0.7
    fit = dict(
        models={
            level: FrozenSelector.fit(
                packets[level], utilities, alpha=0.1, default_action="R4"
            ).to_dict()
            for level in LEVELS
        },
        contexts=32,
        independent_lineages=16,
        final_test_used=False,
    )
    prepared = dict(
        test_panel_status="READY",
        data_id="CPU_FIXTURE_ONLY",
        panels={"T": [{"prompt_id": f"fixture-T{i}"} for i in range(288)]},
    )
    precision = {"N": 12, "mc": {"m": 16}}
    monkeypatch.setattr(workflow, "implementation_id", lambda: "CPU_FIXTURE_IMPLEMENTATION")
    root = tmp_path / "experiment"
    return config, root, packets, utilities, fit, prepared, precision


def _freeze(fixture):
    config, root, _, _, fit, prepared, precision = fixture
    return workflow.freeze_selectors(config, root, fit, precision, prepared, "CPU_FIXTURE_SOURCE")


def _test_packets(fixture):
    config, root, *_ = fixture
    spec = config["origins"]["test_pool"][0]
    origin = f"{spec['seed']}_t32"
    for level in LEVELS:
        _write(root / "prestate" / origin / f"{level}.json", _packet(spec, 32, level).to_dict())
    return origin


def test_complete_freeze_decision_and_independent_outcomes(fixture):
    frozen = _freeze(fixture)
    config, root, *_ = fixture
    origin = _test_packets(fixture)
    decision = workflow.decide(config, root, origin)
    assert decision["choices"] == dict.fromkeys(LEVELS, "R4")
    assert decision["made_before_run"] is True
    assert decision["freeze_id"] == frozen["freeze_id"]
    _write(root / "unrelated_E_outcomes.json", {"R4": 0.0, "R0": 1.0})
    assert workflow.decide(config, root, origin) == decision
    assert frozen["test_n"] == 12 and frozen["test_draws"] == 16
    assert all(v == "prospective-features-v1" for v in frozen["feature_schemas"].values())


def test_missing_and_corrupt_packets_fall_back_with_reasons(fixture):
    _freeze(fixture)
    config, root, *_ = fixture
    origin = _test_packets(fixture)
    (root / "prestate" / origin / f"{LEVELS[0]}.json").unlink()
    (root / "prestate" / origin / f"{LEVELS[1]}.json").write_text("{invalid")
    path = root / "prestate" / origin / f"{LEVELS[2]}.json"
    data = json.loads(path.read_text())
    data["current"]["unknown_future_value"] = 1
    _write(path, data)
    decision = workflow.decide(config, root, origin)
    for level in LEVELS[:3]:
        assert decision["details"][level]["status"] == "FALLBACK_BEST_STATIC"
        assert decision["details"][level]["reason"]
        assert decision["choices"][level] == "R4"
    assert decision["details"][LEVELS[3]]["status"] == "POINT_DECISION_NOT_SAFETY_CERTIFICATE"


def test_all_missing_still_uses_registered_origin_and_default(fixture):
    _freeze(fixture)
    config, root, *_ = fixture
    decision = workflow.decide(config, root, "63001_t32")
    assert decision["lineage_id"] == 63001
    assert all(v["status"] == "FALLBACK_BEST_STATIC" for v in decision["details"].values())
    with pytest.raises(ValueError, match="registered"):
        workflow.decide(config, root, "../unregistered")


@pytest.mark.parametrize(
    "field,value",
    [
        ("origin_id", "63002_t96"),
        ("lineage_id", "63002"),
        ("source_recipe", "R1"),
        ("history_step", 23),
        ("feature_level", LEVELS[3]),
    ],
)
def test_bad_identity_never_hidden_by_corruption_fallback(fixture, field, value):
    _freeze(fixture)
    config, root, *_ = fixture
    origin = _test_packets(fixture)
    path = root / "prestate" / origin / f"{LEVELS[0]}.json"
    data = json.loads(path.read_text())
    data[field] = value
    data["current"] = {}
    _write(path, data)
    with pytest.raises(ValueError, match="registration"):
        workflow.decide(config, root, origin)
    assert not (root / "decisions" / f"{origin}.json").exists()


def test_different_valid_sample_budget_rejected(fixture):
    _freeze(fixture)
    config, root, *_ = fixture
    origin = _test_packets(fixture)
    spec = config["origins"]["test_pool"][0]
    _write(
        root / "prestate" / origin / f"{LEVELS[3]}.json", _packet(spec, 32, LEVELS[3], 2).to_dict()
    )
    with pytest.raises(ValueError, match="identical current observations: n"):
        workflow.decide(config, root, origin)


def test_future_artifacts_block_first_decision_without_reading_results(fixture, monkeypatch):
    _freeze(fixture)
    config, root, *_ = fixture
    origin = _test_packets(fixture)
    future = TaskRegistry(root, config).branch_output(origin, "R0", 1) / "checkpoint_H32.pt"
    future.parent.mkdir(parents=True)
    future.write_bytes(b"unreadable future endpoint fixture")
    real_read = workflow._read
    reads = []

    def observed_read(path):
        reads.append(str(path))
        return real_read(path)

    monkeypatch.setattr(workflow, "_read", observed_read)
    with pytest.raises(ValueError, match="artifacts already exist"):
        workflow.decide(config, root, origin)
    assert not any("branches" in p for p in reads)
    assert not (root / "decisions" / f"{origin}.json").exists()


def test_freeze_rejects_missing_refit_context_and_changed_implementation(fixture, monkeypatch):
    config, root, packets, utilities, fit, prepared, precision = fixture
    fit["models"][LEVELS[0]] = FrozenSelector.fit(
        packets[LEVELS[0]][:-1], utilities[:-1], alpha=0.1, default_action="R4"
    ).to_dict()
    with pytest.raises(ValueError, match="refit all"):
        workflow.freeze_selectors(config, root, fit, precision, prepared, "fixture")
    with pytest.raises(ValueError, match="Implementation changed"):
        workflow.verify_frozen_execution({"implementation_id": "old"})


def test_precision_nested_tuning_excludes_held_lineage_from_every_stage(fixture, monkeypatch):
    config, root, packets, utilities, fit, *_ = fixture
    monkeypatch.setattr(workflow, "development_evidence", lambda *_: (packets, utilities, []))
    result = workflow.precision_from_development(config, root, fit)
    assert result["hyperparameter_source"] == "NESTED_LINEAGE_HOLDOUT_ALPHA_STATIC_AND_COEFFICIENTS"
    assert len(result["planning_rows"]) == 16 and len(result["planning_folds"]) == 32
    for fold in result["planning_folds"]:
        assert fold["held_lineage"] not in fold["fit_lineages"] + fold["tuning_lineages"]
        assert not set(fold["fit_lineages"]) & set(fold["tuning_lineages"])
        assert fold["alpha"] == 10
    assert result["N"] == 32 and result["status"] == "SD_UNIDENTIFIED"
    assert result["planning_limitations"]
    _write(root / "branches/63001_t32/R0/repeat_1/attempt.json", {"fixture": True})
    with pytest.raises(ValueError, match="future artifacts"):
        workflow.precision_from_development(config, root, fit)


def test_development_loader_rejects_cpu_fixture_training_receipt(fixture):
    config, root, packets, *_ = fixture
    origin = packets[LEVELS[0]][0].origin_id
    for level in LEVELS:
        _write(root / "prestate" / origin / f"{level}.json", packets[level][0].to_dict())
    branch = TaskRegistry(root, config).branch_output(origin, "R0", 1)
    _write(branch / "COMPLETE.json", {"status": "CPU_FIXTURE_COMPLETE", "steps": 32})
    with pytest.raises(ValueError, match="real H32"):
        workflow.development_evidence(config, root)


def test_fit_to_freeze_to_predecision_pipeline_with_synthetic_labels(fixture, monkeypatch):
    config, root, packets, utilities, _, prepared, precision = fixture
    monkeypatch.setattr(workflow, "development_evidence", lambda *_: (packets, utilities, []))
    fit = workflow.fit_selectors(config, root, root / "fit")
    assert fit["contexts"] == 32 and fit["independent_lineages"] == 16
    assert fit["final_test_used"] is False
    assert all(model["alpha"] == 10 for model in fit["models"].values())
    freeze = workflow.freeze_selectors(config, root, fit, precision, prepared, "CPU_FIXTURE_SOURCE")
    origin = _test_packets(fixture)
    decision = workflow.decide(config, root, origin)
    assert decision["freeze_id"] == freeze["freeze_id"]
    assert set(decision["choices"].values()) == {"R4"}


def test_development_loader_rejects_misbound_training_identity(fixture):
    config, root, packets, *_ = fixture
    origin = packets[LEVELS[0]][0].origin_id
    for level in LEVELS:
        _write(root / "prestate" / origin / f"{level}.json", packets[level][0].to_dict())
    branch = TaskRegistry(root, config).branch_output(origin, "R0", 1)
    _write(
        branch / "COMPLETE.json",
        dict(
            status="TRAINING_COMPLETE",
            steps=32,
            identity=dict(
                origin_id=origin, lineage_id=61002, recipe_id="R0", repeat=1, role="development"
            ),
        ),
    )
    with pytest.raises(ValueError, match="training identity"):
        workflow.development_evidence(config, root)
