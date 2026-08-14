from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compbias.estimation.crossed_risk_estimator import estimate_crossed_risks
from compbias.interventions.transport_audit import audit_synthetic_transport
from compbias.trajectories.fork_replay import fork_natural_mediators
from compbias.trajectories.natural_sampler import collect_natural_mediators
from compbias.trajectories.records import CrossedRiskRecord
from compbias.trajectories.synthetic_mediator import build_synthetic_mediators


@dataclass(frozen=True)
class _NaturalOutput:
    raw: str
    parsed: dict[str, object]
    parser_confidence: float
    parse_failed: bool
    error_type: str
    task_severity: float
    answer: str
    reward: int
    transport_signature: tuple[float, ...]
    activation_ref: str | None = None


@dataclass(frozen=True)
class _ForkOutput:
    answer: str
    reward: int
    metadata: dict[str, object]


class _FakeAdapter:
    def natural_forward(self, sample, interface, seed):
        del interface
        error = "truth" if seed % 2 == 0 else "offset:+2"
        return _NaturalOutput(
            raw=f'{{"error":"{error}"}}',
            parsed={"error": error},
            parser_confidence=0.99,
            parse_failed=False,
            error_type=error,
            task_severity=float(error != "truth"),
            answer=str(sample["answer"]),
            reward=int(error == "truth"),
            transport_signature=(float(seed % 2), 0.0),
        )

    def fork_continuation(self, record, interface, seed, *, image_access):
        del interface, seed
        assert image_access is False
        return _ForkOutput(
            answer=record.original_answer,
            reward=int(record.error_type == "truth"),
            metadata={"image_access": False, "replayed_exact_state": True},
        )

    def synthetic_mediator(self, sample, interface, error_type, seed):
        del interface, seed
        return {
            "synthetic_mediator": {"error": error_type},
            "construction_method": "scene_json_edit",
            "transport_signature": (5.0, 5.0),
            "nearest_natural_state_ids": (f"{sample['sample_id']}:natural:0",),
        }


def _sample() -> dict[str, object]:
    return {
        "sample_id": "sample-1",
        "image_path": "artifacts/datasets/images/sample-1.png",
        "question": "What is shown?",
        "gold_scene": {"value": 5},
        "answer": 5,
    }


def test_natural_collection_fork_and_synthetic_paths_are_disjoint() -> None:
    adapter = _FakeAdapter()
    natural = collect_natural_mediators(
        adapter,
        (_sample(),),
        interface_id="scene-json-v1",
        checkpoint_id="base",
        n_mediators=4,
        seeds=(10, 11, 12, 13),
    )
    forks = fork_natural_mediators(
        adapter,
        natural,
        interface_id="scene-json-v1",
        n_forks=3,
        seeds=(20, 21, 22),
    )
    synthetic = build_synthetic_mediators(
        adapter,
        (_sample(),),
        interface_id="scene-json-v1",
        checkpoint_id="base",
        error_types=("offset:+2",),
        seeds=(30,),
    )

    assert len(natural) == 4
    assert len(forks) == 12
    assert len(synthetic) == 1
    assert {row.source_kind for row in forks} == {"natural"}
    assert all(row.replay_fidelity_metadata["image_access"] is False for row in forks)
    assert synthetic[0].mediator_record_id not in {row.mediator_record_id for row in natural}


def test_transport_audit_detects_off_support_synthetic_signatures() -> None:
    rng = np.random.default_rng(9)
    natural = rng.normal(0.0, 0.1, size=(80, 3))
    synthetic = rng.normal(3.0, 0.1, size=(80, 3))
    report = audit_synthetic_transport(
        natural_signatures=natural,
        synthetic_signatures=synthetic,
        natural_rewards=np.ones(80),
        synthetic_rewards=np.zeros(80),
        natural_error_types=("offset:+2",) * 80,
        synthetic_error_types=("offset:+2",) * 80,
        bootstrap_draws=2_000,
        confidence=0.95,
        seed=18,
    )

    assert report.error_support_overlap == 1.0
    assert report.two_sample_accuracy > 0.95
    assert report.reward_gap == -1.0
    assert report.off_support_stress_test
    assert report.reward_gap_ci_high < 0.0


def test_crossed_risk_estimator_uses_paired_sample_cells() -> None:
    records = tuple(
        CrossedRiskRecord(
            sample_id=sample_id,
            interface_id="scene-json-v1",
            perception_source=perception,
            reasoner_source=reasoner,
            loss=loss,
            reward=1.0 - loss,
            seed=0,
        )
        for sample_id, adjustment in (("a", 0.0), ("b", 0.02))
        for perception, reasoner, loss in (
            ("model", "model", 0.20 + adjustment),
            ("oracle", "model", 0.30 + adjustment),
            ("model", "oracle", 0.40 + adjustment),
            ("oracle", "oracle", 0.00 + adjustment),
        )
    )
    result = estimate_crossed_risks(records)

    assert len(result.per_sample) == 2
    assert result.aggregate.interaction < 0.0
    assert result.maximum_identity_residual < 1e-12
