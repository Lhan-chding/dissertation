"""Deterministic CPU neural diagnostic for truthful/compensatory equilibria.

The experiment uses two sigmoid units as the smallest differentiable reduction
of the modular perceiver/reasoner: ``p`` is truthful-perception probability and
``q`` is canonical-reasoning probability.  Outcome-only training maximizes the
coordination reward ``p*q + (1-p)*(1-q)``.  This retains the two basins needed
for the diagnostic while keeping the acceptance gate fast and auditable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal

from compbias.envs.cva_world.canonical_solver import solve
from compbias.envs.cva_world.generator import (
    GeneratorConfig,
    generate_dataset,
    generate_error_mechanism_counterfactuals,
)
from compbias.envs.cva_world.schema import CVASample, SemanticSplit, TaskFamily
from compbias.envs.cva_world.verifier import verify_answer
from compbias.interventions.counterfactual import pair_error_mechanism_shift
from compbias.interventions.error_catalog import (
    apply_catalog_error,
    validate_error_catalog,
)

NeuralProfile = Literal["truth_aligned", "flat", "spurious"]
NeuralTrainingMode = Literal["perception_only", "reasoning_only", "joint"]
_VALID_PROFILES = frozenset({"truth_aligned", "flat", "spurious"})
_VALID_MODES = frozenset({"perception_only", "reasoning_only", "joint"})
MAX_NEURAL_STEPS = 512
_IID_ERROR_MECHANISM = "offset_plus_2"
_OOD_ERROR_MECHANISM = "offset_minus_2"
CorrectnessMatrix = tuple[tuple[bool, bool], tuple[bool, bool]]


@dataclass(frozen=True, slots=True)
class NeuralExperimentConfig:
    """Immutable configuration for one CPU-sized diagnostic run."""

    profile: NeuralProfile | str
    mode: NeuralTrainingMode | str
    seed: int
    steps: int = 48
    device: str = "cpu"
    learning_rate: float = 0.8

    def __post_init__(self) -> None:
        if self.profile not in _VALID_PROFILES:
            raise ValueError(
                f"unknown profile {self.profile!r}; expected one of "
                f"{', '.join(sorted(_VALID_PROFILES))}"
            )
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"unknown mode {self.mode!r}; expected one of {', '.join(sorted(_VALID_MODES))}"
            )
        if self.mode == "joint" and self.profile != "spurious":
            raise ValueError(
                "joint mode models the spurious-coordination basin and requires "
                "profile='spurious'; compare error profiles with perception_only"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if self.steps > MAX_NEURAL_STEPS:
            raise ValueError(f"steps must be at most {MAX_NEURAL_STEPS}")
        if self.device != "cpu":
            raise ValueError("the small-neural diagnostic requires device='cpu'")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, Real)
            or not math.isfinite(float(self.learning_rate))
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be a positive finite number")


@dataclass(frozen=True, slots=True)
class NeuralCheckpoint:
    """One immutable error-coupling checkpoint."""

    step: int
    truthful_perception_probability: float
    canonical_reasoning_probability: float
    perception_loss: float
    reasoning_loss: float
    coupling: float
    outcome_loss: float
    iid_accuracy: float
    ood_accuracy: float


@dataclass(frozen=True, slots=True)
class NeuralExperimentResult:
    """Summary of a completed diagnostic run."""

    profile: str
    mode: str
    seed: int
    perception_shift: float
    equilibrium_mode: str
    iid_accuracy: float
    ood_accuracy: float
    perception_accuracy: float
    canonical_reasoning_accuracy: float
    paired_sample_id: str
    iid_error_mechanism: str
    ood_error_mechanism: str
    shifted_factors: tuple[str, ...]
    iid_correctness_matrix: CorrectnessMatrix
    ood_correctness_matrix: CorrectnessMatrix
    history: tuple[NeuralCheckpoint, ...]


@dataclass(frozen=True, slots=True)
class _PairedMechanism:
    sample_id: str
    source_mechanism: str
    counterfactual_mechanism: str
    shifted_factors: tuple[str, ...]
    iid_correctness: CorrectnessMatrix
    ood_correctness: CorrectnessMatrix


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _initial_probabilities(config: NeuralExperimentConfig) -> tuple[float, float]:
    if config.mode == "perception_only":
        fixed_reasoner = {
            "truth_aligned": 0.85,
            "flat": 0.5,
            "spurious": 0.15,
        }[config.profile]
        return 0.5, fixed_reasoner
    if config.mode == "reasoning_only":
        fixed_perceiver = {
            "truth_aligned": 0.85,
            "flat": 0.5,
            "spurious": 0.15,
        }[config.profile]
        return fixed_perceiver, 0.5

    # Alternate across the separatrix while retaining a small deterministic,
    # seed-local offset.  No process-global RNG state is read or modified.
    truthful_basin = config.seed % 2 == 0
    offset = ((config.seed * 37) % 9 - 4) * 0.002
    center = 0.67 if truthful_basin else 0.33
    return center + offset, center - offset


def _nontruth_error_id(sample: CVASample) -> str:
    errors = validate_error_catalog(sample.error_catalog)
    identifiers = tuple(error.error_id for error in errors if error.error_id != "truth")
    if len(identifiers) != 1:
        raise ValueError("the two-state neural diagnostic requires exactly one non-truth error")
    return identifiers[0]


def _state_scenes(sample: CVASample) -> tuple[object, object]:
    return (
        apply_catalog_error(sample.scene, sample.error_catalog, "truth"),
        apply_catalog_error(
            sample.scene,
            sample.error_catalog,
            _nontruth_error_id(sample),
        ),
    )


def _action_correctness(
    sample: CVASample,
    *,
    learned_compensation: float,
) -> CorrectnessMatrix:
    rows: list[tuple[bool, bool]] = []
    for perceived_scene in _state_scenes(sample):
        canonical_prediction = solve(perceived_scene, sample.question, sample.task_family)
        if isinstance(canonical_prediction, bool) or not isinstance(canonical_prediction, Real):
            raise TypeError("the scalar neural diagnostic requires numeric answers")
        predictions = (
            canonical_prediction,
            float(canonical_prediction) + learned_compensation,
        )
        rows.append(
            tuple(verify_answer(sample, prediction).is_correct for prediction in predictions)
        )
    return (rows[0], rows[1])


def _paired_mechanism(seed: int) -> _PairedMechanism:
    source = generate_dataset(
        GeneratorConfig(
            seed=seed,
            samples_per_family_per_split=1,
            splits=(SemanticSplit.IID_TEST,),
            task_families=(TaskFamily.DIGIT_OFFSET,),
            visual_styles=("baseline", "size_compact", "layout_shifted"),
            train_error_mechanism=_IID_ERROR_MECHANISM,
            ood_error_mechanism=_OOD_ERROR_MECHANISM,
        )
    )[0]
    counterfactual = generate_error_mechanism_counterfactuals(
        (source,), counterfactual_error_mechanism=_OOD_ERROR_MECHANISM
    )[0]
    pair = pair_error_mechanism_shift((source,), (counterfactual,))[0]
    erroneous_scene = _state_scenes(source)[1]
    erroneous_prediction = solve(erroneous_scene, source.question, source.task_family)
    canonical_prediction = solve(source.scene, source.question, source.task_family)
    if (
        isinstance(erroneous_prediction, bool)
        or not isinstance(erroneous_prediction, Real)
        or isinstance(canonical_prediction, bool)
        or not isinstance(canonical_prediction, Real)
    ):
        raise TypeError("the scalar neural diagnostic requires numeric answers")
    learned_compensation = float(canonical_prediction) - float(erroneous_prediction)
    return _PairedMechanism(
        sample_id=pair.sample_id,
        source_mechanism=pair.source_error_mechanism,
        counterfactual_mechanism=pair.counterfactual_error_mechanism,
        shifted_factors=pair.shifted_factors,
        iid_correctness=_action_correctness(source, learned_compensation=learned_compensation),
        ood_correctness=_action_correctness(
            counterfactual, learned_compensation=learned_compensation
        ),
    )


def _expected_accuracy(p: float, q: float, correctness: CorrectnessMatrix) -> float:
    state_probabilities = (p, 1.0 - p)
    action_probabilities = (q, 1.0 - q)
    return sum(
        state_probability * action_probability * float(correctness[state][action])
        for state, state_probability in enumerate(state_probabilities)
        for action, action_probability in enumerate(action_probabilities)
    )


def _checkpoint(
    step: int,
    p: float,
    q: float,
    paired: _PairedMechanism,
) -> NeuralCheckpoint:
    iid_accuracy = _expected_accuracy(p, q, paired.iid_correctness)
    ood_accuracy = _expected_accuracy(p, q, paired.ood_correctness)
    perception_loss = 1.0 - p
    reasoning_loss = 1.0 - q
    target_outcome_loss = 1.0 - iid_accuracy
    coupling = 0.5 * (target_outcome_loss - perception_loss - reasoning_loss)
    outcome_loss = perception_loss + reasoning_loss + 2.0 * coupling
    return NeuralCheckpoint(
        step=step,
        truthful_perception_probability=p,
        canonical_reasoning_probability=q,
        perception_loss=perception_loss,
        reasoning_loss=reasoning_loss,
        coupling=coupling,
        outcome_loss=outcome_loss,
        iid_accuracy=iid_accuracy,
        ood_accuracy=ood_accuracy,
    )


def run_neural_experiment(config: NeuralExperimentConfig) -> NeuralExperimentResult:
    """Run outcome-only functional gradient ascent and return fresh results."""

    if not isinstance(config, NeuralExperimentConfig):
        raise TypeError("config must be a NeuralExperimentConfig")
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - base-only install
        raise ModuleNotFoundError(
            "run_neural_experiment requires torch; install compbias[neural]"
        ) from error

    initial_p, initial_q = _initial_probabilities(config)
    p_logit = torch.tensor(_logit(initial_p), dtype=torch.float64, requires_grad=True)
    q_logit = torch.tensor(_logit(initial_q), dtype=torch.float64, requires_grad=True)
    paired = _paired_mechanism(config.seed)
    iid_correctness = torch.tensor(paired.iid_correctness, dtype=torch.float64)
    history = [_checkpoint(0, initial_p, initial_q, paired)]

    train_p = config.mode in {"perception_only", "joint"}
    train_q = config.mode in {"reasoning_only", "joint"}
    for step in range(1, config.steps + 1):
        p = torch.sigmoid(p_logit)
        q = torch.sigmoid(q_logit)
        state_probabilities = torch.stack((p, 1.0 - p))
        action_probabilities = torch.stack((q, 1.0 - q))
        reward = torch.sum(
            state_probabilities[:, None] * action_probabilities[None, :] * iid_correctness
        )
        variables = tuple(
            variable
            for variable, trainable in ((p_logit, train_p), (q_logit, train_q))
            if trainable
        )
        gradients = torch.autograd.grad(reward, variables)
        gradient_by_id = {
            id(variable): gradient for variable, gradient in zip(variables, gradients, strict=True)
        }
        if train_p:
            p_logit = (
                (p_logit + config.learning_rate * gradient_by_id[id(p_logit)])
                .detach()
                .requires_grad_(True)
            )
        if train_q:
            q_logit = (
                (q_logit + config.learning_rate * gradient_by_id[id(q_logit)])
                .detach()
                .requires_grad_(True)
            )
        final_p = float(torch.sigmoid(p_logit).detach())
        final_q = float(torch.sigmoid(q_logit).detach())
        history.append(_checkpoint(step, final_p, final_q, paired))

    final = history[-1]
    equilibrium_mode = (
        "truthful"
        if final.truthful_perception_probability + final.canonical_reasoning_probability >= 1.0
        else "compensatory"
    )
    return NeuralExperimentResult(
        profile=str(config.profile),
        mode=str(config.mode),
        seed=config.seed,
        perception_shift=final.perception_loss - history[0].perception_loss,
        equilibrium_mode=equilibrium_mode,
        iid_accuracy=final.iid_accuracy,
        ood_accuracy=final.ood_accuracy,
        perception_accuracy=final.truthful_perception_probability,
        canonical_reasoning_accuracy=final.canonical_reasoning_probability,
        paired_sample_id=paired.sample_id,
        iid_error_mechanism=paired.source_mechanism,
        ood_error_mechanism=paired.counterfactual_mechanism,
        shifted_factors=paired.shifted_factors,
        iid_correctness_matrix=paired.iid_correctness,
        ood_correctness_matrix=paired.ood_correctness,
        history=tuple(history),
    )


__all__ = [
    "NeuralCheckpoint",
    "NeuralExperimentConfig",
    "NeuralExperimentResult",
    "run_neural_experiment",
]
