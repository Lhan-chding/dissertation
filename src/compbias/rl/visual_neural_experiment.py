"""Image-backed CPU experiment for Phase-C compensability diagnostics.

Unlike the scalar coordination fixture in :mod:`compbias.rl.neural_experiment`,
this module renders a real CVA-World scene with PIL, converts those pixels to a
tensor, and trains the existing CNN-perceiver/MLP-reasoner model.  The only
training objective is expected final-answer reward.  State-specific reasoner
success is measured by injecting the two one-hot perceived states, so the MLP
cannot reread the image during that intervention.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from compbias.envs.cva_world.canonical_solver import solve
from compbias.envs.cva_world.generator import (
    GeneratorConfig,
    generate_dataset,
    generate_error_mechanism_counterfactuals,
)
from compbias.envs.cva_world.renderer import RenderConfig, render_sample
from compbias.envs.cva_world.schema import CVASample, SemanticSplit, TaskFamily
from compbias.envs.cva_world.verifier import verify_answer
from compbias.interventions.counterfactual import pair_error_mechanism_shift
from compbias.interventions.error_catalog import (
    apply_catalog_error,
    validate_error_catalog,
)
from compbias.models.modular_neural import ModularPerceiverReasoner, set_training_mode

VisualProfile = Literal["truth_aligned", "spurious"]
VisualTrainingMode = Literal["perception_only", "reasoning_only", "joint"]
_VALID_PROFILES = frozenset({"truth_aligned", "spurious"})
_VALID_MODES = frozenset({"perception_only", "reasoning_only", "joint"})
CorrectnessMatrix = tuple[tuple[bool, bool], tuple[bool, bool]]


def _integer(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        limit = f" between {minimum} and {maximum}" if maximum is not None else f" >= {minimum}"
        raise ValueError(f"{name} must be{limit}")
    return value


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class VisualNeuralConfig:
    """Closed configuration for one deterministic image-backed run."""

    profile: VisualProfile | str
    mode: VisualTrainingMode | str
    seed: int
    steps: int = 32
    image_size: int = 16
    hidden_dim: int = 4
    perception_learning_rate: float = 0.35
    reasoning_learning_rate: float = 0.35
    device: str = "cpu"
    scene_seed: int = 23
    iid_error_mechanism: str = "offset_plus_2"
    ood_error_mechanism: str = "offset_minus_2"

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
            raise ValueError("joint mode requires profile='spurious'")
        _integer(self.seed, "seed", minimum=0)
        _integer(self.steps, "steps", minimum=1, maximum=512)
        _integer(self.image_size, "image_size", minimum=8, maximum=128)
        _integer(self.hidden_dim, "hidden_dim", minimum=2, maximum=128)
        _integer(self.scene_seed, "scene_seed", minimum=0)
        object.__setattr__(
            self,
            "perception_learning_rate",
            _positive_float(self.perception_learning_rate, "perception_learning_rate"),
        )
        object.__setattr__(
            self,
            "reasoning_learning_rate",
            _positive_float(self.reasoning_learning_rate, "reasoning_learning_rate"),
        )
        if self.device != "cpu":
            raise ValueError("device must be 'cpu' for the Phase-C acceptance experiment")
        iid_mechanism = _nonempty_string(self.iid_error_mechanism, "iid_error_mechanism")
        ood_mechanism = _nonempty_string(self.ood_error_mechanism, "ood_error_mechanism")
        if iid_mechanism == ood_mechanism:
            raise ValueError("IID and OOD error mechanisms must differ")


@dataclass(frozen=True, slots=True)
class VisualNeuralSweepConfig:
    """Shared settings and preregistered gates for a ten-seed sweep."""

    seeds: tuple[int, ...] = tuple(range(10))
    steps: int = 32
    image_size: int = 16
    hidden_dim: int = 4
    perception_learning_rate: float = 0.35
    reasoning_learning_rate: float = 0.35
    device: str = "cpu"
    scene_seed: int = 23
    iid_error_mechanism: str = "offset_plus_2"
    ood_error_mechanism: str = "offset_minus_2"
    minimum_profile_shift: float = 0.02
    minimum_joint_iid_accuracy: float = 0.70
    minimum_ood_gap_margin: float = 0.20
    ood_gap_bootstrap_resamples: int = 10_000
    ood_gap_confidence_level: float = 0.95
    ood_gap_bootstrap_seed: int = 20_260_814

    def __post_init__(self) -> None:
        if isinstance(self.seeds, (str, bytes)):
            raise TypeError("seeds must be a sequence of integers")
        try:
            seeds = tuple(self.seeds)
        except TypeError as error:
            raise TypeError("seeds must be a sequence of integers") from error
        if len(seeds) < 10:
            raise ValueError("seeds must contain at least 10 values")
        for seed in seeds:
            _integer(seed, "seed", minimum=0)
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        object.__setattr__(self, "seeds", seeds)
        _integer(self.steps, "steps", minimum=1, maximum=512)
        _integer(self.image_size, "image_size", minimum=8, maximum=128)
        _integer(self.hidden_dim, "hidden_dim", minimum=2, maximum=128)
        _integer(self.scene_seed, "scene_seed", minimum=0)
        _integer(
            self.ood_gap_bootstrap_resamples,
            "ood_gap_bootstrap_resamples",
            minimum=1_000,
            maximum=1_000_000,
        )
        _integer(self.ood_gap_bootstrap_seed, "ood_gap_bootstrap_seed", minimum=0)
        for field_name in (
            "perception_learning_rate",
            "reasoning_learning_rate",
            "minimum_profile_shift",
            "minimum_joint_iid_accuracy",
            "minimum_ood_gap_margin",
            "ood_gap_confidence_level",
        ):
            object.__setattr__(
                self, field_name, _positive_float(getattr(self, field_name), field_name)
            )
        if self.minimum_joint_iid_accuracy > 1.0:
            raise ValueError("minimum_joint_iid_accuracy must not exceed 1")
        if self.minimum_ood_gap_margin > 1.0:
            raise ValueError("minimum_ood_gap_margin must not exceed 1")
        if self.ood_gap_confidence_level >= 1.0:
            raise ValueError("ood_gap_confidence_level must be less than 1")
        if self.device != "cpu":
            raise ValueError("device must be 'cpu' for the Phase-C acceptance experiment")
        iid_mechanism = _nonempty_string(self.iid_error_mechanism, "iid_error_mechanism")
        ood_mechanism = _nonempty_string(self.ood_error_mechanism, "ood_error_mechanism")
        if iid_mechanism == ood_mechanism:
            raise ValueError("IID and OOD error mechanisms must differ")


@dataclass(frozen=True, slots=True)
class VisualNeuralCheckpoint:
    """One expected-reward and error-coupling measurement."""

    step: int
    truthful_perception_probability: float
    truthful_state_success_probability: float
    erroneous_state_success_probability: float
    l_p: float
    l_r: float
    coupling: float
    l_o: float
    iid_accuracy: float
    ood_accuracy: float


@dataclass(frozen=True, slots=True)
class VisualNeuralRunResult:
    """Immutable result for one trained CNN/MLP pair."""

    profile: str
    mode: str
    seed: int
    input_source: str
    image_tensor_shape: tuple[int, ...]
    image_sha256: str
    ood_image_sha256: str
    iid_error_mechanism: str
    ood_error_mechanism: str
    shifted_factors: tuple[str, ...]
    paired_sample_id: str
    iid_correctness_matrix: CorrectnessMatrix
    ood_correctness_matrix: CorrectnessMatrix
    perception_shift: float
    equilibrium_mode: str
    iid_accuracy: float
    ood_accuracy: float
    conv_parameter_delta: float
    reasoner_parameter_delta: float
    history: tuple[VisualNeuralCheckpoint, ...]


@dataclass(frozen=True, slots=True)
class VisualNeuralGateSummary:
    """Preregistered Phase-C exit-gate summary."""

    fixed_profiles_opposite: bool
    two_joint_equilibria: bool
    three_training_modes: bool
    perception_frozen_in_reasoning_only: bool
    reasoner_frozen_in_perception_only: bool
    convolution_updated: bool
    joint_iid_accuracy_high: bool
    compensatory_ood_gap_larger: bool
    fixed_seed_reproducible: bool
    truthful_joint_runs: int
    compensatory_joint_runs: int
    mean_truthful_ood_gap: float
    mean_compensatory_ood_gap: float
    ood_gap_difference: float
    ood_gap_difference_ci_low: float
    ood_gap_difference_ci_high: float
    ood_gap_bootstrap_method: str
    ood_gap_bootstrap_resamples: int
    ood_gap_confidence_level: float
    ood_gap_bootstrap_seed: int
    minimum_joint_iid_accuracy: float
    passed: bool


@dataclass(frozen=True, slots=True)
class VisualNeuralSweepResult:
    """Immutable collection of profile controls, joint runs, and gates."""

    config: VisualNeuralSweepConfig
    runs: tuple[VisualNeuralRunResult, ...]
    gates: VisualNeuralGateSummary


@dataclass(frozen=True, slots=True)
class VisualNeuralArtifactPaths:
    """Paths written by :func:`write_visual_neural_artifacts`."""

    metrics_json: Path
    runs_csv: Path
    trajectories_csv: Path
    figure_png: Path


@dataclass(frozen=True, slots=True)
class _PairedVisualInputs:
    source: CVASample
    counterfactual: CVASample
    source_image: Image.Image
    counterfactual_image: Image.Image
    sample_id: str
    shifted_factors: tuple[str, ...]
    iid_correctness: CorrectnessMatrix
    ood_correctness: CorrectnessMatrix


def _torch_module() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - base-only install
        raise ModuleNotFoundError(
            "visual neural experiments require torch; install compbias[neural]"
        ) from error
    return torch


def _nontruth_error_id(sample: CVASample) -> str:
    errors = validate_error_catalog(sample.error_catalog)
    identifiers = tuple(error.error_id for error in errors if error.error_id != "truth")
    if len(identifiers) != 1:
        raise ValueError("the two-state visual diagnostic requires exactly one non-truth error")
    return identifiers[0]


def _state_scenes(sample: CVASample) -> tuple[Mapping[str, object], Mapping[str, object]]:
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
            raise TypeError("the visual neural diagnostic requires numeric answers")
        predictions = (
            canonical_prediction,
            float(canonical_prediction) + learned_compensation,
        )
        rows.append(
            tuple(verify_answer(sample, prediction).is_correct for prediction in predictions)
        )
    return (rows[0], rows[1])


def _paired_inputs(config: VisualNeuralConfig) -> _PairedVisualInputs:
    dataset = generate_dataset(
        GeneratorConfig(
            seed=config.scene_seed,
            samples_per_family_per_split=1,
            splits=(SemanticSplit.IID_TEST,),
            task_families=(TaskFamily.DIGIT_OFFSET,),
            visual_styles=("baseline", "size_compact", "layout_shifted"),
            train_error_mechanism=config.iid_error_mechanism,
            ood_error_mechanism=config.ood_error_mechanism,
        )
    )
    sample = dataset[0]
    counterfactual = generate_error_mechanism_counterfactuals(
        dataset,
        counterfactual_error_mechanism=config.ood_error_mechanism,
    )[0]
    pair = pair_error_mechanism_shift((sample,), (counterfactual,))[0]
    erroneous_scene = _state_scenes(sample)[1]
    erroneous_prediction = solve(erroneous_scene, sample.question, sample.task_family)
    canonical_prediction = solve(sample.scene, sample.question, sample.task_family)
    if (
        isinstance(erroneous_prediction, bool)
        or not isinstance(erroneous_prediction, Real)
        or isinstance(canonical_prediction, bool)
        or not isinstance(canonical_prediction, Real)
    ):
        raise TypeError("the visual neural diagnostic requires numeric answers")
    learned_compensation = float(canonical_prediction) - float(erroneous_prediction)
    render_config = RenderConfig(
        width=config.image_size,
        height=config.image_size,
        style="baseline",
        seed=config.scene_seed,
    )
    iid_image = render_sample(sample, render_config)
    ood_image = render_sample(counterfactual, render_config)
    if iid_image.mode != "RGB" or ood_image.mode != "RGB":
        raise AssertionError("CVA renderer must return RGB images")
    if iid_image.tobytes() != ood_image.tobytes():
        raise AssertionError("paired OOD intervention unexpectedly changed image pixels")
    return _PairedVisualInputs(
        source=sample,
        counterfactual=counterfactual,
        source_image=iid_image,
        counterfactual_image=ood_image,
        sample_id=pair.sample_id,
        shifted_factors=pair.shifted_factors,
        iid_correctness=_action_correctness(sample, learned_compensation=learned_compensation),
        ood_correctness=_action_correctness(
            counterfactual, learned_compensation=learned_compensation
        ),
    )


def _image_sha256(image: Image.Image) -> str:
    payload = image.mode.encode("ascii") + str(image.size).encode("ascii") + image.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _pil_to_tensor(image: Image.Image, torch: Any) -> Any:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")
    if image.mode != "RGB":
        raise ValueError("image must use RGB mode")
    pixels = np.array(image, dtype=np.float32, copy=True)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("image must have exactly three channels")
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).contiguous() / 255.0


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _initialize_reasoner(
    model: ModularPerceiverReasoner,
    config: VisualNeuralConfig,
    torch: Any,
) -> None:
    """Initialize the MLP as a genuine state-conditioned reasoner.

    The two fixed profiles are experimental treatments.  Joint runs instead
    draw both initial state-success logits from a seed-local normal generator;
    no branch is selected from the seed or from a desired final label.
    """

    first = model.reasoning[0]
    final = model.reasoning[2]
    if config.mode == "perception_only":
        state_success = {
            "truth_aligned": (0.80, 0.20),
            "spurious": (0.20, 0.80),
        }[config.profile]
        success_logits = tuple(_logit(value) for value in state_success)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed + 104_729)
        draws = torch.randn((2,), generator=generator, dtype=torch.float32)
        success_logits = tuple(float(value) for value in (1.35 * draws))

    with torch.no_grad():
        for parameter in model.reasoning.parameters():
            parameter.zero_()
        first.weight[0, 0] = 1.0
        first.weight[1, 1] = 1.0
        final.weight[0, 0] = success_logits[0]
        final.weight[1, 1] = success_logits[1]


def _build_model(config: VisualNeuralConfig, torch: Any) -> ModularPerceiverReasoner:
    # Module constructors use torch's process-global generator.  fork_rng
    # restores it on exit, while still giving each run a deterministic model.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = ModularPerceiverReasoner(
            image_channels=3,
            image_size=config.image_size,
            num_perceived_states=2,
            num_reasoning_actions=2,
            hidden_dim=config.hidden_dim,
        )
    _initialize_reasoner(model, config, torch)
    return set_training_mode(model, config.mode)


def _parameter_snapshot(parameters: Any) -> tuple[Any, ...]:
    return tuple(parameter.detach().clone() for parameter in parameters)


def _parameter_delta(before: tuple[Any, ...], after: Any) -> float:
    after_tuple = tuple(after)
    if len(before) != len(after_tuple):
        raise AssertionError("parameter block changed shape during training")
    if not before:
        return 0.0
    return max(
        float(torch_after.detach().sub(torch_before).abs().max())
        for torch_before, torch_after in zip(before, after_tuple, strict=True)
    )


def _conv_parameters(model: ModularPerceiverReasoner, torch: Any) -> tuple[Any, ...]:
    return tuple(
        parameter
        for module in model.perception.modules()
        if isinstance(module, torch.nn.Conv2d)
        for parameter in module.parameters(recurse=False)
    )


def _forward_probabilities(
    model: ModularPerceiverReasoner,
    images: Any,
    torch: Any,
) -> tuple[Any, Any]:
    natural = model(images=images)
    state_probabilities = torch.softmax(natural["perception_logits"], dim=-1)[0]
    injected = model(perceived_state=torch.tensor([0, 1], dtype=torch.long, device=images.device))
    action_probabilities = torch.softmax(injected["reasoning_logits"], dim=-1)
    return state_probabilities, action_probabilities


def _expected_accuracy(
    state_probabilities: Any,
    action_probabilities: Any,
    correctness: CorrectnessMatrix,
    torch: Any,
) -> Any:
    correctness_tensor = torch.tensor(
        correctness,
        dtype=torch.float64,
        device=state_probabilities.device,
    )
    return torch.sum(
        state_probabilities.to(dtype=torch.float64)[:, None]
        * action_probabilities.to(dtype=torch.float64)
        * correctness_tensor
    )


def _checkpoint(
    step: int,
    p_truthful: float,
    truthful_success: float,
    erroneous_success: float,
    iid_accuracy: float,
    ood_accuracy: float,
) -> VisualNeuralCheckpoint:
    # Scalar errors are e_p in {0,+1} and e_r in {0,-1}.  A compensator
    # therefore cancels an erroneous perceived state exactly on IID data.
    l_p = 1.0 - p_truthful
    l_r = p_truthful * (1.0 - truthful_success) + (1.0 - p_truthful) * erroneous_success
    coupling = -(1.0 - p_truthful) * erroneous_success
    l_o = l_p + l_r + 2.0 * coupling
    if not math.isclose(iid_accuracy, 1.0 - l_o, abs_tol=1.0e-6):
        raise AssertionError("executed IID accuracy disagrees with the error decomposition")
    return VisualNeuralCheckpoint(
        step=step,
        truthful_perception_probability=p_truthful,
        truthful_state_success_probability=truthful_success,
        erroneous_state_success_probability=erroneous_success,
        l_p=l_p,
        l_r=l_r,
        coupling=coupling,
        l_o=l_o,
        iid_accuracy=iid_accuracy,
        ood_accuracy=ood_accuracy,
    )


def _measure(
    model: ModularPerceiverReasoner,
    iid_image_tensor: Any,
    ood_image_tensor: Any,
    paired: _PairedVisualInputs,
    step: int,
    torch: Any,
) -> VisualNeuralCheckpoint:
    with torch.no_grad():
        iid_state, iid_actions = _forward_probabilities(model, iid_image_tensor, torch)
        ood_state, ood_actions = _forward_probabilities(model, ood_image_tensor, torch)
        iid_accuracy = _expected_accuracy(iid_state, iid_actions, paired.iid_correctness, torch)
        ood_accuracy = _expected_accuracy(ood_state, ood_actions, paired.ood_correctness, torch)
    return _checkpoint(
        step,
        float(iid_state[0]),
        float(iid_actions[0, 0]),
        float(iid_actions[1, 1]),
        float(iid_accuracy),
        float(ood_accuracy),
    )


def run_visual_neural_experiment(config: VisualNeuralConfig) -> VisualNeuralRunResult:
    """Train one real CNN/MLP model using expected outcome reward only."""

    if not isinstance(config, VisualNeuralConfig):
        raise TypeError("config must be a VisualNeuralConfig")
    torch = _torch_module()
    paired = _paired_inputs(config)
    iid_image_tensor = _pil_to_tensor(paired.source_image, torch)
    ood_image_tensor = _pil_to_tensor(paired.counterfactual_image, torch)
    model = _build_model(config, torch)
    model.train()
    conv_parameters = _conv_parameters(model, torch)
    reasoning_parameters = tuple(model.reasoning.parameters())
    conv_before = _parameter_snapshot(conv_parameters)
    reasoner_before = _parameter_snapshot(reasoning_parameters)
    optimizer_groups = []
    trainable_perception = tuple(
        parameter for parameter in model.perception.parameters() if parameter.requires_grad
    )
    if trainable_perception:
        optimizer_groups.append(
            {"params": trainable_perception, "lr": config.perception_learning_rate}
        )
    trainable_reasoner = tuple(
        parameter for parameter in reasoning_parameters if parameter.requires_grad
    )
    if trainable_reasoner:
        optimizer_groups.append(
            {"params": trainable_reasoner, "lr": config.reasoning_learning_rate}
        )
    if not optimizer_groups:
        raise AssertionError("the selected training mode has no trainable parameters")
    optimizer = torch.optim.SGD(optimizer_groups)
    history = [
        _measure(
            model,
            iid_image_tensor,
            ood_image_tensor,
            paired,
            0,
            torch,
        )
    ]

    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        iid_state, iid_actions = _forward_probabilities(model, iid_image_tensor, torch)
        expected_reward = _expected_accuracy(
            iid_state,
            iid_actions,
            paired.iid_correctness,
            torch,
        )
        (-expected_reward).backward()
        optimizer.step()
        history.append(
            _measure(
                model,
                iid_image_tensor,
                ood_image_tensor,
                paired,
                step,
                torch,
            )
        )

    final = history[-1]
    initial = history[0]
    equilibrium_mode = (
        "truthful" if final.truthful_perception_probability >= 0.5 else "compensatory"
    )
    return VisualNeuralRunResult(
        profile=str(config.profile),
        mode=str(config.mode),
        seed=config.seed,
        input_source="cva_renderer_pil",
        image_tensor_shape=tuple(iid_image_tensor.shape),
        image_sha256=_image_sha256(paired.source_image),
        ood_image_sha256=_image_sha256(paired.counterfactual_image),
        iid_error_mechanism=config.iid_error_mechanism,
        ood_error_mechanism=config.ood_error_mechanism,
        shifted_factors=paired.shifted_factors,
        paired_sample_id=paired.sample_id,
        iid_correctness_matrix=paired.iid_correctness,
        ood_correctness_matrix=paired.ood_correctness,
        perception_shift=(
            final.truthful_perception_probability - initial.truthful_perception_probability
        ),
        equilibrium_mode=equilibrium_mode,
        iid_accuracy=final.iid_accuracy,
        ood_accuracy=final.ood_accuracy,
        conv_parameter_delta=_parameter_delta(conv_before, conv_parameters),
        reasoner_parameter_delta=_parameter_delta(reasoner_before, reasoning_parameters),
        history=tuple(history),
    )


def _run_config_from_sweep(
    config: VisualNeuralSweepConfig,
    *,
    profile: VisualProfile,
    mode: VisualTrainingMode,
    seed: int,
) -> VisualNeuralConfig:
    return VisualNeuralConfig(
        profile=profile,
        mode=mode,
        seed=seed,
        steps=config.steps,
        image_size=config.image_size,
        hidden_dim=config.hidden_dim,
        perception_learning_rate=config.perception_learning_rate,
        reasoning_learning_rate=config.reasoning_learning_rate,
        device=config.device,
        scene_seed=config.scene_seed,
        iid_error_mechanism=config.iid_error_mechanism,
        ood_error_mechanism=config.ood_error_mechanism,
    )


def _mean(values: tuple[float, ...]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _bootstrap_gap_difference_interval(
    truthful_gaps: tuple[float, ...],
    compensatory_gaps: tuple[float, ...],
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap the difference of paired IID/OOD seed-level gaps by basin."""

    if not truthful_gaps or not compensatory_gaps:
        raise ValueError("both equilibrium groups are required for the OOD bootstrap")
    generator = np.random.default_rng(seed)
    truthful = np.asarray(truthful_gaps, dtype=np.float64)
    compensatory = np.asarray(compensatory_gaps, dtype=np.float64)
    truthful_samples = generator.choice(
        truthful,
        size=(resamples, truthful.size),
        replace=True,
    ).mean(axis=1)
    compensatory_samples = generator.choice(
        compensatory,
        size=(resamples, compensatory.size),
        replace=True,
    ).mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(compensatory_samples - truthful_samples, (alpha, 1.0 - alpha))
    return float(lower), float(upper)


def run_visual_neural_sweep(
    config: VisualNeuralSweepConfig | None = None,
) -> VisualNeuralSweepResult:
    """Run the two fixed-profile controls and at least ten joint seeds."""

    if config is None:
        config = VisualNeuralSweepConfig()
    if not isinstance(config, VisualNeuralSweepConfig):
        raise TypeError("config must be a VisualNeuralSweepConfig or None")
    fixed_seed = config.seeds[0]
    truth_aligned = run_visual_neural_experiment(
        _run_config_from_sweep(
            config,
            profile="truth_aligned",
            mode="perception_only",
            seed=fixed_seed,
        )
    )
    spurious = run_visual_neural_experiment(
        _run_config_from_sweep(
            config,
            profile="spurious",
            mode="perception_only",
            seed=fixed_seed,
        )
    )
    reasoning_only = run_visual_neural_experiment(
        _run_config_from_sweep(
            config,
            profile="spurious",
            mode="reasoning_only",
            seed=fixed_seed,
        )
    )
    joint = tuple(
        run_visual_neural_experiment(
            _run_config_from_sweep(config, profile="spurious", mode="joint", seed=seed)
        )
        for seed in config.seeds
    )
    repeated = run_visual_neural_experiment(
        _run_config_from_sweep(config, profile="spurious", mode="joint", seed=fixed_seed)
    )
    truthful_joint = tuple(result for result in joint if result.equilibrium_mode == "truthful")
    compensatory_joint = tuple(
        result for result in joint if result.equilibrium_mode == "compensatory"
    )
    truthful_gaps = tuple(result.iid_accuracy - result.ood_accuracy for result in truthful_joint)
    compensatory_gaps = tuple(
        result.iid_accuracy - result.ood_accuracy for result in compensatory_joint
    )
    fixed_profiles_opposite = (
        truth_aligned.perception_shift >= config.minimum_profile_shift
        and spurious.perception_shift <= -config.minimum_profile_shift
    )
    two_equilibria = bool(truthful_joint and compensatory_joint)
    all_runs = (truth_aligned, spurious, reasoning_only, *joint)
    three_training_modes = {result.mode for result in all_runs} == {
        "perception_only",
        "reasoning_only",
        "joint",
    }
    perception_frozen = (
        reasoning_only.conv_parameter_delta == 0.0 and reasoning_only.reasoner_parameter_delta > 0.0
    )
    reasoner_frozen = all(
        result.reasoner_parameter_delta == 0.0 for result in (truth_aligned, spurious)
    )
    convolution_updated = all(
        result.conv_parameter_delta > 0.0 for result in (truth_aligned, spurious, *joint)
    )
    minimum_joint_accuracy = min(result.iid_accuracy for result in joint)
    joint_accuracy_high = minimum_joint_accuracy >= config.minimum_joint_iid_accuracy
    mean_truthful_gap = _mean(truthful_gaps)
    mean_compensatory_gap = _mean(compensatory_gaps)
    gap_difference = mean_compensatory_gap - mean_truthful_gap
    if two_equilibria:
        gap_ci_low, gap_ci_high = _bootstrap_gap_difference_interval(
            truthful_gaps,
            compensatory_gaps,
            resamples=config.ood_gap_bootstrap_resamples,
            confidence_level=config.ood_gap_confidence_level,
            seed=config.ood_gap_bootstrap_seed,
        )
    else:
        gap_ci_low = gap_ci_high = 0.0
    ood_gate = (
        two_equilibria
        and gap_difference >= config.minimum_ood_gap_margin
        and gap_ci_low > config.minimum_ood_gap_margin
    )
    reproducible = joint[0] == repeated
    passed = all(
        (
            fixed_profiles_opposite,
            two_equilibria,
            three_training_modes,
            perception_frozen,
            reasoner_frozen,
            convolution_updated,
            joint_accuracy_high,
            ood_gate,
            reproducible,
        )
    )
    gates = VisualNeuralGateSummary(
        fixed_profiles_opposite=fixed_profiles_opposite,
        two_joint_equilibria=two_equilibria,
        three_training_modes=three_training_modes,
        perception_frozen_in_reasoning_only=perception_frozen,
        reasoner_frozen_in_perception_only=reasoner_frozen,
        convolution_updated=convolution_updated,
        joint_iid_accuracy_high=joint_accuracy_high,
        compensatory_ood_gap_larger=ood_gate,
        fixed_seed_reproducible=reproducible,
        truthful_joint_runs=len(truthful_joint),
        compensatory_joint_runs=len(compensatory_joint),
        mean_truthful_ood_gap=mean_truthful_gap,
        mean_compensatory_ood_gap=mean_compensatory_gap,
        ood_gap_difference=gap_difference,
        ood_gap_difference_ci_low=gap_ci_low,
        ood_gap_difference_ci_high=gap_ci_high,
        ood_gap_bootstrap_method="paired_iid_ood_seed_gap_stratified_by_equilibrium",
        ood_gap_bootstrap_resamples=config.ood_gap_bootstrap_resamples,
        ood_gap_confidence_level=config.ood_gap_confidence_level,
        ood_gap_bootstrap_seed=config.ood_gap_bootstrap_seed,
        minimum_joint_iid_accuracy=minimum_joint_accuracy,
        passed=passed,
    )
    return VisualNeuralSweepResult(
        config=config,
        runs=all_runs,
        gates=gates,
    )


def _summary_row(result: VisualNeuralRunResult) -> dict[str, object]:
    initial, final = result.history[0], result.history[-1]
    return {
        "profile": result.profile,
        "mode": result.mode,
        "seed": result.seed,
        "equilibrium_mode": result.equilibrium_mode,
        "initial_p": initial.truthful_perception_probability,
        "final_p": final.truthful_perception_probability,
        "perception_shift": result.perception_shift,
        "truthful_state_success": final.truthful_state_success_probability,
        "erroneous_state_success": final.erroneous_state_success_probability,
        "iid_accuracy": result.iid_accuracy,
        "ood_accuracy": result.ood_accuracy,
        "ood_gap": result.iid_accuracy - result.ood_accuracy,
        "conv_parameter_delta": result.conv_parameter_delta,
        "reasoner_parameter_delta": result.reasoner_parameter_delta,
        "image_sha256": result.image_sha256,
        "ood_image_sha256": result.ood_image_sha256,
        "iid_error_mechanism": result.iid_error_mechanism,
        "ood_error_mechanism": result.ood_error_mechanism,
        "paired_sample_id": result.paired_sample_id,
        "iid_correctness_matrix": json.dumps(result.iid_correctness_matrix),
        "ood_correctness_matrix": json.dumps(result.ood_correctness_matrix),
    }


def _trajectory_rows(sweep: VisualNeuralSweepResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in sweep.runs:
        for checkpoint in result.history:
            rows.append(
                {
                    "profile": result.profile,
                    "mode": result.mode,
                    "seed": result.seed,
                    **asdict(checkpoint),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("CSV rows must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_figure(path: Path, sweep: VisualNeuralSweepResult) -> None:
    # Avoid pyplot and backend mutation; construct an isolated Agg canvas.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8.0, 3.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    trajectory_axis, endpoint_axis = figure.subplots(1, 2)
    for result in sweep.runs:
        if result.mode != "joint":
            continue
        color = "#2878B5" if result.equilibrium_mode == "truthful" else "#D95319"
        trajectory_axis.plot(
            [checkpoint.step for checkpoint in result.history],
            [checkpoint.truthful_perception_probability for checkpoint in result.history],
            color=color,
            alpha=0.55,
            linewidth=1.1,
        )
    trajectory_axis.axhline(0.5, color="#555555", linestyle="--", linewidth=0.8)
    trajectory_axis.set(xlabel="Step", ylabel="Truthful perception p", ylim=(0.0, 1.0))
    joint = tuple(result for result in sweep.runs if result.mode == "joint")
    endpoint_axis.scatter(
        [result.iid_accuracy for result in joint],
        [result.ood_accuracy for result in joint],
        c=["#2878B5" if result.equilibrium_mode == "truthful" else "#D95319" for result in joint],
        s=28,
    )
    endpoint_axis.plot((0.0, 1.0), (0.0, 1.0), color="#777777", linestyle=":")
    endpoint_axis.set(
        xlabel="IID expected accuracy",
        ylabel="Mechanism-OOD expected accuracy",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, format="png")
    figure.clear()


def write_visual_neural_artifacts(
    sweep: VisualNeuralSweepResult,
    *,
    output_root: str | os.PathLike[str] | None = None,
    paths: VisualNeuralArtifactPaths | None = None,
    provenance: Mapping[str, object] | None = None,
    overwrite: bool = False,
) -> VisualNeuralArtifactPaths:
    """Persist JSON, two CSV tables, and a PNG without mutating ``sweep``."""

    if not isinstance(sweep, VisualNeuralSweepResult):
        raise TypeError("sweep must be a VisualNeuralSweepResult")
    if (output_root is None) == (paths is None):
        raise ValueError("provide exactly one of output_root or paths")
    if paths is None:
        if isinstance(output_root, bool):
            raise TypeError("output_root must be path-like")
        try:
            root = Path(output_root)  # type: ignore[arg-type]
        except TypeError as error:
            raise TypeError("output_root must be path-like") from error
        if root.exists() and not root.is_dir():
            raise ValueError("output_root must be a directory")
        paths = VisualNeuralArtifactPaths(
            metrics_json=root / "artifacts/metrics/visual_neural_summary.json",
            runs_csv=root / "artifacts/predictions/visual_neural_runs.csv",
            trajectories_csv=root / "artifacts/predictions/visual_neural_trajectories.csv",
            figure_png=root / "artifacts/figures/visual_neural_trajectories.png",
        )
    elif not isinstance(paths, VisualNeuralArtifactPaths):
        raise TypeError("paths must be a VisualNeuralArtifactPaths")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be boolean")
    registered_paths = tuple(Path(value) for value in asdict(paths).values())
    existing = tuple(path for path in registered_paths if path.exists())
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"artifact already exists; set overwrite=True to replace it: {rendered}"
        )
    if provenance is not None and not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping or None")
    summary_rows = [_summary_row(result) for result in sweep.runs]
    trajectory_rows = _trajectory_rows(sweep)
    _write_csv(paths.runs_csv, summary_rows)
    _write_csv(paths.trajectories_csv, trajectory_rows)
    fixed = [row for row in summary_rows if row["mode"] == "perception_only"]
    freeze_controls = [row for row in summary_rows if row["mode"] != "joint"]
    joint = [row for row in summary_rows if row["mode"] == "joint"]
    payload = {
        "schema_version": 1,
        "experiment": "visual_neural_phase_c",
        "config": asdict(sweep.config),
        "gates": asdict(sweep.gates),
        "provenance": {
            "input_source": "cva_renderer_pil",
            "model": "ModularPerceiverReasoner",
            "training_objective": "outcome_only_expected_reward",
            "reasoner_intervention": "injected_one_hot_state",
            "ood_evaluation": "paired_catalog_execution",
            "shifted_factors": ["error_mechanism"],
            "paired_images_byte_identical": all(
                result.image_sha256 == result.ood_image_sha256 for result in sweep.runs
            ),
            **({} if provenance is None else dict(provenance)),
        },
        "fixed_profile_runs": fixed,
        "freeze_control_runs": freeze_controls,
        "joint_runs": joint,
    }
    paths.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    paths.metrics_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_figure(paths.figure_png, sweep)
    return paths


__all__ = [
    "VisualNeuralArtifactPaths",
    "VisualNeuralCheckpoint",
    "VisualNeuralConfig",
    "VisualNeuralGateSummary",
    "VisualNeuralRunResult",
    "VisualNeuralSweepConfig",
    "VisualNeuralSweepResult",
    "run_visual_neural_experiment",
    "run_visual_neural_sweep",
    "write_visual_neural_artifacts",
]
