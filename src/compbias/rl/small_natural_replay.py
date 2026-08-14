"""Multi-scene PIL/CNN natural-mediator replay experiment for v2 Phase 1."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from compbias.envs.cva_world.generator import GeneratorConfig, generate_dataset
from compbias.envs.cva_world.renderer import RenderConfig, render_sample
from compbias.envs.cva_world.schema import SemanticSplit, TaskFamily
from compbias.interventions.transport_audit import audit_synthetic_transport
from compbias.models.modular_neural import ModularPerceiverReasoner, set_training_mode
from compbias.theory.crossed_risk import CrossedRiskResult, crossed_risk_decomposition


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


@dataclass(frozen=True, slots=True)
class SmallNaturalReplayConfig:
    """Frozen confirmatory design, with smaller values allowed only for tests/pilots."""

    samples_per_family_per_split: int = 40
    realizations_per_semantic: int = 8
    n_mediators: int = 32
    n_forks: int = 32
    training_seeds: tuple[int, ...] = tuple(range(20))
    training_steps: int = 16
    batch_size: int = 256
    image_size: int = 16
    hidden_dim: int = 8
    learning_rate: float = 0.20
    bootstrap_draws: int = 10_000
    data_seed: int = 20_260_814
    synthetic_error_mass: float = 0.80
    confirmatory: bool = True
    splits: tuple[SemanticSplit, ...] = tuple(SemanticSplit)
    task_families: tuple[TaskFamily, ...] = tuple(TaskFamily)

    def __post_init__(self) -> None:
        _integer(
            self.samples_per_family_per_split,
            "samples_per_family_per_split",
            1,
            200,
        )
        _integer(self.realizations_per_semantic, "realizations_per_semantic", 2, 16)
        _integer(self.n_mediators, "n_mediators", 2, 64)
        _integer(self.n_forks, "n_forks", 2, 64)
        _integer(self.training_steps, "training_steps", 1, 128)
        _integer(self.batch_size, "batch_size", 4, 4096)
        _integer(self.image_size, "image_size", 8, 64)
        _integer(self.hidden_dim, "hidden_dim", 2, 64)
        _integer(self.bootstrap_draws, "bootstrap_draws", 1_000, 100_000)
        _integer(self.data_seed, "data_seed", 0, 2**31 - 1)
        object.__setattr__(self, "learning_rate", _positive(self.learning_rate, "learning_rate"))
        if (
            isinstance(self.synthetic_error_mass, bool)
            or not isinstance(self.synthetic_error_mass, Real)
            or not 0.5 < float(self.synthetic_error_mass) < 1.0
        ):
            raise ValueError("synthetic_error_mass must lie strictly between 0.5 and 1")
        object.__setattr__(self, "synthetic_error_mass", float(self.synthetic_error_mass))
        if not isinstance(self.confirmatory, bool):
            raise TypeError("confirmatory must be boolean")
        seeds = tuple(self.training_seeds)
        if (
            not seeds
            or len(seeds) > 64
            or len(set(seeds)) != len(seeds)
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds
            )
        ):
            raise ValueError("training_seeds must contain 1 to 64 unique non-negative integers")
        object.__setattr__(self, "training_seeds", seeds)
        try:
            splits = tuple(SemanticSplit(value) for value in self.splits)
            families = tuple(TaskFamily(value) for value in self.task_families)
        except (TypeError, ValueError) as error:
            raise ValueError("splits and task_families must use registered values") from error
        if not splits or len(set(splits)) != len(splits):
            raise ValueError("splits must be non-empty and unique")
        if not families or len(set(families)) != len(families):
            raise ValueError("task_families must be non-empty and unique")
        object.__setattr__(self, "splits", splits)
        object.__setattr__(self, "task_families", families)
        workload = self.semantic_state_count * len(seeds) * self.n_mediators * self.n_forks
        if workload > 50_000_000:
            raise ValueError("natural replay workload exceeds the 50M continuation budget")
        if self.confirmatory and (
            self.semantic_state_count < 1_000
            or self.realizations_per_semantic < 8
            or self.n_mediators < 32
            or self.n_forks < 32
            or len(seeds) < 20
            or len(families) < 5
        ):
            raise ValueError("confirmatory design must meet every v2 Phase-1 minimum")

    @property
    def semantic_state_count(self) -> int:
        return self.samples_per_family_per_split * len(self.splits) * len(self.task_families)

    @property
    def visual_realization_count(self) -> int:
        return self.semantic_state_count * self.realizations_per_semantic


@dataclass(frozen=True, slots=True)
class ErrorFamilyCompensability:
    error_family: str
    natural_mediator_count: int
    c_sel: float
    c_fork: float
    c_syn: float
    mediator_gap: float
    transport_gap: float


@dataclass(frozen=True, slots=True)
class SmallNaturalReplaySeedResult:
    seed: int
    model_sha256: str
    c_sel_error: float
    c_fork_error: float
    c_syn_error: float
    mediator_gap: float
    transport_gap: float
    initial_error_probability: float
    final_error_probability: float
    selection_error_ratio: float
    iid_accuracy: float
    ood_accuracy: float
    crossed_risk: CrossedRiskResult
    transport_two_sample_accuracy: float
    transport_off_support: bool
    by_error_family: tuple[ErrorFamilyCompensability, ...]


@dataclass(frozen=True, slots=True)
class SmallNaturalReplayResult:
    config: SmallNaturalReplayConfig
    input_source: str
    model_path: str
    semantic_state_count: int
    visual_realization_count: int
    natural_mediator_count: int
    forked_continuation_count: int
    synthetic_mediator_count: int
    error_families: tuple[str, ...]
    seed_results: tuple[SmallNaturalReplaySeedResult, ...]


def _torch_module() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - base-only installation
        raise ModuleNotFoundError(
            "small natural replay requires torch; install compbias[neural]"
        ) from error
    return torch


def _pil_tensor(samples: tuple[Any, ...], config: SmallNaturalReplayConfig, torch: Any) -> Any:
    images: list[np.ndarray] = []
    for index, sample in enumerate(samples):
        image = render_sample(
            sample,
            RenderConfig(
                width=config.image_size,
                height=config.image_size,
                style=sample.split_keys.visual_style,
                seed=config.data_seed + index,
            ),
        )
        pixels = np.asarray(image, dtype=np.float32)
        if pixels.shape != (config.image_size, config.image_size, 3):
            raise AssertionError("CVA renderer returned an unexpected RGB image shape")
        images.append(np.array(pixels, copy=True))
    array = np.stack(images, axis=0).transpose(0, 3, 1, 2) / 255.0
    return torch.from_numpy(np.ascontiguousarray(array))


def _initialize_model(config: SmallNaturalReplayConfig, seed: int, torch: Any) -> Any:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ModularPerceiverReasoner(
            image_channels=3,
            image_size=config.image_size,
            num_perceived_states=2,
            num_reasoning_actions=2,
            hidden_dim=config.hidden_dim,
        )
    set_training_mode(model, "joint")
    first = model.reasoning[0]
    final = model.reasoning[2]
    with torch.no_grad():
        for parameter in model.reasoning.parameters():
            parameter.zero_()
        first.weight[0, 0] = 1.0
        first.weight[1, 1] = 1.0
        final.weight[0, 0] = math.log(0.75 / 0.25)
        final.weight[1, 1] = math.log(0.90 / 0.10)
    return model


def _probabilities(model: Any, images: Any, torch: Any) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        natural = model(images=images)
        state = torch.softmax(natural["perception_logits"], dim=-1).cpu().numpy()
        injected = model(perceived_state=torch.tensor([0, 1], dtype=torch.long))
        actions = torch.softmax(injected["reasoning_logits"], dim=-1).cpu().numpy()
    return np.array(state, copy=True), np.array(actions, copy=True)


def _train(
    config: SmallNaturalReplayConfig,
    images: Any,
    *,
    seed: int,
    torch: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    model = _initialize_model(config, seed, torch)
    initial_state, _ = _probabilities(model, images, torch)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.data_seed + seed * 104_729)
    model.train()
    for _ in range(config.training_steps):
        indices = torch.randint(
            0,
            len(images),
            (min(config.batch_size, len(images)),),
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        natural = model(images=images[indices])
        state = torch.softmax(natural["perception_logits"], dim=-1)
        injected = model(perceived_state=torch.tensor([0, 1], dtype=torch.long))
        actions = torch.softmax(injected["reasoning_logits"], dim=-1)
        expected_reward = torch.mean(state[:, 0] * actions[0, 0] + state[:, 1] * actions[1, 1])
        (-expected_reward).backward()
        optimizer.step()
    final_state, action_probabilities = _probabilities(model, images, torch)
    synthetic_state = torch.tensor(
        [[1.0 - config.synthetic_error_mass, config.synthetic_error_mass]],
        dtype=images.dtype,
    )
    with torch.no_grad():
        synthetic_actions = (
            torch.softmax(model(perceived_state=synthetic_state)["reasoning_logits"], dim=-1)
            .cpu()
            .numpy()[0]
        )
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return (
        initial_state,
        final_state,
        np.vstack((action_probabilities, synthetic_actions)),
        digest.hexdigest(),
    )


def _semantic_key(sample: Any) -> str:
    marker = sample.sample_id.rsplit("_r", 1)
    if len(marker) != 2:
        raise ValueError("generated sample_id lacks a realization suffix")
    return marker[0]


def _seed_result(
    config: SmallNaturalReplayConfig,
    samples: tuple[Any, ...],
    initial_state: np.ndarray,
    final_state: np.ndarray,
    actions: np.ndarray,
    *,
    seed: int,
    model_sha256: str,
) -> SmallNaturalReplaySeedResult:
    groups: dict[str, list[int]] = {}
    families: dict[str, str] = {}
    for index, sample in enumerate(samples):
        key = _semantic_key(sample)
        groups.setdefault(key, []).append(index)
        families[key] = sample.task_family.value
    ordered = tuple(sorted(groups))
    initial_error = np.asarray(
        [initial_state[groups[key], 1].mean() for key in ordered], dtype=np.float64
    )
    final_error = np.asarray(
        [final_state[groups[key], 1].mean() for key in ordered], dtype=np.float64
    )
    rng = np.random.default_rng(config.data_seed + seed * 65_537)
    error_counts = rng.binomial(config.n_mediators, final_error)
    truth_counts = config.n_mediators - error_counts
    truth_success_probability = float(actions[0, 0])
    error_success_probability = float(actions[1, 1])
    error_successes = rng.binomial(error_counts, error_success_probability)
    if int(error_counts.sum()) == 0 or int(truth_counts.sum()) == 0:
        raise AssertionError("natural sampling did not cover both truth and error states")
    c_sel_error = float(error_successes.sum() / error_counts.sum())
    fork_error_successes = rng.binomial(
        error_counts * config.n_forks,
        error_success_probability,
    )
    c_fork_error = float(fork_error_successes.sum() / (error_counts.sum() * config.n_forks))
    synthetic_success_probability = float(actions[2, 1])
    synthetic_successes = rng.binomial(
        config.n_forks,
        synthetic_success_probability,
        size=len(ordered),
    )
    c_syn_error = float(synthetic_successes.mean() / config.n_forks)

    by_error_family: list[ErrorFamilyCompensability] = []
    for family in sorted(set(families.values())):
        family_indices = np.asarray(
            [index for index, key in enumerate(ordered) if families[key] == family],
            dtype=np.int64,
        )
        family_error_count = int(error_counts[family_indices].sum())
        if family_error_count == 0:
            raise AssertionError(f"natural sampling has no error mediator for family {family!r}")
        family_c_sel = float(error_successes[family_indices].sum() / family_error_count)
        family_c_fork = float(
            fork_error_successes[family_indices].sum() / (family_error_count * config.n_forks)
        )
        family_c_syn = float(synthetic_successes[family_indices].mean() / config.n_forks)
        by_error_family.append(
            ErrorFamilyCompensability(
                error_family=family,
                natural_mediator_count=family_error_count,
                c_sel=family_c_sel,
                c_fork=family_c_fork,
                c_syn=family_c_syn,
                mediator_gap=family_c_sel - family_c_fork,
                transport_gap=family_c_syn - family_c_fork,
            )
        )

    mean_initial_error = float(initial_error.mean())
    mean_final_error = float(final_error.mean())
    selection_ratio = mean_final_error / mean_initial_error
    iid_accuracy = float(
        np.mean(
            (1.0 - final_error) * truth_success_probability
            + final_error * error_success_probability
        )
    )
    ood_accuracy = float(np.mean((1.0 - final_error) * truth_success_probability))
    crossed = crossed_risk_decomposition(
        l_mm=1.0 - iid_accuracy,
        l_om=1.0 - truth_success_probability,
        l_mo=mean_final_error,
        l_oo=0.0,
    )

    natural_signature_rows: list[tuple[float, ...]] = []
    natural_reward_rows: list[float] = []
    natural_types: list[str] = []
    for index, key in enumerate(ordered):
        count = int(error_counts[index])
        natural_signature_rows.extend(
            [(0.0, 1.0, float(actions[1, 0]), float(actions[1, 1]))] * count
        )
        natural_reward_rows.extend([error_success_probability] * count)
        natural_types.extend([families[key]] * count)
    synthetic_signature = (
        1.0 - config.synthetic_error_mass,
        config.synthetic_error_mass,
        float(actions[2, 0]),
        float(actions[2, 1]),
    )
    synthetic_signatures = np.tile(np.asarray(synthetic_signature), (len(ordered), 1))
    transport = audit_synthetic_transport(
        natural_signatures=np.asarray(natural_signature_rows),
        synthetic_signatures=synthetic_signatures,
        natural_rewards=np.asarray(natural_reward_rows),
        synthetic_rewards=np.full(len(ordered), synthetic_success_probability),
        natural_error_types=tuple(natural_types),
        synthetic_error_types=tuple(families[key] for key in ordered),
        bootstrap_draws=config.bootstrap_draws,
        confidence=0.95,
        seed=config.data_seed + seed,
    )
    return SmallNaturalReplaySeedResult(
        seed=seed,
        model_sha256=model_sha256,
        c_sel_error=c_sel_error,
        c_fork_error=c_fork_error,
        c_syn_error=c_syn_error,
        mediator_gap=c_sel_error - c_fork_error,
        transport_gap=c_syn_error - c_fork_error,
        initial_error_probability=mean_initial_error,
        final_error_probability=mean_final_error,
        selection_error_ratio=selection_ratio,
        iid_accuracy=iid_accuracy,
        ood_accuracy=ood_accuracy,
        crossed_risk=crossed,
        transport_two_sample_accuracy=transport.two_sample_accuracy,
        transport_off_support=transport.off_support_stress_test,
        by_error_family=tuple(by_error_family),
    )


def run_small_natural_replay(
    config: SmallNaturalReplayConfig | None = None,
) -> SmallNaturalReplayResult:
    """Run the CPU naturalization experiment using PIL pixels and a real CNN/MLP."""

    if config is None:
        config = SmallNaturalReplayConfig()
    if not isinstance(config, SmallNaturalReplayConfig):
        raise TypeError("config must be SmallNaturalReplayConfig or None")
    generator_config = GeneratorConfig(
        seed=config.data_seed,
        samples_per_family_per_split=config.samples_per_family_per_split,
        splits=config.splits,
        task_families=config.task_families,
        realizations_per_semantic=config.realizations_per_semantic,
    )
    samples = generate_dataset(generator_config)
    if len(samples) != config.visual_realization_count:
        raise AssertionError("generated visual realization count disagrees with the frozen design")
    torch = _torch_module()
    images = _pil_tensor(samples, config, torch)
    seed_results: list[SmallNaturalReplaySeedResult] = []
    for seed in config.training_seeds:
        initial_state, final_state, actions, model_sha256 = _train(
            config,
            images,
            seed=seed,
            torch=torch,
        )
        seed_results.append(
            _seed_result(
                config,
                samples,
                initial_state,
                final_state,
                actions,
                seed=seed,
                model_sha256=model_sha256,
            )
        )
    error_families = tuple(sorted(family.value for family in config.task_families))
    seed_count = len(config.training_seeds)
    return SmallNaturalReplayResult(
        config=config,
        input_source="cva_renderer_pil",
        model_path="cnn_perceiver_to_image_blind_mlp_reasoner",
        semantic_state_count=config.semantic_state_count,
        visual_realization_count=len(samples),
        natural_mediator_count=config.semantic_state_count * config.n_mediators * seed_count,
        forked_continuation_count=(
            config.semantic_state_count * config.n_mediators * config.n_forks * seed_count
        ),
        synthetic_mediator_count=config.semantic_state_count * seed_count,
        error_families=error_families,
        seed_results=tuple(seed_results),
    )
