"""Finite-action CPU laboratory, isolated from the production Qwen adapters.

The only input to the shared scorer is a 37-feature prompt and a 7-feature
candidate. Event labels are used outside the scorer for rewards and evaluation.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.constraint_solver import solve
from src.generate_worlds import FAMILIES, OPERATIONS, generate_dataset
from src.verifiers import executor, fiber_size

INTERFACES = ("SYMBOLIC_PROXY", "EVIDENCE_PROXY")
EVENTS = ("X", "S", "W", "I")
PAIRS = tuple(itertools.combinations(range(4), 2))
LOG_FIELDS = (
    "update_norm",
    "grad_norm_preclip",
    "grad_norm_postclip",
    "optimizer_step",
    "train_X_fraction",
    "train_S_fraction",
    "train_W_fraction",
    "train_I_fraction",
    "zero_advantage_fraction",
    "no_x_mixed_fraction",
    "operation_index",
)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def prompt_features(scene: dict, interface: str) -> list[float]:
    if interface not in INTERFACES:
        raise ValueError("unknown toy interface")
    cue = scene["cue"]
    known_mask, known = [0.0] * 4, [0.0] * 4
    if "known_index" in cue:
        known_mask[cue["known_index"]] = 1.0
        known[cue["known_index"]] = cue["known_value"] / 99.0
    edges = {tuple(sorted((a, b))): value for a, b, value in cue.get("edges", [])}
    evidence = interface == "EVIDENCE_PROXY"
    values = (
        [v / 99.0 for v in scene["observed_world"]]
        + [float(scene["constraint_family"] == f) for f in FAMILIES]
        + known_mask
        + known
        + [float(pair in edges) for pair in PAIRS]
        + [edges.get(pair, 0) / 198.0 for pair in PAIRS]
        + [float(interface == name) for name in INTERFACES]
        + ([v / 99.0 for v in scene["truth_world"]] if evidence else [0.0] * 4)
        + [float(evidence)]
        + [float(scene["operation"] == name) for name in OPERATIONS]
    )
    if len(values) != 37:
        raise AssertionError("prompt feature contract violated")
    return values


def same_answer_alternatives(truth, operation, limit=2):
    """Deterministic full-domain answer-fiber construction, including boundaries."""
    answer = executor(truth, operation)
    truth = tuple(truth)
    result = []

    def add(world):
        world = tuple(world)
        if world != truth and world not in result and executor(world, operation) == answer:
            result.append(world)
        return len(result) >= limit

    if operation == "range4":
        # These realize every possible range, including range zero.
        for lower in range(100 - answer):
            for pattern in itertools.product((0, answer), repeat=4):
                if max(pattern) - min(pattern) != answer:
                    continue
                if add(tuple(lower + v for v in pattern)):
                    return [list(x) for x in result]
    else:
        for a in range(100):
            for b in range(100):
                for c in range(100):
                    d = answer - a - b - c if operation == "sum4" else a + b - c - answer
                    if 0 <= d <= 99 and add((a, b, c, d)):
                        return [list(x) for x in result]
    return [list(x) for x in result]


def _candidates(scene, seed):
    truth = tuple(scene["truth_world"])
    observed = tuple(scene["observed_world"])
    operation, answer = scene["operation"], executor(truth, scene["operation"])
    worlds = [truth, observed]
    alternatives = same_answer_alternatives(truth, operation, limit=3)
    if not alternatives and fiber_size(answer, operation) > 1:
        raise RuntimeError("nonempty S fiber could not be constructed")
    for candidate in alternatives:
        if sum(executor(w, operation) == answer and w != truth for w in worlds) >= 2:
            break
        if tuple(candidate) not in worlds:
            worlds.append(tuple(candidate))
    rng = np.random.default_rng(int(_digest([seed, scene["base_scene_id"], "candidates"])[:16], 16))
    while len(worlds) < 14:
        candidate = tuple(int(v) for v in rng.integers(0, 100, size=4))
        if candidate not in worlds and executor(candidate, operation) != answer:
            worlds.append(candidate)
    actions = [list(w) for w in worlds] + ["INVALID_A", "INVALID_B"]
    order = rng.permutation(16)
    actions = [actions[int(index)] for index in order]
    categories = [
        3
        if isinstance(w, str)
        else 0
        if tuple(w) == truth
        else 1
        if executor(w, operation) == answer
        else 2
        for w in actions
    ]
    return actions, categories


@dataclass(frozen=True)
class ToyDataset:
    train_features: np.ndarray
    train_categories: np.ndarray
    probe_features: np.ndarray
    probe_categories: np.ndarray
    train_metadata: list[dict]
    probe_metadata: list[dict]
    scenes: dict


def build_toy_dataset(out: Path, seed=20260915) -> ToyDataset:
    out = Path(out)
    generate_dataset(out, seed=seed, sizes={"train": 72, "control": 36}, render=False)
    scenes = {
        name: [json.loads(line) for line in (out / f"{name}.jsonl").read_text().splitlines()]
        for name in ("train", "control")
    }
    all_scenes = scenes["train"] + scenes["control"]
    if len({tuple(s["truth_world"]) for s in all_scenes}) != 108:
        raise ValueError("train/probe truth collision")
    if any(solve(s["observed_world"], s["cue"]) != [s["truth_world"]] for s in all_scenes):
        raise ValueError("unique repair validation failed")
    values = {}
    for split in ("train", "control"):
        features, categories, metadata = [], [], []
        for scene in scenes[split]:
            interfaces = (
                ((INTERFACES[0] if scene["interface"] == "SYMBOLIC_FRESH" else INTERFACES[1]),)
                if split == "train"
                else INTERFACES
            )
            actions, labels = _candidates(scene, seed)
            for interface in interfaces:
                prompt = prompt_features(scene, interface)
                candidate_features = [
                    ([v / 99.0 for v in a] + [1.0, 0.0, 0.0])
                    if isinstance(a, list)
                    else [0.0] * 4 + [0.0, float(a == "INVALID_A"), float(a == "INVALID_B")]
                    for a in actions
                ]
                features.append([prompt + candidate for candidate in candidate_features])
                categories.append(labels)
                metadata.append(
                    {
                        "prompt_id": _digest([scene["base_scene_id"], interface])[:32],
                        "base_scene_id": scene["base_scene_id"],
                        "interface": interface,
                        "constraint_family": scene["constraint_family"],
                        "operation": scene["operation"],
                        "chart_type": scene["chart_type"],
                        "truth_structure_hash": scene["truth_structure_hash"],
                        "group": FAMILIES.index(scene["constraint_family"]) * 2
                        + INTERFACES.index(interface),
                        "weight": 1.0 / 72,
                        "solution_count": scene["solution_count"],
                        "S_status": "STRUCTURALLY_EMPTY_S" if 1 not in labels else "PRESENT",
                        "actions": actions,
                        "event_categories": [EVENTS[c] for c in labels],
                    }
                )
        values[split] = (
            np.asarray(features, dtype=np.float64),
            np.asarray(categories, dtype=np.int8),
            metadata,
        )
    return ToyDataset(
        *values["train"][:2],
        *values["control"][:2],
        values["train"][2],
        values["control"][2],
        scenes,
    )


def make_model(seed=7001):
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = torch.nn.Sequential(
            torch.nn.Linear(44, 16, dtype=torch.float64),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 1, dtype=torch.float64),
        ).to(device="cpu")
    return model


def flatten_parameters(model):
    return np.concatenate([p.detach().numpy().ravel() for p in model.parameters()]).copy()


def set_parameters(model, theta):
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (737,) or not np.isfinite(theta).all():
        raise ValueError("finite 737-vector required")
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.from_numpy(theta[offset : offset + p.numel()].reshape(p.shape)))
            offset += p.numel()


def parameter_layout(model):
    offset, rows = 0, []
    for name, p in model.named_parameters():
        rows.append(
            {
                "name": name,
                "shape": list(p.shape),
                "offset": offset,
                "numel": p.numel(),
                "dtype": "float64",
            }
        )
        offset += p.numel()
    return rows


def event_probabilities(model, features, categories):
    x = torch.as_tensor(features, dtype=torch.float64, device="cpu")
    labels = torch.as_tensor(categories, dtype=torch.long, device="cpu")
    scores = model(x).squeeze(-1)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.stack([(probabilities * (labels == event)).sum(-1) for event in range(4)], dim=-1)


def _numpy_forward(theta, features, categories):
    theta, x, c = np.asarray(theta), np.asarray(features), np.asarray(categories)
    if theta.shape != (737,) or x.shape[-2:] != (16, 44) or c.shape != x.shape[:-1]:
        raise ValueError("invalid parameter, candidate feature or identity shape")
    hidden = np.tanh(x @ theta[:704].reshape(16, 44).T + theta[704:720])
    score = hidden @ theta[720:736] + theta[736]
    score = score - score.max(-1, keepdims=True)
    action = np.exp(score)
    action /= action.sum(-1, keepdims=True)
    indicator = np.eye(4)[c]
    p = np.einsum("na,nae->ne", action, indicator)
    return hidden, action, indicator, p


def event_probabilities_from_theta(theta, features, categories):
    return _numpy_forward(theta, features, categories)[3]


def event_jacobian(theta, features, categories):
    """Exact analytic probability Jacobian [prompt,event,parameter], oracle only."""
    theta, x = np.asarray(theta), np.asarray(features)
    hidden, action, indicator, p = _numpy_forward(theta, x, categories)
    hidden_derivative = (1 - hidden**2) * theta[720:736]
    ds_w1 = (hidden_derivative[..., :, None] * x[..., None, :]).reshape(len(x), 16, 704)
    score_jacobian = np.concatenate(
        (ds_w1, hidden_derivative, hidden, np.ones((*hidden.shape[:-1], 1))), axis=-1
    )
    weights = action[..., None] * (indicator - p[:, None, :])
    return np.einsum("nae,nap->nep", weights, score_jacobian, optimize=True)


def sample_bank(model, features, categories, rng, B=4, K=8, prompt_indices=None):
    indices = (
        rng.integers(len(features), size=B)
        if prompt_indices is None
        else np.asarray(prompt_indices)
    )
    with torch.no_grad():
        logits = model(torch.as_tensor(features[indices], dtype=torch.float64)).squeeze(-1)
        logp = torch.log_softmax(logits, -1).numpy()
    actions = np.asarray([rng.choice(16, size=K, p=np.exp(row)) for row in logp])
    return {
        "features": features[indices].copy(),
        "prompt_indices": indices.copy(),
        "actions": actions,
        "categories": np.take_along_axis(categories[indices], actions, axis=1),
        "old_logp": np.take_along_axis(logp, actions, axis=1),
    }


def snapshot(model, optimizer, rng):
    return {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "numpy_rng": copy.deepcopy(rng.bit_generator.state),
        "torch_rng": torch.random.get_rng_state().clone(),
        "gradients": [
            None if p.grad is None else p.grad.detach().clone() for p in model.parameters()
        ],
    }


def restore(model, optimizer, rng, state):
    model.load_state_dict(state["model"])
    # Optimizer loading may retain tensor references when dtype/device match.
    # Copy on every fork so an update cannot mutate the frozen anchor itself.
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    for p, gradient in zip(model.parameters(), state["gradients"], strict=True):
        p.grad = None if gradient is None else gradient.clone()
    rng.bit_generator.state = copy.deepcopy(state["numpy_rng"])
    torch.random.set_rng_state(state["torch_rng"])


def adam_vectors(model, optimizer):
    m, v, steps = [], [], []
    for p in model.parameters():
        state = optimizer.state.get(p, {})
        m.append(state.get("exp_avg", torch.zeros_like(p)).detach().numpy().ravel())
        v.append(state.get("exp_avg_sq", torch.zeros_like(p)).detach().numpy().ravel())
        step = state.get("step", 0)
        steps.append(int(step.item()) if hasattr(step, "item") else int(step))
    if len(set(steps)) != 1:
        raise RuntimeError("optimizer parameter steps differ")
    return np.concatenate(m).copy(), np.concatenate(v).copy(), steps[0]


def perform_step(model, optimizer, bank, lam=0.0, policy="joint", operation_index=0):
    labels = torch.as_tensor(bank["categories"], dtype=torch.long)
    auxiliary = torch.full((len(labels), 1), float(lam), dtype=torch.float64)
    if policy == "no_x_off_before_joint_normalization":
        auxiliary *= (labels == 0).any(-1, keepdim=True)
    elif policy != "joint":
        raise ValueError("unknown intervention policy")
    rewards = 2.0 * (labels == 0) + auxiliary * (labels != 3)
    centered = rewards - rewards.mean(-1, keepdim=True)
    variance = centered.square().mean(-1, keepdim=True)
    advantages = (centered / torch.sqrt(variance + 1e-8)).detach()
    optimizer.zero_grad(set_to_none=False)
    before = flatten_parameters(model)
    new_logp = torch.log_softmax(
        model(torch.as_tensor(bank["features"], dtype=torch.float64)).squeeze(-1), -1
    )
    selected = new_logp.gather(-1, torch.as_tensor(bank["actions"], dtype=torch.long))
    ratio = (selected - torch.as_tensor(bank["old_logp"], dtype=torch.float64)).exp()
    loss = -torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages).mean()
    loss.backward()
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    g = np.concatenate([p.grad.detach().numpy().ravel() for p in model.parameters()]).copy()
    pre = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
    post = float(np.sqrt(sum(float(p.grad.square().sum()) for p in model.parameters())))
    g_clipped = np.concatenate([p.grad.detach().numpy().ravel() for p in model.parameters()]).copy()
    optimizer.step()
    d = flatten_parameters(model) - before
    if not np.isfinite(d).all():
        raise FloatingPointError("nonfinite CPU update")
    zero = float((variance == 0).double().mean())
    no_x_mixed = float(
        (~(labels == 0).any(-1) & (labels == 3).any(-1) & (labels != 3).any(-1)).double().mean()
    )
    counts = np.bincount(labels.numpy().ravel(), minlength=4)
    logs = np.asarray(
        [
            np.linalg.norm(d),
            pre,
            post,
            adam_vectors(model, optimizer)[2],
            *(counts / counts.sum()),
            zero,
            no_x_mixed,
            operation_index,
        ],
        dtype=np.float64,
    )
    return {
        "d": d,
        "g": g,
        "g_clipped": g_clipped,
        "logs": logs,
        "loss": float(loss.detach()),
        "loss_denominator": labels.numel(),
        "zero_advantage_fraction": zero,
        "train_category_counts": counts,
        "advantages": advantages.numpy(),
        "rewards": rewards.numpy(),
        "grad_norm_preclip": pre,
        "grad_norm_postclip": post,
    }
