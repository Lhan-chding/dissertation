"""Scene-paired factorial effects, cluster bootstrap, and Holm correction."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[int(quantile * (len(ordered) - 1))]


def _pair_effects(rows: Sequence[Mapping[str, object]], *, outcome: str) -> dict[str, list[float]]:
    grouped: dict[str, dict[tuple[str, str, str], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        pair_id = row.get("pair_id")
        verifier = row.get("verifier")
        validity = row.get("validity_channel")
        decoder = row.get("eval_decoder")
        value = row.get(outcome)
        if (
            not isinstance(pair_id, str)
            or verifier not in {"answer", "state"}
            or validity not in {"binary", "lex"}
            or decoder not in {"free", "constrained"}
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError("Study C3 factorial row is malformed")
        grouped[pair_id][(str(verifier), str(validity), str(decoder))].append(float(value))
    required = {
        (verifier, validity, decoder)
        for verifier in ("answer", "state")
        for validity in ("binary", "lex")
        for decoder in ("free", "constrained")
    }
    effects: dict[str, list[float]] = defaultdict(list)
    for pair_id, cells in sorted(grouped.items()):
        replicate_counts = {len(values) for values in cells.values()}
        if set(cells) != required or len(replicate_counts) != 1 or 0 in replicate_counts:
            raise ValueError(f"pair {pair_id} lacks the full paired condition/factor grid")
        mean = {key: sum(values) / len(values) for key, values in cells.items()}

        def average(
            selected: Sequence[tuple[str, str, str]],
            *,
            cells: Mapping[tuple[str, str, str], float] = mean,
        ) -> float:
            return sum(cells[key] for key in selected) / len(selected)

        state = [key for key in required if key[0] == "state"]
        answer = [key for key in required if key[0] == "answer"]
        lex = [key for key in required if key[1] == "lex"]
        binary = [key for key in required if key[1] == "binary"]
        constrained = [key for key in required if key[2] == "constrained"]
        free = [key for key in required if key[2] == "free"]
        effects["verifier"].append(average(state) - average(answer))
        effects["validity_channel"].append(average(lex) - average(binary))
        effects["decoder"].append(average(constrained) - average(free))
        effects["verifier_x_validity"].append(
            average([("state", "lex", decoder) for decoder in ("free", "constrained")])
            - average([("state", "binary", decoder) for decoder in ("free", "constrained")])
            - average([("answer", "lex", decoder) for decoder in ("free", "constrained")])
            + average([("answer", "binary", decoder) for decoder in ("free", "constrained")])
        )
        effects["verifier_x_decoder"].append(
            average([("state", validity, "constrained") for validity in ("binary", "lex")])
            - average([("state", validity, "free") for validity in ("binary", "lex")])
            - average([("answer", validity, "constrained") for validity in ("binary", "lex")])
            + average([("answer", validity, "free") for validity in ("binary", "lex")])
        )
    if not effects:
        raise ValueError("Study C3 factorial analysis requires paired scenes")
    return dict(effects)


def paired_factorial_effects(
    rows: Sequence[Mapping[str, object]], *, outcome: str, resamples: int, seed: int
) -> dict[str, object]:
    effects = _pair_effects(rows, outcome=outcome)
    if type(resamples) is not int or resamples <= 0 or type(seed) is not int:
        raise ValueError("Study C3 bootstrap parameters are invalid")
    rng = random.Random(seed)
    result: dict[str, object] = {}
    for name, pair_values in effects.items():
        estimate = sum(pair_values) / len(pair_values)
        draws = [
            sum(rng.choice(pair_values) for _ in pair_values) / len(pair_values)
            for _ in range(resamples)
        ]
        signs = [
            sum((1 if rng.random() < 0.5 else -1) * value for value in pair_values)
            / len(pair_values)
            for _ in range(resamples)
        ]
        p_value = (1 + sum(abs(value) >= abs(estimate) for value in signs)) / (resamples + 1)
        result[name] = {
            "estimate": estimate,
            "bootstrap_95_ci": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
            "sign_flip_p_value": p_value,
        }
    pair_count = len(next(iter(effects.values())))
    return {
        "outcome": outcome,
        "pair_count": pair_count,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "effects": result,
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    if not p_values or any(not 0.0 <= float(value) <= 1.0 for value in p_values.values()):
        raise ValueError("Holm correction requires named probabilities")
    ordered = sorted((float(value), key) for key, value in p_values.items())
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (value, key) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[key] = running
    return {key: adjusted[key] for key in p_values}


__all__ = ["holm_adjust", "paired_factorial_effects"]
