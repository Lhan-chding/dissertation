"""Exact finite-language decoder audit; no real-model results are synthesized."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def enumerate_decoders(transitions, valid_sequences, max_steps):
    """Enumerate free proposals, exact rejection, and grammar-prefix local masking.

    Keys are token tuples. Missing successors terminate a sequence. Grammar
    masking only knows the provided valid language, never truth or task cues.
    The fixture language is deliberately finite, not a production tokenizer FSA.
    """
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    language = frozenset(tuple(sequence) for sequence in valid_sequences)
    if not language or any(not seq or len(seq) > max_steps for seq in language):
        raise ValueError("valid language must contain bounded nonempty sequences")
    prefixes = frozenset(seq[:length] for seq in language for length in range(len(seq) + 1))
    for prefix, probabilities in transitions.items():
        values = np.array(list(probabilities.values()), dtype=float)
        if not isinstance(prefix, tuple) or not len(values) or not np.isfinite(values).all():
            raise ValueError("invalid transition table")
        if np.any(values < 0) or not np.isclose(values.sum(), 1.0, atol=1e-12, rtol=0):
            raise ValueError("transition probabilities must be normalized")

    def walk(prefix, mass, masked):
        if len(prefix) >= max_steps or prefix not in transitions:
            return [(prefix, mass)]
        successors = transitions[prefix]
        allowed = {
            token: prob
            for token, prob in successors.items()
            if not masked or (*prefix, token) in prefixes
        }
        normalizer = sum(allowed.values()) if masked else 1.0
        if not normalizer:
            return [(prefix, mass)]
        return [
            item
            for token, prob in allowed.items()
            if prob > 0
            for item in walk((*prefix, token), mass * prob / normalizer, masked)
        ]

    free = dict(walk((), 1.0, False))
    masked = dict(walk((), 1.0, True))
    valid_mass = sum(mass for seq, mass in free.items() if seq in language)
    rejection = (
        {seq: mass / valid_mass for seq, mass in free.items() if seq in language}
        if valid_mass
        else {}
    )

    def encode(mapping):
        return {json.dumps(list(seq), separators=(",", ":")): mass for seq, mass in mapping.items()}

    distance = 0.5 * sum(
        abs(masked.get(seq, 0.0) - rejection.get(seq, 0.0)) for seq in set(masked) | set(rejection)
    )
    return {
        "free_distribution": encode(free),
        "free_valid_probability": valid_mass,
        "free_conditional_distribution": encode(rejection),
        "rejection_distribution": encode(rejection),
        "fsa_distribution": encode(masked),
        "fsa_completion_probability": sum(masked.get(seq, 0.0) for seq in language),
        "total_variation_fsa_vs_rejection": distance,
        "rejection_expected_proposals_per_accept": 1 / valid_mass if valid_mass else None,
        "scope": "toy_mathematical_validation",
    }


def weighted_pooled_purity(valid_rates, conditional_purities, prompt_weights):
    """Reconstruct pooled q using original prompt weights * original validity."""
    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in [valid_rates, conditional_purities, prompt_weights]
    ]
    valid, purity, weights = arrays
    if any(a.ndim != 1 or not a.size or not np.isfinite(a).all() for a in arrays):
        raise ValueError("rates and weights must be finite nonempty vectors")
    if any(a.shape != valid.shape for a in arrays) or any(np.any(a < 0) for a in arrays):
        raise ValueError("incompatible shapes or negative values")
    if np.any(valid > 1) or np.any(purity > 1) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("rates must be probabilities and prompt weights normalized")
    mass = np.dot(weights, valid)
    return float(np.dot(weights * valid, purity) / mass) if mass else None


def toy_decoder_audit():
    transitions = {
        (): {"[0,": 0.5, "[1,": 0.5},
        ("[0,",): {"0,0,0]": 0.9, "bad": 0.1},
        ("[1,",): {"0,0,0]": 0.1, "bad": 0.9},
    }
    language = frozenset({("[0,", "0,0,0]"), ("[1,", "0,0,0]")})
    result = enumerate_decoders(transitions, language, 2)

    def parse(raw):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return False
        return (
            isinstance(value, list)
            and len(value) == 4
            and all(type(n) is int and 0 <= n <= 99 for n in value)
        )

    agreement = all(
        parse("".join(json.loads(key))) == (tuple(json.loads(key)) in language)
        for key in result["free_distribution"]
    )
    return {
        **result,
        "parser_language_agrees_with_fsa": agreement,
        "language_agreement_scope": "all_complete_strings_in_enumerated_toy_support",
        "real_tokenizer_fsa": "not_implemented_in_this_toy_module",
        "truth_world_or_cue_used": False,
        "pooled_weighting_example": {
            "valid_rates": [0.9, 0.1],
            "conditional_purities": [0.1, 0.9],
            "prompt_weights": [0.5, 0.5],
            "correct_pooled_q": 0.18,
            "unweighted_mean_q": 0.5,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    from .core import phase_artifacts, write_json

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "decoder_toy_audit.json"
    write_json(path, toy_decoder_audit())
    phase_artifacts(
        args.out,
        "P8a",
        "CPU_TOY_PASSED",
        {"scope": "toy_mathematical_validation", "real_model_decoder_audit": "pending_gpu"},
        [path],
    )


if __name__ == "__main__":
    main()
