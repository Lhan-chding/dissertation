"""Executable contracts for the 2x2 perception--reasoning game."""

from dataclasses import FrozenInstanceError

import pytest

from compbias.envs.symbolic_game import (
    CoordinationSolution,
    PerceptionMode,
    ReasoningMode,
    SymbolicGame,
    SymbolicOutcome,
)
from compbias.theory.coordination import CoordinationParams, reward


def test_all_four_action_pairs_execute_the_registered_reward_matrix() -> None:
    game = SymbolicGame(delta=0.2, epsilon=0.7)

    fixtures = (
        (
            PerceptionMode.TRUTHFUL,
            ReasoningMode.CANONICAL,
            1.0,
            CoordinationSolution.TRUTHFUL,
        ),
        (
            PerceptionMode.TRUTHFUL,
            ReasoningMode.COMPENSATOR,
            0.8,
            CoordinationSolution.MISMATCHED,
        ),
        (
            PerceptionMode.ERRONEOUS,
            ReasoningMode.CANONICAL,
            0.3,
            CoordinationSolution.MISMATCHED,
        ),
        (
            PerceptionMode.ERRONEOUS,
            ReasoningMode.COMPENSATOR,
            1.0,
            CoordinationSolution.COMPENSATORY,
        ),
    )
    for perception, reasoning, expected_reward, expected_solution in fixtures:
        outcome = game.execute(perception, reasoning)

        assert game.reward(perception, reasoning) == pytest.approx(expected_reward)
        assert outcome.perception_mode is perception
        assert outcome.reasoning_mode is reasoning
        assert outcome.reward == pytest.approx(expected_reward)
        assert outcome.solution is expected_solution
        assert outcome.is_coordinated is (expected_solution is not CoordinationSolution.MISMATCHED)

    for actual_row, expected_row in zip(
        game.reward_matrix,
        ((1.0, 0.8), (0.3, 1.0)),
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)


def test_expected_reward_delegates_to_the_authoritative_coordination_equation() -> None:
    game = SymbolicGame(delta=0.2, epsilon=0.7)
    p_truthful = 0.3
    q_canonical = 0.8
    params = CoordinationParams(delta=game.delta, epsilon=game.epsilon)

    actual = game.expected_reward(p_truthful=p_truthful, q_canonical=q_canonical)

    assert actual == pytest.approx(reward(p_truthful, q_canonical, params))
    assert actual == pytest.approx(0.596)


def test_point_mass_expected_rewards_equal_executed_outcomes() -> None:
    game = SymbolicGame(delta=0.35, epsilon=0.6)

    for perception in PerceptionMode:
        for reasoning in ReasoningMode:
            p_truthful = float(perception is PerceptionMode.TRUTHFUL)
            q_canonical = float(reasoning is ReasoningMode.CANONICAL)

            assert game.execute(perception, reasoning).reward == pytest.approx(
                game.expected_reward(
                    p_truthful=p_truthful,
                    q_canonical=q_canonical,
                )
            )


def test_game_preserves_the_theory_domain_for_penalties_above_one() -> None:
    game = SymbolicGame(delta=2.0, epsilon=0.5)

    assert game.execute("T", "K").reward == pytest.approx(-1.0)
    assert game.reward_matrix[1][0] == pytest.approx(0.5)


def test_public_outcome_record_rejects_a_solution_inconsistent_with_its_actions() -> None:
    with pytest.raises(ValueError, match="action pair"):
        SymbolicOutcome(
            perception_mode=PerceptionMode.TRUTHFUL,
            reasoning_mode=ReasoningMode.CANONICAL,
            reward=1.0,
            solution=CoordinationSolution.COMPENSATORY,
        )


def test_game_and_outcomes_are_immutable() -> None:
    game = SymbolicGame(delta=0.2, epsilon=0.7)
    outcome = game.execute("T", "C")

    with pytest.raises(FrozenInstanceError):
        game.delta = 0.5  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        outcome.reward = 0.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        game.reward_matrix[0][0] = 0.0  # type: ignore[index]


@pytest.mark.parametrize(
    ("delta", "epsilon"),
    [
        (0.0, 0.5),
        (0.5, 0.0),
        (-0.1, 0.5),
        (True, 0.5),
        (0.5, float("nan")),
        (float("inf"), 0.5),
        pytest.param(10**10_000, 0.5, id="overflowing-real"),
    ],
)
def test_game_rejects_penalties_outside_the_outcome_reward_contract(
    delta: object, epsilon: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        SymbolicGame(delta=delta, epsilon=epsilon)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("p_truthful", "q_canonical"),
    [
        (-0.1, 0.5),
        (1.1, 0.5),
        (0.5, float("nan")),
        (True, 0.5),
        (0.5, "0.2"),
    ],
)
def test_expected_reward_rejects_malformed_probabilities(
    p_truthful: object, q_canonical: object
) -> None:
    game = SymbolicGame(delta=0.2, epsilon=0.7)

    with pytest.raises((TypeError, ValueError)):
        game.expected_reward(  # type: ignore[arg-type]
            p_truthful=p_truthful,
            q_canonical=q_canonical,
        )


@pytest.mark.parametrize(
    ("perception", "reasoning"),
    [("unknown", "C"), ("T", "unknown"), (1, "C"), ("T", None)],
)
def test_execute_rejects_unknown_action_modes(perception: object, reasoning: object) -> None:
    game = SymbolicGame(delta=0.2, epsilon=0.7)

    with pytest.raises(ValueError, match="mode"):
        game.execute(perception, reasoning)  # type: ignore[arg-type]
