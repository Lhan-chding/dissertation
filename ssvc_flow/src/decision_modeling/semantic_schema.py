"""Frozen per-completion semantic map, using the actual N and L parsers."""

from __future__ import annotations

from ..constraint_solver import satisfies
from ..verifiers import annotate

EVENTS = ("X", "S", "W", "I")
REWARD_CHANNELS = ("X", "A", "V", "C")


def relation_counts(world, cue):
    """Count only supplied reliable relations; never inspect the truth world.

    Validate the cue even for invalid completions, so a broken denominator is a
    data error rather than silently rewarding/penalizing a model completion.
    """
    if not isinstance(cue, dict):
        raise ValueError("N-track relation reward requires an explicit cue")
    satisfies([0, 0, 0, 0], cue)
    family = cue["family"]
    denominator = {"duplicate_encoding": 1, "trend": 2}.get(family)
    if family == "cross_series":
        denominator = len(cue["edges"])
    if world is None:
        return 0, denominator
    if family == "duplicate_encoding":
        results = [world[cue["known_index"]] == cue["known_value"]]
    elif family == "cross_series":
        results = [world[i] + world[j] == total for i, j, total in cue["edges"]]
    else:
        a, b, c, d = world
        results = [a - 2 * b + c == 0, b - 2 * c + d == 0]
    return sum(results), denominator


def semantic_features(raw, prompt):
    """Return Phi; invalid N text is never extracted or repaired.

    L remains on its historical C3 P1 parser/executor. The new N relationship
    reward has no registered L definition and is explicitly not applicable.
    """
    scene = prompt.get("scene", prompt)
    track = prompt.get("track", scene.get("track", "N"))
    if track == "L":
        from ..legacy_frozen import annotate_legacy

        annotation = annotate_legacy(raw, scene)
        truth, observed = scene["truth"], scene["observation"]
        numerator = denominator = relation = None
        parser = "C3.semantic_action_parser.parse_p1"
    elif track == "N":
        truth, observed = scene["truth_world"], scene["observed_world"]
        annotation = annotate(raw, truth, scene["operation"], scene["cue"])
        numerator, denominator = relation_counts(annotation["parsed_world"], scene["cue"])
        relation = numerator / denominator
        parser = "N.verifiers.strict_parse"
    else:
        raise ValueError(f"unregistered track: {track}")
    parsed = annotation["parsed_world"]
    event = annotation["category"]
    valid = int(event != "I")
    result = {
        "event": event,
        "parsed_world": parsed,
        "relation_score": relation,
        "relation_numerator": numerator,
        "relation_denominator": denominator,
        "relation_status": "MEASURED" if denominator is not None else "NOT_APPLICABLE_L",
        "single_edit": int(
            parsed is not None and sum(a != b for a, b in zip(parsed, observed, strict=True)) == 1
        ),
        "coordinate_correct": [int(parsed is not None and parsed[j] == truth[j]) for j in range(4)],
        "copy_observation": int(parsed is not None and list(parsed) == list(observed)),
        "answer_correct": int(event in {"X", "S"}),
        "valid": valid,
        "parser_id": parser,
    }
    validate_features(result)
    return result


def validate_features(features):
    event = features["event"]
    if event not in EVENTS:
        raise ValueError("unknown event")
    if features["answer_correct"] != int(event in {"X", "S"}) or features["valid"] != int(
        event != "I"
    ):
        raise ValueError("event/derived-variable nesting violation")
    if features["single_edit"] not in (0, 1) or features["single_edit"] > features["valid"]:
        raise ValueError("single edit requires a valid action")
    numerator, denominator = features["relation_numerator"], features["relation_denominator"]
    if denominator is not None:
        if (
            type(denominator) is not int
            or denominator <= 0
            or type(numerator) is not int
            or not 0 <= numerator <= denominator
        ):
            raise ValueError("invalid relation counts")
        if features["relation_score"] != numerator / denominator or (
            event == "I" and numerator != 0
        ):
            raise ValueError("relation score inconsistent with strict validity/counts")
    return features


def reward_vector(features):
    validate_features(features)
    if features["relation_score"] is None:
        raise ValueError("four-channel N reward is not registered for L")
    return [
        int(features["event"] == "X"),
        features["answer_correct"],
        features["valid"],
        features["relation_score"],
    ]


phi = semantic_features
