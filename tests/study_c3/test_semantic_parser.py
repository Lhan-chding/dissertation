from __future__ import annotations

import pytest

from compensability.study_c3.semantic_action_parser import (
    parse_complete_action_any_domain,
    parse_p0,
    parse_p1,
    parse_p2,
)


@pytest.mark.parametrize(
    ("completion", "p0", "p1"),
    [
        ("2,3,4,5", (2, 3, 4, 5), (2, 3, 4, 5)),
        ("2, 3, 4, 5", None, (2, 3, 4, 5)),
        ("[2,3,4,5]", None, (2, 3, 4, 5)),
        ("(2,3,4,5)", None, (2, 3, 4, 5)),
        ("v1=2,v2=3,v3=4,v4=5", None, (2, 3, 4, 5)),
        ("v1:2 v2:3 v3:4 v4:5", None, (2, 3, 4, 5)),
        ("v1=2,v2=3,v3=4", None, None),
        ("9,24,12,14", None, None),
        ("Let's solve...", None, None),
    ],
)
def test_registered_parser_fixtures(
    completion: str,
    p0: tuple[int, int, int, int] | None,
    p1: tuple[int, int, int, int] | None,
) -> None:
    assert parse_p0(completion) == p0
    assert parse_p1(completion) == p1


def test_complete_action_parser_keeps_domain_failure_separate() -> None:
    assert parse_complete_action_any_domain("9,24,12,14") == (9, 24, 12, 14)
    assert parse_complete_action_any_domain("v1=2,v2=3,v3=4") is None


def test_p1_is_anchored_and_does_not_mine_prose_or_label_indices() -> None:
    assert parse_p1("The answer is 2,3,4,5 because...") is None
    assert parse_p1("v1=2 v2=3 v3=4") is None
    assert parse_p1("v1=2,v2=3,v3=4,v4=5 extra") is None


def test_p2_uses_first_explicit_action_segment_without_cross_segment_joining() -> None:
    assert parse_p2("Action: [2,3,4,5]\nExplanation follows") == (2, 3, 4, 5)
    assert parse_p2("2,3\n4,5") is None
    assert parse_p2("Numbers in prose: 2 3 4 5") is None
    assert parse_p2("v1=2,v2=3,v3=4\nv4=5") is None

