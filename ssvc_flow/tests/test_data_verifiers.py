import itertools

import pytest

from src.verifiers import annotate, classify, executor, fiber_size, strict_parse
from src.constraint_solver import solve


@pytest.mark.parametrize('raw', ['[1,2,3,true]', '[1,2,3,4.0]', '[1,2,3,"4"]', '```json\n[1,2,3,4]\n```', '[1,2,3,4] extra', '[1,2,3,4,5]', '[1,2,3]', '[1,2,3,100]', '[-1,2,3,4]', '[1,2,3,4', '[1,2,3,NaN]', '[1,2,3,4][1,2,3,4]', 'null', '{}'])
def test_parser_rejects_non_action(raw):
    assert strict_parse(raw) is None


def test_full_json_parser_accepts_standard_whitespace():
    assert strict_parse(' \n[ 1, 2, 3, 4 ]\t') == [1, 2, 3, 4]
    assert strict_parse('[0,0,99,99]') == [0, 0, 99, 99]
    assert strict_parse('[1,2,3,4]', domain_size=4) is None


def test_categories_partition_and_nested_events():
    truth = [1, 2, 3, 4]
    examples = {'X': '[1,2,3,4]', 'S': '[2,1,3,4]', 'W': '[1,2,3,5]', 'I': 'Answer: [1,2,3,4]'}
    for category, raw in examples.items():
        assert classify(raw, truth, 'sum4') == category
        result = annotate(raw, truth, 'sum4', {'family': 'trend'})
        assert result['category'] == category
        assert result['syntax_valid'] == (category != 'I')
    assert annotate('[1,2,3,5]', truth, 'sum4', {'family': 'trend'})['constraint_satisfaction'] is False
    assert annotate('bad', truth, 'sum4')['executor_answer'] is None


@pytest.mark.parametrize('operation', ['sum4', 'difference_pairs', 'range4'])
def test_fiber_size_matches_exhaustive_small_domain(operation):
    counts = {}
    for world in itertools.product(range(5), repeat=4):
        answer = executor(world, operation)
        counts[answer] = counts.get(answer, 0) + 1
    for answer, expected in counts.items():
        assert fiber_size(answer, operation, domain_size=5) == expected
    assert fiber_size(1000, operation, domain_size=5) == 0


def test_executor_rejects_unknown_operation_and_invalid_world():
    with pytest.raises(ValueError):
        executor([1, 2, 3, 4], 'bad')
    with pytest.raises(ValueError):
        executor([1, 2, 3], 'sum4')
    with pytest.raises(ValueError):
        fiber_size(1, 'bad')


@pytest.mark.parametrize('truth,observed,cue', [
    ([3, 7, 11, 15], [3, 9, 11, 15], {'family': 'trend'}),
    ([2, 9, 3, 7], [2, 8, 3, 7], {'family': 'duplicate_encoding', 'known_index': 1, 'known_value': 9}),
    ([2, 9, 3, 7], [2, 9, 6, 7], {'family': 'cross_series', 'edges': [[0, 1, 11], [0, 2, 5], [0, 3, 9]]}),
    ([2, 9, 3, 7], [2, 9, 6, 7], {'family': 'cross_series', 'edges': [[0, 1, 11], [1, 2, 12], [2, 3, 10]]}),
])
def test_independent_solver_unique_solution(truth, observed, cue):
    assert solve(observed, cue) == [truth]


def test_no_cue_is_396_candidates_and_diagnostics_not_coerced():
    assert len(solve([1, 2, 3, 4], None)) == 396
    assert solve([1, 2, 3, 4], {'family': 'duplicate_encoding', 'known_index': 0, 'known_value': 101}) == []
    multiple = solve([1, 2, 3, 4], {'family': 'duplicate_encoding', 'known_index': 0, 'known_value': 1})
    assert len(multiple) == 297
    with pytest.raises(ValueError):
        solve([1, 2, 3, 4], {'family': 'unknown'})
