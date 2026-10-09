import pytest

from sr_f1.chartqa import iter_chartqa_slots, relaxed_accuracy, score_chartqa


def test_author_numeric_zero_percent_and_no_answer_extraction():
    assert relaxed_accuracy("100", "105") == 1
    assert relaxed_accuracy("100", "105.001") == 0
    assert relaxed_accuracy("-100", "-95") == 1
    assert relaxed_accuracy("0", "0.0") == 1
    assert relaxed_accuracy("0", "0.001") == 0
    assert relaxed_accuracy("5%", ".05") == 0
    assert relaxed_accuracy("Alpha", "alpha") == 1
    assert relaxed_accuracy("1", "The answer is 1") == 0
    assert score_chartqa("1", "1.0") == {"relaxed_accuracy": 1, "exact_match": 0}


def test_chartqa_cannot_silently_use_convenience_subset():
    with pytest.raises(ValueError, match="partial"):
        list(iter_chartqa_slots([], "SRF1_COMMON_START"))
