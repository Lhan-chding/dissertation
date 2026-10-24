import copy

import pytest

from sr_f4 import c_local_replay


def corpus():
    task = dict(qid="q", world_id="w", root_id="w", split="C_FP_TRAIN_R1",
                world=dict(categories=["January", "February"],
                           series=dict(Alpha=[40, 20], Beta=[25, 30]), domain=[10, 90]),
                query=dict(family="TOPK", selection_series="Alpha", value="Beta",
                           direction="largest", k=1, aggregate="sum"))
    for b in range(3):
        for step in range(1, 17):
            for gi in range(16):
                gid = f"CPF-T-b{b}-SELECT/s{step:03d}/g{gi:05d}"
                records, observations = [], []
                for draw in range(8):
                    j = int(draw < 4)
                    rid = f"{gid}/d{draw:02d}"
                    records.append(dict(row_id=rid, group_id=gid, qid="q", block=b, step=step,
                                        arm="SELECT", draw=draw, full_text="not new protocol",
                                        score=dict(J=j, presentation=1, finish=1)))
                    observations.append(dict(row_id=rid, j=j, perception="PASS",
                                             selection="PASS" if j else "FAIL",
                                             protocol_valid=True, finished=True, defined=True))
                yield dict(source="COMPLETED_FPRIME_SELECT_TRAIN_ONLY", block=b, step=step,
                           group_index=gi, task=task, records=records, observations=observations,
                           old_diagnostic=dict(active=True, selection_signal=False))


def test_expansion_does_not_make_outcome_confounded_selection_shufflable(monkeypatch):
    monkeypatch.setattr(c_local_replay, "rows", lambda path: corpus())
    result = c_local_replay.replay("fixture")
    assert result["counts"]["expanded_selection_signal_groups"] == 768
    assert result["counts"]["joint_J_P_swappable_groups"] == 0
    assert result["counts"]["joint_J_P_changed_rows"] == 0
    assert result["strict_new_measurement"]["selection"] == {"UNKNOWN": 6144}


@pytest.mark.parametrize("fault", ["incomplete", "duplicate", "endpoint", "row_identity"])
def test_replay_rejects_partial_or_wrong_corpus(monkeypatch, fault):
    first = copy.deepcopy(next(corpus()))
    values = [first]
    if fault == "duplicate":
        values.append(first)
    elif fault == "endpoint":
        first["task"]["split"] = "C_FP_TEST_R1"
    elif fault == "row_identity":
        first["records"][0]["row_id"] = "wrong"
    monkeypatch.setattr(c_local_replay, "rows", lambda path: values)
    with pytest.raises(ValueError):
        c_local_replay.replay("fixture")
