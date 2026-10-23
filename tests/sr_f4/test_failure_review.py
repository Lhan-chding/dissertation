import copy

import pytest

from sr_f4.failure_review import VERSION, overlay, resolve, run, validate_bindings


def inputs():
    key = dict(review_id="r", row_id="a", target_categories=["Jan"], selection_series="Alpha")
    blind = dict(review_id="r", world=dict(categories=["Jan", "Feb"]), work="Selected: Jan")
    review = dict(
        review_id="r", reviewer="one", annotation_source="AI_REVIEW", P="PASS",
        extraction_status="KNOWN", adopted_categories=["Jan"], selection_series="Alpha",
        evidence_quote=["Selected: Jan"], rationale="Explicit adoption",
    )
    return key, blind, review


def test_target_comparison_happens_after_extraction_without_mutating_reviews():
    key, blind, review = inputs()
    saved = copy.deepcopy(review)
    yes, summary = resolve([key], [blind], [review])
    no, _ = resolve([dict(key, target_categories=["Feb"])], [blind], [review])
    wrong_series, _ = resolve([dict(key, selection_series="Beta")], [blind], [review])
    assert yes[0]["S"] == "PASS"
    assert no[0]["S"] == wrong_series[0]["S"] == "FAIL"
    assert review == saved
    assert summary["human_review_count"] == 0
    assert not summary["production_gate_modified"]


@pytest.mark.parametrize("change", [
    dict(annotation_source="HUMAN"), dict(evidence_quote=["not observed"]),
    dict(adopted_categories=["Mar"]), dict(adopted_categories=["Jan", "Jan"]),
    dict(extraction_status="UNKNOWN"), dict(rationale=""), dict(P="MAYBE"),
])
def test_invalid_or_invented_review_evidence_rejected(change):
    key, blind, review = inputs()
    with pytest.raises(ValueError):
        resolve([key], [blind], [dict(review, **change)])


def test_incomplete_and_duplicate_review_rejected():
    key, blind, review = inputs()
    for rows in ([], [review, review]):
        with pytest.raises(ValueError):
            resolve([key], [blind], rows)


def test_disagreement_requires_preserved_explicit_adjudication():
    key, blind, review = inputs()
    second = dict(review, reviewer="two", extraction_status="UNKNOWN", adopted_categories=None)
    with pytest.raises(ValueError, match="Unresolved"):
        resolve([key], [blind], [review, second])
    decision = dict(second, annotation_source="AI_ADJUDICATION", rationale="Conflict unresolved")
    rows, summary = resolve([key], [blind], [review, second], [decision])
    assert rows[0]["S"] == "UNKNOWN"
    assert rows[0]["had_disagreement"] and rows[0]["adjudicated"]
    assert summary["second_review_rows"] == 1
    assert summary["disagreements"] == ["r"]


def group():
    return dict(
        source="C_TRAIN_PROBE", group_id="g", block="b0", family="TOPK", world_id="w",
        step=0, cell="T-S", A_J=[0.0], verifier_version="old", scalar="STALE_DERIVED_REWARD",
        labels=[dict(
            row_id="a", verifier_version="old", normalization=dict(
                j=0, protocol_valid=True, finished=True, defined=True, sufficient=False,
            ), native_score=dict(J=0), dimensions={d: dict(status="UNKNOWN") for d in "PSOE"},
        )],
    )


def test_overlay_keeps_j_and_eligibility_and_discards_stale_derived_rewards():
    key, blind, review = inputs()
    resolved, _ = resolve([key], [blind], [review])
    original = group()
    saved = copy.deepcopy(original)
    result = overlay([original], resolved)[0]
    label = result["labels"][0]
    assert original == saved
    assert "scalar" not in result
    assert label["normalization"] == saved["labels"][0]["normalization"]
    assert label["native_score"] == dict(J=0)
    assert label["dimensions"]["E"]["status"] == "UNKNOWN"
    assert label["dimensions"]["S"]["status"] == "PASS"
    assert label["verifier_version"] == VERSION
    assert label["measurement_mode"] == "AI_REVIEWED_FAILURE"
    assert not result["frozen_gate_eligible"]


def test_overlay_must_exactly_cover_candidate_failures_and_only_c():
    key, blind, review = inputs()
    resolved, _ = resolve([key], [blind], [review])
    for groups, rows in (([group()], []), ([dict(group(), source="B2_DEV")], resolved)):
        with pytest.raises(ValueError):
            overlay(groups, rows)
    g = group()
    g["labels"][0]["normalization"]["j"] = 1
    with pytest.raises(ValueError):
        overlay([g], resolved)


def test_overlay_cannot_launder_mismatched_or_mixed_input_versions():
    key, blind, review = inputs()
    resolved, _ = resolve([key], [blind], [review])
    with pytest.raises(ValueError, match="versions disagree"):
        overlay([dict(group(), verifier_version="different")], resolved)
    second = group()
    second.update(group_id="g2", verifier_version="other")
    second["labels"][0].update(row_id="b", verifier_version="other")
    with pytest.raises(ValueError, match="different input"):
        overlay([group(), second], [*resolved, dict(resolved[0], row_id="b")])


def test_run_rejects_partial_cache_before_writing(tmp_path):
    import json

    key, blind, review = inputs()
    for name, row in (("KEY_PRIVATE.jsonl", key), ("BLIND_BATCH_1.jsonl", blind),
                      ("REVIEWER_1.jsonl", review), ("replay.jsonl", group())):
        (tmp_path / name).write_text(json.dumps(row) + "\n")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="complete frozen C cache"):
        run(tmp_path, tmp_path / "replay.jsonl", output)
    assert not output.exists()


def test_review_bindings_reject_swapped_keys_and_changed_world_or_native_j():
    import json

    keys, blinded, labels, raw, tasks = [], [], [], [], []
    for i, category in enumerate(("Jan", "Feb")):
        qid, row_id = f"q{i}", f"a{i}"
        world = dict(categories=["Jan", "Feb"])
        obj = dict(evidence=[], work=f"Selected: {category}")
        task = dict(qid=qid, world_id=f"w{i}", pool="C_TRAIN", world=world,
                    query=dict(selection_series="Alpha"), target_sets=[category])
        tasks.append(task)
        raw.append(dict(row_id=row_id, qid=qid, score=dict(J=0), full_text=json.dumps(obj)))
        labels.append(dict(row_id=row_id, qid=qid, world_id=task["world_id"],
                           native_score=dict(J=0), normalization=dict(j=0)))
        keys.append(dict(review_id=f"r{i}", row_id=row_id, qid=qid, world_id=task["world_id"],
                         query=task["query"], target_categories=[category],
                         selection_series="Alpha"))
        blinded.append(dict(review_id=f"r{i}", world=world, **obj))
    groups = [dict(source="C_TRAIN_PROBE", labels=labels)]
    compact = [dict(rows=raw)]
    validate_bindings(keys, blinded, groups, compact, tasks)
    swapped = [dict(keys[0], row_id="a1"), dict(keys[1], row_id="a0")]
    with pytest.raises(ValueError, match="does not match"):
        validate_bindings(swapped, blinded, groups, compact, tasks)
    with pytest.raises(ValueError, match="does not match"):
        validate_bindings(keys, [dict(blinded[0], world={}), blinded[1]], groups, compact, tasks)
    labels[0]["normalization"]["j"] = 1
    with pytest.raises(ValueError, match="native score"):
        validate_bindings(keys, blinded, groups, compact, tasks)
