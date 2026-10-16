"""Exact amendment bytes, frozen-data invariants and immutable candidate gates."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sr_f3_ce import contract
from sr_f3_ce.masks import target_spans
from sr_f3_ce.prompts import (
    CONTRACT,
    CONTRACT_V1,
    CONTRACT_V2,
    generation_contract,
    prefill_for,
    prompt_for_revision,
    prompt_from_frozen,
    question_from_prompt,
)
from sr_f3_ce.protocol_revision import (
    REQUEST_SHA256,
    _safe_package_files,
    build_revision_candidate,
    revision_config,
    revision_from_config,
    serialize_teacher_target,
    teacher_target,
    validate_revision_config,
    validate_revision_manifests,
    verify_revision_package,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "artifacts/sr_f3_ce_20261016/plan"
REQUEST = Path("/Users/louis/Downloads/CLAUDE_SR_F3_CE_PROTOCOL_FIX_FOR_CODEX_zh.md")
EXPECTED_V2 = 'All displayed quantities are integer counts in [10,90]. Use the printed category order for ties. Empty sum or count is 0; other aggregates of an empty set are null. A one-element range is 0. Exact integers, finite decimals, and quoted fractions such as "98/3" are accepted. Do not use external tools. Include enough true chart facts to verify the selection and the answer; reporting all relevant facts is allowed. You may add a short "work" string after the evidence and before the answer; it is optional. Output exactly one JSON object whose top-level keys are "evidence", then optionally "work", then "answer". The value of "evidence" is a list; each list entry has exactly the keys "series", "category", and "value".'  # noqa: E501
WORLD = {
    "categories": ["January", "February", "March", "April"],
    "domain": [10, 90],
    "series": {"Alpha": [26, 81, 13, 65], "Beta": [76, 22, 29, 60]},
}
QUERY = {"family": "INTERVAL", "start": 1, "end": 3, "aggregate": "mean"}


def test_v2_matches_the_provided_literal_not_its_inaccurate_diff_description():
    assert CONTRACT_V2 == EXPECTED_V2
    assert '{"series"' not in CONTRACT_V2
    assert CONTRACT_V2.endswith('the keys "series", "category", and "value".')
    assert "true chart facts" in CONTRACT_V2 and "true chart facts" not in CONTRACT_V1
    assert 'You may add a short "work" string' in CONTRACT_V2
    assert "You may put a short work string" in CONTRACT_V1
    assert CONTRACT == CONTRACT_V1
    assert "6" not in CONTRACT_V2 and "8" not in CONTRACT_V2.replace('"98/3"', "")


@pytest.mark.parametrize("revision", ["v1", "v2", "c2"])
def test_other_protocols_are_identical_bytes_and_c2_only_changes_structured_prefill(revision):
    original = "What is the mean?\n" + CONTRACT_V1
    revised = prompt_for_revision(original, revision)
    assert question_from_prompt(revised) == "What is the mean?"
    assert prompt_from_frozen(revised, "structured") == revised
    for protocol in ("answer_only", "natural_work"):
        assert prompt_from_frozen(revised, protocol) == prompt_from_frozen(original, protocol)
        assert generation_contract(protocol, revision) == generation_contract(protocol, "v1")
    assert prefill_for("structured", revision) == (
        '{\n  "evidence": [' if revision == "c2" else "{"
    )
    assert revised == "What is the mean?\n" + (EXPECTED_V2 if revision == "v2" else CONTRACT_V1)


def test_unknown_contract_revision_or_multiline_question_fail_closed():
    with pytest.raises(ValueError, match="revision"):
        prompt_for_revision("Q?\n" + CONTRACT_V1, "v3")
    with pytest.raises(ValueError, match="known frozen"):
        question_from_prompt("Q?\n" + CONTRACT_V2 + " one extra word")
    with pytest.raises(ValueError, match="one nonempty line"):
        question_from_prompt("Q?\nInjected question\n" + CONTRACT_V2)


@pytest.mark.parametrize("revision", ["v2", "c2"])
@pytest.mark.parametrize("with_work", [False, True])
@pytest.mark.parametrize("answer_kind", ["integer", "fraction", "null"])
def test_exact_indent_two_teacher_and_unambiguous_field_spans(revision, with_work, answer_kind):
    query = dict(QUERY)
    world = copy.deepcopy(WORLD)
    if answer_kind == "fraction":
        world["series"]["Alpha"][1] = 82
    if answer_kind == "null":
        query.update(positive_only=True)
        world["series"]["Alpha"] = [10] * 4
    before = contract.support_target(world, query, with_work)
    target = teacher_target(world, query, with_work, revision)
    assert target["object"] == before["object"]
    assert target["text"] == json.dumps(before["object"], indent=2, ensure_ascii=False)
    assert target["text"].startswith('{\n  "evidence": [')
    assert target_spans(target["text"]) == (target["object"], target["spans"])
    answer_start, answer_end = target["spans"]["answer"]
    assert target["text"][answer_start:answer_end].startswith('"answer": ')
    assert answer_start >= len(prefill_for("structured", revision))
    if answer_kind == "fraction":
        assert target["object"]["answer"] == "49/3"
        assert '"answer": "49/3"' in target["text"]
    if answer_kind == "null":
        assert target["object"]["answer"] is None
        assert '"answer": null' in target["text"]
    if with_work:
        lo, hi = target["spans"]["work"]
        assert json.loads(target["text"][lo:hi]) == target["object"]["work"]
        assert lo >= len(prefill_for("structured", revision))
    assert contract.support_target(world, query, with_work) == before
    assert serialize_teacher_target(before, "v1") == before


def test_work_text_cannot_spoof_root_answer_span():
    original = contract.support_target(WORLD, QUERY, True)
    original["object"]["work"] += '\n  "answer": 999'
    original["text"] = contract.canonical(original["object"])
    _, original["spans"] = target_spans(original["text"])
    changed = serialize_teacher_target(original, "v2")
    lo, hi = changed["spans"]["answer"]
    assert changed["text"][lo:hi] == '"answer": 16\n}'
    malformed = copy.deepcopy(original)
    malformed["spans"]["answer"][0] -= 1
    with pytest.raises(ValueError, match="disagree"):
        serialize_teacher_target(malformed, "v2")


@pytest.fixture(scope="module")
def original_config():
    if not PLAN.is_dir():
        pytest.skip("Private original user package is not installed")
    return json.loads((PLAN / "config/SR_F3_CE.json").read_text())


@pytest.mark.parametrize("revision", ["v2", "c2"])
def test_only_registered_config_mutations(original_config, revision):
    before = copy.deepcopy(original_config)
    changed = revision_config(original_config, revision)
    assert revision_from_config(changed) == revision
    validate_revision_config(original_config, changed, revision)
    assert original_config == before
    assert changed["teacher_serialization"] == "json.dumps(indent=2)"
    changed["training"]["lr"] *= 2
    with pytest.raises(ValueError, match="Unregistered configuration"):
        validate_revision_config(original_config, changed, revision)
    c2 = revision_config(original_config, "c2")
    c2["generation"]["prefill_by_protocol"]["answer_only"] = c2["generation"]["prefill"]
    with pytest.raises(ValueError, match="preserve separate"):
        revision_from_config(c2)


def test_original_inventory_hash_and_parse_use_the_same_bytes(tmp_path, monkeypatch):
    import shutil

    if not PLAN.is_dir():
        pytest.skip("Private original user package is not installed")
    copied = tmp_path / "plan"
    shutil.copytree(PLAN, copied)
    manifest = copied / "PACKAGE_SHA256.json"
    original_manifest = manifest.read_bytes()
    changed_file = copied / "manifests/BUILD_SUMMARY.json"
    changed_file.write_bytes(changed_file.read_bytes() + b"\n")
    changed_inventory = json.loads(original_manifest)
    changed_inventory["manifests/BUILD_SUMMARY.json"] = hashlib.sha256(
        changed_file.read_bytes()
    ).hexdigest()
    swapped_manifest = json.dumps(changed_inventory).encode("utf-8")
    read_bytes = Path.read_bytes
    manifest_reads = 0

    def swap_after_hash_read(path):
        nonlocal manifest_reads
        if path == manifest:
            manifest_reads += 1
            if manifest_reads == 1:
                # A separate later JSON read would consume this changed inventory.
                captured = read_bytes(path)
                path.write_bytes(swapped_manifest)
                return captured
            if manifest_reads == 2:
                # Restore the authentic inventory before the full file hash pass.
                path.write_bytes(original_manifest)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", swap_after_hash_read)
    with pytest.raises(ValueError, match="content differs from frozen authority"):
        _safe_package_files(copied)
    assert manifest_reads == 2
    assert read_bytes(manifest) == original_manifest


@pytest.fixture(scope="module")
def candidates(tmp_path_factory):
    if not PLAN.is_dir() or not REQUEST.is_file():
        pytest.skip("Private original package and exact user amendment are not installed")
    parent = tmp_path_factory.mktemp("protocol-revision-candidates")
    original_hashes = {
        str(path.relative_to(PLAN)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in PLAN.rglob("*")
        if path.is_file()
    }
    results = {}
    for revision in ("v2", "c2"):
        output = parent / revision / "plan"
        result = build_revision_candidate(PLAN, REQUEST, output, revision)
        assert result["status"] == "PASS"
        results[revision] = output
    after = {
        str(path.relative_to(PLAN)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in PLAN.rglob("*")
        if path.is_file()
    }
    assert original_hashes == after
    return results


@pytest.mark.parametrize("revision", ["v2", "c2"])
def test_actual_reference_generator_all_rows_and_offline_authority(candidates, revision):
    candidate = candidates[revision]
    verified = verify_revision_package(candidate)
    assert verified["revision"] == revision
    assert verified["request_sha256"] == REQUEST_SHA256
    difference = verified["manifest_difference"]
    assert difference["tasks"] == 4096
    assert difference["teacher_targets"] == 768
    assert difference["support_teacher_rows_changed"] == 256
    assert difference["nonstructured_prompts_byte_identical"] == 8192
    assert difference["prompt_rows_changed"] == (4096 if revision == "v2" else 0)
    assert difference["unregistered_differences"] == 0
    assert verified["selected_for_science"] is False
    receipt = json.loads((candidate / "PROTOCOL_REVISION.json").read_text())
    assert receipt["reference_validation"]["build"]["returncode"] == 0
    assert receipt["reference_validation"]["rebuild"]["returncode"] == 0
    assert receipt["reference_validation"]["tests"]["returncode"] == 0
    assert len(receipt["reference_validation"]["byte_identical_rebuild"]) == 8
    with pytest.raises(ValueError, match="fresh path"):
        build_revision_candidate(PLAN, REQUEST, candidate, revision)


@pytest.mark.parametrize("field", ["answer", "question", "root_id", "world", "teacher_object"])
def test_unrelated_manifest_changes_fail_even_if_the_candidate_inventory_is_rehashed(
    candidates, tmp_path, field
):
    import shutil

    candidate = candidates["v2"]
    copied = tmp_path / "manifests"
    shutil.copytree(candidate / "manifests", copied)
    file = copied / "TASKS_AUDIT_ONLY.jsonl"
    rows = [json.loads(line) for line in file.read_text().splitlines()]
    if field == "teacher_object":
        row = next(row for row in rows if row["teacher"] is not None)
        row["teacher"]["GOLD"]["object"]["answer"] = 999
    elif field == "world":
        rows[0]["world"]["series"]["Alpha"][0] += 1
    else:
        rows[0][field] = 999 if field == "answer" else "changed"
    file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=r"Unregistered manifest|Incorrect teacher"):
        validate_revision_manifests(PLAN / "manifests", copied, "v2")


def test_reference_scorer_edit_rejected_after_rehash(candidates, tmp_path):
    import shutil

    copied = tmp_path / "plan"
    shutil.copytree(candidates["v2"], copied)
    file = copied / "reference/contract.py"
    file.write_text(file.read_text().replace("out['J']=int(", "out['J']=int(1 or "))
    inventory = json.loads((copied / "PACKAGE_SHA256.json").read_text())
    inventory["reference/contract.py"] = hashlib.sha256(file.read_bytes()).hexdigest()
    (copied / "PACKAGE_SHA256.json").write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="Unregistered reference/document"):
        verify_revision_package(copied)


def test_input_schema_and_schedule_changes_are_not_allowed(candidates, tmp_path):
    import shutil

    copied = tmp_path / "manifests"
    shutil.copytree(candidates["c2"] / "manifests", copied)
    path = copied / "SCHEDULE_b0.json"
    schedule = json.loads(path.read_text())
    schedule[0]["support_qids"].reverse()
    path.write_text(json.dumps(schedule))
    with pytest.raises(ValueError, match="Unregistered manifest change"):
        validate_revision_manifests(PLAN / "manifests", copied, "c2")
