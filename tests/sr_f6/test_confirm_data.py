import copy
import hashlib
import json
from collections import Counter

import pytest

from sr_f6 import confirm_data as data
from sr_f6.common import digest, file_hash
from sr_f6.interface_v3_protocol import public_prompt
from srf6_ref import design


@pytest.fixture(scope="module")
def bank():
    return design.build_bank(data.SPLIT, 8, 20261028)


def write(path, value, *, rows=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in value) if rows else json.dumps(value))


def history_fixture(project, *, collision=None):
    roots = [project / name for name in ("original", "f5", "f4pilot", "f4main")]
    world = collision or dict(
        categories=["historical_A", "historical_B"],
        series=dict(Alpha=[12, 18], Beta=[84, 78]),
        domain=[10, 90],
    )
    task = dict(world=world, query={"secret_test_query": "MUST_NOT_RETURN"}, gold_answer="SECRET")
    original = roots[0]
    sources = []
    original_task = original / "DATA/TEST_ID/TASKS_PRIVATE.jsonl"
    write(original_task, [task], rows=True)
    write(
        original / "DATA/MANIFEST.json",
        dict(
            files=[
                dict(
                    path="DATA/TEST_ID/TASKS_PRIVATE.jsonl",
                    sha256=file_hash(original_task),
                    bytes=original_task.stat().st_size,
                )
            ]
        ),
    )
    write(
        original / "IDENTITY.json",
        dict(
            data_manifest=dict(
                path="DATA/MANIFEST.json", sha256=file_hash(original / "DATA/MANIFEST.json")
            )
        ),
    )
    for i, root in enumerate(roots[1:]):
        name = "TASKS_PRIVATE.jsonl" if i == 0 else "TASKS.jsonl"
        path = root / "DATA" / name
        write(path, [task, dict(pool="CHARTQA", answer="PRIVATE_CHARTQA")], rows=True)
        sources.append(dict(path=str(path), sha256=file_hash(path), world_rows=1))
    write(original / "PREFLIGHT/HISTORICAL_WORLD_AUDIT.json", dict(status="PASS", sources=sources))
    return roots


def test_full_new_bank_recomputes_original_math_native_score_and_finite_library(bank):
    report = data._audit_roots(bank)
    assert report["split"] == "DEV_SCHEMA_CONFIRM"
    assert report["master_seed"] == 20261028
    assert report["roots"] == 48
    assert report["questions"] == 384
    assert report["forced_anchors"] == 18
    assert report["dual_oracle_comparisons"] == report["native_full_fact_regressions"] == 384
    assert report["family_questions"] == dict.fromkeys(design.FAMILIES, 64)
    assert report["finite_candidate_audit"]["strata"]["all"]["strict_gain_roots"] == 18
    assert report["finite_candidate_audit"]["model_error_coverage"] == "NOT_RUN"
    assert report["processor_readability"] == "NOT_RUN"


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("factor", "Both factors"),
        ("gold", "Stored gold"),
        ("descriptor", "edit-cost"),
        ("coverage", "coverage differs"),
        ("active", "inventory"),
        ("query", "public query"),
        ("forced", "forced-anchor"),
    ],
)
def test_audit_rejects_changed_registered_semantics(bank, mutation, match):
    roots = copy.deepcopy(bank)
    first = roots[0]
    if mutation == "factor":
        first["tasks"].pop("P1")
    elif mutation == "gold":
        first["tasks"]["P1"][1]["gold_answer"] = "99999"
    elif mutation == "descriptor":
        first["tasks"]["P1"][1]["edit_descriptor"] = ["different"]
    elif mutation == "coverage":
        first["certificate"]["coverage_P1"] = -1
    elif mutation == "active":
        first["certificate"]["active_rules"] = []
    elif mutation == "query":
        for factor in first["tasks"].values():
            for task in factor:
                task["query"] = {**task["query"], "aggregate": "mean"}
    else:
        first["certificate"]["engineered_collision_anchor"] = False
    with pytest.raises(ValueError, match=match):
        data._audit_roots(roots)


def test_historical_exclusion_projects_world_only_and_preserves_bound_sources(tmp_path, bank):
    roots = history_fixture(tmp_path)
    before = {p: p.read_bytes() for root in roots for p in root.rglob("*") if p.is_file()}
    report = data._historical(bank, roots)
    assert report["status"] == "PASS"
    assert report["checked_question_rows"] == 384
    assert len(report["sources"]) == 4
    assert report["unique_historical_worlds"] == 1
    assert len(report["bound_files"]) == 7
    assert "SECRET" not in json.dumps(report)
    assert "MUST_NOT_RETURN" not in json.dumps(report)
    assert before == {p: p.read_bytes() for p in before}
    checked = data.verify_bound_files(report["bound_files"], boundary=tmp_path)
    assert checked["content_rehashed"] == 0


def test_history_checks_renaming_and_reordering_not_labels(tmp_path, bank):
    old = copy.deepcopy(bank[0]["tasks"]["P0"][0]["world"])
    old["categories"] = list(reversed(old["categories"]))
    for series in old["series"]:
        old["series"][series].reverse()
    old["categories"] = [f"historical_{i}" for i in range(len(old["categories"]))]
    roots = history_fixture(tmp_path, collision=old)
    report = data._historical(bank, roots)
    assert report["status"] == "FAIL"
    assert bank[0]["tasks"]["P0"][0]["qid"] in report["collisions"]


def test_history_rejects_changed_bound_bytes_without_looking_at_labels(tmp_path, bank):
    roots = history_fixture(tmp_path)
    path = roots[2] / "DATA/TASKS.jsonl"
    path.write_text(path.read_text().replace("84", "72"))
    with pytest.raises(PermissionError, match="content changed"):
        data._historical(bank, roots)


def test_history_requires_every_explicit_source(tmp_path, bank):
    roots = history_fixture(tmp_path)
    with pytest.raises(ValueError, match="three historical"):
        data._historical(bank, roots[:-1])
    with pytest.raises(ValueError, match="bound inventory"):
        data._historical(bank, [roots[0], roots[2], roots[1], roots[3]])


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    project = tmp_path_factory.mktemp("confirm-data").resolve()
    historical = history_fixture(project)
    root = project / "new"
    root.mkdir()
    receipt = data.build(root, historical_roots=historical)
    return root, receipt, historical


def test_build_real_sources_and_exact_new_review_plan(built):
    root, receipt, _ = built
    public, private = data.load_inputs(root)
    assert receipt["status"] == "CPU_PREPARED_NOT_RELEASED"
    assert receipt["data_manifest_sha256"] == file_hash(root / "DATA/MANIFEST.json")
    assert len(public) == len(private) == 384
    assert len(Counter((r["root_id"], r["factor"]) for r in public)) == 96
    assert all(n == 4 for n in Counter((r["root_id"], r["factor"]) for r in public).values())
    assert all(
        r["split"] == data.SPLIT and r["prompt_version"] == "srf6-interface-v3" for r in public
    )
    first = public[0]
    task = private[first["qid"]]
    assert first["model_input"]["prompt"] == public_prompt(
        task["world"]["categories"], task["query"]
    )
    payload, _ = data.render_source(task["world"])
    assert (root / first["model_input"]["image_file"]).read_bytes() == payload
    assert first["model_input"]["image_sha256"] == hashlib.sha256(payload).hexdigest()
    plan = json.loads((root / "DATA/PROCESSOR_REVIEW_PLAN.json").read_text())
    assert len(plan["items"]) == plan["count"] == 48
    assert {r["qid"] for r in plan["items"]} == {
        r["qid"] for r in public if r["factor"] == "P1" and r["variant"] == 0
    }
    assert all(
        r["visual_review"] == "NOT_RUN"
        and r["processor_image"] is None
        and r["processor_routing"] is None
        for r in plan["items"]
    )
    assert plan["historical_visual_review_reused"] is False


def test_verify_actual_built_data_and_reuse_binding_stats(built):
    root, _, _ = built
    report = data.verify_data(root)
    assert report["status"] == "PASS"
    assert report["data_binding"]["content_rehashed"] == 0
    assert report["historical_binding"]["content_rehashed"] == 0
    assert report["processor_readability"] == "NOT_RUN"


def test_build_never_overwrites_data_or_writes_into_history(built, monkeypatch):
    root, _, historical = built

    def forbidden(*args):
        raise AssertionError("constructor must not be reached")

    monkeypatch.setattr(design, "build_bank", forbidden)
    with pytest.raises(FileExistsError, match="Preserve"):
        data.build(root, historical_roots=historical)
    with pytest.raises(ValueError, match="separate"):
        data.build(historical[0], historical_roots=historical)


def test_collision_saves_failure_and_fixed_bank_without_reroll(tmp_path, bank, monkeypatch):
    historical = history_fixture(tmp_path, collision=bank[0]["tasks"]["P0"][0]["world"])
    root = tmp_path / "new"
    root.mkdir()
    calls = []

    def fixed(*args):
        calls.append(args)
        return copy.deepcopy(bank)

    monkeypatch.setattr(design, "build_bank", fixed)
    with pytest.raises(RuntimeError, match="without reseeding"):
        data.build(root, historical_roots=historical)
    assert calls == [("DEV_SCHEMA_CONFIRM", 8, 20261028)]
    assert (
        json.loads((root / "DATA/AUDIT.json").read_text())["status"] == "HISTORICAL_COLLISION_STOP"
    )
    assert not (root / "DATA/images").exists()
    assert digest(data._rows(root / "DATA/ROOTS.jsonl")) == digest(bank)


def test_load_inputs_rejects_missing_or_duplicate_question(built, tmp_path):
    source, _, _ = built
    (tmp_path / "DATA" / data.SPLIT).mkdir(parents=True)
    public = data._rows(source / "DATA" / data.SPLIT / "MODEL_INPUTS.jsonl")
    private = data._rows(source / "DATA" / data.SPLIT / "TASKS_PRIVATE.jsonl")
    public[-1] = public[0]
    write(tmp_path / "DATA" / data.SPLIT / "MODEL_INPUTS.jsonl", public, rows=True)
    write(tmp_path / "DATA" / data.SPLIT / "TASKS_PRIVATE.jsonl", private, rows=True)
    with pytest.raises(ValueError, match="384 distinct"):
        data.load_inputs(tmp_path)
