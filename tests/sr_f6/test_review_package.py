"""Synthetic persistence only: no actual model, review, or scientific gate."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from sr_f6 import census, census_evidence, review, review_package
from sr_f6.common import digest, file_hash
from srf6_ref.census import summarize
from srf6_ref.core import FAMILIES
from srf6_ref.design import make_root
from srf6_ref.measurement import full_truth_response


def save(root, relative, value, *, lines=False):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(review_package._payload(value, lines=lines))
    return {"path": relative, "sha256": file_hash(path)}


def read(root, reference, *, lines=False):
    text = (root / reference["path"]).read_text()
    return [json.loads(line) for line in text.splitlines()] if lines else json.loads(text)


@pytest.mark.parametrize("lines", [False, True])
@pytest.mark.parametrize("number", ["1e999", "-1e999"])
def test_package_reader_rejects_exponent_overflow(tmp_path, lines, number):
    path = tmp_path / ("input.jsonl" if lines else "input.json")
    path.write_text('{"value":' + number + "}\n")
    with pytest.raises(ValueError, match="Nonfinite evidence"):
        review_package._Reader(tmp_path).read(path.name, lines=lines)


@pytest.mark.parametrize("lines", [False, True])
def test_package_reader_preserves_finite_numbers_and_full_text_strings(tmp_path, lines):
    item = dict(large=1e308, negative=-1e308, full_text='{"answer":1e999,"answer":-1e999}')
    ref = save(
        tmp_path, "input.jsonl" if lines else "input.json", [item] if lines else item, lines=lines
    )
    result = review_package._Reader(tmp_path).read(ref["path"], lines=lines)
    assert result == ([item] if lines else item)


def training_evidence(root):
    certificates = [
        dict(
            root_id=f"TRAIN_SYNTHETIC_{i:04d}",
            certificate=dict(root_id=f"TRAIN_SYNTHETIC_{i:04d}", coverage_P0=0, coverage_P1=0.1),
        )
        for i in range(288)
    ]
    audit = dict(
        status="PASS",
        model_calls=0,
        independent_native_oracles=2,
        cross_split_world_duplicates=0,
        splits={
            name: dict(
                roots=roots,
                questions=questions,
                factors=["P0", "P1"],
                family_questions=dict.fromkeys(FAMILIES, per_family),
            )
            for name, roots, questions, per_family in (
                ("DEV_CAL", 48, 384, 64),
                ("TRAIN", 288, 2304, 384),
            )
        },
        training_manipulation=dict(
            status="PASS",
            roots=288,
            mean_gap=sum([0.1] * 288) / 288,
            strict_gain_fraction=1,
            model_outputs_used=False,
        ),
    )
    return [
        save(root, "DATA/CPU_AUDIT.json", audit),
        save(root, "DATA/ROOT_CERTIFICATES_PRIVATE.jsonl", certificates, lines=True),
    ]


class SyntheticPackage:
    """Mock only phase/census verified boundaries; real sampler/support/writer."""

    def __init__(self, root, monkeypatch):
        self.root, self.calls, self.phase_calls = root, [], []
        proto = make_root("DEV_CAL", "BINDING", 0, 2026102701)["tasks"]["P0"][0]
        truth = full_truth_response(proto["world"], proto["query"])
        tasks, records, raw = {}, [], {}
        for parent, factor, family in review.BLOCKS:
            for index in range(8):
                rid = f"DEV_CAL_{family}_{index:04d}"
                for variant in range(4):
                    qid = f"{rid}_{factor}_v{variant}"
                    tasks[qid] = dict(
                        proto, qid=qid, root_id=rid, factor=factor, family=family, variant=variant
                    )
                    for draw in range(8):
                        row_id = f"{parent}/{qid}/{draw}"
                        eos = draw == 7
                        item = dict(
                            parent=parent,
                            factor=factor,
                            family=family,
                            root_id=rid,
                            qid=qid,
                            row_id=row_id,
                            draw=draw,
                            group_id=f"{parent}/{qid}",
                            phase="PARENT_CENSUS",
                            full_text=truth + " " * draw,
                            seed=draw,
                            tokens=[1, 2],
                            model_input={"prompt": "SYNTHETIC TEST"},
                            finish_reason="length" if eos else "balanced",
                            truncated=False,
                            balanced_token_count=None if eos else 2,
                            boundary_token_count=None if eos else 2,
                            policy_hash="a" * 64,
                            A=1,
                            J=int(not eos),
                            V=1,
                            finished=int(not eos),
                            S=None if eos else 1,
                            B=None if eos else 1,
                        )
                        records.append(item)
                        raw[row_id] = dict(
                            item,
                            finish_reason="eos" if eos else "balanced",
                            execution_origin="SYNTHETIC_TEST",
                        )
        refs = training_evidence(root)
        refs.append(
            save(root, "DATA/DEV_CAL/TASKS_PRIVATE.jsonl", list(tasks.values()), lines=True)
        )
        data_ref = save(root, "DATA/MANIFEST.json", {"files": refs})
        spec_ref = save(root, "CENSUS_SPEC_FROZEN.json", {"origin": "SYNTHETIC_TEST"})
        summary = summarize(records)
        summary.update(
            status="COMPLETE_NOT_GATE_SIGNED",
            source_rows=9216,
            frozen_spec_sha256=spec_ref["sha256"],
            optimizer_updates=0,
            real_measurement_review="NOT_RUN",
        )
        self.verified = dict(
            status="VERIFIED_STORED_CENSUS_NOT_GATE_SIGNED",
            training_authorized=False,
            actual_recorded_calls=9216,
            model_calls_by_this_validator=0,
            optimizer_updates_by_this_validator=0,
            input_producer=dict(
                phase="CENSUS_S0", identity={"synthetic": True}, identity_sha256="f" * 64
            ),
            summary=summary,
            records=records,
            raw_by_row=raw,
            inputs=[spec_ref, data_ref, refs[-1]],
        )
        self.bind_sources(data_ref, monkeypatch)

        def validate(root, *, origin_deployment):
            assert Path(root) == self.root and origin_deployment == "SYNTHETIC_S0"
            self.calls.append(origin_deployment)
            return self.verified

        monkeypatch.setattr(census_evidence, "validate_saved_census", validate)

    def bind_sources(self, data_ref, monkeypatch):
        deployment = review_package._deployment()
        files = [
            {"path": str(p.relative_to(deployment)), "sha256": file_hash(p)}
            for p in sorted((deployment / "src").rglob("*.py"))
        ]
        current = save(self.root, "PHASES/PILOT_V1/SOURCE_MANIFEST.json", {"files": files})
        recipe = save(
            self.root,
            "PHASES/CENSUS_REVIEW_V1/SOURCE_MANIFEST.json",
            {"files": [next(row for row in files if row["path"] == "src/sr_f6/review.py")]},
        )
        origin = dict(
            phase="CENSUS_S0",
            deployment="SYNTHETIC_S0",
            identity={"path": "IDENTITY.json", "sha256": "f" * 64},
        )
        self.recipe_identity = dict(
            phase="CENSUS_REVIEW_V1",
            deployment="SYNTHETIC_86_DEPLOYMENT",
            revision=review_package.RECIPE_REVISION,
            origin=origin,
            source_manifest=recipe,
            data_manifest=data_ref,
            config_sha256="a" * 64,
        )
        recipe_ref = save(self.root, "PHASES/CENSUS_REVIEW_V1/IDENTITY.json", self.recipe_identity)
        spec = dict(
            status="REGISTERED_REVIEW_SAMPLER_NOT_EXECUTED",
            context=self.recipe_identity,
            phase_identity_sha256=recipe_ref["sha256"],
            version=review.VERSION,
            seed=2026102701,
            seed_domains=review.DOMAINS,
            required_count=360,
            per_parent_factor=60,
            per_family_within_cell=10,
            root_cap=4,
            root_cap_scope="parent_factor",
            unique_key=["parent", "factor", "qid", "exact_full_text"],
            raw_source=(
                "Original CENSUS/raw records, aligned by frozen qid and draw; "
                "never group/scorer-normalized finish_reason"
            ),
            labels_visible_to_blind_extractors=False,
            review_status="NOT_RUN",
            model_calls=0,
            optimizer_updates=0,
            training_allowed=False,
        )
        spec_ref = save(self.root, review_package.SPEC_PATH, spec)
        self.identity = dict(
            phase="PILOT_V1",
            deployment=str(deployment),
            revision="1" * 40,
            origin=origin,
            source_manifest=current,
            data_manifest=data_ref,
            config_sha256="a" * 64,
            upstream=[
                dict(
                    producer_phase="CENSUS_REVIEW_V1",
                    identity_sha256=recipe_ref["sha256"],
                    evidence=[spec_ref],
                )
            ],
        )
        phase_ref = save(self.root, "PHASES/PILOT_V1/IDENTITY.json", self.identity)
        self.kwargs = dict(
            phase_identity_sha256=phase_ref["sha256"],
            sampler_phase_identity_sha256=recipe_ref["sha256"],
            review_spec_sha256=spec_ref["sha256"],
        )
        self.producer = dict(
            phase="PILOT_V1",
            identity=self.identity,
            identity_sha256=phase_ref["sha256"],
            deployment=str(deployment),
            status="BOUND_NOT_AUTHORIZED",
            training_allowed=False,
        )
        self.recipe = dict(
            phase="CENSUS_REVIEW_V1",
            identity=self.recipe_identity,
            identity_sha256=recipe_ref["sha256"],
            deployment=self.recipe_identity["deployment"],
            status="BOUND_NOT_AUTHORIZED",
            training_allowed=False,
        )

        def verify(root, phase, *, deployment, expected_identity_sha256):
            self.phase_calls.append(phase)
            value = self.producer if phase == "PILOT_V1" else self.recipe
            assert Path(root) == self.root and str(deployment) == value["deployment"]
            assert expected_identity_sha256 == value["identity_sha256"]
            return copy.deepcopy(value)

        monkeypatch.setattr(review_package.phases, "verify_phase", verify)

    def write(self):
        return review_package.write_package(self.root, **self.kwargs)

    def verify(self, reference):
        return review_package.verify_package(self.root, reference, **self.kwargs)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    return SyntheticPackage(tmp_path, monkeypatch)


def test_writer_replays_fixed_full_census_and_preserves_adverse_support(fixture):
    before = digest(fixture.verified)
    receipt = fixture.write()
    assert fixture.calls == ["SYNTHETIC_S0"]
    package = read(fixture.root, receipt["package"])
    support = read(fixture.root, package["support"])
    assert receipt["status"] == "BLIND_PACKAGE_ONLY"
    assert receipt["execution_origin"] == "SYNTHETIC_TEST"
    assert receipt["review_status"] == "REVIEW_NOT_RUN" and not receipt["training_allowed"]
    assert support["status"] == "INSUFFICIENT_REGISTERED_EXPOSURE"
    assert support["failures"] and support["gate_status"] == "NOT_SIGNED"
    assert support["observed"]["cells"]["b0/P0"]["S_known"] == 1344
    assert package["context"]["phase"] == package["producer"]["phase"] == "PILOT_V1"
    assert package["sampler_recipe"]["phase"] == "CENSUS_REVIEW_V1"
    assert package["sampler_recipe"]["identity"]["revision"] == review_package.RECIPE_REVISION
    assert package["input_producer"]["phase"] == "CENSUS_S0"
    assert digest(fixture.verified) == before
    result = fixture.verify(receipt["package"])
    assert len(fixture.calls) == 2
    assert result["status"] == "VERIFIED_REVIEW_PACKAGE_NOT_GATE_SIGNED"
    assert result["reviewed_answers"] == 0 and result["gate_status"] == "NOT_SIGNED"
    assert len(result["artifacts"]["blind"]) == 360
    assert len(result["artifacts"]["tasks"]) == 384
    ledger = result["artifacts"]["ledger"]
    blind = result["artifacts"]["blind"]
    selected = {row["anonymous_id"]: row for row in ledger["selected"]}
    assert ledger["seed"] == 2026102701
    for item in blind:
        assert set(item) == {
            "anonymous_id",
            "raw_text",
            "protocol_version",
            "generation_finish_reason",
        }
        original = fixture.verified["raw_by_row"][selected[item["anonymous_id"]]["row_id"]]
        assert item["raw_text"] == original["full_text"]
        assert item["generation_finish_reason"] == original["finish_reason"]
    eos = [item for item in blind if item["generation_finish_reason"] == "eos"]
    assert eos and all(
        selected[item["anonymous_id"]]["program_labels"]["finish"] == 0 for item in eos
    )
    packet = read(fixture.root, package["judgment_tasks"], lines=True)
    assert all(set(item) == {"anonymous_id", "world", "query"} for item in packet)
    assert [item["anonymous_id"] for item in packet] == [item["anonymous_id"] for item in blind]
    assert fixture.phase_calls == ["PILOT_V1", "CENSUS_REVIEW_V1"] * 2


def test_retry_is_exact_immutable_and_private_directories_are_separate(fixture):
    receipt = fixture.write()
    folder = fixture.root / review_package.DESTINATION
    stats = {str(p): p.stat().st_mtime_ns for p in folder.rglob("*")}
    assert fixture.write() == receipt
    assert {str(p): p.stat().st_mtime_ns for p in folder.rglob("*")} == stats
    for path in (folder, *folder.rglob("*")):
        assert path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)
    assert (folder / "private").is_dir() and (folder / "blind").is_dir()


@pytest.mark.parametrize(
    "mutation", ["text", "labels", "permission", "missing", "extra", "symlink"]
)
def test_existing_package_corruption_is_preserved_and_rejected(fixture, mutation):
    receipt = fixture.write()
    folder = fixture.root / review_package.DESTINATION
    target = folder / "blind/EXTRACTION.jsonl"
    if mutation == "text":
        target.write_bytes(target.read_bytes() + b" ")
    elif mutation == "labels":
        target = folder / "private/selection.json"
        value = json.loads(target.read_text())
        value["selected"][0]["program_labels"]["S"] = 0
        target.write_bytes(review_package._payload(value))
    elif mutation == "permission":
        target.chmod(0o644)
    elif mutation == "missing":
        target.unlink()
    elif mutation == "extra":
        target = folder / "extra"
        target.write_text("SYNTHETIC")
    else:
        target.unlink()
        target.symlink_to(folder / "private/selection.json")
    with pytest.raises(PermissionError):
        fixture.verify(receipt["package"])
    with pytest.raises(PermissionError):
        fixture.write()
    assert target.exists() if mutation != "missing" else not target.exists()


def test_complete_stage_can_resume_but_partial_stage_never_repaired(fixture):
    receipt = fixture.write()
    target = fixture.root / review_package.DESTINATION
    stage = target.with_name(".REVIEW_PACKAGE.staging")
    target.rename(stage)
    assert fixture.write() == receipt and target.exists() and not stage.exists()
    target.rename(stage)
    (stage / "blind/EXTRACTION.jsonl").unlink()
    with pytest.raises(PermissionError, match="Partial package"):
        fixture.write()
    assert stage.exists() and not target.exists()


def test_fixed_seed_and_destination_have_no_production_override(fixture):
    with pytest.raises(TypeError):
        review_package.write_package(fixture.root, seed=42, **fixture.kwargs)
    with pytest.raises(TypeError):
        review_package.write_package(fixture.root, destination="other", **fixture.kwargs)
    with pytest.raises(PermissionError, match="fixed v2 package"):
        fixture.verify({"path": "somewhere/BINDING.json", "sha256": "a" * 64})


@pytest.mark.parametrize(
    "mutation", ["short", "wrong_origin", "technical_failure", "insufficient_unique"]
)
def test_incomplete_or_unbound_census_cannot_publish(fixture, mutation):
    value = fixture.verified
    if mutation == "short":
        value["records"].pop()
    elif mutation == "wrong_origin":
        value["input_producer"]["identity_sha256"] = "e" * 64
    elif mutation == "technical_failure":
        value["status"] = "INCOMPLETE"
    else:
        for row in value["records"]:
            row["source_row_id"] = "one-explicit-provenance-alias"
            row.update(full_text="same", A=0, J=0, V=0, S=None, B=None, finished=0)
            value["raw_by_row"][row["row_id"]]["full_text"] = "same"
    with pytest.raises((PermissionError, ValueError)):
        fixture.write()
    assert not (fixture.root / review_package.DESTINATION).exists()


def test_current_loaded_sampler_must_match_frozen_recipe_bytes(fixture, monkeypatch):
    original = review_package.file_hash
    monkeypatch.setattr(
        review_package,
        "file_hash",
        lambda p: "0" * 64 if Path(p) == Path(review.__file__) else original(p),
    )
    with pytest.raises(PermissionError, match="sampler bytes"):
        fixture.write()
    assert fixture.calls == []


def test_loaded_module_from_another_checkout_is_rejected(fixture, monkeypatch):
    import sys
    import types

    alien = fixture.root / "alien.py"
    alien.write_text("# SYNTHETIC\n")
    monkeypatch.setitem(
        sys.modules,
        "sr_f6.synthetic_foreign",
        types.SimpleNamespace(__name__="sr_f6.synthetic_foreign", __file__=str(alien)),
    )
    with pytest.raises(PermissionError, match="outside the verified deployment"):
        fixture.write()
    assert fixture.calls == []


def test_lazy_dependency_from_other_checkout_is_rejected_before_publication(fixture, monkeypatch):
    import sys
    import types

    original = census_evidence.validate_saved_census
    alien = fixture.root / "late_dependency.py"
    alien.write_text("# SYNTHETIC\n")

    def late_import(root, *, origin_deployment):
        result = original(root, origin_deployment=origin_deployment)
        monkeypatch.setitem(
            sys.modules,
            "sr_f6.synthetic_late_foreign",
            types.SimpleNamespace(__name__="sr_f6.synthetic_late_foreign", __file__=str(alien)),
        )
        return result

    monkeypatch.setattr(census_evidence, "validate_saved_census", late_import)
    with pytest.raises(PermissionError, match="outside the verified deployment"):
        fixture.write()
    assert fixture.calls == ["SYNTHETIC_S0"]
    assert not (fixture.root / review_package.DESTINATION).exists()


def test_complete_census_change_cannot_trigger_a_second_selection(fixture):
    receipt = fixture.write()
    path = fixture.root / receipt["package"]["path"]
    before = path.read_bytes()
    row = fixture.verified["records"][0]
    row["full_text"] += " " * 20
    fixture.verified["raw_by_row"][row["row_id"]]["full_text"] = row["full_text"]
    with pytest.raises(PermissionError):
        fixture.write()
    assert path.read_bytes() == before


@pytest.mark.parametrize("mutation", ["data", "recipe", "spec", "unbound_spec", "producer"])
def test_source_and_recipe_pins_fail_closed_before_sampling(fixture, mutation):
    if mutation == "data":
        (fixture.root / "DATA/DEV_CAL/TASKS_PRIVATE.jsonl").write_text("{}\n")
    elif mutation == "recipe":
        fixture.recipe_identity["revision"] = "0" * 40
    elif mutation == "spec":
        (fixture.root / review_package.SPEC_PATH).write_text("{}")
    elif mutation == "unbound_spec":
        fixture.identity["upstream"] = []
    else:
        fixture.producer["phase"] = "CENSUS_REVIEW_V1"
    with pytest.raises(PermissionError):
        fixture.write()
    assert not (fixture.root / review_package.DESTINATION).exists()


def test_exclusive_lock_prevents_second_writer(fixture):
    import fcntl

    path = fixture.root / "PHASES/PILOT_V1/.review-package.lock"
    with path.open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            fixture.write()
    assert fixture.calls == []


def test_atomic_publication_does_not_replace_even_an_empty_racing_target(tmp_path):
    stage, target = tmp_path / "stage", tmp_path / "target"
    stage.mkdir()
    target.mkdir()
    (stage / "synthetic").write_text("SYNTHETIC")
    with pytest.raises(FileExistsError):
        review_package._publish(stage, target)
    assert stage.is_dir() and target.is_dir() and list(target.iterdir()) == []
    target.rmdir()
    review_package._publish(stage, target)
    assert not stage.exists() and (target / "synthetic").read_text() == "SYNTHETIC"


def test_package_root_cannot_be_an_alias(fixture):
    alias = fixture.root / "alias"
    alias.symlink_to(fixture.root, target_is_directory=True)
    with pytest.raises(PermissionError, match="Canonical run root"):
        review_package.write_package(alias, **fixture.kwargs)


def test_real_saved_census_validator_is_called_and_detects_raw_mutation(fixture, monkeypatch):
    """Real full 9216 persistence replay; only deployment verification is mocked."""
    spec = importlib.util.spec_from_file_location(
        "synthetic_census_evidence_fixture", Path(__file__).with_name("test_census_evidence.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_freeze = census.freeze_spec

    def freeze(root, parents):
        refs = [
            {"path": relative, "sha256": file_hash(root / relative)}
            for relative in (
                "DATA/CPU_AUDIT.json",
                "DATA/ROOT_CERTIFICATES_PRIVATE.jsonl",
                "DATA/DEV_CAL/TASKS_PRIVATE.jsonl",
                "DATA/DEV_CAL/MODEL_INPUTS.jsonl",
            )
        ]
        save(root, "DATA/MANIFEST.json", {"files": refs})
        return original_freeze(root, parents)

    monkeypatch.setattr(census, "freeze_spec", freeze)
    # Replace this test's toy specification with the full persistence fixture.
    (fixture.root / "CENSUS_SPEC_FROZEN.json").unlink()
    context, public = module.build_complete_fixture(fixture.root)
    data_hash = file_hash(fixture.root / "DATA/MANIFEST.json")
    context["data_manifest"]["sha256"] = data_hash
    task_map = {t["qid"]: t for t in census.rows(fixture.root / "DATA/DEV_CAL/TASKS_PRIVATE.jsonl")}
    for path in (fixture.root / "runtime_identity").glob("*.json"):
        value = json.loads(path.read_text())
        value["data_hash"] = data_hash
        path.write_bytes(review_package._payload(value))
    for parent in census.PARENTS:
        for item in public:
            relative = f"CENSUS/raw/{parent}/{item['qid']}.jsonl"
            rows = census.rows(fixture.root / relative)
            for row in rows:
                row.update(data_hash=data_hash, execution_origin="SYNTHETIC_TEST")
                if row["finish_reason"] == "eos":
                    row["full_text"] = row["full_text"][:-1]
            save(fixture.root, relative, rows, lines=True)
            group_relative = f"CENSUS/groups/{parent}/{item['qid']}.json"
            group = read(fixture.root, {"path": group_relative})
            group["records"] = census.annotate_group(rows, task_map[item["qid"]], group["spec"])
            group["credit"] = census.counterfactual(group["records"], group["spec"])
            save(fixture.root, group_relative, group)
    fixture.bind_sources({"path": "DATA/MANIFEST.json", "sha256": data_hash}, monkeypatch)
    monkeypatch.setattr(
        census_evidence.phases,
        "verify_origin",
        lambda root, *, deployment: dict(identity=context, identity_sha256="f" * 64),
    )
    # Undo only the synthetic validator boundary; retain actual source boundary mocks.
    monkeypatch.setattr(census_evidence, "validate_saved_census", ACTUAL_VALIDATE)
    raw_path = fixture.root / f"CENSUS/raw/b0/{public[0]['qid']}.jsonl"
    payload = raw_path.read_bytes()
    for prefix, message in (
        (b'{"full_text":"FORGED_DUPLICATE_VALUE",', "Duplicate JSON"),
        (b'{"nonfinite":NaN,', "Nonfinite JSON"),
    ):
        raw_path.write_bytes(payload.replace(b"{", prefix, 1))
        with pytest.raises(PermissionError, match=message):
            fixture.write()
        assert not (fixture.root / review_package.DESTINATION).exists()
        raw_path.write_bytes(payload)
    receipt = fixture.write()
    result = fixture.verify(receipt["package"])
    assert result["selection_provenance"] == "RECOMPUTED_FROM_VERIFIED_FULL_CENSUS"
    assert len(result["inputs"]) > 3484
    original = census.rows(raw_path)
    original[0]["full_text"] += " tamper"
    save(fixture.root, str(raw_path.relative_to(fixture.root)), original, lines=True)
    with pytest.raises(PermissionError, match="independently persisted original raw"):
        fixture.verify(receipt["package"])


ACTUAL_VALIDATE = census_evidence.validate_saved_census
