"""Synthetic full-matrix fixtures: no actual confirmation or independent review."""

from __future__ import annotations

import copy
import errno
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from sr_f6 import review_package as publication
from sr_f6.common import digest
from sr_f6_followup import review_package, sampling, source

OBJECT = dict(
    evidence=[dict(series="Alpha", category="X", value=1)],
    selected=["X"],
    bindings=[dict(category="X", operand=1)],
    aggregate="sum",
    answer=1,
)
RAW = json.dumps(OBJECT, separators=(",", ":"))
LABELS = dict(A=1, J=1, S=1, B=1, V=1, finish=1)


class Fixture:
    def __init__(self, root, monkeypatch):
        self.root = root
        self.records, self.raw, tasks = [], {}, {}
        for parent, factor, family in sampling.BLOCKS:
            for r in range(8):
                root_id = f"DEV_SCHEMA_CONFIRM_{family}_{r:04d}"
                for v in range(4):
                    qid = f"{root_id}_{factor}_v{v}"
                    tasks[qid] = dict(
                        qid=qid,
                        split=sampling.SPLIT,
                        root_id=root_id,
                        family=family,
                        factor=factor,
                        world=dict(categories=["X"], series=dict(Alpha=[1], Beta=[1])),
                        query=dict(aggregate="sum", k=1, value="Alpha"),
                        gold_answer="1",
                        gold_selected=["X"],
                        gold_bindings={"X": "1"},
                    )
                    for draw in range(8):
                        row_id = f"{parent}/{qid}/{draw}"
                        row = dict(
                            phase=sampling.PHASE,
                            split=sampling.SPLIT,
                            parent=parent,
                            factor=factor,
                            family=family,
                            root_id=root_id,
                            qid=qid,
                            draw=draw,
                            row_id=row_id,
                            full_text=RAW,
                            raw_text=RAW,
                            seed=100 + draw,
                            tokens=[10, 20],
                            model_input={"prompt": qid},
                            finish_reason="balanced",
                            execution_origin="SYNTHETIC_TEST",
                            score=dict(
                                **LABELS,
                                finish_reason="balanced",
                                unknown_reasons={},
                                witness={"gold_selected": ["X"], "gold_answer": "1"},
                            ),
                            **LABELS,
                        )
                        self.records.append(row)
                        self.raw[row_id] = copy.deepcopy(row)
        self.consumer = dict(
            deployment=str(root / "new-consumer"),
            revision="c" * 40,
            source_manifest_sha256="d" * 64,
            registration_sha256="e" * 64,
        )
        self.producer = dict(
            phase=sampling.PHASE,
            deployment=str(source.PRODUCER_DEPLOYMENT),
            revision=source.PRODUCER_REVISION,
        )
        self.evidence = dict(
            status="VERIFIED_STORED_CONFIRMATION_NOT_GATE_SIGNED",
            input_producer=self.producer,
            registration_sha256="f" * 64,
            source_rows=9216,
            actual_recorded_calls=9216,
            records=self.records,
            raw_by_row=self.raw,
            inputs=[],
            external_inputs=[],
        )
        self.support = dict(
            status="VERIFIED_SUPPORT_PASS_REVIEW_PENDING",
            consumer=self.consumer,
            records_sha256=digest(self.records),
            raw_records_sha256=digest(self.raw),
            dependencies=[],
        )
        self.context = dict(
            root=root,
            consumer=self.consumer,
            producer_context={"tasks": tasks},
            registration={"spec": {"sha256": "a" * 64}},
            evidence=self.evidence,
            records=self.records,
            raw_by_row=self.raw,
            support_binding=self.support,
            support_binding_ref={"path": str(root / "SUPPORT_BINDING.json"), "sha256": "b" * 64},
        )
        self.calls = []

        def require(root, purpose):
            assert Path(root) == self.root and purpose == "review"
            self.calls.append(purpose)
            return self.context

        monkeypatch.setattr(source, "require_support", require)
        monkeypatch.setattr(source, "_root", lambda root: Path(root))
        ledger, blind = sampling.build_review(
            self.records, raw_by_row=self.raw, source_identity={"historical": "a" * 64}
        )
        self.history = dict(
            references=[],
            package={},
            ledger=copy.deepcopy(ledger),
            blind=copy.deepcopy(blind),
            tasks=[
                dict(
                    anonymous_id=row["anonymous_id"],
                    world=tasks[row["qid"]]["world"],
                    query=tasks[row["qid"]]["query"],
                )
                for row in ledger["selected"]
            ],
            summary={"status": "BLOCKING_FINDINGS"},
        )
        monkeypatch.setattr(source, "repair_history", lambda: copy.deepcopy(self.history))

    def write(self, path, value, *, lines=False):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = (
            "".join(json.dumps(v) + "\n" for v in value) if lines else json.dumps(value)
        ).encode()
        target.write_bytes(raw)
        return dict(path=path, sha256=hashlib.sha256(raw).hexdigest())

    def read(self, ref):
        path = self.root / ref["path"]
        return (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.suffix == ".jsonl"
            else json.loads(path.read_text())
        )

    def package(self):
        result = review_package.write_package(self.root)
        self.verification = review_package.verify_package(self.root)
        return result


@pytest.fixture
def fx(tmp_path, monkeypatch):
    return Fixture(tmp_path, monkeypatch)


def test_full_matrix_fixed_package_and_distinct_producer(fx):
    before = copy.deepcopy(fx.records)
    first = fx.package()
    assert review_package.write_package(fx.root) == first
    assert fx.records == before
    out = fx.verification
    assert out["input_producer"] == fx.producer and out["consumer"] == fx.consumer
    assert out["training_allowed"] is False
    ledger = out["artifacts"]["ledger"]
    assert ledger["raw_count"] == 9216 and ledger["unique_units"] == 1152
    assert ledger["seed"] == 2026102701 and ledger["selected_count"] == 360
    selected = ledger["selected"]
    assert Counter((r["parent"], r["factor"], r["family"]) for r in selected) == dict.fromkeys(
        sampling.BLOCKS, 10
    )
    assert max(Counter((r["parent"], r["factor"], r["root_id"]) for r in selected).values()) <= 4
    for item in out["artifacts"]["blind"]:
        assert set(item) == {
            "anonymous_id",
            "raw_text",
            "protocol_version",
            "generation_finish_reason",
        }
        assert item["raw_text"] == RAW
    packet = fx.read(out["artifacts"]["package"]["judgment_tasks"])
    assert all(set(r) == {"anonymous_id", "world", "query"} for r in packet)
    assert all("DEV_CAL" not in r["qid"] for r in selected)


def test_support_failure_creates_no_package(fx, monkeypatch):
    def fail(*args, **kwargs):
        raise PermissionError("Support FAIL preserved")

    monkeypatch.setattr(source, "require_support", fail)
    with pytest.raises(PermissionError, match="FAIL"):
        review_package.write_package(fx.root)
    assert not (fx.root / "REVIEW").exists()


@pytest.mark.parametrize("mutation", ["phase", "producer", "matrix", "raw_hash", "task"])
def test_wrong_source_or_incomplete_matrix_rejected(fx, mutation):
    if mutation == "phase":
        fx.records[0]["phase"] = "PARENT_CENSUS"
        fx.support["records_sha256"] = digest(fx.records)
    elif mutation == "producer":
        fx.producer["revision"] = "5" * 40
    elif mutation == "matrix":
        fx.records.pop()
    elif mutation == "raw_hash":
        fx.raw[next(iter(fx.raw))]["full_text"] = "replacement"
    else:
        for task in fx.context["producer_context"]["tasks"].values():
            task["split"] = "DEV_CAL"
    with pytest.raises((PermissionError, ValueError)):
        review_package.write_package(fx.root)


@pytest.mark.parametrize("kind", ["partial", "tamper", "symlink"])
def test_immutable_package_preserves_partial_or_tampered_bytes(fx, kind):
    directory = fx.root / review_package.DESTINATION
    if kind == "partial":
        directory.mkdir(parents=True)
        (directory / "partial").write_text("preserve")
    else:
        fx.package()
        path = directory / "blind/EXTRACTION.jsonl"
        if kind == "symlink":
            path.unlink()
            path.symlink_to(directory / "private/selection.json")
        else:
            path.write_text("changed")
    with pytest.raises(PermissionError):
        review_package.write_package(fx.root)
    assert directory.exists()


@pytest.mark.parametrize("error", [errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP])
def test_ceph_fallback_preserves_exact_complete_publication(fx, monkeypatch, error):
    def unsupported(*args):
        raise OSError(error, "fixture unsupported")

    monkeypatch.setattr(publication, "_rename_exclusive", unsupported)
    fx.package()
    assert (fx.root / "REVIEW/.PACKAGE.staging").is_dir()
    assert review_package.verify_package(fx.root)["package_ref"] == fx.verification["package_ref"]


def test_support_checked_before_sampling_and_no_model_hashing(fx, monkeypatch):
    calls = []
    original = sampling.build_review

    def build(*args, **kwargs):
        assert fx.calls == ["review"]
        calls.append("selection")
        return original(*args, **kwargs)

    monkeypatch.setattr(sampling, "build_review", build)

    def forbidden(*args, **kwargs):
        pytest.fail("Package must adopt validated external pins, never rehash model payloads")

    monkeypatch.setattr(source, "_hash", forbidden)
    review_package.write_package(fx.root)
    assert calls == ["selection"]


def test_foreign_partial_stage_preserved_even_beside_complete_package(fx):
    fx.package()
    stage = fx.root / "REVIEW/.PACKAGE.staging"
    stage.mkdir(mode=0o700)
    (stage / "partial").write_text("preserve")
    with pytest.raises(PermissionError):
        review_package.verify_package(fx.root)
    assert (stage / "partial").read_text() == "preserve"


@pytest.mark.parametrize("part", ["ledger", "blind", "tasks"])
def test_repair_cannot_replace_old_selection_text_or_task(fx, part):
    if part == "ledger":
        fx.history[part]["selected"][0]["row_id"] = "different-original"
    elif part == "blind":
        fx.history[part][0]["raw_text"] = "different-original"
    else:
        fx.history[part][0]["query"] = {"aggregate": "max"}
    with pytest.raises(PermissionError, match="preserve every original"):
        review_package.write_package(fx.root)
    assert not (fx.root / "REVIEW").exists()


def test_program_claims_bind_stored_witnesses_but_stay_out_of_blind_packet(fx):
    fx.package()
    artifacts = fx.verification["artifacts"]
    assert len(artifacts["program_claims"]) == 360
    row = artifacts["program_claims"][0]
    assert row["labels"] == LABELS and row["original_score"]["witness"]["gold_answer"] == "1"
    assert "labels" not in artifacts["blind"][0]
    assert artifacts["package"]["review_method"]["path"].endswith("method/REVIEW_METHOD.md")
