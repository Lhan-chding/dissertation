"""CPU fixtures exercise persistence; none are actual evaluation or gate evidence."""

from __future__ import annotations

import copy
import json

import pytest

from sr_f6 import evaluation
from srf6_ref.core import FAMILIES, seed_for
from srf6_ref.design import make_root
from srf6_ref.measurement import full_truth_response


@pytest.fixture(scope="module")
def prototype():
    return make_root("DEV_EVAL", "BINDING", 0, 2026102713)["tasks"]["P0"][0]


def save(root, relative, value, lines=False):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in value) if lines else json.dumps(value)
    )
    return evaluation._reference(root, path)


@pytest.fixture
def fixture(tmp_path, prototype):
    root = tmp_path.resolve()
    public, private = [], []
    for family in FAMILIES:
        for index in range(4):
            rid = f"DEV_EVAL_{family}_{index:04d}"
            for variant in range(4):
                task = dict(
                    copy.deepcopy(prototype),
                    qid=f"{rid}_P0_v{variant}",
                    root_id=rid,
                    family=family,
                    variant=variant,
                    factor="P0",
                    split="DEV_EVAL",
                )
                private.append(task)
                public.append(
                    dict(
                        **{
                            k: task[k]
                            for k in ("qid", "root_id", "family", "variant", "factor", "split")
                        },
                        protocol="structured_v4",
                        model_input=dict(
                            prompt=task["qid"],
                            image_file="DATA/images/SYNTHETIC.png",
                            image_sha256="f" * 64,
                        ),
                    )
                )
    refs = {
        "public": save(root, "DATA/DEV_EVAL/MODEL_INPUTS.jsonl", public, True),
        "private": save(root, "DATA/DEV_EVAL/TASKS_PRIVATE.jsonl", private, True),
    }
    endpoints = [
        dict(
            kind="FROZEN_PARENT",
            endpoint_id=f"{p}/parent",
            parent=p,
            checkpoint_step=0,
            responses=192,
            tensor_sha256="e" * 64,
        )
        for p in ("b0", "b1", "b2")
    ]
    endpoints += [
        dict(
            kind="PILOT_ENDPOINT",
            endpoint_id=f"{p}/{a}/step12",
            parent=p,
            checkpoint_step=12,
            responses=192,
            run_id=f"SYNTHETIC/PILOT/{p}/r0/{a}",
        )
        for p in ("b0", "b1", "b2")
        for a in ("U10", "U11", "C_SHUF")
    ]
    plan = dict(
        split="DEV_EVAL",
        questions=96,
        draws_per_question=2,
        checkpoint_count=12,
        responses=2304,
        formal_test_access=False,
        recovery_endpoint_is_additional_checkpoint=False,
        descriptive_summary=copy.deepcopy(evaluation.SUMMARY_RECIPE),
        inputs=refs,
        endpoints=endpoints,
        seeds_by_parent={
            p: [
                dict(
                    qid=qid,
                    seeds=[seed_for(2026102701, "pilot_dev_eval", p, qid, d) for d in range(2)],
                )
                for qid in sorted(t["qid"] for t in public)
            ]
            for p in ("b0", "b1", "b2")
        },
    )
    return (
        root,
        {"phase": "PILOT_V1", "evaluation": plan},
        full_truth_response(prototype["world"], prototype["query"]),
    )


class SyntheticRuntime:
    def __init__(self, truth, *, partial=False, technical=False, mutate=False):
        self.truth, self.partial, self.technical, self.mutate = truth, partial, technical, mutate
        self.calls = 0

    def generate_group(self, model_input, run_root, seeds, *, protocol, on_completion):
        self.calls += 1
        ident = self.identity
        result = []
        for draw, seed in enumerate(seeds):
            reason = "balanced" if draw == 0 else "eos"
            row = dict(
                full_text=self.truth if draw == 0 else '{"answer":NaN,"answer":0}',
                tokens=[1, 2],
                raw_tokens=[1, 2],
                sampler_logprobs=[-0.25, -0.5],
                model_input=copy.deepcopy(model_input),
                seed=seed,
                group_seed=seeds[0],
                group_row_index=draw,
                group_size=2,
                protocol=protocol,
                execution_mode="EVALUATION",
                execution_origin="SYNTHETIC_TEST",
                task_id="SYNTHETIC-evaluation",
                policy_tensor_hash=ident["policy_tensor_sha256"],
                policy_hash=ident["policy_tensor_sha256"],
                policy_step=ident["checkpoint_step"],
                step=ident["checkpoint_step"],
                source_sha256=ident["source_sha256"],
                data_hash=ident["data_sha256"],
                phase="PILOT_V1",
                phase_identity_sha256=ident["phase_identity_sha256"],
                run_root=str(run_root),
                run_id=ident["evaluation_run_id"],
                adapter_identity=dict(parent=ident["parent"], step=ident["checkpoint_step"]),
                completion_token_count=2,
                max_new_tokens=1536,
                finish_reason=reason,
                truncated=False,
                balanced_token_count=2 if reason == "balanced" else None,
                boundary_token_count=2 if reason == "balanced" else None,
                prompt_tensor_hash="d" * 64,
                native_prompt_inputs=dict(input_ids=[[1, 2]], attention_mask=[[1, 1]]),
                sampler_logprob_source="actual_generate_raw_logits_selected_token",
                generation_status="TECHNICAL_INVALID" if self.technical else "COMPLETE",
                technical_validation_errors=["SYNTHETIC_FAILURE"] if self.technical else [],
            )
            if ident["checkpoint_commit_sha256"] is not None:
                row.update(
                    checkpoint_commit_sha256=ident["checkpoint_commit_sha256"],
                    checkpoint_run_id=ident["checkpoint_run_id"],
                    checkpoint_step=12,
                )
            on_completion(row)
            result.append(row)
            if self.partial:
                raise RuntimeError("SYNTHETIC interrupt after one persisted draw")
            if self.mutate:
                row["full_text"] += " changed after callback"
        return result


def run(fixture, runtime, endpoint="b0/parent"):
    root, spec, _ = fixture
    return evaluation.run_synthetic_endpoint(root, spec=spec, endpoint_id=endpoint, runtime=runtime)


def test_complete_raw_preservation_exact_retry_and_unknown_denominators(fixture):
    root, _, truth = fixture
    runtime = SyntheticRuntime(truth)
    result = run(fixture, runtime)
    assert result["status"] == "SYNTHETIC_ONLY" and result["responses"] == 192
    assert runtime.calls == 96 and not (root / evaluation.DIRECTORY).exists()
    assert result["metrics"]["S"]["known"] == result["metrics"]["B"]["known"] == 96
    assert result["metrics"]["S"]["unknown"] == 96
    assert result["metrics"]["A"]["known"] == 192
    directory = root / evaluation.SYNTHETIC_DIRECTORY / "b0/parent"
    paths = list(directory.rglob("*.json"))
    before = {p: p.stat().st_mtime_ns for p in paths}
    retry = SyntheticRuntime(truth)
    assert run(fixture, retry) == result and retry.calls == 0
    assert before == {p: p.stat().st_mtime_ns for p in paths}
    raw = evaluation._read(next(directory.glob("groups/*/draw-1.json")))
    assert raw["full_text"] == '{"answer":NaN,"answer":0}'
    assert "score" not in raw and "A" not in raw and raw["finish_reason"] == "eos"
    assert result["gate_status"] == "NOT_SIGNED" and not result["training_authorized"]


@pytest.mark.parametrize("failure", ["partial", "technical", "mutate"])
def test_interrupted_invalid_or_changed_rows_preserved_without_redraw(fixture, failure):
    root, _, truth = fixture
    runtime = SyntheticRuntime(truth, **{failure: True})
    with pytest.raises((RuntimeError, PermissionError)):
        run(fixture, runtime)
    directory = root / evaluation.SYNTHETIC_DIRECTORY / "b0/parent"
    assert list(directory.glob("groups/*/STARTED.json"))
    assert list(directory.glob("groups/*/draw-0.json"))
    assert not (directory / "COMPLETE.json").exists()
    retry = SyntheticRuntime(truth)
    with pytest.raises(evaluation.IncompleteEvaluation):
        run(fixture, retry)
    assert retry.calls == 0


def test_complete_cache_raw_tamper_is_rejected_and_not_replaced(fixture):
    root, _, truth = fixture
    run(fixture, SyntheticRuntime(truth))
    path = next((root / evaluation.SYNTHETIC_DIRECTORY / "b0/parent").glob("groups/*/draw-0.json"))
    row = evaluation._read(path)
    row["seed"] += 1
    path.write_text(json.dumps(row))
    before = path.read_bytes()
    retry = SyntheticRuntime(truth)
    with pytest.raises(PermissionError, match="identity differs"):
        run(fixture, retry)
    assert retry.calls == 0 and path.read_bytes() == before


def test_durable_reservation_blocks_reopened_or_manually_rearmed_partial(fixture):
    root, spec, truth = fixture
    with pytest.raises(RuntimeError):
        run(fixture, SyntheticRuntime(truth, partial=True))
    directory = root / evaluation.SYNTHETIC_DIRECTORY / "b0/parent"
    identity = evaluation._read(directory / "IDENTITY.json")
    public, tasks = evaluation._inputs(root, spec["evaluation"])
    with evaluation.EvaluationLedger(root, identity, spec["evaluation"], public, tasks) as ledger:
        item = ledger.schedule[0]
        ledger.pending = copy.deepcopy(item)
        ledger._active_reservation = evaluation.digest(ledger._start(item))
        with pytest.raises(PermissionError, match="UNKNOWN/partial"):
            ledger.authorize_generation(public[item["qid"]]["model_input"], item["seeds"])
    assert len(list(directory.glob("groups/*/GENERATING.json"))) == 1


def test_summary_recipe_cannot_change_after_registration(fixture):
    _, spec, truth = fixture
    spec["evaluation"]["descriptive_summary"]["bootstrap_replicates"] = 500
    runtime = SyntheticRuntime(truth)
    with pytest.raises(PermissionError, match="frozen evaluation summary"):
        run(fixture, runtime)
    assert runtime.calls == 0
    with pytest.raises(PermissionError, match="summary seed is frozen"):
        evaluation.summarize_reports([], master_seed=42)


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "missing",
        "drop_last",
        "duplicate",
        "unknown",
        "unreferenced_process",
        "split_attempt",
        "wrong_second_seed",
        "wrong_parent",
    ],
)
def test_terminal_native_call_counter_rejects_missing_extra_and_unknown(fixture, mutation):
    """Private consumer CPU fixture; never an actual runtime/admission claim."""
    root, spec, truth = fixture
    report = run(fixture, SyntheticRuntime(truth))
    public, tasks = evaluation._inputs(root, spec["evaluation"])
    with evaluation.EvaluationLedger(
        root, report["identity"], spec["evaluation"], public, tasks
    ) as ledger:
        groups = ledger.completed()
        task_id = "SYNTHETIC-evaluation"
        rows = [
            dict(at="SYNTHETIC", task_id=task_id, parent="b0", **copy.deepcopy(event))
            for group in groups
            for event in group["expected_accounting"]["events"]
        ]
        if mutation == "drop_last":
            rows.pop()
        elif mutation == "duplicate":
            rows.append(copy.deepcopy(rows[0]))
        elif mutation == "unknown":
            rows.append(
                dict(task_id=task_id, parent="b0", kind="unknown_calls", count=1, metadata={})
            )
        elif mutation == "split_attempt":
            rows[0]["count"] = 1
            rows.append(copy.deepcopy(rows[0]))
        elif mutation == "wrong_second_seed":
            rows[2]["metadata"]["group_seed"] += 1
        elif mutation == "wrong_parent":
            rows[0]["parent"] = "b2"
        if mutation != "missing":
            save(
                root,
                (ledger.directory / "accounting" / f"{task_id}.jsonl").relative_to(root).as_posix(),
                rows,
                True,
            )
        if mutation == "unreferenced_process":
            save(
                root,
                (ledger.directory / "accounting/UNKNOWN-process.jsonl")
                .relative_to(root)
                .as_posix(),
                [dict(rows[0], task_id="UNKNOWN-process")],
                True,
            )
        if mutation is not None:
            with pytest.raises(PermissionError):
                evaluation._accounting_evidence(ledger, groups)
        else:
            result = evaluation._accounting_evidence(ledger, groups)
            assert result["actual_recorded_calls"] == 192 and result["process_count"] == 1
            assert len(result["inputs"]) == 1
    assert report["status"] == "SYNTHETIC_ONLY"


def test_endpoint_or_panel_changes_fail_before_generation(fixture):
    _, spec, truth = fixture
    runtime = SyntheticRuntime(truth)
    with pytest.raises(PermissionError, match="Unregistered"):
        run(fixture, runtime, endpoint="b0/U11/RECOVERY/step12")
    spec["evaluation"]["inputs"]["public"]["path"] = "DATA/TEST_ID/MODEL_INPUTS.jsonl"
    with pytest.raises(PermissionError, match="Only DEV_EVAL"):
        run(fixture, runtime)
    assert runtime.calls == 0


def test_ledger_lock_is_live_and_wrong_pending_input_cannot_generate(fixture):
    root, spec, truth = fixture
    report = run(fixture, SyntheticRuntime(truth))
    public, tasks = evaluation._inputs(root, spec["evaluation"])

    def ledger():
        return evaluation.EvaluationLedger(
            root, report["identity"], spec["evaluation"], public, tasks
        )

    with ledger() as active:
        with pytest.raises(BlockingIOError), ledger():
            pass
        with pytest.raises(PermissionError, match="No unconsumed"):
            active.authorize_generation({}, [1, 2])
    with pytest.raises(PermissionError, match="Live evaluation lock"):
        active._locked()


def test_symlinked_endpoint_is_rejected(fixture, tmp_path):
    root, _, truth = fixture
    target = root / evaluation.SYNTHETIC_DIRECTORY / "b0"
    target.mkdir(parents=True)
    other = root / "outside"
    other.mkdir()
    (target / "parent").symlink_to(other, target_is_directory=True)
    with pytest.raises(PermissionError, match="Symlink"):
        run(fixture, SyntheticRuntime(truth))
    assert list(other.iterdir()) == []


def test_full_summary_pairs_all_roots_and_preserves_unknowns(fixture):
    _, spec, truth = fixture
    base = run(fixture, SyntheticRuntime(truth))
    reports = []
    for endpoint in spec["evaluation"]["endpoints"]:
        value = copy.deepcopy(base)
        value["endpoint"] = endpoint
        seeds = {
            r["qid"]: r["seeds"] for r in spec["evaluation"]["seeds_by_parent"][endpoint["parent"]]
        }
        for row in value["records"]:
            row["seed"] = seeds[row["qid"]][row["draw"]]
            if "/U11/" in endpoint["endpoint_id"]:
                row["score"]["A"] = 0
        reports.append(value)
    summary = evaluation.summarize_reports(reports)
    assert summary["status"] == "SYNTHETIC_ONLY" and summary["responses"] == 2304
    change = summary["differences_from_parent"]["b0/U11/step12"]
    assert change["A"]["difference"] == -0.5 and change["A"]["paired_denominator"] == 192
    assert change["A"]["interval"]["lower"] == change["A"]["interval"]["upper"] == -0.5
    assert change["S"]["paired_denominator"] == change["S"]["excluded_unknown_pairs"] == 96
    assert not change["A"]["interval"]["training_repeat_variance_estimated"]
    assert summary["significance_gate"] is None
    with pytest.raises(PermissionError, match="All 12"):
        evaluation.summarize_reports(reports[:-1])
    reports[-1]["records"][0]["seed"] += 1
    with pytest.raises(PermissionError, match="seeds differ"):
        evaluation.summarize_reports(reports)


def test_recovery_lane_and_missing_commit_are_not_endpoint_sources(tmp_path):
    root = tmp_path.resolve()
    relative = "runs/b0/r0/U11/RUN_IDENTITY.json"
    identity = dict(
        phase="PILOT_V1",
        lane="RECOVERY",
        max_steps=12,
        run_id="SYNTHETIC/PILOT/b0/r0/U11",
        parent="b0",
    )
    ref = save(root, f"{evaluation.REGISTRATION_DIRECTORY}/{relative}", identity)
    ref["path"] = relative
    spec = dict(
        phase="PILOT_V1",
        paths=[
            dict(
                run_id=identity["run_id"], run_directory="runs/b0/r0/U11", transaction_identity=ref
            )
        ],
    )
    endpoint = dict(kind="PILOT_ENDPOINT", run_id=identity["run_id"], parent="b0")
    with pytest.raises(PermissionError, match="PRIMARY"), evaluation._source(root, spec, endpoint):
        pass
    identity.update(
        lane="PRIMARY",
        experiment_id="SYNTHETIC",
        phase="PILOT_V1",
        arm="U11",
        repeat=0,
        source_sha256="a" * 64,
        data_sha256="b" * 64,
        config_sha256="c" * 64,
        reference_sha256="d" * 64,
    )
    ref = save(root, f"{evaluation.REGISTRATION_DIRECTORY}/{relative}", identity)
    ref["path"] = relative
    spec["paths"][0]["transaction_identity"] = ref
    with (
        pytest.raises(PermissionError, match="All published primary steps"),
        evaluation._source(root, spec, endpoint),
    ):
        pass


def test_production_has_no_runtime_override_and_requires_admission(fixture, monkeypatch):
    root, _, truth = fixture
    with pytest.raises(TypeError):
        evaluation.run_endpoint(root, runtime=SyntheticRuntime(truth))

    def reject(*args, **kwargs):
        raise PermissionError("SYNTHETIC admission is not scientific authorization")

    monkeypatch.setattr(evaluation, "_admitted", reject)
    # Import-only dependency stubs do not authorize/load any model.
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "sr_f6.access", SimpleNamespace(open_evaluation_access=None))
    with pytest.raises(PermissionError, match="SYNTHETIC admission"):
        evaluation.run_endpoint(
            root,
            endpoint_id="b0/parent",
            task_id="fixture",
            phase_identity_sha256="a" * 64,
            pilot_spec_sha256="b" * 64,
            admission_ref=dict(path="fake", sha256="c" * 64),
        )
