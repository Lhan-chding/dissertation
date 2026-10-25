"""Operations-only bridge regressions; no scheduler/model calls or suite reruns."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/sr_f5/advance_recipe_v2.py"
SPEC = importlib.util.spec_from_file_location("advance_recipe_v2_test", SCRIPT)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)
REVISION = "a" * 40


@pytest.fixture
def example(tmp_path, monkeypatch):
    root, v1, code = (tmp_path / name for name in ("v2", "v1", "science"))
    for path in (root, v1, code):
        path.mkdir()
    launcher = tmp_path / "operations.py"
    launcher.write_text("# operations-only fixture\n")
    for key, path in (("PROJECT", tmp_path), ("ROOT", root), ("V1", v1)):
        monkeypatch.setattr(bridge, key, path)
    config = {"numerics": {"recipe_version": 2}}
    registration = dict(
        identity=dict(
            new_source_revision=REVISION,
            new_root=str(root),
            recipe_version=2,
            new_config_digest=bridge.digest(config),
            numerics=config["numerics"],
        )
    )
    bridge.write(v1 / "CONTINUATION/RECIPE_V2.json", registration)
    bridge.write(v1 / "CONTINUATION/RECIPE_V2_CONFIG.json", config)
    bridge.write(root / "CONFIG.json", config)
    bridge.write(
        root / "CPU_READY.json", dict(status="CPU_READY_FOR_G2_PREFLIGHT", code_revision=REVISION)
    )
    bridge.write(
        root / "CPU_TESTS.json",
        dict(
            status="PASS",
            code_revision=REVISION,
            returncode=0,
            model_calls=0,
            test_paths=["tests/sr_f5"],
        ),
    )
    bridge.write(root / "G2_SPEC_FROZEN.json", dict(code_revision=REVISION))
    bridge.write(root / "ENVIRONMENT_LOCK.json", dict(code_revision=REVISION))
    bridge.write(root / "DATA/HISTORICAL_WORLD_AUDIT.json", dict(status="PASS"))
    bridge.write(root / "GATES/G2.json", dict(status="NOT_RUN"))
    for pair in (0, 1):
        job = str(100 + pair)
        bridge.write(
            root / f"G2_SUBMISSION_PAIR{pair}.json",
            dict(
                status="RELEASED_VERIFIED",
                job_id=job,
                pair=pair,
                shards=[2 * pair, 2 * pair + 1],
                expected_answers=2304,
                science_revision=REVISION,
                source_code=str(code),
                registered_recipe=registration,
                allocation=dict(
                    JobId=job,
                    JobName=f"LH-SRF5-G2-P{pair}",
                    UserId="user(1)",
                    Account="rose",
                    QOS=bridge.QOS,
                    ReqTRES="gpu=2",
                    WorkDir=str(code),
                    Command=str(code / "scripts/sr_f5/calibration.sbatch"),
                    StdOut=str(root / "slurm" / f"G2-P{pair}-{job}.log"),
                ),
            ),
        )
    state = SimpleNamespace(
        active=set(),
        terminal="COMPLETED",
        exit_code="0:0",
        capacity=(1, 1),
        gate="PASS",
        aggregates=0,
        freezes=0,
        submits=0,
        frozen_checks=0,
    )

    def aggregate(actual_root):
        assert actual_root == root
        state.aggregates += 1
        gate = dict(status=state.gate, errors=[])
        report = dict(gate=gate, actual_response_count=4608)
        bridge.write(root / "G2/METRICS.json", report)
        bridge.write(root / "GATES/G2.json", dict(gate, at=f"time-{state.aggregates}"))
        return report

    def freeze(actual_root, revision):
        assert actual_root == root and revision == REVISION
        state.freezes += 1
        frozen = dict(
            code_revision=revision, gate_digest=bridge.digest(bridge.read(root / "GATES/G2.json"))
        )
        bridge.write(root / "G3/FROZEN.json", frozen)
        return frozen

    def require_frozen(actual_root):
        assert actual_root == root
        state.frozen_checks += 1
        frozen = bridge.read(root / "G3/FROZEN.json")
        assert frozen["gate_digest"] == bridge.digest(bridge.read(root / "GATES/G2.json"))
        return frozen

    def submit(args, **kwargs):
        assert args.code == code and args.root == root and args.revision == REVISION
        state.submits += 1
        receipt = dict(
            status=bridge.VERIFIED,
            code_revision=REVISION,
            source_code=str(code),
            pool_plan_digest=bridge.digest({"plan": 1}),
            jobs=[dict(pair=0, job_id="200"), dict(pair=1, job_id="201")],
        )
        bridge.write(root / "G3_SUBMISSION_DUAL_GPU.json", receipt)
        return receipt

    deps = SimpleNamespace(
        aggregate=aggregate,
        freeze=freeze,
        require_frozen=require_frozen,
        require_plan=lambda *args: {"plan": 1},
        science_digest=bridge.digest,
        submit=submit,
        capacity=lambda *args: (*state.capacity, {}),
        gpu_count=lambda value: int(value.split("=")[1]),
        now=lambda: "fixture-time",
        slurm=SimpleNamespace(
            user_job_ids=lambda user: state.active,
            terminal=lambda job: (
                None
                if state.terminal is None
                else dict(
                    job_id=job,
                    state=state.terminal,
                    exit_code=state.exit_code,
                    user="user",
                    account="rose",
                    qos=bridge.QOS,
                )
            ),
        ),
    )

    def run():
        return bridge.advance(
            SimpleNamespace(code=code, revision=REVISION), deps=deps, user="user", launcher=launcher
        )

    return SimpleNamespace(root=root, code=code, state=state, deps=deps, run=run)


def test_complete_pair_reuses_cpu_acceptance_and_submits_once(example):
    assert example.run()["status"] == "G3_SUBMITTED_AFTER_REAL_G2_PASS"
    assert (example.state.aggregates, example.state.freezes, example.state.submits) == (1, 1, 1)
    gate = (example.root / "GATES/G2.json").read_bytes()
    assert example.run()["status"] == "G3_ALREADY_SUBMITTED_DO_NOT_DUPLICATE"
    assert (example.state.aggregates, example.state.freezes, example.state.submits) == (1, 1, 1)
    assert (example.root / "GATES/G2.json").read_bytes() == gate


@pytest.mark.parametrize(
    "kind,status",
    [
        ("active", "WAITING_FOR_G2"),
        ("failed", "G2_TECHNICAL_FAILURE_REQUIRES_DIAGNOSIS"),
        ("unknown", "WAITING_FOR_ACCOUNTING"),
    ],
)
def test_active_failed_unknown_jobs_never_aggregate(example, kind, status):
    if kind == "active":
        example.state.active = {"100"}
    elif kind == "failed":
        example.state.terminal = "FAILED"
    else:
        example.state.terminal = None
    assert example.run()["status"] == status
    assert example.state.aggregates == example.state.submits == 0


@pytest.mark.parametrize("state", ["RUNNING", "PENDING", "COMPLETING", "UNRECOGNIZED"])
def test_queue_accounting_race_does_not_fabricate_terminal_failure(example, state):
    example.state.terminal = state
    assert example.run()["status"] == "WAITING_FOR_ACCOUNTING"
    assert example.state.aggregates == 0


def test_absent_pair_waits_but_unknown_submission_requires_diagnosis(example):
    path = example.root / "G2_SUBMISSION_PAIR1.json"
    path.unlink()
    assert example.run()["status"] == "WAITING_FOR_G2_PAIR_RECEIPT"
    bridge.write(path, dict(status="SUBMISSION_UNKNOWN"))
    assert example.run()["status"] == "G2_SUBMISSION_REQUIRES_DIAGNOSIS"
    assert example.state.aggregates == 0


@pytest.mark.parametrize(
    "field,value", [("job_id", "999"), ("shards", [0, 1]), ("source_code", "/other")]
)
def test_pair_identity_mismatch_fails_closed(example, field, value):
    path = example.root / "G2_SUBMISSION_PAIR1.json"
    receipt = bridge.read(path)
    receipt[field] = value
    bridge.write(path, receipt)
    with pytest.raises(PermissionError, match="pair allocation"):
        example.run()
    assert example.state.aggregates == 0


def test_after_freeze_capacity_retry_preserves_gate_digest(example):
    example.state.capacity = (4, 3)
    assert example.run()["status"] == "WAITING_FOR_G3_CAPACITY"
    gate = (example.root / "GATES/G2.json").read_bytes()
    example.state.capacity = (1, 1)
    assert example.run()["status"] == "G3_SUBMITTED_AFTER_REAL_G2_PASS"
    assert example.state.aggregates == example.state.freezes == 1
    assert example.state.frozen_checks == 1
    assert (example.root / "GATES/G2.json").read_bytes() == gate


def test_gate_failure_stops_v2_and_reuses_unchanged_aggregate(example):
    example.state.gate = "FAIL"
    assert example.run()["status"] == "STOP_RECIPE_V2_FEASIBILITY"
    assert example.run()["status"] == "STOP_RECIPE_V2_FEASIBILITY"
    assert example.state.aggregates == 1
    assert example.state.freezes == example.state.submits == 0
    metrics = bridge.read(example.root / "G2/METRICS.json")
    metrics["changed"] = True
    bridge.write(example.root / "G2/METRICS.json", metrics)
    with pytest.raises(PermissionError, match="evidence changed"):
        example.run()


@pytest.mark.parametrize("partial", ["started", "orphan_metrics", "orphan_gate"])
def test_interrupted_aggregation_never_overwrites_evidence(example, partial):
    if partial == "started":
        bridge.write(
            example.root / "CONTINUATION/RECIPE_V2_G2_AGGREGATION.json", dict(status="STARTED")
        )
    elif partial == "orphan_metrics":
        bridge.write(example.root / "G2/METRICS.json", {"partial": True})
    else:
        bridge.write(example.root / "GATES/G2.json", dict(status="PASS"))
    assert "DIAGNOSIS" in example.run()["status"]
    assert example.state.aggregates == 0


def test_partial_g3_submission_short_circuits_before_aggregation(example):
    bridge.write(example.root / "G3_SUBMISSION_DUAL_GPU.json", dict(status="SUBMITTING", jobs=[]))
    assert example.run()["status"] == "G3_SUBMISSION_REQUIRES_DIAGNOSIS_NO_RETRY"
    assert example.state.aggregates == example.state.submits == 0


def test_cpu_acceptance_must_cover_existing_scientific_revision(example):
    path = example.root / "CPU_TESTS.json"
    receipt = bridge.read(path)
    receipt["test_paths"] = ["unrelated-test"]
    bridge.write(path, receipt)
    with pytest.raises(PermissionError, match="CPU acceptance"):
        example.run()
    assert example.state.aggregates == 0
