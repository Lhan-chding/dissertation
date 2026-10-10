"""Bounded source repair retains prior registration, baseline and exact raw answers."""

import json
import sys

import pytest

from sr_f12.orchestration import TEACHER_QOS, Controller, file_hash, save_json
from sr_f12.protocol import AMENDMENT_ID, object_hash
from sr_f12.source_revision import prepare_technical_repair, resolve_source_revision


class FakeSlurm:
    def __init__(self, command):
        self.command = command
        self.jobs = {}
        self.terminals = {
            "825": dict(
                job_id="825",
                state="FAILED",
                exit_code="1:0",
                user="alice",
                account="rose",
                qos=TEACHER_QOS,
                name="srf12-TECHNICAL",
            ),
            "824": dict(
                job_id="824",
                state="FAILED",
                exit_code="2:0",
                user="alice",
                account="rose",
                qos="cpu",
                name="srf12-controller",
            ),
        }

    def queue(self, user):
        return self.jobs

    def terminal(self, job):
        return self.terminals.get(job)

    def show(self, job):
        assert job == "824"
        return dict(Command=str(self.command), UserId="alice(1)")


@pytest.fixture
def repair(tmp_path, monkeypatch):
    import sr_f12.source_revision as revision

    root = tmp_path / "run"
    root.mkdir()
    old, new = root / "code", root / "code_repair"
    for code in (old, new):
        (code / "src/sr_f12").mkdir(parents=True)
        (code / "src/sr_f12/runtime.py").write_text("unchanged generation")
        (code / "src/sr_f12/training.py").write_text("def objective(): return 1\n")
        (code / "src/sr_f12/protocol.py").write_text("unchanged science")
    (old / "src/sr_f12/runner.py").write_text("old implementation")
    (new / "src/sr_f12/runner.py").write_text("fixed technical selection")

    def manifest(code, commit):
        data = dict(
            git_commit=commit,
            files={str(p.relative_to(code)): file_hash(p) for p in code.rglob("*.py")},
        )
        path = code / "SOURCE.json"
        save_json(path, data)
        return path

    old_manifest = manifest(old, "a" * 40)
    new_manifest = manifest(new, "b" * 40)
    save_json(root / "config/SR_F1_2.json", {"scientific": "unchanged"})
    save_json(
        root / "AMENDMENT.json",
        dict(
            amendment_id=AMENDMENT_ID,
            source_commit="a" * 40,
            config_sha256=object_hash({"scientific": "unchanged"}),
        ),
    )
    save_json(
        root / "orchestration/CONFIG.json",
        dict(code_root=str(old), source_manifest=str(old_manifest), source_commit="a" * 40),
    )
    command = root / "cpu.sbatch"
    command.write_text("cpu controller")
    slurm = FakeSlurm(command)
    task = root / "orchestration/tasks/TECHNICAL"
    save_json(task / "STATE.json", dict(task_id="TECHNICAL", job_id="825", status="FAILED"))
    save_json(task / "INTENT.json", dict(job_name="srf12-TECHNICAL"))
    save_json(task / "TERMINAL.json", slurm.terminals["825"])
    for i in range(3):
        directory = root / "orchestration/tasks" / f"BASELINE_{i}"
        save_json(
            directory / "STATE.json",
            dict(task_id=f"BASELINE_{i}", job_id=str(826 + i), status="RUNNING"),
        )
        save_json(
            directory / "INTENT.json",
            dict(source_sha256=object_hash(json.loads(old_manifest.read_text()))),
        )
    raw = root / "technical/preflight/rollouts"
    raw.mkdir(parents=True)
    for i in range(4):
        (raw / f"{i}.json").write_text(json.dumps({"original_tokens": list(range(i, i + 8))}))
    (root / "technical/failure.log").write_text("failed batch-single gate")
    raw_hashes = {p.name: file_hash(p) for p in raw.glob("*.json")}
    monkeypatch.setattr(
        revision, "_raw32", lambda root: dict(files=raw_hashes, records=32, policy_hash="f" * 64)
    )
    monkeypatch.setattr(revision, "generation_identity", lambda code: {"generation": "same"})
    args = dict(
        revision_id="r0001",
        code_root=new,
        source_manifest=new_manifest,
        source_commit="b" * 40,
        controller_job_id="824",
        slurm=slurm,
        user="alice",
    )
    return root, args, old_manifest, raw_hashes


def test_repair_preserves_original_and_32_answers_and_baseline(repair):
    root, args, _, hashes = repair
    amendment = (root / "AMENDMENT.json").read_bytes()
    config = (root / "orchestration/CONFIG.json").read_bytes()
    baseline = {
        (p.parent.name, p.name): p.read_bytes()
        for p in (root / "orchestration/tasks").glob("BASELINE_*/*.json")
    }
    result = prepare_technical_repair(root, **args)
    assert result["reused_records"] == 32
    assert (root / "AMENDMENT.json").read_bytes() == amendment
    assert (root / "orchestration/CONFIG.json").read_bytes() == config
    assert not (root / "orchestration/tasks/TECHNICAL").exists()
    assert (root / "technical_incidents/source_repair_r0001/task/TERMINAL.json").exists()
    assert (
        root / "technical_incidents/source_repair_r0001/technical/failure.log"
    ).read_text() == "failed batch-single gate"
    assert {
        p.name: file_hash(p) for p in (root / "technical/preflight/rollouts").glob("*.json")
    } == hashes
    assert {
        (p.parent.name, p.name): p.read_bytes()
        for p in (root / "orchestration/tasks").glob("BASELINE_*/*.json")
    } == baseline
    source = resolve_source_revision(root, "b" * 40)
    assert source["registration_source_commit"] == "a" * 40
    assert source["revision_id"] == "r0001"
    assert prepare_technical_repair(root, **args) == result
    with pytest.raises(PermissionError):
        resolve_source_revision(root, "a" * 40)


def test_running_old_job_cannot_be_repaired(repair):
    root, args, _, _ = repair
    args["slurm"].jobs["825"] = {}
    with pytest.raises(PermissionError, match="not both terminal"):
        prepare_technical_repair(root, **args)
    assert (root / "technical/failure.log").exists()


def test_runtime_generation_source_must_not_change(repair):
    root, args, _, _ = repair
    path = args["code_root"] / "src/sr_f12/runtime.py"
    path.write_text("changed sampling")
    manifest = json.loads(args["source_manifest"].read_text())
    manifest["files"]["src/sr_f12/runtime.py"] = file_hash(path)
    save_json(args["source_manifest"], manifest)
    with pytest.raises(PermissionError, match="unapproved executable"):
        prepare_technical_repair(root, **args)


def test_scientific_config_cannot_change(repair):
    root, args, _, _ = repair
    save_json(root / "config/SR_F1_2.json", {"scientific": "altered"})
    with pytest.raises(PermissionError, match="configuration was changed"):
        prepare_technical_repair(root, **args)


def test_actual_pilot_checkpoint_excludes_zero_update_repair(repair):
    root, args, _, _ = repair
    path = root / "technical/SRF1_2_TECHNICAL_R0/checkpoints/step-01.pt"
    path.parent.mkdir(parents=True)
    path.write_text("has update")
    with pytest.raises(PermissionError, match="zero-update preflight"):
        prepare_technical_repair(root, **args)


def test_revision_cannot_target_live_baseline_source(repair):
    root, args, old_manifest, _ = repair
    args.update(code_root=root / "code", source_manifest=old_manifest)
    with pytest.raises(PermissionError, match="separate deployment"):
        prepare_technical_repair(root, **args)


def test_interrupted_move_resumes_without_reexecuting_old_job(repair, monkeypatch):
    import sr_f12.source_revision as revision

    root, args, _, hashes = repair
    real_rename = revision.os.rename
    moves = []

    def interrupt(source, destination):
        if len(moves) == 1:
            raise OSError("injected maintenance interruption")
        moves.append(str(source))
        return real_rename(source, destination)

    monkeypatch.setattr(revision.os, "rename", interrupt)
    with pytest.raises(OSError, match="injected"):
        prepare_technical_repair(root, **args)
    with pytest.raises(PermissionError, match="incomplete"):
        resolve_source_revision(root)
    monkeypatch.setattr(revision.os, "rename", real_rename)
    prepare_technical_repair(root, **args)
    assert {
        p.name: file_hash(p) for p in (root / "technical/preflight/rollouts").glob("*.json")
    } == hashes
    assert resolve_source_revision(root)["source_commit"] == "b" * 40


def test_new_controller_keeps_original_config_and_existing_baseline_registry(repair, monkeypatch):
    root, args, _, _ = repair
    prepare_technical_repair(root, **args)
    original = (root / "orchestration/CONFIG.json").read_bytes()
    model = root / "model"
    model.mkdir()
    controller = Controller(
        root,
        args["code_root"],
        model,
        args["source_manifest"],
        "b" * 40,
        python=sys.executable,
        slurm=args["slurm"],
        user="alice",
    )
    assert (root / "orchestration/CONFIG.json").read_bytes() == original
    assert (root / "orchestration/revisions/r0001/CONFIG.json").exists()
    assert len(controller.attempts()) == 3
    assert all(row["task_id"].startswith("BASELINE_") for row in controller.attempts())


def test_source_chain_authorization_cannot_be_altered(repair):
    root, args, _, _ = repair
    prepare_technical_repair(root, **args)
    path = root / "source_revisions/r0001/AUTHORIZATION.json"
    authorization = json.loads(path.read_text())
    authorization["scope"] = "alter_scientific_recipe"
    save_json(path, authorization)
    with pytest.raises(PermissionError, match="Broken or out-of-scope"):
        resolve_source_revision(root)


def test_objective_change_is_not_a_microbatch_repair(repair):
    root, args, _, _ = repair
    path = args["code_root"] / "src/sr_f12/training.py"
    path.write_text("def objective(): return 2\n")
    manifest = json.loads(args["source_manifest"].read_text())
    manifest["files"]["src/sr_f12/training.py"] = file_hash(path)
    save_json(args["source_manifest"], manifest)
    with pytest.raises(PermissionError, match="numerical objective"):
        prepare_technical_repair(root, **args)


def test_raw32_checks_actual_schedule_input_and_token_identity(tmp_path, monkeypatch):
    import sr_f1.data as data
    import sr_f12.runner as runner
    from sr_f12.source_revision import _raw32

    schedule = [dict(step=1, slot=i, qid=str(i), rollout_seeds=list(range(8))) for i in range(4)]
    inputs = {
        str(i): dict(qid=str(i), text="question", image_file="images/x.png") for i in range(4)
    }
    monkeypatch.setattr(runner, "technical_schedule", lambda root: schedule)
    monkeypatch.setattr(data, "load_inputs", lambda root: inputs)
    for slot in schedule:
        qid = slot["qid"]
        records = [
            dict(
                row=inputs[qid],
                qid=qid,
                root=str(tmp_path.resolve()),
                group_row_index=i,
                group_seed=0,
                tokens=[90, 91],
                sampler_logprobs=[-1.0, -2.0],
            )
            for i in range(8)
        ]
        identity = dict(
            run_id="PREFLIGHT",
            step=1,
            slot=slot["slot"],
            qid=qid,
            policy_hash="f" * 64,
            seeds=list(range(8)),
            input_hash=object_hash(inputs[qid]),
        )
        save_json(
            tmp_path / "technical/preflight/rollouts" / f"01-{slot['slot']:02d}.json",
            dict(identity=identity, records=records, sha256=object_hash(records)),
        )
    assert _raw32(tmp_path)["records"] == 32
    path = tmp_path / "technical/preflight/rollouts/01-00.json"
    saved = json.loads(path.read_text())
    saved["records"][0]["group_row_index"] = 3
    saved["sha256"] = object_hash(saved["records"])
    save_json(path, saved)
    with pytest.raises(PermissionError, match="answer metadata"):
        _raw32(tmp_path)


def test_only_exact_side_effect_free_numerical_exception_is_allowed(tmp_path):
    from sr_f12.source_revision import training_identity

    directory = tmp_path / "src/sr_f12"
    directory.mkdir(parents=True)
    source = directory / "training.py"
    source.write_text("def objective(): return 1\n")
    original = training_identity(tmp_path)
    exception = (
        "class MicrobatchNumericalMismatch(RuntimeError):\n"
        '    "A measured numerical gate failure."\n'
    )
    source.write_text(exception + "def objective(): return 1\n")
    assert training_identity(tmp_path) == original
    source.write_text(
        exception + "    def __init__(self): print('side effect')\n" + "def objective(): return 1\n"
    )
    with pytest.raises(PermissionError, match="unexpected behavior"):
        training_identity(tmp_path)
