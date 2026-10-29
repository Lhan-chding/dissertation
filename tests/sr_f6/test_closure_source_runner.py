"""Synthetic registration and readonly boundary checks; never actual admission."""

import copy
import json
import os
from types import SimpleNamespace

import pytest

from sr_f6_closure import runner, source


@pytest.mark.parametrize("path", ["../old/result.json", "../other/file", "new/../../old"])
def test_write_escape_is_denied(tmp_path, path):
    root = tmp_path / "new"
    root.mkdir()
    with pytest.raises(PermissionError):
        runner.guard_write("open", (str(root / path), "w", os.O_WRONLY), (root,))


def test_original_read_allowed_new_write_allowed_external_write_denied(tmp_path):
    root = tmp_path / "new"
    root.mkdir()
    runner.guard_write("open", (str(tmp_path / "old"), "r", 0), (root,))
    runner.guard_write("open", (str(root / "result"), "w", os.O_WRONLY), (root,))
    with pytest.raises(PermissionError):
        runner.guard_write("open", (str(tmp_path / "old"), "w", os.O_WRONLY), (root,))
    (root / "alias").symlink_to(tmp_path / "old")
    with pytest.raises(PermissionError):
        runner.guard_write("open", (str(root / "alias"), "w", os.O_WRONLY), (root,))


@pytest.mark.parametrize(
    "event,args",
    [
        ("os.remove", ("file", 42)),
        ("os.mkdir", ("dir", 0o700, 42)),
        ("os.rename", ("a", "b", -1, 42)),
        ("os.chmod", ("a", 0o600, 42)),
        ("subprocess.Popen", ("echo",)),
        ("os.system", ("echo",)),
        ("os.fork", ()),
        ("socket.connect", ("host",)),
    ],
)
def test_descriptor_mutations_and_external_execution_denied(tmp_path, event, args):
    with pytest.raises(PermissionError):
        runner.guard_write(event, args, (tmp_path,))


@pytest.fixture
def registered(tmp_path, monkeypatch):
    original = source.original
    root = tmp_path / "closure"
    code = tmp_path / "code"
    package = code / "src/sr_f6_closure"
    scripts = code / "scripts/sr_f6_closure"
    package.mkdir(parents=True)
    scripts.mkdir(parents=True)
    for name in (
        "__init__.py",
        "source.py",
        "runner.py",
        "projection.py",
        "admission_projection.py",
    ):
        (package / name).write_text("# synthetic source\n")
    for name in ("bootstrap.py", "verify.sbatch"):
        (scripts / name).write_text("# synthetic launcher\n")
    monkeypatch.setattr(original, "PROJECT", tmp_path)
    monkeypatch.setattr(source, "ROOT", root)
    monkeypatch.setattr(source, "deployment", lambda revision: code)
    monkeypatch.setattr(source, "module_origins", lambda *args: None)
    monkeypatch.setattr(source, "original_context", lambda: {"consumer": {"revision": "synthetic"}})
    monkeypatch.setattr(source, "contract", lambda: {"scope": "SYNTHETIC_TEST_ONLY"})
    source.register("a" * 40)
    return root, code


def test_registration_is_immutable_and_never_permission(registered):
    root, code = registered
    identity = source.verify_registered()
    assert identity["repair_verifier"]["deployment"] == str(code)
    assert json.loads((root / "REGISTRATION.json").read_text())["training_allowed"] is False
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(PermissionError, match="Preserve"):
        source.register("a" * 40)
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}


@pytest.mark.parametrize("mutation", ["bytes", "member", "permission", "contract", "origin"])
def test_registered_changes_rejected(registered, monkeypatch, mutation):
    root, code = registered
    if mutation == "bytes":
        (code / "src/sr_f6_closure/runner.py").write_text("# changed\n")
    elif mutation == "member":
        (code / "src/sr_f6_closure/extra.py").write_text("# extra\n")
    elif mutation == "permission":
        path = root / "REGISTRATION.json"
        value = json.loads(path.read_text())
        value["training_allowed"] = True
        path.write_text(json.dumps(value))
    elif mutation == "contract":
        (root / "CONTRACT.json").write_text("{}")
    else:
        monkeypatch.setattr(source, "original_context", lambda: {"consumer": {"revision": "other"}})
    with pytest.raises(PermissionError):
        source.verify_registered()


def test_history_has_only_actual_unknown_and_exact_three_reader_errors(monkeypatch):
    original = source.original
    reads = {
        original.ROOT / "ADMISSION/ASSESSMENT.json": dict(
            status="UNKNOWN",
            gate_status="NOT_SIGNED",
            **source.PERMISSIONS,
            checks={"numerical": dict.fromkeys(("b0", "b1", "b2"), "INCOMPLETE")},
        ),
        source.OPS / "T03_VERIFY_DIAGNOSTIC.json": dict(
            report=dict(
                status="TECHNICAL_INVALID",
                errors=[
                    dict(task_id=f"LH-SRF6-T03-233723-{p}", error="KeyError: 'row_id'")
                    for p in ("b0", "b1", "b2")
                ],
            )
        ),
    }
    monkeypatch.setattr(
        original, "_reference", lambda path: dict(path=str(path), sha256=source.HISTORY[path])
    )
    monkeypatch.setattr(original, "_read", lambda path: copy.deepcopy(reads[path]))
    assert len(source.history()) == 4
    reads[original.ROOT / "ADMISSION/ASSESSMENT.json"]["status"] = "PASS"
    with pytest.raises(PermissionError):
        source.history()


def test_module_origins_bind_main_spec_and_path(tmp_path, monkeypatch):
    code = tmp_path / "code"
    package = code / "src/sr_f6_closure"
    package.mkdir(parents=True)
    module_file = package / "source.py"
    module_file.touch()
    bootstrap = code / "scripts/sr_f6_closure/bootstrap.py"
    bootstrap.parent.mkdir(parents=True)
    bootstrap.touch()
    fake = SimpleNamespace(
        __name__="sr_f6_closure",
        __file__=str(module_file),
        __spec__=SimpleNamespace(name="sr_f6_closure", origin=str(module_file)),
        __path__=[str(package)],
    )
    monkeypatch.setattr(source.original, "PROJECT", tmp_path)
    monkeypatch.setattr(
        source.sys,
        "modules",
        {"__main__": SimpleNamespace(__file__=str(bootstrap)), "sr_f6_closure": fake},
    )
    monkeypatch.setattr(
        source.sys,
        "path",
        [
            str(source.original.PRODUCER_DEPLOYMENT / "src"),
            str(source.ORIGINAL_DEPLOYMENT / "src"),
            str(code / "src"),
        ],
    )
    files = [dict(path=str(module_file))]
    source.module_origins(code, files)
    fake.__spec__.origin = str(tmp_path / "shadow.py")
    with pytest.raises(PermissionError):
        source.module_origins(code, files)


@pytest.fixture
def runner_fixture(tmp_path, monkeypatch):
    root = tmp_path / "closure"
    cache = tmp_path / "sr_f6_operations_20261027/CLOSURE_REPAIR_V1_20261029/cache_123"
    (cache / "tmp").mkdir(parents=True)
    monkeypatch.setattr(source, "PROJECT", tmp_path)
    monkeypatch.setattr(source.original, "PROJECT", tmp_path)
    monkeypatch.setattr(source, "ROOT", root)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_JOB_NAME", "LH-SRF6-Closure-Repair")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TMPDIR", str(cache / "tmp"))
    # No persistent audit hook in the test interpreter; guard is tested above.
    monkeypatch.setattr(runner.sys, "addaudithook", lambda hook: None)
    identity = {"scope": "SYNTHETIC_TEST_ONLY"}

    def register(revision):
        root.mkdir(exist_ok=False)
        return copy.deepcopy(identity)

    monkeypatch.setattr(source, "register", register)
    monkeypatch.setattr(source, "verify_registered", lambda: copy.deepcopy(identity))
    monkeypatch.setattr(source, "history", lambda: ["SYNTHETIC_UNCHANGED_HISTORY"])
    return root


@pytest.mark.parametrize("status", ["CONFIRMATION_GATES_PASS", "FAIL", "INCONCLUSIVE", "UNKNOWN"])
def test_runner_seals_distinct_result_once_without_permission(runner_fixture, monkeypatch, status):
    root = runner_fixture
    numerical = dict(status=status, scope="SYNTHETIC_TEST_ONLY")
    calls = []

    def evaluate(old_root):
        calls.append(old_root)
        return dict(
            assessment=dict(status=status, numerical_sha256=runner.digest(numerical)),
            numerical=numerical,
        )

    monkeypatch.setattr(runner.admission_projection, "evaluate", evaluate)
    receipt = runner.execute("a" * 40)
    assert len(calls) == 1
    assert receipt["status"] == status
    assert all(receipt[k] is False for k in source.PERMISSIONS)
    report = json.loads((root / "ASSESSMENT.json").read_text())
    assert report["old_admission_status"] == "UNKNOWN_NOT_MODIFIED"
    assert report["optimizer_updates"] == report["new_model_calls"] == 0
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(FileExistsError):
        runner.execute("a" * 40)
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}


def test_runner_preserves_partial_failure_no_success_receipt(runner_fixture, monkeypatch):
    def fail(old_root):
        raise PermissionError("SYNTHETIC_SOURCE_DRIFT")

    monkeypatch.setattr(runner.admission_projection, "evaluate", fail)
    with pytest.raises(PermissionError, match="SYNTHETIC_SOURCE_DRIFT"):
        runner.execute("a" * 40)
    assert {p.name for p in runner_fixture.iterdir()} == {"ATTEMPT.json", "FAILURE.json"}
