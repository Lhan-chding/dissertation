"""Synthetic local operations regressions; no server, scheduler or scientific PASS."""

import ast
import hashlib
import importlib.util
import io
import json
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.sr_f6.pilot_v3_closure_operations import contracts
from sr_f6.phases import source_delta


def payload(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def load_builder():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "scripts/sr_f6/pilot_v3_closure_operations/build_prepare_submission.py"
    spec = importlib.util.spec_from_file_location("closure_ops_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def closure(tmp_path, monkeypatch):
    blobs = {}
    prefix = contracts.CLOSURE

    def write(name, value):
        relative = prefix + "/" + name + ".json"
        blobs[relative] = payload(value)
        return dict(
            path=str(contracts.PROJECT / relative),
            sha256=hashlib.sha256(blobs[relative]).hexdigest(),
        )

    manifest = write("SOURCE_MANIFEST", {"scope": "SYNTHETIC_TEST_ONLY"})
    contract = write("CONTRACT", {"scope": "SYNTHETIC_TEST_ONLY"})
    registration_value = dict(
        revision=contracts.REPAIR_REVISION,
        deployment=str(contracts.PROJECT / "sr_f6_closure_code_1879f8c"),
        source_manifest=manifest,
        contract=contract,
        status="BOUND_NOT_AUTHORIZED",
        **dict.fromkeys(contracts.PERMISSIONS, False),
    )
    registration = write("REGISTRATION", registration_value)
    identity = dict(
        original_execution_consumer={"scope": "SYNTHETIC_TEST_ONLY"},
        repair_verifier=dict(
            revision=contracts.REPAIR_REVISION,
            deployment=registration_value["deployment"],
            registration=registration,
            source_manifest=manifest,
            contract=contract,
        ),
    )
    write("ATTEMPT", {"scope": "SYNTHETIC_TEST_ONLY"})
    numerical_value = dict(
        status="PASS",
        scope="SERVER_ACTUAL",
        parents={p: dict(status="PASS") for p in ("b0", "b1", "b2")},
        generated_responses=0,
        optimizer_updates=0,
    )
    numerical = write("NUMERICAL", numerical_value)
    canonical = json.dumps(
        numerical_value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    original = write(
        "ORIGINAL_RULE_ASSESSMENT",
        dict(
            status="CONFIRMATION_GATES_PASS", numerical_sha256=hashlib.sha256(canonical).hexdigest()
        ),
    )
    assessment = write(
        "ASSESSMENT",
        dict(
            status="CONFIRMATION_GATES_PASS",
            identity=identity,
            original_rule_assessment=original,
            numerical=numerical,
            old_admission_status="UNKNOWN_NOT_MODIFIED",
            new_model_calls=0,
            optimizer_updates=0,
            **dict.fromkeys(contracts.PERMISSIONS, False),
        ),
    )
    receipt_value = dict(
        schema="srf6-independent-technical-closure-v1",
        status="CONFIRMATION_GATES_PASS",
        gate_status="CORRECTED_CONFIRMATION_FACTS_SEALED",
        identity=identity,
        assessment=assessment,
        **dict.fromkeys(contracts.PERMISSIONS, False),
    )
    receipt = write("RECEIPT", receipt_value)
    blobs[contracts.CLOSURE_OPS + "/CPU_VERIFY_ATTEMPT.json"] = payload(
        {"scope": "SYNTHETIC_TEST_ONLY"}
    )
    blobs[contracts.CLOSURE_OPS + "/CPU_VERIFY_SUBMISSION.json"] = payload(
        dict(status="SUBMITTED", job_id="233920", requested_gpus=0)
    )
    # Only test pins are changed; production constants pin the actual originals.
    monkeypatch.setattr(contracts, "REGISTRATION_PIN", registration["sha256"])
    monkeypatch.setattr(contracts, "RECEIPT_PIN", receipt["sha256"])
    return blobs


@pytest.fixture
def builder(tmp_path, monkeypatch, closure):
    repo = Path(__file__).resolve().parents[2]
    module = load_builder()
    base = tmp_path / "pilot_v3_closure_20261029"
    base.mkdir()
    revision = "c" * 40
    code = str(contracts.PROJECT / "sr_f6_code_ccccccc")
    before = {"files": [dict(path="src/x.py", bytes=1, sha256="a" * 64)]}
    after = {
        "files": [
            dict(path="src/sr_f6/historical_closure.py", bytes=1, sha256="d" * 64),
            dict(path="src/x.py", bytes=2, sha256="b" * 64),
        ]
    }
    for path in (repo / "scripts/sr_f6/pilot_v3_closure_operations").iterdir():
        if path.suffix in {".py", ".sh", ".sbatch"}:
            raw = path.read_bytes()
            after["files"].append(
                dict(
                    path=str(path.relative_to(repo)),
                    bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(),
                )
            )
    after["files"].sort(key=lambda row: row["path"])
    reasons = {
        name: "Synthetic local request test; no actual source or authority"
        for name in [row["path"] for row in after["files"]]
    }
    values = {
        "CODE_REVISION.json": dict(revision=revision, deployment=code),
        "ORIGINAL_SOURCE_MANIFEST.json": before,
        "EXPECTED_SOURCE_MANIFEST.json": after,
        "SOURCE_CHANGE_REASONS.json": reasons,
        "EXPECTED_DELTA.json": source_delta(before, after, reasons=reasons),
        "PREPARE_REFERENCE_INPUTS.json": dict(
            actual_source_revision=revision,
            source_deployment=code,
            root=str(contracts.ROOT),
            origin=dict(
                phase="CENSUS_S0",
                deployment=str(contracts.PROJECT / "sr_f6_code_017b60a"),
                identity=dict(path="IDENTITY.json", sha256="a" * 64),
                source_manifest=dict(path="SOURCE_MANIFEST.json", sha256="a" * 64),
            ),
            upstream=[
                dict(
                    producer_phase="CENSUS_S0",
                    identity_sha256="a" * 64,
                    evidence=[dict(path="PREFLIGHT/CENSUS_READY.json", sha256="a" * 64)],
                )
            ],
            parent_references={
                p: dict(path=f"runtime_identity/CENSUS-{p}-0-of-2.json", sha256="a" * 64)
                for p in ("b0", "b1", "b2")
            },
        ),
    }
    for name, value in values.items():
        (base / name).write_bytes(payload(value))
    upstream = base.parent / "closure_repair_20261029"
    for name, raw in closure.items():
        p = upstream / "delivery" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    delivery = dict(
        status="CONFIRMATION_GATES_PASS",
        job=["233920", "COMPLETED", "0:0"],
        source_revision=contracts.REPAIR_REVISION,
        files={
            name: dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            for name, raw in closure.items()
        },
    )
    (upstream / "DELIVERY.json").write_bytes(payload(delivery))
    monkeypatch.setattr(module, "base", base)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert argv[:2] in (["bash", "-n"], ["git", "ls-files"], ["git", "diff"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: "e" * 40)
    source_calls = []
    monkeypatch.setattr(
        module,
        "verify_published_producer",
        lambda revision, manifest: source_calls.append((revision, manifest)),
    )
    monkeypatch.setattr(module.sys, "argv", ["builder"])
    return module, base, calls, source_calls


def test_validate_only_has_no_activation_or_remote_commands(builder, monkeypatch, capsys):
    module, base, calls, source_calls = builder
    monkeypatch.setattr(module.sys, "argv", ["builder", "--validate-only"])
    module.main()
    assert json.loads(capsys.readouterr().out)["status"] == "LOCAL_TEMPLATES_VALID_NOT_ACTIVATED"
    assert calls == [["bash", "-n"]] and source_calls == []
    assert not (base / "submit_prepare_remote.py").exists()


def test_missing_real_source_identity_cannot_create_request(builder):
    module, base, _, _ = builder
    (base / "CODE_REVISION.json").unlink()
    with pytest.raises(FileNotFoundError, match="CODE_REVISION"):
        module.main()
    assert not (base / "submit_prepare_remote.py").exists()


@pytest.mark.parametrize("status", ["UNKNOWN", "FAIL", "REVIEW_COMPLETE_NOT_GATE_SIGNED"])
def test_old_or_incomplete_status_cannot_activate(builder, status):
    module, base, _, _ = builder
    p = base.parent / "closure_repair_20261029/DELIVERY.json"
    v = json.loads(p.read_text())
    v["status"] = status
    p.write_bytes(payload(v))
    with pytest.raises(PermissionError, match="completed corrected"):
        module.main()
    assert not (base / "submit_prepare_remote.py").exists()


def test_delivery_content_tamper_stops_before_activation(builder):
    module, base, _, _ = builder
    p = base.parent / "closure_repair_20261029/delivery" / contracts.CLOSURE / "NUMERICAL.json"
    p.write_bytes(b"{}")
    with pytest.raises(PermissionError, match="bytes changed"):
        module.main()
    assert not (base / "submit_prepare_remote.py").exists()


def test_complete_synthetic_delivery_produces_only_once_bound_request(builder, capsys):
    module, base, _, source_calls = builder
    module.main()
    assert json.loads(capsys.readouterr().out)["status"] == "ACTIVATED_NOT_SUBMITTED"
    request = json.loads((base / "ACTIVATED_PREPARE_REQUEST.json").read_text())
    rendered = (base / "submit_prepare_remote.py").read_text()
    assert request["producer_revision"] == "c" * 40 and source_calls[0][0] == "c" * 40
    assert request["server_actions"] == request["model_calls"] == 0
    assert request["corrected_confirmation_receipt_ref"]["path"].endswith(
        "/" + contracts.CLOSURE + "/RECEIPT.json"
    )
    assert hashlib.sha256(rendered.encode()).hexdigest() == request["sha256"]
    before = (base / "submit_prepare_remote.py").read_bytes()
    with pytest.raises(PermissionError, match="partial activated"):
        module.main()
    assert before == (base / "submit_prepare_remote.py").read_bytes()


def test_existing_partial_request_cannot_be_reassembled(builder):
    module, base, _, _ = builder
    (base / "submit_prepare_remote.py").write_text("# preserved partial")
    with pytest.raises(PermissionError, match="partial activated"):
        module.main()
    assert not (base / "ACTIVATED_PREPARE_REQUEST.json").exists()


def test_stale_old_confirmation_fields_are_not_silently_reused(builder):
    module, base, _, _ = builder
    p = base / "PREPARE_REFERENCE_INPUTS.json"
    v = json.loads(p.read_text())
    v["confirmation_receipt_sha256"] = None
    p.write_bytes(payload(v))
    with pytest.raises(PermissionError, match="Old UNKNOWN"):
        module.main()
    assert not (base / "submit_prepare_remote.py").exists()


def test_unexplained_source_delta_blocks_activation(builder):
    module, base, calls, _ = builder
    (base / "SOURCE_CHANGE_REASONS.json").write_text("{}")
    with pytest.raises(PermissionError, match="Every source difference"):
        module.main()
    assert calls == [["bash", "-n"]]


@pytest.mark.parametrize("missing", ["origin", "upstream", "parent_references"])
def test_incomplete_original_s0_references_cannot_create_request(builder, missing):
    module, base, _, _ = builder
    path = base / "PREPARE_REFERENCE_INPUTS.json"
    value = json.loads(path.read_text())
    del value[missing]
    path.write_bytes(payload(value))
    with pytest.raises(PermissionError, match=r"[Oo]riginal"):
        module.main()
    assert not (base / "submit_prepare_remote.py").exists()


@pytest.mark.parametrize(
    "record",
    [
        dict(revision="HEAD", deployment=str(contracts.PROJECT / "sr_f6_code_HEAD")),
        dict(revision="c" * 40, deployment="/tmp/sr_f6_code_ccccccc"),
        dict(
            revision="763172582c4e16a0d9583deecbbe72b0817a4a37",
            deployment=str(contracts.PROJECT / "sr_f6_code_7631725"),
        ),
    ],
)
def test_unbound_or_old_producer_rejected(record):
    with pytest.raises(PermissionError):
        contracts.producer(record)


def test_exact_membership_and_fixed_actual_pins(closure):
    assert contracts.closure_facts(closure)["closure_job_id"] == "233920"
    with pytest.raises(PermissionError, match="ten-member"):
        contracts.closure_facts(dict(closure, extra=b"{}"))
    changed = dict(closure)
    changed[contracts.CLOSURE + "/RECEIPT.json"] = b"{}"
    with pytest.raises(PermissionError, match="old UNKNOWN"):
        contracts.closure_facts(changed)


def test_templates_use_bound_dynamic_deployment_and_terminal_prerequisites():
    root = Path(__file__).resolve().parents[2] / "scripts/sr_f6/pilot_v3_closure_operations"
    submit = (root / "submit_prepare_remote.template.py").read_text()
    prepare = (root / "prepare_pilot_remote.template.py").read_text()
    wrapper = (root / "prepare_pilot.sbatch").read_text()
    assert "contracts.producer(" in submit and "phases._deployment(candidate, revision)" in submit
    assert "SRF6_PRODUCER_REVISION=" in submit and "SRF6_PRODUCER_DEPLOYMENT=" in submit
    assert "${revision:0:7}" in wrapper and "CUDA_VISIBLE_DEVICES" in wrapper
    assert "historical_closure.freeze_confirmation_binding" in prepare
    assert "corrected_confirmation_receipt_ref=inputs[" in prepare
    assert "tests/sr_f6/test_historical_closure.py" in prepare
    assert 'jobs = {"233723", "233849", "233873", "233920"}' in submit
    assert "SUBMISSION_UNKNOWN_TIMEOUT" in submit
    assert "No retry or overwrite" in submit
    assert "snapshot = capacity()" in submit
    assert "sr_f6_code_7631725" not in submit + prepare + wrapper


def test_selected_producer_must_contain_exact_local_operation_bytes(builder):
    module, base, _, _ = builder
    path = base / "EXPECTED_SOURCE_MANIFEST.json"
    manifest = json.loads(path.read_text())
    target = next(row for row in manifest["files"] if row["path"].endswith("/prepare_pilot.sbatch"))
    target["sha256"] = "f" * 64
    path.write_bytes(payload(manifest))
    reasons = json.loads((base / "SOURCE_CHANGE_REASONS.json").read_text())
    before = json.loads((base / "ORIGINAL_SOURCE_MANIFEST.json").read_text())
    (base / "EXPECTED_DELTA.json").write_bytes(
        payload(source_delta(before, manifest, reasons=reasons))
    )
    with pytest.raises(PermissionError, match="Local operations differ"):
        module.main()
    assert not (base / "submit_prepare_remote.py").exists()


@pytest.mark.parametrize("mode", ["valid", "unpublished", "wrong_archive", "missing_bridge"])
def test_selected_actual_commit_is_published_and_exact_archive(monkeypatch, mode):
    module = load_builder()
    files = {"src/sr_f6/historical_closure.py": b"# synthetic bridge", "src/x.py": b"# synthetic"}
    if mode == "missing_bridge":
        del files["src/sr_f6/historical_closure.py"]
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        for name, raw in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            tar.addfile(member, io.BytesIO(raw))
    calls = []

    def check_output(argv, **kwargs):
        calls.append(argv)
        if argv[1] == "cat-file":
            return "commit"
        if argv[1] == "branch":
            return "existing-branch"
        if argv[1] == "ls-remote":
            return "d" * 40 + "\trefs/heads/existing-branch"
        assert argv[1] == "archive" and argv[3] == "c" * 40
        return stream.getvalue()

    def run(argv, **kwargs):
        calls.append(argv)
        assert argv == ["git", "merge-base", "--is-ancestor", "c" * 40, "d" * 40]
        if mode == "unpublished":
            raise subprocess.CalledProcessError(1, argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "check_output", check_output)
    monkeypatch.setattr(module.subprocess, "run", run)
    expected = {
        "files": [
            dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            for name, raw in sorted(files.items())
        ]
    }
    if mode == "wrong_archive":
        expected["files"][0]["sha256"] = "f" * 64
    if mode == "valid":
        module.verify_published_producer("c" * 40, expected)
        assert any(call[1] == "ls-remote" for call in calls)
    else:
        with pytest.raises((PermissionError, subprocess.CalledProcessError)):
            module.verify_published_producer("c" * 40, expected)


@pytest.mark.parametrize("error", [subprocess.TimeoutExpired(["sbatch"], 25), OSError("transport")])
def test_unknown_scheduler_outcome_is_recorded_once_without_retry(error):
    """Execute the real template submission branch with a failing local scheduler stub."""
    repo = Path(__file__).resolve().parents[2]
    tree = ast.parse(
        (
            repo / "scripts/sr_f6/pilot_v3_closure_operations/submit_prepare_remote.template.py"
        ).read_text()
    )
    outer = next(node for node in tree.body if isinstance(node, ast.With))
    index = next(i for i, node in enumerate(outer.body) if isinstance(node, ast.Try))
    program = compile(
        ast.Module(body=outer.body[index:], type_ignores=[]), "synthetic_scheduler", "exec"
    )
    attempts, receipts = [], []

    def run(argv, **kwargs):
        attempts.append((argv, kwargs))
        raise error

    env = dict(
        subprocess=SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired),
        argv=["sbatch", "SYNTHETIC_TEST_ONLY"],
        now=lambda: "SYNTHETIC_TEST_ONLY",
        ops=Path("/SYNTHETIC_TEST_ONLY"),
        write_once=lambda path, value: receipts.append((path, value)),
        json=json,
        re=__import__("re"),
    )
    with pytest.raises(SystemExit) as stopped:
        exec(program, env)
    assert stopped.value.code == 1 and len(attempts) == len(receipts) == 1
    assert receipts[0][1]["status"].startswith("SUBMISSION_UNKNOWN_")
    assert receipts[0][1]["job_id"] is None and receipts[0][1]["requested_gpus"] == 0
