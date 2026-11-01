"""Transport failures must never turn into a second remote mutation."""

import base64
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/sr_f6_post_b_delivery/client.py"
    spec = importlib.util.spec_from_file_location("delivery_client_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    private = tmp_path.resolve() / "private"
    private.mkdir()
    monkeypatch.setattr(module, "PRIVATE", private)
    return module


def test_bootstrap_authenticates_before_executing(client):
    payload = b"def deploy(package, revision, binding):\n return {'status':'FIXTURE_ONLY'}\n"
    revision = "a" * 40
    manifest = dict(
        revision=revision,
        deployment="/unused",
        files={
            client.OPERATOR: dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        },
    )
    package = dict(
        manifest=manifest, contents={client.OPERATOR: base64.b64encode(payload).decode()}
    )
    binding = hashlib.sha256(
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()

    def run(value):
        return subprocess.run(
            [sys.executable, "-I", "-B", "-c", client.BOOTSTRAP, value, revision],
            input=json.dumps(package),
            capture_output=True,
            text=True,
            timeout=10,
        )

    result = run(binding)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"status": "FIXTURE_ONLY"}
    result = run("0" * 64)
    assert result.returncode != 0 and not result.stdout
    package["contents"][client.OPERATOR] = base64.b64encode(b"print('BAD_EXECUTED')").decode()
    result = run(binding)
    assert result.returncode != 0 and not result.stdout


def test_orphan_attempt_prevents_mutation(client):
    client.save(client.PRIVATE / "DEPLOY_one_ATTEMPT.json", {"action": "deploy"})
    with pytest.raises(PermissionError, match="Unknown/orphan"):
        client.prior_action("deploy")


@pytest.mark.parametrize("status", ["SUBMITTED", "SUBMISSION_UNKNOWN", "FAILED"])
def test_submission_not_retried(client, status):
    client.save(client.PRIVATE / "SUBMIT_one_ATTEMPT.json", {})
    client.save(
        client.PRIVATE / "SUBMIT_one_RESULT.json",
        dict(returncode=0, stdout=json.dumps(dict(status=status))),
    )
    with pytest.raises(PermissionError, match="forbids resubmission"):
        client.prior_action("submit")


def test_only_confirmed_capacity_decline_allows_later_submit(client):
    client.save(client.PRIVATE / "SUBMIT_one_ATTEMPT.json", {})
    client.save(
        client.PRIVATE / "SUBMIT_one_RESULT.json",
        dict(returncode=0, stdout='{"status":"NO_LEGAL_CAPACITY"}'),
    )
    client.prior_action("submit")


def test_timeout_preserves_partial_output_and_blocks_second_deploy(client, monkeypatch):
    identity = dict(revision="a" * 40, binding_sha256="b" * 64, deployment="/unused")
    monkeypatch.setattr(client, "source", lambda: (identity, {}))
    calls = []

    def timeout(argv, **kwargs):
        calls.append(argv)
        assert kwargs["timeout"] == 50 and argv[-2] == "eee-cluster"
        raise subprocess.TimeoutExpired(argv, 50, output=b'{"partial":', stderr=b"diagnostic")

    monkeypatch.setattr(client.subprocess, "run", timeout)
    path, result = client.transport("deploy")
    assert path.exists() and result["status"] == "UNKNOWN_PRESERVE_NO_RETRY"
    assert result["stdout"] == '{"partial":' and result["stderr"] == "diagnostic"
    with pytest.raises(PermissionError, match="already attempted"):
        client.transport("deploy")
    assert len(calls) == 1


def test_unpushed_revision_rejected_before_preparation(client, monkeypatch):
    revision = "a" * 40
    monkeypatch.setattr(
        client,
        "git",
        lambda *args: (revision + "\n").encode() if args[0] == "rev-parse" else b"other\tbranch\n",
    )
    with pytest.raises(PermissionError, match="Pushed branch"):
        client.prepare(revision)
    assert not list(client.PRIVATE.iterdir())


def test_local_execution_error_still_records_unknown_result(client, monkeypatch):
    identity = dict(revision="a" * 40, binding_sha256="b" * 64, deployment="/unused")
    monkeypatch.setattr(client, "source", lambda: (identity, {}))

    def failed(*args, **kwargs):
        raise FileNotFoundError("ssh executable unavailable")

    monkeypatch.setattr(client.subprocess, "run", failed)
    path, result = client.transport("deploy")
    assert path.exists() and result["error_type"] == "FileNotFoundError"
    assert result["status"] == "UNKNOWN_PRESERVE_NO_RETRY"
    assert len(list(client.PRIVATE.glob("DEPLOY_*_ATTEMPT.json"))) == 1
    with pytest.raises(PermissionError, match="already attempted"):
        client.transport("deploy")


def test_package_cannot_substitute_uncommitted_operator(client, monkeypatch):
    revision = "a" * 40
    contents = {name: b"committed\n" for name in client.MEMBERS}
    manifest = dict(
        revision=revision,
        deployment=client.PROJECT + "/sr_f6_post_b_delivery_code_" + revision[:7],
        files={
            name: dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
            for name, value in contents.items()
        },
    )
    package = dict(
        manifest=manifest,
        contents={name: base64.b64encode(value).decode() for name, value in contents.items()},
    )
    binding = hashlib.sha256(
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    client.save(
        client.PRIVATE / "SOURCE.json",
        dict(revision=revision, deployment=manifest["deployment"], binding_sha256=binding),
    )
    client.save(client.PRIVATE / "PACKAGE.json", package)
    monkeypatch.setattr(client, "git", lambda *args: b"different immutable source\n")
    with pytest.raises(PermissionError, match="not committed source"):
        client.source()
