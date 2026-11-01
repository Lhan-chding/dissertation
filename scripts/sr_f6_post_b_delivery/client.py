"""Bounded local transport; private receipts never enter the Git source package."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import importlib.util
import json
import re
import shlex
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PRIVATE = REPO / ".local/pilot_post_b_delivery_20261101"
BRANCH = "sr-f6-semantic-factorial-v1"
PROJECT = "/projects/_ssd/varunssd/louis-ssvc"
PYTHON = PROJECT + "/envs/ssvc-py312/bin/python3.12"
OPERATOR = "scripts/sr_f6_post_b_delivery/operations.py"
MEMBERS = (
    OPERATOR,
    "scripts/sr_f6_post_b_delivery/audit.py",
    "scripts/sr_f6_post_b_delivery/launch.sbatch",
    "scripts/sr_f6_post_b_delivery/recovery_readonly.py",
)
BOOTSTRAP = r"""
import base64,hashlib,json,sys
package=json.loads(sys.stdin.buffer.read(4*1024*1024+1))
manifest=package['manifest']
encoded=(json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
assert hashlib.sha256(encoded).hexdigest()==sys.argv[1]
assert manifest['revision']==sys.argv[2]
name='scripts/sr_f6_post_b_delivery/operations.py'
payload=base64.b64decode(package['contents'][name],validate=True)
assert len(payload)==manifest['files'][name]['bytes']
assert hashlib.sha256(payload).hexdigest()==manifest['files'][name]['sha256']
scope={'__name__':'authenticated_delivery_bootstrap','__file__':manifest['deployment']+'/'+name}
exec(compile(payload,name,'exec'),scope)
print(json.dumps(scope['deploy'](package,sys.argv[2],sys.argv[1]),sort_keys=True,allow_nan=False))
"""


def require(value, message):
    if not value:
        raise PermissionError(message)


def plain(path):
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "Canonical absolute path required")
    for part in (*reversed(path.parents), path):
        require(not part.is_symlink(), "Symlink forbidden")
    return path


def save(path, value):
    path = plain(path)
    require(path.parent == PRIVATE, "Private receipt directory required")
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def git(*args):
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, check=True, timeout=50
    ).stdout


def prepare(revision):
    require(re.fullmatch(r"[0-9a-f]{40}", revision), "Full committed revision required")
    require(git("rev-parse", "HEAD").decode().strip() == revision, "HEAD differs")
    expected = f"{revision}\trefs/heads/{BRANCH}"
    require(
        git("ls-remote", "origin", f"refs/heads/{BRANCH}").decode().strip() == expected,
        "Pushed branch revision is not verified",
    )
    contents = {name: git("show", f"{revision}:{name}") for name in MEMBERS}
    for name, data in contents.items():
        require(plain(REPO / name).read_bytes() == data, "Working source differs from commit")
    spec = importlib.util.spec_from_file_location("delivery_package_builder", REPO / OPERATOR)
    operator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(operator)
    package = operator.make_package(revision, contents)
    binding = operator.digest(operator.encode(package["manifest"]))
    plain(PRIVATE).mkdir()  # Existing/partial preparation requires reconciliation.
    source = dict(
        revision=revision,
        deployment=package["manifest"]["deployment"],
        binding_sha256=binding,
        ls_remote=expected,
    )
    save(PRIVATE / "PACKAGE.json", package)
    save(PRIVATE / "SOURCE.json", source)
    return source


def source():
    identity = json.loads(plain(PRIVATE / "SOURCE.json").read_bytes())
    revision = identity["revision"]
    require(re.fullmatch(r"[0-9a-f]{40}", revision), "Invalid revision")
    require(re.fullmatch(r"[0-9a-f]{64}", identity["binding_sha256"]), "Invalid binding")
    require(
        identity["deployment"] == PROJECT + "/sr_f6_post_b_delivery_code_" + revision[:7],
        "Unexpected deployment",
    )
    package = json.loads(plain(PRIVATE / "PACKAGE.json").read_bytes())
    payload = (
        json.dumps(package["manifest"], sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    require(hashlib.sha256(payload).hexdigest() == identity["binding_sha256"], "Package changed")
    require(package["manifest"]["revision"] == revision, "Package revision differs")
    require(package["manifest"]["deployment"] == identity["deployment"], "Package path differs")
    require(set(package["contents"]) == set(MEMBERS), "Package membership differs")
    require(set(package["manifest"]["files"]) == set(MEMBERS), "Manifest membership differs")
    for name in MEMBERS:
        original = git("show", f"{revision}:{name}")
        require(
            base64.b64decode(package["contents"][name], validate=True) == original,
            "Package is not committed source",
        )
        require(
            package["manifest"]["files"][name]
            == dict(bytes=len(original), sha256=hashlib.sha256(original).hexdigest()),
            "Manifest is not committed source",
        )
    return identity, package


def prior_action(action):
    attempts = sorted(PRIVATE.glob(action.upper() + "_*_ATTEMPT.json"))
    results = sorted(PRIVATE.glob(action.upper() + "_*_RESULT.json"))
    require(
        {p.name.replace("_ATTEMPT", "_RESULT") for p in attempts} == {p.name for p in results},
        "Unknown/orphan operation: preserve and reconcile; never retry",
    )
    if action in ("deploy", "freeze"):
        require(not attempts, "One-shot action already attempted")
    if action == "submit":
        for path in results:
            result = json.loads(plain(path).read_bytes())
            require(
                result.get("returncode") == 0
                and json.loads(result["stdout"]).get("status") == "NO_LEGAL_CAPACITY",
                "Prior submitted/failed/unknown attempt forbids resubmission",
            )


def contract_hash():
    results = sorted(PRIVATE.glob("FREEZE_*_RESULT.json"))
    require(len(results) == 1, "Unique successful freeze receipt required")
    result = json.loads(plain(results[0]).read_bytes())
    require(result.get("returncode") == 0, "Freeze result unknown or failed")
    frozen = json.loads(result["stdout"])
    require(frozen["status"] == "FROZEN_NOT_SUBMITTED", "Freeze not established")
    value = frozen["contract_ref"]
    require(
        value["path"]
        == PROJECT + "/sr_f6_operations_20261027/PILOT_POST_B_DELIVERY_20261101/CONTRACT.json"
        and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
        "Unexpected contract reference",
    )
    return value["sha256"]


def transport(action):
    require(action in ("deploy", "freeze", "submit", "read", "reconcile"), "Unknown action")
    identity, package = source()
    if action != "read":
        prior_action(action)
    if action == "deploy":
        remote = [
            PYTHON,
            "-I",
            "-B",
            "-c",
            BOOTSTRAP,
            identity["binding_sha256"],
            identity["revision"],
        ]
        stdin = json.dumps(package)
    else:
        remote = [
            PYTHON,
            "-I",
            "-B",
            identity["deployment"] + "/" + OPERATOR,
            action,
            "--revision",
            identity["revision"],
            "--binding-sha256",
            identity["binding_sha256"],
        ]
        if action in ("submit", "reconcile"):
            remote += ["--contract-sha256", contract_hash()]
        stdin = None
    argv = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=12",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=2",
        "eee-cluster",
        shlex.join(remote),
    ]
    at = dt.datetime.now(dt.timezone.utc)
    stem = action.upper() + "_" + at.strftime("%Y%m%dT%H%M%S%f")
    save(
        PRIVATE / (stem + "_ATTEMPT.json"),
        dict(at=at.isoformat(), action=action, source=identity, argv=argv),
    )
    try:
        result = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=50)
        receipt = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except subprocess.TimeoutExpired as exc:

        def decoded(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value

        receipt = dict(
            status="UNKNOWN_PRESERVE_NO_RETRY",
            stdout=decoded(exc.stdout),
            stderr=decoded(exc.stderr),
        )
    except OSError as exc:
        receipt = dict(
            status="UNKNOWN_PRESERVE_NO_RETRY", error_type=type(exc).__name__, error=str(exc)
        )
    receipt.update(at=at.isoformat(), action=action, source=identity)
    path = PRIVATE / (stem + "_RESULT.json")
    save(path, receipt)
    return path, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "deploy", "freeze", "submit", "read", "reconcile")
    )
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.action == "prepare":
        require(args.revision is not None, "Committed revision required")
        print(json.dumps(prepare(args.revision), indent=2))
    else:
        path, receipt = transport(args.action)
        print(path)
        if receipt.get("returncode") == 0:
            value = json.loads(receipt["stdout"])
            print(
                json.dumps(
                    {
                        key: value[key]
                        for key in ("status", "job_id", "contract_ref", "submitted_by_user")
                        if key in value
                    },
                    indent=2,
                )
            )
        else:
            print(json.dumps(receipt, indent=2))
            raise SystemExit(1)


if __name__ == "__main__":
    main()
