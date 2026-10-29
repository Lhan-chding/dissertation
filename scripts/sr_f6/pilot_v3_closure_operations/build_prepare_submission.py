"""Assemble a unique CPU request from actual corrected closure and committed source.

Validation alone never creates a request. Activation requires the complete
delivered repair result and an exact source manifest derived from a published
producer commit. Missing producer identity or evidence is never synthesized.
"""

import argparse
import ast
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

repo = Path(__file__).resolve().parents[3]
base = repo / ".local/pilot_v3_closure_20261029"
sys.path[:0] = [str(repo / "src"), str(repo)]
from scripts.sr_f6.pilot_v3_closure_operations import contracts  # noqa: E402
from sr_f6.phases import source_delta  # noqa: E402


def read(name):
    return json.loads((base / name).read_text())


def _source_member(name):
    path = Path(name)
    return (
        (name.startswith("src/") or name.startswith("tests/sr_f6/")) and path.suffix == ".py"
    ) or (name.startswith("scripts/sr_f6/") and path.suffix in {".py", ".sh", ".sbatch"})


def verify_published_producer(revision, expected):
    """Bind only executable source bytes from the selected published commit."""
    kind = subprocess.check_output(["git", "cat-file", "-t", revision], cwd=repo, text=True).strip()
    contracts.require(kind == "commit", "Selected producer must be a real Git commit")
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repo, text=True
    ).strip()
    contracts.require(bool(branch), "Existing origin branch required")
    remote = subprocess.check_output(
        ["git", "ls-remote", "origin", "refs/heads/" + branch], cwd=repo, text=True
    ).split()
    contracts.require(
        len(remote) == 2 and remote[1] == "refs/heads/" + branch,
        "Actual origin branch could not be verified",
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, remote[0]], cwd=repo, check=True
    )
    archive = subprocess.check_output(
        ["git", "archive", "--format=tar", revision, "src", "tests/sr_f6", "scripts/sr_f6"],
        cwd=repo,
    )
    members = []
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for item in tar.getmembers():
            if item.isdir() or not _source_member(item.name):
                continue
            contracts.require(
                item.isfile() and ".." not in Path(item.name).parts,
                "Regular committed source members required",
            )
            raw = tar.extractfile(item).read()
            members.append(
                dict(path=item.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            )
    members.sort(key=lambda row: row["path"])
    contracts.require(
        members == expected["files"], "Producer manifest differs from committed archive"
    )
    names = {row["path"] for row in members}
    contracts.require(
        "src/sr_f6/historical_closure.py" in names,
        "Actual corrected-closure production bridge is missing",
    )


def verify_local_operations(expected, templates):
    """The exact programs rendered here must belong to the selected producer."""
    members = {row["path"]: row for row in expected["files"]}
    local = [p for p in templates.iterdir() if p.suffix in {".py", ".sh", ".sbatch"}]
    prefix = str(templates.relative_to(repo)) + "/"
    declared = {name for name in members if name.startswith(prefix)}
    contracts.require(
        declared == {str(p.relative_to(repo)) for p in local},
        "Selected producer operations membership differs",
    )
    for path in local:
        raw = path.read_bytes()
        row = members[str(path.relative_to(repo))]
        contracts.require(
            not path.is_symlink()
            and row["bytes"] == len(raw)
            and row["sha256"] == hashlib.sha256(raw).hexdigest(),
            "Local operations differ from selected producer: " + path.name,
        )


def delivered_closure():
    upstream = base.parent / "closure_repair_20261029"
    delivery = json.loads((upstream / "DELIVERY.json").read_text())
    contracts.require(
        delivery["status"] == "CONFIRMATION_GATES_PASS"
        and delivery["job"] == ["233920", "COMPLETED", "0:0"]
        and delivery["source_revision"] == contracts.REPAIR_REVISION,
        "Actual completed corrected closure delivery required",
    )
    contracts.require(
        set(delivery["files"]) == contracts.EXPECTED_FILES,
        "Exact corrected closure membership required",
    )
    blobs = {}
    for name, row in delivery["files"].items():
        path = upstream / "delivery" / name
        contracts.require(
            not any(p.is_symlink() for p in (path, *path.parents)),
            "Symlinked local delivery forbidden",
        )
        raw = path.read_bytes()
        contracts.require(
            len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"],
            "Delivered closure bytes changed",
        )
        blobs[name] = raw
    actual = {
        str(p.relative_to(upstream / "delivery"))
        for p in (upstream / "delivery").rglob("*")
        if p.is_file()
    }
    contracts.require(actual == contracts.EXPECTED_FILES, "Unregistered delivery file")
    return contracts.closure_facts(blobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    options = parser.parse_args()
    templates = Path(__file__).resolve().parent
    program = (templates / "prepare_pilot_remote.template.py").read_text()
    wrapper = (templates / "prepare_pilot.sbatch").read_text()
    template = (templates / "submit_prepare_remote.template.py").read_text()
    for value in (program, template):
        ast.parse(value)
    subprocess.run(["bash", "-n"], input=wrapper, text=True, check=True)
    revision, code = contracts.producer(read("CODE_REVISION.json"))
    manifest, reasons, delta = (
        read(name)
        for name in (
            "EXPECTED_SOURCE_MANIFEST.json",
            "SOURCE_CHANGE_REASONS.json",
            "EXPECTED_DELTA.json",
        )
    )
    contracts.require(
        source_delta(read("ORIGINAL_SOURCE_MANIFEST.json"), manifest, reasons=reasons) == delta,
        "Declared source delta differs",
    )
    marker = "# ACTIVATED_PAYLOAD_INSERTION_POINT"
    contracts.require(template.count(marker) == 1, "Unique activation insertion point required")
    if options.validate_only:
        print(
            json.dumps(
                dict(
                    status="LOCAL_TEMPLATES_VALID_NOT_ACTIVATED",
                    producer_revision=revision,
                    source_members=len(manifest["files"]),
                    server_actions=0,
                    model_calls=0,
                )
            )
        )
        return
    paths = [base / name for name in ("submit_prepare_remote.py", "ACTIVATED_PREPARE_REQUEST.json")]
    contracts.require(
        not any(p.exists() or p.is_symlink() for p in paths),
        "Preserve existing or partial activated request",
    )
    tracked = [str(p.relative_to(repo)) for p in sorted(templates.iterdir()) if p.is_file()]
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", *tracked],
        cwd=repo,
        stdout=subprocess.DEVNULL,
        check=True,
    )
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *tracked], cwd=repo, check=True)
    operations_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    verify_published_producer(revision, manifest)
    verify_local_operations(manifest, templates)
    closure = delivered_closure()
    inputs = read("PREPARE_REFERENCE_INPUTS.json")
    contracts.require(
        inputs["actual_source_revision"] == revision
        and inputs["source_deployment"] == str(code)
        and inputs["root"] == str(contracts.ROOT),
        "Actual producer inputs differ",
    )
    contracts.require(
        not any(
            k in inputs
            for k in (
                "confirmation_receipt_sha256",
                "confirmation_receipt_path",
                "confirmation_receipt_ref",
                "followup_registration_ref",
            )
        ),
        "Old UNKNOWN receipt fields cannot activate corrected preparation",
    )
    contracts.original_references(inputs)
    inputs.update(closure)
    inputs.update(
        status="ACTUAL_CORRECTED_CLOSURE_DELIVERED_REQUEST_NOT_SUBMITTED",
        operations_revision=operations_revision,
        expected_source_manifest=manifest,
        source_change_reasons=reasons,
        expected_delta=delta,
    )
    rendered = template.replace(
        marker,
        "inputs = "
        + repr(inputs)
        + "\nprogram = "
        + repr(program)
        + "\nwrapper = "
        + repr(wrapper),
    )
    ast.parse(rendered)
    with paths[0].open("x") as stream:
        stream.write(rendered)
    with paths[1].open("x") as stream:
        json.dump(
            dict(
                status="ACTIVATED_NOT_SUBMITTED",
                filename=paths[0].name,
                sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                corrected_confirmation_receipt_ref=closure["corrected_confirmation_receipt_ref"],
                closure_registration_ref=closure["closure_registration_ref"],
                producer_revision=revision,
                source_deployment=str(code),
                operations_revision=operations_revision,
                model_calls=0,
                server_actions=0,
            ),
            stream,
            indent=2,
        )
        stream.write("\n")
    print(json.dumps(dict(status="ACTIVATED_NOT_SUBMITTED", filename=paths[0].name)))


if __name__ == "__main__":
    main()
