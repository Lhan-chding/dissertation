"""Assemble an operations request only from delivered actual upstream PASS.

--validate-only checks prospective templates and source delta without creating
an activated request, contacting the server, or inventing missing evidence.
"""

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[3]
base = repo / ".local/pilot_v3_20261029"
sys.path.insert(0, str(repo / "src"))
from sr_f6.phases import source_delta  # noqa: E402


def read(name):
    return json.loads((base / name).read_text())


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--validate-only", action="store_true")
    options = args.parse_args()
    templates = Path(__file__).resolve().parent
    program = (templates / "prepare_pilot_remote.template.py").read_text()
    wrapper = (templates / "prepare_pilot.sbatch").read_text()
    template = (templates / "submit_prepare_remote.template.py").read_text()
    for value in (program, template):
        ast.parse(value)
    subprocess.run(["bash", "-n"], input=wrapper, text=True, check=True)
    manifest, reasons, delta = (
        read(n)
        for n in (
            "EXPECTED_SOURCE_MANIFEST.json",
            "SOURCE_CHANGE_REASONS.json",
            "EXPECTED_DELTA.json",
        )
    )
    assert source_delta(read("ORIGINAL_SOURCE_MANIFEST.json"), manifest, reasons=reasons) == delta
    marker = "# ACTIVATED_PAYLOAD_INSERTION_POINT"
    assert template.count(marker) == 1
    if options.validate_only:
        print(
            json.dumps(
                dict(
                    status="LOCAL_OPERATIONS_TEMPLATES_VALID_NOT_ACTIVATED",
                    source_changes=len(reasons),
                    source_members=len(manifest["files"]),
                    program_sha256=hashlib.sha256(program.encode()).hexdigest(),
                    wrapper_sha256=hashlib.sha256(wrapper.encode()).hexdigest(),
                    server_actions=0,
                    actual_model_calls=0,
                )
            )
        )
        return
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
    upstream = base.parent / "confirm_followup_v2_20261029"
    delivery = json.loads((upstream / "CONFIRMATION_GATE_DELIVERY.json").read_text())
    assert delivery["status"] == "CONFIRMATION_GATES_PASS"
    assert len(delivery["accounting"]) == 2 and "233723" in delivery["accounting"]
    assert all(v == ["COMPLETED", "0:0"] for v in delivery["accounting"].values())
    receipt_bytes = (upstream / "delivery/ADMISSION/RECEIPT.json").read_bytes()
    pin = delivery["files"]["ADMISSION/RECEIPT.json"]["sha256"]
    assert hashlib.sha256(receipt_bytes).hexdigest() == pin
    receipt = json.loads(receipt_bytes)
    assert receipt["status"] == delivery["status"]
    assert receipt["gate_status"] == "CONFIRMATION_FACTS_SEALED"
    assert all(
        receipt[k] is False
        for k in ("generation_allowed", "training_allowed", "evaluation_allowed")
    )
    inputs = read("PREPARE_REFERENCE_INPUTS.json")
    assert inputs["actual_source_revision"] == read("CODE_REVISION.json")["revision"]
    assert inputs["confirmation_receipt_sha256"] is None
    inputs.pop("confirmation_receipt_sha256")
    inputs["status"] = "ACTUAL_UPSTREAM_DELIVERED_REQUEST_NOT_SUBMITTED"
    inputs["operations_revision"] = operations_revision
    inputs["confirmation_receipt_ref"] = dict(
        path=inputs.pop("confirmation_receipt_path"), sha256=pin
    )
    inputs["confirmation_admission_job_id"] = next(
        j for j in delivery["accounting"] if j != "233723"
    )
    inputs["expected_source_manifest"] = manifest
    inputs["source_change_reasons"] = reasons
    inputs["expected_delta"] = delta
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
    path = base / "submit_prepare_remote.py"
    with path.open("x") as stream:
        stream.write(rendered)
    with (base / "ACTIVATED_PREPARE_REQUEST.json").open("x") as stream:
        json.dump(
            dict(
                status="ACTIVATED_NOT_SUBMITTED",
                filename=path.name,
                sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                actual_confirmation_receipt=inputs["confirmation_receipt_ref"],
                producer_revision=inputs["actual_source_revision"],
                operations_revision=operations_revision,
                actual_model_calls=0,
                server_actions=0,
            ),
            stream,
            indent=2,
        )
        stream.write("\n")
    print(json.dumps(dict(status="ACTIVATED_NOT_SUBMITTED", filename=path.name)))


if __name__ == "__main__":
    main()
