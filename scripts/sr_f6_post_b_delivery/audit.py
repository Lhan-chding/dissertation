"""CPU-only delivery verification through the unchanged frozen PILOT_V3 verifier.

This is deliberately not pilot_audit.audit: that final fact boundary requires DEV.
No training, generation, recovery preparation, comparison publication or admission
is performed here. A separately pinned read-only recovery consumer retains the
original comparison assertions and consumes existing historical proof bytes.
A new output directory is an irreversible attempt boundary.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import importlib.util
import json
import os
import re
import resource
import stat
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

BASE = Path("/projects/_ssd/varunssd/louis-ssvc")
ROOT = BASE / "sr_f6_semantic_factorial_20261027"
DEPLOYMENT = BASE / "sr_f6_code_b7e7b06"
OPS = BASE / "sr_f6_operations_20261027"
B_OPS = OPS / "PILOT_ROUND_B_20261031"
NEW_OPS = OPS / "PILOT_POST_B_DELIVERY_20261101"
REGISTRATION = "PHASE_RUNS/PILOT_V3/PILOT_FROZEN"
PLAN = "PHASE_RUNS/PILOT_V3/JOBS/PLAN.json"
PLAN_SHA = "56102cd5b10d93bde1996527e39fa1ec30943fc2618cab2838f03d5fd550d4f4"
SPEC_SHA = "4936b29e3a897334c1c4e67aad436bf46bdea9a105354d0155484d2acc20994c"
PHASE_SHA = "d9f061d1cf08e14c485083bbed421a412a29868fd8305ad4c265fb6404562284"
ADMISSION_SHA = "b65831ad6ec57c829a69a9ad0a87a6aaf6a4a173e2d921cf953462ca7519b2a6"
MANIFEST_SHA = "c6f694582b152e5ccb3e0b303f4ace50c4432ec2ac884fccfd42b37de704bb87"
B_COMPLETE_SHA = "0047dabc1db4484e9949d1f367896e02df07b3d4ae6e99d9636d25b7c87bd77f"
B_CONTRACT_SHA = "daa002017251a8ec756cf3ded430f661693ea6703c94937426812a3706ef328e"
RECOVERY_CONSUMER_SHA = "56c50f7c8c9d710c27d3471aa590667c5776aa922869ae39212c629fdaba2a9c"
RECOVERY_DIRECTORY = "PHASE_RUNS/PILOT_V3/RECOVERY/b0/r0/U11"
SCHEMA = "srf6-post-b-delivery-v1"
PAIRS = {(p, a) for p in ("b0", "b1", "b2") for a in ("U10", "U11", "C_SHUF")}
B_TASKS = {"PRIMARY-b2-U10", "PRIMARY-b2-U11", "PRIMARY-b2-C_SHUF", "RECOVERY-b0-U11"}


def require(value, message):
    if not value:
        raise PermissionError(message)


def plain(path):
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "Absolute plain path required")
    for item in (*reversed(path.parents), path):
        require(not item.is_symlink(), f"Symlink forbidden: {item}")
    return path


def under(base, value):
    base = plain(base)
    relative = Path(value)
    path = plain(relative if relative.is_absolute() else base / relative)
    require(path != base and path.is_relative_to(base), "Reference escapes its root")
    return path


def read_bytes(path):
    path = plain(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "Regular file required")
        return stream.read()


def sha(path):
    return hashlib.sha256(read_bytes(path)).hexdigest()


def bound(base, reference):
    require(set(reference) == {"path", "sha256"}, "Exact hash reference required")
    require(re.fullmatch("[0-9a-f]{64}", reference["sha256"]), "SHA256 required")
    data = read_bytes(under(base, reference["path"]))
    require(hashlib.sha256(data).hexdigest() == reference["sha256"], "Reference hash differs")
    return json.loads(data)


def ref(path):
    return {"path": str(path), "sha256": sha(path)}


def write(path, value):
    path = under(NEW_OPS / "RESULT", path)
    with path.open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    return ref(path)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def usage():
    return {
        "process_lifetime_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "maxrss_unit": "bytes" if sys.platform == "darwin" else "KiB",
        "maxrss_scope": "PROCESS_LIFETIME_HIGH_WATER_NOT_STAGE_INCREMENT",
    }


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=20, check=True)
    return result.stdout


def scheduler():
    job = os.environ.get("SLURM_JOB_ID", "")
    require(re.fullmatch("[1-9][0-9]*", job), "A dedicated Slurm CPU job is required")
    allocation = command(["scontrol", "show", "job", "-o", job])
    fields = dict(token.split("=", 1) for token in allocation.split() if "=" in token)
    require(fields.get("JobId") == job and fields.get("JobState") == "RUNNING", "Audit not running")
    require(fields.get("JobName") == "LH-SRF6-PostB-Delivery", "Wrong audit allocation")
    require(
        fields.get("Account") == "rose"
        and fields.get("Partition") == "cluster02"
        and fields.get("QOS") == "soujanya-poria-startfund-2026-03",
        "Wrong audit account, partition or QoS",
    )
    require(fields.get("NumCPUs") == "4" and fields.get("Requeue") == "0", "Wrong CPU allocation")
    require(fields.get("TimeLimit") == "02:00:00", "Wrong audit time limit")
    for field in ("ReqTRES", "AllocTRES"):
        values = dict(
            part.split("=", 1) for part in fields.get(field, "").split(",") if "=" in part
        )
        require(values.get("cpu") == "4" and values.get("mem") == "12G", "Wrong audit resources")
        require(
            not any("gpu" in key and value != "0" for key, value in values.items()), "GPU forbidden"
        )
    accounting = command(["sacct", "-j", "237289", "-X", "-n", "-P", "-o", "JobID,State,ExitCode"])
    require(
        [line.strip("|").split("|") for line in accounting.splitlines() if line.strip()]
        == [["237289", "COMPLETED", "0:0"]],
        "B lacks actual successful terminal accounting",
    )
    queue = command(["squeue", "-h", "-u", "varun024", "-o", "%i"])
    require("237289" not in queue.split(), "B is still queued")
    return {
        "at": now(),
        "job_id": job,
        "allocation": allocation,
        "b_accounting": accounting,
        "queue": queue,
    }


def verify_b(plan):
    outer_ref = {"path": "round_b_COMPLETE.json", "sha256": B_COMPLETE_SHA}
    outer = bound(B_OPS, outer_ref)
    require(
        outer["job_id"] == "237289" and outer["status"] == "ROUND_B_COMPLETE_NOT_GATE_SIGNED",
        "Wrong B completion",
    )
    require(
        outer["contract_ref"] == {"path": str(B_OPS / "CONTRACT.json"), "sha256": B_CONTRACT_SHA},
        "B contract differs",
    )
    bound(B_OPS, outer["contract_ref"])
    require(
        len(outer["completed"]) == 4
        and {item["task_id"] for item in outer["completed"]} == B_TASKS,
        "Incomplete B closure",
    )
    for path in [
        B_OPS / "round_b_FAILURE.json",
        *(B_OPS / "LANES" / str(i) / "FAILURE.json" for i in range(2)),
    ]:
        require(not plain(path).exists(), "New B failure exists")
    for worker, queue in enumerate(outer["queues"]):
        folder = B_OPS / "LANES" / str(worker)
        started = json.loads(read_bytes(folder / "STARTED.json"))
        complete = json.loads(read_bytes(folder / "COMPLETE.json"))
        require(
            started["job_id"] == complete["job_id"] == "237289"
            and started["worker"] == complete["worker"] == worker
            and started["queue"] == queue
            and complete["status"] == "QUEUE_COMPLETE_NOT_GATE_SIGNED"
            and complete["completed"] == [r for r in outer["completed"] if r["task_id"] in queue],
            "B lane closure differs",
        )
    receipts = []
    recovery_producer = None
    for item in outer["completed"]:
        task_id = item["task_id"]
        task_dir = ROOT / "PHASE_RUNS/PILOT_V3/JOBS/TASKS" / task_id
        execution_dir = B_OPS / "EXECUTIONS" / task_id
        require(
            item["task_complete_ref"]["path"] == str(task_dir / "COMPLETE.json")
            and item["execution_complete_ref"]["path"] == str(execution_dir / "COMPLETE.json")
            and item["storage_facts_ref"]["path"] == str(execution_dir / "STORAGE_FACTS.json"),
            "B closure paths differ",
        )
        require(not plain(execution_dir / "FAILURE.json").exists(), "New execution failure exists")
        task = bound(ROOT, item["task_complete_ref"])
        execution = bound(B_OPS, item["execution_complete_ref"])
        storage = bound(B_OPS, item["storage_facts_ref"])
        expected = next(t for t in plan["tasks"] if t["task_id"] == task_id)
        for receipt in (task, execution):
            require(
                receipt["status"] == "TASK_COMPLETE_NOT_GATE_SIGNED"
                and receipt["task"] == expected
                and receipt["plan_ref"] == {"path": PLAN, "sha256": PLAN_SHA},
                "B task receipt differs",
            )
        require(task["producer"] == execution["producer"], "B producer differs")
        producer = bound(ROOT, task["producer"])
        require(
            execution["storage_facts_ref"] == item["storage_facts_ref"], "Storage reference differs"
        )
        offload = storage["offload"]
        require(
            storage["status"] == "ACTUAL_STORAGE_CONTEXT_COMPLETE_NOT_SCIENTIFIC_PASS"
            and storage["contract_ref"] == outer["contract_ref"]
            and offload == execution["offload"],
            "Actual storage provenance differs",
        )
        require(
            offload["closed"] is True
            and offload["storage_failures"] == []
            and offload["cleanup_errors"] == []
            and all(
                offload[k] == 0 for k in ("live_device_bytes", "live_host_bytes", "live_disk_bytes")
            ),
            "Storage did not close cleanly",
        )
        if task_id == "RECOVERY-b0-U11":
            saved = {
                key: bound(ROOT, producer[key])
                for key in (
                    "training_ref",
                    "binding_ref",
                    "comparison_ref",
                    "recovery_execution_ref",
                )
            }
            comparison = saved["comparison_ref"]
            require(
                comparison["status"] == "PASS"
                and comparison["scope"] == "SERVER_ACTUAL"
                and comparison["exact"] is True
                and comparison["independent_process_verified"] is True
                and comparison["new_generated_responses"] == 1024,
                "Saved recovery receipt lacks registered exactness",
            )
            recovery_producer = {
                "scope": "HASH_BOUND_SAVED_RECEIPTS_ONLY_NOT_RECOMPUTED",
                "producer_ref": task["producer"],
                "producer": producer,
                "receipts": saved,
            }
        receipts.append(item)
    return {
        "outer_ref": {"path": str(B_OPS / outer_ref["path"]), "sha256": B_COMPLETE_SHA},
        "receipts": receipts,
    }, recovery_producer


def source_and_pins():
    plan = bound(ROOT, {"path": PLAN, "sha256": PLAN_SHA})
    identity = bound(ROOT, {"path": "PHASES/PILOT_V3/IDENTITY.json", "sha256": PHASE_SHA})
    admission_ref = {"path": "PHASES/PILOT_V3/ADMISSION.json", "sha256": ADMISSION_SHA}
    bound(ROOT, admission_ref)
    spec = bound(ROOT, {"path": REGISTRATION + "/PILOT_SPEC_FROZEN.json", "sha256": SPEC_SHA})
    manifest = bound(ROOT, {"path": "PHASES/PILOT_V3/SOURCE_MANIFEST.json", "sha256": MANIFEST_SHA})
    require(
        identity["deployment"] == str(DEPLOYMENT) and identity["root"] == str(ROOT), "Wrong roots"
    )
    require(
        len(spec["paths"]) == 9 and {(p["parent"], p["arm"]) for p in spec["paths"]} == PAIRS,
        "Exact nine primary paths required",
    )
    require(
        len(manifest["files"]) == 680 and len({f["path"] for f in manifest["files"]}) == 680,
        "Frozen source inventory differs",
    )
    for item in manifest["files"]:
        data = read_bytes(under(DEPLOYMENT, item["path"]))
        require(
            len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"],
            "Frozen source bytes differ",
        )
    context = {
        "phase": "PILOT_V3",
        "status": "BOUND_NOT_AUTHORIZED",
        "training_allowed": False,
        "deployment": str(DEPLOYMENT),
        "identity": identity,
        "identity_sha256": PHASE_SHA,
    }
    return plan, spec, context, admission_ref


def existing_transaction(transactions, directory, identity):
    directory = plain(directory)
    require(
        directory.is_dir() and plain(directory / "checkpoints").is_dir(), "Existing run required"
    )
    for name in ("RUN_IDENTITY.json", ".run.lock"):
        require(
            stat.S_ISREG(plain(directory / name).stat().st_mode), "Existing identity/lock required"
        )
    require(
        json.loads(read_bytes(directory / "RUN_IDENTITY.json")) == identity,
        "Existing identity differs",
    )
    return transactions.RunTransactions(directory, identity)


def load_recovery_consumer():
    path = plain(Path(__file__).parent / "recovery_readonly.py")
    require(sha(path) == RECOVERY_CONSUMER_SHA, "Recovery consumer differs from driver pin")
    spec = importlib.util.spec_from_file_location("srf6_delivery_recovery_readonly", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(plain(module.__file__) == path, "Wrong recovery consumer origin")
    require(sha(path) == RECOVERY_CONSUMER_SHA, "Recovery consumer changed during import")
    return module


def recovery_passed(value):
    comparison = value["comparison"]
    return (
        value["status"] == "RECOVERY_FACTS_RECHECKED_NOT_GATE_SIGNED"
        and comparison["status"] == "PASS"
        and comparison["exact"] is True
        and comparison["differences"] == []
        and comparison["independent_process_verified"] is True
        and comparison["new_generated_responses"] == 1024
        and value["comparison_history"]["status"] == "ALL_SAVED_FACTS_VERIFIED"
        and value["new_responses"] == value["new_updates"] == 0
    )


def verify_training(output, spec, context, admission_ref, recovery_saved):
    require(
        not any(name == "sr_f6" or name.startswith("sr_f6.") for name in sys.modules),
        "Start a fresh interpreter; do not reuse imported research modules",
    )
    sys.path.insert(0, str(DEPLOYMENT / "src"))
    audit = importlib.import_module("sr_f6.pilot_audit")
    transactions = importlib.import_module("sr_f6.transactions")
    torch = importlib.import_module("torch")
    require(
        torch.get_num_threads() == 1 and not torch.cuda.is_initialized(), "CPU-only OMP1 required"
    )
    for module in (audit, transactions):
        require(
            plain(module.__file__)
            == DEPLOYMENT / "src" / (module.__name__.replace(".", "/") + ".py"),
            "Wrong imported verifier",
        )
    consumer = load_recovery_consumer()
    paths = spec["paths"]
    runs, locked, snapshots, measurements = [], {}, {}, []
    with ExitStack() as stack:
        for item in paths:
            identity = bound(ROOT / REGISTRATION, item["transaction_identity"])
            directory = under(ROOT, item["audit_root"])
            tx = stack.enter_context(existing_transaction(transactions, directory, identity))
            locked[(item["parent"], item["arm"])] = tx
            snapshots[directory] = audit._snapshot(directory)
        primary = locked[("b0", "U11")]
        recovery_directory = under(ROOT, RECOVERY_DIRECTORY)
        recovery_tx = stack.enter_context(
            existing_transaction(transactions, recovery_directory, primary.identity)
        )
        snapshots[recovery_directory] = audit._snapshot(recovery_directory)
        for item in paths:
            tx = locked[(item["parent"], item["arm"])]
            started, before_usage = time.perf_counter(), usage()
            write(
                output / f"PRIMARY-{item['parent']}-{item['arm']}_STARTED.json",
                {"at": now(), "parent": item["parent"], "arm": item["arm"]},
            )
            facts = audit._verify_run(
                tx,
                ROOT,
                attempt_pins=dict(
                    context=context["identity"],
                    phase_identity_sha256=PHASE_SHA,
                    pilot_spec_sha256=SPEC_SHA,
                    admission_ref=admission_ref,
                ),
            )
            require(
                audit._snapshot(tx.directory) == snapshots[tx.directory],
                "Run changed during verification",
            )
            runs.append(facts)
            measurement = dict(
                stage=f"PRIMARY-{item['parent']}-{item['arm']}",
                elapsed_seconds=time.perf_counter() - started,
                usage_before=before_usage,
                usage_after=usage(),
            )
            measurements.append(measurement)
            write(
                output / f"PRIMARY-{item['parent']}-{item['arm']}.json",
                {"facts": facts, "measurement": measurement},
            )
        started, before_usage = time.perf_counter(), usage()
        write(
            output / "RECOVERY_STARTED.json",
            {"at": now(), "consumer_sha256": RECOVERY_CONSUMER_SHA},
        )
        producer = recovery_saved["producer"]
        recovered = consumer.verify_saved_recovery(
            primary,
            recovery_tx,
            root=ROOT,
            phase_context=context,
            pilot_spec_sha256=SPEC_SHA,
            binding_ref=producer["binding_ref"],
            comparison_ref=producer["comparison_ref"],
            recovery_execution_ref=producer["recovery_execution_ref"],
        )
        measurement = dict(
            stage="RECOVERY",
            elapsed_seconds=time.perf_counter() - started,
            usage_before=before_usage,
            usage_after=usage(),
        )
        measurements.append(measurement)
        write(output / "RECOVERY.json", {"facts": recovered, "measurement": measurement})
        require(
            recovery_passed(recovered), "Recovery recheck or full comparison history incomplete"
        )
        require(
            all(audit._snapshot(path) == snapshot for path, snapshot in snapshots.items()),
            "Primary/recovery inventory changed during verification",
        )
        require(not torch.cuda.is_initialized(), "Unexpected CUDA initialization")
        return {
            "runs": runs,
            "recovery": recovered,
            "exposure": audit._exposure(runs),
            "paths": 9,
            "logical_updates": 108,
            "primary_responses": 13824,
            "measurements": measurements,
        }


def run(contract_path, contract_sha256):
    require(
        os.environ.get("OMP_NUM_THREADS") == "1" and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "Explicit OMP1 and hidden CUDA devices required",
    )
    require(
        plain(contract_path) == NEW_OPS / "CONTRACT.json", "Fixed independent contract required"
    )
    contract = bound(NEW_OPS, {"path": str(contract_path), "sha256": contract_sha256})
    require(
        set(contract)
        == {
            "schema",
            "output_directory",
            "driver_sha256",
            "source_deployment",
            "b_job_id",
            "round_b_complete_sha256",
        },
        "Unknown contract fields",
    )
    require(
        contract["schema"] == SCHEMA
        and contract["output_directory"] == str(NEW_OPS / "RESULT")
        and contract["source_deployment"] == str(DEPLOYMENT)
        and contract["b_job_id"] == "237289"
        and contract["round_b_complete_sha256"] == B_COMPLETE_SHA,
        "Contract differs",
    )
    require(
        sha(under(BASE, __file__)) == contract["driver_sha256"],
        "Driver differs from frozen contract",
    )
    output = plain(contract["output_directory"])
    output.mkdir()  # Exclusive: any partial/failed/completed attempt prevents reuse.
    write(
        output / "INTENT.json",
        {"at": now(), "contract_ref": ref(contract_path), "pid": os.getpid()},
    )
    try:
        scheduling = scheduler()
        write(output / "STARTED.json", scheduling)
        plan, spec, context, admission_ref = source_and_pins()
        closure, producer = verify_b(plan)
        write(output / "B_CLOSURE.json", closure)
        facts = verify_training(output, spec, context, admission_ref, producer)
        result = dict(
            schema=SCHEMA,
            status="DELIVERY_FACTS_NOT_SCIENTIFIC_GATE",
            recovery_recheck_status="EXACT_SAVED_FACTS_AND_HISTORY_VERIFIED",
            full_delivery_verified=True,
            at=now(),
            gate_status="NOT_SIGNED",
            evaluation_authorized=False,
            formal_authorized=False,
            new_responses=0,
            new_updates=0,
            cpu_threads=1,
            contract_ref=ref(contract_path),
            scheduler=scheduling,
            b_closure=closure,
            recovery_saved_receipt_closure=producer,
            **facts,
        )
        result_ref = write(output / "RESULT.json", result)
        write(
            output / "COMPLETE.json",
            {
                "status": "DELIVERY_FACTS_NOT_SCIENTIFIC_GATE",
                "full_delivery_verified": True,
                "recovery_recheck_status": "EXACT_SAVED_FACTS_AND_HISTORY_VERIFIED",
                "at": now(),
                "result_ref": result_ref,
                "gate_status": "NOT_SIGNED",
            },
        )
        return result
    except BaseException as exc:
        write(
            output / "FAILURE.json",
            {
                "at": now(),
                "status": "DELIVERY_VERIFICATION_INCOMPLETE",
                "scientific_gate_status": "NOT_EVALUATED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "new_responses": 0,
                "new_updates": 0,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--contract-sha256", required=True)
    args = parser.parse_args()
    result = run(args.contract, args.contract_sha256)
    print(json.dumps({"status": result["status"], "gate_status": result["gate_status"]}))


if __name__ == "__main__":
    main()
