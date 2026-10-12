#!/usr/bin/env python3
"""Submit one bound SR-F2 CPU job, held until verified; never retry a submission.

Repeated invocations only observe or release the existing, authenticated held job.
The preflight job never submits its controller; that is a separate explicit call.
"""

from __future__ import annotations

import argparse
import fcntl
import getpass
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sr_f2.protocol import PARENT_ROOT, verify_registration
from sr_f12.evaluation import verify_source_manifest
from sr_f12.orchestration import (
    TERMINAL,
    file_hash,
    gpu_count,
    memory_gib,
    now,
    read_json,
    save_json,
    slurm_fields,
)
from sr_f12.protocol import object_hash

CPU_QOS = "override-limits-but-killable"
CPU_NODE = "cpu-2"
KINDS = ("preflight", "controller")


def verify_phase(directory, phase, code):
    """Reuse a successful CPU phase only after rehashing its complete source ledger."""
    directory, code = Path(directory), Path(code)
    cohort = phase == "chartqa122"
    receipt_path = directory / ("COHORT_VERIFICATION.json" if cohort else "SUMMARY.json")
    receipt = read_json(receipt_path)
    if receipt.get("status") != "PASS":
        raise PermissionError("Existing CPU phase is not PASS; preserved without retry")
    source_path = directory / "SOURCE_MANIFEST.json"
    if file_hash(source_path) != receipt.get("source_manifest_sha256"):
        raise PermissionError("CPU phase source ledger changed")
    entries = read_json(source_path).get("files")
    if not isinstance(entries, list) or not entries:
        raise PermissionError("CPU phase has no complete source ledger")
    seen = set()
    for entry in entries:
        path = Path(entry.get("source_path", entry.get("path", "")))
        canonical = path.resolve(strict=True)
        if (not path.is_absolute() or str(path) in seen or not canonical.is_file()
                or entry.get("canonical_path", str(canonical)) != str(canonical)
                or file_hash(canonical) != entry["sha256"]
                or canonical.stat().st_size != entry["bytes"]):
            raise PermissionError("Successful CPU phase source changed")
        seen.add(str(path))
    script = code / "scripts/sr_f2" / ("verify_chartqa_cohort.py" if cohort else "audit_prior.py")
    if file_hash(script) != receipt.get("script_sha256"):
        raise PermissionError("Successful CPU phase used a different implementation")
    if cohort:
        rows = receipt.get("rows", {})
        if (receipt.get("count") != 122 or rows.get("path") != "COHORT_ROWS.jsonl"
                or rows.get("count") != 122
                or receipt.get("cohort_identity_verified") is not True
                or receipt.get("raw_output_count") != 1586
                or receipt.get("raw_evidence_status") != "RAW_AND_GOLD_REVERIFIED"
                or file_hash(directory / "COHORT_ROWS.jsonl") != rows.get("sha256")):
            raise PermissionError("Successful ChartQA phase lacks complete raw/gold evidence")
        with (directory / "COHORT_ROWS.jsonl").open() as stream:
            if sum(bool(line.strip()) for line in stream) != 122:
                raise PermissionError("ChartQA phase row count differs")
    else:
        if (receipt.get("phase") != phase
                or Path(receipt["root"]).resolve() != Path(PARENT_ROOT).resolve()
                or file_hash(directory / "AUDIT.jsonl") != receipt.get("audit_jsonl_sha256")):
            raise PermissionError("Successful CPU phase output/identity changed")
    return dict(phase=phase, receipt_sha256=file_hash(receipt_path))


def verify_prepared_inputs(root):
    from sr_f2.protocol import validate_config, verify_input_files

    identity = verify_input_files(root)
    if (identity.get("status") != "PASS" or Path(identity.get("root", "")).resolve() != root
            or Path(identity.get("prior_root", "")).resolve() != Path(PARENT_ROOT).resolve()):
        raise PermissionError("Prepared input receipt is not PASS")
    if file_hash(Path(PARENT_ROOT) / "INPUT_IDENTITY.json") != identity["prior_identity_sha256"]:
        raise PermissionError("Prepared parent input identity changed")
    manifest = read_json(root / "DIAGNOSTIC_MANIFEST.json")
    if object_hash(manifest) != identity["diagnostic_manifest_sha256"]:
        raise PermissionError("Prepared diagnostic manifest changed")
    spec = importlib.util.spec_from_file_location(
        "_sr_f2_prepare", Path(__file__).with_name("prepare.py")
    )
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    model = Path(identity["model_path"]).resolve(strict=True)
    for item in identity["model_files"]:
        path = prepare.model_file(model, item["name"])
        if file_hash(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise PermissionError("Prepared shared base-model bytes changed")
    validate_config(root / "config/SR_F2.json")
    return dict(stage="prepare", receipt_sha256=file_hash(root / "INPUT_IDENTITY.json"))


class CpuSlurm:
    def run(self, args):
        return subprocess.run(args, text=True, capture_output=True, check=False, timeout=60)

    def queue(self, user):
        result = self.run(["squeue", "--noheader", "--user", user, "--format=%i"])
        if result.returncode:
            raise RuntimeError("CPU-job queue observation failed: " + result.stderr)
        jobs = result.stdout.split()
        if len(jobs) != len(set(jobs)) or any(not job.isdigit() for job in jobs):
            raise PermissionError("Unrecognized Slurm job identifier")
        return {job: self.show(job) for job in jobs}

    def show(self, job):
        result = self.run(["scontrol", "show", "job", str(job), "--oneliner"])
        if result.returncode:
            raise RuntimeError("CPU allocation identity unavailable: " + result.stderr)
        return slurm_fields(result.stdout)

    def terminal(self, job):
        result = self.run([
            "sacct", "-X", "-n", "-P", "-j", str(job),
            "--format=JobIDRaw,State,ExitCode,User%100,Account%100,QOS%100,JobName%100",
        ])
        if result.returncode:
            raise RuntimeError("CPU accounting observation failed: " + result.stderr)
        rows = [row.split("|") for row in result.stdout.splitlines() if row.strip()]
        rows = [row for row in rows if row[0] == str(job)]
        if len(rows) != 1:
            return None
        row = rows[0]
        if len(row) < 7:
            raise PermissionError("Incomplete CPU accounting identity")
        return dict(job_id=row[0], state=row[1].split()[0].rstrip("+"), exit_code=row[2],
                    user=row[3], account=row[4], qos=row[5], name=row[6])


def verify_cpu_allocation(fields, intent, user, *, held=False):
    expected = dict(JobId=intent["job_id"], JobName=intent["job_name"],
                    Account="rose", Partition="cluster02", QOS=CPU_QOS,
                    Command=intent["script"], Comment=intent["comment"],
                    ReqNodeList=CPU_NODE, NumCPUs="4", NumNodes="1")
    for key, value in expected.items():
        if fields.get(key) != value:
            raise PermissionError("CPU allocation identity differs: " + key)
    if fields.get("UserId", "").split("(")[0] != user:
        raise PermissionError("CPU allocation owner differs")
    tres = fields.get("ReqTRES", "")
    if gpu_count(tres) != 0 or memory_gib(tres) != 24:
        raise PermissionError("CPU allocation must have no GPU and exactly 24 GiB")
    expected_time = "04:00:00" if intent["kind"] == "preflight" else "3-00:00:00"
    if fields.get("TimeLimit") != expected_time:
        raise PermissionError("CPU job time limit differs")
    if held and (fields.get("JobState"), fields.get("Reason")) != ("PENDING", "JobHeldUser"):
        raise PermissionError("CPU job was not submitted held")


@contextmanager
def job_lease(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "SUBMIT.lock").open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another submitter holds this CPU-job lease") from None
        yield


class CpuSubmitter:
    def __init__(self, root, kind, python, source_manifest, *, slurm=None, user=None):
        if kind not in KINDS:
            raise ValueError("Only preflight and controller CPU jobs are registered")
        self.root = Path(root).resolve(strict=True)
        self.code = (self.root / "code").resolve(strict=True)
        self.python = str(Path(python).resolve(strict=True))
        self.manifest = Path(source_manifest).resolve(strict=True)
        self.kind = kind
        self.slurm = slurm or CpuSlurm()
        self.user = user or getpass.getuser()
        self.directory = self.root / "deployment" / "cpu_jobs" / kind
        if not self.code.is_relative_to(self.root) or not self.manifest.is_relative_to(self.root):
            raise PermissionError("CPU source and manifest must be within this SR-F2 run")
        prior = Path(PARENT_ROOT).resolve()
        if self.root == prior or self.root.is_relative_to(prior) or prior.is_relative_to(self.root):
            raise PermissionError("CPU submission cannot target the prior experiment")

    def identity(self):
        if (self.root / "STOP").exists():
            raise PermissionError("SR-F2 STOP exists")
        return verify_source_manifest(self.manifest, self.code)

    def commands(self):
        prefix = [self.python]

        def script(name):
            return str(self.code / "scripts/sr_f2" / name)
        if self.kind == "controller":
            return [[*prefix, script("controller.py"), "--root", str(self.root),
                     "--python", self.python]]
        prior = str(Path(PARENT_ROOT))
        audit = self.root / "prior_audit"
        return [
            [*prefix, script("prepare.py"), "--root", str(self.root), "--prior-root", prior],
            [*prefix, script("audit_prior.py"), "--root", prior, "--phase", "blind-classify",
             "--output", str(audit / "blind")],
            [*prefix, script("audit_prior.py"), "--root", prior, "--phase", "rescore",
             "--blind-receipt", str(audit / "blind"), "--output", str(audit / "rescore")],
            [*prefix, script("verify_chartqa_cohort.py"), "--source",
             str(self.root / "private_inputs/audit_chartqa_sensitivity_rows.jsonl"),
             "--definition", str(self.code / "docs/sr_f2/CHARTQA_122_USER_CLARIFICATION.json"),
             "--raw-root", prior, "--output", str(audit / "chartqa122")],
            [*prefix, script("cpu_preflight.py"), "--root", str(self.root),
             "--source-manifest", str(self.manifest)],
        ]

    def validate_new_job(self):
        commands = self.commands()
        scripts = [*(Path(command[1]) for command in commands),
                   self.code / "scripts/sr_f2/submit_cpu.py"]
        for path in scripts:
            if not path.is_file():
                raise FileNotFoundError("Required deployed CPU script is absent: " + str(path))
        if self.kind == "preflight":
            for name in ("runs", "technical", "history", "diagnostics", "sealed", "released"):
                if (self.root / name).exists() and any((self.root / name).rglob("*")):
                    raise PermissionError("Fresh CPU preparation cannot follow GPU work")
            definition = self.code / "docs/sr_f2/CHARTQA_122_USER_CLARIFICATION.json"
            private = self.root / "private_inputs/audit_chartqa_sensitivity_rows.jsonl"
            if not definition.is_file() or not private.is_file():
                raise FileNotFoundError("ChartQA clarified definition/source must be deployed")
        else:
            preflight = CpuSubmitter(self.root, "preflight", self.python, self.manifest,
                                     slurm=self.slurm, user=self.user)
            with job_lease(preflight.directory):
                if not (preflight.directory / "INTENT.json").exists():
                    raise PermissionError("CPU preflight submission evidence is absent")
                result = preflight.observe(release_held=False)
            if result.get("status") != "COMPLETE":
                raise PermissionError("CPU preflight needs verified Slurm/artifact completion")
            registration = verify_registration(self.root)
            if (registration["source_manifest_sha256"] != self.identity()
                    or Path(registration["code_root"]).resolve(strict=True) != self.code):
                raise PermissionError("Controller source differs from completed preflight")
        return commands

    def check_intent(self, intent):
        expected = dict(kind=self.kind, root=str(self.root), code_root=str(self.code),
                        python=self.python, source_manifest=str(self.manifest), user=self.user)
        if any(intent.get(k) != value for k, value in expected.items()):
            raise PermissionError("Existing CPU intent belongs to different execution arguments")
        if intent.get("source_manifest_sha256") != self.identity():
            raise PermissionError("CPU registered source changed after submission intent")
        if file_hash(intent["script"]) != intent["script_sha256"]:
            raise PermissionError("CPU submitted script changed")

    def submit(self):
        with job_lease(self.directory):
            if (self.directory / "INTENT.json").exists():
                return self.observe()
            if (self.directory / "STATE.json").exists():
                raise PermissionError("CPU state exists without its submission intent")
            source_hash = self.identity()
            commands = self.validate_new_job()
            script = self.directory / "run.sbatch"
            guard = (
                "import sys; from sr_f12.evaluation import verify_source_manifest; "
                "assert verify_source_manifest(sys.argv[1], sys.argv[2]) == sys.argv[3], "
                "'Deployed CPU source changed'"
            )
            body = "#!/bin/bash\nset -euo pipefail\n"
            body += "export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1\n"
            body += "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4\n"
            body += "export PYTHONPATH=" + shlex.quote(str(self.code / "src")) + "\n"
            body += "cd " + shlex.quote(str(self.code)) + "\n"
            body += "test ! -e " + shlex.quote(str(self.root / "STOP")) + "\n"
            body += shlex.join([self.python, "-c", guard, str(self.manifest),
                                str(self.code), source_hash]) + "\n"
            for command in commands:
                if self.kind == "controller":
                    body += "test ! -e " + shlex.quote(str(self.root / "STOP")) + "\n"
                    body += shlex.join(command) + "\n"
            if self.kind == "preflight":
                body += shlex.join([
                    self.python, str(self.code / "scripts/sr_f2/submit_cpu.py"),
                    "--root", str(self.root), "--kind", "preflight", "--python", self.python,
                    "--source-manifest", str(self.manifest), "--execute-preflight",
                ]) + "\n"
            with script.open("x") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            token = object_hash([str(self.root), self.kind, source_hash])[:20]
            intent = dict(kind=self.kind, root=str(self.root), code_root=str(self.code),
                          python=self.python, source_manifest=str(self.manifest), user=self.user,
                          source_manifest_sha256=source_hash, script=str(script),
                          script_sha256=file_hash(script), commands=commands, created=now(),
                          job_name="srf2-cpu-" + self.kind,
                          comment="srf2-cpu-" + token)
            args = ["sbatch", "--hold", "--parsable", "--account=rose", "--partition=cluster02",
                    "--qos=" + CPU_QOS, "--nodelist=" + CPU_NODE, "--nodes=1", "--ntasks=1",
                    "--cpus-per-task=4", "--mem=24G",
                    "--time=" + ("04:00:00" if self.kind == "preflight" else "3-00:00:00"),
                    "--job-name=" + intent["job_name"], "--comment=" + intent["comment"],
                    "--output=" + str(self.directory / "slurm-%j.log"), str(script)]
            intent["sbatch"] = args
            save_json(self.directory / "INTENT.json", intent, exclusive=True)
            state = dict(kind=self.kind, status="SUBMISSION_UNKNOWN")
            save_json(self.directory / "STATE.json", state)
            try:
                result = self.slurm.run(args)
            except Exception as error:
                save_json(self.directory / "SUBMISSION_EXCEPTION.json",
                          dict(error=repr(error), at=now()), exclusive=True)
                raise
            save_json(self.directory / "SUBMISSION.json", dict(returncode=result.returncode,
                      stdout=result.stdout, stderr=result.stderr, at=now()), exclusive=True)
            match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?\s*", result.stdout)
            if result.returncode or not match:
                return state
            state.update(job_id=match[1], status="HELD")
            save_json(self.directory / "STATE.json", state)
            return self.release(state, intent)

    def release(self, state, intent):
        self.check_intent(intent)
        fields = self.slurm.show(state["job_id"])
        verify_cpu_allocation(fields, dict(intent, job_id=state["job_id"]), self.user, held=True)
        if (self.directory / "RELEASE_INTENT.json").exists():
            state["status"] = "RELEASE_UNKNOWN"
            save_json(self.directory / "STATE.json", state)
            return state
        save_json(self.directory / "ALLOCATION_VERIFIED.json",
                  dict(job_id=state["job_id"], fields=fields), exclusive=True)
        save_json(self.directory / "RELEASE_INTENT.json", dict(job_id=state["job_id"], at=now()),
                  exclusive=True)
        try:
            result = self.slurm.run(["scontrol", "release", state["job_id"]])
        except Exception as error:
            state["status"] = "RELEASE_UNKNOWN"
            save_json(self.directory / "STATE.json", state)
            save_json(self.directory / "RELEASE_EXCEPTION.json",
                      dict(error=repr(error), at=now()), exclusive=True)
            raise
        save_json(self.directory / "RELEASE.json", dict(job_id=state["job_id"],
                  returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                  at=now()), exclusive=True)
        state["status"] = "SUBMITTED" if result.returncode == 0 else "RELEASE_UNKNOWN"
        save_json(self.directory / "STATE.json", state)
        return state

    def observe(self, *, release_held=True):
        intent = read_json(self.directory / "INTENT.json")
        self.check_intent(intent)
        state_path = self.directory / "STATE.json"
        state = read_json(state_path) if state_path.exists() else dict(
            kind=self.kind, status="SUBMISSION_UNKNOWN")
        if state.get("kind") != self.kind:
            raise PermissionError("CPU state identity differs")
        if state["status"] == "COMPLETE":
            if self.verify_completion(state, intent) != read_json(
                self.directory / "VERIFIED_COMPLETE.json"
            ):
                raise PermissionError("Completed CPU evidence changed since acceptance")
            return state
        if state["status"] == "FAILED":
            return state
        queue = self.slurm.queue(self.user)
        job = state.get("job_id")
        if not job:
            matches = [jid for jid, row in queue.items() if row.get("Comment") == intent["comment"]]
            if len(matches) > 1:
                raise PermissionError("Multiple CPU jobs match one submission intent")
            if not matches:
                return state
            job = matches[0]
            verify_cpu_allocation(queue[job], dict(intent, job_id=job), self.user)
            state.update(job_id=job, status="HELD")
            save_json(self.directory / "RECONCILED.json", dict(job_id=job, at=now()),
                      exclusive=True)
            save_json(state_path, state)
        if job in queue:
            verify_cpu_allocation(queue[job], dict(intent, job_id=job), self.user)
            if queue[job].get("Reason") == "JobHeldUser":
                return self.release(state, intent) if release_held else state
            state["status"] = queue[job]["JobState"]
        else:
            terminal = self.slurm.terminal(job)
            if terminal is None or terminal["state"] not in TERMINAL:
                state["status"] = "ACCOUNTING_UNKNOWN"
            else:
                expected = dict(job_id=job, user=self.user, account="rose", qos=CPU_QOS,
                                name=intent["job_name"])
                if any(terminal.get(k) != value for k, value in expected.items()):
                    raise PermissionError("CPU accounting identity differs")
                save_json(self.directory / "TERMINAL.json", terminal, exclusive=True)
                if terminal["state"] == "COMPLETED" and terminal["exit_code"] == "0:0":
                    try:
                        proof = self.verify_completion(state, intent)
                    except Exception as error:
                        save_json(self.directory / "VERIFICATION_FAILURE.json",
                                  dict(error=repr(error), at=now()), exclusive=True)
                        state.update(status="FAILED", failure="ARTIFACT_VERIFICATION")
                    else:
                        save_json(self.directory / "VERIFIED_COMPLETE.json", proof, exclusive=True)
                        state["status"] = "COMPLETE"
                else:
                    state.update(status="FAILED", failure=terminal)
        save_json(state_path, state)
        return state

    def verify_completion(self, state, intent):
        allocation = read_json(self.directory / "ALLOCATION_VERIFIED.json")
        release = read_json(self.directory / "RELEASE_INTENT.json")
        if allocation.get("job_id") != state["job_id"] or release.get("job_id") != state["job_id"]:
            raise PermissionError("CPU completion was not bound before release")
        verify_cpu_allocation(allocation["fields"], dict(intent, job_id=state["job_id"]), self.user)
        registration = verify_registration(self.root)
        if (registration["source_manifest_sha256"] != intent["source_manifest_sha256"]
                or Path(registration["code_root"]).resolve(strict=True) != self.code):
            raise PermissionError("CPU completion registration differs")
        if self.kind == "preflight":
            paths = [self.root / "CPU_ACCEPTANCE.json",
                     self.root / "prior_audit/inventory/SUMMARY.json",
                     self.root / "prior_audit/blind/SUMMARY.json",
                     self.root / "prior_audit/rescore/SUMMARY.json",
                     self.root / "prior_audit/chartqa122/COHORT_VERIFICATION.json"]
            for path in paths:
                if read_json(path).get("status") != "PASS":
                    raise PermissionError("CPU preflight evidence did not pass: " + str(path))
            cohort = read_json(paths[-1])
            if (cohort.get("count") != 122
                    or cohort.get("cohort_identity_verified") is not True
                    or cohort.get("raw_output_count") != 1586
                    or cohort.get("raw_evidence_status") != "RAW_AND_GOLD_REVERIFIED"):
                raise PermissionError("ChartQA cohort lacks all 122 reverified raw/gold records")
            worker = read_json(self.directory / "WORKER_COMPLETE.json")
            if (worker.get("status") != "COMPLETE"
                    or worker.get("source_manifest_sha256") != intent["source_manifest_sha256"]
                    or worker.get("registration_sha256") != object_hash(registration)):
                raise PermissionError("Bound CPU preflight execution did not complete")
            paths.append(self.directory / "WORKER_COMPLETE.json")
            artifacts = [dict(path=str(p.relative_to(self.root)), sha256=file_hash(p))
                         for p in paths]
        else:
            done = read_json(self.root / "released/COMPLETE.json")
            if (done.get("status") != "COMPLETE_COVERAGE_VERIFIED"
                    or done.get("experiment_complete") is not True):
                raise PermissionError("Controller exited without complete report delivery")
            artifacts = done.get("files", [])
            if not artifacts:
                raise PermissionError("Report delivery has no file evidence")
            names = set()
            for ref in artifacts:
                relative = Path(ref["path"])
                path = (self.root / relative).resolve(strict=True)
                if (relative.is_absolute() or ".." in relative.parts
                        or not path.is_relative_to(self.root) or str(relative) in names
                        or file_hash(path) != ref["sha256"] or path.stat().st_size != ref["bytes"]):
                    raise PermissionError("Report delivery path, size or hash differs")
                names.add(str(relative))
            expected = {"released/" + name for name in (
                "AUTHORIZATION.json", "SCORES.jsonl", "OLD_SCORES.jsonl",
                "DIAGNOSTIC_SCORES.jsonl", "results.json", "REPORT.md", "trajectories.png",
            )}
            if names != expected:
                raise PermissionError("Controller completion lacks required report deliverables")
        return dict(kind=self.kind, job_id=state["job_id"], status="COMPLETE",
                    registration_sha256=object_hash(registration), artifacts=artifacts)

    def execute_preflight(self):
        """CPU-only ordered work: reuse verified PASS phases, preserve every failure."""
        if self.kind != "preflight":
            raise PermissionError("Only the preflight allocation may execute CPU phases")
        state = read_json(self.directory / "STATE.json")
        intent = read_json(self.directory / "INTENT.json")
        self.check_intent(intent)
        if not state.get("job_id") or state["job_id"] != os.environ.get("SLURM_JOB_ID"):
            raise PermissionError("Preflight must execute inside its bound Slurm job")
        verify_cpu_allocation(self.slurm.show(state["job_id"]),
                              dict(intent, job_id=state["job_id"]), self.user)
        commands = self.commands()
        receipts = []

        def execute(command):
            if (self.root / "STOP").exists():
                raise PermissionError("SR-F2 STOP exists")
            result = subprocess.run(command, check=False)
            if result.returncode:
                raise RuntimeError("CPU phase failed; outputs preserved without retry")

        if not (self.root / "INPUT_IDENTITY.json").exists():
            execute(commands[0])
        receipts.append(verify_prepared_inputs(self.root))
        for command, directory, phase in zip(
            commands[1:4], ("blind", "rescore", "chartqa122"),
            ("blind-classify", "rescore", "chartqa122"), strict=True,
        ):
            output = self.root / "prior_audit" / directory
            if not output.exists():
                execute(command)
            receipts.append(verify_phase(output, phase, self.code))
        # Its reuse_cpu_acceptance validates the exact source, all three test commands,
        # original logs and native prompts before reusing a previous successful check.
        execute(commands[4])
        registration = verify_registration(self.root)
        return save_json(self.directory / "WORKER_COMPLETE.json",
                         dict(status="COMPLETE", source_manifest_sha256=self.identity(),
                              registration_sha256=object_hash(registration), phases=receipts),
                         exclusive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=KINDS)
    parser.add_argument("--python", required=True)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--execute-preflight", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    submitter = CpuSubmitter(args.root, args.kind, args.python, args.source_manifest)
    result = submitter.execute_preflight() if args.execute_preflight else submitter.submit()
    print(json.dumps(result, sort_keys=True))
    failed = ("FAILED", "SUBMISSION_UNKNOWN", "RELEASE_UNKNOWN", "ACCOUNTING_UNKNOWN")
    if result["status"] in failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
