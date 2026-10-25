"""Bounded read-only activity checks for the exact project cleanup candidates."""

import json
import os
import socket
import subprocess
from pathlib import Path

from sr_f4.common import now, read, write
from sr_f4.pool import Slurm

ROOT = Path("/projects/_ssd/varunssd/louis-ssvc")
SOURCE = ROOT / "sr_f3_ce_protocol_fix_20261016/selected_v2"
CACHE = ROOT / "cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct"
AUDIT = ROOT / "storage_audits/cleanup_20261025_project_only"


def targeted(path):
    path = Path(str(path).removesuffix(" (deleted)"))
    if path == CACHE or CACHE in path.parents:
        return True
    try:
        rel = path.relative_to(SOURCE)
    except ValueError:
        return False
    parts = rel.parts
    return bool(parts) and (
        parts[0] in ("monitor", "engine")
        or (len(parts) >= 3 and parts[0] == "runs" and parts[2] in ("rollouts", "support"))
        or str(rel) in ("release/ENDPOINT_RAW.jsonl.gz", "release/ENDPOINT_SCORES.jsonl")
    )


def main():
    if ROOT.resolve(strict=True) != ROOT:
        raise PermissionError("Project root is not canonical")
    slurm = Slurm()
    import getpass

    ids = slurm.user_job_ids(getpass.getuser())
    jobs = {key: slurm.show(key) for key in sorted(ids)}
    conflicts = [
        key
        for key, row in jobs.items()
        if any(targeted(row.get(field, "")) for field in ("WorkDir", "Command"))
    ]
    handles, unavailable, session_daemons = [], [], []
    process_count = 0
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            process_count += 1
            for fd in (proc / "fd").iterdir():
                try:
                    destination = os.readlink(fd)
                    if targeted(destination):
                        handles.append(dict(pid=int(proc.name), fd=fd.name, path=destination))
                except FileNotFoundError:
                    pass
            for line in (proc / "maps").read_text().splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and fields[5].startswith(str(ROOT)) and targeted(fields[5]):
                    handles.append(dict(pid=int(proc.name), mapping=True, path=fields[5]))
        except FileNotFoundError:
            continue
        except PermissionError:
            # These login session services are not experiment workers. Their
            # protected maps are inaccessible even to their owning account.
            identity = subprocess.run(
                ["ps", "-p", proc.name, "-o", "comm=,args="],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip().split(maxsplit=1)
            if identity in (["(sd-pam)", "(sd-pam)"], ["sshd", f"sshd: {getpass.getuser()}@notty"]):
                session_daemons.append(dict(pid=int(proc.name), identity=identity))
            elif (proc / "status").exists():
                unavailable.append(int(proc.name))
    config = read(ROOT / "sr_f5_semantic_j_20261025/CONFIG.json")
    protected = []
    paths = [Path(config["model"]["model_path"]) / "config.json"]
    for adapter in config["model"]["parents"].values():
        paths.extend(
            Path(adapter) / name for name in ("adapter_config.json", "adapter_model.safetensors")
        )
    paths.extend(
        [
            ROOT / "sr_f4_consolidated_20261022/DATA/TASKS.jsonl",
            ROOT / "sr_f4_consolidated_20261022/DATA/IMAGE_MANIFEST.jsonl",
            ROOT / "sr_f4_consolidated_20261022/DATA/CHARTQA_IMAGES.jsonl",
            ROOT / "sr_f4_c_local_pilot_20261024/DATA/TASKS.jsonl",
            SOURCE / "external/chartqa/TASKS_AUDIT_ONLY.jsonl",
            ROOT / "envs/ssvc-py312/bin/python3.12",
        ]
    )
    for path in paths:
        stat = path.stat()
        protected.append(
            dict(
                path=str(path),
                bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                inode=stat.st_ino,
                device=stat.st_dev,
            )
        )
    write(AUDIT / "PROTECTED_BEFORE.json", protected)
    report = dict(
        at=now(),
        status="PASS" if not (handles or conflicts or unavailable) else "BLOCKED",
        project_root=str(ROOT),
        checked_roots=[str(SOURCE), str(CACHE)],
        candidate_filter="archive allowlisted F3 output prefixes plus obsolete Qwen2.5 cache only",
        open_handles=handles,
        inaccessible_owned_processes=unavailable,
        excluded_login_session_daemons=session_daemons,
        local_processes_checked=process_count,
        host=socket.gethostname(),
        activity_check_scope=("Current login descriptors/maps plus user-wide Slurm jobs"),
        remote_compute_process_visibility=("NOT_AVAILABLE; SSH denied, no job steps launched"),
        active_job_evidence=dict(
            status="PASS" if not conflicts else "BLOCKED",
            project_root=str(ROOT),
            conflicting_jobs=conflicts,
            jobs=jobs,
            source="actual squeue/scontrol and local process/open-handle checks",
        ),
        scientific_dependencies_protected=True,
        limitation=(
            "No assertion of all-host exclusive access; completed F3 files require "
            "byte match before unlink; model cache is regenerable"
        ),
    )
    write(AUDIT / "OPEN_HANDLES.json", report)
    print(
        json.dumps(
            {k: report[k] for k in ("status", "open_handles", "inaccessible_owned_processes")}
        )
    )
    if report["status"] != "PASS":
        raise RuntimeError("Cleanup preflight blocked")


if __name__ == "__main__":
    main()
