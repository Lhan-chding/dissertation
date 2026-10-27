"""Losslessly compact one stopped experiment's copied raw evidence.

The CLI has a fixed project boundary and a fixed source directory. It first
creates and verifies an archive, and only a separate --apply invocation may
unlink its exact unchanged members. No model, checkpoint or unrelated directory
is accepted as a source. All source directories remain in place.
"""

from __future__ import annotations

import argparse
import fcntl
import getpass
import gzip
import hashlib
import json
import os
import re
import socket
import stat
import subprocess
import tarfile
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path("/projects/_ssd/varunssd/louis-ssvc")
SOURCE = "sr_f3_ce_20261016/prior/raw"
AUDIT = "storage_audits/cleanup_20261027_srf3_prior_raw"
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def now():
    return datetime.now(timezone.utc).isoformat()


def parts(relative):
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("Invalid relative path")
    fields = relative.split("/")
    if PurePosixPath(relative).is_absolute() or any(p in ("", ".", "..") for p in fields):
        raise ValueError("Unsafe relative path")
    return fields


def source_member(relative):
    fields = parts(relative)
    prefix = parts(SOURCE)
    if fields[: len(prefix)] != prefix or len(fields) <= len(prefix):
        raise PermissionError("Outside the exact registered raw directory")
    return relative


def canonical_root(root):
    root = Path(root)
    if not root.is_absolute() or root.resolve(strict=True) != root:
        raise PermissionError("Project root must be canonical")
    for path in (*reversed(root.parents), root):
        if stat.S_ISLNK(path.lstat().st_mode):
            raise PermissionError("Symlink in project ancestors")
    return root


def make_audit_directory(root):
    """Create only through pinned non-symlink directory descriptors."""
    current = os.open(canonical_root(root), DIR_FLAGS)
    try:
        for name in parts(AUDIT):
            with suppress(FileExistsError):
                os.mkdir(name, dir_fd=current)
            following = os.open(name, DIR_FLAGS, dir_fd=current)
            os.close(current)
            current = following
    finally:
        os.close(current)
    return Path(root) / AUDIT


@contextmanager
def parent_fd(root, relative):
    fields = parts(relative)
    current = os.open(canonical_root(root), DIR_FLAGS)
    try:
        for item in fields[:-1]:
            following = os.open(item, DIR_FLAGS, dir_fd=current)
            os.close(current)
            current = following
        yield current, fields[-1]
    finally:
        os.close(current)


def metadata(value):
    return {
        key: getattr(value, "st_" + key)
        for key in ("dev", "ino", "mode", "size", "mtime_ns", "ctime_ns", "nlink", "blocks")
    }


def same_source_stat(root, expected, current):
    """Translate only mount-local st_dev; every other recorded field must match.

    The project is mounted at the same canonical path on several login hosts.
    Their device numbers differ even though the file inode and all byte-change
    indicators are the same. The current file must be on the current root mount.
    Content identity remains the previously verified member SHA256.
    """
    return current["dev"] == Path(root).stat().st_dev and {
        key: value for key, value in current.items() if key != "dev"
    } == {key: value for key, value in expected.items() if key != "dev"}


def unchanged_inventory(root, entries):
    observed = inventory(root)
    return len(observed) == len(entries) and all(
        live["path"] == archived["path"] and same_source_stat(root, archived["stat"], live["stat"])
        for live, archived in zip(observed, entries, strict=True)
    )


@contextmanager
def regular_file(root, relative):
    with parent_fd(root, relative) as (directory, name):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise PermissionError("Only regular files are allowed")
            yield stream, directory, name


def digest(stream):
    result = hashlib.sha256()
    while chunk := stream.read(8 << 20):
        result.update(chunk)
    return result.hexdigest()


def write_once(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def inventory(root):
    root = canonical_root(root)
    source = root / SOURCE
    if source.resolve(strict=True) != source:
        raise PermissionError("Source is not canonical")
    entries = []
    for directory, dirs, files in os.walk(source, followlinks=False):
        for name in sorted(dirs):
            if not stat.S_ISDIR((Path(directory) / name).lstat().st_mode):
                raise PermissionError("Symlink or non-directory inside source")
        for name in sorted(files):
            path = Path(directory) / name
            relative = source_member(str(path.relative_to(root)))
            value = path.lstat()
            if not stat.S_ISREG(value.st_mode):
                raise PermissionError("Nonregular source member")
            entries.append({"path": relative, "stat": metadata(value)})
    if not entries:
        raise ValueError("Empty source is not a cleanup candidate")
    return sorted(entries, key=lambda row: row["path"])


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, check=True)
    return result.stdout


def preflight(root):
    """Actual scheduler state and current-host handles, with explicit limits."""
    root = canonical_root(root)
    stop = json.loads((root / "sr_f3_ce_20261016/STOP").read_text())
    if stop.get("status") != "STOPPED_PROTOCOL_FAILURE":
        raise PermissionError("The source experiment is not the registered stopped run")
    target = str(root / SOURCE)
    ids = command(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i"]).split()
    jobs = {job: command(["scontrol", "show", "job", "-o", job]) for job in ids}
    conflicts = [job for job, value in jobs.items() if target in value]
    # No unrelated job is changed, even when it shares the Unix account/QoS.
    configs = {}
    for relative in (
        "sr_f5_semantic_j_20261025/CONFIG.json",
        "sr_f5_semantic_j_recipe_v2_20261026/CONFIG.json",
    ):
        payload = (root / relative).read_bytes()
        if target in payload.decode():
            raise PermissionError("Current configured dependency references cleanup source")
        configs[relative] = hashlib.sha256(payload).hexdigest()
    handles = []
    inaccessible = []
    excluded_daemons = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            for fd in (proc / "fd").iterdir():
                try:
                    location = os.readlink(fd)
                    if location == target or location.startswith(target + "/"):
                        handles.append({"pid": int(proc.name), "fd": fd.name, "path": location})
                except FileNotFoundError:
                    pass
            for line in (proc / "maps").read_text().splitlines():
                if target in line:
                    handles.append({"pid": int(proc.name), "mapping": line})
        except FileNotFoundError:
            pass
        except PermissionError:
            identity = subprocess.run(
                ["ps", "-p", proc.name, "-o", "comm=,args="],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            if identity.startswith("(sd-pam)") or re.fullmatch(r"sshd\s+sshd: \S+@notty", identity):
                excluded_daemons.append({"pid": int(proc.name), "identity": identity})
            elif (proc / "status").exists():
                inaccessible.append({"pid": int(proc.name), "identity": identity})
    result = {
        "at": now(),
        "hostname": socket.gethostname(),
        "source": target,
        "stop": stop,
        "jobs": jobs,
        "conflicting_jobs": conflicts,
        "current_f5_config_hashes": configs,
        "open_handles": handles,
        "inaccessible_owned_processes": inaccessible,
        "excluded_login_daemons": excluded_daemons,
        "scope": "Current login /proc plus all user Slurm jobs; not all-host process visibility",
        "status": "PASS" if not (conflicts or handles or inaccessible) else "BLOCKED",
    }
    if result["status"] != "PASS":
        raise PermissionError(json.dumps(result))
    return result


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()

    def read(self, size=-1):
        chunk = self.stream.read(size)
        self.hash.update(chunk)
        return chunk


def archive_sources(root, destination, entries):
    """Every archive member is a regular file with a safe unique project path."""
    seen = set()
    output = []
    with Path(destination).open("xb") as raw:
        with (
            gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as compressed,
            tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive,
        ):
            for index, entry in enumerate(entries):
                relative = source_member(entry["path"])
                if relative in seen:
                    raise ValueError("Duplicate source member")
                seen.add(relative)
                with regular_file(root, relative) as (stream, directory, name):
                    before = metadata(os.fstat(stream.fileno()))
                    if before != entry["stat"]:
                        raise PermissionError("Source changed before archive: " + relative)
                    info = tarfile.TarInfo(relative)
                    info.size = before["size"]
                    info.mode = stat.S_IMODE(before["mode"])
                    info.mtime = before["mtime_ns"] / 1e9
                    reader = HashingReader(stream)
                    archive.addfile(info, reader)
                    after = metadata(os.fstat(stream.fileno()))
                    current = metadata(os.stat(name, dir_fd=directory, follow_symlinks=False))
                    if after != before or current != before:
                        raise PermissionError("Source changed while archiving: " + relative)
                    output.append({**entry, "sha256": reader.hash.hexdigest()})
                if (index + 1) % 5000 == 0:
                    print(json.dumps({"stage": "ARCHIVING", "files": index + 1}), flush=True)
        raw.flush()
        os.fsync(raw.fileno())
    return output


def verify_archive(path, entries):
    expected = {row["path"]: row for row in entries}
    if len(expected) != len(entries):
        raise ValueError("Duplicate manifest member")
    seen = set()
    total = 0
    with gzip.open(path, "rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r|") as archive:
            for member in archive:
                name = source_member(member.name)
                if name in seen or name not in expected or not member.isfile():
                    raise PermissionError("Unexpected, duplicate or nonregular archive member")
                row = expected[name]
                if member.size != row["stat"]["size"]:
                    raise ValueError("Archive member size differs")
                stream = archive.extractfile(member)
                if stream is None or digest(stream) != row["sha256"]:
                    raise ValueError("Archive member SHA256 differs")
                seen.add(name)
                total += member.size
        # tar EOF alone does not prove the gzip trailer CRC/read-to-EOF check.
        while compressed.read(8 << 20):
            pass
    if seen != set(expected):
        raise ValueError("Archive has missing members")
    with Path(path).open("rb") as stream:
        archive_hash = digest(stream)
    return {
        "status": "PASS",
        "members": len(seen),
        "source_bytes": total,
        "archive_bytes": Path(path).stat().st_size,
        "archive_sha256": archive_hash,
        "gzip_crc_read_to_eof": True,
        "all_member_sha256": True,
    }


def delete_member(root, entry):
    """Reuse the verified archive digest; stable source identity guards the unlink."""
    relative = source_member(entry["path"])
    with regular_file(root, relative) as (stream, directory, name):
        before = metadata(os.fstat(stream.fileno()))
        if not same_source_stat(root, entry["stat"], before):
            raise PermissionError("Source changed before unlink: " + relative)
        if (
            metadata(os.fstat(stream.fileno())) != before
            or metadata(os.stat(name, dir_fd=directory, follow_symlinks=False)) != before
        ):
            raise PermissionError("Source changed during final check")
        os.unlink(name, dir_fd=directory)
        return {
            "path": relative,
            "status": "DELETED_ARCHIVED_SOURCE",
            "sha256": entry["sha256"],
            "logical_bytes": before["size"],
            "lastlink_allocated_bytes": before["blocks"] * 512 if before["nlink"] == 1 else 0,
        }


def cached_archive_verification(destination, entries):
    """Reuse completed verification if the archive predates its proof unchanged.

    The proof is created only after full gzip EOF/CRC and member SHA validation.
    This avoids hashing gigabytes again during the deletion stage. A changed
    archive or source fails closed; it is never automatically re-certified.
    """
    destination = Path(destination)
    proof_path = destination / "PREDELETE_PROOF.json"
    proof = json.loads(proof_path.read_text())
    original = json.loads((destination / "ARCHIVE_VERIFIED.json").read_text())
    value = (destination / "RAW_COMPLETE.tar.gz").lstat()
    if (
        not stat.S_ISREG(value.st_mode)
        or proof.get("status") != "PASS"
        or not isinstance(proof.get("hostname"), str)
        or proof.get("archive") != original
        or original.get("status") != "PASS"
        or original.get("gzip_crc_read_to_eof") is not True
        or original.get("all_member_sha256") is not True
        or original.get("members") != len(entries)
        or original.get("source_bytes") != sum(row["stat"]["size"] for row in entries)
        or value.st_size != original.get("archive_bytes")
        or max(value.st_mtime_ns, value.st_ctime_ns) > proof_path.stat().st_mtime_ns
    ):
        raise PermissionError("No unchanged fully verified archive available")
    return original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = canonical_root(ROOT)
    destination = make_audit_directory(root)
    lock_fd = os.open(destination / "EXECUTION.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock_fd, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evidence = preflight(root)
        if not args.apply:
            write_once(destination / "PREFLIGHT.json", evidence)
            entries = inventory(root)
            write_once(destination / "INVENTORY.json", entries)
            entries = archive_sources(root, destination / "RAW_COMPLETE.tar.gz", entries)
            write_once(destination / "ALLOWLIST.json", entries)
            check = verify_archive(destination / "RAW_COMPLETE.tar.gz", entries)
            write_once(destination / "ARCHIVE_VERIFIED.json", check)
            write_once(
                destination / "PREDELETE_PROOF.json",
                {
                    "status": "PASS",
                    "hostname": socket.gethostname(),
                    "archive": check,
                    "source_files": len(entries),
                    "source_bytes": check["source_bytes"],
                    "stop_status": evidence["stop"]["status"],
                    "source_stat_guard": (
                        "Each source checked around archive reads; checked again before unlink"
                    ),
                },
            )
            print(json.dumps(check), flush=True)
            return
        archive_preflight = json.loads((destination / "PREFLIGHT.json").read_text())
        evidence["archive_hostname"] = archive_preflight["hostname"]
        evidence["current_project_mount_device"] = root.stat().st_dev
        evidence["identity_comparison"] = (
            "same canonical paths and all stat fields except mount-local st_dev"
        )
        write_once(destination / "APPLY_PREFLIGHT.json", evidence)
        entries = json.loads((destination / "ALLOWLIST.json").read_text())
        actual = cached_archive_verification(destination, entries)
        # Validate the full inventory before the first unlink, not just a prefix.
        if not unchanged_inventory(root, entries):
            raise PermissionError("Source inventory changed since archive")
        logical = allocated = count = 0
        with (destination / "DELETE_JOURNAL.jsonl").open("x") as journal:
            for row in entries:
                result = delete_member(root, row)
                journal.write(json.dumps(result) + "\n")
                journal.flush()
                logical += result["logical_bytes"]
                allocated += result["lastlink_allocated_bytes"]
                count += 1
                if count % 5000 == 0:
                    os.fsync(journal.fileno())
                    print(json.dumps({"stage": "DELETING_VERIFIED", "files": count}), flush=True)
            os.fsync(journal.fileno())
        remaining = [str(p) for p in (root / SOURCE).rglob("*") if p.is_file() or p.is_symlink()]
        receipt = {
            "status": "COMPLETE" if not remaining else "INCOMPLETE",
            "at": now(),
            "source": SOURCE,
            "archive": str(destination / "RAW_COMPLETE.tar.gz"),
            "deleted_files": count,
            "logical_bytes_removed": logical,
            "lastlink_allocated_bytes_released": allocated,
            "archive_bytes": actual["archive_bytes"],
            "net_logical_bytes_saved": logical - actual["archive_bytes"],
            "remaining_non_directories": remaining,
            "archive_verification": actual,
            "note": (
                "Allocated blocks are estimates, not a shared filesystem free-space delta; "
                "audit files consume additional bytes."
            ),
        }
        write_once(destination / "APPLY_RECEIPT.json", receipt)
        write_once(root / SOURCE / "ARCHIVED_LOCATION.json", receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
