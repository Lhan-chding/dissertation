#!/usr/bin/env python3
"""Select one sealed CPU request by its exact Slurm array index.

The list is a JSON array of {"path": ..., "sha256": ...} entries. Relative
request paths resolve beside the list. Requests are never edited or combined;
the existing CPU request runner performs its own hash and command checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def _index(value, name):
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        return int(value)
    raise ValueError(f"{name} must be a nonnegative ASCII integer index")


def _sha256(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("sha256 must contain exactly 64 lowercase hexadecimal digits")
    return value


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate list entry field")
        result[key] = value
    return result


def select_request(request_list, sha256, index, task_id):
    """Read-only validation, usable by tiny fixtures without a Slurm allocation."""
    index = _index(index, "explicit index")
    if index != _index(task_id, "Slurm array task"):
        raise ValueError("explicit index differs from Slurm array task index")
    path = Path(request_list).resolve()
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _sha256(sha256):
        raise ValueError("CPU request list hash mismatch")
    rows = json.loads(payload, object_pairs_hook=_unique_keys)
    if not isinstance(rows, list) or not rows:
        raise ValueError("CPU request list must be a nonempty ordered JSON array")
    normalized, seen_paths, seen_hashes = [], set(), set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError("CPU request list entry requires exactly path and sha256")
        if not isinstance(row["path"], str) or not row["path"].strip() or "\x00" in row["path"]:
            raise ValueError("CPU request entry path must be a nonempty filesystem path")
        request = Path(row["path"])
        request = (request if request.is_absolute() else path.parent / request).resolve()
        request_hash = _sha256(row["sha256"])
        if request in seen_paths or request_hash in seen_hashes:
            raise ValueError("duplicate request path or payload in CPU array list")
        seen_paths.add(request)
        seen_hashes.add(request_hash)
        normalized.append((request, request_hash))
    if index >= len(normalized):
        raise ValueError("CPU request array index is out of range")
    request, request_hash = normalized[index]
    if hashlib.sha256(request.read_bytes()).hexdigest() != request_hash:
        raise ValueError("selected CPU request hash mismatch")
    return {
        "request_list": str(path),
        "request_list_sha256": sha256,
        "request_count": len(normalized),
        "request_index": index,
        "request": str(request),
        "request_sha256": request_hash,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-list", required=True)
    parser.add_argument("--sha256", required=True, help="SHA256 of the complete ordered list")
    parser.add_argument("--index", required=True)
    args = parser.parse_args(argv)
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if sys.platform != "linux" or re.fullmatch(r"[0-9]+", job_id) is None:
        raise ValueError("CPU array execution requires Linux and a numeric Slurm job ID")
    if any(os.environ.get(name) != "" for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")):
        raise ValueError("CPU array execution requires empty GPU visibility variables")
    selected = select_request(
        args.request_list, args.sha256, args.index, os.environ.get("SLURM_ARRAY_TASK_ID")
    )
    print(
        json.dumps(
            {
                "schema": "ssvc-v3-cpu-array-dispatch-1",
                **selected,
                "slurm_job_id": job_id,
                "slurm_array_task_id": os.environ["SLURM_ARRAY_TASK_ID"],
                "requests_modified": False,
            }
        ),
        flush=True,
    )
    wrapper = Path(__file__).resolve().with_name("run_modeling_v3_cpu_request.py")
    result = subprocess.run(
        [
            sys.executable,
            str(wrapper),
            "--request",
            selected["request"],
            "--sha256",
            selected["request_sha256"],
        ],
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from None
