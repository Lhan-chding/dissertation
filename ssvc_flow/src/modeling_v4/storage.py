"""Read filesystem space and inherited Ceph quotas before a phase or large write."""

from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path


def storage_status(path):
    path = Path(path).resolve()
    while not path.exists() and path != path.parent:
        path = path.parent
    free = shutil.disk_usage(path).free
    limits, unavailable = [], []
    read_attribute = getattr(os, "getxattr", None)
    if read_attribute is None:
        unavailable.append({"path": str(path), "error": "QUOTA_XATTR_API_UNAVAILABLE"})
    # A missing xattr on the leaf never implies that ancestor quotas are absent.
    for directory in (path, *path.parents):
        if read_attribute is None:
            break
        try:
            limit = int(read_attribute(directory, "ceph.quota.max_bytes"))
            if limit <= 0:
                continue
            used = int(read_attribute(directory, "ceph.dir.rbytes"))
            if used < 0:
                raise ValueError("Negative Ceph directory usage")
            limits.append(
                {
                    "path": str(directory),
                    "limit_bytes": limit,
                    "used_bytes": used,
                    "available_bytes": max(0, limit - used),
                }
            )
        except OSError as exc:
            no_attribute = {errno.ENODATA, errno.ENOTSUP, getattr(errno, "ENOATTR", errno.ENODATA)}
            if exc.errno not in no_attribute:
                unavailable.append({"path": str(directory), "errno": exc.errno})
        except ValueError:
            unavailable.append({"path": str(directory), "error": "INVALID_QUOTA_ATTRIBUTE"})
    return {
        "path": str(path),
        "filesystem_free_bytes": free,
        "ancestor_quotas": limits,
        "available_bytes": min([free, *[r["available_bytes"] for r in limits]]),
        "unreadable_quota_attributes": unavailable,
        "status": "MEASURED" if not unavailable else "PARTIAL_QUOTA_READ",
        "reservation_created": False,
    }


def require_space(path, required_bytes):
    if type(required_bytes) is not int or required_bytes < 0:
        raise ValueError("Actual required bytes must be a nonnegative integer")
    result = storage_status(path)
    if result["available_bytes"] < required_bytes:
        raise OSError(
            errno.ENOSPC,
            f"Need {required_bytes} bytes, measured available {result['available_bytes']}; "
            "preserve raw data and move scratch or provide storage before continuing",
            str(path),
        )
    return {**result, "required_bytes": required_bytes}
