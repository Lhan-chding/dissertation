"""Bind the two MAIN GPU jobs to disjoint subsets of the frozen request manifest.

Each job owns one starting seed. Endpoints follow the existing controller order
within that seed, and requests within each endpoint retain the manifest order.
The plan contains identities only and never grants terminal admission.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


STARTS = (11031, 11032)
DESIGN_FREEZE_SHA256 = "6c4b948c533fdd736e089bf3836378f4dbcb76ac66f1d480fa979eaca56c84e0"
MAIN_MANIFEST_SHA256 = "cf4c4fd307d90de3475c5e01885b92196dbf0e481c84461f626246befd77d884"
FULL_PANELS = {"list8": 768, "swap8": 128, "authority8": 384, "prefix10": 576, "retention8": 96}
PRIMARY_PANEL = {"list8": 256}


def lane_endpoints(start_seed):
    """Original allocation order restricted to a single starting seed."""
    if type(start_seed) is not int or start_seed not in STARTS:
        raise ValueError("Unknown MAIN starting seed")
    return ((start_seed, "START", 0),) + tuple(
        (start_seed, arm, step) for arm in ("T0", "T1") for step in (8, 16, 24)
    )


def object_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _historical_seed(row):
    # Legacy srf6_ref.core.seed_for; matching draws intentionally share this
    # seed across endpoints and views. Request IDs, not RNG seeds, are unique.
    parts = [2026110301, "eval", "b0", row["panel"], row["root_id"], row["draw"]]
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _partition(rows):
    groups = defaultdict(list)
    manifest_ids = set()
    for row in rows:
        start, arm, step = row["start_seed"], row["arm"], row["step"]
        if (row["split"] != "MAIN" or type(start) is not int or type(step) is not int
                or type(row["draw"]) is not int or type(row["seed"]) is not int):
            raise ValueError("Invalid MAIN request identity")
        request = row["request_id"]
        expected = (f"SAT1/MAIN/{start}/{arm}/s{step:02d}/{row['panel']}/"
                    f"{row['root_id']}/{row['view']}/d{row['draw']:02d}")
        if request != expected or request in manifest_ids:
            raise ValueError("Misassigned or repeated MAIN request ID: " + str(request))
        if row["seed"] != _historical_seed(row):
            raise ValueError("MAIN draw seed differs from registered pairing: " + request)
        groups[start, arm, step].append(row)
        manifest_ids.add(request)

    expected_endpoints = {ep for start in STARTS for ep in lane_endpoints(start)}
    if set(groups) != expected_endpoints:
        raise ValueError("Missing or unexpected MAIN endpoint")
    lanes = {}
    assigned = set()
    for start in STARTS:
        endpoints = []
        ordered_lane_ids = []
        for endpoint in lane_endpoints(start):
            _, arm, step = endpoint
            selected = groups[endpoint]
            ids = [r["request_id"] for r in selected]
            panels = Counter(r["panel"] for r in selected)
            if panels != (FULL_PANELS if step in (0, 24) else PRIMARY_PANEL):
                raise ValueError(f"MAIN panel or response allocation differs at {endpoint}")
            endpoints.append(dict(start_seed=start, arm=arm, step=step, responses=len(ids),
                                  request_ids_sha256=object_sha(ids), rows_sha256=object_sha(selected),
                                  panels=dict(panels)))
            ordered_lane_ids.extend(ids)
        lane_ids = set(ordered_lane_ids)
        if len(lane_ids) != 6880 or len(ordered_lane_ids) != 6880 or assigned & lane_ids:
            raise ValueError("MAIN lane allocation is incomplete or overlapping")
        assigned.update(lane_ids)
        lanes[str(start)] = dict(responses=len(ordered_lane_ids),
                                 request_ids_sha256=object_sha(ordered_lane_ids), endpoints=endpoints)

    if assigned != manifest_ids or len(assigned) != 13760:
        raise ValueError("MAIN lanes do not cover the exact original request set")
    return dict(schema="sr-satdose-parallel-main-plan-v1",
                scope="FROZEN_MAIN_REQUEST_COVERAGE_ONLY", design_freeze_sha256=DESIGN_FREEZE_SHA256,
                main_manifest_sha256=MAIN_MANIFEST_SHA256, unique_requests=len(assigned),
                endpoint_count=len(groups), overlapping_requests=0, omitted_requests=0,
                manifest_request_ids_sha256=object_sha([r["request_id"] for r in rows]),
                lanes=lanes)


def build_plan(main_manifest_path):
    """Read only frozen bytes. Path may be a local copy; its SHA must match the freeze."""
    path = Path(main_manifest_path).absolute()
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError("Noncanonical MAIN manifest path")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MAIN_MANIFEST_SHA256:
        raise ValueError("MAIN manifest differs from frozen SHA-256")
    return _partition([json.loads(line) for line in raw.splitlines()])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_plan(args.manifest), sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
