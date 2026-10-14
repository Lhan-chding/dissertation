#!/usr/bin/env python3
"""Audit preserved SR-F2 artifacts using original frozen definitions, without GPU work."""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", "--original-root", dest="root", required=True, type=Path)
    parser.add_argument("--frozen-code-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    original = args.root.resolve(strict=True)
    frozen = args.frozen_code_root.resolve(strict=True)
    extension = Path(__file__).resolve().parents[2]
    # Recovery is a separate namespace; all scientific imports come from frozen code.
    sys.path[:0] = [str(frozen / "src"), str(extension / "src")]
    import sr_f2.orchestration
    from sr_f2_recovery.audit import audit_recovery

    if Path(sr_f2.orchestration.__file__).resolve() != frozen / "src/sr_f2/orchestration.py":
        raise PermissionError("Scientific definitions did not load from frozen code")
    registration = json.loads((original / "REGISTRATION.json").read_text())
    if Path(registration["code_root"]).resolve(strict=True) != frozen:
        raise PermissionError("Requested frozen code root differs from original registration")
    output = args.output.absolute()
    if output.exists():
        raise FileExistsError("Audit output already exists; preserve original audit")
    result = audit_recovery(original)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "audit_path": str(output),
                "counts": result["counts"],
                "scientific_scores_read": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
