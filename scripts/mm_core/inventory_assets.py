#!/usr/bin/env python3
"""Create a bounded historical asset inventory without inference or extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mm_core.inventory import inventory_assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--server-evidence", type=Path)
    parser.add_argument("--model-receipt", type=Path)
    args = parser.parse_args()
    archives = {}
    for value in args.archive:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in archives:
            parser.error("each archive must have a unique nonempty NAME=PATH")
        archives[name] = Path(path)
    result = inventory_assets(
        args.repo_root,
        args.output_dir,
        archives=archives,
        server_evidence=args.server_evidence,
        model_receipt=args.model_receipt,
    )
    print(f"{len(result['contracts'])} contracts; selected path {result['selected_path']}")


if __name__ == "__main__":
    main()
