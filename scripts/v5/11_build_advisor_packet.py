#!/usr/bin/env python3
"""Build the local advisor packet from completed v5 result artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compensability_v5.evaluation.build_v5_tables import build_advisor_packet  # noqa: E402


def _fixture_results() -> dict[str, object]:
    return {
        "support_results": {"complete": True},
        "confirmation_results": {"complete": True},
        "reward_results": {"complete": True},
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_optional_result(path: Path | None, label: str) -> dict[str, object] | None:
    if path is None or not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink JSON file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fixture-dry-run", action="store_true")
    parser.add_argument("--support-results", type=Path)
    parser.add_argument("--confirmation-results", type=Path)
    parser.add_argument("--reward-results", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/v5/evaluation/advisor_packet.json",
    )
    arguments = parser.parse_args()
    if arguments.fixture_dry_run:
        if arguments.execute:
            print("BLOCKED: --fixture-dry-run and --execute are mutually exclusive")
            return 2
        packet = build_advisor_packet(_fixture_results())
        print(json.dumps({"status": packet["status"]}))
        return 0
    output_root = arguments.output
    if not arguments.execute:
        print("BLOCKED: Advisor packet write requires explicit --execute.")
        return 2
    if output_root.exists() or output_root.is_symlink():
        print(f"BLOCKED: output already exists; overwrite forbidden: {output_root}")
        return 2
    try:
        loaded = {
            "support_results": _read_optional_result(arguments.support_results, "support results"),
            "confirmation_results": _read_optional_result(
                arguments.confirmation_results, "confirmation results"
            ),
            "reward_results": _read_optional_result(arguments.reward_results, "reward results"),
        }
        packet = build_advisor_packet(
            {name: payload for name, payload in loaded.items() if payload is not None}
        )
        output_root.mkdir(parents=True)
        filename = (
            "status.json"
            if packet["status"] == "BLOCKED_MISSING_RESULTS"
            else "advisor_packet.json"
        )
        _write_json(output_root / filename, packet)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"BLOCKED: {error}")
        return 2
    print(f"{packet['status']}: output={output_root / filename}")
    return 2 if packet["status"] == "BLOCKED_MISSING_RESULTS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
