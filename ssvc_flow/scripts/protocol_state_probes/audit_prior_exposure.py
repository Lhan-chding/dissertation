"""Read-only audit of executed JSONL evidence for registered U22 identifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def audit(ids, roots):
    needles = {v.encode() for v in set(ids) | set(ids.values())}
    pattern = re.compile(b"|".join(re.escape(n) for n in sorted(needles)))
    files, hits, errors = [], [], []
    seen = set()
    for root in roots:
        if not Path(root).is_dir():
            errors.append({"path": str(root), "error": "Missing audit root"})
            continue
        for path in sorted(Path(root).rglob("*.jsonl")):
            if any(p in {"cache", ".git", "node_modules", "__pycache__"} for p in path.parts):
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                st = path.stat()
                matches = 0
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for line_no, line in enumerate(stream, 1):
                        digest.update(line)
                        if not pattern.search(line):
                            continue
                        row = json.loads(line)
                        # Only actual output records constitute exposure, not a prompt list.
                        output_keys = {
                            "token_ids",
                            "generated_token_ids",
                            "completion",
                            "text",
                            "raw_text",
                            "output_text",
                            "raw_output",
                            "response",
                        }
                        executable = (
                            bool(output_keys & row.keys()) or "semantic" in row or "event" in row
                        )
                        hits.append(
                            {
                                "path": resolved,
                                "line": line_no,
                                "row_sha256": hashlib.sha256(line).hexdigest(),
                                "executed_output_candidate": executable,
                                "keys": sorted(row),
                                "prompt_id": row.get("prompt_id"),
                                "base_scene_id": row.get("base_scene_id"),
                                "role": row.get("role"),
                                "step": row.get("step"),
                            }
                        )
                        matches += 1
                files.append(
                    {
                        "path": resolved,
                        "bytes": st.st_size,
                        "mtime_ns": st.st_mtime_ns,
                        "matching_rows": matches,
                        "sha256": digest.hexdigest(),
                    }
                )
            except (OSError, ValueError) as exc:
                errors.append({"path": str(path), "error": repr(exc)})
    if not files:
        errors.append({"error": "No JSONL execution evidence available"})
    executed = [h for h in hits if h["executed_output_candidate"]]
    return {
        "schema": "protocol-probes-prior-exposure-v1",
        "status": "CONTAMINATED_DOWNGRADED"
        if executed
        else ("UNVERIFIED" if errors else "VERIFIED_UNTOUCHED"),
        "audit_complete": not errors,
        "verified": not errors,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "roots": [str(Path(r).resolve()) for r in roots],
        "prompt_ids": sorted(ids),
        "base_scene_ids": sorted(set(ids.values())),
        "original_split": "train",
        "split_role": "development_diagnostic" if executed else "untouched_outcome_replication",
        "files_scanned": len(files),
        "bytes_scanned": sum(f["bytes"] for f in files),
        "executed_matching_rows": len(executed),
        "metadata_matching_rows": len(hits) - len(executed),
        "errors": errors,
        "matches": hits,
        "file_inventory": files,
        "scope_limit": (
            "JSONL execution evidence in the named historical roots; "
            "no claim about deleted or inaccessible evidence."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", required=True)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = audit(json.loads(Path(args.ids).read_text()), args.roots)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"matches", "file_inventory", "prompt_ids", "base_scene_ids"}
            }
        )
    )


if __name__ == "__main__":
    main()
