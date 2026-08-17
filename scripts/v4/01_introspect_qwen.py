"""Verify and introspect the exact pinned Qwen snapshot on the server."""

from __future__ import annotations

import argparse
from pathlib import Path

from _guards import (
    CONFIG_PATH,
    PACKAGE_LOCK_PATH,
    ROOT,
    blocked_unless_execute,
    validate_server_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--package-lock", type=Path, default=PACKAGE_LOCK_PATH)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/model/ModelScope/Qwen/Qwen2.5-VL-3B-Instruct"),
    )
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    arguments = parser.parse_args()
    if blocked_unless_execute(arguments.execute):
        return 2
    canonical_artifact_root = ROOT / "artifacts"
    output = arguments.artifact_root / "v4"
    targets = (output / "model_introspection.json", output / "module_manifest.txt")
    try:
        if (
            arguments.artifact_root.is_symlink()
            or arguments.artifact_root.resolve() != canonical_artifact_root.resolve()
            or output.is_symlink()
        ):
            raise RuntimeError(
                f"artifact root must be the canonical repository path: {canonical_artifact_root}"
            )
        if any(path.exists() or path.is_symlink() for path in targets):
            raise FileExistsError("refusing to overwrite Qwen introspection evidence")
        validate_server_inputs(
            config=arguments.config,
            package_lock=arguments.package_lock,
            model_path=arguments.model_path,
            inputs=(),
            input_sha256=(),
            expected_input_sha256=(),
            require_raw_evidence=False,
        )
        from compensability_v4.qwen.introspect_model import write_model_introspection
        from compensability_v4.qwen.model_loader import load_pinned_qwen

        model, _processor = load_pinned_qwen(model_path=arguments.model_path)
        write_model_introspection(model, output)
    except Exception as error:
        print(f"BLOCKED: {error}")
        return 2
    print(f"READY: Qwen runtime introspection written under {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
