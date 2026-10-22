"""Build the production code snapshot with inherited runtime resource files."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

ARCHIVE_PATHS = (
    "src",
    "docs/sr_f4",
    "scripts/sr_f4",
    "tests/sr_f4",
    "pyproject.toml",
    "docs/sr_f1/package/PACKAGE_SHA256.json",
    "docs/sr_f1/package/reference",
    "docs/sr_f1/package/config",
    "docs/sr_f1/amendments",
)


def build_archive(repository, output, *, revision="HEAD"):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    subprocess.run(
        ["git", "archive", "-o", str(temporary), "--format=tar", revision, *ARCHIVE_PATHS],
        cwd=repository,
        check=True,
    )
    os.replace(temporary, output)
    return output


def check_imports(code, python):
    """Resolve the actual production import chain; no model load or generation."""
    code = Path(code).resolve()
    script = "\n".join(
        (
            "from pathlib import Path",
            "from sr_f3_ce.training import fresh_optimizer",
            "from sr_f4.runtime import Runtime",
            "from sr_f4.run import execute",
            "from sr_f4.replay import replay_stage",
            "from sr_f4.reporting import report",
            "from sr_f1.contract import PACKAGE, load_config",
            "assert PACKAGE.is_relative_to(Path.cwd())",
            "assert load_config()",
            "print('PRODUCTION_IMPORTS_RESOLVED; model_calls=0')",
        )
    )
    return subprocess.run(
        [str(python), "-c", script],
        cwd=code,
        env=dict(os.environ, PYTHONPATH=str(code / "src"), PYTHONDONTWRITEBYTECODE="1"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    print(build_archive(args.repository, args.output, revision=args.revision))
