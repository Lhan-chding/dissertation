"""Package committed development code with transitive native contract assets."""

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATHS = (
    "src",
    "scripts",
    "configs",
    "tests/sr_f5",
    "docs/sr_f5",
    "pyproject.toml",
    "docs/sr_f1/package",
    "docs/sr_f1/amendments",
)


def package(revision, output):
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", revision + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            "--output=" + str(Path(output).resolve()),
            commit,
            *PATHS,
        ],
        cwd=ROOT,
        check=True,
    )
    return commit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(package(args.revision, args.output))
