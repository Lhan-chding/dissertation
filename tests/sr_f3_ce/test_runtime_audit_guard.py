"""Actual audit CLI boundaries; no model calls or scientific records are required."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/sr_f3_ce/audit_engine_runtime.py"
SOURCE_NAMES = (
    "src/sr_f3_ce/prompts.py",
    "src/sr_f3_ce/contract.py",
    "src/sr_f3_ce/runtime.py",
    "src/sr_f3_ce/data.py",
    "src/sr_f3_ce/evaluation.py",
    "src/sr_f3_ce/runner.py",
    "src/sr_f1/runtime.py",
    "src/sr_f1/json_protocol.py",
    "src/sr_f12/runtime.py",
    "src/mm_core/vl_runtime.py",
)


def _environment():
    env = dict(os.environ)
    for name in ("PYTHONDONTWRITEBYTECODE", "PYTHONOPTIMIZE", "PYTHONPYCACHEPREFIX"):
        env.pop(name, None)
    return env


def _source_hashes(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _frozen_fixture(tmp_path):
    repo, audit, output = (tmp_path / name for name in ("frozen_source", "audit", "output"))
    for name in SOURCE_NAMES:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name in ("contract.py", "prompts.py"):
            marker = "FROZEN_" + path.stem.upper() + "_DONT_WRITE_BYTECODE="
            path.write_text("import sys\n" + f"print({marker!r} + str(sys.dont_write_bytecode))\n")
        else:
            path.write_text("# Identity-only fixture; never imported.\n")
    manifest = audit / "snapshot/deployment/SOURCE_MANIFEST.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {"path": name, "sha256": digest}
                    for name, digest in _source_hashes(repo).items()
                ]
            }
        )
    )
    args = [
        str(SCRIPT),
        "--repository",
        str(repo),
        "--audit-root",
        str(audit),
        "--output",
        str(output),
    ]
    return repo, output, args


def test_optimized_cli_rejects_before_import_or_output(tmp_path):
    repo, output, args = _frozen_fixture(tmp_path)
    before = _source_hashes(repo)
    result = subprocess.run(
        [sys.executable, "-O", *args],
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert "Optimized Python is unsupported: audit assertions must remain enabled" in result.stderr
    assert "FROZEN_" not in result.stdout
    assert not output.exists()
    assert _source_hashes(repo) == before


def test_default_cli_dynamic_import_keeps_frozen_tree_without_pyc(tmp_path):
    repo, output, args = _frozen_fixture(tmp_path)
    before = _source_hashes(repo)
    result = subprocess.run(
        [sys.executable, *args],
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    # Both real SourceFileLoader imports complete before the intentionally absent
    # plan input stops the minimal audit. No -B flag or external env guard is used.
    assert "FROZEN_CONTRACT_DONT_WRITE_BYTECODE=True" in result.stdout
    assert "FROZEN_PROMPTS_DONT_WRITE_BYTECODE=True" in result.stdout
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr and "TASKS_AUDIT_ONLY.jsonl" in result.stderr
    assert output.is_dir()
    assert _source_hashes(repo) == before
    assert not list(repo.rglob("*.pyc"))
    assert not list(repo.rglob("__pycache__"))
