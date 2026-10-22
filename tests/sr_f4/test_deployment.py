import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from sr_f4.deployment import build_archive, check_imports


def test_snapshot_contains_inherited_runtime_resources(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    broken = tmp_path / "broken"
    broken.mkdir()
    source = tmp_path / "source.tar"
    subprocess.run(
        ["git", "archive", "-o", str(source), "HEAD", "src"], cwd=repository, check=True
    )
    with tarfile.open(source) as archive:
        archive.extractall(broken, filter="data")
    with pytest.raises(subprocess.CalledProcessError) as failure:
        check_imports(broken, sys.executable)
    assert "docs/sr_f1/package/PACKAGE_SHA256.json" in failure.value.stderr
    complete = tmp_path / "complete"
    complete.mkdir()
    with tarfile.open(build_archive(repository, tmp_path / "complete.tar")) as archive:
        archive.extractall(complete, filter="data")
    assert "model_calls=0" in check_imports(complete, sys.executable)
