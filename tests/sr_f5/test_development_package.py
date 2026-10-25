"""The deployed archive must import the real pilot without repository assets."""

import os
import runpy
import subprocess
import sys
import tarfile
from pathlib import Path


def test_archive_contains_native_contract_import_dependencies(tmp_path):
    repository = Path(__file__).parents[2]
    script = runpy.run_path(str(repository / "scripts/sr_f5/package_development.py"))
    archive = tmp_path / "deployment.tar"
    script["package"]("HEAD", archive)
    deployment = tmp_path / "deployment"
    with tarfile.open(archive) as bundle:
        bundle.extractall(deployment, filter="data")
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import sr_f5.pilot, sr_f1.contract; "
                "assert Path(sr_f5.pilot.__file__).is_relative_to(Path.cwd()); "
                "assert sr_f1.contract.PACKAGE.is_relative_to(Path.cwd()); "
                "print('PACKAGED_NATIVE_IMPORT_PASS')"
            ),
        ],
        cwd=deployment,
        env=dict(os.environ, PYTHONPATH=str(deployment / "src"), PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        check=True,
    )
    assert process.stdout.strip() == "PACKAGED_NATIVE_IMPORT_PASS"
