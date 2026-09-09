import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.audit_r1_runtime import finalize_evidence, main
from src.core import file_hash


def test_failure_keeps_required_evidence_distinct_from_measurements(tmp_path):
    (tmp_path / "likelihood_parity.json").write_text(
        '{"status":"FAIL","paths":{"cache":{"passed":false}}}'
    )
    finalize_evidence(tmp_path, "FAIL", {"error": "pre-update check failed"}, "CPU_AUDIT")
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "FAIL"
    assert json.loads((tmp_path / "training_smoke.json").read_text())["status"] == "NOT_MEASURED"
    assert (
        json.loads((tmp_path / "likelihood_parity.json").read_text())["paths"]["cache"]["passed"]
        is False
    )
    assert (tmp_path / "manifest.json").exists()


def test_final_manifest_hashes_the_terminal_progress(tmp_path):
    (tmp_path / "progress.json").write_text('{"status":"RUNNING"}')
    finalize_evidence(tmp_path, "FAIL", {"error": "fixture failure"}, "CPU_AUDIT")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert any(row["path"] == "progress.json" for row in manifest["files"])
    for row in manifest["files"]:
        artifact = tmp_path / row["path"]
        assert row["sha256"] == file_hash(artifact)
        assert row["bytes"] == artifact.stat().st_size


def test_no_cuda_failure_is_reported_without_loading_weights(tmp_path):
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("src.audit_r1_runtime.load_adapter") as loader,
    ):
        assert (
            main(
                [
                    "--data-root",
                    str(tmp_path),
                    "--out",
                    str(tmp_path / "r1"),
                    "--smoke-out",
                    str(tmp_path / "r1/smoke"),
                ]
            )
            == 1
        )
    loader.assert_not_called()
    status = json.loads((tmp_path / "r1/status.json").read_text())
    assert status["execution_kind"] == "CPU_AUDIT"
    assert "CUDA" in status["details"]["error"]
    with pytest.raises(ValueError, match="overwrite"):
        main(
            [
                "--data-root",
                str(tmp_path),
                "--out",
                str(tmp_path / "r1"),
                "--smoke-out",
                str(tmp_path / "r1/smoke"),
            ]
        )


@pytest.mark.parametrize("exit_code", [0, 7])
def test_slurm_wrapper_initializes_scratch_and_preserves_failure(tmp_path, exit_code):
    repo = tmp_path / "dissertation-ssvc/ssvc_flow"
    (repo / "src").mkdir(parents=True)
    (repo / "src/__init__.py").write_text("")
    (repo / "src/audit_r1_runtime.py").write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        "out=Path(sys.argv[sys.argv.index('--out')+1])\n"
        "(out/'environment.json').write_text(json.dumps({k:os.environ[k] "
        "for k in ('TMPDIR','TMP','TEMP','HF_HOME')}))\n"
        f"sys.exit({exit_code})\n"
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    git = binaries / "git"
    git.write_text("#!/bin/sh\nprintf '%s\\n' cpu-fixture-commit\n")
    git.chmod(0o755)
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "HF_HOME")
    }
    env.update(
        SSVC_WORK=str(tmp_path),
        SSVC_PYTHON=sys.executable,
        SLURM_JOB_ID="123",
        PATH=str(binaries) + os.pathsep + os.environ["PATH"],
    )
    result = subprocess.run(
        ["bash", str(Path(__file__).parents[1] / "scripts/ntu_r1.sbatch")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == exit_code, result.stderr
    run = tmp_path / "runs/NEXT_20260909/R1_server_123_attempt_0"
    measured = json.loads((run / "environment.json").read_text())
    assert measured["TMP"] == measured["TEMP"] == measured["TMPDIR"] == str(run / "tmp")
    assert measured["HF_HOME"] == str(tmp_path / "cache/huggingface")
    assert f"exit_code={exit_code}\n" in (run / "result.txt").read_text()
    archive = Path(str(run) + ".tar.gz")
    assert Path(str(archive) + ".sha256").is_file()
    with tarfile.open(archive) as stream:
        assert not any("/tmp" in name for name in stream.getnames())
