"""Memory measurements are bounded observations, never scientific acceptance."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "delivery_profile", ROOT / "scripts/sr_f3_ce/profile_delivery_memory.py"
)
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


@pytest.mark.parametrize("platform,raw,expected", [("darwin", 4096, 4096), ("linux", 4, 4096)])
def test_rss_unit_normalization(platform, raw, expected):
    assert profile.rss_bytes(raw, platform) == expected


def test_fixture_full_shape_and_independent_allocations():
    builder = SimpleNamespace()
    root, api, dataset, training, monitor = profile.synthetic_fixture(builder, 3)
    assert root.name.startswith("SYNTHETIC")
    assert len(training) == 768 * 16
    assert len(monitor) == 51 * 64
    assert sum(item[2] for item in training) == 98304
    assert sum(item[2] for item in monitor) == 26112
    first = builder.read(training[0][0])["records"]
    second = builder.read(training[0][0])["records"]
    assert first == second
    assert first[0]["tokens"] is not first[1]["tokens"]
    assert first[0]["tokens"][0] is not second[0]["tokens"][0]
    assert dataset.tasks.keys() == {"fixture_q"}
    assert api.runner.work_occurrence(first)["records"] == 8
    path = root / "runs" / training[0][1]["run_id"] / "work_occurrence/step-01.json"
    receipt = builder.read(path)
    assert receipt["summary"]["records"] == 128
    assert receipt["raw_group_sha256"] == [builder.sha_file(p) for p, _, _ in training[:16]]
    with pytest.raises(ValueError, match="Unexpected"):
        builder.read(root / "unregistered.json")


def test_real_payload_shape_is_only_synthetic_and_not_modified(tmp_path):
    raw = tmp_path / "sample.json"
    original = json.dumps(
        {
            "records": [
                {
                    "tokens": [42, 43],
                    "nested": {"logprobs": [-0.2]},
                    "qid": "real_q",
                    "full_text": "real text",
                }
            ]
        }
    )
    raw.write_text(original)
    builder = SimpleNamespace()
    _, _, _, training, _ = profile.synthetic_fixture(builder, 999, raw)
    rows = builder.read(training[0][0])["records"]
    assert rows[0]["tokens"] == [42, 43]
    assert rows[0]["nested"] == {"logprobs": [-0.2]}
    assert rows[0]["qid"] == "fixture_q"
    assert rows[0]["full_text"] != "real text"
    rows[0]["nested"]["logprobs"].append(-0.3)
    assert rows[1]["nested"]["logprobs"] == [-0.2]
    assert raw.read_text() == original


def test_full_synthetic_trajectory_has_real_denominator_checks():
    builder = profile.load_builder(ROOT / "scripts/sr_f3_ce/build_delivery.py")
    result = builder.work_occurrence_trajectory(*profile.synthetic_fixture(builder, 0))
    assert len(result["training"]) == 768
    assert len(result["monitor"]) == 51
    assert all(p["summary"]["records"] == 128 for p in result["training"])
    assert all(p["summary"]["records"] == 512 for p in result["monitor"])
    assert result["new_model_calls"] == 0


def test_real_wrapper_runs_fresh_child_and_keeps_profile_separate(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-allocation")
    builder = tmp_path / "builder.py"
    builder.write_text(
        "import os\ndef build(root, output, **kwargs):\n"
        "    return {'pid': os.getpid(), 'checks': {'fixture': 'PASS'}}\n"
    )
    receipt = tmp_path / "profile.json"
    result = profile.main(
        [
            "--builder",
            str(builder),
            "--mode",
            "real-build",
            "--root",
            str(tmp_path),
            "--output",
            str(tmp_path / "archive.zip"),
            "--receipt",
            str(receipt),
        ]
    )
    assert result["pid"] != os.getpid()
    assert result["exitcode"] == 0
    assert result["peak_rss_bytes"] > 0
    assert result["status"] == "REAL_BUILD_RETURNED"
    assert result["builder_sha256"] == profile.sha_file(builder)
    assert "not Slurm/cgroup total or resource minimum" in result["measurement_scope"]
    assert json.loads(receipt.read_text()) == result
    before = receipt.read_bytes()
    with pytest.raises(SystemExit):
        profile.main(["--builder", str(builder), "--receipt", str(receipt)])
    assert receipt.read_bytes() == before


def test_failed_child_not_reported_as_success(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-allocation")
    builder = tmp_path / "builder.py"
    builder.write_text("def build(*args, **kwargs):\n    raise ValueError('fixture failure')\n")
    receipt = tmp_path / "failure.json"
    with pytest.raises(SystemExit):
        profile.main(
            [
                "--builder",
                str(builder),
                "--mode",
                "real-build",
                "--root",
                str(tmp_path),
                "--output",
                str(tmp_path / "archive.zip"),
                "--receipt",
                str(receipt),
            ]
        )
    result = json.loads(receipt.read_text())
    assert result["status"] == "PROFILE_FAILED"
    assert result["exitcode"] != 0
    assert "result_sha256" not in result
    assert result["builder_sha256"] == profile.sha_file(builder)
    assert result["profile_source_sha256"] == profile.sha_file(profile.__file__)
    assert result["elapsed_wall_seconds"] >= 0
    assert result["peak_rss_bytes"] is None
    assert result["current_rss_bytes"] is None
    assert "may miss the true peak" in result["measurement_scope"]
    assert "fixture failure" in result["stderr_tail"]
    assert len(result["stderr_tail"]) <= 2048
    assert len(result["stderr_sha256_utf8"]) == 64
    assert len(result["stdout_sha256_utf8"]) == 64
    assert result["command"][1] == "-B"
    assert result["started_at_utc"] <= result["finished_at_utc"]
    assert result["platform"] == profile.sys.platform


@pytest.mark.parametrize("extra", [["--token-length", "-1"], ["--mode", "real-build"]])
def test_invalid_profile_request_rejected(tmp_path, extra):
    builder = tmp_path / "builder.py"
    builder.write_text("raise AssertionError('must not load')\n")
    with pytest.raises(SystemExit):
        profile.main(["--builder", str(builder), *extra])


def test_linux_real_build_rejects_unallocated_login_node(tmp_path, monkeypatch):
    builder = tmp_path / "builder.py"
    builder.write_text("raise AssertionError('must not load')\n")
    monkeypatch.setattr(profile.sys, "platform", "linux")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(SystemExit):
        profile.main(
            [
                "--builder",
                str(builder),
                "--mode",
                "real-build",
                "--root",
                str(tmp_path),
                "--output",
                str(tmp_path / "not-created.zip"),
            ]
        )
    assert not (tmp_path / "not-created.zip").exists()


@pytest.mark.parametrize("with_revision", [False, True])
def test_real_build_revision_manifest_exact_optional_passthrough(
    tmp_path, monkeypatch, with_revision
):
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-allocation")
    builder = tmp_path / "builder.py"
    manifest = tmp_path / "revision manifest.json"
    manifest.write_text('{"fixture":true}\n')
    revision_parameter = ", delivery_revision_manifest" if with_revision else ""
    returned = "str(delivery_revision_manifest)" if with_revision else "None"
    builder.write_text(
        "def build(root, output, *, helper_manifest, extension_manifest, operation_identity"
        + revision_parameter
        + "):\n"
        + "    return {'revision': "
        + returned
        + "}\n"
    )
    argv = [
        "--builder",
        str(builder),
        "--mode",
        "real-build",
        "--root",
        str(tmp_path),
        "--output",
        str(tmp_path / "output.zip"),
    ]
    if with_revision:
        argv += ["--delivery-revision-manifest", str(manifest)]
    result = profile.main(argv)
    assert result["result_sha256"] == profile.canonical_digest(
        {"revision": str(manifest) if with_revision else None}
    )


def test_bounded_log_keeps_full_hash_only_small_tail():
    import hashlib
    import io

    raw = "abc" * 100000 + "final diagnostic"
    tail, digest = profile.bounded_log(io.StringIO(raw), tail_chars=10)
    assert tail == raw[-10:]
    assert digest == hashlib.sha256(raw.encode()).hexdigest()
