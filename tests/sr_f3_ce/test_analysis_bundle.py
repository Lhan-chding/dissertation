import gzip
import hashlib
import importlib.util
import struct
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "scripts/sr_f3_ce/analysis_bundle.py"
spec = importlib.util.spec_from_file_location("analysis_bundle_test", SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def pin(data):
    return dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def raw():
    return dict(
        full_text="invalid original retained",
        raw_text="invalid original retained",
        tokens=[1, 2],
        model_input=dict(prompt="original"),
        finish_reason="length",
        max_new_tokens=2,
    )


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(
        m,
        "EXPECTED",
        dict(training_groups=2, training_records=16, support=1, monitor=2, engine=2, endpoint=3),
    )
    root = tmp_path / "source"
    root.mkdir()
    files = {}

    def add(name, value, category):
        data = value if isinstance(value, bytes) else m.encoded(value)
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files[name] = dict(**pin(data), category=category)

    for i in range(2):
        add(
            f"runs/run/rollouts/{i:02d}-00.json",
            dict(records=[raw() for _ in range(8)]),
            "training_raw",
        )
    add(
        "runs/run/support/01-q.json",
        dict(
            target=dict(text="teacher", object={"answer": 1}, spans={}),
            tokens=[1, 2],
            answer_mask=[0, 1],
            work_mask=[1, 0],
            mask_audit={},
        ),
        "support",
    )
    add("monitor/m/step-00/groups/g.json", dict(records=[raw(), raw()]), "monitor_raw")
    add("engine/b0/groups/g.json", dict(records=[raw(), raw()]), "engine_raw")
    add(
        "release/ENDPOINT_RAW.jsonl.gz",
        gzip.compress(b"".join(m.encoded(raw()).replace(b"\n", b"") + b"\n" for _ in range(3))),
        "release",
    )
    add(
        "release/RAW_RESCORING.json",
        dict(
            status="PASS",
            records=3,
            mismatches=[],
            gzip_read_to_eof=True,
            raw_sha256=files["release/ENDPOINT_RAW.jsonl.gz"]["sha256"],
        ),
        "release",
    )
    item = dict(
        arm1="EXEC",
        arm0="RL",
        scope="REPAIR",
        effect_pp=-1.0,
        per_block_pp=[-3, 1, -1],
        CI95_pp=[-3, 1],
        CI_simultaneous_pp=[-4, 2],
        paired_t_sensitivity=dict(CI95_pp=[-5, 3]),
    )
    add(
        "release/PRIMARY_ANALYSIS.json",
        dict(primary={k: item for k in ("C1", "C2", "C3")}, adverse_results_retained=True),
        "release",
    )
    add(
        "release/EXTERNAL_ANALYSIS.json",
        dict(
            CHARTQA=dict(
                model=dict(v2=dict(exact_match=0.1), typed_status_counts=dict(UNRESOLVED=4))
            )
        ),
        "release",
    )
    add(
        "release/PREFIX_DIAGNOSTICS.json",
        dict(
            panels=dict(
                model=dict(
                    records=3,
                    mean_generated_tokens=2,
                    budgets={"128": dict(value_correct=0.1, finish=0.2, presentation=0.3)},
                )
            )
        ),
        "release",
    )
    add("release/ENDPOINT_SCORES.jsonl", b'{"retained":true}\n', "release")
    add("release/RELEASE.json", dict(status="RELEASED"), "release")
    add("release/RELEASE_SCORED.json", dict(status="RELEASED"), "release")
    add(
        "release/MANIFEST.json",
        dict(
            files=[
                dict(path=Path(n).name, **{k: v[k] for k in ("sha256", "bytes")})
                for n, v in files.items()
                if n.startswith("release/")
            ]
        ),
        "release",
    )
    manifest = tmp_path / "manifest.json"

    def save():
        manifest.write_bytes(m.encoded(dict(files=files)))

    save()
    return SimpleNamespace(
        root=root,
        files=files,
        manifest=manifest,
        output=tmp_path / "result.zip",
        add=add,
        save=save,
    )


def build(case):
    case.save()
    return m.build(case.root, case.manifest, case.output)


def test_roundtrip_preserves_bytes_counts_and_boundaries(case):
    result = build(case)
    assert result["counts"] == m.EXPECTED
    assert result["fresh_scoring"] == "NOT_RUN"
    assert result["server_delivery_gates"] == "NOT_RUN_OR_CLAIMED"
    assert set(result["checks"].values()) == {"PASS"}
    assert m.verify(case.output)["archive"] == result["archive"]
    assert Path(str(case.output) + ".verification.json").exists()
    assert Path(str(case.output) + ".sha256").read_text().startswith(result["archive"]["sha256"])
    with zipfile.ZipFile(case.output) as z:
        for name in case.files:
            assert z.read(name) == (case.root / name).read_bytes()
        report = z.read(m.GENERATED + "REPORT_ZH.md").decode()
        for word in ("C1", "C2", "C3", "-3", "UNRESOLVED", "128", "未确认", "完整原分析附录"):
            assert word in report
    with pytest.raises(ValueError, match="exists"):
        build(case)


@pytest.mark.parametrize(
    "name",
    [
        "../x",
        "/x",
        "a//b",
        "a/./b",
        "a/../b",
        "C:/x",
        "a\\b",
        "models/x.json",
        "runs/r/checkpoints/state.pt",
        "weight.pt",
    ],
)
def test_unsafe_or_forbidden_before_read(case, name):
    case.files[name] = dict(sha256="0" * 64, bytes=1, category="evidence")
    with pytest.raises(ValueError):
        build(case)
    assert not case.output.exists()


def test_corrupt_source_rejected_and_partial_removed(case):
    (case.root / "release/ENDPOINT_RAW.jsonl.gz").write_bytes(b"bad")
    with pytest.raises(ValueError, match="Input hash"):
        build(case)
    assert not case.output.exists()
    assert not list(case.output.parent.glob(".analysis_*"))


def test_missing_science_row_fails_fixed_coverage(case):
    del case.files["runs/run/rollouts/00-00.json"]
    with pytest.raises(ValueError, match="coverage"):
        build(case)


def test_invalid_raw_fields_rejected(case):
    bad = raw()
    del bad["tokens"]
    case.add("runs/run/rollouts/00-00.json", dict(records=[bad] * 8), "training_raw")
    with pytest.raises(ValueError, match="token"):
        build(case)


def test_support_masks_required(case):
    case.add(
        "runs/run/support/01-q.json",
        dict(
            target=dict(text="x", object={"answer": 1}, spans={}),
            tokens=[1],
            answer_mask=[],
            work_mask=[1],
            mask_audit={},
        ),
        "support",
    )
    with pytest.raises(ValueError, match="mask"):
        build(case)


def test_symlink_rejected(case, tmp_path):
    p = case.root / "runs/run/rollouts/00-00.json"
    other = tmp_path / "outside"
    p.rename(other)
    p.symlink_to(other)
    with pytest.raises(ValueError, match="Symlink"):
        build(case)


def test_original_rescore_cannot_bind_other_payload(case):
    case.add(
        "release/RAW_RESCORING.json",
        dict(status="PASS", records=3, mismatches=[], gzip_read_to_eof=True, raw_sha256="0" * 64),
        "release",
    )
    with pytest.raises(ValueError):
        build(case)


def test_gzip_truncation_fails_even_with_rebound_file_hash(case):
    case.add("other/raw.gz", gzip.compress(b"raw")[:-4], "evidence")
    with pytest.raises((EOFError, gzip.BadGzipFile)):
        build(case)


def test_crc_corruption_rejected(case):
    build(case)
    with zipfile.ZipFile(case.output) as z:
        cursor = z.start_dir
    data = bytearray(case.output.read_bytes())
    original = struct.unpack_from("<I", data, cursor + 16)[0]
    struct.pack_into("<I", data, cursor + 16, original ^ 1)
    case.output.write_bytes(data)
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        m.verify(case.output)


def test_no_automatic_collection(case):
    (case.root / "unselected.txt").write_text("unrelated")
    build(case)
    with zipfile.ZipFile(case.output) as z:
        assert "unselected.txt" not in z.namelist()


def test_manifest_duplicate_json_keys_rejected(case):
    case.manifest.write_text('{"files":{},"files":{}}')
    with pytest.raises(ValueError, match="Duplicate"):
        m.build(case.root, case.manifest, case.output)


def test_lightweight_checkpoint_receipt_json_allowed(case):
    case.add(
        "parents/model/source_receipts/checkpoints/commit-96.json", {"sha256": "a" * 64}, "identity"
    )
    build(case)


def test_task_artifact_hash_binding_rejected(case):
    case.add(
        "tasks/train--x/COMPLETE.json",
        {"artifacts": [{"path": "runs/run/rollouts/00-00.json", "bytes": 1, "sha256": "0" * 64}]},
        "receipt",
    )
    with pytest.raises(ValueError, match="Task artifact binding"):
        build(case)


def test_panel_coverage_binding_rejected(case):
    case.add(
        "monitor/m/step-00/COVERAGE.json",
        {"groups": [{"group_id": "g", "sha256": "0" * 64}]},
        "receipt",
    )
    with pytest.raises(ValueError, match="COVERAGE group binding"):
        build(case)


def test_receipt_links_verified(case):
    name = "runs/run/rollouts/00-00.json"
    case.add(
        "tasks/train--x/COMPLETE.json",
        {"artifacts": [{"path": name, **{k: case.files[name][k] for k in ("sha256", "bytes")}}]},
        "receipt",
    )
    case.add(
        "monitor/m/step-00/COVERAGE.json",
        {
            "groups": [
                {"group_id": "g", "sha256": case.files["monitor/m/step-00/groups/g.json"]["sha256"]}
            ]
        },
        "receipt",
    )
    result = build(case)
    assert result["receipt_links"] == dict(task_artifact_links=1, panel_group_links=1)


def test_scheduler_gpu_total_not_double_counted_and_pending_not_final(case):
    snapshot = dict(
        returncode=0,
        at_utc="2026-10-21T09:00:00+00:00",
        stdout=(
            "1|COMPLETED|0:0|2026-10-20T00:00:00|2026-10-20T00:00:00|"
            "2026-10-20T01:00:00|3600|cpu=16,gres/gpu=4,gres/gpu:pro6000=4|gpu-node|teacher\n"
            "2|PENDING|0:0|2026-10-20T00:00:00|Unknown|Unknown|0||None assigned|teacher\n"
        ),
    )
    case.add("delivery_context/SCHEDULER_COST_SNAPSHOT.json", snapshot, "cost")
    build(case)
    with zipfile.ZipFile(case.output) as package:
        report = package.read(m.GENERATED + "REPORT_ZH.md").decode()
    assert "合计\uff1a4.0" in report
    assert "2026-10-20T08:00:00+08:00" in report
    assert "总墙钟尚不能" in report
