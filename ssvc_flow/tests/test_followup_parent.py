"""Restricted parent evidence is JSON-only, bounded, immutable and fail-closed."""

import hashlib
import stat
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "reports/SSVC_GPT_PRO_DELIVERY_20260914"
WARM = PARENT / "02_data/R3_warm"


def test_directory_and_zip_metadata_audit_never_promote_raw():
    from src.followup_parent import audit_parent_evidence

    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in WARM.glob("*.json")}
    direct = audit_parent_evidence(PARENT, project_root=ROOT / "ssvc_flow")
    archived = audit_parent_evidence(PARENT.with_suffix(".zip"))
    assert direct["parent_metadata_verified"] is archived["parent_metadata_verified"] is True
    assert direct["status"] == archived["status"] == "BLOCKED_MISSING_PARENT_RAW"
    for key in (
        "parent_raw_verified",
        "raw_tensors_verified",
        "gpu_smoke_passed",
        "gpu_started",
        "training_started",
    ):
        assert direct[key] is archived[key] is False
    assert len(direct["bank_composition"]) == 5
    assert direct["source_files"]
    assert before == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in WARM.glob("*.json")
    }


def test_missing_parent_metadata_is_explicit(tmp_path):
    from src.followup_parent import audit_parent_evidence

    with pytest.raises((ValueError, FileNotFoundError), match=r"metadata|runtime_lock"):
        audit_parent_evidence(tmp_path)


def test_origin_cannot_be_overridden_by_caller():
    from src.followup_parent import audit_parent_evidence

    with pytest.raises(ValueError, match=r"origin"):
        audit_parent_evidence(WARM, required_origin="0" * 64)
    with pytest.raises(ValueError, match=r"read.only"):
        audit_parent_evidence(WARM, readonly=False)


@pytest.mark.parametrize(
    "name",
    [
        "../evil.json",
        "/evil.json",
        "a/../evil.json",
        "a//evil.json",
        "a\\evil.json",
        "C:/evil.json",
    ],
)
def test_restricted_archive_rejects_unsafe_names(tmp_path, name):
    from src.followup_parent import RestrictedParentReader

    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(name, "{}")
    with pytest.raises(ValueError, match=r"path"):
        RestrictedParentReader(archive)


def test_archive_duplicate_symlink_and_file_directory_collision(tmp_path):
    from src.followup_parent import RestrictedParentReader

    for fault in ("duplicate", "symlink", "ancestor"):
        archive = tmp_path / (fault + ".zip")
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("a.json", "{}")
            if fault == "duplicate":
                with pytest.warns(UserWarning):
                    z.writestr("a.json", "{}")
            elif fault == "symlink":
                info = zipfile.ZipInfo("link.json")
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                z.writestr(info, "a.json")
            else:
                z.writestr("a.json/b.json", "{}")
        with pytest.raises(ValueError):
            RestrictedParentReader(archive)


def test_bounded_json_rejects_hash_type_duplicate_keys_nan_and_symlink(tmp_path):
    from src.followup_parent import RestrictedParentReader

    p = tmp_path / "a.json"
    p.write_text('{"ok": true}')
    with RestrictedParentReader(tmp_path) as reader:
        assert reader.read_json("a.json") == {"ok": True}
        with pytest.raises(ValueError, match=r"hash"):
            reader.read_json("a.json", expected_sha256="0" * 64)
        for name in ("../a.json", "a.txt"):
            with pytest.raises(ValueError):
                reader.read_json(name)
        for bad in ("[]", '{"a":1,"a":2}', '{"x":NaN}'):
            p.write_text(bad)
            with pytest.raises(ValueError):
                reader.read_json("a.json")
    p.write_text("{}" * 100)
    with (
        RestrictedParentReader(tmp_path, max_file_bytes=10) as reader,
        pytest.raises(ValueError, match=r"limit"),
    ):
        reader.read_json("a.json")
    p.unlink()
    p.symlink_to(WARM / "runtime_lock.json")
    with RestrictedParentReader(tmp_path) as reader, pytest.raises(ValueError, match=r"symlink"):
        reader.read_json("a.json")


def test_zip_declared_sizes_and_compression_are_bounded(tmp_path):
    from src.followup_parent import RestrictedParentReader

    archive = tmp_path / "large.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("a.json", " " * 10000)
    with pytest.raises(ValueError, match=r"limit|compression"):
        RestrictedParentReader(archive, max_file_bytes=100)
    with pytest.raises(ValueError, match=r"limit|compression"):
        RestrictedParentReader(archive, max_total_bytes=100)
    with pytest.raises(ValueError, match=r"compression"):
        RestrictedParentReader(archive, max_compression_ratio=2)


@pytest.mark.parametrize(
    "fault", ["phase", "step", "model", "B", "K", "plan", "identity", "composition"]
)
def test_metadata_schema_and_cross_file_binding_rejects_tampering(fault):
    from src.followup_parent import load_parent_metadata, validate_parent_metadata

    docs = load_parent_metadata(WARM)
    runtime, bank, candidates = (
        docs[k] for k in ("runtime_lock", "bank_manifest", "candidate_manifest")
    )
    if fault == "phase":
        runtime["identity"]["phase"] = "R3-cold"
    elif fault == "step":
        runtime["identity"]["checkpoint_step"] = 65
    elif fault == "model":
        runtime["config"]["model"]["revision"] = "other"
    elif fault in ("B", "K"):
        runtime["config"]["R3"][fault] = 2
    elif fault == "plan":
        bank["plan"]["banks"][3][0] = "other"
    elif fault == "identity":
        bank["identity"]["data_hash"] = "0" * 64
    else:
        candidates["banks"]["3"]["group_composition"]["prompts"][2]["counts"]["W"] = 5
    with pytest.raises(ValueError):
        validate_parent_metadata(runtime, bank, candidates)


@pytest.mark.parametrize(
    "section,key,value", [("R3", "B", 4.0), ("R3", "K", 8.0), ("R4", "seed", 17.0)]
)
def test_parent_integer_contract_rejects_float_aliases(section, key, value):
    from src.followup_parent import load_parent_metadata, validate_parent_metadata

    docs = load_parent_metadata(WARM)
    docs["runtime_lock"]["config"][section][key] = value
    with pytest.raises(ValueError, match=r"B4|K8|seed"):
        validate_parent_metadata(
            docs["runtime_lock"], docs["bank_manifest"], docs["candidate_manifest"]
        )
