"""Packaging revision identity checks; fixtures never claim scientific completion."""

import copy
import zipfile

import pytest
from test_delivery import delivery, write

OLD = "35800b02b1ef1eff2378e4c933393b4ea2668e4d"
NEW = "a" * 40


@pytest.fixture
def revision(tmp_path, monkeypatch):
    old, new = tmp_path / "operations" / OLD, tmp_path / "operations" / NEW
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "build_delivery.py").write_text("# retained original builder\n")
    (old / "dispatch_registered.py").write_text("# retained dispatcher\n")
    builder = new / "build_delivery.py"
    builder.write_text("# independently revised packaging builder\n")
    monkeypatch.setattr(delivery, "__file__", str(builder))
    for name in ("RUN_IDENTITY.json", "deployment/SOURCE_MANIFEST.json", "EXECUTION_FREEZE.json"):
        write(tmp_path / name, {"fixture": name})
    helper = old / "OPERATION_SOURCES.json"
    write(
        helper,
        dict(
            schema_version=1,
            experiment_id=delivery.EXPERIMENT,
            files={p.name: delivery.sha_file(p) for p in old.glob("*.py")},
        ),
    )
    identity = old / "OPERATION_IDENTITY.json"
    write(
        identity,
        dict(
            operation_sources_sha256=delivery.sha_file(helper),
            run_identity_sha256=delivery.sha_file(tmp_path / "RUN_IDENTITY.json"),
            source_manifest_sha256=delivery.sha_file(tmp_path / "deployment/SOURCE_MANIFEST.json"),
        ),
    )
    paths = dict(
        builder=builder,
        original_builder=old / "build_delivery.py",
        helper_manifest=helper,
        operation_identity=identity,
        run_identity=tmp_path / "RUN_IDENTITY.json",
        scientific_source_manifest=tmp_path / "deployment/SOURCE_MANIFEST.json",
        execution_freeze=tmp_path / "EXECUTION_FREEZE.json",
    )
    manifest = new / "DELIVERY_REVISION.json"
    value = dict(
        schema_version=1,
        experiment_id=delivery.EXPERIMENT,
        root=str(tmp_path),
        revision_commit=NEW,
        scope="DELIVERY_PACKAGING_AND_VERIFICATION_ONLY",
        bindings={
            name: dict(path=str(path.relative_to(tmp_path)), sha256=delivery.sha_file(path))
            for name, path in paths.items()
        },
    )
    write(manifest, value)
    return tmp_path, manifest, value, paths


def load(fixture):
    root, manifest, _, paths = fixture
    return delivery.load_delivery_revision(
        root,
        manifest,
        helper_manifest=paths["helper_manifest"],
        operation_identity=paths["operation_identity"],
    )


def test_revision_preserves_original_helpers_and_identifies_actual_builder(revision):
    _root, _manifest, value, paths = revision
    checked = load(revision)
    helpers, receipt = delivery.helper_sources(paths["helper_manifest"], delivery_revision=checked)
    assert helpers["build_delivery.py"] == paths["original_builder"]
    assert receipt["original_manifest_sha256"] == value["bindings"]["helper_manifest"]["sha256"]
    assert checked.identity["bindings"]["builder"]["sha256"] == delivery.sha_file(paths["builder"])
    assert set(checked.sources) == {
        "delivery_revision/DELIVERY_REVISION.json",
        "delivery_revision/source/build_delivery.py",
    }
    with pytest.raises(PermissionError, match="does not bind"):
        delivery.helper_sources(paths["helper_manifest"])


@pytest.mark.parametrize(
    "field",
    [
        "builder",
        "original_builder",
        "helper_manifest",
        "operation_identity",
        "run_identity",
        "scientific_source_manifest",
        "execution_freeze",
    ],
)
def test_each_bound_source_change_is_rejected(revision, field):
    revision[3][field].write_bytes(revision[3][field].read_bytes() + b"\n")
    with pytest.raises(PermissionError, match="binding differs"):
        load(revision)


@pytest.mark.parametrize(
    "field,value",
    [
        ("scope", "SCIENCE"),
        ("root", "/wrong"),
        ("revision_commit", OLD),
        ("schema_version", 2),
        ("experiment_id", "OTHER"),
    ],
)
def test_revision_identity_cannot_change(revision, field, value):
    changed = copy.deepcopy(revision[2])
    changed[field] = value
    write(revision[1], changed)
    with pytest.raises(PermissionError, match="identity or scope"):
        load(revision)


def test_missing_binding_rejected(revision):
    changed = copy.deepcopy(revision[2])
    del changed["bindings"]["execution_freeze"]
    write(revision[1], changed)
    with pytest.raises(PermissionError, match="bindings incomplete"):
        load(revision)


def test_redirect_to_other_builder_rejected(revision, monkeypatch):
    monkeypatch.setattr(delivery, "__file__", str(revision[3]["original_builder"]))
    with pytest.raises(PermissionError, match="actual builder"):
        load(revision)


def test_revision_rejects_unbound_cli_operation(revision):
    with pytest.raises(PermissionError, match="actual builder"):
        delivery.load_delivery_revision(
            revision[0],
            revision[1],
            helper_manifest=None,
            operation_identity=revision[3]["operation_identity"],
        )


def test_revision_rejects_broken_historical_link_even_with_new_hash(revision):
    _root, manifest, value, paths = revision
    changed = delivery.read(paths["operation_identity"])
    changed["run_identity_sha256"] = "0" * 64
    write(paths["operation_identity"], changed)
    value = copy.deepcopy(value)
    value["bindings"]["operation_identity"]["sha256"] = delivery.sha_file(
        paths["operation_identity"]
    )
    write(manifest, value)
    with pytest.raises(PermissionError, match="historical identity linkage"):
        load(revision)


def test_revision_sources_are_streamed_and_rechecked(revision, tmp_path):
    checked = load(revision)
    output = tmp_path / "revision.zip"
    writer = delivery.DeliveryZip(output)
    for name, (path, record) in checked.sources.items():
        writer.add_path(name, path, record)
    writer.finish()
    with zipfile.ZipFile(output) as archive:
        assert archive.read("delivery_revision/DELIVERY_REVISION.json") == revision[1].read_bytes()
        assert (
            archive.read("delivery_revision/source/build_delivery.py")
            == revision[3]["builder"].read_bytes()
        )
    revision[3]["builder"].write_text("# changed\n")
    writer = delivery.DeliveryZip(tmp_path / "changed.zip")
    try:
        source, record = checked.sources["delivery_revision/source/build_delivery.py"]
        with pytest.raises(PermissionError, match="changed"):
            writer.add_path("new.py", source, record)
    finally:
        writer.archive.close()


def test_external_receipt_explicitly_carries_revision(revision):
    checked = load(revision)
    output = revision[0] / "fixture.zip"
    output.write_bytes(b"fixture only")
    provenance = dict(
        run_identity_sha256="run",
        scientific_source_manifest_sha256="science",
        helper_manifest_sha256="old-helper",
        delivery_revision=checked.identity,
    )
    receipt = delivery._external_receipt(
        output, dict(checks={"raw_rescoring_chain": "PASS"}, entries=1), provenance, {"fixture": 1}
    )
    assert receipt["delivery_revision"] == checked.identity
    assert receipt["helper_manifest_sha256"] == "old-helper"
