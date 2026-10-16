import pytest

from sr_f3_ce.bootstrap import verify_native_receipt_revision
from sr_f3_ce.protocol_revision import revision_config


@pytest.mark.parametrize("revision", ["v2", "c2"])
def test_p0_cannot_reuse_native_receipts_from_another_revision(revision):
    config = revision_config({"generation": {"prefill": "{"}}, revision)
    with pytest.raises(PermissionError, match="different protocol revision"):
        verify_native_receipt_revision(config, {}, {"serialization": "compact"})
    matching = {"protocol_revision": revision}
    assert (
        verify_native_receipt_revision(config, matching, dict(matching, serialization="indent_2"))
        == revision
    )
    with pytest.raises(PermissionError, match="different protocol revision"):
        verify_native_receipt_revision(config, matching, dict(matching, serialization="newline"))
    with pytest.raises(PermissionError, match="different protocol revision"):
        verify_native_receipt_revision(config, {}, dict(matching, serialization="indent_2"))


def test_original_native_receipt_keeps_registered_v1_compatibility():
    assert verify_native_receipt_revision({}, {}, {"serialization": "compact"}) == "v1"
