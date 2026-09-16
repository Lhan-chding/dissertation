import errno
from types import SimpleNamespace

import pytest

from src.modeling_v4.storage import require_space, storage_status


def test_missing_leaf_attribute_does_not_hide_parent_quota(tmp_path, monkeypatch):
    leaf = tmp_path / "campaign"
    leaf.mkdir()
    monkeypatch.setattr("shutil.disk_usage", lambda path: SimpleNamespace(free=1000000))

    def get_attribute(path, name):
        if path == tmp_path:
            return b"1000" if name.endswith("max_bytes") else b"900"
        raise OSError(errno.ENODATA, "no attribute")

    monkeypatch.setattr("os.getxattr", get_attribute, raising=False)
    result = storage_status(leaf)
    assert result["available_bytes"] == 100
    assert result["ancestor_quotas"][0]["path"] == str(tmp_path)
    with pytest.raises(OSError, match="preserve raw data"):
        require_space(leaf, 101)
    assert require_space(leaf, 50)["reservation_created"] is False


def test_permission_error_is_partial_not_unlimited_space(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.disk_usage", lambda path: SimpleNamespace(free=1000))

    def denied(*args):
        raise OSError(errno.EACCES, "permission")

    monkeypatch.setattr("os.getxattr", denied, raising=False)
    result = storage_status(tmp_path)
    assert result["status"] == "PARTIAL_QUOTA_READ"
    assert result["unreadable_quota_attributes"]
