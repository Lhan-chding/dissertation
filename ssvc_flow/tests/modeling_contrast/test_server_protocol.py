"""The approved server revision changes resources, never scientific choices."""

import copy

import pytest

from src.modeling_contrast import protocol


def test_exact_server_revision_keeps_scientific_protocol():
    original = protocol.load_config()
    server = protocol.load_config(protocol.ROOT / "configs/modeling_contrast/protocol_server.json")
    assert protocol.config_sha256(original) == protocol.CONFIG_SHA256
    assert protocol.config_sha256(server) == protocol.SERVER_CONFIG_SHA256
    assert server["resources"]["max_added_output_gib"] == 6
    normalized = copy.deepcopy(server)
    normalized.pop("execution_amendment")
    normalized["resources"]["max_added_output_gib"] = 3
    assert normalized == original
    assert protocol.validate_config(server)["total_optimizer_updates"] == 6080
    forecast = dict(
        wall_seconds=5000, peak_ram_gib=3, added_output_bytes=4 * 2**30, temporary_bytes=0
    )
    assert not protocol.resource_gate(forecast, original)["passed"]
    assert protocol.resource_gate(forecast, server)["passed"]


@pytest.mark.parametrize(
    "field,value", [("max_added_output_gib", 7), ("max_ram_gib", 16), ("cpu_threads", 2)]
)
def test_server_revision_rejects_unapproved_resource_changes(field, value):
    server = protocol.load_config(protocol.ROOT / "configs/modeling_contrast/protocol_server.json")
    server["resources"][field] = value
    with pytest.raises(ValueError, match="locked protocol"):
        protocol.validate_config(server)


def test_server_revision_rejects_tuning_or_reformatted_config(tmp_path):
    path = protocol.ROOT / "configs/modeling_contrast/protocol_server.json"
    server = protocol.load_config(path)
    server["fresh_cpu"]["steps"] = 65
    with pytest.raises(ValueError, match="locked protocol"):
        protocol.validate_config(server)
    altered = tmp_path / "server.json"
    altered.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="byte"):
        protocol.load_config(altered)
