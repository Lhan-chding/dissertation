from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sr_f1.runtime import (
    SavedActivationOffload,
    actual_compute_cuda_identity,
    configure_compute_execution,
    load_compute_replica,
)


def repair():
    return dict(
        gpu_count=4,
        compute_device_indices=[0, 1, 2, 3],
        repair_sha256="a" * 64,
        recompute_policy="native_pure_delta_rule_nonreentrant_recompute_v1",
        activation_storage="LOCAL_GPU48G_CPU48G_EXACT_EXTERNAL_DISK",
        gpu_activation_budget_bytes=48 << 30,
        cpu_activation_budget_bytes=48 << 30,
    )


def test_compute_policy_is_explicit_and_leaves_original_defaults():
    runtime = SimpleNamespace(identity={})
    configure_compute_execution(runtime, repair())
    assert runtime.activation_gpu_devices == (0,)
    assert runtime.activation_local_compute_storage is True
    assert runtime.compute_recompute_enabled
    assert runtime.identity["compute_parallel"]["gpu_count"] == 4
    assert SavedActivationOffload.default_gpu_budget_bytes == 80 << 30


@pytest.mark.parametrize(
    "key,value",
    [
        ("recompute_policy", "changed"),
        ("gpu_activation_budget_bytes", 80 << 30),
        ("cpu_activation_budget_bytes", 0),
        ("activation_storage", "remote"),
    ],
)
def test_compute_policy_rejects_unauthenticated_memory_topology(key, value):
    current = repair()
    current[key] = value
    with pytest.raises(PermissionError):
        configure_compute_execution(SimpleNamespace(identity={}), current)


def test_compute_identity_checks_all_physical_devices():
    props = [
        SimpleNamespace(
            name="NVIDIA RTX PRO 6000",
            uuid=f"GPU-{i:032x}",
            total_memory=96 << 30,
            major=12,
            minor=0,
        )
        for i in range(4)
    ]
    output = "\n".join(f"{p.name}, {p.uuid}, 580.1" for p in props)
    with (
        patch("torch.cuda.device_count", return_value=4),
        patch("torch.cuda.get_device_properties", side_effect=lambda i: props[i]),
        patch("sr_f1.runtime.subprocess.run", return_value=SimpleNamespace(stdout=output)),
    ):
        observed = actual_compute_cuda_identity(repair())
        assert len(observed["compute_devices"]) == 4
        assert all(d["role"] == "sequence_compute" for d in observed["compute_devices"])
        props[3].uuid = props[2].uuid
        with pytest.raises(PermissionError, match="distinct physical"):
            actual_compute_cuda_identity(repair())


def test_local_activation_storage_cannot_silently_apply_to_cpu():
    with pytest.raises(PermissionError, match="replica compute GPU"):
        SavedActivationOffload(torch.nn.Linear(2, 2), gpu_devices=(0,), local_compute_storage=True)


def test_replica_cannot_change_cuda_visibility_after_initialization():
    with patch("torch.cuda.is_initialized", return_value=True), pytest.raises(PermissionError):
        load_compute_replica({}, "/tmp", 1, "GPU-abcdef")
