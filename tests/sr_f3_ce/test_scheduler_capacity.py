"""Operational scheduler capacity tests; no SSH, allocation, or model calls."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

QOS = "soujanya-poria-startfund-2026-03"
QOS_FIELDS = (
    "Name",
    "GrpTRES",
    "MaxTRESPU",
    "MaxTRES",
    "GrpJobs",
    "GrpSubmit",
    "MaxJobsPU",
    "MaxSubmitPU",
    "Flags",
)
ASSOC_FIELDS = (
    "Cluster",
    "Account",
    "User",
    "Partition",
    "GrpTRES",
    "MaxTRES",
    "MaxTRESPerNode",
    "GrpJobs",
    "GrpSubmit",
    "MaxJobs",
    "MaxSubmit",
)
MEMORY_PER_TASK = 90 * 1024


def table(fields, rows):
    return "\n".join(
        ["|".join(fields), *("|".join(row.get(k, "") for k in fields) for row in rows)]
    )


def qos_text(**overrides):
    value = dict(
        Name=QOS,
        GrpTRES="gres/gpu:pro6000=8,gres/gpu:6000ada=0,gres/gpu:a100=0",
        MaxSubmitPU="5",
    )
    value.update(overrides)
    return table(QOS_FIELDS, [value])


def assoc_text(**overrides):
    value = dict(Cluster="eee", Account="rose", User="alice", Partition="")
    value.update(overrides)
    return table(ASSOC_FIELDS, [value])


def node_text(name="pro6000-1", gpus=8, memory=760832, cpus=64, features="highmem", state="mix"):
    return f"{name}|gpu:pro6000:{gpus}(S:0-1)|{memory}|{cpus}|{features}|{state}\n"


@pytest.fixture(scope="module")
def capacity_module():
    path = Path(__file__).resolve().parents[2] / "scripts/sr_f3_ce/scheduler_capacity.py"
    spec = importlib.util.spec_from_file_location("scheduler_capacity_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def derive(module, *, qos=None, assoc=None, nodes=None):
    return module.derive(
        qos_text() if qos is None else qos,
        assoc_text() if assoc is None else assoc,
        node_text() if nodes is None else nodes,
        user="alice",
    )


class FakeScheduler:
    def __init__(self):
        self.qos = qos_text()
        self.assoc = assoc_text()
        self.nodes = node_text()
        self.commands = []
        self.fail_at = None
        self.timeout_at = None

    def __call__(self, command, **kwargs):
        self.commands.append((list(command), dict(kwargs)))
        assert command[0] in {"sacctmgr", "sinfo"}
        assert kwargs == dict(capture_output=True, text=True, timeout=20, check=False)
        if len(self.commands) == self.timeout_at:
            raise subprocess.TimeoutExpired(command, 20)
        if len(self.commands) == self.fail_at:
            return SimpleNamespace(returncode=1, stdout="", stderr="scheduler unavailable")
        if command[0] == "sinfo":
            body = self.nodes
        else:
            assert command[:3] == ["sacctmgr", "-P", "show"]
            assert command[3] in {"qos", "assoc"}
            body = self.qos if command[3] == "qos" else self.assoc
        return SimpleNamespace(returncode=0, stdout=body, stderr="")


@pytest.fixture
def capacity_env(capacity_module, tmp_path):
    scheduler = FakeScheduler()
    now = [100.0]
    value = dict(root=str(tmp_path), user="alice", account="rose", partition="cluster02")
    instance = capacity_module.Capacity(value, execute=scheduler, clock=lambda: now[0])
    return SimpleNamespace(
        module=capacity_module,
        root=tmp_path,
        instance=instance,
        scheduler=scheduler,
        now=now,
        value=value,
    )


def intent_for(env, answer):
    return dict(
        capacity_snapshot_path=answer["capacity_snapshot_path"],
        capacity_snapshot_sha256=answer["capacity_snapshot_sha256"],
        capacity_limits={key: answer[key] for key in env.module.LIMIT_KEYS},
        max_tasks_per_job=answer["max_tasks_per_job"],
    )


def snapshots(env):
    return list((env.root / "operations/capacity/snapshots").glob("*.json"))


def test_actual_headers_eight_gpu_five_submissions_and_one_node(capacity_module):
    limits, nodes = derive(capacity_module)
    assert limits == dict(
        gpu_limit=8,
        submitted_limit=5,
        max_tasks_per_job=8,
        gpu_per_user_limit=None,
    )
    assert nodes == [dict(node="pro6000-1", max_tasks=8, state="mix")]


@pytest.mark.parametrize("gpu_limit", [0, 3, 8, 12, 20])
def test_limits_follow_scheduler_without_five_or_eight_gpu_ceiling(capacity_module, gpu_limit):
    limits, _ = derive(
        capacity_module,
        qos=qos_text(GrpTRES=f"gres/gpu:pro6000={gpu_limit}"),
        nodes=node_text(gpus=24, memory=24 * MEMORY_PER_TASK, cpus=96),
    )
    assert limits["gpu_limit"] == gpu_limit
    assert limits["max_tasks_per_job"] == gpu_limit


@pytest.mark.parametrize(
    ("tres", "expected"),
    [
        ("gres/gpu:pro6000=8,gres/gpu:a100=0,gres/gpu:l40=0", 8),
        ("gres/gpu=6,gres/gpu:pro6000=8", 6),
        ("gres/gpu=8,gres/gpu:pro6000=8", 8),
        ("gres/gpu=0,gres/gpu:pro6000=8", 0),
        ("", None),
        ("gres/gpu:pro6000=UNLIMITED", None),
    ],
)
def test_typed_gpu_limits_are_intersections_not_sums(capacity_module, tres, expected):
    limits, _ = derive(capacity_module, qos=qos_text(GrpTRES=tres))
    assert limits["gpu_limit"] == expected
    assert limits["max_tasks_per_job"] == (8 if expected is None else expected)


def test_per_user_gpu_limit_is_distinct_from_group_limit(capacity_env):
    env = capacity_env
    env.scheduler.qos = qos_text(MaxTRESPU="gres/gpu:pro6000=4")
    env.scheduler.assoc = assoc_text(GrpTRES="gres/gpu:pro6000=3", MaxSubmit="4")
    limits = env.instance.limits()
    assert limits["gpu_limit"] == 8
    assert limits["gpu_per_user_limit"] == 3
    assert limits["max_tasks_per_job"] == 3
    assert limits["submitted_limit"] == 4
    assert env.instance.headroom(dict(gpus=5, user_gpus=1, submitted=2), limits) == {
        "gpus": 2,
        "submitted": 2,
    }
    with pytest.raises(PermissionError, match="Per-user GPU accounting"):
        env.instance.headroom(dict(gpus=5, submitted=2), limits)


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (dict(gpus=8, submitted=1), dict(gpus=0, submitted=4)),
        (dict(gpus=1, submitted=5), dict(gpus=7, submitted=0)),
        (dict(gpus=3, submitted=2), dict(gpus=5, submitted=3)),
    ],
)
def test_gpu_and_submitted_job_headroom_are_independent(capacity_env, usage, expected):
    assert capacity_env.instance.assert_within(usage) == expected


@pytest.mark.parametrize("usage", [dict(gpus=9, submitted=1), dict(gpus=1, submitted=6)])
def test_actual_over_limit_reservations_are_rejected(capacity_env, usage):
    with pytest.raises(PermissionError, match="Actual reservations exceed"):
        capacity_env.instance.assert_within(usage)


@pytest.mark.parametrize(
    ("qos_changes", "assoc_changes", "expected"),
    [
        ({"MaxTRES": "cpu=12,mem=720G,gres/gpu:pro6000=8"}, {}, 3),
        ({"MaxTRES": "mem=270G"}, {}, 3),
        ({"MaxTRES": "mem=276480M"}, {}, 3),
        ({"MaxTRES": "mem=283115520K"}, {}, 3),
        ({"MaxTRES": "mem=1T"}, {}, 8),
        ({}, {"MaxTRES": "gres/gpu:pro6000=2"}, 2),
        ({}, {"MaxTRESPerNode": "cpu=8,mem=720G"}, 2),
        ({"MaxTRES": "gres/gpu:pro6000=0"}, {}, 0),
    ],
)
def test_per_job_limits_include_worker_cpu_and_memory(
    capacity_module,
    qos_changes,
    assoc_changes,
    expected,
):
    limits, _ = derive(
        capacity_module,
        qos=qos_text(**qos_changes),
        assoc=assoc_text(**assoc_changes),
    )
    assert limits["max_tasks_per_job"] == expected


@pytest.mark.parametrize(
    ("nodes", "expected"),
    [
        (node_text(memory=4 * MEMORY_PER_TASK), 4),
        (node_text(cpus=12), 3),
        (node_text(memory=MEMORY_PER_TASK - 1), 0),
        (node_text(gpus=2) + node_text(name="second", gpus=3), 3),
        (node_text(state="drain") + node_text(name="second", gpus=2), 2),
        (node_text(state="down*"), 0),
        (node_text(state="unknown"), 0),
        (node_text(features="lowmem"), 0),
        (node_text(features="fast,highmem"), 8),
        ("", 0),
    ],
)
def test_worker_capacity_must_fit_a_single_eligible_node(capacity_module, nodes, expected):
    limits, _ = derive(capacity_module, nodes=nodes)
    assert limits["max_tasks_per_job"] == expected


@pytest.mark.parametrize(
    "raw",
    ["", "Name|GrpTRES", "Name|Name\nx|x", "Name|GrpTRES\nx", "bad\nheader"],
)
def test_malformed_or_missing_qos_output_fails_closed(capacity_module, raw):
    with pytest.raises(PermissionError):
        derive(capacity_module, qos=raw)


@pytest.mark.parametrize(
    "changes",
    [
        {"GrpTRES": "gres/gpu:pro6000=unknown"},
        {"GrpTRES": "gres/gpu:pro6000=8,gres/gpu:pro6000=4"},
        {"GrpTRES": "gres/gpu:pro6000"},
        {"MaxSubmitPU": "unknown"},
        {"MaxTRES": "cpu=-2"},
        {"MaxTRES": "mem=unknown"},
        {"Name": "some-other-qos"},
    ],
)
def test_unknown_or_ambiguous_limits_are_not_unlimited(capacity_module, changes):
    with pytest.raises(PermissionError):
        derive(capacity_module, qos=qos_text(**changes))


@pytest.mark.parametrize("flags", ["Relative", "DenyOnLimit,Relative", "relative"])
def test_relative_qos_percentages_are_not_physical_gpu_counts(capacity_module, flags):
    with pytest.raises(PermissionError, match="Relative"):
        derive(capacity_module, qos=qos_text(Flags=flags))


@pytest.mark.parametrize(
    "changes",
    [{"User": "other"}, {"Account": "other"}, {"Partition": "other"}],
)
def test_missing_selected_association_fails_closed(capacity_module, changes):
    with pytest.raises(PermissionError, match="association is unavailable"):
        derive(capacity_module, assoc=assoc_text(**changes))


@pytest.mark.parametrize(
    ("qos_changes", "assoc_changes"),
    [
        ({"GrpSubmit": "20"}, {}),
        ({"GrpTRES": "gres/gpu:pro6000=8,cpu=32"}, {}),
        ({"GrpTRES": "gres/gpu:pro6000=8,mem=720G"}, {}),
        ({"MaxTRESPU": "node=1"}, {}),
        ({}, {"GrpTRES": "cpu=32"}),
    ],
)
def test_additional_aggregate_limits_require_explicit_accounting(
    capacity_module,
    qos_changes,
    assoc_changes,
):
    with pytest.raises(PermissionError):
        derive(capacity_module, qos=qos_text(**qos_changes), assoc=assoc_text(**assoc_changes))


@pytest.mark.parametrize(
    "nodes",
    [
        node_text(memory=-1),
        node_text(cpus=-1),
        node_text(memory="unknown"),
        node_text(cpus="unknown"),
        node_text() + node_text(gpus=4),
        "node|gpu:pro6000:8|760832|32|highmem\n",
        "node|gpu:pro6000:8,gpu:pro6000:4|760832|32|highmem|idle\n",
    ],
)
def test_malformed_or_duplicate_node_capacity_fails_closed(capacity_module, nodes):
    with pytest.raises(PermissionError):
        derive(capacity_module, nodes=nodes)


def test_limit_discovery_records_only_small_readonly_scheduler_queries(capacity_env):
    env = capacity_env
    answer = env.instance.limits()
    commands = [command for command, _ in env.scheduler.commands]
    assert len(commands) == 3
    assert commands[0][:4] == ["sacctmgr", "-P", "show", "qos"]
    assert "name=" + QOS in commands[0]
    assert "user=alice" in commands[1]
    assert "account=rose" in commands[1]
    assert commands[2] == ["sinfo", "-h", "-N", "-p", "cluster02", "-o", "%N|%G|%m|%c|%f|%t"]
    path = Path(answer["capacity_snapshot_path"])
    assert path.parent == env.root / "operations/capacity/snapshots"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == answer["capacity_snapshot_sha256"]
    record = json.loads(path.read_text())
    assert record["root"] == str(env.root)
    assert record["qos"] == QOS
    assert record["observed_at"] == 100.0
    assert record["limits"] == {key: answer[key] for key in env.module.LIMIT_KEYS}
    assert [row["command"] for row in record["observations"]] == commands


def test_cache_is_fresh_for_less_than_sixty_seconds_and_returns_copies(capacity_env):
    env = capacity_env
    answer = env.instance.limits()
    answer["gpu_limit"] = 999
    env.now[0] += 59
    assert env.instance.limits()["gpu_limit"] == 8
    assert len(env.scheduler.commands) == 3
    env.now[0] += 1
    env.scheduler.qos = qos_text(GrpTRES="gres/gpu:pro6000=4")
    assert env.instance.limits()["gpu_limit"] == 4
    assert len(env.scheduler.commands) == 6
    assert len(snapshots(env)) == 2


def test_fresh_query_and_clock_rollback_do_not_reuse_cache(capacity_env):
    env = capacity_env
    env.instance.limits()
    env.instance.limits(fresh=True)
    env.now[0] -= 1
    env.instance.limits()
    assert len(env.scheduler.commands) == 9
    assert len(snapshots(env)) == 3


def test_constructor_preserves_its_scope_if_caller_mutates_input(capacity_env):
    env = capacity_env
    env.value["user"] = "someone-else"
    answer = env.instance.limits()
    assert "user=alice" in env.scheduler.commands[1][0]
    assert env.instance.verify_snapshot(intent_for(env, answer))["gpu_limit"] == 8


@pytest.mark.parametrize("failed_command", [1, 2, 3])
def test_failed_scheduler_query_creates_no_allocation_or_snapshot(capacity_env, failed_command):
    env = capacity_env
    env.scheduler.fail_at = failed_command
    with pytest.raises(RuntimeError, match="scheduler unavailable"):
        env.instance.limits()
    assert len(env.scheduler.commands) == failed_command
    assert snapshots(env) == []
    assert env.instance._cached is None


def test_timeout_is_not_treated_as_unlimited_capacity(capacity_env):
    env = capacity_env
    env.scheduler.timeout_at = 1
    with pytest.raises(subprocess.TimeoutExpired):
        env.instance.limits()
    assert snapshots(env) == []
    assert env.instance._cached is None


def test_failed_fresh_query_does_not_return_cached_capacity(capacity_env):
    env = capacity_env
    answer = env.instance.limits()
    original = Path(answer["capacity_snapshot_path"]).read_bytes()
    env.scheduler.fail_at = 4
    with pytest.raises(RuntimeError):
        env.instance.limits(fresh=True)
    assert len(snapshots(env)) == 1
    assert Path(answer["capacity_snapshot_path"]).read_bytes() == original


def test_historical_capacity_snapshot_remains_valid_after_live_limit_drop(capacity_env):
    env = capacity_env
    first = env.instance.limits()
    historical_intent = intent_for(env, first)
    old_path = Path(first["capacity_snapshot_path"])
    old_bytes = old_path.read_bytes()
    env.scheduler.qos = qos_text(GrpTRES="gres/gpu:pro6000=4")
    current = env.instance.limits(fresh=True)
    assert current["gpu_limit"] == 4
    assert current["capacity_snapshot_path"] != first["capacity_snapshot_path"]
    query_count = len(env.scheduler.commands)
    historical = env.instance.verify_snapshot(historical_intent)
    assert historical["gpu_limit"] == 8
    assert historical["max_tasks_per_job"] == 8
    assert len(env.scheduler.commands) == query_count
    assert old_path.read_bytes() == old_bytes
    with pytest.raises(PermissionError, match="Actual reservations exceed"):
        env.instance.assert_within(dict(gpus=8, submitted=2), current)


@pytest.mark.parametrize("change", ["hash", "limits", "pool_size", "boolean_pool_size"])
def test_snapshot_claim_binding_rejects_mismatch(capacity_env, change):
    env = capacity_env
    intent = intent_for(env, env.instance.limits())
    if change == "hash":
        intent["capacity_snapshot_sha256"] = "0" * 64
    elif change == "limits":
        intent["capacity_limits"]["gpu_limit"] = 12
    elif change == "pool_size":
        intent["max_tasks_per_job"] = 4
    else:
        intent["max_tasks_per_job"] = True
    with pytest.raises(PermissionError):
        env.instance.verify_snapshot(intent)


@pytest.mark.parametrize("change", ["root", "derived_limits", "derived_nodes", "observations"])
def test_valid_hash_cannot_hide_wrong_scope_or_wrong_derivation(capacity_env, change):
    env = capacity_env
    intent = intent_for(env, env.instance.limits())
    path = Path(intent["capacity_snapshot_path"])
    record = json.loads(path.read_text())
    if change == "root":
        record["root"] = str(env.root / "other")
    elif change == "derived_limits":
        record["limits"]["gpu_limit"] = 12
    elif change == "derived_nodes":
        record["eligible_nodes"][0]["max_tasks"] = 12
    else:
        record["observations"] = record["observations"][:2]
    path.write_text(json.dumps(record))
    intent["capacity_snapshot_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(PermissionError):
        env.instance.verify_snapshot(intent)


def test_snapshot_must_resolve_inside_its_own_retained_directory(capacity_env):
    env = capacity_env
    intent = intent_for(env, env.instance.limits())
    outside = env.root / "outside.json"
    outside.write_bytes(Path(intent["capacity_snapshot_path"]).read_bytes())
    symlink = env.root / "operations/capacity/snapshots/escape.json"
    symlink.symlink_to(outside)
    intent["capacity_snapshot_path"] = str(symlink)
    with pytest.raises(PermissionError, match="Retained scheduler capacity"):
        env.instance.verify_snapshot(intent)


def test_explicit_unlimited_limits_remain_distinct_from_zero(capacity_env):
    env = capacity_env
    env.scheduler.qos = qos_text(GrpTRES="", MaxSubmitPU="UNLIMITED")
    unlimited = env.instance.limits()
    assert unlimited["max_tasks_per_job"] == 8
    assert env.instance.headroom(dict(gpus=12, submitted=6), unlimited) == {
        "gpus": math.inf,
        "submitted": math.inf,
    }
    env.scheduler.qos = qos_text(GrpTRES="gres/gpu:pro6000=0", MaxSubmitPU="0")
    zero = env.instance.limits(fresh=True)
    assert env.instance.headroom(dict(gpus=0, submitted=0), zero) == {
        "gpus": 0,
        "submitted": 0,
    }
    assert zero["max_tasks_per_job"] == 0


def test_partition_and_unpartitioned_user_association_limits_intersect(capacity_module):
    base = dict(Cluster="eee", Account="rose", User="alice")
    assoc = table(
        ASSOC_FIELDS,
        [dict(base, MaxSubmit="4"), dict(base, Partition="cluster02", MaxSubmit="3")],
    )
    limits, _ = derive(capacity_module, assoc=assoc)
    assert limits["submitted_limit"] == 3


def test_readonly_snapshot_verification_does_not_modify_intent(capacity_env):
    env = capacity_env
    intent = intent_for(env, env.instance.limits())
    original = copy.deepcopy(intent)
    env.instance.verify_snapshot(intent)
    assert intent == original
