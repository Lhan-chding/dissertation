"""J23 identity wrapper around the unchanged, audited SER-J2 numerical engine.

The student serialization schema remains ``ser-j2-student-state-v1`` for byte
format compatibility. It is not experiment identity: newly created students
always carry SER_J23_20261008 and the full logical arm in their identity.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from ..exposure_substitution.runtime import (
    SERRuntime as LegacyRuntime,
)
from ..exposure_substitution.runtime import (
    load_student_into_backend as load_student_into_backend,
)
from ..exposure_substitution.runtime import (
    read_student_checkpoint as read_student_checkpoint,
)


class SERRuntime(LegacyRuntime):
    def __init__(self, *args, **kwargs):
        from .schema import ARMS, EXPERIMENT_ID

        identity = kwargs.get("identity", {})
        if (
            identity.get("experiment_id") != EXPERIMENT_ID
            or identity.get("logical_arm_id") not in ARMS
            or identity.get("arm") != identity.get("logical_arm_id")
        ):
            raise ValueError("New J23 runtime requires its genuine experiment/logical-arm identity")
        super().__init__(*args, **kwargs)


def source_binding(run, *, role="trainer"):
    from .schema import verify_source_bound

    return verify_source_bound(Path(run).resolve(), role=role)


def build_runtime(run, backend, parent, block, arm, *, technical=False):
    from ..modeling_v3.io import canonical_hash
    from .schedule import ExplicitScheduleSampler
    from .schema import ARMS, EXPERIMENT_ID, training_rows, verify_frozen
    from .training import SCHEDULE_SEEDS, encode_rows

    if parent not in ("S96", "REP96") or type(block) is not int or block not in range(3):
        raise ValueError("Registered parent and block required")
    if arm not in ARMS:
        raise ValueError("Full J23 logical arm ID required")
    run = Path(run).resolve()
    source = source_binding(run)
    frozen = None if technical else verify_frozen(run, role="trainer")
    rows = training_rows(run, arm, technical=technical)
    encoded = encode_rows(backend, rows)
    sampler = ExplicitScheduleSampler.from_run(run, block, arm, technical=technical)
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "phase_id": EXPERIMENT_ID,
        "logical_arm_id": arm,
        "parent": parent,
        "block": block,
        "arm": arm,
        "source_bound_hash": canonical_hash(source),
        "frozen_plan_hash": None if technical else canonical_hash(frozen),
        "schedule_id": sampler.schedule_id,
        "seed": SCHEDULE_SEEDS[block],
        "encoded_training_hash": canonical_hash({k: v.__dict__ for k, v in encoded.items()}),
        "parent_checkpoint_sha256": backend.receipt["checkpoint"]["sha256"],
        "loss": "completion_sequence_mean_slots16_donor_isolated_44314",
        "steps": 256,
        "technical_only": technical,
    }
    return SERRuntime.from_frozen_backend(
        backend,
        seed=SCHEDULE_SEEDS[block],
        identity=identity,
        sampler=sampler,
        encoded=encoded,
        metadata=rows,
    )


def legacy_run_directory(run, *, technical=False, machine=None):
    from .schema import read_training_bound

    registered = read_training_bound(run, "machine.json", technical=technical)
    if machine is not None:
        candidate = (
            json.loads(Path(machine).read_text())
            if not isinstance(machine, dict)
            else copy.deepcopy(machine)
        )
        if candidate != registered:
            raise ValueError("Machine override differs from bound J23 machine")
    path = registered.get("legacy_run") or registered.get("legacy_run_path")
    if not isinstance(path, str) or not path:
        raise ValueError("Bound machine must explicitly name legacy_run")
    return Path(path).resolve()


def load_parent_backend(run, parent, machine=None, allow_gpu=False, *, technical=False):
    """Preserve the historical loader checks without opening any target manifests.

    The historical ``verify_frozen`` hashes its entire archive, including truth.
    Workers instead verify its immutable header and only the loader control files;
    the exact same FrozenBackend performs parent restoration and numerical checks.
    """
    from ..exposure_substitution.schema import (
        _frozen_header,
        file_digest,
    )
    from ..exposure_substitution.schema import (
        read_bound as read_legacy_bound,
    )
    from ..modeling_v3.io import canonical_hash
    from ..protocol_state_probes.inference import FrozenBackend
    from .schema import read_training_bound

    if allow_gpu is not True:
        raise PermissionError("J23 inherited model execution requires --allow-gpu")
    if parent not in ("S96", "REP96"):
        raise ValueError("Unregistered inherited parent")
    source_binding(run, role="control")
    legacy_run = legacy_run_directory(run, technical=technical, machine=machine)
    registered = read_training_bound(run, "machine.json", technical=technical)
    expected_legacy_plan = registered.get("legacy_frozen_plan_sha256")
    if (
        not isinstance(expected_legacy_plan, str)
        or len(expected_legacy_plan) != 64
        or file_digest(legacy_run / "FROZEN_PLAN.json") != expected_legacy_plan
    ):
        raise ValueError("Bound historical frozen-plan bytes are missing or have changed")
    frozen = _frozen_header(legacy_run)
    historical_machine = read_legacy_bound(legacy_run, "machine.json")
    for key in ("runtime_path", "frozen_protocol_path"):
        external = frozen.get("external_inputs", {}).get(key, {})
        if external.get("status") != "VERIFIED_BYTES" or file_digest(
            historical_machine[key]
        ) != external.get("sha256"):
            raise ValueError("Unverified or modified historical external input: " + key)
    if frozen.get("later_exclusion_status") != "VERIFIED_EXPLICIT_SCAN":
        raise ValueError("Historical explicit exclusion audit is required")
    records = read_legacy_bound(legacy_run, "parent_availability.json")["checkpoints"]
    selected = [record for record in records if record["checkpoint_id"] == parent]
    if len(selected) != 1 or selected[0].get("status") != "AVAILABLE":
        raise ValueError("Registered original parent is unavailable")
    historical = json.loads(Path(historical_machine["frozen_protocol_path"]).read_text())
    if (
        historical_machine.get("frozen_protocol_hash") is not None
        and canonical_hash(historical) != historical_machine["frozen_protocol_hash"]
    ):
        raise ValueError("Historical frozen protocol binding differs")
    return FrozenBackend(
        historical_machine["runtime_path"], selected[0], historical, allow_gpu=True
    )
