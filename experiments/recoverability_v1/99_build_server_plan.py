#!/usr/bin/env python3
"""Build the metadata-only handoff plan; this command cannot execute a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compbias.io.manifests import manifest_sha256
from compbias.recoverability.config import load_recoverability_protocol
from compbias.recoverability.design import build_design_report
from compbias.recoverability.evidence import verify_protocol_lock


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_recoverability_protocol(args.config)
    root = Path(__file__).resolve().parents[2]
    lock = verify_protocol_lock(args.protocol_lock, repository_root=root)
    design = build_design_report(protocol)
    payload = {
        "artifact_type": "recoverability_v1_local_server_plan",
        "schema_version": 1,
        "status": protocol.status,
        "config_sha256": manifest_sha256(protocol),
        "protocol_lock_verified": lock.verified,
        "phase_n_scenes": design.phase_n_scenes,
        "bridge_scenes": design.bridge_scenes,
        "bridge_protocol_trajectories": design.bridge_protocol_trajectories,
        "bridge_model_calls": design.bridge_model_calls,
        "phase_c_intake_scenes": design.phase_c_intake_scenes,
        "selected_independent_scenes": design.selected_independent_scenes,
        "total_downstream_forks": design.total_downstream_forks,
        "gpu_invoked": False,
        "training_invoked": False,
        "server_execution_permitted": False,
        "next_stage": "server_measurement_bridge",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
