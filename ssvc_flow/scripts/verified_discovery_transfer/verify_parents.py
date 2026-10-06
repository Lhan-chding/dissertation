"""Verify the two existing parents on CPU without loading the base model."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.modeling_v3.io import sha256_file
from src.optimizer_fork import state_hash
from src.protocol_state_probes.checkpoint_catalog import check_checkpoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    records = json.loads(Path(args.manifest).read_text())
    records = [row for row in records if row["id"] in {"S96", "REP96"}]
    if {r["id"] for r in records} != {"S96", "REP96"}:
        raise ValueError("Both registered parents are required")
    result = check_checkpoints(records)
    import torch

    for row in result["checkpoints"]:
        if row["status"] != "AVAILABLE":
            continue
        spec = row["checkpoint"]
        if sha256_file(spec["path"]) != spec["sha256"]:
            raise ValueError("Parent file digest differs: " + row["checkpoint_id"])
        payload = torch.load(spec["path"], map_location="cpu", weights_only=True)
        if payload["identity"] != spec["identity"]:
            raise ValueError("Parent identity differs")
        if payload["state_hash"] != spec["state_hash"]:
            raise ValueError("Parent state binding differs")
        state = payload["state"]
        if state_hash(state) != spec["state_hash"]:
            raise ValueError("Parent full state digest differs")
        params = state["parameters"]
        if len(params) != 192 or any("lora_" not in k for k in params):
            raise ValueError("Expected exact 96-module LoRA parameter set")
        if any(not torch.isfinite(v).all() for v in params.values()):
            raise ValueError("Nonfinite parent LoRA")
        row["checkpoint_content_verified"] = True
        row["parameter_inventory"] = {
            k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in params.items()
        }
        row["complete_state_fields"] = sorted(state)
        row["state_metadata"] = state["metadata"]
        del state, payload
    result["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["checkpoint_tensors_loaded"] = True
    result["model_loaded"] = False
    result["scope"] = (
        "CPU file, complete state and parameter identity; GPU restore remains separate"
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "parents": [
                    {
                        "id": r["checkpoint_id"],
                        "status": r["status"],
                        "content_verified": r.get("checkpoint_content_verified", False),
                    }
                    for r in result["checkpoints"]
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
