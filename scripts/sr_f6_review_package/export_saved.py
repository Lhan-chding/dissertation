"""Read-only saved-data projection; stdout is a ZIP, never run model/scoring code."""

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path("/projects/_ssd/varunssd/louis-ssvc/sr_f6_semantic_factorial_20261027")
KEEP = {
    "generated_text",
    "raw_text",
    "full_text",
    "full_decoded_text",
    "generated_text_with_special_tokens",
    "assistant_prefill",
    "prefill",
    "finish_reason",
    "truncated",
    "generation_status",
    "technical_validation_errors",
    "seed",
    "group_seed",
    "group_row_index",
    "group_size",
    "seed_semantics",
    "prompt_token_count",
    "completion_token_count",
    "generation_seconds",
    "sampling_parameters",
    "sampling_hash",
    "max_new_tokens",
    "model_input",
    "prompt_sha256",
    "prompt_version",
    "policy_hash",
    "policy_tensor_hash",
    "parent_content_hash",
    "source_hash",
    "source_sha256",
    "data_hash",
    "run_id",
    "step",
    "phase",
    "execution_mode",
    "protocol",
    "sr_f6_protocol_version",
    "format_protocol_error",
    "boundary_char",
    "balanced_cut_char",
    "final_tag_cut_char",
    "input_overlay_sha256",
    "pilot_spec_sha256",
}


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


def export(section):
    if section not in ("data", "b0", "b1", "b2", "train_b0", "train_b1", "train_b2"):
        raise ValueError("Unknown fixed section")
    sources = []
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:

        def read(path):
            path = Path(path)
            if not path.resolve().is_relative_to(ROOT.resolve()) or path.is_symlink():
                raise ValueError("Source escapes frozen root")
            data = path.read_bytes()
            sources.append(
                {
                    "source": str(path.relative_to(ROOT)),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            return data

        def copy(path, name):
            archive.writestr(name, read(path))

        if section == "data":
            for panel in ("DEV_EVAL", "DEV_PILOT"):
                public = ROOT / f"PHASE_RUNS/PILOT_V3/INPUTS_V3/{panel}/MODEL_INPUTS.jsonl"
                copy(public, f"04_DESIGN/{panel}_MODEL_INPUTS.jsonl")
                copy(
                    ROOT / f"DATA/{panel}/TASKS_PRIVATE.jsonl",
                    f"04_DESIGN/{panel}_TASKS_PRIVATE.jsonl",
                )
                images = {}
                for line in public.read_bytes().splitlines():
                    row = json.loads(line)["model_input"]
                    if row.get("image_file"):
                        images[row["image_file"]] = row["image_sha256"]
                for rel, expected in images.items():
                    data = read(ROOT / rel)
                    if hashlib.sha256(data).hexdigest() != expected:
                        raise ValueError("Image hash differs")
                    name = f"04_DESIGN/{rel}"
                    if name not in archive.namelist():
                        archive.writestr(name, data)
            copy(
                ROOT / "PHASE_RUNS/PILOT_V3/PILOT_FROZEN/PILOT_SPEC_FROZEN.json",
                "04_DESIGN/PILOT_SPEC_FROZEN.json",
            )
            copy(ROOT / "PHASE_RUNS/PILOT_V3/JOBS/PLAN.json", "04_DESIGN/PLAN.json")
        elif section.startswith("train_"):
            parent = section[-2:]
            for arm in ("U10", "U11", "C_SHUF"):
                run = ROOT / f"PHASE_RUNS/PILOT_V3/PILOT_FROZEN/runs/{parent}/r0/{arm}"
                for step in range(1, 13):
                    copy(
                        run / f"rollouts/step-{step:03d}/credit.json",
                        f"03_TRAINING/{parent}_{arm}/step-{step:03d}_credit.json",
                    )
                    # Preserve the original full UPDATE, including probability references.
                    copy(
                        run / f"checkpoints/step-{step:03d}/UPDATE.json",
                        f"03_TRAINING/{parent}_{arm}/step-{step:03d}_UPDATE.json",
                    )
        else:
            for arm in ("parent", "U10", "U11", "C_SHUF"):
                suffix = "parent" if arm == "parent" else f"{arm}/step12"
                endpoint = ROOT / f"PHASE_RUNS/PILOT_V3/EVALUATION/{section}/{suffix}"
                report_bytes = read(endpoint / "COMPLETE.json")
                report = json.loads(report_bytes)
                archive.writestr(f"02_EVALUATION/{section}_{arm}_COMPLETE.json", report_bytes)
                projected = []
                for record in report["records"]:
                    raw_path = endpoint / "groups" / record["qid"] / f"draw-{record['draw']}.json"
                    raw_bytes = read(raw_path)
                    raw = json.loads(raw_bytes)
                    selected = {k: v for k, v in raw.items() if k in KEEP}
                    if not any(
                        isinstance(selected.get(k), str) and selected[k]
                        for k in ("full_text", "raw_text", "generated_text")
                    ):
                        raise ValueError("Missing answer text")
                    projected.append(
                        {
                            "endpoint": f"{section}/{suffix}",
                            "record": record,
                            "answer": selected,
                            "source": sources[-1],
                            "omitted_raw_fields": sorted(set(raw) - KEEP),
                        }
                    )
                if len(projected) != 192:
                    raise ValueError("Unexpected endpoint record count")
                archive.writestr(
                    f"02_EVALUATION/{section}_{arm}_answers.jsonl",
                    b"".join(encoded(row) for row in projected),
                )
        archive.writestr(f"05_PROVENANCE/{section}_SOURCE_INDEX.json", encoded(sources))
    sys.stdout.buffer.write(buffer.getvalue())


if __name__ == "__main__":
    export(sys.argv[1])
