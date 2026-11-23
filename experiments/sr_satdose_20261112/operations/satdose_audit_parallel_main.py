#!/usr/bin/env python3
"""CPU-only final audit and original-order request index for the two MAIN lanes.

The remote process reads Slurm and existing evidence and streams a gzip result.
The local driver preserves that stream and verifies its hashes and full coverage.
It publishes a successful local receipt only after all fourteen endpoints pass.
There is no scheduler mutation, generation, or partial behavioral scoring CLI.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1")
EXTERNAL = ROOT / "external_parallel_main_v1"
RELEASE = ROOT / "execution_releases/parallel_main_v1/RELEASE.json"
REMOTE_PYTHON = "/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python"
AUDITOR = "operations/satdose_audit_parallel_main.py"
CONTROL = "execution/parallel_main_control.py"
MANIFEST_SHA = "cf4c4fd307d90de3475c5e01885b92196dbf0e481c84461f626246befd77d884"
FROZEN_IDS_SHA = "0a7498ba3657e736ac013ca6734283f6d18a642c648f35d13d357be98ae4e36d"
QOS = "soujanya-poria-startfund-2026-03"
SEEDS = (11031, 11032)
WAVES = (("START", 0), ("T0", 8), ("T0", 16), ("T0", 24), ("T1", 8), ("T1", 16), ("T1", 24))
MAX_BUNDLE_BYTES = 32 * 1024 * 1024
STATUS = "PARALLEL_MAIN_ALL_13760_CAPTURES_INDEPENDENTLY_VERIFIED"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_json(raw):
    def pairs(items):
        obj = dict(items)
        require(len(items) == len(obj), "Duplicate JSON key")
        return obj

    def reject(value):
        raise ValueError("Nonfinite JSON number " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def ordinary(path):
    require(
        path.resolve(strict=True) == path and path.is_file(),
        "Noncanonical evidence file: " + str(path),
    )
    return path


def source_bytes(path, expected):
    raw = ordinary(path).read_bytes()
    require(digest(raw) == expected, "SHA mismatch: " + str(path))
    return raw


def reference(root, path):
    return dict(path=str(path.relative_to(root)), sha256=digest(ordinary(path).read_bytes()))


def release_module(release_sha):
    release = load_json(source_bytes(RELEASE, release_sha))
    here = EXTERNAL / AUDITOR
    require(Path(__file__).resolve(strict=True) == here, "Unexpected remote auditor location")
    source_bytes(here, release["parallel_sources"][AUDITOR])
    path = EXTERNAL / CONTROL
    raw = source_bytes(path, release["parallel_sources"][CONTROL])
    spec = importlib.util.spec_from_file_location("satdose_audited_parallel_main", path)
    controller = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), "exec"), controller.__dict__)
    serial, validated, terminal, planner = controller.bootstrap(RELEASE, release_sha)
    require(
        validated == release and controller.WAVES == WAVES and tuple(controller.SEEDS) == SEEDS,
        "Release or wave/seed definitions changed",
    )
    return controller, serial, release, terminal, planner


def exact_membership(expected, actual, label):
    """Reject a compensated omission/duplicate even when total length is exact."""
    want, have = Counter(expected), Counter(actual)
    missing = sorted((want - have).elements())
    extra = sorted((have - want).elements())
    duplicate = sorted(key for key, count in have.items() if count > 1)
    require(
        not missing and not extra and not duplicate,
        f"{label}: missing {len(missing)} {missing[:3]}, extra {len(extra)} {extra[:3]}, "
        f"duplicate IDs {len(duplicate)} {duplicate[:3]}",
    )


def scheduler_rows(raw, job, name):
    rows = [line.split("|") for line in raw.splitlines() if line.strip()]
    require(all(len(row) == 13 for row in rows), "Malformed scheduler accounting for " + str(job))
    ids = [row[0] for row in rows]
    exact_membership(
        [str(job), f"{job}.batch", f"{job}.extern", f"{job}.0"], ids, "Slurm job and steps"
    )
    parent = next(row for row in rows if row[0] == str(job))
    require(
        parent[3:8] == [name, "rose", QOS, "cluster02", "varun024"], "Wrong scheduler job identity"
    )
    for row in rows:
        require(
            row[1:3] == ["COMPLETED", "0:0"]
            and row[10] == parent[10]
            and all(v not in ("", "Unknown", "None") for v in row[8:11]),
            "Unfinished, failed or misplaced scheduler record: " + row[0],
        )
    for column in (11, 12):
        tres = dict(entry.split("=", 1) for entry in parent[column].split(","))
        require(
            tres.get("gres/gpu:pro6000") == "2"
            and tres.get("gres/gpu") == "2"
            and tres.get("cpu") == "8"
            and tres.get("node") == "1",
            "Different job resource allocation",
        )
    return rows


def accounting(job, seed, arm, step):
    cmd = [
        "sacct",
        "-j",
        str(job),
        "-nP",
        "--format=JobIDRaw,State%40,ExitCode,JobName%120,Account%40,QOS%80,Partition,User,Start,End,NodeList,ReqTRES%160,AllocTRES%160",
    ]
    env = {k: v for k, v in os.environ.items() if k != "SLURM_CLUSTERS"}
    obs = subprocess.run(cmd, capture_output=True, text=True, timeout=90, check=True, env=env)
    require(not obs.stderr.strip(), "Slurm accounting warning for " + str(job))
    name = f"LH-SATDOSE-MAIN-{seed}-{arm}-s{step:02d}"
    return dict(argv=cmd, rows=scheduler_rows(obs.stdout, job, name))


def endpoint_metadata(
    m, terminal, observed, key, path, job, release_ref, pair_ref, released_ref, count, ids_sha
):
    seed, arm, step = key
    result_ref = m.ref(path / "RESULT.json")
    result = observed.json(result_ref)
    require(
        result["schema"] == "sr-satdose-main-result-v1"
        and result["status"] == "MAIN_ENDPOINT_COMPLETE"
        and result["permission"] == "NO_TRAINING_OR_SUCCESSOR_EXPERIMENT"
        and (result["start_seed"], result["arm"], result["step"]) == key
        and result["responses"] == count
        and result["job_id"] == job
        and result["release"] == release_ref
        and result["pair"] == pair_ref
        and result["released"] == released_ref
        and result["terminal"] == terminal.receipt
        and result["design_freeze_sha256"] == m.FREEZE_SHA,
        f"Endpoint {key} has foreign or incomplete RESULT",
    )
    for field, location in [
        ("receipt", "endpoint/RECEIPT.json"),
        ("endpoint_binding", "ENDPOINT_BINDING.json"),
        ("close", "CLOSE.json"),
    ]:
        require(observed.ref(result[field]) == path / location, "Relocated endpoint " + field)
    claim_ref = m.ref(path / "CLAIM.json")
    claim = observed.json(claim_ref)
    require(
        claim["release"] == release_ref
        and claim["pair"] == pair_ref
        and claim["job_id"] == job
        and claim["released"] == released_ref
        and claim["terminal"] == terminal.receipt
        and claim["process_identity"] == result["process_identity"]
        and claim["request_ids_sha256"] == ids_sha
        and claim["output"] == str(path / "endpoint")
        and (claim["start_seed"], claim["arm"], claim["step"]) == key
        and claim["responses"] == count,
        f"Endpoint {key} claim does not match RESULT",
    )
    binding = observed.json(result["endpoint_binding"])
    ep = binding["endpoint"]
    require(
        (ep["start_seed"], ep["arm"], ep["step"]) == key
        and binding["scope"] == "MAIN"
        and binding["manifest_sha256"] == MANIFEST_SHA
        and binding["request_ids_sha256"] == ids_sha
        and binding["checkpoint"] == result["checkpoint"],
        "Foreign endpoint or checkpoint binding",
    )
    original_sources = terminal.original["sources"]
    sources = {
        k: dict(
            path=str(terminal.package / "execution" / (k + ".py")),
            sha256=original_sources["execution/" + k + ".py"],
        )
        for k in (
            "evaluation_runtime",
            "evaluation_loader",
            "training_state",
            "training_transaction",
        )
    }
    sources["authority"] = dict(path=str(EXTERNAL / CONTROL), sha256=m.sha(EXTERNAL / CONTROL))
    runtime_ref = m.ref(path / "endpoint/EVALUATION_RUNTIME.json")
    runtime = observed.json(runtime_ref)
    require(
        binding["execution_sources_sha256"]
        == runtime["execution_sources_sha256"]
        == m.backend.object_sha(sources)
        and runtime["authority_source"] == sources["authority"]
        and runtime["endpoint"] == ep
        and runtime["design_freeze_sha256"] == m.FREEZE_SHA
        and runtime["requests"] == count
        and runtime["request_ids_sha256"] == ids_sha
        and runtime["optimizer_constructed"] is False
        and runtime["scheduler_constructed"] is False
        and runtime["authorization"]["terminal"] == terminal.receipt,
        "Actual evaluation runtime, terminal authority or source binding differs",
    )
    if arm == "START":
        require(binding["checkpoint"] is None, "Unexpected START checkpoint")
    else:
        expected_commit = terminal.audit_spec["trusted_commits"][f"{seed}/{arm}"][step - 1]
        require(
            binding["checkpoint"]["commit"] == expected_commit,
            "Checkpoint differs from terminal audit",
        )
    close = observed.json(result["close"])
    require(
        close["closed"] is True
        and close["scope"] == "MAIN"
        and close["attempted_requests"] == count
        and all(close[k] == v for k, v in ep.items())
        and close["request_ids_sha256"] == ids_sha
        and all(
            type(close[k]) is int and close[k] == 0
            for k in ("optimizer_steps", "scheduler_steps", "parameter_updates")
        ),
        "Invalid close evidence",
    )
    receipt = observed.json(result["receipt"])
    require(
        receipt["endpoint_binding_sha256"] == result["endpoint_binding"]["sha256"]
        and receipt["endpoint_before"] == receipt["endpoint_after"] == ep
        and receipt["responses"] == count
        and observed.ref(receipt["raw"]) == path / "endpoint" / "RAW.jsonl"
        and observed.ref(receipt["accounting"]) == path / "endpoint" / "accounting.jsonl",
        "Invalid raw/receipt binding",
    )
    require(
        list(m.evaluation.core.read_jsonl(path / "endpoint" / "accounting.jsonl"))
        == close["accounting"],
        "CLOSE accounting differs from actual events",
    )
    return dict(
        result=result_ref,
        receipt=result["receipt"],
        binding=result["endpoint_binding"],
        close=result["close"],
        claim=claim_ref,
        job_id=job,
        raw=receipt["raw"],
        accounting=receipt["accounting"],
        runtime=runtime_ref,
        process_identity=result["process_identity"],
    )


def evaluate_endpoint(m, observed, meta, config, freeze, selected):
    receipt = observed.json(meta["receipt"])
    binding = observed.json(meta["binding"])
    result = m.evaluation.analyze_completed_endpoint(
        config,
        freeze,
        ROOT,
        observed.ref(meta["receipt"]),
        expected_receipt_sha256=meta["receipt"]["sha256"],
        endpoint_binding_path=observed.ref(meta["binding"]),
        expected_endpoint_binding_sha256=meta["binding"]["sha256"],
    )
    scores = result["rows"]
    ids = [row["request_id"] for row in selected]
    actual = [row["request_id"] for row in scores]
    exact_membership(ids, actual, "Scored requests")
    require(
        ids == actual
        and result["summary"]["status"] == "COMPLETE_ENDPOINT_CPU_SCORED"
        and result["summary"]["scope"] == "MAIN"
        and result["summary"]["metrics"]["assigned"] == len(ids)
        and result["summary"]["endpoint"] == binding["endpoint"],
        "Independent scoring order or endpoint differs",
    )
    directory = observed.ref(meta["receipt"]).parent
    # Inspect the full slot so an orphan side capture outside endpoint/ also fails.
    suffixes = (".BEGIN.json", ".CAPTURE.json", ".COMPLETE.json")
    for item in directory.parent.rglob("*"):
        if item.name.endswith(suffixes):
            require(
                item.parent == directory, "Orphan capture outside registered endpoint: " + str(item)
            )
    m._capture_inventory(directory, ids)
    compact = []
    for manifest, score, ref in zip(selected, scores, receipt["complete_files"], strict=True):
        require(
            all(type(score.get(k)) is type(v) and score[k] == v for k, v in manifest.items())
            and score["original_complete"] == ref,
            "Scored manifest fields or original capture changed",
        )
        rid = score["request_id"]
        stem = digest(rid.encode())
        require(observed.ref(ref) == directory / (stem + ".COMPLETE.json"), "Relocated COMPLETE")
        capture = observed.path(
            directory / (stem + ".CAPTURE.json"), score["native_capture_sha256"]
        )
        require(str(capture) == score["native_capture_path"], "Relocated native CAPTURE")
        observed.path(directory / (stem + ".BEGIN.json"))
        compact.append(
            dict(
                request_id=rid,
                job_id=meta["job_id"],
                complete_sha256=ref["sha256"],
                native_capture_sha256=score["native_capture_sha256"],
                score_sha256=digest(canonical(score)),
            )
        )
    meta["independent_summary"] = result["summary"]
    return compact


def audit(release_sha):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU isolation required")
    controller, m, release, terminal, planner = release_module(release_sha)
    release_ref = dict(path=str(RELEASE.relative_to(ROOT)), sha256=release_sha)
    observed = m.TA.Observed(ROOT)
    observed.ref(release_ref)
    observed.tree(EXTERNAL)
    for relative, sha256 in release["parallel_sources"].items():
        observed.path(EXTERNAL / relative, sha256)
    m._hold_free()
    freeze_path = observed.path(ROOT / "DESIGN_FREEZE.json", m.FREEZE_SHA)
    freeze = m.read(freeze_path)
    config = observed.ref(freeze["artifacts"]["config"])
    manifest = observed.ref(freeze["artifacts"]["main_manifest"])
    require(
        freeze["artifacts"]["main_manifest"]["sha256"] == MANIFEST_SHA, "Wrong frozen MAIN manifest"
    )
    plan = planner.build_plan(manifest)
    require(plan == observed.json(release["plan"]), "Planned lanes differ from the manifest")
    rows = list(m.evaluation.core.read_jsonl(manifest))
    groups = m.evaluation.validate_manifest(rows, "MAIN")
    slots = m._existing_slots()
    exact_membership([m.endpoint_key(*key) for key in groups], list(slots), "MAIN endpoint slots")
    previous = {}
    jobs = {}
    metadata = {}
    waves = []
    processes = set()
    # Validate every Slurm result and completion before consulting behavioral scores.
    for wave, (arm, step) in enumerate(WAVES):
        controller.wave_healthy(wave)
        directory = controller.wave_directory(wave)
        pair_ref = m.ref(directory / "PAIR.json")
        pair = observed.json(pair_ref)
        controller.validate_pair(pair, release_ref, wave)
        require(
            pair["previous_results"] == previous, "Broken predecessor chain for wave " + str(wave)
        )
        released_ref = m.ref(directory / "RELEASE_RESULT.json")
        released = observed.json(released_ref)
        require(
            released["status"] == "MAIN_PARALLEL_PAIR_RELEASE_CONFIRMED"
            and released["wave"] == wave
            and released["jobs"] == pair["jobs"]
            and released["pair"] == pair_ref,
            "Unverified pair release for wave " + str(wave),
        )
        require(
            observed.ref(released["release_response"]) == directory / "RELEASE_RESPONSE.json"
            and released["release"] == release_ref,
            "Relocated scheduler response or release",
        )
        response = observed.json(released["release_response"])
        ids = [str(pair["jobs"][str(seed)]) for seed in SEEDS]
        require(
            response["returncode"] == 0
            and response["argv"] == ["scontrol", "release", ",".join(ids)],
            "Recorded Slurm release failed or named different jobs",
        )
        waves.append(dict(pair=pair_ref, released=released_ref, jobs=pair["jobs"]))
        for seed in SEEDS:
            key = (seed, arm, step)
            name = m.endpoint_key(*key)
            job = pair["jobs"][str(seed)]
            require(job not in jobs, "Slurm job ID reused between endpoints")
            held_ref = pair["held_refs"][str(seed)]
            require(
                observed.ref(held_ref) == directory / f"HELD_{seed}.json",
                "Relocated held-job identity",
            )
            held = observed.json(held_ref)["fields"]
            post = released["post_release"][str(seed)]["fields"]
            for state in (held, post):
                require(
                    state["JobId"] == str(job)
                    and state["JobName"] == f"LH-SATDOSE-MAIN-{seed}-{arm}-s{step:02d}"
                    and state["Account"] == "rose"
                    and state["QOS"] == QOS
                    and state["Partition"] == "cluster02",
                    "Foreign scheduler evidence",
                )
            require(
                held["JobState"] == "PENDING"
                and held["Reason"] == "JobHeldUser"
                and held["Priority"] == "0"
                and post.get("Reason") not in ("JobHeldUser", "JobHeldAdmin"),
                "Invalid held/released pair",
            )
            jobs[job] = accounting(job, seed, arm, step)
            ids = [row["request_id"] for row in groups[key]]
            meta = endpoint_metadata(
                m,
                terminal,
                observed,
                key,
                slots[name],
                job,
                release_ref,
                pair_ref,
                released_ref,
                len(ids),
                planner.object_sha(ids),
            )
            identity = canonical(meta["process_identity"])
            require(identity not in processes, "Evaluation process reused between endpoints")
            processes.add(identity)
            observed.tree(slots[name])
            previous[name] = meta["result"]
            metadata[name] = meta
    records = []
    for key, selected in groups.items():
        records.extend(
            evaluate_endpoint(
                m, observed, metadata[m.endpoint_key(*key)], config, freeze_path, selected
            )
        )
    original_ids = [row["request_id"] for row in rows]
    exact_membership(original_ids, [row["request_id"] for row in records], "Combined audited rows")
    by_id = {row["request_id"]: row for row in records}
    ordered = [by_id[rid] for rid in original_ids]
    require(digest(canonical(original_ids)) == FROZEN_IDS_SHA, "Original manifest order changed")
    terminal.stable()
    m._hold_free()
    require(m._existing_slots() == slots, "MAIN slots changed during independent audit")
    for wave in range(len(WAVES)):
        controller.wave_healthy(wave)
    observed.stable()
    return dict(
        status=STATUS,
        verified_at_utc=datetime.now(timezone.utc).isoformat(),
        release=release_ref,
        auditor_sha256=digest(ordinary(Path(__file__)).read_bytes()),
        terminal=terminal.receipt,
        manifest=freeze["artifacts"]["main_manifest"],
        plan=plan,
        scheduler=jobs,
        waves=waves,
        endpoints=metadata,
        rows=ordered,
        authority="READ_ONLY_AUDIT_NO_SUCCESSOR",
    )


def validate_bundle(bundle, release_sha, auditor_sha):
    require(
        bundle["status"] == STATUS
        and bundle["auditor_sha256"] == auditor_sha
        and bundle["release"] == dict(path=str(RELEASE.relative_to(ROOT)), sha256=release_sha)
        and bundle["manifest"]["sha256"] == MANIFEST_SHA
        and bundle["authority"] == "READ_ONLY_AUDIT_NO_SUCCESSOR",
        "Foreign audit stream",
    )
    rows = bundle["rows"]
    ids = [row["request_id"] for row in rows]
    require(
        len(ids) == len(set(ids)) == 13760 and digest(canonical(ids)) == FROZEN_IDS_SHA,
        "Local downloaded request IDs differ from all 13,760 original ordered IDs",
    )
    endpoints = bundle["endpoints"]
    require(
        set(endpoints) == {f"{s}/{a}/{t}" for s in SEEDS for a, t in WAVES},
        "Missing/extra endpoint",
    )
    waves = bundle["waves"]
    require(len(waves) == len(WAVES), "Missing or extra paired scheduler release")
    for i, ((arm, step), wave) in enumerate(zip(WAVES, waves, strict=True)):
        expected = {str(s): endpoints[f"{s}/{arm}/{step}"]["job_id"] for s in SEEDS}
        require(wave["jobs"] == expected, "Wave assigns different jobs to endpoint lanes")
        for field, name in (("pair", "PAIR.json"), ("released", "RELEASE_RESULT.json")):
            ref = wave[field]
            path = RELEASE.parent.relative_to(ROOT) / f"wave-{i:02d}" / name
            require(
                ref["path"] == str(path) and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]),
                "Wave authorization references differ",
            )
    ids_by_job = {}
    for key, endpoint in endpoints.items():
        job = endpoint["job_id"]
        require(
            type(job) is int and job > 0 and job not in ids_by_job, "Foreign or duplicate job ID"
        )
        ids_by_job[job] = []
        s, a, t = key.split("/")
        rows_of_job = bundle["scheduler"][str(job)]["rows"]
        scheduler_rows(
            "\n".join("|".join(r) for r in rows_of_job),
            job,
            f"LH-SATDOSE-MAIN-{s}-{a}-s{int(t):02d}",
        )
    require(
        set(bundle["scheduler"]) == {str(j) for j in ids_by_job},
        "Unexpected scheduler IDs in download",
    )
    for row in rows:
        rid, job = row["request_id"], row["job_id"]
        fields = rid.split("/")
        key = f"{fields[2]}/{fields[3]}/{int(fields[4][1:])}"
        require(
            endpoints[key]["job_id"] == job
            and all(
                re.fullmatch(r"[0-9a-f]{64}", row[k])
                for k in ("complete_sha256", "native_capture_sha256", "score_sha256")
            ),
            "Row on wrong lane/job or missing capture hash",
        )
        ids_by_job[job].append(rid)
    for seed in SEEDS:
        lane_count = 0
        for arm, step in WAVES:
            endpoint = endpoints[f"{seed}/{arm}/{step}"]
            actual = ids_by_job[endpoint["job_id"]]
            expected = 1952 if step in (0, 24) else 256
            require(
                len(actual) == expected
                and endpoint["independent_summary"]["metrics"]["assigned"] == expected
                and digest(canonical(actual))
                == endpoint["independent_summary"]["evidence"]["request_ids_sha256"],
                "Endpoint count/order differs in downloaded index",
            )
            lane_count += len(actual)
        require(lane_count == 6880, "Lane count differs")


def write_new(path, data):
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def verify_download(output, release_sha, source_sha):
    require(output.resolve(strict=True) == output, "Noncanonical download directory")
    original_intent = load_json(ordinary(output / "INTENT.json").read_bytes())
    require(
        original_intent == dict(release_sha256=release_sha, auditor_sha256=source_sha),
        "Foreign audit attempt",
    )
    bundle_file = ordinary(output / "BUNDLE.json.gz")
    require(
        bundle_file.stat().st_size <= MAX_BUNDLE_BYTES, "Compressed audit stream exceeds budget"
    )
    with gzip.open(bundle_file, "rb") as stream:
        raw = stream.read(MAX_BUNDLE_BYTES + 1)
    require(len(raw) <= MAX_BUNDLE_BYTES, "Uncompressed audit stream exceeds budget")
    # This completion marker is sent only after the remote gzip footer and flush.
    marker = b"SATDOSE_PARALLEL_AUDIT_SHA256="
    observed = [
        line[len(marker) :].strip()
        for line in ordinary(output / "SSH_STDERR.log").read_bytes().splitlines()
        if line.startswith(marker)
    ]
    require(observed == [digest(raw).encode()], "Missing or mismatched remote audit content digest")
    bundle = load_json(raw)
    validate_bundle(bundle, release_sha, source_sha)
    final = dict(
        status=STATUS,
        responses=13760,
        lanes={"11031": 6880, "11032": 6880},
        endpoints=14,
        jobs=14,
        waves=7,
        release=bundle["release"],
        terminal=bundle["terminal"],
        bundle=reference(output, bundle_file),
        compressed_bytes=bundle_file.stat().st_size,
        content_sha256=digest(raw),
        auditor_sha256=source_sha,
        authority="READ_ONLY_AUDIT_NO_SUCCESSOR",
    )
    receipt = output / "AUDIT_RECEIPT.json"
    if receipt.exists():
        require(load_json(ordinary(receipt).read_bytes()) == final, "Prior audit receipt differs")
    else:
        write_new(receipt, canonical(final) + b"\n")
    return final


def download(output, release_sha):
    require(
        output.is_absolute() and output.parent.resolve(strict=True) == output.parent,
        "New canonical output directory required",
    )
    output.mkdir(mode=0o700, exist_ok=False)
    source_sha = digest(ordinary(Path(__file__).resolve()).read_bytes())
    write_new(
        output / "INTENT.json",
        canonical(dict(release_sha256=release_sha, auditor_sha256=source_sha)) + b"\n",
    )
    remote = EXTERNAL / AUDITOR
    # Verify remote source bytes with the independently known local SHA before execution.
    bootstrap = (
        "import hashlib,pathlib,sys; p=pathlib.Path(sys.argv[1]); raw=p.read_bytes(); "
        "assert p.resolve(strict=True)==p and hashlib.sha256(raw).hexdigest()==sys.argv[2]; "
        'sys.argv=[str(p),"--release-sha256",sys.argv[3],"--stream"]; '
        'exec(compile(raw,str(p),"exec"),{"__name__":"__main__","__file__":str(p)})'
    )
    command = [
        "env",
        "CUDA_VISIBLE_DEVICES=",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        REMOTE_PYTHON,
        "-I",
        "-B",
        "-c",
        bootstrap,
        str(remote),
        source_sha,
        release_sha,
    ]
    ssh = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=4",
        "eee-cluster",
        shlex.join(command),
    ]
    with (
        (output / "BUNDLE.json.gz").open("xb") as out,
        (output / "SSH_STDERR.log").open("xb") as err,
    ):
        process = subprocess.run(ssh, stdout=out, stderr=err, check=False)
        out.flush()
        os.fsync(out.fileno())
        err.flush()
        os.fsync(err.fileno())
    require(process.returncode == 0, "SSH audit failed; inspect retained SSH_STDERR.log")
    return verify_download(output, release_sha, source_sha)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release-sha256", required=True)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--ssh-host", choices=["eee-cluster"])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument(
        "--verify-download",
        action="store_true",
        help="Verify the recorded complete local stream without SSH",
    )
    mode.add_argument("--stream", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    try:
        require(
            re.fullmatch(r"[0-9a-f]{64}", args.release_sha256),
            "A trusted release SHA-256 is required",
        )
        if args.stream:
            require(not args.output_dir and not args.ssh_host, "Unexpected stream arguments")
            # Keep diagnostics from the original native CPU scorer out of the binary archive.
            from contextlib import redirect_stdout

            with redirect_stdout(sys.stderr):
                bundle = audit(args.release_sha256)
            content = canonical(bundle)
            with gzip.GzipFile(fileobj=sys.stdout.buffer, mode="wb", mtime=0) as out:
                out.write(content)
            sys.stdout.buffer.flush()
            print("SATDOSE_PARALLEL_AUDIT_SHA256=" + digest(content), file=sys.stderr, flush=True)
            return 0
        if args.execute:
            require(
                args.output_dir is not None and args.ssh_host == "eee-cluster",
                "Supply --ssh-host eee-cluster and a new absolute --output-dir",
            )
            result = download(args.output_dir, args.release_sha256)
        elif args.verify_download:
            require(
                args.output_dir is not None and not args.ssh_host,
                "Provide existing download directory",
            )
            result = verify_download(
                args.output_dir,
                args.release_sha256,
                digest(ordinary(Path(__file__).resolve()).read_bytes()),
            )
        else:
            result = dict(
                status="PLAN_ONLY",
                jobs=14,
                waves=7,
                planned_responses=13760,
                model_calls=0,
                scheduler_mutations=0,
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(dict(status="PARALLEL_MAIN_AUDIT_INCOMPLETE", error=str(exc))),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
