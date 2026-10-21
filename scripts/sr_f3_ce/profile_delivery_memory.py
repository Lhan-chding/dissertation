#!/usr/bin/env python3
"""Profile one delivery builder in a fresh process; never submit or rerun science.

Synthetic mode exercises the complete 768 training / 51 monitor panel shape with
on-demand raw dictionaries. It replaces file reads/hashes and monitor scoring with
explicit fixture equivalents, while using the real work-occurrence summarizer.
Its result is NOT a minimum resource requirement for the complete real delivery.
Real-build mode invokes only the selected builder.build, with its normal guards.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def rss_bytes(raw, platform=None):
    """getrusage reports bytes on macOS and KiB on Linux."""
    return int(raw) if (platform or sys.platform) == "darwin" else int(raw) * 1024


def load_builder(path):
    spec = importlib.util.spec_from_file_location("profiled_delivery_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_fixture(builder, token_length, raw_sample=None):
    from sr_f3_ce import evaluation, runner

    if token_length < 0:
        raise ValueError("token length must be nonnegative")
    payloads = None
    if raw_sample is not None:
        sample = json.loads(Path(raw_sample).read_text())
        if not sample.get("records") or not all(isinstance(row, dict) for row in sample["records"]):
            raise ValueError("raw sample requires nonempty record objects")
        payloads = [json.dumps(row) for row in sample["records"]]
    root = Path("/SYNTHETIC_SR_F3_CE_NOT_EXPERIMENT")
    models = evaluation.endpoint_models()
    training, monitor, counts = [], [], {}
    receipt_sources = {}
    for model in models:
        mid = model["model_id"]
        if model["arm"] != "PARENT":
            for step in range(1, 65):
                sources = []
                for slot in range(16):
                    path = root / "runs" / mid / "rollouts" / f"{step:02d}-{slot:02d}.json"
                    training.append((path, {"run_id": mid}, 8))
                    counts[path] = 8
                    sources.append(path)
                receipt_sources[
                    root / "runs" / mid / "work_occurrence" / f"step-{step:02d}.json"
                ] = (step, sources)
        for step in (0,) if model["arm"] == "PARENT" else (16, 32, 48, 64):
            for slot in range(64):
                path = root / "monitor" / mid / f"step-{step:02d}" / "groups" / f"{slot:02d}.json"
                monitor.append((path, {}, 8))
                counts[path] = 8

    def rows(count, length):
        texts = ('{"answer":1,"work":"reason"}', '{"answer":1}', "invalid", '{"work":""}')
        result = []
        for index in range(count):
            text = texts[index % len(texts)]
            row = (
                json.loads(payloads[index % len(payloads)])
                if payloads
                else dict(
                    tokens=[1024 + token for token in range(length)],
                    logprobs=[-0.0001 * (token + 1) for token in range(length)],
                )
            )
            row.update(
                full_text=text,
                raw_text=text,
                qid="fixture_q",
                finish_reason="eos",
                score={"value_parseable": int('"answer"' in text)},
            )
            result.append(row)
        return result

    summary = runner.work_occurrence(rows(128, 0))

    def fixture_hash(path):
        # Virtual source identity; deliberately not claimed as a real file hash.
        return canonical_digest(["SYNTHETIC_SOURCE", str(path), token_length])

    def fixture_read(path):
        path = Path(path)
        if path in receipt_sources:
            step, sources = receipt_sources[path]
            return dict(
                step=step,
                summary=summary,
                raw_group_sha256=[fixture_hash(source) for source in sources],
            )
        if path not in counts:
            raise ValueError("Unexpected synthetic source: " + str(path))
        return {"records": rows(counts[path], token_length)}

    builder.read = fixture_read
    builder.sha_file = fixture_hash
    builder.slot_revision_kwargs = lambda root: {"revision": "v2"}
    api = SimpleNamespace(
        evaluation=evaluation,
        runner=runner,
        contract=SimpleNamespace(
            score=lambda text, *args, **kwargs: {"value_parseable": int('"answer"' in text)}
        ),
    )
    dataset = SimpleNamespace(tasks={"fixture_q": {"world": {}, "query": {}}})
    return root, api, dataset, training, monitor


def run_child(args):
    started = time.monotonic()
    if args.source_root:
        sys.path.insert(0, str(args.source_root.resolve()))
    builder = load_builder(args.builder)
    if args.mode == "synthetic":
        from sr_f3_ce import contract, evaluation, runner

        result = builder.work_occurrence_trajectory(
            *synthetic_fixture(builder, args.token_length, args.raw_sample)
        )
        evidence = dict(
            status="SYNTHETIC_PROFILE_ONLY",
            fixture_sources={
                name: dict(
                    path=str(Path(module.__file__).resolve()), sha256=sha_file(module.__file__)
                )
                for name, module in (
                    ("runner", runner),
                    ("evaluation", evaluation),
                    ("contract", contract),
                )
            },
            training_panels=len(result["training"]),
            monitor_panels=len(result["monitor"]),
            training_records=sum(p["summary"]["records"] for p in result["training"]),
            monitor_records=sum(p["summary"]["records"] for p in result["monitor"]),
            token_length=args.token_length if args.raw_sample is None else None,
            raw_sample_sha256=sha_file(args.raw_sample) if args.raw_sample else None,
            limitation="Synthetic IO/hash/scoring; not full delivery resource acceptance",
        )
    else:
        revision_kwargs = (
            {"delivery_revision_manifest": args.delivery_revision_manifest}
            if args.delivery_revision_manifest is not None
            else {}
        )
        result = builder.build(
            args.root,
            args.output,
            helper_manifest=args.helper_manifest,
            extension_manifest=args.extension_manifest,
            operation_identity=args.operation_identity,
            **revision_kwargs,
        )
        evidence = dict(status="REAL_BUILD_RETURNED", output=str(args.output))
    evidence.update(
        builder=str(args.builder.resolve()),
        builder_sha256=sha_file(args.builder),
        result_sha256=canonical_digest(result),
        elapsed_seconds=time.monotonic() - started,
        peak_rss_bytes=rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        platform=sys.platform,
        python=sys.version,
        pid=os.getpid(),
        current_rss_bytes=process_rss_bytes(os.getpid()),
        exitcode=0,
        science_execution_requested=False,
        profile_source_sha256=sha_file(Path(__file__)),
    )
    print(json.dumps(evidence, sort_keys=True))
    return evidence


def process_rss_bytes(pid):
    """Read one process RSS only; never scan unrelated processes."""
    try:
        if sys.platform.startswith("linux"):
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
            return None
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=False
        )
        return int(result.stdout.strip()) * 1024 if result.returncode == 0 else None
    except (OSError, ValueError):
        return None


def bounded_log(stream, tail_chars=65536):
    """Hash all decoded diagnostic text while retaining only a bounded tail."""
    stream.seek(0)
    digest, tail = hashlib.sha256(), ""
    for chunk in iter(lambda: stream.read(65536), ""):
        digest.update(chunk.encode("utf-8"))
        tail = (tail + chunk)[-tail_chars:]
    return tail, digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--mode", choices=("synthetic", "real-build"), default="synthetic")
    parser.add_argument("--token-length", type=int, default=64)
    parser.add_argument(
        "--raw-sample", type=Path, help="One existing group for synthetic record payload shape only"
    )
    parser.add_argument(
        "--source-root", type=Path, help="Source directory for fixture sr_f3_ce modules"
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--helper-manifest", type=Path)
    parser.add_argument("--extension-manifest", type=Path)
    parser.add_argument("--operation-identity", type=Path)
    parser.add_argument("--delivery-revision-manifest", type=Path)
    parser.add_argument("--receipt", type=Path, help="New profile JSON file; refuses to overwrite")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.token_length < 0:
        parser.error("--token-length must be nonnegative")
    if not args.builder.is_file():
        parser.error("--builder must exist")
    if args.mode == "real-build" and (not args.root or not args.output):
        parser.error("real-build requires --root and --output")
    if (
        args.mode == "real-build"
        and sys.platform.startswith("linux")
        and not os.environ.get("SLURM_JOB_ID")
    ):
        parser.error(
            "real-build on Linux requires an allocated Slurm job; login-node runs forbidden"
        )
    if args.receipt and args.receipt.exists():
        parser.error("profile receipt already exists")
    if args.child:
        return run_child(args)
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        *(sys.argv[1:] if argv is None else argv),
        "--child",
    ]
    identity_at_launch = dict(
        builder=str(args.builder.resolve()),
        builder_sha256=sha_file(args.builder),
        profile_source_sha256=sha_file(Path(__file__)),
    )
    input_paths = {}
    for name in (
        "raw_sample",
        "helper_manifest",
        "extension_manifest",
        "operation_identity",
        "delivery_revision_manifest",
    ):
        path = getattr(args, name)
        if path is not None:
            input_paths[name] = dict(
                path=str(path), sha256=sha_file(path) if path.is_file() else None
            )
    started_at = datetime.now(timezone.utc).isoformat()
    launched = time.monotonic()
    observed_peak, last_rss = None, None
    with tempfile.TemporaryFile(mode="w+") as stdout, tempfile.TemporaryFile(mode="w+") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        while process.poll() is None:
            observed = process_rss_bytes(process.pid)
            if observed is not None:
                last_rss = observed
                observed_peak = observed if observed_peak is None else max(observed_peak, observed)
            time.sleep(0.2)
        output, stdout_sha256 = bounded_log(stdout)
        diagnostics, stderr_sha256 = bounded_log(stderr)
    if diagnostics:
        sys.stderr.write(diagnostics)
    if process.returncode:
        failure = dict(
            status="PROFILE_FAILED",
            exitcode=process.returncode,
            **identity_at_launch,
            mode=args.mode,
            command=command,
            started_at_utc=started_at,
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            platform=sys.platform,
            python=sys.version,
            token_length=args.token_length,
            inputs=input_paths,
            stdout_sha256_utf8=stdout_sha256,
            stderr_sha256_utf8=stderr_sha256,
            stderr_tail=diagnostics[-2048:],
            elapsed_wall_seconds=time.monotonic() - launched,
            peak_rss_bytes=None,
            current_rss_bytes=None,
            measurement_scope=(
                "Parent sampled child RSS only; no child peak receipt on failure; "
                "samples are a lower bound and may miss the true peak (including OOM); "
                "null means unavailable"
            ),
            science_execution_requested=False,
            observed_peak_rss_bytes=observed_peak,
            last_observed_rss_bytes=last_rss,
        )
        if args.receipt:
            with args.receipt.open("x") as stream:
                json.dump(failure, stream, sort_keys=True)
        print(json.dumps(failure))
        raise SystemExit(process.returncode if process.returncode > 0 else 1)
    result = json.loads(output.strip().splitlines()[-1])
    result.update(
        observed_peak_rss_bytes=observed_peak,
        last_observed_rss_bytes=last_rss,
        measurement_scope="builder child only; not Slurm/cgroup total or resource minimum",
    )
    if args.receipt:
        with args.receipt.open("x") as stream:
            json.dump(result, stream, sort_keys=True)
            stream.write("\n")
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
