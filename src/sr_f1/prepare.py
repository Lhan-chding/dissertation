"""Materialize the authenticated SR-F1 inputs, without loading model weights."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from mm_core.execution import atomic_json, object_hash, read_json, utc_now

from .amendment import (
    effective_plan,
    protocol_amendment,
    register_amendment,
    validate_plan,
    verify_protocol_route,
)
from .contract import PACKAGE, PLAN_ID, file_hash, load_config, reference_module
from .data import bounded_path, load_inputs, load_schedule, load_tasks

FONT_HASH = "57f73e11f51999432bf7ab22ce55b6f945d5eca1bf824404cfa9ec2e3718c84e"
BOLD_HASH = "a4c5bc453ca281d90ea079e596da7ae0dfeb5777497c29ec254e76d97ff6f890"
PACKAGE_MANIFEST_HASH = "832e6cb387af2b0525b55cd076c5a0eb7ad87dd697c42e67b81f194ea580fa65"
SOURCE_ROOT = Path(__file__).resolve().parents[2]


def verify_package():
    if file_hash(PACKAGE / "PACKAGE_SHA256.json") != PACKAGE_MANIFEST_HASH:
        raise PermissionError("Uploaded package hash inventory itself changed")
    declared = read_json(PACKAGE / "PACKAGE_SHA256.json")
    if not declared:
        raise PermissionError("Empty uploaded package inventory")
    for name, expected in declared.items():
        path = bounded_path(PACKAGE, name)
        if not path.is_file() or file_hash(path) != expected:
            raise PermissionError("Uploaded package changed: " + name)
    return {
        "status": "PASS",
        "checked_files": len(declared),
        "package_manifest_sha256": file_hash(PACKAGE / "PACKAGE_SHA256.json"),
    }


def source_identity():
    files = {}
    for folder in ("src/sr_f1", "scripts/sr_f1", "src/mm_core", "src/mm_dev"):
        for path in sorted((SOURCE_ROOT / folder).rglob("*")):
            if not path.is_file() or path.suffix not in (".py", ".sh", ".sbatch"):
                continue
            if path.is_symlink():
                raise PermissionError("Source symlinks are forbidden")
            files[str(path.relative_to(SOURCE_ROOT))] = file_hash(path)
    for name in ("pyproject.toml", "uv.lock"):
        path = SOURCE_ROOT / name
        if path.is_file():
            files[name] = file_hash(path)
    # Protocol amendments are executable provenance, not merely commentary.
    for path in sorted((SOURCE_ROOT / "docs/sr_f1/amendments").rglob("*")):
        if path.is_file() and path.suffix in (".md", ".json"):
            if path.is_symlink():
                raise PermissionError("Amendment source symlinks are forbidden")
            files[str(path.relative_to(SOURCE_ROOT))] = file_hash(path)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=SOURCE_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        deployment = SOURCE_ROOT / "SOURCE_DEPLOYMENT.json"
        receipt = read_json(deployment) if deployment.exists() else {}
        commit, dirty = receipt.get("source_commit"), receipt.get("source_dirty_files", [])
        if receipt.get("source_tree_sha256") not in (None, object_hash(files)):
            raise PermissionError(
                "Deployment source-tree receipt differs from actual code"
            ) from None
        if receipt.get("source_file_hashes") not in (None, files):
            raise PermissionError(
                "Deployment source-file receipt differs from actual code"
            ) from None
    return dict(
        source_commit=commit,
        source_tree_sha256=object_hash(files),
        source_file_hashes=files,
        source_dirty_files=dirty,
    )


def _copy_registered(root):
    """Never overwrite a conflicting run file or modify the uploaded package."""
    for folder in ("manifests", "config"):
        for source in sorted((PACKAGE / folder).rglob("*")):
            if not source.is_file():
                continue
            target = bounded_path(root, str(source.relative_to(PACKAGE)))
            if target.exists():
                if file_hash(source) != file_hash(target):
                    raise PermissionError("Existing registered input differs: " + str(target))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)


def _run_reference_checks(root):
    """Run reference validators only against the isolated copy."""
    (root / "validation").mkdir(exist_ok=True)
    result = subprocess.run(
        [sys.executable, str(PACKAGE / "reference/validate_manifest.py")],
        cwd=root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    (root / "validation/MANIFEST_VALIDATION_LOG.txt").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("Authenticated manifest validator failed; see validation log")
    with tempfile.TemporaryDirectory(prefix="sr-f1-rebuild-") as temporary:
        destination = Path(temporary) / "manifests"
        result = subprocess.run(
            [
                sys.executable,
                str(PACKAGE / "reference/build_manifest.py"),
                "--out",
                str(destination),
            ],
            capture_output=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError("Manifest rebuild failed: " + result.stderr[-2000:])
        identical = {
            str(path.relative_to(PACKAGE / "manifests")): file_hash(path)
            == file_hash(destination / path.relative_to(PACKAGE / "manifests"))
            for path in sorted((PACKAGE / "manifests").rglob("*"))
            if path.is_file() and path.name != "DIAGNOSTIC_INPUTS_AUDIT_ONLY.jsonl"
        }
        if not all(identical.values()):
            raise PermissionError("Frozen manifests are not reproducible")
    for seed in load_config()["training"]["seeds"]:
        load_schedule(root, seed)
    report = dict(
        status="PASS",
        byte_identical=identical,
        independent_validation=read_json(root / "validation/MANIFEST_VALIDATION.json"),
    )
    atomic_json(root / "validation/REPRODUCIBILITY.json", report)
    return report


def render_inputs(root, font_path, bold_path):
    from PIL import Image, ImageFont
    from PIL import __version__ as pillow_version

    fonts = [
        (Path(font_path).resolve(strict=True), FONT_HASH),
        (Path(bold_path).resolve(strict=True), BOLD_HASH),
    ]
    for path, expected in fonts:
        if file_hash(path) != expected:
            raise PermissionError("Renderer font differs from uploaded examples: " + str(path))
    renderer = reference_module("render_charts")
    renderer.FONT_PATH, renderer.FONT_BOLD = map(lambda entry: str(entry[0]), fonts)
    tasks = load_tasks(root)
    unique = {}
    for task in tasks.values():
        previous = unique.setdefault(task["image_file"], task)
        if any(previous[key] != task[key] for key in ("world", "chart", "style")):
            raise PermissionError("One image route maps to distinct rendered worlds")
    if len(unique) != 2890:
        raise PermissionError("Expected exactly 2890 distinct registered chart images")
    rows = []
    for relative, task in sorted(unique.items()):
        path = bounded_path(root, relative)
        # For a partial CPU preparation, verify existing images by fresh rendering.
        with tempfile.TemporaryDirectory(prefix="sr-f1-render-") as temporary:
            temporary_path = Path(temporary) / "image.png"
            record = renderer.render(task, temporary_path)
            if path.exists() and file_hash(path) != record["sha256"]:
                raise PermissionError("Existing rendered image differs: " + relative)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(temporary_path, path)
        with Image.open(path) as image:
            if image.size != (1024, 768) or image.info:
                raise PermissionError("Unexpected PNG canvas or metadata: " + relative)
        record.update(
            file=relative,
            font_bold_sha256=BOLD_HASH,
            pool=task["pool"],
            family=task["family"],
            chart=task["chart"],
            style=task["style"],
            root_id=task["root_id"],
            series_count=len(task["world"]["series"]),
        )
        rows.append(record)
    blank = root / "images/diagnostic_blank.png"
    with tempfile.TemporaryDirectory(prefix="sr-f1-blank-") as temporary:
        path = Path(temporary) / "blank.png"
        Image.new("RGB", (1024, 768), "white").save(path)
        if blank.exists() and file_hash(blank) != file_hash(path):
            raise PermissionError("Existing diagnostic blank image changed")
        if not blank.exists():
            shutil.copyfile(path, blank)
    rows.append(
        dict(
            file="images/diagnostic_blank.png",
            sha256=file_hash(blank),
            width=1024,
            height=768,
            type="diagnostic_blank_not_scientific_world",
        )
    )
    example_checks = []
    for example in read_json(PACKAGE / "examples/RENDER_RECEIPT.json"):
        relative = example["file"].removeprefix("examples/")
        with (
            Image.open(PACKAGE / example["file"]) as original,
            Image.open(root / relative) as rendered,
        ):
            same = original.convert("RGB").tobytes() == rendered.convert("RGB").tobytes()
        example_checks.append(dict(file=relative, pixel_identical=same))
    receipt = dict(
        status="RENDERED",
        chart_count=2890,
        image_count=2891,
        images=rows,
        example_pixel_matches=sum(r["pixel_identical"] for r in example_checks),
        reference_example_pixel_comparison=example_checks,
        reference_example_comparison_is_acceptance_gate=False,
        original_package_bold_and_freetype_identity="NOT_RECORDED",
        renderer_environment={
            "Pillow": pillow_version,
            "FreeType": ImageFont.core.freetype2_version,
            "renderer_sha256": file_hash(PACKAGE / "reference/render_charts.py"),
        },
        fonts={
            "normal": dict(path=str(fonts[0][0]), sha256=FONT_HASH),
            "bold": dict(path=str(fonts[1][0]), sha256=BOLD_HASH),
        },
        font_bytes_must_not_be_distributed=True,
    )
    atomic_json(root / "RENDER_RECEIPT.json", receipt)
    return receipt


def model_identity(model_path, runtime, plan=None):
    """Fresh actual file hashes; no inherited run identity or inference outputs."""
    from mm_core.vl_runtime import model_file_path

    snapshot = Path(model_path).resolve(strict=True)
    plan = validate_plan(load_config() if plan is None else plan)
    amendment = protocol_amendment(plan)
    if runtime.identity.get("protocol_amendment") != amendment:
        raise PermissionError("Native processor identity differs from the registered protocol")
    if snapshot.name != plan["model"]["revision"]:
        raise PermissionError("Model snapshot directory must identify the frozen revision")
    files = []
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(snapshot))
        actual = model_file_path(snapshot, relative)
        files.append(dict(name=relative, bytes=actual.stat().st_size, sha256=file_hash(actual)))
    weights = {
        r["name"]: r["sha256"]
        for r in files
        if r["name"].endswith((".safetensors", "index.json")) or r["name"] == "config.json"
    }
    if not any(name.endswith(".safetensors") for name in weights):
        raise PermissionError("Actual model weights are missing")
    weight_hash = object_hash(weights)
    if weight_hash != plan["model"]["prior_composite_weight_hash"]:
        raise PermissionError("Fresh composite model hash differs from fixed 9B snapshot")
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "numpy", "Pillow", "safetensors"):
        packages[name] = importlib.metadata.version(name)
    return dict(
        runtime.identity,
        plan_id=PLAN_ID,
        status="ACTUAL_CPU_VERIFIED",
        checked_at=utc_now(),
        model_id=plan["model"]["id"],
        model_revision=plan["model"]["revision"],
        revision=plan["model"]["revision"],
        model_path=str(snapshot),
        model_files=files,
        model_weights_hash=weight_hash,
        packages=packages,
        python=platform.python_version(),
        model_weights_loaded=False,
        model_calls=0,
        cuda_context_initialized=False,
    )


def label_metrics(fonts):
    """Measure every visible label's ink height in its actual renderer font."""
    from PIL import ImageFont

    labels = [(fonts["normal"]["path"], 18, str(value)) for value in range(0, 101, 10)]
    labels += [
        (fonts["normal"]["path"], 18, text)
        for text in ("Unit: count", "Category order: January, February, March, April")
    ]
    labels += [
        (fonts["normal"]["path"], 22, text)
        for text in ("Alpha", "Beta", "Gamma", "January", "February", "March", "April")
    ]
    labels += [(fonts["bold"]["path"], 27, "Quarterly quantities")]
    records = []
    for path, size, text in labels:
        box = ImageFont.truetype(path, size).getmask(text).getbbox()
        if box is None:
            raise PermissionError("Visible label has no ink")
        records.append(dict(text=text, font_size=size, height_pixels=box[3] - box[1]))
    minimum = min(r["height_pixels"] for r in records)
    if minimum < 9:
        raise PermissionError("Major label ink height is below nine pixels")
    return dict(minimum_label_height_pixels=minimum, labels=records)


def isolation_tests(runtime, row, root):
    """Run both negative sidecar and positive text/pixel input-tensor controls."""
    base = runtime.prepare(row, root)
    polluted = copy.deepcopy(row)
    polluted.update(
        answer="DO_NOT_PASS_TO_MODEL",
        required_evidence=[{"value": -987654}],
        root_id="DO_NOT_PASS_TO_MODEL",
        root_label="DO_NOT_PASS_TO_MODEL",
        qid="MUTATED_ID",
    )
    negative = runtime.prepare(polluted, root)
    question = dict(row, text=row["text"] + " Return only the requested JSON object.")
    positive_text = runtime.prepare(question, root)
    positive_image = runtime.prepare(dict(row, image_file="images/diagnostic_blank.png"), root)

    def fingerprint(prepared):
        return prepared["routing"]["input_tensor_hash"]

    checks = dict(
        sidecar_tensor_invariant=fingerprint(base) == fingerprint(negative),
        question_tensor_changes=fingerprint(base) != fingerprint(positive_text),
        image_tensor_changes=fingerprint(base) != fingerprint(positive_image),
        image_pixels_change=base["routing"]["processed_pixel_sha256"]
        != positive_image["routing"]["processed_pixel_sha256"],
        question_pixels_invariant=base["routing"]["processed_pixel_sha256"]
        == positive_text["routing"]["processed_pixel_sha256"],
        qid_not_in_prompt=row["qid"] not in base["chat_text"],
        path_not_in_prompt=row["image_file"] not in base["chat_text"],
    )
    if not all(checks.values()):
        raise PermissionError("Model input isolation control failed: " + str(checks))
    return dict(
        status="PASS",
        checks=checks,
        hashes={
            "base": fingerprint(base),
            "sidecar": fingerprint(negative),
            "question": fingerprint(positive_text),
            "blank_image": fingerprint(positive_image),
        },
    )


def processor_preflight(root, runtime, rendered, plan=None):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    from mm_dev.data import reconstruct_rgb

    plan = validate_plan(load_config() if plan is None else plan)
    if torch.cuda.is_initialized():
        raise PermissionError("CPU preparation must not initialize a CUDA context")
    torch.set_num_threads(2)
    labels = label_metrics(rendered["fonts"])
    inputs = load_inputs(root)
    processed, routes, qa = {}, [], []
    qa_keys = set()
    image_rows = {r["file"]: r for r in rendered["images"]}
    for index, row in enumerate(inputs.values()):
        prepared = runtime.prepare(row, root)
        routing = prepared["routing"]
        verify_protocol_route(routing, plan)
        relative = row["image_file"]
        if (
            routing["source_image_sha256"] != image_rows[relative]["sha256"]
            or routing["processed_size"] != [1024, 768]
            or routing["pixels_per_count"] < 3
        ):
            raise PermissionError("Native processor image identity/readability failed")
        if relative not in processed:
            rgb = reconstruct_rgb(
                prepared["inputs"]["pixel_values"],
                routing["image_grid_thw"],
                runtime.processor.image_processor,
            )
            with Image.open(root / relative) as original:
                error = int(
                    np.abs(
                        np.asarray(rgb, dtype=np.int16)
                        - np.asarray(original.convert("RGB"), dtype=np.int16)
                    ).max()
                )
            if error > 1 or rgb.size != (1024, 768):
                raise PermissionError(
                    "Actual native processor changed chart pixels or clipped labels"
                )
            processed[relative] = dict(
                file=relative,
                processed_size=list(rgb.size),
                processed_pixel_sha256=routing["processed_pixel_sha256"],
                processed_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                max_inverse_rgb_error=error,
                image_token_count=routing["image_token_count"],
                image_grid_thw=routing["image_grid_thw"],
                pixels_per_count=routing["pixels_per_count"],
                minimum_label_height_pixels=labels["minimum_label_height_pixels"],
            )
            r = image_rows[relative]
            key = (r["pool"], r["family"], r["chart"], r["style"], f"{r['series_count']}series")
            if key not in qa_keys:
                qa_keys.add(key)
                path = "qa/processed/" + relative.removeprefix("images/")
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                rgb.save(root / path)
                qa.append(
                    dict(path=path, sha256=file_hash(root / path), original=relative, key=list(key))
                )
        elif processed[relative]["processed_pixel_sha256"] != routing["processed_pixel_sha256"]:
            raise PermissionError("Processor pixels depend on question or sidecar")
        routes.append(dict(qid=row["qid"], image_file=relative, **routing))
        if (index + 1) % 200 == 0:
            print(
                json.dumps(dict(stage="CPU_PROCESSOR", questions=index + 1, images=len(processed))),
                flush=True,
            )
    blank = runtime.prepare(
        dict(next(iter(inputs.values())), image_file="images/diagnostic_blank.png"), root
    )
    if blank["routing"]["processed_size"] != [1024, 768]:
        raise PermissionError("Diagnostic blank lost its native visual channel")
    verify_protocol_route(blank["routing"], plan)
    controls = isolation_tests(runtime, next(iter(inputs.values())), root)
    # Deterministic contact sheets cover every pool/family/chart/style combination.
    sheets = []
    for start in range(0, len(qa), 12):
        sheet = Image.new("RGB", (1024, 840), "white")
        draw = ImageDraw.Draw(sheet)
        for position, record in enumerate(qa[start : start + 12]):
            x, y = (position % 4) * 256, (position // 4) * 280
            with Image.open(root / record["path"]) as rgb:
                sheet.paste(rgb.resize((256, 192)), (x, y + 36))
            draw.text((x + 3, y + 3), " / ".join(record["key"][:2]), fill="black")
            draw.text((x + 3, y + 17), " / ".join(record["key"][2:]), fill="black")
        relative = f"qa/contact_{start // 12:03d}.png"
        sheet.save(root / relative)
        sheets.append(dict(path=relative, sha256=file_hash(root / relative)))
    for name, values in (
        ("PROCESSOR_INPUT_ROUTES.jsonl", routes),
        ("PROCESSED_IMAGES.jsonl", list(processed.values())),
    ):
        with (root / name).open("w") as handle:
            for value in values:
                handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    report = dict(
        status="PASS",
        processor_status="ACTUAL_CPU_PROCESSOR_VERIFIED",
        question_count=len(routes),
        chart_image_count=len(processed),
        blank_image_verified=True,
        blank_image_routing=blank["routing"],
        readability=labels,
        no_crop_pixel_reconstruction_verified=True,
        isolation=controls,
        qa_samples=qa,
        qa_contact_sheets=sheets,
        processor_identity=runtime.identity,
        model_calls=0,
        model_weights_loaded=False,
        cuda_context_initialized=False,
    )
    if len(routes) != 4800 or len(processed) != 2890 or torch.cuda.is_initialized():
        raise PermissionError("Incomplete CPU native processor preflight")
    atomic_json(root / "PROCESSOR_PREFLIGHT.json", report)
    return report


def prepare_run(
    root,
    *,
    font_path,
    bold_path,
    operator,
    model_path=None,
    local_only=False,
    defer_freeze=False,
    plan=None,
):
    from .freeze import freeze_execution, verify_execution

    root = Path(root).resolve()
    if root == PACKAGE or root.is_relative_to(PACKAGE):
        raise PermissionError("Run root must not be inside the uploaded package")
    root.mkdir(parents=True, exist_ok=True)
    plan = effective_plan(root) if plan is None else validate_plan(plan)
    frozen = root / "EXECUTION_FREEZE.json"
    if frozen.exists() and read_json(frozen).get("status") == "FROZEN":
        return verify_execution(plan, root)
    package = verify_package()
    _copy_registered(root)
    register_amendment(root, plan)
    reference = _run_reference_checks(root)
    rendered = render_inputs(root, font_path, bold_path)
    manifest = dict(
        plan_id=PLAN_ID,
        prepared_at=utc_now(),
        status="LOCAL_PREPARED",
        package=package,
        reference_validation=reference,
        render_receipt_sha256=file_hash(root / "RENDER_RECEIPT.json"),
        image_count=2891,
        question_count=4800,
        processor_verified=False,
        package_input_hashes={
            str(p.relative_to(PACKAGE)): file_hash(p)
            for folder in ("config", "manifests")
            for p in sorted((PACKAGE / folder).rglob("*"))
            if p.is_file()
        },
        **source_identity(),
    )
    atomic_json(root / "SOURCE_AND_RENDER_MANIFEST.json", manifest)
    if local_only:
        atomic_json(
            frozen,
            dict(
                plan_id=PLAN_ID,
                status="LOCAL_PREPARED",
                prepared_at=utc_now(),
                processor_verified=False,
                model_identity_verified=False,
                model_calls=0,
                gpu_execution_authorized=False,
            ),
        )
        return read_json(frozen)
    if not model_path:
        raise ValueError("Real CPU preparation requires --model-path; otherwise use --local-only")
    from .runtime import SRRuntime

    amendment = protocol_amendment(plan)
    runtime = (
        SRRuntime.processor_only(model_path, protocol_amendment=amendment)
        if amendment is not None
        else SRRuntime.processor_only(model_path)
    )
    identity = model_identity(model_path, runtime, plan=plan)
    atomic_json(root / "MODEL_ENVIRONMENT_IDENTITY.json", identity)
    processor_preflight(root, runtime, rendered, plan=plan)
    manifest.update(
        status="CPU_VERIFIED",
        processor_verified=True,
        processor_preflight_sha256=file_hash(root / "PROCESSOR_PREFLIGHT.json"),
        model_environment_identity_sha256=file_hash(root / "MODEL_ENVIRONMENT_IDENTITY.json"),
        **source_identity(),
    )
    atomic_json(root / "SOURCE_AND_RENDER_MANIFEST.json", manifest)
    if defer_freeze:
        atomic_json(
            frozen,
            dict(
                plan_id=PLAN_ID,
                status="CPU_VERIFIED",
                prepared_at=utc_now(),
                processor_verified=True,
                model_identity_verified=True,
                model_calls=0,
                gpu_execution_authorized=False,
            ),
        )
        return read_json(frozen)
    return freeze_execution(root, operator=operator, plan=plan)
