"""Frozen author ChartQAPro subset and conservative, gold-blind type auditing.

Selection reads only native category, question count, and image/question identity.
Gold is loaded only after the selected identity manifest has been published.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import re
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

from sr_f12.evaluation import durable_json

from .contract import normalize_typed
from .evaluation import EXPERIMENT_ID, object_hash, sha_file

DATASET_ID = "ahmed-masry/ChartQAPro"
DATASET_REVISION = "e27c2874825874d6767d2bbc538ed4f0dc2c64c2"
PARQUET_SHA256 = "feb03a5579e49114b45350b75aed7f674e285f795a425c83e22f400095c91489"
PARQUET_URL = (
    f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}"
    "/data/test-00000-of-00001.parquet"
)
AUTHOR_REVISION = "f0c861b695adce63ffc76e247a49919d2dcaed8b"
SCORER_SHA256 = "fe21d33f076a394765935b6231c2b918c321d4d197d618374ab2ab6b2cc96a71"
SCORER_URL = (
    f"https://raw.githubusercontent.com/vis-nlp/ChartQAPro/{AUTHOR_REVISION}"
    "/evaluate_predictions.py"
)
VLM_EVAL_REVISION = "94bae05b7e24c09981072ab0598e8f25466f8f48"
PROMPT_SOURCE_SHA256 = "9bfa75feb8c70341f788a7d85095df9a18cca08c04a7ab793d802b66afd2db83"
ANLS_WHEEL_SHA256 = "f6273fc31188784cd3f6ac083412f70be095ab531847580a88245ee836abe811"
AUTHOR_LICENSE_SHA256 = "5d4f41bc5c9d0df5553524f50d86adffd0d929d733a0663200e0306bb48e0394"
VLM_EVAL_LICENSE_SHA256 = "5f0538b314b13e062d1fed3991ec1868cc71c49d0838cd7bc6ef563be06a5ad3"
ANLS_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/16/6f/"
    "bd3b85c757a57c38955ae0c705357aac1ec282f5281cff7679d00620440c/"
    "anls-0.0.2-py3-none-any.whl"
)
TYPE_KINDS = ("YEAR", "COUNT_INT", "PERCENT_POINT", "RATIO_FRACTION", "GENERAL_NUMERIC")
REVIEW_METHODS = ("independent_visual_annotation", "independent_contract_rule_check")


def _write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise PermissionError(f"Immutable external artifact differs: {path}")
        return
    with path.open("xb") as stream:
        stream.write(data)


def acquire_chartqapro(directory):
    """Download only pinned public sources; failures are explicit external-only state."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sources = {
        "test-00000-of-00001.parquet": (PARQUET_URL, PARQUET_SHA256),
        "evaluate_predictions.py": (SCORER_URL, SCORER_SHA256),
        "anls-0.0.2-py3-none-any.whl": (ANLS_WHEEL_URL, ANLS_WHEEL_SHA256),
        "vlmeval_chartqapro_utils.py": (
            f"https://raw.githubusercontent.com/open-compass/VLMEvalKit/{VLM_EVAL_REVISION}/vlmeval/dataset/utils/chartqapro.py",
            PROMPT_SOURCE_SHA256,
        ),
        "AUTHOR_LICENSE": (
            f"https://raw.githubusercontent.com/vis-nlp/ChartQAPro/{AUTHOR_REVISION}/LICENSE",
            AUTHOR_LICENSE_SHA256,
        ),
        "VLM_EVAL_LICENSE": (
            f"https://raw.githubusercontent.com/open-compass/VLMEvalKit/{VLM_EVAL_REVISION}/LICENSE",
            VLM_EVAL_LICENSE_SHA256,
        ),
    }
    assets = {}
    try:
        for name, (url, expected) in sources.items():
            path = directory / name
            if not path.exists():
                partial = path.with_suffix(path.suffix + ".partial")
                with (
                    urllib.request.urlopen(url, timeout=60) as source,
                    partial.open("wb") as target,
                ):
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(chunk)
                if sha_file(partial) != expected:
                    raise ValueError("Downloaded official asset differs from its pinned SHA256")
                partial.rename(path)
            if sha_file(path) != expected:
                raise ValueError("Local official asset differs from its pinned SHA256")
            assets[name] = dict(url=url, sha256=expected, bytes=path.stat().st_size)
    except (OSError, urllib.error.URLError) as error:
        receipt = dict(
            status="UNAVAILABLE",
            experiment_id=EXPERIMENT_ID,
            reason=f"{type(error).__name__}: {error}",
            assets=assets,
            external_transfer_certified=False,
        )
        durable_json(directory / ("UNAVAILABLE_" + object_hash(receipt)[:16] + ".json"), receipt)
        return receipt
    receipt = dict(
        status="SOURCES_AVAILABLE",
        experiment_id=EXPERIMENT_ID,
        dataset_id=DATASET_ID,
        dataset_revision=DATASET_REVISION,
        author_revision=AUTHOR_REVISION,
        assets=assets,
        selection_frozen=False,
        external_transfer_certified=False,
    )
    durable_json(directory / "SOURCE_IDENTITY.json", receipt)
    return receipt


def _pixel_hash(image_bytes):
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb = image.convert("RGB")
        return hashlib.sha256(str(rgb.size).encode() + b"|RGB|" + rgb.tobytes()).hexdigest()


def audit_history_images(input_identity_path, image_root, output_path, *, chartqa_manifest):
    """CPU-only, read-only traversal of every image in the supplied prior inventory.

    Run on an authorized compute worker for a large remote inventory. No model,
    score, answer, or new TEST output is read by this identity audit.
    """
    root = Path(image_root).resolve(strict=True)
    identity = json.loads(Path(input_identity_path).read_text())
    files = identity.get("files", [])
    image_files = [
        r
        for r in files
        if Path(r["path"]).suffix.lower()
        in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff")
    ]
    if not image_files or len({r["path"] for r in files}) != len(files):
        raise ValueError("Nonempty, unique prior file inventory required")
    chartqa = [json.loads(s) for s in Path(chartqa_manifest).read_text().splitlines()]
    if len(chartqa) != 2500:
        raise ValueError("Complete prior ChartQA question manifest required")
    inventory = {r["path"]: r["sha256"] for r in image_files}
    if any(inventory.get(r["image_file"]) != r["image_sha256"] for r in chartqa):
        raise PermissionError("Historical ChartQA image identities differ from prior inventory")
    rows = []
    for entry in sorted(image_files, key=lambda r: r["path"]):
        relative = Path(entry["path"])
        path = (root / relative).resolve(strict=True)
        if relative.is_absolute() or not path.is_relative_to(root):
            raise PermissionError("Historical image escapes the verified prior root")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != entry["sha256"] or len(raw) != entry["bytes"]:
            raise PermissionError("Historical image changed from its recorded bytes")
        rows.append(dict(path=entry["path"], image_sha256=digest, pixel_sha256=_pixel_hash(raw)))
    receipt = dict(
        status="COMPLETE",
        source_manifest_sha256=sha_file(input_identity_path),
        chartqa_manifest_sha256=sha_file(chartqa_manifest),
        source_root=str(root),
        checked_images=len(rows),
        rows=rows,
        image_sha256=sorted({r["image_sha256"] for r in rows}),
        pixel_sha256=sorted({r["pixel_sha256"] for r in rows}),
        pixel_hash_definition="SHA256(str(RGB.size).encode()+b'|RGB|'+RGB.tobytes())",
        scope="all images in supplied historical inventory; no unknown-source independence claim",
        model_calls=0,
        new_test_outputs_accessed=False,
    )
    durable_json(output_path, receipt)
    return receipt


def prepare_chartqa(prior_manifest, image_root, output_root, *, history_receipt, type_reviews=()):
    """Freeze all 2,500 original ChartQA inputs with unchanged historical question text.

    Images are referenced below the new experiment root and must already have
    been copied with their original identity. Gold lives only in the audit file.
    """
    from sr_f1.chartqa import PROMPT

    rows = [json.loads(s) for s in Path(prior_manifest).read_text().splitlines()]
    if (
        len(rows) != 2500
        or len({r["qid"] for r in rows}) != 2500
        or Counter(r["subset"] for r in rows) != {"human": 1250, "augmented": 1250}
    ):
        raise ValueError("ChartQA must preserve the complete original 1250+1250 questions")
    if history_receipt.get("status") != "COMPLETE" or history_receipt.get(
        "chartqa_manifest_sha256"
    ) != sha_file(prior_manifest):
        raise PermissionError("ChartQA requires its verified historical image receipt")
    by_image = {r["image_sha256"]: r["pixel_sha256"] for r in history_receipt["rows"]}
    root = Path(image_root).resolve(strict=True)
    inputs, tasks = [], []
    for row in rows:
        path = (root / row["image_file"]).resolve(strict=True)
        if (
            not path.is_relative_to(root)
            or sha_file(path) != row["image_sha256"]
            or row["image_sha256"] not in by_image
        ):
            raise PermissionError("ChartQA image is missing or changed")
        inputs.append(
            dict(
                qid=row["qid"],
                prompt=PROMPT + "\n" + row["question"],
                image_file=row["image_file"],
                image_sha256=row["image_sha256"],
            )
        )
        tasks.append(
            dict(
                row,
                pool="CHARTQA",
                family="EXTERNAL",
                mode="external",
                root_id=row["image_id"],
                image_cluster=by_image[row["image_sha256"]],
            )
        )
    output = Path(output_root)
    registry = build_type_registry(rows, type_reviews, outputs_unseen=True)
    durable_json(output / "TYPE_REGISTRY.json", registry)
    for name, values in (("MODEL_INPUTS.jsonl", inputs), ("TASKS_AUDIT_ONLY.jsonl", tasks)):
        _write_bytes(
            output / name,
            "".join(
                json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in values
            ).encode(),
        )
    receipt = dict(
        status="FROZEN",
        pool="CHARTQA",
        image_count=2500,
        unique_image_count=len({r["image_id"] for r in rows}),
        protocol="chartqa_plain",
        image_root=str(root),
        source_manifest_sha256=sha_file(prior_manifest),
        history_receipt_sha256=object_hash(history_receipt),
        inputs_sha256=sha_file(output / "MODEL_INPUTS.jsonl"),
        audit_tasks_sha256=sha_file(output / "TASKS_AUDIT_ONLY.jsonl"),
        type_registry_sha256=object_hash(registry),
        typed_unresolved_count=registry["unresolved"],
        scoring="historical_code + v2 whitespace-stripped + frozen typed audit",
        scope="development-reused regression benchmark; not unseen confirmation",
    )
    durable_json(output / "FREEZE.json", receipt)
    return receipt


def blind_candidates(parquet_path):
    """Read an explicit four-column projection; gold and Year are never consulted."""
    import pyarrow.parquet as pq

    if sha_file(parquet_path) != PARQUET_SHA256:
        raise PermissionError("The frozen author dataset revision/hash is required")
    rows = pq.read_table(
        parquet_path, columns=["Question Type", "Question", "Paragraph", "image"]
    ).to_pylist()
    if len(rows) != 1948 or Counter(row["Question Type"] for row in rows) != {
        "Factoid": 1081,
        "Conversational": 311,
        "Fact Checking": 244,
        "Multi Choice": 214,
        "Hypothetical": 98,
    }:
        raise ValueError("Author dataset category inventory differs")
    candidates = []
    for index, row in enumerate(rows):
        if row["Question Type"] != "Factoid" or len(row["Question"]) != 1:
            continue
        content = row["image"]
        sha = hashlib.sha256(content).hexdigest()
        candidates.append(
            dict(
                qid=f"CHARTQAPRO_{DATASET_REVISION}_test_{index:06d}",
                source_row_index=index,
                image_id=sha,
                image_sha256=sha,
                image_pixel_sha256=_pixel_hash(content),
                question_type="Factoid",
                question_count=1,
                question=row["Question"][0],
                paragraph=row["Paragraph"] or "",
                image_bytes=content,
            )
        )
    return candidates


def select_chartqapro(candidates, *, prior_image_sha256=(), prior_pixel_sha256=(), limit=256):
    """Pure deterministic identity selection. Extra keys, including gold, are ignored."""
    if type(limit) is not int or not 1 <= limit <= 256:
        raise ValueError("Predefined external subset limit must be 1..256")
    old_sha, old_pixels = set(prior_image_sha256), set(prior_pixel_sha256)
    by_image, identities = defaultdict(list), set()
    for row in candidates:
        if row["qid"] in identities:
            raise ValueError("Duplicate native question identity")
        identities.add(row["qid"])
        if row["question_type"] != "Factoid" or row["question_count"] != 1:
            continue
        if row["image_sha256"] in old_sha or row.get("image_pixel_sha256") in old_pixels:
            continue
        by_image[row["image_id"]].append(row)
    # Native source has no image_id or qid. The frozen adapter explicitly uses
    # image-byte SHA and source revision + split + original row index instead.
    images = sorted(
        by_image,
        key=lambda image_id: (
            hashlib.sha256(("SRF3CE-external|" + image_id).encode()).hexdigest(),
            image_id,
        ),
    )[:limit]
    selected = []
    for image_id in images:
        row = min(
            by_image[image_id],
            key=lambda x: (hashlib.sha256(x["qid"].encode()).hexdigest(), x["qid"]),
        )
        selected.append(
            {
                k: row[k]
                for k in (
                    "qid",
                    "source_row_index",
                    "image_id",
                    "image_sha256",
                    "image_pixel_sha256",
                    "question_type",
                    "question_count",
                    "question",
                    "paragraph",
                )
            }
        )
    return selected


def official_direct_prompt(question, paragraph, prompt_source):
    """Use author-recommended Direct Factoid template with native non-answer context."""
    if not isinstance(question, str) or not isinstance(paragraph, str):
        raise ValueError("Native single question and paragraph text required")
    if sha_file(prompt_source) != PROMPT_SOURCE_SHA256:
        raise PermissionError("Frozen author-recommended prompt source differs")
    tree = ast.parse(Path(prompt_source).read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prompt_context"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(prompt_source), "exec"), namespace
    )
    # The Factoid/Direct branch never reads answers. Passing [] makes accidental
    # adoption of a conversational/gold-history path fail instead of leaking it.
    prompt = namespace["prompt_context"]([question], [], "Factoid", "Direct")
    return paragraph + "\n" + prompt if paragraph else prompt


def freeze_chartqapro(
    parquet_path,
    output_root,
    *,
    history_receipt,
    scorer_source,
    prompt_source,
    anls_wheel,
    type_reviews=(),
):
    """Publish selection before loading gold, and retain source bytes and exact inputs."""
    if (
        history_receipt.get("status") != "COMPLETE"
        or not history_receipt.get("source_manifest_sha256")
        or "image_sha256" not in history_receipt
        or "pixel_sha256" not in history_receipt
    ):
        raise RuntimeError(
            "Complete prior-used-image identity audit is needed before external freeze"
        )
    for path, expected in (
        (scorer_source, SCORER_SHA256),
        (prompt_source, PROMPT_SOURCE_SHA256),
        (anls_wheel, ANLS_WHEEL_SHA256),
    ):
        if sha_file(path) != expected:
            raise PermissionError("Official source/dependency identity differs")
    root = Path(output_root)
    candidates = blind_candidates(parquet_path)
    selected = select_chartqapro(
        candidates,
        prior_image_sha256=history_receipt["image_sha256"],
        prior_pixel_sha256=history_receipt["pixel_sha256"],
    )
    selection = dict(
        status="SELECTION_FROZEN",
        experiment_id=EXPERIMENT_ID,
        dataset_id=DATASET_ID,
        dataset_revision=DATASET_REVISION,
        parquet_sha256=PARQUET_SHA256,
        image_count=len(selected),
        selected=selected,
        history_receipt_sha256=object_hash(history_receipt),
        image_id_definition="SHA256 of original author image bytes (no native id exists)",
        qid_definition="CHARTQAPRO_<dataset revision>_test_<zero-based 6-digit source row>",
        image_selection='ascending SHA256("SRF3CE-external|image_id")',
        within_image_selection="ascending SHA256(qid)",
        eligibility="native Factoid (=paper Reasoning), one question",
        gold_used_for_selection=False,
        model_output_used_for_selection=False,
        near_duplicate_claim="byte/pixel exact identities only; pHash is not an independence proof",
    )
    durable_json(root / "SELECTION.json", selection)
    # With no independent visual annotations every row remains UNRESOLVED. This
    # retains official scores and never invents a type from a keyword or gold.
    type_registry = build_type_registry(selected, type_reviews, outputs_unseen=True)
    durable_json(root / "TYPE_REGISTRY.json", type_registry)
    # Only after publication may source gold be loaded to an audit-only sidecar.
    import pyarrow.parquet as pq

    gold = pq.read_table(parquet_path, columns=["Answer", "Year"]).to_pylist()
    source_by_qid = {row["qid"]: row for row in candidates}
    tasks, inputs = [], []
    for row in selected:
        image_file = "images/" + row["image_id"] + ".img"
        _write_bytes(root / image_file, source_by_qid[row["qid"]]["image_bytes"])
        native = gold[row["source_row_index"]]
        if not isinstance(native["Answer"], list) or len(native["Answer"]) != 1:
            raise ValueError("Selected native single-turn task has unexpected answer structure")
        prompt = official_direct_prompt(row["question"], row["paragraph"], prompt_source)
        inputs.append(
            dict(
                qid=row["qid"],
                prompt=prompt,
                image_file=image_file,
                image_sha256=row["image_sha256"],
            )
        )
        tasks.append(
            dict(
                row,
                pool="CHARTQAPRO",
                family="EXTERNAL",
                mode="external",
                root_id=row["image_id"],
                image_cluster=row["image_pixel_sha256"],
                official=dict(
                    Answer=native["Answer"], Year=native["Year"], **{"Question Type": "Factoid"}
                ),
            )
        )
    for name, rows in (("MODEL_INPUTS.jsonl", inputs), ("TASKS_AUDIT_ONLY.jsonl", tasks)):
        _write_bytes(
            root / name,
            "".join(
                json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows
            ).encode(),
        )
    for name, source in (
        ("evaluate_predictions.py", scorer_source),
        ("prompt_source.py", prompt_source),
        ("anls-0.0.2-py3-none-any.whl", anls_wheel),
    ):
        _write_bytes(root / "sources" / name, Path(source).read_bytes())
    for name, expected in (
        ("AUTHOR_LICENSE", AUTHOR_LICENSE_SHA256),
        ("VLM_EVAL_LICENSE", VLM_EVAL_LICENSE_SHA256),
    ):
        license_path = Path(scorer_source).parent / name
        if not license_path.is_file() or sha_file(license_path) != expected:
            raise RuntimeError("Retained official-source license bytes are required")
        _write_bytes(root / "sources" / name, license_path.read_bytes())
    receipt = dict(
        status="FROZEN" if selected else "UNAVAILABLE",
        experiment_id=EXPERIMENT_ID,
        reason=None if selected else "No eligible novel image roots after identity exclusion",
        image_count=len(selected),
        selection_sha256=object_hash(selection),
        inputs_sha256=sha_file(root / "MODEL_INPUTS.jsonl"),
        audit_tasks_sha256=sha_file(root / "TASKS_AUDIT_ONLY.jsonl"),
        scorer_sha256=SCORER_SHA256,
        scorer_revision=AUTHOR_REVISION,
        prompt_source_sha256=PROMPT_SOURCE_SHA256,
        prompt_revision=VLM_EVAL_REVISION,
        prompt_join="native Paragraph followed by LF and unmodified Direct/Factoid prompt_context",
        anls_version="0.0.2",
        anls_wheel_sha256=ANLS_WHEEL_SHA256,
        type_registry_sha256=object_hash(type_registry),
        typed_unresolved_count=type_registry["unresolved"],
        scope="predefined subset, not full ChartQAPro",
        external_transfer_certified=False,
    )
    durable_json(root / "FREEZE.json", receipt)
    return receipt


class ExternalInputStore:
    """Exact frozen external strings/images, without any audit-sidecar routing."""

    def __init__(self, root):
        root = Path(root).resolve(strict=True)
        freeze = json.loads((root / "FREEZE.json").read_text())
        self.image_root = Path(freeze.get("image_root", root)).resolve(strict=True)
        self.protocol = freeze.get("protocol", "chartqapro_official")
        path = root / "MODEL_INPUTS.jsonl"
        if freeze.get("status") != "FROZEN" or sha_file(path) != freeze["inputs_sha256"]:
            raise PermissionError("Frozen external model-input identity required")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self._inputs = {r["qid"]: r for r in rows}
        if len(rows) != len(self._inputs) or len(rows) != freeze["image_count"]:
            raise ValueError("External input coverage differs")
        registry = json.loads((root / "TYPE_REGISTRY.json").read_text())
        if object_hash(registry) != freeze["type_registry_sha256"]:
            raise PermissionError("Pre-output external type registry changed")

    def model_input(self, qid, protocol=None):
        if protocol is not None and protocol != self.protocol:
            raise PermissionError("External input cannot inherit synthetic assumptions")
        row = self._inputs[qid]
        return {k: row[k] for k in ("prompt", "image_file", "image_sha256")}


def load_official_scorer(scorer_source, anls_wheel):
    """Execute only pinned author functions plus pinned ANLS; leave environment intact."""
    if sha_file(scorer_source) != SCORER_SHA256 or sha_file(anls_wheel) != ANLS_WHEEL_SHA256:
        raise PermissionError("Official scorer or ANLS source bytes changed")
    anls_namespace = {"List": list}
    with zipfile.ZipFile(anls_wheel) as archive:
        distance_tree = ast.parse(archive.read("anls/metrics/dist.py"))
        score_tree = ast.parse(archive.read("anls/metrics/score.py"))
    exec(
        compile(
            ast.Module(
                body=[n for n in distance_tree.body if isinstance(n, ast.FunctionDef)],
                type_ignores=[],
            ),
            "anls-0.0.2/metrics/dist.py",
            "exec",
        ),
        anls_namespace,
    )
    anls_namespace["NL"] = anls_namespace["normalized_levenshtein_distance"]
    exec(
        compile(
            ast.Module(
                body=[n for n in score_tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[]
            ),
            "anls-0.0.2/metrics/score.py",
            "exec",
        ),
        anls_namespace,
    )
    namespace = dict(
        ast=ast,
        re=re,
        List=list,
        Optional=Optional,
        Any=Any,
        Tuple=tuple,
        Dict=dict,
        anls_score=anls_namespace["anls_score"],
    )
    tree = ast.parse(Path(scorer_source).read_text())
    names = {
        "fix_list_format",
        "parse_to_list",
        "to_float",
        "evaluate_single_answer",
        "relaxed_correctness_chartqapro",
        "evaluate_predictions_chartqapro",
    }
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in functions} != names:
        raise ValueError("Pinned official function inventory differs")
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(scorer_source), "exec"), namespace
    )
    return namespace["evaluate_predictions_chartqapro"]


def blind_item_identity(item):
    return object_hash(
        {
            k: item.get(k)
            for k in ("qid", "question", "image_sha256", "paragraph", "interface_contract")
        }
    )


def build_type_registry(items, reviews, *, outputs_unseen):
    """Two independent evidence-backed annotations; absence/disagreement is UNRESOLVED.

    No keyword classifier, gold argument, model output, or numeric value can choose
    a type. Review identities bind both the visible image and full question.
    """
    if outputs_unseen is not True:
        raise PermissionError("Type registry must freeze before model-output access")
    grouped = defaultdict(list)
    for review in reviews:
        grouped[review["qid"]].append(review)
    registry = []
    for item in items:
        identity = blind_item_identity(item)
        annotations = grouped[item["qid"]]
        valid = [
            r
            for r in annotations
            if r.get("input_identity") == identity
            and r.get("method") in REVIEW_METHODS
            and r.get("evidence")
            and r.get("reviewer_id")
            and r.get("kind") in TYPE_KINDS
            and r.get("comparison") in ("EXACT", "RELATIVE_5PCT")
            and r.get("used_gold") is False
            and r.get("used_model_output") is False
        ]
        identities = {r["reviewer_id"] for r in valid}
        rules = {(r["kind"], r["comparison"]) for r in valid}
        resolved = len(identities) >= 2 and len(rules) == 1 and len(valid) == len(annotations)
        kind, comparison = next(iter(rules)) if resolved else ("UNRESOLVED", None)
        if kind in ("YEAR", "COUNT_INT") and comparison != "EXACT":
            kind, comparison, resolved = "UNRESOLVED", None, False
        registry.append(
            dict(
                qid=item["qid"],
                input_identity=identity,
                kind=kind,
                comparison=comparison,
                status="RESOLVED" if resolved else "UNRESOLVED",
                reviews=annotations,
                reason="independent_agreement"
                if resolved
                else "insufficient_or_ambiguous_type_evidence",
            )
        )
    if len({r["qid"] for r in registry}) != len(registry) or set(grouped) - {
        r["qid"] for r in registry
    }:
        raise ValueError("Duplicate items or unregistered type review")
    return dict(
        status="FROZEN",
        experiment_id=EXPERIMENT_ID,
        outputs_unseen=True,
        rows=registry,
        unresolved=sum(r["status"] == "UNRESOLVED" for r in registry),
        evidence_scope="question plus visible image and explicit interface only",
    )


def typed_audit(prediction, gold, registry_row, *, original_score):
    result = dict(
        original_score=original_score,
        typed_score=None,
        type=registry_row["kind"],
        status="UNRESOLVED",
    )
    if registry_row.get("status") != "RESOLVED":
        return result
    try:
        target = normalize_typed(gold, registry_row["kind"])
        value = normalize_typed(prediction, registry_row["kind"])
    except (TypeError, ValueError, ZeroDivisionError):
        return dict(
            result, status="OUTPUT_UNRESOLVED", reason="not_an_unambiguous_registered_scalar"
        )
    rule = registry_row["comparison"]
    if rule == "EXACT":
        score = int(target == value)
    elif rule == "RELATIVE_5PCT" and registry_row["kind"] not in ("YEAR", "COUNT_INT"):
        score = (
            int(target == value)
            if target == 0
            else int(abs((value - target) / target) <= Fraction(1, 20))
        )
    else:
        raise ValueError("Unregistered typed comparison")
    return dict(
        result,
        typed_score=score,
        status="RESOLVED",
        normalized_prediction=str(value),
        normalized_gold=str(target),
        comparison=rule,
    )


def score_external(row, task, *, type_registry, official_scorer=None):
    """Keep historical/v2/typed scores distinct, with unresolved audit rows retained."""
    raw = row["raw_text"]
    registry = {x["qid"]: x for x in type_registry["rows"]}
    if type_registry.get("status") != "FROZEN" or not type_registry.get("outputs_unseen"):
        raise PermissionError("External typed audit needs a pre-output frozen registry")
    if row["qid"] not in registry:
        raise RuntimeError("Missing registry row cannot be silently assigned GENERAL_NUMERIC")
    if row["qid"] != task["qid"] or registry[row["qid"]]["input_identity"] != blind_item_identity(
        task
    ):
        raise PermissionError("External type registry input identity differs")
    if task["pool"] == "CHARTQA":
        from sr_f1.chartqa import score_chartqa

        gold = str(task["answer"])
        legacy, v2 = score_chartqa(gold, raw), score_chartqa(gold.strip(), raw.strip())
        audit = typed_audit(raw, gold, registry[row["qid"]], original_score=v2)
        return dict(
            historical_code=legacy,
            v2=v2,
            typed_audit=audit,
            benchmark_role="development-reused regression",
            finish=int(row["finish_reason"] == "eos"),
        )
    if task["pool"] != "CHARTQAPRO" or official_scorer is None:
        raise RuntimeError("Frozen official ChartQAPro evaluator required")
    official = official_scorer([dict(task["official"], prediction=raw)])
    if any(
        not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
        for v in official.values()
    ):
        raise RuntimeError("Official scorer failure cannot be recoded as a model error")
    audit = typed_audit(
        raw,
        task["official"]["Answer"][-1],
        registry[row["qid"]],
        original_score=official["Overall"],
    )
    return dict(
        official=official["Overall"],
        typed_audit=audit,
        scorer_sha256=SCORER_SHA256,
        benchmark_role="predefined novel-source subset",
        finish=int(row["finish_reason"] == "eos"),
    )
