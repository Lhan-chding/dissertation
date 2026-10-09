"""Author-pinned ChartQA test acquisition, scoring and image-cluster inference.

Run ``python -m sr_f1.chartqa --root RUN_ROOT`` before model evaluation. Only the
2,500-question test split is acquired; tables/annotations never enter a prompt.
The original ChartQA T5/VL-T5 README warns their inline metric is exact accuracy.
The relaxed metric here conforms to the same authors' UniChart implementation,
whose immutable source is retained and hashed in the dataset receipt.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

from .evaluation import (
    encoded,
    evaluation_checkpoint,
    evaluation_model_identity,
    generation_config,
    model_ids,
    read_jsonl,
    sha_file,
    validate_raw,
)

DATASET_REVISION = "af8b6f5c08c95085271561c2a3f9d15f2b5a9031"
AUTHOR_REPO_REVISION = "044eabfc306abfe9340c5741f0093aefc5973d06"
SCORER_REVISION = "56dd59af1c3ebec1773fe8a4bf43694cd565130c"
SCORER_SHA256 = "22f627ab2396fc43c3edaf568148ea6c1f3edfbbc592fdef14668bfdf2b5e4ab"
PARQUET_SHA256 = "84889bae30aaf6e24ff899e1604221606c0b65e658b6c70a7d02c79f50fc87df"
SCORER_URL = (
    f"https://raw.githubusercontent.com/vis-nlp/UniChart/{SCORER_REVISION}/model/chartqa_model.py"
)
PARQUET_URL = f"https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/{DATASET_REVISION}/data/test-00000-of-00001.parquet"
PROMPT = (
    "Answer the question using the chart. Give only the short final answer, without explanation."
)


def relaxed_accuracy(target, prediction):
    """Mirror author compute_metric, including numeric-zero fallback behaviour.

    Whitespace around both raw strings is stripped as by the author inference
    wrapper; no number extraction, percent conversion or answer repair occurs.
    """
    gold, pred = str(target).strip(" "), str(prediction).strip(" ")
    try:
        gold = float(gold)
        pred = float(pred)
        return int(abs(gold - pred) / abs(gold) <= 0.05)
    except (ValueError, ZeroDivisionError, OverflowError):
        return int(str(gold).lower() == str(pred).lower())


def score_chartqa(target, prediction):
    return dict(
        relaxed_accuracy=relaxed_accuracy(target, prediction),
        exact_match=int(str(target).strip(" ").lower() == str(prediction).strip(" ").lower()),
    )


def verify_author_scorer(path):
    """Execute only the pinned, hash-verified seven-line metric, not its imports."""
    path = Path(path)
    if sha_file(path) != SCORER_SHA256:
        raise ValueError("Author metric source bytes changed")
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ChartQAModule")
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "compute_metric"
    )
    method.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    cases = [
        ("100", "105"),
        ("100", "105.01"),
        ("-100", "-95"),
        ("0", "0.0"),
        ("0", "0.0001"),
        ("Alpha", "alpha"),
        ("5%", "5%"),
        ("5%", "0.05"),
        ("1,000", "1000"),
        ("1", "the answer is 1"),
        ("0", "-0"),
        ("nan", "nan"),
    ]
    results = []
    for target, prediction in cases:
        author = int(namespace["compute_metric"](None, target, prediction))
        local = relaxed_accuracy(target, prediction)
        if author != local:
            raise ValueError("Local scorer differs from pinned author implementation")
        results.append(dict(target=target, prediction=prediction, score=local))
    return dict(
        status="VERIFIED_AGAINST_PINNED_AUTHOR_FUNCTION",
        cases=results,
        source_url=SCORER_URL,
        source_sha256=SCORER_SHA256,
        relative_tolerance=0.05,
        normalization="strip ASCII spaces, then author float/lower behaviour",
        zero_truth="author numeric conversion then division-by-zero string fallback",
        percent_handling="no conversion",
        implementation="UniChart ChartQAModule.compute_metric",
    )


def _download(url, path, expected_sha256=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=60) as source, partial.open("wb") as target:
            for block in iter(lambda: source.read(1048576), b""):
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
        partial.replace(path)
    actual = sha_file(path)
    if expected_sha256 and actual != expected_sha256:
        raise ValueError("Downloaded asset hash differs from frozen identity")
    return dict(url=url, sha256=actual, bytes=path.stat().st_size)


def acquire(root):
    """Network/permission failures produce an explicit external-only failure receipt."""
    root = Path(root).resolve()
    destination = root / "external/chartqa"
    destination.mkdir(parents=True, exist_ok=True)
    receipt_path = destination / "IDENTITY.json"
    try:
        import pyarrow.parquet as pq
        from PIL import Image

        parquet = destination / "test.parquet"
        assets = {"test_parquet": _download(PARQUET_URL, parquet, PARQUET_SHA256)}
        author = {}
        for subset in ("human", "augmented"):
            url = f"https://raw.githubusercontent.com/vis-nlp/ChartQA/{AUTHOR_REPO_REVISION}/ChartQA%20Dataset/test/test_{subset}.json"
            path = destination / f"test_{subset}.json"
            assets[subset] = _download(url, path)
            author[subset] = json.loads(path.read_text())
            if len(author[subset]) != 1250:
                raise ValueError("Author split identity differs from registered 1250+1250")
        scorer = destination / "author_chartqa_model.py"
        assets["scorer"] = _download(SCORER_URL, scorer, SCORER_SHA256)
        metric_receipt = verify_author_scorer(scorer)
        source_rows = pq.read_table(parquet).to_pylist()
        if len(source_rows) != 2500 or Counter(r["type"] for r in source_rows) != {
            "human": 1250,
            "augmented": 1250,
        }:
            raise ValueError("Dataset must contain the complete 2500-question author test split")
        for subset in author:
            expected = Counter((r["imgname"], r["query"], str(r["label"])) for r in author[subset])
            actual = Counter(
                (r["imgname"], r["query"], str(r["label"]))
                for r in source_rows
                if r["type"] == subset
            )
            if expected != actual:
                raise ValueError("HF question identities do not match pinned author JSON")
        records, images = [], {}
        for i, source in enumerate(source_rows):
            payload = source["image"]
            image_hash = hashlib.sha256(payload).hexdigest()
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()
            with Image.open(io.BytesIO(payload)) as image:
                width, height = image.size
            image_file = f"external/chartqa/images/{image_hash}.png"
            path = root / image_file
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and sha_file(path) != image_hash:
                raise ValueError("Existing external image hash mismatch")
            if not path.exists():
                path.write_bytes(payload)
            images[image_hash] = dict(
                image_file=image_file, sha256=image_hash, width=width, height=height
            )
            records.append(
                dict(
                    qid=f"chartqa-{i:04d}",
                    subset=source["type"],
                    image_id=image_hash,
                    image_file=image_file,
                    image_sha256=image_hash,
                    source_imgname=source["imgname"],
                    question=source["query"],
                    answer=source["label"],
                )
            )
        manifest = destination / "TEST_MANIFEST.jsonl"
        contents = "".join(encoded(r) + "\n" for r in records)
        if manifest.exists() and manifest.read_text() != contents:
            raise ValueError("Existing ChartQA manifest identity changed")
        manifest.write_text(contents)
        receipt = dict(
            status="AVAILABLE_VERIFIED",
            dataset="ahmed-masry/ChartQA",
            dataset_revision=DATASET_REVISION,
            author_repository_revision=AUTHOR_REPO_REVISION,
            question_count=2500,
            subset_counts={"human": 1250, "augmented": 1250},
            unique_image_count=len(images),
            assets=assets,
            images=list(images.values()),
            manifest_sha256=sha_file(manifest),
            scoring=metric_receipt,
            tables_or_annotations_as_input=False,
            pretrained_contamination_excluded=False,
        )
    except Exception as exc:
        # A corrupted identity is retained and visible, never substituted by another benchmark.
        receipt = dict(
            status="UNAVAILABLE",
            exception_type=type(exc).__name__,
            reason=str(exc),
            dataset_revision=DATASET_REVISION,
            main_matrix_may_continue=True,
            external_generalization_claim_allowed=False,
        )
    if receipt_path.exists() and receipt_path.read_text() != encoded(receipt) + "\n":
        history = destination / "acquisition_history"
        history.mkdir(exist_ok=True)
        (history / (sha_file(receipt_path) + ".json")).write_bytes(receipt_path.read_bytes())
    receipt_path.write_text(encoded(receipt) + "\n")
    return receipt


def load_verified_manifest(root):
    root = Path(root)
    receipt = json.loads((root / "external/chartqa/IDENTITY.json").read_text())
    if receipt["status"] != "AVAILABLE_VERIFIED":
        raise ValueError("External ChartQA is unavailable")
    manifest = root / "external/chartqa/TEST_MANIFEST.jsonl"
    if sha_file(manifest) != receipt["manifest_sha256"]:
        raise ValueError("ChartQA manifest no longer has the acquired identity")
    rows = list(read_jsonl(manifest))
    if len(rows) != 2500:
        raise ValueError("Incomplete external test panel")
    return rows


def iter_chartqa_slots(manifest, model_id):
    if model_id not in model_ids():
        raise ValueError("Unregistered model")
    if len(manifest) != 2500:
        raise ValueError("No partial ChartQA convenience subsets")
    for item in manifest:
        yield dict(
            slot_id=f"{model_id}|CHARTQA|{item['qid']}",
            model_id=model_id,
            pool="CHARTQA",
            protocol="plain_answer",
            qid=item["qid"],
            draw=0,
            image_id=item["image_id"],
            subset=item["subset"],
            generation=generation_config("plain_answer"),
            sampling_seed=0,
            step=0 if model_id == "SRF1_COMMON_START" else 96,
        )


def _evaluate_chartqa(runtime, root, model_id, boundary):
    root = Path(root).resolve()
    manifest = load_verified_manifest(root)
    items = {r["qid"]: r for r in manifest}
    path = root / "raw/chartqa" / f"{model_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = {}
    if path.exists():
        for row in read_jsonl(path):
            if row["slot_id"] in seen:
                raise ValueError("Duplicate ChartQA output slot")
            seen[row["slot_id"]] = row
    identity = getattr(runtime, "stable_model_identity", None)
    if identity is None:
        identity = getattr(runtime, "identity", None)
    identity = identity() if callable(identity) else identity
    completed = 0
    with path.open("a") as stream:
        for slot in iter_chartqa_slots(manifest, model_id):
            if boundary["requested"] or (root / "STOP").exists():
                reason = "STOP_REQUESTED" if (root / "STOP").exists() else "PREEMPTION"
                return evaluation_checkpoint(root, path, completed + len(seen), reason)
            item = items[slot["qid"]]
            text = PROMPT + "\n" + item["question"]
            image_path = (root / item["image_file"]).resolve()
            if not image_path.is_relative_to(root) or sha_file(image_path) != item["image_sha256"]:
                raise ValueError("External image identity invalid")
            input_hash = hashlib.sha256(
                encoded(dict(text=text, image_hash=item["image_sha256"])).encode()
            ).hexdigest()
            if slot["slot_id"] in seen:
                previous = seen.pop(slot["slot_id"])
                validate_raw(previous, slot)
                if previous["input_hash"] != input_hash or (
                    identity
                    and evaluation_model_identity(previous["model_identity"])
                    != evaluation_model_identity(identity)
                ):
                    raise ValueError("ChartQA resume identity changed")
                completed += 1
                continue
            persisted = []

            def persist(
                completion,
                slot=slot,
                text=text,
                input_hash=input_hash,
                item=item,
                persisted=persisted,
            ):
                row = dict(completion)
                row.update(slot)
                row.update(
                    status="GENERATED",
                    text=text,
                    input_hash=input_hash,
                    image_file=item["image_file"],
                    image_hash=item["image_sha256"],
                    information_granted="original_chart_and_question",
                )
                row.setdefault("model_identity", identity)
                if "adapter_identity" not in row:
                    adapter = getattr(runtime, "adapter_identity", None)
                    row["adapter_identity"] = adapter() if callable(adapter) else adapter
                stream.write(encoded(row) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                persisted.append(row)

            runtime.generate(
                text=text,
                image_path=image_path,
                run_root=root,
                seed=0,
                generation=slot["generation"],
                protocol="plain_answer",
                on_completion=persist,
            )
            if len(persisted) != 1:
                raise RuntimeError("Expected one durable callback per actual generation")
            validate_raw(persisted[0], slot)
            completed += 1
            if boundary["requested"] or (root / "STOP").exists():
                reason = "STOP_REQUESTED" if (root / "STOP").exists() else "PREEMPTION"
                return evaluation_checkpoint(root, path, completed, reason)
    if seen:
        raise ValueError("Unexpected ChartQA slots")
    return dict(
        status="COMPLETE",
        artifacts=[str(path.relative_to(root))],
        metadata=dict(generated_slots=2500, scores_released=False),
    )


def evaluate_chartqa(runtime, root, model_id):
    from .training import stop_at_committed_boundary

    with stop_at_committed_boundary() as boundary:
        return _evaluate_chartqa(runtime, root, model_id, boundary)


def chartqa_summary(rows):
    groups = {}
    for model in sorted({r["model_id"] for r in rows}):
        groups[model] = {}
        for subset in ("human", "augmented", "combined"):
            panel = [
                r
                for r in rows
                if r["model_id"] == model and (subset == "combined" or r["subset"] == subset)
            ]
            expected = 2500 if subset == "combined" else 1250
            if len(panel) != expected:
                raise ValueError("Incomplete ChartQA panel is technical missingness")
            clusters = sorted({r["image_id"] for r in panel})
            rng = np.random.default_rng(812901)
            sampled = rng.integers(0, len(clusters), (5000, len(clusters)))
            result = dict(question_count=len(panel), image_cluster_count=len(clusters))
            for metric in ("relaxed_accuracy", "exact_match"):
                sums = np.array(
                    [
                        sum(r["independent_score"][metric] for r in panel if r["image_id"] == key)
                        for key in clusters
                    ]
                )
                sizes = np.array([sum(r["image_id"] == key for r in panel) for key in clusters])
                bootstrap = sums[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)
                result[metric] = dict(
                    mean=float(sums.sum() / sizes.sum()),
                    CI95=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                )
            result["scope"] = (
                "image-cluster measurement uncertainty; pretraining contamination not excluded"
            )
            groups[model][subset] = result
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    receipt = acquire(args.root)
    print(encoded({k: v for k, v in receipt.items() if k not in ("images", "assets", "scoring")}))
    return 0 if receipt["status"] == "AVAILABLE_VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
