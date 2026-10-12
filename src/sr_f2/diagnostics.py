"""Fixed, outcome-independent SR-F2 panels and diagnostic prompts.

The historical TEST_ID has 128 roots: v0/v3, both charts means 512 prompts,
not the document's 1024. This discrepancy is explicit, never silently padded.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from sr_f1.contract import FAMILIES
from sr_f1.evaluation import _seed, neutral_padding
from sr_f12.protocol import BASELINE, object_hash, scientific_matrix

EXPERIMENT_ID = "SR-F2-20261013"
STEPS = tuple(range(0, 97, 8))
PROTOCOL_VARIANTS = ("v0", "v3")
FREE_TEMPLATE = (
    "You may first write anything that helps you solve the question. "
    'On the final line, give the answer as {"answer": X}.'
)
LIST_TEMPLATE = (
    "First list only the chart values you need to answer the question. "
    'Then, on the final line, give the answer as {"answer": X}.'
)
NUMBER_RULES = (
    "Counts are integers. Express non-integer means as exact fraction strings "
    'such as "7/3", not rounded decimals. For an empty set: sum/count are 0; '
    "mean/range/max/min are null. A single-value range is 0. "
    "Resolve ranking ties by the category order printed on the chart."
)
NEUTRAL_TEMPLATE = (
    "This is a chart-reading exercise. Read the question carefully and use "
    "the chart. The chart belongs to a collection of exercises. "
)


def _rows(tasks):
    rows = list(tasks.values()) if isinstance(tasks, dict) else list(tasks)
    if len({t["qid"] for t in rows}) != len(rows):
        raise ValueError("Duplicate question identifier")
    return rows


def _select(tasks, pool, variants, expected_per_family):
    rows = sorted(
        (t for t in _rows(tasks) if t["pool"] == pool and t["variant"] in variants),
        key=lambda t: t["qid"],
    )
    if Counter(t["family"] for t in rows) != {f: expected_per_family for f in FAMILIES}:
        raise ValueError(f"Incomplete {pool} fixed panel")
    return rows


def monitor_panel(tasks):
    """Every MONITOR root, both charts, v0/v3: exactly 32 prompts per family."""
    return _select(tasks, "MONITOR", PROTOCOL_VARIANTS, 32)


def protocol_panel(tasks, *, protocol_variants=PROTOCOL_VARIANTS):
    if tuple(protocol_variants) != PROTOCOL_VARIANTS:
        raise ValueError("Protocol variants require the explicit v0/v3 registration")
    return _select(tasks, "TEST_ID", protocol_variants, 64 * len(protocol_variants))


def interval_panel(tasks):
    rows = sorted(
        (t for t in _rows(tasks) if t["pool"] == "TEST_COMPOSITION" and t["family"] == "INTERVAL"),
        key=lambda t: t["qid"],
    )
    roots = Counter(t["root_id"] for t in rows)
    if len(rows) != 32 or len(roots) != 16 or set(roots.values()) != {2}:
        raise ValueError("INTERVAL requires all 16 roots and both chart renderings")
    if any(t["query"]["aggregate"] != "mean" or t["variant"] != "v0" for t in rows):
        raise ValueError("INTERVAL diagnostic is mean on original composition prompts only")
    return rows


def generation_config(protocol):
    caps = {
        "evidence_answer": 768,
        "answer_only": 768,
        "plain_answer": 128,
        "free": 256,
        "list": 128,
    }
    if protocol not in caps:
        raise ValueError("Unknown SR-F2 protocol")
    return dict(
        do_sample=protocol != "plain_answer",
        temperature=1,
        top_p=1,
        top_k=0,
        max_new_tokens=caps[protocol],
        num_return_sequences=1,
    )


def protocol_prompt(row, protocol):
    if protocol not in ("free", "list"):
        raise ValueError("Working-space protocol required")
    question, sep, _ = row["plain_text"].partition("\n\n")
    if not sep or not question:
        raise ValueError("Cannot isolate question from original prompt")
    template = FREE_TEMPLATE if protocol == "free" else LIST_TEMPLATE
    return question + "\n\n" + template + " " + NUMBER_RULES


def interval_views(task, row):
    query, world = task["query"], task["world"]
    if query["family"] != "INTERVAL" or query["aggregate"] != "mean":
        raise ValueError("INTERVAL mean required")
    indices = list(range(query["start"] + (not query["include_start"]), query["end"] + 1))
    if len(indices) != 3:
        raise ValueError("Exactly three interval categories required")
    categories, series = world["categories"], world["series"]
    pairs = [
        f"{categories[i]}: Alpha={series['Alpha'][i]}, Beta={series['Beta'][i]}" for i in indices
    ]
    differences = [series["Alpha"][i] - series["Beta"][i] for i in indices]
    diffs = [f"{categories[i]}: {d}" for i, d in zip(indices, differences, strict=True)]
    operands = [str(d) if d >= 0 else f"({d})" for d in differences]
    suffixes = {
        "V0": "",
        "V1": "The exact chart values inside the stated interval are: " + "; ".join(pairs) + ".",
        "V2": "The Alpha minus Beta differences inside the stated interval are: "
        + "; ".join(diffs)
        + ".",
        "V3": "The complete expression for the requested answer is: ("
        + " + ".join(operands)
        + ")/3.",
        "V4": NEUTRAL_TEMPLATE,
    }
    grants = {
        "V0": "original_chart_and_question",
        "V1": "six_gold_chart_values",
        "V2": "three_selected_signed_differences",
        "V3": "complete_unevaluated_expression",
        "V4": "neutral_text_token_matched_to_V1",
    }
    result = []
    for view, suffix in suffixes.items():
        item = dict(
            qid=task["qid"],
            view=view,
            text=row["text"] + ("\n\n" + suffix if suffix else ""),
            image_file=row["image_file"],
            information_granted=grants[view],
            allowed_in_training=False,
        )
        if view == "V4":
            item.update(
                text=row["text"],
                runtime_neutral_padding=dict(
                    target_text=suffixes["V1"],
                    candidate=NEUTRAL_TEMPLATE,
                    exact_tokens_required=True,
                    matched_view="V1",
                ),
            )
        result.append(item)
    return result


def build_diagnostic_manifest(tasks, inputs, *, protocol_variants=PROTOCOL_VARIANTS):
    monitor = monitor_panel(tasks)
    protocols = protocol_panel(tasks, protocol_variants=protocol_variants)
    interval = interval_panel(tasks)
    manifest = dict(
        experiment_id=EXPERIMENT_ID,
        selection="all_roots_fixed_chart_and_variant_labels_no_outcomes",
        monitor_qids=[t["qid"] for t in monitor],
        monitor_per_family=32,
        monitor_K=4,
        monitor_steps=list(STEPS),
        protocol_qids=[t["qid"] for t in protocols],
        protocol_K=2,
        protocol_variants=list(protocol_variants),
        protocol_prompt_count=len(protocols),
        protocol_prompt_count_in_document=1024,
        protocol_count_discrepancy=len(protocols) != 1024,
        protocol_discrepancy_reason="128 roots x 2 charts x selected variant count",
        protocol_models=[BASELINE] + [m["model_id"] for m in scientific_matrix()],
        protocol_templates=dict(free=FREE_TEMPLATE, list=LIST_TEMPLATE, number_rules=NUMBER_RULES),
        generation={
            p: generation_config(p) for p in ("evidence_answer", "answer_only", "free", "list")
        },
        interval_qids=[t["qid"] for t in interval],
        interval_K=4,
        interval_views=["V0", "V1", "V2", "V3", "V4"],
        diagnostic_inputs=[item for t in interval for item in interval_views(t, inputs[t["qid"]])],
        neutral_matching="exact_native_token_count_of_complete_prompt_against_V1",
    )
    return dict(manifest, manifest_sha256=object_hash(manifest))


def _slots(panel, model_id, step, pool, protocol, k, view=None):
    for task in panel:
        for draw in range(k):
            parts = [model_id, step, pool, protocol, view or "original", task["qid"], draw]
            yield dict(
                slot_id="|".join(map(str, parts)),
                model_id=model_id,
                step=step,
                pool=pool,
                protocol=protocol,
                view=view,
                qid=task["qid"],
                root_id=task["root_id"],
                family=task["family"],
                chart=task["chart"],
                variant=task["variant"],
                draw=draw,
                samples_per_prompt=k,
                sampling_seed=_seed(EXPERIMENT_ID, step, pool, protocol, view, task["qid"], draw),
                group_seed=_seed(EXPERIMENT_ID, step, pool, protocol, view, task["qid"], 0),
                generation=generation_config(protocol),
            )


def monitor_slots(tasks, model_id, step):
    if step not in STEPS:
        raise ValueError("Monitor only at steps 0,8,...,96")
    return list(_slots(monitor_panel(tasks), model_id, step, "MONITOR_LITE", "evidence_answer", 4))


def _historical_step(model_id):
    if model_id == BASELINE:
        return 0
    if model_id not in {m["model_id"] for m in scientific_matrix()}:
        raise ValueError("Diagnostics apply only to the 13 historical models")
    return 96


def protocol_slots(tasks, model_id, *, protocol_variants=PROTOCOL_VARIANTS):
    panel = protocol_panel(tasks, protocol_variants=protocol_variants)
    return [
        s
        for p in ("free", "list")
        for s in _slots(panel, model_id, _historical_step(model_id), "TEST_ID", p, 2)
    ]


def interval_slots(tasks, model_id):
    return [
        s
        for v in ("V0", "V1", "V2", "V3", "V4")
        for s in _slots(
            interval_panel(tasks),
            model_id,
            _historical_step(model_id),
            "TEST_COMPOSITION",
            "evidence_answer",
            4,
            v,
        )
    ]


def prepare_input(slot, inputs, diagnostics, tokenizer=None):
    row = inputs[slot["qid"]]
    image = Path(row["image_file"])
    synthetic = bool(image.parts) and image.parts[0] == "images"
    chartqa = image.parts[:3] == ("external", "chartqa", "images")
    if (
        image.is_absolute()
        or ".." in image.parts
        or not (chartqa if slot["protocol"] == "plain_answer" else synthetic)
    ):
        raise ValueError("Unsafe registered image route")
    if slot["protocol"] == "plain_answer":
        if slot.get("view") or slot["pool"] != "CHARTQA":
            raise PermissionError("ChartQA requires original greedy external input")
        return dict(
            text=row["text"],
            image_file=row["image_file"],
            information_granted="original_chartqa_image_and_question",
        )
    if slot.get("view"):
        records = (
            diagnostics.get("diagnostic_inputs", diagnostics)
            if isinstance(diagnostics, dict)
            else diagnostics
        )
        if isinstance(records, dict):
            diagnostic = records[(slot["qid"], slot["view"])]
        else:
            hits = [d for d in records if (d["qid"], d["view"]) == (slot["qid"], slot["view"])]
            if len(hits) != 1:
                raise ValueError("Diagnostic prompt coverage differs")
            diagnostic = hits[0]
        if diagnostic["allowed_in_training"] is not False:
            raise PermissionError("Intervention prompt may not enter training")
        if diagnostic["image_file"] != row["image_file"]:
            raise PermissionError("Intervention must preserve the chart")
        output = {k: diagnostic[k] for k in ("text", "image_file", "information_granted")}
        if slot["view"] == "V4":
            if tokenizer is None:
                raise ValueError("Frozen native tokenizer required for V4")
            pad = diagnostic["runtime_neutral_padding"]
            output["text"], match = neutral_padding(
                diagnostic["text"], pad["target_text"], pad["candidate"], tokenizer
            )
            if match["token_difference"] != 0:
                raise ValueError("V4 requires exactly equal complete-prompt token length to V1")
            output["token_matching"] = dict(match, matched_view="V1", exact_tokens_required=True)
        return output
    protocol = slot["protocol"]
    text = (
        protocol_prompt(row, protocol)
        if protocol in ("free", "list")
        else row["plain_text" if protocol == "answer_only" else "text"]
    )
    return dict(
        text=text, image_file=row["image_file"], information_granted="original_chart_and_question"
    )
