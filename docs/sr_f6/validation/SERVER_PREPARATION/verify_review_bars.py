"""Read actual coloured pixel spans, independently of renderer metric receipts."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

base = Path(__file__).parent
receipt = json.loads((base / "PREFLIGHT/PROCESSOR.json").read_text())
worlds = {
    r["qid"]: r["world"] for r in json.loads((base / "REVIEW_WORLDS.json").read_text())["items"]
}
colours = {"Alpha": (63, 110, 156), "Beta": (184, 112, 48)}
items = []
for item in receipt["images"]:
    world = worlds[item["qid"]]
    measurements = {}
    for kind, key in [("source", "source_file"), ("processed", "processed_file")]:
        arr = np.array(Image.open(base / item[key]).convert("RGB")).astype("int16")
        sy, sx = arr.shape[0] / 896, arr.shape[1] / 1280
        ymin, ymax = round(125 * sy), round(755 * sy)
        bars = []
        for series, colour in colours.items():
            mask = np.max(np.abs(arr[ymin:ymax] - colour), axis=2) <= 2
            xs = np.flatnonzero(mask.any(axis=0))
            segments = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
            assert len(segments) == len(world["categories"]), (
                item["qid"],
                kind,
                series,
                len(segments),
            )
            for i, seg in enumerate(segments):
                x = round(float(np.median(seg)))
                top = ymin + int(np.flatnonzero(mask[:, x])[0])
                value = (755 * sy - top) / (6.3 * sy)
                expected = world["series"][series][i]
                assert abs(value - expected) <= 0.5, (item["qid"], kind, series, i, value, expected)
                target_x = (
                    110
                    + (1225 - 110) / len(segments) * (i + 0.5)
                    + (-0.5 if series == "Alpha" else 0.5) * 37 * 1.12
                ) * sx
                assert abs(x - target_x) <= 2, (item["qid"], kind, series, i, x, target_x)
                bars.append(
                    {
                        "category": world["categories"][i],
                        "series": series,
                        "world_value": expected,
                        "pixel_estimate": round(value, 4),
                        "absolute_error": round(abs(value - expected), 4),
                    }
                )
        measurements[kind] = bars
    items.append(
        {
            "qid": item["qid"],
            "category_order": world["categories"],
            "status": "PASS",
            "measurements": measurements,
        }
    )
report = {
    "status": "PASS",
    "reviewer_type": "PROGRAM",
    "method": (
        "Locate contiguous RGB colour spans in actual source and processed pixels; "
        "infer heights from fixed 0-100 chart axes; compare every left-to-right series "
        "bar with actual server world values. No renderer metrics or content hashes "
        "are used. Visual review separately verifies text/legend readability."
    ),
    "height_tolerance_chart_units": 0.5,
    "x_center_tolerance_pixels": 2,
    "source_worlds": "actual server DEV_CAL/TASKS_PRIVATE.jsonl for the 48 registered qids",
    "images": 48,
    "pixel_bars_checked": sum(len(v) for x in items for v in x["measurements"].values()),
    "max_absolute_height_error": max(
        b["absolute_error"] for x in items for values in x["measurements"].values() for b in values
    ),
    "items": items,
}
(base / "PIXEL_VALUE_CHECK.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n"
)
print({k: v for k, v in report.items() if k != "items"})
