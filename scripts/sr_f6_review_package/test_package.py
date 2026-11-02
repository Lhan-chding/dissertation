"""Small regression checks for the saved-evidence exporter, with no remote I/O."""

import importlib.util
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_zip_path_boundary():
    build = load("build")
    assert build.safe("02_EVALUATION/b0_U11_answers.jsonl")
    for path in ("/absolute", "../escape", "a/../../escape", "a\\escape"):
        assert not build.safe(path)


def test_nested_model_input_image_is_included(tmp_path, monkeypatch):
    exporter = load("export_saved")
    monkeypatch.setattr(exporter, "ROOT", tmp_path)
    image = b"saved-image-bytes"
    image_file = tmp_path / "DATA/images/chart.png"
    image_file.parent.mkdir(parents=True)
    image_file.write_bytes(image)
    for panel in ("DEV_EVAL", "DEV_PILOT"):
        public = tmp_path / f"PHASE_RUNS/PILOT_V3/INPUTS_V3/{panel}/MODEL_INPUTS.jsonl"
        public.parent.mkdir(parents=True)
        public.write_text(
            json.dumps(
                {
                    "model_input": {
                        "image_file": "DATA/images/chart.png",
                        "image_sha256": exporter.hashlib.sha256(image).hexdigest(),
                    }
                }
            )
            + "\n"
        )
        private = tmp_path / f"DATA/{panel}/TASKS_PRIVATE.jsonl"
        private.parent.mkdir(parents=True)
        private.write_text("{}\n")
    for rel in ("PILOT_FROZEN/PILOT_SPEC_FROZEN.json", "JOBS/PLAN.json"):
        path = tmp_path / "PHASE_RUNS/PILOT_V3" / rel
        path.parent.mkdir(parents=True)
        path.write_text("{}\n")
    buffer = io.BytesIO()
    monkeypatch.setattr(exporter.sys, "stdout", SimpleNamespace(buffer=buffer))
    exporter.export("data")
    with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive:
        assert archive.testzip() is None
        assert archive.read("04_DESIGN/DATA/images/chart.png") == image
        assert archive.namelist().count("04_DESIGN/DATA/images/chart.png") == 1
