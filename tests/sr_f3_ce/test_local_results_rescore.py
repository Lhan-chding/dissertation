"""Portable data-only gates fail on altered answers, inventories and source code."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f3_ce import contract, evaluation

REPO = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


local = load("local_results_rescore_test", REPO / "scripts/sr_f3_ce/local_results_rescore.py")
delivery = load("local_results_original_delivery", REPO / "scripts/sr_f3_ce/build_delivery.py")
fixtures = load("local_results_delivery_fixtures", Path(__file__).with_name("test_delivery.py"))


def training_fixture():
    task = dict(
        world=dict(categories=["a", "b"], domain=[0, 10], series=dict(Alpha=[1, 2], Beta=[2, 1])),
        query=dict(family="TOPK", aggregate="sum", k=1, direction="largest"),
    )
    request = dict(prompt="frozen prompt", image_file="images/q.png", image_sha256="a" * 64)
    text = contract.support_target(task["world"], task["query"], False)["text"]
    score = contract.score(
        text, task["world"], task["query"], finish_reason="eos", protocol="structured"
    )
    rows = [
        dict(
            qid="q",
            tokens=[1],
            sampler_logprobs=[-0.1],
            max_new_tokens=1024,
            raw_text=text,
            full_text=text,
            model_input=request,
            finish_reason="eos",
            generation_status="COMPLETE",
            technical_validation_errors=[],
            group_row_index=index,
            score=score,
            reward=score["J"],
        )
        for index in range(8)
    ]
    value = dict(records=rows, records_sha256=evaluation.object_hash(rows))
    return value, task, request, SimpleNamespace(evaluation=evaluation, contract=contract)


def test_training_rescores_genuine_production_answers():
    value, task, request, api = training_fixture()
    local._training_group(value, "q", task, request, api, delivery)


@pytest.mark.parametrize(
    "change", ["score", "reward", "text", "qid", "technical", "index", "logprobs", "prompt"]
)
def test_training_semantic_tampering_even_with_recomputed_group_digest(change):
    value, task, request, api = training_fixture()
    value = copy.deepcopy(value)
    row = value["records"][0]
    if change == "score":
        row["score"]["J"] = 1 - row["score"]["J"]
    elif change == "reward":
        row["reward"] = 1 - row["reward"]
    elif change == "text":
        row["full_text"] = "invalid answer"
    elif change == "qid":
        row["qid"] = "unregistered"
    elif change == "technical":
        row["technical_validation_errors"] = ["missing observation"]
    elif change == "index":
        row["group_row_index"] = 1
    elif change == "logprobs":
        row["sampler_logprobs"] = []
    else:
        row["model_input"]["prompt"] = "changed"
    value["records_sha256"] = evaluation.object_hash(value["records"])
    with pytest.raises(PermissionError):
        local._training_group(value, "q", task, request, api, delivery)


def test_training_missing_answer_blocks():
    value, task, request, api = training_fixture()
    value["records"].pop()
    value["records_sha256"] = evaluation.object_hash(value["records"])
    with pytest.raises(PermissionError, match="count/checksum"):
        local._training_group(value, "q", task, request, api, delivery)


def test_training_checksum_tampering_blocks():
    value, task, request, api = training_fixture()
    value["records_sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="count/checksum"):
        local._training_group(value, "q", task, request, api, delivery)


def test_original_endpoint_release_verifier_reused(tmp_path):
    groups, tasks, scorer = fixtures.release_fixture(tmp_path)
    checked = delivery.verify_release_stream(tmp_path, iter(groups), tasks, scorer)
    assert checked["records"] == 1
    assert checked["gzip_read_to_eof"] is True


def test_original_endpoint_wrong_production_score_blocks(tmp_path):
    groups, tasks, _ = fixtures.release_fixture(tmp_path)
    with pytest.raises(PermissionError, match="rescoring"):
        delivery.verify_release_stream(tmp_path, iter(groups), tasks, lambda *_: {"J": 99})


def test_original_endpoint_missing_group_blocks(tmp_path):
    _, tasks, scorer = fixtures.release_fixture(tmp_path)
    with pytest.raises(PermissionError, match="extra endpoint"):
        delivery.verify_release_stream(tmp_path, iter([]), tasks, scorer)


def endpoint_fixture(root):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    model = dict(model_id="FIXTURE")
    slot = dict(qid="q", protocol="structured", pool="TEST_ID", model_id="FIXTURE")
    group = dict(group_id="g", slots=[slot])
    request = dict(prompt="frozen")
    row = dict(slot, model_input=request)
    directory = root / "sealed/endpoint/FIXTURE"
    provenance = dict(model=model, model_identity="m", adapter_identity="a")
    write(root / "RUN_IDENTITY.json", {})
    write(root / "external/PANELS.json", {})
    write(directory / "MODEL_IDENTITY.json", provenance)
    payload = directory / "groups/g.json"
    write(payload, dict(group_id="g", records=[row], records_sha256=evaluation.object_hash([row])))
    entry = dict(
        group_id="g",
        path=str(payload.relative_to(root)),
        sha256=local._sha(payload),
        bytes=payload.stat().st_size,
    )
    manifest = dict(
        status="SEALED_COMPLETE",
        model_id="FIXTURE",
        raw_groups=[entry],
        model_identity_sha256=local._sha(directory / "MODEL_IDENTITY.json"),
        external_panels_sha256=local._sha(root / "external/PANELS.json"),
    )
    write(directory / "RAW_MANIFEST.json", manifest)
    fake = SimpleNamespace(
        endpoint_models=lambda: [model],
        group_slots=lambda _: [group],
        object_hash=evaluation.object_hash,
        validate_record=lambda *_a, **_kw: None,
    )
    store = SimpleNamespace(model_input=lambda *_a, **_kw: request)
    return SimpleNamespace(evaluation=fake), store, payload, manifest, directory


def test_endpoint_original_group_stream_preserved(tmp_path):
    api, store, _, _, _ = endpoint_fixture(tmp_path)
    result = list(local._endpoint_groups(tmp_path, [], api, store, {}))
    assert len(result) == 1
    assert result[0][0]["group_id"] == "g"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "bytes", "prompt"])
def test_endpoint_original_group_tamper_or_coverage_blocks(tmp_path, mutation):
    api, store, payload, manifest, directory = endpoint_fixture(tmp_path)
    if mutation == "missing":
        manifest["raw_groups"] = []
    elif mutation == "duplicate":
        manifest["raw_groups"] *= 2
    elif mutation == "bytes":
        payload.write_text("{}")
    else:
        value = json.loads(payload.read_text())
        value["records"][0]["model_input"]["prompt"] = "changed"
        value["records_sha256"] = evaluation.object_hash(value["records"])
        payload.write_text(json.dumps(value))
        manifest["raw_groups"][0].update(sha256=local._sha(payload), bytes=payload.stat().st_size)
    (directory / "RAW_MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(PermissionError):
        list(local._endpoint_groups(tmp_path, [], api, store, {}))


@pytest.mark.parametrize("relative", ["../secret", "/etc/passwd", "a/../secret", "a//../secret"])
def test_path_escape_rejected(tmp_path, relative):
    with pytest.raises(PermissionError):
        local._bounded(tmp_path, relative)


def test_source_manifest_not_self_authenticating(tmp_path):
    (tmp_path / "deployment").mkdir()
    (tmp_path / "deployment/SOURCE_MANIFEST.json").write_text('{"files": []}')
    with pytest.raises(PermissionError, match="source manifest differs"):
        local._source_api(tmp_path)


def test_external_relocation_preserves_frozen_input_and_checks_image(tmp_path):
    directory = tmp_path / "external/panel"
    (directory / "images").mkdir(parents=True)
    image = directory / "images/q.img"
    image.write_bytes(b"frozen source image")
    original = Path("/registered/server")
    freeze = dict(image_root="/registered/server/external/panel", protocol="chartqa_plain")
    inputs = {"q": dict(prompt="frozen", image_file="images/q.img", image_sha256=local._sha(image))}
    store = local._ExternalInputs(tmp_path, directory, freeze, inputs, original)
    assert store.model_input("q", protocol="chartqa_plain") == inputs["q"]
    with pytest.raises(PermissionError, match="protocol"):
        store.model_input("q", protocol="structured")
    image.write_bytes(b"corrupted")
    fresh = local._ExternalInputs(tmp_path, directory, freeze, inputs, original)
    with pytest.raises(PermissionError, match="image changed"):
        fresh.model_input("q")


def test_external_relocation_cannot_escape_registered_run(tmp_path):
    with pytest.raises(PermissionError, match="outside"):
        local._ExternalInputs(
            tmp_path, tmp_path, {"image_root": "/other/run"}, {}, Path("/registered/server")
        )


def test_subprocess_failure_not_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(
        local.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(returncode=1, stderr="source mismatch"),
    )
    with pytest.raises(RuntimeError, match="source mismatch"):
        local.verify_results(tmp_path)


def test_subprocess_partial_coverage_not_pass(monkeypatch, tmp_path):
    response = SimpleNamespace(returncode=0, stdout=json.dumps(dict(status="PASS", records=233339)))
    monkeypatch.setattr(local.subprocess, "run", lambda *_a, **_k: response)
    with pytest.raises(PermissionError, match="incomplete acceptance"):
        local.verify_results(tmp_path)
