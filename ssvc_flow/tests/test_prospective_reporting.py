"""Synthetic receipt fixtures test provenance validation, never model efficacy."""

import copy
import csv
import json

import pytest

from src.modeling_v3.io import canonical_hash
from src.prospective_selection import jobs
from src.prospective_selection.features import LEVELS
from src.prospective_selection.protocol import load_protocol
from src.prospective_selection.reporting import development_report, final_report


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def training(config, directory, identity, recipe, steps=32):
    checkpoint = {"state_hash": f"fixture-state-{directory.name}", "path": "fixture-only"}
    manifest = {
        "kind": "PROSPECTIVE_TRAINING",
        "fixture": False,
        "runtime_identity": {"execution_kind": "REAL_CUDA_MODEL"},
        "identity": identity,
        "config_hash": canonical_hash(config),
        "recipe": recipe,
        "steps": steps,
        "B": 4,
        "K": 8,
        "Lnorm": 64,
        "initial_state_hash": "fixture-initial",
    }
    complete = {
        "status": "TRAINING_COMPLETE",
        "identity": identity,
        "steps": steps,
        "training_outputs": steps * 32,
        "checkpoint": checkpoint,
        "checkpoints": {str(steps): checkpoint},
        "elapsed_training_seconds": 10,
    }
    write(directory / "MANIFEST.json", manifest)
    write(directory / "COMPLETE.json", complete)
    return manifest, complete


@pytest.fixture
def report_fixture(tmp_path, monkeypatch):
    config = load_protocol()
    frozen = {
        "freeze_id": "fixture-freeze",
        "test_n": 12,
        "test_draws": 16,
        "best_static": "R0",
        "T_panel": {"prompts": [{"prompt_id": str(i)} for i in range(288)]},
    }
    decisions, tasks = {}, {}
    root = tmp_path / "runs"

    class FakeRegistry:
        def __init__(self, root, config):
            self.root = root

        def load_freeze(self):
            return frozen

        def decision(self, origin):
            return decisions[origin]

        def task(self, task_id):
            return tasks[task_id]

        def branch_output(self, origin, recipe, repeat):
            return root / "branches" / origin / recipe / str(repeat)

    monkeypatch.setattr(jobs, "TaskRegistry", FakeRegistry)
    registry = FakeRegistry(root, config)
    branches = []
    for index, spec in enumerate(config["origins"]["test_pool"][:12]):
        lid, step = spec["seed"], spec["anchors"][0]
        origin = f"{lid}_t{step}"
        decisions[origin] = {
            "lineage_id": lid,
            "origin_id": origin,
            "origin_step": step,
            "source_recipe": spec["source_recipe"],
            "made_before_run": True,
            "choices": {**dict.fromkeys(LEVELS, "R3"), LEVELS[2]: "R0"},
        }
        for recipe in ("R0", "R3", *config["actions"]["external_baselines"]):
            for repeat in (1, 2):
                branch = registry.branch_output(origin, recipe, repeat)
                branches.append(branch)
                payload = {
                    "lineage_id": lid,
                    "origin_id": origin,
                    "recipe_id": recipe,
                    "repeat": repeat,
                    "role": "locked_test",
                }
                task_id = f"{origin}-{recipe}-{repeat}"
                receipt = {
                    "task_id": task_id,
                    "kind": "branch",
                    "payload": payload,
                    "freeze_id": frozen["freeze_id"],
                    "decision_id": jobs.content_id(decisions[origin]),
                }
                tasks[task_id] = receipt
                _, train = training(
                    config, branch, {**payload, "authorization_receipt": receipt}, recipe
                )
                identity = {
                    "lineage_id": lid,
                    "origin_id": origin,
                    "policy_id": f"{origin}:{recipe}:{repeat}:H32",
                    "panel_id": "T",
                    "horizon": 32,
                    "repeat": repeat,
                    "role": "evaluation",
                    "draws": 16,
                    "checkpoint_state_hash": train["checkpoint"]["state_hash"],
                    "prompts_hash": canonical_hash(frozen["T_panel"]["prompts"]),
                    "config_hash": canonical_hash(config),
                }
                evaluation = {
                    "status": "EVALUATION_COMPLETE",
                    "identity": identity,
                    "generated_outputs": 288 * 16,
                    "elapsed_sampling_seconds": 5,
                    "summary": {
                        "J": 0.5 + (0.03 + index * 0.001 if recipe == "R3" else 0),
                        "samples": 288 * 16,
                        "prompts": 288,
                    },
                }
                write(branch / "evaluations/H32/MANIFEST.json", identity)
                write(branch / "evaluations/H32/COMPLETE.json", evaluation)
                write(branch / "EVAL_H32.json", evaluation)
    return config, root, tmp_path / "report", branches, frozen


def test_final_report_answers_comparisons_and_emits_recomputable_csv(report_fixture):
    config, root, out, _, _ = report_fixture
    result = final_report(config, root, out)
    assert result["primary"]["n_independent_lineages"] == 12
    for name, rows in (("primary_comparison.csv", 12), ("secondary_holm.csv", 4)):
        with (out / name).open() as stream:
            assert len(list(csv.DictReader(stream))) == rows
    chinese = (out / "FINAL_DECISION_zh.md").read_text()
    assert "额外修复结构是否改善未见训练原点" in chinese
    assert "固定配方" in chinese and "GDPO" in chinese and "SAW" in chinese
    assert "直接修复奖励" in chinese and "Holm" in chinese
    assert "```json" not in chinese


@pytest.mark.parametrize(
    "field,value",
    [
        ("panel_id", "D"),
        ("horizon", 8),
        ("lineage_id", 999),
        ("origin_id", "another-origin"),
        ("repeat", 2),
        ("policy_id", "other-policy"),
        ("config_hash", "other-config"),
        ("checkpoint_state_hash", "other-checkpoint"),
        ("prompts_hash", "other-panel"),
        ("draws", 32),
        ("role", "predecision"),
    ],
)
def test_final_rejects_self_consistent_wrong_evaluation_identity(report_fixture, field, value):
    config, root, out, branches, _ = report_fixture
    branch = branches[0]
    evaluation = json.loads((branch / "EVAL_H32.json").read_text())
    evaluation["identity"][field] = value
    # Alter all three receipts together: internal agreement cannot bypass frozen expectations.
    write(branch / "EVAL_H32.json", evaluation)
    write(branch / "evaluations/H32/COMPLETE.json", evaluation)
    write(branch / "evaluations/H32/MANIFEST.json", evaluation["identity"])
    with pytest.raises(ValueError, match="Evaluation identity mismatch"):
        final_report(config, root, out)
    assert not (out / "FINAL_DECISION_zh.md").exists()


@pytest.mark.parametrize("field,value", [("fixture", True), ("config_hash", "wrong")])
def test_final_rejects_spoofed_training_manifest(report_fixture, field, value):
    config, root, out, branches, _ = report_fixture
    path = branches[0] / "MANIFEST.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    write(path, manifest)
    with pytest.raises(ValueError, match="Training manifest"):
        final_report(config, root, out)


def test_final_does_not_relabel_cpu_execution_as_real(report_fixture):
    config, root, out, branches, _ = report_fixture
    path = branches[0] / "MANIFEST.json"
    manifest = json.loads(path.read_text())
    manifest["runtime_identity"]["execution_kind"] = "CPU_FIXTURE"
    write(path, manifest)
    with pytest.raises(ValueError, match="Training execution"):
        final_report(config, root, out)


def test_final_rejects_authorization_from_other_decision(report_fixture):
    config, root, out, branches, _ = report_fixture
    branch = branches[0]
    for filename in ("MANIFEST.json", "COMPLETE.json"):
        value = json.loads((branch / filename).read_text())
        value["identity"]["authorization_receipt"]["decision_id"] = "other-decision"
        write(branch / filename, value)
    with pytest.raises(ValueError, match="authorization"):
        final_report(config, root, out)


def test_final_rejects_nonreal_evaluation_status(report_fixture):
    config, root, out, branches, _ = report_fixture
    branch = branches[0]
    evaluation = json.loads((branch / "EVAL_H32.json").read_text())
    evaluation["status"] = "CPU_FIXTURE_COMPLETE"
    write(branch / "EVAL_H32.json", evaluation)
    write(branch / "evaluations/H32/COMPLETE.json", evaluation)
    with pytest.raises(ValueError, match="Evaluation completion"):
        final_report(config, root, out)


def test_final_missing_evidence_and_replaced_sidecar_fail(report_fixture):
    config, root, out, branches, _ = report_fixture
    path = branches[0] / "EVAL_H32.json"
    ev = json.loads(path.read_text())
    altered = copy.deepcopy(ev)
    altered["summary"]["J"] = 0.99
    write(path, altered)
    with pytest.raises(ValueError, match="sidecar"):
        final_report(config, root, out)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        final_report(config, root, out)
    assert not (out / "FINAL_DECISION_zh.md").exists()


def test_development_cannot_mark_fixture_source_as_first_four_complete(tmp_path):
    config = load_protocol()
    spec = config["origins"]["development"][0]
    source = tmp_path / "sources" / str(spec["seed"])
    _, complete = training(
        config,
        source,
        {"lineage_id": spec["seed"], "source_recipe": spec["source_recipe"], "role": "source"},
        spec["source_recipe"],
        96,
    )
    complete["status"] = "CPU_FIXTURE_COMPLETE"
    write(source / "COMPLETE.json", complete)
    with pytest.raises(ValueError, match="Training completion"):
        development_report(config, tmp_path, tmp_path / "report", first_four=True)


def test_development_missing_results_remain_incomplete(tmp_path):
    result = development_report(load_protocol(), tmp_path, tmp_path / "report", first_four=True)
    assert result["status"] == "INCOMPLETE"
    assert result["branches_measured"] == 0 and result["expected_branches"] == 88
