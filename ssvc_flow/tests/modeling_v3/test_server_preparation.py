import json
from pathlib import Path

import pytest

from src.modeling_v3.server_preparation import prepare_gpu


@pytest.mark.parametrize(
    "provenance",
    [
        {
            "execution_kind": "LOCAL_SMALL_ENGINEERING_TESTS",
            "platform": "Darwin",
            "slurm_job_id": None,
        },
        {"execution_kind": "SERVER_CPU", "platform": "Darwin", "slurm_job_id": "123"},
        {"execution_kind": "SERVER_CPU", "platform": "Linux-6.8", "slurm_job_id": None},
        {"execution_kind": "SERVER_CPU", "platform": "Linux-6.8", "slurm_job_id": ""},
    ],
)
def test_gpu_preparation_rejects_nonserver_acceptance_before_other_inputs(tmp_path, provenance):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "result.json").write_text(
        json.dumps({**provenance, "exit_code": 0, "sources_unchanged_during_run": True})
    )
    out = tmp_path / "q4"
    with pytest.raises(ValueError, match="server Slurm CPU"):
        prepare_gpu({}, {}, evidence, out)
    assert not out.exists()


def test_pending_q4_authorization_cannot_reach_runtime(tmp_path, monkeypatch):
    from src.modeling_v3 import vlm_campaign as campaign
    from src.modeling_v3.io import canonical_hash

    stage = {"phase": "Q4", "operator_authorization": "PENDING_FIRST_GPU_CONFIRMATION"}
    path = tmp_path / "stage.json"
    path.write_text(json.dumps(stage))
    binding = {"path": str(path), "sha256": campaign.file_hash(path)}
    plan = {
        "source_hash": canonical_hash(campaign.source_hashes()),
        "config_hash": canonical_hash({}),
    }
    plan["plan_hash"] = canonical_hash(plan)
    stage.update(operations=["vlm-smoke"], q4_plan_hash=plan["plan_hash"])
    path.write_text(json.dumps(stage))
    binding["sha256"] = campaign.file_hash(path)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(campaign, "_technical_gate", lambda *a: None)

    def forbidden_runtime(*args, **kwargs):
        pytest.fail("pending authorization reached runtime loading")

    monkeypatch.setattr(campaign, "load_runtime", forbidden_runtime)
    with pytest.raises(PermissionError, match="first GPU authorization"):
        campaign.vlm_smoke(
            {},
            {
                "v3_stage_lock": binding,
                "q4_bridge_plan": {"path": str(plan_path), "sha256": campaign.file_hash(plan_path)},
            },
            out=tmp_path / "out",
            allow_gpu=True,
            acknowledge_new_experiment=True,
        )


def test_q4_authorization_preserves_prepared_stage_and_binds_reviewed_handoff(tmp_path):
    from src.modeling_v3.io import canonical_hash, sha256_file, source_identity
    from src.modeling_v3.server_preparation import record_q4_authorization

    stage = {
        "phase": "Q4",
        "operator_authorization": "PENDING_FIRST_GPU_CONFIRMATION",
        "config_hash": canonical_hash({}),
        "source_hash": source_identity()["sha256"],
        "operations": ["vlm-smoke"],
        "q4_plan_hash": "frozen-plan",
    }
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(stage))
    bindings = {"v3_stage_lock": {"path": str(path), "sha256": sha256_file(path)}}
    handoff = tmp_path / "reviewed.md"
    handoff.write_text("Engineering fixture: exact command and measured timing review")
    handoff_binding = {"path": str(handoff), "sha256": sha256_file(handoff)}
    with pytest.raises(ValueError, match="confirmation"):
        record_q4_authorization(
            {},
            bindings,
            confirmation_text="",
            reviewed_handoff=handoff_binding,
            out=tmp_path / "bad",
        )
    resolved = record_q4_authorization(
        {},
        bindings,
        confirmation_text="Fixture authorization, no real GPU authorized",
        reviewed_handoff=handoff_binding,
        out=tmp_path / "authorized",
    )
    assert json.loads(path.read_text()) == stage
    authorized = json.loads(Path(resolved["v3_stage_lock"]["path"]).read_text())
    assert authorized["operator_authorization"] == "CONFIRMED_FIRST_GPU_SUBMISSION"
    assert authorized["q4_plan_hash"] == stage["q4_plan_hash"]
    assert authorized["first_gpu_confirmation"]["reviewed_handoff"] == handoff_binding
