from __future__ import annotations

import json
from pathlib import Path

import yaml

from compbias.audit.frozen_components import VLMRegimeSpec
from scripts.train_vlm_rl_regime import main

PINNED_MODEL = "66285546d2b821cf421d4f5eb2576359d3770cd3"


def test_regime_claim_boundaries_distinguish_acquisition_readout_and_mediator() -> None:
    lm_only = VLMRegimeSpec(
        regime_id="lm_only",
        vision_update="frozen",
        projector_update="frozen",
        language_update="lora",
    )
    projector_lm = VLMRegimeSpec(
        regime_id="projector_lm",
        vision_update="frozen",
        projector_update="full",
        language_update="lora",
    )
    vision_lora = VLMRegimeSpec(
        regime_id="vision_lora",
        vision_update="lora",
        projector_update="full",
        language_update="lora",
    )

    assert lm_only.acquisition_frozen
    assert lm_only.interface_regime("post_projector_activation") == "F0"
    assert lm_only.interface_regime("natural_evidence_prefix") == "F1"
    assert projector_lm.acquisition_frozen
    assert projector_lm.interface_regime("post_projector_activation") == "F1"
    assert vision_lora.interface_regime("post_projector_activation") == "F2"
    assert "acquisition_improvement" not in projector_lm.allowed_claims
    assert "readout_change" in projector_lm.allowed_claims


def _config(tmp_path: Path, regime: str) -> dict[str, object]:
    updates = {
        "lm_only": ("frozen", "frozen", "lora", 24),
        "projector_lm": ("frozen", "full", "lora", 32),
        "vision_lora": ("lora", "full", "lora", 48),
        "max_end_to_end": ("full", "full", "full", 80),
    }
    vision, projector, language, vram = updates[regime]
    return {
        "schema_version": 2,
        "experiment": f"qwen25vl3b_v2_{regime}",
        "model": {
            "name": "Qwen/Qwen2.5-VL-3B-Instruct",
            "revision": PINNED_MODEL,
        },
        "regime": {
            "id": regime,
            "vision_update": vision,
            "projector_update": projector,
            "language_update": language,
        },
        "training": {
            "framework": "verl",
            "algorithm": "grpo",
            "checkpoints": ["base", "10pct", "25pct", "50pct", "75pct", "final"],
            "seeds": [0, 1, 2],
            "pilot_inputs": 100,
            "mediators_per_input": 4,
            "forks_per_mediator": 4,
        },
        "interfaces": [
            "natural_evidence_prefix",
            "structured_scene_json",
            "post_projector_activation",
            "mid_fusion_activation",
            "pre_answer_evidence_state",
        ],
        "resources": {"minimum_gpu_vram_gb": vram, "precision": "bf16"},
        "gate": {
            "execution_permitted": False,
            "require_phase_d_human_review": True,
            "require_authenticated_extension": True,
            "require_target_gpu_smoke": True,
        },
        "output_plan": str(tmp_path / f"{regime}.json"),
    }


def test_each_vlm_regime_cli_stops_at_authenticated_large_gpu_boundary(tmp_path: Path) -> None:
    for regime in ("lm_only", "projector_lm", "vision_lora", "max_end_to_end"):
        config_path = tmp_path / f"{regime}.yaml"
        config_path.write_text(
            yaml.safe_dump(_config(tmp_path, regime), sort_keys=False),
            encoding="utf-8",
        )
        assert main(["--config", str(config_path)]) == 2
        plan = json.loads((tmp_path / f"{regime}.json").read_text(encoding="utf-8"))

        assert plan["schema_version"] == 2
        assert plan["execution_permitted"] is False
        assert plan["large_gpu_started"] is False
        assert plan["requires_large_gpu"] is True
        assert plan["model"]["revision"] == PINNED_MODEL
        assert plan["checkpoints"] == ["base", "10pct", "25pct", "50pct", "75pct", "final"]
        assert {
            "c_sel",
            "c_fork",
            "c_syn",
            "D_P",
            "D_R",
            "Gamma",
            "selection_ratio",
            "epsilon_alg",
            "iid_ood_gap",
        }.issubset(plan["required_metrics"])
        assert "authenticated_execution_extension_not_implemented" in plan["blockers"]


def test_vlm_regime_cli_rejects_attempt_to_enable_execution(tmp_path: Path) -> None:
    config = _config(tmp_path, "lm_only")
    config["gate"]["execution_permitted"] = True
    config_path = tmp_path / "unsafe.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    assert main(["--config", str(config_path)]) == 3
    assert not (tmp_path / "lm_only.json").exists()
