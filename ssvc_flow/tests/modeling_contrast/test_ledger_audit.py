import json

from src.modeling_contrast.ledger_audit import audit_recorded_costs


def test_physical_cost_does_not_sum_overlapping_acquisition_scenarios(tmp_path):
    packet = tmp_path / "N2/packets/unit"
    packet.mkdir(parents=True)
    (packet / "packets.json").write_text(
        json.dumps(
            {
                "packets": [{"cost": {"generated_actions": 10, "reused_actions": 20}}],
                "C1_supplementary_costs": [{"cost": {"generated_actions": 3}}],
            }
        )
    )
    cost = tmp_path / "N2/cost/unit.json"
    cost.parent.mkdir()
    cost.write_text(
        json.dumps(
            {
                "fit_by_bank_budget": {"8": {"generated_actions": 999}},
                "direct_four_evaluation_banks": {"generated_actions": 30},
            }
        )
    )
    result = audit_recorded_costs(tmp_path)
    assert result["operation_counts"]["generated_actions"] == 43
    assert result["operation_counts"]["reused_actions"] == 20
    assert result["packet_ledger_count"] == 1


def test_parent_jacobian_cost_schema_is_explicitly_adapted(tmp_path):
    stage = tmp_path / "oracle_diagnostics"
    stage.mkdir()
    (stage / "SUMMARY.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": 1,
                "cost": {
                    "jacobian_calls": 2,
                    "batched_vjp_calls": 2,
                    "scored_prompt_policies": 144,
                    "action_score_values": 2304,
                    "logical_vjp_directions": 432,
                    "optimizer_updates": 0,
                },
            }
        )
    )
    result = audit_recorded_costs(tmp_path)
    assert result["operation_counts"]["Jacobian_VJP_calls"] == 2
    assert result["operation_counts"]["physical_policy_forwards"] == 144
    assert result["operation_counts"]["logical_vjp_directions"] == 432
