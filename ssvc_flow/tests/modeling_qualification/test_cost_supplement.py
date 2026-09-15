from src.modeling_qualification.cost_supplement import allocate_panel_costs


def test_allocated_counts_and_io_reconstruct_measured_totals():
    summary = {
        "exact_panel_calls": 100,
        "exact_seconds": 2.0,
        "branch_seconds": 3.0,
        "measurement_seconds": 5.0,
        "output_io_seconds": 7.0,
    }
    manifest = {
        "trajectories": [1, 2],
        "anchors": [8, 24, 40],
        "bank_roles": {"fit": list(range(8)), "diagnostic": [8, 9], "evaluation": [10, 11, 12, 13]},
    }
    sizes = {"observations_file": 100, "oracle_file": 200, "rng_file": 300, "counts_file": 400}
    value = allocate_panel_costs(summary, manifest, sizes, [16, 64, 256], 5)
    assert value["status"] == "ALLOCATED_ESTIMATE"
    assert value["finite_count_panels_allocated"] == 1500
    assert value["measurement_total_reconstructed_seconds"] == 5.0
    assert abs(value["output_io_total_reconstructed_seconds"] - 7.0) < 1e-12
    assert value["not_a_per_configuration_measurement"]
