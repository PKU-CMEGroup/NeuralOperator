import numpy as np

from utility.time_dependent_no.euler2d import (
    inspect_cpg_trajectory,
    load_cpg_primitive_sequence,
    make_cpg_graph_frame,
)
from utility.time_dependent_no.euler2d_synthetic import (
    SyntheticEuler2DConfig,
    make_degraded_synthetic_euler2d_prediction,
    make_synthetic_cpg_trajectory,
)
from utility.time_dependent_no.euler2d_fixture import run_euler2d_fixture_diagnostics
from utility.time_dependent_no.euler2d_metrics import shock_centroid_distance, shock_smearing_metrics


def test_synthetic_fixture_matches_cpg_schema_and_frame_loader():
    config = SyntheticEuler2DConfig(nx=8, ny=6, num_steps=5)
    group = make_synthetic_cpg_trajectory(config)

    summary = inspect_cpg_trajectory(group, name="fixture")
    primitive = load_cpg_primitive_sequence(group)
    frame = make_cpg_graph_frame(group, 1, num_steps=2)

    assert summary.missing_keys == []
    assert summary.num_time_steps == 5
    assert summary.num_nodes == 48
    assert summary.num_edges == 82
    assert primitive.shape == (5, 48, 4)
    assert frame["x"].shape == (48, 6)
    assert frame["future_primitives"].shape == (2, 48, 4)
    assert np.min(primitive[..., 0]) > 0.0
    assert np.min(primitive[..., 3]) > 0.0


def test_degraded_synthetic_predictions_activate_intended_diagnostics():
    config = SyntheticEuler2DConfig(nx=16, ny=10, num_steps=6)
    group = make_synthetic_cpg_trajectory(config)
    target = load_cpg_primitive_sequence(group)
    lagged = make_degraded_synthetic_euler2d_prediction(
        config, "lagged_shock", target=target
    )
    smeared = make_degraded_synthetic_euler2d_prediction(
        config, "smeared_shock", target=target
    )

    lagged_distance = shock_centroid_distance(
        lagged[-1], target[-1], group["pos"], group["edges"]
    )
    smeared_metrics = shock_smearing_metrics(smeared[-1], target[-1], group["edges"])

    assert lagged_distance > 0.0
    assert smeared_metrics["strength_ratio"] < 1.0
    assert smeared_metrics["thickness_ratio"] >= 1.0


def test_fixture_diagnostic_payload_has_expected_checks():
    config = SyntheticEuler2DConfig(nx=10, ny=8, num_steps=6)

    payload = run_euler2d_fixture_diagnostics(config)

    assert payload["evidence_class"] == "local_smoke_only"
    assert set(payload["cases"]) == {
        "perfect",
        "lagged_shock",
        "smeared_shock",
        "boundary_leak",
        "positivity_failure",
    }
    assert all(payload["checks"].values())
    assert payload["cases"]["perfect"]["summary"]["final_relative_l2"] == 0.0
    assert not payload["cases"]["positivity_failure"]["positivity"]["all_positive"]

