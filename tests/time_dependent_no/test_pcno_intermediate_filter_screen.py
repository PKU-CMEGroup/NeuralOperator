from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from pcno.pcno import PCNO, compute_Fourier_modes
from scripts.time_dependent_no.analyze_pcno_intermediate_filter_screen import (
    ARMS,
    _case_rows,
    _error_map_figure,
    _nonnegative_ratio,
    _predict_population,
    _primary_figure,
    _profile_figure,
    _save,
    _spectrum_figure,
    _structure_figure,
    _style,
    build_decision,
)
from scripts.time_dependent_no.fit_pcno_shock_representation import (
    RegisteredCase,
    build_batch,
)
from utility.time_dependent_no.pcno_intermediate_filter_screen import (
    NATIVE_REPLAY,
    REGISTERED_INTERVENTIONS,
    filter_channel_first,
    prepare_fixed_physical_operator,
    replay_pcno_intervention,
)
from utility.time_dependent_no.pcno_resolution_pathways import (
    graph_ball_device_invariants,
)
from utility.time_dependent_no.pcno_shock_representation import (
    ANCHORS,
    DOMAIN_LENGTHS,
    build_structured_cell_grid,
    make_translated_front_case,
    pcno_aux_tensors,
    pcno_case_input,
)


def _toy_problem() -> tuple[PCNO, object, torch.Tensor, tuple[torch.Tensor, ...]]:
    torch.manual_seed(29)
    grid = build_structured_cell_grid((32, 16))
    case = make_translated_front_case(
        grid,
        "pulse",
        position=ANCHORS[1] + 0.875 * grid.hx,
    )
    modes = torch.as_tensor(
        compute_Fourier_modes(2, [2, 2], list(DOMAIN_LENGTHS)),
        dtype=torch.float32,
    )
    model = PCNO(
        ndims=2,
        modes=modes,
        nmeasures=1,
        layers=[4, 4, 4],
        fc_dim=5,
        in_dim=4,
        out_dim=1,
    ).eval()
    return model, grid, pcno_case_input(case, grid), pcno_aux_tensors(grid)


def test_channel_first_filter_bypass_dc_and_inference_guard() -> None:
    grid = build_structured_cell_grid((32, 16))
    xx, yy = np.meshgrid(grid.x_centers, grid.y_centers, indexing="xy")
    field = 2.0 + np.cos(2.0 * np.pi * 12.0 * xx) * np.cos(
        2.0 * np.pi * 4.0 * yy / grid.lengths[1]
    )
    values = torch.as_tensor(field.reshape(1, 1, -1), dtype=torch.float32)

    bypass = filter_channel_first(values, grid, kind="zero")
    smooth = filter_channel_first(values, grid, kind="smooth")
    constant = torch.full_like(values, 3.25)

    assert bypass is values
    torch.testing.assert_close(
        filter_channel_first(constant, grid, kind="smooth"),
        constant,
        rtol=0.0,
        atol=2.0e-6,
    )
    assert torch.max(torch.abs(smooth - values)).item() > 0.1
    with pytest.raises(RuntimeError, match="inference-only"):
        filter_channel_first(values.requires_grad_(), grid, kind="smooth")


def test_native_replay_closes_and_every_arm_is_single_location() -> None:
    model, grid, model_input, aux = _toy_problem()
    original = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    direct = model(model_input, aux)
    replay = replay_pcno_intervention(
        model,
        model_input,
        aux,
        grid,
        intervention=NATIVE_REPLAY,
    )
    torch.testing.assert_close(replay, direct, rtol=1.0e-5, atol=1.0e-6)

    fixed = prepare_fixed_physical_operator(
        grid,
        device=model_input.device,
        dtype=model_input.dtype,
    )
    invariants = graph_ball_device_invariants(fixed)
    assert invariants["maximum_row_sum_error"] < 2.0e-7
    assert invariants["maximum_constant_error"] < 2.0e-7

    changes = {}
    for arm in REGISTERED_INTERVENTIONS:
        output = replay_pcno_intervention(
            model,
            model_input,
            aux,
            grid,
            intervention=arm,
            fixed_operator=fixed if arm == "R_grad_fixed_physical" else None,
        )
        assert output.shape == direct.shape
        assert torch.isfinite(output).all()
        changes[arm] = torch.max(torch.abs(output - direct)).item()

    assert all(value > 1.0e-8 for value in changes.values())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0.0, atol=0.0)


def test_invalid_interventions_fail_closed() -> None:
    model, grid, model_input, aux = _toy_problem()
    with pytest.raises(ValueError, match="unknown intervention"):
        replay_pcno_intervention(
            model,
            model_input,
            aux,
            grid,
            intervention="F0",
        )
    with pytest.raises(ValueError, match="prepared operator"):
        replay_pcno_intervention(
            model,
            model_input,
            aux,
            grid,
            intervention="R_grad_fixed_physical",
        )
    with pytest.raises(ValueError, match="nonfinal"):
        replay_pcno_intervention(
            model,
            model_input,
            aux,
            grid,
            intervention="S_postactivation",
            selected_layer=1,
        )
    assert _nonnegative_ratio(0.0, 0.0) == 1.0
    assert np.isinf(_nonnegative_ratio(1.0, 0.0))
    assert _nonnegative_ratio(2.0, 4.0) == 0.5


def test_batched_screen_reuses_native_output_and_emits_finite_metrics() -> None:
    model, grid, _, _ = _toy_problem()
    records = []
    for index, family in enumerate(("step", "pulse")):
        position = ANCHORS[1] + 0.875 * grid.hx
        records.append(
            RegisteredCase(
                case_id=f"held_{family}_a1_phase_0p875",
                split="held",
                family=family,
                anchor_index=1,
                phase=0.875,
                position=position,
                case=make_translated_front_case(grid, family, position=position),
            )
        )

    predictions, replay_error = _predict_population(
        model,
        records,
        grid,
        torch.device("cpu"),
        batch_size=2,
    )
    model_input, _, aux = build_batch(records, grid, torch.device("cpu"))
    direct = (
        model(model_input, aux)[..., 0].detach().numpy().reshape(2, grid.ny, grid.nx)
    )

    assert tuple(predictions) == ARMS
    np.testing.assert_array_equal(predictions["F0"], direct)
    assert replay_error < 1.0e-6
    for arm, values in predictions.items():
        rows = _case_rows(records, values, grid, arm=arm)
        assert len(rows) == 2
        assert all(np.isfinite(row["scalars"]["relative_increment_l2"]) for row in rows)


def test_zero_effect_cannot_pass_the_registered_decision_gate() -> None:
    grid = build_structured_cell_grid((64, 32))
    records = []
    for split, phase in (("train", 0.0), ("held", 0.875)):
        for family in ("step", "pulse"):
            for anchor_index, anchor in enumerate((0.25, 0.375, 0.5)):
                position = anchor + phase * grid.hx
                records.append(
                    RegisteredCase(
                        case_id=f"{split}_{family}_a{anchor_index}_phase_{phase}",
                        split=split,
                        family=family,
                        anchor_index=anchor_index,
                        phase=phase,
                        position=position,
                        case=make_translated_front_case(
                            grid, family, position=position
                        ),
                    )
                )
    exact = np.stack([record.case.increment for record in records])
    baseline_rows = _case_rows(records, exact, grid, arm="F0")
    runs = {}
    for seed in (1701, 1702, 1703):
        seed_runs = {}
        for arm in ARMS:
            seed_runs[arm] = {
                "native_rows": copy.deepcopy(baseline_rows),
                "nested_rows": [],
            }
        runs[f"s{seed}"] = seed_runs

    decision = build_decision(
        runs,
        (1701, 1702, 1703),
        strict_replay_pass=True,
    )
    assert decision["eligible_for_later_training"] == []
    assert decision["eligible_count"] == 0
    assert all(not row["screen_pass"] for row in decision["arms"].values())


def test_registered_visualization_package_renders(tmp_path) -> None:
    grid = build_structured_cell_grid((64, 32))
    records = []
    for split, phase in (("train", 0.0), ("held", 0.875)):
        for family in ("step", "pulse"):
            for anchor_index, anchor in enumerate(ANCHORS):
                position = anchor + phase * grid.hx
                records.append(
                    RegisteredCase(
                        case_id=f"{split}_{family}_a{anchor_index}_phase_{phase}",
                        split=split,
                        family=family,
                        anchor_index=anchor_index,
                        phase=phase,
                        position=position,
                        case=make_translated_front_case(
                            grid, family, position=position
                        ),
                    )
                )
    exact = np.stack([record.case.increment for record in records])
    xx = np.tile(grid.x_centers, (grid.ny, 1))
    prediction = exact + 2.0e-3 * np.sin(2.0 * np.pi * 12.0 * xx)[None, ...]
    baseline_rows = _case_rows(records, prediction, grid, arm="F0")
    runs = {
        f"s{seed}": {
            arm: {
                "native_rows": copy.deepcopy(baseline_rows),
                "nested_rows": [],
            }
            for arm in ARMS
        }
        for seed in (1701, 1702, 1703)
    }
    arrays = {f"s1701__native__{arm}": prediction.copy() for arm in ARMS}
    decision = build_decision(
        runs,
        (1701, 1702, 1703),
        strict_replay_pass=True,
    )

    _style()
    figures = (
        _primary_figure(decision, (1701, 1702, 1703)),
        _structure_figure(runs, (1701, 1702, 1703)),
        _profile_figure(arrays, records, grid, 1701),
        _error_map_figure(arrays, records, grid, 1701),
        _spectrum_figure(arrays, records, grid, 1701),
    )
    files = []
    for index, figure in enumerate(figures):
        files.extend(_save(figure, tmp_path, f"figure_{index}"))
    assert len(files) == 10
    assert all(path.is_file() and path.stat().st_size > 0 for path in files)
