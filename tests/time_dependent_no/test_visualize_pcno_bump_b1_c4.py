from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import scripts.time_dependent_no.visualize_pcno_bump_b1_c4 as visualizer
from scripts.time_dependent_no.visualize_pcno_bump_b1_c4 import (
    _configure_matplotlib,
    _field_scales,
    free_pressure_fields,
    one_step_residual_fields,
    plot_loss_curves,
    render_comparison_animation,
)


def _bundle() -> dict[str, np.ndarray]:
    reference = np.zeros((80, 3, 4), dtype=np.float32)
    for call in range(80):
        reference[call, :, 0] = 2.0 + call
        reference[call, :, 1] = 0.1
        reference[call, :, 2] = 0.2
        reference[call, :, 3] = 20.0 + 2.0 * call
    free128 = reference.copy()
    free256 = reference.copy()
    free128[1:, :, 3] += np.arange(1, 80, dtype=np.float32)[:, None]
    free256[1:, :, 3] -= np.arange(1, 80, dtype=np.float32)[:, None]
    teacher128 = reference[1:].copy()
    teacher256 = reference[1:].copy()
    teacher128[:, :, 0] += 0.25
    teacher256[:, :, 0] -= 0.5
    teacher128[:, :, 3] += 1.5
    teacher256[:, :, 3] -= 2.0
    return {
        "trajectory": np.asarray("synthetic"),
        "positions": np.asarray([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
        "physical_times": np.arange(80, dtype=np.float64) * 0.1,
        "reference_states_conservative": reference,
        "n128_free_states_conservative": free128,
        "n256_free_states_conservative": free256,
        "n128_teacher_forced_next_conservative": teacher128,
        "n256_teacher_forced_next_conservative": teacher256,
    }


def test_one_step_residual_uses_exact_reference_current_not_free_rollout() -> None:
    arrays = _bundle()
    fields = one_step_residual_fields(arrays, "density")
    assert np.allclose(fields["truth"], 1.0)
    assert np.allclose(fields["n128"], 1.25)
    assert np.allclose(fields["n128_error"], 0.25)
    assert np.allclose(fields["n256"], 0.5)
    assert np.allclose(fields["n256_error"], -0.5)

    arrays["n128_free_states_conservative"][:] = 1.0e6
    unchanged = one_step_residual_fields(arrays, "density")
    assert np.array_equal(unchanged["n128"], fields["n128"])
    assert np.array_equal(unchanged["n128_error"], fields["n128_error"])


def test_free_pressure_fields_are_separate_accumulated_states() -> None:
    fields = free_pressure_fields(_bundle())
    assert fields["truth"].shape == (79, 3)
    assert not np.allclose(fields["n128_error"], 0.0)
    assert not np.allclose(fields["n256_error"], 0.0)
    scales = _field_scales(fields, mode="free_pressure")
    assert scales["field"][0] < scales["field"][1]
    assert scales["error"][0] < 0.0 < scales["error"][1]


def _curve(count: int) -> dict:
    steps = (1_280, 20_480, 38_400, 40_960)
    rows = []
    for index, step in enumerate(steps):
        rows.append(
            {
                "optimizer_step": step,
                "online_train_one_step_relative_l2": 0.2 / (index + 1),
                "fixed_seen_train_one_step_relative_l2": 0.3 / (index + 1),
                "fixed_validation_one_step_relative_l2": 0.4 / (index + 1),
                "rollout_h79_relative_l2": 0.8 / (index + 1),
            }
        )
    return {"rows": rows, "comparable_rows": rows, "selected_step": 38_400}


def test_side_by_side_loss_plot_writes_png_and_pdf(tmp_path: Path) -> None:
    outputs = plot_loss_curves(
        {128: _curve(128), 256: _curve(256)},
        {
            128: {8_192: 0.7, 20_480: 0.4, 38_400: 0.2, 40_960: 0.21},
            256: {16_384: 0.8, 20_480: 0.5, 38_400: 0.22, 40_960: 0.24},
        },
        tmp_path,
    )
    assert len(outputs) == 4
    assert {Path(row["path"]).suffix for row in outputs} == {".png", ".pdf"}
    assert all((tmp_path / row["path"]).stat().st_size > 0 for row in outputs)


def test_one_step_residual_rejects_nonconservative_component() -> None:
    with pytest.raises(ValueError, match="unsupported conservative component"):
        one_step_residual_fields(_bundle(), "pressure")


def test_mp4_renderer_pads_odd_frame_dimension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, animation = _configure_matplotlib()
    if not animation.writers.is_available("ffmpeg"):
        pytest.skip("ffmpeg is unavailable")
    one_frame = {
        name: np.zeros((1, 3), dtype=np.float64)
        for name in ("truth", "n128", "n128_error", "n256", "n256_error")
    }
    monkeypatch.setattr(
        visualizer,
        "one_step_residual_fields",
        lambda arrays, component: one_frame,
    )
    output = tmp_path / "tiny.mp4"
    record = render_comparison_animation(
        _bundle(),
        output,
        mode="one_step_residual",
        component="density",
        fps=20,
        dpi=20,
    )
    assert record["frames"] == 1
    assert output.stat().st_size > 0
