from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.time_dependent_no.visualize_pcno_bump_b1_c5_repeatability import (
    BF16_EXECUTIONS,
    EXPECTED_EXECUTIONS,
    HORIZONS,
    PRIMARY_ROLES,
    TRAJECTORY_COUNTS,
    horizon_crossovers,
    physical_event_counts,
    plot_horizon_profiles,
    plot_repeatability_summary,
)


def _profiles() -> dict[tuple[str, int, str], np.ndarray]:
    result = {}
    for execution_index, execution in enumerate(EXPECTED_EXECUTIONS):
        for count in TRAJECTORY_COUNTS:
            selected = np.asarray([0.02, 0.03, 0.04, 0.05], dtype=np.float64)
            selected += 1.0e-4 * execution_index + 1.0e-5 * count
            terminal = selected + np.asarray([-0.002, -0.001, 0.001, 0.003])
            result[execution, count, "selected"] = selected
            result[execution, count, "terminal"] = terminal
    return result


def _cell_rows() -> list[dict[str, str]]:
    rows = []
    for count in TRAJECTORY_COUNTS:
        for role_index, role in enumerate(PRIMARY_ROLES):
            base = 0.04 + 0.004 * (count == 256) + 0.001 * role_index
            row = {
                "trajectory_count": str(count),
                "checkpoint_role": role,
                "all_execution_h79_range": "0.0005",
                "fp32_h79": str(base + 0.0003),
            }
            for index, execution in enumerate(BF16_EXECUTIONS):
                row[f"{execution}_h79"] = str(base + index * 1.0e-4)
            rows.append(row)
    return rows


def _case_points() -> list[dict]:
    return [
        {
            "trajectory": "25",
            "n128_mean": 0.03,
            "n256_mean": 0.05,
            "stable_winner": True,
            "winner": "n128",
        },
        {
            "trajectory": "83",
            "n128_mean": 0.04,
            "n256_mean": 0.07,
            "stable_winner": True,
            "winner": "n128",
        },
        {
            "trajectory": "152",
            "n128_mean": 0.08,
            "n256_mean": 0.04,
            "stable_winner": True,
            "winner": "n256",
        },
        {
            "trajectory": "30",
            "n128_mean": 0.045,
            "n256_mean": 0.046,
            "stable_winner": False,
            "winner": "",
        },
    ]


def _event_rows() -> list[dict[str, str]]:
    rows = []
    for count in TRAJECTORY_COUNTS:
        for role in PRIMARY_ROLES:
            for index in range(28):
                row = {
                    "trajectory_count": str(count),
                    "checkpoint_role": role,
                    "event_stable": "True",
                }
                for execution in EXPECTED_EXECUTIONS:
                    row[f"{execution}_call"] = ""
                if index == 0:
                    for execution in EXPECTED_EXECUTIONS:
                        row[f"{execution}_call"] = "60"
                elif index == 1:
                    row["event_stable"] = "False"
                    row["bf16_1_call"] = "79"
                rows.append(row)
    return rows


def test_crossovers_distinguish_early_improvement_from_tail_degradation() -> None:
    profiles = _profiles()
    values = horizon_crossovers(profiles)
    for count in TRAJECTORY_COUNTS:
        assert values[str(count)]["h20_terminal_minus_selected"] < 0.0
        assert values[str(count)]["h79_terminal_minus_selected"] > 0.0


def test_event_counts_separate_stable_absence_presence_and_instability() -> None:
    counts = physical_event_counts(_event_rows())
    for count in TRAJECTORY_COUNTS:
        for role in PRIMARY_ROLES:
            assert counts[count, role] == {
                "stable_no_violation": 26,
                "stable_violation": 1,
                "precision_sensitive": 1,
            }


def test_repeatability_plots_write_pdf_and_png(tmp_path: Path) -> None:
    profiles = _profiles()
    crossovers = horizon_crossovers(profiles)
    events = physical_event_counts(_event_rows())
    analysis = {
        "selected_count_ordering": {"smallest_cross_count_h79_gap": 0.005},
    }
    outputs = plot_horizon_profiles(profiles, crossovers, tmp_path)
    outputs.extend(
        plot_repeatability_summary(
            _cell_rows(), _case_points(), events, analysis, tmp_path
        )
    )

    assert len(outputs) == 4
    assert {Path(record["path"]).suffix for record in outputs} == {".png", ".pdf"}
    assert all((tmp_path / record["path"]).stat().st_size > 0 for record in outputs)
    assert tuple(HORIZONS) == (20, 40, 60, 79)
