from __future__ import annotations

import pytest

from scripts.time_dependent_no.compare_pcno_structural_replicates import _comparison


def test_relative_comparison_is_symmetric_and_thresholded() -> None:
    discrepancy, passed = _comparison(10.0, 10.05, mode="relative", tolerance=0.01)
    assert discrepancy == pytest.approx(0.05 / 10.05)
    assert passed
    discrepancy, passed = _comparison(10.0, 10.2, mode="relative", tolerance=0.01)
    assert discrepancy == pytest.approx(0.2 / 10.2)
    assert not passed


def test_absolute_comparison_can_use_coarse_cell_scale() -> None:
    discrepancy, passed = _comparison(
        0.004,
        0.0044,
        mode="absolute",
        tolerance=0.1,
        scale=0.008,
    )
    assert discrepancy == pytest.approx(0.05)
    assert passed


def test_comparison_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unsupported comparison mode"):
        _comparison(1.0, 1.0, mode="unknown", tolerance=1.0)
