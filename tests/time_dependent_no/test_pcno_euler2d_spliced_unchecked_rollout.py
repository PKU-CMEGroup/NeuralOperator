from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.time_dependent_no.diagnose_pcno_euler2d_spliced_unchecked_rollout import (
    CONTINUATION_CALLS,
    assemble_spliced_predictions,
    exact_device_state,
)
from scripts.time_dependent_no.diagnose_pcno_euler2d_unchecked_rollout import (
    EXPECTED_FIRST_EXCLUDED_CALL,
    EXPECTED_RECURRENT_CALLS,
)


def test_splice_preserves_exact_frozen_prefix() -> None:
    prefix = np.arange(EXPECTED_FIRST_EXCLUDED_CALL * 2 * 4, dtype=np.float32).reshape(
        EXPECTED_FIRST_EXCLUDED_CALL, 2, 4
    )
    continuation = np.full((CONTINUATION_CALLS, 2, 4), -3.25, dtype=np.float32)

    result = assemble_spliced_predictions(prefix, continuation)

    assert result.shape == (EXPECTED_RECURRENT_CALLS, 2, 4)
    np.testing.assert_array_equal(result[:EXPECTED_FIRST_EXCLUDED_CALL], prefix)
    np.testing.assert_array_equal(result[EXPECTED_FIRST_EXCLUDED_CALL:], continuation)


def test_splice_rejects_wrong_continuation_length() -> None:
    prefix = np.zeros((EXPECTED_FIRST_EXCLUDED_CALL, 2, 4), dtype=np.float32)
    continuation = np.zeros((CONTINUATION_CALLS - 1, 2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="calls 34--79"):
        assemble_spliced_predictions(prefix, continuation)


def test_exact_device_state_round_trip_is_bit_exact_on_cpu() -> None:
    failed = np.asarray(
        [[1.25, -0.0, 3.5, np.nextafter(np.float32(0.0), np.float32(1.0))]],
        dtype=np.float32,
    )
    template = torch.zeros((1, 1, 4), dtype=torch.float32)

    current, metadata = exact_device_state(
        failed, template=template, device=torch.device("cpu")
    )

    np.testing.assert_array_equal(current[0].numpy(), failed)
    assert metadata["cpu_device_cpu_exact"] is True
    assert metadata["cpu_device_cpu_max_abs"] == 0.0


def test_exact_device_state_rejects_non_float32_template() -> None:
    failed = np.zeros((1, 4), dtype=np.float32)
    template = torch.zeros((1, 1, 4), dtype=torch.float64)

    with pytest.raises(ValueError, match="requires float32"):
        exact_device_state(failed, template=template, device=torch.device("cpu"))
