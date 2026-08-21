"""Fixed late-window SP19 correction for W26-L5.

The schedule is part of the experiment contract, not a configurable runtime
choice. The function accepts no reference state or true residual.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from utility.time_dependent_no.pcno_cross_resolution_correction import (
    NativeIncrementBasis,
    ResolutionContract,
    prepare_common_native_inputs,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    FixedCosineProjector,
)
from utility.time_dependent_no.pcno_fine_discrepancy_correction import (
    FineDiscrepancyStep,
)
from utility.time_dependent_no.pcno_resolution_transfer import Resolution
from utility.time_dependent_no.pcno_sparse_modal_correction import (
    FROZEN_ACTIVE_CELLS,
    SPARSE_POLICY,
    ModalCell,
    sparse_modal_correction,
    synchronized_sparse_modal_step,
)

HORIZON = 30
ACTIVE_START_INPUT_CALL = 15
SCHEDULED_POLICY = "sp19_late15_fine_away_half"


def _strict_input_call(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("input_call must be an integer")
    call = int(value)
    if not 0 <= call < HORIZON:
        raise ValueError(f"input_call must be in [0, {HORIZON - 1}]")
    return call


def correction_active(input_call: int) -> bool:
    """Return the frozen half-horizon schedule decision."""

    return _strict_input_call(input_call) >= ACTIVE_START_INPUT_CALL


def synchronized_scheduled_sparse_modal_step(
    native_state: np.ndarray,
    *,
    input_call: int,
    contract: ResolutionContract,
    projector: FixedCosineProjector,
    predictor: Callable[[Resolution, np.ndarray], np.ndarray],
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    active_cells: Sequence[ModalCell] = FROZEN_ACTIVE_CELLS,
) -> FineDiscrepancyStep:
    """Advance one native state with one early or two late model calls."""

    call = _strict_input_call(input_call)
    if call >= ACTIVE_START_INPUT_CALL:
        return synchronized_sparse_modal_step(
            native_state,
            contract=contract,
            projector=projector,
            predictor=predictor,
            policy=SPARSE_POLICY,
            volumes=volumes,
            component_scale=component_scale,
            active_cells=active_cells,
        )

    prepared = prepare_common_native_inputs(native_state, contract=contract)
    native_prediction = np.asarray(
        predictor(
            contract.native,
            np.array(prepared.model_inputs[contract.native], copy=True),
        ),
        dtype=np.float64,
    )
    expected_shape = (contract.native[0] * contract.native[1], 4)
    if (
        native_prediction.shape != expected_shape
        or not np.isfinite(native_prediction).all()
    ):
        raise ValueError(
            f"native prediction must be finite with shape {expected_shape}"
        )
    native_current = np.asarray(
        prepared.model_inputs[contract.native], dtype=np.float64
    )
    native_increment = native_prediction - native_current
    zero = np.zeros_like(native_increment)
    basis = NativeIncrementBasis(
        native_increment=native_increment,
        coarse_on_native=np.array(native_increment, copy=True),
        fine_on_native=np.array(native_increment, copy=True),
        native_minus_coarse=np.array(zero, copy=True),
        fine_minus_native=np.array(zero, copy=True),
    )
    correction, audit = sparse_modal_correction(
        basis,
        projector,
        policy="zero",
        volumes=volumes,
        component_scale=component_scale,
        active_cells=active_cells,
    )
    return FineDiscrepancyStep(
        prepared_inputs=prepared,
        predictions={contract.native: native_prediction},
        basis=basis,
        correction=correction,
        audit=audit,
        next_native_state=native_prediction + correction,
    )
