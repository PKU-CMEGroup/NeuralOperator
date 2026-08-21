"""PlanarDet rollout diagnostics for the released cumulative-pressure field.

The paper does not release its front/cell-size implementation.  These functions
therefore define an independent, train-before-validation contract: a fixed
pressure-threshold front, fixed-pressure thickness, fixed stations, and a
shock-relative transverse spectrum.  They are diagnostics, not conservation or
an exact reproduction of the paper's structure metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from utility.time_dependent_no.realm_planardet import PLANARDET_FIELDS

PMAX_BASELINE_PA = 1.0e5
PMAX_FRONT_THRESHOLD_PA = 1.0e6
PMAX_THICKNESS_LOW_PA = 2.0e5
PMAX_THICKNESS_HIGH_PA = 2.0e6
FRONT_ACTIVE_TRANSVERSE_FRACTION = 0.9
SHOCK_STRIP_NEAR_M = 5.0e-4
SHOCK_STRIP_FAR_M = 4.0e-3
MIN_SHOCK_STRIP_COLUMNS = 16
CELL_WAVELENGTH_MIN_M = 4.0e-4
CELL_WAVELENGTH_MAX_M = 8.0e-3
HIGHPASS_WAVELENGTH_MAX_M = 8.0e-4
ARRIVAL_STATIONS_M = (0.114, 0.120, 0.126)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _json_float_array(values: np.ndarray) -> list[float | None]:
    return [_finite_or_none(float(value)) for value in values.reshape(-1)]


def _validate_axes(x: np.ndarray, y: np.ndarray, shape: tuple[int, int]) -> None:
    if x.shape != (shape[1],) or y.shape != (shape[0],):
        raise ValueError("coordinate axes do not match decoded spatial shape")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("coordinate axes must be finite")
    for axis, name in ((x, "x"), (y, "y")):
        differences = np.diff(axis)
        if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
            raise ValueError(f"{name} coordinate axis must be strictly monotone")


def _leading_positions(
    field: np.ndarray, x: np.ndarray, threshold: float
) -> np.ndarray:
    active = field > threshold
    positions = np.where(active, x[None, :], -np.inf).max(axis=1)
    positions[~np.any(active, axis=1)] = np.nan
    return positions


def _arrival_time(
    front_position: np.ndarray,
    times: np.ndarray,
    station: float,
) -> float:
    finite = np.isfinite(front_position)
    crossing = np.flatnonzero(finite & (front_position >= station))
    if crossing.size == 0:
        return math.nan
    index = int(crossing[0])
    if index == 0 or not finite[index - 1]:
        return float(times[index])
    x0 = float(front_position[index - 1])
    x1 = float(front_position[index])
    t0 = float(times[index - 1])
    t1 = float(times[index])
    if x1 <= x0:
        return t1
    fraction = min(max((station - x0) / (x1 - x0), 0.0), 1.0)
    return t0 + fraction * (t1 - t0)


@dataclass(frozen=True)
class TransverseSpectrum:
    status: str
    selected_columns: int
    dominant_mode: int | None
    dominant_wavelength_m: float
    highpass_fraction: float
    mode_numbers: np.ndarray
    normalized_power: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "selected_columns": self.selected_columns,
            "dominant_mode": self.dominant_mode,
            "dominant_wavelength_m": _finite_or_none(self.dominant_wavelength_m),
            "highpass_fraction": _finite_or_none(self.highpass_fraction),
            "mode_numbers": self.mode_numbers.astype(int).tolist(),
            "normalized_power": _json_float_array(self.normalized_power),
        }


def shock_attached_transverse_spectrum(
    pmax: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
) -> TransverseSpectrum:
    """Compute a fixed physical-band spectrum 0.5--4 mm behind the front."""

    field = np.asarray(pmax, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError("pMax frame must have shape [y, x]")
    _validate_axes(np.asarray(x), np.asarray(y), field.shape)
    if not np.isfinite(field).all():
        return TransverseSpectrum(
            "nonfinite_field",
            0,
            None,
            math.nan,
            math.nan,
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    active_fraction = np.mean(field > PMAX_FRONT_THRESHOLD_PA, axis=0)
    active_columns = active_fraction >= FRONT_ACTIVE_TRANSVERSE_FRACTION
    if not np.any(active_columns):
        return TransverseSpectrum(
            "front_absent",
            0,
            None,
            math.nan,
            math.nan,
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    front = float(np.max(x[active_columns]))
    selected = (
        active_columns
        & (x <= front - SHOCK_STRIP_NEAR_M)
        & (x >= front - SHOCK_STRIP_FAR_M)
    )
    selected_columns = int(np.count_nonzero(selected))
    if selected_columns < MIN_SHOCK_STRIP_COLUMNS:
        return TransverseSpectrum(
            "insufficient_shock_strip",
            selected_columns,
            None,
            math.nan,
            math.nan,
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    signal = np.log(np.maximum(field[:, selected], PMAX_BASELINE_PA) / PMAX_BASELINE_PA)
    signal -= signal.mean(axis=0, keepdims=True)
    power = np.mean(np.abs(np.fft.rfft(signal, axis=0)) ** 2, axis=1)
    dy = float(abs(np.median(np.diff(y))))
    frequencies = np.fft.rfftfreq(field.shape[0], d=dy)
    with np.errstate(divide="ignore"):
        wavelengths = 1.0 / frequencies
    valid = (
        (wavelengths >= CELL_WAVELENGTH_MIN_M)
        & (wavelengths <= CELL_WAVELENGTH_MAX_M)
        & np.isfinite(wavelengths)
    )
    modes = np.flatnonzero(valid)
    if modes.size == 0 or float(power[valid].sum()) <= 0.0:
        return TransverseSpectrum(
            "zero_band_energy",
            selected_columns,
            None,
            math.nan,
            math.nan,
            modes,
            np.zeros(modes.size, dtype=np.float64),
        )
    band_power = power[valid]
    normalized = band_power / band_power.sum()
    peak_offset = int(np.argmax(band_power))
    dominant_mode = int(modes[peak_offset])
    highpass = wavelengths[modes] <= HIGHPASS_WAVELENGTH_MAX_M
    return TransverseSpectrum(
        "ok",
        selected_columns,
        dominant_mode,
        float(wavelengths[dominant_mode]),
        float(normalized[highpass].sum()),
        modes,
        normalized,
    )


def planardet_structure_series(
    decoded_sequence: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
) -> dict[str, Any]:
    """Summarize one decoded sequence including its initial frame."""

    state = np.asarray(decoded_sequence)
    if state.ndim != 4 or state.shape[1] != len(PLANARDET_FIELDS):
        raise ValueError("decoded_sequence must have shape [time, 13, y, x]")
    if times.shape != (state.shape[0],) or not np.isfinite(times).all():
        raise ValueError("times must be a finite vector matching the sequence")
    _validate_axes(np.asarray(x), np.asarray(y), state.shape[2:])
    pmax = np.asarray(state[:, 12], dtype=np.float64)
    front = np.full(state.shape[0], np.nan, dtype=np.float64)
    thickness = np.full_like(front, np.nan)
    active_fraction = np.full_like(front, np.nan)
    spectra: list[TransverseSpectrum] = []
    for frame in range(state.shape[0]):
        field = pmax[frame]
        if np.isfinite(field).all():
            middle = _leading_positions(field, x, PMAX_FRONT_THRESHOLD_PA)
            low = _leading_positions(field, x, PMAX_THICKNESS_LOW_PA)
            high = _leading_positions(field, x, PMAX_THICKNESS_HIGH_PA)
            if np.isfinite(middle).any():
                front[frame] = float(np.nanmedian(middle))
            valid_thickness = np.isfinite(low) & np.isfinite(high)
            if np.any(valid_thickness):
                thickness[frame] = float(
                    np.median(
                        np.maximum(low[valid_thickness] - high[valid_thickness], 0.0)
                    )
                )
            active_fraction[frame] = float(np.mean(field > PMAX_FRONT_THRESHOLD_PA))
        spectra.append(shock_attached_transverse_spectrum(field, x=x, y=y))

    pmax_difference = np.diff(pmax, axis=0)
    finite_front = np.isfinite(front)
    speed = math.nan
    if np.count_nonzero(finite_front) >= 2:
        speed = float(np.polyfit(times[finite_front], front[finite_front], 1)[0])
    arrivals = {
        f"x_{station:.3f}_m": _finite_or_none(_arrival_time(front, times, station))
        for station in ARRIVAL_STATIONS_M
    }
    cell_sizes = np.asarray(
        [spectrum.dominant_wavelength_m for spectrum in spectra], dtype=np.float64
    )
    highpass_fractions = np.asarray(
        [spectrum.highpass_fraction for spectrum in spectra], dtype=np.float64
    )

    def finite_mean(values: np.ndarray) -> float | None:
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else None

    return {
        "contract": {
            "pMax_front_threshold_pa": PMAX_FRONT_THRESHOLD_PA,
            "pMax_thickness_low_pa": PMAX_THICKNESS_LOW_PA,
            "pMax_thickness_high_pa": PMAX_THICKNESS_HIGH_PA,
            "arrival_stations_m": list(ARRIVAL_STATIONS_M),
            "shock_strip_near_m": SHOCK_STRIP_NEAR_M,
            "shock_strip_far_m": SHOCK_STRIP_FAR_M,
            "cell_wavelength_band_m": [
                CELL_WAVELENGTH_MIN_M,
                CELL_WAVELENGTH_MAX_M,
            ],
            "highpass_wavelength_max_m": HIGHPASS_WAVELENGTH_MAX_M,
            "paper_metric_reproduction_claim": False,
        },
        "normalized_or_decoded_nonfinite_count": int(
            state.size - np.count_nonzero(np.isfinite(state))
        ),
        "pMax_nonpositive_count": int(np.count_nonzero(pmax <= 0.0)),
        "pMax_decrease_count": int(np.count_nonzero(pmax_difference < 0.0)),
        "pMax_max_decrease_pa": (
            float(max(0.0, -float(np.nanmin(pmax_difference))))
            if pmax_difference.size
            else 0.0
        ),
        "pMax_peak_pa_per_frame": _json_float_array(np.nanmax(pmax, axis=(1, 2))),
        "front_position_m": _json_float_array(front),
        "front_thickness_m": _json_float_array(thickness),
        "front_active_fraction": _json_float_array(active_fraction),
        "least_squares_front_speed_m_per_s": _finite_or_none(speed),
        "arrival_time_s": arrivals,
        "cell_size_proxy_m": [
            _finite_or_none(spectrum.dominant_wavelength_m) for spectrum in spectra
        ],
        "mean_cell_size_proxy_m": finite_mean(cell_sizes),
        "shock_attached_highpass_fraction": [
            _finite_or_none(spectrum.highpass_fraction) for spectrum in spectra
        ],
        "mean_shock_attached_highpass_fraction": finite_mean(highpass_fractions),
        "shock_attached_spectra": [spectrum.summary() for spectrum in spectra],
    }


def compare_planardet_structure(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
) -> dict[str, Any]:
    """Compare two decoded sequences under the same independent contract."""

    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth decoded sequences must share shape")
    prediction_summary = planardet_structure_series(prediction, x=x, y=y, times=times)
    truth_summary = planardet_structure_series(truth, x=x, y=y, times=times)
    prediction_pmax = np.asarray(prediction[:, 12], dtype=np.float64)
    truth_pmax = np.asarray(truth[:, 12], dtype=np.float64)
    difference = prediction_pmax - truth_pmax
    axes = (1, 2)
    numerator = np.sqrt(np.sum(np.square(difference), axis=axes))
    denominator = np.sqrt(np.sum(np.square(truth_pmax), axis=axes))
    relative_l2 = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0.0,
    )

    def numeric_series(summary: dict[str, Any], key: str) -> np.ndarray:
        return np.asarray(
            [math.nan if value is None else value for value in summary[key]],
            dtype=np.float64,
        )

    front_error = numeric_series(
        prediction_summary, "front_position_m"
    ) - numeric_series(truth_summary, "front_position_m")
    thickness_error = numeric_series(
        prediction_summary, "front_thickness_m"
    ) - numeric_series(truth_summary, "front_thickness_m")
    cell_error = numeric_series(
        prediction_summary, "cell_size_proxy_m"
    ) - numeric_series(truth_summary, "cell_size_proxy_m")
    predicted_peak = np.nanmax(prediction_pmax, axis=axes)
    truth_peak = np.nanmax(truth_pmax, axis=axes)
    peak_bias = prediction_pmax.mean(axis=axes) - truth_pmax.mean(axis=axes)
    arrival_error = {}
    for key, truth_value in truth_summary["arrival_time_s"].items():
        prediction_value = prediction_summary["arrival_time_s"][key]
        arrival_error[key] = (
            None
            if truth_value is None or prediction_value is None
            else float(prediction_value - truth_value)
        )
    predicted_mean_cell = prediction_summary["mean_cell_size_proxy_m"]
    truth_mean_cell = truth_summary["mean_cell_size_proxy_m"]
    predicted_mean_highpass = prediction_summary[
        "mean_shock_attached_highpass_fraction"
    ]
    truth_mean_highpass = truth_summary["mean_shock_attached_highpass_fraction"]
    return {
        "prediction": prediction_summary,
        "truth": truth_summary,
        "pMax_relative_l2_per_frame": _json_float_array(relative_l2),
        "pMax_peak_bias_pa_per_frame": _json_float_array(predicted_peak - truth_peak),
        "pMax_mean_bias_pa_per_frame": _json_float_array(peak_bias),
        "front_position_error_mm_per_frame": _json_float_array(front_error * 1.0e3),
        "front_thickness_error_mm_per_frame": _json_float_array(
            thickness_error * 1.0e3
        ),
        "cell_size_proxy_error_mm_per_frame": _json_float_array(cell_error * 1.0e3),
        "arrival_time_error_s": arrival_error,
        "mean_cell_size_proxy_error_mm": (
            None
            if predicted_mean_cell is None or truth_mean_cell is None
            else float((predicted_mean_cell - truth_mean_cell) * 1.0e3)
        ),
        "mean_shock_attached_highpass_fraction_error": (
            None
            if predicted_mean_highpass is None or truth_mean_highpass is None
            else float(predicted_mean_highpass - truth_mean_highpass)
        ),
        "front_speed_error_m_per_s": (
            None
            if prediction_summary["least_squares_front_speed_m_per_s"] is None
            or truth_summary["least_squares_front_speed_m_per_s"] is None
            else float(
                prediction_summary["least_squares_front_speed_m_per_s"]
                - truth_summary["least_squares_front_speed_m_per_s"]
            )
        ),
    }


__all__ = [
    "ARRIVAL_STATIONS_M",
    "CELL_WAVELENGTH_MAX_M",
    "CELL_WAVELENGTH_MIN_M",
    "FRONT_ACTIVE_TRANSVERSE_FRACTION",
    "HIGHPASS_WAVELENGTH_MAX_M",
    "MIN_SHOCK_STRIP_COLUMNS",
    "PMAX_BASELINE_PA",
    "PMAX_FRONT_THRESHOLD_PA",
    "PMAX_THICKNESS_HIGH_PA",
    "PMAX_THICKNESS_LOW_PA",
    "SHOCK_STRIP_FAR_M",
    "SHOCK_STRIP_NEAR_M",
    "TransverseSpectrum",
    "compare_planardet_structure",
    "planardet_structure_series",
    "shock_attached_transverse_spectrum",
]
