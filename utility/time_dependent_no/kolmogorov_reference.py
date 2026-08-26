"""Fail-closed fixed-grid reference map for forced 2D Kolmogorov flow.

The state is real, mean-zero vorticity on a periodic square.  The reference
map is deliberately finite-dimensional: it advances the 2/3-dealiased Fourier
Galerkin state with adaptive-step RK4 for one declared macro step.  Callers
must choose explicitly between requiring a canonical state and projecting an
arbitrary array into the canonical state space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KolmogorovReferenceConfig:
    """Numerical and physical contract for one fixed-grid reference map."""

    resolution: int = 64
    domain_length: float = 2.0 * math.pi
    viscosity: float = 1.0e-3
    linear_drag: float = 0.1
    forcing_amplitude: float = 1.0
    forcing_wavenumber: int = 4
    macro_dt: float = 0.1122
    dt_max: float = 2.0e-3
    cfl: float = 0.4
    canonical_tolerance: float = 1.0e-11
    max_substeps: int = 100_000

    def validated(self) -> KolmogorovReferenceConfig:
        """Return this immutable config after checking the full contract."""

        if (
            isinstance(self.resolution, bool)
            or not isinstance(self.resolution, int)
            or self.resolution < 12
            or self.resolution % 2
        ):
            raise ValueError("resolution must be an even integer at least 12")
        if (
            isinstance(self.forcing_wavenumber, bool)
            or not isinstance(self.forcing_wavenumber, int)
            or self.forcing_wavenumber < 1
            or self.forcing_wavenumber > self.resolution // 3
        ):
            raise ValueError("forcing wavenumber must lie inside the retained band")
        finite_positive = {
            "domain_length": self.domain_length,
            "viscosity": self.viscosity,
            "macro_dt": self.macro_dt,
            "dt_max": self.dt_max,
            "cfl": self.cfl,
            "canonical_tolerance": self.canonical_tolerance,
        }
        for name, value in finite_positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.cfl > 1.0:
            raise ValueError("cfl must not exceed 1")
        if not math.isfinite(self.linear_drag) or self.linear_drag < 0.0:
            raise ValueError("linear_drag must be finite and nonnegative")
        if not math.isfinite(self.forcing_amplitude):
            raise ValueError("forcing_amplitude must be finite")
        if (
            isinstance(self.max_substeps, bool)
            or not isinstance(self.max_substeps, int)
            or self.max_substeps < 1
        ):
            raise ValueError("max_substeps must be a positive integer")
        return self


@dataclass(frozen=True)
class KolmogorovStepDiagnostics:
    """Accounting for one macro step of the reference map."""

    substeps: int
    minimum_substep: float
    maximum_substep: float
    input_projection_relative_l2: float


@dataclass(frozen=True)
class KolmogorovStep:
    """Canonical next state and numerical accounting for one macro step."""

    state: np.ndarray
    diagnostics: KolmogorovStepDiagnostics


@dataclass(frozen=True)
class VorticityDiagnostics:
    """State diagnostics with unambiguous periodic-grid normalization."""

    mean_vorticity: float
    kinetic_energy: float
    enstrophy: float
    palinstrophy: float


def resize_dealiased_vorticity(state: np.ndarray, target_resolution: int) -> np.ndarray:
    """Resize a periodic vorticity polynomial and apply the target 2/3 mask.

    Fourier coefficients are transferred by their integer mode labels.  On
    upsampling, this is exact trigonometric interpolation for the source
    polynomial.  On downsampling, modes outside the target's dealiased state
    space are deliberately discarded.
    """

    values = np.asarray(state)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("state must be a square two-dimensional array")
    source_resolution = values.shape[0]
    if source_resolution < 12 or source_resolution % 2:
        raise ValueError("source resolution must be even and at least 12")
    if (
        isinstance(target_resolution, bool)
        or not isinstance(target_resolution, int)
        or target_resolution < 12
        or target_resolution % 2
    ):
        raise ValueError("target resolution must be even and at least 12")
    if values.dtype != np.float64 or np.iscomplexobj(values):
        raise ValueError("state must be real float64")
    if not np.all(np.isfinite(values)):
        raise ValueError("state must contain only finite values")

    source_modes = np.rint(
        np.fft.fftfreq(source_resolution) * source_resolution
    ).astype(int)
    target_cutoff = target_resolution // 3
    source_hat = np.fft.fft2(values) / source_resolution**2
    target_hat = np.zeros((target_resolution, target_resolution), dtype=np.complex128)
    for source_x, mode_x in enumerate(source_modes):
        if abs(mode_x) > target_cutoff:
            continue
        target_x = mode_x % target_resolution
        for source_y, mode_y in enumerate(source_modes):
            if abs(mode_y) > target_cutoff:
                continue
            target_y = mode_y % target_resolution
            target_hat[target_x, target_y] = source_hat[source_x, source_y]
    target_hat[0, 0] = 0.0
    resized = np.fft.ifft2(target_hat * target_resolution**2).real
    return np.ascontiguousarray(resized, dtype=np.float64)


class KolmogorovReferenceStepper:
    """Pseudo-spectral RK4 transition for damped, forced 2D vorticity.

    The equation is

        omega_t + u dot grad(omega)
            = nu Laplacian(omega) - alpha omega - A q cos(q y),

    where ``q = 2 pi k / L``.  Also, ``u = (psi_y, -psi_x)`` and
    ``-Laplacian(psi) = omega``.
    Arrays use ``(x, y)`` axis order.
    """

    def __init__(self, config: KolmogorovReferenceConfig) -> None:
        self.config = config.validated()
        n = self.config.resolution
        length = self.config.domain_length
        modes = 2.0 * math.pi * np.fft.fftfreq(n, d=length / n)
        self._kx, self._ky = np.meshgrid(modes, modes, indexing="ij")
        self._k2 = self._kx**2 + self._ky**2
        self._inverse_k2 = np.zeros_like(self._k2)
        nonzero = self._k2 > 0.0
        self._inverse_k2[nonzero] = 1.0 / self._k2[nonzero]

        cutoff = (n // 3) * (2.0 * math.pi / length)
        self._dealias = (np.abs(self._kx) <= cutoff) & (np.abs(self._ky) <= cutoff)
        y = np.arange(n, dtype=np.float64) * (length / n)
        forcing_wave = (
            2.0 * math.pi * self.config.forcing_wavenumber / self.config.domain_length
        )
        forcing = -(
            self.config.forcing_amplitude
            * forcing_wave
            * np.cos(forcing_wave * y)[None, :]
        )
        forcing = np.broadcast_to(forcing, (n, n))
        self._forcing_hat = np.fft.fft2(forcing) * self._dealias
        self._forcing_hat[0, 0] = 0.0
        self._maximum_retained_k2 = float(np.max(self._k2[self._dealias]))
        self._decay = self.config.viscosity * self._k2 + self.config.linear_drag
        self._shell_index = np.floor(
            np.sqrt(self._k2) / (2.0 * math.pi / length) + 1.0e-12
        ).astype(int)

    @property
    def state_shape(self) -> tuple[int, int]:
        """Shape of one physical-space vorticity state."""

        n = self.config.resolution
        return n, n

    def _validate_array(
        self, state: np.ndarray, *, require_float64: bool
    ) -> np.ndarray:
        values = np.asarray(state)
        if values.shape != self.state_shape:
            raise ValueError(f"state must have shape {self.state_shape}")
        if np.iscomplexobj(values):
            raise ValueError("state must be real")
        if require_float64 and values.dtype != np.float64:
            raise ValueError("canonical solver states must use float64")
        values = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("state must contain only finite values")
        return values

    def canonicalize(self, state: np.ndarray) -> np.ndarray:
        """Project a finite real array to the mean-zero dealiased state space."""

        values = self._validate_array(state, require_float64=False)
        state_hat = np.fft.fft2(values)
        state_hat *= self._dealias
        state_hat[0, 0] = 0.0
        canonical = np.fft.ifft2(state_hat).real
        return np.ascontiguousarray(canonical, dtype=np.float64)

    def projection_relative_l2(self, state: np.ndarray) -> float:
        """Return the relative change made by canonicalization."""

        values = self._validate_array(state, require_float64=False)
        canonical = self.canonicalize(values)
        denominator = max(
            float(np.linalg.norm(values)),
            math.sqrt(values.size) * np.finfo(np.float64).eps,
        )
        return float(np.linalg.norm(canonical - values) / denominator)

    def _canonical_input(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        values = self._validate_array(state, require_float64=True)
        canonical = self.canonicalize(values)
        denominator = max(
            float(np.linalg.norm(values)),
            math.sqrt(values.size) * np.finfo(np.float64).eps,
        )
        relative_change = float(np.linalg.norm(canonical - values) / denominator)
        if relative_change > self.config.canonical_tolerance:
            raise ValueError(
                "state is outside the canonical mean-zero dealiased space: "
                f"relative projection change {relative_change:.6e}"
            )
        return np.ascontiguousarray(values, dtype=np.float64), relative_change

    def _velocity(self, omega_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        psi_hat = omega_hat * self._inverse_k2
        velocity_x = np.fft.ifft2(1j * self._ky * psi_hat).real
        velocity_y = np.fft.ifft2(-1j * self._kx * psi_hat).real
        return velocity_x, velocity_y

    def _rhs(self, omega_hat: np.ndarray) -> np.ndarray:
        velocity_x, velocity_y = self._velocity(omega_hat)
        omega_x = np.fft.ifft2(1j * self._kx * omega_hat).real
        omega_y = np.fft.ifft2(1j * self._ky * omega_hat).real
        advection_hat = np.fft.fft2(velocity_x * omega_x + velocity_y * omega_y)
        advection_hat *= self._dealias
        rhs = -advection_hat - self._decay * omega_hat + self._forcing_hat
        rhs *= self._dealias
        rhs[0, 0] = 0.0
        return rhs

    def _stable_substep(self, omega_hat: np.ndarray, remaining: float) -> float:
        velocity_x, velocity_y = self._velocity(omega_hat)
        maximum_x_speed = float(np.max(np.abs(velocity_x)))
        maximum_y_speed = float(np.max(np.abs(velocity_y)))
        spacing = self.config.domain_length / self.config.resolution
        advective_rate = (maximum_x_speed + maximum_y_speed) / spacing
        advective_limit = self.config.cfl / max(
            advective_rate, np.finfo(np.float64).tiny
        )
        decay_rate = (
            self.config.viscosity * self._maximum_retained_k2 + self.config.linear_drag
        )
        diffusive_limit = self.config.cfl / max(decay_rate, np.finfo(np.float64).tiny)
        return float(
            min(remaining, self.config.dt_max, advective_limit, diffusive_limit)
        )

    def _advance(
        self, canonical: np.ndarray, projection_change: float
    ) -> KolmogorovStep:
        omega_hat = np.fft.fft2(canonical)
        omega_hat *= self._dealias
        omega_hat[0, 0] = 0.0
        elapsed = 0.0
        substeps: list[float] = []
        while elapsed < self.config.macro_dt:
            if len(substeps) >= self.config.max_substeps:
                raise RuntimeError("reference step exceeded max_substeps")
            remaining = self.config.macro_dt - elapsed
            dt = self._stable_substep(omega_hat, remaining)
            if not math.isfinite(dt) or dt <= 0.0:
                raise RuntimeError("reference solver produced an invalid substep")
            k1 = self._rhs(omega_hat)
            k2 = self._rhs(omega_hat + 0.5 * dt * k1)
            k3 = self._rhs(omega_hat + 0.5 * dt * k2)
            k4 = self._rhs(omega_hat + dt * k3)
            omega_hat = omega_hat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            omega_hat *= self._dealias
            omega_hat[0, 0] = 0.0
            elapsed += dt
            if abs(elapsed - self.config.macro_dt) <= 8.0 * np.finfo(float).eps:
                elapsed = self.config.macro_dt
            substeps.append(dt)

        state = np.fft.ifft2(omega_hat).real
        if not np.all(np.isfinite(state)):
            raise RuntimeError("reference solver produced a nonfinite state")
        return KolmogorovStep(
            state=np.ascontiguousarray(state, dtype=np.float64),
            diagnostics=KolmogorovStepDiagnostics(
                substeps=len(substeps),
                minimum_substep=float(min(substeps)),
                maximum_substep=float(max(substeps)),
                input_projection_relative_l2=projection_change,
            ),
        )

    def advance_canonical(self, state: np.ndarray) -> KolmogorovStep:
        """Advance one state, failing if it is not already canonical float64."""

        canonical, projection_change = self._canonical_input(state)
        return self._advance(canonical, projection_change)

    def advance_projected(self, state: np.ndarray) -> KolmogorovStep:
        """Explicitly project an arbitrary finite real state, then advance it."""

        values = self._validate_array(state, require_float64=False)
        projection_change = self.projection_relative_l2(values)
        canonical = self.canonicalize(values)
        return self._advance(canonical, projection_change)

    def rollout_canonical(
        self, state: np.ndarray, steps: int
    ) -> tuple[np.ndarray, tuple[KolmogorovStepDiagnostics, ...]]:
        """Return the initial state and ``steps`` repeated canonical transitions."""

        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        canonical, _ = self._canonical_input(state)
        states = [canonical]
        records: list[KolmogorovStepDiagnostics] = []
        current = canonical
        for _ in range(steps):
            result = self.advance_canonical(current)
            states.append(result.state)
            records.append(result.diagnostics)
            current = result.state
        return np.stack(states, axis=0), tuple(records)

    def diagnostics_canonical(self, state: np.ndarray) -> VorticityDiagnostics:
        """Evaluate energy and gradient diagnostics on a canonical state."""

        checked, _ = self._canonical_input(state)
        canonical = self.canonicalize(checked)
        omega_hat = np.fft.fft2(canonical)
        velocity_x, velocity_y = self._velocity(omega_hat)
        omega_x = np.fft.ifft2(1j * self._kx * omega_hat).real
        omega_y = np.fft.ifft2(1j * self._ky * omega_hat).real
        return VorticityDiagnostics(
            mean_vorticity=float(np.mean(canonical)),
            kinetic_energy=float(0.5 * np.mean(velocity_x**2 + velocity_y**2)),
            enstrophy=float(0.5 * np.mean(canonical**2)),
            palinstrophy=float(0.5 * np.mean(omega_x**2 + omega_y**2)),
        )

    def kinetic_energy_spectrum_canonical(
        self, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return shell wavenumbers and kinetic energy summing to total energy."""

        checked, _ = self._canonical_input(state)
        canonical = self.canonicalize(checked)
        omega_hat = np.fft.fft2(canonical)
        psi_hat = omega_hat * self._inverse_k2
        velocity_x_hat = 1j * self._ky * psi_hat
        velocity_y_hat = -1j * self._kx * psi_hat
        mode_energy = (
            0.5
            * (np.abs(velocity_x_hat) ** 2 + np.abs(velocity_y_hat) ** 2)
            / self.config.resolution**4
        )
        shell_energy = np.bincount(
            self._shell_index.ravel(), weights=mode_energy.ravel()
        )
        fundamental = 2.0 * math.pi / self.config.domain_length
        shell_wavenumbers = fundamental * np.arange(shell_energy.size)
        return shell_wavenumbers, shell_energy

    def laminar_vorticity(self) -> np.ndarray:
        """Return the exact steady laminar state for the configured forcing."""

        n = self.config.resolution
        length = self.config.domain_length
        y = np.arange(n, dtype=np.float64) * (length / n)
        wave = (
            2.0 * math.pi * self.config.forcing_wavenumber / self.config.domain_length
        )
        denominator = self.config.viscosity * wave**2 + self.config.linear_drag
        amplitude = -(self.config.forcing_amplitude * wave) / denominator
        state = amplitude * np.cos(wave * y)[None, :]
        return self.canonicalize(np.broadcast_to(state, (n, n)))
