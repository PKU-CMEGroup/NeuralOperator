"""Four-call conditional PDE-Refiner adaptation on the fixed PCNO state space.

The released PDEArena rule uses four DDPM v-prediction levels, white Gaussian
noise and no clipping. Here the candidate is (omega_next-omega)/train_scale;
conditioning uses one Markov state, the known forcing and a four-level one-hot
code. Only the final physical state is restricted, exactly as for the baseline.
This is a PCNO adaptation, not a reproduction of the published U-Net experiment.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.functional import one_hot

from utility.time_dependent_no.pcno_kolmogorov import (
    PeriodicVorticityPCNO, _state, canonicalize_vorticity,
)


class RefinerSchedule:
    """Released K=3 exponential-beta, fixed-small DDPM reverse process.

    PDEArena calls beta_min ``min_noise_std``; it is the minimum *variance*
    in this construction. Keep this distinction explicit rather than squaring
    the released default a second time. No dependency on diffusers is needed.
    """

    def __init__(self):
        self.betas = tuple(4e-7 ** (k / 3) for k in (3, 2, 1, 0))
        self.alphas_cumprod = tuple(math.prod(1-b for b in self.betas[:k+1])
                                   for k in range(4))

    def factors(self, levels, reference):
        if (levels.shape != (reference.shape[0],) or levels.dtype != torch.int64
                or levels.device != reference.device or torch.any((levels < 0) | (levels > 3))):
            raise ValueError("levels must be aligned int64 [batch] in 0..3")
        # Form the small noise coefficient before casting: subtracting a rounded
        # FP32 alpha from one changes sigma_min by about 2%. Training and reverse
        # sampling must use the same coefficients, including at this last level.
        signal = reference.new_tensor([math.sqrt(a) for a in self.alphas_cumprod])
        sigma = reference.new_tensor([math.sqrt(1-a) for a in self.alphas_cumprod])
        return signal[levels].reshape(-1, 1, 1), sigma[levels].reshape(-1, 1, 1)

    def presentation(self, clean, noise, levels):
        if clean.shape != noise.shape or clean.dtype != noise.dtype or clean.device != noise.device:
            raise ValueError("clean residual and noise must align")
        signal, sigma = self.factors(levels, clean)
        return signal * clean + sigma * noise, signal * noise - sigma * clean

    def reconstruct(self, sample, velocity, levels):
        signal, sigma = self.factors(levels, sample)
        return signal * sample - sigma * velocity

    def step(self, sample, velocity, level, noise):
        if type(level) is not int or level not in (0, 1, 2, 3):
            raise ValueError("reverse level must be an integer in 0..3")
        if sample.shape != velocity.shape or sample.dtype != velocity.dtype or sample.device != velocity.device:
            raise ValueError("sample and velocity must align")
        if level == 0 and noise is not None:
            raise ValueError("last reverse step is deterministic")
        if level > 0 and (noise is None or noise.shape != sample.shape
                          or noise.dtype != sample.dtype or noise.device != sample.device):
            raise ValueError("stochastic reverse step requires aligned noise")
        a = self.alphas_cumprod[level]
        previous_a = self.alphas_cumprod[level-1] if level else 1.
        beta = 1-a/previous_a
        clean = math.sqrt(a)*sample - math.sqrt(1-a)*velocity
        result = math.sqrt(previous_a)*beta/(1-a)*clean
        result = result + math.sqrt(1-beta)*(1-previous_a)/(1-a)*sample
        if level:
            result = result + math.sqrt((1-previous_a)/(1-a)*beta)*noise
        return result


def noise_tape(reference, generator):
    """Four independent white fields: initial, reverse 3, reverse 2, reverse 1.

    Supply the same tape for a clean/plus/minus response triplet. Fresh tapes
    belong to different physical steps; an ensemble mean is not a deployed map.
    """
    return tuple(torch.randn(reference.shape, dtype=reference.dtype, device="cpu",
                             generator=generator).to(reference.device) for _ in range(4))


class PeriodicVorticityRefiner(PeriodicVorticityPCNO):
    def __init__(self, resolution, *, train_scale, modes=12, width=64, depth=4, fc_dim=128):
        super().__init__(resolution, train_scale=train_scale, modes=modes,
                         width=width, depth=depth, fc_dim=fc_dim)
        self.pcno.fc0 = nn.Linear(7, width)
        self.pcno.in_dim = 7
        self.pcno.normal_params = list(self.pcno.parameters())
        self.schedule = RefinerSchedule()

    @torch.no_grad()
    def initialize_from_parent(self, parent):
        """Copy its body; add zero candidate/level weights; negate the last head.

        At level 3 the initial clean residual prediction equals the parent raw
        increment. This does not imply the untrained four-call map is competent.
        """
        state = {k: v.detach().clone() for k, v in parent.state_dict().items()}
        weight = state["pcno.fc0.weight"]
        state["pcno.fc0.weight"] = torch.cat((weight, weight.new_zeros((weight.shape[0], 5))), 1)
        for name in ("pcno.fc2.weight", "pcno.fc2.bias"):
            state[name] = -state[name]
        self.load_state_dict(state, strict=True)

    def forward(self, current, candidate, levels):
        n = _state(current)
        if (n != self.resolution or current.dtype != self.nodes.dtype
                or current.device != self.nodes.device):
            raise ValueError("current state must match model grid/device/dtype")
        if (candidate.shape != current.shape or candidate.dtype != current.dtype
                or candidate.device != current.device or not torch.isfinite(candidate).all()):
            raise ValueError("candidate must be a finite aligned field")
        self.schedule.factors(levels, current)  # Validate mixed per-example levels.
        batch = len(current)
        features = torch.cat((
            (current/self.train_scale).reshape(batch, n*n, 1),
            self.forcing_shape.unsqueeze(0).expand(batch, -1, -1),
            candidate.reshape(batch, n*n, 1),
            one_hot(levels, 4).to(current.dtype)[:, None, :].expand(-1, n*n, -1)), -1)
        aux = tuple(v.unsqueeze(0).expand(batch, *v.shape) for v in (
            self.node_mask, self.nodes, self.node_weights, self.directed_edges,
            self.edge_gradient_weights))
        if self._fourier_cache is None:
            with torch.no_grad():
                self._fourier_cache = self.pcno.prepare_fourier_tensors(
                    self.nodes.unsqueeze(0), self.node_weights.unsqueeze(0))
        fourier = tuple(v.expand(batch, *v.shape[1:]) for v in self._fourier_cache)
        return self.pcno(features, aux, fourier_tensors=fourier).reshape_as(current)

    def transition(self, current, tape):
        if len(tape) != 4 or any(v.shape != current.shape or v.dtype != current.dtype
                or v.device != current.device or not torch.isfinite(v).all() for v in tape):
            raise ValueError("four finite aligned noise fields are required")
        candidate = tape[0]
        for level in (3, 2, 1, 0):
            levels = torch.full((len(current),), level, device=current.device, dtype=torch.int64)
            velocity = self(current, candidate, levels)
            candidate = self.schedule.step(candidate, velocity, level,
                                           tape[4-level] if level else None)
        raw = current + self.train_scale*candidate
        restricted = canonicalize_vorticity(raw)
        return dict(raw_next=raw, next_state=restricted, projection_residual=raw-restricted)
