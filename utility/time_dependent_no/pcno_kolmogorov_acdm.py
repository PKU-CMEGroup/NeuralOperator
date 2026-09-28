"""One-state PCNO adaptation of autoregressive conditional diffusion.

ACDM's joint epsilon objective and noisy-condition replacement are retained;
the backbone and one-state/known-forcing interface are project adaptations.
Source semantics: tum-pbs/autoreg-pde-diffusion, commit 123e71b8d8f0f8cdb53b6bf29201b050c6446299,
src/turbpred/model_diffusion.py and model_diffusion_blocks.py.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.functional import one_hot

from utility.time_dependent_no.pcno_kolmogorov import (
    PeriodicVorticityPCNO, _state, canonicalize_vorticity,
)


class ACDMSchedule:
    """Twenty source-schedule DDPM steps with fixed-small posterior variance."""

    steps = 20

    def __init__(self):
        self.betas = tuple(.0025+(.5-.0025)*k/19 for k in range(self.steps))
        self.alphas_cumprod = tuple(math.prod(1-b for b in self.betas[:k+1])
                                   for k in range(self.steps))

    def factors(self, levels, reference):
        if (levels.shape != (len(reference),) or levels.dtype != torch.int64
                or levels.device != reference.device
                or torch.any((levels < 0) | (levels >= self.steps))):
            raise ValueError("levels must be aligned int64 [batch] in 0..19")
        signal = reference.new_tensor([math.sqrt(a) for a in self.alphas_cumprod])
        sigma = reference.new_tensor([math.sqrt(1-a) for a in self.alphas_cumprod])
        shape = (-1,) + (1,)*(reference.ndim-1)
        return signal[levels].reshape(shape), sigma[levels].reshape(shape)

    def add_noise(self, clean, noise, levels):
        if clean.shape != noise.shape or clean.dtype != noise.dtype or clean.device != noise.device:
            raise ValueError("clean field and noise must align")
        signal, sigma = self.factors(levels, clean)
        return signal*clean + sigma*noise

    def reconstruct(self, sample, epsilon, levels):
        signal, sigma = self.factors(levels, sample)
        return (sample-sigma*epsilon)/signal

    def step(self, sample, epsilon, level, noise):
        if type(level) is not int or not 0 <= level < self.steps:
            raise ValueError("reverse level must be an integer in 0..19")
        if sample.shape != epsilon.shape or sample.dtype != epsilon.dtype or sample.device != epsilon.device:
            raise ValueError("sample and predicted noise must align")
        if level == 0 and noise is not None:
            raise ValueError("last reverse step is deterministic")
        if level and (noise is None or noise.shape != sample.shape
                      or noise.dtype != sample.dtype or noise.device != sample.device):
            raise ValueError("intermediate step requires aligned posterior noise")
        beta, abar = self.betas[level], self.alphas_cumprod[level]
        previous = self.alphas_cumprod[level-1] if level else 1.
        mean = (sample-beta/math.sqrt(1-abar)*epsilon)/math.sqrt(1-beta)
        if level:
            mean = mean + math.sqrt(beta*(1-previous)/(1-abar))*noise
        return mean


def noise_tape(reference, generator):
    """Initial successor, fixed condition noise, then nineteen posterior fields.

    Reuse the entire tape across paired input probes. Draw a fresh tape at each
    physical step. The fixed condition noise across reverse levels follows the
    released sampler; it is never replaced by the model's conditioning output.
    """
    return tuple(torch.randn(reference.shape, dtype=reference.dtype, device="cpu",
                             generator=generator).to(reference.device) for _ in range(21))


class PeriodicVorticityACDM(PeriodicVorticityPCNO):
    def __init__(self, resolution, *, train_scale, modes=12, width=64, depth=4, fc_dim=128):
        if fc_dim <= 0:
            raise ValueError("ACDM uses the shared positive-width output head")
        super().__init__(resolution, train_scale=train_scale, modes=modes,
                         width=width, depth=depth, fc_dim=fc_dim)
        self.pcno.fc0 = nn.Linear(23, width)
        self.pcno.fc2 = nn.Linear(fc_dim, 2)
        self.pcno.in_dim, self.pcno.out_dim = 23, 2
        self.pcno.normal_params = list(self.pcno.parameters())
        self.schedule = ACDMSchedule()

    @torch.no_grad()
    def initialize_from_parent(self, parent):
        """Warm-start the body, zero new lifting columns and the joint noise head."""
        state = {k: v.detach().clone() for k, v in parent.state_dict().items()}
        weight = state["pcno.fc0.weight"]
        state["pcno.fc0.weight"] = torch.cat((weight, weight.new_zeros((len(weight), 21))), 1)
        state["pcno.fc2.weight"] = torch.zeros_like(self.pcno.fc2.weight)
        state["pcno.fc2.bias"] = torch.zeros_like(self.pcno.fc2.bias)
        self.load_state_dict(state, strict=True)

    def forward(self, joint, levels):
        if joint.ndim != 4 or joint.shape[1] != 2:
            raise ValueError("joint normalized field must have shape [batch,2,n,n]")
        n = _state(joint[:, 0])
        if (n != self.resolution or joint.dtype != self.nodes.dtype
                or joint.device != self.nodes.device or not torch.isfinite(joint).all()):
            raise ValueError("joint field must be finite and match model grid/device/dtype")
        self.schedule.factors(levels, joint)
        batch = len(joint)
        features = torch.cat((joint[:, 0].reshape(batch, n*n, 1),
            self.forcing_shape.unsqueeze(0).expand(batch, -1, -1),
            joint[:, 1].reshape(batch, n*n, 1),
            one_hot(levels, 20).to(joint.dtype)[:, None, :].expand(-1, n*n, -1)), -1)
        aux = tuple(v.unsqueeze(0).expand(batch, *v.shape) for v in (
            self.node_mask, self.nodes, self.node_weights, self.directed_edges,
            self.edge_gradient_weights))
        if self._fourier_cache is None:
            with torch.no_grad():
                self._fourier_cache = self.pcno.prepare_fourier_tensors(
                    self.nodes.unsqueeze(0), self.node_weights.unsqueeze(0))
        fourier = tuple(v.expand(batch, *v.shape[1:]) for v in self._fourier_cache)
        return self.pcno(features, aux, fourier_tensors=fourier).reshape(batch, n, n, 2).permute(0, 3, 1, 2)

    def transition(self, current, tape):
        _state(current)
        if len(tape) != 21 or any(v.shape != current.shape or v.dtype != current.dtype
                or v.device != current.device or not torch.isfinite(v).all() for v in tape):
            raise ValueError("twenty-one finite aligned noise fields are required")
        candidate, condition = tape[0], current/self.train_scale
        for level in range(19, -1, -1):
            levels = torch.full((len(current),), level, device=current.device, dtype=torch.int64)
            noised_condition = self.schedule.add_noise(condition, tape[1], levels)
            epsilon = self(torch.stack((noised_condition, candidate), 1), levels)
            candidate = self.schedule.step(candidate, epsilon[:, 1], level,
                                           tape[21-level] if level else None)
        raw = self.train_scale*candidate
        restricted = canonicalize_vorticity(raw)
        return dict(raw_next=raw, next_state=restricted, projection_residual=raw-restricted)
