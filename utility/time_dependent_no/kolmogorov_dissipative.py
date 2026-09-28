"""Dissipative-prior representatives for the fixed Kolmogorov comparison."""

import torch

from utility.time_dependent_no.pcno_kolmogorov import canonicalize_vorticity

# Qualified analytic envelope: nu=.01, drag=.1, L=2*pi, vorticity forcing RMS=sqrt(8).
# The allowance concerns the sampled reference discretization, not a solver theorem.
RMS_BOUND = 25.712973861328997 * (1 + 1e-5)
SHELL_INNER = 1.1 * RMS_BOUND
SHELL_OUTER = 4.36 * SHELL_INNER
CONTRACTION = 0.5
SHELL_WEIGHT = 1e-4


def shell_samples(shape, generator):
    """Isotropic canonical directions and uniform radius (not uniform volume)."""
    white = torch.randn(shape, generator=generator, dtype=torch.float32)
    direction = canonicalize_vorticity(white)
    rms = direction.double().square().mean((-2, -1), keepdim=True).sqrt().float()
    radius = SHELL_INNER + (SHELL_OUTER - SHELL_INNER) * torch.rand(
        (shape[0], 1, 1), generator=generator)
    return direction * (radius / rms)


def shell_error(prediction, inputs):
    """Per-sample relative squared error to a synthetic contraction target."""
    error = (prediction - CONTRACTION * inputs).square().mean((-2, -1))
    return (error / inputs.square().mean((-2, -1))).mean()


class EnvelopeProjection(torch.nn.Module):
    """Project a canonical model output onto the qualified all-time RMS ball."""

    def __init__(self, model, record_calls=False):
        super().__init__()
        self.model = model
        self.record_calls = record_calls
        self.calls = []

    @property
    def train_scale(self):
        return self.model.train_scale

    def forward(self, inputs):
        output = self.model(inputs)
        proposal = output['next_state']
        if not torch.isfinite(proposal).all():
            raise FloatingPointError('nonfinite proposal before envelope projection')
        rms = proposal.double().square().mean((-2, -1), keepdim=True).sqrt()
        factor = (RMS_BOUND / rms.clamp_min(torch.finfo(torch.float64).tiny)).clamp(max=1)
        projected = proposal * factor.to(proposal.dtype)
        if self.record_calls:
            self.calls.append(dict(proposal_rms=rms.flatten().tolist(),
                                   factors=factor.flatten().tolist()))
        # raw_next retains the underlying network output for restriction diagnostics.
        return dict(output, next_state=projected)
