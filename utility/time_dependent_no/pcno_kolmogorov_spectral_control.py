"""A layer-norm restriction inspired by McCabe et al., adapted to dense PCNO.

This controls channel matrices, not the complete residual transition or graph
derivative. It does not reproduce ReFNO's separable blocks or filter reordering.
"""

import torch
from torch import nn
from torch.nn.utils import parametrize


class MatrixNormCap(nn.Module):
    def forward(self, weight):
        norm = torch.linalg.matrix_norm(weight.flatten(1), ord=2)
        return weight / norm.clamp_min(1)


class BoundedSpectralConv(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.weights_c = original.weights_c
        self.weights_s = original.weights_s
        self.weights_0 = original.weights_0

    def effective_weights(self):
        weight = torch.complex(self.weights_c, self.weights_s)
        norms = torch.linalg.matrix_norm(weight.permute(2, 3, 0, 1), ord=2)
        weight = weight / norms.clamp_min(1)[None, None]
        zero = self.weights_0
        zero_norm = torch.linalg.matrix_norm(zero.permute(2, 3, 0, 1), ord=2)
        return weight, zero / zero_norm.clamp_min(1)[None, None]

    def forward(self, x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0):
        weight, zero = self.effective_weights()
        xhat = torch.complex(torch.einsum('bix,bxkw->bikw', x, wbases_c),
                             -torch.einsum('bix,bxkw->bikw', x, wbases_s))
        yhat = torch.einsum('bikw,iokw->bokw', xhat, weight)
        mean = torch.einsum('bix,bxkw->bikw', x, wbases_0)
        mean = torch.einsum('bikw,iokw->bokw', mean, zero)
        return (torch.einsum('bokw,bxkw->box', mean, bases_0)
                + 2 * torch.einsum('bokw,bxkw->box', yhat.real, bases_c)
                - 2 * torch.einsum('bokw,bxkw->box', yhat.imag, bases_s))


def restrict_channel_maps(model):
    """Cap each dense Fourier-mode matrix and each pointwise channel matrix."""
    for i, layer in enumerate(model.pcno.sp_convs):
        model.pcno.sp_convs[i] = BoundedSpectralConv(layer)
    for layer in list(model.pcno.modules()):
        if isinstance(layer, (nn.Linear, nn.Conv1d)):
            parametrize.register_parametrization(layer, 'weight', MatrixNormCap())
    return model
