import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
import copy


class Transformer1d(nn.Module):
    def __init__(
        self,
        width,
        width_ffn,
        delta_x,
        num_encoders,
        num_heads,
        attention_type,
        decoder_type,
        num_docoderlayers,
        modes,
        bases,
        wbases,
    ):
        super(Transformer1d, self).__init__()
        self.width = width
        self.fc0 = nn.Linear(2, self.width)

        # self.delta_x = self._get_deltax(pos)
        self.delta_x = delta_x

        self.num_encoders = num_encoders
        self.num_heads = num_heads
        self.encoders = nn.ModuleList(
            [
                AttentionEncoder(
                    dim_in=self.width,
                    delta_x=self.delta_x,
                    num_heads=self.num_heads,
                    attention_type=attention_type,
                    width_ffn=width_ffn,
                    ln_eps=1e-05,
                )
                for _ in range(self.num_encoders)
            ]
        )

        self.modes = modes
        self.decoder_type = decoder_type
        if decoder_type == "Fourier":
            self.decoder = FourierDecoder(
                dim_in=self.width,
                modes1=self.modes // 2,
                num_docoderlayers=num_docoderlayers,
            )
        elif decoder_type == "Galerkin":
            self.decoder = GalerkinDecoder(
                dim_in=self.width,
                modes=self.modes,
                num_docoderlayers=num_docoderlayers,
                bases=bases,
                wbases=wbases,
            )
        else:
            self.decoder = LinearDecoder(
                dim_in=self.width, num_docoderlayers=num_docoderlayers
            )
        print("attention type:", attention_type)
        print("decoder type:", decoder_type)

    def forward(self, x):
        x = self.fc0(x)

        for layer in self.encoders:
            x = layer(x)
        x = self.decoder(x)
        return x


class AttentionEncoder(nn.Module):
    def __init__(self, dim_in, delta_x, num_heads, attention_type, width_ffn, ln_eps):
        super(AttentionEncoder, self).__init__()
        self.num_heads = num_heads
        self.d_n = dim_in
        self.d_k = self.d_n // self.num_heads

        self.linears = nn.ModuleList(
            [copy.deepcopy(nn.Linear(self.d_n, self.d_n)) for _ in range(3)]
        )
        # self._reset_parameters()

        self.delta_x = delta_x

        self.attention_type = attention_type
        if attention_type == "Fourier":
            self.layernorms_Q = nn.ModuleList(
                [nn.LayerNorm(self.d_k, eps=ln_eps) for _ in range(self.num_heads)]
            )
            self.layernorms_K = nn.ModuleList(
                [nn.LayerNorm(self.d_k, eps=ln_eps) for _ in range(self.num_heads)]
            )
        elif attention_type == "Galerkin":
            self.layernorms_K = nn.ModuleList(
                [nn.LayerNorm(self.d_k, eps=ln_eps) for _ in range(self.num_heads)]
            )
            self.layernorms_V = nn.ModuleList(
                [nn.LayerNorm(self.d_k, eps=ln_eps) for _ in range(self.num_heads)]
            )

        self.width_ffn = width_ffn
        self.ffn = nn.Sequential(
            nn.Linear(self.d_n, self.width_ffn),
            nn.SiLU(),
            nn.Linear(self.width_ffn, self.d_n),
        )

    def forward(self, x):
        batch_size = x.size(0)
        query, key, value = [
            layer(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            for layer, x in zip(self.linears, (x, x, x))
        ]

        key = torch.stack(
            [
                layernorm(x)
                for layernorm, x in zip(
                    self.layernorms_K,
                    (key[:, i, ...] for i in range(self.num_heads)),
                )
            ],
            dim=1,
        )
        if self.attention_type == "Fourier":
            query = torch.stack(
                [
                    layernorm(x)
                    for layernorm, x in zip(
                        self.layernorms_Q,
                        (query[:, i, ...] for i in range(self.num_heads)),
                    )
                ],
                dim=1,
            )
        elif self.attention_type == "Galerkin":
            value = torch.stack(
                [
                    layernorm(x)
                    for layernorm, x in zip(
                        self.layernorms_K,
                        (value[:, i, ...] for i in range(self.num_heads)),
                    )
                ],
                dim=1,
            )

        key = torch.einsum("bhxd,x->bhxd", key, self.delta_x)

        x1 = self._attention(query, key, value, self.attention_type)
        x2 = x1.transpose(1, 2).contiguous().view(batch_size, -1, self.d_n)

        x = x + x2
        x3 = self.ffn(x)
        x = x + x3

        return x

    def _attention(self, query, key, value, attention_type):
        grid_size = query.size(-2)

        if attention_type == "Fourier":
            scores = torch.matmul(query, key.transpose(-2, -1))
            p_attn = scores / grid_size
            p_attn = F.dropout(p_attn)
            out = torch.matmul(p_attn, value)
        elif attention_type == "Galerkin":
            scores = torch.matmul(key.transpose(-2, -1), value)
            p_attn = scores / grid_size
            p_attn = F.dropout(p_attn)
            out = torch.matmul(query, p_attn)

        return out

    # def _reset_parameters(self):
    #     for param in self.linears.parameters():
    #         if param.ndim > 1:
    #             xavier_uniform_(param, gain=0.01)
    #             param.data += 0.01 * torch.diag(
    #                 torch.ones(param.size(-1), dtype=torch.float)
    #             )
    #         else:
    #             constant_(param, 0)


class FourierDecoder(nn.Module):
    def __init__(self, dim_in, num_docoderlayers, modes1):
        super(FourierDecoder, self).__init__()
        self.width = dim_in // 2
        self.modes1 = modes1

        self.fourier_layers = nn.ModuleList(
            [
                FourierLayer1d(
                    dim_in=dim_in,
                    dim_out=self.width,
                    modes1=self.modes1,
                )
            ]
        )
        for _ in range(num_docoderlayers - 1):
            self.fourier_layers.append(
                FourierLayer1d(
                    dim_in=self.width,
                    dim_out=self.width,
                    modes1=self.modes1,
                )
            )

        self.ffn = nn.Sequential(
            nn.Linear(self.width, self.width), nn.SiLU(), nn.Linear(self.width, 1)
        )

    def forward(self, x):
        for layer in self.fourier_layers:
            x = layer(x)
        x = self.ffn(x)
        return x


class GalerkinDecoder(nn.Module):
    def __init__(self, dim_in, modes, num_docoderlayers, bases, wbases):
        super(GalerkinDecoder, self).__init__()
        self.width = dim_in // 2
        self.modes = modes
        self.bases = bases
        self.wbases = wbases

        self.galerkin_layers = nn.ModuleList(
            [
                GalerkinLayer1d(
                    dim_in=dim_in,
                    dim_out=self.width,
                    modes=self.modes,
                    bases=bases,
                    wbases=wbases,
                )
            ]
        )
        for _ in range(num_docoderlayers - 1):
            self.galerkin_layers.append(
                GalerkinLayer1d(
                    dim_in=self.width,
                    dim_out=self.width,
                    modes=self.modes,
                    bases=bases,
                    wbases=wbases,
                )
            )

        self.ffn = nn.Sequential(
            nn.Linear(self.width, self.width), nn.SiLU(), nn.Linear(self.width, 1)
        )

    def forward(self, x):
        for layer in self.galerkin_layers:
            x = layer(x)
        x = self.ffn(x)
        return x


class LinearDecoder(nn.Module):
    def __init__(self, dim_in, num_docoderlayers):
        super(LinearDecoder, self).__init__()
        self.width = 128

        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim_in, self.width),
                    nn.SiLU(),
                )
            ]
        )
        for _ in range(num_docoderlayers - 1):
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(self.width, self.width),
                    nn.SiLU(),
                )
            )
        self.dropout = nn.Dropout(p=0.1)
        self.out = nn.Linear(self.width, 1)

    def forward(self, x):
        for layer in self.ffn_layers:
            x = layer(x)
            x = self.dropout(x)

        x = self.out(x)
        return x


class FourierLayer1d(nn.Module):
    def __init__(self, dim_in, dim_out, modes1):
        super(FourierLayer1d, self).__init__()
        self.modes1 = modes1
        self.dim_out = dim_out

        self.weights = nn.Parameter(
            torch.rand(dim_in, dim_out, self.modes1, dtype=torch.cfloat)
        )
        xavier_normal_(self.weights, gain=1 / (dim_in * dim_out))

        self.act = nn.SiLU()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        res = self.linear(x)

        batch_size = x.shape[0]
        grid_size = x.shape[1]

        x_ft = torch.fft.rfft(x, dim=1)
        out_ft = torch.zeros(
            batch_size,
            grid_size // 2 + 1,
            self.dim_out,
            dtype=torch.cfloat,
        )
        out_ft[:, : self.modes1, :] = torch.einsum(
            "bki,iok->bko", x_ft[:, : self.modes1, :], self.weights
        )
        x = torch.fft.irfft(out_ft, n=grid_size, dim=1)

        x = self.act(x + res)
        return x


class GalerkinLayer1d(nn.Module):
    def __init__(self, dim_in, dim_out, modes, bases, wbases):
        super(GalerkinLayer1d, self).__init__()
        self.bases = bases
        self.wbases = wbases
        self.modes = modes
        self.dim_out = dim_out

        self.scale = 1 / (dim_in * dim_out)
        self.weights = nn.Parameter(
            torch.rand(dim_in, dim_out, self.modes, dtype=torch.float)
        )
        xavier_normal_(self.weights, gain=1 / (dim_in * dim_out))

        self.act = nn.SiLU()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        res = self.linear(x)
        bases, wbases = self.bases, self.wbases

        coeff = torch.einsum("bxi,xk->bki", x, wbases)
        coeff = torch.einsum("bki,iok->bko", coeff, self.weights)
        x = torch.real(torch.einsum("bko,xk->bxo", coeff, bases))

        x = self.act(x + res)
        return x
