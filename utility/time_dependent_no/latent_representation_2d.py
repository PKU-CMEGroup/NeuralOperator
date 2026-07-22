"""Bounded 2D latent-representation diagnostics for Line 4A.

This module owns encoders and decoders only. It intentionally contains no
latent transition, rollout correction, or data-assimilation implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from utility.time_dependent_no.pcno_euler2d import (
    Euler2DNormalization,
    conservative_admissibility,
)

LINE3_HANDOFF_SCHEMA = "line3_to_line4_physical_baseline_handoff_v1"
LINE3_HANDOFF_SHA256 = (
    "4b56baefe0f61fd635668c51e7dd71e82a783450a51131de74bcb7c7875cc9c5"
)
LINE4_REPRESENTATION_SCHEMA = "line4a_euler2d_representation_v1"
LINE4_TOKEN_NX = 25
LINE4_TOKEN_NY = 10
LINE4_TOKEN_CHANNELS = 20
LINE4_LATENT_SIZE = LINE4_TOKEN_NX * LINE4_TOKEN_NY * LINE4_TOKEN_CHANNELS


def sha256_file(path: str | Path) -> str:
    """Return the byte-level digest used by the frozen artifact contracts."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_line3_handoff(
    path: str | Path,
    *,
    expected_sha256: str = LINE3_HANDOFF_SHA256,
    expected_data_manifest_digest: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless the frozen Line-3-to-Line-4 gate is complete."""

    handoff_path = Path(path)
    digest = sha256_file(handoff_path)
    _require(digest == expected_sha256, "Line-3 handoff digest mismatch")
    payload = json.loads(handoff_path.read_text(encoding="utf-8"))
    _require(payload.get("schema") == LINE3_HANDOFF_SCHEMA, "wrong handoff schema")
    _require(payload.get("status") == "frozen", "Line-3 handoff is not frozen")
    _require(
        payload.get("line4_training_truth_authorized") is True,
        "Line-4 training truth is not authorized",
    )
    _require(
        payload.get("line4_transition_training_authorized") is False,
        "representation audit must not consume a transition authorization",
    )
    _require(
        payload.get("line4_front_candidate_available") is False,
        "this pilot must not silently consume an unvalidated front candidate",
    )

    dependencies = payload.get("dependencies", {})
    _require(
        dependencies.get("line4_reconstruction_and_closure_tests_may_reopen") is True,
        "reconstruction and closure tests were not released",
    )
    _require(
        dependencies.get(
            "line4_transition_training_still_requires_representation_gates"
        )
        is True,
        "transition dependency gate is missing",
    )
    _require(
        dependencies.get("line4_assimilation_remains_blocked_by_raw_open_loop_gate")
        is True,
        "assimilation dependency gate is missing",
    )

    truth = payload.get("training_truth", {})
    manifest_digest = str(truth.get("data_manifest_digest", ""))
    _require(bool(manifest_digest), "training-truth manifest digest is missing")
    if expected_data_manifest_digest is not None:
        _require(
            manifest_digest == expected_data_manifest_digest,
            "training-truth manifest digest mismatch",
        )
    _require(
        truth.get("state_convention") == "conservative_[rho,rho_u,rho_v,E]",
        "unexpected state convention",
    )
    resolution = truth.get("resolution_contract", {})
    _require(resolution.get("stored_grid") == [250, 100], "wrong stored grid")
    _require(resolution.get("node_counts") == [25000], "wrong stored node count")
    _require(
        truth.get("weight_provenance") == "validated_physical_cell_volume_normalized",
        "physical volume weights are not authorized",
    )

    baseline = payload.get("physical_baseline", {})
    _require(baseline.get("split") == "validation", "baseline is not validation-only")
    _require(baseline.get("calls") == 60, "baseline is not the H60 contract")
    _require(baseline.get("raw_recurrence") is True, "baseline recurrence is not raw")
    _require(
        baseline.get("boundary_mode") == "model_all_nodes",
        "baseline boundary contract mismatch",
    )
    interventions = baseline.get("inference_interventions", {})
    _require(bool(interventions), "baseline intervention ledger is missing")
    _require(
        all(value is False for value in interventions.values()),
        "baseline contains an inference intervention",
    )
    _require(bool(baseline.get("aggregates")), "baseline aggregates are missing")
    _require(
        len(baseline.get("front_and_smooth_hierarchy", [])) == 24,
        "front/smooth hierarchy is incomplete",
    )
    _require(
        len(baseline.get("grouped_parameter_ood", [])) == 8,
        "grouped validation summary is incomplete",
    )
    _require(bool(baseline.get("cost")), "baseline cost contract is missing")
    reference_contract = baseline.get("reference_contract", {})
    _require(
        len(reference_contract.get("reference_checks", [])) == 24,
        "reference checks are incomplete",
    )

    return {
        "schema": LINE3_HANDOFF_SCHEMA,
        "sha256": digest,
        "data_manifest_digest": manifest_digest,
        "normalization_digest": str(truth.get("normalization_digest", "")),
        "grouped_split_digest": str(truth.get("grouped_split_digest", "")),
        "training_truth_authorized": True,
        "front_candidate_available": False,
        "transition_training_authorized": False,
        "assimilation_authorized": False,
    }


@dataclass(frozen=True)
class TokenGeometry:
    """Resolution-independent physical tokenization of a batched point cloud."""

    nodes: torch.Tensor
    volumes: torch.Tensor
    token_ids: torch.Tensor
    token_volumes: torch.Tensor
    global_coordinates: torch.Tensor
    local_coordinates: torch.Tensor
    log_relative_volume: torch.Tensor
    token_nx: int
    token_ny: int
    domain: tuple[float, float, float, float]

    @property
    def num_tokens(self) -> int:
        return self.token_nx * self.token_ny


def _batched_nodes(nodes: torch.Tensor) -> torch.Tensor:
    if nodes.ndim == 2:
        nodes = nodes.unsqueeze(0)
    if nodes.ndim != 3 or nodes.shape[-1] != 2:
        raise ValueError("nodes must have shape [N,2] or [B,N,2]")
    if not torch.isfinite(nodes).all():
        raise ValueError("nodes must be finite")
    return nodes


def _batched_volumes(
    node_volumes: torch.Tensor,
    *,
    batch_size: int,
    num_nodes: int,
) -> torch.Tensor:
    volumes = node_volumes
    while volumes.ndim > 2 and volumes.shape[-1] == 1:
        volumes = volumes.squeeze(-1)
    if volumes.ndim == 2 and volumes.shape == (num_nodes, 1):
        volumes = volumes[:, 0]
    if volumes.ndim == 1:
        volumes = volumes.unsqueeze(0)
    if volumes.ndim != 2 or volumes.shape[-1] != num_nodes:
        raise ValueError("node_volumes must provide one scalar per node")
    if volumes.shape[0] == 1 and batch_size > 1:
        volumes = volumes.expand(batch_size, -1)
    if volumes.shape[0] != batch_size:
        raise ValueError("node and volume batch dimensions disagree")
    if not torch.isfinite(volumes).all() or not torch.all(volumes > 0.0):
        raise ValueError("node volumes must be finite and positive")
    return volumes


def build_token_geometry(
    nodes: torch.Tensor,
    node_volumes: torch.Tensor,
    *,
    token_nx: int = LINE4_TOKEN_NX,
    token_ny: int = LINE4_TOKEN_NY,
    domain: Sequence[float] = (0.0, 2.0, 0.0, 1.0),
) -> TokenGeometry:
    """Assign nodes to a fixed physical lattice, independent of mesh resolution."""

    if token_nx < 1 or token_ny < 1:
        raise ValueError("token lattice dimensions must be positive")
    if len(domain) != 4:
        raise ValueError("domain must contain xmin, xmax, ymin, ymax")
    xmin, xmax, ymin, ymax = (float(value) for value in domain)
    if not (xmin < xmax and ymin < ymax):
        raise ValueError("domain bounds must be ordered")

    batched_nodes = _batched_nodes(nodes)
    batch_size, num_nodes, _ = batched_nodes.shape
    volumes = _batched_volumes(
        node_volumes,
        batch_size=batch_size,
        num_nodes=num_nodes,
    ).to(device=batched_nodes.device, dtype=batched_nodes.dtype)

    unit_x = (batched_nodes[..., 0] - xmin) / (xmax - xmin)
    unit_y = (batched_nodes[..., 1] - ymin) / (ymax - ymin)
    tolerance = 32.0 * torch.finfo(batched_nodes.dtype).eps
    in_domain = (
        (unit_x >= -tolerance)
        & (unit_x <= 1.0 + tolerance)
        & (unit_y >= -tolerance)
        & (unit_y <= 1.0 + tolerance)
    )
    if not bool(torch.all(in_domain)):
        raise ValueError("nodes fall outside the declared physical domain")
    unit_x = unit_x.clamp(0.0, 1.0)
    unit_y = unit_y.clamp(0.0, 1.0)
    index_x = torch.floor(unit_x * token_nx).to(torch.long).clamp(max=token_nx - 1)
    index_y = torch.floor(unit_y * token_ny).to(torch.long).clamp(max=token_ny - 1)
    token_ids = index_y * token_nx + index_x
    num_tokens = token_nx * token_ny

    token_volumes = volumes.new_zeros((batch_size, num_tokens))
    token_volumes.scatter_add_(1, token_ids, volumes)
    token_counts = volumes.new_zeros((batch_size, num_tokens))
    token_counts.scatter_add_(1, token_ids, torch.ones_like(volumes))
    if not bool(torch.all(token_volumes > 0.0)):
        raise ValueError("every physical token must contain positive volume")

    center_x = (index_x.to(batched_nodes.dtype) + 0.5) / token_nx
    center_y = (index_y.to(batched_nodes.dtype) + 0.5) / token_ny
    local_coordinates = torch.stack(
        (
            2.0 * token_nx * (unit_x - center_x),
            2.0 * token_ny * (unit_y - center_y),
        ),
        dim=-1,
    )
    global_coordinates = torch.stack((2.0 * unit_x - 1.0, 2.0 * unit_y - 1.0), dim=-1)
    gathered_token_volume = torch.gather(token_volumes, 1, token_ids)
    gathered_token_count = torch.gather(token_counts, 1, token_ids)
    relative_volume = volumes * gathered_token_count / gathered_token_volume

    return TokenGeometry(
        nodes=batched_nodes,
        volumes=volumes,
        token_ids=token_ids,
        token_volumes=token_volumes,
        global_coordinates=global_coordinates,
        local_coordinates=local_coordinates,
        log_relative_volume=relative_volume.log().unsqueeze(-1),
        token_nx=token_nx,
        token_ny=token_ny,
        domain=(xmin, xmax, ymin, ymax),
    )


def repeat_token_geometry(geometry: TokenGeometry, batch_size: int) -> TokenGeometry:
    """Broadcast one fixed geometry across a homogeneous state batch."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    current = geometry.nodes.shape[0]
    if current == batch_size:
        return geometry
    if current != 1:
        raise ValueError("only a singleton geometry batch can be broadcast")

    def expanded(value: torch.Tensor) -> torch.Tensor:
        return value.expand(batch_size, *value.shape[1:])

    return TokenGeometry(
        nodes=expanded(geometry.nodes),
        volumes=expanded(geometry.volumes),
        token_ids=expanded(geometry.token_ids),
        token_volumes=expanded(geometry.token_volumes),
        global_coordinates=expanded(geometry.global_coordinates),
        local_coordinates=expanded(geometry.local_coordinates),
        log_relative_volume=expanded(geometry.log_relative_volume),
        token_nx=geometry.token_nx,
        token_ny=geometry.token_ny,
        domain=geometry.domain,
    )


def token_weighted_mean(values: torch.Tensor, geometry: TokenGeometry) -> torch.Tensor:
    """Compute physical-volume means for each spatial token."""

    if values.ndim == 2:
        values = values.unsqueeze(0)
    if values.ndim != 3:
        raise ValueError("values must have shape [N,C] or [B,N,C]")
    if values.shape[:2] != geometry.nodes.shape[:2]:
        raise ValueError("value and geometry shapes disagree")
    if not torch.isfinite(values).all():
        raise ValueError("token aggregation received nonfinite values")
    batch_size, _, channels = values.shape
    token_sums = values.new_zeros((batch_size, geometry.num_tokens, channels))
    token_index = geometry.token_ids.unsqueeze(-1).expand(-1, -1, channels)
    token_sums.scatter_add_(
        1,
        token_index,
        values * geometry.volumes.unsqueeze(-1),
    )
    return token_sums / geometry.token_volumes.unsqueeze(-1)


def gather_token_values(
    token_values: torch.Tensor,
    geometry: TokenGeometry,
) -> torch.Tensor:
    """Gather a token field back to the nodes without smoothing between tokens."""

    if token_values.ndim != 3:
        raise ValueError("token_values must have shape [B,T,C]")
    if token_values.shape[:2] != (
        geometry.nodes.shape[0],
        geometry.num_tokens,
    ):
        raise ValueError("token value and geometry shapes disagree")
    channels = token_values.shape[-1]
    index = geometry.token_ids.unsqueeze(-1).expand(-1, -1, channels)
    return torch.gather(token_values, 1, index)


class SpatialTokenAutoencoder(nn.Module):
    """Matched generic or conservative-moment spatial-token autoencoder."""

    _VARIANTS = {"generic", "conservative_moment"}

    def __init__(
        self,
        normalization: Euler2DNormalization | Mapping[str, Any],
        *,
        variant: str,
        token_nx: int = LINE4_TOKEN_NX,
        token_ny: int = LINE4_TOKEN_NY,
        token_channels: int = LINE4_TOKEN_CHANNELS,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        if variant not in self._VARIANTS:
            raise ValueError(f"variant must be one of {sorted(self._VARIANTS)}")
        if token_channels < 4:
            raise ValueError("token_channels must leave four conservative coordinates")
        if token_nx < 1 or token_ny < 1 or hidden_channels < 1:
            raise ValueError("lattice and hidden dimensions must be positive")
        if not isinstance(normalization, Euler2DNormalization):
            normalization = Euler2DNormalization.from_mapping(normalization)

        self.variant = variant
        self.token_nx = int(token_nx)
        self.token_ny = int(token_ny)
        self.token_channels = int(token_channels)
        self.hidden_channels = int(hidden_channels)
        self.gamma = float(normalization.gamma)
        self.register_buffer(
            "state_mean",
            torch.as_tensor(normalization.state_mean, dtype=torch.float32).view(
                1, 1, 4
            ),
        )
        self.register_buffer(
            "state_scale",
            torch.as_tensor(normalization.state_scale, dtype=torch.float32).view(
                1, 1, 4
            ),
        )

        self.cell_encoder = nn.Sequential(
            nn.Linear(9, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, token_channels),
        )
        self.encoder_mixer = nn.Sequential(
            nn.Conv2d(token_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, token_channels, kernel_size=3, padding=1),
        )
        self.decoder_mixer = nn.Sequential(
            nn.Conv2d(token_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, token_channels, kernel_size=3, padding=1),
        )
        self.point_decoder = nn.Sequential(
            nn.Linear(token_channels + 5, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 4),
        )
        final_layer = self.point_decoder[-1]
        if not isinstance(final_layer, nn.Linear):
            raise AssertionError("point decoder final layer contract changed")
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    @property
    def latent_size(self) -> int:
        return self.token_nx * self.token_ny * self.token_channels

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate_geometry(self, geometry: TokenGeometry) -> None:
        if (geometry.token_nx, geometry.token_ny) != (self.token_nx, self.token_ny):
            raise ValueError("model and physical token lattices disagree")

    def _to_grid(self, token_values: torch.Tensor) -> torch.Tensor:
        batch_size = token_values.shape[0]
        return token_values.view(
            batch_size,
            self.token_ny,
            self.token_nx,
            self.token_channels,
        ).permute(0, 3, 1, 2)

    @staticmethod
    def _from_grid(grid: torch.Tensor) -> torch.Tensor:
        return (
            grid.permute(0, 2, 3, 1)
            .contiguous()
            .view(
                grid.shape[0],
                grid.shape[2] * grid.shape[3],
                grid.shape[1],
            )
        )

    def encode(
        self,
        conservative: torch.Tensor,
        geometry: TokenGeometry,
    ) -> torch.Tensor:
        """Encode once; no decode-reencode projection is performed."""

        self._validate_geometry(geometry)
        if conservative.ndim == 2:
            conservative = conservative.unsqueeze(0)
        if conservative.shape != (*geometry.nodes.shape[:2], 4):
            raise ValueError("conservative state and geometry shapes disagree")
        if not torch.isfinite(conservative).all():
            raise ValueError("encoder received nonfinite state")
        normalized = (conservative - self.state_mean) / self.state_scale
        cell_features = torch.cat(
            (
                normalized,
                geometry.global_coordinates,
                geometry.local_coordinates,
                geometry.log_relative_volume,
            ),
            dim=-1,
        )
        pooled = token_weighted_mean(self.cell_encoder(cell_features), geometry)
        learned = pooled + self._from_grid(self.encoder_mixer(self._to_grid(pooled)))
        if self.variant == "generic":
            return learned
        moments = token_weighted_mean(conservative, geometry)
        normalized_moments = (moments - self.state_mean) / self.state_scale
        return torch.cat((normalized_moments, learned[..., 4:]), dim=-1)

    def decode(self, code: torch.Tensor, geometry: TokenGeometry) -> torch.Tensor:
        """Decode raw values; inadmissibility remains visible to the caller."""

        self._validate_geometry(geometry)
        expected_shape = (
            geometry.nodes.shape[0],
            geometry.num_tokens,
            self.token_channels,
        )
        if code.shape != expected_shape:
            raise ValueError(f"code must have shape {expected_shape}")
        if not torch.isfinite(code).all():
            raise ValueError("decoder received nonfinite code")
        token_context = code + self._from_grid(self.decoder_mixer(self._to_grid(code)))
        point_context = gather_token_values(token_context, geometry)
        point_features = torch.cat(
            (
                point_context,
                geometry.global_coordinates,
                geometry.local_coordinates,
                geometry.log_relative_volume,
            ),
            dim=-1,
        )
        decoded_standard = self.point_decoder(point_features)
        if self.variant == "generic":
            return self.state_mean + self.state_scale * decoded_standard

        coarse_standard = gather_token_values(code[..., :4], geometry)
        coarse = self.state_mean + self.state_scale * coarse_standard
        residual = self.state_scale * decoded_standard
        residual = residual - gather_token_values(
            token_weighted_mean(residual, geometry),
            geometry,
        )
        return coarse + residual

    def forward(
        self,
        conservative: torch.Tensor,
        geometry: TokenGeometry,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        code = self.encode(conservative, geometry)
        return self.decode(code, geometry), code

    def contract(self) -> dict[str, Any]:
        fixed_channels = 4 if self.variant == "conservative_moment" else 0
        return {
            "schema": LINE4_REPRESENTATION_SCHEMA,
            "variant": self.variant,
            "spatially_indexed": True,
            "token_lattice": [self.token_nx, self.token_ny],
            "token_channels": self.token_channels,
            "latent_size": self.latent_size,
            "fixed_conservative_channels": fixed_channels,
            "learned_channels": self.token_channels - fixed_channels,
            "front_variables": 0,
            "decoder_constraint": (
                "exact_token_conservative_mean"
                if self.variant == "conservative_moment"
                else "none"
            ),
            "resolution_contract": "fixed_physical_tokens_variable_query_nodes",
            "raw_decode": True,
            "clipping": False,
            "floors": False,
            "limiter": False,
            "decode_reencode_projection": False,
            "parameter_count": self.parameter_count,
        }


def scaled_volume_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    geometry: TokenGeometry,
    state_scale: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    """Physical-volume mean-square error after fixed component scaling."""

    if prediction.shape != target.shape or prediction.shape != (
        *geometry.nodes.shape[:2],
        4,
    ):
        raise ValueError("prediction, target, and geometry shapes disagree")
    scale = torch.as_tensor(
        state_scale, dtype=prediction.dtype, device=prediction.device
    )
    if scale.numel() != 4 or not torch.all(scale > 0.0):
        raise ValueError("state_scale must contain four positive entries")
    scaled_square = ((prediction - target) / scale.view(1, 1, 4)).square()
    weighted_sum = (scaled_square * geometry.volumes.unsqueeze(-1)).sum()
    return weighted_sum / (4.0 * geometry.volumes.sum())


def fit_frozen_decoder_code(
    model: SpatialTokenAutoencoder,
    target: torch.Tensor,
    geometry: TokenGeometry,
    initial_code: torch.Tensor,
    *,
    fixed_channels: int = 4,
    max_iter: int = 250,
    max_eval: int = 320,
    history_size: int = 20,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit one state's free code against a frozen raw decoder.

    This is a representation-capacity diagnostic, not an encoder, transition,
    rollout, or analysis algorithm. The returned code is obtained by declared
    per-state fitting and must never be counted as autonomous forecast evidence.
    """

    if model.variant != "conservative_moment":
        raise ValueError("code reachability is defined for the conservative model")
    if target.shape[0] != 1 or initial_code.shape[0] != 1:
        raise ValueError("code reachability fits exactly one state at a time")
    if target.shape != (*geometry.nodes.shape[:2], 4):
        raise ValueError("target and geometry shapes disagree")
    if initial_code.shape != (
        1,
        geometry.num_tokens,
        model.token_channels,
    ):
        raise ValueError("initial code and model shapes disagree")
    if fixed_channels != 4 or fixed_channels >= model.token_channels:
        raise ValueError("the four conservative moment channels must stay fixed")
    if max_iter < 1 or max_eval < max_iter or history_size < 1:
        raise ValueError("invalid fixed L-BFGS contract")
    if not torch.isfinite(target).all() or not torch.isfinite(initial_code).all():
        raise ValueError("code reachability received nonfinite inputs")

    parameters = list(model.parameters())
    original_requires_grad = [parameter.requires_grad for parameter in parameters]
    original_training = model.training
    fixed = initial_code[..., :fixed_channels].detach().clone()
    free = nn.Parameter(initial_code[..., fixed_channels:].detach().clone())
    trace: list[float] = []
    optimizer = torch.optim.LBFGS(
        [free],
        lr=1.0,
        max_iter=max_iter,
        max_eval=max_eval,
        tolerance_grad=1.0e-7,
        tolerance_change=1.0e-9,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )

    def assembled_code() -> torch.Tensor:
        return torch.cat((fixed, free), dim=-1)

    try:
        model.eval()
        for parameter in parameters:
            parameter.requires_grad_(False)
        with torch.no_grad():
            initial_loss = scaled_volume_mse(
                model.decode(initial_code, geometry),
                target,
                geometry,
                model.state_scale,
            )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            prediction = model.decode(assembled_code(), geometry)
            loss = scaled_volume_mse(
                prediction,
                target,
                geometry,
                model.state_scale,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite frozen-decoder code-fit loss")
            loss.backward()
            trace.append(float(loss.detach().cpu()))
            return loss

        optimizer.step(closure)
        fitted_code = assembled_code().detach()
        with torch.no_grad():
            final_loss = scaled_volume_mse(
                model.decode(fitted_code, geometry),
                target,
                geometry,
                model.state_scale,
            )
    finally:
        for parameter, requires_grad in zip(
            parameters,
            original_requires_grad,
            strict=True,
        ):
            parameter.requires_grad_(requires_grad)
        model.train(original_training)

    if not torch.equal(fitted_code[..., :fixed_channels], fixed):
        raise AssertionError("fixed conservative channels changed during code fitting")
    initial_value = float(initial_loss.detach().cpu())
    final_value = float(final_loss.detach().cpu())
    return fitted_code, {
        "optimizer": "lbfgs_strong_wolfe",
        "learning_rate": 1.0,
        "max_iter": max_iter,
        "max_eval": max_eval,
        "history_size": history_size,
        "tolerance_grad": 1.0e-7,
        "tolerance_change": 1.0e-9,
        "fixed_channels": fixed_channels,
        "free_channels": model.token_channels - fixed_channels,
        "closure_evaluations": len(trace),
        "initial_scaled_volume_mse": initial_value,
        "final_scaled_volume_mse": final_value,
        "loss_ratio": final_value / max(initial_value, torch.finfo(target.dtype).tiny),
        "loss_trace": trace,
        "per_state_fitting": True,
        "forecast_evidence": False,
    }


def volume_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    geometry: TokenGeometry,
) -> torch.Tensor:
    """Physical-volume relative L2 over conservative components."""

    if prediction.shape != target.shape or prediction.shape != (
        *geometry.nodes.shape[:2],
        4,
    ):
        raise ValueError("prediction, target, and geometry shapes disagree")
    weights = geometry.volumes.unsqueeze(-1)
    numerator = ((prediction - target).square() * weights).sum()
    denominator = (target.square() * weights).sum()
    return torch.sqrt(numerator / denominator.clamp_min(torch.finfo(target.dtype).tiny))


def reconstruction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    geometry: TokenGeometry,
    *,
    state_scale: torch.Tensor | Sequence[float],
    gamma: float = 1.4,
) -> dict[str, Any]:
    """Report fidelity and admissibility without repairing the decoded state."""

    admissibility = conservative_admissibility(prediction, gamma=gamma)
    predicted_moments = token_weighted_mean(prediction, geometry)
    target_moments = token_weighted_mean(target, geometry)
    moment_error = predicted_moments - target_moments
    token_weights = geometry.token_volumes.unsqueeze(-1)
    moment_numerator = (moment_error.square() * token_weights).sum()
    moment_denominator = (target_moments.square() * token_weights).sum()
    global_prediction = (prediction * geometry.volumes.unsqueeze(-1)).sum(dim=1)
    global_target = (target * geometry.volumes.unsqueeze(-1)).sum(dim=1)
    global_relative = torch.linalg.vector_norm(
        global_prediction - global_target, dim=-1
    )
    global_relative /= torch.linalg.vector_norm(global_target, dim=-1).clamp_min(
        torch.finfo(target.dtype).tiny
    )
    return {
        "scaled_volume_mse": float(
            scaled_volume_mse(prediction, target, geometry, state_scale).detach().cpu()
        ),
        "relative_l2": float(
            volume_relative_l2(prediction, target, geometry).detach().cpu()
        ),
        "token_moment_relative_l2": float(
            torch.sqrt(moment_numerator / moment_denominator.clamp_min(1.0e-30))
            .detach()
            .cpu()
        ),
        "token_moment_max_abs": float(moment_error.abs().max().detach().cpu()),
        "global_budget_relative_l2_mean": float(global_relative.mean().detach().cpu()),
        "admissible_fraction": float(
            admissibility["admissible"].to(torch.float64).mean().detach().cpu()
        ),
        "minimum_density": float(admissibility["density"].min().detach().cpu()),
        "minimum_internal_energy": float(
            admissibility["internal_energy"].min().detach().cpu()
        ),
        "minimum_pressure": float(admissibility["pressure"].min().detach().cpu()),
        "intervention_applied": False,
    }


@dataclass(frozen=True)
class ChannelWhitener:
    """A shared channel gauge; it never mixes physical token locations."""

    mean: torch.Tensor
    inverse_square_root: torch.Tensor
    ridge: float

    def transform(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 3 or codes.shape[-1] != self.mean.numel():
            raise ValueError("codes and whitener channel dimensions disagree")
        centered = codes - self.mean.to(device=codes.device, dtype=codes.dtype)
        matrix = self.inverse_square_root.to(device=codes.device, dtype=codes.dtype)
        return torch.matmul(centered, matrix)


def fit_channel_whitener(
    codes: torch.Tensor,
    *,
    ridge: float = 1.0e-6,
) -> ChannelWhitener:
    """Fit a C-by-C channel whitener rather than a 5000-by-5000 code gauge."""

    if codes.ndim != 3 or codes.shape[0] < 2:
        raise ValueError("codes must have shape [S,T,C] with at least two samples")
    if ridge <= 0.0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and positive")
    flat = codes.detach().to(dtype=torch.float64).reshape(-1, codes.shape[-1])
    if not torch.isfinite(flat).all():
        raise ValueError("whitener received nonfinite codes")
    mean = flat.mean(dim=0)
    centered = flat - mean
    covariance = centered.T @ centered / max(flat.shape[0] - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    scale = eigenvalues.max().clamp_min(torch.finfo(eigenvalues.dtype).eps)
    floored = eigenvalues.clamp_min(ridge * scale)
    inverse_square_root = (eigenvectors * floored.rsqrt().unsqueeze(0)) @ eigenvectors.T
    return ChannelWhitener(
        mean=mean,
        inverse_square_root=inverse_square_root,
        ridge=float(ridge),
    )


def _numeric_group_ids(
    groups: Sequence[str | int] | None,
    sample_count: int,
    device: torch.device,
) -> torch.Tensor:
    if groups is None:
        return torch.arange(sample_count, dtype=torch.long, device=device)
    if len(groups) != sample_count:
        raise ValueError("one exclusion group is required per sample")
    lookup: dict[str | int, int] = {}
    identifiers: list[int] = []
    for group in groups:
        if group not in lookup:
            lookup[group] = len(lookup)
        identifiers.append(lookup[group])
    return torch.as_tensor(identifiers, dtype=torch.long, device=device)


def _neighbor_ambiguity(
    conditioning: torch.Tensor,
    future: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    neighbors: int,
    chunk_size: int,
) -> dict[str, Any]:
    sample_count, conditioning_dimension = conditioning.shape
    if len(torch.unique(group_ids)) <= neighbors:
        raise ValueError("not enough independent groups for the requested neighbors")
    future_errors: list[torch.Tensor] = []
    current_distances: list[torch.Tensor] = []
    neighbor_indices: list[torch.Tensor] = []
    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        distance = torch.cdist(conditioning[start:stop], conditioning)
        blocked = group_ids[start:stop, None] == group_ids[None, :]
        distance = distance.masked_fill(blocked, torch.inf)
        selected_distance, selected_index = torch.topk(
            distance,
            k=neighbors,
            dim=1,
            largest=False,
            sorted=True,
        )
        if not torch.isfinite(selected_distance).all():
            raise ValueError("independent nearest neighbors could not be selected")
        selected_future = future[selected_index]
        target_future = future[start:stop, None, :]
        future_error = torch.sqrt(
            (selected_future - target_future).square().mean(dim=-1)
        )
        future_errors.append(future_error)
        current_distances.append(selected_distance / np.sqrt(conditioning_dimension))
        neighbor_indices.append(selected_index)
    future_rms = torch.cat(future_errors, dim=0)
    current_rms = torch.cat(current_distances, dim=0)
    indices = torch.cat(neighbor_indices, dim=0)
    return {
        "neighbors": int(neighbors),
        "sample_count": int(sample_count),
        "current_neighbor_rms_mean": float(current_rms.mean().cpu()),
        "future_neighbor_rms_mean": float(future_rms.mean().cpu()),
        "future_neighbor_rms_median": float(future_rms.median().cpu()),
        "future_neighbor_rms_p90": float(torch.quantile(future_rms, 0.9).cpu()),
        "neighbor_indices": indices.cpu().tolist(),
    }


def conditional_future_diagnostics(
    current_codes: torch.Tensor,
    future_codes: torch.Tensor,
    *,
    whitener: ChannelWhitener,
    exclusion_groups: Sequence[str | int] | None = None,
    previous_codes: torch.Tensor | None = None,
    neighbors: int = 4,
    chunk_size: int = 64,
) -> dict[str, Any]:
    """Measure encoded-future ambiguity, with one bounded history comparison."""

    if current_codes.shape != future_codes.shape or current_codes.ndim != 3:
        raise ValueError("current and future codes must share shape [S,T,C]")
    if neighbors < 1 or chunk_size < 1:
        raise ValueError("neighbors and chunk_size must be positive")
    sample_count = current_codes.shape[0]
    whitened_current = whitener.transform(current_codes).reshape(sample_count, -1)
    whitened_future = whitener.transform(future_codes).reshape(sample_count, -1)
    group_ids = _numeric_group_ids(
        exclusion_groups,
        sample_count,
        whitened_current.device,
    )
    result: dict[str, Any] = {
        "current_only": _neighbor_ambiguity(
            whitened_current,
            whitened_future,
            group_ids,
            neighbors=neighbors,
            chunk_size=chunk_size,
        ),
        "whitening_contract": "shared_channel_covariance_no_spatial_mixing",
    }
    if previous_codes is not None:
        if previous_codes.shape != current_codes.shape:
            raise ValueError("previous codes must match current code shape")
        whitened_previous = whitener.transform(previous_codes).reshape(sample_count, -1)
        conditioned = torch.cat((whitened_previous, whitened_current), dim=-1)
        history = _neighbor_ambiguity(
            conditioned,
            whitened_future,
            group_ids,
            neighbors=neighbors,
            chunk_size=chunk_size,
        )
        current_error = result["current_only"]["future_neighbor_rms_mean"]
        history_error = history["future_neighbor_rms_mean"]
        history["future_ambiguity_ratio_to_current_only"] = (
            history_error / current_error
        )
        history["material_improvement_at_20_percent"] = (
            history_error <= 0.8 * current_error
        )
        result["one_previous_code"] = history
    return result


def empirical_decoder_gains(
    model: SpatialTokenAutoencoder,
    code: torch.Tensor,
    geometry: TokenGeometry,
    directions: torch.Tensor,
    *,
    epsilon: float = 1.0e-3,
) -> dict[str, Any]:
    """Measure local raw-decoder gains in the fixed physical norm."""

    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    if directions.ndim == code.ndim:
        directions = directions.unsqueeze(0)
    if directions.ndim != code.ndim + 1 or directions.shape[1:] != code.shape:
        raise ValueError("directions must have shape [K,B,T,C]")
    if not torch.isfinite(directions).all():
        raise ValueError("decoder perturbation directions must be finite")

    was_training = model.training
    model.eval()
    gains: list[float] = []
    try:
        with torch.no_grad():
            baseline = model.decode(code, geometry)
            for direction in directions:
                direction_rms = direction.square().mean().sqrt()
                if not bool(direction_rms > 0.0):
                    raise ValueError("decoder perturbation directions must be nonzero")
                normalized_direction = direction / direction_rms
                perturbed = model.decode(
                    code + epsilon * normalized_direction, geometry
                )
                scaled_change = (perturbed - baseline) / model.state_scale
                output_rms = torch.sqrt(
                    (scaled_change.square() * geometry.volumes.unsqueeze(-1)).sum()
                    / (4.0 * geometry.volumes.sum())
                )
                gains.append(float((output_rms / epsilon).cpu()))
    finally:
        model.train(was_training)
    gain_array = np.asarray(gains, dtype=np.float64)
    return {
        "epsilon": float(epsilon),
        "directions": int(len(gains)),
        "mean_gain": float(gain_array.mean()),
        "median_gain": float(np.median(gain_array)),
        "maximum_gain": float(gain_array.max()),
        "raw_decode": True,
        "intervention_applied": False,
    }


class WeightedSnapshotPOD(nn.Module):
    """Fixed-mesh weighted POD control with an explicit effective-rank contract."""

    def __init__(
        self,
        *,
        modes: torch.Tensor,
        weighted_mean: torch.Tensor,
        feature_weights: torch.Tensor,
        state_mean: torch.Tensor,
        state_scale: torch.Tensor,
        num_nodes: int,
        requested_rank: int,
    ) -> None:
        super().__init__()
        if modes.ndim != 2 or modes.shape[1] != 4 * num_nodes:
            raise ValueError("POD modes do not match the declared mesh")
        if modes.shape[0] != requested_rank:
            raise ValueError("POD effective rank does not match requested rank")
        self.num_nodes = int(num_nodes)
        self.requested_rank = int(requested_rank)
        self.register_buffer("modes", modes)
        self.register_buffer("weighted_mean", weighted_mean)
        self.register_buffer("feature_weights", feature_weights)
        self.register_buffer("state_mean", state_mean.reshape(1, 1, 4))
        self.register_buffer("state_scale", state_scale.reshape(1, 1, 4))

    @classmethod
    def fit(
        cls,
        states: torch.Tensor,
        node_volumes: torch.Tensor,
        normalization: Euler2DNormalization | Mapping[str, Any],
        *,
        rank: int,
        eigenvalue_relative_floor: float = 1.0e-10,
    ) -> "WeightedSnapshotPOD":
        """Fit through the snapshot Gram matrix; no reconstruction target is used."""

        if states.ndim != 3 or states.shape[-1] != 4:
            raise ValueError("states must have shape [S,N,4]")
        if rank < 1 or states.shape[0] <= rank:
            raise ValueError("centered rank requires at least rank + 1 snapshots")
        if eigenvalue_relative_floor <= 0.0:
            raise ValueError("eigenvalue_relative_floor must be positive")
        if not isinstance(normalization, Euler2DNormalization):
            normalization = Euler2DNormalization.from_mapping(normalization)
        volumes = _batched_volumes(
            node_volumes,
            batch_size=1,
            num_nodes=states.shape[1],
        )[0].to(device=states.device, dtype=states.dtype)
        normalized_volume = volumes / volumes.sum()
        feature_weights = normalized_volume.sqrt().repeat_interleave(4)
        state_mean = torch.as_tensor(
            normalization.state_mean,
            dtype=states.dtype,
            device=states.device,
        )
        state_scale = torch.as_tensor(
            normalization.state_scale,
            dtype=states.dtype,
            device=states.device,
        )
        normalized = (states - state_mean.view(1, 1, 4)) / state_scale.view(1, 1, 4)
        weighted = normalized.reshape(states.shape[0], -1) * feature_weights
        weighted_mean = weighted.mean(dim=0)
        centered = weighted - weighted_mean
        gram = centered @ centered.T
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        order = torch.argsort(eigenvalues, descending=True)[:rank]
        selected_values = eigenvalues[order]
        threshold = eigenvalue_relative_floor * eigenvalues.max()
        if not bool(torch.all(selected_values > threshold)):
            effective = int(torch.count_nonzero(selected_values > threshold))
            raise ValueError(
                f"requested POD rank {rank} has only {effective} resolved directions"
            )
        selected_vectors = eigenvectors[:, order]
        modes = selected_vectors.T @ centered
        modes = modes / selected_values.sqrt().unsqueeze(-1)
        return cls(
            modes=modes,
            weighted_mean=weighted_mean,
            feature_weights=feature_weights,
            state_mean=state_mean,
            state_scale=state_scale,
            num_nodes=states.shape[1],
            requested_rank=rank,
        )

    @property
    def latent_size(self) -> int:
        return self.requested_rank

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim == 2:
            states = states.unsqueeze(0)
        if states.shape[1:] != (self.num_nodes, 4):
            raise ValueError("POD input does not match the fitted mesh")
        normalized = (states - self.state_mean) / self.state_scale
        weighted = normalized.reshape(states.shape[0], -1) * self.feature_weights
        return (weighted - self.weighted_mean) @ self.modes.T

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        if code.ndim == 1:
            code = code.unsqueeze(0)
        if code.ndim != 2 or code.shape[-1] != self.requested_rank:
            raise ValueError("POD code has the wrong rank")
        weighted = self.weighted_mean + code @ self.modes
        normalized = (weighted / self.feature_weights).view(
            code.shape[0], self.num_nodes, 4
        )
        return self.state_mean + self.state_scale * normalized

    def contract(self) -> dict[str, Any]:
        return {
            "schema": LINE4_REPRESENTATION_SCHEMA,
            "variant": "pod",
            "latent_size": self.requested_rank,
            "effective_rank": self.modes.shape[0],
            "spatially_indexed": False,
            "resolution_contract": "fixed_mesh_only",
            "raw_decode": True,
            "clipping": False,
            "floors": False,
            "limiter": False,
            "decode_reencode_projection": False,
        }


def _state_dict_digest(
    state: Mapping[str, torch.Tensor],
    prefixes: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    selected = [
        name
        for name in sorted(state)
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    ]
    if not selected:
        raise ValueError("component digest selected no tensors")
    for name in selected:
        tensor = state[name].detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def representation_component_digests(
    model: SpatialTokenAutoencoder,
) -> dict[str, str]:
    state = model.state_dict()
    shared_buffers = ("state_mean", "state_scale")
    return {
        "encoder_sha256": _state_dict_digest(
            state,
            (*shared_buffers, "cell_encoder", "encoder_mixer"),
        ),
        "decoder_sha256": _state_dict_digest(
            state,
            (*shared_buffers, "decoder_mixer", "point_decoder"),
        ),
    }


def representation_artifact_ledger(
    model: SpatialTokenAutoencoder,
    *,
    split: str,
    trajectory: str,
    physical_states_file: str,
    latent_states_file: str,
    valid_length: int,
    failure_cause: str,
    seed: int,
) -> dict[str, Any]:
    """Create the mandatory ledger while keeping analysis explicitly absent."""

    if split not in {"train", "validation"}:
        raise ValueError("the representation gate may use train or validation only")
    if valid_length < 0:
        raise ValueError("valid_length cannot be negative")
    if not trajectory or not physical_states_file or not latent_states_file:
        raise ValueError("trajectory and state artifact names are required")
    return {
        "schema": LINE4_REPRESENTATION_SCHEMA,
        "split": split,
        "trajectory": trajectory,
        "physical_states_file": physical_states_file,
        "latent_states_file": latent_states_file,
        **representation_component_digests(model),
        "valid_length": int(valid_length),
        "failure_cause": str(failure_cause),
        "training_seed": int(seed),
        "observation_contract": {"status": "not_applicable_representation_gate"},
        "ensemble_seeds": [],
        "analysis_applied": False,
        "analysis_flags": [],
        "interventions": {
            "clipping": False,
            "floors": False,
            "limiter": False,
            "decode_reencode_projection": False,
            "assimilation_reset": False,
        },
        "transition_present": False,
        "representation_contract": model.contract(),
    }
