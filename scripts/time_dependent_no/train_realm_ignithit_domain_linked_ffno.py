"""Train the isolated D089 domain-linked direct FFNO for REALM IgnitHIT.

The entry point reuses the frozen D088 loop inside this process after binding a
distinct model, run identity, schemas, source manifest, and structure gate. It
does not modify the byte-frozen parent source. Real-data or GPU execution needs
separate A3 authorization.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no import train_realm_ignithit_ffno as parent
from utility.time_dependent_no.realm_benchmark import canonical_json_sha256
from utility.time_dependent_no.realm_domain_link import (
    BoxCoxDomainLink,
    DomainLinkedMap,
)
from utility.time_dependent_no.realm_ffno import RealmFFNO2d, RealmFFNOConfig
from utility.time_dependent_no.realm_ignithit import (
    BOX_COX_LAMBDA,
    PRIMARY_BOX_COX_EPSILON,
    sha256_file,
)

RUN_ID = "d089_realm_ignithit_p1d_domain_link_direct_seed0_5000_20260813a"
BEST_CHECKPOINT_SCHEMA = "d089_ignithit_domain_link_ffno_best_v1"
LAST_CHECKPOINT_SCHEMA = "d089_ignithit_domain_link_ffno_last_v1"
BASE_FLOOR = PRIMARY_BOX_COX_EPSILON**BOX_COX_LAMBDA
PARENT_SOURCE_SHA256 = (
    "a3fa0c0e831765ce24c6ae3aa11e3d2afcf67cc242de0a508381dc0b25b43f1e"
)
PARENT_NORMALIZER_ARRAYS_SHA256 = parent.NORMALIZER_ARRAYS_SHA256
PARENT_STD_CORRECTION = parent.STD_CORRECTION
PARENT_SCALE_STABILIZER = parent.SCALE_STABILIZER

_PARENT_RUN_ID = parent.RUN_ID
_PARENT_BEST_CHECKPOINT_SCHEMA = parent.BEST_CHECKPOINT_SCHEMA
_PARENT_LAST_CHECKPOINT_SCHEMA = parent.LAST_CHECKPOINT_SCHEMA
_PARENT_MODEL_FACTORY = parent.RealmFFNO2d
_PARENT_FROZEN_TRAINING_CONTRACT = parent.frozen_training_contract
_PARENT_TRAINING_CONTRACT = _PARENT_FROZEN_TRAINING_CONTRACT()
_PARENT_SOURCE_MANIFEST = parent._source_manifest
_PARENT_RUNTIME_MANIFEST = parent._runtime_manifest
_PARENT_NORMALIZERS_FROM_ARRAYS = parent._normalizers_from_arrays
_PARENT_RUN_VALIDATION = parent._run_validation
_PARENT_WRITE_JSON_ATOMIC = parent._write_json_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normalizer-arrays", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="resume only from OUTPUT_DIR/last.pt under the exact D089 contract",
    )
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="operational planned stop at a registered validation step",
    )
    return parser


def build_domain_link(
    normalizer_state: Mapping[str, Any],
) -> BoxCoxDomainLink:
    required = {
        "mean",
        "scale",
        "transformed_channels",
        "box_cox_lambda",
        "box_cox_epsilon",
        "std_correction",
        "scale_stabilizer",
        "source_sha256",
    }
    if set(normalizer_state) != required:
        raise ValueError("normalizer state differs from the frozen D089 contract")
    if float(normalizer_state["box_cox_lambda"]) != BOX_COX_LAMBDA:
        raise ValueError("normalizer Box-Cox lambda differs from D089")
    if float(normalizer_state["box_cox_epsilon"]) != PRIMARY_BOX_COX_EPSILON:
        raise ValueError("normalizer Box-Cox epsilon differs from D089")
    if tuple(normalizer_state["transformed_channels"]) != tuple(range(8)):
        raise ValueError("normalizer transformed channels differ from D089")
    if int(normalizer_state["std_correction"]) != PARENT_STD_CORRECTION:
        raise ValueError("normalizer standard-deviation correction differs from D089")
    if float(normalizer_state["scale_stabilizer"]) != PARENT_SCALE_STABILIZER:
        raise ValueError("normalizer scale stabilizer differs from D089")
    if normalizer_state["source_sha256"] != PARENT_NORMALIZER_ARRAYS_SHA256:
        raise ValueError("normalizer source digest differs from D089")
    return BoxCoxDomainLink(
        normalizer_state["mean"],
        normalizer_state["scale"],
        transformed_channels=tuple(normalizer_state["transformed_channels"]),
        box_cox_lambda=BOX_COX_LAMBDA,
        base_floor=BASE_FLOOR,
        channel_axis=1,
    )


def build_model(normalizer_state: Mapping[str, Any]) -> nn.Module:
    return DomainLinkedMap(
        RealmFFNO2d(RealmFFNOConfig()),
        build_domain_link(normalizer_state),
    )


def frozen_training_contract() -> dict[str, Any]:
    actual_parent_sha256 = sha256_file(Path(parent.__file__).resolve())
    if actual_parent_sha256 != PARENT_SOURCE_SHA256:
        raise RuntimeError("the byte-frozen D088 parent trainer source differs")
    payload = copy.deepcopy(_PARENT_TRAINING_CONTRACT)
    parent_config_digest = payload.pop("canonical_payload_sha256")
    payload.update(
        {
            "schema": "d089_ignithit_domain_link_ffno_reconstruction_v1",
            "run_id": RUN_ID,
            "claim_kind": "domain_linked_direct_baseline_amendment",
            "checkpoint_schemas": {
                "best": BEST_CHECKPOINT_SCHEMA,
                "last": LAST_CHECKPOINT_SCHEMA,
            },
            "selection_requires_all_decoded_finite_admissible_and_bounded": True,
            "parent_config_digest": parent_config_digest,
            "output_parameterization": {
                "name": "box_cox_domain_link_softplus_normalized_v1",
                "transformed_channels": list(range(8)),
                "box_cox_lambda": BOX_COX_LAMBDA,
                "base_floor": BASE_FLOOR,
                "formula": (
                    "lower=((base_floor-1)/lambda-mean)/scale; "
                    "shift=softplus_inverse(-lower); "
                    "normalized=lower+softplus(raw+shift)"
                ),
                "raw_zero_maps_to_normalized_zero": True,
                "trainable_parameters": 0,
                "untransformed_channels": "identity",
                "truth_conditioning": False,
                "recurrence_repair": False,
            },
            "parent_source_sha256": PARENT_SOURCE_SHA256,
            "parent_attempt": _PARENT_RUN_ID,
            "inherited_storage_schemas": {
                "input_manifest": "d088_ignithit_ffno_direct_inputs_v1",
                "reason": "exact matched input format and digest",
            },
        }
    )
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def source_manifest() -> dict[str, Any]:
    if sha256_file(Path(parent.__file__).resolve()) != PARENT_SOURCE_SHA256:
        raise RuntimeError("the byte-frozen D088 parent trainer source differs")
    paths = (
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_benchmark.py",
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_ignithit.py",
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_ffno.py",
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_domain_link.py",
        REPO_ROOT / "scripts" / "time_dependent_no" / "train_realm_ignithit_ffno.py",
        Path(__file__).resolve(),
    )
    payload: dict[str, Any] = {
        "schema": "d089_ignithit_domain_link_ffno_executed_source_v1",
        "files": [
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _run_validation(*args: Any, **kwargs: Any) -> dict[str, Any]:
    validation = _PARENT_RUN_VALIDATION(*args, **kwargs)
    model = args[0] if args else kwargs.get("model")
    if not isinstance(model, DomainLinkedMap):
        raise TypeError("D089 validation requires a domain-linked model")
    validation.update(domain_link_diagnostics(model))
    return require_eligible_validation(validation)


def domain_link_diagnostics(model: DomainLinkedMap) -> dict[str, Any]:
    """Report static guarantees without performing another model call."""

    link = model.output_link
    base_floor = torch.as_tensor(
        link.base_floor,
        dtype=link.mean.dtype,
        device=link.mean.device,
    )
    normalized_zero = torch.zeros(
        (1, link.mean.numel(), 1, 1),
        dtype=link.mean.dtype,
        device=link.mean.device,
    )
    zero_mapping_error = float(link(normalized_zero).abs().max().item())
    return {
        "domain_link": {
            "transformed_channels": list(link.transformed_channels),
            "inverse_base_floor_configured": link.base_floor,
            "inverse_base_floor_realized_float": float(base_floor.item()),
            "physical_species_floor_configured": PRIMARY_BOX_COX_EPSILON,
            "normalized_zero_mapping_max_abs_error": zero_mapping_error,
            "raw_nonfinite_policy": "raise_before_link",
            "raw_proposal_diagnostic_api": "model.forward_raw",
        }
    }


def require_eligible_validation(
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject a validation before it can participate in model selection."""

    required_flags = (
        "all_normalized_finite",
        "all_decoded_finite",
        "all_released_state_admissible",
        "all_bounded_10x_train_max",
    )
    if any(flag not in validation for flag in required_flags):
        raise ValueError("validation is missing a registered eligibility flag")
    if not (all(validation[flag] is True for flag in required_flags)):
        raise RuntimeError("validation failed the finite/admissible/bounded gate")
    if "realm_npe_mean" not in validation:
        raise ValueError("validation is missing the registered selection metric")
    if not math.isfinite(float(validation["realm_npe_mean"])):
        raise RuntimeError("validation selection metric is nonfinite")
    return dict(validation)


def _runtime_manifest(device: Any) -> dict[str, Any]:
    payload = _PARENT_RUNTIME_MANIFEST(device)
    payload.pop("canonical_payload_sha256")
    payload["schema"] = "d089_ignithit_domain_link_ffno_runtime_v1"
    payload["parent_runtime_format"] = "d088_ignithit_ffno_direct_runtime_v1"
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _json_payload(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if path.name == "history.json":
        result["schema"] = "d089_ignithit_domain_link_ffno_history_v1"
    elif path.name in {"status.json", "summary.json"}:
        result.update(
            {
                "schema": "d089_ignithit_domain_link_ffno_status_v1",
                "claim_kind": "domain_linked_direct_baseline_amendment",
                "parent_attempt": _PARENT_RUN_ID,
                "anti_claims": [
                    "this is an amended direct baseline, not the original raw direct baseline",
                    "this is a declared released-source reconstruction, not paper-faithful history",
                    "validation is a model-selection population, not untouched test evidence",
                    "released-state admissibility is not complete composition conservation",
                    "a partial run is not a baseline result",
                    "this run alone is not a direct-versus-residual comparison",
                ],
            }
        )
    elif path.name == "final_hash_manifest.json":
        result["schema"] = "d089_ignithit_domain_link_ffno_final_hash_manifest_v1"
    return result


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _PARENT_WRITE_JSON_ATOMIC(path, _json_payload(path, payload))


@contextmanager
def activated_contract() -> Iterator[None]:
    """Temporarily bind D089 into the frozen D088 execution engine."""

    captured_normalizer: dict[str, Mapping[str, Any]] = {}

    def normalizers_from_arrays(path: Path):
        result = _PARENT_NORMALIZERS_FROM_ARRAYS(path)
        captured_normalizer["state"] = result[3]
        return result

    def model_factory() -> nn.Module:
        normalizer_state = captured_normalizer.get("state")
        if normalizer_state is None:
            raise RuntimeError("D089 model construction preceded normalizer preflight")
        return build_model(normalizer_state)

    replacements = {
        "RUN_ID": RUN_ID,
        "BEST_CHECKPOINT_SCHEMA": BEST_CHECKPOINT_SCHEMA,
        "LAST_CHECKPOINT_SCHEMA": LAST_CHECKPOINT_SCHEMA,
        "RealmFFNO2d": model_factory,
        "frozen_training_contract": frozen_training_contract,
        "_source_manifest": source_manifest,
        "_runtime_manifest": _runtime_manifest,
        "_normalizers_from_arrays": normalizers_from_arrays,
        "_run_validation": _run_validation,
        "_write_json_atomic": _write_json_atomic,
    }
    originals = {
        "RUN_ID": _PARENT_RUN_ID,
        "BEST_CHECKPOINT_SCHEMA": _PARENT_BEST_CHECKPOINT_SCHEMA,
        "LAST_CHECKPOINT_SCHEMA": _PARENT_LAST_CHECKPOINT_SCHEMA,
        "RealmFFNO2d": _PARENT_MODEL_FACTORY,
        "frozen_training_contract": _PARENT_FROZEN_TRAINING_CONTRACT,
        "_source_manifest": _PARENT_SOURCE_MANIFEST,
        "_runtime_manifest": _PARENT_RUNTIME_MANIFEST,
        "_normalizers_from_arrays": _PARENT_NORMALIZERS_FROM_ARRAYS,
        "_run_validation": _PARENT_RUN_VALIDATION,
        "_write_json_atomic": _PARENT_WRITE_JSON_ATOMIC,
    }
    try:
        for name, expected in originals.items():
            current = getattr(parent, name)
            matches = current is expected if callable(expected) else current == expected
            if not matches:
                raise RuntimeError("D088 parent module changed before D089 activation")
        for name, value in replacements.items():
            setattr(parent, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(parent, name, value)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(Path(parent.__file__).resolve()) != PARENT_SOURCE_SHA256:
        raise RuntimeError("the byte-frozen D088 parent trainer source differs")
    with activated_contract():
        status = parent.run_training(args)
    return _json_payload(Path("status.json"), status)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        status = run_training(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(status, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
