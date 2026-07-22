from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.time_dependent_no.train_cpg_legal_boundary as legal_training
from scripts.time_dependent_no.train_cpg_legal_boundary import (
    POLICY_VALIDATION_SCHEMA,
    _concatenate_policies,
    _legal_recurrent_outflow_mach_min,
    _record_step_loss,
    _require_frozen_normalizers,
    _train_epoch,
    _validate_all_primitive_frames,
    _validate_policy_summary,
    parse_args,
)
from utility.time_dependent_no.cpg_mesh_contract import (
    BoundaryStencil,
    apply_torch_boundary_policy,
    build_torch_boundary_policy,
)


def test_legal_training_parser_rejects_empty_training_schedule(tmp_path: Path):
    validation = tmp_path / "summary.json"
    args = [
        "--reference-repo",
        str(tmp_path),
        "--dataset-root",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "out"),
        "--graph-policy-validation-summary",
        str(validation),
        "--stage1-epochs",
        "0",
        "--stage2-epochs",
        "0",
    ]
    with pytest.raises(ValueError, match="at least one epoch"):
        parse_args(args)
    assert parse_args([*args, "--audit-only"]).audit_only is True


@pytest.mark.parametrize("value", ["nan", "inf"])
def test_legal_training_parser_rejects_nonfinite_learning_rates(
    tmp_path: Path, value: str
):
    args = [
        "--reference-repo",
        str(tmp_path),
        "--dataset-root",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "output"),
        "--graph-policy-validation-summary",
        str(tmp_path / "summary.json"),
        "--stage1-lr",
        value,
    ]
    with pytest.raises(ValueError, match="must be positive"):
        parse_args(args)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("v1", np.nan, "nonfinite"),
        ("rho", -1.0, "inadmissible"),
        ("pres", 0.0, "inadmissible"),
    ],
)
def test_all_frame_primitive_preflight_rejects_bad_middle_state(
    field: str, value: float, match: str
):
    group = {
        name: np.ones((5, 3, 1), dtype=np.float64)
        for name in ("rho", "v1", "v2", "pres")
    }
    group[field][2, 1, 0] = value

    with pytest.raises(ValueError, match=match):
        _validate_all_primitive_frames(group, "case")


def test_legal_training_parser_defaults_to_effective_batch_three(tmp_path: Path):
    args = parse_args(
        [
            "--reference-repo",
            str(tmp_path),
            "--dataset-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--graph-policy-validation-summary",
            str(tmp_path / "summary.json"),
        ]
    )
    assert args.batch_size == 1
    assert args.gradient_accumulation_steps == 3
    assert args.teacher_forcing_batch_size == 3
    with pytest.raises(ValueError, match="effective batch size three"):
        parse_args(
            [
                "--reference-repo",
                str(tmp_path),
                "--dataset-root",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--graph-policy-validation-summary",
                str(tmp_path / "summary.json"),
                "--gradient-accumulation-steps",
                "2",
            ]
        )
    with pytest.raises(ValueError, match="teacher-forcing batch size"):
        parse_args(
            [
                "--reference-repo",
                str(tmp_path),
                "--dataset-root",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--graph-policy-validation-summary",
                str(tmp_path / "summary.json"),
                "--teacher-forcing-batch-size",
                "2",
            ]
        )


def test_legal_training_parser_rejects_nontraining_split(tmp_path: Path):
    with pytest.raises(ValueError, match="pinned to --split train"):
        parse_args(
            [
                "--reference-repo",
                str(tmp_path),
                "--dataset-root",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--graph-policy-validation-summary",
                str(tmp_path / "summary.json"),
                "--split",
                "validation",
            ]
        )


@pytest.mark.parametrize("bad_dt", [0.01, float("nan"), float("inf")])
def test_legal_training_parser_rejects_unimplemented_timestep(
    tmp_path: Path, bad_dt: float
):
    with pytest.raises(ValueError, match="hardcodes dt"):
        parse_args(
            [
                "--reference-repo",
                str(tmp_path),
                "--dataset-root",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--graph-policy-validation-summary",
                str(tmp_path / "summary.json"),
                "--dt",
                str(bad_dt),
            ]
        )


def test_record_step_loss_backpropagates_without_retaining_step_graphs():
    torch = pytest.importorskip("torch")
    weight = torch.nn.Parameter(torch.tensor(1.0))
    total = torch.zeros(())
    total = _record_step_loss(total, weight * 2.0, 0.5)
    total = _record_step_loss(total, weight * 3.0, 0.5)

    assert total.item() == pytest.approx(5.0)
    assert total.requires_grad is False
    assert weight.grad.item() == pytest.approx(2.5)


def test_train_epoch_accumulates_normal_node_weighted_microbatches(monkeypatch):
    torch = pytest.importorskip("torch")

    class Model:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def train(self):
            return None

    model = Model()
    graphs = [
        SimpleNamespace(
            x=torch.zeros((normal_nodes, 1)),
            loss_scale=loss_scale,
            num_graphs=1,
        )
        for normal_nodes, loss_scale in ((1, 2.0), (3, 4.0), (2, 10.0))
    ]

    def fake_microbatch_loss(*, model, graph, **_kwargs):
        return (
            model.weight * graph.loss_scale,
            torch.tensor(0.5),
            torch.tensor(0.25),
            None,
            int(graph.x.shape[0]),
            graph.num_graphs,
        )

    monkeypatch.setattr(legal_training, "_microbatch_loss", fake_microbatch_loss)
    optimizer = torch.optim.SGD([model.weight], lr=0.1)
    result = _train_epoch(
        api={"torch": torch},
        model=model,
        loader=graphs,
        optimizer=optimizer,
        transformer=None,
        policies=(),
        device=torch.device("cpu"),
        noise_std=None,
        teacher_forcing=False,
        max_batches=None,
        accumulation_steps=3,
    )

    assert model.weight.item() == pytest.approx(13.0 / 30.0)
    assert result["mean_total_loss"] == pytest.approx(17.0 / 3.0)
    assert result["batches"] == 1
    assert result["microbatches"] == 3
    assert result["samples"] == 3
    assert result["mean_gradient_norm"] == pytest.approx(17.0 / 3.0)
    assert result["min_recurrent_outflow_normal_mach"] is None


def test_train_epoch_rejects_partial_accumulation_group(monkeypatch):
    torch = pytest.importorskip("torch")

    class Model:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def train(self):
            return None

    def fail_if_called(**_kwargs):
        raise AssertionError("partial group must fail before a forward pass")

    monkeypatch.setattr(legal_training, "_microbatch_loss", fail_if_called)
    model = Model()
    graphs = [
        SimpleNamespace(x=torch.zeros((1, 1)), num_graphs=1),
        SimpleNamespace(x=torch.zeros((1, 1)), num_graphs=1),
    ]
    with pytest.raises(RuntimeError, match="incomplete"):
        _train_epoch(
            api={"torch": torch},
            model=model,
            loader=graphs,
            optimizer=torch.optim.SGD([model.weight], lr=0.1),
            transformer=None,
            policies=(),
            device=torch.device("cpu"),
            noise_std=None,
            teacher_forcing=False,
            max_batches=None,
            accumulation_steps=3,
        )


def test_train_epoch_rejects_nonfinite_gradient_before_optimizer_step(
    monkeypatch,
):
    torch = pytest.importorskip("torch")

    class Model:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def train(self):
            return None

    def fake_microbatch_loss(*, model, graph, **_kwargs):
        return (
            torch.sqrt(model.weight),
            torch.tensor(0.5),
            torch.tensor(0.25),
            None,
            int(graph.x.shape[0]),
            graph.num_graphs,
        )

    monkeypatch.setattr(legal_training, "_microbatch_loss", fake_microbatch_loss)
    model = Model()
    optimizer = torch.optim.SGD([model.weight], lr=0.1)
    graphs = [
        SimpleNamespace(x=torch.zeros((1, 1)), num_graphs=1) for _ in range(3)
    ]
    with pytest.raises(FloatingPointError, match="nonfinite gradients"):
        _train_epoch(
            api={"torch": torch},
            model=model,
            loader=graphs,
            optimizer=optimizer,
            transformer=None,
            policies=(),
            device=torch.device("cpu"),
            noise_std=None,
            teacher_forcing=False,
            max_batches=None,
            accumulation_steps=3,
        )
    assert model.weight.item() == 0.0


def test_sequential_multistep_requires_frozen_normalizers():
    torch = pytest.importorskip("torch")

    def normalizer(accumulations: float, maximum: float):
        return SimpleNamespace(
            _num_accumulations=torch.tensor(accumulations),
            _max_accumulations=maximum,
        )

    model = SimpleNamespace(
        _output_normalizer=normalizer(9.0, 10.0),
        _node_normalizer=normalizer(10.0, 10.0),
    )
    with pytest.raises(RuntimeError, match="frozen normalizers"):
        _require_frozen_normalizers(model)

    model._output_normalizer._num_accumulations.fill_(10.0)
    _require_frozen_normalizers(model)


def test_policy_validation_requires_every_raw_mesh_case(tmp_path: Path):
    path = tmp_path / "summary.json"
    cases = [
        {
            "trajectory_key": str(index),
            "raw_case_id": index + 1,
            "mesh": {
                "graph_native_boundary_mapping": (
                    "verified_against_abaqus_boundary"
                ),
                "graph_native_normal_max_abs_error": 1.0e-6,
                "graph_native_normal_coherence_max_abs_error": 2.0e-6,
            },
            "config": {
                "save_dt": 0.025,
                "gamma": 1.4,
                "rho_inf": 1.4,
                "p_inf": 1.0,
            },
            "boundary": {
                "causal_outflow_normal_mach_min": 2.0,
                "causal_outflow_evidence": (
                    "all_frames_extrapolated_from_interior_stencil"
                ),
                "boundary_stencil_sha256": "b" * 64,
            },
        }
        for index in range(20)
    ]
    path.write_text(
        json.dumps(
            {
                "schema": POLICY_VALIDATION_SCHEMA,
                "status": "valid",
                "case_count": 20,
                "case_offset": 1,
                "max_boundary_source_hops": 3,
                "dataset_sha256": "a" * 64,
                "graph_mesh_identity": "verified_all_selected_cases",
                "boundary_geometry": "verified_all_selected_cases",
                "temporal_alignment_evidence": (
                    "sampled_first_midpoint_last_not_full_temporal_identity"
                ),
                "static_graph_evidence": (
                    "all_frames_exact_release_facing_metadata"
                ),
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    result = _validate_policy_summary(path)
    assert result["case_count"] == 20
    assert result["max_normal_abs_error"] == pytest.approx(1.0e-6)
    assert result["max_boundary_source_hops"] == 3

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cases"][0]["mesh"] = {}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not validated"):
        _validate_policy_summary(path)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda payload: payload["cases"][0]["config"].update(gamma=1.3),
            "different physical configuration",
        ),
        (
            lambda payload: payload["cases"][0]["mesh"].update(
                graph_native_normal_max_abs_error=float("nan")
            ),
            "must be finite",
        ),
        (
            lambda payload: payload["cases"][1].update(
                trajectory_key=payload["cases"][0]["trajectory_key"]
            ),
            "zero-to-one-based",
        ),
        (
            lambda payload: payload["cases"][0]["boundary"].update(
                causal_outflow_normal_mach_min=0.9
            ),
            "not outward-supersonic",
        ),
        (
            lambda payload: payload.update(max_boundary_source_hops=2),
            "source-hop contract",
        ),
        (
            lambda payload: payload["cases"][0]["boundary"].update(
                boundary_stencil_sha256="invalid"
            ),
            "stencil digest",
        ),
        (
            lambda payload: payload["cases"][0].update(raw_case_id=99),
            "zero-to-one-based",
        ),
    ],
)
def test_policy_validation_rejects_incompatible_contracts(
    tmp_path: Path, mutator, match: str
):
    path = tmp_path / "summary.json"
    cases = [
        {
            "trajectory_key": str(index),
            "raw_case_id": index + 1,
            "mesh": {
                "graph_native_boundary_mapping": (
                    "verified_against_abaqus_boundary"
                ),
                "graph_native_normal_max_abs_error": 1.0e-6,
                "graph_native_normal_coherence_max_abs_error": 2.0e-6,
            },
            "config": {
                "save_dt": 0.025,
                "gamma": 1.4,
                "rho_inf": 1.4,
                "p_inf": 1.0,
            },
            "boundary": {
                "causal_outflow_normal_mach_min": 2.0,
                "causal_outflow_evidence": (
                    "all_frames_extrapolated_from_interior_stencil"
                ),
                "boundary_stencil_sha256": "b" * 64,
            },
        }
        for index in range(20)
    ]
    payload = {
        "schema": POLICY_VALIDATION_SCHEMA,
        "status": "valid",
        "case_count": 20,
        "case_offset": 1,
        "max_boundary_source_hops": 3,
        "dataset_sha256": "a" * 64,
        "graph_mesh_identity": "verified_all_selected_cases",
        "boundary_geometry": "verified_all_selected_cases",
        "temporal_alignment_evidence": (
            "sampled_first_midpoint_last_not_full_temporal_identity"
        ),
        "static_graph_evidence": "all_frames_exact_release_facing_metadata",
        "cases": cases,
    }
    mutator(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        _validate_policy_summary(path)


def test_policy_validation_rejects_legacy_v1_for_new_training(tmp_path: Path):
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "schema": "cpg_bump_mesh_provenance_audit_v1",
                "status": "valid",
                "cases": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires the hardened v2"):
        _validate_policy_summary(path)


def _policy(torch, mach: float):
    node_type = np.asarray([3, 0, 1, 2], dtype=np.int64)
    node_normal = np.asarray([[-1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    stencil = BoundaryStencil(
        target_nodes=np.asarray([2, 3], dtype=np.int64),
        target_rows=np.asarray([0, 1], dtype=np.int64),
        source_nodes=np.asarray([1, 1], dtype=np.int64),
        weights=np.asarray([1.0, 1.0], dtype=np.float64),
        fallback_target_count=0,
    )
    return build_torch_boundary_policy(
        torch=torch,
        device=torch.device("cpu"),
        node_type=node_type,
        node_normal=node_normal,
        wall_normal_coherence=np.ones(4),
        stencil=stencil,
        mach=mach,
        config={"gamma": 1.4, "rho_inf": 1.4, "p_inf": 1.0},
    )


def test_concatenated_policy_matches_independent_graph_policies():
    torch = pytest.importorskip("torch")
    policies = [_policy(torch, 2.5), _policy(torch, 3.0)]
    state_a = torch.tensor(
        [
            [9.0, 9.0, 9.0, 9.0],
            [2.0, 3.0, 4.0, 5.0],
            [8.0, 8.0, 8.0, 8.0],
            [7.0, 7.0, 7.0, 7.0],
        ]
    )
    state_b = state_a + 1.0
    expected = torch.cat(
        [
            apply_torch_boundary_policy(torch, state_a, policies[0]),
            apply_torch_boundary_policy(torch, state_b, policies[1]),
        ]
    )
    combined = _concatenate_policies(
        torch,
        policies=policies,
        policy_indices=torch.tensor([0, 1]),
        ptr=torch.tensor([0, 4, 8]),
        device=torch.device("cpu"),
    )
    actual = apply_torch_boundary_policy(torch, torch.cat([state_a, state_b]), combined)
    torch.testing.assert_close(actual, expected)
    assert combined["freestream"].shape == (2, 4)
    assert combined["outflow_rows"].numel() == 2
    assert combined["gamma"] == pytest.approx(1.4)


def test_recurrent_training_rejects_nonsupersonic_outflow():
    torch = pytest.importorskip("torch")
    policy = _policy(torch, 2.5)
    state = torch.tensor(
        [
            [1.4, 2.5, 0.0, 1.0],
            [2.0, 3.0, 0.0, 5.0],
            [2.0, 3.0, 0.0, 5.0],
            [2.0, 3.0, 0.0, 5.0],
        ]
    )
    assert _legal_recurrent_outflow_mach_min(
        torch, state, policy, step=0
    ) > 1.0

    invalid_replaced_boundary = state.clone()
    invalid_replaced_boundary[[0, 2, 3], 0] = -1.0
    assert _legal_recurrent_outflow_mach_min(
        torch, invalid_replaced_boundary, policy, step=0
    ) > 1.0

    state[1, 1] = 0.5
    with pytest.raises(FloatingPointError, match="non-supersonic recurrent outflow"):
        _legal_recurrent_outflow_mach_min(torch, state, policy, step=0)
