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
    _record_step_loss,
    _train_epoch,
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
        accumulation_steps=2,
    )

    assert model.weight.item() == pytest.approx(-0.35)
    assert result["mean_total_loss"] == pytest.approx(5.0)
    assert result["batches"] == 2
    assert result["microbatches"] == 3
    assert result["samples"] == 3


def test_policy_validation_requires_every_raw_mesh_case(tmp_path: Path):
    path = tmp_path / "summary.json"
    case = {
        "mesh": {
            "graph_native_boundary_mapping": "verified_against_abaqus_boundary",
            "graph_native_normal_max_abs_error": 1.0e-6,
            "graph_native_normal_coherence_max_abs_error": 2.0e-6,
        }
    }
    path.write_text(
        json.dumps(
            {
                "schema": POLICY_VALIDATION_SCHEMA,
                "status": "valid",
                "case_count": 20,
                "dataset_sha256": "test-digest",
                "graph_mesh_identity": "verified_all_selected_cases",
                "boundary_geometry": "verified_all_selected_cases",
                "cases": [case] * 20,
            }
        ),
        encoding="utf-8",
    )
    result = _validate_policy_summary(path)
    assert result["case_count"] == 20
    assert result["max_normal_abs_error"] == pytest.approx(1.0e-6)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cases"][0] = {"mesh": {}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not validated"):
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
