from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import run_kolmogorov_corrective_followup as runner


def test_early_coverage_changes_times_without_changing_input_count():
    rows=runner.early_rows(32)
    assert rows == runner.early_rows(32)
    assert len(rows)==1024
    for p in range(32):
        t=[r["input_step"] for r in rows if r["path_index"]==p]
        assert len(t)==len(set(t))==32
        assert set(range(1,8)) <= set(t)
        assert sum(x<32 for x in t)==16
        assert all(1<=x<512 for x in t)
    assert {r["mixed_length"] for r in rows}=={1}


def test_gaussian_profile_preserves_direction_and_does_not_rescale_native_errors():
    center=np.ones((4,4),dtype=np.float32)
    gaussian=center+np.arange(16,dtype=np.float32).reshape(4,4)/10
    before=gaussian.copy()
    actual=runner.profile_gaussian(center,gaussian,.2,.4)
    np.testing.assert_allclose(actual-center,(gaussian-center)*.5,atol=1e-7)
    np.testing.assert_array_equal(gaussian,before)
    np.testing.assert_array_equal(runner.profile_gaussian(center,gaussian,0,.4),center)


def test_paired_targets_distinguish_exact_recovery_from_exact_dynamics():
    clean=np.arange(24,dtype=float).reshape(2,3,4)
    dynamics=clean+2
    rec=runner.paired_scores(clean,dynamics,clean,2)
    dyn=runner.paired_scores(dynamics,dynamics,clean,2)
    np.testing.assert_allclose(rec["rec_mse"],0)
    np.testing.assert_allclose(rec["dyn_mse"],1)
    np.testing.assert_allclose(rec["cross_term"],-2)
    np.testing.assert_allclose(dyn["rec_mse"],1)
    np.testing.assert_allclose(dyn["dyn_mse"],0)
    with pytest.raises(ValueError,match="identical shapes"):
        runner.paired_scores(clean,dynamics,clean[:1],2)


def test_resume_preserves_optimizer_sampler_and_random_state():
    torch.manual_seed(11)
    model=torch.nn.Linear(3,1)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    samplers=[runner.base.EpochSampler(16,4,17),runner.base.EpochSampler(8,4,1701)]
    def update(m,o,ss):
        a,b=[s.next_indices() for s in ss]
        x=torch.randn(4,3)+torch.tensor(a[:,None],dtype=torch.float32)/16
        y=torch.tensor(b[:,None],dtype=torch.float32)/8
        o.zero_grad(); (m(x)-y).square().mean().backward(); o.step()
        return a,b
    for _ in range(3): update(model,optimizer,samplers)
    saved=copy.deepcopy(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
        samplers=[s.state_dict() for s in samplers],torch_rng=torch.get_rng_state(),cuda_rng=[],update=3))
    expected_indices=update(model,optimizer,samplers)
    expected=copy.deepcopy(model.state_dict())
    restored=torch.nn.Linear(3,1)
    opt=torch.optim.Adam(restored.parameters(),lr=.7)
    ss=[runner.base.EpochSampler(16,4,99),runner.base.EpochSampler(8,4,98)]
    assert runner.restore_training_state(saved,restored,opt,ss)==3
    actual_indices=update(restored,opt,ss)
    for a,b in zip(expected_indices,actual_indices): np.testing.assert_array_equal(a,b)
    for key in expected: torch.testing.assert_close(restored.state_dict()[key],expected[key],rtol=0,atol=0)
    assert opt.param_groups[0]["lr"]==.01


def test_population_rms_pools_squares_before_root():
    from scripts.time_dependent_no.plot_kolmogorov_corrective_followup import pool
    values=np.array([[1.,4.],[9.,16.],[100.,100.]])
    np.testing.assert_allclose(pool(values,[True,True,False]),np.sqrt([5.,10.]))


def test_comparison_table_separates_populations_and_horizons(tmp_path):
    import csv
    from scripts.time_dependent_no.plot_kolmogorov_corrective_followup import comparison_tables
    original=np.repeat(np.array([[4.],[1.],[9.]]),33,axis=1)
    continued=np.repeat(np.array([[1.],[.25],[4.]]),33,axis=1)
    continued[1:,9:]=np.array([[4.],[16.]])
    continued[:,0]=1000.  # Initial-state entries must not enter a rollout metric.
    comparison_tables(tmp_path,tmp_path,
        {"pretrained":{"rollout_mse":original},"clean_continuation":{"rollout_mse":continued}},
        {"populations":["train","validation","validation"]})
    with (tmp_path/"comparisons.csv").open() as f:
        rows={(r["population"],int(r["horizon"])):r for r in csv.DictReader(f)}
    assert len(rows)==4
    assert float(rows["train",32]["ratio"])==.5
    assert float(rows["validation",8]["ratio"])==pytest.approx(np.sqrt(17/40))
    assert float(rows["validation",32]["ratio"])==pytest.approx(np.sqrt(257/160))
    assert int(rows["validation",8]["improved_cases"])==2
    assert int(rows["validation",32]["improved_cases"])==0


@pytest.mark.parametrize("target",["clean","recovery","dynamics"])
def test_followup_training_uses_exact_branch_inputs_and_targets(tmp_path,monkeypatch,target):
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.slope=torch.nn.Parameter(torch.tensor(.9))
            self.register_buffer("train_scale",torch.tensor(1.))
        def forward(self,x):
            y=self.slope*x
            return dict(next_state=y,raw_next=y)
    rng=np.random.default_rng(7)
    study=object.__new__(runner.Study)
    study.out=tmp_path; study.device="cpu"; study.scale=1.
    study.train=rng.normal(size=(2,9,4,4)).astype(np.float32)
    study.train_hash="synthetic"; study.sources={}; study.registry={}
    study.original=dict(model_config={})
    study.old_model=lambda _: (Toy(),dict(synthetic=True))
    center=rng.normal(size=(16,4,4)).astype(np.float32)
    data=dict(rows=[dict(path_index=i//8,input_step=i%8,mixed_length=1) for i in range(16)],
              centers=center,inputs=center+3,clean_targets=center*2,dynamics=(center+3)*2,
              identity=dict(synthetic=True))
    original_update=runner.base.update
    calls=[]
    def checked(model,optimizer,batches,device):
        calls.append(tuple((x.numpy().copy(),y.numpy().copy()) for x,y in batches))
        return original_update(model,optimizer,batches,device)
    monkeypatch.setattr(runner.base,"update",checked)
    study.train_run("fixture",data,target,steps=2)
    records=[json.loads(line) for line in (tmp_path/"training/fixture/updates.jsonl").read_text().splitlines()]
    for record,batches in zip(records,calls):
        first,second=np.array(record["clean_indices"]),np.array(record["displaced_indices"])
        p,t=np.divmod(first,8)
        np.testing.assert_array_equal(batches[0][0],study.train[p,t])
        np.testing.assert_array_equal(batches[0][1],study.train[p,t+1])
        np.testing.assert_array_equal(batches[1][0],data["centers" if target=="clean" else "inputs"][second])
        np.testing.assert_array_equal(batches[1][1],data["dynamics" if target=="dynamics" else "clean_targets"][second])
    result=json.loads((tmp_path/"training/fixture/result.json").read_text())
    assert result["status"]=="completed" and result["updates_completed"]==2
    assert set(result["measurements"])=={"0","2"}
    saved=torch.load(tmp_path/"training/fixture/checkpoint_000002.pt",weights_only=False)
    assert saved["update"]==2
    assert all(float(s["step"])==2 for s in saved["optimizer"]["state"].values())
