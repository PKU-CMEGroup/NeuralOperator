# P1 Path-Conditioned Tube Diagnostic Preregistration

Updated: 2026-08-26

Status: retained A1 supporting metric algebra and synthetic CPU contracts; no
real checkpoint, dataset array, remote process, training, or sealed-population
evaluation is authorized or reported by this document, and A2 is not the
automatic next scientific gate

## Decision Context

D094 is closed as retrospective development evidence for a replicated
one-step/rollout separation. P0-RS-A2 then showed that the registered native
coarse finite-volume step is not accurate enough to serve as the trusted
fine-evolve/restrict target from displaced states. The stronger solver-relative
response diagnostic therefore remains unavailable under the retained bump and
dynamic-FV contracts.

The owner initially selected the question of finding a quantity that predicts
rollout performance beyond one-step accuracy before opening the broader
architecture, training, and inference-strategy ladders. This registration takes
the resulting narrower path-conditioned route. Under the newer corrector-centred
programme, it is retained as a bounded phenotype/prediction screen and possible
intervention selector rather than the thesis center or automatic next gate. It
is informed by the owner-provided unpublished tubular-stability draft, which was
read locally and is not copied, quoted, or source-bound here.

## Problem Anchor

Bottom-line problem:

> Identify a short-prefix, model-independent observable of a learned one-step
> map that predicts an unseen long-horizon ranking after one-step accuracy and
> early rollout accuracy have been accounted for.

Must-solve bottleneck:

> Clean reference-state error does not characterize how the deployed map reacts
> to finite deviations created by previous model errors.

Non-goals:

- no universal stability theorem, attractor claim, or ID/OOD definition;
- no claim that the retained bump graph reconstructs a Trixi-DG restart state;
- no solver-relative off-path dynamics claim without `S(u + eta)`;
- no data-scaling, optimizer, representation, gradient, or architecture cause;
- no physical conservation claim from bump proxy weights;
- no new training, architecture panel, historical-test access, or checkpoint
  selection under A1; and
- no tangent/normal manifold claim without a validated projector.

Success condition:

> A score frozen from an H20-or-shorter common deviation bank must improve a
> genuinely prospective H79 architecture/seed ranking beyond both fixed-
> validation one-step error and H20, on more than one independent comparison.

## Why The Diagnostic Is Narrower Than Solver-Relative Response

For a reference input `u`, archived next reference state `u_plus`, learned map
`G`, and finite reachable displacement `eta`, define

    d = G(u) - u_plus
    p = G(u + eta) - G(u).

The draft's transverse-gain idea motivates measuring finite-amplitude response
along deviations that a learned rollout can actually generate. P0-RS-A2 blocks
the corresponding trusted-solver quantity because `S(u + eta)` is unavailable.
The present diagnostic instead measures distance to the same archived reference
path. It may therefore reward a retracting map whose off-path physics is wrong.
It is a candidate predictor of reference-rollout error, not deployment-response
fidelity in the stronger physical sense.

This limitation is a falsifier, not wording to hide. If the score predicts
rollout but a later restartable system shows that it rewards response-inaccurate
retraction, the paper must retain the narrower path-conditioned claim.

## Primary Probe Quantity

Use one fixed component-scaled, proxy-weighted inner product `W` shared by all
models inside a matched comparison. Define

    kappa_G(u, eta)
      = [||G(u + eta) - u_plus||_W - ||G(u) - u_plus||_W]
        / ||eta||_W.

`kappa_G` is the signed finite-amplitude **tube-escape slope**. Lower is better:

- positive values mean that the displacement increases next-reference error;
- negative values mean response/defect cancellation or retraction; and
- zero displacement is unresolved and is never hidden behind an epsilon.

Also report

    q_sec = ||p||_W / ||eta||_W
    c_dp  = <d, p>_W / (||d||_W ||p||_W).

The exact identity

    ||d + p||_W^2 = ||d||_W^2 + ||p||_W^2 + 2 <d, p>_W

shows why learned gain alone is insufficient: the same response norm can help or
harm depending on alignment with the clean defect. It also yields

    -q_sec <= kappa_G <= q_sec.

The following remain secondary components rather than substitute primary
scores:

- clean defect magnitude `||d||_W`;
- displaced combined defect `||d+p||_W`;
- squared excess per input energy;
- learned secant gain `q_sec`;
- defect-response cosine `c_dp`;
- finite-output admissibility and boundedness; and
- amplitude-response curves.

Do not use a global Lipschitz bound or a random-direction JVP as the primary
quantity. They can penalize physically real expanding directions, miss
finite-amplitude nonlinearity, and ignore which directions the rollout injects.

## Norm And Deployment Map

The evaluated map is the complete deployed one-call state map after residual
reconstruction and the frozen `causal_nodal_physical` boundary policy. Direct,
residual, gradient, no-gradient, Fourier-factorized, MLP, or future attention
implementations are compared only through this common state-map interface.

Within each matched trajectory-count comparison:

- use the same count-specific D094 component scale for every architecture and
  seed in that comparison;
- use the same reconstructed bump proxy weights and active-node mask;
- average over active proxy mass and components exactly once;
- label these as proxy-weighted state metrics, not physical-volume or
  conservation metrics; and
- do not use cross-count scalar rankings as primary evidence because D094
  intentionally fitted count-specific normalizers.

## Frozen Short-Prefix Deviation Bank

The intended A2 development bank uses only already open D094 development
objects. An A2 amendment must bind exact checkpoint bytes, descriptors, split,
data, evaluator source, normalizers, and command before any model call.

Semantic contract:

1. Work separately at each `n in {8,16,32,64,128,256}`.
2. Donors are the two seed-`20260718` exact-`64n` PCNO/PCFNO maps at the same
   `n`.
3. Recipients are the seed-`20260812` and seed-`20260813` exact-`64n`
   PCNO/PCFNO maps. Donor and recipient checkpoint sets are disjoint.
4. Use the recipient seed's registered 28-case outside-selection development
   cohort. Historical test remains sealed.
5. Generate each donor's free path only through H20 from exact frame zero.
6. Probe outputs at calls `{4, 8, 12, 16, 20}`. The reference input and donor
   path input are the corresponding pre-call states.
7. Form `eta = a * (u_donor - u_reference)` for
   `a in {0.5, 1.0, 2.0}`.
8. Before any recipient call, freeze the subset with finite, boundary-consistent,
   Euler-admissible inputs and nonzero scaled displacement. Input filtering may
   depend only on the reference state, donor state, amplitude, and frozen
   physical contract--never on a recipient output or H79 result.
9. Every recipient at the same `n` sees the identical surviving probe keys.
10. No state, error, metric, event, or selection information after H20 enters
    bank construction or score computation.

The future A2 preflight must report the survival count in every
`(case, amplitude)` cell. It stops before recipient inference if any cell is
empty, identities duplicate, donor and recipient sets overlap, the common-bank
keys differ across recipients, a zero displacement survives, or a named source
or artifact binding fails.

## A0 Live Manifest Audit

The 2026-08-26 local ignored-artifact audit verifies the current manifest
surface rather than inferring availability from historical prose:

- the B1-C2 retrieval SHA-256 manifest currently rehashes at
  `707e3ef2e7160becd4f7ef950caa80d20ee6e78e053d6a0215c3fdc3849f630d`;
- its matrix receipt currently hashes to
  `5d7aa6d027849c080c1084fdac8f632a04456eebf2d6a81a629992fb63c3187b`
  and binds 24 unique exact-`64n` sentinel hashes for seeds `20260812` and
  `20260813` across six counts and two architectures;
- the retained B1-C2 tar inventory contains all 72 registered checkpoint files:
  24 best, 24 last, and 24 exact-`64n` sentinels. This audit listed the archive
  and rehashed its manifest; it did not extract or independently rehash all
  5.6 GB of inner payloads;
- the seed-`20260718` compact tar rehashes at
  `db6aa41498b6d47ba83d9df50db3e45202d892861828b3bb9628bb3ee8836c2f`,
  and its matrix receipt rehashes at
  `cc935ce0495d9c153152ca280bbd650d47b4589cd7ddaf3c1afd6d03e05756b9`;
  that receipt binds all 12 exact-`64n` sentinel identities and hashes, but the
  retrieval manifest explicitly records `checkpoints_included=false`; and
- the retained B1-C5 fixed-map/map--path roots contain result and source
  packets but no selected/terminal checkpoint bytes.

Therefore the proposed A2 bank is not locally executable. Before an A2
amendment can be immutable, a separately approved read-only recovery must
rehash the 12 seed-`20260718` sentinel checkpoint bytes and the four B1-C5
selected/terminal checkpoint bytes against their retained receipts. Revising
the donor split merely to avoid that recovery would change the registered
cross-fitting design after outcomes are known and is not automatic.

## Primary Model Score

For each recipient, case, and amplitude, take the 0.9 quantile of `kappa_G` over
the frozen donor/call probes using NumPy's `linear` quantile definition. Average
those cell values with equal weight over cases and amplitudes:

    T_G = mean_(case, amplitude) Q_0.9[kappa_G].

Lower `T_G` predicts lower H79 error. Equal cell weighting prevents a case or
amplitude with more surviving probes from silently receiving more weight. The
0.9 quantile is a reachable-tail analogue of a worst transverse gain while
avoiding a single-probe maximum. Mean, median, `q_sec`, cosine, and the full
amplitude curve are secondary ablations and may not replace the primary score
after target access.

## Development And Audit Separation

All existing D094 H79 outcomes are already open, so neither stage below is a
prospective claim.

- Seed `20260812` is the development panel. It may be used to check the frozen
  algebra, numerical margins, and whether the score is obviously noninformative.
- Seed `20260813` is a pseudo-held audit panel. The code and score are frozen
  before its H79 table is joined. This is an out-of-sample development check,
  not a genuinely prospective result.
- The retained B1-C5 selected-versus-terminal `n={128,256}` crossover is an
  additional hard retrospective check. Its donors must come from a disjoint
  replication seed. The score is computed from H20-or-shorter probes before the
  already-known H79 direction is joined.

## Required Baselines And Outcomes

For every matched recipient pair, report separately:

1. fixed-validation one-step state error;
2. H20 free-rollout state error on the same cohort;
3. `T_G`;
4. H79 state error, joined only after the score artifact is finalized;
5. shock position, strength, and thickness at H20 and H79;
6. admissibility, boundedness, finiteness, and completion; and
7. parameter count, model calls, peak memory, and elapsed time.

Primary comparisons are within `(seed, n)` PCNO/PCFNO pairs and within the two
B1-C5 selected/terminal pairs. A cross-count regression or pooled correlation is
descriptive only.

## P1-A Routing Rule

The path-conditioned candidate is retained for a genuinely prospective P1-B
only if all of the following hold:

1. all source, checkpoint, split, normalizer, common-bank, closure, finiteness,
   and repeatability gates pass;
2. the pseudo-held seed score is no worse than the better of one-step and H20
   on all-pair winner accuracy;
3. on the subset where one-step and H20 agree but H79 reverses, `T_G` improves
   winner accuracy and the subset contains at least two matched comparisons;
4. it predicts the H79-harmful direction in both B1-C5 selected/terminal
   crossovers while using no state after H20; and
5. the direction is not carried only by nonfinite outputs, one case, one
   amplitude, or one donor architecture.

Failure stops this exact scalar and retains the decomposition as a diagnostic.
It does not automatically launch JVPs, longer prefixes, another aggregation,
more architectures, more data, or a higher-fidelity solver project. A revised
quantity requires a new registration before reusing the opened outcomes.

## Prospective P1-B Boundary

A successful retrospective gate still supports no paper claim. P1-B requires a
new architecture/seed panel whose H79 targets remain unopened until:

- training and one-step/H20 selection are final;
- donor bank, score code, checkpoint descriptors, and source manifest are
  immutable;
- all `T_G` artifacts and predicted pairwise rankings are finalized; and
- the exact target-opening command is separately approved.

The unchanged score must predict more than one independent H79 ranking beyond
both baselines. Only then may the architecture component ladder begin as a
mediator study. Training and inference strategies remain later stages and do
not enter P1-A.

## A1 Implementation Surface

- `utility/time_dependent_no/path_conditioned_tube.py` implements the exact
  metric algebra, identity/bound checks, and rectangular cross-fitted tail
  aggregation.
- `tests/time_dependent_no/test_path_conditioned_tube.py` uses synthetic CPU
  fixtures to check sign, closure, scaling, zero displacement, invalid geometry,
  donor/recipient leakage, prefix leakage, and common-bank completeness.

A1 creates no experiment entry point because the exact A2 checkpoint/archive
bindings have not yet been recovered into an immutable execution contract. The
live audit above identifies the missing bytes exactly. This keeps the
implementation smaller than a speculative evaluator and prevents an
unregistered real-model call.

## Claims Not Supported By A1

- that `T_G` predicts any retained or unseen rollout;
- that the owner-provided draft's tubular assumptions hold for bump states;
- that D094 errors are normal rather than tangent to a physical state family;
- that lower `T_G` means more faithful off-path PDE dynamics;
- that PCNO, PCFNO, FFNO, or any component is intrinsically more stable;
- that data scaling, capacity, optimization, or architecture causes the score;
  or
- that the architecture, training, or inference ladder is authorized.
