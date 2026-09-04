# Time-Dependent Neural Operators: Handoff

Updated: 2026-09-05

Status: replaceable operational snapshot. Scientific authority is
[RESEARCH_DIRECTION_DECISION.md](RESEARCH_DIRECTION_DECISION.md). Historical
run evidence belongs in the experiment index and verified Git or artifact
anchors.

## Read Order

1. repository-root `AGENTS.md`;
2. [RESEARCH_DIRECTION_DECISION.md](RESEARCH_DIRECTION_DECISION.md);
3. this handoff;
4. [MECHANISTIC_DIAGNOSTIC_TRACKER.md](MECHANISTIC_DIAGNOSTIC_TRACKER.md);
5. [README.md](README.md);
6. ignored `LOCAL_CONTEXT.md` privately, if present; and
7. live `git status --short --branch` plus relevant source/artifact manifests.

For active planning, then read:

1. [PROJECT_PLAN.md](PROJECT_PLAN.md);
2. [the paper plan](../../paper/PAPER_PLAN.md);
3. [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md);
4. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md);
5. [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md);
6. [B3B4_NACA_CORRECTIVE_EXTENSION_PREREGISTRATION.md](B3B4_NACA_CORRECTIVE_EXTENSION_PREREGISTRATION.md)
   and its JSON contract;
7. [B5_BUMP_SOLVER_FREE_TRANSFER_PREREGISTRATION.md](B5_BUMP_SOLVER_FREE_TRANSFER_PREREGISTRATION.md)
   and its JSON contract;
8. [PARKED_EXPERIMENTS.md](PARKED_EXPERIMENTS.md); and
9. [WEEKLY_RESEARCH_PLAN.md](WEEKLY_RESEARCH_PLAN.md).

Current explicit owner direction outranks all snapshots. No planning document
independently authorizes code, solver, model, remote, data, or sealed work.

## Current Thesis

The project now has exactly two headline claims:

- conditional non-identifiability of deployment-relevant off-trace response
  from clean one-step supervision in a sufficiently rich class; and
- prospective empirical value of frozen response/forcing/drift diagnostics for
  predicting rollout ranking reversals and regime-dependent intervention value
  beyond clean error, with PCNO fixed as the PDE backbone.

The project studies endogenous self-composition drift from in-distribution
initial conditions. Exogenous OOD initial-condition evaluation and online OOD
detection are out of scope. Offline reference-proximity diagnostics are allowed
only as experimental instruments.

Corrective mechanism is the umbrella term. Embedded mechanisms change training
or architecture; operational correctors are separately attributable deployed
maps/kernels. All final claims attach to the complete deployed transition.
A response-controlled tube is a region where response is identified by data,
constrained by structure, or restored by correction. Tube retention and
tangent/path accuracy remain separate obligations.

The central explanation is explicitly two-axis. Clean one-step error probes
the learned map on clean reference inputs--the formal object behind the
tangent/path intuition--but does not identify displaced-input response in
general. Accurate rollout additionally requires a transverse retention or
recovery property in the complete deployed map. Neither clean accuracy nor
tube retention alone is sufficient. Formal text uses “on-reference evolution”
unless tangent geometry has been qualified; the familiar tangent interpretation
is motivation, not the novelty claim.

Recovery and dynamics relabeling are outcome-neutral competitors:

- recovery uses `x=u+eta -> Phi(u)`;
- relabeling uses `x=u+eta -> Phi(x)`.

Substantial trusted normal response does not guarantee that relabeling wins the
ID rollout. Recovery may suppress a harmful displacement; it may also erase
meaningful phase dynamics. The experiment predicts the ranking before the
designated long horizon and reports both target fidelity and ID rollout quality.

## Current Empirical Programme

Required:

1. B1 exact tangent-normal ODE laboratory as a theory/measurement check
   (**complete and verified 2026-08-30**);
2. B2 compact learned ODE realization with one learner, three paired seeds,
   and only the smallest recovery/relabeling/explicit contrast needed for
   mechanism calibration (**complete and verified 2026-08-30**);
3. B3 mandatory fixed-PCNO diagnosis on one qualified, reasonably complex PDE
   selected through the hard readiness screen, ending in a problem-specific
   intervention-hypothesis freeze;
4. B4 representative embedded and operational intervention study, trained-model
   ranking freeze, and long-horizon reveal in that same case study; and
5. B5 selected reduced transfer/no-harm study on the Supersonic Bump.

SU2 Unsteady NACA0012 has completed the first R0 screen and is rejected as the
primary hero case under the frozen decision rule. Its valid negative result may
be retained as supplementary one-step/rollout-discrepancy evidence, and no
corrector study follows under that closed R0 identity. A distinct owner-approved
successor, `B3B4_NACA_CM_20260901A`, now makes NACA the primary fixed-PCNO case
for the narrower corrective-mechanism question. Its frozen comparison is
exactly `CLEAN`, `IID_RECOVERY`, `ERROR_SUBSPACE_RECOVERY`,
`DETACHED_PUSHFORWARD`, and `PATH_PROJECTION`. All twelve learned-arm/seed
training packets and the development-only result packet are locally verified;
the latter is bound by final-manifest SHA256
`09e3ae7e5896039e2230073c1ead5e5f9808c0cc9b47d0b15ff337b26b3e0ed1`.
This is closed parent evidence. The completed open-development extension
`B3B4_NACA_CM_EXT_20260902A` separately evaluates literal
`MP_PDE_PUSHFORWARD_M01`, curriculum/EMA depth-`1--3` exposure, paired frozen-
bank recovery versus SU2 dynamics relabeling, and
`PCNO_PDEREFINER_K3_VPRED`. Prospective and sealed populations remain unopened.

All extension gates closed at open-development scope. The corrected pilot and
all 476 paired-bank solver calls passed; 18 Attempt-D training packets were
evaluated without mutation by Attempt E. Source tests, GPU smoke, common
evaluation, retrieval, and independent packet audit passed under final-manifest
SHA256 `15475da78b4182754fe959eadc175434ffca18ef1558a45ea06cbf9c93ca7370`.
The local training-summary retrieval contains verified metadata/history only,
not checkpoints. Exact identities are centralized in
[EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md). No online solver call, online
defect trigger, prospective read, or sealed read occurred.

Path projection ranks first and PDE-Refiner second on both primary metrics at
every seed. Paired recovery is the safest one-call learned arm. Dynamics
relabeling and both new pushforward protocols are heterogeneous and sometimes
harmful. This is strong rollout evidence but only partial mediator evidence:
trained recovery and relabeling were not compared on common bank inputs against
both stored targets, and the one-prefix gain does not certify late stability.

The active secondary study is
`B5_BUMP_SOLVER_FREE_TRANSFER_20260905A`. It treats NACA's projection ranking
as problem-specific and asks whether solver-free corrective information helps
on geometry-dependent, advection-dominated bump trajectories where raw global
PCA is not a legitimate deployed map. Local Stage-0A/B identity/reducibility
and synthetic implementation are authorized. Stage 0C and Stage 1 are a frozen
design envelope that requires the amendments and resource gate in the B5
contract. Stage 1 is defined as `CLEAN`,
`IID_RECOVERY`, `CURRICULUM_EMA_PREFIX_K13`, and
`PREFIX_ERROR_CORRECTOR_K13`; a bump PDE-Refiner port is conditional. No remote
resource or dataset-scale launch is selected, and the historical test remains
sealed. The B5 preregistration and JSON contract are authoritative.

The field study is cutoff-bounded and comprehensive across declared method
families, while empirical implementations are representative. Its maintained
source ledger is [paper/LITERATURE_AUDIT.md](../../paper/LITERATURE_AUDIT.md),
verified through 2026-09-01. The deterministic proof line has completed audit.
The former W26 five-line programme is evidence history, not an active queue.

No surveyed SU2 alternative was clearly better before the honest NACA screen.
Turbulent square cylinder remains a possible fallback only under a pinned
production contract; laminar von Karman is a cheap solver/control fallback;
shock-vortex remains a tangential/no-harm control, while bump has the bounded
B5 secondary role above. Derived figures and
claim-bounded manuscript integration are complete. If the corresponding claim
is retained, an evaluation-only common-bank response assay must be frozen
separately. No fallback activates automatically. These roles are planning
decisions, not new empirical findings.

## Live Repository State At Rewrite

- branch: `time-dependent-no`;
- HEAD at the start of this documentation revision:
  `28e9f3b4faaea8ef9770f9899153a0d0c0b17c5f`;
- branch was 63 commits ahead of `origin/time-dependent-no`;
- active documents and a proposed historical-deletion set already contained
  uncommitted user-owned changes; and
- `paper/` was ignored and user-owned.

No manuscript source, experiment code, data, checkpoint, or scientific
artifact was moved during that rewrite. The proposed historical deletions were
not committed: tracked records remain available for reproducibility but are not
active sources of scientific direction.

Re-run Git status before any later edit or handoff; this prose is not the source
of truth for live file status.

## Verified M1 State

The M1 documents were stale at the start of this rewrite: they described the
R2-POP R1 process as authorized but pending. The live ignored receipts show the
following.

### Qualified parent reference

- spatial result SHA-256:
  `f32b2d4defd549de3035cbafa6bc1e129c73b43bfce18a0ac021072d16805ed4`;
- spatial artifact-manifest SHA-256:
  `9f8de40a0b919b41ea09806e8c892c59cebea605dd815dd093d2e5f58dd67c84`;
- spatial source commit:
  `7480ad89c6737817e73324651574866598474afb`;
- temporal result SHA-256:
  `460eb8d32df28f58ef2497f86b4374ec16477d88fa6fa872149c395d5c847196`;
- temporal artifact-manifest SHA-256:
  `af0b7bb197736bce337c7b92f189a056ed4cf6ee636d28d2a57cb471febb5412`;
- temporal source commit:
  `6f64fc5b3ad41e24e5f2bce973eb5b555bb74dfe`; and
- all recorded source blobs and all six referenced Q1/R1 parent result/manifest
  hashes independently reverified.

These packets qualify only the registered finite-grid N256 reference over the
frozen trajectories. They do not prove continuum convergence or qualify a
fresh population.

The temporal launch receipt retains an exit-code-capture failure even though
the complete packet independently passes. Preserve that caveat; do not rewrite
the raw receipt.

### Unqualified population

Launch 0 closed as `incomplete_8h_wall_time_cap_before_packet` with no result,
manifest, or retained series.

R1 used requested source commit equal to current HEAD and passed startup source
and parent-packet checks before solver progress. Its final receipt records:

- status `runner_failed_confirmed_nonzero`;
- exit code `1` with capture available;
- elapsed `22307.064 s`;
- no result, artifact manifest, retained series, or output directory;
- empty stdout;
- progress-only stderr with no traceback;
- seeds `2026083101`–`2026083103` last at `1792/5120` and seed
  `2026083104` last at `2048/5120`; and
- null `infrastructure_classification`.

R1 receipt SHA-256 is
`b7f976e9a02824da44358a63ce856ee0c8f2ed1d98f3a2e18c49513389c1a98b`;
its event/stderr SHA-256 is
`49e45a82c876932e09218f80ce27d329093dec022899ecd2336e6ea4cd0553c3`.

This is not a population sampling-law failure, stationarity result, or completed
infrastructure classification. End-of-run source stability cannot be verified
without a packet. Q2 remains closed, and no further retry is authorized.

The old M1 model plan fixed residual FNO. The current PDE experiment fixes
PCNO. Only the verified reference packets may be considered as parent evidence;
any future model stage needs a new PCNO identity and readiness contract.

### Live primary-PDE readiness

The 2026-08-31 Stage-0 NACA audit freezes the public tutorial configuration,
fixed mesh, and binary restart triplet 497/498/499. The files match their
declared sizes and SHA256 values; mesh/restart coordinates agree exactly; and
the complete evolved RANS--SA state and second-order history contract are
present. The maintained parser and focused synthetic tests pass.

The preparation-only harness remains non-executing and preserves canonical
target 499 externally. The audited native runner/evaluator/comparator has now
also completed two independently prepared final cases, C and D, under the
pinned SU2 `v8.5.0` identity. Both return successfully and produce the same
native output bytes. Against the strict canonical target, the five evolved
fields have component-balanced relative L2 discrepancy `4.3264e-7` under
farfield-dominated physical-area weighting, `5.9447e-4` under uniform-node
weighting, and `8.9081e-4` in the fixed near-body/wake box; local throughput is
`1.92`--`1.96` transitions/s.

The audited trajectory runner subsequently produced 1,501 native states over
indices 499--1999. The packet occupies `3,326,514,699` bytes, ran for
`5631.17 s`, and has ordered-state aggregate SHA256
`9adbcfd53c4812009d3c5e5081730b709a2f63f48237760390e2c8559ab200ef`.
The trajectory receipt SHA256 is
`3d857d6ddca4dca111ecc716b2c85ff23f727a4fafc97d39f24db340d2b2672b`.

The unchanged model-blind analyzer returned `PHASE_ONLY_SUPPORTED` under
contract SHA256
`f442065172beb014ebcd4f3823abe1e2764fb2b74bbad348a8dd94d6b4c903f5`.
The CL period is `34.6543` steps and 31 complete cycles remain after the
selected burn boundary. Uniform-node recurrence has median/q90
`0.003845/0.004076`; near-body/wake has `0.003827/0.004038`. The required
physical-area secondary view fails (`0.110995/0.146430`, q90
cycle-to-template `1.255897`) because the farfield-weighted orbit amplitude is
extremely small. This failure is part of the result and was not used to rescue
or veto the two frozen gated views. Analysis SHA256 is
`b14ef6d66c5e520633ea94954ff3b243a47c3185743de43a5ff10d1c7f74069f`.
An earlier sandboxed attempt failed before reading the packet because its
temporary directory was inaccessible; its `INVALID_ARTIFACT` receipt is
preserved as infrastructure evidence, and the analyzer was unchanged before
the successful rerun.

The 32-anchor population, content-disjoint role blocks, 208-step horizon,
complete-BDF2 baseline, metrics, and access rules subsequently passed
independent contract audit at SHA256
`94069e65ef520d31860735b6c17f2b7515da1c4acd34d68518a2a16b3181fa87`.
The train/development dataset, full-resolution AutoDL smoke, exact three-seed
training schedule, and unchanged development-only evaluator then completed.
The retrieved archive has SHA256
`7eb53ec50a454630f5461c82593322830d9f995029809c359ff89585d671ef17`;
its evaluation manifest has SHA256
`37e43b315c9160b4d3525b4b6d7991881faacdae5629235361c64e837046725a`.
Independent audit verified every declared file/self-hash and all live source,
dataset, contract, replay, smoke, authorization, device, seed, completion, and
access bindings, then recomputed the raw metric grids.

Clean next-state relative L2 is `0.001181/0.001233/0.001190`. All 208-step
rollouts are finite and severe late, but their early train-state-scale window
errors `0.2669/0.2368/0.2458` all fail the strict `<0.15` gate. The frozen
verdict is `R0_REJECT_NACA_AS_HERO_CORRECTION_NECESSITY_CASE`. Native replay
also misses the strict quarter-margin in the uniform-node and near-body/wake
views. The small persistence terminal error at step 208 is a phase alias; the
registered window metrics govern. Prospective/sealed data were not opened, and
there was no horizon extension, corrector run, or post-outcome tuning.

### Inactive candidate-source compatibility

The 2026-08-30 audit independently rehashed current bytes rather than relying
on the historical prose:

- both Kolmogorov retained result files match their artifact manifests;
- the live Kolmogorov reference solver and reference test match the retained
  spatial and temporal source manifests;
- live shock fine/coarse solvers, family/shard code, and baseline evaluator
  match the retained D074 source manifest; and
- the current shock PCNO wrapper and trainer do not match that old manifest.

This does not promote either inactive candidate. Kolmogorov still lacks a PCNO
pipeline and resolved projection semantics; shock-vortex still lacks a trusted
arbitrary-state restart. The reverified state does not restore the former
shock-first implementation order.

## Manuscript State

`paper/main.tex` and `paper/sections/` now render a 34-page theory, ODE, and
primary-PDE manuscript:

1. Introduction;
2. Why one-step accuracy does not control rollout;
3. Corrective-mechanism framework;
4. Existing methods and framework predictions, including the conditional
   two-region hypothesis;
5. framework-guided corrector design;
6. mechanism calibration in exact and learned ODE maps;
7. the primary fixed-PCNO PDE case study; and
8. limitations and conclusion.

The primary-PDE section contains the distinct NACA successor's frozen problem
statement, diagnostics, five-arm method contract, signed predictions, and
verified open-development outcome. No prospective or sealed result has been
opened. The completed R0 negative remains documentation-only under its original
identity and is not shown in the active manuscript. Execution protocol, gates,
packet mechanics, source hashes, and
manuscript TODOs live only in maintained Markdown and artifacts.

The manuscript now integrates the verified extension outcome, its limitations,
and the two audited extension figures. The resulting 34-page build is clean and
the figure pages passed direct visual inspection.

The manuscript now defines on-reference defect, learned/trusted response,
response defect, corrective-mechanism classes, retention, and path accuracy.
The ODE section is organized by scientific questions and integrates the
requested two-row defect-landscape figure with actual training supports and
recomputed trajectories. Appendix A retains the completed deterministic proof
audit and its refinements; Appendix B
contains only supporting taxonomy and proxy limitations. The active draft no
longer presents the fixed-radius nonlinear pilot as PDE-style noise evidence.
It builds with no undefined references, overfull boxes, or visible planning
markup. The pre-tightening paper is preserved in
`paper/archive/pre_incisive_20260830.zip`; the immediately preceding nonlinear
draft and figure are preserved under
`paper/archive/pre_gaussian_correction_20260830/`.

The canonical ODE packet is `corrective_ode_study_20260830d`. Its manifest
SHA-256 is
`54d65dfe6f7186768d270453be780f5f1fe07d74c8042f607b3278e9439edb99`;
all 23 output hashes and all three packet-bound source hashes verify. The live
runner and focused test still match their recorded hashes; the maintained
`EXPERIMENT_PLAN.md` has since changed for the PDE successor, so the affine
packet is not claimed to be fully current-checkout compatible. Its bounded
scientific evidence remains valid under the recorded source identity. The
derived visualization packet is
`corrective_ode_landscapes_20260830e`, with manifest SHA-256
`256e175d8d77e352a2fb545e85ce4470e603a9af06cfb8370e1df03562f6ff0a`
and the same verified parent binding. Independent code-to-intent and visual
audits passed, as did all 16 focused tests and Ruff. Treat these results as
supporting calibration and mechanism realization only, never as an ODE-to-PDE
ranking claim.

The nonlinear stress packet is
`corrective_ode_nonlinear_stress_20260830a`, with manifest SHA-256
`8c1458d44dedb5dfa170d7e2acd7b84efb74541c2d87eaa49e0276373abf9f40`;
all 31 outputs and five current sources verify. Its frozen classification is
`response_qualification_failed`. DYN recovered the C/M/E response ordering
and every normalized in-tube fidelity gate passed, but one expansive model
missed the clean-trace/comparability gate and two mixed models narrowly missed
the near-zero response band. The P3--P5 rollout patterns are retained only as
descriptive internal evidence. Its discrete training radii are not Gaussian
noise and must not be presented as a PDE augmentation prescription. The packet
remains immutable. Dense landscapes and state-resolved trajectories are in the
separate parent-bound packet
`corrective_ode_nonlinear_diagnostics_20260830c`, whose manifest SHA-256 is
`84f06ee329112a88302fce595c4c8f7f39b0fb995201fd718e0eabaf27a1521c`.
It verifies 12 outputs, two current sources, all 15 checkpoint hashes, and all
nine selected learned/reference trajectory digests. Five focused tests, Ruff,
independent code-to-intent audit, and final visual review pass. The packet is
explicitly retrospective, non-Gaussian, and excluded from paper evidence. The
prior `...b` packet remains immutable provenance; `...c` is its palette-corrected
successor and matches the manuscript visual grammar.

The Gaussian normal-noise successor is complete under packet
`corrective_ode_gaussian_normal_noise_20260830a`, manifest SHA-256
`8902430e479caae18e877407cfbc40c8ddd231bb55dc310bd765990a5af46172`.
It verifies 72 outputs, 51 checkpoints, six current sources, the independent
audit, frozen nonlinear parent, and inherited forcing signs. Only the normal
coordinate is corrupted, with registered standard deviations
`{0.005,0.01,0.02,0.04}` and no clipping or resampling. RECOVERY and DYN both
pass their fixed-radius Gaussian density--fidelity signatures: median Spearman
is `1.0`, with `17/18` and `53/54` positive sampling units, respectively.
Larger local density is therefore associated with lower fitted error in this
laboratory. Clean fidelity and forced retention are not monotone in scale, and
no scale passes the stricter all-seed response gate. Every rollout outcome is
descriptive. The packet supports qualitative scale-response reasoning, not a
universal variance, a qualified rollout ranking, or ODE-to-PDE transfer.

No executed ODE packet contains genuine pushforward, multistep-loss, or
model-prefix-exposure training. `DYN-RELABEL` uses prescribed displaced inputs
and trusted displaced-state targets. The active PDE extension, rather than a
new ODE arm, now compares literal MP-PDE exposure with a separately named
curriculum/EMA protocol. The closed `DETACHED_PUSHFORWARD` result remains a
one-prefix stress test and does not reject multi-step training generally.

## Parked State

[PARKED_EXPERIMENTS.md](PARKED_EXPERIMENTS.md) is the current parking ledger.
It parks forward work in W26-L1–L5, architecture expansion, boundary/filter/loss
programmes, REALM/HydroGym execution, exhaustive corrector reproduction,
online/exogenous OOD work, broad PDE suites, and sealed/test openings.
The frozen deterministic `PATH_PROJECTION` operational corrector is part of
B4 parent evidence. The exact `PCNO_PDEREFINER_K3_VPRED` adaptation is active
under the extension; exhaustive learned-corrector reproduction remains parked.

Historical outcomes remain recoverable and claim-bearing at their bounded
scope. Parking caused no artifact deletion.

## Historical Replay Constraints

Tracked historical preregistrations and the detailed tracker remain available
for bounded provenance lookup. Their presence does not reactivate parked work.
Any identity whose exact historical bytes are missing from the checkout and
verified Git or artifact anchors remains unrecoverable and must not be
reconstructed under an old hash; future execution requires a new identity and
source manifest.

Rewriting the decision and tracker changes provenance-document hashes. Legacy
D094 tooling that hard-codes old hashes is expected to fail its old source
contract against the new checkout. Leave that tooling unchanged; tracked prior
document bytes are recoverable only from a verified Git or artifact anchor,
and any future replay needs a new registered identity.

## Immediate Owner Inspection

Recommended review order:

1. [RESEARCH_DIRECTION_DECISION.md](RESEARCH_DIRECTION_DECISION.md);
2. [PROJECT_PLAN.md](PROJECT_PLAN.md);
3. [paper/PAPER_PLAN.md](../../paper/PAPER_PLAN.md);
4. [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md);
5. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md);
6. [B3B4_NACA_CORRECTIVE_EXTENSION_PREREGISTRATION.md](B3B4_NACA_CORRECTIVE_EXTENSION_PREREGISTRATION.md);
7. [B3B4_NACA_CORRECTIVE_EXTENSION_CONTRACT.json](B3B4_NACA_CORRECTIVE_EXTENSION_CONTRACT.json);
8. [B5_BUMP_SOLVER_FREE_TRANSFER_PREREGISTRATION.md](B5_BUMP_SOLVER_FREE_TRANSFER_PREREGISTRATION.md);
9. [B5_BUMP_SOLVER_FREE_TRANSFER_CONTRACT.json](B5_BUMP_SOLVER_FREE_TRANSFER_CONTRACT.json);
10. [R0_PRIMARY_PDE_READINESS_CONTRACT.md](R0_PRIMARY_PDE_READINESS_CONTRACT.md);
11. [PARKED_EXPERIMENTS.md](PARKED_EXPERIMENTS.md);
12. [WEEKLY_RESEARCH_PLAN.md](WEEKLY_RESEARCH_PLAN.md).

Then inspect `paper/main.pdf`, especially the ODE calibration and primary-PDE
case study,
together with the verified affine landscape packet under
`artifacts/time_dependent_no/corrective_ode_landscapes_20260830e/` and the
separate nonlinear diagnostic packet under
`artifacts/time_dependent_no/corrective_ode_nonlinear_diagnostics_20260830c/`,
then inspect the Gaussian scale summary and figure under
`artifacts/time_dependent_no/corrective_ode_gaussian_normal_noise_20260830a/`.
For the completed PDE screen, inspect
`artifacts/time_dependent_no/naca_pcno_r0_results_20260901a/runs/evaluation/result.json`
and the 2026-09-01 closeout entry in `EXPERIMENT_TRACKER.md`; the large archive
is retained under `artifacts/time_dependent_no/` with its verified hash.
Open-development artifact, manuscript, and exact mounted-checkout WSL
verification are complete for the five-arm parent. The extension pilot, bank,
training, common evaluation, corrected presentation derivative, and manuscript
integration are also complete and verified. Its two main-paper figures passed
direct visual inspection in the 34-page build. If the trusted
recovery--relabel response claim is retained, freeze an evaluation-only assay
on identical displaced bank inputs before executing it. A prospective reveal
remains a later decision.
The verified parent development result finds finite rollouts throughout;
uniform paired gains from
`PATH_PROJECTION`, helpful `IID_RECOVERY`, an error-subspace arm that improves
`CLEAN` but not IID recovery, and a falsified, seed-unstable detached-prefix
mechanism. Treat the zero post-projection path residual only as a construction
check. Do not infer physical off-manifold drift or arbitrary displaced-state
SU2 response. The immutable replay
`development_replay_v2_81646368` contains `rollout_replay.npz` at SHA256
`8def4971abb669339f9a8ea5d5b44f8a6d0ddf792f0745cb893c03c460e73bd3`
and `replay_manifest.json` at SHA256
`f509b21c5fb448d2aa56a5ac135bdb0d85f337f3be7873b49c94ee3838a8120c`.
The historical visualization `development_visualization_v2_81646368` has
canonical manifest payload
`2887f7fbf79489ff674e97f2bcac0d4397f4f70a24b18236a881e8b497c46daf`
and 11 rendered outputs. It remains valid provenance. The final
presentation-only packet is `development_visualization_v3_e2eccdbf`; it
contains the same replay bytes, pins the historical producer, and replaces the
paper's two PDE PDFs with embedded non-Type-3 versions. Static figures use the
verified development evaluation; the three seed animations are qualitative
only. Every protected-access flag is false. The then-current 35-page parent
`paper/main.pdf` has
SHA256 `79f64c09030a85160777bea72af896b7f98d141bc081c893398807c6993f6c66`;
the detailed packet ledger is in `EXPERIMENT_TRACKER.md`.

Do not extend the closed R0 horizon, tune its baseline after outcome, or
reinterpret that negative identity. Preserve the complete R0 packet only as
documentation-level evidence for the narrower discrepancy/structure result;
it is not part of the active manuscript. The audited
portable `prospective_freeze` is complete; its final-manifest SHA256 is
`eef941053e549aaa4b5e2fd080ba0fbd73eed4bda4095a3d941f76d4e607217f`.
The owner-only provisional folder `prospective_freeze_private_acl_c5efa3f4`
and the pre-hardening `prospective_freeze_stale_pre_hardening_09b470be` are
preserved only as superseded provenance.
Prospective and sealed NACA values remain unopened, and a separate owner
decision is still required before either reveal. Shock-vortex remains a possible tangential/
no-harm control; no fallback run follows automatically.
B2-GN is complete; do not rerun, tune, or reinterpret its descriptive rollout
rows under the closed identity.

Before execution, an agent must audit each implementation against its intended
target, information source, composition order, recurrent state, and identity
ablation, and must ask the owner when scientific intent is ambiguous.

## Privacy And Authorization

- `LOCAL_CONTEXT.md` remains ignored, private, unquoted, and uncommitted.
- No private host, credential, machine path, dataset location, or unpublished
  manuscript content may be sent to an external AI without explicit approval.
- The closed R0 work downloaded and audited the pinned public NACA tutorial resources,
  ran local CPU tests, executed two final pinned native replays, generated the
  local 499--1999 solver trajectory, completed its model-blind phase analysis,
  froze and audited the PCNO population/baseline, materialized the bounded open
  dataset, and completed the authorized AutoDL smoke, three-seed training,
  evaluation, retrieval, and independent packet audit. It did not open
  prospective/sealed values or run a corrector under the R0 identity. The
  distinct successor has closed calibration, smoke, all twelve learned-arm
  training packets, and the independently verified development result packet.
  The extension has separately closed its pilot, bank, 18 training packets, and
  Attempt-E development result. Protected populations remain unopened, and no
  completion statement for either lineage authorizes a reveal.
- B5 currently authorizes only its declared local Stage-0A/B and synthetic
  implementation scope. A remote or dataset-scale run requires a separately
  selected compute resource and launch receipt; the historical bump test is
  not authorized.
