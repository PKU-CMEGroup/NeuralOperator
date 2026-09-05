# Active Experiment Tracker

Updated: 2026-09-05

Status: current block and authorization tracker. Historical evidence remains in
`MECHANISTIC_DIAGNOSTIC_TRACKER.md`. This file does not authorize execution.

## Status Vocabulary

| Status | Meaning |
| --- | --- |
| `PLAN_REVIEW` | Design exists and awaits owner review. |
| `IN_PROGRESS` | A named, bounded work scope is authorized and active. |
| `IN_PROGRESS_OPEN_ONLY` | Only open-role work is authorized; the row states whether it is active or complete, and protected reveal is not authorized. |
| `PLANNED_NOT_AUTHORIZED` | Scientifically selected; no implementation or execution approval. |
| `BLOCKED_BY_PREREQUISITE` | A named readiness condition is absent. |
| `EVIDENCE_ONLY` | Historical evidence, not a prospective active block. |
| `PARKED` | Preserved outside the critical path. |
| `COMPLETE` | The exact block and required packet are complete. |

Authorization is separate from scientific status.

## Active Blocks

| Block | Claim role | Status | Authorization | Next admissible action |
| --- | --- | --- | --- | --- |
| T0 documentation and paper skeleton | C1/C2 framing | `COMPLETE` | Documentation/manuscript edits only. | Owner inspects active files; fix any scientific-language objection. |
| R0 NACA first screen | C2 prerequisite | `COMPLETE` with negative delayed-failure decision | The frozen NACA screen completed and passed independent packet audit. | Preserve the exact R0 identity and verdict; use it only as immutable parent evidence. |
| [NACA corrective successor](B3B4_NACA_CORRECTIVE_SUCCESSOR_PREREGISTRATION.md) | Primary fixed-PCNO PDE case | `COMPLETE` at open-development scope | Development evaluation, replay, final presentation packet, portable freeze, manuscript, and exact mounted-checkout WSL verification are complete; protected reveal is not authorized. | Owner inspects the evidence and separately decides whether to reveal prospective outcomes. |
| [NACA corrective extension](B3B4_NACA_CORRECTIVE_EXTENSION_PREREGISTRATION.md) | Missing B4 mechanism contrasts | `COMPLETE` at open-development scope | All pilot, bank, source, smoke, training, common-evaluator, retrieval, packet-audit, corrected-visualization, and claim-bounded manuscript gates closed under `B3B4_NACA_CM_EXT_20260902A`; protected roles are closed. | Owner inspects the manuscript. Freeze a separate evaluation-only mechanism assay before claiming identical-bank trusted-response fidelity. |
| B1 exact ODE | C1 boundary and measurement calibration | `COMPLETE` | Closed under the frozen B1 contract. | Preserve the verified packet and use only as supporting calibration. |
| B2 learned ODE | Supporting mechanism realization | `COMPLETE` | Closed under the frozen three-seed/four-arm B2 contract. | Freeze the implementation and packets; continue paper writing without inferring ODE-to-PDE ranking transfer. |
| [B2-NL fixed-support nonlinear pilot](B2_NL_NONLINEAR_ODE_STRESS_PREREGISTRATION.md) | Historical response-regime stress | `COMPLETE` | Closed under the frozen local-CPU contract; no PDE or remote work occurred. | Preserve the packet unchanged. Its fixed-radius support is excluded from the active draft and may be visualized only as an internal diagnostic. |
| [B2-GN Gaussian normal-noise successor](B2_GN_GAUSSIAN_NORMAL_NOISE_PREREGISTRATION.md) | Supporting scale-response study | `COMPLETE` | Closed under the audited normal-only local-CPU contract. No PDE, GPU, remote, or sealed work occurred. | Preserve the packet and use its scale-response results only as supporting ODE evidence. No rollout ranking is mechanism-qualified. |
| B3 primary-PDE diagnosis/freeze | Main C2 diagnosis | `COMPLETE` | NACA successor open-population diagnosis, signed predictions, audit, and source-bound freeze are closed. | Preserve the verified packet and its claim boundary. |
| B4 primary-PDE interventions/reveal | Main C2 test | `COMPLETE` at open-development scope | Parent and extension experiments, corrected figures, and claim-bounded manuscript integration are complete on open development roles. Prospective and sealed evidence remain unopened and unestablished. | Owner inspects the evidence and separately decides whether any protected reveal is warranted. |
| [B5 Supersonic-Bump comparison](B5_BUMP_SOLVER_FREE_COMPARISON.md) | Secondary transfer/no-harm evidence | `IN_PROGRESS`: comparison implementation | Stage 0B passed on AutoDL (32/32 above rank 7; median rank 58). Owner authorizes the solver-free comparison; focused CPU tests pass. | Close full-mesh GPU smoke, launch the fixed paired pilot, and retrieve all outcomes. Historical test remains sealed. |

## Candidate Readiness

| Candidate | Verified live state | Missing gate | Status |
| --- | --- | --- | --- |
| SU2 Unsteady NACA0012 | Stage 0 through audited R0 evaluation completed. Every rollout is finite and severely wrong late, but all three fail the old early-accuracy threshold (`0.2669/0.2368/0.2458 >= 0.15`). | The delayed-failure phenotype is absent; state-space drift and ripple cause remain unmeasured. | Preserve `R0_REJECT_NACA_AS_HERO_CORRECTION_NECESSITY_CASE`; later corrective results belong only to the distinct successor/extension identities. |
| Turbulent square cylinder | Public SU2 case material exists. | A pinned, reproducible production configuration and all downstream gates. | Unselected candidate; owner/mentor choice and a new contract are required. |
| Laminar von Karman cylinder | Official unsteady SU2 tutorial route exists. | R0 population, PCNO, and transverse-failure qualification. | Cheap solver/control fallback, not assumed to be the hero case. |
| Shock-vortex / supersonic bump | Retained infrastructure and retrospective evidence; B5 now has an executable solver-free comparison amendment. | No trusted arbitrary-state restart; only stored trajectories supply training targets. | Bump's paired AutoDL pilot is owner-authorized. Shock-vortex remains a control candidate; historical test stays sealed. |

NACA0012 has completed a valid negative R0 delayed-failure screen. Its distinct
successor and extension now provide completed open-development evidence for the
narrower fixed-PCNO question of how corrective mechanisms change recurrent
response, path, structure, and rollout. The visible ripples are not assumed to
be manifold-normal. Prospective diagnostic selection remains untested.

## Historical Evidence In The Current Story

| Evidence | Use |
| --- | --- |
| D094 | Retrospective one-step/rollout discordance and self-composition motivation; never prospective C2 evidence. |
| M1 reference packets | Finite-grid Kolmogorov parent evidence only; old residual-FNO model stage and failed population attempts do not authorize continuation. |
| P0 native-coarse restart | Explains why the old shock coarse map is not a trusted displaced-state reference. |
| P1 path-conditioned tube | Solver-free vocabulary and synthetic plumbing only; not `Phi(u+eta)` fidelity. |

## Owner Decisions Still Required

- [x] Select SU2 Unsteady NACA0012 as the first R0 screen and conditional
      primary case after the bounded candidate review (2026-08-31).
- [x] Rewrite the NACA R0 solver/state/failure/population/resource contract and
      freeze the public-resource manifest (2026-08-31).
- [x] Approve the exact B1 ODE system and compact B2 arm set (2026-08-30).
- [x] Approve the separate B2-NL nonlinear response-regime stress test and
      local CPU execution after independent audit (2026-08-30).
- [x] Approve the exact B2-GN Gaussian covariance and scale-sweep contract
      (normal-only, 2026-08-30);
      report both standard deviation and variance, and do not use a long-rollout
      outcome to select the displayed scale.
- [x] Issue the complete NACA R0 verdict and do not promote it as the primary
      PDE (`R0_REJECT_NACA_AS_HERO_CORRECTION_NECESSITY_CASE`, 2026-09-01).
- [x] Select NACA as the primary case for the distinct corrective successor and
      approve open-population implementation/execution (2026-09-01).
- [x] Approve the B3 representative intervention set and signed predictions in
      `B3B4_NACA_CM_20260901A` (2026-09-01).
- [x] Approve successor implementation, resource smoke, and bounded open-role
      training/evaluation (2026-09-01).
- [x] Approve `B3B4_NACA_CM_EXT_20260902A` implementation and train/development
      execution, with the full relabeling bank gated by the train-only SU2
      pilot (2026-09-02).
- [x] Select the Supersonic Bump for bounded solver-free B5 secondary transfer
      and authorize local Stage-0A/B contract, analyzer, and synthetic-test work
      only (2026-09-05).
- [x] Select AutoDL, complete the B5 train-only audit, and authorize the solver-free
      comparison under `B5_BUMP_SOLVER_FREE_COMPARISON_20260905A`.
- [ ] Decide whether to approve the prospective reveal; the portable prediction
      packet is closed, but its existence is not reveal authorization.
- [ ] Approve any sealed/test access as a still later, separate decision.

## Attempt E Development Result

The evaluation-only retry for `B3B4_NACA_CM_EXT_20260902A` completed under
final-manifest SHA256
`15475da78b4182754fe959eadc175434ffca18ef1558a45ea06cbf9c93ca7370`.
It reused the 18 Attempt-D training packets without mutation. Source tests,
GPU smoke, evaluation, raw-table recomputation, NPZ CRC/shape/finiteness, and
local/remote hash-parity checks passed. All values below are development-only:
each seed is the median over the same eight anchors, followed by the descriptive
median `[minimum, maximum]` over seeds 17, 29, and 43. The range is not a
confidence interval.

| Primary deployment | Rollout AUC | Late-window error |
| --- | ---: | ---: |
| `PATH_PROJECTION`, corrected | 0.01453 [0.01144, 0.01519] | 0.02477 [0.01950, 0.02530] |
| `PCNO_PDEREFINER_K3_VPRED`, EMA sampler-median | 0.64862 [0.63807, 0.69877] | 0.97480 [0.94732, 1.04325] |
| `DETACHED_PUSHFORWARD` | 1.01607 [0.83671, 1.36423] | 1.81917 [1.39480, 3.77995] |
| `PAIRED_RECOVERY` | 1.06803 [0.97535, 1.15693] | 1.69931 [1.54134, 1.93490] |
| `DYNAMICS_RELABEL` | 1.09442 [1.01955, 1.32361] | 1.71306 [1.67811, 2.95696] |
| `CURRICULUM_EMA_PUSHFORWARD_K13`, EMA | 1.15916 [0.94547, 5.29715] | 2.03362 [1.51489, 22.24117] |
| `CLEAN_EMA`, EMA | 1.26954 [1.08723, 289.58284] | 1.87478 [1.68988, 838.24174] |
| inherited `CLEAN` | 1.28764 [1.02310, 1.37085] | 2.21296 [1.55229, 2.21709] |
| `MP_PDE_PUSHFORWARD_M01` | 1.64431 [1.05054, 2.41758] | 4.92573 [1.75293, 8.13172] |

The result supports three development-case conclusions. First, one-step
ranking does not determine rollout ranking: `CLEAN_EMA` seed 43 has essentially
the same horizon-one error as inherited clean but 283-times larger AUC, whereas
PDE-Refiner has 5.36-times larger central horizon-one error than clean but about
half its AUC. Second, path projection ranks first and PDE-Refiner second on both
primary metrics at every seed; paired recovery is the safest one-call learned
intervention. Third, literal MP-PDE and curriculum exposure are heterogeneous,
so this experiment does not reject pushforward or multistep training as a
family.

The mechanism evidence is narrower. Recovery preserves clean fidelity and
reduces one-prefix response without validity harm. Relabeling lowers the same
gain only with large clean-path and optimization harm. The evaluator did not
run the trained recovery and relabeling models on identical bank inputs against
both stored targets, so trusted displaced-response fidelity remains untested.
The one-prefix gain is not a stability certificate: it misses the coherent late
threshold failure of `CLEAN_EMA` seed 43. PDE-Refiner improves each reverse step
near the clean path but is not a monotone projection at late reached states.
All numerical states are finite, yet six deployment/seed variants develop
nonpositive density; finite and physically admissible are therefore distinct.
Prospective and sealed populations remain unopened.

## Execution Log

- 2026-09-05: The first B5 comparison smoke (`5ce08f7`) passed all 23 remote
  synthetic tests, then stopped before scientific array access because source
  collection treated PyTorch's virtual `_classes.py` filename as repository
  source. The attempt is retained as infrastructure-only. The collector now
  ignores relative virtual filenames, with a regression test; all 24 focused
  CPU tests pass. A new source snapshot and smoke attempt are required.

- 2026-09-05: B5 Stage 0B completed successfully on AutoDL in 229 seconds.
  The retrieved packet independently closes at final-manifest SHA256
  `479eaa3c28fd7118dc291cb779a0d27b96977e6cad7a00b60e358567505a9bf3`.
  All 32 training trajectories exceed rank 7 at 99.9% variance; median is 58,
  range 49--62. Rank 7 captures median 90.6%, so this is a long spectral tail,
  not proof of high intrinsic dimension. The owner then requests the solver-free
  intervention comparison. Its new amendment freezes five training runs and
  eight deployed systems, including a shared Clean-EMA control and mapped PCA
  ranks 7/32. Twenty-three focused synthetic CPU tests and Ruff pass before
  full-mesh GPU smoke. Generated launch/results belong under the ignored
  `b5_bump_comparison_20260905a` artifact directory.

- 2026-09-05: owner selected the Supersonic Bump for a bounded second-PDE
  transfer/no-harm study and clarified that NACA's projection ranking is a
  consequence of its special nearly linear low-rank path, not a universal
  conclusion. `B5_BUMP_SOLVER_FREE_TRANSFER_20260905A` freezes train-only
  per-trajectory native-mesh temporal POD as a diagnostic and declares four
  intended solver-free systems. Only Stage 0A/B local implementation and
  synthetic tests are currently executable. Stage 0C and Stage 1 require
  identity/algorithm amendments; remote execution has no named resource and
  the historical test remains sealed.

- 2026-09-04: a presentation-only v3 derivative rerendered the extension's
  exact replay (SHA256
  `79886d205f21ed64ae9727a0a70ad3692ae1962bad221fbb488800da618878ec`)
  as unfiltered piecewise-linear native-node fields on a deterministic quad
  triangulation. Scientific arrays are unchanged: the checkerboard in the old
  flat-cell view was a rendering artifact. The presentation-contract and
  render-manifest canonical SHA256 values are respectively
  `35efce1ac222312ec3bc8a3db8905995152c086a0fb791c3273c77d17c9136c0`
  and `4562919f14b24e9b925a9b0bfaf0c599700af78ac06cfcd8cbeb7ccb90b1d0f2`;
  the corrected horizon-208 PDF is
  `54c87cf2995eacb224b9346e133e88c03be24eba750d9a0572bd3e24deb3d5bf`.
  A separate train-only packet, `naca_dataset_pca_20260904b`, analyzes the 238
  train-current states in 72,880 normalized model coordinates. The first
  `2/4/6` PCs explain `96.1294/99.8007/99.9809%` of variance; the complete
  BDF2 view is nearly identical, while area weighting needs seven PCs for
  `99.9%`. This supports strong low-rank, path-like variance concentration,
  not an intrinsic manifold dimension. Its final-manifest SHA256 is
  `e83cd7aaeedcc24b2cefc1683da222f6730af7ed3bb702e470650766750916f7`
  and summary canonical SHA256 is
  `f536320d111587e0988d2b582114231782ca4d76b06d4a3fbe88444e44f9d9ff`.
  Both figures are integrated in the private 34-page manuscript (SHA256
  `966c2aad4b02d01c198555008e4db87d3ec583390461d05df4a782758c9e5f95`),
  which has no undefined references/citations, overfull boxes, Type-3 fonts,
  or unembedded fonts. Prospective and sealed populations remain unopened.

- 2026-09-03: the corrected extension presentation derivative
  `naca_corrective_extension_visualization_results_20260903b` passed hash,
  format, and direct visual audit. Its presentation-receipt canonical SHA256 is
  `7be542ef9ead7d8f7835686450795aaf32b944fe87a6a4f0eb42c398bd208b62`;
  its remote parent receipt is
  `07360658a8f8e085515bb69550e2e6fada17a9f21f503cbac7513bc81cb6ee53`,
  render-manifest canonical SHA256 is
  `12b3054ffbdebf047e8e63d4957217889697d0ad40902a2cd2d71f204d8857a2`,
  and exact replay SHA256 is
  `79886d205f21ed64ae9727a0a70ad3692ae1962bad221fbb488800da618878ec`.
  The two exact-snapshot PDFs are integrated in the clean 33-page manuscript;
  its SHA256 is
  `d7ed11861fec76148131f8b81881f0a2feb67d069630d7d90b98dc736b4eece6`.
  The three animations are independent qualitative rerolls. Scientific replay
  and evaluation are unchanged, and every protected-access flag is false.

- 2026-09-03: Attempt E closed the extension's evaluation-only retry and the
  exact result packet above. The local metadata retrieval contains all 126
  non-checkpoint JSON files from the 18 training packets; their hashes,
  self-hashes, source bindings, and final-manifest identities verify. No
  checkpoint was duplicated locally. Any new common-bank response assay
  requires a frozen evaluation-only contract before execution.

- 2026-09-02: the extension trainer/evaluator source closure and final
  code-to-intent audit passed. A first audit correctly blocked execution because
  the trainer used relabel-receipt semantics without binding that module's
  bytes; the repaired closure now includes and live-rehashes it. The trainer
  file/source-set SHA256 values are
  `0d79bfa9d5312770d81574d6d30020d3a708a3332a77b367a5625f4a609ad852`
  and `9a2911d11e19d9d3a1f0dba4866d8964fe412380b1d0a46bae45dacb51694f8a`;
  the evaluator file/source-set values are
  `f9409fb5edcba78f69636e62e6329b64df06b7b63f8a4e2725bab57623cdfa8f`
  and `46670860fb9eb1e1271449dc6cb86350f5163eaf954837122c1c3d6380e6b1e4`.
  The minimal 37-file deployment/test archive is `292,836` bytes at SHA256
  `3ece0808eb82b53940353479dc4d0d86a4abc2d31eed61cd460237ad986dd6ea`;
  a clean extraction reproduces both source-set digests exactly.
  The final integrated extension partition passes `75` tests; separate R0 and
  parent compatibility partitions pass `45` with four skips and `11`
  respectively, and Ruff check/format-check pass. The evaluator recomputes all
  extension deployments plus inherited `CLEAN`, `DETACHED_PUSHFORWARD`, and
  `PATH_PROJECTION`; no scalar result is spliced from the parent evaluator. No
  extension GPU training or development evaluation has occurred.

- 2026-09-02: the complete train-only paired displaced-BDF2 bank closed with
  all `476/476` fixed-SU2 calls successful, no redraw or dropped case, every
  per-field realized-scale ratio inside its frozen bounds, and no protected
  access. The final-manifest and compact-NPZ SHA256 values are respectively
  `9202d9738ce2549918eb548b3ca2cd78541c8b892e9a6959883b4f4ec9e04293`
  and `04bb27b633930874890f2bd34d8acd0c60dcc9c1e520aac1b395979eb607fc53`.
  An independent streaming pass reverified all `478` registered files. This
  unlocks paired training; it is not a learned-model result.

- 2026-09-02: corrected pilot R1 completed all `11/11` registered train-only
  SU2 calls and passed replay, convergence, admissibility, repeatability,
  auxiliary-invariance, displacement-scale, zero-control, response-separation,
  and authority-immutability gates. Its final-manifest and pilot-receipt SHA256
  values are respectively
  `f55dbab47943d34f0a34ccc40f54bab2f3676faa9bee28f29b5d7a10868b5603`
  and `a5a388c4718b1ce8a78ce6cf36482a7289b035fed84bc8c7b0e54c9695d00aa4`.
  The earlier configuration-only attempt is preserved under final-manifest
  SHA256 `6b1ab039a47d91e42325fdc53f97f230c3387fde315df1920a6a9ccf4f55e8b3`;
  it produced no numerical SU2 result and is not scientific evidence.

- 2026-09-02: `B3B4_NACA_CM_EXT_20260902A` was frozen as a distinct
  train/development-only extension. It preserves the closed five-arm parent and
  registers literal `MP_PDE_PUSHFORWARD_M01`, separate
  `CURRICULUM_EMA_PUSHFORWARD_K13`, paired frozen-bank `PAIRED_RECOVERY` versus
  `DYNAMICS_RELABEL`, and `PCNO_PDEREFINER_K3_VPRED`, with `CLEAN_EMA` as the
  EMA control. Focused CPU tests and independent code-to-intent audit precede
  execution. The then-pending 476-input paired bank was gated on the eleven-call
  train-only SU2 restart pilot passing every registered replay, convergence,
  admissibility, determinism, scale, and response-separation gate. No solver
  case could be silently dropped. Solver labels are offline only; no prospective,
  sealed, online-solver, or online-defect-trigger access is authorized. This
  registration is not an implementation or result claim.

- 2026-09-02: `B3B4_NACA_CM_20260901A` reached local open-development
  closeout. The development evaluation remains bound by final-manifest SHA256
  `09e3ae7e5896039e2230073c1ead5e5f9808c0cc9b47d0b15ff337b26b3e0ed1`.
  Its exact archived evaluator is
  `development_source_snapshot/evaluate_pcno_naca0012_successor.py` at SHA256
  `2630a7548de796ff1f11a23ed395ae454f2df87c1d6fed0d2d706c4769a7de8c`;
  the snapshot manifest is
  `2ca73afd4b44be7af0af3b3ee876b71b82ed4b393daf362231b0a215e62f6ce6`,
  and the development source-set identity is
  `df7949e9a1eb81dcb0af4e95e07a14d23c9f4abd205eac8f950dcbe5e453ff13`.
  Independent verification closed the declared inventory, hashes, source/data/
  checkpoint bindings, metric cardinalities, identity parity, and protected-role
  flags. Every evaluated rollout is finite. `PATH_PROJECTION` improves both
  primary metrics in all 24 paired anchor-seed comparisons and improves the
  pressure and graph-Dirichlet diagnostics; its zero post-correction path
  residual is true by construction and is not outcome evidence.
  `IID_RECOVERY` helps relative to `CLEAN`; `ERROR_SUBSPACE_RECOVERY` improves
  `CLEAN` but not `IID_RECOVERY`. The registered `DETACHED_PUSHFORWARD`
  mechanism prediction is falsified and its rollout behavior is seed-unstable.
  These are open-development learned-map/corrector findings, not evidence that
  visible flow features are physically off-manifold or a trusted SU2 response
  assay at arbitrary displaced states.

  The immutable replay `development_replay_v2_81646368` contains
  `rollout_replay.npz` at SHA256
  `8def4971abb669339f9a8ea5d5b44f8a6d0ddf792f0745cb893c03c460e73bd3`
  and `replay_manifest.json` at SHA256
  `f509b21c5fb448d2aa56a5ac135bdb0d85f337f3be7873b49c94ee3838a8120c`.
  Historical `development_visualization_v2_81646368` remains valid provenance:
  raw manifest SHA256
  `bc070b378ebe006e1fbaaf7796980b7a86fb133b1b9d5d35f15c9660702fe49b`,
  canonical payload
  `2887f7fbf79489ff674e97f2bcac0d4397f4f70a24b18236a881e8b497c46daf`,
  and producer
  `816463686ff7e53df8787a5cc31dc3cac8d731d40858820b2d6f986935a640bd`.
  Its earlier `corrective_tradeoffs.pdf` and
  `density_signed_error_h208.pdf` hashes are
  `4a996fca73cbc527c97a92b819c0a2272a8b9c5a355c2ef5eedf8b7680b52e72`
  and `dedca6a1cde51aa4957a531dd2d0b7ed76dbbe497eeffffcb188c2a7e442b0f8`.

  Final presentation-only packet `development_visualization_v3_e2eccdbf` has
  exactly 14 files and 11 rendered outputs. Its raw manifest SHA256 is
  `ea27126baf53c0621c7c55ae865f0bb65f55f40b9f8e551e499153bfe050d309`,
  canonical payload is
  `f6bbbacf5c400a6ec4d173b7e543285b970b1b61d763baa046a2a111220da776`,
  and current renderer is
  `e2eccdbf2619a1966bb939f53a0d38cb472cf4a18f5e1e31b0ac390323ece867`.
  It explicitly pins the historical producer above and contains byte-identical
  replay files. Final `corrective_tradeoffs.pdf` and
  `density_signed_error_h208.pdf` have SHA256
  `0e4304fab66c04dbae2a4c08ffd9fe09e4fdf6bbf0547bcae37a33b02419f572`
  and `b24332f34534efb0615522d71854885247a7b6e0f10d11f070a34b0544c3aacd`.
  All eight PDFs contain no Type-3 fonts; all three MP4 files validate. Static
  figures use verified development evaluation, while animations are qualitative.

  The canonical portable `prospective_freeze` contains exactly three files.
  Its raw final-manifest, freeze-file, source-manifest, canonical freeze-payload,
  and evaluator source-set SHA256 values are respectively
  `eef941053e549aaa4b5e2fd080ba0fbd73eed4bda4095a3d941f76d4e607217f`,
  `afaa206c6a6a449e48d41a090adbb055f2fe9824255ba7ae57f86af7b030fa6a`,
  `6aafec1880dbeee0a0c8762fcab341f8520d1fbfb26afa2655303935c2620a1c`,
  `e0f56dbe219fbaade339db24eb1907c055e7d77c8d8d95297dba4549aafbfdfb`,
  and `1cd504a1482e07100ac8ab63818ee919da089b5d396fa5c9bc350b36104a4f62`.
  Its current evaluator is
  `fbe8e61328fdbeec5ca9a28855d9f47f20d5ec2fbed46f53ffe975b8a26d29a0`.
  Inherited/readable ACLs and false prospective/sealed flags are verified.

  Two earlier freezes are preserved only as superseded provenance. The
  owner-only `prospective_freeze_private_acl_c5efa3f4` has raw final-manifest,
  freeze-file, source-manifest, payload, and source-set SHA256 values
  `c5efa3f435eb8ed5831c69c09675afcc30cb35622df4a2931c3ba4e7b1e7a28c`,
  `028e11e26659093e7f245a6e0810b245ebd33460b8e5d450322fe9e263557a17`,
  `f448118ad02e07aaa5a512dfbc7493edcb0ac7a11cce2c959daecae016e3e081`,
  `8d0c2845d33916481ea23379a0a1c28f0b5cecdb1338002818327e8c86eba8c3`,
  and `d343f10736b60047808def780b35fa6fbcb737b394083959e4767d285b6e29f9`.
  Pre-hardening `prospective_freeze_stale_pre_hardening_09b470be` has raw
  final-manifest, freeze-file, source-manifest, payload, and source-set SHA256
  values
  `09b470bef70f6362b52d9249eed27de9c5380985e63903c66fa7958f43eeca09`,
  `8307f9e66a3139ea439949f9093705ab3a5daadaeba566a822b97e4f0e034488`,
  `6c952c672e593d338d4debd2fe796d6eb449b2a704bc005c21387b95c00902d4`,
  `f8921371b0d8e2cc9b3f8fcc2238a2d04ff8a32e214890ce143e371cf9b35934`,
  and `bb66ec8ba904ab71bcb9c0a65181c9809ec232014456e60861106afdf708f8a2`.

  The integrated `paper/main.pdf` has 35 pages and SHA256
  `79f64c09030a85160777bea72af896b7f98d141bc081c893398807c6993f6c66`.
  It has no undefined references/citations, overfull boxes, `VERIFY` markers,
  Type-3 fonts, or unembedded fonts. Windows 68 focused tests and Ruff
  check/format-check pass with the same explicit `E402` exception described
  below for the deliberate provenance-first imports.

  Exact mounted-current-checkout WSL Ubuntu verification also passes under
  Python `3.10.12`, Torch `2.8.0+cpu`, pytest `9.0.2`, and Ruff `0.12.12`.
  Pre/post SHA256 identities match for the evaluator
  (`fbe8e61328fdbeec5ca9a28855d9f47f20d5ec2fbed46f53ffe975b8a26d29a0`),
  visualizer
  (`e2eccdbf2619a1966bb939f53a0d38cb472cf4a18f5e1e31b0ac390323ece867`),
  evaluator test
  (`75e75a09622edd3c57f5bf5d27852a1a8bcbb9dbccd60b5704145387953907f4`),
  visualization test
  (`a7d2bdfa36a7a44045fe41427a2f92df694df1959da05322bfae4b2a0ddc0cd3`),
  and successor utility
  (`8f788a25b25895fa2ad48dfc437ef051b051d3548ad2d340ef0e7370a3180cca`).
  Ruff format-check passes for the four source/test files. Ruff check passes
  with explicit `--ignore E402`; those imports follow deliberate source-byte
  provenance capture. All 68 focused tests pass in `20.46 s`. Both exact WSL
  temporary directories were deleted and their absence verified.

  An earlier AutoDL exact-four-file attempt verified those four file hashes but
  could not run checks because its dependency closure was incomplete. It is not
  a Linux test pass; its temporary root was deleted. No protected population,
  data, or checkpoint was opened. No prospective authorization artifact exists;
  prospective and sealed populations remain unopened, and C2 prospective
  generalization is unclaimed.

- 2026-09-01: the frozen NACA PCNO R0 screen completed on one RTX 5090 under
  CUDA 12.8. A full-resolution forward/backward/optimizer smoke passed, followed
  by exact sequential seeds `17/29/43`, each completing 100 epochs with best
  checkpoint at epoch 100, and the unchanged development-only evaluator. The
  dataset final-manifest SHA256 is
  `1cb5fd2d751bdf1ca27cde5e1a1457973a525c19a98dbadd7cc3738c3efe2e4b`;
  training final-manifest SHA256 values are
  `5fc86d129ca456aed1a8b8870020f93d5e0bf3e3d41881ec5ea3ef87418f3071`,
  `3078accaf90c7c0c2cd653171ca6f938ec42bfe1e01b23bb2f22f8711699d020`,
  and `0ab746f1ca30f72471146165ee6bbfc2dd20f7978e52bc4d62b3cc12680c8e50`.
  The result archive is `900,044,171` bytes with SHA256
  `7eb53ec50a454630f5461c82593322830d9f995029809c359ff89585d671ef17`;
  evaluation manifest SHA256 is
  `37e43b315c9160b4d3525b4b6d7991881faacdae5629235361c64e837046725a`,
  and canonical result payload SHA256 is
  `11aac10b5fb7781eeb64a0e3fb5d09ef8ce30f02e271a20de0cfdaa64d9b3630`.
  Independent audit rehashed all 32 declared files and 21 JSON self-hashes,
  verified the live source, dataset, contract, replay, smoke, authorization,
  device, seed, completion, and access bindings, and recomputed the raw grids.
  Clean next-state relative L2 is `0.001181/0.001233/0.001190`; early
  train-state-scale window error is `0.2669/0.2368/0.2458` versus persistence
  `0.3364`; late error is `1.8564/1.7023/1.7319` versus persistence `0.3365`.
  All runs remain finite and all satisfy the severe-late gate, but none passes
  the strict early `<0.15` gate, so the frozen decision is
  `R0_REJECT_NACA_AS_HERO_CORRECTION_NECESSITY_CASE`. Native replay also misses
  the strict quarter-margin in uniform-node and near-body/wake views. The tiny
  persistence error at terminal step 208 is a near-period alias; registered
  window metrics control interpretation. No horizon extension, prospective or
  sealed access, corrector run, or post-outcome retuning occurred.

- 2026-08-31: before any PCNO population materialization or model outcome, the
  single-attractor baseline was frozen in
  `R0_NACA_PCNO_BASELINE_CONTRACT.json`, SHA256
  `94069e65ef520d31860735b6c17f2b7515da1c4acd34d68518a2a16b3181fa87`.
  Independent audit recomputed the 32 phase anchors, source crossings, maximum
  phase-rounding error, 238 train/development transitions, complete BDF2 and
  208-step containment, six-period ratio, and 16-channel input. It also closed
  windowed non-aliased gates, persistence advantage, failed-rollout accounting,
  qualified shoelace measures, five-state pressure-only force diagnostics,
  train/development-only access, precision/reproducibility, and the paired
  replay-margin rule. Prospective/sealed states remain unopened; no dataset,
  checkpoint, training, or scientific model result was produced.

- 2026-08-31: before scientific analysis, the fixed-system attractor pilot was
  frozen and then clarified in `R0_NACA_PHASE_PILOT_CONTRACT.json` (current
  SHA256 `f442065172beb014ebcd4f3823abe1e2764fb2b74bbad348a8dd94d6b4c903f5`).
  It uses only solver states/history, requires eight stationary recurrent
  cycles, freezes extension/resource-cap outcomes, and treats 32 phase anchors
  as finite quadrature on one attractor rather than independent trajectories.
  No PCNO artifact or outcome may enter this decision.
- 2026-08-31: the audited trajectory runner completed output indices
  499--1999: 1,501 native states, `3,326,514,699` bytes, and ordered-state
  aggregate SHA256 `9adbcfd53c4812009d3c5e5081730b709a2f63f48237760390e2c8559ab200ef`.
  The run took `5631.17 s` (`0.2665` written states/s); the short one-step
  throughput above is therefore not used as a production-rate estimate.
  Receipt SHA256 is
  `3d857d6ddca4dca111ecc716b2c85ff23f727a4fafc97d39f24db340d2b2672b`.
- 2026-08-31: the unchanged audited analyzer returned
  `PHASE_ONLY_SUPPORTED` on the full trajectory. The CL clock gives period
  `34.6543` steps; 31 complete cycles remain after the selected burn boundary.
  Uniform-node recurrence has median/q90 component-balanced errors
  `0.003845/0.004076`; near-body/wake has `0.003827/0.004038`. The required
  physical-area secondary view fails (`0.110995/0.146430`, q90
  cycle-to-template `1.255897`) because its farfield-weighted orbit amplitudes
  are extremely small; it did not qualify or rescue the decision. Analysis
  SHA256 is
  `b14ef6d66c5e520633ea94954ff3b243a47c3185743de43a5ff10d1c7f74069f`.
  A preceding sandbox-only temp-directory permission failure is preserved as
  `INVALID_ARTIFACT` receipt SHA256
  `e5f4196fa98cfcb17b7ccc8e2874ef5e5ce436e643ce40c8d16c43988c2f2dc4`;
  it consumed no packet data and is infrastructure evidence only.
- 2026-08-31: after 47 focused tests, Ruff, and three independent audits, final
  native cases `su2_naca0012_native_replay_20260831_c` and `_d` both completed
  execution/evaluation under the pinned SU2 `v8.5.0` identity. Their outputs
  are byte-identical (`a39f0b4a...7fab`); the five evolved fields have
  component-balanced relative L2 discrepancy `4.3264e-7` with physical-area
  weights, `5.9447e-4` with uniform-node weights, and `8.9081e-4` in the fixed
  near-body/wake box against canonical target 499; observed local throughput is
  `1.92`--`1.96` transitions/s.
  The comparison packet revalidates both receipts and reports zero
  native-to-native numerical error. The model-relative quarter-margin,
  arbitrary-state restart, population, support, and PCNO gates remain open; no
  corrective-mechanism outcome is implied. Qualification addendum
  `R0_SU2_NACA_NATIVE_REPLAY_QUALIFICATION.json` records that the canonical
  target has no producer-bound integrated coefficients, freezes the common
  target/replay surface functional and auxiliary-field scope, and projects
  storage without rewriting the immutable C/D receipts; its SHA256 is
  `8269354d4a021b921f6c1d99c4f0dd416896b62c23fb5f9beb77fab0f0e960d6`.
- 2026-08-31: the preparation-only NACA replay harness passed 14 focused tests,
  Ruff, and an independent fail-closed audit. The regenerated ignored preflight
  stages only histories 497/498, keeps canonical target 499 external, freezes a
  distinct output stem and non-compact restart schema, and records all native
  solver claims as false. No SU2 executable or scientific evaluator was run.
- 2026-08-31: owner authorized an honest SU2 attempt after asking whether any
  time-dependent benchmark was clearly better. The bounded review found none
  with a better combined scientific fit and completion risk, so Unsteady
  NACA0012 was frozen as the first R0 screen and conditional primary case.
  Stage 0 independently bound and validated the public license, configuration,
  mesh, and restart triplet 497/498/499, including native binary layout, mesh
  coordinates, complete RANS--SA evolved state, and BDF2 history. No native
  solver replay, dataset generation, PCNO training, or PDE result occurred.
  The next milestone is the non-destructive native replay harness and its
  independent code-to-intent audit.
- 2026-08-30: final ODE evidence synthesis reverified every canonical B1/B2,
  landscape, nonlinear, Gaussian, and derived-diagnostic packet against current
  sources. The final-paper ODE core is the exact separation results, learned
  affine target comparison, and affine defect landscape. The Gaussian study
  contributes only its registered density--fidelity association and failed
  all-seed scale gate. Fixed-support nonlinear rollouts, nonlinear diagnostic
  plots, Gaussian planar atlases, and state-only proxy negative controls remain
  documentation or visualization assets rather than paper evidence. No new ODE
  run, post-outcome retuning, optimal-scale claim, rollout ranking, or PDE
  transfer claim is opened.

- 2026-08-30: ODE result-to-claim closeout found no claim-critical additional
  arm. CLEAN, RECOVERY, DYN-RELABEL, and `CLEAN+C_0` already isolate clean-only
  supervision, embedded return, trusted displaced-state supervision, and a
  separately attributable operational projection; the exact study also covers
  finite correction strength and the recovery--relabeling crossover. kNN and
  learned samplers would duplicate approximate projection in this known-circle
  geometry, while pushforward and the relabeling--recovery hybrid are reserved
  for the primary PDE study. The ODE implementation and verified packets are
  frozen; visualization scripts are retained for later aesthetic revisions.

- 2026-08-30: B2-GN closed under canonical packet
  `corrective_ode_gaussian_normal_noise_20260830a`, manifest
  `8902430e479caae18e877407cfbc40c8ddd231bb55dc310bd765990a5af46172`.
  Independent verification closes 72 outputs, 51 checkpoints, six exact current
  sources, the audited parent, and the exact forcing-sign inheritance. Six
  focused tests and Ruff pass. The fixed-radius density--fidelity signatures
  pass for RECOVERY (median Spearman `1.0`, `17/18` positive) and DYN (median
  `1.0`, `53/54` positive). No scale passes the all-seed response gate: most
  misses are the preregistered cross-regime clean-error ratio, and seed 29 also
  exceeds the `2e-3` DYN-M clean-error ceiling at `sigma_train=0.005` and
  `0.02`. All rollout rows therefore remain descriptive. At the largest scale,
  `115/576000` Gaussian draws exceed the declared `|r|=0.15` tube, close to the
  theoretical tail, while none exceeds the `0.5` safety radius. No threshold,
  scale, architecture, or outcome was changed after reveal.

- 2026-08-30: owner authorized B2-GN with normal-coordinate Gaussian input
  corruption only. The frozen standard deviations are
  `{0.005,0.01,0.02,0.04}`; clean/noisy rows are mixed 64/64 with paired
  fresh-per-update draws, and recurrent forcing remains a separate Rademacher
  assay. Implementation and the bounded local-CPU run may proceed only after
  focused tests and an independent code-to-intent `AUDIT_PASS`.

- 2026-08-30: palette-corrected nonlinear diagnostics were regenerated under
  successor packet `corrective_ode_nonlinear_diagnostics_20260830c`. It uses
  the manuscript's `magma`, reference, trusted-path, and categorical arm
  colors while preserving the exact parent computations. The prior `...b`
  packet remains immutable historical provenance.

- 2026-08-30: the parent-bound visualization packet
  `corrective_ode_nonlinear_diagnostics_20260830b` was generated after visual
  review and independently verifies 12 outputs, two current sources, all 15
  checkpoints, and nine learned/reference trajectory digests under manifest
  `942459c972a4935fe89dd95bf055dac78cb3552db1ba5cb864886a0ab9dd36db`.
  Its three regime atlases show trusted-flow defect, clean-return error, output
  distance to the clean set, actual fixed support, and learned/trusted paths;
  its trajectory diagnostic shows retention and visited-state defects. It is
  explicitly retrospective, non-Gaussian, and excluded from active paper
  evidence.

- 2026-08-30: owner clarified that fixed radii near `0.15` are not the intended
  analogue of PDE noise augmentation and directed a Gaussian-input successor
  with an explicit perturbation-scale hyperparameter. B2-NL remains a valid frozen
  record of its exact deterministic-support contract, but it is no longer
  active draft evidence. The missing nonlinear defect landscapes, trajectories,
  and visited-state defects are recoverable from its verified checkpoints as
  a separate read-only diagnostic packet. A new training run requires B2-GN;
  old samples will not be relabelled as Gaussian.

- 2026-08-30: B2-NL closed under canonical packet
  `corrective_ode_nonlinear_stress_20260830a`, which independently verifies 31
  outputs and five current sources under manifest
  `8c1458d44dedb5dfa170d7e2acd7b84efb74541c2d87eaa49e0276373abf9f40`.
  Three final audits returned `AUDIT_PASS`; nine focused tests and Ruff passed.
  The frozen scientific classification is `response_qualification_failed`:
  one seed's expansive DYN clean error was `2.421e-3` against `2e-3`, its
  cross-regime clean-error ratio was `4.207` against `2`, and two mixed-regime
  log-gains were `-0.525` and `-0.529` outside the `|lambda| <= 0.5` band.
  All signed target effects and all nine normalized in-tube DYN fidelity gates
  passed. Rollout descriptors matched P3--P5, including C/M/E residence
  `1.000/0.998/0.229` at `sigma=5e-3`, but remain descriptive by contract.
  No thresholds, architecture, or training settings were changed after reveal.

- 2026-08-30: owner authorized proceeding to the harder learned-ODE setting
  after inspecting the completed defect landscapes. B2-NL was preregistered
  under a separate identity; its exact source freeze remains pending the
  independent audit, while the verified B1/B2 packets remain current-source
  compatible. The test varies trusted clean-manifold normal response across
  contractive, mixed, and expansive regimes and separately scores robustness
  to the clean path and fidelity to a same-forcing trusted path. No PDE,
  remote, dataset, or sealed action is included.

- 2026-08-30: generated the requested learned-ODE defect-landscape
  visualization before any harder setting. The read-only derived renderer
  verified the canonical B1/B2 parent, reconstructed its training bank, and
  evaluated all nine checkpoints on a common phase--radius grid. The final
  packet corrective_ode_landscapes_20260830e binds parent manifest
  54d65dfe6f7186768d270453be780f5f1fe07d74c8042f607b3278e9439edb99
  and has manifest
  256e175d8d77e352a2fb545e85ce4470e603a9af06cfb8370e1df03562f6ff0a.
  Independent code-to-intent and visual audits passed. The two heatmap rows
  deliberately separate trusted-flow defect from clean-return error; neither
  is an ID/OOD label or online rule.
- 2026-08-30: tightened the manuscript around the corrective-mechanism thesis.
  Execution protocol, packet mechanics, readiness gates, placeholder PDE
  prose, and manuscript TODOs were removed from the rendered paper and retained
  in project Markdown. The ODE section now motivates each assay, defines both
  defect landscapes, explains why noiseless labels do not imply exact fitted
  response coefficients, interprets the reached trajectories, and states the
  ODE-to-PDE limitation explicitly.
- 2026-08-30: B1/B2 closed. Independent code-to-intent review returned
  `AUDIT_PASS`; 13 focused tests passed. Canonical packet
  `corrective_ode_study_20260830d` verified 23 output hashes and three current
  source hashes under manifest
  `54d65dfe6f7186768d270453be780f5f1fe07d74c8042f607b3278e9439edb99`.
  Every exact identity/inequality and every frozen learned mechanism check
  passed. CLEAN had the smallest mean clean one-step error among learned
  predictors but one of three seeds exited the tube catastrophically;
  RECOVERY learned return, DYN-RELABEL learned trusted displaced response, and
  `CLEAN+C_0` separated explicit retention from normal-dynamics fidelity. The
  continuous-phase metric is explicitly a lifted-coordinate path error, with
  chordal state-space error reported separately. The result is supporting ODE
  evidence only; no PDE action was taken.
- 2026-08-30: owner authorized completion of the ODE part and corresponding
  manuscript sections. B1/B2 were frozen before implementation: exact
  scenarios, learned data/target contracts, architecture, seeds, updates,
  checkpoints, metrics, signed predictions, null-ID-rollout handling, and
  artifact verification are now explicit. This authorization is local and
  ODE-only; primary-PDE selection remains deferred.
- 2026-08-30: owner clarified the two-axis thesis: clean one-step supervision
  probes on-reference evolution, while accurate rollout also requires a
  transverse retention/recovery property supplied by an embedded or operational
  corrective mechanism; neither small clean error nor tube retention alone is
  sufficient. The on-reference/tangent observation is background, not the
  novelty claim.
- 2026-08-30: primary-PDE selection was deferred pending the owner's mentor
  discussion. Shock-vortex was reclassified provisionally as a tangential/
  no-harm control. Kolmogorov, balanced rotating shallow water, and one of
  Lambda--Omega/Cahn--Hilliard remain literature candidates only. No candidate-
  specific implementation or scientific execution is authorized.
- 2026-08-30: owner accepted the plan and officially started the project. R0
  source/artifact audit and contract drafting began. No solver, test, dataset,
  model, checkpoint, training, remote, prospective, or sealed action was
  performed. Candidate-specific implementation remains gated by the mentor/owner
  PDE decision and review of the rewritten candidate-specific R0 contract.
- 2026-08-30: documentation and manuscript structure only. Exact prior plans
  and paper sources were archived. Live candidate source/result manifests were
  checked. No solver, model, checkpoint, scientific array, training, remote,
  download, or sealed/test action was performed.
