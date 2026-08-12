# D088 W26-L4 REALM Benchmark Contract Preregistration

Status: D088 allocated to W26-L4 on 2026-08-12 after the P1a source review and
noncollision check. P1a and the restricted-internal-research P1b
train/validation reference replay are complete and all P1b gates passed. On
2026-08-12 the owner authorized P1c A3-smoke only and selected an alternate
personal GPU workstation because AutoDL will be occupied for 48 hours. The
contract below was frozen before the first P1c model output. P1c completed with
all gates passing. The direct-baseline A1 trainer and synthetic CPU gates are
also complete. The owner then authorized the exact A3 direct-baseline attempt;
it stopped at the registered step-100 decoded-nonfinite validation gate. Test
objects remained absent, no completed direct baseline exists, and the
direct-versus-residual comparison remains unauthorized.

## Question And Scope

This line asks whether predicting a normalized-state increment improves
long-horizon IgnitHIT validation rollouts over a matched direct-next-state FFNO.
P1a and P1b do **not** answer that question. They independently fix and replay
the contracts that must exist before a fair reproduction or comparison can run:

- metadata-only dataset identity and sealed-test exclusion;
- Box-Cox/z-score preprocessing and exact recurrence algebra;
- paper-average and released-source-sum metrics as different quantities;
- decoded structure, invariant-domain, boundedness, boundary-band, and spectral
  diagnostics; and
- synthetic CPU gates for every implemented semantic.

P1a may create only:

1. this preregistration;
2. `utility/time_dependent_no/realm_benchmark.py`;
3. `scripts/time_dependent_no/audit_realm_benchmark.py`; and
4. `tests/time_dependent_no/test_realm_benchmark.py`.

It cannot copy official REALM source, download any repository object, inspect a
test object, use a checkpoint or real trajectory, contact AutoDL, train a model,
allocate a stable result ID, or make a scientific performance claim.

## Frozen Primary Bindings

| Object | Frozen identity | Role |
| --- | --- | --- |
| paper | `arXiv:2512.18595v2`; retained PDF SHA-256 `148971e6eef3782f1562b0eaa045ce8608ded673ffa13c0bfd99acbc39f5b137` | paper-formula contract |
| official source | `deepflame-ai/REALM` commit `7d00523dbda7823efa03c20be36692c947a417b5` | historical source behavior only; no code is copied |
| IgnitHIT release | `realm-bench/realm-bench-IgnitHIT` revision `a0736b4d8c6c58a2688127e32addc30085e824c3` | first staged case |
| open manifest | canonical entry SHA-256 `85e8dcdaf4a5f7726ac5a7ad18a1bc131785fc1212acc9318ff69b94be965205`; normalized JSON file SHA-256 `2a35e6551ba68ed473932f3458e8d5cf54fa046cd768afd42e6af8042e7dc0d3`; canonical JSON payload SHA-256 `0eb711bd3fb170244855271fe250387d17cfa6a9070e436b87130254753cf0cf`; 552,023,019 bytes | exact metadata plus 26 train and five validation trajectories |
| sealed payload | five test trajectories; 89,545,596 bytes | must remain absent through training and selection |

The public code tree and dataset cards do not provide a verified license grant.
That remains an acquisition/reuse blocker unless the authors clarify it or the
owner records a separate restricted internal-research disposition. Public
visibility alone is not treated as permission.

The released IgnitHIT channel order is exactly:

`H, H2, H2O, H2O2, HO2, O, O2, OH, T, rho, Ux, Uy`.

The registered contiguous metric groups are `chem=0:8`, `T=8:9`, `rho=9:10`,
`u=10:12`, and an absent `p=12:12`. Missing pressure is omitted from a grouped
metric; it is never imputed as zero.

## Manifest And Split Contract

The metadata-only audit input is JSON schema
`realm_huggingface_manifest_v1` with exactly:

```json
{
  "schema": "realm_huggingface_manifest_v1",
  "repository": "owner/repository",
  "revision": "40-hex-git-commit",
  "entries": [
    {"path": "relative/posix/path", "size": 0, "oid": "...", "lfs": false}
  ]
}
```

Each path is canonical, relative POSIX text. Sizes are nonnegative integers.
An LFS entry requires a 64-hex SHA-256 OID; every other entry requires a 40-hex
Git object ID. Duplicate paths fail. Canonical manifest bytes are UTF-8 lines
`path<TAB>size<TAB>oid<LF>` sorted by path. The exact repository, revision,
canonical digest, and byte total must all match.

Any path component equal to `test`, case-insensitively, fails before digest
acceptance. A trajectory path of the form `data/<split>/...` permits only
`train` or `val`; another split fails. The audit command reads the supplied JSON
only. It performs no network request and opens no referenced object. Its JSON
result contains no local absolute path.

Canonical source, data, configuration, and runtime mappings use sorted compact
JSON, UTF-8, no NaN/Infinity, followed by SHA-256. P1b must bind the actual
source-file hashes, package/runtime flags, and downloaded-object hashes to those
canonical mappings before an array is decoded.

## Preprocessing Contract

The rollout/evaluator tensor contract is `[case, call, channel, y, x]`; train
statistics reduce case, frame, and both spatial axes, preserving only channel.
Only the 26 registered train trajectories may enter those statistics.
Validation and sealed test values are prohibited.

For each of the first eight species channels, with an explicitly supplied
positive clamp `epsilon` and `lambda=0.1`,

`B(x) = ((max(x, epsilon))^lambda - 1) / lambda`.

The implementation also supports the tested logarithmic limit at `lambda=0`.
Inverse decoding uses

`B_inverse(z) = (lambda * z + 1)^(1/lambda)`.

A negative inverse base is either a hard failure or a reason-coded NaN under an
explicit diagnostic policy. A zero base is valid for positive `lambda` and
decodes to zero. There is no silent absolute value, clipping, or complex/even
power repair.

Per-channel transformed train means and standard deviations use PyTorch's
sample standard deviation (`correction=1`). The source-compatible scale policy
is `scale=1` when `std < 1e-10`, otherwise `scale=std+1e-10`. Encode and decode
are `(B(U)-mean)/scale` and its exact inverse; non-species channels skip Box-Cox.

The historical source uses `epsilon=1e-40` while its runtime encoder defaults to
`1e-8`. P1a tests both but does not invent which generated the paper row. P1b
must freeze one primary epsilon after the train-only schema/statistics replay,
before any model output or validation score is computed. The alternative is a
separately named source-sensitivity channel.

## Recurrence And Exposure Contract

Let `Z` be the normalized state and `x` the same static coordinate tensor at
both calls. The two registered parameterizations are:

- direct: `Z_hat_next = F(Z_current, x)`;
- residual: `Z_hat_next = Z_current + R(Z_current, x)`.

The residual is a transformed-normalized state increment, not a conservative,
flux, or PDE residual. A zero residual returns the current state exactly.

`one_call` performs one learned transition. `two_call_final` performs call one
inside `no_grad`, detaches its returned recurrent state, feeds that exact state
to call two, and exposes only call two to gradient-based loss. Both calls use
the identical static coordinates. The recurrence API has no truth/target
argument, so future truth cannot be fed back through this surface.

The released trainer's nominal two-step option actually executes one learned
call. It is reported only as released-source `one_call`. A genuine
`two_call_final` run is separately named and must be applied to both arms.

## Metric Semantics

For case `i`, returned call `h`, and each present group `g`, define

`m(i,h,g) = mean_(channel in g, y, x) (Z_hat - Z_truth)^2`

and `q(i,h) = sum_g m(i,h,g)`. Truth must be finite. Any nonfinite prediction
element makes its affected group/call and total error `+Infinity`; it is not
dropped or converted to a finite mean.

| Name | Exact aggregation | Interpretation |
| --- | --- | --- |
| `realm_npe_mean` | `mean_i mean_h q(i,h)` | primary paper-formula compatibility channel |
| `realm_npe_sum_source` | `mean_i sum_h q(i,h)` | pinned released-source compatibility channel |
| group curves | `m(i,h,g)` without early cross-case pooling | localization evidence |
| `corr_decoded` | spatial Pearson per decoded case/call/channel, observable channels then calls within case, then equal case weight | structural paper channel |

Pearson entries are reason-coded `ok`, `truth_nonfinite`,
`prediction_nonfinite`, `constant_truth`, or `constant_prediction`. Unobservable
channels are not coerced to zero. If a whole case has no observable entry, its
case value and the population result are null rather than silently dropping the
case. Counts and reason codes accompany any later aggregate.

All scientific summaries are case-first. No accepted-prefix or available-case
pool may change cohort membership with horizon.

## Decoded Diagnostic Semantics

These diagnostics remain distinct from accuracy, conservation, and general
physical validity.

| Diagnostic | Frozen operational definition | P1b resolution |
| --- | --- | --- |
| admissibility | native finiteness; each released species `>=0`; `T>0`; `rho>0`; `p>0` only when present | all 31 open trajectories pass the released-state check; omitted species and pressure still block a stronger composition/pressure claim |
| boundedness | inclusive `abs(channel) <= expansion_factor * train_quantile(abs(channel))` | train-only `quantile=1.0`, expansion `10`; exact channel limits are bound in `normalizer_arrays.npz` |
| boundary band | inclusive distance to any Cartesian domain edge `<= physical band_width`; report decoded boundary/interior MSE and ratio | `dx=dy=0.00039269961416721344` native units; width `0.0015707984566688538`; units and centering unresolved |
| front high/low regions | `field>=high` and `field<=low`; transition band is strictly `low<field<high` | train-only linear quantiles: `T=[595.7160034179688,2480.06005859375]`, `OH=[1.0604499376705017e-22,0.013010700233280659]` |
| front area/length | grid-cell area proxy `count*dx*dy`; neighbor high-mask crossings weighted by `dy` across x and `dx` across y | native spacing resolved; physical units and whether released values are cell- or point-centered remain unresolved |
| front position/thickness/strength | gradient-magnitude centroid; transition area/interface length; high mean minus low mean; missing interfaces/regions reason-coded | train-only thresholds and reporting fields (`T`, `OH`, or both) |
| spectrum | full 2-D `fft2(norm="ortho")`; radial cycles per physical unit; half-open bands except inclusive last edge; final edge covers the grid maximum; Parseval closure | non-demeaned native-coordinate edges `[0,159.1547272908376,318.3094545816752,636.6189091633504,1273.2378183267008,1800.630190803951]`; physical units unresolved |

The area and interface quantities are operational grid proxies until the actual
released grid centering/extent contract is verified. Released-species
nonnegativity is not a complete species simplex or elemental-conservation test.
Boundary-band error is descriptive and cannot establish boundary-condition
learning without deployable boundary/forcing provenance. Lower high-frequency
energy alone cannot establish improvement if position, thickness, or strength
regresses.

## Provenance And Comparison Matrix

| Field | Direct arm | Residual arm | Current status |
| --- | --- | --- | --- |
| paper/source/dataset identities | exact bindings above | identical | resolved at metadata level |
| train/validation population | pinned 26/5 split object manifest | identical | P1b opened all 31 open trajectories; exact inventory passed |
| sealed test | absent | absent | mandatory |
| channel/group contract | 12 channels and slices above | identical | resolved |
| normalizer axes/formula | train-only contract above | identical | resolved |
| Box-Cox epsilon | explicit runtime argument | identical | primary `1e-8`; separately named `1e-40` source-statistics sensitivity |
| architecture/capacity/init | FFNO-M reconstruction | exact paired tensors | unresolved until P1c |
| optimizer/history/seed/budget | matched reconstructed baseline | identical | original paper values partly unresolved |
| target parameterization | direct next normalized state | normalized-state increment | registered factor |
| recurrence/exposure | exact deployed state; named one/two-call mode | identical except addition algebra | resolved synthetically |
| precision/runtime | exact future model runtime manifest | identical | P1b CPU replay bound to Python 3.12.2, NumPy 2.5.0, PyTorch 2.8.0+cu128; model runtime remains unresolved |
| evaluator | this independent contract plus source hashes | identical | synthetic and real-data reference semantics passed |
| selection | validation only; test unavailable | identical | five validation trajectories opened only for identity replay; no model selection executed |

The absence of the original IgnitHIT seed, numerical weight decay, presentation
history, immutable command, checkpoint, and training log blocks an exact paper
reproduction claim. A reconstruction can be source-faithful to declared fields
without being the original run.

## A1 Verification Gates And Stop Rules

P1a goes only if all of the following pass on CPU synthetic fixtures:

1. float64 Box-Cox/inverse closure, clamp/zero behavior, and inverse-domain
   failure policy;
2. train-only reduction axes, sample-std, zero-variance policy, and
   encode/decode closure;
3. exact channel slices and paper-average/source-sum identities;
4. direct, zero-residual, exact two-call detachment/static-input semantics, and
   no truth parameter;
5. canonical manifest ordering/hash/OID checks, exact open split allowlist, and
   hard test-path rejection;
6. case-first errors/correlations, reason-coded unobservable channels, and
   nonfinite propagation;
7. admissibility and inclusive boundedness;
8. front translation, boundary-band separation, and FFT/Parseval/band
   invariants;
9. CLI `--help`, focused pytest, and Ruff on only the three Python files; and
10. a final privacy/diff check proving that only the four authorized files are
    staged and no local path, credential, data, array, checkpoint, or result is
    included.

Stop P1a on any failed semantic, hidden data/network access, official-source
copy, test-path acceptance, noncanonical manifest behavior, or need to alter a
shared user-owned file. A passing synthetic suite is software-contract evidence
only; it is not a benchmark result.

### A1 implementation closeout

The frozen implementation sources are:

| File | SHA-256 |
| --- | --- |
| `utility/time_dependent_no/realm_benchmark.py` | `479c82e05bb668a1e7e94780a977dea3af9387e6440da0a5851f83b3050166b5` |
| `scripts/time_dependent_no/audit_realm_benchmark.py` | `32d0ee21c92f7a3bd0a52e383803bf00afc2f6c3689bf99bab47edf2551ebb25` |
| `tests/time_dependent_no/test_realm_benchmark.py` | `41bba6e310376280bc5209709c0cea13b7da13f7af4e1d513f56661ea20957e8` |

On 2026-08-12, the focused CPU suite passed `39/39`, Ruff reported no
violations on the three Python files, and the standalone audit CLI `--help`
returned successfully. No real manifest, dataset object, model, checkpoint,
network endpoint, remote host, or GPU was used. These outcomes close P1a's
software gates only.

## Artifacts, Cost, And Next Gate

The four registered files are the complete long-lived A1 artifact set. Tests
may create one task-specific pytest temporary directory, which is removed after
verification. P1a has cost class `XS`: CPU synthetic only, no network/storage
payload and no GPU.

### P1b frozen real-data replay values

The following choices were frozen after inspecting only the three released
metadata/statistics objects and one smallest train trajectory, before opening a
validation trajectory or producing any model output:

- primary train/runtime Box-Cox clamp: `1e-8`, matching the released runtime
  default and used consistently for both fitted statistics and encoding;
- separately named source-statistics sensitivity clamp: `1e-40`; it cannot be
  pooled with the primary channel or selected from validation performance;
- boundedness envelope: per-channel train maximum absolute decoded value
  (`quantile=1.0`) with an inclusive `10x` expansion factor;
- temperature and OH front bands: pooled train-only decoded `q=0.10` and
  `q=0.90`, with the numerical values recorded before any model execution;
- boundary band: four released Cartesian coordinate spacings from every edge,
  inclusive; it remains a native-coordinate band until coordinate units and
  point-versus-cell centering are authoritative;
- coordinate uniformity: median-spacing relative tolerance `3e-4`, frozen to
  admit observed float32 coordinate jitter while rejecting grid drift;
- spectra: full non-demeaned FFT energy in native cycles per released
  coordinate unit, with radial edges at `0`, `1/8`, `1/4`, `1/2`, and `1` times
  the smaller axis Nyquist, followed by the exact maximum radial grid
  frequency; and
- replay closure gates: float64 encode/decode/re-encode maximum normalized
  error `<=1e-10`, FP32 maximum normalized error `<=1e-4`, exact zero
  self-prediction grouped MSE, and decoded self-correlation within `1e-6` of
  one for every observable case/call/channel.

The one-trajectory clamp audit is pilot information only: it showed no negative
or zero released species in that file, but several species values below
`1e-8`, so the two historical clamps are materially different. The complete
train/validation audit remains the registered P1b result. No tolerance may be
changed after a validation trajectory is opened.

The registered P1b disposition is restricted internal research while public
source and dataset reuse terms remain unresolved. It permits metadata plus
train/validation acquisition of exactly 552,023,019 bytes at the pinned
revision and the schema/statistics/reference-metric replay in this document.
It does not establish redistribution rights, permit copying official source,
or authorize any test acquisition.

### P1b result and artifact inventory

Run `d088_realm_ignithit_p1b_20260812a` completed on AutoDL CPU on 2026-08-12.
The remote host had no outbound route, so the first acquisition attempt stopped
before completing any file and left no partial object. The identical frozen
manifest was then acquired locally, transferred as a closed tree, and rechecked
on AutoDL in idempotent mode: 34/34 files were reused only after their registered
Git-blob or LFS digest passed; zero bytes were downloaded remotely. This is an
infrastructure fallback, not a population or source change.

The real-data contract is `[30,12,128,128]` float32 per trajectory, with 26
train and five validation trajectories. The coordinates have shape
`[2,128,128]`, order `[y,x]`, and descend on both axes. There are 30 released
times from `1e-5` through `3e-4` at native cadence `1e-5`. Coordinate/time units
and grid centering remain unresolved. Train statistics used 12,779,520 samples
per channel. All 31 trajectories were native-finite and passed the operational
released-state admissibility check.

| Gate | Result |
| --- | --- |
| exact open inventory and sealed-test absence | pass; 34 files, 552,023,019 bytes, zero `data/test` files |
| metadata/schema and all open native states | pass; exact keys/order/shape/dtype and 31/31 finite/admissible |
| float64 normalized re-encode | pass; maximum error `2.44249065417534e-15` (`<=1e-10`) |
| float32 normalized re-encode | pass; maximum error `9.5367431640625e-7` (`<=1e-4`) |
| validation identity grouped errors | pass; both primary mean and source-sum maxima exactly `0` |
| validation decoded self-correlation | pass; minimum `1.0` across 1,740 observable entries |

No model or checkpoint was loaded, and the GPU was unused. Therefore D088 P1b
establishes only that the open data, preprocessing, metric, and diagnostic
reference contracts are executable. It provides no direct-versus-residual
performance result, paper reproduction, generalization result, conservation
result, or evidence about the sealed test population.

Before the final replay, the combined focused CPU suite passed `57/57` and Ruff
format/check passed on the six REALM Python files. A final review added a
fail-closed guard preventing the summary output from entering or overwriting the
input manifest, exact data root, or closed report directory. The corrected
source passed that guard on AutoDL and reran from a fresh report directory. Its
schema, statistics, normalizer-array, reference-replay, and runtime artifacts
are byte-identical to the preserved pre-guard attempt; this is same-data/runtime
replay evidence, not independent confirmation.

The final executed-source canonical manifest SHA-256 is
`9c7f9047c1c83342007999942314ea92d63446b48387d79a9f1e84a442fcb48d`.
Its four files are:

| Executed file | SHA-256 |
| --- | --- |
| `utility/time_dependent_no/realm_benchmark.py` | `479c82e05bb668a1e7e94780a977dea3af9387e6440da0a5851f83b3050166b5` |
| `utility/time_dependent_no/realm_ignithit.py` | `4a50c4901b592ec9363209e87fae03d171078a2ce0663e41ef4ce4b1e772312f` |
| `scripts/time_dependent_no/audit_realm_benchmark.py` | `463b7dab409f567ca1bd3e62eb7be4a2357b15b79bd48dcc80633c14c963057b` |
| `scripts/time_dependent_no/acquire_realm_ignithit.py` | `4acf17d44eb4185d8e97df917c129c65da2474721a9f77f7bb3ed1a15441fc5d` |

The compact report set is closed by
`reports/final_hash_manifest.json`, SHA-256
`70037f06bc6e04d00a4dcfb954b1a3cef7cbfdd7c0aa6abb8fe2036331341b53`:

| Retained compact artifact | SHA-256 |
| --- | --- |
| `reports/schema_report.json` | `d0497b0f3bfa39e7bfc0892062c43a23aa528a8cde168343b8be9462381fc3b3` |
| `reports/train_statistics.json` | `5b8c82fee3059edaa0ad36e4369db8ba28573cf35c1b12a658248f54191db5b9` |
| `reports/reference_replay.json` | `e17534d415f614ac3a02c0bc541555bf321cf95fcc6cb896df6c5781ce417538` |
| `reports/normalizer_arrays.npz` | `368d243b5e0f71b6380ee5f49fb9f5cf2724baa94ddece620d3ceae55285ca20` |
| `reports/source_manifest.json` | `a983822b45f33af4d63f72f091fac2f1a7dc2e151398a0ccafab59d5cca00f4c` |
| `reports/runtime_manifest.json` | `1fd14d3f2208ed252d6b5155f61a45f1df59358e1e9f8e915e123d3a795b03de` |

Raw trajectories and compact reports remain ignored artifacts and are not
committed. Public dataset/source reuse terms, original-run training provenance,
physical coordinate/time units, grid centering, and a model-runtime contract
remain unresolved.

## P1c Frozen Personal-GPU Engineering Smoke

P1c uses attempt label `d088_realm_ignithit_p1c_personalgpu_20260812a` under
D088. It is an engineering smoke, not a new stable scientific result ID.
Current explicit owner direction selects an alternate personal GPU workstation
for this attempt and supersedes the branch's default AutoDL resource only for
this smoke. Its machine-specific SSH alias remains private.

### Exact execution bindings

| Field | Frozen value |
| --- | --- |
| data | the exact P1b 34-object open tree; canonical entry SHA-256 `85e8dcdaf4a5f7726ac5a7ad18a1bc131785fc1212acc9318ff69b94be965205`; 552,023,019 bytes; no test object |
| normalizer | `normalizer_arrays.npz` SHA-256 `368d243b5e0f71b6380ee5f49fb9f5cf2724baa94ddece620d3ceae55285ca20`; primary Box-Cox/z-score channel contract from P1b |
| model | independent FFNO-M reconstruction; 12 state plus two static coordinate inputs; width 128; four residual factorized Fourier blocks; 32 modes per axis; factor-four two-layer pointwise MLP with ReLU and final LayerNorm; 128-wide GELU head; 12 outputs |
| parameter gate | target 8,936,500 within 0.5%; reconstructed exact expectation 8,936,460 |
| initialization | seed `20260812`; PyTorch linear/LayerNorm defaults and Xavier-normal real/imaginary spectral tensors; no checkpoint loaded |
| map and recurrence | direct next normalized state; one learned call per released transition; the returned normalized proposal is the next input without clipping, repair, teacher forcing, or future truth |
| coordinates | subtract each coordinate channel's own minimum and divide both by coordinate-channel-zero range |
| runtime | exactly one visible RTX 5060 Ti CUDA device on the owner-selected personal workstation; FP32; autocast and TF32 disabled; deterministic algorithms on; cuDNN deterministic on and benchmark off; `CUBLAS_WORKSPACE_CONFIG=:4096:8`; exact Python/PyTorch/CUDA/cuDNN/device properties retained |
| optimizer smoke | Adam with `lr=1e-3`, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay zero, AMSGrad/foreach/fused off; no scheduler or clipping |
| real train work | ordered 26 registered train cases, each frame 0 to frame 1; microbatch 1; accumulation 26; effective batch 26; exactly one optimizer step |
| validation work | first registered validation group `phi=_t_15_3_t`; start at frame 0; autoregressive H29 over all 29 matching-truth transitions; no selection or tuning use |
| hard cap | 900 end-to-end seconds, equal to 0.25 GPU-hour; external `timeout 900s` is also required |

The run order is fixed: exact source/data/normalizer/runtime preflight, one
synthetic `[1,12,128,128]` forward/backward memory probe, the one registered
optimizer step, then the H29 validation rollout. Stop before the next stage on
any source, data, split, normalizer, coordinate, shape, parameter-count, device,
precision, finiteness, recurrence, or budget failure. A returned nonfinite
normalized or decoded validation state is retained as the failure call and is
not fed back. An infrastructure failure before a returned state is not a
scientific stability event.

Retain only canonical config, input/source/runtime manifests, per-phase time and
peak-memory values, compact per-call validation diagnostics, hashes, stdout, and
stderr. Write no checkpoint. Report the two normalized error conventions,
decoded correlation, released-state admissibility, and 10-times-training-envelope
boundedness separately. These are engineering diagnostics from a random model
after one optimizer step: they cannot support baseline accuracy, convergence,
selection, stability, conservation, benchmark-superiority, or residual-method
claims.

The authorized committed surface is exactly:

1. this preregistration;
2. `docs/time_dependent_no/W26_L4_REALM_BENCHMARK_PLAN.md` for the owner-selected
   resource/status reconciliation;
3. `utility/time_dependent_no/realm_ffno.py`;
4. `scripts/time_dependent_no/smoke_realm_ignithit_ffno.py`; and
5. `tests/time_dependent_no/test_realm_ffno.py`.

The full direct baseline and direct-versus-residual comparison remain later,
separately named authorizations.

### P1c result and artifact inventory

Run `d088_realm_ignithit_p1c_personalgpu_20260812a` executed once on 2026-08-12
from pre-execution commit `67b84f1c124cf133776ccbd67ef01336c939ee61`.
Before transfer, Ruff format/check passed and the three focused REALM CPU files
passed `64/64` tests. The four executed source files, 34-object open tree, and
normalizer were independently rehashed after transfer. There were zero test
paths. A first detached-launch shell command failed during local/SSH quoting
before creating a session, report, log, or GPU process; it is infrastructure
history, not a second model execution. The corrected frozen command then exited
zero with empty stderr and all ten registered gates passing.

| Result channel | Verified value |
| --- | --- |
| model/runtime | 8,936,460 trainable parameters; FP32; PyTorch `2.11.0+cu130`; CUDA `13.0`; RTX 5060 Ti; deterministic/TF32/autocast bindings exact |
| synthetic probe | finite forward/backward; `0.327168` s; peak allocated/reserved `486,580,224 / 595,591,168` bytes |
| registered train work | exactly one optimizer step; effective batch 26; finite predictions/gradients/parameters; pre-step grouped loss `3.35034162`; `3.321702` s; peak allocated/reserved `522,123,776 / 664,797,184` bytes |
| H29 validation | 29/29 normalized-finite, decoded-finite, released-state-admissible, and 10-times-envelope-bounded calls; maximum recorded envelope ratio `0.12392933` |
| engineering accuracy | `realm_npe_mean=8.66303635`; `realm_npe_sum_source=251.22805786`; decoded case-first correlation `0.00706968` |
| wall time | `6.602508` s end to end, below the 900-second cap |
| privacy/artifacts | no test object opened; no checkpoint loaded or written; zero checkpoint-like files in the isolated target |

The one-step accuracy is deliberately reported but is not a baseline outcome.
Its poor error/correlation and simultaneous operational admissibility illustrate
why accuracy, finiteness, admissibility, and boundedness must remain separate.
Passing P1c establishes only that the exact open data, reconstructed FFNO map,
one-step optimizer path, direct recurrence, and H29 evaluator fit and execute on
this runtime.

The executed-source canonical manifest SHA-256 is
`a59f3dd1e2500f4cf2f3371e3d6af5d7c104f7513b057da349fafa6e991f2cce`.
The compact result set is anchored by
`reports/final_hash_manifest.json`, SHA-256
`931be66bbbcfbb29012e01895d2a19a9a1e17fdcc09cf34f514989e437d8a209`:

| Retained compact artifact | SHA-256 |
| --- | --- |
| `reports/config.json` | `351c92dc0e9a18f9545aaa60678cf2e8984485667c4b22121c9ef0324c124286` |
| `reports/input_manifest.json` | `28b8ccd8e544e14239a866451b9222a7a4897d6fd115cb5dc959297f29cfaba1` |
| `reports/source_manifest.json` | `42df6b50664780bd3e1d2fc40396f0f50f845cd73fe2e81fefe64477e86b5a7b` |
| `reports/runtime_manifest.json` | `abfb9aa8265d06e22ca309f4a634f3b821e1cb71ed3c15f6173a94abae3ba7b8` |
| `reports/smoke_metrics.json` | `f7994847c3a48cdd1873a4c7990fb187d0adea3357c87cf134d3a3cf5168ee09` |
| `reports/summary.json` | `3cc33270c5d11c7ade4fbff267be426b9a9391d5b39e019ca5e80fdc8f3ab258` |

Measured train-step time projects to `4.61` raw GPU-hours for 5,000 optimizer
steps and `18.45` raw GPU-hours for 20,000, before validation, checkpoint, and
I/O overhead. The first fits the current 12-hour cap; the second does not. These
are linear engineering projections from one step, not benchmark runtimes. Since
the selected paper row's iteration count and numerical weight decay are still
unresolved, P1c does not choose between those histories.

## P1c Direct-Baseline Declared Reconstruction

Status: frozen on 2026-08-12 before trainer execution or real-data training.
This section authorizes A1 source and synthetic CPU tests only. The future run
label is `d088_realm_ignithit_p1c_direct_seed0_5000_20260812a`; it remains an
attempt under D088 and is not allocated as a new stable result ID.

The primary paper establishes Adam with weight decay, OneCycleLR, batch 26,
nominal two-step rollout, and selected IgnitHIT `max_lr=1e-3`, but it does not
numerically disclose weight decay or iteration count. The pinned public source
defaults to seed 0, 5,000 iterations, Adam weight decay 0, and a 5,001-step
OneCycleLR. Its nominal two-step loop executes only one learned call and samples
starts 0--27, omitting the otherwise legal one-call pair 28->29. Therefore the
system below is a **declared released-source reconstruction with a corrected
one-call exposure domain**, not a paper-faithful reproduction.

| Field | Frozen reconstruction |
| --- | --- |
| model/data/normalizer | exact P1c FFNO-M, open 26-train/5-validation IgnitHIT tree, coordinate transform, and primary P1b normalizer |
| seed/runtime | seed `0`; one visible CUDA device; FP32; autocast and TF32 off; deterministic algorithms on; cuDNN deterministic on/benchmark off; `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| target/map | direct next normalized state; one learned call; group-summed normalized MSE over `chem`, `T`, `rho`, and `u`; no pressure channel exists |
| presentation budget | 5,000 optimizer steps; 26 trajectory-pair presentations per step; 130,000 total presentations |
| frame sampling | all 29 adjacent starts `0..28` have positive support; one common start per effective batch; no future truth enters the model |
| case ordering | begin from the retained ordered 26 train keys each step, shuffle with a dedicated Python MT19937 seeded `0`, then draw the common frame start; save and restore that RNG state |
| batching | microbatch 1; accumulate the mean of 26 case losses; effective batch 26; one optimizer/scheduler step after all 26 cases |
| optimizer | PyTorch Adam; `max_lr` argument `1e-3`; betas `(0.9,0.999)`; epsilon `1e-8`; weight decay `0`; AMSGrad/foreach/fused off; no gradient clipping |
| scheduler | PyTorch OneCycleLR; `total_steps=5001`; `max_lr=1e-3`; `pct_start=0.3`; cosine annealing; cycle momentum on; base/max momentum `0.85/0.95`; div/final-div factors `25/10000`; scheduler steps after each optimizer step |
| validation schedule | completed steps `1`, `50`, `100`, ..., `5000`; all five open validation trajectories; frame-0 start; H29 direct recurrence; matching released truth only |
| selection | strict improvement in case-first `realm_npe_mean`; ties retain the earlier checkpoint; source-sum NPE and decoded correlation are reported but not selected |
| checkpoints | `best.pt` is inference-only and contains the deployable model/normalizer plus exact provenance; `last.pt` is the distinct resumable state with optimizer, scheduler, ordering RNG, Python/NumPy/Torch/CUDA RNG, completed step, best step/score/model digest, and history; `last.pt` is committed first, so an interrupted improved-best write is recoverable only when its exact current-model digest agrees |
| resume | same output directory and exact config/input/executed-source/runtime digests only; restore all states before the next sample; a partial run cannot be reported as the baseline result |

The trainer must reject a test path anywhere under the closed data root, a
nonempty new output directory, an output inside the data tree, manifest or
normalizer drift, a non-CUDA scientific run, more or fewer than one visible GPU,
nonfinite loss/gradient/parameter/proposal, parameter-count drift, missing or
duplicated validation keys, and resume-contract drift. It writes compact JSON
manifests/history/summary plus best and last checkpoints; it does not write
rollout tensors or media.

A1 synthetic tests must cover all 29 adjacent pairs, deterministic case/time
ordering, effective-batch-26 loss parity, exact proposal recurrence, case-first
five-case validation aggregation, test-path rejection, best/last schema
separation, and uninterrupted-versus-resumed deterministic equality. No A1 test
may open the real trajectory tree, instantiate a CUDA context, or execute the
default 8.9M-parameter training loop.

### Direct-baseline A1 implementation closeout

Status: **COMPLETE; ALL SOURCE AND SYNTHETIC CPU GATES PASS; GPU RUN NOT
AUTHORIZED** on 2026-08-12.

The complete new executable surface is:

| Artifact | SHA-256 | Owner/invocation |
| --- | --- | --- |
| `scripts/time_dependent_no/train_realm_ignithit_ffno.py` | `a3fa0c0e831765ce24c6ae3aa11e3d2afcf67cc242de0a508381dc0b25b43f1e` | narrow future A3 entry point; exact manifest, data root, P1b normalizer arrays, and new output directory are mandatory |
| `tests/time_dependent_no/test_train_realm_ignithit_ffno.py` | `4890a37d857f41b5611a50932efc6aca1c3beda1615c42e1fc071d9693e2dc32` | synthetic CPU contract, checkpoint-integrity, and resume tests |

`ruff format --check` and `ruff check` passed for both files. The focused
REALM CPU suite passed `74/74`: benchmark metrics, IgnitHIT closed-tree/data
contracts, FFNO shape/capacity contracts, and the new trainer tests. The new
tests establish exact 29-pair support, deterministic sampling and resume,
effective-batch loss/gradient parity, direct recurrence, case-first validation,
sealed-test/output guards, safe CLI help, and distinct best/last ownership with
structured-state digests. The write transaction stores authoritative `last.pt`
before a newly improved `best.pt`; resume may reconstruct a missing or stale
best only when the last state is itself the registered best and its model digest
and validation row agree exactly.

No real trajectory array was opened; the new trainer tests did not invoke
`run_training` or instantiate its default model; and no CUDA context, network
endpoint, remote host, or scientific checkpoint was used. No new utility,
configuration, generated report, or artifact directory was added. These results
authorize no claim about trained accuracy or stability.

### Direct-baseline A3 terminal result

Attempt `d088_realm_ignithit_p1c_direct_seed0_5000_20260812a` was explicitly
authorized and launched on 2026-08-12 on the owner-selected personal GPU
resource. The prelaunch source, open-input, normalizer, split, sealed-test,
runtime, occupancy, output, and session gates all passed. One GPU was visible;
the external hard timeout was 43,200 seconds; no `--stop-after-step` override was
used.

Exact execution identity:

| Field | Frozen observed value |
| --- | --- |
| run signature | `2c08721a2b769ca30d25717eea4ee5027e78bc03a73c95f55921d3cf6c25d76f` |
| config/input/source/runtime digests | `9949335070d23ecd8719a94c67d327e0aa53ce421ea1dffa27892b7fba599fd7`; `08d80fc972ae5ace8f4c3c5392c4fd36b53187f2d2a7655fff820257a2ec0074`; `a6f71c1c906ebbe8ab36566a8dcc28dd09d1feb1ac0b887bb548ec66570725ee`; `35e3f3f4bf8098eb73fd6b51fc42828a7ba59cee2c3200b0fea28b0bbd4371aa` |
| completed retained state | step 50; best step 50; case-first H29 validation NPE `4.562549114227295` |
| step-50 structure gates | normalized finite; decoded finite; released-state admissible; bounded below 10x train envelope on all five cases and 29 calls |
| terminal event | next registered validation at step 100: normalized H29 proposals completed finite, then prediction decode returned a nonfinite physical value |
| terminal disposition | registered stop; no step-100 history row/checkpoint/status/final manifest; no automatic resume |

The exact retained compact/artifact hashes are:

| Artifact | SHA-256 |
| --- | --- |
| captured error log | `5563b18c795982e21c9454185fa143c32adb7f846a204339f74a41bbb15900e5` |
| `config.json` | `cc40b001c341aa4d9fcd97c686cd3ca0e2b43aa439f4b4f58a7fb8a5f9887dd6` |
| `input_manifest.json` | `87e212a322790d78de480b4b338fbf5f3c34ae3247b2062d273899e1179758f4` |
| `source_manifest.json` | `c5f7e4c136115eb59e0d482961ab1a4c363a9515267d1154433fa20562cc47b0` |
| `runtime_manifest.json` | `7ea508c45fa1a2a74a9dc5fd0f131686373c3024651fbb2d383ce5f199849f43` |
| `history.json` | `24876f3313fcdc1bfe68d8d86b496fcadd6286a99dfcd2a9283ddc5c87db4641` |
| `best.pt` | `7a94eee88d3ed903b610bdeb3888b144d294cbcc9f90eec807cd005c43480fe4` |
| `last.pt` | `437e304b0c280488c08dcb727ea7de0431bee363df805821785bc09f8fe13832` |

Independent post-stop verification recomputed and passed the last-model,
best-model, best-normalizer, best/last link, provenance-signature, common
provenance, and step-50 identity checks. The hard cap was not reached, GPU
memory did not exhaust, and no test object or residual arm was used.

This is a failed completion gate, not evidence that the final 5,000-step model
would necessarily be inaccurate and not evidence that REALM, FFNO, or direct
state prediction generally fails. The failure is localized only to the frozen
optimizer/history and early step-100 H29 decoded rollout. The current failure
path does not retain the step-100 model or identify the first case/call/channel,
so cause claims about Box--Cox margin, recurrent amplitude, or a specific field
remain unsupported. The minimum decisive follow-up is a separately registered
exact step-50-to-100 replay with failure-state localization; it may not silently
continue this attempt or relax physical validity.

## P1c Step-100 Decode-Failure Localization

Status: **A1 COMPLETE; ALL SOURCE AND SYNTHETIC CPU GATES PASS; SHORT GPU
REPLAY NOT AUTHORIZED** on 2026-08-12. This is a diagnostic attempt under D088, not a new
stable result ID. Its frozen label is
`d088_realm_ignithit_p1c_decode_localization_20260812a`.

The parent identity is exact: attempt
`d088_realm_ignithit_p1c_direct_seed0_5000_20260812a`, step-50 `last.pt`
SHA-256 `437e304b0c280488c08dcb727ea7de0431bee363df805821785bc09f8fe13832`,
step-50 `best.pt` SHA-256
`7a94eee88d3ed903b610bdeb3888b144d294cbcc9f90eec807cd005c43480fe4`,
and run signature
`2c08721a2b769ca30d25717eea4ee5027e78bc03a73c95f55921d3cf6c25d76f`.
The registered parent config/input/source/runtime canonical digests are,
respectively,
`9949335070d23ecd8719a94c67d327e0aa53ce421ea1dffa27892b7fba599fd7`,
`08d80fc972ae5ace8f4c3c5392c4fd36b53187f2d2a7655fff820257a2ec0074`,
`a6f71c1c906ebbe8ab36566a8dcc28dd09d1feb1ac0b887bb548ec66570725ee`,
and `35e3f3f4bf8098eb73fd6b51fc42828a7ba59cee2c3200b0fea28b0bbd4371aa`.

The replay is deliberately narrow:

- restore the exact step-50 model, Adam, OneCycleLR, case-order RNG, and global
  RNG states after all parent file, checkpoint-internal, source, open-input,
  normalizer, runtime, sealed-test, and output-isolation checks pass;
- execute exactly optimizer steps 51 through 100 with the frozen direct
  trainer's sampling, one-step objective, effective batch 26, and step order;
- retain an inference-only step-100 model-state checkpoint and its structured
  digest before decoding;
- run only the existing five-case, frame-0, H29 direct recurrence and require
  every normalized proposal to be finite;
- decode with the unchanged primary P1b normalizer and
  `inverse_domain_policy="nan"`; then stop, whether the terminal event is
  reproduced or not;
- never resume the parent output in place, continue beyond step 100, retry with
  changed numerics, load a test object, or execute the residual arm.

The first decoded-nonfinite location is the lexicographic minimum over frozen
validation-case order, one-based call, released channel index, row, and column.
The compact localization record binds the case key, call, field, normalized
value, pre-inverse transformed value and finiteness class, inverse Box--Cox base
`1 + 0.1 z` and domain margin where applicable, decoded nonfinite class, total
invalid-point count, affected cases/calls/channels, first event per case, and a
call-by-call pre-failure trace for the globally first case/channel. Mechanism
labels are operational only: negative Box--Cox base is an inverse-domain
violation; finite nonnegative base with nonfinite power is inverse-power
overflow; a nonfinite pre-inverse value is transformed overflow; and a
nontransformed channel can only receive a linear-decode overflow label. These
labels localize numerical decoding; they do not establish why training created
the normalized trajectory.

If no decoded nonfinite value appears, the result is a reason-coded
nonreproduction under the exact replay. It is not permission to continue
training. Any normalized nonfinite proposal, parent/source/runtime mismatch,
wrong restored step, missing sealed-test guarantee, or failure before a
returned decoded tensor is an infrastructure/contract stop and yields no
scientific localization row.

The A1 implementation surface is limited to one entry point,
`scripts/time_dependent_no/diagnose_realm_ignithit_decode_failure.py`, one
synthetic CPU test,
`tests/time_dependent_no/test_diagnose_realm_ignithit_decode_failure.py`, and
this preregistration plus the maintained W26-L4 plan. The original trainer and
all reusable REALM utilities remain byte-identical. Focused CPU gates cover the
fixed parent/replay constants and knob-free CLI, parent hash and output guards,
exact restore/continuation parity on a tiny model, normalized-finite
enforcement, lexicographic localization, inverse-domain and inverse-overflow
classification, untransformed-channel behavior, JSON-finite serialization,
nonreproduction, step-100 checkpoint linkage, and safe `--help`. No A1 check
may open the real trajectory tree, load the scientific checkpoint, initialize
CUDA, or contact a remote host.

### Decode-localization A1 closeout

The complete new executable surface is:

| Artifact | Bytes | SHA-256 | Owner/invocation |
| --- | ---: | --- | --- |
| `scripts/time_dependent_no/diagnose_realm_ignithit_decode_failure.py` | 35,259 | `1635164736867bfa40692f4e857ba8af97e96f8dc85486e34d8d072ce82487f4` | exact future short A3 entry point; requires the open manifest/tree, P1b normalizer arrays, exact parent output, and a new isolated output |
| `tests/time_dependent_no/test_diagnose_realm_ignithit_decode_failure.py` | 16,974 | `cbb4d5b100d58c1a04900bd58722211caea55567f56c29462b3a705445e41063` | synthetic CPU source, guard, replay-parity, localization, and artifact-schema gates |

The parent trainer remains byte-identical at SHA-256
`a3fa0c0e831765ce24c6ae3aa11e3d2afcf67cc242de0a508381dc0b25b43f1e`.
`ruff format --check` and `ruff check` pass for both new files. The full
maintained REALM synthetic CPU suite passes `85/85`: generic benchmark,
IgnitHIT data-contract, FFNO, direct-trainer, and decode-localization tests.
The new 11-test subset additionally closes exact current parent config/source
identity, wrong-step/provenance rejection, output nonoverlap, bounded invalid
mask reduction, all five lexicographic coordinates, JSON-finite nonfinite
classes, exact fresh-process step-50 continuation parity, and inference-only
step-100 checkpoint linkage.

No real trajectory array or scientific checkpoint was opened. No CUDA context,
network endpoint, remote host, generated scientific artifact, test object, or
residual model was used. A future successful A3 writes only `contract.json`,
`parent_identity.json`, `source_manifest.json`, `runtime_manifest.json`,
`replay_trace.json`, `step100_model.pt`, `localization.json`, `summary.json`,
and `final_hash_manifest.json` in a new attempt directory. These source-only
results authorize no stability, accuracy, or failure-mechanism claim.
