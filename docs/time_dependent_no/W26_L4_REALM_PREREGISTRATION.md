# W26-L4-P1a REALM Benchmark Contract Preregistration

Status: A1 local-contract implementation authorized 2026-08-12; frozen before
any REALM trajectory, checkpoint, model output, AutoDL host, or GPU is accessed.
This file has no D-series identity. `D088` remains unallocated.

## Question And Scope

This line asks whether predicting a normalized-state increment improves
long-horizon IgnitHIT validation rollouts over a matched direct-next-state FFNO.
P1a does **not** answer that question. It independently fixes the contracts that
must exist before a fair reproduction or comparison can run:

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
| open manifest | SHA-256 `85e8dcdaf4a5f7726ac5a7ad18a1bc131785fc1212acc9318ff69b94be965205`; 552,023,019 bytes | exact metadata plus 26 train and five validation trajectories |
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

| Diagnostic | Frozen operational definition | P1b value still required |
| --- | --- | --- |
| admissibility | native finiteness; each released species `>=0`; `T>0`; `rho>0`; `p>0` only when present | whether omitted species permit any stronger composition diagnostic |
| boundedness | inclusive `abs(channel) <= expansion_factor * train_quantile(abs(channel))` | train-only quantile and expansion factor |
| boundary band | inclusive distance to any Cartesian domain edge `<= physical band_width`; report decoded boundary/interior MSE and ratio | physical extent, spacing, band width, and channel set |
| front high/low regions | `field>=high` and `field<=low`; transition band is strictly `low<field<high` | train-derived field choice and thresholds |
| front area/length | grid-cell area proxy `count*dx*dy`; neighbor high-mask crossings weighted by `dy` across x and `dx` across y | physical spacing and whether released values are cell- or point-centered |
| front position/thickness/strength | gradient-magnitude centroid; transition area/interface length; high mean minus low mean; missing interfaces/regions reason-coded | train-only thresholds and reporting fields (`T`, `OH`, or both) |
| spectrum | full 2-D `fft2(norm="ortho")`; radial cycles per physical unit; half-open bands except inclusive last edge; final edge covers the grid maximum; Parseval closure | physical `dx,dy`, demeaning policy, and fixed band edges |

The area and interface quantities are operational grid proxies until the actual
released grid centering/extent contract is verified. Released-species
nonnegativity is not a complete species simplex or elemental-conservation test.
Boundary-band error is descriptive and cannot establish boundary-condition
learning without deployable boundary/forcing provenance. Lower high-frequency
energy alone cannot establish improvement if position, thickness, or strength
regresses.

## Provenance And Comparison Matrix

| Field | Direct arm | Residual arm | P1a status |
| --- | --- | --- | --- |
| paper/source/dataset identities | exact bindings above | identical | resolved at metadata level |
| train/validation population | pinned 26/5 split object manifest | identical | object IDs bound by digest; arrays unopened |
| sealed test | absent | absent | mandatory |
| channel/group contract | 12 channels and slices above | identical | resolved |
| normalizer axes/formula | train-only contract above | identical | resolved |
| Box-Cox epsilon | explicit runtime argument | identical | unresolved until P1b, before outputs |
| architecture/capacity/init | FFNO-M reconstruction | exact paired tensors | unresolved until P1c |
| optimizer/history/seed/budget | matched reconstructed baseline | identical | original paper values partly unresolved |
| target parameterization | direct next normalized state | normalized-state increment | registered factor |
| recurrence/exposure | exact deployed state; named one/two-call mode | identical except addition algebra | resolved synthetically |
| precision/runtime | exact future runtime manifest | identical | unresolved until AutoDL preflight |
| evaluator | this independent contract plus future source hashes | identical | synthetic semantics implemented |
| selection | validation only; test unavailable | identical | registered; not executed |

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

After source review passes, the smallest next authorization is:

1. allocate noncolliding stable ID `D088` to W26-L4; and
2. separately authorize P1b metadata plus train/validation acquisition of
   exactly 552,023,019 bytes at the pinned revision, under a recorded license
   disposition.

That later authorization still would not permit test acquisition, AutoDL/GPU,
training, or a direct-versus-residual scientific comparison. Those remain P1c
and P2 decisions after real-data schema/metric replay closes.
