# D093 W26-L4 PlanarDet Architecture/Exposure Record

Updated: 2026-08-21

Status: closed single-seed, single-open-validation evidence record; not a live
queue or authorization

## Registered Comparison

D093 reuses the completed D092-R1 `PCNO-7` run as its seven-trajectory PCNO
anchor and adds five cells: `PCFNO-7`, `FFNO-7`, `PCNO-3`, `PCFNO-3`, and
`FFNO-3`. All evaluations use the one open PlanarDet validation trajectory
`sampling_phi1-290`; `test_object_opened=false` throughout.

The suffix `-3` or `-7` denotes the number of unique supervised training
trajectories, not optimizer updates or batch size. Every optimizer step has
seven microbatch presentations at one shared registered time window. The
three-trajectory arm repeats its three active conditions within those seven
presentations, accumulates all seven gradients, and then takes one optimizer
step. Both arms use normalization fit on all seven released training
trajectories. D093 is therefore a condition-diversity/exposure comparison with
a transductive normalizer control, not a clean dataset-size scaling law.

`PCFNO` is the matched PCNO implementation with the differential/gradient
branch disabled; it is not the REALM paper's vanilla FNO. The D093 `FFNO` also
uses the common residual target and D093 coordinate/training contract, so its
numbers are not a paper-faithful REALM reproduction.

The reused D092 anchor is a result-level comparison, not a byte-identical
source match. Both source manifests use schema
`realm_planardet_pcno_source_snapshot_v1`, but D092 binds 17 files with payload
`1c48afecdde31d5998695438a39b19ddb328f91b63e272c26ecaed0c30e6f502`, while
D093 binds 18 files with payload
`8cbdc9dad419e236926bca76dfca51689229fe096e0b5d57ff7410650bf7c5b5`.
In particular, D093's `realm_pcno.py` adds optional gradient-disable support.
The active-gradient PCNO path is intended to preserve D092 semantics, but the
six-cell matrix is not source-matched bytewise.

## Retained Results

All sums below cover the 49 registered target calls. `A/B` gives registered
admissible/bounded call counts, each out of 49. The endpoint is the last
retained successful training step; `incomplete` means the 5,000-step training
contract did not complete.

| Model | Unique train trajectories | Best step | Endpoint | Truth-input H49 sum | Free H49 sum | Truth-input A/B | Free A/B | Free decoded finite |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| PCNO-7 | 7 | 950 | 5,000 complete | 6.09925 | 88.13821 | 49/49 | 7/49 | yes |
| PCFNO-7 | 7 | 700 | 1,900 incomplete | 10.67000 | 167.42062 | 49/33 | 1/28 | no |
| FFNO-7 | 7 | 850 | 5,000 complete | 1.36466 | 32.53910 | 34/49 | 4/49 | yes |
| PCNO-3 | 3 | 850 | 5,000 complete | 7.06828 | 85.74666 | 49/48 | 4/49 | yes |
| PCFNO-3 | 3 | 700 | 2,000 incomplete | 12.00864 | 67.19994 | 49/29 | 3/37 | no |
| FFNO-3 | 3 | 850 | 1,950 incomplete | 2.03325 | 51.50276 | 45/49 | 5/49 | yes |

The frozen-initial-state persistence control, defined as predicting `U(t0)` at
all 49 targets without truth refresh, has normalized H49 sum `421.91979`.
Every selected model beats that weak control. This does not make persistence a
matched learned baseline.

## Evidence And Interpretation

- The seven-condition cell has a lower selected truth-input sum than the
  three-condition cell for PCNO, PCFNO, and FFNO in this matrix.
- Only FFNO's seven-condition cell also has a lower selected free-rollout sum.
  PCNO changes from `85.74666` to `88.13821`, and PCFNO from `67.19994` to
  `167.42062`; the seven-condition cell is not uniformly better under free
  recurrence in this matrix.
- FFNO is best in this matrix on the selected truth-input and free sums, but its
  free sum `32.53910` remains 2.59 times the paper's numerical PlanarDet FFNO
  validation value `12.577`. The contracts differ, so this ratio is context,
  not a direct benchmark comparison.
- PCNO has lower selected truth-input error than PCFNO at both condition
  counts, while their selected free-rollout ordering changes with the condition
  count. These data do not isolate a gradient-branch cause or establish that
  the branch prevents smearing or improves long-horizon stability.
- Late training and validation strongly decouple in every cell. PCFNO-7,
  PCFNO-3, and FFNO-3 stop after later truth-input validation proposals fail
  finite physical decoding. None of the six selected free rollouts is fully
  admissible. The matrix exposes an alignment problem among optimization,
  truth-input selection, autoregressive reliability, and decoded validity.

This supports only a partial descriptive claim: on one seed and one open
validation condition, each seven-condition cell has lower selected truth-input
error than its three-condition counterpart, while the selected long-horizon
ordering is family-dependent and can reverse. It does not support general data
scaling, architecture superiority, a conservation claim, a paper-faithful
REALM reproduction, or a sealed-test claim.

Minimum missing evidence includes multiple seeds and sparse free-rollout-aware
checkpoint selection under a newly preregistered selection contract, a
paper-faithful direct-state FFNO control, a capacity-controlled gradient
ablation, and another open validation condition or newly preregistered
non-sealed population. This record does not authorize those runs.

## Provenance And Local Replay Boundary

- Training source manifest:
  `artifacts/time_dependent_no/d093_planardet_arch_data_scaling_20260817a/source_manifest_8cbdc9dad419e236.json`;
  canonical payload SHA-256
  `8cbdc9dad419e236926bca76dfca51689229fe096e0b5d57ff7410650bf7c5b5`.
- Evaluation source manifest:
  `artifacts/time_dependent_no/d093_planardet_arch_data_scaling_20260817a/evaluation_20260820/source_manifest.json`;
  canonical payload SHA-256
  `bc666902e18ccaa63a9b0446fdab8dd40bd2edf921fe50d41148d9b7dc1d6f97`.
- Later retained persistence-visualization source manifest:
  `artifacts/time_dependent_no/d093_planardet_arch_data_scaling_20260817a/visualization_persistence_deployment_20260820a/source_manifest.json`;
  canonical payload SHA-256
  `31e1dcdc9b4678a1ad94064f4e090f9da5f42702ebb9f93b933c2385800d69b9`.
  Its summary and final-manifest file SHA-256 values are respectively
  `08b59d34224c3d96dc4ea3ae8d6d5cbc8b3ba0264975af800cf387b880d4af1e`
  and `39c85c0715269bb92b9011e796762d04a90a75f2168e4aee81930a89bdf4333c`.
  Those files now remain as members beneath `visualization_persistence_20260820a/`
  in `visualization_persistence_deployment_20260820a/visualization_persistence_output_bundle.tar.gz`,
  archive SHA-256
  `4f0c64c65610108efc4f592c4f957d8e22146957fdc1e453c2bc93c4cfacd079`.
- Result-to-claim record:
  `artifacts/time_dependent_no/d093_planardet_arch_data_scaling_20260817a/result_to_claim_20260820.json`;
  file SHA-256
  `47a8fe9cc1a6fd61773d9b771b48965a6a1f6176770f1a340bcea980ddfe07df`.
  This immutable record predates the corrected seven-presentations-per-step
  sampler accounting and binds the older visualization source payload
  `068d33baf214ba0f153ab1eb906f45c623670977727ea15cd6c6a770be7ab5dc`
  and `visualization_deployment_20260820/visualization_output_bundle.tar.gz`
  bundle `bb222a867bdf5c5089909eca48fe7ae5ddd6d3266b90359c67592bbf39e44f0`,
  not the later persistence packet used for the table above.

Only `PCNO-3` and `FFNO-7` are marked as reaching the 5,000-step D093 endpoint.
For each, eight retained files named by its final manifest hash-match, but the
manifest also names one locally absent `last.pt`; neither is a locally complete
training packet. The three interrupted cells retain partial histories and
selected checkpoints. Their retrieved `status.json` files are stale at step 50
and best step 0; the later histories/evaluator closure reach `PCFNO-7` step
1,900/best 700, `PCFNO-3` step 2,000/best 700, and `FFNO-3` step 1,950/best 850,
which are the values used above. Each of the five evaluation metadata closures
rehashes its three retained files, but its final manifest also names two absent
normalized teacher/free prediction arrays. The closed JSON/CSV summaries
support the table above; full packet closure and exact local replay from those
arrays do not.

The 2026-08-21 cleanup removed only three expanded copies after member-by-member
archive verification. Its append-only receipt is
`artifacts/time_dependent_no/RETENTION_MANIFEST_20260821b.json`, SHA-256
`de4d33468b7501188b9bf18cb3cea94dc50b4354455de4d38f935684407598c6`.
The root evaluation source manifest and parent persistence source manifest named
above remain directly present.

The retained visualization archives are authoritative for their figures. A
later checkout version of the visualizer is not bound by those archives and
must not be used to retroactively reinterpret their outputs. Bytewise replay
should use the frozen source archives rather than the current checkout.
