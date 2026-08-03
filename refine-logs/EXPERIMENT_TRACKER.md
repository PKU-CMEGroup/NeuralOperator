# D072 Experiment Tracker

This is the execution ledger for `EXPERIMENT_PLAN.md`. Scientific status is
updated only from manifest-bound open-population artifacts.

| ID | Stage | Status | Verified evidence / blocking request |
| --- | --- | --- | --- |
| D072-P0-DYN-MANIFEST | Dynamic source and split provenance | passed | Family ID and embedded digest match the registered 135-case family; retained run contract binds 84 train, 24 open validation, 27 sealed test, shard digest, normalizer digest, and 13,400 updates. |
| D072-P0-DYN-SEAL | Open-only collar publication path | passed locally | Adapter now accepts and records `--splits train validation`, rejects outside populations in the output manifest, and has a focused CPU regression test. No test array was opened. |
| D072-P0-DYN-FIELDS | Dynamic real collar shards | passed | Open-only derived manifest SHA-256 `6d190423b7f367c16c3080bda2ef5a27840463c4a6421c23864f870c941f9339`; 84 train plus 24 open-validation trajectories, 1,404 inherited hardlinks, fixed `ell=0.05`, and an empty test split. The adapter filtered splits before constructing a trajectory path and opened no test array. |
| D072-P0-BUMP-SPLIT | Bump native geometry split | passed | Retained metadata show 270 unique train and 30 unique open-validation geometry digests with zero overlap; Mach ranges overlap. |
| D072-P0-BUMP-GEOMETRY | Bump primary width and boundary provenance | passed | Training-only audit SHA-256 `fd21f095201f9adaabced11884089d0274bf79b9f9793810e7abfba5391f3699` binds 270 keys, checks every saved geometry frame, opens no state/target array and no validation geometry, and freezes `ell=0.05=0.05*median(min(x_span,y_span))`. Derived open manifest SHA-256 is `6b7176d5f7f926af2fe05dc7a2e62a76292501461e47ac721782de4c8431ab4a`; it contains 270 train, 30 open validation, no test, and wall/outflow/inflow channels. |
| D072-P0-SOURCE | Current D072 source snapshot | passed | Isolated source schema v5 binds source-set digest `5c31b97d840397b000e7bf60ffed8a0d2b748d28816f97a575a522019f78a194`, provenance-set digest `6012dd888560dbb9b8c6eebfcfa676e6f750b77313b2bf5b0d718a986bef2687`, and archive SHA-256 `5c721899d28df92c06659d96a8917bb81e9a67749ddcb2c9f513b4e414f8a3f0` on both assigned machines. Dirty remote repositories were not modified. |
| D072-P1-DYN-SMOKE | Dynamic CUDA N0/G1/S1 smoke | passed | All three 64-presentation smokes completed H2 on open validation with identical stream SHA-256 `2cd572bd8749347a222d65d8748ba35462414dd0d4dd609e0de7cecaebcd6aeb`; N0/G1/S1 in-dimensions are 8/9/10 and exact matched initialization passes. Excluded from science. |
| D072-P1-DYN-CORE | Dynamic three-seed core | completed, pending aggregate audit | All nine N0/G1/S1 run directories contain summaries; the final arm finished at 2026-08-03 23:45 CST and the assigned GPU is idle. This is completion evidence only: run-contract verification and result comparison remain pending, and no method claim has been accepted. |
| D072-P2-BUMP-SMOKE | Bump CUDA N0/G1/S1 smoke | passed | All arms complete H2 with admissibility 1.0, normalizer digest `d2d07a4000acc3cfd45ff105a19e7437177d553cee4c18d50858f5572de7efec`, identical stream SHA-256 `5e5eb512f3d4a615ff79bbea4f524065932d3d2e814bd212ef5a462575f0d77e`, and in-dimensions 8/9/11. A post-run bookkeeping assertion was corrected without retraining: homogeneous sampling recorded 64, not 16, optimizer steps; corrected validator SHA-256 is `e87fdbcffec937eb2610e2fd293c899a3cbfec30ecd0eab65d279f66dbc45d62`. |
| D072-P2-BUMP-CORE | Bump three-seed core | running unattended | Seed-20260718 N0 completed its registered 40 epochs, 853,200 presentations, and 216,000 optimizer steps. The original shell queue then stopped only because its post-run wrapper incorrectly required `raw_recurrence=true`; the preserved causal policy correctly records `raw_recurrence=false` with autonomous closure. A uniquely named resume queue validated N0 and launched the remaining eight arms unchanged; seed-20260718 G1 reached active CUDA training with finite epoch-0 metrics. Measured N0 throughput gives a nominal finish near 2026-08-05 17:00 CST and a conservative window through early 2026-08-06. |
| D072-P3-CONTROLS | Linear-kernel and width controls | conditional | Run only after the corresponding core matrix; use the registered expansion trigger. |
| D072-P4-MECHANISM | Frozen interventions and hooks | pending | Requires trained checkpoints and no-hook/hooked equivalence. |
| D072-P5-EVAL | Resolution/geometry/rotation metrics | pending | Dynamic common-source only; bump native graphs only; rotation diagnostic only. |
| D072-P6-VIS | All-frame figures and animations | pending | Every comparable frame, fixed scales, scalar metrics, and failure markers required. |

## Satisfied bump retrieval contract

The following request was satisfied by the hash-bound training-only audit above.
It remains here as the exact provenance contract: produce one JSON manifest
without copying or decoding conservative states:

- schema name and generation command/source snapshot digest;
- source HDF5 identity: byte size, modification time, and available immutable
  source digest or the existing bound source-artifact-set digest;
- exact ordered training keys and their split-manifest digest;
- for each of the 270 training keys only: coordinate-array digest, connectivity
  digest, node-type-array digest, `x_min`, `x_max`, `y_min`, `y_max`,
  `x_span`, `y_span`, node count, boundary-cycle digest, per-semantic segment
  count and total polyline length, and any mixed-endpoint count;
- explicit code map `0 normal, 1 wall, 2 outflow, 3 inflow`;
- confirmation that the boundary cycle is derived from released graph
  connectivity and that mixed endpoint edges enter both adjoining semantic
  subsets;
- aggregate training-only
  `median(min(x_span, y_span))` and the resulting
  `ell = 0.05 * median(min(x_span, y_span))`;
- a digest over the canonical JSON payload.

Do not include state values, target summaries, validation outcomes, model
outputs, or any sealed population. Stop if a training key lacks coordinates,
connectivity, type tags, or an unambiguous boundary cycle.

## Current local verification

- Final boundary-field, source-artifact, and trainer regression: 59 tests passed
  in 56.48 seconds after source-snapshot-v5 and causal-policy integration.
- The additional maintained runtime suite passed 6 tests. The broader legacy
  `pcno_test.py` check passed 8 tests and retained one unrelated pre-existing
  structured-element ordering failure; D072 does not touch that conversion path.
- Worktree is intentionally dirty and contains concurrent user work; no file has
  been staged, committed, or reverted.
