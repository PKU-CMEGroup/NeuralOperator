# Historical Documentation Archive

Files in this directory preserve superseded research language and execution
state. They are evidence, not current authorization or a live experiment queue.
Do not rewrite a dated snapshot; corrections belong in a newer active document
or a later dated archive.

## Authority Snapshots

Each pair below is a byte-preserved copy of the active decision and diagnostic
tracker immediately before a documentation compaction.

### Through 2026-08-01

- `RESEARCH_DIRECTION_DECISION_through_2026-08-01.md`
  - SHA-256:
    `5B446C93FA7487F7B97C895FC3CC38309E549F496656FDD87FC6FBC60BC6A538`
- `MECHANISTIC_DIAGNOSTIC_TRACKER_through_2026-08-01.md`
  - SHA-256:
    `D055F2EEB398224CEFDBDD86D3A230C501DFA4069D65DF188C7F8183F0F821CE`

### Through 2026-08-11

This pair includes the detailed D064--D086 record and the owner-selected weekly
program immediately before the second active-document compaction.

- `RESEARCH_DIRECTION_DECISION_through_2026-08-11.md`
  - SHA-256:
    `D4E46F69A418F8557190F63D80B0A886BC359A45DE058121CDDA38ADD57A2DE7`
- `MECHANISTIC_DIAGNOSTIC_TRACKER_through_2026-08-11.md`
  - SHA-256:
    `CA8CF201FD2CD76AE086241BCF858C36EC7F20375A2AD5E9F97A811D213CBC03`

## Frozen D072 Execution Records

The directory `d072_refine_logs_frozen_2026-08-03/` preserves the original
generic root refinement logs together so their sibling relationship remains
intact:

- `EXPERIMENT_PLAN.md`
  - SHA-256:
    `D61C6128DD565269FDB81C409B629225232526A2A056876451C631F870115986`
- `EXPERIMENT_TRACKER.md`
  - SHA-256:
    `82B9EAB15C34E37380A66DA1F8040A9806DEAC115467C458E7695A6E9273EC00`

Their text says that D072 was executing or pending because that was true when
they were frozen. D072 and its D084 follow-up are now closed; use the active
documents for current status.

## Superseded Line Plans

- `W26_L4_REALM_BENCHMARK_PLAN_through_D091.md` preserves the W26-L4 REALM
  execution plan through D091. It is historical evidence, not a live queue or
  experiment authorization. Resolve its relative links against the original
  `docs/time_dependent_no/` base directory.

## Retired Coordination Scaffolding

`CODEX_KICKSTART_PROMPTS.md` and `findings.md` were transient coordination
notes, not claim-bearing evidence or current authorization. They were retired
on 2026-08-21 after active links were removed. Dated snapshots may still name
the former prompt file; recover exact historical bytes from Git when needed
rather than treating that link as a live instruction surface.

## Retrieval And Link Resolution

Use the compact active [research decision](../RESEARCH_DIRECTION_DECISION.md),
[handoff](../HANDOFF.md), [experiment index](../MECHANISTIC_DIAGNOSTIC_TRACKER.md),
and [weekly plan](../WEEKLY_RESEARCH_PLAN.md) for routine context. Search this
directory by stable run ID only when exact historical contracts, metrics,
hashes, or dated recommendations are needed.

The authority snapshots were copied without changing their bytes. Resolve
relative links inside them against their original base directory,
`docs/time_dependent_no/`, rather than `docs/time_dependent_no/history/`.
Likewise, plain repository paths in the frozen D072 logs retain their original
meaning. These rules preserve exact hashes while active documents provide
working navigation.
