# W26-L1 H320 Reference-Free Rollout Record

Date recorded: 2026-08-21

Status: closed descriptive evidence at the registered single-seed, 30-case
scope

## Scope and population

The retained packets extend the D087 bump recurrence from H79 to H320 for the
same 30 open validation trajectories and compare PCNO with the functional
no-gradient PCFNO control. The retained prefixes close against the registered
H79 trajectories; truth is available only through call 79. Consequently,
H80--H320 is reference-free recurrence evidence.

The four working identities are:

- `W26-L1-PCFNO-H320-A1-S20260718`;
- `W26-L1-PCNO-H320-A1R1-S20260718`;
- `W26-L1-PCFNO-PCNO-H320-C1-S20260718`; and
- `W26-L1-PCFNO-PCNO-H320-NORM-P1-POSTHOC`.

## Retained artifact receipts

All 43 files named by the five final manifests were rehashed locally by path,
size, and SHA-256 with no missing or mismatched member.

The retained packet directories are:

- `artifacts/time_dependent_no/w26_l1_pcfno_h320_s20260718_20260814a_r1/`;
- `artifacts/time_dependent_no/w26_l1_pcno_h320_s20260718_20260814a_r1/`;
- `artifacts/time_dependent_no/w26_l1_pcfno_pcno_h320_comparison_s20260718_20260814a_r1/`;
- `artifacts/time_dependent_no/w26_l1_pcfno_pcno_h320_prediction_norm_20260814a/`;
- `artifacts/time_dependent_no/w26_l1_pcfno_h320_s20260718_20260814a_r1_visuals/`.

| Packet | Summary SHA-256 | Final-manifest SHA-256 |
| --- | --- | --- |
| PCFNO H320 rollout | `fd1908d4d49bdc2b934e913f8d0d6e01fd7fbde9dbbfe4abd1bdfd22822fef16` | `69c1b017e18cce9371f1c8b6f71763b021a36e9c75453a7242ae1b2df707c8a4` |
| PCNO H320 rollout | `a205848a01f5d0f4ee36b34097d174ced4f603282335edfb1cc60f7fb884ef94` | `6db0133d432cfee1b29029a7c6d677ee2ca590d109bf4b171bbc69b4be6e94ef` |
| matched comparison | `1c97b907ad23b6b40f0f01fd2034f5f7bfa6bfaa5618e127ecee47c546775536` | `b78110ec4e83c9c4cd38a7576897d80cfb0454efb4449294f162ebf5b4b3f0d1` |
| posthoc prediction norm | `e79a31126afd15f78157c6b01d3a8d7ea994122aceae10a938dc742704c0516a` | `e6a46b661a0370313888060910fefacb37353f599bbc1b2892d903c7d4f4102f` |
| PCFNO visualization | `bec4c7125ab0d8f5e3e50700d02bddd76f2a77aa77ca5e0c34c1e559b1a4dda2` | `a1f8aa9d4d6dbba4a1c4f24e090d42b3ac1ffaf227a65147b165c94095f1521e` |

The PCFNO and PCNO run-contract SHA-256 values are respectively
`fefa37e8a96e8ce60c22035270e7bc418a599caacd4e02938499d51729446afb`
and `8d411e09c13e1570cadc08c2fa69e7ebd78dfedaa8b9a4c68c8410663e265d5f`.
Both rollout summaries declare `accuracy_evaluated=false`; the PCFNO summary
also records bitwise equality for every checked H79 prefix.

## Descriptive result

| First event | PCNO median call | PCFNO median call | Paired later count, PCNO / PCFNO / tie |
| --- | ---: | ---: | ---: |
| inadmissible state | 139.5 | 108 | 27 / 3 / 0 |
| boundedness failure | 197.5 | 149 | 28 / 2 / 0 |
| nonfinite state | 320, right-censored | 320, mixed | 13 / 0 / 17 |

PCNO remains finite through H320 on 30/30 trajectories; PCFNO remains finite
on 17/30. The posthoc prediction-size diagnostic shows both families departing
strongly from their H1--H79 reference scale, with PCFNO generally departing
earlier and more severely. That diagnostic uses no future truth and is not an
error, phase, shock-position, admissibility, or conservation metric.

## Claim boundary

This supports only a bounded statement: on one seed and the registered 30-case
open population, the active-gradient PCNO recurrence reaches the registered
admissibility, boundedness, and finiteness events later than the functional
no-gradient control more often than the reverse. It does not establish H80--H320
accuracy, physical validity, conservation, asymptotic stability, a causal
gradient-path effect, seed generality, or a general neural-operator claim.

## Replay gap

Two source-bound preregistration documents are absent from the checkout, all Git
history, and every retained archive:

- `W26_L1_PCFNO_H320_PREREGISTRATION.md`, expected SHA-256
  `db5cc052a470d06fd3360b84f20c569421c949914ace6774bc4dfaf13894eb3b`;
- `W26_L1_PCFNO_PCNO_H320_COMPARISON_PREREGISTRATION.md`, expected SHA-256
  `a25aad5a9ce2c35b6b4968ebe0e6f3225a70f1f692fd6fb8abd4dcc42c0480a7`.

All three H320 evaluators also invoke the historical B1 frozen-source verifier.
That verifier binds exact versions of the active tracker, decision, and Euler2D
trainer at SHA-256 values
`c3014df67a4e7d5ed8f1d48b1c76f664b5a7604557d7ce0b783ca77a19cc7ee4`,
`dafa88e4bca3b174ff9711272259e6a449dab9d744dac24d0254e6a61d7c2de3`,
and `7f7733b421f74411a420d753b51a5c56a798f578462806a37054bee16bc82318`.
Those three historical byte versions are likewise absent from the checkout,
Git history, and all retained archives, so the current verifier fails closed.

The retained packets remain evidence for the bounded outcomes above, but exact
old-identity source replay is unavailable. These documents must not be
reconstructed under their historical hashes. Any future H320 run requires a
new preregistration, identity, and source manifest.
