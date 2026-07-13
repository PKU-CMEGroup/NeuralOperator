# Section 1.2 Corrected Baseline Protocol

This note supersedes legacy Section 1.2 rows produced before the corrected CPGNet/FNO implementations landed.

## Deprecation Rules

- Rows with `model=cpgnet` and no `model_implementation` must be treated as deprecated legacy rows. They came from the old `CPGStyleEuler1DHead` pilot and must not be reported as CPGNet.
- Rows with `model=cpg_style_pilot` are explicit pilot ablations only and are not paper-facing CPGNet baselines.
- Rows with `model=fno` must report `model_implementation`. Width-32/modes-8 runs are legacy controls, not the main FNO baseline.
- Corrected Section 1.2 tables must include `model_implementation`, `target_type`, `input_noise_std`, `step_stride`, `rollout_final_frame`, `seed_count`, one-step loss, rollout mean/final L2, shock error, conservation error, and density/pressure positivity diagnostics.

## One-Seed Corrected Sweep

Set these on AutoDL before launching commands:

```bash
DATA=/path/to/euler1d_dataset.npz
BASE_OUT=artifacts/time_dependent_no/section12_corrected_v1
SEED=20260707
```

Strong corrected FNO, main `limited_residual` target:

```bash
for noise in 0.003 0.02; do
  for cfg in 64:16:4 64:24:4 96:24:4; do
    IFS=: read -r width modes layers <<< "$cfg"
    python scripts/time_dependent_no/train_euler1d_target_ladder.py \
      --data-path "$DATA" \
      --output-dir "$BASE_OUT/fno_w${width}_m${modes}_l${layers}_noise${noise}_seed${SEED}" \
      --model fno \
      --target limited_residual \
      --epochs 80 \
      --train-cases 384 --val-cases 64 --test-cases 64 \
      --step-stride 4 --rollout-final-frame 80 \
      --input-noise-std "$noise" \
      --fno-width "$width" --fno-modes "$modes" --fno-layers "$layers" \
      --seed "$SEED" --device cuda --gpu 0 --fail-fast
  done
done
```

Raw residual FNO control only:

```bash
for noise in 0.003 0.02; do
  python scripts/time_dependent_no/train_euler1d_target_ladder.py \
    --data-path "$DATA" \
    --output-dir "$BASE_OUT/fno_raw_residual_w64_m24_l4_noise${noise}_seed${SEED}" \
    --model fno \
    --target residual \
    --epochs 80 \
    --train-cases 384 --val-cases 64 --test-cases 64 \
    --step-stride 4 --rollout-final-frame 80 \
    --input-noise-std "$noise" \
    --fno-width 64 --fno-modes 24 --fno-layers 4 \
    --seed "$SEED" --device cuda --gpu 0 --fail-fast
done
```

Faithful 1D CPGNet-style adaptation:

```bash
for noise in 0.003 0.02; do
  python scripts/time_dependent_no/train_euler1d_target_ladder.py \
    --data-path "$DATA" \
    --output-dir "$BASE_OUT/cpgnet_h128_mp12_limited_residual_noise${noise}_seed${SEED}" \
    --model cpgnet \
    --target limited_residual \
    --epochs 80 \
    --train-cases 384 --val-cases 64 --test-cases 64 \
    --step-stride 4 --rollout-final-frame 80 \
    --input-noise-std "$noise" \
    --cpg-hidden-dim 128 --cpg-message-passing-steps 12 --cpg-mlp-layers 3 \
    --seed "$SEED" --device cuda --gpu 0 --fail-fast
done
```

## Three-Seed Confirmation

After ranking the one-seed sweep, repeat only the best one or two corrected configs:

```bash
for seed in 20260707 20260708 20260709; do
  python scripts/time_dependent_no/train_euler1d_target_ladder.py \
    --data-path "$DATA" \
    --output-dir "$BASE_OUT/best_config_seed${seed}" \
    --model fno \
    --target limited_residual \
    --epochs 80 \
    --train-cases 384 --val-cases 64 --test-cases 64 \
    --step-stride 4 --rollout-final-frame 80 \
    --input-noise-std 0.003 \
    --fno-width 64 --fno-modes 24 --fno-layers 4 \
    --seed "$seed" --device cuda --gpu 0 --fail-fast
done
```

## Completed One-Seed AutoDL Sweep

Run `section12_corrected_autodl_20260710_222716` completed on 2026-07-12 with the corrected source synced to AutoDL. The dataset was generated as a 512-case 1D Euler `.npz` with shape `[512, 101, 256, 3]` and zero solver fallback steps. All rows below use train/val/test `384/64/64`, stride `4`, rollout frame `80`, seed count `1`, and rollout-validation checkpoint selection.

| Run | Target | Noise | Best Epoch | One-Step Loss | Rollout Mean L2 | Rollout Final L2 | Shock MAE | Conservation Final | Min Rho | Min P | Positivity | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| FNO 64/16/4 | limited_residual | 0.003 | 80 | 4.65e-05 | 3.53e-02 | 5.84e-02 | 1.00e-02 | 2.40e-02 | 1.02e-01 | 1.04e-08 | 5 pressure nonpositive | reject |
| FNO 64/24/4 | limited_residual | 0.003 | 80 | 4.27e-05 | 3.05e-02 | 5.02e-02 | 4.80e-03 | 1.70e-02 | 1.02e-01 | 7.66e-02 | clean | candidate |
| FNO 96/24/4 | limited_residual | 0.003 | 33 | 6.43e-05 | 2.84e-02 | 5.45e-02 | 1.35e-02 | 1.49e-02 | 1.04e-01 | 9.61e-02 | clean | candidate |
| FNO 64/16/4 | limited_residual | 0.02 | 79 | 6.05e-05 | 2.37e-02 | 4.24e-02 | 2.96e-03 | 2.00e-02 | 1.01e-01 | 8.93e-02 | clean | candidate |
| FNO 64/24/4 | limited_residual | 0.02 | 79 | 6.43e-05 | 2.16e-02 | 3.72e-02 | 2.51e-03 | 1.89e-02 | 1.00e-01 | 2.24e-02 | clean | candidate |
| FNO 96/24/4 | limited_residual | 0.02 | 58 | 6.97e-05 | 3.52e-02 | 4.93e-02 | 4.31e-03 | 2.15e-02 | 9.70e-02 | 1.00e-08 | 3 pressure nonpositive | reject |
| FNO 64/24/4 | residual control | 0.003 | 59 | 4.78e-05 | 1.70e+03 | 3.41e+04 | 4.92e-03 | 8.06e+02 | 1.00e-08 | 1.00e-08 | 2 density, 13 pressure nonpositive | reject |
| FNO 64/24/4 | residual control | 0.02 | 79 | 6.53e-05 | 2.18e-02 | 3.60e-02 | 2.78e-03 | 2.23e-02 | 1.02e-01 | 3.51e-02 | clean | candidate |
| CPGNet h128/mp12 | limited_residual | 0.003 | 1 | 9.84e-02 | 5.62e-01 | 7.11e-01 | 4.85e-01 | 4.46e-01 | 1.45e-01 | 2.22e-01 | clean | candidate |
| CPGNet h128/mp12 | limited_residual | 0.02 | 62 | 9.83e-02 | 5.63e-01 | 7.08e-01 | 4.85e-01 | 4.30e-01 | 1.48e-01 | 2.19e-01 | clean | candidate |

Immediate read:

- The corrected strong FNO baseline is much stronger than the faithful 1D CPGNet adaptation on this Section 1.2 target-ladder dataset. Best paper-facing FNO is `64/24/4`, `limited_residual`, noise `0.02`: final rollout L2 `3.72e-02`, shock MAE `2.51e-03`, clean positivity.
- The raw residual control at noise `0.02` is numerically best by final rollout L2 (`3.60e-02`), but it is a control, not the main target. Its conservation final error is worse than the best limited residual row (`2.23e-02` vs `1.89e-02`), and the same raw residual target catastrophically fails at noise `0.003`.
- Noise `0.02` helps the stable FNO rows in this sweep. It improves final rollout L2 for `64/16/4` from `5.84e-02` to `4.24e-02`, and for `64/24/4` from `5.02e-02` to `3.72e-02`.
- Wider FNO `96/24/4` is not better here. At noise `0.02` it triggers pressure floor/positivity flags, so it should not be the main baseline despite having a reasonable final L2.
- The faithful CPGNet-style 1D adaptation is honest but not competitive in this setting: final rollout L2 stays around `0.71` and shock MAE around `0.485`. It does preserve positivity, so the failure is accuracy/optimization or target-fit, not immediate physical invalidity.
## Analysis Command

```bash
mapfile -t RUN_DIRS < <(
  find "$BASE_OUT" -mindepth 2 -maxdepth 2 -name summary.csv -printf '%h\n' \
    | grep -v '/smoke_'
)

python scripts/time_dependent_no/analyze_euler1d_target_ladder.py \
  "${RUN_DIRS[@]}" \
  --output-dir artifacts/time_dependent_no/section12_corrected_v1_analysis \
  --group-by all
```

Do not pass `"$BASE_OUT"/*` directly: that includes `logs/` and other non-run directories. Use `selector_ranked.csv` and `analysis.md` for the Section 1.2 table draft. The analyzer marks unlabeled legacy `cpgnet` summaries as `deprecated_result=yes`.
