# FullTrainable16 vs FullTrainable8 Wind-bin L8-best Summary

Date: 2026-07-05

Status: FullTrainable8 is promoted to the current descent true-observation
mainline. The legacy default before this promotion was frozen 32D, not
trainable 16D.

## Setup

- Models: `FullTrainable16` vs `FullTrainable8`.
- Checkpoint rule: `ckpt_best_l8.pt`, selected after reaching curriculum L8.
- Episodes: 256 per wind bin per model.
- Wind bins: 0-2, 2-4, 4-6, 6-8, 8-10 m/s.
- Eval setting: descent phase, true observation, no vision, no observation predictor, no cable latent predictor.
- Control setting: PID residual base enabled.
- Environment: default L8 descent curriculum, 10 cable segments, 3 rebars, fixed target XY.
- Important test-side dimension setting:
  - 16D: `--cable-encoder-output-dim 16`, actor/critic obs_dim = 60.
  - 8D: `--cable-encoder-output-dim 8`, actor/critic obs_dim = 52.

## Files

- Combined raw metrics: `combined_wind_bins_process_metrics.csv`
- Combined readable table: `combined_wind_bins_process_metrics_readable.csv`
- JSON readable table: `combined_wind_bins_process_metrics.json`
- Raw 16D CSV: `full_trainable16_l8best_wind_bins.csv`
- Raw 8D CSV: `full_trainable8_l8best_wind_bins.csv`
- Logs: `wind_bins_l8best.stdout.log`
- Plots: `plots_full16/`, `plots_full8/`

## Main Results

`Main SR` is broad success rate, i.e. strict insertion success plus lucky insert. `Strict SR` is the perfect/strict success rate.

| Model | Wind (m/s) | Main SR | Strict SR | Avg reward | Avg steps | Avg KE (mJ) | P95 KE (mJ) | Avg swing (deg) | P95 swing (deg) | Avg EE acc (m/s^2) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullTrainable16 | 0-2 | 100.0% | 100.0% | 40.8 | 95.1 | 12.2 | 58.6 | 1.01 | 3.26 | 0.061 |
| FullTrainable16 | 2-4 | 100.0% | 100.0% | 40.4 | 95.2 | 13.1 | 64.8 | 1.06 | 3.29 | 0.063 |
| FullTrainable16 | 4-6 | 96.9% | 93.0% | 34.2 | 97.5 | 15.3 | 80.6 | 1.22 | 3.53 | 0.066 |
| FullTrainable16 | 6-8 | 66.0% | 53.9% | 2.3 | 115.4 | 15.9 | 88.2 | 1.53 | 3.84 | 0.062 |
| FullTrainable16 | 8-10 | 34.0% | 19.9% | -30.5 | 166.1 | 14.2 | 78.3 | 1.84 | 3.96 | 0.051 |
| FullTrainable8 | 0-2 | 96.1% | 96.1% | 38.9 | 101.7 | 12.2 | 48.0 | 0.90 | 3.14 | 0.045 |
| FullTrainable8 | 2-4 | 98.8% | 98.4% | 40.4 | 96.6 | 12.5 | 48.7 | 0.93 | 3.12 | 0.047 |
| FullTrainable8 | 4-6 | 92.6% | 85.2% | 30.8 | 106.5 | 12.4 | 51.1 | 1.05 | 3.13 | 0.046 |
| FullTrainable8 | 6-8 | 75.8% | 58.2% | 8.1 | 128.8 | 11.1 | 48.3 | 1.33 | 3.25 | 0.041 |
| FullTrainable8 | 8-10 | 44.5% | 29.7% | -20.2 | 165.8 | 9.0 | 40.2 | 1.63 | 3.15 | 0.034 |

## 8D minus 16D

| Wind (m/s) | Main SR delta | Strict SR delta | Reward delta | Avg KE delta | P95 KE delta |
|---:|---:|---:|---:|---:|---:|
| 0-2 | -3.9 pp | -3.9 pp | -1.9 | -0.1 mJ | -10.7 mJ |
| 2-4 | -1.2 pp | -1.6 pp | +0.1 | -0.6 mJ | -16.1 mJ |
| 4-6 | -4.3 pp | -7.8 pp | -3.4 | -2.9 mJ | -29.5 mJ |
| 6-8 | +9.8 pp | +4.3 pp | +5.8 | -4.8 mJ | -39.9 mJ |
| 8-10 | +10.5 pp | +9.8 pp | +10.2 | -5.1 mJ | -38.1 mJ |

## Failure Distribution

| Model | Wind (m/s) | Termination counts |
|---|---:|---|
| FullTrainable16 | 0-2 | insertion_success 256 |
| FullTrainable16 | 2-4 | insertion_success 256 |
| FullTrainable16 | 4-6 | insertion_success 238, lucky 10, stuck_on_rebar 8 |
| FullTrainable16 | 6-8 | insertion_success 138, lucky 31, stuck_on_rebar 87 |
| FullTrainable16 | 8-10 | insertion_success 51, lucky 36, stuck_on_rebar 126, timeout 43 |
| FullTrainable8 | 0-2 | insertion_success 246, timeout 10 |
| FullTrainable8 | 2-4 | insertion_success 252, lucky 1, stuck_on_rebar 1, timeout 2 |
| FullTrainable8 | 4-6 | insertion_success 218, lucky 19, stuck_on_rebar 5, timeout 14 |
| FullTrainable8 | 6-8 | insertion_success 149, lucky 45, stuck_on_rebar 47, timeout 15 |
| FullTrainable8 | 8-10 | insertion_success 76, lucky 38, stuck_on_rebar 113, timeout 27, early_stop 2 |

## Test Fairness Notes

- The two models use the same script, wind bins, seed, episodes per bin, L8 reset distribution, 10-segment rope, and PID residual base.
- The raw CSV confirms identical wind mean/std for both models in every bin.
- Sim2real modules are disabled for both: logs show `pred=0.00ms`, `clp=0.00ms`, `vis=0%`, `rope=0%`.
- The only intended test-side difference is cable encoder latent dimension: 16D vs 8D.

## Interpretation

- Low wind is saturated for both, but 16D is slightly cleaner at 0-4 m/s.
- At 4-6 m/s, 16D remains better in strict and broad success.
- At 6-10 m/s, 8D is better: it has higher broad success, higher strict success, lower swing kinetic energy, lower swing angle, and lower EE acceleration.
- 8D reduces `stuck_on_rebar` substantially in 6-8 m/s: 87 -> 47. It also reduces timeout at 8-10 m/s: 43 -> 27.
- Current evidence suggests 8D may be a better capacity/regularization point for high-wind robustness, while 16D preserves slightly better low/mid-wind precision.

## Caveat

This is an evaluation of the selected L8-best checkpoints, not a pure architecture-capacity theorem. The 8D run reached a higher realtime L8-best training SR than the 16D run, so the observed 8D advantage may reflect both latent dimensionality and the resulting optimization trajectory.
