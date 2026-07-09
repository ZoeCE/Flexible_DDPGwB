# Wind-bin L8-best Test Summary

Date: 2026-07-04

## Setup

- Models: `NoCable` vs `FullTrainable16`.
- Checkpoint rule: best checkpoint after reaching curriculum L8, i.e. `ckpt_best_l8.pt`.
- Episodes: 256 per wind bin per model.
- Wind bins: 0-2, 2-4, 4-6, 6-8, 8-10 m/s.
- Eval setting: descent phase, true observation, no vision, no observation predictor, no cable latent predictor.
- Control setting: PID residual base enabled.
- Environment: default L8 descent curriculum, 10 cable segments, 3 rebars, fixed target XY.
- Wind setting: `wind_speed_max=10`, variable wind enabled with speed/dir random walk matching the training profile.

## Files

- Combined raw metrics: `combined_wind_bins_process_metrics.csv`
- Combined readable table: `combined_wind_bins_process_metrics_readable.csv`
- JSON readable table: `combined_wind_bins_process_metrics.json`
- Raw NoCable CSV: `nocable_l8best_wind_bins.csv`
- Raw FullTrainable16 CSV: `full_trainable16_l8best_wind_bins.csv`
- Logs: `wind_bins_l8best.stdout.log`
- Plots: `plots_nocable/`, `plots_full16/`

## Main Results

`Main SR` is broad success rate, i.e. strict insertion success plus lucky insert. `Strict SR` is the perfect/strict success rate.

| Model | Wind (m/s) | Main SR | Strict SR | Avg reward | Avg steps | Avg KE (mJ) | P95 KE (mJ) | Avg swing (deg) | P95 swing (deg) | Avg EE acc (m/s^2) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NoCable | 0-2 | 100.0% | 99.6% | 39.8 | 95.7 | 11.6 | 51.9 | 1.09 | 3.51 | 0.047 |
| NoCable | 2-4 | 96.9% | 96.1% | 36.8 | 97.9 | 11.4 | 50.8 | 1.10 | 3.49 | 0.048 |
| NoCable | 4-6 | 74.6% | 66.8% | 12.3 | 107.2 | 11.3 | 49.5 | 1.25 | 3.54 | 0.048 |
| NoCable | 6-8 | 28.1% | 12.1% | -34.4 | 133.4 | 9.6 | 43.5 | 1.43 | 3.36 | 0.042 |
| NoCable | 8-10 | 22.7% | 12.9% | -40.8 | 179.0 | 8.4 | 42.3 | 1.77 | 3.58 | 0.034 |
| FullTrainable16 | 0-2 | 99.2% | 99.2% | 40.1 | 96.6 | 12.2 | 57.8 | 1.01 | 3.24 | 0.061 |
| FullTrainable16 | 2-4 | 100.0% | 100.0% | 40.4 | 95.3 | 13.3 | 66.0 | 1.06 | 3.35 | 0.063 |
| FullTrainable16 | 4-6 | 97.3% | 93.4% | 34.5 | 96.8 | 15.3 | 80.8 | 1.22 | 3.57 | 0.065 |
| FullTrainable16 | 6-8 | 69.1% | 56.6% | 4.2 | 110.9 | 16.3 | 88.7 | 1.52 | 3.80 | 0.063 |
| FullTrainable16 | 8-10 | 33.6% | 16.8% | -32.7 | 173.4 | 13.5 | 76.0 | 1.79 | 3.74 | 0.049 |

## FullTrainable16 minus NoCable

| Wind (m/s) | Main SR delta | Strict SR delta | Reward delta | Stuck-on-rebar reduction |
|---:|---:|---:|---:|---:|
| 0-2 | -0.8 pp | -0.4 pp | +0.3 | - |
| 2-4 | +3.1 pp | +3.9 pp | +3.6 | 8 -> 0 |
| 4-6 | +22.7 pp | +26.6 pp | +22.2 | 64 -> 7 |
| 6-8 | +41.0 pp | +44.5 pp | +38.6 | 183 -> 79 |
| 8-10 | +10.9 pp | +3.9 pp | +8.1 | 163 -> 121 |

## Failure Distribution

| Model | Wind (m/s) | Termination counts |
|---|---:|---|
| NoCable | 0-2 | insertion_success 255, lucky 1 |
| NoCable | 2-4 | insertion_success 246, lucky 2, stuck_on_rebar 8 |
| NoCable | 4-6 | insertion_success 171, lucky 20, stuck_on_rebar 64, timeout 1 |
| NoCable | 6-8 | insertion_success 31, lucky 41, stuck_on_rebar 183, timeout 1 |
| NoCable | 8-10 | insertion_success 33, lucky 25, stuck_on_rebar 163, timeout 34, early_stop 1 |
| FullTrainable16 | 0-2 | insertion_success 254, timeout 2 |
| FullTrainable16 | 2-4 | insertion_success 256 |
| FullTrainable16 | 4-6 | insertion_success 239, lucky 10, stuck_on_rebar 7 |
| FullTrainable16 | 6-8 | insertion_success 145, lucky 32, stuck_on_rebar 79 |
| FullTrainable16 | 8-10 | insertion_success 43, lucky 43, stuck_on_rebar 121, timeout 47, early_stop 2 |

## Interpretation

- FullTrainable16 is clearly better once wind exceeds 4 m/s. The strongest separation is 6-8 m/s: main SR improves from 28.1% to 69.1%, strict SR from 12.1% to 56.6%.
- Cable information mainly reduces `stuck_on_rebar`. This is especially visible in 4-6 m/s and 6-8 m/s, where stuck failures drop from 64 to 7 and from 183 to 79.
- NoCable has lower swing kinetic energy and lower EE acceleration, but this is not necessarily better behavior. It often gets stuck before completing insertion, so the lower energy partly reflects a more conservative or stalled trajectory.
- Both policies degrade sharply at 8-10 m/s. FullTrainable16 still improves broad success, but strict success remains only 16.8%, so this region is outside the current robust operating range.
- The practical ability boundary for the current FullTrainable16 policy is around 6-8 m/s: it is still useful but no longer near-saturated. A high-wind focused finetune should target 6-10 m/s if this regime is important.

## Training Note

The trainable-8D cable encoder run is still in progress and is not included in this completed test summary.
