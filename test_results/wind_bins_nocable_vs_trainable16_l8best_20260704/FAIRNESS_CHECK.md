# Test Fairness Check

Date: 2026-07-04

## Verdict

This wind-bin test is fair for the intended question:

> Does adding true cable information through the trainable 16D cable encoder improve the policy over the NoCable policy?

Yes. The test keeps the environment, task difficulty, wind sequence, reset sequence, controller base, and sim2real modules aligned. The intended variable is whether the policy receives cable information.

## What Is Matched

- Same phase: descent.
- Same algorithm at evaluation: PPO residual policy.
- Same checkpoint rule: `ckpt_best_l8.pt`, selected after reaching curriculum L8.
- Same episode count: 256 episodes per wind bin.
- Same wind bins: 0-2, 2-4, 4-6, 6-8, 8-10 m/s.
- Same random seed: `280704`.
- Same wind sampling code: `case_seed = seed + 1009 * bin_id`.
- Same sampled wind statistics in the raw CSV:

| Wind bin | NoCable wind mean/std | Full16 wind mean/std |
|---:|---:|---:|
| 0-2 | 1.0225 / 0.5695 | 1.0225 / 0.5695 |
| 2-4 | 2.9653 / 0.5792 | 2.9653 / 0.5792 |
| 4-6 | 4.8972 / 0.5900 | 4.8972 / 0.5900 |
| 6-8 | 7.0019 / 0.5520 | 7.0019 / 0.5520 |
| 8-10 | 9.0219 / 0.5620 | 9.0219 / 0.5620 |

- Same L8 initialization:
  - init XY: +/-20 mm
  - init payload XY velocity: +/-0.012 m/s
  - init tilt: +/-0.005 rad
  - XY tolerance: 5 mm
  - z/tilt/yaw tolerance: 20 mm / 0.050 rad / 0.080 rad
- Same MuJoCo cable model during eval:
  - 10 cable segments
  - segment length 0.04 m
  - total chain length 0.4 m
- Same controller setting:
  - PID residual mode enabled
  - residual dq scale 0.8
  - residual acc max XY/Z 0.6 / 0.9
  - base descent vertical speed 0.03 m/s
  - max steps 300
- Same sim2real disable flags:
  - vision disabled
  - observation predictor disabled
  - cable latent predictor disabled
  - logs show `pred=0.00ms`, `clp=0.00ms`, `vis=0%`, `rope=0%`.

## Intended Difference

The only intentional test-side difference is:

- `NoCable`: evaluated with `--disable-cable-obs`, so the policy input removes cable latent dimensions.
- `FullTrainable16`: evaluated with the trainable 16D cable encoder enabled, so the policy receives true cable-state information encoded into 16 dimensions.

This is fair for a cable-information ablation. It is not a parameter-count-matched architecture test, and it is not a deploy-sensor test with vision/marker reconstruction.

## Does Full16D Really Beat NoCable?

Yes, but mainly in medium/high wind. Low wind is saturated for both.

`Main SR` is broad success rate, including lucky insert. `Strict SR` is strict insertion success.

| Wind bin | NoCable Main SR | Full16 Main SR | Main delta | NoCable Strict SR | Full16 Strict SR | Strict delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0-2 | 100.0% | 99.2% | -0.8 pp | 99.6% | 99.2% | -0.4 pp |
| 2-4 | 96.9% | 100.0% | +3.1 pp | 96.1% | 100.0% | +3.9 pp |
| 4-6 | 74.6% | 97.3% | +22.7 pp | 66.8% | 93.4% | +26.6 pp |
| 6-8 | 28.1% | 69.1% | +41.0 pp | 12.1% | 56.6% | +44.5 pp |
| 8-10 | 22.7% | 33.6% | +10.9 pp | 12.9% | 16.8% | +3.9 pp |

The strongest evidence is from 4-8 m/s:

- 4-6 m/s: Full16 improves main SR by 22.7 percentage points and strict SR by 26.6 points.
- 6-8 m/s: Full16 improves main SR by 41.0 points and strict SR by 44.5 points.

At 8-10 m/s, Full16 still improves broad/main success, but strict success remains low for both models. This bin should be interpreted as beyond or near the edge of the current robust operating range.

## Failure Evidence

Full16D mainly reduces `stuck_on_rebar` failures:

| Wind bin | NoCable stuck | Full16 stuck |
|---:|---:|---:|
| 2-4 | 8 | 0 |
| 4-6 | 64 | 7 |
| 6-8 | 183 | 79 |
| 8-10 | 163 | 121 |

This matches the intended role of cable information: it helps the policy infer payload/cable configuration under disturbance and avoid insertion/contact failure.

## Caveat

The CSV stores aggregate results per bin, not per-episode paired success/failure labels. Because of that, an exact paired test such as McNemar cannot be run from the saved files. However, the raw CSV confirms identical wind mean/std per bin, and the benchmark code deterministically reuses the same wind and reset seed arrays for both runs. The aggregate differences in 4-8 m/s are large enough that the qualitative conclusion is robust.
