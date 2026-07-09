# Controlled-Variable Ladder Summary

Cell format: broad success / strict success (%). Broad includes lucky insertions.

| Profile | Added factor | Wind | Rope | Ratio | PID | Damped-PD | MPC | Mainline | NoCable | Mainline P2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ctrl-c0-base` | base | 0 | 4 | 8.0 | 0.0 / 0.0 | 0.0 / 0.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |
| `ctrl-c1-init` | init | 0 | 4 | 8.0 | 6.2 / 0.0 | 18.8 / 0.0 | 93.8 / 93.8 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |
| `ctrl-c2-wind2` | wind 2 | 2 | 4 | 8.0 | 9.4 / 3.1 | 12.5 / 0.0 | 96.9 / 96.9 | 100.0 / 100.0 | 100.0 / 100.0 | 93.8 / 93.8 |
| `ctrl-c3-wind6` | wind 6 | 6 | 4 | 8.0 | 9.4 / 3.1 | 3.1 / 0.0 | 50.0 / 46.9 | 100.0 / 100.0 | 100.0 / 100.0 | 34.4 / 25.0 |
| `ctrl-c3-wind8` | wind 8 | 8 | 4 | 8.0 | 3.1 / 0.0 | 3.1 / 0.0 | 43.8 / 37.5 | 96.9 / 75.0 | 100.0 / 100.0 | 37.5 / 9.4 |
| `ctrl-c4-rope10-w6` | rope 10 | 6 | 10 | 8.0 | 9.4 / 0.0 | 3.1 / 0.0 | 0.0 / 0.0 | 93.8 / 93.8 | 100.0 / 100.0 | 65.6 / 50.0 |
| `ctrl-c4-rope10-w8` | rope 10 | 8 | 10 | 8.0 | 6.2 / 0.0 | 6.2 / 0.0 | 0.0 / 0.0 | 100.0 / 65.6 | 100.0 / 100.0 | 25.0 / 6.2 |
| `ctrl-c5-ratio4-w6` | ratio 4 | 6 | 10 | 4.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 84.4 / 68.8 | 100.0 / 93.8 | 59.4 / 37.5 |
| `ctrl-c5-ratio4-w8` | ratio 4 | 8 | 10 | 4.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 40.6 / 31.2 | 81.2 / 62.5 | 21.9 / 9.4 |

Generated files:

- `controlled_ladder_summary.csv`
- `controlled_ladder_success.png`
- `controlled_ladder_heatmap_broad.png`
- `controlled_ladder_failure_composition.png`
