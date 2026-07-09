# Controlled-Variable Ladder Summary

Cell format: broad success / strict success (%). Broad includes lucky insertions.

| Profile | Added factor | Wind | Rope | Ratio | PID | Damped-PD | MPC | Mainline | NoCable | Mainline P2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ctrl-c0-base` | base | 0 | 4 | 8.0 | 100.0 / 0.0 | 100.0 / 0.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |
| `ctrl-c1-init` | init | 0 | 4 | 8.0 | 56.2 / 0.0 | 59.4 / 0.0 | 96.9 / 96.9 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |
| `ctrl-c2-rope10` | rope 10 | 0 | 10 | 8.0 | 28.1 / 0.0 | 34.4 / 0.0 | 0.0 / 0.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |
| `ctrl-c3-wind2` | wind 2 | 2 | 10 | 8.0 | 25.0 / 3.1 | 34.4 / 0.0 | 0.0 / 0.0 | 100.0 / 100.0 | 100.0 / 100.0 | 96.9 / 96.9 |
| `ctrl-c4-wind6` | wind 6 | 6 | 10 | 8.0 | 31.2 / 3.1 | 18.8 / 0.0 | 0.0 / 0.0 | 93.8 / 93.8 | 100.0 / 100.0 | 62.5 / 56.2 |
| `ctrl-c4-wind8` | wind 8 | 8 | 10 | 8.0 | 25.0 / 0.0 | 28.1 / 0.0 | 0.0 / 0.0 | 100.0 / 68.8 | 100.0 / 100.0 | 28.1 / 12.5 |
| `ctrl-c5-ratio4-w6` | ratio 4 | 6 | 10 | 4.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 84.4 / 65.6 | 100.0 / 93.8 | 59.4 / 40.6 |
| `ctrl-c5-ratio4-w8` | ratio 4 | 8 | 10 | 4.0 | 0.0 / 0.0 | 3.1 / 0.0 | 0.0 / 0.0 | 31.2 / 28.1 | 75.0 / 53.1 | 9.4 / 3.1 |

Generated files:

- `controlled_ladder_summary.csv`
- `controlled_ladder_success.png`
- `controlled_ladder_heatmap_broad.png`
- `controlled_ladder_failure_composition.png`
