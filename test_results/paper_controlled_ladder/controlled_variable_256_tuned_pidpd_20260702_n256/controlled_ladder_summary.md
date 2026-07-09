# Controlled-Variable Ladder Summary

Cell format: broad success / strict success (%). Broad includes lucky insertions.

| Profile | Added factor | Wind | Rope | Ratio | PID | Damped-PD | MPC | Mainline | NoCable | Mainline P2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ctrl-c0-base` | base | 0 | 4 | 8.0 | 100.0 / 0.0 | 100.0 / 0.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |
| `ctrl-c1-init` | init | 0 | 4 | 8.0 | 58.2 / 0.8 | 61.3 / 0.8 | 95.3 / 95.3 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |
| `ctrl-c2-wind2` | wind 2 | 2 | 4 | 8.0 | 60.9 / 0.0 | 68.8 / 0.0 | 94.5 / 94.5 | 100.0 / 100.0 | 100.0 / 100.0 | 94.1 / 94.1 |
| `ctrl-c3-wind6` | wind 6 | 6 | 4 | 8.0 | 64.8 / 0.4 | 62.5 / 0.0 | 66.4 / 63.3 | 99.2 / 99.2 | 100.0 / 100.0 | 47.3 / 30.5 |
| `ctrl-c3-wind8` | wind 8 | 8 | 4 | 8.0 | 64.5 / 1.2 | 62.1 / 0.0 | 39.1 / 32.4 | 97.7 / 74.2 | 100.0 / 100.0 | pending |
| `ctrl-c4-rope10-w6` | rope 10 | 6 | 10 | 8.0 | pending | pending | pending | pending | pending | pending |
| `ctrl-c4-rope10-w8` | rope 10 | 8 | 10 | 8.0 | pending | pending | pending | pending | pending | pending |
| `ctrl-c5-ratio4-w6` | ratio 4 | 6 | 10 | 4.0 | pending | pending | pending | pending | pending | pending |
| `ctrl-c5-ratio4-w8` | ratio 4 | 8 | 10 | 4.0 | pending | pending | pending | pending | pending | pending |

Generated files:

- `controlled_ladder_summary.csv`
- `controlled_ladder_success.png`
- `controlled_ladder_heatmap_broad.png`
- `controlled_ladder_failure_composition.png`
