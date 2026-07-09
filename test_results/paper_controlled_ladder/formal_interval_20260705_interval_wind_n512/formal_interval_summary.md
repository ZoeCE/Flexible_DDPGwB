# Formal Interval Ladder Summary

Cell format: broad success / strict success (%).

## Main Controlled Comparison

| Level | Wind interval | PID | Damped-PD | MPC | Residual RL Full-8D |
|---|---:|---:|---:|---:|---:|
| C0 base | 0-2 m/s | 60.7 / 0.0 | 40.2 / 0.0 | 92.4 / 65.8 | 100.0 / 100.0 |
| C1 + init randomization | 0-2 m/s | 58.8 / 0.4 | 65.6 / 0.8 | 83.0 / 72.5 | 100.0 / 100.0 |
| C2 + 10-segment rope | 0-2 m/s | 25.8 / 0.4 | 24.0 / 0.4 | 0.0 / 0.0 | 100.0 / 100.0 |
| C3 + high wind | 6-10 m/s | 25.2 / 0.8 | 24.0 / 0.4 | 0.0 / 0.0 | 92.8 / 92.0 |
| C4 + ratio 4 precision | 6-10 m/s | 1.2 / 0.0 | 0.4 / 0.0 | 0.0 / 0.0 | 77.3 / 68.2 |

## High-Wind Highest-Difficulty Ablation

| Variant | Broad / strict success (%) | Avg KE (mJ) | Avg angle (deg) |
|---|---:|---:|---:|
| Residual RL Full-8D | 79.3 / 69.5 | 10.6 | 1.47 |
| w/o Cable | 61.3 / 48.2 | 8.8 | 1.61 |
| Full-16D | 74.6 / 67.0 | 15.7 | 1.69 |
| w/o LSTM (MLP, trainable 8D) | 81.6 / 66.6 | 13.2 | 1.80 |
