# Recovered Project Memory (2026-08-05)

## Sources checked

- Repository `README.md` and `test_results/` formal records.
- Global Codex history under `C:/Users/xyzha/.codex/sessions/` and SQLite
  state/log databases. The project-local `.codex/` directory is empty.
- Frozen Figure 1 assets in
  `test_results/figure_assets/pipeline_paper_direction_success_search_20260725_v6/`.

## Paper state recovered

- Main task: flexible cable-suspended payload precision insertion with a KUKA
  arm and residual RL descent control.
- Headline controlled-variable ladder: C1 initial randomization, C2 ten-segment
  rope, C3 high wind (6-10 m/s), C4 ratio-4 precision insertion.
- Main comparison methods: PID, Damped-PD, MPC, and Residual RL Full8D.
- Current policy mainline: trainable 8D cable encoder with PID-base residual PPO;
  the 5 Hz observation / 10 Hz control world-model branch is an ablation.
- Frozen Figure 1 trajectory: NMPC cruise followed by true-observation 8D
  residual PPO descent, wind 0 m/s, episode 3, handoff step 118, strict
  insertion success at step 215; 4.5 mm lateral error and 16.1 mm insertion.

## Generated artifact

`outputs/figure1_complete/figure1_complete.png` and `.pdf` are the complete
paper Figure 1 composition based on the supplied sketch. Rebuild with:

```text
python scripts/analysis/build_figure1_complete.py
```
