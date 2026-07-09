# Controlled-Variable Descent Comparison

Date: 2026-07-02

This folder is for the replacement paper comparison ladder.  The old five-level
paper ladder mixed several difficulty factors in the same step.  This ladder
uses cumulative controlled variables so adjacent rows differ by one intended
factor.  The final order is: initial disturbance, rope-model complexity, wind,
then hole precision.

## Design

All profiles keep four rebars/four holes, fixed target XY, no obstacles, the
same robot/payload, the same descent reward, and the same trained PID residual
base for RL policies.  Broad success includes lucky insertions; strict success
is the clean/perfect success rate.

| Profile | Added factor | Rebar/hole ratio | Hole diameter | Rope | Init randomization | Wind |
|---|---|---:|---:|---:|---:|---:|
| `ctrl-c0-base` | Base | 8.0 | 40 mm | 4 seg | 0 mm / 0 rad | 0 m/s |
| `ctrl-c1-init` | Initial disturbance | 8.0 | 40 mm | 4 seg | +/-16 mm / +/-0.004 rad | 0 m/s |
| `ctrl-c2-rope10` | Rope model | 8.0 | 40 mm | 10 seg | +/-16 mm / +/-0.004 rad | 0 m/s |
| `ctrl-c3-wind2` | Wind starts | 8.0 | 40 mm | 10 seg | +/-16 mm / +/-0.004 rad | 2 m/s |
| `ctrl-c4-wind6` | Higher wind branch | 8.0 | 40 mm | 10 seg | +/-16 mm / +/-0.004 rad | 6 m/s |
| `ctrl-c4-wind8` | Higher wind branch | 8.0 | 40 mm | 10 seg | +/-16 mm / +/-0.004 rad | 8 m/s |
| `ctrl-c5-ratio4-w6` | Hole precision | 4.0 | 20 mm | 10 seg | +/-16 mm / +/-0.004 rad | 6 m/s |
| `ctrl-c5-ratio4-w8` | Hole precision | 4.0 | 20 mm | 10 seg | +/-16 mm / +/-0.004 rad | 8 m/s |

Implementation notes:

- The physical socket hole size is changed for this controlled experiment.
- The curriculum XY tolerance and physical rebar-alignment tolerance follow the
  hole clearance.
- The policy-observation normalization tolerances are not retuned with the
  profile, so task difficulty is not confounded with a large input-scale change.
- The 6 m/s and 8 m/s branches are both run so the final paper plot can choose
  the branch that shows a useful degradation curve without saturating at zero.

## Compared Methods

| Method | Meaning | Checkpoint / setting |
|---|---|---|
| `PID` | Traditional true-observation PID expert | `--algo expert --descent-base-expert traditional_pid`, tuned by `tuned_traditional_experts_20260702.json` |
| `DampedPD` | Traditional true-observation damped PD expert | `--algo expert --descent-base-expert damped_pd`, tuned by `tuned_traditional_experts_20260702.json` |
| `MPC` | Traditional true-observation shooting-MPC expert | `--algo expert --descent-base-expert mpc` |
| `Mainline` | Full ideal-observation PID+Residual RL | `saves/descent_ppo_paper_trueobs_residual_pretrain_online_20260627_010320/ckpt_latest.pt` |
| `NoCable` | Ideal-observation PID+Residual RL without cable latent | `saves/descent_ablation_trueobs_no_cable_latent_pretrain_20260627_224522/ckpt_latest.pt` |
| `MainlineP2` | Full PID+Residual RL with ObsPred every other step | `saves/descent_ablation_fullobs_obspred_p2_ft_20260629_145921/ckpt_latest.pt` plus matching `ckpt_latest_obs_predictor.pt` |

Use `ckpt_latest.pt` for this controlled-variable ablation set, matching the
current ablation registry rule.

## Commands

First run a 32-episode smoke gate.  The older 1-episode smoke only checked that
the code path did not crash; it is not a stable success-rate estimate.

```powershell
wsl -e bash -lc "cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB && EPISODES=32 RUN_ID=smoke32_20260702_v2 OUT=test_results/paper_controlled_ladder/smoke/controlled_variable_smoke32_20260702_v2_n32 bash test_results/paper_controlled_ladder/launch_controlled_variable_ladder_20260702.sh"
```

Audit the generated configuration before launching a formal run:

```powershell
wsl -e bash -lc "cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB && source /home/xyzha/miniconda3/etc/profile.d/conda.sh && conda activate vsdrl_env_5060 && python test_results/paper_controlled_ladder/audit_controlled_ladder_config.py --out test_results/paper_controlled_ladder/config_audit_20260702.csv"
```

Launch the full 256-episode-per-group run from Windows PowerShell:

```powershell
wsl -e bash -lc "cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB && EPISODES=256 bash test_results/paper_controlled_ladder/launch_controlled_variable_ladder_20260702.sh"
```

Observe progress:

```powershell
wsl -e bash -lc "cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB && bash test_results/paper_controlled_ladder/status_controlled_variable_ladder_20260702.sh"
```

Generate or refresh tables and figures for the latest run:

```powershell
wsl -e bash -lc "cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB && bash test_results/paper_controlled_ladder/status_controlled_variable_ladder_20260702.sh --report"
```

## Current Gate Status

- `20260702_controlled_v1_n256` was stopped before producing JSON summaries
  because the P2 command inherited `--disable-obs-predictor` from common args.
- The runner was fixed so only non-P2 variants pass `--disable-obs-predictor`.
  `MainlineP2` now has `obs_predictor_enabled=True`, `obs_period=2`, and
  target mode `non_cable_latent`.
- Config audit saved to `config_audit_20260702.csv`.
- Full 32-episode smoke after the P2 fix completed:
  `test_results/paper_controlled_ladder/smoke/controlled_variable_smoke32_20260702_v2_n32`.
- PID and Damped-PD were tuned on `ctrl-c0-base`, `ctrl-c1-init`, and
  the old `ctrl-c2-wind2` with 16 episodes/profile.  Source data:
  `tuning_pid_pd_20260702/tuning_results.csv`.
- Fixed traditional baseline params are saved in
  `tuned_traditional_experts_20260702.json`.  The runner applies them only to
  `PID` and `DampedPD`; MPC and all RL variants are unchanged.
- Tuned PID/Damped-PD 32-episode smoke completed:
  `test_results/paper_controlled_ladder/smoke/controlled_variable_smoke32_tuned_pidpd_20260702_n32`.
  Main broad success pattern: C0=100% for both, C1/C2/C3 nonzero, C4 drops
  after switching to 10 rope segments, C5 drops to 0 after tightening ratio to
  4.0.
- Formal 256-episode run with the old order was stopped after the order change
  request and should not be used as final data:
  `test_results/paper_controlled_ladder/controlled_variable_256_tuned_pidpd_20260702_n256`.
- Current ladder definition has 8 profiles x 6 methods = 48 groups.
- Rope-first config audit saved to
  `test_results/paper_controlled_ladder/config_audit_ropefirst_20260702.csv`.
- Rope-first 32-episode smoke started:
  `test_results/paper_controlled_ladder/smoke/controlled_variable_smoke32_ropefirst_20260702_n32`.

Key non-variable alignment after the audit:

- target/start: `[-0.3, 0.2]` and `[0.3, 0.15]`;
- control rate: `10 Hz`;
- payload mass: `4.7 kg`;
- four rebars/four holes and fixed target XY;
- actual obstacles: `0`, while default obstacle slots remain configured for
  observation-layout compatibility;
- Mainline/MainlineP2 use 76D full cable observation; NoCable uses 44D with the
  cable encoder removed;
- vision and CableLatPred are disabled for this controlled comparison;
- P2 is the only variant with ObsPred enabled.
