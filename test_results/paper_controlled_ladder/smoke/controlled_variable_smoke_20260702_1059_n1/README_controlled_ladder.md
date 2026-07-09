# Controlled-Variable Descent Comparison

Date: 2026-07-02

This folder is for the replacement paper comparison ladder.  The old five-level
paper ladder mixed several difficulty factors in the same step.  This ladder
uses cumulative controlled variables so adjacent rows differ by one intended
factor.

## Design

All profiles keep four rebars/four holes, fixed target XY, no obstacles, the
same robot/payload, the same descent reward, and the same trained PID residual
base for RL policies.  Broad success includes lucky insertions; strict success
is the clean/perfect success rate.

| Profile | Added factor | Rebar/hole ratio | Hole diameter | Rope | Init randomization | Wind |
|---|---|---:|---:|---:|---:|---:|
| `ctrl-c0-base` | Base | 8.0 | 40 mm | 4 seg | 0 mm / 0 rad | 0 m/s |
| `ctrl-c1-init` | Initial disturbance | 8.0 | 40 mm | 4 seg | +/-16 mm / +/-0.004 rad | 0 m/s |
| `ctrl-c2-wind2` | Wind starts | 8.0 | 40 mm | 4 seg | +/-16 mm / +/-0.004 rad | 2 m/s |
| `ctrl-c3-wind6` | Higher wind branch | 8.0 | 40 mm | 4 seg | +/-16 mm / +/-0.004 rad | 6 m/s |
| `ctrl-c3-wind8` | Higher wind branch | 8.0 | 40 mm | 4 seg | +/-16 mm / +/-0.004 rad | 8 m/s |
| `ctrl-c4-rope10-w6` | Rope model | 8.0 | 40 mm | 10 seg | +/-16 mm / +/-0.004 rad | 6 m/s |
| `ctrl-c4-rope10-w8` | Rope model | 8.0 | 40 mm | 10 seg | +/-16 mm / +/-0.004 rad | 8 m/s |
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
| `PID` | Traditional true-observation PID expert | `--algo expert --descent-base-expert traditional_pid` |
| `DampedPD` | Traditional true-observation damped PD expert | `--algo expert --descent-base-expert damped_pd` |
| `MPC` | Traditional true-observation shooting-MPC expert | `--algo expert --descent-base-expert mpc` |
| `Mainline` | Full ideal-observation PID+Residual RL | `saves/descent_ppo_paper_trueobs_residual_pretrain_online_20260627_010320/ckpt_latest.pt` |
| `NoCable` | Ideal-observation PID+Residual RL without cable latent | `saves/descent_ablation_trueobs_no_cable_latent_pretrain_20260627_224522/ckpt_latest.pt` |
| `MainlineP2` | Full PID+Residual RL with ObsPred every other step | `saves/descent_ablation_fullobs_obspred_p2_ft_20260629_145921/ckpt_latest.pt` plus matching `ckpt_latest_obs_predictor.pt` |

Use `ckpt_latest.pt` for this controlled-variable ablation set, matching the
current ablation registry rule.

## Commands

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

