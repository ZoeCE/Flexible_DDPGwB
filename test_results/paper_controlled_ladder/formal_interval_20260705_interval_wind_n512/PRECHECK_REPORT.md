# Formal Interval Ladder Precheck

Generated: 2026-07-06 17:56:36

## Required L8-Best Checkpoints

- Full8D: `OK` `saves/descent_ppo_paper_trueobs_residual_pretrain_trainable8_seed270627_20260704/ckpt_best_l8.pt`
- NoCable: `OK` `saves/descent_ablation_trueobs_no_cable_latent_pretrain_l8best_20260702_parallel_l8best/ckpt_best_l8.pt`
- Full16D: `OK` `saves/descent_ppo_paper_trueobs_residual_pretrain_trainable16_seed270627_20260703/ckpt_best_l8.pt`
- MLP8D: `OK` `saves/descent_ablation_fullobs_mlp_trainable8_seed270627_20260705/ckpt_best_l8.pt`

- Traditional overrides: `OK` `test_results/paper_controlled_ladder/tuned_traditional_experts_20260702.json`

## Skipped Formal Candidates

- MainlineP2: No ckpt_best_l8.pt is registered for the ObsPred P2 finetune. Do not substitute ckpt_latest.pt for the formal L8-best protocol.
- NoPID: No successful L8-best no-PID/full-action checkpoint exists; current no-PID runs are recorded as failed evidence only.
- LegacyMLP32: The historical A5 MLP checkpoint used a frozen 32D cable encoder. It is preserved in old JSON files but excluded from the formal paper summary after the matched trainable-8D MLP run became available.

## Protocol

- Wind is sampled uniformly per episode from the level interval.
- All variants in one level reuse the same wind, wind-direction, and reset-seed schedule.
- Main comparison: PID, Damped-PD, MPC, Full8D Residual RL.
- High-wind ablation: Full8D, NoCable, Full16D, matched MLP8D on C4 only.
- The formal runner is single-environment and serial to avoid shared MuJoCo XML races.
