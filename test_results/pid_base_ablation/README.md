# PID Base Ablation: No-PID Student

Goal: compare the mature descent `PID base + RL residual` design against the
same PPO network used as a direct EE-acceleration policy without the PID base.

Protocol:

1. Distill a no-PID student from a mature full-ideal-observation PID+RL teacher.
   The teacher still provides the supervised label, but DAgger-style student
   roll-in can take over part of the environment rollout. Teacher rollout steps
   use PID+RL action semantics; student roll-in steps use no-PID full EE
   acceleration semantics.
2. The default no-PID distillation target is
   `descent_stable_acc_plus_actor`: a mature-descent-controller-inspired full
   EE acceleration target plus a scaled teacher residual. The target includes
   target attraction, swing/catch damping, settling/contact hold, z descent
   gating, near-target slowdown, and tilt/yaw-aware descent suppression.
3. Fine-tune the distilled student with PPO under true environment observations.
   Vision, observation predictor, and cable latent predictor are disabled.
4. Evaluate the fine-tuned no-PID student on the paper five-level ladder with
   the same true-observation setting used for traditional experts.

Default teacher:

`saves/descent_ppo_variable_wind_true_obs_finetune_20260531_110809/ckpt_best.pt`

Override it by exporting `TEACHER=/path/to/ckpt.pt` before running the scripts.

Main scripts:

- `run_distill_trueobs_no_pid.sh`
- `run_finetune_trueobs_no_pid.sh`
- `run_distill_then_finetune_no_pid_trueobs.sh`
- `run_eval_ladder_no_pid_trueobs.sh`

The official long-run outputs should stay under:

- `saves/descent_pidbase_ablation_no_pid_distill_fullact_trueobs_*`
- `saves/descent_pidbase_ablation_no_pid_finetune_fullact_trueobs_*`
- `test_results/pid_base_ablation/eval_ladder_no_pid_trueobs/`

New DAgger/stable-target runs use:

- `saves/descent_pidbase_ablation_no_pid_distill_dagger_stable_trueobs_*`
- `saves/descent_pidbase_ablation_no_pid_finetune_dagger_stable_trueobs_*`

Terminal-aware DAgger runs use:

- `saves/descent_pidbase_ablation_no_pid_distill_dagger_terminal_noup_trueobs_*`
- `saves/descent_pidbase_ablation_no_pid_finetune_dagger_terminal_noup_trueobs_*`
- `saves/descent_pidbase_ablation_no_pid_distill_dagger_terminal_nolift_trueobs_*`
- `saves/descent_pidbase_ablation_no_pid_finetune_dagger_terminal_nolift_trueobs_*`
- `saves/descent_pidbase_ablation_no_pid_distill_dagger_terminal_trueobs_*`
- `saves/descent_pidbase_ablation_no_pid_finetune_dagger_terminal_trueobs_*`

For no-PID full-action PPO finetune, use a larger free action RMS and a much
smaller action-magnitude penalty than the residual PID+RL policy. The default
finetune script now uses `ACTION_RMS_FREE=1.0` and `ACTION_MAG_COEF=0.02`.
