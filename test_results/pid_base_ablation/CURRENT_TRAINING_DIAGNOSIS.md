# Current no-PID teacher distill / finetune diagnosis

Date: 2026-06-25

## Run status

- Stopped the active no-PID PPO finetune run after it remained at 0% success for 4419 episodes.
- Distill run:
  `saves/descent_pidbase_ablation_no_pid_distill_fullact_trueobs_20260625_205240`
- Finetune run:
  `saves/descent_pidbase_ablation_no_pid_finetune_fullact_trueobs_20260625_212842`
- Smoke eval of distilled student:
  `test_results/pid_base_ablation/distill_fullact_no_pid_smoke_eval_n8.log`

## Main evidence

Distillation converged as a supervised problem:

- Final samples: 300000
- Best loss: 0.001641
- Distill rollout level: L4/8
- Teacher-rollout success near the end: about 95-100%

But this success is from teacher-controlled rollouts. In `train_policy_distill`,
the environment still steps with `teacher_actions`, while the student only learns
from observations visited by the teacher.

Closed-loop no-PID behavior is not solved:

- Distilled student smoke eval, no wind, eval curriculum level 0, 8 episodes:
  0% success, average reward -93.28, average 260.4 steps.
- PPO finetune after distill:
  4419 episodes, 0% success, still L0/8.
- PPO first 100 episodes:
  average reward -85.0, average 192.3 steps.
- PPO last 100 episodes:
  average reward -18.9, average 41.1 steps.
- PPO failure counts in text log:
  `instability:tilt` 2337, `instability:yaw` 1581, `instability:vel` 452,
  `insertion_success` 0.

The PPO stage did not recover the task. It moved toward a short-horizon local
solution that gets less dense penalty before quickly becoming unstable.

## Likely root causes

1. Teacher-state distribution shift

   Distillation samples are generated under PID+RL teacher closed-loop control.
   The no-PID student is never used to roll the environment during distillation.
   When finetune starts, the student immediately visits states that the teacher
   rarely visits, especially higher tilt, yaw, payload velocity, and cable energy.

2. The no-PID target action is only an approximation

   The current full-action target is reconstructed by
   `_descent_base_acc_from_phase_obs()` plus a scaled teacher residual. It is not
   the true inverse of the mature PID+RL controller's final `delta_q`, and it does
   not faithfully encode several stabilizing rules from the mature descent
   controller.

3. Missing low-level descent stabilizers

   The mature descent controller contains settling, contact freeze, integral
   clipping, target attraction, swing/catch damping, yaw alignment, and z descent
   gating. The current no-PID execution path is a simpler EE acceleration
   integrator. Removing the PID base therefore removes more than a small base
   action: it removes much of the closed-loop stabilizing structure.

4. Reward is residual-oriented

   The current descent reward includes a "residual intervention regularizer" for
   action magnitude and smoothness. In PID+RL this discourages unnecessary
   residuals. In no-PID full-action training, sustained action is sometimes
   necessary for stabilization, so the same regularizer can bias the policy toward
   overly small or short-sighted actions.

5. PPO starts from a poor success signal

   The no-PID student has 0% success before finetune. PPO then receives mostly
   instability and timeout outcomes, with no insertion successes to anchor the
   curriculum. This makes the L0 curriculum effectively stuck.

## Recommended optimization order

1. Add a closed-loop smoke gate before long PPO finetune

   After distillation, run 16-32 deterministic no-PID true-observation episodes at
   no wind and curriculum level 0. Continue to PPO only if success is non-zero and
   instability is not dominant.

2. Add DAgger-style student roll-in

   During distillation, mix teacher roll-in and student roll-in. When the student
   controls the system and drifts into off-distribution states, still compute a
   teacher or heuristic recovery target from that state and add it to the
   supervised batch.

3. Improve no-PID target generation

   The target should include the stabilizing terms that the mature controller
   already uses:

   - settling hold in the first descent steps,
   - swing/catch damping from `payload_xy - ee_xy` and payload velocity,
   - xy velocity limiting,
   - z descent gate based on xy error, payload speed, yaw error, and near-target
     slowdown,
   - contact/floor-region freeze behavior,
   - yaw/tilt-aware descent suppression.

4. Use a no-PID-specific warmup curriculum

   Before the normal descent curriculum, train on a stability-and-alignment stage:

   - no wind,
   - small initial xy/velocity/tilt perturbation,
   - slower or gated z descent,
   - looser insertion success at first,
   - only advance when closed-loop stability and non-zero success are present.

5. Make reward mode aware of full-action no-PID

   For no-PID full-action training, reduce or disable the residual-style action
   magnitude regularizer at the start. Keep smoothness, but scale it as a safety
   term rather than as a residual-use penalty.

6. Lower PPO disturbance after distill

   Use a smaller actor learning rate and lower exploration at the beginning, but
   treat this as secondary. It will not fix the core distribution shift by itself.

## Immediate conclusion

Do not continue the current finetune configuration. The next useful experiment is
not a longer PPO run; it is a new pretraining pipeline with student roll-in and a
more faithful no-PID full-action target, followed by a short closed-loop smoke
test before any long finetune.

## DAgger/stable-target run diagnosis

Run:
`saves/descent_pidbase_ablation_no_pid_distill_dagger_stable_trueobs_20260625_225616`

This run was stopped at about 182k samples because success collapsed after
student roll-in began.

Success by student roll-in fraction:

- `RI=0`: 257 episodes, 92.6% success, average reward 37.4.
- `0<RI<=0.05`: 125 episodes, 16.8% success, average reward -34.6.
- `0.05<RI<=0.10`: 97 episodes, about 0% success, average reward -59.1.
- `0.10<RI<=0.20`: 233 episodes, 0% success, average reward -73.0.
- `RI>0.20`: 530 episodes, 0% success, average reward -102.9.

Main failure modes at high roll-in:

- Early stop from large xy miss, often 25-160 mm.
- Tilt/yaw instability.
- Timeout with payload still too high or off target.
- Occasional lucky-rebar insert rejection, which means some trajectories still
  reached the plate/rebar region but did not satisfy clean-insert constraints.

Interpretation:

The stable target itself did not break teacher-only rollout; before roll-in the
teacher-controlled trajectories still succeeded. The collapse starts when
student actions are injected. Per-step roll-in is too aggressive for a suspended
payload: even a 5% per-step probability gives several student actions per
episode, and one poor action can create swing/yaw error that the mixed rollout
does not recover from. At 35% roll-in, nearly every episode contains many
student actions, so the log measures an unstable mixed controller rather than a
healthy teacher trajectory with small DAgger perturbations.

Next adjustment:

- Keep the stable full-action target.
- Use much longer teacher-only warmup.
- Reduce final student roll-in probability by at least one order of magnitude.
- Cap student roll-in steps per episode, or force several teacher recovery steps
  after each student step.
- Rename/interpret the logged `teacher_success` metric as mixed-rollout success
  once roll-in is enabled.

## Protected DAgger / stable-target run diagnosis

Run:
`saves/descent_pidbase_ablation_no_pid_distill_dagger_stable_trueobs_20260625_232748`

This run completed normally:

- Samples: 300000
- Duration: 40.7 min
- Best loss: 0.017332
- Final curriculum: L2/8
- Checkpoints: `ckpt_best.pt`, `ckpt_final.pt`, `ckpt_latest.pt`

Training statistics by sample stage:

- `0-50k`: 547 episodes, 92.1% mixed-rollout success, L0, roll-in 0.0000.
- `50-100k`: 548 episodes, 95.4% mixed-rollout success, average level 0.66,
  roll-in 0.0000.
- `100-160k`: 622 episodes, 80.9% mixed-rollout success, average level 1.82,
  roll-in 0.0024.
- `160-220k`: 559 episodes, 58.3% mixed-rollout success, L2, roll-in 0.0059.
- `220-300k`: 686 episodes, 36.2% mixed-rollout success, L2, roll-in 0.0102.

Roll-in sensitivity:

- `RI=0`: 2003 episodes, 95.6% success, average reward 39.4.
- `0<RI<=0.01`: 375 episodes, 6.4% success, average reward -38.1.
- `0.01<RI<=0.02`: 566 episodes, 27.4% success, average reward -20.3.
- `RI>0.02`: 18 episodes, 55.6% success, average reward 7.9.

The last 1000 mixed-rollout terminations were:

- `insertion_success`: 408
- `stuck_on_rebar`: 310
- `lucky_rebar_insert_failure`: 211
- `timeout`: 44
- `early_stop`: 26

Closed-loop no-PID smoke evaluations after this run:

- `ckpt_best.pt`, paper L1 profile with eval curriculum level 0, no wind,
  16 episodes: 0% success, 16/16 timeout, average reward -98.42.
- `ckpt_final.pt`, same setting, 8 episodes: 0% success, 8/8 timeout,
  average reward -100.64.

Interpretation:

Protected DAgger fixed the earlier catastrophic instability. The student is now
stable in closed-loop evaluation, with average swing angle about 2.3 degrees and
no early tilt/yaw failure in the smoke tests. However, it still does not finish
the insertion without the PID base. The dominant training failures also moved
from instability to contact and terminal insertion quality: stuck on rebar,
lucky-contact rejection, and timeout near the final descent region.

This means the next bottleneck is not "more PPO from the same checkpoint". The
next bottleneck is terminal-contact-aware target generation and a no-PID reward /
curriculum that can teach decisive but gentle insertion. PPO should only be
started after a short no-PID smoke gate shows non-zero success at L0.

## Terminal-aware target follow-up

### Terminal lift target

Run:
`saves/descent_pidbase_ablation_no_pid_distill_dagger_terminal_trueobs_20260626_100958`

This run reached L4/8 with good mixed-rollout statistics and best loss 0.013696,
but closed-loop no-PID evaluation failed catastrophically:

- `ckpt_best.pt`, paper L1 profile with eval curriculum level 0, no wind,
  16 episodes: 0% success, 16/16 instability.
- Terminal diagnostics showed payload heights around 0.46-0.70 m, far above the
  descent target.

Interpretation: the explicit "bad contact lift" target was unsafe for direct
full-action no-PID control. The student generalized the upward recovery label
into high-altitude lifting and swing/yaw instability.

The failed run logs were preserved as:

- `test_results/pid_base_ablation/distill_dagger_terminal_lift_failed_20260626_100958.log`
- `test_results/pid_base_ablation/eval_terminal_lift_failed_best_paperl1_cur0_wind0_n16.log`

### Terminal no-lift target

Run:
`saves/descent_pidbase_ablation_no_pid_distill_dagger_terminal_nolift_trueobs_20260626_105556`

The lift label was removed. Training again looked healthy internally:

- Samples: 300000
- Final curriculum: L4/8
- Best loss: 0.013031
- Last 1000 mixed-rollout terminations: 844 insertion successes, 88 lucky
  rejects, 61 stuck-on-rebar failures, 3 early stops, 3 timeouts.

Closed-loop no-PID evaluation still failed:

- Before the no-upward safety constraint, `ckpt_best.pt` had 0/16 success and
  16/16 instability.
- After adding the no-upward-z descent safety constraint, the same checkpoint
  became stable but still had 0/8 success. It descended to the floor region while
  remaining about 10 cm off target.

Interpretation: the safety constraint fixes the high-altitude instability, but
the policy still does not learn reliable horizontal recovery from its own
off-distribution states.

### Terminal no-upward DAgger

Run:
`saves/descent_pidbase_ablation_no_pid_distill_dagger_terminal_noup_trueobs_20260626_114331`

This run used stronger student roll-in with no-upward-z safety. It is not a good
candidate:

- Because `WIND_MAX=0`, the curriculum jumped directly to L8/8.
- Success collapsed as roll-in increased: final stage mixed-rollout success was
  about 17%.
- Last 1000 mixed-rollout terminations were dominated by stuck-on-rebar and
  lucky-rebar failures.
- Closed-loop eval, L0/no wind, 8 episodes: 0% success. Failures were timeout or
  early stop; terminal diagnostics showed either large xy miss or near-target
  contact with excessive tilt.

### Current conclusion

Do not start no-PID PPO finetune from any of the terminal-aware distillation
checkpoints above. The smoke gate remains at 0% success.

The useful code changes to keep are:

- terminal diagnostics in `test_phase.py`,
- the no-PID-only `no_upward_z` descent safety constraint,
- saved scripts and logs that clearly distinguish stable, terminal-lift,
  terminal-nolift, and terminal-noup experiments.

The terminal-aware target itself should not be treated as solved. The evidence
now supports the paper argument that the PID base is carrying essential
closed-loop structure. Matching the same PPO network directly to full EE
acceleration is not enough; it needs a different action parameterization,
stronger structured controller, or substantially different staged recovery
training.
