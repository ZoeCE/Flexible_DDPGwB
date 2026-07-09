# Paper Final Experiment Design

Date: 2026-07-05

This note consolidates the current descent evidence into a smaller paper-ready
experiment plan.  It supersedes the old mixed-factor five-level ladder as the
main comparison design, but keeps the old official sim2real run as supporting
evidence.

All paper-facing result tables should be emitted as LaTeX tables from now on.
The current table set is stored in:

`test_results/paper_final_experiment_tables_20260705.tex`

## Current Evidence

- The old paper ladder showed a strong headline result: full sim2real residual
  RL achieved 100/100/96.9/90.6/78.1% strict success from L1 to L5, while
  tuned traditional experts degraded sharply.  This is useful evidence, but the
  old ladder mixed rope complexity, wind, geometry, and precision in the same
  difficulty changes.
- Traditional baselines are now usable as representative non-learning
  controllers, not state-of-the-art crane controllers.  MPC is strong on easy
  tasks but loses robustness as the model mismatch grows; PID/Damped-PD often
  rely on broad/lucky insertion and have near-zero strict success in controlled
  smoke.
- Cable-state information is meaningful under high wind.  In the L8 wind-bin
  tests, full cable latent greatly reduced stuck-on-rebar failures compared
  with no-cable policies.
- The current true-observation mainline is trainable 8D CableEncoder, selected
  by L8-best checkpoint.  Legacy frozen 32D is historical only.  All new
  mainline evaluations must explicitly use the 8D checkpoint and matching
  dimensions.
- Controlled-variable smoke confirms that the new ladder is the right causal
  design, but the existing controlled runner still refers to historical 32D
  mainline rows in places.  The formal run must update those rows before use.

## Literature Positioning

The closest adjacent literature falls into five groups.  The user's related
work notes emphasize that reviewers will likely ask why this is not simply a
pure end-to-end RL problem, so the final comparison must explicitly include a
pure-RL/no-PID row.

1. Cable-suspended payload RL.  Recent quadrotor-payload works such as FLARE,
   RoVerFly, and CrazyMARL show that RL can work for suspended-payload flight,
   trajectory tracking, or cooperative transport.  Their limitation for our
   paper is that they do not solve multi-hole precision insertion with contact,
   small clearances, flexible rope modeling, and wind.
2. Residual robot learning.  Johannink et al. and Silver et al. show that a
   learned residual can improve an imperfect conventional controller on
   contact-rich manipulation, partial observability, sensor noise, and model
   mismatch.  Our work should be positioned as bringing this idea to
   cable-suspended precision insertion, where the base controller is useful for
   safe descent but cannot model flexible cable/contact/wind effects.
   References:
   - https://arxiv.org/abs/1812.03201
   - https://arxiv.org/abs/1812.06298
3. Flexible-cable and suspended-payload control.  Geometric control and
   POD/NMPC work model the cable explicitly and can provide strong guarantees
   or high-quality tracking when the model/state assumptions hold.  Our
   advantage is not replacing all model-based control; it is handling a
   contact-rich insertion task with wind, multi-segment rope, visual/latent
   sensing, and model mismatch using a deployable residual policy.
   References:
   - https://arxiv.org/abs/1407.8164
   - https://arxiv.org/abs/2403.17565
4. Learning for suspended payloads.  Model-based meta-RL work shows that
   suspended payload dynamics are hard enough to need adaptation.  Our task is
   complementary: instead of only transport/tracking, the payload must be
   inserted into small holes while the cable remains flexible and disturbed.
   Reference:
   - https://arxiv.org/abs/2004.11345
5. Sim2real and deployable robot learning.  Domain randomization, dexterous
   manipulation, and world-model robot learning motivate training in simulation
   with robust perception/dynamics handling.  Our specific contribution is a
   task-tailored sim2real stack: visual rope/payload sensing, cable latent
   prediction, and residual control under a classical safety prior.
   References:
   - https://arxiv.org/abs/1703.06907
   - https://arxiv.org/abs/1808.00177
   - https://arxiv.org/abs/2206.14176

## Core Claims To Highlight

The experiments should make these claims easy to see:

1. Structured residual learning is necessary: PID/PD/MPC alone either fail
   strict insertion or degrade sharply under rope/wind/precision changes; pure
   RL/no-PID also fails to form a reliable insertion policy; PID+Residual RL is
   the robust middle ground.
2. Cable dynamics matter: removing the cable latent should increase
   stuck-on-rebar and reduce strict success, especially under 10-segment rope,
   wind, and tighter holes.
3. Compact learned cable representation is better than simply adding more raw
   capacity: the 8D trainable encoder is the current best robustness point and
   should be the mainline; 16D/legacy 32D are supporting capacity evidence.
4. Sim2real modules are deployable approximations, not just oracle state:
   vision and CableLatPred should be evaluated as a deployment-gap experiment
   against the ideal true-observation upper bound.
5. Failure modes are part of the contribution: success rates alone are not
   enough.  Stuck-on-rebar, lucky insert, timeout, swing energy, and swing angle
   explain why the residual policy works.

## Main Paper Comparison

Use one cumulative controlled-variable ladder.  Adjacent levels should add one
intended difficulty factor.

| Level | Added factor | Hole/rebar ratio | Rope | Init disturbance | Wind |
|---|---|---:|---:|---:|---:|
| C0 | base insertion | 8.0 | 4 seg | none | 0 m/s |
| C1 | randomized initial state | 8.0 | 4 seg | L5-scale init | 0 m/s |
| C2 | flexible rope model | 8.0 | 10 seg | L5-scale init | 0 m/s |
| C3 | moderate wind | 8.0 | 10 seg | L5-scale init | 2 m/s |
| C4 | strong wind | 8.0 | 10 seg | L5-scale init | 6 m/s |
| C5 | precision insertion | 4.0 | 10 seg | L5-scale init | 6 m/s |

Run the 8 m/s branch as a stress-test supplement, not the main ladder, unless
6 m/s saturates after switching to the 8D mainline.

Main methods:

| Method | Role |
|---|---|
| PID | simple classical baseline, true observation |
| Damped-PD | anti-swing classical baseline, true observation |
| MPC | simplified model-based baseline, true observation |
| Pure RL / no-PID | end-to-end RL baseline without the stabilizing PID prior |
| Ours-Full8D | current ideal-observation PID+residual RL mainline |

Keep Ours-Sim2Real as a separate deployment table/figure because it uses
vision and CableLatPred and should not be mixed with true-observation causal
ablations.  Evaluate it on geometry-compatible levels first.  If it is tested
on the ratio-4 precision level, label that row as OOD precision generalization
unless a matching finetune is performed.

Primary metric:

- main success = strict insertion success + lucky insertion;
- strict success = clean insertion only.

Always report both.  PID/Damped-PD can look good under broad success while
having almost zero strict success.

Secondary metrics:

- stuck-on-rebar, timeout, early-stop/instability distribution;
- average and p95 swing kinetic energy;
- average and p95 swing angle;
- EE acceleration;
- average steps;
- final distance-to-fixture / worst rebar error where available.

## Ablation Design

Keep ablations grouped by mechanism rather than toggling every minor flag.

### A. Controller Prior

Question: is the PID residual structure necessary?

Variants:

- PID only;
- Ours-Full8D;
- Pure RL/no-PID full-action policy.

This is now a main-text ablation, not only a supplement.  It directly answers
the likely reviewer question raised by recent pure-RL suspended-payload work:
why not train an end-to-end policy?  Current no-PID evidence is strongly
negative, but it still needs a clean paper-row evaluation on the controlled
ladder.

Do not revive MPC-base residual as a main ablation unless a smoke test shows
nonzero L8 learning; current evidence says this branch is not clean enough.

### B. Cable-State Representation

Question: does explicit cable state help the residual policy?

Variants:

- Ours-Full8D;
- Ours-NoCable L8-best;
- optional capacity supplement: 16D trainable vs 8D trainable.

Best evidence target: high-wind and precision levels, where cable state should
reduce stuck-on-rebar and improve strict success.

This should be the second central ablation group for the paper because it
directly connects to the flexible-cable literature.  Run it on C2, C4, C5, and
the 6-8/8-10 m/s wind bins if compute allows.

### C. Temporal Memory

Question: does recurrent memory matter beyond instantaneous observations?

Variants:

- Ours-Full8D LSTM;
- A5-Full8D-MLP, same budget and reward.

Important caveat: the completed A5 run used the legacy 32D cable encoder, so it
is not a clean comparison to the current 8D mainline.  For the main paper, redo
or finetune a true Full8D-MLP variant; otherwise label the existing A5 result
as legacy/supplementary.  Evaluate on C1, C4, and C5 first; run full ladder only
if smoke shows a meaningful separation.

### D. Sim2Real Sensing / Prediction

Question: what is lost when ideal observations are replaced by deployable
sensing and prediction?

Variants:

- true-observation Ours-Full8D upper bound;
- full sim2real vision + CableLatPred deployment model;
- ObsPred P2 as a low-rate sensing stress test;
- ObsPred P3 only as supplement if P2 is not already too degraded.

Do not present P2/P3 as the core method unless they improve over true
observation under a well-defined sensor-delay setting.  Their clean role is
robustness/stress analysis.

For the main paper, prefer one clean deployment comparison:

- TrueObs-Full8D upper bound;
- Vision + CableLatPred full sim2real model;
- P2 only if the story is specifically about low-rate sensing.

Move P3 to supplementary material unless P2 is near-saturated.

### E. Future World-Model Extension

The proposed WMNext8D feature can be tested after the main paper data are
stable.  Treat it as an extension, not as a prerequisite for the current paper.
Its first fair comparison is Ours-Full8D versus Ours-Full8D+WMNext8D on C4/C5
and high-wind bins.

## Formal Test Protocol

Before any formal run:

1. Update the controlled runner to use
   `saves/descent_ppo_paper_trueobs_residual_pretrain_trainable8_seed270627_20260704/ckpt_best_l8.pt`
   for the mainline.
2. Pass explicit `--cable-encoder-output-dim 8 --trainable-cable-encoder` for
   Full8D.
3. Use the L8-best NoCable checkpoint for no-cable ablations.
4. Run config audit and confirm: target/start XY, payload mass, obstacle slots,
   actual obstacles, rope segments, hole size, wind, PID residual mode, obs_dim,
   and sim2real flags.
5. Run 32 episodes per level/method smoke.
6. Formal main comparison: 256 episodes per level/method minimum; use 512 for
   the final headline if compute allows or if confidence intervals are close.
7. Formal ablations: 256 episodes per selected profile/variant is enough unless
   the effect size is small.
8. Export all result tables in LaTeX using the same style as
   `paper_final_experiment_tables_20260705.tex`.

Figure rules:

- Show broad/main success as the bar height because this matches the current
  insertion-success definition.
- Overlay strict success as a marker or outlined bar.  Do not hide it in a
  separate table; PID/PD can have high broad success while nearly all outcomes
  are lucky/reject cases.
- Put failure composition next to the success plot.  The most important
  mechanism-level failure is stuck-on-rebar.
- Include process metrics only where they support the mechanism: swing energy,
  swing angle, and EE acceleration are enough.

## Recommended Execution Order

1. Fix and audit the controlled runner for Full8D mainline and L8-best ckpts.
2. Run C0-C5 smoke for PID, Damped-PD, MPC, Pure RL/no-PID, and Ours-Full8D.
3. If 6 m/s saturates, add the 8 m/s branch; otherwise keep 8 m/s as supplement.
4. Run the main 256/512 comparison.
5. Run cable ablation: Full8D vs NoCable on C2/C4/C5 plus wind bins.
6. Run memory ablation: Full8D-LSTM vs a matched Full8D-MLP on C1/C4/C5.
7. Run sim2real deployment/stress: full sim2real and P2 on the same selected
   profiles where geometry is compatible.
8. Generate paper figures: success with Wilson CI, strict-success overlay,
   failure composition, and process-metric panels.
