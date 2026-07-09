# Parallel Testing Note

Date: 2026-07-06

## Question

Can we speed up evaluation by testing multiple environments in parallel when
they use the same difficulty level and therefore the same MuJoCo XML?

## Short Answer

For the current official runs, keep the evaluator single-environment and
serial.  Same difficulty reduces the risk because the generated XML content is
identical, but it does not remove the race: every `CableRobotEnvWithObstacles`
construction still calls the rope/XML generator and writes shared files under
`assets/`.

## Evidence In Code

- `mujoco_env_new.CableRobotEnvWithObstacles.__init__` imports
  `assets.generate_four_cables_with_plate` and the logs show each test case
  rewrites:
  - `assets/demo_fourCable_withSteel_withSensor_cylinder.xml`
  - `assets/iiwa14_four_cables_with_plate.xml`
- `vec_env.py` already contains a worker `init_lock` because concurrent
  workers previously caused XML parse errors when one process read a partially
  written shared XML.
- After initialization, reset-time scene generation is mostly worker-local, but
  initialization is enough to make naive multi-process testing unsafe.

## Safe Parallel Options

1. **Conservative official path, current choice**
   - Run single-environment serial evaluation.
   - Slow but avoids shared XML races and makes the result easiest to audit.

2. **Same-level parallel with serialized initialization**
   - Add a file lock around environment construction for each process.
   - After all processes finish initialization, rollouts can run in parallel.
   - Still needs a smoke test because summary writing, GPU model loading, and
     process cleanup introduce new failure modes.

3. **Proper vectorized evaluator**
   - Reuse the existing `SubprocVecEnv` style with its `init_lock`.
   - Implement deterministic schedule partitioning across workers.
   - Aggregate per-episode results into exactly the same summary schema.
   - This is the cleanest long-term speed-up path, but it is code work and
     should be validated against the current serial evaluator on a small
     n=32/n=64 smoke.

## Recommendation

Do not change the official 512-episode ablation currently running.  If
evaluation speed becomes a bottleneck after the paper tables stabilize, build a
separate vectorized evaluator and verify it against the serial evaluator on the
same C4 schedule before using it for paper numbers.
