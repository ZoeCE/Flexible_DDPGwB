import unittest

import numpy as np

import mujoco_env


class DescentLogicTests(unittest.TestCase):
    def test_descent_requires_consecutive_stable_steps_before_starting(self):
        env = mujoco_env.CableRobotEnv.__new__(mujoco_env.CableRobotEnv)
        env.current_mocap_pos = np.array([0.2, 0.3, 1.0], dtype=float)
        env.current_mocap_vel = np.zeros(3, dtype=float)
        env.descent_active = False
        env.descent_trigger_dist = 0.02
        env.descent_trigger_vel = 0.08
        env.descent_ready_counter = 0
        env.descent_ready_steps_required = 3
        env.success_qz_threshold = 0.01
        env.descent_slow_qz = 0.08
        env.descent_fast_vz = -0.2
        env.descent_slow_vz = -0.05
        env.descent_hold_z = 0.42
        env.descent_pause_dist = 0.08
        env.descent_pause_vel = 0.20

        env._compute_auto_z_accel(dist_xy=0.01, vel_xy=0.01, q_z=0.20)
        env._compute_auto_z_accel(dist_xy=0.01, vel_xy=0.01, q_z=0.20)
        self.assertFalse(env.descent_active)

        env._compute_auto_z_accel(dist_xy=0.01, vel_xy=0.01, q_z=0.20)
        self.assertTrue(env.descent_active)

    def test_descent_stays_latched_after_touchdown_sequence_starts(self):
        env = mujoco_env.CableRobotEnv.__new__(mujoco_env.CableRobotEnv)
        env.current_mocap_pos = np.array([0.2, 0.3, 1.0], dtype=float)
        env.current_mocap_vel = np.zeros(3, dtype=float)
        env.descent_active = False
        env.descent_trigger_dist = 0.05
        env.descent_trigger_vel = 0.15
        env.descent_ready_counter = 0
        env.descent_ready_steps_required = 1
        env.success_qz_threshold = 0.01
        env.descent_slow_qz = 0.08
        env.descent_fast_vz = -0.2
        env.descent_slow_vz = -0.05
        env.descent_hold_z = 0.42
        env.descent_pause_dist = 0.08
        env.descent_pause_vel = 0.20

        az_start = env._compute_auto_z_accel(dist_xy=0.01, vel_xy=0.01, q_z=0.20)
        self.assertTrue(env.descent_active)
        self.assertLess(az_start, 0.0)

        az_continue = env._compute_auto_z_accel(dist_xy=0.08, vel_xy=0.20, q_z=0.06)
        self.assertTrue(env.descent_active)
        self.assertLess(az_continue, 0.0)


if __name__ == "__main__":
    unittest.main()
