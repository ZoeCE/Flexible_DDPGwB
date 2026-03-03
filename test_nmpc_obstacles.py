# -*- coding: utf-8 -*-
"""
带障碍物避碰的 NMPC 测试脚本。

使用 mujoco_env_obstacles 与 nmpc_controller_obstacles，从当前位置移动到目标位置，
路径上存在障碍物，NMPC 约束负载不与障碍物碰撞。
独立于原有 test.py，不覆盖原代码。
"""

import time
import argparse
import numpy as np

from mujoco_env_obstacles import CableRobotEnvWithObstacles
from nmpc_controller_obstacles import NMPCControllerObstacles


def run_test_nmpc_obstacles(n_episodes=10, render=False, n_obstacles=3,
                            obstacle_seed=42):
    env = CableRobotEnvWithObstacles(
        render=render,
        latency_steps=1,
        force_noise_level=0.08,
        control_freq_hz=10,
        init_velocity_scale=0.08,
        init_position_range=0.06,
        n_obstacles=n_obstacles,
        obstacle_radius_range=(0.04, 0.07),
        path_width=0.12,
        obstacle_seed=obstacle_seed,
    )
    nmpc = NMPCControllerObstacles(
        dt=env.control_dt,
        N=20,
        L=0.6,
        u_max=0.5,
        n_obstacles_max=max(5, n_obstacles),
        obstacle_margin=0.05,
    )

    success_count = 0
    total_steps_success = 0
    collision_count = 0

    print("*******************************************")
    print("NMPC with Obstacle Avoidance")
    print(f"Episodes: {n_episodes}, Render: {render}, Obstacles: {n_obstacles}")
    print("*******************************************")

    start_time = time.time()
    for ep in range(n_episodes):
        obs = env.reset()
        obstacles = env.get_obstacles()
        target_pos = env.target_pos
        step = 0
        episode_collision = False

        while True:
            nmpc_state = obs[:8]
            action = nmpc.get_action(nmpc_state, target_pos, obstacles)
            next_obs, reward, done, success = env.step(action)
            obs = next_obs
            step += 1

            # 简单碰撞检测：负载 (qx,qy) 与任意障碍物距离 < 半径 视为碰撞
            qx, qy = obs[4], obs[5]
            for (ox, oy, r) in obstacles:
                d = np.hypot(qx - ox, qy - oy)
                if d < r:
                    episode_collision = True
                    break

            if render:
                time.sleep(0.02)

            if done or step >= 200:
                if episode_collision:
                    collision_count += 1
                if success:
                    success_count += 1
                    total_steps_success += step
                if render:
                    print(f"Ep {ep+1}: Steps={step}, Success={success}, Collision={episode_collision}")
                break

        if not render and (ep + 1) % 5 == 0:
            print(f"Progress: {ep+1}/{n_episodes} | SR: {success_count/(ep+1)*100:.1f}% | Collisions: {collision_count}")

    elapsed = time.time() - start_time
    avg_steps = total_steps_success / success_count if success_count > 0 else 0
    print("\n" + "="*40)
    print("Result (NMPC + Obstacles):")
    print(f"  Episodes:     {n_episodes}")
    print(f"  Success:      {success_count} ({success_count/n_episodes*100:.2f}%)")
    print(f"  Collisions:   {collision_count}")
    print(f"  Avg steps:    {avg_steps:.1f}")
    print(f"  Time:        {elapsed:.2f}s")
    print("="*40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test NMPC with path obstacles")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--obstacles", type=int, default=3, help="Number of obstacles per episode")
    parser.add_argument("--seed", type=int, default=42, help="Obstacle RNG seed")
    args = parser.parse_args()
    run_test_nmpc_obstacles(
        n_episodes=args.episodes,
        render=args.render,
        n_obstacles=args.obstacles,
        obstacle_seed=args.seed,
    )
