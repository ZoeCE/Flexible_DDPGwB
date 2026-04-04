#!/usr/bin/env python3
"""
测试新 pipeline：IK 自动求解 + 2段绳索模型。
用法:
    python test_ik_pipeline.py              # 无渲染，跑 5 个 episode
    python test_ik_pipeline.py --render     # 开启 GUI 渲染
    python test_ik_pipeline.py --episodes 20
"""

import argparse
import numpy as np
import mujoco

from mujoco_env_new import CableRobotEnvWithObstacles
from config import DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Test IK + new env pipeline")
    parser.add_argument("--render", action="store_true", help="Enable MuJoCo GUI")
    parser.add_argument("--episodes", type=int, default=5, help="Number of reset episodes")
    parser.add_argument("--no-ik", action="store_true", help="Disable IK, use default qpos")
    args = parser.parse_args()

    # 构造配置覆盖
    override = {
        "sim": {"render": args.render},
        "reset": {"ik_enabled": not args.no_ik},
    }

    env = CableRobotEnvWithObstacles(config=override)

    print(f"Model: nq={env.model.nq}, nv={env.model.nv}")
    print(f"IK enabled: {env.config['reset']['ik_enabled']}")
    print(f"IK height above prefab: {env.config['reset']['ik_height_above_prefab']} m")
    print("=" * 60)

    for ep in range(args.episodes):
        obs = env.reset()

        # 读取 warmup 后的末端状态
        ee_pos = env.data.site_xpos[env.ee_site_id].copy()
        ee_mat = env.data.site_xmat[env.ee_site_id].reshape(3, 3)
        ee_z = ee_mat[:, 2]

        prefab_pos = env.data.body("prefab").xpos.copy()

        xy_err = np.linalg.norm(ee_pos[:2] - prefab_pos[:2])
        height_diff = ee_pos[2] - prefab_pos[2]
        z_dot = np.dot(ee_z, [0, 0, -1])  # 1.0 = 完美朝下

        print(
            f"Ep {ep+1:2d} | "
            f"EE=({ee_pos[0]:+.3f}, {ee_pos[1]:+.3f}, {ee_pos[2]:.3f}) | "
            f"Prefab=({prefab_pos[0]:+.3f}, {prefab_pos[1]:+.3f}, {prefab_pos[2]:.3f}) | "
            f"XY_err={xy_err:.4f}m | "
            f"dZ={height_diff:.3f}m | "
            f"Z_down={z_dot:.4f}"
        )

        # 简单跑几步验证 step 不报错
        for _ in range(10):
            action = np.zeros(env.action_dim)
            result = env.step(action)
            obs, reward, done = result[0], result[1], result[2]
            if done:
                break

    print("=" * 60)
    print("Pipeline test completed.")


if __name__ == "__main__":
    main()
