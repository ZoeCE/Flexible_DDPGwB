# ==============================================================================
# config.py — 三阶段 RL 控制架构全局配置 v8 (重构版)
#
# v8 重构核心:
#   1. 删除所有已废弃模块的参数 (ORCA expert, dual-RL, shadow obstacle, ...)
#   2. 三阶段聚焦于「噪声/风力鲁棒性」课程学习
#   3. 课程晋级条件: success_rate ≥ threshold AND episodes_at_level ≥ min_eps
#   4. 风力上限统一: 训练/测试最大 2.0 N
#   5. wandb 日志精简: PPO 训练指标 + 课程参数 + 训练表现 三类
#
# 各阶段统一课程结构 (噪声等级 0 → max):
#   level 0: 无噪声/无风   → 建立基础能力
#   level 1: 弱噪声/弱风   → 提升鲁棒性
#   level 2: 中等噪声/中风 → 增强适应
#   level 3: 强噪声/强风   → 最终鲁棒水平
# ==============================================================================

import numpy as np

DEFAULT_CONFIG = {

    # ==========================================================================
    # 1. 仿真与控制
    # ==========================================================================
    "sim": {
        "physics_dt":       0.002,
        "control_freq_hz":  10,
        "max_steps":        400,
        "render":           False,
        "substep_skip":     1,
    },

    # ==========================================================================
    # 2. 空间与动作
    # ==========================================================================
    "space": {
        "action_dim":        7,
        "action_space_high": [2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054],
        "action_space_low":  [-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054],
        "ee_action_high":    [0.5, 0.5, 2.0, 2.0],
        "dq_max": [0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12],
        "dq_scale_insertion": 0.5,
    },

    # ==========================================================================
    # 3. 任务与初始状态
    # ==========================================================================
    "task": {
        "start_pos_mocap":    [0.3, 0.15, 0.5],
        "start_quat_mocap":   [1.0, 0.0, 0.0, 0.0],
        "default_start_xy":   [0.3, 0.15],
        "default_target_xy":  [-0.3, 0.2],
        "init_position_range": 0.02,
        "init_velocity_scale": 0.08,
    },

    # ==========================================================================
    # 4. 场景生成 (cruise 全程使用 3 障碍物, 不再做障碍课程)
    # ==========================================================================
    "scene": {
        "n_obstacles":        3,
        "radius_range":       (0.020, 0.045),
        "obstacle_z_center":  0.15,
        "obstacle_halfheight": 0.15,
        "endpoint_z_offset":  0.025,
        "seed":               None,
    },

    # ==========================================================================
    # 5. A* 寻路 + 3D 轨迹规划
    # ==========================================================================
    "planning": {
        "workspace_radius":  0.65,
        "payload_radius":    0.075,
        "planning_margin":   0.04,
        "planning_grid_res": 0.025,
        "path_width":        0.80,
        "corridor_side":          +1,
        "corridor_forbid_margin": 0.05,
        "bounds_margin":     0.5,
        "max_expansions":    100000,
        "payload_z_cruise":  0.25,
        "target_z_descent":  0.12,
        "num_descent_steps": 15,
        # [v12.2] num_lift_steps 3 → 6
        # 原 3 个航点 (z=0.157, 0.203, 0.25) 间距太大, NMPC 跟踪粗糙;
        # 6 个 (z=0.133, 0.156, 0.180, 0.203, 0.227, 0.250) 让 lift 更平稳.
        "num_lift_steps":    6,
    },

    # ==========================================================================
    # 6. Step 安全判定
    # ==========================================================================
    "step_logic": {
        "look_ahead_dist":    0.25,
        "out_of_bounds_dist": 2.0,
        "crash_z_threshold":  0.03,
        "crash_vz_threshold": -0.5,
        "crash_grace_steps":  30,
        "instability_check":     True,
        "instability_grace_steps": 15,
        "swing_xy_max":          0.35,
        "payload_vel_max":       2.0,
        "payload_tilt_max":      1.0,
        "payload_yaw_max":       1.2,
        "instability_penalty":   -5.0,
        "reward_clip_min":       -5.0,
        "reward_clip_max":        5.0,
    },

    # ==========================================================================
    # 7. 系统延迟 (latency_steps 仅 env 使用)
    # ==========================================================================
    "noise": {
        "latency_steps":     1,
        "force_noise_level": 0.0,   # DEPRECATED, 实际噪声由 curriculum.* 控制
    },

    # ==========================================================================
    # 8. 重置参数
    # ==========================================================================
    "reset": {
        "init_qpos_arm": [
            -0.71972016,  0.29057466, -1.10685585,
             1.53851657,  2.87907443,  1.73055121, -1.85194252
        ],
        "init_qpos_prefab": [0.3, 0.15, 0.1, 1.0, 0.0, 0.0, 0.0],
        "warmup_steps":      50,
        "mocap_init_z":      0.6,
        "ik_enabled":        True,
        "ik_height_above_prefab": 0.6,
        "ik_target_quat":    [0.0, 1.0, 0.0, 0.0],
        "ik_max_iter":       5000,
        "ik_tol_pos":        1e-4,
        "ik_tol_rot":        1e-3,
    },

    # ==========================================================================
    # 9. Prefab 几何
    # ==========================================================================
    "prefab": {
        "shape":              "socket",
        "box_half_size":      [0.05, 0.05, 0.1],
        "cylinder_radius":    0.04,
        "cylinder_half_height": 0.1,
        "mesh_prefix":        "hollow_cylinder_convex",
        "mesh_count":         64,
        "socket_half_size":   [0.05, 0.05, 0.1],
        "socket_hole_size":   [0.014, 0.014],
        "socket_hole_depth":  0.06,
        "socket_hole_positions": [
            [ 0.035,  0.035],
            [ 0.035, -0.035],
            [-0.035,  0.035],
            [-0.035, -0.035],
        ],
        "mass":               1.0,
        "lift_site_offset":   0.1,
        "lift_site_spread":   0.05,
    },

    # ==========================================================================
    # 10. Target — rebar 模式
    # ==========================================================================
    "target": {
        "mode":               "rebar",
        "shape":              "box",
        "box_half_size":      [0.05, 0.05, 0.1],
        "cylinder_radius":    0.02,
        "cylinder_half_height": 0.1,
        "rgba":               [0.8, 0.0, 0.0, 0.4],
        "rebar_positions": [
            [ 0.035,  0.035],
            [ 0.035, -0.035],
            [-0.035,  0.035],
            [-0.035, -0.035],
        ],
        "rebar_radius":       0.0025,
        "rebar_half_height":  0.01,
        "rebar_rgba":         [0.8, 0.0, 0.0, 1.0],
    },

    # ==========================================================================
    # 11. 绳索
    # ==========================================================================
    "rope": {
        "num_segments":    10,
        "segment_length":  0.04,
        "damping":         0.05,
        "capsule_radius":  0.004,
        "segment_mass":    0.01,
        "plate_half_size": [0.05, 0.05, 0.01],
        "plate_mass":      0.1,
        "hook_offset":     0.05,
    },

    # ==========================================================================
    # 12. NMPC 底层控制器 (cruise base + descent expert 都用)
    # ==========================================================================
    "controller": {
        "N":                    20,
        "dt":                   0.1,
        "L":                    0.5,
        "u_max_xy":             1.2,
        "u_max_z":              2.5,
        "u_max_yaw":            2.0,
        "arrival_threshold_xy": 0.08,
        "arrival_threshold_z":  0.08,
    },

    # ==========================================================================
    # 13. 插入任务成功判定
    # ==========================================================================
    "insertion": {
        "entry_z":               0.16,
        "target_payload_z":      0.10,
        "success_z_tolerance":   0.020,
        "xy_tolerance":          0.0045,
        "tilt_tolerance":        0.05,
        "yaw_tolerance":         0.08,
        "hold_steps":            3,
        "vel_xy_tolerance":      0.05,
        "vel_z_tolerance":       0.05,
        "require_floor_contact": False,
        "partial_dist_scale":    0.05,

        # 退火 (descent 课程最高 level 后用)
        "xy_tolerance_train_end":      0.005,
        "tilt_tolerance_train_start":  0.12,
        "tilt_tolerance_train_end":    0.05,
        "yaw_tolerance_train_start":   0.15,
        "yaw_tolerance_train_end":     0.08,
        "xy_tolerance_anneal_steps":   500_000,
    },

    # ==========================================================================
    # 14. 风力 (env 内部; 训练用 set_wind_force 每 episode 覆盖)
    # ==========================================================================
    "wind": {
        "enabled":            False,
        "F_max":              2.0,        # [v8] 全局上限 2.0 N
        "theta_rate_std":     0.15,
        "force_rate_std":     0.1,
        "seed":               123,
    },

    # ==========================================================================
    # 15. 旧版 env 内置奖励 (env 内部 _compute_reward 用)
    # ==========================================================================
    "reward": {
        "success_bonus":          50.0,
        "soft_success_bonus":     25.0,
        "timeout_penalty":        -2.0,
        "collision_penalty":      -10.0,
        "out_of_bounds_penalty":  -5.0,
        "crash_penalty":          -8.0,
        "instability_penalty":    -5.0,
        "use_mujoco_contact":     True,
        "joint_limit_margin":     0.1,
    },

    # ==========================================================================
    # 16. Pipeline 阶段切换条件 (test_phase.py 用)
    # ==========================================================================
    "phase_transition": {
        "lift_to_cruise_z_threshold":    0.23,
        "lift_to_cruise_tilt_max":       0.15,
        "lift_to_cruise_swing_vel_max":  0.08,
        "lift_to_cruise_payload_vel_max": 0.15,
        "cruise_to_descent_xy_dist":     0.10,
        "cruise_to_descent_tilt_max":    0.15,
        "cruise_to_descent_swing_vel_max": 0.20,
        "cruise_to_descent_payload_vel_max": 0.25,
        "cruise_to_descent_fallback_xy_dist":   0.15,
        "cruise_to_descent_fallback_swing_max": 0.30,
        "cruise_to_descent_fallback_step_frac": 0.85,
    },

    # ==========================================================================
    # 17. EE 加速度控制器
    # ==========================================================================
    # [v12 关键修复] EE 控制器速度上限 + 锚定 与 JointSpaceExpert 同步.
    # 之前 train 用 EEAccController (vel_max_xy=0.25, alpha=0.08), 但 test/expert
    # 用 JointSpaceExpert (vel_max_xy=0.15, alpha=0.10). 这导致 NMPC 在"想象的车"
    # (≤0.15)上规划, 实际车(≤0.25)更快 → 车跑过头, NMPC 反复补偿 → 振荡发散.
    # 现在统一为 JointSpaceExpert 的值: train 路径与 test/expert 路径行为一致.
    "ee_control": {
        "acc_max_xy":         0.8,
        "acc_max_z":          1.5,
        "vel_max_xy":         0.15,   # v11 0.25 → v12 0.15 (与 JointSpaceExpert 一致)
        "vel_max_z":          0.20,   # v11 0.15 → v12 0.20 (与 JointSpaceExpert 一致)
        "vel_max_z_descent":  0.03,
        "anchor_alpha":       0.10,   # 新增, 替代 EEAccController 硬编码 0.08
        "integrator_dt":      0.1,
    },

    # ==========================================================================
    # 17b. Cruise Z/Yaw PID
    # ==========================================================================
    "cruise_z_pid": {
        "kp_z":               3.0,
        "ki_z":               0.3,
        "kd_z":               0.5,
        "z_integral_max":     0.03,
        "z_correction_max":   0.10,
        "kp_yaw":             0.3,
        "kd_yaw":             0.1,
        "yaw_correction_max": 0.1,
        "floor_z_threshold":  0.04,
        "floor_vz_threshold": -0.15,
    },

    # ==========================================================================
    # 18. Phase 1: Lift RL (v12: NMPC base + RL residual 3D acc)
    # ==========================================================================
    # [v12] 架构改为 NMPC + residual, 与 cruise/descent 统一
    # - NMPC 单独跑能完成纯垂直 lift (path 起点 -> z_cruise, xy 不变)
    # - RL 残差: 在 NMPC 基础上做小幅微调 (主要抗风/抗噪)
    # - 用户要求: RL 残差权重小, 只是提高稳定性
    "lift_rl": {
        "obs_dim":            263,  # [v12.3] +240 for cable seg states (4x10x6: rel_pos + lin_vel)
        "action_dim":         3,
        "init_xy_range":      0.01,
        "init_z_range":       0.01,
        "init_vel_range":     0.0,
        "max_steps":          200,
        "target_z_cruise":    0.25,

        # [v12] 架构: NMPC + RL 残差
        "use_nmpc_base":             True,
        "residual_acc_max_xy":       0.08,    # xy 残差 (sim2real 抗风用, 与 cruise 一致)
        "residual_acc_max_z":        0.10,    # z 残差 (微调 lift 速度, 应远 < acc_max_z=1.5)
        # 残差总和上限 (clip after base + residual)
        "total_acc_max_xy":          0.60,
        "total_acc_max_z":           1.50,

        # log_std phase-specific (新 lift residual 用 zero-init + 小 std)
        "lift_log_std_init":         -2.0,   # std exp(-2.0) ≈ 0.135
        "lift_log_std_floor_init":   -3.0,
        "lift_log_std_floor_final":  -3.0,

        "reward": {
            # ── v12.4 Lift reward (与 cruise 共用防摆主体) ───────────────────
            # 核心: NMPC 已能 lift, RL 残差只需抗扰. 防摆 reward 与 cruise 完全一致.
            # 终止 reward 量级 ≤ 5 (v11.4 经验, 防 PPO Q 发散).
            #
            # 1. z_approach: 引导 z 接近 cruise 高度 (差分, lift 特有, 主导)
            "z_approach_coef":            2.0,
            # 2. swing_energy 强惩罚 (与 cruise 完全一致)
            "swing_energy_thresh":        0.010,
            "swing_energy_penalty_coef":  4.0,
            "swing_energy_penalty_max":   0.10,
            # [v12.4 新增] swing_improve 差分鼓励主动减摆 (与 cruise 一致)
            "swing_improve_coef":         20.0,
            "swing_improve_max":           0.04,
            "swing_worsen_max":            0.04,
            # 3. [v12.3] cable_ke_penalty (与 cruise 一致)
            "cable_ke_thresh":            0.05,
            "cable_ke_penalty_coef":      1.0,
            "cable_ke_penalty_max":       0.08,
            # 4. tilt 惩罚 (lift 阶段防止 payload 翻倒)
            "tilt_coef":                  0.5,
            "tilt_max_for_penalty":       0.3,
            # 5. action 残差限制 (与 cruise 一致)
            "action_magnitude_coef":      0.02,
            "action_smoothness_coef":     0.04,
            # 6. xy drift (lift 特有: NMPC 已保证, 小奖励)
            "xy_drift_coef":              1.0,
            # 7. 成功判定参数
            "success_z_tol":              0.025,
            "success_vz_max":             0.08,
            "success_swing_energy_thresh": 0.020,
            "xy_max_dist_success":        0.12,
            "hold_steps":                 3,
            # 8. 终止 reward (量级与 cruise 一致)
            "success_bonus":              5.0,
            "step_penalty":               0.0,    # 用户要求 "不加 penalty"
            "instability_penalty":       -1.0,
            "crash_penalty":             -1.5,
        },
    },

    # ==========================================================================
    # 19. Phase 1: Cruise RL (统一阶段: NMPC 抬升 + 平移)  [v13.0]
    #
    # [v13.0 真正合并] 之前是 lift + cruise 两段, 现在合并为 cruise 一段:
    #   - 路径: (start_xy, 0.11) → (start_xy, z_cruise) → (target_xy, z_cruise)
    #   - NMPC 自动处理 lift→cruise 边界 (controller.py tracker has lift WP 检测)
    #   - RL 残差: 3D acc (xy + z), 因为 lift 阶段需要 z 残差控制
    # [v9] BC 预训练完全移除. cruise 残差 actor 在创建时自动应用输出层零初始化,
    # 训练初期 RL 残差 ≈ 0, 让 NMPC 单独工作; PPO 梯度逐步训练出小幅微调.
    # 来源: Jeon et al. 2025 (Residual MPC), Ankile et al. 2024 (ResiP).
    # ==========================================================================
    "cruise_rl": {
        "obs_dim":            278,  # [v12.3] +240 for cable seg states
        "action_dim":         3,    # [v13.0] 2 → 3 (xy + z), 因为合并了 lift 的 z 残差
        "init_xy_range":      0.01,
        "init_z_range":       0.01,
        "init_vel_range":     0.0,
        "max_steps":          700,  # [v13.0] 500 → 700 (lift 200 + cruise 500 合并)
        "min_steps_for_success": 5,
        "estimated_full_dist_m": 0.41,
        "z_lock_height":      0.25,
        "target_z_cruise":    0.25,  # [v13.0] 从 lift_rl 搬来 (统一 z 目标高度)

        # 架构: NMPC base + RL 残差 [v9 缩小残差幅度]
        "use_nmpc_base":         True,
        "residual_acc_max_xy_rl": 0.08,   # [v9] 原 0.25 → 0.08 (base 量级 ~0.5 的 16%)
        "residual_acc_max_z_rl":  0.10,   # [v13.0] 新增 z 残差上限 (从 lift_rl 搬来)
        "residual_acc_max_xy":   0.60,    # [v9] 原 0.80 → 0.60
        "total_acc_max_z":       1.50,    # [v13.0] z 总和上限 (lift 阶段需要)
        # [v9] BC 已完全移除. cruise 残差 actor 在 PPOPhaseAgent.__init__ 中
        # 总是自动应用输出层零初始化 (见 phase_agent.py _zero_init_residual_actor).
        # [v10] phase-specific log_std 设置 (覆盖全局), 让零初始化真正生效
        "cruise_log_std_init":        -2.0,    # std exp(-2.0) ≈ 0.135
        "cruise_log_std_floor_init":  -3.0,    # 几乎不限制下降
        "cruise_log_std_floor_final": -3.0,

        "reward": {
            # ─────────────────────────────────────────────────────────────────
            # [v11.2] 基于文献彻底重设计 (Olesen 2026, Jeon 2025, Mysore 2021).
            # 设计原则: residual RL 应用简单 reward, 不与 base controller 冲突.
            # ─────────────────────────────────────────────────────────────────

            # 1. swing_improve (差分) — 鼓励主动减摆
            #    [v11.4] cap 0.20 → 0.04 (与其他 reward 一起 ÷5 防 Q 发散)
            "swing_improve_coef":   20.0,    # v11.2 50 → v11.4 20
            "swing_improve_max":     0.04,   # v11.2 0.20 → v11.4 0.04
            "swing_worsen_max":      0.04,   # 同上

            # 2. swing_energy 绝对惩罚 — 核心: 保护 NMPC 稳定性
            #    [v11.2] threshold 50mJ → 10mJ (NMPC normal 4mJ 的 2.5x)
            #            coef 2.0 → 8.0 (强惩罚, 让 RL 学到"不能让 swing 涨")
            #    [v11.4] cap 0.50 → 0.10 (P0: 减小 reward magnitude 防 Q 发散)
            "swing_energy_thresh":        0.010,   # 50mJ → 10mJ (用户建议)
            "swing_energy_penalty_coef":  4.0,     # v11.2 8.0 → v11.4 4.0
            "swing_energy_penalty_max":   0.10,    # v11.2 0.50 → v11.4 0.10
            # [v12.3 新增] 绳索动能惩罚
            "cable_ke_thresh":            0.05,
            "cable_ke_penalty_coef":      1.0,
            "cable_ke_penalty_max":       0.10,

            # 3. [v11.2 新增, v11.4 缩小] action_magnitude_penalty
            #    Olesen et al. 2026: residual policy 应默认 0, 仅必要时介入
            "action_magnitude_coef":      0.02,    # v11.2 0.05 → v11.4 0.02

            # 4. [v11.2 新增, v11.4 缩小] action_smoothness_penalty (CAPS)
            "action_smoothness_coef":     0.04,    # v11.2 0.10 → v11.4 0.04

            # 5. calm_bonus — 引导持续低摆动 (强信号)
            "calm_energy_thresh":   0.005,
            "calm_base_bonus":      0.02,    # v11.2 0.03 → v11.4 0.02
            "near_goal_radius":     0.10,
            "near_goal_calm_bonus": 0.03,    # v11.2 0.05 → v11.4 0.03

            # ── 已删除项 (v11.2 起) ──────────────────────────────────────────
            "pbrs_coef":          0.0,
            "pbrs_gamma":         0.99,
            "step_penalty":       0.0,
            "obs_repulse_coef":   0.0,
            "obs_repulse_d0":     0.10,
            "obs_repulse_linear": True,
            "milestone_radius":   0.15,
            "milestone_bonus":    0.0,

            # ── [v13.0] lift 段引导 (payload z < z_cruise 时启用) ────────────
            # 用于合并的 lift 阶段, payload 仍在低空时优先引导上升
            "z_approach_coef":            2.0,     # 差分系数 (从 lift_rl 搬来)
            "tilt_coef":                  0.5,     # tilt 惩罚 (lift 时防翻倒)
            "tilt_max_for_penalty":       0.3,

            # 6. 终止信号 (v11.4 全部 ÷5, P0 防 Q 发散)
            # 理论 Q ∈ [success/(1-γ), collision/(1-γ)] = [+500, -200] (γ=0.99)
            # 即使 Q 偶发偏离, critic_loss 量级保持 < 100
            "collision_penalty":   -2.0,     # v11.2 -10 → v11.4 -2
            "instability_penalty": -1.0,     # v11.2 -5 → v11.4 -1
            "success_bonus":        5.0,     # v11.2 20 → v11.4 5
            # [v13.1] 用户反馈: expert 到达终点上方但没成功率 → 放宽标准
            "success_radius":       0.12,    # v13.0 0.10 → v13.1 0.12 (xy 容差放到 12cm)
            "success_hold_steps":   2,       # v13.0 3 → v13.1 2 (减少持续要求)
            # z 容差: z_success_tol = z_lock_height * z_success_tol_frac
            # z_lock=0.25, tol_frac=0.40 → z_success_tol=0.10 (z ∈ [0.15, 0.35])
            # 之前 0.20 → 0.05 太严, expert NMPC 不一定能稳定停在 [0.20, 0.30]
            "z_success_tol_frac":   0.40,    # v13.0 0.20 → v13.1 0.40 (z 容差 5cm → 10cm)
            # [v13.1] 成功判定的 vel/swing 阈值, 比 phase_transition 严格,
            # 但比 cruise_to_descent 略宽松, 允许 payload 还在轻微衰减时判成功
            "success_swing_vel_max":   0.30,  # 之前用 cruise_to_descent 的 0.20, 现在改 0.30
            "success_payload_vel_max": 0.35,  # 之前 0.25, 现在 0.35
            "success_tilt_max":         0.20, # 之前 0.15, 现在 0.20 (~11.5°)
        },
    },

    # ==========================================================================
    # 20. Phase 3: Descent RL (PID base + RL residual delta_q)
    # ==========================================================================
    "descent_rl": {
        "obs_dim":            269,  # [v12.3] +240 for cable seg states
        "action_dim":         3,
        "init_xy_range":      0.030,
        "init_z_range":       0.005,
        "init_tilt_range":    0.010,
        "init_vel_range":     0.030,
        "max_steps":          500,

        # 架构: PID + residual RL
        "pid_residual_mode":   True,
        "residual_dq_scale":   0.30,
        "residual_acc_max_xy": 0.30,
        "residual_acc_max_z":  0.50,

        # [v10] descent 因 action_scale=[0.5,0.5,1.0] 较大, 需更紧 floor
        "descent_log_std_init":        -1.5,    # std exp(-1.5) ≈ 0.22
        "descent_log_std_floor_init":  -1.5,
        "descent_log_std_floor_final": -3.0,

        # [v11 Path 2] PPO-HER (Crowder et al. 2024, arXiv:2410.22524)
        # [v11.3] max_eps 5 → 16, 让 HER 真正补偿 sparse reward
        "ppo_her_enabled":             True,
        "ppo_her_xy_tol_relabel":      0.020,  # 0.015 → 0.020 (更宽松, 更易 relabel)
        "ppo_her_min_displacement":    0.003,  # 0.005 → 0.003
        "ppo_her_max_episodes":         16,    # 5 → 16 (≥ 1 个 rollout 大量 relabel)

        # [v12.3 关键回退] 用户反馈: "很久之前版本可以 work, 现在出错了"
        # 嫌疑 #1: action_scale 缩太小. v8: 0.5/1.0, v11.4: 0.10/0.20 → 缩 80%
        # 在 5mm 精度时, RL 残差需要"小且精准"的修正, 太小则无力修正
        # 但太大又会破坏 PID. 取中庸: 0.20/0.40 (v8 的 40%, v11.4 的 200%)
        # 来源: Ankile et al. 2024 (ResiP) — residual scale 应 5-30% of base
        # PID base xy 量级 ~0.5, 取 40% = 0.20 ✓
        "acc_max_xy":         0.20,           # v11.4 0.10 → v12.3 0.20
        "acc_max_z":          0.40,           # v11.4 0.20 → v12.3 0.40
        "vel_max_z":          0.05,

        "reward": {
            # ─────────────────────────────────────────────────────────────────
            # [v12.3] 用户核心需求 (descent): 防摆 + 抗扰 + 稳定 SR
            # 重点强化: swing_ke (防摆) + xy_align (与钢筋对准) + 抗扰 robust
            # 
            # [v12.3 用户决定] 不加 step_penalty (保持 v11.3 设计):
            #   - 用户明确表态 "不加 penalty"
            #   - 改为通过单一精度课程 + cable obs/reward 让 RL 直接学
            # ─────────────────────────────────────────────────────────────────

            # 1. XY 对准差分 (PID 已经在 align, 给小奖励)
            "xy_align_coef":      1.0,
            # 2. Z 下降差分 (gate by xy aligned)
            "z_descent_coef":     1.5,
            "z_descent_xy_gate":  0.030,
            "z_rise_penalty_coef": 0.5,
            "z_rise_max_penalty":  0.05,

            # 3. 防摆 (用户核心需求, v12 加强)
            "swing_ke_coef":      1.5,    # v11.4 0.8 → v12 1.5 (用户要求强化防摆)
            "swing_ke_max":       0.25,   # 有界, 避免单步惩罚过大
            # [v12.3 新增] 绳索动能惩罚
            "cable_ke_thresh":            0.05,
            "cable_ke_penalty_coef":      0.8,
            "cable_ke_penalty_max":       0.06,

            # 4. 稳定性 (tilt + yaw)
            "tilt_coef":          0.4,
            "yaw_coef":           0.5,

            # 5. [v12 新增] 钢筋对准 bonus (rebar align)
            # 用户要求: 与地面钢筋对准给奖励, 这是 descent 段精度的关键
            # 当 xy_dist < 5mm 时给持续 bonus (每步, 鼓励 hold 在对准状态)
            "rebar_align_radius":  0.010,   # 1cm 内开始给
            "rebar_align_bonus":   0.02,    # 每步小奖励, 一 ep cap ~10 步累积 0.2
            "rebar_close_radius":  0.005,   # 5mm 内给双倍 bonus

            # 6. 高精度 bonus (近目标高斯, 比 rebar_align 更窄)
            "precision_bonus_coef":  2.5,
            "precision_bonus_sigma": 0.005,

            # 7. [v11.3] step_penalty=0, 防 dead critic
            "step_penalty":       0.0,

            # 8. [v11.3 起] action penalties (residual RL 标准做法)
            "action_magnitude_coef":   0.01,
            "action_smoothness_coef":  0.02,

            # 9. 终止信号
            "success_bonus":      12.0,
            "instability_penalty": -0.8,
            "crash_penalty":      -1.2,
        },
    },

    # ==========================================================================
    # 21. PPO 超参数
    # ==========================================================================
    "ppo": {
        # [v12.3] obs_dim 263-278 (含 240 维绳索状态), hidden_dim 升 256→384
        # 经验法则: hidden_dim ≥ obs_dim 才能充分编码
        "hidden_dim":            384,
        "n_layers":              3,
        "lr_actor":              1e-4,
        "lr_critic":             3e-4,
        "gamma":                 0.99,
        "gae_lambda":            0.95,
        "clip_eps":              0.2,
        "value_loss_coef":       0.5,
        "entropy_coef":          0.01,
        "max_grad_norm":         0.5,
        "n_steps":               1024,
        "n_epochs":              6,
        "batch_size":            256,
        "use_lstm":              True,
        "seq_len":               8,
        "lstm_dim":              128,
        "normalize_advantages":  True,
        "use_obs_norm":          True,
        "obs_norm_clip":         10.0,
        "obs_norm_warm_start":   5000,
        "log_std_init":         -0.5,
        "log_std_min":          -3.0,
        "log_std_max":           0.0,
        "target_kl":             0.02,
        # [v10] floor 放宽: 不再硬性限制 cruise 残差 std 必须 ≥ 0.61
        "log_std_floor_init":   -1.0,    # v9 -0.5 → v10 -1.0 (std 0.61 → 0.37)
        "log_std_floor_final":  -2.5,    # v9 -2.0 → v10 -2.5
        "log_std_floor_steps":   500_000,  # v9 1M → v10 0.5M (更快放宽)
        "entropy_coef_start":         0.05,
        "entropy_coef_end":           0.005,
        "entropy_coef_anneal_steps":  600_000,
        "plasticity_reset_interval":  200_000,
    },

    # ==========================================================================
    # 22. SAC 超参数
    # ==========================================================================
    "sac": {
        # [v12.3] hidden_dim 升 256→384 for obs_dim ~270 with cable states
        "hidden_dim":           384,
        "n_layers":             3,
        # [v11 vec fix] lr 降一半, 对抗 vec 模式 batch 内相关性导致的 critic 高方差
        "lr_actor":             1.5e-4,    # v10 3e-4 → 1.5e-4
        "lr_critic":            1.5e-4,    # v10 3e-4 → 1.5e-4
        "lr_alpha":             1e-4,
        "gamma":                0.99,
        "tau":                  0.005,
        "buffer_size":          500_000,
        # [v11 vec fix] batch 增大, 减少 vec 模式 8 同步 transitions 引入的相关性影响
        "batch_size":           512,       # v10 256 → 512
        "warmup_steps":         5000,
        "warmup_mode":          "expert",
        "auto_alpha":           True,
        # [v11.4] alpha_init 从 0.2 降到 0.05, 与 zero-init residual actor 配合
        # 减少初期 entropy 项对 Q target 的负向贡献
        "alpha_init":           0.05,      # 0.2 → 0.05
        # [v11.4 关键] target_entropy_ratio 0.5 → 1.6
        # 原因: zero-init actor (log_std=-2, std=0.135) 下 tanh-squashed log_pi ≈ +1.2/sample.
        # 标准 target_entropy = -|A| = -dim (Haarnoja 2018) 假设 uniform 探索, 但我们 actor
        # std 远小于 uniform → log_pi 持续 > target → alpha 单调爆炸到 5+.
        # 修复: 将 target_entropy 设为 log_pi_actual 附近, ratio=1.6 给 target = -1.6*dim:
        #   cruise (dim=2): -3.2, 与 log_pi=+1.2 之差 = -4.4 (alpha 会下降, 不再发散)
        #   descent (dim=3): -4.8
        # 参考: Ray.io discrete SAC 讨论 (target 过大致 alpha 发散), 与本任务同理.
        "target_entropy_ratio": 1.6,        # 0.5 → 1.6 (P0 修复 alpha divergence)
        # [v11.4 双保险] log_alpha hard bounds: 即使 target_entropy 失配, alpha 也不爆.
        # min=log(1e-4)≈-9.2, max=log(1.0)=0 → α ∈ [1e-4, 1.0]
        "log_alpha_min":        -9.2,
        "log_alpha_max":         0.0,
        "critic_grad_clip":     1.0,
        "her_k":               4,
        "her_reward_scale":    0.3,
        "actor_grad_clip":      1.0,
        "reward_scale":         1.0,
        # [v11 vec] update_interval=1 (每步 update), updates_per_step=N
        # vec 模式: 每步 N transitions 进入 buffer, 必须 N updates 才能 keep up
        "update_interval":      1,
        "updates_per_step":     2,         # 单进程默认 2; vec 自动 = n_envs
        "use_obs_norm":         True,
        "obs_norm_clip":        10.0,
        "obs_norm_warm_start":  5000,
    },

    # ==========================================================================
    # 23. 训练全局参数
    # ==========================================================================
    "train": {
        "total_timesteps":       2_000_000,
        "save_interval":         50,
        "eval_interval":         100,
        "eval_episodes":         10,
        "log_smooth_win":        20,
        "gpu_id":                0,
        # [v11 Path 3] Vectorized env (CPU-parallel)
        # n_envs=1 → DummyVecEnv (与原版完全一致, 默认安全)
        # n_envs>1 → SubprocVecEnv (需要重构 train loop, 见 README)
        # 建议: n_envs ≤ 物理 CPU 核心数 - 1
        "n_envs":                1,
        "vec_env_start_method":  "forkserver",  # 'spawn'/'fork'/'forkserver'
        "algo":                  "ppo",
        "seed":                  42,
    },

    # ==========================================================================
    # 24. 课程学习 [v9] 三类噪声 + 风力 统一管理 (BC 已完全移除)
    #
    # 每个 level 同时定义:
    #   obs_noise   — 观测噪声 σ (加到 normalized obs 上)
    #   act_noise   — 执行噪声 σ (加到 delta_q 上, rad/step)
    #   force_noise — 环境噪声力 σ (N, 加在 payload 上, 随机方向)
    #   wind_max    — 风力上限 (N, 每 episode 在 [0, wind_max] 均匀采样)
    #
    # 晋级条件: (sr ≥ sr_threshold) AND (eps_at_level ≥ min_eps)
    # 硬上限: eps_at_level ≥ hard_cap_eps 时强制晋级 (防卡死)
    # 倒退机制 (descent): SR 跌破 regression_sr_threshold 持续 regression_min_eps 后回退一级
    # ==========================================================================
    "curriculum": {
        "enabled": True,

        # ── [v12.5] 课程倒退机制 — 用户反馈: 倒退导致策略反复反弹破坏稳定 ───
        # 用户原话: "反复反弹由于回退机制导致, 破坏其稳定性很快达到 0 成功率"
        # 解决: 关闭所有倒退, 改用 hysteresis: 学坏时延长当前级训练而非倒退
        # 文献: Narvekar et al. 2020 (Curriculum Learning Survey, arXiv:2003.04960)
        #       —— 反复倒退/晋级会让 on-policy RL (PPO) 灾难性发散
        "descent_regression_enabled": False,   # v12.5: 关闭
        "lift_regression_enabled":    False,
        "cruise_regression_enabled":  False,

        # ── [v12.5] 三段课程统一策略 ─────────────────────────────────────────
        # 1. 噪声维度: 只保留 wind_max (用户要求 + Pinto 2017 robust adversarial RL)
        #    - 取消 obs_noise: 测量噪声本可由 LSTM actor 隐式滤掉
        #    - 取消 act_noise: 执行噪声在真实系统是 motor backlash, 不大
        #    - 取消 force_noise: 已被 wind 包含 (都是外力)
        #    - 风力 wind_max 是真正的 sim2real 主要 gap (环境扰动)
        # 2. 课程: 6 级渐进 (每级风力增量 ≤ 0.5N), 不是 2 级突变
        #    旧 v12.3 L0→L1 跳 0→1N 导致 PPO 策略灾难性发散
        # 3. 晋级条件: 严格 (SR ≥ 阈值 + min_eps 充分训练)
        # 4. 不倒退: 学坏 → 延长当前级 (hard_cap_eps 增大)

        # ── Lift 课程 (6 级渐进, 风力 0 → 2.0N) ──────────────────────────────
        # 用户上限 ≤ 2N, 每级增量 0.4N
        "lift_levels": [
            {"wind_max": 0.00},
            {"wind_max": 0.40},
            {"wind_max": 0.80},
            {"wind_max": 1.20},
            {"wind_max": 1.60},
            {"wind_max": 2.00},
        ],
        "lift_sr_threshold":  0.75,
        "lift_min_eps":       200,    # v12.3: 150 → v12.5: 200 (每级训练更久)
        "lift_hard_cap_eps":  1500,   # v12.3: 1000 → v12.5: 1500
        "lift_stats_window":  60,     # v12.3: 50 → v12.5: 60

        # ── Cruise 课程 (6 级渐进, 风力 0 → 2.0N) ────────────────────────────
        "cruise_levels": [
            {"wind_max": 0.00},
            {"wind_max": 0.40},
            {"wind_max": 0.80},
            {"wind_max": 1.20},
            {"wind_max": 1.60},
            {"wind_max": 2.00},
        ],
        "cruise_sr_threshold":  0.60,
        "cruise_min_eps":       250,   # v12.3: 200 → v12.5: 250
        "cruise_hard_cap_eps":  2000,  # v12.3: 1500 → v12.5: 2000
        "cruise_stats_window":  80,

        # ── Descent 课程 (5 级渐进, 风力 0 → 1.0N) ───────────────────────────
        # descent 精度要求高 (5mm), 风力上限低于 lift/cruise
        # 每级增量 0.25N, 渐进幅度最小, 避免精度破坏
        "descent_levels": [
            {"init_xy": 0.020, "init_vel": 0.012, "init_tilt": 0.005, "xy_tol": 0.005,
             "wind_max": 0.00},
            {"init_xy": 0.020, "init_vel": 0.012, "init_tilt": 0.005, "xy_tol": 0.005,
             "wind_max": 0.25},
            {"init_xy": 0.020, "init_vel": 0.012, "init_tilt": 0.005, "xy_tol": 0.005,
             "wind_max": 0.50},
            {"init_xy": 0.020, "init_vel": 0.012, "init_tilt": 0.005, "xy_tol": 0.005,
             "wind_max": 0.75},
            {"init_xy": 0.020, "init_vel": 0.012, "init_tilt": 0.005, "xy_tol": 0.005,
             "wind_max": 1.00},
        ],
        "descent_sr_threshold":  0.80,
        "descent_min_eps":       250,   # v12.3: 200 → v12.5: 250
        "descent_hard_cap_eps":  2000,  # v12.3: 1500 → v12.5: 2000
        "descent_stats_window":   80,   # v12.3: 60 → v12.5: 80

        # OmniReset (descent 仍用; 早期阶段从目标附近开始, 加速学习)
        "omnireset_enabled":          True,
        "omnireset_near_goal_prob":   0.10,
        "omnireset_near_goal_xy":     0.012,
        "omnireset_near_goal_z_offset": 0.05,
    },

    # ==========================================================================
    # 26. 测试默认参数 (test_phase.py 用; 噪声/风力默认全部 0)
    # ==========================================================================
    "test": {
        "n_episodes":         20,
        "render":             False,
        "n_obstacles":        3,
        "obstacle_seed":      21,
        "save_paths":         False,
        "save_paths_dir":     "test_results",
        "wind_force":         0.0,
        "wind_direction":     0.0,
        "obs_noise":          0.0,
        "act_noise":          0.0,
        "force_noise":        0.0,
    },
}