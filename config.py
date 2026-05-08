# ==============================================================================
# config.py — 三阶段 RL 控制架构全局配置 v3
#
# v3 主要变更 (相对 v2):
#
# [ARCH-CRUISE] Cruise 段架构变更:
#   取消 SwingDampingController residual 模式
#   RL 直接输出完整 xy 加速度 (不再与底层防摆叠加)
#   SwingDamping 保留用于监控, 但不参与控制
#
# [ARCH-DESCENT] Descent 段架构变更:
#   用 JointSpaceExpert.compute_delta_q_target() 作为 base PID controller
#   RL 输出 3D 残差 delta_q (在 PID delta_q 基础上按比例叠加)
#   残差限幅: residual_dq_scale=0.30 (RL 贡献不超过总 delta_q 的 30%)
#
# [REW-CRUISE] Cruise 奖励重构 v3:
#   z_dev_coef/vz_penalty_coef 归零 (PID 负责 Z, reward 不重复)
#   swing_energy 上限降到 0.5/step (不压倒 PBRS)
#   pbrs_coef 提升到 20 (绝对主导导航信号)
#   alive_bonus 完全去除
#
# [REW-DESCENT] Descent 奖励重构 v3:
#   z_unconditional_frac 归零 (防止不对准就下降)
#   rebar_sdf 仅在 dtf<8mm AND z_reached 时激活
#   near_goal_focus 上限 0.10/step (低于 success_bonus)
#   step_penalty 加大到 -0.02
#
# [CUR-CRUISE] Cruise 课程重构:
#   全程距离 (cruise_dist_curriculum=False)
#   0 障碍物 (obstacle_enabled=False)
#   success_radius 从 0.20m 开始
#
# [CUR-DESCENT] Descent 课程重构:
#   init_xy_start 2mm (更接近 BC 场景)
#   tol_mult=3.0
#   BC epochs: descent=20, cruise=40
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
    # 3. 任务与随机化初始状态
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
    # 4. 场景生成
    # ==========================================================================
    "scene": {
        "n_obstacles":        3,
        "radius_range":       (0.006, 0.015),
        "obstacle_z_center":  0.15,
        "obstacle_halfheight": 0.15,
        "endpoint_z_offset":  0.025,
        "seed":               None,
    },

    # ==========================================================================
    # 5. A* 寻路与 3D 轨迹规划
    # ==========================================================================
    "planning": {
        "workspace_radius":  0.50,
        "payload_radius":    0.075,
        "planning_margin":   0.03,
        "planning_grid_res": 0.025,
        "path_width":        0.6,
        "corridor_side":          +1,
        "corridor_forbid_margin": 0.05,
        "bounds_margin":     0.5,
        "max_expansions":    100000,
        "payload_z_cruise":  0.25,
        "target_z_descent":  0.12,
        "num_descent_steps": 15,
        "num_lift_steps":    3,
    },

    # ==========================================================================
    # 6. Step 逻辑判定
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
    # 7. 系统延迟与噪声
    # ==========================================================================
    "noise": {
        "latency_steps":     1,
        "force_noise_level": 0.1,
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
    # 9. Prefab 几何参数
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
        "rebar_radius":       0.0025,  # [v3.4] 3mm→2.5mm: 略降难度, xy_tol 4→4.5mm
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
    # 12. NMPC 底层控制器
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
        "xy_tolerance":          0.0045,  # [v3.4] 随 rebar_radius 2.5mm 更新: 7-2.5=4.5mm
        "tilt_tolerance":        0.05,
        "yaw_tolerance":         0.08,
        "hold_steps":            3,
        "vel_xy_tolerance":      0.05,
        "vel_z_tolerance":       0.05,
        "require_floor_contact": False,
        "partial_dist_scale":    0.05,

        # [v3.4] xy_tol 改为 per-level 查表, 不再使用 tol_mult 计算
        # tol_mult 保留字段但 reward 层不再读取 (由 rstate.current_xy_tol 直接提供)
        "xy_tol_range_multiplier":     3.0,   # DEPRECATED - 不再使用
        "xy_tolerance_train_end":      0.005,

        "tilt_tolerance_train_start":  0.12,
        "tilt_tolerance_train_end":    0.05,
        "yaw_tolerance_train_start":   0.15,
        "yaw_tolerance_train_end":     0.08,
        "xy_tolerance_anneal_steps":   500_000,
    },

    # ==========================================================================
    # 14. 风力扰动
    # ==========================================================================
    "wind": {
        "enabled":            False,
        "F_max":              1.0,
        "theta_rate_std":     0.15,
        "force_rate_std":     0.1,
        "seed":               123,
        "curriculum_start":   0,
        "curriculum_end":     200_000,
    },

    # ==========================================================================
    # 15. 旧版奖励配置（兼容 env 内部 _compute_reward）
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
    # 16. 阶段切换逻辑
    # ==========================================================================
    "phase_transition": {
        "lift_to_cruise_z_threshold":    0.23,
        "lift_to_cruise_tilt_max":       0.15,
        "lift_to_cruise_swing_vel_max":  0.08,
        "lift_to_cruise_payload_vel_max": 0.15,
        "cruise_to_descent_xy_dist":     0.06,
        "cruise_to_descent_tilt_max":    0.15,
        "cruise_to_descent_swing_vel_max": 0.10,
        "cruise_to_descent_payload_vel_max": 0.15,
    },

    # ==========================================================================
    # 17. EE 加速度控制器
    # ==========================================================================
    "ee_control": {
        "acc_max_xy":         0.8,
        "acc_max_z":          1.5,
        "vel_max_xy":         0.25,
        "vel_max_z":          0.15,
        "vel_max_z_descent":  0.03,
        "integrator_dt":      0.1,
    },

    # ==========================================================================
    # 17b. Cruise 段 PID 控制器 (Z/Yaw)
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
    # 17c. Cruise 段防摆控制器 (v3: 仅监控, 不参与控制)
    # ==========================================================================
    "cruise_swing_damping": {
        "kd_vel":         3.5,
        "kp_pos":         0.3,
        "acc_max":        0.7,
        "adaptive":       True,
        "energy_ref":     0.02,
        "gain_max_scale": 2.0,
    },

    # ==========================================================================
    # 18. Phase 1: Lift RL
    # ==========================================================================
    "lift_rl": {
        "obs_dim":            23,
        "action_dim":         3,
        "init_xy_range":      0.01,
        "init_z_range":       0.01,
        "init_vel_range":     0.0,
        "max_steps":          100,
        "target_z_cruise":    0.25,
        "reward": {
            "z_approach_coef":    5.0,
            "swing_ke_coef":      2.0,
            "success_bonus":      10.0,
            "step_penalty":       -0.01,
            "instability_penalty": -3.0,
            "crash_penalty":      -5.0,
        },
    },

    # ==========================================================================
    # 19. Phase 2: Cruise RL [v3]
    #
    # 架构: RL 直接输出完整 xy 加速度 (不是残差)
    # 课程: 全程距离, 0 障碍物, 大成功半径
    # ==========================================================================
    "cruise_rl": {
        "obs_dim":            38,
        "action_dim":         2,
        "init_xy_range":      0.01,
        "init_z_range":       0.01,
        "init_vel_range":     0.0,
        "max_steps":          400,
        "min_steps_for_success": 5,
        "estimated_full_dist_m": 0.41,
        "z_lock_height":      0.25,

        # [v3.5] acc_max_xy: 0.80→0.60
        # 降低 25%, 配合 log_std_max=-0.3 控制动作幅度
        # damp_gain 一直满增益 (2.0) 说明 RL 动作引发摆动过大
        "residual_acc_max_xy": 0.60,

        "reward": {
            # ── 导航 (主导) ─────────────────────────────────────────────────
            "pbrs_coef":          20.0,   # [v3] 15→20: 主导导航信号
            "pbrs_gamma":         0.99,

            # ── 障碍物排斥 (保留 reward, 0 障碍物训练时无实际惩罚) ──────────
            # 目的: 让 critic 的值函数已经见过障碍物相关的 reward 项,
            # 加入障碍物时 critic 不需要重新适应
            "obs_repulse_coef":   0.3,
            "obs_repulse_d0":     0.10,
            "obs_repulse_linear": True,

            # ── 摆动约束 (软约束, 严格限幅防止压倒 PBRS) ───────────────────
            # [v3] 正常导航摆动 ~0.1-0.3J → penalty ~0.05-0.15/step
            # PBRS 最大 ~0.5/step → 摆动惩罚不超过 PBRS 的 30%
            "swing_energy_thresh":        0.15,   # [v3] 0.08→0.15
            "swing_energy_penalty_coef":  1.0,    # [v3] 2.0→1.0
            # 单步摆动惩罚硬上限: 0.5/step
            "swing_energy_penalty_max":   0.50,   # [v3] 新增: 硬上限
            "swing_ke_coef":      0.0,

            # ── Z/Vz (PID 控制, reward 不重复惩罚) ──────────────────────────
            "z_dev_coef":         0.0,    # [v3] 1.0→0: PID 负责 Z
            "vz_penalty_coef":    0.0,    # [v3] 1.0→0

            # ── 时间惩罚 ────────────────────────────────────────────────────
            "alive_bonus":        0.0,    # [v3] 完全去除
            "step_penalty":       -0.01,

            # ── 碰撞 / 失稳 ─────────────────────────────────────────────────
            "collision_penalty":  -10.0,
            "success_bonus":      8.0,
            "instability_penalty": -3.0,

            # ── 成功半径 [v3] ───────────────────────────────────────────────
            # 起始 0.20m (更松), 随 SR 提升收紧
            # [v3.4] success_radius 随障碍物课程收紧:
            #   n_obs=0: 0.20m, n_obs=1: 0.15m, n_obs=2: 0.12m, n_obs=3: 0.10m
            # 见 CurriculumManager.get_current_success_radius()
            "success_radius_start": 0.20,
            "success_radius_end":   0.10,
            # success_radius_test 已弃用 (测试改用 phase_transition 条件)
            "success_radius_test":  0.10,   # [v3.4] 与 success_radius_end 对齐
            "success_radius_anneal_steps": 9_999_999,

            # ── 里程碑 / 近目标 ─────────────────────────────────────────────
            "milestone_radius":   0.15,
            "milestone_bonus":    1.5,
            "near_goal_radius":   0.10,
            "z_success_tol_frac": 0.15,
            "near_goal_bonus":    0.05,
        },
    },

    # ==========================================================================
    # 20. Phase 3: Descent RL [v3]
    #
    # 架构: JointSpaceExpert PID 作为 base controller
    #       RL 输出 3D 残差 delta_q (residual_dq_scale 限幅)
    # 奖励: 纯差分, 无 per-step 持续奖励 exploit
    # ==========================================================================
    "descent_rl": {
        "obs_dim":            29,
        "action_dim":         3,
        "init_xy_range":      0.030,
        "init_z_range":       0.005,
        "init_tilt_range":    0.010,
        "init_vel_range":     0.030,
        "max_steps":          500,

        # [v3 ARCH-DESCENT] PID + residual RL
        "pid_residual_mode":   True,
        # RL 残差缩放: rl_delta_q_contribution = residual_dq_scale * pid_delta_q_norm
        # PID 主导动作, RL 只做精细修正
        "residual_dq_scale":   0.30,      # [v3] RL 贡献上限 = 30% of PID action magnitude
        # 作为 EEAccController 的补充限幅 (backup)
        "residual_acc_max_xy": 0.30,
        "residual_acc_max_z":  0.50,

        "acc_max_xy":         0.5,
        "acc_max_z":          1.0,
        "vel_max_z":          0.05,

        "reward": {
            # ── XY 对准 (差分) ───────────────────────────────────────────────
            "xy_align_coef":      4.0,

            # ── Z 下降 (差分, 严格门控) ──────────────────────────────────────
            "z_descent_coef":     6.0,
            "z_unconditional_frac": 0.0,  # [v3] 0.15→0: 完全移除无条件下降
            "z_descent_xy_gate":  0.030,  # 30mm 内才给 z 奖励

            # [v3] 上升惩罚 (适度, PID 已阻止激进上升)
            "z_rise_penalty_coef": 2.0,
            "z_rise_max_penalty":  0.2,

            # ── 稳定性 ───────────────────────────────────────────────────────
            "swing_ke_coef":      3.0,
            "tilt_coef":          1.5,
            "yaw_coef":           2.0,

            # ── [v3.2] 移除所有 per-step 持续正向奖励 ────────────────────────
            # xy_gauss / rebar_sdf / near_goal_focus 均已删除.
            # 根因: 这些奖励随 episode 长度线性累积, 导致失败ep(500步)总reward
            # 远超成功ep(150步), 造成 reward 与 SR 反相关.
            # 替代: 成功时给一次性精准度奖励 (precision_bonus)

            # 保留字段供兼容 (reward 函数中不再使用)
            "xy_gauss_sigma":     0.020,   # [v3.2 UNUSED] 已移除 per-step gauss
            "xy_gauss_coef":      0.0,     # [v3.2] 归零
            "rebar_sdf_enabled":  False,   # [v3.2] 已移除 per-step sdf
            "rebar_sdf_coef":     0.0,     # [v3.2] 归零
            "near_goal_focus_enabled": False,   # [v3.2] 已移除 per-step focus
            "near_goal_focus_coef":    0.0,     # [v3.2] 归零

            # ── 成功时一次性精准度奖励 (替代 per-step 高斯) ─────────────────
            # 仅在 insertion_success 时给一次, 不影响 episode 长度
            # dtf=0mm → +10, dtf=5mm → +6.1, dtf=10mm → +0.8
            "precision_bonus_coef":  10.0,
            "precision_bonus_sigma": 0.005,   # 5mm sigma

            # ── 时间惩罚 (加大, 逼迫快速完成) ──────────────────────────────
            "step_penalty":       -0.02,  # [v3] -0.005→-0.02

            # ── 终止 ────────────────────────────────────────────────────────
            "success_bonus":      50.0,
            "instability_penalty": -3.0,
            "crash_penalty":      -5.0,
        },
    },

    # ==========================================================================
    # 21. PPO 超参数
    # ==========================================================================
    "ppo": {
        "hidden_dim":            256,
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
        "log_std_min":          -2.0,
        # [v3.5] log_std_max: 0.0→-0.3
        # std 上限从 1.0→0.74, 动作幅度降低 26%
        # 防止 cruise logstd 单调上升到 -0.12 导致摆动失控
        "log_std_max":          -0.3,
        "target_kl":             0.03,
        "log_std_floor_init":   -0.5,
        "log_std_floor_final":  -1.5,
        "log_std_floor_steps":   800_000,
        # [v3.5] entropy 退火加速: 3M→800k steps, start 0.03→0.01
        # 原退火在 20k ep 训练结束时仅完成 67%, entropy_coef 仍高达 0.016
        # 持续高 entropy_coef 是 logstd 单调上升的根本原因
        "entropy_coef_start":         0.01,
        "entropy_coef_end":           0.001,
        "entropy_coef_anneal_steps":  800_000,
        "plasticity_reset_interval":  200_000,
    },

    # ==========================================================================
    # 22. SAC 超参数
    # ==========================================================================
    "sac": {
        "hidden_dim":           256,
        "n_layers":             3,
        "lr_actor":             3e-4,
        "lr_critic":            3e-4,
        "lr_alpha":             1e-4,
        "gamma":                0.99,
        "tau":                  0.005,
        "buffer_size":          500_000,
        "batch_size":           256,
        "warmup_steps":         5000,
        "warmup_mode":          "expert",
        "auto_alpha":           True,
        "alpha_init":           0.2,
        "target_entropy_ratio": 0.5,
        "critic_grad_clip":     1.0,
        "her_k":               4,
        "her_reward_scale":    0.3,
        "actor_grad_clip":      1.0,
        "reward_scale":         1.0,
        "update_interval":      1,
        "updates_per_step":     2,
        "use_obs_norm":         True,
        "obs_norm_clip":        10.0,
        "obs_norm_warm_start":  5000,
    },

    # ==========================================================================
    # 23. 训练总体参数
    # ==========================================================================
    "train": {
        "total_timesteps":       2_000_000,
        "save_interval":         50,
        "eval_interval":         100,
        "eval_episodes":         10,
        "log_smooth_win":        20,
        "gpu_id":                0,
        "n_envs":                1,
        "algo":                  "ppo",
        "seed":                  42,
    },

    # ==========================================================================
    # 24. BC 预训练配置 [v3]
    # ==========================================================================
    "bc_pretrain": {
        "enabled":           True,
        "n_episodes":        1000,
        "n_epochs":          20,          # [v3] descent: 60→20, 防过拟合/熵崩塌
        "n_epochs_cruise":   40,          # [v3] cruise 专用 epoch 数
        "lr":                3e-4,
        "lr_decay":          0.5,
        "lr_decay_interval": 20,
        "batch_size":        512,
        "eval_interval":     5,           # [v3] 更早检测过拟合
        "eval_episodes":     30,
        "patience":          3,           # [v3] 5→3: 更早停止
        "epsilon_start":     0.3,
        "epsilon_end":       0.0,
        "loss_threshold":    0.30,
        "n_dagger_rounds":   2,
    },

    # ==========================================================================
    # 25. 课程学习配置 [v3]
    # ==========================================================================
    "curriculum": {
        "enabled":                    True,

        # 风力课程
        "wind_start_frac":            0.0,
        "wind_end_frac":              1.0,
        "wind_anneal_steps":          200_000,

        # 噪声课程
        "noise_start_scale":          0.0,
        "noise_end_scale":            1.0,
        "noise_anneal_steps":         500_000,

        # ── Cruise 课程 [v3] ─────────────────────────────────────────────────
        # 障碍物课程: 从 0 个真实障碍物开始, 渐进到 3 个
        # [v3 SHADOW] 即使真实障碍物 = 0, 仍会在 reward 中注入 shadow_obstacles
        # (随机采样的虚拟障碍物, 不参与物理碰撞), 让 critic 提前学习障碍物特征
        # 真实障碍物数量由课程晋级条件控制: 0→1→2→3
        "obstacle_enabled":           True,    # [v3] 启用障碍物课程
        "obstacle_start_n":           0,       # 从 0 个真实障碍物开始
        "obstacle_max_n":             3,
        "obstacle_unlock_dist_frac":  0.65,
        "obstacle_hard_cap_grad_steps": 999_999_999,
        "obstacle_hard_cap_eps":      999_999,
        "obstacle_level_warmup_eps":  300,
        "perf_window":                200,    # [v3.3] 300→200: 更快响应 SR 变化
        # [v3.3] 修正: 图中 n_obs=2 阶段 SR 稳定在 0.55-0.60
        #   obs_sr_thresh=0.65: 永远无法满足 (n_obs=2场景下SR天然更低)
        #   obs_sr_thresh=0.50: 过低, obs_min_eps=500 会导致 level 0 仅500ep就晋级
        #   obs_sr_thresh=0.55: n_obs=2 时 SR~0.57-0.60 可满足, obs_min_eps=1000 保证充分训练
        "perf_sr_threshold":          0.55,   # [v3.3] 0.65→0.55: n_obs=2 SR约0.57-0.60可触发
        "perf_reward_threshold":      -15.0,
        # [v3.5] 2000: 防止 n_obs=2→3 跳太快
        # 图中 2→3 几乎同时发生, agent 来不及适应就面对 3 个障碍物
        "perf_min_episodes_per_level": 1500,
        "perf_hard_cap_steps":        999_999_999,

        # [v3 SHADOW OBSTACLES] 0 真实障碍物阶段使用的"影子障碍物"配置
        # shadow_obstacles 在每个 episode 开始时随机采样, 不参与物理碰撞,
        # 但参与 reward 中的 obs_repulse 计算, 让 critic 提前学习障碍物特征
        # 晋级到 n_real>=1 后, shadow_obstacles 被真实障碍物替代
        "shadow_obstacle_enabled":    True,    # 0 真实障碍物时启用影子障碍物
        "shadow_obstacle_n":          3,       # 影子障碍物数量 (= obstacle_max_n)
        "shadow_obstacle_r_min":      0.006,   # 半径范围 (同真实障碍物)
        "shadow_obstacle_r_max":      0.015,

        # 距离课程: 关闭, 全程训练
        "cruise_dist_curriculum":     False,   # [v3] 关闭距离课程
        "cruise_dist_start_frac":     1.0,
        "cruise_dist_end_frac":       1.0,
        "cruise_dist_anneal_steps":   9_999_999,
        "cruise_dist_sr_threshold":   0.65,

        # ── Descent 课程 [v3] ────────────────────────────────────────────────
        # OmniReset
        "omnireset_enabled":          True,
        "omnireset_near_goal_prob":   0.10,    # [v3] 0.15→0.10
        "omnireset_near_goal_xy":     0.012,
        "omnireset_near_goal_z_offset": 0.05,

        # ── Descent 初始化课程 [v3.4 重设计] ─────────────────────────────────
        # 核心修改:
        #   1. 每个 Level 单独定义 (init_xy, init_vel, init_tilt, xy_tol)
        #      tol 与 init_xy 解耦, 独立递减收紧
        #   2. Level 间距从线性改为近似对数间距 (2→5→10→20→30mm)
        #      小偏差阶段有足够训练密度, 大偏差阶段也有覆盖
        #   3. xy_tol 随 level 递减 (15→12→10→8→6mm)
        #      Level 0-2: tol > init_xy, PID 基础能力即可成功, 快速积累成功经验
        #      Level 3-4: tol < init_xy, agent 必须主动对准才能成功
        #   4. window=50 (30→50): 减少 SR 估计噪声
        #   5. min_eps=200 (80→200): 给每个 level 足够稳定时间
        #
        # Level 参数表:
        #   Lv  init_xy   init_vel  init_tilt  xy_tol   任务难度
        #    0    2mm       0mm/s     0.06°      15mm    PID即可, 快速建立信心
        #    1    5mm       5mm/s     0.11°      12mm    PID基本可达
        #    2   10mm      10mm/s     0.23°      10mm    PID刚好可达, RL开始需要
        #    3   20mm      15mm/s     0.34°       8mm    必须主动对准
        #    4   30mm      20mm/s     0.46°       6mm    接近真实插入要求(5mm)
        "descent_init_curriculum":    True,

        # per-level 参数 (替代旧的 start/end 线性插值)
        # 格式: [level0, level1, level2, level3, level4]
        "descent_init_xy_ranges":   [0.002, 0.005, 0.010, 0.020, 0.030],
        "descent_init_vel_ranges":  [0.000, 0.005, 0.010, 0.015, 0.020],
        "descent_init_tilt_ranges": [0.001, 0.002, 0.004, 0.006, 0.008],
        "descent_init_xy_tols":     [0.015, 0.012, 0.010, 0.008, 0.006],

        # 兼容旧字段 (不再使用, 保留避免 KeyError)
        "descent_init_xy_start":    0.002,
        "descent_init_xy_end":      0.030,
        "descent_init_vel_start":   0.000,
        "descent_init_vel_end":     0.020,
        "descent_init_tilt_start":  0.001,
        "descent_init_tilt_end":    0.008,

        # Level 晋级配置
        "descent_cur_levels":         5,
        "descent_cur_min_eps":        200,     # [v3.4] 80→200: 每 level 最少200ep
        "descent_cur_sr_threshold":   0.50,
        "descent_cur_hard_cap":       300_000,
        "descent_cur_stats_window":   50,      # [v3.4] 新增: SR 统计窗口 (30→50)

        # Bootstrapped PBRS (关闭)
        "bootstrapped_pbrs_enabled":  False,
        "bootstrapped_pbrs_coef":     0.5,
    },

    # ==========================================================================
    # 26. 测试配置
    # ==========================================================================
    "test": {
        "n_episodes":         20,
        "render":             False,
        "n_obstacles":        3,
        "obstacle_seed":      21,
        "save_paths":         False,
        "save_paths_dir":     "test_results",
    },

    # ==========================================================================
    # 27. 兼容旧版
    # ==========================================================================
    "ppo_planner": {
        "hidden_dim": 256, "n_layers": 3,
        "lr_actor": 3e-4, "lr_critic": 1e-3,
        "gamma": 0.99, "gae_lambda": 0.97,
        "clip_eps": 0.2, "value_loss_coef": 0.5,
        "entropy_coef": 0.01, "max_grad_norm": 0.5,
        "n_steps": 2048, "n_epochs": 4, "batch_size": 256,
        "normalize_advantages": True, "use_obs_norm": True,
        "obs_norm_clip": 10.0, "log_std_init": -1.0,
        "log_std_min": -4.0, "log_std_max": 0.5, "target_kl": 0.03,
    },
    "sac_planner": {
        "hidden_dim": 256, "n_layers": 3,
        "lr_actor": 3e-4, "lr_critic": 3e-4, "lr_alpha": 1e-4,
        "gamma": 0.99, "tau": 0.005, "buffer_size": 500_000,
        "batch_size": 256, "warmup_steps": 5000, "warmup_mode": "expert",
        "auto_alpha": True, "alpha_init": 0.2, "target_entropy_ratio": 0.3,
        "critic_grad_clip": 1.0, "actor_grad_clip": 1.0,
        "reward_scale": 1.0, "update_interval": 1, "updates_per_step": 2,
        "use_obs_norm": True, "obs_norm_clip": 10.0,
    },
    "planner": {
        "obs_dim": 49, "action_dim": 5,
        "hidden_dim": 256, "n_layers": 3,
        "target_range_xy": 0.15, "target_range_z": 0.10,
        "speed_range": [0.05, 0.5], "waypoint_horizon": 5,
    },
}