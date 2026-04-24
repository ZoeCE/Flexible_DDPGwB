# ==============================================================================
# config.py — 项目全局配置（V3 修复版：Reward重构 + obs增强 + BC修复）
#
# ══════════════════════════════════════════════════════════════════════════════
# V3 变更摘要
# ══════════════════════════════════════════════════════════════════════════════
#
# [V3-OBS]  state_dim 从 10+3n+26 → 10+3n+26+9 = 54（n=3时）
#   新增: payload_tilt(替换0.), payload_yaw(替换0.)
#         phase_encode(3), progress(1), z_error(1), rebar_errors(4)
#
# [V3-RWD] 全新分层阶段感知Reward
#   - 对数势能替代二次势能（近距离梯度自然放大）
#   - Per-rebar对准奖励（最差钢筋约束）
#   - 条件门控下降（不对准就下降受罚）
#   - 阶段自适应系数（巡航/对准/插入三阶段）
#   - 轻微时间步惩罚（激励效率）
#
# [V3-BC]  BC预训练修复
#   - 学习率降低 3e-4 → 1e-4
#   - Smooth L1 Loss + per-joint归一化
#   - 分阶段样本加权
#
# [V3-PPO] PPO超参微调
#   - log_std_init: -2.5 → -1.5（更多探索）
#   - entropy_coef: 0.005 → 0.01
#   - bc_coef_init: 10.0 → 5.0（BC标签不可靠，降低依赖）
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np

DEFAULT_CONFIG = {

    # ==========================================================================
    # 1. 仿真与控制
    # ==========================================================================
    "sim": {
        "physics_dt":       0.002,
        "control_freq_hz":  10,
        "max_steps":        300,
        "render":           False,
        "substep_skip":     1,
    },

    # ==========================================================================
    # 2. 空间与动作（关节空间 7D）
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
        "instability_grace_steps": 50,
        "swing_xy_max":          0.25,
        "payload_vel_max":       2.0,
        "payload_tilt_max":      1.0,
        "payload_yaw_max":       1.2,
        "instability_penalty":   -5.0,

        "reward_clip_min":       -5.0,
        "reward_clip_max":        5.0,
    },

    # ==========================================================================
    # 7. 奖励函数系数 — V3 分层阶段感知Reward
    # ==========================================================================
    "reward": {
        # ── Layer 0: Safety终止 ──────────────────────────────────
        "success_bonus":          50.0,
        "soft_success_bonus":     20.0,
        "timeout_penalty":        -2.0,
        "collision_penalty":      -10.0,
        "out_of_bounds_penalty":  -5.0,
        "crash_penalty":          -8.0,
        "instability_penalty":    -5.0,

        # ── Layer 1: 对数势能引导 ────────────────────────────────
        "goal_log_coef_xy":       3.0,
        "goal_log_coef_z":        2.0,
        "goal_log_eps":           0.002,

        # ── 障碍物排斥 APF（保留） ───────────────────────────────
        "obstacle_rho_0":           0.10,
        "obstacle_d_min":           0.005,
        "obstacle_apf_coef":        3.0e-4,
        "obstacle_apf_max":         1.0,
        "obstacle_penalty_coef":    0.02,
        "obstacle_penalty_scale":   0.05,

        # ── Per-rebar对准 ────────────────────────────────────────
        "rebar_align_coef":         5.0,
        "rebar_worst_weight":       0.7,
        "rebar_activate_dist":      0.03,

        # ── 条件门控下降 ─────────────────────────────────────────
        "descent_reward_coef":      3.0,
        "descent_penalty_coef":     2.0,
        "descent_align_thresh_xy":  0.008,
        "descent_align_thresh_pose": 0.06,

        # ── Layer 2: 姿态惩罚（阶段自适应） ─────────────────────
        "tilt_penalty_coef_cruise":     0.15,
        "tilt_penalty_coef_insertion":  0.5,
        "yaw_penalty_coef_cruise":      0.2,
        "yaw_penalty_coef_insertion":   0.6,

        # ── 防摆 ──────────────────────────────────────────────────
        "swing_penalty_coef":      0.05,
        "swing_penalty_scale":     0.05,
        "angvel_penalty_coef":     0.02,
        "angvel_penalty_scale":    0.5,

        # ── 控制平滑 ──────────────────────────────────────────────
        "action_rate_coef_cruise":     0.1,
        "action_rate_coef_insertion":  0.5,
        "joint_smooth_coef":           0.01,

        # ── 时间步惩罚 ───────────────────────────────────────────
        "step_penalty":            -0.003,
        "step_penalty_insertion":  -0.001,

        # ── 航点 ─────────────────────────────────────────────────
        "waypoint_bonus":         1.0,

        # ── MuJoCo 碰撞检测开关 ──────────────────────────────────
        "use_mujoco_contact":        True,

        # ── 关节极限（保留兼容） ─────────────────────────────────
        "joint_limit_penalty":   -0.05,
        "joint_limit_margin":     0.1,
        "joint_smooth_penalty":  -0.005,
    },

    # ==========================================================================
    # 8. 系统延迟与噪声
    # ==========================================================================
    "noise": {
        "latency_steps":     1,
        "force_noise_level": 0.1,
    },

    # ==========================================================================
    # 9. 重置参数
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
    # 10. Prefab 几何参数 — socket 模式
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
    # 11. Target — rebar 模式
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
        "rebar_radius":       0.003,
        "rebar_half_height":  0.01,
        "rebar_rgba":         [0.8, 0.0, 0.0, 1.0],
    },

    # ==========================================================================
    # 12. 绳索生成
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
    # 13. NMPC 控制器（BC 标签生成器）
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
    # 14. PPO Agent 超参数 — V3 微调
    # ==========================================================================
        "ppo_agent": {
        "hidden_dim":            512,        # 256 → 512
        "n_layers":              3,          # 2 → 3
        "lr_actor":              2e-5,       # 稍降，大网络防振荡
        "lr_critic":             2e-4,
        "gamma":                 0.99,
        "gae_lambda":            0.98,       # 0.95→0.98 提高远期信用
        "clip_eps":              0.1,        # 0.15→0.1 更保守
        "value_loss_coef":       0.5,
        "entropy_coef":          0.005,      # 0.01→0.005 降低探索强度
        "max_grad_norm":         0.5,
        "n_steps":               2048,
        "n_epochs":              4,
        "batch_size":            256,
        "normalize_advantages":  True,
        "behavior_clone":        True,
        "bc_coef_init":          0.8,        # 5.0→0.8，避免过分束缚
        "bc_coef_final":         0.01,       # 0.1→0.01，最终近乎消失
        "bc_anneal_steps":       1_500_000,
        "bc_loss_type":          "mse",
        "use_obs_norm":          True,
        "obs_norm_clip":         10.0,
        "log_std_init":         -4.5,
        "log_std_min":          -5.0,
        "log_std_max":          -2.0,
        "target_kl":             0.03,
    },

    # ==========================================================================
    # 14b. 课程学习配置
    # ==========================================================================
    "curriculum": {
        "enabled":                    True,
        "bc_n_obstacles":             0,
        "ramp_mode":                  "performance",
        "perf_window":                30,
        "perf_sr_threshold":          0.6,
        "perf_reward_threshold":      25.0,
        "perf_min_episodes_per_level": 100,
        "perf_regression_tol":       -10.0,
        "perf_hard_cap_steps":       600_000,
        "milestones": [
            (0,          0),
            (400_000,    1),
            (700_000,    2),
            (1_100_000,  3),
            (1_500_000,  4),
            (2_000_000,  5),
        ],
        "ppo_max_n_obstacles":        3,
        "ppo_stable_n_obstacles":     0,
        "ppo_stable_timesteps":       400_000,
        "ppo_ramp_start_timesteps":   400_000,
        "ppo_ramp_end_timesteps":     2_000_000,
        "ramp_step_size":             1,
    },

    # ==========================================================================
    # 15. TD3 Agent 超参数
    # ==========================================================================
    "td3_agent": {
        "hidden_dim":           256,
        "buffer_size":          300_000,
        "batch_size":           256,
        "gamma":                0.99,
        "tau":                  0.005,
        "policy_noise":         0.2,
        "noise_clip":           0.5,
        "policy_freq":          2,
        "critic_grad_clip":     1.0,
        "actor_grad_clip":      1.0,
        "critic_loss_type":     "huber",
        "target_q_clip":        30.0,
        "behavior_clone":       True,
        "bc_alpha":             2.5,
        "epsilon_init":         1.0,
        "epsilon_min":          0.05,
        "epsilon_delta":        5e-6,
        "use_reward_norm":      False,
        "lr_actor":             3e-4,
        "lr_critic":            3e-4,
    },

    # ==========================================================================
    # 16. 训练流程超参数
    # ==========================================================================
    "train": {
        "n_episodes":            8000,
        "total_timesteps":       5_000_000,
        "warmup_episodes":       50,
        "explore_noise":         0.1,
        "min_buffer_to_train":   2048,
        "grad_updates_per_step": 1,
        "save_interval":         50,
        "eval_interval":         100,
        "eval_episodes":         10,
        "log_smooth_win":        20,
        "gpu_id":                0,
        "n_envs":                1,
    },

    # ==========================================================================
    # 17. 测试专用参数
    # ==========================================================================
    "test": {
        "n_episodes":         20,
        "render":             False,
        "n_obstacles":        0,
        "obstacle_seed":      21,
        "save_paths":         False,
        "save_paths_dir":     "test_paths",
        "ckpt_path":          None,
        "policy_type":        "ppo",
    },

    # ==========================================================================
    # 18. BC 预训练参数 — V3 修复
    # ==========================================================================
    "bc_pretrain": {
        "n_episodes":   500,
        "n_epochs":     100,
        "lr":           1e-4,       # [V3] 3e-4→1e-4 避免噪声标签过拟合
        "batch_size":   256,
    },

    # ==========================================================================
    # 19. 插入任务成功判定
    # ==========================================================================
    "insertion": {
        "entry_z":               0.16,
        "target_payload_z":      0.10,
        "success_z_tolerance":   0.020,
        "hold_steps":            3,
        "xy_tolerance":          0.005,
        "tilt_tolerance":        0.08,
        "yaw_tolerance":         0.06,
        "vel_xy_tolerance":      0.08,
        "vel_z_tolerance":       0.08,
        "require_floor_contact": False,
        "partial_dist_scale":    0.05,
    },
}