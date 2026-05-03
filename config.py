# ==============================================================================
# config.py — 三阶段 RL 控制架构全局配置
#
# ══════════════════════════════════════════════════════════════════════════════
# 架构概览
# ══════════════════════════════════════════════════════════════════════════════
#
# 三阶段分治:
#   Phase 1 - Lift:   从起始位置提升到巡航高度
#   Phase 2 - Cruise: 在巡航高度平移到终点上方(避障)
#   Phase 3 - Descent: 精准下降插入钢筋
#
# 每个阶段:
#   - 独立 RL 模型 (PPO / SAC)
#   - 统一控制接口: RL 输出 EE 3D 加速度 → 积分 → IK → 关节角
#   - 独立 reward 函数
#   - BC 预训练接口
#   - 风力/噪声课程学习接口
#
# 阶段切换由固定逻辑判断:
#   Lift → Cruise:  payload z >= z_cruise 且 摆动幅度/速度 < 阈值
#   Cruise → Descent: payload xy 到达终点上方 且 摆动幅度/速度 < 阈值
# ══════════════════════════════════════════════════════════════════════════════

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
    # 5. A* 寻路与 3D 轨迹规划 (保留用于专家/BC)
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
        "instability_grace_steps": 15,   # ★ 80→15, 不再浪费前80步
        "swing_xy_max":          0.35,   # 0.25→0.35, 平移段 EE 与 payload 有滞后
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
    # 9. Prefab 几何参数 — socket 模式
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
        "rebar_radius":       0.003,
        "rebar_half_height":  0.01,
        "rebar_rgba":         [0.8, 0.0, 0.0, 1.0],
    },

    # ==========================================================================
    # 11. 绳索生成
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
    # 12. NMPC 底层控制器 (固定不变, 用于BC专家)
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
        "success_z_tolerance":   0.030,    # ★ 0.020→0.030, 更宽松便于早期探索
        "hold_steps":            2,         # ★ 3→2, 更容易触发成功
        "xy_tolerance":          0.015,    # ★ 0.005→0.015 (15mm), 关键修复! 5mm几乎不可能
        "tilt_tolerance":        0.12,     # ★ 0.08→0.12
        "yaw_tolerance":         0.10,     # ★ 0.06→0.10
        "vel_xy_tolerance":      0.10,
        "vel_z_tolerance":       0.10,
        "require_floor_contact": False,
        "partial_dist_scale":    0.05,
        # ★ 分阶段容差: 训练早期用宽松容差, 后期收紧
        "xy_tolerance_train_start": 0.030, # 训练开始时 3cm
        "xy_tolerance_train_end":   0.010, # 训练结束时 1cm
        "xy_tolerance_anneal_steps": 1_000_000,
    },

    # ==========================================================================
    # 14. 风力扰动
    # ==========================================================================
    "wind": {
        "enabled":            False,   # 默认关闭, 先无风训练
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

    # ══════════════════════════════════════════════════════════════════════════
    # 三阶段 RL 架构配置
    # ══════════════════════════════════════════════════════════════════════════

    # ==========================================================================
    # 16. 阶段切换逻辑
    # ==========================================================================
    "phase_transition": {
        # Lift → Cruise
        "lift_to_cruise_z_threshold":    0.23,      # payload_z >= 此值
        "lift_to_cruise_tilt_max":       0.15,      # rad
        "lift_to_cruise_swing_vel_max":  0.08,      # m/s  (相对速度)
        "lift_to_cruise_payload_vel_max": 0.15,     # m/s

        # Cruise → Descent
        "cruise_to_descent_xy_dist":     0.06,      # 0.03→0.06, 6cm 范围
        "cruise_to_descent_tilt_max":    0.15,      # 0.10→0.15
        "cruise_to_descent_swing_vel_max": 0.10,    # 0.05→0.10
        "cruise_to_descent_payload_vel_max": 0.15,  # 0.10→0.15
    },

    # ==========================================================================
    # 17. 统一 RL 控制接口: EE 3D 加速度
    # ==========================================================================
    # 每阶段 RL 输出 3D EE 加速度 → 积分得到 EE 速度/位置 → IK 得到关节角
    "ee_control": {
        "acc_max_xy":         0.5,      # 1.0→0.5, 柔和加速 (pendulum T=1.4s)
        "acc_max_z":          1.5,      # 2.0→1.5
        "vel_max_xy":         0.15,     # 0.2→0.15, 更稳定的巡航速度
        "vel_max_z":          0.15,
        "vel_max_z_descent":  0.03,
        "integrator_dt":      0.1,
    },

    # ==========================================================================
    # 17b. ★ Cruise 段 PID 控制器 (Z 高度保持 + Yaw 角度保持)
    # ==========================================================================
    "cruise_z_pid": {
        # Z-axis PID: 控制 payload z ≈ z_cruise
        "kp_z":               2.0,      # 比例增益
        "ki_z":               0.5,      # 积分增益 (消除稳态误差)
        "kd_z":               0.3,      # 微分增益 (阻尼)
        "z_integral_max":     0.05,     # 积分抗饱和上限
        "z_correction_max":   0.08,     # 最大 EE z 修正量 (m)

        # Yaw-axis PD: 控制 payload yaw ≈ 0
        "kp_yaw":             1.0,
        "kd_yaw":             0.2,
        "yaw_correction_max": 0.3,      # rad

        # Floor collision detection (payload 坠落检测)
        "floor_z_threshold":  0.04,     # payload z < 此值视为坠落
        "floor_vz_threshold": -0.15,    # payload vz < 此值视为快速下坠
    },

    # ==========================================================================
    # 18. Phase 1: Lift RL 配置
    # ==========================================================================
    "lift_rl": {
        # ── 观测维度 ──
        # ee_pos(3) + ee_vel(3) + pl_pos(3) + pl_vel(3) +
        # ee_pl_offset(3) + tilt/yaw/tilt_rate/yaw_rate(4) +
        # start_xy(2) + target_z(1) + z_error(1) = 23
        "obs_dim":            23,
        "action_dim":         3,        # EE 加速度 (ax, ay, az)

        # ── 初始化随机范围 ──
        "init_xy_range":      0.01,     # 0.03→0.01
        "init_z_range":       0.01,     # 0.02→0.01
        "init_vel_range":     0.0,

        # ── 训练参数 ──
        "max_steps":          100,      # 提升阶段最多步数
        "target_z_cruise":    0.25,     # 目标巡航高度

        # ── Reward 参数 ──
        "reward": {
            "z_approach_coef":    5.0,      # 接近巡航高度奖励系数
            "swing_ke_coef":      2.0,      # 摆动动能惩罚
            "success_bonus":      10.0,     # 成功到达巡航高度
            "step_penalty":       -0.01,    # 每步小惩罚
            "instability_penalty": -5.0,    # 失稳惩罚
            "crash_penalty":      -5.0,     # 坠毁惩罚
        },
    },

    # ==========================================================================
    # 19. Phase 2: Cruise RL 配置
    # ==========================================================================
    "cruise_rl": {
        # ── 观测维度 ──
        # ee_xy(2) + ee_vxy(2) + pl_xy(2) + pl_vxy(2) +
        # ee_pl_offset_xy(2) + tilt/yaw/tilt_rate/yaw_rate(4) +
        # target_xy(2) + target_dist(1) +
        # obstacles(3*n_obs_max=9) + joint_q(7) = 33
        "obs_dim":            33,       # 33 for n_obstacles=3
        "action_dim":         2,        # EE 加速度 (ax, ay), z 锁定

        # ── 初始化随机范围 ──
        "init_xy_range":      0.01,     # 0.03→0.01, 更小随机偏移
        "init_z_range":       0.01,     # 0.02→0.01
        "init_vel_range":     0.0,      # 0.03→0, 零初始速度 (让agent从静止学起)

        # ── 训练参数 ──
        "max_steps":          300,      # 平移阶段最多步数
        "z_lock_height":      0.25,     # 锁定高度

        # ── Reward 参数 v3 ──
        "reward": {
            # PBRS 势能差分 (唯一距离驱动力, 删除了 dist_coef 和 direction_coef)
            "pbrs_coef":          15.0,     # Φ(s) = -dist, r = γΦ(s')-Φ(s)
            "pbrs_gamma":         0.99,

            # 障碍物排斥
            "obs_repulse_coef":   2.0,
            "obs_repulse_d0":     0.12,

            # 摆动惩罚 (低权重)
            "swing_ke_coef":      0.3,

            # 存活/时间
            "alive_bonus":        0.01,
            "step_penalty":       -0.005,

            # 终端
            "collision_penalty":  -10.0,
            "success_bonus":      50.0,
            "instability_penalty": -5.0,

            # 渐进成功半径
            "success_radius_start": 0.20,
            "success_radius_end":   0.06,
            "success_radius_anneal_steps": 500_000,

            # 里程碑
            "milestone_radius":   0.15,
            "milestone_bonus":    5.0,
        },
    },

    # ==========================================================================
    # 20. Phase 3: Descent RL 配置
    # ==========================================================================
    "descent_rl": {
        "obs_dim":            29,
        "action_dim":         3,        # EE 加速度 (ax, ay, az)

        # ── 初始化随机范围 ──
        "init_xy_range":      0.005,    # ★ 0.01→0.005, 更靠近目标开始
        "init_z_range":       0.01,
        "init_tilt_range":    0.01,     # ★ 0.02→0.01
        "init_vel_range":     0.0,

        # ── 训练参数 ──
        "max_steps":          300,      # ★ 200→300, 给更多时间下降
        "acc_max_xy":         0.5,      # ★ 1.0→0.5, 下降段需精细控制
        "acc_max_z":          1.0,      # ★ 1.5→1.0
        "vel_max_z":          0.05,     # ★ 0.03→0.05, 0.03太慢且不稳定

        # ── Reward 参数 ──
        "reward": {
            "xy_align_coef":      3.0,      # ★ 5.0→3.0, 降低 xy 惩罚绝对值防梯度压制
            "z_descent_coef":     5.0,      # ★ 3.0→5.0, 更鼓励下降
            "swing_ke_coef":      2.0,      # ★ 3.0→2.0, 适当降低防止 agent 完全不动
            "tilt_coef":          1.0,      # ★ 2.0→1.0
            "success_bonus":      50.0,     # ★ 30→50, 强化成功信号
            "soft_success_bonus": 20.0,     # ★ 15→20
            "step_penalty":       -0.001,   # ★ -0.002→-0.001
            "instability_penalty": -5.0,
            "crash_penalty":      -5.0,
        },
    },

    # ==========================================================================
    # 21. PPO 超参数 (三阶段共享基础配置)
    # ==========================================================================
    "ppo": {
        "hidden_dim":            256,
        "n_layers":              3,
        "lr_actor":              3e-4,
        "lr_critic":             1e-3,
        "gamma":                 0.99,
        "gae_lambda":            0.95,
        "clip_eps":              0.2,
        "value_loss_coef":       0.5,
        # ★ 熵系数提高: 0.05→0.01 (注意: 更高熵系数反而导致策略发散, 这里用适中值)
        # ★ 真正防止熵坍缩靠 log_std_floor annealing, 而非过高 entropy_coef
        "entropy_coef":          0.02,
        "max_grad_norm":         0.5,
        "n_steps":               2048,
        "n_epochs":              10,
        "batch_size":            256,      # ★ 128→256
        "normalize_advantages":  True,
        "use_obs_norm":          True,
        "obs_norm_clip":         10.0,
        # ★ log_std 设置: 配合 BC 后的 RL 微调
        # BC 完成后策略已经有方向感, 初始 std 不用太大
        "log_std_init":         -0.5,      # ★ 0.0→-0.5, BC后 std≈0.6, 适中探索
        "log_std_min":          -3.0,      # ★ -2.0→-3.0, 允许更精细动作
        "log_std_max":           1.0,
        "target_kl":             0.05,     # ★ 0.02→0.05, 放宽 early stop, 让更新充分
        # ★ log_std_floor annealing 参数
        "log_std_floor_init":   -0.5,      # ★ 初始下限 (std≈0.6), 防熵早崩
        "log_std_floor_final":  -3.0,      # ★ 最终下限
        "log_std_floor_steps":   800_000,  # ★ 在 800k 步内线性退火
    },

    # ==========================================================================
    # 22. SAC 超参数 (三阶段共享基础配置)
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
        "warmup_mode":          "expert",   # "expert" 或 "random"
        "auto_alpha":           True,
        "alpha_init":           0.2,
        "target_entropy_ratio": 0.3,
        "critic_grad_clip":     1.0,
        "actor_grad_clip":      1.0,
        "reward_scale":         1.0,
        "update_interval":      1,
        "updates_per_step":     2,
        "use_obs_norm":         True,
        "obs_norm_clip":        10.0,
    },

    # ==========================================================================
    # 23. 训练总体参数
    # ==========================================================================
    "train": {
        "total_timesteps":       2_000_000,   # 每个阶段的训练步数
        "save_interval":         50,
        "eval_interval":         100,
        "eval_episodes":         10,
        "log_smooth_win":        20,
        "gpu_id":                0,
        "n_envs":                1,
        "algo":                  "ppo",       # "ppo" 或 "sac"
        "seed":                  42,
    },

    # ==========================================================================
    # 24. BC 预训练配置 (三阶段共享)
    # ==========================================================================
    "bc_pretrain": {
        "enabled":           True,
        # ★ 大幅增加 BC 数据量: 300→1000 episodes
        "n_episodes":        1000,
        # ★ 增加 epoch 数: 80→200, 配合 epsilon 退火 + 独立 eval 早停
        "n_epochs":          200,
        "lr":                3e-4,       # ★ 1e-4→3e-4, 更快收敛
        "lr_decay":          0.5,        # ★ 每 50 epoch LR 减半
        "lr_decay_interval": 50,
        "batch_size":        512,        # ★ 256→512
        "max_buffer_size":   500_000,    # ★ 200k→500k
        # ★ 独立 eval: 每 eval_interval epoch 用专家 eval 一次, early stop
        "eval_interval":     20,
        "eval_episodes":     30,
        "patience":          5,          # 连续 5 次 eval loss 不降则停止
        # ★ epsilon 退火: BC 训练中加入动作噪声, 模拟 DAgger
        "epsilon_start":     0.3,        # 初始探索比例 (30% 随机动作)
        "epsilon_end":       0.0,        # 结束时纯 BC
        # ★ 验证损失阈值: loss < 此值认为 BC 足够好
        "loss_threshold":    0.02,
    },

    # ==========================================================================
    # 25. 课程学习配置
    # ==========================================================================
    "curriculum": {
        "enabled":                    False,  # 默认关闭, 先无风无障碍训练到稳定
        # 风力课程
        "wind_start_frac":            0.0,    # 初始风力倍率
        "wind_end_frac":              1.0,    # 最终风力倍率
        "wind_anneal_steps":          500_000,
        # 噪声课程
        "noise_start_scale":          0.0,
        "noise_end_scale":            1.0,
        "noise_anneal_steps":         500_000,
        # 障碍物课程 (仅 cruise 阶段)
        "obstacle_enabled":           True,
        "obstacle_start_n":           0,
        "obstacle_max_n":             3,
        "perf_window":                30,
        "perf_sr_threshold":          0.5,
        "perf_reward_threshold":      10.0,
        "perf_min_episodes_per_level": 100,
        "perf_hard_cap_steps":        600_000,
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
    # 27. 兼容旧版 — 保留的配置项
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