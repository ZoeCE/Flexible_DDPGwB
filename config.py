# ==============================================================================
# config.py — 三阶段 RL 控制架构全局配置 v2 (优化版)
#
# 主要变更 (相对 v1):
#   [OPT-ENT]  PPO 熵崩塌修复:
#              entropy_coef 0.05→0.15, log_std_min -2.5→-1.5,
#              log_std_floor_init -0.5→0.0, target_kl 0.015→0.03
#   [OPT-REW]  Cruise reward 重平衡:
#              alive_bonus 0.08→0.02, pbrs_coef 10→20,
#              去除存活套利 → 到达驱动更强
#   [OPT-REW]  Descent reward 重平衡:
#              z_descent 无条件权重提升, xy_align 解耦 align_factor,
#              step_penalty 修正
#   [OPT-CUR]  Cruise 课程学习: 距离课程 (初始化随机范围渐进扩大)
#   [OPT-CUR]  Descent 课程学习: 初始化 xy_range/tilt/vel 渐进扩大
#   [OPT-BC]   BC 预训练: n_epochs 200→60, patience 8→5 (防 descent 过拟合)
#   [OPT-SAC]  SAC 超参: target_entropy_ratio 0.3→0.5 (更高目标熵)
#   [OPT-INIT] Descent init_xy_range 0.010→0.030, init_vel_range 0→0.03
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
        "rebar_radius":       0.003,
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
        "xy_tolerance":          0.004,
        "tilt_tolerance":        0.05,
        "yaw_tolerance":         0.08,
        "hold_steps":            3,

        "vel_xy_tolerance":      0.05,
        "vel_z_tolerance":       0.05,
        "require_floor_contact": False,
        "partial_dist_scale":    0.05,

        # [OPT-CUR] 训练容差退火 — 关键设计原则：
        # xy_tol_start >= max_xy_range (30mm) → 保证所有课程级别的成功可达
        # anneal_steps足够长 → 随课程自然退化，不产生cliff效应
        # 精度由 precision_coef 奖励驱动，而非仅靠 xy_tol 收紧
        #
        # v7问题根因：
        #   xy_tol_start=15mm, anneal=80k → level2跳升时xy_range=16mm>xy_tol=14.7mm
        #   agent在level2无法成功 → SR骤降至0 (cliff效应)
        # v8修复：
        #   xy_tol_start=40mm, anneal=500k
        #   level4(100k steps)时 xy_tol=33mm > xy_range=30mm ✅ 全程覆盖
        "xy_tolerance_train_start":    0.040,   # [v8] 15→40mm: 覆盖最大课程级别(30mm)+安全余量
        "xy_tolerance_train_end":      0.005,   # 最终物理要求
        "tilt_tolerance_train_start":  0.12,    # [v8] 恢复更宽松的起始值
        "tilt_tolerance_train_end":    0.05,    # 对齐物理要求(0.050rad)
        "yaw_tolerance_train_start":   0.15,    # [v8] 宽松起始
        "yaw_tolerance_train_end":     0.08,    # 对齐物理要求(0.080rad)
        "xy_tolerance_anneal_steps":   500_000, # [v8] 80k→500k: 避免cliff效应
                                                 # 在100k steps训练时 xy_tol≈33mm (仍覆盖level4)
                                                 # 精度驱动靠precision_coef奖励，不靠快速收紧tol
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
    # 17b. Cruise 段 PID 控制器
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
    # 17c. Cruise 段底层防摆控制器 (SwingDampingController) [v4 新增]
    # ==========================================================================
    # 设计原则: 纯物理阻尼, 独立于 RL, 可单独验证
    # 验证方法: 令 RL 残差=0, 观察 swing_energy 是否单调下降 (有阻尼振荡)
    # RL 在此基础上输出残差修正 (路径规划 + 精细防摆调整)
    "cruise_swing_damping": {
        "kd_vel":         2.5,    # 速度阻尼增益 (主要参数)
                                   # [v6] 3.5→2.5: 降回合理范围
                                   # 物理分析: kd_vel*gain_max=3.5*4.0=14 > 临界阻尼c_cr=9.9
                                   # 过大增益导致自激振荡 (damp_gain满幅1.5-4.0振荡是证据)
        "kp_pos":         0.3,    # 位移恢复增益 (辅助参数)
                                   # [v6] 0.5→0.3: 降低以减少与kd的耦合振荡
        "acc_max":        0.5,    # 单独防摆最大修正量 (m/s^2)
                                   # [v6] 0.6→0.5: 略微降低，防止过冲
        "adaptive":       True,   # 自适应增益: 摆动能量大时增大阻尼
        "energy_ref":     0.05,   # 参考能量 (J), 超过此值增益开始增大
                                   # [v6] 0.02→0.05: 恢复原值，0.02触发过早导致gain常年在最大
        "gain_max_scale": 2.0,    # 自适应增益最大倍数
                                   # [v6] 4.0→2.0: 这是关键修复
                                   # kd_vel*gain_max = 2.5*2.0 = 5.0 < c_cr=9.9 (安全范围)
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
            "instability_penalty": -3.0,  # [FIX2] -5→-3: 缩小reward范围
            "crash_penalty":      -5.0,
        },
    },

    # ==========================================================================
    # 19. Phase 2: Cruise RL
    # ==========================================================================
    "cruise_rl": {
        "obs_dim":            38,
        "action_dim":         2,

        # 初始化由课程管理器动态覆盖, 此为最大值
        "init_xy_range":      0.01,
        "init_z_range":       0.01,
        "init_vel_range":     0.0,

        "max_steps":          400,
        "min_steps_for_success": 5,
        "estimated_full_dist_m": 0.41,
        "z_lock_height":      0.25,

        # [v4 新增] RL 残差加速度上限 (独立于底层防摆控制器)
        # 总 acc_xy = base_acc (防摆) + residual_acc (RL)
        # residual 上限 = acc_max_xy * 0.5, 防止 RL 残差过大覆盖底层防摆
        "residual_acc_max_xy": 0.4,  # m/s^2 (底层防摆 acc_max=0.4, RL 残差同等上限)
        "reward": {
            # PBRS 距离驱动
            "pbrs_coef":          15.0,  # [v6] 10→15: 增强导航信号，平衡减小的障碍物惩罚
            "pbrs_gamma":         0.99,

            # [v6] 障碍物排斥重设计: 从指数排斥改为线性软排斥
            # 原因: 旧指数势 coef*(1/d-1/d0)² 在d=5cm时penalty=60-300, 远超PBRS(0.05/step)
            # 新线性排斥: penalty = coef * max(0, 1-d/d0), d=5cm时penalty≤0.25 (合理)
            # 硬碰撞由 collision_penalty=-5 负责, 软排斥只做导向
            "obs_repulse_coef":   0.5,   # [v6] 3.5→0.5: 线性排斥系数(量级完全不同!)
            "obs_repulse_d0":     0.10,  # [v6] 0.15→0.10: 线性模式感知范围10cm
            "obs_repulse_linear": True,  # [v6] 新增: 启用线性排斥模式

            # [v6] 摆动能量约束 (KE+PE 软上界)
            "swing_energy_thresh":        0.10,   # J [v6] 0.05→0.10: 阈值过严时惩罚信号主导reward,掩盖导航
            "swing_energy_penalty_coef":  2.0,    # [v6] 6.0→2.0: 大幅降低,防止能量惩罚成为主要信号
            # 物理分析: 正常摆动0.1-0.5J时,penalty=2*(excess+0.5*excess²)
            # 在0.2J时: penalty=2*(0.1+0.005)=0.21/step → 合理范围
            # 旧 swing_ke_coef 已废弃, 由 swing_energy_* 替代
            "swing_ke_coef":      0.0,    # [v5 废弃, 保留兼容]

            "z_dev_coef":         1.5,   # [v6] 2.0→1.5: 适度恢复，避免掩盖PBRS
            "vz_penalty_coef":    2.0,   # [v6] 3.0→2.0: 适度恢复

            # [OPT] alive_bonus 0.08→0.02, step_penalty -0.002→-0.01
            # 目标: 净 step 效果为轻微负值 (-0.008/步), 消除存活套利
            "alive_bonus":        0.02,
            "step_penalty":       -0.01,

            "collision_penalty":  -10.0,  # [v9] -8→-10: 进一步遏制冲刺行为
            "success_bonus":      8.0,   # [FIX2] 20→8: VL=150根因, 进一步降低return方差
            "instability_penalty": -3.0,  # [FIX2] -5→-3: 缩小reward范围

            # [OPT-CUR] 渐进成功半径 (由课程管理器动态管理)
            "success_radius_start": 0.12,
            "success_radius_end":   0.10,  # [v4] 0.15→0.10: 测试时使用此值作为严格判定上界
            "success_radius_test":  0.06,   # 独立测试专用: 严格的 6cm 判定半径
            "success_radius_anneal_steps": 9_999_999, # [FIX] 禁用时间驱动, 跟随 dist 课程

            "milestone_radius":   0.15,
            "milestone_bonus":    1.5,  # [FIX2] 3.0→1.5: 配合success_bonus=8

            # [OPT] 新增: 到达奖励 (distance < threshold 时持续正奖励)
            "near_goal_radius":   0.10,
            # [FIX] z 成功容差: z_lock * z_success_tol_frac
            "z_success_tol_frac": 0.15,  # [FIX] 0.10→0.15: z 容差 ±37mm (原 ±25mm)
            "near_goal_bonus":    0.05,
        },
    },

    # ==========================================================================
    # 20. Phase 3: Descent RL
    # ==========================================================================
    "descent_rl": {
        "obs_dim":            29,
        "action_dim":         3,

        # [OPT-INIT] 扩大初始化随机范围, 配合课程学习
        # 训练初期由课程管理器从小值开始, 逐步扩大到这里的最大值
        "init_xy_range":      0.030,   # [OPT] 0.010→0.030
        "init_z_range":       0.005,
        "init_tilt_range":    0.010,   # [OPT] 0.005→0.010
        "init_vel_range":     0.030,   # [OPT] 0.0→0.030

        "max_steps":          300,
        "acc_max_xy":         0.5,
        "acc_max_z":          1.0,
        "vel_max_z":          0.05,

        # [OPT-REW] Reward v3
        "reward": {
            "xy_align_coef":      4.0,
            # [OPT] z_descent_coef 6.0→8.0: 更强 z 下降驱动
            "z_descent_coef":     6.0,   # [v9] 10→6: 降低，防止激进下降激励绳摆
            # [v5] 新增: z上升惩罚 — 阻止agent学到「上升」行为
            # 测试日志显示payload从250mm升到600mm+，是当前最核心问题
            "z_rise_penalty_coef": 3.0,  # [v9] 5→3: 适度降低上升惩罚
            "z_rise_max_penalty":  0.3,  # [v5] 单步最大上升惩罚 (防极端case主导reward)
            # [v5] z_unconditional_frac: 进一步提升无条件下降激励
            "z_unconditional_frac": 0.15,  # [v9] 0.25→0.15: 降低无条件下降激励
            "swing_ke_coef":      3.0,   # [v4] ×3: 稳定优先
            "tilt_coef":          1.5,   # [v4] 0.5→1.5: 稳定优先
            "yaw_coef":           2.0,   # [v9] 0.5→2.0: 强化yaw旋转惩罚
            # [v4] 高斯对准奖励: 在 dtf < 15mm 形成强吸引势阱
            "xy_gauss_sigma":     0.020,   # [v7] 扩大高斯范围
            "xy_gauss_coef":      1.5,     # [v7] 增强高斯奖励
            "z_descent_xy_gate":  0.030,  # [v4.1] 0.010→0.030: 30mm 内才需同步对准+下降
            "near_target_bonus":  1.5,     # [v7] 1.0→1.5
            "precision_coef":     4.0,     # [v7] z到位后精细对准奖励系数
            "success_bonus":      50.0,
            "soft_success_bonus": 8.0,   # [v5] 25→8: 去除超时套利，逼迫真实插入学习
            "step_penalty":       -0.005,   # [OPT] -0.001→-0.005
            "instability_penalty": -3.0,  # [FIX2] -5→-3: 缩小reward范围
            "crash_penalty":      -5.0,
        },
    },

    # ==========================================================================
    # 21. PPO 超参数 [v4 — logstd 硬约束 + 低初始熵]
    # ==========================================================================
    "ppo": {
        "hidden_dim":            256,
        "n_layers":              3,
        "lr_actor":              1e-4,
        "lr_critic":             1e-3,
        "gamma":                 0.99,
        "gae_lambda":            0.95,
        "clip_eps":              0.2,
        "value_loss_coef":       0.5,
        "entropy_coef":          0.01,
        "max_grad_norm":         0.5,
        "n_steps":               512,
        "n_epochs":              4,
        "batch_size":            256,
        "normalize_advantages":  True,
        "use_obs_norm":          True,
        "obs_norm_clip":         10.0,
        "obs_norm_warm_start":   5000,
        "log_std_init":         -0.5,
        # [v4 FIX-LOGSTD] 硬约束: max=0.3 防熵爆炸(原max=1.0导致logstd爬升至+1)
        # min=-2.0 (std≥0.135) 保留足够探索能力
        "log_std_min":          -2.0,
        "log_std_max":           0.3,
        "target_kl":             0.01,
        "log_std_floor_init":   -0.5,
        "log_std_floor_final":  -1.5,
        "log_std_floor_steps":   800_000,
        "entropy_coef_start":         0.02,   # [v6] 0.01→0.02: 更高初始熵鼓励探索
        "entropy_coef_end":           0.005,  # [v6] 0.002→0.005: 最终保留更多探索
        "entropy_coef_anneal_steps":  2_000_000,  # [v6] 1M→2M: 延长退火过程
        "plasticity_reset_interval":  200_000,
    },

    # ==========================================================================
    # 22. SAC 超参数 [OPT-SAC]
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
        # [OPT-SAC] target_entropy_ratio 0.3→0.5: 鼓励更高目标熵
        "target_entropy_ratio": 0.5,
        "critic_grad_clip":     1.0,

        # [HER] Hindsight Experience Replay params (descent SAC 专用)
        "her_k":               4,      # 每条 transition 重标注 k 个目标
        "her_reward_scale":    0.3,    # HER 奖励 = success_bonus * scale

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
    # 24. BC 预训练配置 [OPT-BC]
    # ==========================================================================
    "bc_pretrain": {
        "enabled":           True,
        "n_episodes":        1000,
        # [OPT-BC] n_epochs 200→60: 防止 descent 过拟合导致熵崩塌
        "n_epochs":          60,
        "lr":                3e-4,
        "lr_decay":          0.5,
        "lr_decay_interval": 20,    # [OPT] 50→20: 配合 epoch 减少
        "batch_size":        512,
        "eval_interval":     10,    # [OPT] 20→10
        "eval_episodes":     30,
        # [OPT-BC] patience 8→5: 更早停止, 避免过拟合
        "patience":          5,
        "epsilon_start":     0.3,
        "epsilon_end":       0.0,
        "loss_threshold":    0.30,
        "n_dagger_rounds":   2,
    },

    # ==========================================================================
    # 25. 课程学习配置 [OPT-CUR] 全面重设计
    # ==========================================================================
    "curriculum": {
        "enabled":                    True,   # [OPT] 默认启用

        # 风力课程 [v4 解耦设计]
        # ─────────────────────────────────────────────────────────────────────
        # 风力是独立的鲁棒性维度, 与主任务学习完全解耦
        # 训练分两阶段:
        #   阶段A (主训练): wind_enabled=False, 只学防摆+导航/插入
        #   阶段B (鲁棒微调): 加载阶段A最佳ckpt, wind从0线性增至目标值
        # 触发条件: 主任务 SR >= 0.6 后才进入阶段B
        # 用法: python train_phase.py --phase cruise --wind (仅鲁棒微调时传入)
        "wind_start_frac":            0.0,
        "wind_end_frac":              1.0,    # 鲁棒微调阶段内从0→100%
        "wind_anneal_steps":          200_000, # 鲁棒微调共200k steps线性增加

        # 噪声课程
        "noise_start_scale":          0.0,
        "noise_end_scale":            1.0,
        "noise_anneal_steps":         500_000,

        # ── Cruise 专属课程 ──────────────────────────────────────────────────
        # 障碍物课程 — 解锁条件: dist_frac 到达阈值后才开始引入
        "obstacle_enabled":           True,   # [FIX2] 重启: 配合解锁条件缓慢引入
        # [v7b] 障碍物课程: 每级别给予充足训练时间
        # 实测: 5k→8k steps内n_obs从0快速增至3, agent未能充分学习每个难度
        # 用户反馈: 每个障碍物数量的训练轮数需要更多
        "obstacle_enabled":           True,
        "obstacle_unlock_dist_frac":  0.70,   # [v6] 必须走完70%路程才引入障碍物
        "obstacle_hard_cap_grad_steps": 8000,  # [v9] 3000→8000: 每级需要8000次梯度更新才强制晋级
                                               # 分析: 图中6-7k内完成3级, 每级仅3000steps=约430ep太少
        "obstacle_hard_cap_eps":      6000,   # [v8] 保留ep计数(备用)
        "obstacle_level_warmup_eps":  200,    # [v8] 切换后200ep为暖机期,不参与SR统计
        "obstacle_start_n":           0,
        "obstacle_max_n":             3,
        "perf_window":                80,     # [v8] 50→80: 更大窗口减少旧level数据污染
        "perf_sr_threshold":          0.55,   # [v7b] 0.50→0.55: 更高的晋级门槛确保充分掌握
        "perf_reward_threshold":      -10.0,
        "perf_min_episodes_per_level": 1000,  # [v8] 800→1000: 确保充分学习
        "perf_hard_cap_steps":        999_999_999,

        # [BOOTSTRAP-V] Bootstrapped PBRS (2025)
        # Critic 值函数作 potential: F(s,s') = γ·V(s') - V(s)
        # 在 PPO descent 中叠加到 reward, 无需手工势能设计
        "bootstrapped_pbrs_enabled":  True,
        "bootstrapped_pbrs_coef":     0.5,   # PBRS 叠加系数 (不宜过大)

        # [OPT-CUR] Cruise 距离课程: 训练初期 agent 从近处开始学习
        # start_xy 到 target_xy 的初始偏移从小到大渐进
        "cruise_dist_curriculum":     True,
        "cruise_dist_start_frac":     0.25,
        "cruise_dist_end_frac":       1.0,
        "cruise_dist_anneal_steps":   9_999_999,
        "cruise_dist_sr_threshold":   0.50,   # [v6] 0.40→0.50: dist课程也需要50%SR才推进

        # ── Descent 专属课程 [OPT-CUR] ──────────────────────────────────────
        # 初始化范围课程: 从小偏差开始, 逐步扩大
        "descent_init_curriculum":    True,
        # xy_range: 0.005m → 0.030m
        "descent_init_xy_start":      0.005,
        "descent_init_xy_end":        0.030,
        # vel_range: 0.0 → 0.030 m/s
        "descent_init_vel_start":     0.000,
        "descent_init_vel_end":       0.030,
        # tilt_range: 0.002 → 0.010 rad
        "descent_init_tilt_start":    0.002,
        "descent_init_tilt_end":      0.010,
        # 每个难度级别的 episode 数
        # [OMNIRESET] OmniReset (Weirdlab 2025) — descent 多层次状态分布
        "omnireset_enabled":          True,
        "omnireset_near_goal_prob":   0.30,  # [v6] 0.25→0.30
        "omnireset_near_goal_xy":     0.012, # [v6] 0.008→0.012
        "omnireset_near_goal_z_offset": 0.05,

        "descent_cur_levels":         5,
        "descent_cur_min_eps":        100,   # [v6] 150→100
        "descent_cur_sr_threshold":   0.50,  # [v6] 0.35→0.50
        "descent_cur_hard_cap":       300_000, # [v6] 400k→300k
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