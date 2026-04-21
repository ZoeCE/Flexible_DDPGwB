# ==============================================================================
# config.py — 项目全局配置（合并版：旧版 PPO 课程 + 新版 socket/rebar 任务）
#
# ══════════════════════════════════════════════════════════════════════════════
# 本版合并说明（关键变更）
# ══════════════════════════════════════════════════════════════════════════════
#
# [MERGE-1] 保留新版 XML 任务定义
#   - prefab.shape = "socket"：底部带 4 个方孔的实心方块
#   - target.mode  = "rebar"  ：地面 4 根钢筋桩，有真实碰撞
#   - 新任务成功判定：4 根钢筋必须成功插入 4 个方孔（姿态对齐 + 下降到位）
#
# [MERGE-2] 保留旧版 PPO 全套收敛机制
#   - v4 PPO 超参数（bc 软锚、log_std、target_kl、lr）
#   - v5 性能触发课程学习（perf_sr_threshold, perf_window, hard_cap）
#   - DAgger 预训练配置（bc_pretrain 节）
#
# [MERGE-3] 姿态约束大幅加强（钢筋插入新物理约束）
#   几何分析：
#     方孔 14×14 mm (半宽 7 mm), 钢筋半径 5 mm → 单边径向余量 2 mm
#     孔间距 70 mm, 4 孔对齐 4 杆 → yaw 偏差 > arctan(2/70) ≈ 1.6° = 0.028 rad 即对不齐
#     tilt 偏差 > arctan(2/60) ≈ 1.9° = 0.033 rad（以孔深 60mm 估算）也会卡住
#   因此：
#     - 成功判定 tilt < 0.08 rad（~4.6°）、yaw < 0.06 rad（~3.4°）
#     - 失稳早停 tilt/yaw 阈值收紧
#     - 姿态惩罚系数保留旧版 v3 的线性+指数复合形式（放大）
#
# [MERGE-4] 障碍物碰撞检测精度提升
#   - 旧版仅检查 payload_xy 中心点到障碍物中心距离 < (orad + payload_radius)
#   - 新版 socket 方块非圆形，使用 AABB 保守外接圆 + 真实 MuJoCo contact 双重判定
#   - payload_radius 改为 socket 外接圆半径（sqrt(hx^2+hy^2) + 安全余量）
#
# [MERGE-5] APF 障碍物参数保留旧版 v3 校准曲线，作用半径与新几何协调
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
        # [DELTA] 每控制步最大关节角变化量（rad/step）
        # 插入任务要求更精细动作 → 0.15 → 0.12（约 69°/s，仍足够灵活）
        "dq_max": [0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12],
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
    # ──────────────────────────────────────────────────────────────────────
    # 【重要】n_obstacles 是环境初始化时的"最大障碍物上限"，
    #   - state_dim = 10 + 3*n_obstacles + 26，一旦 env 初始化即固定
    #   - 课程学习运行时 set_curriculum_n_obstacles(n) 可在 [0, n_obstacles] 内调整
    #   - test 节的 n_obstacles 必须 ≤ scene.n_obstacles（否则 state_dim 不匹配）
    # 【state_dim 计算】(n=5): 10 + 15 + 26 = 51
    # ==========================================================================
    "scene": {
        "n_obstacles":        3,            # 课程最高难度上限（state_dim 所依赖）
        "radius_range":       (0.006, 0.015),
        "obstacle_z_center":  0.15,
        "obstacle_halfheight": 0.15,
        "endpoint_z_offset":  0.025,
        "seed":               None,
    },

    # ==========================================================================
    # 5. A* 寻路与 3D 轨迹规划（严格机械臂工作空间约束）
    # ==========================================================================
    # 【工作空间推导】
    #   iiwa14 水平工作半径 ~0.55m (保守值，倒立抓取 + 绳长 0.4m)
    #   减去安全裕量 0.05m → 有效半径 0.50m
    #   起点 (0.3, 0.15) 半径 335mm  ✓
    #   终点 (-0.3, 0.2) 半径 361mm  ✓
    #   
    # 【path_width 推导】
    #   起终点中点 (0, 0.175)，中点半径 175mm
    #   |mid| + path_width/2 + payload_radius ≤ workspace_radius
    #   → path_width ≤ 2×(0.50 - 0.175 - 0.075) = 0.50m
    #   保守取 0.40m（±0.2m 散布）
    # ==========================================================================
    "planning": {
        # [WS-1] 机械臂工作半径（A* 硬约束）
        "workspace_radius":  0.50,

        # payload_radius = socket 外接圆 sqrt(0.05²+0.05²)=0.0707 + 安全余量 → 0.075
        "payload_radius":    0.075,
        "planning_margin":   0.03,
        "planning_grid_res": 0.025,

        # [WS-2] path_width 严格推导：2×(ws_r - mid_r - payload_r) = 2×(0.5-0.175-0.075) = 0.5m
        #        保守取 0.4m（±0.2m 散布），避免 A* 把路径规划到工作空间外
        "path_width":        0.6,

        # [CORR] 走廊方向约束（障碍物与 A* 走廊同侧）
        # perp = (-dy, dx)；对 start(0.3,0.15)→target(-0.3,0.2), perp≈(0,-1)
        #   +1: 障碍物 & 走廊都在 perp 正侧（即 -Y，靠近底座）
        #   -1: 障碍物 & 走廊都在 perp 负侧（即 +Y，远离底座）
        "corridor_side":          +1,
        "corridor_forbid_margin": 0.05,

        "bounds_margin":     0.5,
        "max_expansions":    100000,

        # [Z-1] 巡航高度 = 0.25（payload 中心 z）
        # payload 顶面 z = 0.25 + 0.1 = 0.35 > obstacle 顶 0.30 ✓
        "payload_z_cruise":  0.25,

        # [Z-2] target_z_descent = A* 轨迹末端 payload 中心 z
        # 新钢筋几何：rebar 总高 20mm（从 z=0 立到 z=0.02）
        # target_z_descent 应高于 "触地+完全插入" 的最终 z（0.10），
        # 给 RL 策略留下降空间。取 0.12 表示 payload 底面 0.02m（刚好钢筋顶端）
        "target_z_descent":  0.12,

        "num_descent_steps": 10,          # 下降航点数（多航点 → 下降平滑）
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

        # ── [v3-STABILITY v2] 失稳早停阈值（针对插入任务收紧）─────────────────────
        # 钢筋插入对姿态极度敏感：tilt>0.14 rad 或 yaw>0.04 rad 就无法插入
        # 但训练早期策略未成熟，过严会导致大量提前终止、失去探索机会
        # 折中：grace 期放宽，grace 后逐步收紧
        "instability_check":     True,
        "instability_grace_steps": 50,
        "swing_xy_max":          0.25,      # 摆角阈值保持宽松
        "payload_vel_max":       2.0,
        "payload_tilt_max":      1.0,       # 旧版 1.2 → 1.0（~57°）收紧
        "payload_yaw_max":       1.2,       # 旧版 1.6 → 1.2（~69°）收紧
        "instability_penalty":   -10.0,

        # 单步奖励裁剪
        "reward_clip_min":       -5.0,
        "reward_clip_max":        5.0,
    },

    # ==========================================================================
    # 7. 奖励函数系数（旧版 v3/v4 APF + 插入任务姿态强化）
    #
    # 设计原则：
    #   1. 连续惩罚用负指数（小偏差≈0，大偏差饱和）
    #   2. 障碍物用 APF 排斥势
    #   3. 吊装物姿态（tilt/yaw）惩罚大幅加强（插入任务核心需求）
    #   4. 新增钢筋"对准奖励"：payload 接近正确插入姿势时给予引导
    #   5. 成功奖励大幅提高（+80），拉开与平台期差距
    # ==========================================================================
    # ==========================================================================
    # 7. 奖励函数系数（PPO 课程学习专用架构）
    #
    # 设计目标：为 PPO 课程学习（0 → N 障碍物）提供清晰梯度信号
    #
    # 架构：
    #   【势能 1】终点吸引（potential-based shaping）：
    #     U_goal_xy = k_goal_xy × ||p_xy - target_xy||²
    #     U_goal_z  = k_goal_z × (p_z - target_z)²   [仅 dtf<激活阈值时]
    #     reward += -(U_t - U_{t-1})   # 势能差分式 shaping，γ=0.99 相容
    #     好处：累积和 = U_start - U_end，无需调节每步量级
    #
    #   【势能 2】障碍物排斥（APF）：
    #     U_obs = Σ k × max(0, 1/d_eff - 1/rho_0)²
    #     保留原调好的曲线，仅在 dist_edge < rho_0 激活
    #
    #   【姿态严格控制】（核心）：
    #     tilt/yaw 线性 + 负指数复合惩罚
    #     payload_tilt_penalty_coef=0.3, scale=0.08（强梯度）
    #     线性项 0.5*tilt（小偏差持续引导）
    #
    #   【防摆】（辅助，不是核心）：
    #     payload_angvel_penalty_coef = 0.04（从旧 0.08 减半）
    #     swing_xy 连续指数惩罚
    #
    #   【成功/失败奖励】（终止信号，量级清晰）：
    #     success_bonus  = +100  ← 拉开与失败差距
    #     collision      = -30
    #     crash          = -20
    #     instability    = -15
    #     timeout        = -5
    #     预期 return 差距 > 220
    # ==========================================================================
    "reward": {
        # ── 终止奖励（量级清晰，拉开差距）──────────────────────────────────
        "success_bonus":          100.0,    # 成功插入 + 触地稳定
        "soft_success_bonus":     40.0,     # "软成功"（几乎成功但超时）给部分奖励
        "timeout_penalty":        -5.0,
        "collision_penalty":      -30.0,    # 撞障碍物（致命）
        "out_of_bounds_penalty":  -15.0,
        "crash_penalty":          -20.0,    # payload 坠落
        "instability_penalty":    -15.0,    # 失稳早停

        # ── 势能 1：终点吸引势能（主奖励信号）──────────────────────────────
        # U_goal_xy = k × dtf²，势能差分式 shaping（Ng et al. 1999）
        # 量级：起终点距离 0.6m → U_start = 1.0 × 0.36 = 0.36
        #       总累计 ≈ +0.36（靠近时递减到 0）
        # 放大后：+36（占成功奖励的约 30%）
        "goal_potential_xy_coef":   100.0,  # U_xy = 100 × dtf²（dtf=0.6→U=36）
        "goal_potential_z_coef":    100.0,  # U_z  = 100 × dz²（dz=0.1→U=1）
        "goal_potential_activate_dist":  0.15,  # dtf<15cm 才启用 Z 势能

        # ── 势能 2：障碍物排斥（APF，保留旧调好的曲线）──────────────────────
        "obstacle_rho_0":           0.10,
        "obstacle_d_min":           0.005,
        "obstacle_apf_coef":        3.0e-4,
        "obstacle_apf_max":         1.0,
        "obstacle_penalty_coef":    0.02,
        "obstacle_penalty_scale":   0.05,

        # ── 姿态严格控制（核心，线性+指数复合）──────────────────────────────
        "payload_tilt_penalty_coef":     0.3,
        "payload_tilt_penalty_scale":    0.08,
        "payload_tilt_linear_coef":      0.5,
        "payload_tilt_linear_clip":      1.0,

        "payload_yaw_penalty_coef":      0.4,
        "payload_yaw_penalty_scale":     0.1,
        "payload_yaw_linear_coef":       0.6,
        "payload_yaw_linear_clip":       1.0,

        # ── 防摆（辅助，不是核心）──────────────────────────────────────────
        "payload_angvel_penalty_coef":   0.04,   # 从旧 0.08 减半（非核心）
        "payload_angvel_penalty_scale":  0.5,
        "swing_penalty_coef":     0.10,          # 旧 0.15 微降
        "swing_penalty_scale":    0.03,
        "verticality_penalty_coef": 0.08,        # 旧 0.10 微降
        "verticality_penalty_scale": 0.15,

        # ── 控制平滑（减小抖动）────────────────────────────────────────────
        "velocity_penalty_coef":  0.03,
        "velocity_penalty_scale": 0.3,
        "joint_smooth_penalty_coef":  0.02,
        "joint_smooth_penalty_scale": 0.1,

        # ── 对准精细奖励（平滑二次，插入阶段激活）────────────────────────────
        # 替代原指数悬崖，避免抖动
        # coef × (1 - dtf/activate_dist)² 全程平滑
        "alignment_activate_dist":   0.05,   # dtf<5cm 激活
        "alignment_xy_coef":         3.0,    # dtf=0 时 +3
        "alignment_pose_coef":       3.0,    # pose_err=0 时 +3
        "alignment_pose_activate":   0.10,   # pose_err<0.10 rad 激活

        # ── 抖动抑制（插入阶段内）──────────────────────────────────────────
        "stability_vel_threshold":   0.05,   # <5cm/s 视为静止
        "stability_bonus":           1.5,
        "action_rate_coef":          0.5,    # Δq_t - Δq_{t-1} 惩罚
        "action_rate_scale":         0.03,

        # ── 进展奖励（旧版保留，辅助 shaping）────────────────────────────────
        "progress_coef":          2.0,    # 旧 5.0 → 2.0（势能差分已提供主梯度）
        "progress_clip":          0.1,
        "waypoint_bonus":         1.0,
        "step_penalty":           0.0,

        # ── MuJoCo 真实接触检测开关 ────────────────────────────────────────
        "use_mujoco_contact":        True,

        # ── 关节极限（保留但当前不启用）──────────────────────────────────
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
    # 10. Prefab（负载）几何参数 — 新版 socket 模式
    # ==========================================================================
    "prefab": {
        # shape: "box" | "cylinder" | "composite" | "socket"
        # 插入任务默认 socket（底部带 4 方孔的方块）
        "shape":              "socket",

        # --- box 模式参数（实心方块，备用）---
        "box_half_size":      [0.05, 0.05, 0.1],

        # --- cylinder 模式参数（实心圆柱，备用）---
        "cylinder_radius":    0.04,
        "cylinder_half_height": 0.1,

        # --- composite 模式参数（多 STL 凸分解拼合体，备用）---
        "mesh_prefix":        "hollow_cylinder_convex",
        "mesh_count":         64,

        # --- socket 模式参数（底部带方孔的方块，插入任务专用）---
        "socket_half_size":   [0.05, 0.05, 0.1],    # 外壳半尺寸 [x, y, z]
        "socket_hole_size":   [0.014, 0.014],        # 方孔截面 [x, y] (m)
        "socket_hole_depth":  0.06,                  # 方孔深度 (m)
        "socket_hole_positions": [                   # 孔中心 XY 坐标（4 孔对 4 杆）
            [ 0.035,  0.035],
            [ 0.035, -0.035],
            [-0.035,  0.035],
            [-0.035, -0.035],
        ],

        # --- 通用参数 ---
        "mass":               1.0,
        "lift_site_offset":   0.1,
        "lift_site_spread":   0.05,
    },

    # ==========================================================================
    # 11. Target（目标位置标记）— 新版 rebar 模式
    # ==========================================================================
    "target": {
        # mode: "visual" (纯视觉，无碰撞) | "rebar" (地面钢筋桩，有碰撞)
        "mode":               "rebar",

        # --- visual 模式参数（备用）---
        "shape":              "box",
        "box_half_size":      [0.05, 0.05, 0.1],
        "cylinder_radius":    0.02,
        "cylinder_half_height": 0.1,
        "rgba":               [0.8, 0.0, 0.0, 0.4],

        # --- rebar 模式参数（地面钢筋桩）---
        # 4 根钢筋位置必须与 prefab.socket_hole_positions 一致
        "rebar_positions": [
            [ 0.035,  0.035],
            [ 0.035, -0.035],
            [-0.035,  0.035],
            [-0.035, -0.035],
        ],
        "rebar_radius":       0.003,     # 钢筋半径 5mm
        "rebar_half_height":  0.01,      # 钢筋半高 20mm（总高 40mm）
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
    # 14. PPO Agent 超参数（旧版 v4，已验证收敛）
    # ==========================================================================
    "ppo_agent": {
        "hidden_dim":            256,
        "n_layers":              2,

        "lr_actor":              1e-4,
        "lr_critic":             3e-4,
        "gamma":                 0.99,
        "gae_lambda":            0.95,
        "clip_eps":              0.15,
        "value_loss_coef":       0.5,
        "entropy_coef":          0.003,
        "max_grad_norm":         0.5,

        "n_steps":               2048,
        "n_epochs":              4,
        "batch_size":            256,
        "normalize_advantages":  True,

        "behavior_clone":        True,
        "bc_coef_init":          10.0,
        "bc_coef_final":         0.3,
        "bc_anneal_steps":       1_500_000,
        "bc_loss_type":          "mse",

        "use_obs_norm":          True,
        "obs_norm_clip":         10.0,

        "log_std_init":         -2.5,
        "log_std_min":          -4.0,
        "log_std_max":          -1.5,

        "target_kl":             0.03,
    },

    # ==========================================================================
    # 14b. [v5-PERFORMANCE] 课程学习配置（旧版收敛良好，完整保留）
    # ==========================================================================
    "curriculum": {
        "enabled":                    True,
        "bc_n_obstacles":             0,
        # ── 性能触发参数（主模式）──
        "ramp_mode":                  "performance",
        "perf_window":                30,
        "perf_sr_threshold":          0.6,      # 60% 成功率（比 0.7 宽松，便于 0→1 首次升级）
        "perf_reward_threshold":      80.0,     # 匹配新奖励量级（成功 ≈150~200，失败 ≈-50）
        "perf_min_episodes_per_level": 100,
        "perf_regression_tol":       -30.0,
        "perf_hard_cap_steps":       600_000,
        # ── 时间触发参数（备用）──
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
    # 15. TD3 Agent 超参数（保留）
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
    # ──────────────────────────────────────────────────────────────────────
    # 【约束】test.n_obstacles 必须 ≤ scene.n_obstacles（即 state_dim 上限）
    #   test.py 启动时会把 config["scene"]["n_obstacles"] 设为此值，
    #   这个值仅定义测试时生成的障碍物数量，不改变 state_dim
    # ==========================================================================
    "test": {
        "n_episodes":         20,
        "render":             False,
        "n_obstacles":        3,        # 必须 ≤ scene.n_obstacles = 5
        "obstacle_seed":      6,
        "save_paths":         False,
        "save_paths_dir":     "test_paths",
        "ckpt_path":          None,
        "policy_type":        "ppo",
    },

    # ==========================================================================
    # 18. BC 预训练参数（Phase 1，旧版保留）
    # ==========================================================================
    "bc_pretrain": {
        "n_episodes":   500,
        "n_epochs":     100,
        "lr":           3e-4,
        "batch_size":   256,
    },

    # ==========================================================================
    # 19. 插入任务成功判定（基于 "触地 + 姿态 + XY" 物理约束）
    #
    # 新几何参数（用户设定）：
    #   钢筋：半径 3mm，总高 20mm（从 z=0 立到 z=0.02）
    #   方孔：半宽 7mm，深度 60mm
    #   单边径向余量 = 4mm
    #
    # payload_z → 底面 z → 插入深度 对照（严格）：
    # ──────────────────────────────────────────
    #   payload_z=0.12 → 底面 z=+0.020（刚接触钢筋顶端，0% 插入）
    #   payload_z=0.11 → 底面 z=+0.010（插入 10mm，50% 深度）
    #   payload_z=0.10 → 底面 z=+0.000（触地，钢筋完全嵌入 20mm，100%）
    #   payload_z<0.10 → 物理不可能（底面 < 地面）
    #
    # 成功判定（用户要求"触地稳定即成功"）：
    #   必须同时满足（连续 hold_steps 步）：
    #     (1) payload_z ∈ [success_z_min, success_z_max]  ← 底面接近地面
    #     (2) dtf < xy_tolerance  ← XY 对准
    #     (3) tilt < tilt_tolerance  ← 竖直
    #     (4) yaw < yaw_tolerance  ← 4 孔对 4 杆
    #     (5) vel_xy < vel_xy_tol, |vz| < vel_z_tol  ← 稳定不运动
    #     (6) (可选) MuJoCo 检测到 floor 接触
    #
    # 容差严格推导（单边余量 4mm，分配到 xy/yaw/tilt）：
    #   xy_tolerance × 1 + yaw_tolerance × 49.5mm + tilt_tolerance × 60mm ≤ 4mm 余量
    #   (49.5mm = 对角半径 sqrt(0.035²+0.035²) 是 yaw 偏差导致的最大 XY 位移杠杆)
    #   (60mm = 孔深，tilt 偏差在孔底处的位移杠杆)
    #   任意两项叠加 ≤ 4mm （第三项为零或很小）
    #
    # 实用容差（考虑 rebar 动力学软纠正）：
    #   xy_tolerance   = 0.003 (3mm)  ← 严格，1mm 安全余量
    #   yaw_tolerance  = 0.04 (2.3°)  ← 位移 ~2mm，与 xy 合计 5mm（略超余量，但动力学纠正）
    #   tilt_tolerance = 0.05 (2.9°)  ← 位移 ~3mm（略宽，因插入后会自动纠正）
    # ==========================================================================
    "insertion": {
        # 进入插入阶段阈值（payload 进入此 z 以下时 in_insertion_phase=True）
        "entry_z":               0.16,     # 底面 z=0.06（钢筋顶 4cm 上方）

        # 成功区间 [success_z_min, success_z_max]（payload 中心 z）
        # 目标 = payload 底面触地（底面 z≈0 → payload_z=0.10）
        "target_payload_z":      0.10,     # 最终成功位置（底面触地）
        "success_z_tolerance":   0.015,    # ±15mm（底面 z 在 [-5mm, +25mm]）
        #   实际触发：payload_z ∈ [0.085, 0.115]
        #   0.115 底面 z=15mm，钢筋已插入 5mm（>25%，物理上会产生触地前的接触）
        #   0.085 底面被地面约束住（MuJoCo 不让穿透）

        # 连续 hold_steps 步满足所有条件 → 成功
        "hold_steps":            5,        # 5 步 @ 10Hz = 0.5s 稳定

        # 姿态 & XY 容差（严格按 4mm 径向余量推导，动力学略放宽）
        "xy_tolerance":          0.003,    # 3mm（单边余量 4mm，留 1mm 安全）
        "tilt_tolerance":        0.05,     # 2.9°（rebar 接触后会纠正）
        "yaw_tolerance":         0.04,     # 2.3°（对角半径 49.5mm × 0.04 ≈ 2mm 位移）

        # 速度容差（成功判定需低速平稳）
        "vel_xy_tolerance":      0.05,     # <5cm/s
        "vel_z_tolerance":       0.05,     # <5cm/s（触地后应静止）

        # 是否启用真实 MuJoCo 地面接触检测
        # True: 必须 payload 与 floor 有真实 contact 才算成功（最严格）
        # False: 仅依赖 payload_z 范围判断
        "require_floor_contact": True,

        # 部分奖励的距离尺度
        "partial_dist_scale":    0.05,     # dtf<5cm 给 Tier-3 部分奖励
    },
}