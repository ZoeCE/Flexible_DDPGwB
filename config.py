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
        "n_obstacles":        5,            # 课程最高难度上限（state_dim 所依赖）
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
        # [WS-1] 机械臂工作半径（A* 硬约束，超出此半径的网格点不可通行）
        "workspace_radius":  0.50,

        # payload_radius = socket 外接圆 sqrt(0.05²+0.05²)=0.0707 + 安全余量 → 0.075
        "payload_radius":    0.075,
        "planning_margin":   0.02,
        "planning_grid_res": 0.02,

        # [WS-2] path_width 从 scene 节移到 planning，严格推导为 0.4m
        "path_width":        1.2,

        "bounds_margin":     0.5,
        "max_expansions":    100000,

        # 巡航高度：从 0.25 → 0.30，保证 payload 顶面 (cruise + 0.1) = 0.40 > obstacle 顶 0.30
        "payload_z_cruise":  0.25,

        # [INS] target_z_descent = 插入入口 z (payload 中心)
        # 物理：payload 底面 = target_z_descent - socket_hz = 0.16 - 0.1 = 0.06m
        # 距 rebar 顶 (z=0.04) 有 20mm 缓冲，策略从此开始自主下降
        "target_z_descent":  0.1,

        "num_descent_steps": 15,          # 8→6（下降更紧凑）
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
    "reward": {
        # ── 终止奖励 ──────────────────────────────────────────────────────────
        "success_bonus":          80.0,     # 旧 50 → 80（插入任务难度更大，奖励相应提高）
        "timeout_penalty":        -10.0,
        "collision_penalty":      -20.0,    # 旧 -15 → -20（新 socket 体积更大，碰撞更致命）
        "out_of_bounds_penalty":  -15.0,
        "crash_penalty":          -20.0,    # 旧 -15 → -20

        # ── 稠密进展奖励 ─────────────────────────────────────────────────────
        "progress_coef":          5.0,
        "progress_clip":          0.1,

        # ── 里程碑奖励 ────────────────────────────────────────────────────────
        "waypoint_bonus":         1.0,

        # ── 每步惩罚 ─────────────────────────────────────────────────────────
        "step_penalty":           0.0,

        # ── 负指数连续惩罚 ───────────────────────────────────────────────────

        "velocity_penalty_coef":  0.03,
        "velocity_penalty_scale": 0.3,

        "swing_penalty_coef":     0.15,
        "swing_penalty_scale":    0.03,

        "verticality_penalty_coef": 0.1,
        "verticality_penalty_scale": 0.15,

        "joint_smooth_penalty_coef":  0.02,
        "joint_smooth_penalty_scale": 0.1,

        # ── [MERGE-3] 吊装物姿态惩罚（插入任务大幅强化）────────────────────
        # 旧版 v4 (coef=0.08, scale=0.15) 对于吊运任务够用，
        # 但插入要求 tilt<0.08，必须更强的梯度信号。
        # 采用：线性项 + 负指数 的复合形式，小偏差有线性梯度，大偏差饱和
        "payload_tilt_penalty_coef":     0.3,     # 0.08 → 0.3（指数项加大 4×）
        "payload_tilt_penalty_scale":    0.08,    # 0.15 → 0.08（scale 缩小，让小偏差也被惩罚）
        "payload_tilt_linear_coef":      0.5,     # 恢复线性项，提供持续梯度
        "payload_tilt_linear_clip":      1.0,     # 线性项上限（防止大偏差时爆炸）

        "payload_yaw_penalty_coef":      0.4,     # 0.06 → 0.4（yaw 对 4 孔插入最敏感）
        "payload_yaw_penalty_scale":     0.1,     # 0.3 → 0.1
        "payload_yaw_linear_coef":       0.6,
        "payload_yaw_linear_clip":       1.0,

        "payload_angvel_penalty_coef":   0.08,    # 0.02 → 0.08 加强角速度抑制
        "payload_angvel_penalty_scale":  0.5,

        # ── [v3-APF] 障碍物 APF 排斥势（保留旧版校准曲线）────────────────────
        # 作用半径调整为与新 payload_radius=0.075 协调
        "obstacle_rho_0":           0.10,   # 0.08 → 0.10（payload 变大，警告半径扩大）
        "obstacle_d_min":           0.005,
        "obstacle_apf_coef":        3.0e-4,
        "obstacle_apf_max":         1.0,
        "obstacle_penalty_coef":    0.02,
        "obstacle_penalty_scale":   0.05,

        # ── [MERGE-3 NEW] 钢筋对准引导奖励（插入任务专属）──────────────────
        # 目标：当 payload 接近目标位置时，激励正确的 XY 对齐和姿态
        # 几何考量：新 xy_tolerance=6mm，scale 取其 1.5-2 倍（稍宽，有梯度）
        # 1) 对准距离奖励：payload_xy 到 target_xy 距离 < alignment_activate_dist 时激活
        # 2) 高度下降奖励：payload_z 接近目标插入 z 时给予渐增奖励
        # 3) 成功插入的精确条件见 _compute_reward 中的终点判定
        "alignment_activate_dist":   0.08,   # XY 距离 < 8cm 激活对准引导
        "alignment_xy_coef":         3.0,    # XY 对准奖励系数
        "alignment_xy_scale":        0.01,   # 1cm 尺度（比 xy_tol 稍宽，有梯度）
        "alignment_pose_coef":       4.0,    # 姿态对准奖励
        "alignment_pose_scale":      0.05,   # 姿态尺度（与 tilt/yaw 容差同量级）

        # ── [MERGE-4 NEW] 真实 MuJoCo 接触检测开关 ─────────────────────────────
        # True = 使用 data.contact 检测 prefab_body 与 obstacle/rebar 的真实接触
        # False = 仅使用 payload_xy 距离（旧版行为）
        "use_mujoco_contact":        True,

        # ── 关节极限惩罚（保留但不启用） ──────────────────────────────────────
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
        "perf_sr_threshold":          0.7,
        "perf_reward_threshold":      60.0,
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
        "ppo_max_n_obstacles":        5,
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
        "n_obstacles":        5,        # 必须 ≤ scene.n_obstacles = 5
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
    # 19. 插入任务成功判定参数（基于 "下降到位并保持稳定"）
    #
    # 物理分析（严格数学推导）：
    # ──────────────────────────────────────
    #   rebar 从 z=0 立起到 z=0.04（total 40mm）
    #   payload 底面 z = payload_center_z - socket_hz = payload_z - 0.1
    #
    #   payload_z=0.16 → 底面 z=0.06（rebar 顶上方 20mm，即将接触）
    #   payload_z=0.14 → 底面 z=0.04（刚接触 rebar 顶端）
    #   payload_z=0.13 → 底面 z=0.03（插入深度 10mm，25% 深度）
    #   payload_z=0.12 → 底面 z=0.02（插入深度 20mm，50% 深度）
    #   payload_z=0.11 → 底面 z=0.01（插入深度 30mm，75% 深度）
    #   payload_z=0.10 → 底面 z=0.00（插入深度 40mm，100% 深度，触地）
    #
    # 成功判定策略：
    # ──────────────────────────────────────
    #   1. "进入插入阶段"：payload_z ≤ entry_z = 0.16 且 XY 对准
    #      → 开始累计 hold_counter（不立即成功，允许 rebar 物理接触）
    #   2. "成功插入"：payload_z ≤ success_z = 0.13 （插入深度 ≥ 25%）
    #      AND XY 对准、姿态稳定、速度低  持续 hold_steps 步
    #      → is_success=True，episode 终止 + 全额奖励
    #   3. reached_final 到达但未满足成功判定 → Tier-2/Tier-3 部分奖励
    #
    # XY/姿态 容差（比旧版宽）：
    # ──────────────────────────────────────
    #   由于在插入过程中 rebar 会物理约束 payload 的 XY 和 yaw，
    #   成功判定时容差可以放宽。策略只需要把 payload 下降到指定深度即可。
    #   xy_tolerance:   0.006 (6mm，因为 rebar 会把 payload 对齐到孔位)
    #   tilt_tolerance: 0.10  (5.7°，rebar 插入后会纠正部分倾斜)
    #   yaw_tolerance:  0.05  (2.9°，rebar 会强约束 yaw)
    # ==========================================================================
    "insertion": {
        # 插入阶段触发阈值（payload 中心 z，进入此阶段开始连续判定）
        "entry_z":               0.12,     # 底面 z=0.06，即将接触 rebar
        # 成功判定阈值（payload 中心 z，低于此视为成功插入）
        "success_z":             0.07,     # 底面 z=0.03，插入深度 ≥ 25%
        # 完全插入阈值（给部分奖励用，越深越好）
        "deep_z":                0.11,     # 底面 z=0.01，插入深度 ≥ 75%

        # 连续保持多少步才判定成功（防止一闪而过）
        "hold_steps":            3,        # 5 步 @ 10Hz = 0.5s

        # XY/姿态 容差（成功判定需同时满足）
        # 此容差在 "插入中" 阶段使用，相对宽松（依赖 rebar 物理约束）
        "xy_tolerance":          0.006,    # 6mm（插入前要 XY 对准，但 rebar 会进一步纠正）
        "tilt_tolerance":        0.10,     # 5.7°（允许一定倾斜，插入后会改善）
        "yaw_tolerance":         0.05,     # 2.9°（rebar 强约束 yaw）

        # 速度容差（成功判定需静止或缓慢下降）
        "vel_xy_tolerance":      0.15,     # 插入时 XY 速度需小
        "vel_z_tolerance":       0.30,     # Z 速度（下降中，允许轻微下降）

        # 部分奖励的距离标度
        "partial_dist_scale":    0.05,     # dtf < 5cm 给 Tier-3 部分奖励
    },
}