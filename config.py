# ==============================================================================
# 修改日志（对照上一版 config.py 的问题）：
#
# [CONFIG-1] "reward" 节新增 "success_bonus": 15.0
#            旧版 step() 中硬编码了 reward += 15.0（成功时），新版通过配置暴露。
#
# [CONFIG-2] "reward" 节新增 "timeout_penalty": -5.0
#            旧版 step() 中硬编码了 reward -= 5.0（超时时），新版通过配置暴露。
#
# 其余所有键值与旧版完全一致，未做任何修改。
# ==============================================================================
import numpy as np
 
DEFAULT_CONFIG = {
    # ==========================================
    # 1. 仿真与控制 (Simulation & Control)
    # ==========================================
    "sim": {
        "physics_dt":      0.002,   # 物理引擎时间步长 (500 Hz)
        "control_freq_hz": 10,      # 控制频率 (10 Hz)，每控制步执行 50 次物理步
        "max_steps":       200,     # 回合最大步数
        "render":          False,   # 是否开启 GUI 渲染
    },
 
    # ==========================================
    # 2. 空间与动作 (Space & Action)
    # ==========================================
    "space": {
        "action_dim":        6,                         # [ax, ay, az, a_roll, a_pitch, a_yaw]
        # 各维度上限
        "action_space_high": [0.5, 0.5, 0.5, 0.5, 2.0, 2.0],
    },
 
    # ==========================================
    # 3. 任务与随机化初始状态 (Task & Randomization)
    # ==========================================
    "task": {
        "start_pos_mocap":    [0.3, 0.2, 1.0],         # 动捕点初始平移位置
        "start_quat_mocap":   [1.0, 0.0, 0.0, 0.0],   # 动捕点初始姿态四元数 (w,x,y,z)
        "default_start_xy":   [0.3, 0.2],              # 负载初始 XY 中心（加噪声前）
        "default_target_xy":  [-0.3, 0.2],             # 目标 XY 中心（子类中不加噪声）
        "init_position_range": 0.01,                   # 初始 XY 位置均匀噪声范围 ±0.08
        "init_velocity_scale": 0.15,                   # 初始速度扰动尺度（当前保留未用）
    },
 
    # ==========================================
    # 4. 环境场景生成 (Scene & Obstacles)
    # 注意：键名为 "scene"，env 中 cfg_scene = self.config["scene"]
    # ==========================================
    "scene": {
        "n_obstacles":       3,                        # 障碍物数量
        "radius_range":      (0.001, 0.003),           # 障碍物半径范围 [r_min, r_max]
        "path_width":        0.02,                     # 障碍物横向分布宽度（在连线两侧）
        "obstacle_z_center": 0.25,                     # 障碍物 Z 轴中心高度
        "obstacle_halfheight": 0.2,                    # 障碍物半高（圆柱半高）
        "endpoint_z_offset": 0.025,                    # 起/终点标记球相对障碍物顶部的偏移
        "seed":              None,                     # 障碍物随机种子（None=每次不同）
    },
 
    # ==========================================
    # 5. A* 寻路与 3D 轨迹规划 (Planning & Geometry)
    # ==========================================
    "planning": {
        "payload_radius":    0.06,    # 负载几何半径（用于障碍物膨胀防撞）
        "planning_margin":   0.05,    # A* 安全边距（在 payload_radius 基础上额外留白）
        "planning_grid_res": 0.02,    # A* 栅格分辨率（单位：米）
        "bounds_margin":     0.05,     # 寻路地图超出首尾点的边界余量
        "max_expansions":    100000,  # A* 最大扩展节点数（防死循环）
 
        "payload_z_cruise":   0.2,   # 负载平移阶段巡航高度（单位：米）
        "target_z_descent":   0.12,   # 终点正上方垂直下潜的最低高度
        "num_descent_steps":  6,      # Z 轴垂直下降段的离散点数量
    },
 
    # ==========================================
    # 6. Step 逻辑判定 (Step Logic)
    # ==========================================
    "step_logic": {
        "look_ahead_dist":    0.1,    # 切换到下一个 Waypoint 的视距阈值（远端航点）
        # 注：尾部 ≤2 个航点时使用硬编码 0.06，以防止过早触发 reached_final
        "out_of_bounds_dist": 2.0,    # 偏离目标超过此距离视为出界并终止
        "crash_z_threshold":  0.12,   # 被判定为砸地坠毁的 Z 轴高度阈值
        "crash_vz_threshold": -0.1,   # 被判定为砸地坠毁的 Z 轴下降速度阈值（负数为下降）
    },
 
    # ==========================================
    # 7. 奖励函数系数 (Reward Shaping)
    # ==========================================
    "reward": {
        "progress_coef":        50.0,   # 靠近当前航点的势能进展奖励系数
        "collision_penalty":    -8.0,   # 撞击障碍物惩罚（负数）
        "out_of_bounds_penalty": -10.0, # 出界惩罚（负数）
        "crash_penalty":        -10.0,  # 砸地/摔机惩罚（负数）
        "action_smooth_penalty": -0.02, # 动作 L2 正则惩罚系数（负数，乘以动作平方和）
        "velocity_penalty_coef":  0.1,  # 速度过快的正则化惩罚系数
        "step_penalty":          -0.02, # 每步生存惩罚（鼓励尽快完成任务）
        # [CONFIG-1 新增] 成功降落的一次性奖励（对应旧版硬编码的 reward += 15.0）
        "success_bonus":         15.0,
        # [CONFIG-2 新增] 超时未成功的惩罚（对应旧版硬编码的 reward -= 5.0）
        "timeout_penalty":       -5.0,
    },
 
    # ==========================================
    # 8. 系统延迟与噪声 (Latency & Noise)
    # ==========================================
    "noise": {
        "latency_steps":    1,    # 动作延迟步数（1 = 延迟 1 个控制周期）
        "force_noise_level": 0.1, # 力输出噪声水平（当前保留未注入，与旧版一致）
    },
 
    # ==========================================
    # 9. 重置参数 (Reset & Init)
    # ==========================================
    "reset": {
        # xyc: 机械臂初始关节角（7 个 revolute 关节，单位：rad）
        # 由 IK 求解器自动计算：末端在 prefab 上方、垂直朝下
        "init_qpos_arm": [
            -0.71972016, 0.29057466, -1.10685585,
            1.53851657, 2.87907443, 1.73055121, -1.85194252
        ],
        # xyc: prefab（负载）初始 free joint 位姿 [x, y, z, qw, qx, qy, qz]
        "init_qpos_prefab": [0.3, 0.2, 0.1, 1.0, 0.0, 0.0, 0.0],
 
        "warmup_steps": 50,    # 物理引擎预热步数（让绳索自然垂落稳定）
        "mocap_init_z": 1.0,   # 动捕点初始强制 Z 高度（单位：米）

        # IK 求解参数：reset 时自动求解机械臂关节角，使末端在 prefab 正上方垂直朝下
        "ik_enabled": True,              # 是否启用自动 IK（False 则使用 init_qpos_arm）
        # NOTE: ik_height_above_prefab 应与 rope.total_length 一致
        #       (= rope.num_segments * rope.segment_length)
        "ik_height_above_prefab": 0.2,   # 末端在 prefab 上方的高度 (m)
        "ik_target_quat": [0.0, 1.0, 0.0, 0.0],  # 末端目标姿态四元数 (wxyz)，绕X轴180°=朝下
        "ik_max_iter": 5000,             # IK 最大迭代次数
        "ik_tol_pos": 1e-4,              # 位置收敛容差 (m)
        "ik_tol_rot": 1e-3,              # 姿态收敛容差 (rad)
    },

    # ==========================================
    # 10. Rope Generation
    # ==========================================
    # ==========================================
    # 10. Prefab (payload) geometry
    # ==========================================
    "prefab": {
        "shape": "box",                      # "box" or "cylinder"
        # box params: half-extents [x, y, z]
        "box_half_size": [0.05, 0.05, 0.1],
        "cylinder_radius": 0.05,             # only used if shape="cylinder"
        "cylinder_half_height": 0.1,         # only used if shape="cylinder"
        "mass": 1.0,
        # lift site offset from prefab center (z = top of shape)
        "lift_site_offset": 0.1,             # z-offset for lift sites
        "lift_site_spread": 0.05,            # xy-offset for lift site corners
    },

    # ==========================================
    # 11. Target (visual goal marker, no collision)
    # ==========================================
    "target": {
        "shape": "box",                      # "box" or "cylinder" (matches prefab)
        # box params
        "box_half_size": [0.05, 0.05, 0.1],
        "cylinder_radius": 0.05,
        "cylinder_half_height": 0.1,
        "rgba": [0.8, 0.0, 0.0, 0.4],       # semi-transparent red
    },

    # ==========================================
    # 12. Rope Generation
    # ==========================================
    "rope": {
        "num_segments":   20,      # capsule segments per rope
        "segment_length": 0.02,    # length of each segment (m), total = num * length
        "damping":        0.02,    # ball joint damping
        "capsule_radius": 0.004,   # visual/collision radius (m)
        "segment_mass":   0.01,    # mass per segment (kg)
        # plate geometry (hook_attachment body)
        "plate_half_size": [0.05, 0.05, 0.01],  # box half-extents [x, y, z]
        "plate_mass":      0.1,                   # plate mass (kg)
        "hook_offset":     0.05,                   # hook site offset from plate center (m)
    },
}