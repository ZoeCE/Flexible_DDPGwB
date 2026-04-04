#!/usr/bin/env python3
"""
用 MuJoCo IK 求解：让 iiwa14 末端 (attachment_site) 位于 prefab 正上方，且末端垂直朝下。

输出可直接粘贴到 config.py 的 init_qpos_arm 中。
"""

import mujoco
import numpy as np
from pathlib import Path

# ── 参数 ──
PREFAB_POS = np.array([0.2, 0.3, 0.5])   # prefab 初始位置
HEIGHT_ABOVE = 0.2                         # 末端在 prefab 上方多少米
TARGET_POS = PREFAB_POS + np.array([0, 0, HEIGHT_ABOVE])  # 目标位置

# 末端垂直朝下：z 轴指向 -Z 方向
# 对应旋转矩阵列向量排列为 [x_axis | y_axis | z_axis]
# 选择: x→+X, y→+Y, z→-Z  （右手系需要调整符号）
# x→+X, y→-Y, z→-Z 是右手系 (绕X旋转180°)
TARGET_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz: 绕X轴旋转180°

# ── 加载模型 ──
xml_path = Path(__file__).resolve().parent / "demo_fourCable_withSteel_withSensor_cylinder.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
assert site_id >= 0, "找不到 attachment_site"

# ── IK 参数 ──
N_JOINTS = 7
MAX_ITER = 5000
STEP_POS = 0.5
STEP_ROT = 0.3
TOL_POS = 1e-4
TOL_ROT = 1e-3

# 初始猜测（当前 config 值）
q0 = np.array([1.10805046, 0.35488624, -2.97538264,
               0.11512695, 2.40048625, 2.3193137, 0.64687807])
data.qpos[:N_JOINTS] = q0.copy()
# 设置 prefab free joint 使其不干扰
data.qpos[-7:] = np.concatenate([PREFAB_POS, [1, 0, 0, 0]])
mujoco.mj_forward(model, data)

print(f"目标位置:   {TARGET_POS}")
print(f"目标四元数: {TARGET_QUAT}  (末端朝下)")
print(f"初始末端位置: {data.site_xpos[site_id]}")
print()


def quat_error(q_target, q_current_mat):
    """计算四元数误差（返回角轴形式的3维误差向量）"""
    # 从 3x3 旋转矩阵恢复四元数
    q_cur = np.zeros(4)
    mujoco.mju_mat2Quat(q_cur, q_current_mat.flatten())
    # 计算误差四元数: q_err = q_target * q_cur^{-1}
    q_cur_inv = q_cur.copy()
    q_cur_inv[1:] *= -1  # 共轭 = 逆（单位四元数）
    q_err = np.zeros(4)
    mujoco.mju_mulQuat(q_err, q_target, q_cur_inv)
    # 确保标量部分为正（短路径）
    if q_err[0] < 0:
        q_err *= -1
    # 角轴: 2 * arctan2(|v|, w) * v/|v|  ≈ 2*v (小角度)
    return 2.0 * q_err[1:]


# ── 雅可比 IK 迭代 ──
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))

for i in range(MAX_ITER):
    mujoco.mj_forward(model, data)

    # 位置误差
    pos_err = TARGET_POS - data.site_xpos[site_id]
    # 姿态误差
    rot_err = quat_error(TARGET_QUAT, data.site_xmat[site_id].reshape(3, 3))

    pos_norm = np.linalg.norm(pos_err)
    rot_norm = np.linalg.norm(rot_err)

    if pos_norm < TOL_POS and rot_norm < TOL_ROT:
        print(f"收敛! 迭代 {i} 次")
        break

    if i % 500 == 0:
        print(f"iter {i:4d}  pos_err={pos_norm:.6f}  rot_err={rot_norm:.6f}")

    # 计算末端雅可比（只取前 N_JOINTS 列）
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    Jp = jacp[:, :N_JOINTS]
    Jr = jacr[:, :N_JOINTS]

    # 关节增量
    dq = STEP_POS * Jp.T @ pos_err + STEP_ROT * Jr.T @ rot_err
    data.qpos[:N_JOINTS] += dq

    # 关节限位裁剪
    for j in range(N_JOINTS):
        jnt_id = j  # iiwa14 前7个关节按顺序排列
        lo = model.jnt_range[jnt_id, 0]
        hi = model.jnt_range[jnt_id, 1]
        if lo < hi:  # 有限位
            data.qpos[j] = np.clip(data.qpos[j], lo, hi)
else:
    print(f"未完全收敛 (iter={MAX_ITER})  pos_err={pos_norm:.6f}  rot_err={rot_norm:.6f}")

# ── 输出结果 ──
mujoco.mj_forward(model, data)
final_pos = data.site_xpos[site_id].copy()
final_mat = data.site_xmat[site_id].reshape(3, 3)
final_quat = np.zeros(4)
mujoco.mju_mat2Quat(final_quat, final_mat.flatten())

print()
print("=" * 60)
print(f"末端位置:   {final_pos}")
print(f"末端四元数: {final_quat}")
print(f"末端 Z 轴:  {final_mat[:, 2]}  (应接近 [0, 0, -1])")
print()
print("关节角 (rad):")
q_result = data.qpos[:N_JOINTS]
print(f"  {list(q_result)}")
print()
print("可粘贴到 config.py 的格式:")
print(f'        "init_qpos_arm": [')
print(f"            {q_result[0]:.8f}, {q_result[1]:.8f}, {q_result[2]:.8f},")
print(f"            {q_result[3]:.8f}, {q_result[4]:.8f}, {q_result[5]:.8f}, {q_result[6]:.8f}")
print(f"        ],")
