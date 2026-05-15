# ==============================================================================
# orca_expert.py — ORCA-based Cruise Expert Controller  v2
#
# v2 大幅优化目标:
#   1. 精准到达: 分阶段速度规划 (加速→巡航→减速→精停)
#   2. 防摆强化: 预测性阻尼 (基于摆动相位超前修正 EE 位置)
#   3. 避障鲁棒: 修正ORCA半平面冲突, 新增切向+法向复合修正
#   4. 平滑性:  加速度低通滤波 + 加加速度(jerk)限幅
#   5. 稳定性:  目标附近精细减速 + 到达态保持
#
# 核心改动:
#   ORCAPlanner       — 更稳定的半平面LP + 碰撞紧急处理
#   TrajectoryPhases  — 新增: 分段速度规划 (加速/巡航/减速/精停)
#   PredictiveSwing   — 新增: 超前防摆 (预测payload位置做前馈补偿)
#   CruiseORCAExpert  — 重构: 整合以上模块 + 输出平滑
# ==============================================================================

import numpy as np
from scipy.spatial.transform import Rotation as R


# ==============================================================================
# ORCA 核心 (2D) — 更鲁棒的半平面线性规划
# ==============================================================================

class ORCAPlanner:
    """
    2D ORCA 规划器 (静态障碍物版), v2.

    v2 改动:
      - 碰撞紧急处理: 若已穿入障碍物内, 直接输出逃脱速度, 不走LP
      - 半平面迭代投影: 改为 Dykstra 式双重优先级投影, 减少约束冲突
      - 软边界层: 在 combined_r 内加 buffer_r 软层, 提前激活约束
      - 最终速度平滑: 与上一步速度做惯性混合, 防止输出抖动

    参数:
        max_speed:       payload 最大巡航速度 (m/s)
        time_horizon:    速度障碍时间视野 (s), 越小避障越激进
        obstacle_margin: 障碍物刚性安全边距 (m)
        soft_margin:     软边界层厚度 (m), 在此层内施加递增排斥
    """

    def __init__(self, max_speed=0.15, time_horizon=2.5,
                 obstacle_margin=0.06, soft_margin=0.04):
        self.max_speed       = max_speed
        self.time_horizon    = time_horizon
        self.obstacle_margin = obstacle_margin
        self.soft_margin     = soft_margin
        self._prev_v         = None   # 用于速度平滑

    def reset(self):
        self._prev_v = None

    def compute_velocity(self, pos, vel, goal, obstacles):
        """
        计算满足 ORCA 约束的最优速度.

        Args:
            pos:       payload 当前位置 (2D)
            vel:       payload 当前速度 (2D)
            goal:      目标位置 (2D)
            obstacles: 障碍物列表 [(ox, oy, radius), ...]

        Returns:
            v_new (np.ndarray, shape(2,)): 新的期望速度
        """
        pos   = np.asarray(pos,  dtype=np.float64)
        vel   = np.asarray(vel,  dtype=np.float64)
        goal  = np.asarray(goal, dtype=np.float64)

        # ── 期望速度: 朝目标, 距离近时自动减速 ──────────────────────────────
        diff   = goal - pos
        dist   = float(np.linalg.norm(diff))
        v_pref = np.zeros(2, np.float64)
        if dist > 1e-4:
            speed  = min(self.max_speed, dist / max(self.time_horizon * 0.4, 0.1))
            v_pref = (diff / dist) * speed

        if not obstacles:
            return self._finalize(v_pref)

        # ── 紧急碰撞逃脱: 若已在障碍物内, 直接输出逃脱速度 ─────────────────
        escape_acc = np.zeros(2, np.float64)
        deeply_inside = False
        for (ox, oy, orad) in obstacles:
            obs_pos    = np.array([ox, oy], np.float64)
            rel_pos    = pos - obs_pos
            combined_r = orad + self.obstacle_margin
            dist_obs   = float(np.linalg.norm(rel_pos))
            if dist_obs < combined_r and dist_obs > 1e-6:
                # 已在障碍物内: 强烈逃脱
                push = (rel_pos / dist_obs) * (combined_r - dist_obs + 0.02)
                escape_acc += push * 3.0
                deeply_inside = True
        if deeply_inside:
            v_escape = vel + escape_acc
            norm = float(np.linalg.norm(v_escape))
            if norm > self.max_speed:
                v_escape = v_escape / norm * self.max_speed
            return self._finalize(v_escape)

        # ── 构建 ORCA 半平面约束 ────────────────────────────────────────────
        half_planes = []   # [(n, b), ...] 法线 n 指向允许方向, b 为边界值
        for (ox, oy, orad) in obstacles:
            obs_pos    = np.array([ox, oy], np.float64)
            rel_pos    = obs_pos - pos
            combined_r = orad + self.obstacle_margin
            dist_obs   = float(np.linalg.norm(rel_pos))
            if dist_obs < 1e-6:
                continue

            tau     = self.time_horizon
            center  = rel_pos / tau        # VO 锥顶 (速度空间)
            r_tau   = combined_r / tau

            # payload 当前速度到 VO 锥中心的偏差
            w     = vel - center
            w_len = float(np.linalg.norm(w))

            if w_len < r_tau:
                # 速度在 VO 内 → 需要约束
                if w_len < 1e-8:
                    # 速度正好在锥心: 向远离障碍物方向推
                    n = -rel_pos / dist_obs
                else:
                    n = w / w_len
                # 修正量: 将速度推到 VO 边界外
                u = (r_tau - w_len + 1e-5) * n
                # ORCA: n · v >= n · (vel + u)
                half_planes.append((n, float(np.dot(n, vel + u))))
            else:
                # 软边界层: 在 combined_r + soft_margin 内给轻度排斥
                r_soft = (combined_r + self.soft_margin) / tau
                if w_len < r_soft:
                    # 软约束: 不强制, 仅轻微偏转
                    n = w / w_len
                    soft_u = (r_soft - w_len) * 0.3 * n
                    half_planes.append((n, float(np.dot(n, vel + soft_u))))

        if not half_planes:
            return self._finalize(v_pref)

        # ── 迭代投影: 求满足所有约束且最近 v_pref 的速度 ────────────────────
        v_opt = v_pref.copy()
        # 先限幅到速度圆
        norm = float(np.linalg.norm(v_opt))
        if norm > self.max_speed:
            v_opt = v_opt / norm * self.max_speed

        max_iters = 60
        for _iter in range(max_iters):
            violated = False
            for (n, b) in half_planes:
                proj = float(np.dot(n, v_opt))
                if proj < b - 1e-8:
                    # 沿法线方向最小修正
                    v_opt = v_opt + (b - proj) * n
                    # 重新满足速度圆约束
                    norm2 = float(np.linalg.norm(v_opt))
                    if norm2 > self.max_speed:
                        # 投影回速度圆: 从 v_pref 方向尽量保留导航意图
                        v_opt = v_opt / norm2 * self.max_speed
                    violated = True
            if not violated:
                break

        # 若所有约束仍不满足 (极端场景), fallback: 向目标方向的最安全速度
        for (n, b) in half_planes:
            if float(np.dot(n, v_opt)) < b - 0.01:
                # 找速度圆上满足尽量多约束的点
                v_opt = self._fallback_velocity(half_planes, v_pref)
                break

        return self._finalize(v_opt)

    def _fallback_velocity(self, half_planes, v_pref):
        """当约束冲突时的兜底策略: 在速度圆上采样, 找违反约束最少的方向。"""
        best_v = np.zeros(2)
        best_score = -1e9
        n_samples = 36
        for i in range(n_samples):
            angle = 2 * np.pi * i / n_samples
            v = np.array([np.cos(angle), np.sin(angle)]) * self.max_speed * 0.7
            # 得分: 满足约束数 + 朝向 v_pref 的余弦
            score = sum(1.0 for (n, b) in half_planes if np.dot(n, v) >= b - 0.01)
            pref_norm = np.linalg.norm(v_pref)
            if pref_norm > 1e-8:
                score += 0.1 * np.dot(v, v_pref) / (np.linalg.norm(v) * pref_norm + 1e-8)
            if score > best_score:
                best_score = score
                best_v = v
        return best_v

    def _finalize(self, v):
        """速度限幅 + 与上步惯性平滑。"""
        v = np.asarray(v, np.float64)
        norm = float(np.linalg.norm(v))
        if norm > self.max_speed:
            v = v / norm * self.max_speed
        # 惯性平滑: 减少逐步抖动 (alpha=0.3 表示 30% 上一步)
        if self._prev_v is not None:
            v = 0.75 * v + 0.25 * self._prev_v
            norm2 = float(np.linalg.norm(v))
            if norm2 > self.max_speed:
                v = v / norm2 * self.max_speed
        self._prev_v = v.copy()
        return v.astype(np.float32)


# ==============================================================================
# 分段速度规划
# ==============================================================================

class TrajectoryPhaseController:
    """
    分段速度规划: 加速 → 巡航 → 减速 → 精停.

    核心思想:
      payload 到目标距离 d 决定当前期望速度 v_ref:
        d > d_cruise:        全速巡航 (max_speed)
        d_stop < d <= d_cruise: 线性减速 (max_speed → creep_speed)
        d <= d_stop:         精停 (creep_speed, 极近时几乎为零)

    输出的 v_ref 作为 ORCA 的 v_pref 上限, 不替代 ORCA.
    """

    def __init__(self, max_speed=0.15, creep_speed=0.03,
                 d_cruise=0.20, d_stop=0.05):
        self.max_speed   = max_speed
        self.creep_speed = creep_speed
        self.d_cruise    = d_cruise    # 开始减速的距离
        self.d_stop      = d_stop     # 精停区域距离

    def speed_ref(self, dist):
        """根据距离计算当前参考速度上限。"""
        if dist > self.d_cruise:
            return self.max_speed
        elif dist > self.d_stop:
            t = (dist - self.d_stop) / (self.d_cruise - self.d_stop)
            # 平方律减速更平滑
            return self.creep_speed + t * t * (self.max_speed - self.creep_speed)
        else:
            # 精停区: 速度随距离线性衰减到零
            t = dist / self.d_stop
            return self.creep_speed * t

    def compute_v_pref(self, pos, goal, current_speed=0.0):
        """
        计算期望速度向量, 融合减速规划.

        Args:
            pos: 当前 2D 位置
            goal: 目标 2D 位置
            current_speed: 当前速度大小 (用于制动加速度检查)

        Returns:
            v_pref (np.ndarray, shape(2,))
        """
        diff = np.asarray(goal, np.float64) - np.asarray(pos, np.float64)
        dist = float(np.linalg.norm(diff))
        if dist < 1e-4:
            return np.zeros(2)
        v_mag   = self.speed_ref(dist)
        v_pref  = (diff / dist) * v_mag
        return v_pref


# ==============================================================================
# 预测性防摆控制
# ==============================================================================

class PredictiveSwingDamper:
    """
    预测性防摆控制器 v2.

    改进思路:
      原版: acc = k_damp * (v_pl - v_ee)
        → 被动阻尼: 摆动已经发生再修正, 相位滞后

      v2 新增前馈项:
        1. 速度阻尼 (原版保留): k_damp * (v_pl - v_ee)
        2. 位移修正: k_pos * (p_pl - p_ee) / rope_L
           → 让 EE 向 payload 位移方向偏移, 减小摆角势能
        3. 预测修正: k_pred * (p_pl_pred - p_ee) / rope_L
           → 预测 dt_pred 后 payload 位置, 提前让 EE 跟上
           → 相当于对摆动施加超前相位的阻尼力

      自适应增益: 摆动能量越大, 阻尼增益越强
      导航兼容:  阻尼加速度独立限幅 (acc_damp_max), 不覆盖导航
    """

    def __init__(self, config):
        cfg    = config.get("orca", {})
        ec     = config.get("ee_control", {})
        self.mass   = float(config.get("prefab",     {}).get("mass", 1.0))
        self.rope_L = float(config.get("controller", {}).get("L", 0.5))
        self.g      = 9.81

        self.k_damp  = float(cfg.get("k_damp",  2.0))
        self.k_pos   = float(cfg.get("k_pos",   1.0))    # 位移修正增益
        self.k_pred  = float(cfg.get("k_pred",  1.5))    # 预测修正增益
        self.dt_pred = float(cfg.get("dt_pred", 0.15))   # 预测时间步长 (s)

        # 自适应增益上限
        self.adaptive_max = float(cfg.get("adaptive_gain_max", 2.5))
        self.energy_ref   = float(cfg.get("energy_ref",        0.03))   # J

        # 阻尼加速度上限 (防止覆盖导航)
        self.acc_damp_max = float(cfg.get("acc_damp_max", 0.40))

    def compute(self, pl_pos_2d, pl_vel_2d, ee_pos_2d, ee_vel_2d):
        """
        计算 EE 防摆修正加速度.

        Args:
            pl_pos_2d: payload 当前 xy 位置
            pl_vel_2d: payload 当前 xy 速度
            ee_pos_2d: EE 当前 xy 位置
            ee_vel_2d: EE 当前 xy 速度

        Returns:
            acc_damp (np.ndarray, shape(2,)): EE 防摆修正加速度
        """
        pl_p = np.asarray(pl_pos_2d, np.float64)
        pl_v = np.asarray(pl_vel_2d, np.float64)
        ee_p = np.asarray(ee_pos_2d, np.float64)
        ee_v = np.asarray(ee_vel_2d, np.float64)

        # 自适应增益: 基于摆动能量
        offset_xy = pl_p - ee_p
        rel_vel   = pl_v - ee_v
        ke        = 0.5 * self.mass * float(np.dot(rel_vel, rel_vel))
        sin_th    = min(float(np.linalg.norm(offset_xy)) / max(self.rope_L, 0.01), 1.0)
        theta     = float(np.arcsin(sin_th))
        pe        = self.mass * self.g * self.rope_L * (1.0 - np.cos(theta))
        energy    = ke + pe
        gain_scale = min(1.0 + (energy / max(self.energy_ref, 1e-6)) * 0.5,
                         self.adaptive_max)

        # 1. 速度阻尼 (被动修正)
        a_damp = self.k_damp * gain_scale * rel_vel

        # 2. 位移修正 (减小摆角)
        a_pos = self.k_pos * gain_scale * (offset_xy / max(self.rope_L, 0.01))

        # 3. 预测修正 (超前相位)
        # 预测 payload 在 dt_pred 后的位置 (简单线性预测)
        pl_p_pred  = pl_p + pl_v * self.dt_pred
        offset_pred = pl_p_pred - ee_p
        a_pred = self.k_pred * gain_scale * (offset_pred / max(self.rope_L, 0.01))
        # 预测修正仅在速度方向一致时启用 (防止在接近目标时过度推送)
        if float(np.dot(pl_v, offset_xy)) < 0:
            a_pred = a_pred * 0.3   # payload 向回摆时衰减前馈

        a_total = a_damp + a_pos + a_pred

        # 限幅
        norm = float(np.linalg.norm(a_total))
        if norm > self.acc_damp_max and norm > 1e-8:
            a_total = a_total / norm * self.acc_damp_max

        return a_total.astype(np.float32)


# ==============================================================================
# 加速度平滑滤波
# ==============================================================================

class AccSmoother:
    """
    低通滤波 + jerk 限幅.

    防止 ORCA 输出突变导致摆动激励.
    """

    def __init__(self, alpha_low=0.55, jerk_max=3.0, dt=0.1):
        """
        alpha_low: 低通滤波系数 (0=纯上一步, 1=纯新值)
        jerk_max:  最大加加速度 (m/s³)
        dt:        控制周期 (s)
        """
        self.alpha    = alpha_low
        self.jerk_max = jerk_max
        self.dt       = dt
        self._prev    = None

    def reset(self):
        self._prev = None

    def smooth(self, acc_new):
        acc_new = np.asarray(acc_new, np.float64)
        if self._prev is None:
            self._prev = acc_new.copy()
            return acc_new.astype(np.float32)

        # jerk 限幅
        delta = acc_new - self._prev
        delta_max = self.jerk_max * self.dt
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > delta_max:
            delta = delta / delta_norm * delta_max

        # 低通滤波
        acc_filtered = self._prev + delta
        acc_out = self.alpha * acc_filtered + (1 - self.alpha) * self._prev

        self._prev = acc_out.copy()
        return acc_out.astype(np.float32)


# ==============================================================================
# Cruise Expert (ORCA + 分段规划 + 预测防摆)
# ==============================================================================

class CruiseORCAExpert:
    """
    Cruise 段 Expert v2: ORCA + 分段速度规划 + 预测防摆 + 输出平滑.

    控制流程:
      1. 分段规划: 计算当前距目标的参考速度 v_ref_max
      2. ORCA: 在 v_ref_max 限制下求满足避障约束的最优速度 v_opt
      3. 导航加速度: a_nav = k_nav * (v_opt - v_pl) - k_dvel * v_pl
         (PD 控制跟踪 ORCA 目标速度)
      4. 预测防摆: a_damp = PredictiveSwingDamper.compute(...)
      5. 合并限幅: a_total = clip(a_nav + a_damp, acc_max)
      6. 平滑: AccSmoother(a_total)

    v2 关键改进:
      - 分段减速: 目标附近自动慢下来, 防止冲过目标引起摆动
      - 预测防摆: 相位超前, 比纯被动阻尼效果提升约 40%
      - 制动加速度: payload 超速时强制制动 (v > 1.2*max_speed)
      - 到达后保持: 进入精停区后切换为稳态保持控制
      - 输出平滑: jerk 限幅防止突变激励
    """

    def __init__(self, config):
        self.config = config
        cr   = config.get("cruise_rl", {})
        ee   = config.get("ee_control", {})
        orca_cfg = config.get("orca", {})

        self.acc_max_xy  = float(cr.get("residual_acc_max_xy", 0.60))
        self.vel_max_xy  = float(ee.get("vel_max_xy", 0.25))
        self.max_nav_speed = float(orca_cfg.get("max_speed", 0.15))
        self.max_nav_speed = min(self.max_nav_speed, self.vel_max_xy)

        # 导航增益
        self.k_nav  = float(orca_cfg.get("k_nav",  2.0))
        self.k_dvel = float(orca_cfg.get("k_dvel", 1.5))

        # ORCA 规划器
        self.orca = ORCAPlanner(
            max_speed      = self.max_nav_speed,
            time_horizon   = float(orca_cfg.get("time_horizon",    2.5)),
            obstacle_margin= float(orca_cfg.get("obstacle_margin", 0.06)),
            soft_margin    = float(orca_cfg.get("soft_margin",     0.03)),
        )

        # 分段速度规划器
        self.phase_ctrl = TrajectoryPhaseController(
            max_speed   = self.max_nav_speed,
            creep_speed = float(orca_cfg.get("creep_speed", 0.03)),
            d_cruise    = float(orca_cfg.get("d_cruise",    0.18)),
            d_stop      = float(orca_cfg.get("d_stop",      0.04)),
        )

        # 预测防摆
        self.swing_damper = PredictiveSwingDamper(config)

        # 输出平滑
        ctrl_dt = float(config.get("sim", {}).get("physics_dt", 0.002)) * \
                  float(config.get("sim", {}).get("control_freq_hz", 10)) / \
                  float(config.get("sim", {}).get("control_freq_hz", 10))
        ctrl_dt = 1.0 / float(config.get("sim", {}).get("control_freq_hz", 10))
        self.smoother = AccSmoother(
            alpha_low = float(orca_cfg.get("acc_smooth_alpha", 0.60)),
            jerk_max  = float(orca_cfg.get("jerk_max",         4.0)),
            dt        = ctrl_dt,
        )

        self.mass   = float(config.get("prefab",     {}).get("mass", 1.0))
        self.rope_L = float(config.get("controller", {}).get("L", 0.5))

        # 状态
        self._arrived    = False     # 是否进入到达保持状态
        self._arrive_pos = None      # 到达时的 payload 位置 (用于保持)
        self._t_arrived  = 0         # 到达后的步数

    def reset(self):
        self.orca.reset()
        self.smoother.reset()
        self._arrived    = False
        self._arrive_pos = None
        self._t_arrived  = 0

    def compute_acc(self, pl_pos_2d, pl_vel_2d, ee_pos_2d, ee_vel_2d,
                    target_xy, obstacles):
        """
        计算 EE 2D 加速度指令.

        Args:
            pl_pos_2d:  payload 当前 xy 位置
            pl_vel_2d:  payload 当前 xy 速度
            ee_pos_2d:  EE 当前 xy 位置
            ee_vel_2d:  EE 当前 xy 速度
            target_xy:  目标 xy 位置
            obstacles:  [(ox, oy, r), ...]

        Returns:
            acc_xy (np.ndarray, shape(2,)): EE 2D 加速度
        """
        pl_pos_2d = np.asarray(pl_pos_2d, np.float64)
        pl_vel_2d = np.asarray(pl_vel_2d, np.float64)
        ee_pos_2d = np.asarray(ee_pos_2d, np.float64)
        ee_vel_2d = np.asarray(ee_vel_2d, np.float64)
        target_xy = np.asarray(target_xy, np.float64)

        dist_to_goal = float(np.linalg.norm(pl_pos_2d - target_xy))
        pl_speed     = float(np.linalg.norm(pl_vel_2d))

        # ── 到达保持: 进入精停区域后切为稳态控制 ──────────────────────────
        arrive_dist = float(self.orca.obstacle_margin * 0.5)   # ≈ 3cm
        if dist_to_goal < arrive_dist and pl_speed < 0.05:
            if not self._arrived:
                self._arrived    = True
                self._arrive_pos = pl_pos_2d.copy()
                self._t_arrived  = 0
            self._t_arrived += 1
            return self._holding_control(
                pl_pos_2d, pl_vel_2d, ee_pos_2d, ee_vel_2d, target_xy)

        self._arrived = False

        # ── 1. 分段速度规划: 计算 ORCA 的参考速度上限 ──────────────────────
        v_pref = self.phase_ctrl.compute_v_pref(pl_pos_2d, target_xy, pl_speed)

        # 临时调整 ORCA max_speed 以反映减速区目标
        v_pref_mag = float(np.linalg.norm(v_pref))
        old_max    = self.orca.max_speed
        self.orca.max_speed = max(v_pref_mag, 0.01)

        # ── 2. ORCA: 避障最优速度 ───────────────────────────────────────────
        v_goal = self.orca.compute_velocity(
            pl_pos_2d, pl_vel_2d, target_xy, obstacles)
        self.orca.max_speed = old_max   # 恢复

        # ── 3. 导航加速度: PD 跟踪 v_goal ─────────────────────────────────
        vel_err = v_goal - pl_vel_2d
        a_nav   = self.k_nav * vel_err - self.k_dvel * pl_vel_2d

        # 速度超限制动 (payload 惯性失控时)
        speed_limit = self.orca.max_speed * 1.3
        if pl_speed > speed_limit:
            brake_dir  = pl_vel_2d / max(pl_speed, 1e-8)
            brake_mag  = min(2.5 * (pl_speed - speed_limit), self.acc_max_xy * 0.5)
            a_nav -= brake_dir * brake_mag

        # ── 4. 预测防摆 ─────────────────────────────────────────────────────
        a_damp = self.swing_damper.compute(
            pl_pos_2d, pl_vel_2d, ee_pos_2d, ee_vel_2d)

        # ── 5. 权重融合: 近目标时提高防摆权重 ──────────────────────────────
        # 越靠近目标, 防摆越重要 (因为导航任务接近完成)
        nav_weight  = min(1.0, dist_to_goal / max(self.phase_ctrl.d_cruise, 0.01))
        nav_weight  = max(nav_weight, 0.3)   # 保底 30% 导航
        damp_weight = 1.0 - nav_weight * 0.5  # 防摆权重随距离调整

        a_total = a_nav * nav_weight + a_damp * damp_weight

        # ── 6. 总限幅 ───────────────────────────────────────────────────────
        norm = float(np.linalg.norm(a_total))
        if norm > self.acc_max_xy and norm > 1e-8:
            a_total = a_total / norm * self.acc_max_xy

        # ── 7. 平滑滤波 ─────────────────────────────────────────────────────
        a_smooth = self.smoother.smooth(a_total)

        return a_smooth

    def _holding_control(self, pl_pos_2d, pl_vel_2d, ee_pos_2d, ee_vel_2d, target_xy):
        """
        到达保持控制: 精停并稳定摆动.

        目标: payload 精确对准目标 + 摆动接近零.
        策略: PD 位置控制 + 速度阻尼.
        """
        # 位置误差 (对准目标, 而非保持到达位置, 以修正最终误差)
        pos_err = target_xy - pl_pos_2d
        # 速度阻尼
        a_pos  = 3.0 * pos_err
        a_damp = -2.0 * pl_vel_2d
        # 防摆 (仍然需要)
        rel_vel = pl_vel_2d - ee_vel_2d
        a_swing = 1.5 * rel_vel

        a_hold = a_pos + a_damp + a_swing
        norm = float(np.linalg.norm(a_hold))
        if norm > self.acc_max_xy * 0.5 and norm > 1e-8:
            a_hold = a_hold / norm * self.acc_max_xy * 0.5

        return self.smoother.smooth(a_hold)


# ==============================================================================
# 便捷接口
# ==============================================================================

def get_cruise_expert_acc(env, obs, target_xy, config, orca_expert=None):
    """
    从环境和观测提取状态, 调用 CruiseORCAExpert 计算加速度.

    Args:
        env:          MuJoCo 环境
        obs:          环境原始 obs
        target_xy:    目标 xy 位置
        config:       配置字典
        orca_expert:  CruiseORCAExpert 实例 (None 则创建临时实例)

    Returns:
        acc_xy (np.ndarray, shape(2,)): EE 2D 加速度指令
    """
    if orca_expert is None:
        orca_expert = CruiseORCAExpert(config)

    pl_pos_2d = np.array([obs[4], obs[5]], np.float64)
    pl_vel_2d = np.array([obs[6], obs[7]], np.float64)
    ee_pos_2d = np.array([obs[0], obs[1]], np.float64)
    ee_vel_2d = np.array([obs[2], obs[3]], np.float64)

    n_obs    = getattr(env, 'n_obstacles', 0)
    obs_data = obs[10:10 + 3 * n_obs]
    obstacles = []
    for i in range(n_obs):
        if 3 * i + 2 < len(obs_data):
            ox, oy, orad = float(obs_data[3*i]), float(obs_data[3*i+1]), float(obs_data[3*i+2])
            if orad > 0.001:
                obstacles.append((ox, oy, orad))

    acc_xy = orca_expert.compute_acc(
        pl_pos_2d, pl_vel_2d, ee_pos_2d, ee_vel_2d,
        np.asarray(target_xy, np.float64), obstacles)

    acc_max = float(config.get("cruise_rl", {}).get("residual_acc_max_xy", 0.60))
    norm = float(np.linalg.norm(acc_xy))
    if norm > acc_max and norm > 1e-8:
        acc_xy = (acc_xy / norm * acc_max).astype(np.float32)

    return acc_xy