"""
Shared StabilityMetrics class for both training (train_phase.py) and testing
(test_phase.py).

[v12.6 引入此独立模块] 之前 StabilityMetrics 只在 test_phase.py, 训练时无法访问.
将其提到 stability_metrics.py 可以让 train_phase.py 也用 (输出到 wandb).

用户痛点 (v12.4 提出):
  "成功率已经无法衡量 RL 对 base controller 的性能提升"
解决:
  在每个训练 episode 末输出细粒度指标到 wandb, 区分"成功" vs "稳定 + 成功".
"""

import numpy as np


class StabilityMetrics:
    """
    细粒度 RL 训练评估指标 (学界标准).

    用法:
      stab = StabilityMetrics()
      for step in episode:
          obs = env.step(...)
          stab.update_step(obs, config, env=env, rl_action=rl_act)
      summary = stab.summary()  # dict, 供 wandb.log 用
      log_line = stab.print_line()  # 控制台单行总结

    指标:
      A. 摆动质量 (Anti-Sway Quality)
         - avg/max/p95/rms_angle: 摆角统计 (deg)
         - avg/max/p95/rms_ke_mJ: 摆动能量 (mJ)
         - integral_ke_mJs: ∫swing_KE dt (mJ·s, 全过程累积摆动)

      B. 绳索状态 (Cable Dynamics)
         - cable_ke_peak/avg/integral: 绳段动能 (m²/s²)

      C. payload 运动 (Motion Smoothness)
         - pl_vel_peak/rms: payload xy 速度 (m/s)

      D. EE 控制 (Control Effort)
         - avg/max_acc: EE 加速度 (m/s²)
         - rl_action_mag_mean/peak: RL 残差大小 (RL 介入程度)

    文献依据:
      - Shen et al. 2011 (Springer): PV/RMS 是 anti-sway 标准指标
      - Olesen 2026 (arXiv:2602.05895): integral 摆动能量评估
      - FLARE 2025 (arXiv:2508.09797): RL 防摆 evaluation
    """
    def __init__(self):
        self.reset_episode()

    def reset_episode(self):
        # 原有曲线
        self.swing_ke_curve   = []   # 单位 mJ
        self.acc_curve        = []   # EE 加速度 m/s²
        self.angle_curve      = []   # payload 摆角 deg
        self._prev_ee_pos     = None
        # [v12.6] 新增曲线
        self.cable_ke_curve   = []   # 绳索段动能总和 (单位 m²/s²)
        self.pl_vel_curve     = []   # payload xy 速度 m/s
        self.rl_action_mag    = []   # RL 残差 norm
        self._dt              = 0.1

    def update_step(self, obs, config, dt=0.1, env=None, rl_action=None):
        """每 env.step() 后调用. obs 是 env._get_obs() 输出."""
        self._dt = float(dt)
        try:
            rope_L = float(config.get("controller", {}).get("L", 0.5))
            mass   = float(config.get("prefab", {}).get("mass", 1.0))
            pl_vxy = np.array([obs[6], obs[7]], np.float64)
            ee_vxy = np.array([obs[2], obs[3]], np.float64)
            # 摆动相对动能 (mJ)
            ke_J   = 0.5 * mass * float(np.dot(pl_vxy - ee_vxy, pl_vxy - ee_vxy))
            pl_xy  = np.array([obs[4], obs[5]], np.float64)
            ee_xy  = np.array([obs[0], obs[1]], np.float64)
            sin_th = float(np.linalg.norm(pl_xy - ee_xy)) / max(rope_L, 0.01)
            angle_deg = float(np.degrees(np.arcsin(min(sin_th, 1.0))))
            self.swing_ke_curve.append(ke_J * 1000)
            self.angle_curve.append(angle_deg)
            # EE 加速度 (有限差分)
            ee_pos = np.array([obs[0], obs[1]], np.float64)
            if self._prev_ee_pos is not None:
                acc_mag = float(np.linalg.norm(
                    (ee_pos - self._prev_ee_pos) / max(dt, 1e-6)))
                self.acc_curve.append(acc_mag)
            self._prev_ee_pos = ee_pos.copy()
            # payload 速度
            self.pl_vel_curve.append(float(np.linalg.norm(pl_vxy)))
            # 绳索段动能 (来自 env 直接读)
            if env is not None and hasattr(env, '_get_cable_energy'):
                try:
                    self.cable_ke_curve.append(float(env._get_cable_energy()))
                except Exception:
                    pass
            # RL 残差大小
            if rl_action is not None:
                a = np.asarray(rl_action, np.float64).ravel()
                self.rl_action_mag.append(float(np.linalg.norm(a)))
        except Exception:
            pass

    def summary(self):
        """返回 wandb-friendly dict (单一 episode 的总结)."""
        if not self.swing_ke_curve:
            return {}
        ke  = np.array(self.swing_ke_curve)
        ang = np.array(self.angle_curve)
        acc = np.array(self.acc_curve) if self.acc_curve else np.zeros(1)
        d = {
            "avg_ke_mJ":   float(np.mean(ke)),
            "max_ke_mJ":   float(np.max(ke)),
            "p95_ke_mJ":   float(np.percentile(ke, 95)),
            "rms_ke_mJ":   float(np.sqrt(np.mean(ke ** 2))),
            "integral_ke_mJs": float(np.sum(ke) * self._dt),
            "avg_angle":   float(np.mean(ang)),
            "max_angle":   float(np.max(ang)),
            "p95_angle":   float(np.percentile(ang, 95)),
            "rms_angle":   float(np.sqrt(np.mean(ang ** 2))),
            "avg_acc":     float(np.mean(acc)),
            "max_acc":     float(np.max(acc)),
        }
        if self.pl_vel_curve:
            pv = np.array(self.pl_vel_curve)
            d["pl_vel_peak"] = float(np.max(pv))
            d["pl_vel_rms"]  = float(np.sqrt(np.mean(pv ** 2)))
        if self.cable_ke_curve:
            ck = np.array(self.cable_ke_curve)
            d["cable_ke_peak"]      = float(np.max(ck))
            d["cable_ke_avg"]       = float(np.mean(ck))
            d["cable_ke_integral"]  = float(np.sum(ck) * self._dt)
        if self.rl_action_mag:
            am = np.array(self.rl_action_mag)
            d["rl_action_mag_mean"] = float(np.mean(am))
            d["rl_action_mag_peak"] = float(np.max(am))
        return d

    def summary_for_wandb(self, phase, prefix="stab"):
        """返回带 wandb namespace 前缀的 dict, 直接喂给 wandb.log."""
        s = self.summary()
        return {f"{prefix}/{phase}/{k}": v for k, v in s.items()}

    def print_line(self):
        if not self.swing_ke_curve:
            return ""
        ke  = np.array(self.swing_ke_curve)
        ang = np.array(self.angle_curve)
        acc = np.array(self.acc_curve) if self.acc_curve else np.zeros(1)
        line = (f"KE avg/max:{np.mean(ke):.1f}/{np.max(ke):.1f}mJ "
                f"θ avg/max:{np.mean(ang):.1f}°/{np.max(ang):.1f}° "
                f"acc avg/max:{np.mean(acc):.2f}/{np.max(acc):.2f}m/s²")
        if self.pl_vel_curve:
            pv = np.array(self.pl_vel_curve)
            line += f" plV peak:{np.max(pv):.2f}m/s"
        if self.cable_ke_curve:
            ck = np.array(self.cable_ke_curve)
            line += f" cKE peak:{np.max(ck):.3f}"
        if self.rl_action_mag:
            am = np.array(self.rl_action_mag)
            line += f" rlA:{np.mean(am):.3f}"
        return line