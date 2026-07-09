"""Traditional descent experts for paper comparison experiments.

These controllers are deliberately separate from ``JointSpaceExpert`` so the
PID expert paired with the residual-RL policy stays unchanged.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Type

import numpy as np
from scipy.spatial.transform import Rotation as R


OBS_EE_X, OBS_EE_Y = 0, 1
OBS_EE_VX, OBS_EE_VY = 2, 3
OBS_PL_X, OBS_PL_Y = 4, 5
OBS_PL_VX, OBS_PL_VY = 6, 7
OBS_EE_Z, OBS_EE_VZ = 19, 20
OBS_PL_Z, OBS_PL_VZ = 21, 22
OBS_EE_YAW, OBS_EE_YAW_V = 27, 28
OBS_TILT, OBS_YAW = 29, 30


DEFAULT_TRADITIONAL_EXPERTS = {
    "common": {
        "target_yaw": 0.0,
        "anchor_alpha": 0.08,
        "vel_max_xy": 0.14,
        "vel_max_z": 0.020,
        "vel_max_yaw": 0.25,
        "terminal_vel_scale": 1.00,
        "terminal_z_margin": 0.060,
        "terminal_vel_min_xy": 0.000,
        "attitude_z_gate": True,
        "attitude_gate_min": 0.18,
        "yaw_gate_scale": 0.85,
        "terminal_align_enabled": True,
        "terminal_align_z_margin": 0.060,
        "terminal_align_xy": 0.075,
        "terminal_align_kp_xy": 2.00,
        "terminal_align_kd_xy": 0.65,
        "terminal_align_blend": 0.90,
        "terminal_yaw_boost_enabled": True,
        "terminal_yaw_boost_z_margin": 0.080,
        "terminal_yaw_kp_scale": 1.80,
        "terminal_yaw_vel_scale": 2.00,
    },
    "pid": {
        "kp_xy": 2.00,
        "ki_xy": 0.55,
        "kd_payload_xy": 0.35,
        "k_swing_xy": 0.25,
        "swing_fade_xy": 0.080,
        "swing_fade_z_margin": 0.050,
        "k_rel_vel_xy": 0.10,
        "integral_limit": 0.018,
        "integral_active_xy": 0.120,
        "z_kp": 0.65,
        "z_kd": 0.35,
        "z_gate": 0.030,
        "z_full_gate": 0.010,
        "z_trickle_gate": 0.080,
        "z_min_gate_frac": 0.20,
        "yaw_kp": 0.65,
        "yaw_kd": 0.10,
    },
    "damped_pd": {
        "kp_xy": 2.05,
        "ki_xy": 0.0,
        "kd_payload_xy": 0.55,
        "k_swing_xy": 0.28,
        "swing_fade_xy": 0.080,
        "swing_fade_z_margin": 0.050,
        "k_rel_vel_xy": 0.12,
        "z_kp": 0.50,
        "z_kd": 0.50,
        "z_gate": 0.025,
        "z_full_gate": 0.008,
        "z_trickle_gate": 0.065,
        "z_min_gate_frac": 0.12,
        "yaw_kp": 0.45,
        "yaw_kd": 0.14,
    },
    "mpc": {
        "N": 14,
        "model_gravity": 9.81,
        "model_damping_xy": 1.15,
        "candidate_scales": [0.0, 0.35, 0.60, 0.85, 1.00],
        "lateral_scales": [-0.35, 0.0, 0.35],
        "kp_feedback_xy": 1.85,
        "kd_feedback_xy": 0.60,
        "k_swing_feedback_xy": 0.35,
        "q_stage_xy": 180.0,
        "q_terminal_xy": 850.0,
        "q_payload_vel_xy": 24.0,
        "q_swing_xy": 38.0,
        "r_cmd_vel_xy": 6.0,
        "cmd_smoothing_alpha": 0.45,
        "cmd_rate_limit_xy": 0.050,
        "feedback_override_xy": 0.060,
        "descent_ready_xy": 0.030,
        "descent_ready_payload_vel": 0.120,
        "recovery_xy": 0.055,
        "recovery_payload_z_margin": 0.030,
        "z_kp": 0.70,
        "z_kd": 0.35,
        "z_gate": 0.035,
        "z_full_gate": 0.012,
        "z_trickle_gate": 0.080,
        "z_min_gate_frac": 0.15,
        "yaw_kp": 0.45,
        "yaw_kd": 0.12,
    },
}


def _deep_update(base, override):
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def traditional_cfg(config, name):
    cfg = _deep_update(
        DEFAULT_TRADITIONAL_EXPERTS,
        config.get("traditional_experts", {}))
    common = copy.deepcopy(cfg.get("common", {}))
    common.update(cfg.get(name, {}))
    return common


def _wrap_pi(x):
    return (float(x) + math.pi) % (2.0 * math.pi) - math.pi


def _clip_norm(vec, max_norm):
    vec = np.asarray(vec, dtype=np.float64).copy()
    max_norm = float(max_norm)
    norm = float(np.linalg.norm(vec))
    if max_norm > 0.0 and norm > max_norm and norm > 1e-12:
        vec *= max_norm / norm
    return vec


def _unit(vec):
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return None
    return vec / norm


def _gate_fraction(xy_err, full_gate, hard_gate, trickle_gate, min_frac):
    xy_err = float(xy_err)
    full_gate = max(float(full_gate), 1e-9)
    hard_gate = max(float(hard_gate), full_gate + 1e-9)
    trickle_gate = max(float(trickle_gate), hard_gate)
    min_frac = float(np.clip(min_frac, 0.0, 1.0))
    if xy_err <= full_gate:
        return 1.0
    if xy_err <= hard_gate:
        return 1.0 - (xy_err - full_gate) / (hard_gate - full_gate)
    if xy_err <= trickle_gate:
        taper = 1.0 - (xy_err - hard_gate) / max(trickle_gate - hard_gate, 1e-9)
        return min_frac * float(np.clip(taper, 0.0, 1.0))
    return 0.0


def _soft_abs_gate(value, full_gate, zero_gate, min_frac=0.0):
    value = abs(float(value))
    full_gate = max(float(full_gate), 1e-9)
    zero_gate = max(float(zero_gate), full_gate + 1e-9)
    min_frac = float(np.clip(min_frac, 0.0, 1.0))
    if value <= full_gate:
        return 1.0
    if value >= zero_gate:
        return min_frac
    frac = 1.0 - (value - full_gate) / (zero_gate - full_gate)
    return min_frac + (1.0 - min_frac) * float(np.clip(frac, 0.0, 1.0))


class _VirtualEEDescentExpert:
    name = "base"

    def __init__(self, config, ik_solver):
        self.config = config
        self.ik_solver = ik_solver
        self.cfg = traditional_cfg(config, self.name)
        self.dt = float(config.get("controller", {}).get(
            "dt", config.get("ee_control", {}).get("integrator_dt", 0.1)))
        sp = config["space"]
        self.dq_max = np.asarray(sp.get("dq_max", [0.12] * 7), dtype=np.float64)
        self.q_low = np.asarray(sp["action_space_low"], dtype=np.float64)
        self.q_high = np.asarray(sp["action_space_high"], dtype=np.float64)
        self._ee_pos = np.zeros(3, dtype=np.float64)
        self._ee_vel = np.zeros(3, dtype=np.float64)
        self._ee_yaw = 0.0
        self._ee_yaw_vel = 0.0
        self._last_q = None
        self._target_xy = np.asarray(
            config.get("task", {}).get("default_target_xy", [0.0, 0.0]),
            dtype=np.float64)
        self._integral_xy = np.zeros(2, dtype=np.float64)
        self.last_info = {}
        self._rebar_radius_xy = self._compute_rebar_radius_xy()

    def _compute_rebar_radius_xy(self):
        points = self.config.get("target", {}).get(
            "rebar_positions",
            self.config.get("prefab", {}).get("socket_hole_positions", []))
        radii = []
        for point in points or []:
            try:
                radii.append(float(np.linalg.norm(np.asarray(point[:2]))))
            except Exception:
                continue
        return max(radii) if radii else 0.0

    def reset(self, env_obs, init_q, env=None):
        if env is not None:
            self._ee_pos = env._get_ee_pos().astype(np.float64)
            self._target_xy = np.asarray(env.target_pos[:2], dtype=np.float64)
            pl_z = float(env.data.body("prefab").xpos[2])
        else:
            self._ee_pos = np.array([
                float(env_obs[OBS_EE_X]),
                float(env_obs[OBS_EE_Y]),
                float(env_obs[OBS_EE_Z]),
            ], dtype=np.float64)
            pl_z = float(env_obs[OBS_PL_Z])
        rope_L = float(self.config.get("controller", {}).get("L", 0.5))
        self._ee_pos[2] = max(
            float(self._ee_pos[2]),
            pl_z + rope_L)
        self._ee_vel = np.zeros(3, dtype=np.float64)
        if env is not None:
            try:
                self._ee_yaw = float(R.from_matrix(
                    env._get_ee_mat()).as_euler("xyz")[2])
            except Exception:
                self._ee_yaw = float(env_obs[OBS_EE_YAW])
        else:
            self._ee_yaw = float(env_obs[OBS_EE_YAW])
        self._ee_yaw_vel = 0.0
        self._last_q = np.asarray(init_q, dtype=np.float64).copy()
        self._integral_xy[:] = 0.0
        self.last_info = {}

    def set_path(self, path):
        return None

    def _state(self, obs, env=None):
        target_xy = (np.asarray(env.target_pos[:2], dtype=np.float64)
                     if env is not None and hasattr(env, "target_pos")
                     else self._target_xy)
        if env is not None and hasattr(env, "data") and hasattr(env, "model"):
            try:
                pl_pos = env.data.body("prefab").xpos.copy()
                dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
                pl_vel = env.data.qvel[dof_idx:dof_idx + 3].copy()
                ee_pos = env._get_ee_pos().astype(np.float64)
                pl_mat = env.data.body("prefab").xmat.reshape(3, 3)
                pl_euler = R.from_matrix(pl_mat).as_euler("xyz")
                pl_yaw = float(pl_euler[2])
                pl_yaw_vel = 0.0
                try:
                    if len(env.data.qvel) >= dof_idx + 6:
                        pl_yaw_vel = float(env.data.qvel[dof_idx + 5])
                except Exception:
                    pl_yaw_vel = 0.0
                ee_mat = env._get_ee_mat()
                ee_yaw = float(R.from_matrix(ee_mat).as_euler("xyz")[2])
                if obs is None:
                    ee_vel_default = np.zeros(3, dtype=np.float64)
                    ee_yaw_vel = 0.0
                else:
                    ee_vel_default = [obs[OBS_EE_VX], obs[OBS_EE_VY], obs[OBS_EE_VZ]]
                    ee_yaw_vel = float(obs[OBS_EE_YAW_V])
                ee_vel = np.asarray(getattr(
                    env, "_ee_vel_cache", ee_vel_default), dtype=np.float64)
                return {
                    "target_xy": target_xy,
                    "pl_xy": pl_pos[:2].astype(np.float64),
                    "pl_vel_xy": pl_vel[:2].astype(np.float64),
                    "pl_z": float(pl_pos[2]),
                    "pl_vz": float(pl_vel[2]),
                    "pl_yaw": pl_yaw,
                    "pl_yaw_vel": pl_yaw_vel,
                    "tilt": float(np.sqrt(pl_euler[0] ** 2 + pl_euler[1] ** 2)),
                    "ee_pos": ee_pos,
                    "ee_vel": ee_vel,
                    "ee_yaw": ee_yaw,
                    "ee_yaw_vel": ee_yaw_vel,
                }
            except Exception:
                pass
        return {
            "target_xy": target_xy,
            "pl_xy": np.array([obs[OBS_PL_X], obs[OBS_PL_Y]], dtype=np.float64),
            "pl_vel_xy": np.array([obs[OBS_PL_VX], obs[OBS_PL_VY]], dtype=np.float64),
            "pl_z": float(obs[OBS_PL_Z]),
            "pl_vz": float(obs[OBS_PL_VZ]),
            "pl_yaw": float(obs[OBS_YAW]),
            "pl_yaw_vel": 0.0,
            "tilt": float(obs[OBS_TILT]),
            "ee_pos": np.array([obs[OBS_EE_X], obs[OBS_EE_Y], obs[OBS_EE_Z]],
                               dtype=np.float64),
            "ee_vel": np.array([obs[OBS_EE_VX], obs[OBS_EE_VY], obs[OBS_EE_VZ]],
                               dtype=np.float64),
            "ee_yaw": float(obs[OBS_EE_YAW]),
            "ee_yaw_vel": float(obs[OBS_EE_YAW_V]),
        }

    def _anchor(self, real_ee):
        alpha = float(np.clip(self.cfg.get("anchor_alpha", 0.08), 0.0, 1.0))
        self._ee_pos = (1.0 - alpha) * self._ee_pos + alpha * real_ee
        target_pz = float(self.config.get("insertion", {}).get(
            "target_payload_z", 0.10))
        min_ee_z = target_pz + float(self.config.get("controller", {}).get("L", 0.5))
        if self._ee_pos[2] < min_ee_z:
            self._ee_pos[2] = min_ee_z
            self._ee_vel[2] = max(0.0, self._ee_vel[2])

    def _effective_yaw_gate(self):
        explicit = self.cfg.get("yaw_descent_gate", None)
        if explicit is not None:
            return max(float(explicit), 1e-6)
        ins = self.config.get("insertion", {})
        paper = self.config.get("paper_eval", {})
        yaw_tol = float(paper.get(
            "yaw_tolerance", ins.get("yaw_tolerance", math.pi)))
        xy_tol = float(paper.get(
            "xy_tolerance",
            ins.get("physical_rebar_xy_tolerance",
                    ins.get("xy_tolerance", 0.02))))
        if self._rebar_radius_xy <= 1e-6:
            return yaw_tol
        rebar_gate = float(self.cfg.get("yaw_gate_scale", 0.85)) * xy_tol / self._rebar_radius_xy
        return max(0.035, min(yaw_tol, rebar_gate))

    def _attitude_gate(self, s, yaw_err):
        if not bool(self.cfg.get("attitude_z_gate", True)):
            return 1.0
        min_frac = float(self.cfg.get("attitude_gate_min", 0.08))
        yaw_gate = self._effective_yaw_gate()
        yaw_frac = _soft_abs_gate(
            yaw_err, 0.55 * yaw_gate, 1.25 * yaw_gate, min_frac)
        ins = self.config.get("insertion", {})
        paper = self.config.get("paper_eval", {})
        tilt_tol = float(paper.get(
            "tilt_tolerance", ins.get("tilt_tolerance", 0.25)))
        if tilt_tol >= 0.75:
            tilt_frac = 1.0
        else:
            tilt_frac = _soft_abs_gate(
                s.get("tilt", 0.0), 0.65 * tilt_tol, 1.30 * tilt_tol, min_frac)
        return float(np.clip(yaw_frac * tilt_frac, min_frac, 1.0))

    def _xy_vel_limit(self, s):
        base = float(self.cfg.get("vel_max_xy", 0.18))
        target_pz = float(self.config.get("insertion", {}).get(
            "target_payload_z", 0.10))
        margin = max(float(self.cfg.get("terminal_z_margin", 0.060)), 1e-6)
        if s["pl_z"] >= target_pz + margin:
            return base
        scale = float(np.clip(self.cfg.get("terminal_vel_scale", 0.70), 0.05, 1.0))
        frac = float(np.clip((s["pl_z"] - target_pz) / margin, 0.0, 1.0))
        limit = base * (scale + (1.0 - scale) * frac)
        return max(limit, float(self.cfg.get("terminal_vel_min_xy", 0.050)))

    def _terminal_align_velocity(self, v_xy, s, err_xy, xy_err_norm):
        if not bool(self.cfg.get("terminal_align_enabled", True)):
            return v_xy
        target_pz = float(self.config.get("insertion", {}).get(
            "target_payload_z", 0.10))
        z_margin = float(self.cfg.get("terminal_align_z_margin", 0.060))
        xy_gate = float(self.cfg.get("terminal_align_xy", 0.075))
        if s["pl_z"] > target_pz + z_margin or xy_err_norm > xy_gate:
            return v_xy
        align = (
            float(self.cfg.get("terminal_align_kp_xy", 1.10)) * err_xy -
            float(self.cfg.get("terminal_align_kd_xy", 0.45)) * s["pl_vel_xy"])
        blend = float(np.clip(self.cfg.get("terminal_align_blend", 0.65), 0.0, 1.0))
        z_frac = 1.0 - float(np.clip(
            (s["pl_z"] - target_pz) / max(z_margin, 1e-6), 0.0, 1.0))
        w = blend * z_frac
        return (1.0 - w) * v_xy + w * align

    def _yaw_command(self, s, yaw_err):
        kp = float(self.cfg.get("yaw_kp", 0.65))
        kd = float(self.cfg.get("yaw_kd", 0.10))
        vmax = float(self.cfg.get("vel_max_yaw", 0.25))
        if bool(self.cfg.get("terminal_yaw_boost_enabled", True)):
            target_pz = float(self.config.get("insertion", {}).get(
                "target_payload_z", 0.10))
            z_margin = float(self.cfg.get("terminal_yaw_boost_z_margin", 0.080))
            if s["pl_z"] <= target_pz + z_margin:
                kp *= float(self.cfg.get("terminal_yaw_kp_scale", 1.35))
                vmax *= float(self.cfg.get("terminal_yaw_vel_scale", 1.50))
        return float(np.clip(
            kp * yaw_err - kd * s.get("pl_yaw_vel", 0.0),
            -vmax, vmax))

    def _solve_delta_q(self, current_q):
        q_start = (self._last_q if self._last_q is not None
                   else np.asarray(current_q, dtype=np.float64))
        q_target = self.ik_solver.solve_4d(
            current_q=q_start.astype(np.float64),
            target_x=float(self._ee_pos[0]),
            target_y=float(self._ee_pos[1]),
            target_z=float(self._ee_pos[2]),
            target_yaw=float(self._ee_yaw),
        )
        if q_target is None or np.any(np.isnan(q_target)):
            q_target = np.asarray(current_q, dtype=np.float64).copy()
        q_target = np.clip(q_target, self.q_low, self.q_high)
        self._last_q = q_target.copy()
        dq = q_target - np.asarray(current_q, dtype=np.float64)
        return np.clip(dq, -self.dq_max, self.dq_max).astype(np.float32)


class DescentPIDExpert(_VirtualEEDescentExpert):
    name = "pid"

    def compute_delta_q_target(self, env_obs, current_q, env=None):
        s = self._state(env_obs, env=env)
        target_pz = float(self.config.get("insertion", {}).get(
            "target_payload_z", 0.10))
        err_xy = s["target_xy"] - s["pl_xy"]
        xy_err_norm = float(np.linalg.norm(err_xy))

        if xy_err_norm <= float(self.cfg.get("integral_active_xy", 0.06)):
            self._integral_xy += err_xy * self.dt
            self._integral_xy = _clip_norm(
                self._integral_xy, float(self.cfg.get("integral_limit", 0.010)))
        else:
            self._integral_xy *= 0.80

        swing_xy = s["pl_xy"] - self._ee_pos[:2]
        rel_vel_xy = s["pl_vel_xy"] - self._ee_vel[:2]
        swing_gain = float(self.cfg.get("k_swing_xy", 1.60))
        swing_fade = float(self.cfg.get("swing_fade_xy", 0.0))
        fade_z_margin = float(self.cfg.get("swing_fade_z_margin", 0.0))
        fade_active = (
            swing_fade > 1e-9 and
            s["pl_z"] <= target_pz + max(0.0, fade_z_margin))
        if fade_active:
            swing_gain *= float(np.clip(xy_err_norm / swing_fade, 0.0, 1.0))
        v_xy = (
            float(self.cfg.get("kp_xy", 1.15)) * err_xy +
            float(self.cfg.get("ki_xy", 0.35)) * self._integral_xy -
            float(self.cfg.get("kd_payload_xy", 0.65)) * s["pl_vel_xy"] +
            swing_gain * swing_xy +
            float(self.cfg.get("k_rel_vel_xy", 0.35)) * rel_vel_xy)
        v_xy = self._terminal_align_velocity(v_xy, s, err_xy, xy_err_norm)
        v_xy = _clip_norm(v_xy, self._xy_vel_limit(s))

        yaw_err = _wrap_pi(float(self.cfg.get("target_yaw", 0.0)) - s["pl_yaw"])
        gate = _gate_fraction(
            xy_err_norm,
            self.cfg.get("z_full_gate", 0.010),
            self.cfg.get("z_gate", 0.030),
            self.cfg.get("z_trickle_gate", 0.080),
            self.cfg.get("z_min_gate_frac", 0.20))
        attitude_gate = self._attitude_gate(s, yaw_err)
        gate *= attitude_gate
        v_z_raw = (
            float(self.cfg.get("z_kp", 0.65)) * (target_pz - s["pl_z"]) -
            float(self.cfg.get("z_kd", 0.35)) * s["pl_vz"])
        v_z = float(np.clip(
            gate * v_z_raw,
            -float(self.cfg.get("vel_max_z", 0.030)),
            float(self.cfg.get("vel_max_z", 0.030))))
        if s["pl_z"] <= target_pz:
            v_z = max(0.0, v_z)

        self._ee_vel[:2] = v_xy
        self._ee_vel[2] = 0.70 * self._ee_vel[2] + 0.30 * v_z
        self._ee_pos += self._ee_vel * self.dt
        min_ee_z = target_pz + float(self.config.get("controller", {}).get("L", 0.5))
        if self._ee_pos[2] < min_ee_z:
            self._ee_pos[2] = min_ee_z
            self._ee_vel[2] = max(0.0, self._ee_vel[2])

        self._ee_yaw_vel = self._yaw_command(s, yaw_err)
        self._ee_yaw += self._ee_yaw_vel * self.dt
        self._anchor(s["ee_pos"])
        self.last_info = {
            "xy_err": xy_err_norm,
            "z_gate": gate,
            "attitude_gate": attitude_gate,
            "swing_gain": swing_gain,
            "v_xy": float(np.linalg.norm(v_xy)),
            "v_z": v_z,
            "yaw_err": yaw_err,
            "yaw_gate": self._effective_yaw_gate(),
        }
        return self._solve_delta_q(current_q)


class DescentDampedPDExpert(DescentPIDExpert):
    name = "damped_pd"


class DescentMPCExpert(_VirtualEEDescentExpert):
    name = "mpc"

    def __init__(self, config, ik_solver):
        super().__init__(config, ik_solver)
        self._last_cmd_xy = np.zeros(2, dtype=np.float64)

    def reset(self, env_obs, init_q, env=None):
        super().reset(env_obs, init_q, env=env)
        self._last_cmd_xy[:] = 0.0

    def _candidate_xy_velocities(self, s):
        vel_max = self._xy_vel_limit(s)
        err_xy = s["target_xy"] - s["pl_xy"]
        swing_xy = s["pl_xy"] - self._ee_pos[:2]
        feedback = (
            float(self.cfg.get("kp_feedback_xy", 1.85)) * err_xy -
            float(self.cfg.get("kd_feedback_xy", 0.60)) * s["pl_vel_xy"] +
            float(self.cfg.get("k_swing_feedback_xy", 0.35)) * swing_xy)
        feedback = _clip_norm(feedback, vel_max)

        dirs = []
        for vec in (err_xy, feedback, -s["pl_vel_xy"], err_xy + 0.5 * swing_xy):
            unit = _unit(vec)
            if unit is not None:
                dirs.append(unit)

        if dirs:
            primary = dirs[0]
            dirs.append(np.array([-primary[1], primary[0]], dtype=np.float64))
            dirs.append(np.array([primary[1], -primary[0]], dtype=np.float64))

        candidates = [
            np.zeros(2, dtype=np.float64),
            feedback,
            0.5 * feedback,
            self._last_cmd_xy,
            0.5 * self._last_cmd_xy,
        ]
        scales = self.cfg.get("candidate_scales", [0.0, 0.35, 0.60, 0.85, 1.00])
        lateral_scales = self.cfg.get("lateral_scales", [-0.35, 0.0, 0.35])
        base = dirs[0] if dirs else np.array([1.0, 0.0], dtype=np.float64)
        lateral = np.array([-base[1], base[0]], dtype=np.float64)
        for scale in scales:
            for lat in lateral_scales:
                cmd = vel_max * (float(scale) * base + float(lat) * lateral)
                candidates.append(_clip_norm(cmd, vel_max))
        for direction in dirs[1:]:
            for scale in scales:
                candidates.append(_clip_norm(vel_max * float(scale) * direction, vel_max))

        unique = []
        seen = set()
        for cmd in candidates:
            cmd = _clip_norm(cmd, vel_max)
            key = tuple(np.round(cmd, 4))
            if key not in seen:
                unique.append(cmd)
                seen.add(key)
        return unique, feedback

    def _predict_xy_cost(self, cmd_xy, s):
        dt = self.dt
        horizon = int(self.cfg.get("N", 14))
        rope_L = max(float(self.config.get("controller", {}).get("L", 0.5)), 1e-3)
        gravity = float(self.cfg.get("model_gravity", 9.81))
        damping = float(self.cfg.get("model_damping_xy", 1.15))
        q_stage = float(self.cfg.get("q_stage_xy", 180.0))
        q_terminal = float(self.cfg.get("q_terminal_xy", 850.0))
        q_vel = float(self.cfg.get("q_payload_vel_xy", 24.0))
        q_swing = float(self.cfg.get("q_swing_xy", 38.0))
        r_cmd = float(self.cfg.get("r_cmd_vel_xy", 6.0))

        payload_xy = s["pl_xy"].copy()
        payload_vel = s["pl_vel_xy"].copy()
        ee_xy = self._ee_pos[:2].copy()
        target_xy = s["target_xy"]
        cost = r_cmd * float(np.dot(cmd_xy, cmd_xy))
        for k in range(max(1, horizon)):
            ee_xy = ee_xy + cmd_xy * dt
            swing_xy = payload_xy - ee_xy
            payload_acc = -(gravity / rope_L) * swing_xy - damping * payload_vel
            payload_vel = payload_vel + payload_acc * dt
            payload_xy = payload_xy + payload_vel * dt
            err_xy = target_xy - payload_xy
            stage_w = 1.0 + 0.05 * k
            cost += stage_w * (
                q_stage * float(np.dot(err_xy, err_xy)) +
                q_vel * float(np.dot(payload_vel, payload_vel)) +
                q_swing * float(np.dot(swing_xy, swing_xy)))
        terminal_err = target_xy - payload_xy
        cost += q_terminal * float(np.dot(terminal_err, terminal_err))
        return cost

    def compute_delta_q_target(self, env_obs, current_q, env=None):
        s = self._state(env_obs, env=env)
        target_pz = float(self.config.get("insertion", {}).get(
            "target_payload_z", 0.10))
        err_xy = s["target_xy"] - s["pl_xy"]
        xy_err = float(np.linalg.norm(err_xy))
        yaw_err = _wrap_pi(float(self.cfg.get("target_yaw", 0.0)) - s["pl_yaw"])
        attitude_gate = self._attitude_gate(s, yaw_err)

        candidates, feedback_xy = self._candidate_xy_velocities(s)
        costs = [self._predict_xy_cost(cmd, s) for cmd in candidates]
        best_i = int(np.argmin(costs))
        cmd_xy = candidates[best_i]
        if xy_err > float(self.cfg.get("feedback_override_xy", 0.060)):
            cmd_xy = feedback_xy
        alpha = float(np.clip(self.cfg.get("cmd_smoothing_alpha", 0.45), 0.0, 1.0))
        rate = float(self.cfg.get("cmd_rate_limit_xy", 0.050))
        cmd_xy = alpha * cmd_xy + (1.0 - alpha) * self._last_cmd_xy
        cmd_xy = np.clip(cmd_xy, self._last_cmd_xy - rate, self._last_cmd_xy + rate)
        self._last_cmd_xy = _clip_norm(
            cmd_xy, float(self.cfg.get("vel_max_xy", 0.18)))

        payload_speed_xy = float(np.linalg.norm(s["pl_vel_xy"]))
        attitude_ready = attitude_gate >= float(self.cfg.get(
            "descent_attitude_ready", 0.80))
        ready_to_descend = (
            xy_err <= float(self.cfg.get("descent_ready_xy", 0.030)) and
            payload_speed_xy <= float(self.cfg.get("descent_ready_payload_vel", 0.120)) and
            attitude_ready)
        cruise_pz = float(self.config.get("planning", {}).get(
            "payload_z_cruise", target_pz + 0.20))
        recover_low = (
            xy_err > float(self.cfg.get("recovery_xy", 0.055)) and
            s["pl_z"] < cruise_pz - float(self.cfg.get(
                "recovery_payload_z_margin", 0.030)))
        z_target = target_pz if ready_to_descend and not recover_low else cruise_pz
        gate = _gate_fraction(
            xy_err,
            self.cfg.get("z_full_gate", 0.012),
            self.cfg.get("z_gate", 0.035),
            self.cfg.get("z_trickle_gate", 0.080),
            self.cfg.get("z_min_gate_frac", 0.15))
        v_z_raw = (
            float(self.cfg.get("z_kp", 0.70)) * (z_target - s["pl_z"]) -
            float(self.cfg.get("z_kd", 0.35)) * s["pl_vz"])
        if z_target > target_pz:
            gate = 1.0
        else:
            gate *= attitude_gate
        v_z = float(np.clip(
            gate * v_z_raw,
            -float(self.cfg.get("vel_max_z", 0.030)),
            float(self.cfg.get("vel_max_z", 0.030))))
        if z_target <= target_pz and s["pl_z"] <= target_pz:
            v_z = max(0.0, v_z)

        self._ee_vel[:2] = self._last_cmd_xy
        self._ee_vel[2] = 0.70 * self._ee_vel[2] + 0.30 * v_z
        self._ee_vel[2] = float(np.clip(
            self._ee_vel[2],
            -float(self.cfg.get("vel_max_z", 0.030)),
            float(self.cfg.get("vel_max_z", 0.030))))
        self._ee_pos += self._ee_vel * self.dt
        min_ee_z = target_pz + float(self.config.get("controller", {}).get("L", 0.5))
        if self._ee_pos[2] < min_ee_z:
            self._ee_pos[2] = min_ee_z
            self._ee_vel[2] = max(0.0, self._ee_vel[2])
        self._ee_yaw_vel = self._yaw_command(s, yaw_err)
        self._ee_yaw += self._ee_yaw_vel * self.dt
        self._anchor(s["ee_pos"])
        self.last_info = {
            "xy_err": xy_err,
            "z_gate": gate,
            "attitude_gate": attitude_gate,
            "attitude_ready": attitude_ready,
            "ready": ready_to_descend,
            "z_target": z_target,
            "cmd_xy": float(np.linalg.norm(self._last_cmd_xy)),
            "feedback_xy": float(np.linalg.norm(feedback_xy)),
            "candidate_cost": float(costs[best_i]),
            "v_z": v_z,
            "yaw_err": yaw_err,
            "yaw_gate": self._effective_yaw_gate(),
        }
        return self._solve_delta_q(current_q)


EXPERT_REGISTRY: Dict[str, Type[_VirtualEEDescentExpert]] = {
    "pid": DescentPIDExpert,
    "damped_pd": DescentDampedPDExpert,
    "mpc": DescentMPCExpert,
}


def make_traditional_expert(name, config, ik_solver):
    key = str(name).strip().lower()
    if key not in EXPERT_REGISTRY:
        valid = ", ".join(sorted(EXPERT_REGISTRY))
        raise ValueError(f"Unknown traditional expert '{name}'. Valid: {valid}")
    return EXPERT_REGISTRY[key](config, ik_solver)
