# ==============================================================================
# train_phase.py — 三阶段独立训练框架 v9 (BC 完全移除版)
#
# v9 重构核心:
#   1. [BC 完全移除] cruise 残差 actor 在 PPOPhaseAgent.__init__ 中自动零初始化,
#      pretrain_bc_cruise / pretrain_bc_descent 函数和相关 CLI 选项全部删除.
#      来源: Jeon et al. 2025 (Residual MPC), Ankile et al. 2024 (ResiP).
#   2. 缩小 cruise 残差幅度: 0.25 → 0.08 (base 31% → 13%, 符合文献最佳实践)
#   3. 缩小课程扰动: wind 2N → 0.5N, 三类噪声 ÷3 (匹配 RL 物理修正能力)
#   4. Descent 课程 6 级 → 5 级 (跳变 3× 改 2×), 加入倒退机制 (SR < 20% 回退一级)
#   5. wandb 日志精简: PPO 训练 + 课程 + 训练表现 三类
# ==============================================================================

import os
import sys
import csv
import copy
import time
import random
import argparse
import re
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque, Counter

from config import DEFAULT_CONFIG
from mujoco_env_new import CableRobotEnvWithObstacles
from controller import JointSpaceExpert
from phase_agent import (
    PPOPhaseAgent, SACPhaseAgent,
    build_cruise_obs, build_descent_obs,
    build_wind_obs, CableEncoder,
    OBS_CABLE_START, OBS_CABLE_TOTAL,
    delay_mdp_extra_dim, adaptation_history_extra_dim,
    PPO_ZERO, SAC_ZERO,
)
from phase_reward import (
    compute_cruise_reward, compute_descent_reward,
    CruiseRewardState, DescentRewardState,
    RewardComponentTracker,
)
from ee_acc_controller import EEAccController, CruiseZYawPID, SwingDampingController
from obs_predictor import (
    build_observation_predictor, predictor_ckpt_path,
    build_cable_latent_predictor, cable_latent_predictor_ckpt_path,
)
from stability_metrics import StabilityMetrics  # [v12.6] 训练时输出细粒度评估指标

import mujoco
from scipy.spatial.transform import Rotation as R


# ==============================================================================
# 工具
# ==============================================================================

def set_global_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wind_speed_to_force(config, speed_mps):
    wind_cfg = config.get("wind", {})
    v = max(0.0, float(speed_mps))
    rho = float(wind_cfg.get("air_density", 1.225))
    cd = float(wind_cfg.get("drag_coefficient", 1.30))
    area = float(wind_cfg.get("projected_area", 0.020))
    force = 0.5 * rho * cd * area * v * v
    return float(min(force, float(wind_cfg.get("F_max", force))))


def wind_obs_scale(config):
    wobs = config.get("wind_obs", {})
    return float(wobs.get("wind_speed_max",
                          config.get("wind", {}).get("speed_max", 16.5)))


def _nested_get(config, section, key, default=None):
    val = config.get(section, {})
    if isinstance(val, dict) and key in val:
        return val[key]
    return default


def _deep_update(dst, src):
    """Recursively merge src into dst."""
    for key, value in (src or {}).items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def _apply_descent_high_freq_time_scale_fixes(overrides, new_freq, new_dt):
    """Patch high-frequency descent branches; default 10Hz stays untouched."""
    base_freq = float(DEFAULT_CONFIG.get("sim", {}).get("control_freq_hz", 10.0))
    base_dt = 1.0 / max(base_freq, 1e-6)
    step_ratio = float(new_freq) / max(base_freq, 1e-6)
    dt_ratio = float(new_dt) / max(base_dt, 1e-6)

    ppo_cfg = DEFAULT_CONFIG.get("ppo", {})
    gamma_base = float(ppo_cfg.get("gamma", 0.99))
    gae_base = float(ppo_cfg.get("gae_lambda", 0.95))
    overrides.setdefault("ppo", {})["gamma"] = gamma_base ** (base_freq / float(new_freq))
    overrides.setdefault("ppo", {})["gae_lambda"] = gae_base ** (base_freq / float(new_freq))
    overrides["ppo"]["n_steps"] = max(
        1, int(round(float(ppo_cfg.get("n_steps", 1024)) * step_ratio)))
    overrides["ppo"]["seq_len"] = max(
        1, int(round(float(ppo_cfg.get("seq_len", 8)) * step_ratio)))
    overrides["ppo"]["log_std_floor_steps"] = max(
        1, int(round(float(ppo_cfg.get("log_std_floor_steps", 800_000)) * step_ratio)))
    overrides["ppo"]["entropy_coef_anneal_steps"] = max(
        1, int(round(float(ppo_cfg.get("entropy_coef_anneal_steps", 1_500_000)) * step_ratio)))
    overrides["ppo"]["plasticity_reset_interval"] = max(
        1, int(round(float(ppo_cfg.get("plasticity_reset_interval", 200_000)) * step_ratio)))
    overrides["ppo"]["obs_norm_warm_start"] = max(
        1, int(round(float(ppo_cfg.get("obs_norm_warm_start", 5000)) * step_ratio)))

    drl = overrides.setdefault("descent_rl", {})
    drl["early_stop_patience"] = max(
        1, int(round(float(DEFAULT_CONFIG["descent_rl"].get(
            "early_stop_patience", 20)) * step_ratio)))
    drl["descent_entropy_coef_anneal_steps"] = max(
        1, int(round(float(DEFAULT_CONFIG["descent_rl"].get(
            "descent_entropy_coef_anneal_steps", 900_000)) * step_ratio)))
    reward_cfg = copy.deepcopy(DEFAULT_CONFIG["descent_rl"].get("reward", {}))
    reward_cfg.update(copy.deepcopy(drl.get("reward", {})))
    reward_cfg["dense_dt_scale"] = dt_ratio
    drl["reward"] = reward_cfg

    ins = overrides.setdefault("insertion", {})
    ins["stuck_fail_patience"] = max(
        1, int(round(float(DEFAULT_CONFIG["insertion"].get(
            "stuck_fail_patience", 25)) * step_ratio)))
    ins["stuck_fail_progress_eps"] = (
        float(DEFAULT_CONFIG["insertion"].get(
            "stuck_fail_progress_eps", 0.0002)) * dt_ratio)
    ins["clean_insert_bad_contact_patience"] = max(
        1, int(round(1.0 * step_ratio)))
    ins["xy_tolerance_anneal_steps"] = max(
        1, int(round(float(DEFAULT_CONFIG["insertion"].get(
            "xy_tolerance_anneal_steps", 500_000)) * step_ratio)))

    print("  [Descent high-frequency time-scale fixes]")
    print(f"    base/control freq: {base_freq:.1f}Hz -> {float(new_freq):.1f}Hz "
          f"(step_ratio={step_ratio:.3f}, dt_ratio={dt_ratio:.3f})")
    print(f"    PPO gamma/gae_lambda: {overrides['ppo']['gamma']:.6f} / "
          f"{overrides['ppo']['gae_lambda']:.6f}")
    print(f"    PPO n_steps/seq_len: {overrides['ppo']['n_steps']} / "
          f"{overrides['ppo']['seq_len']}")
    print(f"    reward dense_dt_scale: {dt_ratio:.3f}")
    print(f"    stuck patience/progress_eps: {ins['stuck_fail_patience']} / "
          f"{ins['stuck_fail_progress_eps']:.6g}")
    print(f"    lucky bad-contact patience: "
          f"{ins['clean_insert_bad_contact_patience']} steps")


def apply_control_frequency_override(overrides, control_freq_hz,
                                     keep_step_budget=False,
                                     scale_limits_with_dt=False,
                                     phase=None):
    """Add train-time control-frequency overrides to a custom config dict."""
    if control_freq_hz is None:
        return
    new_freq = float(control_freq_hz)
    if new_freq <= 0.0:
        raise ValueError("--control-freq-hz must be positive")
    old_freq = float(_nested_get(
        overrides, "sim", "control_freq_hz",
        DEFAULT_CONFIG.get("sim", {}).get("control_freq_hz", 10.0)))
    old_dt = 1.0 / max(old_freq, 1e-6)
    new_dt = 1.0 / new_freq
    step_ratio = new_freq / max(old_freq, 1e-6)
    dt_ratio = new_dt / max(old_dt, 1e-6)

    overrides.setdefault("sim", {})["control_freq_hz"] = new_freq
    overrides.setdefault("controller", {})["dt"] = new_dt
    overrides.setdefault("ee_control", {})["integrator_dt"] = new_dt

    if not keep_step_budget:
        sections = ["sim", "cruise_rl", "descent_rl"]
        if phase in ("cruise", "descent"):
            sections = ["sim", f"{phase}_rl"]
        for section in sections:
            base = DEFAULT_CONFIG.get(section, {})
            if not isinstance(base, dict) or "max_steps" not in base:
                continue
            current_steps = int(_nested_get(
                overrides, section, "max_steps", base["max_steps"]))
            overrides.setdefault(section, {})["max_steps"] = max(
                1, int(round(current_steps * step_ratio)))

    if scale_limits_with_dt:
        sp_base = DEFAULT_CONFIG.get("space", {})
        dq_src = _nested_get(overrides, "space", "dq_max",
                             sp_base.get("dq_max", [0.12] * 7))
        overrides.setdefault("space", {})["dq_max"] = [
            float(v) * dt_ratio for v in np.asarray(dq_src, dtype=np.float64)
        ]
        ctrl_base = DEFAULT_CONFIG.get("controller", {})
        for key in ("action_rate_limit_xy", "action_rate_limit_z",
                    "action_rate_limit_yaw"):
            if key not in ctrl_base and key not in overrides.get("controller", {}):
                continue
            current = float(_nested_get(
                overrides, "controller", key, ctrl_base.get(key, 0.0)))
            overrides.setdefault("controller", {})[key] = current * dt_ratio

    base_control_freq = float(DEFAULT_CONFIG.get("sim", {}).get(
        "control_freq_hz", 10.0))
    is_high_freq_descent = (
        phase == "descent" and
        new_freq > base_control_freq + 1e-6 and
        abs(base_control_freq - 10.0) < 1e-6
    )
    if is_high_freq_descent:
        _apply_descent_high_freq_time_scale_fixes(overrides, new_freq, new_dt)

    print("\n[Train control frequency override]")
    print(f"  control_freq_hz: {old_freq:.1f} -> {new_freq:.1f}")
    print(f"  action period: {old_dt:.3f}s -> {new_dt:.3f}s")
    print(f"  controller.dt / ee_control.integrator_dt: {new_dt:.3f}s")
    if keep_step_budget:
        print("  max_steps: unchanged")
    else:
        print(f"  max_steps scaled by {step_ratio:.3f} to preserve episode time")
    print(f"  per-step dq/rate limits scaled with dt: {bool(scale_limits_with_dt)}")


class EpisodeStats:
    def __init__(self, window=50, extra_windows=None, baseline_window=50):
        extra_windows = extra_windows or []
        if isinstance(extra_windows, (int, float)):
            extra_windows = [int(extra_windows)]
        windows = [int(window)] + [int(w) for w in extra_windows]
        windows = sorted({max(1, w) for w in windows})
        self.primary_window = max(1, int(window))
        self.trend_windows = windows
        self.window = max(windows)
        self.baseline_window = max(1, int(baseline_window))
        self._data = {}
        self._baseline = {}
    def update(self, **kwargs):
        for k, v in kwargs.items():
            try:
                value = float(v)
            except Exception:
                continue
            if not np.isfinite(value):
                continue
            if k not in self._data:
                self._data[k] = deque(maxlen=self.window)
                self._baseline[k] = []
            self._data[k].append(value)
            if len(self._baseline[k]) < self.baseline_window:
                self._baseline[k].append(value)
    def mean(self, key, window=None):
        d = self._data.get(key)
        if not d:
            return 0.0
        vals = list(d)
        if window is not None:
            vals = vals[-max(1, int(window)):]
        return float(np.mean(vals)) if vals else 0.0
    def baseline_mean(self, key):
        d = self._baseline.get(key)
        return float(np.mean(d)) if d else 0.0
    def improvement(self, key, window=None, lower_is_better=False):
        base = self.baseline_mean(key)
        cur = self.mean(key, window=window)
        return float(base - cur) if lower_is_better else float(cur - base)
    def success_rate(self, window=None):
        return self.mean("success", window=window)
    def wandb_trends(self, phase):
        out = {}
        perf_names = {
            "reward": ("reward", False),
            "success": ("sr", False),
            "steps": ("steps", True),
            "dist_to_goal_cm": ("dist_to_goal_cm", True),
        }
        for w in self.trend_windows:
            for key, (name, lower_is_better) in perf_names.items():
                if key not in self._data:
                    continue
                out[f"trend/{phase}/{name}_ma{w}"] = self.mean(key, window=w)
                out[f"trend/{phase}/{name}_improve_ma{w}"] = self.improvement(
                    key, window=w, lower_is_better=lower_is_better)

            for key in sorted(k for k in self._data if k.startswith("stab_")):
                name = key[len("stab_"):]
                out[f"trend_stab/{phase}/{name}_ma{w}"] = self.mean(key, window=w)
                out[f"trend_stab/{phase}/{name}_improve_ma{w}"] = (
                    self.improvement(key, window=w, lower_is_better=True))
        return out


def _make_episode_stats(config):
    train_cfg = config.get("train", {})
    return EpisodeStats(
        window=int(train_cfg.get("log_smooth_win", 200)),
        extra_windows=train_cfg.get("log_trend_windows", [50, 200, 500]),
        baseline_window=int(train_cfg.get("log_baseline_episodes", 50)))


def _update_episode_stats(stats, reward, steps, success,
                          dist_to_goal_cm=0.0, stab_summary=None):
    payload = {
        "reward": reward,
        "steps": steps,
        "success": float(success),
        "dist_to_goal_cm": dist_to_goal_cm,
    }
    for k, v in (stab_summary or {}).items():
        payload[f"stab_{k}"] = v
    stats.update(**payload)


def _new_vision_acc():
    return {
        "steps": 0,
        "valid": 0,
        "active_cameras": [],
        "reprojection_px": [],
        "depth_rmse_m": [],
        "depth_support": [],
        "age_steps": [],
        "failures": Counter(),
    }


def _as_finite_float(value, default=None):
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _safe_percentile(vals, q, default=0.0):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.percentile(arr, q))


def _safe_mean(vals, default=0.0):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.mean(arr))


def _update_vision_acc(acc, info):
    if acc is None:
        return
    vision = info.get("vision", None) if isinstance(info, dict) else None
    if not isinstance(vision, dict) or not vision:
        return
    acc["steps"] += 1
    valid = bool(vision.get("valid", False))
    acc["valid"] += int(valid)
    acc["active_cameras"].append(int(vision.get("active_cameras", 0) or 0))
    acc["age_steps"].append(_as_finite_float(
        vision.get("age_steps", 0.0), 0.0))
    acc["depth_support"].append(_as_finite_float(
        vision.get("depth_support", 0.0), 0.0))
    reproj = _as_finite_float(vision.get("reprojection_error", None), None)
    if reproj is not None:
        acc["reprojection_px"].append(reproj)
    depth_rmse = _as_finite_float(vision.get("depth_rmse", None), None)
    if depth_rmse is not None:
        acc["depth_rmse_m"].append(depth_rmse)
    if not valid:
        reason = str(vision.get("failure_reason", "") or "invalid")
        acc["failures"][reason] += 1


def _summarize_vision_acc(acc):
    if not acc or int(acc.get("steps", 0)) <= 0:
        return {}
    steps = max(1, int(acc.get("steps", 0)))
    failures = acc.get("failures", Counter())
    failure_reason = ""
    failure_count = 0
    if failures:
        failure_reason, failure_count = failures.most_common(1)[0]
    active = np.asarray(acc.get("active_cameras", []), dtype=np.float64)
    depth_support = np.asarray(acc.get("depth_support", []), dtype=np.float64)
    age = np.asarray(acc.get("age_steps", []), dtype=np.float64)
    valid_steps = int(acc.get("valid", 0))
    return {
        "samples": steps,
        "valid_rate": float(valid_steps) / float(steps),
        "valid_steps": valid_steps,
        "active_cameras_mean": float(np.mean(active)) if active.size else 0.0,
        "active_cameras_min": float(np.min(active)) if active.size else 0.0,
        "active_cameras_max": float(np.max(active)) if active.size else 0.0,
        "reprojection_px_mean": _safe_mean(acc.get("reprojection_px", [])),
        "reprojection_px_p95": _safe_percentile(
            acc.get("reprojection_px", []), 95),
        "depth_rmse_mm_mean": 1000.0 * _safe_mean(
            acc.get("depth_rmse_m", [])),
        "depth_rmse_mm_p95": 1000.0 * _safe_percentile(
            acc.get("depth_rmse_m", []), 95),
        "depth_support_mean": float(np.mean(depth_support))
            if depth_support.size else 0.0,
        "age_steps_mean": float(np.mean(age)) if age.size else 0.0,
        "failure_steps": int(steps - valid_steps),
        "top_failure": str(failure_reason),
        "top_failure_count": int(failure_count),
    }


def _vision_summary_wandb_metrics(summary, phase, worker_id):
    if not summary:
        return {}
    return {
        f"vision/{phase}/valid_rate": summary["valid_rate"],
        f"vision/{phase}/valid_steps": summary["valid_steps"],
        f"vision/{phase}/samples": summary["samples"],
        f"vision/{phase}/active_cameras_mean": summary["active_cameras_mean"],
        f"vision/{phase}/active_cameras_min": summary["active_cameras_min"],
        f"vision/{phase}/active_cameras_max": summary["active_cameras_max"],
        f"vision/{phase}/reprojection_px_mean": summary["reprojection_px_mean"],
        f"vision/{phase}/reprojection_px_p95": summary["reprojection_px_p95"],
        f"vision/{phase}/depth_rmse_mm_mean": summary["depth_rmse_mm_mean"],
        f"vision/{phase}/depth_rmse_mm_p95": summary["depth_rmse_mm_p95"],
        f"vision/{phase}/depth_support_mean": summary["depth_support_mean"],
        f"vision/{phase}/age_steps_mean": summary["age_steps_mean"],
        f"vision/{phase}/failure_steps": summary["failure_steps"],
        f"vision/{phase}/top_failure_count": summary["top_failure_count"],
        f"vision/{phase}/worker_id": int(worker_id),
    }


def _new_rope_marker_acc():
    return {
        "steps": 0,
        "markers_total": [],
        "visible": [],
        "valid_after_noise": [],
        "dropout": [],
        "camera_estimates": [],
        "cameras": [],
        "sources": Counter(),
    }


def _update_rope_marker_acc(acc, info):
    if acc is None:
        return
    debug = (info.get("rope_marker_feature_debug", None)
             if isinstance(info, dict) else None)
    if not isinstance(debug, dict) or not debug:
        return
    acc["steps"] += 1
    total = max(0, int(debug.get("markers_total", 0) or 0))
    visible = max(0, int(debug.get("visible", 0) or 0))
    valid = max(0, int(debug.get(
        "valid_after_noise", debug.get("visible", 0)) or 0))
    acc["markers_total"].append(total)
    acc["visible"].append(min(visible, total) if total > 0 else visible)
    acc["valid_after_noise"].append(min(valid, total) if total > 0 else valid)
    acc["dropout"].append(max(0, int(debug.get("dropout", 0) or 0)))
    acc["camera_estimates"].append(max(
        0, int(debug.get("camera_estimates", 0) or 0)))
    acc["cameras"].append(max(0, int(debug.get("cameras", 0) or 0)))
    source = str(debug.get("source", "") or "unknown")
    acc["sources"][source] += 1


def _summarize_rope_marker_acc(acc):
    if not acc or int(acc.get("steps", 0)) <= 0:
        return {}
    total = np.asarray(acc.get("markers_total", []), dtype=np.float64)
    visible = np.asarray(acc.get("visible", []), dtype=np.float64)
    valid = np.asarray(acc.get("valid_after_noise", []), dtype=np.float64)
    denom = np.maximum(total, 1.0)
    source = ""
    if acc.get("sources"):
        source = acc["sources"].most_common(1)[0][0]
    return {
        "samples": int(acc.get("steps", 0)),
        "source": source,
        "markers_total_mean": _safe_mean(total),
        "visible_mean": _safe_mean(visible),
        "valid_after_noise_mean": _safe_mean(valid),
        "visible_rate": _safe_mean(visible / denom),
        "valid_rate": _safe_mean(valid / denom),
        "dropout_mean": _safe_mean(acc.get("dropout", [])),
        "camera_estimates_mean": _safe_mean(acc.get("camera_estimates", [])),
        "cameras_mean": _safe_mean(acc.get("cameras", [])),
    }


def _rope_marker_summary_wandb_metrics(summary, phase, worker_id):
    if not summary:
        return {}
    return {
        f"rope_marker/{phase}/samples": summary["samples"],
        f"rope_marker/{phase}/markers_total_mean":
            summary["markers_total_mean"],
        f"rope_marker/{phase}/visible_mean": summary["visible_mean"],
        f"rope_marker/{phase}/valid_after_noise_mean":
            summary["valid_after_noise_mean"],
        f"rope_marker/{phase}/visible_rate": summary["visible_rate"],
        f"rope_marker/{phase}/valid_rate": summary["valid_rate"],
        f"rope_marker/{phase}/dropout_mean": summary["dropout_mean"],
        f"rope_marker/{phase}/camera_estimates_mean":
            summary["camera_estimates_mean"],
        f"rope_marker/{phase}/cameras_mean": summary["cameras_mean"],
        f"rope_marker/{phase}/worker_id": int(worker_id),
    }


def _lucky_reject_enabled(config):
    cfg = config.get("insertion", {})
    if bool(cfg.get("strict_lucky_reject_always", True)):
        return True
    return bool(cfg.get("train_reject_lucky_rebar_insert", True))


def _maybe_update_lucky_reject_schedule(config, phase, sr, window_n=0):
    """Auto-restore strict lucky-insert rejection after bootstrap SR recovers."""
    cfg = config.get("insertion", {})
    if bool(cfg.get("strict_lucky_reject_always", True)):
        return {
            f"lucky/{phase}/reject_enabled": 1.0,
            f"lucky/{phase}/strict_always": 1.0,
            f"lucky/{phase}/auto_enable": 0.0,
        }
    if not bool(cfg.get("lucky_reject_auto_enable", False)):
        return {
            f"lucky/{phase}/reject_enabled": float(_lucky_reject_enabled(config)),
            f"lucky/{phase}/strict_always": 0.0,
            f"lucky/{phase}/auto_enable": 0.0,
        }
    threshold = float(cfg.get("lucky_reject_enable_sr", 0.70))
    min_window = max(1, int(cfg.get("lucky_reject_min_window", 1)))
    enabled = _lucky_reject_enabled(config)
    if (not enabled) and int(window_n) >= min_window and float(sr) >= threshold:
        cfg["train_reject_lucky_rebar_insert"] = True
        enabled = True
        print(f"  [LuckyReject-{phase}] enabled: SR={float(sr):.0%} "
              f">= {threshold:.0%}, window_n={int(window_n)}")
    return {
        f"lucky/{phase}/reject_enabled": float(enabled),
        f"lucky/{phase}/strict_always": 0.0,
        f"lucky/{phase}/auto_enable": 1.0,
        f"lucky/{phase}/enable_sr": threshold,
        f"lucky/{phase}/window_n": int(window_n),
    }


class Logger:
    """轻量 wandb 包装器, 容错: 没装 wandb 也能跑。"""
    def __init__(self, log_dir, project="phase_rl", run_name=None):
        self._wandb = None; self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        try:
            import wandb
            self._wandb = wandb
            run_info = self._find_existing_wandb_run(log_dir)
            wandb_id = os.environ.get("WANDB_RUN_ID") or run_info.get("id")
            wandb_project = os.environ.get("WANDB_PROJECT") or \
                run_info.get("project") or project
            wandb_name = os.environ.get("WANDB_NAME")
            if wandb_name is None and wandb_id is None:
                wandb_name = run_name or os.path.basename(log_dir)
            self._wandb.init(project=wandb_project, name=wandb_name,
                             id=wandb_id, dir=log_dir, config={},
                             resume="allow")
        except Exception:
            pass

    @staticmethod
    def _find_existing_wandb_run(log_dir):
        wandb_dir = os.path.join(log_dir, "wandb")
        if not os.path.isdir(wandb_dir):
            return {}
        candidates = []
        try:
            for name in os.listdir(wandb_dir):
                path = os.path.join(wandb_dir, name)
                if not name.startswith("run-") or not os.path.isdir(path):
                    continue
                run_id = name.rsplit("-", 1)[-1]
                if run_id:
                    candidates.append((os.path.getmtime(path), path, run_id))
        except Exception:
            return {}
        if not candidates:
            return {}
        _, run_path, run_id = max(candidates, key=lambda x: x[0])
        info = {"id": run_id}
        debug_log = os.path.join(run_path, "logs", "debug.log")
        try:
            with open(debug_log, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "finishing run " not in line:
                        continue
                    m = re.search(r"/([^/]+)/([^/\s]+)\s*$", line.strip())
                    if m:
                        info["project"] = m.group(1)
        except Exception:
            pass
        return info

    def update_config(self, cfg):
        if self._wandb:
            flat = {}
            def _f(d, pre=""):
                for k, v in d.items():
                    key = f"{pre}{k}"
                    if isinstance(v, dict): _f(v, key + "/")
                    else: flat[key] = v
            _f(cfg)
            try: self._wandb.config.update(flat, allow_val_change=True)
            except Exception: pass
    def log(self, step, metrics):
        if self._wandb:
            try: self._wandb.log(metrics, step=step)
            except Exception: pass
    def close(self):
        if self._wandb:
            try: self._wandb.finish()
            except Exception: pass


def _read_csv_progress(*paths):
    for path in paths:
        if not path or not os.path.exists(path) or os.path.getsize(path) <= 0:
            continue
        try:
            last = None
            with open(path, "r", newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    last = row
            if last is None:
                continue
            ep = int(float(last.get("episode", 0))) + 1
            ts = int(float(last.get("total_steps", 0)))
            return ep, ts, path
        except Exception:
            continue
    return 0, 0, None


def _resolve_resume_training_target(total_timesteps, current_steps,
                                    resume_ckpt=None, progress_log=None,
                                    label="train"):
    """Resolve the loop target after loading a checkpoint.

    When fine-tuning from an old checkpoint into a fresh dated log directory,
    users usually mean --timesteps as the number of new environment steps.  The
    checkpoint may already have a larger global step counter, so a plain
    while ts < total_timesteps would otherwise exit immediately.
    """
    target = int(total_timesteps)
    current = int(current_steps)
    if not resume_ckpt or current <= 0:
        return target

    if progress_log is None and target <= current:
        requested = max(target, 0)
        target = current + requested
        print(f"  [Resume target] {label}: new log dir starts at "
              f"ts={current}; treating --timesteps={requested} as "
              f"additional steps -> target_total_steps={target}")
    else:
        print(f"  [Resume target] {label}: target_total_steps={target}, "
              f"remaining_steps={max(target - current, 0)}")
    return target


def _read_curriculum_progress(path, phase):
    """Read the last curriculum state from a training CSV or run directory."""
    if not path:
        return None
    candidates = []
    if os.path.isdir(path):
        candidates.extend([
            os.path.join(path, f"{phase}_ppo_vec_log.csv"),
            os.path.join(path, f"{phase}_ppo_log.csv"),
            os.path.join(path, f"{phase}_sac_vec_log.csv"),
            os.path.join(path, f"{phase}_sac_log.csv"),
        ])
    else:
        candidates.append(path)

    for cand in candidates:
        if not cand or not os.path.exists(cand) or os.path.getsize(cand) <= 0:
            continue
        last = None
        with open(cand, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                last = row
        if not last:
            continue

        def _as_float(name, default=None):
            try:
                value = last.get(name, "")
                return float(value) if value != "" else default
            except Exception:
                return default

        def _as_int(name, default=None):
            value = _as_float(name, None)
            return int(value) if value is not None else default

        wind = _as_float("wind_speed_max", None)
        if wind is None:
            continue
        return {
            "path": cand,
            "wind": wind,
            "level": _as_int("lvl", None),
            "eps_at_level": _as_int("cur_eps", None),
            "sr": _as_float("cur_sr", None),
        }
    return None


def _init_csv_log(path, header, append_existing=False):
    if append_existing and os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)


def make_dated_log_dir(base_log_dir, timestamp=None):
    stamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
    base = os.path.normpath(str(base_log_dir))
    candidate = f"{base}_{stamp}"
    if not os.path.exists(candidate):
        return candidate
    for idx in range(1, 1000):
        alt = f"{candidate}_{idx:02d}"
        if not os.path.exists(alt):
            return alt
    raise RuntimeError(f"Could not allocate a unique log dir for {candidate}")


def parse_xy_range_arg(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = max(0.0, float(value))
        return [v, v]
    text = str(value).strip()
    if not text:
        return None
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    vals = [max(0.0, float(p)) for p in parts]
    if len(vals) == 1:
        return [vals[0], vals[0]]
    if len(vals) >= 2:
        return vals[:2]
    return None


def save_checkpoint(agent, log_dir, episode, tag="", obs_predictor=None,
                    cable_latent_predictor=None):
    fname = f"ckpt_{tag}.pt" if tag else f"ckpt_ep{episode}.pt"
    path = os.path.join(log_dir, fname)
    latest_path = os.path.join(log_dir, "ckpt_latest.pt")
    agent.save(path)
    if os.path.abspath(path) != os.path.abspath(latest_path):
        agent.save(latest_path)
    if obs_predictor is not None:
        try:
            _save_obs_predictor_checkpoint(obs_predictor, predictor_ckpt_path(path))
            if os.path.abspath(path) != os.path.abspath(latest_path):
                _save_obs_predictor_checkpoint(
                    obs_predictor, predictor_ckpt_path(latest_path))
        except Exception as e:
            print(f"[WARN] obs predictor save failed: {e}")
    if cable_latent_predictor is not None:
        try:
            _save_cable_latent_predictor_checkpoint(
                cable_latent_predictor, cable_latent_predictor_ckpt_path(path))
            if os.path.abspath(path) != os.path.abspath(latest_path):
                _save_cable_latent_predictor_checkpoint(
                    cable_latent_predictor,
                    cable_latent_predictor_ckpt_path(latest_path))
        except Exception as e:
            print(f"[WARN] cable latent predictor save failed: {e}")
    return path


def _best_checkpoint_ready(cur, cur_info, phase):
    min_window = max(1, int(getattr(cur, "_ramp_min_window", 1) or 1))
    window_n = int(cur_info.get(f"cur/{phase}/ramp_window_n", 0) or 0)
    return window_n >= min_window


def _save_obs_predictor_checkpoint(obs_predictor, path):
    if isinstance(obs_predictor, (list, tuple)):
        predictors = list(obs_predictor)
        if not any(p is not None and getattr(p, "enabled", True)
                   for p in predictors):
            return
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save({
            "type": "ensemble",
            "predictors": [
                p.state_dict() if p is not None else None for p in predictors
            ],
        }, path)
        return
    obs_predictor.save(path)


def _load_obs_predictor_checkpoint(obs_predictor, path):
    if obs_predictor is None or not os.path.exists(path):
        return False

    if isinstance(obs_predictor, (list, tuple)):
        predictors = [p for p in obs_predictor if p is not None]
        if not predictors:
            return False
        device = getattr(predictors[0], "device", None)
        ck = torch.load(path, map_location=device, weights_only=False)
        if isinstance(ck, dict) and ck.get("type") == "ensemble":
            states = ck.get("predictors", [])
            for p, state in zip(obs_predictor, states):
                if p is not None and state is not None:
                    p.load_state_dict(state)
        else:
            for p in predictors:
                p.load_state_dict(ck)
        return True

    obs_predictor.load(path)
    return True


def _obs_predictor_checkpoint_config(config):
    path = config.get("observation_predictor", {}).get("checkpoint", "")
    return str(path).strip() if path else ""


def _candidate_obs_predictor_checkpoints(config, resume_ckpt=None):
    explicit = _obs_predictor_checkpoint_config(config)
    seen = set()
    out = []
    for path in [explicit, predictor_ckpt_path(resume_ckpt) if resume_ckpt else ""]:
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _try_load_obs_predictor_checkpoint(obs_predictor, config, resume_ckpt=None,
                                       label="ObsPredictor"):
    explicit = _obs_predictor_checkpoint_config(config)
    explicit_norm = os.path.normpath(explicit) if explicit else ""
    for path in _candidate_obs_predictor_checkpoints(config, resume_ckpt):
        if not os.path.exists(path):
            msg = f"  [{label}] checkpoint not found: {path}"
            if explicit_norm and os.path.normpath(path) == explicit_norm:
                raise FileNotFoundError(msg.strip())
            print(msg)
            continue
        try:
            if _load_obs_predictor_checkpoint(obs_predictor, path):
                print(f"  [{label}] resumed from {path}")
                return path
        except Exception as e:
            print(f"  [WARN] {label} resume failed from {path}: {e}")
            if explicit_norm and os.path.normpath(path) == explicit_norm:
                raise
    return None


def _save_cable_latent_predictor_checkpoint(cable_latent_predictor, path):
    if isinstance(cable_latent_predictor, (list, tuple)):
        predictors = list(cable_latent_predictor)
        if not any(p is not None and getattr(p, "enabled", True)
                   for p in predictors):
            return
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save({
            "type": "ensemble",
            "predictors": [
                p.state_dict() if p is not None else None for p in predictors
            ],
        }, path)
        return
    cable_latent_predictor.save(path)


def _load_cable_latent_predictor_checkpoint(cable_latent_predictor, path):
    if cable_latent_predictor is None or not os.path.exists(path):
        return False

    if isinstance(cable_latent_predictor, (list, tuple)):
        predictors = [p for p in cable_latent_predictor if p is not None]
        if not predictors:
            return False
        device = getattr(predictors[0], "device", None)
        ck = torch.load(path, map_location=device, weights_only=False)
        if isinstance(ck, dict) and ck.get("type") == "ensemble":
            states = ck.get("predictors", [])
            for p, state in zip(cable_latent_predictor, states):
                if p is not None and state is not None:
                    p.load_state_dict(state)
        else:
            for p in predictors:
                p.load_state_dict(ck)
        return True

    cable_latent_predictor.load(path)
    return True


def _cable_latent_predictor_checkpoint_config(config):
    path = config.get("cable_latent_predictor", {}).get("checkpoint", "")
    return str(path).strip() if path else ""


def _candidate_cable_latent_predictor_checkpoints(config, resume_ckpt=None):
    explicit = _cable_latent_predictor_checkpoint_config(config)
    seen = set()
    out = []
    for path in [
        explicit,
        cable_latent_predictor_ckpt_path(resume_ckpt) if resume_ckpt else "",
    ]:
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _try_load_cable_latent_predictor_checkpoint(
        cable_latent_predictor, config, resume_ckpt=None,
        label="CableLatPred"):
    explicit = _cable_latent_predictor_checkpoint_config(config)
    explicit_norm = os.path.normpath(explicit) if explicit else ""
    for path in _candidate_cable_latent_predictor_checkpoints(config, resume_ckpt):
        if not os.path.exists(path):
            msg = f"  [{label}] checkpoint not found: {path}"
            if explicit_norm and os.path.normpath(path) == explicit_norm:
                raise FileNotFoundError(msg.strip())
            print(msg)
            continue
        try:
            if _load_cable_latent_predictor_checkpoint(
                    cable_latent_predictor, path):
                print(f"  [{label}] resumed from {path}")
                return path
        except Exception as e:
            print(f"  [WARN] {label} resume failed from {path}: {e}")
            if explicit_norm and os.path.normpath(path) == explicit_norm:
                raise
    return None


def _try_load_critic_cable_encoder(agent, config):
    if not getattr(agent, "use_asymmetric_critic", False):
        return None
    enc = getattr(agent, "critic_cable_encoder", None)
    if enc is None:
        return None
    path = str(config.get("asymmetric_critic", {}).get(
        "cable_encoder_ckpt", "") or "").strip()
    if not path:
        return None
    if not os.path.exists(path):
        print(f"  [AsymCritic] critic cable encoder checkpoint not found: {path}")
        return None
    try:
        ck = torch.load(path, map_location=getattr(agent, "device", None),
                        weights_only=False)
        state = ck.get("critic_cable_encoder") or ck.get("cable_encoder") or {}
        if state:
            enc.load_state_dict(state)
            print(f"  [AsymCritic] critic cable encoder loaded from {path}")
            return path
        print(f"  [AsymCritic] no cable encoder state in {path}")
    except Exception as e:
        print(f"  [AsymCritic] failed to load critic cable encoder: {e}")
    return None


def _save_obs_predictor_tag(obs_predictor, log_dir, tag):
    path = os.path.join(log_dir, f"ckpt_{tag}_obs_predictor.pt")
    _save_obs_predictor_checkpoint(obs_predictor, path)
    return path


_OBS_PRED_LATENT_TARGET_MODES = {"non_cable_latent", "compact_latent", "latent"}


def _obs_pred_target_mode(config):
    return str(config.get("observation_predictor", {}).get(
        "target_mode", "raw")).lower()


def _obs_pred_uses_cable_latent_target(config):
    return _obs_pred_target_mode(config) in _OBS_PRED_LATENT_TARGET_MODES


def _obs_pred_non_cable_dim(config, raw_obs_dim=None):
    cfg = config.get("observation_predictor", {})
    dim = max(0, int(cfg.get("non_cable_dim", OBS_CABLE_START)))
    if raw_obs_dim is not None:
        dim = min(dim, max(0, int(raw_obs_dim)))
    return dim


def _obs_pred_cable_latent_dim(config, agent=None):
    if agent is not None and hasattr(agent, "_cable_out_dim"):
        return int(getattr(agent, "_cable_out_dim"))
    return max(0, int(config.get("observation_predictor", {}).get(
        "cable_latent_dim", 32)))


def _obs_pred_target_dim(config, raw_obs=None, agent=None):
    raw_dim = None if raw_obs is None else int(
        np.asarray(raw_obs, dtype=np.float32).reshape(-1).size)
    if not _obs_pred_uses_cable_latent_target(config):
        if raw_dim is None:
            raise ValueError("raw_obs is required for raw obs predictor target")
        return raw_dim
    return (_obs_pred_non_cable_dim(config, raw_dim) +
            _obs_pred_cable_latent_dim(config, agent))


def _agent_encode_cable(agent, cable_component, config):
    latent_dim = _obs_pred_cable_latent_dim(config, agent)
    arr = np.asarray(cable_component, dtype=np.float32).reshape(-1)
    if arr.size == latent_dim:
        latent = arr
    elif agent is not None and hasattr(agent, "encode_cable"):
        if arr.size < OBS_CABLE_TOTAL:
            arr = np.pad(arr, (0, OBS_CABLE_TOTAL - arr.size))
        elif arr.size > OBS_CABLE_TOTAL:
            arr = arr[:OBS_CABLE_TOTAL]
        latent = np.asarray(agent.encode_cable(arr), dtype=np.float32).reshape(-1)
    else:
        latent = np.zeros(latent_dim, dtype=np.float32)
    if latent.size < latent_dim:
        latent = np.pad(latent, (0, latent_dim - latent.size))
    return latent[:latent_dim].astype(np.float32)


def _obs_pred_target_vector(config, env_obs, phase_obs=None, agent=None):
    env_arr = np.asarray(env_obs, dtype=np.float32).reshape(-1)
    if not _obs_pred_uses_cable_latent_target(config):
        return env_arr.copy()
    head_dim = _obs_pred_non_cable_dim(config, env_arr.size)
    head = env_arr[:head_dim].copy()
    if head.size < head_dim:
        head = np.pad(head, (0, head_dim - head.size))
    if phase_obs is not None and len(phase_obs) > 1 and phase_obs[1] is not None:
        cable_component = phase_obs[1]
    elif env_arr.size >= OBS_CABLE_START + OBS_CABLE_TOTAL:
        cable_component = env_arr[OBS_CABLE_START:OBS_CABLE_START + OBS_CABLE_TOTAL]
    else:
        cable_component = np.zeros(OBS_CABLE_TOTAL, dtype=np.float32)
    cable_latent = _agent_encode_cable(agent, cable_component, config)
    return np.concatenate([head, cable_latent]).astype(np.float32)


def _obs_pred_visible_env_obs(config, true_env_obs, pred_target, used_prediction):
    true_arr = np.asarray(true_env_obs, dtype=np.float32).reshape(-1)
    if not bool(used_prediction):
        return true_arr.copy()
    pred = np.asarray(pred_target, dtype=np.float32).reshape(-1)
    if not _obs_pred_uses_cable_latent_target(config):
        return pred.copy()
    out = true_arr.copy()
    head_dim = _obs_pred_non_cable_dim(config, out.size)
    copy_dim = min(head_dim, pred.size, out.size)
    if copy_dim > 0:
        out[:copy_dim] = pred[:copy_dim]
    cable_end = min(out.size, OBS_CABLE_START + OBS_CABLE_TOTAL)
    if cable_end > OBS_CABLE_START:
        out[OBS_CABLE_START:cable_end] = 0.0
    return out.astype(np.float32)


def _obs_pred_cable_latent_from_vec(config, pred_target, used_prediction=True):
    if (not bool(used_prediction)) or (not _obs_pred_uses_cable_latent_target(config)):
        return None
    pred = np.asarray(pred_target, dtype=np.float32).reshape(-1)
    head_dim = _obs_pred_non_cable_dim(config, None)
    if pred.size <= head_dim:
        return None
    return pred[head_dim:].astype(np.float32)


def _phase_obs_with_cable_component(phase_obs, cable_component):
    if phase_obs is None or cable_component is None:
        return phase_obs
    return (phase_obs[0], np.asarray(cable_component, dtype=np.float32).reshape(-1),
            phase_obs[2], phase_obs[3], phase_obs[4], phase_obs[5])


def _use_rope_marker_features(config):
    return bool(config.get("cable_latent_predictor", {}).get(
        "use_rope_marker_features", False))


def _rope_marker_feature_source_config(config):
    clp = config.get("cable_latent_predictor", {})
    markers = config.get("rope_markers", {})
    source = str(clp.get(
        "rope_marker_feature_source",
        markers.get("feature_source", "site")) or "").strip().lower()
    if not source:
        source = str(markers.get("feature_source", "site")).strip().lower()
    return source or "site"


def _phase_visible_no_cable_vector(phase_obs, rope_marker_features=None):
    if phase_obs is None:
        return np.zeros(0, dtype=np.float32)
    core = np.asarray(phase_obs[0], dtype=np.float32).reshape(-1)
    wind = np.asarray(phase_obs[2], dtype=np.float32).reshape(-1)
    parts = [core, wind]
    if rope_marker_features is not None:
        parts.append(np.asarray(
            rope_marker_features, dtype=np.float32).reshape(-1))
    return np.concatenate(parts).astype(np.float32)


def _phase_obs_with_predicted_cable_latent(
        phase_obs, cable_latent_predictor, action_for_history, agent, config,
        phase=None, target_phase_obs=None, rope_marker_features=None):
    if phase_obs is None or cable_latent_predictor is None:
        return phase_obs, None
    visible = _phase_visible_no_cable_vector(
        phase_obs, rope_marker_features=rope_marker_features)
    target_src = target_phase_obs if target_phase_obs is not None else phase_obs
    target_latent = _agent_encode_cable(agent, target_src[1], config)
    pred_latent, info = cable_latent_predictor.predict_and_update(
        visible, action=action_for_history, phase=phase,
        target_latent=target_latent)
    return _phase_obs_with_cable_component(phase_obs, pred_latent), info


def _buffer_cable_for_agent(agent, cable_component):
    arr = np.asarray(cable_component, dtype=np.float32).reshape(-1)
    if arr.size == OBS_CABLE_TOTAL:
        return arr.copy()
    return np.zeros(OBS_CABLE_TOTAL, dtype=np.float32)


# ==============================================================================
# Reward / Obs 调度
# ==============================================================================

REWARD_FNS = {
    "cruise":  compute_cruise_reward,
    "descent": compute_descent_reward,
}
REWARD_STATES = {
    "cruise":  CruiseRewardState,
    "descent": DescentRewardState,
}


def build_phase_obs(phase, env_obs, env, start_xy, target_xy, prev_tilt, prev_yaw,
                    wind_obs=None, base_action=None):
    """[v14.0] 返回 (core_obs, cable_raw, wind_obs, tilt, yaw)."""
    if phase == "cruise":
        return build_cruise_obs(env_obs, env, target_xy, prev_tilt, prev_yaw,
                                wind_obs, base_action=base_action)
    elif phase == "descent":
        return build_descent_obs(env_obs, env, target_xy, prev_tilt, prev_yaw,
                                 wind_obs, base_action=base_action)
    raise ValueError(f"Unknown phase: {phase}")


def get_last_nmpc_action(expert):
    """Return the latest 4D NMPC action for residual-policy observations."""
    base = getattr(expert, "last_action_4d", None)
    if base is None:
        return np.zeros(4, dtype=np.float32)
    base = np.asarray(base, dtype=np.float32).reshape(-1)
    if base.size < 4:
        base = np.pad(base, (0, 4 - base.size))
    return base[:4].astype(np.float32)


def clip_cruise_residual(action, config):
    """Clip cruise residual RL output to the small authority around NMPC."""
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    rm_xy = float(config["cruise_rl"].get("residual_acc_max_xy_rl", 0.08))
    rm_z = float(config["cruise_rl"].get("residual_acc_max_z_rl", 0.10))
    return np.array([
        float(np.clip(arr[0] if arr.size > 0 else 0.0, -rm_xy, rm_xy)),
        float(np.clip(arr[1] if arr.size > 1 else 0.0, -rm_xy, rm_xy)),
        float(np.clip(arr[2] if arr.size > 2 else 0.0, -rm_z, rm_z)),
    ], dtype=np.float64)


# ==============================================================================
# 课程管理器 [v8] 统一噪声/风力 + descent 初始化精度
# ==============================================================================

class CurriculumManager:
    """
    统一课程管理器, 每个阶段独立 level 计数.

    每个 level 注入 4 类扰动:
      - obs_noise:   normalize_obs 后的高斯噪声 σ
      - act_noise:   delta_q 上的高斯噪声 σ (rad/step)
      - force_noise: payload 上随机方向力 σ (N)
      - wind_max:    风力上限 (N), 每 episode 从 [0, wind_max] 均匀采样

    descent 额外字段: init_xy / init_vel / init_tilt / xy_tol

    晋级条件:
      (sr >= sr_threshold) AND (eps_at_level >= min_eps)
      OR  eps_at_level >= hard_cap_eps (防卡死)

    [v14.2] 连续课程模式:
      wind_max 不再跳变, 而是在每个 episode 根据 SR 平滑增长:
      - 当 SR >= sr_threshold: wind_max 按 ramp_rate 线性增长
      - 当 SR < sr_threshold: wind_max 暂停增长 (等待 RL 适应)
      - wind_max 不会回退 (避免 PPO on-policy 分布混乱)
      旧 level 结构仅用于 descent 的精度参数, wind 完全由连续 ramp 控制
    """
    def __init__(self, config, phase):
        self.config = config
        cur = config.get("curriculum", {})
        self.enabled = bool(cur.get("enabled", True))
        self.phase   = phase

        self.levels      = list(cur.get(f"{phase}_levels", []))
        self.sr_thresh   = float(cur.get(f"{phase}_sr_threshold", 0.7))
        self.min_eps     = int(cur.get(f"{phase}_min_eps",        150))
        self.hard_cap    = int(cur.get(f"{phase}_hard_cap_eps",   1000))
        self.stats_win   = int(cur.get(f"{phase}_stats_window",   50))

        self.level_idx    = 0
        self.eps_at_level = 0
        self._sr_window   = deque(maxlen=self.stats_win)

        # 上一 episode 采样的风力 (供日志)
        self._last_wind_speed = 0.0
        self._last_wind_dir   = 0.0
        self._last_wind_min   = 0.0

        # ── [v14.2] 连续课程参数 ─────────────────────────────────────────────
        # wind_max 从 levels[0] 的值开始, 按 ramp_rate 每 episode 线性增长
        _wind_levels = [float(lv.get("wind_max", 0.0)) for lv in self.levels] if self.levels else [0.0]
        _configured_min = _wind_levels[0] if _wind_levels else 0.0
        self._wind_min  = max(_configured_min, float(cur.get(
            f"{phase}_ramp_min_wind", _configured_min)))
        self._wind_max  = _wind_levels[-1] if _wind_levels else 0.0
        self._wind_cur  = self._wind_min   # 当前 wind_max (平滑增长)
        # ramp_rate: 每成功 episode 增加的 wind_max 幅度
        # 设计: 从 min 到 max 需要 ~500-800 成功 episode
        _total_range = self._wind_max - self._wind_min
        _target_success_eps = int(cur.get(f"{phase}_ramp_episodes", 600))
        self._wind_ramp_rate = _total_range / max(_target_success_eps, 1)
        # SR 低于此值时暂停增长
        self._ramp_sr_thresh = float(cur.get(f"{phase}_ramp_sr_threshold",
                                              self.sr_thresh * 0.8))
        self._ramp_warmup_sr_thresh = float(cur.get(
            f"{phase}_ramp_warmup_sr_threshold", self._ramp_sr_thresh))
        self._ramp_warmup_scale = float(cur.get(
            f"{phase}_ramp_warmup_scale", 0.0))
        self._ramp_min_window = int(cur.get(
            f"{phase}_ramp_min_window", min(50, self.stats_win)))
        self._wind_focus_enabled = bool(cur.get(
            f"{phase}_wind_focus_enabled", False))
        self._wind_focus_start_level = int(cur.get(
            f"{phase}_wind_focus_start_level", 6))
        self._wind_focus_full_level = int(cur.get(
            f"{phase}_wind_focus_full_level", self._wind_focus_start_level + 2))
        self._wind_focus_start_min = float(cur.get(
            f"{phase}_wind_focus_start_min", 3.0))
        self._wind_focus_full_min = float(cur.get(
            f"{phase}_wind_focus_full_min", 5.0))
        self._wind_focus_final_min = float(cur.get(
            f"{phase}_wind_focus_final_min", 8.0))
        start_wind = cur.get(f"{phase}_start_wind", cur.get("start_wind", None))
        if start_wind is not None:
            # Allow explicit recovery runs from easier-than-L0 wind without
            # flattening the level table via --wind-speed-max 0.
            self._wind_min = min(self._wind_min, float(start_wind))
        start_level = cur.get(f"{phase}_start_level", cur.get("start_level", None))
        start_eps = cur.get(f"{phase}_start_eps_at_level",
                            cur.get("start_eps_at_level", None))
        if start_wind is not None or start_level is not None:
            self.set_progress(wind=start_wind, level=start_level, eps=start_eps)
        # 最低 SR 时减速 (不回退, 但增速降到 0)
        self._ramp_pause_count = 0  # 连续暂停的 episode 数

    # ── 当前 level 信息 ──────────────────────────────────────────────────────
    def _level_from_wind(self, wind):
        if not self.levels:
            return 0
        wind_levels = [float(lv.get("wind_max", 0.0)) for lv in self.levels]
        idx = 0
        for j, wl in enumerate(wind_levels):
            if float(wind) >= wl * 0.9:
                idx = j
        return int(max(0, min(idx, len(self.levels) - 1)))

    def set_progress(self, wind=None, level=None, eps=None):
        """Initialize curriculum progress when transferring a trained policy."""
        if wind is not None:
            w = float(wind)
            self._wind_cur = float(np.clip(w, self._wind_min, self._wind_max))
        if level is None:
            level = self._level_from_wind(self._wind_cur)
        if self.levels:
            self.level_idx = int(max(0, min(int(level), len(self.levels) - 1)))
        else:
            self.level_idx = 0
        if eps is not None:
            self.eps_at_level = max(0, int(eps))
        print(f"  [Curriculum-{self.phase}] start at L{self.level_idx}/"
              f"{self.n_levels-1}, wind={self._wind_cur:.2f}/"
              f"{self._wind_max:.1f}m/s")

    @property
    def n_levels(self):
        return max(1, len(self.levels))
    @property
    def current_level(self):
        if not self.levels:
            return {"obs_noise": 0.0, "act_noise": 0.0, "force_noise": 0.0, "wind_max": 0.0}
        # [v14.2] 用连续 wind_cur 替换 level 中的 wind_max
        idx = min(self.level_idx, len(self.levels) - 1)
        lvl = dict(self.levels[idx])  # copy
        lvl["wind_max"] = self._wind_cur
        return lvl
    @property
    def is_max_level(self):
        return self.level_idx >= len(self.levels) - 1
    @property
    def wind_current(self):
        """[v14.2] 当前连续 wind_max 值."""
        return self._wind_cur

    @property
    def wind_sample_min(self):
        return self._compute_wind_sample_min()

    def _compute_wind_sample_min(self):
        if (not self._wind_focus_enabled) or self._wind_cur <= 0.0:
            return 0.0
        if not self.levels or self.level_idx < self._wind_focus_start_level:
            return 0.0

        start_level = self._wind_focus_start_level
        full_level = max(start_level, self._wind_focus_full_level)
        if self.level_idx < full_level:
            span = max(1, full_level - start_level)
            t = (self.level_idx - start_level) / span
            wind_min = self._wind_focus_start_min + t * (
                self._wind_focus_full_min - self._wind_focus_start_min)
        else:
            wind_levels = [
                float(lv.get("wind_max", 0.0)) for lv in self.levels
            ]
            full_idx = min(max(0, full_level), len(wind_levels) - 1)
            full_wind = wind_levels[full_idx]
            denom = max(1e-6, self._wind_max - full_wind)
            t = float(np.clip((self._wind_cur - full_wind) / denom, 0.0, 1.0))
            wind_min = self._wind_focus_full_min + t * (
                self._wind_focus_final_min - self._wind_focus_full_min)

        return float(np.clip(wind_min, 0.0, max(0.0, self._wind_cur - 1e-6)))

    # ── 每 episode 采样 ──────────────────────────────────────────────────────
    def sample_episode_perturbations(self):
        if not self.enabled:
            return {"obs_noise": 0.0, "act_noise": 0.0,
                    "force_noise": 0.0, "wind_min": 0.0, "wind_speed": 0.0,
                    "wind_max": 0.0, "wind_dir": 0.0}
        lvl = self.current_level
        # [v14.2] wind_max 来自连续 ramp
        wind_max = self._wind_cur
        wind_min = self._compute_wind_sample_min()
        ws = float(np.random.uniform(wind_min, wind_max)) if wind_max > 0 else 0.0
        wd = float(np.random.uniform(0.0, 2 * np.pi))
        self._last_wind_min = wind_min
        self._last_wind_speed = ws
        self._last_wind_dir   = wd
        return {
            "obs_noise":   float(lvl.get("obs_noise",   0.0)),
            "act_noise":   float(lvl.get("act_noise",   0.0)),
            "force_noise": float(lvl.get("force_noise", 0.0)),
            "wind_min":    wind_min,
            "wind_speed":  ws,
            "wind_max":    wind_max,
            "wind_dir":    wd,
        }

    # ── descent 专属: 初始化精度参数 ────────────────────────────────────────
    def get_descent_init(self):
        if self.phase != "descent" or not self.levels:
            return None
        lvl = self.current_level
        return {
            "xy_range":   float(lvl.get("init_xy",   0.030)),
            "vel_range":  float(lvl.get("init_vel",  0.020)),
            "tilt_range": float(lvl.get("init_tilt", 0.008)),
            "xy_tol":     float(lvl.get("xy_tol",    0.005)),
        }

    # ── [v14.2] 连续推进逻辑 ────────────────────────────────────────────────
    def update(self, success):
        """每 episode 末调用. 连续课程: 根据 SR 平滑增长 wind_max."""
        if not self.enabled:
            return False
        self.eps_at_level += 1
        self._sr_window.append(float(success))

        # 统计 SR. Do not assume a neutral 50% SR while the window is small:
        # that made high-wind descent ramp before the policy had earned it.
        sr = float(np.mean(self._sr_window)) if self._sr_window else 0.0
        enough_ramp_stats = len(self._sr_window) >= max(1, self._ramp_min_window)

        # ── 连续 wind ramp ──
        ramp_scale = 0.0
        if enough_ramp_stats:
            if sr >= self._ramp_sr_thresh:
                ramp_scale = 1.0
            elif sr >= self._ramp_warmup_sr_thresh:
                ramp_scale = max(0.0, self._ramp_warmup_scale)

        if ramp_scale > 0.0:
            self._wind_cur = min(
                self._wind_cur + self._wind_ramp_rate * ramp_scale,
                self._wind_max)
            self._ramp_pause_count = 0
        else:
            # SR 太低 → 暂停增长 (不回退)
            self._ramp_pause_count += 1

        # ── 阶梯 level 推进 (仅用于 descent 精度参数) ──
        # 根据 wind_cur 位置推断当前应属哪个 level
        if self.levels:
            _wind_levels = [float(lv.get("wind_max", 0.0)) for lv in self.levels]
            _new_idx = 0
            for j, wl in enumerate(_wind_levels):
                if self._wind_cur >= wl * 0.9:  # 90% 阈值
                    _new_idx = j
            if _new_idx > self.level_idx:
                self.level_idx = _new_idx
                self._just_promoted = True
                print(f"  [Curriculum-{self.phase}] → L{self.level_idx} "
                      f"(wind={self._wind_cur:.2f}m/s, SR={sr:.0%})")

        # 定期打印
        if self.eps_at_level % 100 == 0:
            print(f"  [Curriculum-{self.phase}] ep={self.eps_at_level} "
                  f"wind=[{self.wind_sample_min:.2f},"
                  f"{self._wind_cur:.2f}]/{self._wind_max:.1f}m/s "
                  f"SR={sr:.0%} n={len(self._sr_window)}/{self._ramp_min_window} "
                  f"pause={self._ramp_pause_count}")

        return False

    def consume_promotion_flag(self):
        flag = getattr(self, '_just_promoted', False)
        self._just_promoted = False
        return flag

    # ── 日志辅助 ─────────────────────────────────────────────────────────────
    def info(self):
        """返回当前课程信息字典 (供 wandb 日志)。"""
        lvl = self.current_level
        sr = float(np.mean(self._sr_window)) if self._sr_window else 0.0
        return {
            f"cur/{self.phase}/level":       self.level_idx,
            f"cur/{self.phase}/eps_at_lvl":  self.eps_at_level,
            f"cur/{self.phase}/sr_window":   sr,
            f"cur/{self.phase}/obs_noise":   float(lvl.get("obs_noise",   0.0)),
            f"cur/{self.phase}/act_noise":   float(lvl.get("act_noise",   0.0)),
            f"cur/{self.phase}/force_noise": float(lvl.get("force_noise", 0.0)),
            f"cur/{self.phase}/wind_min":    self.wind_sample_min,
            f"cur/{self.phase}/wind_max":    float(lvl.get("wind_max",    0.0)),
            f"cur/{self.phase}/last_wind":   self._last_wind_speed,
            f"cur/{self.phase}/ramp_window_n": len(self._sr_window),
        }


# ==============================================================================
# 三阶段独立物理初始化 (移除 cruise_dist_curriculum, ORCA 相关)
# ==============================================================================

def _sync_env_internal_state(env):
    env.current_step = 0
    env.current_wp_idx = 0
    env.reached_final = False
    env.last_dist = None
    env.last_wp_idx = -1
    env._wp_just_advanced = False
    env._termination_reason = None
    env._prev_q = env.data.qpos[:7].copy().astype(np.float32)
    env._prev_delta_q = np.zeros(env.action_dim, dtype=np.float32)
    env._insertion_hold_counter = 0
    env._in_insertion_phase = False
    env._best_insertion_z = 10.0
    env._prev_goal_potential = None
    env._prev_phi_z = None
    env._prev_descent_depth = 0.0
    env._prev_ref_dist = None
    env._prev_ee_pos = env._get_ee_pos().copy()
    env._ee_vel_cache = np.zeros(3)
    mat = env._get_ee_mat()
    env._prev_ee_euler = R.from_matrix(mat).as_euler('xyz').copy()
    env._ee_euler_vel_cache = np.zeros(3)
    init_q = env.data.qpos[:7].copy().astype(np.float32)
    env.action_queue.clear()
    for _ in range(max(1, env.latency_steps + 1)):
        env.action_queue.append(init_q.copy())


def reset_for_phase(env, phase, config,
                    override_init_xy_range=None,
                    override_init_vel_range=None,
                    override_init_tilt_range=None,
                    rng_seed=None):
    """统一三阶段物理初始化 (移除 cruise_dist_curriculum, ORCA)。"""
    max_xml_retries = 5
    obs = None; planned_path = None
    for _xml_retry in range(max_xml_retries):
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            obs = env.reset()
            planned_path = env.get_planned_path()
            sys.stdout.close(); sys.stdout = old_stdout
            break
        except (ValueError, RuntimeError) as _xml_err:
            sys.stdout.close(); sys.stdout = old_stdout
            _err_str = str(_xml_err)
            if any(s in _err_str.lower() for s in ["empty file", "xml error", "xml"]):
                if _xml_retry < max_xml_retries - 1:
                    time.sleep(0.05 * (_xml_retry + 1)); continue
            raise
        except Exception:
            sys.stdout.close(); sys.stdout = old_stdout; raise

    if obs is None or planned_path is None:
        return None, None

    phase_cfg = config.get(f"{phase}_rl", {})
    rng = np.random.default_rng(rng_seed)

    if phase == "cruise":
        # [v13.0 合并 lift] cruise 现在从低空 (z ≈ 0.11) 起步, 接管 lift 任务
        # NMPC 自动处理 lift→cruise 边界 (controller.py tracker 已识别 lift WP)
        z_cruise   = float(config["planning"]["payload_z_cruise"])
        init_pref_z = float(config["reset"]["init_qpos_prefab"][2])  # 默认 0.10
        start_xy = np.asarray(getattr(
            env, "episode_start_xy", env.default_start_xy), dtype=np.float64).copy()
        rope_L = float(config["controller"].get("L", 0.5))
        # EE 起步: 比 payload 高一个 rope_L
        ee_z   = init_pref_z + rope_L
        seed_q = np.array(config["reset"]["init_qpos_arm"], np.float64)
        init_q = env.ik_solver.solve_4d(seed_q, float(start_xy[0]), float(start_xy[1]),
                                         ee_z, 0.0)
        if init_q is None or np.any(np.isnan(init_q)):
            init_q = seed_q.copy()

        env.data.qpos[:7] = init_q
        env.data.qvel[:7] = 0.0
        env.data.ctrl[:7] = init_q

        pref_jnt  = env.model.body("prefab").jntadr[0]
        qpos_addr = env.model.jnt_qposadr[pref_jnt]
        dof_idx   = env.model.jnt_dofadr[pref_jnt]

        xy_range = override_init_xy_range if override_init_xy_range is not None \
                   else float(phase_cfg.get("init_xy_range", 0.01))
        noise_xy = rng.uniform(-xy_range, xy_range, 2)
        env.data.qpos[qpos_addr]   = start_xy[0] + noise_xy[0]
        env.data.qpos[qpos_addr+1] = start_xy[1] + noise_xy[1]
        env.data.qpos[qpos_addr+2] = init_pref_z   # [v13.0] 低空起步
        env.data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]
        env.data.qvel[dof_idx:dof_idx+6] = 0.0

        has_viewer = (getattr(env, 'render_mode', False)
                      and getattr(env, 'viewer', None) is not None)
        prefab_hold_xy = env.data.qpos[qpos_addr:qpos_addr+2].copy()
        for _ in range(60):
            env.data.qpos[:7] = init_q
            env.data.qvel[:7] = 0.0
            env.data.qpos[qpos_addr]   = prefab_hold_xy[0]
            env.data.qpos[qpos_addr+1] = prefab_hold_xy[1]
            env.data.qpos[qpos_addr+2] = init_pref_z   # [v13.0]
            env.data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]
            env.data.qvel[dof_idx:dof_idx+6] = 0.0
            mujoco.mj_step(env.model, env.data)
            if has_viewer: env.viewer.sync()
        mujoco.mj_forward(env.model, env.data)

        vel_range = override_init_vel_range if override_init_vel_range is not None \
                    else float(phase_cfg.get("init_vel_range", 0.0))
        if vel_range > 0:
            env.data.qvel[dof_idx:dof_idx+2] += rng.uniform(-vel_range, vel_range, 2)
        mujoco.mj_forward(env.model, env.data)
        _sync_env_internal_state(env)
        if hasattr(env, "reset_vision_state"):
            env.reset_vision_state()
        obs = env._get_obs()

    elif phase == "descent":
        z_cruise = float(config["planning"]["payload_z_cruise"])
        target_xy = env.target_pos.copy()
        rope_L = float(config["controller"].get("L", 0.5))
        ee_z   = z_cruise + rope_L
        seed_q = np.array(config["reset"]["init_qpos_arm"], np.float64)
        init_q = env.ik_solver.solve_4d(seed_q,
            float(target_xy[0]), float(target_xy[1]), ee_z, 0.0)
        if init_q is None or np.any(np.isnan(init_q)):
            init_q = seed_q.copy()

        env.data.qpos[:7] = init_q
        env.data.qvel[:7] = 0.0
        env.data.ctrl[:7] = init_q

        pref_jnt  = env.model.body("prefab").jntadr[0]
        qpos_addr = env.model.jnt_qposadr[pref_jnt]
        dof_idx   = env.model.jnt_dofadr[pref_jnt]

        xy_range = override_init_xy_range if override_init_xy_range is not None \
                   else float(phase_cfg.get("init_xy_range", 0.010))
        noise_xy = rng.uniform(-xy_range, xy_range, 2)
        env.data.qpos[qpos_addr]   = target_xy[0] + noise_xy[0]
        env.data.qpos[qpos_addr+1] = target_xy[1] + noise_xy[1]
        env.data.qpos[qpos_addr+2] = z_cruise
        env.data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]
        env.data.qvel[dof_idx:dof_idx+6] = 0.0

        has_viewer = (getattr(env, 'render_mode', False)
                      and getattr(env, 'viewer', None) is not None)
        prefab_hold_xy2 = env.data.qpos[qpos_addr:qpos_addr+2].copy()
        for _ in range(60):
            env.data.qpos[:7] = init_q
            env.data.qvel[:7] = 0.0
            env.data.qpos[qpos_addr]   = prefab_hold_xy2[0]
            env.data.qpos[qpos_addr+1] = prefab_hold_xy2[1]
            env.data.qpos[qpos_addr+2] = z_cruise
            env.data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]
            env.data.qvel[dof_idx:dof_idx+6] = 0.0
            mujoco.mj_step(env.model, env.data)
            if has_viewer: env.viewer.sync()
        mujoco.mj_forward(env.model, env.data)

        vel_range = override_init_vel_range if override_init_vel_range is not None \
                    else float(phase_cfg.get("init_vel_range", 0.0))
        if vel_range > 0:
            env.data.qvel[dof_idx:dof_idx+2] += rng.uniform(-vel_range, vel_range, 2)

        tilt_range = override_init_tilt_range if override_init_tilt_range is not None \
                     else float(phase_cfg.get("init_tilt_range", 0.005))
        if tilt_range > 0:
            tilt_noise = rng.uniform(-tilt_range, tilt_range, 2)
            env.data.qpos[qpos_addr+3] = 1.0
            env.data.qpos[qpos_addr+4] = tilt_noise[0] * 0.5
            env.data.qpos[qpos_addr+5] = tilt_noise[1] * 0.5
            env.data.qpos[qpos_addr+6] = 0.0
            qnorm = np.linalg.norm(env.data.qpos[qpos_addr+3:qpos_addr+7])
            env.data.qpos[qpos_addr+3:qpos_addr+7] /= qnorm
        mujoco.mj_forward(env.model, env.data)
        _sync_env_internal_state(env)
        if hasattr(env, "reset_vision_state"):
            env.reset_vision_state()
        obs = env._get_obs()

    else:
        raise ValueError(f"Unknown phase: {phase}")

    return obs, planned_path


def reset_for_descent_with_cur(env, config, cur):
    """Descent 段 reset, 含课程注入 + OmniReset 概率。"""
    cur_cfg = config.get("curriculum", {})
    omnireset_enabled = bool(cur_cfg.get("omnireset_enabled", True))
    near_prob = float(cur_cfg.get("omnireset_near_goal_prob", 0.10))

    desc_init = cur.get_descent_init() if cur is not None else None

    if omnireset_enabled and np.random.rand() < near_prob:
        near_xy       = float(cur_cfg.get("omnireset_near_goal_xy", 0.012))
        return reset_for_phase(env, "descent", config,
            override_init_xy_range=near_xy,
            override_init_vel_range=0.005,
            override_init_tilt_range=0.001)
    if desc_init is not None:
        return reset_for_phase(env, "descent", config,
            override_init_xy_range=desc_init["xy_range"],
            override_init_vel_range=desc_init["vel_range"],
            override_init_tilt_range=desc_init["tilt_range"])
    return reset_for_phase(env, "descent", config)


# ==============================================================================
# 专家辅助
# ==============================================================================

def _advance_expert_to_nearest_wp(expert, planned_path, pl_pos):
    if planned_path is None or len(planned_path) == 0:
        return
    dists = [np.linalg.norm(pl_pos - wp) for wp in planned_path]
    nearest_idx = int(np.argmin(dists))
    expert.tracker.current_idx = nearest_idx


def collect_expert_acc(expert, env, obs, current_q, phase, config):
    """收集专家 EE 加速度 (用于 BC 标签 [descent] 或 SAC warmup [cruise/descent])。"""
    if phase == "cruise":
        if bool(config.get("cruise_rl", {}).get("nmpc_residual_mode", True)):
            return np.zeros(int(config["cruise_rl"].get("action_dim", 3)),
                            dtype=np.float32)
        action_4d = expert.tracker.compute_ee_acceleration(obs, target_yaw=0.0)
        acc_max_xy = float(config["ee_control"].get("acc_max_xy", 0.8))
        return np.clip(
            np.array([float(action_4d[0]), float(action_4d[1])], dtype=np.float32),
            -acc_max_xy, acc_max_xy)
    elif phase == "descent":
        target_xy = env.target_pos.copy()
        target_pz = float(config["insertion"]["target_payload_z"])
        pl_pos = env.data.body('prefab').xpos.copy()
        pl_xy  = pl_pos[:2]; pl_z = float(pl_pos[2])
        dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
        pl_vel  = env.data.qvel[dof_idx:dof_idx+3].copy()
        acc_max_xy = float(config["descent_rl"].get(
            "residual_acc_max_xy", config["descent_rl"].get("acc_max_xy", 0.5)))
        acc_max_z  = float(config["descent_rl"].get(
            "residual_acc_max_z", config["descent_rl"].get("acc_max_z",  1.0)))
        diff_xy = target_xy - pl_xy
        dist_xy = float(np.linalg.norm(diff_xy))
        if dist_xy > 0.002:
            dir_xy   = diff_xy / dist_xy
            vel_proj = float(np.dot(pl_vel[:2], dir_xy))
            acc_mag  = 0.6 * min(dist_xy, 0.05) / 0.05 - 0.8 * vel_proj
            acc_xy   = dir_xy * np.clip(acc_mag, -1.0, 1.0) * acc_max_xy
        else:
            acc_xy = -pl_vel[:2] * 1.5
        acc_xy = np.clip(acc_xy, -acc_max_xy, acc_max_xy)
        align_factor = float(np.exp(-dist_xy / 0.01))
        z_error = pl_z - target_pz
        if z_error > 0.005 and align_factor > 0.3:
            acc_z = -(0.5 * min(z_error, 0.15) + 0.3 * max(float(pl_vel[2]), 0)) * align_factor
        else:
            acc_z = -float(pl_vel[2]) * 1.0
        acc_z = np.clip(acc_z, -acc_max_z, acc_max_z)
        return np.array([acc_xy[0], acc_xy[1], acc_z], dtype=np.float32)
    return np.zeros(config[f"{phase}_rl"]["action_dim"], dtype=np.float32)


# ==============================================================================
# Descent: PID base + RL residual delta_q 合并
# ==============================================================================

def _apply_descent_pid_residual(expert, rl_act, obs, env, config, current_q,
                                pid_dq=None):
    """Descent 段 PID + RL residual 动作合并.

    [v12.3 关键修复] 修复 RL authority 随 PID 收敛而消失的 bug:

    旧逻辑 (有 bug):
        max_residual_norm = residual_dq_scale * max(|pid_dq|, dq_max_avg * 0.1)
        # 当 pid_dq 很小 (PID 收敛到稳态时), max_residual_norm 也很小
        # → RL 残差只有 pid_dq 的 30%, 即 0.3-1mm
        # 但 descent 最后 1-2mm 的精度修正恰恰需要 RL 介入
        # → RL 没有"力量"做最后修正 → SR 卡在课程过渡处

    新逻辑 (v12.3):
        max_residual_norm = residual_dq_scale * dq_max_avg
        # 让 RL 残差 始终 有充分 authority (默认 30% × 0.12 = 0.036, 约 36mm)
        # 这样 RL 残差能在任何时候做出 ~5mm 级别的精度修正
        # 不会因 PID 收敛而 RL "失去能量"

    来源:
        Ankile et al. 2024 (ResiP, arXiv:2407.16677): residual 应有恒定且
        足够的 authority, 不能与 base 输出耦合.
        Alakuijala 2021 (arXiv:2106.08050): residual scale 应独立设定.
    """
    if pid_dq is None:
        try:
            pid_dq = expert.compute_delta_q_target(obs, current_q.astype(np.float64))
        except Exception:
            pid_dq = np.zeros(7, dtype=np.float32)
    pid_dq = np.asarray(pid_dq, dtype=np.float32)

    residual_dq_scale = float(config["descent_rl"].get("residual_dq_scale", 0.30))
    dq_max     = np.array(config["space"].get("dq_max", [0.12]*7), dtype=np.float32)
    acc_max_xy = float(config["descent_rl"].get(
        "residual_acc_max_xy", config["descent_rl"].get("acc_max_xy", 0.20)))
    acc_max_z  = float(config["descent_rl"].get(
        "residual_acc_max_z",  config["descent_rl"].get("acc_max_z",  0.40)))

    # rl_act 来自 actor, 范围 [-acc_max_xy, +acc_max_xy] (xy) 和 [-acc_max_z, +acc_max_z] (z)
    rl_norm_xy = rl_act[:2] / max(acc_max_xy, 1e-6)   # 归一化到 [-1, 1]
    rl_norm_z  = float(rl_act[2]) / max(acc_max_z, 1e-6) if len(rl_act) > 2 else 0.0

    # [v12.3 修复] max_residual_norm 与 PID 输出解耦
    # 旧: max_residual_norm = residual_dq_scale * max(|pid_dq|, dq_max_avg * 0.1)
    #     → 当 PID 收敛 (|pid_dq| 小) 时, max_residual_norm 也变小, RL 失去 authority
    # 新: max_residual_norm = residual_dq_scale * dq_max_avg
    #     → RL authority 与 PID 输出无关, 始终有足够"力量"做精度修正
    dq_max_avg = float(np.mean(dq_max))                          # 默认 0.12
    max_residual_norm = residual_dq_scale * dq_max_avg           # 默认 0.036

    rl_dq = np.zeros(7, dtype=np.float32)
    rl_dq[0] = rl_norm_xy[0] * max_residual_norm / np.sqrt(3)
    rl_dq[1] = rl_norm_xy[1] * max_residual_norm / np.sqrt(3)
    rl_dq[2] = rl_norm_z      * max_residual_norm / np.sqrt(3)
    rl_dq = np.clip(rl_dq, -dq_max * residual_dq_scale, dq_max * residual_dq_scale)

    dq_total = np.clip(pid_dq + rl_dq, -dq_max, dq_max)
    return dq_total, pid_dq


# ==============================================================================
# 噪声注入辅助
# ==============================================================================

def _add_obs_noise(norm_obs, sigma):
    if sigma <= 0: return norm_obs
    return (norm_obs + np.random.normal(0, sigma, norm_obs.shape).astype(np.float32))


def _delay_mdp_enabled(config):
    return bool(config.get("delay_mdp", {}).get("enabled", False))


class DelayMDPObservationState:
    """Held-observation state for delay-robust PPO without a predictor."""
    def __init__(self, config):
        cfg = config.get("delay_mdp", {})
        self.enabled = bool(cfg.get("enabled", False))
        self.period = max(1, int(cfg.get("measurement_period_steps", 2)))
        self.warmup_true_steps = max(0, int(cfg.get("warmup_true_steps", 0)))
        self.include_obs_age = bool(cfg.get("include_obs_age", True))
        self.action_history_steps = max(0, int(cfg.get("action_history_steps", 4)))
        self.action_dim = max(0, int(cfg.get("action_dim", 7)))
        dq_max = np.asarray(config.get("space", {}).get("dq_max", [0.12] * 7),
                            dtype=np.float32).reshape(-1)
        if dq_max.size < self.action_dim:
            dq_max = np.pad(dq_max, (0, self.action_dim - dq_max.size),
                            constant_values=float(np.mean(dq_max)) if dq_max.size else 1.0)
        self.action_scale = np.maximum(dq_max[:self.action_dim], 1e-6)
        self.action_history = deque(maxlen=self.action_history_steps)
        self.visible_obs = None
        self.obs_age_steps = 0

    def reset(self, true_obs):
        if true_obs is not None:
            self.visible_obs = np.asarray(true_obs, dtype=np.float32).reshape(-1).copy()
        else:
            self.visible_obs = None
        self.obs_age_steps = 0
        self.action_history.clear()

    def feature_dim(self):
        return delay_mdp_extra_dim({"delay_mdp": {
            "enabled": self.enabled,
            "include_obs_age": self.include_obs_age,
            "action_history_steps": self.action_history_steps,
            "action_dim": self.action_dim,
        }})

    def features(self):
        if not self.enabled:
            return np.zeros(0, dtype=np.float32)
        parts = []
        if self.include_obs_age:
            denom = max(self.period - 1, 1)
            parts.append(np.array([
                min(float(self.obs_age_steps) / float(denom), 1.0)
            ], dtype=np.float32))
        if self.action_history_steps > 0 and self.action_dim > 0:
            seq = list(self.action_history)
            while len(seq) < self.action_history_steps:
                seq.insert(0, np.zeros(self.action_dim, dtype=np.float32))
            parts.append(np.concatenate(seq, axis=0).astype(np.float32))
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts, axis=0).astype(np.float32)

    def append_features(self, encoded_obs):
        encoded_obs = np.asarray(encoded_obs, dtype=np.float32).reshape(-1)
        if not self.enabled:
            return encoded_obs
        return np.concatenate([encoded_obs, self.features()]).astype(np.float32)

    def record_action(self, dq_cmd):
        if not self.enabled or self.action_history_steps <= 0 or self.action_dim <= 0:
            return
        dq = np.asarray(dq_cmd if dq_cmd is not None else [], dtype=np.float32).reshape(-1)
        if dq.size < self.action_dim:
            dq = np.pad(dq, (0, self.action_dim - dq.size))
        dq = dq[:self.action_dim]
        dq_norm = np.clip(dq / self.action_scale, -2.0, 2.0).astype(np.float32)
        self.action_history.append(dq_norm)

    def observe_result(self, true_next_obs, next_step_index):
        true_next_obs = np.asarray(true_next_obs, dtype=np.float32).reshape(-1)
        if not self.enabled:
            self.visible_obs = true_next_obs.copy()
            self.obs_age_steps = 0
            return true_next_obs.copy(), False
        has_measurement = (
            next_step_index <= self.warmup_true_steps or
            self.period <= 1 or
            int(next_step_index) % self.period == 0
        )
        if has_measurement or self.visible_obs is None:
            self.visible_obs = true_next_obs.copy()
            self.obs_age_steps = 0
            return self.visible_obs.copy(), False
        self.obs_age_steps += 1
        return self.visible_obs.copy(), True


class AdaptationHistoryState:
    """Deployable action-history features for no-cable hidden-dynamics inference."""
    def __init__(self, config):
        cfg = config.get("adaptation_history", {})
        self.enabled = bool(cfg.get("enabled", False))
        self.action_history_steps = max(0, int(cfg.get("action_history_steps", 8)))
        self.action_dim = max(0, int(cfg.get("action_dim", 7)))
        dq_max = np.asarray(config.get("space", {}).get("dq_max", [0.12] * 7),
                            dtype=np.float32).reshape(-1)
        if dq_max.size < self.action_dim:
            dq_max = np.pad(dq_max, (0, self.action_dim - dq_max.size),
                            constant_values=float(np.mean(dq_max)) if dq_max.size else 1.0)
        self.action_scale = np.maximum(dq_max[:self.action_dim], 1e-6)
        self.action_history = deque(maxlen=self.action_history_steps)

    def reset(self):
        self.action_history.clear()

    def feature_dim(self):
        return adaptation_history_extra_dim({"adaptation_history": {
            "enabled": self.enabled,
            "action_history_steps": self.action_history_steps,
            "action_dim": self.action_dim,
        }})

    def features(self):
        if not self.enabled or self.action_history_steps <= 0 or self.action_dim <= 0:
            return np.zeros(0, dtype=np.float32)
        seq = list(self.action_history)
        while len(seq) < self.action_history_steps:
            seq.insert(0, np.zeros(self.action_dim, dtype=np.float32))
        return np.concatenate(seq, axis=0).astype(np.float32)

    def append_features(self, encoded_obs):
        encoded_obs = np.asarray(encoded_obs, dtype=np.float32).reshape(-1)
        if not self.enabled:
            return encoded_obs
        return np.concatenate([encoded_obs, self.features()]).astype(np.float32)

    def record_action(self, dq_cmd):
        if not self.enabled or self.action_history_steps <= 0 or self.action_dim <= 0:
            return
        dq = np.asarray(dq_cmd if dq_cmd is not None else [], dtype=np.float32).reshape(-1)
        if dq.size < self.action_dim:
            dq = np.pad(dq, (0, self.action_dim - dq.size))
        dq = dq[:self.action_dim]
        dq_norm = np.clip(dq / self.action_scale, -2.0, 2.0).astype(np.float32)
        self.action_history.append(dq_norm)


def _add_act_noise(dq, sigma):
    if sigma <= 0: return dq
    return (dq + np.random.normal(0, sigma, dq.shape).astype(dq.dtype))


def _apply_episode_wind_env(env, pert):
    ws = float(pert.get("wind_speed", 0.0))
    wd = float(pert.get("wind_dir", 0.0))
    if ws > 0.0 and hasattr(env, 'set_wind_speed'):
        env.set_wind_speed(ws, wd)
    else:
        if hasattr(env, 'clear_wind'):
            env.clear_wind()
        else:
            try: env.set_wind_curriculum(0.0)
            except Exception: pass


def _apply_episode_wind_vec(vec, idx, pert):
    ws = float(pert.get("wind_speed", 0.0))
    wd = float(pert.get("wind_dir", 0.0))
    if ws > 0.0 and hasattr(vec, 'set_wind_speed'):
        vec.set_wind_speed(idx, ws, wd)
    else:
        if hasattr(vec, 'clear_wind'):
            vec.clear_wind(idx)
        else:
            try: vec.set_wind_curriculum(idx, 0.0)
            except Exception: pass


# ==============================================================================
# PPO 训练
# ==============================================================================

def make_phase_env_and_controllers(phase, config, worker_id=0):
    """[v11 Path 3] 工厂函数: 构造 env + 所有 base controllers (按 phase).

    供单进程 train_ppo / SubprocVecEnv worker 共用. 每次调用返回独立实例.

    Args:
        phase:     "cruise" / "descent"
        config:    完整 config dict (deepcopy 后)
        worker_id: 用于 seed 偏移 (0 = main, >0 = subproc worker)
    Returns:
        (env, controllers_dict, config) — controllers 中含 'phase' 键, worker 可读
    """
    import copy as _copy
    cfg = _copy.deepcopy(config) if worker_id > 0 else config
    if worker_id > 0:
        cfg["train"]["seed"] = int(cfg["train"].get("seed", 42)) + worker_id * 1000
        set_global_seed(cfg["train"]["seed"])
    env = CableRobotEnvWithObstacles(config=cfg)
    controllers = {
        "phase":   phase,                       # [v11 fix] worker 读取 phase 用
        "expert":  JointSpaceExpert(cfg, env.ik_solver),
        "ee_ctrl": EEAccController(cfg, env.ik_solver),
        "z_pid":   CruiseZYawPID(cfg)          if phase == "cruise" else None,
        "swing_d": SwingDampingController(cfg) if phase == "cruise" else None,
    }
    return env, controllers, cfg


def train_ppo(phase, log_dir, config, resume_ckpt=None):
    """PPO 训练入口. [v11] n_envs > 1 自动 dispatch 到 train_ppo_vec."""
    n_envs = int(config["train"].get("n_envs", 1))
    if n_envs > 1:
        # [v11 Path 3] 多环境并行
        return train_ppo_vec(phase, log_dir, config, resume_ckpt=resume_ckpt,
                             n_envs=n_envs)
    return _train_ppo_single(phase, log_dir, config, resume_ckpt=resume_ckpt)


def _make_obs_pred_pretrain_curriculum(config, phase):
    cur = CurriculumManager(config, phase)
    cfg = config.get("observation_predictor", {})
    if bool(cfg.get("pretrain_final_curriculum", True)) and cur.levels:
        cur.level_idx = len(cur.levels) - 1
        cur._wind_cur = cur._wind_max
    return cur


def _apply_pretrain_wind_floor(config, cur, pert):
    cfg = config.get("observation_predictor", {})
    wind_min = max(0.0, float(cfg.get("pretrain_wind_min", 0.0)))
    wind_max = max(wind_min, float(getattr(cur, "wind_current", 0.0)))
    if wind_min <= 0.0 or wind_max <= 0.0:
        return pert
    wd = float(pert.get("wind_dir", np.random.uniform(0.0, 2 * np.pi)))
    ws = float(np.random.uniform(min(wind_min, wind_max), wind_max))
    pert["wind_min"] = min(wind_min, wind_max)
    pert["wind_max"] = wind_max
    pert["wind_speed"] = ws
    pert["wind_dir"] = wd
    return pert


def train_obs_predictor_pretrain(phase, log_dir, config, resume_ckpt=None,
                                 n_envs=4):
    """Pretrain only the delayed-observation predictor with a frozen policy."""
    from vec_env import make_vec_env
    T = int(config["train"].get("total_timesteps", 300_000))
    SI = int(config["train"].get("save_interval", 100))
    start_method = config["train"].get("vec_env_start_method", "forkserver")
    pred_cfg = config.setdefault("observation_predictor", {})
    pred_cfg["enabled"] = True
    pred_cfg["train_enabled"] = True
    deterministic_policy = bool(pred_cfg.get(
        "pretrain_policy_deterministic", True))

    print(f"\n{'='*60}\n  ObsPredictor PRETRAIN [VEC n_envs={n_envs}] | "
          f"{phase.upper()} | {T} steps | {log_dir}\n{'='*60}\n")
    if not resume_ckpt or not os.path.exists(resume_ckpt):
        raise ValueError("--resume-ckpt is required for obs_pred pretraining")

    agent = PPOPhaseAgent(phase, config=config)
    agent.load(resume_ckpt)
    print(f"  [Policy] frozen residual policy loaded from {resume_ckpt}")

    def _make_one(wid):
        return make_phase_env_and_controllers(phase, config, worker_id=wid)

    vec = make_vec_env(_make_one, n_envs=n_envs, start_method=start_method)
    shared_cur = _make_obs_pred_pretrain_curriculum(config, phase)
    curs = [shared_cur for _ in range(n_envs)]
    obs_list = [None] * n_envs
    sxy_list = [None] * n_envs
    txy_list = [None] * n_envs
    rstate_list = [None] * n_envs
    perts = [None] * n_envs
    pt_list = [0.0] * n_envs
    py_list = [0.0] * n_envs
    ep_rewards = [0.0] * n_envs
    ep_steps = [0] * n_envs
    ep_suc = [False] * n_envs
    ep_term = ["running"] * n_envs
    obs_histories = [agent.make_obs_history() for _ in range(n_envs)]
    cable_histories = [agent.make_cable_history() for _ in range(n_envs)]

    def _reset_one_env(i, max_retries=10):
        for _retry in range(max_retries):
            pert = curs[i].sample_episode_perturbations()
            pert = _apply_pretrain_wind_floor(config, curs[i], pert)
            if phase == "descent":
                _di = curs[i].get_descent_init()
                obs, pp = vec.reset_for_descent(i, cur_init=_di)
            else:
                obs, pp = vec.reset(i)
            if obs is not None:
                break
        else:
            raise RuntimeError(f"Worker {i}: obs predictor reset failed")
        vec.set_force_noise(i, pert["force_noise"])
        _apply_episode_wind_vec(vec, i, pert)
        cq = vec.get_qpos(i)
        vec.reset_controllers(i, obs, cq, pp)
        sxy = None
        txy = np.asarray(vec.env_attr(i, "target_pos"), np.float32)[:2]
        rs = REWARD_STATES[phase]()
        if phase == "descent":
            _di = curs[i].get_descent_init()
            if _di is not None:
                rs.current_xy_range = _di["xy_range"]
                rs.current_xy_tol = _di["xy_tol"]
                rs.current_descent_level = curs[i].level_idx
                rs.descent_n_levels = curs[i].n_levels
        return obs, sxy, txy, rs, pert

    try:
        for i in range(n_envs):
            obs_list[i], sxy_list[i], txy_list[i], rstate_list[i], perts[i] = \
                _reset_one_env(i)
            obs_histories[i].reset()
            cable_histories[i].clear()

        obs_dim = _obs_pred_target_dim(config, obs_list[0], agent)
        obs_predictors = [
            build_observation_predictor(
                config, phase, obs_dim, device=getattr(agent, "device", None))
            for _ in range(n_envs)
        ]
        first_pred = next((p for p in obs_predictors if p is not None), None)
        if first_pred is None:
            raise RuntimeError("observation predictor did not build")
        print(f"  [ObsPredictor-PRETRAIN] true obs every "
              f"{first_pred.measurement_period} control steps, obs_dim={obs_dim}, "
              f"target_mode={_obs_pred_target_mode(config)}, predictors={n_envs}")
        _try_load_obs_predictor_checkpoint(
            obs_predictors, config, None, label="ObsPredictor-PRETRAIN")
        for i, p in enumerate(obs_predictors):
            if p is not None:
                p.reset(obs_list[i])

        logger = Logger(log_dir, project="obs_predictor_pretrain",
                        run_name=f"{phase}_obs_pred_pretrain")
        logger.update_config(config)
        lf = os.path.join(log_dir, f"{phase}_obs_pred_pretrain_log.csv")
        _init_csv_log(lf, [
            "episode", "total_steps", "ep_reward", "success", "steps",
            "pred_loss", "pred_nll", "pred_raw_nll", "pred_huber",
            "pred_rmse_norm",
            "pred_rmse_raw", "pred_rmse_non_cable", "pred_rmse_cable_latent",
            "pred_rmse_non_cable_norm", "pred_rmse_cable_latent_norm",
            "pred_log_std_mean", "pred_log_std_min", "pred_std_mean",
            "hidden_frac", "lvl", "wind_speed", "wind_speed_min",
            "wind_speed_max", "worker_id", "termination",
        ], append_existing=False)

        phase_obs_cache = [None] * n_envs
        for i in range(n_envs):
            phase_obs_cache[i] = vec.build_phase_obs_remote(
                i, phase, obs_list[i], sxy_list[i], txy_list[i],
                pt_list[i], py_list[i])

        ts = 0
        ep_count = 0
        best_rmse = float("inf")
        t0 = time.time()
        window_loss = deque(maxlen=500)
        window_rmse = deque(maxlen=500)
        window_hidden = deque(maxlen=500)
        pred_acc = [
            {"loss": 0.0, "nll": 0.0, "raw_nll": 0.0, "huber": 0.0,
             "rmse_norm": 0.0,
             "rmse_raw": 0.0, "rmse_non_cable": 0.0,
             "rmse_cable_latent": 0.0, "rmse_non_cable_norm": 0.0,
             "rmse_cable_latent_norm": 0.0,
             "log_std_mean": 0.0, "log_std_min": 0.0, "std_mean": 0.0,
             "updates": 0, "hidden": 0, "steps": 0}
            for _ in range(n_envs)
        ]

        while ts < T:
            actions_list = []
            for i in range(n_envs):
                _core_i, _cable_i, _wind_i, _, _, _base_i = phase_obs_cache[i]
                po = agent.encode_obs(_core_i, _cable_i, _wind_i)
                no = agent.normalize_obs(po, update=False)
                act, _lp, _val = agent.act_with_history(
                    no, obs_histories[i], deterministic=deterministic_policy)
                cable_histories[i].append(_cable_i.copy())
                actions_list.append(act)

            payloads = []
            for i in range(n_envs):
                payloads.append({
                    'phase': phase,
                    'rl_action': actions_list[i],
                    'obs': obs_list[i],
                    'current_q': vec.get_qpos(i),
                    'start_xy': sxy_list[i],
                    'target_xy': txy_list[i],
                    'prev_tilt': pt_list[i],
                    'prev_yaw': py_list[i],
                    'rstate': rstate_list[i],
                    'act_noise': perts[i]["act_noise"],
                    'base_dq': phase_obs_cache[i][5],
                    'train_reject_lucky_rebar_insert':
                        _lucky_reject_enabled(config),
                })

            if hasattr(vec, 'remotes'):
                for i, p in enumerate(payloads):
                    vec.remotes[i].send(('rl_step', p))
                results = [vec._check_recv(vec.remotes[i].recv(), i)
                           for i in range(n_envs)]
            else:
                results = [vec.rl_step(i, payloads[i]) for i in range(n_envs)]

            for i, res in enumerate(results):
                true_next_obs = res['new_obs']
                obs_pred = obs_predictors[i]
                if obs_pred is not None:
                    dq_cmd = res.get('delta_q', None)
                    if dq_cmd is None:
                        dq_cmd = np.zeros(obs_pred.action_dim, dtype=np.float32)
                    current_target = _obs_pred_target_vector(
                        config, obs_list[i], phase_obs_cache[i], agent)
                    true_target = _obs_pred_target_vector(
                        config, true_next_obs, None, agent)
                    obs_pred.predict_next(current_target, dq_cmd, phase)
                    _visible_unused, info = obs_pred.observe_result(
                        true_target, next_step_index=ep_steps[i] + 1)
                    pred_acc[i]["steps"] += 1
                    pred_acc[i]["hidden"] += int(info.used_prediction)
                    if info.trained:
                        pred_acc[i]["loss"] += info.loss
                        pred_acc[i]["nll"] += info.nll
                        pred_acc[i]["raw_nll"] += info.raw_nll
                        pred_acc[i]["huber"] += info.huber
                        pred_acc[i]["rmse_norm"] += info.rmse_norm
                        pred_acc[i]["rmse_raw"] += info.rmse_raw
                        pred_acc[i]["rmse_non_cable"] += info.rmse_non_cable
                        pred_acc[i]["rmse_cable_latent"] += info.rmse_cable_latent
                        pred_acc[i]["rmse_non_cable_norm"] += (
                            info.rmse_non_cable_norm)
                        pred_acc[i]["rmse_cable_latent_norm"] += (
                            info.rmse_cable_latent_norm)
                        pred_acc[i]["log_std_mean"] += info.log_std_mean
                        pred_acc[i]["log_std_min"] += info.log_std_min
                        pred_acc[i]["std_mean"] += info.std_mean
                        pred_acc[i]["updates"] += 1
                        window_loss.append(info.loss)
                        window_rmse.append(info.rmse_norm)
                    window_hidden.append(float(info.used_prediction))

                # The frozen policy always receives the true simulator obs.
                obs_list[i] = true_next_obs
                rstate_list[i] = res['rstate']
                done = bool(res['done'])
                if res.get('info', {}).get('nan_detected', False):
                    done = True
                mx = int(config[f"{phase}_rl"]["max_steps"])
                if ep_steps[i] >= mx - 1:
                    done = True
                    if not res['success']:
                        ep_term[i] = "timeout"
                elif done:
                    ep_term[i] = res.get('termination', 'done')
                ep_suc[i] = ep_suc[i] or bool(res['success'])
                ep_rewards[i] += float(res['reward'])
                ep_steps[i] += 1
                ts += 1

                next_phase_obs = None
                if not done:
                    if res.get('new_core_obs') is not None:
                        next_phase_obs = (
                            res['new_core_obs'], res['new_cable_raw'],
                            res['new_wind_obs'], res['new_tilt'],
                            res['new_yaw'], res.get('new_base_dq'))
                    else:
                        next_phase_obs = vec.build_phase_obs_remote(
                            i, phase, obs_list[i], sxy_list[i], txy_list[i],
                            pt_list[i], py_list[i])
                if not done and next_phase_obs is not None:
                    phase_obs_cache[i] = next_phase_obs
                    pt_list[i] = next_phase_obs[3]
                    py_list[i] = next_phase_obs[4]

                if done or ts >= T:
                    _acc = pred_acc[i]
                    den = max(int(_acc["updates"]), 1)
                    step_den = max(int(_acc["steps"]), 1)
                    pred_loss = _acc["loss"] / den
                    pred_rmse = _acc["rmse_norm"] / den
                    hidden_frac = _acc["hidden"] / step_den
                    ep_count += 1
                    curs[i].update(ep_suc[i])
                    with open(lf, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([
                            ep_count, ts, f"{ep_rewards[i]:.3f}",
                            int(ep_suc[i]), ep_steps[i],
                            f"{pred_loss:.6f}",
                            f"{(_acc['nll'] / den):.6f}",
                            f"{(_acc['raw_nll'] / den):.6f}",
                            f"{(_acc['huber'] / den):.6f}",
                            f"{pred_rmse:.6f}",
                            f"{(_acc['rmse_raw'] / den):.6f}",
                            f"{(_acc['rmse_non_cable'] / den):.6f}",
                            f"{(_acc['rmse_cable_latent'] / den):.6f}",
                            f"{(_acc['rmse_non_cable_norm'] / den):.6f}",
                            f"{(_acc['rmse_cable_latent_norm'] / den):.6f}",
                            f"{(_acc['log_std_mean'] / den):.6f}",
                            f"{(_acc['log_std_min'] / den):.6f}",
                            f"{(_acc['std_mean'] / den):.6f}",
                            f"{hidden_frac:.3f}", curs[i].level_idx,
                            f"{perts[i].get('wind_speed', 0.0):.4f}",
                            f"{perts[i].get('wind_min', curs[i].wind_sample_min):.4f}",
                            f"{perts[i].get('wind_max', curs[i].wind_current):.4f}",
                            i, ep_term[i],
                        ])
                    metrics = {
                        f"obs_pred_pretrain/{phase}/loss_ep": pred_loss,
                        f"obs_pred_pretrain/{phase}/nll_ep":
                            _acc["nll"] / den,
                        f"obs_pred_pretrain/{phase}/raw_nll_ep":
                            _acc["raw_nll"] / den,
                        f"obs_pred_pretrain/{phase}/huber_ep":
                            _acc["huber"] / den,
                        f"obs_pred_pretrain/{phase}/rmse_norm_ep": pred_rmse,
                        f"obs_pred_pretrain/{phase}/rmse_raw_ep":
                            _acc["rmse_raw"] / den,
                        f"obs_pred_pretrain/{phase}/rmse_non_cable_ep":
                            _acc["rmse_non_cable"] / den,
                        f"obs_pred_pretrain/{phase}/rmse_cable_latent_ep":
                            _acc["rmse_cable_latent"] / den,
                        f"obs_pred_pretrain/{phase}/rmse_non_cable_norm_ep":
                            _acc["rmse_non_cable_norm"] / den,
                        f"obs_pred_pretrain/{phase}/rmse_cable_latent_norm_ep":
                            _acc["rmse_cable_latent_norm"] / den,
                        f"obs_pred_pretrain/{phase}/log_std_mean_ep":
                            _acc["log_std_mean"] / den,
                        f"obs_pred_pretrain/{phase}/log_std_min_ep":
                            _acc["log_std_min"] / den,
                        f"obs_pred_pretrain/{phase}/std_mean_ep":
                            _acc["std_mean"] / den,
                        f"obs_pred_pretrain/{phase}/hidden_frac_ep": hidden_frac,
                        f"obs_pred_pretrain/{phase}/success": float(ep_suc[i]),
                        f"obs_pred_pretrain/{phase}/reward": ep_rewards[i],
                        f"obs_pred_pretrain/{phase}/wind_speed":
                            float(perts[i].get("wind_speed", 0.0)),
                        f"obs_pred_pretrain/{phase}/wind_min":
                            curs[i].wind_sample_min,
                        f"obs_pred_pretrain/{phase}/wind_max": curs[i].wind_current,
                        f"obs_pred_pretrain/{phase}/window_loss":
                            float(np.mean(window_loss)) if window_loss else 0.0,
                        f"obs_pred_pretrain/{phase}/window_rmse_norm":
                            float(np.mean(window_rmse)) if window_rmse else 0.0,
                        f"obs_pred_pretrain/{phase}/window_hidden_frac":
                            float(np.mean(window_hidden)) if window_hidden else 0.0,
                    }
                    logger.log(ts, metrics)
                    if ep_count % 25 == 0:
                        print(f"[ObsPredPretrain] Ep{ep_count:5d} "
                              f"[{ts}] loss:{pred_loss:.4f} "
                              f"rmse:{pred_rmse:.4f} hidden:{hidden_frac:.0%} "
                              f"W:{perts[i].get('wind_speed', 0.0):.2f}m/s "
                              f"term:{ep_term[i]}")
                    if pred_rmse > 0.0 and pred_rmse < best_rmse:
                        best_rmse = pred_rmse
                        _save_obs_predictor_tag(obs_predictors, log_dir, "best")
                    if ep_count > 0 and ep_count % SI == 0:
                        _save_obs_predictor_tag(obs_predictors, log_dir, "latest")

                    obs_list[i], sxy_list[i], txy_list[i], rstate_list[i], perts[i] = \
                        _reset_one_env(i)
                    pt_list[i] = 0.0
                    py_list[i] = 0.0
                    obs_histories[i].reset()
                    cable_histories[i].clear()
                    if obs_predictors[i] is not None:
                        obs_predictors[i].reset(obs_list[i])
                    pred_acc[i] = {
                        "loss": 0.0, "nll": 0.0, "raw_nll": 0.0,
                        "huber": 0.0,
                        "rmse_norm": 0.0, "rmse_raw": 0.0,
                        "rmse_non_cable": 0.0, "rmse_cable_latent": 0.0,
                        "rmse_non_cable_norm": 0.0,
                        "rmse_cable_latent_norm": 0.0,
                        "log_std_mean": 0.0,
                        "log_std_min": 0.0,
                        "std_mean": 0.0,
                        "updates": 0,
                        "hidden": 0, "steps": 0,
                    }
                    ep_rewards[i] = 0.0
                    ep_steps[i] = 0
                    ep_suc[i] = False
                    ep_term[i] = "running"
                    phase_obs_cache[i] = vec.build_phase_obs_remote(
                        i, phase, obs_list[i], sxy_list[i], txy_list[i],
                        pt_list[i], py_list[i])

        _save_obs_predictor_tag(obs_predictors, log_dir, "latest")
        _save_obs_predictor_tag(obs_predictors, log_dir, "final")
        print(f"\n[{phase.upper()}-OBS-PRED] Done: {ts} steps, "
              f"{(time.time()-t0)/60:.1f} min, best_rmse={best_rmse:.4f}")
        logger.close()
        return obs_predictors
    finally:
        try:
            vec.close()
        except Exception:
            pass


def _train_ppo_single(phase, log_dir, config, resume_ckpt=None):
    T  = int(config["train"].get("total_timesteps", 2_000_000))
    SI = int(config["train"]["save_interval"])
    EI = int(config["train"].get("eval_interval", 100))

    print(f"\n{'='*60}\n  PPO | {phase.upper()} | {T} steps | {log_dir}\n{'='*60}\n")

    env    = CableRobotEnvWithObstacles(config=config)
    agent  = PPOPhaseAgent(phase, config=config)
    expert = JointSpaceExpert(config, env.ik_solver)
    ectl   = EEAccController(config, env.ik_solver)
    z_pid   = CruiseZYawPID(config)          if phase == "cruise" else None
    swing_d = SwingDampingController(config) if phase == "cruise" else None
    _cruise_nmpc_base = (phase == "cruise" and
        bool(config.get("cruise_rl", {}).get("use_nmpc_base", False)))
    _cruise_nmpc_residual = (phase == "cruise" and
        bool(config.get("cruise_rl", {}).get(
            "nmpc_residual_mode", _cruise_nmpc_base)))
    _descent_pid_residual = (phase == "descent" and
        bool(config.get("descent_rl", {}).get("pid_residual_mode", True)))
    cur = CurriculumManager(config, phase)

    # [v9] resume_ckpt 用于断点续训, 不再用于加载 BC 权重
    if resume_ckpt and os.path.exists(resume_ckpt):
        agent.load(resume_ckpt); print(f"  Resumed from ckpt: {resume_ckpt}")
        if bool(config.get("train", {}).get("reset_optimizer_on_resume", False)):
            agent.reset_adam_state("reset_optimizer_on_resume")

    obs_predictor_enabled = bool(config.get("observation_predictor", {}).get("enabled", False))
    if obs_predictor_enabled and _delay_mdp_enabled(config):
        raise ValueError("delay_mdp and observation_predictor are mutually exclusive")
    obs_predictor = None
    _obs_pred_resume_loaded = False
    delay_mdp_state = DelayMDPObservationState(config)
    if delay_mdp_state.enabled:
        print(f"  [Delay-MDP] enabled: true obs every "
              f"{delay_mdp_state.period} control steps, "
              f"action_history={delay_mdp_state.action_history_steps}, "
              f"extra_obs_dim={delay_mdp_state.feature_dim()}")

    logger = Logger(log_dir, project=f"phase_rl_v9", run_name=f"{phase}_ppo")
    logger.update_config(config)
    stats = _make_episode_stats(config)
    ep = 0; ts = 0; best = 0.0; t0 = time.time()
    _ppo_update_count = 0

    lf = os.path.join(log_dir, f"{phase}_ppo_log.csv")
    _progress_ep, _progress_ts, _progress_log = (0, 0, None)
    if resume_ckpt:
        _progress_ep, _progress_ts, _progress_log = _read_csv_progress(
            lf, os.path.join(log_dir, f"{phase}_ppo_vec_log.csv"))
        if _progress_log:
            ep = max(ep, _progress_ep)
            ts = max(ts, _progress_ts, int(getattr(agent, "total_steps", 0)))
            agent.total_steps = max(int(getattr(agent, "total_steps", 0)), ts)
            print(f"  Resumed log progress: ep={ep}, ts={ts} from {_progress_log}")
        else:
            ts = max(ts, int(getattr(agent, "total_steps", 0)))
            agent.total_steps = max(int(getattr(agent, "total_steps", 0)), ts)
            if ts > 0:
                print(f"  Resumed checkpoint progress: ts={ts} "
                      f"(new log dir, no CSV to append)")
    T = _resolve_resume_training_target(
        T, ts, resume_ckpt=resume_ckpt, progress_log=_progress_log,
        label=f"{phase}-ppo")
    _ppo_header = ["episode", "total_steps", "ep_reward", "avg_reward",
                   "sr", "steps", "pol_loss", "val_loss", "ent",
                   "lvl", "wind_speed", "wind_speed_min",
                   "wind_speed_max", "cur_sr", "cur_eps"]
    _init_csv_log(lf, _ppo_header, append_existing=(
        resume_ckpt and os.path.abspath(_progress_log or "") == os.path.abspath(lf)))

    while ts < T:
        # ── 每 episode: 采样课程扰动 ───────────────────────────────────────────
        pert = cur.sample_episode_perturbations()

        # ── 物理初始化 ────────────────────────────────────────────────────────
        if phase == "descent":
            obs, pp = reset_for_descent_with_cur(env, config, cur)
        else:
            obs, pp = reset_for_phase(env, phase, config)
        if obs is None:
            continue
        if obs_predictor_enabled and obs_predictor is None:
            obs_dim = _obs_pred_target_dim(config, obs, agent)
            obs_predictor = build_observation_predictor(
                config, phase, obs_dim, device=getattr(agent, "device", None))
            if obs_predictor is not None:
                print(f"  [ObsPredictor] enabled: true obs every "
                      f"{obs_predictor.measurement_period} control steps, "
                      f"obs_dim={obs_dim}, "
                      f"target_mode={_obs_pred_target_mode(config)}")
                if not _obs_pred_resume_loaded:
                    _try_load_obs_predictor_checkpoint(
                        obs_predictor, config, resume_ckpt,
                        label="ObsPredictor")
                    _obs_pred_resume_loaded = True
        if obs_predictor is not None:
            obs_predictor.reset(obs)
        if delay_mdp_state.enabled:
            delay_mdp_state.reset(obs)
        env.set_force_noise(pert["force_noise"])
        _apply_episode_wind_env(env, pert)

        cq = env.data.qpos[:7].copy()
        expert.reset(obs, cq, env=env)
        if pp is not None:
            # [v12.2] lift 时只把"lift 段"喂给 tracker, 防止 look-ahead 跨段拉走 xy
            expert.set_path(pp)
            plp = env.data.body('prefab').xpos.copy()
            _advance_expert_to_nearest_wp(expert, pp, plp)
        ectl.reset(env._get_ee_pos(), cq)

        if z_pid is not None:
            _pl_z   = float(env.data.body('prefab').xpos[2])
            _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
            _pl_yaw = float(R.from_matrix(_pl_mat).as_euler('xyz')[2])
            z_pid.reset(_pl_z, _pl_yaw)

        sxy = np.asarray(getattr(
            env, "episode_start_xy", env.default_start_xy), dtype=np.float32).copy()
        txy = env.target_pos.copy()
        pt, py = 0.0, 0.0

        rs = REWARD_STATES[phase]()
        if hasattr(rs, 'total_steps_global'):
            rs.total_steps_global = ts
        if phase == "descent":
            _di = cur.get_descent_init()
            if _di is not None:
                rs.current_xy_range = _di["xy_range"]
                rs.current_xy_tol   = _di["xy_tol"]
                rs.current_descent_level = cur.level_idx
                rs.descent_n_levels      = cur.n_levels

        if hasattr(agent, 'reset_history'):
            agent.reset_history()

        rew_tracker = RewardComponentTracker(phase)
        # [v12.6] 细粒度 RL 评估指标 tracker
        stab = StabilityMetrics()
        er = 0.0; es = 0; suc = False; term_reason = "running"
        mx = int(config[f"{phase}_rl"]["max_steps"])
        obs_pred_loss_sum = 0.0
        obs_pred_nll_sum = 0.0
        obs_pred_huber_sum = 0.0
        obs_pred_rmse_norm_sum = 0.0
        obs_pred_rmse_raw_sum = 0.0
        obs_pred_rmse_non_cable_sum = 0.0
        obs_pred_rmse_cable_latent_sum = 0.0
        obs_pred_rmse_non_cable_norm_sum = 0.0
        obs_pred_rmse_cable_latent_norm_sum = 0.0
        obs_pred_updates = 0
        obs_pred_hidden_steps = 0
        obs_pred_cable_latent = None
        obs_pred_phase_obs = None
        delay_mdp_hidden_steps = 0
        delay_mdp_total_steps = 0

        # ── Episode 主循环 ───────────────────────────────────────────────────
        # [v14.0] 获取 wind obs (每 episode 更新一次, 因为 wind 可能在 step 间变化)
        _wf_max = wind_obs_scale(config)

        def _env_step_with_obs_predictor(dq_cmd):
            if obs_predictor is not None:
                current_target = _obs_pred_target_vector(
                    config, obs, obs_pred_phase_obs, agent)
                obs_predictor.predict_next(current_target, dq_cmd, phase)
            return env.step(dq_cmd)

        rd = False
        while not rd:
            cq = env.data.qpos[:7].copy().astype(np.float32)
            base_dq_for_obs = None
            if phase == "descent" and _descent_pid_residual:
                try:
                    base_dq_for_obs = expert.compute_delta_q_target(
                        obs, cq.astype(np.float64))
                except Exception:
                    base_dq_for_obs = np.zeros(7, dtype=np.float32)
            elif phase == "cruise" and _cruise_nmpc_residual:
                base_dq_for_obs = get_last_nmpc_action(expert)
            # [v14.0] 构建 obs: (core, cable_raw, wind, tilt, yaw)
            wobs = build_wind_obs(env, _wf_max)
            core, cable_raw, wobs, pt, py = build_phase_obs(
                phase, obs, env, sxy, txy, pt, py, wind_obs=wobs,
                base_action=base_dq_for_obs)
            # [v14.0] 通过 CableEncoder 编码后合并
            po = agent.encode_obs(core, cable_raw, wobs)
            cable_for_policy = (obs_pred_cable_latent
                                if obs_pred_cable_latent is not None
                                else cable_raw)
            obs_pred_phase_obs = (
                core, cable_for_policy, wobs, pt, py, base_dq_for_obs)
            po = agent.encode_obs(core, cable_for_policy, wobs)
            po = delay_mdp_state.append_features(po)
            no = agent.normalize_obs(po, update=True)
            no_noisy = _add_obs_noise(no, pert["obs_noise"])

            ree = env._get_ee_pos()

            # ── Cruise [v13.0 合并 lift]: 根据 payload 高度切换两种模式 ──────
            #   低空 (payload_z < z_cruise - 0.03): lift 模式 — expert.compute_delta_q_target
            #     (单一积分器路径, 与 test 一致, NMPC 自动垂直上升)
            #   高空 (payload_z 接近 z_cruise): cruise 模式 — 原 lock_z + xy 残差
            if phase == "cruise" and _cruise_nmpc_residual:
                act, lp, val = agent.act(
                    no_noisy, deterministic=False)
                _act_arr = np.asarray(act, np.float32)
                _res3 = clip_cruise_residual(_act_arr, config)
                dq = expert.compute_delta_q_target(
                    obs, cq.astype(np.float64), residual_acc=_res3)
                _cruise_reward_base = get_last_nmpc_action(expert)
                dq = _add_act_noise(dq, pert["act_noise"])
                no2, _, _, _, ei = _env_step_with_obs_predictor(dq)
                rw, dn, sc, ri = compute_cruise_reward(
                    env, no2, config, rs, tracker=rew_tracker,
                    rl_action=_res3, base_action=_cruise_reward_base)
                rew_tracker.step()

            elif phase == "cruise" and z_pid is not None:
                _pl_pos  = env.data.body('prefab').xpos
                _dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
                _pl_vz   = float(env.data.qvel[_dof_idx + 2])
                _pl_mat  = env.data.body('prefab').xmat.reshape(3, 3)
                _pl_euler = R.from_matrix(_pl_mat).as_euler('xyz')
                _pl_yaw  = float(_pl_euler[2])
                _pl_yaw_rate = float(env.data.qvel[_dof_idx + 5]) \
                    if _dof_idx + 5 < len(env.data.qvel) else 0.0
                _z_cruise = float(config["cruise_rl"].get("target_z_cruise", 0.25))
                _is_lift_phase = float(_pl_pos[2]) < _z_cruise - 0.03

                # cruise 模式需要 z_pid 反馈, lift 模式不需要 (NMPC 自处理 z)
                if not _is_lift_phase:
                    _z_corr, _tgt_yaw, _falling = z_pid.compute(
                        float(_pl_pos[2]), _pl_vz, _pl_yaw, _pl_yaw_rate)
                    if _falling:
                        rw = -5.0
                        if agent.use_lstm:
                            agent.obs_history.push(no_noisy)
                            agent.push_cable_raw(
                                _buffer_cable_for_agent(agent, cable_for_policy))
                            val = agent.get_value_for_obs_sequence(
                                agent.obs_history.get_sequence())
                        else:
                            val = agent.get_value_for_state(no_noisy)
                        agent.add_to_buffer(
                            no_noisy, np.zeros(agent.action_dim, np.float32),
                            rw, 1.0, val, 0.0,
                            cable_raw=_buffer_cable_for_agent(
                                agent, cable_for_policy))
                        er += rw; es += 1; ts += 1; agent.total_steps = ts
                        rd = True; break
                else:
                    _z_corr, _tgt_yaw = 0.0, 0.0

                if swing_d is not None and not _is_lift_phase:
                    _pl_vel  = env.data.qvel[_dof_idx:_dof_idx+3].copy()
                    _ee_vel  = getattr(env, '_ee_vel_cache', np.zeros(3))
                    swing_d.compute(_pl_pos, ree, _pl_vel, _ee_vel)

                act, lp, val = agent.act(
                    no_noisy, deterministic=False)
                _act_arr = np.asarray(act, np.float32)
                _cruise_reward_base = get_last_nmpc_action(expert)

                if _is_lift_phase:
                    # ── Lift 模式: 用 expert.compute_delta_q_target + 3D residual_acc
                    # 与 test_phase / v12.1 lift fix 完全一致路径
                    _rm_xy = float(config["cruise_rl"].get("residual_acc_max_xy_rl", 0.08))
                    _rm_z  = float(config["cruise_rl"].get("residual_acc_max_z_rl",  0.10))
                    # action_dim=3, 索引 [0,1,2] = xy_residual + z_residual
                    _res3 = np.array([
                        float(np.clip(_act_arr[0], -_rm_xy, _rm_xy)),
                        float(np.clip(_act_arr[1], -_rm_xy, _rm_xy)),
                        float(np.clip(_act_arr[2], -_rm_z,  _rm_z)) if len(_act_arr) > 2 else 0.0,
                    ], np.float64)
                    dq = expert.compute_delta_q_target(obs, cq, residual_acc=_res3)
                    _cruise_reward_base = get_last_nmpc_action(expert)
                else:
                    # ── Cruise 模式: NMPC 输出 + xy 残差 + lock_z
                    if _cruise_nmpc_base:
                        try:
                            _a4 = expert.tracker.compute_ee_acceleration(obs, target_yaw=_tgt_yaw)
                            _base = np.array([float(_a4[0]), float(_a4[1])], np.float32)
                        except Exception:
                            _base = np.zeros(2, np.float32)
                        _cruise_reward_base = np.array(
                            [_base[0], _base[1], 0.0, 0.0], dtype=np.float32)
                        _res_max = float(config["cruise_rl"].get("residual_acc_max_xy_rl", 0.08))
                        _rl_clip = np.clip(_act_arr[:2], -_res_max, _res_max)
                        _comb = _base + _rl_clip
                        _amax = float(config["cruise_rl"].get("residual_acc_max_xy", 0.60))
                        _cn = float(np.linalg.norm(_comb))
                        if _cn > _amax: _comb = _comb / _cn * _amax
                        a3 = np.array([_comb[0], _comb[1], 0.0])
                    else:
                        a3 = np.array([_act_arr[0], _act_arr[1], 0.0])

                    dq = ectl.compute_delta_q(
                        a3, cq, ree, lock_z=True,
                        z_lock_height=float(config["cruise_rl"]["z_lock_height"]),
                        z_pid_correction=_z_corr, target_yaw=_tgt_yaw,
                        base_acc_xy=None, residual_mode=False)

                dq = _add_act_noise(dq, pert["act_noise"])

                no2, _, _, _, ei = _env_step_with_obs_predictor(dq)
                # [v13.0] rl_action 传完整 3D 维度
                rw, dn, sc, ri = compute_cruise_reward(env, no2, config, rs,
                                                       tracker=rew_tracker,
                                                       rl_action=_act_arr,
                                                       base_action=_cruise_reward_base)
                rew_tracker.step()

            # ── Descent: PID base + RL residual ──────────────────────────────
            elif phase == "descent" and _descent_pid_residual:
                act, lp, val = agent.act(
                    no_noisy, deterministic=False)
                dq, _pid_dq = _apply_descent_pid_residual(
                    expert, act, obs, env, config, cq,
                    pid_dq=base_dq_for_obs)

                dq = _add_act_noise(dq, pert["act_noise"])
                no2, _, _, _, ei = _env_step_with_obs_predictor(dq)
                # [v11.3] 传 rl_action 给 descent reward (action_magnitude/smoothness penalty)
                rw, dn, sc, ri = compute_descent_reward(env, no2, config, rs,
                                                        tracker=rew_tracker,
                                                        rl_action=act)
                rew_tracker.step()

            # ── Lift v12: NMPC vertical base + RL residual 3D ────────────────
            # [v12 fix] 用 expert.compute_delta_q_target(..., residual_acc=...)
            # 这样积分器/速度限制/锚定/IK 都和 test_phase expert-only 完全一致.
            # 修复了之前 expert.tracker + EEAccController 拼接的"双积分器漂移"问题.
            else:
                raise ValueError(f"Unsupported phase: {phase}")

            no2_true = no2
            if obs_predictor is not None:
                true_target = _obs_pred_target_vector(
                    config, no2_true, None, agent)
                visible_target, _obs_pred_info = obs_predictor.observe_result(
                    true_target, next_step_index=es + 1)
                no2 = _obs_pred_visible_env_obs(
                    config, no2_true, visible_target,
                    _obs_pred_info.used_prediction)
                obs_pred_cable_latent = _obs_pred_cable_latent_from_vec(
                    config, visible_target, _obs_pred_info.used_prediction)
                if _obs_pred_info.trained:
                    obs_pred_loss_sum += _obs_pred_info.loss
                    obs_pred_nll_sum += _obs_pred_info.nll
                    obs_pred_huber_sum += _obs_pred_info.huber
                    obs_pred_rmse_norm_sum += _obs_pred_info.rmse_norm
                    obs_pred_rmse_raw_sum += _obs_pred_info.rmse_raw
                    obs_pred_rmse_non_cable_sum += _obs_pred_info.rmse_non_cable
                    obs_pred_rmse_cable_latent_sum += _obs_pred_info.rmse_cable_latent
                    obs_pred_rmse_non_cable_norm_sum += (
                        _obs_pred_info.rmse_non_cable_norm)
                    obs_pred_rmse_cable_latent_norm_sum += (
                        _obs_pred_info.rmse_cable_latent_norm)
                    obs_pred_updates += 1
                if _obs_pred_info.used_prediction:
                    obs_pred_hidden_steps += 1
            elif delay_mdp_state.enabled:
                delay_mdp_state.record_action(dq)
                no2, _held_obs = delay_mdp_state.observe_result(
                    no2_true, next_step_index=es + 1)
                delay_mdp_total_steps += 1
                if _held_obs:
                    delay_mdp_hidden_steps += 1

            done = dn or ei.get("nan_detected", False)
            if es >= mx - 1:
                done = True
                if not ri.get("termination"):
                    if phase == "descent":
                        rw += float(config["descent_rl"]["reward"].get(
                            "timeout_penalty", 0.0))
                    ri["termination"] = "timeout"
            if sc: suc = True
            if done and ri.get("termination"):
                term_reason = ri["termination"]

            # [v12.6] 细粒度 RL 评估指标更新 (no2 是 env._get_obs() raw 输出)
            stab.update_step(no2_true, config, env=env, rl_action=act)

            # [v14.1] push cable_raw to history (for LSTM seq buffer)
            agent.push_cable_raw(_buffer_cable_for_agent(agent, cable_for_policy))
            agent.add_to_buffer(no_noisy, act, rw, float(done), val, lp,
                                cable_raw=_buffer_cable_for_agent(
                                    agent, cable_for_policy))

            er += rw; es += 1; ts += 1; agent.total_steps = ts
            obs = no2
            agent._update_entropy_coef(global_ts=ts)

            if agent.buffer.full:
                if done:
                    lv = 0.0
                else:
                    # [v14.0] 重新构建 obs
                    _wobs2 = build_wind_obs(env, _wf_max)
                    _base2 = None
                    if phase == "descent" and _descent_pid_residual:
                        try:
                            _cq2 = env.data.qpos[:7].copy().astype(np.float32)
                            _base2 = expert.compute_delta_q_target(
                                no2, _cq2.astype(np.float64))
                        except Exception:
                            _base2 = np.zeros(7, dtype=np.float32)
                    elif phase == "cruise" and _cruise_nmpc_residual:
                        _base2 = get_last_nmpc_action(expert)
                    _core2, _cable2, _wobs2, _, _ = build_phase_obs(
                        phase, no2, env, sxy, txy, pt, py, wind_obs=_wobs2,
                        base_action=_base2)
                    _cable2_for_policy = (obs_pred_cable_latent
                                          if obs_pred_cable_latent is not None
                                          else _cable2)
                    ns_ = agent.encode_obs(_core2, _cable2_for_policy, _wobs2)
                    ns_ = delay_mdp_state.append_features(ns_)
                    nn_ = agent.normalize_obs(ns_, update=False)
                    nn_noisy = _add_obs_noise(nn_, pert["obs_noise"])
                    if agent.use_lstm:
                        seq = list(agent.obs_history.buffer)
                        seq.append(nn_noisy.copy())
                        while len(seq) > agent.seq_len:
                            seq.pop(0)
                        while len(seq) < agent.seq_len:
                            seq.insert(0, np.zeros(agent.obs_dim, dtype=np.float32))
                        lv = agent.get_value_for_obs_sequence(seq)
                    else:
                        lv = agent.get_value_for_state(nn_noisy)
                agent.buffer.compute_returns_and_advantages(
                    lv, agent.gamma, agent.gae_lambda)
                agent.update(global_ts=ts)
                _ppo_update_count += 1
                rd = True
            if done:
                rd = True

        # ── Episode 末: 更新统计 / 课程 / 日志 ───────────────────────────────
        cur.update(suc)
        # [v10] 课程倒退则清 Adam 状态, 避免死局轨迹累积的二阶矩阻碍恢复
        # [v14.0] 课程晋级时 entropy boost: 临时提高 entropy_coef 给新分布探索空间
        if cur.consume_promotion_flag():
            _boost_val = agent.entropy_coef_start * 0.6
            agent.entropy_coef = max(agent.entropy_coef, _boost_val)
            print(f"  [v14.0 entropy boost] Level {cur.level_idx}: "
                  f"entropy_coef → {agent.entropy_coef:.4f}")

        r = agent._last_result
        mark = "✅" if suc else "❌"
        pl_pos_now = env.data.body('prefab').xpos
        dist_to_goal = float(np.linalg.norm(pl_pos_now[:2] - txy)) \
            if phase in ("cruise", "descent") else 0.0
        stab_summary = stab.summary()
        _update_episode_stats(
            stats, reward=er, steps=es, success=suc,
            dist_to_goal_cm=dist_to_goal * 100.0,
            stab_summary=stab_summary)
        ar = stats.mean("reward"); sr = stats.success_rate()

        cur_info = cur.info()
        lucky_metrics = _maybe_update_lucky_reject_schedule(
            config, phase,
            cur_info.get(f"cur/{phase}/sr_window", sr),
            cur_info.get(f"cur/{phase}/ramp_window_n", 0))
        cur_str = (f"L{cur.level_idx}/{cur.n_levels-1} ep{cur.eps_at_level} "
                   f"W[{cur_info['cur/%s/wind_min' % phase]:.1f},"
                   f"{cur_info['cur/%s/wind_max' % phase]:.1f}]")
        obs_pred_metrics = {}
        if obs_predictor is not None:
            _op_den = max(obs_pred_updates, 1)
            obs_pred_metrics = {
                f"obs_pred/{phase}/loss": obs_pred_loss_sum / _op_den,
                f"obs_pred/{phase}/nll": obs_pred_nll_sum / _op_den,
                f"obs_pred/{phase}/huber": obs_pred_huber_sum / _op_den,
                f"obs_pred/{phase}/rmse_norm": obs_pred_rmse_norm_sum / _op_den,
                f"obs_pred/{phase}/rmse_raw": obs_pred_rmse_raw_sum / _op_den,
                f"obs_pred/{phase}/rmse_non_cable":
                    obs_pred_rmse_non_cable_sum / _op_den,
                f"obs_pred/{phase}/rmse_cable_latent":
                    obs_pred_rmse_cable_latent_sum / _op_den,
                f"obs_pred/{phase}/rmse_non_cable_norm":
                    obs_pred_rmse_non_cable_norm_sum / _op_den,
                f"obs_pred/{phase}/rmse_cable_latent_norm":
                    obs_pred_rmse_cable_latent_norm_sum / _op_den,
                f"obs_pred/{phase}/updates": obs_pred_updates,
                f"obs_pred/{phase}/hidden_frac": (
                    obs_pred_hidden_steps / max(es, 1)),
            }
        delay_mdp_metrics = {}
        if delay_mdp_state.enabled:
            delay_mdp_metrics = {
                f"delay_mdp/{phase}/hidden_frac": (
                    delay_mdp_hidden_steps / max(delay_mdp_total_steps, 1)),
                f"delay_mdp/{phase}/obs_age_steps": delay_mdp_state.obs_age_steps,
                f"delay_mdp/{phase}/action_history_steps":
                    delay_mdp_state.action_history_steps,
                f"delay_mdp/{phase}/measurement_period_steps":
                    delay_mdp_state.period,
            }

        print(f"Ep{ep:4d} [{ts:7d}] {mark} R:{er:6.2f}({ar:5.2f}) SR:{sr*100:4.0f}% "
              f"S:{es:3d} dist:{dist_to_goal*100:.1f}cm "
              f"W:{pert.get('wind_speed', 0.0):.2f}m/s "
              f"[{cur_str}] | {term_reason}")
        print(f"       PPO PL:{r.policy_loss:6.3f} VL:{r.value_loss:6.3f} "
              f"E:{r.entropy_loss:6.3f} KL:{r.approx_kl:.4f}")
        if obs_predictor is not None:
            print(f"       ObsPred loss:{obs_pred_metrics[f'obs_pred/{phase}/loss']:.4f} "
                  f"rmse_raw:{obs_pred_metrics[f'obs_pred/{phase}/rmse_raw']:.4f} "
                  f"hidden:{obs_pred_metrics[f'obs_pred/{phase}/hidden_frac']:.0%}")
        if delay_mdp_state.enabled:
            print(f"       DelayMDP held:{delay_mdp_metrics[f'delay_mdp/{phase}/hidden_frac']:.0%} "
                  f"period:{delay_mdp_state.period} "
                  f"hist:{delay_mdp_state.action_history_steps}")
        # ── wandb 日志 ──────────────────────────────────────────────────────
        # [v10] 诊断指标: actor std + RL action 量级
        try:
            _std_per_dim = agent.actor.get_log_std_per_dim()
            _actor_std_mean = float(np.exp(_std_per_dim).mean())
            _actor_std_max  = float(np.exp(_std_per_dim).max())
        except Exception:
            _actor_std_mean = _actor_std_max = 0.0

        log_metrics = {
            # 训练表现
            f"{phase}/reward":       er,
            f"{phase}/avg_reward":   ar,
            f"{phase}/sr":           sr,
            f"{phase}/steps":        es,
            f"{phase}/dist_to_goal_cm": dist_to_goal * 100,
            # PPO 训练指标
            "ppo/pol_loss":    r.policy_loss,
            "ppo/val_loss":    r.value_loss,
            "ppo/ent_loss":    r.entropy_loss,
            "ppo/approx_kl":   r.approx_kl,
            "ppo/clip_frac":   r.clip_fraction,
            "ppo/ent_coef":    r.entropy_coef_used,
            # [v10] 诊断指标
            f"diag/{phase}/actor_std_mean": _actor_std_mean,
            f"diag/{phase}/actor_std_max":  _actor_std_max,
        }
        # 课程指标
        log_metrics.update(cur_info)
        log_metrics.update(lucky_metrics)
        log_metrics.update(obs_pred_metrics)
        log_metrics.update(delay_mdp_metrics)
        # reward 分项
        log_metrics.update(rew_tracker.episode_summary())
        # [v12.6] 细粒度 RL 评估指标 (anti-sway quality + RL intervention)
        log_metrics.update({
            f"stab/{phase}/{k}": v for k, v in stab_summary.items()
        })
        log_metrics.update(stats.wandb_trends(phase))
        logger.log(ep, log_metrics)

        with open(lf, "a", newline="") as f:
            csv.writer(f).writerow([ep, ts, f"{er:.3f}", f"{ar:.3f}", f"{sr:.3f}", es,
                                    f"{r.policy_loss:.4f}", f"{r.value_loss:.4f}",
                                    f"{r.entropy_loss:.4f}",
                                    cur.level_idx, f"{pert.get('wind_speed', 0.0):.4f}",
                                    f"{pert.get('wind_min', cur_info['cur/%s/wind_min' % phase]):.4f}",
                                    f"{pert.get('wind_max', cur_info['cur/%s/wind_max' % phase]):.4f}",
                                    f"{cur_info['cur/%s/sr_window' % phase]:.3f}",
                                    cur.eps_at_level])
        if ep > 0 and ep % SI == 0:
            save_checkpoint(agent, log_dir, ep, tag="latest",
                            obs_predictor=obs_predictor)
        if _best_checkpoint_ready(cur, cur_info, phase) and sr > best:
            best = sr
            save_checkpoint(agent, log_dir, ep, tag="best",
                            obs_predictor=obs_predictor)
        ep += 1

    save_checkpoint(agent, log_dir, ep, tag="final",
                    obs_predictor=obs_predictor)
    print(f"\n[{phase.upper()}-PPO] Done: {ts} steps, {(time.time()-t0)/60:.1f} min, "
          f"best_sr={best*100:.0f}%")
    logger.close(); env.close()
    return agent


# ==============================================================================
# [v11 Path 3] PPO 训练 — 向量化版本 (SubprocVecEnv 并行)
# ==============================================================================

def train_ppo_vec(phase, log_dir, config, resume_ckpt=None, n_envs=4):
    """[v11 Path 3] PPO 训练 with SubprocVecEnv.

    每个 worker 子进程独立运行 env + base controllers (NMPC / PID / Z-PID 等).
    主进程负责: agent inference, PPO update, 课程, wandb, HER (descent).

    与 _train_ppo_single 的差异:
    - 每 step 通过 vec.rl_step(idx, payload) 远程调用 worker
    - n_envs 个 episode 同时进行 (各自独立 curriculum 状态/reward tracker)
    - rollout buffer 接受来自所有 envs 的 transitions

    注意:
    - HER 暂不支持 vec 模式 (n_envs > 1 时自动禁用)
    - 推荐 n_envs ≤ CPU 物理核数 - 1
    """
    from vec_env import make_vec_env
    T  = int(config["train"].get("total_timesteps", 2_000_000))
    SI = int(config["train"]["save_interval"])
    EI = int(config["train"].get("eval_interval", 100))
    start_method = config["train"].get("vec_env_start_method", "forkserver")

    print(f"\n{'='*60}\n  PPO [VEC n_envs={n_envs}] | {phase.upper()} | "
          f"{T} steps | {log_dir}\n{'='*60}\n")

    # ── 主进程: agent only, 不持有 env ────────────────────────────────────────
    agent = PPOPhaseAgent(phase, config=config)
    if resume_ckpt and os.path.exists(resume_ckpt):
        agent.load(resume_ckpt); print(f"  Resumed: {resume_ckpt}")
        if bool(config.get("train", {}).get("reset_optimizer_on_resume", False)):
            agent.reset_adam_state("reset_optimizer_on_resume")
    _try_load_critic_cable_encoder(agent, config)
    obs_predictor_enabled = bool(config.get("observation_predictor", {}).get(
        "enabled", False))
    cable_latent_predictor_enabled = bool(config.get(
        "cable_latent_predictor", {}).get("enabled", False))
    if obs_predictor_enabled and _delay_mdp_enabled(config):
        raise ValueError("delay_mdp and observation_predictor are mutually exclusive")
    delay_mdp_enabled = _delay_mdp_enabled(config)
    if delay_mdp_enabled:
        _tmp_delay = DelayMDPObservationState(config)
        print(f"  [Delay-MDP-VEC] enabled: true obs every "
              f"{_tmp_delay.period} control steps, "
              f"action_history={_tmp_delay.action_history_steps}, "
              f"extra_obs_dim={_tmp_delay.feature_dim()}")
    adaptation_enabled = (
        bool(config.get("adaptation_history", {}).get("enabled", False)) and
        not cable_latent_predictor_enabled
    )
    if cable_latent_predictor_enabled:
        print("  [CableLatPred-VEC] enabled: visible history -> cable latent; "
              "actor cable slot uses predicted latent")
        if _use_rope_marker_features(config):
            print("  [CableLatPred-VEC] rope marker features enabled: "
                  f"source={_rope_marker_feature_source_config(config)}")
    if adaptation_enabled:
        _tmp_adapt = AdaptationHistoryState(config)
        print(f"  [AdaptHistory-VEC] enabled: action_history="
              f"{_tmp_adapt.action_history_steps}, "
              f"extra_obs_dim={_tmp_adapt.feature_dim()}")
    if getattr(agent, "use_asymmetric_critic", False):
        print(f"  [AsymCritic-VEC] actor_obs_dim={agent.obs_dim}, "
              f"critic_obs_dim={agent.critic_obs_dim}, "
              f"critic_base_obs_dim={agent.critic_base_obs_dim}")
    # ── 启动 n_envs 个 worker (每个独立持有 env + controllers) ───────────────
    def _make_one(wid):
        return make_phase_env_and_controllers(phase, config, worker_id=wid)
    vec = make_vec_env(_make_one, n_envs=n_envs, start_method=start_method)

    # Vectorized training shares one curriculum so the ramp follows global
    # episode outcomes instead of eight slow per-worker windows.
    shared_cur = CurriculumManager(config, phase)
    curs = [shared_cur for _ in range(n_envs)]
    ep_rewards = [0.0] * n_envs; ep_steps = [0] * n_envs
    ep_suc     = [False] * n_envs
    ep_term    = ["running"] * n_envs
    obs_list   = [None] * n_envs
    sxy_list   = [None] * n_envs; txy_list = [None] * n_envs
    pt_list    = [0.0] * n_envs;  py_list  = [0.0] * n_envs
    rstate_list= [None] * n_envs
    obs_histories = [agent.make_obs_history() for _ in range(n_envs)]
    critic_histories = [agent.make_critic_obs_history() for _ in range(n_envs)]
    cable_histories = [agent.make_cable_history() for _ in range(n_envs)]
    obs_predictors = [None] * n_envs
    cable_latent_predictors = [None] * n_envs
    policy_phase_obs_cache = [None] * n_envs
    cable_latent_action_dim = int(config.get(
        "cable_latent_predictor", {}).get("action_dim", 7))
    cable_latent_last_actions = [
        np.zeros(cable_latent_action_dim, dtype=np.float32)
        for _ in range(n_envs)
    ]
    cable_latent_use_rope_markers = (
        cable_latent_predictor_enabled and _use_rope_marker_features(config))
    rope_marker_feature_cache = [None] * n_envs
    delay_states = [DelayMDPObservationState(config) for _ in range(n_envs)]
    adaptation_states = [AdaptationHistoryState(config) for _ in range(n_envs)]
    if not adaptation_enabled:
        for _s in adaptation_states:
            _s.enabled = False

    def _new_obs_pred_acc():
        return {
            "loss": 0.0,
            "nll": 0.0,
            "huber": 0.0,
            "rmse_norm": 0.0,
            "rmse_raw": 0.0,
            "rmse_non_cable": 0.0,
            "rmse_cable_latent": 0.0,
            "rmse_non_cable_norm": 0.0,
            "rmse_cable_latent_norm": 0.0,
            "updates": 0,
            "hidden": 0,
            "steps": 0,
        }

    obs_pred_acc = [_new_obs_pred_acc() for _ in range(n_envs)]

    def _new_cable_latent_pred_acc():
        return {
            "loss": 0.0,
            "huber": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "rmse_norm": 0.0,
            "pred_norm": 0.0,
            "target_norm": 0.0,
            "updates": 0,
            "steps": 0,
        }

    def _accumulate_cable_latent_pred(acc, info):
        if info is None:
            return
        acc["steps"] += 1
        acc["pred_norm"] += float(getattr(info, "pred_norm", 0.0))
        acc["target_norm"] += float(getattr(info, "target_norm", 0.0))
        if getattr(info, "trained", False):
            acc["loss"] += float(info.loss)
            acc["huber"] += float(info.huber)
            acc["mse"] += float(info.mse)
            acc["rmse"] += float(info.rmse)
            acc["rmse_norm"] += float(info.rmse_norm)
            acc["updates"] += 1

    cable_latent_pred_acc = [
        _new_cable_latent_pred_acc() for _ in range(n_envs)
    ]

    def _new_delay_mdp_acc():
        return {"steps": 0, "hidden": 0}

    delay_mdp_acc = [_new_delay_mdp_acc() for _ in range(n_envs)]
    vision_acc = [_new_vision_acc() for _ in range(n_envs)]
    rope_marker_acc = [_new_rope_marker_acc() for _ in range(n_envs)]

    # ── 课程 + 物理初始化 (每 worker) ─────────────────────────────────────────
    def _reset_one_env(i, max_retries=10):
        for _retry in range(max_retries):
            pert = curs[i].sample_episode_perturbations()
            # reset
            if phase == "descent":
                _di = curs[i].get_descent_init()
                if _di is not None:
                    obs, pp = vec.reset_for_descent(i, cur_init=_di)
                else:
                    obs, pp = vec.reset_for_descent(i, cur_init=None)
            else:
                obs, pp = vec.reset(i)
            if obs is not None:
                break  # reset 成功
        else:
            raise RuntimeError(f"Worker {i}: reset_for_phase 重试 {max_retries} 次仍失败")
        vec.set_force_noise(i, pert["force_noise"])
        _apply_episode_wind_vec(vec, i, pert)
        # [v11 KEY FIX] reset worker 内的 expert / ee_ctrl / z_pid.
        # 之前漏掉, 导致 NMPC tracker / PID 保留上 episode 状态, base action 错误,
        # 是 vec 模式 SR=0 的根本原因.
        cq = vec.get_qpos(i)
        vec.reset_controllers(i, obs, cq, pp)
        # start_xy / target_xy
        sxy = None
        tp_attr = vec.env_attr(i, "target_pos")
        txy = np.asarray(tp_attr, np.float32)[:2]
        # rstate: 按 phase 选用对应的 RewardState 类 ([v11 fix])
        rs = REWARD_STATES[phase]()
        if phase == "descent":
            _di = curs[i].get_descent_init()
            if _di is not None:
                rs.current_xy_range = _di["xy_range"]
                rs.current_xy_tol   = _di["xy_tol"]
                rs.current_descent_level = curs[i].level_idx
                rs.descent_n_levels      = curs[i].n_levels
        return obs, sxy, txy, rs, pert

    # 初始化所有 worker
    perts = [None] * n_envs
    for i in range(n_envs):
        obs_list[i], sxy_list[i], txy_list[i], rstate_list[i], perts[i] = \
            _reset_one_env(i)
        pt_list[i] = 0.0; py_list[i] = 0.0
        obs_histories[i].reset(); critic_histories[i].reset()
        cable_histories[i].clear(); adaptation_states[i].reset()
        delay_states[i].reset(obs_list[i])
        ep_rewards[i] = 0.0; ep_steps[i] = 0
        ep_suc[i] = False; ep_term[i] = "running"

    if obs_predictor_enabled:
        obs_dim = _obs_pred_target_dim(config, obs_list[0], agent)
        obs_predictors = [
            build_observation_predictor(
                config, phase, obs_dim, device=getattr(agent, "device", None))
            for _ in range(n_envs)
        ]
        first_pred = next((p for p in obs_predictors if p is not None), None)
        if first_pred is not None:
            print(f"  [ObsPredictor-VEC] enabled: true obs every "
                  f"{first_pred.measurement_period} control steps, "
                  f"obs_dim={obs_dim}, "
                  f"target_mode={_obs_pred_target_mode(config)}, "
                  f"predictors={n_envs}")
            _try_load_obs_predictor_checkpoint(
                obs_predictors, config, resume_ckpt,
                label="ObsPredictor-VEC")
            for i, p in enumerate(obs_predictors):
                if p is not None:
                    p.reset(obs_list[i])

    logger = Logger(log_dir, project=f"phase_rl_v11_vec", run_name=f"{phase}_ppo_vec")
    logger.update_config(config)
    stats = _make_episode_stats(config)
    timing_acc = {
        "rl_action_s": 0.0, "rl_action_calls": 0,
        "obs_pred_s": 0.0, "obs_pred_calls": 0,
    }
    ts = 0; ep_count = 0; t0 = time.time(); best = 0.0

    lf = os.path.join(log_dir, f"{phase}_ppo_vec_log.csv")
    _progress_ep, _progress_ts, _progress_log = (0, 0, None)
    if resume_ckpt:
        _progress_ep, _progress_ts, _progress_log = _read_csv_progress(
            lf, os.path.join(log_dir, f"{phase}_ppo_log.csv"))
        if _progress_log:
            ep_count = max(ep_count, _progress_ep)
            ts = max(ts, _progress_ts, int(getattr(agent, "total_steps", 0)))
            agent.total_steps = max(int(getattr(agent, "total_steps", 0)), ts)
            print(f"  Resumed log progress: ep={ep_count}, ts={ts} "
                  f"from {_progress_log}")
        else:
            ts = max(ts, int(getattr(agent, "total_steps", 0)))
            agent.total_steps = max(int(getattr(agent, "total_steps", 0)), ts)
            if ts > 0:
                print(f"  Resumed checkpoint progress: ts={ts} "
                      f"(new log dir, no CSV to append)")
    T = _resolve_resume_training_target(
        T, ts, resume_ckpt=resume_ckpt, progress_log=_progress_log,
        label=f"{phase}-ppo-vec")
    _ppo_vec_header = ["episode", "total_steps", "ep_reward", "avg_reward",
                       "sr", "steps", "pol_loss", "val_loss", "ent",
                       "lvl", "wind_speed", "wind_speed_min",
                       "wind_speed_max", "cur_sr",
                       "cur_eps", "worker_id", "rl_action_ms", "obs_pred_ms",
                       "compute_hz_est", "vision_valid_rate",
                       "vision_active_cameras_mean", "vision_reproj_px_mean",
                       "vision_reproj_px_p95", "vision_depth_rmse_mm_mean",
                       "vision_depth_rmse_mm_p95", "vision_failure_steps",
                       "vision_top_failure", "rope_marker_source",
                       "rope_marker_visible_rate", "rope_marker_valid_rate",
                       "rope_marker_visible_mean",
                       "rope_marker_valid_after_noise_mean",
                       "rope_marker_camera_estimates_mean",
                       "rope_marker_cameras_mean",
                       "rope_marker_dropout_mean"]
    _init_csv_log(lf, _ppo_vec_header, append_existing=(
        resume_ckpt and os.path.abspath(_progress_log or "") == os.path.abspath(lf)))

    try:
        # 首次为每个 worker 通过 worker 端 build_phase_obs (因为主进程没有 env)
        phase_obs_cache = [None] * n_envs
        critic_phase_obs_cache = [None] * n_envs
        for i in range(n_envs):
            phase_obs_cache[i] = vec.build_phase_obs_remote(
                i, phase, obs_list[i], sxy_list[i], txy_list[i],
                pt_list[i], py_list[i])
            critic_phase_obs_cache[i] = phase_obs_cache[i]
            policy_phase_obs_cache[i] = phase_obs_cache[i]
            if cable_latent_use_rope_markers and hasattr(
                    vec, "get_rope_marker_features"):
                rope_marker_feature_cache[i] = vec.get_rope_marker_features(i)

        if cable_latent_predictor_enabled:
            visible_dim = int(_phase_visible_no_cable_vector(
                phase_obs_cache[0],
                rope_marker_features=rope_marker_feature_cache[0]
                if cable_latent_use_rope_markers else None).size)
            latent_dim = _obs_pred_cable_latent_dim(config, agent)
            cable_latent_predictors = [
                build_cable_latent_predictor(
                    config, phase, visible_dim, latent_dim=latent_dim,
                    action_dim=cable_latent_action_dim,
                    device=getattr(agent, "device", None))
                for _ in range(n_envs)
            ]
            first_clp = next(
                (p for p in cable_latent_predictors if p is not None), None)
            if first_clp is not None:
                print(f"  [CableLatPred-VEC] visible_dim={visible_dim}, "
                      f"latent_dim={latent_dim}, predictors={n_envs}")
                _try_load_cable_latent_predictor_checkpoint(
                    cable_latent_predictors, config, resume_ckpt,
                    label="CableLatPred-VEC")
                for i, p in enumerate(cable_latent_predictors):
                    if p is None:
                        continue
                    p.reset(_phase_visible_no_cable_vector(
                        phase_obs_cache[i],
                        rope_marker_features=rope_marker_feature_cache[i]
                        if cable_latent_use_rope_markers else None))
                    policy_phase_obs_cache[i], _clp_info = (
                        _phase_obs_with_predicted_cable_latent(
                            phase_obs_cache[i], p,
                            cable_latent_last_actions[i], agent, config,
                            phase=phase,
                            target_phase_obs=critic_phase_obs_cache[i],
                            rope_marker_features=rope_marker_feature_cache[i]
                            if cable_latent_use_rope_markers else None))
                    _accumulate_cable_latent_pred(
                        cable_latent_pred_acc[i], _clp_info)

        while ts < T:
            # ── 主进程: 用缓存的 phase obs 做 RL inference ──────────────────
            actions_list = []; lps_list = []; vals_list = []; obs_noisy_list = []
            critic_obs_list = []
            _rl_action_t0 = time.perf_counter()
            for i in range(n_envs):
                # [v15] phase_obs_cache[i] = (core, cable_raw, wind, tilt, yaw, base_dq)
                _core_i, _cable_i, _wind_i, _, _, _base_i = (
                    policy_phase_obs_cache[i])
                po = agent.encode_obs(_core_i, _cable_i, _wind_i)
                po = delay_states[i].append_features(po)
                po = adaptation_states[i].append_features(po)
                no = agent.normalize_obs(po, update=True)
                no_noisy = _add_obs_noise(no, perts[i]["obs_noise"])
                critic_no = None
                if getattr(agent, "use_asymmetric_critic", False):
                    _ccore_i, _ccable_i, _cwind_i, _, _, _ = critic_phase_obs_cache[i]
                    cpo = agent.encode_critic_obs(_ccore_i, _ccable_i, _cwind_i)
                    cpo = delay_states[i].append_features(cpo)
                    cpo = adaptation_states[i].append_features(cpo)
                    critic_no = agent.normalize_critic_obs(cpo, update=True)
                act, lp, val = agent.act_with_history(
                    no_noisy, obs_histories[i],
                    deterministic=False,
                    critic_norm_obs=critic_no,
                    critic_obs_history=critic_histories[i])
                cable_histories[i].append(
                    _buffer_cable_for_agent(agent, _cable_i))
                obs_noisy_list.append(no_noisy)
                critic_obs_list.append(critic_no)
                actions_list.append(act); lps_list.append(lp); vals_list.append(val)
            timing_acc["rl_action_s"] += time.perf_counter() - _rl_action_t0
            timing_acc["rl_action_calls"] += n_envs

            # ── 并发发送 rl_step 到所有 worker ────────────────────────────────
            payloads = []
            for i in range(n_envs):
                cq = vec.get_qpos(i)
                payloads.append({
                    'phase': phase,
                    'rl_action': actions_list[i],
                    'obs': obs_list[i],
                    'current_q': cq,
                    'start_xy': sxy_list[i],
                    'target_xy': txy_list[i],
                    'prev_tilt': pt_list[i],
                    'prev_yaw':  py_list[i],
                    'rstate':    rstate_list[i],
                    'act_noise': perts[i]["act_noise"],
                    'base_dq':    phase_obs_cache[i][5],
                    'train_reject_lucky_rebar_insert':
                        _lucky_reject_enabled(config),
                })

            # 异步发送 (SubprocVecEnv); DummyVecEnv 内联
            if hasattr(vec, 'remotes'):
                for i, p in enumerate(payloads):
                    vec.remotes[i].send(('rl_step', p))
                results = [vec._check_recv(vec.remotes[i].recv(), i)
                           for i in range(n_envs)]
            else:
                results = [vec.rl_step(i, payloads[i]) for i in range(n_envs)]

            # ── 处理每个 env 的结果 ─────────────────────────────────────────
            for i, res in enumerate(results):
                rw = res['reward']; done = res['done']; suc_step = res['success']
                _update_vision_acc(vision_acc[i], res.get('info', {}))
                if cable_latent_use_rope_markers:
                    _update_rope_marker_acc(
                        rope_marker_acc[i], res.get('info', {}))
                if cable_latent_use_rope_markers:
                    _rmf = (res.get('info', {}) or {}).get(
                        "rope_marker_features", None)
                    if _rmf is not None:
                        rope_marker_feature_cache[i] = _rmf
                true_next_obs = res['new_obs']
                visible_next_obs = true_next_obs
                visible_cable_latent = None
                obs_pred_info = None
                obs_pred = obs_predictors[i]
                if obs_pred is not None:
                    _pred_t0 = time.perf_counter()
                    dq_cmd = res.get('delta_q', None)
                    if dq_cmd is None:
                        dq_cmd = np.zeros(obs_pred.action_dim, dtype=np.float32)
                    current_target = _obs_pred_target_vector(
                        config, obs_list[i], phase_obs_cache[i], agent)
                    true_phase_obs = None
                    if res.get('new_core_obs') is not None:
                        true_phase_obs = (
                            res['new_core_obs'], res['new_cable_raw'],
                            res['new_wind_obs'], res['new_tilt'],
                            res['new_yaw'], res.get('new_base_dq'))
                    true_target = _obs_pred_target_vector(
                        config, true_next_obs, true_phase_obs, agent)
                    obs_pred.predict_next(current_target, dq_cmd, phase)
                    visible_target, obs_pred_info = obs_pred.observe_result(
                        true_target, next_step_index=ep_steps[i] + 1)
                    visible_next_obs = _obs_pred_visible_env_obs(
                        config, true_next_obs, visible_target,
                        obs_pred_info.used_prediction)
                    visible_cable_latent = _obs_pred_cable_latent_from_vec(
                        config, visible_target, obs_pred_info.used_prediction)
                    timing_acc["obs_pred_s"] += time.perf_counter() - _pred_t0
                    timing_acc["obs_pred_calls"] += 1
                    _acc = obs_pred_acc[i]
                    _acc["steps"] += 1
                    _acc["hidden"] += int(obs_pred_info.used_prediction)
                    if obs_pred_info.trained:
                        _acc["loss"] += obs_pred_info.loss
                        _acc["nll"] += obs_pred_info.nll
                        _acc["huber"] += obs_pred_info.huber
                        _acc["rmse_norm"] += obs_pred_info.rmse_norm
                        _acc["rmse_raw"] += obs_pred_info.rmse_raw
                        _acc["rmse_non_cable"] += obs_pred_info.rmse_non_cable
                        _acc["rmse_cable_latent"] += obs_pred_info.rmse_cable_latent
                        _acc["rmse_non_cable_norm"] += (
                            obs_pred_info.rmse_non_cable_norm)
                        _acc["rmse_cable_latent_norm"] += (
                            obs_pred_info.rmse_cable_latent_norm)
                        _acc["updates"] += 1
                elif delay_states[i].enabled:
                    delay_states[i].record_action(res.get('delta_q', None))
                    visible_next_obs, _held_obs = delay_states[i].observe_result(
                        true_next_obs, next_step_index=ep_steps[i] + 1)
                    _dacc = delay_mdp_acc[i]
                    _dacc["steps"] += 1
                    _dacc["hidden"] += int(_held_obs)
                adaptation_states[i].record_action(res.get('delta_q', None))
                obs_list[i] = visible_next_obs
                rstate_list[i] = res['rstate']

                # 终止: episode 满 max_steps?
                mx = int(config[f"{phase}_rl"]["max_steps"])
                if ep_steps[i] >= mx - 1:
                    done = True
                    if res['termination'] == 'running':
                        if phase == "descent":
                            rw += float(config["descent_rl"]["reward"].get(
                                "timeout_penalty", 0.0))
                        res['termination'] = 'timeout'
                if suc_step: ep_suc[i] = True
                if done and res['termination']:
                    ep_term[i] = res['termination']

                # [v14.0] 加入 buffer (用刚才 RL inference 时的 phase obs)
                _core_prev, _cable_prev, _wind_prev, _, _, _ = (
                    policy_phase_obs_cache[i])
                po_prev = agent.encode_obs(_core_prev, _cable_prev, _wind_prev)
                po_prev = delay_states[i].append_features(po_prev)
                po_prev = adaptation_states[i].append_features(po_prev)
                no_buf = agent.normalize_obs(po_prev, update=False)
                no_buf_noisy = obs_noisy_list[i]
                next_phase_obs = None
                next_critic_phase_obs = None
                if not done:
                    if obs_pred is not None or delay_states[i].enabled:
                        next_phase_obs = vec.build_phase_obs_remote(
                            i, phase, obs_list[i], sxy_list[i], txy_list[i],
                            pt_list[i], py_list[i])
                        next_phase_obs = _phase_obs_with_cable_component(
                            next_phase_obs, visible_cable_latent)
                    elif res.get('new_core_obs') is not None:
                        next_phase_obs = (
                            res['new_core_obs'], res['new_cable_raw'],
                            res['new_wind_obs'], res['new_tilt'],
                            res['new_yaw'], res.get('new_base_dq'))
                    if res.get('new_core_obs') is not None:
                        next_critic_phase_obs = (
                            res['new_core_obs'], res['new_cable_raw'],
                            res['new_wind_obs'], res['new_tilt'],
                            res['new_yaw'], res.get('new_base_dq'))
                    else:
                        next_critic_phase_obs = next_phase_obs
                next_policy_phase_obs = next_phase_obs
                if (not done and next_phase_obs is not None and
                        cable_latent_predictors[i] is not None):
                    next_policy_phase_obs, _clp_info = (
                        _phase_obs_with_predicted_cable_latent(
                            next_phase_obs, cable_latent_predictors[i],
                            res.get('delta_q', None), agent, config,
                            phase=phase,
                            target_phase_obs=next_critic_phase_obs,
                            rope_marker_features=rope_marker_feature_cache[i]
                            if cable_latent_use_rope_markers else None))
                    _accumulate_cable_latent_pred(
                        cable_latent_pred_acc[i], _clp_info)
                next_val = 0.0
                if (getattr(agent, "use_asymmetric_critic", False) and
                        next_critic_phase_obs is not None):
                    cpo_next = agent.encode_critic_obs(
                        next_critic_phase_obs[0], next_critic_phase_obs[1],
                        next_critic_phase_obs[2])
                    cpo_next = delay_states[i].append_features(cpo_next)
                    cpo_next = adaptation_states[i].append_features(cpo_next)
                    cno_next = agent.normalize_critic_obs(cpo_next, update=False)
                    if agent.use_lstm:
                        cseq = list(critic_histories[i].buffer)
                        cseq.append(cno_next.copy())
                        while len(cseq) > agent.seq_len:
                            cseq.pop(0)
                        while len(cseq) < agent.seq_len:
                            cseq.insert(0, np.zeros(agent.critic_obs_dim, dtype=np.float32))
                        next_val = agent.get_value_for_critic_obs_sequence(cseq)
                    else:
                        next_val = agent.get_value_for_critic_state(cno_next)
                elif next_policy_phase_obs is not None:
                    next_po = agent.encode_obs(
                        next_policy_phase_obs[0], next_policy_phase_obs[1],
                        next_policy_phase_obs[2])
                    next_po = delay_states[i].append_features(next_po)
                    next_po = adaptation_states[i].append_features(next_po)
                    next_no = agent.normalize_obs(next_po, update=False)
                    next_no_noisy = _add_obs_noise(next_no, perts[i]["obs_noise"])
                    if agent.use_lstm:
                        seq = list(obs_histories[i].buffer)
                        seq.append(next_no_noisy.copy())
                        while len(seq) > agent.seq_len:
                            seq.pop(0)
                        while len(seq) < agent.seq_len:
                            seq.insert(0, np.zeros(agent.obs_dim, dtype=np.float32))
                        next_val = agent.get_value_for_obs_sequence(seq)
                    else:
                        next_val = agent.get_value_for_state(next_no_noisy)
                if not agent.buffer.full:
                    agent.add_to_buffer_with_history(
                        no_buf_noisy, actions_list[i], rw, float(done),
                        vals_list[i], lps_list[i],
                        obs_history=obs_histories[i],
                        cable_history=cable_histories[i],
                        cable_raw=_buffer_cable_for_agent(agent, _cable_prev),
                        next_value=next_val, env_id=i,
                        critic_obs_history=(critic_histories[i] if getattr(
                            agent, "use_asymmetric_critic", False) else None),
                        critic_norm_obs=critic_obs_list[i])

                ep_rewards[i] += rw; ep_steps[i] += 1; ts += 1
                agent.total_steps = ts
                if res.get('delta_q', None) is not None:
                    _last_dq = np.asarray(
                        res.get('delta_q'), dtype=np.float32).reshape(-1)
                    if _last_dq.size < cable_latent_action_dim:
                        _last_dq = np.pad(
                            _last_dq,
                            (0, cable_latent_action_dim - _last_dq.size))
                    cable_latent_last_actions[i] = (
                        _last_dq[:cable_latent_action_dim].astype(np.float32))

                # [v14.0] 更新 phase_obs_cache: 用 worker 返回的新 obs
                if not done and next_phase_obs is not None:
                    phase_obs_cache[i] = next_phase_obs
                    pt_list[i] = next_phase_obs[3]; py_list[i] = next_phase_obs[4]
                if not done and next_policy_phase_obs is not None:
                    policy_phase_obs_cache[i] = next_policy_phase_obs
                if not done and next_critic_phase_obs is not None:
                    critic_phase_obs_cache[i] = next_critic_phase_obs

                # ── Episode 结束: log + 重置 ──────────────────────────────
                if done:
                    _stab_sum = res.get('stab_summary') or {}
                    _pl_xy_done = np.asarray(res.get('pl_xy', np.zeros(2)),
                                             np.float32)
                    _dist_cm = float(np.linalg.norm(_pl_xy_done - txy_list[i]) * 100.0) \
                        if phase in ("cruise", "descent") else 0.0
                    _update_episode_stats(
                        stats, reward=ep_rewards[i], steps=ep_steps[i],
                        success=ep_suc[i], dist_to_goal_cm=_dist_cm,
                        stab_summary=_stab_sum)
                    curs[i].update(ep_suc[i])
                    ar = stats.mean("reward"); sr = stats.success_rate()
                    cur_info = curs[i].info()
                    lucky_metrics = _maybe_update_lucky_reject_schedule(
                        config, phase,
                        cur_info.get(f"cur/{phase}/sr_window", sr),
                        cur_info.get(f"cur/{phase}/ramp_window_n", 0))
                    r = agent._last_result
                    mark = "✅" if ep_suc[i] else "❌"
                    cur_str = (f"L{curs[i].level_idx}/{curs[i].n_levels-1} "
                               f"ep{curs[i].eps_at_level} "
                               f"W[{cur_info['cur/%s/wind_min' % phase]:.1f},"
                               f"{cur_info['cur/%s/wind_max' % phase]:.1f}]")
                    vision_summary = _summarize_vision_acc(vision_acc[i])
                    rope_marker_summary = _summarize_rope_marker_acc(
                        rope_marker_acc[i])
                    print(f"[w{i}] Ep{ep_count:4d} [{ts:7d}] {mark} R:{ep_rewards[i]:6.2f}"
                          f"({ar:5.2f}) SR:{sr*100:4.0f}% S:{ep_steps[i]:3d} "
                          f"W:{perts[i].get('wind_speed', 0.0):.2f}m/s [{cur_str}] | {ep_term[i]}")
                    if vision_summary:
                        _vf = str(vision_summary.get("top_failure", "") or "-")
                        print(f"       Vision w{i} "
                              f"valid:{vision_summary['valid_rate']*100:.1f}% "
                              f"cams:{vision_summary['active_cameras_mean']:.2f} "
                              f"reproj:{vision_summary['reprojection_px_mean']:.2f}/"
                              f"{vision_summary['reprojection_px_p95']:.2f}px "
                              f"depth:{vision_summary['depth_rmse_mm_mean']:.1f}/"
                              f"{vision_summary['depth_rmse_mm_p95']:.1f}mm "
                              f"fail:{vision_summary['failure_steps']} {_vf}")
                    if rope_marker_summary:
                        print(f"       RopeMarker w{i} "
                              f"src:{rope_marker_summary['source']} "
                              f"valid:{rope_marker_summary['valid_rate']*100:.1f}% "
                              f"visible:{rope_marker_summary['visible_rate']*100:.1f}% "
                              f"markers:{rope_marker_summary['valid_after_noise_mean']:.1f}/"
                              f"{rope_marker_summary['markers_total_mean']:.1f} "
                              f"cam_est:{rope_marker_summary['camera_estimates_mean']:.1f} "
                              f"cams:{rope_marker_summary['cameras_mean']:.1f} "
                              f"drop:{rope_marker_summary['dropout_mean']:.1f}")
                    obs_pred_metrics = {}
                    if obs_predictors[i] is not None:
                        _op = obs_pred_acc[i]
                        _op_den = max(int(_op["updates"]), 1)
                        _op_steps = max(int(_op["steps"]), 1)
                        obs_pred_metrics = {
                            f"obs_pred/{phase}/loss": _op["loss"] / _op_den,
                            f"obs_pred/{phase}/nll": _op["nll"] / _op_den,
                            f"obs_pred/{phase}/huber": _op["huber"] / _op_den,
                            f"obs_pred/{phase}/rmse_norm": _op["rmse_norm"] / _op_den,
                            f"obs_pred/{phase}/rmse_raw": _op["rmse_raw"] / _op_den,
                            f"obs_pred/{phase}/rmse_non_cable":
                                _op["rmse_non_cable"] / _op_den,
                            f"obs_pred/{phase}/rmse_cable_latent":
                                _op["rmse_cable_latent"] / _op_den,
                            f"obs_pred/{phase}/rmse_non_cable_norm":
                                _op["rmse_non_cable_norm"] / _op_den,
                            f"obs_pred/{phase}/rmse_cable_latent_norm":
                                _op["rmse_cable_latent_norm"] / _op_den,
                            f"obs_pred/{phase}/updates": int(_op["updates"]),
                            f"obs_pred/{phase}/hidden_frac": _op["hidden"] / _op_steps,
                            f"obs_pred/{phase}/worker_id": i,
                        }
                        print(f"       ObsPred w{i} "
                              f"loss:{obs_pred_metrics[f'obs_pred/{phase}/loss']:.4f} "
                              f"rmse_raw:{obs_pred_metrics[f'obs_pred/{phase}/rmse_raw']:.4f} "
                              f"hidden:{obs_pred_metrics[f'obs_pred/{phase}/hidden_frac']:.0%}")
                    cable_latent_pred_metrics = {}
                    if cable_latent_predictors[i] is not None:
                        _cp = cable_latent_pred_acc[i]
                        _cp_den = max(int(_cp["updates"]), 1)
                        _cp_steps = max(int(_cp["steps"]), 1)
                        cable_latent_pred_metrics = {
                            f"cable_latent_pred/{phase}/loss":
                                _cp["loss"] / _cp_den,
                            f"cable_latent_pred/{phase}/huber":
                                _cp["huber"] / _cp_den,
                            f"cable_latent_pred/{phase}/mse":
                                _cp["mse"] / _cp_den,
                            f"cable_latent_pred/{phase}/rmse":
                                _cp["rmse"] / _cp_den,
                            f"cable_latent_pred/{phase}/rmse_norm":
                                _cp["rmse_norm"] / _cp_den,
                            f"cable_latent_pred/{phase}/pred_norm":
                                _cp["pred_norm"] / _cp_steps,
                            f"cable_latent_pred/{phase}/target_norm":
                                _cp["target_norm"] / _cp_steps,
                            f"cable_latent_pred/{phase}/updates":
                                int(_cp["updates"]),
                            f"cable_latent_pred/{phase}/worker_id": i,
                        }
                        print(f"       CableLatPred w{i} "
                              f"loss:{cable_latent_pred_metrics[f'cable_latent_pred/{phase}/loss']:.4f} "
                              f"rmse:{cable_latent_pred_metrics[f'cable_latent_pred/{phase}/rmse']:.4f} "
                              f"rmse_norm:{cable_latent_pred_metrics[f'cable_latent_pred/{phase}/rmse_norm']:.4f}")
                    delay_mdp_metrics = {}
                    if delay_states[i].enabled:
                        _dacc = delay_mdp_acc[i]
                        _d_steps = max(int(_dacc["steps"]), 1)
                        delay_mdp_metrics = {
                            f"delay_mdp/{phase}/hidden_frac":
                                _dacc["hidden"] / _d_steps,
                            f"delay_mdp/{phase}/obs_age_steps":
                                delay_states[i].obs_age_steps,
                            f"delay_mdp/{phase}/action_history_steps":
                                delay_states[i].action_history_steps,
                            f"delay_mdp/{phase}/measurement_period_steps":
                                delay_states[i].period,
                            f"delay_mdp/{phase}/worker_id": i,
                        }
                        print(f"       DelayMDP w{i} "
                              f"held:{delay_mdp_metrics[f'delay_mdp/{phase}/hidden_frac']:.0%} "
                              f"period:{delay_states[i].period} "
                              f"hist:{delay_states[i].action_history_steps}")
                    try:
                        _std = agent.actor.get_log_std_per_dim()
                        _std_mean = float(np.exp(_std).mean())
                    except Exception:
                        _std_mean = 0.0
                    _rl_ms = 1000.0 * timing_acc["rl_action_s"] / max(
                        1, int(timing_acc["rl_action_calls"]))
                    _pred_ms = 1000.0 * timing_acc["obs_pred_s"] / max(
                        1, int(timing_acc["obs_pred_calls"]))
                    _compute_ms = _rl_ms + _pred_ms
                    _compute_hz = 1000.0 / max(_compute_ms, 1e-9)
                    log_metrics = {
                        f"{phase}/reward":   ep_rewards[i],
                        f"{phase}/avg_reward": ar,
                        f"{phase}/sr":       sr,
                        f"{phase}/steps":    ep_steps[i],
                        "ppo/pol_loss":      r.policy_loss,
                        "ppo/val_loss":      r.value_loss,
                        "ppo/ent_loss":      r.entropy_loss,
                        "ppo/approx_kl":     r.approx_kl,
                        "ppo/clip_frac":     r.clip_fraction,
                        "ppo/ent_coef":      r.entropy_coef_used,
                        f"diag/{phase}/actor_std_mean": _std_mean,
                        "diag/vec/worker_id": i,
                        f"timing/{phase}/rl_action_ms": _rl_ms,
                        f"timing/{phase}/obs_pred_ms": _pred_ms,
                        f"timing/{phase}/compute_hz_est": _compute_hz,
                        "diag/ppo/asymmetric_critic": float(getattr(
                            agent, "use_asymmetric_critic", False)),
                        "diag/ppo/adaptation_history": float(adaptation_enabled),
                        "diag/ppo/cable_latent_predictor": float(
                            cable_latent_predictor_enabled),
                        "diag/ppo/actor_obs_dim": int(getattr(agent, "obs_dim", 0)),
                        "diag/ppo/critic_obs_dim": int(getattr(agent, "critic_obs_dim", 0)),
                    }
                    log_metrics.update(obs_pred_metrics)
                    log_metrics.update(cable_latent_pred_metrics)
                    log_metrics.update(delay_mdp_metrics)
                    log_metrics.update(_vision_summary_wandb_metrics(
                        vision_summary, phase, i))
                    log_metrics.update(_rope_marker_summary_wandb_metrics(
                        rope_marker_summary, phase, i))
                    log_metrics.update(cur_info)
                    log_metrics.update(lucky_metrics)
                    # [v12.6] vec 模式 stab: worker 在 done 时返回 stab_summary, 直接喂 wandb
                    if _stab_sum:
                        log_metrics.update({
                            f"stab/{phase}/{k}": v for k, v in _stab_sum.items()
                        })
                    log_metrics.update(stats.wandb_trends(phase))
                    logger.log(ep_count, log_metrics)

                    with open(lf, "a", newline="") as f:
                        csv.writer(f).writerow([ep_count, ts, f"{ep_rewards[i]:.3f}",
                            f"{ar:.3f}", f"{sr:.3f}", ep_steps[i],
                            f"{r.policy_loss:.4f}", f"{r.value_loss:.4f}",
                            f"{r.entropy_loss:.4f}",
                            curs[i].level_idx, f"{perts[i].get('wind_speed', 0.0):.4f}",
                            f"{perts[i].get('wind_min', cur_info['cur/%s/wind_min' % phase]):.4f}",
                            f"{perts[i].get('wind_max', cur_info['cur/%s/wind_max' % phase]):.4f}",
                            f"{cur_info['cur/%s/sr_window' % phase]:.3f}",
                            curs[i].eps_at_level, i,
                            f"{_rl_ms:.4f}", f"{_pred_ms:.4f}",
                            f"{_compute_hz:.1f}",
                            f"{vision_summary.get('valid_rate', 0.0):.4f}",
                            f"{vision_summary.get('active_cameras_mean', 0.0):.3f}",
                            f"{vision_summary.get('reprojection_px_mean', 0.0):.4f}",
                            f"{vision_summary.get('reprojection_px_p95', 0.0):.4f}",
                            f"{vision_summary.get('depth_rmse_mm_mean', 0.0):.4f}",
                            f"{vision_summary.get('depth_rmse_mm_p95', 0.0):.4f}",
                            int(vision_summary.get('failure_steps', 0)),
                            str(vision_summary.get('top_failure', "") or ""),
                            str(rope_marker_summary.get('source', "") or ""),
                            f"{rope_marker_summary.get('visible_rate', 0.0):.4f}",
                            f"{rope_marker_summary.get('valid_rate', 0.0):.4f}",
                            f"{rope_marker_summary.get('visible_mean', 0.0):.3f}",
                            f"{rope_marker_summary.get('valid_after_noise_mean', 0.0):.3f}",
                            f"{rope_marker_summary.get('camera_estimates_mean', 0.0):.3f}",
                            f"{rope_marker_summary.get('cameras_mean', 0.0):.3f}",
                            f"{rope_marker_summary.get('dropout_mean', 0.0):.3f}"])

                    ep_count += 1
                    if ep_count > 0 and ep_count % SI == 0:
                        save_checkpoint(agent, log_dir, ep_count, tag="latest",
                                        obs_predictor=obs_predictors,
                                        cable_latent_predictor=(
                                            cable_latent_predictors))
                    if _best_checkpoint_ready(curs[i], cur_info, phase) and sr > best:
                        best = sr
                        save_checkpoint(agent, log_dir, ep_count, tag="best",
                                        obs_predictor=obs_predictors,
                                        cable_latent_predictor=(
                                            cable_latent_predictors))

                    # 重置此 worker 的 env + 状态
                    obs_list[i], sxy_list[i], txy_list[i], rstate_list[i], perts[i] = \
                        _reset_one_env(i)
                    pt_list[i] = 0.0; py_list[i] = 0.0
                    obs_histories[i].reset(); critic_histories[i].reset()
                    cable_histories[i].clear(); adaptation_states[i].reset()
                    delay_states[i].reset(obs_list[i])
                    if obs_predictors[i] is not None:
                        obs_predictors[i].reset(obs_list[i])
                    obs_pred_acc[i] = _new_obs_pred_acc()
                    cable_latent_pred_acc[i] = _new_cable_latent_pred_acc()
                    delay_mdp_acc[i] = _new_delay_mdp_acc()
                    vision_acc[i] = _new_vision_acc()
                    rope_marker_acc[i] = _new_rope_marker_acc()
                    ep_rewards[i] = 0.0; ep_steps[i] = 0
                    ep_suc[i] = False; ep_term[i] = "running"
                    # 重新 build phase obs (新 episode 起点)
                    phase_obs_cache[i] = vec.build_phase_obs_remote(
                        i, phase, obs_list[i], sxy_list[i], txy_list[i],
                        pt_list[i], py_list[i])
                    critic_phase_obs_cache[i] = phase_obs_cache[i]
                    policy_phase_obs_cache[i] = phase_obs_cache[i]
                    if cable_latent_use_rope_markers and hasattr(
                            vec, "get_rope_marker_features"):
                        rope_marker_feature_cache[i] = (
                            vec.get_rope_marker_features(i))
                    cable_latent_last_actions[i] = np.zeros(
                        cable_latent_action_dim, dtype=np.float32)
                    if cable_latent_predictors[i] is not None:
                        cable_latent_predictors[i].reset(
                            _phase_visible_no_cable_vector(
                                phase_obs_cache[i],
                                rope_marker_features=rope_marker_feature_cache[i]
                                if cable_latent_use_rope_markers else None))
                        policy_phase_obs_cache[i], _clp_info = (
                            _phase_obs_with_predicted_cable_latent(
                                phase_obs_cache[i], cable_latent_predictors[i],
                                cable_latent_last_actions[i], agent, config,
                                phase=phase,
                                target_phase_obs=critic_phase_obs_cache[i],
                                rope_marker_features=rope_marker_feature_cache[i]
                                if cable_latent_use_rope_markers else None))
                        _accumulate_cable_latent_pred(
                            cable_latent_pred_acc[i], _clp_info)

                agent._update_entropy_coef(global_ts=ts)

            # ── Buffer 满? 触发 PPO update ─────────────────────────────────
            if agent.buffer.full:
                agent.buffer.compute_returns_and_advantages(
                    0.0, agent.gamma, agent.gae_lambda)
                agent.update(global_ts=ts)
    finally:
        vec.close()
        try: logger.close()
        except Exception: pass

    save_checkpoint(agent, log_dir, ep_count, tag="final",
                    obs_predictor=obs_predictors,
                    cable_latent_predictor=cable_latent_predictors)
    print(f"\n[{phase.upper()}-PPO-VEC] Done: {ts} steps, {(time.time()-t0)/60:.1f} min, "
          f"best_sr={best*100:.0f}%")
    return agent


# ==============================================================================
# [v11 Path 3] SAC 训练 — 向量化版本 (SubprocVecEnv 并行)
# ==============================================================================

def train_sac_vec(phase, log_dir, config, resume_ckpt=None, n_envs=4):
    """[v11 Path 3] SAC 训练 with SubprocVecEnv.

    设计:
    - N 个 worker 子进程独立运行 env + base controllers
    - 主进程: SAC agent (actor + 2 critics + replay buffer)
    - 每步: 主进程为 N 个 worker 各生成 1 action, 并发发送 rl_step,
            收回 N 个 transitions, 全部加入共享 replay buffer
    - 每 update_interval 步: SAC train_step (off-policy 优化)
    - Warmup: 用 actor 输出 + 高斯噪声 (cruise actor 零初始化, 起步即接近 NMPC)

    与 train_ppo_vec 的差异:
    - 不需要 phase_obs_cache (SAC act 是 stateless)
    - HER 仅在 descent 单 env 时支持; vec descent 仅记 episode-level HER (worker 末 flush)
    - update 频率独立于 buffer 满 (off-policy)
    """
    from vec_env import make_vec_env
    T  = int(config["train"].get("total_timesteps", 2_000_000))
    SI = int(config["train"]["save_interval"])
    EI = int(config["train"].get("eval_interval", 100))
    WU = int(config["sac"].get("warmup_steps", 5000))
    start_method = config["train"].get("vec_env_start_method", "forkserver")

    print(f"\n{'='*60}\n  SAC [VEC n_envs={n_envs}] | {phase.upper()} | "
          f"{T} steps | warmup={WU} | {log_dir}\n{'='*60}\n")

    # ── 主进程: 仅 agent ─────────────────────────────────────────────────────
    agent = SACPhaseAgent(phase, config=config)
    if resume_ckpt and os.path.exists(resume_ckpt):
        agent.load(resume_ckpt); print(f"  Resumed: {resume_ckpt}")
        if bool(config.get("train", {}).get("reset_optimizer_on_resume", False)):
            agent.reset_adam_state("reset_optimizer_on_resume")

    # ── 启动 worker (每个独立持有 env + controllers) ─────────────────────────
    def _make_one(wid):
        return make_phase_env_and_controllers(phase, config, worker_id=wid)
    vec = make_vec_env(_make_one, n_envs=n_envs, start_method=start_method)

    # Share one curriculum across vectorized workers for a true global ramp.
    shared_cur = CurriculumManager(config, phase)
    curs = [shared_cur for _ in range(n_envs)]
    ep_rewards = [0.0] * n_envs; ep_steps = [0] * n_envs
    ep_suc     = [False] * n_envs
    ep_term    = ["running"] * n_envs
    obs_list   = [None] * n_envs
    sxy_list   = [None] * n_envs; txy_list = [None] * n_envs
    pt_list    = [0.0] * n_envs;  py_list  = [0.0] * n_envs
    rstate_list= [None] * n_envs

    def _reset_one_env(i, max_retries=10):
        for _retry in range(max_retries):
            pert = curs[i].sample_episode_perturbations()
            if phase == "descent":
                _di = curs[i].get_descent_init()
                obs, pp = vec.reset_for_descent(i, cur_init=_di)
            else:
                obs, pp = vec.reset(i)
            if obs is not None:
                break
        else:
            raise RuntimeError(f"Worker {i}: reset_for_phase 重试 {max_retries} 次仍失败")
        vec.set_force_noise(i, pert["force_noise"])
        _apply_episode_wind_vec(vec, i, pert)
        # [v11 KEY FIX] reset worker 内的 expert / ee_ctrl / z_pid (同 PPO vec)
        cq = vec.get_qpos(i)
        vec.reset_controllers(i, obs, cq, pp)
        sxy = None
        tp_attr = vec.env_attr(i, "target_pos")
        txy = np.asarray(tp_attr, np.float32)[:2]
        # [v11 fix] 按 phase 选用对应的 RewardState 类
        rs = REWARD_STATES[phase]()
        if phase == "descent":
            _di = curs[i].get_descent_init()
            if _di is not None:
                rs.current_xy_range = _di["xy_range"]
                rs.current_xy_tol   = _di["xy_tol"]
                rs.current_descent_level = curs[i].level_idx
                rs.descent_n_levels      = curs[i].n_levels
        return obs, sxy, txy, rs, pert

    perts = [None] * n_envs
    for i in range(n_envs):
        obs_list[i], sxy_list[i], txy_list[i], rstate_list[i], perts[i] = \
            _reset_one_env(i)
        pt_list[i] = 0.0; py_list[i] = 0.0

    logger = Logger(log_dir, project=f"phase_rl_v11_vec", run_name=f"{phase}_sac_vec")
    logger.update_config(config)
    stats = _make_episode_stats(config)
    ts = 0; ep_count = 0; t0 = time.time(); best = 0.0; onf = False
    update_interval = int(config["sac"].get("update_interval", 1))

    lf = os.path.join(log_dir, f"{phase}_sac_vec_log.csv")
    with open(lf, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "total_steps", "ep_reward", "avg_reward",
                                "sr", "steps", "cl", "al", "alpha", "lvl", "wind_speed",
                                "wind_speed_min", "wind_speed_max",
                                "cur_sr", "cur_eps", "worker_id"])

    try:
        # 首轮 build phase obs (远程, 因主进程无 env)
        phase_obs_cache = [None] * n_envs
        for i in range(n_envs):
            phase_obs_cache[i] = vec.build_phase_obs_remote(
                i, phase, obs_list[i], sxy_list[i], txy_list[i],
                pt_list[i], py_list[i])

        while ts < T:
            # ── 主进程: 为每个 worker 生成 action ─────────────────────────────
            actions_list = []; obs_noisy_list = []
            for i in range(n_envs):
                # [v14.0]
                _core_i, _cable_i, _wind_i, _, _, _ = phase_obs_cache[i]
                po = agent.encode_obs(_core_i, _cable_i, _wind_i)
                no = agent.normalize_obs(po, update=not onf)
                no_noisy = _add_obs_noise(no, perts[i]["obs_noise"])
                if ts < WU:
                    # Warmup: actor zero-init + 较大噪声 (cruise residual ~0 + noise)
                    act_base = agent.act(no_noisy, deterministic=False)
                    noise = np.random.normal(0, 0.1, len(act_base)).astype(np.float32)
                    act = (act_base + noise).astype(np.float32)
                else:
                    act = agent.act(no_noisy, deterministic=False)
                obs_noisy_list.append(no_noisy)
                actions_list.append(act)
            if ts >= WU and not onf:
                agent._freeze_obs_norm = True; onf = True

            # ── 并发发送 rl_step ──────────────────────────────────────────────
            payloads = []
            for i in range(n_envs):
                cq = vec.get_qpos(i)
                payloads.append({
                    'phase': phase,
                    'rl_action': actions_list[i],
                    'obs': obs_list[i],
                    'current_q': cq,
                    'start_xy': sxy_list[i],
                    'target_xy': txy_list[i],
                    'prev_tilt': pt_list[i],
                    'prev_yaw':  py_list[i],
                    'rstate':    rstate_list[i],
                    'act_noise': perts[i]["act_noise"],
                    'base_dq':    phase_obs_cache[i][5],
                    'train_reject_lucky_rebar_insert':
                        _lucky_reject_enabled(config),
                })

            if hasattr(vec, 'remotes'):
                for i, p in enumerate(payloads):
                    vec.remotes[i].send(('rl_step', p))
                results = [vec._check_recv(vec.remotes[i].recv(), i)
                           for i in range(n_envs)]
            else:
                results = [vec.rl_step(i, payloads[i]) for i in range(n_envs)]

            # ── 处理每个 worker 的 transition ─────────────────────────────────
            for i, res in enumerate(results):
                rw = res['reward']; done = res['done']; suc_step = res['success']
                obs_list[i] = res['new_obs']
                rstate_list[i] = res['rstate']

                mx = int(config[f"{phase}_rl"]["max_steps"])
                if ep_steps[i] >= mx - 1:
                    done = True
                    if res['termination'] == 'running':
                        res['termination'] = 'timeout'
                if suc_step: ep_suc[i] = True
                if done and res['termination']:
                    ep_term[i] = res['termination']

                # [v14.0] 计算 normalized obs: SAC remember 用 norm_obs + norm_next_obs
                _core_prev, _cable_prev, _wind_prev, _, _, _ = phase_obs_cache[i]
                po_prev = agent.encode_obs(_core_prev, _cable_prev, _wind_prev)
                no_prev_noisy = obs_noisy_list[i]
                if res.get('new_core_obs') is not None:
                    new_po = agent.encode_obs(
                        res['new_core_obs'], res['new_cable_raw'], res['new_wind_obs'])
                    new_no = agent.normalize_obs(new_po, update=False)
                    new_no = _add_obs_noise(new_no, perts[i]["obs_noise"])
                else:
                    new_no = no_prev_noisy  # falling 等异常 case
                pl_xy_now = res.get('pl_xy', np.zeros(2, np.float32))
                achieved_pos = np.array([pl_xy_now[0], pl_xy_now[1],
                                          float(config.get("insertion", {}).get(
                                              "target_payload_z", 0.10))], np.float32)
                agent.remember(no_prev_noisy, actions_list[i], new_no,
                               rw, float(done), achieved_pos=achieved_pos)

                ep_rewards[i] += rw; ep_steps[i] += 1; ts += 1
                agent.total_steps = ts

                if not done:
                    phase_obs_cache[i] = (
                        res['new_core_obs'], res['new_cable_raw'],
                        res['new_wind_obs'], res['new_tilt'], res['new_yaw'],
                        res.get('new_base_dq'))
                    pt_list[i] = res['new_tilt']; py_list[i] = res['new_yaw']

                # Episode 结束
                if done:
                    # HER flush (descent)
                    if phase == "descent":
                        agent.flush_episode_her(
                            txy_list[i],
                            float(config["insertion"]["target_payload_z"]))

                    _stab_sum = res.get('stab_summary') or {}
                    _pl_xy_done = np.asarray(res.get('pl_xy', np.zeros(2)),
                                             np.float32)
                    _dist_cm = float(np.linalg.norm(_pl_xy_done - txy_list[i]) * 100.0) \
                        if phase in ("cruise", "descent") else 0.0
                    _update_episode_stats(
                        stats, reward=ep_rewards[i], steps=ep_steps[i],
                        success=ep_suc[i], dist_to_goal_cm=_dist_cm,
                        stab_summary=_stab_sum)
                    curs[i].update(ep_suc[i])
                    # [v11 vec fix] 倒退时 reset Adam state, 同 PPO vec.
                    ar = stats.mean("reward"); sr = stats.success_rate()
                    r = agent._last_result
                    mark = "✅" if ep_suc[i] else "❌"
                    cur_str = f"L{curs[i].level_idx}/{curs[i].n_levels-1} ep{curs[i].eps_at_level}"
                    print(f"[w{i}] Ep{ep_count:4d} [{ts:7d}] {mark} R:{ep_rewards[i]:6.2f}"
                          f"({ar:5.2f}) SR:{sr*100:4.0f}% S:{ep_steps[i]:3d} "
                          f"W:{perts[i].get('wind_speed', 0.0):.2f}m/s [{cur_str}] | {ep_term[i]}")
                    print(f"       SAC CL:{r.critic_loss:.4f} AL:{r.actor_loss:.4f} "
                          f"α:{agent.alpha:.4f}")

                    cur_info = curs[i].info()
                    log_metrics = {
                        f"{phase}/reward":   ep_rewards[i],
                        f"{phase}/avg_reward": ar,
                        f"{phase}/sr":       sr,
                        f"{phase}/steps":    ep_steps[i],
                        "sac/critic_loss":   r.critic_loss,
                        "sac/actor_loss":    r.actor_loss,
                        "sac/alpha_loss":    r.alpha_loss,
                        "sac/alpha":         r.alpha,
                        "sac/q_mean":        r.q_mean,
                        # [v11.4] Q-divergence 早期检测
                        "sac/target_q_mean": getattr(agent, '_last_target_q', 0.0),
                        "sac/q_target_gap":  r.q_mean - getattr(agent, '_last_target_q', 0.0),
                        "diag/vec/worker_id": i,
                    }
                    log_metrics.update(cur_info)
                    # [v12.6] vec 模式 stab: worker 返回 stab_summary
                    if _stab_sum:
                        log_metrics.update({
                            f"stab/{phase}/{k}": v for k, v in _stab_sum.items()
                        })
                    log_metrics.update(stats.wandb_trends(phase))
                    logger.log(ep_count, log_metrics)

                    with open(lf, "a", newline="") as f:
                        csv.writer(f).writerow([ep_count, ts, f"{ep_rewards[i]:.3f}",
                            f"{ar:.3f}", f"{sr:.3f}", ep_steps[i],
                            f"{r.critic_loss:.5f}", f"{r.actor_loss:.5f}",
                            f"{agent.alpha:.5f}",
                            curs[i].level_idx, f"{perts[i].get('wind_speed', 0.0):.4f}",
                            f"{perts[i].get('wind_min', cur_info['cur/%s/wind_min' % phase]):.4f}",
                            f"{perts[i].get('wind_max', cur_info['cur/%s/wind_max' % phase]):.4f}",
                            f"{cur_info['cur/%s/sr_window' % phase]:.3f}",
                            curs[i].eps_at_level, i])

                    ep_count += 1
                    if ep_count > 0 and ep_count % SI == 0:
                        save_checkpoint(agent, log_dir, ep_count, tag="latest")
                    if _best_checkpoint_ready(curs[i], cur_info, phase) and sr > best:
                        best = sr; save_checkpoint(agent, log_dir, ep_count, tag="best")

                    # Reset this worker
                    obs_list[i], sxy_list[i], txy_list[i], rstate_list[i], perts[i] = \
                        _reset_one_env(i)
                    pt_list[i] = 0.0; py_list[i] = 0.0
                    ep_rewards[i] = 0.0; ep_steps[i] = 0
                    ep_suc[i] = False; ep_term[i] = "running"
                    phase_obs_cache[i] = vec.build_phase_obs_remote(
                        i, phase, obs_list[i], sxy_list[i], txy_list[i],
                        pt_list[i], py_list[i])

            # ── SAC train_step (off-policy update) ────────────────────────────
            # [v11 vec fix] 每个外层 step 收集 n_envs 个 transitions; 必须做
            # n_envs 次 update 才能 keep up, 否则 buffer 涨太快、critic 训练量
            # 落后, 表现为 critic_loss 涨到 100-300 (单进程仅 5-10).
            # 来源: SAC 原论文 (Haarnoja 2018) 推荐 1 update per 1 env step.
            if ts >= WU and ts % update_interval == 0:
                _n_updates = n_envs  # vec 模式: n_envs updates per outer step
                for _ in range(_n_updates):
                    agent.train_step()

    finally:
        vec.close()
        try: logger.close()
        except Exception: pass

    save_checkpoint(agent, log_dir, ep_count, tag="final")
    print(f"\n[{phase.upper()}-SAC-VEC] Done: {ts} steps, "
          f"{(time.time()-t0)/60:.1f} min, best_sr={best*100:.0f}%")
    return agent


# ==============================================================================
# SAC 训练 (单进程)
# ==============================================================================

def train_sac(phase, log_dir, config, resume_ckpt=None):
    T  = int(config["train"].get("total_timesteps", 2_000_000))
    SI = int(config["train"]["save_interval"])
    EI = int(config["train"].get("eval_interval", 100))
    WU = int(config["sac"].get("warmup_steps", 5000))

    print(f"\n{'='*60}\n  SAC | {phase.upper()} | {T} steps | warmup={WU} | "
          f"{log_dir}\n{'='*60}\n")

    env    = CableRobotEnvWithObstacles(config=config)
    agent  = SACPhaseAgent(phase, config=config)
    expert = JointSpaceExpert(config, env.ik_solver)
    ectl   = EEAccController(config, env.ik_solver)
    z_pid   = CruiseZYawPID(config)          if phase == "cruise" else None
    swing_d = SwingDampingController(config) if phase == "cruise" else None
    _cruise_nmpc_base = (phase == "cruise" and
        bool(config.get("cruise_rl", {}).get("use_nmpc_base", False)))
    _cruise_nmpc_residual = (phase == "cruise" and
        bool(config.get("cruise_rl", {}).get(
            "nmpc_residual_mode", _cruise_nmpc_base)))
    _descent_pid_residual = (phase == "descent" and
        bool(config.get("descent_rl", {}).get("pid_residual_mode", True)))
    # [v12] lift 也支持 NMPC base + RL 残差
    cur = CurriculumManager(config, phase)

    # [v9] resume_ckpt 用于断点续训
    if resume_ckpt and os.path.exists(resume_ckpt):
        agent.load(resume_ckpt); print(f"  Resumed from ckpt: {resume_ckpt}")
        if bool(config.get("train", {}).get("reset_optimizer_on_resume", False)):
            agent.reset_adam_state("reset_optimizer_on_resume")

    logger = Logger(log_dir, project=f"phase_rl_v9", run_name=f"{phase}_sac")
    logger.update_config(config)
    stats = _make_episode_stats(config)
    ep = 0; ts = 0; best = 0.0; t0 = time.time(); onf = False

    lf = os.path.join(log_dir, f"{phase}_sac_log.csv")
    with open(lf, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "total_steps", "ep_reward", "avg_reward",
                                "sr", "steps", "cl", "al", "alpha", "lvl", "wind_speed",
                                "wind_speed_min", "wind_speed_max",
                                "cur_sr", "cur_eps"])

    while ts < T:
        pert = cur.sample_episode_perturbations()

        if phase == "descent":
            obs, pp = reset_for_descent_with_cur(env, config, cur)
        else:
            obs, pp = reset_for_phase(env, phase, config)
        if obs is None:
            continue
        env.set_force_noise(pert["force_noise"])
        _apply_episode_wind_env(env, pert)

        cq = env.data.qpos[:7].copy()
        expert.reset(obs, cq, env=env)
        if pp is not None:
            # [v12.2] lift 时只把"lift 段"喂给 tracker, 防止 look-ahead 跨段拉走 xy
            expert.set_path(pp)
            plp = env.data.body('prefab').xpos.copy()
            _advance_expert_to_nearest_wp(expert, pp, plp)
        ectl.reset(env._get_ee_pos(), cq)

        if z_pid is not None:
            _pl_z   = float(env.data.body('prefab').xpos[2])
            _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
            _pl_yaw = float(R.from_matrix(_pl_mat).as_euler('xyz')[2])
            z_pid.reset(_pl_z, _pl_yaw)

        sxy = np.asarray(getattr(
            env, "episode_start_xy", env.default_start_xy), dtype=np.float32).copy()
        txy = env.target_pos.copy()
        pt, py = 0.0, 0.0

        rs = REWARD_STATES[phase]()
        if hasattr(rs, 'total_steps_global'):
            rs.total_steps_global = ts
        if phase == "descent":
            _di = cur.get_descent_init()
            if _di is not None:
                rs.current_xy_range = _di["xy_range"]
                rs.current_xy_tol   = _di["xy_tol"]
                rs.current_descent_level = cur.level_idx
                rs.descent_n_levels      = cur.n_levels

        rew_tracker = RewardComponentTracker(phase)
        # [v12.6] 细粒度 RL 评估指标 tracker
        stab = StabilityMetrics()
        er = 0.0; es = 0; suc = False; term_reason = "running"
        mx = int(config[f"{phase}_rl"]["max_steps"])
        cached_base_dq = None

        while True:
            cq = env.data.qpos[:7].copy().astype(np.float32)
            base_dq_for_obs = cached_base_dq
            if phase == "descent" and _descent_pid_residual and base_dq_for_obs is None:
                try:
                    base_dq_for_obs = expert.compute_delta_q_target(
                        obs, cq.astype(np.float64))
                except Exception:
                    base_dq_for_obs = np.zeros(7, dtype=np.float32)
            elif phase == "cruise" and _cruise_nmpc_residual:
                base_dq_for_obs = get_last_nmpc_action(expert)
            cached_base_dq = None
            # [v14.0]
            _wobs = build_wind_obs(env, wind_obs_scale(config))
            core, cable_raw, _wobs, pt, py = build_phase_obs(
                phase, obs, env, sxy, txy, pt, py, wind_obs=_wobs,
                base_action=base_dq_for_obs)
            po = agent.encode_obs(core, cable_raw, _wobs)
            no = agent.normalize_obs(po, update=not onf)
            no_noisy = _add_obs_noise(no, pert["obs_noise"])

            if ts < WU:
                try:
                    act = collect_expert_acc(expert, env, obs, cq, phase, config)
                    act = act + np.random.normal(0, 0.1, len(act)).astype(np.float32)
                except Exception:
                    act = np.zeros(agent.action_dim, np.float32)
            else:
                act = agent.act(no_noisy, deterministic=False)
            if ts == WU and not onf:
                agent._freeze_obs_norm = True; onf = True

            ree = env._get_ee_pos()
            _cruise_reward_base = None
            _cruise_reward_action = act

            if phase == "cruise" and _cruise_nmpc_residual:
                _res3 = clip_cruise_residual(act, config)
                dq = expert.compute_delta_q_target(
                    obs, cq.astype(np.float64), residual_acc=_res3)
                _cruise_reward_base = get_last_nmpc_action(expert)
                _cruise_reward_action = _res3

            elif phase == "cruise" and z_pid is not None:
                _pl_pos  = env.data.body('prefab').xpos
                _dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
                _pl_vz   = float(env.data.qvel[_dof_idx + 2])
                _pl_mat  = env.data.body('prefab').xmat.reshape(3, 3)
                _pl_euler = R.from_matrix(_pl_mat).as_euler('xyz')
                _pl_yaw  = float(_pl_euler[2])
                _pl_yaw_rate = float(env.data.qvel[_dof_idx + 5]) \
                    if _dof_idx + 5 < len(env.data.qvel) else 0.0
                _z_corr, _tgt_yaw, _falling = z_pid.compute(
                    float(_pl_pos[2]), _pl_vz, _pl_yaw, _pl_yaw_rate)
                if _falling:
                    rw = -5.0
                    _wobs2 = build_wind_obs(env, wind_obs_scale(config))
                    _c2, _cb2, _w2, _, _ = build_phase_obs(
                        phase, obs, env, sxy, txy, pt, py, wind_obs=_wobs2,
                        base_action=get_last_nmpc_action(expert))
                    npo = agent.encode_obs(_c2, _cb2, _w2)
                    nn_ = agent.normalize_obs(npo, update=False)
                    nn_ = _add_obs_noise(nn_, pert["obs_noise"])
                    agent.remember(no_noisy, act, nn_, rw, 1.0)
                    er += rw; es += 1; ts += 1; agent.total_steps = ts; break

                if swing_d is not None:
                    _pl_vel  = env.data.qvel[_dof_idx:_dof_idx+3].copy()
                    _ee_vel  = getattr(env, '_ee_vel_cache', np.zeros(3))
                    swing_d.compute(_pl_pos, ree, _pl_vel, _ee_vel)

                if _cruise_nmpc_base:
                    try:
                        _a4 = expert.tracker.compute_ee_acceleration(obs, target_yaw=_tgt_yaw)
                        _ba = np.array([float(_a4[0]), float(_a4[1])], np.float32)
                    except Exception:
                        _ba = np.zeros(2, np.float32)
                    _cruise_reward_base = np.array(
                        [_ba[0], _ba[1], 0.0, 0.0], dtype=np.float32)
                    # [v11 Path 1] SAC cruise 默认与 PPO 对齐
                    _rm = float(config["cruise_rl"].get("residual_acc_max_xy_rl", 0.08))
                    _cm = _ba + np.clip(act[:2], -_rm, _rm)
                    _am = float(config["cruise_rl"].get("residual_acc_max_xy", 0.60))
                    _cn = float(np.linalg.norm(_cm))
                    if _cn > _am: _cm = _cm / _cn * _am
                    a3 = np.array([_cm[0], _cm[1], 0.0])
                else:
                    a3 = np.array([act[0], act[1], 0.0])
                dq = ectl.compute_delta_q(
                    a3, cq, ree, lock_z=True,
                    z_lock_height=float(config["cruise_rl"]["z_lock_height"]),
                    z_pid_correction=_z_corr, target_yaw=_tgt_yaw,
                    base_acc_xy=None, residual_mode=False)

            elif phase == "descent":
                if _descent_pid_residual:
                    dq, _pid_dq = _apply_descent_pid_residual(
                        expert, act, obs, env, config, cq,
                        pid_dq=base_dq_for_obs)
                else:
                    dq = ectl.compute_delta_q(act, cq, ree)
            else:
                raise ValueError(f"Unsupported phase: {phase}")

            dq = _add_act_noise(dq, pert["act_noise"])
            no2, _, _, _, ei = env.step(dq)

            if phase == "cruise":
                # [v11.2] 传 rl_action 给 cruise reward (action_magnitude/smoothness penalty)
                rw, dn, sc, ri = compute_cruise_reward(env, no2, config, rs,
                                                       tracker=rew_tracker,
                                                       rl_action=_cruise_reward_action,
                                                       base_action=_cruise_reward_base)
            elif phase == "descent":
                # [v11.3] 传 rl_action 给 descent reward
                rw, dn, sc, ri = compute_descent_reward(env, no2, config, rs,
                                                        tracker=rew_tracker,
                                                        rl_action=act)
            else:
                raise ValueError(f"Unsupported phase: {phase}")
            rew_tracker.step()

            done = dn or ei.get("nan_detected", False)
            if es >= mx - 1:
                done = True; ri.setdefault("termination", "timeout")
            if sc: suc = True
            if done and ri.get("termination"): term_reason = ri["termination"]

            # [v12.6] 细粒度 RL 评估指标更新
            stab.update_step(no2, config, env=env, rl_action=act)

            _wobs3 = build_wind_obs(env, wind_obs_scale(config))
            _base3 = None
            if phase == "descent" and _descent_pid_residual and not done:
                try:
                    _cq3 = env.data.qpos[:7].copy().astype(np.float32)
                    _base3 = expert.compute_delta_q_target(
                        no2, _cq3.astype(np.float64))
                except Exception:
                    _base3 = np.zeros(7, dtype=np.float32)
                cached_base_dq = _base3
            elif phase == "cruise" and _cruise_nmpc_residual:
                _base3 = get_last_nmpc_action(expert)
            _c3, _cb3, _w3, _, _ = build_phase_obs(
                phase, no2, env, sxy, txy, pt, py, wind_obs=_wobs3,
                base_action=_base3)
            npo = agent.encode_obs(_c3, _cb3, _w3)
            nn_ = agent.normalize_obs(npo, update=False)
            nn_ = _add_obs_noise(nn_, pert["obs_noise"])
            _pl_pos_now = env.data.body('prefab').xpos.copy()
            agent.remember(no_noisy, act, nn_, rw, float(done),
                           achieved_pos=_pl_pos_now)
            if ts >= WU and ts % agent.update_interval == 0:
                agent.train_step()
            er += rw; es += 1; ts += 1; agent.total_steps = ts; obs = no2
            if done:
                if phase == "descent":
                    agent.flush_episode_her(txy,
                        float(config["insertion"]["target_payload_z"]))
                break

        cur.update(suc)
        r = agent._last_result

        mark = "✅" if suc else "❌"
        pl_pos_now = env.data.body('prefab').xpos
        dist_to_goal = float(np.linalg.norm(pl_pos_now[:2] - txy)) \
            if phase in ("cruise", "descent") else 0.0
        stab_summary = stab.summary()
        _update_episode_stats(
            stats, reward=er, steps=es, success=suc,
            dist_to_goal_cm=dist_to_goal * 100.0,
            stab_summary=stab_summary)
        ar = stats.mean("reward"); sr = stats.success_rate()
        cur_info = cur.info()
        print(f"Ep{ep:4d} [{ts:7d}] {mark} R:{er:6.2f}({ar:5.2f}) SR:{sr*100:4.0f}% "
              f"S:{es:3d} dist:{dist_to_goal*100:.1f}cm "
              f"W:{pert.get('wind_speed', 0.0):.2f}m/s L{cur.level_idx} | {term_reason}")
        print(f"       SAC CL:{r.critic_loss:7.4f} AL:{r.actor_loss:7.4f} "
              f"α:{agent.alpha:.4f}")

        log_metrics = {
            f"{phase}/reward":       er,
            f"{phase}/avg_reward":   ar,
            f"{phase}/sr":           sr,
            f"{phase}/steps":        es,
            f"{phase}/dist_to_goal_cm": dist_to_goal * 100,
            "sac/critic_loss": r.critic_loss,
            "sac/actor_loss":  r.actor_loss,
            "sac/alpha_loss":  r.alpha_loss,
            "sac/alpha":       r.alpha,
            "sac/q_mean":      r.q_mean,
            # [v11.4] Q-divergence 早期检测 (q_mean 应 ≈ target_q, gap 大说明 critic 落后)
            "sac/target_q_mean": getattr(agent, '_last_target_q', 0.0),
            "sac/q_target_gap":  r.q_mean - getattr(agent, '_last_target_q', 0.0),
        }
        log_metrics.update(cur_info)
        log_metrics.update(rew_tracker.episode_summary())
        # [v12.6] 细粒度 RL 评估指标
        log_metrics.update({
            f"stab/{phase}/{k}": v for k, v in stab_summary.items()
        })
        log_metrics.update(stats.wandb_trends(phase))
        logger.log(ep, log_metrics)

        with open(lf, "a", newline="") as f:
            csv.writer(f).writerow([ep, ts, f"{er:.3f}", f"{ar:.3f}", f"{sr:.3f}", es,
                                    f"{r.critic_loss:.5f}", f"{r.actor_loss:.5f}",
                                    f"{agent.alpha:.5f}",
                                    cur.level_idx, f"{pert.get('wind_speed', 0.0):.4f}",
                                    f"{pert.get('wind_min', cur_info['cur/%s/wind_min' % phase]):.4f}",
                                    f"{pert.get('wind_max', cur_info['cur/%s/wind_max' % phase]):.4f}",
                                    f"{cur_info['cur/%s/sr_window' % phase]:.3f}",
                                    cur.eps_at_level])
        if ep > 0 and ep % SI == 0:
            save_checkpoint(agent, log_dir, ep, tag="latest")
        if (ep > 0 and ep % EI == 0 and
                _best_checkpoint_ready(cur, cur_info, phase) and sr > best):
            best = sr; save_checkpoint(agent, log_dir, ep, tag="best")
        ep += 1

    save_checkpoint(agent, log_dir, ep, tag="final")
    print(f"\n[{phase.upper()}-SAC] Done: {ts} steps, {(time.time()-t0)/60:.1f} min, "
          f"best_sr={best*100:.0f}%")
    logger.close(); env.close()
    return agent



# ==============================================================================
# 训练入口
# ==============================================================================

def _make_policy_distill_teacher_config(config, phase):
    teacher_config = copy.deepcopy(config)
    teacher_config.setdefault("adaptation_history", {})["enabled"] = False
    teacher_config.setdefault("asymmetric_critic", {})["enabled"] = False
    teacher_config.setdefault("cable_encoder", {})
    teacher_config["cable_encoder"].update({
        "enabled": True,
        "zero_obs": False,
        "output_dim": int(DEFAULT_CONFIG.get("cable_encoder", {}).get(
            "output_dim", 32)),
    })
    if phase == "descent":
        teacher_config.setdefault("descent_rl", {})["obs_dim"] = int(
            DEFAULT_CONFIG.get("descent_rl", {}).get("obs_dim", 76))
    elif phase == "cruise":
        teacher_config.setdefault("cruise_rl", {})["obs_dim"] = int(
            DEFAULT_CONFIG.get("cruise_rl", {}).get("obs_dim", 77))
    teacher_config.setdefault("observation_predictor", {})["enabled"] = False
    teacher_config.setdefault("delay_mdp", {})["enabled"] = False
    teacher_config.setdefault("cable_latent_predictor", {})["enabled"] = False
    return teacher_config


def _distill_actor_step(agent, batch_obs, batch_actions):
    if not batch_obs:
        return None
    dev = agent.device
    obs_b = torch.as_tensor(np.asarray(batch_obs, dtype=np.float32),
                            dtype=torch.float32, device=dev)
    act_b = torch.as_tensor(np.asarray(batch_actions, dtype=np.float32),
                            dtype=torch.float32, device=dev)
    if agent.use_lstm:
        pred, _, _, _ = agent.actor.get_action(obs_b, deterministic=True)
    else:
        pred, _, _ = agent.actor.get_action(obs_b, deterministic=True)
    loss = F.mse_loss(pred, act_b)
    agent.opt_actor.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), agent.max_grad_norm)
    agent.opt_actor.step()
    return float(loss.detach().cpu().item())


def train_policy_distill(phase, log_dir, config, teacher_ckpt=None, n_envs=4):
    """Distill a full-cable teacher policy into the current student config."""
    from vec_env import make_vec_env

    dist_cfg = config.setdefault("policy_distill", {})
    teacher_ckpt = teacher_ckpt or str(dist_cfg.get("teacher_ckpt", "") or "")
    if not teacher_ckpt or not os.path.exists(teacher_ckpt):
        raise ValueError("--teacher-ckpt is required for --algo distill")

    T = int(config["train"].get("total_timesteps", 300_000))
    SI = int(config["train"].get("save_interval", 50))
    batch_size = max(1, int(dist_cfg.get("batch_size", 1024)))
    deterministic_teacher = bool(dist_cfg.get("deterministic_teacher", True))
    start_method = config["train"].get("vec_env_start_method", "forkserver")

    print(f"\n{'='*60}\n  Policy DISTILL [VEC n_envs={n_envs}] | "
          f"{phase.upper()} | {T} samples | {log_dir}\n{'='*60}\n")
    print(f"  Teacher: {teacher_ckpt}")
    print(f"  Student obs_dim={config[f'{phase}_rl']['obs_dim']} "
          f"cable_enabled={config.get('cable_encoder', {}).get('enabled', True)}")

    student = PPOPhaseAgent(phase, config=config)
    teacher_config = _make_policy_distill_teacher_config(config, phase)
    teacher = PPOPhaseAgent(phase, config=teacher_config)
    teacher.load(teacher_ckpt)
    teacher.actor.eval()
    teacher.critic.eval()
    for p in teacher.actor.parameters():
        p.requires_grad = False
    for p in teacher.critic.parameters():
        p.requires_grad = False

    def _make_one(wid):
        return make_phase_env_and_controllers(phase, config, worker_id=wid)

    vec = make_vec_env(_make_one, n_envs=n_envs, start_method=start_method)
    curs = [CurriculumManager(config, phase) for _ in range(n_envs)]
    obs_list = [None] * n_envs
    sxy_list = [None] * n_envs
    txy_list = [None] * n_envs
    rstate_list = [None] * n_envs
    perts = [None] * n_envs
    pt_list = [0.0] * n_envs
    py_list = [0.0] * n_envs
    ep_rewards = [0.0] * n_envs
    ep_steps = [0] * n_envs
    ep_suc = [False] * n_envs
    ep_term = ["running"] * n_envs
    student_histories = [student.make_obs_history() for _ in range(n_envs)]
    teacher_histories = [teacher.make_obs_history() for _ in range(n_envs)]
    adaptation_states = [AdaptationHistoryState(config) for _ in range(n_envs)]
    cable_latent_predictor_enabled = bool(config.get(
        "cable_latent_predictor", {}).get("enabled", False))
    if cable_latent_predictor_enabled:
        for _s in adaptation_states:
            _s.enabled = False
    cable_latent_predictors = [None] * n_envs
    policy_phase_obs_cache = [None] * n_envs
    cable_latent_action_dim = int(config.get(
        "cable_latent_predictor", {}).get("action_dim", 7))
    cable_latent_last_actions = [
        np.zeros(cable_latent_action_dim, dtype=np.float32)
        for _ in range(n_envs)
    ]
    cable_latent_pred_updates = [0] * n_envs
    cable_latent_pred_rmse = [0.0] * n_envs
    cable_latent_use_rope_markers = (
        cable_latent_predictor_enabled and _use_rope_marker_features(config))
    rope_marker_feature_cache = [None] * n_envs
    vision_acc = [_new_vision_acc() for _ in range(n_envs)]

    def _reset_one_env(i, max_retries=10):
        for _retry in range(max_retries):
            pert = curs[i].sample_episode_perturbations()
            if phase == "descent":
                _di = curs[i].get_descent_init()
                obs, pp = vec.reset_for_descent(i, cur_init=_di)
            else:
                obs, pp = vec.reset(i)
            if obs is not None:
                break
        else:
            raise RuntimeError(f"Worker {i}: policy distill reset failed")
        vec.set_force_noise(i, pert["force_noise"])
        _apply_episode_wind_vec(vec, i, pert)
        cq = vec.get_qpos(i)
        vec.reset_controllers(i, obs, cq, pp)
        txy = np.asarray(vec.env_attr(i, "target_pos"), np.float32)[:2]
        rs = REWARD_STATES[phase]()
        if phase == "descent":
            _di = curs[i].get_descent_init()
            if _di is not None:
                rs.current_xy_range = _di["xy_range"]
                rs.current_xy_tol = _di["xy_tol"]
                rs.current_descent_level = curs[i].level_idx
                rs.descent_n_levels = curs[i].n_levels
        return obs, None, txy, rs, pert

    try:
        for i in range(n_envs):
            obs_list[i], sxy_list[i], txy_list[i], rstate_list[i], perts[i] = \
                _reset_one_env(i)
            student_histories[i].reset()
            teacher_histories[i].reset()
            adaptation_states[i].reset()

        logger = Logger(log_dir, project="phase_policy_distill",
                        run_name=f"{phase}_no_cable_distill")
        logger.update_config(config)
        lf = os.path.join(log_dir, f"{phase}_policy_distill_log.csv")
        _init_csv_log(lf, [
            "episode", "total_steps", "ep_reward", "success", "steps",
            "distill_loss", "loss_window", "lvl", "wind_speed",
            "wind_speed_min", "wind_speed_max", "cur_sr", "cur_eps",
            "worker_id", "termination", "vision_valid_rate",
            "vision_active_cameras_mean", "vision_reproj_px_mean",
            "vision_reproj_px_p95", "vision_depth_rmse_mm_mean",
            "vision_depth_rmse_mm_p95", "vision_failure_steps",
        ], append_existing=False)

        phase_obs_cache = [None] * n_envs
        for i in range(n_envs):
            phase_obs_cache[i] = vec.build_phase_obs_remote(
                i, phase, obs_list[i], sxy_list[i], txy_list[i],
                pt_list[i], py_list[i])
            policy_phase_obs_cache[i] = phase_obs_cache[i]
            if cable_latent_use_rope_markers and hasattr(
                    vec, "get_rope_marker_features"):
                rope_marker_feature_cache[i] = vec.get_rope_marker_features(i)

        if cable_latent_predictor_enabled:
            visible_dim = int(_phase_visible_no_cable_vector(
                phase_obs_cache[0],
                rope_marker_features=rope_marker_feature_cache[0]
                if cable_latent_use_rope_markers else None).size)
            latent_dim = _obs_pred_cable_latent_dim(config, student)
            cable_latent_predictors = [
                build_cable_latent_predictor(
                    config, phase, visible_dim, latent_dim=latent_dim,
                    action_dim=cable_latent_action_dim,
                    device=getattr(student, "device", None))
                for _ in range(n_envs)
            ]
            first_clp = next(
                (p for p in cable_latent_predictors if p is not None), None)
            if first_clp is not None:
                print(f"  [CableLatPred-DISTILL] visible_dim={visible_dim}, "
                      f"latent_dim={latent_dim}, predictors={n_envs}")
                if cable_latent_use_rope_markers:
                    print("  [CableLatPred-DISTILL] rope marker features "
                          "enabled: "
                          f"source={_rope_marker_feature_source_config(config)}")
                _try_load_cable_latent_predictor_checkpoint(
                    cable_latent_predictors, config, None,
                    label="CableLatPred-DISTILL")
                for i, p in enumerate(cable_latent_predictors):
                    if p is None:
                        continue
                    p.reset(_phase_visible_no_cable_vector(
                        phase_obs_cache[i],
                        rope_marker_features=rope_marker_feature_cache[i]
                        if cable_latent_use_rope_markers else None))
                    policy_phase_obs_cache[i], _clp_info = (
                        _phase_obs_with_predicted_cable_latent(
                            phase_obs_cache[i], p,
                            cable_latent_last_actions[i], student, config,
                            phase=phase,
                            target_phase_obs=phase_obs_cache[i],
                            rope_marker_features=rope_marker_feature_cache[i]
                            if cable_latent_use_rope_markers else None))
                    if _clp_info is not None and _clp_info.trained:
                        cable_latent_pred_updates[i] += 1
                        cable_latent_pred_rmse[i] += float(_clp_info.rmse)

        ts = 0
        ep_count = 0
        best_loss = float("inf")
        batch_obs = []
        batch_actions = []
        loss_window = deque(maxlen=200)
        t0 = time.time()

        while ts < T:
            teacher_actions = []
            for i in range(n_envs):
                _core, _cable, _wind, _, _, _base = phase_obs_cache[i]

                teacher_po = teacher.encode_obs(_core, _cable, _wind)
                teacher_no = teacher.normalize_obs(teacher_po, update=False)
                t_act, _, _ = teacher.act_with_history(
                    teacher_no, teacher_histories[i],
                    deterministic=deterministic_teacher)

                _score, _scable, _swind, _, _, _ = policy_phase_obs_cache[i]
                student_po = student.encode_obs(_score, _scable, _swind)
                student_po = adaptation_states[i].append_features(student_po)
                student_no = student.normalize_obs(student_po, update=True)
                student_histories[i].push(student_no)
                if student.use_lstm:
                    batch_obs.append(student_histories[i].get_sequence())
                else:
                    batch_obs.append(student_no.copy())
                batch_actions.append(np.asarray(t_act, dtype=np.float32).copy())
                teacher_actions.append(t_act)

            if len(batch_obs) >= batch_size:
                loss = _distill_actor_step(student, batch_obs, batch_actions)
                if loss is not None:
                    loss_window.append(loss)
                batch_obs.clear()
                batch_actions.clear()

            payloads = []
            for i in range(n_envs):
                payloads.append({
                    'phase': phase,
                    'rl_action': teacher_actions[i],
                    'obs': obs_list[i],
                    'current_q': vec.get_qpos(i),
                    'start_xy': sxy_list[i],
                    'target_xy': txy_list[i],
                    'prev_tilt': pt_list[i],
                    'prev_yaw': py_list[i],
                    'rstate': rstate_list[i],
                    'act_noise': perts[i]["act_noise"],
                    'base_dq': phase_obs_cache[i][5],
                    'train_reject_lucky_rebar_insert':
                        _lucky_reject_enabled(config),
                })

            if hasattr(vec, 'remotes'):
                for i, p in enumerate(payloads):
                    vec.remotes[i].send(('rl_step', p))
                results = [vec._check_recv(vec.remotes[i].recv(), i)
                           for i in range(n_envs)]
            else:
                results = [vec.rl_step(i, payloads[i]) for i in range(n_envs)]

            for i, res in enumerate(results):
                _update_vision_acc(vision_acc[i], res.get('info', {}))
                if cable_latent_use_rope_markers:
                    _rmf = (res.get('info', {}) or {}).get(
                        "rope_marker_features", None)
                    if _rmf is not None:
                        rope_marker_feature_cache[i] = _rmf
                adaptation_states[i].record_action(res.get('delta_q', None))
                obs_list[i] = res['new_obs']
                rstate_list[i] = res['rstate']
                done = bool(res['done'])
                mx = int(config[f"{phase}_rl"]["max_steps"])
                if ep_steps[i] >= mx - 1:
                    done = True
                    if res.get('termination') == 'running':
                        res['termination'] = 'timeout'
                ep_suc[i] = ep_suc[i] or bool(res['success'])
                if done:
                    ep_term[i] = res.get('termination', 'done')
                ep_rewards[i] += float(res['reward'])
                ep_steps[i] += 1
                ts += 1
                student.total_steps = ts

                next_phase_obs = None
                if not done:
                    if res.get('new_core_obs') is not None:
                        next_phase_obs = (
                            res['new_core_obs'], res['new_cable_raw'],
                            res['new_wind_obs'], res['new_tilt'],
                            res['new_yaw'], res.get('new_base_dq'))
                    else:
                        next_phase_obs = vec.build_phase_obs_remote(
                            i, phase, obs_list[i], sxy_list[i], txy_list[i],
                            pt_list[i], py_list[i])
                if not done and next_phase_obs is not None:
                    phase_obs_cache[i] = next_phase_obs
                    pt_list[i] = next_phase_obs[3]
                    py_list[i] = next_phase_obs[4]
                    policy_phase_obs_cache[i] = next_phase_obs
                    if cable_latent_predictors[i] is not None:
                        policy_phase_obs_cache[i], _clp_info = (
                            _phase_obs_with_predicted_cable_latent(
                                next_phase_obs, cable_latent_predictors[i],
                                res.get('delta_q', None), student, config,
                                phase=phase,
                                target_phase_obs=next_phase_obs,
                                rope_marker_features=rope_marker_feature_cache[i]
                                if cable_latent_use_rope_markers else None))
                        if _clp_info is not None and _clp_info.trained:
                            cable_latent_pred_updates[i] += 1
                            cable_latent_pred_rmse[i] += float(_clp_info.rmse)
                    if res.get('delta_q', None) is not None:
                        _last_dq = np.asarray(
                            res.get('delta_q'), dtype=np.float32).reshape(-1)
                        if _last_dq.size < cable_latent_action_dim:
                            _last_dq = np.pad(
                                _last_dq,
                                (0, cable_latent_action_dim - _last_dq.size))
                        cable_latent_last_actions[i] = (
                            _last_dq[:cable_latent_action_dim].astype(np.float32))

                if done or ts >= T:
                    ep_count += 1
                    curs[i].update(ep_suc[i])
                    cur_info = curs[i].info()
                    vision_summary = _summarize_vision_acc(vision_acc[i])
                    loss_avg = float(np.mean(loss_window)) if loss_window else 0.0
                    last_loss = float(loss_window[-1]) if loss_window else 0.0
                    mark = "✅" if ep_suc[i] else "❌"
                    print(f"[distill w{i}] Ep{ep_count:4d} [{ts:7d}] "
                          f"{mark} R:{ep_rewards[i]:6.2f} "
                          f"L:{last_loss:.5f}/{loss_avg:.5f} "
                          f"S:{ep_steps[i]:3d} W:{perts[i].get('wind_speed', 0.0):.2f} "
                          f"L{curs[i].level_idx}/{curs[i].n_levels-1} | {ep_term[i]}")
                    if cable_latent_predictors[i] is not None:
                        print(f"       CableLatPred w{i} "
                              f"rmse:{(cable_latent_pred_rmse[i] / max(cable_latent_pred_updates[i], 1)):.4f} "
                              f"updates:{cable_latent_pred_updates[i]}")
                    with open(lf, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([
                            ep_count, ts, f"{ep_rewards[i]:.3f}",
                            int(ep_suc[i]), ep_steps[i],
                            f"{last_loss:.8f}", f"{loss_avg:.8f}",
                            curs[i].level_idx,
                            f"{perts[i].get('wind_speed', 0.0):.4f}",
                            f"{perts[i].get('wind_min', cur_info[f'cur/{phase}/wind_min']):.4f}",
                            f"{perts[i].get('wind_max', cur_info[f'cur/{phase}/wind_max']):.4f}",
                            f"{cur_info[f'cur/{phase}/sr_window']:.3f}",
                            curs[i].eps_at_level, i, ep_term[i],
                            f"{vision_summary.get('valid_rate', 0.0):.4f}",
                            f"{vision_summary.get('active_cameras_mean', 0.0):.3f}",
                            f"{vision_summary.get('reprojection_px_mean', 0.0):.4f}",
                            f"{vision_summary.get('reprojection_px_p95', 0.0):.4f}",
                            f"{vision_summary.get('depth_rmse_mm_mean', 0.0):.4f}",
                            f"{vision_summary.get('depth_rmse_mm_p95', 0.0):.4f}",
                            int(vision_summary.get('failure_steps', 0)),
                        ])
                    logger.log(ep_count, {
                        f"distill/{phase}/loss": last_loss,
                        f"distill/{phase}/loss_window": loss_avg,
                        f"distill/{phase}/teacher_success": float(ep_suc[i]),
                        f"distill/{phase}/teacher_reward": ep_rewards[i],
                        f"distill/{phase}/teacher_steps": ep_steps[i],
                        f"distill/{phase}/level": curs[i].level_idx,
                        f"distill/{phase}/wind_speed":
                            perts[i].get('wind_speed', 0.0),
                        f"distill/{phase}/vision_valid_rate":
                            vision_summary.get('valid_rate', 0.0),
                        f"cable_latent_pred/{phase}/rmse":
                            (cable_latent_pred_rmse[i] /
                             max(cable_latent_pred_updates[i], 1)),
                        f"cable_latent_pred/{phase}/updates":
                            cable_latent_pred_updates[i],
                    })
                    if ep_count > 0 and ep_count % SI == 0:
                        save_checkpoint(
                            student, log_dir, ep_count, tag="latest",
                            cable_latent_predictor=cable_latent_predictors)
                    if loss_window and loss_avg < best_loss:
                        best_loss = loss_avg
                        save_checkpoint(
                            student, log_dir, ep_count, tag="best",
                            cable_latent_predictor=cable_latent_predictors)

                    obs_list[i], sxy_list[i], txy_list[i], rstate_list[i], perts[i] = \
                        _reset_one_env(i)
                    pt_list[i] = 0.0
                    py_list[i] = 0.0
                    ep_rewards[i] = 0.0
                    ep_steps[i] = 0
                    ep_suc[i] = False
                    ep_term[i] = "running"
                    student_histories[i].reset()
                    teacher_histories[i].reset()
                    adaptation_states[i].reset()
                    vision_acc[i] = _new_vision_acc()
                    phase_obs_cache[i] = vec.build_phase_obs_remote(
                        i, phase, obs_list[i], sxy_list[i], txy_list[i],
                        pt_list[i], py_list[i])
                    policy_phase_obs_cache[i] = phase_obs_cache[i]
                    if cable_latent_use_rope_markers and hasattr(
                            vec, "get_rope_marker_features"):
                        rope_marker_feature_cache[i] = (
                            vec.get_rope_marker_features(i))
                    cable_latent_last_actions[i] = np.zeros(
                        cable_latent_action_dim, dtype=np.float32)
                    cable_latent_pred_updates[i] = 0
                    cable_latent_pred_rmse[i] = 0.0
                    if cable_latent_predictors[i] is not None:
                        cable_latent_predictors[i].reset(
                            _phase_visible_no_cable_vector(
                                phase_obs_cache[i],
                                rope_marker_features=rope_marker_feature_cache[i]
                                if cable_latent_use_rope_markers else None))
                        policy_phase_obs_cache[i], _clp_info = (
                            _phase_obs_with_predicted_cable_latent(
                                phase_obs_cache[i], cable_latent_predictors[i],
                                cable_latent_last_actions[i], student, config,
                                phase=phase,
                                target_phase_obs=phase_obs_cache[i],
                                rope_marker_features=rope_marker_feature_cache[i]
                                if cable_latent_use_rope_markers else None))
                        if _clp_info is not None and _clp_info.trained:
                            cable_latent_pred_updates[i] += 1
                            cable_latent_pred_rmse[i] += float(_clp_info.rmse)

        if batch_obs:
            loss = _distill_actor_step(student, batch_obs, batch_actions)
            if loss is not None:
                loss_window.append(loss)
        save_checkpoint(student, log_dir, ep_count, tag="final",
                        cable_latent_predictor=cable_latent_predictors)
        print(f"\n[{phase.upper()}-DISTILL] Done: {ts} samples, "
              f"{(time.time()-t0)/60:.1f} min, best_loss={best_loss:.6f}")
        try:
            logger.close()
        except Exception:
            pass
        return student
    finally:
        vec.close()


def train(phase, log_dir, algo="ppo", custom_config=None, resume_ckpt=None):
    """[v11] BC 完全移除. cruise 残差用 actor 输出层零初始化 (自动); descent 用 PID 残差.

    Args:
        phase:        "cruise" / "descent"
        log_dir:      日志和 checkpoint 目录
        algo:         "ppo" 或 "sac"
        custom_config: 覆盖 DEFAULT_CONFIG 的字段 (CLI 参数)
        resume_ckpt:   断点续训用的 checkpoint 路径 (可选)
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    if custom_config:
        _deep_update(config, custom_config)
    ins_cfg = config.setdefault("insertion", {})
    if bool(ins_cfg.get("strict_lucky_reject_always", True)):
        ins_cfg["train_reject_lucky_rebar_insert"] = True
        ins_cfg["lucky_reject_auto_enable"] = False
    set_global_seed(config["train"].get("seed", 42))
    os.makedirs(log_dir, exist_ok=True)
    _cf = float(config.get("sim", {}).get("control_freq_hz", 10.0))
    print(f"  [control] {phase}: {_cf:.1f}Hz, "
          f"action_dt={1.0 / max(_cf, 1e-6):.3f}s, "
          f"controller.dt={float(config.get('controller', {}).get('dt', 0.1)):.3f}s, "
          f"ee_dt={float(config.get('ee_control', {}).get('integrator_dt', 0.1)):.3f}s, "
          f"max_steps={int(config.get(f'{phase}_rl', {}).get('max_steps', 0))}")
    ins_cfg = config.get("insertion", {})
    if bool(ins_cfg.get("strict_lucky_reject_always", True)):
        print("  [LuckyReject] strict always: lucky insert is never counted as success")
    elif bool(ins_cfg.get("lucky_reject_auto_enable", False)):
        print("  [LuckyReject] bootstrap: "
              f"enabled_now={_lucky_reject_enabled(config)}, "
              f"auto_enable_sr={float(ins_cfg.get('lucky_reject_enable_sr', 0.70)):.2f}, "
              f"min_window={int(ins_cfg.get('lucky_reject_min_window', 1))}")

    n_envs = int(config["train"].get("n_envs", 1))
    if algo == "obs_pred":
        if n_envs < 1:
            n_envs = 1
        return train_obs_predictor_pretrain(
            phase, log_dir, config, resume_ckpt=resume_ckpt, n_envs=n_envs)
    if algo == "distill":
        if n_envs < 1:
            n_envs = 1
        teacher_ckpt = config.get("policy_distill", {}).get(
            "teacher_ckpt", resume_ckpt)
        return train_policy_distill(
            phase, log_dir, config, teacher_ckpt=teacher_ckpt, n_envs=n_envs)
    if algo == "ppo":
        # [v11 Path 3] n_envs > 1 时启用 SubprocVecEnv 并行版本
        if n_envs > 1:
            print(f"\n  [v11 Path 3] 启用 vectorized PPO with n_envs={n_envs}")
            return train_ppo_vec(phase, log_dir, config,
                                 resume_ckpt=resume_ckpt, n_envs=n_envs)
        return train_ppo(phase, log_dir, config, resume_ckpt=resume_ckpt)
    elif algo == "sac":
        # [v11 Path 3] SAC 也支持 vec
        if n_envs > 1:
            print(f"\n  [v11 Path 3] 启用 vectorized SAC with n_envs={n_envs}")
            return train_sac_vec(phase, log_dir, config,
                                 resume_ckpt=resume_ckpt, n_envs=n_envs)
        return train_sac(phase, log_dir, config, resume_ckpt=resume_ckpt)
    else:
        raise ValueError(f"Unknown algo: {algo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="[v13.0] 两阶段 RL 训练: cruise (NMPC 抬升+平移) + descent (PID 下降)")
    parser.add_argument("--profile",   type=str, default=None,
                        help="named training profile from DEFAULT_CONFIG['train_profiles']")
    parser.add_argument("--phase",     type=str, default=None,
                        choices=["cruise", "descent"],
                        help="train cruise or descent")
    parser.add_argument("--algo",      type=str, default=None,
                        choices=["ppo", "sac", "obs_pred", "distill"])
    parser.add_argument("--log-dir",   type=str, default=None)
    parser.add_argument("--resume-in-place", action="store_true",
                        help="write into --log-dir exactly; default creates a dated copy dir")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--render",    action="store_true")
    parser.add_argument("--gpu",       type=int, default=0)
    parser.add_argument("--resume-ckpt", type=str, default=None,
                        help="[v11] 断点续训 checkpoint 路径 (替代旧 --bc-ckpt)")
    parser.add_argument("--teacher-ckpt", type=str, default=None,
                        help="teacher policy checkpoint for --algo distill")
    parser.add_argument("--distill-batch-size", type=int, default=None,
                        help="supervised policy distillation batch size")
    parser.add_argument("--reset-optimizer-on-resume", action="store_true",
                        help="reset Adam state after loading --resume-ckpt")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--no-curriculum", action="store_true",
                        help="禁用课程学习 (level 永远停在 0)")
    parser.add_argument("--target-xy-randomize", action="store_true",
                        help="enable endpoint target XY randomization for training")
    parser.add_argument("--disable-target-xy-randomize", action="store_true",
                        help="force fixed endpoint target XY for training")
    parser.add_argument("--target-xy-range", type=str, default=None,
                        help="endpoint target XY randomization half-range; scalar or x,y in meters")
    parser.add_argument("--n-envs",    type=int, default=None,
                        help="[v11 Path 3] 并行环境数 (默认 1 = 单进程, "
                             ">1 = SubprocVecEnv 多进程)")
    parser.add_argument("--control-freq-hz", type=float, default=None,
                        help="train-time residual+controller frequency; 5 means one action every 0.2s")
    parser.add_argument("--keep-step-budget", action="store_true",
                        help="do not scale max_steps when overriding control frequency")
    parser.add_argument("--scale-limits-with-control-dt", action="store_true",
                        help="scale per-step dq/rate limits with control period to preserve per-second limits")
    parser.add_argument("--obs-predictor", action="store_true",
                        help="enable LSTM observation predictor for delayed real-device observations")
    parser.add_argument("--disable-obs-predictor", action="store_true",
                        help="force-disable the observation predictor")
    parser.add_argument("--obs-period", type=int, default=None,
                        help="true environment observation period in control steps")
    parser.add_argument("--delay-mdp", action="store_true",
                        help="use held observations with obs-age/action-history features; no predictor")
    parser.add_argument("--delay-action-history-steps", type=int, default=None,
                        help="number of executed joint actions appended to Delay-MDP observations")
    parser.add_argument("--disable-delay-obs-age", action="store_true",
                        help="do not append the normalized observation age feature")
    parser.add_argument("--vision", action="store_true",
                        help="enable RGB-D vision payload-state observations")
    parser.add_argument("--disable-vision", action="store_true",
                        help="force-disable RGB-D vision observations")
    parser.add_argument("--vision-latency-steps", type=int, default=None,
                        help="sensor latency in control steps for RGB-D vision")
    parser.add_argument("--vision-processing-delay-steps", type=int, default=None,
                        help="compute delay in control steps for RGB-D fusion")
    parser.add_argument("--vision-period", type=int, default=None,
                        help="RGB-D measurement period in control steps")
    parser.add_argument("--vision-pos-noise", type=float, default=None,
                        help="isotropic XY position noise std for RGB-D vision (m)")
    parser.add_argument("--vision-dropout", type=float, default=None,
                        help="per-camera RGB-D dropout probability")
    parser.add_argument("--zero-cable-obs", action="store_true",
                        help="keep cable latent dims but feed zeros for ablation")
    parser.add_argument("--disable-cable-obs", action="store_true",
                        help="remove cable latent dims; requires matching obs_dim/checkpoint")
    parser.add_argument("--asymmetric-critic", action="store_true",
                        help="train actor on deploy obs while critic sees full cable-latent obs")
    parser.add_argument("--critic-cable-encoder-ckpt", type=str, default=None,
                        help="checkpoint providing the frozen cable encoder for asymmetric critic")
    parser.add_argument("--adaptation-history", action="store_true",
                        help="deprecated alias for --cable-latent-predictor")
    parser.add_argument("--adapt-history-steps", type=int, default=None,
                        help="deprecated; kept for old commands")
    parser.add_argument("--adapt-history-action-dim", type=int, default=None,
                        help="action dimension for CableLatPred history input")
    parser.add_argument("--cable-latent-predictor", action="store_true",
                        help="estimate the 32D cable latent from visible observations/history")
    parser.add_argument("--cable-latent-use-rope-markers", action="store_true",
                        help="append reconstructed rope marker features to CableLatPred input")
    parser.add_argument("--rope-marker-feature-source", type=str, default=None,
                        choices=[
                            "site", "rgbd", "vision", "backproject",
                            "backprojection", "opencv_rgbd", "opencv",
                            "color_rgbd", "color"],
                        help=("rope marker feature source: oracle site, ideal "
                              "RGB-D back-projection, or OpenCV RGB-D detection"))
    parser.add_argument("--rope-marker-pos-noise", type=float, default=None,
                        help="isotropic lab-frame marker position noise std in meters")
    parser.add_argument("--rope-marker-dropout", type=float, default=None,
                        help="per-marker dropout probability after visibility/depth checks")
    parser.add_argument("--rope-marker-outlier-prob", type=float, default=None,
                        help="per-marker outlier probability after visibility/depth checks")
    parser.add_argument("--rope-marker-outlier-std", type=float, default=None,
                        help="isotropic lab-frame outlier noise std in meters")
    parser.add_argument("--rope-marker-pixel-noise", type=float, default=None,
                        help="marker center pixel noise std before depth back-projection")
    parser.add_argument("--rope-marker-quantize-px", action="store_true",
                        help="round marker center detections to integer pixels before back-projection")
    parser.add_argument("--disable-cable-latent-predictor", action="store_true",
                        help="force-disable the cable latent predictor")
    parser.add_argument("--cable-latent-predictor-ckpt", type=str, default=None,
                        help="explicit CableLatPred checkpoint to load")
    parser.add_argument("--cable-latent-predictor-lr", type=float, default=None,
                        help="override CableLatPred learning rate")
    parser.add_argument("--cable-latent-predictor-hidden-dim", type=int, default=None,
                        help="override CableLatPred recurrent hidden dimension")
    parser.add_argument("--cable-latent-predictor-huber-coef", type=float, default=None,
                        help="override CableLatPred Huber loss coefficient")
    parser.add_argument("--cable-latent-predictor-mse-coef", type=float, default=None,
                        help="override CableLatPred MSE loss coefficient")
    parser.add_argument("--obs-predictor-lr", type=float, default=None,
                        help="override observation predictor learning rate")
    parser.add_argument("--obs-predictor-target-mode", type=str, default=None,
                        choices=["raw", "non_cable_latent"],
                        help="predictor target: full raw env obs, or non-cable raw obs + cable latent")
    parser.add_argument("--obs-predictor-non-cable-weight", type=float, default=None,
                        help="loss weight for non-cable predictor target dimensions")
    parser.add_argument("--obs-predictor-cable-latent-weight", type=float, default=None,
                        help="loss weight for cable-latent predictor target dimensions")
    parser.add_argument("--obs-predictor-nll-coef", type=float, default=None,
                        help="observation predictor Gaussian NLL loss coefficient")
    parser.add_argument("--obs-predictor-huber-coef", type=float, default=None,
                        help="observation predictor Huber loss coefficient")
    parser.add_argument("--obs-predictor-log-std-min", type=float, default=None,
                        help="minimum predictor log std; higher values reduce overconfidence")
    parser.add_argument("--obs-predictor-log-std-max", type=float, default=None,
                        help="maximum predictor log std")
    parser.add_argument("--obs-predictor-nll-error-clip", type=float, default=None,
                        help="clip standardized residual inside predictor NLL; <=0 disables")
    parser.add_argument("--obs-predictor-grad-clip", type=float, default=None,
                        help="override observation predictor gradient clipping norm")
    parser.add_argument("--obs-predictor-ckpt", type=str, default=None,
                        help="explicit observation predictor checkpoint to load")
    parser.add_argument("--obs-pred-pretrain-wind-min", type=float, default=None,
                        help="minimum wind speed during obs_pred pretraining")
    parser.add_argument("--obs-pred-pretrain-stochastic-policy",
                        action="store_true",
                        help="use stochastic residual policy actions in obs_pred pretraining")
    parser.add_argument("--obs-pred-pretrain-full-curriculum",
                        action="store_true",
                        help="use normal curriculum instead of final-level obs_pred pretraining")
    parser.add_argument("--payload-mass", type=float, default=None,
                        help="override payload mass in kg")
    parser.add_argument("--wind-speed-max", type=float, default=None,
                        help="override wind speed curriculum cap in m/s")
    parser.add_argument("--variable-wind", action="store_true",
                        help="vary wind continuously around each sampled episode wind")
    parser.add_argument("--wind-speed-band-abs", type=float, default=None,
                        help="variable wind absolute speed band (m/s)")
    parser.add_argument("--wind-speed-band-frac", type=float, default=None,
                        help="variable wind fractional speed band")
    parser.add_argument("--wind-speed-rate-std", type=float, default=None,
                        help="variable wind speed random-walk std")
    parser.add_argument("--wind-dir-band-rad", type=float, default=None,
                        help="variable wind direction band around initial direction (rad)")
    parser.add_argument("--wind-dir-rate-std", type=float, default=None,
                        help="variable wind direction random-walk std")
    parser.add_argument("--curriculum-from-log", type=str, default=None,
                        help="initialize curriculum wind/level from a previous run CSV or run dir")
    parser.add_argument("--curriculum-start-wind", type=float, default=None,
                        help="initialize the continuous curriculum wind cap in m/s")
    parser.add_argument("--curriculum-start-level", type=int, default=None,
                        help="initialize curriculum level; inferred from wind if omitted")
    parser.add_argument("--curriculum-ramp-episodes", type=int, default=None,
                        help="override successful episodes needed to ramp min wind to max wind")
    parser.add_argument("--curriculum-ramp-sr-threshold", type=float, default=None,
                        help="override SR threshold for full curriculum wind ramp")
    parser.add_argument("--curriculum-ramp-warmup-sr-threshold", type=float, default=None,
                        help="override SR threshold for partial curriculum wind ramp")
    parser.add_argument("--curriculum-ramp-warmup-scale", type=float, default=None,
                        help="override partial curriculum wind ramp scale")
    parser.add_argument("--curriculum-ramp-min-window", type=int, default=None,
                        help="override minimum episodes before wind ramp can move")
    parser.add_argument("--high-wind-focus", action="store_true",
                        help="sample curriculum wind from a rising lower bound for high-wind finetuning")
    parser.add_argument("--wind-focus-start-level", type=int, default=None,
                        help="level where high-wind focused sampling starts")
    parser.add_argument("--wind-focus-full-level", type=int, default=None,
                        help="level where the post-ramp lower-bound schedule starts")
    parser.add_argument("--wind-focus-start-min", type=float, default=None,
                        help="wind lower bound at the focus start level")
    parser.add_argument("--wind-focus-full-min", type=float, default=None,
                        help="wind lower bound at the focus full level")
    parser.add_argument("--wind-focus-final-min", type=float, default=None,
                        help="final wind lower bound once curriculum wind max reaches its cap")
    parser.add_argument("--ppo-lr-actor", type=float, default=None,
                        help="override PPO actor learning rate, also after checkpoint resume")
    parser.add_argument("--ppo-lr-critic", type=float, default=None,
                        help="override PPO critic learning rate, also after checkpoint resume")
    parser.add_argument("--freeze-obs-norm", action="store_true",
                        help="keep checkpoint observation normalization fixed during PPO")
    parser.add_argument("--descent-residual-dq-scale", type=float, default=None,
                        help="override descent residual dq scale")
    parser.add_argument("--descent-residual-acc-xy", type=float, default=None,
                        help="override descent residual xy acceleration cap")
    parser.add_argument("--descent-residual-acc-z", type=float, default=None,
                        help="override descent residual z acceleration cap")
    parser.add_argument("--descent-action-rms-free", type=float, default=None,
                        help="override free normalized action RMS before penalty")
    parser.add_argument("--descent-action-magnitude-coef", type=float, default=None,
                        help="override descent action magnitude penalty coefficient")
    parser.add_argument("--descent-z-soft-gate-full", type=float, default=None,
                        help="XY error below which descent PID uses full z speed")
    parser.add_argument("--descent-z-hard-gate", type=float, default=None,
                        help="XY error above which descent PID blocks z speed except trickle")
    parser.add_argument("--descent-z-min-speed-frac", type=float, default=None,
                        help="minimum z-speed fraction used in the trickle gate")
    parser.add_argument("--descent-z-trickle-xy-gate", type=float, default=None,
                        help="outer XY gate where descent z trickle tapers to zero")
    parser.add_argument("--descent-base-v-max-z", type=float, default=None,
                        help="override descent PID maximum payload z descent speed")
    parser.add_argument("--descent-alignment-z-gate", type=float, default=None,
                        help="XY gate for descent z-progress reward")
    parser.add_argument("--descent-premature-descent-xy-gate", type=float, default=None,
                        help="XY gate for premature descent penalty")
    parser.add_argument("--disable-lucky-until-sr", type=float, default=None,
                        help="deprecated no-op; lucky insert rejection now stays strict")
    parser.add_argument("--lucky-reject-min-window", type=int, default=None,
                        help="minimum curriculum SR window size before auto-enabling lucky insert rejection")
    args = parser.parse_args()

    profile = None
    cc = {}
    if args.profile:
        profiles = DEFAULT_CONFIG.get("train_profiles", {})
        if args.profile not in profiles:
            names = ", ".join(sorted(profiles)) or "(none)"
            raise ValueError(
                f"unknown --profile {args.profile!r}; available profiles: {names}")
        profile = copy.deepcopy(profiles[args.profile])
        desc = str(profile.get("description", "")).strip()
        print(f"[profile] {args.profile}" + (f": {desc}" if desc else ""))
        _deep_update(cc, profile.get("config", {}))
        if args.phase is None:
            args.phase = profile.get("phase")
        if args.algo is None:
            args.algo = profile.get("algo", "ppo")
        if args.log_dir is None and profile.get("log_dir"):
            args.log_dir = profile.get("log_dir")
        if args.resume_ckpt is None and profile.get("resume_ckpt"):
            args.resume_ckpt = profile.get("resume_ckpt")
        if (args.curriculum_from_log is None and
                profile.get("curriculum_from_log")):
            args.curriculum_from_log = profile.get("curriculum_from_log")

    if args.phase is None:
        raise ValueError("--phase is required unless --profile supplies it")
    if args.algo is None:
        args.algo = "ppo"

    if args.log_dir is None and args.phase == "descent" and args.algo == "ppo":
        base_ld = "saves/descent_ppo_physical"
    else:
        base_ld = args.log_dir or f"saves/{args.phase}_{args.algo}"
    if args.resume_in_place:
        ld = base_ld
        print(f"[log-dir] resume-in-place: saving directly to {ld}")
    else:
        ld = make_dated_log_dir(base_ld)
        print(f"[log-dir] dated output: {ld}")
    if args.phase == "descent" and args.algo == "ppo":
        old_descent_dir = os.path.normpath("saves/descent_ppo")
        if os.path.normpath(ld) == old_descent_dir:
            raise ValueError(
                "拒绝覆盖 saves/descent_ppo。请使用新的 --log-dir, "
                "例如 saves/descent_ppo_physical_next")
    if args.render:    cc.setdefault("sim", {})["render"]   = True
    if args.gpu != 0:  cc.setdefault("train", {})["gpu_id"] = args.gpu
    if args.timesteps: cc.setdefault("train", {})["total_timesteps"] = args.timesteps
    cc.setdefault("train", {})["seed"] = args.seed
    if args.teacher_ckpt is not None:
        cc.setdefault("policy_distill", {})["teacher_ckpt"] = str(args.teacher_ckpt)
    if args.distill_batch_size is not None:
        cc.setdefault("policy_distill", {})["batch_size"] = int(
            args.distill_batch_size)
    if args.no_curriculum:
        cc.setdefault("curriculum", {})["enabled"] = False
    task_cc = cc.setdefault("task", {})
    if bool(args.disable_target_xy_randomize):
        task_cc["target_xy_randomize"] = False
        task_cc["target_xy_range"] = [0.0, 0.0]
    elif bool(args.target_xy_randomize):
        task_cc["target_xy_randomize"] = True
    target_xy_range = parse_xy_range_arg(args.target_xy_range)
    if target_xy_range is not None:
        task_cc["target_xy_range"] = target_xy_range
        if max(target_xy_range) > 0.0 and not bool(args.disable_target_xy_randomize):
            task_cc["target_xy_randomize"] = True
    if args.n_envs is not None:
        cc.setdefault("train", {})["n_envs"] = args.n_envs
    if args.reset_optimizer_on_resume:
        cc.setdefault("train", {})["reset_optimizer_on_resume"] = True
    if profile is not None:
        apply_control_frequency_override(
            cc, profile.get("control_freq_hz"),
            keep_step_budget=bool(profile.get("keep_step_budget", False)),
            scale_limits_with_dt=bool(
                profile.get("scale_limits_with_control_dt", False)),
            phase=args.phase)
    apply_control_frequency_override(
        cc, args.control_freq_hz,
        keep_step_budget=bool(args.keep_step_budget),
        scale_limits_with_dt=bool(args.scale_limits_with_control_dt),
        phase=args.phase)
    if args.ppo_lr_actor is not None:
        cc.setdefault("ppo", {})["lr_actor"] = float(args.ppo_lr_actor)
    if args.ppo_lr_critic is not None:
        cc.setdefault("ppo", {})["lr_critic"] = float(args.ppo_lr_critic)
    if args.freeze_obs_norm:
        cc.setdefault("ppo", {})["freeze_obs_norm"] = True
    if args.obs_predictor:
        cc.setdefault("observation_predictor", {})["enabled"] = True
    if args.algo == "obs_pred":
        cc.setdefault("observation_predictor", {})["enabled"] = True
    if args.delay_mdp:
        cc.setdefault("delay_mdp", {})["enabled"] = True
        cc.setdefault("observation_predictor", {})["enabled"] = False
    if args.disable_obs_predictor:
        cc.setdefault("observation_predictor", {})["enabled"] = False
    if args.obs_period is not None:
        cc.setdefault("observation_predictor", {})["measurement_period_steps"] = int(
            args.obs_period)
        cc.setdefault("delay_mdp", {})["measurement_period_steps"] = int(
            args.obs_period)
    if args.delay_action_history_steps is not None:
        cc.setdefault("delay_mdp", {})["action_history_steps"] = int(
            args.delay_action_history_steps)
    if args.disable_delay_obs_age:
        cc.setdefault("delay_mdp", {})["include_obs_age"] = False
    if args.vision:
        cc.setdefault("vision", {})["enabled"] = True
    if args.disable_vision:
        cc.setdefault("vision", {})["enabled"] = False
    if args.vision_latency_steps is not None:
        cc.setdefault("vision", {})["latency_steps"] = int(
            args.vision_latency_steps)
    if args.vision_processing_delay_steps is not None:
        cc.setdefault("vision", {})["processing_delay_steps"] = int(
            args.vision_processing_delay_steps)
    if args.vision_period is not None:
        cc.setdefault("vision", {})["measurement_period_steps"] = int(
            args.vision_period)
    if args.vision_pos_noise is not None:
        _vn = max(0.0, float(args.vision_pos_noise))
        cc.setdefault("vision", {})["position_noise_std"] = [_vn, _vn, _vn * 1.5]
    if args.vision_dropout is not None:
        cc.setdefault("vision", {})["dropout_prob"] = float(args.vision_dropout)
    if args.zero_cable_obs:
        ce_cc = cc.setdefault("cable_encoder", {})
        ce_cc["enabled"] = True
        ce_cc["zero_obs"] = True
        ce_cc.setdefault("output_dim", 32)
    if args.disable_cable_obs:
        ce_cc = cc.setdefault("cable_encoder", {})
        ce_cc["enabled"] = False
        ce_cc["zero_obs"] = True
        ce_cc["output_dim"] = 0
        if args.phase == "descent":
            cc.setdefault("descent_rl", {})["obs_dim"] = 44
        elif args.phase == "cruise":
            cc.setdefault("cruise_rl", {})["obs_dim"] = 45
    if args.asymmetric_critic:
        ac_cc = cc.setdefault("asymmetric_critic", {})
        ac_cc["enabled"] = True
        if args.phase in ("descent", "cruise"):
            ac_cc.setdefault("critic_obs_dim", int(
                DEFAULT_CONFIG.get(f"{args.phase}_rl", {}).get("obs_dim",
                                                               cc.get(f"{args.phase}_rl", {}).get("obs_dim", 0))))
    if args.critic_cable_encoder_ckpt is not None:
        cc.setdefault("asymmetric_critic", {})["cable_encoder_ckpt"] = str(
            args.critic_cable_encoder_ckpt)
    cable_latent_pred_requested = (
        args.cable_latent_predictor or args.adaptation_history or
        args.adapt_history_steps is not None
    )
    if cable_latent_pred_requested:
        clp_cc = cc.setdefault("cable_latent_predictor", {})
        clp_cc["enabled"] = True
        ce_cc = cc.setdefault("cable_encoder", {})
        ce_cc["enabled"] = True
        ce_cc["zero_obs"] = False
        ce_cc["output_dim"] = int(DEFAULT_CONFIG.get(
            "cable_encoder", {}).get("output_dim", 32))
        if args.phase in ("descent", "cruise"):
            cc.setdefault(f"{args.phase}_rl", {})["obs_dim"] = int(
                DEFAULT_CONFIG.get(f"{args.phase}_rl", {}).get(
                    "obs_dim", cc.get(f"{args.phase}_rl", {}).get("obs_dim", 0)))
        cc.setdefault("adaptation_history", {})["enabled"] = False
        if args.adapt_history_action_dim is not None:
            clp_cc["action_dim"] = int(args.adapt_history_action_dim)
    if args.cable_latent_use_rope_markers:
        clp_cc = cc.setdefault("cable_latent_predictor", {})
        clp_cc["use_rope_marker_features"] = True
        cc.setdefault("rope_markers", {})["enabled"] = True
    if args.rope_marker_feature_source is not None:
        _src = str(args.rope_marker_feature_source).strip().lower()
        cc.setdefault("rope_markers", {})["feature_source"] = _src
        cc.setdefault("cable_latent_predictor", {})[
            "rope_marker_feature_source"] = _src
    if args.rope_marker_pos_noise is not None:
        _rn = max(0.0, float(args.rope_marker_pos_noise))
        cc.setdefault("rope_markers", {})[
            "feature_pos_noise_std"] = [_rn, _rn, _rn]
    if args.rope_marker_dropout is not None:
        cc.setdefault("rope_markers", {})["feature_dropout_prob"] = min(
            1.0, max(0.0, float(args.rope_marker_dropout)))
    if args.rope_marker_outlier_prob is not None:
        cc.setdefault("rope_markers", {})["feature_outlier_prob"] = min(
            1.0, max(0.0, float(args.rope_marker_outlier_prob)))
    if args.rope_marker_outlier_std is not None:
        cc.setdefault("rope_markers", {})["feature_outlier_std"] = max(
            0.0, float(args.rope_marker_outlier_std))
    if args.rope_marker_pixel_noise is not None:
        cc.setdefault("rope_markers", {})["feature_pixel_noise_std"] = max(
            0.0, float(args.rope_marker_pixel_noise))
    if args.rope_marker_quantize_px:
        cc.setdefault("rope_markers", {})["feature_quantize_px"] = True
    if args.disable_cable_latent_predictor:
        cc.setdefault("cable_latent_predictor", {})["enabled"] = False
    if args.obs_predictor_lr is not None:
        cc.setdefault("observation_predictor", {})["lr"] = float(
            args.obs_predictor_lr)
    if args.obs_predictor_target_mode is not None:
        cc.setdefault("observation_predictor", {})["target_mode"] = str(
            args.obs_predictor_target_mode)
    if args.obs_predictor_non_cable_weight is not None:
        cc.setdefault("observation_predictor", {})[
            "non_cable_loss_weight"] = float(args.obs_predictor_non_cable_weight)
    if args.obs_predictor_cable_latent_weight is not None:
        cc.setdefault("observation_predictor", {})[
            "cable_latent_loss_weight"] = float(
                args.obs_predictor_cable_latent_weight)
    if args.obs_predictor_nll_coef is not None:
        cc.setdefault("observation_predictor", {})["nll_coef"] = float(
            args.obs_predictor_nll_coef)
    if args.obs_predictor_huber_coef is not None:
        cc.setdefault("observation_predictor", {})["huber_coef"] = float(
            args.obs_predictor_huber_coef)
    if args.obs_predictor_log_std_min is not None:
        cc.setdefault("observation_predictor", {})["log_std_min"] = float(
            args.obs_predictor_log_std_min)
    if args.obs_predictor_log_std_max is not None:
        cc.setdefault("observation_predictor", {})["log_std_max"] = float(
            args.obs_predictor_log_std_max)
    if args.obs_predictor_nll_error_clip is not None:
        cc.setdefault("observation_predictor", {})["nll_error_clip"] = float(
            args.obs_predictor_nll_error_clip)
    if args.obs_predictor_grad_clip is not None:
        cc.setdefault("observation_predictor", {})["grad_clip"] = float(
            args.obs_predictor_grad_clip)
    if args.obs_predictor_ckpt is not None:
        cc.setdefault("observation_predictor", {})["checkpoint"] = str(
            args.obs_predictor_ckpt)
    if args.cable_latent_predictor_lr is not None:
        cc.setdefault("cable_latent_predictor", {})["lr"] = float(
            args.cable_latent_predictor_lr)
    if args.cable_latent_predictor_hidden_dim is not None:
        cc.setdefault("cable_latent_predictor", {})["hidden_dim"] = int(
            args.cable_latent_predictor_hidden_dim)
    if args.cable_latent_predictor_huber_coef is not None:
        cc.setdefault("cable_latent_predictor", {})["huber_coef"] = float(
            args.cable_latent_predictor_huber_coef)
    if args.cable_latent_predictor_mse_coef is not None:
        cc.setdefault("cable_latent_predictor", {})["mse_coef"] = float(
            args.cable_latent_predictor_mse_coef)
    if args.cable_latent_predictor_ckpt is not None:
        cc.setdefault("cable_latent_predictor", {})["checkpoint"] = str(
            args.cable_latent_predictor_ckpt)
    if args.obs_pred_pretrain_wind_min is not None:
        cc.setdefault("observation_predictor", {})["pretrain_wind_min"] = float(
            args.obs_pred_pretrain_wind_min)
    if args.obs_pred_pretrain_stochastic_policy:
        cc.setdefault("observation_predictor", {})[
            "pretrain_policy_deterministic"] = False
    if args.obs_pred_pretrain_full_curriculum:
        cc.setdefault("observation_predictor", {})[
            "pretrain_final_curriculum"] = False
    if args.payload_mass is not None:
        cc.setdefault("prefab", {})["mass"] = float(args.payload_mass)
    if args.wind_speed_max is not None:
        _ws = float(args.wind_speed_max)
        cc.setdefault("wind", {})["speed_max"] = _ws
        cc.setdefault("wind_obs", {})["wind_speed_max"] = _ws
        cur_cc = cc.setdefault("curriculum", {})
        levels_key = f"{args.phase}_levels"
        base_levels = copy.deepcopy(DEFAULT_CONFIG.get("curriculum", {}).get(
            levels_key, []))
        if base_levels:
            base_max = max(float(lv.get("wind_max", 0.0)) for lv in base_levels)
            base_max = max(base_max, 1e-6)
            for lv in base_levels:
                lv["wind_max"] = min(
                    _ws, float(lv.get("wind_max", 0.0)) / base_max * _ws)
            cur_cc[levels_key] = base_levels
            cur_cc[f"{args.phase}_ramp_min_wind"] = min(
                _ws, float(base_levels[0].get("wind_max", 0.0)))
    wind_cfg = cc.setdefault("wind", {})
    if args.variable_wind:
        wind_cfg["test_wind_variable"] = True
    for arg_name, cfg_key in [
        ("wind_speed_band_abs", "test_speed_band_abs"),
        ("wind_speed_band_frac", "test_speed_band_frac"),
        ("wind_speed_rate_std", "test_speed_rate_std"),
        ("wind_dir_band_rad", "test_dir_band_rad"),
        ("wind_dir_rate_std", "test_dir_rate_std"),
    ]:
        val = getattr(args, arg_name, None)
        if val is not None:
            wind_cfg[cfg_key] = float(val)
    cur_cc = cc.setdefault("curriculum", {})
    if args.curriculum_from_log is not None:
        progress = _read_curriculum_progress(args.curriculum_from_log, args.phase)
        if progress is None:
            raise ValueError(
                f"could not read curriculum progress from {args.curriculum_from_log}")
        cur_cc[f"{args.phase}_start_wind"] = float(progress["wind"])
        if progress.get("level") is not None:
            cur_cc[f"{args.phase}_start_level"] = int(progress["level"])
        if progress.get("eps_at_level") is not None:
            cur_cc[f"{args.phase}_start_eps_at_level"] = int(
                progress["eps_at_level"])
        print("[curriculum] start from "
              f"{progress['path']}: wind={progress['wind']:.4f}, "
              f"level={progress.get('level')}, eps={progress.get('eps_at_level')}")
    if args.curriculum_start_wind is not None:
        cur_cc[f"{args.phase}_start_wind"] = float(args.curriculum_start_wind)
    if args.curriculum_start_level is not None:
        cur_cc[f"{args.phase}_start_level"] = int(args.curriculum_start_level)
    if args.curriculum_ramp_episodes is not None:
        cur_cc[f"{args.phase}_ramp_episodes"] = int(
            args.curriculum_ramp_episodes)
    if args.curriculum_ramp_sr_threshold is not None:
        cur_cc[f"{args.phase}_ramp_sr_threshold"] = float(
            args.curriculum_ramp_sr_threshold)
    if args.curriculum_ramp_warmup_sr_threshold is not None:
        cur_cc[f"{args.phase}_ramp_warmup_sr_threshold"] = float(
            args.curriculum_ramp_warmup_sr_threshold)
    if args.curriculum_ramp_warmup_scale is not None:
        cur_cc[f"{args.phase}_ramp_warmup_scale"] = float(
            args.curriculum_ramp_warmup_scale)
    if args.curriculum_ramp_min_window is not None:
        cur_cc[f"{args.phase}_ramp_min_window"] = int(
            args.curriculum_ramp_min_window)
    if args.high_wind_focus:
        cur_cc[f"{args.phase}_wind_focus_enabled"] = True
    for _arg_name, _cfg_suffix, _cast in [
        ("wind_focus_start_level", "wind_focus_start_level", int),
        ("wind_focus_full_level", "wind_focus_full_level", int),
        ("wind_focus_start_min", "wind_focus_start_min", float),
        ("wind_focus_full_min", "wind_focus_full_min", float),
        ("wind_focus_final_min", "wind_focus_final_min", float),
    ]:
        _val = getattr(args, _arg_name, None)
        if _val is not None:
            cur_cc[f"{args.phase}_{_cfg_suffix}"] = _cast(_val)
    if args.phase == "descent":
        drl_cc = cc.setdefault("descent_rl", {})
        if args.descent_residual_dq_scale is not None:
            drl_cc["residual_dq_scale"] = float(args.descent_residual_dq_scale)
        if args.descent_residual_acc_xy is not None:
            v = float(args.descent_residual_acc_xy)
            drl_cc["residual_acc_max_xy"] = v
            drl_cc["acc_max_xy"] = v
        if args.descent_residual_acc_z is not None:
            v = float(args.descent_residual_acc_z)
            drl_cc["residual_acc_max_z"] = v
            drl_cc["acc_max_z"] = v
        if args.descent_action_rms_free is not None:
            drl_cc.setdefault("reward", {})["action_rms_free"] = float(
                args.descent_action_rms_free)
        if args.descent_action_magnitude_coef is not None:
            drl_cc.setdefault("reward", {})["action_magnitude_coef"] = float(
                args.descent_action_magnitude_coef)
        for arg_name, cfg_key in [
            ("descent_z_soft_gate_full", "z_soft_gate_full"),
            ("descent_z_hard_gate", "z_hard_gate"),
            ("descent_z_min_speed_frac", "z_min_speed_frac"),
            ("descent_z_trickle_xy_gate", "z_trickle_xy_gate"),
            ("descent_base_v_max_z", "base_v_max_z"),
        ]:
            val = getattr(args, arg_name, None)
            if val is not None:
                drl_cc[cfg_key] = float(val)
        reward_cc = drl_cc.setdefault("reward", {})
        if args.descent_alignment_z_gate is not None:
            reward_cc["alignment_z_gate"] = float(
                args.descent_alignment_z_gate)
        if args.descent_premature_descent_xy_gate is not None:
            reward_cc["premature_descent_xy_gate"] = float(
                args.descent_premature_descent_xy_gate)
    if args.disable_lucky_until_sr is not None:
        print("[LuckyReject] --disable-lucky-until-sr is deprecated and ignored; "
              "strict lucky rejection stays enabled.")
        ins_cc = cc.setdefault("insertion", {})
        ins_cc["strict_lucky_reject_always"] = True
        ins_cc["train_reject_lucky_rebar_insert"] = True
        ins_cc["lucky_reject_auto_enable"] = False
    if args.lucky_reject_min_window is not None:
        cc.setdefault("insertion", {})["lucky_reject_min_window"] = int(
            args.lucky_reject_min_window)

    train(args.phase, ld, algo=args.algo, custom_config=cc or None,
          resume_ckpt=args.resume_ckpt)
