"""Evaluate traditional descent experts for paper comparison baselines.

Example:
    python compare_traditional_experts.py --experts pid,damped_pd,mpc \
        --task-profile easy-single-rebar --episodes 20 --tune
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import time
from collections import Counter

import numpy as np
from scipy.spatial.transform import Rotation as R

from config import DEFAULT_CONFIG
from mujoco_env_new import CableRobotEnvWithObstacles
from phase_reward import DescentRewardState, compute_descent_reward
from train_phase import reset_for_phase
from traditional_experts import make_traditional_expert


PAPER_TASK_START_XY = [0.30, 0.15]
PAPER_TASK_TARGET_XY = [-0.30, 0.20]
PAPER_PAYLOAD_Z_CRUISE = float(DEFAULT_CONFIG["planning"]["payload_z_cruise"])
PAPER_TARGET_Z_DESCENT = float(DEFAULT_CONFIG["planning"]["target_z_descent"])
PAPER_DESCENT_MAX_STEPS = int(DEFAULT_CONFIG["descent_rl"]["max_steps"])
PAPER_PAYLOAD_MASS = float(DEFAULT_CONFIG["prefab"]["mass"])
DEFAULT_OBS_OBSTACLE_SLOTS = int(DEFAULT_CONFIG["scene"]["n_obstacles"])
REBAR4_POSITIONS = [
    [0.035, 0.035], [0.035, -0.035],
    [-0.035, 0.035], [-0.035, -0.035],
]


TASK_PROFILES = {
    "easy-single-rebar": {
        "description": "No wind, one centered rebar, wide socket, simplified damped cable.",
        "sim": {"max_steps": 420},
        "scene": {"n_obstacles": 0, "seed": 21},
        "noise": {"latency_steps": 0},
        "task": {
            "default_start_xy": [0.30, 0.15],
            "default_target_xy": [0.25, 0.15],
            "init_position_range": 0.0,
            "target_xy_randomize": False,
        },
        "planning": {
            "payload_z_cruise": 0.32,
            "target_z_descent": 0.10,
            "num_descent_steps": 8,
        },
        "rope": {
            "num_segments": 4,
            "segment_length": 0.10,
            "damping": 0.20,
            "segment_mass": 0.015,
        },
        "controller": {
            "L": 0.45,
            "N": 10,
            "u_max_xy": 0.45,
            "u_max_z": 0.80,
        },
        "prefab": {
            "socket_half_size": [0.05, 0.05, 0.10],
            "socket_hole_size": [0.060, 0.060],
            "socket_hole_depth": 0.080,
            "socket_hole_positions": [[0.0, 0.0]],
            "mass": 3.0,
        },
        "target": {
            "rebar_positions": [[0.0, 0.0]],
            "rebar_radius": 0.0025,
            "rebar_half_height": 0.010,
        },
        "insertion": {
            "target_payload_z": 0.10,
            "success_z_tolerance": 0.025,
            "xy_tolerance": 0.030,
            "xy_tolerance_train_end": 0.030,
            "physical_rebar_xy_tolerance": 0.030,
            "tilt_tolerance": 0.90,
            "yaw_tolerance": 3.20,
            "tilt_tolerance_train_start": 0.90,
            "tilt_tolerance_train_end": 0.90,
            "yaw_tolerance_train_start": 3.20,
            "yaw_tolerance_train_end": 3.20,
            "hold_steps": 2,
            "physical_success_requires_floor_contact": False,
            "physical_success_requires_insert_depth": False,
            "physical_success_requires_rebar_alignment": True,
            "train_reject_lucky_rebar_insert": False,
            "strict_lucky_reject_always": False,
            "stuck_fail_enabled": False,
        },
        "descent_rl": {
            "init_xy_range": 0.006,
            "init_vel_range": 0.0,
            "init_tilt_range": 0.0,
            "max_steps": 420,
            "early_stop_xy_fail": 0.280,
            "early_stop_patience": 120,
        },
        "step_logic": {
            "instability_grace_steps": 30,
            "swing_xy_max": 0.70,
            "payload_tilt_max": 1.8,
            "payload_yaw_max": 3.2,
        },
    },
    "easy-four-rebar": {
        "description": "No wind, four rebars, relaxed holes, simplified damped cable.",
        "sim": {"max_steps": 420},
        "scene": {"n_obstacles": 0, "seed": 21},
        "noise": {"latency_steps": 0},
        "task": {
            "default_start_xy": [0.30, 0.15],
            "default_target_xy": [0.25, 0.15],
            "init_position_range": 0.0,
            "target_xy_randomize": False,
        },
        "rope": {
            "num_segments": 6,
            "segment_length": 0.067,
            "damping": 0.14,
            "segment_mass": 0.012,
        },
        "planning": {
            "payload_z_cruise": 0.30,
            "target_z_descent": 0.10,
        },
        "controller": {"L": 0.45},
        "prefab": {
            "socket_hole_size": [0.024, 0.024],
            "mass": 3.6,
        },
        "target": {
            "rebar_radius": 0.0025,
        },
        "insertion": {
            "success_z_tolerance": 0.024,
            "xy_tolerance": 0.010,
            "xy_tolerance_train_end": 0.010,
            "physical_rebar_xy_tolerance": 0.010,
            "tilt_tolerance": 0.18,
            "yaw_tolerance": 0.35,
            "tilt_tolerance_train_start": 0.18,
            "tilt_tolerance_train_end": 0.18,
            "yaw_tolerance_train_start": 0.35,
            "yaw_tolerance_train_end": 0.35,
            "hold_steps": 2,
            "physical_success_requires_floor_contact": False,
            "physical_success_requires_insert_depth": False,
            "train_reject_lucky_rebar_insert": False,
            "strict_lucky_reject_always": False,
            "stuck_fail_enabled": False,
        },
        "descent_rl": {
            "init_xy_range": 0.014,
            "init_vel_range": 0.0,
            "init_tilt_range": 0.002,
            "max_steps": 420,
            "early_stop_xy_fail": 0.100,
            "early_stop_patience": 80,
        },
        "step_logic": {
            "instability_grace_steps": 30,
            "swing_xy_max": 0.55,
            "payload_tilt_max": 1.4,
            "payload_yaw_max": 3.2,
        },
    },
    "nominal-four-rebar": {
        "description": "Default four-rebar geometry without wind or obstacles.",
        "scene": {"n_obstacles": 0, "seed": 21},
        "task": {"init_position_range": 0.0},
        "wind": {"enabled": False},
        "descent_rl": {
            "init_xy_range": 0.020,
            "init_vel_range": 0.0,
            "init_tilt_range": 0.005,
            "max_steps": 300,
        },
    },
}


TUNE_PRESETS = {
    "pid": [
        {"kp_xy": 1.50, "ki_xy": 0.35, "kd_payload_xy": 0.30,
         "k_swing_xy": 0.15, "k_rel_vel_xy": 0.05, "vel_max_xy": 0.12,
         "vel_max_z": 0.016, "z_gate": 0.030,
         "swing_fade_xy": 0.070, "swing_fade_z_margin": 0.045},
        {"kp_xy": 2.00, "ki_xy": 0.55, "kd_payload_xy": 0.35,
         "k_swing_xy": 0.25, "k_rel_vel_xy": 0.10, "vel_max_xy": 0.14,
         "vel_max_z": 0.020, "z_gate": 0.030,
         "swing_fade_xy": 0.080, "swing_fade_z_margin": 0.050},
        {"kp_xy": 2.50, "ki_xy": 0.70, "kd_payload_xy": 0.45,
         "k_swing_xy": 0.35, "k_rel_vel_xy": 0.12, "vel_max_xy": 0.16,
         "vel_max_z": 0.018, "z_gate": 0.024,
         "swing_fade_xy": 0.090, "swing_fade_z_margin": 0.050},
        {"kp_xy": 2.80, "ki_xy": 0.90, "kd_payload_xy": 0.55,
         "k_swing_xy": 0.42, "k_rel_vel_xy": 0.18, "vel_max_xy": 0.18,
         "vel_max_z": 0.014, "z_gate": 0.018, "z_trickle_gate": 0.050,
         "integral_limit": 0.012, "swing_fade_xy": 0.070,
         "swing_fade_z_margin": 0.035, "yaw_kp": 1.20,
         "vel_max_yaw": 0.80},
        {"kp_xy": 1.80, "ki_xy": 0.75, "kd_payload_xy": 0.70,
         "k_swing_xy": 0.55, "k_rel_vel_xy": 0.25, "vel_max_xy": 0.10,
         "vel_max_z": 0.012, "z_gate": 0.014, "z_trickle_gate": 0.040,
         "integral_limit": 0.010, "swing_fade_xy": 0.055,
         "swing_fade_z_margin": 0.030, "yaw_kp": 1.60,
         "vel_max_yaw": 1.20},
    ],
    "damped_pd": [
        {"kp_xy": 1.75, "ki_xy": 0.0, "kd_payload_xy": 0.50, "k_swing_xy": 0.22,
         "swing_fade_xy": 0.070, "swing_fade_z_margin": 0.050,
         "k_rel_vel_xy": 0.08,
         "vel_max_xy": 0.13, "vel_max_z": 0.018},
        {"kp_xy": 2.05, "ki_xy": 0.0, "kd_payload_xy": 0.55, "k_swing_xy": 0.28,
         "swing_fade_xy": 0.080, "swing_fade_z_margin": 0.050,
         "k_rel_vel_xy": 0.12,
         "vel_max_xy": 0.14, "vel_max_z": 0.020},
        {"kp_xy": 2.35, "ki_xy": 0.0, "kd_payload_xy": 0.65, "k_swing_xy": 0.32,
         "swing_fade_xy": 0.090, "swing_fade_z_margin": 0.050,
         "k_rel_vel_xy": 0.16,
         "vel_max_xy": 0.15, "vel_max_z": 0.018},
        {"kp_xy": 2.80, "ki_xy": 0.0, "kd_payload_xy": 0.85, "k_swing_xy": 0.42,
         "swing_fade_xy": 0.070, "swing_fade_z_margin": 0.035,
         "k_rel_vel_xy": 0.24, "vel_max_xy": 0.16, "vel_max_z": 0.014,
         "z_gate": 0.018, "yaw_kp": 1.20, "vel_max_yaw": 0.80},
        {"kp_xy": 1.60, "ki_xy": 0.0, "kd_payload_xy": 0.90, "k_swing_xy": 0.58,
         "swing_fade_xy": 0.055, "swing_fade_z_margin": 0.030,
         "k_rel_vel_xy": 0.30, "vel_max_xy": 0.10, "vel_max_z": 0.012,
         "z_gate": 0.014, "yaw_kp": 1.60, "vel_max_yaw": 1.20},
    ],
    "mpc": [
        {"N": 10, "model_damping_xy": 1.00, "kp_feedback_xy": 1.55,
         "kd_feedback_xy": 0.50, "k_swing_feedback_xy": 0.25,
         "q_stage_xy": 150.0, "q_terminal_xy": 650.0, "q_payload_vel_xy": 22.0,
         "q_swing_xy": 32.0, "r_cmd_vel_xy": 7.0, "vel_max_xy": 0.14,
         "vel_max_z": 0.018, "cmd_rate_limit_xy": 0.040},
        {"N": 14, "model_damping_xy": 1.15, "kp_feedback_xy": 1.85,
         "kd_feedback_xy": 0.60, "k_swing_feedback_xy": 0.35,
         "q_stage_xy": 180.0, "q_terminal_xy": 850.0, "q_payload_vel_xy": 24.0,
         "q_swing_xy": 38.0, "r_cmd_vel_xy": 6.0, "vel_max_xy": 0.16,
         "vel_max_z": 0.020, "cmd_rate_limit_xy": 0.050},
        {"N": 18, "model_damping_xy": 1.30, "kp_feedback_xy": 2.10,
         "kd_feedback_xy": 0.70, "k_swing_feedback_xy": 0.45,
         "q_stage_xy": 210.0, "q_terminal_xy": 1050.0, "q_payload_vel_xy": 28.0,
         "q_swing_xy": 45.0, "r_cmd_vel_xy": 5.5, "vel_max_xy": 0.16,
         "vel_max_z": 0.018, "cmd_rate_limit_xy": 0.045},
        {"N": 22, "model_damping_xy": 1.55, "kp_feedback_xy": 2.40,
         "kd_feedback_xy": 0.90, "k_swing_feedback_xy": 0.60,
         "q_stage_xy": 260.0, "q_terminal_xy": 1350.0, "q_payload_vel_xy": 36.0,
         "q_swing_xy": 58.0, "r_cmd_vel_xy": 7.0, "vel_max_xy": 0.13,
         "vel_max_z": 0.014, "cmd_rate_limit_xy": 0.035,
         "descent_ready_xy": 0.018, "descent_ready_payload_vel": 0.060,
         "feedback_override_xy": 0.035, "yaw_kp": 1.20,
         "vel_max_yaw": 0.80},
        {"N": 16, "model_damping_xy": 0.95, "kp_feedback_xy": 2.80,
         "kd_feedback_xy": 0.55, "k_swing_feedback_xy": 0.25,
         "q_stage_xy": 280.0, "q_terminal_xy": 1600.0, "q_payload_vel_xy": 18.0,
         "q_swing_xy": 26.0, "r_cmd_vel_xy": 4.5, "vel_max_xy": 0.18,
         "vel_max_z": 0.012, "cmd_rate_limit_xy": 0.055,
         "descent_ready_xy": 0.014, "descent_ready_payload_vel": 0.080,
         "feedback_override_xy": 0.030, "yaw_kp": 1.60,
         "vel_max_yaw": 1.20},
    ],
}


WIND_BASELINE_PRESET_IDX = {
    # Best 24 mm no-wind presets from the boundary sweep; keep fixed when
    # measuring wind robustness.
    "pid": 4,
    "damped_pd": 0,
    "mpc": 1,
}


WIND_BASELINE_PRESET_IDX_BY_TASK = {
    # Re-tuned on the 24 mm task with the default 10-segment rope.
    "easy-four-rebar-10seg": {
        "pid": 2,
        "damped_pd": 4,
        "mpc": 1,
    },
}


FIXED_BEST_PRESET_IDX = {
    # Fixed for the paper ladder: no per-level retuning during evaluation.
    # Chosen from previous boundary / 24mm-10seg sweeps.
    "pid": 2,
    "damped_pd": 4,
    "mpc": 1,
}


def deep_update(base, override):
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def parse_experts(text):
    return [x.strip().lower() for x in str(text).split(",") if x.strip()]


def parse_float_list(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _install_ladder_profiles():
    rebar4 = REBAR4_POSITIONS

    TASK_PROFILES["single-rebar-46mm"] = deep_update(
        TASK_PROFILES["easy-single-rebar"], {
            "description": (
                "No wind, one centered rebar, 46mm socket hole, "
                "same simplified cable."),
            "prefab": {"socket_hole_size": [0.046, 0.046]},
            "insertion": {
                "xy_tolerance": 0.020,
                "xy_tolerance_train_end": 0.020,
                "physical_rebar_xy_tolerance": 0.020,
            },
            "descent_rl": {"init_xy_range": 0.010},
        })

    TASK_PROFILES["single-rebar-24mm"] = deep_update(
        TASK_PROFILES["easy-single-rebar"], {
            "description": (
                "No wind, one centered rebar, 24mm socket hole, "
                "6-segment cable."),
            "sim": {"max_steps": 420},
            "rope": {
                "num_segments": 6,
                "segment_length": 0.067,
                "damping": 0.14,
                "segment_mass": 0.012,
            },
            "prefab": {
                "socket_hole_size": [0.024, 0.024],
                "mass": 3.6,
            },
            "insertion": {
                "success_z_tolerance": 0.024,
                "xy_tolerance": 0.010,
                "xy_tolerance_train_end": 0.010,
                "physical_rebar_xy_tolerance": 0.010,
                "tilt_tolerance": 0.45,
                "tilt_tolerance_train_start": 0.45,
                "tilt_tolerance_train_end": 0.45,
            },
            "descent_rl": {
                "init_xy_range": 0.012,
                "max_steps": 420,
                "early_stop_xy_fail": 0.140,
                "early_stop_patience": 100,
            },
            "step_logic": {"payload_tilt_max": 1.4, "swing_xy_max": 0.55},
        })

    TASK_PROFILES["four-rebar-30mm"] = deep_update(
        TASK_PROFILES["easy-single-rebar"], {
            "description": (
                "No wind, four rebars, 30mm holes, relaxed yaw/tilt, "
                "6-segment cable."),
            "sim": {"max_steps": 420},
            "rope": {
                "num_segments": 6,
                "segment_length": 0.067,
                "damping": 0.14,
                "segment_mass": 0.012,
            },
            "prefab": {
                "socket_hole_size": [0.030, 0.030],
                "socket_hole_positions": rebar4,
                "mass": 3.6,
            },
            "target": {"rebar_positions": rebar4},
            "insertion": {
                "success_z_tolerance": 0.024,
                "xy_tolerance": 0.012,
                "xy_tolerance_train_end": 0.012,
                "physical_rebar_xy_tolerance": 0.012,
                "tilt_tolerance": 0.30,
                "yaw_tolerance": 0.70,
                "tilt_tolerance_train_start": 0.30,
                "tilt_tolerance_train_end": 0.30,
                "yaw_tolerance_train_start": 0.70,
                "yaw_tolerance_train_end": 0.70,
            },
            "descent_rl": {
                "init_xy_range": 0.012,
                "init_tilt_range": 0.002,
                "max_steps": 420,
                "early_stop_xy_fail": 0.120,
                "early_stop_patience": 100,
            },
            "step_logic": {"payload_tilt_max": 1.4, "swing_xy_max": 0.55},
        })

    TASK_PROFILES["four-rebar-20mm"] = deep_update(
        TASK_PROFILES["easy-four-rebar"], {
            "description": (
                "No wind, four rebars, 20mm holes, tighter yaw/tilt, "
                "8-segment cable."),
            "sim": {"max_steps": 420},
            "rope": {
                "num_segments": 8,
                "segment_length": 0.050,
                "damping": 0.10,
                "segment_mass": 0.011,
            },
            "prefab": {
                "socket_hole_size": [0.020, 0.020],
                "mass": 4.0,
            },
            "insertion": {
                "success_z_tolerance": 0.022,
                "xy_tolerance": 0.008,
                "xy_tolerance_train_end": 0.008,
                "physical_rebar_xy_tolerance": 0.008,
                "tilt_tolerance": 0.14,
                "yaw_tolerance": 0.25,
                "tilt_tolerance_train_start": 0.14,
                "tilt_tolerance_train_end": 0.14,
                "yaw_tolerance_train_start": 0.25,
                "yaw_tolerance_train_end": 0.25,
            },
            "descent_rl": {
                "init_xy_range": 0.016,
                "init_tilt_range": 0.004,
                "max_steps": 420,
                "early_stop_xy_fail": 0.080,
                "early_stop_patience": 80,
            },
            "step_logic": {
                "payload_tilt_max": 1.2,
                "payload_yaw_max": 2.0,
                "swing_xy_max": 0.45,
            },
        })

    TASK_PROFILES["easy-four-rebar-10seg"] = deep_update(
        TASK_PROFILES["easy-four-rebar"], {
            "description": (
                "No wind, four rebars, 24mm holes, default 10-segment cable."),
            "rope": {
                "num_segments": 10,
                "segment_length": 0.040,
                "damping": 0.05,
                "segment_mass": 0.010,
            },
        })


def _paper_level_profile(description, hole_m, rope_segments, init_xy_m,
                         init_tilt_rad, xy_tol_m, yaw_tol_rad, tilt_tol_rad,
                         wind_mps, rope_damping, rope_mass, z_tol_m=0.024,
                         early_stop_xy_fail=0.12):
    segment_length = 0.40 / max(1, int(rope_segments))
    return {
        "description": description,
        "sim": {"max_steps": PAPER_DESCENT_MAX_STEPS},
        "test": {"actual_obstacles": 0},
        "task": {
            "default_start_xy": PAPER_TASK_START_XY,
            "default_target_xy": PAPER_TASK_TARGET_XY,
            "init_position_range": 0.0,
            "target_xy_randomize": False,
        },
        "planning": {
            "payload_z_cruise": PAPER_PAYLOAD_Z_CRUISE,
            "target_z_descent": PAPER_TARGET_Z_DESCENT,
        },
        "rope": {
            "num_segments": int(rope_segments),
            "segment_length": segment_length,
            "damping": float(rope_damping),
            "segment_mass": float(rope_mass),
        },
        "prefab": {
            "socket_hole_size": [float(hole_m), float(hole_m)],
            "socket_hole_positions": REBAR4_POSITIONS,
            "mass": PAPER_PAYLOAD_MASS,
        },
        "target": {
            "rebar_positions": REBAR4_POSITIONS,
            "rebar_radius": 0.0025,
        },
        "paper_eval": {
            "nominal_hole_size": float(hole_m),
            "xy_tolerance": float(xy_tol_m),
            "z_tolerance": float(z_tol_m),
            "tilt_tolerance": float(tilt_tol_rad),
            "yaw_tolerance": float(yaw_tol_rad),
            "wind_speed": float(wind_mps),
        },
        "insertion": {
            "target_payload_z": 0.10,
            "success_z_tolerance": float(z_tol_m),
            "xy_tolerance": float(xy_tol_m),
            "xy_tolerance_train_end": float(xy_tol_m),
            "physical_rebar_xy_tolerance": float(xy_tol_m),
            "tilt_tolerance": float(tilt_tol_rad),
            "yaw_tolerance": float(yaw_tol_rad),
            "tilt_tolerance_train_start": float(tilt_tol_rad),
            "tilt_tolerance_train_end": float(tilt_tol_rad),
            "yaw_tolerance_train_start": float(yaw_tol_rad),
            "yaw_tolerance_train_end": float(yaw_tol_rad),
            "hold_steps": 2,
            "physical_success_requires_floor_contact": False,
            "physical_success_requires_insert_depth": False,
            "physical_success_requires_rebar_alignment": True,
            "train_reject_lucky_rebar_insert": False,
            "strict_lucky_reject_always": False,
            "stuck_fail_enabled": False,
        },
        "descent_rl": {
            "init_xy_range": float(init_xy_m),
            "init_vel_range": 0.0,
            "init_tilt_range": float(init_tilt_rad),
            "max_steps": PAPER_DESCENT_MAX_STEPS,
            "early_stop_xy_fail": float(early_stop_xy_fail),
            "early_stop_patience": 100,
        },
        "curriculum": {
            "descent_levels": [{
                "init_xy": float(init_xy_m),
                "init_vel": 0.0,
                "init_tilt": float(init_tilt_rad),
                "xy_tol": float(xy_tol_m),
                "wind_max": float(wind_mps),
            }],
            "descent_start_level": 0,
            "descent_start_wind": float(wind_mps),
        },
        "vision": {
            "enabled": False,
            "apply_to_env_obs": False,
        },
        "observation_predictor": {"enabled": False},
        "cable_latent_predictor": {"enabled": False},
        "delay_mdp": {"enabled": False},
        "step_logic": {
            "instability_grace_steps": 30,
            "swing_xy_max": 0.55,
            "payload_tilt_max": 1.4,
            "payload_yaw_max": 3.2,
        },
    }


def _install_paper_ladder_profiles():
    TASK_PROFILES["paper-l1"] = _paper_level_profile(
        "L1: four rebars, nominal 46mm gate, 4 rope segments, true obs.",
        hole_m=0.046, rope_segments=4, init_xy_m=0.010,
        init_tilt_rad=0.0, xy_tol_m=0.020, yaw_tol_rad=3.20,
        tilt_tol_rad=0.90, wind_mps=0.0, rope_damping=0.20,
        rope_mass=0.015, z_tol_m=0.025, early_stop_xy_fail=0.20)
    TASK_PROFILES["paper-l2"] = _paper_level_profile(
        "L2: four rebars, nominal 30mm gate, 6 rope segments, true obs.",
        hole_m=0.030, rope_segments=6, init_xy_m=0.012,
        init_tilt_rad=0.002, xy_tol_m=0.012, yaw_tol_rad=0.70,
        tilt_tol_rad=0.30, wind_mps=0.0, rope_damping=0.14,
        rope_mass=0.012, z_tol_m=0.024, early_stop_xy_fail=0.12)
    TASK_PROFILES["paper-l3"] = _paper_level_profile(
        "L3: four rebars, nominal 24mm gate, 6 rope segments, 2m/s wind.",
        hole_m=0.024, rope_segments=6, init_xy_m=0.014,
        init_tilt_rad=0.002, xy_tol_m=0.010, yaw_tol_rad=0.35,
        tilt_tol_rad=0.18, wind_mps=2.0, rope_damping=0.14,
        rope_mass=0.012, z_tol_m=0.024, early_stop_xy_fail=0.10)
    TASK_PROFILES["paper-l4"] = _paper_level_profile(
        "L4: four rebars, nominal 24mm gate, 10 rope segments, 4m/s wind.",
        hole_m=0.024, rope_segments=10, init_xy_m=0.016,
        init_tilt_rad=0.003, xy_tol_m=0.010, yaw_tol_rad=0.35,
        tilt_tol_rad=0.18, wind_mps=4.0, rope_damping=0.05,
        rope_mass=0.010, z_tol_m=0.024, early_stop_xy_fail=0.10)
    TASK_PROFILES["paper-l5"] = _paper_level_profile(
        "L5: four rebars, nominal 20mm gate, 10 rope segments, 6m/s wind.",
        hole_m=0.020, rope_segments=10, init_xy_m=0.016,
        init_tilt_rad=0.004, xy_tol_m=0.007, yaw_tol_rad=0.20,
        tilt_tol_rad=0.12, wind_mps=6.0, rope_damping=0.05,
        rope_mass=0.010, z_tol_m=0.022, early_stop_xy_fail=0.08)


_install_ladder_profiles()
_install_paper_ladder_profiles()

TASK_LADDERS = {
    "default-descent-no-wind": [
        "easy-single-rebar",
        "single-rebar-46mm",
        "single-rebar-24mm",
        "four-rebar-30mm",
        "easy-four-rebar",
        "four-rebar-20mm",
        "nominal-four-rebar",
    ],
    "paper-trueobs-fixedbest": [
        "paper-l1",
        "paper-l2",
        "paper-l3",
        "paper-l4",
        "paper-l5",
    ],
}


def build_config(args, extra_traditional=None):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    profile = TASK_PROFILES[args.task_profile]
    cfg = deep_update(cfg, profile)
    if extra_traditional:
        cfg = deep_update(cfg, {"traditional_experts": extra_traditional})
    cfg["sim"]["render"] = bool(args.render)
    cfg["wind"]["enabled"] = False
    cfg["wind"]["test_wind_variable"] = bool(
        getattr(args, "wind_variable", False))
    cfg.setdefault("vision", {})["enabled"] = False
    cfg.setdefault("vision", {})["apply_to_env_obs"] = False
    cfg.setdefault("observation_predictor", {})["enabled"] = False
    cfg.setdefault("cable_latent_predictor", {})["enabled"] = False
    cfg.setdefault("delay_mdp", {})["enabled"] = False
    cfg.setdefault("test", {})
    arg_obstacles = getattr(args, "obstacles", None)
    if arg_obstacles is None:
        actual_obstacles = int(cfg["test"].get("actual_obstacles", 0))
    else:
        actual_obstacles = int(arg_obstacles)
    cfg["test"]["actual_obstacles"] = actual_obstacles
    cfg["scene"]["n_obstacles"] = max(
        DEFAULT_OBS_OBSTACLE_SLOTS, actual_obstacles)
    if args.max_steps is not None:
        cfg["sim"]["max_steps"] = int(args.max_steps)
        cfg["descent_rl"]["max_steps"] = int(args.max_steps)
    cfg["scene"]["seed"] = int(args.seed)
    return cfg


def paper_profile_wind(config):
    peval = config.get("paper_eval", {})
    if "wind_speed" in peval:
        return float(peval["wind_speed"])
    levels = config.get("curriculum", {}).get("descent_levels", [])
    if levels:
        return float(levels[0].get("wind_max", 0.0))
    return 0.0


def fixed_best_preset(expert_name):
    presets = TUNE_PRESETS.get(expert_name, [{}])
    idx = FIXED_BEST_PRESET_IDX.get(expert_name)
    if idx is None or idx >= len(presets):
        return {}, ""
    return copy.deepcopy(presets[idx]), str(idx)


def select_preset(expert_name, args):
    if bool(getattr(args, "tune", False)) or getattr(args, "preset_mode", "") == "tune":
        return tune_expert(expert_name, args), "tune"
    if bool(getattr(args, "ladder_no_tune", False)) or getattr(args, "preset_mode", "") == "none":
        return {}, "none"
    preset, idx = fixed_best_preset(expert_name)
    label = f"fixed-best:{idx}" if idx else "fixed-best:none"
    return preset, label


def final_metrics(env, config):
    pl_pos = env.data.body("prefab").xpos.copy()
    ee_pos = env._get_ee_pos()
    pl_mat = env.data.body("prefab").xmat.reshape(3, 3)
    pl_euler = R.from_matrix(pl_mat).as_euler("xyz")
    tilt = float(np.sqrt(pl_euler[0] ** 2 + pl_euler[1] ** 2))
    yaw = abs(float(pl_euler[2]))
    target_xy = env.target_pos.copy()
    dtf = float(np.linalg.norm(pl_pos[:2] - target_xy))
    target_pz = float(config.get("insertion", {}).get("target_payload_z", 0.10))
    try:
        _, worst_rebar, mean_rebar = env._compute_rebar_errors(pl_pos[:2], pl_mat)
    except Exception:
        worst_rebar = dtf
        mean_rebar = dtf
    try:
        cable_ke = float(env._get_cable_energy())
    except Exception:
        cable_ke = 0.0
    return {
        "target_x_m": float(target_xy[0]),
        "target_y_m": float(target_xy[1]),
        "payload_x_m": float(pl_pos[0]),
        "payload_y_m": float(pl_pos[1]),
        "ee_x_m": float(ee_pos[0]),
        "ee_y_m": float(ee_pos[1]),
        "ee_z_m": float(ee_pos[2]),
        "final_dtf_m": dtf,
        "final_z_m": float(pl_pos[2]),
        "final_z_err_m": abs(float(pl_pos[2]) - target_pz),
        "final_tilt_rad": tilt,
        "final_yaw_abs_rad": yaw,
        "worst_rebar_err_m": float(worst_rebar),
        "mean_rebar_err_m": float(mean_rebar),
        "cable_ke": cable_ke,
    }


def run_one_episode(env, expert, config, seed, verbose=False, trace_every=0,
                    wind_speed_mps=0.0, wind_direction_rad=None):
    obs, _planned_path = reset_for_phase(
        env, "descent", config, rng_seed=int(seed))
    if obs is None:
        return {
            "success": False,
            "steps": 0,
            "reward": 0.0,
            "termination": "reset_failed",
            **{},
        }
    if hasattr(env, "clear_wind"):
        env.clear_wind()
    wind_speed_mps = float(wind_speed_mps or 0.0)
    wind_direction = 0.0
    if wind_speed_mps > 1e-9 and hasattr(env, "set_wind_speed"):
        if wind_direction_rad is None:
            rng = np.random.default_rng(int(seed) + 92821)
            wind_direction = float(rng.uniform(0.0, 2.0 * np.pi))
        else:
            wind_direction = float(wind_direction_rad)
        env.set_wind_speed(wind_speed_mps, direction_rad=wind_direction)

    current_q = env.data.qpos[:7].copy().astype(np.float32)
    expert.reset(obs, current_q, env=env)

    rstate = DescentRewardState()
    rstate.total_steps_global = 0
    levels = config.get("curriculum", {}).get("descent_levels", [])
    if levels:
        rstate.current_xy_tol = float(levels[0].get(
            "xy_tol",
            config.get("insertion", {}).get("xy_tolerance_train_end", 0.005)))
        rstate.descent_n_levels = max(1, len(levels))
    else:
        rstate.current_xy_tol = float(config.get("insertion", {}).get(
            "xy_tolerance_train_end",
            config.get("insertion", {}).get("xy_tolerance", 0.005)))
        rstate.descent_n_levels = 1
    rstate.current_descent_level = 0

    ep_reward = 0.0
    success = False
    term = "timeout"
    steps = 0
    max_steps = int(config.get("descent_rl", {}).get(
        "max_steps", config.get("sim", {}).get("max_steps", 300)))

    for step in range(max_steps):
        current_q = env.data.qpos[:7].copy().astype(np.float32)
        dq = expert.compute_delta_q_target(obs, current_q, env=env)
        next_obs, _, env_done, _env_trunc, env_info = env.step(dq)
        reward, done, r_success, r_info = compute_descent_reward(
            env, next_obs, config, rstate, rl_action=None)
        ep_reward += float(reward)
        steps = step + 1
        obs = next_obs
        if trace_every and (step % int(trace_every) == 0 or r_success):
            pl = env.data.body("prefab").xpos.copy()
            ee = env._get_ee_pos()
            target = env.target_pos.copy()
            dtf = float(np.linalg.norm(pl[:2] - target))
            info = getattr(expert, "last_info", {})
            print(
                f"      t={step + 1:03d} "
                f"pl=({pl[0]:+.3f},{pl[1]:+.3f},{pl[2]:+.3f}) "
                f"ee=({ee[0]:+.3f},{ee[1]:+.3f},{ee[2]:+.3f}) "
                f"dtf={dtf*1000:5.1f}mm info={info}")
        if r_success:
            success = True
            term = str(r_info.get("termination", "success"))
            break
        if r_info.get("termination"):
            term = str(r_info["termination"])
        if env_info.get("nan_detected", False):
            term = "nan_detected"
            break
        if done or env_done:
            term = term or str(env_info.get("termination_reason", "done"))
            break

    row = {
        "success": bool(success),
        "steps": int(steps),
        "reward": float(ep_reward),
        "termination": term,
        "wind_speed_mps": float(wind_speed_mps),
        "wind_direction_rad": float(wind_direction),
        "wind_force_n": float(getattr(env, "wind_F", 0.0)),
    }
    row.update(final_metrics(env, config))
    if verbose:
        mark = "OK" if success else "FAIL"
        print(
            f"    {mark} steps={steps:3d} "
            f"dtf={row['final_dtf_m']*1000:5.1f}mm "
            f"zerr={row['final_z_err_m']*1000:5.1f}mm "
            f"rebar={row['worst_rebar_err_m']*1000:5.1f}mm "
            f"term={term}")
    return row


def evaluate_expert(expert_name, config, episodes, seed,
                    verbose=False, trace_every=0,
                    wind_speed_mps=0.0, wind_direction_rad=None):
    env_config = copy.deepcopy(config)
    env = CableRobotEnvWithObstacles(config=env_config)
    actual_obstacles = int(config.get("test", {}).get("actual_obstacles", 0))
    if hasattr(env, "set_curriculum_n_obstacles"):
        env.set_curriculum_n_obstacles(actual_obstacles)
    expert = make_traditional_expert(expert_name, config, env.ik_solver)
    rows = []
    try:
        for ep in range(int(episodes)):
            row = run_one_episode(
                env, expert, config, seed=int(seed) + ep, verbose=verbose,
                trace_every=trace_every, wind_speed_mps=wind_speed_mps,
                wind_direction_rad=wind_direction_rad)
            row["expert"] = expert_name
            row["episode"] = ep
            rows.append(row)
    finally:
        if getattr(env, "viewer", None) is not None:
            try:
                env.viewer.close()
            except Exception:
                pass
    return rows


def summarize(rows):
    if not rows:
        return {"sr": 0.0, "n": 0}
    success = np.array([float(r["success"]) for r in rows], dtype=np.float64)
    steps = np.array([float(r["steps"]) for r in rows], dtype=np.float64)
    dtf = np.array([float(r["final_dtf_m"]) for r in rows], dtype=np.float64)
    zerr = np.array([float(r["final_z_err_m"]) for r in rows], dtype=np.float64)
    rebar = np.array([float(r["worst_rebar_err_m"]) for r in rows], dtype=np.float64)
    terms = Counter(str(r["termination"]).split(":")[0] for r in rows)
    return {
        "n": len(rows),
        "sr": float(np.mean(success)),
        "avg_steps": float(np.mean(steps)),
        "dtf_mm": float(np.mean(dtf) * 1000.0),
        "zerr_mm": float(np.mean(zerr) * 1000.0),
        "rebar_mm": float(np.mean(rebar) * 1000.0),
        "top_terms": terms.most_common(3),
    }


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def tune_expert(expert_name, args):
    presets = TUNE_PRESETS.get(expert_name, [{}])
    best = None
    print(f"\n[TUNE] {expert_name}: {len(presets)} presets, "
          f"{args.tune_episodes} episodes each")
    for i, preset in enumerate(presets):
        cfg = build_config(args, extra_traditional={expert_name: preset})
        wind_speed_mps = paper_profile_wind(cfg)
        rows = evaluate_expert(
            expert_name, cfg, args.tune_episodes, args.seed, verbose=False,
            trace_every=0, wind_speed_mps=wind_speed_mps,
            wind_direction_rad=getattr(args, "wind_direction", None))
        summ = summarize(rows)
        print(f"  preset {i}: SR={summ['sr']*100:5.1f}% "
              f"dtf={summ['dtf_mm']:5.1f}mm "
              f"rebar={summ['rebar_mm']:5.1f}mm "
              f"steps={summ['avg_steps']:5.1f} cfg={preset}")
        score = (summ["sr"], -summ["dtf_mm"], -summ["rebar_mm"], -summ["avg_steps"])
        if best is None or score > best["score"]:
            best = {"preset": preset, "summary": summ, "score": score}
    print(f"  best: SR={best['summary']['sr']*100:.1f}% cfg={best['preset']}")
    return best["preset"]


def ladder_profiles(args):
    if args.ladder_profiles:
        profiles = parse_experts(args.ladder_profiles)
    else:
        profiles = list(TASK_LADDERS[str(args.ladder_name)])
    unknown = [p for p in profiles if p not in TASK_PROFILES]
    if unknown:
        valid = ", ".join(sorted(TASK_PROFILES))
        raise ValueError(f"Unknown ladder profile(s): {unknown}. Valid: {valid}")
    return profiles


def run_ladder(args):
    experts = parse_experts(args.experts)
    profiles = ladder_profiles(args)
    stop_sr = float(args.stop_sr)
    active = {name: True for name in experts}
    all_rows = []
    summary_rows = []

    print(f"[LADDER] {args.ladder_name}: {' -> '.join(profiles)}")
    print(f"[LADDER] stop threshold: SR < {stop_sr * 100:.1f}% "
          f"after evaluation")

    for level_idx, profile_name in enumerate(profiles):
        profile = TASK_PROFILES[profile_name]
        print(f"\n[LEVEL {level_idx}] {profile_name}: {profile['description']}")
        for expert_name in experts:
            if not active.get(expert_name, False):
                print(f"  [SKIP] {expert_name}: already below threshold")
                continue

            local_args = copy.copy(args)
            local_args.task_profile = profile_name
            local_args.seed = int(args.seed) + level_idx * int(args.seed_stride)
            preset, preset_label = select_preset(expert_name, local_args)
            cfg = build_config(local_args, extra_traditional={expert_name: preset})
            wind_speed_mps = paper_profile_wind(cfg)

            print(f"\n[EVAL-L{level_idx}] {expert_name}: "
                  f"profile={profile_name}, episodes={args.episodes}, "
                  f"wind={wind_speed_mps:.1f}m/s, preset={preset_label}")
            rows = evaluate_expert(
                expert_name, cfg, args.episodes, local_args.seed,
                verbose=args.verbose, trace_every=args.trace_every,
                wind_speed_mps=wind_speed_mps,
                wind_direction_rad=args.wind_direction)
            for row in rows:
                row["task_profile"] = profile_name
                row["difficulty_level"] = level_idx
                row["preset_label"] = preset_label
                row["preset"] = repr(preset)
                row["true_env_obs"] = True
                row["actual_obstacles"] = int(cfg.get("test", {}).get(
                    "actual_obstacles", 0))
                row["obs_obstacle_slots"] = int(cfg.get("scene", {}).get(
                    "n_obstacles", 0))
            all_rows.extend(rows)

            summ = summarize(rows)
            status = "active"
            if summ["sr"] < stop_sr:
                active[expert_name] = False
                status = "below_stop_sr"
            summary_row = {
                "difficulty_level": level_idx,
                "task_profile": profile_name,
                "expert": expert_name,
                "status": status,
                "preset_label": preset_label,
                "preset": repr(preset),
                "wind_speed_mps": float(wind_speed_mps),
                "actual_obstacles": int(cfg.get("test", {}).get(
                    "actual_obstacles", 0)),
                "obs_obstacle_slots": int(cfg.get("scene", {}).get(
                    "n_obstacles", 0)),
                "n": int(summ["n"]),
                "success_rate": float(summ["sr"]),
                "avg_steps": float(summ["avg_steps"]),
                "dtf_mm": float(summ["dtf_mm"]),
                "zerr_mm": float(summ["zerr_mm"]),
                "rebar_mm": float(summ["rebar_mm"]),
                "top_terms": repr(summ["top_terms"]),
            }
            summary_rows.append(summary_row)
            print(f"  [RESULT] SR={summ['sr']*100:5.1f}% "
                  f"avg_steps={summ['avg_steps']:5.1f} "
                  f"dtf={summ['dtf_mm']:5.1f}mm "
                  f"zerr={summ['zerr_mm']:5.1f}mm "
                  f"rebar={summ['rebar_mm']:5.1f}mm "
                  f"status={status} terms={summ['top_terms']}")

        if not any(active.values()):
            print("\n[LADDER] all experts are below threshold; stopping.")
            break

    stamp = time.strftime("%Y%m%d_%H%M%S")
    detail_out = args.out or os.path.join(
        "test_results", f"traditional_ladder_detail_{stamp}.csv")
    summary_out = args.summary_out or os.path.join(
        "test_results", f"traditional_ladder_summary_{stamp}.csv")
    write_csv(detail_out, all_rows)
    write_csv(summary_out, summary_rows)
    print(f"\n[CSV] detail:  {detail_out}")
    print(f"[CSV] summary: {summary_out}")


def wind_baseline_preset(expert_name, task_profile=None):
    presets = TUNE_PRESETS.get(expert_name, [{}])
    idx_map = WIND_BASELINE_PRESET_IDX_BY_TASK.get(
        task_profile, WIND_BASELINE_PRESET_IDX)
    idx = idx_map.get(expert_name)
    if idx is None or idx >= len(presets):
        return {}, ""
    return copy.deepcopy(presets[idx]), str(idx)


def run_wind_ladder(args):
    experts = parse_experts(args.experts)
    speeds = parse_float_list(args.wind_speeds)
    all_rows = []
    summary_rows = []

    print(f"[WIND] task={args.task_profile}, speeds={speeds} m/s")
    if args.wind_direction is None:
        print("[WIND] direction: random uniform per episode")
    else:
        print(f"[WIND] direction: fixed {float(args.wind_direction):.3f} rad")
    print("[WIND] presets: fixed 24mm no-wind baseline presets")

    for level_idx, speed in enumerate(speeds):
        print(f"\n[WIND-L{level_idx}] speed={speed:.2f} m/s")
        for expert_name in experts:
            preset, preset_idx = wind_baseline_preset(
                expert_name, task_profile=args.task_profile)
            cfg = build_config(args, extra_traditional={expert_name: preset})
            rows = evaluate_expert(
                expert_name, cfg, args.episodes, args.seed + level_idx * 1000,
                verbose=args.verbose, trace_every=args.trace_every,
                wind_speed_mps=speed, wind_direction_rad=args.wind_direction)
            for row in rows:
                row["task_profile"] = args.task_profile
                row["wind_level"] = level_idx
                row["preset_idx"] = preset_idx
                row["preset"] = repr(preset)
            all_rows.extend(rows)

            summ = summarize(rows)
            wind_force = float(np.mean([
                float(r.get("wind_force_n", 0.0)) for r in rows
            ])) if rows else 0.0
            summary_row = {
                "wind_level": level_idx,
                "wind_speed_mps": float(speed),
                "wind_force_n": wind_force,
                "task_profile": args.task_profile,
                "expert": expert_name,
                "preset_idx": preset_idx,
                "preset": repr(preset),
                "n": int(summ["n"]),
                "success_rate": float(summ["sr"]),
                "avg_steps": float(summ["avg_steps"]),
                "dtf_mm": float(summ["dtf_mm"]),
                "zerr_mm": float(summ["zerr_mm"]),
                "rebar_mm": float(summ["rebar_mm"]),
                "top_terms": repr(summ["top_terms"]),
            }
            summary_rows.append(summary_row)
            print(f"  {expert_name:9s} SR={summ['sr']*100:5.1f}% "
                  f"F={wind_force:5.3f}N "
                  f"dtf={summ['dtf_mm']:5.1f}mm "
                  f"zerr={summ['zerr_mm']:5.1f}mm "
                  f"rebar={summ['rebar_mm']:5.1f}mm "
                  f"terms={summ['top_terms']}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    detail_out = args.out or os.path.join(
        "test_results", f"traditional_wind_detail_{stamp}.csv")
    summary_out = args.summary_out or os.path.join(
        "test_results", f"traditional_wind_summary_{stamp}.csv")
    write_csv(detail_out, all_rows)
    write_csv(summary_out, summary_rows)
    print(f"\n[CSV] detail:  {detail_out}")
    print(f"[CSV] summary: {summary_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=str, default="pid,damped_pd,mpc")
    parser.add_argument("--task-profile", type=str, default="easy-single-rebar",
                        choices=sorted(TASK_PROFILES))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--preset-mode", type=str, default="fixed-best",
                        choices=["fixed-best", "none", "tune"],
                        help="traditional expert preset selection")
    parser.add_argument("--tune-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--obstacles", type=int, default=None,
                        help="actual obstacle count; obs slots stay at default")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--trace-every", type=int, default=0)
    parser.add_argument("--ladder", action="store_true")
    parser.add_argument("--ladder-name", type=str,
                        default="default-descent-no-wind",
                        choices=sorted(TASK_LADDERS))
    parser.add_argument("--ladder-profiles", type=str, default=None)
    parser.add_argument("--ladder-no-tune", action="store_true")
    parser.add_argument("--seed-stride", type=int, default=100)
    parser.add_argument("--stop-sr", type=float, default=0.10)
    parser.add_argument("--wind-ladder", action="store_true")
    parser.add_argument("--wind-speeds", type=str,
                        default="0,1,2,3.5,5,6.5,8,10")
    parser.add_argument("--wind-direction", type=float, default=None)
    parser.add_argument("--wind-variable", action="store_true")
    parser.add_argument("--summary-out", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.wind_ladder:
        run_wind_ladder(args)
        return

    if args.ladder:
        run_ladder(args)
        return

    experts = parse_experts(args.experts)
    profile = TASK_PROFILES[args.task_profile]
    print(f"[TASK] {args.task_profile}: {profile['description']}")

    all_rows = []
    selected_presets = {}
    for expert_name in experts:
        preset, preset_label = select_preset(expert_name, args)
        selected_presets[expert_name] = {
            "label": preset_label,
            "preset": preset,
        }
        cfg = build_config(args, extra_traditional={expert_name: preset})
        wind_speed_mps = paper_profile_wind(cfg)
        print(f"\n[EVAL] {expert_name}: episodes={args.episodes}, "
              f"wind={wind_speed_mps:.1f}m/s, preset={preset_label}")
        rows = evaluate_expert(
            expert_name, cfg, args.episodes, args.seed, verbose=args.verbose,
            trace_every=args.trace_every, wind_speed_mps=wind_speed_mps,
            wind_direction_rad=args.wind_direction)
        for row in rows:
            row["task_profile"] = args.task_profile
            row["preset_label"] = preset_label
            row["preset"] = repr(preset)
            row["true_env_obs"] = True
            row["actual_obstacles"] = int(cfg.get("test", {}).get(
                "actual_obstacles", 0))
            row["obs_obstacle_slots"] = int(cfg.get("scene", {}).get(
                "n_obstacles", 0))
        all_rows.extend(rows)
        summ = summarize(rows)
        print(f"  SR={summ['sr']*100:5.1f}% "
              f"avg_steps={summ['avg_steps']:5.1f} "
              f"dtf={summ['dtf_mm']:5.1f}mm "
              f"zerr={summ['zerr_mm']:5.1f}mm "
              f"rebar={summ['rebar_mm']:5.1f}mm "
              f"terms={summ['top_terms']}")

    out = args.out
    if not out:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join(
            "test_results",
            f"traditional_experts_{args.task_profile}_{stamp}.csv")
    write_csv(out, all_rows)
    print(f"\n[CSV] {out}")
    print(f"[SELECTED] {selected_presets}")


if __name__ == "__main__":
    main()
