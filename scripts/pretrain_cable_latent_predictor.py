#!/usr/bin/env python3
"""Fixed-budget predictor-only CableLatPred pretraining.

This script rolls out a frozen policy/checkpoint and updates only the online
CableLatPred ensemble.  The PPO policy and CableEncoder are not optimized here.
"""

import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import DEFAULT_CONFIG
from obs_predictor import build_cable_latent_predictor
from phase_agent import PPOPhaseAgent
from train_phase import (
    Logger,
    REWARD_STATES,
    _apply_episode_wind_vec,
    _apply_pretrain_wind_floor,
    _deep_update,
    _init_csv_log,
    _lucky_reject_enabled,
    _make_obs_pred_pretrain_curriculum,
    _obs_pred_cable_latent_dim,
    _phase_obs_with_predicted_cable_latent,
    _phase_visible_no_cable_vector,
    _rope_marker_feature_source_config,
    _save_cable_latent_predictor_checkpoint,
    _use_rope_marker_features,
    make_phase_env_and_controllers,
)


def build_config(args):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    profile = cfg.get("train_profiles", {}).get(args.profile, {})
    _deep_update(cfg, profile.get("config", {}))
    cfg.setdefault("train", {})["total_timesteps"] = int(args.timesteps)
    cfg.setdefault("train", {})["n_envs"] = int(args.n_envs)
    cfg.setdefault("train", {})["seed"] = int(args.seed)
    cfg.setdefault("vision", {})["enabled"] = False
    cfg.setdefault("observation_predictor", {})["enabled"] = False
    ce = cfg.setdefault("cable_encoder", {})
    ce["enabled"] = True
    ce["zero_obs"] = False
    ce["output_dim"] = int(args.cable_encoder_output_dim)
    ce["trainable"] = True
    ce["lr"] = float(args.cable_encoder_lr)
    cfg.setdefault("descent_rl", {})["obs_dim"] = 44 + int(args.cable_encoder_output_dim)
    clp = cfg.setdefault("cable_latent_predictor", {})
    clp["enabled"] = True
    clp["train_enabled"] = True
    clp["latent_dim"] = int(args.cable_encoder_output_dim)
    clp["use_rope_marker_features"] = bool(args.use_rope_markers)
    if args.use_rope_markers:
        cfg.setdefault("rope_markers", {})["enabled"] = True
        cfg.setdefault("rope_markers", {})["feature_source"] = str(args.rope_marker_source)
        clp["rope_marker_feature_source"] = str(args.rope_marker_source)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="descent_variable_wind_true_obs")
    ap.add_argument("--phase", default="descent")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--resume-ckpt", required=True)
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=270707)
    ap.add_argument("--cable-encoder-output-dim", type=int, default=8)
    ap.add_argument("--cable-encoder-lr", type=float, default=1e-4)
    ap.add_argument("--use-rope-markers", action="store_true")
    ap.add_argument("--rope-marker-source", default="site")
    args = ap.parse_args()

    from vec_env import make_vec_env

    cfg = build_config(args)
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, "resolved_config.json"),
              "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    phase = args.phase
    T = int(args.timesteps)
    n_envs = int(args.n_envs)
    start_method = cfg.get("train", {}).get("vec_env_start_method", "spawn")
    print(f"\n{'=' * 60}\n  CableLatPred PRETRAIN [VEC n_envs={n_envs}] | "
          f"{phase.upper()} | {T} steps | {args.log_dir}\n{'=' * 60}\n")
    print(f"  [Policy] frozen untrained/mainline-init policy loaded from {args.resume_ckpt}")
    agent = PPOPhaseAgent(phase, config=cfg)
    agent.load(args.resume_ckpt)

    def _make_one(wid):
        return make_phase_env_and_controllers(phase, cfg, worker_id=wid)

    vec = make_vec_env(_make_one, n_envs=n_envs, start_method=start_method)
    cur = _make_obs_pred_pretrain_curriculum(cfg, phase)
    curs = [cur for _ in range(n_envs)]
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
    last_actions = [np.zeros(7, dtype=np.float32) for _ in range(n_envs)]
    rope_marker_cache = [None] * n_envs
    use_markers = _use_rope_marker_features(cfg)

    def reset_one(i, max_retries=10):
        for _ in range(max_retries):
            pert = _apply_pretrain_wind_floor(cfg, curs[i],
                                              curs[i].sample_episode_perturbations())
            if phase == "descent":
                obs, pp = vec.reset_for_descent(i, cur_init=curs[i].get_descent_init())
            else:
                obs, pp = vec.reset(i)
            if obs is not None:
                break
        else:
            raise RuntimeError(f"worker {i}: CableLatPred pretrain reset failed")
        vec.set_force_noise(i, pert["force_noise"])
        _apply_episode_wind_vec(vec, i, pert)
        vec.reset_controllers(i, obs, vec.get_qpos(i), pp)
        txy = np.asarray(vec.env_attr(i, "target_pos"), np.float32)[:2]
        rs = REWARD_STATES[phase]()
        if phase == "descent":
            di = curs[i].get_descent_init()
            if di is not None:
                rs.current_xy_range = di["xy_range"]
                rs.current_xy_tol = di["xy_tol"]
                rs.current_descent_level = curs[i].level_idx
                rs.descent_n_levels = curs[i].n_levels
        return obs, None, txy, rs, pert

    logger = None
    try:
        for i in range(n_envs):
            obs_list[i], sxy_list[i], txy_list[i], rstate_list[i], perts[i] = reset_one(i)
            obs_histories[i].reset()

        phase_obs_cache = [None] * n_envs
        for i in range(n_envs):
            phase_obs_cache[i] = vec.build_phase_obs_remote(
                i, phase, obs_list[i], sxy_list[i], txy_list[i], pt_list[i], py_list[i])
            if use_markers and hasattr(vec, "get_rope_marker_features"):
                rope_marker_cache[i] = vec.get_rope_marker_features(i)

        visible_dim = int(_phase_visible_no_cable_vector(
            phase_obs_cache[0],
            rope_marker_features=rope_marker_cache[0] if use_markers else None).size)
        latent_dim = _obs_pred_cable_latent_dim(cfg, agent)
        predictors = [
            build_cable_latent_predictor(
                cfg, phase, visible_dim, latent_dim=latent_dim,
                action_dim=7, device=getattr(agent, "device", None))
            for _ in range(n_envs)
        ]
        print(f"  [CableLatPred-PRETRAIN] visible_dim={visible_dim}, "
              f"latent_dim={latent_dim}, predictors={n_envs}")
        if use_markers:
            print("  [CableLatPred-PRETRAIN] rope marker features enabled: "
                  f"source={_rope_marker_feature_source_config(cfg)}")
        for i, pred in enumerate(predictors):
            pred.reset(_phase_visible_no_cable_vector(
                phase_obs_cache[i],
                rope_marker_features=rope_marker_cache[i] if use_markers else None))

        logger = Logger(args.log_dir, project="phase_rl_v11_ablation",
                        run_name=os.environ.get(
                            "WANDB_NAME", "full8d_cable_latpred_pretrain"))
        logger.update_config(cfg)
        lf = os.path.join(args.log_dir, f"{phase}_cable_latpred_pretrain_log.csv")
        _init_csv_log(lf, [
            "episode", "total_steps", "ep_reward", "success", "steps",
            "pred_loss", "pred_huber", "pred_mse", "pred_rmse",
            "pred_rmse_norm", "pred_norm", "target_norm",
            "lvl", "wind_speed", "wind_speed_min", "wind_speed_max",
            "worker_id", "termination",
        ], append_existing=False)

        ts = 0
        ep_count = 0
        best_rmse = float("inf")
        acc = [
            {"loss": 0.0, "huber": 0.0, "mse": 0.0, "rmse": 0.0,
             "rmse_norm": 0.0, "pred_norm": 0.0, "target_norm": 0.0,
             "updates": 0, "steps": 0}
            for _ in range(n_envs)
        ]
        t0 = time.time()

        while ts < T:
            actions = []
            for i in range(n_envs):
                core, cable, wind, _, _, _ = phase_obs_cache[i]
                po = agent.encode_obs(core, cable, wind)
                no = agent.normalize_obs(po, update=False)
                act, _, _ = agent.act_with_history(
                    no, obs_histories[i], deterministic=True)
                actions.append(act)

            payloads = []
            for i in range(n_envs):
                payloads.append({
                    "phase": phase,
                    "rl_action": actions[i],
                    "obs": obs_list[i],
                    "current_q": vec.get_qpos(i),
                    "start_xy": sxy_list[i],
                    "target_xy": txy_list[i],
                    "prev_tilt": pt_list[i],
                    "prev_yaw": py_list[i],
                    "rstate": rstate_list[i],
                    "act_noise": perts[i]["act_noise"],
                    "base_dq": phase_obs_cache[i][5],
                    "train_reject_lucky_rebar_insert": _lucky_reject_enabled(cfg),
                })
            if hasattr(vec, "remotes"):
                for i, p in enumerate(payloads):
                    vec.remotes[i].send(("rl_step", p))
                results = [vec._check_recv(vec.remotes[i].recv(), i)
                           for i in range(n_envs)]
            else:
                results = [vec.rl_step(i, payloads[i]) for i in range(n_envs)]

            for i, res in enumerate(results):
                if use_markers:
                    rmf = (res.get("info", {}) or {}).get("rope_marker_features", None)
                    if rmf is not None:
                        rope_marker_cache[i] = rmf
                done = bool(res["done"])
                obs_list[i] = res["new_obs"]
                rstate_list[i] = res["rstate"]
                ep_rewards[i] += float(res["reward"])
                ep_steps[i] += 1
                ep_suc[i] = ep_suc[i] or bool(res["success"])
                if done:
                    ep_term[i] = res.get("termination", "done")
                ts += 1

                if not done:
                    if res.get("new_core_obs") is not None:
                        next_phase_obs = (
                            res["new_core_obs"], res["new_cable_raw"],
                            res["new_wind_obs"], res["new_tilt"],
                            res["new_yaw"], res.get("new_base_dq"))
                    else:
                        next_phase_obs = vec.build_phase_obs_remote(
                            i, phase, obs_list[i], sxy_list[i], txy_list[i],
                            pt_list[i], py_list[i])
                    _, info = _phase_obs_with_predicted_cable_latent(
                        next_phase_obs, predictors[i], res.get("delta_q", None),
                        agent, cfg, phase=phase, target_phase_obs=next_phase_obs,
                        rope_marker_features=rope_marker_cache[i] if use_markers else None)
                    if info is not None:
                        acc[i]["steps"] += 1
                        acc[i]["pred_norm"] += float(getattr(info, "pred_norm", 0.0))
                        acc[i]["target_norm"] += float(getattr(info, "target_norm", 0.0))
                        if getattr(info, "trained", False):
                            for k in ("loss", "huber", "mse", "rmse", "rmse_norm"):
                                acc[i][k] += float(getattr(info, k, 0.0))
                            acc[i]["updates"] += 1
                    phase_obs_cache[i] = next_phase_obs
                    pt_list[i] = next_phase_obs[3]
                    py_list[i] = next_phase_obs[4]

                if done or ts >= T:
                    ep_count += 1
                    den = max(int(acc[i]["updates"]), 1)
                    sden = max(int(acc[i]["steps"]), 1)
                    row = [
                        ep_count, ts, f"{ep_rewards[i]:.3f}", int(ep_suc[i]),
                        ep_steps[i], f"{acc[i]['loss']/den:.6f}",
                        f"{acc[i]['huber']/den:.6f}",
                        f"{acc[i]['mse']/den:.6f}",
                        f"{acc[i]['rmse']/den:.6f}",
                        f"{acc[i]['rmse_norm']/den:.6f}",
                        f"{acc[i]['pred_norm']/sden:.6f}",
                        f"{acc[i]['target_norm']/sden:.6f}",
                        curs[i].level_idx,
                        f"{perts[i].get('wind_speed', 0.0):.4f}",
                        f"{perts[i].get('wind_min', 0.0):.4f}",
                        f"{perts[i].get('wind_max', 0.0):.4f}",
                        i, ep_term[i],
                    ]
                    with open(lf, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(row)
                    logger.log(ep_count, {
                        f"cable_latpred_pretrain/{phase}/loss": acc[i]["loss"] / den,
                        f"cable_latpred_pretrain/{phase}/rmse": acc[i]["rmse"] / den,
                        f"cable_latpred_pretrain/{phase}/rmse_norm": acc[i]["rmse_norm"] / den,
                        f"cable_latpred_pretrain/{phase}/success": float(ep_suc[i]),
                        f"cable_latpred_pretrain/{phase}/worker_id": i,
                    })
                    if acc[i]["rmse"] / den < best_rmse:
                        best_rmse = acc[i]["rmse"] / den
                        _save_cable_latent_predictor_checkpoint(
                            predictors, os.path.join(
                                args.log_dir, "ckpt_best_cable_latent_predictor.pt"))
                    _save_cable_latent_predictor_checkpoint(
                        predictors, os.path.join(
                            args.log_dir, "ckpt_latest_cable_latent_predictor.pt"))
                    if ep_count % 25 == 0:
                        print(f"[CableLatPredPretrain] Ep{ep_count:5d} "
                              f"[{ts:7d}] rmse:{acc[i]['rmse']/den:.4f} "
                              f"rmse_norm:{acc[i]['rmse_norm']/den:.4f} "
                              f"W:{perts[i].get('wind_speed',0.0):.2f} "
                              f"term:{ep_term[i]}")
                    obs_list[i], sxy_list[i], txy_list[i], rstate_list[i], perts[i] = reset_one(i)
                    phase_obs_cache[i] = vec.build_phase_obs_remote(
                        i, phase, obs_list[i], sxy_list[i], txy_list[i], 0.0, 0.0)
                    if use_markers and hasattr(vec, "get_rope_marker_features"):
                        rope_marker_cache[i] = vec.get_rope_marker_features(i)
                    predictors[i].reset(_phase_visible_no_cable_vector(
                        phase_obs_cache[i],
                        rope_marker_features=rope_marker_cache[i] if use_markers else None))
                    obs_histories[i].reset()
                    pt_list[i] = 0.0
                    py_list[i] = 0.0
                    ep_rewards[i] = 0.0
                    ep_steps[i] = 0
                    ep_suc[i] = False
                    ep_term[i] = "running"
                    acc[i] = {k: 0.0 for k in acc[i]}
                    acc[i]["updates"] = 0
                    acc[i]["steps"] = 0

        _save_cable_latent_predictor_checkpoint(
            predictors, os.path.join(args.log_dir, "ckpt_final_cable_latent_predictor.pt"))
        print(f"\n[CableLatPred-PRETRAIN] Done: {ts} steps, "
              f"{(time.time()-t0)/60:.1f} min, best_rmse={best_rmse:.6f}")
    finally:
        try:
            vec.close()
        finally:
            try:
                if logger is not None:
                    logger.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
