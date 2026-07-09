#!/usr/bin/env python3
"""Create an untrained Full8D mainline PPO checkpoint.

The checkpoint is intentionally untrained but has the exact policy/cable
encoder architecture used by the current mainline: PPO LSTM + PID residual
base + trainable 8D CableEncoder.
"""

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import DEFAULT_CONFIG
from phase_agent import PPOPhaseAgent
from train_phase import _deep_update


def build_full8d_config(seed):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    profile = cfg.get("train_profiles", {}).get("descent_variable_wind_true_obs", {})
    _deep_update(cfg, profile.get("config", {}))
    cfg.setdefault("train", {})["seed"] = int(seed)
    cfg.setdefault("train", {})["n_envs"] = 8
    cfg.setdefault("train", {})["total_timesteps"] = 8_000_000
    cfg.setdefault("vision", {})["enabled"] = False
    cfg.setdefault("observation_predictor", {})["enabled"] = False
    cfg.setdefault("cable_latent_predictor", {})["enabled"] = False
    ce = cfg.setdefault("cable_encoder", {})
    ce["enabled"] = True
    ce["zero_obs"] = False
    ce["output_dim"] = 8
    ce["trainable"] = True
    ce["lr"] = 1e-4
    cfg.setdefault("descent_rl", {})["obs_dim"] = 52
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=270716)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = build_full8d_config(args.seed)
    agent = PPOPhaseAgent("descent", config=cfg)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    agent.save(args.out)
    with open(os.path.join(os.path.dirname(args.out), "resolved_config_init.json"),
              "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"saved untrained Full8D init checkpoint: {args.out}")
    print("obs_dim=52 cable_encoder.output_dim=8 trainable=True total_steps=0")


if __name__ == "__main__":
    main()
