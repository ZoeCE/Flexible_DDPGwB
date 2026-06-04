import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PHASE_ONE_HOT_DIM = 3  # keep checkpoint-compatible input width
PHASE_TO_ID = {"cruise": 1, "descent": 2}


@dataclass
class ObsPredictionStepInfo:
    enabled: bool = False
    used_prediction: bool = False
    measured_step: bool = True
    trained: bool = False
    loss: float = 0.0
    nll: float = 0.0
    raw_nll: float = 0.0
    huber: float = 0.0
    rmse_norm: float = 0.0
    rmse_raw: float = 0.0
    rmse_non_cable: float = 0.0
    rmse_cable_latent: float = 0.0
    rmse_non_cable_norm: float = 0.0
    rmse_cable_latent_norm: float = 0.0
    log_std_mean: float = 0.0
    log_std_min: float = 0.0
    std_mean: float = 0.0


class LSTMObservationPredictorNet(nn.Module):
    """One-step raw-observation predictor with a diagonal Gaussian head."""

    def __init__(self, input_dim, obs_dim, hidden_dim=128, n_layers=1,
                 dropout=0.0, log_std_min=-5.0, log_std_max=1.0):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=float(dropout) if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_delta = nn.Linear(hidden_dim, obs_dim)
        self.log_std = nn.Linear(hidden_dim, obs_dim)
        nn.init.zeros_(self.mean_delta.weight)
        nn.init.zeros_(self.mean_delta.bias)
        nn.init.constant_(self.log_std.bias, -1.5)

    def forward(self, x, hx=None):
        y, hx_new = self.lstm(x, hx)
        feat = self.head(y[:, -1, :])
        mean_delta = self.mean_delta(feat)
        log_std = self.log_std(feat).clamp(self.log_std_min, self.log_std_max)
        return mean_delta, log_std, hx_new


class OnlineObservationPredictor:
    """
    Online LSTM predictor for missing control-cycle observations.

    The RL/controller path sees a true observation only every
    `measurement_period_steps` control steps. Intermediate observations are
    predicted. In simulation, the hidden true observation is still used as a
    supervised target and by reward/termination logic, but it is not returned to
    the RL/controller path.
    """

    def __init__(self, config, phase, obs_dim, action_dim=7, device=None):
        cfg = config.get("observation_predictor", {})
        self.config = config
        self.cfg = cfg
        self.phase = phase
        self.enabled = bool(cfg.get("enabled", False))
        self.measurement_period = max(1, int(cfg.get("measurement_period_steps", 2)))
        self.warmup_true_steps = max(0, int(cfg.get("warmup_true_steps", 0)))
        self.train_on_all_steps = bool(cfg.get("train_on_all_steps", True))
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.target_mode = str(cfg.get("target_mode", "raw")).lower()
        self.non_cable_dim = max(0, int(cfg.get("non_cable_dim", 54)))
        self.cable_latent_dim = max(0, int(cfg.get("cable_latent_dim", 32)))
        self.device = device or torch.device(
            f"cuda:{config.get('train', {}).get('gpu_id', 0)}"
            if torch.cuda.is_available() and config.get("train", {}).get("gpu_id", 0) >= 0
            else "cpu")

        self.obs_clip = float(cfg.get("obs_norm_clip", 8.0))
        self.action_clip = float(cfg.get("action_norm_clip", 2.0))
        self.huber_coef = float(cfg.get("huber_coef", 0.25))
        self.nll_coef = float(cfg.get("nll_coef", 1.0))
        self.nll_error_clip = float(cfg.get("nll_error_clip", 0.0))
        self.grad_clip = float(cfg.get("grad_clip", 1.0))
        self.train_enabled = bool(cfg.get("train_enabled", True))

        self.obs_scale = self._build_obs_scale(self.obs_dim, cfg)
        self.loss_weights_np = self._build_loss_weights(self.obs_dim, cfg)
        self.loss_weights = torch.from_numpy(self.loss_weights_np).to(
            self.device).view(1, -1)
        dq_max = np.asarray(
            config.get("space", {}).get("dq_max", [0.12] * self.action_dim),
            dtype=np.float32)
        if dq_max.size < self.action_dim:
            dq_max = np.pad(dq_max, (0, self.action_dim - dq_max.size),
                            constant_values=float(np.mean(dq_max)))
        self.action_scale = np.maximum(dq_max[:self.action_dim], 1e-6)

        input_dim = self.obs_dim + self.action_dim + 3 + 1
        self.net = LSTMObservationPredictorNet(
            input_dim=input_dim,
            obs_dim=self.obs_dim,
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            n_layers=int(cfg.get("n_layers", 1)),
            dropout=float(cfg.get("dropout", 0.0)),
            log_std_min=float(cfg.get("log_std_min", -5.0)),
            log_std_max=float(cfg.get("log_std_max", 1.0)),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=float(cfg.get("lr", 3e-4)),
            eps=1e-5,
            weight_decay=float(cfg.get("weight_decay", 0.0)),
        )

        self.hidden = None
        self._pending = None
        self._pending_pred_obs = None
        self._last_returned_measured = True
        self._episode_step = 0

    @staticmethod
    def _uses_non_cable_latent_target(cfg):
        mode = str(cfg.get("target_mode", "raw")).lower()
        return mode in ("non_cable_latent", "compact_latent", "latent")

    @staticmethod
    def _build_raw_obs_scale(obs_dim, cfg):
        scale = np.ones(obs_dim, dtype=np.float32)
        if obs_dim >= 10:
            scale[:10] = np.asarray(
                [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
                dtype=np.float32)
        if obs_dim >= 31:
            scale[19:31] = np.asarray(
                [1.0, 2.0, 1.0, 2.0, np.pi, 3.0, np.pi, 3.0,
                 np.pi, 3.0, np.pi, np.pi],
                dtype=np.float32)
        if obs_dim >= 45:
            scale[31:38] = np.pi
            scale[38:45] = 3.0
        if obs_dim >= 54:
            scale[45:48] = 1.0
            scale[48:50] = 1.0
            scale[50:54] = float(cfg.get("rebar_error_scale", 0.05))
        if obs_dim > 54:
            pattern = np.asarray([0.5, 0.5, 0.5, 2.0, 2.0, 2.0], dtype=np.float32)
            n = obs_dim - 54
            scale[54:] = np.resize(pattern, n)
        return scale

    @staticmethod
    def _build_obs_scale(obs_dim, cfg):
        if OnlineObservationPredictor._uses_non_cable_latent_target(cfg):
            head_dim = min(max(0, int(cfg.get("non_cable_dim", 54))), obs_dim)
            scale = np.ones(obs_dim, dtype=np.float32)
            if head_dim > 0:
                scale[:head_dim] = OnlineObservationPredictor._build_raw_obs_scale(
                    head_dim, cfg)
            if obs_dim > head_dim:
                scale[head_dim:] = float(cfg.get(
                    "cable_latent_scale", cfg.get("latent_scale", 1.0)))
        else:
            scale = OnlineObservationPredictor._build_raw_obs_scale(obs_dim, cfg)
        override = cfg.get("global_scale", None)
        if override is not None:
            scale[:] = float(override)
        return np.maximum(scale, 1e-6)

    @staticmethod
    def _build_loss_weights(obs_dim, cfg):
        weights = np.ones(obs_dim, dtype=np.float32)
        if OnlineObservationPredictor._uses_non_cable_latent_target(cfg):
            head_dim = min(max(0, int(cfg.get("non_cable_dim", 54))), obs_dim)
            weights[:head_dim] = float(cfg.get("non_cable_loss_weight", 2.0))
            if obs_dim > head_dim:
                weights[head_dim:] = float(cfg.get("cable_latent_loss_weight", 1.0))
        override = cfg.get("global_loss_weight", None)
        if override is not None:
            weights[:] = float(override)
        return np.maximum(weights, 1e-6)

    def reset(self, initial_obs):
        self.hidden = None
        self._pending = None
        self._pending_pred_obs = None
        self._last_returned_measured = True
        self._episode_step = 0

    def _detach_hidden(self):
        if self.hidden is None:
            return
        self.hidden = tuple(h.detach() for h in self.hidden)

    def _phase_one_hot(self, phase):
        out = np.zeros(PHASE_ONE_HOT_DIM, dtype=np.float32)
        out[PHASE_TO_ID.get(phase, PHASE_TO_ID.get(self.phase, 1))] = 1.0
        return out

    def _norm_obs_np(self, obs):
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if arr.size != self.obs_dim:
            raise ValueError(f"obs dim mismatch: got {arr.size}, expected {self.obs_dim}")
        return np.clip(arr / self.obs_scale, -self.obs_clip, self.obs_clip)

    def _denorm_obs_np(self, obs_norm):
        return (np.asarray(obs_norm, dtype=np.float32) * self.obs_scale).astype(np.float32)

    def _build_input_np(self, obs, action, phase):
        obs_norm = self._norm_obs_np(obs)
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        if act.size < self.action_dim:
            act = np.pad(act, (0, self.action_dim - act.size))
        act = np.clip(act[:self.action_dim] / self.action_scale,
                      -self.action_clip, self.action_clip)
        measured_flag = np.asarray([1.0 if self._last_returned_measured else 0.0],
                                   dtype=np.float32)
        return np.concatenate([obs_norm, act, self._phase_one_hot(phase),
                               measured_flag]).astype(np.float32), obs_norm

    def predict_next(self, current_obs, action, phase=None):
        if not self.enabled:
            self._pending = None
            self._pending_pred_obs = None
            return np.asarray(current_obs, dtype=np.float32).copy()

        phase = phase or self.phase
        self._detach_hidden()
        x_np, obs_norm_np = self._build_input_np(current_obs, action, phase)
        x = torch.from_numpy(x_np).to(self.device).view(1, 1, -1)
        curr_norm = torch.from_numpy(obs_norm_np).to(self.device).view(1, -1)

        if self.train_enabled:
            mean_delta, log_std, hidden_new = self.net(x, self.hidden)
            pred_norm = curr_norm + mean_delta
            self.hidden = hidden_new
            pred_norm_np = pred_norm.detach().cpu().numpy().reshape(-1)
            pred_obs = self._denorm_obs_np(
                np.clip(pred_norm_np, -self.obs_clip, self.obs_clip))
            self._pending = {
                "pred_norm": pred_norm,
                "log_std": log_std,
                "pred_obs": pred_obs,
            }
        else:
            with torch.no_grad():
                mean_delta, log_std, hidden_new = self.net(x, self.hidden)
                pred_norm = curr_norm + mean_delta
                self.hidden = tuple(h.detach() for h in hidden_new)
                pred_norm_np = pred_norm.cpu().numpy().reshape(-1)
                pred_obs = self._denorm_obs_np(
                    np.clip(pred_norm_np, -self.obs_clip, self.obs_clip))
            self._pending = None

        self._pending_pred_obs = pred_obs
        return pred_obs.copy()

    def _supervise_pending(self, true_next_obs, hidden_target):
        info = ObsPredictionStepInfo(enabled=self.enabled)
        if not self.enabled or self._pending is None:
            return info
        if (not self.train_on_all_steps) and (not hidden_target):
            self._pending = None
            return info

        target_np = self._norm_obs_np(true_next_obs)
        target = torch.from_numpy(target_np).to(self.device).view(1, -1)
        pred_norm = self._pending["pred_norm"]
        log_std = self._pending["log_std"]
        pred_obs = self._pending["pred_obs"]

        err = pred_norm - target
        std = log_std.exp().clamp_min(1e-4)
        weights = self.loss_weights.to(pred_norm.device)
        weight_den = weights.sum().clamp_min(1e-6)
        z = err / std
        raw_nll = 0.5 * (z.pow(2) + 2.0 * log_std)
        raw_nll = (raw_nll * weights).sum() / weight_den
        if self.nll_error_clip > 0.0:
            z = z.clamp(-self.nll_error_clip, self.nll_error_clip)
        nll = 0.5 * (z.pow(2) + 2.0 * log_std)
        nll = (nll * weights).sum() / weight_den
        huber = F.smooth_l1_loss(pred_norm, target, reduction="none")
        huber = (huber * weights).sum() / weight_den
        loss = self.nll_coef * nll + self.huber_coef * huber

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()
        self._detach_hidden()

        err_norm = (pred_norm.detach() - target).cpu().numpy().reshape(-1)
        err_raw = pred_obs - np.asarray(true_next_obs, dtype=np.float32).reshape(-1)
        info.trained = True
        info.loss = float(loss.detach().cpu().item())
        info.nll = float(nll.detach().cpu().item())
        info.raw_nll = float(raw_nll.detach().cpu().item())
        info.huber = float(huber.detach().cpu().item())
        log_std_det = log_std.detach()
        std_det = std.detach()
        info.log_std_mean = float(log_std_det.mean().cpu().item())
        info.log_std_min = float(log_std_det.min().cpu().item())
        info.std_mean = float(std_det.mean().cpu().item())
        info.rmse_norm = float(np.sqrt(np.mean(np.square(err_norm))))
        info.rmse_raw = float(np.sqrt(np.mean(np.square(err_raw))))
        if self._uses_non_cable_latent_target(self.cfg):
            split = min(self.non_cable_dim, err_raw.size)
            if split > 0:
                info.rmse_non_cable = float(
                    np.sqrt(np.mean(np.square(err_raw[:split]))))
                info.rmse_non_cable_norm = float(
                    np.sqrt(np.mean(np.square(err_norm[:split]))))
            if err_raw.size > split:
                info.rmse_cable_latent = float(
                    np.sqrt(np.mean(np.square(err_raw[split:]))))
                info.rmse_cable_latent_norm = float(
                    np.sqrt(np.mean(np.square(err_norm[split:]))))
        self._pending = None
        return info

    def should_use_prediction(self, next_step_index):
        if not self.enabled or self.measurement_period <= 1:
            return False
        if int(next_step_index) <= self.warmup_true_steps:
            return False
        return (int(next_step_index) % self.measurement_period) != 0

    def observe_result(self, true_next_obs, next_step_index):
        hidden_target = self.should_use_prediction(next_step_index)
        info = self._supervise_pending(true_next_obs, hidden_target)
        info.enabled = self.enabled
        info.used_prediction = bool(hidden_target and self._pending_pred_obs is not None)
        info.measured_step = not info.used_prediction

        if info.used_prediction:
            next_obs = self._pending_pred_obs.copy()
            self._last_returned_measured = False
        else:
            next_obs = np.asarray(true_next_obs, dtype=np.float32).copy()
            self._last_returned_measured = True
        self._episode_step = int(next_step_index)
        self._pending_pred_obs = None
        return next_obs, info

    def state_dict(self):
        return {
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "phase": self.phase,
            "cfg": dict(self.cfg),
        }

    def load_state_dict(self, state):
        if isinstance(state, dict) and state.get("type") == "ensemble":
            states = [s for s in state.get("predictors", []) if s is not None]
            if not states:
                raise ValueError("empty obs predictor ensemble checkpoint")
            state = states[0]
        self.net.load_state_dict(state["net"])
        if "optimizer" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
                lr = float(self.cfg.get("lr", 3e-4))
                weight_decay = float(self.cfg.get("weight_decay", 0.0))
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                    group["weight_decay"] = weight_decay
            except Exception:
                pass

    def save(self, path):
        if not self.enabled:
            return
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        ck = torch.load(path, map_location=map_location or self.device,
                        weights_only=False)
        self.load_state_dict(ck)


def predictor_ckpt_path(agent_ckpt_path):
    root, ext = os.path.splitext(agent_ckpt_path)
    if not ext:
        ext = ".pt"
    return f"{root}_obs_predictor{ext}"


def build_observation_predictor(config, phase, obs_dim, device=None):
    cfg = config.get("observation_predictor", {})
    if not bool(cfg.get("enabled", False)):
        return None
    return OnlineObservationPredictor(
        config=config,
        phase=phase,
        obs_dim=obs_dim,
        action_dim=int(cfg.get("action_dim", 7)),
        device=device,
    )
