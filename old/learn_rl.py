import numpy as np
import torch
import csv
import os
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn
 
from agent import PureRLAgent
from old.mujoco_env import CableRobotEnvWithObstacles
 
 
# ===========================================================================
# 日志封装（与 learn.py 完全相同，复制粘贴无改动）
# ===========================================================================
 
class Logger:
    def __init__(self, log_dir, project="cable_robot", run_name=None,
                 use_wandb=True, use_tb=True):
        self._wandb  = None
        self._writer = None
        self.log_dir = log_dir
 
        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
                self._wandb.init(
                    project=project,
                    name=run_name or os.path.basename(log_dir),
                    dir=log_dir, config={}, resume="allow",
                )
                print("[Logger] wandb 初始化成功。")
            except Exception as e:
                print(f"[Logger] wandb 不可用，跳过：{e}")
                self._wandb = None
 
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(log_dir, "tb")
                self._writer = SummaryWriter(log_dir=tb_dir)
                print(f"[Logger] TensorBoard 初始化成功，日志目录：{tb_dir}")
            except Exception as e:
                print(f"[Logger] TensorBoard 不可用，跳过：{e}")
                self._writer = None
 
        if self._wandb is None and self._writer is None:
            print("[Logger] wandb 与 TensorBoard 均不可用，仅记录 CSV。")
 
    def update_config(self, cfg: dict):
        if self._wandb is not None:
            self._wandb.config.update(cfg)
 
    def log(self, step: int, metrics: dict):
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)
        if self._writer is not None:
            for k, v in metrics.items():
                self._writer.add_scalar(k, v, global_step=step)
 
    def close(self):
        if self._wandb is not None:
            self._wandb.finish()
        if self._writer is not None:
            self._writer.close()
 
 
# ===========================================================================
# 纯 RL 训练主函数
# ===========================================================================
 
def train_rl(log_dir):
    """
    纯 TD3 训练，无专家引导。
 
    探索策略说明：
      - Warmup 阶段（前 WARMUP_STEPS 步）：均匀随机动作 U(-max_action, max_action)
        目的是快速、无偏地填充 replay buffer，避免早期 Actor 未收敛时的系统性偏差。
      - 正式训练阶段：Actor 输出 + 高斯噪声 N(0, σ²)，σ 随训练线性退火。
        这是标准 TD3 的探索方式，兼顾利用（Actor 输出）和探索（随机扰动）。
 
    为什么不用 epsilon-greedy：
      纯RL没有专家可以切换，epsilon-greedy 退化为"随机动作 vs Actor"的切换。
      连续动作空间中，均匀随机动作（大噪声）集中在早期 Warmup，正式阶段用
      小方差高斯噪声更有利于在 Actor 已经有一定质量后做精细探索。
    """
 
    # ------------------------------------------------------------------
    # 1. 环境初始化（参数与 learn.py 完全一致，保证公平对比）
    # ------------------------------------------------------------------
    env = CableRobotEnvWithObstacles(
        render=False,
        n_obstacles=3,
        latency_steps=1,
        force_noise_level=0.08,
        control_freq_hz=10,
        init_velocity_scale=0.08,
        init_position_range=0.00,
        obstacle_radius_range=(0.01, 0.02),
        path_width=0.12,
        default_start_xy=[0.2, 0.2],
        default_target_xy=[0.5, 0.5],
        payload_radius=0.10,
        planning_margin=0.10,
        planning_grid_res=0.02
    )
 
    MAX_ACTION = 0.5
    STATE_DIM  = 23   # 与 learn.py 相同：10(base) + 9(obs) + 4(z_info)
    ACTION_DIM = 3    # ax, ay, az
 
    # ------------------------------------------------------------------
    # 2. Agent 初始化
    # ------------------------------------------------------------------
    agent = PureRLAgent(
        log_dir=log_dir,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        max_action=MAX_ACTION,
    )
 
    # ------------------------------------------------------------------
    # 3. 超参数
    # ------------------------------------------------------------------
    n_episodes   = 20000
    MAX_EP_STEPS = 150
 
    # Warmup：前 WARMUP_STEPS 环境步使用均匀随机动作填充 buffer
    # 不用 episode 数而用 step 数，是因为纯RL早期每局可能很短（频繁 done）
    WARMUP_STEPS = 15000
 
    # 高斯探索噪声：σ 从 EXPLORE_NOISE_START 线性退火到 EXPLORE_NOISE_END
    # 退火总步数 = n_episodes * MAX_EP_STEPS（与 epsilon 退火逻辑对等）
    EXPLORE_NOISE_START = 0.1   # 初始较大，鼓励广泛探索
    EXPLORE_NOISE_END   = 0.02  # 最终较小，收敛后精细利用
    total_train_steps   = n_episodes * MAX_EP_STEPS
    noise_decay_rate    = (EXPLORE_NOISE_START - EXPLORE_NOISE_END) / total_train_steps
 
    # ------------------------------------------------------------------
    # 4. 日志
    # ------------------------------------------------------------------
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
 
    logger = Logger(
        log_dir=log_dir,
        project="cable_robot_pure_rl",
        run_name=os.path.basename(log_dir),
        use_wandb=True,
        use_tb=True,
    )
    logger.update_config({
        "mode":               "pure_rl",
        "n_episodes":         n_episodes,
        "max_steps_per_ep":   MAX_EP_STEPS,
        "state_dim":          STATE_DIM,
        "action_dim":         ACTION_DIM,
        "max_action":         MAX_ACTION,
        "warmup_steps":       WARMUP_STEPS,
        "explore_noise_start": EXPLORE_NOISE_START,
        "explore_noise_end":   EXPLORE_NOISE_END,
        "gamma":              agent.gamma,
        "tau":                agent.tau,
        "batch_size":         agent.batch_size,
        "policy_noise":       agent.policy_noise,
        "policy_freq":        agent.policy_freq,
        "base_boot":          agent.base_boot,
        "behavior_clone":     agent.behavior_clone,
    })
 
    log_file = os.path.join(log_dir, 'log.csv')
    with open(log_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['episode', 'frames', 'return', 'Lc', 'La', 'Lbc',
                         'explore_noise', 'success'])
 
    # ------------------------------------------------------------------
    # 5. 训练主循环
    # ------------------------------------------------------------------
    frames_total  = 0
    loss_c = loss_a = loss_bc = 0.0
 
    SMOOTH_WIN  = 20
    reward_hist = []
    steps_hist  = []
 
    # 当前探索噪声（随 frames_total 退火）
    current_noise = EXPLORE_NOISE_START
 
    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        auto_refresh=False
    )
 
    with progress:
        task_id = progress.add_task('[cyan]Pure RL Training...', total=n_episodes * MAX_EP_STEPS)
 
        for episode in range(n_episodes):
 
            state = env.reset()
            # 纯RL不需要 NMPC 路径，env 内部 planned_path 存在不影响 obs 输出
            # 但我们不给 agent 提供路径信息，确保 agent 从零学习
 
            episode_reward  = 0.0
            step_count      = 0
            episode_success = False
 
            while True:
                # --------------------------------------------------------
                # Warmup：均匀随机动作，不走 Actor
                # --------------------------------------------------------
                if frames_total < WARMUP_STEPS:
                    action_exec = np.random.uniform(
                        -MAX_ACTION, MAX_ACTION, size=ACTION_DIM
                    ).astype(np.float32)
                    # act() 返回值仅用于获取 base_action 占位
                    _, _, zero_base = agent.act(state)
                    base_action_val    = zero_base
                    next_base_action_val = zero_base
 
                else:
                    # --------------------------------------------------------
                    # 正式训练：Actor 输出 + 高斯噪声探索
                    # --------------------------------------------------------
                    actor_action, _, zero_base = agent.act(state)
 
                    # 退火高斯噪声（在外部注入，agent.act 内部不含噪声）
                    current_noise = max(
                        EXPLORE_NOISE_END,
                        EXPLORE_NOISE_START - noise_decay_rate * frames_total
                    )
                    noise = np.random.normal(0, current_noise, size=ACTION_DIM).astype(np.float32)
                    action_exec = np.clip(actor_action + noise, -MAX_ACTION, MAX_ACTION)
 
                    base_action_val      = zero_base   # 零占位
                    next_base_action_val = zero_base   # 零占位
 
                # --------------------------------------------------------
                # 环境交互
                # --------------------------------------------------------
                if np.isnan(action_exec).any():
                    action_exec = np.zeros(ACTION_DIM, dtype=np.float32)
 
                next_state, reward, done, success = env.step(action_exec)
 
                if success:
                    episode_success = True
 
                # 存入 buffer（base_action 全为零，train() 不会使用）
                agent.remember(
                    state, action_exec,
                    base_action_val, next_base_action_val,
                    next_state, reward, done
                )
 
                # --------------------------------------------------------
                # 训练（Warmup 结束后才开始）
                # --------------------------------------------------------
                if frames_total >= WARMUP_STEPS and agent.buffer.size > agent.batch_size:
                    loss_c, loss_a, loss_bc = agent.train(1)
 
                state          = next_state
                episode_reward += reward
                step_count     += 1
                frames_total   += 1
 
                progress.update(task_id, advance=1)
 
                if done or step_count >= MAX_EP_STEPS:
                    break
 
            # --------------------------------------------------------------
            # Episode 指标汇总
            # --------------------------------------------------------------
            reward_hist.append(episode_reward)
            steps_hist.append(step_count)
            if len(reward_hist) > SMOOTH_WIN:
                reward_hist.pop(0)
                steps_hist.pop(0)
            avg_reward = float(np.mean(reward_hist))
            avg_steps  = float(np.mean(steps_hist))
 
            logger.log(episode, {
                "reward/episode":      episode_reward,
                "reward/avg20":        avg_reward,
                "steps/episode":       step_count,
                "steps/avg20":         avg_steps,
                "loss/critic":         loss_c,
                "loss/actor":          loss_a,
                "loss/behavior_clone": loss_bc,   # 纯RL模式下恒为 0
                "explore/noise_sigma": current_noise,
                "frames_total":        frames_total,
            })
 
            progress.refresh()
 
            warmup_mark = " [warmup]" if frames_total <= WARMUP_STEPS else ""
            status_mark = "✅ 成功" if episode_success else "❌ 失败"
            print(f"Ep: {episode:4d} | {status_mark}{warmup_mark} | "
                  f"R: {episode_reward:7.2f} (avg20: {avg_reward:7.2f}) "
                  f"| Steps: {step_count:3d} | σ: {current_noise:.4f}")
 
            with open(log_file, "a+", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([episode, frames_total, episode_reward,
                                 loss_c, loss_a, loss_bc,
                                 current_noise, int(episode_success)])
 
            if episode > 0 and episode % 50 == 0:
                torch.save(agent.actor, os.path.join(log_dir, f'actor_ep{episode}.pt'))
                torch.save(agent.actor, os.path.join(log_dir, 'actor.pt'))
                torch.save(agent.critic, os.path.join(log_dir, 'critic.pt'))
 
    logger.close()
 
 
# ===========================================================================
# 入口
# ===========================================================================
 
if __name__ == '__main__':
    log_dir = 'saves/pure_rl_experiment'
    train_rl(log_dir)