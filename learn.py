import numpy as np
import torch
import csv
import os
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from agent_old import WBAgent
from mujoco_env_new import CableRobotEnvWithObstacles
from nmpc_controller_new import NMPCTrajectoryTracker

from ipdb import set_trace as xxxx

nmpc_instance = None

# ===========================================================================
# 轻量日志封装：优先 wandb，其次 TensorBoard，两者都没有则仅输出到 CSV。
# 使用方式：在 train() 开头调用 Logger(...)，之后只用 logger.log(step, dict)。
# ===========================================================================
class Logger:
    def __init__(self, log_dir, project="cable_robot", run_name=None,
                 use_wandb=True, use_tb=True):
        self._wandb  = None
        self._writer = None
        self.log_dir = log_dir

        # ---- 尝试初始化 wandb ----
        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
                self._wandb.init(
                    project=project,
                    name=run_name or os.path.basename(log_dir),
                    dir=log_dir,
                    config={},
                    resume="allow",
                )
                print("[Logger] wandb 初始化成功。")
            except Exception as e:
                print(f"[Logger] wandb 不可用，跳过：{e}")
                self._wandb = None

        # ---- 尝试初始化 TensorBoard ----
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(log_dir, "tb")
                self._writer = SummaryWriter(log_dir=tb_dir)
                print(f"[Logger] TensorBoard 初始化成功，日志目录：{tb_dir}")
                print(f"         启动命令：tensorboard --logdir {tb_dir}")
            except Exception as e:
                print(f"[Logger] TensorBoard 不可用，跳过：{e}")
                self._writer = None

        if self._wandb is None and self._writer is None:
            print("[Logger] wandb 与 TensorBoard 均不可用，仅记录 CSV。")

    def update_config(self, cfg: dict):
        """向 wandb run 写入超参数"""
        if self._wandb is not None:
            self._wandb.config.update(cfg)

    def log(self, step: int, metrics: dict):
        """同时向 wandb 和 TensorBoard 写入一批指标。"""
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


def nmpc_wrapper(state_input, env_wp_idx=None):
    """
    专家控制器包装器：严格适配 23 维输入，输出 3 维动作 (ax, ay, az)
    """
    global nmpc_instance
    # 确保兼容 torch Tensor 或 numpy array
    if isinstance(state_input, torch.Tensor):
        state_input = state_input.cpu().numpy()
        
    is_batch = len(state_input.shape) > 1
    states = state_input if is_batch else [state_input]

    actions = []
    for s in states:
        # 直接调用更新后的 3D NMPC 控制器
        act = nmpc_instance.compute_action(s, target_yaw=0.0)
        actions.append(act)

    actions = np.array(actions, dtype=np.float32)
    return actions if is_batch else actions[0]


def train(log_dir):
    global nmpc_instance

    # 实例化 3D 环境，包含障碍物
    # 通过 config 字典覆盖 DEFAULT_CONFIG 中的参数
    custom_config = {
        "sim": {
            "render": False  # 在这里设置是否渲染
        }
    }
    env = CableRobotEnvWithObstacles(config=custom_config)
    nmpc_instance = NMPCTrajectoryTracker()

    MAX_ACTION = 0.5
    # 维度严格对齐环境的 23 维和控制器的 3 维输出
    STATE_DIM  = 31  # 10(base) + 9(obs) + 4(z_info)
    ACTION_DIM = 6    # ax, ay, az

    def env_aware_nmpc_wrapper(state_input):
        # 核心逻辑：只有在单步交互（非 batch）时，才强制让 NMPC 读取环境同步的路点索引
        is_batch = len(getattr(state_input, 'shape', [])) > 1
        current_wp = None if is_batch else getattr(env, 'current_wp_idx', None)
        return nmpc_wrapper(state_input, env_wp_idx=current_wp)

    agent = WBAgent(
        log_dir=log_dir,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        max_action=MAX_ACTION,
        mixed_q=True,
        base_boot=True,
        behavior_clone=True,
        base_controller_func=env_aware_nmpc_wrapper
    )

    agent.delta       = 5e-6
    agent.epsilon_min = 0.1

    WARMUP_EPISODES = 100
    EXPLORE_NOISE   = 0.05
    n_episodes      = 4000

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 初始化日志器
    # ------------------------------------------------------------------
    logger = Logger(
        log_dir=log_dir,
        project="cable_robot_wbagent",
        run_name=os.path.basename(log_dir),
        use_wandb=True,
        use_tb=True,
    )
    logger.update_config({
        "n_episodes":      n_episodes,
        "max_steps_per_ep": 200,
        "state_dim":       STATE_DIM,
        "action_dim":      ACTION_DIM,
        "max_action":      MAX_ACTION,
        "warmup_episodes": WARMUP_EPISODES,
        "explore_noise":   EXPLORE_NOISE,
        "delta":           agent.delta,
        "epsilon_min":     agent.epsilon_min,
        "gamma":           agent.gamma,
        "tau":             agent.tau,
        "batch_size":      agent.batch_size,
    })

    # CSV
    log_file = os.path.join(log_dir, 'log.csv')
    with open(log_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['episode', 'frames', 'return', 'Lc', 'La', 'Lbc', 'ratio', 'eps'])

    frames_total = 0
    loss_c = loss_a = loss_bc = 0.0

    SMOOTH_WIN  = 20
    reward_hist = []
    steps_hist  = []

    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        auto_refresh=False
    )

    with progress:
        task_id = progress.add_task('[red]Training...', total=n_episodes * 200)

        for episode in range(n_episodes):
            if episode < WARMUP_EPISODES:
                agent.delta   = 0.0
                agent.epsilon = 1.0
            elif episode == WARMUP_EPISODES:
                agent.delta   = 5e-6
                agent.epsilon = 1.0

            state = env.reset()
            # 直接删掉 nmpc_instance.reset_state_machine() 相关的代码
            # 获取当前目标位置（用于喂给新版 NMPC）
            target_pos = env.target_pos
            target_yaw = 0.0  # 或者从 info 里取
            if hasattr(env, "get_planned_path"):
                nmpc_instance.set_path(env.get_planned_path())

            episode_reward = 0.0
            step_count     = 0
            ratio_count    = 0
            episode_success = False

            while True:
                # ------------------------------------------------------------------
                # 1. 动态获取当前环境的 3D 目标航点 (与环境内部状态机完美对齐)
                # ------------------------------------------------------------------
                if env._planned_path is not None and not env.reached_final:
                    current_target_3d = env._planned_path[env.current_wp_idx]
                else:
                    current_target_3d = env.target_pos  # 最终目标

                # 替换旧版 env_aware_nmpc_wrapper，直接使用新版即时控制器
                base_action_val = nmpc_instance.compute_action(state, target_yaw=0.0)

                # ------------------------------------------------------------------
                # 2. Agent 决策与探索噪声注入
                # ------------------------------------------------------------------
                # Agent 根据当前 state 预测动作 (如果内部未启用专家机制，act_val可忽略)
                action, is_network, _ = agent.act(state)

                if not is_network:
                    action_exec = action.copy()
                    ratio_count += 1
                else:
                    # 探索噪声依然适配 3D (6D动作)
                    action_exec = action + np.random.normal(0, EXPLORE_NOISE, size=ACTION_DIM)

                # 动作裁剪与 NaN 异常保护 (若网络崩溃，用专家动作托底)
                action_exec = np.clip(action_exec, -MAX_ACTION, MAX_ACTION)
                if np.isnan(action_exec).any():
                    action_exec = base_action_val.copy()
                    ratio_count += 1

                # ------------------------------------------------------------------
                # 3. 环境执行 3D 动作 (完美对接 Gymnasium 5 元组返回)
                # ------------------------------------------------------------------
                # 修正点：env.step 返回 obs, reward, terminated, truncated, info
                next_state, reward, terminated, truncated, info = env.step(action_exec)
                
                # 合并终止条件与提取成功标志
                done = terminated or truncated
                success = info.get("is_success", False)
                
                if success:
                    episode_success = True

                # ------------------------------------------------------------------
                # 4. 计算 Next State 的专家动作 (用于入池)
                # ------------------------------------------------------------------
                if not done:
                    # 环境步进后，航点 idx 可能已更新，需重新获取以保证严格同步
                    if env._planned_path is not None and not env.reached_final:
                        next_target_3d = env._planned_path[env.current_wp_idx]
                    else:
                        next_target_3d = env.target_pos
                        
                    next_base_action_val = nmpc_instance.compute_action(next_state, target_yaw=0.0)
                else:
                    next_base_action_val = np.zeros(ACTION_DIM, dtype=np.float32)

                # ------------------------------------------------------------------
                # 5. 存入回放池并触发训练
                # ------------------------------------------------------------------
                # 保持你原来的参数顺序：state, action, base_action, next_base_action, next_state, reward, done
                agent.remember(state, action_exec, base_action_val, next_base_action_val, next_state, reward, done)

                if agent.buffer.size > 1024:
                    loss_c, loss_a, loss_bc = agent.train(1)

                # ------------------------------------------------------------------
                # 6. 状态流转与统计更新
                # ------------------------------------------------------------------
                state          = next_state
                episode_reward += reward
                step_count     += 1
                frames_total   += 1

                progress.update(task_id, advance=1)

                # env.step 内部已经处理了超时逻辑并返回 terminated=True，直接判定 done 即可
                if done:
                    break
            # ----------------------------------------------------------------
            # Episode 指标汇总
            # ----------------------------------------------------------------
            ratio = ratio_count / max(step_count, 1)

            reward_hist.append(episode_reward)
            steps_hist.append(step_count)
            if len(reward_hist) > SMOOTH_WIN:
                reward_hist.pop(0)
                steps_hist.pop(0)
            avg_reward = float(np.mean(reward_hist))
            avg_steps  = float(np.mean(steps_hist))

            logger.log(episode, {
                "reward/episode":       episode_reward,
                "reward/avg20":         avg_reward,
                "steps/episode":        step_count,
                "steps/avg20":          avg_steps,
                "loss/critic":          loss_c,
                "loss/actor":           loss_a,
                "loss/behavior_clone":  loss_bc,
                "explore/epsilon":      agent.epsilon,
                "explore/ratio":        ratio,
                "frames_total":         frames_total,
            })

            progress.refresh()
            
            status_mark = "✅ 成功" if episode_success else "❌ 失败"
            print(f"Ep: {episode:4d} | {status_mark} | R: {episode_reward:7.2f} (avg20: {avg_reward:7.2f}) "
                  f"| Steps: {step_count:3d} | Ratio: {ratio:.2f} | Eps: {agent.epsilon:.4f}")

            with open(log_file, "a+", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([episode, frames_total, episode_reward,
                                 loss_c, loss_a, loss_bc, ratio, agent.epsilon])

            if episode > 0 and episode % 50 == 0:
                torch.save(agent.actor,  os.path.join(log_dir, f'actor_ep{episode}.pt'))
                torch.save(agent.actor,  os.path.join(log_dir, 'actor.pt'))
                torch.save(agent.critic, os.path.join(log_dir, 'critic.pt'))

    logger.close()


if __name__ == '__main__':
    log_dir = 'saves/nmpc_experiment'
    train(log_dir)