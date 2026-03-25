import numpy as np
import torch
import csv
import os
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from agent import WBAgent
from mujoco_env import CableRobotEnvWithObstacles
from nmpc_controller import NMPCTrajectoryTracker

nmpc_instance = None

def nmpc_wrapper(state_input):
    """
    专家控制器包装器：适配 23 维输入，输出 3 维动作 (ax, ay, az)
    """
    global nmpc_instance
    is_batch = len(state_input.shape) > 1
    states = state_input if is_batch else [state_input]

    actions = []
    for s in states:
        nmpc_s  = s[:8]
        z, vz   = s[-4], s[-3]
        act     = nmpc_instance.get_tracking_action(nmpc_s, z, vz)
        actions.append(act)

    actions = np.array(actions, dtype=np.float32)
    return actions if is_batch else actions[0]

def train(log_dir):
    global nmpc_instance

    env           = CableRobotEnvWithObstacles(render=False, n_obstacles=3)
    nmpc_instance = NMPCTrajectoryTracker()

    MAX_ACTION = 0.5
    STATE_DIM  = 23   # 10(base) + 9(obs) + 4(z_info)
    ACTION_DIM = 3    # ax, ay, az

    agent = WBAgent(
        log_dir=log_dir,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        max_action=MAX_ACTION,
        mixed_q=True,
        base_boot=True,
        behavior_clone=True,
        base_controller_func=nmpc_wrapper
    )

    # ------------------------------------------------------------------
    # Epsilon 退火完全交由 agent.act() 内部的线性衰减管理（每步 -delta）。
    # learn.py 不再做任何 epsilon 操作，消除双重衰减。
    #
    # 若需调整退火速率，修改此处即可（初始化后覆盖默认值）：
    #   agent.delta       = 5e-6   # 每步衰减量：(1.0-0.1)/delta = 退火总步数
    #   agent.epsilon_min = 0.1
    # ------------------------------------------------------------------
    agent.delta       = 5e-6
    agent.epsilon_min = 0.1

    # 热身期：前 N 集强制 epsilon=1.0，填充高质量专家经验
    # 实现方式：暂时把 delta 置 0，热身结束后恢复
    WARMUP_EPISODES = 50
    EXPLORE_NOISE   = 0.05   # 网络动作探索噪声标准差

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'log.csv')

    with open(log_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['episode', 'frames', 'return', 'Lc', 'La', 'Lbc', 'ratio', 'eps'])

    n_episodes   = 2000
    frames_total = 0
    loss_c = loss_a = loss_bc = 0.0

    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        auto_refresh=False
    )

    with progress:
        task_id = progress.add_task('[red]Training...', total=n_episodes * 150)

        for episode in range(n_episodes):

            # 热身期：冻结 epsilon（delta=0），保证 ratio≈1.0 稳定填充 buffer
            # 热身结束后恢复 delta，让 epsilon 从当前值开始自然线性退火
            if episode < WARMUP_EPISODES:
                agent.delta = 0.0          # 冻结，不衰减
                agent.epsilon = 1.0        # 每集重置（保证热身期全程为 1.0）
            elif episode == WARMUP_EPISODES:
                agent.delta   = 5e-6       # 恢复正常衰减速率
                agent.epsilon = 1.0        # 从 1.0 重新开始退火

            state = env.reset()
            nmpc_instance.reset_state_machine()
            if hasattr(env, "get_planned_path"):
                nmpc_instance.set_trajectory(env.get_planned_path())

            episode_reward = 0
            step_count     = 0
            ratio_count    = 0

            while True:
                # act() 内部完成：专家/网络选择 + 线性衰减 epsilon
                action, is_network, base_action_val = agent.act(state)

                if not is_network:
                    # 专家动作：不加噪声，保持示范准确性
                    action_exec = action.copy()
                    ratio_count += 1
                else:
                    # 网络动作：加小幅探索噪声
                    action_exec = action + np.random.normal(0, EXPLORE_NOISE, size=ACTION_DIM)

                # 安全 Clip & 防 NaN
                action_exec = np.clip(action_exec, -MAX_ACTION, MAX_ACTION)
                if np.isnan(action_exec).any():
                    action_exec = base_action_val.copy()
                    ratio_count += 1   # 退回专家同样计入 ratio

                # 环境步进
                next_state, reward, done, success = env.step(action_exec)

                # 获取下一步专家动作（用于 base_boot 的 Critic 目标计算）
                next_base_action_val = nmpc_wrapper(next_state) if not done \
                                       else np.zeros(ACTION_DIM, dtype=np.float32)

                # 存入 Buffer
                agent.remember(state, action_exec, base_action_val,
                               next_base_action_val, next_state, reward, done)

                # 训练（Buffer 足够后才开始）
                if agent.buffer.size > 1024:
                    loss_c, loss_a, loss_bc = agent.train(1)

                state          = next_state
                episode_reward += reward
                step_count     += 1
                frames_total   += 1

                progress.update(task_id, advance=1)

                if done or step_count >= 150:
                    break

            ratio = ratio_count / max(step_count, 1)
            progress.refresh()
            print(f"Ep: {episode:3d} | R: {episode_reward:6.2f} | Steps: {step_count:3d} "
                  f"| Ratio: {ratio:.2f} | Eps: {agent.epsilon:.4f}")

            with open(log_file, "a+", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([episode, frames_total, episode_reward,
                                 loss_c, loss_a, loss_bc, ratio, agent.epsilon])

            if episode > 0 and episode % 50 == 0:
                torch.save(agent.actor,  os.path.join(log_dir, f'actor_ep{episode}.pt'))
                torch.save(agent.actor,  os.path.join(log_dir, 'actor.pt'))
                torch.save(agent.critic, os.path.join(log_dir, 'critic.pt'))

if __name__ == '__main__':
    log_dir = 'saves/nmpc_experiment'
    train(log_dir)