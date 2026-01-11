import numpy as np
import torch
import csv
import os
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from agent import WBAgent
from mujoco_env import CableRobotEnv
from nmpc_controller import NMPCController

nmpc_instance = None
env_target_pos = None 

def nmpc_wrapper(state_input):
    global nmpc_instance, env_target_pos
    is_batch = len(state_input.shape) > 1
    if not is_batch:
        states = [state_input]
    else:
        states = state_input

    actions = []
    for s in states:
        nmpc_state = s[:8]
        act = nmpc_instance.get_action(nmpc_state, env_target_pos)
        actions.append(act)
    
    actions = np.array(actions, dtype=np.float32)
    if not is_batch:
        return actions[0]
    return actions

def train(log_dir):
    global nmpc_instance, env_target_pos
    
    env = CableRobotEnv(render=False) 
    nmpc_instance = NMPCController()
    
    # 明确最大动作范围
    MAX_ACTION = 0.5
    
    agent = WBAgent(
        log_dir=log_dir,
        state_dim=10,
        action_dim=2,
        max_action=MAX_ACTION, # 传入 max_action
        mixed_q=True,
        base_boot=True,
        behavior_clone=True,
        base_controller_func=nmpc_wrapper
    )

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'log.csv')
    
    with open(log_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['episode', 'frames', 'return', 'Lc', 'La', 'Lbc', 'ratio', 'test'])

    n_episodes = 2000
    max_frames_total = 2e5
    frames_total = 0
    
    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        auto_refresh=False
    )
    
    with progress:
        task_id = progress.add_task('[red]Training...', total=max_frames_total)
        
        for episode in range(n_episodes):
            state = env.reset()
            env_target_pos = env.target_pos 
            
            episode_reward = 0
            step_count = 0
            ratio_count = 0
            
            while True:
                # act 返回: 动作, 是否网络, Base动作
                action, is_network, base_action_val = agent.act(state)
                
                if not is_network:
                    ratio_count += 1
                    # 【策略优化】如果是 Base 动作，不加噪声或加极小噪声，保持专家示范的准确性
                    action_exec = action 
                else:
                    # 如果是 Network 动作，加噪声探索
                    action_exec = action + np.random.normal(0, 0.05, size=2)
                
                # 统一 Clip
                action_exec = np.clip(action_exec, -MAX_ACTION, MAX_ACTION)
                
                next_state, reward, done, success = env.step(action_exec)
                
                # 存入 Buffer
                agent.remember(state, action_exec, base_action_val, next_state, reward, done)
                
                state = next_state
                episode_reward += reward
                step_count += 1
                frames_total += 1
                
                loss_c, loss_a, loss_bc = agent.train(1)                    
                progress.update(task_id, advance=1)
                progress.refresh()
                
                if done or step_count >= 150:
                    break
            
            ratio = ratio_count / step_count
            print(f"Ep: {episode} | R: {episode_reward:.2f} | Steps: {step_count} | Ratio: {ratio:.2f} | Eps: {agent.epsilon:.2f}")
            
            with open(log_file, "a+", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([episode, frames_total, episode_reward, loss_c, loss_a, loss_bc, ratio, 0])
            
            if episode % 50 == 0:
                torch.save(agent.actor, os.path.join(log_dir, 'actor.pt'))
                torch.save(agent.critic, os.path.join(log_dir, 'critic.pt'))

if __name__ == '__main__':
    log_dir = 'saves/nmpc_experiment'
    train(log_dir)