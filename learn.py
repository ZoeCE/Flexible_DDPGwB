import numpy as np
import torch
import csv
import os
import time
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

def evaluate_policy(env, agent, n_eval_episodes=100):
    success_count = 0
    for _ in range(n_eval_episodes):
        state = env.reset()
        done = False
        step = 0
        while not done and step < 200:
            action, _, _ = agent.act(state, test=True)
            action = np.clip(action, -agent.max_action, agent.max_action)
            state, reward, done, success = env.step(action)
            step += 1
            if success:
                success_count += 1
                done = True
    return success_count / n_eval_episodes

def train(log_dir, seed=0):
    global nmpc_instance, env_target_pos
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 【参数调整】降低扰动，确保物理稳定
    # 原因：0.15 m/s 初速 + 0.3 rad/s 角速度导致仿真频繁崩溃
    # 解决：降低到 0.08 m/s 和 0.15 rad/s，保持挑战性但更稳定
    env = CableRobotEnv(
        render=False,
        latency_steps=1,                    # 1步延迟 = 0.1s 传输延迟
        force_noise_level=0.08,             # 0.08N 随机外力（从 0.1 降低）
        control_freq_hz=10,                 # 10Hz 控制频率
        init_velocity_scale=0.08,           # 0.08m/s 初速（从 0.15 降低）
        init_position_range=0.06            # ±6cm 位置（从 ±8cm 降低）
    )
    
    eval_env = CableRobotEnv(
        render=False,
        latency_steps=1,
        force_noise_level=0.08,
        control_freq_hz=10,
        init_velocity_scale=0.08,
        init_position_range=0.06
    )
    
    
    nmpc_instance = NMPCController()
    
    MAX_ACTION = 0.5
    
    agent = WBAgent(
        log_dir=log_dir,
        state_dim=10,
        action_dim=2,
        max_action=MAX_ACTION,
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
        writer.writerow(['episode', 'frames', 'train_return', 'train_success', 'test_success_rate', 'ratio', 'epsilon'])

    n_episodes = 2000
    max_frames_total = 2e5
    frames_total = 0
    best_test_sr = -1.0
    train_success_count = 0
    
    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        auto_refresh=False
    )
    
    with progress:
        task_id = progress.add_task('[red]Training...', total=n_episodes)
        
        for episode in range(n_episodes):
            state = env.reset()
            env_target_pos = env.target_pos 
            
            episode_reward = 0
            step_count = 0
            ratio_count = 0
            
            while True:
                action, is_network, base_action_val = agent.act(state)
                
                if not is_network:
                    ratio_count += 1
                    action_exec = action 
                else:
                    action_exec = action + np.random.normal(0, 0.05, size=2)
                
                action_exec = np.clip(action_exec, -MAX_ACTION, MAX_ACTION)
                
                next_state, reward, done, success = env.step(action_exec)
                
                agent.remember(state, action_exec, base_action_val, next_state, reward, done)
                
                state = next_state
                episode_reward += reward
                step_count += 1
                frames_total += 1
                
                agent.train(2)
                
                if done or step_count >= 150:
                    break
            
            is_success = 1 if episode_reward > 0.5 else 0
            train_success_count += is_success
            current_ratio = ratio_count / step_count
            
            test_sr = np.nan
            if (episode + 1) % 30 == 0:
                print(f"\n[Evaluation] Episode {episode+1}: Running 100 test episodes...")
                test_sr = evaluate_policy(eval_env, agent, n_eval_episodes=100)
                
                if test_sr > best_test_sr:
                    best_test_sr = test_sr
                    torch.save(agent.actor, os.path.join(log_dir, 'actor_best.pt'))
                    print(f"  >>> New Best Model Saved! SR: {test_sr*100:.1f}%")
                else:
                    print(f"  --- Current SR: {test_sr*100:.1f}% (Best: {best_test_sr*100:.1f}%)")
            
            with open(log_file, "a+", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([episode, frames_total, episode_reward, is_success, test_sr, current_ratio, agent.epsilon])
            
            progress.update(task_id, advance=1)
            progress.refresh()
            
            if (episode + 1) % 10 == 0:
                print(f"Ep: {episode+1} | Train_R: {episode_reward:.2f} | Ratio: {current_ratio:.2f} | Eps: {agent.epsilon:.2f}")

            if frames_total >= max_frames_total:
                break

    print("\n" + "="*40)
    print("Training Finished. Starting Final Evaluation (1000 Episodes)...")
    
    best_model_path = os.path.join(log_dir, 'actor_best.pt')
    if os.path.exists(best_model_path):
        try:
            # 【修复】正确处理 device
            device_str = f"cuda:{agent.device}" if torch.cuda.is_available() else "cpu"
            best_actor = torch.load(best_model_path, map_location=device_str, weights_only=False)
            
            original_actor = agent.actor
            agent.actor = best_actor
            
            final_sr = evaluate_policy(eval_env, agent, n_eval_episodes=1000)
            print(f"Final Test Performance (Table I): {final_sr*100:.2f}%")
            
            agent.actor = original_actor
        except Exception as e:
            print(f"Error loading best model: {e}")
    else:
        print("No best model found.")
    print("="*40 + "\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    args = parser.parse_args()
    
    log_dir = f'saves/nmpc_experiment/seed_{args.seed}'
    train(log_dir, seed=args.seed)