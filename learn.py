import numpy as np
import torch
import csv
import os
import sys
import contextlib
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn

from agent import WBAgent
from mujoco_env import CableRobotEnv
from nmpc_controller import NMPCController

@contextlib.contextmanager
def silence_stderr():
    fd = sys.stderr.fileno()
    def_handler = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, fd)
    try:
        yield
    finally:
        os.dup2(def_handler, fd)
        os.close(devnull)
        os.close(def_handler)

nmpc_instance = None
env_target_pos = None 

def nmpc_wrapper(state_input):
    global nmpc_instance, env_target_pos
    is_batch = len(state_input.shape) > 1
    states = state_input if is_batch else [state_input]

    actions =[]
    for s in states:
        nmpc_state = s[:8]
        with silence_stderr():
            act = nmpc_instance.get_action(nmpc_state, env_target_pos)
        if np.isnan(act).any() or np.isinf(act).any():
            act = np.zeros(2)
        actions.append(act)
    
    actions = np.array(actions, dtype=np.float32)
    return actions if is_batch else actions[0]

def evaluate_policy(env, agent, n_eval_episodes=100):
    success_count = 0
    for _ in range(n_eval_episodes):
        with silence_stderr():
            state = env.reset()
            
        done = False
        step = 0
        while not done and step < 200:
            action, _, _ = agent.act(state, test=True)
            action = np.clip(action, -agent.max_action, agent.max_action)
            
            with silence_stderr():
                state, reward, done, info = env.step(action)
                
            step += 1
            if info.get("success", False):
                success_count += 1
                done = True
            if info.get("physics_crash", False):
                done = True
                
    return success_count / n_eval_episodes

def train(log_dir, seed=0, enable_init_rand=True, enable_process_noise=True):
    global nmpc_instance, env_target_pos
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 【保持地狱难度】0.35m 初始范围，无过程噪声
    HUGE_INIT_RANGE = 0.25
    
    with silence_stderr():
        env = CableRobotEnv(
            render=False, latency_steps=1, force_noise_level=0.0, control_freq_hz=10,
            init_velocity_scale=0.05, init_position_range=HUGE_INIT_RANGE,
            enable_init_randomization=enable_init_rand,
            enable_process_noise=False
        )
        eval_env = CableRobotEnv(
            render=False, latency_steps=1, force_noise_level=0.0, control_freq_hz=10,
            init_velocity_scale=0.05, init_position_range=HUGE_INIT_RANGE,
            enable_init_randomization=enable_init_rand,
            enable_process_noise=False
        )
    
    nmpc_instance = NMPCController()
    MAX_ACTION = 0.5
    
    agent = WBAgent(
        log_dir=log_dir, state_dim=10, action_dim=2, max_action=MAX_ACTION,
        mixed_q=True, base_boot=True, behavior_clone=True, base_controller_func=nmpc_wrapper
    )

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'log.csv')
    
    # 【新增】记录 pred_loss
    with open(log_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['episode', 'frames', 'train_return', 'train_success', 'test_success_rate', 'ratio', 'epsilon', 'init_dist', 'avg_swing', 'pred_loss'])

    max_frames_total = int(2e5)
    frames_total = 0
    episode = 0
    best_test_sr = -1.0
    
    progress = Progress(
        TextColumn("[bold magenta]Training OURS (Predictor)..."),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        auto_refresh=True 
    )
    
    with progress:
        task_id = progress.add_task("Training...", total=max_frames_total)
        
        while frames_total < max_frames_total:
            with silence_stderr():
                state = env.reset()
            nmpc_instance.reset() 
            env_target_pos = env.target_pos 
            
            episode_reward = 0
            step_count = 0
            ratio_count = 0
            ep_pred_loss = 0 # 记录本回合的平均预测误差
            
            while True:
                action, is_network, base_action_val = agent.act(state)
                
                if not is_network:
                    ratio_count += 1
                    action_exec = action 
                else:
                    action_exec = action + np.random.normal(0, 0.05, size=2)
                
                action_exec = np.clip(action_exec, -MAX_ACTION, MAX_ACTION)
                
                with silence_stderr():
                    next_state, reward, done, info = env.step(action_exec)
                
                if info.get("physics_crash", False) or np.isnan(next_state).any() or np.isinf(next_state).any():
                    break 
                
                agent.remember(state, action_exec, base_action_val, next_state, reward, done)
                
                state = next_state
                episode_reward += reward
                step_count += 1
                frames_total += 1
                
                # 接收预测器的 Loss
                _, _, _, l_pred = agent.train(2)
                ep_pred_loss += l_pred
                
                progress.update(task_id, completed=frames_total)
                
                if done or step_count >= 150:
                    break
            
            episode += 1
            is_success = 1 if episode_reward > 0.5 else 0 
            current_ratio = ratio_count / max(1, step_count)
            avg_pred_loss = ep_pred_loss / max(1, step_count)
            
            init_dist = env.current_init_dist
            avg_swing = env.get_avg_swing()
            test_sr = np.nan
            
            if episode % 30 == 0:
                test_sr = evaluate_policy(eval_env, agent, n_eval_episodes=100)
                if test_sr > best_test_sr:
                    best_test_sr = test_sr
                    # 【请替换为新代码】：把双核大脑一起打包保存！
                    torch.save({
                        'actor': agent.actor,
                        'predictor': agent.predictor
                    }, os.path.join(log_dir, 'model_best.pt'))
                    progress.console.print(f"[bold green]Ep {episode:4d} | Frames {frames_total:6d} | New Best SR: {test_sr*100:.1f}% | PredLoss: {avg_pred_loss:.4f}[/bold green]")
                else:
                    progress.console.print(f"[bold yellow]Ep {episode:4d} | Frames {frames_total:6d} | Test SR: {test_sr*100:.1f}% (Best: {best_test_sr*100:.1f}%) | PredLoss: {avg_pred_loss:.4f}[/bold yellow]")
            
            with open(log_file, "a+", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([episode, frames_total, episode_reward, is_success, test_sr, current_ratio, agent.epsilon, init_dist, avg_swing, avg_pred_loss])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--enable_init_rand', type=int, default=1)
    parser.add_argument('--enable_process_noise', type=int, default=1)
    args = parser.parse_args()
    
    init_str = 'initRand' if args.enable_init_rand else 'noInitRand'
    noise_str = 'procNoise' if args.enable_process_noise else 'noProcNoise'
    
    # 【核心修改】保存到 ours_experiment 文件夹，与 baseline 区分开！
    log_dir = f'saves/ours_experiment/{init_str}_{noise_str}/seed_{args.seed}'
    
    train(log_dir, seed=args.seed, enable_init_rand=bool(args.enable_init_rand), enable_process_noise=bool(args.enable_process_noise))