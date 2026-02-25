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

# ==========================================
# 【黑魔法】OS 级别屏蔽 C++ stderr 输出
#  用于拦截 MuJoCo 底层的非致命警告，保护进度条界面不被破坏
# ==========================================
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

    actions = []
    for s in states:
        nmpc_state = s[:8]
        # 屏蔽 NMPC 内部可能触发的警告
        with silence_stderr():
            act = nmpc_instance.get_action(nmpc_state, env_target_pos)
            
        # 安全防护
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
                # 注意这里适配了新的环境返回值 (obs, reward, done, info)
                state, reward, done, info = env.step(action)
                
            step += 1
            if info.get("success", False):
                success_count += 1
                done = True
                
            # 测试时如果发生崩溃也直接结束当前测试回合
            if info.get("physics_crash", False):
                done = True
                
    return success_count / n_eval_episodes

def train(log_dir, seed=0, enable_init_rand=True, enable_process_noise=True):
    global nmpc_instance, env_target_pos
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    with silence_stderr():
        env = CableRobotEnv(
            render=False, latency_steps=1, force_noise_level=0.08, control_freq_hz=10,
            init_velocity_scale=0.08, init_position_range=0.06,
            enable_init_randomization=enable_init_rand,
            enable_process_noise=enable_process_noise
        )
        eval_env = CableRobotEnv(
            render=False, latency_steps=1, force_noise_level=0.08, control_freq_hz=10,
            init_velocity_scale=0.08, init_position_range=0.06,
            enable_init_randomization=enable_init_rand,
            enable_process_noise=enable_process_noise
        )
    
    nmpc_instance = NMPCController()
    MAX_ACTION = 0.5
    
    agent = WBAgent(
        log_dir=log_dir, state_dim=10, action_dim=2, max_action=MAX_ACTION,
        mixed_q=True, base_boot=True, behavior_clone=True, base_controller_func=nmpc_wrapper
    )

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'log.csv')
    
    with open(log_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['episode', 'frames', 'train_return', 'train_success', 'test_success_rate', 'ratio', 'epsilon', 'init_dist', 'avg_swing'])

    max_frames_total = int(2e5)
    frames_total = 0
    episode = 0
    best_test_sr = -1.0
    
    # 【华丽且高效的 rich 进度条】
    # 开启 auto_refresh=True，交由 rich 的后台线程管理刷新，既不闪烁也不拖慢训练速度
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        auto_refresh=True 
    )
    
    with progress:
        task_id = progress.add_task("Training RL Agent...", total=max_frames_total)
        
        while frames_total < max_frames_total:
            with silence_stderr():
                state = env.reset()
            nmpc_instance.reset() 
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
                
                # 核心物理步进，屏蔽 C++ 报错，解包 info 字典
                with silence_stderr():
                    next_state, reward, done, info = env.step(action_exec)
                
                # ====================================================
                # 【防污染核心逻辑】：一旦发生物理崩溃，直接丢弃该步并结束回合
                # ====================================================
                if info.get("physics_crash", False) or np.isnan(next_state).any() or np.isinf(next_state).any():
                    break # 跳过 agent.remember，直接结束当前失败的 episode
                
                # 正常状态存入经验池
                agent.remember(state, action_exec, base_action_val, next_state, reward, done)
                
                state = next_state
                episode_reward += reward
                step_count += 1
                frames_total += 1
                
                agent.train(2)
                
                # 仅更新进度数值，不再手动调用 refresh()，避免性能瓶颈
                progress.update(task_id, completed=frames_total)
                
                # 环境 done 或者达到步数上限则结束回合
                if done or step_count >= 150:
                    break
            
            episode += 1
            # 判断训练回合是否成功（可根据你的业务逻辑调整）
            is_success = 1 if episode_reward > 0.5 else 0 
            current_ratio = ratio_count / max(1, step_count)
            
            init_dist = env.current_init_dist
            avg_swing = env.get_avg_swing()
            test_sr = np.nan
            
            # 每 30 回合进行测试并打印
            if episode % 30 == 0:
                test_sr = evaluate_policy(eval_env, agent, n_eval_episodes=100)
                if test_sr > best_test_sr:
                    best_test_sr = test_sr
                    torch.save(agent.actor, os.path.join(log_dir, 'actor_best.pt'))
                    progress.console.print(f"[bold green]Ep {episode:4d} | Frames {frames_total:6d} | New Best SR: {test_sr*100:.1f}% | Saved![/bold green]")
                else:
                    progress.console.print(f"[bold yellow]Ep {episode:4d} | Frames {frames_total:6d} | Test SR: {test_sr*100:.1f}% (Best: {best_test_sr*100:.1f}%)[/bold yellow]")
            
            with open(log_file, "a+", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([episode, frames_total, episode_reward, is_success, test_sr, current_ratio, agent.epsilon, init_dist, avg_swing])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--enable_init_rand', type=int, default=1)
    parser.add_argument('--enable_process_noise', type=int, default=1)
    args = parser.parse_args()
    
    init_str = 'initRand' if args.enable_init_rand else 'noInitRand'
    noise_str = 'procNoise' if args.enable_process_noise else 'noProcNoise'
    log_dir = f'saves/nmpc_experiment/{init_str}_{noise_str}/seed_{args.seed}'
    
    train(log_dir, seed=args.seed, enable_init_rand=bool(args.enable_init_rand), enable_process_noise=bool(args.enable_process_noise))