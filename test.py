import torch
import numpy as np
import argparse
import time
import os
import sys

# 引入环境和控制器
from mujoco_env import CableRobotEnv
from nmpc_controller import NMPCController
# 引入 Agent 网络定义 (必须，否则 torch.load 报错)
from agent import FastActor 

def get_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")

def run_test(mode, log_dir, n_episodes, render, device_id=0):
    """
    统一的测试主循环
    """
    # 1. 初始化环境
    env = CableRobotEnv(render=render)
    
    # 2. 初始化策略 (Actor 或 Base)
    actor_model = None
    nmpc_controller = None
    
    if mode == 'actor':
        print(f"Loading Actor model from {log_dir}/actor.pt ...")
        device = get_device(device_id)
        model_path = os.path.join(log_dir, 'actor.pt')
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return
        try:
            # weights_only=False 解决 PyTorch 2.6+ 兼容性
            actor_model = torch.load(model_path, map_location=device, weights_only=False)
            actor_model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    elif mode == 'base':
        print("Initializing NMPC Controller (Base)...")
        nmpc_controller = NMPCController()
    
    # 3. 开始测试循环
    success_count = 0
    total_steps_success = 0
    
    print("*******************************************")
    print(f"Start Testing [{mode.upper()}] for {n_episodes} episodes...")
    print(f"Render: {'ON' if render else 'OFF'}")
    print("*******************************************")
    
    start_time = time.time()
    
    for i in range(n_episodes):
        obs = env.reset()
        target_pos = env.target_pos # 仅 Base 需要用到绝对坐标
        step = 0
        episode_reward = 0
        
        while True:
            # --- 策略决策 ---
            if mode == 'actor':
                # RL Agent: 输入 State -> 输出 2D Action
                s_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = actor_model(s_tensor).cpu().numpy()[0]
            else:
                # NMPC Base: 输入 State + Target -> 输出 2D Action
                nmpc_state = obs[:8]
                action = nmpc_controller.get_action(nmpc_state, target_pos)
            
            # --- 环境交互 ---
            # 无论是 Actor 还是 Base，都发送 2D 动作
            # 环境会自动判断是否满足下降条件
            next_obs, reward, done, success = env.step(action)
            
            obs = next_obs
            episode_reward += reward
            step += 1
            
            # 渲染延时，方便肉眼观察
            if render:
                time.sleep(0.02)
            
            # --- 结束判定 ---
            if done or step >= 200:
                # 只有在渲染模式下才打印每一局的详情，避免刷屏
                if render:
                    print(f"Ep {i+1}: Steps={step}, R={reward:.2f}, Success={success}")
                
                if success: 
                    success_count += 1
                    total_steps_success += step
                break
        
        # 进度条 (非渲染模式下显示)
        if not render and (i+1) % 10 == 0:
            print(f"Progress: {i+1}/{n_episodes} | Current SR: {success_count/(i+1)*100:.1f}%")

    end_time = time.time()
    avg_steps = total_steps_success / success_count if success_count > 0 else 0
    
    print("\n" + "="*30)
    print(f"Final Result [{mode.upper()}]:")
    print(f"Total Episodes: {n_episodes}")
    print(f"Success Rate:   {success_count}/{n_episodes} ({success_count/n_episodes*100:.2f}%)")
    print(f"Avg Steps:      {avg_steps:.1f}")
    print(f"Time Elapsed:   {end_time - start_time:.2f}s")
    print("="*30 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test RL Agent or NMPC Base Controller")
    
    # 1. 选择模式: actor (RL) 或 base (NMPC)
    parser.add_argument('--mode', type=str, default='actor', choices=['actor', 'base'], 
                        help='Choose policy to test: "actor" or "base"')
    
    # 2. 是否开启动画: 加上 --render 就开启，不加就关闭
    parser.add_argument('--render', action='store_true', 
                        help='Enable visualization (slows down testing)')
    
    # 3. 测试局数
    parser.add_argument('--episodes', type=int, default=10, 
                        help='Number of episodes to run')
    
    # 4. 模型路径 (仅 actor 模式需要)
    parser.add_argument('--dir', type=str, default='saves/nmpc_experiment', 
                        help='Directory containing actor.pt')
    
    args = parser.parse_args()
    
    run_test(args.mode, args.dir, args.episodes, args.render)