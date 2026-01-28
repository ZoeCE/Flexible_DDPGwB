import torch
import numpy as np
import argparse
import time
import os
import sys

from mujoco_env import CableRobotEnv
from nmpc_controller import NMPCController
from agent import FastActor 

def get_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")

def run_test(mode, log_dir, n_episodes, render, device_id=0):
    # 【关键】测试时也要开启延迟和扰动，模拟真实物理
    # 使用与训练一致的参数（已降低扰动）
    env = CableRobotEnv(
        render=render,
        latency_steps=1,
        force_noise_level=0.08,
        control_freq_hz=10,
        init_velocity_scale=0.08,
        init_position_range=0.06
    )
    
    actor_model = None
    nmpc_controller = None
    
    if mode == 'actor':
        print(f"Loading Actor model from {log_dir}/actor_best.pt ...") # 优先加载 best
        device = get_device(device_id)
        model_path = os.path.join(log_dir, 'actor_best.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join(log_dir, 'actor.pt') # 降级加载
            
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return
        try:
            actor_model = torch.load(model_path, map_location=device, weights_only=False)
            actor_model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    elif mode == 'base':
        print("Initializing NMPC Controller (Base)...")
        nmpc_controller = NMPCController()
    
    success_count = 0
    total_steps_success = 0
    
    print("*******************************************")
    print(f"Start Testing [{mode.upper()}] for {n_episodes} episodes...")
    print(f"Render: {'ON' if render else 'OFF'}")
    print("*******************************************")
    
    start_time = time.time()
    
    for i in range(n_episodes):
        obs = env.reset()
        target_pos = env.target_pos
        step = 0
        
        while True:
            if mode == 'actor':
                s_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = actor_model(s_tensor).cpu().numpy()[0]
            else:
                nmpc_state = obs[:8]
                action = nmpc_controller.get_action(nmpc_state, target_pos)
            
            next_obs, reward, done, success = env.step(action)
            obs = next_obs
            step += 1
            
            if render:
                time.sleep(0.02)
            
            if done or step >= 200:
                if render:
                    print(f"Ep {i+1}: Steps={step}, R={reward:.2f}, Success={success}")
                
                if success: 
                    success_count += 1
                    total_steps_success += step
                break
        
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='actor', choices=['actor', 'base'])
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--dir', type=str, default='saves/nmpc_experiment/seed_1')
    
    args = parser.parse_args()
    run_test(args.mode, args.dir, args.episodes, args.render)