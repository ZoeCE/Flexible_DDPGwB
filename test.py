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

# 【修复 1】在函数定义中加入 enable_init_rand 和 enable_process_noise 参数
def run_test(mode, log_dir, n_episodes, render, enable_init_rand=True, enable_process_noise=True, device_id=0):
    
    # 【修复 2】将这两个参数传递给环境，确保测试时的扰动配置与命令行一致
    env = CableRobotEnv(
        render=render,
        latency_steps=1,
        force_noise_level=0.08,
        control_freq_hz=10,
        init_velocity_scale=0.08,
        init_position_range=0.06,
        enable_init_randomization=enable_init_rand,
        enable_process_noise=enable_process_noise
    )
    
    actor_model = None
    nmpc_controller = None
    
    if mode == 'actor':
        print(f"Loading Actor model from {log_dir}/actor_best.pt ...") 
        device = get_device(device_id)
        model_path = os.path.join(log_dir, 'actor_best.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join(log_dir, 'actor.pt') 
            
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
    print(f"Init Rand: {'ON' if enable_init_rand else 'OFF'} | Process Noise: {'ON' if enable_process_noise else 'OFF'}")
    print("*******************************************")
    
    start_time = time.time()
    
    for i in range(n_episodes):
        obs = env.reset()
        
        # 【关键修复】如果是测试 NMPC，每个回合必须重置历史解，防止空间跳跃导致崩溃
        if mode == 'base' and nmpc_controller is not None:
            nmpc_controller.reset()
            
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
                # 防止 NMPC 输出 NaN
                if np.isnan(action).any():
                    action = np.zeros(2)
            
            # 适配新的环境返回值 (obs, reward, done, info)
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            step += 1
            
            if render:
                time.sleep(0.02)
            
            # 如果发生物理崩溃，直接结束当前回合
            if info.get("physics_crash", False):
                if render:
                    print(f"Ep {i+1}: Physics Crash! Steps={step}")
                break
                
            if done or step >= 200:
                success = info.get("success", False)
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
    parser.add_argument('--dir', type=str, default='saves/nmpc_experiment/initRand_procNoise/seed_1')
    parser.add_argument('--enable_init_rand', type=int, default=1,
                        help='Enable init randomization (1=yes, 0=no)')
    parser.add_argument('--enable_process_noise', type=int, default=1,
                        help='Enable process noise (1=yes, 0=no)')
    
    args = parser.parse_args()
    run_test(args.mode, args.dir, args.episodes, args.render,
             enable_init_rand=bool(args.enable_init_rand),
             enable_process_noise=bool(args.enable_process_noise))