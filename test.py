import torch
import numpy as np
import argparse
import time
import os
import sys
import contextlib

from mujoco_env import CableRobotEnv
from nmpc_controller import NMPCController
# 导入我们新的网络结构
from agent import FastActor, ForwardPredictor 

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

def get_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")

def run_test(mode, log_dir, n_episodes, render, enable_init_rand=True, enable_process_noise=True, device_id=0):
    
    with silence_stderr():
        env = CableRobotEnv(
            render=render,
            latency_steps=1,
            force_noise_level=0.08 if enable_process_noise else 0.0,
            control_freq_hz=10,
            init_velocity_scale=0.05,
            init_position_range=0.35 if enable_init_rand else 0.06,
            enable_init_randomization=enable_init_rand,
            enable_process_noise=enable_process_noise
        )
    
    actor_model = None
    predictor_model = None
    nmpc_controller = None
    device = get_device(device_id)
    
    if mode == 'actor':
        # 【核心修改】加载包含 Actor 和 Predictor 的综合字典
        model_path = os.path.join(log_dir, 'actor_best.pt')
        if not os.path.exists(model_path):
            print(f"Error: 找不到模型文件 {model_path}")
            return
            
        print(f"Loading Dual-Brain Model from {model_path} ...") 
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        actor_model = checkpoint['actor']
        predictor_model = checkpoint['predictor']
        
        actor_model.eval()
        predictor_model.eval()
        
    elif mode == 'base':
        print("Initializing NMPC Controller (Base)...")
        nmpc_controller = NMPCController()
    
    success_count = 0
    total_steps_success = 0
    
    print("*******************************************")
    print(f"Start Testing [{mode.upper()}] for {n_episodes} episodes...")
    print(f"Render: {'ON' if render else 'OFF'}")
    print("*******************************************")
    
    for i in range(n_episodes):
        with silence_stderr():
            obs = env.reset()
            
        if mode == 'base' and nmpc_controller is not None:
            nmpc_controller.reset()
            
        target_pos = env.target_pos
        step = 0
        
        while True:
            if mode == 'actor':
                s_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    # 【核心修改】先用 Predictor 提取物理直觉，再给 Actor
                    emb = predictor_model.get_embedding(s_tensor)
                    action = actor_model(s_tensor, emb).cpu().numpy()[0]
            else:
                nmpc_state = obs[:8]
                with silence_stderr():
                    action = nmpc_controller.get_action(nmpc_state, target_pos)
                if np.isnan(action).any():
                    action = np.zeros(2)
            
            with silence_stderr():
                next_obs, reward, done, info = env.step(action)
            obs = next_obs
            step += 1
            
            if render:
                time.sleep(0.02)
            
            if info.get("physics_crash", False):
                if render: print(f"Ep {i+1}: Physics Crash! Steps={step}")
                break
                
            is_success = info.get("success", False)
            if done or is_success:
                if render: print(f"Ep {i+1}: Steps={step}, R={reward:.2f}, Success={is_success}")
                if is_success: 
                    success_count += 1
                    total_steps_success += step
                break

    avg_steps = total_steps_success / success_count if success_count > 0 else 0
    print(f"\nFinal Result [{mode.upper()}]: Success Rate: {success_count}/{n_episodes} ({success_count/n_episodes*100:.2f}%)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='actor', choices=['actor', 'base'])
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--dir', type=str, default='saves/ours_experiment/initRand_noProcNoise/seed_1')
    parser.add_argument('--enable_init_rand', type=int, default=1)
    parser.add_argument('--enable_process_noise', type=int, default=0)
    
    args = parser.parse_args()
    run_test(args.mode, args.dir, args.episodes, args.render,
             enable_init_rand=bool(args.enable_init_rand),
             enable_process_noise=bool(args.enable_process_noise))