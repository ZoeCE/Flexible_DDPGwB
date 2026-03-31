import torch
import numpy as np
import argparse
import time
import os
import sys

from ipdb import set_trace as xxxx

# 引入环境和控制器
from mujoco_env import CableRobotEnv, CableRobotEnvWithObstacles
from nmpc_controller import NMPCController, NMPCTrajectoryTracker
# 引入 Agent 网络定义 (必须，否则 torch.load 报错)
from agent import FastActor 

def get_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")

def run_test(mode, log_dir, n_episodes, render, device_id=0):
    """
    统一的测试主循环 (针对基础 2D 环境，无障碍物)
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
                # RL Agent: 输入 State -> 输出 Action
                s_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = actor_model(s_tensor).cpu().numpy()[0]
            else:
                # NMPC Base: 输入 State + Target -> 输出 Action
                nmpc_state = obs[:8]
                action = nmpc_controller.get_action(nmpc_state, target_pos)
            
            # --- 环境交互 ---
            next_obs, reward, done, success = env.step(action)
            
            obs = next_obs
            episode_reward += reward
            step += 1
            
            # 渲染延时，方便肉眼观察
            if render:
                time.sleep(0.02)
            
            # --- 结束判定 ---
            if done or step >= 500:
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


def run_test_obstacles(mode, log_dir, n_episodes=10, render=False, n_obstacles=3,
                       obstacle_seed=42, save_paths_dir=None,
                       payload_radius=0.2, planning_margin=0.2, planning_grid_res=0.02,
                       default_start_xy=None, default_target_xy=None, device_id=0):
    """带障碍物避碰的 NMPC 测试：CableRobotEnvWithObstacles + NMPCTrajectoryTracker。"""
    
    # 严格保留原有环境参数
    env = CableRobotEnvWithObstacles(
        render=render,
        latency_steps=1,
        force_noise_level=0.08,
        control_freq_hz=10,
        init_velocity_scale=0.08,
        init_position_range=0.00,
        n_obstacles=n_obstacles,
        obstacle_radius_range=(0.01, 0.02),
        path_width=0.12,
        obstacle_seed=obstacle_seed,
        default_start_xy=default_start_xy or [0.2, 0.2],
        default_target_xy=default_target_xy or [0.5, 0.5],
        payload_radius=payload_radius,
        planning_margin=planning_margin,
        planning_grid_res=planning_grid_res,
    )

    actor_model = None
    nmpc_controller = None

    # 模型加载与初始化
    if mode == 'actor_obstacles':
        print(f"Loading Actor model from {log_dir}/actor.pt ...")
        device = get_device(device_id)
        model_path = os.path.join(log_dir, 'actor.pt')
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return
        
        try:
            actor_model = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            actor_model = torch.load(model_path, map_location=device)
        actor_model.eval()
        
    elif mode in ['obstacles', 'obstacles_base']:
        print("Initializing NMPC Trajectory Tracker (Following 3D path)...")
        nmpc_controller = NMPCTrajectoryTracker()

    if save_paths_dir is not None:
        os.makedirs(save_paths_dir, exist_ok=True)

    # 统计变量
    success_count = 0
    total_steps_success = 0
    collision_count = 0
    
    print("*******************************************")
    print(f"Start Testing [{mode.upper()}] for {n_episodes} episodes...")
    print(f"Render: {'ON' if render else 'OFF'}, Obstacles: {n_obstacles}")
    print("*******************************************")
    
    start_time = time.time()
    
    for ep in range(n_episodes):
        obs = env.reset()
        step = 0
        episode_collision = False
        target_pos = env.target_pos # 获取绝对终点
        
        # 获取当前环境生成的障碍物列表，用于后续碰撞统计
        obstacles = env.get_obstacles()
        
        # 加载环境规划的轨迹
        if mode in ['obstacles', 'obstacles_base', 'actor_obstacles'] and hasattr(env, "get_planned_path"):
            planned_path = env.get_planned_path()
            if planned_path is not None and len(planned_path) > 0:
                if nmpc_controller:
                    nmpc_controller.set_trajectory(planned_path)
            else:
                if nmpc_controller:
                    nmpc_controller.set_trajectory([])
        
        # 确保每回合重置 NMPC 内部的状态机
        if nmpc_controller and hasattr(nmpc_controller, 'reset_state_machine'):
            nmpc_controller.reset_state_machine()
            
        while True:
            # === 1. 动作计算逻辑 (全面适配 3D 接口) ===
            if mode == 'actor_obstacles':
                s_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = actor_model(s_tensor).cpu().numpy()[0]
            else:
                # [针对 3D NMPC 的修改]：向控制器传入 env_wp_idx 保证航点一致性
                current_wp = getattr(env, 'current_wp_idx', None)
                action = nmpc_controller.get_tracking_action(obs, env_wp_idx=current_wp)
            
            # === 2. 环境推演 ===
            next_obs, reward, done, success = env.step(action)
            obs = next_obs
            step += 1
            
            # === 3. 碰撞与渲染处理 ===
            if reward <= -5.0 and not success: 
                episode_collision = True
                
            if render:
                time.sleep(0.01) 
                
            if done or step >= 500:
                if render:
                    status = "✅ Success" if success else "❌ Failed"
                    col_status = " (Collision!)" if episode_collision else ""
                    print(f"Ep {ep+1:3d} | {status}{col_status} | Reward: {reward:7.2f} | Steps: {step:3d}")
                
                if success:
                    success_count += 1
    
    print("\n" + "="*50)
    print(f"Final Result [{mode.upper()}]:")
    print(f"Total Episodes: {n_episodes}")
    print(f"Success Rate:   {success_count}/{n_episodes} ({success_count/n_episodes*100:.2f}%)")
    print(f"Collision Rate: {collision_count}/{n_episodes} ({collision_count/n_episodes*100:.2f}%)")
    print(f"Avg Steps:      {avg_steps:.1f}")
    print(f"Time Elapsed:   {end_time - start_time:.2f}s")
    print("="*50 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Cable Robot Policy')
    
    # 恢复原有的所有 mode choices
    parser.add_argument('--mode', type=str, default='actor_obstacles', 
                        choices=['base', 'actor', 'obstacles', 'obstacles_base', 'actor_obstacles'],
                        help='Test mode')
    parser.add_argument('--render', action='store_true', help='Enable MuJoCo rendering')
    parser.add_argument('--episodes', type=int, default=10, help='Number of test episodes')
    parser.add_argument('--dir', type=str, default='saves/nmpc_experiment_0.95', help='Directory with actor.pt')

    # 障碍物模式专用参数
    parser.add_argument('--obstacles', type=int, default=3, help='[obstacles] Number of obstacles per episode')
    parser.add_argument('--seed', type=int, default=42, help='[obstacles] Obstacle RNG seed')
    parser.add_argument('--save_paths_dir', type=str, default=None,
                        help='[obstacles] Save planned 3D paths as CSV to this dir')
    parser.add_argument('--payload_radius', type=float, default=0.10, help='[obstacles] Payload safety radius (m)')
    parser.add_argument('--planning_margin', type=float, default=0.10, help='[obstacles] Planning margin (m)')
    parser.add_argument('--planning_grid_res', type=float, default=0.02, help='[obstacles] Grid resolution (m)')

    args = parser.parse_args()

    # 路由及参数传递：基础 2D 环境与 3D 障碍物环境的分流
    if args.mode in ['actor', 'base']:
        run_test(mode=args.mode, log_dir=args.dir, n_episodes=args.episodes, render=args.render)
    elif args.mode in ['obstacles', 'obstacles_base', 'actor_obstacles']:
        run_test_obstacles(
            mode=args.mode,
            log_dir=args.dir,
            n_episodes=args.episodes,
            render=args.render,
            n_obstacles=args.obstacles,
            obstacle_seed=args.seed,
            save_paths_dir=args.save_paths_dir,
            payload_radius=args.payload_radius,
            planning_margin=args.planning_margin,
            planning_grid_res=args.planning_grid_res
        )
    else:
        run_test(args.mode, args.dir, args.episodes, args.render)