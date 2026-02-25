#!/usr/bin/env python3
"""
扰动分离测试脚本
用于对比"初始化随机扰动"与"过程随机扰动"对控制性能的影响
"""

import numpy as np
import torch
import os
import sys
import contextlib
from mujoco_env import CableRobotEnv
from nmpc_controller import NMPCController

# ==========================================
# 【黑魔法】OS 级别屏蔽 C++ stderr 输出
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

def test_configuration(config_name, enable_init, enable_noise, n_episodes=100):
    """测试特定配置下的 NMPC 控制器性能"""
    print(f"\n{'='*60}")
    print(f"测试配置: {config_name}")
    print(f"  初始化随机扰动: {'启用' if enable_init else '禁用'}")
    print(f"  过程随机噪声:   {'启用' if enable_noise else '禁用'}")
    print(f"{'='*60}")
    
    with silence_stderr():
        env = CableRobotEnv(
            render=False,
            latency_steps=1,
            force_noise_level=0.08,
            control_freq_hz=10,
            init_velocity_scale=0.08,
            init_position_range=0.06,
            enable_init_randomization=enable_init,
            enable_process_noise=enable_noise
        )
    
    nmpc = NMPCController()
    
    success_count = 0
    total_steps =[]
    total_rewards =[]
    
    for ep in range(n_episodes):
        with silence_stderr():
            obs = env.reset()
            
        nmpc.reset() # 【关键】重置 NMPC 历史解
        target_pos = env.target_pos
        
        episode_reward = 0
        step = 0
        
        while step < 200:
            nmpc_state = obs[:8]
            with silence_stderr():
                action = nmpc.get_action(nmpc_state, target_pos)
                
            if np.isnan(action).any():
                action = np.zeros(2)
            
            with silence_stderr():
                # 【修复】正确接收 info 字典
                next_obs, reward, done, info = env.step(action)
                
            obs = next_obs
            episode_reward += reward
            step += 1
            
            # 如果发生物理崩溃，直接算作失败并结束
            if info.get("physics_crash", False):
                break
                
            # 正确解析 success 标志
            is_success = info.get("success", False)
            
            if done or is_success:
                if is_success:
                    success_count += 1
                    total_steps.append(step)
                break
        
        total_rewards.append(episode_reward)
        
        if (ep + 1) % 20 == 0:
            print(f"  进度: {ep+1}/{n_episodes} | 当前成功率: {success_count/(ep+1)*100:.1f}%")
    
    # 统计结果
    success_rate = success_count / n_episodes
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps) if total_steps else 0
    
    print(f"\n结果:")
    print(f"  成功率:       {success_rate*100:.2f}% ({success_count}/{n_episodes})")
    print(f"  平均奖励:     {avg_reward:.4f}")
    print(f"  平均步数:     {avg_steps:.1f} (仅成功案例)")
    
    return {
        'config': config_name,
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'enable_init': enable_init,
        'enable_noise': enable_noise
    }

def main():
    print("\n" + "="*60)
    print("扰动分离对比实验")
    print("目标: 定位控制失败的根本原因")
    print("="*60)
    
    configs =[
        ("理想情况 (无扰动)", False, False),
        ("仅初始化扰动", True, False),
        ("仅过程噪声", False, True),
        ("完整扰动 (两者都有)", True, True),
    ]
    
    results =[]
    
    for config_name, enable_init, enable_noise in configs:
        result = test_configuration(config_name, enable_init, enable_noise, n_episodes=100)
        results.append(result)
    
    print("\n" + "="*60)
    print("对比总结")
    print("="*60)
    print(f"{'配置':<25} {'成功率':<12} {'平均奖励':<12} {'平均步数':<10}")
    print("-"*60)
    
    for r in results:
        print(f"{r['config']:<25} {r['success_rate']*100:>6.2f}%     {r['avg_reward']:>8.4f}     {r['avg_steps']:>6.1f}")
    
    print("\n" + "="*60)
    print("分析结论:")
    print("="*60)
    
    ideal = results[0]['success_rate']
    init_only = results[1]['success_rate']
    noise_only = results[2]['success_rate']
    both = results[3]['success_rate']
    
    print(f"1. 理想情况成功率: {ideal*100:.1f}%")
    print(f"2. 仅初始化扰动导致成功率下降: {(ideal-init_only)*100:.1f}%")
    print(f"3. 仅过程噪声导致成功率下降: {(ideal-noise_only)*100:.1f}%")
    print(f"4. 完整扰动成功率: {both*100:.1f}%")
    
    if (ideal - init_only) > (ideal - noise_only):
        print("\n结论: 初始化扰动是主要问题 - 路径规划能力不足")
    elif (ideal - noise_only) > (ideal - init_only):
        print("\n结论: 过程噪声是主要问题 - 抗干扰能力不足")
    else:
        print("\n结论: 两种扰动影响相当 - 需要同时改进")
    
    print("="*60 + "\n")

if __name__ == '__main__':
    main()