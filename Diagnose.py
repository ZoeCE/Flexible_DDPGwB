"""诊断脚本：检查 PPO 预训练后 actor 为什么失败。"""
import copy
import numpy as np
import torch
from config import DEFAULT_CONFIG
from mujoco_env_new import CableRobotEnvWithObstacles
from controller import JointSpaceExpert
from agent import PPOAgent

config = copy.deepcopy(DEFAULT_CONFIG)
config["scene"]["seed"] = 42

env = CableRobotEnvWithObstacles(config=config)
STATE_DIM = env.state_dim
ACTION_DIM = config["space"]["action_dim"]

# 先测试纯专家
expert = JointSpaceExpert(config, env.ik_solver)
dq_max = np.array(config["space"].get("dq_max", [0.1]*7), dtype=np.float32)

print("="*60)
print("测试 1: 纯专家 (compute_delta_q_target) 执行")
print("="*60)

for ep in range(5):
    obs = env.reset()
    current_q = env.data.qpos[:7].copy()
    expert.reset(obs, current_q, env=env)
    planned_path = env.get_planned_path()
    if planned_path is None:
        print(f"  Ep {ep}: 路径规划失败"); continue
    expert.set_path(planned_path)
    n_wps = len(planned_path)
    
    ep_reward = 0; step = 0; ep_success = False
    termination_reason = "timeout"
    max_wp = 0
    
    while True:
        current_q = env.data.qpos[:7].copy().astype(np.float32)
        bc_dq = expert.compute_delta_q_target(obs, current_q)
        
        next_obs, reward, terminated, truncated, info = env.step(bc_dq)
        done = terminated or truncated
        ep_reward += reward; step += 1
        
        if info.get("is_success"):
            ep_success = True; termination_reason = "success"
        if info.get("is_collision"):
            termination_reason = "collision"
        max_wp = max(max_wp, info.get("current_wp_idx", 0))
        
        # 检查终止时的状态
        if done:
            payload_xy = np.array([next_obs[4], next_obs[5]])
            payload_z = env.data.body('prefab').xpos[2]
            pl_vxy = np.array([next_obs[6], next_obs[7]])
            dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
            pl_vz = env.data.qvel[dof_idx + 2]
            target = env.target_pos
            dtf = np.linalg.norm(payload_xy - target)
            vel_xy = np.linalg.norm(pl_vxy)
            
            if not ep_success and info.get("reached_final", False):
                termination_reason = f"reached_final_but_failed(dtf={dtf:.3f},vxy={vel_xy:.3f},vz={pl_vz:.3f})"
            elif step >= config["sim"]["max_steps"]:
                termination_reason = f"timeout(wp={max_wp}/{n_wps},dtf={dtf:.3f})"
            
            crash_z = config["step_logic"]["crash_z_threshold"]
            if payload_z < crash_z:
                termination_reason = f"crash_z(z={payload_z:.3f}<{crash_z})"
            
            print(f"  Ep {ep}: {termination_reason} | R={ep_reward:.2f} | steps={step} | wp={max_wp}/{n_wps}")
            break
        
        obs = next_obs

print()
print("="*60)
print("测试 2: 检查 actor delta_q 输出范围和 NaN")
print("="*60)

# 加载最新 checkpoint（如果存在）
import os, glob
ckpt_dirs = ["saves/ppo_run", "saves/td3_run"]
for d in ckpt_dirs:
    ckpt = os.path.join(d, "ckpt_latest.pt")
    if not os.path.exists(ckpt):
        continue
    
    print(f"\n加载 checkpoint: {ckpt}")
    
    # 自动判断 agent 类型
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    is_td3 = "target_actor" in ck
    algo_name = "TD3" if is_td3 else "PPO"
    print(f"  检测到 {algo_name} checkpoint")
    
    if is_td3:
        from agent import TD3Agent
        agent = TD3Agent(None, STATE_DIM, ACTION_DIM, config=config)
    else:
        agent = PPOAgent(None, STATE_DIM, ACTION_DIM, config=config)
    agent.load(ckpt, map_location=torch.device("cpu"))
    
    obs = env.reset()
    expert2 = JointSpaceExpert(config, env.ik_solver)
    current_q = env.data.qpos[:7].copy()
    expert2.reset(obs, current_q, env=env)
    planned_path = env.get_planned_path()
    if planned_path is not None and len(planned_path) > 0:
        expert2.set_path(planned_path)
    
    for step in range(10):
        norm_obs = agent.normalize_obs(obs, update=False)
        if is_td3:
            action, _ = agent.act(norm_obs)
        else:
            action, _, _ = agent.act(norm_obs, deterministic=True)
            
            # 专家参考
            current_q = env.data.qpos[:7].copy().astype(np.float32)
            expert_dq = expert2.compute_delta_q_target(obs, current_q)
            
            diff = np.abs(action - expert_dq)
            print(f"  Step {step}: actor_dq_max={np.max(np.abs(action)):.4f} "
                  f"expert_dq_max={np.max(np.abs(expert_dq)):.4f} "
                  f"diff_max={np.max(diff):.4f} diff_mean={np.mean(diff):.4f} "
                  f"nan={np.any(np.isnan(action))}")
            
            obs, _, done, _, _ = env.step(action)
            if done: break

env.close()