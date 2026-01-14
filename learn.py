import numpy as np
import torch
import csv
import os
import time
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from agent import WBAgent
from mujoco_env import CableRobotEnv
from nmpc_controller import NMPCController

# 全局变量用于 Wrapper
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

# --- 核心函数：定期测试 (Test Performance) ---
def evaluate_policy(env, agent, n_eval_episodes=100):
    """
    对应论文：每隔一定步数，暂停训练，进行无噪声测试
    """
    success_count = 0
    
    for _ in range(n_eval_episodes):
        state = env.reset()
        done = False
        step = 0
        
        while not done and step < 200:
            # test=True: 不加噪声，纯 Actor 策略
            action, _, _ = agent.act(state, test=True)
            
            # 确保动作范围
            action = np.clip(action, -agent.max_action, agent.max_action)
            
            state, reward, done, success = env.step(action)
            step += 1
            
            if success:
                success_count += 1
                done = True
                
    return success_count / n_eval_episodes

def train(log_dir, seed=0):
    global nmpc_instance, env_target_pos
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 初始化环境
    env = CableRobotEnv(render=False) 
    eval_env = CableRobotEnv(render=False) # 独立的测试环境
    
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
    
    # 日志文件
    log_file = os.path.join(log_dir, 'log.csv')
    
    # CSV 表头：
    # episode: 当前回合
    # frames: 总步数
    # train_return: 训练时的带惩罚总分
    # train_success: 训练时是否成功 (0/1)
    # test_success_rate: 定期测试的平均成功率 (0.0-1.0)
    with open(log_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['episode', 'frames', 'train_return', 'train_success', 'test_success_rate', 'ratio', 'epsilon'])

    n_episodes = 2000
    max_frames_total = 2e5
    frames_total = 0
    
    best_test_sr = -1.0 # 记录历史最好成绩
    
    # 训练统计
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
            
            # --- 1. 训练循环 (Training Episode) ---
            while True:
                # act 返回: 动作, 是否网络, Base动作
                action, is_network, base_action_val = agent.act(state)
                
                if not is_network:
                    ratio_count += 1
                    action_exec = action 
                else:
                    # 训练时加噪声
                    action_exec = action + np.random.normal(0, 0.05, size=2)
                
                action_exec = np.clip(action_exec, -MAX_ACTION, MAX_ACTION)
                
                next_state, reward, done, success = env.step(action_exec)
                
                agent.remember(state, action_exec, base_action_val, next_state, reward, done)
                
                state = next_state
                episode_reward += reward
                step_count += 1
                frames_total += 1
                
                # 训练网络
                agent.train(2)
                
                if done or step_count >= 150:
                    break
            
            # --- 2. 记录训练表现 (Training Performance) ---
            is_success = 1 if episode_reward > 0.5 else 0
            train_success_count += is_success
            current_ratio = ratio_count / step_count
            
            # --- 3. 定期测试 (Test Performance) ---
            # 论文逻辑：每 30 个 episode 测试一次
            test_sr = np.nan # 默认空值，方便画图处理
            
            if (episode + 1) % 30 == 0:
                print(f"\n[Evaluation] Episode {episode+1}: Running 100 test episodes...")
                test_sr = evaluate_policy(eval_env, agent, n_eval_episodes=100)
                
                # 【关键】保存最佳模型
                if test_sr > best_test_sr:
                    best_test_sr = test_sr
                    torch.save(agent.actor, os.path.join(log_dir, 'actor_best.pt'))
                    print(f"  >>> New Best Model Saved! SR: {test_sr*100:.1f}%")
                else:
                    print(f"  --- Current SR: {test_sr*100:.1f}% (Best: {best_test_sr*100:.1f}%)")
            
            # --- 4. 写入日志 ---
            with open(log_file, "a+", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([episode, frames_total, episode_reward, is_success, test_sr, current_ratio, agent.epsilon])
            
            progress.update(task_id, advance=1)
            progress.refresh()
            
            # 打印简报
            if (episode + 1) % 10 == 0:
                print(f"Ep: {episode+1} | Train_R: {episode_reward:.2f} | Ratio: {current_ratio:.2f} | Eps: {agent.epsilon:.2f}")

            if frames_total >= max_frames_total:
                break

    # --- 5. 训练结束：最终大考 (Final Evaluation) ---
    print("\n" + "="*40)
    print("Training Finished. Starting Final Evaluation (1000 Episodes)...")
    print("Loading 'actor_best.pt'...")
    
    best_model_path = os.path.join(log_dir, 'actor_best.pt')
    if os.path.exists(best_model_path):
        # 加载最佳模型
        # 注意：这里需要处理 weights_only 问题，或者确保 agent 类定义一致
        try:
            best_actor = torch.load(best_model_path, map_location=agent.device, weights_only=False)
            # 临时替换 agent 的 actor 进行测试
            original_actor = agent.actor
            agent.actor = best_actor
            
            final_sr = evaluate_policy(eval_env, agent, n_eval_episodes=1000)
            
            print(f"Final Test Performance (Table I): {final_sr*100:.2f}%")
            
            # 恢复
            agent.actor = original_actor
        except Exception as e:
            print(f"Error loading best model: {e}")
    else:
        print("No best model found (maybe training crashed?).")
    print("="*40 + "\n")

if __name__ == '__main__':
    # 支持简单的命令行参数来跑不同的 Seed
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    args = parser.parse_args()
    
    # 日志目录区分 Seed
    log_dir = f'saves/nmpc_experiment/seed_{args.seed}'
    train(log_dir, seed=args.seed)