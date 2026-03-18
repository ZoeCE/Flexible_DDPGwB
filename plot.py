import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def smooth(data, window=20):
    """滑动平均，让曲线更平滑好看"""
    return data.rolling(window=window, min_periods=1).mean()

def plot_comparison():
    # 定义两个实验的日志路径 (注意：这里是无过程噪声的地狱难度)
    baseline_log = 'saves/nmpc_experiment/initRand_noProcNoise/seed_1/log.csv'
    ours_log = 'saves/ours_experiment/initRand_noProcNoise/seed_1/log.csv'
    
    # 【请修改】填入你之前测试 NMPC 在地狱难度下的真实成功率 (比如 0.15)
    NMPC_BASELINE_SR = 0.15 

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # 读取数据
    df_base = pd.read_csv(baseline_log) if os.path.exists(baseline_log) else None
    df_ours = pd.read_csv(ours_log) if os.path.exists(ours_log) else None

    if df_ours is None:
        print("找不到 Ours 的日志文件，请检查路径！")
        return

    metrics =[
        {'col': 'test_success_rate', 'title': 'Test Success Rate (Evaluation)', 'ylabel': 'Success Rate', 'smooth': 3},
        {'col': 'train_success', 'title': 'Train Success Rate (Interaction)', 'ylabel': 'Success Rate', 'smooth': 50},
        {'col': 'avg_swing', 'title': 'Process Disturbance: Average Swing ($D_{swing}$)', 'ylabel': 'Swing Distance (m)', 'smooth': 50},
        {'col': 'ratio', 'title': 'Base Controller Usage Ratio', 'ylabel': 'Ratio', 'smooth': 20}
    ]

    colors = {'Baseline': '#1f77b4', 'Ours': '#d62728'} # 蓝 vs 红

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        col = metric['col']
        
        # 画 Baseline (普通 RL)
        if df_base is not None and col in df_base.columns:
            if col == 'test_success_rate':
                df_clean = df_base.dropna(subset=[col])
                ax.plot(df_clean['frames'], smooth(df_clean[col], metric['smooth']), 
                        color=colors['Baseline'], linewidth=2.5, alpha=0.6, label='Baseline (DDPG+BC)')
            else:
                ax.plot(df_base['frames'], smooth(df_base[col], metric['smooth']), 
                        color=colors['Baseline'], linewidth=2.5, alpha=0.6, label='Baseline (DDPG+BC)')

        # 画 Ours (预测网络 RL)
        if col in df_ours.columns:
            if col == 'test_success_rate':
                df_clean = df_ours.dropna(subset=[col])
                ax.plot(df_clean['frames'], smooth(df_clean[col], metric['smooth']), 
                        color=colors['Ours'], linewidth=3.0, label='Ours (Latent Forward Dynamics)')
            else:
                ax.plot(df_ours['frames'], smooth(df_ours[col], metric['smooth']), 
                        color=colors['Ours'], linewidth=3.0, label='Ours (Latent Forward Dynamics)')

        # 画 NMPC 基准线
        if 'success' in col:
            ax.axhline(y=NMPC_BASELINE_SR, color='gray', linestyle='--', linewidth=2, label=f'NMPC Baseline ({NMPC_BASELINE_SR*100:.0f}%)')
            ax.set_ylim(0, 1.05)

        ax.set_title(metric['title'], fontsize=15, fontweight='bold')
        ax.set_xlabel('Environment Steps', fontsize=12)
        ax.set_ylabel(metric['ylabel'], fontsize=12)
        ax.legend(fontsize=11)
        ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    save_path = 'saves/comparison_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 完美对比图已生成！保存在: {save_path}")

if __name__ == '__main__':
    plot_comparison()