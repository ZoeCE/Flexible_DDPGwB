import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def smooth(data, window=20):
    """滑动平均，让曲线更平滑好看"""
    return data.rolling(window=window, min_periods=1).mean()

def plot_large_range_result(log_dir='saves/nmpc_experiment/initRand_noProcNoise/seed_1'):
    # 注意上面的路径：因为你关了过程噪声，所以文件夹变成了 initRand_noProcNoise
    
    log_file = os.path.join(log_dir, 'log.csv')
    if not os.path.exists(log_file):
        print(f"找不到日志文件: {log_file}，请确认训练是否已经产生数据。")
        return

    df = pd.read_csv(log_file)
    
    # ==========================================
    # 【关键修改】这里填入你刚才用 test.py 测试 NMPC 得到的真实成功率！
    # 假设 NMPC 在地狱难度下只有 15% 的成功率，你就改成 0.15
    # ==========================================
    NMPC_BASELINE_SR = 0.34

    # 设置学术论文风格
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # ==========================================
    # 图 1: Test Success Rate (闭卷考试)
    # ==========================================
    ax = axes[0]
    df_test = df.dropna(subset=['test_success_rate']) # 过滤掉没有测试的回合
    if not df_test.empty:
        # 加上轻微平滑，消除毛刺
        y_smooth = smooth(df_test['test_success_rate'], window=3)
        ax.plot(df_test['frames'], y_smooth, color='#d62728', linewidth=2.5, label='RL Agent (Test)')
    
    ax.axhline(y=NMPC_BASELINE_SR, color='gray', linestyle='--', linewidth=2, label=f'NMPC Baseline ({NMPC_BASELINE_SR*100:.0f}%)')
    ax.set_title('Test Success Rate (Large Init Range)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Environment Steps', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)

    # ==========================================
    # 图 2: Train Success Rate (平时作业)
    # ==========================================
    ax = axes[1]
    train_sr_smooth = smooth(df['train_success'], window=50)
    ax.plot(df['frames'], train_sr_smooth, color='#1f77b4', linewidth=2, alpha=0.8, label='RL Agent (Train)')
    ax.axhline(y=NMPC_BASELINE_SR, color='gray', linestyle='--', linewidth=2, label='NMPC Baseline')
    ax.set_title('Train Success Rate (Interaction)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Environment Steps', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)

    # ==========================================
    # 图 3: Base Controller Usage Ratio (断奶过程)
    # ==========================================
    ax = axes[2]
    ratio_smooth = smooth(df['ratio'], window=20)
    ax.plot(df['frames'], ratio_smooth, color='#2ca02c', linewidth=2.5)
    ax.set_title('Base Controller Usage Ratio', fontsize=14, fontweight='bold')
    ax.set_xlabel('Environment Steps', fontsize=12)
    ax.set_ylabel('Ratio (0 to 1)', fontsize=12)
    ax.set_ylim(0, 1.05)

    # ==========================================
    # 图 4: Average Swing (平稳性证明)
    # ==========================================
    ax = axes[3]
    swing_smooth = smooth(df['avg_swing'], window=50)
    ax.plot(df['frames'], swing_smooth, color='#ff7f0e', linewidth=2.5)
    ax.set_title('Average Swing Distance ($D_{swing}$)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Environment Steps', fontsize=12)
    ax.set_ylabel('Swing Distance (m)', fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(log_dir, 'large_range_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 绘图成功！图表已保存至: {save_path}")

if __name__ == '__main__':
    plot_large_range_result()