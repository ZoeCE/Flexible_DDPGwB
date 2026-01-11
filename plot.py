import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def smooth(data, window=50):
    """对数据进行滑动平均平滑"""
    return data.rolling(window=window, min_periods=1).mean()

def plot_training_results(log_dir):
    log_path = os.path.join(log_dir, 'log.csv')
    
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        return

    # 读取数据
    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 设置绘图风格
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # 1. 绘制 Success Rate (Return)
    # 因为 Reward 是 0 或 1，滑动平均后就是成功率
    window_size = 50
    success_rate = smooth(df['return'], window=window_size)
    
    axes[0].plot(df['frames'], df['return'], alpha=0.2, color='gray', label='Raw Reward')
    axes[0].plot(df['frames'], success_rate, color='royalblue', linewidth=2, label=f'Success Rate (MA-{window_size})')
    axes[0].set_ylabel('Success Rate / Reward')
    axes[0].set_title('Training Performance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. 绘制 Ratio (Base Controller Usage) & Epsilon
    # 注意：log.csv 里没有直接记 epsilon，但 ratio 反映了 epsilon 的效果
    axes[1].plot(df['frames'], df['ratio'], color='crimson', linewidth=2, label='Base Controller Ratio')
    axes[1].set_ylabel('Ratio')
    axes[1].set_title('Base Controller Usage Ratio')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. 绘制 Loss (Critic & Actor)
    # 对 Loss 也做一点平滑，因为波动很大
    loss_c_smooth = smooth(df['Lc'], window=20)
    loss_a_smooth = smooth(df['La'], window=20)
    loss_bc_smooth = smooth(df['Lbc'], window=20)

    axes[2].plot(df['frames'], loss_c_smooth, label='Critic Loss', color='orange')
    axes[2].plot(df['frames'], loss_bc_smooth, label='BC Loss', color='green')
    # Actor Loss 通常是负的 Q 值，画在一起可能比例不对，可以考虑双轴，这里先画在一起
    # axes[2].plot(df['frames'], loss_a_smooth, label='Actor Loss', color='purple', linestyle='--')
    
    axes[2].set_ylabel('Loss')
    axes[2].set_xlabel('Environment Steps')
    axes[2].set_title('Training Losses')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log') # Loss 变化范围大，用对数坐标看细节

    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(log_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='saves/nmpc_experiment', help='Path to log directory')
    args = parser.parse_args()
    
    plot_training_results(args.dir)