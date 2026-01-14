import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse

def smooth(data, window=20):
    """滑动平均"""
    return data.rolling(window=window, min_periods=1).mean()

def read_all_seeds(root_dir):
    """读取所有 Seed 的日志并合并"""
    all_files = glob.glob(os.path.join(root_dir, 'seed_*', 'log.csv'))
    if not all_files:
        print(f"No log files found in {root_dir}")
        return None

    data_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            # 只需要关键列
            df = df[['frames', 'train_success', 'test_success_rate', 'ratio']]
            data_list.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    return data_list

def plot_paper_style(root_dir):
    data_list = read_all_seeds(root_dir)
    if not data_list:
        return

    # 设置绘图风格
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 定义要画的指标
    metrics = [
        {'col': 'train_success', 'title': 'Training Performance (Interaction)', 'ylabel': 'Success Rate', 'smooth': 50},
        {'col': 'test_success_rate', 'title': 'Test Performance (Eval)', 'ylabel': 'Success Rate', 'smooth': 1}, # Test 本身就是平均值，不用平滑
        {'col': 'ratio', 'title': 'Base Controller Usage', 'ylabel': 'Ratio', 'smooth': 20}
    ]

    # 统一 X 轴 (Frames)
    # 创建一个公共的 X 轴网格
    max_frames = min([df['frames'].max() for df in data_list])
    common_x = np.linspace(0, max_frames, 200) # 插值成 200 个点

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        col = metric['col']
        
        interpolated_y = []
        for df in data_list:
            # 处理 NaN (Test 数据是稀疏的)
            if col == 'test_success_rate':
                # 去掉 NaN 行
                df_clean = df.dropna(subset=[col])
                x = df_clean['frames']
                y = df_clean[col]
            else:
                x = df['frames']
                y = df[col]
                if metric['smooth'] > 1:
                    y = smooth(y, metric['smooth'])
            
            # 线性插值到公共 X 轴
            if len(x) > 1:
                y_interp = np.interp(common_x, x, y)
                interpolated_y.append(y_interp)

        if not interpolated_y:
            continue

        # 计算 Mean 和 Std
        y_matrix = np.array(interpolated_y)
        y_mean = np.mean(y_matrix, axis=0)
        y_std = np.std(y_matrix, axis=0)

        # 绘图
        ax.plot(common_x, y_mean, linewidth=2, color='royalblue', label='Ours (DDPG+Base)')
        ax.fill_between(common_x, y_mean - y_std, y_mean + y_std, color='royalblue', alpha=0.2)

        # 如果是成功率图，画 Base Controller 基线
        if 'success' in col:
            ax.axhline(y=0.605, color='gray', linestyle='--', label='Base Controller (60.5%)')
            ax.set_ylim(0, 1.05)

        ax.set_title(metric['title'])
        ax.set_xlabel('Environment Steps')
        ax.set_ylabel(metric['ylabel'])
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(root_dir, 'paper_result.png')
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='saves/nmpc_experiment', help='Root directory containing seed folders')
    args = parser.parse_args()
    
    plot_paper_style(args.dir)