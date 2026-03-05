import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def smooth(data, window=20):
    return data.rolling(window=window, min_periods=1).mean()

def plot_ablation_study(root_dir='saves/nmpc_experiment'):
    # 定义四种消融实验的文件夹名称和图例标签
    conditions = {
        'Full Disturbance': 'initRand_procNoise',
        'Init Rand Only': 'initRand_noProcNoise',
        'Process Noise Only': 'noInitRand_procNoise',
        'Ideal (No Disturbance)': 'noInitRand_noProcNoise'
    }
    
    colors =['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'] # 红, 橙, 绿, 蓝
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    metrics =[
        {'col': 'test_success_rate', 'title': 'Test Success Rate (Evaluation)', 'ylabel': 'Success Rate', 'smooth': 1},
        {'col': 'train_success', 'title': 'Train Success Rate (Interaction)', 'ylabel': 'Success Rate', 'smooth': 50},
        {'col': 'avg_swing', 'title': 'Process Disturbance: Average Swing ($D_{swing}$)', 'ylabel': 'Swing Distance (m)', 'smooth': 50},
        {'col': 'ratio', 'title': 'Base Controller Usage Ratio', 'ylabel': 'Ratio', 'smooth': 20}
    ]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        col = metric['col']
        
        for c_idx, (label, folder) in enumerate(conditions.items()):
            # 假设我们读取 seed_1 的数据 (如果有多个seed可以扩展求平均)
            file_path = os.path.join(root_dir, folder, 'seed_1', 'log.csv')
            if not os.path.exists(file_path):
                print(f"Warning: File not found {file_path}")
                continue
                
            df = pd.read_csv(file_path)
            
            if col == 'test_success_rate':
                df_clean = df.dropna(subset=[col])
                x = df_clean['frames']
                y = df_clean[col]
            else:
                x = df['frames']
                y = smooth(df[col], metric['smooth'])
                
            ax.plot(x, y, linewidth=2.5, color=colors[c_idx], label=label, alpha=0.85)

        ax.set_title(metric['title'], fontsize=14, fontweight='bold')
        ax.set_xlabel('Environment Steps', fontsize=12)
        ax.set_ylabel(metric['ylabel'], fontsize=12)
        ax.legend(fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    save_path = os.path.join(root_dir, 'ablation_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {save_path}")

if __name__ == '__main__':
    plot_ablation_study()