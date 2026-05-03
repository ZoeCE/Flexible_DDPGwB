# ==============================================================================
# plot_results.py — 测试轨迹可视化
#
# 用法:
#   # 可视化单阶段测试轨迹
#   python plot_results.py --dir test_results/lift
#   python plot_results.py --dir test_results/cruise
#   python plot_results.py --dir test_results/descent
#
#   # 可视化流水线轨迹
#   python plot_results.py --dir test_results/pipeline
#
#   # 指定回合数
#   python plot_results.py --dir test_results/lift --max-eps 5
#
#   # 保存图片而不显示
#   python plot_results.py --dir test_results/lift --save-only
# ==============================================================================

import os
import glob
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

# 使用 Agg 后端 (无需 GUI) 当 --save-only
# matplotlib.use('Agg')


def load_episodes(result_dir, max_eps=None):
    """加载所有 episode npz 文件。"""
    files = sorted(glob.glob(os.path.join(result_dir, "ep_*.npz")))
    if max_eps is not None:
        files = files[:max_eps]
    episodes = []
    for f in files:
        data = dict(np.load(f, allow_pickle=True))
        episodes.append(data)
    return episodes


def plot_single_phase(episodes, phase, save_path=None):
    """为单阶段绘制可视化。"""
    n_eps = len(episodes)
    if n_eps == 0:
        print("无数据")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"{phase.upper()} 阶段专家基准测试 ({n_eps} 回合)", fontsize=14)

    # ── (0,0) XY 轨迹 ──────────────────────────────────────────
    ax = axes[0, 0]
    ax.set_title("Payload XY 轨迹")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect('equal')

    # 绘制工作空间
    ws_circle = plt.Circle((0, 0), 0.5, fill=False, linestyle='--',
                            color='gray', alpha=0.5, label='工作空间')
    ax.add_patch(ws_circle)

    colors = plt.cm.viridis(np.linspace(0, 1, n_eps))
    for i, ep in enumerate(episodes):
        pl = ep["payload"]
        success = bool(ep.get("success", False))
        style = '-' if success else '--'
        alpha = 0.8 if success else 0.4
        ax.plot(pl[:, 0], pl[:, 1], style, color=colors[i], alpha=alpha,
                linewidth=1.2, label=f'Ep{i} {"✅" if success else "❌"}' if i < 6 else None)
        ax.plot(pl[0, 0], pl[0, 1], 'o', color=colors[i], markersize=4)
        ax.plot(pl[-1, 0], pl[-1, 1], 's', color=colors[i], markersize=4)

    ax.plot(0, 0, 'k+', markersize=10, label='基座')
    if n_eps <= 8:
        ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.2, 0.55)

    # ── (0,1) Z 高度变化 ───────────────────────────────────────
    ax = axes[0, 1]
    ax.set_title("Payload Z 高度")
    ax.set_xlabel("Step")
    ax.set_ylabel("Z (m)")

    for i, ep in enumerate(episodes):
        pl = ep["payload"]
        success = bool(ep.get("success", False))
        style = '-' if success else '--'
        ax.plot(pl[:, 2], style, color=colors[i], alpha=0.6, linewidth=1)

    # 标注关键高度
    ax.axhline(y=0.25, color='green', linestyle=':', alpha=0.5, label='巡航高度')
    ax.axhline(y=0.10, color='red', linestyle=':', alpha=0.5, label='目标高度')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (1,0) 摆动速度 ─────────────────────────────────────────
    ax = axes[1, 0]
    ax.set_title("摆动相对速度 ||v_pl - v_ee||")
    ax.set_xlabel("Step")
    ax.set_ylabel("Swing Vel (m/s)")

    for i, ep in enumerate(episodes):
        swing = ep.get("swing_vel", np.zeros(len(ep["payload"])))
        success = bool(ep.get("success", False))
        ax.plot(swing, color=colors[i], alpha=0.5, linewidth=0.8)

    ax.axhline(y=0.08, color='orange', linestyle=':', alpha=0.5, label='lift 切换阈值')
    ax.axhline(y=0.05, color='red', linestyle=':', alpha=0.5, label='cruise 切换阈值')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (1,1) Tilt 姿态 ────────────────────────────────────────
    ax = axes[1, 1]
    ax.set_title("Payload Tilt 角度")
    ax.set_xlabel("Step")
    ax.set_ylabel("Tilt (rad)")

    for i, ep in enumerate(episodes):
        tilt = ep.get("tilt", np.zeros(len(ep["payload"])))
        success = bool(ep.get("success", False))
        ax.plot(tilt, color=colors[i], alpha=0.5, linewidth=0.8)

    ax.axhline(y=0.15, color='orange', linestyle=':', alpha=0.5, label='lift 切换阈值')
    ax.axhline(y=0.10, color='red', linestyle=':', alpha=0.5, label='cruise 切换阈值')
    ax.axhline(y=0.08, color='purple', linestyle=':', alpha=0.5, label='插入容差')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_pipeline(episodes, save_path=None):
    """为流水线绘制可视化。"""
    n_eps = len(episodes)
    if n_eps == 0:
        print("无数据")
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"全流水线专家基准测试 ({n_eps} 回合)", fontsize=14)

    phase_colors = {"lift": "#2196F3", "cruise": "#FF9800", "descent": "#4CAF50"}

    # ── (0,0) XY 轨迹 (全部, 按阶段着色) ──────────────────────
    ax = axes[0, 0]
    ax.set_title("Payload XY 轨迹 (按阶段着色)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect('equal')

    ws_circle = plt.Circle((0, 0), 0.5, fill=False, linestyle='--',
                            color='gray', alpha=0.5)
    ax.add_patch(ws_circle)

    for i, ep in enumerate(episodes):
        pl = ep["payload"]
        if "phase" in ep:
            phases = ep["phase"]
            for p_name, p_color in phase_colors.items():
                mask = (phases == p_name)
                if np.any(mask):
                    indices = np.where(mask)[0]
                    ax.plot(pl[indices, 0], pl[indices, 1], '-',
                            color=p_color, alpha=0.4, linewidth=1)
        else:
            ax.plot(pl[:, 0], pl[:, 1], '-', alpha=0.3, linewidth=0.8)

    # 图例
    for p_name, p_color in phase_colors.items():
        ax.plot([], [], '-', color=p_color, linewidth=2, label=p_name)
    ax.plot(0, 0, 'k+', markersize=10, label='基座')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.2, 0.55)

    # ── (0,1) Z 高度 ──────────────────────────────────────────
    ax = axes[0, 1]
    ax.set_title("Payload Z 高度")
    ax.set_xlabel("Step")
    ax.set_ylabel("Z (m)")

    for i, ep in enumerate(episodes):
        pl = ep["payload"]
        success = bool(ep.get("success", False))
        ax.plot(pl[:, 2], alpha=0.4, linewidth=0.8,
                color='green' if success else 'red')

    ax.axhline(y=0.25, color='blue', linestyle=':', alpha=0.5, label='巡航高度')
    ax.axhline(y=0.10, color='red', linestyle=':', alpha=0.5, label='目标高度')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (0,2) 摆动速度 ────────────────────────────────────────
    ax = axes[0, 2]
    ax.set_title("摆动相对速度")
    ax.set_xlabel("Step")
    ax.set_ylabel("Swing Vel (m/s)")

    for i, ep in enumerate(episodes):
        swing = ep.get("swing_vel", np.zeros(len(ep["payload"])))
        ax.plot(swing, alpha=0.3, linewidth=0.6)

    ax.grid(True, alpha=0.3)

    # ── (1,0) Tilt ─────────────────────────────────────────────
    ax = axes[1, 0]
    ax.set_title("Payload Tilt")
    ax.set_xlabel("Step")
    ax.set_ylabel("Tilt (rad)")

    for i, ep in enumerate(episodes):
        tilt = ep.get("tilt", np.zeros(len(ep["payload"])))
        ax.plot(tilt, alpha=0.3, linewidth=0.6)

    ax.axhline(y=0.08, color='red', linestyle=':', alpha=0.5, label='插入容差')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (1,1) 各阶段成功率 + 步数 ──────────────────────────────
    ax = axes[1, 1]
    ax.set_title("各阶段统计")

    phase_names = ["lift", "cruise", "descent"]
    srs = []
    avg_steps = []
    for p in phase_names:
        sr = np.mean([bool(ep.get(f"{p}_success", False)) for ep in episodes])
        srs.append(sr * 100)
        s = np.mean([float(ep.get(f"{p}_steps", 0)) for ep in episodes])
        avg_steps.append(s)

    x = np.arange(3)
    bars = ax.bar(x, srs, 0.5, color=[phase_colors[p] for p in phase_names],
                  alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in phase_names])
    ax.set_ylabel("成功率 (%)")
    ax.set_ylim(0, 110)

    for bar, sr, s in zip(bars, srs, avg_steps):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f'{sr:.0f}%\n({s:.0f}步)',
                ha='center', va='bottom', fontsize=9)

    ax.grid(True, alpha=0.3, axis='y')

    # ── (1,2) 回合奖励分布 ─────────────────────────────────────
    ax = axes[1, 2]
    ax.set_title("回合总奖励分布")

    rewards = [float(ep.get("reward", 0.)) for ep in episodes]
    successes = [bool(ep.get("success", False)) for ep in episodes]
    colors_ep = ['green' if s else 'red' for s in successes]

    ax.bar(range(len(rewards)), rewards, color=colors_ep, alpha=0.6)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.axhline(y=np.mean(rewards), color='blue', linestyle='--',
               alpha=0.5, label=f'平均: {np.mean(rewards):.1f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_stats_table(episodes, phase, save_path=None):
    """打印并保存统计表格。"""
    n = len(episodes)
    if n == 0:
        return

    rewards = [float(ep.get("reward", 0.)) for ep in episodes]
    successes = [bool(ep.get("success", False)) for ep in episodes]
    steps_list = [len(ep["payload"]) for ep in episodes]

    sr = np.mean(successes) * 100
    avg_r = np.mean(rewards)
    std_r = np.std(rewards)
    avg_s = np.mean(steps_list)

    # 终止原因
    from collections import Counter
    terms = Counter()
    for ep in episodes:
        t = str(ep.get("termination", "unknown"))
        terms[t.split(":")[0]] += 1

    print(f"\n{'─'*50}")
    print(f"  {phase.upper()} 统计汇总 ({n} 回合)")
    print(f"{'─'*50}")
    print(f"  成功率:       {sr:.1f}%")
    print(f"  平均奖励:     {avg_r:.2f} ± {std_r:.2f}")
    print(f"  平均步数:     {avg_s:.1f}")
    print(f"  终止原因:     {dict(terms)}")
    print(f"{'─'*50}\n")


def main():
    parser = argparse.ArgumentParser(description="测试轨迹可视化")
    parser.add_argument("--dir", type=str, required=True,
                        help="轨迹保存目录 (如 test_results/lift)")
    parser.add_argument("--max-eps", type=int, default=None,
                        help="最多显示的回合数")
    parser.add_argument("--save-only", action="store_true",
                        help="仅保存图片, 不弹出窗口")
    parser.add_argument("--output", type=str, default=None,
                        help="图片保存路径 (默认: <dir>/plot.png)")
    args = parser.parse_args()

    if args.save_only:
        matplotlib.use('Agg')

    episodes = load_episodes(args.dir, args.max_eps)
    if not episodes:
        print(f"未找到轨迹文件: {args.dir}")
        return

    print(f"加载了 {len(episodes)} 个回合")

    # 检测是否为 pipeline 数据
    is_pipeline = "phase" in episodes[0] or "lift_success" in episodes[0]

    # 推断 phase 名称
    phase = os.path.basename(args.dir.rstrip('/'))

    save_path = args.output or os.path.join(args.dir, "plot.png")

    if is_pipeline:
        plot_pipeline(episodes, save_path if args.save_only else None)
    else:
        plot_single_phase(episodes, phase, save_path if args.save_only else None)

    plot_stats_table(episodes, phase)

    if not args.save_only and args.output:
        # 也保存一份
        if is_pipeline:
            plot_pipeline(episodes, args.output)
        else:
            plot_single_phase(episodes, phase, args.output)


if __name__ == "__main__":
    main()