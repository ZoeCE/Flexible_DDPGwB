# Vision 训练数据分析报告 2026-06-07

## 数据范围

本报告比较两组有效 vision+predictor descent 训练：

| 组别 | 目录 | 训练设置 | 状态 |
|---|---|---|---|
| A baseline | `saves/descent_ppo_10hz_vision_pred_small_endpoint_ft_20260606_090431` | `ppo-lr-actor=1e-5`, `ppo-lr-critic=3e-5`, 默认 action 正则 | 930,396 / 1,000,000 追加步数，约 93.0% |
| B fast | `saves/descent_ppo_10hz_vision_pred_small_endpoint_fast_ft_20260607_005307` | `ppo-lr-actor=2e-5`, `ppo-lr-critic=6e-5`, `action_rms_free=0.40`, `action_magnitude_coef=0.18` | 299,913 / 300,000 追加步数，已完成 |

修复前的 `descent_ppo_10hz_vision_pred_small_endpoint_fast_ft_20260607_004655` 因配置浅合并导致 `success_bonus` 缺失而崩溃，CSV 无有效 episode，排除在性能比较之外。

## 总体结论

B fast 分支在更少训练步数内达到了与 A baseline 后期几乎相同的成功率，并且平均奖励略好。说明提高 PPO 学习率并放松 action 正则确实加快了从 20% 成功率区域向 40% 成功率区域的过渡。

两组的视觉质量都稳定，不是当前训练瓶颈。真正瓶颈仍然是插入末端的接触顺序：`stuck_on_rebar` 和 `lucky_rebar_insert_failure` 合计仍占约 58-59%。

## 关键指标对比

| 指标 | A first500 | A last500 | B first500 | B last500 |
|---|---:|---:|---:|---:|
| 成功率 | 9.0% | 40.2% | 23.2% | 40.4% |
| 平均奖励 | -42.72 | -6.33 | -20.17 | -4.30 |
| 平均步数 | 107.3 | 90.1 | 101.4 | 90.8 |
| rolling SR 末值 | 11% | 36% | 19% | 39% |
| vision valid | 100.0% | 100.0% | 100.0% | 100.0% |
| 平均有效相机 | 2.877 | 2.848 | 2.863 | 2.846 |
| reproj mean/p95 | 0.185/0.291 px | 0.179/0.285 px | 0.178/0.286 px | 0.178/0.286 px |
| depth RMSE mean/p95 | 0.547/0.812 mm | 0.527/0.800 mm | 0.531/0.801 mm | 0.524/0.794 mm |

## 训练速度判断

A baseline 从 9.0% 提升到 40.2%，用了约 930k 追加控制步。  
B fast 从 resume 后第一段就达到 23.2%，300k 内到 40.4%。

更公平地看，B 是从 A 的 `ckpt_best` 继续训练，起点更强；但它在 300k 内稳定保持并略微提升到 40% 区间，说明 fast 设置没有破坏已有策略，且收敛速度比保守设置更高。

## 失败模式对比

### A last500

| 终止原因 | 占比 | 几何特征 |
|---|---:|---|
| `insertion_success` | 40.2% | `dtf=2.61mm`, `rebar=3.44mm`, `z=101.6mm`, `insert=18.35mm` |
| `stuck_on_rebar` | 29.8% | `dtf=8.42mm`, `rebar=9.36mm`, `z=119.9mm`, `insert=0.10mm` |
| `lucky_rebar_insert_failure` | 29.2% | `dtf=2.95mm`, `rebar=3.65mm`, `z=100.4mm`, `insert=19.70mm` |
| `early_stop` | 0.8% | `dtf≈29.8mm`, `z≈116mm` |

### B last500

| 终止原因 | 占比 | 几何特征 |
|---|---:|---|
| `insertion_success` | 40.4% | `dtf=2.66mm`, `rebar=3.38mm`, `z=101.8mm`, `insert=18.21mm` |
| `stuck_on_rebar` | 30.8% | `dtf=8.40mm`, `rebar=9.31mm`, `z=119.8mm`, `insert=0.19mm` |
| `lucky_rebar_insert_failure` | 27.0% | `dtf=2.77mm`, `rebar=3.49mm`, `z=100.4mm`, `insert=19.62mm` |
| `early_stop` | 1.6% | `dtf≈29.0mm`, `z≈115mm` |
| `timeout` | 0.2% | 已基本消失 |

B 相比 A 的后 500：

- 成功率基本持平：40.4% vs 40.2%。
- 平均奖励更好：-4.30 vs -6.33。
- `lucky_rebar_insert_failure` 略低：27.0% vs 29.2%。
- `stuck_on_rebar` 略高：30.8% vs 29.8%，差异很小。

## 视觉与预测器表现

两组 vision 指标几乎一致：

- `vision_valid_rate=100%`
- 有效相机约 2.85 个
- reprojection 误差约 0.18 px
- depth RMSE 约 0.52-0.53 mm

这说明侧面 tag + 三相机方案已经足够稳定。当前成功率上限主要不是视觉定位精度，而是控制策略对最后接触阶段的处理。

ObsPredictor 在 B 组耗时更高：

| 指标 | A all | A last500 | B all | B last500 |
|---|---:|---:|---:|---:|
| `rl_action_ms` | 1.81 | 2.01 | 2.55 | 2.52 |
| `obs_pred_ms` | 5.75 | 6.62 | 8.99 | 8.95 |
| `compute_hz_est` | 132.7 | 115.9 | 86.8 | 87.2 |

B 的 wall-clock 更慢，但训练质量更好。若后续追求吞吐，可以单独检查 WSL/CPU 负载、wandb IO、worker 初始化和 predictor 推理耗时。

## PPO 状态

| 指标 | A last500 | B last500 |
|---|---:|---:|
| policy loss | -0.0074 | -0.0070 |
| value loss | 5.660 | 5.519 |
| entropy | 0.864 | 0.870 |

两组没有明显策略崩溃迹象。B 的 entropy 略高，说明放松 action 正则没有让策略过早塌缩，探索仍然足够。

## 解释

当前策略已经能稳定到达插入区域，timeout 几乎消失。剩余失败集中在两个状态：

1. `stuck_on_rebar`：payload 在 `dtf≈8-9mm` 时过早下降到 rebar 顶部，横向误差还未进入 4.5mm 物理门限。
2. `lucky_rebar_insert_failure`：payload 已经在几何上接近成功，但之前发生过不干净 rebar 接触，被 strict lucky reject 判为失败。

这说明下一阶段优化重点应从“全局下降/定位”转向“末端插入接触顺序”。继续提高 LR 的收益可能有限，下一步更可能需要收紧 z 下降门控或加强插入前横向对准约束。

## 建议

1. 保留 B fast 设置作为新的默认 fine-tune 分支。它以 300k 训练量达到 A 后期水平，性价比更高。
2. 不建议放松 strict lucky reject。否则成功率会快速变好，但会学习到不干净接触插入。
3. 下一轮建议做“保守 z 门控”A/B：
   - `z_soft_gate_full`: 0.015 -> 0.006~0.008
   - `z_hard_gate`: 0.035 -> 0.018~0.022
   - `z_min_speed_frac`: 0.25 -> 0.10
   - `alignment_z_gate` / `premature_descent_xy_gate`: 0.012 -> 0.007
4. 指标监控重点：
   - `stuck_on_rebar` 是否从约 30% 降到 20% 以下
   - `lucky_rebar_insert_failure` 是否不再上升
   - `insertion_success` 是否突破 50%
   - `dtf` at stuck 是否从 8-9mm 降到 5-6mm

