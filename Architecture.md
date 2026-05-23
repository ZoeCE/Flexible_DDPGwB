# Cable-Suspended Payload RL Training Framework — v13.0

> 2 阶段架构: **cruise** (NMPC 抬升+平移) + **descent** (PID 下降+插入)

---

## 一、快速开始

```bash
# 训练 (显式指定新 log-dir, 避免覆盖当前成功 ckpt)
python train_phase.py --phase cruise  --algo ppo --n-envs 8 --timesteps 2500000 --log-dir saves/cruise_ppo_next
python train_phase.py --phase descent --algo ppo --n-envs 8 --timesteps 3000000 --log-dir saves/descent_ppo_next

# 测试单段
python test_phase.py --phase cruise  --algo ppo --ckpt saves/cruise_ppo_next/ckpt_best.pt --episodes 20
python test_phase.py --phase descent --algo ppo --ckpt saves/descent_ppo_next/ckpt_best.pt --episodes 20

# 测试完整流水线 (cruise → descent)
python test_phase.py --phase pipeline \
    --cruise-ckpt saves/cruise_ppo_next/ckpt_best.pt \
    --descent-ckpt saves/descent_ppo_next/ckpt_best.pt \
    --cruise-algo ppo --descent-algo ppo \
    --episodes 30
```

**Checkpoint 保护**: 当前成功的 descent checkpoint 保留在 `saves/descent_ppo/`
中。下一次训练默认使用 `saves/descent_ppo_next/`; 除非明确要覆盖,
不要把 `--log-dir` 指回 `saves/descent_ppo`.

**注意**: `--phase lift` 仍被接受, 但会自动重定向到 `cruise`(并打印提示)。
旧脚本无需立即修改, 但建议尽快迁移。

---

## 二、架构总览 (v13.0)

### Phase 1: `cruise` — 抬升 + 平移统一段

```
起点: (start_xy, z=0.11)         ←—— payload 在地面附近
       │
       │ NMPC 自动垂直抬升 (lift sub-phase)
       │ tracker 检测 lift WP, 防止跨越 lift→cruise 边界 (v12.2)
       ↓
       (start_xy, z=z_cruise=0.25)
       │
       │ NMPC 水平平移 (cruise sub-phase) + lock_z 保持高度
       │ 加入 RL 残差 (3D acc) 抗扰
       ↓
终点: (target_xy, z=z_cruise)     ←—— payload 到达目标位置上方
```

- **Base Controller**: NMPC (统一, 内部识别 lift/cruise 两个子阶段)
- **RL 残差**: 3D acc (xy + z)
  - 低空段 (payload_z < z_cruise - 0.03):
    `expert.compute_delta_q_target(obs, cq, residual_acc=res3)` — 与 test_phase 完全一致
  - 高空段:
    `EEAccController.compute_delta_q(..., lock_z=True)` — 锁高度专心 xy
- **奖励**: 防摆 (swing_energy + cable_ke) + z_approach (低空引导) + tilt
- **成功条件**: payload 到达 `target_xy` ± success_radius

### Phase 2: `descent` — 下降 + 插入

```
起点: (target_xy, z=0.25)        ←—— 接续 cruise 终点
       │
       │ PID 3D base + RL 残差精度修正
       │ HER 回放经验加速学习
       ↓
终点: 插入钢筋 (xy_tol = 5mm)
```

- **Base Controller**: PID (3D 位置控制)
- **RL 残差**: 3D acc, **与 PID 输出解耦** (v12.3 fix)
- **奖励**: swing_ke + xy_align + rebar_align + precision_bonus
- **成功条件**: xy 误差 < 5mm + z 到达 target_z

---

## 三、关键设计 (v13.0 整合所有版本改进)

### 3.1 训练课程 (v12.5)

只保留风力扰动, 渐进 6 级 (cruise) / 5 级 (descent):

```python
# cruise (合并 lift): 6 级渐进, 风力 0 → 2N
[0.00, 0.40, 0.80, 1.20, 1.60, 2.00]

# descent: 5 级渐进, 风力 0 → 1N (descent 精度高更敏感)
[0.00, 0.25, 0.50, 0.75, 1.00]
```

**关键**: 课程倒退 (`regression`) 全部关闭, 避免 PPO 灾难性发散.
学坏时延长当前级训练 (`hard_cap_eps` 增大).

### 3.2 NMPC 与 EE 控制器一致性 (v12.1)

`EEAccController` 参数与 `JointSpaceExpert` 完全对齐:
- `vel_max_xy = 0.15` (一致)
- `vel_max_z = 0.20` (一致)
- `anchor_alpha = 0.10` (一致)

`JointSpaceExpert.compute_delta_q_target(obs, cq, residual_acc=res3)` 接受 3D 残差注入,
**与 test_phase 走完全相同的代码路径**.

### 3.3 绳索观测 + 能量惩罚 (v12.3)

每个 segment 的运动状态加入 obs:
- 4 根绳 × 10 段 = 40 个 link points
- 每段: `rel_pos (3) + lin_vel (3)` = 6 维
- 总 **240 维 cable obs**

`cable_ke_penalty`: 三段都加, 防止 RL 让 cable 高频振动.

### 3.4 细粒度 RL 评估指标 (v12.6)

成功率 (SR) 无法衡量 "RL vs 纯 base controller" 的改进.
新增 15 个指标输出到 wandb:

| 类别 | 指标 |
|------|------|
| 摆动质量 | `avg/max/p95/rms_ke_mJ`, `integral_ke_mJs`, `avg/max/p95/rms_angle` |
| 绳索动能 | `cable_ke_peak/avg/integral` |
| payload 运动 | `pl_vel_peak/rms` |
| EE 控制 | `avg/max_acc` |
| RL 介入 | `rl_action_mag_mean/peak` |

wandb 名称: `stab/{phase}/{metric}`

### 3.5 Descent 残差权威性修复 (v12.3)

旧 bug: `max_residual_norm = residual_dq_scale * max(|pid_dq|, ...)`
让 RL 残差随 PID 收敛而消失, 无法做最后 5mm 精度修正.

修复: `max_residual_norm = residual_dq_scale * dq_max_avg` —
RL 始终有恒定 ~36mm 权威.

### 3.6 单一精度训练 (v12.3, Ankile 2024 ResiP 方法)

descent 不再逐层降低精度, 直接在 **5mm 最终精度** 上训练:
- L0: 5mm + 无噪声
- L1-L4: 5mm + 渐进风力

---

## 四、超参数

### 主要配置

| 参数 | 值 | 说明 |
|------|----|----|
| `cruise_rl.action_dim` | 3 | xy + z 残差 |
| `cruise_rl.max_steps` | 700 | lift 段 + cruise 段总长度 |
| `cruise_rl.residual_acc_max_xy_rl` | 0.08 | xy 残差上限 |
| `cruise_rl.residual_acc_max_z_rl` | 0.10 | z 残差上限 (低空段用) |
| `cruise_rl.target_z_cruise` | 0.25 | 巡航高度 |
| `descent_rl.action_dim` | 3 | xy + z 残差 |
| `descent_rl.acc_max_xy` | 0.20 | 残差最大加速度 xy |
| `descent_rl.acc_max_z` | 0.40 | 残差最大加速度 z |
| `descent_rl.residual_dq_scale` | 0.30 | 残差占 base 比例 |

### PPO

| 参数 | 值 |
|------|----|
| hidden_dim | 384 (适配 ~270 维 obs) |
| n_layers | 3 |
| seq_len (LSTM) | 8 |
| lstm_dim | 128 |
| n_epochs | 6 |
| batch_size | 256 |
| target_kl | 0.02 |
| clip_eps | 0.2 |
| entropy_coef anneal | 0.05 → 0.005 over 600k steps |

### SAC (备用)

PPO 为主算法. SAC 仍可用 (`--algo sac`).

---

## 五、常用任务

### 5.1 完整训练流程

```bash
# Step 1: 训练 cruise (合并的抬升+平移段)
python train_phase.py --phase cruise --algo ppo --n-envs 8 --timesteps 2500000 --log-dir saves/cruise_ppo_next

# Step 2: 训练 descent
# 注意: 不要写入 saves/descent_ppo, 该目录保留当前成功 ckpt.
python train_phase.py --phase descent --algo ppo --n-envs 8 --timesteps 3000000 --log-dir saves/descent_ppo_next

# Step 3: 测试单段 SR + 细粒度指标
python test_phase.py --phase cruise  --algo ppo --ckpt saves/cruise_ppo_next/ckpt_best.pt --episodes 30
python test_phase.py --phase descent --algo ppo --ckpt saves/descent_ppo_next/ckpt_latest.pt --episodes 30 --render

# Step 4: 测试 pipeline (cruise → descent)
python test_phase.py --phase pipeline \
    --cruise-ckpt saves/cruise_ppo_next/ckpt_best.pt \
    --descent-ckpt saves/descent_ppo_next/ckpt_best.pt \
    --cruise-algo ppo --descent-algo ppo \
    --episodes 50 \
    --wind-force 1.5
```

### 5.2 baseline 对比 (RL vs 纯 NMPC/PID)

```bash
# 跑 expert-only baseline 收集 stab metrics
python test_phase.py --phase cruise --algo expert --episodes 30
python test_phase.py --phase descent --algo expert --episodes 30

# 跑 RL 训练后的版本
python test_phase.py --phase cruise --algo ppo --ckpt saves/cruise_ppo_next/ckpt_best.pt --episodes 30
python test_phase.py --phase descent --algo ppo --ckpt saves/descent_ppo_next/ckpt_best.pt --episodes 30
```

对比 `stab/cruise/max_angle`, `integral_ke_mJs` 等指标 →
量化 RL 防摆改善程度.

### 5.3 风力鲁棒性测试

```bash
# 不同风力下测试 (sim2real gap 评估)
for wind in 0.0 0.5 1.0 1.5 2.0; do
    python test_phase.py --phase pipeline \
        --cruise-ckpt saves/cruise_ppo_next/ckpt_best.pt \
        --descent-ckpt saves/descent_ppo_next/ckpt_best.pt \
        --cruise-algo ppo --descent-algo ppo \
        --episodes 20 --wind-force $wind
done
```

---

## 六、文件结构

```
.
├── README.md                       # 本文件
├── config.py                       # 所有超参数
├── controller.py                   # JointSpaceExpert + NMPCTrajectoryTracker
├── ee_acc_controller.py            # EE 加速度积分控制器
├── mujoco_env_new.py               # 环境 + obs (含 240 维 cable obs)
├── phase_agent.py                  # PPO / SAC agent + build_*_obs
├── phase_reward.py                 # reward 函数 (lift kept 仅供兼容)
├── train_phase.py                  # 训练入口 (PPO/SAC × single/vec)
├── test_phase.py                   # 测试入口 (single phase + pipeline)
├── vec_env.py                      # SubprocVecEnv 多进程
├── stability_metrics.py            # 共享 StabilityMetrics class
└── docs/                           # 设计文档
    ├── v12_4_lift_cruise_unified.md
    ├── v12_5_curriculum_smooth_progression.md
    ├── v12_6_train_test_metrics_adaptation.md
    └── v13_0_two_phase_merge.md    # 本次合并文档
```

---

## 七、版本历史关键改动

| 版本 | 改动 |
|------|------|
| v8-v10 | 原始 3 阶段 (lift/cruise/descent), 大量 reward 实验 |
| v11.2 | cruise reward 重设计 (CAPS + Olesen 2026) |
| v11.4 | SAC alpha 失控修复 + Q-divergence 防护 |
| v12.1 | train↔test NMPC 路径一致性修复 (核心 bug) |
| v12.2 | Lift expert 早期平移修复 (tracker 跨越保护) |
| v12.3 | 240 维 cable obs + descent residual 权威性修复 |
| v12.4 | Lift+cruise reward 合并 (统一防摆主体) |
| v12.5 | 课程平滑改造 (只保留风力, 6 级渐进, 关闭倒退) |
| v12.6 | 细粒度 stab metrics 集成到 train/test |
| **v13.0** | **完整合并: 2 阶段架构 (cruise + descent)** |

---

## 八、文献依据

1. **Ankile et al. 2024** (ResiP). arXiv:2407.16677 — 残差 RL 单一精度训练.
2. **Olesen et al. 2026** (Crane RL). arXiv:2602.05895 — anti-sway residual reward.
3. **Mysore et al. 2021** (CAPS). arXiv:2012.06644 — action smoothness regularization.
4. **Kotaru et al. 2017**. arXiv:1711.04895 — multi-link cable modeling.
5. **Goodarzi et al. 2014**. arXiv:1407.8164 — geometric control cable payload.
6. **FLARE 2025**. arXiv:2508.09797 — RL anti-sway for cable-suspended quadrotor.
7. **Pinto et al. 2017**. arXiv:1703.02702 — Robust adversarial RL (单一扰动维度).
8. **Narvekar et al. 2020**. arXiv:2003.04960 — Curriculum learning survey.

---

## 九、已知问题 + 提醒

1. **旧 checkpoints 不兼容**: obs_dim 263-278 (含 240 维 cable) + hidden 384,
   v12.0 之前的 ckpt 完全无法 load. 必须从头训练.

2. **VecEnv 模式 (`--n-envs > 1`)**: stab metrics 通过 remote 返回, 工作正常但
   每 ep 多一次序列化开销 (< 1ms, 可忽略).

3. **`--phase lift` 兼容**: 仍可用, 自动重定向到 cruise.
   旧 lift checkpoint 即使能加载也会 fail (action_dim 不一致).

4. **保留代码**: `phase_reward.py::compute_lift_reward` 等保留以兼容历史调用,
   但训练实际只调 cruise + descent. 长期会清理.

5. **课程层数与时长**: cruise (6 级) / descent (5 级) 都加大了 min_eps 和 hard_cap_eps,
   单段训练需要 200-300 万 step 才能跑完全部课程. 不要中途停训.
