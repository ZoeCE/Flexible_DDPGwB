# 三阶段 RL 控制架构设计文档 v4

## 1. 架构总览

```
┌──────────────────────────────────────────────────────────────────────┐
│                    三阶段分治控制系统 v4                              │
│                                                                      │
│  ┌─────────────┐    ┌──────────────────────┐    ┌────────────────┐  │
│  │  Phase 1    │    │  Phase 2             │    │  Phase 3       │  │
│  │  Lift RL    │───>│  Cruise              │───>│  Descent RL    │  │
│  │  (提升)     │    │  (平移避障+防摆)     │    │  (精准下降)    │  │
│  └──────┬──────┘    └──────┬───────────────┘    └──────┬─────────┘  │
│         │                  │                            │            │
│    EE acc(3D)    ┌─────────┴──────────┐          PID+residual       │
│    ax,ay,az      │  ORCA Expert (BC)  │          EE acc(3D)        │
│    差分 reward   │  CruiseDualRLAgent │                            │
│    KE+PE 摆动    │  ┌──────────────┐  │                            │
│    XY差分惩罚    │  │ planner_RL   │  │                            │
│                  │  │ (导航+避障)  │  │                            │
│                  │  ├──────────────┤  │                            │
│                  │  │ swing_RL     │  │                            │
│                  │  │ (防摆残差)   │  │                            │
│                  │  └──────────────┘  │                            │
│                  └────────────────────┘                            │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              EE Acceleration Controller                      │   │
│  │  acc → 积分 → EE vel/pos → IK → q_target → delta_q         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │           Detailed WandB Reward Tracking                     │   │
│  │  各阶段: z_approach / xy_drift / swing_energy / step_penalty │   │
│  │  cruise: pbrs_nav / obs_repulsion / swing_penalty           │   │
│  │  descent: xy_align / z_descent / swing_ke / tilt / yaw      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

## 2. v4 主要变更

### 2.1 Phase 1: Lift RL — 增强 Reward

**原问题**: 纯 KE 摆动惩罚忽略势能; XY 绝对值惩罚导致"提升=漂移=惩罚"局部最优

**v4 修改**:
- 摆动能量: KE + PE (完整机械能), 与 cruise/descent 统一
- XY 惩罚: 差分形式 `(prev_dtf - dtf) × coef`, 靠近 start_xy → 正, 远离 → 负
- 成功判定: z 双边 ±25mm + |vz|<5cm/s + swing_energy<50mJ + hold_steps=3
- WandB 分项: `z_approach`, `xy_drift_reward`, `swing_energy_penalty`, `swing_energy_J`

### 2.2 Phase 2: Cruise — ORCA Expert + 双 RL

**原架构**: 单 RL agent, 多目标 reward (导航+防摆冲突)

**v4 架构**:
```
ORCA Expert (orca_expert.py):
  - ORCAPlanner: 基于速度障碍半平面约束的 2D 速度规划
  - CruiseORCAExpert: ORCA + 防摆阻尼 (用于 BC 数据收集)
  - 替代原 tracker-based expert, 更真实的避障行为

CruiseDualRLAgent:
  - planner_agent: PPO/SAC, 2D xy acc, 主用导航 reward
    reward: pbrs_nav (PBRS 势能差分) + obs_repulsion + 轻量防摆
  - swing_agent:   PPO/SAC, 2D xy acc (残差, 更小 acc_max)
    reward: 摆动能量惩罚 + 摆动速度惩罚 (无导航 reward)
  - total_acc = planner_acc + clip(swing_acc, ±swing_acc_max)
```

**两网络 Reward 分离设计**:
| Agent | 主要 Reward | 不包含 |
|-------|------------|--------|
| planner | PBRS导航 + 避障 + 轻量防摆 | 不强调防摆 |
| swing_rl | 摆动能量惩罚 + 摆动速度 | 无导航目标 |

**优势**: reward 信号更清晰, 避免多目标梯度冲突; 可独立诊断各功能学习效果

### 2.3 Phase 3: Descent — 残差 RL (不变 + 增强日志)

框架与 v3 相同 (PID base + RL residual delta_q), 新增 WandB 分项追踪:
- `xy_align_reward`, `z_descent_reward`, `swing_ke_penalty`
- `tilt_penalty`, `yaw_penalty`, `step_penalty`, `success_bonus`
- `xy_dist_mm`, `z_dist_mm`, `swing_ke_J`

## 3. ORCA 算法说明

### orca_expert.py

**ORCAPlanner** (2D 静态障碍物版):
```python
# 核心流程
for obstacle in obstacles:
    # 构建速度障碍 VO (Velocity Obstacle)
    # 计算到 VO 边界的最短修正向量 u
    # 添加半平面约束: n · v >= n·(vel + u)
# 迭代投影求满足所有约束的最近 v_pref
v_opt = LP_solve(half_planes, v_pref, max_speed)
```

**CruiseORCAExpert**:
```python
acc = k_nav * (v_orca - v_pl) + k_damp * (v_pl - v_ee)
#     导航项: 跟踪 ORCA 目标速度    防摆项: EE 跟随 payload
```

### BC 数据收集 (pretrain_bc_cruise)
- 使用 ORCA expert 代替原 tracker-based expert
- planner_agent 接受 ORCA acc 作为 BC 标签
- swing_agent BC 标签为 0 (从零防摆残差开始)

## 4. WandB 分项 Reward 曲线

### RewardComponentTracker (phase_reward.py)

每个 episode 结束后汇总并上报以下分项:

**Lift**:
- `lift/rew/z_approach` — z 接近主导奖励
- `lift/rew/xy_drift_reward` — XY 差分奖励
- `lift/rew/swing_energy_penalty` — 摆动能量惩罚
- `lift/rew/swing_energy_J` — 摆动能量监控 (J)
- `lift/rew/step_penalty` — 步惩罚
- `lift/rew/success_bonus` — 成功奖励
- 以上均有 `_per_step` 版本 (每步平均)

**Cruise (planner)**:
- `cruise/rew/pbrs_nav` — PBRS 导航信号
- `cruise/rew/obs_repulsion` — 障碍物排斥
- `cruise/rew/swing_penalty_planner` — 轻量防摆
- `cruise/rew/near_goal_bonus` — 近目标奖励
- `cruise/rew/milestone_bonus` — 里程碑

**Cruise (swing_rl)**:
- `cruise_swing/rew/swing_energy_penalty` — 防摆主导
- `cruise_swing/rew/swing_ke_J` — 动能监控
- `cruise_swing/rew/swing_pe_J` — 势能监控
- `cruise_swing/rew/swing_angle_deg` — 摆角监控
- `cruise_swing/rew/swing_vel_penalty` — 摆速惩罚

**Descent**:
- `descent/rew/xy_align_reward` — XY 对准差分
- `descent/rew/z_descent_reward` — Z 下降差分
- `descent/rew/swing_ke_penalty` — 摆动动能惩罚
- `descent/rew/tilt_penalty` — 姿态惩罚
- `descent/rew/yaw_penalty` — 偏航惩罚
- `descent/rew/xy_dist_mm` — XY 距离监控 (mm)
- `descent/rew/z_dist_mm` — Z 高度监控 (mm)
- `descent/rew/precision_bonus` — 成功精准度奖励

## 5. 训练命令

```bash
# Phase 1: Lift (PPO / SAC)
python train_phase.py --phase lift --algo ppo --log-dir saves/lift_ppo
python train_phase.py --phase lift --algo sac --log-dir saves/lift_sac

# Phase 2: Cruise (Residual RL, PPO / SAC)
python train_phase.py --phase cruise --algo ppo --log-dir saves/cruise_residual

# Phase 3: Descent (PID + 残差 RL)
python train_phase.py --phase descent --algo ppo --log-dir saves/descent_ppo
python train_phase.py --phase descent --algo sac --log-dir saves/descent_sac
```

## 6. 测试命令

```bash
# 单阶段专家 (cruise 使用 ORCA expert)
python test_phase.py --phase lift    --algo expert --render
python test_phase.py --phase cruise --algo expert --render --obstacles 3 --episodes 20
python test_phase.py --phase descent --algo expert --render

# 单阶段 RL 测试
python test_phase.py --phase lift --algo ppo --ckpt saves/lift_ppo/ckpt_best.pt --render
python test_phase.py --phase cruise --algo ppo --ckpt saves/cruise_residual/ckpt_best.pt --render --obstacles 3
python test_phase.py --phase descent --algo ppo --ckpt saves/descent_ppo/ckpt_best.pt --render

# 有风测试
python test_phase.py --phase cruise --algo expert \
    --obstacles 3 --episodes 20 \
    --wind-force 2.0 --render

# 完整流水线
python test_phase.py --phase pipeline \
    --lift-ckpt   saves/lift_ppo/ckpt_best.pt \
    --cruise-ckpt saves/cruise_residual/ckpt_best.pt \
    --descent-ckpt saves/descent_ppo/ckpt_best.pt \
    --lift-algo ppo --cruise-algo ppo --descent-algo ppo --render
```

## 7. 文件结构 v4

```
project/
├── config.py               # 全局配置 (新增: cruise_rl.use_dual_rl)
├── controller.py           # Base Controller (不修改)
├── mujoco_env_new.py       # MuJoCo 仿真环境 (不修改)
├── ee_acc_controller.py    # EE 加速度控制器 (不修改)
├── orca_expert.py          # ★ NEW: ORCA 路径规划 + 防摆 Expert
├── phase_agent.py          # ★ 新增 CruiseDualRLAgent
├── phase_reward.py         # ★ 新增 RewardComponentTracker + tracked reward 函数
├── train_phase.py          # ★ 支持双 RL Cruise + ORCA BC + 详细 wandb 日志
├── test_phase.py           # ★ 支持 CruiseDualRLAgent + ORCA expert 测试
└── Architecture.md         # ★ 本文档 v4
```

## 8. 关键设计决策 v4

### 8.1 为什么 Cruise 用双 RL 而不是单 RL 多目标?

| 方式 | 问题 | v4 解决方案 |
|------|------|-------------|
| 单 RL, 导航+防摆 reward | 梯度方向冲突, reward 量级需精细调参 | 分离网络, 独立 reward 信号 |
| 单 RL, 只有导航 | 忽略防摆, 运动可能导致摆动失控 | swing_rl 专门处理防摆 |
| 双 RL 叠加 | 两者可能互相干扰 | swing_acc 限幅保证主导权在 planner |

### 8.2 为什么 ORCA 替代原 tracker-based expert?

1. **物理一致性**: ORCA 基于速度约束, 输出的 acc 物理意义更清晰
2. **障碍物感知**: 原 expert 不感知障碍物; ORCA 显式绕障
3. **防摆**: ORCA expert 内置阻尼控制, BC 数据质量更高
4. **可扩展**: ORCA 参数化, 易于调整探索/保守程度

### 8.3 WandB 分项曲线的分析方法

```
诊断 Lift SR 低:
  z_approach 高但 SR 低 → 检查 xy_drift_reward (是否 XY 漂移)
  z_approach 低 → 检查 swing_energy_J (摆动过大阻碍上升)

诊断 Cruise SR 低:
  pbrs_nav 高但 SR 低 → 检查 obs_repulsion (碰撞?)
  planner SR 高但 swing 仍大 → swing_rl 学习失败, 检查 swing_energy_J

诊断 Descent SR 低:
  xy_align 高但 z_descent 低 → 对准不足 30mm, 检查 xy_dist_mm
  z_descent 高但 SR 低 → 检查 tilt_penalty + yaw_penalty (姿态问题)
```