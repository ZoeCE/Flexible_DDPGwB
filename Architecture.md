# 三阶段 RL 控制架构设计文档

## 1. 架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                    三阶段分治控制系统                              │
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐  │
│  │  Phase 1    │    │  Phase 2     │    │  Phase 3           │  │
│  │  Lift RL    │───>│  Cruise RL   │───>│  Descent RL        │  │
│  │  (提升)     │    │  (平移避障)  │    │  (精准下降插入)    │  │
│  └──────┬──────┘    └──────┬───────┘    └─────┬──────────────┘  │
│         │                  │                   │                 │
│    EE acc(3D)         EE acc(2D)          EE acc(3D)            │
│    ax, ay, az         ax, ay (z锁)       ax, ay, az            │
│         │                  │                   │                 │
│         ▼                  ▼                   ▼                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              EE Acceleration Controller                  │   │
│  │  acc → 积分 → EE vel/pos → IK → q_target → delta_q     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│                     ┌─────────────┐                              │
│                     │  MuJoCo Env │                              │
│                     └──────┬──────┘                              │
│                            │                                     │
│                            ▼                                     │
│                  ┌──────────────────┐                            │
│                  │ Phase Rewards    │                            │
│                  │ (各阶段独立)     │                            │
│                  └──────────────────┘                            │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           固定阶段切换逻辑 (Phase Transition)            │   │
│  │  Lift→Cruise: z≥阈值 & tilt<阈值 & swing_vel<阈值      │   │
│  │  Cruise→Descent: xy_dist<阈值 & tilt<阈值 & vel<阈值   │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## 2. 三阶段详细设计

### 2.1 Phase 1: Lift RL (提升阶段)

**目标**: 将吊装物从起始位置安全提升到巡航高度

**初始化**:
- 标准起点, 在 `default_start_xy ± init_xy_range` 范围内随机化
- 无初始速度

**RL 输出**: EE 3D 加速度 `(ax, ay, az)`
- 经 EEAccController 积分 → IK → 关节角

**观测** (23D):
| 维度 | 内容 | 说明 |
|------|------|------|
| 3 | ee_pos | 末端位置 |
| 3 | ee_vel | 末端速度 |
| 3 | pl_pos | 吊装物位置 |
| 3 | pl_vel | 吊装物速度 |
| 3 | ee_pl_offset | 末端-吊装物偏移 |
| 4 | tilt, yaw, tilt_rate, yaw_rate | 姿态及变化率 |
| 2 | start_xy | 起点坐标 |
| 1 | target_z | 目标巡航高度 |
| 1 | z_error | 当前z与目标z的误差 |

**Reward**:
- `z_approach`: 差分奖励, 越接近巡航高度越正
- `swing_ke`: 摆动动能惩罚 = -c * 0.5m * ||v_pl - v_ee||²
- `success_bonus`: 到达巡航高度且稳定
- `step_penalty`: 每步小惩罚

**成功条件**: payload_z ≥ 阈值 AND tilt < 阈值 AND swing_vel < 阈值

### 2.2 Phase 2: Cruise RL (平移阶段)

**目标**: 在巡航高度避障到达终点上方

**初始化**:
- 从起点上方巡航高度随机化 (通过专家先提升到巡航高度)
- 小随机初始速度

**RL 输出**: EE 2D 加速度 `(ax, ay)`, Z 维锁定

**观测** (33D for n_obstacles=3):
| 维度 | 内容 |
|------|------|
| 2 | ee_xy |
| 2 | ee_vxy |
| 2 | pl_xy |
| 2 | pl_vxy |
| 2 | ee_pl_offset_xy |
| 4 | tilt, yaw, tilt_rate, yaw_rate |
| 2 | target_xy |
| 1 | target_dist |
| 9 | obstacles (3×3) |
| 7 | joint_q |

**Reward**:
- `goal_potential`: 目标吸引势能差分 (Φ = -k/(d+ε), r = Φ(t-1) - Φ(t))
- `obs_repulse`: 障碍物排斥势 (d < d0 时激活)
- `swing_ke`: 摆动动能惩罚
- `collision_penalty`: 碰撞障碍物 → 大惩罚 + reset
- `success_bonus`: 到达终点上方且稳定

**成功条件**: xy_dist < 阈值 AND tilt < 阈值 AND vel < 阈值

### 2.3 Phase 3: Descent RL (下降阶段)

**目标**: 精准下降插入钢筋

**初始化**:
- 从终点上方巡航高度随机化 (通过专家先导航到终点上方)
- 小随机初始摆动

**RL 输出**: EE 3D 加速度 `(ax, ay, az)`, 速度限制更严格

**观测** (29D):
| 维度 | 内容 |
|------|------|
| 3 | ee_pos |
| 3 | ee_vel |
| 3 | pl_pos |
| 3 | pl_vel |
| 3 | ee_pl_offset |
| 4 | tilt, yaw, tilt_rate, yaw_rate |
| 2 | target_xy |
| 1 | target_z |
| 2 | pl_target_xy_err |
| 1 | z_error |
| 4 | rebar_errors |

**Reward**:
- `xy_align`: 连续 xy 对准惩罚
- `z_descent`: 差分下降奖励 (xy 对准时才给)
- `swing_ke`: 摆动动能惩罚 (高权重)
- `tilt`: 姿态惩罚
- `success_bonus`: 插入成功
- `soft_success_bonus`: 超时时按完成度给分

**成功条件**: z/xy/tilt/yaw 全部满足容差, 保持 ≥3 步

## 3. 统一控制接口: EE 加速度控制器

```python
# EEAccController 工作流程
acc = rl_agent.act(obs)      # RL 输出加速度
ee_vel += acc * dt            # 积分得速度
ee_pos += ee_vel * dt         # 积分得位置
ee_pos = anchor(ee_pos, real) # 软锚定防漂移
q_target = IK(ee_pos)         # 逆运动学
delta_q = clip(q_target - q)  # 关节角增量
env.step(delta_q)             # 执行
```

**设计优势**:
1. 所有阶段统一控制接口, 仅 acc 范围和维度不同
2. 积分器自然保证运动连续性
3. 软锚定防止积分误差累积
4. IK 保证关节空间可行性

## 4. 阶段切换逻辑

```python
# 由环境端固定逻辑判断, 非 RL 学习
if phase == "lift":
    if payload_z >= 0.23 and tilt < 0.15 and swing_vel < 0.08:
        phase = "cruise"

elif phase == "cruise":
    if xy_dist < 0.03 and tilt < 0.10 and swing_vel < 0.05:
        phase = "descent"
```

**切换时动作**:
- 重置 reward state
- 重置 EE 加速度控制器 (清零速度, 同步位置)
- 切换到对应阶段的 RL 模型

## 5. 训练框架

### 5.1 每阶段独立训练

```bash
# Phase 1: Lift
python train_phase.py --phase lift --algo ppo --log-dir saves/lift_ppo

# Phase 2: Cruise
python train_phase.py --phase cruise --algo ppo --log-dir saves/cruise_ppo

# Phase 3: Descent
python train_phase.py --phase descent --algo ppo --log-dir saves/descent_ppo
```

### 5.2 训练流程

```
对每个阶段:
  1. BC 预训练 (可选)
     - 用 NMPC 专家收集数据
     - 从专家轨迹提取 EE 加速度标签
     - 监督学习 actor
  
  2. RL 训练 (PPO 或 SAC)
     - 阶段专属环境初始化
     - 阶段专属 reward
     - wandb 实时可视化
  
  3. 课程学习
     - 风力: 从 0 → 100% 渐进
     - 障碍物 (仅 cruise): 0 → 1 → 2 → 3 渐进
```

### 5.3 PPO vs SAC 选择建议

| 阶段 | 推荐 | 原因 |
|------|------|------|
| Lift | PPO | 简单任务, PPO 更稳定 |
| Cruise | SAC | 需要更好的探索 (避障) |
| Descent | PPO/SAC | 高精度, 需要精细调参对比 |

## 6. 测试

### 6.1 单阶段测试

```bash
python test_phase.py --phase lift --algo ppo --ckpt saves/lift_ppo/ckpt_best.pt --render
python test_phase.py --phase cruise --algo ppo --ckpt saves/cruise_ppo_01/ckpt_latest.pt --render
python test_phase.py --phase descent --algo ppo --ckpt saves/descent_ppo/ckpt_latest.pt --render
# ── MuJoCo 实时渲染 ──
python test_phase.py --phase lift    --algo expert  --render
python test_phase.py --phase cruise  --algo expert  --render --obstacles 3
python test_phase.py --phase descent --algo expert  --render
```

### 6.2 全流水线测试

```bash
python test_phase.py --phase pipeline \
    --lift-ckpt saves/lift_ppo/ckpt_best.pt \
    --cruise-ckpt saves/cruise_sac/ckpt_best.pt \
    --descent-ckpt saves/descent_ppo/ckpt_best.pt \
    --lift-algo ppo --cruise-algo sac --descent-algo ppo
```

### 6.3 专家基准

```bash
python test_phase.py --phase pipeline --algo expert
```

## 7. 文件结构

```
project/
├── config.py               # 全局配置 (三阶段 RL + 共享参数)
├── controller.py           # Base Controller (MPC+IK, 不修改)
├── mujoco_env_new.py       # MuJoCo 仿真环境 (不修改)
├── ee_acc_controller.py    # ★ EE 加速度控制器 (acc→IK→delta_q)
├── phase_agent.py          # ★ 三阶段统一 Agent (PPO+SAC)
├── phase_reward.py         # ★ 三阶段独立奖励函数
├── train_phase.py          # ★ 独立训练脚本 (BC+RL+课程)
├── test_phase.py           # ★ 测试脚本 (单阶段+流水线)
└── Architecture.md         # ★ 本文档
```

## 8. 关键设计决策

### 8.1 为什么 RL 输出加速度而非位置?

1. **运动连续性**: 加速度通过积分自然保证速度/位置连续
2. **物理直觉**: 加速度对应力, 更容易学习物理规律
3. **统一接口**: 所有阶段用相同的控制方式, 仅维度不同
4. **探索安全**: 加速度的 tanh 限幅 + 速度限幅 = 双重安全

### 8.2 为什么平移段锁死 Z?

1. **降低学习难度**: 2D 比 3D 更容易学习
2. **物理合理性**: 巡航阶段保持高度不变是工程常识
3. **防止误操作**: 避免 RL 学到"先下降再上升"的无效策略
4. **与下降段解耦**: 高度控制完全交给 descent phase

### 8.3 为什么每阶段初始化要用专家?

1. **避免分布偏移**: 如果随机初始化中间状态, 可能得到物理上不合理的配置
2. **保持绳索张力**: 直接设置 payload 位置会破坏绳索约束
3. **真实过渡**: 专家跑到的状态是真正可以从上一阶段过渡来的
4. **随机扰动**: 在专家到达后添加小扰动, 增加多样性

### 8.4 为什么删除旧模块?

| 删除 | 原因 |
|------|------|
| planner_agent.py | 被 phase_agent.py 替代 |
| residual_agent.py | 三阶段架构不需要残差微调 |
| reward.py | 被 phase_reward.py 替代 |
| train_planner.py | 被 train_phase.py 替代 |
| test_planner.py | 被 test_phase.py 替代 |
| agent.py | 旧版端到端, 不再需要 |
| PPOlearn.py | 旧版训练循环, 不再需要 |
| swing_controller.py | 底层防摆已由 EEAccController 内置 |
| train_swing_controller.py | 配套删除 |

**保留**: controller.py (NMPC+IK, 用于BC专家), mujoco_env_new.py (仿真环境)