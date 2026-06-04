# Flexible_DDPGwB - Cable-Suspended Payload RL Framework

> 当前主线: **cruise** 负责 NMPC 抬升+平移到钢筋上方, **descent** 负责 PID+residual RL 对准、下降和物理插入。
> 本 README 已合并原 `Architecture.md` 的架构说明和常用指令, 作为当前唯一主文档维护。

---

## 1. 快速开始

先进入 WSL 项目目录和 conda 环境:

```bash
cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB
conda activate vsdrl_env_5060
```

最常用的 pipeline 测试:

```bash
# cruise 用 expert/NMPC, descent 用 PPO residual RL
python test_phase.py --phase pipeline \
  --cruise-algo expert \
  --descent-algo ppo \
  --descent-ckpt saves/descent_ppo/ckpt_latest.pt \
  --episodes 10 --render --wait-for-space
```

加入可调恒定风力:

```bash
python test_phase.py --phase pipeline \
  --cruise-algo expert \
  --descent-algo ppo \
  --descent-ckpt saves/descent_ppo/ckpt_latest.pt \
  --episodes 10 --render --wait-for-space \
  --wind-speed 10 --wind-dir 0.0
```

`--wind-speed` 单位是 m/s; `--wind-dir` 是弧度。`0.0` 约为 +x 方向, `1.5708` 约为 +y 方向。不写 `--wind-dir` 时使用随机方向。

---

## 2. 当前架构

### 2.1 Cruise: NMPC 抬升 + 平移

Cruise 现在合并了原来的 lift 和水平 cruise:

```text
payload 起点: (start_xy, z ~= 0.11)
    |
    | NMPC 垂直抬升
    v
(start_xy, z_cruise ~= 0.25)
    |
    | NMPC 水平平移
    v
(target_xy, z_cruise)  # 钢筋正上方
```

核心控制路径:

```python
expert.compute_delta_q_target(obs, current_q, residual_acc=res3)
```

当 `cruise-algo expert` 时, `residual_acc=None`, 即纯 NMPC/expert。
当 `cruise-algo ppo/sac` 时, residual RL 输出小幅 3D 加速度残差, 叠加在 NMPC 输出之前。

当前 NMPC 的关键优化:

- 代价函数中加入 `U_prev` 和 jerk 惩罚, 抑制控制量第一拍突变和来回翻转。
- 平滑前瞻参考点 `ref_smoothing_alpha`, 避免 waypoint 切换导致参考点跳变。
- action smoothing + rate limit, 限制 NMPC 输出小范围高速抖动。
- 终点附近 settle deadband, 在 payload 足够接近、速度和摆动都较小时压掉微小控制量。
- pipeline 中的 cruise expert 已与单独 `--phase cruise --algo expert` 对齐, 不再使用 pipeline-only controller/ee_control 覆盖。

### 2.2 Descent: PID base + residual RL 插入

Descent 从钢筋上方开始:

```text
payload 起点: (target_xy + small noise, z_cruise)
    |
    | PID base 控制下降和对准
    | PPO/SAC residual RL 修正 xy/z
    v
物理插入钢筋
```

成功判定已从“只到达钢筋位置”改为更接近真实任务:

- xy、z、tilt、yaw 等指标在容差内;
- 并且检测到训练 reward 使用的物理插入/地面接触成功条件;
- 若 payload 卡在钢筋上且不再产生有效插入进展, 会提前判负。

这保证渲染里能看到吊装物真正插入钢筋, 而不是停在钢筋上方。

### 2.3 Pipeline

Pipeline 当前是两阶段:

```text
cruise:  NMPC/expert 或 NMPC + residual RL
handoff: 到达钢筋上方后切换
descent: PID + residual RL 或纯 expert
```

Pipeline 会禁用 3D 轨迹里的主动下降段:

```python
config["planning"]["disable_descent_segment"] = True
```

也就是说, cruise 的目标就是“稳定到达钢筋上方”, 下降和插入由 descent 阶段接管。

---

## 3. Reward 与 RL 设计

### 3.1 Cruise residual RL

Cruise residual RL 的目标不是替代 NMPC, 而是在 NMPC 出现风扰、小幅抖动、控制环效应时做小幅补偿。

观测中包含:

- 常规 EE / payload 状态;
- NMPC 当前 base action;
- 240 维绳索观测, 经过 cable encoder;
- 风力观测。

Reward 重点:

- `swing_energy_penalty`: 抑制 payload 摆动动能。
- `cable_ke_penalty`: 抑制绳索高频振动。
- `swing_improve`: 奖励每一步相对上一时刻的消摆改进。
- `action_rms_free`: 给小 residual 一个免费区间, 让 RL 有空间介入。
- `loop_counter_reward`: 当 NMPC base action 翻转/抖动时, 奖励 residual 反向抵消。
- `rel_vel_damping_reward`: 奖励 residual 阻尼 payload 和 EE 的相对速度。
- `loop_jitter_penalty`: 显式惩罚 NMPC base action 的抖动。

碰撞障碍物只给很小惩罚, 不再把 collision reward 与高度强绑定。原因是 cruise collision 多数是 base NMPC 轨迹/控制表现导致, 不希望 RL 学成“为了避免碰撞强行改变高度或破坏稳定性”。

### 3.2 Descent residual RL

Descent 的设计保持当前成功版本:

- PID 提供稳定下降和粗对准;
- residual RL 保持足够控制权威, 不随 PID 收敛而消失;
- reward 主要关注插入误差、xy 精度、z 进度、姿态稳定、物理插入;
- 当前 descent 测试表现很好, 后续默认不要轻易改 reward 和成功判定。

### 3.3 课程学习

Cruise 使用连续风力课程, 从极小风力开始逐步增加, 让 residual RL 先学“在近似无扰下不破坏 NMPC”, 再学“有风时消摆和阻尼”。

Descent 使用精度固定的课程, 重点保持最终 5mm 对准能力, 风力逐步增强。

---

## 4. 日志与 W&B 指标

为了能看出 RL 是否真的改善移动过程稳定性, 训练日志窗口已加大:

- `train.log_smooth_win = 200`
- `train.log_trend_windows = [50, 200, 500]`
- `train.log_baseline_episodes = 50`

重点关注这些指标:

```text
stab/{phase}/avg_ke_mJ
stab/{phase}/max_ke_mJ
stab/{phase}/p95_ke_mJ
stab/{phase}/integral_ke_mJs
stab/{phase}/avg_angle_deg
stab/{phase}/max_angle_deg
stab/{phase}/p95_angle_deg
stab/{phase}/cable_ke_peak
stab/{phase}/cable_ke_avg
stab/{phase}/pl_vel_peak
stab/{phase}/avg_acc
stab/{phase}/max_acc
stab/{phase}/rl_action_mag_mean
stab/{phase}/rl_action_mag_peak
```

Cruise reward 还会记录:

```text
cruise/rew/action_rms_norm
cruise/rew/loop_counter_reward
cruise/rew/rel_vel_damping_reward
cruise/rew/loop_jitter_penalty
cruise/rew/swing_energy_penalty
cruise/rew/cable_ke_penalty
```

判断 cruise residual RL 是否有效时, 不只看平均 reward, 更要看 `max/p95 swing angle`, `integral_ke_mJs`, `cable_ke_peak`, `pl_vel_peak` 是否随训练下降。

---

## 5. 常用命令

### 5.1 RL 训练

Cruise PPO:

```bash
python train_phase.py --phase cruise \
  --algo ppo \
  --n-envs 8 \
  --timesteps 2500000 \
  --log-dir saves/cruise_ppo_next
```

Descent PPO:

```bash
python train_phase.py --phase descent \
  --algo ppo \
  --n-envs 8 \
  --timesteps 3000000 \
  --log-dir saves/descent_ppo_next
```

注意: 当前成功的 descent checkpoint 在 `saves/descent_ppo/`, 不要无意中覆盖。新训练建议写入 `saves/descent_ppo_next` 或其他新目录。

### 5.2 单阶段测试

Cruise 纯 expert/NMPC:

```bash
python test_phase.py --phase cruise \
  --algo expert \
  --episodes 30 --render --wait-for-space
```

Cruise residual RL:

```bash
python test_phase.py --phase cruise \
  --algo ppo \
  --ckpt saves/cruise_ppo/ckpt_latest.pt \
  --episodes 30 --render --wait-for-space
```

Descent residual RL:

```bash
python test_phase.py --phase descent \
  --algo ppo \
  --ckpt saves/descent_ppo/ckpt_latest.pt \
  --episodes 30 --render --wait-for-space
```

Descent 纯 expert/PID:

```bash
python test_phase.py --phase descent \
  --algo expert \
  --episodes 30 --render --wait-for-space
```

### 5.3 阶段风力测试

Cruise expert 加风:

```bash
python test_phase.py --phase cruise \
  --algo expert \
  --episodes 30 --render --wait-for-space \
  --wind-speed 10 --wind-dir 0.0
```

Cruise RL 加风:

```bash
python test_phase.py --phase cruise \
  --algo ppo \
  --ckpt saves/cruise_ppo/ckpt_latest.pt \
  --episodes 30 --render --wait-for-space \
  --wind-speed 10 --wind-dir 0.0
```

Descent RL 加风:

```bash
python test_phase.py --phase descent \
  --algo ppo \
  --ckpt saves/descent_ppo/ckpt_latest.pt \
  --episodes 30 --render --wait-for-space \
  --wind-speed 10 --wind-dir 0.0
```

风力扫描:

```bash
for wind in 0 2 4 6 8 10; do
  python test_phase.py --phase descent \
    --algo ppo \
    --ckpt saves/descent_ppo/ckpt_latest.pt \
    --episodes 30 \
    --wind-speed $wind
done
```

### 5.4 Pipeline 测试

Cruise expert + descent RL:

```bash
python test_phase.py --phase pipeline \
  --cruise-algo expert \
  --descent-algo ppo \
  --descent-ckpt saves/descent_ppo/ckpt_latest.pt \
  --episodes 10 --render --wait-for-space
```

Cruise expert + descent RL + 风力:

```bash
python test_phase.py --phase pipeline \
  --cruise-algo expert \
  --descent-algo ppo \
  --descent-ckpt saves/descent_ppo/ckpt_latest.pt \
  --episodes 10 --render --wait-for-space \
  --wind-speed 10 --wind-dir 0.0
```

全程 RL, 即 cruise residual RL + descent residual RL:

```bash
python test_phase.py --phase pipeline \
  --cruise-algo ppo \
  --cruise-ckpt saves/cruise_ppo/ckpt_latest.pt \
  --descent-algo ppo \
  --descent-ckpt saves/descent_ppo/ckpt_latest.pt \
  --episodes 10 --render --wait-for-space
```

全程 RL + 风力:

```bash
python test_phase.py --phase pipeline \
  --cruise-algo ppo \
  --cruise-ckpt saves/cruise_ppo/ckpt_latest.pt \
  --descent-algo ppo \
  --descent-ckpt saves/descent_ppo/ckpt_latest.pt \
  --episodes 10 --render --wait-for-space \
  --wind-speed 10 --wind-dir 0.0
```

纯 expert 全流程 + 风力:

```bash
python test_phase.py --phase pipeline \
  --cruise-algo expert \
  --descent-algo expert \
  --episodes 10 --render --wait-for-space \
  --wind-speed 10 --wind-dir 0.0
```

Pipeline 风力扫描, cruise expert + descent RL:

```bash
for wind in 0 2 4 6 8 10; do
  python test_phase.py --phase pipeline \
    --cruise-algo expert \
    --descent-algo ppo \
    --descent-ckpt saves/descent_ppo/ckpt_latest.pt \
    --episodes 10 \
    --wind-speed $wind
done
```

Pipeline 风力扫描, 全程 RL:

```bash
for wind in 0 2 4 6 8 10; do
  python test_phase.py --phase pipeline \
    --cruise-algo ppo \
    --cruise-ckpt saves/cruise_ppo/ckpt_latest.pt \
    --descent-algo ppo \
    --descent-ckpt saves/descent_ppo/ckpt_latest.pt \
    --episodes 10 \
    --wind-speed $wind
done
```

### 5.5 可视化暂停

只有在命令中显式加入下面参数时才会暂停:

```bash
--wait-for-space
```

必须同时开 `--render`。每个 episode 初始化和第一帧同步后, 点击 MuJoCo 渲染窗口并按 `Space` 才开始执行。默认不等待, 保持原来的自动运行行为。

### 5.6 插入目标和 cruise 高度调节

Pipeline 默认使用:

```text
--pipeline-cruise-z 0.25
--pipeline-insert-target-z 0.10
--pipeline-insert-depth-min 0.025
```

测试阶段不再追加独立收尾控制段; 若要观察更深插入, 应调低训练/测试共用的目标 z, 例如:

```bash
python test_phase.py --phase pipeline \
  --cruise-algo expert \
  --descent-algo ppo \
  --descent-ckpt saves/descent_ppo/ckpt_latest.pt \
  --episodes 10 --render --wait-for-space \
  --pipeline-insert-target-z 0.09
```

---

## 6. 主要文件

```text
config.py             # 全局配置: controller/reward/curriculum/train/test
controller.py         # JointSpaceExpert, NMPCTrajectoryTracker, NMPCController4D
ee_acc_controller.py  # EE acceleration controller, cruise z/yaw PID, swing damping
mujoco_env_new.py     # MuJoCo 环境、路径生成、风力、接触、物理 step
phase_agent.py        # PPO/SAC agent, cable encoder, phase obs 构建
phase_reward.py       # cruise/descent reward 和 success 判定
train_phase.py        # 训练入口、reset_for_phase、课程、W&B logging
test_phase.py         # 单阶段测试、pipeline 测试、渲染暂停、风力测试
vec_env.py            # 并行环境, 与单环境训练路径保持一致
stability_metrics.py  # 稳定性指标统计
```

---

## 7. 最近关键修改记录

| 模块 | 修改 |
|------|------|
| `controller.py` | NMPC 加入 U_prev jerk 惩罚、参考点低通、action rate limit、终点 settle deadband。 |
| `test_phase.py` | pipeline cruise expert 与单独 cruise expert 对齐, 删除 pipeline-only controller 覆盖。 |
| `test_phase.py` | 新增 `--wait-for-space`, 渲染初始化后按空格开始。 |
| `phase_reward.py` | cruise reward 新增 base action 相关项: loop counter, relative velocity damping, loop jitter。 |
| `phase_reward.py` | descent 成功判定保留物理插入/地面接触逻辑。 |
| `phase_agent.py` | cruise 观测包含 NMPC action、绳索、风力; 支持 residual RL。 |
| `train_phase.py` | W&B 平均窗口和趋势窗口增大, 更容易观察稳定性指标改善。 |
| `vec_env.py` | 并行训练路径同步 cruise NMPC residual 和 reward 所需 base action。 |

---

## 8. 使用建议

1. 改 cruise NMPC 前, 先跑:

```bash
python test_phase.py --phase cruise --algo expert --episodes 30 --render
```

如果单段 cruise expert 表现好, pipeline 表现差, 优先检查 pipeline 是否又引入了额外 controller 覆盖或 handoff 条件过严。

2. 改 descent reward 前, 先保护当前成功 ckpt:

```bash
cp -r saves/descent_ppo saves/descent_ppo_backup
```

Descent 当前表现很好, 默认只做测试和小范围可视化参数调整。

3. 观察 RL 是否有帮助时, 不只看 SR。Cruise 尤其要看:

```text
p95/max swing angle
integral_ke_mJs
cable_ke_peak
pl_vel_peak
loop_counter_reward
rel_vel_damping_reward
```

4. 如果 VS Code/WSL 出问题, 最稳的打开方式是:

```bash
cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB
code .
```
