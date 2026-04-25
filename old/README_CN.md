# 基于基座控制器的长时域稀疏奖励机械臂任务学习

本文档为项目 **BCLearning_with_nmpc** 的详细中文说明，涵盖每个代码文件的功能与主要实现逻辑。

# PPO 训练
python PPOlearn.py --algo ppo --log-dir saves/ppo_run --timesteps 5000000

# TD3 训练（对比实验）
python PPOlearn.py --algo td3 --log-dir saves/td3_run --episodes 8000

# 测试 PPO 策略
python test.py --mode ppo --log-dir saves/ppo_run --episodes 20 --render

# 测试专家控制器（NMPC 基准）
python test.py --mode nmpc --episodes 20 --render

# 手动控制（调试）
python test.py --mode manual --obstacles 3

---

## 一、项目概述

本项目实现论文 **"Learning of Long-Horizon Sparse-Reward Robotic Manipulator Tasks with Base Controllers"**（TNNLS）中的方法，核心思想是：

- **基座控制器（Base Controller）**：使用 NMPC（非线性模型预测控制）作为“老师”，在稀疏奖励、长时域任务中提供稳定的基础策略。
- **强化学习智能体（DDPG + 行为克隆）**：在模仿 NMPC 的基础上，通过 Q 值引导探索，学习超越基座控制器的策略。

当前代码已从 **PyBullet + Kuka 机械臂** 迁移到 **MuJoCo + 缆索机器人（Cable Robot）** 环境，并增加了延迟、外力扰动、随机初始化等 Sim2Real 相关设置。

### 论文信息

- **标题**：Learning of Long-Horizon Sparse-Reward Robotic Manipulator Tasks with Base Controllers  
- **作者**：Guangming Wang, Minjian Xin, Wenhua Wu, Zhe Liu, Hesheng Wang  
- **期刊**：IEEE Transactions on Neural Networks and Learning Systems (TNNLS), 2022  

---

## 二、项目目录结构

```
BCLearning_with_nmpc/
├── agent.py              # 智能体：Actor/Critic 网络、经验回放、WBAgent 逻辑
├── kuka.py               # PyBullet 版 Kuka 机械臂与环境（原论文实现，当前未用）
├── learn.py              # 训练入口：环境创建、NMPC 封装、训练循环、评估与保存
├── mujoco_env.py         # MuJoCo 缆索机器人环境（含 CableRobotEnvWithObstacles 与 2D 路径规划）
├── nmpc_controller.py    # NMPC 基座控制器（含 NMPCControllerObstacles 避障版）
├── plot.py               # 读取 log.csv，绘制训练曲线（成功率、Base 使用比例等）
├── test.py               # 测试脚本：Actor / NMPC base / NMPC+障碍物 三种模式
├── IMPLEMENTATION_GUIDE.md # 实现与参数修改说明（英文）
├── README.md             # 原版英文 README
├── README_CN.md          # 本文件：详细中文说明
├── models/               # PyBullet 用 URDF/SDF 模型（Kuka、桌子、方块、杯子等）
├── assets2/              # MuJoCo 缆索机器人场景（XML、网格等）
└── saves/                # 训练日志与模型保存目录
    └── nmpc_experiment/
        └── seed_1/
            ├── log.csv       # 每 episode 的指标
            └── actor_best.pt # 验证集上表现最好的 Actor
```

---

## 三、各文件功能与实现详解

### 3.1 `agent.py` — 智能体与网络结构

本文件实现 **WBAgent（With-Base Agent）**：带基座控制器的 DDPG 智能体，包含网络、经验回放和训练逻辑。

#### 3.1.1 工具函数

| 函数 | 功能 |
|------|------|
| `opt_cuda(t, device)` | 若可用 GPU，将张量 `t` 移到 `cuda:device`，否则保持 CPU。 |
| `np_to_tensor(n, device)` | 将 NumPy 数组转为 FloatTensor 并放到指定设备。 |
| `soft_update(target, source, tau)` | 对 `target` 和 `source` 的对应参数做软更新：`target = (1-tau)*target + tau*source`，用于更新 Target 网络。 |

#### 3.1.2 `FastActor`（Actor 网络）

- **输入**：状态 `s`，维度 `state_dim`（本项目中为 10）。
- **结构**：三层全连接 `(state_dim → 256 → 256 → action_dim)`，激活 ReLU，最后一层 Tanh。
- **输出**：动作 `a = Tanh(·) * max_action`，即范围 `[-max_action, max_action]`（本项目 `max_action=0.5`）。
- **作用**：策略网络，根据当前状态输出连续二维动作（缆索机器人的 XY 加速度指令）。

#### 3.1.3 `Critic`（Q 网络）

- **输入**：状态 `state` 与动作 `action`，拼接为 `(state_dim + action_dim)` 维。
- **结构**：两层全连接 `(state_dim+action_dim → 256 → 256 → 1)`，最后一层 Sigmoid。
- **输出**：标量 Q 值（0~1），用于评估 (s,a) 的好坏。
- **作用**：在 DDPG 中用于 TD 目标与梯度，在 WBAgent 中还用于“选基座动作还是网络动作”（mixed_q）以及行为克隆时的权重。

#### 3.1.4 `ReplayBufferFast`（经验回放）

- **存储内容**：`(sta1, sta2, acts, base_acts, rews, done)`，即当前状态、下一状态、执行的动作、基座控制器动作、奖励、是否结束。
- **容量**：100000 条；采样时随机取 `batch_size`（默认 256）条。
- **方法**：
  - `store(...)`：存入一条转移，指针循环覆盖。
  - `sample_batch(batch_size)`：随机下标，返回一个字典，包含 `sta1, sta2, acts, base_acts, rews, done`。

**设计要点**：同时存 `acts` 和 `base_acts`，以便训练时做“只在基座比网络好时模仿基座”的加权行为克隆。

#### 3.1.5 `WBAgent` 类

**初始化参数（部分）**：

- `state_dim=10`, `action_dim=2`, `max_action=0.5`：与缆索环境一致。
- `mixed_q`：是否用 Q 值比较“基座动作”与“网络动作”，决定执行谁。
- `base_boot`：Critic 的 TD 目标是否对“下一状态基座动作”的 Q 取 max，用于 Bootstrap。
- `behavior_clone`：是否在 Actor 损失中加入“模仿基座”的 BC 项。
- `base_controller_func`：基座控制器函数，输入状态（或状态列表），输出动作（或动作数组），即 NMPC 的封装接口。

**内部构造**：

- 创建 Actor / Target Actor、Critic / Target Critic，并做一次 `soft_update(·,·,1)` 使 target 与主网络一致。
- Adam 优化器，学习率 1e-3；折扣 `gamma=0.99`，软更新系数 `tau=0.005`；探索率 `epsilon` 从 1 线性衰减，步长 `delta=5e-6`，下限 0.1。
- **`lmbda=0.5`**：Actor 总损失中“Q 值项”的权重，即 `La = Lbc - lmbda * Q(s,a)`，使策略在模仿基座的同时更追求高 Q 值。

**`act(self, s, test=False)`**：

1. 若有基座控制器，先算 `action_b = self.base(s)`。
2. 用当前 Actor 得到 `action_net`。
3. 若 `test=True`：直接返回 `(action_net, True, action_b)`，用于评估。
4. 否则按 `epsilon` 做探索：
   - 以概率 `epsilon` 执行基座动作 `action_b`，返回 `(action_b, False, action_b)`。
   - 以概率 `1-epsilon`：若 `mixed_q` 为真，比较 `Q(s, action_b)` 与 `Q(s, action_net)`；若基座 Q 更高则仍执行基座动作；否则执行 `action_net`，返回 `(action_net, True, action_b)`。
5. 每次调用后 `epsilon = max(epsilon - delta, 0.1)`。

**`remember(...)`**：将一条转移存入 ReplayBuffer（包含当前状态、实际执行动作、基座动作、下一状态、奖励、done）。

**`train(self, frame)`**（每轮调用 2 个梯度步）：

1. **Critic 更新**：
   - 从 buffer 采样 batch。
   - 用 target_actor 得到下一状态动作 `a_next`，计算 `back_up = target_critic(sn, a_next)`。
   - 若 `base_boot`，再算 `back_up_d = target_critic(sn, base_action_n)`，取 `back_up = max(back_up, back_up_d)`。
   - TD 目标：`yi = ri + (1-d)*gamma*back_up`。
   - 损失：MSE(`critic(si, ai)`, `yi`)，反向传播并更新 Critic，再对 target_critic 做软更新。

2. **Actor 更新**：
   - 若 `behavior_clone`：
     - 计算 `q_base = critic(si, base_action)`，`a = actor(si)`，`q_a = critic(si, a)`。
     - 权重 `xi = ReLU(sign(q_base - q_a))`：只在“基座比当前策略好”的样本上模仿。
     - BC 损失：`Lbc = mean((a - base_action)^2 * xi) / max(sum(xi), 1)`。
     - 总损失：`La = Lbc - lmbda * mean(q_a)`，即模仿基座 + 最大化 Q。
   - 否则：`La = -mean(critic(si, actor(si)))`，标准 DDPG。
   - 反向传播更新 Actor，再对 target_actor 做软更新。

返回值：平均的 Critic 损失、Actor 损失、BC 损失（用于打日志）。

---

### 3.2 `kuka.py` — PyBullet 机械臂环境（原版）

此文件为**原论文**使用的 PyBullet + Kuka 机械臂实现，当前 `learn.py` 已改为使用 MuJoCo 的 `CableRobotEnv`，因此**训练/测试主流程不再依赖本文件**，仅作保留参考。

#### 3.2.1 `Kuka` 类

- 加载 Kuka 的 SDF 模型，设置关节限位、末端执行器初始位姿与夹爪角度。
- `getObservation()`：读取末端 link 位姿、夹爪角度与力，组成观测向量。
- `applyAction(motorCommands)`：将 (dx, dy, dz, da, df) 转为末端目标位姿与夹爪角度，通过逆运动学得到关节目标，再用 PyBullet 位置控制执行。

#### 3.2.2 `Object` 类

- 加载 URDF 物体；`reset()` 时在桌面范围内随机放置；`pos_and_euler()` 返回位置与欧拉角。

#### 3.2.3 `KukaCamEnvBase` 及子类

- 搭建场景（地面、桌子、Kuka、两个物体），支持图像观测（RGB/深度/双视角）和关节/物体状态。
- `step(action)` 将 5 维动作 (dx, dy, dz, da, df) 应用 3 个物理步，再根据子类 `reward()` 判断完成与奖励。
- **KukaCamEnv1**：绿块叠到紫块上（块对块）。
- **KukaCamEnv2**：绿块放进杯子。
- **KukaCamEnv3**：小杯放进大杯。

本仓库中 `learn.py` / `test.py` 已不调用这些类，而是使用 `mujoco_env.CableRobotEnv`。

---

### 3.3 `mujoco_env.py` — MuJoCo 缆索机器人环境

本文件实现**缆索机器人（Cable Robot）**的 MuJoCo 仿真，带控制频率、延迟、外力噪声和随机初始化，用于与 NMPC + DDPG 配合训练与测试。

#### 3.3.1 模型与参数

- **场景**：`assets2/demo_fourCable_withSteel_withSensor_cylinder.xml`，包含 mocap（控制点）、prefab（负载）、rebar_base（目标）等 body。
- **物理**：`physics_dt=0.002`（500 Hz）；控制频率 `control_freq_hz`（默认 10 Hz），`control_dt=1/control_freq_hz`，`frame_skip = control_dt / physics_dt`。
- **状态维度**：10 — 例如 mocap 位置/速度、负载位置/速度、目标相对负载的偏差等（具体以 `_get_obs()` 为准）。
- **动作维度**：2 — XY 平面加速度指令；Z 轴由环境内部根据“是否接近目标且速度小”在内部用简单规则生成（下沉或维持高度）。

#### 3.3.2 `reset()`

- 从 keyframe 0 重置数据；随机化目标位置、负载初始位置与姿态、负载初始线速度与角速度、mocap 初始位置与速度、action buffer 的初始填充（小随机动作）、预热步数（带小外力）等。
- 目的：增加初始状态多样性，减轻过拟合，更贴近 Sim2Real。

#### 3.3.3 `step(action)`

1. **动作处理**：若 `action` 为 2 维，则裁剪到 `[-action_space_high, action_space_high]`，并根据当前观测（如到目标距离、速度）在内部补全 Z 轴加速度，得到 3 维 `processed_action`。
2. **延迟**：将 `processed_action` 压入 `action_buffer`，执行时取的是**最早未执行的那条**（即延迟 `latency_steps` 步）。
3. **动力学**：用延迟后的加速度对 mocap 速度积分，再对位置积分；对速度和位置做裁剪，防止发散；将 mocap 位置写回 `data.mocap_pos`。
4. **物理**：执行 `frame_skip` 次 `mj_step`，每步可加外力噪声（Sim2Real），然后清空外力。
5. **观测与奖励**：`_get_obs()` 得到 10 维状态；`_compute_reward(obs)` 得到 reward、done、success（例如目标距离、速度、高度条件满足则 success=True，reward 加 1）。
6. 若步数达到 `max_steps` 则强制 `done=True`。

#### 3.3.4 `_get_obs()`

- 从 `current_mocap_pos/vel`、`data.qpos/qvel`（负载）、`target_pos` 等组装 10 维向量，例如：mocap 的 x,y,vx,vy，负载的 x,y,vx,vy，以及目标相对负载的 tx, ty。具体顺序以代码为准。
- 类型：`np.float32`。

#### 3.3.5 `_compute_reward(obs)`

- 根据目标距离、负载速度、高度等判断是否“到达且稳定”，给出稀疏奖励（如成功 +1，否则小惩罚）和 `success` 布尔。
- 与 NMPC 的目标定义一致，便于基座控制器与 RL 共用同一任务。

#### 3.3.6 `CableRobotEnvWithObstacles` 与 2D 路径规划（同文件内）

- **类**：`CableRobotEnvWithObstacles(CableRobotEnv)`，在 `mujoco_env.py` 中与基类同文件。
- **功能**：每次 `reset()` 时在起点—目标路径附近采样若干圆形障碍物，生成带静态障碍物的临时 XML 并重新加载 MuJoCo 模型；同时在 XY 平面用栅格 A\* 做 2D 避障路径规划。
- **关键参数**：`n_obstacles`、`obstacle_radius_range`、`path_width`、`payload_radius`、`planning_margin`、`planning_grid_res`、`default_start_xy`、`default_target_xy`。
- **接口**：`get_obstacles()` 返回当前 episode 障碍物列表 `(x, y, radius)`；`get_planned_path()` 返回 2D 规划路径 `(N, 2)` 或 `None`。障碍物圆心与起点/终点的最小距离会考虑负载安全半径，避免障碍过于靠近起终点。

---

### 3.4 `nmpc_controller.py` — NMPC 基座控制器

使用 **CasADi** 建立非线性模型与约束，用 **IPOPT** 求解 NMPC，为缆索机器人提供 XY 加速度指令，作为“基座控制器”。

#### 3.4.1 模型与离散化

- **状态** `x`（8 维）：mocap 位置 (p_x, p_y)、速度 (v_px, v_py)，负载位置 (q_x, q_y)、速度 (v_qx, v_qy)。
- **控制** `u`（2 维）：ax, ay（XY 加速度）。
- **连续动力学**：mocap 为双积分器；负载为二维摆动 + 阻尼（与缆长、重力相关），例如  
  `d(v_qx)/dt = -omega_sq*(q_x - p_x) - damping*v_qx`，y 同理。  
  与 `mujoco_env` 中物理一致（简化版）。
- **离散化**：4 阶 Runge-Kutta，步长 `dt=0.1`（与 10 Hz 控制一致）。

#### 3.4.2 目标与约束

- **代价**：跟踪目标位置 (P_ref)、抑制摆动 (q 与 p 的偏差)、抑制 mocap 速度、控制量正则。
- **约束**：动力学等式约束（RK4 离散）+ 初始状态等于当前观测到的状态；控制量上下界 `±u_max`（0.5）。

#### 3.4.3 接口 `get_action(state, target_pos)`

- **输入**：`state` 为 8 维（与 NMPC 状态一致），`target_pos` 为 2 维目标 (x, y)。
- **参数**：将 `state` 与 `target_pos` 拼成 `p`，作为 NLP 的参数；可选 warm-start 用上次解。
- **输出**：取解中第一段控制 `u_opt` 的 2 维，作为当前步的基座动作；若求解失败则返回零向量。

`learn.py` 中通过 `nmpc_wrapper(state_input)` 调用：从 10 维状态里取前 8 维给 NMPC，目标由全局 `env_target_pos` 传入。

#### 3.4.4 `NMPCControllerObstacles`（同文件内）

- **类**：`NMPCControllerObstacles`，在 `nmpc_controller.py` 中与 `NMPCController` 同文件。
- **用途**：带障碍物避碰的 NMPC，动力学与代价与基座版一致，额外在预测时域内对每一步、每个障碍物施加不等式约束：负载到障碍物中心距离 ≥ `r_safe = 障碍物半径 + obstacle_margin`。
- **接口**：`get_action(state, target_pos, obstacles)`，其中 `obstacles` 为 `list of (x, y, radius)`。不足 `n_obstacles_max` 时用 `(0,0,0)` 填充，`r_safe=0` 表示该槽位不约束。

---

### 3.5 `learn.py` — 训练入口

#### 3.5.1 `nmpc_wrapper(state_input)`

- 兼容单状态与批量状态：若 `state_input` 为一维则转为单元素列表，对每个状态取前 8 维，调用 `nmpc_instance.get_action(nmpc_state, env_target_pos)`，再按输入是否为批量返回单个动作或动作数组。
- `nmpc_instance` 与 `env_target_pos` 在 `train()` 里设置，保证每 episode 目标一致。

#### 3.5.2 `evaluate_policy(env, agent, n_eval_episodes=100)`

- 不探索（`agent.act(state, test=True)`），运行 `n_eval_episodes` 个 episode，每 episode 最多 200 步，统计成功次数，返回成功率。

#### 3.5.3 `train(log_dir, seed=0)`

1. **固定随机种子**：`torch`、`np`。
2. **环境**：创建 `CableRobotEnv`（训练与评估各一个），参数包括 `latency_steps=1`、`force_noise_level=0.08`、`control_freq_hz=10`、`init_velocity_scale=0.08`、`init_position_range=0.06` 等（相对 IMPLEMENTATION_GUIDE 略保守，提高稳定性）。
3. **基座与智能体**：实例化 `NMPCController()` 和 `WBAgent(..., base_controller_func=nmpc_wrapper)`，`mixed_q=True`, `base_boot=True`, `behavior_clone=True`。
4. **日志**：在 `log_dir` 下创建 `log.csv`，表头包含 episode、frames、train_return、train_success、test_success_rate、ratio、epsilon。
5. **主循环**：
   - 每 episode：`env.reset()`，记录 `env_target_pos`；循环内 `agent.act(state)` 得到动作（可能来自基座或网络），加小高斯噪声后裁剪，`env.step(action_exec)`，`agent.remember(...)`，`agent.train(2)`；直到 done 或步数达到 150。
   - 每 30 个 episode 做一次评估（100 个 test episode），若当前成功率超过历史最佳则保存 `actor_best.pt`。
   - 每 10 个 episode 打印当前 episode 回报、基座使用比例、epsilon。
6. **结束**：若存在 `actor_best.pt`，加载最佳 Actor 再跑 1000 个 test episode，输出最终成功率。

运行方式：`python learn.py --seed 1`，日志与模型保存在 `saves/nmpc_experiment/seed_1/`。

---

### 3.6 `plot.py` — 训练曲线绘制

- **数据**：从 `root_dir` 下所有 `seed_*/log.csv` 读取，提取 `frames, train_success, test_success_rate, ratio`。
- **处理**：按 `frames` 对齐到公共 X 轴（插值）；对 train_success、ratio 做滑动平均（窗口可配置）；test_success_rate 不平滑（已是评估均值）。
- **绘图**：三个子图 — 训练成功率、测试成功率、Base 使用比例；若为成功率图可画一条 Base Controller 基线（如 52.7%）；输出均值 ± 标准差。
- **保存**：`root_dir/paper_result.png`，dpi=300。

使用：`python plot.py --dir saves/nmpc_experiment`。

---

### 3.7 `test.py` — 测试脚本

- **模式**：`--mode actor`、`--mode base` 或 `--mode obstacles`。  
  - **actor**：加载指定目录下的 `actor_best.pt`（若无则 `actor.pt`），用 `FastActor` 前向得到动作，每步 `env.step(action)`。  
  - **base**：不加载网络，仅用 `NMPCController().get_action(obs[:8], target_pos)` 作为动作。  
  - **obstacles**：使用 `CableRobotEnvWithObstacles` 与 `NMPCControllerObstacles`，路径上随机生成障碍物，NMPC 带避障约束；可统计成功率与碰撞次数，并可选择将每局 2D 规划路径导出为 CSV。
- **环境**：actor/base 使用 `CableRobotEnv`；obstacles 使用 `CableRobotEnvWithObstacles`（同文件内），参数含延迟与扰动。
- **障碍物模式专用参数**：`--obstacles`（每局障碍物数量）、`--seed`（障碍物随机种子）、`--save_paths_dir`（导出规划路径 CSV 的目录）、`--payload_radius`、`--planning_margin`、`--planning_grid_res`。
- **统计**：运行 `n_episodes`（默认 10），统计成功次数、平均成功时步数、总耗时；obstacles 模式额外统计碰撞次数。

使用示例：  
- 测试 Actor：`python test.py --mode actor --dir saves/nmpc_experiment/seed_1 --episodes 100`  
- 测试纯 NMPC：`python test.py --mode base --episodes 100`  
- 测试 NMPC+障碍物：`python test.py --mode obstacles --episodes 10 --obstacles 3 --render`  
- 导出规划路径：`python test.py --mode obstacles --save_paths_dir saves/paths --episodes 5`  
- 可视化：任意模式加 `--render`。

---

## 四、数据流与训练流程概览

1. **环境**：`CableRobotEnv` 提供 10 维状态、2 维动作（或内部补全为 3 维）、稀疏奖励与 success 信号；reset 时随机化目标、负载与 mocap 状态；step 时带延迟与外力。
2. **基座**：NMPC 根据当前 8 维状态与目标位置输出 2 维加速度，经 `nmpc_wrapper` 传给 Agent。
3. **Agent**：以 ε 概率用基座动作、否则用网络动作（且可被 Q 比较否决）；转移存入 buffer；每步训练 2 次（Critic TD + Actor BC+Q），并衰减 ε。
4. **评估与保存**：每 30 个 episode 评估 100 次，保存最佳 Actor 为 `actor_best.pt`；训练结束后用最佳模型再评估 1000 次得到论文表格用成功率。

---

## 五、依赖与运行

### 5.1 依赖

- Python 3.x（建议 3.8+）
- PyTorch（含 CUDA 可选）
- MuJoCo（`mujoco`）
- CasADi（`casadi`）
- NumPy、pandas、matplotlib
- rich（进度条，`learn.py`）
- 原论文环境还用到 PyBullet（仅当使用 `kuka.py` 时）

### 5.2 常用命令

```bash
# 训练（默认 seed=1，日志在 saves/nmpc_experiment/seed_1/）
python learn.py --seed 1

# 测试训练好的 Actor（100 个 episode）
python test.py --mode actor --dir saves/nmpc_experiment/seed_1 --episodes 100

# 仅测试 NMPC 基座
python test.py --mode base --episodes 100

# 测试 NMPC + 障碍物避碰（带渲染）
python test.py --mode obstacles --episodes 10 --obstacles 3 --render

# 障碍物模式并导出规划路径 CSV
python test.py --mode obstacles --save_paths_dir saves/paths --episodes 5

# 绘制训练曲线（多 seed 时会在同一目录下找 seed_*/log.csv）
python plot.py --dir saves/nmpc_experiment
```

### 5.3 `test.py` 完整命令行参数与示例

`test.py` 支持三种模式，参数如下（未给出的选项使用默认值）。

| 参数 | 类型 | 默认值 | 适用模式 | 说明 |
|------|------|--------|----------|------|
| `--mode` | str | `actor` | 全部 | 策略：`actor` / `base` / `obstacles` |
| `--render` | flag | 关闭 | 全部 | 开启动画 |
| `--episodes` | int | 10 | 全部 | 测试 episode 数 |
| `--dir` | str | `saves/nmpc_experiment` | actor | 含 `actor.pt` 的目录 |
| `--obstacles` | int | 3 | obstacles | 每局障碍物数量 |
| `--seed` | int | 42 | obstacles | 障碍物随机种子 |
| `--save_paths_dir` | str | None | obstacles | 规划路径 CSV 输出目录 |
| `--payload_radius` | float | 0.06 | obstacles | 负载安全半径 (m) |
| `--planning_margin` | float | 0.02 | obstacles | 规划裕度 (m) |
| `--planning_grid_res` | float | 0.02 | obstacles | 规划栅格分辨率 (m) |

**最完整调用示例**（按模式各举一例，参数写全）：

```bash
# actor：加载指定目录模型，100 局，开渲染
python test.py --mode actor --dir saves/nmpc_experiment/seed_1 --episodes 100 --render

# base：纯 NMPC，50 局，开渲染
python test.py --mode base --episodes 50 --render

# obstacles：障碍物模式，参数写全（20 局、5 个障碍、种子 123、导出路径、规划参数、开渲染）
python test.py --mode obstacles --episodes 20 --obstacles 5 --seed 123 \
  --save_paths_dir saves/planned_paths \
  --payload_radius 0.06 --planning_margin 0.02 --planning_grid_res 0.02 \
  --render
```

仅需默认行为时，可简写，例如：  
`python test.py --mode base`、`python test.py --mode obstacles --render`。

---

## 六、障碍物环境与 2D 路径规划

本节内容合并自原 `README_obstacles.md`，说明带障碍物的环境与 NMPC 避障测试的用法。

### 8.1 障碍物建模与 XML

- 基础场景为 `assets2/demo_fourCable_withSteel_withSensor_cylinder.xml`。`CableRobotEnvWithObstacles` 在每次 `reset()` 时：
  - 在起点—目标路径附近调用 `_sample_obstacles_on_path(...)` 采样若干**圆形障碍物** `(ox, oy, r)`，半径在 `obstacle_radius_range` 内；障碍物圆心与起、终点的距离不少于 `payload_radius + planning_margin + r`，避免过于靠近起终点。
  - 调用 `_build_xml_with_obstacles(...)` 在 worldbody 中插入静态圆柱障碍物（与 `rebar_base` 同风格），并将 `rebar_base` 的 xy 设为当前目标点；临时 XML 写入 `assets2/` 后加载，用毕删除。
- 障碍物在 MuJoCo 中为真实 3D 圆柱碰撞体，XY 投影为圆盘。

### 8.2 2D 路径规划与安全半径

- **payload_radius**：负载在 XY 平面的等效安全半径。**planning_margin**：规划层额外裕度。膨胀半径 \(r_{\text{eff}} = r_{\text{obstacle}} + \text{payload\_radius} + \text{planning\_margin}\)。
- **plan_path_2d(start_xy, target_xy, obstacles, ...)**：栅格 A\*，8 邻接；起终点投影到最近可通行格；失败时退化为直线。
- NMPC 避障约束：\((q_x - o_x)^2 + (q_y - o_y)^2 \ge r_{\text{safe}}^2\)，\(r_{\text{safe}} = r_{\text{obstacle}} + \text{obstacle\_margin}\)。通常 `planning_margin >= obstacle_margin`，规划更保守。

### 8.3 环境接口与测试

- **get_obstacles()**：当前 episode 障碍物列表 `(x, y, radius)`。**get_planned_path()**：当前 2D 规划路径 `(N, 2)` 或 `None`。
- 测试：`python test.py --mode obstacles --episodes 10 --obstacles 3`；加 `--save_paths_dir DIR` 可把每局规划路径导出为 `path_ep1.csv` 等（列：idx, x, y）。
- 可调参数：`--payload_radius`、`--planning_margin`、`--planning_grid_res`（以及环境侧的 `default_start_xy` / `default_target_xy` 需在代码中传入，见 `run_test_obstacles`）。

---

## 七、引用

若在研究中使用了本代码或原论文方法，请引用：

```bibtex
@article{wang2022learning,
  title={Learning of Long-Horizon Sparse-Reward Robotic Manipulator Tasks With Base Controllers},
  author={Wang, Guangming and Xin, Minjian and Wu, Wenhua and Liu, Zhe and Wang, Hesheng},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022},
  publisher={IEEE}
}
```

---

## 八、与实现指南的对应关系

更细的参数含义、Frame Skip / Action Buffer 原理、随机化设计及调参建议见 **`IMPLEMENTATION_GUIDE.md`**。本 README_CN 侧重“每个文件做什么、关键函数与数据流”，便于快速理解整体和定位到具体代码。
