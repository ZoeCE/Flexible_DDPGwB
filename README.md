残差分层强化学习框架 — 完整使用手册
本手册覆盖基于 PPO 的底层防摆 RL 控制器和高层残差 Planner 的训练、测试与评估。

目录
1. 概述

2. 安装与依赖

3. 配置说明

4. 训练指令

4.1 底层防摆控制器 (Swing Controller)

4.2 高层残差 Planner (Layered PPO)

4.3 端到端 PPO 训练（无分层）

5. 测试指令

5.1 测试底层控制器 (swing-only)

5.2 测试分层策略 (layered)

5.3 测试 NMPC 专家基准

5.4 测试端到端 PPO / TD3

5.5 手动控制

6. 风力扰动专项测试

7. 完整训练流程建议

8. 常见问题

1. 概述
本项目实现残差分层强化学习，用于索驱动机器人高精度钢筋插入任务：

底层防摆 RL 控制器 (swing_controller.py)：接收局部观测（无全局地图），输出 Δq_base，负责路径跟踪与 payload 防摆。

高层残差 Planner (PPOlearn.py + agent.py)：接收全局观测（54D）并叠加底层输出，输出残差 Δq_res，合成最终动作 Δq = Δq_base + α·Δq_res，负责避障与精细插入。

环境 (mujoco_env_new.py)：带缓慢连续变化的风力扰动，支持课程学习。

测试脚本 (test.py)：支持多种模式，自动输出防摆指标并与 NMPC 专家对比。

2. 安装与依赖
# 核心依赖
pip install torch numpy scipy mujoco
# 控制器依赖（可选，用于 NMPC 专家）
pip install casadi
# 日志与可视化（可选）
pip install tensorboard wandb
首次运行前会自动生成绳索模型（assets/generate_four_cables_with_plate.py），确保 assets/ 目录完整。

3. 配置说明
所有超参数集中在 config.py，关键新增配置节：

wind ：风力扰动参数（F_max, theta_rate_std, force_rate_std, curriculum_start/end）

swing_controller ：底层控制器的网络结构、PPO 参数

swing_controller_reward ：底层专属奖励系数

residual ：残差缩放系数 alpha 及 BC 零目标模式

修改配置时可直接编辑 config.py 或通过命令行覆盖部分参数（如 --wind-fmax）。

4. 训练指令
4.1 底层防摆控制器 (Swing Controller)
# 完整训练（推荐）—— 包含 BC 暖启动 + PPO，风力逐步增强
python train_swing_controller.py --log-dir saves/swing_ctrl --timesteps 1000000

# 跳过 BC 暖启动（直接 PPO）
python train_swing_controller.py --no-bc --log-dir saves/swing_ctrl

# 自定义最大风力
python train_swing_controller.py --wind-fmax 2.5 --log-dir saves/swing_ctrl_highwind

# 指定 GPU
python train_swing_controller.py --gpu 1 --log-dir saves/swing_ctrl
输出文件 (均保存在 --log-dir 下)：

ckpt_bc.pt : BC 后保存，可作暖启动结果

ckpt_best.pt : 基于 tilt RMS 最小的最优模型

ckpt_final.pt : 训练结束时最终模型

swing_train_log.csv : 包含 tilt_rms, swing_vel, policy_loss 等

4.2 高层残差 Planner (Layered PPO)
前提：已训练并保存底层控制器 (ckpt_best.pt)。

# 标准分层残差训练（底层冻结，只训练高层）
python PPOlearn.py --algo ppo \
  --log-dir saves/ppo_layered \
  --swing-ckpt saves/swing_ctrl/ckpt_best.pt \
  --timesteps 5000000

# 如果想继续训练已有 Planner
python PPOlearn.py --algo ppo \
  --log-dir saves/ppo_layered \
  --swing-ckpt saves/swing_ctrl/ckpt_best.pt \
  --bc-ckpt saves/ppo_layered/ckpt_bc_pretrained.pt
说明：--swing-ckpt 一旦指定，训练自动进入分层模式；否则仍为原来的端到端 PPO。

输出：

ckpt_best.pt 等，保存的是 高层 Planner 的权重（与底层的 SwingControllerAgent 分离）。

CSV 日志同原有格式。

4.3 端到端 PPO 训练（无分层）
# 不加载底层控制器，训练原始 54D 输入 PPO
python PPOlearn.py --algo ppo --log-dir saves/ppo_e2e
其他参数可参考原有训练流程（课程学习、BC 预训练等均不变）。

5. 测试指令
所有测试使用 test.py，通过 --mode 指定评估模式。

5.1 测试底层控制器 (swing-only)
# 基础评估（20 回合，默认风力）
python test.py --mode swing-only \
  --swing-ckpt saves/swing_ctrl/ckpt_best.pt \
  --episodes 20

# 渲染并加大风力
python test.py --mode swing-only \
  --swing-ckpt saves/swing_ctrl/ckpt_best.pt \
  --episodes 10 --render --wind-fmax 2.0

# 改变风向游走速率
python test.py --mode swing-only \
  --swing-ckpt saves/swing_ctrl/ckpt_best.pt \
  --wind-theta-std 0.4
自动对比 NMPC：测试结束后，脚本会自动运行相同场景的 NMPC 专家，并打印 tilt RMS 对比表（RL / NMPC 比值），无需额外操作。

输出指标：

成功率、碰撞率、平均步数

Tilt RMS、Max Tilt、Swing Velocity、Swing Offset、Path Error RMS

5.2 测试分层策略 (layered)
# 需要同时提供高层 Planner 和底层 Controller checkpoint
python test.py --mode layered \
  --ckpt saves/ppo_layered/ckpt_best.pt \
  --swing-ckpt saves/swing_ctrl/ckpt_best.pt \
  --episodes 20

# 渲染模式
python test.py --mode layered \
  --ckpt saves/ppo_layered/ckpt_best.pt \
  --swing-ckpt saves/swing_ctrl/ckpt_best.pt \
  --render --episodes 5
参数：--alpha 可覆盖残差缩放系数（代码中暂未加入，如需可修改 test.py 添加）。

5.3 测试 NMPC 专家基准
# 默认风力 (F_max=1.0)
python test.py --mode nmpc --episodes 20

# 强风下的 NMPC
python test.py --mode nmpc --episodes 10 --wind-fmax 2.5

# 剧烈风向变化
python test.py --mode nmpc --episodes 10 --wind-theta-std 0.3

# 带渲染
python test.py --mode nmpc --episodes 5 --render
输出：每回合奖励、步数、成功/碰撞，最终汇总成功率、碰撞率、平均步数。

5.4 测试端到端 PPO / TD3
# 测试端到端 PPO
python test.py --mode ppo --ckpt saves/ppo_e2e/ckpt_best.pt --episodes 20

# 测试 TD3
python test.py --mode td3 --ckpt saves/td3_run/ckpt_best.pt
5.5 手动控制
bash
python test.py --mode manual --render
键盘方向键控制 EE 移动，空格暂停，关闭窗口退出。

6. 风力扰动专项测试
# 1. 微风 (F_max=0.5)
python test.py --mode nmpc --episodes 10 --wind-fmax 0.5

# 2. 强风 (F_max=2.5)
python test.py --mode nmpc --episodes 10 --wind-fmax 2.5

# 3. 风向剧烈变化
python test.py --mode nmpc --episodes 10 --wind-theta-std 0.4

# 4. 组合
python test.py --mode nmpc --episodes 10 --wind-fmax 2.0 --wind-theta-std 0.25
底层控制器对比：将上述命令中的 --mode nmpc 替换为 --mode swing-only --swing-ckpt saves/swing_ctrl/ckpt_best.pt，即可比较 RL 与 NMPC 的抗风性能。

7. 完整训练流程建议
# Step 1: 训练底层防摆控制器
python train_swing_controller.py --log-dir saves/swing_ctrl --timesteps 1000000

# Step 2: 评估底层防摆性能（可对比 NMPC）
python test.py --mode swing-only --swing-ckpt saves/swing_ctrl/ckpt_best.pt --episodes 30

# Step 3: 训练高层残差 Planner
python PPOlearn.py --algo ppo --log-dir saves/ppo_layered \
  --swing-ckpt saves/swing_ctrl/ckpt_best.pt --timesteps 5000000

# Step 4: 测试完整分层策略
python test.py --mode layered \
  --ckpt saves/ppo_layered/ckpt_best.pt \
  --swing-ckpt saves/swing_ctrl/ckpt_best.pt \
  --episodes 30 --render
8. 常见问题
Q: 训练时 GPU 内存不足？
A: 可减小 n_steps (如 1024) 或 batch_size (256 → 128)，或减小 hidden_dim。

Q: 底层控制器训练不稳定，reward 不升？
A: 尝试降低 lr_actor (如 2e-4)，增加 BC 预训练轮数，或检查风力课程是否过早开启。

Q: 分层 Planner 残差输出很小，几乎无修正？
A: 确认 bc_target_zero=True 且 BC 权重 (bc_coef_init_residual) 不要过大（建议 0.3），同时 residual_scale 初始值 0.1 是合理的。适当增加 entropy_coef 也可促进探索。

Q: 测试时找不到 checkpoint？
A: 将 checkpoint 路径写全，或放入默认的 saves/ 目录下；test.py 会自动查找 ckpt_latest.pt / ckpt_best.pt，也可通过 --ckpt 和 --swing-ckpt 明确指定。

Q: 想用 NMPC 模式评估防摆指标？
A: 目前 NMPC 模式仅输出总奖励、成功率等，如需详细 tilt RMS，可修改 test.py 中对应部分，或使用 swing-only 模式并加载一个未训练的底层控制器（随机动作）作为对比，但不建议。更好的方式是在 test.py 的 NMPC 分支中添加指标记录。