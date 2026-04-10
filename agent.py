import numpy as np
import torch
import torch.nn as nn
import random
import os

# 导入全局默认配置（作为 fallback 备用）
from config import DEFAULT_CONFIG

# ===========================================================================
# 工具函数
# ===========================================================================

def opt_cuda(t, device=None):
    if torch.cuda.is_available():
        if device is None:
            return t.cuda()
        return t.to(device)
    else:
        return t


def np_to_tensor(n, device=None):
    return opt_cuda(torch.from_numpy(n).type(torch.FloatTensor), device)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


# ===========================================================================
# 网络定义
# ===========================================================================

class FastActor(nn.Module):
    """
    策略网络（Actor）。
    相较于原版，去除了硬编码的 256 和 max_action=0.5。
    通过 hidden_dim 参数和从 config 读取的多维 max_action 张量进行初始化。
    """
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(FastActor, self).__init__()
        
        # 使用 register_buffer 确保护 max_action 张量能够随模型一同被 .cuda() 移动到正确设备
        if isinstance(max_action, (float, int)):
            self.register_buffer('max_action', torch.tensor(max_action, dtype=torch.float32))
        else:
            self.register_buffer('max_action', torch.tensor(max_action, dtype=torch.float32))

        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, s):
        # 逐维度乘以动作空间上限（广播机制）
        return self.fc(s) * self.max_action


class TwinCritic(nn.Module):
    """
    双重 Critic 网络（TD3 核心）。
    同时维护两个独立的 Q 网络（Q1、Q2），训练时取两者最小值计算目标，
    从根源上消除过估计问题。网络宽度由 config 中的 hidden_dim 决定。
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(TwinCritic, self).__init__()

        # Q1 网络
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 网络（结构与 Q1 完全相同，但参数独立初始化）
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, s, a):
        """返回两个 Q 值，Critic 更新时同时使用。"""
        sa = torch.cat([s, a], dim=1)
        return self.q1_net(sa), self.q2_net(sa)

    def Q1(self, s, a):
        """
        仅返回 Q1，供 Actor 梯度更新与 BC 差异计算使用。
        TD3 中 Actor 固定跟随 Q1 更新，避免引入 Q2 带来的额外方差。
        """
        sa = torch.cat([s, a], dim=1)
        return self.q1_net(sa)


# ===========================================================================
# 经验回放池
# ===========================================================================

class ReplayBuffer:
    """
    循环经验回放池，接口与原版完全一致，支持 base_action / next_base_action
    字段，以兼容 Base Bootstrapping 和 Behavior Clone 机制。
    """
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state          = np.zeros((max_size, state_dim))
        self.action         = np.zeros((max_size, action_dim))
        self.base_action    = np.zeros((max_size, action_dim))
        self.next_base_action = np.zeros((max_size, action_dim))
        self.next_state     = np.zeros((max_size, state_dim))
        self.reward         = np.zeros((max_size, 1))
        self.done           = np.zeros((max_size, 1))

    def add(self, state, action, base_action, next_base_action, next_state, reward, done):
        self.state[self.ptr]            = state
        self.action[self.ptr]           = action
        self.base_action[self.ptr]      = base_action
        self.next_base_action[self.ptr] = next_base_action
        self.next_state[self.ptr]       = next_state
        self.reward[self.ptr]           = reward
        self.done[self.ptr]             = done

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[ind],
            self.action[ind],
            self.base_action[ind],
            self.next_base_action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.done[ind],
        )


# ===========================================================================
# TD3 Agent
# ===========================================================================

class WBAgent:
    """
    TD3（Twin Delayed Deep Deterministic Policy Gradient）Agent。
    此版本已完全兼容项目全局配置 config.py，移除了所有的硬编码超参数。
    """

    def __init__(self, log_dir, state_dim, action_dim, config=None, base_controller_func=None):
        if config is None:
            config = DEFAULT_CONFIG

        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.config     = config
        
        agent_cfg = config["agent"]
        space_cfg = config["space"]

        # 动作空间上限（不再硬编码为 0.5，支持按维度定义的高低限）
        self.max_action = space_cfg["action_space_high"]

        # Agent 运行模式配置
        self.mixed_q        = agent_cfg["mixed_q"]
        self.base_boot      = agent_cfg["base_boot"]
        self.behavior_clone = agent_cfg["behavior_clone"]
        self.base_controller_func = base_controller_func

        # 网络宽度配置
        hidden_dim = agent_cfg["hidden_dim"]

        # ---------- Actor ----------
        self.actor        = opt_cuda(FastActor(state_dim, action_dim, self.max_action, hidden_dim))
        self.target_actor = opt_cuda(FastActor(state_dim, action_dim, self.max_action, hidden_dim))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=agent_cfg["lr_actor"])

        # ---------- Twin Critic ----------
        self.critic        = opt_cuda(TwinCritic(state_dim, action_dim, hidden_dim))
        self.target_critic = opt_cuda(TwinCritic(state_dim, action_dim, hidden_dim))
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=agent_cfg["lr_critic"])

        # ---------- Replay Buffer ----------
        self.buffer = ReplayBuffer(agent_cfg["buffer_size"], state_dim, action_dim)

        # ---------- 基础超参数 ----------
        self.batch_size = agent_cfg["batch_size"]
        self.gamma      = agent_cfg["gamma"]
        self.tau        = agent_cfg["tau"]

        # ---------- Epsilon 探索 ----------
        self.epsilon     = agent_cfg["epsilon_init"]
        self.epsilon_min = agent_cfg["epsilon_min"]
        self.delta       = agent_cfg["epsilon_delta"]

        # ---------- TD3 专属超参数 ----------
        self.policy_noise = agent_cfg["policy_noise"]
        self.noise_clip   = agent_cfg["noise_clip"]
        self.policy_freq  = agent_cfg["policy_freq"]

        self.total_it    = 0       # 全局训练步数计数器

        # 缓存最近一次 Actor 和 BC 的 Loss，用于延迟更新时保持外部接口稳定
        self._last_actor_loss = 0.0
        self._last_bc_loss    = 0.0

    # -----------------------------------------------------------------------
    # 动作选择
    # -----------------------------------------------------------------------

    def act(self, state):
        """
        根据 epsilon 决定使用专家（NMPC）还是 Actor 网络输出动作。
        """
        with torch.no_grad():
            s_tensor    = np_to_tensor(state.reshape(1, -1))
            actor_action = self.actor(s_tensor).cpu().data.numpy().flatten()
            base_action  = (self.base_controller_func(state)
                            if self.base_controller_func
                            else np.zeros(self.action_dim))

            if random.random() < self.epsilon:
                return base_action, False, base_action
            else:
                return actor_action, True, base_action

    # -----------------------------------------------------------------------
    # 经验存储
    # -----------------------------------------------------------------------

    def remember(self, state, action, base_action, next_base_action,
                 next_state, reward, done):
        self.buffer.add(state, action, base_action, next_base_action,
                        next_state, reward, done)

    # -----------------------------------------------------------------------
    # TD3 训练核心
    # -----------------------------------------------------------------------

    def train(self, iterations):
        """
        执行 iterations 次 TD3 更新。
        返回 (avg_critic_loss, actor_loss, bc_loss)。
        """
        if self.buffer.size < self.batch_size:
            return 0.0, 0.0, 0.0

        total_Lc = 0.0

        for _ in range(iterations):
            self.total_it += 1

            # ---------- 采样 ----------
            (state, action, base_action, next_base_action,
             next_state, reward, done) = self.buffer.sample(self.batch_size)

            si               = np_to_tensor(state).float()
            ai               = np_to_tensor(action).float()
            base_action_t    = np_to_tensor(base_action).float()
            next_base_act_t  = np_to_tensor(next_base_action).float()
            s_i              = np_to_tensor(next_state).float()
            ri               = np_to_tensor(reward).float()
            di               = np_to_tensor(done).float()

            # ================================================================
            # 【TD3 机制 1】Target Policy Smoothing
            # 对目标策略的输出动作加截断高斯噪声，使 Critic 的 Q 目标曲面更平滑
            # ================================================================
            with torch.no_grad():
                noise = (
                    torch.randn_like(ai) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)

                # 使用张量支持的截断方式，适配多维不同的 max_action
                next_action = self.target_actor(s_i) + noise
                next_action = torch.max(
                    torch.min(next_action, self.target_actor.max_action), 
                    -self.target_actor.max_action
                )

                # ============================================================
                # 【TD3 机制 2】Clipped Double-Q（双 Critic 取最小值）
                # ============================================================
                target_Q1, target_Q2 = self.target_critic(s_i, next_action)
                target_Q = torch.min(target_Q1, target_Q2)

                # ------------------------------------------------------------
                # 【保留原机制】Base Bootstrapping
                # ------------------------------------------------------------
                if self.base_boot:
                    tQ1_base, tQ2_base = self.target_critic(s_i, next_base_act_t)
                    target_Q_base = torch.min(tQ1_base, tQ2_base)
                    target_Q = 0.5 * target_Q + 0.5 * target_Q_base # zxy

                yi = ri + self.gamma * (1.0 - di) * target_Q

            # ================================================================
            # Critic 更新
            # ================================================================
            current_Q1, current_Q2 = self.critic(si, ai)
            Lc = nn.MSELoss()(current_Q1, yi) + nn.MSELoss()(current_Q2, yi)

            self.optimizer_critic.zero_grad()
            Lc.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.optimizer_critic.step()

            total_Lc += Lc.item()

            # ================================================================
            # target_critic 每步 soft update
            # ================================================================
            soft_update(self.target_critic, self.critic, self.tau)

            # ================================================================
            # 【TD3 机制 3】Delayed Policy Updates（延迟 Actor 更新）
            # ================================================================
            if self.total_it % self.policy_freq == 0:

                self.optimizer_actor.zero_grad()

                if self.behavior_clone:
                    # --------------------------------------------------------
                    # 【保留原机制】Behavior Cloning（行为克隆）
                    # --------------------------------------------------------
                    with torch.no_grad():
                        q_base = self.critic.Q1(si, base_action_t)

                    a   = self.actor(si)
                    q_a = self.critic.Q1(si, a)

                    with torch.no_grad():
                        # xi = 1 当 Q(base) > Q(actor)，即专家比当前策略更优
                        xi = nn.ReLU()(torch.sign(q_base - q_a))

                    # BC Loss：只对 xi=1 的样本计算 MSE，并归一化
                    Lbc = (
                        ((a - base_action_t) ** 2).mean(dim=1, keepdim=True) * xi
                    ).sum() / max(xi.sum().item(), 1)

                    # Actor 总 Loss：BC 监督 + Policy Gradient（Q1 最大化）
                    La = Lbc - 0.02 * q_a.mean()

                    self._last_bc_loss = Lbc.item()

                else:
                    a  = self.actor(si)
                    La = -self.critic.Q1(si, a).mean()
                    self._last_bc_loss = 0.0

                La.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.optimizer_actor.step()

                self._last_actor_loss = La.item()

                # target_actor soft update 随 Actor 延迟进行
                soft_update(self.target_actor, self.actor, self.tau)

        # ---------- Epsilon 衰减 ----------
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.delta

        return total_Lc / iterations, self._last_actor_loss, self._last_bc_loss


# ===========================================================================
# PureRLAgent —— 纯 TD3，无专家依赖
# ===========================================================================

class PureRLAgent(WBAgent):
    """
    纯强化学习 Agent（TD3），完全不依赖 NMPC 专家控制器。
    """

    def __init__(self, log_dir, state_dim, action_dim, config=None, base_controller_func=None):
        
        # 将传入参数透传给父类 WBAgent 初始化网络与核心组件
        super().__init__(
            log_dir=log_dir,
            state_dim=state_dim,
            action_dim=action_dim,
            config=config,
            base_controller_func=base_controller_func,
        )

        # 强制关闭专家相关机制，覆写父类从 config 中读取的配置
        self.mixed_q = False
        self.base_boot = False
        self.behavior_clone = False

        # epsilon 在纯RL模式下无意义，固定为 0 防止误用
        self.epsilon     = 0.0
        self.epsilon_min = 0.0
        self.delta       = 0.0

    # -----------------------------------------------------------------------
    # 动作选择（纯 Actor 输出）
    # -----------------------------------------------------------------------

    def act(self, state):
        """
        返回 (actor_action, True, zero_base_action)。
        """
        with torch.no_grad():
            s_tensor     = np_to_tensor(state.reshape(1, -1))
            actor_action = self.actor(s_tensor).cpu().data.numpy().flatten()
            zero_base    = np.zeros(self.action_dim, dtype=np.float32)
        return actor_action, True, zero_base