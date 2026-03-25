import numpy as np
import torch
import torch.nn as nn
import random

def opt_cuda(t, device):
    if torch.cuda.is_available():
        cuda = "cuda:" + str(device)
        return t.cuda(cuda)
    else:
        return t

def np_to_tensor(n, device):
    return opt_cuda(torch.from_numpy(n).type(torch.FloatTensor), device)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class FastActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=0.5):
        super(FastActor, self).__init__()
        self.max_action = max_action
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh())

    def forward(self, s):
        return self.fc(s) * self.max_action

class FastCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(FastCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1))

    def forward(self, s, a):
        return self.fc(torch.cat([s, a], 1))

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr  = 0
        self.size = 0
        self.state            = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.action           = np.zeros((max_size, action_dim), dtype=np.float32)
        self.base_action      = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_base_action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state       = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.reward           = np.zeros((max_size, 1),          dtype=np.float32)
        self.not_done         = np.zeros((max_size, 1),          dtype=np.float32)

    def add(self, state, action, base_action, next_base_action, next_state, reward, done):
        self.state[self.ptr]             = state
        self.action[self.ptr]            = action
        self.base_action[self.ptr]       = base_action
        self.next_base_action[self.ptr]  = next_base_action
        self.next_state[self.ptr]        = next_state
        self.reward[self.ptr]            = reward
        self.not_done[self.ptr]          = 1. - done
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            np_to_tensor(self.state[ind],            device),
            np_to_tensor(self.action[ind],           device),
            np_to_tensor(self.base_action[ind],      device),
            np_to_tensor(self.next_base_action[ind], device),
            np_to_tensor(self.next_state[ind],       device),
            np_to_tensor(self.reward[ind],           device),
            np_to_tensor(self.not_done[ind],         device),
        )

class WBAgent:
    def __init__(self, log_dir, state_dim, action_dim, max_action=0.5,
                 base_boot=True, mixed_q=True, behavior_clone=True,
                 base_controller_func=None):

        self.device      = 0 if torch.cuda.is_available() else "cpu"
        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.max_action  = max_action

        # ---------- Hyperparameters ----------
        self.gamma      = 0.99
        self.tau        = 0.005
        self.batch_size = 256

        # Epsilon 线性衰减（与参考 agent 完全一致）：
        #   每次调用 act() 减少固定量 delta，而非按集数乘法衰减。
        #   从 1.0 → epsilon_min 约需 (1.0-0.1)/5e-6 = 180,000 步，
        #   对应 2000集×150步 训练量的约 60%，退火速率平稳可控。
        self.epsilon     = 1.0
        self.epsilon_min = 0.1
        self.delta       = 5e-6   # 可在外部按需调整

        # ---------- Flags ----------
        self.base_boot            = base_boot
        self.mixed_q              = mixed_q
        self.behavior_clone       = behavior_clone
        self.base_controller_func = base_controller_func

        # ---------- Networks ----------
        self.actor = opt_cuda(FastActor(state_dim, action_dim, max_action), self.device)
        self.target_actor = opt_cuda(FastActor(state_dim, action_dim, max_action), self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = opt_cuda(FastCritic(state_dim, action_dim), self.device)
        self.target_critic = opt_cuda(FastCritic(state_dim, action_dim), self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.buffer = ReplayBuffer(state_dim, action_dim)

    def act(self, state, test=False):
        """
        动作选择，逻辑与参考 agent 保持一致：

        1. 先获取专家动作与网络动作。
        2. test=True → 直接返回网络动作，不衰减 epsilon。
        3. 以 epsilon 概率执行专家动作（纯探索分支）。
        4. 否则进入 mixed_q 分支：用 Critic 比较两者 Q 值，
           若专家更优则执行专家，否则执行网络动作。
        5. 每次进入非 test 路径，epsilon 线性衰减一次。

        返回: (action, is_network, base_action)
        """
        base_action = np.zeros(self.action_dim, dtype=np.float32)
        if self.base_controller_func is not None:
            base_action = self.base_controller_func(state)

        s_t = np_to_tensor(state.reshape(1, -1), self.device)
        with torch.no_grad():
            action_net = self.actor(s_t).cpu().data.numpy().flatten()

        if test:
            return action_net, True, base_action

        # 每步线性衰减一次（与参考 agent delta 机制相同）
        self.epsilon = max(self.epsilon - self.delta, self.epsilon_min)

        if np.random.uniform(0, 1) < self.epsilon:
            # 专家分支：epsilon 仍较高时优先用专家填充 buffer
            return base_action, False, base_action
        else:
            if self.mixed_q:
                # Q 值对比：Critic 判断谁更优
                ba_t = np_to_tensor(base_action.reshape(1, -1), self.device)
                na_t = np_to_tensor(action_net.reshape(1, -1),  self.device)
                with torch.no_grad():
                    q_base = self.critic(s_t, ba_t).item()
                    q_net  = self.critic(s_t, na_t).item()
                if q_base > q_net:
                    return base_action, False, base_action
            return action_net, True, base_action

    def remember(self, state, action, base_action, next_base_action, next_state, reward, done):
        self.buffer.add(state, action, base_action, next_base_action, next_state, reward, done)

    def train(self, iterations):
        total_Lc = total_La = total_Lbc = 0.0

        for _ in range(iterations):
            si, ai, base_action, next_base_action_n, sn, ri, d = \
                self.buffer.sample(self.batch_size, self.device)

            # ------ Critic update ------
            with torch.no_grad():
                a_next  = self.target_actor(sn)
                back_up = self.target_critic(sn, a_next)
                if self.base_boot:
                    back_up_d = self.target_critic(sn, next_base_action_n)
                    back_up   = torch.max(back_up, back_up_d)
                yi = ri + d * self.gamma * back_up

            Lc = ((self.critic(si, ai) - yi) ** 2).mean()
            self.optimizer_critic.zero_grad()
            Lc.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.optimizer_critic.step()
            soft_update(self.target_critic, self.critic, self.tau)
            total_Lc += Lc.item()

            # ------ Actor update ------
            self.optimizer_actor.zero_grad()
            if self.behavior_clone:
                with torch.no_grad():
                    q_base = self.critic(si, base_action)
                a   = self.actor(si)
                q_a = self.critic(si, a)
                with torch.no_grad():
                    xi = nn.ReLU()(torch.sign(q_base - q_a))
                Lbc = (((a - base_action) ** 2).mean(dim=1, keepdim=True) * xi).sum() \
                      / max(xi.sum().item(), 1)
                La  = Lbc - 0.02 * q_a.mean()
            else:
                a   = self.actor(si)
                La  = -self.critic(si, a).mean()
                Lbc = torch.tensor(0.0)

            La.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.optimizer_actor.step()
            soft_update(self.target_actor, self.actor, self.tau)

            total_La  += La.item()
            total_Lbc += Lbc.item() if self.behavior_clone else 0.0

        return total_Lc / iterations, total_La / iterations, total_Lbc / iterations