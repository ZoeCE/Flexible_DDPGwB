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
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()) 

    def forward(self, s):
        return self.fc(s) * self.max_action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid())

    def forward(self, state, action):
        return self.fc(torch.cat((state, action), dim=1))

class ReplayBufferFast:
    def __init__(self, state_dim, action_dim, size):
        self.sta1_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.sta2_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.base_acts_buf = np.zeros([size, action_dim], dtype=np.float32) 
        self.rews_buf = np.zeros([size, 1], dtype=np.float32)
        self.done_buf = np.zeros([size, 1], dtype=np.bool_)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, sta, act, base_act, next_sta, rew, done):
        self.sta1_buf[self.ptr] = sta
        self.sta2_buf[self.ptr] = next_sta
        self.acts_buf[self.ptr] = act
        self.base_acts_buf[self.ptr] = base_act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(sta1=self.sta1_buf[idxs],
                    sta2=self.sta2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    base_acts=self.base_acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

class WBAgent:
    def __init__(self, log_dir, state_dim=10, action_dim=2, max_action=0.5, device=0, 
                 mixed_q=True, base_boot=True, behavior_clone=True, 
                 base_controller_func=None):
        
        self.device = device
        self.mixed_q = mixed_q
        self.base_boot = base_boot
        self.behavior_clone = behavior_clone
        self.base = base_controller_func 
        self.max_action = max_action

        self.buffer = ReplayBufferFast(state_dim, action_dim, size=100000)
        
        self.actor = opt_cuda(FastActor(state_dim, action_dim, max_action), self.device)
        self.target_actor = opt_cuda(FastActor(state_dim, action_dim, max_action), self.device)
        soft_update(self.target_actor, self.actor, 1)
        
        self.critic = opt_cuda(Critic(state_dim, action_dim), self.device)
        self.target_critic = opt_cuda(Critic(state_dim, action_dim), self.device)
        soft_update(self.target_critic, self.critic, 1)
        
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.gamma = 0.99
        self.tau = 0.005
        self.epsilon = 1.0
        self.delta = 5e-6 
        self.batch_size = 256
        
        # 【关键修改】Q-Loss 的权重系数
        # 原来是 0.02 (太小)，现在改为 0.5（爆炸性遗忘，适中到0.25）
        # 这意味着 Agent 会更看重 "拿高分" 而不仅仅是 "模仿 NMPC"
        self.lmbda = 0.25 

    def act(self, s, test=False):
        if self.base is not None:
            action_b = self.base(s) 
        else:
            action_b = np.zeros(2)

        s_tensor = np_to_tensor(s, self.device).unsqueeze(dim=0)
        with torch.no_grad():
            action_net = self.actor(s_tensor).squeeze().cpu().numpy()

        if test:
            return action_net, True, action_b

        if np.random.uniform(0, 1) < self.epsilon:
            self.epsilon = max(self.epsilon - self.delta, 0.1)
            return action_b, False, action_b
        else:
            self.epsilon = max(self.epsilon - self.delta, 0.1)
            if self.mixed_q:
                action_b_tensor = np_to_tensor(action_b, self.device).unsqueeze(dim=0)
                action_net_tensor = np_to_tensor(action_net, self.device).unsqueeze(dim=0)
                with torch.no_grad():
                    q_base = self.critic(s_tensor, action_b_tensor)
                    q_net = self.critic(s_tensor, action_net_tensor)
                if q_base.item() > q_net.item():
                    return action_b, False, action_b
            return action_net, True, action_b

    def remember(self, state, action, base_action, next_state, reward, done):
        self.buffer.store(state, action, base_action, next_state, [reward], [done])

    def train(self, frame):
        steps = 2 
        if self.buffer.size < self.batch_size:
            return 0, 0, 0

        total_Lc = total_La = total_Lbc = 0
        
        for i in range(steps):
            batch = self.buffer.sample_batch(batch_size=self.batch_size)
            si = np_to_tensor(batch['sta1'], self.device)
            sn = np_to_tensor(batch['sta2'], self.device)
            ai = np_to_tensor(batch['acts'], self.device)
            ri = np_to_tensor(batch['rews'], self.device)
            d = np_to_tensor(batch['done'], self.device)
            
            base_action = np_to_tensor(batch['base_acts'], self.device)
            base_action_n = base_action 

            # Critic Update
            self.optimizer_critic.zero_grad()
            with torch.no_grad():
                a_next = self.target_actor(sn)
                back_up = self.target_critic(sn, a_next)
                if self.base_boot:
                    back_up_d = self.target_critic(sn, base_action_n)
                    back_up = torch.max(back_up, back_up_d)
                yi = ri + (1 - d) * self.gamma * back_up
            
            Lc = ((self.critic(si, ai) - yi) ** 2).mean()
            Lc.backward()
            self.optimizer_critic.step()
            soft_update(self.target_critic, self.critic, self.tau)
            total_Lc += Lc.item()

            # Actor Update
            self.optimizer_actor.zero_grad()
            if self.behavior_clone:
                with torch.no_grad():
                    q_base = self.critic(si, base_action)
                a = self.actor(si)
                q_a = self.critic(si, a)
                with torch.no_grad():
                    xi = nn.ReLU()(torch.sign(q_base - q_a))
                
                # BC Loss: 模仿 NMPC
                Lbc = (((a - base_action) ** 2).mean(dim=1, keepdim=True) * xi).sum() / max(xi.sum().item(), 1)
                
                # 【关键修改】Total Loss = BC Loss - lambda * Q_Value
                # 增大 lambda (0.5) 让 Agent 更想去最大化 Q 值
                La = Lbc - self.lmbda * q_a.mean()
            else:
                a = self.actor(si)
                La = - self.critic(si, a).mean()
            
            La.backward()
            self.optimizer_actor.step()
            soft_update(self.target_actor, self.actor, self.tau)
            
            if self.behavior_clone:
                total_Lbc += Lbc.item()
            total_La += La.item()

        return total_Lc / steps, total_La / steps, total_Lbc / steps