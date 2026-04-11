import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# 1. 预训练的 Transformer 架构 (特征提取器)
# =====================================================================
class StableTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.2):
        super().__init__()
        self.nhead = nhead;
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model);
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model);
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model);
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_model * 4, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        nx = self.norm1(x)
        B, L, D = nx.shape
        q = self.q_proj(nx).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(nx).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(nx).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if mask is not None: attn = attn.masked_fill(mask[:L, :L] == 0, -1e9)
        attn = torch.softmax(attn, dim=-1)
        attn_out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        x = x + self.dropout(self.out_proj(attn_out))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class QuantDCEncoder(nn.Module):
    def __init__(self, num_features=7, d_model=64, nhead=8, num_layers=2):
        super().__init__()
        self.feature_projection = nn.Linear(num_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([StableTransformerLayer(d_model, nhead) for _ in range(num_layers)])
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 2)
        )


# =====================================================================
# 2. TD3 网络的 Actor 与 Critic
# =====================================================================
class Actor(nn.Module):
    def __init__(self, encoder, hidden_dim=64, action_dim=1):
        super(Actor, self).__init__()
        self.encoder = encoder
        # 冻结 Transformer 参数，防止被 RL 的高方差梯度破坏
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.l1 = nn.Linear(32, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state_seq):
        with torch.no_grad():
            x = self.encoder.input_norm(self.encoder.feature_projection(state_seq))
            seq_len = x.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len)).to(state_seq.device)
            for layer in self.encoder.layers:
                x = layer(x, mask=mask)
            # 取最后一层隐藏状态过线性层得到 32 维特征
            hidden_feature = self.encoder.prediction_head[0:3](x[:, -1, :])

        a = F.relu(self.l1(hidden_feature))
        a = torch.sigmoid(self.l2(a))  # 输出 0~1 的仓位权重
        return a


class Critic(nn.Module):
    def __init__(self, encoder, hidden_dim=64, action_dim=1):
        super(Critic, self).__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Q1 网络
        self.l1 = nn.Linear(32 + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 网络 (用于缓解高估)
        self.q2_l1 = nn.Linear(32 + action_dim, hidden_dim)
        self.q2_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state_seq, action):
        with torch.no_grad():
            x = self.encoder.input_norm(self.encoder.feature_projection(state_seq))
            seq_len = x.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len)).to(state_seq.device)
            for layer in self.encoder.layers:
                x = layer(x, mask=mask)
            hidden_feature = self.encoder.prediction_head[0:3](x[:, -1, :])

        sa = torch.cat([hidden_feature, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.q2_l1(sa))
        q2 = F.relu(self.q2_l2(q2))
        q2 = self.l3(q2)
        return q1, q2

    def Q1(self, state_seq, action):
        # 仅供 Actor 延迟更新时调用
        with torch.no_grad():
            x = self.encoder.input_norm(self.encoder.feature_projection(state_seq))
            seq_len = x.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len)).to(state_seq.device)
            for layer in self.encoder.layers:
                x = layer(x, mask=mask)
            hidden_feature = self.encoder.prediction_head[0:3](x[:, -1, :])

        sa = torch.cat([hidden_feature, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


# =====================================================================
# 3. 序列经验回放池 (Sequence Replay Buffer)
# =====================================================================
class ReplayBuffer(object):
    def __init__(self, state_dim=(30, 7), action_dim=1, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, *state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, *state_dim))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.not_done[ind]).to(device)
        )


# =====================================================================
# 4. TD3 核心算法管理器
# =====================================================================
class TD3(object):
    def __init__(self, encoder_path, action_dim=1, lr=3e-4, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5,
                 policy_freq=2):
        # 初始化 Encoder 并加载权重
        self.encoder = QuantDCEncoder(num_features=7).to(device)
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        self.encoder.eval()  # Encoder 永远在 Eval 模式

        self.actor = Actor(self.encoder, action_dim=action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(self.encoder, action_dim=action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        # 用于与环境交互，输入 state 为 Numpy 数组
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # 1. 采样一个 Batch 的历史数据
        state, action, reward, next_state, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # 2. 目标策略平滑 (Target Policy Smoothing) - 给 action 加点噪音防止过拟合
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            # 由于我们的仓位权重在 0~1 之间，所以 action 加上 noise 后必须 clamp 在 [0, 1]
            next_action = (self.actor_target(next_state) + noise).clamp(0.0, 1.0)

            # 3. 计算双 Q 网络的目标值
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # 4. 更新 Critic 网络
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 5. 延迟更新 Actor 网络
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 6. 软更新 Target 网络 (EMA)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        torch.save(self.actor.state_dict(), filename + "_actor.pth")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic.pth", map_location=device))
        self.actor.load_state_dict(torch.load(filename + "_actor.pth", map_location=device))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)


# =================测试=================
if __name__ == "__main__":
    import os

    print("===========================================")
    print("    正在测试 TD3 核心强化学习框架")
    print("===========================================")

    # 指向你刚才预训练好的权重
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    encoder_path = os.path.join(BASE_DIR,"dissertation", "../transformer", "models", "encoder_pretrained.pth")

    if not os.path.exists(encoder_path):
        print(f"[!] 请先完成预训练，或者确认预训练权重在此路径: {encoder_path}")
    else:
        # 1. 实例化 TD3
        agent = TD3(encoder_path=encoder_path, action_dim=1)
        print("✅ TD3 Agent 初始化成功 (已冻结 Transformer 参数)")

        # 2. 模拟环境状态观测 (30天，7个特征)
        dummy_state = np.random.randn(30, 7)

        # 3. 动作选择测试
        action = agent.select_action(dummy_state)
        print(f"✅ Select Action 测试成功 | 建议 BTC 仓位: {action[0]:.4f}")

        # 4. 经验回放池测试
        buffer = ReplayBuffer(state_dim=(30, 7), action_dim=1)
        for _ in range(300):
            buffer.add(dummy_state, action, np.array([1.5]), dummy_state, 0)
        print(f"✅ Replay Buffer 测试成功 | 当前容量: {buffer.size}")

        # 5. 训练迭代测试
        agent.train(buffer, batch_size=256)
        print("✅ Train 迭代测试成功 | Actor 与 Critic 均未报错")
        print("\n🎉 系统脑机接口已完全打通，随时可以接入 MuSA 交易环境进行实盘推演训练！")