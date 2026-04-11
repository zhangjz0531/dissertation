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
# 2. TD3 网络的 Actor 与 Critic (抗过拟合版)
# =====================================================================
# 🚨 核心修改：将 hidden_dim 从 64 降低到 32，限制模型容量
class Actor(nn.Module):
    def __init__(self, encoder, hidden_dim=32, action_dim=1):
        super(Actor, self).__init__()
        self.encoder = copy.deepcopy(encoder)
        for param in self.encoder.parameters():
            param.requires_grad = True

        self.l1 = nn.Linear(32, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state_seq):
        x = self.encoder.input_norm(self.encoder.feature_projection(state_seq))
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(state_seq.device)
        for layer in self.encoder.layers:
            x = layer(x, mask=mask)
        hidden_feature = self.encoder.prediction_head[0:3](x[:, -1, :])

        a = F.relu(self.l1(hidden_feature))
        a = torch.sigmoid(self.l2(a))
        return a


class Critic(nn.Module):
    def __init__(self, encoder, hidden_dim=32, action_dim=1):
        super(Critic, self).__init__()
        self.encoder = copy.deepcopy(encoder)
        for param in self.encoder.parameters():
            param.requires_grad = True

        self.l1 = nn.Linear(32 + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        self.q2_l1 = nn.Linear(32 + action_dim, hidden_dim)
        self.q2_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state_seq, action):
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


class TD3(object):
    def __init__(self, encoder_path, action_dim=1, lr=1e-4, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5,
                 policy_freq=2):
        self.base_encoder = QuantDCEncoder(num_features=7).to(device)
        self.base_encoder.load_state_dict(torch.load(encoder_path, map_location=device))

        self.actor = Actor(self.base_encoder, action_dim=action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        # 🚨 核心修改：加入 weight_decay=1e-4，强迫模型抗过拟合
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=1e-4)

        self.critic = Critic(self.base_encoder, action_dim=action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, weight_decay=1e-4)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, reward, next_state, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(0.0, 1.0)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

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