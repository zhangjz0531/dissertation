import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


# ==================== 1. 稳定的 Transformer 架构 ====================
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
    def __init__(self, num_features, d_model=64, nhead=8, num_layers=2):
        super().__init__()
        self.feature_projection = nn.Linear(num_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([StableTransformerLayer(d_model, nhead) for _ in range(num_layers)])
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 2)
        )

    def forward(self, src):
        x = self.input_norm(self.feature_projection(src))
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(src.device)
        for layer in self.layers:
            x = layer(x, mask=mask)
        hidden_state = x[:, -1, :]
        prediction = self.prediction_head(hidden_state)
        return prediction

    # ==================== 2. 数据准备 (严防数据泄露 & 保存指纹) ====================


def prepare_pretrain_data(window_size=30):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    btc_path = os.path.join(BASE_DIR, "download_data", "esg_data", "ESG_1D_Final.csv")

    if not os.path.exists(btc_path):
        print(f"[!] 找不到文件: {btc_path}")
        exit()

    df_btc = pd.read_csv(btc_path)

    # 🚨 核心修复 1：只用 2024 年之前的数据预训练，绝不偷看未来
    df_btc = df_btc[df_btc['date'] < '2024-01-01'].reset_index(drop=True)

    tech_indicators = ['dc_trend', 'dc_event', 'dc_drawdown', 'dc_T', 'dc_TMV', 'dc_R']
    features_list = [df_btc[tech].values for tech in tech_indicators]
    features_list.append(df_btc['sentiment_score'].values)

    features_data = np.column_stack(features_list).astype(np.float32)

    # 🚨 核心修复 2：计算均值和标准差，并保存为 npz 文件供 RL 使用
    train_mean = features_data.mean(axis=0)
    train_std = features_data.std(axis=0) + 1e-8
    features_data = (features_data - train_mean) / train_std

    # 强制保存到根目录的 models 文件夹下
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    stats_path = os.path.join(models_dir, "feature_stats.npz")
    np.savez(stats_path, mean=train_mean, std=train_std)
    print(f"[*] ✅ 归一化指纹已保存至: {stats_path}")

    targets_data = np.column_stack(
        [df_btc['target_return'].values, np.zeros_like(df_btc['target_return'].values)]).astype(np.float32) * 100.0

    X, Y = [], []
    for i in range(len(features_data) - window_size):
        X.append(features_data[i: i + window_size])
        Y.append(targets_data[i + window_size - 1])

    return torch.tensor(np.array(X)), torch.tensor(np.array(Y)), features_data.shape[1]


# ==================== 3. 监督预训练循环 ====================
if __name__ == "__main__":
    X, Y, num_features = prepare_pretrain_data(window_size=30)
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=True)

    model = QuantDCEncoder(num_features=num_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print(f"🚀 Transformer 1D 严格样本内预训练启动 | 特征维度: {num_features}")
    model.train()

    start_time = time.time()
    for epoch in range(100):
        epoch_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

    print(f"\n✅ 预训练总耗时: {(time.time() - start_time):.2f} 秒")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_save_path = os.path.join(BASE_DIR, "models", "encoder_pretrained.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ 纯净版预训练编码器权重已存至: {model_save_path}")