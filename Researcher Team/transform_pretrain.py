import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🚨 11 维超级情报特征
MACRO_FEATURES = [
    'dc_trend', 'dc_event', 'dc_drawdown', 'dc_T', 'dc_TMV', 'dc_R',
    'RSI', 'MACD_Pct', 'sentiment_score', 'interest_rate', 'credit_stress'
]
NUM_FEATURES = len(MACRO_FEATURES)

# 🚨 智能绝对路径：直接定位你生成数据的确切物理位置，拒绝搬文件！
DATA_PATH = r"D:\python\dissertation\Data Acquisition\download_data\esg_data\SPY_Macro_State.csv"
MODELS_DIR = r"D:\python\dissertation\models"


class MacroStateDataset(Dataset):
    def __init__(self, df, window_size=30):
        self.data = df[MACRO_FEATURES].values.astype(np.float32)
        returns = df['close'].pct_change().shift(-1).fillna(0).values.astype(np.float32)
        self.targets = returns
        self.window_size = window_size

    def __len__(self): return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.window_size]
        y = self.targets[idx + self.window_size - 1]
        return torch.tensor(x), torch.tensor(y)


# =====================================================================
# 🧠 研究团队核心大脑：多空辩论 Transformer
# =====================================================================
class StableTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.2):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
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
    def __init__(self, num_features=NUM_FEATURES, d_model=64, nhead=8, num_layers=2):
        super().__init__()
        self.feature_projection = nn.Linear(num_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([StableTransformerLayer(d_model, nhead) for _ in range(num_layers)])
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.input_norm(self.feature_projection(x))
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        for layer in self.layers: x = layer(x, mask=mask)
        out = self.prediction_head(x[:, -1, :])
        return out.squeeze(-1)


def train_research_team():
    print("==================================================")
    print("  🔬 唤醒【研究团队】：开启多空结构化辩论 (Transformer)")
    print("==================================================\n")

    if not os.path.exists(DATA_PATH):
        print(f"[!] 致命错误：智能路径寻址失败，找不到数据底座 {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    df_train = df[df['date'] < '2024-01-01'].copy().reset_index(drop=True)

    print(f"[*] 成功跨文件夹调取分析师报告！共包含 {len(df_train)} 天的 11 维特征矩阵。")

    dataset = MacroStateDataset(df_train, window_size=30)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

    model = QuantDCEncoder(num_features=NUM_FEATURES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    epochs = 20
    print(f"\n[*] 研究员开始辩论 (Device: {device})...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    -> [辩论轮次 {epoch + 1:02d}/{epochs}] 多空预测分歧度 (MSE Loss): {avg_loss:.6f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "encoder_pretrained.pth")
    torch.save(model.state_dict(), save_path)

    print(f"\n✅ 【研究团队】辩论结束！市场见解已跨文件夹凝聚至: {save_path}")


if __name__ == "__main__":
    train_research_team()