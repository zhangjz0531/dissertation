from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


TaskType = Literal["classification", "regression", "multitask"]


@dataclass
class LSTMConfig:
    n_features: int
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.3
    bidirectional: bool = False
    use_layernorm: bool = True
    head_dropout: float = 0.3
    use_asset_embedding: bool = True
    n_assets: int = 16
    asset_emb_dim: int = 8
    task_type: TaskType = "classification"
    out_dim: int = 1


class LSTMBaseline(nn.Module):
    """
    简化版 LSTM（去掉 pack 逻辑，因为当前数据是定长窗口 mask 全 True）。
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = False,
        use_layernorm: bool = True,
        head_dropout: float = 0.3,
        use_asset_embedding: bool = True,
        n_assets: int = 16,
        asset_emb_dim: int = 8,
        task_type: TaskType = "classification",
    ):
        super().__init__()

        self.config = LSTMConfig(
            n_features=int(n_features), hidden_size=int(hidden_size),
            num_layers=int(num_layers), dropout=float(dropout),
            bidirectional=bool(bidirectional), use_layernorm=bool(use_layernorm),
            head_dropout=float(head_dropout), use_asset_embedding=bool(use_asset_embedding),
            n_assets=int(n_assets), asset_emb_dim=int(asset_emb_dim),
            task_type=str(task_type),
            out_dim=1 if task_type in {"classification", "regression"} else 2,
        )

        self.task_type = str(task_type).strip().lower()
        if self.task_type not in {"classification", "regression", "multitask"}:
            raise ValueError(f"未知 task_type: {task_type}")

        self.use_asset_embedding = bool(use_asset_embedding)
        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if self.bidirectional else 1

        lstm_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=int(n_features),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bool(bidirectional),
        )
        backbone_dim = int(hidden_size) * self.num_directions

        if self.use_asset_embedding:
            self.asset_emb = nn.Embedding(int(n_assets), int(asset_emb_dim))
            backbone_dim += int(asset_emb_dim)
        else:
            self.asset_emb = None

        self.norm = nn.LayerNorm(backbone_dim) if use_layernorm else nn.Identity()
        self.head_dropout = nn.Dropout(float(head_dropout))

        if self.task_type in {"classification", "regression"}:
            self.head = nn.Linear(backbone_dim, 1)
        else:
            self.head = nn.Linear(backbone_dim, 2)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        if self.asset_emb is not None:
            nn.init.normal_(self.asset_emb.weight, mean=0.0, std=0.02)
        if isinstance(self.head, nn.Linear):
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def encode(self, X, mask=None, asset_int=None):
        if X.dim() != 3:
            raise ValueError(f"X 应为 [B, L, F]，收到: {tuple(X.shape)}")
        _, (h_n, _) = self.lstm(X)
        if self.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h = h_n[-1]
        if self.asset_emb is not None:
            if asset_int is None:
                raise ValueError("use_asset_embedding=True 时必须传 asset_int")
            emb = self.asset_emb(asset_int.long())
            h = torch.cat([h, emb], dim=-1)
        h = self.norm(h)
        h = self.head_dropout(h)
        return h

    def forward(self, X, mask=None, asset_int=None, return_hidden=False):
        h = self.encode(X=X, mask=mask, asset_int=asset_int)
        out = self.head(h)
        if self.task_type in {"classification", "regression"}:
            out = out.squeeze(-1)
        if return_hidden:
            return out, h
        return out

    def summary(self) -> Dict:
        return {
            "config": asdict(self.config),
            "task_type": self.task_type,
            "n_params_total": int(sum(p.numel() for p in self.parameters())),
            "n_params_trainable": int(sum(p.numel() for p in self.parameters() if p.requires_grad)),
        }
