from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


TaskType = Literal["classification", "regression", "multitask"]


@dataclass
class TransformerConfig:
    n_features: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.3
    attn_dropout: float = 0.2
    max_len: int = 256
    use_layernorm: bool = True
    pooling: str = "mean"
    causal: bool = False
    use_asset_embedding: bool = True
    n_assets: int = 16
    asset_emb_dim: int = 8
    task_type: TaskType = "classification"
    out_dim: int = 1


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, attn_dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=attn_dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None, need_weights=False):
        attn_out, attn_weights = self.attn(
            query=x, key=x, value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False if need_weights else True,
        )
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x, attn_weights


class TimeSeriesTransformer(nn.Module):
    """
    Encoder-style time-series Transformer for sequence classification.
    关键修复:
      * causal=False (默认) -> self-attn 每个 token 可看全窗口历史
      * pooling='mean' (默认) -> 对噪声更稳
      * use_asset_embedding=True (默认) -> 跨资产 pooling 也能学个体差异
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.3,
        attn_dropout: float = 0.2,
        max_len: int = 256,
        use_layernorm: bool = True,
        pooling: str = "mean",
        causal: bool = False,
        use_asset_embedding: bool = True,
        n_assets: int = 16,
        asset_emb_dim: int = 8,
        task_type: TaskType = "classification",
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")
        if pooling not in {"last", "mean"}:
            raise ValueError("pooling 必须是 last 或 mean")

        self.config = TransformerConfig(
            n_features=int(n_features), d_model=int(d_model), n_heads=int(n_heads),
            n_layers=int(n_layers), d_ff=int(d_ff), dropout=float(dropout),
            attn_dropout=float(attn_dropout), max_len=int(max_len),
            use_layernorm=bool(use_layernorm), pooling=str(pooling), causal=bool(causal),
            use_asset_embedding=bool(use_asset_embedding), n_assets=int(n_assets),
            asset_emb_dim=int(asset_emb_dim), task_type=str(task_type),
            out_dim=1 if task_type in {"classification", "regression"} else 2,
        )

        self.task_type = str(task_type).strip().lower()
        if self.task_type not in {"classification", "regression", "multitask"}:
            raise ValueError(f"未知 task_type: {task_type}")

        self.pooling = str(pooling)
        self.causal = bool(causal)
        self.use_asset_embedding = bool(use_asset_embedding)

        self.input_proj = nn.Linear(int(n_features), int(d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, int(max_len), int(d_model)))
        self.input_dropout = nn.Dropout(float(dropout))

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=int(d_model), n_heads=int(n_heads), d_ff=int(d_ff),
                dropout=float(dropout), attn_dropout=float(attn_dropout),
            )
            for _ in range(int(n_layers))
        ])

        backbone_dim = int(d_model)
        if self.use_asset_embedding:
            self.asset_emb = nn.Embedding(int(n_assets), int(asset_emb_dim))
            backbone_dim += int(asset_emb_dim)
        else:
            self.asset_emb = None

        self.norm = nn.LayerNorm(backbone_dim) if use_layernorm else nn.Identity()
        self.head_dropout = nn.Dropout(float(dropout))

        if self.task_type in {"classification", "regression"}:
            self.head = nn.Linear(backbone_dim, 1)
        else:
            self.head = nn.Linear(backbone_dim, 2)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        if self.asset_emb is not None:
            nn.init.normal_(self.asset_emb.weight, mean=0.0, std=0.02)
        if isinstance(self.head, nn.Linear):
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    @staticmethod
    def _build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    @staticmethod
    def _build_key_padding_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if mask.dim() == 3:
            step_valid = mask.any(dim=-1)
        elif mask.dim() == 2:
            step_valid = mask.bool()
        else:
            return None
        kpm = ~step_valid
        if kpm.sum().item() == 0:
            return None
        return kpm

    def _pool_hidden(self, h, key_padding_mask):
        if self.pooling == "last":
            if key_padding_mask is None:
                return h[:, -1, :]
            valid = ~key_padding_mask
            lengths = valid.long().sum(dim=1).clamp(min=1)
            idx = lengths - 1
            return h[torch.arange(h.size(0), device=h.device), idx, :]
        if key_padding_mask is None:
            return h.mean(dim=1)
        valid = (~key_padding_mask).float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        return (h * valid).sum(dim=1) / denom

    def encode(self, X, mask=None, asset_int=None, return_attn=False):
        if X.dim() != 3:
            raise ValueError(f"X 应为 [B, L, F]，收到: {tuple(X.shape)}")
        B, L, n_feat = X.shape
        if L > self.config.max_len:
            raise ValueError(f"L={L} > max_len={self.config.max_len}")

        attn_mask = self._build_causal_mask(L, X.device) if self.causal else None
        key_padding_mask = self._build_key_padding_mask(mask)

        h = self.input_proj(X) + self.pos_emb[:, :L, :]
        h = self.input_dropout(h)

        attn_list = [] if return_attn else None
        for block in self.blocks:
            h, attn = block(
                x=h, attn_mask=attn_mask,
                key_padding_mask=key_padding_mask, need_weights=return_attn,
            )
            if return_attn and attn is not None:
                attn_list.append(attn)

        pooled = self._pool_hidden(h, key_padding_mask=key_padding_mask)
        if self.asset_emb is not None:
            if asset_int is None:
                raise ValueError("use_asset_embedding=True 时必须传 asset_int")
            emb = self.asset_emb(asset_int.long())
            pooled = torch.cat([pooled, emb], dim=-1)
        pooled = self.norm(pooled)
        pooled = self.head_dropout(pooled)
        return pooled, attn_list

    def forward(self, X, mask=None, asset_int=None, return_hidden=False, return_attn=False):
        hidden, attn_list = self.encode(X=X, mask=mask, asset_int=asset_int, return_attn=return_attn)
        out = self.head(hidden)
        if self.task_type in {"classification", "regression"}:
            out = out.squeeze(-1)
        if return_hidden and return_attn:
            return out, hidden, attn_list
        if return_hidden:
            return out, hidden
        if return_attn:
            return out, attn_list
        return out

    def summary(self) -> Dict:
        return {
            "config": asdict(self.config),
            "task_type": self.task_type,
            "n_params_total": int(sum(p.numel() for p in self.parameters())),
            "n_params_trainable": int(sum(p.numel() for p in self.parameters() if p.requires_grad)),
        }
