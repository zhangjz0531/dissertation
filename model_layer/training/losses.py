from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    BCEWithLogitsLoss(pos_weight=neg/pos)
    警告：如果同时打开 label_smoothing 和 pos_weight，会双重扭曲 → 日志提示。
    """

    def __init__(
        self,
        pos_weight: Optional[float] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction

        if pos_weight is None:
            self.register_buffer("pos_weight_tensor", None)
        else:
            if self.label_smoothing > 0:
                import warnings
                warnings.warn("pos_weight 与 label_smoothing 同时开启会双重扭曲分布，建议二选一")
            self.register_buffer(
                "pos_weight_tensor",
                torch.tensor(float(pos_weight), dtype=torch.float32),
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.float().view(-1)
        if self.label_smoothing > 0:
            eps = self.label_smoothing
            targets = targets * (1.0 - eps) + 0.5 * eps
        return F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight_tensor,
            reduction=self.reduction,
        )


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(
        self,
        pos_weight: Optional[float] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction
        if pos_weight is None:
            self.register_buffer("pos_weight_tensor", None)
        else:
            self.register_buffer(
                "pos_weight_tensor",
                torch.tensor(float(pos_weight), dtype=torch.float32),
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.float().view(-1)
        if self.label_smoothing > 0:
            eps = self.label_smoothing
            targets = targets * (1.0 - eps) + 0.5 * eps
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight_tensor, reduction="none",
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_factor = (1.0 - p_t).pow(self.gamma)
        loss = focal_factor * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class HuberRegressionLoss(nn.Module):
    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.delta = float(delta)
        self.reduction = reduction

    def forward(self, preds, targets):
        return F.huber_loss(
            preds.view(-1), targets.float().view(-1),
            delta=self.delta, reduction=self.reduction,
        )


class MSERegressionLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, preds, targets):
        return F.mse_loss(preds.view(-1), targets.float().view(-1), reduction=self.reduction)


class MultiTaskDirRetLoss(nn.Module):
    def __init__(
        self,
        pos_weight: Optional[float] = None,
        cls_loss_name: str = "bce",
        reg_loss_name: str = "huber",
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.cls_weight = float(cls_weight)
        self.reg_weight = float(reg_weight)

        cls_loss_name = cls_loss_name.lower()
        if cls_loss_name == "bce":
            self.cls_loss = WeightedBCEWithLogitsLoss(
                pos_weight=pos_weight, label_smoothing=label_smoothing, reduction="mean",
            )
        elif cls_loss_name == "focal":
            self.cls_loss = BinaryFocalLossWithLogits(
                pos_weight=pos_weight, gamma=focal_gamma,
                label_smoothing=label_smoothing, reduction="mean",
            )
        else:
            raise ValueError(cls_loss_name)

        reg_loss_name = reg_loss_name.lower()
        if reg_loss_name == "huber":
            self.reg_loss = HuberRegressionLoss(delta=huber_delta, reduction="mean")
        elif reg_loss_name == "mse":
            self.reg_loss = MSERegressionLoss(reduction="mean")
        else:
            raise ValueError(reg_loss_name)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if preds.ndim != 2 or preds.shape[1] != 2:
            raise ValueError(f"preds 应为 [N,2]，收到 {tuple(preds.shape)}")
        if targets.ndim != 2 or targets.shape[1] != 2:
            raise ValueError(f"targets 应为 [N,2]，收到 {tuple(targets.shape)}")
        cls = self.cls_loss(preds[:, 0], targets[:, 0])
        reg = self.reg_loss(preds[:, 1], targets[:, 1])
        return self.cls_weight * cls + self.reg_weight * reg


def build_loss(
    task_type: str = "classification",
    loss_name: str = "bce",
    pos_weight: Optional[float] = None,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
    huber_delta: float = 1.0,
    cls_weight: float = 1.0,
    reg_weight: float = 1.0,
) -> nn.Module:
    task_type = task_type.lower()
    loss_name = loss_name.lower()

    if task_type == "classification":
        if loss_name == "bce":
            return WeightedBCEWithLogitsLoss(
                pos_weight=pos_weight, label_smoothing=label_smoothing, reduction="mean",
            )
        if loss_name == "focal":
            return BinaryFocalLossWithLogits(
                pos_weight=pos_weight, gamma=focal_gamma,
                label_smoothing=label_smoothing, reduction="mean",
            )
        raise ValueError(loss_name)

    if task_type == "regression":
        if loss_name == "huber":
            return HuberRegressionLoss(delta=huber_delta, reduction="mean")
        if loss_name == "mse":
            return MSERegressionLoss(reduction="mean")
        raise ValueError(loss_name)

    if task_type == "multitask":
        mapping = {
            "bce_huber": ("bce", "huber"),
            "focal_huber": ("focal", "huber"),
            "bce_mse": ("bce", "mse"),
            "focal_mse": ("focal", "mse"),
        }
        if loss_name not in mapping:
            raise ValueError(loss_name)
        cls_n, reg_n = mapping[loss_name]
        return MultiTaskDirRetLoss(
            pos_weight=pos_weight, cls_loss_name=cls_n, reg_loss_name=reg_n,
            cls_weight=cls_weight, reg_weight=reg_weight,
            focal_gamma=focal_gamma, label_smoothing=label_smoothing,
            huber_delta=huber_delta,
        )

    raise ValueError(task_type)
