from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from data_layer.config import PANEL_DIR
    DEFAULT_TENSOR_ROOT = PANEL_DIR.parent / "tensors_excess_H5_dir_only"
except Exception:
    DEFAULT_TENSOR_ROOT = Path("./tensors")

warnings.filterwarnings("ignore")


ViewMode = Literal["sequence", "last_step", "flatten"]
LabelMode = Literal["dir_only", "ret_only", "multitask"]


@dataclass
class ClassWeightInfo:
    pos_rate: float
    neg_rate: float
    pos_weight: float
    n_pos: int
    n_neg: int
    n_total: int


def _load_bundle(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"找不到 tensor bundle: {path}")
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    return obj


def _normalize_split(s: str) -> str:
    s = str(s).strip().lower()
    if s not in {"train", "val", "test"}:
        raise ValueError(f"split 必须是 train/val/test")
    return s


def _normalize_view_mode(s: str) -> str:
    s = str(s).strip().lower()
    if s not in {"sequence", "last_step", "flatten"}:
        raise ValueError(f"view_mode 必须是 sequence/last_step/flatten")
    return s


def _normalize_label_mode(s: str) -> str:
    s = str(s).strip().lower()
    if s not in {"dir_only", "ret_only", "multitask"}:
        raise ValueError(f"label_mode 必须是 dir_only/ret_only/multitask")
    return s


def _compute_binary_class_weight_from_y(y: torch.Tensor) -> ClassWeightInfo:
    y = y.detach().cpu().float().view(-1)
    n_total = int(y.numel())
    n_pos = int((y == 1).sum().item())
    n_neg = int((y == 0).sum().item())
    if n_total == 0:
        return ClassWeightInfo(0.0, 0.0, 1.0, 0, 0, 0)
    pos_rate = float(n_pos / n_total)
    neg_rate = float(n_neg / n_total)
    pos_weight = 1.0 if n_pos == 0 else float(n_neg / max(n_pos, 1))
    return ClassWeightInfo(pos_rate, neg_rate, pos_weight, n_pos, n_neg, n_total)


def compute_class_weight_from_train_bundle(tensor_root: str | Path = DEFAULT_TENSOR_ROOT) -> ClassWeightInfo:
    tensor_root = Path(tensor_root)
    bundle = _load_bundle(tensor_root / "train.pt")
    y = bundle["y"]
    if y.ndim == 2:
        y = y[:, 0]
    return _compute_binary_class_weight_from_y(y.float().view(-1))


def _apply_view_mode(X: torch.Tensor, mask: torch.Tensor, view_mode: ViewMode) -> Tuple[torch.Tensor, torch.Tensor]:
    if view_mode == "sequence":
        return X, mask
    if view_mode == "last_step":
        return X[:, -1, :], mask[:, -1, :]
    if view_mode == "flatten":
        n, l, f = X.shape
        return X.reshape(n, l * f), mask.reshape(n, l * f)
    raise ValueError(view_mode)


def load_numpy_split(
    tensor_root: str | Path = DEFAULT_TENSOR_ROOT,
    split: str = "train",
    view_mode: ViewMode = "last_step",
    label_mode: LabelMode = "dir_only",
) -> Tuple[np.ndarray, np.ndarray, List[Dict], Dict]:
    tensor_root = Path(tensor_root)
    split = _normalize_split(split)
    view_mode = _normalize_view_mode(view_mode)
    label_mode = _normalize_label_mode(label_mode)

    bundle = _load_bundle(tensor_root / f"{split}.pt")
    X = bundle["X"].float()
    y = bundle["y"]
    mask = bundle["mask"]
    meta = list(bundle["meta"])

    X, mask = _apply_view_mode(X, mask, view_mode)
    X_np = X.detach().cpu().numpy().astype(np.float32)
    X_np = np.where(mask.detach().cpu().numpy().astype(bool), X_np, 0.0)

    if label_mode == "dir_only":
        if y.ndim == 2:
            y = y[:, 0]
        y_np = y.detach().cpu().numpy().astype(np.int64)
    elif label_mode == "ret_only":
        if y.ndim == 1:
            raise ValueError("当前 bundle 不含 ret 标签")
        y_np = y[:, 1].detach().cpu().numpy().astype(np.float32)
    else:
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("当前 bundle 不是 multitask 格式")
        y_np = y.detach().cpu().numpy().astype(np.float32)

    info = {
        "split": split, "view_mode": view_mode, "label_mode": label_mode,
        "X_shape": tuple(X_np.shape), "y_shape": tuple(y_np.shape),
        "n_samples": int(X_np.shape[0]), "feature_names": list(bundle["feature_names"]),
        "main_horizon": int(bundle.get("main_horizon", 0)),
        "target_style": bundle.get("target_style"),
        "dir_target_col": bundle.get("dir_target_col"),
        "ret_target_col": bundle.get("ret_target_col"),
    }
    return X_np, y_np, meta, info


class TensorBundleDataset(Dataset):
    def __init__(
        self,
        tensor_root: str | Path = DEFAULT_TENSOR_ROOT,
        split: str = "train",
        view_mode: ViewMode = "sequence",
        label_mode: LabelMode = "dir_only",
        include_meta: bool = True,
        include_asset_id: bool = True,
    ):
        super().__init__()
        self.tensor_root = Path(tensor_root)
        self.split = _normalize_split(split)
        self.view_mode = _normalize_view_mode(view_mode)
        self.label_mode = _normalize_label_mode(label_mode)
        self.include_meta = bool(include_meta)
        self.include_asset_id = bool(include_asset_id)

        bundle = _load_bundle(self.tensor_root / f"{self.split}.pt")
        self.X_raw = bundle["X"].float()
        self.mask_raw = bundle["mask"].bool()
        self.y_raw = bundle["y"]
        self.meta = list(bundle["meta"])
        self.feature_names = list(bundle["feature_names"])
        self.lookback = int(bundle["lookback"])
        self.main_horizon = int(bundle["main_horizon"])
        self.bundle_label_mode = str(bundle["label_mode"])
        self.target_style = bundle.get("target_style")
        self.dir_target_col = bundle.get("dir_target_col")
        self.ret_target_col = bundle.get("ret_target_col")

        self.X, self.mask = _apply_view_mode(self.X_raw, self.mask_raw, self.view_mode)
        self.y = self._prepare_y(self.y_raw, self.label_mode)

        if self.include_asset_id and len(self.meta) > 0 and "asset_int" in self.meta[0]:
            self.asset_int = torch.tensor(
                [int(m["asset_int"]) for m in self.meta], dtype=torch.long
            )
        else:
            self.asset_int = None

    @staticmethod
    def _prepare_y(y: torch.Tensor, label_mode: LabelMode) -> torch.Tensor:
        if label_mode == "dir_only":
            if y.ndim == 2:
                y = y[:, 0]
            return y.float().view(-1)
        if label_mode == "ret_only":
            if y.ndim != 2 or y.shape[1] < 2:
                raise ValueError("当前 bundle 不含 return 标签")
            return y[:, 1].float().view(-1)
        if label_mode == "multitask":
            if y.ndim != 2 or y.shape[1] != 2:
                raise ValueError("当前 bundle 不是 multitask [N,2] 格式")
            return y.float()
        raise ValueError(label_mode)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Dict:
        item = {"X": self.X[idx], "y": self.y[idx], "mask": self.mask[idx]}
        if self.asset_int is not None:
            item["asset_int"] = self.asset_int[idx]
        if self.include_meta:
            item["meta"] = self.meta[idx]
        return item

    @property
    def n_features(self) -> int:
        return int(self.X.shape[-1])

    @property
    def n_samples(self) -> int:
        return int(len(self))

    @property
    def n_assets(self) -> int:
        if self.asset_int is None:
            return 1
        return int(self.asset_int.max().item()) + 1


def tensor_bundle_collate(batch: List[Dict]) -> Dict:
    X = torch.stack([b["X"] for b in batch], dim=0)
    y = torch.stack([b["y"] if torch.is_tensor(b["y"]) else torch.tensor(b["y"]) for b in batch], dim=0)
    mask = torch.stack([b["mask"] for b in batch], dim=0)
    out = {"X": X, "y": y.float(), "mask": mask}
    if "asset_int" in batch[0]:
        out["asset_int"] = torch.stack([b["asset_int"] for b in batch], dim=0)
    if "meta" in batch[0]:
        out["meta"] = [b["meta"] for b in batch]
    return out


def build_dataloader(
    tensor_root: str | Path = DEFAULT_TENSOR_ROOT,
    split: str = "train",
    view_mode: ViewMode = "sequence",
    label_mode: LabelMode = "dir_only",
    batch_size: int = 128,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    include_meta: bool = True,
    include_asset_id: bool = True,
) -> Tuple[TensorBundleDataset, DataLoader]:
    ds = TensorBundleDataset(
        tensor_root=tensor_root, split=split, view_mode=view_mode, label_mode=label_mode,
        include_meta=include_meta, include_asset_id=include_asset_id,
    )
    if shuffle is None:
        shuffle = (split == "train")
    dl = DataLoader(
        ds, batch_size=int(batch_size), shuffle=bool(shuffle),
        num_workers=int(num_workers), pin_memory=bool(pin_memory),
        drop_last=bool(drop_last), collate_fn=tensor_bundle_collate,
    )
    return ds, dl


def build_all_dataloaders(
    tensor_root: str | Path = DEFAULT_TENSOR_ROOT,
    view_mode: ViewMode = "sequence",
    label_mode: LabelMode = "dir_only",
    train_batch_size: int = 128,
    eval_batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = False,
    include_meta: bool = True,
    include_asset_id: bool = True,
) -> Dict:
    train_ds, train_dl = build_dataloader(
        tensor_root=tensor_root, split="train", view_mode=view_mode, label_mode=label_mode,
        batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
        include_meta=include_meta, include_asset_id=include_asset_id,
    )
    val_ds, val_dl = build_dataloader(
        tensor_root=tensor_root, split="val", view_mode=view_mode, label_mode=label_mode,
        batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
        include_meta=include_meta, include_asset_id=include_asset_id,
    )
    test_ds, test_dl = build_dataloader(
        tensor_root=tensor_root, split="test", view_mode=view_mode, label_mode=label_mode,
        batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
        include_meta=include_meta, include_asset_id=include_asset_id,
    )
    class_weight_info = None
    if label_mode == "dir_only":
        class_weight_info = _compute_binary_class_weight_from_y(train_ds.y)
    return {
        "train_dataset": train_ds, "val_dataset": val_ds, "test_dataset": test_ds,
        "train_loader": train_dl, "val_loader": val_dl, "test_loader": test_dl,
        "class_weight_info": class_weight_info,
    }
