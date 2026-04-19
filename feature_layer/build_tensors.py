from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from data_layer.config import PANEL_DIR

try:
    from feature_layer.dataset import WindowTensorDataset
except Exception:
    from dataset import WindowTensorDataset  # 兼容直接运行

warnings.filterwarnings("ignore")


# ============================================================
# 默认路径
# ============================================================
DEFAULT_WINDOW_ROOT = PANEL_DIR.parent / "tensors" / "window_L64_H5_excess_dir_only"
DEFAULT_SCALER_PATH = PANEL_DIR.parent / "tensors" / "scaler.pkl"
DEFAULT_OUTPUT_BASE = PANEL_DIR.parent


# ============================================================
# 工具函数
# ============================================================
def _parse_csv_feature_names(s: str) -> Optional[List[str]]:
    s = str(s).strip()
    if not s:
        return None
    parts = [x.strip() for x in s.split(",")]
    parts = [x for x in parts if x]
    return parts if parts else None


def _load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"找不到 JSON 文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_window_meta(window_root: Path) -> Dict:
    meta_path = window_root / "feature_meta.json"
    if meta_path.exists():
        return _load_json(meta_path)
    return {}


def _build_default_output_dir(window_meta: Dict, label_mode: str, output_base: Path) -> Path:
    target_style = str(window_meta.get("target_style", "absolute")).lower()
    main_horizon = int(window_meta.get("main_horizon", 5))
    return output_base / f"tensors_{target_style}_H{main_horizon}_{str(label_mode).lower()}"


def _dataset_to_bundle(ds: WindowTensorDataset) -> Dict:
    meta_records = ds.meta_df.to_dict(orient="records")
    extra_meta = ds.feature_meta if hasattr(ds, "feature_meta") else {}

    bundle = {
        "X": ds.X.float(),
        "y": ds.y.clone(),
        "mask": ds.mask.bool(),
        "meta": meta_records,
        "feature_names": list(ds.selected_feature_names),
        "feature_indices": list(ds.feature_indices),
        "split": str(ds.split),
        "label_mode": str(ds.label_mode),
        "lookback": int(ds.lookback),
        "main_horizon": int(ds.main_horizon),
        "target_style": str(extra_meta.get("target_style", "absolute")),
        "dir_target_col": extra_meta.get("dir_target_col"),
        "ret_target_col": extra_meta.get("ret_target_col"),
    }
    return bundle


def _bundle_summary(bundle: Dict) -> Dict:
    X: torch.Tensor = bundle["X"]
    y: torch.Tensor = bundle["y"]
    mask: torch.Tensor = bundle["mask"]

    info = {
        "split": str(bundle["split"]),
        "label_mode": str(bundle["label_mode"]),
        "target_style": str(bundle.get("target_style", "absolute")),
        "X_shape": tuple(X.shape),
        "y_shape": tuple(y.shape),
        "mask_shape": tuple(mask.shape),
        "n_samples": int(X.shape[0]),
        "lookback": int(bundle["lookback"]),
        "n_features": int(X.shape[-1]),
    }

    if bundle["label_mode"] == "dir_only" and y.numel() > 0:
        info["dir_positive_rate"] = float(y.float().mean().item())
    elif bundle["label_mode"] == "ret_only" and y.numel() > 0:
        info["ret_mean"] = float(y.float().mean().item())
        info["ret_std"] = float(y.float().std(unbiased=False).item())
    elif bundle["label_mode"] == "multitask" and y.numel() > 0:
        info["dir_positive_rate"] = float(y[:, 0].float().mean().item())
        info["ret_mean"] = float(y[:, 1].float().mean().item())
        info["ret_std"] = float(y[:, 1].float().std(unbiased=False).item())

    return info


def print_bundle_report(report: Dict) -> None:
    print("=" * 72)
    print("  Tensor Bundle 概况")
    print("=" * 72)
    print(f"  window_root : {report['window_root']}")
    print(f"  output_dir  : {report['output_dir']}")
    print(f"  target_style: {report['target_style']}")
    print(f"  label_mode  : {report['label_mode']}")
    print(f"  lookback    : {report['lookback']}")
    print(f"  main_horizon: {report['main_horizon']}")
    print(f"  n_features  : {report['n_features']}")
    if report.get("dir_target_col") is not None:
        print(f"  dir_target  : {report['dir_target_col']}")
    if report.get("ret_target_col") is not None:
        print(f"  ret_target  : {report['ret_target_col']}")
    print()

    for split in ["train", "val", "test"]:
        item = report["splits"][split]
        print(f"[{split}]")
        print(f"  X shape : {item['X_shape']}")
        print(f"  y shape : {item['y_shape']}")
        print(f"  mask    : {item['mask_shape']}")
        if "dir_positive_rate" in item:
            print(f"  pos_rate: {item['dir_positive_rate']:.4f}")
        if "ret_mean" in item:
            print(f"  ret_mean: {item['ret_mean']:+.4f}")
            print(f"  ret_std : {item['ret_std']:.4f}")
        print()


# ============================================================
# 主逻辑
# ============================================================
def build_tensor_bundles(
    window_root: str | Path = DEFAULT_WINDOW_ROOT,
    output_dir: str | Path | None = None,
    label_mode: str = "dir_only",
    feature_names: Optional[Sequence[str]] = None,
    feature_indices: Optional[Sequence[int]] = None,
    scaler_path: Optional[str | Path] = DEFAULT_SCALER_PATH,
    copy_scaler: bool = True,
) -> Dict:
    """
    把 windowing.py 产出的分 split 张量整理成标准 train.pt / val.pt / test.pt。

    默认不再写入一个通用裸目录 outputs/tensors，避免不同标签定义互相覆盖。
    默认目录命名为：
      outputs/tensors_{target_style}_H{main_horizon}_{label_mode}
    """
    window_root = Path(window_root)
    if not window_root.exists():
        raise FileNotFoundError(f"window_root 不存在: {window_root}")

    window_meta = _infer_window_meta(window_root)
    if output_dir is None:
        output_dir = _build_default_output_dir(window_meta=window_meta, label_mode=label_mode, output_base=DEFAULT_OUTPUT_BASE)
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundles: Dict[str, Dict] = {}
    split_reports: Dict[str, Dict] = {}

    for split in ["train", "val", "test"]:
        ds = WindowTensorDataset(
            tensor_root=window_root,
            split=split,
            label_mode=label_mode,
            feature_names=feature_names,
            feature_indices=feature_indices,
            include_meta=True,
        )
        bundle = _dataset_to_bundle(ds)
        bundles[split] = bundle
        split_reports[split] = _bundle_summary(bundle)
        torch.save(bundle, output_dir / f"{split}.pt")

    target_style = str(bundles["train"].get("target_style", window_meta.get("target_style", "absolute"))).lower()
    dir_target_col = bundles["train"].get("dir_target_col")
    ret_target_col = bundles["train"].get("ret_target_col")

    feature_meta = {
        "source_window_root": str(window_root),
        "label_mode": str(label_mode),
        "target_style": target_style,
        "lookback": int(bundles["train"]["lookback"]),
        "main_horizon": int(bundles["train"]["main_horizon"]),
        "dir_target_col": dir_target_col,
        "ret_target_col": ret_target_col,
        "feature_names": list(bundles["train"]["feature_names"]),
        "feature_indices": list(bundles["train"]["feature_indices"]),
        "n_features": int(len(bundles["train"]["feature_names"])),
        "splits": {
            s: {
                "n_samples": int(bundles[s]["X"].shape[0]),
                "X_shape": list(bundles[s]["X"].shape),
                "y_shape": list(bundles[s]["y"].shape),
                "mask_shape": list(bundles[s]["mask"].shape),
            }
            for s in ["train", "val", "test"]
        },
    }
    with open(output_dir / "feature_meta.json", "w", encoding="utf-8") as f:
        json.dump(feature_meta, f, ensure_ascii=False, indent=2)

    report = {
        "window_root": str(window_root),
        "output_dir": str(output_dir),
        "target_style": target_style,
        "label_mode": str(label_mode),
        "lookback": int(bundles["train"]["lookback"]),
        "main_horizon": int(bundles["train"]["main_horizon"]),
        "dir_target_col": dir_target_col,
        "ret_target_col": ret_target_col,
        "n_features": int(len(bundles["train"]["feature_names"])),
        "splits": split_reports,
    }
    with open(output_dir / "bundle_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if copy_scaler and scaler_path is not None:
        scaler_path = Path(scaler_path)
        if scaler_path.exists():
            shutil.copy2(scaler_path, output_dir / "scaler.pkl")
        else:
            print(f"[WARN] scaler 不存在，跳过复制: {scaler_path}")

    print_bundle_report(report)
    print(f"✅ 已保存 train.pt / val.pt / test.pt 到: {output_dir}")
    return report


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Build standard train.pt / val.pt / test.pt bundles.")
    parser.add_argument("--window_root", type=str, default=str(DEFAULT_WINDOW_ROOT), help="windowing.py 输出目录")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="标准 tensor bundle 输出目录；若为空，自动生成 tensors_{target_style}_H{main_horizon}_{label_mode}",
    )
    parser.add_argument(
        "--label_mode",
        type=str,
        default="dir_only",
        choices=["dir_only", "ret_only", "multitask"],
        help="标签模式；默认 dir_only",
    )
    parser.add_argument("--feature_names", type=str, default="", help="逗号分隔的特征名列表")
    parser.add_argument("--feature_indices", type=int, nargs="*", default=None, help="特征索引子集")
    parser.add_argument("--scaler_path", type=str, default=str(DEFAULT_SCALER_PATH), help="scaler.pkl 路径")
    parser.add_argument("--no_copy_scaler", action="store_true", help="不复制 scaler.pkl")
    args = parser.parse_args()

    feature_names = _parse_csv_feature_names(args.feature_names)
    output_dir = None if not str(args.output_dir).strip() else args.output_dir

    build_tensor_bundles(
        window_root=args.window_root,
        output_dir=output_dir,
        label_mode=args.label_mode,
        feature_names=feature_names,
        feature_indices=args.feature_indices,
        scaler_path=args.scaler_path,
        copy_scaler=not args.no_copy_scaler,
    )


if __name__ == "__main__":
    main()
