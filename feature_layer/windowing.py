from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from data_layer.config import PANEL_DIR

try:
    from feature_layer.scaler import HybridRobustScaler
except Exception:
    from scaler import HybridRobustScaler  # 兼容直接运行

warnings.filterwarnings("ignore")


# ============================================================
# 默认路径
# ============================================================
DEFAULT_PANEL_PATH = PANEL_DIR / "panel_daily_stock_main.parquet"
DEFAULT_SCALER_PATH = PANEL_DIR.parent / "tensors" / "scaler.pkl"
DEFAULT_OUTPUT_ROOT = PANEL_DIR.parent / "tensors"


# ============================================================
# 工具函数
# ============================================================
def _load_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺失必要列: {missing}")


def _sort_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if out["timestamp"].isna().any():
        bad = int(out["timestamp"].isna().sum())
        raise ValueError(f"timestamp 有 {bad} 个无法解析的值")

    out["asset_id"] = out["asset_id"].astype(str).str.upper().str.strip()
    out = (
        out.sort_values(["asset_id", "timestamp"])
        .drop_duplicates(subset=["asset_id", "timestamp"], keep="last")
        .reset_index(drop=True)
    )
    return out


def _safe_stack_3d(arrs: List[np.ndarray], lookback: int, n_features: int) -> torch.Tensor:
    if len(arrs) == 0:
        return torch.zeros((0, lookback, n_features), dtype=torch.float32)
    return torch.tensor(np.stack(arrs).astype(np.float32), dtype=torch.float32)


def _safe_stack_2d(arrs: List[np.ndarray], width: int) -> torch.Tensor:
    if len(arrs) == 0:
        return torch.zeros((0, width), dtype=torch.float32)
    return torch.tensor(np.stack(arrs).astype(np.float32), dtype=torch.float32)


def _safe_stack_1d(arrs: List[float]) -> torch.Tensor:
    if len(arrs) == 0:
        return torch.zeros((0,), dtype=torch.float32)
    return torch.tensor(np.asarray(arrs, dtype=np.float32), dtype=torch.float32)


def _safe_stack_mask(arrs: List[np.ndarray], lookback: int, n_features: int) -> torch.Tensor:
    if len(arrs) == 0:
        return torch.zeros((0, lookback, n_features), dtype=torch.bool)
    return torch.tensor(np.stack(arrs).astype(bool), dtype=torch.bool)


def _summarize_split_y(y: torch.Tensor, task_mode: str) -> Dict:
    if y.numel() == 0:
        return {"n_samples": 0}

    if task_mode == "dir_only":
        return {
            "n_samples": int(y.shape[0]),
            "dir_positive_rate": float(y.float().mean().item()),
        }

    if task_mode == "ret_only":
        return {
            "n_samples": int(y.shape[0]),
            "ret_mean": float(y.float().mean().item()),
            "ret_std": float(y.float().std(unbiased=False).item()),
        }

    dir_y = y[:, 0]
    ret_y = y[:, 1]
    return {
        "n_samples": int(y.shape[0]),
        "dir_positive_rate": float(dir_y.float().mean().item()),
        "ret_mean": float(ret_y.float().mean().item()),
        "ret_std": float(ret_y.float().std(unbiased=False).item()),
    }


def _resolve_target_cols(target_style: str, main_horizon: int, task_mode: str) -> Tuple[str, Optional[str]]:
    h = int(main_horizon)
    style = str(target_style).strip().lower()
    mode = str(task_mode).strip().lower()

    if style == "absolute":
        dir_col = f"target_dir_{h}"
        ret_col = f"target_ret_{h}"
    elif style == "excess":
        dir_col = f"target_excess_dir_{h}"
        ret_col = f"target_excess_ret_{h}"
    elif style == "band":
        dir_col = f"target_band_dir_{h}"
        # band 方向通常配超额收益做联合目标
        ret_col = f"target_excess_ret_{h}"
    else:
        raise ValueError(f"未知 target_style: {target_style}")

    if mode == "dir_only":
        return dir_col, None
    if mode == "ret_only":
        return ret_col, None
    if mode == "multitask":
        return dir_col, ret_col
    raise ValueError(f"未知 task_mode: {task_mode}")


def _build_default_output_dir(output_root: Path, lookback: int, main_horizon: int, target_style: str, task_mode: str) -> Path:
    return output_root / f"window_L{int(lookback)}_H{int(main_horizon)}_{str(target_style).lower()}_{str(task_mode).lower()}"


# ============================================================
# 核心：构建滚动窗口
# ============================================================
def build_windows(
    panel_df: pd.DataFrame,
    scaler: HybridRobustScaler,
    lookback: int = 64,
    main_horizon: int = 5,
    task_mode: str = "multitask",
    target_style: str = "excess",
) -> Tuple[Dict[str, Dict[str, object]], Dict]:
    """
    输出:
      split_data = {
        "train": {"X":..., "y":..., "mask":..., "meta": DataFrame},
        "val":   {...},
        "test":  {...},
      }

    规则:
      - 以窗口最后一个 bar 的 split 决定样本归属
      - 窗口历史允许跨越更早 split（这不构成泄露，因为都是过去信息）
      - 只有 end row 的 is_usable_for_model == 1 才生成样本
      - mask 由原始 panel 的缺失情况构造，不再一律全 1
    """
    task_mode = str(task_mode).strip().lower()
    if task_mode not in {"dir_only", "ret_only", "multitask"}:
        raise ValueError("task_mode 必须是 'dir_only' / 'ret_only' / 'multitask'")

    panel = _sort_panel(panel_df)
    dir_or_ret_col, ret_col = _resolve_target_cols(target_style=target_style, main_horizon=main_horizon, task_mode=task_mode)

    required = [
        "asset_id",
        "timestamp",
        "split",
        "is_usable_for_model",
        dir_or_ret_col,
    ]
    if task_mode == "multitask" and ret_col is not None:
        required.append(ret_col)

    _require_columns(panel, required)

    feature_cols = list(scaler.feature_cols_)
    missing_feat = [c for c in feature_cols if c not in panel.columns]
    if missing_feat:
        raise ValueError(f"panel 缺少 scaler 需要的特征列: {missing_feat}")

    # 先从原始 panel 构造缺失 mask，再做数值缩放
    raw_mask_panel = panel[feature_cols].notna().astype(bool)
    scaled_panel = scaler.transform(panel)

    n_features = len(feature_cols)
    splits = ["train", "val", "test"]
    buckets = {s: {"X": [], "y": [], "mask": [], "meta": []} for s in splits}

    asset_ids = sorted(scaled_panel["asset_id"].astype(str).unique().tolist())
    asset_to_id = {a: i for i, a in enumerate(asset_ids)}

    total_skipped_no_history = 0
    total_skipped_unusable = 0
    total_skipped_bad_target = 0

    for asset_id, sub_scaled in scaled_panel.groupby("asset_id", sort=False):
        sub_scaled = sub_scaled.sort_values("timestamp").reset_index(drop=True)
        sub_raw_mask = raw_mask_panel.loc[sub_scaled.index].reset_index(drop=True)
        # 上面 reset_index 后索引错了，因此重新按 asset/time 对齐更安全
        sub_raw_mask = raw_mask_panel.loc[
            panel[(panel["asset_id"].astype(str).str.upper().str.strip() == str(asset_id))]
            .sort_values("timestamp").index
        ].reset_index(drop=True)

        X_all = sub_scaled[feature_cols].to_numpy(dtype=np.float32)
        mask_all = sub_raw_mask[feature_cols].to_numpy(dtype=bool)
        split_arr = sub_scaled["split"].astype(str).to_numpy()
        usable_arr = pd.to_numeric(sub_scaled["is_usable_for_model"], errors="coerce").fillna(0).to_numpy()
        ts_arr = pd.to_datetime(sub_scaled["timestamp"]).dt.strftime("%Y-%m-%d").to_numpy()

        y_primary = pd.to_numeric(sub_scaled[dir_or_ret_col], errors="coerce").to_numpy(dtype=np.float32)
        y_secondary = None
        if task_mode == "multitask" and ret_col is not None:
            y_secondary = pd.to_numeric(sub_scaled[ret_col], errors="coerce").to_numpy(dtype=np.float32)

        for end_idx in range(len(sub_scaled)):
            if end_idx < lookback - 1:
                total_skipped_no_history += 1
                continue

            if usable_arr[end_idx] != 1:
                total_skipped_unusable += 1
                continue

            split_name = split_arr[end_idx]
            if split_name not in buckets:
                continue

            X_win = X_all[end_idx - lookback + 1 : end_idx + 1]
            mask_win = mask_all[end_idx - lookback + 1 : end_idx + 1]
            if X_win.shape[0] != lookback or mask_win.shape[0] != lookback:
                total_skipped_no_history += 1
                continue

            if task_mode == "dir_only":
                if not np.isfinite(y_primary[end_idx]):
                    total_skipped_bad_target += 1
                    continue
                y_win = float(y_primary[end_idx])
            elif task_mode == "ret_only":
                if not np.isfinite(y_primary[end_idx]):
                    total_skipped_bad_target += 1
                    continue
                y_win = float(y_primary[end_idx])
            else:
                if (not np.isfinite(y_primary[end_idx])) or (y_secondary is None) or (not np.isfinite(y_secondary[end_idx])):
                    total_skipped_bad_target += 1
                    continue
                y_win = np.array([y_primary[end_idx], y_secondary[end_idx]], dtype=np.float32)

            buckets[split_name]["X"].append(X_win.astype(np.float32))
            buckets[split_name]["mask"].append(mask_win.astype(bool))
            buckets[split_name]["y"].append(y_win)
            buckets[split_name]["meta"].append(
                {
                    "asset_id": str(asset_id),
                    "asset_int": int(asset_to_id[str(asset_id)]),
                    "timestamp": str(ts_arr[end_idx]),
                    "split": str(split_name),
                    "end_row_in_asset": int(end_idx),
                }
            )

    split_data: Dict[str, Dict[str, object]] = {}
    for split_name in splits:
        if task_mode in {"dir_only", "ret_only"}:
            y_tensor = _safe_stack_1d(buckets[split_name]["y"])
        else:
            y_tensor = _safe_stack_2d(buckets[split_name]["y"], width=2)

        X_tensor = _safe_stack_3d(buckets[split_name]["X"], lookback=lookback, n_features=n_features)
        mask_tensor = _safe_stack_mask(buckets[split_name]["mask"], lookback=lookback, n_features=n_features)
        meta_df = pd.DataFrame(buckets[split_name]["meta"])

        split_data[split_name] = {
            "X": X_tensor,
            "y": y_tensor,
            "mask": mask_tensor,
            "meta": meta_df,
        }

    meta_summary = {
        "lookback": int(lookback),
        "main_horizon": int(main_horizon),
        "task_mode": str(task_mode),
        "target_style": str(target_style).lower(),
        "dir_target_col": None,
        "ret_target_col": None,
        "feature_cols": feature_cols,
        "n_features": int(n_features),
        "asset_to_id": asset_to_id,
        "skipped_no_history": int(total_skipped_no_history),
        "skipped_unusable": int(total_skipped_unusable),
        "skipped_bad_target": int(total_skipped_bad_target),
        "split_summary": {s: _summarize_split_y(split_data[s]["y"], task_mode) for s in splits},
    }

    resolved_dir_col, resolved_ret_col = _resolve_target_cols(target_style=target_style, main_horizon=main_horizon, task_mode="multitask")
    if task_mode == "dir_only":
        meta_summary["dir_target_col"] = dir_or_ret_col
    elif task_mode == "ret_only":
        meta_summary["ret_target_col"] = dir_or_ret_col
    else:
        meta_summary["dir_target_col"] = resolved_dir_col
        meta_summary["ret_target_col"] = resolved_ret_col

    return split_data, meta_summary


# ============================================================
# 保存
# ============================================================
def save_window_data(split_data: Dict[str, Dict[str, object]], meta_summary: Dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, obj in split_data.items():
        torch.save(obj["X"], output_dir / f"{split_name}_X.pt")
        torch.save(obj["y"], output_dir / f"{split_name}_y.pt")
        torch.save(obj["mask"], output_dir / f"{split_name}_mask.pt")
        meta_df: pd.DataFrame = obj["meta"]
        meta_df.to_csv(output_dir / f"{split_name}_meta.csv", index=False, encoding="utf-8-sig")

    with open(output_dir / "feature_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_summary, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Window 数据已保存到: {output_dir}")


# ============================================================
# 打印报告
# ============================================================
def print_window_report(split_data: Dict[str, Dict[str, object]], meta_summary: Dict) -> None:
    print("=" * 72)
    print("  滚动窗口样本概况")
    print("=" * 72)
    print(f"  lookback     : {meta_summary['lookback']}")
    print(f"  main_horizon : {meta_summary['main_horizon']}")
    print(f"  task_mode    : {meta_summary['task_mode']}")
    print(f"  target_style : {meta_summary['target_style']}")
    if meta_summary.get("dir_target_col") is not None:
        print(f"  dir_target   : {meta_summary['dir_target_col']}")
    if meta_summary.get("ret_target_col") is not None:
        print(f"  ret_target   : {meta_summary['ret_target_col']}")
    print(f"  n_features   : {meta_summary['n_features']}")

    print("\n  跳过统计:")
    print(f"    skipped_no_history : {meta_summary['skipped_no_history']}")
    print(f"    skipped_unusable   : {meta_summary['skipped_unusable']}")
    print(f"    skipped_bad_target : {meta_summary['skipped_bad_target']}")

    print("\n  各 split 样本:")
    for split_name in ["train", "val", "test"]:
        X = split_data[split_name]["X"]
        y = split_data[split_name]["y"]
        mask = split_data[split_name]["mask"]
        summary = meta_summary["split_summary"][split_name]

        print(f"    [{split_name}]")
        print(f"      X shape = {tuple(X.shape)}")
        print(f"      y shape = {tuple(y.shape)}")
        print(f"      mask    = {tuple(mask.shape)}")

        if "dir_positive_rate" in summary and summary["dir_positive_rate"] is not None:
            print(f"      dir_positive_rate = {summary['dir_positive_rate']:.4f}")
        if "ret_mean" in summary and summary["ret_mean"] is not None:
            print(f"      ret_mean          = {summary['ret_mean']:+.4f}")
            print(f"      ret_std           = {summary['ret_std']:.4f}")
    print()


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Build rolling windows from scaled main panel.")
    parser.add_argument("--panel", type=str, default=str(DEFAULT_PANEL_PATH), help="输入主面板路径")
    parser.add_argument("--scaler", type=str, default=str(DEFAULT_SCALER_PATH), help="已训练好的 scaler.pkl 路径")
    parser.add_argument("--lookback", type=int, default=64, help="滚动窗口长度，例如 64")
    parser.add_argument("--main_horizon", type=int, default=5, help="主任务 horizon，例如 5")
    parser.add_argument(
        "--task_mode",
        type=str,
        default="dir_only",
        choices=["dir_only", "ret_only", "multitask"],
        help="标签模式：dir_only / ret_only / multitask",
    )
    parser.add_argument(
        "--target_style",
        type=str,
        default="excess",
        choices=["absolute", "excess", "band"],
        help="目标风格：absolute / excess / band",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="输出目录；若为空则自动生成为 tensors/window_L{lookback}_H{h}_{target_style}_{task_mode}",
    )
    args = parser.parse_args()

    panel_path = Path(args.panel)
    scaler_path = Path(args.scaler)
    if not panel_path.exists():
        raise FileNotFoundError(f"panel 不存在: {panel_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"scaler 不存在: {scaler_path}")
    if args.lookback <= 1:
        raise ValueError("lookback 必须 > 1")
    if args.main_horizon <= 0:
        raise ValueError("main_horizon 必须 > 0")

    if args.output_dir.strip():
        output_dir = Path(args.output_dir)
    else:
        output_dir = _build_default_output_dir(
            output_root=DEFAULT_OUTPUT_ROOT,
            lookback=int(args.lookback),
            main_horizon=int(args.main_horizon),
            target_style=str(args.target_style),
            task_mode=str(args.task_mode),
        )

    print(f"加载 panel : {panel_path}")
    panel_df = _load_any(panel_path)
    print(f"行数 {len(panel_df)}, 列数 {panel_df.shape[1]}")

    print(f"加载 scaler: {scaler_path}")
    scaler = HybridRobustScaler.load(scaler_path)

    split_data, meta_summary = build_windows(
        panel_df=panel_df,
        scaler=scaler,
        lookback=int(args.lookback),
        main_horizon=int(args.main_horizon),
        task_mode=str(args.task_mode),
        target_style=str(args.target_style),
    )

    print_window_report(split_data, meta_summary)
    save_window_data(split_data, meta_summary, output_dir)


if __name__ == "__main__":
    main()
