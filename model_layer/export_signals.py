from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from data_layer.config import PANEL_DIR
    DEFAULT_TENSOR_ROOT = PANEL_DIR.parent / "tensors_excess_H5_dir_only"
except Exception:
    DEFAULT_TENSOR_ROOT = Path("./tensors_excess_H5_dir_only")

try:
    from data_layer.config import PANEL_DIR as _PANEL_DIR_FOR_EXPORT
    DEFAULT_TARGET_PANEL_PATH = _PANEL_DIR_FOR_EXPORT / "panel_daily_stock_main.parquet"
except Exception:
    DEFAULT_TARGET_PANEL_PATH = Path("./panel_daily_stock_main.parquet")

try:
    from model_layer.data.loader import TensorBundleDataset, tensor_bundle_collate
    from model_layer.models.lstm import LSTMBaseline
    from model_layer.models.transformer import TimeSeriesTransformer
    from model_layer.evaluation.calibration import calibrate_signals_dataframe
except Exception:
    from data.loader import TensorBundleDataset, tensor_bundle_collate
    from models.lstm import LSTMBaseline
    from models.transformer import TimeSeriesTransformer
    from evaluation.calibration import calibrate_signals_dataframe

from torch.utils.data import DataLoader


ModelName = Literal["lstm", "transformer"]
TaskType = Literal["classification", "multitask"]


def get_device(s="auto"):
    s = str(s).strip().lower()
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def load_ckpt(path):
    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")



def load_bundle_meta(tensor_root: str | Path, split: str = "train") -> Dict:
    path = Path(tensor_root) / f"{split}.pt"
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(path, map_location="cpu")
    return bundle


def load_target_panel_subset(target_panel_path: str | Path, main_horizon: int) -> pd.DataFrame:
    path = Path(target_panel_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 target panel: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    keep_cols = ["asset_id", "timestamp", "split"]
    for c in [
        f"target_ret_{main_horizon}",
        f"target_excess_ret_{main_horizon}",
        f"target_dir_{main_horizon}",
        f"target_excess_dir_{main_horizon}",
        f"target_band_dir_{main_horizon}",
    ]:
        if c in df.columns:
            keep_cols.append(c)

    sub = df[keep_cols].copy()
    sub["asset_id"] = sub["asset_id"].astype(str)
    sub["timestamp"] = pd.to_datetime(sub["timestamp"], errors="coerce")
    if "split" in sub.columns:
        sub["split"] = sub["split"].astype(str)
    return sub.drop_duplicates(subset=["asset_id", "timestamp"], keep="last").reset_index(drop=True)


def enrich_signals_with_panel_targets(
    signals_df: pd.DataFrame,
    target_panel_path: str | Path,
    main_horizon: int,
) -> pd.DataFrame:
    panel_sub = load_target_panel_subset(target_panel_path, main_horizon=main_horizon)
    out = signals_df.copy()
    out["asset_id"] = out["asset_id"].astype(str)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.merge(panel_sub, on=["asset_id", "timestamp"], how="left", suffixes=("", "_panel"))

    rename_map = {}
    if f"target_ret_{main_horizon}" in out.columns:
        rename_map[f"target_ret_{main_horizon}"] = f"y_true_ret_{main_horizon}"
    if f"target_excess_ret_{main_horizon}" in out.columns:
        rename_map[f"target_excess_ret_{main_horizon}"] = f"y_true_excess_ret_{main_horizon}"
    if f"target_dir_{main_horizon}" in out.columns:
        rename_map[f"target_dir_{main_horizon}"] = f"y_true_abs_dir_{main_horizon}"
    if f"target_excess_dir_{main_horizon}" in out.columns:
        rename_map[f"target_excess_dir_{main_horizon}"] = f"y_true_excess_dir_{main_horizon}"
    if f"target_band_dir_{main_horizon}" in out.columns:
        rename_map[f"target_band_dir_{main_horizon}"] = f"y_true_band_dir_{main_horizon}"
    out = out.rename(columns=rename_map)
    return out

def build_model_from_run_config(run_config: Dict, n_features: int) -> nn.Module:
    model_name = str(run_config["model_name"]).strip().lower()
    task_type = str(run_config["task_type"]).strip().lower()
    n_assets = int(run_config.get("n_assets", 16))

    if model_name == "lstm":
        return LSTMBaseline(
            n_features=n_features,
            hidden_size=int(run_config.get("hidden_size", 64)),
            num_layers=int(run_config.get("num_layers", 1)),
            dropout=float(run_config.get("dropout", 0.3)),
            bidirectional=bool(run_config.get("bidirectional", False)),
            use_layernorm=not bool(run_config.get("no_layernorm", False)),
            head_dropout=float(run_config.get("head_dropout", 0.3)),
            use_asset_embedding=bool(run_config.get("use_asset_embedding", True)),
            n_assets=n_assets,
            asset_emb_dim=int(run_config.get("asset_emb_dim", 8)),
            task_type=task_type,
        )
    if model_name == "transformer":
        return TimeSeriesTransformer(
            n_features=n_features,
            d_model=int(run_config.get("d_model", 64)),
            n_heads=int(run_config.get("n_heads", 4)),
            n_layers=int(run_config.get("n_layers", 2)),
            d_ff=int(run_config.get("d_ff", 128)),
            dropout=float(run_config.get("dropout", 0.3)),
            attn_dropout=float(run_config.get("attn_dropout", 0.2)),
            max_len=int(run_config.get("max_len", 256)),
            use_layernorm=not bool(run_config.get("no_layernorm", False)),
            pooling=str(run_config.get("pooling", "mean")),
            causal=bool(run_config.get("causal", False)),
            use_asset_embedding=bool(run_config.get("use_asset_embedding", True)),
            n_assets=n_assets,
            asset_emb_dim=int(run_config.get("asset_emb_dim", 8)),
            task_type=task_type,
        )
    raise ValueError(model_name)


@torch.no_grad()
def infer_split(
    model: nn.Module,
    ds: TensorBundleDataset,
    task_type: str,
    device: torch.device,
    batch_size: int = 256,
    return_attn: bool = False,
):
    model.eval()
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False,
                    collate_fn=tensor_bundle_collate)

    all_logits = []
    all_targets = []
    meta_rows = []
    attn_list = []

    for batch in dl:
        X = batch["X"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        asset_int = batch.get("asset_int", None)
        if asset_int is not None:
            asset_int = asset_int.to(device, non_blocking=True)

        use_ae = bool(getattr(model, "use_asset_embedding", False))
        forward_kwargs = dict(X=X, mask=mask, asset_int=asset_int if use_ae else None)

        if return_attn and isinstance(model, TimeSeriesTransformer):
            preds, attn = model(**forward_kwargs, return_attn=True)
            if attn is not None and len(attn) > 0:
                last = attn[-1].detach().cpu().float().numpy()  # [B, H, L, L]
                last_head_mean = last.mean(axis=1)
                attn_list.append(last_head_mean)
        else:
            preds = model(**forward_kwargs)

        if task_type == "classification":
            all_logits.append(preds.detach().cpu().view(-1).float())
            all_targets.append(y.detach().cpu().view(-1).float())
        elif task_type == "multitask":
            all_logits.append(preds[:, 0].detach().cpu().view(-1).float())
            all_targets.append(y[:, 0].detach().cpu().view(-1).float())
        else:
            raise ValueError(task_type)

        if "meta" in batch:
            for m in batch["meta"]:
                meta_rows.append(dict(m))

    logits = torch.cat(all_logits, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy().astype(np.int64)
    probs = torch.sigmoid(torch.from_numpy(logits).float()).numpy().astype(np.float32)
    attn_stack = None
    if return_attn and len(attn_list) > 0:
        attn_stack = np.concatenate(attn_list, axis=0)
    return {
        "meta": meta_rows, "logits": logits.astype(np.float32),
        "probs": probs, "targets": targets, "attn_last_layer": attn_stack,
    }


def build_signals_df(
    model_name: str, seed: int, task_type: str, main_horizon: int,
    result_by_split: Dict[str, Dict],
) -> pd.DataFrame:
    rows = []
    for split, pack in result_by_split.items():
        meta = pack["meta"]
        logits = pack["logits"]
        probs = pack["probs"]
        targets = pack["targets"]
        n = len(meta)
        for i in range(n):
            m = meta[i]
            rows.append({
                "model_name": model_name,
                "seed": int(seed),
                "task_type": task_type,
                "split": split,
                "asset_id": m.get("asset_id"),
                "asset_int": int(m.get("asset_int", 0)),
                "timestamp": m.get("timestamp"),
                f"y_true_dir_{main_horizon}": int(targets[i]),
                "logit_raw": float(logits[i]),
                "prob_raw": float(probs[i]),
                "prob_calibrated": float("nan"),  # 占位，后面 calibration 会填
            })
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["split", "asset_id", "timestamp"]).reset_index(drop=True)
    return df


def export_run(
    run_dir,
    tensor_root=DEFAULT_TENSOR_ROOT,
    ckpt_name="best.pt",
    splits=("train", "val", "test"),
    device="auto",
    batch_size=256,
    save_attention=False,
    calibration_method="isotonic",
    target_panel_path=DEFAULT_TARGET_PANEL_PATH,
):
    run_dir = Path(run_dir)
    ckpt_path = run_dir / "checkpoints" / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    device = get_device(device)
    ckpt = load_ckpt(ckpt_path)
    run_config = ckpt["run_config"]
    task_type = str(run_config["task_type"]).strip().lower()
    model_name = str(run_config["model_name"]).strip().lower()
    seed = int(run_config.get("seed", 42))
    best_threshold = float(ckpt.get("best_threshold", run_config.get("threshold", 0.5)))

    if task_type not in {"classification", "multitask"}:
        raise ValueError(task_type)

    bundle_meta = load_bundle_meta(tensor_root=tensor_root, split=splits[0])
    target_style = bundle_meta.get("target_style")
    dir_target_col = bundle_meta.get("dir_target_col")
    ret_target_col = bundle_meta.get("ret_target_col")

    # 第一个 split 用于拿 feature 维度
    first_ds = TensorBundleDataset(
        tensor_root=tensor_root, split=splits[0],
        view_mode="sequence", label_mode="dir_only" if task_type == "classification" else "multitask",
        include_meta=True, include_asset_id=True,
    )
    n_features = int(first_ds.n_features)

    model = build_model_from_run_config(run_config, n_features=n_features).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 推理
    result_by_split: Dict[str, Dict] = {}
    for s in splits:
        ds = TensorBundleDataset(
            tensor_root=tensor_root, split=s,
            view_mode="sequence",
            label_mode="dir_only" if task_type == "classification" else "multitask",
            include_meta=True, include_asset_id=True,
        )
        return_attn = bool(save_attention and model_name == "transformer")
        pack = infer_split(
            model=model, ds=ds, task_type=task_type, device=device,
            batch_size=batch_size, return_attn=return_attn,
        )
        result_by_split[s] = pack

    # DataFrame
    main_horizon = int(first_ds.main_horizon)
    df = build_signals_df(
        model_name=model_name, seed=seed, task_type=task_type,
        main_horizon=main_horizon, result_by_split=result_by_split,
    )
    df["best_threshold"] = float(best_threshold)
    df["target_style"] = target_style if target_style is not None else "unknown"
    df["dir_target_col"] = dir_target_col if dir_target_col is not None else ""
    df["ret_target_col"] = ret_target_col if ret_target_col is not None else ""

    df = enrich_signals_with_panel_targets(
        signals_df=df,
        target_panel_path=target_panel_path,
        main_horizon=main_horizon,
    )

    # ============================================================
    # 关键修复：强制调 calibration 填充 prob_calibrated
    # ============================================================
    cal_summary = {}
    if calibration_method != "none":
        cal_result = calibrate_signals_dataframe(
            df=df.drop(columns=["prob_calibrated"]),
            method=calibration_method,
            y_col=f"y_true_dir_{main_horizon}",
            split_col="split",
            prob_col="prob_raw",
            logit_col="logit_raw",
            threshold=best_threshold,
        )
        df = cal_result["signals"]
        cal_summary = cal_result["summary"]
        # 保存 calibrator
        cal_path = run_dir / f"calibrator_{calibration_method}.pkl"
        cal_result["calibrator"].save(cal_path)
        with open(run_dir / f"calibration_{calibration_method}.summary.json", "w", encoding="utf-8") as f:
            json.dump(cal_summary, f, ensure_ascii=False, indent=2)

    # save
    out_dir = run_dir / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)
    pq_path = out_dir / f"signals_{model_name}_seed{seed}.parquet"
    csv_path = out_dir / f"signals_{model_name}_seed{seed}.csv"
    df.to_parquet(pq_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    if save_attention:
        for s, pack in result_by_split.items():
            if pack.get("attn_last_layer") is not None:
                np.save(out_dir / f"attention_{model_name}_seed{seed}_{s}.npy",
                        pack["attn_last_layer"])

    summary = {
        "run_dir": str(run_dir),
        "ckpt_path": str(ckpt_path),
        "model_name": model_name,
        "seed": seed,
        "task_type": task_type,
        "best_threshold": float(best_threshold),
        "splits": list(splits),
        "signals_parquet": str(pq_path),
        "signals_csv": str(csv_path),
        "target_panel_path": str(target_panel_path),
        "target_style": target_style,
        "dir_target_col": dir_target_col,
        "ret_target_col": ret_target_col,
        "calibration": cal_summary,
    }
    with open(out_dir / "export_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[export] {pq_path}")
    if cal_summary:
        for s in ["val", "test"]:
            if s in cal_summary.get("post", {}):
                pre = cal_summary["pre"][s]
                post = cal_summary["post"][s]
                print(f"  [{s}] pre ece={pre['ece']:.4f} brier={pre['brier']:.4f}  "
                      f"post ece={post['ece']:.4f} brier={post['brier']:.4f}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--tensor_root", type=str, default=str(DEFAULT_TENSOR_ROOT))
    p.add_argument("--ckpt_name", type=str, default="best.pt")
    p.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"])
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--save_attention", action="store_true")
    p.add_argument("--calibration_method", type=str, default="isotonic",
                   choices=["none", "platt", "isotonic", "temperature"])
    p.add_argument("--target_panel_path", type=str, default=str(DEFAULT_TARGET_PANEL_PATH))
    args = p.parse_args()

    export_run(
        run_dir=args.run_dir, tensor_root=args.tensor_root,
        ckpt_name=args.ckpt_name, splits=tuple(args.splits),
        device=args.device, batch_size=args.batch_size,
        save_attention=args.save_attention,
        calibration_method=args.calibration_method,
        target_panel_path=args.target_panel_path,
    )


if __name__ == "__main__":
    main()
