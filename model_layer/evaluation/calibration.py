from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

CalibrationMethod = Literal["none", "platt", "isotonic", "temperature"]


def _ensure_1d(x, name="x"):
    arr = np.asarray(x)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.shape}")
    return arr


def _sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _logit_np(p, eps=1e-8):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def compute_ece(y_true, prob, n_bins=15):
    y_true = _ensure_1d(y_true).astype(np.int64)
    prob = np.clip(_ensure_1d(prob).astype(np.float64), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    if len(prob) == 0:
        return 0.0
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            m = (prob >= lo) & (prob < hi)
        else:
            m = (prob >= lo) & (prob <= hi)
        if not np.any(m):
            continue
        conf = float(np.mean(prob[m]))
        acc = float(np.mean(y_true[m]))
        ece += abs(acc - conf) * float(np.mean(m))
    return float(ece)


def binary_metrics(y_true, prob, threshold=0.5, ece_bins=15):
    y_true = _ensure_1d(y_true).astype(np.int64)
    prob = np.clip(_ensure_1d(prob).astype(np.float64), 1e-8, 1 - 1e-8)
    pred = (prob >= float(threshold)).astype(np.int64)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-12))
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    mcc = float(((tp * tn) - (fp * fn)) / denom)
    out = {
        "acc": float((pred == y_true).mean()),
        "precision": precision, "recall": recall, "f1": f1, "mcc": mcc,
        "brier": float(brier_score_loss(y_true, prob)),
        "ece": float(compute_ece(y_true, prob, ece_bins)),
        "positive_rate_pred": float(pred.mean()),
        "positive_rate_true": float(y_true.mean()),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, prob))
    except Exception:
        out["auc"] = float("nan")
    try:
        out["nll"] = float(log_loss(y_true, prob, labels=[0, 1]))
    except Exception:
        out["nll"] = float("nan")
    return out


def fit_temperature_scalar(logits, y_true, max_iter=200, lr=0.01):
    if torch is None:
        raise ImportError("temperature scaling requires torch")
    x = torch.tensor(_ensure_1d(logits).astype(np.float32))
    y = torch.tensor(_ensure_1d(y_true).astype(np.float32))
    log_temp = torch.tensor([0.0], requires_grad=True)
    opt = torch.optim.Adam([log_temp], lr=float(lr))
    best_loss, best_temp = float("inf"), 1.0
    for _ in range(int(max_iter)):
        opt.zero_grad()
        temp = torch.exp(log_temp).clamp(min=1e-3, max=100.0)
        loss = F.binary_cross_entropy_with_logits(x / temp, y)
        loss.backward()
        opt.step()
        cur_loss = float(loss.detach().cpu().item())
        cur_temp = float(temp.detach().cpu().item())
        if cur_loss < best_loss:
            best_loss, best_temp = cur_loss, cur_temp
    return float(best_temp)


@dataclass
class CalibratorConfig:
    method: CalibrationMethod = "isotonic"
    threshold: float = 0.5
    ece_bins: int = 15


class BinaryCalibrator:
    def __init__(self, method: CalibrationMethod = "isotonic", threshold: float = 0.5, ece_bins: int = 15):
        method = str(method).lower()
        if method not in {"none", "platt", "isotonic", "temperature"}:
            raise ValueError(method)
        self.config = CalibratorConfig(method=method, threshold=float(threshold), ece_bins=int(ece_bins))
        self.is_fitted_ = False
        self.platt_model_ = None
        self.isotonic_model_ = None
        self.temperature_ = None

    def fit(self, y_true, prob_raw=None, logit_raw=None):
        y_true = _ensure_1d(y_true).astype(np.int64)
        m = self.config.method
        if m == "none":
            self.is_fitted_ = True
            return self
        if m == "platt":
            if logit_raw is None:
                if prob_raw is None:
                    raise ValueError("platt 需要 logit_raw 或 prob_raw")
                logit_raw = _logit_np(prob_raw)
            X = _ensure_1d(logit_raw).reshape(-1, 1).astype(np.float64)
            model = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
            model.fit(X, y_true)
            self.platt_model_ = model
            self.is_fitted_ = True
            return self
        if m == "isotonic":
            if prob_raw is None:
                if logit_raw is None:
                    raise ValueError("isotonic 需要 prob_raw 或 logit_raw")
                prob_raw = _sigmoid_np(logit_raw)
            X = _ensure_1d(prob_raw).astype(np.float64)
            model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
            model.fit(X, y_true)
            self.isotonic_model_ = model
            self.is_fitted_ = True
            return self
        if m == "temperature":
            if logit_raw is None:
                if prob_raw is None:
                    raise ValueError("temperature 需要 logit_raw 或 prob_raw")
                logit_raw = _logit_np(prob_raw)
            self.temperature_ = fit_temperature_scalar(logit_raw, y_true)
            self.is_fitted_ = True
            return self
        raise ValueError(m)

    def predict_proba(self, prob_raw=None, logit_raw=None):
        if not self.is_fitted_:
            raise RuntimeError("未 fit")
        m = self.config.method
        if m == "none":
            if prob_raw is not None:
                return np.asarray(prob_raw, dtype=np.float64)
            return _sigmoid_np(logit_raw)
        if m == "platt":
            if logit_raw is None:
                logit_raw = _logit_np(prob_raw)
            X = _ensure_1d(logit_raw).reshape(-1, 1).astype(np.float64)
            return self.platt_model_.predict_proba(X)[:, 1]
        if m == "isotonic":
            if prob_raw is None:
                prob_raw = _sigmoid_np(logit_raw)
            return self.isotonic_model_.predict(_ensure_1d(prob_raw).astype(np.float64))
        if m == "temperature":
            if logit_raw is None:
                logit_raw = _logit_np(prob_raw)
            return _sigmoid_np(_ensure_1d(logit_raw).astype(np.float64) / float(self.temperature_))
        raise ValueError(m)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "config": asdict(self.config), "is_fitted": self.is_fitted_,
                "platt_model": self.platt_model_, "isotonic_model": self.isotonic_model_,
                "temperature": self.temperature_,
            }, f)

    @classmethod
    def load(cls, path):
        with open(Path(path), "rb") as f:
            payload = pickle.load(f)
        obj = cls(**payload["config"])
        obj.is_fitted_ = bool(payload["is_fitted"])
        obj.platt_model_ = payload.get("platt_model")
        obj.isotonic_model_ = payload.get("isotonic_model")
        obj.temperature_ = payload.get("temperature")
        return obj

    def summary(self) -> Dict:
        return {
            "config": asdict(self.config),
            "is_fitted": bool(self.is_fitted_),
            "has_platt_model": self.platt_model_ is not None,
            "has_isotonic_model": self.isotonic_model_ is not None,
            "temperature": None if self.temperature_ is None else float(self.temperature_),
        }


def load_signals(path):
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(path.suffix)


def save_signals(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        raise ValueError(path.suffix)


def calibrate_signals_dataframe(
    df: pd.DataFrame,
    method: CalibrationMethod = "isotonic",
    y_col: str = "y_true_dir_5",
    split_col: str = "split",
    prob_col: str = "prob_raw",
    logit_col: str = "logit_raw",
    threshold: float = 0.5,
    ece_bins: int = 15,
) -> Dict:
    if prob_col not in df.columns and logit_col not in df.columns:
        raise ValueError(f"至少需要 {prob_col} 或 {logit_col} 其中之一")
    for c in [y_col, split_col]:
        if c not in df.columns:
            raise ValueError(f"缺列: {c}")

    out_df = df.copy()
    out_df[split_col] = out_df[split_col].astype(str)

    val_df = out_df[out_df[split_col] == "val"].copy()
    test_df = out_df[out_df[split_col] == "test"].copy()
    if len(val_df) == 0:
        raise ValueError("val 子集为空；calibration 必须 fit 在 val 上")

    y_val = val_df[y_col].to_numpy()
    prob_val = val_df[prob_col].to_numpy() if prob_col in val_df.columns else None
    logit_val = val_df[logit_col].to_numpy() if logit_col in val_df.columns else None

    pre_metrics = {
        "val": binary_metrics(
            y_val,
            prob_val if prob_val is not None else _sigmoid_np(logit_val),
            threshold=threshold, ece_bins=ece_bins,
        )
    }
    if len(test_df) > 0:
        y_test = test_df[y_col].to_numpy()
        prob_test = test_df[prob_col].to_numpy() if prob_col in test_df.columns else None
        logit_test = test_df[logit_col].to_numpy() if logit_col in test_df.columns else None
        pre_metrics["test"] = binary_metrics(
            y_test,
            prob_test if prob_test is not None else _sigmoid_np(logit_test),
            threshold=threshold, ece_bins=ece_bins,
        )

    calibrator = BinaryCalibrator(method=method, threshold=threshold, ece_bins=ece_bins)
    calibrator.fit(y_true=y_val, prob_raw=prob_val, logit_raw=logit_val)

    full_prob_raw = out_df[prob_col].to_numpy() if prob_col in out_df.columns else None
    full_logit_raw = out_df[logit_col].to_numpy() if logit_col in out_df.columns else None
    out_df["prob_calibrated"] = calibrator.predict_proba(prob_raw=full_prob_raw, logit_raw=full_logit_raw)

    post_metrics = {}
    for s in ["val", "test"]:
        sub = out_df[out_df[split_col] == s]
        if len(sub) == 0:
            continue
        post_metrics[s] = binary_metrics(
            sub[y_col].to_numpy(), sub["prob_calibrated"].to_numpy(),
            threshold=threshold, ece_bins=ece_bins,
        )

    return {
        "calibrator": calibrator,
        "signals": out_df,
        "summary": {
            "method": method, "threshold": threshold, "ece_bins": ece_bins,
            "n_val": int(len(val_df)), "n_test": int(len(test_df)),
            "pre": pre_metrics, "post": post_metrics,
            "calibrator": calibrator.summary(),
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--signals_path", type=str, required=True)
    p.add_argument("--method", type=str, default="isotonic",
                   choices=["none", "platt", "isotonic", "temperature"])
    p.add_argument("--output_signals", type=str, default="")
    p.add_argument("--output_calibrator", type=str, default="")
    p.add_argument("--output_summary", type=str, default="")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--ece_bins", type=int, default=15)
    args = p.parse_args()

    sp = Path(args.signals_path)
    df = load_signals(sp)
    out_sig = Path(args.output_signals) if args.output_signals else sp.with_name(
        sp.stem + f"_calibrated_{args.method}" + sp.suffix)
    out_cal = Path(args.output_calibrator) if args.output_calibrator else sp.with_name(
        f"calibrator_{args.method}.pkl")
    out_sum = Path(args.output_summary) if args.output_summary else sp.with_name(
        sp.stem + f"_calibration_{args.method}.summary.json")

    r = calibrate_signals_dataframe(
        df=df, method=args.method,
        threshold=args.threshold, ece_bins=args.ece_bins,
    )
    save_signals(r["signals"], out_sig)
    r["calibrator"].save(out_cal)
    with open(out_sum, "w", encoding="utf-8") as f:
        json.dump(r["summary"], f, ensure_ascii=False, indent=2)

    print(f"signals -> {out_sig}")
    print(f"calibrator -> {out_cal}")
    print(f"summary -> {out_sum}")
    for s in ["val", "test"]:
        if s in r["summary"]["post"]:
            pre = r["summary"]["pre"][s]
            post = r["summary"]["post"][s]
            print(f"[{s}] pre ece={pre['ece']:.4f} brier={pre['brier']:.4f} auc={pre['auc']:.4f}")
            print(f"[{s}] post ece={post['ece']:.4f} brier={post['brier']:.4f} auc={post['auc']:.4f}")


if __name__ == "__main__":
    main()
