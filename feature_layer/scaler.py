from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
#  Adaptive Hybrid Robust Scaler
#  - 对称特征: robust / adaptive z-score
#  - 非负偏态 / 长尾特征: sign-log1p + adaptive z-score
#  - 离散标签: 不变
#
#  核心修复点：
#  1) 不再只用 MAD 当 scale，避免稀疏右偏列 scale 塌缩
#  2) 对 dc_osv_* / macd_hist 等长尾列也做 log 压缩
# ============================================================
class HybridRobustScaler:
    """
    混合 robust scaler：

    transform 流程（对每列）：
      1. 如果列名匹配 LOG_TRANSFORM 规则 -> 先 sign(x) * log1p(|x|)
      2. z = (x_t - median) / adaptive_scale
         其中 adaptive_scale = max(1.4826*MAD, IQR/1.349, (q95-median)/2.5, eps)
      3. clip 到 ±clip_value
      4. NaN -> 0

    为什么不用纯 MAD：
      对 credit_stress 这种大量接近 0、少数尖峰的列，MAD 往往极小，
      会导致 scale 塌缩，进而大量样本被打到 +5 clip。
    """

    def __init__(
        self,
        clip_value: float = 5.0,
        skip_cols: Optional[List[str]] = None,
        log_cols: Optional[List[str]] = None,
        eps: float = 1e-8,
    ):
        self.clip_value = float(clip_value)
        self.skip_cols = list(skip_cols or [])
        self.log_cols = list(log_cols or [])
        self.eps = float(eps)

        self.feature_cols_: List[str] = []
        self.scaled_cols_: List[str] = []
        self.logged_cols_: List[str] = []
        self.skipped_cols_: List[str] = []

        self.median_: Dict[str, float] = {}
        self.mad_: Dict[str, float] = {}
        self.scale_: Dict[str, float] = {}
        self.scale_source_: Dict[str, str] = {}

        self.fitted_: bool = False

    @staticmethod
    def _signed_log1p(x: np.ndarray) -> np.ndarray:
        """
        对称 log 压缩：
          sign(x) * log(1 + |x|)
        既能压正尾，也能压负尾。
        """
        return np.sign(x) * np.log1p(np.abs(x))

    def _maybe_log(self, c: str, x: np.ndarray) -> np.ndarray:
        if c in self.log_cols:
            return self._signed_log1p(x)
        return x

    @staticmethod
    def _compute_adaptive_scale(x_t: np.ndarray, med: float, eps: float) -> Tuple[float, str]:
        """
        自适应尺度：
          - robust_scale = 1.4826 * MAD
          - iqr_scale    = (q75 - q25) / 1.349
          - tail_scale   = (q95 - median) / 2.5

        取三者最大值，避免：
          - MAD 对稀疏右偏列过小
          - IQR 对极端长尾不够敏感
          - q95 对普通对称列不够稳
        """
        mad = float(np.median(np.abs(x_t - med)))
        robust_scale = 1.4826 * mad

        q25, q75 = np.quantile(x_t, [0.25, 0.75])
        iqr_scale = float((q75 - q25) / 1.349)

        q95 = float(np.quantile(x_t, 0.95))
        tail_scale = float((q95 - med) / 2.5)

        candidates = [
            ("mad", robust_scale),
            ("iqr", iqr_scale),
            ("q95", tail_scale),
        ]
        candidates = [(name, val) for name, val in candidates if np.isfinite(val) and val > eps]

        if not candidates:
            return 1.0, "unit"

        source, scale = max(candidates, key=lambda kv: kv[1])
        return float(scale), source

    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> "HybridRobustScaler":
        self.feature_cols_ = list(feature_cols)
        self.scaled_cols_ = []
        self.logged_cols_ = []
        self.skipped_cols_ = []

        self.median_ = {}
        self.mad_ = {}
        self.scale_ = {}
        self.scale_source_ = {}

        for c in self.feature_cols_:
            if c in self.skip_cols:
                self.skipped_cols_.append(c)
                continue

            x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
            x = x[np.isfinite(x)]

            if len(x) == 0:
                self.skipped_cols_.append(c)
                self.median_[c] = 0.0
                self.mad_[c] = 1.0
                self.scale_[c] = 1.0
                self.scale_source_[c] = "unit"
                continue

            x_t = self._maybe_log(c, x)
            med = float(np.median(x_t))
            mad = float(np.median(np.abs(x_t - med)))
            scale, scale_source = self._compute_adaptive_scale(x_t, med, self.eps)

            self.median_[c] = med
            self.mad_[c] = mad
            self.scale_[c] = scale
            self.scale_source_[c] = scale_source

            self.scaled_cols_.append(c)
            if c in self.log_cols:
                self.logged_cols_.append(c)

        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Scaler 未 fit")

        out = df.copy()

        for c in self.scaled_cols_:
            x = pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=np.float64)
            x_t = self._maybe_log(c, x)

            scale = self.scale_.get(c, 1.0) + self.eps
            z = (x_t - self.median_[c]) / scale
            z = np.clip(z, -self.clip_value, self.clip_value)
            z = np.where(np.isfinite(z), z, 0.0)

            out[c] = z.astype(np.float32)

        for c in self.skipped_cols_:
            x = pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=np.float64)
            x = np.where(np.isfinite(x), x, 0.0)
            out[c] = x.astype(np.float32)

        return out

    def fit_transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        return self.fit(df, feature_cols).transform(df)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "clip_value": self.clip_value,
                    "skip_cols": self.skip_cols,
                    "log_cols": self.log_cols,
                    "eps": self.eps,
                    "feature_cols": self.feature_cols_,
                    "scaled_cols": self.scaled_cols_,
                    "logged_cols": self.logged_cols_,
                    "skipped_cols": self.skipped_cols_,
                    "median": self.median_,
                    "mad": self.mad_,
                    "scale": self.scale_,
                    "scale_source": self.scale_source_,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "HybridRobustScaler":
        with open(path, "rb") as f:
            obj = pickle.load(f)

        s = cls(
            clip_value=obj["clip_value"],
            skip_cols=obj["skip_cols"],
            log_cols=obj.get("log_cols", []),
            eps=obj["eps"],
        )
        s.feature_cols_ = obj["feature_cols"]
        s.scaled_cols_ = obj["scaled_cols"]
        s.logged_cols_ = obj.get("logged_cols", [])
        s.skipped_cols_ = obj["skipped_cols"]
        s.median_ = obj["median"]
        s.mad_ = obj["mad"]

        if "scale" in obj:
            s.scale_ = obj["scale"]
        else:
            # 向后兼容旧版 scaler.pkl
            s.scale_ = {k: 1.4826 * float(v) for k, v in s.mad_.items()}

        if "scale_source" in obj:
            s.scale_source_ = obj["scale_source"]
        else:
            s.scale_source_ = {k: "mad" for k in s.scale_.keys()}

        s.fitted_ = True
        return s


# 向后兼容
RobustScaler = HybridRobustScaler


# ============================================================
#  特征列识别
# ============================================================
NON_FEATURE_COLS = {
    "asset_id",
    "timestamp",
    "split",
    "is_warmup",
    "is_usable_for_model",
    "row_num_in_asset",
    "wf_fold",
    "wf_split",
    "open",
    "high",
    "low",
    "close",
    "volume",   # 原始成交量不直接送进 scaler
}

# 已是离散标签
SKIP_SCALE_PREFIXES = (
    "dc_event_",
    "dc_trend_",
)

# 需要 log 压缩的列：右偏严重 / 长尾 / 稀疏正值 / 对称长尾
LOG_TRANSFORM_PREFIXES = (
    "dc_age_",
    "dc_density_",
    "dc_osv_",
    "credit_stress",
    "vix_level",
    "ust10y",
    "macd_hist",
)


def detect_feature_cols(df: pd.DataFrame) -> List[str]:
    feats = []
    for c in df.columns:
        if c in NON_FEATURE_COLS:
            continue
        if c.startswith("target_") or c.startswith("benchmark_"):
            continue
        feats.append(c)
    return feats


def detect_skip_scale_cols(feature_cols: List[str]) -> List[str]:
    return [c for c in feature_cols if any(c.startswith(p) for p in SKIP_SCALE_PREFIXES)]


def detect_log_cols(feature_cols: List[str]) -> List[str]:
    return [c for c in feature_cols if any(c.startswith(p) for p in LOG_TRANSFORM_PREFIXES)]


# ============================================================
#  Sanity report
# ============================================================
def sanity_report(
    scaler: HybridRobustScaler,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict:
    report = {
        "n_features_total": len(scaler.feature_cols_),
        "n_features_scaled": len(scaler.scaled_cols_),
        "n_features_logged": len(scaler.logged_cols_),
        "n_features_skipped": len(scaler.skipped_cols_),
        "scaled_cols": scaler.scaled_cols_,
        "logged_cols": scaler.logged_cols_,
        "skipped_cols": scaler.skipped_cols_,
        "scale_source": scaler.scale_source_,
        "per_split": {},
    }

    print("\n  Scaler 训练完成:")
    print(f"    特征总数  : {report['n_features_total']}")
    print(f"    归一化列  : {report['n_features_scaled']}")
    print(f"    log 列    : {report['n_features_logged']} ({scaler.logged_cols_})")
    print(f"    跳过列    : {report['n_features_skipped']} ({scaler.skipped_cols_})")

    # 看看 adaptive scale 来源分布
    source_counts = pd.Series(list(scaler.scale_source_.values())).value_counts().to_dict()
    print(f"    尺度来源  : {source_counts}")

    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        scaled = scaler.transform(df)
        scaled_only = scaled[scaler.scaled_cols_].to_numpy(dtype=np.float64)

        info = {
            "n_rows": int(len(scaled)),
            "global_mean": float(np.nanmean(scaled_only)),
            "global_std": float(np.nanstd(scaled_only)),
            "global_min": float(np.nanmin(scaled_only)),
            "global_max": float(np.nanmax(scaled_only)),
            "n_nan_after_scale": int(np.isnan(scaled_only).sum()),
            "n_clipped_high": int((scaled_only >= scaler.clip_value - 1e-6).sum()),
            "n_clipped_low": int((scaled_only <= -scaler.clip_value + 1e-6).sum()),
        }
        report["per_split"][split_name] = info

        print(
            f"\n  [{split_name:5s}] rows={info['n_rows']:6d}  "
            f"mean={info['global_mean']:+.4f}  std={info['global_std']:.4f}  "
            f"range=[{info['global_min']:+.2f}, {info['global_max']:+.2f}]  "
            f"NaN_after={info['n_nan_after_scale']}  "
            f"clipped_hi={info['n_clipped_high']}  clipped_lo={info['n_clipped_low']}"
        )

    train_scaled = scaler.transform(train_df)
    clip_rates = {}
    for c in scaler.scaled_cols_:
        x = train_scaled[c].to_numpy(dtype=np.float64)
        rate = float(((x >= scaler.clip_value - 1e-6) | (x <= -scaler.clip_value + 1e-6)).mean())
        clip_rates[c] = rate

    top_clipped = sorted(clip_rates.items(), key=lambda kv: -kv[1])[:8]
    print("\n  Train 集 clip 比例 top 8:")
    for c, r in top_clipped:
        flag = "⚠️" if r > 0.05 else "✓"
        log_tag = " [log]" if c in scaler.logged_cols_ else ""
        src_tag = f" [{scaler.scale_source_.get(c, 'na')}]"
        print(f"    {flag}  {c:24s} {r*100:5.2f}%{log_tag}{src_tag}")

    return report


# ============================================================
#  CLI
# ============================================================
if __name__ == "__main__":
    from data_layer.config import PANEL_DIR

    panel_path = PANEL_DIR / "panel_daily_stock_main.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"找不到 panel: {panel_path}")

    print(f"加载 panel: {panel_path}")
    df = pd.read_parquet(panel_path)
    print(f"行数 {len(df)}, 列数 {df.shape[1]}")

    feat_cols = detect_feature_cols(df)
    skip_cols = detect_skip_scale_cols(feat_cols)
    log_cols = detect_log_cols(feat_cols)

    print(f"\n识别特征列: {len(feat_cols)} 个")
    print(f"  跳过归一化: {skip_cols}")
    print(f"  log 压缩  : {log_cols}")

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    print(f"\n切分: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    scaler = HybridRobustScaler(
        clip_value=5.0,
        skip_cols=skip_cols,
        log_cols=log_cols,
    )
    scaler.fit(train_df, feat_cols)
    sanity_report(scaler, train_df, val_df, test_df)

    out_path = PANEL_DIR.parent / "tensors" / "scaler.pkl"
    scaler.save(out_path)
    print(f"\n✅ Scaler 已保存: {out_path}")