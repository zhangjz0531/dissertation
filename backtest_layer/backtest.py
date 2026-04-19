"""
Layer 5: Backtest + Visualization + Significance Tests

输入:
  --daily_returns   layer4_daily_returns.parquet
  --signals         layer4_signals.parquet   (含 prob_calibrated / y_true_dir_5)
  --output_dir      输出目录

输出:
  layer5_cumulative_returns.png   累计净值曲线（val+test）
  layer5_drawdown.png             回撤曲线
  layer5_rolling_sharpe.png       滚动60日Sharpe
  layer5_performance_table.csv    完整绩效指标表
  layer5_significance.json        DM检验 + Bootstrap
  layer5_monthly_returns.csv      月度收益热力图数据
  layer5_monthly_heatmap.png      月度收益热力图
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import norm

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

STRATEGY_COLORS = {
    "DC-only":      "#6c757d",
    "AI-only":      "#0d6efd",
    "Fusion-Gate":  "#198754",
    "Fusion-Vote":  "#fd7e14",
    "Fusion-Conf":  "#6f42c1",
}
DEFAULT_COLOR = "#333333"


# ============================================================
# 绩效工具
# ============================================================

def performance_metrics(r: pd.Series, periods: int = 252) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {}
    ann_ret = r.mean() * periods
    ann_vol = r.std() * math.sqrt(periods)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else float("nan")
    cumr = (1 + r).cumprod()
    mdd = float(((cumr - cumr.cummax()) / cumr.cummax()).min())
    calmar = ann_ret / abs(mdd) if abs(mdd) > 1e-10 else float("nan")
    sortino_denom = r[r < 0].std() * math.sqrt(periods)
    sortino = ann_ret / sortino_denom if sortino_denom > 1e-10 else float("nan")
    return {
        "ann_return":   round(float(ann_ret), 6),
        "ann_vol":      round(float(ann_vol), 6),
        "sharpe":       round(float(sharpe), 4),
        "sortino":      round(float(sortino), 4),
        "mdd":          round(float(mdd), 6),
        "calmar":       round(float(calmar), 4),
        "hit_rate":     round(float((r > 0).mean()), 4),
        "total_return": round(float(cumr.iloc[-1] - 1), 6),
        "n_days":       int(len(r)),
    }


def sharpe_bootstrap_ci(r: pd.Series, n_boot: int = 2000,
                         block: int = 20, seed: int = 42,
                         periods: int = 252):
    """Moving block bootstrap for Sharpe CI"""
    rng = np.random.default_rng(seed)
    arr = r.dropna().values
    n = len(arr)
    if n < block * 2:
        return {"sharpe": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}

    orig_sharpe = arr.mean() / arr.std() * math.sqrt(periods) if arr.std() > 0 else float("nan")
    n_blocks = math.ceil(n / block)
    boot_sharpes = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block + 1, 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        b = arr[idx]
        s = b.mean() / b.std() * math.sqrt(periods) if b.std() > 0 else 0.0
        boot_sharpes.append(s)
    boot = np.array(boot_sharpes)
    return {
        "sharpe": round(float(orig_sharpe), 4),
        "ci_lo":  round(float(np.quantile(boot, 0.025)), 4),
        "ci_hi":  round(float(np.quantile(boot, 0.975)), 4),
        "n_boot": n_boot,
    }


# ============================================================
# DM检验 (returns-based，不需要prob)
# ============================================================

def dm_test_returns(r1: np.ndarray, r2: np.ndarray, horizon: int = 5) -> dict:
    """Diebold-Mariano on squared daily return errors (vs 0 benchmark)"""
    l1 = r1 ** 2   # loss: how far from zero
    l2 = r2 ** 2
    d = l1 - l2    # positive = model1 worse
    T = len(d)
    d_bar = float(np.mean(d))
    lag = max(horizon - 1, 0)
    gamma0 = float(np.mean((d - d_bar) ** 2))
    var_hat = gamma0
    for k in range(1, lag + 1):
        cov = float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
        var_hat += 2.0 * (1.0 - k / (lag + 1.0)) * cov
    dm_stat = d_bar / math.sqrt(max(var_hat / max(T, 1), 1e-12))
    p_value = float(2.0 * (1.0 - norm.cdf(abs(dm_stat))))
    return {
        "dm_stat": round(float(dm_stat), 4),
        "p_value": round(float(p_value), 6),
        "mean_diff_r1_minus_r2": round(float(d_bar), 8),
        "interpretation": "r1 better" if dm_stat < 0 else "r2 better",
    }


# ============================================================
# 图表
# ============================================================

def plot_cumulative_returns(daily: pd.DataFrame, output_path: Path,
                            splits: list = None, title: str = "Cumulative Net Returns"):
    if splits is None:
        splits = ["val", "test"]

    sub = daily[daily["split"].isin(splits)].copy()
    sub = sub.sort_values("timestamp")

    strategies = sub["strategy"].unique().tolist()
    fig, ax = plt.subplots(figsize=(13, 5))

    for strat in strategies:
        s = sub[sub["strategy"] == strat].sort_values("timestamp")
        cumr = (1 + s["net_ret"]).cumprod()
        color = STRATEGY_COLORS.get(strat, DEFAULT_COLOR)
        ax.plot(s["timestamp"], cumr, label=strat, color=color, linewidth=1.8)

    # val/test separator
    if "val" in splits and "test" in splits:
        val_end = daily[daily["split"] == "val"]["timestamp"].max()
        ax.axvline(val_end, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(val_end, ax.get_ylim()[0], "val|test", fontsize=9,
                color="gray", ha="center", va="bottom")

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative Return (net of costs)")
    ax.set_xlabel("")
    ax.legend(frameon=False, fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}x"))
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {output_path}")


def plot_drawdown(daily: pd.DataFrame, output_path: Path,
                  splits: list = None):
    if splits is None:
        splits = ["val", "test"]

    sub = daily[daily["split"].isin(splits)].copy()
    strategies = sub["strategy"].unique().tolist()

    fig, ax = plt.subplots(figsize=(13, 4))
    for strat in strategies:
        s = sub[sub["strategy"] == strat].sort_values("timestamp")
        cumr = (1 + s["net_ret"]).cumprod()
        dd = (cumr - cumr.cummax()) / cumr.cummax()
        color = STRATEGY_COLORS.get(strat, DEFAULT_COLOR)
        ax.fill_between(s["timestamp"], dd, 0, alpha=0.25, color=color)
        ax.plot(s["timestamp"], dd, label=strat, color=color, linewidth=1.4)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Drawdown", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {output_path}")


def plot_rolling_sharpe(daily: pd.DataFrame, output_path: Path,
                        window: int = 60, splits: list = None):
    if splits is None:
        splits = ["val", "test"]

    sub = daily[daily["split"].isin(splits)].copy()
    strategies = sub["strategy"].unique().tolist()

    fig, ax = plt.subplots(figsize=(13, 4))
    for strat in strategies:
        s = sub[sub["strategy"] == strat].sort_values("timestamp")
        rolling = s["net_ret"].rolling(window)
        rs = rolling.mean() / rolling.std() * math.sqrt(252)
        color = STRATEGY_COLORS.get(strat, DEFAULT_COLOR)
        ax.plot(s["timestamp"], rs, label=strat, color=color, linewidth=1.4)

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sharpe (annualised)")
    ax.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {output_path}")


def plot_monthly_heatmap(daily: pd.DataFrame, strategy: str,
                         output_path: Path, split: str = "test"):
    sub = daily[(daily["strategy"] == strategy) & (daily["split"] == split)].copy()
    sub["timestamp"] = pd.to_datetime(sub["timestamp"])
    sub["year"] = sub["timestamp"].dt.year
    sub["month"] = sub["timestamp"].dt.month

    monthly = sub.groupby(["year", "month"])["net_ret"].apply(
        lambda r: float((1 + r).prod() - 1)
    ).reset_index()
    monthly.columns = ["year", "month", "ret"]

    pivot = monthly.pivot(index="year", columns="month", values="ret")
    pivot.columns = [f"{m:02d}" for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(max(10, pivot.shape[1] * 0.8),
                                    max(3, pivot.shape[0] * 0.6)))
    vmax = max(abs(pivot.values[np.isfinite(pivot.values)]).max(), 0.01)
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=9)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=7,
                        color="black" if abs(v) < vmax * 0.6 else "white")

    plt.colorbar(im, ax=ax, format=mticker.FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.set_title(f"Monthly Net Returns — {strategy} ({split})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {output_path}")


# ============================================================
# 主流程
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--daily_returns", type=str, required=True,
                   help="layer4_daily_returns.parquet")
    p.add_argument("--output_dir",    type=str, required=True)
    p.add_argument("--report_splits", type=str, nargs="+", default=["val", "test"])
    p.add_argument("--rolling_window", type=int, default=60)
    p.add_argument("--n_boot",        type=int, default=2000)
    p.add_argument("--boot_block",    type=int, default=20)
    p.add_argument("--horizon",       type=int, default=5)
    p.add_argument("--baseline",      type=str, default="DC-only",
                   help="DM检验的baseline策略名")
    p.add_argument("--heatmap_strategy", type=str, default="Fusion-Gate",
                   help="月度热力图使用哪个策略")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.daily_returns}")
    daily = pd.read_parquet(args.daily_returns)
    daily["timestamp"] = pd.to_datetime(daily["timestamp"])
    strategies = daily["strategy"].unique().tolist()
    print(f"  strategies: {strategies}")
    print(f"  splits:     {daily['split'].unique().tolist()}")

    # ── 绩效表 ────────────────────────────────────────────────
    print("\n[performance table]")
    perf_rows = []
    sharpe_ci_rows = []

    for strat in strategies:
        for split in args.report_splits:
            r = daily[(daily["strategy"] == strat) & (daily["split"] == split)]["net_ret"]
            m = performance_metrics(r)
            if not m:
                continue
            ci = sharpe_bootstrap_ci(r, n_boot=args.n_boot,
                                      block=args.boot_block, seed=42)
            perf_rows.append({"strategy": strat, "split": split, **m})
            sharpe_ci_rows.append({
                "strategy": strat, "split": split,
                **{f"sharpe_{k}": v for k, v in ci.items()},
            })
            print(f"  {strat:16s} {split:5s} | sharpe={m['sharpe']:6.3f} "
                  f"[{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}]  "
                  f"ann_ret={m['ann_return']:.4f}  mdd={m['mdd']:.4f}")

    perf_df = pd.DataFrame(perf_rows)
    ci_df   = pd.DataFrame(sharpe_ci_rows)
    perf_df.to_csv(out / "layer5_performance_table.csv", index=False, encoding="utf-8-sig")
    ci_df.to_csv(out / "layer5_sharpe_ci.csv", index=False, encoding="utf-8-sig")

    # ── DM显著性检验：每个策略 vs baseline ───────────────────
    print(f"\n[DM tests vs baseline={args.baseline}]")
    dm_results = {}
    for split in args.report_splits:
        dm_results[split] = {}
        base_r = daily[(daily["strategy"] == args.baseline) &
                        (daily["split"] == split)]["net_ret"].values

        for strat in strategies:
            if strat == args.baseline:
                continue
            strat_r = daily[(daily["strategy"] == strat) &
                             (daily["split"] == split)]["net_ret"].values
            n = min(len(base_r), len(strat_r))
            if n < 10:
                continue
            res = dm_test_returns(base_r[:n], strat_r[:n], horizon=args.horizon)
            dm_results[split][f"{args.baseline}_vs_{strat}"] = res
            sig = "***" if res["p_value"] < 0.01 else ("**" if res["p_value"] < 0.05
                   else ("*" if res["p_value"] < 0.10 else ""))
            print(f"  {split:5s} | {args.baseline:12s} vs {strat:16s} | "
                  f"DM={res['dm_stat']:6.3f}  p={res['p_value']:.4f} {sig}  "
                  f"({res['interpretation']})")

    with open(out / "layer5_significance.json", "w", encoding="utf-8") as f:
        json.dump(dm_results, f, ensure_ascii=False, indent=2)

    # ── 月度收益 CSV ──────────────────────────────────────────
    all_monthly = []
    for strat in strategies:
        for split in args.report_splits:
            sub = daily[(daily["strategy"] == strat) &
                         (daily["split"] == split)].copy()
            sub["year"]  = sub["timestamp"].dt.year
            sub["month"] = sub["timestamp"].dt.month
            m = sub.groupby(["year", "month"])["net_ret"].apply(
                lambda r: float((1 + r).prod() - 1)
            ).reset_index()
            m.columns = ["year", "month", "monthly_ret"]
            m["strategy"] = strat
            m["split"]    = split
            all_monthly.append(m)

    monthly_df = pd.concat(all_monthly, ignore_index=True)
    monthly_df.to_csv(out / "layer5_monthly_returns.csv",
                       index=False, encoding="utf-8-sig")

    # ── 图表 ──────────────────────────────────────────────────
    print("\n[plots]")
    plot_cumulative_returns(daily, out / "layer5_cumulative_returns.png",
                            splits=args.report_splits)
    plot_drawdown(daily, out / "layer5_drawdown.png",
                  splits=args.report_splits)
    plot_rolling_sharpe(daily, out / "layer5_rolling_sharpe.png",
                        window=args.rolling_window, splits=args.report_splits)
    plot_monthly_heatmap(daily, args.heatmap_strategy,
                         out / "layer5_monthly_heatmap.png",
                         split="test")

    print(f"\n[all outputs saved to] {out}")


if __name__ == "__main__":
    main()