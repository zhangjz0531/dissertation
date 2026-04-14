
import argparse
import base64
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# Dissertation dashboard generator (FINAL)
#
# Final system:
#   H5 Transformer v5 base strategy
#   No overlay enabled in final practical system
#
# This script creates a self-contained HTML dashboard for the final
# dissertation demo / appendix / viva presentation.
# =========================================================


# -----------------------------
# Default paths for the current final run
# -----------------------------
DEFAULT_FEATURE_DATA = r"D:\python\dissertation\Data Acquisition\cleaned_datasets\main_experiment_h5.csv"
DEFAULT_FIXED_PREDICTIONS = r"D:\python\dissertation\Model Runs\final_run_20260413\fixed_predictions_transformer_v5\transformer_predictions_all_splits_fixed.csv"
DEFAULT_MODEL_SUMMARY = r"D:\python\dissertation\Model Runs\final_run_20260413\main_experiment_h5_v5_transformer_seq30_lr0.0003_wd0.0005_do0.2\metrics_summary.json"
DEFAULT_OVERLAY_SUMMARY = r"D:\python\dissertation\Model Runs\final_run_20260413\transformer_predictions_all_splits_fixed_transformer_overlay_selector_v2\metrics_summary.json"
DEFAULT_OUT_HTML = r"D:\python\dissertation\Model Runs\final_run_20260413\dissertation_trading_dashboard_transformer_final.html"


# -----------------------------
# Helpers
# -----------------------------
def safe_read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(p)


def safe_read_json(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def annualized_return(final_equity: float, total_days: int) -> float:
    total_days = max(1, int(total_days))
    if final_equity <= 0:
        return -1.0
    return float(final_equity ** (252.0 / total_days) - 1.0)


def sharpe_ratio(period_returns: List[float], horizon: int) -> float:
    if len(period_returns) <= 1:
        return 0.0
    arr = np.array(period_returns, dtype=float)
    std = arr.std(ddof=1)
    if std <= 1e-12:
        return 0.0
    return float((arr.mean() / std) * np.sqrt(252.0 / horizon))


def max_drawdown_from_equity_curve(equity_curve: List[float]) -> float:
    eq = np.array(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(eq)
    dd = eq / np.maximum(running_max, 1e-12) - 1.0
    return float(dd.min())


def turnover(prev_w: Dict[str, float], new_w: Dict[str, float]) -> float:
    names = set(prev_w.keys()) | set(new_w.keys())
    return float(sum(abs(prev_w.get(n, 0.0) - new_w.get(n, 0.0)) for n in names))


def fmt_pct(x: Optional[float], digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "N/A"
    return f"{x * 100:.{digits}f}%"


def fmt_num(x: Optional[float], digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "N/A"
    return f"{x:.{digits}f}"


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# -----------------------------
# Load inputs
# -----------------------------
def load_inputs(
    fixed_predictions_path: str,
    feature_data_path: str,
    model_summary_path: str,
    overlay_summary_path: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Optional[Dict]]:
    pred = safe_read_csv(fixed_predictions_path)
    feat = safe_read_csv(feature_data_path)
    model_summary = safe_read_json(model_summary_path)

    overlay_summary = None
    if overlay_summary_path:
        overlay_path = Path(overlay_summary_path)
        if overlay_path.exists():
            overlay_summary = safe_read_json(overlay_summary_path)

    pred["date"] = pd.to_datetime(pred["date"])
    feat["date"] = pd.to_datetime(feat["date"])

    # normalize prob column
    if "pred_prob" not in pred.columns:
        cand = [c for c in pred.columns if "prob" in c.lower()]
        if len(cand) == 0:
            raise ValueError("Predictions file must contain 'pred_prob' or equivalent probability column.")
        pred = pred.rename(columns={cand[0]: "pred_prob"})

    # normalize future_return if already in predictions
    if "future_return" not in pred.columns:
        ret_cols = [c for c in feat.columns if c.startswith("future_return_")]
        if len(ret_cols) != 1:
            raise ValueError("Feature file must contain exactly one future_return_* column.")
        ret_col = ret_cols[0]
        pred = pred.merge(
            feat[["date", "stock", "split", ret_col]].rename(columns={ret_col: "future_return"}),
            on=["date", "stock", "split"],
            how="left",
        )

    if pred["future_return"].isna().any():
        raise ValueError("Predictions data contains missing future_return after merge.")

    return pred, feat, model_summary, overlay_summary


# -----------------------------
# Strategy extraction / backtest
# -----------------------------
def extract_base_strategy(model_summary: Dict) -> Dict:
    # expected from transformer v5 script
    block = model_summary.get("best_strategy_checkpoint", {})
    selected = block.get("selected_strategy", {})
    return {
        "mode": selected.get("mode", "topk"),
        "top_k": int(selected.get("top_k", 2)),
        "min_prob": float(selected.get("min_prob", 0.60)),
        "threshold": float(selected.get("threshold", 0.50)),
        "constraints_satisfied": bool(selected.get("constraints_satisfied", True)),
        "val_strategy": selected.get("val_strategy", {}),
        "test_strategy": selected.get("test_strategy", {}),
    }


def compute_base_weights(day_df: pd.DataFrame, mode: str, top_k: int, min_prob: float, threshold: float) -> Dict[str, float]:
    day = day_df.sort_values("pred_prob", ascending=False).copy()
    if mode == "topk":
        chosen = day[day["pred_prob"] >= min_prob].head(top_k)
    elif mode == "threshold":
        chosen = day[day["pred_prob"] >= threshold].copy()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if len(chosen) == 0:
        return {}
    w = 1.0 / len(chosen)
    return {row["stock"]: w for _, row in chosen.iterrows()}


def infer_horizon_from_feature_df(feature_df: pd.DataFrame) -> int:
    return_cols = [c for c in feature_df.columns if c.startswith("future_return_")]
    if len(return_cols) != 1:
        return 5
    col = return_cols[0]
    if col.endswith("1d"):
        return 1
    if col.endswith("5d"):
        return 5
    return 5


def backtest_final_base(
    pred_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    mode: str,
    top_k: int,
    min_prob: float,
    threshold: float,
    transaction_cost_bps: float = 10.0,
) -> Tuple[Dict, pd.DataFrame]:
    horizon = infer_horizon_from_feature_df(feature_df)
    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    rebalance_dates = sorted(df["date"].drop_duplicates().tolist())[::horizon]
    tc = transaction_cost_bps / 10000.0

    prev_w = {}
    equity = 1.0
    equity_curve = [equity]
    period_returns = []
    turns = []
    rows = []

    for dt in rebalance_dates:
        day = df[df["date"] == dt].copy()
        day = day.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        new_w = compute_base_weights(day, mode=mode, top_k=top_k, min_prob=min_prob, threshold=threshold)

        ret_map = day.set_index("stock")["future_return"].to_dict()
        gross = float(sum(new_w[s] * ret_map.get(s, 0.0) for s in new_w.keys()))
        turn = turnover(prev_w, new_w)
        net = gross - tc * turn

        equity *= (1.0 + net)
        equity_curve.append(equity)
        period_returns.append(net)
        turns.append(turn)

        top1_stock = day.iloc[0]["stock"] if len(day) > 0 else ""
        top1_prob = float(day.iloc[0]["pred_prob"]) if len(day) > 0 else np.nan
        rows.append({
            "date": pd.Timestamp(dt),
            "selected_stocks": "|".join(sorted(new_w.keys())),
            "n_holdings": len(new_w),
            "top1_stock": top1_stock,
            "top1_prob": top1_prob,
            "gross_return": gross,
            "net_return": net,
            "turnover": turn,
            "equity": equity,
            "drawdown": equity / max(equity_curve) - 1.0,
            "exposure": float(sum(new_w.values())),
        })
        prev_w = new_w

    metrics = {
        "periods": int(len(period_returns)),
        "cumulative_return": float(equity_curve[-1] - 1.0),
        "annualized_return": annualized_return(equity_curve[-1], len(period_returns) * horizon),
        "sharpe": sharpe_ratio(period_returns, horizon=horizon),
        "max_drawdown": max_drawdown_from_equity_curve(equity_curve),
        "win_rate": float(np.mean(np.array(period_returns) > 0)) if period_returns else 0.0,
        "avg_turnover": float(np.mean(turns)) if turns else 0.0,
        "avg_exposure": float(np.mean([r["exposure"] for r in rows])) if rows else 0.0,
        "horizon": horizon,
    }
    return metrics, pd.DataFrame(rows)


# -----------------------------
# Charts
# -----------------------------
def make_equity_chart(actions_df: pd.DataFrame, title: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = pd.to_datetime(actions_df["date"])
    y = actions_df["equity"].astype(float).values
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    return fig_to_base64(fig)


def make_drawdown_chart(actions_df: pd.DataFrame, title: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.0))
    x = pd.to_datetime(actions_df["date"])
    y = actions_df["drawdown"].astype(float).values
    ax.plot(x, y)
    ax.axhline(0.0, linestyle="--", alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    return fig_to_base64(fig)


def make_turnover_chart(actions_df: pd.DataFrame, title: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.0))
    x = pd.to_datetime(actions_df["date"])
    y = actions_df["turnover"].astype(float).values
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Turnover")
    ax.grid(True, alpha=0.3)
    return fig_to_base64(fig)


def make_pred_distribution_chart(test_pred_df: pd.DataFrame, title: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.0))
    vals = test_pred_df["pred_prob"].astype(float).values
    ax.hist(vals, bins=30)
    ax.set_title(title)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    return fig_to_base64(fig)


# -----------------------------
# HTML assembly
# -----------------------------
def latest_snapshot(pred_df: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    latest_date = pred_df["date"].max()
    latest_pred = pred_df[pred_df["date"] == latest_date].copy()
    latest_feat = feat_df[feat_df["date"] == latest_date].copy()

    cols = ["date", "stock"]
    for c in ["dc_trend", "mkt_dc_trend", "vix_z_60", "credit_stress"]:
        if c in latest_feat.columns:
            cols.append(c)

    latest = latest_pred.merge(latest_feat[cols], on=["date", "stock"], how="left")
    latest = latest.sort_values("pred_prob", ascending=False).reset_index(drop=True)
    return latest


def df_to_html_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    show = df.head(max_rows).copy()
    return show.to_html(index=False, classes="styled-table")


def render_dashboard(
    out_html: str,
    model_summary: Dict,
    overlay_summary: Optional[Dict],
    base_strategy: Dict,
    test_metrics: Dict,
    val_metrics_from_summary: Dict,
    test_metrics_from_summary: Dict,
    test_actions: pd.DataFrame,
    test_pred_df: pd.DataFrame,
    latest_df: pd.DataFrame,
    charts: Dict[str, str],
) -> None:
    overlay_note = "Overlay summary not found."
    overlay_decision = "N/A"

    if overlay_summary is not None:
        selected_overlay = overlay_summary.get("selected_overlay_params", {})
        overlay_decision = selected_overlay.get("candidate_type", "unknown")
        reason = selected_overlay.get("selection_reason", "")
        overlay_note = f"Overlay selector result: <strong>{overlay_decision}</strong>"
        if reason:
            overlay_note += f" ({reason})"

    latest_top = latest_df[["stock", "pred_prob"]].head(5).copy()
    latest_top["pred_prob"] = latest_top["pred_prob"].map(lambda x: f"{x:.4f}")

    market_state_rows = []
    if len(latest_df) > 0:
        row0 = latest_df.iloc[0]
        for c, label in [
            ("dc_trend", "Top-stock DC trend"),
            ("mkt_dc_trend", "Market DC trend"),
            ("vix_z_60", "VIX z-score (60)"),
            ("credit_stress", "Credit stress"),
        ]:
            if c in latest_df.columns:
                val = row0[c]
                market_state_rows.append({"Metric": label, "Value": "N/A" if pd.isna(val) else f"{val:.4f}"})

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dissertation Trading Dashboard - Final Transformer System</title>
<style>
body {{
  font-family: Arial, sans-serif;
  margin: 24px;
  background: #f7f7f7;
  color: #111;
}}
h1, h2, h3 {{
  color: #111;
}}
.section {{
  background: #fff;
  padding: 18px 22px;
  margin-bottom: 18px;
  border-radius: 12px;
  box-shadow: 0 1px 5px rgba(0,0,0,0.08);
}}
.cards {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 14px;
}}
.card {{
  background: #fafafa;
  border: 1px solid #e6e6e6;
  border-radius: 12px;
  padding: 14px;
}}
.metric-label {{
  font-size: 13px;
  color: #666;
  margin-bottom: 6px;
}}
.metric-value {{
  font-size: 26px;
  font-weight: bold;
}}
.small-note {{
  color: #444;
  font-size: 14px;
}}
img.chart {{
  max-width: 100%;
  border-radius: 8px;
  border: 1px solid #ddd;
}}
.styled-table {{
  border-collapse: collapse;
  width: 100%;
  margin-top: 12px;
}}
.styled-table th, .styled-table td {{
  border: 1px solid #ddd;
  padding: 8px 10px;
  text-align: left;
}}
.styled-table th {{
  background: #f0f0f0;
}}
.code-like {{
  background: #f4f4f4;
  padding: 8px 10px;
  border-radius: 8px;
  font-family: Consolas, monospace;
  white-space: pre-wrap;
}}
</style>
</head>
<body>

<div class="section">
  <h1>Final Dissertation Trading Dashboard</h1>
  <p class="small-note">
    Final practical system: <strong>H5 Transformer v5 base strategy</strong>.  
    This dashboard reflects the final model selection after the overlay experiments.
  </p>
</div>

<div class="section">
  <h2>Executive Summary</h2>
  <div class="cards">
    <div class="card">
      <div class="metric-label">Final system</div>
      <div class="metric-value">Transformer v5</div>
      <div class="small-note">Base strategy without additional overlay</div>
    </div>
    <div class="card">
      <div class="metric-label">Horizon</div>
      <div class="metric-value">H{test_metrics.get('horizon', 5)}</div>
      <div class="small-note">5-day prediction horizon</div>
    </div>
    <div class="card">
      <div class="metric-label">Test cumulative return</div>
      <div class="metric-value">{fmt_pct(test_metrics['cumulative_return'])}</div>
      <div class="small-note">Net of transaction-cost assumption in backtest</div>
    </div>
    <div class="card">
      <div class="metric-label">Test Sharpe</div>
      <div class="metric-value">{fmt_num(test_metrics['sharpe'])}</div>
      <div class="small-note">Risk-adjusted performance</div>
    </div>
    <div class="card">
      <div class="metric-label">Test max drawdown</div>
      <div class="metric-value">{fmt_pct(test_metrics['max_drawdown'])}</div>
      <div class="small-note">Peak-to-trough decline</div>
    </div>
    <div class="card">
      <div class="metric-label">Validation constraint status</div>
      <div class="metric-value">{"Pass" if base_strategy['constraints_satisfied'] else "Fail"}</div>
      <div class="small-note">Formal model-selection rule</div>
    </div>
  </div>
</div>

<div class="section">
  <h2>Final Strategy Configuration</h2>
  <div class="code-like">mode={base_strategy['mode']}, top_k={base_strategy['top_k']}, min_prob={base_strategy['min_prob']:.2f}, threshold={base_strategy['threshold']:.2f}</div>
  <p class="small-note">
    {overlay_note}.  
    Final practical deployment decision: <strong>{"No additional overlay is enabled" if overlay_decision == "no_overlay" else "Overlay decision should be reviewed from selector output"}</strong>.
  </p>
</div>

<div class="section">
  <h2>Validation vs Test Metrics (Selected Transformer v5 Strategy)</h2>
  <div class="cards">
    <div class="card">
      <div class="metric-label">Validation cumulative return</div>
      <div class="metric-value">{fmt_pct(val_metrics_from_summary.get('cumulative_return'))}</div>
    </div>
    <div class="card">
      <div class="metric-label">Validation Sharpe</div>
      <div class="metric-value">{fmt_num(val_metrics_from_summary.get('sharpe'))}</div>
    </div>
    <div class="card">
      <div class="metric-label">Validation max drawdown</div>
      <div class="metric-value">{fmt_pct(val_metrics_from_summary.get('max_drawdown'))}</div>
    </div>
    <div class="card">
      <div class="metric-label">Test cumulative return</div>
      <div class="metric-value">{fmt_pct(test_metrics_from_summary.get('cumulative_return'))}</div>
    </div>
    <div class="card">
      <div class="metric-label">Test Sharpe</div>
      <div class="metric-value">{fmt_num(test_metrics_from_summary.get('sharpe'))}</div>
    </div>
    <div class="card">
      <div class="metric-label">Test max drawdown</div>
      <div class="metric-value">{fmt_pct(test_metrics_from_summary.get('max_drawdown'))}</div>
    </div>
  </div>
</div>

<div class="section">
  <h2>Test Equity Curve</h2>
  <img class="chart" src="data:image/png;base64,{charts['equity']}">
</div>

<div class="section">
  <h2>Test Drawdown Curve</h2>
  <img class="chart" src="data:image/png;base64,{charts['drawdown']}">
</div>

<div class="section">
  <h2>Test Turnover</h2>
  <img class="chart" src="data:image/png;base64,{charts['turnover']}">
</div>

<div class="section">
  <h2>Prediction Distribution (Test Split)</h2>
  <img class="chart" src="data:image/png;base64,{charts['pred_dist']}">
</div>

<div class="section">
  <h2>Latest Top Ranked Stocks</h2>
  {df_to_html_table(latest_top, max_rows=5)}
</div>

<div class="section">
  <h2>Latest Market / DC Snapshot</h2>
  {df_to_html_table(pd.DataFrame(market_state_rows), max_rows=10) if market_state_rows else "<p class='small-note'>No additional market-state columns available.</p>"}
</div>

<div class="section">
  <h2>Recent Rebalance Actions</h2>
  {df_to_html_table(test_actions[['date', 'selected_stocks', 'n_holdings', 'top1_stock', 'top1_prob', 'net_return', 'turnover', 'equity', 'drawdown']].tail(12).sort_values('date', ascending=False), max_rows=12)}
</div>

<div class="section">
  <h2>Interpretation</h2>
  <p class="small-note">
    This dashboard presents the final dissertation system after the model and overlay experiments were consolidated.
    The final practical choice is the <strong>base Transformer v5 strategy</strong>.
    The overlay-selection experiments did not produce a configuration that improved the base strategy sufficiently under the imposed risk-control framework,
    so the final deployable configuration does not activate an additional overlay layer.
  </p>
</div>

</body>
</html>
"""
    Path(out_html).write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate final dissertation dashboard HTML for Transformer v5 base system.")
    parser.add_argument("--fixed_predictions_path", type=str, default=DEFAULT_FIXED_PREDICTIONS)
    parser.add_argument("--feature_data_path", type=str, default=DEFAULT_FEATURE_DATA)
    parser.add_argument("--model_summary_path", type=str, default=DEFAULT_MODEL_SUMMARY)
    parser.add_argument("--overlay_summary_path", type=str, default=DEFAULT_OVERLAY_SUMMARY)
    parser.add_argument("--out_html", type=str, default=DEFAULT_OUT_HTML)
    parser.add_argument("--transaction_cost_bps", type=float, default=10.0)
    args = parser.parse_args()

    pred, feat, model_summary, overlay_summary = load_inputs(
        fixed_predictions_path=args.fixed_predictions_path,
        feature_data_path=args.feature_data_path,
        model_summary_path=args.model_summary_path,
        overlay_summary_path=args.overlay_summary_path,
    )

    base_strategy = extract_base_strategy(model_summary)
    test_pred_df = pred[pred["split"] == "test"].copy()
    test_metrics, test_actions = backtest_final_base(
        pred_df=test_pred_df,
        feature_df=feat,
        mode=base_strategy["mode"],
        top_k=base_strategy["top_k"],
        min_prob=base_strategy["min_prob"],
        threshold=base_strategy["threshold"],
        transaction_cost_bps=args.transaction_cost_bps,
    )

    charts = {
        "equity": make_equity_chart(test_actions, "Final Transformer v5 - Test Equity Curve"),
        "drawdown": make_drawdown_chart(test_actions, "Final Transformer v5 - Test Drawdown"),
        "turnover": make_turnover_chart(test_actions, "Final Transformer v5 - Test Turnover"),
        "pred_dist": make_pred_distribution_chart(test_pred_df, "Test Prediction Distribution"),
    }

    latest_df = latest_snapshot(pred, feat)

    val_metrics_from_summary = base_strategy.get("val_strategy", {})
    test_metrics_from_summary = base_strategy.get("test_strategy", {})

    render_dashboard(
        out_html=args.out_html,
        model_summary=model_summary,
        overlay_summary=overlay_summary,
        base_strategy=base_strategy,
        test_metrics=test_metrics,
        val_metrics_from_summary=val_metrics_from_summary,
        test_metrics_from_summary=test_metrics_from_summary,
        test_actions=test_actions,
        test_pred_df=test_pred_df,
        latest_df=latest_df,
        charts=charts,
    )

    print(f"Dashboard saved to: {args.out_html}")


if __name__ == "__main__":
    main()
