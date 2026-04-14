
import argparse
import base64
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_FINAL_DIR = r"D:\python\dissertation\Model Runs\final_run_20260413\final_transformer_optimized_system"
DEFAULT_FEATURE_DATA = r"D:\python\dissertation\Data Acquisition\cleaned_datasets\main_experiment_h5.csv"
DEFAULT_FIXED_PREDICTIONS = r"D:\python\dissertation\Model Runs\final_run_20260413\fixed_predictions_transformer_v5\transformer_predictions_all_splits_fixed.csv"
DEFAULT_OUT_HTML = r"D:\python\dissertation\Model Runs\final_run_20260413\dissertation_trading_dashboard_transformer_optimized_final_v2.html"


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


def fmt_pct(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "N/A"
    try:
        if not np.isfinite(float(x)):
            return "N/A"
    except Exception:
        return "N/A"
    return f"{float(x) * 100:.{digits}f}%"


def fmt_num(x: Optional[float], digits: int = 3) -> str:
    if x is None:
        return "N/A"
    try:
        if not np.isfinite(float(x)):
            return "N/A"
    except Exception:
        return "N/A"
    return f"{float(x):.{digits}f}"


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def load_inputs(final_dir: str, feature_data_path: str, fixed_predictions_path: str):
    final_dir = Path(final_dir)

    manifest = safe_read_json(str(final_dir / "final_system_manifest.json"))
    metrics = safe_read_json(str(final_dir / "final_metrics_summary.json"))
    val_actions = safe_read_csv(str(final_dir / "final_val_actions.csv"))
    test_actions = safe_read_csv(str(final_dir / "final_test_actions.csv"))

    feat = safe_read_csv(feature_data_path)
    pred = safe_read_csv(fixed_predictions_path)

    feat["date"] = pd.to_datetime(feat["date"])
    pred["date"] = pd.to_datetime(pred["date"])
    val_actions["date"] = pd.to_datetime(val_actions["date"])
    test_actions["date"] = pd.to_datetime(test_actions["date"])

    if "pred_prob" not in pred.columns:
        cand = [c for c in pred.columns if "prob" in c.lower()]
        if not cand:
            raise ValueError("Predictions file must contain pred_prob or equivalent.")
        pred = pred.rename(columns={cand[0]: "pred_prob"})

    return manifest, metrics, val_actions, test_actions, pred, feat


def make_line_chart(actions_df: pd.DataFrame, y_col: str, title: str, y_label: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = pd.to_datetime(actions_df["date"])
    y = actions_df[y_col].astype(float).values
    ax.plot(x, y)
    if y_col == "drawdown":
        ax.axhline(0.0, linestyle="--", alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    return fig_to_base64(fig)


def make_hist_chart(pred_df: pd.DataFrame, title: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.0))
    vals = pred_df["pred_prob"].astype(float).values
    ax.hist(vals, bins=30)
    ax.set_title(title)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    return fig_to_base64(fig)


def df_to_html_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    show = df.head(max_rows).copy()
    return show.to_html(index=False, classes="styled-table")


def parse_selected_stocks(cell) -> List[str]:
    if pd.isna(cell):
        return []
    text = str(cell).strip()
    if text == "":
        return []
    return [x for x in text.split("|") if x]


def build_latest_actual_portfolio(test_actions: pd.DataFrame, pred: pd.DataFrame, feat: pd.DataFrame):
    if len(test_actions) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.NaT

    last_action = test_actions.sort_values("date").iloc[-1].copy()
    rebalance_date = pd.Timestamp(last_action["date"])
    selected_names = parse_selected_stocks(last_action.get("selected_stocks", ""))
    exposure = float(last_action.get("exposure", 0.0))
    n_holdings = int(last_action.get("n_holdings", 0))

    actual_weights = {}
    if n_holdings > 0 and exposure > 0:
        w = exposure / n_holdings
        actual_weights = {name: w for name in selected_names}

    pred_day = pred[pred["date"] == rebalance_date].copy().sort_values("pred_prob", ascending=False).reset_index(drop=True)
    feat_day = feat[feat["date"] == rebalance_date].copy()

    rank_view = pred_day.copy()
    rank_view["held_in_actual_portfolio"] = rank_view["stock"].map(lambda s: 1 if s in actual_weights else 0)
    rank_view["actual_weight"] = rank_view["stock"].map(lambda s: float(actual_weights.get(s, 0.0)))
    rank_view = rank_view[["date", "stock", "pred_prob", "held_in_actual_portfolio", "actual_weight"]].copy()

    market_cols = ["date", "stock"]
    for c in ["dc_trend", "mkt_dc_trend", "vix_z_60", "credit_stress"]:
        if c in feat_day.columns:
            market_cols.append(c)

    if len(rank_view) > 0 and len(feat_day) > 0:
        top_stock = rank_view.iloc[0]["stock"]
        market_snapshot = feat_day[feat_day["stock"] == top_stock][market_cols].copy()
    else:
        market_snapshot = pd.DataFrame(columns=market_cols)

    return rank_view, market_snapshot, rebalance_date


def render_dashboard(out_html: str, manifest: Dict, metrics: Dict, val_actions: pd.DataFrame, test_actions: pd.DataFrame, pred: pd.DataFrame, feat: pd.DataFrame) -> None:
    final_cfg = manifest.get("execution_layer", {})
    val_metrics = metrics.get("validation_metrics", {})
    test_metrics = metrics.get("test_metrics", {})
    all_metrics = metrics.get("all_period_metrics", {})

    test_pred = pred[pred["split"] == "test"].copy()

    charts = {
        "equity": make_line_chart(test_actions, "equity", "Final Optimized System - Test Equity Curve", "Equity"),
        "drawdown": make_line_chart(test_actions, "drawdown", "Final Optimized System - Test Drawdown", "Drawdown"),
        "turnover": make_line_chart(test_actions, "turnover", "Final Optimized System - Test Turnover", "Turnover"),
        "pred_dist": make_hist_chart(test_pred, "Test Prediction Distribution"),
    }

    latest_portfolio_view, market_snapshot, rebalance_date = build_latest_actual_portfolio(test_actions, pred, feat)

    latest_top = latest_portfolio_view[["stock", "pred_prob", "held_in_actual_portfolio", "actual_weight"]].head(8).copy()
    if "pred_prob" in latest_top.columns:
        latest_top["pred_prob"] = latest_top["pred_prob"].map(lambda x: f"{float(x):.4f}")
    if "actual_weight" in latest_top.columns:
        latest_top["actual_weight"] = latest_top["actual_weight"].map(lambda x: f"{float(x):.4f}")

    recent_actions = test_actions[[c for c in ["date", "selected_stocks", "n_holdings", "net_return", "turnover", "equity", "drawdown", "exposure", "newly_added", "kept_existing"] if c in test_actions.columns]].tail(12).sort_values("date", ascending=False).copy()

    actual_portfolio_note = (
        f"Table below shows the actual ranked signal list and actual holdings at the last test rebalance date: <strong>{rebalance_date.date()}</strong>."
        if pd.notna(rebalance_date) else
        "No final rebalance snapshot available."
    )

    html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dissertation Trading Dashboard - Final Optimized Transformer System</title>
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
  line-height: 1.55;
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
    Final practical system: <strong>H5 Transformer v5 fixed predictions + validation-only optimized sticky execution</strong>.
    This is the final frozen deployable version used for the dissertation.
  </p>
</div>

<div class="section">
  <h2>Executive Summary</h2>
  <div class="cards">
    <div class="card">
      <div class="metric-label">Final system</div>
      <div class="metric-value">Transformer v5 + Sticky Execution</div>
      <div class="small-note">Frozen optimized execution layer</div>
    </div>
    <div class="card">
      <div class="metric-label">Validation cumulative return</div>
      <div class="metric-value">{fmt_pct(val_metrics.get("cumulative_return"))}</div>
      <div class="small-note">Final validation economic performance</div>
    </div>
    <div class="card">
      <div class="metric-label">Validation Sharpe</div>
      <div class="metric-value">{fmt_num(val_metrics.get("sharpe"))}</div>
      <div class="small-note">Risk-adjusted validation performance</div>
    </div>
    <div class="card">
      <div class="metric-label">Validation max drawdown</div>
      <div class="metric-value">{fmt_pct(val_metrics.get("max_drawdown"))}</div>
      <div class="small-note">Validation downside control</div>
    </div>
    <div class="card">
      <div class="metric-label">Test cumulative return</div>
      <div class="metric-value">{fmt_pct(test_metrics.get("cumulative_return"))}</div>
      <div class="small-note">Final hold-out test performance</div>
    </div>
    <div class="card">
      <div class="metric-label">Test Sharpe</div>
      <div class="metric-value">{fmt_num(test_metrics.get("sharpe"))}</div>
      <div class="small-note">Final hold-out Sharpe</div>
    </div>
    <div class="card">
      <div class="metric-label">Test max drawdown</div>
      <div class="metric-value">{fmt_pct(test_metrics.get("max_drawdown"))}</div>
      <div class="small-note">Final hold-out drawdown</div>
    </div>
    <div class="card">
      <div class="metric-label">Test turnover</div>
      <div class="metric-value">{fmt_num(test_metrics.get("avg_turnover"))}</div>
      <div class="small-note">Execution discipline after optimization</div>
    </div>
  </div>
</div>

<div class="section">
  <h2>Final Frozen Execution Configuration</h2>
  <div class="code-like">{json.dumps(final_cfg, ensure_ascii=False, indent=2)}</div>
  <p class="small-note">
    This final execution layer keeps only the strongest signals, introduces holding persistence,
    and limits unnecessary switching. It is optimized using validation only and then applied once to test.
  </p>
</div>

<div class="section">
  <h2>All-Period Reference Metrics</h2>
  <div class="cards">
    <div class="card">
      <div class="metric-label">All-period annualized return</div>
      <div class="metric-value">{fmt_pct(all_metrics.get("annualized_return"))}</div>
    </div>
    <div class="card">
      <div class="metric-label">All-period Sharpe</div>
      <div class="metric-value">{fmt_num(all_metrics.get("sharpe"))}</div>
    </div>
    <div class="card">
      <div class="metric-label">All-period max drawdown</div>
      <div class="metric-value">{fmt_pct(all_metrics.get("max_drawdown"))}</div>
    </div>
    <div class="card">
      <div class="metric-label">All-period turnover</div>
      <div class="metric-value">{fmt_num(all_metrics.get("avg_turnover"))}</div>
    </div>
  </div>
  <p class="small-note">
    These all-period figures are provided as a descriptive reference only. The formal dissertation claims are based
    primarily on validation, test, walk-forward diagnostics, and transaction-cost-aware evaluation.
  </p>
</div>

<div class="section">
  <h2>Test Equity Curve</h2>
  <img class="chart" src="data:image/png;base64,{charts["equity"]}">
</div>

<div class="section">
  <h2>Test Drawdown Curve</h2>
  <img class="chart" src="data:image/png;base64,{charts["drawdown"]}">
</div>

<div class="section">
  <h2>Test Turnover</h2>
  <img class="chart" src="data:image/png;base64,{charts["turnover"]}">
</div>

<div class="section">
  <h2>Prediction Distribution (Test Split)</h2>
  <img class="chart" src="data:image/png;base64,{charts["pred_dist"]}">
</div>

<div class="section">
  <h2>Latest Actual Portfolio at Last Test Rebalance</h2>
  <p class="small-note">{actual_portfolio_note}</p>
  {df_to_html_table(latest_top, max_rows=8)}
</div>

<div class="section">
  <h2>Latest Market / DC Snapshot</h2>
  {df_to_html_table(market_snapshot, max_rows=5) if len(market_snapshot) > 0 else "<p class='small-note'>No additional market-state columns available.</p>"}
</div>

<div class="section">
  <h2>Recent Test Rebalance Actions</h2>
  {df_to_html_table(recent_actions, max_rows=12)}
</div>

<div class="section">
  <h2>Interpretation</h2>
  <p class="small-note">
    The final system combines Transformer-based probability forecasts with a more conservative execution layer.
    Compared with the earlier frozen configuration, this optimized execution reduces turnover, improves validation drawdown control,
    and improves the final hold-out test Sharpe and cumulative return. The result should still be interpreted as a regime-sensitive,
    weak-predictive but economically tradable system rather than a universally stable forecasting engine.
  </p>
</div>

</body>
</html>
'''
    Path(out_html).write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate final dashboard for optimized Transformer system (fixed latest portfolio display).")
    parser.add_argument("--final_dir", type=str, default=DEFAULT_FINAL_DIR)
    parser.add_argument("--feature_data_path", type=str, default=DEFAULT_FEATURE_DATA)
    parser.add_argument("--fixed_predictions_path", type=str, default=DEFAULT_FIXED_PREDICTIONS)
    parser.add_argument("--out_html", type=str, default=DEFAULT_OUT_HTML)
    args = parser.parse_args()

    manifest, metrics, val_actions, test_actions, pred, feat = load_inputs(
        final_dir=args.final_dir,
        feature_data_path=args.feature_data_path,
        fixed_predictions_path=args.fixed_predictions_path,
    )

    render_dashboard(
        out_html=args.out_html,
        manifest=manifest,
        metrics=metrics,
        val_actions=val_actions,
        test_actions=test_actions,
        pred=pred,
        feat=feat,
    )

    print(f"Dashboard saved to: {args.out_html}")


if __name__ == "__main__":
    main()
