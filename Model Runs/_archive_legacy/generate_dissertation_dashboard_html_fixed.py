
import json
import math
import base64
import argparse
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Dissertation HTML Dashboard Generator
#
# No Streamlit required.
# Generates a self-contained local HTML report that can be opened
# directly in a browser.
#
# Recommended inputs:
#   --base_predictions_path
#       fixed_predictions/lstm_predictions_all_splits_fixed.csv
#   --feature_data_path
#       cleaned_datasets/main_experiment_h5.csv
#   --overlay_dir
#       lstm_predictions_all_splits_fixed_offline_policy_overlay_v2b
#
# Output:
#   dissertation_trading_dashboard.html
# =========================================================


DEFAULT_BASE_PREDICTIONS = r"D:\python\dissertation\Model Runs\fixed_predictions\lstm_predictions_all_splits_fixed.csv"
DEFAULT_FEATURE_DATA = r"D:\python\dissertation\Data Acquisition\cleaned_datasets\main_experiment_h5.csv"
DEFAULT_OVERLAY_DIR = r"D:\python\dissertation\Model Runs\lstm_predictions_all_splits_fixed_offline_policy_overlay_v2b"


def read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_inputs(base_predictions_path: Path, feature_data_path: Path, overlay_dir: Path):
    base_pred = safe_read_csv(base_predictions_path)
    feat_df = safe_read_csv(feature_data_path)
    metrics_summary = read_json(overlay_dir / "metrics_summary.json")
    test_actions = safe_read_csv(overlay_dir / "test_overlay_actions.csv")
    val_actions = safe_read_csv(overlay_dir / "val_overlay_actions.csv")
    return base_pred, feat_df, metrics_summary, test_actions, val_actions


def select_rebalance_dates(pred_df: pd.DataFrame, split_name: str, horizon: int = 5) -> List[pd.Timestamp]:
    split_df = pred_df[pred_df["split"] == split_name].copy()
    unique_dates = sorted(pd.to_datetime(split_df["date"]).drop_duplicates().tolist())
    return unique_dates[::horizon]


def build_ranked_rebalance_frame(pred_df: pd.DataFrame, split_name: str, horizon: int = 5) -> pd.DataFrame:
    split_df = pred_df[pred_df["split"] == split_name].copy()
    rebalance_dates = select_rebalance_dates(pred_df, split_name=split_name, horizon=horizon)

    rows = []
    for dt in rebalance_dates:
        day = split_df[split_df["date"] == dt].copy()
        if day.empty:
            continue
        day = day.sort_values("pred_prob", ascending=False).reset_index(drop=True)

        row = {
            "date": pd.Timestamp(dt),
            "top1_stock": day.loc[0, "stock"],
            "top1_prob": float(day.loc[0, "pred_prob"]),
            "top1_future_return": float(day.loc[0, "future_return"]),
            "top2_stock": day.loc[1, "stock"] if len(day) > 1 else day.loc[0, "stock"],
            "top2_prob": float(day.loc[1, "pred_prob"]) if len(day) > 1 else float(day.loc[0, "pred_prob"]),
            "top2_future_return": float(day.loc[1, "future_return"]) if len(day) > 1 else float(day.loc[0, "future_return"]),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def merge_overlay_with_ranked(test_actions: pd.DataFrame, ranked_df: pd.DataFrame) -> pd.DataFrame:
    out = test_actions.copy()
    out["date"] = pd.to_datetime(out["date"])
    ranked_df = ranked_df.copy()
    ranked_df["date"] = pd.to_datetime(ranked_df["date"])
    out = out.merge(ranked_df, on="date", how="left")
    return out


def add_context_features(signal_df: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [
        "date", "stock",
        "dc_event", "dc_trend", "dc_tmv",
        "mkt_dc_event", "mkt_dc_trend",
        "vix_z_60", "credit_stress",
        "return_21d_cs_z", "rsi_14_cs_z", "macd_hist_pct_cs_z",
        "close"
    ]
    use_cols = [c for c in feat_cols if c in feat_df.columns]
    tmp = feat_df[use_cols].copy()

    out = signal_df.copy()
    out = out.merge(
        tmp.rename(columns={"stock": "top1_stock"}),
        on=["date", "top1_stock"],
        how="left",
    )
    return out


def action_to_display_stocks(row: pd.Series) -> str:
    action = str(row.get("action_name", ""))
    if action == "cash":
        return "-"
    if action == "top1_full":
        return str(row.get("top1_stock", "-"))
    if action == "top2_equal":
        return f"{row.get('top1_stock', '-')}, {row.get('top2_stock', '-')}"
    return "-"


def action_to_display_prob(row: pd.Series) -> str:
    action = str(row.get("action_name", ""))
    if action == "cash":
        return "-"
    if action == "top1_full":
        return f"{row.get('top1_prob', np.nan):.3f}"
    if action == "top2_equal":
        return f"{row.get('top1_prob', np.nan):.3f}, {row.get('top2_prob', np.nan):.3f}"
    return "-"


def build_decision_reason(row: pd.Series, base_min_prob: float) -> str:
    action = str(row.get("action_name", ""))
    top1 = row.get("top1_stock", "-")
    top1_prob = row.get("top1_prob", np.nan)
    top2 = row.get("top2_stock", "-")
    top2_prob = row.get("top2_prob", np.nan)

    dc_event = row.get("dc_event", np.nan)
    dc_trend = row.get("dc_trend", np.nan)
    mkt_dc_trend = row.get("mkt_dc_trend", np.nan)
    vix_z = row.get("vix_z_60", np.nan)
    credit_stress = row.get("credit_stress", np.nan)

    parts = []
    if pd.notna(top1_prob):
        parts.append(f"Top signal was {top1} with probability {top1_prob:.3f}.")
    if pd.notna(top2_prob):
        parts.append(f"Second-ranked signal was {top2} with probability {top2_prob:.3f}.")
    parts.append(f"Base threshold was {base_min_prob:.2f}.")
    if pd.notna(dc_event) and pd.notna(dc_trend):
        parts.append(f"Top stock DC state: dc_event={dc_event:.0f}, dc_trend={dc_trend:.0f}.")
    if pd.notna(mkt_dc_trend):
        parts.append(f"Market DC trend={mkt_dc_trend:.0f}.")
    if pd.notna(vix_z):
        parts.append(f"VIX z-score={vix_z:.2f}.")
    if pd.notna(credit_stress):
        parts.append(f"Credit stress={credit_stress:.4f}.")
    if action == "cash":
        parts.append("Overlay chose cash, preferring no exposure on this rebalance date.")
    elif action == "top1_full":
        parts.append(f"Overlay chose full exposure to the top-ranked stock: {top1}.")
    elif action == "top2_equal":
        parts.append(f"Overlay diversified across the top two ranked stocks: {top1} and {top2}.")
    else:
        parts.append("Overlay action unavailable.")
    return " ".join(parts)


def metrics_table(metrics_summary: Dict) -> pd.DataFrame:
    base = metrics_summary["test_base_metrics"]
    overlay = metrics_summary["test_overlay_metrics"]

    rows = [
        ["Cumulative Return", base["cumulative_return"], overlay["cumulative_return"], overlay["cumulative_return"] - base["cumulative_return"]],
        ["Annualized Return", base["annualized_return"], overlay["annualized_return"], overlay["annualized_return"] - base["annualized_return"]],
        ["Sharpe", base["sharpe"], overlay["sharpe"], overlay["sharpe"] - base["sharpe"]],
        ["Max Drawdown", base["max_drawdown"], overlay["max_drawdown"], overlay["max_drawdown"] - base["max_drawdown"]],
        ["Win Rate", base["win_rate"], overlay["win_rate"], overlay["win_rate"] - base["win_rate"]],
        ["Avg Turnover", base.get("avg_turnover", np.nan), overlay.get("avg_turnover", np.nan), overlay.get("avg_turnover", np.nan) - base.get("avg_turnover", np.nan)],
    ]
    return pd.DataFrame(rows, columns=["Metric", "Base H5 LSTM", "Overlay v2b", "Overlay - Base"])


def format_percent(x) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:.2%}"


def format_num(x) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:.4f}"


def fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_equity_chart(test_actions: pd.DataFrame) -> str:
    df = test_actions.copy().sort_values("date")
    if df.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pd.to_datetime(df["date"]), df["overlay_equity"], label="Overlay")
    ax.plot(pd.to_datetime(df["date"]), df["base_equity"], label="Base H5 LSTM")
    ax.set_title("Test Period Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig_to_base64(fig)


def make_stock_chart(selected_stock: str, feat_df: pd.DataFrame, merged_test: pd.DataFrame) -> str:
    price_df = feat_df[feat_df["stock"] == selected_stock].copy().sort_values("date")
    if "split" in price_df.columns:
        price_df = price_df[price_df["split"] == "test"].copy()

    if price_df.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(price_df["date"], price_df["close"], label=f"{selected_stock} Close")

    if "dc_event" in price_df.columns:
        event_df = price_df[price_df["dc_event"] != 0]
        if not event_df.empty:
            ax.scatter(event_df["date"], event_df["close"], s=18, label="DC Event")

    sig = merged_test[
        (merged_test["top1_stock"] == selected_stock) | (merged_test["top2_stock"] == selected_stock)
    ].copy()

    if not sig.empty:
        marker_map = {
            "cash": "x",
            "top1_full": "^",
            "top2_equal": "s",
        }

        # Avoid close/close_x/close_y merge collisions by renaming first
        price_marker_df = price_df[["date", "close"]].rename(columns={"close": "price_close"})

        for action_name, g in sig.groupby("action_name"):
            merged = g.merge(price_marker_df, on="date", how="left")
            if "price_close" not in merged.columns:
                continue

            ax.scatter(
                merged["date"],
                merged["price_close"],
                s=35,
                marker=marker_map.get(action_name, "o"),
                label=f"Overlay {action_name}"
            )

    ax.set_title(f"{selected_stock}: Price, DC Events and Overlay Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig_to_base64(fig)


def dataframe_to_html(df: pd.DataFrame, percent_cols: List[str] = None, number_cols: List[str] = None, max_rows: int = None) -> str:
    tmp = df.copy()
    percent_cols = percent_cols or []
    number_cols = number_cols or []

    if max_rows is not None:
        tmp = tmp.head(max_rows)

    for c in percent_cols:
        if c in tmp.columns:
            tmp[c] = tmp[c].apply(format_percent)
    for c in number_cols:
        if c in tmp.columns:
            tmp[c] = tmp[c].apply(format_num)

    return tmp.to_html(index=False, escape=False, classes="dataframe")


def build_html(
    metrics_summary: Dict,
    overview_df: pd.DataFrame,
    merged_test: pd.DataFrame,
    equity_chart_b64: str,
    stock_charts: Dict[str, str],
    feature_data_path: Path,
    base_predictions_path: Path,
    overlay_dir: Path,
) -> str:
    base_test = metrics_summary["test_base_metrics"]
    overlay_test = metrics_summary["test_overlay_metrics"]
    excess_test = metrics_summary["test_excess_cumulative_return"]
    base_min_prob = float(metrics_summary.get("base_strategy_definition", {}).get("base_min_prob", 0.62))

    overview_html = dataframe_to_html(
        overview_df,
        number_cols=["Base H5 LSTM", "Overlay v2b", "Overlay - Base"]
    )

    trade_log_cols = [
        "date", "action_name", "selected_stocks", "selected_probs",
        "overlay_net_return", "base_net_return", "overlay_equity",
        "base_equity", "overlay_drawdown", "overlay_turnover", "base_turnover"
    ]
    trade_log_df = merged_test[trade_log_cols].copy()
    trade_log_html = dataframe_to_html(
        trade_log_df,
        percent_cols=["overlay_net_return", "base_net_return", "overlay_drawdown"],
        number_cols=["overlay_equity", "base_equity", "overlay_turnover", "base_turnover"],
        max_rows=120
    )

    explanation_rows = []
    for _, row in merged_test.head(20).iterrows():
        explanation_rows.append({
            "date": str(pd.Timestamp(row["date"]).date()),
            "action_name": row["action_name"],
            "selected_stocks": row["selected_stocks"],
            "selected_probs": row["selected_probs"],
            "overlay_net_return": row["overlay_net_return"],
            "base_net_return": row["base_net_return"],
            "decision_reason": row["decision_reason"],
        })
    explanation_df = pd.DataFrame(explanation_rows)
    explanation_html = dataframe_to_html(
        explanation_df,
        percent_cols=["overlay_net_return", "base_net_return"],
    )

    stock_sections = []
    for stock, img_b64 in stock_charts.items():
        if not img_b64:
            continue
        stock_sections.append(f"""
        <div class="section">
          <h3>{stock}</h3>
          <img class="chart" src="data:image/png;base64,{img_b64}" />
        </div>
        """)

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Dissertation Trading Dashboard</title>
<style>
body {{
  font-family: Arial, Helvetica, sans-serif;
  margin: 0;
  padding: 0;
  background: #f7f8fb;
  color: #1f2937;
}}
.container {{
  width: 1200px;
  margin: 0 auto;
  padding: 24px;
}}
h1, h2, h3 {{
  margin-top: 0;
}}
.card-grid {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 24px;
}}
.card {{
  background: white;
  border-radius: 14px;
  padding: 18px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}}
.metric-label {{
  font-size: 13px;
  color: #6b7280;
}}
.metric-value {{
  font-size: 28px;
  font-weight: 700;
  margin-top: 6px;
}}
.section {{
  background: white;
  border-radius: 14px;
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}}
.chart {{
  width: 100%;
  max-width: 1100px;
}}
table.dataframe {{
  border-collapse: collapse;
  width: 100%;
  font-size: 13px;
}}
table.dataframe th, table.dataframe td {{
  border: 1px solid #e5e7eb;
  padding: 8px;
  text-align: left;
  vertical-align: top;
}}
table.dataframe th {{
  background: #f3f4f6;
}}
.meta {{
  font-size: 13px;
  color: #4b5563;
  line-height: 1.6;
}}
.small {{
  font-size: 12px;
  color: #6b7280;
}}
</style>
</head>
<body>
<div class="container">
  <h1>Dissertation Trading Dashboard</h1>
  <p class="meta">
    Base system: <strong>H5 LSTM</strong><br/>
    Policy enhancement: <strong>Offline Policy Overlay v2b</strong><br/>
    Base threshold: <strong>{base_min_prob:.2f}</strong>
  </p>

  <div class="card-grid">
    <div class="card">
      <div class="metric-label">Base Test Return</div>
      <div class="metric-value">{format_percent(base_test["cumulative_return"])}</div>
    </div>
    <div class="card">
      <div class="metric-label">Overlay Test Return</div>
      <div class="metric-value">{format_percent(overlay_test["cumulative_return"])}</div>
    </div>
    <div class="card">
      <div class="metric-label">Excess Return</div>
      <div class="metric-value">{format_percent(excess_test)}</div>
    </div>
    <div class="card">
      <div class="metric-label">Overlay Test Sharpe</div>
      <div class="metric-value">{format_num(overlay_test["sharpe"])}</div>
    </div>
  </div>

  <div class="section">
    <h2>Portfolio Overview</h2>
    {overview_html}
  </div>

  <div class="section">
    <h2>Equity Curve (Test Period)</h2>
    <img class="chart" src="data:image/png;base64,{equity_chart_b64}" />
  </div>

  <div class="section">
    <h2>Replay Trade Log</h2>
    {trade_log_html}
  </div>

  <div class="section">
    <h2>Decision Explanation Samples</h2>
    {explanation_html}
  </div>

  <div class="section">
    <h2>Stock-Level Signal Views</h2>
    <p class="small">Charts show test-period close price, DC events, and dates where the overlay selected the stock.</p>
  </div>

  {''.join(stock_sections)}

  <div class="section">
    <h2>Data Sources</h2>
    <p class="meta">
      Base predictions: {base_predictions_path}<br/>
      Feature data: {feature_data_path}<br/>
      Overlay directory: {overlay_dir}
    </p>
    <p class="small">
      This HTML report is a no-Streamlit fallback dashboard for dissertation demonstration and paper-trading style replay.
    </p>
  </div>
</div>
</body>
</html>
"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate a self-contained HTML dashboard for the dissertation trading system.")
    parser.add_argument("--base_predictions_path", type=str, default=DEFAULT_BASE_PREDICTIONS)
    parser.add_argument("--feature_data_path", type=str, default=DEFAULT_FEATURE_DATA)
    parser.add_argument("--overlay_dir", type=str, default=DEFAULT_OVERLAY_DIR)
    parser.add_argument("--out_html", type=str, default=r"D:\python\dissertation\Model Runs\dissertation_trading_dashboard.html")
    args = parser.parse_args()

    base_predictions_path = Path(args.base_predictions_path)
    feature_data_path = Path(args.feature_data_path)
    overlay_dir = Path(args.overlay_dir)
    out_html = Path(args.out_html)

    base_pred, feat_df, metrics_summary, test_actions, val_actions = load_inputs(
        base_predictions_path=base_predictions_path,
        feature_data_path=feature_data_path,
        overlay_dir=overlay_dir,
    )

    horizon = int(metrics_summary.get("horizon", 5))
    base_min_prob = float(metrics_summary.get("base_strategy_definition", {}).get("base_min_prob", 0.62))

    ranked_test = build_ranked_rebalance_frame(base_pred, split_name="test", horizon=horizon)
    merged_test = merge_overlay_with_ranked(test_actions, ranked_test)
    merged_test = add_context_features(merged_test, feat_df)
    merged_test["selected_stocks"] = merged_test.apply(action_to_display_stocks, axis=1)
    merged_test["selected_probs"] = merged_test.apply(action_to_display_prob, axis=1)
    merged_test["decision_reason"] = merged_test.apply(lambda row: build_decision_reason(row, base_min_prob), axis=1)

    overview_df = metrics_table(metrics_summary)
    equity_chart_b64 = make_equity_chart(test_actions)

    stock_list = sorted(feat_df["stock"].dropna().unique().tolist())
    stock_charts = {}
    for stock in stock_list:
        stock_charts[stock] = make_stock_chart(stock, feat_df, merged_test)

    html = build_html(
        metrics_summary=metrics_summary,
        overview_df=overview_df,
        merged_test=merged_test,
        equity_chart_b64=equity_chart_b64,
        stock_charts=stock_charts,
        feature_data_path=feature_data_path,
        base_predictions_path=base_predictions_path,
        overlay_dir=overlay_dir,
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")

    print(f"Saved HTML dashboard to: {out_html}")


if __name__ == "__main__":
    main()
