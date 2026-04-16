
from __future__ import annotations

"""
Mainline defense dashboard builder

Executive:
- main transformer training result
- deployable frozen config
- execution optimization result
- benchmark-aware overlay result

Research:
- optional bootstrap CI
- optional block-length sensitivity
"""

import argparse
import json
import math
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


PALETTE = {
    "train": "#1f4e79",
    "deploy": "#2563eb",
    "exec": "#ff7f0e",
    "overlay": "#2ca02c",
    "benchmark": "#6b7280",
    "robust": "#9467bd",
    "bg": "#f6f7fb",
    "card": "#ffffff",
    "text": "#111827",
    "muted": "#6b7280",
    "border": "#e5e7eb",
    "warn_bg": "#fff7ed",
    "warn_border": "#fdba74",
}


@dataclass
class RunPaths:
    train_dir: Optional[Path]
    deployable_dir: Optional[Path]
    final_system_dir: Optional[Path]
    overlay_dir: Optional[Path]
    bootstrap_ci_dir: Optional[Path]
    block_sens_dir: Optional[Path]
    out_dir: Path


@dataclass
class Inputs:
    train_summary: Optional[dict]
    train_comparison: Optional[pd.DataFrame]

    deployable_manifest: Optional[dict]
    deployable_config_comparison: Optional[pd.DataFrame]
    deployable_candidate_summary: Optional[pd.DataFrame]
    deployable_overall_metric_summary: Optional[pd.DataFrame]

    final_system_manifest: Optional[dict]

    overlay_summary: Optional[dict]
    overlay_strategy_comparison: Optional[pd.DataFrame]

    bootstrap_summary: Optional[dict]
    bootstrap_static: Optional[pd.DataFrame]
    bootstrap_sticky: Optional[pd.DataFrame]
    bootstrap_diff: Optional[pd.DataFrame]

    block_diff: Optional[pd.DataFrame]
    block_strategy: Optional[pd.DataFrame]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_latest_run_dir(base_dir: Path, pattern: str = "*") -> Optional[Path]:
    if not base_dir.exists():
        return None
    candidates = [p for p in base_dir.glob(pattern) if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def safe_read_json(path: Path, warnings: List[str]) -> Optional[dict]:
    try:
        if not path.exists():
            warnings.append(f"缺失 JSON：{path}")
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        warnings.append(f"读取 JSON 失败：{path} | {e}")
        return None


def safe_read_csv(path: Path, warnings: List[str]) -> Optional[pd.DataFrame]:
    try:
        if not path.exists():
            warnings.append(f"缺失 CSV：{path}")
            return None
        return pd.read_csv(path)
    except Exception as e:
        warnings.append(f"读取 CSV 失败：{path} | {e}")
        return None


def first_existing_file(parent: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = parent / n
        if p.exists():
            return p
    return None


def fmt_pct(x: Any, digits: int = 2) -> str:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return "NA"
        return f"{v * 100:.{digits}f}%"
    except Exception:
        return "NA"


def fmt_num(x: Any, digits: int = 4) -> str:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return "NA"
        return f"{v:.{digits}f}"
    except Exception:
        return "NA"


def fmt_int(x: Any) -> str:
    try:
        return str(int(float(x)))
    except Exception:
        return "NA"


def normalize_key(key: str) -> str:
    return str(key).strip().lower().replace(" ", "_")


def find_first_key(obj: Any, targets: List[str]) -> Any:
    normalized_targets = {normalize_key(t) for t in targets}

    def _walk(x: Any) -> Any:
        if isinstance(x, dict):
            for k, v in x.items():
                if normalize_key(k) in normalized_targets:
                    return v
            for _, v in x.items():
                found = _walk(v)
                if found is not None:
                    return found
        elif isinstance(x, list):
            for item in x:
                found = _walk(item)
                if found is not None:
                    return found
        return None

    return _walk(obj)


def pick_row(df: Optional[pd.DataFrame], split: Optional[str] = None, strategy: Optional[str] = None) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    sub = df.copy()
    if split is not None and "split" in sub.columns:
        sub = sub[sub["split"].astype(str).str.lower() == split.lower()]
    if strategy is not None and "strategy" in sub.columns:
        sub = sub[sub["strategy"].astype(str).str.lower() == strategy.lower()]
    if sub.empty:
        return None
    return sub.iloc[0]


def card_html(title: str, value: str, subtitle: str = "", color_key: str = "train") -> str:
    border = PALETTE.get(color_key, PALETTE["train"])
    return f"""
    <div class="kpi-card" style="border-left: 6px solid {border};">
      <div class="kpi-title">{title}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-subtitle">{subtitle}</div>
    </div>
    """


def warning_box(msg: str) -> str:
    return f"""
    <div class="warning">
      <div class="warning-title">⚠ 提示</div>
      <div class="warning-msg">{msg}</div>
    </div>
    """


def df_to_table_html(df: Optional[pd.DataFrame], table_id: str) -> str:
    if df is None or df.empty:
        return warning_box("表格数据为空。")
    return df.to_html(index=False, escape=True, table_id=table_id, classes="tbl")


def dict_to_kv_df(d: Optional[Dict[str, Any]], order: Optional[List[str]] = None) -> pd.DataFrame:
    if not isinstance(d, dict) or not d:
        return pd.DataFrame(columns=["Item", "Value"])
    keys = list(d.keys())
    if order:
        ordered = [k for k in order if k in d] + [k for k in keys if k not in order]
    else:
        ordered = keys
    rows = []
    for k in ordered:
        v = d.get(k)
        if isinstance(v, float):
            if any(t in normalize_key(k) for t in ["return", "drawdown", "rate", "turnover", "gap", "share"]):
                vv = fmt_pct(v, 2)
            else:
                vv = fmt_num(v, 4)
        else:
            vv = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
        rows.append({"Item": k, "Value": vv})
    return pd.DataFrame(rows)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plot_to_div(fig: Optional[go.Figure], div_id: str, include_plotlyjs: bool, filename_stub: str) -> str:
    if fig is None:
        return warning_box(f"图表缺失：{div_id}")
    config = {
        "displaylogo": False,
        "responsive": True,
        "toImageButtonOptions": {"format": "png", "filename": filename_stub, "scale": 3},
    }
    return fig.to_html(full_html=False, include_plotlyjs=True if include_plotlyjs else False, config=config, div_id=div_id)


def discover_runs(project_root: Path) -> RunPaths:
    model_runs = project_root / "Model Runs"

    train_parent = model_runs / "experiments" / "main_transformer_h5"
    deploy_parent = model_runs / "evaluation" / "transformer_deployable"
    final_parent = model_runs / "execution" / "final_system"
    overlay_parent = model_runs / "execution" / "benchmark_aware_overlay"

    train_dir = (train_parent / "main_transformer") if (train_parent / "main_transformer").exists() else find_latest_run_dir(train_parent, "*")
    deploy_dir = (deploy_parent / "main_transformer_deployable") if (deploy_parent / "main_transformer_deployable").exists() else find_latest_run_dir(deploy_parent, "*")
    final_dir = (final_parent / "main_final_system") if (final_parent / "main_final_system").exists() else find_latest_run_dir(final_parent, "*")
    overlay_dir = (overlay_parent / "main_benchmark_aware_overlay") if (overlay_parent / "main_benchmark_aware_overlay").exists() else find_latest_run_dir(overlay_parent, "*")

    bootstrap_ci_dir = find_latest_run_dir(model_runs / "robustness" / "bootstrap_ci", "run_*")
    block_sens_dir = find_latest_run_dir(model_runs / "robustness" / "bootstrap_block_sensitivity", "run_*")

    out_dir = ensure_dir(model_runs / "reporting")
    return RunPaths(
        train_dir=train_dir,
        deployable_dir=deploy_dir,
        final_system_dir=final_dir,
        overlay_dir=overlay_dir,
        bootstrap_ci_dir=bootstrap_ci_dir,
        block_sens_dir=block_sens_dir,
        out_dir=out_dir,
    )


def load_inputs(runs: RunPaths, warnings: List[str]) -> Inputs:
    train_summary = None
    train_comparison = None
    if runs.train_dir:
        train_summary = safe_read_json(runs.train_dir / "metrics_summary.json", warnings)
        train_comparison = safe_read_csv(runs.train_dir / "model_comparison.csv", warnings)
    else:
        warnings.append("未找到主线训练输出目录（experiments/main_transformer_h5/main_transformer）。")

    deployable_manifest = None
    deployable_config_comparison = None
    deployable_candidate_summary = None
    deployable_overall_metric_summary = None
    if runs.deployable_dir:
        deployable_manifest = safe_read_json(runs.deployable_dir / "final_system_manifest.json", warnings)
        deployable_config_comparison = safe_read_csv(runs.deployable_dir / "deployable_config_comparison.csv", warnings)
        deployable_candidate_summary = safe_read_csv(runs.deployable_dir / "candidate_summary.csv", warnings)
        deployable_overall_metric_summary = safe_read_csv(runs.deployable_dir / "overall_metric_summary.csv", warnings)
    else:
        warnings.append("未找到 deployable 评估目录（evaluation/transformer_deployable/main_transformer_deployable）。")

    final_system_manifest = None
    if runs.final_system_dir:
        manifest_path = first_existing_file(runs.final_system_dir, ["final_system_manifest.json", "system_manifest.json"])
        if manifest_path:
            final_system_manifest = safe_read_json(manifest_path, warnings)
        else:
            warnings.append(f"未找到 execution 最终系统 manifest：{runs.final_system_dir}")
    else:
        warnings.append("未找到 final system 目录（execution/final_system/main_final_system）。")

    overlay_summary = None
    overlay_strategy_comparison = None
    if runs.overlay_dir:
        overlay_summary_path = first_existing_file(runs.overlay_dir, ["benchmark_aware_overlay_summary.json", "overlay_summary.json"])
        if overlay_summary_path:
            overlay_summary = safe_read_json(overlay_summary_path, warnings)
        else:
            warnings.append(f"未找到 overlay summary JSON：{runs.overlay_dir}")
        overlay_strategy_comparison = safe_read_csv(runs.overlay_dir / "strategy_comparison_val_test.csv", warnings)
    else:
        warnings.append("未找到 benchmark-aware overlay 目录（execution/benchmark_aware_overlay/main_benchmark_aware_overlay）。")

    bootstrap_summary = None
    bootstrap_static = None
    bootstrap_sticky = None
    bootstrap_diff = None
    if runs.bootstrap_ci_dir:
        bootstrap_summary = safe_read_json(runs.bootstrap_ci_dir / "bootstrap_summary.json", warnings)
        bootstrap_static = safe_read_csv(runs.bootstrap_ci_dir / "bootstrap_static_test.csv", warnings)
        bootstrap_sticky = safe_read_csv(runs.bootstrap_ci_dir / "bootstrap_sticky_test.csv", warnings)
        bootstrap_diff = safe_read_csv(runs.bootstrap_ci_dir / "bootstrap_difference_test.csv", warnings)

    block_diff = None
    block_strategy = None
    if runs.block_sens_dir:
        block_diff = safe_read_csv(runs.block_sens_dir / "block_length_difference_summary.csv", warnings)
        block_strategy = safe_read_csv(runs.block_sens_dir / "block_length_strategy_summary.csv", warnings)

    return Inputs(
        train_summary=train_summary,
        train_comparison=train_comparison,
        deployable_manifest=deployable_manifest,
        deployable_config_comparison=deployable_config_comparison,
        deployable_candidate_summary=deployable_candidate_summary,
        deployable_overall_metric_summary=deployable_overall_metric_summary,
        final_system_manifest=final_system_manifest,
        overlay_summary=overlay_summary,
        overlay_strategy_comparison=overlay_strategy_comparison,
        bootstrap_summary=bootstrap_summary,
        bootstrap_static=bootstrap_static,
        bootstrap_sticky=bootstrap_sticky,
        bootstrap_diff=bootstrap_diff,
        block_diff=block_diff,
        block_strategy=block_strategy,
    )


def make_mainline_metric_bar(data: Inputs) -> Optional[go.Figure]:
    series = []

    train_row = pick_row(data.train_comparison)
    if train_row is not None:
        series.append({
            "name": "Main Transformer",
            "cumret": train_row.get("test_strategy_cumulative_return", np.nan),
            "sharpe": train_row.get("test_strategy_sharpe", np.nan),
            "mdd": train_row.get("test_strategy_max_drawdown", np.nan),
            "turnover": train_row.get("test_strategy_avg_turnover", np.nan),
        })

    fsm = data.final_system_manifest or {}
    current_test = find_first_key(fsm, ["current_test_metrics"])
    optimized_test = find_first_key(fsm, ["optimized_test_metrics"])
    if isinstance(current_test, dict):
        series.append({
            "name": "Frozen Execution",
            "cumret": current_test.get("cumulative_return", np.nan),
            "sharpe": current_test.get("sharpe", np.nan),
            "mdd": current_test.get("max_drawdown", np.nan),
            "turnover": current_test.get("avg_turnover", np.nan),
        })
    if isinstance(optimized_test, dict):
        series.append({
            "name": "Optimized Execution",
            "cumret": optimized_test.get("cumulative_return", np.nan),
            "sharpe": optimized_test.get("sharpe", np.nan),
            "mdd": optimized_test.get("max_drawdown", np.nan),
            "turnover": optimized_test.get("avg_turnover", np.nan),
        })

    overlay_test = pick_row(data.overlay_strategy_comparison, split="test", strategy="benchmark_aware_active_overlay")
    benchmark_test = pick_row(data.overlay_strategy_comparison, split="test", strategy="universe_equal_weight_buy_hold")
    if benchmark_test is not None:
        series.append({
            "name": "Benchmark",
            "cumret": benchmark_test.get("cumulative_return", np.nan),
            "sharpe": benchmark_test.get("sharpe", np.nan),
            "mdd": benchmark_test.get("max_drawdown", np.nan),
            "turnover": benchmark_test.get("avg_turnover", np.nan),
        })
    if overlay_test is not None:
        series.append({
            "name": "Overlay Final",
            "cumret": overlay_test.get("cumulative_return", np.nan),
            "sharpe": overlay_test.get("sharpe", np.nan),
            "mdd": overlay_test.get("max_drawdown", np.nan),
            "turnover": overlay_test.get("avg_turnover", np.nan),
        })

    if not series:
        return None

    names = [s["name"] for s in series]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[s["cumret"] for s in series], name="Cumulative Return"))
    fig.add_trace(go.Bar(x=names, y=[s["sharpe"] for s in series], name="Sharpe"))
    fig.add_trace(go.Bar(x=names, y=[s["mdd"] for s in series], name="Max Drawdown"))
    fig.add_trace(go.Bar(x=names, y=[s["turnover"] for s in series], name="Avg Turnover"))
    fig.update_layout(
        template="plotly_white",
        barmode="group",
        height=460,
        margin=dict(l=40, r=20, t=40, b=50),
        title="Mainline test metrics across system layers",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    return fig


def make_overlay_val_test_bar(data: Inputs) -> Optional[go.Figure]:
    val = pick_row(data.overlay_strategy_comparison, split="val", strategy="benchmark_aware_active_overlay")
    test = pick_row(data.overlay_strategy_comparison, split="test", strategy="benchmark_aware_active_overlay")
    bench_val = pick_row(data.overlay_strategy_comparison, split="val", strategy="universe_equal_weight_buy_hold")
    bench_test = pick_row(data.overlay_strategy_comparison, split="test", strategy="universe_equal_weight_buy_hold")
    if val is None and test is None:
        return None

    categories = ["Validation", "Test"]
    overlay_cum = [
        val.get("cumulative_return", np.nan) if val is not None else np.nan,
        test.get("cumulative_return", np.nan) if test is not None else np.nan,
    ]
    bench_cum = [
        bench_val.get("cumulative_return", np.nan) if bench_val is not None else np.nan,
        bench_test.get("cumulative_return", np.nan) if bench_test is not None else np.nan,
    ]
    overlay_shp = [
        val.get("sharpe", np.nan) if val is not None else np.nan,
        test.get("sharpe", np.nan) if test is not None else np.nan,
    ]
    bench_shp = [
        bench_val.get("sharpe", np.nan) if bench_val is not None else np.nan,
        bench_test.get("sharpe", np.nan) if bench_test is not None else np.nan,
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=categories, y=bench_cum, name="Benchmark CumRet", marker_color=PALETTE["benchmark"]))
    fig.add_trace(go.Bar(x=categories, y=overlay_cum, name="Overlay CumRet", marker_color=PALETTE["overlay"]))
    fig.add_trace(go.Scatter(x=categories, y=bench_shp, name="Benchmark Sharpe", mode="lines+markers", yaxis="y2",
                             line=dict(color=_hex_to_rgba(PALETTE["benchmark"], 0.85), width=3)))
    fig.add_trace(go.Scatter(x=categories, y=overlay_shp, name="Overlay Sharpe", mode="lines+markers", yaxis="y2",
                             line=dict(color=_hex_to_rgba(PALETTE["overlay"], 0.85), width=3)))

    fig.update_layout(
        template="plotly_white",
        height=460,
        margin=dict(l=40, r=50, t=40, b=50),
        title="Benchmark-aware overlay vs benchmark",
        barmode="group",
        yaxis=dict(title="Cumulative Return"),
        yaxis2=dict(title="Sharpe", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    return fig


def make_bootstrap_violin(boot_static: pd.DataFrame, boot_sticky: pd.DataFrame, col: str, title: str) -> Optional[go.Figure]:
    if boot_static is None or boot_sticky is None:
        return None
    if col not in boot_static.columns or col not in boot_sticky.columns:
        return None

    fig = go.Figure()
    fig.add_trace(go.Violin(
        y=boot_static[col].astype(float),
        name="Static",
        line_color=PALETTE["train"],
        fillcolor=_hex_to_rgba(PALETTE["train"], 0.25),
        box_visible=True,
        meanline_visible=True,
        points=False
    ))
    fig.add_trace(go.Violin(
        y=boot_sticky[col].astype(float),
        name="Sticky",
        line_color=PALETTE["exec"],
        fillcolor=_hex_to_rgba(PALETTE["exec"], 0.25),
        box_visible=True,
        meanline_visible=True,
        points=False
    ))
    fig.update_layout(
        template="plotly_white",
        title=title,
        margin=dict(l=40, r=20, t=60, b=40),
        height=420,
        yaxis_title=col
    )
    return fig


def make_block_sensitivity_figure(block_diff: pd.DataFrame, y_mean: str, y_lo: str, y_hi: str, title: str) -> Optional[go.Figure]:
    if block_diff is None or block_diff.empty:
        return None
    needed = {"block_length", y_mean, y_lo, y_hi}
    if not needed.issubset(set(block_diff.columns)):
        return None

    df = block_diff.sort_values("block_length").copy()
    x = df["block_length"].astype(int).values
    mean = df[y_mean].astype(float).values
    lo = df[y_lo].astype(float).values
    hi = df[y_hi].astype(float).values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=hi, mode="lines",
        line=dict(color=_hex_to_rgba(PALETTE["robust"], 0.0), width=0),
        showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=x, y=lo, mode="lines", fill="tonexty",
        fillcolor=_hex_to_rgba(PALETTE["robust"], 0.18),
        line=dict(color=_hex_to_rgba(PALETTE["robust"], 0.0), width=0),
        showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=x, y=mean, mode="lines+markers",
        line=dict(color=PALETTE["robust"], width=3),
        marker=dict(size=7),
        name="mean"
    ))
    fig.update_layout(
        template="plotly_white",
        title=title,
        margin=dict(l=40, r=20, t=60, b=40),
        height=420,
        xaxis_title="block length",
        yaxis_title="difference"
    )
    return fig


def build_executive_view(data: Inputs) -> str:
    train_row = pick_row(data.train_comparison)
    train_cfg = None
    if train_row is not None:
        train_cfg = {
            "mode": train_row.get("selected_strategy_mode"),
            "top_k": train_row.get("selected_top_k"),
            "min_prob": train_row.get("selected_min_prob"),
            "threshold": train_row.get("selected_threshold"),
            "constraints_satisfied": train_row.get("selected_constraints_satisfied"),
        }

    deploy_cfg = None
    if data.deployable_manifest:
        deploy_cfg = find_first_key(data.deployable_manifest, ["frozen_config", "recommended_frozen_config", "final_frozen_config", "current_frozen_config"])
        if not isinstance(deploy_cfg, dict):
            deploy_cfg = None

    current_cfg = None
    optimized_exec_cfg = None
    current_val = None
    optimized_val = None
    current_test = None
    optimized_test = None
    if data.final_system_manifest:
        current_cfg = find_first_key(data.final_system_manifest, ["current_frozen_config", "frozen_prediction_layer", "frozen_config"])
        optimized_exec_cfg = find_first_key(data.final_system_manifest, ["optimized_execution_config", "execution_layer"])
        current_val = find_first_key(data.final_system_manifest, ["current_validation_metrics"])
        optimized_val = find_first_key(data.final_system_manifest, ["optimized_validation_metrics"])
        current_test = find_first_key(data.final_system_manifest, ["current_test_metrics"])
        optimized_test = find_first_key(data.final_system_manifest, ["optimized_test_metrics"])

    overlay_cfg = None
    val_overlay_vs_bench = None
    test_overlay_vs_bench = None
    if data.overlay_summary:
        overlay_cfg = find_first_key(data.overlay_summary, ["selected_overlay_config", "overlay_config"])
        val_overlay_vs_bench = find_first_key(data.overlay_summary, ["validation_overlay_vs_benchmark", "val_relative"])
        test_overlay_vs_bench = find_first_key(data.overlay_summary, ["test_overlay_vs_benchmark", "test_relative"])

    overlay_test = pick_row(data.overlay_strategy_comparison, split="test", strategy="benchmark_aware_active_overlay")
    benchmark_test = pick_row(data.overlay_strategy_comparison, split="test", strategy="universe_equal_weight_buy_hold")

    cards = []
    if train_row is not None:
        cards.append(card_html("Mainline training Sharpe", fmt_num(train_row.get("test_strategy_sharpe"), 4), "Prediction layer (test)", "train"))
        cards.append(card_html("Mainline cumulative return", fmt_pct(train_row.get("test_strategy_cumulative_return"), 2), "Prediction layer (test)", "train"))
        cards.append(card_html("Mainline max drawdown", fmt_pct(train_row.get("test_strategy_max_drawdown"), 2), "Prediction layer (test)", "train"))
    if isinstance(optimized_test, dict):
        cards.append(card_html("Optimized execution Sharpe", fmt_num(optimized_test.get("sharpe"), 4), "Execution layer (test)", "exec"))
        cards.append(card_html("Optimized execution return", fmt_pct(optimized_test.get("cumulative_return"), 2), "Execution layer (test)", "exec"))
    if overlay_test is not None:
        cards.append(card_html("Overlay final Sharpe", fmt_num(overlay_test.get("sharpe"), 4), "Portfolio layer (test)", "overlay"))
        cards.append(card_html("Overlay final return", fmt_pct(overlay_test.get("cumulative_return"), 2), "Portfolio layer (test)", "overlay"))
        cards.append(card_html("Overlay final max drawdown", fmt_pct(overlay_test.get("max_drawdown"), 2), "Portfolio layer (test)", "overlay"))
    if benchmark_test is not None and overlay_test is not None:
        gap = float(overlay_test.get("cumulative_return", np.nan)) - float(benchmark_test.get("cumulative_return", np.nan))
        cards.append(card_html("Overlay excess return vs benchmark", fmt_pct(gap, 2), "Portfolio enhancement (test)", "overlay"))

    kpi_html = f"""
    <div class="grid-3">
      {''.join(cards)}
    </div>
    """

    layer_rows = []
    if train_row is not None:
        layer_rows.append({
            "Layer": "Prediction",
            "System": "Main Transformer",
            "Config": f"topk={fmt_int(train_row.get('selected_top_k'))}, min_prob={fmt_num(train_row.get('selected_min_prob'), 2)}, thr={fmt_num(train_row.get('selected_threshold'), 2)}",
            "Test CumRet": fmt_pct(train_row.get("test_strategy_cumulative_return"), 2),
            "Test Sharpe": fmt_num(train_row.get("test_strategy_sharpe"), 4),
            "Test MDD": fmt_pct(train_row.get("test_strategy_max_drawdown"), 2),
            "Turnover": fmt_pct(train_row.get("test_strategy_avg_turnover"), 2),
        })
    if isinstance(current_test, dict):
        layer_rows.append({
            "Layer": "Execution",
            "System": "Frozen Execution",
            "Config": json.dumps(current_cfg, ensure_ascii=False) if isinstance(current_cfg, dict) else "NA",
            "Test CumRet": fmt_pct(current_test.get("cumulative_return"), 2),
            "Test Sharpe": fmt_num(current_test.get("sharpe"), 4),
            "Test MDD": fmt_pct(current_test.get("max_drawdown"), 2),
            "Turnover": fmt_pct(current_test.get("avg_turnover"), 2),
        })
    if isinstance(optimized_test, dict):
        layer_rows.append({
            "Layer": "Execution",
            "System": "Optimized Sticky Execution",
            "Config": json.dumps(optimized_exec_cfg, ensure_ascii=False) if isinstance(optimized_exec_cfg, dict) else "NA",
            "Test CumRet": fmt_pct(optimized_test.get("cumulative_return"), 2),
            "Test Sharpe": fmt_num(optimized_test.get("sharpe"), 4),
            "Test MDD": fmt_pct(optimized_test.get("max_drawdown"), 2),
            "Turnover": fmt_pct(optimized_test.get("avg_turnover"), 2),
        })
    if benchmark_test is not None:
        layer_rows.append({
            "Layer": "Portfolio",
            "System": "Benchmark",
            "Config": "Universe equal-weight buy-and-hold",
            "Test CumRet": fmt_pct(benchmark_test.get("cumulative_return"), 2),
            "Test Sharpe": fmt_num(benchmark_test.get("sharpe"), 4),
            "Test MDD": fmt_pct(benchmark_test.get("max_drawdown"), 2),
            "Turnover": fmt_pct(benchmark_test.get("avg_turnover"), 2),
        })
    if overlay_test is not None:
        layer_rows.append({
            "Layer": "Portfolio",
            "System": "Benchmark-aware Active Overlay",
            "Config": json.dumps(overlay_cfg, ensure_ascii=False) if isinstance(overlay_cfg, dict) else "NA",
            "Test CumRet": fmt_pct(overlay_test.get("cumulative_return"), 2),
            "Test Sharpe": fmt_num(overlay_test.get("sharpe"), 4),
            "Test MDD": fmt_pct(overlay_test.get("max_drawdown"), 2),
            "Turnover": fmt_pct(overlay_test.get("avg_turnover"), 2),
        })

    layer_df = pd.DataFrame(layer_rows) if layer_rows else pd.DataFrame()

    config_cards = []
    if train_cfg:
        config_cards.append(f"<div class='card'><h2>Training-selected config</h2>{df_to_table_html(dict_to_kv_df(train_cfg, ['mode','top_k','min_prob','threshold','constraints_satisfied']), 'tbl_train_cfg')}</div>")
    if deploy_cfg:
        config_cards.append(f"<div class='card'><h2>Deployable frozen config</h2>{df_to_table_html(dict_to_kv_df(deploy_cfg, ['mode','top_k','min_prob','threshold']), 'tbl_deploy_cfg')}</div>")
    if optimized_exec_cfg:
        config_cards.append(f"<div class='card'><h2>Optimized execution config</h2>{df_to_table_html(dict_to_kv_df(optimized_exec_cfg, ['mode','top_k','entry_prob','exit_prob','threshold','switch_buffer','min_hold_periods','max_new_names_per_rebalance']), 'tbl_exec_cfg')}</div>")
    if overlay_cfg:
        config_cards.append(f"<div class='card'><h2>Selected overlay config</h2>{df_to_table_html(dict_to_kv_df(overlay_cfg, ['mode','tilt_budget','n_overweight','n_underweight','min_weight','max_weight','turnover_cap']), 'tbl_overlay_cfg')}</div>")

    exec_cards = []
    if isinstance(current_val, dict):
        exec_cards.append(f"<div class='card'><h2>Current validation execution metrics</h2>{df_to_table_html(dict_to_kv_df(current_val), 'tbl_cur_val')}</div>")
    if isinstance(optimized_val, dict):
        exec_cards.append(f"<div class='card'><h2>Optimized validation execution metrics</h2>{df_to_table_html(dict_to_kv_df(optimized_val), 'tbl_opt_val')}</div>")
    if isinstance(current_test, dict):
        exec_cards.append(f"<div class='card'><h2>Current test execution metrics</h2>{df_to_table_html(dict_to_kv_df(current_test), 'tbl_cur_test')}</div>")
    if isinstance(optimized_test, dict):
        exec_cards.append(f"<div class='card'><h2>Optimized test execution metrics</h2>{df_to_table_html(dict_to_kv_df(optimized_test), 'tbl_opt_test')}</div>")

    rel_cards = []
    if isinstance(val_overlay_vs_bench, dict):
        rel_cards.append(f"<div class='card'><h2>Validation overlay vs benchmark</h2>{df_to_table_html(dict_to_kv_df(val_overlay_vs_bench), 'tbl_val_rel')}</div>")
    if isinstance(test_overlay_vs_bench, dict):
        rel_cards.append(f"<div class='card'><h2>Test overlay vs benchmark</h2>{df_to_table_html(dict_to_kv_df(test_overlay_vs_bench), 'tbl_test_rel')}</div>")

    fig_layers = make_mainline_metric_bar(data)
    fig_overlay = make_overlay_val_test_bar(data)

    layers_div = plot_to_div(fig_layers, "fig_mainline_layers", include_plotlyjs=True, filename_stub="mainline_layers")
    overlay_div = plot_to_div(fig_overlay, "fig_overlay", include_plotlyjs=False, filename_stub="overlay_vs_benchmark")

    return f"""
    {kpi_html}
    <div style="height:14px;"></div>

    <div class="card">
      <h2>Mainline system summary</h2>
      {df_to_table_html(layer_df, "tbl_mainline_summary")}
      <div class="small">主线按 prediction → execution → portfolio overlay 三层结构展示。</div>
    </div>

    <div style="height:14px;"></div>
    <div class="grid-2">
      <div class="card"><h2>Mainline layer comparison</h2>{layers_div}</div>
      <div class="card"><h2>Overlay enhancement view</h2>{overlay_div}</div>
    </div>

    <div style="height:14px;"></div>
    <div class="grid-2">
      {''.join(config_cards) if config_cards else warning_box("未能读取主线配置表。")}
    </div>

    <div style="height:14px;"></div>
    <div class="grid-2">
      {''.join(exec_cards) if exec_cards else warning_box("未能读取 execution 优化指标表。")}
    </div>

    <div style="height:14px;"></div>
    <div class="grid-2">
      {''.join(rel_cards) if rel_cards else warning_box("未能读取 overlay 相对 benchmark 的附加指标。")}
    </div>
    """


def build_research_view(data: Inputs) -> str:
    if data.bootstrap_summary is None and data.block_diff is None:
        return warning_box("未找到 robustness 输入。Research 页将显示降级提示。")

    fig_bs_cum = make_bootstrap_violin(data.bootstrap_static, data.bootstrap_sticky, "cumulative_return", "Bootstrap distribution: cumulative return")
    fig_bs_shp = make_bootstrap_violin(data.bootstrap_static, data.bootstrap_sticky, "sharpe", "Bootstrap distribution: sharpe")
    fig_block_c = make_block_sensitivity_figure(data.block_diff, "cumret_diff_mean", "cumret_diff_ci_5", "cumret_diff_ci_95", "Block-length sensitivity: CumRet diff (Sticky - Static)")
    fig_block_s = make_block_sensitivity_figure(data.block_diff, "sharpe_diff_mean", "sharpe_diff_ci_5", "sharpe_diff_ci_95", "Block-length sensitivity: Sharpe diff (Sticky - Static)")

    bs_cum_div = plot_to_div(fig_bs_cum, "fig_bs_cum", include_plotlyjs=False, filename_stub="bootstrap_cumret")
    bs_shp_div = plot_to_div(fig_bs_shp, "fig_bs_shp", include_plotlyjs=False, filename_stub="bootstrap_sharpe")
    block_c_div = plot_to_div(fig_block_c, "fig_block_c", include_plotlyjs=False, filename_stub="block_cumret")
    block_s_div = plot_to_div(fig_block_s, "fig_block_s", include_plotlyjs=False, filename_stub="block_sharpe")

    paired = (data.bootstrap_summary or {}).get("paired_difference_summary", {}) if isinstance(data.bootstrap_summary, dict) else {}
    paired_rows = []
    if paired:
        paired_rows = [
            ["CumRet (Sticky-Static)", paired.get("cumret_diff_mean"), paired.get("cumret_diff_ci_5"), paired.get("cumret_diff_ci_95"), paired.get("prob_static_beats_sticky_cumret")],
            ["Sharpe (Sticky-Static)", paired.get("sharpe_diff_mean"), paired.get("sharpe_diff_ci_5"), paired.get("sharpe_diff_ci_95"), paired.get("prob_static_beats_sticky_sharpe")],
            ["MDD (Sticky-Static)", paired.get("mdd_diff_mean"), paired.get("mdd_diff_ci_5"), paired.get("mdd_diff_ci_95"), paired.get("prob_static_beats_sticky_mdd")],
        ]
    paired_df = pd.DataFrame(paired_rows, columns=["Metric", "Mean", "CI_5", "CI_95", "P(Static wins)"]) if paired_rows else pd.DataFrame()

    block_table = pd.DataFrame()
    if data.block_diff is not None and not data.block_diff.empty:
        cols = [c for c in [
            "block_length",
            "cumret_diff_mean", "cumret_diff_ci_5", "cumret_diff_ci_95", "prob_static_beats_sticky_cumret",
            "sharpe_diff_mean", "sharpe_diff_ci_5", "sharpe_diff_ci_95", "prob_static_beats_sticky_sharpe",
        ] if c in data.block_diff.columns]
        if cols:
            block_table = data.block_diff[cols].sort_values("block_length").copy()

    return f"""
    <div class="grid-2">
      <div class="card"><h2>Bootstrap cumulative return</h2>{bs_cum_div}</div>
      <div class="card"><h2>Bootstrap sharpe</h2>{bs_shp_div}</div>
    </div>

    <div style="height:14px;"></div>
    <div class="grid-2">
      <div class="card"><h2>Block sensitivity: CumRet</h2>{block_c_div}</div>
      <div class="card"><h2>Block sensitivity: Sharpe</h2>{block_s_div}</div>
    </div>

    <div style="height:14px;"></div>
    <div class="grid-2">
      <div class="card"><h2>Paired difference summary</h2>{df_to_table_html(paired_df, 'tbl_paired')}</div>
      <div class="card"><h2>Block-length difference table</h2>{df_to_table_html(block_table, 'tbl_block')}</div>
    </div>
    """


def build_css() -> str:
    return f"""
    <style>
      :root {{
        --bg: {PALETTE["bg"]};
        --card: {PALETTE["card"]};
        --text: {PALETTE["text"]};
        --muted: {PALETTE["muted"]};
        --border: {PALETTE["border"]};
        --warn-bg: {PALETTE["warn_bg"]};
        --warn-border: {PALETTE["warn_border"]};
      }}
      body {{
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC",
                     "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif;
        background: var(--bg);
        color: var(--text);
      }}
      .container {{
        max-width: 1320px;
        margin: 0 auto;
        padding: 22px 18px 40px 18px;
      }}
      .header {{
        display: flex;
        justify-content: space-between;
        gap: 14px;
        align-items: center;
        margin-bottom: 14px;
      }}
      .title h1 {{
        margin: 0;
        font-size: 34px;
      }}
      .title .meta {{
        margin-top: 6px;
        color: var(--muted);
        font-size: 13px;
      }}
      .toolbar {{
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }}
      .btn {{
        cursor: pointer;
        border: 1px solid var(--border);
        background: #fff;
        padding: 9px 12px;
        border-radius: 10px;
        font-size: 13px;
        color: var(--text);
      }}
      .btn:hover {{
        border-color: #cbd5e1;
        background: #fafafa;
      }}
      .btn-primary {{
        border-color: #111827;
        background: #111827;
        color: #fff;
      }}
      .tabs {{
        display: flex;
        gap: 8px;
        margin: 12px 0 16px 0;
      }}
      .tab {{
        border: 1px solid var(--border);
        background: #fff;
        border-radius: 12px;
        padding: 10px 14px;
        cursor: pointer;
        font-size: 14px;
      }}
      .tab[aria-selected="true"] {{
        background: #111827;
        color: #fff;
        border-color: #111827;
      }}
      .tab-content {{
        display: none;
      }}
      .tab-content.active {{
        display: block;
      }}
      .grid-2 {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
      }}
      .grid-3 {{
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 14px;
      }}
      .card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px 14px 10px 14px;
        box-shadow: 0 1px 0 rgba(17, 24, 39, 0.02);
      }}
      .card h2 {{
        margin: 0 0 10px 0;
        font-size: 18px;
      }}
      .kpi-card {{
        background: #fff;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px 14px 12px 14px;
      }}
      .kpi-title {{
        font-size: 13px;
        color: var(--muted);
        margin-bottom: 6px;
      }}
      .kpi-value {{
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 2px;
      }}
      .kpi-subtitle {{
        font-size: 12px;
        color: var(--muted);
      }}
      .tbl {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }}
      .tbl th, .tbl td {{
        border: 1px solid var(--border);
        padding: 8px 10px;
        text-align: left;
        vertical-align: top;
      }}
      .tbl th {{
        background: #f8fafc;
        font-weight: 600;
      }}
      .warning {{
        background: var(--warn-bg);
        border: 1px solid var(--warn-border);
        border-radius: 14px;
        padding: 12px 12px;
      }}
      .warning-title {{
        font-weight: 700;
        margin-bottom: 6px;
      }}
      .warning-msg {{
        font-size: 13px;
        color: #7c2d12;
        white-space: pre-wrap;
      }}
      .small {{
        color: var(--muted);
        font-size: 12px;
      }}
      @media print {{
        body {{
          background: #fff;
        }}
        .no-print {{
          display: none !important;
        }}
        .tab-content {{
          display: none !important;
        }}
        .tab-content.print-target {{
          display: block !important;
        }}
        .card, .kpi-card {{
          box-shadow: none !important;
          break-inside: avoid;
          page-break-inside: avoid;
        }}
      }}
    </style>
    """


def build_js() -> str:
    return """
    <script>
      function setActiveTab(tabId) {
        const tabs = document.querySelectorAll('.tab');
        const contents = document.querySelectorAll('.tab-content');

        tabs.forEach(t => {
          const isActive = (t.getAttribute('data-tab') === tabId);
          t.setAttribute('aria-selected', isActive ? 'true' : 'false');
        });

        contents.forEach(c => {
          const isActive = (c.id === tabId);
          c.classList.toggle('active', isActive);
          c.classList.toggle('print-target', isActive);
        });

        setTimeout(() => {
          try {
            if (window.Plotly) {
              const active = document.getElementById(tabId);
              const graphs = active.querySelectorAll('.plotly-graph-div');
              graphs.forEach(gd => {
                try { Plotly.Plots.resize(gd); } catch (e) {}
              });
            }
          } catch (e) {}
        }, 250);
      }

      document.addEventListener('keydown', (e) => {
        if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
        const tabs = Array.from(document.querySelectorAll('.tab'));
        const idx = tabs.findIndex(t => t.getAttribute('aria-selected') === 'true');
        if (idx < 0) return;
        const next = (e.key === 'ArrowRight') ? (idx + 1) % tabs.length : (idx - 1 + tabs.length) % tabs.length;
        setActiveTab(tabs[next].getAttribute('data-tab'));
      });

      function printToPDF() {
        window.print();
      }
    </script>
    """


def build_page_html(title: str, meta_lines: List[str], executive_html: str, research_html: str, combined: bool) -> str:
    css = build_css()
    js = build_js()
    meta_html = "<br/>".join(meta_lines)

    if combined:
        tabs_html = """
        <div class="tabs no-print" role="tablist" aria-label="Dashboard Tabs">
          <button class="tab" role="tab" aria-selected="true" data-tab="tab-exec" onclick="setActiveTab('tab-exec')">Executive</button>
          <button class="tab" role="tab" aria-selected="false" data-tab="tab-research" onclick="setActiveTab('tab-research')">Research</button>
        </div>
        """
        content_html = f"""
        <div id="tab-exec" class="tab-content active print-target">{executive_html}</div>
        <div id="tab-research" class="tab-content">{research_html}</div>
        """
    else:
        tabs_html = ""
        content_html = f"<div id='single' class='tab-content active print-target'>{executive_html}{research_html}</div>"

    return f"""
    <!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>{title}</title>
      {css}
    </head>
    <body>
      <div class="container">
        <div class="header no-print">
          <div class="title">
            <h1>{title}</h1>
            <div class="meta">{meta_html}</div>
          </div>
          <div class="toolbar">
            <button class="btn btn-primary" onclick="printToPDF()">打印/保存为 PDF（当前视图）</button>
            <button class="btn" onclick="setActiveTab('tab-exec')">切到 Executive</button>
            <button class="btn" onclick="setActiveTab('tab-research')">切到 Research</button>
          </div>
        </div>
        {tabs_html}
        {content_html}
        <div class="small" style="margin-top:14px;">
          说明：Executive 页为当前主线答辩版；Research 页为可选稳健性/附录页面。
        </div>
      </div>
      {js}
    </body>
    </html>
    """


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mainline defense dashboard.")
    parser.add_argument("--project_root", type=str, default=r"D:\python\dissertation")
    parser.add_argument("--horizon", type=int, default=5, help="kept for compatibility")
    parser.add_argument("--write_split_pages", action="store_true", help="also write dashboard_main.html and dashboard_robustness.html")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    warnings: List[str] = []

    runs = discover_runs(project_root)
    data = load_inputs(runs, warnings)

    executive_html = build_executive_view(data)
    research_html = build_research_view(data)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_lines = [
        f"生成时间：{now}",
        f"train_dir：{str(runs.train_dir) if runs.train_dir else 'NA'}",
        f"deployable_dir：{str(runs.deployable_dir) if runs.deployable_dir else 'NA'}",
        f"final_system_dir：{str(runs.final_system_dir) if runs.final_system_dir else 'NA'}",
        f"overlay_dir：{str(runs.overlay_dir) if runs.overlay_dir else 'NA'}",
    ]

    if warnings:
        exec_warn = "<div class='card'><h2>Warnings</h2>" + warning_box("\n".join(warnings)) + "</div>"
        executive_html = exec_warn + "<div style='height:14px;'></div>" + executive_html

    combined_html = build_page_html(
        title="Dissertation Trading Dashboard",
        meta_lines=meta_lines,
        executive_html=executive_html,
        research_html=research_html,
        combined=True,
    )
    out_combined = runs.out_dir / "dashboard_combined.html"
    out_combined.write_text(combined_html, encoding="utf-8")
    print(f"[OK] Wrote: {out_combined}")

    if args.write_split_pages:
        main_html = build_page_html(
            title="Dissertation Trading Dashboard (Executive)",
            meta_lines=meta_lines,
            executive_html=executive_html,
            research_html="",
            combined=False,
        )
        out_main = runs.out_dir / "dashboard_main.html"
        out_main.write_text(main_html, encoding="utf-8")
        print(f"[OK] Wrote: {out_main}")

        robust_html = build_page_html(
            title="Dissertation Trading Dashboard (Research)",
            meta_lines=meta_lines,
            executive_html="",
            research_html=research_html,
            combined=False,
        )
        out_robust = runs.out_dir / "dashboard_robustness.html"
        out_robust.write_text(robust_html, encoding="utf-8")
        print(f"[OK] Wrote: {out_robust}")

    if warnings:
        print("\n[WARNINGS]")
        for w in warnings:
            print(" -", w)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("FATAL ERROR")
        print(traceback.format_exc())
