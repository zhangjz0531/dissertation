from __future__ import annotations

"""
build_final_dashboard.py

目标：
- 将 Executive 主展示与 Research 鲁棒性分析合并为单一自包含 HTML（答辩级）
- 数据源自动定位最新 run_*（ablations / robustness）
- 支持 Tab 切换（Executive / Research）
- 支持 PNG 导出（Plotly modebar + toImageButtonOptions）
- 支持 PDF 导出（浏览器打印保存为 PDF；可选 Kaleido 内嵌图级 PDF/PNG 下载）
- 缺失文件时优雅降级（生成警告卡片，不中断全局输出）
- 输出：
    - dashboard_combined.html（必出）
    - dashboard_main.html（可选）
    - dashboard_robustness.html（可选）
"""

import argparse
import base64
import json
import math
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ---------------------------
# 配色与全局样式（与你要求的映射一致）
# ---------------------------
PALETTE = {
    "static": "#1f4e79",      # deep blue
    "sticky": "#ff7f0e",      # orange
    "robust": "#2ca02c",      # green
    "sens": "#9467bd",        # purple
    "bg": "#f6f7fb",
    "card": "#ffffff",
    "text": "#111827",
    "muted": "#6b7280",
    "border": "#e5e7eb",
    "warn_bg": "#fff7ed",
    "warn_border": "#fdba74",
}


# ---------------------------
# 数据结构
# ---------------------------
@dataclass
class RunPaths:
    ablation_run: Optional[Path]
    bootstrap_ci_run: Optional[Path]
    block_sens_run: Optional[Path]
    out_dir: Path


@dataclass
class Inputs:
    static_actions: Optional[pd.DataFrame]
    sticky_actions: Optional[pd.DataFrame]
    ablation_summary: Optional[dict]

    bootstrap_summary: Optional[dict]
    bootstrap_static: Optional[pd.DataFrame]
    bootstrap_sticky: Optional[pd.DataFrame]
    bootstrap_diff: Optional[pd.DataFrame]

    block_diff: Optional[pd.DataFrame]
    block_strategy: Optional[pd.DataFrame]


# ---------------------------
# 工具函数：文件/目录发现
# ---------------------------
def find_latest_run_dir(base_dir: Path, pattern: str = "run_*") -> Optional[Path]:
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


def pick_first_existing(parent: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = parent / n
        if p.exists():
            return p
    return None


# ---------------------------
# 指标计算
# ---------------------------
def annualized_return(final_equity: float, periods: int, horizon: int) -> float:
    if periods <= 0:
        return float("nan")
    total_days = max(1, periods * horizon)
    if final_equity <= 0:
        return -1.0
    return float(final_equity ** (252.0 / total_days) - 1.0)


def sharpe_ratio(returns: np.ndarray, horizon: int) -> float:
    if returns.size <= 1:
        return 0.0
    std = returns.std(ddof=1)
    if std <= 1e-12:
        return 0.0
    return float((returns.mean() / std) * math.sqrt(252.0 / horizon))


def equity_curve_from_net_returns(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty or "net_return" not in df.columns:
        return None
    r = df["net_return"].astype(float).values
    equity = np.cumprod(1.0 + r)
    return pd.Series(equity)


def drawdown_from_equity(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return dd


def compute_metrics(actions: pd.DataFrame, horizon: int) -> Dict[str, float]:
    out = {
        "periods": float("nan"),
        "cumulative_return": float("nan"),
        "annualized_return": float("nan"),
        "sharpe": float("nan"),
        "max_drawdown": float("nan"),
        "win_rate": float("nan"),
        "avg_turnover": float("nan"),
    }
    if actions is None or actions.empty or "net_return" not in actions.columns:
        return out

    r = actions["net_return"].astype(float).values
    eq = np.cumprod(1.0 + r)
    dd = (eq / np.maximum.accumulate(eq)) - 1.0

    out["periods"] = int(len(r))
    out["cumulative_return"] = float(eq[-1] - 1.0)
    out["annualized_return"] = annualized_return(eq[-1], len(r), horizon)
    out["sharpe"] = sharpe_ratio(r, horizon)
    out["max_drawdown"] = float(dd.min())
    out["win_rate"] = float((r > 0).mean())

    if "turnover" in actions.columns:
        out["avg_turnover"] = float(actions["turnover"].astype(float).mean())

    return out


# ---------------------------
# Plotly 图表构建
# ---------------------------
def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _get_x_axis(df: pd.DataFrame) -> np.ndarray:
    if df is not None and "date" in df.columns:
        try:
            return pd.to_datetime(df["date"]).values
        except Exception:
            pass
    # fallback：用序号
    if df is None:
        return np.arange(0)
    return np.arange(len(df))


def make_equity_figure(static_df: pd.DataFrame, sticky_df: pd.DataFrame) -> Optional[go.Figure]:
    if static_df is None and sticky_df is None:
        return None

    fig = go.Figure()
    if static_df is not None and "net_return" in static_df.columns:
        x = _get_x_axis(static_df)
        eq = equity_curve_from_net_returns(static_df)
        fig.add_trace(go.Scatter(
            x=x, y=eq, mode="lines", name="Static",
            line=dict(color=PALETTE["static"], width=3)
        ))
    if sticky_df is not None and "net_return" in sticky_df.columns:
        x = _get_x_axis(sticky_df)
        eq = equity_curve_from_net_returns(sticky_df)
        fig.add_trace(go.Scatter(
            x=x, y=eq, mode="lines", name="Sticky",
            line=dict(color=PALETTE["sticky"], width=3)
        ))

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        yaxis_title="Equity (起始=1.0)",
    )
    return fig


def make_drawdown_figure(static_df: pd.DataFrame, sticky_df: pd.DataFrame) -> Optional[go.Figure]:
    if static_df is None and sticky_df is None:
        return None

    fig = go.Figure()
    if static_df is not None and "net_return" in static_df.columns:
        x = _get_x_axis(static_df)
        eq = equity_curve_from_net_returns(static_df)
        dd = drawdown_from_equity(eq)
        fig.add_trace(go.Scatter(
            x=x, y=dd, mode="lines", name="Static DD",
            line=dict(color=PALETTE["static"], width=3)
        ))
    if sticky_df is not None and "net_return" in sticky_df.columns:
        x = _get_x_axis(sticky_df)
        eq = equity_curve_from_net_returns(sticky_df)
        dd = drawdown_from_equity(eq)
        fig.add_trace(go.Scatter(
            x=x, y=dd, mode="lines", name="Sticky DD",
            line=dict(color=PALETTE["sticky"], width=3)
        ))

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        yaxis_title="Drawdown",
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
        line_color=PALETTE["static"],
        fillcolor=_hex_to_rgba(PALETTE["static"], 0.25),
        box_visible=True,
        meanline_visible=True,
        points=False
    ))
    fig.add_trace(go.Violin(
        y=boot_sticky[col].astype(float),
        name="Sticky",
        line_color=PALETTE["sticky"],
        fillcolor=_hex_to_rgba(PALETTE["sticky"], 0.25),
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


def make_bootstrap_ci_bar(summary: dict, metric_key: str, title: str) -> Optional[go.Figure]:
    """
    summary: bootstrap_summary.json 内的 static_summary / sticky_summary
    metric_key: e.g. 'cumret' -> 需要 mean/ci_5/ci_95
    """
    if summary is None:
        return None

    s0 = summary.get("static_summary")
    s1 = summary.get("sticky_summary")
    if not s0 or not s1:
        return None

    mean0 = s0.get(f"{metric_key}_mean")
    lo0 = s0.get(f"{metric_key}_ci_5")
    hi0 = s0.get(f"{metric_key}_ci_95")

    mean1 = s1.get(f"{metric_key}_mean")
    lo1 = s1.get(f"{metric_key}_ci_5")
    hi1 = s1.get(f"{metric_key}_ci_95")

    if any(v is None for v in [mean0, lo0, hi0, mean1, lo1, hi1]):
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Static", "Sticky"],
        y=[mean0, mean1],
        marker_color=[PALETTE["static"], PALETTE["sticky"]],
        error_y=dict(
            type="data",
            symmetric=False,
            array=[hi0 - mean0, hi1 - mean1],
            arrayminus=[mean0 - lo0, mean1 - lo1],
            thickness=1.2,
            width=3
        ),
        name="mean & 5-95%"
    ))
    fig.update_layout(
        template="plotly_white",
        title=title,
        margin=dict(l=40, r=20, t=60, b=40),
        height=420,
        yaxis_title=metric_key
    )
    return fig


def make_block_sensitivity_figure(block_diff: pd.DataFrame, y_mean: str, y_lo: str, y_hi: str, title: str) -> Optional[go.Figure]:
    if block_diff is None or block_diff.empty:
        return None
    needed = { "block_length", y_mean, y_lo, y_hi }
    if not needed.issubset(set(block_diff.columns)):
        return None

    df = block_diff.sort_values("block_length").copy()
    x = df["block_length"].astype(int).values
    mean = df[y_mean].astype(float).values
    lo = df[y_lo].astype(float).values
    hi = df[y_hi].astype(float).values

    fig = go.Figure()
    # CI band
    fig.add_trace(go.Scatter(
        x=x, y=hi,
        mode="lines",
        line=dict(color=_hex_to_rgba(PALETTE["sens"], 0.0), width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=x, y=lo,
        mode="lines",
        fill="tonexty",
        fillcolor=_hex_to_rgba(PALETTE["sens"], 0.18),
        line=dict(color=_hex_to_rgba(PALETTE["sens"], 0.0), width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    # mean line
    fig.add_trace(go.Scatter(
        x=x, y=mean,
        mode="lines+markers",
        line=dict(color=PALETTE["sens"], width=3),
        marker=dict(size=7),
        name="mean (Sticky - Static)"
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


def make_winprob_bar(diff_summary: dict) -> Optional[go.Figure]:
    if not diff_summary:
        return None

    # 使用“static wins”的概率（更符合最终系统叙事）
    p_c = diff_summary.get("prob_static_beats_sticky_cumret")
    p_s = diff_summary.get("prob_static_beats_sticky_sharpe")
    p_m = diff_summary.get("prob_static_beats_sticky_mdd")
    if any(v is None for v in [p_c, p_s, p_m]):
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["CumRet", "Sharpe", "MDD"],
        y=[p_c, p_s, p_m],
        marker_color=PALETTE["robust"],
        name="P(Static beats Sticky)"
    ))
    fig.update_layout(
        template="plotly_white",
        title="胜率（配对Bootstrap）：P(Static > Sticky)",
        margin=dict(l=40, r=20, t=60, b=40),
        height=420,
        yaxis=dict(range=[0, 1.0])
    )
    return fig


# ---------------------------
# Kaleido 静态资源内嵌（可选）
# ---------------------------
def try_export_static_assets(fig: go.Figure, name: str, warnings: List[str], scale: int = 3) -> Dict[str, str]:
    """
    返回 {f"{name}_png": base64, f"{name}_pdf": base64}
    若环境无 Kaleido 或导出失败，返回空 dict。
    """
    if fig is None:
        return {}

    try:
        # 仅在真正需要时导入；无 kaleido 也不影响 HTML 生成
        import kaleido  # noqa: F401
    except Exception:
        return {}

    assets: Dict[str, str] = {}
    try:
        png_bytes = fig.to_image(format="png", scale=scale)
        pdf_bytes = fig.to_image(format="pdf")
        assets[f"{name}_png"] = base64.b64encode(png_bytes).decode("ascii")
        assets[f"{name}_pdf"] = base64.b64encode(pdf_bytes).decode("ascii")
    except Exception as e:
        warnings.append(f"Kaleido 导出失败（{name}）：{e}")
    return assets


# ---------------------------
# HTML 拼装
# ---------------------------
def fmt_pct(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x * 100:.{digits}f}%"


def fmt_num(x: float, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{digits}f}"


def card_html(title: str, value: str, subtitle: str = "", color_key: str = "static") -> str:
    border = PALETTE[color_key]
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
      <div class="warning-title">⚠ 数据缺失/降级</div>
      <div class="warning-msg">{msg}</div>
    </div>
    """


def df_to_table_html(df: pd.DataFrame, table_id: str) -> str:
    if df is None or df.empty:
        return warning_box("表格数据为空。")
    # 轻量表格 HTML（便于打印）
    return df.to_html(index=False, escape=True, table_id=table_id, classes="tbl")


def plot_to_div(fig: go.Figure, div_id: str, include_plotlyjs: bool, filename_stub: str) -> str:
    """
    返回 fig 的 HTML div（full_html=False），用于嵌入自定义页面。
    include_plotlyjs=True 只对第一个图启用，确保整体自包含离线可用。
    """
    if fig is None:
        return warning_box(f"图表缺失：{div_id}")

    config = {
        "displaylogo": False,
        "responsive": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": filename_stub,
            "scale": 3
        }
    }
    return fig.to_html(
        full_html=False,
        include_plotlyjs=True if include_plotlyjs else False,
        config=config,
        div_id=div_id
    )


def build_css() -> str:
    return f"""
    <style>
      :root {{
        --static: {PALETTE["static"]};
        --sticky: {PALETTE["sticky"]};
        --robust: {PALETTE["robust"]};
        --sens: {PALETTE["sens"]};
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
        max-width: 1280px;
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
        letter-spacing: 0.2px;
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
        border-color: #cbd5e1;
        background: #111827;
        color: #fff;
      }}

      .btn-primary:hover {{
        background: #0b1220;
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

      .download-row {{
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin: 8px 0 6px 0;
      }}

      /* 打印/导出 PDF：只打印当前 tab 内容 */
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


def build_js(embedded_assets: Dict[str, str]) -> str:
    # 将 base64 静态资源放到 window.__STATIC_ASSETS__，用于下载按钮
    assets_json = json.dumps(embedded_assets, ensure_ascii=False)

    return f"""
    <script>
      window.__STATIC_ASSETS__ = {assets_json};

      function setActiveTab(tabId) {{
        const tabs = document.querySelectorAll('.tab');
        const contents = document.querySelectorAll('.tab-content');

        tabs.forEach(t => {{
          const isActive = (t.getAttribute('data-tab') === tabId);
          t.setAttribute('aria-selected', isActive ? 'true' : 'false');
        }});

        contents.forEach(c => {{
          const isActive = (c.id === tabId);
          c.classList.toggle('active', isActive);
          c.classList.toggle('print-target', isActive);
        }});

        // 触发 Plotly resize，避免隐藏容器渲染尺寸异常
        setTimeout(() => {{
          try {{
            if (window.Plotly) {{
              const active = document.getElementById(tabId);
              const graphs = active.querySelectorAll('.plotly-graph-div');
              graphs.forEach(gd => {{
                try {{ Plotly.Plots.resize(gd); }} catch (e) {{}}
              }});
            }}
          }} catch (e) {{}}
        }}, 250);
      }}

      // 键盘无障碍：左右键切换 tab
      document.addEventListener('keydown', (e) => {{
        if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
        const tabs = Array.from(document.querySelectorAll('.tab'));
        const idx = tabs.findIndex(t => t.getAttribute('aria-selected') === 'true');
        if (idx < 0) return;
        const next = (e.key === 'ArrowRight') ? (idx + 1) % tabs.length : (idx - 1 + tabs.length) % tabs.length;
        setActiveTab(tabs[next].getAttribute('data-tab'));
      }});

      function b64ToBlob(b64Data, contentType) {{
        contentType = contentType || '';
        const sliceSize = 1024;
        const byteCharacters = atob(b64Data);
        const byteArrays = [];

        for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {{
          const slice = byteCharacters.slice(offset, offset + sliceSize);
          const byteNumbers = new Array(slice.length);
          for (let i = 0; i < slice.length; i++) {{
            byteNumbers[i] = slice.charCodeAt(i);
          }}
          const byteArray = new Uint8Array(byteNumbers);
          byteArrays.push(byteArray);
        }}
        return new Blob(byteArrays, {{ type: contentType }});
      }}

      function downloadEmbedded(key, filename, mime) {{
        const b64 = window.__STATIC_ASSETS__[key];
        if (!b64) {{
          alert('未内嵌该静态资源（可能未安装 Kaleido 或导出失败）。你仍可使用图表右上角下载 PNG，或用“打印/保存为 PDF”。');
          return;
        }}
        const blob = b64ToBlob(b64, mime);
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }}

      function printToPDF() {{
        window.print();
      }}
    </script>
    """


def build_page_html(
    title: str,
    meta_lines: List[str],
    executive_html: str,
    research_html: str,
    combined: bool,
    embedded_assets: Dict[str, str],
) -> str:
    css = build_css()
    js = build_js(embedded_assets)

    meta_html = "<br/>".join([f"{m}" for m in meta_lines])

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
        content_html = f"""
        <div id="single" class="tab-content active print-target">{executive_html}{research_html}</div>
        """

    html = f"""
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
          说明：本页面可离线打开；图表右上角可下载 PNG（高分辨率可由 scale 控制）。如需图级 PDF 下载，可安装 Kaleido 后重新生成（脚本会自动内嵌 PDF）。 
        </div>
      </div>
      {js}
    </body>
    </html>
    """
    return html


# ---------------------------
# 主流程：发现 -> 读取 -> 构建 -> 输出
# ---------------------------
def discover_runs(project_root: Path) -> RunPaths:
    model_runs = project_root / "Model Runs"
    ablation_base = model_runs / "ablations" / "sticky_execution"
    ci_base = model_runs / "robustness" / "bootstrap_ci"
    block_base = model_runs / "robustness" / "bootstrap_block_sensitivity"
    out_dir = model_runs / "reporting"
    out_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        ablation_run=find_latest_run_dir(ablation_base, "run_*"),
        bootstrap_ci_run=find_latest_run_dir(ci_base, "run_*"),
        block_sens_run=find_latest_run_dir(block_base, "run_*"),
        out_dir=out_dir
    )


def load_inputs(runs: RunPaths, warnings: List[str]) -> Inputs:
    # ---- ablation
    static_actions = None
    sticky_actions = None
    ablation_summary = None

    if runs.ablation_run:
        static_path = pick_first_existing(runs.ablation_run, ["frozen_static_test_actions.csv", "static_test_actions.csv"])
        sticky_path = pick_first_existing(runs.ablation_run, ["sticky_test_actions.csv"])
        if static_path:
            static_actions = safe_read_csv(static_path, warnings)
        else:
            warnings.append(f"未找到 static test actions（{runs.ablation_run}）")

        if sticky_path:
            sticky_actions = safe_read_csv(sticky_path, warnings)
        else:
            warnings.append(f"未找到 sticky test actions（{runs.ablation_run}）")

        ablation_summary = safe_read_json(runs.ablation_run / "ablation_summary.json", warnings)
    else:
        warnings.append("未找到 ablation run（ablations/sticky_execution/run_*）。")

    # ---- bootstrap ci
    bootstrap_summary = None
    boot_static = None
    boot_sticky = None
    boot_diff = None

    if runs.bootstrap_ci_run:
        bootstrap_summary = safe_read_json(runs.bootstrap_ci_run / "bootstrap_summary.json", warnings)
        boot_static = safe_read_csv(runs.bootstrap_ci_run / "bootstrap_static_test.csv", warnings)
        boot_sticky = safe_read_csv(runs.bootstrap_ci_run / "bootstrap_sticky_test.csv", warnings)
        boot_diff = safe_read_csv(runs.bootstrap_ci_run / "bootstrap_difference_test.csv", warnings)
    else:
        warnings.append("未找到 bootstrap_ci run（robustness/bootstrap_ci/run_*）。")

    # ---- block sensitivity
    block_diff = None
    block_strategy = None
    if runs.block_sens_run:
        block_diff = safe_read_csv(runs.block_sens_run / "block_length_difference_summary.csv", warnings)
        block_strategy = safe_read_csv(runs.block_sens_run / "block_length_strategy_summary.csv", warnings)
    else:
        warnings.append("未找到 block_sensitivity run（robustness/bootstrap_block_sensitivity/run_*）。")

    return Inputs(
        static_actions=static_actions,
        sticky_actions=sticky_actions,
        ablation_summary=ablation_summary,
        bootstrap_summary=bootstrap_summary,
        bootstrap_static=boot_static,
        bootstrap_sticky=boot_sticky,
        bootstrap_diff=boot_diff,
        block_diff=block_diff,
        block_strategy=block_strategy
    )


def build_executive_view(inputs: Inputs, horizon: int, embedded_assets: Dict[str, str], warnings: List[str]) -> str:
    s_metrics = compute_metrics(inputs.static_actions, horizon)
    t_metrics = compute_metrics(inputs.sticky_actions, horizon)

    # KPI cards（你可按论文叙事微调）
    kpi_cards = []
    kpi_cards.append(card_html("最终系统（Static）累计收益", fmt_pct(s_metrics["cumulative_return"]), "Test Period", "static"))
    kpi_cards.append(card_html("最终系统（Static）年化收益", fmt_pct(s_metrics["annualized_return"]), f"horizon={horizon}", "static"))
    kpi_cards.append(card_html("最终系统（Static）Sharpe", fmt_num(s_metrics["sharpe"], 4), "risk-free=0（默认）", "static"))
    kpi_cards.append(card_html("最终系统（Static）最大回撤", fmt_pct(s_metrics["max_drawdown"]), "越接近0越好", "static"))
    kpi_cards.append(card_html("对照（Sticky）累计收益", fmt_pct(t_metrics["cumulative_return"]), "Test Period", "sticky"))

    if (not math.isnan(s_metrics["cumulative_return"])) and (not math.isnan(t_metrics["cumulative_return"])):
        excess = s_metrics["cumulative_return"] - t_metrics["cumulative_return"]
        kpi_cards.append(card_html("超额收益（Static - Sticky）", fmt_pct(excess), "正值=Static更优", "robust"))
    else:
        kpi_cards.append(card_html("超额收益（Static - Sticky）", "NA", "缺少输入无法计算", "robust"))

    kpi_html = f"""
    <div class="grid-3">
      {''.join(kpi_cards)}
    </div>
    """

    # Portfolio Overview table
    overview = pd.DataFrame([
        ["Cumulative Return", s_metrics["cumulative_return"], t_metrics["cumulative_return"], (s_metrics["cumulative_return"] - t_metrics["cumulative_return"]) if (not math.isnan(s_metrics["cumulative_return"]) and not math.isnan(t_metrics["cumulative_return"])) else float("nan")],
        ["Annualized Return",  s_metrics["annualized_return"],  t_metrics["annualized_return"],  (s_metrics["annualized_return"] - t_metrics["annualized_return"]) if (not math.isnan(s_metrics["annualized_return"]) and not math.isnan(t_metrics["annualized_return"])) else float("nan")],
        ["Sharpe",             s_metrics["sharpe"],             t_metrics["sharpe"],             (s_metrics["sharpe"] - t_metrics["sharpe"]) if (not math.isnan(s_metrics["sharpe"]) and not math.isnan(t_metrics["sharpe"])) else float("nan")],
        ["Max Drawdown",       s_metrics["max_drawdown"],       t_metrics["max_drawdown"],       (s_metrics["max_drawdown"] - t_metrics["max_drawdown"]) if (not math.isnan(s_metrics["max_drawdown"]) and not math.isnan(t_metrics["max_drawdown"])) else float("nan")],
        ["Win Rate",           s_metrics["win_rate"],           t_metrics["win_rate"],           (s_metrics["win_rate"] - t_metrics["win_rate"]) if (not math.isnan(s_metrics["win_rate"]) and not math.isnan(t_metrics["win_rate"])) else float("nan")],
        ["Avg Turnover",       s_metrics["avg_turnover"],       t_metrics["avg_turnover"],       (s_metrics["avg_turnover"] - t_metrics["avg_turnover"]) if (not math.isnan(s_metrics["avg_turnover"]) and not math.isnan(t_metrics["avg_turnover"])) else float("nan")],
    ], columns=["Metric", "Static", "Sticky", "Static - Sticky"])

    # 格式化显示（百分比/数值）
    def _fmt_row(metric: str, v: float) -> str:
        if metric in {"Cumulative Return", "Annualized Return", "Max Drawdown", "Win Rate", "Avg Turnover"}:
            return fmt_pct(v, 2)
        return fmt_num(v, 4)

    overview_fmt = overview.copy()
    for col in ["Static", "Sticky", "Static - Sticky"]:
        overview_fmt[col] = [
            _fmt_row(m, float(v)) if v == v else "NA"
            for m, v in zip(overview["Metric"], overview[col])
        ]

    overview_html = f"""
    <div class="card">
      <h2>Portfolio Overview（Test）</h2>
      {df_to_table_html(overview_fmt, "tbl_overview")}
      <div class="small">说明：指标由 test actions 的 net_return 计算；交易成本若已计入 net_return，则这里不重复扣除。</div>
    </div>
    """

    # Equity & Drawdown figs（关键图第一个 include_plotlyjs=True，保证页面离线自包含）
    eq_fig = make_equity_figure(inputs.static_actions, inputs.sticky_actions)
    dd_fig = make_drawdown_figure(inputs.static_actions, inputs.sticky_actions)

    # 可选：内嵌静态资源（PNG/PDF）用于按钮下载
    embedded_assets.update(try_export_static_assets(eq_fig, "equity", warnings))
    embedded_assets.update(try_export_static_assets(dd_fig, "drawdown", warnings))

    eq_div = plot_to_div(eq_fig, "fig_equity", include_plotlyjs=True, filename_stub="equity_curve")
    dd_div = plot_to_div(dd_fig, "fig_drawdown", include_plotlyjs=False, filename_stub="drawdown_curve")

    eq_panel = f"""
    <div class="card">
      <h2>Equity Curve（Test）</h2>
      <div class="download-row no-print">
        <button class="btn" onclick="downloadEmbedded('equity_png','equity_curve.png','image/png')">下载PNG（内嵌）</button>
        <button class="btn" onclick="downloadEmbedded('equity_pdf','equity_curve.pdf','application/pdf')">下载PDF（内嵌）</button>
      </div>
      {eq_div}
    </div>
    """

    dd_panel = f"""
    <div class="card">
      <h2>Drawdown Curve（Test）</h2>
      <div class="download-row no-print">
        <button class="btn" onclick="downloadEmbedded('drawdown_png','drawdown_curve.png','image/png')">下载PNG（内嵌）</button>
        <button class="btn" onclick="downloadEmbedded('drawdown_pdf','drawdown_curve.pdf','application/pdf')">下载PDF（内嵌）</button>
      </div>
      {dd_div}
    </div>
    """

    charts_html = f"""
    <div class="grid-2">
      {eq_panel}
      {dd_panel}
    </div>
    """

    # Final decision table（优先来自 ablation_summary.json；缺失则用默认兜底）
    frozen_cfg = {}
    if inputs.ablation_summary and isinstance(inputs.ablation_summary, dict):
        frozen_cfg = inputs.ablation_summary.get("frozen_static_cfg", {}) or {}

    top_k = frozen_cfg.get("top_k", 2)
    min_prob = frozen_cfg.get("min_prob", 0.54)
    threshold = frozen_cfg.get("threshold", 0.5)

    decision_df = pd.DataFrame([
        ["Prediction Model", "Transformer H5（若需从 manifest 自动读取可扩展）"],
        ["Execution Layer", "Static（最终部署）"],
        ["Top-K", str(top_k)],
        ["Min Prob", str(min_prob)],
        ["Threshold", str(threshold)],
        ["Decision", "保留 Static；Sticky 作为 ablation/robustness 负结果展示"],
    ], columns=["Item", "Value"])

    decision_html = f"""
    <div class="card">
      <h2>Final Decision（Deployment）</h2>
      {df_to_table_html(decision_df, "tbl_decision")}
      <div class="small">说明：该表用于答辩“冻结最终系统配置”。</div>
    </div>
    """

    # Provenance / warnings
    warn_html = ""
    if warnings:
        warn_html = f"""
        <div class="card">
          <h2>Data Warnings / Provenance</h2>
          {warning_box("\\n".join(warnings))}
          <div class="small">提示：若你希望完全固定某次 run，请在脚本里手动指定 run 目录或添加 CLI 参数。</div>
        </div>
        """

    return f"""
    {kpi_html}
    <div style="height:14px;"></div>
    {overview_html}
    <div style="height:14px;"></div>
    {charts_html}
    <div style="height:14px;"></div>
    {decision_html}
    <div style="height:14px;"></div>
    {warn_html}
    """


def build_research_view(inputs: Inputs, embedded_assets: Dict[str, str], warnings: List[str]) -> str:
    # Bootstrap figures
    bs_summary = inputs.bootstrap_summary or {}
    paired = bs_summary.get("paired_difference_summary", {}) if isinstance(bs_summary, dict) else {}

    fig_bs_cum = make_bootstrap_violin(inputs.bootstrap_static, inputs.bootstrap_sticky, "cumulative_return", "Bootstrap 分布：Cumulative Return")
    fig_bs_shp = make_bootstrap_violin(inputs.bootstrap_static, inputs.bootstrap_sticky, "sharpe", "Bootstrap 分布：Sharpe")
    fig_ci_cum = make_bootstrap_ci_bar(bs_summary, "cumret", "Bootstrap 汇总：CumRet mean & 5-95%")
    fig_ci_shp = make_bootstrap_ci_bar(bs_summary, "sharpe", "Bootstrap 汇总：Sharpe mean & 5-95%")

    # Block sensitivity figure（差值随 block length）
    fig_block_c = make_block_sensitivity_figure(
        inputs.block_diff,
        y_mean="cumret_diff_mean",
        y_lo="cumret_diff_ci_5",
        y_hi="cumret_diff_ci_95",
        title="Block-Length Sensitivity：CumRet diff（Sticky - Static）"
    )
    fig_block_s = make_block_sensitivity_figure(
        inputs.block_diff,
        y_mean="sharpe_diff_mean",
        y_lo="sharpe_diff_ci_5",
        y_hi="sharpe_diff_ci_95",
        title="Block-Length Sensitivity：Sharpe diff（Sticky - Static）"
    )

    fig_win = make_winprob_bar(paired)

    # 内嵌静态资源（可选）
    embedded_assets.update(try_export_static_assets(fig_bs_cum, "bs_cum", warnings))
    embedded_assets.update(try_export_static_assets(fig_ci_cum, "ci_cum", warnings))
    embedded_assets.update(try_export_static_assets(fig_block_c, "block_cum", warnings))
    embedded_assets.update(try_export_static_assets(fig_win, "winprob", warnings))

    # Plotly div（注意：这里 include_plotlyjs=False，因为 Executive 的第一个图已加载）
    bs_cum_div = plot_to_div(fig_bs_cum, "fig_bs_cum", include_plotlyjs=False, filename_stub="bootstrap_cumret")
    bs_shp_div = plot_to_div(fig_bs_shp, "fig_bs_shp", include_plotlyjs=False, filename_stub="bootstrap_sharpe")
    ci_cum_div = plot_to_div(fig_ci_cum, "fig_ci_cum", include_plotlyjs=False, filename_stub="bootstrap_ci_cumret")
    ci_shp_div = plot_to_div(fig_ci_shp, "fig_ci_shp", include_plotlyjs=False, filename_stub="bootstrap_ci_sharpe")
    block_c_div = plot_to_div(fig_block_c, "fig_block_c", include_plotlyjs=False, filename_stub="block_sensitivity_cumret")
    block_s_div = plot_to_div(fig_block_s, "fig_block_s", include_plotlyjs=False, filename_stub="block_sensitivity_sharpe")
    win_div = plot_to_div(fig_win, "fig_winprob", include_plotlyjs=False, filename_stub="win_probability")

    # Paired diff table（来自 bootstrap_summary.json）
    paired_rows = []
    if paired:
        paired_rows = [
            ["CumRet (Sticky-Static)", paired.get("cumret_diff_mean"), paired.get("cumret_diff_ci_5"), paired.get("cumret_diff_ci_95"), paired.get("prob_static_beats_sticky_cumret")],
            ["Sharpe (Sticky-Static)", paired.get("sharpe_diff_mean"), paired.get("sharpe_diff_ci_5"), paired.get("sharpe_diff_ci_95"), paired.get("prob_static_beats_sticky_sharpe")],
            ["MDD (Sticky-Static)", paired.get("mdd_diff_mean"), paired.get("mdd_diff_ci_5"), paired.get("mdd_diff_ci_95"), paired.get("prob_static_beats_sticky_mdd")],
        ]
    paired_df = pd.DataFrame(paired_rows, columns=["Metric", "Mean", "CI_5", "CI_95", "P(Static wins)"]) if paired_rows else pd.DataFrame()

    # block-length diff table（可选输出简版）
    block_table = pd.DataFrame()
    if inputs.block_diff is not None and not inputs.block_diff.empty:
        cols = [c for c in [
            "block_length",
            "cumret_diff_mean", "cumret_diff_ci_5", "cumret_diff_ci_95", "prob_static_beats_sticky_cumret",
            "sharpe_diff_mean", "sharpe_diff_ci_5", "sharpe_diff_ci_95", "prob_static_beats_sticky_sharpe",
        ] if c in inputs.block_diff.columns]
        block_table = inputs.block_diff[cols].sort_values("block_length").copy()

    paired_html = f"""
    <div class="card">
      <h2>Paired Difference Summary（Sticky - Static）</h2>
      {df_to_table_html(paired_df, "tbl_paired")}
      <div class="small">说明：差值为 Sticky-Static；若均值<0 且 P(Static wins) 较高，则支持最终选择 Static。</div>
    </div>
    """

    block_tbl_html = f"""
    <div class="card">
      <h2>Block-Length Difference Table（精选列）</h2>
      {df_to_table_html(block_table, "tbl_block")}
      <div class="small">说明：用于展示结论不依赖单一 block length。</div>
    </div>
    """ if not block_table.empty else f"""
    <div class="card">
      <h2>Block-Length Difference Table</h2>
      {warning_box("缺少 block_length_difference_summary.csv 或列不全，无法生成表格。")}
    </div>
    """

    # Panels
    p1 = f"""
    <div class="card">
      <h2>Bootstrap 分布（CumRet）</h2>
      <div class="download-row no-print">
        <button class="btn" onclick="downloadEmbedded('bs_cum_png','bootstrap_cumret.png','image/png')">下载PNG（内嵌）</button>
        <button class="btn" onclick="downloadEmbedded('bs_cum_pdf','bootstrap_cumret.pdf','application/pdf')">下载PDF（内嵌）</button>
      </div>
      {bs_cum_div}
    </div>
    """
    p2 = f"""
    <div class="card">
      <h2>Bootstrap 分布（Sharpe）</h2>
      {bs_shp_div}
    </div>
    """
    p3 = f"""
    <div class="card">
      <h2>Bootstrap CI 汇总（CumRet）</h2>
      <div class="download-row no-print">
        <button class="btn" onclick="downloadEmbedded('ci_cum_png','bootstrap_ci_cumret.png','image/png')">下载PNG（内嵌）</button>
        <button class="btn" onclick="downloadEmbedded('ci_cum_pdf','bootstrap_ci_cumret.pdf','application/pdf')">下载PDF（内嵌）</button>
      </div>
      {ci_cum_div}
    </div>
    """
    p4 = f"""
    <div class="card">
      <h2>Bootstrap CI 汇总（Sharpe）</h2>
      {ci_shp_div}
    </div>
    """
    p5 = f"""
    <div class="card">
      <h2>Block-length Sensitivity（CumRet diff）</h2>
      <div class="download-row no-print">
        <button class="btn" onclick="downloadEmbedded('block_cum_png','block_sensitivity_cumret.png','image/png')">下载PNG（内嵌）</button>
        <button class="btn" onclick="downloadEmbedded('block_cum_pdf','block_sensitivity_cumret.pdf','application/pdf')">下载PDF（内嵌）</button>
      </div>
      {block_c_div}
    </div>
    """
    p6 = f"""
    <div class="card">
      <h2>Block-length Sensitivity（Sharpe diff）</h2>
      {block_s_div}
    </div>
    """
    p7 = f"""
    <div class="card">
      <h2>Win Probability（配对Bootstrap）</h2>
      <div class="download-row no-print">
        <button class="btn" onclick="downloadEmbedded('winprob_png','win_probability.png','image/png')">下载PNG（内嵌）</button>
        <button class="btn" onclick="downloadEmbedded('winprob_pdf','win_probability.pdf','application/pdf')">下载PDF（内嵌）</button>
      </div>
      {win_div}
    </div>
    """

    # 如果核心 research 输入缺失，给提示
    if inputs.bootstrap_summary is None and inputs.block_diff is None:
        return warning_box("Research 视图缺少 robustness 输入（bootstrap_summary.json / block_length_difference_summary.csv）。请先运行 robustness 脚本或指定 ablation/robustness 目录。")

    return f"""
    <div class="grid-2">
      {p1}{p2}
    </div>
    <div style="height:14px;"></div>
    <div class="grid-2">
      {p3}{p4}
    </div>
    <div style="height:14px;"></div>
    <div class="grid-2">
      {p5}{p6}
    </div>
    <div style="height:14px;"></div>
    <div class="grid-2">
      {p7}{paired_html}
    </div>
    <div style="height:14px;"></div>
    {block_tbl_html}
    """


def main():
    parser = argparse.ArgumentParser(description="Build merged Executive+Research defense-ready dashboard (self-contained HTML).")
    parser.add_argument("--project_root", type=str, default=r"D:\python\dissertation")
    parser.add_argument("--horizon", type=int, default=5, help="rebalance horizon in trading days (default=5)")
    parser.add_argument("--write_split_pages", action="store_true", help="also write dashboard_main.html and dashboard_robustness.html")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    warnings: List[str] = []

    runs = discover_runs(project_root)
    if runs.ablation_run is None:
        warnings.append("ablation_run 未找到：无法生成完整 Executive 对比。")
    if runs.bootstrap_ci_run is None:
        warnings.append("bootstrap_ci_run 未找到：Research 中 bootstrap 面板将降级。")
    if runs.block_sens_run is None:
        warnings.append("block_sens_run 未找到：Research 中 block sensitivity 面板将降级。")

    inputs = load_inputs(runs, warnings)

    embedded_assets: Dict[str, str] = {}  # 可选：Kaleido 生成的 PNG/PDF 的 base64

    exec_html = build_executive_view(inputs, horizon=args.horizon, embedded_assets=embedded_assets, warnings=warnings)
    research_html = build_research_view(inputs, embedded_assets=embedded_assets, warnings=warnings)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_lines = [
        f"生成时间：{now}",
        f"ablation_run：{str(runs.ablation_run) if runs.ablation_run else 'NA'}",
        f"bootstrap_ci_run：{str(runs.bootstrap_ci_run) if runs.bootstrap_ci_run else 'NA'}",
        f"block_sensitivity_run：{str(runs.block_sens_run) if runs.block_sens_run else 'NA'}",
        f"horizon：{args.horizon}",
    ]

    # Combined
    combined_html = build_page_html(
        title="Dissertation Trading Dashboard",
        meta_lines=meta_lines,
        executive_html=exec_html,
        research_html=research_html,
        combined=True,
        embedded_assets=embedded_assets
    )
    out_combined = runs.out_dir / "dashboard_combined.html"
    out_combined.write_text(combined_html, encoding="utf-8")
    print(f"[OK] Wrote: {out_combined}")

    # Optional split pages (非必须)
    if args.write_split_pages:
        main_html = build_page_html(
            title="Dissertation Trading Dashboard (Executive)",
            meta_lines=meta_lines,
            executive_html=exec_html,
            research_html="",
            combined=False,
            embedded_assets=embedded_assets
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
            embedded_assets=embedded_assets
        )
        out_robust = runs.out_dir / "dashboard_robustness.html"
        out_robust.write_text(robust_html, encoding="utf-8")
        print(f"[OK] Wrote: {out_robust}")

    # 将 warnings 同步输出到控制台，便于快速定位缺失文件
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
