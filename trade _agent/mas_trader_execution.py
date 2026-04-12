import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
import matplotlib.pyplot as plt
from collections import deque

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
rl_folder_path = os.path.join(PROJECT_ROOT, "Reinforcement Learning & Blind Test")
sys.path.append(rl_folder_path)

try:
    from td3_agent import TD3
except ImportError:
    print("[!] 致命错误：找不到 td3_agent")
    exit()

warnings.filterwarnings('ignore')

MACRO_PATH = r"D:\python\dissertation\Data Acquisition\download_data\esg_data\SPY_Macro_State.csv"
MICRO_PATH = r"D:\python\dissertation\Data Acquisition\download_data\esg_data\MUSA_Top10_Panel.csv"
ENCODER_PATH = r"D:\python\dissertation\models\encoder_pretrained.pth"
TD3_DIR = r"D:\python\dissertation\models\td3_final"


def run_scenario(df_macro, df_micro, agent, start_date, end_date, scenario_name, results_dir):
    print(f"\n[{scenario_name}] 🚀 交易员代理进场，执行 {start_date} 至 {end_date} 模拟盘交易...")

    available_features = [
        'dc_trend', 'dc_event', 'dc_drawdown', 'dc_T', 'dc_TMV', 'dc_R',
        'RSI', 'MACD_Pct', 'sentiment_score', 'interest_rate', 'credit_stress'
    ]

    df_test_macro = df_macro[(df_macro['date'] >= start_date) & (df_macro['date'] <= end_date)].copy().reset_index(
        drop=True)
    if df_test_macro.empty: return

    test_dates = sorted(df_test_macro['date'].unique())
    window_size = 30
    macro_state_buffer = deque(maxlen=window_size)
    first_state = df_test_macro.loc[0, available_features].values.astype(np.float32)
    for _ in range(window_size): macro_state_buffer.append(first_state)

    cap_bh = [100000.0];
    ret_bh_list = []
    cap_td3 = [100000.0];
    ret_td3_list = []
    cap_musa = [100000.0];
    ret_musa_list = []
    cap_mas = [100000.0];
    ret_mas_list = []

    dates_plot = []
    last_top_3 = []
    last_exposure = 1.0
    exposure_history = []

    for i in range(1, len(test_dates)):
        today_date_str = test_dates[i]
        yesterday_date_str = test_dates[i - 1]
        today_dt = pd.to_datetime(today_date_str)
        yesterday_dt = pd.to_datetime(yesterday_date_str)

        current_state_seq = np.array(macro_state_buffer).astype(np.float32)

        # 🧠 MAS 原生指令 1：风控经理 TD3 决定宏观敞口 (不再被任何人干预)
        td3_action = agent.select_action(current_state_seq)[0]
        macro_exposure = 0.2 + (td3_action * 0.8)
        exposure_history.append(macro_exposure)

        # 🧠 MAS 原生指令 2：分析师团队 MUSA 选出多因子龙头
        lookback_dt = today_dt - pd.Timedelta(days=90)
        past_data = df_micro[(df_micro['date'] >= lookback_dt) & (df_micro['date'] <= yesterday_dt)]

        def calc_multifactor_score(df_chunk):
            if len(df_chunk) < 20: return 0.0
            fund_score = df_chunk['net_margin_Z'].iloc[-1] * 2.0
            tech_score = df_chunk['RSI_Z'].iloc[-1] + df_chunk['MACD_Pct_Z'].iloc[-1]
            return fund_score + tech_score

        if not past_data.empty:
            metrics = past_data.groupby('stock').apply(calc_multifactor_score)
            top_3_tickers = metrics.nlargest(3).index.tolist()
        else:
            top_3_tickers = last_top_3

        # --- [环境结算与正确的数学摩擦计算] ---
        today_prices = df_micro[df_micro['date'] == today_dt]
        yesterday_prices = df_micro[df_micro['date'] == yesterday_dt]
        merged = pd.merge(today_prices[['stock', 'close']], yesterday_prices[['stock', 'close']], on='stock',
                          suffixes=('_today', '_yest'))

        selected_ret_data = merged[merged['stock'].isin(top_3_tickers)]
        avg_micro_return = selected_ret_data['close_today'].sub(selected_ret_data['close_yest']).div(
            selected_ret_data['close_yest']).mean() if not selected_ret_data.empty else 0.0
        bench_ret = (df_test_macro.loc[i, 'close'] - df_test_macro.loc[i - 1, 'close']) / df_test_macro.loc[
            i - 1, 'close']

        change_count = len(set(top_3_tickers) - set(last_top_3)) if last_top_3 else 3

        # 🚨 已修复的摩擦公式：换股手续费只针对真实持仓比例计算！
        stock_turnover_cost_musa = (change_count / 3.0) * 0.0002
        stock_turnover_cost_mas = (change_count / 3.0) * 0.0002 * macro_exposure
        macro_turnover_cost = abs(macro_exposure - last_exposure) * 0.0002

        # 空仓资金产生 4% 的年化无风险收益
        daily_rf = 0.04 / 252
        cash_return = (1.0 - macro_exposure) * daily_rf

        net_bh = bench_ret
        net_td3 = (macro_exposure * bench_ret) + cash_return - macro_turnover_cost
        net_musa = avg_micro_return - stock_turnover_cost_musa

        # 原生 MAS 收益：TD3仓位 * MUSA精选股 + 现金利息 - 精准摩擦成本
        net_mas = (macro_exposure * avg_micro_return) + cash_return - stock_turnover_cost_mas - macro_turnover_cost

        cap_bh.append(cap_bh[-1] * (1 + net_bh))
        cap_td3.append(cap_td3[-1] * (1 + net_td3))
        cap_musa.append(cap_musa[-1] * (1 + net_musa))
        cap_mas.append(cap_mas[-1] * (1 + net_mas))

        ret_bh_list.append(net_bh)
        ret_td3_list.append(net_td3)
        ret_musa_list.append(net_musa)
        ret_mas_list.append(net_mas)

        dates_plot.append(today_date_str)
        macro_state_buffer.append(df_test_macro.loc[i, available_features].values.astype(np.float32))
        last_top_3 = top_3_tickers
        last_exposure = macro_exposure

    avg_exp = np.mean(exposure_history) * 100
    print(f"    -> 📊 [诊断雷达] 深度重训后的 TD3 平均敞口仓位: {avg_exp:.1f}%")

    def get_metrics(rets, caps):
        rets = np.array(rets)
        rf_daily = 0.04 / 252
        sharpe = ((np.mean(rets) - rf_daily) / (np.std(rets) + 1e-9)) * np.sqrt(252)
        mdd = np.min((caps - np.maximum.accumulate(caps)) / np.maximum.accumulate(caps)) * 100
        years = len(rets) / 252.0
        cagr = ((caps[-1] / caps[0]) ** (1 / years) - 1) * 100 if years > 0 else 0.0
        return (caps[-1] / caps[0] - 1) * 100, cagr, sharpe, mdd

    ret_b, cagr_b, sh_b, mdd_b = get_metrics(ret_bh_list, cap_bh)
    ret_t, cagr_t, sh_t, mdd_t = get_metrics(ret_td3_list, cap_td3)
    ret_m, cagr_m, sh_m, mdd_m = get_metrics(ret_musa_list, cap_musa)
    ret_mas, cagr_mas, sh_mas, mdd_mas = get_metrics(ret_mas_list, cap_mas)

    report_text = (
        f"================ {scenario_name.replace('_', ' ')} ================\n"
        f"Period: {start_date} to {end_date} (TD3 Avg Exposure: {avg_exp:.1f}%)\n"
        "--------------------------------------------------------------------------------------------------\n"
        f"{'Strategy Variant':<28} | {'Total Return':<15} | {'CAGR (Annual)':<15} | {'Sharpe (Rf=4%)':<15} | {'Max Drawdown':<15}\n"
        "--------------------------------------------------------------------------------------------------\n"
        f"{'[1] Benchmark (SPY)':<28} | {ret_b:>14.2f}% | {cagr_b:>14.2f}% | {sh_b:>14.2f} | {mdd_b:>14.2f}%\n"
        f"{'[2] Pure TD3 (Risk Only)':<28} | {ret_t:>14.2f}% | {cagr_t:>14.2f}% | {sh_t:>14.2f} | {mdd_t:>14.2f}%\n"
        f"{'[3] Pure MUSA (Alpha Only)':<28} | {ret_m:>14.2f}% | {cagr_m:>14.2f}% | {sh_m:>14.2f} | {mdd_m:>14.2f}%\n"
        f"{'[4] Full MAS Framework':<28} | {ret_mas:>14.2f}% | {cagr_mas:>14.2f}% | {sh_mas:>14.2f} | {mdd_mas:>14.2f}%\n"
        "==================================================================================================\n\n"
    )
    print(report_text)

    with open(os.path.join(results_dir, "MAS_Trading_Report.txt"), "a", encoding="utf-8") as f:
        f.write(report_text)

    plt.figure(figsize=(14, 7), dpi=300)
    plt_dates = pd.to_datetime(dates_plot)
    plt.plot(plt_dates, cap_bh[1:], label=f'Benchmark SPY (CAGR: {cagr_b:.1f}%)', color='gray', linestyle='--',
             alpha=0.8)
    plt.plot(plt_dates, cap_td3[1:], label=f'Pure TD3 Risk (CAGR: {cagr_t:.1f}%)', color='#1f77b4', alpha=0.8)
    plt.plot(plt_dates, cap_musa[1:], label=f'Pure MUSA Alpha (CAGR: {cagr_m:.1f}%)', color='#2ca02c', alpha=0.8)
    plt.plot(plt_dates, cap_mas[1:], label=f'Multi-Agent System (CAGR: {cagr_mas:.1f}%)', color='#d62728',
             linewidth=2.5)

    plt.title(
        f'MAS Collaborative Trading Backtest: {scenario_name.replace("_", " ")} ({start_date[:4]}-{end_date[:4]})',
        fontsize=16, pad=15, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value (USD)', fontsize=12)
    plt.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"MAS_Chart_{scenario_name}.png"))
    plt.close()


def main():
    print("==================================================")
    print("  🏆 唤醒【交易员代理】: 多智能体终极实盘盲测")
    print("==================================================\n")

    results_dir = os.path.join(PROJECT_ROOT, "MAS_Final_Results")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "MAS_Trading_Report.txt")
    if os.path.exists(report_path): os.remove(report_path)

    df_macro = pd.read_csv(MACRO_PATH)
    df_micro = pd.read_csv(MICRO_PATH)
    df_micro['date'] = pd.to_datetime(df_micro['date'])

    agent = TD3(encoder_path=ENCODER_PATH, action_dim=1)
    agent.load(TD3_DIR)

    run_scenario(df_macro, df_micro, agent, '2024-01-01', '2026-12-31', "Bull_Market", results_dir)
    run_scenario(df_macro, df_micro, agent, '2022-01-01', '2022-12-31', "Bear_Market_Crash", results_dir)

    print(f"\n🎉 全剧终！纯粹版 MAS 架构完整测试通过。所有报告与图表已保存至: {results_dir}")


if __name__ == "__main__":
    main()