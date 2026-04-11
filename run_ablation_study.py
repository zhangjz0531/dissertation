import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
import matplotlib.pyplot as plt
from collections import deque

# 设置路径，导入 TD3 Agent
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rl_folder_path = os.path.join(BASE_DIR, "Reinforcement Learning & Blind Test")
sys.path.append(rl_folder_path)

from td3_agent import TD3

warnings.filterwarnings('ignore')


def run_scenario(df_macro, df_micro, agent, start_date, end_date, scenario_name, ablation_dir):
    print(f"\n[{scenario_name}] 🚀 正在执行 {start_date} 至 {end_date} 的漫游回测...")

    available_features = ['dc_trend', 'dc_event', 'dc_drawdown', 'dc_T', 'dc_TMV', 'dc_R', 'sentiment_score']

    # 截取该剧本的时间段
    df_test_macro = df_macro[(df_macro['date'] >= start_date) & (df_macro['date'] <= end_date)].copy().reset_index(
        drop=True)
    if df_test_macro.empty:
        print(f"[!] 警告：未找到 {scenario_name} 的大盘数据！")
        return

    test_dates = sorted(df_test_macro['date'].unique())

    # 初始化状态 Buffer
    window_size = 30
    macro_state_buffer = deque(maxlen=window_size)
    first_state = df_test_macro.loc[0, available_features].values.astype(np.float32)
    for _ in range(window_size):
        macro_state_buffer.append(first_state)

    # 核心资产追踪列表
    cap_bh = [100000.0];
    ret_bh_list = []
    cap_td3 = [100000.0];
    ret_td3_list = []
    cap_musa = [100000.0];
    ret_musa_list = []
    cap_hier = [100000.0];
    ret_hier_list = []

    dates_plot = []
    last_top_3 = []
    last_exposure = 1.0

    for i in range(1, len(test_dates)):
        today_date_str = test_dates[i]
        yesterday_date_str = test_dates[i - 1]
        today_dt = pd.to_datetime(today_date_str)
        yesterday_dt = pd.to_datetime(yesterday_date_str)

        # --- [宏观风控] ---
        current_state_seq = np.array(macro_state_buffer).astype(np.float32)
        td3_action = agent.select_action(current_state_seq)[0]
        macro_exposure = np.power(td3_action, 2)  # 仓位映射：20% ~ 100%

        # --- [微观选股] ---
        lookback_dt = today_dt - pd.Timedelta(days=90)
        past_data = df_micro[(df_micro['date'] >= lookback_dt) & (df_micro['date'] <= yesterday_dt)]

        def calc_momentum_factor(x):
            if len(x) < 20: return 0.0
            ret = (x.iloc[-1] - x.iloc[0]) / x.iloc[0]
            vol = x.pct_change().std() + 1e-6
            return ret / vol

        if not past_data.empty:
            metrics = past_data.groupby('stock')['close'].apply(calc_momentum_factor)
            top_3_tickers = metrics.nlargest(3).index.tolist()
        else:
            top_3_tickers = last_top_3

        # --- [结算收益] ---
        today_prices = df_micro[df_micro['date'] == today_dt]
        yesterday_prices = df_micro[df_micro['date'] == yesterday_dt]
        merged = pd.merge(today_prices[['stock', 'close']], yesterday_prices[['stock', 'close']], on='stock',
                          suffixes=('_today', '_yest'))

        selected_ret_data = merged[merged['stock'].isin(top_3_tickers)]
        avg_micro_return = selected_ret_data['close_today'].sub(selected_ret_data['close_yest']).div(
            selected_ret_data['close_yest']).mean() if not selected_ret_data.empty else 0.0

        bench_ret = (df_test_macro.loc[i, 'close'] - df_test_macro.loc[i - 1, 'close']) / df_test_macro.loc[
            i - 1, 'close']

        # --- [摩擦成本] ---
        change_count = len(set(top_3_tickers) - set(last_top_3)) if last_top_3 else 3
        stock_turnover_cost = (change_count / 3.0) * 0.0002
        macro_turnover_cost = abs(macro_exposure - last_exposure) * 0.0002

        # --- [资产更新] ---
        net_bh = bench_ret
        net_td3 = (macro_exposure * bench_ret) - macro_turnover_cost
        net_musa = avg_micro_return - stock_turnover_cost
        net_hier = (macro_exposure * avg_micro_return) - stock_turnover_cost - macro_turnover_cost

        cap_bh.append(cap_bh[-1] * (1 + net_bh))
        cap_td3.append(cap_td3[-1] * (1 + net_td3))
        cap_musa.append(cap_musa[-1] * (1 + net_musa))
        cap_hier.append(cap_hier[-1] * (1 + net_hier))

        ret_bh_list.append(net_bh)
        ret_td3_list.append(net_td3)
        ret_musa_list.append(net_musa)
        ret_hier_list.append(net_hier)

        dates_plot.append(today_date_str)
        macro_state_buffer.append(df_test_macro.loc[i, available_features].values.astype(np.float32))
        last_top_3 = top_3_tickers
        last_exposure = macro_exposure

    # --- [学术指标] ---
    def get_metrics(rets, caps):
        rets = np.array(rets)
        rf_daily = 0.04 / 252  # 4% 年化无风险利率
        sharpe = ((np.mean(rets) - rf_daily) / (np.std(rets) + 1e-9)) * np.sqrt(252)
        mdd = np.min((caps - np.maximum.accumulate(caps)) / np.maximum.accumulate(caps)) * 100
        return (caps[-1] / caps[0] - 1) * 100, sharpe, mdd

    ret_b, sh_b, mdd_b = get_metrics(ret_bh_list, cap_bh)
    ret_t, sh_t, mdd_t = get_metrics(ret_td3_list, cap_td3)
    ret_m, sh_m, mdd_m = get_metrics(ret_musa_list, cap_musa)
    ret_h, sh_h, mdd_h = get_metrics(ret_hier_list, cap_hier)

    report_text = (
        f"================ {scenario_name.replace('_', ' ')} ================\n"
        f"Period: {start_date} to {end_date}\n"
        "-------------------------------------------------------------------------------------\n"
        f"{'Strategy Variant':<28} | {'Total Return':<15} | {'Sharpe (Rf=4%)':<15} | {'Max Drawdown':<15}\n"
        "-------------------------------------------------------------------------------------\n"
        f"{'[1] Benchmark (SPY)':<28} | {ret_b:>14.2f}% | {sh_b:>14.2f} | {mdd_b:>14.2f}%\n"
        f"{'[2] Pure TD3 (Macro Only)':<28} | {ret_t:>14.2f}% | {sh_t:>14.2f} | {mdd_t:>14.2f}%\n"
        f"{'[3] Pure MUSA (Micro Only)':<28} | {ret_m:>14.2f}% | {sh_m:>14.2f} | {mdd_m:>14.2f}%\n"
        f"{'[4] Hierarchical (Ours)':<28} | {ret_h:>14.2f}% | {sh_h:>14.2f} | {mdd_h:>14.2f}%\n"
        "=====================================================================================\n\n"
    )
    print(report_text)

    report_path = os.path.join(ablation_dir, "Comprehensive_Metrics_Report.txt")
    with open(report_path, "a", encoding="utf-8") as f:  # 用 "a" 追加模式写入
        f.write(report_text)

    # --- [绘图] ---
    plt.figure(figsize=(14, 7), dpi=300)
    plt_dates = pd.to_datetime(dates_plot)

    plt.plot(plt_dates, cap_bh[1:], label=f'Benchmark SPY (Ret: {ret_b:.1f}%)', color='gray', linestyle='--', alpha=0.8)
    plt.plot(plt_dates, cap_td3[1:], label=f'Pure TD3 (Ret: {ret_t:.1f}%)', color='#1f77b4', alpha=0.8)
    plt.plot(plt_dates, cap_musa[1:], label=f'Pure MUSA (Ret: {ret_m:.1f}%)', color='#2ca02c', alpha=0.8)
    plt.plot(plt_dates, cap_hier[1:], label=f'Hierarchical TD3+MUSA (Ret: {ret_h:.1f}%)', color='#d62728',
             linewidth=2.5)

    title_text = f'Ablation Study: {scenario_name.replace("_", " ")} ({start_date[:4]}-{end_date[:4]})'
    plt.title(title_text, fontsize=16, pad=15, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value (USD)', fontsize=12)
    plt.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)

    save_path = os.path.join(ablation_dir, f"Ablation_Chart_{scenario_name}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 图表保存成功: Ablation_Chart_{scenario_name}.png")


def main():
    print("==================================================")
    print("  🎓 学术级消融实验：双剧本压力测试 (Bull vs Bear)")
    print("==================================================\n")

    ablation_dir = os.path.join(BASE_DIR, "Ablation_Results")
    os.makedirs(ablation_dir, exist_ok=True)

    # 清空之前的旧报告
    report_path = os.path.join(ablation_dir, "Comprehensive_Metrics_Report.txt")
    if os.path.exists(report_path):
        os.remove(report_path)

    # --- 统一加载数据与模型 (节省IO时间) ---
    macro_path = os.path.join(BASE_DIR, "download_data", "esg_data", "SPY_1D_Final.csv")
    df_macro = pd.read_csv(macro_path)
    available_features = ['dc_trend', 'dc_event', 'dc_drawdown', 'dc_T', 'dc_TMV', 'dc_R', 'sentiment_score']

    for feat in available_features:
        df_macro[feat] = pd.to_numeric(df_macro[feat], errors='coerce').fillna(0).astype(np.float32)

    stats = np.load(os.path.join(BASE_DIR, "models", "feature_stats.npz"))
    df_macro[available_features] = (df_macro[available_features] - stats['mean'].astype(np.float32)) / stats[
        'std'].astype(np.float32)

    encoder_path = os.path.join(BASE_DIR, "models", "encoder_pretrained.pth")
    agent = TD3(encoder_path=encoder_path, action_dim=1)
    agent.load(os.path.join(BASE_DIR, "models", "td3_final"))

    micro_path = os.path.join(BASE_DIR, "download_data", "esg_data", "MUSA_Top10_Panel.csv")
    df_micro = pd.read_csv(micro_path)
    df_micro['date'] = pd.to_datetime(df_micro['date'])

    # ==========================================
    # 🧪 剧本 1：2024-2026 单边疯牛市 (验证超额收益)
    # ==========================================
    run_scenario(
        df_macro=df_macro, df_micro=df_micro, agent=agent,
        start_date='2024-01-01', end_date='2026-12-31',
        scenario_name="Bull_Market", ablation_dir=ablation_dir
    )

    # ==========================================
    # 🧪 剧本 2：2022 美联储暴力加息熊市 (验证风控能力)
    # ==========================================
    run_scenario(
        df_macro=df_macro, df_micro=df_micro, agent=agent,
        start_date='2022-01-01', end_date='2022-12-31',
        scenario_name="Bear_Market_Crash", ablation_dir=ablation_dir
    )

    print(f"\n🎉 所有的压力测试已完成！请前往 {ablation_dir} 提取论文材料。")


if __name__ == "__main__":
    main()