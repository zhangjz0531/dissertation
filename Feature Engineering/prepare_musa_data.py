import pandas as pd
import os
import glob


def build_musa_panel():
    print("==================================================")
    print("    微观选股数据池 (Micro Panel Data) 组装器")
    print("==================================================\n")

    # 动态获取项目根目录
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stocks_dir = os.path.join(BASE_DIR, "dataload", "stocks")

    # 寻找所有的个股 csv 文件
    all_files = glob.glob(os.path.join(stocks_dir, "*_1d.csv"))
    if not all_files:
        print("[!] 未找到个股文件，请确认 dataload/stocks 目录下是否有数据。")
        return

    df_list = []

    for file in all_files:
        # 提取股票代码，例如从 AAPL_1d.csv 提取出 AAPL
        ticker = os.path.basename(file).split('_')[0]
        df = pd.read_csv(file)

        # 新增一列 'stock' 以区分不同股票
        df['stock'] = ticker

        # 统一时间列名
        if 'start_time' in df.columns:
            df.rename(columns={'start_time': 'date'}, inplace=True)

        # 格式化时间为 YYYY-MM-DD
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df_list.append(df)

    # 将 10 只股票的数据垂直拼接在一起
    panel_df = pd.concat(df_list, ignore_index=True)

    # 🚨 核心逻辑：按日期升序，同日期下按股票代码升序 (这是 MUSA/RL 框架的标准格式)
    panel_df = panel_df.sort_values(by=['date', 'stock']).reset_index(drop=True)

    # 提取 MUSA 截面选股所需的标准列
    cols = ['date', 'stock', 'open', 'high', 'low', 'close', 'volume']
    available_cols = [c for c in cols if c in panel_df.columns]
    final_df = panel_df[available_cols]

    # 保存至终极目录 esg_data
    out_dir = os.path.join(BASE_DIR, "download_data", "esg_data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "MUSA_Top10_Panel.csv")

    final_df.to_csv(out_path, index=False)

    print(f"✅ 微观股票池重组完成！")
    print(f"[*] 共处理股票数: {len(all_files)} 只")
    print(f"[*] 面板数据总行数: {len(final_df)} 行")
    print(f"[*] MUSA 选股底表已保存至: {out_path}")


if __name__ == "__main__":
    build_musa_panel()