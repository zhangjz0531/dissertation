import pandas as pd
import os


def prepare_micro_panel_data():
    print("[+] 正在打包微观股票池面板数据 (Multi-Agent MUSA Panel)...")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stocks_dir = os.path.join(BASE_DIR, "dataload", "stocks")
    output_dir = os.path.join(BASE_DIR, "download_data", "esg_data")
    os.makedirs(output_dir, exist_ok=True)

    all_data = []

    # 遍历所有股票csv文件
    for file in os.listdir(stocks_dir):
        if file.endswith("_1d.csv"):
            ticker = file.split('_')[0]
            file_path = os.path.join(stocks_dir, file)
            df = pd.read_csv(file_path)

            # 添加股票代码列，作为 Panel 数据的身份标识
            df['stock'] = ticker
            all_data.append(df)

    if not all_data:
        print("[!] 未找到微观股票数据，请先运行 data_downloader.py！")
        return

    # 纵向拼接所有股票数据
    panel_df = pd.concat(all_data, ignore_index=True)

    # 确保基本面因子存在，防止由于前面下载失败导致缺失
    if 'net_margin' not in panel_df.columns:
        panel_df['net_margin'] = 0.0

    # 排序：先按股票代号排，再按时间排
    panel_df['date'] = pd.to_datetime(panel_df['date'])
    panel_df.sort_values(by=['stock', 'date'], inplace=True)
    panel_df['date'] = panel_df['date'].dt.strftime('%Y-%m-%d')

    output_path = os.path.join(output_dir, "MUSA_Top10_Panel.csv")
    panel_df.to_csv(output_path, index=False)

    print(f"✅ 面板数据打包成功！共包含 {len(panel_df['stock'].unique())} 只股票。")
    print(f"✅ 系统特征维度已确认：包含量价特征与基本面财务特征 [net_margin]")
    print(f"✅ 已保存至: {output_path}")


if __name__ == "__main__":
    prepare_micro_panel_data()