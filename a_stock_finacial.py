# @Author: CY
# @Time : 2026/1/5 11:58
# ==========================================
# 获取全A股的财务报表数据
# 用来进行五因子数据的计算
# ==========================================

import baostock as bs
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm



# ================= 配置 =================
# 我们需要 2022-2025 的因子，因此财务数据需要从 2021 开始 (用于计算 2022 的增长率)
START_YEAR = 2021
END_YEAR = 2025
OUTPUT_DIR = "data"
OUTPUT_FILE = f"{OUTPUT_DIR}/all_a_stocks_financial_ff5.csv"
RAW_FILE = f"{OUTPUT_DIR}/all_a_stocks_financial_ff5_raw.csv"


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def fetch_financial_data_robust():
    if os.path.exists(RAW_FILE):
        os.remove(RAW_FILE)
    raw_header_written = False

    print("=== 开始获取全A股财务数据 (Baostock) ===")

    # 1. 登录 Baostock
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录失败: {lg.error_msg}")
        return

    # 2. 获取全市场股票列表
    # 使用最近的一个交易日来获取当前上市的所有股票
    print("正在获取股票列表...")
    rs = bs.query_all_stock(day="2025-12-31")

    stock_list = []
    while (rs.error_code == '0') & rs.next():
        stock_list.append(rs.get_row_data())

    df_stocks = pd.DataFrame(stock_list, columns=rs.fields)
    # 过滤指数和空代码，只保留股票 (sh.6... 或 sz.0.../sz.3...)
    df_stocks = df_stocks[df_stocks['code'].str.startswith(('sh.6', 'sz.0', 'sz.3'))]
    all_codes = df_stocks['code'].tolist()

    print(f"共获取到 {len(all_codes)} 只A股股票，准备下载财务报表...")
    print(f"时间范围: {START_YEAR} - {END_YEAR}")

    # 3. 准备时间遍历 (Year, Quarter)
    # 生成需要查询的 (年份, 季度) 列表
    periods = []
    for year in range(START_YEAR, END_YEAR + 1):
        for q in [1, 2, 3, 4]:
            # 跳过未来的时间 (假设当前时间是 2026年初)
            if year == 2026 and q > 1: continue
            periods.append((year, q))

    all_financial_records = []

    # 4. 循环获取 (这是最耗时的部分)
    # 进度条显示
    pbar = tqdm(all_codes, desc="Downloading")

    try:
        for code in pbar:
            for year, quarter in periods:
                try:
                    # --- A. 获取盈利能力 (Profit) -> NetProfit (RMW) ---
                    # 字段: code, pubDate, statDate, netProfit, ...
                    rs_profit = bs.query_profit_data(code=code, year=year, quarter=quarter)

                    # --- B. 获取资产负债 (Balance) -> TotalAssets (CMA), TotalEquity (HML) ---
                    # 字段: code, pubDate, statDate, totalAssets, totalShareholderEquity, ...
                    rs_balance = bs.query_balance_data(code=code, year=year, quarter=quarter)

                    # 解析 Profit
                    profit_data = {}
                    while (rs_profit.error_code == '0') & rs_profit.next():
                        row = rs_profit.get_row_data()
                        # 映射列名与值
                        data_dict = dict(zip(rs_profit.fields, row))
                        profit_data = data_dict

                    # 解析 Balance
                    balance_data = {}
                    while (rs_balance.error_code == '0') & rs_balance.next():
                        row = rs_balance.get_row_data()
                        data_dict = dict(zip(rs_balance.fields, row))
                        balance_data = data_dict

                    # 如果两者都有数据，合并
                    if profit_data and balance_data:
                        # 校验：必须有公告日期 (pubDate)，否则无法对齐防止未来函数
                        pub_date = profit_data.get('pubDate', '')
                        if pub_date == '':
                            pub_date = balance_data.get('pubDate', '')

                        if pub_date == '': continue # 丢弃无公告日期的废数据

                        record = {
                            'code': code.split('.')[-1], # 去掉 sh./sz. 前缀，保持与 AkShare 统一
                            'publish_date': pub_date,
                            'report_date': profit_data.get('statDate'),
                            'NetProfit': profit_data.get('netProfit'),
                            'TotalAssets': balance_data.get('totalAssets'),
                            'TotalEquity': balance_data.get('totalShareholderEquity')
                        }
                        # 追加落盘（避免中断丢数据）
                        pd.DataFrame([record]).to_csv(
                            RAW_FILE,
                            mode="a",
                            index=False,
                            header=(not raw_header_written),
                            encoding="utf-8-sig"
                        )
                        raw_header_written = True
                        all_financial_records.append(record)

                except Exception as e:
                    # 某只股票某个季度失败不影响整体
                    continue
    finally:
        bs.logout()
    print(f"\n下载完成，共收集到 {len(all_financial_records)} 条财务记录。")

    # 5. 数据清洗与保存
    if os.path.exists(RAW_FILE):
        df_fin = pd.read_csv(RAW_FILE)

        # 类型转换 (Baostock 返回的都是字符串)
        cols_to_float = ['NetProfit', 'TotalAssets', 'TotalEquity']
        for col in cols_to_float:
            df_fin[col] = pd.to_numeric(df_fin[col], errors='coerce')

        # 去除无效数据 (例如资产为0或空)
        df_fin.dropna(subset=cols_to_float, inplace=True)

        # 转换日期格式
        df_fin['publish_date'] = pd.to_datetime(df_fin['publish_date'])
        df_fin['report_date'] = pd.to_datetime(df_fin['report_date'])

        # 去重：同一只股票同一个发布日期的重复数据 (通常保留最后一条)
        df_fin.drop_duplicates(subset=['code', 'publish_date', 'report_date'], keep='last', inplace=True)

        # 保存
        df_fin.to_csv(OUTPUT_FILE, index=False)
        print(f"文件已保存至: {OUTPUT_FILE}")
        print("前5行预览:")
        print(df_fin.head())
    else:
        print("未获取到有效数据，请检查网络或时间范围。")
        print("raw 文件不存在：可能运行中途还没写入任何记录。")

if __name__ == "__main__":
    fetch_financial_data_robust()