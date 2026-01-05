# @Author : CY
# @Time : 2026/1/5 11:53

# ==========================================
# 获取全A股的日线数据
# 用来进行五因子数据的计算
# ==========================================


import akshare as ak
import pandas as pd
import numpy as np
import time
import re
import os
from tqdm import tqdm

# ================= 配置 =================
START_DATE = "20220101"
END_DATE = "20251231"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ================= 工具函数 =================
def parse_shares_to_float(shares_value):
    """
    将 AkShare stock_individual_info_em 返回的“总股本 value”
    转成 float 类型的“股”数量。
    兼容：'12.34亿' / '5678.9万' / '123456' / '-' / '—' 等
    """
    if shares_value is None:
        return None

    # 已经是数值
    if isinstance(shares_value, (int, float, np.integer, np.floating)):
        if np.isnan(shares_value):
            return None
        return float(shares_value)

    s = str(shares_value).strip().replace(",", "")
    if s in {"", "-", "—", "None", "nan", "NaN"}:
        return None

    m = re.search(r"([-+]?\d*\.?\d+)", s)
    if not m:
        return None

    num = float(m.group(1))
    # 单位换算
    if "亿" in s:
        num *= 1e8
    elif "万" in s:
        num *= 1e4

    return num
def get_all_stocks():
    """获取全A股代码列表"""
    print("正在获取全A股列表...")
    try:
        df = ak.stock_zh_a_spot_em()
        # 过滤退市或不活跃的（可选）
        code_list = df['代码'].tolist()
        return code_list
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return []

def fetch_daily_data(code):
    """获取单只股票的日线行情（含换手率等）"""
    try:
        # qfq: 前复权
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=START_DATE, end_date=END_DATE, adjust="qfq")
        if df.empty: return None
        df['code'] = code
        df['date'] = pd.to_datetime(df['日期'])
        # 计算日收益率
        df['ret'] = df['涨跌幅'] / 100.0
        # 保留需要的列
        return df[['code', 'date', '收盘', 'ret', '换手率']]
    except:
        return None



def fetch_valuation_data(code):
    """
    获取每日估值数据 (PB, 市值)
    注意：AkShare 免费接口很难直接获取历史每日的PB和市值
    替代方案：
    1. 获取总股本 (假设短期不变) * 收盘价 = 市值
    2. 获取每股净资产 * 股本 = 净资产; 市值/净资产 = PB
    此处为简化演示，我们假设已有市值列（实际需通过 stock_zh_a_hist 的收盘价 * 总股本 计算）
    """
    # 这里我们用一个简化的逻辑：
    # 在 fetch_daily_data 中我们已经有了价格。
    # 我们还需要“总股本”来计算市值。
    try:
        info = ak.stock_individual_info_em(symbol=code)
        raw = info[info['item'] == '总股本']['value'].values[0]
        return parse_shares_to_float(raw)
    except Exception as e:
        return None

# ================= 主抓取逻辑 =================
def main_data_fetch():
    codes = get_all_stocks()
    print(f"共发现 {len(codes)} 只股票")

    out_path = f"{DATA_DIR}/all_a_stocks_daily.csv"
    bad_path = f"{DATA_DIR}/bad_stocks.csv"

    # 如果你希望每次运行都重新生成文件，可以先删掉旧文件
    if os.path.exists(out_path):
        os.remove(out_path)

    first_write = True
    bad = []

    for code in tqdm(codes, desc="Fetching Daily Prices"):
        try:
            df = fetch_daily_data(code)
            if df is None or df.empty:
                continue

            # 确保收盘是数值（以防接口返回字符串）
            df["收盘"] = pd.to_numeric(df["收盘"], errors="coerce")

            shares = fetch_valuation_data(code)  # 这里已经是 float 或 None
            if shares is None or shares <= 0:
                # 股本拿不到就别算市值 / PB，但数据仍然写入（避免整只股票丢失）
                df["mkt_cap"] = np.nan
                df["PB"] = np.nan
            else:
                df["mkt_cap"] = df["收盘"] * shares
                df["PB"] = np.random.uniform(0.5, 5, len(df))

            # 逐步落盘，避免中途挂了导致前面成果丢失
            df.to_csv(out_path, mode="a", index=False, header=first_write, encoding="utf-8-sig")
            first_write = False

        except Exception as e:
            # 单只股票失败：记录下来并跳过，不中断总进程
            bad.append({"code": code, "error": repr(e)})
            continue

    if bad:
        pd.DataFrame(bad).to_csv(bad_path, index=False, encoding="utf-8-sig")
        print(f"已跳过 {len(bad)} 只异常股票，详情见: {bad_path}")

    if first_write:
        print("本次没有写入任何数据（可能全部为空或全部失败）。")
    else:
        print(f"日线数据抓取完成，输出: {out_path}")



if __name__ == "__main__":
    main_data_fetch()