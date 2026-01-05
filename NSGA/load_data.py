# @Author : CY
# @Time : 2026/1/5 15:46
# ==========================================
# 第一部分：数据加载与预处理 (Data Preprocessing)
# 功能：读取CSV，过滤时间窗口，对齐数据，将行业转为数字编码
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

def load_and_process_data():
    print("【步骤1】正在读取并处理数据...")

    # --- 1. 读取日收益率数据 (源头修复：指定 code 为字符串) ---
    try:
        # dtype={'code': str} 确保 '000001' 读进来还是 '000001'
        df_ret = pd.read_csv("data/stock_returns_daily.csv", dtype={'code': str})
    except FileNotFoundError:
        raise FileNotFoundError("缺失文件: stock_returns_daily.csv")

    df_ret['timestamps'] = pd.to_datetime(df_ret['timestamps'])

    # 过滤时间范围 (2024-12-02 到 2025-12-26)
    start_date = "2022-01-01"
    end_date = "2025-12-26"
    mask = (df_ret['timestamps'] >= start_date) & (df_ret['timestamps'] <= end_date)
    df_ret = df_ret.loc[mask]

    # --- 关键修复：ChangePCT 是百分数(%) -> 小数比率 ---
    df_ret["ChangePCT"] = pd.to_numeric(df_ret["ChangePCT"], errors="coerce") / 100.0

    # --- 学术更稳：极端/不可能值直接置空，不参与当日等权 ---
    # r <= -100% 会导致 (1+r)<=0，净值穿0；属于数据异常/口径异常
    df_ret.loc[df_ret["ChangePCT"] <= -0.999999, "ChangePCT"] = np.nan

    # 转换为矩阵：行=日期，列=股票代码
    df_ret['code'] = df_ret['code'].astype(str)
    returns_matrix = df_ret.pivot(index='timestamps', columns='code', values='ChangePCT')
    returns_matrix = returns_matrix.fillna(0.0)  # 填充缺失数据

    # --- 2. 读取因子得分数据 (源头修复) ---
    try:
        df_scores = pd.read_csv("data/csi300_3factor_scores_winsorized.csv", dtype={'code': str})
    except FileNotFoundError:
        raise FileNotFoundError("缺失文件: csi300_3factor_scores_winsorized.csv")

    df_scores['code'] = df_scores['code'].astype(str)
    df_scores = df_scores.drop_duplicates(subset=['code']).set_index('code')

    # --- 3. 读取行业分类数据 (源头修复) ---
    try:
        df_industry = pd.read_csv("data/csi300_with_industry_2022_2025.csv", dtype={'code': str})
    except FileNotFoundError:
        raise FileNotFoundError("缺失文件: csi300_with_industry_2022_2025.csv")

    df_industry['code'] = df_industry['code'].astype(str)
    df_industry = df_industry.drop_duplicates(subset=['code']).set_index('code')

    # 保存一份代码到名称的映射，方便后续展示
    # (如果有 name 列的话)
    if 'name' in df_industry.columns:
        code_to_name = df_industry['name'].to_dict()
    else:
        code_to_name = {}

    # 4. 数据对齐 (取三者交集)
    valid_codes = returns_matrix.columns
    valid_codes = valid_codes.intersection(df_scores.index)
    valid_codes = valid_codes.intersection(df_industry.index)

    print(f"   - 时间范围: {start_date} 至 {end_date}")
    print(f"   - 有效股票数量: {len(valid_codes)}")

    # 提取最终数据
    returns_matrix = returns_matrix[valid_codes]
    scores_series = df_scores.loc[valid_codes, 'final_score']
    industry_raw = df_industry.loc[valid_codes, 'sw_level1']

    # 将行业名称映射为整数 ID (0, 1, 2...)
    industry_codes, industry_uniques = pd.factorize(industry_raw)
    sectors_series = pd.Series(industry_codes, index=valid_codes)

    print(f"   - 覆盖行业数量: {len(industry_uniques)}")

    return returns_matrix, scores_series, sectors_series, valid_codes, industry_uniques, code_to_name