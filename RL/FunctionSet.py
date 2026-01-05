# @Author : CY
# @Time : 2026/1/5 16:14
import numpy as np
import pandas as pd
from scipy.stats import linregress


class LongTermOperators:
    """
    专为中长期持仓权重优化设计的算子库
    """

    # --- 基础运算 ---
    @staticmethod
    def add(x, y): return x + y

    @staticmethod
    def sub(x, y): return x - y

    @staticmethod
    def mul(x, y): return x * y

    @staticmethod
    def div(x, y): return x / (y + 1e-8)  # 防止除零

    # --- 改进的时序算子 (Long Windows) ---
    @staticmethod
    def ts_mean(x, d):
        """简单移动平均，窗口 d 建议取 60, 120, 250"""
        return x.rolling(window=d, min_periods=d // 2).mean()

    @staticmethod
    def ts_ema(x, d):
        """指数移动平均，对近期反应更灵敏"""
        return x.ewm(span=d, adjust=False).mean()

    @staticmethod
    def ts_momentum(x, d):
        """长期动量：(现价 / d天前价格) - 1"""
        return x / x.shift(d) - 1

    # --- 新增：风险类算子 ---
    @staticmethod
    def ts_downside_std(x, d):
        """
        下行波动率：衡量下行风险
        x: 通常输入为日收益率序列
        """
        # 将正收益置为0，只计算负收益的波动
        negative_ret = x.clip(upper=0)
        return negative_ret.rolling(window=d, min_periods=d // 2).std()

    @staticmethod
    def ts_max_drawdown(x, d):
        """
        滚动最大回撤
        x: 价格序列
        """
        roll_max = x.rolling(window=d, min_periods=d // 2).max()
        drawdown = (x - roll_max) / roll_max
        # 我们通常希望最大回撤越小越好，所以返回 drawdown (是负数)，
        # 或者返回 abs(min(drawdown))
        return drawdown.rolling(window=d, min_periods=d // 2).min()

    # --- 新增：趋势质量算子 ---
    @staticmethod
    def ts_trend_strength(x, d):
        """
        计算长期趋势的强度 (Sharpe Proxy of Price)
        Mean(Returns) / Std(Returns) * sqrt(252)
        """
        ret = x.pct_change()
        mu = ret.rolling(window=d).mean()
        sigma = ret.rolling(window=d).std()
        return (mu / (sigma + 1e-8)) * np.sqrt(252)

    # --- 改进的截面算子 ---
    @staticmethod
    def cross_zscore(x):
        """
        截面 Z-Score 标准化
        x: (Time, Stocks) 矩阵
        axis=1 表示在股票维度操作
        """
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        return (x - mean) / (std + 1e-8)

    @staticmethod
    def ind_neutralize(x, ind_matrix):
        """
        行业中性化 (简化版)
        x: 因子值矩阵
        ind_matrix: 行业 Dummy 矩阵 (Stocks, Industries)
        """
        # 实际代码通常用线性回归取残差，这里用减去行业均值模拟
        # 假设 x 是 DataFrame
        # 这一步通常在 Alpha 解析器外层处理，因为涉及额外数据输入
        pass