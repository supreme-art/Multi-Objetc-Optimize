import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class AkShareBacktestEngine:
    def __init__(self, risk_free_rate=0.02):
        self.risk_free_rate = risk_free_rate

    def _format_date(self, date_str):
        return date_str.replace("-", "")

    def get_data(self, stock_codes, start_date, end_date):
        """
        获取数据 (保持不变)
        """
        s_date = self._format_date(start_date)
        e_date = self._format_date(end_date)
        all_data = {}
        print(f"正在获取 {len(stock_codes)} 只股票数据...")

        for i, code in enumerate(stock_codes):
            clean_code = code.split(".")[0]
            try:
                df = ak.stock_zh_a_hist(symbol=clean_code, period="daily", start_date=s_date, end_date=e_date, adjust="qfq")
                if not df.empty:
                    df['日期'] = pd.to_datetime(df['日期'])
                    df.set_index('日期', inplace=True)
                    all_data[code] = df['收盘']
                    print(f"[{i+1}/{len(stock_codes)}] {code} 成功")
            except Exception as e:
                print(f"[{i+1}/{len(stock_codes)}] {code} 失败: {e}")

        if not all_data: return pd.DataFrame(), pd.Series()

        # 数据清洗
        df_stocks = pd.DataFrame(all_data).ffill()

        # 获取基准
        try:
            df_bench = ak.index_zh_a_hist(symbol="000300", period="daily", start_date=s_date, end_date=e_date)
            df_bench['日期'] = pd.to_datetime(df_bench['日期'])
            df_bench.set_index('日期', inplace=True)
            bench_series = df_bench['收盘']
        except:
            bench_series = pd.Series(1, index=df_stocks.index)

        # 对齐
        common_index = df_stocks.index.intersection(bench_series.index)
        if common_index.empty: common_index = df_stocks.index

        return df_stocks.loc[common_index], bench_series.reindex(common_index).ffill()

    def calculate_metrics(self, nav_series):
        """
        【关键修改】计算修复日期 (Recovery Date)
        """
        total_ret = nav_series.iloc[-1] - 1
        days = (nav_series.index[-1] - nav_series.index[0]).days
        ann_ret = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0

        daily_ret = nav_series.pct_change().dropna()
        ann_vol = daily_ret.std() * np.sqrt(250)
        sharpe = (ann_ret - self.risk_free_rate) / ann_vol if ann_vol != 0 else 0

        # --- 最大回撤及修复时间计算 ---
        peak_series = nav_series.cummax()          # 历史峰值序列
        drawdown = (peak_series - nav_series) / peak_series
        max_dd = drawdown.max()

        # 1. 定位谷底 (Valley)
        valley_date = drawdown.idxmax()

        # 2. 定位峰值 (Peak, 谷底之前最后一次创行高)
        peak_val = peak_series.loc[valley_date]
        # 找到等于该峰值的最后一天（即下跌开始前的那一天）
        start_date = nav_series.loc[:valley_date][nav_series == peak_val].index[-1]

        # 3. 定位修复日 (Recovery, 谷底之后第一次回到峰值)
        # 截取谷底之后的数据
        post_valley_data = nav_series.loc[valley_date:]
        # 找到第一个 >= 峰值的日期
        recovery_mask = post_valley_data >= peak_val

        if recovery_mask.any():
            recovery_date = recovery_mask.idxmax()
            is_recovered = True
            recovery_days = (recovery_date - valley_date).days # 爬坑耗时
        else:
            recovery_date = None
            is_recovered = False
            recovery_days = (nav_series.index[-1] - valley_date).days # 至今未修复天数

        return {
            'Total Return': total_ret,
            'Annual Return': ann_ret,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd,
            'DD Start Date': start_date,    # 峰值日
            'DD Valley Date': valley_date,  # 谷底日
            'DD Recovery Date': recovery_date, # 修复日 (可能为None)
            'Is Recovered': is_recovered,
            'Recovery Days': recovery_days
        }

    def run(self, stock_codes, start_date, end_date):
        df_stocks, df_bench = self.get_data(stock_codes, start_date, end_date)
        if df_stocks.empty: return

        portfolio_ret = df_stocks.pct_change().fillna(0).mean(axis=1)
        portfolio_nav = (1 + portfolio_ret).cumprod()
        benchmark_nav = df_bench / df_bench.iloc[0]

        metrics = self.calculate_metrics(portfolio_nav)
        self._plot_results(portfolio_nav, benchmark_nav, metrics)
        return metrics

    def _plot_results(self, port_nav, bench_nav, metrics):
        """
        【关键修改】可视化逻辑：区分下跌段(绿)和修复段(金)
        """
        fig, ax = plt.subplots(figsize=(12, 7))

        # 1. 基础曲线
        ax.plot(port_nav.index, port_nav.values, label='Portfolio', color='#d62728', linewidth=2, zorder=2)
        ax.plot(bench_nav.index, bench_nav.values, label='Benchmark (CSI 300)', color='gray', linestyle='--', alpha=0.6, zorder=1)

        # 2. 超额收益区域
        ax.fill_between(port_nav.index, port_nav.values, bench_nav.values,
                         where=(port_nav.values >= bench_nav.values),
                         interpolate=True, color='red', alpha=0.05)
        ax.fill_between(port_nav.index, port_nav.values, bench_nav.values,
                         where=(port_nav.values < bench_nav.values),
                         interpolate=True, color='green', alpha=0.05)

        # --- 3. 最大回撤区间可视化 (三段式) ---
        s_date = metrics['DD Start Date']   # Peak
        v_date = metrics['DD Valley Date']  # Valley
        r_date = metrics['DD Recovery Date']# Recovery

        # A. 下跌阶段 (Peak -> Valley): 绿色
        ax.axvspan(s_date, v_date, color='green', alpha=0.15, label='Drawdown Phase')

        # B. 修复阶段 (Valley -> Recovery): 金色
        if metrics['Is Recovered']:
            ax.axvspan(v_date, r_date, color='gold', alpha=0.2, label='Recovery Phase')
            end_marker_date = r_date
            end_marker_val = port_nav.loc[r_date]
            rec_text = f"Recovered in {metrics['Recovery Days']} days"
        else:
            # 如果没修复，画到最后一天，并标记为红色
            ax.axvspan(v_date, port_nav.index[-1], color='gray', alpha=0.1, label='Not Recovered Yet')
            end_marker_date = port_nav.index[-1]
            end_marker_val = port_nav.iloc[-1]
            rec_text = f"Not Recovered ({metrics['Recovery Days']}+ days)"

        # C. 标记关键点 (峰值、谷底、修复点)
        s_val = port_nav.loc[s_date]
        v_val = port_nav.loc[v_date]

        # 画点
        ax.scatter([s_date, v_date, end_marker_date], [s_val, v_val, end_marker_val],
                   c=['green', 'red', 'gold'], s=50, zorder=3, edgecolors='k')

        # 画虚线连接
        ax.plot([s_date, v_date, end_marker_date], [s_val, v_val, end_marker_val],
                color='green', linestyle=':', linewidth=1)

        # D. 文字标注 (箭头指向谷底)
        mid_date = v_date # 标注位置放在谷底附近
        dd_percent = metrics['Max Drawdown']

        # 动态生成标注文本
        anno_text = (f"Max DD: -{dd_percent:.2%}\n"
                     f"Fall: {(v_date - s_date).days} days\n"
                     f"Repair: {rec_text}")

        ax.annotate(anno_text,
                    xy=(v_date, v_val),
                    xytext=(v_date, v_val - 0.08), # 文字向下偏移
                    arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.6),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    ha='center', fontsize=9)

        # 标题与格式
        title_str = (f"Backtest Performance\n"
                     f"Sharpe: {metrics['Sharpe Ratio']:.2f} | "
                     f"Max DD: {metrics['Max Drawdown']:.2%} | "
                     f"Ann Ret: {metrics['Annual Return']:.2%}")

        ax.set_title(title_str, fontsize=12, pad=15)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left', fontsize=9)

        plt.tight_layout()
        plt.show()

# ===========================
# 运行示例
# ===========================
if __name__ == "__main__":
    # 示例: 选几个波动大的股票更容易看到回撤和修复效果
    # 600519(茅台), 300750(宁德), 601888(中免)
    my_stocks = ['600519', '300750', '601888']

    engine = AkShareBacktestEngine()
    print("开始回测...")
    # 稍微拉长一点时间窗口，确保有完整的跌-涨周期
    engine.run(my_stocks, "2024-01-01", "2025-10-01")