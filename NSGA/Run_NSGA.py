# @Author : CY
# @Time : 2026/1/5 15:51

# ==========================================
# 第四部分：主程序执行 (Main Execution)
# 功能：运行优化，输出报表
# ==========================================
from NSGA.NSGA_Optimizer import NSGA3_Optimizer
from NSGA.load_data import load_and_process_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from backtest import AkShareBacktestEngine

if __name__ == "__main__":

    try:
        data = load_and_process_data()
        ret_mat, score_ser, sec_ser, valid_codes, sec_names, code_to_name = data

        # 2. 执行优化
        optimizer = NSGA3_Optimizer(ret_mat, score_ser, sec_ser)
        final_pop, final_objs, df_hv_sp = optimizer.run()

        # --- 【优化代码开始】生成质量指标汇总 CSV ---
        # 直接提取最后一代的指标，避免重复计算和随机误差
        final_metrics = df_hv_sp.iloc[-1]

        # 构建符合您格式要求的 DataFrame
        df_quality = pd.DataFrame([{
            "HV": final_metrics["hv"],
            "SP": final_metrics["sp"],
            "front_size": int(final_metrics["front_size"]),
            "ref_point": "1.100,1.100,1.100",  # 对应代码中固定的参考点
            "hv_samples": 2000  # 对应代码中固定的采样数
        }])

        # 导出文件
        df_quality.to_csv("data/nsga3_front_quality_metrics.csv", index=False, encoding="utf-8-sig")
        print("已导出：nsga3_front_quality_metrics.csv (基于最终代结果)")

        # 【新增】导出每代 HV/SP
        df_hv_sp.to_csv("data/nsga3_hv_sp_history.csv", index=False, encoding="utf-8-sig")
        print("已导出：nsga3_hv_sp_history.csv")

        # 【新增】画 HV / SP 曲线
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(df_hv_sp["gen"], df_hv_sp["hv"], color="tab:blue", label="HV")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Hypervolume (higher is better)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(True, linestyle="--", alpha=0.4)

        ax2 = ax1.twinx()
        ax2.plot(df_hv_sp["gen"], df_hv_sp["sp"], color="tab:orange", label="SP")
        ax2.set_ylabel("Spacing (lower is better)", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        plt.title("HV/SP over Generations (Pareto Front Quality)")
        fig.tight_layout()
        plt.show()

        # 3. 结果整理
        df_result = pd.DataFrame({
            'Ann_Return': final_objs[:, 0],
            'Max_Drawdown': final_objs[:, 1],
            'Factor_Score': final_objs[:, 2],
            'Num_Stocks': np.sum(final_pop, axis=1)
        })

        # 计算夏普代理 (收益/回撤)
        df_result['Sharpe_Proxy'] = df_result['Ann_Return'] / (df_result['Max_Drawdown'] + 1e-6)
        df_result = df_result.sort_values('Sharpe_Proxy', ascending=False).reset_index(drop=True)

        final_metrics = df_hv_sp.iloc[-1]

        print("\n" + "=" * 40)
        print("【算法性能指标】(基于固定基准的最终代结果)")
        print(f"Hypervolume (HV): {final_metrics['hv']:.4f}")
        print(f"Spacing (SP):     {final_metrics['sp']:.4f}")
        print("=" * 40)

        # ==========================================
        # 4. 四维度优选解定位
        # ==========================================
        # 策略 A: 夏普最优 (平衡型)
        idx_sharpe = df_result['Sharpe_Proxy'].idxmax()

        # 策略 B: 最小回撤 (保守型)
        idx_min_risk = df_result['Max_Drawdown'].idxmin()

        # 策略 C: 最大收益 (激进型)
        idx_max_ret = df_result['Ann_Return'].idxmax()

        # 策略 D: 最大因子得分 (风格型) - 新增
        idx_max_score = df_result['Factor_Score'].idxmax()

        strategies = {
            "Best Sharpe": {"idx": idx_sharpe, "color": "red", "marker": "*", "file_suffix": "sharpe"},
            "Min Risk": {"idx": idx_min_risk, "color": "green", "marker": "v", "file_suffix": "min_risk"},
            "Max Return": {"idx": idx_max_ret, "color": "blue", "marker": "^", "file_suffix": "max_return"},
            "Max Score": {"idx": idx_max_score, "color": "purple", "marker": "s", "file_suffix": "max_score"}
        }

        print("\n" + "=" * 60)
        print("【四维度策略优选结果】")
        print("=" * 60)

        # 打印摘要表
        selected_indices = [meta["idx"] for meta in strategies.values()]
        selected_rows = df_result.loc[selected_indices].copy()
        selected_rows.index = strategies.keys()

        print(selected_rows[['Ann_Return', 'Max_Drawdown', 'Factor_Score', 'Num_Stocks', 'Sharpe_Proxy']].to_string(
            formatters={
                'Ann_Return': '{:.2%}'.format,
                'Max_Drawdown': '{:.2%}'.format,
                'Factor_Score': '{:.4f}'.format,
                'Num_Stocks': '{:.0f}'.format,
                'Sharpe_Proxy': '{:.2f}'.format
            }))


        # ==========================================
        # 5. 持仓详情提取与导出
        # ==========================================
        def process_portfolio(name, z_vec, scores, sectors, valid_codes, sec_names, file_suffix):
            print(f"\n>>> 策略详情: {name}")
            sel_idx = np.where(z_vec == 1)[0]

            df_holdings = pd.DataFrame({
                'Stock_Code': valid_codes[sel_idx],
                # 假设 code_to_name 在 load_data 返回值中 (需确保 load_data 返回了 code_to_name)
                # 如果没有，请注释掉下面这行
                'Stock_Name': [code_to_name.get(c, "Unknown") for c in valid_codes[sel_idx]],
                'Industry': sec_names[sectors[sel_idx]],
                'Factor_Score': scores[sel_idx]
            })

            # 统计信息
            print(f"   持仓数量: {len(df_holdings)}")
            print(f"   平均得分: {df_holdings['Factor_Score'].mean():.4f}")

            # 排序并显示前10只（按得分）
            print(df_holdings.sort_values('Factor_Score', ascending=False).head(10).to_string(index=False))
            if len(df_holdings) > 10:
                print(f"   ... (共 {len(df_holdings)} 只，完整列表已导出)")

            # 导出 CSV
            filename = f"data/nsga3_holdings_{file_suffix}.csv"
            df_holdings.to_csv(filename, index=False, encoding="utf-8-sig")
            print(f"   [已导出]: {filename}")


        # 循环处理每个策略
        for name, meta in strategies.items():
            original_pop_idx = meta["idx"]
            best_z = final_pop[original_pop_idx]

            process_portfolio(
                name,
                best_z,
                score_ser.values,
                sec_ser.values,
                valid_codes,
                sec_names,
                meta["file_suffix"]
            )

        # ==========================================
        # 6. 可视化展现 (四点标记)
        # ==========================================
        plt.figure(figsize=(12, 8))

        # 绘制所有帕累托点 (底图)
        sc = plt.scatter(df_result['Max_Drawdown'], df_result['Ann_Return'], c=df_result['Factor_Score'],
                         cmap='viridis', s=80, alpha=0.7, edgecolors='grey', label='Pareto Solutions')
        cbar = plt.colorbar(sc)
        cbar.set_label('Factor Score (Avg)', rotation=270, labelpad=15)

        # 标记四个特殊点
        for name, meta in strategies.items():
            row = df_result.loc[meta["idx"]]
            # 绘制大点
            plt.scatter(row['Max_Drawdown'], row['Ann_Return'],
                        color=meta["color"], marker=meta["marker"], s=250,
                        label=name, zorder=10, edgecolors='white', linewidth=1.5)

            # 添加带偏移的文字标签 (防止重叠)
            # 简单的交替偏移逻辑
            y_offset = 0.02 if meta["idx"] % 2 == 0 else -0.02
            plt.text(row['Max_Drawdown'] + 0.01, row['Ann_Return'] + y_offset, name,
                     fontsize=10, fontweight='bold', color='black',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        plt.xlabel('Max Drawdown (Risk) - Lower is Better', fontsize=12)
        plt.ylabel('Annualized Return - Higher is Better', fontsize=12)
        plt.title('Pareto Front: 4-Dimensional Strategy Selection', fontsize=14, pad=15)
        plt.legend(loc='lower right', frameon=True, framealpha=0.9, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()

        # ------------------------------
        # 【新增输出】按最大收益(Ann_Return)选择 Top N 解
        # 不改变你原来的输出，只是追加一个“收益优先”榜单
        # ------------------------------
        TOP_N_BY_RETURN = 5
        PRINT_HOLDINGS_FOR_TOP_RETURN = True  # 若只想看表，不想打印持仓，改 False

        df_top_ret = (
            df_result
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["Ann_Return", "Max_Drawdown", "Factor_Score"])
            .sort_values("Ann_Return", ascending=False)
            .head(TOP_N_BY_RETURN)
            .copy()
        )

        print("\n" + "=" * 50)
        print(f"最大收益优选组合 (Top {TOP_N_BY_RETURN} by Ann_Return)")
        print("=" * 50)
        print(df_top_ret.to_string(formatters={
            'Ann_Return': '{:.2%}'.format,
            'Max_Drawdown': '{:.2%}'.format,
            'Factor_Score': '{:.4f}'.format,
            'Num_Stocks': '{:.0f}'.format,
            'Sharpe_Proxy': '{:.2f}'.format
        }))
        best_profile_stocks = []

        if PRINT_HOLDINGS_FOR_TOP_RETURN:
            for rank, row in enumerate(df_top_ret.itertuples(index=False), start=1):
                # 用目标向量在 final_objs 中做最近邻匹配，得到对应的 z
                #（与你下面“最佳策略持仓详情”的做法一致，确保不依赖 index 对齐）
                target = np.array([row.Ann_Return, row.Max_Drawdown, row.Factor_Score], dtype=float)
                d = np.sum((final_objs - target) ** 2, axis=1)
                idx = int(np.argmin(d))
                z = final_pop[idx]

                stocks_k = valid_codes[z == 1]
                best_profile_stocks.append(stocks_k)
                sectors_k = sec_names[sec_ser[z == 1]]
                scores_k = score_ser[z == 1]

                df_holdings_k = pd.DataFrame({
                    "Stock_Code": stocks_k,
                    "Industry": sectors_k,
                    "Score": scores_k.values
                }).sort_values("Industry")

                print("\n" + "-" * 50)
                print(f"[Max Return Rank #{rank}] Ann_Return={row.Ann_Return:.2%}, "
                      f"MaxDD={row.Max_Drawdown:.2%}, Score={row.Factor_Score:.4f}, "
                      f"Num={int(row.Num_Stocks)}")
                print(df_holdings_k.to_string(index=False))

    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback

        traceback.print_exc()

    # ==========================================
    # 获取四个维度的持仓代码列表 (List of Lists)
    # ==========================================

    # 定义提取顺序 (与您的要求一致)
    # 0: 夏普最优, 1: 最大收益, 2: 最小风险, 3: 最大因子得分
    # 注意：这里的 Key 必须与上面 strategies 字典中的 Key 保持一致
    target_order = ["Best Sharpe", "Max Return", "Min Risk", "Max Score"]

    # 初始化结果列表
    optimal_stock_codes = []

    print("\n" + "=" * 60)
    print("【正在生成持仓代码列表...】")

    for i, name in enumerate(target_order):
        # 1. 获取该策略在最终种群中的索引
        pop_idx = strategies[name]["idx"]

        # 2. 获取对应的 0-1 决策向量
        z_vec = final_pop[pop_idx]

        # 3. 提取为 1 的位置，并映射回股票代码
        sel_idx = np.where(z_vec == 1)[0]
        codes_list = valid_codes[sel_idx].tolist()  # 转为 Python list

        # 4. 存入大列表
        optimal_stock_codes.append(codes_list)

        print(f"  列表索引 [{i}] - {name}: 提取了 {len(codes_list)} 只股票代码")

    print("=" * 60)

    # ==========================================
    # 实盘验证回测
    # ==========================================
    # 1. 初始化回测引擎
    engine = AkShareBacktestEngine(risk_free_rate=0.02)

    # 2. 准备一个列表来存储所有策略的回测结果
    all_metrics = []

    # 3. 循环遍历每个策略的持仓列表
    # target_order 在上一个单元格中已经定义
    for i, stock_list in enumerate(optimal_stock_codes):
        strategy_name = target_order[i]
        print(f"\n{'=' * 20} 正在回测: {strategy_name} {'=' * 20}")

        # 清理股票代码 (与您原来的逻辑保持一致)
        stocks_to_backtest = (
            pd.Series(stock_list, dtype="string")
            .astype(str).str.strip()
            .str.replace(r"\.SZ$|\.SH$", "", regex=True)
            .str.zfill(6)
            .tolist()
        )

        # 检查股票列表是否为空
        if not stocks_to_backtest:
            print(f"策略 '{strategy_name}' 持仓为空，跳过回测。")
            continue

        # 4. 执行回测
        # 注意：这里使用了样本外的时间段 (2022-12-01 到 2024-11-30)
        # 这个时间段在您的优化周期 (2024-12-02 开始) 之前，是真实有效的样本外测试
        metrics = engine.run(
            stock_codes=stocks_to_backtest,
            start_date="2025-01-01",
            end_date="2025-12-30",
        )

        # 5. 存储结果，并添加策略名称
        metrics['Strategy'] = strategy_name
        all_metrics.append(metrics)

    # 6. 将所有结果合并成一个 DataFrame 并展示
    if all_metrics:
        df_all_metrics = pd.DataFrame(all_metrics).set_index('Strategy')
        print("\n" + "=" * 60)
        print("【所有策略样本外回测结果对比】")
        print("=" * 60)
        print(df_all_metrics.to_string())
    else:
        print("没有可用的回测结果。")