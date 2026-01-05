# @Author: CY
# @Time : 2026/1/5 14:57
# ==========================================
# 获取全A股的财务数据-akshare接口
# 用来进行五因子数据的计算
# ==========================================
import akshare as ak
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import traceback

# ================= 配置 =================
# 我们需要 2022-2025 的因子，因此财务数据需要从 2021 开始 (用于计算 2022 的增长率)
START_YEAR = 2021
END_YEAR = 2025
OUTPUT_DIR = "data"
OUTPUT_FILE = f"{OUTPUT_DIR}/all_a_stocks_financial_ff5_ak.csv"


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def fetch_financial_data_robust():
    print("=== 开始获取全A股财务数据 (AkShare/东财 报告期全市场) ===")
    print(f"时间范围: {START_YEAR} - {END_YEAR}")

    # raw 文件：抓取阶段实时追加写入，避免中断丢失
    RAW_FILE = f"{OUTPUT_DIR}/all_a_stocks_financial_ff5_raw_ak.csv"

    # 你如果希望“每次运行都重新生成”，就删掉旧文件
    if os.path.exists(RAW_FILE):
        os.remove(RAW_FILE)
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    raw_header_written = False

    def _call_ak(possible_names, **kwargs):
        """
        更鲁棒地调用 akshare 接口：
        - 优先尝试原 kwargs；
        - 尝试常见日期参数名（date, report_date, reportDate, rpt, end_date 等）；
        - 尝试位置参数（如 fn(rpt)）；
        - 尝试 YYYY-MM-DD 格式、int 格式、无参调用；
        - 记录并打印最后错误栈，方便定位。
        """
        import inspect

        last_err = None
        param_names = ["date", "report_date", "reportDate", "rpt", "trade_date", "period", "end_date"]
        # 从 kwargs 提取第一个值作为 rpt 值（兼容现有调用 style: date=rpt）
        rpt_val = None
        if kwargs:
            try:
                rpt_val = next(iter(kwargs.values()))
            except StopIteration:
                rpt_val = None

        # 生成候选日期表示
        date_candidates = []
        if rpt_val is not None:
            s = str(rpt_val)
            date_candidates.append(s)
            # 如果是像 20221231 的纯数字，尝试带短横的形式和 int
            if len(s) == 8 and s.isdigit():
                date_candidates.append(f"{s[:4]}-{s[4:6]}-{s[6:]}")
                try:
                    date_candidates.append(int(s))
                except Exception:
                    pass

        for name in possible_names:
            fn = getattr(ak, name, None)
            if fn is None:
                continue

            # 1) 直接用原 kwargs（保持向后兼容）
            try:
                return fn(**kwargs)
            except Exception as e:
                last_err = e

            # 2) 尝试常见关键字名传递 rpt_val
            if rpt_val is not None:
                for p in param_names:
                    try:
                        return fn(**{p: rpt_val})
                    except Exception as e:
                        last_err = e
                        continue

            # 3) 尝试位置参数调用（各种日期格式）
            for v in date_candidates:
                try:
                    return fn(v)
                except Exception as e:
                    last_err = e
                    continue

            # 4) 尝试不带参数调用（某些版本接口可能无参数）
            try:
                return fn()
            except Exception as e:
                last_err = e

            # 5) 如果函数签名可读，尝试按签名构造（如只接受一个位置参数）
            try:
                sig = inspect.signature(fn)
                params = sig.parameters
                # 如果函数只接受一个非-kwargs 参数，尝试把 rpt_val 以位置方式传入
                non_default_params = [p for p in params.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
                if len(non_default_params) == 1 and rpt_val is not None:
                    try:
                        return fn(rpt_val)
                    except Exception as e:
                        last_err = e
            except Exception:
                pass

        # 全部尝试失败，打印最后错误栈便于排查
        if last_err is not None:
            traceback.print_exception(type(last_err), last_err, last_err.__traceback__)
        raise RuntimeError(f"AkShare 接口调用失败，尝试过: {possible_names}, last_err={repr(last_err)}")

    def _pick_col(df: pd.DataFrame, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # 生成报告期列表（季度末）
    quarter_ends = {1: "0331", 2: "0630", 3: "0930", 4: "1231"}
    report_dates = []
    for y in range(START_YEAR, END_YEAR + 1):
        for q in (1, 2, 3, 4):
            report_dates.append(f"{y}{quarter_ends[q]}")

    # 逐报告期抓取（每个报告期 2 次请求：利润表 + 资产负债表）
    for rpt in tqdm(report_dates, desc="By report_date"):
        try:
            # 利润表（全市场，按报告期）
            # 新旧命名兼容：stock_lrb_em / stock_em_lrb
            df_lrb = _call_ak(
                ["stock_lrb_em", "stock_em_lrb", "stock_profit_sheet_by_report_em"],
                date=rpt
            )

            # 资产负债表（全市场，按报告期）
            # 新旧命名兼容：stock_zcfz_em / stock_em_zcfz
            df_zcfz = _call_ak(
                ["stock_zcfz_em", "stock_em_zcfz", "stock_balance_sheet_by_report_em"],
                date=rpt
            )

            if df_lrb is None or df_lrb.empty or df_zcfz is None or df_zcfz.empty:
                continue

            # --- 识别关键列（不同版本字段可能略有差异，做候选兼容） ---
            code_col_lrb = _pick_col(df_lrb, ["股票代码", "代码", "证券代码"])
            pub_col_lrb  = _pick_col(df_lrb, ["公告日期", "公告日", "披露日期"])
            net_col      = _pick_col(df_lrb, ["净利润", "净利润(元)", "归母净利润", "归属于母公司所有者的净利润"])

            code_col_zcfz = _pick_col(df_zcfz, ["股票代码", "代码", "证券代码"])
            pub_col_zcfz  = _pick_col(df_zcfz, ["公告日期", "公告日", "披露日期"])
            assets_col    = _pick_col(df_zcfz, ["资产-总资产", "总资产", "资产总计", "资产总额"])
            equity_col    = _pick_col(df_zcfz, ["股东权益合计", "所有者权益合计", "股东权益(或所有者权益)合计", "归属于母公司股东权益合计"])

            # 如果关键列缺失，跳过该报告期（不影响整体）
            if not (code_col_lrb and pub_col_lrb and net_col and code_col_zcfz and assets_col and equity_col):
                continue

            # --- 裁剪并规范字段 ---
            lrb_sub = df_lrb[[code_col_lrb, pub_col_lrb, net_col]].copy()
            lrb_sub.rename(columns={
                code_col_lrb: "code",
                pub_col_lrb: "publish_date_lrb",
                net_col: "NetProfit"
            }, inplace=True)

            zcfz_cols = [code_col_zcfz, assets_col, equity_col]
            if pub_col_zcfz:
                zcfz_cols.append(pub_col_zcfz)

            zcfz_sub = df_zcfz[zcfz_cols].copy()
            rename_map = {
                code_col_zcfz: "code",
                assets_col: "TotalAssets",
                equity_col: "TotalEquity",
            }
            if pub_col_zcfz:
                rename_map[pub_col_zcfz] = "publish_date_zcfz"
            zcfz_sub.rename(columns=rename_map, inplace=True)

            # 过滤出 A 股常见 6 位代码（0/3/6 开头）
            lrb_sub["code"] = lrb_sub["code"].astype(str).str.strip()
            zcfz_sub["code"] = zcfz_sub["code"].astype(str).str.strip()

            # 兼容带后缀（如 600519.SH 或 600519.SZ）的代码，提取最后的 6 位数字
            lrb_sub["code"] = lrb_sub["code"].str.extract(r'(\d{6})$', expand=False)
            zcfz_sub["code"] = zcfz_sub["code"].str.extract(r'(\d{6})$', expand=False)

            # 丢弃无法提取的行
            lrb_sub = lrb_sub.dropna(subset=["code"])
            zcfz_sub = zcfz_sub.dropna(subset=["code"])

            # 只保留以 0/3/6 开头的 A 股（字符串形式）
            lrb_sub = lrb_sub[lrb_sub["code"].str.startswith(("0", "3", "6"))]
            zcfz_sub = zcfz_sub[zcfz_sub["code"].str.startswith(("0", "3", "6"))]

            # 合并：优先用利润表的公告日期，没有就用资产负债表的
            merged = pd.merge(lrb_sub, zcfz_sub, on="code", how="inner")
            if "publish_date_zcfz" in merged.columns:
                merged["publish_date"] = merged["publish_date_lrb"].combine_first(merged["publish_date_zcfz"])
            else:
                merged["publish_date"] = merged["publish_date_lrb"]

            merged["report_date"] = rpt

            # 保持你原脚本相同的输出列
            out = merged[["code", "publish_date", "report_date", "NetProfit", "TotalAssets", "TotalEquity"]].copy()

            # 追加写入 raw（避免中断丢数据）
            out.to_csv(
                RAW_FILE,
                mode="a",
                index=False,
                header=(not raw_header_written),
            )
            raw_header_written = True

        except Exception as e:
            # 单个报告期失败不影响整体
            print(f"[ERROR] rpt={rpt} 失败: {e}")
            traceback.print_exc()

            continue

    # -------- 最终清洗与输出（保持你原逻辑/格式）--------
    if not os.path.exists(RAW_FILE):
        print("未获取到有效数据（raw 文件不存在），请检查网络或接口是否可用。")
        return

    df_fin = pd.read_csv(RAW_FILE)

    cols_to_float = ["NetProfit", "TotalAssets", "TotalEquity"]
    for col in cols_to_float:
        df_fin[col] = pd.to_numeric(df_fin[col], errors="coerce")
    df_fin.dropna(subset=cols_to_float, inplace=True)

    df_fin["publish_date"] = pd.to_datetime(df_fin["publish_date"], errors="coerce")
    df_fin["report_date"] = pd.to_datetime(df_fin["report_date"], errors="coerce")
    df_fin.dropna(subset=["publish_date", "report_date"], inplace=True)

    df_fin.drop_duplicates(subset=["code", "publish_date", "report_date"], keep="last", inplace=True)

    df_fin.to_csv(OUTPUT_FILE, index=False)
    print(f"文件已保存至: {OUTPUT_FILE}")
    print("前5行预览:")
    print(df_fin.head())

if __name__ == "__main__":
    fetch_financial_data_robust()