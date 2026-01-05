# @Author : CY
# @Time : 2026/1/5 16:15
from enum import Enum


class TokenType(Enum):
    BEG = 0  # 开始符号
    SEP = 1  # 结束符号 (Separator)
    FEATURE = 2  # 基础量价特征 (Open, Close...)
    CONSTANT = 3  # 常数 (包括时间窗口参数 60, 250 等)
    OPERATOR = 4  # 算子 (Add, TS_Mean...)

    # 原文可能有 TimeDelta，但在长周期中，建议将其归类为 CONSTANT 以简化处理