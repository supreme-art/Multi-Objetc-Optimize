# @Author : CY
# @Time : 2026/1/5 16:16
from RL.TokenType import TokenType


class TokenVocabulary:
    def __init__(self):
        self.idx_to_token = {}
        self.token_to_idx = {}
        self.token_types = {}  # 记录每个Token的类型
        self.unary_ops = set()  # 一元算子集合
        self.binary_ops = set()  # 二元算子集合
        self.rolling_ops = set()  # 滚动算子 (需要时间窗口参数)

        self._build_vocab()

    def _build_vocab(self):
        # 1. 特殊符号
        self._add_token('<BEG>', TokenType.BEG)
        self._add_token('<SEP>', TokenType.SEP)

        # 2. 基础特征 (针对NSGA选出的股票)
        features = ['open', 'high', 'low', 'close', 'volume', 'vwap']
        for f in features:
            self._add_token(f, TokenType.FEATURE)

        # 3. 常数 (关键改进：为了适应长周期，剔除短周期，增加年化周期)
        # 10, 30 -> 剔除; 60(季), 120(半年), 250(年) -> 新增
        constants = [
            '60', '120', '250',  # 时间窗口
            '0.01', '0.05', '0.5', '1.0'  # 数值常数，用于加权或阈值
        ]
        for c in constants:
            self._add_token(c, TokenType.CONSTANT)

        # 4. 算子 (对应上一节设计的 FunctionSet)

        # 4.1 一元/截面算子 (One operand)
        unary = ['Abs', 'Log', 'Sign', 'ZScore', 'Rank', 'Distance_To_Best']
        for op in unary:
            self._add_token(op, TokenType.OPERATOR)
            self.unary_ops.add(op)

        # 4.2 二元算子 (Two operands)
        binary = ['Add', 'Sub', 'Mul', 'Div', 'Greater', 'Less']
        for op in binary:
            self._add_token(op, TokenType.OPERATOR)
            self.binary_ops.add(op)

        # 4.3 滚动算子 (需要 Feature + Constant)
        # 注意：在RPN中，TS_Mean(close, 60) 写作 [close, 60, TS_Mean]
        # 因此，这些算子在语法上等同于二元算子 (接收一个序列和一个常数)
        rolling = [
            'TS_Mean', 'TS_Std', 'TS_Sum', 'TS_Prod',
            'TS_Max', 'TS_Min', 'TS_EMA',
            # --- 新增的风险与趋势算子 ---
            'TS_Downside_Std',  # 下行波动
            'TS_Max_Drawdown',  # 最大回撤
            'TS_Trend_Strength'  # 趋势强度
        ]
        for op in rolling:
            self._add_token(op, TokenType.OPERATOR)
            self.rolling_ops.add(op)  # 标记为滚动算子，采样时需特殊处理

    def _add_token(self, name, type_):
        idx = len(self.idx_to_token)
        self.idx_to_token[idx] = name
        self.token_to_idx[name] = idx
        self.token_types[idx] = type_

    def size(self):
        return len(self.idx_to_token)