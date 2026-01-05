# @Author : CY
# @Time : 2026/1/5 16:17
import numpy as np

from RL.TokenType import TokenType


class ActionMaskGenerator:
    def __init__(self, vocab: TokenVocabulary, max_len=30):
        self.vocab = vocab
        self.max_len = max_len

    def get_valid_mask(self, current_tokens: list):
        """
        根据当前已生成的 token 列表，返回下一步合法 token 的 bool mask
        True 表示可选，False 表示不可选
        """
        vocab_size = self.vocab.size()
        mask = np.zeros(vocab_size, dtype=bool)

        # 1. 模拟 RPN 栈状态
        # stack 存储: 'series' (时间序列), 'const' (常数)
        stack = []

        try:
            for t_idx in current_tokens:
                if t_idx == self.vocab.token_to_idx['<BEG>']:
                    continue

                t_type = self.vocab.token_types[t_idx]
                t_name = self.vocab.idx_to_token[t_idx]

                if t_type == TokenType.FEATURE:
                    stack.append('series')
                elif t_type == TokenType.CONSTANT:
                    stack.append('const')
                elif t_type == TokenType.OPERATOR:
                    # 弹出操作数
                    if t_name in self.vocab.unary_ops:
                        if len(stack) < 1: raise ValueError
                        op_type = stack.pop()
                        stack.append(op_type)  # 结果类型通常保持不变
                    else:  # Binary or Rolling
                        if len(stack) < 2: raise ValueError
                        op2 = stack.pop()
                        op1 = stack.pop()
                        # Rolling 算子通常返回 series，Binary 视情况而定
                        stack.append('series')
        except:
            # 如果历史序列已经非法（理论上不应发生），全封禁或重置
            return mask

            # 2. 根据当前 Stack 状态决定下一步可选什么

        # A. 允许放置 Feature 或 Constant (只要长度没超标)
        if len(current_tokens) < self.max_len:
            for idx, t_type in self.vocab.token_types.items():
                if t_type in [TokenType.FEATURE, TokenType.CONSTANT]:
                    mask[idx] = True

        # B. 允许放置 Operator (必须有足够的操作数)
        # B.1 一元算子：栈里至少有1个元素
        if len(stack) >= 1:
            # 只有当栈顶是 series 时才允许用复杂算子（避免对常数取Log等无意义操作）
            if stack[-1] == 'series':
                for op in self.vocab.unary_ops:
                    mask[self.vocab.token_to_idx[op]] = True

        # B.2 二元算子/滚动算子：栈里至少有2个元素
        if len(stack) >= 2:
            op2_type = stack[-1]  # 栈顶
            op1_type = stack[-2]  # 次栈顶

            # 普通二元算子 (Add, Sub...)
            for op in self.vocab.binary_ops:
                mask[self.vocab.token_to_idx[op]] = True

            # 滚动算子 (TS_Mean...) 特殊约束：
            # 必须是 [Series, Constant, Operator] 的形式
            # 即栈顶必须是 Const (时间窗口)，次栈顶是 Series
            if op1_type == 'series' and op2_type == 'const':
                for op in self.vocab.rolling_ops:
                    mask[self.vocab.token_to_idx[op]] = True

        # C. 允许 SEP (结束)
        # 只有当栈里只剩 1 个元素（即最终结果），且该结果是 Series 时，才允许结束
        if len(stack) == 1 and stack[0] == 'series' and len(current_tokens) > 2:
            mask[self.vocab.token_to_idx['<SEP>']] = True

        return mask