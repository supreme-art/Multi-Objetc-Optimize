# @Author : CY
# @Time : 2026/1/5 15:44
import Config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

# ==========================================
# 第三部分：核心算法实现 (NSGA-III Core)
# 功能：实现目标计算、修复算子、交叉变异及选择逻辑
# ==========================================
class NSGA3_Optimizer:
    def __init__(self, returns, scores, sectors):
        self.returns = returns.values  # (T, N)
        self.scores = scores.values  # (N,)
        self.sectors = sectors.values  # (N,)
        self.n_stocks = len(scores)

        # 生成风险分层参考点
        self.ref_points = self._generate_risk_stratified_refs()

        # 记录历史 (保留您的CSV输出需求)
        self.hv_sp_history = []

        # 【新增】基因保护机制相关
        # protected_mask[i, j] = True 表示第 i 个个体的第 j 只股票是“优质核心”，需保护
        self.protected_masks = np.zeros((Config.N_POP, self.n_stocks), dtype=bool)
        self.window_size = 5     # 滑动窗口大小 (5只股票)
        self.protect_freq = 10   # 每10代更新一次保护名单

        # 【新增】计算个股质量分级 (0:劣质, 1:普通, 2:优质)
        # 这一步只在初始化做一次，不影响迭代速度
        self.stock_tiers = self._calculate_stock_quality_tiers()

        # 【新增】全局测量基准 (防止 HV 剧烈抖动)
        # 初始设为无穷大/无穷小，后续动态更新
        self.global_min_objs = np.full(3, np.inf)
        self.global_max_objs = np.full(3, -np.inf)

    def _calculate_stock_quality_tiers(self):
        """
        【新增】基于因子分和波动率，给股票分级
        Tier 2 (Elite): 高分低波 (前 30%)
        Tier 0 (Weak):  低分高波 (后 30%)
        Tier 1 (Mid):   中间 (40%)
        """
        # 1. 计算个股历史波动率 (风险代理)
        # axis=0 对时间维度求标准差
        volatility = np.nanstd(self.returns, axis=0)
        # 填充 nan (防止报错)
        volatility[np.isnan(volatility)] = np.nanmean(volatility)

        # 2. 归一化 (越小越好 -> 转为越大越好)
        # 分数归一化 (0-1)
        s_min, s_max = np.min(self.scores), np.max(self.scores)
        norm_score = (self.scores - s_min) / (s_max - s_min + 1e-9)

        # 波动率归一化 (0-1)，注意：波动率越小越好，所以取反
        v_min, v_max = np.min(volatility), np.max(volatility)
        norm_vol = 1 - (volatility - v_min) / (v_max - v_min + 1e-9)

        # 3. 综合质量分 = 0.7 * 因子分 + 0.3 * 低波分
        # (权重可调，假设我们更看重因子)
        quality_score = 0.7 * norm_score + 0.3 * norm_vol

        # 4. 分位数分级
        tiers = np.ones(self.n_stocks, dtype=int) # 默认为 1 (普通)

        th_high = np.percentile(quality_score, 70) # Top 30%
        th_low = np.percentile(quality_score, 30)  # Bottom 30%

        tiers[quality_score >= th_high] = 2  # 优质
        tiers[quality_score <= th_low] = 0   # 劣质

        print(f"  [Stock Quality] Elite: {np.sum(tiers==2)}, Mid: {np.sum(tiers==1)}, Weak: {np.sum(tiers==0)}")
        return tiers

    def _generate_risk_stratified_refs(self):
        """生成30个风险分层参考点"""
        """
        【改进】混合参考点：偏好点 + 均匀点 + 边界点
        作用：填补解集空隙，提升 SP；拉伸边界，提升 HV
        """
        refs = []

        # 1. 业务偏好点 (保持原有逻辑，占约 40%)
        # 保守
        for _ in range(12):
            r = np.array([0.2, 0.6, 0.2]) + np.random.rand(3) * 0.05
            refs.append(r / r.sum())
        # 平衡
        for _ in range(12):
            r = np.array([0.33, 0.33, 0.33]) + np.random.rand(3) * 0.05
            refs.append(r / r.sum())
        # 激进
        for _ in range(6):
            if np.random.rand() > 0.5: r = np.array([0.6, 0.1, 0.3])
            else: r = np.array([0.3, 0.1, 0.6])
            refs.append(r / r.sum())

        # 2. 【新增】均匀随机点 (占约 50%)
        # 填补偏好点之间的“无人区”，让前沿连成一条线
        n_uniform = 40
        for _ in range(n_uniform):
            r = np.random.rand(3)
            refs.append(r / r.sum())

        # 3. 【新增】极端边界点 (Anchor Points)
        # 强行拉伸 HV 的边界
        refs.append(np.array([0.98, 0.01, 0.01])) # 极度重收益
        refs.append(np.array([0.01, 0.98, 0.01])) # 极度重风险
        refs.append(np.array([0.01, 0.01, 0.98])) # 极度重得分

        return np.array(refs)

    def calculate_objectives(self, z):
        sel_idx = np.where(z == 1)[0]
        # 空解惩罚
        if len(sel_idx) == 0:
            return np.array([-999.0, 999.0, -999.0])

        # 1. 因子得分
        f_score = float(np.mean(self.scores[sel_idx]))

        # 2. 收益与回撤
        # 使用 nanmean 处理停牌/缺失数据
        sel_ret = self.returns[:, sel_idx]
        port_daily_ret = np.nanmean(sel_ret, axis=1)

        # 兜底检查
        if np.all(np.isnan(port_daily_ret)):
            return np.array([-999.0, 999.0, -999.0])

        # 净值计算 (Log累加更稳定)
        log_nav = np.cumsum(np.log1p(port_daily_ret))
        nav = np.exp(log_nav)

        # 年化收益
        ann_ret = nav[-1] ** (Config.TRADING_DAYS / len(nav)) - 1

        # 最大回撤
        peak = np.maximum.accumulate(nav)
        dd = (peak - nav) / peak
        max_dd = float(np.max(dd))

        return np.array([ann_ret, max_dd, f_score])

    def _calculate_window_metrics(self, indices):
        """
        【新增】计算种子窗口的指标 (局部评估)
        用于判断这 5-10 只股票组成的小组合是否足够优秀
        """
        f_score = float(np.mean(self.scores[indices]))

        sel_ret = self.returns[:, indices]
        port_daily_ret = np.nanmean(sel_ret, axis=1)

        # 简单计算年化收益和波动率 (作为风险代理，比回撤计算快)
        ann_ret = np.mean(port_daily_ret) * Config.TRADING_DAYS
        ann_vol = np.std(port_daily_ret) * np.sqrt(Config.TRADING_DAYS)

        # 返回: [收益, 风险(波动率), 得分]
        return ann_ret, ann_vol, f_score

    def update_protected_fragments(self, pop):
        """
        【新增】核心思想实现：识别并锁定优质片段
        步骤 1 & 3: 滑动窗口扫描，标记优质子集
        """
        # 重置保护掩码
        self.protected_masks = np.zeros((len(pop), self.n_stocks), dtype=bool)

        # 收集所有窗口的指标以确定相对阈值
        all_window_metrics = []
        window_mapping = [] # (pop_idx, stock_indices)

        for i, ind in enumerate(pop):
            sel_idx = np.where(ind == 1)[0]
            if len(sel_idx) < self.window_size:
                continue

            # 滑动窗口 (Step 1)
            # 假设 sel_idx 是 [0, 5, 12, ...]，窗口就是 [0,5,12,15,20]
            for start in range(len(sel_idx) - self.window_size + 1):
                window_indices = sel_idx[start : start + self.window_size]

                ret, vol, score = self._calculate_window_metrics(window_indices)
                all_window_metrics.append([ret, vol, score])
                window_mapping.append((i, window_indices))

        if not all_window_metrics:
            return

        # 确定动态阈值 (Top 20% 收益，Bottom 20% 风险，Top 20% 得分)
        # 只要满足其中两个条件，或者综合表现优异，即判定为“非支配窗口”
        metrics = np.array(all_window_metrics)

        th_ret = np.percentile(metrics[:, 0], 70) # 收益前30%
        th_vol = np.percentile(metrics[:, 1], 40) # 风险后40% (越小越好)
        th_score = np.percentile(metrics[:, 2], 60) # 得分前40%

        # 筛选优质窗口
        count_protected = 0
        for idx, (ret, vol, score) in enumerate(metrics):
            # 判定逻辑：高收益低风险 OR 高收益高分 OR 低风险高分
            is_good = 0
            if ret > th_ret: is_good += 1
            if vol < th_vol: is_good += 1
            if score > th_score: is_good += 1

            if is_good >= 2: # 满足两项及以上
                pop_idx, indices = window_mapping[idx]
                # 标记为保护 (Step 1 end)
                self.protected_masks[pop_idx, indices] = True
                count_protected += 1

        print(f"  [Gene Protection] Identified {count_protected} elite fragments across population.")

    def _crossover_with_protection(self, p1, p2, p1_idx, p2_idx, gen, max_gen):
        """
        【改进】带有保护机制的交叉算子
        步骤 2: 强制保留保护区域
        """
        # 1. 基础交叉 (均匀交叉或两点交叉)
        if np.random.rand() > Config.PC:
            child = p1.copy()
        else:
            # 均匀交叉
            mask = np.random.rand(self.n_stocks) < 0.5
            child = p1.copy()
            child[mask] = p2[mask]

        # 2. 【关键】强制覆盖受保护的基因 (Step 2)
        # 如果父本1在某位置有保护，子代必须继承父本1的该位置值
        mask1 = self.protected_masks[p1_idx]
        mask2 = self.protected_masks[p2_idx]

        # 逻辑：受保护的基因片段意味着“这个股票在这个组合里很好”，所以必须保留
        # 如果 p1 说要保，就用 p1 的；如果 p2 说要保，就用 p2 的
        # 如果两者都要保，通常它们的值都是 1（选中），所以冲突不大

        child[mask1] = p1[mask1]
        child[mask2] = p2[mask2]

        # 3. 【新增】生成子代的保护 Mask
        # 子代的保护区 = 父母保护区的并集 (只要父母任意一方说要保，子代就得保)
        child_mask = mask1 | mask2

        # 3. 修复 (传入 gen)
        return self.repair(child, child_mask, gen, max_gen), child_mask

    def _roulette_select(self, indices, n_select, current_gen=0, max_gen=100, is_add=True):
        """
        【改进】退火式贪婪选择
        前期(gen=0) epsilon较低(如0.5)，允许更多随机探索
        后期(gen=max) epsilon趋近1.0，强制锁定最优，减少波动
        """
        if len(indices) == 0: return []

        cand_scores = self.scores[indices]
        if is_add: sorted_args = np.argsort(cand_scores)[::-1]
        else: sorted_args = np.argsort(cand_scores)
        sorted_indices = indices[sorted_args]

        selected = []

        # 【动态 Epsilon】线性增长: 0.5 -> 0.95
        progress = current_gen / max(1, max_gen)
        epsilon = 0.5 + 0.45 * progress

        pool_size = max(1, min(len(sorted_indices), 5))

        for _ in range(n_select):
            if np.random.rand() < epsilon:
                # 贪婪模式
                choice = sorted_indices[0]
            else:
                # 随机模式
                # 再次防止越界
                curr_pool = min(pool_size, len(sorted_indices))
                choice = sorted_indices[np.random.randint(0, curr_pool)]

            selected.append(choice)
            # 移除已选
            sorted_indices = sorted_indices[sorted_indices != choice]
            if len(sorted_indices) == 0: break

        return np.array(selected)

    def repair(self, z,protected_mask=None, current_gen=0, max_gen=100):
        """
        【改进】感知保护机制的修复算子
        参数:
            z: 个体基因
            protected_mask: (可选) 布尔数组，True表示该位置受保护，不建议修改
        """
        z_new = z.copy()

        # 如果没有提供掩码，创建一个全 False 的
        if protected_mask is None:
            protected_mask = np.zeros(self.n_stocks, dtype=bool)

        # --- 1. 数量约束 (10-30) ---
        while True:
            k = int(np.sum(z_new))
            if k < Config.K_MIN:
                # 加仓：从没买的里面选
                candidates = np.where(z_new == 0)[0]
                if len(candidates) == 0: break
                to_add = self._roulette_select(candidates, 1, current_gen, max_gen, is_add=True)
                z_new[to_add] = 1

            elif k > Config.K_MAX:
                # 减仓：【关键修改】优先剔除“未受保护”的持仓
                # 只有当非保护持仓删光了，才被迫删保护持仓

                # 找出所有持仓
                held_indices = np.where(z_new == 1)[0]

                # 分成两类：受保护的 / 没受保护的
                unprotected_held = held_indices[~protected_mask[held_indices]]
                protected_held = held_indices[protected_mask[held_indices]]

                if len(unprotected_held) > 0:
                    # 优先在没保护的里面删
                    to_remove = self._roulette_select(unprotected_held, 1, current_gen, max_gen, is_add=False)
                else:
                    # 没办法，只能动保护的了 (极少发生)
                    to_remove = self._roulette_select(protected_held, 1, current_gen, max_gen, is_add=False)

                z_new[to_remove] = 0
            else:
                break

        # --- 2. 行业覆盖约束 ---
        # (逻辑同理：如果需要剔除平衡数量，优先剔除非保护的)
        sel_idx = np.where(z_new == 1)[0]
        curr_secs = np.unique(self.sectors[sel_idx])

        if len(curr_secs) < Config.MIN_SECTORS:
            missing = np.setdiff1d(np.unique(self.sectors), curr_secs)
            cand = np.where(np.isin(self.sectors, missing) & (z_new==0))[0]

            if len(cand) > 0:
                n_need = Config.MIN_SECTORS - len(curr_secs)
                to_add = self._roulette_select(cand, n_need, current_gen, max_gen,is_add=True)
                z_new[to_add] = 1

                # 平衡数量
                if np.sum(z_new) > Config.K_MAX:
                    # 重新获取持仓
                    held_indices = np.where(z_new == 1)[0]
                    unprotected_held = held_indices[~protected_mask[held_indices]]
                    protected_held = held_indices[protected_mask[held_indices]]

                    n_del = np.sum(z_new) - Config.K_MAX

                    # 尝试从非保护中删除
                    if len(unprotected_held) >= n_del:
                        to_remove = self._roulette_select(unprotected_held, n_del,current_gen, max_gen, is_add=False)
                        z_new[to_remove] = 0
                    else:
                        # 混着删
                        z_new[unprotected_held] = 0 # 全删
                        rem_del = n_del - len(unprotected_held)
                        to_remove = self._roulette_select(protected_held, rem_del,current_gen, max_gen, is_add=False)
                        z_new[to_remove] = 0

        # 3. 行业占比
        sel_idx = np.where(z_new == 1)[0]
        k = len(sel_idx)
        unique_secs = np.unique(self.sectors[sel_idx])
        for s in unique_secs:
            cnt = np.sum(self.sectors[sel_idx] == s)
            max_allow = max(1, int(k * Config.MAX_SECTOR_RATIO))
            if cnt > max_allow:
                in_sec = sel_idx[self.sectors[sel_idx] == s]
                to_remove = self._roulette_select(in_sec, cnt - max_allow, is_add=False)
                z_new[to_remove] = 0

        # 兜底
        if np.sum(z_new) < Config.K_MIN:
            cand = np.where(z_new == 0)[0]
            to_add = self._roulette_select(cand, Config.K_MIN - np.sum(z_new), is_add=True)
            z_new[to_add] = 1

        return z_new

    def _crossover(self, p1, p2):
        if np.random.rand() > Config.PC: return p1.copy()
        pt1, pt2 = np.sort(np.random.randint(0, self.n_stocks, 2))
        child = p1.copy()
        child[pt1:pt2] = p2[pt1:pt2]
        return self.repair(child)

    def _adaptive_mutation_with_protection(self, z, gen, max_gen, protected_mask):
        """
        【改进】结合了自适应 + 保护机制的变异
        """
        z_mut = z.copy()
        progress = gen / max_gen
        current_base_pm = Config.PM * (1 - 0.5 * progress)

        for i in range(self.n_stocks):
            # 如果该基因位被保护，且当前已经持有(z=1)，强制跳过变异
            # 含义：这个好股票必须留着，不允许随机翻转成0
            if protected_mask[i] and z[i] == 1:
                continue

            # 原有的质量导向逻辑
            tier = self.stock_tiers[i]
            is_held = (z[i] == 1)
            prob = current_base_pm

            if is_held:
                if tier == 2: prob *= 0.1
                elif tier == 0: prob *= 3.0
            else:
                if tier == 2: prob *= 3.0
                elif tier == 0: prob *= 0.1

            if np.random.rand() < prob:
                z_mut[i] = 1 - z_mut[i]

        # 修复 (传入 gen)
        return self.repair(z_mut, protected_mask, gen, max_gen)

    def _environmental_selection(self, pop, objs):
        """
        【关键修改3】增加去重逻辑，防止单一解占满种群
        在 NSGA-III 的环境选择阶段，如果我们在去重时过于激进，可能会把优秀的解误删。我们可以优化去重逻辑：优先保留目标函数更好（支配层级更高）的个体。
        """
        N = len(pop)
        min_objs = np.column_stack([-objs[:, 0], objs[:, 1], -objs[:, 2]])

        # 1. 快速非支配排序
        dom_count = np.zeros(N)
        for i in range(N):
            diff = min_objs - min_objs[i]
            is_dominated = np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1)
            dom_count[i] = np.sum(is_dominated)

        # 2. 排序
        sorted_idx = np.argsort(dom_count)

        # 3. 【改进】基于目标空间的智能去重
        selected_indices = []
        # 使用列表存储已选解的目标值，用于距离比对
        selected_objs = []

        # 距离阈值 (根据归一化后的量级设定，这里粗略设为绝对值差异)
        # 比如收益差 < 0.05%, 回撤差 < 0.1%, 得分差 < 0.001 视为雷同
        threshold = np.array([0.0005, 0.001, 0.001])

        for idx in sorted_idx:
            curr_obj = min_objs[idx] # 注意这是 [ -Ret, DD, -Score ]

            is_duplicate = False
            for sel_obj in selected_objs:
                # 计算曼哈顿距离或切比雪夫距离
                diff = np.abs(curr_obj - sel_obj)
                # 如果三项指标都极其接近，视为重复
                if np.all(diff < threshold):
                    is_duplicate = True
                    break

            if not is_duplicate:
                selected_indices.append(idx)
                selected_objs.append(curr_obj)

            if len(selected_indices) >= Config.N_POP:
                break

        # 如果去重后不够，再从剩下的里面补 (优先补 dom_count 小的)
        if len(selected_indices) < Config.N_POP:
            remaining = [i for i in sorted_idx if i not in selected_indices]
            n_need = Config.N_POP - len(selected_indices)
            selected_indices.extend(remaining[:n_need])

        return pop[selected_indices], objs[selected_indices]

    def _compute_hv_sp_for_population(self, objs):
        """
        【修改】移除繁重的基准估计，使用动态范围归一化
        """
        # 转最小化
        min_objs = np.column_stack([-objs[:, 0], objs[:, 1], -objs[:, 2]])

        # 1. 更新全局基准 (关键步骤!)
        # 用当前代的最值去撑大全局最值
        curr_min = np.min(min_objs, axis=0)
        curr_max = np.max(min_objs, axis=0)

        self.global_min_objs = np.minimum(self.global_min_objs, curr_min)
        self.global_max_objs = np.maximum(self.global_max_objs, curr_max)

        # 2. 提取前沿
        unique_objs = np.unique(min_objs, axis=0)
        if len(unique_objs) < 2: return 0.0, 0.0, len(unique_objs)

        # 3. 使用全局基准归一化 (而非当前代基准)
        denom = self.global_max_objs - self.global_min_objs
        denom[denom == 0] = 1.0

        # 限制在 [0, 1] 范围内 (防止因更新滞后导致的轻微越界)
        norm = np.clip((unique_objs - self.global_min_objs) / denom, 0.0, 1.0)

        # SP 计算
        dists = []
        for i in range(len(norm)):
            d = np.sum((norm - norm[i])**2, axis=1)
            d[i] = np.inf
            dists.append(np.sqrt(np.min(d)))
        d_mean = np.mean(dists)
        sp = np.sqrt(np.sum((dists - d_mean)**2) / (len(unique_objs)-1))

        # HV 计算
        ref = np.array([1.1, 1.1, 1.1])
        samps = np.random.uniform(0, 1.1, (2000, 3))
        dom = 0
        for s in samps:
            if np.any(np.all(norm <= s, axis=1)): dom += 1
        hv = dom / 2000

        return hv, sp, len(unique_objs)

    def run(self):
        """
        启发式种群初始化 (Heuristic Initialization)
        不要全靠随机生成第一代。我们可以在初始种群中手动注入一些高质量的“种子选手”。
        这样算法就是站在巨人的肩膀上进化，而不是从零开始
        :return:
        """
        print("【步骤2】算法初始化...")
        pop = []
        # 【修改】删除了原先的 baseline 估计部分，直接开始

        # --- 【新增】注入几只“种子选手” (精英解) ---
        # 种子1: 全市场因子分最高的 Top 20 组合
        # 种子注入 (传入 gen=0)
        z1 = np.zeros(self.n_stocks, int); z1[np.argsort(self.scores)[::-1][:20]] = 1
        pop.append(self.repair(z1, None, 0, Config.N_GEN))

        # 种子2: 全市场因子分最高的 Top 10 组合 (更集中)
        z_seed2 = np.zeros(self.n_stocks, dtype=int)
        top_score_idx2 = np.argsort(self.scores)[::-1][:10]
        z_seed2[top_score_idx2] = 1
        pop.append(self.repair(z_seed2, None, 0, Config.N_GEN))

        # 种子3: 随机选几个行业，买入该行业最高分的龙头 (模拟行业轮动)
        z_seed3 = np.zeros(self.n_stocks, dtype=int)
        uniq_sectors = np.unique(self.sectors)
        # 随机选 6 个行业
        chosen_secs = np.random.choice(uniq_sectors, 6, replace=False)
        for s in chosen_secs:
            # 选该行业第一名
            in_sec = np.where(self.sectors == s)[0]
            best_in_sec = in_sec[np.argmax(self.scores[in_sec])]
            z_seed3[best_in_sec] = 1
        pop.append(self.repair(z_seed3, None, 0, Config.N_GEN))

        while len(pop) < Config.N_POP:
            z = np.zeros(self.n_stocks, dtype=int)
            k = np.random.randint(Config.K_MIN, Config.K_MAX + 1)
            z[np.random.choice(self.n_stocks, k, replace=False)] = 1
            pop.append(self.repair(z, None, 0, Config.N_GEN))
        pop = np.array(pop)

        objs = np.array([self.calculate_objectives(ind) for ind in pop])

        # 记录初始状态
        hv, sp, fs = self._compute_hv_sp_for_population(objs)
        self.hv_sp_history.append({"gen": 0, "hv": hv, "sp": sp, "front_size": fs})

        self.protected_masks = np.zeros((Config.N_POP, self.n_stocks), dtype=bool)

        print(f"【步骤3】开始迭代 (共 {Config.N_GEN} 代)...")
        for gen in tqdm(range(Config.N_GEN)):
            offspring = []

            # Step 3: 每 10 代更新保护名单
            if gen % self.protect_freq == 0:
                self.update_protected_fragments(pop)

            for _ in range(Config.N_POP):
                # 记录父母索引，以便查找保护掩码
                idx1, idx2 = np.random.choice(Config.N_POP, 2, replace=False)
                p1, p2 = pop[idx1], pop[idx2]

                # Step 2: 使用带保护的交叉
                child, child_mask = self._crossover_with_protection(p1, p2, idx1, idx2, gen, Config.N_GEN)
                child = self._adaptive_mutation_with_protection(child, gen, Config.N_GEN, child_mask)
                offspring.append(child)

            offspring = np.array(offspring)
            off_objs = np.array([self.calculate_objectives(ind) for ind in offspring])

            # 合并
            mixed_pop = np.vstack([pop, offspring])
            mixed_objs = np.vstack([objs, off_objs])

            # 环境选择 (含去重)
            pop, objs = self._environmental_selection(mixed_pop, mixed_objs)

            # 【新增】鲶鱼效应 (Catfish Effect)
            # 每 5 代，强制把最后 5 名(拥挤度或层级最差的)替换为纯随机个体
            # 作用：防止种群同质化，引入外部基因
            # 【改进】鲶鱼效应的克制：只在前期进行，且后期完全停止
            # 设定为前 80% 阶段才允许鲶鱼
            if gen % 5 == 0 and gen < Config.N_GEN * 0.8:
                n_catfish = 3 # 减少数量，温和一点
                catfish_pop = []
                for _ in range(n_catfish):
                    z = np.zeros(self.n_stocks, dtype=int)
                    k = np.random.randint(Config.K_MIN, Config.K_MAX + 1)
                    z[np.random.choice(self.n_stocks, k, replace=False)] = 1
                    # 鲶鱼是新来的，gen设为0让它有高随机性
                    catfish_pop.append(self.repair(z, None, 0, Config.N_GEN))

                # 替换尾部
                pop[-n_catfish:] = np.array(catfish_pop)
                objs[-n_catfish:] = np.array([self.calculate_objectives(ind) for ind in catfish_pop])

            # 记录历史
            hv, sp, fs = self._compute_hv_sp_for_population(objs)
            self.hv_sp_history.append({"gen": gen + 1, "hv": hv, "sp": sp, "front_size": fs})

        # 循环结束后 ...
        print("【步骤4】执行终局局部搜索 (Polishing)...")

        # 对最终种群进行一轮精细打磨
        final_pop = pop.copy()
        final_objs = objs.copy()

        # 找出当前非支配的个体进行优化 (只优化精英，节省时间)
        # 简单起见，对全种群做一次尝试
        for i in tqdm(range(len(final_pop)), desc="Local Search"):
            current_z = final_pop[i]
            current_obj = final_objs[i] # [AnnRet, MaxDD, Score]

            # 尝试动作：卖出持仓中分最低的，买入没持仓中分最高的
            sel_idx = np.where(current_z == 1)[0]
            unsel_idx = np.where(current_z == 0)[0]

            if len(sel_idx) == 0 or len(unsel_idx) == 0: continue

            # 找最差的持仓 (按得分)
            worst_held = sel_idx[np.argmin(self.scores[sel_idx])]
            # 找最好的备选
            best_unheld = unsel_idx[np.argmax(self.scores[unsel_idx])]

            # 构造新解
            new_z = current_z.copy()
            new_z[worst_held] = 0
            new_z[best_unheld] = 1
            new_z = self.repair(new_z) # 确保约束

            # 评估
            new_obj = self.calculate_objectives(new_z)

            # 判定：是否支配原解？
            # 目标方向：Ret(Max), DD(Min), Score(Max)
            # 支配条件：Ret'>=Ret, DD'<=DD, Score'>=Score 且至少一个更好
            better_ret = new_obj[0] >= current_obj[0]
            better_dd = new_obj[1] <= current_obj[1]
            better_score = new_obj[2] >= current_obj[2]

            if better_ret and better_dd and better_score:
                # 如果其中有一个显著更好 (比如提升 > 0.01%)
                if (new_obj[0] > current_obj[0] + 1e-4) or \
                   (new_obj[1] < current_obj[1] - 1e-4) or \
                   (new_obj[2] > current_obj[2] + 1e-4):
                    # 更新！
                    final_pop[i] = new_z
                    final_objs[i] = new_obj

        # 重新进行一次环境选择整理
        pop, objs = self._environmental_selection(final_pop, final_objs)

        # 重新计算最终指标
        hv, sp, fs = self._compute_hv_sp_for_population(objs)
        self.hv_sp_history.append({"gen": Config.N_GEN + 1, "hv": hv, "sp": sp, "front_size": fs})

        return pop, objs, pd.DataFrame(self.hv_sp_history)