# -*- coding: utf-8 -*-
"""
行业标准动态计算模块
==================

从历史合同数据中统计行业标准，用于风险识别。

**现状：本模块没有现役调用方，别当现役逻辑读。**
唯一的调用方是 `agents/document_agent`，而该 Agent 已在阶段 4 从图里摘除
（它那 9 个工具全是合同审核专用的，与电商售后无关）。文件保留未删，
若日后恢复合同审核场景，把 `_register_document_agent` 接回来即可。

术语：retriever（检索器）= 负责从知识库召回文档的组件；
本模块拿它去捞"历史合同"当统计样本 —— 用检索器做抽样，是这一层最反直觉的地方。
"""

import re
import json
import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class IndustryStandardsCalculator:
    """行业标准计算器"""

    def __init__(self, retriever=None):
        """
        初始化

        Args:
            retriever: 向量检索器（retriever：负责召回文档的组件，此处用来取历史合同）
        """
        self.retriever = retriever
        self.standards_cache = {}  # 缓存（cache）：按 contract_type 存算过的结果，避免重复统计

    def calculate_standards(
        self,
        contract_type: Optional[str] = None,
        force_refresh: bool = False
    ) -> Dict:
        """
        计算行业标准

        Args:
            contract_type: 合同类型（如：软件开发、硬件采购等）
            force_refresh: 是否强制刷新缓存

        Returns:
            行业标准字典
        """
        cache_key = contract_type or "all"

        # 检查缓存
        if not force_refresh and cache_key in self.standards_cache:
            logger.info(f"[行业标准] 使用缓存数据: {cache_key}")
            return self.standards_cache[cache_key]

        logger.info(f"[行业标准] 开始计算: {cache_key}")

        try:
            # 1. 检索历史合同
            contracts = self._retrieve_contracts(contract_type)

            if not contracts or len(contracts) < 3:
                logger.warning(f"[行业标准] 历史合同数量不足（{len(contracts)}份），使用默认标准")
                return self._get_default_standards()

            logger.info(f"[行业标准] 检索到 {len(contracts)} 份历史合同")

            # 2. 提取各项指标
            penalty_rates = []
            prepayments = []
            warranties = []
            delivery_times = []
            amounts = []

            for contract in contracts:
                content = contract.get("content", "")

                # 提取违约金比例
                penalty = self._extract_penalty_rate(content)
                if penalty:
                    penalty_rates.append(penalty)

                # 提取预付款比例
                prepayment = self._extract_prepayment(content)
                if prepayment:
                    prepayments.append(prepayment)

                # 提取质保期
                warranty = self._extract_warranty_period(content)
                if warranty:
                    warranties.append(warranty)

                # 提取交付时间
                delivery = self._extract_delivery_time(content)
                if delivery:
                    delivery_times.append(delivery)

                # 提取合同金额
                amount = self._extract_amount(content)
                if amount:
                    amounts.append(amount)

            # 3. 计算统计指标
            standards = {
                "sample_size": len(contracts),
                "contract_type": contract_type or "全部类型",
                "penalty_rate": self._calculate_stats(penalty_rates, "违约金比例"),
                "prepayment": self._calculate_stats(prepayments, "预付款比例"),
                "warranty": self._calculate_stats(warranties, "质保期"),
                "delivery_time": self._calculate_stats(delivery_times, "交付时间"),
                "amount": self._calculate_stats(amounts, "合同金额"),
            }

            # 4. 缓存结果
            self.standards_cache[cache_key] = standards

            logger.info(f"[行业标准] 计算完成")
            return standards

        except Exception as e:
            logger.error(f"[行业标准] 计算失败: {e}", exc_info=True)
            return self._get_default_standards()

    def _retrieve_contracts(self, contract_type: Optional[str]) -> List[Dict]:
        """检索历史合同"""
        if not self.retriever:
            return []

        try:
            # 构建查询
            if contract_type:
                query = f"{contract_type}合同"
            else:
                query = "采购合同 软件开发 硬件采购 云服务"

            # 检索更多合同用于统计。
            # top_k（取前 k 条）= 只要得分最高的 50 份；样本量直接决定后面统计的可信度，
            # 而 retrieve 的排序是语义相关性、不是随机抽样，样本天然有偏 —— 这是本方案的弱点。
            results = self.retriever.retrieve(query, top_k=50)
            return results

        except Exception as e:
            logger.error(f"检索历史合同失败: {e}")
            return []

    def _extract_penalty_rate(self, text: str) -> Optional[float]:
        """
        提取违约金比例

        示例：
        - "每延期一天，按合同总额的0.5%支付违约金" → 0.5
        - "延期违约金1.5%/天" → 1.5
        """
        patterns = [
            r'违约金.*?(\d+\.?\d*)%[/／]天',
            r'每延期一天.*?(\d+\.?\d*)%',
            r'按合同总额的(\d+\.?\d*)%.*?违约',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                rate = float(match.group(1))
                # 合理性检查（0.1% - 5%）：越界值整条丢弃（返回 None），不是截断到边界 ——
                # 截断会凭空造出一个不存在的样本，把均值往中间拉。
                if 0.1 <= rate <= 5.0:
                    return rate

        return None

    def _extract_prepayment(self, text: str) -> Optional[float]:
        """
        提取预付款比例

        示例：
        - "支付合同总额的30%作为预付款" → 30
        - "预付款40%" → 40
        """
        patterns = [
            r'预付款.*?(\d+)%',
            r'支付.*?(\d+)%.*?预付',
            r'合同总额的(\d+)%作为预付',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                rate = float(match.group(1))
                # 合理性检查（10% - 80%）
                if 10 <= rate <= 80:
                    return rate

        return None

    def _extract_warranty_period(self, text: str) -> Optional[int]:
        """
        提取质保期（月）

        示例：
        - "质保期为12个月" → 12
        - "质保期：24个月" → 24
        """
        patterns = [
            r'质保期.*?(\d+)\s*个?月',
            r'质保.*?(\d+)\s*个?月',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                months = int(match.group(1))
                # 合理性检查（1 - 60个月）
                if 1 <= months <= 60:
                    return months

        return None

    def _extract_delivery_time(self, text: str) -> Optional[int]:
        """
        提取交付时间（月）

        示例：
        - "6个月内完成" → 6
        - "交付时间：4个月" → 4
        """
        patterns = [
            r'(\d+)\s*个?月内.*?完成',
            r'交付.*?(\d+)\s*个?月',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                months = int(match.group(1))
                # 合理性检查（1 - 24个月）
                if 1 <= months <= 24:
                    return months

        return None

    def _extract_amount(self, text: str) -> Optional[float]:
        """
        提取合同金额（万元）

        示例：
        - "合同总金额：人民币800,000元" → 80
        - "150万元" → 150
        """
        patterns = [
            r'(\d+)万元',
            r'(\d{1,3}(?:,\d{3})+)元',  # 带逗号的金额
            r'合同.*?金额.*?(\d+)元',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                amount_str = match.group(1).replace(',', '')
                amount = float(amount_str)

                # 转换为万元
                # 靠 pattern 文本里有没有「万」判断单位，所以上面三条的顺序不能调换：
                # 必须先试「万元」再试纯「元」，否则 150万 会被当成 150 元。
                if '万' not in pattern:
                    amount = amount / 10000

                # 合理性检查（1万 - 10000万）
                if 1 <= amount <= 10000:
                    return amount

        return None

    def _calculate_stats(self, values: List[float], name: str) -> Dict:
        """计算统计指标：mean（均值）/ median（中位数）/ std（标准差，越小说明行业越统一）"""
        if not values:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
                "std": None
            }

        values_array = np.array(values)

        stats = {
            "count": len(values),
            "mean": float(np.mean(values_array)),
            "median": float(np.median(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "std": float(np.std(values_array))
        }

        logger.info(f"  {name}: 均值={stats['mean']:.2f}, 范围=[{stats['min']:.2f}, {stats['max']:.2f}], 样本数={stats['count']}")

        return stats

    def _get_default_standards(self) -> Dict:
        """
        获取默认行业标准（基于经验值）

        注意（事实性说明，未改逻辑）：这里各项 count 都是 0，而 `_format_standards_text`
        只在 count > 0 时才输出 —— 所以这些经验值经 `format_standards_for_prompt`
        渲染后只剩一行标题，数字其实到不了 prompt 里。
        """
        logger.info("[行业标准] 使用默认标准")

        return {
            "sample_size": 0,
            "contract_type": "默认标准",
            "penalty_rate": {
                "count": 0,
                "mean": 0.4,
                "median": 0.3,
                "min": 0.3,
                "max": 0.5,
                "std": 0.1
            },
            "prepayment": {
                "count": 0,
                "mean": 30.0,
                "median": 30.0,
                "min": 20.0,
                "max": 40.0,
                "std": 5.0
            },
            "warranty": {
                "count": 0,
                "mean": 12.0,
                "median": 12.0,
                "min": 6.0,
                "max": 24.0,
                "std": 3.0
            },
            "delivery_time": {
                "count": 0,
                "mean": 6.0,
                "median": 6.0,
                "min": 3.0,
                "max": 12.0,
                "std": 2.0
            },
            "amount": {
                "count": 0,
                "mean": 100.0,
                "median": 80.0,
                "min": 10.0,
                "max": 500.0,
                "std": 50.0
            }
        }

    def format_standards_for_prompt(self, standards: Dict) -> str:
        """格式化行业标准为 prompt（提示词：喂给 LLM 的指令文本）"""
        if standards["sample_size"] == 0:
            return "【行业标准】（基于经验值）\n" + self._format_standards_text(standards)
        else:
            return f"【行业标准】（基于{standards['sample_size']}份历史合同统计）\n" + self._format_standards_text(standards)

    def _format_standards_text(self, standards: Dict) -> str:
        """格式化标准文本"""
        lines = []

        # 违约金
        penalty = standards["penalty_rate"]
        if penalty["count"] > 0:
            lines.append(f"- 违约金标准：{penalty['mean']:.2f}%/天（范围：{penalty['min']:.2f}%-{penalty['max']:.2f}%）")

        # 预付款
        prepayment = standards["prepayment"]
        if prepayment["count"] > 0:
            lines.append(f"- 预付款比例：{prepayment['mean']:.1f}%（范围：{prepayment['min']:.1f}%-{prepayment['max']:.1f}%）")

        # 质保期
        warranty = standards["warranty"]
        if warranty["count"] > 0:
            lines.append(f"- 质保期：{warranty['mean']:.1f}个月（范围：{warranty['min']:.0f}-{warranty['max']:.0f}个月）")

        # 交付时间
        delivery = standards["delivery_time"]
        if delivery["count"] > 0:
            lines.append(f"- 交付时间：{delivery['mean']:.1f}个月（范围：{delivery['min']:.0f}-{delivery['max']:.0f}个月）")

        return "\n".join(lines)


# 便捷函数
def calculate_industry_standards(retriever, contract_type: Optional[str] = None) -> Dict:
    """
    计算行业标准（便捷函数）

    Args:
        retriever: 向量检索器
        contract_type: 合同类型

    Returns:
        行业标准字典
    """
    calculator = IndustryStandardsCalculator(retriever)
    return calculator.calculate_standards(contract_type)
