# -*- coding: utf-8 -*-
"""
Tavily Search 服务
==================

封装 Tavily Search API，提供网络搜索能力
"""

import os
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class TavilySearchService:
    """Tavily Search 服务"""

    def __init__(self, api_key: str = None):
        """
        初始化

        Args:
            api_key: Tavily API Key
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.client = None

        if not self.api_key:
            logger.warning("Tavily API Key 未配置，将使用模拟数据")
        else:
            self._init_client()

    def _init_client(self):
        """初始化 Tavily 客户端"""
        try:
            from tavily import TavilyClient
            self.client = TavilyClient(api_key=self.api_key)
            logger.info("✓ Tavily Search 初始化完成")
        except ImportError:
            logger.error("tavily-python 未安装，请运行: pip install tavily-python")
            self.client = None
        except Exception as e:
            logger.error(f"初始化 Tavily 失败: {e}")
            self.client = None

    def web_search(self, query: str, count: int = 5) -> Dict[str, Any]:
        """
        网络搜索

        Args:
            query: 搜索查询
            count: 返回结果数量

        Returns:
            搜索结果
        """
        if not self.client:
            return self._get_mock_data(query)

        try:
            logger.info(f"Tavily Search: {query}")

            # 调用 Tavily API
            response = self.client.search(
                query=query,
                max_results=count,
                search_depth="basic"  # basic 或 advanced
            )

            return self._format_results(response)

        except Exception as e:
            logger.error(f"Tavily Search 失败: {e}")
            return {"success": False, "error": str(e)}

    def _format_results(self, response: Dict) -> Dict[str, Any]:
        """格式化搜索结果"""
        try:
            results = response.get("results", [])

            if not results:
                return {"success": True, "results": "未找到相关结果"}

            # 格式化为文本
            formatted = "【搜索结果】\n\n"
            for i, result in enumerate(results, 1):
                title = result.get("title", "")
                content = result.get("content", "")
                url = result.get("url", "")

                formatted += f"{i}. {title}\n"
                formatted += f"{content}\n"
                formatted += f"来源: {url}\n\n"

            return {"success": True, "results": formatted}

        except Exception as e:
            logger.error(f"格式化结果失败: {e}")
            return {"success": False, "error": str(e)}

    def _get_mock_data(self, query: str) -> Dict[str, Any]:
        """获取模拟数据"""
        mock_results = {
            "劳动合同法": {
                "success": True,
                "results": """
【搜索结果】劳动合同法最新规定（2026年）

1. 违约金上限：不得超过劳动者月工资的3倍
2. 试用期规定：
   - 3个月以下合同：不得约定试用期
   - 3个月-1年：试用期不超过1个月
   - 1-3年：试用期不超过2个月
   - 3年以上：试用期不超过6个月
3. 经济补偿：每满一年支付一个月工资
4. 加班费：工作日1.5倍，休息日2倍，法定节假日3倍

来源：中华人民共和国人力资源和社会保障部
更新时间：2026-01-15
                """
            },
            "软件开发合同": {
                "success": True,
                "results": """
【搜索结果】软件开发合同行业标准（2026年）

1. 付款方式：通常采用 3-3-3-1 模式
   - 签约后支付30%
   - 需求确认后支付30%
   - 开发完成后支付30%
   - 验收通过后支付10%

2. 知识产权：
   - 定制开发：著作权归委托方
   - 通用模块：开发方保留权利

3. 质保期：通常为验收后12个月

4. 违约责任：
   - 延期交付：每日支付合同金额0.5%违约金
   - 质量问题：免费修复或退款

来源：中国软件行业协会
更新时间：2026-02-20
                """
            }
        }

        # 简单匹配
        for keyword, data in mock_results.items():
            if keyword in query:
                logger.info(f"使用模拟数据: {keyword}")
                return data

        return {
            "success": True,
            "results": f"未找到关于 '{query}' 的相关信息（模拟数据）"
        }
