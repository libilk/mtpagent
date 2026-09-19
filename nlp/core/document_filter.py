# -*- coding: utf-8 -*-
"""
文档元数据管理
==============

支持文档分层和角色过滤。

**现状：过滤逻辑写了，但运行时不会触发 —— 别按"正在生效的权限控制"读。**
[enhanced_entry.py](../langgraph_orchestrator/enhanced_entry.py) 把 `doc_filter` 硬编码
传 `None`，而 `KnowledgeAgent` 只在 document_filter 为真时才启用过滤 —— 于是本文件的
`filter_by_*` 三个方法在现役流程里都走不到。类本身没问题，属"有代码没接线"。

术语：audience（受众）= 读文档的人的角色；domain（领域）= 文档属于哪条业务线。
过滤的思路是**先猜"用户是谁"再裁可见范围**，所以猜错就会多滤或漏滤。
"""

import json
import logging
from typing import List, Dict, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)


class Domain(str, Enum):
    """文档领域（Enum：一组固定的命名常量；继承 str 是为了能直接与字符串比较、序列化）"""
    TECH = "tech"  # 技术
    SALES = "sales"  # 销售
    PRODUCT = "product"  # 产品
    CUSTOMER_SERVICE = "customer_service"  # 客服
    GENERAL = "general"  # 通用


class Audience(str, Enum):
    """目标受众"""
    DEVELOPER = "developer"  # 开发人员
    SALES_REP = "sales_rep"  # 销售人员
    CUSTOMER_SERVICE_REP = "customer_service_rep"  # 客服人员
    PRODUCT_MANAGER = "product_manager"  # 产品经理
    END_USER = "end_user"  # 终端用户
    ALL = "all"  # 所有人


class DocumentMetadata:
    """文档元数据"""

    def __init__(
        self,
        doc_id: str,
        title: str,
        domain: Domain,
        audiences: List[Audience],
        tags: Optional[List[str]] = None,
        priority: int = 0
    ):
        """
        初始化

        Args:
            doc_id: 文档ID
            title: 文档标题
            domain: 文档领域
            audiences: 目标受众列表
            tags: 标签列表
            priority: 优先级（数字越大优先级越高）
        """
        self.doc_id = doc_id
        self.title = title
        self.domain = domain
        self.audiences = audiences
        self.tags = tags or []
        self.priority = priority

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "domain": self.domain,
            "audiences": self.audiences,
            "tags": self.tags,
            "priority": self.priority
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DocumentMetadata":
        """从字典创建"""
        return cls(
            doc_id=data["doc_id"],
            title=data["title"],
            domain=data["domain"],
            audiences=data["audiences"],
            tags=data.get("tags", []),
            priority=data.get("priority", 0)
        )


class DocumentFilter:
    """文档过滤器"""

    def __init__(self):
        """初始化"""
        self.metadata_store = {}  # {doc_id: DocumentMetadata}

    def add_document(self, metadata: DocumentMetadata):
        """
        添加文档元数据

        Args:
            metadata: 文档元数据
        """
        self.metadata_store[metadata.doc_id] = metadata
        logger.debug(f"添加文档元数据: {metadata.doc_id}")

    def filter_by_audience(
        self,
        doc_ids: List[str],
        audience: Audience
    ) -> List[str]:
        """
        根据受众过滤文档

        Args:
            doc_ids: 文档ID列表
            audience: 目标受众

        Returns:
            过滤后的文档ID列表
        """
        filtered = []
        for doc_id in doc_ids:
            metadata = self.metadata_store.get(doc_id)
            if metadata:
                # 检查是否匹配受众
                if audience in metadata.audiences or Audience.ALL in metadata.audiences:
                    filtered.append(doc_id)
            else:
                # 没有元数据的文档默认保留 —— 这是 fail-open（失败放行）：
                # 漏登记元数据 = 谁都看得到，而不是谁都看不到。想收紧权限时这里是关键点。
                filtered.append(doc_id)

        logger.debug(f"受众过滤: {len(doc_ids)} -> {len(filtered)} (audience={audience})")
        return filtered

    def filter_by_domain(
        self,
        doc_ids: List[str],
        domains: List[Domain]
    ) -> List[str]:
        """
        根据领域过滤文档

        Args:
            doc_ids: 文档ID列表
            domains: 领域列表

        Returns:
            过滤后的文档ID列表
        """
        filtered = []
        for doc_id in doc_ids:
            metadata = self.metadata_store.get(doc_id)
            if metadata:
                if metadata.domain in domains or metadata.domain == Domain.GENERAL:
                    filtered.append(doc_id)
            else:
                # 没有元数据的文档默认保留
                filtered.append(doc_id)

        logger.debug(f"领域过滤: {len(doc_ids)} -> {len(filtered)} (domains={domains})")
        return filtered

    def filter_by_tags(
        self,
        doc_ids: List[str],
        tags: List[str]
    ) -> List[str]:
        """
        根据标签过滤文档

        Args:
            doc_ids: 文档ID列表
            tags: 标签列表

        Returns:
            过滤后的文档ID列表
        """
        filtered = []
        for doc_id in doc_ids:
            metadata = self.metadata_store.get(doc_id)
            if metadata:
                # 检查是否有交集
                if set(metadata.tags) & set(tags):
                    filtered.append(doc_id)
            else:
                filtered.append(doc_id)

        return filtered

    def get_metadata(self, doc_id: str) -> Optional[DocumentMetadata]:
        """获取文档元数据"""
        return self.metadata_store.get(doc_id)

    def save_to_file(self, file_path: str):
        """保存元数据到文件"""
        data = {
            doc_id: metadata.to_dict()
            for doc_id, metadata in self.metadata_store.items()
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"元数据已保存到: {file_path}")

    def load_from_file(self, file_path: str):
        """从文件加载元数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for doc_id, metadata_dict in data.items():
                metadata = DocumentMetadata.from_dict(metadata_dict)
                self.metadata_store[doc_id] = metadata
            logger.info(f"从文件加载了 {len(data)} 条元数据")
        except FileNotFoundError:
            logger.warning(f"元数据文件不存在: {file_path}")
        except Exception as e:
            logger.error(f"加载元数据失败: {e}")


def infer_audience_from_query(query: str) -> Audience:
    """
    从查询推断用户角色

    Args:
        query: 用户查询

    Returns:
        推断的受众

    注意：**返回 `Audience.ALL` 不等于"不过滤"** —— 下游 `filter_by_audience` 会拿它
    当普通角色比对，只有显式标了 ALL 受众的文档才留下。这是最容易读反的一处。
    """
    query_lower = query.lower()

    # 开发人员关键词
    dev_keywords = ["代码", "api", "接口", "函数", "算法", "bug", "调试", "编程"]
    if any(kw in query_lower for kw in dev_keywords):
        return Audience.DEVELOPER

    # 销售人员关键词
    sales_keywords = ["价格", "报价", "客户", "销售", "合同", "商务", "竞品"]
    if any(kw in query_lower for kw in sales_keywords):
        return Audience.SALES_REP

    # 客服人员关键词
    cs_keywords = ["工单", "投诉", "退款", "售后", "问题", "咨询"]
    if any(kw in query_lower for kw in cs_keywords):
        return Audience.CUSTOMER_SERVICE_REP

    # 默认返回ALL
    return Audience.ALL
