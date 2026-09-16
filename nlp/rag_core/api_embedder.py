"""
API Embedding 模块
==================

通过调用商业 AI 的 Embedding API 实现文本向量化。
支持通义千问、OpenAI、智谱等主流服务商。
"""

import logging
from typing import List, Optional, Dict
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ['APIEmbedder', 'EMBEDDING_PROVIDERS']

# 预设的 Embedding API 提供商配置
EMBEDDING_PROVIDERS = {
    'dashscope': {
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'default_model': 'text-embedding-v3',
        'dimension': 1024,  # text-embedding-v3 默认维度
        'model_dimensions': {
            'text-embedding-v2': 1536,
            'text-embedding-v3': 1024,
        },
    },
    'openai': {
        'base_url': 'https://api.openai.com/v1',
        'default_model': 'text-embedding-3-small',
        'dimension': 1536,
    },
    'zhipu': {
        'base_url': 'https://open.bigmodel.cn/api/paas/v4',
        'default_model': 'embedding-3',
        'dimension': 2048,
    },
}


class APIEmbedder:
    """
    API Embedding 生成器

    通过 OpenAI 兼容接口调用各类商业 Embedding 服务。
    与本地 SemanticRetriever 的 encode_texts 接口一致，可直接替换。
    """

    def __init__(
        self,
        api_key: str,
        provider: str = 'dashscope',
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        timeout: int = 60,
    ):
        """
        初始化 API Embedder

        Args:
            api_key: API 密钥
            provider: 服务商名称（dashscope/openai/zhipu）
            base_url: 自定义 API 地址
            model: 模型名称
            dimension: 向量维度
            timeout: 请求超时（秒）
        """
        self.api_key = api_key
        self.provider = provider.lower()
        self.timeout = timeout

        # 获取服务商配置
        provider_config = EMBEDDING_PROVIDERS.get(self.provider, {})

        # 设置 API 地址
        self.base_url = base_url or provider_config.get('base_url')
        if not self.base_url:
            raise ValueError(f"未知的服务商: {provider}，且未提供 base_url")

        # 设置模型
        self.model = model or provider_config.get('default_model', 'text-embedding-v3')

        # 设置向量维度（优先使用显式指定，其次按模型查找，最后用提供商默认值）
        if dimension:
            self.embedding_dim = dimension
        else:
            model_dims = provider_config.get('model_dimensions', {})
            self.embedding_dim = model_dims.get(self.model, provider_config.get('dimension', 1024))

        # 尝试导入 openai 库
        try:
            import openai
            self.use_openai_lib = True
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
            logger.info(f"使用 openai 库调用 {self.provider} Embedding API")
        except ImportError:
            self.use_openai_lib = False
            logger.info(f"openai 库未安装，使用 requests 调用 {self.provider} Embedding API")

        # 使用统计
        self.usage_stats = {
            'total_calls': 0,
            'total_tokens': 0,
        }

        logger.info(f"API Embedder 初始化完成")
        logger.info(f"  服务商: {self.provider}")
        logger.info(f"  模型: {self.model}")
        logger.info(f"  向量维度: {self.embedding_dim}")

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 10,  # 降低到 10，符合通义千问限制
        show_progress: bool = False
    ) -> np.ndarray:
        """
        将文本列表编码为向量（与本地模型接口一致）

        Args:
            texts: 文本列表
            batch_size: 批处理大小（通义千问建议 25 以下）
            show_progress: 是否显示进度

        Returns:
            向量数组 (n, dimension)
        """
        if not texts:
            return np.array([])

        all_embeddings = []

        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            if show_progress and len(texts) > batch_size:
                logger.info(f"编码进度: {i}/{len(texts)}")

            try:
                batch_embeddings = self._encode_batch(batch_texts)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"批次 {i}-{i+len(batch_texts)} 编码失败: {e}")
                # 失败时返回零向量
                all_embeddings.extend([np.zeros(self.embedding_dim) for _ in batch_texts])

        return np.array(all_embeddings)

    def _encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        编码一批文本

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        if self.use_openai_lib:
            return self._encode_with_openai(texts)
        else:
            return self._encode_with_requests(texts)

    def _encode_with_openai(self, texts: List[str]) -> List[np.ndarray]:
        """使用 openai 库编码"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )

            # 更新统计
            self.usage_stats['total_calls'] += 1
            if hasattr(response, 'usage') and response.usage:
                self.usage_stats['total_tokens'] += response.usage.total_tokens

            # 提取向量
            embeddings = [np.array(item.embedding) for item in response.data]
            return embeddings

        except Exception as e:
            logger.error(f"OpenAI 库调用失败: {e}")
            raise

    def _encode_with_requests(self, texts: List[str]) -> List[np.ndarray]:
        """使用 requests 库编码"""
        import requests
        import json

        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json; charset=utf-8"
        }

        payload = {
            "model": self.model,
            "input": texts,
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                timeout=self.timeout
            )

            # 打印详细错误信息
            if response.status_code != 200:
                logger.error(f"API 错误响应: {response.text}")

            response.raise_for_status()
            result = response.json()

            # 更新统计
            self.usage_stats['total_calls'] += 1
            if 'usage' in result:
                self.usage_stats['total_tokens'] += result['usage'].get('total_tokens', 0)

            # 提取向量
            embeddings = [np.array(item['embedding']) for item in result['data']]
            return embeddings

        except Exception as e:
            logger.error(f"Requests 调用失败: {e}")
            raise

    def encode_query(self, query: str) -> np.ndarray:
        """
        编码单个查询（与本地模型接口一致）

        Args:
            query: 查询文本

        Returns:
            查询向量
        """
        return self.encode_texts([query])[0]

    def get_embedding_dim(self) -> int:
        """
        获取向量维度（与本地模型接口一致）

        Returns:
            向量维度
        """
        return self.embedding_dim

    def get_usage_stats(self) -> Dict:
        """
        获取使用统计

        Returns:
            统计信息字典
        """
        return {
            'provider': self.provider,
            'model': self.model,
            'dimension': self.embedding_dim,
            'usage': self.usage_stats.copy()
        }

    # 方法别名，兼容不同接口
    def embed_query(self, query: str) -> List[float]:
        """embed_query别名，调用encode_query"""
        return self.encode_query(query).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """embed_documents别名，调用encode_texts"""
        return self.encode_texts(texts).tolist()
