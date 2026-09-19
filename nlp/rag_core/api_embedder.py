"""
API Embedding 模块
==================

通过调用商业 AI 的 Embedding API 实现文本向量化。
支持通义千问、OpenAI、智谱等主流服务商。

本文件只有一个类，读懂它只需抓住一件事：**它把「文本 → 向量」这一步外包给了远程 API。**

两个容易读歪的点：
- 「支持三家服务商」指的是下面那张配置表里列了三家，但**本项目实际只跑 dashscope（通义千问）**，
  调用处写死了 provider="dashscope"；openai / zhipu 两项没有任何地方使用。所以这不是多厂商适配工程，
  只是「配置表 + 默认值」的写法，别按多适配层的复杂度去读。
- 「OpenAI 兼容接口」是行业通行说法：指各家服务商都照 OpenAI 的 HTTP 协议收发请求，
  于是可以统一用 openai 这个 SDK 去调它们，**不需要各装各家自己的 SDK**（本项目就没装 dashscope 包）。
"""

import logging
from typing import List, Optional, Dict
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ['APIEmbedder', 'EMBEDDING_PROVIDERS']

# 预设的 Embedding API 提供商配置
# 三家都遵循同一套 OpenAI 兼容协议，所以同一份调用代码能通吃，差别只在地址、模型名、维度。
# 本项目只用得到 'dashscope' 这一项；另两项留在这里当下拉选项，没有代码路径会选到它们。
# dimension / model_dimensions：同一家的不同模型维度不同（如 v2=1536、v3=1024），所以维度是「跟着模型走」的。
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
    API Embedding 生成器（embedder，向量化模型）

    通过 OpenAI 兼容接口调用各类商业 Embedding 服务。
    与本地 SemanticRetriever 的 encode_texts 接口一致，可直接替换。

    这层「接口一致」是刻意的契约：调用方只认 encode_texts / encode_query / get_embedding_dim 三个方法，
    所以把本地模型换成远程 API（或反过来）时，上层检索器一行都不用改。
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
        # 隐式契约：这个维度必须和向量库里已存的向量维度一致 —— 建库时用哪个模型，检索时就得用同一个模型。
        # 换模型（v2 的 1536 维 ↔ v3 的 1024 维）会让新旧向量没法比较，只能重新建库，这是最容易踩的坑。
        if dimension:
            self.embedding_dim = dimension
        else:
            model_dims = provider_config.get('model_dimensions', {})
            self.embedding_dim = model_dims.get(self.model, provider_config.get('dimension', 1024))

        # 尝试导入 openai 库
        # openai 是本项目的硬依赖（requirements.txt 里没注释掉），所以实际总是走 try 这条分支；
        # 而 requests 那条备用通道本项目跑不到 —— 读的时候知道它存在、不必当主逻辑读。
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
        # token（词元）= LLM 计费与长度计量的最小单位，按 token 数收费，所以这里单独累计 total_tokens。
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
            向量数组 (n, dimension)，行顺序与入参 texts 严格一一对应 ——
            调用方靠下标把向量对回原文，所以下面的失败兜底必须按位补足，不能少塞。
        """
        if not texts:
            return np.array([])

        all_embeddings = []

        # 分批处理：一次请求捎带多条能省往返开销，但服务商对单批条数/长度有上限，超限会整批报错。
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
                # 为什么不抛出异常：补零是为了让返回数组的行数仍与 texts 对齐，调用方按位取不会错位。
                # 代价是调用方分不清「这条编码失败了」还是「它真的不相关」—— 零向量与任何向量相似度都是 0，
                # 通常会被下游的分数阈值滤掉，于是表现为「静默少了几条结果」。
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
        """使用 requests 库编码

        备用通道：只在 openai 库装不上时才会被 _encode_batch 选中（见 __init__ 里的 use_openai_lib）。
        它绕开 SDK、直接手拼 HTTP 请求打同一套 OpenAI 兼容协议，本项目不会走到这里。
        """
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
    # embed_query / embed_documents 是 LangChain 那套 Embeddings 接口约定的方法名：
    # 外部组件（如 LangChain 的向量库封装）按这两个名字来认，所以这里保留别名，但内部实现就是上面的 encode_*。
    # 注意返回类型不同：encode_* 给的是 numpy 数组，别名给的是普通 list（便于直接序列化）。
    def embed_query(self, query: str) -> List[float]:
        """embed_query别名，调用encode_query"""
        return self.encode_query(query).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """embed_documents别名，调用encode_texts"""
        return self.encode_texts(texts).tolist()
