"""
语义检索器模块
==============

使用 API Embedding 将文本编码为向量，然后通过向量相似度找到最相关的文档

检索链路的形状（记住这条线，本文件就不难读）：
    query 文本 --embedder.encode_query--> 向量 --与库里已存的 chunk 向量逐个比相似度--> 排序取 top_k
所以「语义检索」不是查关键词，而是把文本压成一串数字后比方向；两句话用词完全不同、
只要意思接近，向量就接近，这是它相对关键词检索的价值所在。

两处容易读错的地方：
- 本文件的 SemanticRetriever（检索器）**不在生产路径上** —— 全仓库无人 import 它，
  真要跑检索时用的是 enhanced_entry.py 里就地定义的 SimpleRetriever（APIEmbedder 出向量 +
  ChromaStore 存/搜）。本文件更像一份可复用的参考实现。
- 「低于阈值宁可返回空，也不硬凑」这道相关性闸门也不在本文件，而在 SimpleRetriever.MIN_VECTOR_SCORE；
  本文件的方法只负责排序，不做分数下限过滤（即可能返回一堆弱相关结果，由调用方自己卡）。
"""

import re
import logging

import jieba
import numpy as np
from typing import Any, List, Tuple, Optional, Dict
from rag_core.api_embedder import APIEmbedder

logger = logging.getLogger(__name__)

__all__ = ['SemanticRetriever']

try:
    # reranker（重排序器）= 对召回结果做精排，把最相关的顶上来（见术语表 D）。
    # 它是可选依赖：装不上就把整块能力关掉（RERANKER_AVAILABLE=False），检索仍能跑 —— 这就是 fallback（降级兜底）。
    from .reranker import RerankerFactory
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("重排序模块不可用")


class SemanticRetriever:
    """语义检索器（retriever）：用 embedding（向量化）后的向量比相似度来召回文档

    和关键词检索的分工：关键词检索擅长「词一字不差地出现」，语义检索擅长「换种说法也能找到」；
    代价是语义相近但不该命中的内容也可能被捞进来，所以排序之后通常还要一道分数阈值。
    """

    def __init__(
        self,
        embedder: APIEmbedder,
        generator: Optional[Any] = None,
        enable_rerank: bool = False,
        reranker_type: str = 'rule',
        reranker_model_path: Optional[str] = None,
        # 兼容旧参数（忽略）
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
    ):
        self.embedder = embedder
        self.embedding_dim = embedder.get_embedding_dim()
        self.generator = generator
        self.enable_rerank = enable_rerank

        logger.info(f"SemanticRetriever 初始化完成，embedding 维度: {self.embedding_dim}")

        self.reranker = None
        if enable_rerank and RERANKER_AVAILABLE:
            try:
                if reranker_type == 'cross_encoder' and reranker_model_path:
                    self.reranker = RerankerFactory.create_reranker(
                        reranker_type='cross_encoder',
                        model_path=reranker_model_path,
                        max_length=max_length
                    )
                else:
                    self.reranker = RerankerFactory.create_reranker(reranker_type='rule')
                logger.info("重排序器初始化成功")
            except Exception as e:
                logger.error(f"重排序器初始化失败: {e}")

    def encode_texts(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """批量编码文本为向量"""
        return self.embedder.encode_texts(texts, show_progress=show_progress)

    def encode_query(self, query: str) -> np.ndarray:
        """编码单个查询为向量"""
        return self.embedder.encode_query(query)

    def expand_query(self, query: str, expansion_terms: List[str]) -> str:
        """
        查询扩展：添加同义词和相关术语，扩展后的查询可以匹配更多使用不同术语但讨论相同主题的文档。
        提高了召回率（找到更多相关文档），但可能会降低精确度（此函数被LLM查询扩展代替）
        recall（召回）/ precision（精确率）是一对要互相让位的指标：多撒网少漏掉，但混进来的噪声也更多。

        Args:
            query: 原始查询 ["机器学习"]
            expansion_terms: 扩展词列表 ["深度学习", "神经网络", "AI"]

        Returns:
            扩展后的查询
        """
        # 简单的查询扩展策略：添加同义词或相关术语
        expanded = query
        for term in expansion_terms:
            if term.lower() not in query.lower():  # 检查该词是否已经在查询中（不区分大小写）
                expanded += f" {term}" # 如果不存在，则在查询后面添加该词（用空格分隔）
                
        # 用户查询："Python" →扩展后："Python 编程 代码 脚本"  
        return expanded

    def rewrite_query(self, query: str) -> List[str]:
        """
        手动查询改写：将一个查询改写成多种表达形式，让检索系统能匹配到更多相关文档（此函数被LLM查询扩展代替）

        Args:
            query: 原始查询

        Returns:
            查询变体列表
        """
        # 初始化变体列表：先把原始查询加入列表
        rewrites = [query]

        # 添加问句形式：例："机器学习是什么" → 添加 "机器学习是什么?"
        if not query.endswith('?') and not query.endswith('？'):
            rewrites.append(query + '?')

        # 添加陈述句形式：例："如何学习Python" → 添加 "学习Python"
        if '如何' in query or '怎么' in query:
            rewrites.append(query.replace('如何', '').replace('怎么', ''))

        # 添加关键词提取形式（移除无意义的停用词）：例："我想学习机器学习的基础" → 添加 "想学习机器学习的基础"（移除了“我”，“的”）
        # 注意：遍历一个字符串拿到的是「单字符」而不是「词」，所以这里是按字剔除停用字，不是按词；
        # 变量名叫 word（词）会有点误导。下面的 stopwords 里混着 '一个' 这类双字项，永远不会被匹配到。
        stopwords = ['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个']
        keywords = [word for word in query if word not in stopwords]
        if len(keywords) > 0:
            rewrites.append(' '.join(keywords))
            
        # set() 去除重复的变体，转回列表返回
        return list(set(rewrites))  # 去重

    def hybrid_search(
        self,
        query: str,
        semantic_results: List[Tuple],
        keyword_results: List[Tuple],
        alpha: float = 0.7
    ) -> List[Tuple]:
        """
        实现了混合检索策略，将语义检索和关键词检索的结果融合在一起
        （此函数未被调用：演示函数使用 纯语义检索（基于 Sentence-BERT + FAISS））
        混合检索 = 0.7 × 语义分数（归一化） + 0.3 × 关键词分数（归一化）

        hybrid retrieval（混合检索）= 向量检索 + 关键词检索一起用（术语表 D）。
        注意它融合的是「加权求和后的分数」，不是 RRF（倒数排名融合）；两种融合方式没有优劣，
        只是加权求和要求两路分数先归一化到可比的量纲上。
        另外文中「Sentence-BERT + FAISS」是历史描述：本仓库既没用 FAISS 也没装 sentence-transformers，
        实际向量库是 Chroma、向量由远程 API 生成。

        Args:
            query: 查询文本
            semantic_results: 语义检索结果（理解查询的语义含义，找到意思相近的文档） [(idx, score, chunk), ...]
            keyword_results: 关键词检索结果（精确匹配关键词，找到包含特定词汇的文档） [(idx, score, chunk), ...]
            alpha: 语义检索权重，0-1之间

        Returns:
            混合检索结果
        """
        # 用户查询："机器学习算法"                                                                                                           
        
        # 语义检索结果（理解语义，找相似意思的文档）：
        # semantic_results = [
        #     (0, 0.85, <Chunk对象A>),  # 文档0："深度学习是机器学习的重要分支"
        #     (2, 0.72, <Chunk对象C>),  # 文档2："神经网络算法的应用"
        #     (5, 0.68, <Chunk对象F>)   # 文档5："人工智能技术发展"
        # ]
        
        # 关键词检索结果（精确匹配关键词）：
        # keyword_results = [
        #     (1, 0.90, <Chunk对象B>),  # 文档1："机器学习算法包括决策树..."
        #     (2, 0.65, <Chunk对象C>),  # 文档2："神经网络算法的应用"
        #     (3, 0.50, <Chunk对象D>)   # 文档3："算法优化方法"
        # ]
        
        # 将结果列表转换为字典，方便通过索引快速查找分数，例：{0: 0.85, 2: 0.72, 5: 0.68}
        semantic_scores = {idx: score for idx, score, _ in semantic_results}
        keyword_scores = {idx: score for idx, score, _ in keyword_results}

        # 获取所有出现在任一检索结果中的文档索引: all_indices = {0, 1, 2, 3, 5}(文档块对应索引)
        all_indices = set(semantic_scores.keys()) | set(keyword_scores.keys())

        # 计算混合分数
        hybrid_results = []
        
        for idx in all_indices:
            sem_score = semantic_scores.get(idx, 0.0) # 获取语义检索分数
            kw_score = keyword_scores.get(idx, 0.0) # 获取关键词检索分数

            # 文档 0（只在语义检索中）
            # idx = 0
            # sem_score =  0.85；kw_score = 0
            # hybrid_score = 0.7 × 0.85 + 0.3 × 0.0 = 0.595
            hybrid_score = alpha * sem_score + (1 - alpha) * kw_score
            
            # 根据文档索引找到对应的文档块对象（chunk）：
            # 之前计算出了混合分数，但还需要找到对应的文档内容才能返回完整结果
            chunk = None
            for i, s, c in semantic_results: # 语义检索结果中查找:遍历 [(0, 0.85, <Chunk对象A>), (2, 0.72, <Chunk对象C>), ...]
                if i == idx:  # 如果索引匹配
                    chunk = c # 获取chunk对象
                    break     # 找到就停止
            if chunk is None:
                for i, s, c in keyword_results: # 第二个循环：从关键词检索结果中查找（备用）
                    if i == idx:
                        chunk = c
                        break

            if chunk is not None:
                hybrid_results.append((idx, hybrid_score, chunk))

        # 按分数排序
        hybrid_results.sort(key=lambda x: x[1], reverse=True)

        return hybrid_results

    def rerank_results(
        self,
        query: str,
        results: List[Tuple],
        top_k: Optional[int] = None
    ) -> List[Tuple]:
        """
        重排序：使用专门的重排序模型对结果进行精细排序

        如果配置了reranker，使用Cross-Encoder或规则重排序
        否则使用简单的相似度重排序

        cross-encoder（交叉编码器）把「问题 + 文档」拼在一起送进模型打分，比各算各的向量更准但更慢，
        所以只用来精排少量候选，不用于全库召回。

        Args:
            query: 查询文本
            results: 初始检索结果
            top_k: 返回top-k个结果

        Returns:
            重排序后的结果
        """
        if not results:
            return []

        # 使用配置的重排序器
        if self.reranker is not None:
            return self.reranker.rerank(query, results, top_k)

        # 后备方案：简单的相似度重排序
        logger.info("使用简单相似度重排序")

        # 提取文档文本
        doc_texts = [chunk.text if hasattr(chunk, 'text') else str(chunk)
                     for _, _, chunk in results]

        # 编码查询和文档
        query_embedding = self.encode_query(query)
        doc_embeddings = self.encode_texts(doc_texts)

        # 计算相似度
        # 这里直接拿 np.dot 当相似度用：只有当向量已是单位长度时，点积才等于 cosine similarity（余弦相似度）；
        # 否则点积会被向量长度影响。此处依赖「embedding 服务返回的向量已归一化」这个隐式前提。
        similarities = np.dot(doc_embeddings, query_embedding)

        # 重新排序
        reranked_results = []
        for i, (idx, old_score, chunk) in enumerate(results):
            new_score = float(similarities[i])
            reranked_results.append((idx, new_score, chunk))

        # 按新分数排序
        reranked_results.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            reranked_results = reranked_results[:top_k]

        return reranked_results

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的语义相似度

        Args:
            text1: 第一个文本
            text2: 第二个文本

        Returns:
            相似度分数 (0-1)
        """
        embeddings = self.encode_texts([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)

    def get_model_info(self) -> Dict:
        """
        获取模型信息

        Returns:
            模型信息字典
        """
        return {
            'model_type': 'api_embedding',
            'embedding_dim': self.embedding_dim,
            'provider': getattr(self.embedder, 'provider', 'unknown'),
            'model_name': getattr(self.embedder, 'model', 'unknown'),
        }

    def expand_query_with_llm(self, query: str, num_terms: int = 3) -> str:
        """
        追加相关词，丰富词汇(比如“机器学习” 查询扩展后：“人工智能、监督学习、模型训练”)
        (generator.generate() 就是调用商业 API 的入口，配置改为 GENERATOR_TYPE='local'，则 self.generator 变成 调用本地模型)

        这是上面 expand_query 那套规则扩展的 LLM 版本：generator（生成器）= 负责补词的 LLM 客户端，
        换掉了写死的同义词表。step 4 用 jieba 分词过滤掉与原查询重复的词，防止把原词又拼一遍。
        Args:
            query: 原始查询
            num_terms: 生成的扩展词数量

        Returns:
            扩展后的查询
        """
        if self.generator is None:
            # 事实说明：日志写「降级使用规则方法」，但此处只把原 query 原样返回，并没有调用 expand_query 那套规则扩展。
            # 文本与行为不符，此处仅记录，不改逻辑。
            logger.warning("未提供生成器，降级使用规则方法")
            return query

        prompt = f"""任务：为查询生成相关的扩展词，帮助检索更多相关文档。

查询：{query}

要求：
1. 生成{num_terms}个与查询主题紧密相关的扩展词或短语
2. 扩展词类型（优先级从高到低）：
   - 同义词和近义概念（如"向量数据库" → "向量索引" "FAISS"）
   - 相关技术（如"向量数据库" → "语义检索" "相似度搜索"）
   - 典型实现（如"向量数据库" → "Milvus" "Pinecone"）
3. 扩展词必须与原查询高度相关，不要生成太宽泛的词
4. 只输出词语，用空格分隔，不要编号、标点、解释

示例：
查询：什么是机器学习？
输出：人工智能 监督学习 模型训练

查询：向量数据库有什么作用？
输出：语义检索 相似度搜索 FAISS

查询：{query}
输出："""

        try:
            expansion = self.generator.generate(
                prompt=prompt,
                max_new_tokens=50,  # 增加长度以生成更多词
                temperature=0.5  # 提高温度增加多样性
            ).strip()

            # 清理生成结果
            # 1. 只取第一行（避免多行输出）
            expansion = expansion.split('\n')[0].strip()

            # 2. 移除编号和标点
            expansion = re.sub(r'^\d+[.、）)\s]+', '', expansion)  # 移除开头的编号
            expansion = re.sub(r'[，、,;；。！？]', ' ', expansion)  # 替换标点为空格

            # 3. 移除多余空格
            expansion = ' '.join(expansion.split())

            # 4. 智能过滤重复词（只过滤完全相同的多字词，保留组合词）
            # 使用jieba分词进行更准确的词语识别
            query_words = set(jieba.lcut(query.replace('？', '').replace('?', '')))
            # 移除停用词
            stop_words = {'的', '是', '了', '在', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',  '看', '好', '自己', '这', '什么', '怎么', '如何', '为什么'}
            # 只保留有意义的关键词（2字以上）
            query_keywords = {w for w in query_words - stop_words if len(w) >= 2}

            expansion_words = expansion.split()
            filtered_words = []
            for w in expansion_words:
                # 只过滤完全相同的多字词，允许包含原查询词的组合词
                if len(w) > 1 and w not in query_keywords:
                    filtered_words.append(w)

            # 如果过滤后为空，降低要求，保留所有2字以上的词
            if not filtered_words:
                filtered_words = [w for w in expansion_words if len(w) >= 2]

            # 5. 限制词数
            filtered_words = filtered_words[:num_terms]
            expansion = ' '.join(filtered_words)

            if expansion:
                expanded_query = f"{query} {expansion}"
                # 日志已在dispatcher中打印，此处不重复
                return expanded_query
            else:
                logger.warning("LLM 未生成有效扩展词，使用原始查询")
                return query

        except Exception as e:
            logger.error(f"LLM 查询扩展失败: {e}，降级使用原始查询")
            return query

    def rewrite_query_with_llm(self, query: str, num_rewrites: int = 3) -> List[str]:
        """
        改写成多种表达来实现多路检索（比如“如何学习” → “怎么学习”，这里未启用这个功能），
        (generator.generate() 就是调用商业 API 的入口，配置改为 GENERATOR_TYPE='local'，则 self.generator 变成 调用本地模型)

        多路检索 = 同一个问题生成几种说法，各自去检索，再把结果合并去重；
        命中面更宽，代价是检索次数按变体数翻倍。

        Args:
            query: 原始查询
            num_rewrites: 生成的改写数量

        Returns:
            查询变体列表（包含原始查询）
        """
        if self.generator is None:
            logger.warning("未提供生成器，降级使用规则方法")
            return self.rewrite_query(query)

        prompt = f"""将"{query}"改写成{num_rewrites}种表达。
要求：只输出改写，每行一个，不要编号和其他内容。

改写："""

        try:
            rewrites_text = self.generator.generate(
                prompt=prompt,
                max_new_tokens=80,  # 减少生成长度
                temperature=0.3  # 降低温度
            ).strip()

            # 解析生成的改写
            rewrites = [query]  # 始终包含原始查询

            lines = rewrites_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 移除编号（1. 2. 一、二、等）
                line = re.sub(r'^[\d一二三四五六七八九十]+[.、）)\s]+', '', line)
                line = line.strip()

                # 移除常见的解释性前缀
                line = re.sub(r'^(改写为|可以改写为|改写成|可以表达为|表达为)[：:]\s*', '', line)
                line = re.sub(r'根据.*?[，,]', '', line)  # 移除"根据...，"
                line = line.strip()

                # 截断：如果包含解释性文本，只取问号或句号前的部分
                if '？' in line:
                    line = line.split('？')[0] + '？'
                elif '?' in line:
                    line = line.split('?')[0] + '?'
                elif '。' in line and len(line.split('。')[0]) > 3:
                    line = line.split('。')[0]

                # 过滤无效改写
                if line and line != query and len(line) > 2 and len(line) < 100:
                    rewrites.append(line)

                # 达到数量限制就停止
                if len(rewrites) >= num_rewrites + 1:
                    break

            # 去重并限制数量
            rewrites = list(dict.fromkeys(rewrites))[:num_rewrites + 1]

            logger.info(f"查询改写: '{query}' -> {len(rewrites)} 个变体")
            return rewrites

        except Exception as e:
            logger.error(f"LLM 查询改写失败: {e}，降级使用规则方法")
            return self.rewrite_query(query)
