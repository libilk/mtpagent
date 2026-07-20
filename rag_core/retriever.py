"""
语义检索器模块
==============

使用 Sentence-BERT 模型将整句文本编码为向量，然后通过向量相似度找到最相关的文档
支持查询扩展、重排序和混合检索策略。
"""
import re
import logging
import jieba
import torch
import numpy as np
from typing import Any, List, Tuple, Optional, Dict
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

__all__ = ['SemanticRetriever']

# 导入重排序器（延迟导入避免循环依赖）
try:
    from .reranker import RerankerFactory

    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("重排序模块不可用")


class SemanticRetriever:
    """
    语义检索器：使用Sentence-BERT进行语义相似度检索

    支持的功能：
    1. 基于预训练模型的语义编码
    2. 查询扩展和改写
    3. 混合检索（语义+关键词）
    4. 结果重排序
    5. 批量编码优化
    """

    def __init__(
            self,  # 预训练模型路径
            model_path: str,  # GPU/CPU
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            max_length: int = 512,
            batch_size: int = 32,
            generator: Optional[Any] = None,  # 新增：可选的文本生成器，用于 LLM 查询优化
            enable_rerank: bool = False,  # 是否启用重排序
            reranker_type: str = 'cross_encoder',  # 重排序器类型
            reranker_model_path: Optional[str] = None  # 重排序模型路径
    ):
        """
        初始化语义检索器

        Args:
            model_path: 预训练模型路径
            device: 计算设备 ('cuda' 或 'cpu')
            max_length: 最大序列长度
            batch_size: 批处理大小
            generator: 可选的文本生成器（用于 LLM 查询优化）
            enable_rerank: 是否启用重排序
            reranker_type: 重排序器类型 ('cross_encoder', 'rule')
            reranker_model_path: 重排序模型路径
        """
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.generator = generator  # 保存生成器引用
        self.enable_rerank = enable_rerank

        logger.info(f"加载语义检索模型: {model_path}")
        logger.info(f"使用设备: {self.device}")

        # 加载tokenizer和模型（只使用本地模型，不连接 HuggingFace）
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

        # 获取模型的隐藏层维度
        self.embedding_dim = self.model.config.hidden_size

        logger.info(f"模型加载完成，嵌入维度: {self.embedding_dim}")

        # 初始化重排序器
        self.reranker = None
        if enable_rerank and RERANKER_AVAILABLE:
            logger.info(f"初始化重排序器: {reranker_type}")
            try:
                if reranker_type == 'cross_encoder' and reranker_model_path:
                    self.reranker = RerankerFactory.create_reranker(
                        reranker_type='cross_encoder',
                        model_path=reranker_model_path,
                        device=device,
                        max_length=max_length
                    )
                else:
                    # 使用规则重排序
                    self.reranker = RerankerFactory.create_reranker(
                        reranker_type='rule'
                    )
                logger.info("重排序器初始化成功")
            except Exception as e:
                logger.error(f"重排序器初始化失败: {e}")
                self.reranker = None
        elif enable_rerank and not RERANKER_AVAILABLE:
            logger.warning("重排序功能已启用，但reranker模块不可用")

    # mean_pooling：将变长序列（不同句子长度不同）转换为固定维度的句子向量，后续用于计算句子相似度
    def mean_pooling(
            self,
            model_output: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        平均池化：对token embeddings进行平均，生成句子embedding

        Args:
            model_output: 模型输出的token embeddings
            attention_mask: 注意力掩码

        Returns:
            句子embedding
        """
        # 获取token embeddings
        #      model_output[0] = [
        #     [[0.12, -0.34, ..., 0.78],   # [CLS] 的768维向量
        #      [0.23, 0.45, ..., 0.34],    # "我" 的768维向量
        #      [-0.11, 0.67, ..., -0.45],  # "爱" 的768维向量
        #      [0.34, -0.23, ..., 0.56],   # "机器" 的768维向量
        #      [0.45, 0.12, ..., 0.67],    # "学习" 的768维向量
        #      [0.56, -0.45, ..., 0.23],   # [SEP] 的768维向量
        #      [0.00, 0.00, ..., 0.00],    # [PAD] 的768维向量
        #      [0.00, 0.00, ..., 0.00]]    # [PAD] 的768维向量
        # ]
        token_embeddings = model_output[0]

        # 扩展attention_mask以匹配token_embeddings的维度
        # 原始: [1, 1, 1, 1, 1, 1, 0, 0]  shape: [1, 8]
        # 扩展后: shape: [1, 8, 768]
        # 每个 token 的 768 维都复制相同的 mask 值
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # 对有效token进行求和:将 padding 位置的向量置零

        # Token向量举例:  Mask:    结果:
        # [0.1, 0.2]  *  1    =  [0.1, 0.2]
        # [0.3, 0.4]  *  1    =  [0.3, 0.4]
        # [0.5, 0.6]  *  1    =  [0.5, 0.6]
        # [0.2, 0.1]  *  0    =  [0.0, 0.0]  ← padding被屏蔽
        # 求和:                  [0.9, 1.2]
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

        #  统计每个句子有多少个有效token:（本例中是6个）
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # 平均池化，因为模型输出是每个 token 的向量，但实际需要整个句子的向量：# [0.9, 1.2] / 6 = [0.15, 0.2]
        return sum_embeddings / sum_mask

    def encode_texts(
            self,
            texts: List[str],
            show_progress: bool = False
    ) -> np.ndarray:
        """
        将知识库的文本列表批量转换为向量矩阵

        Args:
            texts: 文本列表
            show_progress: 是否显示进度

        Returns:
            向量矩阵 (n_texts, embedding_dim)

        输入: ["我爱机器学习", "深度学习很有趣", ...]
        输出: [[0.12, -0.34, ...], [0.45, 0.23, ...], ...]  # shape: (n, 768)
        """
        #  空列表返回空矩阵
        if not texts:
            return np.empty((0, self.embedding_dim))

        all_embeddings = []

        # 批量处理（每次32个文本）
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]  # texts[i:i + 32]  # 取一批

            # Tokenize（文本 → 数字）
            # 结果: {'input_ids': [[101, 2769, ...]], 'attention_mask': [[1, 1, ...]]}
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,  # 补齐到相同长度
                truncation=True,  # 超长截断
                max_length=self.max_length,  # 最大长度
                return_tensors='pt'  # 返回PyTorch张量
            )

            # 移动到 GPU/CPU
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            # 前向传播: 来自 model_path 指向的预训练模型文件
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # 池化得到的句子embedding
            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

            # 将每个向量归一化为单位向量（长度=1）（用于余弦相似度）
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            # 转为 NumPy 并收集
            all_embeddings.append(embeddings.cpu().numpy())

            if show_progress and (i + self.batch_size) % 100 == 0:
                logger.info(f"已编码 {min(i + self.batch_size, len(texts))}/{len(texts)} 个文本")

        # texts = ["我爱机器学习", "深度学习很有趣", "NLP很强大"]
        # embeddings = retriever.encode_texts(texts)

        # embeddings.shape = (3, 768)
        # embeddings[0] = [0.12, -0.34, ..., 0.78]  # "我爱机器学习"的向量
        # embeddings[1] = [0.45, 0.23, ..., -0.12] # "深度学习很有趣"的向量
        # embeddings[2] = [-0.11, 0.67, ..., 0.34] # "NLP很强大"的向量
        # 合并所有批次: [[第一句的向量], [第二句的向量], ...] （用预训练的模型转换的）
        return np.vstack(all_embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        """
        用户问题转化为对应向量：这个嵌入向量可以与文档的嵌入向量进行比较，找到语义相似的内容

        Args:
            query: 查询文本：

        Returns:
            查询向量 (embedding_dim,)
        """
        # 比如query = "什么是机器学习"
        # encode_texts 返回：[[0.12, -0.34, ..., 0.78]]（二维数组，只有1行）
        embedding = self.encode_texts([query])

        # embedding[0] = [0.12, -0.34, ..., 0.78]
        return embedding[0]

    def expand_query(self, query: str, expansion_terms: List[str]) -> str:
        """
        查询扩展：添加同义词和相关术语，扩展后的查询可以匹配更多使用不同术语但讨论相同主题的文档。
        提高了召回率（找到更多相关文档），但可能会降低精确度（此函数被LLM查询扩展代替）

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
                expanded += f" {term}"  # 如果不存在，则在查询后面添加该词（用空格分隔）

        # 用户查询："Python" →扩展后："Python 编程 代码 脚本"
        return expanded

    def rewrite_query(self, query: str) -> List[str]:
        """
        手动查询改写：将一个查询改写成多种表达形式，
        、让检索系统能匹配到更多相关文档（此函数被LLM查询扩展代替）

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
            sem_score = semantic_scores.get(idx, 0.0)  # 获取语义检索分数
            kw_score = keyword_scores.get(idx, 0.0)  # 获取关键词检索分数

            # 文档 0（只在语义检索中）
            # idx = 0
            # sem_score =  0.85；kw_score = 0
            # hybrid_score = 0.7 × 0.85 + 0.3 × 0.0 = 0.595
            hybrid_score = alpha * sem_score + (1 - alpha) * kw_score

            # 根据文档索引找到对应的文档块对象（chunk）：
            # 之前计算出了混合分数，但还需要找到对应的文档内容才能返回完整结果
            chunk = None
            for i, s, c in semantic_results:  # 语义检索结果中查找:遍历 [(0, 0.85, <Chunk对象A>), (2, 0.72, <Chunk对象C>), ...]
                if i == idx:  # 如果索引匹配
                    chunk = c  # 获取chunk对象
                    break  # 找到就停止
            if chunk is None:
                for i, s, c in keyword_results:  # 第二个循环：从关键词检索结果中查找（备用）
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
            'model_type': self.model.config.model_type,
            'embedding_dim': self.embedding_dim,
            'max_length': self.max_length,
            'device': self.device,
            'vocab_size': self.model.config.vocab_size
        }

    def expand_query_with_llm(self, query: str, num_terms: int = 3) -> str:
        """
        追加相关词，丰富词汇(比如“机器学习” 查询扩展后：“人工智能、监督学习、模型训练”)
        (generator.generate() 就是调用商业 API 的入口，配置改为 GENERATOR_TYPE='local'，则 self.generator 变成 调用本地模型)
        Args:
            query: 原始查询
            num_terms: 生成的扩展词数量

        Returns:
            扩展后的查询
        """
        if self.generator is None:
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
            stop_words = {'的', '是', '了', '在', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很',
                          '到', '说', '要', '去', '你', '会', '着', '看', '好', '自己', '这', '什么', '怎么', '如何',
                          '为什么'}
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
