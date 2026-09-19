"""
BM25关键词检索器模块
===================

实现基于BM25算法的关键词检索。
支持中文分词、停用词过滤和多种评分策略。

BM25（关键词检索算法）= 按词频与稀有度打分，擅长**精确字符串**匹配。

为什么它必须和向量检索并存：embedding（向量化）擅长"意思相近"，但对型号、
订单号、条款编号这类精确串反而不牢靠 —— 向量会把 "X100" 和 "X200" 看成近义。
所以检索链路里专门留一路纯词面匹配，两路结果再融合（见 hybrid_retriever.py）。
这也解释了本文件为何不做语义理解：它存在的价值就是"不懂意思、只认字面"。

本文件是这一路关键词检索的独立实现，只依赖 jieba（中文分词库）与词频统计，
不做任何模型推理。语料库由 enhanced_entry 从 ChromaDB 原文一次性灌入后 fit()。
"""

import math
import jieba
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)

__all__ = ['BM25Retriever']


class BM25Retriever:
    """
    BM25检索器：基于BM25算法的关键词检索

    BM25是一种基于概率的信息检索排序函数，广泛应用于搜索引擎。
    相比TF-IDF，BM25对词频进行了饱和处理，避免了长文档的不公平优势。

    retriever（检索器）= 负责 recall（召回）—— 从全库里把可能相关的捞出来的那一步。
    本类是关键词那一路的检索器，只按字面命中打分，不含任何语义判断。

    两个参数各解决一个反直觉问题，这才是 BM25 比"直接数词频"高明的地方：
    - k1 饱和：同一个词出现 10 次并不比 5 次重要一倍，词频收益要递减；
    - b 归一化：长文档天然容易命中更多词，必须按篇幅打折，否则长文档永远赢。

    支持的功能：
    1. 中文分词（基于jieba）
    2. 停用词过滤
    3. BM25评分算法
    4. 可调参数（k1, b）
    5. 批量检索

    算法参数：
    - k1: 控制词频饱和度，通常在1.2-2.0之间，默认1.5
    - b: 控制文档长度归一化，0-1之间，默认0.75
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        use_stopwords: bool = True,
        custom_stopwords: Optional[Set[str]] = None
    ):
        """
        初始化BM25检索器

        Args:
            k1: BM25参数k1，控制词频饱和度（默认1.5）
            b: BM25参数b，控制文档长度归一化（默认0.75）
            use_stopwords: 是否使用停用词过滤
            custom_stopwords: 自定义停用词集合
        """
        self.k1 = k1
        self.b = b
        self.use_stopwords = use_stopwords

        # 停用词设置
        self.stopwords = self._load_stopwords() if use_stopwords else set()
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)

        # 文档统计信息
        self.corpus = []  # 原始文档列表
        self.corpus_size = 0  # 文档总数
        self.avgdl = 0  # 平均文档长度
        self.doc_freqs = defaultdict(int)  # 词文档频率: term -> df
        self.idf = {}  # IDF值: term -> idf
        self.doc_len = []  # 每个文档的长度
        self.doc_tokens = []  # 每个文档的分词结果

        logger.info(f"初始化BM25检索器 (k1={k1}, b={b}, 停用词={use_stopwords})")

    def _load_stopwords(self) -> Set[str]:
        """
        加载停用词表

        Returns:
            停用词集合
        """
        # 常用中文停用词（可扩展）
        stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '个',
            '们', '中', '来', '为', '能', '对', '与', '于', '之', '及',
            '或', '但', '而', '等', '以', '因为', '所以', '如果', '这样',
            '什么', '怎么', '如何', '哪里', '为什么', '吗', '吧', '呢', '啊'
        }
        return stopwords

    def tokenize(self, text: str) -> List[str]:
        """
        文本分词和预处理

        这里只做**词面切分**，不做同义替换或语义归一：BM25 的匹配单位就是切出来的
        token（词元），分词质量直接决定关键词路线能否命中（中文尤其吃这一步）。

        Args:
            text: 输入文本

        Returns:
            分词后的token列表
        """
        # 使用jieba分词
        tokens = jieba.lcut(text.lower())

        # 过滤停用词和空白字符
        if self.use_stopwords:
            tokens = [t for t in tokens if t.strip() and t not in self.stopwords]
        else:
            tokens = [t for t in tokens if t.strip()]

        return tokens

    def fit(self, corpus: List[str]):
        """
        在语料库上训练BM25模型

        corpus（语料库）= 全部 chunk（文本块）的原文集合，由调用方一次性灌入。
        这里的"训练"只是数词频、建统计，不涉及模型参数，所以很快、也能随时重建
        （文档一变就得重新 fit，否则 IDF 与平均长度会失真）。

        Args:
            corpus: 文档块的纯文本内容
        """
        logger.info(f"开始训练BM25模型，文档数: {len(corpus)}")
        # 1. 保存语料库基本信息
        self.corpus = corpus  # 保存原始文档
        self.corpus_size = len(corpus) # 文档总数
        self.doc_len = [] # 每个文档的长度
        self.doc_tokens = [] # 每个文档的分词结果
        self.doc_freqs = defaultdict(int) # 词文档频率统计

        # 分词并统计
        for doc in corpus:
            tokens = self.tokenize(doc) # 对文档调用jieba分词
            self.doc_tokens.append(tokens) # 保存分词结果
            self.doc_len.append(len(tokens)) # 保存文档长度

            # 统计词文档频率（每个文档只计数一次）
            unique_tokens = set(tokens)  # 去重，确保每个词在一个文档中只计数一次（ IDF 统计的是"多少个文档包含这个词"，而不是"这个词总共出现多少次"）
            for token in unique_tokens:
                self.doc_freqs[token] += 1 # 记录包含该词的文档数

        # 计算平均文档长度：用于 BM25 公式中的长度归一化：短文档中出现的词更重要，长文档中出现的词要打折
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0

        # 调用 _calc_idf() 方法 - 计算 IDF 值（越稀有的词，区分度越高）
        self._calc_idf()
        # 训练后的数据结构
        # self.corpus = [原始文档列表]
        # self.corpus_size = 3
        # self.doc_tokens = [每个文档的分词列表]
        # self.doc_len = [4, 5, 5]
        # self.avgdl = 4.67
        # self.doc_freqs = {"机器": 2, "学习": 2, ...}
        # self.idf = {"机器": 0.470, "学习": 0.470, "人工智能": 0.981, ...}
        # 训练完成后，就可以调用 search() 或 retrieve() 进行检索了。        

        logger.info(f"BM25训练完成 - 文档数: {self.corpus_size}, "
                   f"平均长度: {self.avgdl:.1f}, "
                   f"唯一词汇: {len(self.doc_freqs)}")

    def _calc_idf(self):
        """
        计算所有词的IDF值，返回idf

        IDF（逆文档频率）= 一个词出现在多少篇文档里。出现得越少，区分度越高，
        所以拿它当权重 —— 这正对应"精确串"的价值：型号/编号往往只在少数文档出现。
        注意 df 统计的是"包含该词的文档数"，不是出现总次数（见 fit 里的去重）。
        IDF公式: log((N - df + 0.5) / (df + 0.5) + 1)
        其中N是文档总数，df是包含该词的文档数
        """
        self.idf = {} # 初始化空字典，用于存储每个词的IDF值

        for term, df in self.doc_freqs.items():
            # 遍历所有词及其文档频率，term: 词语（如"机器"、"学习"）
            # self.corpus_size：文档总数 N： df：包含该词的文档数，公式：log((N - df + 0.5) / (df + 0.5) + 1)
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
            
            # self.idf = {
            # '机器': 0.87,
            # '算法': 1.38,
            # ...
            # }
            # - df=2 的词（如"机器"、"学习"）→ IDF ≈ 0.875（出现在2篇文档中）
            # - df=1 的词（大部分词）→ IDF ≈ 1.386（只出现在1篇文档中）
            # - 出现越少的词，IDF值越高，检索时权重越大
            self.idf[term] = idf

    def get_scores(self, query: str) -> np.ndarray:
        """
        计算查询与所有文档的BM25分数

        query（查询）= 用户问的那句话。这里把它分词后，逐个词累加对每篇文档的贡献：
        查询里没在语料库出现过的词会被跳过（下面 if token not in self.idf）。

        Args:
            query: 查询文本

        Returns:
            一维数组，长度等于文档总数，每个元素是对应文档的 BM25 分数
        """
        # 查询分词："机器学习和深度学习" → ["机器", "学习", "深度"]（自动过滤停用词（如"和"））
        query_tokens = self.tokenize(query)

        # 初始化分数数组：创建一个全零数组，长度等于文档总数
        scores = np.zeros(self.corpus_size)

        # 遍历查询中的每个词（如果词不在 IDF 字典中（说明语料库中没有这个词），跳过）
        for token in query_tokens:
            
            # 语料库中的词及对应idf值：idf(出现越少的词，IDF值越高，检索时权重越大) = {
            # '机器': 0.87,
            # '算法': 1.38,..
            # }
            if token not in self.idf:
                continue

            # 获取该词的IDF:逆文档频率:出现在越少文档中的词，IDF 值越高
            idf = self.idf[token]

            #  计算该词对每个文档的贡献分数
            for doc_idx in range(self.corpus_size):
                # 计算词频
                doc_tokens = self.doc_tokens[doc_idx]
                tf = doc_tokens.count(token) # 词频：该词在文档中出现的次数
                # 文档不包含该词，跳过
                if tf == 0:
                    continue 

                # BM25评分：计算查询与文档集合的相关性得分
                # 获取当前文档的词数（token数量）,例：doc_len = 150（该文档有150个词）
                doc_len = self.doc_len[doc_idx]
                # 计算分子: tf： 词频:词在文档中出现越多，得分越高
                numerator = tf * (self.k1 + 1)
                # 计算分母（长度归一化）:doc_len / self.avgdl  # 文档长度相对于平均长度的比例:针对超长的文档（需要惩罚）
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))

                scores[doc_idx] += idf * (numerator / denominator)
                
        # scores = [1.924, 1.850, 0.0, 0.0, 0.0]
        # # 文档0和文档1都包含"机器学习"，得分最高： 返回一维numpy数组，长度等于文档总数，每个元素是对应文档的BM25分数
        # ⚠️ 反直觉点：BM25 分数没有上界，也没有跨语料的可比性（换一批文档分数就变），
        # 所以它只能用来**排序**，别当成"相关性百分比"去读或去卡阈值。
        return scores

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        return_scores: bool = True
    ) -> List[Tuple[int, float]]:
        """
        从语料库中检索与查询最相关的 top-k 个文档（按照get_scores得到的分数），返回文档索引和分

        top_k（取前 k 条）= 只保留得分最高的 k 个结果。本方法返回 (索引, 分数) 而非文本，
        文本由上层 search() 再映射 —— 这样上层还能自己重排或按元数据过滤。

        Args:
            query: 查询文本（用户的问题或搜索词）
            top_k: 返回top-k个结果
            return_scores: 是否返回分数

        Returns:
            [(doc_idx, score), ...]，按分数降序排列
        """
        #  get_scores() 方法计算用户问题与语料库中所有文档的 BM25 相关性分数
        scores = self.get_scores(query)

        # 获取top-k索引
        top_indices = np.argsort(scores)[::-1][:top_k]

        # 构建结果
        if return_scores:
            results = [(int(idx), float(scores[idx])) for idx in top_indices]
        else:
            results = [(int(idx), 0.0) for idx in top_indices]
        # results：[(2, 3.1), (0, 2.5), (3, 1.2)]
        # 即文档 2 最相关（分数 3.1），其次是文档 0（分数 2.5），然后是文档 3（分数 1.2）
        return results

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        检索相关文档，返回 (文本, 分数) 格式

        Args:
            query: 查询文本
            top_k: 返回top-k个结果

        Returns:
            [(doc_text, score), ...]，按分数降序排列
        """
        # 先调用 retrieve 获取 (doc_idx, score)
        idx_results = self.retrieve(query, top_k=top_k, return_scores=True)

        # 转换为 (doc_text, score) 格式
        results = []
        for idx, score in idx_results:
            if idx < len(self.corpus):
                results.append((self.corpus[idx], score))

        return results

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[List[Tuple[int, float]]]:
        """
        批量检索

        Args:
            queries: 查询列表
            top_k: 每个查询返回top-k个结果

        Returns:
            每个查询的检索结果列表
        """
        results = []
        for query in queries:
            result = self.retrieve(query, top_k=top_k)
            results.append(result)

        return results

    def get_top_terms(self, query: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        获取查询中最重要的词（按IDF排序）

        Args:
            query: 查询文本
            top_n: 返回top-n个词

        Returns:
            [(term, idf), ...]，按IDF降序排列
        """
        tokens = self.tokenize(query)

        # 获取每个词的IDF
        term_idfs = []
        for token in set(tokens):
            if token in self.idf:
                term_idfs.append((token, self.idf[token]))

        # 按IDF排序
        term_idfs.sort(key=lambda x: x[1], reverse=True)

        return term_idfs[:top_n]

    def explain_score(self, query: str, doc_idx: int) -> Dict:
        """
        解释某个文档的BM25分数计算过程

        Args:
            query: 查询文本
            doc_idx: 文档索引

        Returns:
            分数解释字典
        """
        query_tokens = self.tokenize(query)
        doc_tokens = self.doc_tokens[doc_idx]
        doc_len = self.doc_len[doc_idx]

        explanation = {
            'doc_idx': doc_idx,
            'doc_length': doc_len,
            'avg_doc_length': self.avgdl,
            'query_tokens': query_tokens,
            'term_scores': {},
            'total_score': 0.0
        }

        total_score = 0.0

        for token in query_tokens:
            if token not in self.idf:
                explanation['term_scores'][token] = {
                    'tf': 0,
                    'idf': 0,
                    'score': 0
                }
                continue

            tf = doc_tokens.count(token)
            idf = self.idf[token]

            # 计算该词的得分
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
            score = idf * (numerator / denominator)

            explanation['term_scores'][token] = {
                'tf': tf,
                'df': self.doc_freqs[token],
                'idf': round(idf, 4),
                'score': round(score, 4)
            }

            total_score += score

        explanation['total_score'] = round(total_score, 4)

        return explanation

    def get_statistics(self) -> Dict:
        """
        获取BM25统计信息

        Returns:
            统计信息字典
        """
        return {
            'corpus_size': self.corpus_size,
            'avg_doc_length': round(self.avgdl, 2),
            'vocabulary_size': len(self.doc_freqs),
            'total_tokens': sum(self.doc_len),
            'k1': self.k1,
            'b': self.b,
            'use_stopwords': self.use_stopwords,
            'stopwords_count': len(self.stopwords)
        }


# ==================== 使用示例 ====================
if __name__ == '__main__':
    # 初始化BM25检索器
    bm25 = BM25Retriever(k1=1.5, b=0.75)

    # 示例语料库
    corpus = [
        "机器学习是人工智能的一个重要分支",
        "深度学习是机器学习的一个子领域",
        "自然语言处理技术在搜索引擎中广泛应用",
        "计算机视觉可以识别图像中的物体",
        "推荐系统使用协同过滤算法"
    ]

    # 训练
    bm25.fit(corpus)

    # 检索
    query = "机器学习和深度学习"
    results = bm25.retrieve(query, top_k=3)

    print(f"查询: {query}")
    print(f"\nTop-3结果:")
    for idx, score in results:
        print(f"  文档{idx}: {corpus[idx][:30]}... (分数: {score:.4f})")

    # 查看IDF值
    print(f"\nIDF值 (按从高到低排序):")
    for term, idf in sorted(bm25.idf.items(), key=lambda x: x[1], reverse=True):
        print(f"  '{term}': {idf:.4f} (df={bm25.doc_freqs[term]})")

    # 解释分数
    print(f"\n分数解释 (文档0):")
    explanation = bm25.explain_score(query, 0)
    print(f"  总分: {explanation['total_score']}")
    print(f"  词项得分:")
    for term, info in explanation['term_scores'].items():
        print(f"    {term}: TF={info['tf']}, IDF={info['idf']}, Score={info['score']}")
