# -*- coding: utf-8 -*-
"""
上下文压缩器
============

压缩检索到的文档，减少LLM输入token

【调用状态】**唯一调用方是降级路径**：knowledge_agent/agent.py:1252
（`_handle_with_optimizations` 第 3 步），且那里写死 `use_llm=False` ——
线上真正跑的只有规则版 `_compress_with_rules`；基于 LLM 的 `_compress_with_llm`
**全仓库无调用方**（它更省 token，但每次压缩要多一次 LLM 调用）。

【取舍一句话】压缩是省 token（词元：LLM 处理文本的最小单位，也是计费与长度单位）
的手段，代价是**可能丢信息**：规则版只留"命中查询词"的句子，那些不命中字面、
但人读起来是必要前提的句子会被整句删掉，答案就可能缺依据。省下的钱和丢掉的信息
得自己权衡 —— 这也是它被放在降级路径、而不是主路径的原因之一。
"""

import logging
from typing import List, Dict
import re

logger = logging.getLogger(__name__)


class ContextCompressor:
    """
    上下文压缩器

    策略：
    1. 提取与查询相关的句子
    2. 过滤无关信息
    3. 保留关键上下文

    两条实现路线，差别在"谁来判相关"：规则版靠词面命中（便宜、但中文几乎失效，
    见 _calculate_relevance）、LLM 版靠模型改写（贵、但真能读懂语义）。
    """

    def __init__(self, llm=None, max_length: int = 2000):
        """
        初始化上下文压缩器

        Args:
            llm: LLM实例（用于智能压缩）
            max_length: 压缩后的最大长度 —— 注意实际是按**每篇文档**分别卡的，
                        不是所有文档加起来的总长度（见 _compress_with_rules）
        """
        self.llm = llm
        self.max_length = max_length

        logger.info(f"上下文压缩器初始化完成 (max_length={max_length})")

    def compress(
        self,
        query: str,
        documents: List[Dict],
        use_llm: bool = False
    ) -> List[Dict]:
        """
        压缩文档上下文

        Args:
            query: 查询文本
            documents: 文档列表
            use_llm: 是否使用LLM压缩

        Returns:
            压缩后的文档列表
        """
        if not documents:
            return documents

        logger.info(f"[上下文压缩] 压缩 {len(documents)} 个文档")

        # 计算原始长度
        original_length = sum(len(doc.get("content", "")) for doc in documents)

        if use_llm and self.llm:
            compressed_docs = self._compress_with_llm(query, documents)
        else:
            compressed_docs = self._compress_with_rules(query, documents)

        # 计算压缩后长度
        compressed_length = sum(len(doc.get("content", "")) for doc in compressed_docs)

        # compression_ratio 是"省掉的比例"（1 - 压缩后/原始），0.7 = 省了 70%，
        # 不是"还剩 70%" —— 名字容易反着读
        compression_ratio = (1 - compressed_length / original_length) if original_length > 0 else 0

        logger.info(f"[上下文压缩] 原始长度: {original_length}, 压缩后: {compressed_length}, 压缩率: {compression_ratio:.1%}")

        return compressed_docs

    def _compress_with_rules(
        self,
        query: str,
        documents: List[Dict]
    ) -> List[Dict]:
        """
        基于规则的压缩

        策略：
        1. 提取包含查询关键词的句子
        2. 保留句子的上下文（前后各一句）—— 防止抽出来的句子变成断章
        3. 按相关性排序 —— 指**挑选顺序**（分高的先入选、先占满长度配额），
           最终输出会按原文顺序重排回去，保证行文连贯、LLM 读得通

        ⚠️ 本方法在中文上会失效：相关性判定见 _calculate_relevance 的说明，
        中文句子的得分几乎恒为 0，因此 selected_indices 常常为空、
        compressed_content 被置成空串。而降级路径传的正是 use_llm=False，
        所以"压缩"这一步实际可能把文档清空。

        Args:
            query: 查询文本
            documents: 文档列表

        Returns:
            压缩后的文档列表
        """
        compressed_docs = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for doc in documents:
            content = doc.get("content", "")

            # 按句子分割
            sentences = self._split_sentences(content)

            # 计算每个句子的相关性分数
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                score = self._calculate_relevance(sentence, query_words)
                sentence_scores.append((i, sentence, score))

            # 按分数排序
            sentence_scores.sort(key=lambda x: x[2], reverse=True)

            # 选择top句子（按上面的相关性降序，分高的先占名额）
            selected_indices = set()
            for i, sentence, score in sentence_scores:
                if score > 0:  # 只选择有相关性的句子
                    # 添加当前句子及其上下文
                    selected_indices.add(i)
                    if i > 0:
                        selected_indices.add(i - 1)  # 前一句
                    if i < len(sentences) - 1:
                        selected_indices.add(i + 1)  # 后一句

                # 检查长度限制
                # 每轮重算一次已选总长；超了就停手（注意配额是"每篇文档"各算各的）。
                # 另外本轮刚加进来的"前后各一句"是在检查之前放进去的，所以最终长度
                # 有可能略微超出 max_length —— 是近似控制，不是硬上限
                current_length = sum(len(sentences[idx]) for idx in selected_indices)
                if current_length > self.max_length:
                    break

            # 按原始顺序重组句子
            selected_sentences = [sentences[i] for i in sorted(selected_indices)]
            compressed_content = "".join(selected_sentences)

            # 创建压缩后的文档
            compressed_doc = doc.copy()
            compressed_doc["content"] = compressed_content
            compressed_doc["compressed"] = True

            compressed_docs.append(compressed_doc)

        return compressed_docs

    def _compress_with_llm(
        self,
        query: str,
        documents: List[Dict]
    ) -> List[Dict]:
        """
        基于LLM的压缩

        让LLM提取与查询相关的关键信息

        与规则版的区别：规则版只做"删句"，LLM 版是"改写式提取"——可以把三句话
        的意思压成一句，压缩率更高，也更不怕中文。代价是每篇文档多一次 LLM 调用
        （而且失败时会原样退回未压缩文档，见下方 except）。
        现状：全仓库无调用方（降级路径写死 use_llm=False）。

        Args:
            query: 查询文本
            documents: 文档列表

        Returns:
            压缩后的文档列表
        """
        compressed_docs = []

        for doc in documents:
            content = doc.get("content", "")

            # 如果内容已经很短，不需要压缩
            if len(content) <= 500:
                compressed_docs.append(doc)
                continue

            try:
                prompt = f"""请从以下文档中提取与查询相关的关键信息，保持原文表达，不要添加额外内容。

查询：{query}

文档：
{content}

提取的关键信息："""

                compressed_content = self.llm.generate(prompt, temperature=0.3, max_tokens=500)

                # 创建压缩后的文档
                compressed_doc = doc.copy()
                compressed_doc["content"] = compressed_content.strip()
                compressed_doc["compressed"] = True

                compressed_docs.append(compressed_doc)

            except Exception as e:
                logger.error(f"LLM压缩失败: {e}")
                # 失败时使用原文档
                compressed_docs.append(doc)

        return compressed_docs

    def _split_sentences(self, text: str) -> List[str]:
        """
        分割句子

        Args:
            text: 文本

        Returns:
            句子列表
        """
        # 按句号、问号、感叹号分割
        # 正则加括号 = 让标点本身也留在结果里，这样下面才能把"句子+标点"拼回去
        sentences = re.split(r'([。！？.!?]+)', text)

        # 重新组合句子和标点
        # ⚠️ 步长 2 的配对循环意味着**最后一段没标点结尾的残句会被丢掉**：
        #    "A。B" → 只得到 "A。"，"B" 消失；而以标点收尾的文本不受影响。
        #    检索回来的 chunk 常是截断的，正好踩在这个坑上（已实测确认）。
        combined = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                combined.append(sentences[i] + sentences[i + 1])
            else:
                combined.append(sentences[i])

        # 过滤空句子
        combined = [s.strip() for s in combined if s.strip()]

        return combined

    def _calculate_relevance(self, sentence: str, query_words: set) -> float:
        """
        计算句子与查询的相关性

        Args:
            sentence: 句子
            query_words: 查询词集合

        Returns:
            相关性分数
        """
        # ⚠️ 这整套打分是英文思路：靠空格切词。中文句子没有空格，
        #    split() 会把整句当成一个"词"，于是与查询词的交集几乎恒为空、
        #    得分恒为 0（已实测："本店支持七天无理由退货。" vs "退货政策是什么" → 0.0）。
        #    后果见 _compress_with_rules：选不出句子，content 被清空。
        #    要修得先解决中文分词（jieba 之类），不是调阈值能救的。
        sentence_lower = sentence.lower()
        sentence_words = set(sentence_lower.split())

        # 计算交集
        intersection = query_words & sentence_words

        # 相关性分数 = 匹配词数 / 查询词数（命中越多分越高，0~1）
        score = len(intersection) / len(query_words) if query_words else 0

        return score
