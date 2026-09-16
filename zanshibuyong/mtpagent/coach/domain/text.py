"""轻量文本相似度。**不引依赖、不联网、结果可复现。**

两个地方要用它,所以抽到 domain 层(谁都能依赖,它不依赖任何人):
1. `evaluation/baselines.py` —— 扁平召回对照组
2. `knowledge/builder.py` —— 概念多了以后,从全量清单里召回相关概念喂给 LLM,
   而不是把上千个概念全塞进 prompt

用字符二元组而不是分词:中文短词用这个比按空格切更靠谱,而且零依赖。
**注意这是字面相似度,不是语义相似度** —— 同义词、上下位关系它看不出来。
"""

import math
from collections import Counter
from typing import Counter as CounterType, List, Optional, Sequence, Tuple


def bigrams(text: str) -> CounterType:
    """字符二元组计数。"""
    cleaned = "".join(str(text).split()).lower()
    if len(cleaned) < 2:
        return Counter([cleaned]) if cleaned else Counter()
    return Counter(cleaned[i : i + 2] for i in range(len(cleaned) - 1))


def cosine(a: CounterType, b: CounterType) -> float:
    """两个二元组向量的余弦相似度。"""
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    numerator = sum(a[token] * b[token] for token in common)
    denominator = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(
        sum(v * v for v in b.values())
    )
    return numerator / denominator if denominator else 0.0


def rank_by_similarity(
    query: str, candidates: Sequence[Tuple[str, str]], top_k: Optional[int] = None
) -> List[Tuple[float, str]]:
    """按与 query 的字面相似度给候选排序。

    Args:
        candidates: [(key, text)] —— text 是要比较的文本(通常 id + 名字 + 描述)

    Returns:
        [(score, key)],分数降序;同分按 key 排序保证**确定性**
    """
    base = bigrams(query)
    scored = [(cosine(base, bigrams(text)), key) for key, text in candidates]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[:top_k] if top_k else scored
