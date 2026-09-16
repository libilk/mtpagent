"""对照组:P5.4 —— 不走图的「扁平召回」根因定位。

**为什么要有它:** 光说"我们的根因定位准"没有意义,得回答
「不用图、只用文本相似度能不能做到同样的事」。这个数字才是 §3.3 说的差异化。

**方法:** 把知识点当成纯文本(名字 + 描述 + id),用字符二元组余弦相似度
召回最相关的若干知识点,再在其中挑掌握度最低的当根因。

**公平性上做的三件事**(否则容易变成打稻草人):
1. 两边用**同一份掌握度**、同一个缺口阈值;
2. 两边都拿到同样宽度的候选池,差的只是「候选怎么选」—— 图用前置闭包,
   扁平用文本相似度。变量被隔离到这一点上;
3. 基线也允许看到目标知识点自己(和图的闭包不同,闭包不含自身)。

**已知的局限(会写进 Limitations):** 这是字面相似度,不是真向量。
真向量(embedding)语义泛化更好,可能强于本基线。要更严格的话应换成
Chroma + 真实 embedding —— `rag_core/chroma_store.py` 已具备,但那要联网调
embedding 服务,评测的可复现性会变差。当前版本选择**可复现优先**,并如实说明。
"""

from typing import Dict, List, Optional, Sequence

from coach import config
from coach.domain import text as text_utils


class FlatRetrievalBaseline:
    """纯文本相似度的根因定位。完全不认识 `edges` 表。"""

    def __init__(self, knowledge):
        self.knowledge = knowledge
        self._vectors: Dict[str, Counter] = {}
        self._texts: Dict[str, str] = {}
        for concept in knowledge.list_concepts():
            text = " ".join(
                filter(
                    None,
                    [concept.id, concept.name, concept.description or "", concept.subject or ""],
                )
            )
            self._texts[concept.id] = text
            self._vectors[concept.id] = text_utils.bigrams(text)

    def similar(self, kp_id: str, top_k: int = 5, include_self: bool = True) -> List[str]:
        """按文本相似度召回。`include_self=False` 用于只找"别处"的相关点。"""
        base = self._vectors.get(kp_id)
        if base is None:
            return []
        scored = []
        for other, vector in self._vectors.items():
            if other == kp_id and not include_self:
                continue
            scored.append((text_utils.cosine(base, vector), other))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [kp for _, kp in scored[:top_k]]

    def root_causes(
        self,
        mastery: Dict[str, float],
        kp_id: str,
        top_k: int = config.ROOT_CAUSE_TOP_K,
        threshold: float = config.GAP_THRESHOLD,
        pool: int = 12,
        observed: Optional[set] = None,
    ) -> List[Dict]:
        """和 knowledge.queries.root_causes 同形,但候选来自文本召回而非前置闭包。

        pool 给得比 top_k 宽,是刻意的:让基线有充分机会从大池子里挑出最弱的。

        observed:有作答记录的知识点集合。给了它就只看这些 ——
        **一个知识点从没被观测过,就没有证据说它弱**,不该把"未知"当成"最弱"。
        不给的话缺失值会落到 0.0,比任何真实观测值都低,基线会去挑一堆
        毫无证据的点(实测踩过,结果是 0.0 这种没法看的数字)。
        """
        candidates = self.similar(kp_id, top_k=pool + 1)
        gaps = []
        for candidate in candidates:
            if candidate == kp_id:
                continue
            if observed is not None and candidate not in observed:
                continue
            value = mastery.get(candidate, 0.0)
            if value < threshold:
                gaps.append({"kp_id": candidate, "mastery": value, "depth": 0})

        # 扁平方案没有 depth 概念,只能按掌握度排
        gaps.sort(key=lambda g: g["mastery"])
        return gaps[:top_k]
