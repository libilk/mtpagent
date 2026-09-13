"""LLM id 漂移的同名归并测试。

实测背景:让 qwen-plus 从一段教材文本抽关系,它给归并排序起 id
`algo.merge_sort`,而人工金标准是 `algo.mergesort`。治理原本只按 id 判重,
于是同一个概念会在图里长出两个节点,前置闭包被污染。

修法分两头:
- 源头:抽取时把已有概念清单喂给模型,要求复用 id
- 兜底:治理层按**名字**归并 —— 同名提案不新增节点,引用别名的边改指规范 id
"""

import pytest

from coach.domain.models import KnowledgePoint
from coach.knowledge import builder
from coach.knowledge.governance import Governance, normalize_name
from coach.knowledge.store import KnowledgeStore


@pytest.fixture
def gov(knowledge) -> Governance:
    return Governance(knowledge)


def run(gov, kind, payload, source="llm_extract"):
    return gov.aggregate(gov.propose(gov.observe(kind, payload, source=source).id).id)


class TestNormalizeName:
    def test_strips_whitespace_and_lowercases(self):
        assert normalize_name("  归并 排序 ") == "归并排序"
        assert normalize_name("Merge Sort") == "mergesort"

    def test_none_and_empty(self):
        assert normalize_name(None) == ""
        assert normalize_name("") == ""


class TestAliasMerging:
    @pytest.fixture(autouse=True)
    def _seed_canonical(self, gov):
        """图里先有金标准的 algo.mergesort / 归并排序。"""
        run(gov, "concept", {"id": "algo.mergesort", "name": "归并排序", "subject": "algorithms"})
        run(gov, "concept", {"id": "algo.dp", "name": "动态规划", "subject": "algorithms"})

    def test_same_name_with_different_id_is_rejected(self, gov, knowledge):
        proposal = run(
            gov, "concept", {"id": "algo.merge_sort", "name": "归并排序", "subject": "algorithms"}
        )

        assert proposal.status == "rejected"
        assert "同名知识点已存在" in proposal.reason
        assert "algo.mergesort" in proposal.reason
        # ★ 图里不该长出第二个节点
        assert knowledge.get_concept("algo.merge_sort") is None
        assert len(knowledge.list_concepts()) == 2

    def test_edge_using_alias_is_rewritten_to_canonical_id(self, gov, knowledge):
        """别名先被见过(概念提案),后续引用它的边要落到规范 id 上。"""
        run(gov, "concept", {"id": "algo.merge_sort", "name": "归并排序", "subject": "algorithms"})

        proposal = run(
            gov, "edge",
            {"from_id": "algo.dp", "to_id": "algo.merge_sort", "type": "PREREQUISITE",
             "confidence": 0.9},
        )

        assert proposal.status == "accepted"
        assert proposal.payload["to_id"] == "algo.mergesort"
        assert knowledge.has_edge("algo.dp", "algo.mergesort", "PREREQUISITE")
        assert not knowledge.has_edge("algo.dp", "algo.merge_sort", "PREREQUISITE")

    def test_observation_keeps_the_original_llm_id(self, gov):
        """提案被归一化,但 observation 里保留 LLM 的原始输出,便于追溯。"""
        observation = gov.observe(
            "concept",
            {"id": "algo.merge_sort", "name": "归并排序", "subject": "algorithms"},
        )

        proposal = gov.aggregate(gov.propose(observation.id).id)

        assert proposal.payload["id"] == "algo.merge_sort"  # 提案记录的仍是原样
        assert proposal.status == "rejected"
        assert gov.get_observation(observation.id).payload["id"] == "algo.merge_sort"

    def test_genuinely_new_concept_still_inserted(self, gov, knowledge):
        proposal = run(
            gov, "concept", {"id": "algo.trie", "name": "前缀树", "subject": "data_structures"}
        )

        assert proposal.status == "accepted"
        assert knowledge.has_concept("algo.trie")

    def test_newly_inserted_concept_joins_name_index_immediately(self, gov, knowledge):
        """同一批里先新建、后重复提交,第二次也要被判成同名。"""
        first = run(gov, "concept", {"id": "algo.trie", "name": "前缀树", "subject": "ds"})
        assert first.status == "accepted"

        second = run(gov, "concept", {"id": "ds.trie", "name": "前缀树", "subject": "ds"})

        assert second.status == "rejected"
        assert "归并到 algo.trie" in second.reason
        assert len(knowledge.list_concepts()) == 3

    def test_edge_with_unknown_alias_still_rejected(self, gov):
        """没见过的 id 且图里没有,仍然按悬空引用拒绝。"""
        proposal = run(
            gov, "edge",
            {"from_id": "algo.dp", "to_id": "algo.nope", "type": "PREREQUISITE",
             "confidence": 0.9},
        )

        assert proposal.status == "rejected"
        assert "引用不存在的知识点" in proposal.reason

    def test_different_names_are_not_merged(self, gov, knowledge):
        """名字不同就不是同一个概念,不能被误并。"""
        proposal = run(
            gov, "concept", {"id": "algo.quicksort", "name": "快速排序", "subject": "algorithms"}
        )

        assert proposal.status == "accepted"
        assert knowledge.has_concept("algo.quicksort")


class TestExtractionPromptCarriesKnownIds:
    @staticmethod
    def _capturing_llm(captured):
        class CapturingLLM:
            def chat_structured(self, messages, tools=None, **_kwargs):
                captured["messages"] = messages
                from llm.llm_client import LLMResponse

                return LLMResponse(content='{"concepts": [], "edges": []}')

        return CapturingLLM()

    def test_prompt_lists_existing_concepts(self, knowledge):
        knowledge.upsert_concept(
            KnowledgePoint(id="algo.mergesort", name="归并排序", subject="algorithms")
        )
        captured = {}

        builder.extract(
            self._capturing_llm(captured),
            "归并排序是分治的经典应用",
            subject="algorithms",
            known_concepts=knowledge.list_concepts(),
        )

        prompt = captured["messages"][-1]["content"]
        assert "请优先复用" in prompt
        assert "algo.mergesort=归并排序" in prompt

    def test_prompt_has_no_listing_without_known_concepts(self):
        captured = {}

        class CapturingLLM:
            def chat_structured(self, messages, tools=None, **_kwargs):
                captured["messages"] = messages
                from llm.llm_client import LLMResponse

                return LLMResponse(content='{"concepts": [], "edges": []}')

        builder.extract(CapturingLLM(), "文本")

        assert "请优先复用" not in captured["messages"][-1]["content"]

    def test_real_extraction_then_ingest_does_not_duplicate(self, knowledge):
        """端到端:模拟 LLM 抽到别名 → 全流程走一遍 → 图里没有重复节点。"""
        g = Governance(knowledge)
        builder.seed_graph(g)
        before = len(knowledge.list_concepts())

        llm_output = {
            "concepts": [
                {"id": "algo.merge_sort", "name": "归并排序", "subject": "algorithms"},
                {"id": "sorts.quicksort", "name": "快速排序", "subject": "algorithms"},
            ],
            # ds.array → 归并排序 这条金标准里没有,应该能落库(且端点被归一化)
            "edges": [
                {"from_id": "ds.array", "to_id": "algo.merge_sort",
                 "type": "PREREQUISITE", "confidence": 0.95},
            ],
        }
        concepts, edges = builder._parse_extraction(llm_output)
        report = builder.ingest(g, concepts, edges, source="llm_extract")

        assert len(knowledge.list_concepts()) == before, "不该新增任何节点"
        assert report.concepts_rejected == 2  # 两个都是别名
        assert report.edges_accepted == 1
        assert knowledge.has_edge("ds.array", "algo.mergesort", "PREREQUISITE")
        assert not knowledge.has_edge("ds.array", "algo.merge_sort", "PREREQUISITE")
        assert report.rejections[0][2].startswith("同名知识点已存在")

    def test_alias_edge_matching_an_existing_edge_is_deduplicated(self, knowledge):
        """别名归并之后如果撞上已有边,照样算重复 —— 归并和判重是串起来的。"""
        g = Governance(knowledge)
        builder.seed_graph(g)
        before = len(knowledge.list_edges())

        # 金标准里已有 algo.recursion → algo.mergesort,这里换成 LLM 的别名写法
        concepts, edges = builder._parse_extraction(
            {
                "concepts": [],
                "edges": [
                    {"from_id": "algo.recursion", "to_id": "algo.merge_sort",
                     "type": "PREREQUISITE", "confidence": 0.9},
                ],
            }
        )
        # 先让别名进别名表(靠一次概念提案)
        builder.ingest(
            g,
            [{"id": "algo.merge_sort", "name": "归并排序", "subject": "algorithms"}],
            [],
            source="llm_extract",
        )
        report = builder.ingest(g, concepts, edges, source="llm_extract")

        assert report.edges_accepted == 0
        assert "重复" in report.rejections[0][2]
        assert len(knowledge.list_edges()) == before
