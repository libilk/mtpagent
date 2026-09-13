"""P2 根因定位测试(work.md §7 P2.5 / §5.5)。

这一组是项目的核心断言:构造已知掌握度分布,问系统"根因在哪",答案必须是对的那个。
"""

import pytest

from coach.knowledge import queries
from coach.knowledge.governance import Governance
from coach.knowledge.store import KnowledgeStore
from coach.domain.models import Edge, KnowledgePoint


@pytest.fixture
def chain(knowledge) -> KnowledgeStore:
    """一条链:base → mid → top 外加一个同级更弱的 sibling。"""
    for kp_id, name in [
        ("base", "基础"),
        ("mid", "中层"),
        ("top", "目标"),
        ("sibling", "同级同层短板"),
    ]:
        knowledge.upsert_concept(KnowledgePoint(id=kp_id, name=name, subject="test"))
    for a, b in [("base", "mid"), ("mid", "top"), ("sibling", "top")]:
        knowledge.upsert_edge(Edge(from_id=a, to_id=b))
    return knowledge


class TestPrereqClosure:
    def test_returns_shortest_depths(self, chain):
        assert queries.prereq_closure(chain, "top", depth=3) == {
            "mid": 1,
            "sibling": 1,
            "base": 2,
        }

    def test_empty_for_root(self, chain):
        assert queries.prereq_closure(chain, "base", depth=3) == {}


class TestDetectGaps:
    def test_only_below_threshold(self, chain):
        mastery = {"mid": 0.9, "sibling": 0.1, "base": 0.5}
        gaps = queries.detect_gaps(chain, mastery, "top", threshold=0.4)

        assert [g["kp_id"] for g in gaps] == ["sibling"]
        assert gaps[0]["depth"] == 1
        assert gaps[0]["mastery"] == pytest.approx(0.1)

    def test_sorted_by_depth_then_mastery(self, chain):
        mastery = {"base": 0.05, "mid": 0.2, "sibling": 0.3}
        gaps = queries.detect_gaps(chain, mastery, "top", threshold=0.4)

        # depth 1 的两个先出(mastery 升序),depth 2 的最后
        assert [g["kp_id"] for g in gaps] == ["mid", "sibling", "base"]

    def test_shallow_weak_ranks_before_deep_weaker(self, chain):
        """深度优先:更浅的 0.3 要排在更深的 0.05 前面。"""
        mastery = {"mid": 0.3, "sibling": 0.9, "base": 0.05}
        gaps = queries.detect_gaps(chain, mastery, "top", threshold=0.4)

        assert [g["kp_id"] for g in gaps] == ["mid", "base"]

    def test_empty_when_no_prereqs(self, chain):
        assert queries.detect_gaps(chain, {}, "base") == []

    def test_returns_name_not_just_id(self, chain):
        gaps = queries.detect_gaps(chain, {"mid": 0.1, "sibling": 0.9, "base": 0.9}, "top")
        assert gaps[0]["name"] == "中层"


class TestRootCauses:
    def test_includes_dependency_path(self, chain):
        mastery = {"mid": 0.9, "sibling": 0.9, "base": 0.1}
        roots = queries.root_causes(chain, mastery, "top", top_k=3)

        assert [r["kp_id"] for r in roots] == ["base"]
        assert roots[0]["path"] == ["top", "mid", "base"]

    def test_respects_top_k(self, chain):
        mastery = {"mid": 0.1, "sibling": 0.1, "base": 0.1}
        assert len(queries.root_causes(chain, mastery, "top", top_k=1)) == 1

    def test_empty_when_all_prereqs_mastered(self, chain):
        mastery = {"mid": 0.9, "sibling": 0.9, "base": 0.9}
        assert queries.root_causes(chain, mastery, "top") == []


class TestShortestPath:
    def test_self_path(self, chain):
        assert queries.shortest_prereq_path(chain, "top", "top") == ["top"]

    def test_multi_hop(self, chain):
        assert queries.shortest_prereq_path(chain, "top", "base") == ["top", "mid", "base"]

    def test_direct(self, chain):
        assert queries.shortest_prereq_path(chain, "top", "sibling") == ["top", "sibling"]

    def test_unreachable_returns_empty(self, chain):
        assert queries.shortest_prereq_path(chain, "base", "top") == []


class TestNextToLearn:
    def test_only_returns_nodes_whose_prereqs_are_ready(self, chain):
        # base 没掌握 → mid 还不能学;base 自己是叶子,可以学
        mastery = {"base": 0.1, "mid": 0.1, "sibling": 0.9}
        picked = queries.next_to_learn(chain, mastery, "top", top_k=5)

        assert [p["kp_id"] for p in picked] == ["base"]

    def test_skips_already_mastered(self, chain):
        mastery = {"base": 0.9, "mid": 0.1, "sibling": 0.9}
        picked = queries.next_to_learn(chain, mastery, "top", top_k=5)

        assert [p["kp_id"] for p in picked] == ["mid"]

    def test_prefers_closest_to_goal(self, chain):
        mastery = {"base": 0.8, "mid": 0.1, "sibling": 0.1}
        picked = queries.next_to_learn(chain, mastery, "top", top_k=5)

        # mid / sibling 都是 depth 1 且前置已备齐,排在 depth 2 的之前
        assert {p["kp_id"] for p in picked} == {"mid", "sibling"}
        assert all(p["depth"] == 1 for p in picked)


class TestExplainGapsTemplate:
    def test_mentions_goal_and_root_cause(self, chain):
        gaps = [{"kp_id": "base", "name": "基础", "depth": 2, "mastery": 0.1,
                 "path": ["top", "mid", "base"]}]
        text = queries.explain_gaps("top", "目标", 0.3, gaps)

        assert "目标" in text
        assert "基础" in text
        assert "top → mid → base" in text

    def test_no_gaps_message(self, chain):
        text = queries.explain_gaps("top", "目标", 0.9, [])
        assert "前置知识都已具备" in text
