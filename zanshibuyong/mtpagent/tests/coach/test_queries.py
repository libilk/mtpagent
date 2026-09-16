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


class TestObservedVersusUnobserved:
    """★ 两态区分:P5 评测里稀疏条件下图 0.4933 反低于随机 0.6267,根因就在这里。

    没被观测过的点,掌握度只是初始值,数值上可能比"考过且确实弱"的点还低。
    旧排序只看数值,于是把真正的弱项挤到后面。
    """

    def test_observed_gap_outranks_shallower_unobserved(self, chain):
        # mid 是 depth1、有证据、掌握度 0.3;base 是 depth2、从没测过、掌握度 0.1
        mastery = {"mid": 0.3, "sibling": 0.9, "base": 0.1}
        observed = {"mid", "sibling"}  # base 没观测过

        gaps = queries.detect_gaps(chain, mastery, "top", observed=observed)

        assert [g["kp_id"] for g in gaps] == ["mid", "base"]
        assert gaps[0]["status"] == "gap"
        assert gaps[1]["status"] == "unobserved"

    def test_without_observed_falls_back_to_old_order(self, chain):
        """不传 observed 就是旧行为,方便不关心两态的调用方。"""
        mastery = {"mid": 0.3, "sibling": 0.9, "base": 0.1}

        gaps = queries.detect_gaps(chain, mastery, "top")

        assert [g["kp_id"] for g in gaps] == ["mid", "base"]
        assert all(g["status"] == "gap" for g in gaps)

    def test_unobserved_still_reported_not_dropped(self, chain):
        """未观测的点不该被丢掉 —— 它是"该去测一下",不是"没问题"。"""
        mastery = {"mid": 0.1, "sibling": 0.1, "base": 0.1}
        observed = set()  # 什么都没测过

        gaps = queries.detect_gaps(chain, mastery, "top", observed=observed)

        assert {g["kp_id"] for g in gaps} == {"mid", "sibling", "base"}
        assert all(g["status"] == "unobserved" for g in gaps)

    def test_root_causes_passes_status_through(self, chain):
        mastery = {"mid": 0.2, "sibling": 0.9, "base": 0.1}
        roots = queries.root_causes(chain, mastery, "top", observed={"mid", "sibling"})

        assert roots[0]["kp_id"] == "mid"
        assert roots[0]["status"] == "gap"
        assert roots[-1]["status"] == "unobserved"

    def test_explain_distinguishes_confirmed_from_untested(self, chain):
        gaps = [
            {"kp_id": "mid", "name": "中层", "depth": 1, "mastery": 0.2,
             "status": "gap", "path": ["top", "mid"]},
            {"kp_id": "base", "name": "基础", "depth": 2, "mastery": 0.1,
             "status": "unobserved", "path": ["top", "mid", "base"]},
        ]

        text = queries.explain_gaps("top", "目标", 0.3, gaps)

        assert "已确认薄弱" in text
        assert "还没有作答记录" in text
        assert "中层" in text and "基础" in text


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

    def test_unobserved_prereq_does_not_block_but_flags_probe(self, chain):
        """★ 冷启动修正:前置**从没测过**不该当成"没准备好"直接卡死。

        旧行为下全新学生只能学叶子;现在未观测的前置不阻塞,但会标成 probe。
        """
        mastery = {"base": 0.1, "mid": 0.1, "sibling": 0.1}
        observed = {"sibling"}  # base / mid 从没测过

        picked = queries.next_to_learn(chain, mastery, "top", top_k=5, observed=observed)
        by_id = {p["kp_id"]: p for p in picked}

        assert "mid" in by_id, "前置 base 只是没测过,不该把 mid 卡死"
        assert by_id["mid"]["action"] == "probe"      # 前置没测过 → 先摸底
        assert by_id["base"]["action"] == "learn"     # 叶子,没有前置
        assert by_id["sibling"]["action"] == "learn"  # 也没有前置

    def test_confirmed_weak_prereq_still_blocks(self, chain):
        """前置**有证据且确实弱** → 仍然不能学,这是对的。"""
        mastery = {"base": 0.1, "mid": 0.2, "sibling": 0.9}
        observed = {"base", "mid", "sibling"}  # base 是已确认的弱项

        picked = queries.next_to_learn(chain, mastery, "top", top_k=5, observed=observed)

        assert "mid" not in [p["kp_id"] for p in picked]

    def test_without_observed_keeps_old_blocking_behaviour(self, chain):
        mastery = {"base": 0.1, "mid": 0.1, "sibling": 0.9}
        picked = queries.next_to_learn(chain, mastery, "top", top_k=5)
        assert [p["kp_id"] for p in picked] == ["base"]

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
