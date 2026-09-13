"""P0 知识图谱测试(work.md §7 P0.8)。

覆盖:环检测、自环、悬空引用、低置信度、重复、前置闭包(深度/去重/不含自身)、
被拒提案不入图、种子图无环。
"""

import pytest

from coach.knowledge import builder
from coach.knowledge.governance import Governance, has_cycle
from coach.knowledge.store import KnowledgeStore


@pytest.fixture
def store(tmp_path):
    s = KnowledgeStore.open(tmp_path / "coach.db")
    yield s
    s.close()


@pytest.fixture
def gov(store):
    return Governance(store)


def ingest(gov, kind, payload, source="test"):
    """跑完 observe → propose → aggregate,返回最终提案。"""
    observation = gov.observe(kind, payload, source=source)
    return gov.aggregate(gov.propose(observation.id).id)


def add_concept(gov, kp_id, subject="test"):
    return ingest(gov, "concept", {"id": kp_id, "name": kp_id, "subject": subject})


def add_edge(gov, from_id, to_id, edge_type="PREREQUISITE", confidence=1.0):
    return ingest(
        gov,
        "edge",
        {
            "from_id": from_id,
            "to_id": to_id,
            "type": edge_type,
            "confidence": confidence,
        },
    )


# ---------------------------------------------------------------- 环检测


class TestCycleDetection:
    def test_three_node_cycle_rejected(self, store, gov):
        for kp in ("A", "B", "C"):
            add_concept(gov, kp)
        assert add_edge(gov, "A", "B").status == "accepted"
        assert add_edge(gov, "B", "C").status == "accepted"

        proposal = add_edge(gov, "C", "A")

        assert proposal.status == "rejected"
        assert "成环" in proposal.reason
        # ★ 关键:被拒的边绝不能进图
        assert not store.has_edge("C", "A", "PREREQUISITE")
        assert len(store.list_edges()) == 2

    def test_self_loop_rejected(self, store, gov):
        add_concept(gov, "A")
        proposal = add_edge(gov, "A", "A")
        assert proposal.status == "rejected"
        assert "自环" in proposal.reason
        assert store.list_edges() == []

    def test_contrasts_may_form_pair(self, store, gov):
        """CONTRASTS 天然可以成对,不该被当环拒绝。"""
        add_concept(gov, "A")
        add_concept(gov, "B")
        assert add_edge(gov, "A", "B", "CONTRASTS").status == "accepted"
        assert add_edge(gov, "B", "A", "CONTRASTS").status == "accepted"

    def test_has_cycle_pure(self):
        assert has_cycle([("A", "B"), ("B", "C"), ("C", "A")]) is True
        assert has_cycle([("A", "B"), ("B", "C")]) is False
        assert has_cycle([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]) is False
        assert has_cycle([]) is False


# ---------------------------------------------------------------- 治理规则


class TestGovernanceRules:
    def test_dangling_reference_rejected(self, store, gov):
        """引用不存在的知识点必须拒绝(§4.4 第 4 条规则)。"""
        add_concept(gov, "A")
        proposal = add_edge(gov, "A", "NOT_THERE")
        assert proposal.status == "rejected"
        assert proposal.reason == "引用不存在的知识点:NOT_THERE"
        assert store.list_edges() == []

        proposal = add_edge(gov, "NOT_THERE", "A")
        assert proposal.status == "rejected"
        assert proposal.reason == "引用不存在的知识点:NOT_THERE"

    def test_low_confidence_rejected(self, store, gov):
        add_concept(gov, "A")
        add_concept(gov, "B")
        proposal = add_edge(gov, "A", "B", confidence=0.59)
        assert proposal.status == "rejected"
        assert "置信度" in proposal.reason

    def test_confidence_at_threshold_accepted(self, store, gov):
        add_concept(gov, "A")
        add_concept(gov, "B")
        assert add_edge(gov, "A", "B", confidence=0.6).status == "accepted"

    def test_duplicate_edge_rejected(self, store, gov):
        add_concept(gov, "A")
        add_concept(gov, "B")
        assert add_edge(gov, "A", "B").status == "accepted"
        proposal = add_edge(gov, "A", "B")
        assert proposal.status == "rejected"
        assert "重复" in proposal.reason

    def test_duplicate_concept_rejected(self, gov):
        assert add_concept(gov, "A").status == "accepted"
        proposal = add_concept(gov, "A")
        assert proposal.status == "rejected"
        assert "重复" in proposal.reason

    def test_unknown_relation_type_rejected(self, store, gov):
        add_concept(gov, "A")
        add_concept(gov, "B")
        proposal = add_edge(gov, "A", "B", edge_type="SUPPORTS")
        assert proposal.status == "rejected"
        assert "未知关系类型" in proposal.reason

    def test_rejected_proposal_is_persisted_with_reason(self, store, gov):
        """被拒的提案要留在库里,能回答'这条关系为什么没进图'。"""
        add_concept(gov, "A")
        proposal = add_edge(gov, "A", "A")

        stored = gov.get_proposal(proposal.id)
        assert stored.status == "rejected"
        assert stored.reason

    def test_aggregate_is_idempotent(self, store, gov):
        add_concept(gov, "A")
        add_concept(gov, "B")
        proposal = add_edge(gov, "A", "B")

        again = gov.aggregate(proposal.id)

        assert again.status == "accepted"
        assert len(store.list_edges()) == 1

    def test_observation_and_proposal_chain_persisted(self, store, gov):
        observation = gov.observe("concept", {"id": "X", "name": "X", "subject": "test"})
        proposal = gov.propose(observation.id)

        assert gov.get_observation(observation.id).id == observation.id
        assert proposal.observation_id == observation.id
        assert proposal.status == "pending"

        gov.aggregate(proposal.id)
        assert store.has_concept("X")


# ---------------------------------------------------------------- 前置闭包


class TestPrereqClosure:
    @pytest.fixture(autouse=True)
    def _build_diamond(self, gov):
        """菱形依赖:A→B→D、A→C→D(A 是所有节点的前置)。"""
        for kp in ("A", "B", "C", "D"):
            add_concept(gov, kp)
        add_edge(gov, "A", "B")
        add_edge(gov, "A", "C")
        add_edge(gov, "B", "D")
        add_edge(gov, "C", "D")

    def test_depth_is_shortest_path(self, store):
        assert store.ancestors("D", 3) == {"B": 1, "C": 1, "A": 2}

    def test_depth_bound_truncates(self, store):
        assert store.ancestors("D", 1) == {"B": 1, "C": 1}

    def test_excludes_self(self, store):
        assert "D" not in store.ancestors("D", 5)

    def test_dedup_when_reachable_via_multiple_paths(self, store):
        """A 能经 B、也能经 C 到达,只应出现一次。"""
        closure = store.ancestors("D", 3)
        assert list(closure).count("A") == 1
        assert closure["A"] == 2

    def test_leaf_has_empty_closure(self, store):
        assert store.ancestors("A", 3) == {}

    def test_descendants_mirror(self, store):
        assert store.descendants("A", 3) == {"B": 1, "C": 1, "D": 2}

    def test_chain_depth(self, gov, store):
        for kp in ("P", "Q", "R", "S"):
            add_concept(gov, kp)
        add_edge(gov, "P", "Q")
        add_edge(gov, "Q", "R")
        add_edge(gov, "R", "S")

        assert store.ancestors("S", 2) == {"R": 1, "Q": 2}
        assert store.ancestors("S", 5) == {"R": 1, "Q": 2, "P": 3}

    def test_related_edges_do_not_leak_into_prereq_traversal(self, gov, store):
        add_concept(gov, "M")
        add_concept(gov, "N")
        add_edge(gov, "M", "N", "RELATED")

        assert store.ancestors("N", 3) == {}
        assert store.ancestors("N", 3, edge_type="RELATED") == {"M": 1}


# ---------------------------------------------------------------- 种子图


class TestGoldenSeedGraph:
    @pytest.fixture
    def seeded(self, store, gov):
        report = builder.seed_graph(gov)
        return store, report

    def test_seed_ingests_without_rejection(self, seeded):
        _, report = seeded
        assert report.edges_rejected == 0
        assert report.concepts_rejected == 0
        assert report.edges_accepted > 0

    def test_seeded_graph_is_acyclic(self, seeded):
        store, _ = seeded
        assert has_cycle(store.prerequisite_pairs()) is False

    def test_seeded_graph_covers_all_relation_types(self, seeded):
        store, _ = seeded
        assert {e.type for e in store.list_edges()} == {
            "PREREQUISITE",
            "RELATED",
            "EXTENDS",
            "CONTRASTS",
        }

    def test_ancestors_of_dp(self, seeded):
        """P2 验收场景的前置:动态规划的前置闭包必须包含函数调用与递归。"""
        store, _ = seeded
        closure = store.ancestors("algo.dp", 3)

        assert closure["algo.recursion"] == 1
        assert closure["ds.array"] == 1
        assert closure["prog.func_call"] == 2

    def test_reseeding_is_idempotent(self, seeded, gov):
        store, _ = seeded
        before = len(store.list_edges())
        report = builder.seed_graph(gov)

        assert report.edges_accepted == 0
        assert len(store.list_edges()) == before

    def test_seed_limit_keeps_graph_consistent(self, store, gov):
        report = builder.seed_graph(gov, limit=15)
        assert report.edges_rejected == 0
        assert has_cycle(store.prerequisite_pairs()) is False
        # 被截掉的概念不该留下悬空边
        ids = {c.id for c in store.list_concepts()}
        for edge in store.list_edges():
            assert edge.from_id in ids and edge.to_id in ids
