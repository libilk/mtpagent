"""写入治理:observation → proposal → aggregator(work.md §4.4)。

核心立场:**LLM 的输出不是事实,是提案**。任何要进图的东西都要走这三步:
    1. observe()   记下原始信号(可能来自 LLM,也可能是人工)
    2. propose()   规则校验:自环 / 重复 / 置信度过低 / 引用不存在的知识点 → 直接 rejected
    3. aggregate() 环检测(DAG 校验)通过才写入 edges

被拒绝的提案留在 proposals 表里(status=rejected + reason),
这是可解释性的一部分:能回答"这条关系为什么没进图"。
"""

import json
import sqlite3
from collections import defaultdict
from typing import Any, Dict, List, Optional

from coach import config
from coach.domain.ids import new_ulid
from coach.domain.models import (
    EDGE_TYPES,
    Edge,
    KnowledgePoint,
    Observation,
    Proposal,
)

_CONCEPT_REQUIRED = ("id", "name", "subject")
_EDGE_REQUIRED = ("from_id", "to_id", "type")


def has_cycle(edges: List[tuple]) -> bool:
    """edges: [(from, to)] 仅 PREREQUISITE。DFS 三色标记(work.md §5.2)。"""
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = defaultdict(int)

    def dfs(u):
        color[u] = GRAY
        for v in graph[u]:
            if color[v] == GRAY:
                return True
            if color[v] == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    return any(color[u] == WHITE and dfs(u) for u in list(graph))


def _payload(row: sqlite3.Row) -> Dict[str, Any]:
    return json.loads(row["payload"]) if row["payload"] else {}


def _proposal_key(kind: str, payload: Dict[str, Any]):
    """提案的唯一键:边用三元组,知识点用 id。判重用。"""
    if kind == "edge":
        return (payload.get("from_id"), payload.get("to_id"), payload.get("type"))
    return payload.get("id")


def _row_to_observation(row: sqlite3.Row) -> Observation:
    return Observation(
        id=row["id"],
        kind=row["kind"],
        payload=json.loads(row["payload"]) if row["payload"] else {},
        source=row["source"],
        observed_at=row["observed_at"] if row["observed_at"] is not None else 0.0,
    )


def _row_to_proposal(row: sqlite3.Row) -> Proposal:
    return Proposal(
        id=row["id"],
        observation_id=row["observation_id"],
        kind=row["kind"],
        payload=json.loads(row["payload"]) if row["payload"] else {},
        status=row["status"],
        reason=row["reason"],
        created_at=row["created_at"] if row["created_at"] is not None else 0.0,
    )


class Governance:
    """三阶段写入治理。依赖 KnowledgeStore 读写图,自身只管流程与规则。"""

    def __init__(self, store):
        self.store = store
        self.conn = store.conn
        # 待审/已接受的提案 key 缓存。判重要查 proposals,而 payload 是 JSON,
        # 没法用 SQL 直接比 —— 每次全表扫会退化成 O(n²)(建种子图时明显)。
        # 缓存在实例内构建一次,之后靠 _insert_proposal 增量维护。
        self._open_keys: Optional[Dict[str, set]] = None

    # ---------------- 阶段 1:观察 ----------------

    def observe(self, kind: str, payload: Dict[str, Any], source: str = "manual") -> Observation:
        """记录一条原始观察(未验证)。kind ∈ {"edge", "concept"}。"""
        obs = Observation(id=new_ulid(), kind=kind, payload=payload, source=source)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO observations (id, kind, payload, source, observed_at)
                VALUES (:id, :kind, :payload, :source, :observed_at)
                """,
                {
                    "id": obs.id,
                    "kind": obs.kind,
                    "payload": json.dumps(obs.payload, ensure_ascii=False),
                    "source": obs.source,
                    "observed_at": obs.observed_at,
                },
            )
        return obs

    def get_observation(self, observation_id: str) -> Optional[Observation]:
        row = self.conn.execute(
            "SELECT * FROM observations WHERE id = ?", (observation_id,)
        ).fetchone()
        return _row_to_observation(row) if row else None

    # ---------------- 阶段 2:提案(规则校验)----------------

    def propose(self, observation_id: str) -> Proposal:
        """把观察变成提案。规则不通过的直接落 rejected,连同 reason。"""
        obs = self.get_observation(observation_id)
        if obs is None:
            raise KeyError(f"观察不存在:{observation_id}")

        reason = self._validate(obs)
        proposal = Proposal(
            id=new_ulid(),
            observation_id=obs.id,
            kind=obs.kind,
            payload=obs.payload,
            status="rejected" if reason else "pending",
            reason=reason,
        )
        self._insert_proposal(proposal)
        return proposal

    def _validate(self, obs: Observation) -> Optional[str]:
        """返回拒绝原因;None 表示通过。"""
        if obs.kind == "edge":
            return self._validate_edge(obs.payload)
        if obs.kind == "concept":
            return self._validate_concept(obs.payload)
        return f"未知的观察类型:{obs.kind}"

    def _validate_edge(self, payload: Dict[str, Any]) -> Optional[str]:
        missing = [k for k in _EDGE_REQUIRED if not payload.get(k)]
        if missing:
            return f"缺少必填字段:{','.join(missing)}"

        from_id, to_id, edge_type = payload["from_id"], payload["to_id"], payload["type"]
        if edge_type not in EDGE_TYPES:
            return f"未知关系类型:{edge_type}"
        if from_id == to_id:
            return "自环:起点与终点相同"

        confidence = payload.get("confidence", 1.0)
        if confidence < config.MIN_EDGE_CONFIDENCE:
            return f"置信度 {confidence} 低于阈值 {config.MIN_EDGE_CONFIDENCE}"

        if not self.store.has_concept(from_id):
            return f"引用不存在的知识点:{from_id}"
        if not self.store.has_concept(to_id):
            return f"引用不存在的知识点:{to_id}"

        if self.store.has_edge(from_id, to_id, edge_type):
            return "重复:该边已存在于图中"
        if self._has_open_proposal("edge", (from_id, to_id, edge_type)):
            return "重复:已有相同边在待审/已接受的提案中"
        return None

    def _validate_concept(self, payload: Dict[str, Any]) -> Optional[str]:
        missing = [k for k in _CONCEPT_REQUIRED if not payload.get(k)]
        if missing:
            return f"缺少必填字段:{','.join(missing)}"
        if self.store.has_concept(payload["id"]):
            return "重复:该知识点已存在于图中"
        if self._has_open_proposal("concept", payload["id"]):
            return "重复:已有相同知识点在待审/已接受的提案中"
        return None

    def _has_open_proposal(self, kind: str, key) -> bool:
        """同 key 的 pending/accepted 提案是否已存在。"""
        return key in self._load_open_keys()[kind]

    def _load_open_keys(self) -> Dict[str, set]:
        if self._open_keys is None:
            self._open_keys = {"edge": set(), "concept": set()}
            rows = self.conn.execute(
                "SELECT kind, payload FROM proposals WHERE status IN ('pending','accepted')"
            ).fetchall()
            for row in rows:
                kind = row["kind"]
                if kind not in self._open_keys:
                    continue
                self._open_keys[kind].add(_proposal_key(kind, _payload(row)))
        return self._open_keys

    # ---------------- 阶段 3:聚合入库 ----------------

    def aggregate(self, proposal_id: str) -> Proposal:
        """环检测通过则入库,否则 rejected。已处理过的提案原样返回(幂等)。"""
        proposal = self.get_proposal(proposal_id)
        if proposal is None:
            raise KeyError(f"提案不存在:{proposal_id}")
        if proposal.status != "pending":
            return proposal

        if proposal.kind == "edge":
            reason = self._aggregate_edge(proposal.payload)
        elif proposal.kind == "concept":
            reason = self._aggregate_concept(proposal.payload)
        else:
            reason = f"未知的提案类型:{proposal.kind}"

        proposal.status = "rejected" if reason else "accepted"
        proposal.reason = reason
        self._update_proposal_status(proposal)
        return proposal

    def _aggregate_edge(self, payload: Dict[str, Any]) -> Optional[str]:
        edge_type = payload["type"]
        if edge_type == "PREREQUISITE":
            candidate = self.store.prerequisite_pairs() + [
                (payload["from_id"], payload["to_id"])
            ]
            if has_cycle(candidate):
                return f"成环:引入 {payload['from_id']}→{payload['to_id']} 后前置依赖成环"

        self.store.upsert_edge(
            Edge(
                from_id=payload["from_id"],
                to_id=payload["to_id"],
                type=edge_type,
                weight=payload.get("weight", 1.0),
                confidence=payload.get("confidence", 1.0),
                source=payload.get("source"),
            )
        )
        return None

    def _aggregate_concept(self, payload: Dict[str, Any]) -> Optional[str]:
        if self.store.has_concept(payload["id"]):
            return "重复:该知识点已存在于图中"
        self.store.upsert_concept(
            KnowledgePoint(
                id=payload["id"],
                name=payload["name"],
                subject=payload["subject"],
                difficulty=payload.get("difficulty", 3.0),
                description=payload.get("description"),
            )
        )
        return None

    # ---------------- 查询辅助 ----------------

    def get_proposal(self, proposal_id: str) -> Optional[Proposal]:
        row = self.conn.execute(
            "SELECT * FROM proposals WHERE id = ?", (proposal_id,)
        ).fetchone()
        return _row_to_proposal(row) if row else None

    def pending_proposals(self) -> List[Proposal]:
        rows = self.conn.execute(
            "SELECT * FROM proposals WHERE status = 'pending' ORDER BY created_at, id"
        ).fetchall()
        return [_row_to_proposal(r) for r in rows]

    def list_proposals(self, status: Optional[str] = None) -> List[Proposal]:
        if status is None:
            rows = self.conn.execute(
                "SELECT * FROM proposals ORDER BY created_at, id"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM proposals WHERE status = ? ORDER BY created_at, id",
                (status,),
            ).fetchall()
        return [_row_to_proposal(r) for r in rows]

    def aggregate_pending(self) -> List[Proposal]:
        """把所有 pending 提案跑一遍 aggregator,返回处理后的提案。"""
        return [self.aggregate(p.id) for p in self.pending_proposals()]

    # ---------------- 落库细节 ----------------

    def _insert_proposal(self, proposal: Proposal) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO proposals (id, observation_id, kind, payload, status, reason, created_at)
                VALUES (:id, :observation_id, :kind, :payload, :status, :reason, :created_at)
                """,
                {
                    "id": proposal.id,
                    "observation_id": proposal.observation_id,
                    "kind": proposal.kind,
                    "payload": json.dumps(proposal.payload, ensure_ascii=False),
                    "status": proposal.status,
                    "reason": proposal.reason,
                    "created_at": proposal.created_at,
                },
            )
        if proposal.status in ("pending", "accepted"):
            self._load_open_keys()[proposal.kind].add(
                _proposal_key(proposal.kind, proposal.payload)
            )

    def _update_proposal_status(self, proposal: Proposal) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE proposals SET status = :status, reason = :reason WHERE id = :id",
                {
                    "id": proposal.id,
                    "status": proposal.status,
                    "reason": proposal.reason,
                },
            )
        # 被拒的提案不该继续占着判重位:等阻断它的那条边被删掉后,
        # 同一条关系应该还能重新提案
        if proposal.status == "rejected" and self._open_keys is not None:
            self._open_keys[proposal.kind].discard(
                _proposal_key(proposal.kind, proposal.payload)
            )
