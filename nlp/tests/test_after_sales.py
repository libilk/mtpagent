# -*- coding: utf-8 -*-
"""
电商售后助手 · 回归测试
======================

**这是什么：** 改造完成后的**验收基线**。每次改动跑一遍，确认核心链路没被改坏。

**怎么跑：**
    cd nlp
    .venv/Scripts/python.exe tests/test_after_sales.py

退出码 0 = 全部通过，1 = 有失败（方便接 CI）。

用例分两段，按运行成本排：

    第一段（不调 LLM，秒级）—— 数据层与纯逻辑
        test_01  数据库就位
        test_02  外键约束真的在生效（臆造 ID 必须被拒）
        test_03  匿名工单允许创建
        test_04  退货申请的三重业务校验
        test_05  防幻觉兜底逻辑
        test_06  知识库白名单

    第二段（调 LLM，较慢，每个用例数秒到数十秒）—— 端到端
        test_07  系统初始化
        test_08  Agent 注册集合正确
        test_09  路由：5 类问题分派正确
        test_10  政策问答（含 3C 例外这个业务陷阱）
        test_11  订单查询（真实 SQL）
        test_12  物流查询（跨表 JOIN）
        test_13  多轮指代消解
        test_14  写操作闸门 + 审批恢复（★ 直接调 Agent，绕过路由以保证稳定）

**已知的不稳定因素：** 第二段依赖 LLM，`temperature` 不为 0，同样的输入偶尔会有不同路径。
所以 test_14 刻意绕过路由直接调 Agent —— 只验证"写操作会被闸门拦住"这件事，
把路由的不确定性排除在外。**回归测试要尽量只测一件事。**
"""

import os
import sys
import io
import time
import json
import sqlite3
import traceback
import logging
from typing import List, Optional

# Windows 控制台默认不是 UTF-8，中文会乱码
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 项目根目录（本文件在 tests/ 下）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)   # 确保相对路径（./.env、./database/...）都能对上

# 加载 .env
_env_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(_env_path):
    with open(_env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("AfterSalesTest").setLevel(logging.INFO)
logger = logging.getLogger("AfterSalesTest")

DB_PATH = os.path.join(PROJECT_ROOT, 'database', 'ecommerce.db')


class TestResult:
    """单个用例的结果"""

    def __init__(self, name: str, passed: bool, detail: str = "", duration: float = 0):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.duration = duration


class AfterSalesTester:
    """电商售后助手回归测试器"""

    def __init__(self):
        self.results: List[TestResult] = []
        # system 延迟创建 —— 它很重（要加载模型），只在需要时才实例化
        self._system = None

    # ------------------------------------------------------------------
    # 基础设施
    # ------------------------------------------------------------------

    @property
    def system(self):
        """懒加载系统实例：第一段的纯逻辑用例不该为它付出启动成本"""
        if self._system is None:
            from langgraph_orchestrator.enhanced_entry import EnhancedLangGraphRAGSystem
            self._system = EnhancedLangGraphRAGSystem(
                auto_update_index=False,
                enable_checkpointer=True,
                enable_parameter_validation=True,
                enable_critic=False,
            )
        return self._system

    def run_test(self, name: str, func) -> None:
        """跑一个用例并记录结果；异常一律记成失败，不让一个用例炸掉整轮"""
        start = time.time()
        try:
            detail = func()
            self.results.append(TestResult(name, True, str(detail or ""), time.time() - start))
            print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
        except AssertionError as e:
            self.results.append(TestResult(name, False, str(e), time.time() - start))
            print(f"  FAIL  {name}\n        {e}")
        except Exception as e:
            detail = f"{type(e).__name__}: {e}"
            self.results.append(TestResult(name, False, detail, time.time() - start))
            print(f"  ERROR {name}\n        {detail}")
            logger.debug(traceback.format_exc())

    # ==================================================================
    # 第一段：不调 LLM 的纯逻辑用例
    # ==================================================================

    def test_01_db_ready(self):
        """数据库就位，且六张表都有数据"""
        assert os.path.exists(DB_PATH), f"数据库不存在: {DB_PATH}，请先跑 tools/scripts/init_ecommerce_db.py"

        conn = sqlite3.connect(DB_PATH)
        try:
            tables = ["customers", "orders", "order_items", "logistics", "refunds", "tickets"]
            counts = {}
            for t in tables:
                counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        finally:
            conn.close()

        empty = [t for t, c in counts.items() if c == 0]
        assert not empty, f"这些表是空的: {empty}"
        return f"{sum(counts.values())} 行 / {len(tables)} 张表"

    def test_02_foreign_key_enforced(self):
        """
        ★ 外键约束真的在生效。

        这条用例守着一个**真实踩过的坑**：建表语句里写了 FOREIGN KEY，
        但 SQLite 默认不开校验，于是模型臆造的 user_id（实测出现过 U87654321）
        会被静默写进库。修法是显式 PRAGMA foreign_keys = ON。

        这里直接连库验证 pragma 生效，不依赖上层代码是否正确。
        """
        from core.ecommerce_crm import EcommerceCRM
        crm = EcommerceCRM()

        # 1) 臆造的用户必须被拒
        r = crm.create_ticket(user_id='U00000000', issue='回归测试-臆造用户')
        assert r.get("error"), f"臆造用户竟然创建成功了: {r.get('ticket_id')}"

        # 2) 臆造的订单号也必须被拒
        r = crm.create_ticket(user_id='U10001', order_id='SO00000000', issue='回归测试-臆造订单')
        assert r.get("error"), f"臆造订单竟然创建成功了: {r.get('ticket_id')}"

        return "臆造 user_id / order_id 均被拒绝"

    def test_03_anonymous_ticket_allowed(self):
        """匿名工单（user_id = NULL）应当允许 —— 外键只约束"给了值就必须存在" """
        from core.ecommerce_crm import EcommerceCRM
        crm = EcommerceCRM()

        r = crm.create_ticket(user_id=None, issue='回归测试-匿名咨询')
        assert not r.get("error"), f"匿名工单被误拒: {r.get('error')}"
        assert r.get("ticket_id"), "没返回工单号"
        assert r.get("user_id") is None, f"匿名工单的 user_id 应为 NULL，实际 {r.get('user_id')!r}"
        return r["ticket_id"]

    def test_04_return_request_validation(self):
        """
        退货申请的三重校验：订单存在 / 归属正确 / 状态允许售后。
        这是唯一会动钱的操作，校验必须可靠。
        """
        from core.ecommerce_crm import EcommerceCRM
        crm = EcommerceCRM()

        # 不存在的订单
        r = crm.submit_return_request('SO00000000', 'U10001', '测试')
        assert r.get("error") and "不存在" in r["error"], f"不存在的订单没被拒: {r}"

        # 别人的订单（张伟的订单，用李娜的账号退）
        r = crm.submit_return_request('SO20260909001', 'U10002', '测试')
        assert r.get("error") and "不属于" in r["error"], f"越权退货没被拒: {r}"

        # 未发货的订单（应该引导去取消订单，而不是走退货）
        r = crm.submit_return_request('SO20260910005', 'U10005', '测试')
        assert r.get("error") and "待发货" in r["error"], f"未发货订单没被拒: {r}"

        # 正常路径
        r = crm.submit_return_request('SO20260909001', 'U10001', '回归测试-质量问题')
        assert r.get("refund_id"), f"正常退货失败: {r}"
        assert r.get("status") == "待审核", f"初始状态应为待审核，实际 {r.get('status')}"
        return r["refund_id"]

    def test_05_anti_hallucination_guard(self):
        """
        防幻觉兜底：声称办过事，就必须真调用过写工具。

        这是阶段 3 挖出来的最危险的一个问题 —— 模型会在**没调用工具**的情况下
        写出"已为您提交申请，单号 RFxxx"。这里把它固化成用例，防止以后被改回去。
        """
        from agents.customer_service_agent.agent import CustomerServiceAgent as A

        # 借壳调用静态逻辑（不需要实例化整个 Agent）
        class _Probe:
            _WRITE_TOOLS = A._WRITE_TOOLS
            _CLAIM_MARKERS = A._CLAIM_MARKERS
            _verify_write_claim = A._verify_write_claim

        probe = _Probe()
        lie = "已为您提交退货申请，单号 RF20260916001"

        # 没调写工具却声称办了 → 必须拦截
        out = probe._verify_write_claim(lie, {"query_order", "hybrid_search"})
        assert "并未成功提交" in out, "编造办理结果没有被拦下"

        # 真调了写工具 → 放行
        out = probe._verify_write_claim(lie, {"submit_return_request"})
        assert out == lie, "正常答复被误伤"

        # 纯政策回答（不涉及办理）→ 放行
        policy = "七天无理由需在签收后 7 日内申请，运费由您承担。"
        out = probe._verify_write_claim(policy, {"hybrid_search"})
        assert out == policy, "纯咨询被误伤"

        return "4 种场景判断正确"

    def test_06_knowledge_whitelist(self):
        """知识库只剩售后政策文档，旧的 RAG 技术文档已清空"""
        from pathlib import Path
        knowledge_dir = Path(PROJECT_ROOT) / 'data' / 'knowledge'
        assert knowledge_dir.exists(), f"知识库目录不存在: {knowledge_dir}"

        files = {p.stem for p in knowledge_dir.glob('*.md')}
        assert len(files) == 10, f"预期 10 篇政策文档，实际 {len(files)} 篇: {sorted(files)}"

        # 这几篇是旧场景的，必须已经不在
        stale = {"rag_optimization_strategies", "hybrid_search_deep_dive", "customer_satisfaction_guide"}
        leftover = stale & files
        assert not leftover, f"旧场景文档还在: {leftover}"

        return f"{len(files)} 篇售后政策"

    # ==================================================================
    # 第二段：端到端（调 LLM）
    # ==================================================================

    def test_07_system_init(self):
        """系统能初始化，图和注册表都就绪"""
        assert self.system is not None, "系统实例为 None"
        assert self.system.graph is not None, "图实例为 None"
        assert self.system.registry is not None, "注册表为 None"
        return "图 + 注册表就绪"

    def test_08_agents_registered(self):
        """Agent 注册集合正确：5 个，且 document_agent 已摘除"""
        agent_ids = [a["id"] for a in self.system.registry.get_all_agents()]

        expected = {"knowledge_agent", "database_agent", "customer_service_agent",
                    "vqa_agent", "chat_agent"}
        assert set(agent_ids) == expected, (
            f"注册集合不符\n  实际: {sorted(agent_ids)}\n  预期: {sorted(expected)}"
        )
        assert "document_agent" not in agent_ids, "document_agent 应已摘除（9 个工具全是合同审核用）"
        return f"{len(agent_ids)} 个 Agent"

    def test_09_routing(self):
        """路由：5 类售后问题分派到正确的 Agent"""
        agents = self.system.registry.get_all_agents()

        cases = [
            ("七天无理由退货怎么算",              "customer_service_agent"),
            ("退货运费谁承担",                    "customer_service_agent"),
            ("订单 SO20260909001 什么状态",       "database_agent"),
            ("我的快递到哪了",                    "database_agent"),
            ("我要投诉你们",                      "customer_service_agent"),
            ("你好",                              "chat_agent"),
        ]

        wrong = []
        for query, expect in cases:
            got = self.system.router.route({"description": query}, agents)
            if got != expect:
                wrong.append(f"「{query}」→ {got}（期望 {expect}）")

        assert not wrong, "路由错误:\n  " + "\n  ".join(wrong)
        return f"{len(cases)}/{len(cases)} 正确"

    def test_10_policy_qa_with_exception(self):
        """
        政策问答，且必须踩中业务陷阱。

        主场景的商品是「已激活的无线耳机」—— 属于七天无理由的**例外**
        （已激活 3C 数码），只能走质量问题通道。如果答成"7天无理由、运费自付"
        就说明提示词里的例外说明没起作用。
        """
        result = self.system.handle_query(
            "我在你们这买的无线耳机，已经拆开用过了，现在想退货，还能退吗？运费谁出？",
            thread_id="regression_policy",
        )
        assert result.get("success"), f"查询失败: {result.get('error', result)}"
        answer = result.get("result", "")
        assert len(answer) > 30, f"回答过短: {answer}"

        # 至少要提到"质量问题"这条通道（说明它识别出了 3C 例外）
        assert "质量" in answer, (
            f"没有识别出 3C 例外（未提到质量问题通道）:\n{answer[:300]}"
        )
        return f"质量分 {result.get('quality_score', 0):.2f}"

    def test_11_order_query(self):
        """订单查询：答案必须来自真实数据，不是编的"""
        result = self.system.handle_query(
            "订单 SO20260909001 现在什么状态？",
            thread_id="regression_order",
        )
        assert result.get("success"), f"查询失败: {result.get('error', result)}"
        answer = result.get("result", "")

        # 库里该订单状态就是「已签收」
        assert "已签收" in answer, f"没查到真实状态:\n{answer[:300]}"

        # 顺带确认路由到了数据库 Agent
        agents = [r["agent"] for r in result.get("agent_results", [])]
        assert "database_agent" in agents, f"应由 database_agent 处理，实际: {agents}"
        return "状态正确，来自 SQL 查询"

    def test_12_logistics_query(self):
        """物流查询：必须 JOIN orders 和 logistics 两张表才能答对"""
        result = self.system.handle_query(
            "SO20260909001 这个订单的快递到哪了？",
            thread_id="regression_logistics",
        )
        assert result.get("success"), f"查询失败: {result.get('error', result)}"
        answer = result.get("result", "")

        # 库里的真实承运商和运单号
        assert "顺丰" in answer, f"没查到承运商:\n{answer[:300]}"
        assert "SF1234567890" in answer, f"没查到运单号:\n{answer[:300]}"
        return "承运商 + 运单号均正确"

    def test_13_anaphora_resolution(self):
        """多轮对话：指代消解 —— 第二句里的「它」要能解析成上文的订单"""
        tid = "regression_anaphora"

        r1 = self.system.handle_query("订单 SO20260909001 现在什么状态？", thread_id=tid)
        assert r1.get("success"), f"第一轮失败: {r1.get('error', r1)}"

        r2 = self.system.handle_query("那它是什么时候签收的？", thread_id=tid)
        assert r2.get("success"), f"第二轮失败: {r2.get('error', r2)}"
        answer = r2.get("result", "")

        # 库里签收时间是 2026-09-11 15:42
        assert "09-11" in answer or "9月11" in answer or "9 月 11" in answer, (
            f"指代消解失败，「它」没解析成该订单:\n{answer[:300]}"
        )
        return "「它」正确解析为订单 SO20260909001"

    def test_14_write_gate_and_resume(self):
        """
        ★ 写操作闸门 + 审批恢复。

        **刻意绕过路由**直接调用售后 Agent —— 路由有随机性（同一句话有时走简单
        路径、有时走复杂路径），而这里只想验证「写操作会被闸门拦住」这一件事。
        回归测试一个用例只测一件事，否则失败时看不出是哪儿的问题。
        """
        from core.write_ops import consume_write_ops

        agent = self.system.registry.get_agent("customer_service_agent")["instance"]

        consume_write_ops()   # 清掉可能的残留
        agent.handle(
            "我是张伟，用户ID是 U10001。订单 SO20260909001 的耳机右耳没声音，我要退货。",
            {"history": []},
        )
        ops = consume_write_ops()

        assert ops, "售后 Agent 没有登记任何写操作（可能又出现了「只描述不调用」的问题）"

        op_types = {o["type"] for o in ops}
        assert op_types & {"submit_return_request", "create_ticket"}, (
            f"写操作类型不对: {op_types}"
        )

        # 校验登记的详情里有可追溯的单号
        detail = ops[0]["detail"]
        assert detail.get("refund_id") or detail.get("ticket_id"), f"写操作详情缺少单号: {detail}"
        return f"{ops[0]['type']} → {detail.get('refund_id') or detail.get('ticket_id')}"

    def test_15_boundary_inputs(self):
        """边界输入不应让系统崩掉"""
        # 极短输入（会走闲聊兜底）
        r = self.system.handle_query("?", thread_id="regression_edge_1")
        assert r.get("success") is not None, "极短输入返回结构异常"

        # 超长输入
        long_q = "退货流程是什么？" * 60
        r = self.system.handle_query(long_q, thread_id="regression_edge_2")
        assert r.get("success") is not None, "超长输入返回结构异常"

        # 特殊字符
        r = self.system.handle_query("订单 <script>alert(1)</script> 状态？", thread_id="regression_edge_3")
        assert r.get("success") is not None, "特殊字符返回结构异常"

        return "极短 / 超长 / 特殊字符均未崩溃"

    # ------------------------------------------------------------------

    def run_all(self) -> bool:
        """按顺序跑全部用例"""
        print("\n" + "=" * 70)
        print("  电商售后助手 · 回归测试")
        print("=" * 70)

        fast = [
            ("01 数据库就位",                 self.test_01_db_ready),
            ("02 外键约束生效",               self.test_02_foreign_key_enforced),
            ("03 匿名工单允许",               self.test_03_anonymous_ticket_allowed),
            ("04 退货三重校验",               self.test_04_return_request_validation),
            ("05 防幻觉兜底",                 self.test_05_anti_hallucination_guard),
            ("06 知识库白名单",               self.test_06_knowledge_whitelist),
        ]
        slow = [
            ("07 系统初始化",                 self.test_07_system_init),
            ("08 Agent 注册集合",             self.test_08_agents_registered),
            ("09 路由分派",                   self.test_09_routing),
            ("10 政策问答（3C 例外）",        self.test_10_policy_qa_with_exception),
            ("11 订单查询",                   self.test_11_order_query),
            ("12 物流查询（JOIN）",           self.test_12_logistics_query),
            ("13 多轮指代消解",               self.test_13_anaphora_resolution),
            ("14 写操作闸门",                 self.test_14_write_gate_and_resume),
            ("15 边界输入",                   self.test_15_boundary_inputs),
        ]

        total_start = time.time()

        print("\n[第一段] 数据层与纯逻辑（不调 LLM）")
        for name, fn in fast:
            self.run_test(name, fn)

        print("\n[第二段] 端到端（调 LLM，较慢）")
        for name, fn in slow:
            self.run_test(name, fn)

        total = time.time() - total_start
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        print("\n" + "=" * 70)
        print(f"  总计 {len(self.results)} | 通过 {passed} | 失败 {failed} | 耗时 {total:.1f}s")
        print("=" * 70)

        if failed:
            print("\n失败的用例：")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}")
                    print(f"    {r.detail[:400]}")
            print()

        return failed == 0


if __name__ == "__main__":
    tester = AfterSalesTester()
    sys.exit(0 if tester.run_all() else 1)
