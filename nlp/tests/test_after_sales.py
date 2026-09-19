# -*- coding: utf-8 -*-
"""
电商售后助手 · 回归测试
======================

**这是什么：** 改造完成后的**验收基线**。每次改动跑一遍，确认核心链路没被改坏。
（回归测试 / regression = 改完之后重跑，确认没把原来好的弄坏 —— 本文件就是干这个的。）

**怎么跑：**
    cd nlp
    .venv/Scripts/python.exe tests/test_after_sales.py

退出码 0 = 全部通过，1 = 有失败（方便接 CI）。

**两段是怎么分的：** 不是命令行开关 —— `run_all()` 里两个写死的用例列表
（`fast` / `slow`）顺序执行，一段跑完再进下一段；想只跑第一段得手动注释掉 slow 循环。

**为什么这么分：** 第一段只碰 SQLite 和纯函数，秒级出结果，改完数据层或政策逻辑可立刻自查；
第二段每个用例都要真调 LLM（大语言模型），整段约 3 分钟。把"便宜且确定"与"贵且随机"
分开，才能不必为每次小改动付真模型的成本。

用例分两段，按运行成本排：

    第一段（不调 LLM，秒级）—— 数据层与纯逻辑
        test_01  数据库就位
        test_02  外键约束真的在生效（臆造 ID 必须被拒）
        test_03  匿名工单允许创建
        test_04  退货申请的三重业务校验
        test_05  防幻觉兜底逻辑
        test_06  知识图谱政策推导（三值逻辑 + 同族覆盖 + 确定性）
        test_06b 知识库白名单

    第二段（调 LLM，较慢，每个用例数秒到数十秒）—— 端到端（E2E：从入口到出口完整跑一遍）
        test_07  系统初始化
        test_08  Agent 注册集合正确
        test_09  路由：5 类问题分派正确
        test_10  政策问答（含 3C 例外这个业务陷阱）
        test_11  订单查询（真实 SQL）
        test_12  物流查询（跨表 JOIN）
        test_13  多轮指代消解
        test_14  写操作登记链路（分段：机制确定性 + 行为不变量）

**已知的不稳定因素：** 第二段依赖 LLM，`temperature`（采样温度：>0 时选词带随机性，
0 才是每次同解）不为 0，同样的输入偶尔会有不同路径。
所以 test_14 刻意绕过路由直接调 Agent —— 只验证"写操作会被闸门拦住"这件事，
把路由的不确定性排除在外。**回归测试要尽量只测一件事。**

**用例怎么读：** 带 ★ 的（02/04/05/06/14）是**正向证明** —— 证明某个机制真的拦住了
一个真实踩过的问题，而不是"功能正常"。03 是 02 的反面：证明约束没有做过头。
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
        """懒加载系统实例：第一段的纯逻辑用例不该为它付出启动成本

        重模块的 import 也一起推迟在这里 —— 第一段因此连依赖加载都不会发生。
        """
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
        """跑一个用例并记录结果；异常一律记成失败，不让一个用例炸掉整轮

        assert（断言）不成立归 FAIL（被测逻辑错），其他异常归 ERROR（环境/接口错）——
        分开是为了看汇总时能一眼判断该去查业务代码还是查环境。
        """
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
        """数据库就位，且六张表都有数据

        防的是"没跑初始化脚本就来排查业务问题"：它失败说明库是空的，
        那么后面用例的失败可能全是这一个根因引起的 —— 先修它再往下看。
        """
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
        """匿名工单（user_id = NULL）应当允许 —— 外键只约束"给了值就必须存在"

        防的是修 test_02 时把校验做过头：堵死臆造 ID 的同时，连匿名咨询也拒了。
        这两条是同一特性的正/反两面，改一处必须回来看另一处。
        """
        from core.ecommerce_crm import EcommerceCRM
        crm = EcommerceCRM()

        r = crm.create_ticket(user_id=None, issue='回归测试-匿名咨询')
        assert not r.get("error"), f"匿名工单被误拒: {r.get('error')}"
        assert r.get("ticket_id"), "没返回工单号"
        assert r.get("user_id") is None, f"匿名工单的 user_id 应为 NULL，实际 {r.get('user_id')!r}"
        return r["ticket_id"]

    def test_04_return_request_validation(self):
        """
        ★ 退货申请的三重校验：订单存在 / 归属正确 / 状态允许售后。
        这是唯一会动钱的操作，校验必须可靠。

        防的是三条校验被删减或合并 —— 漏掉任何一条，不该退的单都会退成功。
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
        ★ 防幻觉兜底：声称办过事，就必须真调用过写工具。

        兜底（fallback）= 主路径不可信时退到的确定性规则。这里"主路径"是 LLM 的自由
        措辞，"兜底"是拿正则去核它有没有对应的工具调用记录 —— 是核验行为，不是核验文字。

        这是阶段 3 挖出来的最危险的一个问题 —— 模型会在**没调用工具**的情况下
        写出"已为您提交申请，单号 RFxxx"。

        下面「应拦下」的第 2 条是**浏览器实测中逃逸过的原句**：
        模型当时写的是「已为您**立即**创建了售后工单」，中间插了个副词，
        而最初的实现用的是固定子串（`"已为您创建" in answer`），9 条规则全部落空。
        现在改成允许插入副词的正则。**这条用例就是为了防止它再次退化。**

        同时「应放行」那几条同样重要 —— 防幻觉不能误伤正常的政策回答。
        """
        from agents.customer_service_agent.agent import CustomerServiceAgent as A

        # 借壳调用，跳过 Agent 的构造（那会加载模型、建索引，很贵）。
        # 注意 _verify_write_claim 其实是**实例方法**，这里靠鸭子类型让 _Probe 冒充 self ——
        # _Probe 因此必须带上 _WRITE_TOOLS / _CLAIM_PATTERNS 这两个同名类属性。
        class _Probe:
            _WRITE_TOOLS = A._WRITE_TOOLS
            _CLAIM_PATTERNS = A._CLAIM_PATTERNS
            _verify_write_claim = A._verify_write_claim

        probe = _Probe()
        NO_WRITE = {"query_order", "hybrid_search"}

        # ---- 应被拦下：声称已办，但本轮没调用写工具 ----
        lies = [
            "已为您提交退货申请，单号 RF20260916001",
            "我已为您立即创建了售后工单并提交了退货申请。",   # ← 曾逃逸的原句
            "已经创建了工单，请等待处理",
            "您的退货单号：RF20260916001",
        ]
        for lie in lies:
            out = probe._verify_write_claim(lie, NO_WRITE)
            assert "并未成功提交" in out, f"编造办理结果没有被拦下: {lie}"

        # ---- 应放行：真调了写工具 ----
        lie = lies[0]
        out = probe._verify_write_claim(lie, {"submit_return_request"})
        assert out == lie, "真办理的答复被误伤"

        # ---- 应放行：正常政策回答（不能被误伤）----
        legits = [
            "七天无理由需在签收后 7 日内提交退货申请，运费由您承担。",
            "您的订单已签收，退货申请需在 7 日内提交。",
            "该商品属于已激活 3C 数码，不适用七天无理由；质量问题可在 15 日内申请。",
            "订单 SO20260909001 状态为「已签收」，下单时间 2026-09-09 14:23。",
        ]
        for text in legits:
            out = probe._verify_write_claim(text, NO_WRITE)
            assert out == text, f"正常回答被误伤: {text}"

        return f"{len(lies)} 条编造全拦下 / {len(legits)} 条正常回答零误伤"

    def test_06_knowledge_graph(self):
        """
        ★ 知识图谱：政策适用性推导。

        **这是不调 LLM 的确定性用例** —— 图谱的价值就在于"算出来的"而不是"猜出来的"，
        所以这里全部是确定断言，而且要求三次结果完全一致（可复现）。

        覆盖三值逻辑（适用 / 排除 / 待确认）的三种情况，以及一个结构性问题（同族覆盖：
        越具体的类别覆盖越泛的，否则家电会同时拿到"15日"和"7日"两条互相打架的时限）。

        最要紧的是"待确认"那一档：条件未知时**不能默认成适用** —— 判错方向恰好是把
        不该退的说成能退，这是唯一危险的错法。
        """
        from core.knowledge_graph import query_policy_applicability

        EARPHONE = "云听 Pro 主动降噪无线耳机"
        PURIFIER = "云净空气净化器 3 代"

        # ---- ① 已激活 + 质量问题：无理由被排除、质量问题通道可用 ----
        r = query_policy_applicability(EARPHONE, {"ACTIVATED": True, "QUALITY_ISSUE": True})
        excluded = {e["policy"] for e in r["excluded"]}
        applies = {e["policy"] for e in r["applies"]}

        assert "七天无理由退货" in excluded, f"3C例外没推导出来: {r}"
        # 这条是修过的：通用政策原先挂在「一般商品」上，3C数码拿不到，agent 就编了「30天」
        assert "质量问题退货（15日）" in applies, f"3C 没拿到通用质量问题退货政策: {applies}"
        # 推导依据要能说出来（"因为耳机属于 3C 数码，无理由被排除"），否则图谱等于白做。
        # 注意合并后的结构：结论在 policy 上，依据在 reasons 里（一个政策可能有多条依据）。
        entry = next(e for e in r["excluded"] if e["policy"] == "七天无理由退货")
        via = {reason["via"] for reason in entry["reasons"]}
        assert "3C数码产品" in via, f"缺少推导依据: {entry}"

        # ---- ② 没说是否激活：必须"待确认"，绝不能默认成可退 ----
        r2 = query_policy_applicability(EARPHONE, {})
        assert any(e["policy"] == "七天无理由退货" for e in r2["uncertain"]), \
            "未知条件没有被标成待确认（危险：等于默认可以退）"
        assert not any(e["policy"] == "七天无理由退货" for e in r2["applies"]), \
            "未知条件被误判成适用"

        # ---- ③ 明确未激活：排除不成立 → 无理由反而适用 ----
        r3 = query_policy_applicability(EARPHONE, {"ACTIVATED": False})
        assert any(e["policy"] == "七天无理由退货" for e in r3["applies"]), \
            "排除条件不成立时没能反向推出适用"

        # ---- ④ 同族覆盖：家电只应拿到 7日/15日，不能同时出现通用的 15日/30日 ----
        r4 = query_policy_applicability(PURIFIER, {"QUALITY_ISSUE": True})
        policies4 = {e["policy"] for e in r4["applies"]}
        assert "家电质量问题退货（7日）" in policies4, f"家电专属时限丢了: {policies4}"
        assert "质量问题退货（15日）" not in policies4, \
            f"同族覆盖失效，通用的15日没被家电的7日盖掉: {policies4}"

        # ---- ⑤ 未登记锚点的商品必须显式报错，不能静默兜底 ----
        # 锚点 = 商品营销名到类别的显式映射（「云听 Pro 主动降噪无线耳机」→「无线耳机」），
        # 营销名和文档里的类别词对不上，所以必须人工登记。静默兜底 = 让漏配悄悄过去。
        r5 = query_policy_applicability("某个没登记过的商品")
        assert r5.get("error"), "未登记锚点的商品被静默处理了（会掩盖漏配）"

        # ---- ⑥ 确定性：同样输入三次，结果必须完全一致 ----
        import json as _json
        sigs = {_json.dumps(query_policy_applicability(EARPHONE, {"ACTIVATED": True}),
                            sort_keys=True, ensure_ascii=False) for _ in range(3)}
        assert len(sigs) == 1, "图谱查询结果不可复现"

        return "排除/适用/待确认三值 + 同族覆盖 + 确定性 全部正确"

    def test_06b_knowledge_whitelist(self):
        """知识库只剩售后政策文档，旧的 RAG 技术文档已清空

        RAG（检索增强生成）= 先检索资料、再让 LLM 基于资料作答；RAG 技术文档留在库里
        会污染政策问答的检索结果。防的是旧场景文档被重新灌回来 —— 篇数用等值断言而非
        "至少"，多一篇也会被抓出来。
        """
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
        """系统能初始化，图和注册表都就绪

        registry（注册中心）= Agent 名册，按名字取实例。图和它后面所有 E2E 用例的地基 ——
        这两个挂了，后面的失败都是同一个根因。
        """
        assert self.system is not None, "系统实例为 None"
        assert self.system.graph is not None, "图实例为 None"
        assert self.system.registry is not None, "注册表为 None"
        return "图 + 注册表就绪"

    def test_08_agents_registered(self):
        """Agent 注册集合正确：5 个，且 document_agent 已摘除

        防的是注册集合被悄悄改动 —— 用精确相等而非"至少包含"，误加一个 Agent 也会失败
        （能动手的 Agent 多一个，能力边界就变了）。
        """
        agent_ids = [a["id"] for a in self.system.registry.get_all_agents()]

        expected = {"knowledge_agent", "database_agent", "customer_service_agent",
                    "vqa_agent", "chat_agent"}
        assert set(agent_ids) == expected, (
            f"注册集合不符\n  实际: {sorted(agent_ids)}\n  预期: {sorted(expected)}"
        )
        assert "document_agent" not in agent_ids, "document_agent 应已摘除（9 个工具全是合同审核用）"
        return f"{len(agent_ids)} 个 Agent"

    def test_09_routing(self):
        """路由：5 类售后问题分派到正确的 Agent

        （列表里其实有 6 条：5 类业务问题 + 1 条「你好」的闲聊兜底。）
        入参是 {"description": query} 而非 query —— router.route 只读 description 字段，
        字段名对不上会静默走偏。防的是改提示词或 Agent 描述后分派整体漂移。
        """
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

        只断言关键词（"质量"）而不比对整句：LLM 每次措辞都不同，锁语义锚点才不脆。
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
        """订单查询：答案必须来自真实数据，不是编的

        防的是退化成"模型凭记忆答" —— 断言的是库里真实存在的状态值，不是回答格式。
        """
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
        """物流查询：必须 JOIN orders 和 logistics 两张表才能答对

        JOIN（联表查询）= 按关联条件把多张表拼起来查。承运商/运单号不在 orders 表里，
        所以答对就等于证明了真去查了库，而不是照提示词编。
        """
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
        """多轮对话：指代消解（anaphora resolution）—— 第二句里的「它」要能解析成上文的订单

        防的是会话记忆串了 thread_id 或历史没带上：两轮用同一个 thread_id，
        第二轮问句里没有任何订单号，答对只能靠记忆。
        """
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

    def test_14_write_ops_registration(self):
        """
        ★ 写操作登记链路。

        **这条用例被重写过一次，原因值得记：**

        原版是"让 Agent 处理一句退货请求，断言它必须产生写操作"。
        实测发现它**时好时坏** —— 模型有时会先要求用户补充凭证再提交
        （那其实是合理行为，不是 bug），于是测试偶发失败。

        **根因是测试设计缺陷：回归测试不该依赖模型的随机决策。**

        所以拆成两段，对应两种不同的确定性：

        A. **机制是确定的** —— 只要写工具被调用，就必须被登记。
           走 Agent 的**工具分发路径**（`_execute_tool_calls`）验证，完全不经过 LLM。
           注意不能直接调 `tool_registry.call_tool()` —— 登记是在分发层做的
           （阶段 3b 为了线程安全特意从工具函数里挪出来），绕过分发层就不会登记。
        B. **"Agent 会不会主动去调工具"是概率性的** —— 所以这一段不强制它必须写，
           只验证**不变量**：只要答复声称"已提交"，就必须真的写过。
           （这正是防幻觉要守的那条线。）

        不变量（invariant）= 无论随机路径怎么走都必须成立的性质。用"声称即必须真做"
        代替"必须发生某事"，测试才不会因为模型换了个合理走法就抖。
        """
        import re
        from core.write_ops import consume_write_ops

        agent = self.system.registry.get_agent("customer_service_agent")["instance"]

        # ---------- A. 机制：走分发层调用写工具 → 必须登记 ----------
        consume_write_ops()   # 读走并清空：不清的话读到的是上一次请求的残留记录
        import json as _json
        agent._execute_tool_calls(
            [{
                "id": "regression_call_1",
                "type": "function",
                "function": {
                    "name": "submit_return_request",
                    "arguments": _json.dumps({
                        "order_id": "SO20260909001",
                        "user_id": "U10001",
                        "reason": "回归测试-机制验证",
                        "refund_type": "退货退款",
                    }, ensure_ascii=False),
                },
            }],
            retrieval_count=0,
        )
        ops = consume_write_ops()

        assert ops, "调用了写工具却没有登记写操作 —— 登记链路坏了"
        assert ops[0]["type"] == "submit_return_request", f"登记类型不对: {ops[0]['type']}"
        refund_id = ops[0]["detail"].get("refund_id")
        assert refund_id, f"登记详情缺少单号: {ops[0]}"

        # ---------- B. 不变量：声称已办 ⇒ 必须真办 ----------
        consume_write_ops()
        answer = agent.handle(
            "我是张伟，用户ID是 U10001。订单 SO20260909001 的耳机右耳没声音，我要退货。",
            {"history": []},
        )
        ops2 = consume_write_ops()

        # 三种结果都算通过：
        #   ① 真办了（ops2 非空）
        #   ② 没办，也没声称办 —— 正常行为（例如先向用户索取凭证）
        #   ③ 没办却声称办了，但兜底已经把它改写成「并未成功提交」
        # 只有一种算失败：声称办了、没真办、**且兜底没拦下**。
        #
        # 注意这里必须判断"兜底是否已触发"：兜底的实现是在原文**前面加上**
        # 一段「并未成功提交」的说明、**再附上原文**，所以原文里那句虚假声明
        # 仍然在 answer 里 —— 只按正则找声称、不看兜底有没有生效，会误判。
        guard_fired = "并未成功提交" in answer
        claims_done = any(re.search(p, answer) for p in agent._CLAIM_PATTERNS)

        if claims_done and not guard_fired:
            assert ops2, (
                "★ 声称已办理、却没有写操作、兜底也没拦下 —— 这才是真问题:\n"
                f"{answer[:200]}"
            )

        if guard_fired:
            acted = "编造被兜底拦下（正确）"
        elif ops2:
            acted = "已办理"
        else:
            acted = "未办理（允许，模型可能先索取凭证）"
        return f"登记链路 OK（{refund_id}）；Agent {acted}"

    def test_15_boundary_inputs(self):
        """边界输入不应让系统崩掉

        防的是异常穿透到最外层（尤其那条 <script>：不该被当成可执行内容处理）。
        只断言返回结构存在、不规定回答内容 —— 这里测的是稳定性，不是答案质量。
        """
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

        # 分段就在这里：两个列表，fast 段全跑完才开始 slow。
        # fast 里的用例都不碰 self.system，所以连重模块的 import 都不会发生。
        fast = [
            ("01 数据库就位",                 self.test_01_db_ready),
            ("02 外键约束生效",               self.test_02_foreign_key_enforced),
            ("03 匿名工单允许",               self.test_03_anonymous_ticket_allowed),
            ("04 退货三重校验",               self.test_04_return_request_validation),
            ("05 防幻觉兜底",                 self.test_05_anti_hallucination_guard),
            ("06 知识图谱（政策推导）",        self.test_06_knowledge_graph),
            ("06b 知识库白名单",              self.test_06b_knowledge_whitelist),
        ]
        slow = [
            ("07 系统初始化",                 self.test_07_system_init),
            ("08 Agent 注册集合",             self.test_08_agents_registered),
            ("09 路由分派",                   self.test_09_routing),
            ("10 政策问答（3C 例外）",        self.test_10_policy_qa_with_exception),
            ("11 订单查询",                   self.test_11_order_query),
            ("12 物流查询（JOIN）",           self.test_12_logistics_query),
            ("13 多轮指代消解",               self.test_13_anaphora_resolution),
            ("14 写操作登记链路",             self.test_14_write_ops_registration),
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
