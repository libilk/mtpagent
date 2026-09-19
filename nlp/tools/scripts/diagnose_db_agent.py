# -*- coding: utf-8 -*-
"""
DatabaseAgent 快速诊断脚本（跳过 MCP 连接，直接测 sqlite3 + ReAct）
================================================================

这个脚本要回答的问题是"数据库问答坏了，坏在哪一段"。它把整条链拆成 5 段，逐段验证：

  1. sqlite3 直连查询      —— 绕开项目所有封装，确认数据库文件和 SQL 本身没问题
  2. SQLiteMCPService 查询 —— 确认项目自己的数据库服务层能返回结构化结果
  3. DatabaseAgent 工具    —— 确认工具确实注册上了、且能直接调通
  4. LLM Function Calling  —— 确认模型真的会输出"工具调用"（函数调用）而不是自由文本
  5. 完整 ReAct            —— 跑目标问题，确认"想一步→调工具→看结果→再想"整条转得起来

输出怎么读：
  - 每段失败只打印 [FAIL] 然后继续，不会中断 —— 目的是让 5 段结果一次看全，
    最后 SUMMARY 逐条列出 [PASS]/[FAIL]。
  - 关键信息是**第一个 FAIL 出现在第几段**：它前面的都过了、它开始不过，问题就在那一段。
    如果第 3 段就 FAIL，后面第 4/5 段的失败是连带的，不必单独查。

运行前提（两个都很容易踩）：
  - 本文件是模块级顺序执行的，没有函数包裹，**import 它就会开跑**；
    它也不是 pytest 用例（没有 test_ 函数），得直接 python 跑。
  - 第 4、5 段会真的发起 LLM API 调用（要 DASHSCOPE_API_KEY、会计费），
    不是纯本地自检。只想验证数据库那条链的话，把这两段跳过更划算。

（事实性说明，只记录不改逻辑）第 1 段查的是 database/chinook.db 的 Customer/Invoice 表，
但本仓库 database/ 下只有 ecommerce.db —— chinook.db 早已不存在。
sqlite3.connect 遇到不存在的路径会**顺手建一个空文件**，随后查询以 "no such table" 失败。
所以第 1 段在当前仓库必然 FAIL，那是脚本过期，不代表环境坏了。
"""

import os
import sys
import io
import json
import logging

# Windows UTF-8 兼容
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(project_root)
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stderr)]
)

# 用字符串常量而不是抛异常来标记结果：每段自己 try/except 兜住，
# 一段挂掉不能影响后面的段，否则就成了"只看得到第一个错误"的调试。
PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def section(name):
    print(f"\n{'='*60}")
    print(f">>> {name}")
    print(f"{'='*60}")


# ========== Test 1: sqlite3 direct query ==========
section("Test 1: SQLite3 direct query")
try:
    import sqlite3
    db_path = os.path.join(project_root, "database", "chinook.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Target SQL
    sql = """
    SELECT
        c.CustomerId,
        c.FirstName || ' ' || c.LastName AS CustomerName,
        ROUND(SUM(i.Total), 2) AS TotalSales,
        COUNT(il.InvoiceLineId) AS TrackCount
    FROM Customer c
    JOIN Invoice i ON c.CustomerId = i.CustomerId
    JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
    GROUP BY c.CustomerId, c.FirstName, c.LastName
    ORDER BY TotalSales DESC
    LIMIT 5
    """
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    conn.close()

    print(f"  Columns: {cols}")
    for r in rows:
        print(f"  {r}")
    print(f"\n  {PASS} sqlite3 direct query works, {len(rows)} rows")
    results.append(("sqlite3 direct", PASS))
except Exception as e:
    print(f"  {FAIL} {e}")
    results.append(("sqlite3 direct", FAIL))

# ========== Test 2: SQLiteMCPService (will use sqlite3 fallback) ==========
section("Test 2: SQLiteMCPService query")
try:
    from core.sqlite_mcp_service import SQLiteMCPService
    # 不传参数 → skip_mcp 取默认值 True，即"根本不去连 MCP"，直接用 sqlite3 降级。
    # 所以本段测的其实是项目服务层的降级路径（也是当前线上实际走的那条），
    # MCP 那条分支没被覆盖到，别看它打印了 MCP 字样就以为 MCP 在生效。
    svc = SQLiteMCPService()
    print(f"  MCP connected: {svc.connected}")

    result = svc.query("SELECT COUNT(*) AS cnt FROM Customer")
    print(f"  Customer count: {result}")
    assert result.get("success"), f"Query failed: {result}"
    assert result.get("data"), "No data returned"
    print(f"  {PASS} SQLiteMCPService query works (via {'MCP' if svc.connected else 'sqlite3 fallback'})")
    results.append(("SQLiteMCPService", PASS))
except Exception as e:
    print(f"  {FAIL} {e}")
    results.append(("SQLiteMCPService", FAIL))

# ========== Test 3: DatabaseAgent tools ==========
# 注意跨段依赖：本段定义的 agent / tools 两个变量会被第 4、5 段直接用。
# 所以本段若 FAIL，第 4/5 段会因变量未定义而 NameError —— 那是连带失败，不是新问题。
section("Test 3: DatabaseAgent tool registration & direct call")
try:
    from llm.llm_client import LLM
    from agents.database_agent.agent import DatabaseAgent

    llm = LLM(model_name="qwen-plus")
    agent = DatabaseAgent(llm=llm)

    mcp_status = "initialized" if agent.sqlite_mcp else "NOT initialized"
    print(f"  sqlite_mcp: {mcp_status}")

    tools = agent.tool_registry.get_tools_schema()
    tool_names = [t["function"]["name"] for t in tools]
    print(f"  Tools ({len(tools)}): {tool_names}")

    # Direct call
    r = agent.query_database("SELECT COUNT(*) AS cnt FROM Customer")
    print(f"  Direct query result:\n  {r}")
    print(f"  {PASS} Tools registered and callable")
    results.append(("Agent tools", PASS))
except Exception as e:
    print(f"  {FAIL} {e}")
    results.append(("Agent tools", FAIL))

# ========== Test 4: LLM Function Calling ==========
section("Test 4: LLM Function Calling format")
try:
    messages = [
        {"role": "system", "content": "You must call tools to query the database. Never answer without calling tools first."},
        {"role": "user", "content": "List all tables in the database"}
    ]

    print(f"  Sending request with {len(tools)} tool definitions...")
    response = llm.chat(messages, tools=tools, temperature=0.1)
    print(f"  Response: {response[:300]}")

    from llm.output_parser import parse_tool_calls
    tc = parse_tool_calls(response)
    if tc:
        for t in tc:
            print(f"  Tool call: {t['function']['name']}({t['function']['arguments']})")
        print(f"  {PASS} LLM correctly issued function call")
        results.append(("Function Calling", PASS))
    else:
        print(f"  {FAIL} LLM did not produce function call, responded with text")
        results.append(("Function Calling", FAIL))
except Exception as e:
    print(f"  {FAIL} {e}")
    results.append(("Function Calling", FAIL))

# ========== Test 5: Full ReAct ==========
section("Test 5: Full ReAct flow (target query)")
try:
    logging.getLogger("agents.database_agent").setLevel(logging.DEBUG)

    query = "查询销售额最高的前5位客户及其购买的曲目数量"
    print(f"  Query: {query}")
    # handle（处理入口）= Agent 对外统一的方法签名，调它等于走完整条 ReAct 循环
    result = agent.handle(query, context={})

    print(f"\n  {'='*50}")
    print(f"  FINAL ANSWER:")
    print(f"  {'='*50}")
    print(f"  {result}")
    print(f"  {'='*50}")

    # Check quality
    # 这里要抓的典型故障是：模型懒得执行，直接把 SQL 当答案吐出来了 ——
    # 所以判据是"有没有 SQL 字样、但没有任何数据/表头痕迹"，而不是 SQL 写得对不对。
    has_data = any(kw in result for kw in ["CustomerId", "客户", "销售额", "曲目", "Total", "Track"])
    has_sql_only = "SELECT" in result.upper() and "GROUP BY" in result.upper() and not has_data

    if has_sql_only:
        print(f"\n  {FAIL} Only SQL output, no execution result!")
        results.append(("Full ReAct", FAIL))
    else:
        print(f"\n  {PASS} ReAct returned data-backed answer")
        results.append(("Full ReAct", PASS))

except Exception as e:
    print(f"  {FAIL} {e}")
    results.append(("Full ReAct", FAIL))

# ========== Summary ==========
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for i, (name, status) in enumerate(results, 1):
    print(f"  {i}. {status} {name}")

failed = sum(1 for _, s in results if s == FAIL)
if failed:
    print(f"\n{FAIL} {failed} test(s) failed")
else:
    print(f"\n{PASS} All tests passed!")
print(f"{'='*60}")
