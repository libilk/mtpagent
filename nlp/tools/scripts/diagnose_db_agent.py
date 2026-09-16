# -*- coding: utf-8 -*-
"""
DatabaseAgent 快速诊断脚本（跳过 MCP 连接，直接测 sqlite3 + ReAct）
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
    result = agent.handle(query, context={})

    print(f"\n  {'='*50}")
    print(f"  FINAL ANSWER:")
    print(f"  {'='*50}")
    print(f"  {result}")
    print(f"  {'='*50}")

    # Check quality
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
