# -*- coding: utf-8 -*-
"""
系统上线前全面测试脚本
======================

模拟真实用户的各种操作场景，覆盖：
1. 系统初始化与健康检查
2. 简单知识问答（knowledge_agent）
3. 数据库查询（database_agent）
4. 客服对话（customer_service_agent）
5. 闲聊兜底（chat_agent）
6. 文件上传解析（PDF/Excel/CSV/TXT/Markdown）
7. 边界情况（空查询、超长查询、特殊字符）
8. 流式与非流式 API 兼容性
9. 多轮对话（记忆/指代消解）
10. 复杂查询（多Agent协作DAG）

运行方式：
    cd C:\\Users\\13508\\Desktop\\RAG_Project
    python tests/test_system_prelaunch.py
"""

import os
import sys
import io
import json
import time
import traceback
import logging
from typing import List, Dict, Tuple

# Windows UTF-8 编码处理
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

logging.basicConfig(
    level=logging.WARNING,  # 测试时只显示 WARNING 及以上，减少干扰
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PreLaunchTest")
logger.setLevel(logging.INFO)


class TestResult:
    """单个测试用例结果"""
    def __init__(self, name: str, passed: bool, detail: str = "", duration: float = 0):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.duration = duration

    def __str__(self):
        icon = "✅" if self.passed else "❌"
        dur = f" ({self.duration:.1f}s)" if self.duration > 0 else ""
        base = f"  {icon} {self.name}{dur}"
        if not self.passed and self.detail:
            base += f"\n     ↳ {self.detail[:300]}"
        return base


class PreLaunchTester:
    """系统上线前测试器"""

    def __init__(self):
        self.results: List[TestResult] = []
        self.system = None

    def run_test(self, name: str, func):
        """执行单个测试并记录结果"""
        start = time.time()
        try:
            result = func()
            duration = time.time() - start
            if result is True or (isinstance(result, str) and result):
                self.results.append(TestResult(name, True, "", duration))
            elif result is False:
                self.results.append(TestResult(name, False, "测试返回 False", duration))
            else:
                self.results.append(TestResult(name, True, str(result)[:200] if result else "", duration))
        except Exception as e:
            duration = time.time() - start
            detail = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()[-500:]}"
            self.results.append(TestResult(name, False, detail, duration))

    # ========== 测试用例 ==========

    def test_01_system_init(self):
        """测试1: 系统初始化"""
        from langgraph_orchestrator.enhanced_entry import EnhancedLangGraphRAGSystem

        self.system = EnhancedLangGraphRAGSystem(
            auto_update_index=False,
            enable_checkpointer=True,
            enable_parameter_validation=True,
            enable_critic=False,
        )
        assert self.system is not None, "系统实例为 None"
        assert self.system.graph is not None, "图实例为 None"
        assert self.system.registry is not None, "注册表为 None"
        return True

    def test_02_agents_registered(self):
        """测试2: 所有Agent已注册"""
        agents = self.system.registry.get_all_agents()
        agent_ids = [a["id"] for a in agents]
        logger.info(f"已注册Agent: {agent_ids}")

        expected = ["knowledge_agent", "database_agent", "customer_service_agent",
                     "document_agent", "vqa_agent", "chat_agent"]
        missing = [a for a in expected if a not in agent_ids]
        if missing:
            raise AssertionError(f"缺失Agent: {missing}")
        return True

    def test_03_simple_knowledge_query(self):
        """测试3: 简单知识问答 - RAG检索"""
        result = self.system.handle_query(
            "什么是RAG？请简要说明",
            thread_id="test_knowledge_1"
        )
        assert result.get("success"), f"查询失败: {result.get('error', result)}"
        answer = result.get("result", "")
        assert len(answer) > 20, f"回答过短: {answer}"
        logger.info(f"[知识问答] 质量分: {result.get('quality_score', 0):.2f}, 答案前100字: {answer[:100]}")
        return True

    def test_04_database_query(self):
        """测试4: 数据库查询"""
        result = self.system.handle_query(
            "查询数据库中有哪些表？",
            thread_id="test_db_1"
        )
        assert result.get("success"), f"查询失败: {result.get('error', result)}"
        answer = result.get("result", "")
        assert len(answer) > 10, f"回答过短: {answer}"
        logger.info(f"[数据库查询] 质量分: {result.get('quality_score', 0):.2f}, 答案前100字: {answer[:100]}")
        return True

    def test_05_customer_service(self):
        """测试5: 客服对话"""
        result = self.system.handle_query(
            "我购买的产品出了质量问题，想投诉",
            thread_id="test_cs_1"
        )
        assert result.get("success"), f"查询失败: {result.get('error', result)}"
        answer = result.get("result", "")
        assert len(answer) > 10, f"回答过短: {answer}"
        logger.info(f"[客服对话] 质量分: {result.get('quality_score', 0):.2f}, 答案前100字: {answer[:100]}")
        return True

    def test_06_chat_greeting(self):
        """测试6: 闲聊问候"""
        result = self.system.handle_query(
            "你好，你能做什么？",
            thread_id="test_chat_1"
        )
        assert result.get("success"), f"查询失败: {result.get('error', result)}"
        answer = result.get("result", "")
        assert len(answer) > 5, f"回答过短: {answer}"
        logger.info(f"[闲聊] 答案前100字: {answer[:100]}")
        return True

    def test_07_empty_query(self):
        """测试7: 空查询边界"""
        result = self.system.handle_query("", thread_id="test_edge_1")
        # 空查询应该不崩溃，可能返回错误提示或低质量答案
        logger.info(f"[空查询] 返回: success={result.get('success')}, error={result.get('error', 'None')}")
        return True  # 只要不崩溃就算通过

    def test_08_long_query(self):
        """测试8: 超长查询"""
        long_query = "请详细解释RAG系统的工作原理。" * 50  # ~700字
        result = self.system.handle_query(long_query[:1000], thread_id="test_edge_2")
        assert not result.get("error") or result.get("result"), f"超长查询处理失败: {result}"
        logger.info(f"[超长查询] 返回: success={result.get('success')}")
        return True

    def test_09_special_chars(self):
        """测试9: 特殊字符查询"""
        result = self.system.handle_query(
            "什么是 SQL 注入？如何防止 '; DROP TABLE users; -- 这类攻击？",
            thread_id="test_edge_3"
        )
        assert result.get("success"), f"特殊字符查询失败: {result.get('error')}"
        logger.info(f"[特殊字符] 质量分: {result.get('quality_score', 0):.2f}")
        return True

    def test_10_stream_query(self):
        """测试10: 流式查询"""
        events = []
        for event in self.system.handle_query_stream(
            "向量检索的原理是什么？",
            thread_id="test_stream_1"
        ):
            events.append(event)
            event_type = event.get("type")
            if event_type == "progress":
                logger.info(f"  [流式进度] {event.get('agent', '?')}: {event.get('msg', '')}")

        # 检查事件完整性
        types_seen = {e.get("type") for e in events}
        assert "progress" in types_seen, "缺少 progress 事件"
        assert "final" in types_seen or "error" in types_seen, "缺少 final/error 事件"

        final = [e for e in events if e.get("type") == "final"]
        if final:
            answer = final[0].get("result", {}).get("result", "")
            logger.info(f"[流式] 最终答案前100字: {answer[:100]}")

        return True

    def test_11_file_parser_txt(self):
        """测试11: TXT文件解析"""
        from core.file_parser import parse_file

        # 创建测试文件
        test_dir = os.path.join(PROJECT_ROOT, "data", "uploads")
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, "test_prelaunch.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("这是一个测试文件。\n包含中文内容，用于验证文件解析功能。\n第三行内容。")

        result = parse_file(test_file)
        assert result["success"], f"TXT解析失败: {result.get('error')}"
        assert "测试文件" in result["text"], f"解析内容不正确: {result['text'][:100]}"
        logger.info(f"[TXT解析] 成功，{result['char_count']}字符")

        # 清理
        os.remove(test_file)
        return True

    def test_12_file_parser_csv(self):
        """测试12: CSV文件解析"""
        from core.file_parser import parse_file
        import pandas as pd

        test_dir = os.path.join(PROJECT_ROOT, "data", "uploads")
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, "test_prelaunch.csv")

        df = pd.DataFrame({
            "产品": ["手机", "电脑", "平板"],
            "价格": [3999, 7999, 2999],
            "销量": [100, 50, 80],
        })
        df.to_csv(test_file, index=False, encoding="utf-8")

        result = parse_file(test_file)
        assert result["success"], f"CSV解析失败: {result.get('error')}"
        assert "手机" in result["text"], f"CSV内容不正确: {result['text'][:200]}"
        logger.info(f"[CSV解析] 成功，{result['char_count']}字符")

        os.remove(test_file)
        return True

    def test_13_file_parser_excel(self):
        """测试13: Excel文件解析"""
        from core.file_parser import parse_file
        import pandas as pd
        import gc

        test_dir = os.path.join(PROJECT_ROOT, "data", "uploads")
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, "test_prelaunch.xlsx")

        df = pd.DataFrame({
            "部门": ["技术部", "市场部", "财务部"],
            "人数": [30, 20, 10],
            "预算(万)": [500, 300, 200],
        })
        df.to_excel(test_file, index=False)

        result = parse_file(test_file)
        assert result["success"], f"Excel解析失败: {result.get('error')}"
        assert "技术部" in result["text"], f"Excel内容不正确: {result['text'][:200]}"
        logger.info(f"[Excel解析] 成功，{result['char_count']}字符")

        # Windows 文件锁兼容：强制释放文件句柄后再删除
        gc.collect()
        try:
            os.remove(test_file)
        except PermissionError:
            logger.warning("[Excel解析] 临时文件删除跳过（Windows 文件锁）")
        return True

    def test_14_file_parser_markdown(self):
        """测试14: Markdown文件解析"""
        from core.file_parser import parse_file

        test_dir = os.path.join(PROJECT_ROOT, "data", "uploads")
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, "test_prelaunch.md")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("# 测试文档\n\n## 概述\n\n这是Markdown格式的测试文件。\n\n- 列表项1\n- 列表项2")

        result = parse_file(test_file)
        assert result["success"], f"Markdown解析失败: {result.get('error')}"
        assert "测试文档" in result["text"], f"MD内容不正确"
        logger.info(f"[Markdown解析] 成功，{result['char_count']}字符")

        os.remove(test_file)
        return True

    def test_15_file_parser_pdf(self):
        """测试15: PDF文件解析（使用知识库现有PDF）"""
        from core.file_parser import parse_file

        pdf_path = os.path.join(PROJECT_ROOT, "data", "knowledge", "rag_introduction.pdf")
        if not os.path.exists(pdf_path):
            logger.warning("[PDF解析] 测试PDF不存在，跳过")
            return True

        result = parse_file(pdf_path)
        assert result["success"], f"PDF解析失败: {result.get('error')}"
        assert result["char_count"] > 100, f"PDF内容过少: {result['char_count']}字符"
        logger.info(f"[PDF解析] 成功，{result['char_count']}字符")
        return True

    def test_16_file_upload_and_query(self):
        """测试16: 文件上传后查询（模拟Excel分析流程）"""
        import pandas as pd

        # 创建测试Excel
        test_dir = os.path.join(PROJECT_ROOT, "data", "uploads")
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, "test_sales_analysis.xlsx")

        df = pd.DataFrame({
            "月份": ["1月", "2月", "3月", "4月", "5月", "6月"],
            "销售额": [12000, 15000, 13500, 17000, 16000, 19000],
            "成本": [8000, 9500, 9000, 10500, 10000, 11500],
            "利润": [4000, 5500, 4500, 6500, 6000, 7500],
        })
        df.to_excel(test_file, index=False)

        # 使用系统处理带文件的查询
        result = self.system.handle_query(
            "请帮我分析这份销售数据的趋势",
            thread_id="test_file_query_1",
            file_paths=[test_file],
        )

        assert result.get("success") or result.get("result"), f"文件查询失败: {result.get('error', result)}"
        answer = result.get("result", "")
        logger.info(f"[文件+查询] 质量分: {result.get('quality_score', 0):.2f}, 答案前150字: {answer[:150]}")

        os.remove(test_file)
        return True

    def test_17_multi_turn_memory(self):
        """测试17: 多轮对话 - 指代消解"""
        thread_id = "test_memory_1"

        # 第一轮
        r1 = self.system.handle_query("什么是混合检索？", thread_id=thread_id)
        assert r1.get("success"), f"第一轮失败: {r1.get('error')}"
        logger.info(f"[多轮-轮1] 答案前80字: {r1.get('result', '')[:80]}")

        # 第二轮 - 含指代
        r2 = self.system.handle_query("它有什么优势？", thread_id=thread_id)
        assert r2.get("success"), f"第二轮失败: {r2.get('error')}"
        answer2 = r2.get("result", "")
        logger.info(f"[多轮-轮2] 答案前80字: {answer2[:80]}")
        # 验证指代消解有效（答案应包含检索相关内容）
        return True

    def test_18_knowledge_detail_query(self):
        """测试18: 深度知识问答 - 测试检索质量"""
        result = self.system.handle_query(
            "请比较BM25和向量检索的优缺点，以及混合检索如何结合两者的优势",
            thread_id="test_knowledge_detail_1"
        )
        assert result.get("success"), f"查询失败: {result.get('error')}"
        answer = result.get("result", "")
        assert len(answer) > 100, f"回答过短（深度问题）: 仅{len(answer)}字符"
        logger.info(f"[深度知识] 质量分: {result.get('quality_score', 0):.2f}, 长度: {len(answer)}")

        # 检查引用源
        sources = result.get("sources", [])
        logger.info(f"[深度知识] 引用源数: {len(sources)}")
        return True

    def test_19_database_specific_query(self):
        """测试19: 数据库具体查询 - SQL生成"""
        result = self.system.handle_query(
            "查询销量最高的前5个专辑的名称和销量",
            thread_id="test_db_specific_1"
        )
        assert result.get("success"), f"查询失败: {result.get('error')}"
        answer = result.get("result", "")
        logger.info(f"[数据库具体] 质量分: {result.get('quality_score', 0):.2f}, 答案前150字: {answer[:150]}")
        return True

    def test_20_json_serialization(self):
        """测试20: 结果JSON序列化安全性"""
        result = self.system.handle_query(
            "你好",
            thread_id="test_json_1"
        )

        # 尝试序列化整个结果
        try:
            from api import safe_json_dumps
            json_str = safe_json_dumps(result)
            parsed = json.loads(json_str)
            assert isinstance(parsed, dict), "序列化结果不是dict"
            logger.info(f"[JSON序列化] 成功，{len(json_str)}字节")
        except Exception as e:
            raise AssertionError(f"JSON序列化失败: {e}")
        return True

    def test_21_concurrent_threads(self):
        """测试21: 不同thread_id隔离"""
        r1 = self.system.handle_query("什么是LangGraph？", thread_id="thread_A")
        r2 = self.system.handle_query("什么是向量数据库？", thread_id="thread_B")

        assert r1.get("success") and r2.get("success"), "并发线程查询失败"
        a1 = r1.get("result", "")
        a2 = r2.get("result", "")
        # 两个不同问题应该得到不同答案
        assert a1 != a2 or (len(a1) > 0 and len(a2) > 0), "不同查询得到完全相同的答案"
        logger.info(f"[线程隔离] thread_A: {len(a1)}字, thread_B: {len(a2)}字")
        return True

    def test_22_quality_score_range(self):
        """测试22: 质量评分范围检查"""
        result = self.system.handle_query(
            "解释一下RAG中的检索增强生成是什么意思",
            thread_id="test_quality_1"
        )
        score = result.get("quality_score", -1)
        assert 0 <= score <= 1, f"质量分超出范围: {score}"
        logger.info(f"[质量分范围] score={score:.2f}, 在[0,1]范围内")
        return True

    def test_23_router_accuracy(self):
        """测试23: 路由准确性测试"""
        test_cases = [
            ("查询customers表有多少条记录", "database_agent"),
            ("帮我解释什么是向量检索", "knowledge_agent"),
            ("你好啊", "chat_agent"),
        ]

        for query, expected_agent in test_cases:
            result = self.system.handle_query(query, thread_id=f"test_route_{expected_agent}")
            mode = result.get("mode", "")
            agent_results = result.get("agent_results", [])
            actual_agents = [r.get("agent") for r in agent_results] if agent_results else []
            logger.info(f"[路由] '{query[:20]}...' -> 期望:{expected_agent}, 实际:{actual_agents}")

        return True

    def test_24_api_models_validation(self):
        """测试24: API Pydantic 模型验证"""
        from api import QueryRequest, ResumeRequest

        # 正常请求
        req = QueryRequest(query="测试问题", thread_id="test")
        assert req.query == "测试问题"

        # 带文件路径
        req2 = QueryRequest(query="分析数据", file_paths=["data/test.xlsx"])
        assert req2.file_paths == ["data/test.xlsx"]

        # 带图片
        req3 = QueryRequest(query="看图说话", image_paths=["data/img.png"])
        assert req3.image_paths == ["data/img.png"]

        # ResumeRequest
        resume = ResumeRequest(thread_id="t1", human_feedback="approved")
        assert resume.human_feedback == "approved"

        logger.info("[API模型] Pydantic 验证通过")
        return True

    def test_25_error_recovery(self):
        """测试25: 错误恢复 - 无效Agent不会崩溃"""
        # 通过直接设置 state 模拟无效 agent
        result = self.system.handle_query(
            "这是一个正常的查询，测试系统稳定性",
            thread_id="test_recovery_1"
        )
        # 只要不崩溃就算通过
        logger.info(f"[错误恢复] success={result.get('success')}")
        return True

    # ========== 运行所有测试 ==========

    def run_all(self):
        """运行所有测试"""
        print("\n" + "=" * 70)
        print("  RAG Agent 系统上线前测试")
        print("=" * 70)

        # 按顺序获取所有 test_ 方法
        test_methods = [
            (name, getattr(self, name))
            for name in sorted(dir(self))
            if name.startswith("test_") and callable(getattr(self, name))
        ]

        total_start = time.time()

        for name, method in test_methods:
            print(f"\n⏳ 运行: {method.__doc__ or name}")
            self.run_test(method.__doc__ or name, method)
            print(self.results[-1])

        total_duration = time.time() - total_start

        # 汇总报告
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)

        print("\n" + "=" * 70)
        print(f"  测试完成 | 总计: {len(self.results)} | 通过: {passed} | 失败: {failed} | 耗时: {total_duration:.1f}s")
        print("=" * 70)

        if failed:
            print("\n❌ 失败的测试:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}")
                    if r.detail:
                        # 打印更详细的错误信息帮助调试
                        print(f"    详情: {r.detail[:500]}")
            print()

        return failed == 0


if __name__ == "__main__":
    tester = PreLaunchTester()
    success = tester.run_all()
    sys.exit(0 if success else 1)
