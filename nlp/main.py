# -*- coding: utf-8 -*-
"""
RAG Agent 多智能体协作系统 - 主入口
====================================

基于 LangGraph 框架的多 Agent 编排系统。
支持任务规划、智能路由、ReAct执行、状态持久化、断点续传、人机共驾。
"""

import os
import re
import sys
import logging
from typing import Optional, List, Tuple
from dotenv import load_dotenv
load_dotenv()

# 设置Windows控制台UTF-8编码
if sys.platform == 'win32':
    import io
    # 强制 stdin/stdout/stderr 使用 UTF-8 编码
    if hasattr(sys.stdin, 'buffer'):
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def load_env():
    """加载.env文件中的环境变量"""
    env_path = '.env'
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


load_env()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """打印欢迎横幅"""
    banner = """
========================================
 RAG Agent 多智能体协作系统 (LangGraph)
========================================

核心特性:
  - LangGraph 状态图编排
  - 任务规划 (LLM生成DAG)
  - 智能路由 (向量相似度 + LLM精排)
  - ReAct执行 (思考-行动-观察循环)
  - 多Agent协作 (Knowledge/Database/Customer/Document)
  - 状态持久化 & 断点续传
  - 人机共驾 (Human-in-the-Loop)

========================================
"""
    print(banner)


def print_help():
    """打印帮助信息"""
    help_text = """
可用命令:
  /help     - 显示帮助信息
  /agents   - 列出所有Agent
  /reset    - 重置对话
  /exit     - 退出系统

直接输入问题即可开始对话。

图片输入（VQA视觉问答）:
  在问题末尾添加 [image:图片路径] 即可分析图片，支持多张：
  分析这张图表的趋势 [image:C:\\data\\chart.png]
  对比这两张图 [image:chart1.png] [image:chart2.png]

文件输入（Excel/CSV/PDF/Word 分析）:
  在问题末尾添加 [file:文件路径] 即可分析文件：
  帮我分析这份数据 [file:C:\\data\\sales.xlsx]
  审核这份合同 [file:contract.pdf]

向量数据库管理:
  python main.py --init-db        - 初始化向量数据库（完全重建）
  python main.py --update-db      - 增量更新向量数据库
  python main.py --no-auto-update - 禁用启动时自动更新
"""
    print(help_text)


def create_system(auto_update_index: bool = True):
    """
    创建系统实例（使用 LangGraph 框架）

    Args:
        auto_update_index: 是否自动更新向量索引

    Returns:
        EnhancedLangGraphRAGSystem 实例
    """
    from langgraph_orchestrator.enhanced_entry import EnhancedLangGraphRAGSystem

    logger.info("使用 LangGraph 编排模式")
    return EnhancedLangGraphRAGSystem(
        auto_update_index=auto_update_index,
        enable_checkpointer=True,
        enable_parameter_validation=True,
        enable_critic=False,
    )




def parse_image_tags(user_input: str) -> Tuple[str, List[str]]:
    """
    从用户输入中解析 [image:路径] 标记

    Args:
        user_input: 原始用户输入

    Returns:
        (纯文本query, 图片路径列表)

    示例:
        "分析趋势 [image:C:\\chart.png]" -> ("分析趋势", ["C:\\chart.png"])
        "对比 [image:a.png] [image:b.png]" -> ("对比", ["a.png", "b.png"])
    """
    pattern = r'\[image:(.+?)\]'
    paths = re.findall(pattern, user_input)
    query = re.sub(pattern, '', user_input).strip()
    return query, paths


def parse_file_tags(user_input: str) -> Tuple[str, List[str]]:
    """
    从用户输入中解析 [file:路径] 标记

    Args:
        user_input: 原始用户输入

    Returns:
        (纯文本query, 文件路径列表)

    示例:
        "分析数据 [file:C:\\data\\sales.xlsx]" -> ("分析数据", ["C:\\data\\sales.xlsx"])
        "审核合同 [file:a.pdf] [file:b.pdf]" -> ("审核合同", ["a.pdf", "b.pdf"])
    """
    pattern = r'\[file:(.+?)\]'
    paths = re.findall(pattern, user_input)
    query = re.sub(pattern, '', user_input).strip()
    return query, paths


# ========== 主函数 ==========

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='RAG Agent 多智能体协作系统 (LangGraph)')
    parser.add_argument('--init-db', action='store_true', help='初始化向量数据库（完全重建）')
    parser.add_argument('--update-db', action='store_true', help='增量更新向量数据库')
    parser.add_argument('--no-auto-update', action='store_true', help='禁用自动更新向量索引')
    args = parser.parse_args()

    # 数据库操作（独立命令）
    if args.init_db:
        logger.info("开始初始化向量数据库...")
        from tools.scripts.init_vector_db import init_vector_db
        init_vector_db()
        logger.info("向量数据库初始化完成")
        return

    if args.update_db:
        logger.info("开始增量更新向量数据库...")
        from tools.scripts.incremental_update import incremental_update
        incremental_update()
        logger.info("向量数据库更新完成")
        return

    # 启动系统
    print_banner()

    try:
        system = create_system(auto_update_index=not args.no_auto_update)
        print("\n系统就绪！输入 /help 查看帮助信息\n")

        # 命令行交互循环
        while True:
            try:
                user_input = input("\n用户> ").strip()

                if not user_input:
                    continue

                if user_input == "/exit":
                    print("\n再见！")
                    break
                elif user_input == "/help":
                    print_help()
                    continue
                elif user_input == "/agents":
                    print("\n" + system.list_agents())
                    continue
                elif user_input == "/reset":
                    print("\n对话已重置")
                    continue

                # 解析图片标记
                query_text, image_paths = parse_image_tags(user_input)
                if not query_text:
                    query_text = user_input

                # 解析文件标记
                query_text, file_paths = parse_file_tags(query_text)
                if not query_text:
                    query_text = user_input

                if image_paths:
                    # 检查图片文件是否存在
                    valid_paths = []
                    for p in image_paths:
                        if os.path.exists(p):
                            valid_paths.append(p)
                        else:
                            print(f"  [警告] 图片文件不存在: {p}")
                    image_paths = valid_paths
                    if valid_paths:
                        print(f"  [VQA] 已加载 {len(valid_paths)} 张图片")

                if file_paths:
                    # 检查文件是否存在
                    valid_files = []
                    for p in file_paths:
                        if os.path.exists(p):
                            valid_files.append(p)
                        else:
                            print(f"  [警告] 文件不存在: {p}")
                    file_paths = valid_files
                    if valid_files:
                        print(f"  [文件] 已加载 {len(valid_files)} 个文件")

                # 处理查询（流式输出）
                print()
                final_result = None
                human_intervention = None

                for event in system.handle_query_stream(
                    query_text,
                    image_paths=image_paths if image_paths else None,
                    file_paths=file_paths if file_paths else None,
                ):
                    event_type = event.get("type")

                    if event_type == "progress":
                        # 实时进度
                        agent = event.get("agent", "system")
                        msg = event.get("msg", "")
                        print(f"  [{agent}] {msg}")

                    elif event_type == "node_done":
                        # 节点完成（静默，避免过多输出）
                        pass

                    elif event_type == "human_intervention":
                        human_intervention = event

                    elif event_type == "final":
                        final_result = event.get("result", {})

                    elif event_type == "error":
                        print(f"\n  错误: {event.get('error', '未知错误')}")

                # 显示结果分隔线
                print("\n" + "=" * 60)

                if human_intervention:
                    # 人工介入交互
                    print(f"\n需要人工介入")
                    print(f"原因: {human_intervention.get('reason', '')}")

                    intervention_data = human_intervention.get('data', {})
                    if intervention_data:
                        print(f"详情: {intervention_data.get('message', '')}")
                        if 'plan' in intervention_data:
                            print(f"任务计划: {intervention_data['plan']}")
                        if 'score' in intervention_data:
                            print(f"当前质量: {intervention_data['score']}")
                        if 'issues' in intervention_data:
                            print(f"Critic问题: {intervention_data['issues']}")

                    # 等待用户输入
                    human_input = input("\n请输入操作指令> ").strip()

                    if human_input.lower() == 'abort':
                        print("\n已终止任务")
                    else:
                        # 恢复执行
                        print("\n继续执行中...")
                        thread_id = human_intervention.get('thread_id', 'default')
                        result = system.resume_after_human_input(
                            thread_id=thread_id,
                            human_feedback=human_input,
                        )

                        # 恢复后可能再次需要人工介入
                        while result.get('human_intervention_required'):
                            print(f"\n再次需要人工介入")
                            print(f"原因: {result.get('intervention_reason', '')}")
                            i_data = result.get('intervention_data', {})
                            if i_data:
                                print(f"详情: {i_data.get('message', '')}")

                            human_input = input("\n请输入操作指令> ").strip()
                            if human_input.lower() == 'abort':
                                print("\n已终止任务")
                                break
                            print("\n继续执行中...")
                            result = system.resume_after_human_input(
                                thread_id=thread_id,
                                human_feedback=human_input,
                            )

                        if not result.get('human_intervention_required'):
                            if result.get('success'):
                                print(f"\n助手> {result.get('result', '')}")
                            else:
                                print(f"\n错误> {result.get('error', '处理失败')}")

                elif final_result:
                    if final_result.get('mode') == 'planning':
                        plan = final_result.get('plan', {})
                        print(f"模式: 规划执行")
                        if plan:
                            print(f"规划思路: {plan.get('reasoning', '')}")
                            print(f"任务数量: {len(plan.get('tasks', []))}")
                    else:
                        print(f"模式: 直接路由")

                    score = final_result.get('quality_score', 0)
                    print(f"质量评分: {score:.2f}")
                    print("=" * 60)
                    print(f"\n助手> {final_result.get('result', '')}")
                else:
                    print("未获取到结果")

            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except EOFError:
                print("\n\n再见！")
                break
            except Exception as e:
                logger.error("处理出错: %s", e, exc_info=True)
                print(f"\n错误: {e}")

    except Exception as e:
        logger.error("系统初始化失败: %s", e, exc_info=True)
        print(f"系统初始化失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
