# -*- coding: utf-8 -*-
"""
RAG（检索增强生成）Agent 多智能体协作系统 —— 命令行（CLI）入口
============================================================

本文件的职责只有两件：做交互式问答循环，以及把"需要人工确认"这件事
落到 input() 上。跑图逻辑一行都不在这里。

与 api.py 的关系：共用同一套 EnhancedLangGraphRAGSystem（同一个类、几乎同一组
构造参数），但是**两个独立进程、各建各的实例** —— 命令行与 Web 之间不共享
会话记忆，也不共享暂停中的断点（checkpointer 是进程内的 MemorySaver 内存存档器）。

查图的四个入口别混（分工见 enhanced_entry.py 文件头）：
  handle_query              → Web /chat（非流式）
  handle_query_stream       → 本文件（同步流式）
  handle_query_stream_async → Web /chat/stream（异步流式）
  resume_after_human_input  → 两边共用的人工恢复通道

命令行独有的约定：图片 / 文件用"标记"写在问题里（[image:路径] / [file:路径]），
由本文件的正则摘出来再拆成两个字段；Web 端对应的是 JSON 字段
image_paths / file_paths，没有这层标记语法。

基于 LangGraph（图编排框架）的多 Agent（智能体）编排系统。
支持任务规划、智能路由、ReAct（推理-行动循环）执行、状态持久化、断点续传、人机共驾。
"""

import os
import re
import sys
import logging
from typing import Optional, List, Tuple
from dotenv import load_dotenv
load_dotenv()

# 设置Windows控制台UTF-8编码
# 为什么需要：Windows 控制台默认 GBK，print 中文/特殊符号会抛 UnicodeEncodeError
# 直接把循环打断；errors='replace' 保证最坏情况是乱码而不是崩溃。
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
    """
    加载.env文件中的环境变量（自带的极简解析，不依赖 python-dotenv 的解析能力）

    与 api.py 的 _load_env 是两份重复实现，且有一处行为差异：这里用 `=`
    直接赋值（覆盖进程里已有的同名变量），api.py 用 setdefault（不覆盖）。
    """
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

    # 与 api.py 的 lifespan 构造参数保持一致（CLI 多一个 auto_update_index 开关）；
    # 想改编排行为（如 enable_critic）要两边一起改，否则 CLI 与 Web 会悄悄分叉。
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

    这是命令行独有的入参方式（Web 端直接用 JSON 字段），所以必须先把标记
    从问题里摘干净再喂给图 —— 否则 "[image:...]" 这段字面量会被当成问题内容。
    路径不做转义，用非贪婪匹配到第一个 `]` 为止，路径里含 `]` 会截断。

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

    与 parse_image_tags 同构，区别只在标记词；调用方是串行解析
    （先用图片函数摘一遍，再把结果交给本函数摘文件标记）。

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
    """
    主函数（三种运行方式，靠 argparse 命令行参数解析分流）

    1. 不带参数          → 进交互式对话循环（本函数的 while True）
    2. --init-db / --update-db → 一次性运维命令，跑完直接 return，不进对话循环
    3. --no-auto-update  → 只改启动行为（关掉启动时的自动向量索引更新），仍进对话循环
    """
    import argparse

    parser = argparse.ArgumentParser(description='RAG Agent 多智能体协作系统 (LangGraph)')
    parser.add_argument('--init-db', action='store_true', help='初始化向量数据库（完全重建）')
    parser.add_argument('--update-db', action='store_true', help='增量更新向量数据库')
    parser.add_argument('--no-auto-update', action='store_true', help='禁用自动更新向量索引')
    args = parser.parse_args()
    # action='store_true'：出现该参数即为 True，不需要跟值（--init-db 而非 --init-db=1）

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
                    # 与 Web 的 GET /agents 不同：这里拿到的是给终端直接打印的
                    # 文本（registry 注册中心渲染好的），不是给前端用的结构化 JSON。
                    print("\n" + system.list_agents())
                    continue
                elif user_input == "/reset":
                    # 事实说明（未改动逻辑）：这一支只打印提示，没有清空任何状态 ——
                    # 既不换 thread_id 也不清记忆，所以后续对话仍接着原来的上下文。
                    # 帮助文案里写的"重置对话"与这里的实际行为不一致。
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
                # 事件类型与 Web 端 /chat/stream 推的 SSE 完全同源（progress 由节点内的
                # get_stream_writer 推出），区别只是这里直接 print，不编成 SSE 帧。
                # 注意：没有传 thread_id → 走默认值 "default"，即整条命令行会话
                # 共用同一份记忆与断点存档（多用户场景才需要各自传不同 thread_id）。
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
                    # 人工介入交互 —— CLI 版的人机共驾。
                    # Web 那边是前端弹审核框、再 POST /resume；这里就是 input() 收一行指令，
                    # 收完调同一个 resume_after_human_input，所以两端的恢复语义完全一致。
                    # reason / data 对应图里的 intervention_reason / intervention_data。
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
                        # 只本地打印，不通知图 —— 图仍停在那个断点上，
                        # 存档留在 checkpointer 里（下次同一 thread_id 会接在它上面）。
                        print("\n已终止任务")
                    else:
                        # 恢复执行。CLI 不做白名单校验，输什么就原样进 human_feedback；
                        # 但只有字面量 "approved" 会置审批标志（同 api.py 那条约定），
                        # 所以别的词可能恢复后被同一条通道再拦一次 —— 下面的 while 即为兜这个。
                        print("\n继续执行中...")
                        thread_id = human_intervention.get('thread_id', 'default')
                        result = system.resume_after_human_input(
                            thread_id=thread_id,
                            human_feedback=human_input,
                        )

                        # 恢复后可能再次需要人工介入：图里有多条互不相干的审批通道
                        # （任务规划审核 / 数据库写操作 / 写操作审批 / 质量不过关 / Critic 失败），
                        # 批过一条只是放行那一条，后面的通道仍可能在下一段执行里再拦一次。
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
                    # mode 只有 planning / simple 两种，源自 state 里的 is_complex
                    # （由复杂度分类器判定）：复杂走规划器拆任务，简单直接路由给单个 Agent。
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
