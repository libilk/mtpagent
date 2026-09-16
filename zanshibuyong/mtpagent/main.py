# -*- coding: utf-8 -*-
"""
合同风险审查 Agent —— 命令行入口
================================

单 Agent 架构：DocumentAgent 以 ReAct 模式自主调用工具审查合同。
  解析合同 → 抽取要素 → 识别风险 → 历史对比 → 风险打分 → 审核结论

用法：
  python main.py --file data/contracts/待审核合同_2026_001_高风险.txt
  python main.py                    # 交互模式，直接输入合同文件路径

评测请看 eval/（见 PLAN.md）。
"""

import os
import sys
import logging

# Windows 控制台 UTF-8
if sys.platform == 'win32':
    import codecs
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


def load_env(*paths: str) -> None:
    """把 .env / api.env 里的键值对加载进环境变量（不覆盖已存在的）"""
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())


logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

BANNER = """
========================================
        合同风险审查 Agent
========================================

  单个 ReAct Agent，自主决定调用哪些工具：

    解析合同 → 抽取要素 → 识别风险
             → 历史对比 → 风险打分 → 审核结论

  可用工具：parse_document / extract_structure / identify_risks
            compare_with_history / calculate_risk_score
            search_similar_contracts / search_by_supplier
            list_contracts / read_contract_file

========================================
"""

HELP = """
输入合同文件路径即可开始审查，例如：
  data/contracts/待审核合同_2026_001_高风险.txt

命令：
  /help    显示本帮助
  /exit    退出
"""


def build_agent(model_name: str = None):
    """按 config/models.yaml 构建合同审查 Agent"""
    import yaml
    from llm.llm_client import LLM
    from llm.function_calling import FunctionCalling
    from agents.document_agent.agent import DocumentAgent

    model_name = model_name or os.getenv('LLM_MODEL', 'qwen-plus')

    model_cfg = {}
    cfg_path = 'config/models.yaml'
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            models = (yaml.safe_load(f) or {}).get('models', {})
        model_cfg = models.get(model_name, {})

    api_key = os.getenv(model_cfg.get('api_key_env', 'DASHSCOPE_API_KEY'))
    if not api_key:
        raise SystemExit(
            "未设置 DASHSCOPE_API_KEY。请写入 .env 或 api.env，或设为环境变量。"
        )

    llm = LLM(
        model_name=model_cfg.get('model_name', model_name),
        api_key=api_key,
        base_url=model_cfg.get('base_url') or DEFAULT_BASE_URL,
        max_tokens=model_cfg.get('max_tokens', 1024),
        temperature=model_cfg.get('temperature', 0.7),
    )

    # retriever=None：未接历史合同库时，检索/历史对比类工具自动降级返回空
    return DocumentAgent(
        llm=llm,
        retriever=None,
        function_calling=FunctionCalling(),
        embedder=None,
    )


def review(agent, file_path: str) -> str:
    """审查一份合同，返回审核结论文本"""
    if not os.path.exists(file_path):
        return f"文件不存在: {file_path}"
    return agent.handle(f"请审核这份合同：{file_path}", {"file_path": file_path})


def main():
    import argparse

    load_env('.env', 'api.env')

    parser = argparse.ArgumentParser(description='合同风险审查 Agent')
    parser.add_argument('--file', help='要审查的合同文件路径（跑一次后退出）')
    parser.add_argument('--model', help='模型名，默认取 LLM_MODEL 或 qwen-plus')
    args = parser.parse_args()

    print(BANNER)

    try:
        agent = build_agent(args.model)
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Agent 初始化失败: {e}", exc_info=True)
        print(f"初始化失败: {e}")
        sys.exit(1)

    # 单次模式
    if args.file:
        print(f"正在审查: {args.file}\n" + "-" * 60)
        print(review(agent, args.file))
        return

    # 交互模式
    print(HELP)
    while True:
        try:
            user_input = input("\n合同> ").strip()
            if not user_input:
                continue
            if user_input in ('/exit', '/quit'):
                print("再见！")
                break
            if user_input == '/help':
                print(HELP)
                continue

            print("\n审查中...\n" + "-" * 60)
            print(review(agent, user_input))

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            logger.error(f"处理出错: {e}", exc_info=True)
            print(f"错误: {e}")


if __name__ == '__main__':
    main()
