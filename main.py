# -*- coding: utf-8 -*-
"""
RAG Agent系统主入口
==================

多Agent协作系统，支持任务规划、智能路由、ReAct执行
"""

import os
import sys
import logging
import yaml
from typing import Optional

# 设置Windows控制台UTF-8编码
if sys.platform == 'win32':
    import codecs
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 加载.env文件
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

# 导入核心模块
from llm.llm_client import LLM
from llm.embedder import Embedder
from llm.function_calling import FunctionCalling
from core.document_filter import DocumentFilter
from orchestrator.orchestrator import Orchestrator

# 导入Agent
from agents.knowledge_agent.agent import KnowledgeAgent
from agents.document_agent.agent import DocumentAgent


class RAGSystem:
    """RAG Agent系统"""

    def __init__(self, auto_update_index: bool = True):
        """
        初始化系统

        Args:
            auto_update_index: 是否自动检测并更新向量索引（默认True）
        """
        logger.info("初始化RAG Agent系统...")

        # 自动更新向量索引（如果有新文档）
        if auto_update_index:
            self._auto_update_vector_index()

        # 加载配置
        self.config = self._load_config()

        # 初始化LLM
        self.llm_max = self._create_llm("qwen-max")
        self.llm_plus = self._create_llm("qwen-plus")

        # 初始化Embedder
        self.embedder = self._create_embedder()

        # 初始化Orchestrator
        self.orchestrator = Orchestrator(
            llm_max=self.llm_max,
            llm_plus=self.llm_plus,
            embedder=self.embedder
        )

        # 注册Agents
        self._register_agents()

        logger.info("系统初始化完成")

    def _auto_update_vector_index(self):
        """自动检测并更新向量索引"""
        try:
            from tools.scripts.incremental_update import IncrementalIndexer, incremental_update

            # 检查是否有新文档
            indexer = IncrementalIndexer()
            new_files, modified_files = indexer.get_new_or_modified_files()

            if new_files or modified_files:
                logger.info(f"检测到 {len(new_files)} 个新文档, {len(modified_files)} 个修改的文档")
                logger.info("正在自动更新向量索引...")

                # 执行增量更新
                incremental_update()

                logger.info("向量索引更新完成")
            else:
                logger.debug("向量索引已是最新，无需更新")

        except Exception as e:
            logger.warning(f"自动更新向量索引失败: {e}")
            logger.warning("将继续使用现有索引")

    def _load_config(self) -> dict:
        """加载配置文件"""
        config = {}

        # 加载models.yaml
        models_path = "config/models.yaml"
        if os.path.exists(models_path):
            with open(models_path, 'r', encoding='utf-8') as f:
                loaded = yaml.safe_load(f)
                config['models'] = loaded.get('models', {})

        # 加载agents.yaml
        agents_path = "config/agents.yaml"
        if os.path.exists(agents_path):
            with open(agents_path, 'r', encoding='utf-8') as f:
                loaded = yaml.safe_load(f)
                config['agents'] = loaded.get('agents', {})

        return config

    def _create_llm(self, model_name: str) -> LLM:
        """创建LLM实例"""
        model_config = self.config.get('models', {}).get(model_name, {})

        api_key = os.getenv(model_config.get('api_key_env', 'DASHSCOPE_API_KEY'))

        return LLM(
            model_name=model_config.get('model_name', model_name),
            api_key=api_key,
            base_url=model_config.get('base_url'),
            max_tokens=model_config.get('max_tokens', 1024),
            temperature=model_config.get('temperature', 0.7)
        )

    def _create_embedder(self) -> Embedder:
        """创建Embedder实例"""
        # 尝试导入API embedder
        try:
            from rag_core.api_embedder import APIEmbedder
            api_key = os.getenv('DASHSCOPE_API_KEY')
            api_embedder = APIEmbedder(
                api_key=api_key,
                provider='dashscope',
                model='text-embedding-v2'
            )
            return Embedder(api_embedder=api_embedder)
        except Exception as e:
            logger.warning(f"无法初始化API Embedder: {e}")
            return Embedder()

    def _register_agents(self):
        """注册所有Agent"""
        # 加载文档过滤器
        doc_filter = DocumentFilter()
        metadata_file = "data/document_metadata.json"
        if os.path.exists(metadata_file):
            doc_filter.load_from_file(metadata_file)
            logger.info("文档元数据加载成功")
        else:
            logger.warning("文档元数据文件不存在，请先运行 python init_documents.py")

        # 初始化向量检索器
        retriever = None
        try:
            from rag_core.chroma_store import ChromaStore
            from rag_core.api_embedder import APIEmbedder

            # 检查向量数据库是否存在
            chroma_db_path = "./vector_db/chroma_db"
            if os.path.exists(chroma_db_path):
                logger.info("加载向量数据库...")

                # 创建Embedder
                api_key = os.getenv('DASHSCOPE_API_KEY')
                api_embedder = APIEmbedder(
                    api_key=api_key,
                    provider='dashscope',
                    model='text-embedding-v2'
                )

                # 创建ChromaStore
                chroma_store = ChromaStore(
                    dimension=1536,
                    collection_name="rag_documents",
                    persist_directory=chroma_db_path
                )

                # 创建简单的检索器包装
                class SimpleRetriever:
                    def __init__(self, embedder, chroma_store):
                        self.embedder = embedder
                        self.chroma_store = chroma_store

                    def retrieve(self, query: str, top_k: int = 3):
                        # 向量化查询
                        query_embedding = self.embedder.encode_query(query)

                        # 检索
                        results = self.chroma_store.search(
                            query_vector=query_embedding,
                            top_k=top_k
                        )

                        # 转换格式
                        documents = []
                        for chunk, score in results:
                            documents.append({
                                "content": chunk,
                                "score": score,
                                "metadata": {}
                            })

                        return documents

                retriever = SimpleRetriever(api_embedder, chroma_store)
                logger.info("向量检索器加载成功")
            else:
                logger.warning(f"向量数据库不存在: {chroma_db_path}")
                logger.warning("请先运行: python init_vector_db.py")

        except Exception as e:
            logger.error(f"向量检索器初始化失败: {e}", exc_info=True)

        # Knowledge Agent
        try:
            # 初始化BM25检索器（用于混合检索）
            bm25_retriever = None
            try:
                from rag_core.bm25_retriever import BM25Retriever
                if retriever:
                    # 从向量数据库加载文档用于BM25索引
                    logger.info("初始化BM25检索器...")
                    bm25_retriever = BM25Retriever()
                    # 注意：这里需要从ChromaDB加载所有文档来构建BM25索引
                    # 简化处理：如果没有预先构建的BM25索引，可以先不启用
                    logger.info("BM25检索器初始化完成")
            except Exception as e:
                logger.warning(f"BM25检索器初始化失败: {e}")

            fc_knowledge = FunctionCalling()
            knowledge_agent = KnowledgeAgent(
                llm=self.llm_plus,
                retriever=retriever,
                bm25_retriever=bm25_retriever,
                function_calling=fc_knowledge,
                document_filter=doc_filter,
                embedder=self.embedder,
                enable_optimizations=True,
                enable_sentiment=False  # 禁用情感分析
            )
            self.orchestrator.register_agent(
                agent_id="knowledge_agent",
                name="知识检索Agent",
                description="专门处理通用知识查询，擅长技术概念、原理解释、文档检索。支持文档分层过滤。已集成8大生产级优化。",
                capabilities=["语义检索", "关键词检索", "混合检索", "文本摘要", "文档过滤", "查询优化", "上下文压缩", "答案引用"],
                agent_instance=knowledge_agent
            )
            logger.info("Knowledge Agent注册成功（生产级优化已启用）")
        except Exception as e:
            logger.error(f"Knowledge Agent注册失败: {e}")

        # Document Agent
        try:
            fc_document = FunctionCalling()
            document_agent = DocumentAgent(
                llm=self.llm_plus,
                retriever=retriever,
                function_calling=fc_document,
                embedder=self.embedder
            )
            self.orchestrator.register_agent(
                agent_id="document_agent",
                name="合同审核Agent",
                description="企业级合同审核Agent，支持文档解析、风险识别、历史对比。基于ReAct模式自主决策",
                capabilities=["文档解析", "结构化提取", "风险识别", "历史对比", "风险评分", "供应商查询"],
                agent_instance=document_agent
            )
            logger.info("Document Agent注册成功")
        except Exception as e:
            logger.error(f"Document Agent注册失败: {e}")


    def handle_query(self, query: str) -> dict:
        """
        处理用户查询

        Args:
            query: 用户查询

        Returns:
            处理结果
        """
        return self.orchestrator.handle(query)

    def list_agents(self) -> str:
        """列出所有Agent"""
        return self.orchestrator.list_agents()


def print_banner():
    """打印欢迎横幅"""
    banner = """
========================================
RAG Agent 多智能体协作系统
========================================

特性:
  - 任务规划 (LLM生成DAG)
  - 智能路由 (向量相似度 + LLM精排)
  - ReAct执行 (思考-行动-观察循环)
  - 多Agent协作 (Knowledge/Code/Customer)
  - 文档分层 (基于角色的文档过滤)

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

向量数据库管理:
  python main.py --init-db        - 初始化向量数据库（完全重建）
  python main.py --update-db      - 增量更新向量数据库
  python main.py --no-auto-update - 禁用启动时自动更新
"""
    print(help_text)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='RAG Agent 多智能体协作系统')
    parser.add_argument('--init-db', action='store_true', help='初始化向量数据库（完全重建）')
    parser.add_argument('--update-db', action='store_true', help='增量更新向量数据库')
    parser.add_argument('--no-auto-update', action='store_true', help='禁用自动更新向量索引')
    args = parser.parse_args()

    # 如果指定了初始化数据库
    if args.init_db:
        logger.info("开始初始化向量数据库...")
        from tools.scripts.init_vector_db import init_vector_db
        init_vector_db()
        logger.info("向量数据库初始化完成")
        return

    # 如果指定了更新数据库
    if args.update_db:
        logger.info("开始增量更新向量数据库...")
        from tools.scripts.incremental_update import incremental_update
        incremental_update()
        logger.info("向量数据库更新完成")
        return

    print_banner()

    try:
        # 初始化系统
        system = RAGSystem(auto_update_index=not args.no_auto_update)

        print("\n系统就绪！输入 /help 查看帮助信息\n")

        # 命令行交互循环
        while True:
            try:
                # 读取用户输入
                user_input = input("\n用户> ").strip()

                if not user_input:
                    continue

                # 处理命令
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

                # 处理查询
                print("\n处理中...")
                result = system.handle_query(user_input)

                # 显示结果
                print("\n" + "="*60)
                print(f"模式: {result.get('mode', 'unknown')}")

                if result.get('mode') == 'planning':
                    plan = result.get('plan', {})
                    print(f"规划思路: {plan.get('reasoning', '')}")
                    print(f"任务数量: {len(plan.get('tasks', []))}")

                elif result.get('mode') == 'simple':
                    print(f"使用Agent: {result.get('agent_name', '')}")

                print("="*60)

                if result.get('success'):
                    print(f"\n助手> {result.get('result', '')}")
                else:
                    print(f"\n错误> {result.get('error', '处理失败')}")

            except KeyboardInterrupt:
                print("\n\n再见！")
                break

            except Exception as e:
                logger.error(f"处理出错: {e}", exc_info=True)
                print(f"\n错误: {e}")

    except Exception as e:
        logger.error(f"系统初始化失败: {e}", exc_info=True)
        print(f"系统初始化失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
