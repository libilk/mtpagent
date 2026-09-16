# -*- coding: utf-8 -*-
"""
FastAPI Web 服务入口
===================

将 LangGraph RAG 系统暴露为 HTTP API，支持：
- 流式聊天接口（SSE）
- 非流式聊天接口
- 人工介入恢复
- Agent 列表查询
"""

import os
import sys
import json
import uuid
import shutil
import logging
from typing import AsyncIterator, Optional, List, Dict, Tuple
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()


class SafeJSONEncoder(json.JSONEncoder):
    """
    安全的 JSON 编码器

    将 LangChain 的 AIMessage/HumanMessage 等不可序列化对象
    自动转为普通 dict/str，避免 SSE 推送时报错。
    """
    def default(self, obj):
        # LangChain BaseMessage 子类（AIMessage, HumanMessage 等）
        try:
            from langchain_core.messages import BaseMessage
            if isinstance(obj, BaseMessage):
                return {"role": getattr(obj, "type", "unknown"), "content": obj.content}
        except ImportError:
            pass
        # 兜底：转 str
        try:
            return str(obj)
        except Exception:
            return repr(obj)


def safe_json_dumps(obj, **kwargs):
    """使用安全编码器进行 JSON 序列化"""
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(obj, cls=SafeJSONEncoder, **kwargs)


# 加载 .env 环境变量（与 main.py 保持一致）
def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

_load_env()

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 导入现有的 RAG 系统
from langgraph_orchestrator.enhanced_entry import EnhancedLangGraphRAGSystem

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局系统实例
system: Optional[EnhancedLangGraphRAGSystem] = None

# 全局问答日志记录器（与 enhanced_entry.py 共享同一数据库）
from core.qa_logger import QALogger
qa_logger: Optional[QALogger] = None

# ========== 文档注册表（用于文档内容查看） ==========

# 知识库目录
KNOWLEDGE_DIR = Path(__file__).parent / "data" / "knowledge"
METADATA_PATH = Path(__file__).parent / "data" / "metadata" / "document_metadata.json"

# 支持的文档扩展名
_DOC_EXTENSIONS = {".md", ".txt", ".pdf", ".xlsx", ".docx"}

# doc_id → { path: Path, title: str, filename: str, ext: str }
_document_registry: Dict[str, Dict] = {}


def _build_document_registry() -> None:
    """
    启动时扫描 data/knowledge/ 目录，构建文档注册表。

    使用文件名 stem 作为 doc_id，同时从 document_metadata.json 读取标题。
    前端只需传 doc_id，后端在注册表中查找，防止路径遍历攻击。
    """
    global _document_registry
    _document_registry.clear()

    if not KNOWLEDGE_DIR.exists():
        logger.warning(f"[DocRegistry] 知识库目录不存在: {KNOWLEDGE_DIR}")
        return

    # 读取元数据（获取标题）
    metadata = {}
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"[DocRegistry] 读取元数据失败: {e}")

    # 扫描知识库文件
    count = 0
    for file_path in KNOWLEDGE_DIR.iterdir():
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext not in _DOC_EXTENSIONS:
            continue

        doc_id = file_path.stem
        meta = metadata.get(doc_id, {})
        title = meta.get("title", doc_id.replace("_", " ").title())

        _document_registry[doc_id] = {
            "path": file_path,
            "title": title,
            "filename": file_path.name,
            "ext": ext,
        }
        count += 1

    logger.info(f"[DocRegistry] 文档注册表构建完成，共 {count} 个文档")


def _extract_document_content(file_path: Path, ext: str) -> Tuple[str, str]:
    """
    提取文档内容。

    Args:
        file_path: 文件绝对路径
        ext: 文件扩展名（含点号，如 '.md'）

    Returns:
        (内容文本, 内容类型) — content_type 为 'markdown' 或 'text'
    """
    if ext == ".md":
        content = file_path.read_text(encoding="utf-8")
        return content, "markdown"

    elif ext == ".txt":
        content = file_path.read_text(encoding="utf-8")
        return content, "text"

    elif ext == ".pdf":
        import pdfplumber
        texts = []
        with pdfplumber.open(str(file_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                # 提取表格
                tables = page.extract_tables()
                table_texts = []
                for table in tables:
                    if table:
                        rows = []
                        for row in table:
                            cells = [str(c) if c else "" for c in row]
                            rows.append(" | ".join(cells))
                        table_texts.append("\n".join(rows))
                combined = page_text
                if table_texts:
                    combined += "\n\n" + "\n\n".join(table_texts)
                if combined.strip():
                    texts.append(f"[第{i+1}页]\n{combined}")
        return "\n\n".join(texts), "text"

    elif ext == ".docx":
        from docx import Document
        doc = Document(str(file_path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs), "text"

    elif ext == ".xlsx":
        import pandas as pd
        xls = pd.ExcelFile(str(file_path))
        parts = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            parts.append(f"[工作表: {sheet_name}]\n{df.to_string(index=False)}")
        return "\n\n".join(parts), "text"

    else:
        raise ValueError(f"不支持的文件类型: {ext}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global system, qa_logger

    # 启动时初始化
    logger.info("=" * 60)
    logger.info("正在初始化 RAG 系统...")
    logger.info("=" * 60)

    try:
        system = EnhancedLangGraphRAGSystem(
            auto_update_index=False,  # 生产环境建议手动更新索引
            enable_checkpointer=True,
            enable_parameter_validation=True,
            enable_critic=False,
        )
        logger.info("✓ RAG 系统初始化完成")
        logger.info("✓ 服务已就绪，可以接收请求")
        logger.info("=" * 60)

        # 构建文档注册表（用于文档内容查看接口）
        _build_document_registry()

        # 初始化问答日志记录器
        qa_logger = QALogger()
        logger.info("✓ 问答日志记录器初始化完成")
    except Exception as e:
        logger.error(f"✗ 系统初始化失败: {e}", exc_info=True)
        raise

    yield

    # 关闭时清理（如果需要）
    logger.info("正在关闭服务...")


# 创建 FastAPI 应用
app = FastAPI(
    title="云集优选 · 售后助手 API",
    description="电商售后多Agent协同系统 —— 售后政策问答、订单物流查询、退换货办理、投诉工单",
    version="1.0.0",
    lifespan=lifespan
)

# 跨域配置（允许前端调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境改成具体域名，如 ["https://your-frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务（图表图片等生成文件）
os.makedirs("data/charts", exist_ok=True)
app.mount("/static/charts", StaticFiles(directory="data/charts"), name="charts")


# ========== 请求/响应模型 ==========

class QueryRequest(BaseModel):
    """聊天请求"""
    query: str = Field(..., description="用户问题", min_length=1)
    thread_id: str = Field(default="default", description="会话ID（用于状态持久化）")
    image_urls: Optional[list] = Field(default=None, description="图片公网URL列表（VQA多模态）")
    image_paths: Optional[list] = Field(default=None, description="本地图片路径列表（VQA多模态）")
    image_base64_list: Optional[list] = Field(default=None, description="base64编码图片列表（VQA多模态）")
    file_paths: Optional[List[str]] = Field(default=None, description="已上传文件路径列表（Excel/CSV，通过 /upload 获得）")


class ResumeRequest(BaseModel):
    """人工介入恢复请求"""
    thread_id: str = Field(..., description="会话ID")
    human_feedback: str = Field(..., description="人工反馈（approved/retry/abort/override）")


# ========== 核心接口 ==========

@app.post("/chat/stream")
async def chat_stream(request: QueryRequest):
    """
    流式聊天接口（SSE - Server-Sent Events）

    前端使用 EventSource 或 fetch 消费实时事件流。

    事件类型：
    - progress: 实时进度（节点执行状态）
    - node_done: 节点完成
    - human_intervention: 需要人工介入
    - final: 最终答案
    - error: 错误信息
    """
    if not system:
        raise HTTPException(status_code=503, detail="系统未初始化，请稍后重试")

    logger.info(f"[Stream] 收到请求 | Query: {request.query[:50]}... | Thread: {request.thread_id}")

    async def event_generator() -> AsyncIterator[str]:
        try:
            # 调用你现有的异步流式方法
            async for event in system.handle_query_stream_async(
                query=request.query,
                thread_id=request.thread_id,
                image_urls=request.image_urls,
                image_paths=request.image_paths,
                image_base64_list=request.image_base64_list,
                file_paths=request.file_paths,
            ):
                # 转换为 SSE 格式：data: {json}\n\n
                try:
                    event_json = safe_json_dumps(event)
                except Exception as ser_err:
                    logger.warning(f"[Stream] 事件序列化失败，跳过: {ser_err}")
                    continue
                yield f"data: {event_json}\n\n"

                # 如果是最终答案，记录日志
                if event.get("type") == "final":
                    result = event.get("result", {})
                    logger.info(f"[Stream] 完成 | Score: {result.get('quality_score', 0):.2f}")

        except Exception as e:
            logger.error(f"[Stream] 处理失败: {e}", exc_info=True)
            error_event = {
                "type": "error",
                "error": str(e),
                "detail": "系统处理异常，请稍后重试"
            }
            yield f"data: {safe_json_dumps(error_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        }
    )


@app.post("/chat")
async def chat(request: QueryRequest):
    """
    非流式聊天接口

    等待所有处理完成后一次性返回结果。
    适合不需要实时进度的场景。
    """
    if not system:
        raise HTTPException(status_code=503, detail="系统未初始化，请稍后重试")

    logger.info(f"[Chat] 收到请求 | Query: {request.query[:50]}... | Thread: {request.thread_id}")

    try:
        result = system.handle_query(
            query=request.query,
            thread_id=request.thread_id,
            image_urls=request.image_urls,
            image_paths=request.image_paths,
            image_base64_list=request.image_base64_list,
            file_paths=request.file_paths,
        )

        logger.info(f"[Chat] 完成 | Success: {result.get('success')} | Score: {result.get('quality_score', 0):.2f}")
        return JSONResponse(content=json.loads(safe_json_dumps(result)))

    except Exception as e:
        logger.error(f"[Chat] 处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/resume")
async def resume_after_human(request: ResumeRequest):
    """
    人工介入后恢复执行

    当系统触发人工介入（human_intervention_required=True）时，
    前端展示审核界面，用户做出决策后调用此接口恢复执行。

    human_feedback 可选值：
    - approved: 批准继续
    - retry: 重新执行
    - abort: 终止任务
    - override: 忽略问题强制通过
    """
    if not system:
        raise HTTPException(status_code=503, detail="系统未初始化，请稍后重试")

    logger.info(f"[Resume] 收到请求 | Thread: {request.thread_id} | Feedback: {request.human_feedback}")

    try:
        result = system.resume_after_human_input(
            thread_id=request.thread_id,
            human_feedback=request.human_feedback
        )

        logger.info(f"[Resume] 完成 | Success: {result.get('success')}")
        return JSONResponse(content=json.loads(safe_json_dumps(result)))

    except Exception as e:
        logger.error(f"[Resume] 处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========== 文件上传接口 ==========

# 允许上传的文件类型
ALLOWED_EXTENSIONS = {
    # Excel / CSV
    ".xlsx", ".xls", ".csv",
    # 图片（统一走 /upload，替代原有的 image_paths 方式）
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
    # 文档
    ".pdf", ".docx", ".doc", ".txt", ".md",
}

# 上传目录
UPLOAD_DIR = "data/uploads"

# 文档类扩展名（上传后自动解析为文本，让 LLM 直接理解）
# Excel/CSV 也纳入解析：小表全量转自然语言，大表自动降级为智能摘要
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".xlsx", ".xls", ".csv"}

# 图片类扩展名
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

# 文件解析后的最大字符数限制（超出则截断并提示）
MAX_PARSED_CHARS = 8000


# 使用公共文件解析模块（与向量数据库入库共用同一套解析逻辑）
from core.file_parser import parse_file as _parse_file


def parse_file_content(file_path: str) -> dict:
    """
    解析文件内容为纯文本（供 LLM 理解）

    内部委托给 core.file_parser.parse_file，与向量数据库入库使用同一套解析逻辑：
    - PDF：pdfplumber + OCR 降级（扫描版）+ 跨页表格合并
    - Word：python-docx 段落提取
    - TXT/MD：直接读取

    Args:
        file_path: 文件绝对路径

    Returns:
        {"success": True, "text": "...", "truncated": False, "char_count": 1234}
        或 {"success": False, "error": "..."}
    """
    return _parse_file(file_path, max_chars=MAX_PARSED_CHARS)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    上传文件（Excel/CSV/图片/文档）

    上传后自动解析文件内容为文本（图片除外），返回 parsed_text 字段。
    前端可将 parsed_text 注入到用户消息中，让 LLM 直接理解文件内容。

    前端使用 multipart/form-data 格式上传文件：
    ```javascript
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    const response = await fetch('/upload', { method: 'POST', body: formData });
    const { file_path, parsed_text, truncated } = await response.json();
    ```

    Returns:
        file_path: 服务器端文件路径（后续 /chat 时使用）
        filename: 原始文件名
        size_kb: 文件大小（KB）
        parsed_text: 解析后的文本内容（图片为 null）
        truncated: 是否因超出 token 限制被截断
        parse_error: 解析失败时的错误信息
    """
    # 1. 校验文件扩展名
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}。支持的格式: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # 2. 校验文件大小（最大 50MB）
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件过大（{len(contents) / 1024 / 1024:.1f}MB），最大允许 50MB"
        )

    # 3. 生成唯一文件名（防止覆盖）
    unique_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    # 4. 确保上传目录存在并保存文件
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(contents)

    file_size_kb = len(contents) / 1024

    logger.info(f"[Upload] 文件上传成功: {file.filename} -> {save_path} ({file_size_kb:.1f}KB)")

    # 5. 自动解析文件内容（图片除外）
    parsed_text = None
    truncated = False
    parse_error = None

    if ext in DOCUMENT_EXTENSIONS:
        result = parse_file_content(save_path)
        if result["success"]:
            parsed_text = result["text"]
            truncated = result.get("truncated", False)
            char_count = result.get("char_count", len(parsed_text))
            logger.info(f"[Upload] 文件解析成功: {char_count} 字符, 截断={truncated}")
        else:
            parse_error = result["error"]
            logger.warning(f"[Upload] 文件解析失败: {parse_error}")

    return {
        "file_path": save_path,
        "filename": file.filename,
        "size_kb": round(file_size_kb, 1),
        "parsed_text": parsed_text,
        "truncated": truncated,
        "parse_error": parse_error,
    }


@app.post("/upload/multiple")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """
    批量上传文件

    前端使用 multipart/form-data 格式上传多个文件：
    ```javascript
    const formData = new FormData();
    formData.append('files', file1);
    formData.append('files', file2);
    const response = await fetch('/upload/multiple', { method: 'POST', body: formData });
    ```

    Returns:
        files: 上传结果列表，每项包含 file_path, filename, size_kb
    """
    results = []
    errors = []

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    for file in files:
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            errors.append({"filename": file.filename, "error": f"不支持的格式: {ext}"})
            continue

        contents = await file.read()
        if len(contents) > 50 * 1024 * 1024:
            errors.append({"filename": file.filename, "error": "文件过大（超过50MB）"})
            continue

        unique_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        save_path = os.path.join(UPLOAD_DIR, unique_name)

        with open(save_path, "wb") as f:
            f.write(contents)

        file_info = {
            "file_path": save_path,
            "filename": file.filename,
            "size_kb": round(len(contents) / 1024, 1),
            "parsed_text": None,
            "truncated": False,
            "parse_error": None,
        }

        # 自动解析文件内容（图片除外）
        if ext in DOCUMENT_EXTENSIONS:
            parse_result = parse_file_content(save_path)
            if parse_result["success"]:
                file_info["parsed_text"] = parse_result["text"]
                file_info["truncated"] = parse_result.get("truncated", False)
            else:
                file_info["parse_error"] = parse_result["error"]

        results.append(file_info)

    logger.info(f"[Upload] 批量上传完成: {len(results)} 成功, {len(errors)} 失败")

    return {"files": results, "errors": errors}


# ========== 辅助接口 ==========

@app.get("/agents")
async def list_agents():
    """
    列出所有可用的 Agent

    返回结构化的 Agent 列表，包含 id、名称、描述、能力、类型等字段，
    方便前端渲染为表格或卡片。
    """
    if not system:
        raise HTTPException(status_code=503, detail="系统未初始化，请稍后重试")

    try:
        agents = system.registry.get_all_agents(enabled_only=False)
        agent_list = []
        for a in agents:
            agent_list.append({
                "id": a.get("id", ""),
                "name": a.get("name", ""),
                "description": a.get("description", ""),
                "capabilities": a.get("capabilities", []),
                "enabled": a.get("enabled", True),
                "agent_type": a.get("agent_type", "atomic"),
                "model": a.get("model", ""),
                "status": a.get("status", "unknown"),
            })
        return {"agents": agent_list}
    except Exception as e:
        logger.error(f"[Agents] 查询失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查"""
    # Redis 连接状态
    redis_status = "not_configured"
    if system and hasattr(system, 'redis_client') and system.redis_client:
        try:
            system.redis_client.ping()
            redis_status = "connected"
        except Exception:
            redis_status = "disconnected"

    return {
        "status": "healthy" if system else "initializing",
        "service": "云集优选 · 售后助手 API",
        "version": "1.0.0",
        "redis": redis_status,
    }


@app.get("/cache/stats")
async def cache_stats():
    """
    查看所有缓存层的统计信息

    返回各层缓存的命中率、大小、Redis状态等
    """
    if not system:
        raise HTTPException(status_code=503, detail="系统未初始化")

    stats = {
        "redis_connected": system.redis_client is not None,
    }

    # 收集所有已注册 Agent 的缓存统计
    try:
        agents = system.registry.get_all_agents()
        for agent_info in agents:
            agent_instance = agent_info.get("instance")
            if agent_instance and hasattr(agent_instance, 'get_optimization_stats'):
                agent_stats = agent_instance.get_optimization_stats()
                if agent_stats.get("optimization_enabled"):
                    stats[agent_info["id"]] = agent_stats.get("cache_stats", {})
    except Exception as e:
        logger.warning(f"[CacheStats] 获取 Agent 缓存统计失败: {e}")

    # LLM 缓存统计
    for llm_name, llm_instance in [("llm_max", system.llm_max), ("llm_plus", system.llm_plus)]:
        if hasattr(llm_instance, 'llm_cache') and llm_instance.llm_cache:
            stats[f"{llm_name}_cache"] = llm_instance.llm_cache.get_stats()

    return stats


@app.post("/cache/clear")
async def cache_clear():
    """
    清空所有缓存

    包括业务层、检索层、LLM层的内存缓存和 Redis 缓存
    """
    if not system:
        raise HTTPException(status_code=503, detail="系统未初始化")

    cleared = []

    # 清空 Agent 缓存
    try:
        agents = system.registry.get_all_agents()
        for agent_info in agents:
            agent_instance = agent_info.get("instance")
            if agent_instance:
                if hasattr(agent_instance, 'cache_manager'):
                    agent_instance.cache_manager.clear_all()
                    cleared.append(f"{agent_info['id']}_cache_manager")
                if hasattr(agent_instance, 'semantic_cache') and agent_instance.semantic_cache:
                    agent_instance.semantic_cache.clear()
                    cleared.append(f"{agent_info['id']}_semantic_cache")
    except Exception as e:
        logger.warning(f"[CacheClear] 清空 Agent 缓存失败: {e}")

    # 清空 LLM 缓存
    for llm_name, llm_instance in [("llm_max", system.llm_max), ("llm_plus", system.llm_plus)]:
        if hasattr(llm_instance, 'llm_cache') and llm_instance.llm_cache:
            llm_instance.llm_cache.clear()
            cleared.append(f"{llm_name}_cache")

    logger.info(f"[CacheClear] 已清空: {cleared}")
    return {"cleared": cleared, "count": len(cleared)}


# ========== 文档内容查看接口 ==========

@app.get("/document/{doc_id}")
async def get_document(doc_id: str):
    """
    获取知识库文档的完整内容

    通过 doc_id（文件名 stem）查找文档，提取文本内容返回。
    前端引用文档卡片点击后调用此接口展示原文。

    安全设计：仅接受 doc_id 字符串，在注册表中查找，不接受任何文件路径。

    Args:
        doc_id: 文档ID（文件名不含扩展名，如 'rag_optimization_strategies'）

    Returns:
        doc_id, title, filename, file_type, content, content_type
    """
    # 在注册表中查找
    doc_info = _document_registry.get(doc_id)
    if not doc_info:
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")

    file_path: Path = doc_info["path"]
    ext: str = doc_info["ext"]

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"文档文件缺失: {doc_info['filename']}")

    try:
        content, content_type = _extract_document_content(file_path, ext)

        logger.info(f"[Document] 获取文档成功: {doc_id} ({len(content)} 字符)")

        return {
            "doc_id": doc_id,
            "title": doc_info["title"],
            "filename": doc_info["filename"],
            "file_type": ext.lstrip("."),
            "content": content,
            "content_type": content_type,
        }
    except Exception as e:
        logger.error(f"[Document] 读取文档失败: {doc_id} - {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文档读取失败: {str(e)}")


@app.get("/monitor")
async def monitor():
    """系统监控页面"""
    monitor_path = os.path.join(os.path.dirname(__file__), "frontend", "monitor.html")
    if os.path.exists(monitor_path):
        return FileResponse(monitor_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="监控页面文件不存在")


@app.get("/")
async def root():
    """根路径 - 返回前端页面"""
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path, media_type="text/html")
    # 前端文件不存在时返回 API 信息
    return {
        "service": "云集优选 · 售后助手 API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "hint": "前端文件不存在，请将 index.html 放入 frontend/ 目录",
        "endpoints": {
            "stream_chat": "POST /chat/stream",
            "chat": "POST /chat",
            "upload": "POST /upload",
            "resume": "POST /resume",
            "agents": "GET /agents"
        }
    }


# ========== 问答日志接口 ==========

@app.get("/logs")
async def get_logs(
    page: int = 1,
    page_size: int = 20,
    keyword: Optional[str] = None,
    mode: Optional[str] = None,
    success_only: Optional[bool] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """
    分页查询问答日志

    Args:
        page: 页码（从 1 开始）
        page_size: 每页条数（默认 20，最大 100）
        keyword: 关键词搜索（搜索问题和答案）
        mode: 筛选模式（simple / planning / error）
        success_only: 仅显示成功记录（true/false）
        start_date: 开始日期（YYYY-MM-DD）
        end_date: 结束日期（YYYY-MM-DD）

    Returns:
        {total, page, page_size, pages, data: [...]}
    """
    if not qa_logger:
        raise HTTPException(status_code=503, detail="日志系统未初始化")

    page_size = min(page_size, 100)  # 防止请求过大
    try:
        result = qa_logger.query_logs(
            page=page,
            page_size=page_size,
            keyword=keyword,
            mode=mode,
            success_only=success_only,
            start_date=start_date,
            end_date=end_date,
        )
        return result
    except Exception as e:
        logger.error(f"[Logs] 查询失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/stats")
async def get_logs_stats():
    """
    获取问答日志统计概览

    Returns:
        总数、成功率、平均评分、平均耗时、今日查询数等
    """
    if not qa_logger:
        raise HTTPException(status_code=503, detail="日志系统未初始化")

    try:
        return qa_logger.get_stats()
    except Exception as e:
        logger.error(f"[LogStats] 查询失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/export")
async def export_logs():
    """
    导出全部问答日志为 CSV 格式下载

    Returns:
        CSV 文件（text/csv）
    """
    if not qa_logger:
        raise HTTPException(status_code=503, detail="日志系统未初始化")

    try:
        import csv
        import io

        records = qa_logger.export_all()

        if not records:
            return JSONResponse(content={"message": "暂无日志记录"})

        # 生成 CSV
        output = io.StringIO()
        fieldnames = list(records[0].keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

        csv_content = output.getvalue()

        from fastapi.responses import Response
        return Response(
            content=csv_content.encode("utf-8-sig"),  # BOM 确保 Excel 正确识别中文
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=qa_logs.csv"},
        )
    except Exception as e:
        logger.error(f"[LogExport] 导出失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========== 异常处理 ==========

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.error(f"未捕获的异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "path": str(request.url)
        }
    )


if __name__ == "__main__":
    import uvicorn

    # 开发环境直接运行
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"  # debug 模式：输出详细请求/响应日志
    )
