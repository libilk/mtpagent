# -*- coding: utf-8 -*-
"""
视觉问答Agent (VQA)
===================

基于通义千问VL多模态模型，支持：
- 单张图片问答
- 图表数据趋势分析
- 多图表对比分析
- 图片内容描述
- OCR文字识别

VQA = 视觉问答（Visual Question Answering）。
多模态模型 = 能同时读文本和图像的模型，这里走的是 OpenAI 兼容的 chat 接口，
把图片编码成字符串塞进消息的 content 列表里，和文字一起发过去。

形态上和别的 Agent 有本质区别 —— 别的 Agent 是"工具式"的（注册若干工具、
走 ReAct 循环去调），本 Agent 不注册任何工具（全文件没有 register_tool），
只发一次 API 请求拿回一段文本。所以它没有 ReAct 循环、没有多轮工具调用，
LLM 在这里只是"兼容占位"（见 __init__ 的 llm 参数）。
"""

import os
import base64  # base64（二进制转文本编码）：把图片字节编成能塞进 JSON 的字符串
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("openai库未安装，VQA Agent不可用")


# ========== 图表分析专用Prompt ==========
# 下面三个 prompt 常量按分析模式分工，正文一字未改。
# 它们都是"角色扮演 + 输出维度清单"的写法：与其让模型自由发挥，不如把要看的
# 维度（基本信息/概览/趋势/发现/结论）列死，输出才会结构一致、可横向比较。

CHART_ANALYSIS_SYSTEM_PROMPT = """你是一个专业的图表数据分析师。请仔细观察图表，从以下维度进行全面分析：

1. **图表基本信息**：图表类型（柱状图/折线图/饼图/散点图等）、标题、坐标轴含义、图例
2. **数据概览**：关键数据点、最大值、最小值、平均水平
3. **趋势分析**：整体趋势（上升/下降/波动/平稳）、转折点、增长率
4. **关键发现**：异常值、显著变化、值得关注的模式
5. **结论建议**：基于数据得出的核心结论

请用结构化的方式输出分析结果，数据要准确，分析要有依据。"""

MULTI_CHART_COMPARE_SYSTEM_PROMPT = """你是一个专业的图表数据分析师。用户提供了多张图表，请进行对比分析：

1. **图表概述**：分别描述每张图表的类型和主要内容
2. **数据对比**：找出各图表之间的共同指标，进行数值对比
3. **趋势对比**：各图表展示的趋势是否一致，有何差异
4. **关联分析**：图表之间是否存在因果关系或相关性
5. **综合结论**：综合多张图表得出的整体结论和建议

请确保对比分析有据可依，结论清晰明确。"""

GENERAL_VQA_SYSTEM_PROMPT = """你是一个视觉分析助手。请仔细观察图片内容，准确回答用户的问题。
如果图片中包含文字，请准确识别。如果是图表，请准确读取数据。
回答要具体、准确，避免模糊表述。"""


class VQAAgent:
    """
    视觉问答Agent

    支持三种分析模式：
    - general: 通用图片问答
    - chart_analysis: 单图表深度分析
    - multi_chart_compare: 多图表对比分析
    """

    def __init__(
        self,
        model_name: str = "qwen-vl-max",
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_tokens: int = 4096,
        llm=None,
    ):
        """
        初始化VQA Agent

        Args:
            model_name: 多模态模型名称（qwen-vl-max / qwen-vl-plus）
            api_key: API密钥（默认读取DASHSCOPE_API_KEY环境变量）
            base_url: API地址
            max_tokens: 最大生成token数（token = 词元：模型处理文本的最小单位，
                        也是单次回答的长度上限）
            llm: 现有LLM实例（用于指代消解等文本任务，保持架构兼容）
                 指代消解 = 把"这张图里它怎么样"里的"它"还原成具体对象；
                 本文件其实只把 llm 存进 self.llm 就再没读过 —— 图片理解完全
                 交给多模态模型，文本任务并不经过这个 llm。
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.llm = llm  # 兼容 make_agent_node 中的指代消解

        if not self.api_key:
            raise ValueError("API密钥未设置，请设置DASHSCOPE_API_KEY环境变量")

        if HAS_OPENAI:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=120,
            )
        else:
            self.client = None
            logger.warning("openai库未安装，VQA Agent将无法调用视觉模型")

    def handle(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        handle（处理入口）：Agent 对外统一的方法签名，兼容 make_agent_node 调用约定。
        图节点只认 handle 这个名字，所以签名必须和别的 Agent 一致 —— 这正是
        AgentProtocol（Agent 接口契约：规定 Agent 必须实现哪些方法）在起作用。

        图片不走 query、走 context：query 只是那句问题，图片在 context 的
        图片字段里（下面 _collect_images 会按优先级合并成一个列表）。

        Args:
            query: 用户问题
            context: 上下文，可包含图片信息：
                - image_urls: List[str] 图片公网URL列表
                - image_paths: List[str] 本地图片路径列表
                - image_base64_list: List[str] base64编码列表
                （也兼容单个值：image_url / image_path / image_base64）

        Returns:
            分析结果文本
        """
        context = context or {}

        # 收集所有图片
        images = self._collect_images(context)
        if not images:
            return (
                "未提供图片。请通过以下方式提供图片：\n"
                "- CLI: 在问题后添加 [image:图片路径]，如: 分析这张图表 [image:C:\\data\\chart.png]\n"
                "- API: 在请求中包含 image_urls 或 image_paths 参数"
            )

        # 判断分析模式
        mode = self._detect_mode(query, images)
        logger.info("[VQAAgent] 模式: %s | 图片数: %d", mode, len(images))

        if mode == "multi_chart_compare":
            return self._multi_chart_compare(query, images)
        elif mode == "chart_analysis":
            return self._chart_analysis(query, images[0])
        else:
            return self._general_vqa(query, images)

    def _detect_mode(self, query: str, images: List[str]) -> str:
        """
        根据问题内容和图片数量自动检测分析模式

        注意：这是纯关键词匹配 + 图片张数判断，不调任何模型。
        好处是零成本零延迟、结果可预测；代价是措辞换一种说法就可能落错模式。
        所以别把 mode 当成"模型理解出的意图"，它只是规则分支。

        Returns:
            "multi_chart_compare" / "chart_analysis" / "general"
        """
        query_lower = query.lower()

        # 多图对比关键词
        compare_keywords = [
            "对比", "比较", "对照", "差异", "不同", "相比",
            "compare", "contrast", "difference", "versus", "vs",
            "哪个更", "哪张", "两张", "几张",
        ]

        # 图表分析关键词
        chart_keywords = [
            "图表", "趋势", "走势", "数据", "增长", "下降",
            "柱状图", "折线图", "饼图", "散点图", "条形图",
            "chart", "trend", "graph", "plot",
            "最大值", "最小值", "平均", "峰值", "波动",
            "同比", "环比", "占比", "比例", "分布",
            "分析", "解读", "读取", "看看",
        ]

        # 多图 + 对比关键词 → 多图对比模式
        if len(images) > 1 and any(kw in query_lower for kw in compare_keywords):
            return "multi_chart_compare"

        # 多图但没有对比关键词，也走多图对比（因为给了多张图通常就是要对比）
        if len(images) > 1:
            return "multi_chart_compare"

        # 单图 + 图表关键词 → 图表深度分析
        if any(kw in query_lower for kw in chart_keywords):
            return "chart_analysis"

        return "general"

    def _general_vqa(self, query: str, images: List[str]) -> str:
        """通用图片问答"""
        content = []

        # 添加图片（支持多图）
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": img}})

        content.append({"type": "text", "text": query})

        messages = [
            {"role": "system", "content": GENERAL_VQA_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        return self._call_vision_api(messages)

    def _chart_analysis(self, query: str, image: str) -> str:
        """单张图表深度分析"""
        user_text = (
            f"用户问题：{query}\n\n"
            "请根据上述问题，对这张图表进行深度分析。"
            "如果用户问题比较笼统，请从趋势、关键数据点、异常值等多个维度全面分析。"
        )

        messages = [
            {"role": "system", "content": CHART_ANALYSIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        return self._call_vision_api(messages)

    def _multi_chart_compare(self, query: str, images: List[str]) -> str:
        """多图表对比分析"""
        content = []

        # 添加所有图片，并标注序号
        for i, img in enumerate(images, 1):
            content.append({"type": "image_url", "image_url": {"url": img}})
            content.append({"type": "text", "text": f"（以上是第{i}张图表）"})

        user_text = (
            f"\n用户问题：{query}\n\n"
            f"以上共{len(images)}张图表，请进行对比分析。"
            "如果用户问题比较笼统，请从数据对比、趋势对比、关联分析等多个维度全面对比。"
        )
        content.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": MULTI_CHART_COMPARE_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        return self._call_vision_api(messages)

    def _call_vision_api(self, messages: List[Dict]) -> str:
        """
        调用千问VL视觉模型API

        整个 Agent 只有这一处真正花钱花时间的地方，且只调一次、没有重试。
        失败时把错误文本当正常答案返回（不抛异常），所以调用方看到的
        "图像分析失败: ..." 本质上是一段质量极低的答案，会照常进入后续评估。

        Args:
            messages: OpenAI格式的消息列表

        Returns:
            模型回答文本
        """
        if not self.client:
            return "VQA Agent不可用：openai库未安装，请执行 pip install openai"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
            )
            answer = response.choices[0].message.content
            logger.info("[VQAAgent] 调用成功，回答长度: %d", len(answer))
            return answer

        except Exception as e:
            logger.error("[VQAAgent] API调用失败: %s", e)
            return f"图像分析失败: {e}"

    # ========== 图片收集与格式转换 ==========

    def _collect_images(self, context: Dict[str, Any]) -> List[str]:
        """
        从context中收集所有图片，统一转换为API可用的URL或data URI

        data URI = 形如 data:image/png;base64,xxxx 的自包含字符串，把"文件类型
        + 图片内容"直接编码进一个字符串。本地图片必须转成这种形式才能跟文字一起
        塞进 JSON 消息体；公网 URL 则可以原样传。

        支持的context字段（优先级从高到低）：
        - image_urls: List[str]     公网URL列表
        - image_paths: List[str]    本地文件路径列表
        - image_base64_list: List[str]  base64编码列表
        - image_url: str            单个公网URL（兼容）
        - image_path: str           单个本地路径（兼容）
        - image_base64: str         单个base64（兼容）
        """
        images = []

        # 批量字段
        for url in (context.get("image_urls") or []):
            if url:
                images.append(url)

        for path in (context.get("image_paths") or []):
            resolved = self._local_path_to_data_uri(path)
            if resolved:
                images.append(resolved)

        for b64 in (context.get("image_base64_list") or []):
            resolved = self._base64_to_data_uri(b64)
            if resolved:
                images.append(resolved)

        # 单值兼容字段
        if not images:
            url = context.get("image_url")
            if url:
                images.append(url)

            path = context.get("image_path")
            if path:
                resolved = self._local_path_to_data_uri(path)
                if resolved:
                    images.append(resolved)

            b64 = context.get("image_base64")
            if b64:
                resolved = self._base64_to_data_uri(b64)
                if resolved:
                    images.append(resolved)

        return images

    def _local_path_to_data_uri(self, path: str) -> Optional[str]:
        """将本地图片文件转换为base64 data URI

        MIME type（文件类型标识，如 image/png）靠扩展名猜，写进 data URI 的前缀里，
        接口据此判断这是不是一张合法图片；认不出的扩展名一律按 image/png 兜底。
        """
        if not os.path.exists(path):
            logger.error("[VQAAgent] 图片文件不存在: %s", path)
            return None

        ext = os.path.splitext(path)[1].lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime = mime_map.get(ext, "image/png")

        try:
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")

            # base64 会把字节数放大到约 4/3，所以乘 0.75 反推原始文件大小；
            # 这行纯粹是给日志看的，不参与任何判断。
            size_kb = len(encoded) * 0.75 / 1024
            logger.info("[VQAAgent] 本地图片已转base64: %s (%.1f KB)", path, size_kb)
            return f"data:{mime};base64,{encoded}"
        except Exception as e:
            logger.error("[VQAAgent] 读取图片失败 %s: %s", path, e)
            return None

    def _base64_to_data_uri(self, b64: str) -> Optional[str]:
        """将base64字符串转换为data URI（如果还不是的话）"""
        if not b64:
            return None
        if b64.startswith("data:image"):
            return b64
        return f"data:image/png;base64,{b64}"
