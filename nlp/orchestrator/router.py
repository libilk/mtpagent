# -*- coding: utf-8 -*-
"""
Agent路由器
===========

router（路由器）= 简单问题：从一堆 Agent 里挑**一个**最合适的
（复杂问题拆成多个任务的活儿在 planner.py，别在这找）。

⚠ 本文件与 langgraph_orchestrator/router.py 是两个不同的 router，极易混淆：
   那个文件是"条件边函数"集合（route_by_complexity / route_to_agent …，图上决定
   下一步走谁，没有类）；本文件是一个类 AgentRouter（用检索的方式选出 Agent）。
   同名不同物，看 import 路径区分。

选法是两级，和 RAG（检索增强生成）的套路一致：
  第 1 级 召回（recall）：embedding（向量化）后算余弦相似度，从全量 Agent 里捞出
          top-3 候选。便宜，但只看语义相近 —— 描述写得不像的 Agent 会被漏掉
  第 2 级 精排（rerank）：让 LLM 读完候选的能力描述做最终选择。贵，但能纠召回级的漏
两级之间还有一道前置拦截：闲聊/问候在向量召回**之前**就被短路给 chat_agent（省一次 embedding）。

历史包袱：route_simple() 已被删除（全项目无调用方的死代码），旧文档/旧笔记里出现
这个名字时，指的是已经不存在的代码。
"""

import json
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class AgentRouter:
    """Agent路由器"""

    def __init__(self, llm, embedder):
        """
        初始化

        Args:
            llm: LLM实例（Qwen-Max）
            embedder: 向量化模型（把文本变成一串数字，语义相近的文本向量也相近）
        """
        self.llm = llm
        self.embedder = embedder
        # 向量索引缓存 {agent_id: vector}；惰性填充 —— 没索引的会在 _vector_recall 里
        # 现场算并回填，所以漏调 index_agents() 也不会崩，只是首次路由会慢一点
        self.agent_vectors = {}  # {agent_id: vector}

    def index_agents(self, agents: List[Dict]):
        """
        为Agent建立向量索引

        索引文本 = description + capabilities —— 这两段是"这个 Agent 能干什么"的语义
        摘要，也就是召回时真正被检索的语料。言下之意：路由选错时第一个该看的是这两段
        文案写得清不清楚，而不是去调相似度算法。

        Args:
            agents: Agent列表
        """
        for agent in agents:
            # 组合描述和能力作为索引文本
            text = f"{agent['description']} {' '.join(agent['capabilities'])}"
            vector = self.embedder.embed(text)
            self.agent_vectors[agent['id']] = vector

        logger.info(f"已为 {len(agents)} 个Agent建立索引")

    def route(
        self,
        task: Dict,
        agents: List[Dict],
        use_llm_rerank: bool = True,
        enable_auto_correct: bool = True
    ) -> str:
        """
        路由任务到Agent（优化版：精排+验证合并为一次LLM调用）

        流程：
        0. 前置拦截：闲聊/问候/极短查询直接路由到 chat_agent
        1. 向量相似度召回 top-3 候选
        2. LLM 一次调用完成精排+验证（输出选择、置信度、理由）

        三级短路早退，越早越便宜：只有 1 个 Agent → 前置拦截命中 → 召回后只剩 1 个候选，
        都直接 return，不花 LLM 调用。所以"这次怎么没看到 LLM 日志"往往是短路生效，不是出错。

        注意 enable_auto_correct 参数在函数体内**从未被读取**：自动纠正已融进
        _llm_select 的提示词（让 LLM 可以从候选之外重选），这个参数只剩签名兼容作用。

        Args:
            task: 任务信息（只需要 description 字段，其余字段本文件不用）
            agents: 可用的Agent列表
            use_llm_rerank: 是否使用LLM精排（False 时退化为"直接取相似度最高的候选"）
            enable_auto_correct: 已失效，见上

        Returns:
            选中的Agent ID
        """
        if not agents:
            raise ValueError("没有可用的Agent")

        # 如果只有一个Agent，直接返回
        if len(agents) == 1:
            return agents[0]['id']

        # ===== 前置拦截：闲聊/问候/极短查询直接路由到 chat_agent =====
        # 拦在向量召回之前，是因为"你好"这类 query 语义稀薄，向量可能莫名其妙地
        # 贴到某个业务 Agent 上；与其让精排去纠，不如一开始就不给它出场机会（还省一次 embedding）
        task_desc = task.get('description', '').strip()
        chat_agent_available = any(a['id'] == 'chat_agent' for a in agents)
        if chat_agent_available and self._is_chitchat(task_desc):
            logger.info(f"[路由] 前置拦截: 闲聊/问候 → chat_agent (query='{task_desc}')")
            return "chat_agent"

        # 第一步：向量相似度召回top-3
        # 取 3 是个折中：候选太少，召回阶段一旦漏了正确的 Agent 就再无补救机会；
        # 候选太多，精排提示词变长、噪声变大（召回阶段本来就是宁滥勿缺）
        candidates = self._vector_recall(task, agents, top_k=3)

        # 如果只有一个候选，直接返回
        if len(candidates) == 1:
            return candidates[0][0]['id']

        # 第二步：LLM精排+验证（合并为一次调用）
        # 精排（rerank）= 对召回结果重新排序/复选，把最合适的顶上来。这里把"排序"和
        # "验证选中的合不合理"合并成一次 LLM 调用 —— 少一次往返，代价是两者绑在一起了
        if use_llm_rerank:
            selected_agent_id = self._llm_select(task, candidates, agents)
        else:
            # 直接返回相似度最高的
            selected_agent_id = candidates[0][0]['id']

        logger.info(f"路由任务到Agent: {selected_agent_id}")
        return selected_agent_id

    @staticmethod
    def _is_chitchat(query: str) -> bool:
        """
        判断是否为闲聊/问候/无实质内容的查询

        纯规则、不问 LLM：这类问题在真实对话里占比不低，用关键词表拦掉最便宜也最稳。
        两条判据是"或"的关系 —— 极短且不含技术词，或整句命中问候白名单；
        所以它是**保守**的（宁可漏判闲聊，也不误伤真问题）。

        Args:
            query: 用户查询文本

        Returns:
            是否为闲聊
        """
        # 极短查询（<=5字符）且不含技术关键词，视为闲聊
        if len(query) <= 5:
            tech_keywords = ["RAG", "SQL", "Agent", "API", "检索", "查询", "数据", "分析", "对比", "合同"]
            if not any(kw.lower() in query.lower() for kw in tech_keywords):
                return True

        # 常见问候和闲聊模式
        chitchat_patterns = [
            "hi", "hello", "hey", "你好", "嗨", "哈喽", "在吗", "在不在",
            "你是谁", "你叫什么", "谢谢", "感谢", "再见", "拜拜", "bye",
            "早上好", "下午好", "晚上好", "早安", "晚安", "good morning",
            "good afternoon", "good evening", "good night",
            "ok", "好的", "嗯", "哦", "呵呵", "哈哈", "666", "厉害",
        ]
        query_lower = query.lower().strip()
        if query_lower in chitchat_patterns:
            return True

        return False

    def _vector_recall(
        self,
        task: Dict,
        agents: List[Dict],
        top_k: int = 3
    ) -> List[Tuple[Dict, float]]:
        """
        向量相似度召回

        只做"粗筛"，不做决定 —— 所以它不排序给谁用，只负责把可能相关的捞齐。
        返回的 similarity 是余弦相似度（比较两个向量方向有多接近，取值 0~1），
        它会被写进精排提示词让 LLM 参考，但不作为最终判据。

        Args:
            task: 任务信息（只取 description 当查询）
            agents: Agent列表
            top_k: 取前 k 条（只保留得分最高的 k 个）

        Returns:
            候选Agent列表 [(agent, similarity), ...]，已按相似度降序
        """
        # 任务描述向量化
        task_desc = task.get('description', '')
        logger.info(f"[RAG步骤1] 用户问题向量化: '{task_desc}'")
        task_vector = self.embedder.embed(task_desc)
        logger.info(f"[RAG步骤1] 向量化完成，维度: {len(task_vector)}")

        # 计算相似度
        similarities = []
        for agent in agents:
            agent_id = agent['id']
            if agent_id not in self.agent_vectors:
                # 如果没有索引，现场计算
                text = f"{agent['description']} {' '.join(agent['capabilities'])}"
                agent_vector = self.embedder.embed(text)
                self.agent_vectors[agent_id] = agent_vector
            else:
                agent_vector = self.agent_vectors[agent_id]

            # 计算余弦相似度
            # 就地在循环内 import：Python 会缓存已加载的模块，重复 import 只是查一次
            # 字典，没有实际开销 —— 看着别扭但不是 bug，别顺手"优化"
            from llm.embedder import cosine_similarity
            sim = cosine_similarity(task_vector, agent_vector)
            similarities.append((agent, sim))

        # 排序并返回top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        candidates = similarities[:top_k]

        logger.info(f"[RAG步骤2] Agent路由完成，候选: {[(c[0]['name'], f'{c[1]:.3f}') for c in candidates]}")
        logger.debug(f"向量召回候选: {[(c[0]['id'], c[1]) for c in candidates]}")
        return candidates

    def _llm_select(
        self,
        task: Dict,
        candidates: List[Tuple[Dict, float]],
        all_agents: List[Dict]
    ) -> str:
        """
        LLM精排+验证（合并为一次调用）

        将原来的 _llm_rerank() + IntentValidator.validate_routing()
        合并为一次 LLM 调用，同时完成选择和验证。

        提示词里特意把"没进候选的 Agent"也列了出来，这是**召回可能漏**的补救：
        向量只认语义相近，字面不像但职责对的 Agent 会被漏掉，所以给 LLM 一次从候选外
        重选的机会（日志里那句"精排修正"就是命中了这条路径）。代价是提示词变长。

        LLM 返回的 confidence 和 reason 只用于写日志，**不参与任何决策** ——
        别误以为低置信度会触发复核（没有这个机制）。

        Args:
            task: 任务信息
            candidates: 向量召回的候选Agent列表
            all_agents: 所有可用Agent列表（用于验证候选之外是否有更合适的）

        Returns:
            选中的Agent ID
        """
        task_desc = task.get('description', '')

        # 构建候选Agent描述
        candidates_desc = "\n".join([
            f"{i+1}. {agent['id']}: {agent['name']}\n"
            f"   描述: {agent['description']}\n"
            f"   能力: {', '.join(agent['capabilities'])}\n"
            f"   相似度: {sim:.3f}"
            for i, (agent, sim) in enumerate(candidates)
        ])

        # 构建所有Agent描述（用于验证是否有候选之外更合适的）
        candidate_ids = {agent['id'] for agent, _ in candidates}
        other_agents = [a for a in all_agents if a['id'] not in candidate_ids]

        other_agents_desc = ""
        if other_agents:
            other_agents_desc = "\n\n其他可用Agent（未进入候选）:\n" + "\n".join([
                f"- {agent['id']}: {agent['name']}\n"
                f"  描述: {agent['description']}\n"
                f"  能力: {', '.join(agent['capabilities'])}"
                for agent in other_agents
            ])

        prompt = (
            f"请为以下任务选择最合适的Agent。\n\n"
            f"任务: {task_desc}\n\n"
            f"候选Agent（按相似度排序）:\n"
            f"{candidates_desc}{other_agents_desc}\n\n"
            "选择规则（这是电商售后场景，按下面的边界判断）：\n"
            "1. 售后政策、退换货规则、运费谁承担、退款时效、会员权益等**咨询类**问题 -> customer_service_agent\n"
            "2. 办理类动作：要退货/换货、提交售后申请、投诉 -> customer_service_agent\n"
            "3. 查询订单/物流/退款等**交易数据**（订单状态、快递到哪了、退款进度、按条件统计订单）-> database_agent\n"
            "4. 与售后无关的通用知识、概念解释、原理说明 -> knowledge_agent\n"
            "5. 图片分析（破损商品照片、快递单、发票截图、图表）-> vqa_agent\n"
            "6. 闲聊、问候、意图不明确 -> chat_agent\n\n"
            "注意 1 和 3 的区别：问「退货政策是什么」→ customer_service_agent；\n"
            "问「我这个订单现在什么状态」→ database_agent。一个问规则，一个查数据。\n\n"
            "请仔细分析任务需求与每个Agent的能力匹配度，选择最合适的Agent。\n"
            "如果候选列表中的Agent都不合适，可以从其他可用Agent中选择。\n\n"
            '只输出JSON格式：\n'
            '{"selected_agent_id": "agent_id", "confidence": "high/medium/low", "reason": "选择理由"}'
        )

        try:
            response = self.llm.generate(prompt, temperature=0.1, max_tokens=200)

            from pydantic import BaseModel, Field
            from llm.langchain_parser import parse_llm_output

            # 模型类定义在函数内：它只为这一次解析服务，放模块级会白白占一个命名空间
            class RouteResult(BaseModel):
                selected_agent_id: str = Field(description="选择的Agent ID")
                confidence: str = Field(default="medium", description="置信度")
                reason: str = Field(default="", description="选择理由")

            result = parse_llm_output(response, RouteResult, fallback=None)

            if result and result.selected_agent_id:
                selected_id = result.selected_agent_id
                confidence = result.confidence
                reason = result.reason

                # 验证ID是否有效（候选 + 全部Agent）
                all_valid_ids = {a['id'] for a in all_agents}
                if selected_id in all_valid_ids:
                    if selected_id not in candidate_ids:
                        logger.info(f"[路由] LLM精排修正: 从候选外选择了更合适的 {selected_id}，理由: {reason}")
                    else:
                        logger.info(f"[路由] LLM选择: {selected_id}（置信度: {confidence}）")
                    return selected_id

            # JSON 解析成功但 ID 无效，降级
            logger.warning(f"LLM返回无效结果，使用相似度最高的")
            return candidates[0][0]['id']

        except Exception as e:
            logger.error(f"LLM精排+验证失败: {e}，使用相似度最高的")
            return candidates[0][0]['id']

    def _llm_rerank(
        self,
        task: Dict,
        candidates: List[Tuple[Dict, float]]
    ) -> str:
        """
        LLM精排

        注意：本方法目前**全项目无调用方** —— 它的活儿已被 _llm_select 并进去
        （后者同时做选择与验证）。保留它是历史遗留，读路由流程时不要把它当成现役的一步。

        Args:
            task: 任务信息
            candidates: 候选Agent列表

        Returns:
            选中的Agent ID
        """
        # 构建提示词
        task_desc = task.get('description', '')

        candidates_desc = "\n".join([
            f"{i+1}. {agent['id']}: {agent['name']}\n"
            f"   描述: {agent['description']}\n"
            f"   能力: {', '.join(agent['capabilities'])}\n"
            f"   相似度: {sim:.3f}"
            for i, (agent, sim) in enumerate(candidates)
        ])

        prompt = (
            f"请选择最适合执行以下任务的Agent。\n\n"
            f"任务: {task_desc}\n\n"
            f"候选Agent:\n{candidates_desc}\n\n"
            f"请分析任务需求，选择最合适的Agent。只输出Agent的ID，不要有其他内容。\n\n"
            f"选择的Agent ID:"
        )

        try:
            response = self.llm.generate(prompt, temperature=0.1, max_tokens=50)
            selected_id = response.strip()

            # 验证ID是否有效
            valid_ids = [agent['id'] for agent, _ in candidates]
            if selected_id in valid_ids:
                logger.debug(f"LLM选择: {selected_id}")
                return selected_id
            else:
                logger.warning(f"LLM返回无效ID: {selected_id}，使用相似度最高的")
                return candidates[0][0]['id']

        except Exception as e:
            logger.error(f"LLM精排失败: {e}，使用相似度最高的")
            return candidates[0][0]['id']
