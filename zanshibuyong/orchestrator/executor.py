# -*- coding: utf-8 -*-
"""
ReAct执行器
===========

执行任务DAG，支持思考-行动-观察循环
优化：LLM语义评估 + 具体反馈注入 + 重复检测 + 简化重试
"""

import logging
from typing import Dict, List, Any, Optional
from llm.output_parser import parse_llm_json

logger = logging.getLogger(__name__)


class ReactExecutor:
    """ReAct执行器"""

    def __init__(self, llm, registry, max_iterations: int = 3):
        """
        初始化

        Args:
            llm: LLM实例（Qwen-Max）
            registry: Agent注册中心
            max_iterations: 最大迭代次数（优化：从5降为3）
        """
        self.llm = llm
        self.registry = registry
        self.max_iterations = max_iterations

    def execute(self, plan: Dict, query: str) -> Dict:
        """
        执行任务计划

        Args:
            plan: 任务计划
            query: 原始查询

        Returns:
            执行结果
        """
        tasks = plan.get("tasks", [])
        if not tasks:
            return {"error": "任务列表为空"}

        # 执行结果存储
        task_results = {}

        # 按依赖关系排序任务（拓扑排序）
        sorted_tasks = self._topological_sort(tasks)

        # 执行每个任务
        for task in sorted_tasks:
            task_id = task["task_id"]
            logger.info(f"执行任务: {task_id} - {task['description']}")

            # 获取依赖任务的结果
            dependencies = {}
            for dep_id in task.get("depends_on", []):
                if dep_id in task_results:
                    dependencies[dep_id] = task_results[dep_id]

            # 执行任务
            result = self._execute_task(task, dependencies, query)
            task_results[task_id] = result

            logger.info(f"任务完成: {task_id}")

        # 汇总结果
        final_result = self._aggregate_results(tasks, task_results, query)

        return {
            "success": True,
            "result": final_result,
            "task_results": task_results
        }

    def _execute_task(
        self,
        task: Dict,
        dependencies: Dict,
        original_query: str
    ) -> Dict:
        """
        执行单个任务（优化版ReAct循环）

        优化点：
        1. 重复检测：结果与上次相同则停止重试
        2. 质量停滞检测：质量提升不明显则停止
        3. LLM语义评估：替代纯长度评估
        4. 具体反馈注入：让Agent知道哪里需要改进
        """
        agent_id = task["agent_id"]
        agent_info = self.registry.get_agent(agent_id)

        if not agent_info:
            return {"error": f"Agent不存在: {agent_id}"}

        agent_instance = agent_info["instance"]

        # 构建任务上下文
        context = {
            "task": task["description"],
            "dependencies": dependencies,
            "original_query": original_query
        }

        previous_results = []  # 历史结果（用于重复检测）
        last_quality = 0.0

        # ReAct循环
        for iteration in range(self.max_iterations):
            logger.info(f"[Executor] ReAct迭代 {iteration + 1}/{self.max_iterations}")

            # 行动（带反馈注入）
            action_result = self._act(agent_instance, context, iteration)

            # 重复检测
            result_key = str(action_result)[:500]
            if result_key in previous_results:
                logger.info("[Executor] 结果与之前重复，停止重试")
                return {
                    "success": True,
                    "result": action_result,
                    "iterations": iteration + 1,
                    "stop_reason": "duplicate_result"
                }
            previous_results.append(result_key)

            # 评估质量（LLM语义评估）
            eval_result = self._evaluate(action_result, context)
            quality = eval_result["score"]
            feedback = eval_result.get("feedback", "")

            logger.info(f"[Executor] 质量评分: {quality:.2f}")

            # 质量足够好，返回结果
            if quality >= 0.7:
                return {
                    "success": True,
                    "result": action_result,
                    "iterations": iteration + 1,
                    "quality": quality
                }

            # 质量停滞检测
            if iteration > 0 and quality - last_quality < 0.05:
                logger.info(f"[Executor] 质量未明显提升({last_quality:.2f} → {quality:.2f})，停止重试")
                return {
                    "success": quality >= 0.5,
                    "result": action_result,
                    "iterations": iteration + 1,
                    "quality": quality,
                    "stop_reason": "quality_stagnation"
                }

            last_quality = quality

            # 更新上下文（注入具体反馈）
            context["feedback"] = f"质量不足({quality:.2f})"
            context["feedback_detail"] = feedback
            context["previous_result"] = action_result

        # 达到最大迭代次数
        return {
            "success": False,
            "result": action_result,
            "iterations": self.max_iterations,
            "quality": quality,
            "message": "达到最大迭代次数"
        }

    def _act(self, agent_instance, context: Dict, iteration: int) -> Any:
        """
        行动步骤（优化：注入具体反馈）

        Args:
            agent_instance: Agent实例
            context: 上下文
            iteration: 当前迭代次数

        Returns:
            行动结果
        """
        try:
            task_desc = context["task"]

            # 如果有上次的具体反馈，注入到任务描述中
            if iteration > 0 and "feedback_detail" in context:
                feedback = context["feedback_detail"]
                task_desc = f"{task_desc}\n\n【改进要求】上次回答的问题：{feedback}，请针对性改进。"

            result = agent_instance.handle(task_desc, context)
            return result
        except Exception as e:
            logger.error(f"Agent执行失败: {e}")
            return {"error": str(e)}

    def _extract_result_str(self, result: Any) -> str:
        """从结果中提取字符串"""
        if isinstance(result, dict):
            if "error" in result:
                return ""
            if "result" in result:
                return str(result["result"])
        return str(result) if result else ""

    def _evaluate(self, result: Any, context: Dict) -> Dict:
        """
        评估结果质量（优化：LLM语义评估 + 基础检查快速通道）

        Returns:
            {"score": float, "feedback": str}
        """
        result_str = self._extract_result_str(result)

        # 快速通道：明显失败
        if not result_str:
            return {"score": 0.0, "feedback": "结果为空，请确保给出实质性回答"}
        if len(result_str) < 10:
            return {"score": 0.2, "feedback": "回答过于简短，请提供更详细的内容"}

        # 快速通道：结果足够长且无错误关键词，直接通过
        error_keywords = ["错误", "失败", "无法", "error", "抱歉"]
        has_error = any(kw in result_str.lower() for kw in error_keywords)
        if len(result_str) >= 200 and not has_error:
            return {"score": 0.85, "feedback": ""}

        # 中间地带：使用LLM语义评估
        try:
            return self._llm_evaluate(result_str, context)
        except Exception as e:
            logger.warning(f"LLM评估失败: {e}，使用基础评估")
            return self._basic_evaluate(result_str, has_error)

    def _llm_evaluate(self, result_str: str, context: Dict) -> Dict:
        """LLM语义评估"""
        prompt = f"""请评估以下回答的质量。

任务要求: {context['task']}
回答内容: {result_str[:1000]}

评估维度（每项0-10分）：
1. 相关性：回答是否针对任务要求
2. 完整性：是否覆盖任务的关键方面
3. 准确性：是否有明显错误或矛盾

请只输出JSON格式：{{"relevance": 分数, "completeness": 分数, "accuracy": 分数, "feedback": "具体改进建议"}}"""

        response = self.llm.generate(prompt, temperature=0.1, max_tokens=200)
        scores = parse_llm_json(response, fallback={
            "relevance": 7,
            "completeness": 7,
            "accuracy": 7,
            "feedback": ""
        })

        # 加权平均
        total = (scores.get("relevance", 7) * 0.4
               + scores.get("completeness", 7) * 0.3
               + scores.get("accuracy", 7) * 0.3) / 10

        # 生成具体反馈
        feedback_parts = []
        if scores.get("relevance", 10) < 6:
            feedback_parts.append("回答偏离了任务要求，请更聚焦于问题本身")
        if scores.get("completeness", 10) < 6:
            feedback_parts.append("回答不够完整，请补充遗漏的方面")
        if scores.get("accuracy", 10) < 6:
            feedback_parts.append("回答可能存在不准确的内容，请验证后修正")

        custom_feedback = scores.get("feedback", "")
        if custom_feedback:
            feedback_parts.append(custom_feedback)

        return {
            "score": total,
            "feedback": "；".join(feedback_parts) if feedback_parts else ""
        }

    def _basic_evaluate(self, result_str: str, has_error: bool) -> Dict:
        """基础评估（降级方案）"""
        length = len(result_str)

        if has_error:
            return {"score": 0.4, "feedback": "回答中包含错误信息，请修正"}
        elif length < 50:
            return {"score": 0.5, "feedback": "回答内容不足，请提供更多细节"}
        elif length < 200:
            return {"score": 0.7, "feedback": ""}
        else:
            return {"score": 0.85, "feedback": ""}

    def _topological_sort(self, tasks: List[Dict]) -> List[Dict]:
        """
        拓扑排序任务

        Args:
            tasks: 任务列表

        Returns:
            排序后的任务列表
        """
        # 构建依赖图
        graph = {task["task_id"]: task.get("depends_on", []) for task in tasks}
        task_map = {task["task_id"]: task for task in tasks}

        # 计算入度（修复：入度 = 该任务依赖的任务数量）
        in_degree = {task_id: len(deps) for task_id, deps in graph.items()}

        # 队列初始化（入度为0的任务）
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        sorted_tasks = []

        while queue:
            # 取出入度为0的任务
            task_id = queue.pop(0)
            sorted_tasks.append(task_map[task_id])

            # 更新依赖该任务的其他任务的入度
            for other_id, deps in graph.items():
                if task_id in deps:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)

        return sorted_tasks

    def _aggregate_results(
        self,
        tasks: List[Dict],
        task_results: Dict,
        query: str
    ) -> str:
        """
        汇总任务结果

        Args:
            tasks: 任务列表
            task_results: 任务结果
            query: 原始查询

        Returns:
            最终结果
        """
        # 如果只有一个任务，直接返回结果
        if len(tasks) == 1:
            task_id = tasks[0]["task_id"]
            result = task_results.get(task_id, {})
            if isinstance(result, dict):
                return result.get("result", str(result))
            return str(result)

        # 多个任务，需要汇总
        results_text = []
        for task in tasks:
            task_id = task["task_id"]
            result = task_results.get(task_id, {})
            if isinstance(result, dict):
                result = result.get("result", str(result))
            results_text.append(f"{task['description']}: {result}")

        # 使用LLM汇总
        try:
            prompt = f"""请根据以下子任务的结果，回答用户的问题。

用户问题: {query}

子任务结果:
{chr(10).join(results_text)}

请综合以上信息，给出完整的回答:"""

            final_answer = self.llm.generate(prompt, temperature=0.5)
            return final_answer

        except Exception as e:
            logger.error(f"结果汇总失败: {e}")
            return "\n\n".join(results_text)
