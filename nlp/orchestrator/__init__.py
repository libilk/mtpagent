# -*- coding: utf-8 -*-
"""
Orchestrator（编排零件包）
==========================

本包装的是"被图节点调用"的零件，它自己不连图、不跑流程 —— 连图和跑流程都在隔壁
langgraph_orchestrator/ 包里（那边的包文档写了分工，也提醒了别把两个包搞混）。

- registry.py       registry（注册中心）= Agent 名册，按名字取实例；同时也是
                    Harness（约束层）的落点 —— 注册时强制校验 AgentProtocol。
- parameter_aligner.py  parameter_aligner（参数对齐器）= 按 parameter_mapping
                    （参数映射）从上游结果抽字段、填成下游能用的入参；
                    它是"参数验证"真正的实现体，编排层只决定何时调用。
- planner.py        复杂问题拆成带依赖的任务列表。
- router.py         简单问题挑一个 Agent。

本文件只放这段说明，不做任何导入 —— 包根不导出名字，要用里面的东西请按
`from orchestrator.registry import AgentRegistry` 这样直接引子模块。
"""
