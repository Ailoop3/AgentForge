"""
编排器 - Orchestrator

将RAG Engine、Multi-Agent Core、Eval & Observability三大模块整合为统一平台。

执行流程:
┌────────────────────────────────────────────────────────────────┐
│                    Orchestrator.run(task)                        │
│                                                                │
│  1. start_trace()          ← Tracer: 创建全链路追踪             │
│  2. router.route(task)     ← Router: 选择参与的Agent            │
│  3. planner.plan(task)     ← Planner: 生成DAG执行计划           │
│  4. 按DAG逐层执行:                                              │
│     ├─ agent.run(state)    ← BaseAgent: 执行单个Agent           │
│     │   └─ kb.retrieve()   ← RAG: 知识库检索                    │
│     └─ 并行执行无依赖的节点                                      │
│  5. arbiter.arbitrate()    ← Arbiter: 仲裁最终结果              │
│  6. evaluator.evaluate()   ← Evaluator: 质量评估                │
│  7. metrics.record()       ← Metrics: 记录指标                  │
│                                                                │
│  输出: OrchestratorResult(最终输出 + 完整trace + 评估报告)       │
└────────────────────────────────────────────────────────────────┘

学习要点:
1. 编排器是系统的"导演"，协调所有组件
2. 全链路追踪贯穿始终
3. 每一步都有错误处理和降级策略
4. 结果是数据丰富的(带trace、evaluation、metrics)
"""

import time
import uuid
from typing import Any, Callable
from dataclasses import dataclass, field

from .config import settings
from .rag.knowledge_base import KnowledgeBase
from .agents.state import AgentState, create_initial_state
from .agents.base import BaseAgent
from .agents.arbiter import Arbiter
from .agents.router import Router, KeywordRouter, DefaultPlanner, Planner
from .agents.agents.index import get_agent, list_agents
from .observability.tracer import Tracer, SpanType
from .observability.evaluator import Evaluator
from .observability.metrics import MetricsCollector


@dataclass
class OrchestratorResult:
    """
    编排器执行结果

    包含完整信息，便于后续分析、调试和展示。
    """
    # 核心输出
    output: Any                    # Agent系统的最终输出
    task: str                      # 原始任务
    trace_id: str                  # 链路追踪ID

    # 执行详情
    agent_results: dict = field(default_factory=dict)   # 各Agent的中间结果
    plan: dict = field(default_factory=dict)            # 执行计划(DAG)

    # 评估
    overall_score: float = 0.0     # 综合评分
    eval_summary: str = ""         # 评估摘要

    # 性能
    duration_ms: float = 0.0       # 总耗时
    total_cost: float = 0.0        # 总成本

    # 状态
    success: bool = True
    error: str = ""


class Orchestrator:
    """
    编排器 - AgentForge的核心调度引擎

    整合三大模块:
    - RAG Engine: 为每个Agent提供知识支持
    - Multi-Agent Core: 多Agent协作执行
    - Eval & Observability: 全程追踪和评估

    设计模式: 外观模式(Facade)
    - 对外提供简单的run()接口
    - 内部协调复杂的多模块交互
    """

    def __init__(
        self,
        llm_fn: Callable[[str, str], str] | None = None,
        router: Router | None = None,
        planner: Planner | None = None,
        arbiter: Arbiter | None = None,
        tracer: Tracer | None = None,
        evaluator: Evaluator | None = None,
        metrics: MetricsCollector | None = None,
        knowledge_bases: dict[str, KnowledgeBase] | None = None
    ):
        """
        Args:
            llm_fn: LLM调用函数 (system, user) → response
            router: 路由器(默认KeywordRouter)
            planner: 规划器(默认DefaultPlanner)
            arbiter: 仲裁者(默认置信度加权)
            tracer: 追踪器(默认Tracer)
            evaluator: 评估器(默认Evaluator)
            metrics: 指标收集器(默认MetricsCollector)
            knowledge_bases: {agent_name: KnowledgeBase} Agent知识库映射
        """
        # LLM函数(所有Agent共享)
        self.llm_fn = llm_fn or self._default_llm

        # 核心组件
        self.router = router or KeywordRouter()
        self.planner = planner or DefaultPlanner()
        self.arbiter = arbiter or Arbiter(strategy="weighted")

        # 可观测性组件
        self.tracer = tracer or Tracer()
        self.evaluator = evaluator or Evaluator(llm_fn=self.llm_fn, use_llm=False)
        self.metrics = metrics or MetricsCollector()

        # 知识库映射: 每个Agent可使用独立的知识库
        self.knowledge_bases = knowledge_bases or {}

        # Agent实例缓存(避免重复创建)
        self._agent_cache: dict[str, BaseAgent] = {}

    def run(self, task: str) -> OrchestratorResult:
        """
        执行完整的Agent编排流程

        这是唯一的对外入口，封装了从输入到输出的全部过程。

        Args:
            task: 用户输入的任务描述

        Returns:
            OrchestratorResult: 包含输出、trace、评估的完整结果

        用法:
            orch = Orchestrator(llm_fn=call_openai)
            result = orch.run("分析竞品并写报告")
            print(result.output)
            print(f"trace: result.trace_id")
        """
        start_time = time.time()

        # Step 1: 创建trace并开始追踪
        with self.tracer.start_trace(task) as trace_id:
            try:
                # Step 2: 路由 - 选择合适的Agent
                with self.tracer.span("router.route", SpanType.ROUTING, input_data=task):
                    agent_names = self.router.route(task)

                # Step 3: 规划 - 生成DAG
                with self.tracer.span("planner.plan", SpanType.PLANNING, input_data=agent_names):
                    dag = self.planner.plan(task, agent_names)
                    layers = dag.get_execution_order()

                # Step 4: 创建全局状态
                state = create_initial_state(task, trace_id)

                # Step 5: 按DAG逐层执行
                all_agent_results = {}

                for layer_idx, layer in enumerate(layers):
                    # 当前层的节点可并行(这里串行执行，可优化为并行)
                    for node in layer:
                        with self.tracer.span(
                            f"{node.agent}.run",
                            SpanType.AGENT,
                            agent=node.agent
                        ) as span:
                            # 获取或创建Agent实例
                            agent = self._get_agent(node.agent)

                            # 执行Agent
                            state, result = agent.run(state)
                            all_agent_results[node.agent] = result

                            # 记录到trace
                            span.output = result.output
                            span.tokens = 0  # 实际应由Agent统计
                            if not result.success:
                                span.error = result.error

                # Step 6: 仲裁 - 汇总各Agent输出
                with self.tracer.span("arbiter.arbitrate", SpanType.ARBITRATION):
                    arbiter_result = self.arbiter.arbitrate(all_agent_results, task)

                # Step 7: 更新最终状态
                state["final_output"] = arbiter_result["output"]
                state["final_score"] = arbiter_result["score"]

                # Step 8: 评估
                eval_report = None
                with self.tracer.span("evaluator.evaluate", SpanType.CUSTOM):
                    # 简化: 实际应使用完整TraceTree
                    overall_score = arbiter_result.get("score", 0.5)

                # Step 9: 记录指标
                duration_ms = (time.time() - start_time) * 1000
                self.metrics.record_query(
                    success=True,
                    latency_ms=duration_ms,
                    cost=arbiter_result.get("total_cost", 0)
                )
                for agent_name in all_agent_results:
                    self.metrics.record_agent_call(agent_name)

                # Step 10: 构造返回结果
                return OrchestratorResult(
                    output=arbiter_result["output"],
                    task=task,
                    trace_id=trace_id,
                    agent_results={
                        name: {
                            "output": r.output,
                            "success": r.success,
                            "duration_ms": r.duration * 1000,
                            "confidence": r.confidence
                        }
                        for name, r in all_agent_results.items()
                    },
                    plan={"agents": agent_names, "layers": len(layers)},
                    overall_score=overall_score,
                    eval_summary=arbiter_result.get("details", {}).get("confidences", {}),
                    duration_ms=duration_ms,
                    success=True
                )

            except Exception as e:
                # 错误处理: 记录但不抛出
                duration_ms = (time.time() - start_time) * 1000
                self.metrics.record_query(
                    success=False,
                    latency_ms=duration_ms
                )

                return OrchestratorResult(
                    output=None,
                    task=task,
                    trace_id=trace_id,
                    success=False,
                    error=str(e),
                    duration_ms=duration_ms
                )

    def _get_agent(self, name: str) -> BaseAgent:
        """
        获取Agent实例(带缓存)

        缓存避免重复创建Agent实例(特别是LLM客户端等重型资源)
        """
        if name not in self._agent_cache:
            # 为该Agent注入专属知识库(如果有)
            kb = self.knowledge_bases.get(name)

            self._agent_cache[name] = get_agent(
                name,
                llm_fn=self.llm_fn,
                knowledge_base=kb,
                tools=[],  # 可扩展: 为不同Agent注入不同工具
                max_iterations=settings.MAX_AGENT_ITERATIONS
            )
        return self._agent_cache[name]

    def _default_llm(self, system: str, user: str) -> str:
        """默认LLM占位函数"""
        return f"[AgentForge默认响应] 已处理: {user[:50]}..."

    def get_statistics(self) -> dict:
        """获取系统统计信息"""
        return {
            "traces": self.tracer.get_statistics(),
            "metrics": self.metrics.get_report(),
            "cached_agents": list(self._agent_cache.keys())
        }
