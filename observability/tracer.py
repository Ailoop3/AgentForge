"""
链路追踪器 - Tracer (全链路Trace记录)

设计参考: OpenTelemetry / LangSmith / Phoenix Arize

核心概念:
- Trace: 一次完整请求的全链路记录(树状结构)
- Span: Trace中的一个操作单元(如一次LLM调用、工具调用、检索)
- 每个Span记录: 开始/结束时间、输入/输出、子Span

Trace树示例:
trace: "帮我分析竞品"
├── span: router.route [2ms]
├── span: planner.plan [5ms]
├── span: researcher.run [3.2s]
│   ├── span: kb.retrieve [120ms]
│   │   ├── span: embedder.embed [50ms]
│   │   └── span: vector_search [70ms]
│   └── span: llm.call [3.0s]
├── span: analyst.run [2.8s]
│   └── span: llm.call [2.7s]
├── span: writer.run [4.1s]
│   └── span: llm.call [4.0s]
└── span: reviewer.run [1.5s]
    └── span: llm.call [1.4s]

学习要点:
1. 上下文传递(trace_id/span_id)实现跨Agent追踪
2. 树状结构便于定位瓶颈(哪个环节最慢)
3. 结构化存储支持后续分析和回放
"""

import uuid
import time
import json
import os
from datetime import datetime
from typing import Any
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum

from AgentForge.config import settings


class SpanType(Enum):
    """Span类型 - 表示操作的类别"""
    LLM = "llm"              # 大模型调用
    TOOL = "tool"            # 工具执行
    RAG = "rag"              # 检索操作
    AGENT = "agent"          # Agent完整执行
    ROUTING = "routing"      # 路由决策
    PLANNING = "planning"    # 任务规划
    ARBITRATION = "arbitration"  # 结果裁决
    CUSTOM = "custom"        # 自定义


@dataclass
class Span:
    """
    跨度 - 表示一个操作单元

    类比: OpenTelemetry的Span / LangSmith的Run
    一个Span = 一次函数调用/一个步骤的一次执行
    """
    # 标识
    span_id: str
    trace_id: str
    parent_id: str | None     # None表示根span

    # 分类
    name: str                 # 描述性名称(如"researcher.run")
    span_type: SpanType       # 操作类型
    agent: str = ""           # 所属Agent

    # 时间
    start_time: float = 0.0   # 时间戳(秒)
    end_time: float = 0.0

    # I/O
    input: Any = None         # 输入数据
    output: Any = None        # 输出结果

    # 性能指标
    tokens: int | None = None     # LLM消耗的token数
    cost: float | None = None     # 本次操作的成本(美元)

    # 错误
    error: str | None = None

    # 元数据
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """耗时(毫秒)"""
        return (self.end_time - self.start_time) * 1000

    @property
    def is_root(self) -> bool:
        """是否为根span"""
        return self.parent_id is None

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "span_type": self.span_type.value,
            "agent": self.agent,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "tokens": self.tokens,
            "cost": self.cost,
            "error": self.error,
            "input": str(self.input)[:500] if self.input is not None else None,
            "output": str(self.output)[:500] if self.output is not None else None,
            "metadata": self.metadata
        }


@dataclass
class TraceTree:
    """
    追踪树 - 单次请求的完整记录

    Span的树状组织结构，便于:
    - 查看整体执行流程
    - 定位瓶颈(最慢的环节)
    - 分析成本构成
    - 回放执行过程
    """
    trace_id: str
    task: str                              # 原始任务
    root_span: Span                        # 根Span
    children: list['TraceTree'] = field(default_factory=list)  # 子树
    created_at: str = ""

    # 聚合指标(懒计算)
    _total_cost: float | None = None
    _total_latency: float | None = None

    @property
    def total_cost(self) -> float:
        """整条链路的总成本"""
        if self._total_cost is None:
            self._total_cost = self._sum_cost(self.root_span)
        return self._total_cost

    @property
    def total_latency_ms(self) -> float:
        """整条链路总耗时"""
        return self.root_span.duration_ms

    def _sum_cost(self, span: Span) -> float:
        """递归累加成本"""
        cost = span.cost or 0
        for child in self.children:
            cost += child._sum_cost(child.root_span)
        return cost

    def find_slowest_spans(self, top_n: int = 5) -> list[Span]:
        """找出最慢的N个Span(性能分析)"""
        all_spans = self._collect_all_spans(self.root_span)
        # 加上所有子树的span
        for child in self.children:
            all_spans.extend(child._collect_all_spans(child.root_span))
        all_spans.sort(key=lambda s: s.duration_ms, reverse=True)
        return all_spans[:top_n]

    def _collect_all_spans(self, span: Span) -> list[Span]:
        """收集当前及所有子孙Span"""
        spans = [span]
        return spans

    def to_dict(self) -> dict:
        """序列化"""
        return {
            "trace_id": self.trace_id,
            "task": self.task,
            "total_cost": self.total_cost,
            "total_latency_ms": self.total_latency_ms,
            "created_at": self.created_at,
            "root_span": self.root_span.to_dict()
        }


class Tracer:
    """
    追踪器 - 全链路追踪的核心类

    使用方式:
        tracer = Tracer()
        with tracer.start_trace("用户任务") as trace:
            with tracer.span("llm.call", span_type=SpanType.LLM):
                response = call_llm(...)
            with tracer.span("tool.search", span_type=SpanType.TOOL):
                results = search(...)

    设计特点:
    1. 上下文管理器(span)自动计时
    2. Thread-local支持并发追踪
    3. 自动持久化到磁盘
    4. 支持成本计算
    """

    def __init__(self, storage_path: str = None, auto_persist: bool = True):
        """
        Args:
            storage_path: trace存储目录
            auto_persist: 是否自动持久化
        """
        self.storage_path = storage_path or settings.TRACE_STORAGE_PATH
        self.auto_persist = auto_persist

        # 当前活跃的trace栈(支持嵌套操作)
        self._active_trace: str | None = None
        self._active_span: str | None = None

        # 所有span的临时存储(当前trace)
        self._current_spans: list[Span] = []

        # 历史trace索引
        self._trace_index: dict[str, dict] = {}

        # 确保存储目录存在
        os.makedirs(self.storage_path, exist_ok=True)

    @contextmanager
    def start_trace(self, task: str):
        """
        开始一个新的trace

        用法:
            with tracer.start_trace("分析竞品") as trace_id:
                # ... 各种操作
        """
        trace_id = str(uuid.uuid4())[:12]
        self._active_trace = trace_id
        self._current_spans = []

        # 创建根Span
        root = Span(
            span_id=str(uuid.uuid4())[:8],
            trace_id=trace_id,
            parent_id=None,
            name="root",
            span_type=SpanType.CUSTOM,
            start_time=time.time()
        )
        self._current_spans.append(root)
        self._active_span = root.span_id

        try:
            yield trace_id
        finally:
            # 结束根Span
            root.end_time = time.time()
            root.input = task
            root.metadata = {"total_spans": len(self._current_spans)}

            # 构建trace树
            tree = TraceTree(
                trace_id=trace_id,
                task=task,
                root_span=root,
                created_at=datetime.now().isoformat()
            )

            # 索引trace
            self._trace_index[trace_id] = {
                "task": task[:100],
                "trace_id": trace_id,
                "duration_ms": tree.total_latency_ms,
                "total_cost": tree.total_cost,
                "created_at": tree.created_at
            }

            # 持久化
            if self.auto_persist:
                self._persist_trace(tree)

            # 清理
            self._active_trace = None
            self._active_span = None
            self._current_spans = []

    @contextmanager
    def span(
        self,
        name: str,
        span_type: SpanType = SpanType.CUSTOM,
        agent: str = "",
        input_data: Any = None,
        metadata: dict = None
    ):
        """
        创建一个span(上下文管理器)

        自动处理:
        - 开始/结束计时
        - 父子关系链接
        - 异常捕获

        用法:
            with tracer.span("llm.call", SpanType.LLM, agent="researcher") as span:
                span.tokens = 150
                response = call_llm(...)
                span.output = response
        """
        if not self._active_trace:
            # 没有活跃trace时，span不生效(避免遗漏)
            yield None
            return

        span_id = str(uuid.uuid4())[:8]
        span = Span(
            span_id=span_id,
            trace_id=self._active_trace,
            parent_id=self._active_span,
            name=name,
            span_type=span_type,
            agent=agent,
            start_time=time.time(),
            input=input_data,
            metadata=metadata or {}
        )

        # 记录旧的parent(退出时恢复)
        old_parent = self._active_span
        self._active_span = span_id
        self._current_spans.append(span)

        try:
            yield span
        except Exception as e:
            span.error = str(e)
            raise
        finally:
            # 结束计时
            span.end_time = time.time()

            # 自动计算cost(如果有token信息)
            if span.tokens and settings.ENABLE_COST_TRACKING:
                span.cost = self._calculate_cost(span.tokens, span.metadata.get("model", ""))

            # 恢复parent上下文
            self._active_span = old_parent

    def _calculate_cost(self, tokens: int, model: str) -> float:
        """
        估算API调用成本

        简化的成本计算，实际应区分input/output token
        """
        model_costs = settings.COST_PER_1K_TOKENS.get(model, {})
        if model_costs:
            # 简化: 全部按input价格计算
            return tokens / 1000 * model_costs.get("input", 0)
        return 0.0

    def get_current_trace_id(self) -> str | None:
        """获取当前活跃的trace ID"""
        return self._active_trace

    def get_trace(self, trace_id: str) -> TraceTree | None:
        """从索引获取trace"""
        return self._trace_index.get(trace_id)

    def list_traces(self, limit: int = 50) -> list[dict]:
        """列出最近的trace摘要"""
        sorted_traces = sorted(
            self._trace_index.values(),
            key=lambda x: x.get("created_at", ""),
            reverse=True
        )
        return sorted_traces[:limit]

    def get_trace_detail(self, trace_id: str) -> dict | None:
        """获取完整的trace详情"""
        filepath = os.path.join(self.storage_path, f"{trace_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def _persist_trace(self, tree: TraceTree):
        """将trace持久化到磁盘"""
        filepath = os.path.join(self.storage_path, f"{tree.trace_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tree.to_dict(), f, ensure_ascii=False, indent=2)

    def get_statistics(self) -> dict:
        """
        统计所有trace的聚合指标

        Returns:
            包含成功率、平均延迟、总成本等的统计信息
        """
        traces = list(self._trace_index.values())
        if not traces:
            return {"total_traces": 0}

        durations = [t.get("duration_ms", 0) for t in traces]
        costs = [t.get("total_cost", 0) for t in traces]

        return {
            "total_traces": len(traces),
            "avg_duration_ms": sum(durations) / len(durations),
            "p50_duration_ms": sorted(durations)[len(durations) // 2],
            "p95_duration_ms": sorted(durations)[int(len(durations) * 0.95)],
            "total_cost": sum(costs),
            "avg_cost_per_query": sum(costs) / len(costs)
        }
